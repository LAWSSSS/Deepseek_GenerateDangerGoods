"""Supplement the experiment ledger without guessing interrupted worker costs."""
import argparse
import json
from pathlib import Path
from display_batch.common import atomic_json,resolve


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def account(out):
    batches=[]
    for batch in sorted(out.glob('batch-*')):
        if not batch.is_dir():continue
        results=batch/'results.jsonl'
        rows=[json.loads(s) for s in results.read_text(encoding='utf8').splitlines() if s] if results.exists() else []
        runs=list(batch.glob('run-*.json'))
        batches.append({'batch':str(batch),'completed_job_records':len(rows),
            'completed_job_seconds':sum(r.get('seconds',0) for r in rows),
            'final_worker_accounting_available':bool(runs),
            'wall_seconds':sum(read(p)['seconds'] for p in runs) if runs else None,
            'render_counts':None if not runs else {k:sum(read(p).get('render_counts',{}).get(k,0) for p in runs)
                for k in ('object_renders','with_shadow_renders','without_shadow_renders')},
            'note':'Missing final worker logs are unknown, not zero. Job elapsed times can overlap and are not added to wall time.'})
    probe=resolve('work/physics_receiver_probe.json')
    probes=read(probe) if probe.exists() else []
    verifications=[read(p) for p in out.glob('verification*/verification.json')]
    summary={'batches':batches,'physics_diagnostic':{'source':str(probe),'experiments':len(probes),
        'seconds':sum(r.get('seconds',0) for r in probes),'new_api_requests':0,'image_renders':0},
        'offline_png_verification':{'runs':len(verifications),'seconds':sum(r.get('seconds',0) for r in verifications),
            'new_api_requests':sum(r.get('new_api_requests',0) for r in verifications),
            'image_renders':sum(r.get('new_render_count',0) for r in verifications)},
        'api_cost_source':str(out/'validation.json'),
        'api_accounting':'All batch-* usage ledgers, including interrupted attempts and retries, are aggregated in validation.cost.',
        'local_compute_cost':None,'local_compute_reason':'No hardware/electricity tariff configured; interrupted worker time not fully recorded.'}
    atomic_json(out/'diagnostic-accounting.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='outputs/综合融合-12类新场景-20261112')
    args=parser.parse_args();print(json.dumps(account(resolve(args.out)),ensure_ascii=False))
