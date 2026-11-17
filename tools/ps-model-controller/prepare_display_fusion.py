"""Create reproducible recovery/regression manifests from existing batch evidence."""
import argparse
import json
from pathlib import Path
from display_batch.common import atomic_json
from display_batch.schema import BatchConfig


def prepare(batches,out,seed):
    out=Path(out).resolve();jobs=[];recover=[];baseline=[];seen=set()
    for folder in map(Path,batches):
        rows=[json.loads(line) for line in (folder/'results.jsonl').read_text(encoding='utf8').splitlines()]
        for row in rows:
            key=(row['background'],row['asset_id'])
            if key in seen:raise ValueError('duplicate baseline pair')
            seen.add(key)
            # Group names are reporting labels only; never influence scene or candidate generation.
            group='regression_53' if '开发' in folder.name else 'regression_36'
            job={'background':row['background'],'asset_id':row['asset_id'],'seed':seed+len(jobs),'group':group}
            jobs.append(job)
            if row.get('reason') in ('no_supported_lying_candidate','no_candidate_survived_local_physics'):
                recover.append({**job,'group':'recovery'})
            baseline.append({'background':row['background'],'asset_id':row['asset_id'],
                'source':str(folder.resolve()),'job_id':row['job_id'],'status':row['status'],
                'reason':row.get('reason'),'had_image':bool(row.get('image'))})
    atomic_json(out/'baseline.json',baseline)
    for label,items in [('recovery',recover),('regression',jobs)]:
        if not items:raise ValueError('empty '+label+' manifest')
        manifest=out/(label+'-manifest.json');atomic_json(manifest,items)
        config=BatchConfig(backgrounds=str(Path(items[0]['background']).parent),count=len(items),
            output=str(out/label),manifest=str(manifest),seed=seed)
        atomic_json(out/(label+'-config.json'),config.model_dump())
    result={'baseline':len(jobs),'recovery':len(recover),'fixed_photography':sum(r['had_image'] for r in baseline),
            'validation':'cross-background regression; no unseen-camera holdout'}
    atomic_json(out/'plan.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261110)
    a=p.parse_args();print(json.dumps(prepare(a.batches,a.out,a.seed),ensure_ascii=False))
