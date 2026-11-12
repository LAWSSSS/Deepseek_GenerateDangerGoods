"""Freeze, run and report the approved 4-asset, 3-background holdout trial."""
import argparse
import asyncio
from collections import Counter
import html
import json
import os
from pathlib import Path
import re
import sqlite3

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from display_batch.common import atomic_json, file_hash, load_catalog, resolve
from display_batch.costs import cumulative
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig
from refine_photoshop import run as run_photoshop
from scrap_composite.photoshop import availability
from validate_coherent_expanded import source_hashes
from validate_geometry_rebuild import REFERENCE

ASSETS = ('3','13','31','33')
OUT = 'outputs/结构化执行留出小样-20261112'
BACKGROUNDS = 'C:/Users/zhaoc/Desktop/背景图'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def freeze(out, backgrounds, knowledge, provider=None):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'frozen.json'
    if manifest.is_file():
        raise ValueError('holdout_already_frozen_use_new_directory')
    sources = sorted(path.resolve() for path in Path(backgrounds).iterdir()
                     if path.suffix.lower() in ('.jpg','.jpeg','.png') and re.search(r'_36_\d+_\d+$',path.stem))
    if len(sources) != 3:
        raise ValueError('expected_exactly_three_camera_36_backgrounds')
    jobs=[{'background':str(path),'asset_id':asset,'group':path.stem.rsplit('_',3)[-3]+'_'+
           '_'.join(path.stem.split('_')[-2:]),'seed':20261112+4*i+j}
          for i,path in enumerate(sources) for j,asset in enumerate(ASSETS)]
    snapshot=out/'knowledge.sqlite'
    if knowledge.is_file():
        with sqlite3.connect(knowledge.as_uri()+'?mode=ro',uri=True) as source, sqlite3.connect(snapshot) as target:
            source.backup(target)
    frozen={
        'jobs':jobs,'backgrounds':{str(p):file_hash(p) for p in sources},
        'source_models':source_hashes(load_catalog()),'knowledge_sha256':file_hash(snapshot) if snapshot.exists() else None,
        'split':{'development_camera':'53','holdout_camera':'36','independent_new_camera':False},
    }
    atomic_json(out/'jobs.json',jobs)
    atomic_json(out/'trial-config.json',{'backgrounds':str(backgrounds),'output':str(out/'png-batch'),
        'manifest':str(out/'jobs.json'),'count':12,'workflow':'evidence_v1','prepare_photoshop':True,
        'geometry_preset':'rebuild','vehicle_reference':REFERENCE,'max_iterations':2,'cpu_concurrency':1,
        'provider_config':provider,'knowledge_db':str(snapshot)})
    atomic_json(manifest,frozen)
    return frozen


def batch_records(folder):
    path=folder/'results.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line] if path.is_file() else []


def batches(out):
    return [out/'png-batch']+sorted(p for p in (out/'attempts').glob('*') if p.is_dir())


async def run(out, attempt=None):
    frozen=read(out/'frozen.json')
    for path,digest in frozen['backgrounds'].items():
        if file_hash(path)!=digest:
            raise ValueError('frozen_background_changed')
    if read(out/'jobs.json')!=frozen['jobs'] or source_hashes(load_catalog())!=frozen['source_models']:
        raise ValueError('frozen_jobs_or_model_changed')
    snapshot=out/'knowledge.sqlite'
    if (file_hash(snapshot) if snapshot.exists() else None)!=frozen['knowledge_sha256']:
        raise ValueError('frozen_knowledge_changed')
    raw=read(out/'trial-config.json')
    if attempt:
        raw['output']=str(out/'attempts'/attempt)
    config=BatchConfig.model_validate(raw)
    await Pipeline(config).run()
    state=availability()
    atomic_json(out/'photoshop-availability.json',state)
    for row in batch_records(Path(config.output)):
        if not row.get('image'):
            continue
        entry=next(e for e in row['iterations'] if e['iteration']==row['best_iteration'])
        manifest=entry.get('composition_manifest')
        folder=out/'photoshop'/row['job_id']
        if manifest and not folder.exists():
            try:
                await run_photoshop(manifest,folder,provider=config.provider_config,iterations=2)
            except (ValueError,RuntimeError,OSError) as error:
                atomic_json(folder/'failure.json',{'status':'failed','reason':str(error)[:180]})
    return report(out)


def report(out):
    frozen=read(out/'frozen.json')
    original=[row for folder in batches(out) for row in batch_records(folder)]
    by_key={(row['background'],row['asset_id']):row for row in original}
    rows=[]
    for job in frozen['jobs']:
        row={**job,**by_key.get((job['background'],job['asset_id']),{})}
        row.setdefault('status','not_run')
        if row.get('job_id'):
            folder=out/'photoshop'/row['job_id']
            for name in ('workflow.json','unavailable.json','failure.json'):
                if (folder/name).is_file():
                    row['photoshop']=read(folder/name)
                    break
        rows.append(row)
    groups={}
    for row in rows:
        totals=groups.setdefault(row['group'],{'planned':0,'png':0,'photo_pass':0,'psd':0})
        totals['planned']+=1
        totals['png']+=bool(row.get('image'))
        ps=row.get('photoshop',{})
        totals['psd']+=bool(ps.get('outputs',{}).get('psd'))
        if ps.get('best_iteration') is not None:
            best=ps['trials'][ps['best_iteration']]['visual_evidence']
            totals['photo_pass']+=int(min(best['scores'][k] for k in ('lighting','texture','material'))>=3 and
                not any(f['severity'] in ('major','fatal') and f['visibility']=='visible' for f in best['findings']))
    scores=sum(g['photo_pass'] for g in groups.values())
    calls=[cumulative(folder) for folder in batches(out)]
    for path in out.glob('photoshop/*/usage-photoshop*.json'):
        if 'cost' in read(path):calls.append(read(path)['cost'])
    result={'planned':12,'states':dict(Counter(row['status'] for row in rows)), 'by_scene':groups,
        'photo_pass':scores,'goal_passed':scores>=10 and all(g['photo_pass']>=3 for g in groups.values()),
        'png_count':sum(g['png'] for g in groups.values()),'psd_count':sum(g['psd'] for g in groups.values()),
        'source_models_unchanged':source_hashes(load_catalog())==frozen['source_models'],
        'api_ledgers':calls,'actual_requests':sum(c['actual_requests'] for c in calls),
        'reported_tokens':sum(c['reported_tokens'] for c in calls), 'jobs':rows}
    currencies={entry['currency'] for entry in calls if entry['currency']}
    known_currency=len(currencies)==1 and not any(c.get('mixed_currency') for c in calls)
    result['api_cost']={'currency':next(iter(currencies)) if known_currency else None,
        'estimated_known_lower':round(sum(c['estimated_api_min'] or 0 for c in calls),8) if known_currency else None,
        'estimated_total_upper':round(sum(c['estimated_api_max'] or 0 for c in calls),8)
            if known_currency and all(c['complete'] for c in calls) else None,
        'complete':known_currency and all(c['complete'] for c in calls),
        'unpriced_or_unreported_calls':sum(c['unpriced_or_unreported_calls'] for c in calls)}
    protocol=out/'interactive-protocol/diagnosis.json'
    if protocol.is_file():
        item=read(protocol)
        result['interactive_protocol']={key:item[key] for key in (
            'status','model','new_api_requests','new_reported_tokens','seconds','credits','currency_cost')}
    atomic_json(out/'validation.json',result)
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>结构化执行留出小样</title>',
          '<style>body{font:16px system-ui;margin:32px auto;padding:0 18px;max-width:1600px}img{max-width:100%;height:auto}section{border-top:1px solid #aaa;margin:24px 0}</style>',
          '<h1>结构化执行留出小样</h1>',
          f'<p>冻结 12 项。PNG {result["png_count"]}，PSD {result["psd_count"]}，Photoshop 摄影达标 {scores}/12。</p>',
          f'<p>小样累计 {result["actual_requests"]} 次请求，{result["reported_tokens"]} token。</p>',
          '<p><a href="validation.json">逐项原因与实际用量</a></p>']
    if protocol.is_file():
        page.append('<p><a href="interactive-protocol/index.html">归档图的交互式结构化诊断</a></p>')
    shown=set()
    for row in rows:
        if row['group'] not in shown:
            page.append('<h2>'+html.escape(row['group'])+'</h2>')
            page.append('<img src="'+html.escape(Path(row['background']).as_uri(),quote=True)+'" alt="原始完整背景">')
            shown.add(row['group'])
        page.append('<section><h3>物品 '+html.escape(row['asset_id'])+'</h3><p>'+
                    html.escape(row['status']+' · '+str(row.get('reason','未完成')) )+'</p>')
        for path in (row.get('image'),row.get('photoshop',{}).get('outputs',{}).get('png',{}).get('path')):
            if path:
                uri=Path(path).resolve().as_uri()
                page.append('<a href="'+html.escape(uri,quote=True)+'"><img src="'+html.escape(uri,quote=True)+'" alt="完整输出"></a>')
        page.append('</section>')
    (out/'index.html').write_text('\n'.join(page),encoding='utf8')
    return {k:v for k,v in result.items() if k!='jobs'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run','report'))
    parser.add_argument('--out',default=OUT)
    parser.add_argument('--backgrounds',default=BACKGROUNDS)
    parser.add_argument('--knowledge',default='work/knowledge/history-v2.sqlite')
    parser.add_argument('--provider')
    parser.add_argument('--attempt',help='New attempt name when implementation changed; preserve earlier ledgers')
    args=parser.parse_args()
    out=resolve(args.out)
    if args.command=='freeze':
        result=freeze(out,resolve(args.backgrounds),resolve(args.knowledge),args.provider)
        result={'jobs':len(result['jobs']),'source_models':len(result['source_models']),'output':str(out)}
    elif args.command=='run':
        if args.attempt and not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}',args.attempt):
            parser.error('invalid attempt name')
        result=asyncio.run(run(out,args.attempt))
    else:
        result=report(out)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
