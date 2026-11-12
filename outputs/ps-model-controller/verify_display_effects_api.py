"""Replay a completed online batch offline and verify optical controls on real EXRs."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash, resolve
from display_batch.compose import compose
from display_batch.framing import final_resolution
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


async def run(source, out, controls_only=False):
    started=time.monotonic()
    original=[json.loads(line) for line in (source/'results.jsonl').read_text(encoding='utf8').splitlines() if line]
    config=BatchConfig.model_validate({**read(source/'config.json'),'output':str(out),'offline':True})
    summary=await Pipeline(config).run() if not controls_only else {'run':{}}
    if not controls_only and (summary['run']['state']!='complete' or summary['cost_total']['actual_requests']):
        raise ValueError('offline_replay_incomplete_or_made_api_requests')
    replay=original if controls_only else [json.loads(line) for line in (out/'results.jsonl').read_text(encoding='utf8').splitlines() if line]
    if len(original)!=len(replay):raise ValueError('job_count_changed')
    for old,new in zip(original,replay,strict=True):
        if old['status']!=new['status'] or old['reason']!=new['reason'] or bool(old.get('image'))!=bool(new.get('image')):
            raise ValueError('offline_outcome_changed')
        if old.get('image') and file_hash(old['image'])!=file_hash(new['image']):
            raise ValueError('offline_image_changed')
    resumed=await Pipeline(config).run() if not controls_only else {'run':{'seconds':0}}
    if not controls_only and (resumed['run'].get('render_counts') or resumed['usage_current_run']['api_calls']):
        raise ValueError('completed_resume_repeated_work')
    pipe=Pipeline(config)
    controls=[]
    try:
        jobs={job['id']:job['payload'] for job in pipe.store.rows()}
        if controls_only:
            inputs=read(source/'inputs.json')
            jobs={r['job_id']:next(j for j in inputs if (j['asset_id'],j['background'])==(r['asset_id'],r['background'])) for r in original}
        for row in replay:
            if not row.get('image'):continue
            job=out/'jobs'/row['job_id'];job.mkdir(parents=True,exist_ok=True)
            input_job=(source if controls_only else out)/'jobs'/row['job_id']
            initial=read(input_job/'iteration-0.json')
            candidate=next(c for c in read(input_job/'candidates.json')['candidates']
                           if c['id']==initial['params']['candidate'])
            asset=read(pipe.cache/'assets'/row['asset_signature']/'result.json')
            candidate['material_classes']=asset['semantic']['material_classes']
            scene=await pipe.scenes.prepare(row['background'])
            seed=jobs[row['job_id']]['seed']
            directory,meta,env,samples,hit=await pipe.render(asset,scene,candidate,initial['params'],final_resolution(candidate))
            if not hit:raise ValueError('optical_controls_should_reuse_renders')
            baseline=compose(scene,candidate,directory,meta,env,samples,initial['params'],seed,
                             job/'verified-initial.png',capture_local=True)
            if not controls_only and file_hash(baseline['image'])!=initial['image_hash']:
                raise ValueError('initial_png_did_not_replay')
            visible=baseline.pop('local_layers')['alpha']>0
            with Image.open(baseline['image']) as im:before=np.asarray(im.convert('RGB'))
            x,y,r,b=baseline['edit_bounds']
            mask=np.zeros(before.shape[:2],bool);mask[y:b,x:r]=visible
            controls_to_check=[('pool_kernel','6')]
            measured=baseline['photo']['effects']
            if initial['params'].get('fog_mode')=='local_auto':
                controls_to_check.append(('fog_gain','0'))
            else:controls_to_check.append(('fog_extra','0.55'))
            for name,value in controls_to_check:
                changed=compose(scene,candidate,directory,meta,env,samples,{**initial['params'],name:value},
                                seed,job/(name+'-control.png'))
                with Image.open(changed['image']) as im:after=np.asarray(im.convert('RGB'))
                pixels=np.any(before!=after,axis=2)
                expected_skip=(name=='pool_kernel' and measured['pool_weight']==0 or
                               name=='fog_gain' and measured['fog_total_max']==0)
                if (not pixels.any() and not expected_skip) or pixels[~mask].any():
                    raise ValueError('optical_control_missing_or_changed_non_object_pixels')
                for field in ('alpha_hash','visible_alpha_hash','foreground_mask_hash','shadow_hash'):
                    if baseline[field]!=changed[field]:raise ValueError('optical_control_changed_'+field)
                controls.append({'asset_id':row['asset_id'],'effect':name,'value':value,
                    'image':changed['image'],'changed_pixels':int(pixels.sum()),'outside_changed':0,
                    'alpha_shadow_foreground_identical':True,'checks':changed['checks'],
                    'effects':changed['photo']['effects'],'visual_status':'unreviewed_control',
                    'skipped_reason':('insufficient_edge_samples_or_clear_air' if expected_skip else None)})
        if any(k.endswith('_renders') and v for k,v in pipe.renders.totals.items()):
            raise ValueError('optical_controls_caused_render')
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()
    report={'offline_same_images':0 if controls_only else len(replay),'controls':controls,'seconds':round(time.monotonic()-started,3),
            'mode':'controls_only' if controls_only else 'replay_and_controls',
            'api_requests':0,'optical_new_renders':0,'offline_run':summary['run'],
            'source':str(source),'completed_resume_new_work':None if controls_only else 0,
            'resume_seconds':resumed['run']['seconds'],
            'note':'控制图只验证参数生效与像素保护，不迁移 AI 分数'}
    atomic_json(out/'effects-verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('controls','offline_run')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--controls-only',action='store_true',help='Only verify current optical code on frozen EXRs; no scores are reused.')
    args=parser.parse_args()
    asyncio.run(run(resolve(args.batch),resolve(args.out),args.controls_only))
