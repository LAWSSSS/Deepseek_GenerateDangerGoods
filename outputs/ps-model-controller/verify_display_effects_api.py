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


async def run(source, out):
    started=time.monotonic()
    original=[json.loads(line) for line in (source/'results.jsonl').read_text(encoding='utf8').splitlines() if line]
    config=BatchConfig.model_validate({**read(source/'config.json'),'output':str(out),'offline':True})
    summary=await Pipeline(config).run()
    if summary['run']['state']!='complete' or summary['cost_total']['actual_requests']:
        raise ValueError('offline_replay_incomplete_or_made_api_requests')
    replay=[json.loads(line) for line in (out/'results.jsonl').read_text(encoding='utf8').splitlines() if line]
    if len(original)!=len(replay):raise ValueError('job_count_changed')
    for old,new in zip(original,replay,strict=True):
        if old['status']!=new['status'] or old['reason']!=new['reason'] or bool(old.get('image'))!=bool(new.get('image')):
            raise ValueError('offline_outcome_changed')
        if old.get('image') and file_hash(old['image'])!=file_hash(new['image']):
            raise ValueError('offline_image_changed')
    pipe=Pipeline(config)
    controls=[]
    try:
        jobs={job['id']:job['payload'] for job in pipe.store.rows()}
        for row in replay:
            if not row.get('image'):continue
            job=out/'jobs'/row['job_id'];initial=read(job/'iteration-0.json')
            candidate=next(c for c in read(job/'candidates.json')['candidates']
                           if c['id']==initial['params']['candidate'])
            asset=read(pipe.cache/'assets'/row['asset_signature']/'result.json')
            candidate['material_classes']=asset['semantic']['material_classes']
            scene=await pipe.scenes.prepare(row['background'])
            seed=jobs[row['job_id']]['seed']
            directory,meta,env,samples,hit=await pipe.render(asset,scene,candidate,initial['params'],final_resolution(candidate))
            if not hit:raise ValueError('optical_controls_should_reuse_renders')
            baseline=compose(scene,candidate,directory,meta,env,samples,initial['params'],seed,
                             job/'verified-initial.png',capture_local=True)
            if file_hash(baseline['image'])!=initial['image_hash']:
                raise ValueError('initial_png_did_not_replay')
            visible=baseline.pop('local_layers')['alpha']>0
            with Image.open(baseline['image']) as im:before=np.asarray(im.convert('RGB'))
            x,y,r,b=baseline['edit_bounds']
            mask=np.zeros(before.shape[:2],bool);mask[y:b,x:r]=visible
            for name,value in (('pool_kernel','6'),('fog_extra','0.55')):
                changed=compose(scene,candidate,directory,meta,env,samples,{**initial['params'],name:value},
                                seed,job/(name+'-control.png'))
                with Image.open(changed['image']) as im:after=np.asarray(im.convert('RGB'))
                pixels=np.any(before!=after,axis=2)
                if not pixels.any() or pixels[~mask].any():
                    raise ValueError('optical_control_missing_or_changed_non_object_pixels')
                for field in ('alpha_hash','visible_alpha_hash','foreground_mask_hash','shadow_hash'):
                    if baseline[field]!=changed[field]:raise ValueError('optical_control_changed_'+field)
                controls.append({'asset_id':row['asset_id'],'effect':name,'value':value,
                    'image':changed['image'],'changed_pixels':int(pixels.sum()),'outside_changed':0,
                    'alpha_shadow_foreground_identical':True,'checks':changed['checks'],
                    'effects':changed['photo']['effects'],'visual_status':'unreviewed_control'})
        if any(k.endswith('_renders') and v for k,v in pipe.renders.totals.items()):
            raise ValueError('optical_controls_caused_render')
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()
    report={'offline_same_images':len(replay),'controls':controls,'seconds':round(time.monotonic()-started,3),
            'api_requests':0,'optical_new_renders':0,'offline_run':summary['run'],
            'source':str(source),'note':'控制图只验证参数生效与像素保护，不迁移 AI 分数'}
    atomic_json(out/'effects-verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('controls','offline_run')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    asyncio.run(run(resolve(args.batch),resolve(args.out)))
