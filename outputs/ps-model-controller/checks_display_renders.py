"""Compare actual cached low/final-resolution Blender outputs; never requests a new render."""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from PIL import Image

from display_batch.common import ROOT,atomic_json
from display_batch.compose import warp
from display_batch.pipeline import Pipeline,final_resolution
from display_batch.schema import BatchConfig


async def main(batch,limit):
    batch=Path(batch);raw=json.loads((batch/'config.json').read_text(encoding='utf8'))
    raw.update(output=str(ROOT/'outputs/多背景批量-渲染回归-20261110'),offline=True)
    pipe=Pipeline(BatchConfig.model_validate(raw))
    async def missing(*args,**kwargs):raise AssertionError('actual render evidence missing; no new render permitted')
    pipe.blender.ask=missing
    records=[];seen=set()
    try:
        for line in (batch/'results.jsonl').read_text(encoding='utf8').splitlines():
            record=json.loads(line)
            if not record['iterations'] or record['asset_id'] in seen:continue
            seen.add(record['asset_id']);first=record['iterations'][0]
            candidates=json.loads((batch/'jobs'/record['job_id']/'candidates.json').read_text(encoding='utf8'))['candidates']
            candidate=next(c for c in candidates if c['id']==first['params']['candidate'])
            asset=await pipe.prepare_asset(record['asset_id']);scene=await pipe.scenes.prepare(record['background'])
            low=await pipe.render(asset,scene,candidate,first['params'],512)
            high=await pipe.render(asset,scene,candidate,first['params'],final_resolution(candidate))
            assert low[-1] and high[-1]
            scale=candidate['full_px'];a=np.asarray(low[1]['anchor_uv']);b=np.asarray(high[1]['anchor_uv'])
            error=float(np.linalg.norm((a-b)*scale));assert error<=1.5
            offset=np.asarray(candidate['anchor_px'])-a*scale
            bounds=first['edit_bounds'];masks=[]
            for rendered in (low,high):
                with Image.open(rendered[0]/'object.png') as image:alpha=np.asarray(image.getchannel('A'),dtype=float)/255
                masks.append(warp(alpha,scale,offset,bounds)>.5)
            overlap=float((masks[0]&masks[1]).sum()/max(1,(masks[0]|masks[1]).sum()))
            assert overlap>=.975,(record['job_id'],overlap)
            records.append({'job_id':record['job_id'],'asset_id':record['asset_id'],'device':high[1]['device'],
                            'resolutions':[low[1]['resolution'],high[1]['resolution']],
                            'anchor_error_px':error,'alpha_iou':overlap,'passed':True})
            if len(records)>=limit:break
        assert records,'no completed renders'
    finally:await pipe.blender.close();await pipe.vision.close();pipe.store.close()
    result={'kind':'actual_cached_blender_renders','checks':records,'new_api_calls':len(pipe.vision.calls),'new_render_calls':0}
    atomic_json(pipe.out/'checks.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--limit',type=int,default=3)
    args=parser.parse_args();asyncio.run(main(args.batch,args.limit))
