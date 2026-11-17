"""Audit camera alignment using cached full-frame or new cropped renders."""
import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from display_batch.ai import Vision
from display_batch.common import atomic_json,resolve
from display_batch.compose import object_window,sample_environment,warp
from display_batch.rendering import RenderCache,reference_path,reference_resolution
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig
from display_batch.worker import Blender


async def main(batch,limit,out,cropped=False):
    batch=Path(batch);raw=json.loads((batch/'config.json').read_text(encoding='utf8'))
    raw.update(output=out,offline=True)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    config=BatchConfig.model_validate(raw);cache=resolve(config.cache)
    vision=Vision(config,cache/'vision',out/'usage.json')
    scenes=Scenes(cache,vision)
    worker=Blender(out/'worker',config.render_device)
    renderer=RenderCache(cache,worker)
    records=[];seen=set()
    try:
        for line in (batch/'results.jsonl').read_text(encoding='utf8').splitlines():
            record=json.loads(line)
            if not record['iterations'] or record['asset_id'] in seen:continue
            seen.add(record['asset_id']);first=record['iterations'][0]
            candidates=json.loads((batch/'jobs'/record['job_id']/'candidates.json').read_text(encoding='utf8'))['candidates']
            candidate=next(c for c in candidates if c['id']==first['params']['candidate'])
            asset=json.loads((cache/'assets'/record['asset_signature']/'result.json').read_text(encoding='utf8'))
            scene=await scenes.prepare(record['background'])
            environment,_=sample_environment(scene,candidate)
            params=first['params']
            old=[]
            for resolution in (512,reference_resolution(candidate)):
                folder,_=reference_path(cache,asset,candidate,params,environment,config.render_device,resolution)
                meta=json.loads((folder/'result.json').read_text(encoding='utf8'))
                old.append((folder,meta))
            low,high=old
            if cropped:
                environment,_=sample_environment(scene,candidate,scene_lighting=True)
                low=await renderer.get(asset,candidate,params,environment,scene['size'],512)
                high=await renderer.get(asset,candidate,params,environment,scene['size'],2048)
            scale=candidate['full_px'];a=np.asarray(low[1]['anchor_uv']);b=np.asarray(high[1]['anchor_uv'])
            error=float(np.linalg.norm((a-b)*scale))
            offset=np.asarray(candidate['anchor_px'])-a*scale
            bounds=first['edit_bounds'];masks=[]
            for rendered in (low,high):
                with Image.open(rendered[0]/'object.png') as image:alpha=np.asarray(image.getchannel('A'),dtype=float)/255
                masks.append(warp(alpha,scale,offset,bounds,window=object_window(rendered[1]))>.5)
            overlap=float((masks[0]&masks[1]).sum()/max(1,(masks[0]|masks[1]).sum()))
            differences=[]
            for reference,other in ((masks[0],masks[1]),(masks[1],masks[0])):
                distance=cv2.distanceTransform((~reference).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                differences.extend(distance[other&~reference].tolist())
            diff=np.zeros((*masks[0].shape,3),np.uint8)
            diff[masks[0]&masks[1]]=(185,185,185)
            diff[masks[0]&~masks[1]]=(255,145,45)
            diff[masks[1]&~masks[0]]=(65,165,255)
            diff_path=out/(record['asset_id']+'-alpha-difference.png')
            Image.fromarray(diff).save(diff_path)
            records.append({'job_id':record['job_id'],'asset_id':record['asset_id'],'device':high[1]['device'],
                            'resolutions':[low[1]['resolution'],high[1]['resolution']],
                            'anchor_error_px':error,'alpha_iou':overlap,
                            'thresholds':{'anchor_error_px':1.5,'alpha_iou':.975},
                            'anchor_pass':error<=1.5,'alpha_pass':overlap>=.975,
                            'mismatch_distance_p95_px':float(np.percentile(differences,95)) if differences else 0.,
                            'mismatch_distance_max_px':max(differences,default=0.),
                            'difference_image':str(diff_path),
                            'passed':error<=1.5 and overlap>=.975})
            if cropped:
                corrected=[]
                for folder,meta in old:
                    n=meta['resolution']
                    with Image.open(folder/'object.png') as image:
                        alpha=np.asarray(image.getchannel('A'),dtype=np.float32)/255
                    corrected.append(warp(alpha,scale,offset,bounds,window={'frame':n,'rect':[0,0,n,n]})>.5)
                records[-1]['old_fullframe_pixel_centres_iou']=float((corrected[0]&corrected[1]).sum()/
                    max(1,(corrected[0]|corrected[1]).sum()))
                records[-1]['render_windows']=[low[1]['object_window'],high[1]['object_window']]
            print(json.dumps({'asset':record['asset_id'],'alpha_iou':overlap,'anchor_error_px':error}),flush=True)
            if len(records)>=limit:break
        assert records,'no completed renders'
    finally:await worker.close();await vision.close()
    result={'kind':'actual_cropped_blender_renders' if cropped else 'actual_cached_blender_renders',
            'checks':records,'new_api_calls':len(vision.calls),'new_render_calls':renderer.totals['object_renders'],
            'passed':all(r['passed'] for r in records),'checked':len(records)}
    atomic_json(out/'checks.json',result);print(json.dumps(result,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--out',required=True)
    parser.add_argument('--cropped',action='store_true',help='also render and check cropped 512/2048 outputs')
    args=parser.parse_args();asyncio.run(main(args.batch,args.limit,args.out,args.cropped))
