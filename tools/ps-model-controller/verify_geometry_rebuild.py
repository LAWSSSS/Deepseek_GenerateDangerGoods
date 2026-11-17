"""Offline PNG replay, camera-risk probes and optional real render alignment checks."""
import argparse
import asyncio
import copy
import json
import os
import time
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from display_batch.common import resolve,atomic_json,file_hash,implementation_manifest
from display_batch.schema import BatchConfig
from display_batch.pipeline import Pipeline
from display_batch.vehicle_scale import prepare as prepare_vehicle,check_projection_interval
from display_batch.compose import compose,sample_environment,warp,object_window
from display_batch.effects import lighting_for,dust_profile
from display_batch.local_atmosphere import prepare as prepare_air
from display_batch.rendering import RENDER_VERSION
from validate_reference_projection import read,records


def source_job(batch,row):
    reuse=row.get('reused_from')
    return Path(reuse['batch'])/'jobs'/reuse['job_id'] if reuse else batch/'jobs'/row['job_id']


def saved_render(cache,entry):
    parts={k:read(cache/RENDER_VERSION/k/v['signature']/'result.json') for k,v in entry['render_cache'].items()}
    obj=parts['object'];shadow=parts['with_shadow'];base=parts['without_shadow']
    assert obj['camera']==shadow['camera']==base['camera']
    assert shadow['window']==base['window'] and shadow['receiver_signature']==base['receiver_signature']
    meta={k:obj[k] for k in ('camera','device','anchor_uv','contact_polygons')}
    meta.update(object_window=obj['window'],shadow_window=shadow['window'],legacy_grid=False,
        part_paths={k:p['path'] for k,p in parts.items()},matte_materials=obj.get('matte_materials',[]),
        matte_profile=dust_profile(float(entry['params']['surface_dust']),entry['params']['weathering']),
        shadow_pair_validation={'status':'pass','camera_equal':True,'window':shadow['window'],
            'receiver_equal':True,'basis':'render_part_metadata_and_EXR_dimensions'})
    return Path(obj['path']),meta


async def verify(batch,out,render_check=False):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    pipe=Pipeline(BatchConfig.model_validate({**read(batch/'config.json'),'output':str(out),'offline':True}))
    rows=[];render_result=None;inputs=read(batch/'inputs.json')
    try:
        for row in records(batch):
            if not row.get('image'):
                assert row['status']=='failed' and row.get('rejections')
                rows.append({'job_id':row['job_id'],'generated':False,'failed_result_not_delivered':True});continue
            job=source_job(batch,row)
            entry=next(e for e in row['iterations'] if e['iteration']==row['best_iteration'])
            assert entry.get('review') and file_hash(row['image'])==entry['image_hash']
            candidate=copy.deepcopy(next(c for c in read(job/'candidates.json')['candidates'] if c['id']==entry['params']['candidate']))
            data=np.load(pipe.cache/'assets'/row['asset_signature']/'geometry.npz')
            scene=await pipe.scenes.prepare(row['background'])
            scene['vehicle_scale']=await prepare_vehicle(scene,pipe.config.vehicle_reference,pipe.vision,pipe.cache)
            scene['atmosphere']=await pipe.atmospheres.prepare(scene)
            interval=check_projection_interval(scene,candidate,candidate['terrain_physics'],data['vertices'],data['faces'])
            candidate['camera_interval_check']=interval
            params=entry['params'];folder,meta=saved_render(pipe.cache,entry)
            env,samples=sample_environment(scene,candidate,local=True)
            env['atmospheric_context']=prepare_air(scene,candidate,env,pipe.cache)
            env=lighting_for(scene,candidate,env,params['effects_lighting'],preserve=True)
            env['calibration']=entry['photo']['calibration']
            seed=next(j['seed'] for j in inputs if j['asset_id']==row['asset_id'] and j['background']==row['background'])
            reproduced=compose(scene,candidate,folder,meta,env,samples,params,seed,out/(row['job_id']+'.png'))
            assert file_hash(reproduced['image'])==entry['image_hash'],'PNG replay mismatch: '+row['job_id']
            for name in ('alpha_hash','visible_alpha_hash','foreground_mask_hash','shadow_hash'):
                assert reproduced[name]==entry[name],name
            assert not any(c['status']=='fail' for c in reproduced['checks'].values())
            with Image.open(row['image']) as image:assert image.size==tuple(scene['size'])
            with Image.open(row['image']) as image:overlay=image.convert('RGB')
            draw=ImageDraw.Draw(overlay,'RGBA')
            for site in scene['supports']:
                if site['id'] not in candidate['support_ids']:continue
                contours,_=cv2.findContours(site['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour)>2:draw.line([tuple(p) for p in contour[:,0]],fill=(40,230,150,255),width=5)
            for poly in candidate['contact_polygons']:
                xy=np.asarray(poly)*candidate['full_px']+candidate['offset']
                if len(xy)>2:draw.polygon([tuple(p) for p in xy],fill=(255,160,40,150))
            ax,ay=candidate['anchor_px'];draw.ellipse((ax-10,ay-10,ax+10,ay+10),outline='yellow',width=4)
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
            draw.text((30,150),f"{row['asset_id']} · 内宽 2.45m；物品 {candidate['scale']['physical_size']['nominal_mm']:.2f}mm · "
                '绿线为估计承托面，橙色为实际接触簇；相机与承托高度待审',font=font,fill='yellow',stroke_width=2,stroke_fill='black')
            diagnostic=out/(row['job_id']+'-support.png');overlay.save(diagnostic)
            rows.append({'job_id':row['job_id'],'asset_id':row['asset_id'],'background':row['background'],
                'generated':True,'exact_png_replay':True,'reviewed_png_hash':entry['image_hash'],
                'geometry_masks_unchanged':True,'checks':reproduced['checks'],
                'camera_interval_check':interval,'support_diagnostic':str(diagnostic)})
            if render_check and render_result is None:
                asset=read(pipe.cache/'assets'/row['asset_signature']/'result-strict.json')
                rendered=[]
                for n in (512,1024):
                    result=await pipe.render(asset,scene,candidate,params,n)
                    rendered.append(result)
                a,b=rendered;error=float(np.linalg.norm(np.subtract(a[1]['anchor_uv'],b[1]['anchor_uv']))*candidate['full_px'])
                masks=[]
                for d,m,*_ in rendered:
                    with Image.open(d/'object.png') as image:alpha=np.asarray(image.getchannel('A'),dtype=np.float32)/255
                    masks.append(warp(alpha,candidate['full_px'],candidate['offset'],entry['edit_bounds'],window=object_window(m))>.5)
                iou=float((masks[0]&masks[1]).sum()/max(1,(masks[0]|masks[1]).sum()))
                render_result={'asset_id':row['asset_id'],'alpha_iou':iou,'threshold':.975,
                    'anchor_error_px':error,'anchor_threshold_px':1.5,
                    'fixed_camera_including_roll':a[1]['camera']==b[1]['camera'],
                    'shadow_grid_unchanged':a[1]['shadow_window']==b[1]['shadow_window'],
                    'passed':bool(iou>=.975 and error<=1.5),'windows':[a[1]['object_window'],b[1]['object_window']]}
                assert render_result['passed'],render_result
            print(json.dumps({'verified':row['job_id'],'camera_risk_probes':len(interval['checks'])}),flush=True)
        assert not pipe.vision.calls,'offline validation sent an API request'
        result={'passed':True,'batch':str(batch),'implementation':implementation_manifest(),'jobs':rows,
            'exact_replays':sum(r.get('exact_png_replay',False) for r in rows),'render_alignment':render_result,
            'new_api_requests':0,'seconds':time.monotonic()-start,
            'render_counts':dict(pipe.renders.totals),'worker_seconds':pipe.blender.seconds,
            'note':'Same reviewed PNG bytes; uncertainty probes only amend evidence, not pixels or scores.'}
        atomic_json(out/'verification.json',result)
        atomic_json(out/'runs'/f'{time.time_ns()}.json',{
            k:v for k,v in result.items() if k not in ('jobs','implementation')})
        print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','implementation')},ensure_ascii=False),flush=True)
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--out',required=True)
    p.add_argument('--render-check',action='store_true');a=p.parse_args()
    asyncio.run(verify(resolve(a.batch),resolve(a.out),a.render_check))
