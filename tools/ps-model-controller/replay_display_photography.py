"""Fixed-geometry photographic replay. Never settles, changes placement, or edits old runs."""
import argparse
import asyncio
import html
import json
import os
from pathlib import Path
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from display_batch.ai import Vision,ProviderUnavailable
from display_batch.common import ROOT,atomic_json,signature,file_hash,implementation_manifest
from display_batch.compose import compose,sample_environment,warp
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig,Review,apply_actions,PHOTO_CONTROLS,MATERIAL_PROFILES,LIGHT_PROFILES
from display_batch.pipeline import REVIEW_PROMPT
from display_batch.rendering import reference_resolution, REFERENCE_RENDER_VERSION
from display_batch.worker import Blender


def photo_pass(review):
    return min(review.scores.lighting,review.scores.material,review.scores.texture)>=3 and not any(
        d.severity in ('major','fatal') and d.code in ('color','lighting','material','texture') for d in review.defects)


def photo_rank(review):
    scores=review.scores
    return (photo_pass(review),min(scores.lighting,scores.material,scores.texture),
            scores.lighting+scores.material+scores.texture)


def old_render(cache,asset,candidate,params,environment):
    keys=('world_color','light_color','key_direction','haze','sample_count','sample_hash','luma_median','basis')
    env={k:environment[k] for k in keys}
    request={'mesh':asset['mesh'],'matrix':candidate['matrix'],'yaw':candidate['yaw'],
        'plane_normal':candidate.get('plane_normal',[0,0,1]),'elevation':candidate['elevation'],
        'environment':env,'resolution':reference_resolution(candidate),
        'roughness_delta':MATERIAL_PROFILES[params['material']],
        'dust_gain':1.6 if params['material']=='weathered' else 1.,'light_gain':LIGHT_PROFILES[params['light']]}
    key=signature('pbr-linear-shadow-v7',asset['mesh_hash'],request,'OPTIX','blender-4.5.3')
    folder=cache/'renders'/key
    if not (folder/'result.json').is_file():raise ValueError('legacy render cache missing: '+key)
    return folder,json.loads((folder/'result.json').read_text(encoding='utf8'))


def color_comparison(scene,candidate,render,meta,old_image,new_image,samples,bounds):
    x,y,r,b=bounds
    alpha=np.asarray(Image.open(render/'object.png').getchannel('A'),np.float32)/255
    offset=np.asarray(candidate['anchor_px'])-np.asarray(meta['anchor_uv'])*candidate['full_px']
    mask=warp(alpha,candidate['full_px'],offset,bounds)>.9
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    mask &= ~((scene['depth_native'][y:b,x:r]>site['depth']+.10)&~site['mask'][y:b,x:r])
    mask &= ~scene['protected'][y:b,x:r]
    values=[]
    for path in (old_image,new_image):
        rgb=np.asarray(Image.open(path).convert('RGB'))[y:b,x:r]/255
        lab=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2LAB)
        values.append(float(np.median(np.linalg.norm(lab[:,:,1:][mask],axis=1))))
    pixels=np.asarray(scene['frame'])[samples]
    lab=cv2.cvtColor(np.float32(pixels[:,None,:]/255),cv2.COLOR_RGB2LAB)
    reference=float(np.median(np.linalg.norm(lab[:,:,1:],axis=2)))
    excess=max(0,values[0]-reference)
    return {'old_chroma':values[0],'new_chroma':values[1],'scene_chroma':reference,
            'baseline_excess':excess,'excess_reduction':1-max(0,values[1]-reference)/excess if excess>2 else None}


def report(out,records,usage,seconds,render_calls=0):
    scored=[r for r in records if r.get('review')]
    summary={'kind':'fixed_geometry_photography','images':len(records),'reviewed':len(scored),
        'photo_pass':sum(r.get('photo_pass',False) for r in records),
        'photo_pass_rate':sum(r.get('photo_pass',False) for r in records)/max(1,len(records)),
        'target':.8,'new_physics_calls':0,'new_render_calls':render_calls,'seconds':round(seconds,3),
        'usage':{k:v for k,v in usage.items() if k!='calls'},'records':records}
    atomic_json(out/'results.json',summary)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    sections=[]
    for row in records:
        old=Image.open(row['old_crop']).convert('RGB');new=Image.open(row['crop']).convert('RGB')
        board=Image.new('RGB',(1000,430),(27,36,41));draw=ImageDraw.Draw(board)
        draw.text((12,8),f"{row['id']} · {row['asset_id']} · 摄影{'达标' if row.get('photo_pass') else '待审'}",font=font,fill='white')
        for i,im in enumerate((old,new)):
            im.thumbnail((480,340));board.paste(im,(i*500+(500-im.width)//2,65+(340-im.height)//2))
            draw.text((i*500+15,36),'原版' if i==0 else '摄影新版',font=font,fill='white')
        compare=out/(row['id']+'-compare.jpg');board.save(compare,quality=90)
        rel=os.path.relpath(row['image'],out).replace('\\','/')
        sections.append(f'<section><a href="{html.escape(rel)}"><img src="{compare.name}" loading="lazy"></a></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>固定摆位摄影对照</title>'
        '<style>body{background:#172128;color:#eee;font:16px sans-serif;margin:24px}img{max-width:100%}</style>'
        f'<h1>固定摆位摄影对照：{len(records)} 张</h1><p>摄影达标 {summary["photo_pass"]}/{len(scored)}；'
        '不改变原图的几何和正式验收状态。点击对照图查看原尺寸新版。</p>'+''.join(sections),encoding='utf8')
    return summary


async def main(args):
    if not 0<=args.max_iterations<=10:raise ValueError('max-iterations must be 0..10')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'work/display-batch-cache';config=BatchConfig(backgrounds='unused',output=str(out))
    vision=Vision(config,cache/'vision',out/('usage-'+time.strftime('%Y%m%d-%H%M%S')+'.json'))
    scenes=Scenes(cache,vision,legacy=True);records=[];start=time.monotonic();count=0;render_calls=0
    blender=Blender(cache/('photo-worker-'+str(os.getpid())))
    atomic_json(out/'implementation.json',implementation_manifest())
    try:
        for batch_path in args.batches:
            batch=Path(batch_path)
            for line in (batch/'results.jsonl').read_text(encoding='utf8').splitlines():
                row=json.loads(line)
                if not row.get('image') or not row.get('iterations'):continue
                if args.limit and count>=args.limit:break
                count+=1;ident=f'{batch.name}-{row["job_id"]}';jobout=out/'jobs'/ident;jobout.mkdir(parents=True,exist_ok=True)
                record_path=jobout/'result.json'
                if record_path.is_file():records.append(json.loads(record_path.read_text(encoding='utf8')));continue
                best_old=next(it for it in row['iterations'] if it['iteration']==row.get('best_iteration',0))
                candidates=json.loads((batch/'jobs'/row['job_id']/'candidates.json').read_text(encoding='utf8'))['candidates']
                candidate=dict(next(c for c in candidates if c['id']==best_old['params']['candidate']))
                candidate['legacy_occlusion']=True
                asset=json.loads((cache/'assets'/row['asset_signature']/'result.json').read_text(encoding='utf8'))
                candidate['material_classes']=asset['semantic']['material_classes']
                scene=await scenes.prepare(row['background'])
                environment,samples=await asyncio.to_thread(sample_environment,scene,candidate)
                render,meta=old_render(cache,asset,candidate,best_old['params'],environment)
                old_render_dir=render
                if args.rerender_materials:
                    candidate.update(fixed_alpha_dir=str(render),fixed_shadow_dir=str(render))
                    request={'mesh':asset['mesh'],'matrix':candidate['matrix'],'yaw':candidate['yaw'],
                        'camera':meta['camera'],'plane_normal':candidate.get('plane_normal',[0,0,1]),
                        'elevation':candidate['elevation'],'environment':environment,'resolution':meta['resolution'],
                        'roughness_delta':MATERIAL_PROFILES[best_old['params']['material']],
                        'dust_gain':1.6,'light_gain':LIGHT_PROFILES[best_old['params']['light']]}
                    new_render=cache/'photo-renders'/signature(REFERENCE_RENDER_VERSION,asset['mesh_hash'],request)
                    new_render.mkdir(parents=True,exist_ok=True);metadata_path=new_render/'result.json'
                    if metadata_path.is_file():new_meta=json.loads(metadata_path.read_text(encoding='utf8'))
                    else:
                        new_meta=await blender.ask('render',{**request,'output':str(new_render)},timeout=240)
                        atomic_json(metadata_path,new_meta);render_calls+=1
                    assert np.linalg.norm(np.subtract(new_meta['anchor_uv'],meta['anchor_uv']))*candidate['full_px']<1e-6
                    render,meta=new_render,new_meta
                params={**best_old['params'],'photo_style':'scrap_gray',**{k:'auto' for k in PHOTO_CONTROLS}}
                versions=[];seen=set();best=None
                for iteration in range(args.max_iterations+1):
                    if signature(params) in seen:break
                    seen.add(signature(params));image=jobout/f'iteration-{iteration}.png'
                    comp=await asyncio.to_thread(compose,scene,candidate,render,meta,environment,samples,params,row.get('seed',17),image)
                    review=None;review_error=None
                    if not args.no_review:
                        try:
                            review=await vision.ask('fixed-photo-review',Review,REVIEW_PROMPT+
                                ' The user accepts existing geometry/pose and requests strong grey dusty blending. '
                                'Assess photography separately; never demand saturated original paint. '
                                'Actions may only adjust the supplied photographic controls.',
                                {'allowed':{k:list(v) for k,v in PHOTO_CONTROLS.items()},'params':params,
                                 'name':asset['name'],'fixed_geometry':True,'photo':comp['photo']},
                                [comp['preview'],comp['crop'],asset['board']])
                        except ProviderUnavailable:raise
                        except (ValueError,RuntimeError) as error:review_error=str(error)
                    entry={**comp,'params':dict(params),'iteration':iteration,'review':review.model_dump() if review else None,
                           'review_error':review_error}
                    versions.append(entry);atomic_json(jobout/f'iteration-{iteration}.json',entry)
                    if best is None or (review and (best['review'] is None or photo_rank(review)>photo_rank(Review.model_validate(best['review'])))):best=entry
                    if not review or photo_pass(review):break
                    actions=[a for a in review.actions if a.kind in PHOTO_CONTROLS]
                    if not actions:break
                    params=apply_actions(params,actions,[candidate['id']])
                record={'id':ident,'asset_id':row['asset_id'],'background':row['background'],'old_image':row['image'],
                    'old_crop':row['crop'],**best,'photo_pass':bool(best['review'] and photo_pass(Review.model_validate(best['review']))),
                    'fixed_candidate':candidate,'versions':versions,'status':'review','formal_status':'unchanged',
                    'supplemental_render':str(render) if args.rerender_materials else None,
                    'comparison':color_comparison(scene,candidate,old_render_dir,meta,row['image'],best['image'],samples,best['edit_bounds'])}
                atomic_json(record_path,record);records.append(record)
                report(out,records,vision.usage(),time.monotonic()-start,render_calls)
                print(json.dumps({'done':len(records),'asset':row['asset_id'],'photo_pass':record['photo_pass'],
                    'chroma':record['comparison']},ensure_ascii=False),flush=True)
        summary=report(out,records,vision.usage(),time.monotonic()-start,render_calls)
        print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
    finally:await blender.close();await vision.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batches',nargs='+',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--max-iterations',type=int,default=2);parser.add_argument('--no-review',action='store_true')
    parser.add_argument('--rerender-materials',action='store_true')
    asyncio.run(main(parser.parse_args()))
