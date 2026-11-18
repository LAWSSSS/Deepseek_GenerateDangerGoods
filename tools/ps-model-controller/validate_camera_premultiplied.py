"""CPU-only fixed-geometry comparison of archived versus premultiplied camera paths."""
import argparse
import asyncio
import copy
import html
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image

import display_batch.compose as compose_module
import display_batch.scene as scene_module
from display_batch.rendering import source_path
from display_batch.common import atomic_json,file_hash,load_catalog,resolve,signature
from display_batch.compose import compose,sample_environment
from display_batch.effects import lighting_for
from display_batch.local import camera_residual
from display_batch.photography import linear
from display_batch.scene import Scenes
from scene_v2 import Perception,digest,key
from validate_coherent_expanded import source_hashes
from verify_geometry_rebuild import saved_render

SOURCE='outputs/综合融合-12类新场景-20261112'
OUTPUT='outputs/相机合成路径-离线16图-20261112-v2'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


class CachedSceneAnswers:
    async def ask(self,kind,schema,prompt,context,images):
        if kind!='scene':raise RuntimeError('offline_camera_audit_forbids_model_requests')
        return schema.model_validate(read(Path(images[0]).parent/'analysis.json'))


def cached_perception(original):
    def cached(engine,source):
        path=engine.cache/f'scene-{key(digest(source),engine.versions)}.npz'
        if not path.is_file():raise RuntimeError('offline_camera_audit_forbids_perception_inference')
        return original(engine,source)
    return cached


def percentiles(values):
    return np.percentile(values,[50,95,100]).tolist() if values.size else None


def sensitivity(args,kwargs,stats,core):
    """Change only hidden background; keep object, camera noise and atmosphere fixed."""
    shaded,rgb,alpha,background,env,params,seed,writable=args
    yy,xx=np.indices(alpha.shape)
    delta=(.012*np.where((xx+yy)%2,1.,-1.)*core)[...,None]
    changed=np.maximum(background+delta,0)
    changed_shaded=np.maximum(shaded+delta,0)
    value={}
    for mode in ('legacy','premultiplied'):
        setting={**params,'camera_pipeline':mode}
        original=camera_residual(shaded,rgb,alpha,background,env,setting,seed,writable,
                                 object_delta=kwargs.get('object_delta'),fixed_sensor=stats)[0]
        modified=camera_residual(changed_shaded,rgb,alpha,changed,env,setting,seed,writable,
                                 object_delta=kwargs.get('object_delta'),fixed_sensor=stats)[0]
        value[mode]=percentiles(np.max(abs(original-modified),axis=2)[core]*255)
    return value


def page(out,result):
    top={k:v for k,v in result.items() if k not in ('cases','source_hashes')}
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>相机合成路径 · 固定几何离线诊断</title>',
        '<style>body{background:#172128;color:#eef2f4;font:16px system-ui;margin:28px auto;padding:0 18px;max-width:1700px}img{display:block;width:100%;height:auto}a{color:#acdfcf}section{margin:30px 0;border-top:1px solid #53636b}p{line-height:1.7}pre{white-space:pre-wrap}</style>',
        '<h1>相机合成路径 · 固定几何离线诊断</h1>',
        '<p>复用已生成的 EXR、摆位、尺度、遮挡与阴影，仅比较相机合成公式。全部为完整监控画面，未新增 GPU 渲染或 API 复核；这些图不是新一轮合格交付。</p>',
        '<p>旧路径会把被遮挡背景的高频残差加回不透明物体。新版先合并物品 RGB、薄雾和池化，再一次过滤预乘透明度的物品图层。</p>',
        '<p><a href="analysis.json">逐图检查、背景独立性反例及耗时</a> · <a href="manifest.json">冻结输入</a></p>',
        '<details><summary>汇总</summary><pre>'+html.escape(json.dumps(top,ensure_ascii=False,indent=2))+'</pre></details>']
    feedback=out/'user-feedback.json'
    if feedback.is_file():
        note=read(feedback)
        body.insert(5,'<p style="padding:16px;background:#782d2d;color:white">用户结论：'+
            html.escape(note['conclusion'])+'。这些旧几何图片仅留作问题诊断，未采用为新效果基线。</p>')
    for row in result['cases']:
        body.append('<section><h2>'+html.escape(row['group']+' · '+row['asset_id']+' · '+row['job_id'])+'</h2>')
        if row.get('error'):
            body.append('<p>'+html.escape(row['error'])+'</p></section>');continue
        p=row['opaque_background_sensitivity_rgb255']
        body.append('<p>被遮挡背景改变后的物体内部响应 P95：旧路径 '+format(p['legacy'][1],'.4f')+
                    '，新版 '+format(p['premultiplied'][1],'.6f')+'（0–255 RGB级）。这是因果诊断，不是视觉评分。</p>')
        for title,key_name in [('当前版本 · 精确回放','before_full'),('预乘合成 · 固定几何离线试验','after_full')]:
            link=html.escape(Path(os.path.relpath(row[key_name],out)).as_posix(),quote=True)
            body.append('<p>'+title+'</p><a href="'+link+'"><img loading="lazy" src="'+link+'" alt="完整画面"></a>')
        body.append('</section>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf8')


async def run(source,out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    batch=Path(read(source/'selected-run.json')['batch'])
    rows=[json.loads(s) for s in (batch/'results.jsonl').read_text(encoding='utf8').splitlines() if s]
    rows=[r for r in rows if r.get('image')]
    hashes=source_hashes(load_catalog());config=read(batch/'config.json');cache=resolve(config['cache'])
    frozen={'source_batch':str(batch),'source_hashes':hashes,'cases':[
        {'job_id':r['job_id'],'asset_id':r['asset_id'],'background':r['background'],
         'image_sha256':file_hash(r['image']),'background_sha256':file_hash(r['background']),
         'best_iteration':r['best_iteration']} for r in rows],
        'comparison':'legacy versus camera_pipeline=premultiplied','geometry':'frozen',
        'camera_source_hash':file_hash(source_path('display_batch/local.py'))}
    if (out/'manifest.json').is_file() and signature(read(out/'manifest.json'))!=signature(frozen):
        raise ValueError('offline_comparison_inputs_changed_use_new_directory')
    atomic_json(out/'manifest.json',frozen)
    scenes=Scenes(cache,CachedSceneAnswers());inputs=read(batch/'inputs.json');cases=[]
    with patch.object(Perception,'scene',cached_perception(Perception.scene)),patch.object(scene_module,'atomic_json',lambda *a,**k:None):
        for row in rows:
            info={k:row[k] for k in ('job_id','asset_id','group')}
            try:
                job=batch/'jobs'/row['job_id']
                entry=next(e for e in row['iterations'] if e['iteration']==row['best_iteration'])
                candidate=copy.deepcopy(next(c for c in read(job/'candidates.json')['candidates'] if c['id']==entry['params']['candidate']))
                scene=await scenes.prepare(row['background']);folder,meta=saved_render(cache,entry)
                env,samples=sample_environment(scene,candidate,local=True);env['integration_mode']='coherent'
                atmosphere_key=entry['photo']['atmosphere']['signature'];af=cache/'local-atmosphere'/atmosphere_key
                env['atmospheric_context']={'signature':atmosphere_key,'folder':str(af),'summary':read(af/'summary.json')}
                env=lighting_for(scene,candidate,env,entry['params']['effects_lighting'],preserve=True)
                env['calibration']=entry['photo']['calibration']
                seed=next(j['seed'] for j in inputs if j['asset_id']==row['asset_id'] and j['background']==row['background'])
                before=out/(row['job_id']+'-legacy.png');after=out/(row['job_id']+'-premultiplied.png')
                capture={}
                def recorder(*args,**kwargs):
                    value=camera_residual(*args,**kwargs);capture.update(args=args,kwargs=kwargs,stats=value[2]);return value
                with patch.object(compose_module,'camera_residual',recorder):
                    old=compose(scene,candidate,folder,meta,env,samples,entry['params'],seed,before,capture_local=True)
                if file_hash(before)!=entry['image_hash']:raise ValueError('legacy_PNG_not_byte_identical')
                newer={**entry['params'],'camera_pipeline':'premultiplied'}
                new=compose(scene,candidate,folder,meta,env,samples,newer,seed,after,capture_local=True)
                if any(old[k]!=new[k] for k in ('alpha_hash','visible_alpha_hash','foreground_mask_hash','shadow_hash')):
                    raise ValueError('frozen_geometry_shadow_or_occlusion_changed')
                with Image.open(before) as im:a=np.asarray(im.convert('RGB'),np.int16)
                with Image.open(after) as im:b=np.asarray(im.convert('RGB'),np.int16)
                x,y,r,bottom=old['edit_bounds'];difference=np.max(abs(a[y:bottom,x:r]-b[y:bottom,x:r]),axis=2)
                alpha=old['local_layers']['alpha'];writable=old['local_layers']['writable']
                depth=cv2.distanceTransform((alpha>.999).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                core=(depth>6)&writable
                outside=alpha==0;dist=cv2.distanceTransform(outside.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                ring=outside&(dist<=3)&writable
                count,labels,stats,_=cv2.connectedComponentsWithStats(outside.astype(np.uint8),connectivity=4)
                centers=np.zeros_like(outside)
                for n in range(1,count):
                    left,top,width,height,_=stats[n]
                    if left>0 and top>0 and left+width<alpha.shape[1] and top+height<alpha.shape[0]:
                        centers|=(labels==n)&(dist>max(3.,4*new['photo']['blur_sigma']))
                protected_changes=int((difference[~writable]>0).sum())
                if protected_changes:raise ValueError('foreground_or_protected_pixels_changed')
                bg=capture['args'][3];sigma=old['photo']['blur_sigma']
                highpass=bg-cv2.GaussianBlur(bg,(0,0),sigma) if sigma else np.zeros_like(bg)
                counterfactual=sensitivity(capture['args'],capture['kwargs'],capture['stats'],core)
                if counterfactual['premultiplied'] and counterfactual['premultiplied'][2]>.002:
                    raise ValueError('opaque_output_depends_on_hidden_background')
                info.update(before_full=str(before),after_full=str(after),exact_legacy_replay=True,
                    geometry_shadow_occlusion_unchanged=True,output_size=list(scene['size']),
                    changed_pixels=int(np.any(a!=b,axis=2).sum()),opaque_core_pixels=int(core.sum()),
                    core_difference_rgb255=percentiles(difference[core]),edge_band_difference_rgb255=percentiles(difference[ring]),
                    hidden_background_highpass_linear=percentiles(np.max(abs(highpass),axis=2)[core]),
                    opaque_background_sensitivity_rgb255=counterfactual,
                    foreground_or_protected_changes=protected_changes,hole_center_pixels=int(centers.sum()),
                    hole_center_changes=int((difference[centers]>0).sum()),
                    old_photo=old['photo'],new_photo=new['photo'],new_program_checks=new['checks'],
                    review_status='offline_unreviewed_not_delivery')
            except Exception as error:
                info['error']=type(error).__name__+': '+str(error)
            cases.append(info)
            print(json.dumps({k:v for k,v in info.items() if k not in ('old_photo','new_photo','new_program_checks')},ensure_ascii=False),flush=True)
    unchanged=hashes==source_hashes(load_catalog())
    result={'cases':cases,'planned':len(rows),'compared':sum('error' not in c for c in cases),
        'new_api_requests':0,'new_render_count':0,'api_cost':0,'local_compute_cost':None,
        'seconds':time.monotonic()-started,'source_hashes':hashes,'source_models_unchanged':unchanged,
        'quality_status':'fixed_geometry_CPU_diagnostic_no_new_visual_acceptance','batch_resumed':False}
    atomic_json(out/'analysis.json',result);page(out,result)
    if not unchanged:raise ValueError('source_models_changed')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',default=SOURCE);parser.add_argument('--out',default=OUTPUT)
    args=parser.parse_args();asyncio.run(run(resolve(args.source),resolve(args.out)))
