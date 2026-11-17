"""Staged, reproducible display evaluation using the prior 138-task baseline."""
import argparse
import asyncio
from collections import Counter
import html
import itertools
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import Field

from display_batch.ai import ProviderUnavailable, Vision
from display_batch.common import atomic_json, file_hash, implementation_manifest, resolve, signature
from display_batch.compose import compose, sample_environment
from display_batch.pipeline import Pipeline, REVIEW_PROMPT, batch_lock
from display_batch.framing import final_resolution
from display_batch.rendering import RenderCache, reference_path
from display_batch.review import evidence_status, inspect
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig, Review, Score, Strict
from display_batch.worker import Blender


ASSETS=('3','6','10','13','19-1','31')
PROFILES={'B':'camera','C':'materials','D':'scene'}
PHOTO_PROMPT="""Judge lighting/color, material and camera texture in each of the four labelled crops.
The four share the same object, placement, camera, scale and foreground occlusion. Use the panorama
for scene context and the model board for material identity. The requested look is faded grey scrap.
Usable is 3, convincing is 4. Check for artificial highlights, flat haze, false grain, loss of metal
response and plastic turning metallic. Mark serious photographic faults and give short concrete evidence.
Use the same standard for all four labels, without inferring which is a new version."""


class PhotoJudgment(Strict):
    lighting: Score
    material: Score
    texture: Score
    serious: bool
    evidence: str = Field(max_length=120)

    def passes(self):
        return not self.serious and min(self.lighting,self.material,self.texture)>=3


class PhotoComparison(Strict):
    A: PhotoJudgment
    B: PhotoJudgment
    C: PhotoJudgment
    D: PhotoJudgment


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def baseline_rows(baseline):
    inputs=read(baseline/'inputs.json')
    rows=[json.loads(line) for line in (baseline/'results.jsonl').read_text(encoding='utf8').splitlines()]
    data={}
    for row in rows:
        i=int(row['job_id'].split('-')[0])-1
        job=inputs[i]
        key=(job['background'],job['asset_id'])
        if key!=(row['background'],row['asset_id']) or key in data or not row.get('image'):
            raise ValueError('baseline requires one completed image for each background/model pair')
        data[key]=(row,job)
    if len(data)!=138 or set(k[1] for k in data)!=set(ASSETS):
        raise ValueError('expected the complete six-model, 23-background baseline')
    return data


def choose_backgrounds(data,cache,seed):
    paths=sorted({k[0] for k in data})
    random.Random(seed).shuffle(paths)
    values=[]
    for path in paths:
        row=data[(path,ASSETS[0])][0]
        answer=read(cache/'scenes'/row['scene_signature']/'analysis.json')
        with Image.open(path) as image:
            gray=np.asarray(ImageOps.exif_transpose(image).convert('L').resize((160,90)))
        p20,p50,p80=np.percentile(gray,[20,50,80])
        slopes=[s['slope_deg'] for s in answer['surfaces']]
        values.append([p50,p80-p20,answer['haze'],len(slopes),np.mean(slopes) if slopes else 0.])
    features=np.asarray(values)
    features=(features-features.mean(axis=0))/np.maximum(features.std(axis=0),1e-6)
    distances=np.linalg.norm(features[:,None]-features[None,:],axis=-1)
    medoids=[int(np.argmin(distances.sum(axis=1)))]
    while len(medoids)<6:
        remaining=[i for i in range(len(paths)) if i not in medoids]
        medoids.append(max(remaining,key=lambda i:distances[i,medoids].min()))
    for _ in range(8):
        groups=np.argmin(distances[:,medoids],axis=1)
        groups[medoids]=np.arange(len(medoids))
        next_medoids=[]
        for i,m in enumerate(medoids):
            members=np.flatnonzero(groups==i)
            next_medoids.append(int(members[np.argmin(distances[np.ix_(members,members)].sum(axis=1))])
                                if len(members) else m)
        if medoids==next_medoids:break
        medoids=next_medoids
    first,second=max(itertools.combinations(medoids,2),key=lambda pair:distances[pair])
    return {'seed':seed,'fixed':[paths[first],paths[second]],'pilot':[paths[i] for i in medoids],
            'all':paths,'features':{p:v for p,v in zip(paths,values)},
            'feature_names':['luma_median','luma_p80_minus_p20','haze','surface_count','mean_slope_deg']}


def selected_geometry(baseline,row):
    old=next(x for x in row['iterations'] if x['iteration']==row['best_iteration'])
    candidates=read(baseline/'jobs'/row['job_id']/'candidates.json')['candidates']
    candidate=next(c for c in candidates if c['id']==old['params']['candidate'])
    return old,candidate


def montage(pictures,path,*,large=False):
    w,h=(960,580) if large else (640,440)
    result=Image.new('RGB',(w*2,h*2),(27,34,40));draw=ImageDraw.Draw(result)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),24 if large else 20) if font_path.is_file() else ImageFont.load_default()
    for i,(name,file) in enumerate(pictures):
        x,y=i%2*w,i//2*h
        with Image.open(file) as image:
            image=ImageOps.contain(image.convert('RGB'),(w-20,h-45),Image.Resampling.LANCZOS)
            result.paste(image,(x+(w-image.width)//2,y+36+(h-42-image.height)//2))
        draw.text((x+12,y+6),name,font=font,fill='white')
    result.save(path,quality=92)


def fixed_audit(scene,old,candidate,meta,variants):
    frame=np.asarray(scene['frame']);old_mask=old['alpha_hash']
    protected=scene['protected'];x,y,r,b=old['edit_bounds']
    outside=np.ones(frame.shape[:2],bool);outside[y:b,x:r]=False
    checks={}
    for label,value in variants.items():
        with Image.open(value['image']) as im:image=np.asarray(im.convert('RGB'))
        correct_size=image.shape==frame.shape
        changed=np.any(image!=frame,axis=2) if correct_size else np.ones_like(outside)
        checks[label]={'size':correct_size,'outside_pixels':int((changed&outside).sum()),
                       'protected_pixels':int((changed&protected).sum()),
                       'alpha_equal':value['alpha_hash']==old_mask,
                       'occlusion_equal':value['visible_alpha_hash']==old['visible_alpha_hash'],
                       'anchor_error_px':value['checks']['anchor']['projection_error_px'],
                       'hard_checks':evidence_status(value['checks'])}
    return {'variants':checks,'pass':all(c['size'] and not c['outside_pixels'] and not c['protected_pixels']
            and c['alpha_equal'] and c['occlusion_equal'] and c['anchor_error_px']<=1.5 and c['hard_checks']!='fail'
            for c in checks.values()),
            'candidate_signature':signature(candidate),'camera_signature':signature(meta['camera'])}


def aov_audit(folder):
    def load(label):
        return cv2.imread(str(folder/(label+'0001.exr')),cv2.IMREAD_UNCHANGED)[:,:,:3]
    combined=load('combined')
    other=load('diffuse')+load('specular')
    with Image.open(folder/'object.png') as image:
        core=np.asarray(image.getchannel('A'))>240
    residual=np.mean(np.abs(combined-other),axis=2)[core]
    return {'median_linear_residual':float(np.median(residual)),
            'p95_linear_residual':float(np.percentile(residual,95)),
            'compositing_basis':'denoised_combined_with_smoothed_specular_guide'}


async def fixed_case(folder,baseline,row,job,scene,asset,render_cache,cache,device,seed,vision):
    old,geometry=selected_geometry(baseline,row)
    params=old['params']
    env,sample=await asyncio.to_thread(sample_environment,scene,geometry)
    original,request=reference_path(cache,asset,geometry,params,env,device)
    old_meta=read(original/'result.json')
    geometry={**geometry,'material_classes':asset['semantic']['material_classes']}
    lighting,_=await asyncio.to_thread(sample_environment,scene,geometry,scene_lighting=True)
    rendered,meta,hit=await render_cache.get(asset,geometry,params,lighting,scene['size'],
                                             request['resolution'],full_frame=True)
    anchor_error=float(np.linalg.norm(np.subtract(meta['anchor_uv'],old_meta['anchor_uv']))*geometry['full_px'])
    if anchor_error>1e-5 or meta['camera']!=old_meta['camera']:
        raise ValueError('fixed_geometry_camera_or_anchor_changed')
    variants={'A':{**old,'image':row['image'],'crop':row['crop']}}
    for label,style in PROFILES.items():
        frozen={**geometry,'fixed_alpha_dir':str(original)}
        env_for_photo=env if label=='B' else lighting
        model='reference' if style=='materials' else 'camera'
        version={**params,'photo_model':model}
        directory,details=(original,old_meta) if label=='B' else (rendered,meta)
        value=await asyncio.to_thread(compose,scene,frozen,directory,details,env_for_photo,
                                      sample,version,job['seed'],folder/(label+'.png'))
        variants[label]={**value,'params':version,'render_dir':str(directory)}
    audit=fixed_audit(scene,old,geometry,old_meta,variants)
    order=list(variants);random.Random(signature(seed,row['job_id'])).shuffle(order)
    montage([(label,variants[key]['crop']) for label,key in zip('ABCD',order)],folder/'blind.jpg')
    montage([(label+'  '+title,variants[label]['crop']) for label,title in zip(
        'ABCD',('旧版','只改摄影','只改渲染','渲染与摄影'))],folder/'compare.jpg',large=True)
    data={'job_id':row['job_id'],'asset_id':row['asset_id'],'background':row['background'],
          'variants':variants,'audit':audit,'old_render':str(original),'new_render':str(rendered),
          'aov':aov_audit(rendered),'render_cache':meta['render_cache'],'render_timings':meta['render_timings'],
                'render_hit':hit,'board':str(folder/'compare.jpg'),'label_map':dict(zip('ABCD',order))}
    atomic_json(folder/'images.json',data)
    return await compare_case(data,folder,asset,vision)


async def compare_case(data,folder,asset,vision):
    old=data['variants']['A']
    result=await vision.ask('photo-ablation',PhotoComparison,PHOTO_PROMPT,
        {'name':asset['name'],'materials':asset['semantic']['material_classes']},
        [old['preview'],folder/'blind.jpg',asset['board']],image_limits=[(768,512),(1280,900),(512,512)],
        max_output_tokens=1400)
    data['comparison']={key:{**getattr(result,label).model_dump(),'pass':getattr(result,label).passes()}
                        for label,key in data['label_map'].items()}
    atomic_json(folder/'comparison.json',data)
    return data


def brief_usage(usage):
    return {k:v for k,v in usage.items() if k!='calls'}


def photo_report(out,rows,selected,usage,counts,timings,elapsed,state):
    reviewed=[r for r in rows if 'review' in r]
    by_asset=Counter(r['asset_id'] for r in reviewed if r['photo_pass'])
    totals={label:sum(r.get('comparison',{}).get(label,{}).get('pass',False) for r in rows) for label in 'ABCD'}
    n=sum(by_asset.values())
    gate=(len(reviewed)==12 and n>=10 and all(by_asset[k]>=1 for k in ASSETS)
          and all(r['audit']['pass'] and r['structure_ok'] and r['plastic_ok'] for r in reviewed))
    summary={'stage':12,'state':state,'planned':12,'completed_groups':len(rows),'reviewed':len(reviewed),
             'selected':selected,'appearance':PROFILES.get(selected),'comparison_photo_pass':totals,
             'photo_pass':n,'pass_by_asset':{k:by_asset[k] for k in ASSETS},'gate':gate,
             'requires_style_confirmation':True,'physics_calls':0,
             'render_counts':dict(counts),'render_seconds':dict(timings),
             'wall_seconds':round(elapsed,3),'usage':brief_usage(usage)}
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'results.json',rows)
    sections=[]
    for row in rows:
        board=Path(row['board']).relative_to(out).as_posix()
        links=[]
        for key,v in row['variants'].items():
            relative=os.path.relpath(v['image'],out).replace('\\','/')
            links.append(f'<a href="{html.escape(relative,quote=True)}">{key} 原尺寸 PNG</a>')
        evidence=row.get('review',{}).get('evidence','')
        mark='摄影达标' if row.get('photo_pass') else '待审'
        sections.append(f'<section><h2>{row["asset_id"]} · {row["job_id"]} · {mark}</h2>'
            f'<p>{html.escape(evidence)}</p><p>{" · ".join(links)}</p><img src="{board}" loading="lazy"></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>12 组固定摆位摄影对照</title>'
        '<style>body{background:#1b2229;color:#eee;font:16px sans-serif;margin:25px}a{color:#8cd7ff}'
        'section{border-top:1px solid #586873;margin-top:22px}img{width:min(100%,1280px)}'
        'h2{font-size:18px}</style><h1>固定摆位摄影对照</h1>'
        f'<p>{len(rows)}/12 组 · 推荐 {selected or "待评估"} · 摄影达标 {n}/{len(reviewed)} · '
        f'状态 {html.escape(state)}</p><p>固定几何与遮挡。新渲染同步更新阴影。</p>'+''.join(sections),encoding='utf8')
    return summary


async def photo12(baseline,out,selection,data,base,cache):
    started=time.monotonic()
    worker=Blender(out/('worker-'+str(os.getpid())),base.render_device)
    render_cache=RenderCache(cache,worker)
    vision=Vision(base,cache/'vision',out/f'usage-{time.time_ns()}.json')
    offline=Vision(base.model_copy(update={'offline':True}),cache/'vision',out/'offline-usage.json')
    scenes=Scenes(cache,offline)
    rows=[];selected=None;state='running'
    try:
        for aid,path in itertools.product(ASSETS,selection['fixed']):
            row,job=data[(path,aid)];folder=out/'cases'/row['job_id'];folder.mkdir(parents=True,exist_ok=True)
            asyncio.current_task().set_name(row['job_id'])
            asset=read(cache/'assets'/row['asset_signature']/'result.json')
            if file_hash(asset['mesh'])!=job['mesh_hash']:raise ValueError('baseline mesh changed: '+aid)
            if (folder/'comparison.json').is_file():
                record=read(folder/'comparison.json')
            elif (folder/'images.json').is_file():
                record=await compare_case(read(folder/'images.json'),folder,asset,vision)
            else:
                scene=await scenes.prepare(path)
                record=await fixed_case(folder,baseline,row,job,scene,asset,render_cache,cache,
                                        base.render_device,selection['seed'],vision)
            rows.append(record)
            print(json.dumps({'event':'comparison','done':len(rows),'asset':aid,'audit':record['audit']['pass'],
                              'photo_pass':{k:v['pass'] for k,v in record['comparison'].items()}},ensure_ascii=False),flush=True)
            photo_report(out,rows,selected,vision.usage(),render_cache.totals,render_cache.timings,time.monotonic()-started,state)
        # C isolates render changes while retaining the old post-process's duplicate dust.
        # Promote a profile with the new camera treatment across the entire cohort.
        def rank(label):
            scores=[r['comparison'][label] for r in rows]
            return (sum(s['pass'] for s in scores),-sum(s['serious'] for s in scores),
                    sum(s['lighting']+s['material']+s['texture'] for s in scores),label=='D')
        selected=max(('B','D'),key=rank)
        for row in rows:
            key=(row['background'],row['asset_id'])
            source,_=data[key];folder=out/'cases'/row['job_id']
            asyncio.current_task().set_name(row['job_id'])
            asset=read(cache/'assets'/source['asset_signature']/'result.json')
            value=row['variants'][selected]
            review=await inspect(vision,REVIEW_PROMPT,asset,value,value['params'],{},Path(value['render_dir']))
            serious=[d for d in review.defects if d.severity in ('major','fatal')]
            row.update(review=review.model_dump(),photo_pass=review.photo_passes(),
                       structure_ok=review.scores.structure>=3 and not any(
                           d.code=='structure' and d.visibility=='visible' for d in serious),
                       plastic_ok=row['asset_id']!='13' or (review.scores.material>=3 and not any(d.code=='material' for d in serious)))
            atomic_json(folder/'selected-review.json',row['review'])
            print(json.dumps({'event':'selected_review','job':row['job_id'],'asset':row['asset_id'],
                              'profile':selected,'photo_pass':row['photo_pass'],'scores':review.scores.model_dump()}),flush=True)
        state='style_review'
    except ProviderUnavailable as error:
        state='paused_provider'
        print(str(error),flush=True)
    finally:
        await worker.close();await offline.close();await vision.close()
        report=photo_report(out,rows,selected,vision.usage(),render_cache.totals,
                            render_cache.timings,time.monotonic()-started,state)
    return report


def recorded_calls(folder,role):
    calls=[]
    for path in folder.glob('usage-*.json'):
        calls.extend(c for c in read(path)['calls'] if c['role']==role and c.get('state')!='not_sent')
    return calls


async def audit_review_format(photo_dir,out,baseline_rows,base,cache):
    result_path=out/'review-format.json'
    if result_path.is_file():return read(result_path)
    full=recorded_calls(photo_dir,'review')
    vision=Vision(base,cache/'vision',out/f'usage-packed-{time.time_ns()}.json')
    checks=[]
    try:
        selected=read(photo_dir/'summary.json')['selected']
        for row in read(photo_dir/'results.json'):
            asyncio.current_task().set_name(row['job_id'])
            old,_=baseline_rows[(row['background'],row['asset_id'])]
            asset=read(cache/'assets'/old['asset_signature']/'result.json')
            value=row['variants'][selected]
            answer=await inspect(vision,REVIEW_PROMPT,asset,value,value['params'],{},
                                 Path(value['render_dir']),compact=True)
            original=Review.model_validate(row['review'])
            important={'structure','material','occlusion','floating','penetration'}
            def serious(review):
                return {d.code for d in review.defects
                        if d.severity in ('major','fatal') and d.visibility=='visible' and d.code in important}
            missed=serious(original)-serious(answer)
            checks.append({'job_id':row['job_id'],'missed_serious':sorted(missed),
                           'full_photo_pass':original.photo_passes(),'compact_photo_pass':answer.photo_passes(),
                           'full_visual_pass':original.passes(),'compact_visual_pass':answer.passes(),
                           'compact_review':answer.model_dump()})
    finally:await vision.close()
    compact=recorded_calls(out,'review-compact')
    from run import token_summary
    result={'cases':checks,'pass':len(checks)==12 and all(not r['missed_serious']
               and (r['full_visual_pass'] or not r['compact_visual_pass']) for r in checks),
            'full_usage':token_summary(full),'compact_usage':token_summary(compact)}
    atomic_json(result_path,result)
    return result


async def render_benchmark(photo_dir,baseline,out,data,config,cache):
    result_path=out/'render-benchmark.json'
    if result_path.is_file():return read(result_path)
    # Both arms start with the same cached asset, scene and resolved geometry.
    out.mkdir(parents=True,exist_ok=True)
    worker=Blender(out/'benchmark-worker',config.render_device)
    modern=RenderCache(out/'benchmark-cache',worker)
    vision=Vision(config.model_copy(update={'offline':True}),cache/'vision',out/'offline-usage.json')
    scenes=Scenes(cache,vision)
    style='scene' if config.appearance=='scene' else 'reference'
    records=[]
    try:
        for row in read(photo_dir/'results.json'):
            previous,_=data[(row['background'],row['asset_id'])]
            old,candidate=selected_geometry(baseline,previous)
            params=old['params']
            asset=read(cache/'assets'/previous['asset_signature']/'result.json')
            scene=await scenes.prepare(row['background'])
            variants=read(baseline/'jobs'/previous['job_id']/'candidates.json')['candidates']
            folder=out/'bench'/previous['job_id']
            folder.mkdir(parents=True,exist_ok=True)
            done=folder/'result.json'
            if done.is_file():
                records.append(read(done));continue
            legacy=[]
            for i,c in enumerate(variants):
                env,_=await asyncio.to_thread(sample_environment,scene,c)
                _,request=reference_path(cache,asset,c,params,env,config.render_device,512)
                legacy.append((request,folder/f'old-{i}'))
            env,_=await asyncio.to_thread(sample_environment,scene,candidate)
            _,request=reference_path(cache,asset,candidate,params,env,config.render_device)
            legacy.append((request,folder/'old-final'))
            previous_seconds=worker.operation_seconds.get('render',0.)
            for request,directory in legacy:
                await worker.ask('render',{**request,'output':str(directory)},timeout=240)
            old_seconds=worker.operation_seconds['render']-previous_seconds
            env,_=await asyncio.to_thread(sample_environment,scene,candidate,scene_lighting=style=='scene')
            before=worker.operation_seconds.get('render_parts',0.)
            _,meta,hit=await modern.get(asset,candidate,params,env,scene['size'],
                                         final_resolution(candidate),appearance=style)
            new_seconds=worker.operation_seconds.get('render_parts',0.)-before
            start=time.monotonic()
            _,_,warm=await modern.get(asset,candidate,params,env,scene['size'],
                                       final_resolution(candidate),appearance=style)
            record={'job_id':previous['job_id'],'reference_gpu_seconds':old_seconds,
                    'optimized_gpu_seconds':new_seconds,'optimized_warm_seconds':time.monotonic()-start,
                    'reference_passes':3*len(legacy),'optimized_passes':sum(not p['hit'] for p in meta['render_cache'].values()),
                    'cold_entry_reused':hit,'warm_cache_verified':warm}
            atomic_json(done,record);records.append(record)
            print(json.dumps({'event':'render_benchmark',**record}),flush=True)
    finally:await worker.close();await vision.close()
    old=sum(r['reference_gpu_seconds'] for r in records);new=sum(r['optimized_gpu_seconds'] for r in records)
    result={'kind':'same_geometry_reference_and_lazy_render','cases':records,
            'reference_gpu_seconds':old,'optimized_gpu_seconds':new,'fraction_saved':1-new/old,
            'comparable_cold':not any(r['cold_entry_reused'] for r in records)}
    atomic_json(result_path,result)
    return result


async def production_stage(stage,root,photo_dir,baseline,selection,data,base,cache,confirmed):
    approval=read(photo_dir/'summary.json')
    if not approval['gate']:
        raise ValueError('阶段 A 尚未达到摄影、结构和像素门槛')
    if not confirmed:
        raise ValueError('先查看 12 组对照并确认整体摄影风格')
    frozen=read(photo_dir/'manifest.json')
    current=implementation_manifest()
    checked={'blender_display_worker.py','blender_worker.py','photography.py','compose.py','geometry.py',
             'framing.py','materials.py','rendering.py','schema.py'}
    frozen_sources={k:v for k,v in frozen['implementation']['sources'].items() if Path(k).name in checked}
    active_sources={k:v for k,v in current['sources'].items() if Path(k).name in checked}
    if (signature(frozen['selection'])!=signature(selection)
        or frozen['baseline_results_hash']!=file_hash(baseline/'results.jsonl')
        or frozen_sources!=active_sources
        or frozen['implementation']['ocio_config_hash']!=current['ocio_config_hash']):
        raise ValueError('摄影确认之后素材或渲染实现已变化，需要重新验证')
    pilot=root/'pilot36'
    out=pilot if stage==36 else root/'regression138'
    out.mkdir(parents=True,exist_ok=True)
    paths=selection['pilot'] if stage==36 else selection['pilot']+[
        p for p in selection['all'] if p not in selection['pilot']]
    manifest=[data[(p,k)][1] for p,k in itertools.product(paths,ASSETS)]
    atomic_json(out/'validation-inputs.json',manifest)
    if stage==36:
        packed=await audit_review_format(photo_dir,out,data,base,cache)
    else:
        gate=read(pilot/'validation.json')
        if not gate['gate']:raise ValueError('阶段 B 未达到扩批门槛')
        packed=read(pilot/'review-format.json')
    config=base.model_copy(update={'output':str(out),'manifest':str(out/'validation-inputs.json'),
        'count':len(manifest),'appearance':approval['appearance'],'review_format':'compact' if packed['pass'] else 'full'})
    summary=await Pipeline(config).run(reuse_from=pilot if stage==138 else None)
    if summary['run']['state']!='complete':return summary
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text(encoding='utf8').splitlines()]
    passes=Counter(r['asset_id'] for r in rows if r.get('photo_status')=='pass')
    from audit_display_outputs import main as audit_pngs
    audit_pngs([out],out/'png-audit.json')
    pngs=read(out/'png-audit.json')
    validation={'stage':stage,'cases':len(rows),'photo_pass':sum(passes.values()),
                'pass_by_asset':{k:passes[k] for k in ASSETS},'review_format':config.review_format,
                'images':summary['generated_pngs'],'pngs_verified':pngs['passed'],'gate':False}
    if stage==36:
        quality=(sum(passes.values())>=29 and all(passes[k]>=4 for k in ASSETS)
                 and len(rows)==36 and pngs['passed']==pngs['pngs'])
        if quality:
            benchmark=await render_benchmark(photo_dir,baseline,out,data,config,cache)
            validation['render_fraction_saved']=benchmark['fraction_saved']
            validation['gate']=benchmark['comparable_cold'] and benchmark['fraction_saved']>=.4
        validation['quality_gate']=quality
    else:
        validation['photo_target_met']=sum(passes.values())>=111
        validation['image_target_met']=summary['generated_pngs']==138
        validation['reused_images']=summary['reused_pngs']
    atomic_json(out/'validation.json',validation)
    return validation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--stage',type=int,choices=[12,36,138],required=True)
    parser.add_argument('--attempt',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--seed',type=int,default=20261110)
    parser.add_argument('--style-confirmed',action='store_true')
    args=parser.parse_args(argv)
    baseline=args.baseline.resolve();root=args.out.resolve()
    data=baseline_rows(baseline)
    base=BatchConfig.model_validate(read(baseline/'config.json'))
    cache=resolve(base.cache)
    selection=choose_backgrounds(data,cache,args.seed)
    out=root/f'photo12-{args.attempt}'
    manifest={'baseline':str(baseline),'selection':selection,'implementation':implementation_manifest(),
              'baseline_results_hash':file_hash(baseline/'results.jsonl')}
    if args.stage==12:
        with batch_lock(out):
            lock=out/'manifest.json'
            if lock.is_file() and signature(read(lock))!=signature(manifest):
                raise ValueError('fixed comparison already belongs to different code/inputs; use a new attempt')
            atomic_json(lock,manifest)
            result=asyncio.run(photo12(baseline,out,selection,data,base,cache))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    try:
        result=asyncio.run(production_stage(args.stage,root,out,baseline,selection,data,base,cache,args.style_confirmed))
    except (OSError,ValueError) as error:
        parser.exit(2,str(error)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
