"""Archive-based seam audit and deterministic fixed-placement local photography pilot."""
import argparse
import asyncio
from collections import Counter
import html
import json
import os
from pathlib import Path
import random
import time
from typing import Literal

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from display_batch.ai import Vision, ProviderUnavailable
from display_batch.common import atomic_json, resolve, signature, implementation_manifest, file_hash
from display_batch.compose import compose, sample_environment, warp
from display_batch.framing import edit_bounds
from display_batch.local import seam_measurement
from display_batch.rendering import reference_path
from display_batch.review import FUSION_PROMPT, evidence_status
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig, Strict, FusionObservation, FusionCue
from display_batch.pipeline import Pipeline
from display_batch.framing import final_resolution
from validate_display_costs import selected_geometry, montage


class PairImage(FusionObservation):
    defects: list[FusionCue] = Field(max_length=3)


class PairReview(Strict):
    A: PairImage
    B: PairImage
    preferred: Literal['A','B','equal']
    evidence: str = Field(max_length=200)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def records(path):return [json.loads(l) for l in (path/'results.jsonl').read_text(encoding='utf8').splitlines()]


async def cached_case(baseline,row,cache,scenes):
    old,candidate=selected_geometry(baseline,row)
    asset=read(cache/'assets'/row['asset_signature']/'result.json')
    scene=await scenes.prepare(row['background'])
    env,_=sample_environment(scene,candidate)
    env.pop('sample_basis',None)  # The supplied archive predates this metadata-only field.
    directory,_=reference_path(cache,asset,candidate,old['params'],env)
    if not (directory/'result.json').is_file():
        raise ValueError('archive_render_missing: '+str(directory))
    return old,dict(candidate),asset,scene,directory,read(directory/'result.json')


def receiver_diagnostic(scene,candidate,directory,meta):
    bounds=edit_bounds(candidate,scene['size']);x,y,r,b=bounds
    alpha=np.float32(Image.open(directory/'object.png').getchannel('A'))/255
    alpha=warp(alpha,candidate['full_px'],candidate['offset'],bounds)
    before=cv2.imread(str(directory/'with-shadow-linear0001.exr'),cv2.IMREAD_UNCHANGED)[:,:,:3].mean(axis=2)
    base=cv2.imread(str(directory/'without-shadow-linear0001.exr'),cv2.IMREAD_UNCHANGED)[:,:,:3].mean(axis=2)
    loss=np.clip(1-before/np.maximum(base,.015),0,.65);loss[base<.015]=0
    loss=warp(loss,candidate['full_px'],candidate['offset'],bounds)
    receiver=warp(np.float32(base>=.015),candidate['full_px'],candidate['offset'],bounds)>.6
    support=np.logical_or.reduce([s['mask'][y:b,x:r] for s in scene['supports'] if s['id'] in candidate['support_ids']])
    receiver&=support&~scene['protected'][y:b,x:r]
    return seam_measurement(loss,receiver,alpha)


def report(out,rows,summary):
    atomic_json(out/'summary.json',summary)
    with (out/'results.jsonl').open('w',encoding='utf8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    sections=[]
    for row in rows:
        picture=Path(row['comparison']).relative_to(out).as_posix()
        png=Path(row['image']).relative_to(out).as_posix()
        samples=Path(row['image']).with_suffix('.samples.png').relative_to(out).as_posix()
        sections.append(f'<section><h2>{row["asset_id"]} / {row["job_id"]} / {row["comparison_preference"]}</h2>'
            f'<p>{html.escape(row["comparison_evidence"])}</p><a href="{png}">原尺寸 PNG</a> · '
            f'<a href="{samples}">局部采样依据</a><img src="{picture}" loading="lazy"></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>局部融合固定摆位对照</title>'
        '<style>body{background:#172128;color:white;font:16px sans-serif;margin:24px}img{max-width:100%}a{color:#9dcfff}</style>'
        f'<h1>固定摆位局部融合 · {len(rows)} 项</h1><p>统一策略，不逐图更换风格。归档基线在左，新版在右。</p>'
        '<a href="summary.json">质量、耗时和用量记录</a>'+''.join(sections),encoding='utf8')


async def main(args):
    out=resolve(args.out);out.mkdir(parents=True,exist_ok=True)
    baseline=resolve(args.baseline);rows=records(baseline)
    config=BatchConfig.model_validate(read(baseline/'config.json')).model_copy(update={
        'output':str(out),'offline':args.offline or args.audit,'appearance':'local','max_iterations':0})
    cache=resolve(config.cache);start=time.monotonic()
    vision=Vision(config,cache/'vision',out/'usage.json');scenes=Scenes(cache,vision)
    source=signature(implementation_manifest(),file_hash(__file__),args.rerender)
    renderer=Pipeline(config) if args.rerender else None
    atomic_json(out/'implementation.json',implementation_manifest())
    try:
        if args.audit:
            audit=[]
            for row in rows:
                _,candidate,_,scene,folder,meta=await cached_case(baseline,row,cache,scenes)
                stats=await asyncio.to_thread(receiver_diagnostic,scene,candidate,folder,meta)
                audit.append({'job':row['job_id'],'asset':row['asset_id'],'background':row['background'],**stats})
            groups={p:np.mean([r['loss_p95'] for r in audit if r['background']==p]) for p in {r['background'] for r in audit}}
            ordered=sorted(groups,key=lambda p:(groups[p],file_hash(p)))
            chosen=[ordered[-1],ordered[len(ordered)//2]]
            selection=[{'job_id':r['job_id'],'seed':item['seed']} for r,item in zip(rows,read(baseline/'inputs.json')) if r['background'] in chosen]
            atomic_json(out/'audit.json',{'records':audit,'seconds':time.monotonic()-start,'usage':vision.usage()})
            atomic_json(out/'selection.json',{'basis':'highest_and_median_receiver_boundary_loss_before_new_images',
                'backgrounds':chosen,'manifest':selection,'baseline':str(baseline)})
            print(json.dumps({'audited':len(audit),'boundary_loss_over_0_04':sum(r['loss_p95']>.04 for r in audit),'selection':selection},ensure_ascii=False));return
        selection=read(resolve(args.selection))
        planned=selection['manifest']
        prior=out/'selection.json'
        if prior.is_file() and read(prior)!=selection:raise ValueError('fixed_selection_changed')
        atomic_json(prior,selection);done=[]
        for item in planned:
            row=next(r for r in rows if r['job_id']==item['job_id'])
            folder=out/'jobs'/row['job_id'];folder.mkdir(parents=True,exist_ok=True)
            finished=folder/'result.json'
            if finished.is_file():
                value=read(finished)
                if value['source_signature']!=source:raise ValueError('implementation_changed_use_new_output')
                done.append(value);continue
            old,candidate,asset,scene,render,meta=await cached_case(baseline,row,cache,scenes)
            candidate['material_classes']=asset['semantic']['material_classes']
            environment,samples=await asyncio.to_thread(sample_environment,scene,candidate,local=True)
            params={**old['params'],'photo_model':'local','atmosphere':'auto'}
            source_render=str(render)
            if renderer:
                candidate['fixed_alpha_dir']=str(render)
                candidate['fixed_alpha_window']=None
                render,meta,environment,samples,_=await renderer.render(asset,scene,candidate,params,final_resolution(candidate))
            comp=await asyncio.to_thread(compose,scene,candidate,render,meta,environment,samples,params,item['seed'],folder/'local.png')
            l,t,r,b=environment['local_context']['bounds']
            old_context=folder/'archive.context.jpg'
            context=Image.open(row['image']).crop((l,t,r,b));context.thumbnail((1200,900));context.save(old_context,quality=94)
            new_context=Path(comp['image']).with_suffix('.context.jpg')
            order=['archive','local'];random.Random(row['job_id']).shuffle(order)
            inputs={'archive':old_context,'local':new_context}
            pair_prompt=FUSION_PROMPT.replace(
                'The first image is the delivered panorama and the second is its crop. Subsequent images are\nreferences for identity and structure. Evaluate seams on the delivered photograph only.',
                'Images 1 and 2 are alternative delivered neighbourhood crops A and B. Image 3 is material identity. Evaluate seams on crops A and B.')
            error=None
            try:
                comparison=await vision.ask('local-pair-recognition',PairReview,pair_prompt+
                    ' Compare A and B: images 1 and 2 are equally scaled neighbourhood crops of the same placement. '
                    'Image 3 is material identity. Faded paint and dust are intentional; do not demand fresh saturated reference colour. '
                    'Judge photography from nearby surfaces, never unrelated distant lighting. '
                    'Compare rectangular patch seams separately from object edges. Keep at most three defects per image. '
                    'Both may still fail; do not infer which is newer. Return preferred A/B/equal and concise evidence.',
                    {'name':asset['name'],'material_classes':asset['semantic']['material_classes']},
                    [inputs[order[0]],inputs[order[1]],asset['board']],plain_json=True)
            except ProviderUnavailable:raise
            except RuntimeError as failure:
                comparison=None;error=str(failure)
            judgments={kind:getattr(comparison,label).review() if comparison else None for label,kind in zip(('A','B'),order)}
            review=judgments['local'];evidence=evidence_status(comp['checks'])
            geometry={'alpha_equal':old['alpha_hash']==comp['alpha_hash'],
                'visible_alpha_equal':old['visible_alpha_hash']==comp['visible_alpha_hash'],
                'anchor_ok':comp['checks']['anchor']['status']=='pass'}
            assert all(geometry.values()),geometry
            picture=folder/'comparison.jpg'
            montage([('压缩包基线',old_context),('局部融合新版',new_context),
                     ('基线全景',row['image']),('新版全景',comp['image'])],picture,large=True)
            record={**comp,'job_id':row['job_id'],'asset_id':row['asset_id'],'background':row['background'],
                'scene_signature':row['scene_signature'],'status':'failed' if evidence=='fail' else 'accepted_display' if evidence=='pass' and review and review.passes() else 'review',
                'source_signature':source,'geometry':geometry,'source_render':source_render,'render':str(render),
                'render_cache':meta.get('render_cache',{}),'render_timings':meta.get('render_timings',{}),
                'params':params,'scores':review.scores.model_dump() if review else None,'review':review.model_dump() if review else None,
                'archive_review':judgments['archive'].model_dump() if comparison else None,'comparison':str(picture),
                'comparison_preference':('equal' if comparison.preferred=='equal' else order['AB'.index(comparison.preferred)]) if comparison else 'unknown',
                'comparison_evidence':comparison.evidence if comparison else error,'review_error':error,'image_hash':file_hash(comp['image']),
                'photo_status':('pass' if review.photo_passes() else 'fail') if review else 'unknown',
                'visual_status':('pass' if review.passes() else 'fail') if review else 'unknown',
                'evidence_status':evidence,'insertion_trace':review.trace if review else 'uncertain','formal_status':'unchanged'}
            atomic_json(finished,record);done.append(record)
            summary={'generated':len(done),'planned':len(planned),'photo_pass':sum(r['photo_status']=='pass' for r in done),
                'visual_pass':sum(r['visual_status']=='pass' for r in done),'preferences':dict(Counter(r['comparison_preference'] for r in done)),
                'counts':dict(Counter(r['status'] for r in done)),'seconds':time.monotonic()-start,
                'new_physics_calls':0,'render':dict(renderer.renders.totals) if renderer else {'object_renders':0},
                'usage':{k:v for k,v in vision.usage().items() if k!='calls'}}
            report(out,done,summary)
            print(json.dumps({'job':row['job_id'],'preference':record['comparison_preference'],'photo':record['photo_status']},ensure_ascii=False),flush=True)
    finally:
        await vision.close()
        if renderer:
            await renderer.blender.close()
            await renderer.vision.close()
            renderer.store.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='outputs/摄影嵌入-验证-v2-20261110/regression')
    parser.add_argument('--out',required=True)
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--selection')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--rerender',action='store_true',help='Use the production local renderer with frozen archive coverage')
    asyncio.run(main(parser.parse_args()))
