"""Frozen full-frame staged diagnostics and 5 x 3 local-adaptation delivery audit."""
import argparse
import asyncio
import copy
import html
import json
from pathlib import Path
import random
import shutil
import time
from typing import Literal
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from display_batch.common import atomic_json, file_hash, resolve, load_catalog, signature, implementation_manifest
from display_batch.schema import BatchConfig, EffectSettings, Strict, Scores
from display_batch.pipeline import Pipeline
from display_batch.compose import compose, sample_environment
from display_batch.effects import lighting_for, dust_profile
from display_batch.framing import render_windows, final_resolution
from display_batch.rendering import RENDER_VERSION
from display_batch.local_atmosphere import prepare as prepare_local
from display_batch.costs import cumulative, label as cost_label

VERSION='adaptation-diagnostic-v1'
STAGES=('归档','仅修阴影','再改局部雾化','再改中度做旧','再改局部受光')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


class VariantScore(Strict):
    label: Literal['V1','V2','V3','V4','V5']
    scores: Scores
    trace: Literal['none','uncertain','visible']
    identity_preserved: bool
    severe_defects: list[str] = Field(max_length=5)
    evidence: str = Field(max_length=180)


class Comparison(Strict):
    variants: list[VariantScore] = Field(min_length=5,max_length=5)
    evidence: str = Field(max_length=200)


def cached_render(cache,iteration,candidate,size):
    entries={k:read(cache/RENDER_VERSION/k/v['signature']/'result.json') for k,v in iteration['render_cache'].items()}
    obj=entries['object'];windows=render_windows(candidate,size,iteration['render_resolution'])
    meta={k:obj[k] for k in ('camera','device','anchor_uv','contact_polygons')}
    meta.update(object_window=windows['object'],shadow_window=windows['shadow'],legacy_grid=False,
                part_paths={k:v['path'] for k,v in entries.items()},matte_materials=obj.get('matte_materials',[]),
                matte_profile=dust_profile(float(iteration['params']['surface_dust'])))
    return Path(obj['path']),meta


def page(out,cases,summary):
    body=['<!doctype html><meta charset="utf-8"><title>尺度、原貌与局部薄雾测试</title>',
          '<style>body{background:#172027;color:#e8eeee;font:16px system-ui;margin:24px}a{color:#9bdcc6}img{width:100%;height:auto;object-fit:contain}section{margin:24px 0}article{margin:18px 0}p{line-height:1.7}</style>',
          '<h1>尺度、原貌与局部薄雾测试</h1><p>全部展示完整监控画面；点击图片查看原尺寸。视觉与物理证据分别记录。</p>']
    if 'jobs' in summary:
        body.append(f"<p>出图 {summary['images']}/{summary['jobs']}；材质、受光与摄影三项达标 {summary['three_score_pass']}/{summary['jobs']}；"
                    f"展示合格 {summary['states']['accepted_display']}，待审 {summary['states']['review']}。"
                    +('已达到本轮门槛。' if summary['target_met'] else '未达到本轮门槛，普通批量默认不切换。')+'</p>')
    if summary.get('cost'):body.append('<p>'+html.escape(cost_label(summary['cost']))+'</p>')
    body.append('<details><summary>完整验证与成本数据</summary><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details>')
    last_group=None
    for case in cases:
        if case.get('group') and case['group']!=last_group:
            last_group=case['group'];body.append('<h2>场景 '+html.escape(last_group)+'</h2>')
        body.append('<section><h2>'+html.escape(case['title'])+'</h2>')
        if case.get('description'):body.append('<p>'+html.escape(case['description'])+'</p>')
        for index,v in enumerate(case['variants']):
            path=Path(v['image']);rel=Path(__import__('os').path.relpath(path,out)).as_posix()
            caption=v['name']
            if case.get('blind_order') is not None:
                label='V'+str(case['blind_order'].index(index)+1)
                score=next(s for s in case['review']['variants'] if s['label']==label)
                s=score['scores'];caption+=f" · 材质 {s['material']} / 受光 {s['lighting']} / 摄影 {s['texture']} · 痕迹 {score['trace']} · "+score['evidence']
            body.append('<article><p>'+html.escape(caption)+'</p><a href="'+html.escape(rel,quote=True)+'"><img loading="lazy" src="'+html.escape(rel,quote=True)+'"></a></article>')
        body.append('</section>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf8')


async def diagnostic(source,out,limit=None):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    cfg=BatchConfig.model_validate({**read(source/'config.json'),'output':str(out),'max_iterations':0})
    pipe=Pipeline(cfg)
    results=[json.loads(s) for s in (source/'results.jsonl').read_text(encoding='utf8').splitlines() if s]
    groups=list(dict.fromkeys(r['group'] for r in read(source/'inputs.json')))
    rows=[r for r in results if r['group']==groups[2]]
    if len(rows)!=5:raise ValueError('third_scene_must_have_five_frozen_assets')
    if limit:rows=rows[:limit]
    inputs=read(source/'inputs.json');cases=[]
    assets={k:file_hash(v['mesh']) for k,v in load_catalog().items()}
    manifest={'version':VERSION,'source':str(source),'jobs':[r['job_id'] for r in rows],
              'stages':STAGES,'source_models':assets,'implementation':implementation_manifest()}
    manifest_key=signature(manifest)
    if (out/'manifest.json').exists() and read(out/'manifest.json')['signature']!=manifest_key:
        raise ValueError('diagnostic_inputs_changed_use_new_output')
    atomic_json(out/'manifest.json',{'signature':manifest_key,**manifest})
    atomic_json(out/'config.json',cfg.model_dump())
    try:
        for row in rows:
            folder=out/row['job_id'];folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():cases.append(read(folder/'result.json'));continue
            job=source/'jobs'/row['job_id'];old=read(job/f"iteration-{row['best_iteration']}.json")
            pose=copy.deepcopy(next(c for c in read(job/'candidates.json')['candidates'] if c['id']==old['params']['candidate']))
            asset=read(pipe.cache/'assets'/row['asset_signature']/'result.json')
            scene=await pipe.scenes.prepare(row['background'])
            seed=next(r['seed'] for r in inputs if r['background']==row['background'] and r['asset_id']==row['asset_id'])
            params=copy.deepcopy(old['params']);source_render,old_meta=cached_render(pipe.cache,old,pose,scene['size'])
            env,samples=sample_environment(scene,pose,local=True)
            env=lighting_for(scene,pose,env,params['effects_lighting'])
            env['calibration']=old['photo']['calibration']
            replay=compose(scene,pose,source_render,old_meta,env,samples,params,seed,folder/'0-archive.png')
            if file_hash(replay['image'])!=file_hash(old['image']):raise ValueError('archived_baseline_did_not_replay')
            variants=[{'name':STAGES[0],'image':replay['image'],'compose':replay}]
            fixed_pose={**pose,'fixed_alpha_dir':str(source_render),'fixed_alpha_window':old_meta['object_window']}
            params.update(shadow_mode='matched_receiver')
            d,meta,_,_,_=await pipe.render(asset,scene,pose,params,old['render_resolution'])
            # Stage 1 changes shadows only; freeze old object RGB and geometric coverage.
            shadow_meta={**old_meta,'shadow_window':meta['shadow_window'],'part_paths':meta['part_paths'],
                         'shadow_pair_validation':meta.get('shadow_pair_validation')}
            current=compose(scene,pose,source_render,shadow_meta,env,samples,params,seed,folder/'1-shadow.png')
            variants.append({'name':STAGES[1],'image':current['image'],'compose':current})
            print(json.dumps({'job':row['job_id'],'stage':'shadow','checks':{k:v['status'] for k,v in current['checks'].items()}},ensure_ascii=False),flush=True)
            scene['atmosphere']=await pipe.atmospheres.prepare(scene)
            env['atmospheric_context']=prepare_local(scene,pose,env,pipe.cache)
            params.update(fog_mode='local_auto',fog_gain='1')
            current=compose(scene,pose,source_render,shadow_meta,env,samples,params,seed,folder/'2-local-fog.png')
            variants.append({'name':STAGES[2],'image':current['image'],'compose':current})
            params.update(weathering='preserve_source',surface_dust='0.5')
            d,meta,_,_,_=await pipe.render(asset,scene,pose,params,old['render_resolution'])
            current=compose(scene,fixed_pose,d,meta,env,samples,params,seed,folder/'3-weathering.png')
            variants.append({'name':STAGES[3],'image':current['image'],'compose':current})
            params['effects_lighting']='local'
            d,meta,env2,samples2,_=await pipe.render(asset,scene,pose,params,old['render_resolution'])
            current=compose(scene,fixed_pose,d,meta,env2,samples2,params,seed,folder/'4-local-light.png')
            variants.append({'name':STAGES[4],'image':current['image'],'compose':current})
            for v in variants[1:]:
                for key in ('alpha_hash','visible_alpha_hash','foreground_mask_hash'):
                    if v['compose'][key]!=replay[key]:raise ValueError('diagnostic_geometry_changed_'+key)
                for key in ('anchor','protected','outside_edit','holes_alpha','size'):
                    if v['compose']['checks'][key]['status']!='pass':raise ValueError('diagnostic_failed_'+key)
            order=list(range(5));random.Random(seed).shuffle(order)
            images=[]
            for i in order:
                images.extend([variants[i]['compose']['preview'],Path(variants[i]['image']).with_suffix('.context.jpg')])
            images.append(asset['board'])
            answer=await pipe.vision.ask('adaptation-blind-comparison',Comparison,
                'Compare five anonymized composites of the same item, frozen pose/size/contact. Each V1..V5 has a panorama then local context; last image is model identity. '
                'Give all five variants seven 0..4 scores (3 usable), insertion trace, preservation of original paint/markings/texture, concrete severe defects. '
                'Moderate dirt is intended, but uniformly grey coatings, wrong haze colour and rectangular shadow patches are defects. '
                'Judge the photographed nearby same-depth surroundings. Do not infer success from order. Respect occlusion and tiny hidden features.',
                {'name':asset['name'],'variants':['V1','V2','V3','V4','V5']},images,
                image_limits=[size for _ in range(5) for size in [(1280,720),(800,700)]]+[(512,512)],
                max_output_tokens=3072,plain_json=True)
            if {v.label for v in answer.variants}!={'V1','V2','V3','V4','V5'}:raise ValueError('incomplete_blind_labels')
            case={'title':row['asset_id']+' · '+row['group'],'job':row['job_id'],'variants':variants,
                  'blind_order':order,'review':answer.model_dump(),'atmosphere':env2.get('atmospheric_context'),
                  'status':'review','fixed_geometry':True}
            atomic_json(folder/'result.json',case);cases.append(case)
            page(out,cases,{'completed':len(cases),'target':len(rows),'cost':cumulative(out,pipe.vision.usage())})
            print(json.dumps({'job':row['job_id'],'stage':'reviewed','scores':[(v.label,v.scores.material,v.scores.lighting,v.trace) for v in answer.variants]},ensure_ascii=False),flush=True)
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()
        atomic_json(out/f'run-{pipe.run_id}.json',{'seconds':time.monotonic()-started,'renders':dict(pipe.renders.totals),
                    'render_seconds':dict(pipe.renders.timings),'completed':len(cases)})
    if assets!={k:file_hash(v['mesh']) for k,v in load_catalog().items()}:raise ValueError('source_models_changed')
    report={'completed':len(cases),'target':len(rows),'source_models_unchanged':len(assets),
            'cost':cumulative(out),'seconds':sum(read(p)['seconds'] for p in out.glob('run-*.json'))}
    atomic_json(out/'summary.json',report);page(out,cases,report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def delivery(source,batch,out,diagnostics,previous_batches=()):
    out.mkdir(parents=True,exist_ok=True);rows=[json.loads(x) for x in (batch/'results.jsonl').read_text(encoding='utf8').splitlines() if x]
    old=[json.loads(x) for x in (source/'results.jsonl').read_text(encoding='utf8').splitlines() if x]
    inputs=read(batch/'inputs.json');expected={(r['asset_id'],r['background']) for r in read(source/'inputs.json')}
    if {(r['asset_id'],r['background']) for r in inputs}!=expected or len(rows)!=15:raise ValueError('cartesian_product_mismatch')
    cases=[];checks=[];score_pass={};defect_counts={};physical={}
    for r in rows:
        baseline=next(x for x in old if (x['asset_id'],x['background'])==(r['asset_id'],r['background']))
        if r.get('image'):
            with Image.open(r['image']) as im:
                if im.size!=(3840,2160):raise ValueError('wrong_output_dimensions')
            for key in ('anchor','protected','outside_edit','holes_alpha','size','projection_size'):
                if r['checks'].get(key,{}).get('status')!='pass':raise ValueError('delivery_check_failed_'+key)
            final=read(batch/'jobs'/r['job_id']/f"iteration-{r['best_iteration']}.json")
            if file_hash(r['image'])!=final['image_hash']:raise ValueError('best_image_changed')
            defects=final['review']['defects']
            for d in defects:
                if d['severity'] in ('major','fatal'):defect_counts[d['code']]=defect_counts.get(d['code'],0)+1
            s=r['scores'];scores_ok=all(s[k]>=3 for k in ('material','lighting','texture'))
            score_pass[r['group']]=score_pass.get(r['group'],0)+int(scores_ok)
            evidence=r['checks']['scale']['evidence'];profile=evidence['physical_size']
            if r['asset_id'] in physical and physical[r['asset_id']]!=profile:raise ValueError('cross_scene_physical_length_changed')
            physical[r['asset_id']]=profile
            candidate=next(c for c in read(batch/'jobs'/r['job_id']/'candidates.json')['candidates'] if c['id']==final['params']['candidate'])
            env=r['photo']['effects'];desc=(f"材质 {s['material']} / 受光 {s['lighting']} / 摄影质感 {s['texture']}；"
                f"插入痕迹 {r['insertion_trace']}；固定估计长度 {profile['nominal_mm']:.2f} mm；"
                f"投影完整长轴 {r['checks']['scale']['measurement_px']:.1f} px；现场尺度 {r['checks']['scale']['status']}；"
                f"总雾量中位数 {env['fog_total_median']:.1%}；浮灰 {final['params']['surface_dust']}。"
                ' 主要缺陷：'+'；'.join(d['evidence'] for d in defects if d['severity'] in ('major','fatal')))
            variants=[{'name':'旧版','image':baseline['image']},{'name':'新版 · '+r['status'],'image':r['image']}]
            # Diagnostic overlays retain the whole original frame, never cropped panels.
            with Image.open(r['background']) as im:frame=im.convert('RGB')
            draw=ImageDraw.Draw(frame);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',29)
            for ref in evidence['references']:
                pts=np.asarray(ref['reference'])*np.subtract(frame.size,1)
                draw.line([tuple(p) for p in pts],fill='#00ffb7',width=5)
                draw.text(tuple(pts[0]),ref['id']+' '+str(ref['reference_mm'])+' mm',font=font,fill='#00ffb7',stroke_width=2,stroke_fill='black')
            ax,ay=candidate['anchor_px'];draw.ellipse((ax-12,ay-12,ax+12,ay+12),outline='red',width=5)
            draw.rectangle(candidate['bounds'],outline='yellow',width=4)
            draw.text((35,105),f"估计长度 {profile['nominal_mm']:.2f} mm | 仅锚点与参照示意，非尺度确认",font=font,fill='yellow',stroke_width=2,stroke_fill='black')
            overlay=out/(r['job_id']+'-scale.png');frame.save(overlay)
            variants.append({'name':'尺度依据 · 完整画面诊断','image':str(overlay)})
            air=r['photo']['atmosphere'];fog=resolve(read(batch/'config.json')['cache'])/'local-atmosphere'/air['signature']/'full-veil.png'
            if fog.is_file():variants.append({'name':'局部雾场 · 彩色区域为估计雾量，非合成效果','image':str(fog)})
            cases.append({'title':r['asset_id'],'group':r['group'],'description':desc,'variants':variants})
        checks.append({k:r.get(k) for k in ('job_id','asset_id','group','status','reason','image','scores','photo_status',
                       'insertion_trace','measured_major_px','checks','photo','new_api_calls')})
    groups=list(dict.fromkeys(r['group'] for r in inputs))
    passed={g:sum(r['photo_status']=='pass' for r in rows if r['group']==g) for g in groups}
    source_hashes=read(diagnostics/'manifest.json')['source_models']
    unchanged=source_hashes=={k:file_hash(v['mesh']) for k,v in load_catalog().items()}
    if not unchanged:raise ValueError('source_hash_mismatch')
    costs=[cumulative(diagnostics),cumulative(batch),*[cumulative(p) for p in previous_batches]]
    complete=all(c['complete'] for c in costs)
    cost={'actual_requests':sum(c['actual_requests'] for c in costs),'reported_tokens':sum(c['reported_tokens'] for c in costs),
          'estimated_api_min':sum(c['estimated_api_min'] or 0 for c in costs),
          'estimated_api_max':sum(c['estimated_api_max'] for c in costs) if complete else None,
          'complete':complete,'currency':'CNY','local_compute_cost':None,
          'unpriced_or_unreported_calls':sum(c['unpriced_or_unreported_calls'] for c in costs)}
    runs=[{'output':str(p),**read(run)} for p in [*previous_batches,batch] for run in p.glob('run-*.json')]
    diagnostic_runs=[read(p) for p in diagnostics.glob('run-*.json')]
    report={'jobs':15,'images':len(cases),'photo_pass_by_scene':passed,'photo_pass':sum(passed.values()),
            'visual_pass':sum(r['visual_status']=='pass' for r in rows),
            'states':{s:sum(r['status']==s for r in rows) for s in ('accepted_display','review','failed')},
            'three_score_pass_by_scene':score_pass,'three_score_pass':sum(score_pass.values()),
            'target_met':sum(score_pass.values())>=12 and min(score_pass.values())>=4 and
                         not any(defect_counts.get(k,0) for k in ('identity_loss','patch_seam','haze_color')),
            'major_or_fatal_defects':defect_counts,'fixed_physical_profiles':physical,
            'source_models_unchanged':len(source_hashes),'cost':cost,'diagnostics':str(diagnostics/'index.html'),
            'attempts':[str(p) for p in previous_batches],
            'generation_seconds':sum(r.get('seconds',0) for r in runs+diagnostic_runs),
            'runs':runs,'diagnostic_runs':diagnostic_runs,
            'batch':str(batch),'baseline':str(source),'note':'完整画面对照；未知物理证据继续待审，未进入生产扩批'}
    atomic_json(out/'validation.json',{'summary':report,'jobs':checks});page(out,cases,report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['diagnostic','delivery'])
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch');p.add_argument('--diagnostics');p.add_argument('--limit',type=int)
    p.add_argument('--previous-batch',action='append',default=[])
    a=p.parse_args()
    if a.mode=='diagnostic':asyncio.run(diagnostic(resolve(a.source),resolve(a.out),a.limit))
    else:delivery(resolve(a.source),resolve(a.batch),resolve(a.out),resolve(a.diagnostics),[resolve(p) for p in a.previous_batch])
