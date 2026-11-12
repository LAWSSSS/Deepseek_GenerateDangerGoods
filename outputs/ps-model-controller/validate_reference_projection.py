"""Frozen audit -> three canaries -> gated 15-case run, with full-frame evidence."""
import argparse
import asyncio
import copy
import html
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from display_batch.common import atomic_json, resolve, file_hash, load_catalog, signature, implementation_manifest
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig
from display_batch.specifications import identify, scene_observations
from display_batch.reference_projection import audit
from display_batch.compose import compose, sample_environment
from display_batch.effects import lighting_for, dust_profile
from display_batch.local_atmosphere import prepare as prepare_air
from display_batch.costs import cumulative, label as cost_label
from validate_display_adaptation import cached_render


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def records(p):return [json.loads(s) for s in (p/'results.jsonl').read_text(encoding='utf8').splitlines() if s]


def config(source,out,**overrides):
    return BatchConfig.model_validate({**read(source/'config.json'),'output':str(out),
        'scale_mode':'reference_projection','contact_mode':'strict','chroma_mode':'local_budget',
        'max_iterations':2,'cpu_concurrency':1,**overrides})


def freeze(source,out):
    out.mkdir(parents=True,exist_ok=True);jobs=read(source/'inputs.json')
    groups=list(dict.fromkeys(j['group'] for j in jobs))
    if len(jobs)!=15 or len(groups)!=3:raise ValueError('expected_original_5_by_3')
    keys=[('22',groups[0]),('31',groups[0]),('3',groups[2])]
    selected=[next(j for j in jobs if (j['asset_id'],j['group'])==k) for k in keys]
    frozen={'version':'reference-trial-v1','source':str(source),'jobs':jobs,'canaries':selected,
            'source_models':{k:file_hash(v['mesh']) for k,v in load_catalog().items()}}
    path=out/'manifest.json'
    if path.exists() and signature(read(path))!=signature(frozen):raise ValueError('frozen_manifest_changed')
    atomic_json(path,frozen);atomic_json(out/'canaries.json',selected);atomic_json(out/'all-jobs.json',jobs)
    return frozen


def write_page(out,sections,summary):
    color_only='cases' in summary
    title='固定旧摆位的颜色诊断（未改尺寸和位置）' if color_only else '实物尺度、局部色度与严格接触'
    scope=('本页仅比较颜色，尺寸、落点、姿态和遮挡全部沿用历史结果。不是新尺度或物理合格交付。'
           if color_only else '估计尺度与合格实物尺度分开；物理失败只保存诊断，不进入交付区。')
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title>',
        '<style>body{background:#152027;color:#e8eeee;font:16px system-ui;margin:28px}img{width:100%;height:auto}a{color:#9cdac6}p{line-height:1.8}section{margin:28px 0;border-top:1px solid #567}pre{white-space:pre-wrap}</style>',
        '<h1>'+title+'</h1><p>'+scope+'全部图片为完整监控画面，可点击查看原尺寸。</p>',
        '<p>'+html.escape(summary.get('message',''))+'</p>',
        '<p>'+html.escape(cost_label(summary.get('cost',{})))+'</p>',
        '<details><summary>全部验证与成本</summary><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details>']
    if (out/'audit.html').exists():body.append('<p><a href="audit.html">尺寸档案与三个场景参照审计</a></p>')
    for section in sections:
        body.append('<section><h2>'+html.escape(section['title'])+'</h2><p>'+html.escape(section.get('description',''))+'</p>')
        for name,path in section.get('images',[]):
            link=html.escape(Path(os.path.relpath(path,out)).as_posix(),quote=True)
            body.append('<p>'+html.escape(name)+'</p><a href="'+link+'"><img loading="lazy" src="'+link+'"></a>')
        body.append('</section>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf8')


def audit_page(out,result):
    body=['<!doctype html><meta charset="utf-8"><title>尺寸来源与参照审计</title>',
          '<style>body{font:16px system-ui;background:#172027;color:#eee;margin:28px}td,th{padding:12px;border:1px solid #667}a{color:#addfcb}img{width:100%;height:auto}pre{white-space:pre-wrap}</style>',
          '<h1>尺寸档案与参照审计</h1><p>厂家尺寸通过仅指物品自身；三个场景均未找到可溯源标尺，不能称现场尺度已标定。</p>',
          '<table><tr><th>物品</th><th>固定长度 mm</th><th>状态和来源</th></tr>']
    for key,row in result['dimensions'].items():
        source=row.get('source');basis='类别估计；非实测规格'
        if source:basis='<a href="'+html.escape(source['url'],quote=True)+'">'+html.escape(source['row'])+'</a> · '+html.escape(row['feature'])
        body.append('<tr><td>'+html.escape(key)+'</td><td>'+format(row['nominal_mm'],'.2f')+'</td><td>'+basis+'</td></tr>')
    body.append('</table><p>新模式均使用倍率 1；历史展示缩放保留在旧模式。气缸和液压缸的行程、伸出状态长度没有确切型号证据，表中仍为估计。</p>')
    for group,row in result['scenes'].items():
        src='scenes/'+group+'-diagnostic.png'
        body.append('<h2>'+html.escape(group)+'</h2><p>'+html.escape(row['visual_observations']['evidence'])+'</p><a href="'+src+'"><img src="'+src+'"></a>')
    body.append('<details><summary>完整来源、排除理由与成本</summary><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre></details>')
    (out/'audit.html').write_text('\n'.join(body),encoding='utf8')


def reference_diagram(scene,out):
    im=scene['frame'].convert('RGB').copy();draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28)
    for site in scene['supports']:
        if not site['id'].startswith('S'):continue
        points=[tuple(np.asarray(p)*np.subtract(scene['size'],1)) for p in site['polygon']]
        draw.line(points+[points[0]],fill=(70,220,170),width=4)
        ends=[tuple(np.asarray(p)*np.subtract(scene['size'],1)) for p in site['reference']]
        draw.line(ends,fill=(255,180,50),width=5)
        text=site['id']+'：尺寸未核实'
        box=draw.textbbox(ends[0],text,font=font);draw.rectangle(box,fill=(20,28,35));draw.text(ends[0],text,font=font,fill='white')
    draw.rectangle((0,0,scene['size'][0],54),fill=(20,28,35))
    draw.text((20,8),'承托与参照诊断：绿线为候选承托，橙线为估计参照；不代表真实尺度或物理通过。',font=font,fill='white')
    out.parent.mkdir(parents=True,exist_ok=True);im.save(out)


async def evidence_audit(source,out):
    started=time.monotonic();frozen=freeze(source,out);pipe=Pipeline(config(source,out));rows=records(source)
    atomic_json(out/'config.json',pipe.config.model_dump());assets={};scenes={}
    try:
        for row in rows:
            key=row['asset_id']
            if key not in assets:
                asset=read(pipe.cache/'assets'/row['asset_signature']/'result.json')
                assets[key]=await identify(asset,pipe.vision,pipe.cache,pipe.config.dimension_registry)
                atomic_json(out/'dimensions'/(key+'.json'),assets[key])
                print(json.dumps({'stage':'dimension','asset':key,'status':assets[key]['status'],
                                  'nominal_mm':assets[key]['nominal_mm']},ensure_ascii=False),flush=True)
            if row['group'] not in scenes:
                scene=await pipe.scenes.prepare(row['background'])
                observations=await scene_observations(scene,pipe.vision,pipe.cache)
                scenes[row['group']]={**audit(scene),'visual_observations':observations}
                atomic_json(out/'scenes'/(row['group']+'.json'),scenes[row['group']])
                reference_diagram(scene,out/'scenes'/(row['group']+'-diagnostic.png'))
        result={'dimensions':assets,'scenes':scenes,'seconds':time.monotonic()-started,
                'cost':cumulative(out,pipe.vision.usage()),'render_calls':0,
                'source_models_unchanged':frozen['source_models']=={k:file_hash(v['mesh']) for k,v in load_catalog().items()}}
        atomic_json(out/'audit.json',result);audit_page(out,result);return result
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()


async def color_controls(source,out,limit=None):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    pipe=Pipeline(config(source,out,offline=True));rows=records(source);cases=[];sections=[]
    inputs=read(source/'inputs.json')
    try:
        for row in rows[:limit]:
            folder=out/row['job_id'];folder.mkdir(exist_ok=True)
            old=read(source/'jobs'/row['job_id']/f"iteration-{row['best_iteration']}.json")
            candidate=copy.deepcopy(next(c for c in read(source/'jobs'/row['job_id']/'candidates.json')['candidates']
                                    if c['id']==old['params']['candidate']))
            asset=read(pipe.cache/'assets'/row['asset_signature']/'result.json')
            candidate['material_classes']=asset['semantic']['material_classes']
            scene=await pipe.scenes.prepare(row['background'])
            seed=next(j['seed'] for j in inputs if j['asset_id']==row['asset_id'] and j['background']==row['background'])
            directory,meta=cached_render(pipe.cache,old,candidate,scene['size'])
            meta['matte_profile']=dust_profile(float(old['params']['surface_dust']),old['params']['weathering'])
            meta['shadow_pair_validation']=old['checks'].get('shadow_pair',{'status':'unknown'})
            env,samples=sample_environment(scene,candidate,local=True)
            env=lighting_for(scene,candidate,env,old['params']['effects_lighting'],preserve=True)
            env['calibration']=old['photo']['calibration']
            scene['atmosphere']=await pipe.atmospheres.prepare(scene)
            env['atmospheric_context']=prepare_air(scene,candidate,env,pipe.cache)
            baseline=compose(scene,candidate,directory,meta,env,samples,old['params'],seed,folder/'baseline.png',capture_local=True)
            visible=baseline.pop('local_layers')['alpha']>0
            new=compose(scene,candidate,directory,meta,env,samples,{**old['params'],'chroma_mode':'local_budget'},seed,folder/'local-chroma.png')
            for k in ('alpha_hash','visible_alpha_hash','foreground_mask_hash','shadow_hash'):
                if baseline[k]!=new[k]:raise ValueError('color_control_changed_'+k)
            before=np.asarray(Image.open(baseline['image']));after=np.asarray(Image.open(new['image']))
            allowed=np.zeros(before.shape[:2],bool);x,y,r,b=new['edit_bounds'];allowed[y:b,x:r]=visible
            changed=np.any(before!=after,axis=2)
            if changed[~allowed].any():raise ValueError('color_changed_outside_visible_object')
            case={'asset_id':row['asset_id'],'group':row['group'],'image':new['image'],
                  'baseline_matches_archive':file_hash(baseline['image'])==file_hash(old['image']),
                  'changed_pixels':int(changed.sum()),'outside_visible_changed':0,
                  'final_chroma':new['photo']['final_chroma'],'recompositions':new['photo']['chroma_recompositions'],
                  'checks':new['checks'],'status':'unreviewed_photography_control_not_delivery',
                  'alpha_holes_foreground_shadow_unchanged':True}
            cases.append(case);atomic_json(folder/'result.json',case)
            sections.append({'title':row['group']+' · '+row['asset_id'],
                'description':'固定旧摆位，仅验证颜色规则。旧图的尺度与物理问题仍然存在，本区不是合格交付图。',
                'images':[('旧摄影',baseline['image']),('局部色度上限',new['image'])]})
            print(json.dumps({'stage':'color','asset':row['asset_id'],'group':row['group'],
                              'chroma':case['final_chroma']['status'],'replay':case['baseline_matches_archive']},ensure_ascii=False),flush=True)
        result={'cases':cases,'seconds':time.monotonic()-started,'render_calls':0,
                'cost':cumulative(out,pipe.vision.usage()),'message':'摄影控制试验：复用 EXR，不代表物理通过。'}
        if pipe.vision.calls:raise ValueError('offline_color_made_api_calls')
        atomic_json(out/'color-controls.json',result);write_page(out,sections,result);return result
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()


def canary_gate(rows):
    return len(rows)==3 and all(r.get('image') and r.get('photo_status')=='pass' and
        all(c['status']!='fail' for c in r.get('checks',{}).values()) and
        not any(d['severity'] in ('major','fatal') for d in next(e for e in r['iterations'] if e['iteration']==r['best_iteration'])['review']['defects'])
        for r in rows)


async def run_trial(source,out):
    frozen=freeze(source,out)
    code=signature(implementation_manifest())[:8]
    first=out/('canary-run-'+code);cfg=config(source,first,manifest=str(out/'canaries.json'),count=3)
    summary=await Pipeline(cfg).run();rows=records(first)
    passed=canary_gate(rows)
    if passed:
        final=out/('complete-run-'+code);cfg=config(source,final,manifest=str(out/'all-jobs.json'),count=15)
        await Pipeline(cfg).run(reuse_from=first);rows=records(final)
    else:final=first
    sections=[];all_rows=[]
    for job in frozen['jobs']:
        row=next((r for r in rows if r['asset_id']==job['asset_id'] and r['background']==job['background']),None)
        if row is None:row={**job,'status':'not_run_gate','reason':'three_canary_gate_not_met'}
        all_rows.append(row);images=[]
        if row.get('image'):
            previous=next(r for r in records(source) if r['asset_id']==row['asset_id'] and r['background']==row['background'])
            images=[('历史结果（只作比较）',previous['image']),('估计尺度待审' if row.get('delivery_group')=='estimated_scale' else '新版完整图',row['image'])]
        elif row.get('job_id'):
            images=[('失败候选诊断，不是合成图',str(p)) for p in sorted((final/'jobs'/row['job_id']).glob('*diagnostic.jpg'))]
            scene_diagram=out/'scenes'/(job['group']+'-diagnostic.png')
            if scene_diagram.exists():images.append(('完整背景参照诊断',str(scene_diagram)))
            physical=final/'physics-models'/(row['asset_id']+'.json')
            if physical.exists():
                poses=read(physical)['poses'];codes=sorted({c for p in poses for c in p['failure_codes']})
                row['reason']+='：'+', '.join(codes)
        sections.append({'title':job['group']+' · '+job['asset_id']+' · '+row['status'],
                         'description':row.get('reason',''),'images':images})
    after={k:file_hash(v['mesh']) for k,v in load_catalog().items()}
    score_pass=sum(r.get('photo_status')=='pass' for r in all_rows)
    per_scene={g:sum(r.get('photo_status')=='pass' for r in all_rows if r['group']==g) for g in {j['group'] for j in frozen['jobs']}}
    attempts=sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith(('canary-run','complete-run')) and (p/'summary.json').exists())
    summaries=[read(p/'summary.json') for p in attempts]
    run_history=[read(r) for p in attempts for r in p.glob('run-*.json')]
    costs=[s['cost_total'] for s in summaries]
    # Each run owns its new calls only; include the separate audit ledger if present.
    if (out/'audit.json').exists():costs.append(read(out/'audit.json')['cost'])
    cost={k:sum(c.get(k,0) or 0 for c in costs) for k in ('actual_requests','reported_tokens','estimated_api_min')}
    cost['estimated_api_max']=sum(c['estimated_api_max'] for c in costs) if all(c.get('estimated_api_max') is not None for c in costs) else None
    currencies={c['currency'] for c in costs if c.get('currency')}
    cost.update(currency=next(iter(currencies)) if len(currencies)==1 else None,local_compute_priced=False,
                complete=all(c.get('complete',False) for c in costs))
    result={'jobs':all_rows,'denominator':15,'canary_gate_passed':passed,
        'images':sum(bool(r.get('image')) for r in all_rows),'photo_pass':score_pass,'per_scene':per_scene,
        'target_met':score_pass>=12 and all(v>=4 for v in per_scene.values()),
        'source_models_unchanged':after==frozen['source_models'],'cost':cost,
        'generation_attempts':[str(p) for p in attempts],
        'generation_wall_seconds':sum(s['seconds'] for s in run_history),
        'generation_gpu_seconds':sum(s['gpu_seconds'] for s in run_history),
        'message':'先行三组未达到门槛，停止其余 12 组；普通批量默认不变。' if not passed else '已完成冻结小样，详见各项证据。'}
    atomic_json(out/'validation.json',result);write_page(out,sections,result)
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','color','run']);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int)
    args=p.parse_args();source=resolve(args.source);out=resolve(args.out)
    if args.stage=='audit':asyncio.run(evidence_audit(source,out))
    elif args.stage=='color':asyncio.run(color_controls(source,out,args.limit))
    else:asyncio.run(run_trial(source,out))


if __name__=='__main__':main()
