"""Frozen 5 x 3 geometry rebuild using the user's 2.45 m inside-width reference."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import numpy as np
from PIL import ImageDraw,ImageFont
from display_batch.common import resolve,atomic_json,file_hash,load_catalog,signature,implementation_manifest
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig
from display_batch.vehicle_scale import prepare as prepare_vehicle
from display_batch.costs import cumulative
from validate_reference_projection import read,records,freeze,write_page

REFERENCE={'vehicle_class':'semitrailer','internal_width_m':2.45,
    'nominal_internal_length_m':12.85,'nominal_internal_height_m':2.2,
    'variable_dimensions':['nominal_internal_length_m','nominal_internal_height_m'],
    'source':{'kind':'user_supplied','reference':'用户于 2026-11-12 确认：半挂车，内宽 2.45 米；长高可能变化。'},
    'catalog_context':{'url':'https://atestsc.rioh.cn/page/notice/guache/a42d3f93-ed7d-4659-aaa4-6f7914f68ef2.html',
                       'use':'category_context_only; user-supplied width takes precedence'}}


def config(source,out,**kwargs):
    return BatchConfig.model_validate({**read(source/'config.json'),'output':str(out),
        'geometry_preset':'rebuild','vehicle_reference':REFERENCE,'max_iterations':2,
        'cpu_concurrency':1,**kwargs})


def diagram(scene,data,path):
    im=scene['frame'].copy();d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',32)
    for i,span in enumerate(data['observation']['width_spans']):
        xy=np.asarray(span)*np.subtract(scene['size'],1)
        d.line([tuple(p) for p in xy],fill='#ffcc40',width=7)
        qualifier='可见内沿' if data['observation']['rim_visibility']=='clear' else '内沿端点为估计'
        d.text(tuple(xy.mean(0)),f'{qualifier} {i+1} · 内宽 2.45 m',font=font,fill='#ffcc40')
    for s in scene['supports']:
        h=data['support_heights'][s['id']]
        d.text(tuple(s['anchor_px']),s['id']+f" 距上沿 {h['nominal_m']:.2f}m（估计）",font=font,fill='#7dffbf')
    d.text((30,95),'用户给定内宽；相机与料堆高度为估计。不是实测深度。',font=font,fill='white')
    im.save(path)


async def audit(source,out,offline=False):
    frozen=freeze(source,out);p=Pipeline(config(source,out/'audit',offline=offline));scenes=[];start=time.monotonic()
    try:
        for job in frozen['jobs']:
            if any(s['group']==job['group'] for s in scenes):continue
            scene=await p.scenes.prepare(job['background'])
            data=await prepare_vehicle(scene,REFERENCE,p.vision,p.cache)
            folder=out/'scenes';folder.mkdir(exist_ok=True)
            atomic_json(folder/(job['group']+'.json'),data)
            diagram(scene,data,folder/(job['group']+'-width.png'))
            scenes.append({'group':job['group'],'evidence':data})
            print(json.dumps({'stage':'vehicle','group':job['group'],'fit':data['calibration']['residual_px'],
                              'elevation':data['calibration']['elevation_deg']},ensure_ascii=False),flush=True)
        atomic_json(out/'audit.json',{'scenes':scenes,'seconds':time.monotonic()-start,
            'cost':cumulative(p.out,p.vision.usage()),'reference':REFERENCE})
    finally:
        await p.blender.close();await p.vision.close();p.store.close()


def gate(rows):
    if len(rows)!=3:return False
    for r in rows:
        if not r.get('image') or any(c.get('status')=='fail' for c in r.get('checks',{}).values()):return False
        iteration=next(v for v in r['iterations'] if v['iteration']==r['best_iteration'])
        if any(d['visibility']=='visible' and d['severity'] in ('major','fatal') and
               d['code'] in ('floating','penetration','occlusion','upright','scale','camera')
               for d in iteration['review']['defects']):return False
    return True


def geometry_delta(before,after,oldroot,newroot):
    def candidate(row,root):
        if not row.get('image'):return None
        entry=next(v for v in row['iterations'] if v['iteration']==row['best_iteration'])
        reused=row.get('reused_from')
        path=(Path(reused['batch'])/'jobs'/reused['job_id'] if reused else root/'jobs'/row['job_id'])/'candidates.json'
        candidates=read(path)['candidates']
        return next(c for c in candidates if c['id']==entry['params']['candidate'])
    old=candidate(before,oldroot);new=candidate(after,newroot)
    if not new:return {'status':'not_generated'}
    movement=float(np.linalg.norm(np.subtract(old['anchor_px'],new['anchor_px'])))
    rotation=float(np.linalg.norm(np.asarray(old['matrix'])[:3,:3]-np.asarray(new['matrix'])[:3,:3]))
    a=before.get('checks',{}).get('scale',{}).get('measurement_px',before.get('measured_major_px',old['target_px']))
    b=after.get('checks',{}).get('scale',{}).get('measurement_px',after.get('measured_major_px',new['target_px']))
    return {'old_major_px':a,'new_major_px':b,'old_anchor_px':old['anchor_px'],'new_anchor_px':new['anchor_px'],
        'measurement_basis':'full_projection_before_foreground_occlusion',
        'old_visible_major_px':before.get('measured_major_px'),'new_visible_major_px':after.get('measured_major_px'),
        'anchor_change_px':movement,'rotation_matrix_change':rotation,
        'size_changed':bool(abs(a-b)>1),'position_changed':bool(movement>1),'orientation_changed':bool(rotation>.01),
        'old_support_ids':old.get('support_ids'),'new_support_ids':new.get('support_ids'),
        'geometry_changed':bool(abs(a-b)>1 or movement>1 or rotation>.01),'scale_evidence':new['scale'],
        'physical_contact':new.get('terrain_physics'),'contact_gap':new.get('contact_gap')}


async def run(source,out):
    frozen=freeze(source,out);code=signature(implementation_manifest())[:10]
    first=out/('canary-'+code)
    await Pipeline(config(source,first,manifest=str(out/'canaries.json'),count=3)).run()
    rows=records(first);passed=gate(rows);final=first
    if passed:
        final=out/('complete-'+code)
        await Pipeline(config(source,final,manifest=str(out/'all-jobs.json'),count=15)).run(reuse_from=first)
        rows=records(final)
    atomic_json(out/'selected-run.json',{'canary':str(first),'final':str(final),'implementation':code})
    report(source,out,first,final)


def report(source,out,first=None,final=None):
    frozen=read(out/'manifest.json')
    if first is None:
        selected=read(out/'selected-run.json');first=Path(selected['canary']);final=Path(selected['final'])
    rows=records(final);passed=gate(records(first))
    verification_path=out/'verification/verification.json'
    verification=read(verification_path) if verification_path.is_file() else None
    if verification and Path(verification['batch']).resolve()!=final.resolve():
        raise ValueError('verification belongs to another generation run')
    verified={r['job_id']:r for r in verification['jobs']} if verification else {}
    previous=records(source);allrows=[];sections=[]
    for job in frozen['jobs']:
        row=next((r for r in rows if r['asset_id']==job['asset_id'] and r['background']==job['background']),None)
        if row is None:row={**job,'status':'not_run_gate','reason':'three_geometry_canaries_not_met'}
        checked=verified.get(row.get('job_id'))
        if checked and checked.get('generated'):
            row['checks']=checked['checks']
            row['evidence_verification']={'same_reviewed_png':checked['exact_png_replay'],
                'camera_interval_check':checked['camera_interval_check'],'report':str(verification_path)}
        old=next(r for r in previous if r['asset_id']==job['asset_id'] and r['background']==job['background'])
        row['geometry_delta']=geometry_delta(old,row,source,final)
        images=[]
        if row.get('image'):
            images=[('旧版完整图，仅作对照',old['image']),('重新计算尺度和摆位；相机及承托高度待审',row['image'])]
            if checked and checked.get('support_diagnostic'):
                images.append(('新承托与接触簇 · 完整画面诊断',checked['support_diagnostic']))
        elif row.get('job_id'):
            images=[('失败候选完整诊断，不是交付图',str(p)) for p in (final/'jobs'/row['job_id']).glob('*diagnostic.jpg')]
        diagnostic=out/'scenes'/(job['group']+'-width.png')
        if diagnostic.exists():images.append(('完整画面尺度依据',str(diagnostic)))
        text=row.get('reason','')
        if row.get('rejections'):text+='；淘汰原因：'+'；'.join(r['reason'] for r in row['rejections'])
        if row['geometry_delta'].get('geometry_changed'):
            d=row['geometry_delta'];text+=f"；长轴 {d['old_major_px']:.1f} → {d['new_major_px']:.1f}px，接触锚点移动 {d['anchor_change_px']:.1f}px"
            if not d['position_changed']:text+='（位置基本未变；尺寸/姿态重新计算）'
        if row.get('scores'):
            s=row['scores'];text+=f"；材质 {s['material']} / 受光 {s['lighting']} / 摄影 {s['texture']}"
            entry=next(e for e in row['iterations'] if e['iteration']==row['best_iteration'])
            text+='；缺陷：'+'；'.join(d['evidence'] for d in entry['review']['defects'] if d['severity'] in ('major','fatal'))
        sections.append({'title':job['group']+' · '+job['asset_id']+' · '+row['status'],'description':text,'images':images})
        allrows.append(row)
    attempts=[p for p in out.iterdir() if p.is_dir() and p.name.startswith(('canary-','complete-')) and (p/'summary.json').exists()]
    costs=[read(p/'summary.json')['cost_total'] for p in attempts]
    if (out/'audit.json').exists():costs.append(read(out/'audit.json')['cost'])
    cost={k:sum(c.get(k,0) or 0 for c in costs) for k in ('actual_requests','reported_tokens','estimated_api_min')}
    cost.update(estimated_api_max=sum(c['estimated_api_max'] for c in costs) if all(c.get('estimated_api_max') is not None for c in costs) else None,
                currency='CNY',complete=all(c.get('complete') for c in costs),local_compute_priced=False)
    per={g:sum(r.get('photo_status')=='pass' for r in allrows if r['group']==g) for g in {j['group'] for j in frozen['jobs']}}
    runs=[read(p) for a in attempts for p in a.glob('run-*.json')]
    diagnostic_runs=[read(p) for p in (out/'verification/runs').glob('*.json')]
    result={'jobs':allrows,'denominator':15,'canary_gate_passed':passed,'run_directory':str(final),
        'generated_images':sum(bool(r.get('image')) for r in allrows),'photo_pass':sum(per.values()),'per_scene':per,
        'states':{s:sum(r['status']==s for r in allrows) for s in sorted({r['status'] for r in allrows})},
        'geometry_changed_images':sum(r['geometry_delta'].get('geometry_changed',False) for r in allrows),
        'target_met':sum(per.values())>=12 and min(per.values())>=4,
        'source_models_unchanged':frozen['source_models']=={k:file_hash(v['mesh']) for k,v in load_catalog().items()},
        'verification':{k:v for k,v in verification.items() if k not in ('jobs','implementation')} if verification else None,
        'cost':cost,'generation_wall_seconds':sum(r['seconds'] for r in runs),
        'generation_blender_seconds':sum(r['gpu_seconds'] for r in runs),
        'render_rpc_seconds':sum(r.get('worker_operations_seconds',{}).get('render_parts',0) for r in runs),
        'simulation_rpc_seconds':sum(r.get('worker_operations_seconds',{}).get('settle',0) for r in runs),
        'render_counts':{k:sum(r.get('render_counts',{}).get(k,0) for r in runs)
            for k in ('object_renders','with_shadow_renders','without_shadow_renders','asset_imports')},
        'offline_verification_accounting':{
            'seconds':sum(r.get('seconds',0) for r in diagnostic_runs),
            'new_api_requests':sum(r.get('new_api_requests',0) for r in diagnostic_runs),
            'object_renders':sum(r.get('render_counts',{}).get('object_renders',0) for r in diagnostic_runs),
            'runs':len(diagnostic_runs),'includes_abandoned_development_time':False},
        'message':f"15 项中实际生成 {sum(bool(r.get('image')) for r in allrows)} 张新图；摄影达标 {sum(per.values())}/15。"
            '内宽采用用户提供的 2.45 米。新几何结果与旧图对比；未知证据保留待审，未切换批量默认。'}
    atomic_json(out/'validation.json',result)
    write_page(out,sections,{k:v for k,v in result.items() if k!='jobs'})
    page=out/'index.html'
    page.write_text(page.read_text(encoding='utf8').replace('<details>',
        '<p><a href="validation.json">完整逐项验证、评分与成本 JSON</a></p><details>',1),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','run','report']);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--offline',action='store_true',help='audit cached scene evidence without sending requests')
    a=p.parse_args();source=resolve(a.source);out=resolve(a.out)
    if a.stage=='report':report(source,out)
    elif a.stage=='audit':asyncio.run(audit(source,out,a.offline))
    elif a.offline:p.error('use audit --offline; generated PNGs require an online review')
    else:asyncio.run(run(source,out))


if __name__=='__main__':main()
