"""Freeze and run 12 reviewed asset classes x 3 content-selected new backgrounds."""
import argparse
import asyncio
from collections import Counter
import html
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime,timezone

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash, implementation_manifest, load_catalog, resolve, signature
from display_batch.costs import cumulative, label as cost_label
from display_batch.schema import BatchConfig
from validate_geometry_rebuild import REFERENCE


SOURCE = 'outputs/批量展示-20261112-110932-c70ec7'
BASELINE = 'outputs/回退低饱和点评-20261111/cases.json'
OUTPUT = 'outputs/综合融合-12类新场景-20261112'
EXCLUDED_GROUPS = ('36_6_24','53_7_35','53_2_12')
SEED = 20261112
VERSION = 'coherent-expanded-content-selection-v1'
FEATURES = ('brightness_mean','contrast_std','dynamic_range','chroma_mean','lab_a_mean','lab_b_mean','edge_density')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def group(path):
    match=re.search(r'(\d+_\d+_\d+)$',Path(path).stem)
    return match.group(1) if match else 'background-'+file_hash(path)[:10]


def features(path):
    """Bounded thumbnail statistics used only for evaluation sample selection."""
    with Image.open(path) as image:
        size=list(image.size)
        image=image.convert('RGB');image.thumbnail((384,216),Image.Resampling.BOX)
        rgb=np.asarray(image,dtype=np.float32)/255
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    gray=cv2.cvtColor(np.uint8(np.round(rgb*255)),cv2.COLOR_RGB2GRAY)
    light=lab[...,0]/100
    values=[float(light.mean()),float(light.std()),float(np.diff(np.percentile(light,[10,90]))[0]),
            float(np.linalg.norm(lab[...,1:],axis=2).mean()),float(lab[...,1].mean()),
            float(lab[...,2].mean()),float(np.mean(cv2.Canny(gray,60,120)>0))]
    return {'background':str(Path(path).resolve()),'group':group(path),'sha256':file_hash(path),
            'size':size,'features':dict(zip(FEATURES,values))}


def select_backgrounds(candidates,count=3,seed=SEED):
    """Seeded max-min sampling on standardized content, independent of output quality."""
    rows=sorted(candidates,key=lambda r:r['sha256'])
    if len(rows)<count:raise ValueError('insufficient_distinct_new_backgrounds')
    matrix=np.asarray([[r['features'][k] for k in FEATURES] for r in rows])
    center=matrix.mean(axis=0);spread=matrix.std(axis=0)
    normalized=(matrix-center)/np.maximum(spread,1e-6)
    rng=np.random.default_rng(seed);selected=[int(rng.integers(len(rows)))]
    distances=[]
    while len(selected)<count:
        nearest=np.min(np.linalg.norm(normalized[:,None]-normalized[selected][None,:],axis=2),axis=1)
        nearest[selected]=-1.
        winner=int(np.argmax(nearest));distances.append(float(nearest[winner]));selected.append(winner)
    return [rows[i] for i in selected],{'method':'seeded_farthest_point_standardized_content',
        'feature_order':list(FEATURES),'mean':center.tolist(),'std':spread.tolist(),
        'seed':seed,'initial_content_hash':rows[selected[0]]['sha256'],'successive_min_distances':distances,
        'selection_uses_generated_results':False,'independent_camera_holdout':False}


def source_hashes(catalog):
    return {k:file_hash(v['mesh']) for k,v in catalog.items() if v.get('available')}


def freeze(source,out,baseline=BASELINE,seed=SEED):
    out.mkdir(parents=True,exist_ok=True)
    path=out/'manifest.json'
    catalog=load_catalog()
    if path.is_file():
        frozen=read(path)
        if frozen['version']!=VERSION or frozen['seed']!=seed or frozen['source']!=str(source.resolve()):
            raise ValueError('frozen_selection_configuration_changed_use_new_directory')
        if frozen['source_models']!=source_hashes(catalog):raise ValueError('frozen_source_models_changed')
        if any(file_hash(r['background'])!=r['sha256'] for r in frozen['selected_backgrounds']):
            raise ValueError('frozen_background_changed')
        if file_hash(resolve(baseline))!=frozen['baseline_sha256'] or file_hash(source/'config.json')!=frozen['source_config_sha256']:
            raise ValueError('frozen_source_configuration_changed')
        if read(out/'all-jobs.json')!=frozen['jobs']:raise ValueError('frozen_job_manifest_changed')
        return frozen
    old=read(resolve(baseline));assets=list(dict.fromkeys(r['asset_id'] for r in old if r['asset_id'] not in ('5','6','7')))
    if len(old)!=24 or len(assets)!=12:raise ValueError('expected_reviewed_twelve_asset_classes')
    if any(k not in catalog or not catalog[k]['available'] for k in assets):raise ValueError('reviewed_asset_unavailable')
    cfg=read(source/'config.json');directory=resolve(cfg['backgrounds'])
    candidates=[];excluded=[];seen=set()
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):continue
        row=features(p)
        if row['sha256'] in seen:
            excluded.append({**row,'reason':'duplicate_image_content'});continue
        seen.add(row['sha256'])
        if row['group'] in EXCLUDED_GROUPS:
            excluded.append({**row,'reason':'latest_three_scene_groups'});continue
        candidates.append(row)
    selected,selection=select_backgrounds(candidates,seed=seed)
    jobs=[{'background':s['background'],'asset_id':a,'group':s['group'],'seed':seed+i*len(assets)+j,
           'background_hash':s['sha256'],'mesh_hash':catalog[a]['mesh_hash']}
          for i,s in enumerate(selected) for j,a in enumerate(assets)]
    hashes=source_hashes(catalog)
    if len(hashes)!=44:raise ValueError('expected_44_source_model_hashes')
    frozen={'version':VERSION,'seed':seed,'source':str(source.resolve()),'source_config_sha256':file_hash(source/'config.json'),
            'baseline':str(resolve(baseline)),'baseline_sha256':file_hash(resolve(baseline)),
            'assets':[{'id':a,'name':catalog[a]['name'],'mesh_hash':hashes[a]} for a in assets],
            'candidate_backgrounds':candidates,'excluded_backgrounds':excluded,'selected_backgrounds':selected,
            'selection':selection,'jobs':jobs,'denominator':36,'source_models':hashes,
            'reference':REFERENCE,'scope':'new_background_content_selection_not_independent_camera_generalization'}
    atomic_json(out/'manifest.json',frozen);atomic_json(out/'all-jobs.json',jobs)
    atomic_json(out/'batch-config.json',configuration(source,out/'batch',out).model_dump())
    return frozen


def configuration(source,batch,out):
    return BatchConfig.model_validate({**read(source/'config.json'),'output':str(batch),'count':36,
        'manifest':str(out/'all-jobs.json'),'geometry_preset':'rebuild','vehicle_reference':REFERENCE,
        'integration_mode':'coherent','camera_pipeline':'premultiplied','support_geometry':'structured',
        'max_iterations':2,'cpu_concurrency':1,'ai_concurrency':2,'offline':False})


def records(batch):
    path=batch/'results.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line] if path.is_file() else []


def total_cost(out):
    batches=sorted(p for p in out.glob('batch-*') if p.is_dir())
    ledgers=[cumulative(p) for p in batches]
    currencies={c['currency'] for c in ledgers if c['currency']}
    complete=all(c['complete'] for c in ledgers) and len(currencies)<=1
    return {'currency':next(iter(currencies)) if len(currencies)==1 else None,
        'actual_requests':sum(c['actual_requests'] for c in ledgers),
        'reported_tokens':sum(c['reported_tokens'] for c in ledgers),
        'estimated_api_min':sum(c['estimated_api_min'] or 0 for c in ledgers) if len(currencies)<=1 else None,
        'estimated_api_max':sum(c['estimated_api_max'] or 0 for c in ledgers) if complete else None,
        'complete':complete,'unpriced_or_unreported_calls':sum(c['unpriced_or_unreported_calls'] for c in ledgers),
        'local_compute_cost':None,'local_compute_basis':'price_not_configured',
        'scope':'all_attempts_in_this_expanded_experiment_only','batches':[str(p) for p in batches]}


def report(out):
    frozen=read(out/'manifest.json');selection=read(out/'selected-run.json') if (out/'selected-run.json').is_file() else {}
    paused=read(out/'paused.json') if (out/'paused.json').is_file() else None
    batch=Path(selection['batch']) if selection else out/'not-started'
    index={(r['asset_id'],r['background']):r for r in records(batch)}
    rows=[]
    for job in frozen['jobs']:
        row=index.get((job['asset_id'],job['background']))
        rows.append(row if row else {**job,'status':'not_run_paused' if paused else 'not_run',
            'reason':'user_paused_generation' if paused else 'not_completed_in_selected_run','image':None})
    runs=[read(p) for p in out.glob('batch-*/run-*.json')]
    counts=Counter()
    for run in runs:counts.update(run.get('render_counts',{}))
    photo=sum(r.get('photo_status')=='pass' for r in rows)
    summary={'denominator':36,'generated_images':sum(bool(r.get('image')) for r in rows),'photo_pass':photo,
        'visual_pass':sum(r.get('visual_status')=='pass' for r in rows),'states':dict(Counter(r['status'] for r in rows)),
        'per_scene':{s['group']:{'denominator':12,'images':sum(bool(r.get('image')) for r in rows if r['group']==s['group']),
                               'photo_pass':sum(r.get('photo_status')=='pass' for r in rows if r['group']==s['group'])}
                     for s in frozen['selected_backgrounds']},
        'per_asset':{a['id']:{'denominator':3,'photo_pass':sum(r.get('photo_status')=='pass' for r in rows if r['asset_id']==a['id'])} for a in frozen['assets']},
        'source_models_count':len(frozen['source_models']),
        'source_models_unchanged':frozen['source_models']==source_hashes(load_catalog()),
        'cost':total_cost(out),'seconds':sum(r.get('seconds',0) for r in runs) if runs else None,
        'blender_seconds':sum(r.get('gpu_seconds',0) for r in runs) if runs else None,'render_counts':dict(counts) if runs else None,
        'api_seconds':sum(r.get('api_seconds',0) for r in runs) if runs else None,'selected_run':selection,
        'generation_paused':paused,'completed_jobs':len(index),
        'local_timing_complete':all(list(p.glob('run-*.json')) for p in out.glob('batch-*') if p.is_dir()),
        'sample_selection':frozen['selection'],'batch_default_changed':False}
    atomic_json(out/'validation.json',{**summary,'jobs':rows})
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
          '<title>12 类物品 × 3 个新场景</title><style>body{max-width:1600px;margin:28px auto;padding:0 20px;background:#172128;color:#eef2f4;font:16px system-ui}img{display:block;width:100%;height:auto}a{color:#a9dfce}section{border-top:1px solid #52616a;margin-top:30px}p{line-height:1.8}pre{white-space:pre-wrap}figure{margin:20px 0}.warning{color:#ffd09a}</style>',
          '<h1>12 类物品 × 3 个新场景 · 完整画面点评</h1>',
          '<p>先冻结场景内容特征和任务，再生成。位置与尺寸重新计算；内宽采用用户提供的 2.45 米，相机与料堆高度仍有估计。未切换批量默认。</p>',
          f'<p>出图 {summary["generated_images"]}/36；摄影达标 {photo}/36；状态 {html.escape(str(summary["states"]))}。失败和未完成任务保留在分母。</p>',
          '<p>'+html.escape(cost_label(summary['cost']))+'；本机渲染/模拟 '+
              (format(summary['blender_seconds'],'.1f')+' 秒（仅含完整日志）' if summary['blender_seconds'] is not None else '日志不完整，不能记为 0 秒')+'。</p>',
          ('<p class="warning">用户已暂停批量生成；已执行 '+str(len(index))+'/36 项，其余不再自动运行。当前策略未达到自然融合要求。</p>' if paused else ''),
          '<p><a href="manifest.json">冻结清单与场景特征</a> · <a href="validation.json">全部评分、淘汰原因与累计成本</a> · <a href="batch-config.json">配置</a></p>']
    names={a['id']:a['name'] for a in frozen['assets']}
    def figure(path,caption):
        try:
            relative=Path(os.path.relpath(path,out)).as_posix()
        except ValueError:
            # Backgrounds are on C: while reports live on D:. Keep a full-size
            # content-addressed copy so the HTML also works when served over HTTP.
            source=Path(path);key=file_hash(source)
            local=out/'reference-backgrounds'/(key+source.suffix.lower())
            local.parent.mkdir(exist_ok=True)
            if not local.is_file():shutil.copy2(source,local)
            elif file_hash(local)!=key:raise ValueError('report_reference_copy_changed')
            relative=local.relative_to(out).as_posix()
        link=html.escape(relative,quote=True)
        return '<figure><a href="'+link+'"><img loading="lazy" src="'+link+'" alt="完整画面"></a><figcaption>'+html.escape(caption)+' · 点击查看原尺寸</figcaption></figure>'
    for s in frozen['selected_backgrounds']:
        body.append('<h2>场景 '+html.escape(s['group'])+'</h2>'+figure(s['background'],'原始背景'))
        for row in (r for r in rows if r['group']==s['group']):
            body.append('<section><h3>'+html.escape(row['asset_id']+' · '+names[row['asset_id']]+' · '+row['status'])+'</h3>')
            detail=row.get('reason','')
            if row.get('scores'):detail+='；评分 '+json.dumps(row['scores'],ensure_ascii=False)
            body.append('<p>'+html.escape(detail)+'</p>')
            if row.get('image'):
                with Image.open(row['image']) as im:
                    if list(im.size)!=s['size']:raise ValueError('report_requires_original_full_frame_dimensions')
                body.append(figure(row['image'],'待点评完整合成' if row['status']=='review' else '完整合成'))
            else:
                body.append('<p class="warning">本项未交付合成图；没有用旧图或失败版本补齐。</p>')
                if row.get('job_id'):
                    for path in sorted((batch/'jobs'/row['job_id']).glob('*diagnostic.jpg'))[:3]:
                        with Image.open(path) as im:
                            ratio=im.width/im.height
                        if abs(ratio-s['size'][0]/s['size'][1])<.01:
                            body.append(figure(path,'失败候选完整诊断，不是合格合成'))
            if row.get('rejections'):body.append('<details><summary>淘汰原因</summary><pre>'+html.escape(json.dumps(row['rejections'],ensure_ascii=False,indent=2))+'</pre></details>')
            body.append('</section>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf8')
    return summary


async def run(source,out,baseline=BASELINE,seed=SEED,*,resume=False):
    from display_batch.pipeline import Pipeline
    marker=out/'paused.json'
    if marker.exists():
        if not resume:raise ValueError('generation_paused_by_user; explicit --resume required')
        atomic_json(out/'previous-pause.json',read(marker));marker.unlink()
    freeze(source,out,baseline,seed)
    code=signature(implementation_manifest(),file_hash(__file__))[:12]
    batch=out/('batch-'+code)
    config=configuration(source,batch,out)
    atomic_json(out/'batch-config.json',config.model_dump())
    atomic_json(out/'selected-run.json',{'batch':str(batch),'implementation':code,'integration_mode':'coherent'})
    try:
        await Pipeline(config).run()
    finally:
        report(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run','report','pause'))
    parser.add_argument('--source',default=SOURCE);parser.add_argument('--out',default=OUTPUT)
    parser.add_argument('--baseline',default=BASELINE);parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--resume',action='store_true',help='Explicitly resume a user-paused experiment')
    args=parser.parse_args();source=resolve(args.source);out=resolve(args.out)
    if args.stage=='run':asyncio.run(run(source,out,args.baseline,args.seed,resume=args.resume))
    elif args.stage=='pause':
        atomic_json(out/'paused.json',{'requested_by':'user','time_utc':datetime.now(timezone.utc).isoformat(),
            'reason':'用户要求先暂停，效果不好，继续优化。',
            'current_execution_policy':'暂停批量生成，先做离线诊断和改进，不自动续跑。'})
        print(json.dumps(report(out),ensure_ascii=False))
    elif args.stage=='freeze':
        frozen=freeze(source,out,args.baseline,args.seed);summary=report(out)
        print(json.dumps({'jobs':len(frozen['jobs']),'assets':[r['id'] for r in frozen['assets']],
                          'backgrounds':[r['group'] for r in frozen['selected_backgrounds']],
                          'source_models':summary['source_models_count'],'out':str(out)},ensure_ascii=False))
    else:print(json.dumps(report(out),ensure_ascii=False))


if __name__=='__main__':main()
