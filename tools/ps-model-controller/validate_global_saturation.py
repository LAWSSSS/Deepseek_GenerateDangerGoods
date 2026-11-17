"""5 x 3 full-frame colour-only comparison; geometry is explicitly historical."""
import argparse
import asyncio
import copy
import html
import json
import os
from pathlib import Path
import time
from typing import Literal

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from display_batch.common import atomic_json, resolve, file_hash, load_catalog, signature
from display_batch.pipeline import Pipeline
from display_batch.schema import Strict, Score
from display_batch.compose import compose, sample_environment
from display_batch.effects import lighting_for, dust_profile
from display_batch.local_atmosphere import prepare as prepare_air
from display_batch.costs import cumulative, label as cost_label
from display_batch.global_saturation import VERSION
from validate_display_adaptation import cached_render
from validate_reference_projection import read, records, config


class PhotoScore(Strict):
    saturation: Score
    lightness: Score
    material: Score
    texture: Score
    identity: Score


class PairReview(Strict):
    a: PhotoScore
    b: PhotoScore
    preferred: Literal['A','B','equal']
    remaining: list[Literal['color_dense','color_washed_out','lightness','haze','material','texture','edge','scale','support']] = Field(max_length=6)
    evidence: str = Field(max_length=240)


PROMPT = '''Compare two photographic versions of the same inserted item. Images are A full panorama,
A detail, B full panorama, B detail. Geometry, size, placement, shadows and foreground are fixed.
Judge saturated/dense colour versus the pale nearby scrap, and whether either image fits naturally.
Do not assume the less colourful image is better: check chalky greys, lost original colour,
labels, metal response, brightness, texture and haze. Score saturation fit, lightness fit,
material credibility, texture and retained identity 0..4 (3 usable, 4 convincing), for each version.
Compare the actual images without inferring which is new. Identify remaining defects, including
any obvious scale/support defect, but this experiment cannot certify physical placement.
Return preferred A/B/equal. Do not infer improvement from numerical measurements.'''


def write_page(out, cases, summary):
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>全图平均饱和度 · 15 组完整画面对照</title>',
        '<style>body{margin:24px;background:#172128;color:#edf0f0;font:16px system-ui}a{color:#a5e3d2}img{width:100%;height:auto}section{border-top:1px solid #607078;margin-top:32px}p{line-height:1.8}.warning{padding:18px;background:#513e24}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #61717b}pre{white-space:pre-wrap}</style>',
        '<h1>全图平均饱和度：5 件 × 3 场景</h1>',
        '<p class="warning">仅测试颜色。两版都沿用历史尺寸、位置、姿态、遮挡和阴影；没有修复尺寸与摆位，不是新物理合格交付。上一轮严格接触的三组仍失败，未绕过门槛生成新场景。</p>',
        '<p>左/上版为上一轮局部色度约束；右/下版在其后增加整张原背景的 HSV 平均饱和度上限。仅降低有色物品像素，保留 Lab 亮度和色相方向。中性标签不稀释物品统计，低饱和物品不加色。背景统计包含料斗外部，作为此次全图方法的明确边界。</p>',
        '<p>图片均为完整 3840×2160 画面，点击打开原尺寸。自动比较使用随机 A/B 顺序；不根据评分逐图改变参数。</p>',
        '<p>'+html.escape(cost_label(summary.get('cost',{})))+'</p>',
        '<p>比较结果：改善 '+str(summary.get('improved',0))+'，相近 '+str(summary.get('equal',0))+'，变差 '+str(summary.get('worse',0))+'，缺少评分 '+str(summary.get('unreviewed',0))+'。颜色比较不等于整体场景验收。</p>',
        '<table><tr><th>背景</th><th>全图平均 S</th><th>全图 S90</th></tr>']
    for group,stats in summary.get('background_statistics',{}).items():
        parts.append(f'<tr><td>{html.escape(group)}</td><td>{stats["mean_s"]:.2%}</td><td>{stats["p90_s"]:.2%}</td></tr>')
    parts.append('</table>')
    for case in cases:
        stat=case['global_saturation'];before=stat['before'];after=stat['after']
        text=('有色物品平均 S：'+format(before['mean_s'],'.2%')+' → '+format(after['mean_s'],'.2%')) if 'mean_s' in before else '有色不透明像素不足，不强制处理'
        parts.append('<section><h2>'+html.escape(case['group']+' · '+case['asset_id'])+'</h2><p>'+html.escape(text)+'；Lab 亮度误差 P95：'+format(case['lightness_error_p95'],'.3f')+' / 100</p>')
        if case.get('identical_pixels'):
            parts.append('<p>两张图片逐像素相同：记为无变化，不进行重复视觉请求。</p>')
        elif case.get('review'):
            parts.append('<p>自动比较：'+html.escape(case['preference'])+'。'+html.escape(case['review']['evidence'])+'</p>')
        elif case.get('review_error'):
            parts.append('<p>未评分：自动复核格式错误，已用完一次重试；不据此认定效果改善。</p>')
        for name,key in [('上一轮局部约束（旧尺寸、旧摆位）','baseline'),('增加全图平均饱和度上限（同尺寸、同摆位）','trial')]:
            src=html.escape(Path(os.path.relpath(case[key],out)).as_posix(),quote=True)
            parts.append('<p>'+name+'</p><a href="'+src+'"><img loading="lazy" src="'+src+'"></a>')
        parts.append('</section>')
    parts.append('<details><summary>参数、验证、逐项评分和成本</summary><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details></html>')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf8')


async def run(source, controls, out, online, previous=()):
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    inputs=read(source/'inputs.json');rows=records(source)
    if len(rows)!=15 or len({r['group'] for r in rows})!=3:raise ValueError('expected_original_5_by_3')
    catalog=load_catalog();hashes={k:file_hash(v['mesh']) for k,v in catalog.items()}
    implementation={p.name:file_hash(p) for p in [Path(__file__),Path(__file__).parent/'display_batch'/'global_saturation.py',Path(__file__).parent/'display_batch'/'compose.py']}
    frozen={'version':VERSION,'source':str(source),'controls':str(controls),'jobs':inputs,'seed':20261112,
            'previous_attempts':[str(p) for p in previous],
            'models':hashes,'implementation':implementation,'color_only':True}
    if (out/'manifest.json').exists():
        prior=read(out/'manifest.json');old=copy.deepcopy(prior);new=copy.deepcopy(frozen)
        # Reporting/recovery edits do not change the composed pixels. Their
        # versions remain recorded; input, compositor and colour hashes must match.
        for value in (old,new):value['implementation'].pop(Path(__file__).name,None)
        if old!=new:raise ValueError('frozen_inputs_changed_use_new_directory')
        atomic_json(out/'runner-revisions'/(implementation[Path(__file__).name]+'.json'),frozen)
    else:atomic_json(out/'manifest.json',frozen)
    old_summary=read(out/'summary.json') if (out/'summary.json').exists() else None
    if old_summary and not list((out/'runs').glob('*.json')):
        atomic_json(out/'runs'/'historical-attempt.json',{'seconds':old_summary['seconds']})
    if old_summary and str(old_summary.get('interrupted','')).startswith('RuntimeError: global-mean-saturation-pair 两次请求'):
        # Migration from the first runner: the failing pair was saved before
        # the request, but the exhausted review had interrupted the loop.
        failed_index=old_summary['completed_cases']
        if failed_index<len(rows):
            path=out/rows[failed_index]['job_id']/'result.json'
            if path.exists():
                case=read(path)
                if not case.get('review'):
                    case['review_error']=old_summary['interrupted'];case['review_retry_exhausted']=True
                    atomic_json(path,case)
    pipe=Pipeline(config(source,out,offline=not online,chroma_mode='global_mean'))
    atomic_json(out/'config.json',pipe.config.model_dump());cases=[];summary={};interrupted=None
    try:
        for index,row in enumerate(rows):
            folder=out/row['job_id'];folder.mkdir(exist_ok=True);saved=folder/'result.json'
            old=read(source/'jobs'/row['job_id']/f"iteration-{row['best_iteration']}.json")
            if saved.exists():
                case=read(saved)
                for key in ('baseline','trial'):
                    if file_hash(case[key])!=case['output_hashes'][key]:raise ValueError('cached_image_changed')
            else:
                candidate=copy.deepcopy(next(c for c in read(source/'jobs'/row['job_id']/'candidates.json')['candidates'] if c['id']==old['params']['candidate']))
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
                baseline=await asyncio.to_thread(compose,scene,candidate,directory,meta,env,samples,
                    {**old['params'],'chroma_mode':'local_budget'},seed,folder/'local-budget.png',capture_local=True)
                visible=baseline.pop('local_layers')['alpha']>0
                archived=controls/row['job_id']/'local-chroma.png'
                if file_hash(baseline['image'])!=file_hash(archived):raise ValueError('baseline_not_bit_identical_to_displayed_control')
                trial=await asyncio.to_thread(compose,scene,candidate,directory,meta,env,samples,
                    {**old['params'],'chroma_mode':'global_mean'},seed,folder/'global-mean.png')
                for key in ('alpha_hash','visible_alpha_hash','foreground_mask_hash','shadow_hash'):
                    if baseline[key]!=trial[key]:raise ValueError('global_saturation_changed_'+key)
                before=np.asarray(Image.open(baseline['image']));after=np.asarray(Image.open(trial['image']))
                allowed=np.zeros(before.shape[:2],bool);x,y,r,b=trial['edit_bounds'];allowed[y:b,x:r]=visible
                changed=np.any(before!=after,axis=2)
                if changed[~allowed].any():raise ValueError('colour_leaked_outside_object')
                l0=cv2.cvtColor(before[y:b,x:r].astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
                l1=cv2.cvtColor(after[y:b,x:r].astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
                lerror=float(np.percentile(abs(l1-l0)[visible],95))
                if lerror>.5:raise ValueError('saturation_changed_lightness')
                if trial['checks']['global_saturation']['status']=='fail':raise ValueError('final_saturation_cap_failed')
                case={'asset_id':row['asset_id'],'group':row['group'],'job_id':row['job_id'],
                    'baseline':baseline['image'],'trial':trial['image'],'baseline_matches_previous':True,
                    'global_saturation':trial['photo']['global_saturation'],'lightness_error_p95':lerror,
                    'changed_pixels':int(changed.sum()),'outside_visible_changed':0,
                    'alpha_holes_foreground_shadow_unchanged':True,'geometry_unchanged':True,
                    'historical_geometry_status':'not_revalidated_not_deliverable',
                    'checks':trial['checks'],'bounds':candidate['bounds'],'size':scene['size'],
                    'output_hashes':{k:file_hash(v) for k,v in [('baseline',baseline['image']),('trial',trial['image'])]}}
                atomic_json(saved,case)
            case['identical_pixels']=case['output_hashes']['baseline']==case['output_hashes']['trial']
            if case['identical_pixels']:
                if case.get('review'):case['raw_ai_preference']=case.get('preference')
                case['preference']='equal';case['preference_basis']='pixel_identity'
                atomic_json(saved,case)
            elif online and not case.get('review') and not case.get('review_error'):
                order=['baseline','trial'];rng=np.random.default_rng(20261112+index);rng.shuffle(order)
                pictures=[]
                for key in order:
                    pictures.extend([case[key],str(Path(case[key]).with_suffix('.crop.jpg'))])
                w,h=case['size'];l,t,r,b=case['bounds']
                try:
                    answer=await pipe.vision.ask('global-mean-saturation-pair',PairReview,PROMPT,
                        {'asset_id':case['asset_id'],'object_region':[l/w,t/h,r/w,b/h],'scope':'colour_only_geometry_fixed'},
                        pictures,image_limits=[(1600,1000),(1000,800),(1600,1000),(1000,800)],max_output_tokens=1300)
                except RuntimeError as error:
                    from display_batch.ai import ProviderUnavailable
                    if isinstance(error,ProviderUnavailable):raise
                    case.update(review_error=str(error),review_retry_exhausted=True)
                else:
                    preference='equal' if answer.preferred=='equal' else order['AB'.index(answer.preferred)]
                    case.update(review=answer.model_dump(),blind_order=order,preference=preference,
                        scores={key:getattr(answer,label.lower()).model_dump() for label,key in zip('AB',order)})
                atomic_json(saved,case)
            cases.append(case)
            print(json.dumps({'asset':case['asset_id'],'scene':case['group'],'applied':case['global_saturation']['applied'],
                'saturation':case['global_saturation']['after'],'preference':case.get('preference','unreviewed')},ensure_ascii=False),flush=True)
    except Exception as error:
        interrupted=type(error).__name__+': '+str(error)[:240]
        raise
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()
        duration=time.monotonic()-started
        atomic_json(out/'runs'/('run-'+str(time.time_ns())+'.json'),{'seconds':duration,'interrupted':interrupted})
        summary={'version':VERSION,'cases':cases,'seconds':sum(read(p)['seconds'] for p in (out/'runs').glob('*.json')),
            'last_run_seconds':duration,'expected_cases':15,
            'completed_cases':len(cases),'improved':sum(c.get('preference')=='trial' for c in cases),
            'equal':sum(c.get('preference')=='equal' for c in cases),'worse':sum(c.get('preference')=='baseline' for c in cases),
            'unreviewed':sum(not c.get('review') and not c.get('identical_pixels') for c in cases),'interrupted':interrupted,
            'background_statistics':{c['group']:c['global_saturation']['background'] for c in cases},
            'source_models_unchanged':hashes=={k:file_hash(v['mesh']) for k,v in catalog.items()},
            'render_calls':0,'physics_calls':0,'new_geometry_deliverables':0,'promoted_default':False,
            'cost':cumulative(out,pipe.vision.usage())}
        summary['current_attempt_cost']=copy.deepcopy(summary['cost'])
        prior=[read(p/'summary.json') for p in previous]
        # Callers list distinct attempts, each counted by its own ledger only.
        costs=[p.get('current_attempt_cost',p['cost']) for p in prior]+[summary['current_attempt_cost']]
        currencies={c['currency'] for c in costs if c.get('currency')}
        summary['cost']={**summary['cost'],
            'actual_requests':sum(c['actual_requests'] for c in costs),
            'reported_tokens':sum(c['reported_tokens'] for c in costs),
            'estimated_api_min':sum(c['estimated_api_min'] or 0 for c in costs),
            'estimated_api_max':sum(c['estimated_api_max'] for c in costs) if all(c['estimated_api_max'] is not None for c in costs) else None,
            'currency':next(iter(currencies)) if len(currencies)==1 else None,
            'complete':all(c['complete'] for c in costs),'attempts':[str(p) for p in previous]+[str(out)]}
        summary['cumulative_wall_seconds']=sum(p['seconds'] for p in prior)+summary['seconds']
        atomic_json(out/'summary.json',summary);write_page(out,cases,summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','background_statistics')},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--controls',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--review',action='store_true')
    parser.add_argument('--previous',action='append',default=[])
    args=parser.parse_args()
    previous=[resolve(p) for p in args.previous]
    if len(set(previous))!=len(previous) or resolve(args.out) in previous:raise ValueError('duplicate_attempt_in_cost_ledger')
    asyncio.run(run(resolve(args.source),resolve(args.controls),resolve(args.out),args.review,previous))


if __name__=='__main__':main()
