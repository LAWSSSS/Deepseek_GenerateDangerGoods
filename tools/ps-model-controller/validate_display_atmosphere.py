"""Fixed-six atmospheric comparisons followed by a gated 18-image batch."""
import argparse
import asyncio
import copy
import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash, implementation_manifest, resolve, signature
from display_batch.framing import render_windows
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig
from validate_display_costs import montage, selected_geometry, ASSETS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def records(folder):
    return [json.loads(line) for line in (Path(folder)/'results.jsonl').read_text(encoding='utf8').splitlines()]


def validation_signature(implementation=None):
    return signature(implementation or implementation_manifest(),file_hash(__file__))


class FixedPipeline(Pipeline):
    def __init__(self, config, baseline):
        super().__init__(config)
        self.baseline = baseline
        self.originals = {(r['background'],r['asset_id']):r for r in records(baseline)}

    async def choose_placements(self, asset, scene, seed):
        row = self.originals[(scene['path'],asset['id'])]
        old, candidate = selected_geometry(self.baseline, row)
        candidate = copy.deepcopy(candidate)
        key = old['render_cache']['object']['signature']
        original = read(self.cache/'render-parts-v1'/'object'/key/'result.json')
        candidate['fixed_alpha_dir'] = original['path']
        candidate['fixed_alpha_window'] = render_windows(candidate,scene['size'],old['render_resolution'])['object']
        candidate['material_classes'] = asset['semantic']['material_classes']
        return [candidate], []

    async def validate_local_pose(self, asset, scene, candidate):
        return candidate


def photo_pass(row):
    scores = row.get('scores', {})
    return bool(row.get('image') and row.get('insertion_trace')=='none'
                and scores.get('lighting',0)>=3 and scores.get('material',0)>=3
                and row.get('visual_status')=='pass')


def fixed_report(out, baseline, summary):
    original = {(r['background'],r['asset_id']):r for r in records(baseline)}
    comparisons = []
    sections = []
    analyses = {}
    for row in records(out):
        source = original[(row['background'],row['asset_id'])]
        check = {'job':row['job_id'], 'asset':row['asset_id'], 'has_image':bool(row.get('image')),
                 'photographic_pass':photo_pass(row)}
        if row.get('image'):
            old = next(v for v in source['iterations'] if v['iteration']==source['best_iteration'])
            best = next(v for v in row['iterations'] if v['iteration']==row['best_iteration'])
            check.update(alpha_equal=old['alpha_hash']==best['alpha_hash'],
                visible_alpha_equal=old['visible_alpha_hash']==best['visible_alpha_hash'],
                foreground_equal=old['foreground_mask_hash']==best['foreground_mask_hash'],
                anchor_error_px=best['checks']['anchor']['projection_error_px'],
                best_hash=file_hash(row['image']),old_hash=file_hash(source['image']))
            check.update(scores=best['review']['scores'],trace=best['review'].get('trace'),
                defects=best['review']['defects'],atmosphere=best['photo'].get('atmosphere'),
                exposure_ev=best['photo'].get('exposure_ev'),checks=best['checks'])
            atmospheric=best['photo'].get('atmosphere')
            if atmospheric:
                source_folder=Path(atmospheric['analysis']).parent
                relative=Path('atmosphere')/atmospheric['signature']
                destination=out/relative
                destination.mkdir(parents=True,exist_ok=True)
                for name in ('estimate.json','veil.png','samples.png'):
                    shutil.copyfile(source_folder/name,destination/name)
                analyses[atmospheric['signature']]=relative
            picture=out/(row['job_id']+'-comparison.jpg')
            montage([('旧版全景',source['image']),('新版全景',row['image']),
                     ('旧版局部',source['crop']),('新版局部',row['crop'])],picture,large=True)
            sections.append(f'<article><h2>{html.escape(row["asset_id"])}</h2>'
                f'<a href="{Path(row["image"]).relative_to(out).as_posix()}">原尺寸 PNG</a>'
                f'<img src="{picture.name}"><pre>{html.escape(json.dumps(check,ensure_ascii=False))}</pre></article>')
        comparisons.append(check)
    exact = len(comparisons)==6 and all(c.get('alpha_equal') and c.get('visible_alpha_equal')
        and c.get('foreground_equal') and c.get('anchor_error_px',99)<=1.5 for c in comparisons)
    good = sum(c['photographic_pass'] for c in comparisons)
    gate = {'stage':6, 'generated':sum(c['has_image'] for c in comparisons), 'photographic_pass':good,
            'fixed_geometry_pass':exact, 'passed':exact and good>=5 and summary['run']['state']=='complete',
            'implementation':validation_signature(read(out/'implementation.json')), 'records':comparisons}
    atomic_json(out/'gate.json',gate)
    page='<!doctype html><meta charset="utf-8"><title>材质受光与薄雾对照</title>'
    page+='<style>body{background:#172128;color:#eee;font:16px sans-serif;margin:30px}img{width:100%}a{color:#8bd5b7}pre{white-space:pre-wrap}</style>'
    page+='<h1>固定六项材质受光与薄雾对照</h1><p>全部任务按冻结清单运行。</p>'
    page+=f'<p>摄影通过 {good}/6 · 几何一致 {exact}</p><a href="index.html">完整批次记录</a>'+''.join(sections)
    page+='<h2>背景局部成像估计</h2><p>色图为 1 − t，黑底白点为数值采样区。数值属于解码 sRGB 下的估计，不能当作真实空气密度。置信度和深度关联见记录。</p>'
    for relative in analyses.values():
        path=relative.as_posix()
        page+=f'<a href="{path}/estimate.json">成像估计记录</a><div><img style="width:48%" src="{path}/veil.png"><img style="width:48%" src="{path}/samples.png"></div>'
    (out/'comparison.html').write_text(page,encoding='utf8')
    return gate


def select_three(baseline, cache):
    rows = records(baseline)
    entries = {}
    for row in rows:
        path=row['background']
        if path in entries:
            continue
        analysis=read(cache/'scenes'/row['scene_signature']/'analysis.json')
        with Image.open(path) as frame:
            image=np.float32(frame.convert('L').resize((128,72)))/255
        p05,p50,p95=np.percentile(image,[5,50,95])
        # Frozen content-based grouping, before new composite images are generated.
        columns=image.reshape(3,24,4,32).mean(axis=(1,3))
        entries[path]={'black_floor':float(p05/max(p95,.1)), 'haze_hint':analysis['haze'],
                       'spatial_spread':float(columns.std()), 'hash':file_hash(path)}
    remaining=sorted(entries,key=lambda p:entries[p]['hash'])
    clear=min(remaining,key=lambda p:entries[p]['black_floor'])
    remaining.remove(clear)
    haze=max(remaining,key=lambda p:entries[p]['black_floor'])
    remaining.remove(haze)
    nonuniform=max(remaining,key=lambda p:entries[p]['spatial_spread'])
    return {'backgrounds':[clear,haze,nonuniform], 'features':entries,
            'group_names':['least_veil','raised_black_floor','spatially_nonuniform'],
            'basis':'content_proxies_not_verified_fog_classes'}


async def main(args):
    out=resolve(args.out)
    out.mkdir(parents=True,exist_ok=True)
    baseline=resolve(args.baseline)
    config=BatchConfig.model_validate(read(baseline/'config.json'))
    values={'output':str(out),'offline':args.offline,'appearance':'native','atmosphere':'auto',
            'max_iterations':2,'count':6,'provider_config':args.provider_config or config.provider_config}
    if args.stage==6:
        jobs=read(baseline/'inputs.json')
        if len(jobs)!=6:
            raise ValueError('固定对照要求六项已完成原始任务')
    else:
        gate=read(resolve(args.pilot)/'gate.json') if args.pilot else {}
        if not gate.get('passed') or gate.get('implementation')!=validation_signature():
            raise ValueError('六项摄影门槛未通过或代码已变化，18 项保持停止')
        selection=select_three(resolve(args.regression_baseline),resolve(config.cache))
        atomic_json(out/'selection.json',selection)
        jobs=[{'background':path,'asset_id':asset,'group':group,'seed':20261110+i*6+j}
              for i,(path,group) in enumerate(zip(selection['backgrounds'],selection['group_names']))
              for j,asset in enumerate(ASSETS)]
        values['count']=18
    manifest=out/'manifest.json'
    if manifest.is_file() and read(manifest)!=jobs:
        raise ValueError('冻结清单发生变化，需要新输出目录')
    atomic_json(manifest,jobs)
    values['manifest']=str(manifest)
    config=config.model_copy(update=values)
    atomic_json(out/'request-config.json',config.model_dump())
    atomic_json(out/'diagnostic.json',{'stage':args.stage,'uniform_strategy_revision':args.revision,
        'max_uniform_revisions':2,'baseline':str(baseline),'validation_signature':validation_signature()})
    runner=FixedPipeline(config,baseline) if args.stage==6 else Pipeline(config)
    summary=await runner.run()
    if args.stage==6:
        gate=fixed_report(out,baseline,summary)
    else:
        rows=records(out)
        per_asset={asset:sum(photo_pass(row) for row in rows if row['asset_id']==asset) for asset in ASSETS}
        gate={'stage':18,'photographic_pass':sum(per_asset.values()),'per_asset':per_asset,
              'passed':len(rows)==18 and sum(per_asset.values())>=15 and min(per_asset.values())>=2
                       and summary['run']['state']=='complete','implementation':validation_signature()}
        atomic_json(out/'gate.json',gate)
    print(json.dumps({'gate':{k:v for k,v in gate.items() if k!='records'},
        'usage':{k:v for k,v in summary['usage_current_run'].items() if k!='calls'},
        'run':summary['run']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='outputs/scene-native-auto-v4-20261110')
    parser.add_argument('--regression-baseline',default='outputs/摄影嵌入-验证-v2-20261110/regression')
    parser.add_argument('--out',required=True)
    parser.add_argument('--stage',type=int,choices=(6,18),default=6)
    parser.add_argument('--pilot')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--revision',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--provider-config')
    asyncio.run(main(parser.parse_args()))
