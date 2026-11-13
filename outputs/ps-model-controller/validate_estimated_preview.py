"""Run a fixed 3-object by 2-background preview with provisional scene scale."""
import argparse
import asyncio
from collections import Counter
import html
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from display_batch.common import atomic_json, file_hash, load_catalog, resolve
from display_batch.costs import cumulative, label as cost_label
from display_batch.estimated_scale import VERSION as SCALE_VERSION
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig
from validate_coherent_expanded import source_hashes
from validate_evidence_workflow import batch_records, read
from validate_geometry_rebuild import REFERENCE

ASSETS = ('3', '22', '31')
SCENES = ('36_6_24', '53_7_35')
OUT = 'outputs/估算尺度-3物品2场景-20261113'


def freeze(out, backgrounds, assets=ASSETS, scene_ids=SCENES):
    if len(assets) != 3 or len(set(assets)) != 3 or len(scene_ids) != 2 or len(set(scene_ids)) != 2:
        raise ValueError('expected_three_distinct_assets_and_two_distinct_scenes')
    sources = {}
    for group in scene_ids:
        choices = [p.resolve() for p in backgrounds.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and p.stem.endswith('_'+group)]
        if len(choices) != 1:
            raise ValueError(f'expected_unique_background_for_{group}')
        sources[group] = {'path':str(choices[0]), 'sha256':file_hash(choices[0])}
    catalog = load_catalog()
    models = source_hashes(catalog)
    names = {key:catalog[key]['name'] for key in assets}
    jobs = [{'background':source['path'], 'asset_id':key, 'group':group, 'seed':20261113+3*i+j}
            for i,(group,source) in enumerate(sources.items()) for j,key in enumerate(assets)]
    frozen = {'backgrounds':sources, 'source_models':models, 'jobs':jobs, 'names':names,
              'basis':'user_authorized_preview_estimates', 'scene_selection':'one background per camera'}
    if (out/'frozen.json').is_file():
        if read(out/'frozen.json') != frozen:
            raise ValueError('preview_inputs_changed_use_a_new_directory')
        return frozen
    atomic_json(out/'jobs.json', jobs)
    atomic_json(out/'config.json', {
        'backgrounds':str(backgrounds), 'output':str(out/'batch'),
        'manifest':str(out/'jobs.json'), 'count':len(jobs), 'workflow':'evidence_v1',
        'geometry_preset':'rebuild', 'vehicle_reference':REFERENCE, 'scale_evidence_policy':'estimate_preview',
        'max_iterations':2, 'cpu_concurrency':1, 'ai_concurrency':1,
    })
    atomic_json(out/'frozen.json', frozen)
    return frozen


def batch_folders(out):
    return [out/'batch', *sorted(p for p in (out/'attempts').glob('*') if p.is_dir())]


async def run(out, attempt=None, retry_failed=False):
    frozen = read(out/'frozen.json')
    if read(out/'jobs.json') != frozen['jobs'] or source_hashes(load_catalog()) != frozen['source_models']:
        raise ValueError('frozen_jobs_or_models_changed')
    for source in frozen['backgrounds'].values():
        if file_hash(source['path']) != source['sha256']:
            raise ValueError('frozen_background_changed')
    raw = read(out/'config.json')
    if attempt:
        raw['output'] = str(out/'attempts'/attempt)
    if retry_failed:
        batch = Path(raw['output'])
        manifest = batch/'retry-jobs.json'
        if not manifest.is_file():
            results = preview_records(out)
            jobs = [job for job in frozen['jobs'] if not results.get(job_key(job), {}).get('image')]
            if not jobs:
                return report(out)
            atomic_json(manifest, jobs)
        jobs = read(manifest)
        if not jobs or any(job not in frozen['jobs'] for job in jobs) or len({job_key(job) for job in jobs})!=len(jobs):
            raise ValueError('retry_jobs_must_be_distinct_frozen_jobs')
        raw.update(manifest=str(manifest), count=len(jobs))
    await Pipeline(BatchConfig.model_validate(raw)).run()
    return report(out)


def job_key(job):
    return job['background'], job['asset_id']


def preview_records(out):
    return {job_key(row):row for batch in batch_folders(out) for row in batch_records(batch)}


def pixels(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'))


def timings(folders):
    runs = [read(path) for folder in folders for path in sorted(folder.glob('run-*.json'))]
    counts = Counter()
    for run in runs:
        counts.update(run.get('render_counts',{}))
    return {'runs':runs, 'wall_seconds':round(sum(run['seconds'] for run in runs),3),
            'worker_seconds':round(sum(run.get('gpu_seconds',0.) for run in runs),3),
            'api_seconds':round(sum(run.get('api_seconds',0.) for run in runs),3), 'render_counts':dict(counts)}


def write_overview(out, frozen, records):
    scenes = list(frozen['backgrounds'])
    assets = list(frozen['names'])
    by_pair = {(row['group'], row['asset_id']):row for row in records}
    width, height, gap, header = 960, 540, 12, 44
    sheet = Image.new('RGB', (2*width+3*gap, 3*(height+header)+4*gap), (242,243,242))
    try:
        font = ImageFont.truetype('msyh.ttc', 25)
    except OSError:
        font = ImageFont.load_default(size=22)
    draw = ImageDraw.Draw(sheet)
    for i, asset in enumerate(assets):
        top = gap+i*(height+header+gap)
        for j, scene in enumerate(scenes):
            left = gap+j*(width+gap)
            row = by_pair[scene, asset]
            image = row.get('preview_file')
            draw.text((left,top+6), f'{asset.zfill(2)}  /  {scene}', fill=(32,38,39),font=font)
            if image:
                with Image.open(image) as source:
                    frame = ImageOps.contain(source.convert('RGB'), (width,height), Image.Resampling.LANCZOS)
                sheet.paste(frame, (left+(width-frame.width)//2, top+header+(height-frame.height)//2))
            else:
                draw.text((left+32,top+header+32),'No deliverable',fill=(100,46,37),font=font)
    sheet.save(out/'overview.jpg',quality=92,subsampling=0)


def report(out):
    frozen = read(out/'frozen.json')
    index = preview_records(out)
    records = []
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>3 件物品 × 2 张背景</title>',
            '<meta name="viewport" content="width=device-width">',
            '<style>body{font:16px system-ui;margin:24px auto;padding:0 16px;max-width:1580px;background:#edf0f0}'
            'img{width:100%;height:auto;display:block}section{background:white;margin:16px 0;padding:14px}'
            'a{color:#16526a}summary{cursor:pointer}</style>',
            '<h1>3 件物品 × 2 张背景</h1>',
            f'<p>原尺寸检查图，估算版本 {SCALE_VERSION}。尺度与现场支撑含估算，每张保留程序检查和视觉反馈。</p>',
            '<p><a href="overview.jpg">六组全画面总览</a> · <a href="validation.json">逐项验证与累计用量</a></p>']
    shown = set()
    for job in frozen['jobs']:
        row = {**job, **index.get(job_key(job), {})}
        row.setdefault('status','not_run')
        records.append(row)
        group, asset = row['group'], row['asset_id']
        bg = Path(row['background'])
        if group not in shown:
            shown.add(group)
            relative = '背景-'+group+bg.suffix
            if not (out/relative).is_file():
                shutil.copy2(bg, out/relative)
            page.append('<h2>'+html.escape(group)+'</h2><details><summary>原始背景</summary>'
                        '<a href="'+relative+'"><img loading="lazy" src="'+relative+'" alt="完整背景"></a></details>')
        page.append('<section><h3>'+html.escape(asset.zfill(2)+' '+frozen['names'][asset])+'</h3>')
        if row.get('image'):
            best = next(entry for entry in row['iterations'] if entry['iteration']==row['best_iteration'])
            checks = best['checks']
            image = Path(row['image'])
            if (row['status']!='review' or not best.get('deliverable') or file_hash(image)!=best['image_hash']
                    or any(check['status']=='fail' for check in checks.values())):
                raise ValueError('preview_failed_delivery_verification')
            exported, source = pixels(image), pixels(bg)
            if exported.shape != source.shape:
                raise ValueError('preview_canvas_mismatch')
            changed = np.any(exported!=source,axis=2)
            x,y,r,b = best['edit_bounds']
            changed_pixels = int(changed.sum())
            outside = changed_pixels-int(changed[y:b,x:r].sum())
            if not changed_pixels or outside:
                raise ValueError('preview_export_pixel_verification_failed')
            row['preview_readback'] = {'sha256':best['image_hash'], 'canvas_px':list(exported.shape[1::-1]),
                                       'changed_pixels':changed_pixels, 'outside_bounds_changed_pixels':outside}
            name = group+'-'+asset.zfill(2)+'.png'
            shutil.copy2(image, out/name)
            row['preview_file'] = str(out/name)
            scale = checks['scale']['evidence']
            basis = {'same_visible_layer_category_estimate_with_vehicle_width_prior':'同层可见参照与车宽先验',
                     'visible_cargo_span_and_user_vehicle_width_approximation':'可见货堆跨度与车宽先验'}
            page.append('<p>待审 · 可见长边 '+str(round(row['measured_major_px']))+
                        ' px · '+html.escape(basis.get(scale['basis'],scale['basis']))+'</p>')
            page.append('<a href="'+name+'"><img loading="lazy" src="'+name+'" alt="原尺寸完整合成"></a>')
            findings = best.get('visual_evidence',{}).get('findings',[])
            page.append('<details><summary>复核记录</summary><ul>'+''.join(
                '<li>'+html.escape(item['evidence'])+'</li>' for item in findings)+'</ul></details>')
        else:
            page.append('<p>'+html.escape(row['status']+' · '+row.get('reason','未完成'))+'</p>')
        page.append('</section>')
    folders = batch_folders(out)
    ledgers = [cumulative(folder) for folder in folders]
    currency = {c['currency'] for c in ledgers if c['currency']}
    compatible = len(currency) <= 1
    complete = compatible and all(c['complete'] for c in ledgers)
    costs = {'currency':next(iter(currency)) if len(currency)==1 else None,
             'actual_requests':sum(c['actual_requests'] for c in ledgers),
             'reported_tokens':sum(c['reported_tokens'] for c in ledgers),
             'estimated_api_min':round(sum(c['estimated_api_min'] or 0 for c in ledgers),8) if compatible else None,
             'estimated_api_max':round(sum(c['estimated_api_max'] or 0 for c in ledgers),8) if complete else None,
             'unpriced_or_unreported_calls':sum(c['unpriced_or_unreported_calls'] for c in ledgers),
             'complete':complete, 'local_compute_cost':None}
    count = sum(bool(row.get('preview_file')) for row in records)
    performance = timings(folders)
    result = {'planned':len(frozen['jobs']), 'generated':count, 'jobs':records,
              'source_models_unchanged':source_hashes(load_catalog())==frozen['source_models'],
              'source_model_count':len(frozen['source_models']), 'cost_total':costs, 'api_ledgers':ledgers,
              'performance':performance}
    atomic_json(out/'validation.json', result)
    write_overview(out,frozen,records)
    page.insert(6,f'<p>出图 {count}/{len(records)}。'+html.escape(cost_label(costs))+
                f'。累计运行 {performance["wall_seconds"]:.1f} 秒，Blender {performance["worker_seconds"]:.1f} 秒。</p>')
    (out/'index.html').write_text('\n'.join(page),encoding='utf8')
    return {'planned':result['planned'], 'generated':count, 'source_models_unchanged':result['source_models_unchanged'],
            'cost_total':costs, 'output':str(out/'index.html')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','run','report'))
    parser.add_argument('--out', default=OUT)
    parser.add_argument('--backgrounds', default='C:/Users/zhaoc/Desktop/背景图')
    parser.add_argument('--assets', nargs=3, default=ASSETS)
    parser.add_argument('--scenes', nargs=2, default=SCENES)
    parser.add_argument('--attempt')
    parser.add_argument('--retry-failed', action='store_true', help='retry only missing or failed outputs')
    args = parser.parse_args()
    if args.attempt and (not args.attempt.isascii() or not args.attempt.replace('-','').replace('_','').isalnum()):
        parser.error('attempt name must contain letters, digits, dashes or underscores')
    if args.retry_failed and (args.command!='run' or not args.attempt):
        parser.error('--retry-failed requires run with a named --attempt')
    out = resolve(args.out)
    if args.command=='freeze':
        frozen = freeze(out, resolve(args.backgrounds), args.assets, args.scenes)
        value = {'planned':len(frozen['jobs']), 'output':str(out)}
    elif args.command=='run':
        value = asyncio.run(run(out,args.attempt,args.retry_failed))
    else:
        value = report(out)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    main()
