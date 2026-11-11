"""Five objects × two scenes: half paint dust, local lighting and stronger independent optics."""
import argparse
import asyncio
import html
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from display_batch.ai import Vision
from display_batch.common import HERE, atomic_json, file_hash, load_catalog, resolve
from display_batch.compose import compose, cv2, sample_environment
from display_batch.costs import cumulative
from display_batch.framing import final_resolution
from display_batch.local import foreground_dust
from display_batch.local_light import estimate_local_light, receiver_calibration
from display_batch.photography import linear, LUMA
from display_batch.pipeline import batch_lock
from display_batch.rendering import RenderCache, render_dependencies
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig
from display_batch.surface_maps import prepare_uv_masks
from display_batch.worker import Blender
from feedback_rollback import BASELINE, ROLLBACK, SIZED, audit_png, placement_masks
from validate_display_dust import ASSETS, frozen_passes, read_rgb
from validate_display_gradients import (apply_camera, export_effect, frame_stats, full_preview,
                                       limit_matte_brightness, specular_p99)
from validate_display_local import read, records
from validate_display_scanned import geometry_check


OUTPUT = 'outputs/加强池化薄雾与局部受光-20261111-v3'
PREVIOUS = 'outputs/薄雾池化梯度与积灰哑光-20261111-v2'
PROFILE = {'version': 'half-paint-dust-v1', 'paint_dust_scale': .5, 'single_deposit': True}
POOL = (2, 4, 6)
FOG = (.12, .24, .36)


def frame(image, title, bounds, core):
    pixels = read_rgb(image)
    preview = Path(image).with_suffix('.jpg')
    full_preview(pixels, preview)
    return {'image': str(image), 'preview': str(preview), 'title': title,
            'image_hash': file_hash(image), 'size': [pixels.shape[1], pixels.shape[0]],
            **frame_stats(pixels, bounds, core)}


async def render_fixed(row, asset, scene, source, archived, renders, cache, photo_env, masks, local=False):
    legacy = row['archived_window'] is None
    resolution = archived['resolution'] if legacy else final_resolution(row['pose'])
    environment = photo_env if local or not legacy else sample_environment(scene, row['pose'])[0]
    environment = {**environment, 'dust_color': photo_env['dust_color'], 'haze': 0.}
    partitions = None
    if row['asset_id'] != '13':
        partitions, _ = await asyncio.to_thread(prepare_uv_masks, asset, cache/'scanned-uv')
    directory, meta, hit = await renders.get(asset, row['pose'], row['params'], environment,
        scene['size'], resolution, appearance='local', full_frame=legacy, matte_partitions=partitions,
        matte_profile=PROFILE if partitions else None, object_only=not local)
    geo = geometry_check(row['pose'], meta, archived, masks[0], row['baseline']['edit_bounds'], directory, masks[3])
    if not geo['pass']:
        raise ValueError(f'changed_geometry: {geo}')
    if partitions and not meta.get('matte_materials'):
        raise ValueError('missing_matte_materials')
    meta.update(anchor_uv=archived['anchor_uv'], contact_polygons=archived['contact_polygons'])
    if not local:
        meta['shadow_window'] = archived.get('shadow_window')
        for part in ('with_shadow', 'without_shadow'):
            meta['part_paths'][part] = archived.get('part_paths', {}).get(part, str(source))
    # Preserve the original sample grid for geometry; new shadow EXR has its own grid.
    meta['legacy_grid'] = legacy
    return directory, meta, {'geometry': geo, 'cache_hit': hit, 'timings': meta['render_timings'],
        'partition': partitions, 'render_dir': str(directory), 'render_meta': meta}


def audit_geometry(result, row, scene, masks):
    alpha, foreground, visible, offset = masks
    audit = audit_png(scene, result, alpha, foreground, visible, row['pose'], offset)
    if not audit['pass'] or any(result[k] != row['baseline'][k] for k in
                               ('alpha_hash','visible_alpha_hash','foreground_mask_hash')):
        raise ValueError('geometry_or_pixel_protection_failed')
    return audit


def compose_all(folder, row, scene, source, archived, shadow_hash, old_render, light_render,
                environment, samples, new_env, light_samples, masks, previous):
    started = time.monotonic()
    bounds = row['baseline']['edit_bounds']
    visible = masks[2]
    core = cv2.erode((visible > .9).astype(np.uint8), np.ones((5,5),np.uint8)) > 0
    original = read_rgb(row['image'])
    pose = {**row['pose'], 'fixed_alpha_dir': str(source), 'fixed_alpha_window': row['archived_window']}
    with tempfile.TemporaryDirectory(prefix='replay-', dir=folder) as temporary:
        a = compose(scene,row['pose'],source,archived,environment,samples,row['params'],row['seed'],
                    Path(temporary)/'A.png',capture_local=True)
        if not np.array_equal(read_rgb(a['image']),original) or a['shadow_hash']!=shadow_hash:
            raise ValueError('archived_replay_changed')
        audit_geometry(a,row,scene,masks)
        layers = a.pop('local_layers')
        old_info = None
        if old_render:
            directory,meta,old_info = old_render
            material = compose(scene,pose,directory,meta,environment,samples,row['params'],row['seed'],
                Path(temporary)/'material.png',capture_local=True,fixed_tone=a['photo'])
            audit_geometry(material,row,scene,masks)
            appearance = material.pop('local_layers')
            appearance,attenuation = limit_matte_brightness(layers,appearance,environment,row['params'],
                row['seed'],core,frame_stats(original,bounds,core)['luma_p95'])
            delta = linear(appearance['rgb'])-linear(layers['rgb'])
            material_frame = export_effect(folder,'paint-half','漆面积灰减半 · 原受光',original,a,layers,delta,
                                           bounds,environment,row['params'],row['seed'],core)
            old_info['brightness_limiter'] = attenuation
        else:
            material_frame = previous['groups']['pool'][0]
        directory,meta,light_info = light_render
        calibration = receiver_calibration(directory,meta,new_env)
        new_env = {**new_env, 'calibration': calibration}
        # Preserve colour restraint. Exposure comes from the receiver, not the object.
        tone = {**a['photo'], 'exposure_ev': calibration['exposure_ev'], 'contrast_gain': 1.}
        light = compose(scene,pose,directory,meta,new_env,samples,row['params'],row['seed'],
                        folder/'local-light.png',capture_local=True,fixed_tone=tone)
        audit = audit_geometry(light,row,scene,masks)
        light_layers = light.pop('local_layers')
        # Keep the same sensor response across both lights and every effect level.
        # compose's optical noise estimator may respond to the new object, so replay
        # its new shaded background with the original noise-estimation RGB.
        sensor_layers = {**layers, 'shaded': light_layers['shaded'], 'fixed_sensor': a['photo']}
        light_delta = linear(light_layers['rgb'])-linear(layers['rgb'])
        zero_pixels,_ = apply_camera(sensor_layers,light_delta,environment,row['params'],row['seed'])
        lx,ly,rx,by = bounds
        # Start from the validated new shadow frame; change only visible object RGB.
        complete = read_rgb(light['image']).copy()
        complete[ly:by,lx:rx][visible>0] = zero_pixels[visible>0]
        Image.fromarray(complete).save(light['image'],compress_level=2)
        audit = audit_geometry(light,row,scene,masks)
        zero = frame(light['image'],'局部受光 · 池化关闭 · 附加薄雾关闭',bounds,core)
        zero.update(pool_kernel=1,pool_weight=0.,fog_weight=0.,fog_extra=0.,fog_total=layers['air']['amount'])
        groups = {'pool':[zero], 'fog':[zero]}
        for kind,levels in (('pool',POOL),('fog',FOG)):
            for i,value in enumerate(levels,1):
                kernel,weight,fog = (value,1.,0.) if kind=='pool' else (1,0.,value)
                effect,details = foreground_dust(light_layers['clear_rgb'],light_layers['rgb'],visible,
                    environment,layers['air'],row['pose']['target_px'],pool_weight=weight,pool_kernel=kernel,
                    fog_fraction=0.,fog_weight=fog)
                title = f'完整 {kernel}×{kernel} 池化' if kind=='pool' else f'附加薄雾 {fog:.0%} 档'
                item = export_effect(folder,f'{kind}-{i}',title,complete,a,sensor_layers,light_delta+effect,
                                      bounds,environment,row['params'],row['seed'],core)
                item.update(pool_kernel=kernel,pool_weight=details['pool_weight_max'],fog_weight=fog,
                    fog_extra=details['fog_extra_max'],fog_total=details['fog_total_median'],effects=details)
                groups[kind].append(item)
            if len({v['image_hash'] for v in groups[kind]}) != 4:
                raise ValueError('effect_levels_not_distinct')
        initial = previous['groups']['pool'][0]
        diagnostics = np.asarray(scene['frame']).copy()
        diagnostics[light_samples] = np.uint8(.6*diagnostics[light_samples]+.4*np.array([30,210,140]))
        full_preview(diagnostics,folder/'light-samples.jpg')
        material_info = {'profile': PROFILE if old_render else None, 'old_light': old_info,
                         'new_light': light_info, 'calibration': calibration}
        if old_render:
            material_info['specular_p99'] = specular_p99(directory,meta,row['pose'],masks[3],bounds,visible)
        result = {'number':row['number'],'asset_id':row['asset_id'],'name':row['name'],'background':row['background'],
            'previous':initial,'material':material_frame,'groups':groups,'audit':audit,'status':'review',
            'light_evidence':new_env['local_irradiance'],'light_samples':str(folder/'light-samples.jpg'),
            'material_info':material_info,'photo':light['photo'],'checks':light['checks'],
            'seconds':round(time.monotonic()-started,3)}
    atomic_json(folder/'result.json',result)
    return result


def frames(case):
    values = [case['previous'],case['material']]
    for group in case['groups'].values():
        values.extend(group)
    return {v['image']:v for v in values}


def report(out,cases,summary):
    def link(path):
        return html.escape(Path(os.path.relpath(path,out)).as_posix(),quote=True)
    def picture(item):
        return (f'<figure><a href="{link(item["image"])}"><img loading="lazy" src="{link(item["preview"])}" '
                f'width="1600" height="900" alt="完整监控画面"></a><figcaption>{html.escape(item["title"])} '
                f'<a href="{link(item["image"])}">3840×2160 原图</a></figcaption></figure>')
    parts = []
    for bg in sorted({c['background'] for c in cases},key=lambda b:min(c['number'] for c in cases if c['background']==b)):
        parts.append(f'<h2>{html.escape(Path(bg).name)}</h2>')
        for c in sorted((c for c in cases if c['background']==bg),key=lambda c:c['number']):
            parts.append(f'<section id="sample-{c["number"]:02d}"><h3>{c["number"]:02d} · {html.escape(c["name"])}</h3>')
            if c.get('error'):
                parts.append(f'<p>{html.escape(c["error"])}</p></section>');continue
            if c['material_info']['profile']:
                parts.append('<details><summary>材质对照 · 相同原受光</summary><div class="frames">'+
                             picture({**c['previous'],'title':'上一轮积灰哑光'})+picture(c['material'])+'</div></details>')
            parts.append('<details open><summary>局部受光对照 · 相同新材质</summary><div class="frames">'+
                         picture({**c['material'],'title':'原受光'})+picture(c['groups']['pool'][0])+'</div></details>')
            for kind,title in (('pool','池化加强 · 附加薄雾关闭'),('fog','薄雾加强 · 池化关闭')):
                images = []
                for item in c['groups'][kind]:
                    caption = item['title']
                    if kind=='fog':
                        caption += f' · 实际追加峰值 {item["fog_extra"]:.1%} · 总雾量中位数 {item["fog_total"]:.1%}'
                    images.append(picture({**item,'title':caption}))
                parts.append(f'<details><summary>{title}</summary><div class="frames">'+''.join(images)+'</div></details>')
            parts.append(f'<details><summary>局部光照取样范围</summary><img src="{link(c["light_samples"])}" '
                         'loading="lazy" width="1600" height="900"><p>绿色为同层局部中性表面样本。光向证据不足时使用柔和环境光。</p></details></section>')
    head = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>加强池化薄雾与局部受光</title><style>body{font:16px system-ui;background:#192329;color:#edf3f5;margin:24px}'
        'a{color:#a9dfca}section{max-width:1800px;border-bottom:1px solid #41555d;padding:12px 0}'
        '.frames{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{margin:0}'
        'img{width:100%;height:auto;object-fit:contain;display:block}figcaption,summary{padding:8px 0}'
        'summary{cursor:pointer}h2{overflow-wrap:anywhere;font-size:1.1rem}@media(max-width:850px){.frames{grid-template-columns:1fr}}</style>'
        '<h1>加强池化薄雾与局部受光</h1><p>五件物品 × 两张背景。全部展示完整画面，点击查看原尺寸图。</p>'
        '<p>先比较材质和受光，再分别查看池化、薄雾。两个梯度的零档相同；每组只改变一个因素。</p>'
        '<p>漆面粉尘覆盖系数减半，漆面保持粗糙；裸金属保留哑光粉尘，塑料保持原材质。局部光照和尺度仍为估计。</p>'
        f'<p>{summary["cases_pass"]}/10 组程序检查通过，{summary["full_frames"]} 张全景。API ¥0；本机时间见账本。</p>'
        '<p><a href="cost-ledger.json">累计成本与耗时</a> · <a href="results.json">完整参数和检查</a></p>')
    (out/'index.html').write_text(head+''.join(parts),encoding='utf8')


async def run(args):
    start = time.monotonic();out = resolve(args.out)
    if out.exists() and any(out.iterdir()) and not (out/'inputs.json').is_file():
        raise ValueError('fresh_output_directory_required')
    out.mkdir(parents=True,exist_ok=True)
    with batch_lock(out):
        return await run_locked(args,out,start)


async def run_locked(args,out,start):
    rollback = resolve(ROLLBACK)
    config = BatchConfig(backgrounds='.',output=str(out),offline=True,render_device=args.device,
                         photo_style='dusty_scrap',max_iterations=0)
    cache = resolve(config.cache)
    catalog = load_catalog();hashes = {k:v['mesh_hash'] for k,v in catalog.items() if v['available']}
    if hashes!=read(rollback/'validation.json')['source_hashes']:
        raise ValueError('source_models_changed')
    rows = [r for r in read(rollback/'cases.json') if r['asset_id'] in ASSETS]
    backgrounds = {r['background'] for r in rows}
    if len(rows)!=10 or len(backgrounds)!=2 or {(r['asset_id'],r['background']) for r in rows}!={
            (asset,bg) for asset in ASSETS for bg in backgrounds}:
        raise ValueError('expected_five_by_two_frozen_cohort')
    history = {int(r['job_id'].split('-')[0]):r for r in records(resolve(BASELINE))}
    passes = {r['number']:frozen_passes(r,history,resolve(SIZED),cache) for r in rows}
    previous = {r['number']:r for r in read(resolve(PREVIOUS)/'results.json')}
    inputs = {'cohort':file_hash(rollback/'cases.json'),'pool':POOL,'fog':FOG,'profile':PROFILE,
        'device':args.device,'renderer':render_dependencies(),'sources':{name:file_hash(HERE/name) for name in
        (Path(__file__).name,'display_batch/compose.py','display_batch/local.py','display_batch/local_light.py',
         'display_batch/photography.py','display_batch/surface_maps.py','validate_display_gradients.py')},
        'passes':{str(i):v[3] for i,v in passes.items()},'previous':file_hash(resolve(PREVIOUS)/'results.json')}
    inputs = json.loads(json.dumps(inputs))
    if (out/'inputs.json').is_file() and read(out/'inputs.json')!=inputs:
        raise ValueError('frozen_inputs_changed_use_new_directory')
    atomic_json(out/'inputs.json',inputs)
    run_id = time.time_ns()
    vision = Vision(config,cache/'vision',out/f'usage-{run_id}.json')
    scenes = Scenes(cache,vision)
    worker = Blender(out/f'worker-{run_id}',args.device);renders = RenderCache(cache,worker)
    cases,reused = [],0
    try:
        for row in sorted(rows,key=lambda r:(int(r['asset_id']),r['number'])):
            number = row['number'];folder = out/'cases'/f'{number:02d}';folder.mkdir(parents=True,exist_ok=True)
            try:
                if file_hash(row['image'])!=row['image_hash'] or file_hash(row['background'])!=row['background_hash']:
                    raise ValueError('original_image_changed')
                asset = read(cache/'assets'/row['asset_signature']/'result.json')
                if asset['mesh_hash']!=hashes[row['asset_id']]:raise ValueError('asset_metadata_changed')
                row['pose']['material_classes'] = asset['semantic']['material_classes']
                if (folder/'result.json').is_file():
                    case = read(folder/'result.json')
                    if any(file_hash(v['image'])!=v['image_hash'] for v in frames(case).values()):
                        raise ValueError('saved_png_changed')
                    reused += 1
                else:
                    scene = await scenes.prepare(row['background'])
                    source,archived,shadow_hash,_ = passes[number]
                    masks = placement_masks(scene,row['pose'],source,row['archived_window'],archived,row['baseline']['edit_bounds'])
                    environment,samples = sample_environment(scene,row['pose'],local=True)
                    new_env,light_samples = estimate_local_light(scene,row['pose'],environment)
                    old_render = None
                    if row['asset_id']!='13':
                        old_render = await render_fixed(row,asset,scene,source,archived,renders,cache,environment,masks)
                    light_render = await render_fixed(row,asset,scene,source,archived,renders,cache,new_env,masks,local=True)
                    case = await asyncio.to_thread(compose_all,folder,row,scene,source,archived,shadow_hash,
                        old_render,light_render,environment,samples,new_env,light_samples,masks,previous[number])
            except (ValueError,OSError,RuntimeError,cv2.error) as error:
                case = {k:row[k] for k in ('number','asset_id','name','background')}
                case.update(status='failed',error=str(error))
            cases.append(case)
            print(json.dumps({'sample':number,'status':case['status'],'error':case.get('error')},ensure_ascii=False),flush=True)
    finally:
        await worker.close();await vision.close()
    if hashes!={k:file_hash(v['mesh']) for k,v in catalog.items() if v['available']}:
        raise ValueError('source_model_changed')
    complete = [c for c in cases if c['status']=='review']
    all_frames = {p:v for c in complete for p,v in frames(c).items()}
    summary = {'cases_pass':len(complete),'full_frames':len(all_frames),
        'new_pngs':sum(Path(p).is_relative_to(out) for p in all_frames),'source_models':len(hashes),
        'render_counts':dict(renders.totals),'timings':dict(renders.timings),
        'seconds':round(time.monotonic()-start,3),'composition_cache_hits':reused,'cost_total':cumulative(out)}
    atomic_json(out/f'run-{run_id}.json',summary)
    runs = [read(p) for p in out.glob('run-*.json')]
    ledger = {'runs':len(runs),'seconds':sum(r['seconds'] for r in runs),'cost_total':cumulative(out),
        'render_counts':{k:sum(r['render_counts'].get(k,0) for r in runs) for k in
                         ('object_renders','with_shadow_renders','without_shadow_renders','object_hits','asset_imports')},
        'timings':{k:sum(r['timings'].get(k,0) for r in runs) for k in
                   ('object_seconds','with_shadow_seconds','without_shadow_seconds','asset_import_seconds')},
        'local_compute_price':None,'comparison_only':True}
    atomic_json(out/'cost-ledger.json',ledger);atomic_json(out/'summary.json',summary);atomic_json(out/'results.json',cases)
    report(out,cases,summary)
    return summary


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=OUTPUT)
    parser.add_argument('--device',choices=('OPTIX','CUDA','CPU'),default='OPTIX')
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(int(result['cases_pass']!=10))
