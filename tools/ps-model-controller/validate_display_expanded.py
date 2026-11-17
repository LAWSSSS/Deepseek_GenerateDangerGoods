"""Test the confirmed camera treatment on a fixed, wider archive sample."""
import argparse
import asyncio
from collections import Counter
import html
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np
from PIL import Image

from display_batch.ai import Vision
from display_batch.common import atomic_json, file_hash, load_catalog, resolve
from display_batch.compose import compose, sample_environment
from display_batch.costs import cumulative
from display_batch.framing import edit_bounds, final_resolution
from display_batch.global_light import estimate_global_mean
from display_batch.local import foreground_dust
from display_batch.local_light import estimate_local_light, receiver_calibration
from display_batch.photography import linear
from display_batch.pipeline import batch_lock
from display_batch.rendering import RenderCache, render_dependencies
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig
from display_batch.surface_dust_trial import dust_profile
from display_batch.surface_maps import prepare_uv_masks
from display_batch.worker import Blender
from feedback_rollback import BASELINE, ROLLBACK, SIZED, audit_png, placement_masks
from validate_display_dust import ASSETS, frozen_passes, read_rgb
from validate_display_gradients import apply_camera
from validate_display_local import cached_case, read, records
from validate_display_local_light import frame
from validate_display_scanned import geometry_check


OUTPUT = 'outputs/浮灰0.5-多场景扩测-20261112-v2'
REGRESSION = 'outputs/摄影嵌入-验证-v2-20261110/regression'
NEW_SCENES = ('36_5_31', '36_6_6', '53_4_15', '53_2_12')
SCENE_ASSETS = ('3', '13', '19-1')
PROFILE = dust_profile(.5)


def lighting_for(scene, pose, environment):
    try:
        local, _ = estimate_local_light(scene, pose, environment, statistic='mean', margin_scale=1.75)
    except ValueError as error:
        if str(error) != 'insufficient_local_neutral_light_evidence':
            raise
        local = {**environment, 'local_irradiance': {
            'bounds': environment['local_context']['bounds'], 'fill_ratio': .65,
            'direction': {'direction': None, 'confidence': 0., 'basis': str(error)}}}
    lighting, _ = estimate_global_mean(scene, local)
    return lighting


def compose_trial(folder, scene, pose, source, archived, reference, rendered,
                  lighting, environment, samples, params, seed, masks, bounds):
    alpha, foreground, visible, offset = masks
    directory, meta, _ = rendered
    calibration = receiver_calibration(directory, meta, lighting)
    geometry = geometry_check(pose, meta, archived, alpha, bounds, directory, offset)
    if not geometry['pass']:
        raise ValueError(f'frozen_geometry_mismatch: {geometry}')
    meta = {**meta, 'anchor_uv': archived['anchor_uv'], 'contact_polygons': archived['contact_polygons']}
    x, y, r, b = bounds
    core = cv2.erode(np.uint8(visible > .9), np.ones((5, 5), np.uint8)) > 0
    path = folder / 'trial.png'
    with tempfile.TemporaryDirectory(prefix='compose-', dir=folder) as temporary:
        baseline = compose(scene, pose, source, archived, environment, samples, params, seed,
                           Path(temporary)/'baseline.png', capture_local=True)
        tone = {**baseline['photo'], 'exposure_ev': calibration['exposure_ev'], 'contrast_gain': 1.}
        sensor = baseline['local_layers']
        frozen = {**pose, 'fixed_alpha_dir': str(source), 'fixed_alpha_window': archived.get('object_window')}
        value = compose(scene, frozen, directory, meta, {**lighting, 'calibration': calibration},
                        samples, params, seed, path, capture_local=True, fixed_tone=tone)
        layers = value['local_layers']
        sensor = {**sensor, 'shaded': layers['shaded'], 'fixed_sensor': baseline['photo']}
        effect, optics = foreground_dust(layers['clear_rgb'], layers['rgb'], visible, environment,
            layers['air'], pose['target_px'], pool_weight=1., pool_kernel=3, fog_weight=.35,
            uniform=True, fog_profile='wide')
        delta = linear(layers['rgb']) - linear(sensor['rgb']) + effect
        pixels, camera = apply_camera(sensor, delta, environment, params, seed)
        output = read_rgb(path).copy()
        output[y:b, x:r][visible > 0] = pixels[visible > 0]
        Image.fromarray(output).save(path, compress_level=2)
    audit = audit_png(scene, value, alpha, foreground, visible, pose, offset)
    if not audit['pass']:
        raise ValueError(f'png_readback_failed: {audit}')
    permitted = np.zeros(scene['size'][::-1], bool)
    permitted[y:b, x:r] = (scene['allowed'][y:b, x:r] | (alpha > 0)) & ~foreground & ~scene['protected'][y:b, x:r]
    if np.any(read_rgb(path)[~permitted] != np.asarray(scene['frame'])[~permitted]):
        raise ValueError('outside_object_and_shadow_pixels_changed')
    before = folder/'archive.png'
    Image.fromarray(read_rgb(reference)).save(before, compress_level=2)
    return {'before': frame(before, '归档合成', bounds, core), 'after': frame(path, '本轮扩测', bounds, core),
            'geometry': geometry, 'audit': audit, 'optics': optics, 'camera': camera,
            'lighting': lighting['local_irradiance'], 'calibration': calibration,
            'checks': value['checks'], 'matte_materials': meta.get('matte_materials', [])}


def report(out, cases, summary):
    def picture(item):
        preview = html.escape(Path(os.path.relpath(item['preview'], out)).as_posix(), quote=True)
        png = html.escape(Path(os.path.relpath(item['image'], out)).as_posix(), quote=True)
        return f'<figure><a href="{png}"><img src="{preview}" loading="lazy" alt="完整画面"></a><figcaption>{item["title"]} · 点击查看原尺寸</figcaption></figure>'

    parts = []
    backgrounds = list(dict.fromkeys(case['background'] for case in cases))
    for index, background in enumerate(backgrounds, 1):
        parts.append(f'<h2>场景 {index} · {html.escape(Path(background).name)}</h2>')
        for case in (row for row in cases if row['background'] == background):
            parts.append(f'<section><h3>{case["number"]:02d} · {html.escape(case["name"])}</h3>')
            if case.get('error'):
                parts.append(f'<p>{html.escape(case["error"])}</p>')
            else:
                optics = case['optics']
                parts.append('<div class="frames">'+picture(case['before'])+picture(case['after'])+'</div>')
                parts.append(f'<p>池化强度 {optics["pool_weight_max"]:.0%}，实际追加雾量 {optics["fog_extra_max"]:.1%}，'
                             f'加浮灰的材质 {len(case["matte_materials"])} 项。物理证据 {"通过" if case["audit"]["evidence"]=="pass" else "待核对"}。</p>')
            parts.append('</section>')
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>浮灰减半扩测</title>'
        '<style>body{margin:24px auto;padding:0 18px;max-width:1600px;background:#172128;color:#edf3f5;font:16px system-ui}'
        'a{color:#b1dfce}h2{overflow-wrap:anywhere}section{border-bottom:1px solid #52616a;padding-bottom:16px}'
        '.frames{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{margin:0}'
        'img{display:block;width:100%;height:auto}figcaption{padding:6px 0}@media(max-width:720px){.frames{grid-template-columns:1fr}}</style>'
        '<h1>浮灰减半扩测</h1><p>使用 3×3 池化、35% 追加薄雾和 0.5 倍表面浮灰。'
        f'{summary["images"]}/{len(cases)} 组通过图片检查，视觉效果待点评。</p>'
        '<p><a href="summary.json">运行与用量</a> · <a href="results.json">逐组检查</a> · <a href="inputs.json">固定样本清单</a></p>'
        +''.join(parts), encoding='utf8')


async def run(out):
    start = time.monotonic()
    rollback = resolve(ROLLBACK)
    regression = resolve(REGRESSION)
    rows = [r for r in read(rollback/'cases.json') if r['asset_id'] not in ASSETS]
    extra = [r for r in records(regression) if r['asset_id'] in SCENE_ASSETS and
             Path(r['background']).stem.endswith(NEW_SCENES)]
    if len(rows) != 14 or len(extra) != 12:
        raise ValueError('expected_fourteen_models_and_twelve_scene_cases')
    old = {int(r['job_id'].split('-')[0]): r for r in records(resolve(BASELINE))}
    seeds = {r['job_id']: job['seed'] for r, job in zip(records(regression), read(regression/'inputs.json'), strict=True)}
    config = BatchConfig(backgrounds='.', output=str(out), offline=True)
    cache = resolve(config.cache)
    catalog = load_catalog()
    hashes = {k: v['mesh_hash'] for k, v in catalog.items() if v['available']}
    if hashes != read(rollback/'validation.json')['source_hashes']:
        raise ValueError('source_models_changed')
    inputs = {'sources': {str(p): file_hash(p) for p in (rollback/'cases.json', regression/'results.jsonl', regression/'inputs.json')},
              'archive_numbers': [r['number'] for r in rows], 'regression_jobs': [r['job_id'] for r in extra],
              'pool_kernel': 3, 'fog_extra': .35, 'profile': PROFILE, 'renderer': render_dependencies(),
              'code': file_hash(__file__)}
    manifest = out/'inputs.json'
    if manifest.is_file() and read(manifest) != inputs:
        raise ValueError('frozen_inputs_changed_use_new_directory')
    atomic_json(manifest, inputs)
    run_id = time.time_ns()
    vision = Vision(config, cache/'vision', out/f'usage-{run_id}.json')
    scenes = Scenes(cache, vision)
    worker = Blender(out/f'worker-{run_id}', config.render_device)
    renders = RenderCache(cache, worker)
    cases = []
    try:
        for number, row in enumerate([*rows, *extra], 1):
            folder = out/'cases'/f'{number:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/'result.json'
            case = {'number': number, 'background': row['background'], 'asset_id': row['asset_id'],
                    'name': catalog[row['asset_id']]['name'], 'source_case': row.get('job_id', row.get('number'))}
            try:
                if path.is_file():
                    case = read(path)
                    if any(file_hash(case[k]['image']) != case[k]['image_hash'] for k in ('before', 'after')):
                        raise ValueError('saved_image_changed')
                else:
                    if number <= len(rows):
                        source, archived, _, _ = frozen_passes(row, old, resolve(SIZED), cache)
                        pose = dict(row['pose'])
                        params, seed = row['params'], row['seed']
                        scene = await scenes.prepare(row['background'])
                        asset = read(cache/'assets'/row['asset_signature']/'result.json')
                    else:
                        best, pose, asset, scene, source, archived = await cached_case(regression, row, cache, scenes)
                        params, seed = best['params'], seeds[row['job_id']]
                    if asset['mesh_hash'] != hashes[row['asset_id']]:
                        raise ValueError('archive_model_mismatch')
                    pose['material_classes'] = asset['semantic']['material_classes']
                    bounds = edit_bounds(pose, scene['size'])
                    window = archived.get('object_window')
                    masks = placement_masks(scene, pose, source, window, archived, bounds)
                    environment, samples = sample_environment(scene, pose, local=True)
                    lighting = lighting_for(scene, pose, environment)
                    partitions, _ = await asyncio.to_thread(prepare_uv_masks, asset, cache/'scanned-uv')
                    render = await renders.get(asset, pose, params, {**lighting, 'haze': 0.}, scene['size'],
                        archived['resolution'] if window is None else final_resolution(pose), appearance='local',
                        full_frame=window is None, matte_partitions=partitions, matte_profile=PROFILE)
                    params = {**params, 'photo_model': 'local', 'photo_style': 'dusty_scrap', 'atmosphere': 'auto'}
                    case.update(await asyncio.to_thread(compose_trial, folder, scene, pose, source, archived,
                        row['image'], render, lighting, environment, samples, params, seed, masks, bounds))
                    atomic_json(path, case)
            except (ValueError, OSError, RuntimeError, KeyError, cv2.error) as error:
                case['error'] = str(error)
            cases.append(case)
            print(f'{number:02d} {case["name"]} {case.get("error", "review")}', flush=True)
    finally:
        await worker.close()
        await vision.close()
    if hashes != {k: file_hash(v['mesh']) for k, v in catalog.items() if v['available']}:
        raise ValueError('source_models_changed_during_run')
    valid = [c for c in cases if 'error' not in c]
    atomic_json(out/f'run-{run_id}.json', {'seconds': round(time.monotonic()-start, 3), 'renders': dict(renders.totals)})
    runs = [read(p) for p in out.glob('run-*.json')]
    counts = Counter()
    for attempt in runs:
        counts.update(attempt['renders'])
    summary = {'images': len(valid), 'planned': len(cases), 'backgrounds': len({c['background'] for c in valid}),
               'assets': len({c['asset_id'] for c in valid}), 'source_models': len(hashes),
               'pool_applied': sum(c['optics']['pool_weight_max'] > 0 for c in valid),
               'fog_35_applied': sum(abs(c['optics']['fog_extra_max']-.35) < 1e-6 for c in valid),
               'seconds': sum(a['seconds'] for a in runs), 'renders': dict(counts), 'cost_total': cumulative(out)}
    atomic_json(out/'results.json', cases)
    atomic_json(out/'summary.json', summary)
    report(out, cases, summary)
    print(summary)
    return len(valid) == len(cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=OUTPUT)
    out = resolve(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    with batch_lock(out):
        raise SystemExit(0 if asyncio.run(run(out)) else 1)
