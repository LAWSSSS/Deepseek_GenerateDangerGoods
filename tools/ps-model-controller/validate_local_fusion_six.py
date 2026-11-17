"""Frozen-geometry six-frame material/airlight comparison; no selection or iteration.

``freeze`` and ``report`` are offline. ``run`` renders the selected historical
geometry once and reviews each actual PNG once (the client's bounded repair
retry still applies). Original images remain evidence, never review substitutes.
"""
import argparse
import asyncio
import copy
import html
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from display_batch.ai import ProviderUnavailable, VisionEndpoint
from display_batch.common import HERE, atomic_json, file_hash, implementation_manifest, load_catalog, resolve, signature
from display_batch.compose import compose, object_window, warp
from display_batch.costs import cumulative, estimate, label
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig
from display_batch.strict_contact import visual_veto
from display_batch.structured import inspect_frame
from verify_geometry_rebuild import saved_render

VERSION = 'fixed-local-fusion-six-v1'
SOURCE = 'outputs/估算尺度-3物品2场景-20261113'
OUTPUT = 'outputs/局部薄雾与金属做旧-六图-20261113'
MASK_KEYS = ('alpha_hash', 'visible_alpha_hash', 'foreground_mask_hash')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def model_hashes():
    return {key: file_hash(row['mesh']) for key, row in load_catalog().items()
            if row.get('available') and Path(row['mesh']).is_file()}


def selected(source):
    jobs = read(source/'validation.json')['jobs']
    pairs = {(str(row['asset_id']), row['group']) for row in jobs}
    groups = sorted({group for _, group in pairs})
    if len(jobs) != 6 or pairs != {(asset, group) for asset in ('3', '22', '31') for group in groups} or len(groups) != 2:
        raise ValueError('baseline must contain the fixed three assets by two backgrounds')
    result = []
    for row in sorted(jobs, key=lambda r: (r['group'], int(r['asset_id']))):
        entry = next(e for e in row['iterations'] if e['iteration'] == row['best_iteration'])
        candidate_file = Path(entry['image']).parent/'candidates.json'
        candidate = next(c for c in read(candidate_file)['candidates'] if c['id'] == entry['params']['candidate'])
        if file_hash(row['image']) != entry['image_hash']:
            raise ValueError(f"baseline published image differs from selected iteration: {row['job_id']}")
        result.append((row, entry, candidate, candidate_file))
    return result


def freeze(source, out):
    source, out = resolve(source), resolve(out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for row, entry, candidate, candidate_file in selected(source):
        items.append({'key': f"{row['group']}-{row['asset_id']}", 'asset_id': str(row['asset_id']),
            'group': row['group'], 'job_id': row['job_id'], 'seed': row['seed'],
            'baseline': row['image'], 'baseline_hash': entry['image_hash'],
            'background': row['background'], 'background_hash': file_hash(row['background']),
            'asset_signature': row['asset_signature'], 'scene_signature': row['scene_signature'],
            'candidate_file': str(candidate_file), 'candidate_file_hash': file_hash(candidate_file),
            'candidate_signature': signature(candidate), 'candidate': candidate,
            'params': entry['params'], 'params_signature': signature(entry['params']),
            'resolution': entry['render_resolution'], 'baseline_entry': entry})
    payload = {'version': VERSION, 'source': str(source), 'source_validation_hash': file_hash(source/'validation.json'),
        'source_models': model_hashes(), 'items': items,
        'policy': 'same six selected images; exact candidate, camera, scale, pose and parameters; no search or model iteration'}
    payload['signature'] = signature(payload)
    path = out/'frozen.json'
    if path.is_file() and read(path) != payload:
        raise ValueError('frozen inputs changed; use a new output directory')
    atomic_json(path, payload)
    write_cached_scene_evidence(source, out, payload)
    return payload


def write_cached_scene_evidence(source, out, frozen):
    """Expose cached scene hints without constructing a client or starting a GPU."""
    from display_batch.atmosphere import VERSION as atmosphere_version
    config = BatchConfig.model_validate(read(source/'config.json'))
    raw = read(config.provider_config or HERE/'batch-config.example.json')
    endpoint = VisionEndpoint.model_validate({**(raw.get('vision') or raw),
        'max_output_tokens': 3072, 'timeout_seconds': 120})
    records = []
    for item in frozen['items']:
        key = signature(atmosphere_version, item['scene_signature'], endpoint.model_dump())
        path = resolve(config.cache)/'atmospheres'/key/'estimate.json'
        summary = read(path) if path.is_file() else {}
        records.append({'case': item['key'], 'scene_key': item['scene_signature'],
            'background': item['background'], 'atmosphere_cache': str(path),
            'visual_evidence': summary.get('visual_evidence'), 'summary': summary,
            'status': 'cached' if summary else 'missing'})
    atomic_json(out/'source-scene-evidence.json', {'scenes': records, 'new_requests': 0})


def pixel_stats(rgb, mask):
    pixels = np.asarray(rgb)[np.asarray(mask, bool)]
    if not len(pixels):
        return {'pixels': 0, 'status': 'unknown'}
    pixels = np.float32(pixels)/255
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    luma = pixels @ np.float32([.2126, .7152, .0722])*255
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    return {'pixels': len(pixels), 'status': 'measured', 'luma_median_255': float(np.median(luma)),
        'luma_p10_p90_255': np.percentile(luma, [10, 90]).tolist(),
        'lab_chroma_median': float(np.median(chroma)), 'lab_chroma_p90': float(np.percentile(chroma, 90)),
        'basis': 'exported_display_RGB_luma_and_CIELAB_chroma; geometric_visible_alpha_gt_0.5'}


def air_summary(photo):
    effects, atmosphere = photo.get('effects', {}), photo.get('atmosphere', {})
    return {'haze': photo.get('haze'), 'fog_total_median': effects.get('fog_total_median'),
        'fog_total_max': effects.get('fog_total_max'), 'transmission_range': atmosphere.get('transmission_range'),
        'color_linear': photo.get('local_air', {}).get('color_linear'), 'atmosphere': atmosphere,
        'local_reference': photo.get('local_context', {}), 'exposure_ev': photo.get('exposure_ev'),
        'blur_sigma': photo.get('blur_sigma'), 'surface_dust': effects.get('requested', {}).get('surface_dust')}


def check_invariants(item, comp, meta, old_meta, scene, candidate, layers, *, estimated_occlusion=False):
    old = item['baseline_entry']
    invariants = {key: old[key] == comp[key] for key in MASK_KEYS}
    geometry = {k: v for k, v in candidate.items() if k not in ('fixed_alpha_dir', 'fixed_alpha_window')}
    invariants['candidate_unchanged'] = signature(geometry) == item['candidate_signature']
    invariants['parameters_unchanged'] = signature(item['params']) == item['params_signature']
    invariants['camera_unchanged'] = meta['camera'] == old_meta['camera']
    with Image.open(comp['image']) as image:
        invariants['canvas_unchanged'] = tuple(image.size) == tuple(scene['size'])
    anchor_error = float(np.linalg.norm(np.subtract(meta['anchor_uv'], old_meta['anchor_uv']))*candidate['full_px'])
    invariants['contact_anchor_unchanged'] = anchor_error <= 1e-6
    occlusion = comp['checks'].get('occlusion', {})
    # Unknown scene ordering can still preserve every identified foreground pixel.
    foreground_preserved = occlusion.get('status') == 'pass' or (
        estimated_occlusion and occlusion.get('status') == 'unknown'
        and occlusion.get('foreground_changed_pixels') == 0)
    invariants['source_pixels_preserved'] = foreground_preserved and all(
        comp['checks'].get(k, {}).get('status') == 'pass' for k in ('protected', 'outside_edit', 'holes_alpha'))
    invariants['edit_bounds_unchanged'] = old['edit_bounds'] == comp['edit_bounds']
    if invariants['edit_bounds_unchanged']:
        bounds = comp['edit_bounds']
        with Image.open(item['baseline']) as image:
            before = np.asarray(image.convert('RGB').crop(bounds))
        with Image.open(comp['image']) as image:
            after = np.asarray(image.convert('RGB').crop(bounds))
        mask = layers['alpha'] > .5
        stats = {'before': pixel_stats(before, mask), 'after': pixel_stats(after, mask),
            'visible_changed_pixels': int(np.any(before != after, axis=2)[mask].sum())}
    else:
        stats = {'status': 'unknown', 'reason': 'inconsistent_edit_bounds'}
    return {'status': 'pass' if all(invariants.values()) else 'fail', 'checks': invariants,
        'contact_anchor_error_px': anchor_error, 'before_major_px': old['measured_major_px'],
        'after_major_px': comp['measured_major_px'], 'geometric_target_px': candidate['target_px'],
        'anchor_px': candidate['anchor_px'], 'fixed_geometry_signature': item['candidate_signature'],
        'shadow_changed': old['shadow_hash'] != comp['shadow_hash'],
        'shadow_policy': 'material/light-dependent shadow is recomputed; contact geometry stays frozen'}, stats


def raw_alpha_audit(item, folder, meta, old_folder, old_meta):
    masks = []
    for directory, data in ((old_folder, old_meta), (folder, meta)):
        with Image.open(directory/'object.png') as image:
            alpha = np.asarray(image.getchannel('A'), np.float32)/255
        c = item['candidate']
        projected = warp(alpha, c['full_px'], c['offset'], item['baseline_entry']['edit_bounds'],
                         window=object_window(data))
        masks.append(projected > .5)
    union = int((masks[0] | masks[1]).sum())
    iou = float((masks[0] & masks[1]).sum()/max(1, union))
    lengths = []
    for mask in masks:
        yy, xx = np.nonzero(mask)
        lengths.append(float(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)) if len(xx) else 0.)
    tolerance = max(3., .05*lengths[0])
    return {'status': 'pass' if union and iou >= .975 and abs(lengths[1]-lengths[0]) <= tolerance else 'fail',
        'iou': iou, 'minimum_iou': .975, 'raw_major_px': lengths, 'major_tolerance_px': tolerance,
        'basis': 'raw_new_render_vs_baseline_before_frozen_alpha; material_adaptive_sampling_edge_check'}


def relative(path, out):
    try:
        return os.path.relpath(path, out).replace('\\', '/')
    except ValueError:
        return Path(path).resolve().as_uri()


def bind_implementation(path, current_signature, implementation):
    if path.exists() and read(path)['signature'] != current_signature:
        raise ValueError('implementation changed in this fixed experiment; use a new output directory')
    atomic_json(path, {'signature': current_signature, 'implementation': implementation})


def write_report(out):
    frozen = read(out/'frozen.json')
    rows = []
    for item in frozen['items']:
        path = out/'cases'/item['key']/'result.json'
        rows.append(read(path) if path.exists() else {**{k: item[k] for k in ('key', 'asset_id', 'group')},
            'status': 'not_run', 'baseline': item['baseline'], 'stop_reason': 'not_run'})
    source_unchanged = model_hashes() == frozen['source_models']
    runs = [read(p) for p in sorted((out/'runs').glob('*.json'))] if (out/'runs').exists() else []
    summary = {'version': VERSION, 'planned': len(rows), 'generated': sum(bool(r.get('image')) for r in rows),
        'reviewed': sum(bool(r.get('review')) for r in rows), 'accepted_display': 0,
        'review': sum(r['status'] == 'review' for r in rows), 'failed': sum(r['status'] == 'failed' for r in rows),
        'photo_passes': sum(bool(r.get('photography_pass')) for r in rows),
        'source_models_unchanged': source_unchanged, 'source_model_count': len(frozen['source_models']),
        'cost_total': cumulative(out), 'runs': runs, 'jobs': rows,
        'note': 'Photography diagnostic only. Estimated scene scale and contact remain review. '
                'Rejected images are diagnostics, not deliverables. Historical scores do not verify new PNGs.'}
    atomic_json(out/'validation.json', summary)
    cards = []
    for row in rows:
        title = f"{row['group']} · {int(row['asset_id']):02d} · {row['status']}"
        images = []
        for name, path in [('修改前（历史已选版本）', row['baseline']),
                           ('修改后（尺度与接触待审）' if row.get('deliverable') else '修改后诊断（不可交付）', row.get('image'))]:
            images.append(f'<figure><figcaption>{html.escape(name)}</figcaption><a href="{html.escape(relative(path,out))}">'
                f'<img loading="lazy" src="{html.escape(relative(path,out))}"></a></figure>' if path else '<p>未生成。</p>')
        details = {key: row.get(key) for key in ('stop_reason', 'metrics', 'air_before', 'air_after', 'invariants', 'review')}
        cards.append(f'<section><h2>{html.escape(title)}</h2>{"".join(images)}<details><summary>测量、复核与停止原因</summary>'
            f'<pre>{html.escape(json.dumps(details,ensure_ascii=False,indent=2))}</pre></details></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>局部薄雾与金属做旧：六图对照</title>'
        '<style>body{max-width:1800px;margin:24px auto;background:#eee;color:#222;font:16px sans-serif}'
        'section{background:white;margin:24px 0;padding:20px}figure{margin:12px 0}img{width:100%;height:auto}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}figcaption{padding:8px}</style>'
        '<h1>局部薄雾与金属做旧：固定几何六图对照</h1>'
        '<p>全部展示完整画面，点击查看原尺寸 PNG。位置、姿态、尺寸和前景遮挡冻结；只更新统一摄影与材质算法。'
        '尺度和接触的未知证据仍待审。完整记录包含所有失败项。</p>'
        f'<p>{html.escape(label(summary["cost_total"]))}</p>' + ''.join(cards), encoding='utf8')
    make_overview(out, rows)
    return summary


def make_overview(out, rows):
    # Every tile preserves the complete frame; no crop is used in the report.
    width, height, label_h = 960, 540, 48
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    sheet = Image.new('RGB', (width*2, (height+label_h)*len(rows)), '#20242b')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        for j, (name, path) in enumerate([('修改前', row['baseline']), ('修改后 '+row['status'], row.get('image'))]):
            x, y = j*width, i*(height+label_h)
            draw.text((x+14, y+10), f"{row['group']} · {int(row['asset_id']):02d} · {name}", fill='white', font=font)
            if path and Path(path).exists():
                with Image.open(path) as im:
                    im = im.convert('RGB'); im.thumbnail((width, height), Image.Resampling.LANCZOS)
                    sheet.paste(im, (x+(width-im.width)//2, y+label_h+(height-im.height)//2))
            else:
                draw.text((x+20, y+label_h+30), '未出图；详见失败记录', fill='white', font=font)
    sheet.save(out/'完整画面对照.png')
    # Preserve full frames in a compact six-image preview as well.
    groups = sorted({row['group'] for row in rows})
    assets = sorted({row['asset_id'] for row in rows}, key=int)
    current = Image.new('RGB', (width*len(groups), (height+label_h)*len(assets)), '#20242b')
    draw = ImageDraw.Draw(current)
    for row in rows:
        x, y = groups.index(row['group'])*width, assets.index(row['asset_id'])*(height+label_h)
        draw.text((x+14, y+10), f"{row['group']} · {int(row['asset_id']):02d} · {row['status']}", fill='white', font=font)
        if row.get('image') and Path(row['image']).exists():
            with Image.open(row['image']) as im:
                im = im.convert('RGB'); im.thumbnail((width, height), Image.Resampling.LANCZOS)
                current.paste(im, (x+(width-im.width)//2, y+label_h+(height-im.height)//2))
    current.save(out/'新版完整画面总览.jpg', quality=95)


async def run(source, out):
    started = time.monotonic(); source, out = resolve(source), resolve(out)
    frozen = freeze(source, out)
    implementation = implementation_manifest()
    extra_sources = {name: file_hash(HERE/name) for name in ('scanned_shader.py',) if (HERE/name).is_file()}
    current_signature = signature(VERSION, implementation, extra_sources, file_hash(__file__))
    binding = out/'implementation.json'
    bind_implementation(binding, current_signature, {**implementation, 'extra_sources': extra_sources})
    config = BatchConfig.model_validate({**read(source/'config.json'), 'output': str(out),
                                        'count': 6, 'max_iterations': 0})
    atomic_json(out/'config.json', config.model_dump(mode='json'))
    pipe = Pipeline(config)
    run_record = {'run_id': pipe.run_id, 'implementation_signature': current_signature,
                  'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'items': []}
    stopped = None
    try:
        for item in frozen['items']:
            case = out/'cases'/item['key']; case.mkdir(parents=True, exist_ok=True)
            result_file = case/'result.json'
            if result_file.exists():
                cached = read(result_file)
                if cached.get('review') and cached.get('image') and file_hash(cached['image']) == cached['image_hash']:
                    run_record['items'].append({'key': item['key'], 'resumed_result': True}); continue
                # A no-review interruption may reuse render/compose caches, but never inherit old scores.
            asyncio.current_task().set_name(item['key'])
            stage = 'load_frozen_inputs'; t = time.monotonic(); timing = {}
            result = {k: item[k] for k in ('key', 'asset_id', 'group', 'baseline', 'baseline_hash')}
            result.update(status='failed', image=None, implementation_signature=current_signature)
            before_counts = dict(pipe.renders.totals); before_calls = len(pipe.vision.calls)
            try:
                shutil.copy2(item['baseline'], case/'before.png')
                result['baseline'] = str(case/'before.png')
                candidate = copy.deepcopy(item['candidate']); params = copy.deepcopy(item['params'])
                asset = read(pipe.cache/'assets'/item['asset_signature']/'result-strict.json')
                if file_hash(asset['mesh']) != frozen['source_models'][item['asset_id']]:
                    raise ValueError('source asset changed')
                asset['physical_size'] = candidate['scale']['physical_size']
                # Frozen scene evidence must already exist; do not pay for a new scene or reselect geometry.
                previous_offline = pipe.vision.offline; pipe.vision.offline = True
                try:
                    scene = await pipe.scenes.prepare(item['background'])
                finally:
                    pipe.vision.offline = previous_offline
                if scene['key'] != item['scene_signature']:
                    raise ValueError('frozen scene signature changed')
                old_folder, old_meta = saved_render(pipe.cache, item['baseline_entry'])
                timing['load_seconds'] = time.monotonic()-t
                atomic_json(case/'candidate.json', candidate)
                stage = 'render'; phase = time.monotonic()
                folder, meta, environment, samples, hit = await pipe.render(asset, scene, candidate, params, item['resolution'])
                timing['render_seconds'] = time.monotonic()-phase
                raw_alpha = raw_alpha_audit(item, folder, meta, old_folder, old_meta)
                # Shader/adaptive sampling must not move geometry or occlusion in this comparison.
                # Independently audit the new raw alpha above before reusing the frozen geometry mask.
                candidate['fixed_alpha_dir'] = str(old_folder)
                candidate['fixed_alpha_window'] = object_window(old_meta)
                atmosphere = scene.get('atmosphere', {})
                atomic_json(case/'scene-evidence.json', {'scene_key': scene['key'], 'background': item['background'],
                    'atmosphere_summary': atmosphere.get('summary', {}),
                    'visual_evidence': atmosphere.get('summary', {}).get('visual_evidence'),
                    'local_context': environment.get('local_context'),
                    'local_atmospheric_context': environment.get('atmospheric_context')})
                atomic_json(case/'render.json', {'folder': str(folder), 'metadata': meta, 'resolution': item['resolution']})
                stage = 'compose'; phase = time.monotonic()
                comp = compose(scene, candidate, folder, meta, environment, samples, params, item['seed'],
                               case/'after.png', capture_local=True)
                layers = comp.pop('local_layers')
                invariants, metrics = check_invariants(item, comp, meta, old_meta, scene, candidate, layers)
                invariants['raw_render_alpha'] = raw_alpha
                if raw_alpha['status'] != 'pass': invariants['status'] = 'fail'
                metrics['local_reference'] = pixel_stats(np.asarray(scene['frame']), samples)
                metrics['local_reference']['basis'] = environment.get('local_context', {})
                result.update(image=comp['image'], image_hash=file_hash(comp['image']), checks=comp['checks'],
                    composition=comp, photo=comp['photo'], params=params, invariants=invariants, metrics=metrics,
                    air_before=air_summary(item['baseline_entry']['photo']), air_after=air_summary(comp['photo']),
                    scene_evidence=str(case/'scene-evidence.json'), render_cache_hit=hit,
                    baseline_review=item['baseline_entry'].get('review'))
                hard_failures = [key for key, value in comp['checks'].items() if value['status'] == 'fail']
                if invariants['status'] != 'pass': hard_failures.append('fixed_geometry_invariants')
                result['program_failures'] = hard_failures
                timing['compose_seconds'] = time.monotonic()-phase
                atomic_json(result_file, result)
                stage = 'final_image_review'; phase = time.monotonic()
                observation, review = await inspect_frame(pipe.vision, asset, comp, result['image_hash'])
                timing['review_seconds'] = time.monotonic()-phase
                vetoes = visual_veto(review)
                hard_failures = sorted(set(hard_failures + [key for key, value in comp['checks'].items() if value['status'] == 'fail']))
                result.update(visual_evidence=observation.model_dump(mode='json'), review=review.model_dump(mode='json'),
                    physical_visual_veto=vetoes, program_failures=hard_failures,
                    photography_pass=review.photo_passes(), visual_pass=review.passes(),
                    status='failed' if hard_failures or vetoes else 'review',
                    deliverable=not bool(hard_failures or vetoes),
                    stop_reason='program_failure' if hard_failures else 'visible_physical_defect' if vetoes else
                        'fixed_one_pass_test_complete; scale_and_contact_evidence_remain_estimated')
                if result['deliverable']:
                    delivered = out/'review'/item['group']/f"{int(item['asset_id']):02d}.png"
                    delivered.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(comp['image'], delivered)
                    result['deliverable_image'] = str(delivered)
            except Exception as error:
                result.update(status='failed' if not result.get('image') or result.get('program_failures') else 'review', deliverable=False,
                    stop_reason=f'{stage}_unavailable', error={'type': type(error).__name__, 'message': str(error)[:500]})
                if isinstance(error, ProviderUnavailable): stopped = 'provider_unavailable'
            finally:
                timing['total_seconds'] = time.monotonic()-t
                result['timing'] = timing
                new_calls = pipe.vision.calls[before_calls:]
                result['actual_requests_this_attempt'] = sum(c.get('state') != 'not_sent' for c in new_calls)
                result['request_cost_this_attempt'] = estimate(new_calls, pipe.vision.usage()['tariff'])
                result['render_counts_this_attempt'] = {k: v-before_counts.get(k, 0) for k, v in pipe.renders.totals.items()}
                atomic_json(result_file, result)
                run_record['items'].append({'key': item['key'], 'status': result['status'], 'timing': timing,
                    'render_counts': result['render_counts_this_attempt'], 'requests': result['actual_requests_this_attempt']})
                print(json.dumps({'key': item['key'], 'status': result['status'], 'reason': result.get('stop_reason'),
                                  'seconds': timing['total_seconds']}, ensure_ascii=False), flush=True)
            if stopped: break
    finally:
        # All actual requests, including failed/format-repair attempts, are persisted by Vision.
        await pipe.vision.close()
        await pipe.blender.close()
        pipe.store.close()
        run_record.update(total_seconds=time.monotonic()-started, stopped=stopped,
            render_counts=dict(pipe.renders.totals), render_timings=pipe.renders.timings,
            blender_seconds=pipe.blender.seconds, blender_operations=pipe.blender.operation_seconds,
            api_usage=pipe.vision.usage(), source_models_unchanged=model_hashes() == frozen['source_models'])
        atomic_json(out/'runs'/f'{pipe.run_id}.json', run_record)
        summary = write_report(out)
        print(json.dumps({'report': str(out/'index.html'), 'generated': summary['generated'],
            'reviewed': summary['reviewed'], 'cost': summary['cost_total']}, ensure_ascii=False), flush=True)
    return summary


def selfcheck(source):
    """Exercise frozen replay metadata and report layout without GPU/network."""
    source = resolve(source)
    config = BatchConfig.model_validate(read(source/'config.json'))
    with tempfile.TemporaryDirectory(prefix='scrap-fusion-six-') as tmp:
        folder = Path(tmp)
        frozen = freeze(source, folder)
        assert len(frozen['items']) == 6 and len(frozen['source_models']) == 44
        assert freeze(source, folder)['signature'] == frozen['signature']
        assert all(signature(item['candidate']) == item['candidate_signature'] and
                   signature(item['params']) == item['params_signature'] for item in frozen['items'])
        binding = folder/'test-implementation.json'
        bind_implementation(binding, 'first', {'algorithm': 1})
        bind_implementation(binding, 'first', {'algorithm': 1})
        try:
            bind_implementation(binding, 'second', {'algorithm': 2})
        except ValueError:
            pass
        else:
            raise AssertionError('changed implementation allowed to resume')
        assert read(binding)['signature'] == 'first'
        rows = []
        for item in frozen['items']:
            render_folder, meta = saved_render(resolve(config.cache), item['baseline_entry'])
            audit = raw_alpha_audit(item, render_folder, meta, render_folder, meta)
            assert audit['status'] == 'pass' and audit['iou'] == 1.
            with Image.open(item['baseline']) as image, Image.open(item['background']) as background:
                assert image.size == background.size
            rows.append({**item, 'image': item['baseline'], 'status': 'layout_test_only'})
        make_overview(folder, rows)
        with Image.open(folder/'完整画面对照.png') as sheet:
            assert sheet.size == (1920, 588*6)
            with Image.open(rows[0]['baseline']) as image:
                tile = image.convert('RGB'); tile.thumbnail((960, 540), Image.Resampling.LANCZOS)
                actual = sheet.crop(((960-tile.width)//2, 48+(540-tile.height)//2,
                    (960-tile.width)//2+tile.width, 48+(540-tile.height)//2+tile.height))
                assert np.array_equal(np.asarray(actual), np.asarray(tile)), 'report cropped or altered full frame'
        black = np.zeros((3, 4, 3), np.uint8)
        stats = pixel_stats(black, np.ones((3, 4), bool))
        assert stats['pixels'] == 12 and stats['luma_median_255'] == 0 and stats['lab_chroma_median'] == 0
        assert pixel_stats(black, np.zeros((3, 4), bool))['status'] == 'unknown'
        summary = write_report(folder)
        assert summary['planned'] == 6 and summary['generated'] == summary['accepted_display'] == 0
        assert summary['cost_total']['actual_requests'] == 0
    return {'passed': True, 'items': 6, 'source_models': 44, 'new_requests': 0, 'new_renders': 0,
        'checks': ['frozen_inputs_repeatable', 'fixed_candidate_and_parameters', 'implementation_resume_guard',
                   'cached_alpha_self_iou', 'full_canvas_dimensions', 'uncropped_report_tiles',
                   'visible_pixel_metrics', 'all_six_retained_when_not_run']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'report', 'selfcheck'))
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--out', default=OUTPUT)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.source, args.out)
        print(json.dumps({'frozen': len(result['items']), 'signature': result['signature']}, ensure_ascii=False))
    elif args.command == 'report':
        print(json.dumps(write_report(resolve(args.out))['cost_total'], ensure_ascii=False))
    elif args.command == 'selfcheck':
        print(json.dumps(selfcheck(args.source), ensure_ascii=False))
    else:
        asyncio.run(run(args.source, args.out))


if __name__ == '__main__':
    main()
