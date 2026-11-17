"""Offline whole-model reference-material comparisons; no API or scene placement.

Bindings identify exact catalog mesh hashes. Historical renderable poses are
reused when available; otherwise an inspection pose is derived from cached mesh
geometry. Such an inspection pose never constitutes a physical placement pass.
"""
import argparse
import asyncio
import copy
import html
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
from PIL import Image, ImageDraw

from display_batch.common import atomic_json, file_hash, load_catalog, resolve, signature
from display_batch.generated_material import load_pack
from display_batch.physical_material import prepare_finish
from display_batch.asset_profiles import load_profile
from display_batch.geometry import pose_points, project
from display_batch.rendering import RenderCache
from display_batch.worker import Blender
from diagnose_realism_assets import (PACK, SOURCE, coverage_stats, exr,
    export_candidate, inspection_camera, neutral_light)


VERSION = 'reference-material-diagnostics-v1'
EXCLUDED = {'5', '6', '7'}
VARIANTS = (('source-left', 'source', False, 0),
            ('new-left', 'physical', False, 0),
            ('new-right', 'physical', True, 0),
            ('new-turned', 'physical', False, 55))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def binding_rows(library, catalog):
    """Reject ambiguous/stale mappings before asset preparation or rendering."""
    if library.get('version') != 'reference-material-library-v1':
        raise ValueError('unsupported reference library version')
    seen = set()
    rows = []
    for mesh_hash, entry in library.get('assets', {}).items():
        aid = str(entry['asset_id'])
        if aid in EXCLUDED:
            raise ValueError('excluded tank asset in reference library: ' + aid)
        if aid in seen:
            raise ValueError('duplicate reference asset: ' + aid)
        if aid not in catalog or catalog[aid]['mesh_hash'] != mesh_hash:
            raise ValueError('reference binding does not match current catalog mesh: ' + aid)
        evidence = entry.get('evidence', {})
        if evidence.get('mapping_status') == 'class_style_only':
            raise ValueError('class style reference cannot become identity binding: ' + aid)
        if 'candidate_asset_ids' in evidence and aid not in evidence['candidate_asset_ids']:
            raise ValueError('asset is only a style target, not an audited identity: ' + aid)
        seen.add(aid)
        rows.append((aid, mesh_hash, entry))
    if not rows:
        raise ValueError('reference library contains no active identity bindings')
    return sorted(rows, key=lambda row: (int(row[0].split('-')[0]), row[0]))


def selected_bindings(library, catalog, asset_ids=None):
    rows = binding_rows(library, catalog)
    if not asset_ids:
        return rows
    ids = {part.strip() for token in asset_ids for part in token.split(',') if part.strip()}
    unknown = ids - {row[0] for row in rows}
    if unknown:
        raise ValueError('requested asset lacks an identity binding: ' + ','.join(sorted(unknown)))
    return [row for row in rows if row[0] in ids]


def inspection_pose(vertices):
    """Orient the principal axis across the image without claiming stability."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('invalid cached inspection geometry')
    centered = points - points.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    axis = basis[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    up = np.eye(3)[int(np.argmin(np.abs(axis)))]
    side = np.cross(up, axis)
    side /= np.linalg.norm(side)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([axis, side, np.cross(axis, side)])
    transformed = points @ matrix[:3, :3].T
    matrix[:3, 3] = [-float(transformed[:, 0].mean()),
                     -float(transformed[:, 1].mean()),
                     -float(transformed[:, 2].min())]
    return {'id': 'asset-inspection-only', 'matrix': matrix.tolist(), 'yaw': 20.,
            'elevation': 23., 'plane_normal': [0, 0, 1],
            'inspection_only': True, 'physics_status': 'not_evaluated'}


def fitted_camera(asset, candidate):
    camera = inspection_camera(asset, candidate)
    vertices = np.load(asset['geometry'])['vertices']
    points, _ = pose_points(vertices, candidate['matrix'], candidate['yaw'])
    uv = project(points, camera)
    max_extent = float(np.max(np.abs(uv - .5)))
    if max_extent > .46:
        half = math.tan(math.radians(camera['fov'] / 2)) * max_extent / .46
        camera['fov'] = math.degrees(2 * math.atan(half))
    return camera


def cached_asset(cache, catalog_asset, records):
    paths = [cache / 'reference-assets' / catalog_asset['mesh_hash'] / 'result.json']
    for row in records:
        if row.get('asset_signature'):
            folder = cache / 'assets' / row['asset_signature']
            paths.extend((folder / 'result-strict.json', folder / 'result.json'))
    # The batch may predate the current asset revision; match only an exact hash.
    paths.extend(sorted((cache / 'assets').glob('*/result-strict.json')))
    paths.extend(sorted((cache / 'assets').glob('*/result.json')))
    seen = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        asset = read(path)
        if (str(asset.get('id', asset.get('number'))) == str(catalog_asset['id'])
                and asset.get('mesh_hash') == catalog_asset['mesh_hash']
                and Path(asset.get('geometry', '')).is_file()
                and Path(asset.get('mesh', '')).is_file()):
            if file_hash(asset['mesh']) != asset['mesh_hash']:
                raise ValueError('cached mesh content differs from recorded hash')
            return path, asset
    raise ValueError('no matching cached geometry for asset ' + str(catalog_asset['id']))


def historical_pose(source, records):
    for row in records:
        if not row.get('image') or row.get('best_iteration') is None:
            continue
        best = next((i for i in row.get('iterations', [])
                     if i['iteration'] == row['best_iteration']), None)
        path = source / 'jobs' / row['job_id'] / 'candidates.json'
        if not best or not path.is_file():
            continue
        candidates = read(path).get('candidates', [])
        candidate = next((c for c in candidates if c['id'] == best['params']['candidate']), None)
        if candidate is not None:
            # A receiver is irrelevant to object-only diagnostics; retain pose,
            # never let historical support or imagery masquerade as validation.
            candidate = copy.deepcopy(candidate)
            candidate.pop('receiver', None)
            candidate.pop('camera', None)
            return candidate, row['job_id']
    return None, None


def physical_length(asset, candidate):
    size = candidate.get('base_pose', {}).get('length_m')
    if size:
        return float(size), 'historical_fixed_physical_length_estimate'
    mm = candidate.get('scale', {}).get('physical_size', {}).get('nominal_mm')
    if mm:
        return float(mm) / 1000., 'historical_nominal_length_estimate'
    interval = asset.get('semantic', {}).get('major_mm')
    if interval and len(interval) == 2 and min(interval) > 0:
        return math.sqrt(float(interval[0]) * float(interval[1])) / 1000., 'cached_category_interval_geometric_midpoint_estimate'
    if asset.get('diagnostic_length_m'):
        return float(asset['diagnostic_length_m']), 'source_GLTF_coordinate_envelope_unverified_estimate_not_scene_scale'
    raise ValueError('no physical texture-scale estimate for asset ' + asset['id'])


async def prepare_missing_geometry(args):
    """Create canonical geometry only when absent, with zero API/physics/render.

    glTF coordinates use metres by convention, but a modeller may have used a
    working envelope. The span is retained only as an explicitly unverified
    texture-scale estimate, never a claimed real-world dimension.
    """
    from display_batch.reference_material import load_library
    import trimesh
    cache = resolve(args.cache)
    catalog = load_catalog()
    selected = selected_bindings(load_library(resolve(args.reference_manifest)), catalog, args.asset_ids)
    worker = None
    prepared = []
    started = time.monotonic()
    try:
        for aid, mesh_hash, _ in selected:
            asset = catalog[aid]
            try:
                cached_asset(cache, asset, [])
                continue
            except ValueError as exc:
                if not str(exc).startswith('no matching cached geometry'):
                    raise
            folder = cache / 'reference-assets' / mesh_hash
            folder.mkdir(parents=True, exist_ok=True)
            if worker is None:
                worker = Blender(resolve(args.output) / 'geometry-worker', 'CPU')
            geometry = folder / 'geometry.npz'
            before = time.monotonic()
            inspection = await worker.ask('export', {'mesh': asset['mesh'], 'geometry': str(geometry)}, timeout=180)
            scene = trimesh.load(asset['mesh'], force='scene', process=False)
            raw_span = float(np.max(np.diff(np.asarray(scene.bounds), axis=0)))
            if not math.isfinite(raw_span) or raw_span <= 0:
                raise ValueError('invalid source coordinate envelope')
            prepared_asset = {**asset, 'id': aid, 'number': aid,
                              'geometry': str(geometry), 'inspection': inspection,
                              'semantic': {'material_classes': [], 'major_mm': [],
                                           'assembly': 'unknown', 'basis': 'no_new_AI_classification'},
                              'poses': [], 'diagnostic_length_m': raw_span,
                              'signature': signature('reference-inspection-only', mesh_hash, file_hash(geometry)),
                              'scene_physics': 'not_evaluated', 'diagnostic_geometry_only': True,
                              'preparation_seconds': round(time.monotonic()-before, 3)}
            atomic_json(folder / 'result.json', prepared_asset)
            prepared.append({'asset_id': aid, 'mesh_hash': mesh_hash, 'geometry': str(geometry),
                             'seconds': prepared_asset['preparation_seconds'], 'renders': 0, 'api_calls': 0,
                             'size_basis': 'source_GLTF_coordinate_envelope_unverified_estimate'})
    finally:
        if worker is not None:
            await worker.close()
    return {'prepared': prepared, 'wall_seconds': round(time.monotonic()-started, 3),
            'worker_seconds': round(worker.seconds, 3) if worker else 0.,
            'renders': 0, 'physics_simulations': 0, 'api_calls': 0}


def freeze(args):
    from display_batch.reference_material import load_library, lookup_reference
    source, cache = resolve(args.source), resolve(args.cache)
    surface_dust = getattr(args, 'surface_dust', .5)
    if not 0 <= surface_dust <= .5:
        raise ValueError('Surface powder must be within 0–0.5')
    library_path = resolve(args.reference_manifest)
    library = load_library(library_path)
    catalog = load_catalog()
    selected = selected_bindings(library, catalog, args.asset_ids)
    variants = [row for row in VARIANTS if row[0] in args.variants]
    if not {'source-left', 'new-left'} <= set(args.variants):
        raise ValueError('diagnostic must include source-left and new-left for a controlled comparison')
    records_path = source / 'results.jsonl'
    records = sorted((json.loads(line) for line in records_path.read_text(encoding='utf8').splitlines()
                      if line.strip()), key=lambda row: row['job_id']) if records_path.is_file() else []
    pack = load_pack(resolve(args.pack))
    cases = []
    failures = []
    for aid, mesh_hash, binding in selected:
        try:
            asset_records = [r for r in records if str(r.get('asset_id')) == aid]
            path, asset = cached_asset(cache, catalog[aid], asset_records)
            matched_records = [r for r in asset_records if r.get('asset_signature') == asset.get('signature')]
            candidate, job = historical_pose(source, matched_records) if args.pose_mode == 'historical' else (None, None)
            if candidate is None:
                candidate = inspection_pose(np.load(asset['geometry'])['vertices'])
            candidate['camera'] = fitted_camera(asset, candidate)
            size, size_basis = physical_length(asset, candidate)
            reference = lookup_reference(library_path, mesh_hash)
            if not reference:
                raise ValueError('validated reference entry unavailable')
            finish = prepare_finish(asset, cache, pack, size,
                                    profile=load_profile(mesh_hash), reference=reference)
            cases.append({'asset_id': aid, 'name': asset['name'], 'mesh_hash': mesh_hash,
                          'asset_record': str(path), 'asset_record_hash': file_hash(path),
                          'geometry_sha256': file_hash(asset['geometry']), 'candidate': candidate,
                          'source_job': job, 'selection': 'first_historical_image_pose' if job else 'cached_geometry_inspection_pose',
                          'reference_binding': binding, 'finish': finish,
                          'size_m': size, 'size_basis': size_basis,
                          'physical_placement': 'not_evaluated_in_model_diagnostic'})
        except (ValueError, FileNotFoundError, KeyError) as exc:
            failures.append({'asset_id': aid, 'stage': 'freeze', 'reason': str(exc)})
    return {'version': VERSION, 'script_sha256': file_hash(__file__),
            'source': str(source), 'source_results_sha256': file_hash(records_path) if records_path.is_file() else None,
            'reference_manifest': str(library_path), 'reference_manifest_sha256': file_hash(library_path),
            'pack_signature': pack['signature'], 'resolution': args.resolution, 'pose_mode': args.pose_mode,
            'surface_dust': surface_dust,
            'catalog_mesh_hashes': {aid: row['mesh_hash'] for aid, row in catalog.items()},
            'selected_asset_ids': [aid for aid, _, _ in selected],
            'cases': cases, 'failures': failures, 'variants': [row[0] for row in variants],
            'planned_object_renders': len(cases) * len(variants),
            'api_calls': 0, 'scope': 'model_material_diagnostic_not_geometry_or_scene_acceptance'}


def comparison_board(out, row):
    sizes = []
    for item in row['renders']:
        with Image.open(out / item['image']) as image:
            sizes.append(image.width)
    side = max(sizes)
    canvas = Image.new('RGB', (side * 2, (side + 42) * math.ceil(len(row['renders'])/2)), (112, 112, 112))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(row['renders']):
        x, y = index % 2 * side, index // 2 * (side + 42)
        with Image.open(out / item['image']) as original:
            image = original.convert('RGBA')
            image.thumbnail((side, side), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (side-image.width)//2, y+42+(side-image.height)//2), image.getchannel('A'))
        draw.text((x+14, y+14), row['asset_id'] + ' / ' + item['tag'], fill='white')
    path = out / row['asset_id'] / 'complete-comparison.png'
    canvas.save(path)
    return path.relative_to(out).as_posix()


def pixel_comparison(first, second):
    with Image.open(first) as im:
        a = np.asarray(im.convert('RGBA'), dtype=np.int16)
    with Image.open(second) as im:
        b = np.asarray(im.convert('RGBA'), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError('comparison canvas differs')
    opaque = (a[..., 3] > 250) & (b[..., 3] > 250)
    foreground = (a[..., 3] > 127) | (b[..., 3] > 127)
    delta = np.abs(a-b)
    return {'alpha_identical': bool(np.array_equal(a[..., 3], b[..., 3])),
            'alpha_iou': float(((a[..., 3] > 127) & (b[..., 3] > 127)).sum()/max(1, foreground.sum())),
            'max_abs_rgba_255': int(delta.max()), 'changed_pixels': int(np.any(delta, axis=2).sum()),
            'changed_pixel_fraction': float(np.any(delta, axis=2).mean()),
            'core_rgb_mae_255': float(delta[..., :3][opaque].mean()) if opaque.any() else None,
            'basis': 'same_camera_pose_lighting_full_canvas_actual_PNG_not_visual_quality_score'}


def write_report(out, frozen, results, summary):
    pieces = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>参考材质模型诊断</title>',
              '<style>body{font:16px sans-serif;margin:30px;background:#ddd;color:#222}img{max-width:100%;height:auto;background:#777}.plates{display:grid;grid-template-columns:repeat(2,minmax(250px,1fr));gap:15px}figure{margin:0}pre{white-space:pre-wrap}a{color:#075d66}</style>',
              '<h1>参考材质：完整模型对照</h1><p>原材质、新材质固定几何与同一视角；另检查换光和转向。无现场薄雾、调色、模糊或合成。参考六视图不作为几何、型号或标牌重建真值。检查视角不证明现场承托成立。</p>',
              '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>']
    for row in results:
        pieces += ['<h2>' + html.escape(row['asset_id'] + ' ' + row['name']) + '</h2>',
                   '<a href="' + row['comparison'] + '">完整对照板</a> · <a href="' + row['asset_id'] + '/candidate.blend">候选 Blender 文件</a><div class="plates">']
        for item in row['renders']:
            uri = html.escape(item['image'], quote=True)
            pieces.append('<figure><figcaption>' + html.escape(item['tag']) + '</figcaption><a href="'+uri+'"><img src="'+uri+'"></a></figure>')
        pieces += ['</div><pre>' + html.escape(json.dumps(row['checks'], ensure_ascii=False, indent=2)) + '</pre>']
    pieces += ['<h2>全部失败项与限制</h2><pre>' + html.escape(json.dumps(
        {'failures': summary.get('failures', frozen['failures']), 'known_limits': summary['known_limits']}, ensure_ascii=False, indent=2)) + '</pre></html>']
    (out / 'index.html').write_text(''.join(pieces), encoding='utf8')


async def run(args):
    out, cache = resolve(args.output), resolve(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    preparation = await prepare_missing_geometry(args)
    if preparation['prepared']:
        # Preserve a cold-preparation ledger when later freeze/resume reuses it.
        atomic_json(out / 'geometry-preparation.json', preparation)
    frozen = freeze(args)
    marker = out / 'frozen.json'
    if marker.is_file() and signature(read(marker)) != signature(frozen):
        raise ValueError('diagnostic inputs changed; use a fresh output directory')
    atomic_json(marker, frozen)
    if args.freeze_only:
        print(json.dumps({'frozen': str(marker), 'assets': len(frozen['cases']),
                          'failures': frozen['failures'], 'planned_object_renders': frozen['planned_object_renders'],
                          'api_calls': 0}, ensure_ascii=False))
        return
    worker = Blender(out / 'worker', args.device)
    renderer = RenderCache(cache, worker)
    results = []
    failures = list(frozen['failures'])
    started = time.monotonic()
    export_seconds = 0.
    try:
        for case in frozen['cases']:
            asset = read(case['asset_record'])
            if file_hash(asset['mesh']) != case['mesh_hash']:
                raise ValueError('source mesh changed after freeze')
            renders, paths = [], {}
            try:
                for tag, mode, right, turn in VARIANTS:
                    if tag not in frozen['variants']:
                        continue
                    candidate = copy.deepcopy(case['candidate'])
                    candidate['yaw'] = float(candidate['yaw']) + turn
                    candidate['camera'] = fitted_camera(asset, candidate)
                    params = {'material': 'original', 'light': 'scene', 'dust': 'auto',
                              'surface_dust': frozen['surface_dust'], 'integration_mode': 'coherent', 'render_detail': 'standard',
                              'asset_diagnostic': mode, 'asset_refinement': 'component_v1',
                              'oxide_gain': 1., 'scratch_gain': 1., 'relief_gain': 1., 'roughness_shift': 0.}
                    finish = case['finish'] if mode == 'physical' else None
                    folder, meta, hit = await renderer.get(asset, candidate, params, neutral_light(right),
                        (3840, 2160), args.resolution, appearance='local', full_frame=True,
                        object_only=True, generated_finish=finish)
                    image = out / case['asset_id'] / (tag + '.png')
                    image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(folder / 'object.png', image)
                    with Image.open(image) as im:
                        if im.size != (args.resolution, args.resolution):
                            raise ValueError('unexpected whole-model canvas dimensions')
                        alpha = np.asarray(im.getchannel('A'), dtype=float) / 255.
                    if not np.any(alpha > .5):
                        raise ValueError('empty diagnostic model')
                    paths[tag] = folder
                    material_checks = coverage_stats(folder, alpha)
                    if (domain_path := folder / 'component-domains0001.exr').is_file():
                        core = alpha > .98
                        domain = exr(domain_path)[core]
                        wear = exr(folder / 'generated-finish0001.exr')[core]
                        material_checks['component_domains'] = dict(zip(('rubber', 'plastic', 'source_print'),
                                                                        domain.mean(axis=0).tolist()))
                        material_checks['mean_layer_weights'] = dict(zip(('oxide', 'exposed_steel', 'powder'),
                                                                         wear.mean(axis=0).tolist()))
                    renders.append({'tag': tag, 'image': image.relative_to(out).as_posix(),
                        'sha256': file_hash(image), 'cache_hit': hit, 'render_cache': meta['render_cache'],
                        'render_timings': meta.get('render_timings'),
                        'source_regions': material_checks,
                        'material_readback': meta.get('material_readback'),
                        'lighting_readback': meta.get('lighting_readback'),
                        'material_metadata': meta.get('generated_finish'),
                        'camera': meta['camera'], 'geometry_signature': signature(case['mesh_hash'], candidate['matrix'], candidate['yaw'])})
                    if tag == 'new-left':
                        export_started = time.monotonic()
                        export_candidate(out, asset, candidate, params, finish, args.resolution)
                        export_seconds += time.monotonic() - export_started
                    print(json.dumps({'asset': case['asset_id'], 'tag': tag, 'cache_hit': hit}, ensure_ascii=False), flush=True)
                comparison = pixel_comparison(out / case['asset_id'] / 'source-left.png',
                                              out / case['asset_id'] / 'new-left.png')
                with Image.open(out / case['asset_id'] / 'new-left.png') as im:
                    core = np.asarray(im.getchannel('A')) > 250
                maps_mae = spec_mae = None
                if 'new-right' in paths:
                    map_images = [exr(paths[tag] / 'generated-finish0001.exr') for tag in ('new-left', 'new-right')]
                    maps_mae = float(np.abs(map_images[0][core] - map_images[1][core]).mean()) if core.any() else None
                    spec = [exr(paths[tag] / 'specular0001.exr') for tag in ('new-left', 'new-right')]
                    spec_mae = float(np.abs(spec[0][core] - spec[1][core]).mean()) if core.any() else None
                checks = {'source_to_new': comparison, 'source_mesh_unchanged': file_hash(asset['mesh']) == case['mesh_hash'],
                          'same_camera_source_new': renders[0]['camera'] == renders[1]['camera'],
                          'texture_map_mae_under_changed_light': maps_mae,
                          'specular_mae_under_changed_light': spec_mae,
                          'relighting_check': 'measured' if 'new-right' in paths else 'not_requested_in_pilot',
                          'geometry_alpha_pass': comparison['alpha_iou'] >= .995,
                          'visual_realism': 'unreviewed', 'scene_contact': 'not_evaluated',
                          'physical_size_basis': case['size_basis'],
                          'protected_control': '13_plastic' if case['asset_id'] == '13' else None}
                row = {'asset_id': case['asset_id'], 'name': case['name'], 'renders': renders, 'checks': checks,
                       'reference_binding': case['reference_binding'], 'material_audit': case['finish']['material_audit'],
                       'size_m': case['size_m'], 'candidate_check': read(out / asset['id'] / 'candidate-check.json')}
                row['comparison'] = comparison_board(out, row)
                results.append(row)
                atomic_json(out / 'results.json', results)
            except Exception as exc:
                failures.append({'asset_id': case['asset_id'], 'stage': 'render_or_validate',
                                 'reason': str(exc), 'completed_plates': [r['image'] for r in renders]})
                atomic_json(out / 'failures.json', failures)
    finally:
        await worker.close()
    # Reload catalog to verify every original model, including unrendered ones.
    current = load_catalog()
    unchanged = {aid: current[aid]['mesh_hash'] == sha for aid, sha in frozen['catalog_mesh_hashes'].items()}
    references = [case['finish']['reference'] for case in frozen['cases']]
    known_limits = [
        'Albedo separation and roughness/micro-relief are estimates, not measured material properties.',
        'Unknown material regions retain their source appearance; report coverage instead of forcing rust.',
        'Inspection pose and cached category dimensions do not establish scene contact or true scale.',
        'Numeric differences and highlight response do not prove photorealistic quality.',
    ]
    if any('packages' in reference for reference in references):
        known_limits.insert(0, 'Library surfaces provide donor textures. Exact alloy, coating and wear are unverified.')
    if any('packages' not in reference for reference in references):
        known_limits.insert(0, 'AI six views provide appearance motifs, not calibrated PBR or hidden geometry truth.')
    summary = {'version': VERSION, 'state': 'complete' if not failures else 'completed_with_failures',
               'selected_assets': len(frozen['selected_asset_ids']), 'completed_assets': len(results),
               'plates': sum(len(row['renders']) for row in results), 'failures': failures,
               'wall_seconds': round(time.monotonic() - started, 3), 'worker_seconds': round(worker.seconds, 3),
               'candidate_export_seconds': round(export_seconds, 3), 'render_counts': dict(renderer.totals),
               'geometry_preparation_this_run': preparation,
               'render_timings': dict(renderer.timings), 'api_calls': 0, 'tokens': 0, 'api_cost': 0,
               'local_compute_cost': 'unknown', 'catalog_hash_checks': unchanged,
               'visual_status': 'unreviewed', 'production_default_changed': False,
               'known_limits': known_limits}
    atomic_json(out / 'summary.json', summary)
    write_report(out, frozen, results, summary)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--cache', default='work/display-batch-cache')
    parser.add_argument('--pack', default=PACK)
    parser.add_argument('--reference-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resolution', type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument('--device', choices=('OPTIX', 'CPU'), default='OPTIX')
    parser.add_argument('--surface-dust', type=float, default=.5, help='Added surface powder strength, 0–0.5')
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--pose-mode', choices=('historical', 'inspection'), default='historical',
                        help='inspection uses a fixed principal-axis side view; neither mode validates scene physics')
    parser.add_argument('--asset-ids', nargs='+', help='explicit subset of audited identity assets; comma-separated IDs also accepted')
    parser.add_argument('--variants', nargs='+', choices=[row[0] for row in VARIANTS],
                        default=[row[0] for row in VARIANTS])
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
