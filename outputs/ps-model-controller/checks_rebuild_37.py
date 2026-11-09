"""Independent checks for the item 37 oil-collector/filter candidate.

This script verifies only artefacts that were actually produced.  Source IoU
values are uncalibrated diagnostics and are never treated as acceptance.  The
six-view reference is missing for this item, so no reference IoU is claimed.
Browser runtime interaction is marked "not executed"; the standalone page is
checked statically and its inline JavaScript syntax is checked with Node.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 37


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def glb_json(path):
    data = Path(path).read_bytes()
    if data[:4] != b'glTF':
        raise ValueError('not a GLB file')
    offset = 12
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], 'little')
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        if kind == b'JSON':
            return json.loads(body.decode('utf8').rstrip('\x00 '))
        offset += 8 + length
    raise ValueError('GLB has no JSON chunk')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    failures = []
    passed = []
    pending = []
    not_executed = []

    def require(condition, message):
        if condition:
            passed.append(message)
        else:
            failures.append(message)
        return bool(condition)

    result = json.loads((out / 'result.json').read_text(encoding='utf8'))
    worker = json.loads((out / 'worker-result.json').read_text(encoding='utf8'))
    textures = json.loads((out / 'texture-checks.json').read_text(encoding='utf8'))
    hashes = json.loads((out / 'hashes.json').read_text(encoding='utf8'))

    require(result.get('number') == NUMBER, 'result number is 37')
    require(result.get('manual_only') is True and
            result.get('enters_automatic_placement') is False and
            result.get('replaces_library_mesh') is False,
            'candidate is manual_only, does not replace the library mesh and '
            'does not enter automatic placement')
    require(result.get('reference', {}).get('role') == 'missing',
            'six-view reference role is missing (no reliable reference)')
    require(result.get('reference', {}).get('file') is None,
            'no similar model is presented as the reference')

    # Inputs and output hashes.
    for key, record in hashes.get('inputs', {}).items():
        path = ROOT / record['path']
        if not require(path.is_file(), f'input file exists: {key}'):
            continue
        require(digest(path) == record['sha256'], f'input hash unchanged: {key}')
    for name, record in hashes.get('outputs', {}).items():
        path = ROOT / name
        if not require(path.is_file(), f'output file exists: {name}'):
            continue
        require(digest(path) == record['sha256'], f'output hash matches: {name}')
    require(result.get('source', {}).get('original_sha256') ==
            hashes['inputs']['original']['sha256'],
            'result original hash equals hashes.json')

    # Core files.
    for name in ('mesh.glb', 'model.blend', 'review-37.html', 'report.md',
                 '六视图-新旧对照.jpg', '关键局部-新旧对照.jpg',
                 '结构开口-隔离检查.jpg', '材质-同光照对照.jpg',
                 '表面反射-变光对照.jpg', '原照-新候选-参考.jpg',
                 'PBR贴图-对照.jpg', '转台-24方向.jpg',
                 'source-measurements.json', 'old-model-audit.json'):
        require((out / name).is_file(), f'required artefact exists: {name}')
    for name in ('black_plastic_base.png', 'black_plastic_mr.png',
                 'black_plastic_normal.png', 'black_matte_base.png',
                 'filter_mesh_base.png', 'filter_mesh_mr.png',
                 'filter_mesh_normal.png', 'smoke_bowl_base.png',
                 'smoke_bowl_mr.png', 'smoke_bowl_normal.png',
                 'brass_base.png', 'brass_mr.png', 'brass_normal.png'):
        require((out / 'textures' / name).is_file(),
                f'PBR texture exists: {name}')

    # Texture values.
    require(textures.get('passed') is True, 'PBR texture numeric checks pass')
    black = textures.get('black_plastic', {})
    require(black.get('mean_rgb', [255])[0] < 70,
            'black plastic base is dark, not over-bright')
    require(black.get('grain_std', 0) > 2.0,
            'black plastic has visible micro-grain variation')
    require(black.get('metallic_mean', 1) <= .02,
            'black plastic is dielectric (not mislabelled metal)')
    require(black.get('roughness_mean', 0) >= .30 and
            black.get('roughness_mean', 1) <= .62,
            'black plastic roughness is plausible')
    matte = textures.get('black_matte', {})
    require(matte.get('roughness_mean', 0) >= .60,
            'matte recess texture is matte')
    mesh_tex = textures.get('filter_mesh', {})
    require(mesh_tex.get('mean_rgb', [0])[0] > 120,
            'filter mesh is light-coloured')
    require(mesh_tex.get('weave_contrast', 0) > 35,
            'filter mesh texture has visible weave contrast')
    require(mesh_tex.get('metallic_mean', 1) <= .25,
            'filter mesh is dielectric-dominant (appearance approximation)')
    bowl = textures.get('smoke_bowl', {})
    require(bowl.get('mean_rgb', [255])[0] < 110,
            'bowl base is smoky/dark')
    require(bowl.get('roughness_mean', 1) < .12,
            'bowl texture roughness is very low (glossy plastic)')
    require(bowl.get('metallic_mean', 1) <= .02,
            'bowl is dielectric')
    brass = textures.get('brass', {})
    require(brass.get('metallic_mean', 0) > .90,
            'brass metallic mean is high')
    require(brass.get('roughness_mean', 1) < .36,
            'brass roughness is plausible')
    require(brass.get('mean_rgb', [0, 0, 0])[0] >
            brass.get('mean_rgb', [0, 0, 0])[1] >
            brass.get('mean_rgb', [0, 0, 0])[2],
            'brass base colour is warm (R > G > B)')

    # GLB geometry.
    scene = trimesh.load(out / 'mesh.glb', force='scene')
    geoms = list(scene.geometry.values())
    require(len(geoms) >= 40, f'GLB has at least 40 geometries ({len(geoms)})')
    # Node transforms carry most of the part offsets.  scene.bounds applies
    # them; concatenating raw geometry vertices would only measure local space.
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    size = np.asarray(scene.extents, dtype=np.float64)
    combined = (scene.to_geometry() if hasattr(scene, 'to_geometry')
                else scene.dump(concatenate=True))
    all_vertices = np.asarray(combined.vertices, dtype=np.float64)
    require(0.90 <= size[2] <= 1.10,
            f'GLB height is about 1 m ({size[2]:.4f} m)')
    require(size[2] >= 1.8 * size[1],
            'height dominates the bracket/cap width')
    require(size[2] >= 1.8 * size[0],
            'height dominates the front-back depth')
    require(np.isfinite(all_vertices).all(), 'GLB vertices are finite')
    require(all(not g.is_empty for g in geoms), 'GLB geometries are non-empty')
    watertight = {}
    raw_watertight = {}
    for name, geometry in scene.geometry.items():
        raw_watertight[name] = bool(geometry.is_watertight)
        merged = trimesh.Trimesh(np.round(geometry.vertices, 7),
                                 geometry.faces, process=True,
                                 validate=False)
        merged.merge_vertices()
        watertight[name] = bool(merged.is_watertight)
    require(all(watertight.values()),
            'all exported GLB surfaces are watertight after merging seam vertices '
            f'({sum(watertight.values())}/{len(watertight)}; raw unmerged '
            f'{sum(raw_watertight.values())}/{len(raw_watertight)})')

    gltf = glb_json(out / 'mesh.glb')
    material_names = [m.get('name', '') for m in gltf.get('materials', [])]
    require(len(material_names) >= 5,
            f'GLB carries the expected material set ({len(material_names)})')
    require(any('黑色工程塑料' in name for name in material_names),
            'black plastic material is present')
    require(any('编织滤网' in name for name in material_names),
            'filter mesh material is present')
    require(any('半透明' in name for name in material_names),
            'translucent bowl material is present')
    require(any('黄铜' in name for name in material_names),
            'brass material is present')
    bowl_materials = [m for m in gltf.get('materials', [])
                      if '半透明' in m.get('name', '')]
    if bowl_materials:
        extension = bowl_materials[0].get('extensions', {}).get(
            'KHR_materials_transmission', {})
        require(extension.get('transmissionFactor', 0) > 0.3,
                'bowl material carries KHR_materials_transmission')
    require(len(gltf.get('images', [])) >= 5,
            f'GLB embeds PBR images ({len(gltf.get("images", []))})')

    # Worker geometry checks.
    closures = worker.get('closures', {})
    require(all(value == 0 for value in closures.values()),
            'Blender manifold check: every part has zero non-manifold edges')
    require(not worker.get('self_intersecting_components'),
            'Blender BVH check: no same-part non-adjacent face overlap')
    require(worker.get('closed_parts') == len(worker.get('parts', [])),
            'worker reports all parts closed')

    # Ray checks of the real openings and structural details.
    rays = worker.get('ray_checks', {})
    bracket = rays.get('bracket', {})
    require(bracket.get('hole_0_clear') is True and
            bracket.get('hole_1_clear') is True,
            'both stadium mounting holes are open along X')
    require(bracket.get('plate_solid') is True,
            'mounting plate is solid away from its holes')
    cap = rays.get('cap', {})
    neck = rays.get('neck', {})
    require(cap.get('solid') is True and cap.get('bead_present') is True,
            'cap is solid and carries its moulded bead')
    require(neck.get('core_solid') is True and
            neck.get('front_rib_solid') is True and
            neck.get('back_rib_present') is True,
            'neck core and front/back ribs exist')
    cage = rays.get('cage', {})
    windows = cage.get('windows', {})
    require(len(windows) == 6, 'six front cage windows were ray-tested')
    require(cage.get('all_front_windows_open') is True,
            'all six front cage windows are open (not painted grooves)')
    require(all(v.get('filter_hit') for v in windows.values()),
            'the white filter element is visible behind all six windows')
    require(cage.get('front_post_solid') is True,
            'the narrow front post is solid')
    require(cage.get('side_post_solid') is True,
            'the wide side post is solid')
    require(cage.get('top_band_solid') is True and
            cage.get('mid_band_solid') is True,
            'horizontal cage bands are solid')
    require(rays.get('filter', {}).get('core_solid') is True,
            'filter core is solid')
    require(rays.get('collar', {}).get('solid') is True,
            'collar ring is solid')
    require(rays.get('bowl', {}).get('wall_hit') is True,
            'translucent bowl wall is present')
    require(rays.get('bowl', {}).get('ribs_present') is True and
            rays.get('bowl', {}).get('rib_count') == 4,
            'four internal bowl ribs are present')
    drain = rays.get('drain', {})
    require(drain.get('neck_solid') is True and
            drain.get('knob_solid') is True,
            'drain neck and knurled knob are solid')
    require(drain.get('brass_solid') is True,
            'brass drain pin is present')
    require(drain.get('brass_blind_recess') is True and
            (drain.get('brass_recess_depth_m') or 0) > .005,
            'brass pin has a measured shallow blind recess')
    require(rays.get('slots', {}).get('present') is True,
            'four decorative slot insets are present')
    objects = rays.get('objects', {})
    require(objects.get('bracket') is True and objects.get('cap') is True and
            objects.get('filter') is True and objects.get('collar') is True and
            objects.get('bowl') is True and objects.get('brass') is True,
            'core part objects exist')

    # Rendered evidence.
    metal = result['checks']['metal']
    require(metal.get('passed') is True,
            'brass highlight moves with the main light position')
    require(metal.get('mean_abs_difference', 0) >= 1.0,
            'brass-only render changes measurably between light positions')
    require(metal.get('metal_pixels', 0) >= 800,
            'brass-only render contains enough pixels to test')
    luma = result['checks']['luma']
    for key, value in luma.items():
        require(value.get('dark_fraction', 1) < 0.35,
                f'{key} is not mostly black (dark fraction '
                f'{value.get("dark_fraction")})')
    require(luma['new-front.png']['median_luma'] > 20,
            'new front render median luma is not crushed to black')
    require(luma['new-material-3q.png']['median_luma'] > 20,
            'new material render median luma is not crushed to black')
    require(luma['new-material-3q.png']['laplacian_variance'] > 20,
            'new material render retains visible detail (absolute threshold)')

    diagnostics = result.get('diagnostics', {})
    require(diagnostics.get('reference_role') == 'missing',
            'diagnostics record the missing six-view reference')
    require('未做参考图 IoU' in diagnostics.get('reference_note', ''),
            'diagnostics state that no reference IoU was claimed')
    require(diagnostics.get('strict', {}).get('acceptance_gate') is False,
            'strict source IoU is flagged as non-acceptance diagnostic')
    require(diagnostics.get('strict', {}).get('available') is True and
            diagnostics.get('strict', {}).get('iou') is not None,
            'strict same-camera IoU was computed from the library alpha mask')
    require((diagnostics.get('new_source_view_fit') or {}).get('iou') is not None,
            'source-view foreground shape diagnostic was computed')
    require((diagnostics.get('best') or {}).get('acceptance_gate') is False,
            'best-view IoU is flagged as non-acceptance diagnostic')
    require((diagnostics.get('source_box_fit') or {}).get('iou') is not None,
            'source box-fit IoU diagnostic was computed')
    require((diagnostics.get('source_rotated_front_fit') or {}).get('iou') is not None,
            'rotated front IoU diagnostic was computed')
    require(diagnostics.get('best', {}).get('iou') is not None,
            '24-view best silhouette diagnostic was computed')

    # Six-view and detail renders.
    for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        for prefix in ('new', 'old'):
            require((out / 'renders' / f'{prefix}-{view}.png').is_file(),
                    f'six-view render exists: {prefix}-{view}')
    for name in ('bracket', 'neck', 'cap', 'cage', 'windows', 'slots',
                 'collar', 'bowl', 'drain', 'overall'):
        require((out / 'renders' / f'detail-{name}.png').is_file(),
                f'new detail render exists: {name}')
        require((out / 'renders' / f'old-detail-{name}.png').is_file(),
                f'old detail render exists: {name}')
    for name in ('isolate-bracket.png', 'isolate-cage.png',
                 'isolate-cap-collar.png', 'isolate-bowl.png',
                 'isolate-drain.png', 'isolate-brass.png'):
        require((out / 'renders' / name).is_file(),
                f'isolation render exists: {name}')
    for i in range(24):
        for prefix in ('turn-new', 'turn-old', 'turn-clay', 'sil-new', 'sil-old'):
            require((out / 'renders' / f'{prefix}-{i:02d}.png').is_file(),
                    f'turntable frame exists: {prefix}-{i:02d}')

    # Standalone page: every local image reference exists, JS syntax via node.
    html = (out / 'review-37.html').read_text(encoding='utf8')
    references = set(re.findall(r'src="([^"]+)"', html))
    references |= set(re.findall(r"src='([^']+)'", html))
    for reference in sorted(references):
        if reference.startswith(('http:', 'https:', 'data:')):
            continue
        if '<' in reference or '{' in reference:
            continue
        require((out / reference).is_file(),
                f'page asset exists: {reference}')
    script_match = re.search(r'<script>(.*?)</script>', html, re.S)
    require(script_match is not None, 'page contains an inline script')
    if script_match:
        js_path = out / 'renders' / 'page-check.js'
        js_path.write_text(script_match.group(1), encoding='utf8')
        node = subprocess.run(['node', '--check', str(js_path)],
                              capture_output=True, text=True)
        require(node.returncode == 0,
                'page JavaScript syntax passes node --check'
                + ('' if node.returncode == 0 else f': {node.stderr.strip()}'))
    require('manual_only' in html and '不能进入正式合成' in html,
            'page carries the manual_only / no formal composite notice')
    require('missing' in html and '无可靠对应参考' in html,
            'page records the missing six-view reference')
    not_executed.append('浏览器运行时交互（无已安装无头浏览器）')
    pending.append('所有 IoU 均为未标定诊断，不是验收通过。')
    pending.append('六视图参考缺失；背面、内部构造、孔深、真实尺寸与落稳待人工复核。')
    pending.append('黄铜变光只覆盖黄铜阀芯；黑色塑料/滤网/半透明杯不适用。')

    checks_record = {
        'number': NUMBER,
        'status': 'passed_with_pending_review' if not failures else 'failed',
        'passed': passed,
        'failed': failures,
        'pending': pending,
        'not_executed': not_executed,
        'metrics': {
            'glb_geometries': len(geoms),
            'glb_watertight_merged': watertight,
            'glb_watertight_raw': raw_watertight,
            'glb_bounds': bounds.tolist(),
            'glb_size': size.tolist(),
            'openings': result.get('openings'),
            'metal': metal,
            'luma': luma,
            'diagnostics': diagnostics,
        },
        'note': ('通过项只覆盖实际执行的网格、射线、贴图、渲染和静态页面检查；'
                 '未标定 IoU、缺失的六视图参考与浏览器运行时交互未被当作通过。'),
    }
    checks_path = out / 'checks-37.json'
    checks_path.write_text(json.dumps(checks_record, ensure_ascii=False, indent=2),
                           encoding='utf8')

    # Recompute hashes after checks-37.json exists.
    outputs = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'hashes.json':
            outputs[str(path.relative_to(ROOT)).replace('\\', '/')] = {
                'sha256': digest(path), 'bytes': path.stat().st_size}
    hashes['outputs'] = outputs
    (out / 'hashes.json').write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf8')

    progress = {
        'number': NUMBER,
        'status': ('candidate_ready_pending_review' if not failures
                   else 'candidate_built_checks_failed'),
        'out': str(out),
        'mesh': str(out / 'mesh.glb'),
        'blend': str(out / 'model.blend'),
        'checks': {
            'passed_count': len(passed),
            'failed_count': len(failures),
            'failed': failures,
            'pending': pending,
            'not_executed': not_executed,
            'record': str(checks_path),
        },
        'pending_review': result.get('pending_review', []),
        'next_number': 38,
    }
    (ROOT / 'work/direct-deepseek/37-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')

    print(json.dumps({'passed': len(passed), 'failed': failures,
                      'pending': pending, 'not_executed': not_executed},
                     ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
