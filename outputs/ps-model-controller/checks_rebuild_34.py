"""Independent checks for the item 34 green water-extinguisher candidate.

This script verifies only artefacts that were actually produced.  The source
IoU values are uncalibrated diagnostics and are never treated as acceptance.
Browser runtime interaction is marked "not executed"; the standalone page is
checked statically and its inline JavaScript syntax is checked with Node when
available.
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
NUMBER = 34


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

    require(result.get('number') == NUMBER, 'result number is 34')
    require(result.get('manual_only') is True and
            result.get('enters_automatic_placement') is False and
            result.get('replaces_library_mesh') is False,
            'candidate is manual_only, does not replace the library mesh and '
            'does not enter automatic placement')
    require(result.get('reference', {}).get('role') == 'visual_match_unverified',
            'six-view reference role is visual_match_unverified (uncalibrated)')

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
    for name in ('mesh.glb', 'model.blend', 'review-34.html', 'report.md',
                 '六视图-新旧对照.jpg', '关键局部-新旧对照.jpg',
                 '结构开口-隔离检查.jpg', '材质-同光照对照.jpg',
                 '表面反射-变光对照.jpg', '原照-新候选-参考.jpg',
                 'PBR贴图-对照.jpg', '转台-24方向.jpg'):
        require((out / name).is_file(), f'required artefact exists: {name}')
    for name in ('paint_green_worn_base.png', 'paint_green_worn_mr.png',
                 'paint_green_worn_normal.png', 'paint_green_clean_base.png',
                 'chrome_base.png', 'chrome_mr.png', 'chrome_normal.png',
                 'brass_mr.png', 'rubber_mr.png', 'label_front_base.png',
                 'label_back_base.png', 'gauge_face_base.png',
                 'yellow_tag_base.png', 'emboss_base.png'):
        require((out / 'textures' / name).is_file(),
                f'PBR texture exists: {name}')

    # Texture values.
    require(textures.get('passed') is True, 'PBR texture numeric checks pass')
    for key in ('paint_green_worn', 'paint_green_clean'):
        paint = textures.get(key, {})
        require(paint.get('green_dominant') is True,
                f'{key} base is green-dominant')
        require(0.0 <= paint.get('rust_fraction', 0) <= 0.14,
                f'{key} rust fraction is localised, not excessive')
        require(0.12 <= paint.get('roughness_mean', 0) <= 0.55,
                f'{key} roughness is plausible for painted metal')
        require(paint.get('metallic_mean', 1) <= 0.30,
                f'{key} is mostly dielectric paint (not bare metal)')
    chrome = textures.get('chrome', {})
    require(chrome.get('metallic_mean', 0) > 0.9,
            'chrome metallic mean is high')
    require(chrome.get('roughness_mean', 1) < 0.25,
            'chrome roughness mean is low')
    rubber = textures.get('rubber', {})
    require(rubber.get('metallic_mean', 1) < 0.05,
            'rubber metallic mean is zero')
    require(0.38 <= rubber.get('roughness_mean', 0) <= 0.72,
            'rubber roughness is matte')
    require(textures.get('label_front', {}).get('ink_fraction', 0) > 0.03,
            'front label has printed ink')
    require(textures.get('label_back', {}).get('ink_fraction', 0) > 0.02,
            'back label has printed ink')
    require(textures.get('gauge_face', {}).get('white_fraction', 0) > 0.35,
            'gauge dial has a white face')
    require(textures.get('gauge_face', {}).get('colored_fraction', 0) > 0.03,
            'gauge dial has coloured pressure zones')
    require(textures.get('yellow_tag', {}).get('passed') is True,
            'yellow tag texture has yellow face and dark border')
    require(textures.get('emboss', {}).get('passed') is True,
            'embossed-text decal has a plausible alpha coverage')

    # GLB geometry.
    scene = trimesh.load(out / 'mesh.glb', force='scene')
    geoms = list(scene.geometry.values())
    require(len(geoms) >= 40, f'GLB has at least 40 geometries ({len(geoms)})')
    all_vertices = np.vstack([g.vertices for g in geoms]) if geoms else np.zeros((0, 3))
    bounds = np.vstack([all_vertices.min(axis=0), all_vertices.max(axis=0)])
    size = bounds[1] - bounds[0]
    require(0.85 <= size[2] <= 1.25,
            f'GLB height is about 1 m ({size[2]:.4f} m)')
    require(size[2] >= 3.0 * size[0],
            'height dominates the body/front-back thickness')
    require(size[2] >= 1.8 * size[1],
            'height still dominates the hose/handle envelope')
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
    require(len(material_names) >= 12,
            f'GLB carries the expected material set ({len(material_names)})')
    require(any('标贴' in name for name in material_names),
            'label material is present')
    require(any('镀铬' in name for name in material_names),
            'chrome material is present')
    require(any('黄铜' in name for name in material_names),
            'brass material is present')
    emboss_materials = [m for m in gltf.get('materials', [])
                        if '压印' in m.get('name', '')]
    require(bool(emboss_materials) and
            emboss_materials[0].get('alphaMode') == 'MASK',
            'embossed-text decal uses alphaMode MASK')
    chrome_materials = [m for m in gltf.get('materials', [])
                        if '镀铬' in m.get('name', '')]
    require(bool(chrome_materials) and
            chrome_materials[0].get('pbrMetallicRoughness', {}).get(
                'metallicRoughnessTexture') is not None,
            'chrome material carries a metallic-roughness texture')
    require(len(gltf.get('images', [])) >= 8,
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
    handle = rays.get('handle', {})
    require(handle.get('gap_clear') is True,
            'squeeze-lever gap is open along the front ray')
    require(handle.get('lower_solid') is True and
            handle.get('upper_solid') is True and
            handle.get('pivot_solid') is True,
            'both handles and the pivot are solid outside the gap')
    safety = rays.get('safety', {})
    require(safety.get('hole_clear') is True,
            'safety-ring hole is open')
    require(safety.get('ring_solid') is True,
            'safety ring is solid away from its hole')
    require(safety.get('tamper_present') is True and
            safety.get('tag_present') is True,
            'tamper strap and yellow tag are present')
    hose = rays.get('hose', {})
    require(hose.get('hose_solid') is True,
            'rubber hose is solid across its section')
    require(hose.get('coupling_solid') is True,
            'chrome hose coupling is present')
    require(hose.get('nozzle_body_solid') is True and
            hose.get('outlet_ring_solid') is True,
            'nozzle body and outlet ring are present')
    require(hose.get('dark_plug_blind') is True,
            'nozzle has a blind dark plug, not a false through hole')
    require(hose.get('plug_ray_clear_outlet') is True,
            'nozzle outlet ring leaves the measured hole clear')
    gauge = rays.get('gauge', {})
    require(gauge.get('glass_hit') is True and gauge.get('bezel_hit') is True,
            'gauge glass and chrome bezel are present')
    require(gauge.get('dial_present') is True and
            gauge.get('needle_present') is True,
            'gauge dial and needle objects exist')
    objects = rays.get('objects', {})
    require(objects.get('front_label') is True and
            objects.get('back_label') is True and
            objects.get('emboss_label') is True,
            'front/back labels and embossed decal exist')

    # Rendered evidence.
    metal = result['checks']['metal']
    require(metal.get('passed') is True,
            'metal highlight moves with the main light position')
    require(metal.get('mean_abs_difference', 0) >= 1.0,
            'metal-only render changes measurably between light positions')
    require(metal.get('metal_pixels', 0) >= 800,
            'metal-only render contains enough metal pixels to test')
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
    require(diagnostics.get('strict', {}).get('acceptance_gate') is False,
            'strict source IoU is flagged as non-acceptance diagnostic')
    require(diagnostics.get('strict', {}).get('available') is True and
            diagnostics.get('strict', {}).get('iou') is not None,
            'strict same-camera IoU was computed from the library alpha mask')
    require((diagnostics.get('new_source_view_fit') or {}).get('iou') is not None,
            'old source-view foreground shape diagnostic was computed')
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
    for name in ('gauge', 'handles', 'hose-top', 'nozzle', 'label-front',
                 'label-back', 'neck', 'bottom', 'overall'):
        require((out / 'renders' / f'detail-{name}.png').is_file(),
                f'new detail render exists: {name}')
        require((out / 'renders' / f'old-detail-{name}.png').is_file(),
                f'old detail render exists: {name}')
    for i in range(24):
        for prefix in ('turn-new', 'turn-old', 'turn-clay', 'sil-new', 'sil-old'):
            require((out / 'renders' / f'{prefix}-{i:02d}.png').is_file(),
                    f'turntable frame exists: {prefix}-{i:02d}')

    # Standalone page: every local image reference exists, JS syntax via node.
    html = (out / 'review-34.html').read_text(encoding='utf8')
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
    not_executed.append('浏览器运行时交互（无已安装无头浏览器）')
    pending.append('所有 IoU 均为未标定诊断，不是验收通过。')
    pending.append('背面、阀内结构、软管内部、标签原始印刷、真实尺寸与落稳待人工复核。')

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
                 '未标定 IoU 与浏览器运行时交互未被当作通过。'),
    }
    checks_path = out / 'checks-34.json'
    checks_path.write_text(json.dumps(checks_record, ensure_ascii=False, indent=2),
                           encoding='utf8')

    # Recompute hashes after checks-34.json exists.
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
        'next_number': 35,
    }
    (ROOT / 'work/direct-deepseek/34-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')

    print(json.dumps({'passed': len(passed), 'failed': failures,
                      'pending': pending, 'not_executed': not_executed},
                     ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
