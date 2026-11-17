"""Independent checks for the item-39 oil-receiver candidate.

Only artefacts that were actually produced are verified.  All silhouette IoU
values are uncalibrated diagnostics, never acceptance.  The standalone page is
checked statically, its inline JavaScript is parsed by Node, and a real headless
Chrome screenshot is verified.
"""
import argparse
import hashlib
import json
import math
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
NUMBER = 39


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


def luma_stats(path):
    image = Image.open(path).convert('RGBA')
    alpha = np.asarray(image.getchannel('A'))
    rgb = np.asarray(image.convert('RGB')).astype(np.float32)
    mask = alpha >= 128
    if not mask.any():
        return {'median_luma': 0, 'dark_fraction': 1, 'laplacian_variance': 0,
                'pixels': 0}
    gray = rgb.mean(axis=2)
    values = gray[mask]
    laplacian = cv2.Laplacian((gray * (mask > 0)).astype(np.uint8), cv2.CV_32F)
    return {'median_luma': round(float(np.median(values)), 1),
            'dark_fraction': round(float((values < 25).mean()), 4),
            'laplacian_variance': round(float(laplacian[mask].var()), 2),
            'pixels': int(mask.sum())}


def image_diff(first_path, second_path):
    first = np.asarray(Image.open(first_path).convert('RGBA')).astype(np.float32)
    second = np.asarray(Image.open(second_path).convert('RGBA')).astype(np.float32)
    mask = (first[..., 3] >= 128) & (second[..., 3] >= 128)
    if not mask.any():
        return {'mean_abs_difference': 0.0, 'pixels': 0, 'changed': False}
    difference = np.abs(first[..., :3] - second[..., :3])[mask].mean()
    return {'mean_abs_difference': round(float(difference), 3),
            'pixels': int(mask.sum()), 'changed': bool(difference >= 1.0)}


def interior_holes(path):
    with Image.open(path) as image:
        alpha = np.asarray(image.getchannel('A'))
    solid = alpha >= 128
    contours, hierarchy = cv2.findContours(solid.astype('uint8'), cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] < 0 or cv2.contourArea(contour) < 16:
                continue
            region = np.zeros(solid.shape, np.uint8)
            cv2.drawContours(region, contours, i, 1, -1)
            region[solid] = 0
            if region.sum() >= 16:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    failures, passed, pending, not_executed = [], [], [], []

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

    require(result.get('number') == NUMBER, 'result number is 39')
    require(result.get('manual_only') is True and
            result.get('enters_automatic_placement') is False and
            result.get('replaces_library_mesh') is False,
            'candidate is manual_only, does not replace the library mesh and '
            'does not enter automatic placement')
    require(result.get('reference_role') == 'visual_match_unverified',
            'six-view reference role is visual_match_unverified')
    reference = Path(result.get('reference', ''))
    require(reference.is_file(), 'six-view reference file exists')
    if reference.is_file():
        require(digest(reference) == result.get('reference_sha256'),
                'six-view reference hash matches result.json')

    for key, record in hashes.get('inputs', {}).items():
        path = Path(record['path'])
        if not require(path.is_file(), f'input file exists: {key}'):
            continue
        require(digest(path) == record['sha256'], f'input hash unchanged: {key}')
    for name, record in hashes.get('outputs', {}).items():
        path = out / name
        if not require(path.is_file(), f'output file exists: {name}'):
            continue
        require(digest(path) == record['sha256'], f'output hash matches: {name}')

    for name in ('mesh.glb', 'model.blend', 'review-39.html', 'result.json',
                 'texture-checks.json', 'request.json', 'worker-result.json',
                 'source-measurements.json', 'old-model-audit.json',
                 '六视图-新旧对照.jpg', '原照-新候选-参考.jpg', '局部检查.jpg',
                 '结构开口-隔离检查.jpg', '材质-同光照对照.jpg',
                 '材质-变光检查.jpg', '表面反射-变光对照.jpg',
                 'PBR贴图-对照.jpg', '转台-24方向.jpg'):
        require((out / name).is_file(), f'required artefact exists: {name}')

    texture_files = ('body_base', 'body_mr', 'body_normal',
                     'paint_base', 'paint_mr', 'paint_normal',
                     'label_base', 'label_mr', 'label_normal',
                     'metal_base', 'metal_mr', 'metal_normal',
                     'brass_base', 'brass_mr', 'brass_normal',
                     'black_base', 'black_mr', 'black_normal',
                     'red_base', 'red_mr', 'red_normal')
    for name in texture_files:
        require((out / f'{name}.png').is_file(), f'PBR texture exists: {name}.png')

    # Texture values.
    require(textures.get('passed') is True, 'PBR texture numeric checks pass')
    body = textures.get('body', {})
    require(.03 <= body.get('rust_fraction', 0) <= .32,
            'body rust coverage is plausible')
    require(body.get('dark_fraction', 1) < .20, 'body is not mostly black')
    require(body.get('metallic_mean', 1) < .16,
            'painted body metallic mean is low (dielectric paint)')
    require(.28 <= body.get('roughness_mean', 0) <= .68,
            'painted body roughness is plausible')
    label = textures.get('label', {})
    require(label.get('metallic_mean', 1) < .05, 'label paper is dielectric')
    require(label.get('mean_luma', 0) >= 105, 'label is bright enough to read')
    require(label.get('mean_chroma', 0) >= 35, 'label keeps its yellow/blue colour')
    metal = textures.get('metal', {})
    require(metal.get('metallic_mean', 0) >= .90, 'stainless texture metallic mean is high')
    require(.22 <= metal.get('roughness_mean', 0) <= .48, 'stainless roughness is plausible')
    brass = textures.get('brass', {})
    require(brass.get('metallic_mean', 0) >= .90, 'brass texture metallic mean is high')
    require(brass.get('mean_rgb', [0, 0, 0])[0] >
            brass.get('mean_rgb', [0, 0, 0])[1] >
            brass.get('mean_rgb', [0, 0, 0])[2], 'brass base colour is warm')
    black = textures.get('black', {})
    require(black.get('metallic_mean', 1) < .05, 'black plastic is dielectric')
    require(black.get('mean_luma', 255) < 70, 'black plastic base is dark')
    paint = textures.get('paint', {})
    require(paint.get('passed') is True, 'paint atlas tile checks pass')

    # GLB geometry and materials.
    scene = trimesh.load(out / 'mesh.glb', force='scene')
    geoms = list(scene.geometry.values())
    require(len(geoms) >= 45, f'GLB has at least 45 geometries ({len(geoms)})')
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    size = np.asarray(scene.extents, dtype=np.float64)
    require(np.isfinite(bounds).all(), 'GLB bounds are finite')
    require(0.60 <= size[2] <= 0.70,
            f'GLB overall height is about 0.65 m ({size[2]:.4f} m)')
    require(size[2] >= 3.2 * size[0] and size[2] >= 3.2 * size[1],
            'height dominates the body diameter in both horizontal axes')
    require(abs(bounds[0, 2]) < 1e-6,
            f'GLB bottom is at z=0 ({bounds[0, 2]:.6f})')
    watertight = {}
    raw_watertight = {}
    for name, geometry in scene.geometry.items():
        raw_watertight[name] = bool(geometry.is_watertight)
        merged = trimesh.Trimesh(np.round(geometry.vertices, 7), geometry.faces,
                                 process=True, validate=False)
        merged.merge_vertices()
        watertight[name] = bool(merged.is_watertight)
    require(all(watertight.values()),
            f'all exported GLB surfaces are watertight after merging seam vertices '
            f'({sum(watertight.values())}/{len(watertight)})')
    gltf = glb_json(out / 'mesh.glb')
    material_names = [m.get('name', '') for m in gltf.get('materials', [])]
    require(len(material_names) >= 8,
            f'GLB carries at least 8 materials ({len(material_names)})')
    for token in ('stainless', 'brass', 'black', 'teal painted', 'HPEOK label'):
        require(any(token in name for name in material_names),
                f'GLB material present: {token}')
    require(len(gltf.get('images', [])) >= 15,
            f'GLB embeds PBR images ({len(gltf.get("images", []))})')

    # Worker topology and part counts.
    require(worker.get('closed_parts') == len(worker.get('parts', [])),
            'worker reports every part closed')
    require(all(value == 0 for value in worker.get('closures', {}).values()),
            'Blender manifold check: zero non-manifold edges in every part')
    require(not worker.get('self_intersecting_components'),
            'Blender BVH check: no same-part non-adjacent face overlap')
    counts = result.get('structural_checks', {}).get('counts', {})
    require(counts.get('bodies') == 1, 'exactly one body solid')
    require(counts.get('weld_seams') == 2, 'two weld seam rings are present')
    require(counts.get('port_bosses') == 2 and counts.get('port_glass') == 2,
            'two port bosses and two sight glasses are present')
    require(counts.get('port_red_dots') == 2 and counts.get('port_screws') == 6,
            'two red indicator dots and six port screws are present')
    require(counts.get('labels') == 1, 'one HPEOK label decal is present')
    require(counts.get('top_parts', 0) >= 10, 'top valve assembly is present')
    require(counts.get('bottom_parts', 0) >= 10, 'bottom drain assembly is present')
    require(counts.get('side_fitting_parts') == 3, 'side service fitting is present')

    require(interior_holes(out / 'renders' / 'silhouette-front.png') == 0,
            'isolated front silhouette has no interior hole')
    require(interior_holes(out / 'renders' / 'silhouette-top.png') == 0,
            'isolated top silhouette has no interior hole')

    # Recompute rendered evidence.
    alpha_path = Path(hashes.get('inputs', {}).get('source_alpha', {}).get('path', ''))
    require(alpha_path.is_file(), 'source alpha is available for the diagnostic')
    if alpha_path.is_file():
        sys.path.insert(0, str(HERE))
        from mesh_checks import source_fidelity
        with Image.open(alpha_path) as alpha, \
                Image.open(out / 'renders' / 'new-source.png') as rendered:
            fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                       np.asarray(rendered.getchannel('A')))
        require(fidelity.get('iou') is not None, 'source-camera silhouette IoU was computed')
        require(fidelity.get('filled_holes') == 0,
                'source-camera diagnostic has no filled holes')
        pending.append('源相机轮廓 IoU %.4f 为未标定诊断，不是验收通过。' % fidelity['iou'])
    else:
        fidelity = result.get('source_fidelity_diagnostic')

    metal = image_diff(out / 'renders' / 'new-metal-light-a.png',
                       out / 'renders' / 'new-metal-light-b.png')
    paint = image_diff(out / 'renders' / 'new-paint-light-a.png',
                       out / 'renders' / 'new-paint-light-b.png')
    require(metal['changed'] and metal['pixels'] >= 800,
            'metal-only highlights move measurably with the key light')
    require(paint['changed'], 'paint-only highlights move measurably with the key light')

    luma = {name: luma_stats(out / 'renders' / name) for name in
            ('new-front.png', 'new-material-close.png', 'detail-label.png')}
    for key, value in luma.items():
        require(value['dark_fraction'] < .35, f'{key} is not mostly black')
        require(value['median_luma'] > 20, f'{key} median luma is not crushed')
    require(luma['new-material-close.png']['laplacian_variance'] > 20,
            'new-material-close retains visible detail')

    # Six-view and source renders.
    for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        for prefix in ('new', 'old'):
            require((out / 'renders' / f'{prefix}-{view}.png').is_file(),
                    f'six-view render exists: {prefix}-{view}')
    for name in ('overall-front', 'top-manifold', 'top-block', 'top-knob',
                 'top-coupler', 'upper-port', 'lower-port', 'label',
                 'weld-upper', 'weld-lower', 'side-fitting', 'bottom-manifold',
                 'bottom-block', 'body-surface'):
        require((out / 'renders' / f'detail-{name}.png').is_file(),
                f'new detail render exists: {name}')
    for name in ('body-only', 'ports-only', 'label-only', 'top-valve-only',
                 'bottom-valve-only', 'side-fitting-only', 'silhouette-front',
                 'silhouette-top'):
        require((out / 'renders' / f'{name}.png').is_file(),
                f'isolation render exists: {name}')
    for i in range(24):
        require((out / 'renders' / f'turn-{i:02d}.png').is_file(),
                f'turntable frame exists: turn-{i:02d}')
    require((out / 'renders' / 'new-source.png').is_file(),
            'new source-view diagnostic render exists')
    require((out / 'renders' / 'old-source.png').is_file(),
            'old source-view diagnostic render exists')

    # Standalone page: local image references exist, JS syntax via Node, browser shot.
    html = (out / 'review-39.html').read_text(encoding='utf8')
    references = set(re.findall(r'src="([^"]+)"', html))
    references |= set(re.findall(r"src='([^']+)'", html))
    for reference_name in sorted(references):
        if reference_name.startswith(('http:', 'https:', 'data:')):
            continue
        if '<' in reference_name or '{' in reference_name:
            continue
        require((out / reference_name).is_file(),
                f'page image asset exists: {reference_name}')
    script = re.search(r'<script>(.*?)</script>', html, re.S)
    require(script is not None, 'page contains an inline script')
    if script:
        js_path = out / 'renders' / 'page-check-39.js'
        js_path.write_text(script.group(1), encoding='utf8')
        node = subprocess.run(['node', '--check', str(js_path)],
                              capture_output=True, text=True)
        require(node.returncode == 0,
                'page JavaScript syntax passes node --check'
                + ('' if node.returncode == 0 else f': {node.stderr.strip()}'))
    require('manual_only' in html and '不能进入正式合成' in html,
            'page carries the manual_only / no formal composite notice')
    require('visual_match_unverified' in html,
            'page records the uncalibrated visual-match reference role')
    require('IoU' in html and '诊断' in html,
            'page labels IoU values as diagnostics')
    browser = result.get('browser_check', {})
    require(browser.get('status') == 'headless_chrome_screenshot_ok',
            'headless Chrome page screenshot was produced')
    screenshot = Path(browser.get('screenshot', ''))
    if screenshot.is_file():
        arr = np.asarray(Image.open(screenshot).convert('RGB'))
        require(arr.std() > 3, 'page screenshot is non-blank')
    else:
        require(False, 'page screenshot file exists')

    # Reference diagnostics are recorded but never acceptance.
    views = result.get('reference_views', [])
    require(len(views) == 6, 'six reference-view diagnostics are recorded')
    for row in views:
        require(row.get('reference_new_mask_iou') is not None,
                f"reference/new diagnostic IoU computed for {row.get('view')}")
        require(row.get('old_new_silhouette_iou') is not None,
                f"old/new silhouette IoU computed for {row.get('view')}")
    pending.append('六视图各面板缩放不一致且未标定；参考-新 IoU 只用于发现形体差异。')
    pending.append('壁厚、内部通道、螺纹、阀芯可动性、背面细节与落稳未验证。')
    pending.append('旧模型材质为单图顶点色，当前 Cycles 对照只用于几何，不评价旧材质。')
    not_executed.append('现场尺度、落稳、支撑/吊架与场景光照融合未执行。')

    internal_diag = {
        'source_camera_iou': fidelity,
        'reference_views': views,
        'metal_response': metal,
        'paint_response': paint,
        'luma': luma,
    }
    checks_record = {
        'number': NUMBER,
        'status': 'passed_with_pending_review' if not failures else 'failed',
        'passed': passed,
        'failed': failures,
        'pending': pending,
        'not_executed': not_executed,
        'metrics': {
            'glb_geometries': len(geoms),
            'glb_bounds': bounds.tolist(),
            'glb_size': size.tolist(),
            'glb_watertight_merged': watertight,
            'glb_watertight_raw': raw_watertight,
            'materials': material_names,
            'texture_checks': textures,
            'structural_counts': counts,
            'diagnostics': internal_diag,
        },
        'note': ('通过项只覆盖实际执行的网格、贴图、渲染、拓扑和静态/浏览器页面检查；'
                 '未标定 IoU、隐藏结构和现场落稳未被当作通过。'),
    }
    checks_path = out / 'checks-39.json'
    checks_path.write_text(json.dumps(checks_record, ensure_ascii=False, indent=2),
                           encoding='utf8')

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
            'source_fidelity_diagnostic': fidelity,
            'metal_response': metal,
            'paint_response': paint,
            'texture_passed': textures.get('passed'),
            'browser': browser.get('status'),
        },
        'pending_review': result.get('pending_review', []),
        'next_number': 40,
        'manual_only': True,
        'replaces_library_mesh': False,
        'enters_automatic_placement': False,
        'hashes': str(out / 'hashes.json'),
        'result': str(out / 'result.json'),
        'notes': ('Item 39 is the last numbered item; next_number 40 follows the '
                  'continuation convention. All IoU values are uncalibrated '
                  'diagnostics; candidate is manual_only.'),
    }
    (ROOT / 'work/direct-deepseek/39-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    (out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                       encoding='utf8')

    # Recompute output hashes only after checks-39.json and progress.json exist.
    outputs = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'hashes.json':
            outputs[str(path.relative_to(out)).replace('\\', '/')] = {
                'sha256': digest(path), 'bytes': path.stat().st_size}
    hashes['outputs'] = outputs
    (out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                     encoding='utf8')

    print(json.dumps({'passed': len(passed), 'failed': failures,
                      'pending': pending, 'not_executed': not_executed,
                      'record': str(checks_path)}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
