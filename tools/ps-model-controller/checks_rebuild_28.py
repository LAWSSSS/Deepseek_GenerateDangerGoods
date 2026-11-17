"""Integrity and technical checks for the item-28 coil-over shock candidate.

This script does not perform visual acceptance or calibrated measurement.  It
verifies that the recorded hashes, files and technical checks agree with the
actual output, and that the candidate is still manual-only and pending review.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 28
VERSION = 'shock28-parametric-v1'
REFERENCE_SHA256 = 'dcc140e00fd1878933b5fc066845f9ec64a23f3bea53e736436f5e97748922d2'
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
DETAILS = ('top-eye', 'top-eye-bore', 'top-seat-notch', 'spring-coils',
           'body-rod', 'lower-seat', 'bottom-eye', 'bottom-eye-bore',
           'overall-3q', 'overall-side')
ISOLATIONS = ('seats-only-top', 'seats-only-bottom', 'top-eye-only',
              'bottom-eye-only', 'spring-only', 'body-rod-only',
              'new-material-front', 'new-clay-front', 'new-clay-3q',
              'new-light-alternate', 'metal-only-light-a', 'metal-only-light-b')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mesh_topology(path):
    scene = trimesh.load(path, force='scene')
    open_parts, nonfinite = [], []
    for node in scene.graph.nodes_geometry:
        _, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not np.isfinite(mesh.vertices).all():
            nonfinite.append(name)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        if not mesh.is_watertight:
            open_parts.append(name)
    return {'parts': len(scene.graph.nodes_geometry),
            'open_parts': open_parts, 'nonfinite_parts': nonfinite,
            'bounds': scene.bounds.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path,
                   default=ROOT / 'outputs/28液压减震器1-六视图重建-v5')
    a = p.parse_args()
    out = a.out.resolve()
    failures = []

    def need(condition, message):
        if not condition:
            failures.append(message)

    result_path = out / 'result.json'
    hashes_path = out / 'hashes.json'
    need(result_path.is_file(), 'result.json missing')
    need(hashes_path.is_file(), 'hashes.json missing')
    if failures:
        print(json.dumps({'status': 'fail', 'failures': failures}, ensure_ascii=False,
                         indent=2))
        return 1
    result = json.loads(result_path.read_text(encoding='utf8'))
    hashes = json.loads(hashes_path.read_text(encoding='utf8'))

    need(result.get('number') == NUMBER, 'number is not 28')
    need(result.get('version') == VERSION, 'version differs')
    need(result.get('manual_only') is True, 'candidate must remain manual_only')
    need(result.get('status') == 'review', 'status must remain review')
    need(result.get('draft') is False, 'final result must not be a draft')
    need(result.get('reference', {}).get('role') == 'visual_match_unverified',
         'six-view reference role is not visual_match_unverified')
    need(result.get('reference', {}).get('sha256') == REFERENCE_SHA256,
         'reference hash differs from the registered value')
    need(result.get('checks', {}).get('manual_visual_review') == 'not_performed',
         'manual visual review must not be claimed as performed')
    need(result.get('checks', {}).get('source_silhouette_turntable_best_iou_diagnostic')
         is not None, 'turntable silhouette diagnostic missing')
    need((result.get('silhouette_diagnostics') or {}).get('passed') is None,
         'silhouette diagnostics must not claim acceptance')

    for key in ('mesh', 'blend'):
        path = out / ('mesh.glb' if key == 'mesh' else 'model.blend')
        need(path.is_file(), f'{key} file missing')
        expected = result.get(f'{key}_sha256')
        if path.is_file() and expected:
            need(digest(path) == expected, f'{key} hash differs from result.json')
    for name in ('base_color.png', 'metallic_roughness.png', 'normal.png',
                 '六视图-新旧对照.jpg', '局部检查.jpg', '分部件隔离.jpg',
                 '材质-同光照对照.jpg', '金属反射-变光对照.jpg',
                 '开孔-结构隔离检查.jpg', '原照-新候选对照.jpg',
                 '原视角诊断.jpg', '转台-24方向.jpg', '材质图集-布局.jpg',
                 'report.md', 'source-measurements.json', 'old-model-audit.json',
                 'texture-evidence.json', 'worker-result.json', 'request.json',
                 'blender.log', '检查页.html', '浏览器核验.json'):
        need((out / name).is_file(), f'required output missing: {name}')
    for which in ('new', 'old'):
        for view in VIEWS:
            need((out / 'renders' / f'{which}-{view}.png').is_file(),
                 f'missing render {which}-{view}')
    for name in DETAILS:
        need((out / 'renders' / f'detail-{name}.png').is_file(),
             f'missing detail render {name}')
    for name in ISOLATIONS:
        need((out / 'renders' / f'{name}.png').is_file(),
             f'missing isolation render {name}')
    for name in ('new-source.png', 'new-source-tilt.png',
                 'source-alpha-diagnostic.png', 'old-overall-3q.png'):
        need((out / 'renders' / name).is_file(), f'missing render {name}')
    for i in range(24):
        need((out / 'renders' / f'turn-{i:02d}.png').is_file(),
             f'missing turntable render {i:02d}')

    # Textures must be real images with variation (not constant black plates).
    for name in ('base_color.png', 'metallic_roughness.png', 'normal.png'):
        im = np.asarray(Image.open(out / name).convert('RGB'))
        need(im.shape[:2] == (2048, 2048), f'{name} is not 2048x2048')
        need(float(im.std()) > 1.0, f'{name} is near-constant')

    topology = mesh_topology(out / 'mesh.glb')
    checks = result['checks']
    need(topology['parts'] == checks['mesh_parts'],
         'mesh part count differs from result.json')
    need(not topology['open_parts'], f'open GLB parts: {topology["open_parts"]}')
    need(checks['mesh_all_closed'] is True, 'mesh_all_closed is not true')
    need(checks['self_intersection_free'] is True, 'self_intersection_free is not true')
    need(checks['eye_bores_through'] is True, 'eye bores are not open')
    need(checks['spring_axis_open'] is True, 'spring axis is not open')
    need(checks['spring_coil_turns'] == 12.0, 'spring turn count differs from 12')
    need(checks['texture_checks_passed'] is True, 'texture checks did not pass')
    need(checks['metal_highlights_move'] is True, 'metal highlights did not move')
    need(checks['required_renders_present'] is True, 'required renders missing')
    metal = result.get('metal_reflection_check') or {}
    need(metal.get('changed') is True, 'metal reflection record is not true')
    need((metal.get('mask_pixels') or 0) > 1000, 'metal reflection mask too small')
    bores = (result.get('isolated_opening_checks') or {}).get('eye_bores') or {}
    for tag in ('top', 'bottom'):
        need(tag in bores, f'eye bore record missing for {tag}')
        if tag in bores:
            for key, row in bores[tag].items():
                need(row.get('through') is True,
                     f'{tag} eye bore {key} is not through')
    browser = result.get('browser_check') or {}
    need(browser.get('status') == 'passed',
         f'browser self-test did not pass: {browser.get("status")} {browser.get("reason")}')

    # Hashes recorded in hashes.json must match the files that actually exist.
    for rel, expected in (hashes.get('outputs') or {}).items():
        path = out / rel
        if path.is_file() and path.name != 'hashes.json':
            need(digest(path) == expected, f'output hash differs: {rel}')
    for rel in ('result.json', 'mesh.glb', 'model.blend'):
        need(rel in (hashes.get('outputs') or {}), f'hash record missing {rel}')

    summary = {
        'status': 'fail' if failures else 'ok',
        'failures': failures,
        'mesh_parts': topology['parts'],
        'open_parts': topology['open_parts'],
        'eye_bores_through': checks['eye_bores_through'],
        'spring_axis_open': checks['spring_axis_open'],
        'texture_checks_passed': checks['texture_checks_passed'],
        'metal_highlights_move': checks['metal_highlights_move'],
        'browser': browser.get('status'),
        'strict_source_iou': checks.get('source_silhouette_strict_iou_diagnostic'),
        'tilted_source_iou': checks.get('source_silhouette_tilted_iou_diagnostic'),
        'turntable_best_iou': checks.get(
            'source_silhouette_turntable_best_iou_diagnostic'),
        'note': ('These checks cover file/hash agreement and listed geometric '
                 'properties only. They do not prove visual acceptance, hidden '
                 'structure, physical settling or field scale.'),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
