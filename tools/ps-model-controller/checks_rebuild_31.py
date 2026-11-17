"""Integrity and technical checks for the item-31 cutaway hydraulic-cylinder candidate.

This script does not perform visual acceptance or calibrated measurement.  It
verifies that the recorded hashes, files and technical checks agree with the
actual output, and that the candidate is still manual-only and pending review.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 31
VERSION = 'cylinder31-parametric-v1'
REFERENCE_SHA256 = '7290a81f89e40a7989b52ecaf0450ffcd1abe9529eb849ce08817e5ac2809ade'
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
DETAILS = ('rear-eye', 'rear-port', 'front-eye', 'gland-stack', 'piston-rings',
           'window', 'tee-port', 'barrel-seam', 'rear-cap', 'rod', 'overall-3q')
ISOLATIONS = ('rear-eye-only', 'front-eye-only', 'piston-rod-only', 'gland-only',
              'tee-only', 'barrel-only', 'new-material-front', 'new-clay-front',
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
                   default=ROOT / 'outputs/31液压缸1-六视图重建-v1')
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

    need(result.get('number') == NUMBER, 'number is not 31')
    need(result.get('version') == VERSION, 'version differs')
    need(result.get('manual_only') is True, 'candidate must remain manual_only')
    need(result.get('status') == 'review', 'status must remain review')
    need(result.get('draft') is False, 'final result must not be a draft')
    need(result.get('reference', {}).get('role') == 'visual_match_unverified',
         'six-view reference role is not visual_match_unverified')
    need(result.get('reference', {}).get('sha256') == REFERENCE_SHA256,
         'reference hash differs from the registered value')
    need(result.get('checks', {}).get('browser_review_executed') is False,
         'browser review must not be claimed as executed')
    need(result.get('checks', {}).get('source_fidelity_acceptance') is False,
         'source fidelity must not be claimed as accepted')
    need(result.get('checks', {}).get('nonmetal_metal_reflection_applicable') is False,
         'non-metal metal-reflection applicability must be false')

    for key in ('mesh', 'blend'):
        path = out / ('mesh.glb' if key == 'mesh' else 'model.blend')
        need(path.is_file(), f'{key} file missing')
        expected = result.get(f'{key}_sha256')
        if path.is_file() and expected:
            need(digest(path) == expected, f'{key} hash differs from result.json')
    for name in ('base_color.png', 'metallic_roughness.png', 'normal.png',
                 '六视图-新旧对照.jpg', '局部检查.jpg', '分部件隔离.jpg',
                 '材质-同光照对照.jpg', '金属反射-变光对照.jpg',
                 '开孔-隔离检查.jpg', '原视角诊断.jpg', '原照-新候选对照.jpg',
                 '转台-24方向.jpg', '材质图集-布局.jpg', '材质图集-对照.jpg',
                 'report.md', 'source-measurements.json', 'old-model-audit.json',
                 'worker-result.json', 'request.json', 'blender.log'):
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
    for i in range(24):
        need((out / 'renders' / f'turn-{i:02d}.png').is_file(),
             f'missing turntable frame {i:02d}')
    need((HERE / 'model-review-31.html').is_file(), 'independent review page missing')
    need((HERE / 'mesh_rebuild_item_31.py').is_file(), 'item-31 driver missing')
    need((HERE / 'mesh_rebuild_31_worker.py').is_file(), 'item-31 worker missing')
    need((HERE / 'cylinder31_config.json').is_file(), 'item-31 config missing')

    topology = mesh_topology(out / 'mesh.glb')
    need(not topology['open_parts'], f"open mesh parts: {topology['open_parts']}")
    need(not topology['nonfinite_parts'], f"non-finite mesh parts: {topology['nonfinite_parts']}")
    need(result['mesh']['parts'] == topology['parts'],
         'worker part count differs from the exported GLB')
    need(result['mesh']['polygons'] > 15000, 'polygon count is unexpectedly low')
    need(result['mesh']['triangles'] > 30000, 'triangle count is unexpectedly low')
    need(not result['worker']['self_intersecting_components'],
         f"self-intersecting parts: {result['worker']['self_intersecting_components']}")

    # source measurements / reference evidence
    need(len(result['measurements'].get('yellow_components', [])) >= 1,
         'yellow paint region was not recorded')
    need(len(result['measurements'].get('steel_components', [])) >= 1,
         'steel eye/rod regions were not recorded')
    need(len(result['measurements'].get('bronze_components', [])) >= 1,
         'the bronze-coloured seal region was not recorded')
    need(result['measurements'].get('min_area_rect', {}).get('length_over_width', 0) > 1.5,
         'photo silhouette ratio is implausible')
    ratio = result['measurements'].get('configured', {}).get('overall_length_over_height', 0)
    need(3.0 <= ratio <= 5.0,
         f'configured overall length/height ratio is implausible: {ratio}')
    need(0.10 <= result['measurements'].get('configured', {}).get('max_height_m', 0) <= 0.25,
         'configured max height is implausible for the working scale')
    need(result['measurements'].get('rust_color_shift', {}).get('brown_fraction', 0) > 0,
         'rust photograph colour evidence missing')
    need(result['old_audit'].get('components', 0) >= 1,
         'old-model audit component count missing')
    need(result['old_audit'].get('vertices', 0) < 20000,
         'old-model audit does not record the low-resolution mesh')
    need(result['audit'].get('source_fidelity', {}).get('filled_holes', 0) >= 1,
         'old-model audit did not record the filled holes')

    # real openings and part structure
    need(result['openings'].get('eye_holes_through') == 2,
         f"eye through holes = {result['openings'].get('eye_holes_through')}")
    need(result['openings'].get('bushings_present') == 2,
         f"bushings = {result['openings'].get('bushings_present')}")
    need(result['openings'].get('rear_rubber_present') is True,
         'rear rubber sleeve not recorded')
    need(result['openings'].get('window', {}).get('open_through_near_wall') is True,
         'cutaway window near wall not recorded as open')
    need(result['openings'].get('window', {}).get('back_wall_intact') is True,
         'cutaway window back wall not recorded as intact')
    need(result['openings'].get('ports', {}).get('mouths_through') == 2,
         f"tee port through mouths = {result['openings'].get('ports', {}).get('mouths_through')}")
    need(result['openings'].get('piston_rings') == 4,
         f"piston rings = {result['openings'].get('piston_rings')}")
    need((result['openings'].get('gland_rings') or 0) >= 5,
         f"gland rings = {result['openings'].get('gland_rings')}")
    need(result['textures'].get('passed') is True, 'texture value checks not passing')
    need(result['metal'].get('passed') is True, 'metal highlight movement check not passing')
    need(result['checks'].get('metal_highlights_move') is True,
         'metal highlight result not recorded as passing')

    # independent texture sample
    base = np.asarray(Image.open(out / 'base_color.png').convert('RGB'))
    mr = np.asarray(Image.open(out / 'metallic_roughness.png').convert('RGB'))
    normal = np.asarray(Image.open(out / 'normal.png').convert('RGB'))
    need(base.shape == (2048, 2048, 3), 'base atlas is not 2048 px')
    need(mr.shape == (2048, 2048, 3), 'metal/roughness atlas is not 2048 px')
    need(normal.shape == (2048, 2048, 3), 'normal atlas is not 2048 px')
    config = json.loads((HERE / 'cylinder31_config.json').read_text(encoding='utf8'))
    for name, idx in config['texture']['tiles'].items():
        col, row = idx % 4, idx // 4
        x0 = col * 512 + 220
        y0 = (3 - row) * 512 + 220
        sample = mr[y0:y0 + 64, x0:x0 + 64].reshape(-1, 3)
        p90 = float(np.percentile(sample[:, 2], 90))
        p95 = float(np.percentile(sample[:, 2], 95))
        if name in ('chrome', 'steel', 'bronze'):
            need(p90 > 220, f'{name} metallic p90 is not high')
        elif name == 'dark_steel':
            need(p90 > 170, f'{name} metallic p90 is not high enough')
        elif name == 'bore':
            need(p90 > 150, f'{name} metallic p90 is not high enough')
        else:
            need(p95 < 15, f'{name} metallic is not ~0')
    need(float(np.std(normal)) > .5, 'normal atlas is flat')

    # hashes of all recorded outputs and inputs
    for rel, record in hashes.get('outputs', {}).items():
        path = ROOT / rel
        need(path.is_file(), f'hashed output missing: {rel}')
        if path.is_file():
            need(digest(path) == record['sha256'], f'hashed output changed: {rel}')
    for name, record in hashes.get('inputs', {}).items():
        path = ROOT / record['path']
        need(path.is_file(), f'hashed input missing: {name}')
        if path.is_file():
            need(digest(path) == record['sha256'], f'hashed input changed: {name}')

    progress_path = ROOT / 'work/direct-deepseek/31-progress.json'
    need(progress_path.is_file(), 'progress file missing')
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding='utf8'))
        need(progress.get('number') == 31, 'progress number is not 31')
        need(progress.get('status') == 'review', 'progress status is not review')
        need(progress.get('mesh') == str(out / 'mesh.glb'),
             'progress mesh path is not the actual absolute path')
        need(progress.get('blend') == str(out / 'model.blend'),
             'progress blend path is not the actual absolute path')
        need(progress.get('next_number') == 32, 'progress next_number is not 32')

    summary = {
        'status': 'pass' if not failures else 'fail',
        'out': str(out),
        'parts': topology['parts'],
        'closed_parts': topology['parts'] - len(topology['open_parts']),
        'eye_holes_through': result['openings'].get('eye_holes_through'),
        'bushings_present': result['openings'].get('bushings_present'),
        'rear_rubber_present': result['openings'].get('rear_rubber_present'),
        'window_open_through': result['openings']['window'].get('open_through_near_wall'),
        'back_wall_intact': result['openings']['window'].get('back_wall_intact'),
        'tee_port_mouths_through': result['openings']['ports'].get('mouths_through'),
        'piston_rings': result['openings'].get('piston_rings'),
        'gland_rings': result['openings'].get('gland_rings'),
        'texture_values': result['textures'].get('passed'),
        'metal_highlights_move': result['metal'].get('passed'),
        'strict_iou_diagnostic': result['diagnostics']['strict']['iou'],
        'front_box_fit_iou_diagnostic': (result['diagnostics'].get('front_box_fit') or {}).get('iou'),
        'turntable_best_box_fit_iou_diagnostic': (result['diagnostics'].get('best') or {}).get('iou'),
        'manual_only': True,
        'visual_acceptance': False,
        'browser_review_executed': False,
        'failures': failures,
        'notes': [
            'Technical integrity only; visual, hidden-surface, scale, stability and scene checks remain pending.',
            'IoU values are uncalibrated diagnostics, not acceptance thresholds.',
            'The candidate does not replace the original model and never enters automatic placement.',
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
