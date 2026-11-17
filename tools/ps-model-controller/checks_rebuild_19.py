"""Integrity and technical checks for the item-19 six-cylinder candidate.

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
NUMBER = 19
VERSION = 'cylinders19-parametric-v1'
REFERENCE_SHA256 = 'db523981aaa81e97f9dc260920408806e6cf0512320cd95d6330078a8bcec8f7'
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')


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
                   default=ROOT / 'outputs/19气瓶组-六视图重建-v4')
    a = p.parse_args()
    out = a.out.resolve()
    failures = []
    notes = []

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

    need(result.get('number') == NUMBER, 'number is not 19')
    need(result.get('version') == VERSION, 'version differs')
    need(result.get('manual_only') is True, 'candidate must remain manual_only')
    need(result.get('status') == 'review', 'status must remain review')
    need(result.get('draft') is False, 'final result must not be a draft')
    need(result.get('reference', {}).get('role') == 'category_reference',
         'six-view reference role is not category_reference')
    need(result.get('reference', {}).get('sha256') == REFERENCE_SHA256,
         'reference hash differs from the registered value')
    need(result.get('checks', {}).get('browser_review_executed') is False,
         'browser review must not be claimed as executed')
    need(result.get('checks', {}).get('source_fidelity_acceptance') is False,
         'source fidelity must not be claimed as accepted')

    for key in ('mesh', 'blend'):
        path = out / ('mesh.glb' if key == 'mesh' else 'model.blend')
        need(path.is_file(), f'{key} file missing')
        expected = result.get(f'{key}_sha256')
        if path.is_file() and expected:
            need(digest(path) == expected, f'{key} hash differs from result.json')
    for name in ('base_color.png', 'metallic_roughness.png', 'normal.png',
                 '六视图-新旧对照.jpg', '局部检查.jpg', '材质-同光照对照.jpg',
                 '金属反射-变光对照.jpg', '开口-隔离检查.jpg', '原视角诊断.jpg',
                 '原照-新候选对照.jpg', '转台-24方向.jpg', '材质图集-布局.jpg',
                 'report.md', 'source-measurements.json', 'old-model-audit.json',
                 'worker-result.json'):
        need((out / name).is_file(), f'required output missing: {name}')
    for which in ('new', 'old'):
        for view in VIEWS:
            need((out / 'renders' / f'{which}-{view}.png').is_file(),
                 f'missing render {which}-{view}')
    for i in range(24):
        need((out / 'renders' / f'turn-{i:02d}.png').is_file(),
             f'missing turntable frame {i:02d}')
    need((HERE / 'model-review-19.html').is_file(), 'independent review page missing')
    need((HERE / 'mesh_rebuild_item_19.py').is_file(), 'item-19 driver missing')
    need((HERE / 'mesh_rebuild_19_worker.py').is_file(), 'item-19 worker missing')

    # independently re-check mesh closedness and the recorded worker counts
    topology = mesh_topology(out / 'mesh.glb')
    need(not topology['open_parts'], f"open mesh parts: {topology['open_parts']}")
    need(not topology['nonfinite_parts'], f"non-finite mesh parts: {topology['nonfinite_parts']}")
    need(result['mesh']['parts'] == topology['parts'],
         'worker part count differs from the exported GLB')
    need(result['mesh']['polygons'] > 10000, 'polygon count is unexpectedly low')
    need(result['mesh']['triangles'] > 20000, 'triangle count is unexpectedly low')
    need(not result['worker']['self_intersecting_components'],
         f"self-intersecting parts: {result['worker']['self_intersecting_components']}")

    # source measurements must match the registered photo coordinates
    for cid, row in result['measurements']['cylinders'].items():
        for name, delta in row['delta_px'].items():
            need(abs(delta) <= 1.5, f'{cid} measured {name} differs by {delta} px')
    need(len(result['old_audit']['fragments']) >= 2,
         'old-model audit did not record the floating fragments')
    need(result['old_audit']['components'] >= 3,
         'old-model audit component count is inconsistent')

    # guard openings, separateness, gap, textures and metal highlights
    need(result['openings'].get('front_holes') == 2,
         f"front guard holes = {result['openings'].get('front_holes')}")
    need(result['openings'].get('back_holes') == 2,
         f"back guard holes = {result['openings'].get('back_holes')}")
    need(result['openings'].get('top_components') == 6,
         f"top body components = {result['openings'].get('top_components')}")
    need(result['body_gap'].get('passed') is True, 'body vertex gaps not passing')
    need(result['textures'].get('passed') is True, 'texture value checks not passing')
    need(result['metal'].get('passed') is True, 'metal highlight movement check not passing')
    need(result['checks'].get('metal_highlights_move') is True,
         'metal highlight result not recorded as passing')

    # texture tile values, re-sampled independently
    base = np.asarray(Image.open(out / 'base_color.png').convert('RGB'))
    mr = np.asarray(Image.open(out / 'metallic_roughness.png').convert('RGB'))
    normal = np.asarray(Image.open(out / 'normal.png').convert('RGB'))
    need(base.shape == (2048, 2048, 3), 'base atlas is not 2048 px')
    need(mr.shape == (2048, 2048, 3), 'metal/roughness atlas is not 2048 px')
    need(normal.shape == (2048, 2048, 3), 'normal atlas is not 2048 px')
    config = json.loads((HERE / 'cylinders19_config.json').read_text(encoding='utf8'))
    for name, idx in config['texture']['tiles'].items():
        col, row = idx % 4, idx // 4
        # the atlas stores tile row 0 at the bottom of the PNG
        x0, y0 = col * 512 + 220, (3 - row) * 512 + 220
        sample = mr[y0:y0 + 64, x0:x0 + 64]
        med = np.median(sample.reshape(-1, 3), axis=0)
        if name in ('brass', 'chrome'):
            need(med[2] > 245, f'{name} metallic is not ~1')
        elif name in ('red', 'green1', 'green2', 'blue1', 'blue2', 'blue3',
                      'gauge', 'plastic_blue', 'label_white'):
            need(med[2] < 8, f'{name} metallic is not 0')
    need(float(np.std(normal)) > .5, 'normal atlas is flat')

    # hashes of all recorded outputs
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

    progress_path = ROOT / 'work/direct-deepseek/19-progress.json'
    need(progress_path.is_file(), 'progress file missing')
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding='utf8'))
        need(progress.get('number') == 19, 'progress number is not 19')
        need(progress.get('status') == 'review', 'progress status is not review')
        need(progress.get('mesh') == str(out / 'mesh.glb'),
             'progress mesh path is not the actual absolute path')
        need(progress.get('blend') == str(out / 'model.blend'),
             'progress blend path is not the actual absolute path')
        need(progress.get('next_number') == 20, 'progress next_number is not 20')

    summary = {
        'status': 'pass' if not failures else 'fail',
        'out': str(out),
        'parts': topology['parts'],
        'closed_parts': topology['parts'] - len(topology['open_parts']),
        'guard_holes_front': result['openings'].get('front_holes'),
        'guard_holes_back': result['openings'].get('back_holes'),
        'top_components': result['openings'].get('top_components'),
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
            'This candidate does not replace the original model and never enters automatic placement.',
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
