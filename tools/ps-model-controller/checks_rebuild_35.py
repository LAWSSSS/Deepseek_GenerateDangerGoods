"""Integrity and technical checks for the item-35 fire-extinguisher candidate.

This script does not perform visual acceptance or calibrated measurement.  It
verifies that the recorded hashes, files and technical diagnostics agree with
the actual output, and that the candidate is still manual-only and pending
review.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 35
VERSION = 'extinguisher35-parametric-v1'
REFERENCE_SHA256 = '3a706420374349abea3a8cff8b8960f94d4287d0aab4ccf94346408b49348292'
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
DETAILS = ('gauge', 'handle', 'nozzle', 'label', 'bottom', 'top', 'seam', 'overall-3q')
ISOLATIONS = ('body-only', 'label-only', 'valve-only', 'gauge-only', 'handle-only',
              'nozzle-only', 'nozzle-through', 'safety-ring', 'top-parts',
              'new-material-front', 'new-clay-front', 'new-material-alternate',
              'metal-light-a', 'metal-light-b')
EXPECTED_PARTS = (
    '35 body', '35 lower rolled seam', '35 upper rolled seam', '35 bottom lip',
    '35 valve collar', '35 valve body', '35 valve bottom flange', '35 valve top nut',
    '35 gauge housing', '35 nozzle boss', '35 chrome gauge bezel', '35 gauge dial',
    '35 gauge needle outline', '35 gauge needle', '35 gauge glass',
    '35 gauge chrome centre hub', '35 gauge centre screw', '35 gauge screw slot',
    '35 handle bracket', '35 pivot boss', '35 chrome pivot pin',
    '35 lower handle', '35 upper handle lever', '35 safety pin',
    '35 red tamper seal', '35 red seal tag', '35 nozzle horn', '35 nozzle collar',
    '35 label',
)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def topology(path):
    scene = trimesh.load(path, force='scene')
    open_parts, nonfinite = [], []
    for node in scene.graph.nodes_geometry:
        _, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not np.isfinite(mesh.vertices).all():
            nonfinite.append(node)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        if not mesh.is_watertight:
            open_parts.append(node)
    return {'parts': len(scene.graph.nodes_geometry),
            'open_parts': open_parts, 'nonfinite_parts': nonfinite,
            'bounds': scene.bounds.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path,
                   default=ROOT / 'outputs/35灭火器2-六视图重建-v4')
    a = p.parse_args()
    out = a.out.resolve()
    failures = []

    def need(condition, message):
        if not condition:
            failures.append(message)

    for name in ('result.json', 'hashes.json', 'request.json', 'worker-result.json',
                 'old-model-audit.json', 'source-measurements.json', 'progress.json',
                 'report.md', 'index.html'):
        need((out / name).is_file(), f'required file missing: {name}')
    if failures:
        print(json.dumps({'status': 'fail', 'failures': failures}, ensure_ascii=False,
                         indent=2))
        return 1
    result = json.loads((out / 'result.json').read_text(encoding='utf8'))
    hashes = json.loads((out / 'hashes.json').read_text(encoding='utf8'))
    worker = json.loads((out / 'worker-result.json').read_text(encoding='utf8'))

    need(result.get('number') == NUMBER, 'number is not 35')
    need(result.get('version') == VERSION, 'version differs')
    need(result.get('manual_only') is True, 'candidate must remain manual_only')
    need(result.get('status') == 'review', 'status must remain review')
    need(result.get('model_review') == 'pending', 'model_review must be pending')
    need(result.get('scene_validation') == 'pending', 'scene_validation must be pending')
    need(result.get('reference', {}).get('role') == 'visual_match_unverified',
         'six-view reference role is not visual_match_unverified')
    need(result.get('reference', {}).get('sha256') == REFERENCE_SHA256,
         'reference hash differs')
    need(result.get('checks', {}).get('source_fidelity_acceptance') is False,
         'source fidelity must not be claimed as accepted')
    need(result.get('checks', {}).get('browser_review_executed') is False,
         'browser review must not be claimed as executed')
    need(result.get('checks', {}).get('scene_validation') is False,
         'scene validation must not be claimed as executed')
    need(result.get('checks', {}).get('settle_validation') is False,
         'settle validation must not be claimed as executed')
    need(result.get('checks', {}).get('measurement_calibration') is False,
         'measurement calibration must not be claimed')
    need(result.get('source_fidelity_diagnostic', {}).get('acceptance_gate') is False,
         'source fidelity diagnostic must not be an acceptance gate')
    need(result.get('source_fidelity_aligned_diagnostic', {}).get('acceptance_gate') is False,
         'aligned diagnostic must not be an acceptance gate')

    mesh_path = out / 'mesh.glb'
    blend_path = out / 'model.blend'
    need(mesh_path.is_file(), 'mesh.glb missing')
    need(blend_path.is_file(), 'model.blend missing')
    if mesh_path.is_file() and result.get('mesh_sha256'):
        need(digest(mesh_path) == result['mesh_sha256'], 'mesh hash differs from result.json')
    if blend_path.is_file() and result.get('blend_sha256'):
        need(digest(blend_path) == result['blend_sha256'], 'blend hash differs from result.json')
    if mesh_path.is_file():
        topology_record = topology(mesh_path)
        need(not topology_record['open_parts'], 'open mesh parts: ' + str(topology_record['open_parts']))
        need(not topology_record['nonfinite_parts'],
             'nonfinite mesh parts: ' + str(topology_record['nonfinite_parts']))
        need(topology_record['parts'] == len(EXPECTED_PARTS),
             f"part count {topology_record['parts']} != {len(EXPECTED_PARTS)}")
        scene = trimesh.load(mesh_path, force='scene')
        names = list(scene.graph.nodes_geometry)
        need(set(EXPECTED_PARTS).issubset(set(names)), 'expected structural part missing')
        need(result['mesh_audit']['parts'] == topology_record['parts'],
             'recorded part count differs from actual mesh')
        need(result['mesh_audit']['bounds'] == topology_record['bounds'],
             'recorded bounds differ from actual mesh')
        need(not result['worker']['self_intersecting_components'],
             'same-part self intersections recorded')

    for name in ('base_color.png', 'metallic_roughness.png', 'normal.png',
                 'label_base.png', 'label_mr.png', 'label_normal.png',
                 'dial_base.png', 'dial_mr.png', 'dial_normal.png',
                 '六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                 '材质-同光照对照.jpg', '材质-变光检查.jpg',
                 '表面反射-变光对照.jpg', '转台-24方向.jpg',
                 '材质图集-布局.jpg', '展签-表盘-贴图.jpg', '原照-新候选对照.jpg'):
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

    need(result['checks'].get('mesh_closed_after_welding') is True,
         'recorded mesh closure is not true')
    need(result['checks'].get('same_part_self_intersections') is True,
         'recorded self-intersection state is not true')
    need(result['isolated_opening_checks'].get('passed') is True,
         'geometry opening checks did not pass')
    need(result['isolated_opening_checks']['nozzle_bore_ray']['passed'] is True,
         'nozzle bore ray check did not pass')
    need(result['isolated_opening_checks']['safety_pin_ring']['passed'] is True,
         'safety pin ring check did not pass')
    need(result['isolated_opening_checks']['concave_bottom']['passed'] is True,
         'concave bottom check did not pass')
    need(result['isolated_opening_render_checks'].get('passed') is True,
         'isolated opening render checks did not pass')
    need(result['isolated_opening_render_checks']['nozzle_through_interior_holes'] == 1,
         'nozzle-through render does not show exactly one hole')
    need(result['isolated_opening_render_checks']['safety_ring_interior_holes'] == 1,
         'safety-ring render does not show exactly one hole')
    need(result['texture_checks'].get('passed') is True, 'atlas texture checks did not pass')
    need(result['label_texture_check'].get('passed') is True, 'label texture check did not pass')
    need(result['dial_texture_check'].get('passed') is True, 'dial texture check did not pass')
    need(result['metal_reflection_check'].get('passed') is True,
         'metal reflection check did not pass')
    need(float(result['metal_reflection_check'].get('shift_px') or 0) >= 2.0,
         'metal highlight shift is below 2 px')
    need(result['structural_checks']['counts'].get('body') == 1, 'body count differs')
    need(result['structural_checks']['counts'].get('rolled_seams') == 2,
         'rolled seam count differs')
    need(result['structural_checks']['counts'].get('handle_bars') == 2,
         'handle bar count differs')
    need(result['structural_checks']['counts'].get('gauge_parts') == 9,
         'gauge part count differs')
    need(result['structural_checks']['counts'].get('nozzle_parts') == 3,
         'nozzle part count differs')

    for key in ('inputs', 'outputs', 'renders'):
        need(key in hashes, f'hashes.json missing {key}')
    if 'outputs' in hashes:
        for name, expected in hashes['outputs'].items():
            path = out / name
            if path.is_file():
                need(digest(path) == expected, f'output hash differs: {name}')
    for key, expected in hashes.get('inputs', {}).items():
        need(isinstance(expected, str) and len(expected) == 64,
             f'input hash malformed: {key}')
    progress = json.loads((out / 'progress.json').read_text(encoding='utf8'))
    need(progress.get('number') == NUMBER, 'progress number differs')
    need(progress.get('status') == 'review', 'progress status differs')
    need(progress.get('next_number') == 36, 'progress next_number differs')
    need(progress.get('manual_only') is True, 'progress manual_only differs')

    summary = {'status': 'pass' if not failures else 'fail',
               'out': str(out), 'number': NUMBER, 'version': VERSION,
               'manual_only': True, 'review': 'pending',
               'failures': failures,
               'checks': result['checks'],
               'source_fidelity_diagnostic': result['source_fidelity_diagnostic'],
               'mesh_audit': result['mesh_audit']}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
