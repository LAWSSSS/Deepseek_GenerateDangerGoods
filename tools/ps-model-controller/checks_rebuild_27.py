"""Integrity and technical checks for the item-27 rear-shock candidate.

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
NUMBER = 27
VERSION = 'shock27-parametric-v1'
REFERENCE_SHA256 = '059cc68bde6adbd04c2cf95d5dea603b97a16db02c58552c8f12ee0244e3f2c7'
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
DETAILS = ('top-eye', 'top-bolt', 'label', 'spring-upper', 'spring-lower',
           'bottom-clevis', 'hose-top', 'hose-run', 'reservoir-banjo',
           'reservoir-cap', 'overall-3q')
ISOLATIONS = ('top-eye-only', 'bottom-clevis-only', 'spring-only',
              'reservoir-only', 'hose-only', 'new-material-front',
              'new-clay-front', 'new-light-alternate', 'metal-only-light-a',
              'metal-only-light-b')


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
                   default=ROOT / 'outputs/27液压减震器-六视图重建-v1')
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

    need(result.get('number') == NUMBER, 'number is not 27')
    need(result.get('version') == VERSION, 'version differs')
    need(result.get('manual_only') is True, 'candidate must remain manual_only')
    need(result.get('status') == 'review', 'status must remain review')
    need(result.get('draft') is False, 'final result must not be a draft')
    need(result.get('reference', {}).get('role') == 'visual_match_unverified',
         'six-view reference role is not visual_match_unverified')
    need(result.get('reference', {}).get('sha256') == REFERENCE_SHA256,
         'reference hash differs from the registered value')
    checks = result.get('checks', {})
    need(checks.get('browser_review_executed') is False,
         'browser review must not be claimed as executed')
    need(checks.get('source_fidelity_acceptance') is False,
         'source fidelity must not be claimed as accepted')
    need(checks.get('nonmetal_metal_reflection_applicable') is False,
         'non-metal metal-reflection applicability must be false')
    for key in ('mesh_closed', 'no_nonfinite', 'no_self_intersections',
                'texture_values', 'top_eye_through', 'bottom_eye_through',
                'two_bushings_present', 'two_bottom_bolts', 'spring_open_helix',
                'reservoir_socket', 'twelve_knurls', 'metal_highlights_move'):
        need(key in checks, f'check flag missing: {key}')

    for key, name in (('mesh', 'mesh.glb'), ('blend', 'model.blend')):
        path = out / name
        need(path.is_file(), f'{key} file missing')
        expected = result.get(f'{key}_sha256')
        if path.is_file() and expected:
            need(digest(path) == expected, f'{key} hash differs from result.json')
    for name in ('base_color.png', 'metallic_roughness.png', 'normal.png',
                 'label_cap.png', 'label_band.png',
                 '六视图-新旧对照.jpg', '局部检查.jpg', '分部件隔离.jpg',
                 '材质-同光照对照.jpg', '金属反射-变光对照.jpg',
                 '开孔-隔离检查.jpg', '原视角诊断.jpg', '原照-新候选对照.jpg',
                 '转台-24方向.jpg', '材质图集-对照.jpg', 'report.md',
                 'source-measurements.json', 'old-model-audit.json',
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
             f'missing turntable render {i:02d}')
    for name in ('label_cap.png', 'label_band.png'):
        path = out / name
        if path.is_file():
            im = Image.open(path)
            need(im.width >= 512 and im.height >= 96, f'{name} too small')

    topo = mesh_topology(out / 'mesh.glb')
    need(not topo['open_parts'], f"open mesh parts: {topo['open_parts']}")
    need(not topo['nonfinite_parts'], f"non-finite mesh parts: {topo['nonfinite_parts']}")
    need(topo['parts'] == len(result.get('worker', {}).get('parts', [])),
         'GLB part count differs from worker record')
    need(result.get('mesh', {}).get('parts') == topo['parts'],
         'result part count differs from GLB')
    need(result.get('mesh', {}).get('parts', 0) >= 25,
         'unexpectedly few parts for the part-based rebuild')

    outputs = hashes.get('outputs', {})
    mesh_rel = str((out / 'mesh.glb').relative_to(ROOT)).replace('\\', '/')
    blend_rel = str((out / 'model.blend').relative_to(ROOT)).replace('\\', '/')
    need(mesh_rel in outputs, 'mesh.glb missing from hashes.json outputs')
    need(blend_rel in outputs, 'model.blend missing from hashes.json outputs')
    if mesh_rel in outputs:
        need(outputs[mesh_rel]['sha256'] == result.get('mesh_sha256'),
             'hashes.json mesh hash differs from result.json')
    need(result.get('checks', {}).get('top_eye_through') is True,
         'top eye through-hole check did not pass')
    need(result.get('checks', {}).get('bottom_eye_through') is True,
         'bottom clevis through-hole check did not pass')
    need(result.get('checks', {}).get('spring_open_helix') is True,
         'spring open-helix check did not pass')
    need(result.get('checks', {}).get('reservoir_socket') is True,
         'reservoir end-socket check did not pass')
    need(result.get('checks', {}).get('texture_values') is True,
         'texture value checks did not pass')

    summary = {
        'status': 'pass' if not failures else 'fail',
        'number': NUMBER,
        'version': VERSION,
        'out': str(out),
        'mesh_sha256': result.get('mesh_sha256'),
        'blend_sha256': result.get('blend_sha256'),
        'parts': topo['parts'],
        'checks': result.get('checks'),
        'failures': failures,
        'note': ('完整性/技术检查只确认文件、哈希、网格闭合和记录一致；'
                 '不构成视觉验收、尺寸测量或正式合成许可。'),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
