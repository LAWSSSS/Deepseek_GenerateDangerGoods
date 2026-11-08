"""Independent artifact checks for the item 24 review candidate.

This script verifies that the recorded files exist, hashes agree, the exported
GLB is parseable, the recorded structural checks are internally consistent and
the six-view/detail evidence was produced.  It does not replace human visual
review, physical placement, scale validation or formal acceptance.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from mesh_rebuild_item import inspect_mesh

REQUIRED_RENDERS = (
    [f'new-{v}.png' for v in ('front', 'back', 'left', 'right', 'top', 'bottom')] +
    [f'old-{v}.png' for v in ('front', 'back', 'left', 'right', 'top', 'bottom')] +
    [f'turn-{i:02d}.png' for i in range(24)] +
    [f'detail-{n}.png' for n in
     ('front-face', 'rear-face', 'port-front', 'port-rear', 'rod-gland',
      'rod-thread', 'label', 'barrel-profile', 'ea-eb', 'side-screw',
      'overall-3q')] +
    ['caps-only-top.png', 'caps-only-bottom.png', 'front-cap-only.png',
     'rear-cap-only.png', 'rod-only.png', 'barrel-only.png', 'barrel-label.png',
     'new-material-front.png', 'new-clay-front.png', 'new-clay-3q.png',
     'new-light-alternate.png', 'metal-only-light-a.png',
     'metal-only-light-b.png', 'source-alpha-diagnostic.png']
)
REQUIRED_BOARDS = (
    '六视图-新旧对照.jpg', '局部检查.jpg', '分部件隔离.jpg',
    '材质-同光照对照.jpg', '金属反射-变光对照.jpg', '开孔-隔离检查.jpg',
    '原照-新候选对照.jpg', '原视角诊断.jpg', '转台-24方向.jpg',
    '材质图集-布局.jpg',
)
REQUIRED_TEXTURES = ('base_color.png', 'metallic_roughness.png', 'normal.png',
                     'label_base_color.png')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path,
                   default=ROOT / 'outputs/24气缸2-六视图重建-v5')
    a = p.parse_args()
    out = a.out.resolve()
    errors = []
    warnings = []
    result_path = out / 'result.json'
    if not result_path.is_file():
        print(json.dumps({'passed': False, 'errors': ['result.json missing']},
                         ensure_ascii=False, indent=2))
        return 1
    result = json.loads(result_path.read_text(encoding='utf8'))
    required_files = ['mesh.glb', 'model.blend', 'hashes.json',
                      'worker-result.json', 'request.json', 'report.md',
                      '检查页.html', 'source-measurements.json',
                      'old-model-audit.json', 'texture-evidence.json'] + \
        list(REQUIRED_TEXTURES) + list(REQUIRED_BOARDS)
    for name in required_files:
        if not (out / name).is_file():
            errors.append(f'missing {name}')
    for name in REQUIRED_RENDERS:
        if not (out / 'renders' / name).is_file():
            errors.append(f'missing renders/{name}')

    if (out / 'mesh.glb').is_file():
        if digest(out / 'mesh.glb') != result.get('mesh_sha256'):
            errors.append('mesh.glb hash differs from result.json')
        info = inspect_mesh(out / 'mesh.glb')
        if info['parts'] != result['checks'].get('mesh_parts'):
            errors.append('GLB part count differs from result.json')
        if info['closed_after_welding_seams'] != info['parts']:
            errors.append('GLB has open parts after seam welding')
    worker = json.loads((out / 'worker-result.json').read_text(encoding='utf8')) \
        if (out / 'worker-result.json').is_file() else {}
    if worker and worker.get('closed_blender_parts') != len(worker.get('parts') or []):
        errors.append('Blender-side closure check did not cover every part')
    if (out / 'model.blend').is_file():
        if digest(out / 'model.blend') != result.get('blend_sha256'):
            errors.append('model.blend hash differs from result.json')

    checks = result.get('checks') or {}
    for key in ('mesh_all_closed', 'self_intersection_free', 'ports_blind_both',
                'rod_hole_through', 'barrel_grooves_present',
                'texture_checks_passed', 'required_renders_present'):
        if checks.get(key) is not True:
            errors.append(f'check not true: {key} ({checks.get(key)!r})')
    if checks.get('self_intersection_total') not in (0,):
        errors.append('self intersections recorded')
    depths = checks.get('port_depths_m') or {}
    for tag, depth in depths.items():
        if not isinstance(depth, (int, float)) or not 0.010 <= depth <= 0.025:
            errors.append(f'port depth out of range: {tag}={depth}')
    if checks.get('metal_highlights_move') is not True:
        warnings.append('metal highlight check did not pass/execute')
    if checks.get('manual_visual_review') != 'not_performed':
        warnings.append('manual visual review state is not "not_performed"')

    silhouette = result.get('silhouette_diagnostics') or {}
    best = silhouette.get('best') or {}
    if best.get('iou') is None:
        warnings.append('no source silhouette diagnostic IoU recorded')
    if silhouette.get('diagnostic_only') is not True:
        warnings.append('silhouette diagnostic is not marked diagnostic_only')

    hashes_path = out / 'hashes.json'
    if hashes_path.is_file():
        hashes = json.loads(hashes_path.read_text(encoding='utf8'))
        for name, value in hashes.get('outputs', {}).items():
            path = out / name
            if path.is_file() and digest(path) != value:
                errors.append(f'hashes.json output mismatch: {name}')
        for group in ('inputs', 'scripts'):
            for key, row in hashes.get(group, {}).items():
                path = Path(row['path'])
                if path.is_file() and digest(path) != row['sha256']:
                    errors.append(f'hashes.json {group} mismatch: {key}')

    browser = (result.get('browser_check') or {}).get('status')
    if browser not in ('passed', 'failed', 'not_run'):
        warnings.append(f'unexpected browser status: {browser!r}')

    summary = {
        'passed': not errors,
        'errors': errors,
        'warnings': warnings,
        'out': str(out),
        'parts': (info or {}).get('parts'),
        'closed_after_welding_seams': (info or {}).get('closed_after_welding_seams'),
        'self_intersection_total': checks.get('self_intersection_total'),
        'port_depths_m': depths,
        'rod_hole_through': checks.get('rod_hole_through'),
        'barrel_groove_depth_m': checks.get('barrel_groove_depth_m'),
        'best_source_iou_diagnostic': best.get('iou'),
        'best_source_iou_mirrored_diagnostic': best.get('iou_mirrored_reference'),
        'metal_highlights_move': checks.get('metal_highlights_move'),
        'browser_check': browser,
        'manual_visual_review': 'not_performed',
        'note': ('检查通过仅表示文件、哈希、拓扑、射线和贴图证据一致；'
                 '候选仍是 manual_only/待审，不是外观、尺寸、落稳或正式合成验收。'),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
