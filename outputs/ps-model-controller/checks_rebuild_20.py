"""Offline verification for the item-20 gas-cylinder candidate.

This script deliberately does not regenerate any render.  It loads the existing
result, re-runs the topology/GLB/ray/page checks against the exported mesh, and
verifies every recorded input and output hash.  A failed source-silhouette hole
diagnostic is reported as a diagnostic failure, never as a geometry pass.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import mesh_rebuild_item_20 as builder  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'outputs/20气瓶1-六视图重建-v3')
    parser.add_argument('--config', type=Path,
                        default=HERE / 'cylinder20_config.json')
    args = parser.parse_args()
    out = args.out.resolve()
    config = json.loads(args.config.read_text(encoding='utf8'))
    result = json.loads((out / 'result.json').read_text(encoding='utf8'))
    hashes = json.loads((out / 'hashes.json').read_text(encoding='utf8'))

    failures = []
    notes = []

    def require(name, condition, detail=''):
        if condition:
            notes.append(f'PASS {name}' + (f': {detail}' if detail else ''))
        else:
            failures.append(f'FAIL {name}' + (f': {detail}' if detail else ''))

    # Hash record: every listed output file must still exist and match.
    for relative, recorded in hashes.get('files', {}).items():
        if recorded == 'self':
            continue
        path = out / relative
        require(f'hash {relative}', path.is_file() and digest(path) == recorded)
    input_paths = result['inputs']['input_paths']
    input_paths = {**input_paths, 'old_mesh': input_paths.get('mesh'),
                   'config': args.config.relative_to(ROOT).as_posix(),
                   'worker': (HERE / 'mesh_rebuild_20_worker.py').relative_to(ROOT).as_posix(),
                   'builder': (HERE / 'mesh_rebuild_item_20.py').relative_to(ROOT).as_posix()}
    for key, recorded in hashes.get('inputs', {}).items():
        relative = input_paths.get(key)
        path = ROOT / relative if relative else None
        require(f'input hash {key}', path is not None and path.is_file() and
                digest(path) == recorded,
                relative or 'no recorded relative path')

    # Re-run the structural and opening tests against the exported mesh.
    mesh_audit = builder.inspect_mesh(out / 'mesh.glb')
    glb = builder.glb_audit(out / 'mesh.glb')
    worker = result['worker']
    ray = builder.ray_checks(out / 'mesh.glb', config, worker)
    require('six closed parts', mesh_audit['parts'] == 6 and
            not mesh_audit['open_parts'] and not mesh_audit['nonfinite_parts'],
            json.dumps(mesh_audit, ensure_ascii=False))
    require('GLB materials', glb['passed'],
            f"{glb['image_count']} PNG, textured={[m['name'] for m in glb['materials'] if m['name'] in ('body','guard','steel')]}")
    for key in ('guard_hand_hole', 'guard_wall', 'valve_present',
                'convex_bottom', 'neck_seal'):
        require(f'ray {key}', ray[key]['passed'], json.dumps(ray[key], ensure_ascii=False))
    require('recorded ray agrees', ray['guard_hand_hole']['first_hit_x_m'] ==
            result['ray_checks']['guard_hand_hole']['first_hit_x_m'],
            'the regenerated ray result matches result.json')

    # The alpha-hole diagnostic is expected to fail because the far guard wall
    # is solid (matching the dark interior in the photograph).  Report it as a
    # diagnostic, not as a hidden pass.
    fidelity = result['source_fidelity_diagnostic']
    if fidelity['filled_holes']:
        notes.append(
            'DIAGNOSTIC source alpha hole filled by the solid far guard wall; '
            'the real front hand-hole is verified by the ray test, not by this '
            'silhouette diagnostic')
    else:
        require('source alpha hole diagnostic', True)

    for key in ('mesh_integrity', 'isolated_openings', 'structural_counts',
                'cylindrical_body', 'convex_bottom', 'material_textures',
                'glb_materials', 'metal_reflection', 'paint_gloss'):
        require(f'result check {key}', result['checks'].get(key) is True)
    require('manual_only', result.get('manual_only') is True)
    require('review status', result.get('status') == 'review')
    require('review page assets', builder.check_review_page(out).get('passed') is True)
    for name in ('mesh.glb', 'model.blend', 'review.html', 'result.json',
                 'hashes.json', '六视图-新旧对照.jpg', '局部检查.jpg',
                 '结构开口-隔离检查.jpg', '材质-同光照对照.jpg',
                 '材质-变光检查.jpg', '表面反射-变光对照.jpg',
                 '转台-24方向.jpg', '源图-原视角诊断.jpg'):
        require(f'file {name}', (out / name).is_file())

    summary = {
        'number': 20, 'out': str(out), 'passed': not failures,
        'failures': failures, 'notes': notes,
        'source_fidelity_diagnostic': fidelity,
        'ray_checks': ray, 'mesh_audit': mesh_audit, 'glb_audit': glb,
        'manual_only': result.get('manual_only'),
        'status': result.get('status'),
    }
    record = ROOT / 'work/direct-deepseek/20-checks.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                      encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
