"""Finalize item 18 result metadata and regenerate the derived records.

This only annotates already-generated evidence (the diagnostic acceptance-gate
flags and the metal-check status), rebuilds the standalone review page, and
recomputes hashes/progress.  It does not rerender, alter the mesh/blend/textures,
or change any pass/fail check.
"""
import json
from pathlib import Path

from scene_v2 import digest
from review18_page import build_page

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/18气压减震器-六视图重建-v1'
RESULT = OUT / 'result.json'
SHARED_CHECKS = {
    'checks_mesh.py': 'passed: 36 tests OK',
    'checks_v2.py': 'passed: 25 tests OK',
    'placement-web/checks_web.py': 'passed: 22 tests OK',
    'checks_rebuild_01.py': ('failed: 2 errors because '
                             'outputs/01阀门-六视图重建-v6 is absent in this worktree; '
                             'unrelated to item 18 and not claimed as passed'),
}


def mark_scores(record):
    strict = record.get('source_fidelity_diagnostic') or {}
    strict['acceptance_gate'] = False
    rolled = record.get('source_fidelity_rolled_diagnostics') or {}
    for entry in rolled.values():
        for key in ('raw', 'bbox_aligned'):
            if isinstance(entry.get(key), dict):
                entry[key]['acceptance_gate'] = False
    best = record.get('source_fidelity_best_bbox_aligned') or {}
    best['acceptance_gate'] = False
    metal = record.get('metal_reflection_check') or {}
    if 'status' not in metal:
        metal['status'] = 'applicable_metal'


def main():
    record = json.loads(RESULT.read_text(encoding='utf8'))
    mark_scores(record)
    record['shared_checks'] = SHARED_CHECKS
    RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
    progress = json.loads((OUT / 'progress.json').read_text(encoding='utf8'))
    progress['source_fidelity_diagnostic'] = record['source_fidelity_diagnostic']
    progress['source_fidelity_best_bbox_aligned'] = \
        record['source_fidelity_best_bbox_aligned']
    progress['shared_checks'] = SHARED_CHECKS
    (OUT / 'progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    external = ROOT / 'work/direct-deepseek/18-progress.json'
    external.write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                        encoding='utf8')
    build_page(OUT)
    files = [path for path in sorted(OUT.rglob('*')) if path.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_18.py',
              'files': {str(path.relative_to(OUT)).replace('\\', '/'): digest(path)
                        for path in files}}
    hashes['files']['hashes.json'] = 'self'
    (OUT / 'hashes.json').write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(OUT),
                      'metal_status': record['metal_reflection_check']['status'],
                      'best_acceptance_gate': record['source_fidelity_best_bbox_aligned']['acceptance_gate']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
