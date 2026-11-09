"""Regenerate only the text/board/page evidence for the item-39 v5 candidate.

This does not rebuild the mesh or textures; it fixes board layout notes and
adds the honest old-model-material limitation, then refreshes the standalone
page, browser screenshot, result and output hashes.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import mesh_rebuild_item_39 as builder
from review39_page import write_page

ROOT = builder.ROOT
OUT = ROOT / 'outputs/39集油器2-六视图重建-v5'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    result = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
    notes = {
        'limitations': ('The old-model comparison renders use neutral lighting and the '
                        'current Cycles import does not fully reproduce its vertex '
                        'colours; they are geometry comparisons, not an old-material '
                        'acceptance.'),
        'pending_review': ('Use the old-model renders for geometry comparison only; its '
                           'vertex colours are not fully reproduced in the current '
                           'Cycles import.'),
    }
    for key, note in notes.items():
        values = result.setdefault(key, [])
        if note not in values:
            values.append(note)
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding='utf8')

    builder.make_report_39(OUT, builder.REFERENCE_PATH, result['reference_views'])
    row = next(r for r in (json.loads(line) for line in
                           (ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
                           .read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == 39)
    builder.make_source_board(OUT, Path(row['original']), builder.REFERENCE_PATH,
                              result['source_fidelity_diagnostic'])
    builder.make_material_boards(OUT, result['metal_response'],
                                 result['paint_response'])

    tables = {
        'checks': result['checks'],
        'fidelity': result['source_fidelity_diagnostic'],
        'reference_views': result['reference_views'],
        'textures': result['texture_checks'],
        'metal': result['metal_response'],
        'paint': result['paint_response'],
        'openings': result['isolated_opening_checks'],
        'structural': result['structural_checks'],
        'browser': {'status': 'pending'},
        'pending': result['pending_review'],
        'luma': result['luma'],
    }
    page = write_page(OUT, result, tables)
    browser = builder.run_browser(OUT, page)
    result['browser_check'] = browser
    result['checks']['browser_page'] = (
        browser.get('status') == 'headless_chrome_screenshot_ok')
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding='utf8')
    tables['browser'] = browser
    write_page(OUT, result, tables)

    hashes = json.loads((OUT / 'hashes.json').read_text(encoding='utf8'))
    outputs = {}
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name != 'hashes.json':
            outputs[str(path.relative_to(OUT)).replace('\\', '/')] = {
                'sha256': digest(path), 'bytes': path.stat().st_size}
    hashes['outputs'] = outputs
    (OUT / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                     encoding='utf8')

    progress = json.loads((OUT / 'progress.json').read_text(encoding='utf8'))
    progress['checks']['browser'] = browser.get('status')
    progress['pending_review'] = result['pending_review']
    (OUT / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    (ROOT / 'work/direct-deepseek/39-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'patched', 'out': str(OUT), 'browser': browser},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
