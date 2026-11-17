"""Create a no-handle revision of item 10 and preserve earlier review decisions."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from user_feedback_rebuild import DECISIONS, ROOT, SEQUENCE, board, compare, glb, run_worker, sha

REMOVED = {'Detached handle bar', 'Handle bar bore liner', 'Handle bar top rim'}
REQUIRED = {'Jack body', 'Cast base', 'Chrome ram', 'Pump handle socket', 'Carry handle loop'}


def node_names(doc):
    return {node['name'] for node in doc.get('nodes', []) if 'mesh' in node and 'name' in node}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('Output must be under the project outputs directory')
    before_bytes = DECISIONS.read_bytes()
    review = json.loads(before_bytes.decode('utf8'))
    if 10 not in review['accepted_numbers'] or '10' in review['needs_revision']:
        raise ValueError('Expected the prior approved item 10 review')
    prior = json.loads(Path(review['revision_manifest']).read_text(encoding='utf8'))
    old_ids = {str(item['id']) for item in prior['items']}
    if old_ids != {'13', '19-1', '19-2', '19-3', '19-4', '19-5', '19-6', '31'}:
        raise ValueError('Prior feedback revisions are incomplete')
    sequence = json.loads(SEQUENCE.read_text(encoding='utf8'))
    source = json.loads(Path(sequence['items']['10']['record']).read_text(encoding='utf8'))
    rows = {r['number']:r for r in (json.loads(line) for line in
            (ROOT / 'outputs/三维模型库-圆润复验/results.jsonl').read_text(encoding='utf8').splitlines())}
    out.mkdir(parents=True, exist_ok=False)
    (out / '审阅结论-更新前.json').write_bytes(before_bytes)
    folder = out / '10-no-bar'
    case = {'id':'10', 'number':10, 'name':'千斤顶（无独立操作杆）', 'kind':'jack',
            'source_blend':source['blend'], 'out':str(folder)}
    run_worker({'out':str(out), 'items':[case], 'summary':'10-worker-summary.json'}, out, '10')
    mesh, blend = folder / 'mesh.glb', folder / 'model.blend'
    evidence = json.loads((folder / 'geometry-check.json').read_text(encoding='utf8'))
    original, revised = glb(Path(source['mesh'])), glb(mesh)
    original_names, revised_names = node_names(original), node_names(revised)
    if not REMOVED.issubset(original_names) or revised_names != original_names - REMOVED:
        raise RuntimeError('The exported GLB does not preserve the original components without the handle')
    if not REQUIRED.issubset(revised_names) or not evidence['retained_geometry_and_transforms_unchanged']:
        raise RuntimeError('The jack body or an essential component changed')
    if any('uri' in a and not str(a['uri']).startswith('data:')
           for a in revised.get('buffers', []) + revised.get('images', [])):
        raise RuntimeError('External GLB resource found')
    board(folder, '10 千斤顶 · 移除独立操作杆')
    compare(Path(source['out']) / 'renders/new-front.png', folder / 'renders/front.png',
            out / '10-修改前后对照.jpg', '10 号千斤顶：去除独立操作杆',
            '修改前：含独立长杆', '修改后：保留千斤顶本体')
    record = {'id':'10', 'number':10, 'name':case['name'], 'quantity':1,
              'status':'revision_pending_review', 'mesh':str(mesh), 'mesh_sha256':sha(mesh),
              'blend':str(blend), 'blend_sha256':sha(blend), 'board':str(folder / '六视图-修订后.jpg'),
              'rust':rows[10]['rust'], 'source_blend_sha256':sha(Path(source['blend'])),
              'evidence':str(folder / 'geometry-check.json')}
    result = {**prior, 'items':[record, *prior['items']],
              'user_accepted_other_numbers':[n for n in prior['user_accepted_other_numbers'] if n != 10],
              'status':'revision_pending_review', 'created':time.strftime('%Y-%m-%d %H:%M:%S')}
    manifest = out / 'manifest.json'
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    review['accepted_numbers'] = [n for n in review['accepted_numbers'] if n != 10]
    review['needs_revision']['10'] = '去除千斤顶旁边独立的操作杆，保留主体的提环、插座和升降柱'
    review['revision_manifest'] = str(manifest)
    review['revision_status'] = '修订完成，待人工复核'
    review['source'] = '用户确认其他物品通过，之后对 10、13、19、31 号提出修订意见'
    review['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    if DECISIONS.read_bytes() != before_bytes:
        raise RuntimeError('Review decisions changed during the rebuild; refusing to overwrite them')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=DECISIONS.parent,
                                     prefix='review-', suffix='.tmp', delete=False) as stream:
        json.dump(review, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        temporary = Path(stream.name)
    os.replace(temporary, DECISIONS)
    print('Revision complete:', folder, '; previous revisions preserved', flush=True)


if __name__ == '__main__':
    main()
