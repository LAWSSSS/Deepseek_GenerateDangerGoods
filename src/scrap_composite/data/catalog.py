"""Resolve published and historical assets without depending on the web app."""
import json
from pathlib import Path

from scrap_composite.runtime import ROOT
from .files import file_hash, read_jsonl
from .inventory import is_storage_tank
from .legacy_feedback import load_feedback

LIBRARY = Path('outputs/三维模型库-圆润复验/results.jsonl')


def _allowed(path, base):
    try:
        return path.is_relative_to(base)
    except (ValueError, OSError):
        return False


def local_file(raw, suffixes, roots):
    path = Path(raw).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError('Unexpected file type')
    if not any(_allowed(path, root.resolve()) for root in roots):
        raise ValueError('File is outside the project')
    return path


def legacy_catalog(root, *, include_archived=True):
    sequence_path = root / 'work/direct-deepseek/sequence.json'
    sequence = json.loads(sequence_path.read_text(encoding='utf8')) if sequence_path.is_file() else {}
    library = {row['number']: row for row in read_jsonl(root / LIBRARY)}
    feedback = load_feedback(root=root)
    catalog = {}
    roots = (root, root.parent / 'parallel-3d-rebuild')
    for number in range(1, 40):
        if not include_archived and is_storage_tank(number):
            continue
        source = library.get(number, {})
        key = str(number)
        entry = {'id': key, 'number': number, 'name': Path(source.get('file', f'{number:02}')).stem,
                 'available': False, 'reference_missing': number == 37,
                 'human_review': 'approved' if number in feedback['approved'] else 'pending',
                 'reason': ''}
        if number in feedback['requested']:
            revised = ([f'19-{index}' for index in range(1, 7)] if number == 19 else [key])
            if not all(item in feedback['revisions'] for item in revised):
                entry['reason'] = '用户要求的修订模型尚未就绪'
                catalog[key] = entry
                continue
            for item_id in revised:
                item = feedback['revisions'][item_id]
                catalog[item_id] = {**entry, 'id':item_id, 'name':item['name'], 'available':True,
                                    'human_review':'revision_pending_review', 'quantity':1,
                                    '_mesh':item['mesh'], '_board':item['board'], '_rust':item['rust']}
            continue
        try:
            if number == 1:
                out = root / 'outputs' / '01阀门-六视图重建-v7'
                mesh = out / 'mesh.glb'
            else:
                item = sequence['items'][f'{number:02}']
                progress_path = local_file(item['record'], {'.json'}, roots)
                progress = json.loads(progress_path.read_text(encoding='utf8'))
                if int(progress.get('number', -1)) != number:
                    raise ValueError('Incorrect item number')
                out = Path(progress['out']).resolve(strict=True)
                if not any(_allowed(out, root.resolve()) for root in roots):
                    raise ValueError('Output is outside the project')
                mesh = Path(progress['mesh'])
            mesh = local_file(mesh, {'.glb'}, roots)
            board = local_file(out / '六视图-新旧对照.jpg', {'.jpg', '.jpeg', '.png'}, roots)
            rust = local_file(source['rust'], {'.jpg', '.jpeg', '.png'}, roots)
            with mesh.open('rb') as stream:
                if stream.read(4) != b'glTF':
                    raise ValueError('Unexpected GLB header')
            entry.update({'available': True, '_mesh': mesh, '_board': board, '_rust': rust})
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            entry['reason'] = '模型文件或对照资料未找到'
        catalog[key] = entry
    return catalog



def load_catalog(*, root=ROOT, include_archived=True, workspace=None):
    root = Path(root)
    rows = legacy_catalog(root, include_archived=include_archived)
    library = {row['number']: row for row in read_jsonl(root / LIBRARY)}
    for row in rows.values():
        row['original'] = library.get(row['number'], {}).get('original')
        for key in ('mesh', 'rust', 'board'):
            row[key] = str(row.get('_' + key, ''))
        row['mesh_hash'] = file_hash(row['mesh']) if row['available'] else None
    from .registry import published_catalog
    rows.update(published_catalog(workspace or root / 'work/modeling', include_archived=include_archived))
    return rows


def mesh_input(row):
    request = {'mesh': row['mesh']}
    if row.get('packed_subject'):
        request.update(packed_subject=row['packed_subject'], mesh_hash=row['mesh_hash'])
    return request
