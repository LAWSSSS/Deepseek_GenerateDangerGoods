"""Read the user's review and validate the locally generated revised models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DECISIONS = ROOT / 'outputs/人工逐件审阅-20261109/用户审阅结论.json'
EXPECTED_BY_NUMBER = {10: {'10'}, 13: {'13'}, 19: {f'19-{i}' for i in range(1, 7)}, 31: {'31'}}
EXPECTED = set().union(*EXPECTED_BY_NUMBER.values())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def inside_outputs(raw, suffixes):
    path = Path(raw).resolve(strict=True)
    base = (ROOT / 'outputs').resolve(strict=True)
    if not path.is_relative_to(base) or not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError('Invalid revision file')
    return path


def load_feedback():
    result = {'approved': set(), 'requested': set(), 'revisions': {}, 'error': None}
    if not DECISIONS.is_file():
        return result
    try:
        review = json.loads(DECISIONS.read_text(encoding='utf8'))
        approved = {int(x) for x in review.get('accepted_numbers', [])}
        requested = {int(x) for x in review.get('needs_revision', {})}
        if not approved.issubset(set(range(1, 40))) or not requested.issubset(EXPECTED_BY_NUMBER) or approved & requested:
            raise ValueError('Inconsistent review decisions')
        result.update({'approved':approved, 'requested':requested})
        if not review.get('revision_manifest'):
            return result
        manifest_path = inside_outputs(review['revision_manifest'], {'.json'})
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        revisions = {}
        for record in manifest.get('items', []):
            key = str(record['id'])
            if key not in EXPECTED or key in revisions or record.get('quantity') != 1:
                raise ValueError('Unexpected revision identity or quantity')
            number = int(key.split('-', 1)[0])
            if record.get('number') != number:
                raise ValueError('Incorrect revision number')
            mesh = inside_outputs(record['mesh'], {'.glb'})
            board = inside_outputs(record['board'], {'.jpg', '.jpeg', '.png'})
            rust = inside_outputs(record['rust'], {'.jpg', '.jpeg', '.png'})
            blend = inside_outputs(record['blend'], {'.blend'})
            if digest(mesh) != record['mesh_sha256'] or digest(blend) != record['blend_sha256']:
                raise ValueError('Revision hash mismatch')
            revisions[key] = {**record, 'mesh':mesh, 'board':board, 'rust':rust, 'blend':blend}
        expected = set().union(*(EXPECTED_BY_NUMBER[number] for number in requested))
        if set(revisions) != expected:
            raise ValueError('Revision set is incomplete')
        result['revisions'] = revisions
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        result['error'] = '用户反馈记录或修订模型不完整'
        result['revisions'] = {}
    return result
