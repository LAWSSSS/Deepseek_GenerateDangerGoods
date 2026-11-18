"""Versioned local modeling library and its durable build queue."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from time import time
from uuid import uuid4

from scrap_composite.runtime import ROOT
from .files import file_hash
from .inventory import is_storage_tank

WORKSPACE = ROOT / 'work/modeling'
REQUIRED_ARTIFACTS = {'render', 'preview', 'board', 'verification'}


def validate_artifacts(record, directory):
    artifacts = record.get('artifacts', {})
    if set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError('候选缺少模型、预览或核验记录')
    for item in artifacts.values():
        path = Path(item['path']).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != item['sha256']:
            raise ValueError('候选文件路径或散列与记录不符')


class Registry:
    def __init__(self, root=WORKSPACE):
        self.root = Path(root).resolve()
        self.path = self.root / 'registry.sqlite3'

    @contextmanager
    def connect(self, *, write=False):
        if write:
            self.root.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=30)
        else:
            db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS assets (
                        id TEXT PRIMARY KEY, number INTEGER NOT NULL, name TEXT NOT NULL, published TEXT
                    );
                    CREATE TABLE IF NOT EXISTS builds (
                        id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, created REAL NOT NULL, record TEXT NOT NULL
                    );
                """)
                db.execute('BEGIN IMMEDIATE')
            with db:
                yield db
        finally:
            db.close()

    def assets(self):
        if not self.path.is_file():
            return {}
        with self.connect() as db:
            return {row['id']: dict(row) for row in db.execute('SELECT * FROM assets ORDER BY number, id')}

    def create(self, name, request, *, asset=None):
        ident = uuid4().hex
        with self.connect(write=True) as db:
            if asset is None:
                number = db.execute('SELECT max(number) FROM assets').fetchone()[0] or 39
                number = max(39, number) + 1
                asset = {'id': str(number), 'number': number, 'name': name}
            db.execute('INSERT OR IGNORE INTO assets(id,number,name) VALUES (:id,:number,:name)', asset)
            row = db.execute('SELECT id,number,name FROM assets WHERE id=?', (asset['id'],)).fetchone()
            record = {'id': ident, 'asset': dict(row), 'request': request, 'state': 'queued',
                      'stage': 'queued', 'created': time(), 'checks': {}}
            db.execute('INSERT INTO builds VALUES (?, ?, ?, ?)',
                       (ident, row['id'], record['created'], json.dumps(record, ensure_ascii=False)))
        return record

    def get(self, ident):
        if not self.path.is_file():
            raise ValueError('建模任务不存在')
        with self.connect() as db:
            row = db.execute('SELECT record FROM builds WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise ValueError('建模任务不存在')
        return json.loads(row[0])

    def builds(self, asset_id):
        if not self.path.is_file():
            return []
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                'SELECT record FROM builds WHERE asset_id=? ORDER BY created DESC', (asset_id,))]

    def save(self, record):
        with self.connect(write=True) as db:
            db.execute('UPDATE builds SET record=? WHERE id=?',
                       (json.dumps(record, ensure_ascii=False, allow_nan=False), record['id']))

    def directory(self, ident):
        if len(ident) != 32 or any(char not in '0123456789abcdef' for char in ident):
            raise ValueError('建模任务编号无效')
        return self.root / 'builds' / ident

    def publish(self, asset_id, ident):
        record = self.get(ident)
        if record['asset']['id'] != asset_id or record['state'] != 'ready':
            raise ValueError('只有该物品完整的建模候选可以发布')
        validate_artifacts(record, self.directory(ident))
        with self.connect(write=True) as db:
            db.execute('UPDATE assets SET published=? WHERE id=?', (ident, asset_id))
        return record


def published_catalog(workspace=WORKSPACE, *, include_archived=True):
    registry = Registry(workspace)
    catalog = {}
    for asset_id, asset in registry.assets().items():
        ident = asset['published']
        archived = is_storage_tank(asset['number'])
        if not ident or (archived and not include_archived):
            continue
        record = registry.get(ident)
        artifacts = record['artifacts']
        mesh = Path(artifacts['render']['path'])
        preview = Path(artifacts['preview']['path'])
        board = Path(artifacts['board']['path'])
        available = all(path.is_file() for path in (mesh, preview, board))
        original = record['request'].get('identity_photo')
        catalog[asset_id] = {
            'id': asset_id, 'number': asset['number'], 'name': asset['name'], 'available': available,
            'quantity': 1, 'human_review': 'published', 'version': ident,
            'reason': '' if available else '发布版本文件缺失',
            '_mesh': preview, '_board': board, '_rust': Path(original or board),
            'mesh': str(mesh), 'mesh_hash': artifacts['render']['sha256'],
            'source_mesh': str(preview), 'source_mesh_hash': artifacts['preview']['sha256'],
            'packed_subject': record['packed_subject'],
            'board': str(board), 'board_hash': artifacts['board']['sha256'],
            'original': original, 'original_hash': record['request'].get('identity_sha256'),
            'rust': str(original or board),
            'dimension': record['request'].get('dimension'),
            'checks': record.get('checks', {}),
            'archived': archived,
        }
    return catalog
