"""Freeze the visible asset version and image bytes when a batch is submitted."""
import json
from pathlib import Path

from .files import atomic_json, file_hash, signature

FILENAME = 'input-snapshot.json'
VERSION = 'batch-inputs-v1'


def snapshot(config, jobs, catalog):
    selected = {job['asset_id'] for job in jobs}
    rows = {key: row for key, row in catalog.items() if key in selected}
    bindings = {}
    for key, row in rows.items():
        files = {}
        for name, digest_key in (('mesh','mesh_hash'), ('board','board_hash'), ('original','original_hash'),
                                 ('source_mesh','source_mesh_hash')):
            path = row.get(name)
            if not path:
                continue
            observed = file_hash(path)
            if row.get(digest_key) and observed != row[digest_key]:
                raise ValueError(f'模型 {key} 的 {name} 与已发布的散列不符')
            files[str(path)] = observed
        bindings[key] = files
    return {'version': VERSION, 'config': config_signature(config), 'jobs': jobs,
            'catalog': rows, 'asset_files': bindings}


def config_signature(config):
    return signature(config.model_dump(exclude={'offline'}))


def read_snapshot(folder, config):
    path = Path(folder) / FILENAME
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding='utf8'))
    if record.get('version') != VERSION or record.get('config') != config_signature(config):
        raise ValueError('批次输入或配置已变化，请使用新的输出目录')
    return record


def save_snapshot(folder, record):
    atomic_json(Path(folder) / FILENAME, record)


def mismatched_inputs(record, job=None):
    def changed(path, digest):
        return not Path(path).is_file() or file_hash(path) != digest

    jobs = (job,) if job is not None else record['jobs']
    backgrounds = {entry['background']: entry['background_hash']
                   for entry in jobs if entry.get('background_hash')}
    for path, digest in backgrounds.items():
        if changed(path, digest):
            return f'背景文件已变更：{Path(path).name}'
    assets = record['asset_files']
    bindings = assets.items() if job is None else [(job['asset_id'], assets.get(job['asset_id'], {}))]
    for key, paths in bindings:
        if any(changed(path, digest) for path, digest in paths.items()):
            return f'已锁定的物品 {key} 文件已变更'
    return None
