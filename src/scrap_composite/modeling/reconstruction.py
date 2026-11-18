"""Photo reconstruction adapter using the existing offline models."""
import os
from pathlib import Path
from types import SimpleNamespace
import sys

from scrap_composite.data.files import file_hash
from scrap_composite.runtime import BLENDER, ROOT, WEIGHTS, enable_legacy_tools


def photo_to_mesh(source, folder, cache):
    enable_legacy_tools()
    from mesh_build import CHECKPOINT_SHA, build_item
    from scene_v2 import Perception

    weights = WEIGHTS / 'TripoSR/model.ckpt'
    if not weights.is_file() or file_hash(weights) != CHECKPOINT_SHA:
        raise ValueError('TripoSR 权重缺失或散列不符')
    source = Path(source)
    matte, _, _ = Perception(WEIGHTS, cache).matte(source)
    cutout = folder / 'cutout.png'
    matte.save(cutout)
    row = {'rust': str(cutout), 'original': str(source)}
    os.environ.setdefault('HF_HOME', str(ROOT/'work/huggingface'))
    build_item(row, folder, SimpleNamespace(
        python=sys.executable, blender=BLENDER, weights=WEIGHTS,
        cache=cache, vendor=ROOT/'work/vendor/TripoSR'))
    if not row.get('mesh'):
        detail = row.get('reconstruction', {}).get('error', '')
        raise ValueError(row.get('reason', '照片重建没有生成网格') + ('。' + detail if detail else ''))
    info = row['reconstruction']
    return {'path': row['mesh'], 'sha256': file_hash(row['mesh']), 'legacy_axes': True,
            'source_fidelity': info['fidelity'], 'source_camera': info.get('source_camera'),
            'execution': info.get('execution'), 'cache_hit': info.get('cache_hit', False)}
