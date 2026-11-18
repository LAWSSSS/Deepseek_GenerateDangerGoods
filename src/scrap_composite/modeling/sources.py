"""Copy only selected input files into the content-addressed local workspace."""
import json
from pathlib import Path
import shutil
import struct
from tempfile import NamedTemporaryFile

from PIL import Image

from scrap_composite.data.files import file_hash

PHOTO_SUFFIXES = {'.jpeg', '.jpg', '.png', '.webp'}
SOURCE_SUFFIXES = PHOTO_SUFFIXES | {'.blend', '.glb'}
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')


def glb_document(path):
    with Path(path).open('rb') as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError('GLB 文件头不完整')
        magic, version, total, length, kind = struct.unpack('<4sIII4s', header)
        if (magic, version, total, kind) != (b'glTF', 2, Path(path).stat().st_size, b'JSON') or length > total - 20:
            raise ValueError('无法读取 glTF 2 GLB')
        document = json.loads(stream.read(length))
    if not document.get('meshes'):
        raise ValueError('GLB 未包含网格')
    for item in [*document.get('buffers', []), *document.get('images', [])]:
        uri = item.get('uri')
        if uri is not None and not uri.startswith('data:'):
            raise ValueError('GLB 含外部资源，请导出包含依赖的文件')
    return document


def stage(folder, name, stream, *, photo_only=False):
    suffix = Path(name).suffix.lower()
    allowed = PHOTO_SUFFIXES if photo_only else SOURCE_SUFFIXES
    if suffix not in allowed:
        raise ValueError('不支持的建模素材格式')
    folder.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=folder, suffix=suffix, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            shutil.copyfileobj(stream, temporary)
        except BaseException:
            temporary.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        if suffix in PHOTO_SUFFIXES:
            with Image.open(pending) as image:
                image.verify()
        elif suffix == '.glb':
            glb_document(pending)
        elif not pending.stat().st_size:
            raise ValueError('Blender 文件为空')
        digest = file_hash(pending)
        path = folder / (digest + suffix)
        if not path.is_file() or file_hash(path) != digest:
            pending.replace(path)
        return {'path': str(path), 'sha256': digest, 'kind': 'photo' if suffix in PHOTO_SUFFIXES else suffix[1:]}
    finally:
        pending.unlink(missing_ok=True)


def stage_path(folder, path, *, photo_only=False):
    path = Path(path)
    with path.open('rb') as stream:
        return stage(folder, path.name, stream, photo_only=photo_only)


def check_source(source):
    path = Path(source['path'])
    if not path.is_file() or file_hash(path) != source['sha256']:
        raise ValueError('建模素材散列已变化或文件缺失')
    return path
