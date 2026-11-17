"""Local, read-only browser viewer for the 39 review candidates."""
from __future__ import annotations

import argparse
import ctypes
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
import webbrowser

from review_feedback import load_feedback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from display_batch.inventory import is_storage_tank
PARALLEL_ROOT = ROOT.parent / 'parallel-3d-rebuild'
VENDOR = HERE / 'node_modules' / 'three'
SEQUENCE = ROOT / 'work' / 'direct-deepseek' / 'sequence.json'
LIBRARY = ROOT / 'outputs' / '三维模型库-圆润复验' / 'results.jsonl'
PORT = 18765
IDENTITY = 'scrap-model-review-viewer-v1'


class ReviewServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR permits a second listener to bind the same port.
    # An exclusive socket makes repeated launcher clicks reuse the first viewer.
    allow_reuse_address = False

    def server_bind(self):
        if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


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


def load_catalog(*, include_archived=True):
    sequence = json.loads(SEQUENCE.read_text(encoding='utf8'))
    library = {entry['number']: entry for entry in (
        json.loads(line) for line in LIBRARY.read_text(encoding='utf8').splitlines() if line.strip()
    )}
    feedback = load_feedback()
    catalog = {}
    roots = (ROOT, PARALLEL_ROOT)
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
                out = ROOT / 'outputs' / '01阀门-六视图重建-v7'
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


class Handler(BaseHTTPRequestHandler):
    server_version = 'ModelReview/1'

    def log_message(self, *_):
        return

    def _bytes(self, code, content_type, data):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy',
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self'; img-src 'self' blob: data:; connect-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, value):
        self._bytes(code, 'application/json; charset=utf-8',
                    json.dumps(value, ensure_ascii=False).encode('utf8'))

    def _file(self, path, kind=None):
        size = path.stat().st_size
        content_type = kind or mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
        if path.suffix.lower() == '.js':
            content_type = 'text/javascript; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            with path.open('rb') as stream:
                shutil.copyfileobj(stream, self.wfile, length=256 * 1024)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        host = (self.headers.get('Host') or '').split(':')[0].lower()
        if host not in ('127.0.0.1', 'localhost'):
            return self._json(403, {'error': '仅允许本机访问'})
        path = urlsplit(self.path).path
        if path == '/api/health':
            return self._json(200, {'app': IDENTITY})
        if path == '/api/items':
            return self._json(200, [
                {key: value for key, value in item.items() if not key.startswith('_')}
                for item in self.server.catalog.values()
            ])
        if path in ('/', '/index.html'):
            return self._file(HERE / 'review-viewer.html', 'text/html; charset=utf-8')
        if path in ('/review-viewer.css', '/review-viewer.js'):
            return self._file(HERE / path[1:])
        asset = re.fullmatch(r'/api/(model|board|rust)/(\d{1,2}(?:-[1-6])?)', path)
        if asset:
            kind, raw = asset.groups()
            parts = raw.split('-', 1)
            key = str(int(parts[0])) + ('-' + parts[1] if len(parts) > 1 else '')
            if key == '19' and '19-1' in self.server.catalog:
                key = '19-1'
            item = self.server.catalog.get(key)
            if not item or not item['available']:
                return self._json(404, {'error': '模型资料不存在'})
            key = {'model': '_mesh', 'board': '_board', 'rust': '_rust'}[kind]
            try:
                file = local_file(item[key], {'.glb'} if kind == 'model' else {'.jpg', '.jpeg', '.png'},
                                  (ROOT, PARALLEL_ROOT))
                return self._file(file, 'model/gltf-binary' if kind == 'model' else None)
            except (OSError, ValueError):
                return self._json(404, {'error': '文件不可用'})
        if path.startswith('/vendor/'):
            try:
                relative = path[len('/vendor/'):]
                if not relative or '\\' in relative:
                    raise ValueError('Invalid vendor path')
                file = local_file(VENDOR / relative, {'.js'}, (VENDOR,))
                return self._file(file)
            except (OSError, ValueError):
                return self._json(404, {'error': '资源不存在'})
        return self._json(404, {'error': '页面不存在'})


def existing_service(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/health', timeout=1) as response:
            return json.load(response).get('app') == IDENTITY
    except (OSError, ValueError, URLError):
        return False


def notify_error(text):
    if os.name == 'nt' and sys.stderr is None:
        ctypes.windll.user32.MessageBoxW(None, text, '三维模型查看器', 0x10)
    else:
        print(text, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--port', type=int, default=PORT)
    args = parser.parse_args()
    address = f'http://127.0.0.1:{args.port}/'
    try:
        catalog = load_catalog(include_archived=False)
        if not VENDOR.is_dir():
            raise RuntimeError('缺少本地三维显示组件。')
        server = ReviewServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        if isinstance(exc, socket.error) and existing_service(args.port):
            if args.open:
                webbrowser.open(address)
            return 0
        notify_error('查看器无法启动，端口可能被其他程序占用。')
        return 1
    except (RuntimeError, ValueError, json.JSONDecodeError, KeyError):
        notify_error('查看器缺少模型清单或三维显示组件。')
        return 1
    server.catalog = catalog
    if args.open:
        webbrowser.open(address)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
