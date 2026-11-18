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

from scrap_composite.data.catalog import local_file, load_catalog
from scrap_composite.runtime import CONTROLLER, ROOT

HERE = Path(__file__).with_name('static')
PARALLEL_ROOT = ROOT.parent / 'parallel-3d-rebuild'
VENDOR = HERE / 'node_modules' / 'three'
if not VENDOR.is_dir():
    VENDOR = CONTROLLER / 'placement-web' / 'node_modules' / 'three'
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
            if getattr(self.server, 'refresh_catalog', False):
                self.server.catalog = load_catalog(include_archived=False)
            return self._json(200, [
                {key: value for key, value in item.items() if not key.startswith('_') and key not in {'mesh', 'board', 'rust', 'original', 'source_mesh'}}
                for item in self.server.catalog.values()
            ])
        if path in ('/', '/index.html'):
            return self._file(HERE / 'review-viewer.html', 'text/html; charset=utf-8')
        if path in ('/review-viewer.css', '/review-viewer.js'):
            return self._file(HERE / path[1:])
        asset = re.fullmatch(r'/api/(model|board|rust)/([0-9]{1,9}(?:-[1-6])?)', path)
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
    server.refresh_catalog = True
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
