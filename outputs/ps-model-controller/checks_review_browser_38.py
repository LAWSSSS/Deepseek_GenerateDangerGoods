"""Headless-browser check for the independent item-38 review page.

A temporary localhost static server serves the generated item-38 output
directory.  Headless Chrome is launched with a DevTools endpoint and the page's
own selftest is polled until it reports old/new switching, clay/material
switching, rotation and a canvas luma sample.  This is a local compatibility
check only: it does not use the shared placement server, does not change shared
pages, and does not prove model or scene acceptance.
"""
import base64
import json
import os
import socket
import struct
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests


CHROME_CANDIDATES = (
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
)


def find_browser():
    for path in CHROME_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


class WebSocket:
    def __init__(self, url, timeout=15):
        parsed = urlparse(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port or 80),
                                             timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (f'GET {parsed.path} HTTP/1.1\r\n'
                   f'Host: {parsed.hostname}:{parsed.port}\r\n'
                   'Upgrade: websocket\r\nConnection: Upgrade\r\n'
                   f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n')
        self.sock.sendall(request.encode())
        response = b''
        while b'\r\n\r\n' not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError('websocket handshake closed')
            response += chunk
        if b' 101 ' not in response.split(b'\r\n')[0]:
            raise ConnectionError(response.split(b'\r\n')[0].decode('latin1'))
        self.sock.settimeout(timeout)

    def _recv_exact(self, count):
        buffer = b''
        while len(buffer) < count:
            chunk = self.sock.recv(count - len(buffer))
            if not chunk:
                raise EOFError('websocket closed')
            buffer += chunk
        return buffer

    def _recv_frame(self):
        first, second = self._recv_exact(2)
        fin = (first >> 7) & 1
        opcode = first & 0x0f
        masked = (second >> 7) & 1
        length = second & 0x7f
        if length == 126:
            length = struct.unpack('>H', self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack('>Q', self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b''
        payload = self._recv_exact(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4]
                            for index, byte in enumerate(payload))
        return fin, opcode, payload

    def recv_message(self):
        chunks = []
        while True:
            fin, opcode, payload = self._recv_frame()
            if opcode == 8:
                raise EOFError('websocket close frame')
            if opcode == 9:
                self.send(payload, opcode=10)
                continue
            if opcode in (1, 0):
                chunks.append(payload)
            if fin:
                break
        return b''.join(chunks).decode('utf-8', errors='replace')

    def send(self, payload, opcode=1):
        data = payload if isinstance(payload, bytes) else payload.encode('utf8')
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        if len(data) < 126:
            header.append(0x80 | len(data))
        elif len(data) < 65536:
            header.append(0x80 | 126)
            header += struct.pack('>H', len(data))
        else:
            header.append(0x80 | 127)
            header += struct.pack('>Q', len(data))
        header += mask
        masked = bytes(byte ^ mask[index % 4]
                       for index, byte in enumerate(data))
        self.sock.sendall(header + masked)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class CDP:
    def __init__(self, url):
        self.ws = WebSocket(url)
        self.serial = 0

    def call(self, method, params=None, timeout=60):
        self.serial += 1
        identifier = self.serial
        self.ws.send(json.dumps({'id': identifier, 'method': method,
                                 'params': params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = json.loads(self.ws.recv_message())
            if message.get('id') == identifier:
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message.get('result', {})
        raise TimeoutError(f'CDP {method} timed out')

    def evaluate(self, expression, await_promise=False):
        result = self.call('Runtime.evaluate', {
            'expression': expression, 'returnByValue': True,
            'awaitPromise': await_promise})
        remote = result.get('result', {})
        if remote.get('subtype') == 'error':
            raise RuntimeError(remote.get('description', 'Runtime error'))
        return remote.get('value')

    def close(self):
        self.ws.close()


def _browser_url(port):
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            response = requests.get(f'http://127.0.0.1:{port}/json/list', timeout=2)
            targets = response.json()
            pages = [target for target in targets
                     if target.get('type') == 'page' and
                     target.get('webSocketDebuggerUrl')]
            if pages:
                return pages[0]['webSocketDebuggerUrl']
        except Exception:
            pass
        time.sleep(.4)
    raise TimeoutError('Chrome DevTools endpoint did not become ready')


def run(out, port=8891):
    out = Path(out).resolve()
    page = out / 'review.html'
    if not page.is_file():
        return {'status': 'not_run', 'reason': 'review.html is missing'}
    browser = find_browser()
    if not browser:
        return {'status': 'not_run', 'reason': 'Chrome/Edge not found'}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out), **kwargs)

        def log_message(self, *args):
            pass

    server = None
    for candidate in (port, port + 1, port + 2):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', candidate), Handler)
            port = candidate
            break
        except OSError:
            server = None
    if server is None:
        return {'status': 'not_run', 'reason': 'local review server could not bind'}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    debug_port = 9223
    profile = out.parents[2] / 'work/direct-deepseek/38-chrome-profile'
    profile.mkdir(parents=True, exist_ok=True)
    chrome = subprocess.Popen(
        [browser, '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
         '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
         '--disable-background-timer-throttling',
         '--window-size=1280,960', f'--remote-debugging-port={debug_port}',
         f'--user-data-dir={profile}', 'about:blank'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f'http://127.0.0.1:{port}/review.html?selftest=1'
    result = {'status': 'not_run', 'url': url, 'browser': browser,
              'note': ('浏览器兼容性检查：仅验证 38 页面加载、新旧切换、灰模/材质和'
                       '拖动旋转；不代表模型验收，也不代表场景放置通过。')}
    cdp = None
    try:
        cdp = CDP(_browser_url(debug_port))
        cdp.call('Page.enable')
        cdp.call('Runtime.enable')
        cdp.call('Page.navigate', {'url': url})
        deadline = time.time() + 180
        status = ''
        step = ''
        while time.time() < deadline:
            try:
                status = cdp.evaluate("document.body.dataset.selftest||''")
                step = cdp.evaluate("document.body.dataset.step||''")
            except Exception:
                pass
            if status in ('passed', 'failed'):
                break
            time.sleep(.5)
        result['step'] = step
        if status in ('passed', 'failed'):
            result['status'] = status
            raw = cdp.evaluate(
                "(document.getElementById('selftest-result')||{}).textContent||''")
            try:
                result['selftest'] = json.loads(raw)
            except json.JSONDecodeError:
                result['selftest'] = {'raw': raw}
        else:
            result['status'] = 'not_run'
            result['reason'] = (f'selftest did not finish within 180 s; '
                                f'last step: {step or "unknown"}')
        screenshot = out / 'renders' / 'browser-review.png'
        try:
            shot = cdp.call('Page.captureScreenshot', {'format': 'png'}, timeout=60)
            screenshot.write_bytes(base64.b64decode(shot['data']))
            result['screenshot'] = str(screenshot)
        except Exception as error:
            result['screenshot_error'] = f'{type(error).__name__}: {error}'
    except Exception as error:
        result['status'] = 'not_run'
        result['reason'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            if cdp:
                cdp.call('Browser.close', timeout=10)
        except Exception:
            pass
        if cdp:
            cdp.close()
        try:
            chrome.wait(timeout=15)
        except subprocess.TimeoutExpired:
            chrome.terminate()
            try:
                chrome.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    result['checked_at'] = datetime.now(timezone.utc).isoformat()
    record = {
        'status': result['status'],
        'checked_at': result['checked_at'],
        'url': result.get('url'),
        'note': result.get('note'),
        'result': result.get('selftest'),
        'reason': result.get('reason'),
        'browser': result.get('browser'),
        'screenshot': result.get('screenshot'),
        'step': result.get('step'),
    }
    (out / '浏览器核验.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parents[2] /
                                'outputs/38集油器1-六视图重建-v1')
    parser.add_argument('--port', type=int, default=8891)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.port), ensure_ascii=False, indent=2))
