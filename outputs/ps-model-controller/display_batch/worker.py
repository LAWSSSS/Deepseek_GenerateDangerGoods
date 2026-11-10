"""Single persistent Blender process. Serialized RPC keeps GPU memory bounded."""
import asyncio
import json
import subprocess
import time
from pathlib import Path
from .common import BLENDER, HERE, ROOT, FileLease, signature


class RenderUnavailable(RuntimeError):
    pass


class Blender:
    def __init__(self, root, device='OPTIX'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.device=device
        self.lock=asyncio.Lock()
        self.process=None
        self.counter=0
        self.log=None
        self.seconds=0.
        self.gpu_lease=FileLease(ROOT/'work/display-batch-gpu.lock')
        self.wait_seconds=0.
        self.operation_seconds={}
        self.start_seconds=0.

    def start(self):
        if self.process is not None and self.process.poll() is None:return
        if self.log:self.log.close()
        self.log=(self.root/'worker.log').open('a',encoding='utf8')
        started=time.monotonic()
        if not BLENDER.is_file():raise RenderUnavailable('Blender executable unavailable')
        self.process=subprocess.Popen([str(BLENDER),'--background','--factory-startup',
            '--python',str(HERE/'blender_display_worker.py'),'--','serve'],
            stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True,
            encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.start_seconds+=time.monotonic()-started

    async def ask(self, operation, payload, timeout=180):
        async with self.lock:
            waiting=time.monotonic()
            if self.device!='CPU':
                while not self.gpu_lease.acquire():await asyncio.sleep(.5)
            self.wait_seconds+=time.monotonic()-waiting
            self.start(); self.counter+=1
            out=self.root/f'rpc-{self.counter:06}-{signature(operation,payload)[:8]}.json'
            if out.exists():out.unlink()
            request={**payload,'operation':operation,'reply':str(out),'device':self.device}
            start=time.monotonic()
            self.process.stdin.write(json.dumps(request,ensure_ascii=True)+'\n');self.process.stdin.flush()
            while not out.is_file():
                if self.process.poll() is not None:raise RenderUnavailable('Blender worker exited; inspect worker.log')
                if time.monotonic()-start>timeout:
                    self.process.kill();self.process.wait();raise TimeoutError('Blender '+operation+' timed out')
                await asyncio.sleep(.15)
            elapsed=time.monotonic()-start;self.seconds+=elapsed
            self.operation_seconds[operation]=self.operation_seconds.get(operation,0)+elapsed
            data=json.loads(out.read_text(encoding='utf8'))
            if data.get('error'):
                error=data['error']
                if any(s in error.lower() for s in ('no device','cuda','optix','out of memory')):
                    raise RenderUnavailable('Blender: '+error)
                raise RuntimeError('Blender: '+error)
            return data

    async def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.stdin.close()
                try:await asyncio.to_thread(self.process.wait,30)
                except subprocess.TimeoutExpired:
                    self.process.kill();await asyncio.to_thread(self.process.wait)
        finally:
            if self.log:self.log.close()
            self.gpu_lease.release()
