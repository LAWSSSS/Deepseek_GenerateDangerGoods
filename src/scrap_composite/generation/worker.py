"""Single persistent Blender process. Serialized RPC keeps GPU memory bounded."""
import asyncio
import json
import subprocess
import time
from pathlib import Path
from .common import BLENDER, HERE, ROOT, FileLease, signature


class RenderUnavailable(RuntimeError):
    pass


class RenderOperationFailed(RuntimeError):
    """Blender rejected a specific asset or render operation."""


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
        self.memory_retries=0
        self.persistent_data=device!='CPU'

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
        waiting=time.monotonic()
        async with self.lock:
            if self.device!='CPU':
                while not self.gpu_lease.acquire():await asyncio.sleep(.5)
            self.wait_seconds+=time.monotonic()-waiting
            request=dict(payload)
            if operation=='render_parts':
                request['persistent_data']=self.persistent_data
            data=await self.roundtrip(operation,request,timeout)
            error=data.get('error','').lower()
            if request.get('persistent_data') and any(s in error for s in ('out of memory','failed to allocate')):
                # Restart only this worker, preserving the global GPU lease and all render settings.
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
                self.persistent_data=False
                self.memory_retries+=1
                data=await self.roundtrip(operation,{**request,'persistent_data':False},timeout)
            if data.get('error'):
                error=data['error']
                if any(s in error.lower() for s in ('no device','cuda','optix','out of memory','failed to allocate')):
                    raise RenderUnavailable('Blender: '+error)
                raise RenderOperationFailed('Blender: '+error)
            return data

    async def roundtrip(self,operation,payload,timeout):
        self.start();self.counter+=1
        out=self.root/f'rpc-{self.counter:06}-{signature(operation,payload)[:8]}.json'
        if out.exists():out.unlink()
        request={**payload,'operation':operation,'reply':str(out),'device':self.device}
        start=time.monotonic()
        try:
            self.process.stdin.write(json.dumps(request,ensure_ascii=True)+'\n');self.process.stdin.flush()
            while not out.is_file():
                if self.process.poll() is not None:raise RenderUnavailable('Blender worker exited; inspect worker.log')
                if time.monotonic()-start>timeout:
                    self.process.kill();await asyncio.to_thread(self.process.wait)
                    raise TimeoutError('Blender '+operation+' timed out')
                await asyncio.sleep(.15)
            return json.loads(out.read_text(encoding='utf8'))
        finally:
            elapsed=time.monotonic()-start;self.seconds+=elapsed
            self.operation_seconds[operation]=self.operation_seconds.get(operation,0)+elapsed

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
