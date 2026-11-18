"""HTTP adapter for the same modeling commands exposed to the CLI."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys

from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from scrap_composite.data.files import FileLease, file_hash
from scrap_composite.modeling.contracts import BuildOptions, Publication
from scrap_composite.modeling.service import ModelingService
from scrap_composite.modeling.sources import VIEWS
from scrap_composite.runtime import ROOT, SOURCE

SERVICE = ModelingService()
STATIC = Path(__file__).with_name('static')
PROCESSES = {}


def launch(ident):
    active = PROCESSES.get(ident)
    if active and active.poll() is None:
        raise ValueError('建模任务正在运行')
    folder = SERVICE.registry.directory(ident)
    lease = FileLease(folder / '.build.lock')
    if not lease.acquire():
        raise ValueError('建模任务正在运行')
    lease.release()
    env = {**os.environ, 'PYTHONPATH': str(SOURCE)}
    cmd = [sys.executable, '-X', 'utf8', '-u', '-m', 'scrap_composite', 'model',
           '--workspace', str(SERVICE.registry.root), 'resume', ident]
    with (folder / 'process.log').open('a', encoding='utf8') as log:
        PROCESSES[ident] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def brief(record):
    ident = record['id']
    state = record['state']
    process = PROCESSES.get(ident)
    if state == 'queued' and process and process.poll() is not None:
        state = 'interrupted'
    if state == 'running' and not (process and process.poll() is None):
        lease = FileLease(SERVICE.registry.directory(ident) / '.build.lock')
        if lease.acquire():
            lease.release()
            state = 'interrupted'
    return {key: record.get(key) for key in ('id', 'asset', 'stage', 'error', 'checks', 'seconds', 'created')} | {
        'state': state, 'urls': {key: f'/api/modeling/builds/{ident}/files/{key}'
                                 for key in record.get('artifacts', {})}}


def error_response(error, status=400):
    return JSONResponse({'error': str(error)[:600]}, status_code=status)


async def page(request):
    return FileResponse(STATIC / 'modeling.html')


async def assets(request):
    return JSONResponse(await asyncio.to_thread(SERVICE.items))


async def history(request):
    result = SERVICE.registry.builds(request.path_params['asset'])
    return JSONResponse([brief(record) for record in result])


def uploaded(form, name):
    value = form.get(name)
    return (value.filename, value.file) if isinstance(value, UploadFile) and value.filename else None


async def create(request):
    try:
        async with request.form(max_files=7, max_fields=12) as form:
            options = BuildOptions.model_validate_json(str(form.get('options') or '{}'))
            views = {key: upload for key in VIEWS if (upload := uploaded(form, key))}
            record = await asyncio.to_thread(SERVICE.submit, name=form.get('name') or None,
                asset_id=form.get('asset_id') or None, source=uploaded(form, 'source'), views=views, options=options)
        launch(record['id'])
        return JSONResponse(brief(record), status_code=202)
    except (OSError, TypeError, ValueError) as error:
        return error_response(error)


async def status(request):
    try:
        return JSONResponse(brief(SERVICE.registry.get(request.path_params['ident'])))
    except ValueError as error:
        return error_response(error, 404)


async def resume(request):
    try:
        ident = request.path_params['ident']
        record = SERVICE.registry.get(ident)
        if record['state'] != 'ready':
            launch(ident)
        return JSONResponse(brief(record), status_code=202)
    except (OSError, ValueError) as error:
        return error_response(error)


async def publish(request):
    try:
        data = Publication.model_validate(await request.json())
        result = await asyncio.to_thread(SERVICE.publish, request.path_params['asset'], data.version)
        return JSONResponse({'asset': result['asset']['id'], 'version': result['id'], 'published': True})
    except (OSError, ValueError) as error:
        return error_response(error)


async def artifact(request):
    try:
        ident = request.path_params['ident']
        record = SERVICE.registry.get(ident)
        data = record['artifacts'][request.path_params['kind']]
        path = Path(data['path']).resolve(strict=True)
        if not path.is_relative_to(SERVICE.registry.directory(ident)) or await asyncio.to_thread(file_hash, path) != data['sha256']:
            raise ValueError('建模文件已变更')
        return FileResponse(path, media_type='model/gltf-binary' if path.suffix == '.glb' else None,
                            headers={'X-Content-Type-Options': 'nosniff'})
    except (KeyError, ValueError, OSError):
        return error_response('建模文件不可用', 404)


routes = [
    Route('/modeling', page),
    Route('/api/modeling/assets', assets),
    Route('/api/modeling/assets/{asset}/builds', history),
    Route('/api/modeling/assets/{asset}/publish', publish, methods=['POST']),
    Route('/api/modeling/builds', create, methods=['POST']),
    Route('/api/modeling/builds/{ident}', status),
    Route('/api/modeling/builds/{ident}/resume', resume, methods=['POST']),
    Route('/api/modeling/builds/{ident}/files/{kind}', artifact),
]
