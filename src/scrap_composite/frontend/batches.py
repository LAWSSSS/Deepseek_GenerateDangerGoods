"""Local batch controls; subprocess receives paths, never credentials in argv."""
from contextlib import closing, suppress
import json
import queue
import re
import threading
from importlib.util import find_spec
from pathlib import Path
import secrets
import subprocess
import sys
import time
import asyncio

from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from scrap_composite.generation.common import ROOT, HERE, atomic_json, load_catalog, FileLease
from scrap_composite.generation.schema import BatchConfig, EffectSettings, EFFECT_ACTIONS
from scrap_composite.data.store import Store
from .batch_worker import record_start_failure
from scrap_composite.data.backgrounds import background_files
from scrap_composite.data.snapshot import FILENAME, save_snapshot, snapshot
from scrap_composite.runtime import resolve

STATIC = Path(__file__).with_name('static')

DATA=ROOT/'work/placement-web/batches'
PROCESSES={}


def config_path(ident):
    if len(ident)!=16 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('批次不存在')
    path=DATA/(ident+'.json')
    if not path.is_file():raise ValueError('批次不存在')
    return path


def lock_held(path):
    lease = FileLease(path)
    if not lease.acquire():
        return True
    lease.release()
    return False


def read_summary(out):
    path = out / 'summary.json'
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else {}


def receipt_path(out, run):
    ident = run.get('id')
    if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', ident):
        raise ValueError('运行记录中的编号无效')
    return out / f'run-{ident}.json'


def reconcile_interruption(out, store, summary):
    """Called with the batch lease and no active web worker."""
    before = store.last_run()
    legacy = summary.get('run') or {}
    if not before and legacy.get('state') == 'complete' and store.remaining():
        legacy = {**legacy, 'state':'running', 'stage':'dispatch', 'error':
                  '历史摘要标为完成，但任务账本仍有未处理项。原运行未记录具体原因。'}
    run = store.interrupt_stale_run(legacy)
    if not run or run['state'] != 'interrupted':
        return run
    receipt = receipt_path(out, run)
    if before and before['state'] != 'running' and receipt.is_file():
        # A rejected configuration can legitimately retain the previous summary.
        # Only the caller's current status view should overlay the new error.
        return run
    if run.get('stage') == 'launch' or not store.counts():
        summary.update(run=run, counts=store.counts(), usage_current_run=None)
        atomic_json(out / 'summary.json', summary)
    else:
        from scrap_composite.generation.pipeline import write_summary
        usage_file = out / f"usage-{run['id']}.json"
        usage = json.loads(usage_file.read_text(encoding='utf8')) if usage_file.is_file() else None
        summary.update(write_summary(out, store, usage, run))
    atomic_json(receipt, run)
    return run


def wait_for_worker(proc):
    reply = queue.Queue(maxsize=1)

    def read_ready():
        reply.put(proc.stdout.readline().strip())
    threading.Thread(target=read_ready, daemon=True).start()
    try:
        if reply.get(timeout=10) != b'ready':
            raise OSError('工作进程在启动握手前退出，请检查批次日志')
    except queue.Empty:
        raise TimeoutError('工作进程启动握手超时') from None


def launch(ident):
    config = config_path(ident)
    parsed = BatchConfig.model_validate_json(config.read_text(encoding='utf8'))
    out = Path(parsed.output)
    database = out / 'jobs.sqlite'
    if database.exists() and not (out / FILENAME).exists():
        raise ValueError('历史批次请在建立任务时的原实现版本续跑')
    lease = FileLease(out / '.batch.lock')
    if not lease.acquire():
        raise ValueError('批次正在运行')
    try:
        if lock_held(out / '.web-worker.lock'):
            raise ValueError('批次正在运行')
        previous = PROCESSES.get(ident)
        if previous:
            previous.poll()  # Reap the prior child if this web process started it.
        with closing(Store(database)) as store:
            reconcile_interruption(out, store, read_summary(out))
            if store.counts() and not store.remaining():
                raise ValueError('所有任务都已处理，当前批次没有可续跑的任务')
            run_id = secrets.token_hex(16)
            run = store.begin_run(run_id, 'launch')
            if run['remaining'] is None and (out / FILENAME).is_file():
                run['remaining'] = len(json.loads((out / FILENAME).read_text(encoding='utf8'))['jobs'])
                store.record_run(run)
            proc = None
            try:
                with (DATA / (ident + '.log')).open('ab') as log:
                    proc = subprocess.Popen([
                        sys.executable, '-X', 'utf8', '-u', str(Path(__file__).with_name('batch_worker.py')),
                        str(out), str(HERE / 'batch_display.py'), str(config), run_id,
                    ], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                wait_for_worker(proc)
                PROCESSES[ident] = proc
                lease.release()
                proc.stdin.write(b'run\n')
                proc.stdin.flush()
            except (OSError, ValueError) as exc:
                # The child has not acknowledged the dispatch. Reap this startup
                # attempt so it cannot acquire the worker lease later.
                if proc and proc.poll() is None:
                    proc.kill()
                    proc.wait()
                record_start_failure(store, out, run_id, exc)
                raise
            finally:
                if proc:
                    with suppress(BrokenPipeError):
                        proc.stdin.close()
                    proc.stdout.close()
    finally:
        lease.release()


async def page(request):
    return FileResponse(STATIC/'batch.html')


async def create(request):
    try:
        raw=await request.json()
        if not isinstance(raw,dict):raise ValueError('请求必须为 JSON 对象')
        test=request.url.path=='/api/batches/test'
        if test:
            raw.setdefault('count',1)
            # Let the coherent preset supply its own defaults. An empty effects
            # object would instantiate legacy manual haze before schema validation.
            if (raw.get('integration_mode','legacy')!='coherent' and raw.get('workflow')!='evidence_v1'
                    and raw.get('review_profile')!='photographer'):
                raw.setdefault('effects',{})
            raw.setdefault('max_iterations',2)
            if raw.get('offline') or ('effects' in raw and raw['effects'] is None):
                raise ValueError('在线测试必须启用效果和在线复核')
        ident=secrets.token_hex(8)
        raw['output']=str(ROOT/'outputs'/('批量展示-'+time.strftime('%Y%m%d-%H%M%S')+'-'+ident[:6]))
        config=BatchConfig.model_validate(raw)
        if test and config.count>15:raise ValueError('在线小样测试每批最多 15 项')
        if not config.manifest and not resolve(config.backgrounds).is_dir():raise ValueError('背景目录不存在')
        from scrap_composite.generation.pipeline import make_jobs
        catalog=await asyncio.to_thread(load_catalog)
        jobs=await asyncio.to_thread(make_jobs,config,catalog)
        if test and len(jobs)>15:raise ValueError('在线小样清单最多 15 项')
        DATA.mkdir(parents=True,exist_ok=True)
        await asyncio.to_thread(save_snapshot, config.output, await asyncio.to_thread(snapshot,config,jobs,catalog))
        atomic_json(DATA/(ident+'.json'),config.model_dump())
        startup_error = None
        try:
            await asyncio.to_thread(launch, ident)
        except OSError as exc:
            # The request already owns an ID and snapshot. Offer that attempt's
            # durable error and retry action through the normal status endpoint.
            if not (Path(config.output) / 'jobs.sqlite').exists():
                raise
            startup_error = str(exc)[:300]
        return JSONResponse({'id':ident,'output':config.output,
            **({'startup_error':startup_error} if startup_error else {}),
            'preset':config.preset,
            'geometry_preset':config.geometry_preset,'vehicle_reference':config.vehicle_reference,
            'integration_mode':config.integration_mode,
            'workflow':config.workflow,
            'review_profile':config.review_profile,
            'photoshop_mode':config.photoshop_mode,
            'effective_modes':{'scale':config.scale_mode,'contact':config.contact_mode,'chroma':config.chroma_mode,
                               'integration':config.integration_mode},
            'effects':config.effects.model_dump() if config.effects else None,
            'status_url':f'/api/batches/{ident}',
            'report_url':f'/api/batches/{ident}/files/index.html'},status_code=201 if startup_error else 202)
    except (ValueError,OSError) as error:
        return JSONResponse({'error':str(error)[:300]},status_code=400)


async def status(request):
    try:
        ident=request.path_params['ident'];path=config_path(ident)
        config=BatchConfig.model_validate_json(path.read_text(encoding='utf8'));out=Path(config.output)
        result={'id':ident,'output':str(out)}
        result['effects']=config.effects.model_dump() if config.effects else None
        result['geometry_preset']=config.geometry_preset
        result['integration_mode']=config.integration_mode
        result['workflow']=config.workflow
        result['review_profile']=config.review_profile
        result['photoshop_mode']=config.photoshop_mode
        result['preset']=config.preset
        result['vehicle_reference']=config.vehicle_reference
        from scrap_composite.generation.collision import VERSION as collision_version
        from scrap_composite.generation.vehicle_scale import VERSION as scale_version
        result['effective_geometry']={'scale_mode':config.scale_mode,'contact_mode':config.contact_mode,
            'collision_version':collision_version if config.contact_mode=='strict' else 'legacy',
            'vehicle_scale_version':scale_version if config.geometry_preset=='rebuild' else None}
        summary = read_summary(out)
        database = out / 'jobs.sqlite'
        running = lock_held(out / '.web-worker.lock')
        if database.is_file():
            with closing(Store(database)) as store:
                run = store.last_run()
                counts = store.counts()
                legacy = summary.get('run') or {}
                unfinished_legacy = not run and (counts.get('running', 0) or
                    (counts.get('queued', 0) and legacy.get('state') == 'complete'))
                may_be_active = (run or legacy).get('state') == 'running' or unfinished_legacy
                files_pending = run and run['state'] == 'interrupted' and not receipt_path(out, run).is_file()
                if not running and (may_be_active or files_pending):
                    lease = FileLease(out / '.batch.lock')
                    if lease.acquire():
                        try:
                            running = lock_held(out / '.web-worker.lock')
                            if not running:
                                summary = read_summary(out)
                                run = reconcile_interruption(out, store, summary)
                        finally:
                            lease.release()
                    else:
                        running = True
                if run is not None:
                    summary['run'] = dict(run)
                    if run['state'] == 'running' and counts:
                        summary['run']['remaining'] = store.remaining()
                previous = store.last_interruption()
                if previous and previous['id'] != (run or {}).get('id'):
                    result['previous_interruption'] = {
                        key: previous.get(key) for key in ('id', 'stage', 'job_id', 'error', 'remaining')
                    }
                summary['counts'] = store.counts()
                result['active'] = [
                    {'id': r['id'], 'stage': r['stage']} for r in store.rows() if r['state'] == 'running'
                ] if running and summary['counts'].get('running') else []
        result.update(summary)
        result['running'] = running
        if result.get('usage_current_run'):
            result['usage_current_run'] = {k:v for k,v in result['usage_current_run'].items() if k != 'calls'}
        counts = result.get('counts', {})
        remaining = counts.get('queued', 0) + counts.get('running', 0)
        legacy = database.exists() and not (out / FILENAME).is_file()
        result['legacy_runtime_required'] = legacy
        # A run can fail during import/preflight before the queue is initialized.
        result['resumable'] = (not legacy and not running and
                               (remaining > 0 or not counts and (out / FILENAME).is_file()))
        result['has_results'] = (out / 'index.html').is_file()
        proc = PROCESSES.get(ident)
        if proc:
            proc.poll()
        return JSONResponse(result)
    except ValueError as error:return JSONResponse({'error':str(error)},status_code=404)


async def resume(request):
    try:
        await asyncio.to_thread(launch, request.path_params['ident'])
        return JSONResponse({'status':'started'},status_code=202)
    except (ValueError, OSError) as error:return JSONResponse({'error':str(error)},status_code=400)


async def artifact(request):
    try:
        config=BatchConfig.model_validate_json(config_path(request.path_params['ident']).read_text(encoding='utf8'))
        root=Path(config.output).resolve();path=(root/request.path_params['path']).resolve()
        if not path.is_relative_to(root) or path.suffix not in ('.png','.jpg','.html','.json','.jsonl','.psd') or not path.is_file():
            raise ValueError('文件不存在')
        return FileResponse(path)
    except (ValueError,OSError):return JSONResponse({'error':'文件不存在'},status_code=404)


async def effect_options(request):
    return JSONResponse({'version':'online-effects-v3','defaults':EffectSettings().model_dump(),
        'presets':['existing','realism_v1'],
        'integration_modes':{'default':'legacy','options':[
            {'value':'legacy','label':'原融合流程'},
            {'value':'coherent','label':'局部受光与反射协同（小样）'}],
            'promoted_default':False,'dependency':'lighting_material_render_shadow_compose_review'},
        'workflows':['legacy','evidence_v1'],
        'photoshop_modes':['off','always','on_issue'],
        'review_profiles':['legacy','photographer'] if find_spec('pydantic_graph') is not None else ['legacy'],
        'recommended':{'scale_mode':'physical_consistent','effects':{
            **EffectSettings().model_dump(),'fog_mode':'local_auto','weathering':'preserve_source',
            'shadow_mode':'matched_receiver','lighting':'local'}},
        'reference_projection_trial':{'scale_mode':'reference_projection','contact_mode':'strict',
                                      'chroma_mode':'local_budget','legacy_display_factors_applied':False},
        'geometry_rebuild_trial':{'geometry_preset':'rebuild','requires':'vehicle_reference',
            'example_reference':{'vehicle_class':'semitrailer','internal_width_m':2.45,
                'nominal_internal_length_m':12.85,'nominal_internal_height_m':2.2,
                'source':{'kind':'user_supplied','reference':'用户提供的车厢内部尺寸，长高仅作辅助估计'}},
            'promoted_default':False},
        'global_saturation_trial':{'chroma_mode':'global_mean','promoted_default':False,
                                   'basis':'full_background_mean_HSV_S_additional_cap_after_local_budget'},
        'schema':EffectSettings.model_json_schema(),'actions':EFFECT_ACTIONS,
        'test_endpoint':'/api/batches/test','max_test_jobs':15,
        'dependencies':{'fog_extra':'compose_and_review','fog_gain':'compose_and_review','pool_kernel':'compose_and_review',
                        'surface_dust':'material_render_shadow_compose_review'},
        'notes':['fog_extra 为追加雾层，非总雾量','pool_kernel=1 关闭池化',
                 'surface_dust 是独立表面浮灰倍率，漆面另乘 0.5，塑料不加额外灰层',
                 'atmosphere=off 同时关闭基础和追加薄雾','未指定 effects 的旧批次保持原行为']})


async def photography_options(request):
    from scrap_composite.photography_agent import interface
    return JSONResponse({**interface(), 'available':find_spec('pydantic_graph') is not None})


async def backgrounds(request):
    try:
        directory = resolve(request.query_params.get('directory') or 'external/backgrounds')
        files = await asyncio.to_thread(background_files, directory)
        return JSONResponse([path.name for path in files])
    except (OSError, ValueError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


routes=[Route('/batch',page),Route('/api/backgrounds',backgrounds),Route('/api/batches',create,methods=['POST']),
        Route('/api/photography/schema',photography_options),
        Route('/api/batches/effects',effect_options),Route('/api/batches/test',create,methods=['POST']),
        Route('/api/batches/{ident}',status),Route('/api/batches/{ident}/resume',resume,methods=['POST']),
        Route('/api/batches/{ident}/files/{path:path}',artifact)]
