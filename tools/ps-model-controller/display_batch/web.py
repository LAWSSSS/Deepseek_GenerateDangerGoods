"""Local batch controls; subprocess receives paths, never credentials in argv."""
import json
from importlib.util import find_spec
from pathlib import Path
import secrets
import subprocess
import sys
import time
import asyncio

from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from .common import ROOT, HERE, atomic_json, load_catalog, FileLease
from .schema import BatchConfig, EffectSettings, EFFECT_ACTIONS
from .store import Store

DATA=ROOT/'work/placement-web/batches'
PROCESSES={}


def config_path(ident):
    if len(ident)!=16 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('批次不存在')
    path=DATA/(ident+'.json')
    if not path.is_file():raise ValueError('批次不存在')
    return path


def launch(ident):
    current=PROCESSES.get(ident)
    if current and current.poll() is None:raise ValueError('批次正在运行')
    config=config_path(ident)
    parsed=BatchConfig.model_validate_json(config.read_text(encoding='utf8'))
    lease=FileLease(Path(parsed.output)/'.batch.lock')
    if not lease.acquire():raise ValueError('批次正在运行')
    lease.release()
    log=(DATA/(ident+'.log')).open('a',encoding='utf8')
    try:
        PROCESSES[ident]=subprocess.Popen([sys.executable,'-X','utf8','-u',str(HERE/'batch_display.py'),
            'run','--config',str(config)],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    finally:log.close()


async def page(request):
    return FileResponse(HERE/'placement-web/batch.html')


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
        if not Path(config.backgrounds).is_dir():raise ValueError('背景目录不存在')
        from .pipeline import make_jobs
        catalog=await asyncio.to_thread(load_catalog)
        jobs=await asyncio.to_thread(make_jobs,config,catalog)
        if test and len(jobs)>15:raise ValueError('在线小样清单最多 15 项')
        DATA.mkdir(parents=True,exist_ok=True)
        atomic_json(DATA/(ident+'.json'),config.model_dump())
        launch(ident)
        return JSONResponse({'id':ident,'output':config.output,
            'preset':config.preset,
            'geometry_preset':config.geometry_preset,'vehicle_reference':config.vehicle_reference,
            'integration_mode':config.integration_mode,
            'workflow':config.workflow,
            'review_profile':config.review_profile,
            'effective_modes':{'scale':config.scale_mode,'contact':config.contact_mode,'chroma':config.chroma_mode,
                               'integration':config.integration_mode},
            'effects':config.effects.model_dump() if config.effects else None,
            'status_url':f'/api/batches/{ident}',
            'report_url':f'/api/batches/{ident}/files/index.html'},status_code=202)
    except (ValueError,OSError) as error:
        return JSONResponse({'error':str(error)[:300]},status_code=400)


async def status(request):
    try:
        ident=request.path_params['ident'];path=config_path(ident)
        config=BatchConfig.model_validate_json(path.read_text(encoding='utf8'));out=Path(config.output)
        result={'id':ident,'output':str(out),'running':bool(ident in PROCESSES and PROCESSES[ident].poll() is None)}
        result['effects']=config.effects.model_dump() if config.effects else None
        result['geometry_preset']=config.geometry_preset
        result['integration_mode']=config.integration_mode
        result['workflow']=config.workflow
        result['review_profile']=config.review_profile
        result['preset']=config.preset
        result['vehicle_reference']=config.vehicle_reference
        from .collision import VERSION as collision_version
        from .vehicle_scale import VERSION as scale_version
        result['effective_geometry']={'scale_mode':config.scale_mode,'contact_mode':config.contact_mode,
            'collision_version':collision_version if config.contact_mode=='strict' else 'legacy',
            'vehicle_scale_version':scale_version if config.geometry_preset=='rebuild' else None}
        lease=FileLease(out/'.batch.lock')
        if not lease.acquire():result['running']=True
        else:lease.release()
        if (out/'summary.json').is_file():result.update(json.loads((out/'summary.json').read_text(encoding='utf8')))
        if result.get('usage_current_run'):
            result['usage_current_run']={k:v for k,v in result['usage_current_run'].items() if k!='calls'}
        if (out/'jobs.sqlite').is_file():
            store=Store(out/'jobs.sqlite')
            result['counts']=store.counts()
            result['active']=[{'id':r['id'],'stage':r['stage']} for r in store.rows() if r['state']=='running']
            store.close()
        result['resumable']=not result['running'] and any(result.get('counts',{}).get(k,0) for k in ('queued','running'))
        result['has_results']=(out/'index.html').is_file()
        proc=PROCESSES.get(ident)
        if proc and proc.poll() not in (None,0):result['process_error']='执行进程提前退出，请检查批次日志或使用新输出目录'
        return JSONResponse(result)
    except ValueError as error:return JSONResponse({'error':str(error)},status_code=404)


async def resume(request):
    try:
        launch(request.path_params['ident']);return JSONResponse({'status':'started'},status_code=202)
    except ValueError as error:return JSONResponse({'error':str(error)},status_code=400)


async def artifact(request):
    try:
        config=BatchConfig.model_validate_json(config_path(request.path_params['ident']).read_text(encoding='utf8'))
        root=Path(config.output).resolve();path=(root/request.path_params['path']).resolve()
        if not path.is_relative_to(root) or path.suffix not in ('.png','.jpg','.html','.json','.jsonl') or not path.is_file():
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


routes=[Route('/batch',page),Route('/api/batches',create,methods=['POST']),
        Route('/api/photography/schema',photography_options),
        Route('/api/batches/effects',effect_options),Route('/api/batches/test',create,methods=['POST']),
        Route('/api/batches/{ident}',status),Route('/api/batches/{ident}/resume',resume,methods=['POST']),
        Route('/api/batches/{ident}/files/{path:path}',artifact)]
