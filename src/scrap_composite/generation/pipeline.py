import asyncio
from contextlib import contextmanager
import html
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import time
import numpy as np

from PIL import Image, ImageOps

from . import VERSION
from .ai import Vision, ProviderUnavailable, token_summary
from .assets import prepare as prepare_asset
from .common import ROOT, HERE, atomic_json, file_hash, load_catalog, resolve, signature, implementation_manifest
from .estimated_scale import allows_estimate, prepare_preview_scale
from .compose import compose, sample_environment
from .scene import Scenes
from .placement import Placement
from .schema import (Review, FusionReview, EffectsReview, Probe, PHOTO_CONTROLS, StructureConfirmation, version_rank)
from .pile import diagnostic, receiver_mesh
from scrap_composite.data.store import Store
from .worker import Blender, RenderUnavailable
from .framing import edit_bounds, final_resolution
from .rendering import RenderCache, RENDER_VERSION
from .review import allowed_actions, evidence_status, PLACEMENT_DEFECTS
from .execution import PhotoshopExecutor, PhotoshopUnavailable, needs_photoshop, photoshop_plan, pixel_hash
from .gate.review import review_frame, review_record
from .gate.feedback import geometry_issue, next_frame
from .materials import material_kind, SURFACE_DUST
from .atmosphere import Atmospheres
from .costs import cumulative, estimate, tariff, label as cost_label
from .policy import asset_id
from scrap_composite.data.inventory import STORAGE_TANK_IDS
from .effects import parameters as effect_parameters, render_identity
from .reference_material import ReferenceInputChanged
from scrap_composite.data.backgrounds import background_files
from scrap_composite.data.snapshot import mismatched_inputs, read_snapshot, save_snapshot, snapshot

COMPOSE_VERSION='scrap-gray-pile-v6-window'


REVIEW_PROMPT='''Critically inspect the actual final composite panorama and enlarged crop against the source object.
This is for realistic human display, NOT training annotations. Exactly one object must lie horizontally inside
the scrap hopper. Score 0..4: support/contact, occlusion, lighting/color, edge, texture/sensor, material, structure.
3 means usable, 4 means convincing. Painted/rusted surfaces must not become chrome; plastic must remain plastic.
Check floating, penetration, excessive standing height, implausible scale/camera, pasted edges, overly clean/smooth
or uniformly matte metals, wrong shadows, lost holes/rods. Do not reject a lying view merely because the source is upright.
Keep identifying features. Severe defects are major/fatal; minor defects may be acceptable.
Select up to three allowed modifications, using exact supplied values. candidate changes must use an offered ID.
Do not invent code, masks, geometry edits or unavailable locations. If no valid repair exists return no actions.
Do not assume a claimed numerical check proves visual quality. Inspect the image itself.'''
REVIEW_PROMPT += ''' Defect scope is image for view/lighting/appearance problems, asset only for certain
intrinsic mesh damage. Do not call a foreshortened part missing merely because the reference is upright.
An absent separately operated handle is intentional when the object name says so. Lack of foreground
occlusion on an open floor is not itself a defect. Contact shadow must agree with the visible supports.'''
REVIEW_PROMPT += ''' The requested photographic style is strongly weathered grey scrap, not saturated showroom paint.
Partial embedding and 10-40 percent foreground occlusion are intentional. A feature behind steel, facing away,
or below readable pixel size has visibility occluded/unassessable: do not claim intrinsic model damage.
Only visible, evidenced geometry loss can be asset scope; a separate unobstructed review must confirm it.
Chroma/dust/contrast/haze/sharpness actions use auto/less/more; more chroma means MORE desaturation.
Do not demand an occluded identifying feature to become visible, and do not confuse dust with missing geometry.'''


@contextmanager
def batch_lock(folder):
    path=Path(folder)/'.batch.lock';path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('该批次正在运行') from None
    try:yield
    finally:
        if os.name=='nt':
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        handle.close()


def make_jobs(config,catalog):
    excluded=STORAGE_TANK_IDS.union(asset_id(k) for k in config.excluded_assets)
    if config.manifest:
        jobs=json.loads(resolve(config.manifest).read_text(encoding='utf8'))
        if not isinstance(jobs,list) or not 1<=len(jobs)<=1000:raise ValueError('invalid manifest size')
    else:
        directory=resolve(config.backgrounds)
        backgrounds=background_files(directory, config.background_files)
        if not backgrounds:raise ValueError('没有可读取背景')
        ids=config.assets or [key for key, row in catalog.items() if row.get('available', True)]
        ids=[asset_id(k) for k in ids if asset_id(k) not in excluded]
        if not ids:raise ValueError('所选模型均已移出生成名单')
        rng=random.Random(config.seed);rng.shuffle(backgrounds);rng.shuffle(ids)
        jobs=[{'background':str(backgrounds[(i//len(ids)+i)%len(backgrounds)]),
               'asset_id':ids[i%len(ids)],'group':'batch'} for i in range(config.count)]
    result=[]
    for i,job in enumerate(jobs):
        k=asset_id(job['asset_id']);path=resolve(job['background'])
        if k in excluded:continue
        if k not in catalog:raise ValueError('未知模型编号：'+k)
        if not catalog[k].get('available', True):raise ValueError('模型不可用：'+k)
        if not path.is_file():raise ValueError('背景不存在：'+str(path))
        seed=job.get('seed',config.seed+i)
        if type(seed) is not int or seed<0:raise ValueError('invalid manifest seed')
        result.append({**job,'background':str(path),'asset_id':k,'background_hash':file_hash(path),
                       'mesh_hash':catalog[k]['mesh_hash'],'seed':seed,
                       **({'model_version':catalog[k]['version']} if catalog[k].get('version') else {})})
    if not result:raise ValueError('任务清单只包含已移出的模型')
    return result


def catalog_preflight(catalog):
    import struct
    result=[]
    for key,row in catalog.items():
        info={'asset_id':key,'available':row['available'],'model_review':row['human_review'],
              'mesh_hash':row['mesh_hash'],'formal_status':'unchanged'}
        if row['available']:
            with open(row.get('source_mesh', row['mesh']),'rb') as f:
                header=f.read(20);length=struct.unpack_from('<I',header,12)[0]
                doc=json.loads(f.read(length))
            info.update(materials=len(doc.get('materials',[])),textures=len(doc.get('textures',[])),
                        mesh_primitives=sum(len(m['primitives']) for m in doc.get('meshes',[])),
                        render_topology_preserved=True)
        if row.get('packed_subject'):
            info.update(source_mesh_hash=row['source_mesh_hash'], packed_subject=row['packed_subject'],
                        source_material_statistics=True)
        result.append(info)
    return result


def write_summary(out,store,usage=None,run=None):
    out=Path(out).resolve()
    rows=store.rows();records=[r['result'] for r in rows if r['result']]
    with (out/'results.jsonl').open('w',encoding='utf8') as f:
        for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    groups={}
    for row in rows:
        group=row['payload'].get('group','batch')
        g=groups.setdefault(group,{'total':0,'accepted_display':0,'review':0,'failed':0,'pending':0,
                                   'photo_pass':0,'visual_pass':0,'unknown_evidence':0})
        g['total']+=1;g[row['state'] if row['state'] in g else 'pending']+=1
        record=row['result'] or {}
        g['photo_pass']+=record.get('photo_status')=='pass'
        g['visual_pass']+=record.get('visual_status')=='pass'
        g['unknown_evidence']+=record.get('evidence_status')=='unknown'
    for group in groups.values():group['acceptance_rate']=group['accepted_display']/max(1,group['total'])
    times=[r['seconds'] for r in records if 'seconds' in r]
    summary={'version':VERSION,'counts':store.counts(),'groups':groups,'formal_acceptance':'not_changed',
             'cost_total':cumulative(out,usage),
             'run':run,'usage_current_run':usage,'seconds_median':statistics.median(times) if times else None,
             'seconds_p95':sorted(times)[min(len(times)-1,math.ceil(len(times)*.95)-1)] if times else None,
             'generated_pngs':sum(bool(r.get('image')) for r in records),
             'reused_pngs':sum(bool(r.get('reused_from') and r.get('image')) for r in records),
             'photoshop_exports':sum(r.get('cost',{}).get('photoshop_exports',0) for r in records),
             'photoshop_cache_hits':sum(r.get('cost',{}).get('photoshop_cache_hits',0) for r in records),
             'photoshop_seconds':round(sum(r.get('timings',{}).get('photoshop_seconds',0.) for r in records),3),
             'photo_pass':sum(r.get('photo_status')=='pass' for r in records),
             'visual_pass':sum(r.get('visual_status')=='pass' for r in records),
             'image_rate':sum(bool(r.get('image')) for r in records)/max(1,len(rows))}
    if any('insertion_trace' in r for r in records):
        summary['insertion_traces']={status:sum(r.get('insertion_trace')==status for r in records)
                                     for status in ('none','uncertain','visible')}
    atomic_json(out/'summary.json',summary)
    sections=[]
    for r in records:
        image=Path(r['image']).relative_to(out).as_posix() if r.get('image') else None
        crop=Path(r['crop']).relative_to(out).as_posix() if r.get('crop') else None
        psd=Path(r['psd']).relative_to(out).as_posix() if r.get('psd') else None
        name=f"{r['job_id']} · {r['asset_id']} · {r['status']}"
        trace={'none':'未见插入痕迹','uncertain':'插入痕迹不确定','visible':'可见插入痕迹'}.get(r.get('insertion_trace'),'')
        history=[]
        for entry in r.get('iterations',[]):
            if entry.get('deliverable') is False:continue
            p=entry.get('params',{})
            if not p.get('effects_enabled'):continue
            history_image=Path(entry['image']).resolve()
            # Reused results keep their original history records, but the API
            # serves only files inside this batch. Its final full PNG is copied.
            if not history_image.is_relative_to(out):continue
            uri=html.escape(history_image.relative_to(out).as_posix(),quote=True)
            label=f"第 {entry['iteration']} 轮 · 追加薄雾 {float(p['fog_extra']):.0%} · {p['pool_kernel']}×{p['pool_kernel']} 池化 · 表面浮灰 ×{p['surface_dust']}"
            history.append('<figure><figcaption>'+html.escape(label)+f'</figcaption><a href="{uri}"><img src="{uri}" loading="lazy"></a></figure>')
        sections.append('<section><h2>'+html.escape(name)+'</h2><p>'+html.escape(' · '.join(x for x in (trace,r.get('reason','')) if x))+'</p>'+
                        (f'<a href="{image}"><img src="{image}" loading="lazy"></a>' if image else '')+
                        (f'<p><a href="{crop}">局部检查图</a></p>' if crop else '')+
                        (f'<p><a href="{psd}">同版本分层 PSD</a></p>' if psd else '')+
                        ('<details><summary>每轮完整画面与实际参数</summary>'+''.join(history)+'</details>' if history else '')+'</section>')
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>多背景批量结果</title><style>body{background:#172128;color:#eee;font:15px sans-serif;margin:30px auto;max-width:1600px;padding:16px}section{border-top:1px solid #456;padding:15px}img{display:block;width:100%;height:auto}a{color:#b1dfce}h2{font-size:16px}</style><h1>多背景批量展示结果</h1><p>完整画面可点击查看原尺寸。场景依据为估计，展示合格不等于旧流程正式验收。</p><pre>'+html.escape(json.dumps(summary['counts'],ensure_ascii=False))+'</pre>'+''.join(sections)
    page=page.replace('<h1>多背景批量展示结果</h1>','<h1>多背景批量展示结果</h1><p>'+html.escape(cost_label(summary['cost_total']))+'</p>')
    (out/'index.html').write_text(page,encoding='utf8')
    return summary


class Pipeline:
    def __init__(self,config,*,catalog=None):
        self.config=config;self.out=resolve(config.output);self.out.mkdir(parents=True,exist_ok=True)
        self.cache=resolve(config.cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.surface_pack = None
        self.reference_snapshot = None
        if config.effects and config.effects.surface_finish in ('generated_pbr', 'physical_layers'):
            from .generated_material import load_pack
            self.surface_pack = load_pack(resolve(config.effects.surface_finish_manifest))
        self.run_id=time.strftime('%Y%m%d-%H%M%S')+f'-{os.getpid()}'
        self.vision=Vision(config,self.cache/'vision',self.out/f'usage-{self.run_id}.json')
        self.blender=Blender(self.cache/('worker-'+str(os.getpid())),config.render_device)
        self.renders=RenderCache(self.cache,self.blender)
        self.scenes=Scenes(self.cache,self.vision,structured=config.support_geometry=='structured',
                           structured_json=config.workflow=='evidence_v1');self.assets={}
        self.atmospheres=Atmospheres(self.cache,self.vision)
        self.catalog=load_catalog() if catalog is None else catalog
        self.store=Store(self.out/'jobs.sqlite')
        self.cpu=asyncio.Semaphore(config.cpu_concurrency)
        self.prepare_lock=asyncio.Lock()
        self.width_gates={}
        self.scene_format_failures={}
        self.model_review={}
        self.structure_locks={}
        review_path=self.out/'model-review.json'
        if review_path.is_file():self.model_review=json.loads(review_path.read_text(encoding='utf8'))

    def prepare_reference_inputs(self, jobs):
        manifest = self.config.effects.reference_material_manifest if self.config.effects else None
        if manifest is None:
            return None
        from .reference_material import freeze_library
        selected = {job['mesh_hash']:job['asset_id'] for job in jobs}
        if any(selected[job['mesh_hash']] != job['asset_id'] for job in jobs):
            raise ValueError('selected asset IDs share an ambiguous reference mesh')
        self.reference_snapshot = freeze_library(resolve(manifest), selected)
        return self.reference_snapshot

    def assert_reference_inputs(self):
        snapshot = getattr(self, 'reference_snapshot', None)
        if snapshot is not None:
            from .reference_material import verify_snapshot
            verify_snapshot(snapshot)

    async def prepare_asset(self,k):
        async with self.prepare_lock:
            if k not in self.assets:
                # Failed expensive preparation is cached too; a thousand jobs must not
                # run the same doomed decomposition or malformed asset request repeatedly.
                row=self.catalog[k]
                key=signature('asset-preparation-failure-v1',row['mesh_hash'],
                    file_hash(row['board']) if row.get('board') else None, row.get('dimension'),
                    self.vision.endpoint.model_dump(),implementation_manifest(),self.config.contact_mode)
                failure_path=self.cache/'asset-failures'/(key+'.json')
                if failure_path.is_file():
                    self.assets[k]=json.loads(failure_path.read_text(encoding='utf8'))
                else:
                    try:self.assets[k]=await prepare_asset(row,self.cache,self.blender,self.vision,strict=self.config.contact_mode=='strict')
                    except (ProviderUnavailable,RenderUnavailable):raise
                    except Exception as error:
                        self.assets[k]={**row,'signature':key,'poses':[],
                            'reason':f'asset_preparation_failed: {type(error).__name__}: {str(error)[:220]}'}
                        atomic_json(failure_path,self.assets[k])
            return self.assets[k]

    async def width_preflight(self, payload):
        from .vehicle_scale import preflight_visible_width
        key=payload['background_hash']
        async with self.prepare_lock:
            if key not in self.width_gates:
                with Image.open(payload['background']) as source:
                    size=ImageOps.exif_transpose(source).size
                try:
                    gate=await preflight_visible_width(payload['background'],size,self.config.vehicle_reference,self.vision)
                except ProviderUnavailable:
                    raise
                except (ValueError,RuntimeError) as failure:
                    gate={'status':'unknown','allow_expensive_steps':False,
                          'risk_codes':['width_observation_unavailable'],'reason':str(failure)[:180]}
                self.width_gates[key]=gate
                atomic_json(self.out/'width-observations'/(key+'.json'),gate)
        return self.width_gates[key]

    async def prepare_scene_scale(self, scene, gate):
        from .vehicle_scale import prepare as prepare_vehicle

        async with self.prepare_lock:
            if 'estimated_preview_scale' in scene:
                return scene['estimated_preview_scale']['projection_preflight'], True
            if gate['allow_expensive_steps']:
                if scene.get('vehicle_scale',{}).get('observation_mode')!='audited':
                    scene['vehicle_scale']=await prepare_vehicle(
                        scene,self.config.vehicle_reference,self.vision,self.cache,observation_mode='audited')
                gate=scene['vehicle_scale']['projection_preflight']
            if (not gate['allow_expensive_steps'] and self.config.scale_evidence_policy=='estimate_preview'
                    and allows_estimate(gate)):
                scene['estimated_preview_scale']=prepare_preview_scale(scene,self.config.vehicle_reference,gate)
                atomic_json(self.out/'scale-scenes'/(scene['key']+'-estimated.json'),scene['estimated_preview_scale'])
                return gate, True
            return gate, False

    async def evidence_scene(self, payload):
        key=payload['background_hash']
        async with self.prepare_lock:
            if key in self.scene_format_failures:
                raise RuntimeError(self.scene_format_failures[key])
            try:
                return await self.scenes.prepare(payload['background'])
            except ProviderUnavailable:
                raise
            except (ValueError,RuntimeError) as failure:
                self.scene_format_failures[key]=str(failure)[:220]
                raise

    async def render(self,asset,scene,candidate,params,resolution,*,sampled=None,shadow_reference=None):
        style=self.render_style
        if style=='native' or params.get('fog_mode')=='local_auto':
            scene['atmosphere']=await self.atmospheres.prepare(scene)
        environment,samples=(sampled if sampled is not None else
            await asyncio.to_thread(sample_environment,scene,candidate,
                scene_lighting=style in ('scene','native'),native=style=='native',local=self.config.appearance=='local'))
        environment={**environment,'integration_mode':params.get('integration_mode','legacy')}
        extras={}
        if params.get('fog_mode')=='local_auto':
            if params.get('realism_version') == 'realism-v1':
                from .scene_atmosphere import prepare
                context=await asyncio.to_thread(prepare,scene,self.cache)
            else:
                from .local_atmosphere import prepare as prepare_local_atmosphere
                context=await asyncio.to_thread(prepare_local_atmosphere,scene,candidate,environment,self.cache)
            environment={**environment,'atmospheric_context':context}
        if params.get('effects_enabled'):
            from .effects import dust_profile, lighting_for
            from .surface_maps import prepare_uv_masks
            if params.get('realism_version') == 'realism-v1':
                from .realism_light import estimate_local_light
                environment,_=await asyncio.to_thread(estimate_local_light,scene,candidate,environment)
            else:
                environment=await asyncio.to_thread(lighting_for,scene,candidate,environment,params['effects_lighting'],
                                                   preserve=params.get('weathering')=='preserve_source')
            if asset.get('packed_subject'):
                if (params.get('surface_finish', 'existing') != 'existing'
                        or any(float(params.get(name, 0.)) for name in ('surface_dust', 'rust_spots'))):
                    raise ValueError('Packed candidate requires existing material and no added weathering')
            elif params.get('surface_finish') in ('generated_pbr', 'physical_layers'):
                from .generated_material import WEATHERED_STEEL_VERSION, finish_regions, load_pack
                path = resolve(params['surface_finish_manifest'])
                pack = self.surface_pack
                if pack is None or Path(pack['manifest']) != path:
                    pack = await asyncio.to_thread(load_pack, path)
                if params.get('surface_finish') == 'physical_layers':
                    from .physical_material import prepare_finish
                    from .asset_profiles import load_profile
                    size = asset.get('physical_size',{}).get('nominal_mm')
                    if size is None:
                        lo,hi=asset['semantic']['major_mm'];size=math.sqrt(lo*hi)
                    profile=load_profile(asset['mesh_hash']) if params.get('asset_refinement') == 'component_v1' else None
                    reference = None
                    if params.get('reference_material_manifest'):
                        from .reference_material import frozen_reference
                        reference = await asyncio.to_thread(frozen_reference,getattr(self,'reference_snapshot',None),
                                                            asset['mesh_hash'],resolve(params['reference_material_manifest']))
                    extras['generated_finish']=await asyncio.to_thread(prepare_finish,asset,self.cache,pack,size/1000,profile=profile,reference=reference)
                    regions=extras['generated_finish']['regions']
                else:
                    regions = await asyncio.to_thread(finish_regions, asset, pack, self.cache)
                    extras['generated_finish'] = {**pack, 'regions': regions}
                if params.get('surface_finish') == 'generated_pbr' and pack['version'] == WEATHERED_STEEL_VERSION and any(
                        item.get('authored_atlas') and item['coverage'][2] > 0
                        for item in regions['materials'].values()):
                    from .surface_context import nearby_weathered_steel
                    reference = await asyncio.to_thread(nearby_weathered_steel, scene, candidate,
                                                        np.asarray(scene['frame'])[samples])
                    environment = {**environment, 'nearby_weathered_steel': reference}
            else:
                partitions,_=await asyncio.to_thread(prepare_uv_masks,asset,self.cache/'scanned-uv')
                extras={'matte_partitions':partitions,'matte_profile':dust_profile(float(params['surface_dust']),params.get('weathering','legacy_matte'))}
        render_candidate=candidate
        if style=='native' or params.get('shadow_mode')=='matched_receiver':
            # Cast onto the visible neighboring support as well. Physics retains
            # its original finite contact receiver and its separate evidence.
            receiver=await asyncio.to_thread(receiver_mesh,scene,candidate,bounds=edit_bounds(candidate,scene['size']))
            render_candidate={**candidate,'receiver':receiver}
        render_environment={**environment,'haze':0.} if self.config.appearance=='local' else environment
        if params.get('render_detail')=='fine':
            resolution=min(2048,64*math.ceil(2*resolution/64))
        folder,meta,hit=await self.renders.get(asset,render_candidate,params,render_environment,scene['size'],resolution,
                                             appearance=style,shadow_reference=shadow_reference,**extras)
        if getattr(self,'reference_snapshot',None) is not None:
            await asyncio.to_thread(self.assert_reference_inputs)
        if params.get('effects_enabled'):
            if params.get('realism_version') == 'realism-v1':
                from .realism_light import receiver_calibration
            else:
                from .local_light import receiver_calibration
            calibration=await asyncio.to_thread(receiver_calibration,folder,meta,environment)
            environment={**environment,'calibration':calibration}
        return folder,meta,environment,samples,hit

    @property
    def render_style(self):
        if self.config.appearance=='native':return 'native'
        if self.config.appearance=='local':return 'local'
        return 'scene' if self.config.appearance in ('materials','scene') else 'reference'

    def placement_service(self):
        return Placement(self.config, self.out, getattr(self,'cache',self.out), self.vision,
                         getattr(self,'blender',None), getattr(self,'prepare_lock',None))

    async def choose_placements(self, asset, scene, seed):
        return await self.placement_service().choose_placements(asset, scene, seed)

    async def validate_local_pose(self, asset, scene, candidate):
        return await self.placement_service().validate_local_pose(asset, scene, candidate)

    async def confirm_structure(self,asset,candidate,scene,defects):
        # One cached, independent unoccluded orbit check per model; no single composite quarantines it.
        key=signature('structure-confirm-v1',asset['mesh_hash'],file_hash(asset['board']))
        folder=self.cache/'structure-confirm'/key;folder.mkdir(parents=True,exist_ok=True)
        answer_path=folder/'answer.json'
        if answer_path.is_file():return StructureConfirmation.model_validate_json(answer_path.read_text(encoding='utf8'))
        environment,_=sample_environment(scene,candidate)
        render_path=folder/'result.json'
        if not render_path.is_file():
            from .common import mesh_input
            metadata=await self.blender.ask('render',{**mesh_input(asset),'matrix':asset['poses'][0]['matrix'],
                'yaw':90,'elevation':45,'environment':environment,'resolution':512,'roughness_delta':0,
                'light_gain':1,'dust_gain':1,'output':str(folder)},timeout=240)
            atomic_json(render_path,metadata)
        answer=await self.vision.ask('structure-confirm',StructureConfirmation,
            'Compare the independent unobstructed render with the CURRENT approved model reference. '
            'Only confirm visible intrinsic mesh damage, never a hidden/rear/tiny feature or intentional revision. '
            'If reference/view is insufficient return unassessable and confirmed_damage=false. Do not evaluate color.',
            {'name':asset['name'],'intentional_revisions':'Use the current model; detached jack handle intentionally removed, '
             'one can/bottle only, cylinder wall intentionally closed.','suspected':[d.model_dump() for d in defects]},
            [folder/'object.png',asset['board']])
        atomic_json(answer_path,answer.model_dump());return answer

    def default_params(self,candidate):
        photo_model={'native':'native','local':'local','camera':'camera','scene':'camera'}.get(self.config.appearance,'reference')
        return {'candidate':candidate,'material':'original','light':'scene','tone':'scene','photo_model':photo_model,
                'photo_style':self.config.photo_style,'atmosphere':self.config.atmosphere,
                'chroma_mode':self.config.chroma_mode,
                'integration_mode':self.config.integration_mode,
                'camera_pipeline':self.config.camera_pipeline,
                **({'realism_version':'realism-v1'} if self.config.preset=='realism_v1' else {}),
                **({'workflow':self.config.workflow} if self.config.workflow!='legacy' else {}),
                **({'review_profile':'photographer'} if getattr(self.config,'review_profile','legacy')=='photographer' else {}),
                **{k:'auto' for k in PHOTO_CONTROLS},**effect_parameters(self.config.effects)}

    @staticmethod
    def finish_image(out,ident,result,best):
        if best.get('deliverable') is False:raise ValueError('physical_failure_is_not_deliverable')
        receipt = best.get('photoshop')
        if receipt:
            model_psd = receipt['outputs']['psd']
            if file_hash(model_psd['path']) != model_psd['sha256'] or file_hash(best['image']) != best['image_hash']:
                raise ValueError('photoshop_artifacts_changed_before_delivery')
        unreviewed=best.get('review_status')=='unknown' and best.get('review') is None
        if unreviewed:
            if evidence_status(best['checks'])=='fail':
                raise ValueError('program_failure_is_not_deliverable')
            result.update(status='review',unreviewed_draft=True,review_status='unknown')
        estimated=(best['checks'].get('scale',{}).get('evidence',{}).get('version')=='reference-projection-v1'
                   and best['checks']['scale']['status']!='pass')
        dest=out/('estimated' if estimated else 'accepted' if result['status']=='accepted_display' else 'review')
        result['delivery_group']='estimated_scale' if estimated else 'normal'
        dest.mkdir(parents=True,exist_ok=True)
        final=dest/(ident+'.png');shutil.copy2(best['image'],final)
        crop=dest/(ident+'.crop.jpg');shutil.copy2(best['crop'],crop)
        review=None
        if not unreviewed:
            review_type=EffectsReview if best['params'].get('effects_enabled') else FusionReview if 'trace' in best['review'] else Review
            review=review_type.model_validate(best['review'])
        result.update(image=str(final),crop=str(crop),best_iteration=best['iteration'],
                      checks=best['checks'],scores=best['review']['scores'] if review is not None else None,
                      measured_major_px=best['measured_major_px'],
                      placement=best.get('placement','surface'),support_ids=best.get('support_ids',[]),
                      occlusion_fraction=best.get('occlusion_fraction'),photo=best.get('photo'),
                      evidence_confidence=best.get('evidence_confidence','unknown'),
                      evidence_status=evidence_status(best['checks']),
                      photo_status=('pass' if review.photo_passes() else 'fail') if review is not None else 'unknown',
                      visual_status=('pass' if review.passes() else 'fail') if review is not None else 'unknown')
        if isinstance(review,(FusionReview,EffectsReview)):result['insertion_trace']=review.trace
        if receipt:
            psd=dest/(ident+'.psd')
            shutil.copy2(receipt['outputs']['psd']['path'], psd)
            result.update(psd=str(psd), photoshop={'psd_sha256':file_hash(psd),
                'png_sha256':file_hash(final), 'readback':receipt['readback'], 'receipt':receipt.get('receipt')})

    async def job(self,job):
        ident=job['id'];payload=job['payload'];start=time.monotonic()
        asyncio.current_task().set_name(ident)
        folder=self.out/'jobs'/ident;folder.mkdir(parents=True,exist_ok=True)
        result={'job_id':ident,'asset_id':payload['asset_id'],'background':payload['background'],
                'group':payload.get('group','batch'),'status':'review','scene_basis':'estimated',
                'formal_status':'unchanged','iterations':[],'timings':{},'candidate_visits':[],
                'photo_status':'unknown','visual_status':'unknown','evidence_status':'unknown',
                'cost':{'logical_reviews':0,'renders':{},'render_cache_hits':{},'asset_imports':0},
                'actions':[]}
        if payload.get('background_hash'):result['background_hash']=payload['background_hash']
        if payload.get('model_version'):result['model_version']=payload['model_version']
        best=None;draft=None
        ps_mode = getattr(getattr(self,'config',None),'photoshop_mode','off')
        result['photoshop_mode'] = ps_mode
        try:
            if getattr(self,'input_snapshot',None):
                issue = await asyncio.to_thread(mismatched_inputs, self.input_snapshot, payload)
                if issue:
                    raise ReferenceInputChanged(issue)
            if getattr(self,'reference_snapshot',None) is not None:
                await asyncio.to_thread(self.assert_reference_inputs)
            held=getattr(self,'model_review',{}).get(payload['asset_id'])
            if held and held.get('independent_confirmation') and held.get('mesh_hash')==payload.get('mesh_hash'):
                result.update(reason='asset_structure_requires_review',model_review=held);return result
            structured=getattr(self.config,'workflow','legacy')=='evidence_v1'
            preview_policy=getattr(self.config,'scale_evidence_policy','observed')=='estimate_preview'
            if preview_policy:result['preview_policy']='estimate_preview'
            scene=None
            if structured:
                result['configuration']={'requested':self.config._requested,'resolved':self.config.model_dump(),
                                         'reasons':self.config._resolution_notes}
                stage_start=time.monotonic()
                self.store.checkpoint(ident,'scene',{})
                if self.config.geometry_preset=='rebuild':
                    gate=await self.width_preflight(payload)
                    result['width_preflight']=gate
                    if not gate['allow_expensive_steps'] and not (preview_policy and allows_estimate(gate)):
                        result.update(status='failed',evidence_status='fail',reason='high_risk_vehicle_projection',
                                      rejections=[{'reason':'high_risk_vehicle_projection','evidence':gate}])
                        return result
                scene=await self.evidence_scene(payload)
                result['timings']['scene_seconds']=round(time.monotonic()-stage_start,3)
                result['scene_signature']=scene['key']
                if self.config.geometry_preset=='rebuild':
                    # Reject unobservable calibration before asset preparation or Blender.
                    gate,estimated=await self.prepare_scene_scale(scene,gate)
                    if estimated:
                        result['scene_basis']='estimated_visible_layer_preview'
                        result['projection_preflight']=gate
                    elif not gate['allow_expensive_steps']:
                        atomic_json(self.out/'scale-scenes'/(scene['key']+'.json'),scene['vehicle_scale'])
                        result.update(status='failed',evidence_status='fail',reason='high_risk_vehicle_projection',
                                      rejections=[{'reason':'high_risk_vehicle_projection','evidence':gate}])
                        return result
            self.store.checkpoint(ident,'asset',job['checkpoint'] or {})
            stage_start=time.monotonic()
            asset=await self.prepare_asset(payload['asset_id'])
            result['timings']['asset_prepare_seconds']=round(time.monotonic()-stage_start,3)
            result['asset_signature']=asset['signature']
            if not asset['poses'] and not (self.config.contact_mode=='strict' and asset.get('axis')):
                result['reason']=asset['reason'];return result
            if scene is None:
                self.store.checkpoint(ident,'scene',{})
                stage_start=time.monotonic()
                scene=await self.scenes.prepare(payload['background'])
                result['timings']['scene_seconds']=round(time.monotonic()-stage_start,3)
            result['scene_signature']=scene['key'];result['physics_signature']=asset['physics_signature']
            async with self.cpu:
                stage_start=time.monotonic()
                candidates,rejected=await self.choose_placements(asset,scene,payload['seed'])
            result['timings']['placement_seconds']=round(time.monotonic()-stage_start,3)
            candidates=sorted(candidates,key=lambda c:(len(set(c.get('uncertainties',[]))),-c['score']))[:3]
            ready={};invalid=set()
            def save_candidates():
                atomic_json(folder/'candidates.json',{'candidates':[ready.get(c['id'],c) for c in candidates],
                    'visited':result['candidate_visits'],'rejected':rejected})
            save_candidates()
            result['rejections']=rejected
            if not candidates:
                result['reason']='no_valid_metric_lying_pose' if any(r.get('reason')=='no_valid_metric_lying_pose' for r in rejected) else 'no_geometric_candidate'
                if self.config.contact_mode=='strict':result.update(status='failed',evidence_status='fail')
                return result
            params=self.default_params(candidates[0]['id'])
            seen=set();seen_images=set();best_rank=None;no_gain=0;reason='iterations_exhausted'
            adjusted_classes=set(SURFACE_DUST)
            if self.render_style=='native':adjusted_classes.add('mixed')
            adjustable=any(material_kind(m['name'],m['metallic'] or 0.,m['metallic_linked'],
                           asset['semantic'].get('material_classes',[])) in adjusted_classes
                           for m in asset.get('inspection',{}).get('materials',[]))
            iteration=0;previous_candidate=None;rendered_parts={};recipe_checked=set()
            ps_pending = None
            ps_base = None
            ps_seen = set()
            while iteration<=self.config.max_iterations:
                psig=signature(params, ps_pending['actions']) if ps_pending else signature(params)
                if psig in seen:reason='repeated_parameters';break
                seen.add(psig);cid=params['candidate']
                if cid not in ready:
                    candidate=next(c for c in candidates if c['id']==cid)
                    result['candidate_visits'].append(cid)
                    if self.config.contact_mode=='strict':
                        await asyncio.to_thread(diagnostic,scene,candidate,folder/(cid+'-initial-diagnostic.jpg'))
                    before=time.monotonic()
                    try:
                        ready[cid]=await self.validate_local_pose(asset,scene,candidate) if 'plane_normal' in candidate else candidate
                    except ValueError as failure:
                        invalid.add(cid);rejected.append({'candidate':cid,'reason':str(failure)})
                    finally:
                        result['timings']['local_physics_seconds']=result['timings'].get('local_physics_seconds',0)+time.monotonic()-before
                    if cid in ready and candidate.get('placement'):
                        await asyncio.to_thread(diagnostic,scene,ready[cid],folder/(cid+'-geometry.jpg'))
                    save_candidates()
                if cid not in invalid:
                    candidate=ready[cid]
                    candidate['material_classes']=asset.get('semantic',{}).get('material_classes',[])
                    if ps_pending:
                        meta, folder_render, hit = ps_pending.pop('meta')
                        comp = await self._ps_frame(ident, folder, ps_base, result, iteration, **ps_pending)
                        image = Path(comp['image'])
                        ps_pending = None
                    else:
                        render_options={}
                        if structured and cid not in recipe_checked:
                            from .structured import initial_recipe
                            sampled=await asyncio.to_thread(sample_environment,scene,candidate,local=True)
                            params,recommendation=initial_recipe(resolve(self.config.knowledge_db or 'work/knowledge/history-v2.sqlite'),
                                                                 asset,candidate,sampled[0],params)
                            result['parameter_recommendation']={**recommendation,'candidate':cid}
                            recipe_checked.add(cid)
                            seen.add(signature(params))
                            render_options['sampled']=sampled
                        rkey=render_identity(params,self.render_style)
                        if getattr(self,'reference_snapshot',None) is not None:
                            await asyncio.to_thread(self.assert_reference_inputs)
                        if rkey not in rendered_parts:
                            before=time.monotonic()
                            rendered=await self.render(asset,scene,candidate,params,final_resolution(candidate),**render_options)
                            result['timings']['render_stage_seconds']=result['timings'].get('render_stage_seconds',0)+time.monotonic()-before
                            rendered_parts[rkey]=rendered
                            for part,info in rendered[1].get('render_cache',{}).items():
                                tally=result['cost']['render_cache_hits' if info['hit'] else 'renders']
                                tally[part]=tally.get(part,0)+1
                            for key,value in rendered[1].get('render_timings',{}).items():
                                result['timings'][key]=result['timings'].get(key,0)+value
                            result['cost']['asset_imports']+=rendered[1].get('asset_imports',0)
                        else:result['cost']['photo_only_reuse']=result['cost'].get('photo_only_reuse',0)+1
                        folder_render,meta,environment,samples,hit=rendered_parts[rkey]
                        image=folder/f'iteration-{iteration}.png'
                        before=time.monotonic()
                        try:
                            async with self.cpu:
                                layer_options={}
                                if structured and self.config.prepare_photoshop:
                                    layer_options['layer_output']=folder/f'layers-{iteration}'
                                comp=await asyncio.to_thread(compose,scene,candidate,folder_render,meta,environment,samples,
                                                             params,payload['seed'],image,**layer_options)
                        except ValueError as failure:
                            if str(failure) not in ('object_fully_occluded','object_too_small'):raise
                            invalid.add(cid);rejected.append({'candidate':cid,'reason':str(failure)})
                        else:
                            if evidence_status(comp['checks'])=='fail':
                                invalid.add(cid);rejected.append({'candidate':cid,'reason':'final_program_checks_failed','checks':comp['checks']})
                        result['timings']['compose_seconds']=result['timings'].get('compose_seconds',0)+time.monotonic()-before
                if cid in invalid:
                    save_candidates()
                    fallback=next((c['id'] for c in candidates if c['id'] not in invalid and c['id'] not in ready),None)
                    if fallback:
                        params=self.default_params(fallback);continue
                    reason='no_candidate_survived_program_checks'
                    if best is None:result.update(status='failed',evidence_status='fail')
                    break
                if not comp.get('photoshop'):
                    ps_base = comp
                    ps_seen = set()
                    if needs_photoshop(ps_mode):
                        comp = await self._ps_frame(ident, folder, comp, result, iteration)
                        image = Path(comp['image'])
                if comp.get('photoshop'):
                    ps_seen.add(signature(comp['execution_state']['parameters']))
                draft={**comp,'iteration':iteration,'params':params.copy(),'render_cache_hit':hit,
                       'render_cache':meta.get('render_cache',{}),'render_resolution':meta.get('resolution'),
                       'image_hash':file_hash(image)}
                if meta.get('packed_subject'):
                    draft['packed_subject'] = meta['packed_subject']
                if structured:
                    from .structured import algorithm_versions
                    draft['algorithm_versions']=algorithm_versions(params,comp['photo'].get('version'))
                pixels = pixel_hash(image) if comp.get('photoshop') else draft['image_hash']
                same_pixels = False
                if comp.get('photoshop') and result['iterations']:
                    same_pixels = pixels == pixel_hash(result['iterations'][-1]['image'])
                repeated = same_pixels or pixels in seen_images
                if result['actions'] and result['actions'][-1]['iteration']==iteration-1:
                    prior = result['iterations'][-1]
                    result['actions'][-1]['effect']={'params':params.copy(),'image_hash':draft['image_hash'],
                        'changed_pixels':not same_pixels and draft['image_hash']!=prior['image_hash'],
                        'repeated_image':repeated}
                if repeated:
                    reason='no_effective_image_change';break
                seen_images.add(pixels)
                atomic_json(folder/f'draft-{iteration}.json',draft)
                self.store.checkpoint(ident,'review_request',{'last_iteration':iteration,
                    'params':params,'draft':str(folder/f'draft-{iteration}.json')})
                choices=[c for c in candidates if c['id'] not in invalid]
                ids=[c['id'] for c in choices]
                allowed=allowed_actions(params,comp,ids,render_style=self.render_style,adjustable=adjustable)
                geometry_board=folder/(cid+'-geometry.jpg') if previous_candidate and previous_candidate!=cid and candidate.get('placement') else None
                result['cost']['logical_reviews']+=1
                profile=getattr(self.config,'review_profile','legacy')
                observation,review=await review_frame(self.vision,asset,comp,draft['image_hash'],params,allowed,
                    structured=structured,profile=profile,prompt=REVIEW_PROMPT,render_dir=folder_render,
                    compact=self.config.review_format=='compact',geometry_board=geometry_board,choices=choices)
                previous_candidate=cid
                entry={**draft, **review_record(observation,review,enforce_physics=(
                    self.config.contact_mode=='strict' or structured and profile=='photographer' or ps_mode!='off'))}
                result['iterations'].append(entry)
                atomic_json(folder/f'iteration-{iteration}.json',entry)
                self.store.checkpoint(ident,'review',{'last_iteration':iteration,'params':params,'record':str(folder/f'iteration-{iteration}.json')})
                if entry.get('deliverable') is False:
                    invalid.add(cid);rejected.append({'candidate':cid,'reason':'visual_physical_failure','defects':entry['physical_visual_veto']})
                    for previous in result['iterations']:
                        if previous['params']['candidate']==cid:previous['deliverable']=False
                    # A later view can expose the same candidate's earlier hidden physical fault.
                    eligible=[e for e in result['iterations'] if e.get('deliverable',True) and e['params']['candidate'] not in invalid]
                    best=max(eligible,key=lambda e:version_rank(e['checks'],Review.model_validate(
                        {k:v for k,v in e['review'].items() if k!='trace'}))) if eligible else None
                    best_rank=None;save_candidates()
                    fallback=next((c['id'] for c in candidates if c['id'] not in invalid and c['id'] not in ready),None)
                    if fallback and iteration<self.config.max_iterations:
                        params=self.default_params(fallback);iteration+=1;continue
                    reason='no_deliverable_candidate_after_physical_veto'
                    if best is None:result.update(status='failed',evidence_status='fail')
                    break
                needs_placement = (profile=='photographer' or ps_mode!='off') and geometry_issue(review)
                if needs_placement:
                    alternative=next((c['id'] for c in choices if c['id']!=cid and c['id'] not in ready),None)
                    if alternative and iteration<self.config.max_iterations:
                        result['actions'].append({'iteration':iteration,'decisions':[{
                            'disposition':'geometry_recheck','stage':'placement','candidate':alternative}]})
                        params=self.default_params(alternative);iteration+=1;continue
                rank=version_rank(comp['checks'],review)
                if isinstance(review,(FusionReview,EffectsReview)):
                    trace_rank={'none':2,'uncertain':1,'visible':0}[review.trace]
                    rank=(*rank,trace_rank)
                if best_rank is None or rank>best_rank:
                    best,best_rank=entry,rank;no_gain=0
                else:no_gain+=1
                if not preview_policy and all(c['status']=='pass' for c in comp['checks'].values()) and review.passes():
                    result['status']='accepted_display';reason='display_checks_passed';best=entry;break
                suspects=[d for d in review.defects if d.code=='structure' and d.scope=='asset'
                          and d.visibility=='visible' and d.severity in ('major','fatal')]
                if suspects:
                    if not hasattr(self,'structure_locks'):self.structure_locks={}
                    async with self.structure_locks.setdefault(payload['asset_id'],asyncio.Lock()):
                        confirmation=await self.confirm_structure(asset,candidate,scene,suspects)
                    entry['structure_confirmation']=confirmation.model_dump()
                    atomic_json(folder/f'iteration-{iteration}.json',entry)
                    if confirmation.confirmed_damage and confirmation.visibility=='visible':
                        if not hasattr(self,'model_review'):self.model_review={}
                        self.model_review[payload['asset_id']]={'mesh_hash':payload.get('mesh_hash'),
                            'job_id':ident,'independent_confirmation':True,'confirmation':confirmation.model_dump(),
                            'defects':[d.model_dump() for d in suspects]}
                        atomic_json(self.out/'model-review.json',self.model_review)
                        reason='asset_structure_requires_review';break
                if (evidence_status(comp['checks'])=='unknown' and review.photo_passes() and review.scores.edge>=3
                    and (not isinstance(review,(FusionReview,EffectsReview)) or review.trace=='none')):
                    unresolved=[d for d in review.defects if d.visibility=='visible' and d.severity in ('major','fatal')]
                    repairable=any(d.code in PLACEMENT_DEFECTS for d in unresolved) and any(v!=cid for v in ids)
                    structure_ok=review.scores.structure>=3 or any(
                        d.code=='structure' and d.visibility!='visible' for d in review.defects)
                    if structure_ok and not repairable and not any(d.code not in PLACEMENT_DEFECTS for d in unresolved):
                        reason='visual_ok_evidence_unresolved';break
                if iteration>=self.config.max_iterations:break
                if no_gain>=2:reason='no_improvement';break
                if not needs_placement and needs_photoshop(ps_mode, observation=observation,
                                                          remaining=self.config.max_iterations-iteration):
                    current = comp.get('execution_state', {}).get('parameters') if comp.get('photoshop') else None
                    if current is None:
                        from scrap_composite.photoshop import LIMITS
                        current = dict.fromkeys(LIMITS, 0)
                    ps_seen.add(signature(current))
                    updates, ps_plan = await photoshop_plan(self.vision, observation, comp, current, ps_seen)
                    if updates:
                        result['actions'].append({'iteration':iteration,'execution_plan':ps_plan,'backend':'photoshop'})
                        ps_pending = {'actions':updates, 'prior':comp.get('photoshop'),
                                      'reviewed':entry, 'meta':(meta,folder_render,hit)}
                        iteration+=1
                        continue
                    if ps_plan:
                        result['photoshop_plan_disposition'] = ps_plan
                updated,actions,plan=await next_frame(self.vision,observation,review,comp,params,allowed,ids,
                    lambda p:signature(p) in seen,structured=structured,
                    dust_in_shader=self.render_style in ('scene','native'),
                    recommendation=result.get('parameter_recommendation'))
                result['actions'].append({'iteration':iteration,'decisions':actions,'execution_plan':plan})
                if updated is None:
                    if structured and profile=='photographer' and plan:
                        reason=plan['reason']
                    else:
                        reason='repeated_parameters' if actions and all(a['disposition']=='repeated_parameters' for a in actions) else 'no_effective_action'
                    break
                params=updated;iteration+=1
            if getattr(self,'reference_snapshot',None) is not None:
                await asyncio.to_thread(self.assert_reference_inputs)
            if best:
                if getattr(self,'input_snapshot',None):
                    issue = await asyncio.to_thread(mismatched_inputs, self.input_snapshot, payload)
                    if issue:
                        raise ReferenceInputChanged(issue)
                self.finish_image(self.out,ident,result,best)
            elif self.config.contact_mode=='strict':
                result.update(status='failed',evidence_status='fail')
            result['reason']=reason
            return result
        except PhotoshopUnavailable as error:
            result.update(reason=str(error), status='review', execution_paused=True)
            raise
        except (ProviderUnavailable,RenderUnavailable,ReferenceInputChanged):
            raise
        except Exception as error:
            result.update(status='failed',reason=f'{type(error).__name__}: {str(error)[:260]}')
            if type(error).__name__ == 'InvalidStructuredReply':
                result.update(status='review',visual_status='unknown',photo_status='unknown',
                              reason='review_invalid_after_one_repair')
            if best:
                if self.config.contact_mode=='strict' or ps_mode!='off':result['status']='review'
                self.finish_image(self.out,ident,result,best)
            elif draft:
                result.update(checks=draft['checks'],draft_record=str(folder/f"draft-{draft['iteration']}.json"),
                    unreviewed_draft=True,measured_major_px=draft['measured_major_px'],
                    evidence_status=evidence_status(draft['checks']))
                if (type(error).__name__=='InvalidStructuredReply' and self.config.preset=='realism_v1'
                        and self.config.uncertain_policy=='render_review'
                        and evidence_status(draft['checks'])!='fail'):
                    entry={**draft,'review':None,'review_status':'unknown','unreviewed_draft':True,
                           'review_error':'invalid_after_one_repair','deliverable':True}
                    result['iterations'].append(entry)
                    atomic_json(folder/f"iteration-{draft['iteration']}.json",entry)
                    self.finish_image(self.out,ident,result,entry)
            return result
        finally:
            result['seconds']=round(time.monotonic()-start,3)
            calls=[c for c in self.vision.calls if c.get('job_id')==ident]
            usage=token_summary(calls)
            result['new_api_calls']=usage['api_calls']
            result['cost']['api_retries']=sum(c.get('attempt',0)>0 and c.get('state')!='not_sent' for c in calls)
            result['cost']['reported_tokens']=usage['reported_tokens']
            result['cost']['unreported_calls']=usage['unreported_calls']
            result['cost']['api_estimate']=estimate(calls,tariff(self.config,self.vision.endpoint) if calls else None)
            result['cost_total_at_output']=cumulative(self.out,self.vision.usage() if calls else None)
            result['timings']['api_seconds']=round(sum(c.get('seconds',0) for c in calls),3)
            atomic_json(folder/'result.json',result)

    async def _ps_frame(self, ident, folder, source, result, iteration, *, actions=None, prior=None,
                        reviewed=None):
        self.store.checkpoint(ident, 'photoshop', {
            'iteration':iteration, 'composition':source.get('composition_manifest'),
            'reviewed_image_sha256':reviewed.get('image_hash') if reviewed else None,
            'actions':actions or {}})
        comp = await PhotoshopExecutor().execute(source, folder/'photoshop', actions=actions,
                                                prior=prior, reviewed=reviewed)
        result['timings']['photoshop_seconds'] = (
            result['timings'].get('photoshop_seconds',0.) + comp['photoshop_seconds'])
        counter='photoshop_cache_hits' if comp['photoshop_cache_hit'] else 'photoshop_exports'
        result['cost'][counter]=result['cost'].get(counter,0)+1
        return comp

    def reuse_completed(self,source,implementation):
        source=Path(source).resolve()
        source_manifest=json.loads((source/'implementation.json').read_text(encoding='utf8'))
        source_config=json.loads((source/'config.json').read_text(encoding='utf8'))
        exclude={'output','manifest','count','offline'}
        source_vision=json.loads((source/'vision-signature.json').read_text(encoding='utf8'))
        source_evidence=source/'dimension-reference-signature.json'
        if source_evidence.is_file() and json.loads(source_evidence.read_text(encoding='utf8'))!=self.dimension_reference_signature():
            raise ValueError('pilot dimension/reference evidence has changed')
        if getattr(self,'reference_snapshot',None) is not None:
            from .reference_material import compatible_snapshot
            snapshot_path=source/'reference-material-snapshot.json'
            if not snapshot_path.is_file() or not compatible_snapshot(
                    self.reference_snapshot,json.loads(snapshot_path.read_text(encoding='utf8'))):
                raise ValueError('pilot reference materials have changed')
        if source_vision['signature']!=signature(self.vision.endpoint.model_dump()) or signature(source_manifest)!=signature(implementation) or {
            k:v for k,v in source_config.items() if k not in exclude
        }!=self.config.model_dump(exclude=exclude):
            raise ValueError('pilot was produced with different code or settings')
        previous=Store(source/'jobs.sqlite')
        try:completed={signature(r['payload']):r for r in previous.rows() if r['result'] and r['result'].get('image')}
        finally:previous.close()
        for row in self.store.rows():
            if row['state']!='queued':continue
            original=completed.get(signature(row['payload']))
            if not original or original['state'] not in ('accepted_display','review'):continue
            result=dict(original['result'])
            entry=next(e for e in result['iterations'] if e['iteration']==result['best_iteration'])
            if file_hash(result['image'])!=entry['image_hash']:
                raise ValueError('pilot result no longer matches its verified PNG')
            self.finish_image(self.out,row['id'],result,{**entry,'image':result['image'],'crop':result['crop']})
            result.update(job_id=row['id'],reused_from={'batch':str(source),'job_id':original['id']},
                          source_seconds=result['seconds'],seconds=0.,timings={},new_api_calls=0,
                          cost={'logical_reviews':0,'renders':{},'render_cache_hits':{},'asset_imports':0,
                                'api_retries':0,'reported_tokens':0,'unreported_calls':0})
            self.store.finish(row['id'],result['status'],result)

    def dimension_reference_signature(self):
        from .specifications import CATALOG
        if self.config.scale_mode!='reference_projection' and self.config.contact_mode!='strict':return None
        return {k:file_hash(resolve(v)) if v else None for k,v in {
            'catalog':str(CATALOG),'dimensions':self.config.dimension_registry,
            'scene_references':self.config.scene_reference_registry}.items()}

    async def run(self,*,reuse_from=None):
        with batch_lock(self.out):
            return await self._run_pinned(reuse_from)

    async def _run_pinned(self, reuse_from):
        started=time.monotonic();state='complete';error=None
        pinned = read_snapshot(self.out, self.config)
        if pinned is None:
            jobs = make_jobs(self.config, self.catalog)
            pinned = snapshot(self.config, jobs, self.catalog)
        else:
            jobs, self.catalog = pinned['jobs'], pinned['catalog']
        self.input_snapshot = pinned
        reference_snapshot=await asyncio.to_thread(self.prepare_reference_inputs,jobs)
        implementation=implementation_manifest()
        inputs = (VERSION,self.config.model_dump(exclude={'offline'}),jobs,
                                    RENDER_VERSION,COMPOSE_VERSION,implementation,self.vision.endpoint.model_dump(),
                                    self.vision.planner.model_dump() if self.config.workflow=='evidence_v1' else None,
                                    self.dimension_reference_signature())
        if self.surface_pack is not None:
            inputs = (*inputs, self.surface_pack['signature'])
        if reference_snapshot is not None:
            inputs = (*inputs, ('reference-material-snapshot', reference_snapshot['signature']))
        config_signature=signature(*inputs)
        self.store.initialize(config_signature,jobs);self.store.recover()
        save_snapshot(self.out, pinned)
        atomic_json(self.out/'config.json',self.config.model_dump())
        atomic_json(self.out/'inputs.json',jobs)
        atomic_json(self.out/'implementation.json',implementation)
        atomic_json(self.out/'vision-signature.json',{'signature':signature(self.vision.endpoint.model_dump())})
        atomic_json(self.out/'dimension-reference-signature.json',self.dimension_reference_signature())
        if reference_snapshot is not None:
            atomic_json(self.out/'reference-material-snapshot.json',reference_snapshot)
        try:
            if issue := await asyncio.to_thread(mismatched_inputs, pinned):
                raise ReferenceInputChanged(issue)
            atomic_json(self.out/'catalog-preflight.json',catalog_preflight(self.catalog))
            if self.config.photoshop_mode == 'always' and self.store.counts().get('queued'):
                await PhotoshopExecutor().preflight()
            if reuse_from:self.reuse_completed(reuse_from,implementation)
            if not self.config.offline and self.store.counts().get('queued'):
                probe_path=self.out/'capability-probe.jpg'
                with Image.open(jobs[0]['background']) as im:
                    im=im.convert('RGB');im.thumbnail((512,512));im.save(probe_path,quality=80)
                probe=await self.vision.ask('capability',Probe,'Confirm the supplied image is visible; describe it briefly.',{},[probe_path],
                                            plain_json=self.config.appearance in ('native','local'))
                if not probe.image_visible:raise ProviderUnavailable('视觉服务未确认图像可见')
            stop=asyncio.Event()
            async def consumer():
                nonlocal state,error
                while not stop.is_set():
                    job=self.store.claim(group_size=4)
                    if not job:return
                    print(json.dumps({'event':'start','job':job['id'],'asset':job['payload']['asset_id']},ensure_ascii=False),flush=True)
                    try:result=await self.job(job)
                    except (ProviderUnavailable,RenderUnavailable,ReferenceInputChanged,PhotoshopUnavailable) as failure:
                        self.store.pause(job['id'])
                        state=('paused_inputs' if isinstance(failure,ReferenceInputChanged) else
                               'paused_provider' if isinstance(failure,ProviderUnavailable) else
                               'paused_photoshop' if isinstance(failure,PhotoshopUnavailable) else 'paused_renderer')
                        error=str(failure);stop.set();return
                    self.store.finish(job['id'],result['status'],result)
                    progress=write_summary(self.out,self.store,self.vision.usage(),{'id':self.run_id,'state':'running'})
                    print(json.dumps({'event':'done','job':job['id'],'status':result['status'],
                                      'reason':result['reason'],'seconds':result['seconds'],
                                      'cost_total':cost_label(progress['cost_total'])},ensure_ascii=False),flush=True)
            await asyncio.gather(*(consumer() for _ in range(self.config.cpu_concurrency)))
        except ProviderUnavailable as failure:
            state='paused_provider';error=str(failure)
        except ReferenceInputChanged as failure:
            state='paused_inputs';error=str(failure)
        except PhotoshopUnavailable as failure:
            state='paused_photoshop';error=str(failure)
        finally:
            await self.blender.close();await self.vision.close()
            summary=write_summary(self.out,self.store,self.vision.usage(),
                {'id':self.run_id,'state':state,'error':error,'seconds':round(time.monotonic()-started,3),
                 'gpu_seconds':round(self.blender.seconds,3)})
            summary['run']['worker_operations_seconds']=self.blender.operation_seconds
            summary['run']['worker_launch_seconds']=self.blender.start_seconds
            summary['run']['gpu_wait_seconds']=self.blender.wait_seconds
            summary['run']['api_seconds']=round(sum(c['seconds'] for c in self.vision.calls),3)
            summary['run']['render_counts']=dict(self.renders.totals)
            summary['run']['render_stage_seconds']=dict(self.renders.timings)
            summary['run']['memory_retries']=self.blender.memory_retries
            if self.config.workflow=='evidence_v1':
                from scrap_composite.knowledge import archive_run
                import sqlite3
                try:
                    summary['knowledge_archive']=archive_run(self.out,resolve('work/knowledge/history-v2.sqlite'))
                except (OSError,ValueError,sqlite3.Error) as failure:
                    summary['knowledge_archive']={'error_kind':type(failure).__name__}
            atomic_json(self.out/f'run-{self.run_id}.json',summary['run'])
            atomic_json(self.out/'summary.json',summary)
            self.store.close()
        return summary
