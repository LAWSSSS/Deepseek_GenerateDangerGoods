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
from .scene import Scenes, choose_candidates, projection_geometry, fitted_masks, fit_checks
from .schema import (Review, FusionReview, EffectsReview, Probe, PHOTO_CONTROLS, StructureConfirmation, version_rank)
from .pile import support_union, diagnostic, receiver_mesh
from .geometry import contact_support
from .store import Store
from .worker import Blender, RenderUnavailable
from .framing import edit_bounds, final_resolution
from .rendering import RenderCache, RENDER_VERSION
from .review import allowed_actions, evidence_status, fusion_actions, inspect, repair, PLACEMENT_DEFECTS
from .materials import material_kind, SURFACE_DUST
from .atmosphere import Atmospheres
from .costs import cumulative, estimate, tariff, label as cost_label
from .policy import asset_id, resize_candidate
from .effects import parameters as effect_parameters, render_identity

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
    excluded={asset_id(k) for k in config.excluded_assets}
    if config.manifest:
        jobs=json.loads(resolve(config.manifest).read_text(encoding='utf8'))
        if not isinstance(jobs,list) or not 1<=len(jobs)<=1000:raise ValueError('invalid manifest size')
    else:
        directory=resolve(config.backgrounds)
        backgrounds=sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
        if not backgrounds:raise ValueError('没有可读取背景')
        ids=config.assets or list(catalog)
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
        if not path.is_file():raise ValueError('背景不存在：'+str(path))
        seed=job.get('seed',config.seed+i)
        if type(seed) is not int or seed<0:raise ValueError('invalid manifest seed')
        result.append({**job,'background':str(path),'asset_id':k,'background_hash':file_hash(path),
                       'mesh_hash':catalog[k]['mesh_hash'],'seed':seed})
    if not result:raise ValueError('任务清单只包含已移出的模型')
    return result


def catalog_preflight(catalog):
    import struct
    result=[]
    for key,row in catalog.items():
        info={'asset_id':key,'available':row['available'],'model_review':row['human_review'],
              'mesh_hash':row['mesh_hash'],'formal_status':'unchanged'}
        if row['available']:
            with open(row['mesh'],'rb') as f:
                header=f.read(20);length=struct.unpack_from('<I',header,12)[0]
                doc=json.loads(f.read(length))
            info.update(materials=len(doc.get('materials',[])),textures=len(doc.get('textures',[])),
                        mesh_primitives=sum(len(m['primitives']) for m in doc.get('meshes',[])),
                        render_topology_preserved=True)
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
                        ('<details><summary>每轮完整画面与实际参数</summary>'+''.join(history)+'</details>' if history else '')+'</section>')
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>多背景批量结果</title><style>body{background:#172128;color:#eee;font:15px sans-serif;margin:30px auto;max-width:1600px;padding:16px}section{border-top:1px solid #456;padding:15px}img{display:block;width:100%;height:auto}a{color:#b1dfce}h2{font-size:16px}</style><h1>多背景批量展示结果</h1><p>完整画面可点击查看原尺寸。场景依据为估计，展示合格不等于旧流程正式验收。</p><pre>'+html.escape(json.dumps(summary['counts'],ensure_ascii=False))+'</pre>'+''.join(sections)
    page=page.replace('<h1>多背景批量展示结果</h1>','<h1>多背景批量展示结果</h1><p>'+html.escape(cost_label(summary['cost_total']))+'</p>')
    (out/'index.html').write_text(page,encoding='utf8')
    return summary


class Pipeline:
    def __init__(self,config):
        self.config=config;self.out=resolve(config.output);self.out.mkdir(parents=True,exist_ok=True)
        self.cache=resolve(config.cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.run_id=time.strftime('%Y%m%d-%H%M%S')+f'-{os.getpid()}'
        self.vision=Vision(config,self.cache/'vision',self.out/f'usage-{self.run_id}.json')
        self.blender=Blender(self.cache/('worker-'+str(os.getpid())),config.render_device)
        self.renders=RenderCache(self.cache,self.blender)
        self.scenes=Scenes(self.cache,self.vision,structured=config.support_geometry=='structured',
                           structured_json=config.workflow=='evidence_v1');self.assets={}
        self.atmospheres=Atmospheres(self.cache,self.vision)
        self.catalog=load_catalog();self.store=Store(self.out/'jobs.sqlite')
        self.cpu=asyncio.Semaphore(config.cpu_concurrency)
        self.prepare_lock=asyncio.Lock()
        self.width_gates={}
        self.scene_format_failures={}
        self.model_review={}
        self.structure_locks={}
        review_path=self.out/'model-review.json'
        if review_path.is_file():self.model_review=json.loads(review_path.read_text(encoding='utf8'))

    async def prepare_asset(self,k):
        async with self.prepare_lock:
            if k not in self.assets:
                # Failed expensive preparation is cached too; a thousand jobs must not
                # run the same doomed decomposition or malformed asset request repeatedly.
                row=self.catalog[k]
                key=signature('asset-preparation-failure-v1',row['mesh_hash'],
                    file_hash(row['board']) if row.get('board') else None,
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

    async def render(self,asset,scene,candidate,params,resolution,*,sampled=None):
        style=self.render_style
        if style=='native' or params.get('fog_mode')=='local_auto':
            scene['atmosphere']=await self.atmospheres.prepare(scene)
        environment,samples=(sampled if sampled is not None else
            await asyncio.to_thread(sample_environment,scene,candidate,
                scene_lighting=style in ('scene','native'),native=style=='native',local=self.config.appearance=='local'))
        environment={**environment,'integration_mode':params.get('integration_mode','legacy')}
        extras={}
        if params.get('fog_mode')=='local_auto':
            from .local_atmosphere import prepare as prepare_local_atmosphere
            context=await asyncio.to_thread(prepare_local_atmosphere,scene,candidate,environment,self.cache)
            environment={**environment,'atmospheric_context':context}
        if params.get('effects_enabled'):
            from .effects import dust_profile, lighting_for
            from .surface_maps import prepare_uv_masks
            environment=await asyncio.to_thread(lighting_for,scene,candidate,environment,params['effects_lighting'],
                                               preserve=params.get('weathering')=='preserve_source')
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
                                             appearance=style,**extras)
        if params.get('effects_enabled'):
            from .local_light import receiver_calibration
            calibration=await asyncio.to_thread(receiver_calibration,folder,meta,environment)
            environment={**environment,'calibration':calibration}
        return folder,meta,environment,samples,hit

    @property
    def render_style(self):
        if self.config.appearance=='native':return 'native'
        if self.config.appearance=='local':return 'local'
        return 'scene' if self.config.appearance in ('materials','scene') else 'reference'

    async def choose_placements(self,asset,scene,seed):
        preview='estimated_preview_scale' in scene
        if self.config.geometry_preset=='rebuild' and not preview:
            from .vehicle_scale import prepare as prepare_vehicle, projection_preflight
            observation_mode='audited' if self.config.integration_mode=='coherent' else 'legacy'
            if ('vehicle_scale' not in scene or
                    scene['vehicle_scale'].get('observation_mode','legacy')!=observation_mode):
                scene['vehicle_scale']=await prepare_vehicle(scene,self.config.vehicle_reference,self.vision,self.cache,
                    observation_mode=observation_mode)
            scene['vehicle_scale']['enforce_projection_preflight']=self.config.integration_mode=='coherent'
            gate=scene['vehicle_scale'].get('projection_preflight') or projection_preflight(scene['vehicle_scale']['calibration'])
            if self.config.integration_mode=='coherent' and not gate['allow_expensive_steps']:
                atomic_json(self.out/'scale-scenes'/(scene['key']+'.json'),{'vehicle_scale':scene['vehicle_scale'],
                    'projection_preflight':gate})
                return [],[{'reason':'high_risk_vehicle_projection','evidence':gate}]
        if self.config.scale_mode=='reference_projection' or self.config.contact_mode=='strict':
            from .specifications import identify, scene_observations
            from .reference_projection import audit
            from .strict_contact import simulate
            from .assets import initial_poses
            async with self.prepare_lock:
                if 'physical_size' not in asset:
                    asset['physical_size']=await identify(asset,self.vision,self.cache,self.config.dimension_registry)
                if self.config.contact_mode=='strict' and not asset.get('metric_prepared'):
                    data=np.load(asset['geometry']);_,initial=initial_poses(data['vertices'])
                    physics=await simulate(asset,self.blender,initial)
                    poses=[p for p in physics['poses'] if p['stable']]
                    unique=[]
                    for p in sorted(poses,key=lambda q:q['height']):
                        if not any(np.linalg.norm(np.asarray(p['matrix'])[:3,:3]-np.asarray(q['matrix'])[:3,:3])<.05 for q in unique):unique.append(p)
                    poses=unique
                    ext=sorted(asset['inspection']['extents'])
                    if poses and ext[-1]/max(ext[-2],1e-6)<1.5:
                        low=min(p['height'] for p in poses);poses=[p for p in poses if p['height']<=low*1.15]
                    asset.update(poses=poses,metric_prepared=True,metric_physics=physics)
                    atomic_json(self.out/'physics-models'/(str(asset['id'])+'.json'),physics)
            scene['reference_audit']=audit(scene,self.config.scene_reference_registry)
            if preview:
                scene['reference_audit']['preview_scale']=scene['estimated_preview_scale']
            else:
                scene['reference_audit']['visual_observations']=await scene_observations(scene,self.vision,self.cache)
            if self.config.geometry_preset=='rebuild' and not preview:
                from .vehicle_scale import prepare as prepare_vehicle
                if 'vehicle_scale' not in scene:
                    scene['vehicle_scale']=await prepare_vehicle(scene,self.config.vehicle_reference,self.vision,self.cache)
                scene['reference_audit']['vehicle_scale']=scene['vehicle_scale']
            atomic_json(self.out/'scale-models'/(str(asset['id'])+'.json'),asset['physical_size'])
            atomic_json(self.out/'scale-scenes'/(scene['key']+'.json'),scene['reference_audit'])
            if not asset['poses']:
                return [],[{'reason':'no_valid_metric_lying_pose','evidence':asset.get('metric_physics')}]
            return await asyncio.to_thread(choose_candidates,asset,scene,seed,self.config)
        if self.config.scale_mode == 'physical_consistent':
            from .consistent_scale import scaled_asset, scene_calibration
            asset = scaled_asset(asset,self.config.asset_scale_factors.get(str(asset.get('id','')),1.))
            atomic_json(self.out/'scale-models'/(str(asset['id'])+'.json'),asset['physical_size'])
            atomic_json(self.out/'scale-scenes'/(scene['key']+'.json'),scene_calibration(scene))
            return await asyncio.to_thread(choose_candidates,asset,scene,seed,self.config)
        candidates,rejected=await asyncio.to_thread(choose_candidates,asset,scene,seed,self.config)
        factor=self.config.asset_scale_factors.get(str(asset.get('id','')),1.)
        if factor!=1:
            candidates=[await asyncio.to_thread(resize_candidate,asset,scene,c,factor) for c in candidates]
        return candidates,rejected

    async def validate_local_pose(self,asset,scene,candidate):
        if candidate.get('placement'):
            return await self.validate_pile_pose(asset,scene,candidate)
        normal=candidate['plane_normal']
        key=signature('terrain-physics-v1',asset['physics_signature'],candidate['matrix'],normal)
        path=Path(asset['folder'])/'terrain'/(key+'.json')
        hit=path.is_file()
        if hit:physics=json.loads(path.read_text(encoding='utf8'))
        else:
            physics=await self.blender.ask('settle',{'geometry':asset['geometry'],
                'proxy':str(Path(asset['folder'])/'collision.json'),'axis':asset['axis'],
                'initial_matrices':[candidate['matrix']],'assembly':asset['semantic']['assembly'],
                'plane_normal':normal},timeout=90)
            atomic_json(path,physics)
        pose=physics['poses'][0]
        if not pose['stable']:raise ValueError('local_plane_'+','.join(pose['failure_codes']))
        data=np.load(asset['geometry']);candidate=dict(candidate);candidate['matrix']=pose['matrix']
        site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
        minimum=1.;middle=candidate['elevation'];world_ppm=candidate['scale']['world_pixels_per_unit']
        for elevation in sorted(set([*candidate['elevation_interval'],middle])):
            geom=projection_geometry(data['vertices'],data['faces'],pose,0,elevation,normal)
            distance=np.linalg.norm(np.array(geom['camera']['eye'])-geom['camera']['target'])
            full_px=world_ppm*2*distance*math.tan(math.radians(geom['camera']['fov']/2))
            fit=fitted_masks(geom,candidate['anchor_px'],full_px,scene)
            if fit is None:raise ValueError('local_plane_out_of_frame')
            covers,protected,inside,occluded=fit_checks(fit,site,scene)
            if not covers or min(covers)<.9 or not protected or inside<.98 or occluded>.3:
                raise ValueError('local_plane_contact_or_clearance_failed')
            minimum=min(minimum,min(covers))
            if elevation==middle:
                x,y,r,b=fit['bounds']
                candidate.update(full_px=full_px,target_px=max(r-x,b-y),offset=fit['offset'],
                                 bounds=fit['bounds'],anchor_uv=geom['anchor'].tolist(),
                                 expected_major_px=float(np.ptp(geom['uv'],axis=0).max()*full_px))
        candidate.update(contact_min=minimum,terrain_physics={**pose,'signature':key,'cache_hit':hit,
                         'plane_normal':normal,'basis':candidate['plane_basis']})
        return candidate

    async def validate_pile_pose(self,asset,scene,candidate):
        from .geometry import transformed,horizontal_angle
        candidate=dict(candidate);candidate['allow_uncertain']=self.config.uncertain_policy=='render_review'
        strict=self.config.contact_mode=='strict'
        receiver=candidate['receiver'];data=np.load(asset['geometry'])
        original=transformed(data['vertices'],candidate['matrix'])
        if not strict and original[:,2].min()<-.004:raise ValueError('confirmed_initial_ground_penetration')
        key=signature('finite-support-physics-v1',asset['physics_signature'],candidate['matrix'],receiver['signature'])
        path=Path(asset['folder'])/'terrain'/(key+'.json');hit=path.is_file()
        if strict:
            from .strict_contact import simulate, validate_receiver, projected_gap
            from .metric_physics import tolerances
            validate_receiver(candidate)
            physics=await simulate(asset,self.blender,[candidate['matrix']],receiver,candidate['plane_normal'])
            simulated=physics['poses'][0];key=physics['signature'];hit=physics['cache_hit']
            if not simulated['stable']:
                raise ValueError('strict_contact_'+','.join(simulated['failure_codes']))
            gap=projected_gap(simulated,candidate)
            candidate['contact_gap']=gap
            if gap.get('contact_evidence_version'):
                atomic_json(self.out/'contact-evidence'/(signature(key,candidate.get('id'))+'.json'),
                    {'candidate_id':candidate.get('id'),'physics_signature':key,'contact_gap':gap})
            if gap['status']=='fail':
                raise ValueError('visible_contact_support_below_90pct' if
                    gap.get('visible_support',{}).get('status')=='fail' else 'visible_contact_gap_over_2px')
            unit=asset['physical_size']['nominal_mm']/1000
            candidate['contact_band']=tolerances(unit)['contact_m']/unit
        elif receiver['patches']:
            if hit:physics=json.loads(path.read_text(encoding='utf8'))
            else:
                physics=await self.blender.ask('settle',{'geometry':asset['geometry'],
                    'proxy':str(Path(asset['folder'])/'collision.json'),'axis':asset['axis'],
                    'initial_matrices':[candidate['matrix']],'assembly':asset['semantic']['assembly'],
                    'plane_normal':[0,0,1],'receiver':receiver},timeout=90)
                atomic_json(path,physics)
            simulated=physics['poses'][0]
        else:simulated={'stable':False,'failure_codes':['receiver_geometry_insufficient']}
        pose={'matrix':candidate['matrix'],'stable':False,'status':'unknown','axis_angle_deg':candidate['axis_angle_deg']}
        if simulated['stable']:
            pose=simulated
        elif not candidate['allow_uncertain']:
            raise ValueError('estimated_support_simulation_'+','.join(simulated['failure_codes']))
        else:
            candidate['uncertainties']=list(candidate['uncertainties'])+['hidden_support_or_height_unknown']
        # Keep the scene camera fixed when a simulated object moves; update the contact pixel.
        middle=candidate['elevation'];camera=candidate['camera'];full=candidate['full_px']
        geom=projection_geometry(data['vertices'],data['faces'],pose,0,middle,
                                 candidate['plane_normal'] if strict else (0,0,1),camera=camera,
                                 contact_band=candidate.get('contact_band',.008),receiver=receiver if strict else None)
        if geom['anchor'] is None:raise ValueError('no_visible_contact_geometry')
        anchor=(np.asarray(candidate['offset'])+geom['anchor']*full).tolist()
        fit=fitted_masks(geom,anchor,full,scene)
        if fit is None:raise ValueError('local_pose_outside_image')
        x,y,r,b=fit['bounds'];mask=fit['hull']
        if np.any(mask&scene['protected'][y:b,x:r]):raise ValueError('protected_region_conflict')
        if (mask&scene['allowed'][y:b,x:r]).sum()/max(1,mask.sum())<.98:raise ValueError('local_pose_outside_hopper')
        cover=contact_support(
            fit['contact'],support_union(scene,candidate['support_ids'],fit['bounds']))
        if not cover or min(cover)<.9:
            if strict or not candidate['allow_uncertain']:raise ValueError('support_coverage_insufficient')
            candidate['uncertainties']=list(candidate['uncertainties'])+['support_coverage_insufficient']
        tilt=horizontal_angle(asset['axis'],pose['matrix'])
        if tilt>15:raise ValueError('confirmed_non_horizontal_pose')
        candidate.update(matrix=pose['matrix'],anchor_px=anchor,offset=fit['offset'],bounds=fit['bounds'],
            target_px=max(r-x,b-y),expected_major_px=float(np.ptp(geom['uv'],axis=0).max()*full),
            anchor_uv=geom['anchor'].tolist(),contact_polygons=[p.tolist() for p in geom['contacts']],
            contact_min=min(cover) if cover else 0,axis_angle_deg=tilt)
        endpoint_risks=[]
        for elevation in ([] if candidate['scale'].get('calibration',{}).get('projectable') else candidate['elevation_interval']):
            endpoint=projection_geometry(data['vertices'],data['faces'],pose,0,elevation,
                                         candidate['plane_normal'] if strict else (0,0,1),
                                         contact_band=candidate.get('contact_band',.008),receiver=receiver if strict else None)
            other=fitted_masks(endpoint,anchor,full,scene)
            if other is None:endpoint_risks.append({'elevation':elevation,'reason':'outside_image'});continue
            ox,oy,rr,bb=other['bounds'];m=other['hull']
            coverage=contact_support(
                other['contact'],support_union(scene,candidate['support_ids'],other['bounds']))
            if np.any(m&scene['protected'][oy:bb,ox:rr]) or not coverage or min(coverage)<.9:
                endpoint_risks.append({'elevation':elevation,'reason':'support_or_clearance_uncertain'})
        candidate['camera_risks']=endpoint_risks
        if candidate['scale'].get('version','').startswith('vehicle-width-projection'):
            from .vehicle_scale import check_projection_interval
            interval=check_projection_interval(scene,candidate,pose,data['vertices'],data['faces'])
            candidate['camera_interval_check']=interval
            candidate['camera_risks']=[{'reason':interval['reason']}]+[
                r for r in interval['checks'] if r['status']!='pass']
        if endpoint_risks and not candidate['allow_uncertain']:raise ValueError('camera_interval_uncertain')
        if receiver['height_confidence']=='unknown':
            candidate['uncertainties']=list(candidate['uncertainties'])+['support_heights_uncertain']
        candidate['terrain_physics']={**pose,'signature':key,'cache_hit':hit,'simulation':simulated,
            'status':'pass' if pose.get('stable') and (receiver['height_confidence']=='measured' if strict else receiver['height_confidence']!='unknown') else 'unknown',
            'basis':receiver['basis'],'hidden_heights':receiver['hidden_heights']}
        return candidate

    async def confirm_structure(self,asset,candidate,scene,defects):
        # One cached, independent unoccluded orbit check per model; no single composite quarantines it.
        key=signature('structure-confirm-v1',asset['mesh_hash'],file_hash(asset['board']))
        folder=self.cache/'structure-confirm'/key;folder.mkdir(parents=True,exist_ok=True)
        answer_path=folder/'answer.json'
        if answer_path.is_file():return StructureConfirmation.model_validate_json(answer_path.read_text(encoding='utf8'))
        environment,_=sample_environment(scene,candidate)
        render_path=folder/'result.json'
        if not render_path.is_file():
            metadata=await self.blender.ask('render',{'mesh':asset['mesh'],'matrix':asset['poses'][0]['matrix'],
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
                **({'workflow':self.config.workflow} if self.config.workflow!='legacy' else {}),
                **({'review_profile':'photographer'} if getattr(self.config,'review_profile','legacy')=='photographer' else {}),
                **{k:'auto' for k in PHOTO_CONTROLS},**effect_parameters(self.config.effects)}

    @staticmethod
    def finish_image(out,ident,result,best):
        if best.get('deliverable') is False:raise ValueError('physical_failure_is_not_deliverable')
        estimated=(best['checks'].get('scale',{}).get('evidence',{}).get('version')=='reference-projection-v1'
                   and best['checks']['scale']['status']!='pass')
        dest=out/('estimated' if estimated else 'accepted' if result['status']=='accepted_display' else 'review')
        result['delivery_group']='estimated_scale' if estimated else 'normal'
        dest.mkdir(parents=True,exist_ok=True)
        final=dest/(ident+'.png');shutil.copy2(best['image'],final)
        crop=dest/(ident+'.crop.jpg');shutil.copy2(best['crop'],crop)
        review_type=EffectsReview if best['params'].get('effects_enabled') else FusionReview if 'trace' in best['review'] else Review
        review=review_type.model_validate(best['review'])
        result.update(image=str(final),crop=str(crop),best_iteration=best['iteration'],
                      checks=best['checks'],scores=best['review']['scores'],
                      measured_major_px=best['measured_major_px'],
                      placement=best.get('placement','surface'),support_ids=best.get('support_ids',[]),
                      occlusion_fraction=best.get('occlusion_fraction'),photo=best.get('photo'),
                      evidence_confidence=best.get('evidence_confidence','unknown'),
                      evidence_status=evidence_status(best['checks']),
                      photo_status='pass' if review.photo_passes() else 'fail',
                      visual_status='pass' if review.passes() else 'fail')
        if isinstance(review,(FusionReview,EffectsReview)):result['insertion_trace']=review.trace

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
        best=None;draft=None
        try:
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
            while iteration<=self.config.max_iterations:
                psig=signature(params)
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
                draft={**comp,'iteration':iteration,'params':params.copy(),'render_cache_hit':hit,
                       'render_cache':meta.get('render_cache',{}),'render_resolution':meta.get('resolution'),
                       'image_hash':file_hash(image)}
                if structured:
                    from .structured import algorithm_versions
                    draft['algorithm_versions']=algorithm_versions(params,comp['photo'].get('version'))
                if result['actions'] and result['actions'][-1]['iteration']==iteration-1:
                    result['actions'][-1]['effect']={'params':params.copy(),'image_hash':draft['image_hash'],
                        'changed_pixels':draft['image_hash']!=result['iterations'][-1]['image_hash'],
                        'repeated_image':draft['image_hash'] in seen_images}
                if draft['image_hash'] in seen_images:
                    reason='no_effective_image_change';break
                seen_images.add(draft['image_hash'])
                atomic_json(folder/f'draft-{iteration}.json',draft)
                self.store.checkpoint(ident,'review_request',{'last_iteration':iteration,
                    'params':params,'draft':str(folder/f'draft-{iteration}.json')})
                choices=[c for c in candidates if c['id'] not in invalid]
                ids=[c['id'] for c in choices]
                allowed=allowed_actions(params,comp,ids,render_style=self.render_style,adjustable=adjustable)
                geometry_board=folder/(cid+'-geometry.jpg') if previous_candidate and previous_candidate!=cid and candidate.get('placement') else None
                result['cost']['logical_reviews']+=1
                observation=None
                if structured:
                    from .structured import inspect_frame
                    profile=getattr(self.config,'review_profile','legacy')
                    observation,review=await inspect_frame(self.vision,asset,comp,draft['image_hash'],
                                                           profile=profile)
                else:
                    review=await inspect(self.vision,REVIEW_PROMPT,asset,comp,params,allowed,folder_render,
                        compact=self.config.review_format=='compact',geometry_board=geometry_board,
                        candidate_evidence=[{'id':c['id'],'placement':c.get('placement'),
                                             'uncertainties':c.get('uncertainties',[])} for c in choices])
                previous_candidate=cid
                entry={**draft,'review':review.model_dump()}
                if observation is not None:
                    if profile=='photographer':
                        entry['photography_diagnosis']=observation.model_dump(mode='json')
                        entry['visual_evidence']=observation.legacy().model_dump(mode='json')
                    else:
                        entry['visual_evidence']=observation.model_dump(mode='json')
                if self.config.contact_mode=='strict' or (structured and profile=='photographer'):
                    from .strict_contact import visual_veto
                    veto=visual_veto(review);entry['deliverable']=not veto
                    entry['physical_visual_veto']=veto
                result['iterations'].append(entry)
                atomic_json(folder/f'iteration-{iteration}.json',entry)
                self.store.checkpoint(ident,'review',{'last_iteration':iteration,'params':params,'record':str(folder/f'iteration-{iteration}.json')})
                if entry.get('deliverable') is False:
                    invalid.add(cid);rejected.append({'candidate':cid,'reason':'visual_physical_failure','defects':veto})
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
                if (structured and profile=='photographer' and any(d.visibility=='visible' and d.code in
                        ('floating','penetration','occlusion','upright','scale','camera') for d in review.defects)):
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
                if structured:
                    from .structured import plan_frame
                    updated,actions,plan=await plan_frame(self.vision,observation,review,comp,params,allowed,ids,
                        lambda p:signature(p) in seen,dust_in_shader=self.render_style in ('scene','native'),
                        recommendation=result.get('parameter_recommendation'))
                    result['actions'].append({'iteration':iteration,'decisions':actions,'execution_plan':plan})
                else:
                    suggestions=fusion_actions(review,comp,allowed) if isinstance(review,FusionReview) else review.actions
                    if not suggestions:reason='no_valid_action';break
                    updated,actions=repair(params,review,allowed,ids,lambda p:signature(p) in seen,
                                           dust_in_shader=self.render_style in ('scene','native'),suggestions=suggestions)
                    result['actions'].append({'iteration':iteration,'decisions':actions})
                if updated is None:
                    if structured and profile=='photographer' and plan:
                        reason=plan['reason']
                    else:
                        reason='repeated_parameters' if actions and all(a['disposition']=='repeated_parameters' for a in actions) else 'no_effective_action'
                    break
                params=updated;iteration+=1
            if best:
                self.finish_image(self.out,ident,result,best)
            elif self.config.contact_mode=='strict':
                result.update(status='failed',evidence_status='fail')
            result['reason']=reason
            return result
        except (ProviderUnavailable,RenderUnavailable):
            raise
        except Exception as error:
            result.update(status='failed',reason=f'{type(error).__name__}: {str(error)[:260]}')
            if best:
                if self.config.contact_mode=='strict':result['status']='review'
                self.finish_image(self.out,ident,result,best)
            elif draft:
                result.update(checks=draft['checks'],draft_record=str(folder/f"draft-{draft['iteration']}.json"),
                    unreviewed_draft=True,measured_major_px=draft['measured_major_px'],
                    evidence_status=evidence_status(draft['checks']))
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

    def reuse_completed(self,source,implementation):
        source=Path(source).resolve()
        source_manifest=json.loads((source/'implementation.json').read_text(encoding='utf8'))
        source_config=json.loads((source/'config.json').read_text(encoding='utf8'))
        exclude={'output','manifest','count','offline'}
        source_vision=json.loads((source/'vision-signature.json').read_text(encoding='utf8'))
        source_evidence=source/'dimension-reference-signature.json'
        if source_evidence.is_file() and json.loads(source_evidence.read_text(encoding='utf8'))!=self.dimension_reference_signature():
            raise ValueError('pilot dimension/reference evidence has changed')
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
        started=time.monotonic();state='complete';error=None
        jobs=make_jobs(self.config,self.catalog)
        implementation=implementation_manifest()
        config_signature=signature(VERSION,self.config.model_dump(exclude={'offline'}),jobs,
                                    RENDER_VERSION,COMPOSE_VERSION,implementation,self.vision.endpoint.model_dump(),
                                    self.vision.planner.model_dump() if self.config.workflow=='evidence_v1' else None,
                                    self.dimension_reference_signature())
        with batch_lock(self.out):
            self.store.initialize(config_signature,jobs);self.store.recover()
            atomic_json(self.out/'config.json',self.config.model_dump())
            atomic_json(self.out/'catalog-preflight.json',catalog_preflight(self.catalog))
            atomic_json(self.out/'inputs.json',jobs)
            atomic_json(self.out/'implementation.json',implementation)
            atomic_json(self.out/'vision-signature.json',{'signature':signature(self.vision.endpoint.model_dump())})
            atomic_json(self.out/'dimension-reference-signature.json',self.dimension_reference_signature())
            try:
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
                        except (ProviderUnavailable,RenderUnavailable) as failure:
                            self.store.pause(job['id']);state='paused_provider' if isinstance(failure,ProviderUnavailable) else 'paused_renderer'
                            error=str(failure);stop.set();return
                        self.store.finish(job['id'],result['status'],result)
                        progress=write_summary(self.out,self.store,self.vision.usage(),{'id':self.run_id,'state':'running'})
                        print(json.dumps({'event':'done','job':job['id'],'status':result['status'],
                                          'reason':result['reason'],'seconds':result['seconds'],
                                          'cost_total':cost_label(progress['cost_total'])},ensure_ascii=False),flush=True)
                await asyncio.gather(*(consumer() for _ in range(self.config.cpu_concurrency)))
            except ProviderUnavailable as failure:
                state='paused_provider';error=str(failure)
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
