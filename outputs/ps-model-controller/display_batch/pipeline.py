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

from PIL import Image, ImageDraw

from . import VERSION
from .ai import Vision, ProviderUnavailable
from .assets import prepare as prepare_asset
from .common import ROOT, HERE, atomic_json, file_hash, load_catalog, resolve, signature, implementation_manifest
from .compose import compose, sample_environment
from .scene import Scenes, choose_candidates, projection_geometry, fitted_masks, fit_checks
from .schema import (BatchConfig, Review, Probe, MATERIAL_PROFILES, LIGHT_PROFILES,
                     TONE_PROFILES, PHOTO_CONTROLS, StructureConfirmation, apply_actions, version_rank)
from .pile import support_union, diagnostic, depth_order
from .geometry import contact_support
from .store import Store
from .worker import Blender, RenderUnavailable

RENDER_VERSION='pbr-linear-patina-v9'
COMPOSE_VERSION='scrap-gray-pile-v5'


def final_resolution(candidate):
    # Sample the whole render frustum at twice its final scale, not just the object bbox.
    pixels=2*candidate.get('full_px',candidate['target_px']*2.2)
    return int(min(2048,max(512,2**math.ceil(math.log2(max(1,pixels))))))


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
    if config.manifest:
        jobs=json.loads(resolve(config.manifest).read_text(encoding='utf8'))
        if not isinstance(jobs,list) or not 1<=len(jobs)<=1000:raise ValueError('invalid manifest size')
    else:
        directory=resolve(config.backgrounds)
        backgrounds=sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
        if not backgrounds:raise ValueError('没有可读取背景')
        ids=config.assets or list(catalog)
        ids=[str(int(k.split('-')[0]))+('-'+k.split('-')[1] if '-' in k else '') for k in ids]
        rng=random.Random(config.seed);rng.shuffle(backgrounds);rng.shuffle(ids)
        jobs=[{'background':str(backgrounds[(i//len(ids)+i)%len(backgrounds)]),
               'asset_id':ids[i%len(ids)],'group':'batch'} for i in range(config.count)]
    result=[]
    for i,job in enumerate(jobs):
        k=str(job['asset_id']);path=resolve(job['background'])
        if k not in catalog:raise ValueError('未知模型编号：'+k)
        if not path.is_file():raise ValueError('背景不存在：'+str(path))
        seed=job.get('seed',config.seed+i)
        if type(seed) is not int or seed<0:raise ValueError('invalid manifest seed')
        result.append({**job,'background':str(path),'asset_id':k,'background_hash':file_hash(path),
                       'mesh_hash':catalog[k]['mesh_hash'],'seed':seed})
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
        g=groups.setdefault(group,{'total':0,'accepted_display':0,'review':0,'failed':0,'pending':0})
        g['total']+=1;g[row['state'] if row['state'] in g else 'pending']+=1
    for group in groups.values():group['acceptance_rate']=group['accepted_display']/max(1,group['total'])
    times=[r['seconds'] for r in records if 'seconds' in r]
    summary={'version':VERSION,'counts':store.counts(),'groups':groups,'formal_acceptance':'not_changed',
             'run':run,'usage_current_run':usage,'seconds_median':statistics.median(times) if times else None,
             'seconds_p95':sorted(times)[min(len(times)-1,math.ceil(len(times)*.95)-1)] if times else None,
             'generated_pngs':sum(bool(r.get('image')) for r in records),
             'image_rate':sum(bool(r.get('image')) for r in records)/max(1,len(rows))}
    atomic_json(out/'summary.json',summary)
    sections=[]
    for r in records:
        image=Path(r['image']).relative_to(out).as_posix() if r.get('image') else None
        crop=Path(r['crop']).relative_to(out).as_posix() if r.get('crop') else None
        name=f"{r['job_id']} · {r['asset_id']} · {r['status']}"
        sections.append('<section><h2>'+html.escape(name)+'</h2><p>'+html.escape(r.get('reason',''))+'</p>'+
                        (f'<a href="{image}"><img src="{image}" loading="lazy"></a>' if image else '')+
                        (f'<img class="crop" src="{crop}" loading="lazy">' if crop else '')+'</section>')
    page='<!doctype html><meta charset="utf-8"><title>多背景批量结果</title><style>body{background:#172128;color:#eee;font:15px sans-serif;margin:30px}section{border-top:1px solid #456;padding:15px}img{max-width:55%;max-height:440px}.crop{max-width:40%}h2{font-size:16px}</style><h1>多背景批量展示结果</h1><p>场景依据为估计；展示合格不等于旧流程正式验收。</p><pre>'+html.escape(json.dumps(summary['counts'],ensure_ascii=False))+'</pre>'+''.join(sections)
    (out/'index.html').write_text(page,encoding='utf8')
    return summary


class Pipeline:
    def __init__(self,config):
        self.config=config;self.out=resolve(config.output);self.out.mkdir(parents=True,exist_ok=True)
        self.cache=resolve(config.cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.run_id=time.strftime('%Y%m%d-%H%M%S')+f'-{os.getpid()}'
        self.vision=Vision(config,self.cache/'vision',self.out/f'usage-{self.run_id}.json')
        self.blender=Blender(self.cache/('worker-'+str(os.getpid())),config.render_device)
        self.scenes=Scenes(self.cache,self.vision);self.assets={}
        self.catalog=load_catalog();self.store=Store(self.out/'jobs.sqlite')
        self.cpu=asyncio.Semaphore(config.cpu_concurrency)
        self.prepare_lock=asyncio.Lock()
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
                    self.vision.endpoint.model_dump(),implementation_manifest())
                failure_path=self.cache/'asset-failures'/(key+'.json')
                if failure_path.is_file():
                    self.assets[k]=json.loads(failure_path.read_text(encoding='utf8'))
                else:
                    try:self.assets[k]=await prepare_asset(row,self.cache,self.blender,self.vision)
                    except (ProviderUnavailable,RenderUnavailable):raise
                    except Exception as error:
                        self.assets[k]={**row,'signature':key,'poses':[],
                            'reason':f'asset_preparation_failed: {type(error).__name__}: {str(error)[:220]}'}
                        atomic_json(failure_path,self.assets[k])
            return self.assets[k]

    async def render(self,asset,scene,candidate,params,resolution):
        environment,samples=await asyncio.to_thread(sample_environment,scene,candidate)
        req={'mesh':asset['mesh'],'matrix':candidate['matrix'],'yaw':candidate['yaw'],
             'plane_normal':candidate.get('plane_normal',[0,0,1]),
             'elevation':candidate['elevation'],'environment':environment,'resolution':resolution,
             'roughness_delta':MATERIAL_PROFILES[params['material']],
             'dust_gain':1.6 if params['material']=='weathered' else 1.,
             'light_gain':LIGHT_PROFILES[params['light']]}
        if candidate.get('camera'):req['camera']=candidate['camera']
        if candidate.get('receiver',{}).get('patches'):req['receiver']=candidate['receiver']
        key=signature(RENDER_VERSION,asset['mesh_hash'],req,self.config.render_device,'blender-4.5.3')
        folder=self.cache/'renders'/key;folder.mkdir(parents=True,exist_ok=True)
        info=folder/'result.json';hit=info.is_file()
        if hit:meta=json.loads(info.read_text(encoding='utf8'))
        else:
            meta=await self.blender.ask('render',{**req,'output':str(folder)},timeout=240)
            atomic_json(info,meta)
        return folder,meta,environment,samples,hit

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
        receiver=candidate['receiver'];data=np.load(asset['geometry'])
        original=transformed(data['vertices'],candidate['matrix'])
        if original[:,2].min()<-.004:raise ValueError('confirmed_initial_ground_penetration')
        key=signature('finite-support-physics-v1',asset['physics_signature'],candidate['matrix'],receiver['signature'])
        path=Path(asset['folder'])/'terrain'/(key+'.json');hit=path.is_file()
        if receiver['patches']:
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
        geom=projection_geometry(data['vertices'],data['faces'],pose,0,middle,camera=camera)
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
            if not candidate['allow_uncertain']:raise ValueError('support_coverage_insufficient')
            candidate['uncertainties']=list(candidate['uncertainties'])+['support_coverage_insufficient']
        tilt=horizontal_angle(asset['axis'],pose['matrix'])
        if tilt>15:raise ValueError('confirmed_non_horizontal_pose')
        candidate.update(matrix=pose['matrix'],anchor_px=anchor,offset=fit['offset'],bounds=fit['bounds'],
            target_px=max(r-x,b-y),expected_major_px=float(np.ptp(geom['uv'],axis=0).max()*full),
            anchor_uv=geom['anchor'].tolist(),contact_polygons=[p.tolist() for p in geom['contacts']],
            contact_min=min(cover) if cover else 0,axis_angle_deg=tilt)
        endpoint_risks=[]
        for elevation in candidate['elevation_interval']:
            endpoint=projection_geometry(data['vertices'],data['faces'],pose,0,elevation)
            other=fitted_masks(endpoint,anchor,full,scene)
            if other is None:endpoint_risks.append({'elevation':elevation,'reason':'outside_image'});continue
            ox,oy,rr,bb=other['bounds'];m=other['hull']
            coverage=contact_support(
                other['contact'],support_union(scene,candidate['support_ids'],other['bounds']))
            if np.any(m&scene['protected'][oy:bb,ox:rr]) or not coverage or min(coverage)<.9:
                endpoint_risks.append({'elevation':elevation,'reason':'support_or_clearance_uncertain'})
        candidate['camera_risks']=endpoint_risks
        if endpoint_risks and not candidate['allow_uncertain']:raise ValueError('camera_interval_uncertain')
        if receiver['height_confidence']=='unknown':
            candidate['uncertainties']=list(candidate['uncertainties'])+['support_heights_uncertain']
        candidate['terrain_physics']={**pose,'signature':key,'cache_hit':hit,'simulation':simulated,
            'status':'pass' if pose.get('stable') and receiver['height_confidence']!='unknown' else 'unknown',
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
        return {'candidate':candidate,'material':'original','light':'scene','tone':'scene',
                'photo_style':self.config.photo_style,**{k:'auto' for k in PHOTO_CONTROLS}}

    async def job(self,job):
        ident=job['id'];payload=job['payload'];start=time.monotonic();calls_before=len(self.vision.calls)
        asyncio.current_task().set_name(ident)
        folder=self.out/'jobs'/ident;folder.mkdir(parents=True,exist_ok=True)
        result={'job_id':ident,'asset_id':payload['asset_id'],'background':payload['background'],
                'group':payload.get('group','batch'),'status':'review','scene_basis':'estimated',
                'formal_status':'unchanged','iterations':[],'timings':{}}
        best=None
        try:
            held=getattr(self,'model_review',{}).get(payload['asset_id'])
            if held and held.get('independent_confirmation') and held.get('mesh_hash')==payload.get('mesh_hash'):
                result.update(reason='asset_structure_requires_review',model_review=held);return result
            self.store.checkpoint(ident,'asset',job['checkpoint'] or {})
            asset=await self.prepare_asset(payload['asset_id'])
            result['timings']['asset_prepare_seconds']=round(time.monotonic()-start,3)
            result['asset_signature']=asset['signature']
            if not asset['poses']:
                result['reason']=asset['reason'];return result
            self.store.checkpoint(ident,'scene',{})
            stage_start=time.monotonic()
            scene=await self.scenes.prepare(payload['background'])
            result['timings']['scene_seconds']=round(time.monotonic()-stage_start,3)
            result['scene_signature']=scene['key'];result['physics_signature']=asset['physics_signature']
            async with self.cpu:
                stage_start=time.monotonic()
                candidates,rejected=await asyncio.to_thread(choose_candidates,asset,scene,payload['seed'],self.config)
            result['timings']['placement_seconds']=round(time.monotonic()-stage_start,3)
            atomic_json(folder/'candidates.json',{'candidates':candidates,'rejected':rejected})
            result['rejections']=rejected
            if not candidates:
                result['reason']='no_geometric_candidate';return result
            # Render at most three coarse candidates; the same ranked winner starts the final loop.
            coarse=[];validated=[]
            for candidate in candidates:
                try:
                    if 'plane_normal' in candidate:
                        candidate=await self.validate_local_pose(asset,scene,candidate)
                    environment,samples=await asyncio.to_thread(sample_environment,scene,candidate) if 'plane_normal' in candidate else (None,None)
                except ValueError as failure:
                    rejected.append({'candidate':candidate['id'],'reason':str(failure)});continue
                params=self.default_params(candidate['id'])
                rendered=await self.render(asset,scene,candidate,params,512)
                image=folder/(candidate['id']+'-coarse.png')
                comp=await asyncio.to_thread(compose,scene,candidate,*rendered[:4],params,payload['seed'],image)
                if any(c['status']=='fail' for c in comp['checks'].values()):
                    rejected.append({'candidate':candidate['id'],'reason':'explicit_composite_check_failed',
                                     'checks':comp['checks']});continue
                if candidate.get('placement'):
                    diagnostic(scene,candidate,folder/(candidate['id']+'-geometry.jpg'))
                coarse.append({'candidate':candidate['id'],'image':str(image),
                               'hard_failures':sum(c['status']!='pass' for c in comp['checks'].values())})
                validated.append(candidate)
            candidates=validated
            atomic_json(folder/'candidates.json',{'candidates':candidates,'rejected':rejected})
            if not candidates:
                result['reason']='no_candidate_survived_local_physics';return result
            atomic_json(folder/'coarse.json',coarse)
            alternatives=Image.new('RGB',(384*len(coarse),360),(40,44,48));draw=ImageDraw.Draw(alternatives)
            for i,item in enumerate(coarse):
                with Image.open(Path(item['image']).with_suffix('.crop.jpg')) as thumb:
                    thumb=thumb.convert('RGB');thumb.thumbnail((370,325))
                    alternatives.paste(thumb,(384*i+(384-thumb.width)//2,30+(325-thumb.height)//2))
                draw.text((384*i+10,8),item['candidate'],fill='white')
            alternatives.save(folder/'alternatives.jpg',quality=90)
            ranked=sorted(zip(candidates,coarse),key=lambda v:(v[1]['hard_failures'],-v[0]['score']))
            params=self.default_params(ranked[0][0]['id'])
            seen=set();seen_images=set();best=None;best_rank=None;no_gain=0;reason='iterations_exhausted'
            for iteration in range(self.config.max_iterations+1):
                psig=signature(params)
                if psig in seen:reason='repeated_parameters';break
                seen.add(psig);candidate=next(c for c in candidates if c['id']==params['candidate'])
                resolution=final_resolution(candidate)
                folder_render,meta,environment,samples,hit=await self.render(asset,scene,candidate,params,resolution)
                image=folder/f'iteration-{iteration}.png'
                comp=await asyncio.to_thread(compose,scene,candidate,folder_render,meta,environment,samples,
                                              params,payload['seed'],image)
                draft={**comp,'iteration':iteration,'params':params.copy(),
                       'render_cache_hit':hit,'image_hash':file_hash(image)}
                atomic_json(folder/f'draft-{iteration}.json',draft)
                if any(c['status']=='fail' for c in comp['checks'].values()):
                    result['rejections'].append({'candidate':candidate['id'],'reason':'final_program_checks_failed',
                        'checks':comp['checks']})
                    reason='final_program_checks_failed'
                    break
                if draft['image_hash'] in seen_images:
                    reason='no_effective_image_change';break
                seen_images.add(draft['image_hash'])
                self.store.checkpoint(ident,'review_request',{'last_iteration':iteration,
                    'params':params,'draft':str(folder/f'draft-{iteration}.json')})
                review=await self.vision.ask('review',Review,REVIEW_PROMPT,
                    {'name':asset['name'],'features':asset['semantic']['key_features'],'params':params,
                     'allowed':{'candidate':[c['id'] for c in candidates], 'material':list(MATERIAL_PROFILES),
                                'light':list(LIGHT_PROFILES),'tone':list(TONE_PROFILES),
                                **{k:list(v) for k,v in PHOTO_CONTROLS.items()}},
                     'profile_meanings':{'clearer':'reduce roughness, stronger existing metal highlights',
                         'softer':'increase roughness, soften specular reflection',
                         'weathered':'increase light surface dust on exposed paint while retaining bare-metal reflections',
                         'defined':'stronger broad directional light', 'darker':'reduce object exposure'},
                     'estimated_size_mm':asset['semantic']['major_mm'],
                     'checks':comp['checks'],'source_note':'The current reconstructed model preserves user-approved revisions; judge missing geometry against this current model, not the pose of an upright source photo.'},
                     [comp['preview'],comp['crop'],folder_render/'object.png',asset['board'],folder/'alternatives.jpg'])
                entry={**draft,'review':review.model_dump()}
                result['iterations'].append(entry)
                atomic_json(folder/f'iteration-{iteration}.json',entry)
                self.store.checkpoint(ident,'review',{'last_iteration':iteration,'params':params,'record':str(folder/f'iteration-{iteration}.json')})
                rank=version_rank(comp['checks'],review)
                if best_rank is None or rank>best_rank:
                    best,best_rank=entry,rank;no_gain=0
                else:no_gain+=1
                if all(c['status']=='pass' for c in comp['checks'].values()) and review.passes():
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
                if iteration>=self.config.max_iterations:break
                if no_gain>=2:reason='no_improvement';break
                if not review.actions:reason='no_valid_action';break
                try:params=apply_actions(params,review.actions,[c['id'] for c in candidates])
                except ValueError:reason='invalid_ai_action';break
            if best:
                dest=self.out/('accepted' if result['status']=='accepted_display' else 'review')
                dest.mkdir(parents=True,exist_ok=True)
                final=dest/(ident+'.png');shutil.copy2(best['image'],final)
                crop=dest/(ident+'.crop.jpg');shutil.copy2(best['crop'],crop)
                result.update(image=str(final),crop=str(crop),best_iteration=best['iteration'],
                              checks=best['checks'],scores=best['review']['scores'],
                              measured_major_px=best['measured_major_px'],
                              placement=best.get('placement','surface'),support_ids=best.get('support_ids',[]),
                              occlusion_fraction=best.get('occlusion_fraction'),photo=best.get('photo'),
                              evidence_confidence=best.get('evidence_confidence','unknown'))
            result['reason']=reason
            return result
        except (ProviderUnavailable,RenderUnavailable):
            raise
        except Exception as error:
            result.update(status='failed',reason=f'{type(error).__name__}: {str(error)[:260]}')
            # Preserve a concrete image even if review/IO later fails. It is never accepted.
            existing=sorted(folder.glob('iteration-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
            if existing:
                dest=self.out/'review';dest.mkdir(exist_ok=True)
                source=Path(best['image']) if best else existing[-1]
                final=dest/(ident+'.png');shutil.copy2(source,final)
                result['image']=str(final)
                cropped=source.with_suffix('.crop.jpg')
                if cropped.is_file():
                    crop=dest/(ident+'.crop.jpg');shutil.copy2(cropped,crop);result['crop']=str(crop)
                if best:
                    result.update(best_iteration=best['iteration'],checks=best['checks'],
                                  scores=best['review']['scores'],measured_major_px=best['measured_major_px'])
                else:
                    draft_path=folder/('draft-'+source.stem.rsplit('-',1)[1]+'.json')
                    if draft_path.is_file():
                        draft=json.loads(draft_path.read_text(encoding='utf8'))
                        result.update(checks=draft['checks'],draft_record=str(draft_path),unreviewed_draft=True,
                                      measured_major_px=draft['measured_major_px'])
            return result
        finally:
            result['seconds']=round(time.monotonic()-start,3)
            result['new_api_calls']=sum(c.get('job_id')==ident for c in self.vision.calls)
            atomic_json(folder/'result.json',result)

    async def run(self):
        started=time.monotonic();state='complete';error=None
        jobs=make_jobs(self.config,self.catalog)
        implementation=implementation_manifest()
        config_signature=signature(VERSION,self.config.model_dump(exclude={'offline'}),jobs,
                                    RENDER_VERSION,COMPOSE_VERSION,implementation)
        with batch_lock(self.out):
            self.store.initialize(config_signature,jobs);self.store.recover()
            atomic_json(self.out/'config.json',self.config.model_dump())
            atomic_json(self.out/'catalog-preflight.json',catalog_preflight(self.catalog))
            atomic_json(self.out/'inputs.json',jobs)
            atomic_json(self.out/'implementation.json',implementation)
            try:
                if not self.config.offline:
                    probe_path=self.out/'capability-probe.jpg'
                    with Image.open(jobs[0]['background']) as im:
                        im=im.convert('RGB');im.thumbnail((512,512));im.save(probe_path,quality=80)
                    probe=await self.vision.ask('capability',Probe,'Confirm the supplied image is visible; describe it briefly.',{},[probe_path])
                    if not probe.image_visible:raise ProviderUnavailable('视觉服务未确认图像可见')
                stop=asyncio.Event()
                async def consumer():
                    nonlocal state,error
                    while not stop.is_set():
                        job=self.store.claim()
                        if not job:return
                        print(json.dumps({'event':'start','job':job['id'],'asset':job['payload']['asset_id']},ensure_ascii=False),flush=True)
                        try:result=await self.job(job)
                        except (ProviderUnavailable,RenderUnavailable) as failure:
                            self.store.pause(job['id']);state='paused_provider' if isinstance(failure,ProviderUnavailable) else 'paused_renderer'
                            error=str(failure);stop.set();return
                        self.store.finish(job['id'],result['status'],result)
                        write_summary(self.out,self.store,self.vision.usage(),{'id':self.run_id,'state':'running'})
                        print(json.dumps({'event':'done','job':job['id'],'status':result['status'],
                                          'reason':result['reason'],'seconds':result['seconds']},ensure_ascii=False),flush=True)
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
                atomic_json(self.out/'summary.json',summary)
                self.store.close()
            return summary
