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
                     TONE_PROFILES, apply_actions, version_rank)
from .store import Store
from .worker import Blender, RenderUnavailable

RENDER_VERSION='pbr-linear-shadow-v7'
COMPOSE_VERSION='highlight-preserving-v3'


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
        result.append({**job,'background':str(path),'asset_id':k,'background_hash':file_hash(path),
                       'mesh_hash':catalog[k]['mesh_hash'],'seed':config.seed+i})
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
             'seconds_p95':sorted(times)[min(len(times)-1,math.ceil(len(times)*.95)-1)] if times else None}
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
        key=signature(RENDER_VERSION,asset['mesh_hash'],req,self.config.render_device,'blender-4.5.3')
        folder=self.cache/'renders'/key;folder.mkdir(parents=True,exist_ok=True)
        info=folder/'result.json';hit=info.is_file()
        if hit:meta=json.loads(info.read_text(encoding='utf8'))
        else:
            meta=await self.blender.ask('render',{**req,'output':str(folder)},timeout=240)
            atomic_json(info,meta)
        return folder,meta,environment,samples,hit

    async def validate_local_pose(self,asset,scene,candidate):
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
            if held and held.get('mesh_hash')==payload.get('mesh_hash'):
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
                candidates,rejected=await asyncio.to_thread(choose_candidates,asset,scene,payload['seed'])
            result['timings']['placement_seconds']=round(time.monotonic()-stage_start,3)
            atomic_json(folder/'candidates.json',{'candidates':candidates,'rejected':rejected})
            result['rejections']=rejected
            if not candidates:
                result['reason']='no_supported_lying_candidate';return result
            # Render at most three coarse candidates; the same ranked winner starts the final loop.
            coarse=[];validated=[]
            for candidate in candidates:
                try:
                    if 'plane_normal' in candidate:
                        candidate=await self.validate_local_pose(asset,scene,candidate)
                    environment,samples=await asyncio.to_thread(sample_environment,scene,candidate) if 'plane_normal' in candidate else (None,None)
                except ValueError as failure:
                    rejected.append({'candidate':candidate['id'],'reason':str(failure)});continue
                params={'candidate':candidate['id'],'material':'original','light':'scene','tone':'scene'}
                rendered=await self.render(asset,scene,candidate,params,512)
                image=folder/(candidate['id']+'-coarse.png')
                comp=await asyncio.to_thread(compose,scene,candidate,*rendered[:4],params,payload['seed'],image)
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
            params={'candidate':ranked[0][0]['id'],'material':'original','light':'scene','tone':'scene'}
            seen=set();best=None;best_rank=None;no_gain=0;reason='iterations_exhausted'
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
                self.store.checkpoint(ident,'review_request',{'last_iteration':iteration,
                    'params':params,'draft':str(folder/f'draft-{iteration}.json')})
                review=await self.vision.ask('review',Review,REVIEW_PROMPT,
                    {'name':asset['name'],'features':asset['semantic']['key_features'],'params':params,
                     'allowed':{'candidate':[c['id'] for c in candidates], 'material':list(MATERIAL_PROFILES),
                                'light':list(LIGHT_PROFILES),'tone':list(TONE_PROFILES)},
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
                if any(d.code=='structure' and d.scope=='asset' and d.severity in ('major','fatal') for d in review.defects):
                    if not hasattr(self,'model_review'):self.model_review={}
                    self.model_review[payload['asset_id']]={'mesh_hash':payload.get('mesh_hash'),
                        'job_id':ident,'defects':[d.model_dump() for d in review.defects if d.scope=='asset']}
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
                              measured_major_px=best['measured_major_px'])
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
