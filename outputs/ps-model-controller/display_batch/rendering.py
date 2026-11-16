"""Publish complete render parts independently and describe their pixel grids."""
import asyncio
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import tempfile
import time

from PIL import Image

from .common import HERE, ROOT, atomic_json, file_hash, signature
from .framing import render_windows
from .schema import LIGHT_PROFILES, MATERIAL_PROFILES, PHOTO_CONTROLS


REFERENCE_RENDER_VERSION = 'pbr-linear-patina-v9'
RENDER_VERSION = 'render-parts-v2-material-layer'
FINISH_PASS = 'generated-finish0001.exr'
HYDRAULIC_PASS = 'hydraulic-finish0001.exr'
CORROSION_PASS = 'corrosion-finish0001.exr'
HANDLING_PASS = 'handling-finish0001.exr'
PART_FILES = {
    'object': ('object.png', 'combined0001.exr', 'diffuse0001.exr', 'specular0001.exr',
               'depth0001.exr', 'metallic0001.exr', 'upward0001.exr'),
    'with_shadow': ('with-shadow-linear0001.exr',),
    'without_shadow': ('without-shadow-linear0001.exr',),
}


def reference_resolution(candidate):
    pixels=2*candidate.get('full_px',candidate['target_px']*2.2)
    return int(min(2048,max(512,2**math.ceil(math.log2(max(1,pixels))))))


def render_request(asset,candidate,params,environment):
    request={'mesh':asset['mesh'],'matrix':candidate['matrix'],'yaw':candidate['yaw'],
             'plane_normal':candidate.get('plane_normal',[0,0,1]),
             'elevation':candidate['elevation'],'environment':environment,
             'roughness_delta':MATERIAL_PROFILES[params['material']],
             'dust_gain':1.6 if params['material']=='weathered' else 1.,
             'light_gain':LIGHT_PROFILES[params['light']]}
    request['integration_mode']=params.get('integration_mode','legacy')
    request['render_detail']=params.get('render_detail','standard')
    request['rust_spots']=float(params.get('rust_spots',0.))
    request['surface_dust']=float(params.get('surface_dust',0.))
    if params.get('rust_pattern','procedural')=='photo' and request['rust_spots']:
        path=ROOT/'work/render-materials/polyhaven/rusty_metal_03/rusty_metal_03_diff_2k.jpg'
        if not path.is_file():raise FileNotFoundError('photo rust texture missing: '+str(path))
        request['rust_scan']={'path':str(path),'sha256':file_hash(path),'source':'Poly Haven rusty_metal_03 CC0'}
        if params.get('rust_relief',False):
            rough=path.with_name('rusty_metal_03_rough_2k.exr')
            if not rough.is_file():raise FileNotFoundError('photo rust roughness missing: '+str(rough))
            height=path.with_name('rusty_metal_03_disp_2k.exr')
            if not height.is_file():raise FileNotFoundError('photo rust height missing: '+str(height))
            request['rust_scan'].update(roughness_path=str(rough),roughness_sha256=file_hash(rough),
                                        height_path=str(height),height_sha256=file_hash(height),
                                        relief_strength=float(params.get('rust_relief_strength',1.)))
            if request['rust_scan']['relief_strength']>1.:
                detail=path.with_name('rusty_metal_03_detail_2k.png')
                if not detail.is_file():raise FileNotFoundError('photo rust detail missing: '+str(detail))
                request['rust_scan'].update(detail_path=str(detail),detail_sha256=file_hash(detail))
    if candidate.get('camera'):request['camera']=candidate['camera']
    if candidate.get('contact_band'):request['contact_band']=candidate['contact_band']
    if candidate.get('receiver',{}).get('patches'):request['receiver']=candidate['receiver']
    return request


def reference_path(cache,asset,candidate,params,environment,device='OPTIX',resolution=None):
    request=render_request(asset,candidate,params,environment)
    request['resolution']=resolution or reference_resolution(candidate)
    key=signature(REFERENCE_RENDER_VERSION,asset['mesh_hash'],request,device,'blender-4.5.3')
    return Path(cache)/'renders'/key,request


@lru_cache(maxsize=1)
def render_dependencies():
    files=['blender_display_worker.py','blender_worker.py','display_batch/geometry.py',
           'display_batch/framing.py','display_batch/materials.py','display_batch/rendering.py','display_batch/contact_geometry.py',
           'display_batch/scanned_shader.py','display_batch/local_light.py']
    return signature('blender-4.5.3',{name:file_hash(HERE/name) for name in files})


@lru_cache(maxsize=1)
def finish_dependencies():
    return signature(*(file_hash(HERE / 'display_batch' / name)
                       for name in ('generated_material.py', 'generated_shader.py', 'surface_maps.py', 'cylinder_wear.py')))


def part_keys(request, device, dependencies, *, include_detail=True):
    windows=request['windows']
    fields=('world_color','light_color','key_direction','light_level','light_spread','reflection_probe','local_irradiance','integration_mode')
    environment=request['environment']
    light={k:environment[k] for k in fields if k in environment}
    light['calibrated']=bool(environment.get('calibration'))
    shared=(RENDER_VERSION,dependencies,device,request['appearance'],request['camera'],
            request['plane_normal'],request.get('receiver',{}).get('patches',[]),light,request['light_gain'],request.get('integration_mode','legacy'))
    if include_detail:
        shared=(*shared,request.get('render_detail','standard'))
    if request.get('contact_band'):shared=(*shared,request['contact_band'])
    surface=(request['mesh_hash'],request['matrix'],request['yaw'],request['roughness_delta'],
             request['dust_gain'],request['dust_control'],request.get('surface_dust',0.),request['material_classes'],
             request.get('rust_spots',0.),request.get('rust_scan'),
             environment.get('dust_color'),environment.get('haze') if request['appearance']!='native' else None,
             environment.get('local_context',{}).get('surface_texture_rms'))
    if request.get('scanned_hdri'):
        shared=(*shared,request['scanned_hdri'])
    if request.get('surface_scans'):
        surface=(*surface,request['surface_scans']['signature'])
    if request.get('matte_partitions'):
        surface=(*surface,request['matte_partitions']['signature'])
    if request.get('matte_profile'):
        surface=(*surface,request['matte_profile'])
    if finish := request.get('generated_finish'):
        surface=(*surface, finish['signature'], finish['regions']['signature'], finish_dependencies())
    return {
        'object': signature('object',shared,surface,windows['object']),
        'with_shadow': signature('with_shadow',shared,surface,windows['shadow']),
        'without_shadow': signature('without_shadow',shared,windows['shadow']),
    }


class RenderCache:
    def __init__(self, cache, worker):
        self.root=Path(cache)/RENDER_VERSION
        self.worker=worker
        self.lock=asyncio.Lock()
        self.totals=Counter()
        self.timings=Counter()

    @staticmethod
    def ready(folder, part, *, extra_files=()):
        marker=folder/'result.json'
        if not marker.is_file():return None,'no_entry'
        try:
            info=json.loads(marker.read_text(encoding='utf8'))
            files = (*PART_FILES[part], *extra_files)
            if all((Path(info['path'])/name).stat().st_size>0 for name in files):
                return info,None
        except (OSError,ValueError,KeyError,TypeError):
            pass
        return None,'incomplete_entry'

    async def get(self, asset, candidate, params, environment, size, resolution, *,
                  appearance='scene', full_frame=False, surface_scans=None, scanned_hdri=None,
                  matte_partitions=None, object_only=False, matte_profile=None, generated_finish=None,
                  shadow_reference=None):
        if not candidate.get('camera'):
            raise ValueError('cropped_render_requires_fixed_camera')
        request=render_request(asset,candidate,params,environment)
        windows=render_windows(candidate,size,resolution,full_frame=full_frame)
        request.update(mesh_hash=asset['mesh_hash'],appearance=appearance,windows=windows,
                       material_classes=asset['semantic']['material_classes'],
                       dust_control=PHOTO_CONTROLS['dust'][params.get('dust','auto')] if appearance in ('scene','native') else 1.)
        if surface_scans is not None:
            request['surface_scans']=surface_scans
        if scanned_hdri is not None:
            request['scanned_hdri']=scanned_hdri
        if matte_partitions is not None:
            request['matte_partitions']=matte_partitions
        if matte_profile is not None:
            request['matte_profile']=matte_profile
        if generated_finish is not None:
            request['generated_finish'] = generated_finish
        keys=part_keys(request,self.worker.device,render_dependencies())
        archived_receiver = None if object_only else self.archived_receiver(request, shadow_reference)
        if object_only:
            keys={'object':keys['object']}
        queued=time.monotonic()
        async with self.lock:
            timings={'render_cache_wait_seconds':time.monotonic()-queued}
            return await self._get(request,keys,timings,full_frame,archived_receiver)

    def archived_receiver(self, request, reference):
        """Replay only the receiver render already keyed by identical geometry and light."""
        if reference is None:
            return None
        dependencies = reference['dependencies']
        include_detail = not reference.get('standard_profile_without_detail_key')
        if not include_detail and request.get('render_detail', 'standard') != 'standard':
            raise ValueError('frozen_receiver_sample_profile_changed')
        old_key = part_keys(request, self.worker.device, dependencies, include_detail=include_detail)['without_shadow']
        if old_key != reference['signature']:
            raise ValueError('frozen_receiver_geometry_or_light_changed')
        ready, _ = self.ready(self.root / 'without_shadow' / old_key, 'without_shadow')
        if not ready:
            raise ValueError('frozen_receiver_reference_missing')
        return ready

    async def _get(self,request,keys,timings,full_frame,shadow_reference=None):
        windows=request['windows']
        finish = request.get('generated_finish')
        object_passes = ()
        if finish:
            object_passes = (FINISH_PASS,)
            if 'hydraulic' in finish['maps']:
                object_passes += (HYDRAULIC_PASS,)
            if finish.get('corrosion'):
                object_passes += (CORROSION_PASS,)
            if 'handling' in finish['maps']:
                object_passes += (HANDLING_PASS,)
        entries={};cache={};missing={};folders={};imports=0;worker_detail={}
        for part,key in keys.items():
            folder=self.root/part/key
            folders[part]=folder
            extra_files = object_passes if part == 'object' else ()
            ready,reason=self.ready(folder,part,extra_files=extra_files)
            if part == 'without_shadow' and not ready and shadow_reference is not None:
                ready,reason=shadow_reference,None
                atomic_json(folder/'result.json', ready)
            cache[part]={'signature':key,'hit':bool(ready),'miss_reason':reason}
            if ready:
                entries[part]=ready
                self.totals[part+'_hits']+=1
            else:
                folder.mkdir(parents=True,exist_ok=True)
                # A killed worker cannot leave a cache hit or mix two attempts.
                missing[part]=tempfile.mkdtemp(prefix='run-',dir=folder)
        if missing:
            response=await self.worker.ask('render_parts',{**request,'parts':missing},timeout=240)
            worker_detail=response.get('render_mesh_detail',{})
            timings.update(response['timings'])
            imports=response['imports']
            self.totals['asset_imports']+=imports
            for part,path in missing.items():
                window=windows['object' if part=='object' else 'shadow']['rect']
                files = (*PART_FILES[part], *object_passes) if part == 'object' else PART_FILES[part]
                if not all((Path(path)/name).is_file() for name in files):
                    raise ValueError('render_part_missing_files: '+part)
                if part=='object':
                    with Image.open(Path(path)/'object.png') as image:
                        if image.size!=(window[2]-window[0],window[3]-window[1]):
                            raise ValueError('unexpected_render_window_size')
                info={'path':path,'camera':response['camera'],'device':response['device'],
                      'anchor_uv':response['anchor_uv'],'contact_polygons':response['contact_polygons'],
                      'window':windows['object' if part=='object' else 'shadow'],
                      'receiver_signature':signature(request.get('receiver',{}).get('patches',[]))}
                if part=='object' and request.get('surface_scans'):
                    info['scanned_materials']=response['scanned_materials']
                if part=='object' and request.get('matte_partitions'):
                    info['matte_materials']=response['matte_materials']
                if part == 'object' and request.get('generated_finish'):
                    info['generated_finish'] = response['generated_finish']
                if part=='with_shadow' and request.get('scanned_hdri'):
                    info.update(shadow_basis=response['shadow_basis'],
                                shadow_catcher_fallback=response['shadow_catcher_fallback'])
                entries[part]=info
                atomic_json(folders[part]/'result.json',info)
                self.totals[part+'_renders']+=1
        self.timings.update(timings)
        obj=entries['object']
        shadow_pair=None
        if 'with_shadow' in entries and 'without_shadow' in entries:
            import cv2
            expected=(windows['shadow']['rect'][3]-windows['shadow']['rect'][1],
                      windows['shadow']['rect'][2]-windows['shadow']['rect'][0])
            for part in ('with_shadow','without_shadow'):
                entry=entries[part]
                if entry['camera']!=obj['camera']:raise ValueError('shadow_camera_mismatch')
                if entry.get('window',windows['shadow'])!=windows['shadow']:raise ValueError('shadow_grid_mismatch')
                frame=cv2.imread(str(Path(entry['path'])/PART_FILES[part][0]),cv2.IMREAD_UNCHANGED)
                if frame is None or frame.shape[:2]!=expected:raise ValueError('shadow_pass_dimensions_mismatch')
            a,b=entries['with_shadow'],entries['without_shadow']
            if a.get('receiver_signature')!=b.get('receiver_signature'):raise ValueError('shadow_receiver_mismatch')
            shadow_pair={'status':'pass','camera_equal':True,'window':windows['shadow'],
                         'receiver_equal':True,'basis':'render_part_metadata_and_EXR_dimensions'}
        window=windows['object']['rect']
        shadow=windows['shadow']['rect']
        meta={k:obj[k] for k in ('camera','device','anchor_uv','contact_polygons')}
        meta.update(resolution=max(window[2]-window[0],window[3]-window[1]),
                    shadow_resolution=max(shadow[2]-shadow[0],shadow[3]-shadow[1]),
                    object_window=windows['object'],shadow_window=windows['shadow'],
                    legacy_grid=full_frame,part_paths={k:v['path'] for k,v in entries.items()},
                    render_cache=cache,render_timings=timings,asset_imports=imports,
                    render_mesh_detail=worker_detail or {'profile':request.get('render_detail','standard'),
                        'rust_spots':float(request.get('rust_spots',0.))})
        if shadow_pair:meta['shadow_pair_validation']=shadow_pair
        if request.get('surface_scans'):
            meta['scanned_materials']=obj.get('scanned_materials',[])
        if request.get('matte_partitions'):
            meta['matte_materials']=obj.get('matte_materials',[])
        if request.get('matte_profile'):
            meta['matte_profile']=request['matte_profile']
        if request.get('generated_finish'):
            meta['generated_finish'] = obj['generated_finish']
        if request.get('scanned_hdri') and 'with_shadow' in entries:
            meta.update({k:entries['with_shadow'].get(k) for k in ('shadow_basis','shadow_catcher_fallback')})
        return Path(obj['path']),meta,not missing
