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

from .common import HERE, atomic_json, file_hash, signature
from .framing import render_windows
from .schema import LIGHT_PROFILES, MATERIAL_PROFILES, PHOTO_CONTROLS


REFERENCE_RENDER_VERSION = 'pbr-linear-patina-v9'
RENDER_VERSION = 'render-parts-v1'
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
    if candidate.get('camera'):request['camera']=candidate['camera']
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
           'display_batch/framing.py','display_batch/materials.py','display_batch/rendering.py']
    return signature('blender-4.5.3',{name:file_hash(HERE/name) for name in files})


def part_keys(request, device, dependencies):
    windows=request['windows']
    fields=('world_color','light_color','key_direction','light_level','light_spread','reflection_probe')
    environment=request['environment']
    light={k:environment[k] for k in fields if k in environment}
    shared=(RENDER_VERSION,dependencies,device,request['appearance'],request['camera'],
            request['plane_normal'],request.get('receiver',{}).get('patches',[]),light,request['light_gain'])
    surface=(request['mesh_hash'],request['matrix'],request['yaw'],request['roughness_delta'],
             request['dust_gain'],request['dust_control'],request['material_classes'],
             environment.get('dust_color'),environment.get('haze'),
             environment.get('sensor',{}).get('surface_detail_rms') if request['appearance']=='native' else None)
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
    def ready(folder, part):
        marker=folder/'result.json'
        if not marker.is_file():return None,'no_entry'
        try:
            info=json.loads(marker.read_text(encoding='utf8'))
            if all((Path(info['path'])/name).stat().st_size>0 for name in PART_FILES[part]):
                return info,None
        except (OSError,ValueError,KeyError,TypeError):
            pass
        return None,'incomplete_entry'

    async def get(self, asset, candidate, params, environment, size, resolution, *,
                  appearance='scene', full_frame=False):
        if not candidate.get('camera'):
            raise ValueError('cropped_render_requires_fixed_camera')
        request=render_request(asset,candidate,params,environment)
        windows=render_windows(candidate,size,resolution,full_frame=full_frame)
        request.update(mesh_hash=asset['mesh_hash'],appearance=appearance,windows=windows,
                       material_classes=asset['semantic']['material_classes'],
                       dust_control=PHOTO_CONTROLS['dust'][params.get('dust','auto')] if appearance in ('scene','native') else 1.)
        keys=part_keys(request,self.worker.device,render_dependencies())
        queued=time.monotonic()
        async with self.lock:
            timings={'render_cache_wait_seconds':time.monotonic()-queued}
            return await self._get(request,keys,timings,full_frame)

    async def _get(self,request,keys,timings,full_frame):
        windows=request['windows']
        entries={};cache={};missing={};folders={};imports=0
        for part,key in keys.items():
            folder=self.root/part/key
            folders[part]=folder
            ready,reason=self.ready(folder,part)
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
            timings.update(response['timings'])
            imports=response['imports']
            self.totals['asset_imports']+=imports
            for part,path in missing.items():
                window=windows['object' if part=='object' else 'shadow']['rect']
                if not all((Path(path)/name).is_file() for name in PART_FILES[part]):
                    raise ValueError('render_part_missing_files: '+part)
                if part=='object':
                    with Image.open(Path(path)/'object.png') as image:
                        if image.size!=(window[2]-window[0],window[3]-window[1]):
                            raise ValueError('unexpected_render_window_size')
                info={'path':path,'camera':response['camera'],'device':response['device'],
                      'anchor_uv':response['anchor_uv'],'contact_polygons':response['contact_polygons']}
                entries[part]=info
                atomic_json(folders[part]/'result.json',info)
                self.totals[part+'_renders']+=1
        self.timings.update(timings)
        obj=entries['object']
        window=windows['object']['rect']
        shadow=windows['shadow']['rect']
        meta={k:obj[k] for k in ('camera','device','anchor_uv','contact_polygons')}
        meta.update(resolution=max(window[2]-window[0],window[3]-window[1]),
                    shadow_resolution=max(shadow[2]-shadow[0],shadow[3]-shadow[1]),
                    object_window=windows['object'],shadow_window=windows['shadow'],
                    legacy_grid=full_frame,part_paths={k:v['path'] for k,v in entries.items()},
                    render_cache=cache,render_timings=timings,asset_imports=imports)
        return Path(obj['path']),meta,not missing
