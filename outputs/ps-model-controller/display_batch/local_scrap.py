"""Local material evidence and reusable, de-lit texture statistics."""
import asyncio
import json
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field, model_validator

from .ai import ProviderUnavailable
from .common import atomic_json, file_hash, signature
from .fusion import reflection_probe
from .photography import LUMA, display, linear
from .schema import Strict, Unit
from .surface_maps import prepare_uv_masks, save_image


VERSION = 'local-scrap-material-v1'
REFERENCE_PROMPT = """Classify the numbered tiles inside the shown surveillance neighbourhood.
Use steel for exposed weathered metal, paint for painted scrap, rust for oxide and dust for
visible deposits of fine grey or brown dust. Use other for concrete, shadows, holes, text,
mixed boundaries, or ambiguous material. Judge only the numbered tiles. Give every tile once.
Only report a direction toward a broad nearby light source when several comparable surfaces
support it. The direction is [x right, y down], components in [-1,1]; use null when uncertain.
Do not infer absolute roughness, HDR lighting, or particle density."""


class PatchMaterial(Strict):
    id: int = Field(ge=1,le=8)
    kind: Literal['steel','paint','rust','dust','other']
    confidence: Unit


class LocalReference(Strict):
    patches: list[PatchMaterial] = Field(max_length=8)
    light_direction: tuple[float,float] | None
    light_confidence: Unit
    evidence: str = Field(max_length=180)

    @model_validator(mode='after')
    def valid_regions(self):
        if len({p.id for p in self.patches})!=len(self.patches):
            raise ValueError('duplicate patch IDs')
        if self.light_direction and any(abs(v)>1 for v in self.light_direction):
            raise ValueError('invalid local light direction')
        return self


def reference_regions(scene,candidate,bounds):
    l,t,r,b = bounds
    rgb = np.asarray(scene['frame'])[t:b,l:r]
    depth = scene['depth_native'][t:b,l:r]
    site = next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    gray = np.float32(rgb)@LUMA
    geometry = (scene['allowed'][t:b,l:r] & ~scene['protected'][t:b,l:r] &
                (abs(depth-site['depth'])<=.065))
    allowed = geometry.copy()
    if not allowed.any():
        return rgb,allowed,[]
    low,high = np.percentile(gray[allowed],[7,98])
    allowed &= (gray>=max(24,low))&(gray<=min(231,high))&(rgb.max(axis=2)<246)
    inner = cv2.erode(geometry.astype(np.uint8),np.ones((5,5),np.uint8))>0
    h,w = gray.shape
    side = min(h,w,round(np.clip(candidate['target_px']*.15,36,104)))
    if side<24:
        return rgb,allowed,[]
    step = max(12,side//2)
    windows = []
    smooth = cv2.GaussianBlur(gray,(0,0),2.5)
    detail = gray-smooth
    for y in range(0,h-side+1,step):
        for x in range(0,w-side+1,step):
            area = np.s_[y:y+side,x:x+side]
            if not inner[area].all() or allowed[area].mean()<.82 or np.ptp(np.percentile(depth[area],[5,95]))>.06:
                continue
            p10,p90 = np.percentile(smooth[area],[10,90])
            # Coherent boundaries and strong shading are poor surface textures.
            if p90-p10>77:
                continue
            mad = np.median(abs(detail[area]-np.median(detail[area])))
            dist = math.hypot(x+side/2+l-candidate['anchor_px'][0],y+side/2+t-candidate['anchor_px'][1])
            score = min(mad,6)*.045-dist/max(h,w)*.52-(p90-p10)*.0015
            windows.append((score,x,y))
    chosen = []
    for _,x,y in sorted(windows,reverse=True):
        if any(abs(x-q[0])<side*.9 and abs(y-q[1])<side*.9 for q in chosen):
            continue
        chosen.append([x,y,x+side,y+side])
        if len(chosen)==8:
            break
    return rgb,allowed,chosen


def reference_board(rgb,windows,path):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for i,(x,y,r,b) in enumerate(windows,1):
        draw.rectangle((x,y,r-1,b-1),outline='#e3bc50',width=2)
        draw.rectangle((x,y,x+17,y+17),fill='#23282b')
        draw.text((x+4,y+2),str(i),fill='white')
    image.thumbnail((780,670),Image.Resampling.LANCZOS)
    width = max(image.width,4*116)
    board = Image.new('RGB',(width,image.height+(math.ceil(len(windows)/4)*116)),'#23282b')
    board.paste(image,(0,0))
    draw = ImageDraw.Draw(board)
    for i,(x,y,r,b) in enumerate(windows):
        tile = Image.fromarray(rgb[y:b,x:r]).resize((104,92),Image.Resampling.BILINEAR)
        left,top = i%4*116,image.height+i//4*116
        board.paste(tile,(left+4,top+19))
        draw.text((left+4,top+2),str(i+1),fill='white')
    save_image(board,path,format='JPEG',quality=92)


def patch_statistics(rgb,valid):
    observed = linear(np.float32(rgb)/255)
    gray = np.log(np.maximum(observed@LUMA,.018))
    gray[~valid] = np.median(gray[valid])
    small = cv2.GaussianBlur(gray,(0,0),1.35)
    medium = cv2.GaussianBlur(gray,(0,0),3.8)
    broad = cv2.GaussianBlur(gray,(0,0),9.)
    crop = np.s_[5:-5,5:-5]
    fine = (small-medium)[crop]
    coarse = (medium-broad)[crop]
    robust = lambda data:float(np.median(abs(data-np.median(data)))*1.4826)
    dx,dy = np.gradient(fine)
    tensor = np.array([[np.mean(dx*dx),np.mean(dx*dy)],[np.mean(dx*dy),np.mean(dy*dy)]])
    values, axes = np.linalg.eigh(tensor)
    direction = axes[:,0]
    gray_display = np.float32(rgb)@LUMA/255
    median,p92,p98 = np.percentile(gray_display[valid],[50,92,98])
    return {'color_linear':np.median(observed[valid],axis=0).tolist(),
            'fine':min(.09,robust(fine)), 'mottle':min(.15,robust(coarse)),
            'direction_rad':float(math.atan2(direction[0],direction[1])),
            'anisotropy':float(np.clip(math.sqrt((values[1]+1e-6)/(values[0]+1e-6)),1,2)),
            'highlight_excess':float(p98-median), 'highlight_spread':float(p92-median)}


def texture_field(key,stats,size=256):
    """Synthesize periodic band-limited fields from texture statistics, not photo pixels."""
    rng = np.random.default_rng(int(key[:16],16))
    fy,fx = np.meshgrid(np.fft.fftfreq(size),np.fft.rfftfreq(size),indexing='ij')
    angle = stats['direction_rad']
    u = fx*math.cos(angle)+fy*math.sin(angle)
    v = -fx*math.sin(angle)+fy*math.cos(angle)
    def band(sx,sy):
        kernel = np.exp(-2*np.pi**2*((u*sx)**2+(v*sy)**2))
        field = np.fft.irfft2(np.fft.rfft2(rng.normal(size=(size,size)))*kernel,s=(size,size))
        return np.clip((field-field.mean())/max(field.std(),1e-6),-2.5,2.5)
    mottle = band(8,6)
    scratches = band(4.5*stats['anisotropy'],1.65)
    deposition = band(11,8)
    return np.uint8(np.rint(np.clip(.5+.17*np.stack((mottle,scratches,deposition),axis=2),0,1)*255))


def summarize_reference(rgb,valid,windows,answer,environment):
    kinds = {k:[] for k in ('steel','paint','rust','dust')}
    used = []
    for mark in answer.patches:
        if mark.kind not in kinds or mark.confidence<.64 or mark.id>len(windows):
            continue
        x,y,r,b = windows[mark.id-1]
        stats = patch_statistics(rgb[y:b,x:r],valid[y:b,x:r])
        kinds[mark.kind].append(stats)
        used.append({**mark.model_dump(),'bounds':[x,y,r,b],'statistics':stats})
    source = kinds['steel'] or kinds['rust'] or kinds['paint']
    usable = source or kinds['dust']
    stats = {k:float(np.median([s[k] for s in usable])) if usable else default
             for k,default in {'fine':.006,'mottle':.012,'direction_rad':0.,'anisotropy':1.,
                               'highlight_excess':0.,'highlight_spread':0.}.items()}
    def color(kind,luminance):
        entries = kinds[kind] or usable
        base = np.median([e['color_linear'] for e in entries],axis=0) if entries else environment['dust_color']
        cast = np.clip(np.asarray(base)/max(.02,float(np.asarray(base)@LUMA)),.77,1.27)
        return (luminance*cast).tolist()
    # These are appearance presets constrained by photo statistics, not recovered BRDFs.
    peak = stats['highlight_excess']
    spread = stats['highlight_spread']
    roughness = float(np.clip(.61-.56*peak+.13*spread,.43,.65))
    dust_samples = kinds['dust']
    dust_luma = float(np.clip(np.median([np.asarray(s['color_linear'])@LUMA for s in dust_samples]),.11,.28)) if dust_samples else .17
    palette = {'metal':color('steel',.43), 'paint':color('paint',.235),
               'rust':color('rust',.16), 'dust':color('dust',dust_luma)}
    return {'status':'estimated' if source else 'unknown',
            'basis':'nearby_verified_surface_tiles' if source else 'conservative_source_material_local_cast',
            'statistics':stats,'palette':palette,'metal_roughness':roughness,'regions':used,
            'coating_strength':.64 if dust_samples else .49,
            'detail_weight':float(np.clip((stats['mottle']+stats['fine'])*2.4,.028,.24)) if usable else .028}


def local_light(scene,candidate,environment,valid,profile,answer):
    air = environment['airborne_dust']
    amount = min(.16,air['amount'])
    # Estimate incident light with a neutral reference, independently of rendered object colour.
    median = np.asarray(environment['dust_color'],np.float32)
    clear = np.clip((median-amount*np.asarray(air['color_linear']))/(1-amount),median*.65,median*1.25)
    target = float(np.clip(np.mean(np.maximum(clear,0)**(1/2.2)),.20,.71))
    values = np.geomspace(.018,.9,64,dtype=np.float32)
    response = display(np.repeat(values[:,None],3,axis=1)).mean(axis=1)
    illumination = float(np.clip(np.interp(target,response,values)/(.28*.65),.32,2.5))
    cast = np.clip(median/max(float(median@LUMA),.02),.92,1.08)
    direction = answer.light_direction if answer.light_confidence>=.68 and profile['status']=='estimated' else None
    env = {**environment,'world_color':(illumination*cast).tolist(),'light_color':(cast/cast.max()).tolist(),
           'light_level':illumination,'key_direction':direction,'light_confidence':answer.light_confidence if direction else 0.,
           'calibration':{'target_luma':target,'illumination':illumination,
                          'basis':'local_neutral_reference_estimate','limited':illumination in (.32,2.5)},
           'lighting_basis':'local_broad_reflection_estimate'}
    bounds = environment['local_context']['bounds']
    l,t,r,b = bounds
    mask = np.zeros(scene['allowed'].shape,bool)
    mask[t:b,l:r] = valid
    probe,coverage = reflection_probe(np.asarray(scene['frame']),mask,candidate,env,bounds=bounds,
                                      airlight=(amount,air['color_linear']))
    env.update(reflection_probe=probe,reflection_coverage=coverage)
    return env


class ScrapMaterials:
    def __init__(self,cache,vision):
        self.root = Path(cache)/VERSION
        self.vision = vision
        self.locks = {}

    async def prepare(self,asset,scene,candidate,environment):
        bounds = environment['local_context']['bounds']
        rgb,valid,windows = await asyncio.to_thread(reference_regions,scene,candidate,bounds)
        key = signature(VERSION,scene['key'],bounds,windows,environment['sample_hash'],
                        self.vision.endpoint.model_dump())
        folder = self.root/'references'/key
        async with self.locks.setdefault(key,asyncio.Lock()):
            profile,answer,hit = await self.prepare_reference(folder,key,rgb,valid,windows,environment)
        async with self.locks.setdefault(asset['mesh_hash'],asyncio.Lock()):
            masks,mask_hit = await asyncio.to_thread(prepare_uv_masks,asset,self.root/'uv')
        env = await asyncio.to_thread(local_light,scene,candidate,environment,valid,profile,answer)
        variant = signature(VERSION,masks['signature'],profile['signature'])
        env['scrap_surface'] = {**profile,'materials':masks['materials'],'uv_manifest':masks['path'],
                                'variant_signature':variant,'reference_cache_hit':hit,'uv_cache_hit':mask_hit}
        return env

    async def prepare_reference(self,folder,key,rgb,valid,windows,environment):
        manifest = folder/'reference.json'
        if manifest.is_file():
            profile = json.loads(manifest.read_text(encoding='utf8'))
            if Path(profile['texture']).is_file() and Path(profile['board']).is_file():
                return profile,LocalReference.model_validate(profile['answer']),True
        folder.mkdir(parents=True,exist_ok=True)
        board = folder/'nearby-scrap.jpg'
        await asyncio.to_thread(reference_board,rgb,windows,board)
        answer = LocalReference(patches=[],light_direction=None,light_confidence=0.,evidence='局部有效样本不足')
        if windows:
            try:
                answer = await self.vision.ask('local-scrap-reference',LocalReference,REFERENCE_PROMPT,
                    {'tile_ids':list(range(1,len(windows)+1))},[board],image_limits=[(900,900)],
                    max_output_tokens=1024,plain_json=True)
            except ProviderUnavailable:
                raise
            except RuntimeError:
                answer.evidence = '材质识别未返回有效结果'
        profile = summarize_reference(rgb,valid,windows,answer,environment)
        tile = folder/'surface-pattern.png'
        save_image(Image.fromarray(texture_field(key,profile['statistics'])),tile)
        profile.update(answer=answer.model_dump(),board=str(board),texture=str(tile),
                       candidates=windows,reference_bounds=environment['local_context']['bounds'],
                       signature=signature(key,profile,file_hash(tile)),manifest=str(manifest))
        atomic_json(manifest,profile)
        return profile,answer,False
