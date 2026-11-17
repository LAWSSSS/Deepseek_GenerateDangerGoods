"""Cached, image-derived airlight and transmission in camera-linear RGB."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from .common import atomic_json, signature
from .photography import linear, display, LUMA
from .schema import AtmosphereAnswer
from .ai import ProviderUnavailable


VERSION = 'local-atmosphere-v3'
PROMPT = """Inspect this original surveillance photograph for atmospheric imaging only.
Distinguish clear air, depth-related haze/dust, screen-position veiling glare, mixed, or uncertain.
Use actual dark objects, repeated steel surfaces at different depths, and light/shadow evidence.
Do not mistake white paint, smooth metal reflections, rust colour, or overexposure for fog.
Mark a few broad haze_regions, illuminating light_regions, and exclude_regions (white surfaces,
specular highlights, text or saturation that would mislead haze estimation). Every region MUST be
a rectangle [left,top,right,bottom] in 0..1. Example haze_regions: [[0.2,0.1,0.8,0.9]].
Use [] if no region is supported. Do not return polygon vertices or isolated points.
light_direction points towards illumination in screen x-right/y-down coordinates, or null.
Report confidence and concise evidence. Do not invent metric distances or numerical transmission."""


def regions_mask(items, shape):
    mask = np.zeros(shape, np.uint8)
    h, w = shape
    for left,top,right,bottom in items:
        p=np.rint(np.asarray([left,top,right,bottom])*[w-1,h-1,w-1,h-1]).astype(int)
        cv2.rectangle(mask,tuple(p[:2]),tuple(p[2:]),1,-1)
    return mask.astype(bool)


def masked_smooth(value, valid, sigma):
    weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    extra = (None,) if value.ndim == 3 else ()
    denominator = np.maximum(weight, 1e-6)[(...,)+extra]
    return cv2.GaussianBlur(value*valid[(...,)+extra], (0, 0), sigma)/denominator, weight


def estimate(frame, depth, allowed, answer):
    """Dark-channel proposals need spatial and depth evidence; coefficients are estimates."""
    height, width = np.asarray(frame).shape[:2]
    factor = min(1., 512/max(width, height))
    size = (max(8, round(width*factor)), max(8, round(height*factor)))
    rgb = cv2.resize(np.float32(frame)/255, size, interpolation=cv2.INTER_AREA)
    radiance = linear(rgb)
    inverse_depth = cv2.resize(np.float32(depth), size)
    valid = cv2.resize(allowed.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)>0
    valid &= ~regions_mask(answer['exclude_regions'], valid.shape)
    luma = rgb@LUMA
    dark = radiance.min(axis=2)
    gradient = cv2.magnitude(cv2.Sobel(luma, cv2.CV_32F, 1, 0), cv2.Sobel(luma, cv2.CV_32F, 0, 1))
    # Smooth bright surfaces lack the dark, textured evidence required by the prior.
    valid &= (rgb.max(axis=2)<.94) & ~((luma>.65)&(gradient<.025))
    evidence = valid & (gradient>.015)
    # Invalid text/foreground pixels must not lower the neighbouring dark-channel minima.
    candidate_dark=np.where(valid,dark,1.).astype(np.float32)
    dark_small = cv2.erode(candidate_dark, np.ones((7, 7), np.uint8))
    dark_large = cv2.erode(candidate_dark, np.ones((21, 21), np.uint8))
    sufficient = int(evidence.sum())>=300
    sample = radiance[valid] if valid.any() else radiance.reshape(-1, 3)
    air = np.clip(np.percentile(sample, 94, axis=0), .16, .88)
    bright = valid & (luma>=np.percentile(luma[valid], 75) if valid.any() else False)
    field, weight = masked_smooth(radiance, bright, 25.)
    field = np.where((weight>.015)[...,None], field, air)
    airlight = np.float32(np.clip(field, air*.90, air*1.12))
    proposal = np.clip(1-.85*(dark_small+dark_large)/2/np.maximum(airlight.min(axis=2), .08), .48, 1.)
    tau = -np.log(proposal)
    fog_mask = regions_mask(answer['haze_regions'], valid.shape)
    if fog_mask.any():
        region = cv2.GaussianBlur(fog_mask.astype(np.float32), (0, 0), 14.)
        tau *= .35+.65*region
    depth_values = inverse_depth[valid] if valid.any() else inverse_depth.ravel()
    near, far = np.percentile(depth_values, [95, 5])
    distance = np.clip((near-inverse_depth)/max(near-far, .01), 0, 1)
    agreement = float(np.median(abs(dark_small[evidence]-dark_large[evidence]))) if sufficient else 1.
    spread = float(np.std(distance[evidence])) if sufficient else 0.
    correlation = 0.
    beta = 0.
    if sufficient and spread>.12 and np.std(tau[evidence])>.01:
        correlation = float(np.corrcoef(distance[evidence], tau[evidence])[0, 1])
        if correlation>.25 and agreement<.08 and answer['kind'] in ('haze', 'mixed'):
            beta = float(np.clip(np.polyfit(distance[evidence], tau[evidence], 1)[0], 0, .6))
    reliable = sufficient and agreement<.08 and answer['confidence']>=.55
    if answer['kind']=='clear' and answer['confidence']>=.55:
        tau[:] = 0
        beta = 0.
        reliable = sufficient
    elif not reliable:
        tau = np.minimum(tau, .16)
        beta = min(beta, .10)
    residual, coverage = masked_smooth(np.maximum(tau-beta*distance, 0), evidence, 19.)
    fallback = float(np.median(tau[evidence])) if evidence.any() else 0.
    screen = np.float32(np.where(coverage>.025, residual, fallback))
    transmission = np.exp(-np.clip(screen+beta*distance, 0, .75)).astype(np.float32)
    summary = {'version':VERSION, 'kind':answer['kind'], 'confidence':'estimated' if reliable else 'unknown',
        'basis':'multiscale_dark_channel_with_masked_local_statistics', 'visual_evidence':answer,
        'sample_pixels':int(evidence.sum()), 'depth_correlation':correlation, 'depth_coefficient':beta,
        'depth_confidence':'estimated' if beta>0 else 'not_applicable' if answer['kind']=='clear' else 'unknown',
        'sampling_confidence':float(answer['confidence'] if reliable and beta>0 else min(answer['confidence'],.5)),
        'depth_near':float(near), 'depth_far':float(far), 'scale_disagreement':agreement,
        'transmission_range':np.percentile(transmission, [5, 50, 95]).tolist(),
        'airlight_camera_linear':np.median(airlight.reshape(-1, 3), axis=0).tolist(),
        'camera_response':'decoded_sRGB_approximation', 'metric_density_known':False}
    return {'screen_tau':screen, 'airlight':airlight, 'transmission':transmission,
            'inverse_depth':inverse_depth, 'sample_mask':evidence.astype(np.uint8)}, summary


class Atmospheres:
    def __init__(self, cache, vision):
        self.root = Path(cache)/'atmospheres'
        self.vision = vision
        self.loaded = {}
        self.locks = {}

    async def prepare(self, scene):
        key = signature(VERSION, scene['key'], self.vision.endpoint.model_dump())
        async with self.locks.setdefault(key, asyncio.Lock()):
            if key in self.loaded:
                return self.loaded[key]
            started = time.monotonic()
            folder = self.root/key
            record, arrays = folder/'estimate.json', folder/'fields.npz'
            if record.is_file() and arrays.is_file():
                summary = json.loads(record.read_text(encoding='utf8'))
                with np.load(arrays) as source:
                    fields = {name:source[name] for name in source.files}
            else:
                image = Path(scene['folder'])/'background.jpg'
                try:
                    answer = await self.vision.ask('scene-photometry', AtmosphereAnswer, PROMPT, {}, [image], plain_json=True)
                except ProviderUnavailable:
                    raise
                except RuntimeError:
                    answer = AtmosphereAnswer(kind='uncertain',confidence=0,haze_regions=[],light_regions=[],
                        exclude_regions=[],light_direction=None,evidence='视觉区域格式重试失败，使用低置信度图像估计')
                fields, summary = await asyncio.to_thread(estimate, np.asarray(scene['frame']), scene['depth_native'],
                    scene['allowed'] & ~scene['protected'], answer.model_dump())
                folder.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=folder, suffix='.npz', delete=False) as output:
                    temporary = output.name
                    np.savez_compressed(output, **fields)
                os.replace(temporary, arrays)
                atomic_json(record, summary)
                diagnostic = cv2.applyColorMap(np.uint8((1-fields['transmission'])*255), cv2.COLORMAP_TURBO)
                Image.fromarray(diagnostic[:,:,::-1]).save(folder/'veil.png')
                Image.fromarray(fields['sample_mask']*255).save(folder/'samples.png')
            result = {**fields, 'summary':summary, 'signature':key, 'folder':str(folder),
                      'seconds':time.monotonic()-started}
            self.loaded[key] = result
            return result


def field_crop(field, bounds, size):
    x, y, r, b = bounds
    width, height = size
    yy, xx = np.mgrid[y:b, x:r].astype(np.float32)
    mx = (xx+.5)*field.shape[1]/width-.5
    my = (yy+.5)*field.shape[0]/height-.5
    return cv2.remap(field, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def transmission_fields(scene, bounds, object_depth, order, haze_gain=1., enabled=True, *, candidate=None):
    estimate = scene['atmosphere']
    summary = estimate['summary']
    screen = field_crop(estimate['screen_tau'], bounds, scene['size'])
    background_t = field_crop(estimate['transmission'], bounds, scene['size'])
    airlight = field_crop(estimate['airlight'], bounds, scene['size'])
    inverse = field_crop(estimate['inverse_depth'], bounds, scene['size'])
    depth_basis = 'local_screen_veil_and_same_layer_fallback'
    if order.get('confidence')=='estimated' and summary['depth_coefficient']>0:
        inverse = np.where(object_depth>0, order['scale']/np.maximum(object_depth, 1e-6)+order['offset'], inverse)
        depth_basis = 'local_inverse_depth_fit'
    elif candidate is not None:
        site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
        inverse=np.full_like(inverse,site['depth'])
    distance = np.clip((summary['depth_near']-inverse)/max(.01, summary['depth_near']-summary['depth_far']), 0, 1)
    object_t = np.exp(-np.clip((screen+summary['depth_coefficient']*distance)*haze_gain, 0, .9))
    if not enabled:
        object_t = np.ones_like(object_t)
        background_t = np.ones_like(background_t)
    return np.float32(object_t), background_t, airlight, depth_basis


def fogged_radiance(clear, transmission, airlight):
    return clear*transmission[...,None]+airlight*(1-transmission[...,None])


def shadowed_background(background, loss, transmission, airlight):
    surface = np.maximum(background-airlight*(1-transmission[...,None]), 0)
    return np.maximum(background-loss[...,None]*surface, 0)


def calibrate_environment(scene, candidate, environment, samples):
    """One local neutral-reflectance assumption determines lighting for every material."""
    estimate = scene['atmosphere']
    small_sample = cv2.resize(samples.astype(np.uint8), estimate['transmission'].shape[::-1],
                              interpolation=cv2.INTER_NEAREST)>0
    rgb = cv2.resize(np.float32(scene['frame'])/255, small_sample.shape[::-1], interpolation=cv2.INTER_AREA)
    observed = linear(rgb)
    t = estimate['transmission'][...,None]
    clear = np.clip((observed-estimate['airlight']*(1-t))/np.maximum(t, .48), observed*.45, observed*1.6)
    clear_rgb = np.clip(np.where(clear<=.0031308, clear*12.92,
        1.055*np.maximum(clear,0)**(1/2.4)-.055),0,1)
    selected = clear[small_sample]
    if len(selected)<20:
        selected = clear[estimate['sample_mask']>0]
    if not len(selected):
        selected = clear.reshape(-1, 3)
    median = np.median(selected, axis=0)
    # AgX is used only to map virtual neutral radiance to the observed working RGB.
    target = float(np.clip(np.median(clear_rgb[small_sample]@LUMA) if small_sample.any() else .45, .12, .78))
    low, high = .005, 2.
    for _ in range(16):
        middle = (low+high)/2
        if float(display(np.full((1, 3), middle, np.float32)).mean())<target:
            low = middle
        else:
            high = middle
    illumination = float(np.clip((low+high)/2/(.18*.65), .18, 4.))
    tint = np.clip(median/max(float(median.mean()), .015), .88, 1.12)
    environment.update(world_color=(illumination*tint).tolist(), light_level=illumination,
        calibration={'exposure_ev':0., 'white_balance':[1., 1., 1.],
        'basis':'same_layer_neutral_reflectance_0.18_estimate', 'target_luma':target,
        'illumination':illumination, 'limited':illumination in (.18, 4.),
        'material_independent':True,'sample_basis':environment.get('sample_basis','same_support_estimate')})
    hints = estimate['summary']['visual_evidence']
    if hints['light_direction'] and hints['confidence']>=.55:
        environment['key_direction'] = np.clip(hints['light_direction'],-1,1).tolist()
        environment['light_confidence'] = hints['confidence']
    # A compensated low-frequency photograph feeds reflection only, never object texture.
    environment['atmosphere_signature'] = estimate['signature']
    return np.uint8(np.rint(clear_rgb*255))
