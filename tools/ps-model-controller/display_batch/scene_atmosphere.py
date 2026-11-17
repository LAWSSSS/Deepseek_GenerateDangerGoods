"""Object-independent camera veil atlas for the opt-in realism workflow.

The inputs are the unedited photograph and its scene evidence. No object colour,
bounding box, identifier or support selection is an input to the fit. Single-image
airlight and relative depth are estimates, not measured radiometry or metres.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import cv2
import numpy as np
from PIL import Image

from .atmosphere import field_crop, regions_mask
from .common import atomic_json, signature, file_hash
from .photography import linear, LUMA

VERSION = 'scene-veil-atlas-v1'
LONG_EDGE = 512
CELL = 32
RADII = (24, 48, 72)
DEPTH_TOLERANCE = .085


def _digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, str(value.dtype))).encode() + value.tobytes()).hexdigest()


def _input(scene):
    rgb = np.asarray(scene['frame'].convert('RGB') if hasattr(scene['frame'], 'convert') else scene['frame'])
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('atmosphere_requires_rgb_background')
    h, w = rgb.shape[:2]
    depth = np.asarray(scene['depth_native'], np.float32)
    allowed = np.asarray(scene['allowed'], bool)
    protected = np.asarray(scene['protected'], bool)
    if any(a.shape != (h, w) for a in (depth, allowed, protected)):
        raise ValueError('atmosphere_scene_shapes_mismatch')
    hints = scene.get('atmosphere', {}).get('summary', {}).get('visual_evidence', {})
    valid = allowed & ~protected & np.isfinite(depth)
    valid &= ~regions_mask(hints.get('exclude_regions', []), (h, w))
    return rgb, depth, valid, hints


def _downsample(rgb, depth, valid):
    h, w = valid.shape
    scale = min(1., LONG_EDGE/max(h, w))
    size = (max(8, round(w*scale)), max(8, round(h*scale)))
    weight = cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    # Excluded text/foreground must not bleed into nearby low-resolution evidence.
    small = cv2.resize(np.float32(rgb)/255*valid[..., None], size, interpolation=cv2.INTER_AREA)
    small /= np.maximum(weight[..., None], 1e-6)
    z = cv2.resize(np.where(valid, depth, 0), size, interpolation=cv2.INTER_AREA)/np.maximum(weight, 1e-6)
    return small, z, weight > .98


def _field(grid, mask, size, fallback):
    """Smooth only local supported cells; unsupported locations retain uncertainty."""
    coverage = cv2.resize(np.float32(mask), size, interpolation=cv2.INTER_LINEAR)
    extra = (None,) if grid.ndim == 3 else ()
    values = cv2.resize(grid*mask[(...,)+extra], size, interpolation=cv2.INTER_LINEAR)
    return np.where((coverage > .2)[(...,)+extra],
                    values/np.maximum(coverage, 1e-6)[(...,)+extra], fallback).astype(np.float32)


def estimate(scene):
    rgb, depth, allowed, hints = _input(scene)
    full_h, full_w = allowed.shape
    small, inverse, valid = _downsample(rgb, depth, allowed)
    h, w = valid.shape
    radiance = linear(small)
    gray = small @ LUMA
    spread = (small.max(axis=2)-small.min(axis=2))/np.maximum(gray, .05)
    # An airlight tint estimate cannot be set by blue steel, white labels or clipped glints.
    valid &= (small.max(axis=2) < .92) & (gray > .035)
    neutral = valid & (spread < .22)
    kind, hint_confidence = hints.get('kind', 'uncertain'), float(hints.get('confidence', 0))
    haze = regions_mask(hints.get('haze_regions', []), (h, w))
    gh, gw = max(2, int(np.ceil(h/CELL))), max(2, int(np.ceil(w/CELL)))
    amounts = np.zeros((gh, gw), np.float32)
    lo_grid, hi_grid = amounts.copy(), amounts.copy()
    air_grid = np.full((gh, gw, 3), .5, np.float32)
    evidence_grid = np.zeros((gh, gw), np.float32)
    supported = np.zeros((gh, gw), bool)
    tile_records = []
    for gy in range(gh):
        for gx in range(gw):
            cy, cx = min(h-1, int((gy+.5)*h/gh)), min(w-1, int((gx+.5)*w/gw))
            center = inverse[max(0,cy-8):min(h,cy+9), max(0,cx-8):min(w,cx+9)]
            center_valid = valid[max(0,cy-8):min(h,cy+9), max(0,cx-8):min(w,cx+9)]
            if center_valid.sum() < 12:
                continue
            z = float(np.median(center[center_valid]))
            fits = []
            for radius in RADII:
                y, x, b, r = max(0,cy-radius), max(0,cx-radius), min(h,cy+radius), min(w,cx+radius)
                mask = neutral[y:b,x:r] & (abs(inverse[y:b,x:r]-z) <= DEPTH_TOLERANCE)
                if mask.sum() < 60:
                    continue
                values = radiance[y:b,x:r][mask]
                lum = values @ LUMA
                q05, q50, q80, q90 = np.percentile(lum, [5, 50, 80, 90])
                contrast = float((q80-q05)/max(q80, .03))
                # A smooth light plate alone does not establish lifted dark detail.
                if q80-q05 < .012 or contrast < .08:
                    continue
                bright = values[(lum >= q50) & (lum <= q90)]
                color = np.mean(bright, axis=0)
                tint = np.clip(color/max(float(color@LUMA), .01), .96, 1.04)
                # Unseen airlight is bounded rather than identified with a reflector.
                air_level = float(np.clip(q90+.12, .30, .78))
                air = air_level*tint
                minimum_air, maximum_air = max(.22,air_level*.8), min(.90,air_level*1.2)
                low = float(np.clip((q05-.030)/max(maximum_air-.030,.05), 0,.5))
                high = float(np.clip((q05-.008)/max(minimum_air-.008,.05), 0,.5))
                alpha = float(np.clip((q05-.015)/max(air_level-.015,.05), 0,.5))
                overlap = float(haze[y:b,x:r][mask].mean())
                visual = kind in ('haze','veil','mixed') and hint_confidence >= .5 and overlap >= .4
                confidence = min(.75, max(.1, contrast)*1.3) if visual else min(.35, contrast*.5)
                if not visual:
                    alpha = min(low, .18)
                if kind == 'clear' and hint_confidence >= .65:
                    alpha = low = high = 0.
                    confidence = min(.8,hint_confidence)
                fits.append((alpha, low, high, air, confidence, int(mask.sum()), float(q05)))
            if len(fits) < 2:
                continue
            disagreement = float(np.ptp([f[0] for f in fits]))
            a = float(np.median([f[0] for f in fits]))
            conf = float(np.median([f[4] for f in fits])*max(0.,1-disagreement/.15))
            if conf < .2:
                a = min(a, min(f[1] for f in fits), .12)
            supported[gy,gx] = True
            amounts[gy,gx] = a
            lo_grid[gy,gx] = min(a, min(f[1] for f in fits))
            hi_grid[gy,gx] = max(a, max(f[2] for f in fits))
            air_grid[gy,gx] = np.median([f[3] for f in fits],axis=0)
            evidence_grid[gy,gx] = conf
            tile_records.append({'center_normalized':[(cx+.5)/w,(cy+.5)/h], 'relative_depth':z,
                'amount':a, 'amount_bounds':[float(lo_grid[gy,gx]),float(hi_grid[gy,gx])],
                'confidence':conf, 'scale_disagreement':disagreement,
                'samples':min(f[5] for f in fits), 'scales':len(fits)})
    amount = _field(amounts,supported,(w,h),0.)
    low = _field(lo_grid,supported,(w,h),0.)
    high = _field(hi_grid,supported,(w,h),.18)
    air = _field(air_grid,supported,(w,h),.5)
    conf = _field(evidence_grid,supported,(w,h),0.)
    if kind == 'clear' and hint_confidence >= .65:
        amount[:] = low[:] = high[:] = 0.
    tau = -np.log(np.maximum(1-amount,.5))
    sample = neutral & (conf > .15)
    depth_values = inverse[valid] if valid.any() else np.float32([0.,1.])
    near, far = np.percentile(depth_values,[95,5])
    distance = np.clip((near-inverse)/max(float(near-far),.01),0,1)
    beta, correlation = 0., 0.
    # Independent tile evidence, not thousands of correlated pixels, supports depth.
    reliable = [t for t in tile_records if t['confidence'] >= .4]
    if len(reliable) >= 12 and near-far > .15:
        dz = np.float32([(near-t['relative_depth'])/(near-far) for t in reliable])
        dt = -np.log(np.maximum(1-np.float32([t['amount'] for t in reliable]),.5))
        if dz.std() > .12 and dt.std() > .025:
            correlation = float(np.corrcoef(dz,dt)[0,1])
            if correlation >= .55:
                beta = float(np.clip(np.polyfit(dz,dt,1)[0],0,.25))
    screen = np.maximum(tau-beta*distance,0).astype(np.float32)
    # Preserve the fitted spatial field; relative-depth decomposition is only used
    # when the object has an independently supported local depth mapping.
    fields={'transmission':(1-amount).astype(np.float32),'screen_tau':screen,
            'airlight':air,'inverse_depth':inverse,'sample_mask':sample.astype(np.uint8),
            'confidence_field':conf,'amount_lower':low,'amount_upper':high}
    summary={'version':VERSION,'bounds':[0,0,full_w,full_h],'field_size':[w,h],
        'confidence':'estimated' if len(reliable)>=3 else 'unknown',
        'basis':'background_only_multiscale_same_layer_dark_tail_atlas',
        'sample_pixels':int(sample.sum()),'supported_tiles':int(supported.sum()),
        'depth_coefficient':beta,'depth_near':float(near),'depth_far':float(far),
        'depth_correlation':correlation,'depth_confidence':'estimated' if beta else 'unknown',
        'airlight_camera_linear':np.median(air.reshape(-1,3),axis=0).tolist(),
        'transmission_range':np.percentile(1-amount,[5,50,95]).tolist(),
        'amount_bounds':[float(low.min()),float(high.max())],
        'uncertainty_basis':'plausible_dark_reflectance_0.008_to_0.030_and_airlight_20pct_not_statistical_CI',
        'material_and_candidate_independent':True,'metric_air_density_known':False,
        'camera_response':'decoded_sRGB_approximation','depth_tolerance':DEPTH_TOLERANCE,
        'visual_evidence':hints,'tiles':tile_records}
    return fields,summary


def prepare(scene, cache):
    rgb, depth, valid, hints = _input(scene)
    key=signature(VERSION, file_hash(__file__), _digest(rgb), _digest(depth), _digest(valid),
                  hints, LONG_EDGE,CELL,RADII,DEPTH_TOLERANCE)
    folder=Path(cache)/'scene-atmosphere'/key
    summary_path, field_path = folder/'summary.json', folder/'fields.npz'
    hit=summary_path.is_file() and field_path.is_file()
    if hit:
        summary=json.loads(summary_path.read_text(encoding='utf8'))
    else:
        fields,summary=estimate(scene)
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder,suffix='.npz',delete=False) as stream:
            temporary=stream.name
            np.savez_compressed(stream,**fields)
        os.replace(temporary,field_path)
        atomic_json(summary_path,summary)
        color=cv2.applyColorMap(np.uint8((1-fields['transmission'])*255),cv2.COLORMAP_TURBO)[:,:,::-1]
        Image.fromarray(color).save(folder/'veil.png')
        Image.fromarray(fields['sample_mask']*255).save(folder/'samples.png')
    return {'folder':str(folder),'signature':key,'summary':summary,'cache_hit':hit}


def fields_for(context,bounds,object_z,order,candidate,scene,gain=1.,enabled=True):
    if not np.isfinite(gain) or not 0 <= gain <= 2:
        raise ValueError('invalid_atmosphere_gain')
    summary=context['summary']; _,_,w,h=summary['bounds']
    with np.load(Path(context['folder'])/'fields.npz',allow_pickle=False) as data:
        fields={name:field_crop(data[name],bounds,(w,h)) for name in
                ('transmission','airlight','inverse_depth','screen_tau','amount_lower','amount_upper')}
    background_t=fields['transmission']
    tau=-np.log(np.maximum(background_t,.5))
    depth_basis='same_pixel_background_field_no_object_dependent_refit'
    beta=float(summary['depth_coefficient'])
    if beta > 0 and order.get('confidence')=='estimated' and all(k in order for k in ('scale','offset')):
        inverse=np.where(object_z>0,order['scale']/np.maximum(object_z,1e-6)+order['offset'],fields['inverse_depth'])
        old_distance=np.clip((summary['depth_near']-fields['inverse_depth'])/max(.01,summary['depth_near']-summary['depth_far']),0,1)
        new_distance=np.clip((summary['depth_near']-inverse)/max(.01,summary['depth_near']-summary['depth_far']),0,1)
        tau=np.maximum(tau+beta*(new_distance-old_distance),0)
        depth_basis='scene_tile_depth_trend_and_verified_local_relative_depth_fit'
    object_t=np.exp(-np.clip(tau*gain,0,-np.log(.5))).astype(np.float32)
    if not enabled:
        object_t[:]=1.;background_t=np.ones_like(background_t)
    # Keep heavy tile diagnostic records in the cached summary, not every review prompt.
    info={k:v for k,v in summary.items() if k!='tiles'}
    info.update(signature=context['signature'],depth_basis=depth_basis,
        local_amount_bounds=[float(fields['amount_lower'].min()),float(fields['amount_upper'].max())])
    return object_t,background_t,fields['airlight'],info
