"""Bounded neighbourhood evidence and camera matching; never sample the whole frame."""
import hashlib
import math

import cv2
import numpy as np

from .photography import LUMA, linear, sensor_statistics
from .fusion import camera_psf, topology

VERSION = 'neighbourhood-fusion-v4-dust'


def neighbourhood(scene, candidate):
    w,h=scene['size'];x,y,r,b=map(int,candidate['bounds'])
    pad=round(np.clip(candidate['target_px']*.30,32*w/3840,200*w/3840))
    pad=max(12,pad)
    bounds=[max(0,x-pad),max(0,y-pad),min(w,r+pad),min(h,b+pad)]
    l,t,rr,bb=bounds
    rgb=np.asarray(scene['frame'])[t:bb,l:rr]
    depth=scene['depth_native'][t:bb,l:rr]
    allowed=scene['allowed'][t:bb,l:rr]&~scene['protected'][t:bb,l:rr]
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    support=np.logical_or.reduce([s['mask'][t:bb,l:rr] for s in scene['supports']
        if s['id'] in candidate.get('support_ids',[site['id']])])
    ring=np.ones(allowed.shape,bool)
    ring[max(0,y-t):min(bb-t,b-t),max(0,x-l):min(rr-l,r-l)]=False
    # Reject foreground depth and saturated highlights before colour/texture statistics.
    safe=allowed&(abs(depth-site['depth'])<=.085)&(rgb.max(axis=2)<240)
    selected=safe&ring&support
    basis='nearby_same_support_and_depth'
    if selected.sum()<100:
        selected=safe&ring
        basis='nearby_same_depth_surface_estimate'
    if selected.sum()<60:
        selected=safe
        basis='original_surface_under_object_local_only'
    if selected.sum()<20:
        selected=allowed&(rgb.max(axis=2)<248)
        basis='bounded_neighbourhood_depth_unknown'
    if selected.sum()<20:
        raise ValueError('insufficient_local_photography_samples')
    gray=np.float32(rgb)@LUMA
    lo,hi=np.percentile(gray[selected],[10,90])
    selected&=(gray>=lo)&(gray<=hi)
    samples=np.zeros((h,w),bool);samples[t:bb,l:rr]=selected
    pixels=np.float32(rgb[selected]);median=np.median(pixels,axis=0)
    luminance=float(median@LUMA)
    tint=np.clip(median/max(float(median.mean()),1),.88,1.12)
    sensor=sensor_statistics(rgb,selected)
    optics=camera_psf(rgb,safe)
    # A raised local black floor can suggest veiling, but is not a density measurement.
    # Keep it restrained; the failed global neutral-reflectance calibration is not used.
    p05,p50,p95=np.percentile(gray[safe] if safe.any() else gray[selected],[5,50,95])
    veil=float(np.clip((p05/255-.18)*.30,0,.10))
    # Airborne dust is inferred locally from a raised dark floor and reduced contrast.
    # Clear, dark-floor neighbourhoods are not forced to have a dust veil.
    raised=float(np.clip((p05/255-.08)/.46,0,1))
    compressed=float(np.clip(1-(p95-p05)/155,0,1))
    dust_veil=float(np.clip(raised*(.12+.16*compressed),0,.28))
    bright=pixels[(pixels@LUMA)>=np.percentile(pixels@LUMA,75)]
    air_rgb=np.median(bright,axis=0) if len(bright) else median
    air_rgb=.75*air_rgb+.25*float(air_rgb@LUMA)
    environment={'world_color':(np.clip(.22+(luminance-100)*.0012,.14,.35)*tint).tolist(),
        'light_color':(tint/tint.max()).tolist(),'key_direction':None,'haze':veil,
        'sample_count':int(selected.sum()),'sample_hash':hashlib.sha256(np.packbits(samples)).hexdigest(),
        'luma_median':luminance,'basis':basis,'sample_basis':basis,
        'dust_color':linear(median/255).tolist(),'sensor':sensor,'camera_psf':optics,
        'airborne_dust':{'amount':dust_veil,'color_linear':linear(air_rgb/255).tolist(),
            'basis':'local_same_depth_raised_dark_floor_and_contrast_estimate',
            'raised_dark_floor':raised,'contrast_compression':compressed},
        'local_context':{'version':VERSION,'bounds':bounds,'sample_pixels':int(selected.sum()),
            'basis':basis,'depth_tolerance':.085,'global_fallback':False,
            'confidence':'unknown' if 'unknown' in basis else 'estimated',
            'luma_range':[float(p05),float(p50),float(p95)],'veil_estimate':veil,
            'light_direction':'unresolved_local_soft_light','surface_texture_rms':sensor['surface_detail_rms']}}
    # A local photograph does not measure the surrounding HDR reflection field.
    # Retain the reference renderer's reflection profile, tinting it only locally.
    environment.update(light_spread=float(np.clip((p95-p05)/max(p95,20),.08,.7)),
                       light_level=float(np.clip(luminance/130,.65,1.35)),
                       lighting_basis='reference_reflection_profile_locally_tinted_estimate')
    return environment,samples


def local_veil(rgb,environment,params):
    """A restrained neighbourhood camera estimate, independent of surface wear."""
    from .schema import PHOTO_CONTROLS
    dusty=params.get('photo_style')=='dusty_scrap'
    dust=environment.get('airborne_dust',{}) if dusty else {}
    amount=float(np.clip(dust.get('amount',environment.get('haze',0))*PHOTO_CONTROLS['haze'][params.get('haze','auto')],0,.32 if dusty else .16))
    if params.get('atmosphere','auto')=='off':amount=0.
    air=np.asarray(dust.get('color_linear',environment['dust_color']),np.float32)
    captured=linear(rgb)*(1-amount)+air*amount
    result=np.where(captured<=.0031308,captured*12.92,1.055*captured**(1/2.4)-.055)
    return np.float32(result),{'amount':amount,'color_linear':air.tolist(),
        'basis':dust.get('basis','bounded_neighbourhood_veiling_estimate_not_air_density'),
        'base_amount':dust.get('amount',environment.get('haze',0)),
        'limit':.32 if dusty else .16,'mode':params.get('atmosphere','auto')}


def pool_object_rgb(rgb, alpha, kernel=2):
    """Average premultiplied cells once; empty cells never contaminate covered RGB."""
    if not isinstance(kernel, int) or not 1 <= kernel <= 64:
        raise ValueError('pool_kernel_must_be_integer_1_to_64')
    if kernel == 1:
        return rgb.copy()
    h, w = alpha.shape
    rgba = np.dstack((rgb * alpha[..., None], alpha))
    rgba = np.pad(rgba, ((0, -h % kernel), (0, -w % kernel), (0, 0)))
    ph, pw = rgba.shape[:2]
    if kernel > 8:
        # Large cells accumulate thousands of weights. Double precision keeps a
        # constant surface constant even when most of the cell is transparent.
        pooled = rgba.reshape(ph//kernel,kernel,pw//kernel,kernel,4).mean(axis=(1,3),dtype=np.float64)
    else:
        pooled = cv2.resize(rgba, (pw // kernel, ph // kernel), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(pooled, (pw, ph), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return np.float32(restored[..., :3] / np.maximum(restored[..., 3:], 1e-6))


def foreground_dust(clear_rgb, veiled_rgb, alpha, environment, local_air, target_px,
                    *, pool_weight=None, fog_fraction=None, pool_kernel=2, fog_weight=None,
                    uniform=False, fog_profile='local'):
    """Return only the change in linear object RGB for the fixed-placement trial."""
    clear = linear(clear_rgb)
    veiled = linear(veiled_rgb)
    core = alpha > .5
    distance = cv2.distanceTransform(np.pad(core.astype(np.uint8), 1), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    interior = np.maximum(distance - 1., 0.)
    context = environment['local_context']
    optics = environment['camera_psf']
    sampled = context['confidence'] == 'estimated' and context['sample_pixels'] >= 20
    if pool_weight is None:
        pool_weight = float(np.clip(.20 + .40 * (optics['sigma_px'] - .65), .15, .35))
    if fog_profile not in ('local','wide'):
        raise ValueError('unknown_fog_profile')
    fog_max = .85 if fog_profile == 'wide' else .36
    if not 0 <= pool_weight <= 1 or (fog_weight is not None and not 0 <= fog_weight <= fog_max):
        raise ValueError('effect_weight_out_of_range')
    if not sampled or optics['edge_samples'] < 12:
        pool_weight = 0.
    fade = np.minimum(interior / 2., 1.)
    pool = pool_weight * fade * fade * (3 - 2 * fade)
    if uniform:
        pool = np.float32(alpha > 0) * pool_weight
    base = local_air['amount']
    delta = np.zeros_like(clear)
    if pool_weight:
        delta = (pool_object_rgb(clear, alpha, pool_kernel) - clear) * pool[..., None] * (1 - base)

    radius = int(np.clip(round(target_px * .012), 2, 6))
    extra_limit = max(0., min(.09, (.32 - base) / max(1 - base, 1e-6)))
    extra = min(.35 * base, extra_limit) if fog_fraction is None else min(.65 * base, extra_limit) * fog_fraction
    if fog_weight is not None:
        # Explicit stronger diagnostic ladder. A clear neighbourhood remains clear.
        local_strength = float(np.clip(base / .15, 0., 1.25))
        extra = min(fog_weight * local_strength, max(0., (.55 - base) / max(1 - base, 1e-6)))
        if fog_profile == 'wide':
            # Explicit review ladder: actual extra-layer opacity, no hidden local multiplier.
            extra = min(fog_weight, max(0., (.90-base)/max(1-base,1e-6))) if base > 0 else 0.
    if not sampled:
        extra = 0.
    fade = np.minimum(interior / radius, 1.)
    haze = extra * fade * fade * (3 - 2 * fade)
    if fog_profile == 'wide':
        # Geometric alpha already feathers the true edge. An unfogged inner rim
        # would become a conspicuous dark outline at this diagnostic strength.
        haze = np.float32(alpha > 0) * extra
    air = np.asarray(local_air['color_linear'], np.float32)
    delta += (air - veiled - delta) * haze[..., None]

    total = base + (1 - base) * haze[core]
    return delta, {'pool_kernel': [pool_kernel, pool_kernel], 'pool_weight': pool_weight,
                   'pool_uniform': uniform, 'fog_profile': fog_profile,
                   'pool_weight_max': float(pool.max()), 'edge_samples': optics['edge_samples'],
                   'camera_sigma_px': optics['sigma_px'], 'fog_color_linear': local_air['color_linear'],
                   'fog_feather_px': 0 if fog_profile=='wide' else radius, 'fog_base': base, 'fog_extra_max': float(haze.max()),
                   'fog_total_median': float(np.median(total)) if total.size else base,
                   'fog_total_max': float(total.max()) if total.size else base,
                   'sample_basis': context['basis'], 'sample_pixels': context['sample_pixels']}


def seam_measurement(loss, receiver, alpha):
    """Measure the unobserved receiver cutoff separately from the object silhouette."""
    distance=cv2.distanceTransform(receiver.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    object_near=cv2.dilate((alpha>.01).astype(np.uint8),np.ones((7,7),np.uint8))>0
    edge=receiver&(distance<=2)&~object_near
    return {'samples':int(edge.sum()),'loss_p95':float(np.percentile(loss[edge],95)) if edge.any() else 0.,
            'basis':'receiver_boundary_outside_object_not_object_edge'}


def camera_residual(shaded, rgb, alpha, background, environment, params, seed, writable, *, object_delta=None,
                    fixed_sensor=None):
    """Filter only the inserted radiance difference, never a pasted rectangular background."""
    from .schema import PHOTO_CONTROLS
    sharpness=PHOTO_CONTROLS['sharpness'][params.get('sharpness','auto')]
    target=environment['camera_psf']['sigma_px']
    nominal=math.sqrt(max(target**2-.38**2,0))/sharpness
    topology_before=topology(alpha>.5)
    sigma=0.;overlap=1.;optical_alpha=alpha.copy()
    for trial in (nominal,nominal*.75,nominal*.5,nominal*.25):
        if trial<=.18:continue
        softened=cv2.GaussianBlur(alpha,(0,0),trial,borderType=cv2.BORDER_CONSTANT)
        a,b=softened>.5,alpha>.5
        iou=float((a&b).sum()/max(1,(a|b).sum()))
        if iou>=.995 and topology(a)==topology_before:
            sigma=trial;optical_alpha=softened;overlap=iou;break
    delta=shaded*(1-alpha[...,None])+linear(rgb)*alpha[...,None]-background
    if sigma:
        delta=cv2.GaussianBlur(delta,(0,0),sigma,borderType=cv2.BORDER_CONSTANT)
    delta*=writable[...,None]
    optical_alpha*=writable
    captured=np.maximum(background+delta,0)
    encoded=np.where(captured<=.0031308,captured*12.92,1.055*captured**(1/2.4)-.055)
    core=cv2.erode((alpha>.9).astype(np.uint8),np.ones((5,5),np.uint8))>0
    gray=np.float32(encoded)@LUMA
    residual=gray-cv2.GaussianBlur(gray,(0,0),.8)
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    flat=core&(gradient<=np.percentile(gradient[core],35)) if core.any() else core
    existing=float(np.median(abs(residual[flat]-np.median(residual[flat])))*1.4826) if flat.any() else 0.
    target_noise=min(.012,environment['sensor']['noise_sigma'])
    added=math.sqrt(max(0,target_noise**2-existing**2))
    if fixed_sensor is not None:
        existing=float(fixed_sensor['existing_noise_sigma'])
        added=float(fixed_sensor['added_noise_sigma'])
        if not np.isfinite([existing,added]).all() or min(existing,added)<0:
            raise ValueError('invalid_fixed_sensor_statistics')
    noise=np.random.default_rng(seed).normal(0,added,alpha.shape).astype(np.float32)
    if object_delta is not None:
        extra=object_delta*alpha[...,None]
        if sigma:
            extra=cv2.GaussianBlur(extra,(0,0),sigma,borderType=cv2.BORDER_CONSTANT)
        visible=(alpha>0)&writable
        # Only the new effect is clipped. Keep the baseline shadow, edge and noise intact.
        changed=np.maximum(captured[visible]+extra[visible],0)
        encoded[visible]=np.where(changed<=.0031308,changed*12.92,1.055*changed**(1/2.4)-.055)
    encoded+=np.clip(noise,-3*target_noise,3*target_noise)[...,None]*optical_alpha[...,None]
    edited=(np.max(abs(delta),axis=2)>1e-7)|(optical_alpha>0)
    return encoded,edited,{'blur_sigma':float(sigma),'measured_camera_sigma':target,
        'optical_iou':overlap,'added_noise_sigma':added,'existing_noise_sigma':existing,
        'geometric_alpha_unchanged':True,'compositing':'filtered_insertion_difference_only'}
