import hashlib
import math
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image

from .geometry import contact_support
from .schema import TONE_PROFILES, PHOTO_CONTROLS
from .photography import match_linear, sensor_statistics
from .pile import depth_order
from .framing import edit_bounds
from .fusion import camera_psf, camera_tones, contact_darkening, optical_layer, reflection_probe
from .atmosphere import (calibrate_environment, transmission_fields, fogged_radiance,
                         shadowed_background)
from .local import neighbourhood, camera_residual, seam_measurement, local_veil, foreground_dust


def contact_shading_supported(candidate,coverage,coherent=False):
    """Estimated receiver height need not erase an otherwise simulated contact.

    This only permits a small contact-light term. It never changes the physical
    evidence status, and cannot supply missing contacts or hide failed dynamics.
    """
    physics=candidate.get('terrain_physics',{})
    if not coverage or min(coverage)<.9 or not physics.get('stable'):return False
    gap=candidate.get('contact_gap',{})
    if candidate.get('receiver',{}).get('contact_evidence_version')=='visible-support-v1':
        if (gap.get('status')=='fail' or gap.get('simulation_status')!='pass' or
                gap.get('visible_support',{}).get('status')!='pass'):
            return False
        return bool(coherent and not physics.get('failure_codes') and physics.get('center_supported')
                    and physics.get('contact_count',0)>0 and physics.get('status') in ('pass','unknown'))
    if physics.get('status')=='pass':return True
    return bool(coherent and physics.get('status')=='unknown'
        and not physics.get('failure_codes') and physics.get('center_supported')
        and physics.get('contact_count',0)>0
        and candidate.get('contact_gap',{}).get('status')=='pass')


def linear(rgb):
    rgb=np.asarray(rgb,dtype=np.float32)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def srgb(rgb):
    rgb=np.maximum(rgb,0)
    return np.where(rgb<=.0031308,rgb*12.92,1.055*rgb**(1/2.4)-.055)


def sample_environment(scene, candidate, *, scene_lighting=False, native=False, local=False):
    if local:
        return neighbourhood(scene,candidate)
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    w,h=scene['size'];x,y,r,b=candidate['bounds'];pad=round(max(r-x,b-y)*.8)
    l,t,rr,bb=max(0,x-pad),max(0,y-pad),min(w,r+pad),min(h,b+pad)
    sample=np.zeros((h,w),bool);sample[t:bb,l:rr]=True
    support_masks=[s['mask'] for s in scene['supports'] if s['id'] in candidate.get('support_ids',[site['id']])]
    support_union=np.logical_or.reduce(support_masks)
    sample &= support_union & ~scene['protected']
    sample[y:b,x:r]=False
    rgb=np.asarray(scene['frame']);lum=rgb@np.array([.2126,.7152,.0722])
    if sample.sum()<100:
        # frame is the untouched photograph: the surface under the future object is valid evidence.
        sample=support_union.copy() & ~scene['protected']
    if sample.sum()<100:raise ValueError('insufficient_same_surface_light_samples')
    lo,hi=np.percentile(lum[sample],[12,92]);sample &= (lum>=lo)&(lum<=hi)
    sample_basis='same_support_estimate'
    if native:
        # Multiple support fragments can span depth. Use the landing layer for calibration.
        comparable=sample & (abs(scene['depth_native']-site['depth'])<=.085)
        chroma=(rgb.max(axis=2).astype(float)-rgb.min(axis=2))/255
        comparable &= (chroma<.22)&(rgb.max(axis=2)<240)
        if comparable.sum()>=100:
            sample=comparable
            sample_basis='same_depth_neutral_surface'
        else:
            sample_basis='support_fallback_insufficient_comparable_material'
    pixels=rgb[sample].astype(float);median=np.median(pixels,axis=0)
    # Keep albedo separate from illumination: only a restrained neutral cast is estimated.
    tint=np.clip(median/max(float(median.mean()),1),.88,1.12)
    luminance=float(np.median(lum[sample]))
    ambient=np.clip(.22+(luminance-100)*.0012,.14,.35)
    answer=scene['answer']
    environment={'world_color':(ambient*tint).tolist(),'light_color':(tint/tint.max()).tolist(),
                 'key_direction':answer['light_direction'] if answer['light_confidence']>=.55 else None,
                 'haze':answer['haze'],'sample_count':int(sample.sum()),'sample_basis':sample_basis,
                 'sample_hash':hashlib.sha256(np.packbits(sample)).hexdigest(),
                 'luma_median':luminance,'basis':'same_support_estimate',
                 'dust_color':linear(median/255).tolist(),
                 'sensor':sensor_statistics(rgb[t:bb,l:rr],sample[t:bb,l:rr]) if sample[t:bb,l:rr].sum()>=100
                          else sensor_statistics(rgb,sample)}
    if scene_lighting:
        # Use a blurred field so individual steel edges do not become light sources.
        field=cv2.resize(np.float32(lum[t:bb,l:rr]),(96,96),interpolation=cv2.INTER_AREA)
        field=cv2.GaussianBlur(field,(0,0),4.)
        region=cv2.resize(sample[t:bb,l:rr].astype(np.uint8),(96,96),interpolation=cv2.INTER_NEAREST)>0
        p20,p80=np.percentile(field[region] if region.sum()>=100 else lum[sample],[20,80])
        environment.update(light_spread=float(np.clip((p80-p20)/max(p80,20),.08,.7)),
                           light_level=float(np.clip(luminance/130,.65,1.35)),
                           light_confidence=answer['light_confidence'],
                           lighting_basis='broad_direction_estimate' if environment['key_direction'] else 'soft_dome_estimate')
    if native:
        # The projection supplies only a local reflection estimate, not measured HDR.
        valid=scene['allowed']&~scene['protected']
        reflection_frame=rgb
        if scene.get('atmosphere'):
            compensated=calibrate_environment(scene,candidate,environment,sample)
            reflection_frame=cv2.resize(compensated,(w,h),interpolation=cv2.INTER_LINEAR)
        probe,fraction=reflection_probe(reflection_frame,valid,candidate,environment)
        environment.update(reflection_probe=probe,reflection_coverage=fraction,
                           lighting_basis='local_ground_reflection_estimate',
                           camera_psf=camera_psf(rgb[t:bb,l:rr],valid[t:bb,l:rr]))
    return environment,sample


def warp(source, full_px, offset, bounds, *, window=None, nearest=False):
    x,y,r,b=bounds
    if window is None:
        scale=full_px/source.shape[1]
        dx,dy=offset[0]-x,offset[1]-y
    else:
        ox,oy,rr,bb=window['rect']
        if source.shape[:2]!=(bb-oy,rr-ox):
            raise ValueError('render_pass_does_not_match_window')
        scale=full_px/window['frame']
        # Camera UVs are pixel edges. OpenCV's integer coordinates are pixel centres.
        dx=offset[0]+(ox+.5)*scale-.5-x
        dy=offset[1]+(oy+.5)*scale-.5-y
        if scale<.8 and not nearest:
            sigma=.45*math.sqrt(1/scale**2-1)
            source=cv2.GaussianBlur(source,(0,0),sigma,borderType=cv2.BORDER_CONSTANT)
    m=np.float32([[scale,0,dx],[0,scale,dy]])
    flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(source,m,(r-x,b-y),flags=flags,borderMode=cv2.BORDER_CONSTANT)


def object_window(meta):
    return None if meta.get('legacy_grid') else meta.get('object_window')


def taper_shadow(loss, width, receiver):
    """Taper a shadow clipped by the estimated receiver or by the edit window."""
    h,w=loss.shape
    if min(h,w)<3:return loss
    xx=np.minimum(np.arange(w),np.arange(w)[::-1])
    yy=np.minimum(np.arange(h),np.arange(h)[::-1])
    boundary=cv2.distanceTransform(receiver.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    distance=np.minimum(boundary,np.minimum(xx[None,:],yy[:,None]))
    t=np.float32(np.clip(distance/width,0,1))
    return loss*t*t*(3-2*t)


def depth_on_image(path, alpha, full_px, offset, bounds, camera, window):
    distance=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if distance is None:raise ValueError('missing_object_depth_pass')
    if distance.ndim==3:distance=distance[:,:,0]
    if window is None:
        ox,oy,n=0,0,distance.shape[1]
    else:
        ox,oy=window['rect'][:2];n=window['frame']
    vv,uu=np.ogrid[oy:oy+distance.shape[0],ox:ox+distance.shape[1]]
    tangent=2*math.tan(math.radians(camera['fov'])/2)
    distance=distance/np.sqrt(1+(((uu+.5)/n-.5)*tangent)**2+(((vv+.5)/n-.5)*tangent)**2)
    distance=np.where((alpha>.001)&np.isfinite(distance)&(distance<1e8),distance,0).astype(np.float32)
    if window is not None:
        # Z is not an antialiased coverage pass. Never interpolate two surfaces'
        # depths or infer transparency from a missing depth sample.
        return warp(distance,full_px,offset,bounds,window=window,nearest=True)
    return warp(distance*alpha,full_px,offset,bounds)/np.maximum(warp(alpha,full_px,offset,bounds),1e-6)


def match_rgb(rgb, alpha, background_pixels, specular_weight, tone, seed):
    core=alpha>.75
    if core.sum()<20:raise ValueError('object_too_small')
    scene_gray=background_pixels@np.array([.2126,.7152,.0722])
    object_gray=rgb@np.array([.2126,.7152,.0722])
    before=float(np.median(object_gray[core]))
    target=float(np.median(scene_gray))/255
    extra_ev,temperature=TONE_PROFILES[tone]
    ev=float(np.clip(math.log2(max(target,.02)/max(before,.02))*.65,-.65,.45))+extra_ev
    # Preserve the high-frequency highlight response; do not clamp it to the support percentile.
    gain=2**(ev*(1-.65*specular_weight))
    rgb=np.maximum(rgb,0)*gain[:,:,None]
    median=np.median(background_pixels,axis=0)
    cast=np.clip(median/max(1.,median.mean()),.94,1.06)
    cast*=np.array([1+temperature*.008,1,1-temperature*.008])
    rgb*=cast
    rgb=np.clip(rgb,0,1)
    lab=cv2.cvtColor(np.float32(rgb),cv2.COLOR_RGB2LAB)
    scene_lab=cv2.cvtColor(np.float32(background_pixels.reshape(-1,1,3)/255),cv2.COLOR_RGB2LAB)
    object_chroma=float(np.median(np.linalg.norm(lab[:,:,1:][core],axis=1)))
    scene_chroma=float(np.median(np.linalg.norm(scene_lab[:,:,1:],axis=2)))
    chroma_gain=float(np.clip(1-.010*max(0,object_chroma-scene_chroma),.60,1.))
    lab[:,:,1:]*=chroma_gain
    rgb=np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),0,1)
    gray=rgb.mean(axis=2);blur=cv2.GaussianBlur(gray,(0,0),.7)
    detail=float(np.median(abs(gray[core]-blur[core])))
    noise=float(np.clip(np.std(scene_gray)*.025/255,.001,.008))
    sigma=.35 if detail>.045 else 0.
    if sigma:
        weight=cv2.GaussianBlur(alpha,(0,0),sigma)
        smooth=cv2.GaussianBlur(rgb*alpha[:,:,None],(0,0),sigma)/np.maximum(weight[:,:,None],1e-5)
        rgb=rgb*.55+smooth*.45
    rng=np.random.default_rng(seed)
    rgb+=np.clip(rng.normal(0,noise,alpha.shape),-3*noise,3*noise)[:,:,None]
    return np.clip(rgb,0,1),{'exposure_ev':ev,'white_balance':cast.tolist(),'noise_sigma':noise,
                           'blur_sigma':sigma,'chroma_gain':chroma_gain,
                           'before_luma':before*255,'target_luma':target*255}


def compose(scene,candidate,render_dir,render_meta,environment,samples,params,seed,out,
            *,dust_overlay=False,capture_local=False,fixed_tone=None,layer_output=None):
    render_dir=Path(render_dir)
    source=np.asarray(Image.open(render_dir/'object.png').getchannel('A'),np.float32)/255
    alpha_dir=Path(candidate.get('fixed_alpha_dir',render_dir))
    fixed_alpha=alpha_dir!=render_dir
    alpha_source=np.asarray(Image.open(alpha_dir/'object.png').getchannel('A'),np.float32)/255 if fixed_alpha else source
    w,h=scene['size'];bounds=edit_bounds(candidate,scene['size'])
    x,y,r,b=bounds
    # Anchor derives from contact geometry, never from the alpha or shadow bounding box.
    offset=np.asarray(candidate['anchor_px'])-np.asarray(render_meta['anchor_uv'])*candidate['full_px']
    full=candidate['full_px'];window=object_window(render_meta)
    color_alpha=warp(source,full,offset,bounds,window=window)
    alpha_window=candidate.get('fixed_alpha_window') if fixed_alpha else window
    alpha=warp(alpha_source,full,offset,bounds,window=alpha_window) if fixed_alpha else color_alpha
    spec=cv2.imread(str(render_dir/'specular0001.exr'),cv2.IMREAD_UNCHANGED)
    diff=cv2.imread(str(render_dir/'diffuse0001.exr'),cv2.IMREAD_UNCHANGED)
    if spec is None or diff is None:raise ValueError('missing_linear_material_passes')
    combined_path=render_dir/'combined0001.exr'
    combined=cv2.imread(str(combined_path),cv2.IMREAD_UNCHANGED) if combined_path.is_file() else None
    radiance=(combined[:,:,:3] if combined is not None else spec[:,:,:3]+diff[:,:,:3])[:,:,::-1]
    # EXR radiance is associated with its float coverage, not the rounded PNG alpha.
    # Keep geometric coverage frozen while avoiding bright/dark fractional edge fringes.
    beauty_alpha=combined[:,:,3] if combined is not None and combined.shape[2]==4 else source
    beauty_coverage=warp(beauty_alpha,full,offset,bounds,window=window)
    raw=warp(radiance,full,offset,bounds,window=window)/np.maximum(beauty_coverage[...,None],1e-6)
    highlight=np.max(spec[:,:,:3],axis=2)/np.maximum(np.max(spec[:,:,:3]+diff[:,:,:3],axis=2),.0001)
    if 'object_window' in render_meta:
        # Beauty is the denoised radiance. The other passes are only smooth guides.
        highlight=cv2.GaussianBlur(highlight*source,(0,0),.85)/np.maximum(cv2.GaussianBlur(source,(0,0),.85),1e-6)
    highlight=warp(highlight*source,full,offset,bounds,window=window)/np.maximum(color_alpha,1e-6)
    channels={}
    for name in ('metallic','upward'):
        path=render_dir/(name+'0001.exr')
        if path.is_file():
            value=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if value.ndim==3:value=value[:,:,0]
            channels[name]=warp(value,full,offset,bounds,window=window)/np.maximum(color_alpha,1e-6)
    photo_environment={**environment,'material_classes':candidate.get('material_classes',[])}
    local=params.get('photo_model')=='local'
    effects=params.get('effects_enabled',False)
    if (dust_overlay or capture_local or effects) and not local:
        raise ValueError('object_effects_require_local_photography')
    reference=params.get('photo_model','reference') in ('reference','local')
    native=params.get('photo_model')=='native'
    reference_pixels=np.asarray(scene['frame'])[samples]
    rgb,photo=match_linear(raw,alpha,reference_pixels,highlight,params,seed,photo_environment,
                           reference=reference,camera_capture=not (native or local),
                           surface_weathering=not any(name in render_meta for name in
                               ('scanned_materials', 'matte_profile', 'generated_finish')),
                           fixed_tone=fixed_tone,**channels)
    photo['radiance_basis']='combined_EXR' if combined is not None else 'cached_diffuse_plus_specular_EXR'
    chroma_limits=chroma_groups=None
    if params.get('chroma_mode') in ('local_budget','global_mean'):
        from .chroma_budget import masks, budget, compress, measure
        from .photography import display
        chroma_limits=budget(reference_pixels,environment.get('local_context',{}).get('confidence','unknown'))
        chroma_groups=masks(display(raw),alpha,channels.get('metallic'),candidate.get('material_classes',[]))
        # This budget owns the FINAL sensor-space colour, with no retention floor.
        # Applying it after the camera residual prevents colour changes leaking into
        # background pixels through the camera PSF, while retaining legacy replay.
        photo['chroma_initial']=measure(rgb,chroma_groups,chroma_limits)
    if local:
        clear_rgb=rgb
        rgb,local_air=local_veil(rgb,environment,{**params,'atmosphere':'off'} if params.get('fog_mode')=='local_auto' else params)
        photo.update(local_air=local_air,haze=local_air['amount'])
        for label,value in PHOTO_CONTROLS['haze'].items():
            photo['effective_controls'][label]['haze']=float(np.clip(local_air['base_amount']*value,0,local_air['limit'])) if params.get('atmosphere','auto')!='off' else 0.
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    support=np.logical_or.reduce([s['mask'][y:b,x:r] for s in scene['supports']
        if s['id'] in candidate.get('support_ids',[site['id']])]);protected=scene['protected'][y:b,x:r]
    order={'confidence':'legacy_estimate','basis':'frozen_original_occlusion'}
    object_z=np.zeros_like(alpha)
    if candidate.get('placement') and not candidate.get('legacy_occlusion'):
        object_z=depth_on_image(alpha_dir/'depth0001.exr',alpha_source,full,offset,bounds,
                                render_meta['camera'],alpha_window)
        foreground,order=depth_order(scene,candidate,bounds,object_z,alpha,refine=True)
        if window is not None and not fixed_alpha:
            order['depth_unresolved_fraction']=float(((object_z<=0)&(alpha>.5)).sum()/max(1,(alpha>.5).sum()))
    else:foreground=(scene['depth_native'][y:b,x:r]>site['depth']+.10)&~support
    foreground &= ~protected
    visible_alpha=alpha*((~foreground)&(~protected))
    if chroma_groups is not None:
        chroma_groups={k:v&(visible_alpha>.95) for k,v in chroma_groups.items()}
    local_atmospheric=None
    if local and params.get('fog_mode')=='local_auto':
        from .local_atmosphere import fields_for
        object_t,background_t,airlight,air_summary=fields_for(environment['atmospheric_context'],bounds,
            object_z,order,candidate,scene,float(params.get('fog_gain',1)),params.get('atmosphere')!='off')
        rgb=srgb(fogged_radiance(linear(clear_rgb),object_t,airlight))
        local_atmospheric=(background_t,airlight)
        core=visible_alpha>.5
        local_air={'amount':float(np.median(1-object_t[core])) if core.any() else 0.,
                   'color_linear':np.median(airlight[core],axis=0).tolist() if core.any() else [0.,0.,0.],
                   'mode':params.get('atmosphere'),'basis':air_summary['basis']}
        photo.update(local_air=local_air,haze=local_air['amount'],
                     atmosphere={**air_summary,'mode':params.get('atmosphere'),'transmission_range':
                         np.percentile(object_t[core],[5,50,95]).tolist() if core.any() else [1.,1.,1.]})
    occlusion_fraction=float(1-visible_alpha.sum()/max(alpha.sum(),1e-6))
    foot=np.zeros((b-y,r-x),np.uint8)
    for poly in render_meta['contact_polygons']:
        coords=(np.asarray(poly)*full+offset-[x,y])*16
        cv2.fillConvexPoly(foot,np.rint(coords).astype(np.int32),255,shift=4)
    cover=contact_support(foot,support)
    shadow_dir=Path(candidate.get('fixed_shadow_dir',render_dir))
    parts={} if candidate.get('fixed_shadow_dir') else render_meta.get('part_paths',{})
    with_path=Path(parts.get('with_shadow',shadow_dir))/'with-shadow-linear0001.exr'
    without_path=Path(parts.get('without_shadow',shadow_dir))/'without-shadow-linear0001.exr'
    if with_path.is_file() and without_path.is_file():
        a=cv2.imread(str(with_path),cv2.IMREAD_UNCHANGED)[:,:,:3].mean(axis=2)
        base=cv2.imread(str(without_path),cv2.IMREAD_UNCHANGED)[:,:,:3].mean(axis=2)
        shadow_basis=render_meta.get('shadow_basis') or 'scene_linear_EXR_ratio'
    else:
        with_shadow=np.asarray(Image.open(shadow_dir/'with-shadow.png').convert('RGB'),dtype=np.float32)/255
        without_shadow=np.asarray(Image.open(shadow_dir/'without-shadow.png').convert('RGB'),dtype=np.float32)/255
        a=linear(with_shadow).mean(axis=2);base=linear(without_shadow).mean(axis=2)
        shadow_basis='legacy_sRGB_decoded_ratio'
    ratio=a if shadow_basis=='cycles_shadow_catcher_ratio' else a/np.maximum(base,.015)
    loss=np.clip(1-ratio,0,.65)
    loss[base<.015]=0
    shadow_window=render_meta.get('shadow_window') if window is not None and parts else None
    loss=warp(loss,full,offset,bounds,window=shadow_window)
    loss *= support & ~foreground & ~protected
    if not reference or local:
        # A finite, image-derived receiver is not an observed cast-shadow edge.
        # Fade its cutoff inside the support without adding shadow to unsupported pixels.
        receiver=warp(np.float32(base>=.015),full,offset,bounds,window=shadow_window)>.6
        receiver &= support & ~foreground & ~protected
        before_seam=seam_measurement(loss,receiver,alpha)
        width=max(24,min(96,round(candidate['target_px']*.10))) if local else max(5,min(28,round(candidate['target_px']*.036)))
        loss=taper_shadow(loss,width,receiver)
        if local:
            photo['patch_boundary']={'before':before_seam,'after':seam_measurement(loss,receiver,alpha),
                'feather_px':width,'object_alpha_changed':False}
            physics=candidate.get('terrain_physics',{})
            supported=contact_shading_supported(candidate,cover,params.get('integration_mode')=='coherent')
            contact=contact_darkening(foot,support,foreground|protected,candidate['target_px'],supported)
            loss=np.maximum(loss,contact)
            photo['contact_ambient_loss']=float(contact.max())
            photo['contact_shading_basis']='verified_simulation_on_estimated_receiver' if supported and physics.get('status')!='pass' else 'measured_or_disabled'
    base_rgb=np.asarray(scene['frame']).copy();crop=base_rgb[y:b,x:r].astype(np.float32)/255
    if native:
        physics=candidate.get('terrain_physics',{})
        supported=bool(cover and min(cover)>=.9 and physics.get('stable') and physics.get('status')=='pass')
        contact=contact_darkening(foot,support,foreground|protected,candidate['target_px'],supported)
        loss=np.maximum(loss,contact)
        local_tones={'basis':'material_luminance_preserved_after_local_light_calibration'}
        if not environment.get('calibration'):
            rgb,local_tones=camera_tones(rgb,alpha,reference_pixels,channels.get('metallic'),photo['polymer_preservation'])
        photo['after_luma']=float(np.median((rgb@np.float32([.2126,.7152,.0722]))[alpha>.85])*255)
        if foreground.any() and order.get('confidence')=='estimated' and (foreground&(alpha>.5)).any():
            # A nearby foreground face screens a small amount of diffuse light.
            distance=cv2.distanceTransform((~foreground).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            rgb*=1-.07*np.exp(-distance[...,None]/max(2.,min(7.,candidate['target_px']*.012)))
        atmospheric=None
        if scene.get('atmosphere'):
            object_t,background_t,airlight,depth_basis=transmission_fields(scene,bounds,object_z,order,
                PHOTO_CONTROLS['haze'][params.get('haze','auto')],params.get('atmosphere','auto')=='auto',candidate=candidate)
            rgb=srgb(fogged_radiance(linear(rgb),object_t,airlight))
            atmospheric=(background_t,airlight)
            summary=scene['atmosphere']['summary']
            photo['atmosphere']={'signature':scene['atmosphere']['signature'],'mode':params.get('atmosphere','auto'),
                'confidence':summary['confidence'],'depth_basis':depth_basis,
                'depth_confidence':summary.get('depth_confidence','unknown'),
                'sampling_confidence':summary.get('sampling_confidence',0.),
                'transmission_range':np.percentile(object_t[alpha>.85],[5,50,95]).tolist(),
                'airlight_camera_linear':np.median(airlight[alpha>.85],axis=0).tolist(),
                'analysis':str(Path(scene['atmosphere']['folder'])/'estimate.json')}
            photo['haze']=float(np.median(1-object_t[alpha>.85]))
            photo['after_luma']=float(np.median((rgb@np.float32([.2126,.7152,.0722]))[alpha>.85])*255)
            for label,value in PHOTO_CONTROLS['haze'].items():
                photo['effective_controls'][label]['haze']=value if photo['haze']>.001 else 0.
        psf=environment['camera_psf']
        sharpness=PHOTO_CONTROLS['sharpness'][params.get('sharpness','auto')]
        premult,optical_alpha,sigma,overlap=optical_layer(rgb,alpha,psf['sigma_px'],sharpness)
        write=(~foreground)&(~protected)&(scene['allowed'][y:b,x:r] | (alpha>0))
        optical_alpha=optical_alpha*write
        premult=premult*write[...,None]
        photo.update(local_tones=local_tones,lighting_basis=environment['lighting_basis'],
                     reflection_coverage=environment['reflection_coverage'],camera_psf=psf,
                     blur_sigma=sigma,optical_iou=overlap,contact_ambient_loss=float(contact.max()))
        for label,value in PHOTO_CONTROLS['sharpness'].items():
            nominal=math.sqrt(max(psf['sigma_px']**2-.38**2,0))/value
            photo['effective_controls'][label]['sharpness']=nominal if nominal>.18 else 0.
    else:
        optical_alpha=visible_alpha
        premult=linear(rgb)*visible_alpha[...,None]
    loss[loss<1/512]=0
    if native:
        photo['visible_shadow_pixels']=int(((loss>.025)&(optical_alpha<.10)).sum())
    shaded=shadowed_background(linear(crop),loss,*atmospheric) if native and atmospheric is not None else linear(crop)*(1-loss[:,:,None])
    if local:
        atmospheric_args=local_atmospheric or (np.full(alpha.shape,1-local_air['amount'],np.float32),
                                               np.asarray(local_air['color_linear'],np.float32))
        shaded=shadowed_background(linear(crop),loss,*atmospheric_args)
    result=srgb(shaded*(1-optical_alpha[:,:,None])+premult)
    if native:
        noise=float(np.clip(environment['sensor']['noise_sigma'],0,.012))
        residual=np.random.default_rng(seed).normal(0,noise,optical_alpha.shape).astype(np.float32)
        result+=residual[...,None]*optical_alpha[...,None]
        photo['added_noise_sigma']=noise
    edited=(optical_alpha>0)|(loss>0)
    photoshop_layers={} if layer_output is not None else None
    if local:
        writable=(~foreground)&(~protected)&(scene['allowed'][y:b,x:r] | (alpha>0))
        object_delta=None
        if effects:
            from .effects import optical_delta
            optical_air={**local_air,'transmission_field':object_t} if local_atmospheric else local_air
            object_delta,details=optical_delta(clear_rgb,rgb,visible_alpha,environment,optical_air,candidate['target_px'],params)
            finish = render_meta.get('generated_finish')
            details.update(surface_dust_applied=bool(
                (finish and finish['dust_strength'] > 0 and finish['materials']) or render_meta.get('matte_materials')),
                matte_profile=render_meta.get('matte_profile'))
            if finish:
                details['generated_finish'] = finish
            photo['effects']=details
        elif dust_overlay:
            object_delta,details=foreground_dust(clear_rgb,rgb,visible_alpha,environment,local_air,candidate['target_px'])
            photo['dust_overlay']=details
        # Treat automatic air as an object-only residual too. Blurring a fogged base
        # would spread fog changes into holes/background and change the noise estimate.
        capture_rgb=clear_rgb if local_atmospheric and effects else rgb
        capture_options={'capture_layers':photoshop_layers} if photoshop_layers is not None else {}
        result,edited,imaging=camera_residual(shaded,capture_rgb,visible_alpha,linear(crop),environment,params,seed,writable,
                                             object_delta=object_delta,**capture_options)
        photo.update(imaging,local_context=environment['local_context'])
        photo['after_luma']=float(np.median((np.clip(result,0,1)@np.float32([.2126,.7152,.0722]))[visible_alpha>.85])*255)
        photo['color_pipeline']='scene_linear_EXR_exposure_WB_AgX_once_local_camera_veil_residual_filter'
        for label,value in PHOTO_CONTROLS['sharpness'].items():
            photo['effective_controls'][label]['sharpness']=math.sqrt(max(environment['camera_psf']['sigma_px']**2-.38**2,0))/value
    local_weather = (render_meta.get('generated_finish') or {}).get('version') == 'generated-pbr-v8'
    if local_weather:
        from .surface_context import match_surface_context
        result,photo['surface_context']=match_surface_context(result,visible_alpha,reference_pixels,
                                                              environment.get('local_context',{}))
    if chroma_groups is not None:
        exported_preview=np.rint(np.clip(result,0,1)*255).astype(np.float32)/255
        initial_final=measure(exported_preview,chroma_groups,chroma_limits)
        photo['chroma_recompositions']=int(initial_final['status']=='fail')
        if initial_final['status']=='fail':
            result,photo['chroma_recomposition']=compress(result,chroma_groups,chroma_limits,visible_alpha)
    saturation_mask=None
    if params.get('chroma_mode')=='global_mean':
        from .global_saturation import background_statistics, apply as apply_global_saturation
        if 'global_saturation_statistics' not in scene:
            scene['global_saturation_statistics']=background_statistics(np.asarray(scene['frame']))
        # Use the exact exported local-budget result as this diagnostic's input.
        result=np.rint(np.clip(result,0,1)*255).astype(np.float32)/255
        result,photo['global_saturation'],saturation_mask=apply_global_saturation(
            result,visible_alpha,scene['global_saturation_statistics'])
    crop_out=np.uint8(np.rint(np.clip(result,0,1)*255))
    original=base_rgb[y:b,x:r].copy()
    base_rgb[y:b,x:r][edited]=crop_out[edited]
    Image.fromarray(base_rgb).save(out)
    with Image.open(out) as saved:
        exported=np.asarray(saved.convert('RGB'))
        exported_size=list(saved.size)
    original_full=np.asarray(scene['frame'])
    if local_weather:
        core = visible_alpha > .85
        if core.any():
            photo['after_luma'] = float(np.median(exported[y:b,x:r][core] @ [.2126, .7152, .0722]))
    changed=np.any(exported!=original_full,axis=2)
    permitted=np.zeros((h,w),bool);permitted[y:b,x:r]=edited
    outside_count=int((changed&~permitted).sum())
    protected_count=int((changed&scene['protected']).sum())
    foreground_changes=int(np.any(exported[y:b,x:r]!=original_full[y:b,x:r],axis=2)[foreground].sum())
    hidden_contacts=float(((foot>0)&foreground).sum()/max(1,(foot>0).sum()))
    anchor_error=float(np.linalg.norm(offset+np.asarray(render_meta['anchor_uv'])*candidate['full_px']-candidate['anchor_px']))
    expected_offset=np.asarray(candidate['offset'])
    projection_error=float(np.linalg.norm(offset-expected_offset))
    yy,xx=np.nonzero(visible_alpha>.5)
    if len(xx)==0:raise ValueError('object_fully_occluded')
    major=float(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1))
    ay,ax=np.nonzero(alpha>.5)
    unoccluded_major=float(max(ax.max()-ax.min()+1,ay.max()-ay.min()+1))
    expected=candidate.get('expected_major_px',unoccluded_major)
    tolerance=max(3.,expected*.05)
    physics=candidate.get('terrain_physics',{})
    tilt=physics.get('axis_angle_deg',candidate.get('axis_angle_deg'))
    checks={
      'lying':{'status':'pass' if tilt is not None and tilt<=15 else 'unknown',
               'measurement_deg':tilt,'threshold_deg':15,'basis':'compound_simulation_on_estimated_plane'},
      'support':{'status':('unknown' if hidden_contacts>.10 or 'support_geometry_uncertain' in candidate.get('uncertainties',[]) else 'pass')
                 if cover and min(cover)>=.9 else 'unknown' if candidate.get('allow_uncertain') else 'fail',
                 'measurements':cover,'threshold':.9,'basis':'estimated_surface','hidden_contact_fraction':hidden_contacts},
      'camera_clearance':candidate.get('camera_interval_check',{
                 'status':'unknown' if candidate.get('camera_risks') else 'pass',
                 'risks':candidate.get('camera_risks',[]),'elevation_interval':candidate['elevation_interval'],'basis':'interval_projection'}),
      'physics':{'status':physics.get('status','pass' if physics.get('stable') else 'unknown'),'evidence':physics,
                 'basis':'estimated_plane_not_measured_irregular_pile'},
      'occlusion':{'status':'fail' if foreground_changes or (native and occlusion_fraction>.45) else 'unknown' if
                   (order.get('confidence')=='unknown' and order.get('occluded_fraction',0)>.02)
                   or order.get('ambiguous_fraction',0)>.25 or order.get('depth_unresolved_fraction',0)>.02 else 'pass',
                   'foreground_changed_pixels':foreground_changes,'depth_evidence':order},
      'scale':{'status':candidate['scale'].get('status','pass') if abs(unoccluded_major-expected)<=tolerance else 'fail',
               'basis':'estimated_local_reference','measurement_px':unoccluded_major,'visible_px':major,
               'expected_px':expected,'tolerance_px':tolerance,'evidence':candidate['scale']},
      'anchor':{'status':'pass' if max(anchor_error,projection_error)<=1.5 else 'fail',
                'error_px':anchor_error,'projection_error_px':projection_error,'threshold_px':1.5},
      'protected':{'status':'pass' if protected_count==0 else 'fail','changed_pixels':protected_count},
      'outside_edit':{'status':'pass' if outside_count==0 else 'fail','changed_pixels':outside_count},
      'holes_alpha':{'status':'pass','basis':'unchanged_object_alpha_before_foreground_mask'},
      'size':{'status':'pass' if exported_size==[w,h] else 'fail','pixels':exported_size},
    }
    if chroma_groups is not None:
        photo['final_chroma']=measure(exported[y:b,x:r].astype(np.float32)/255,chroma_groups,chroma_limits)
        checks['local_chroma']=photo['final_chroma']
    if saturation_mask is not None:
        from .global_saturation import measure as measure_global_saturation
        measured=measure_global_saturation(exported[y:b,x:r].astype(np.float32)/255,
            saturation_mask,scene['global_saturation_statistics'])
        photo['global_saturation']['after']=measured
        checks['global_saturation']=measured
    if candidate.get('strict_contact'):
        checks['contact_gap']=candidate.get('contact_gap',{'status':'unknown','reason':'no_metric_gap_audit'})
        if not physics.get('stable'):
            checks['physics']['status']='fail'
        if not cover or min(cover)<.9:checks['support']['status']='fail'
    if photo.get('atmosphere'):
        atmosphere=photo['atmosphere']
        checks['atmosphere']={'status':'pass' if atmosphere['confidence']=='estimated' and atmosphere['mode']=='auto' else 'unknown',
            'basis':'image_estimated_airlight_and_transmission','evidence':atmosphere}
    if local:
        context=environment['local_context']
        checks['local_samples']={'status':'unknown' if context['confidence']=='unknown' else 'pass','evidence':context}
        checks['patch_boundary']={'status':'pass' if photo['patch_boundary']['after']['loss_p95']<=.04 else 'unknown',
            'evidence':photo['patch_boundary'],'threshold_linear_loss':.04}
        if params.get('shadow_mode')=='matched_receiver':
            from .seams import audit_boundary
            audit=audit_boundary(original,exported[y:b,x:r],visible_alpha,receiver)
            photo['patch_boundary']['exported']=audit
            checks['patch_boundary']={'status':'pass' if audit['pass'] else 'fail',
                                      'evidence':audit,'basis':'exported_multiband_receiver_and_crop_boundary'}
            checks['shadow_pair']=render_meta.get('shadow_pair_validation',{'status':'unknown','basis':'legacy_metadata_not_available'})
    if candidate['scale'].get('fixed_physical_size'):
        checks['projection_size']={'status':'pass' if abs(unoccluded_major-expected)<=tolerance else 'fail',
                                  'measurement_px':unoccluded_major,'expected_px':expected,'tolerance_px':tolerance}
        checks['scale']['status']=candidate['scale']['status']
        checks['scale']['basis']='scene_scale_evidence_separate_from_projected_pixel_agreement'
    if params.get('review_profile') == 'photographer':
        from scrap_composite.photography_measurements import measurements
        photo['review_measurements'] = measurements(
            photo, object_pixels=exported[y:b,x:r][visible_alpha>.95], reference_pixels=reference_pixels,
            visible_shadow_pixels=int(((loss>.025)&(visible_alpha<.1)).sum()),
            lighting=environment.get('local_irradiance', {}).get('direction'),
            specular_share=highlight[visible_alpha>.95],
        ).model_dump(mode='json', exclude_none=True)
    crop_image=Image.fromarray(base_rgb).crop((x,y,r,b));crop_image.thumbnail((1000,800));crop_image.save(Path(out).with_suffix('.crop.jpg'),quality=92)
    thumb=Image.fromarray(base_rgb);thumb.thumbnail((1536,1024));thumb.save(Path(out).with_suffix('.full.jpg'),quality=88)
    if candidate.get('placement'):
        overlay=np.asarray(scene['frame'])[y:b,x:r].copy()
        overlay[support]=np.uint8(overlay[support]*.6+np.array([20,210,120])*.4)
        overlay[foreground]=np.uint8(overlay[foreground]*.5+np.array([65,140,255])*.5)
        overlay[foot>0]=[255,170,40]
        Image.fromarray(overlay).save(Path(out).with_suffix('.occlusion.png'))
    if local:
        l,t,rr,bb=environment['local_context']['bounds']
        context_image=Image.fromarray(base_rgb).crop((l,t,rr,bb));context_image.thumbnail((1200,900))
        context_image.save(Path(out).with_suffix('.context.jpg'),quality=94)
        overlay=np.asarray(scene['frame'])[t:bb,l:rr].copy()
        mask=samples[t:bb,l:rr]
        overlay[mask]=np.uint8(overlay[mask]*.45+np.array([30,210,140])*.55)
        Image.fromarray(overlay).save(Path(out).with_suffix('.samples.png'))
    placement=candidate.get('placement','surface')
    if native:
        # Report the observed final coverage after local settling, not the rough shortlist label.
        placement='embedded' if .08<=occlusion_fraction<=.40 else ('combined' if len(candidate.get('support_ids',[]))>1 else 'surface')
    result = {'checks':checks,'photo':photo,'measured_major_px':major,'edit_bounds':bounds,
            'placement':placement,'support_ids':candidate.get('support_ids',[site['id']]),
            'evidence_confidence':candidate.get('evidence_confidence','estimated'),
            'uncertainties':candidate.get('uncertainties',[]),'depth_order':order,'shadow_basis':shadow_basis,
            'alpha_hash':hashlib.sha256(alpha.tobytes()).hexdigest(),
            'visible_alpha_hash':hashlib.sha256(visible_alpha.tobytes()).hexdigest(),
            'foreground_mask_hash':hashlib.sha256(np.packbits(foreground)).hexdigest(),
            'shadow_hash':hashlib.sha256(loss.tobytes()).hexdigest(),
            'scene_basis':'estimated','alpha_pixels':int((visible_alpha>.5).sum()),
            'occlusion_fraction':occlusion_fraction,
            'image':str(out),'crop':str(Path(out).with_suffix('.crop.jpg')),
            'preview':str(Path(out).with_suffix('.full.jpg'))}
    if capture_local:
        result['local_layers']={'clear_rgb':clear_rgb,'rgb':rgb,'alpha':visible_alpha,'shaded':shaded,
                                'background':linear(crop),'writable':writable,'air':local_air}
    if photoshop_layers and all(c['status']!='fail' for c in checks.values()):
        from scrap_composite.layers import save_bundle
        try:
            proof=save_bundle(layer_output,scene,out,bounds,photoshop_layers['plate'],photoshop_layers['optical_alpha'],
                              alpha,foreground,checks)
            result['composition_manifest']=str(proof)
        except ValueError as error:
            result['photoshop_layer_error']=str(error)
    return result
