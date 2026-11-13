"""Scene-linear exposure, one OCIO display transform, and bounded sensor matching."""
from functools import lru_cache
import math
import cv2
import numpy as np

from .common import BLENDER
from .schema import TONE_PROFILES, PHOTO_CONTROLS

VERSION='scrap-gray-camera-v8-atmosphere'
REFERENCE_VERSION='scrap-gray-linear-v4-materials'
LUMA=np.array([.2126,.7152,.0722],np.float32)


def linear(rgb):
    rgb=np.maximum(np.asarray(rgb,np.float32),0)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


@lru_cache(maxsize=1)
def display_processor():
    import PyOpenColorIO as ocio
    config=ocio.Config.CreateFromFile(str(BLENDER.parent/'4.5/datafiles/colormanagement/config.ocio'))
    transform=ocio.DisplayViewTransform(src='scene_linear',display='sRGB',view='AgX')
    return config.getProcessor(transform).getDefaultCPUProcessor()


def display(rgb):
    import PyOpenColorIO as ocio
    data=np.ascontiguousarray(np.maximum(rgb,0),dtype=np.float32)
    shape=data.shape;data=data.reshape(-1,1,3)
    display_processor().apply(ocio.PackedImageDesc(data,1,len(data),3))
    return np.clip(data.reshape(shape),0,1)


def normalized_blur(rgb,alpha,sigma):
    weight=cv2.GaussianBlur(alpha,(0,0),sigma)
    return cv2.GaussianBlur(rgb*alpha[...,None],(0,0),sigma)/np.maximum(weight[...,None],1e-6)


def sensor_statistics(frame,mask):
    """Noise comes from flat regions; gradients describe sharpness independently of texture."""
    gray=np.asarray(frame,dtype=np.float32)@LUMA/255
    smooth=cv2.GaussianBlur(gray,(0,0),1.)
    gradient=cv2.magnitude(cv2.Sobel(smooth,cv2.CV_32F,1,0),cv2.Sobel(smooth,cv2.CV_32F,0,1))
    valid=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
    if valid.sum()<20:valid=mask
    values=gradient[valid]
    flat=valid&(gradient<=np.percentile(values,35))
    residual=(gray-smooth)[flat]
    noise=float(np.clip(np.median(abs(residual-np.median(residual)))*1.4826,.0005,.025))
    edges=valid&(gradient>=np.percentile(values,75))
    edge_energy=float(np.median(gradient[edges])) if edges.any() else .02
    mid=valid&(gradient<=np.percentile(values,85))
    texture=float(np.clip(np.median(abs(gray-cv2.GaussianBlur(gray,(0,0),2.))[mid])*1.4826,.008,.065))
    pixels=np.asarray(frame)[mask]
    labs=cv2.cvtColor(np.float32(pixels[:,None,:]/255),cv2.COLOR_RGB2LAB)[:,0]
    return {'noise_sigma':noise,'edge_energy':edge_energy,'flat_pixels':int(flat.sum()),
            'luma_iqr':float(np.diff(np.percentile(gray[valid],[25,75]))[0]),
            'chroma':float(np.median(np.linalg.norm(labs[:,1:],axis=1))),
            'lab_ab':np.median(labs[:,1:],axis=0).tolist(),'surface_detail_rms':texture}


def control(params,name):
    return PHOTO_CONTROLS[name][params.get(name,'auto')]


def match_linear(raw,alpha,pixels,specular_weight,params,seed,environment,metallic=None,upward=None,
                 *,reference=False,camera_capture=True,surface_weathering=True,fixed_tone=None):
    classes=set(environment.get('material_classes',[]))
    polymer=bool(classes) and classes.issubset({'plastic','rubber'})
    core=alpha>.85
    if core.sum()<20:raise ValueError('object_too_small')
    highlight=np.clip(specular_weight,0,1)
    surface=core&(highlight<.75)
    if surface.sum()<20:surface=core
    initial=display(raw)
    initial_luma=initial@LUMA
    if not reference:
        lo,hi=np.percentile(initial_luma[surface],[16,94])
        lit=surface&(initial_luma>=lo)&(initial_luma<=hi)
        if lit.sum()<20:lit=surface
    else:lit=surface
    before=float(np.median(initial_luma[lit]))
    target=float(np.percentile(pixels@LUMA,50 if reference else 62)/255)
    median=np.median(pixels,axis=0)
    cast=np.clip(median/max(1.,median.mean()),.87,1.13)
    extra_ev,temperature=TONE_PROFILES[params.get('tone','scene')]
    calibration=environment.get('calibration')
    if calibration:
        cast=np.asarray(calibration['white_balance'],np.float32)
    cast*=np.array([1+temperature*.008,1,1-temperature*.008])
    # Solve exposure in radiance, evaluating but never recursively re-applying AgX.
    sample=raw[lit][::max(1,int(lit.sum()/1024))]*cast
    scene_weight=.8 if reference else .70
    goal=scene_weight*target+(1-scene_weight)*before
    low,high=-1.5,2.
    if fixed_tone is not None:
        base_ev=fixed_tone['exposure_ev']-extra_ev
    elif calibration:
        base_ev=calibration['exposure_ev']
    else:
        for _ in range(9):
            middle=(low+high)/2
            if np.median(display(sample*2**middle)@LUMA)<goal:low=middle
            else:high=middle
        base_ev=(low+high)/2
    ev=float(base_ev+extra_ev if reference else np.clip(base_ev+extra_ev,-1.5,2.))
    preserve=params.get('weathering')=='preserve_source'
    requested_ev=ev
    coherent=params.get('integration_mode')=='coherent'
    if preserve:
        # Calibration is shared by every material under this light. It is not an
        # object-median brightness correction; only the extra photographic action
        # is limited to half a stop. Legacy replay keeps its original clipping.
        ev=float(base_ev+np.clip(extra_ev,-.5,.5)) if coherent and calibration else float(np.clip(ev,-.5,.5))
    exposure_error=float(np.median(display(sample*2**ev)@LUMA)-goal)
    exposure_limited=bool(abs(exposure_error)>.035 and (ev>1.99 or ev<-1.49))
    if calibration:
        exposure_error=0.
        exposure_limited=calibration['limited']
    if preserve and abs(requested_ev-ev)>1e-5:
        exposure_limited=True
        exposure_error=ev-requested_ev
    radiance=np.maximum(raw,0)*cast*2**ev
    # Local camera veil belongs after display conversion, never in material radiance.
    haze_base=0. if calibration or environment.get('local_context') else environment.get('haze',0)*(1 if reference else .42)
    haze_limit=.3 if reference else .16
    haze=float(np.clip(haze_base*control(params,'haze'),0,haze_limit))
    veil=linear(median/255)
    radiance=radiance*(1-haze)+veil*haze
    rgb=display(radiance)
    stats=environment.get('sensor',{})
    lab=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2LAB)
    scene_lab=cv2.cvtColor(np.float32(pixels[:,None,:]/255),cv2.COLOR_RGB2LAB)[:,0]
    scene_ab=np.median(scene_lab[:,1:],axis=0)
    source_chroma=float(np.median(np.linalg.norm(lab[:,:,1:][surface],axis=1)))
    scene_chroma=float(np.median(np.linalg.norm(scene_lab[:,1:],axis=1)))
    dusty=params.get('photo_style')=='dusty_scrap'
    gray_style=params.get('photo_style','scrap_gray') in ('scrap_gray','dusty_scrap')
    chroma_floor,chroma_ceiling=(.08,.35) if dusty else (.20,.55 if gray_style else 1.)
    ratio=(scene_chroma+2)/max(source_chroma,1)
    base_gain=float(np.clip(ratio*(.65 if dusty else 1.),chroma_floor,chroma_ceiling) if gray_style else np.clip(ratio,.6,1.))
    gain=float(np.clip(base_gain/control(params,'chroma'),chroma_floor,chroma_ceiling))
    if preserve:
        # Retain hue/markings; only excess colour is reduced, never force neutral metal grey.
        gain=float(np.clip((scene_chroma+6)/max(source_chroma,1),.40,1.) / control(params,'chroma'))
        gain=float(np.clip(gain,.35,1.))
        if polymer:gain=max(.75,gain)
    if fixed_tone is not None:
        gain=fixed_tone['chroma_gain']
    # Low-frequency color is altered; original small-scale luminance detail survives.
    sigma=max(1.5,min(alpha.shape)*.018)
    smooth=normalized_blur(lab,alpha,sigma)
    low_l=smooth[:,:,0]
    luma_iqr=float(np.diff(np.percentile(low_l[surface],[25,75]))[0])
    scene_iqr=float(np.diff(np.percentile(scene_lab[:,0],[25,75]))[0])
    base_contrast=float(np.clip(scene_iqr/max(luma_iqr,1),(.82 if polymer else .60) if reference else .88,1.10))
    if calibration:
        context=environment.get('local_context',{})
        if (params.get('workflow')=='evidence_v1' and context.get('confidence')=='estimated'
                and context.get('sample_pixels',0)>=64):
            # Match low-frequency spread gently. Keep texture and highlight protection below.
            base_contrast=float(np.clip(1+.45*(base_contrast-1),.86,1.04))
        else:
            base_contrast=1.
    if params.get('contrast_matching')=='local':
        context=environment.get('local_context',{})
        if context.get('sample_pixels',0)>=64:
            base_contrast=float(np.clip(scene_iqr/max(luma_iqr,1),.65,1.15))
    contrast=1+(base_contrast-1)*control(params,'contrast')
    if fixed_tone is not None:
        contrast=fixed_tone['contrast_gain']
    med_l=float(np.median(low_l[surface]))
    protected=highlight*np.clip((lab[:,:,0]-med_l)/20,0,1)
    new_l=med_l+(low_l-med_l)*contrast+(lab[:,:,0]-low_l)
    lab[:,:,0]=lab[:,:,0]*protected+new_l*(1-protected)
    metal=np.clip(metallic if metallic is not None else highlight,0,1)
    chroma_mask=1-.65*protected
    if not reference:
        excess=np.linalg.norm(lab[:,:,1:],axis=2)-scene_chroma-1.5
        chroma_mask *= np.clip(excess/7,0,1)*(1-.78*metal)*(1-.6*highlight)
    modified=lab[:,:,1:]*gain+scene_ab*(1-gain)*(.5 if gray_style else .15)
    if preserve:modified=lab[:,:,1:]*gain
    lab[:,:,1:]=lab[:,:,1:]*(1-chroma_mask[...,None])+modified*chroma_mask[...,None]
    chroma_cap=None
    if dusty and not preserve:
        # Keep luminance/texture and material reflection intact; bound residual colour
        # including coloured highlights by measured neighbouring chroma, not global tint.
        chroma_cap=float(np.clip(np.percentile(np.linalg.norm(scene_lab[:,1:],axis=1),85)*.8+2,3,10))
        magnitude=np.linalg.norm(lab[:,:,1:],axis=2)
        lab[:,:,1:]*=np.minimum(1,chroma_cap/np.maximum(magnitude,1e-4))[...,None]
    rgb=np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),0,1)
    rng=np.random.default_rng(seed)
    dust_mean=wear_rms=0.
    if reference and surface_weathering:
        # Historical photographic treatment for reproducible comparison boards.
        field=cv2.resize(rng.random((9,9)).astype(np.float32),alpha.shape[::-1],interpolation=cv2.INTER_CUBIC)
        orientation=np.clip(upward if upward is not None else .45+.3*field,0,1)
        dust=np.clip((.16+.30*field)*(.4+.6*orientation)*(1-.45*metal)*(1-protected)*control(params,'dust'),0,.55)
        if polymer:dust*=.35
        rgb=(rgb*(1-dust[...,None])+(median/255)*dust[...,None]).astype(np.float32)
        micro=rng.normal(0,1,alpha.shape).astype(np.float32)
        fine=cv2.GaussianBlur(micro,(0,0),.45)
        medium=cv2.GaussianBlur(micro,(0,0),1.5)
        coarse=cv2.resize(rng.normal(0,1,(18,18)).astype(np.float32),alpha.shape[::-1],interpolation=cv2.INTER_CUBIC)
        grain=fine/max(float(fine.std()),1e-5)*.65+medium/max(float(medium.std()),1e-5)*.25+coarse*.3
        if environment.get('local_context'):
            # Neighbouring detail bounds the supplement; never copy scene pixels as texture.
            amplitude=float(np.clip(stats.get('surface_detail_rms',.018)*.9,.006,.04))
        else:
            amplitude=float(np.clip(stats.get('surface_detail_rms',.018)*1.8,.025,.075))
        detail_protect=np.clip(abs(initial_luma-cv2.GaussianBlur(initial_luma,(0,0),1.))/.035,0,.8)
        wear_mask=(1-.6*metal)*(1-protected)*(1-detail_protect)*(.5+.5*field)
        if polymer:wear_mask*=.35
        wear=np.clip(grain,-2.,2.)*amplitude*control(params,'dust')*wear_mask
        rgb=np.clip(rgb+wear[...,None],0,1).astype(np.float32)
        dust_mean=float(dust[core].mean())
        wear_rms=float(np.std(wear[core]))
    gray=rgb@LUMA
    edge_gray=cv2.GaussianBlur(gray,(0,0),1.)
    grad=cv2.magnitude(cv2.Sobel(edge_gray,cv2.CV_32F,1,0),cv2.Sobel(edge_gray,cv2.CV_32F,0,1))
    interior=cv2.erode(core.astype(np.uint8),np.ones((3,3),np.uint8))>0
    edge=float(np.percentile(grad[interior],80)) if interior.any() else 0
    reference_edge=float(stats.get('edge_energy',.06))
    base_blur=float(np.clip((edge/max(reference_edge,.008)-1)*.20,0,.75 if reference else .46))
    blur_sigma=base_blur/control(params,'sharpness')
    if camera_capture and blur_sigma>.05:
        soft=normalized_blur(rgb,alpha,blur_sigma)
        weight=(1-.7*protected)*.65
        rgb=rgb*(1-weight[...,None])+soft*weight[...,None]
    noise=float(stats.get('noise_sigma',.003))
    flat=interior
    if not reference and interior.any():
        flat=interior&(grad<=np.percentile(grad[interior],35))
    residual=(gray-cv2.GaussianBlur(gray,(0,0),.8))[flat]
    existing=float(np.median(abs(residual))*1.4826) if len(residual) else 0
    added=math.sqrt(max(0,noise**2-min(existing,noise)**2)) if camera_capture else 0.
    noise_field=rng.normal(0,added,alpha.shape).astype(np.float32)
    rgb+=np.clip(noise_field,-3*noise,3*noise)[...,None]*(1-.45*protected[...,None])
    rgb=np.clip(rgb,0,1).astype(np.float32)
    final_lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    after_chroma=float(np.median(np.linalg.norm(final_lab[:,:,1:][surface],axis=1)))
    effective={}
    for label,value in PHOTO_CONTROLS['haze'].items():
        blur=base_blur/value
        effective[label]={'haze':float(np.clip(haze_base*value,0,haze_limit)),
            'chroma':float(np.clip(base_gain/value,chroma_floor,chroma_ceiling)) if chroma_mask[core].max()>.01 else 1.,
            'contrast':1+(base_contrast-1)*value,'sharpness':blur if blur>.05 else 0.}
    return rgb,{'version':environment.get('local_context',{}).get('version') or (REFERENCE_VERSION if reference else VERSION),
        'color_pipeline':'scene_linear_EXR_calibration_AgX_once_camera_linear_atmosphere' if calibration else 'scene_linear_EXR_exposure_WB_haze_then_OCIO_AgX_once',
        'calibration':calibration,
        'exposure_ev':ev,'calibration_ev':float(base_ev) if calibration else None,
        'photographic_adjustment_ev':float(ev-base_ev),'integration_mode':params.get('integration_mode','legacy'),
        'white_balance':cast.tolist(),'chroma_gain':gain,'contrast_gain':contrast,
        'contrast_reference':{'mode':params.get('contrast_matching','legacy'),
                              'object_low_frequency_iqr':luma_iqr,'nearby_iqr':scene_iqr},
        'exposure_limited':exposure_limited,'exposure_error':exposure_error,'effective_controls':effective,
        'dust_mean':dust_mean,'haze':haze,'noise_sigma':noise,'added_noise_sigma':added,
        'blur_sigma':blur_sigma,'before_luma':before*255,'target_luma':target*255,
        'after_luma':float(np.median((rgb@LUMA)[surface])*255),'source_chroma':source_chroma,
        'scene_chroma':scene_chroma,'after_chroma':after_chroma,'local_chroma_cap':chroma_cap,
        'excess_chroma_reduction':float(1-max(0,after_chroma-scene_chroma)/max(.001,source_chroma-scene_chroma)),
        'alpha_unchanged':True,'specular_protection':True,'surface_wear_rms':wear_rms,
        'sensor':stats,'polymer_preservation':polymer}
