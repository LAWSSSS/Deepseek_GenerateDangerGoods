"""Scene-linear exposure, one OCIO display transform, and bounded sensor matching."""
from functools import lru_cache
import math
import cv2
import numpy as np

from .common import BLENDER
from .schema import TONE_PROFILES, PHOTO_CONTROLS

VERSION='scrap-gray-linear-v4-materials'
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


def match_linear(raw,alpha,pixels,specular_weight,params,seed,environment,metallic=None,upward=None):
    classes=set(environment.get('material_classes',[]))
    polymer=bool(classes) and classes.issubset({'plastic','rubber'})
    core=alpha>.85
    if core.sum()<20:raise ValueError('object_too_small')
    highlight=np.clip(specular_weight,0,1)
    surface=core&(highlight<.75)
    if surface.sum()<20:surface=core
    initial=display(raw)
    before=float(np.median((initial@LUMA)[surface]))
    target=float(np.median(pixels@LUMA)/255)
    median=np.median(pixels,axis=0)
    cast=np.clip(median/max(1.,median.mean()),.87,1.13)
    extra_ev,temperature=TONE_PROFILES[params.get('tone','scene')]
    cast*=np.array([1+temperature*.008,1,1-temperature*.008])
    # Solve exposure in radiance, evaluating but never recursively re-applying AgX.
    sample=raw[surface][::max(1,int(surface.sum()/1024))]*cast
    goal=.80*target+.20*before
    low,high=-1.5,2.
    for _ in range(9):
        middle=(low+high)/2
        if np.median(display(sample*2**middle)@LUMA)<goal:low=middle
        else:high=middle
    ev=float((low+high)/2+extra_ev)
    radiance=np.maximum(raw,0)*cast*2**ev
    haze=float(np.clip(environment.get('haze',0)*control(params,'haze'),0,.3))
    veil=linear(median/255)
    radiance=radiance*(1-haze)+veil*haze
    rgb=display(radiance)
    stats=environment.get('sensor',{})
    lab=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2LAB)
    scene_lab=cv2.cvtColor(np.float32(pixels[:,None,:]/255),cv2.COLOR_RGB2LAB)[:,0]
    scene_ab=np.median(scene_lab[:,1:],axis=0)
    source_chroma=float(np.median(np.linalg.norm(lab[:,:,1:][surface],axis=1)))
    scene_chroma=float(np.median(np.linalg.norm(scene_lab[:,1:],axis=1)))
    gray_style=params.get('photo_style','scrap_gray')=='scrap_gray'
    ratio=(scene_chroma+2)/max(source_chroma,1)
    gain=float(np.clip(ratio,.20,.55) if gray_style else np.clip(ratio,.6,1.))
    gain=float(np.clip(gain/control(params,'chroma'),.20,.55 if gray_style else 1.))
    # Low-frequency color is altered; original small-scale luminance detail survives.
    sigma=max(1.5,min(alpha.shape)*.018)
    smooth=normalized_blur(lab,alpha,sigma)
    low_l=smooth[:,:,0]
    luma_iqr=float(np.diff(np.percentile(low_l[surface],[25,75]))[0])
    scene_iqr=float(np.diff(np.percentile(scene_lab[:,0],[25,75]))[0])
    contrast=float(np.clip(scene_iqr/max(luma_iqr,1),.82 if polymer else .60,1.10))
    contrast=1+(contrast-1)*control(params,'contrast')
    med_l=float(np.median(low_l[surface]))
    protected=highlight*np.clip((lab[:,:,0]-med_l)/20,0,1)
    new_l=med_l+(low_l-med_l)*contrast+(lab[:,:,0]-low_l)
    lab[:,:,0]=lab[:,:,0]*protected+new_l*(1-protected)
    chroma_mask=1-.65*protected
    modified=lab[:,:,1:]*gain+scene_ab*(1-gain)*(.5 if gray_style else .15)
    lab[:,:,1:]=lab[:,:,1:]*(1-chroma_mask[...,None])+modified*chroma_mask[...,None]
    rgb=np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),0,1)
    # Uneven photographic dust. Shader dust uses the same scene tint when available.
    rng=np.random.default_rng(seed)
    field=cv2.resize(rng.random((9,9)).astype(np.float32),(alpha.shape[1],alpha.shape[0]),interpolation=cv2.INTER_CUBIC)
    metal=np.clip(metallic if metallic is not None else highlight,0,1)
    orientation=np.clip(upward if upward is not None else .45+.3*field,0,1)
    dust=np.clip((.16+.30*field)*(.4+.6*orientation)*(1-.45*metal)*(1-protected)*control(params,'dust'),0,.55)
    if polymer:dust*=.35
    rgb=(rgb*(1-dust[...,None])+(median/255)*dust[...,None]).astype(np.float32)
    # Bounded surface wear lives in RGB, independently of sensor noise. Never copy steel edges
    # or text from the photograph onto the object; only transfer measured spatial amplitudes.
    micro=rng.normal(0,1,alpha.shape).astype(np.float32)
    fine=cv2.GaussianBlur(micro,(0,0),.45)
    medium=cv2.GaussianBlur(micro,(0,0),1.5)
    coarse=cv2.resize(rng.normal(0,1,(18,18)).astype(np.float32),alpha.shape[::-1],interpolation=cv2.INTER_CUBIC)
    grain=fine/max(float(fine.std()),1e-5)*.65+medium/max(float(medium.std()),1e-5)*.25+coarse*.3
    amplitude=float(np.clip(stats.get('surface_detail_rms',.018)*1.8,.025,.075))
    original_detail=abs((initial@LUMA)-cv2.GaussianBlur(initial@LUMA,(0,0),1.))
    detail_protect=np.clip(original_detail/.035,0,.8)
    wear_mask=(1-.6*metal)*(1-protected)*(1-detail_protect)*(.5+.5*field)
    if polymer:wear_mask*=.35
    wear=np.clip(grain,-2.,2.)*amplitude*control(params,'dust')*wear_mask
    rgb=np.clip(rgb+wear[...,None],0,1).astype(np.float32)
    gray=rgb@LUMA
    edge_gray=cv2.GaussianBlur(gray,(0,0),1.)
    grad=cv2.magnitude(cv2.Sobel(edge_gray,cv2.CV_32F,1,0),cv2.Sobel(edge_gray,cv2.CV_32F,0,1))
    border=(alpha>.05)&(alpha<.95)
    interior=cv2.erode(core.astype(np.uint8),np.ones((3,3),np.uint8))>0
    edge=float(np.percentile(grad[interior],80)) if interior.any() else 0
    reference=float(stats.get('edge_energy',.06))
    blur_sigma=float(np.clip((edge/max(reference,.008)-1)*.20,0,.75)/control(params,'sharpness'))
    if blur_sigma>.05:
        soft=normalized_blur(rgb,alpha,blur_sigma)
        weight=(1-.7*protected)*.65
        rgb=rgb*(1-weight[...,None])+soft*weight[...,None]
    noise=float(stats.get('noise_sigma',.003))
    residual=(gray-cv2.GaussianBlur(gray,(0,0),.8))[interior]
    existing=float(np.median(abs(residual))*1.4826) if len(residual) else 0
    added=math.sqrt(max(0,noise**2-min(existing,noise)**2))
    noise_field=rng.normal(0,added,alpha.shape).astype(np.float32)
    rgb+=np.clip(noise_field,-3*noise,3*noise)[...,None]*(1-.45*protected[...,None])
    rgb=np.clip(rgb,0,1).astype(np.float32)
    final_lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    after_chroma=float(np.median(np.linalg.norm(final_lab[:,:,1:][surface],axis=1)))
    return rgb,{'version':VERSION,'color_pipeline':'scene_linear_EXR_exposure_WB_haze_then_OCIO_AgX_once',
        'exposure_ev':ev,'white_balance':cast.tolist(),'chroma_gain':gain,'contrast_gain':contrast,
        'dust_mean':float(dust[core].mean()),'haze':haze,'noise_sigma':noise,'added_noise_sigma':added,
        'blur_sigma':blur_sigma,'before_luma':before*255,'target_luma':target*255,
        'after_luma':float(np.median((rgb@LUMA)[surface])*255),'source_chroma':source_chroma,
        'scene_chroma':scene_chroma,'after_chroma':after_chroma,
        'excess_chroma_reduction':float(1-max(0,after_chroma-scene_chroma)/max(.001,source_chroma-scene_chroma)),
        'alpha_unchanged':True,'specular_protection':True,'surface_wear_rms':float(np.std(wear[core])),
        'sensor':stats,'polymer_preservation':polymer}
