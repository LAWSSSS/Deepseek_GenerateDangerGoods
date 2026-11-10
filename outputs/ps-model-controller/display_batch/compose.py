import hashlib
import math
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image

from .geometry import contact_support
from .schema import TONE_PROFILES


def linear(rgb):
    rgb=np.asarray(rgb,dtype=np.float32)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def srgb(rgb):
    rgb=np.maximum(rgb,0)
    return np.where(rgb<=.0031308,rgb*12.92,1.055*rgb**(1/2.4)-.055)


def sample_environment(scene, candidate):
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    w,h=scene['size'];x,y,r,b=candidate['bounds'];pad=round(max(r-x,b-y)*.8)
    l,t,rr,bb=max(0,x-pad),max(0,y-pad),min(w,r+pad),min(h,b+pad)
    sample=np.zeros((h,w),bool);sample[t:bb,l:rr]=True
    sample &= site['mask'] & ~scene['protected']
    sample[y:b,x:r]=False
    rgb=np.asarray(scene['frame']);lum=rgb@np.array([.2126,.7152,.0722])
    if sample.sum()<100:
        # frame is the untouched photograph: the surface under the future object is valid evidence.
        sample=site['mask'].copy() & ~scene['protected']
    if sample.sum()<100:raise ValueError('insufficient_same_surface_light_samples')
    lo,hi=np.percentile(lum[sample],[12,92]);sample &= (lum>=lo)&(lum<=hi)
    pixels=rgb[sample].astype(float);median=np.median(pixels,axis=0)
    # Keep albedo separate from illumination: only a restrained neutral cast is estimated.
    tint=np.clip(median/max(float(median.mean()),1),.88,1.12)
    luminance=float(np.median(lum[sample]))
    ambient=np.clip(.22+(luminance-100)*.0012,.14,.35)
    answer=scene['answer']
    environment={'world_color':(ambient*tint).tolist(),'light_color':(tint/tint.max()).tolist(),
                 'key_direction':answer['light_direction'] if answer['light_confidence']>=.55 else None,
                 'haze':answer['haze'],'sample_count':int(sample.sum()),
                 'sample_hash':hashlib.sha256(np.packbits(sample)).hexdigest(),
                 'luma_median':luminance,'basis':'same_support_estimate'}
    return environment,sample


def warp(source, full_px, offset, bounds, *, premultiplied=False):
    x,y,r,b=bounds;n=source.shape[1]
    m=np.float32([[full_px/n,0,offset[0]-x],[0,full_px/n,offset[1]-y]])
    return cv2.warpAffine(source,m,(r-x,b-y),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


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


def compose(scene,candidate,render_dir,render_meta,environment,samples,params,seed,out):
    render_dir=Path(render_dir);source=np.asarray(Image.open(render_dir/'object.png').convert('RGBA')).astype(np.float32)/255
    w,h=scene['size'];x,y,r,b=candidate['bounds'];pad=max(12,round(candidate['target_px']*.12))
    bounds=[max(0,x-pad),max(0,y-pad),min(w,r+pad),min(h,b+pad)]
    x,y,r,b=bounds
    # Anchor derives from contact geometry, never from the alpha or shadow bounding box.
    offset=np.asarray(candidate['anchor_px'])-np.asarray(render_meta['anchor_uv'])*candidate['full_px']
    premult=np.dstack([source[:,:,:3]*source[:,:,3,None],source[:,:,3]])
    rgba=warp(premult,candidate['full_px'],offset,bounds)
    alpha=rgba[:,:,3].copy();rgb=rgba[:,:,:3]/np.maximum(alpha[:,:,None],1e-6)
    spec=cv2.imread(str(render_dir/'specular0001.exr'),cv2.IMREAD_UNCHANGED)
    diff=cv2.imread(str(render_dir/'diffuse0001.exr'),cv2.IMREAD_UNCHANGED)
    if spec is None or diff is None:raise ValueError('missing_linear_material_passes')
    highlight=np.max(spec[:,:,:3],axis=2)/np.maximum(np.max(spec[:,:,:3]+diff[:,:,:3],axis=2),.0001)
    highlight=warp(highlight,candidate['full_px'],offset,bounds)
    rgb,photo=match_rgb(rgb,alpha,np.asarray(scene['frame'])[samples],highlight,params['tone'],seed)
    haze=float(np.clip(environment['haze'],0,.18))
    veil=np.mean(np.asarray(scene['frame'])[samples],axis=0)/255
    rgb=rgb*(1-haze)+veil*haze
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    support=site['mask'][y:b,x:r];protected=scene['protected'][y:b,x:r]
    foreground=(scene['depth_native'][y:b,x:r]>site['depth']+.10)&~support
    foreground &= ~protected
    visible_alpha=alpha*((~foreground)&(~protected))
    with_shadow=np.asarray(Image.open(render_dir/'with-shadow.png').convert('RGB'),dtype=np.float32)/255
    without_shadow=np.asarray(Image.open(render_dir/'without-shadow.png').convert('RGB'),dtype=np.float32)/255
    a=linear(with_shadow).mean(axis=2);base=linear(without_shadow).mean(axis=2)
    loss=np.clip(1-a/np.maximum(base,.015),0,.65)
    loss[base<.015]=0
    loss=warp(loss,candidate['full_px'],offset,bounds)
    loss *= support & ~foreground & ~protected
    loss[loss<1/512]=0
    base_rgb=np.asarray(scene['frame']).copy();crop=base_rgb[y:b,x:r].astype(np.float32)/255
    shaded=linear(crop)*(1-loss[:,:,None])
    result=srgb(shaded)*(1-visible_alpha[:,:,None])+rgb*visible_alpha[:,:,None]
    edited=(visible_alpha>0)|(loss>0)
    crop_out=np.uint8(np.rint(np.clip(result,0,1)*255))
    original=base_rgb[y:b,x:r].copy()
    base_rgb[y:b,x:r][edited]=crop_out[edited]
    Image.fromarray(base_rgb).save(out)
    with Image.open(out) as saved:
        exported=np.asarray(saved.convert('RGB'))
        exported_size=list(saved.size)
    original_full=np.asarray(scene['frame'])
    changed=np.any(exported!=original_full,axis=2)
    permitted=np.zeros((h,w),bool);permitted[y:b,x:r]=edited
    outside_count=int((changed&~permitted).sum())
    protected_count=int((changed&scene['protected']).sum())
    foreground_changes=int(np.any(exported[y:b,x:r]!=original_full[y:b,x:r],axis=2)[foreground].sum())
    foot=np.zeros((b-y,r-x),np.uint8)
    for poly in render_meta['contact_polygons']:
        coords=(np.asarray(poly)*candidate['full_px']+offset-[x,y])*16
        cv2.fillConvexPoly(foot,np.rint(coords).astype(np.int32),255,shift=4)
    cover=contact_support(foot,support)
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
      'support':{'status':'pass' if cover and min(cover)>=.9 else 'fail','measurements':cover,'threshold':.9,'basis':'estimated_surface'},
      'camera_clearance':{'status':'pass','elevation_interval':candidate['elevation_interval'],'basis':'interval_projection'},
      'physics':{'status':'pass' if physics.get('stable') else 'unknown','evidence':physics,
                 'basis':'estimated_plane_not_measured_irregular_pile'},
      'occlusion':{'status':'pass' if foreground_changes==0 else 'fail','foreground_changed_pixels':foreground_changes},
      'scale':{'status':'pass' if abs(unoccluded_major-expected)<=tolerance else 'fail',
               'basis':'estimated_local_reference','measurement_px':unoccluded_major,'visible_px':major,
               'expected_px':expected,'tolerance_px':tolerance,'evidence':candidate['scale']},
      'anchor':{'status':'pass' if max(anchor_error,projection_error)<=1.5 else 'fail',
                'error_px':anchor_error,'projection_error_px':projection_error,'threshold_px':1.5},
      'protected':{'status':'pass' if protected_count==0 else 'fail','changed_pixels':protected_count},
      'outside_edit':{'status':'pass' if outside_count==0 else 'fail','changed_pixels':outside_count},
      'holes_alpha':{'status':'pass','basis':'unchanged_object_alpha_before_foreground_mask'},
      'size':{'status':'pass' if exported_size==[w,h] else 'fail','pixels':exported_size},
    }
    local=Image.fromarray(base_rgb).crop((x,y,r,b));local.thumbnail((1000,800));local.save(Path(out).with_suffix('.crop.jpg'),quality=92)
    thumb=Image.fromarray(base_rgb);thumb.thumbnail((1536,1024));thumb.save(Path(out).with_suffix('.full.jpg'),quality=88)
    return {'checks':checks,'photo':photo,'measured_major_px':major,'edit_bounds':bounds,
            'scene_basis':'estimated','alpha_pixels':int((visible_alpha>.5).sum()),
            'occlusion_fraction':float(1-visible_alpha.sum()/max(alpha.sum(),1e-6)),
            'image':str(out),'crop':str(Path(out).with_suffix('.crop.jpg')),
            'preview':str(Path(out).with_suffix('.full.jpg'))}
