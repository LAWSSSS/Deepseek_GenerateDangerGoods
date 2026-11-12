"""Local camera veil fields. Reflecting steel is not an airlight measurement."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from .common import signature, atomic_json
from .photography import linear, LUMA
from .atmosphere import masked_smooth, field_crop, regions_mask

VERSION='bounded-atmosphere-v1'


def estimate(scene,candidate,environment):
    l,t,r,b=environment['local_context']['bounds']
    frame=np.float32(np.asarray(scene['frame'])[t:b,l:r])/255
    factor=min(1.,256/max(frame.shape[:2]));size=(max(8,round(frame.shape[1]*factor)),max(8,round(frame.shape[0]*factor)))
    rgb=cv2.resize(frame,size,interpolation=cv2.INTER_AREA);radiance=linear(rgb)
    depth=cv2.resize(scene['depth_native'][t:b,l:r],size)
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    valid=(scene['allowed']&~scene['protected'])[t:b,l:r]
    coarse=scene.get('atmosphere',{}).get('summary',{}).get('visual_evidence',{})
    if coarse.get('exclude_regions'):
        excluded=regions_mask(coarse['exclude_regions'],scene['allowed'].shape)[t:b,l:r]
        valid=valid&~excluded
    valid=cv2.resize(valid.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    gray=rgb@LUMA
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    valid&=(abs(depth-site['depth'])<=.085)&(rgb.max(axis=2)<.93)&(gray>.045)
    valid&=~((gray>.68)&(gradient<.02))
    spread=(rgb.max(axis=2)-rgb.min(axis=2))/np.maximum(gray,.10)
    neutral=valid&(spread<.16)
    confidence='estimated' if neutral.sum()>=60 and valid.sum()>=150 else 'unknown'
    sample=neutral if neutral.sum()>=20 else valid
    if not sample.any():
        return {'transmission':np.ones(gray.shape,np.float32),'airlight':np.full((*gray.shape,3),.4,np.float32),
                'sample_mask':np.zeros(gray.shape,np.uint8),'inverse_depth':depth}, {
                'version':VERSION,'bounds':[l,t,r,b],'confidence':'unknown','sample_pixels':0,
                'basis':'no_local_neutral_evidence','depth_coefficient':0.,'depth_confidence':'unknown',
                'transmission_range':[1.,1.,1.],'depth_near':0.,'depth_far':0.}
    # Air colour is supported by neutral material agreement, not the brightest metal.
    colors=radiance[sample];lum=colors@LUMA
    bright=colors[lum>=np.percentile(lum,65)]
    air=np.mean(bright,axis=0);level=max(float(air@LUMA),.06)
    tint=np.clip(air/level,.94,1.06) if neutral.sum()>=20 else np.ones(3)
    air=np.float32(np.clip(level*tint,.06,.85))
    a_field=np.broadcast_to(air,(*gray.shape,3)).copy()
    dark=np.where(valid,radiance.min(axis=2),1.).astype(np.float32)
    small=cv2.erode(dark,np.ones((5,5),np.uint8));large=cv2.erode(dark,np.ones((13,13),np.uint8))
    agreement=float(np.median(abs(small[valid]-large[valid])))
    # A local dark floor must agree across scales; texture/white paint cannot alone add fog.
    amount=np.clip(((small+large)*.5/max(float(air.min()),.06)-.025)*.78,0,.50)
    evidence=valid&(gradient>.012)
    smooth,weight=masked_smooth(amount,evidence,8.)
    fallback=float(np.median(amount[evidence])) if evidence.any() else 0.
    amount=np.where(weight>.015,smooth,fallback)
    if coarse.get('kind')=='clear' and coarse.get('confidence',0)>=.65:amount[:]=0.
    if agreement>.08:confidence='unknown'
    if confidence=='unknown':amount=np.minimum(amount,.15)
    amount=np.float32(np.clip(amount,0,.50));tau=-np.log(np.maximum(1-amount,.5))
    near,far=np.percentile(depth[valid],[95,5]);distance=np.clip((near-depth)/max(near-far,.01),0,1)
    beta=0.;correlation=0.
    if evidence.sum()>100 and np.std(distance[evidence])>.12 and np.std(tau[evidence])>.015:
        correlation=float(np.corrcoef(distance[evidence],tau[evidence])[0,1])
        if correlation>.35 and confidence=='estimated':
            beta=float(np.clip(np.polyfit(distance[evidence],tau[evidence],1)[0],0,.3))
    screen=np.maximum(tau-beta*distance,0).astype(np.float32)
    fields={'transmission':np.exp(-(screen+beta*distance)).astype(np.float32),
            'screen_tau':screen,'airlight':a_field,'sample_mask':sample.astype(np.uint8),'inverse_depth':depth}
    summary={'version':VERSION,'bounds':[l,t,r,b],'confidence':confidence,'sample_pixels':int(sample.sum()),
             'basis':'bounded_neutral_airlight_multiscale_dark_floor','depth_confidence':'estimated' if beta else 'unknown',
             'depth_coefficient':beta,'depth_near':float(near),'depth_far':float(far),'depth_correlation':correlation,
             'scale_disagreement':agreement,'airlight_camera_linear':air.tolist(),
             'transmission_range':np.percentile(fields['transmission'],[5,50,95]).tolist(),
             'surface_reflection_excluded':True,'metric_air_density_known':False}
    return fields,summary


def prepare(scene,candidate,environment,cache):
    context=environment['local_context']
    key=signature(VERSION,scene['key'],context['bounds'],candidate['site_id'],
                  scene.get('atmosphere',{}).get('signature'))
    folder=Path(cache)/'local-atmosphere'/key;path=folder/'summary.json'
    if path.exists() and (folder/'fields.npz').exists():summary=json.loads(path.read_text(encoding='utf8'))
    else:
        fields,summary=estimate(scene,candidate,environment)
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'fields.npz',**fields);atomic_json(path,summary)
        # Full-frame diagnostics are used in the report, without cropping the scene.
        rgb=np.asarray(scene['frame']).copy();l,t,r,b=summary['bounds']
        veil=cv2.resize(1-fields['transmission'],(r-l,b-t))
        color=cv2.applyColorMap(np.uint8(veil*255),cv2.COLORMAP_TURBO)[:,:,::-1]
        rgb[t:b,l:r]=np.uint8(rgb[t:b,l:r]*.5+color*.5)
        Image.fromarray(rgb).save(folder/'full-veil.png')
    return {'folder':str(folder),'signature':key,'summary':summary}


def fields_for(context,bounds,object_z,order,candidate,scene,gain=1.,enabled=True):
    summary=context['summary'];l,t,r,b=summary['bounds'];x,y,rr,bb=bounds
    translated=[x-l,y-t,rr-l,bb-t]
    with np.load(Path(context['folder'])/'fields.npz') as data:
        fields={k:field_crop(data[k],translated,(r-l,b-t)) for k in data.files if k!='sample_mask'}
    air=fields['airlight'];background_t=fields['transmission']
    inverse=fields['inverse_depth'];beta=summary['depth_coefficient']
    depth_basis='same_layer_fallback'
    if beta>0 and order.get('confidence')=='estimated':
        inverse=np.where(object_z>0,order['scale']/np.maximum(object_z,1e-6)+order['offset'],inverse)
        depth_basis='verified_local_relative_depth_fit'
    else:
        site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
        inverse=np.full_like(inverse,site['depth'])
    distance=np.clip((summary['depth_near']-inverse)/max(.01,summary['depth_near']-summary['depth_far']),0,1)
    tau=fields.get('screen_tau',-np.log(np.maximum(background_t,.5)))+beta*distance
    object_t=np.exp(-np.clip(tau*gain,0,-np.log(.45))).astype(np.float32)
    if not enabled:object_t[:]=1.;background_t=np.ones_like(background_t)
    return object_t,background_t,air,{**summary,'signature':context['signature'],'depth_basis':depth_basis}
