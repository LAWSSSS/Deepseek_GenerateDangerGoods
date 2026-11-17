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
COHERENT_VERSION='bounded-atmosphere-v4-scoped-veil'


def local_veil_evidence(radiance,valid,coarse,haze_mask=None):
    """Distinguish a raised dark floor with structure from a flat light surface.

    Low contrast is an expected consequence of veil, so it cannot by itself veto
    the estimate. Require dark/bright structure in several separate local tiles
    and cached visual evidence instead. These are camera-veil *estimates*, not a
    material identification or an absolute atmospheric-density measurement.
    """
    dark=radiance.min(axis=2)
    scoped=np.zeros_like(valid) if haze_mask is None else np.asarray(haze_mask,bool)&valid
    if scoped.shape!=valid.shape:raise ValueError('haze_mask_must_match_local_valid_shape')
    overlap=float(scoped.sum()/max(1,valid.sum()))
    tiles=[]
    for ys in np.array_split(np.arange(valid.shape[0]),4):
        for xs in np.array_split(np.arange(valid.shape[1]),4):
            mask=scoped[np.ix_(ys,xs)]
            total=valid[np.ix_(ys,xs)].sum()
            if mask.sum()<30 or mask.sum()<total*.5:continue
            values=dark[np.ix_(ys,xs)][mask]
            lo,hi=np.percentile(values,[10,70])
            tiles.append((float(lo),float(hi-lo),float((hi-lo)/max(hi,.03))))
    structured=[row for row in tiles if row[1]>.009 and row[2]>.06]
    fraction=len(structured)/max(1,len(tiles))
    visual=(coarse.get('kind') in ('haze','veil','mixed') and
            float(coarse.get('confidence',0))>=.5 and overlap>=.5)
    supported=bool(visual and len(structured)>=3 and fraction>=.6 and
                   np.median([row[0] for row in structured])>.035)
    return {'supports_veil':supported,'visual_veil_evidence':bool(visual),
            'local_haze_overlap':overlap,'local_haze_pixels':int(scoped.sum()),
            'minimum_local_haze_overlap':.5,'visual_scope_basis':'normalized_regions_projected_into_local_valid_pixels',
            'sample_tiles':len(tiles),'structured_tiles':len(structured),
            'structured_fraction':float(fraction),
            'tile_dark_floor_linear':float(np.median([row[0] for row in structured])) if structured else None,
            'basis':'bounded_same_layer_distributed_dark_structure_inside_visual_haze_regions'}


def local_airlight(scene,bounds,site,coarse,fallback):
    """Estimate scatter colour from several neutral bright surfaces near the site.

    A small contact patch can be entirely dark steel. Its mean is a surface
    reflectance measurement, not an airlight measurement. Search a bounded 2x
    envelope, retaining same-layer and protection exclusions, and require three
    independent tiles before replacing the contact-neighbourhood fallback.
    """
    l,t,r,b=bounds;w,h=scene['frame'].size
    half_w=min(r-l,w*.22);half_h=min(b-t,h*.22)
    cx,cy=(l+r)*.5,(t+b)*.5
    area=[max(0,int(cx-half_w)),max(0,int(cy-half_h)),min(w,int(cx+half_w)),min(h,int(cy+half_h))]
    x,y,rr,bb=area
    rgb=np.float32(np.asarray(scene['frame'])[y:bb,x:rr])/255
    factor=min(1.,320/max(rgb.shape[:2]));size=(max(8,round(rgb.shape[1]*factor)),max(8,round(rgb.shape[0]*factor)))
    rgb=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    depth=cv2.resize(scene['depth_native'][y:bb,x:rr],size)
    valid=(scene['allowed']&~scene['protected'])[y:bb,x:rr]
    if coarse.get('exclude_regions'):
        valid=valid&~regions_mask(coarse['exclude_regions'],scene['allowed'].shape)[y:bb,x:rr]
    valid=cv2.resize(valid.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    gray=rgb@LUMA
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    spread=(rgb.max(axis=2)-rgb.min(axis=2))/np.maximum(gray,.1)
    valid&=(abs(depth-site['depth'])<=.085)&(rgb.max(axis=2)<.90)&(gray>.08)&(spread<.16)
    valid&=~((gray>.68)&(gradient<.02))
    radiance=linear(rgb);tiles=[]
    for ys in np.array_split(np.arange(valid.shape[0]),4):
        for xs in np.array_split(np.arange(valid.shape[1]),4):
            mask=valid[np.ix_(ys,xs)]
            if mask.sum()<50:continue
            colors=radiance[np.ix_(ys,xs)][mask];lum=colors@LUMA
            lo,hi=np.percentile(lum,[65,90])
            select=colors[(lum>=lo)&(lum<=hi)]
            if len(select)>=10:tiles.append(np.mean(select,axis=0))
    info={'bounds':area,'sample_pixels':int(valid.sum()),'independent_tiles':len(tiles),
          'basis':'bounded_same_layer_neutral_bright_tiles','confidence':'unknown',
          'fallback_airlight_camera_linear':np.asarray(fallback).tolist()}
    if len(tiles)<3:return fallback,info
    colors=np.float32(tiles);levels=colors@LUMA
    # Upper-middle tiles permit illuminated steel, while an isolated bright
    # reflection cannot set airlight. Keep a bound when true airlight is unseen.
    threshold=min(float(np.percentile(levels,65)),float(np.sort(levels)[-3]))
    old_level=max(float(np.asarray(fallback)@LUMA),.06)
    level=float(np.clip(threshold,old_level,min(.85,old_level*1.8)))
    # The wider envelope establishes brightness only. Keep the immediate
    # same-layer neutral colour, so a nearby cool metal face cannot turn the
    # object's veil blue when the contact neighbourhood is warm grey.
    tint=np.asarray(fallback)/old_level
    result=np.float32(np.clip(level*tint,.06,.85))
    info.update(confidence='estimated',level_ratio=level/old_level,
                airlight_camera_linear=result.tolist(),metric_airlight_known=False)
    return result,info


def constrain_dark_floor(amount,radiance,valid,air,coarse,haze_mask=None):
    """A reflective patch's minimum is not evidence of atmospheric extinction.

    Bound the field by the neighbourhood's dark tail, and require an observable
    dark/bright range. This remains a single-image estimate, never air density.
    """
    values=radiance.min(axis=2)[valid]
    if len(values)<60:
        return np.minimum(amount,.08),{'confidence':'unknown','fog_upper_bound':.08,'basis':'insufficient_dark_samples'}
    dark,bright=np.percentile(values,[5,80])
    contrast=float((bright-dark)/max(bright,.03))
    informative=contrast>.28 and dark<max(float(air.min())*.55,.04)
    evidence=local_veil_evidence(radiance,valid,coarse,haze_mask)
    # I_dark = t*J_dark + (1-t)*A. A small nonzero dark reflectance
    # prevents black material from being mistaken for completely opaque haze.
    # Do not veto a structured veiled neighbourhood because its dark floor is
    # above 55% of the estimated airlight: that was capping every hazy site at 10%.
    supported=informative or evidence['supports_veil']
    upper=float(np.clip((dark-.012)/max(float(air.min())-.012,.05)*.92,0,.50))
    unscoped_upper=min(upper,.35 if informative else .10)
    if not evidence['supports_veil']:upper=unscoped_upper
    if coarse.get('kind')=='clear' and coarse.get('confidence',0)>=.65:upper=0.
    elif coarse.get('kind') in ('uncertain',None):upper=min(upper,.18)
    # The bound is computed from all same-layer valid pixels, not a selected bright
    # surface. It cannot increase the original estimate or create a spatial edge.
    bound=upper
    if evidence['supports_veil']:
        # The visual polygon is evidence, not a hard compositing mask. Smooth
        # its confidence, and do not propagate stronger veil to another part of
        # the local neighbourhood simply because it shares the same crop.
        weight=cv2.GaussianBlur(np.float32(haze_mask),(0,0),8.,borderType=cv2.BORDER_REPLICATE)
        bound=np.minimum(upper,unscoped_upper+(upper-unscoped_upper)*weight)
    return np.minimum(amount,bound),{'confidence':'estimated' if supported else 'unknown',
        'fog_upper_bound':upper,'dark_tail_linear':float(dark),'relative_local_contrast':contrast,
        'basis':'same_layer_dark_tail_bound_with_distributed_veil_evidence',
        'bright_surface_ambiguous':not supported,'local_veil_evidence':evidence}


def estimate(scene,candidate,environment):
    coherent=environment.get('integration_mode')=='coherent'
    version=COHERENT_VERSION if coherent else VERSION
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
                'version':version,'bounds':[l,t,r,b],'confidence':'unknown','sample_pixels':0,
                'basis':'no_local_neutral_evidence','depth_coefficient':0.,'depth_confidence':'unknown',
                'transmission_range':[1.,1.,1.],'depth_near':0.,'depth_far':0.}
    # Air colour is supported by neutral material agreement, not the brightest metal.
    colors=radiance[sample];lum=colors@LUMA
    bright=colors[lum>=np.percentile(lum,65)]
    air=np.mean(bright,axis=0);level=max(float(air@LUMA),.06)
    tint=np.clip(air/level,.94,1.06) if neutral.sum()>=20 else np.ones(3)
    air=np.float32(np.clip(level*tint,.06,.85))
    air_evidence={}
    if coherent:
        air,air_evidence=local_airlight(scene,[l,t,r,b],site,coarse,air)
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
    constraint={}
    if coherent:
        scoped=regions_mask(coarse.get('haze_regions') or [],scene['allowed'].shape)[t:b,l:r]
        scoped=cv2.resize(scoped.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
        amount,constraint=constrain_dark_floor(amount,radiance,valid,air,coarse,scoped)
        if constraint['confidence']=='unknown':confidence='unknown'
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
    summary={'version':version,'bounds':[l,t,r,b],'confidence':confidence,'sample_pixels':int(sample.sum()),
             'basis':'bounded_neutral_airlight_multiscale_dark_floor','depth_confidence':'estimated' if beta else 'unknown',
             'depth_coefficient':beta,'depth_near':float(near),'depth_far':float(far),'depth_correlation':correlation,
             'scale_disagreement':agreement,'airlight_camera_linear':air.tolist(),
             'transmission_range':np.percentile(fields['transmission'],[5,50,95]).tolist(),
             'surface_reflection_excluded':True,'metric_air_density_known':False,
             'airlight_evidence':air_evidence,'dark_floor_constraint':constraint}
    return fields,summary


def prepare(scene,candidate,environment,cache):
    context=environment['local_context']
    version=COHERENT_VERSION if environment.get('integration_mode')=='coherent' else VERSION
    dependencies=[version,scene['key'],context['bounds'],candidate['site_id'],
                  scene.get('atmosphere',{}).get('signature')]
    if version==COHERENT_VERSION:
        dependencies.append(scene.get('atmosphere',{}).get('summary',{}).get('visual_evidence',{}))
    key=signature(*dependencies)
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
