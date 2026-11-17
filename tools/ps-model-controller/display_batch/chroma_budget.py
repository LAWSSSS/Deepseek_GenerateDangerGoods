"""Local Lab chroma budgets; luminance, hue and geometric coverage are independent."""
import cv2
import numpy as np
from .common import signature

VERSION='local-chroma-quantiles-v1'


def lab_of(rgb):
    return cv2.cvtColor(np.clip(rgb,0,1).astype(np.float32),cv2.COLOR_RGB2LAB)


def masks(source_rgb,alpha,metallic,classes):
    lab=lab_of(source_rgb);chroma=np.linalg.norm(lab[...,1:],axis=2)
    visible=alpha>.95
    labels=(chroma<4)&(lab[...,0]>65)
    classes=set(classes)
    if not classes.intersection({'steel','chrome','paint','rust'}):return {}
    metal=np.zeros(alpha.shape,bool) if metallic is None else np.asarray(metallic)>.35
    # Neutral labels never contribute to the paint statistics; ambiguous plastic/rubber
    # areas are excluded rather than classified as metal from their brightness alone.
    paint=(chroma>5)&~metal if 'paint' in classes and 'plastic' not in classes else np.zeros_like(metal)
    if 'plastic' in classes and int((metal&visible).sum())<24:return {}
    if not classes.intersection({'plastic','rubber'}):
        metal |= ~paint&~labels
    return {'paint':visible&paint&~labels, 'metal':visible&metal&~paint&~labels}


def budget(pixels,confidence='estimated'):
    lab=lab_of(np.asarray(pixels,np.float32).reshape(-1,1,3)/255)[:,0]
    chroma=np.linalg.norm(lab[:,1:],axis=1)
    if len(chroma)<20:
        return {'version':VERSION,'status':'unknown','reason':'insufficient_local_color_samples'}
    p50,p90=np.percentile(chroma,[50,90])
    result={'version':VERSION,'status':'unknown' if confidence=='unknown' else 'pass',
        'environment_c50':float(p50),'environment_c90':float(p90),
        'limit_c50':float(1.5*p50+2),'limit_c90':float(1.5*p90+2),
        'sample_count':len(chroma),'rule':'1.5 * local_same_depth_quantile + 2',
        'quantization_tolerance':.25}
    result['signature']=signature(VERSION,result)
    return result


def measure(rgb,groups,limits):
    if 'limit_c50' not in limits:return {'status':'unknown','groups':{},'budget':limits}
    chroma=np.linalg.norm(lab_of(rgb)[...,1:],axis=2);rows={}
    for name,mask in groups.items():
        if mask.sum()<24:
            rows[name]={'status':'unknown','pixels':int(mask.sum()),'reason':'material_region_too_small'};continue
        p50,p90=np.percentile(chroma[mask],[50,90]);eps=limits['quantization_tolerance']
        rows[name]={'status':'pass' if p50<=limits['limit_c50']+eps and p90<=limits['limit_c90']+eps else 'fail',
                    'pixels':int(mask.sum()),'c50':float(p50),'c90':float(p90)}
    status='fail' if any(r['status']=='fail' for r in rows.values()) else limits['status']
    if groups and all(r['status']=='unknown' for r in rows.values()):status='unknown'
    return {'version':VERSION,'status':status,'groups':rows,'budget':limits,
            'basis':'separate_metal_and_colored_paint_no_white_label_dilution' if groups else 'nonmetal_no_forced_desaturation'}


def compress(rgb,groups,limits,alpha):
    result=np.asarray(rgb,np.float32).copy();before=measure(result,groups,limits)
    if 'limit_c50' not in limits:return result,before
    lab=lab_of(result);gains={};factor=np.ones(alpha.shape,np.float32)
    for name,mask in groups.items():
        row=before['groups'][name]
        if 'c50' not in row:continue
        gain=min(1.,limits['limit_c50']/max(row['c50'],1e-6),limits['limit_c90']/max(row['c90'],1e-6))
        if gain<1:gain*=.97
        gains[name]=float(gain)
        factor[mask]=gain
    active=np.logical_or.reduce(list(groups.values())) if groups else np.zeros(alpha.shape,bool)
    weights=cv2.GaussianBlur(active.astype(np.float32),(0,0),1.)
    edge=(alpha>0)&(alpha<=.95)&(weights>1e-5)
    extended=cv2.GaussianBlur(factor*active,(0,0),1.)/np.maximum(weights,1e-5)
    factor[edge]=extended[edge];active|=edge
    lab[...,1:]*=factor[...,None]
    converted=np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),0,1)
    active&=alpha>0
    # Outside the material masks use bit-identical RGB, including holes/foreground.
    result[active]=converted[active]
    after=measure(result,groups,limits);after.update(gains=gains,before=before)
    return result,after
