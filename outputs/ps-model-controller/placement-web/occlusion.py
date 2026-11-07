"""Conservative native-pixel refinement of a suspected thin occluder."""
import cv2
import numpy as np
from PIL import Image


def thin_line_seeds(frame, center, object_size, *, limit=4):
    """Propose paired image edges in native pixels; never decide depth order."""
    rgb=np.asarray(frame.convert('RGB'))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
    edges=cv2.Canny(gray,45,130)
    minimum=max(18,round(object_size*.13))
    raw=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(15,minimum//2),
                        minLineLength=minimum,maxLineGap=max(3,round(object_size*.025)))
    if raw is None:return []
    lines=[]
    for x1,y1,x2,y2 in raw[:,0,:]:
        a=np.array([x1,y1],dtype=float);b=np.array([x2,y2],dtype=float)
        v=b-a;length=float(np.linalg.norm(v))
        if not length:continue
        v/=length
        if v[0]<0 or (v[0]==0 and v[1]<0):v=-v
        mid=(a+b)/2
        if np.linalg.norm(mid-np.asarray(center))>object_size*.85:continue
        lines.append((length,a,b,v))
    lines.sort(key=lambda x:x[0],reverse=True)
    options=[]
    for i,(_,a,b,u) in enumerate(lines[:80]):
        normal=np.array([-u[1],u[0]])
        t0,t1=sorted((float(a@u),float(b@u)))
        n0=float(((a+b)/2)@normal)
        for _,c,d,v in lines[i+1:80]:
            if abs(float(u@v))<.985:continue
            n1=float(((c+d)/2)@normal);width=abs(n1-n0)
            if not 2<=width<=max(8,min(22,object_size*.06)):continue
            s0,s1=sorted((float(c@u),float(d@u)))
            lo,hi=max(t0,s0),min(t1,s1)
            length=hi-lo
            if length<minimum or length<.55*min(t1-t0,s1-s0):continue
            poly=np.array([u*lo+normal*n0,u*hi+normal*n0,u*hi+normal*n1,u*lo+normal*n1])
            seed=np.zeros(gray.shape,np.uint8)
            cv2.fillConvexPoly(seed,np.rint(poly).astype('int32'),1)
            options.append((length/(1+width*.03),seed.astype(bool)))
    options.sort(key=lambda x:x[0],reverse=True)
    chosen=[]
    for _,mask in options:
        if any(np.count_nonzero(mask&other)/max(1,np.count_nonzero(mask|other))>.4 for other in chosen):continue
        chosen.append(mask)
        if len(chosen)>=limit:break
    return chosen


def refine_thin_mask(engine, frame, seed, rect, *, radius=8):
    """Refine within a bounded corridor; do not assert front/back order."""
    left,top,right,bottom=rect
    original=np.asarray(seed,dtype=bool)
    part=original[top:bottom,left:right]
    if not part.any():
        return original.copy(),{'status':'unknown','reason':'局部候选为空'}
    distance=cv2.distanceTransform(part.astype('uint8'),cv2.DIST_L2,5)
    if float(np.median(distance[part]))>6:
        return original.copy(),{'status':'not_thin','reason':'候选不是细长区域'}
    try:
        refined=np.asarray(engine.refine_crop(frame.crop(rect),part),dtype=bool).copy()
    except Exception as exc:
        return original.copy(),{'status':'unknown','reason':str(exc)[:160]}
    if refined.shape!=part.shape:
        return original.copy(),{'status':'unknown','reason':'细化结果尺寸不一致'}
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    allowed=cv2.dilate(part.astype('uint8'),kernel)>0
    refined &= allowed
    overlap=np.count_nonzero(refined&part)/max(1,np.count_nonzero(refined|part))
    ratio=np.count_nonzero(refined)/max(1,np.count_nonzero(part))
    if overlap<.35 or not .35<=ratio<=2.0:
        return original.copy(),{'status':'unknown','reason':'细化结果偏离原候选'}
    result=original.copy();result[top:bottom,left:right]=refined
    return result,{'status':'refined','method':'native_seeded_segmentation',
                   'seed_iou':round(float(overlap),4),'area_ratio':round(float(ratio),4),
                   'order':'unknown'}
