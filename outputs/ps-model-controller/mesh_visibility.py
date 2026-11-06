"""Compare the visible triangles of a settled mesh to its source view."""
import cv2
import numpy as np


def raster(triangles, ids, camera, size=256):
    triangles = np.asarray(triangles,dtype=np.float64)
    ids = np.asarray(ids,dtype=np.int32)
    eye = np.asarray(camera["eye"],dtype=np.float64)
    target = np.asarray(camera.get("target",[0,0,0]),dtype=np.float64)
    up = np.asarray(camera.get("up",[0,0,1]),dtype=np.float64)
    forward=target-eye
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,up)
    if np.linalg.norm(right)<1e-7:
        raise ValueError("Camera up is parallel to its direction")
    right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    delta=triangles-eye
    depth=delta@forward
    if camera.get("orthographic_scale"):
        factor=size/float(camera["orthographic_scale"])
        x=size/2+factor*(delta@right)
        y=size/2-factor*(delta@up)
    else:
        focal=.5*size/np.tan(np.deg2rad(camera.get("fov_deg",40))*.5)
        x=size/2+focal*(delta@right)/np.maximum(depth,1e-5)
        y=size/2-focal*(delta@up)/np.maximum(depth,1e-5)
    x-=size*camera.get("shift_x",0)
    y+=size*camera.get("shift_y",0)
    xy=np.stack([x,y],axis=2)
    valid=np.all(depth>.01,axis=1)&np.all(np.isfinite(xy),axis=(1,2))
    order=np.flatnonzero(valid)
    order=order[np.argsort(depth[order].mean(axis=1))[::-1]]
    labels=np.zeros((size,size),np.int32)
    for index in order:
        polygon=np.clip(np.rint(xy[index]),-32760,32760).astype(np.int32)
        cv2.fillConvexPoly(labels,polygon,int(ids[index])+1)
    return labels


def coverage(path, source_camera, posed_camera, source_alpha, posed_alpha, reference_alpha=None):
    with np.load(path) as data:
        original=raster(data["source"],data["source_ids"],source_camera,size=source_alpha.shape[0])
        posed=raster(data["posed"],data["posed_ids"],posed_camera)
    expected=[]
    for actual,alpha in ((original,source_alpha),(posed,posed_alpha)):
        observed=cv2.resize(np.asarray(alpha,dtype=np.uint8),actual.shape[::-1],interpolation=cv2.INTER_AREA)>=128
        mask=actual>0
        union=np.logical_or(observed,mask).sum()
        expected.append(float(np.logical_and(observed,mask).sum()/union) if union else 0.)
    if min(expected)<.90 or not np.any(posed):
        return {"known":False,"reason":"visibility raster does not match actual renders",
                "raster_iou":[round(v,4) for v in expected],"source_visible_fraction":None}
    source_known=original>0
    if reference_alpha is not None:
        observed=cv2.resize(np.asarray(reference_alpha,dtype=np.uint8),original.shape[::-1],
                            interpolation=cv2.INTER_AREA)>=128
        source_known &= observed
    visible=np.unique(original[source_known])
    fraction=float(np.isin(posed[posed>0],visible).mean())
    return {"known":True,"source_visible_fraction":round(fraction,4),
            "raster_iou":[round(v,4) for v in expected]}


def orbit_estimates(path, source_camera, posed_camera, source_alpha, reference_alpha):
    """Rank possible horizontal rotations; estimates alone cannot pass a pose."""
    target=np.asarray(posed_camera["target"])
    delta=np.asarray(posed_camera["eye"])-target
    with np.load(path) as data:
        original=raster(data["source"],data["source_ids"],source_camera,size=source_alpha.shape[0])
        observed=(np.asarray(source_alpha)>=128)&(np.asarray(reference_alpha)>=128)
        seen=np.unique(original[observed&(original>0)])
        triangles,ids=data["posed"],data["posed_ids"]
    for degrees in range(30,360,30):
        cosine,sine=np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))
        eye=target+np.array([cosine*delta[0]-sine*delta[1],sine*delta[0]+cosine*delta[1],delta[2]])
        posed=raster(triangles,ids,{**posed_camera,"eye":eye.tolist()})
        pixels=posed[posed>0]
        if pixels.size:
            yield degrees,float(np.isin(pixels,seen).mean())
