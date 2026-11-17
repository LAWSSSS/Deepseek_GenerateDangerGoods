"""True-mesh contact bands on independent receiver planes, without bridging gaps."""
import numpy as np
from .geometry import project


def clip(poly, normal, threshold):
    result=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        da=float(a@normal-threshold);db=float(b@normal-threshold)
        if da<=0:result.append(a)
        if (da<=0)!=(db<=0):result.append(a+(b-a)*(da/(da-db)))
    return np.asarray(result).reshape(-1,3)


def patches(points,faces,receiver,band):
    result=[];seen=set()
    for piece in receiver['patches']:
        v=np.asarray(piece['vertices'],float);face=piece['faces'][0]
        n=np.cross(v[face[1]]-v[face[0]],v[face[2]]-v[face[0]]);n/=np.linalg.norm(n)
        if n[2]<0:n=-n
        origin=v[face[0]];distance=float(origin@n)
        key=tuple(np.round(np.r_[n,distance],7))
        if key in seen:continue
        seen.add(key);heights=points@n-distance
        near=(heights[faces].min(1)<=band)&(heights[faces].max(1)>=-band)
        for f in faces[near]:
            poly=clip(points[f],n,distance+band)
            if len(poly)<3:continue
            poly=clip(poly,-n,-distance+band)
            if len(poly)<3:continue
            poly-=((poly@n)-distance)[:,None]*n
            result.append(poly)
    return result


def description(points,faces,camera,receiver,band):
    world=patches(points,faces,receiver,band)
    if not world:return [],None
    vertices=np.concatenate(world);anchor=(vertices.min(0)+vertices.max(0))/2
    return [project(p,camera) for p in world],project(anchor[None],camera)[0]
