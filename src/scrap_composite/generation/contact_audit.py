"""Triangle/edge contact witnesses and area/line support, usable inside Blender."""
import math
import numpy as np
from .contact_geometry import clip


def contact_witnesses(points, faces, receiver_vertices, receiver_faces, band):
    """Clip real triangles near each receiver plane; include edge and face interiors."""
    samples = [points]
    planes = set()
    for face in receiver_faces:
        tri = receiver_vertices[face[:3]]
        n = np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(n)
        if length < 1e-12:continue
        n /= length
        if n[2] < .15:continue
        d = float(tri[0]@n);key=tuple(np.round(np.r_[n,d],9))
        if key in planes:continue
        planes.add(key)
        heights=points@n-d
        near=(heights[faces].min(1)<=band)&(heights[faces].max(1)>=-band)
        for f in faces[near]:
            poly=clip(points[f],n,d+band)
            if len(poly)<2:continue
            poly=clip(poly,-n,-d+band)
            if len(poly)<2:continue
            samples.extend((poly,(poly+np.roll(poly,-1,axis=0))*.5,poly.mean(0)[None]))
    return np.unique(np.round(np.concatenate(samples),10),axis=0)


def hull2(points):
    p=sorted(set(map(tuple,np.asarray(points,float))))
    if len(p)<=2:return np.asarray(p).reshape(-1,2)
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for v in p:
        while len(lower)>=2 and cross(lower[-2],lower[-1],v)<=0:lower.pop()
        lower.append(v)
    for v in reversed(p):
        while len(upper)>=2 and cross(upper[-2],upper[-1],v)<=0:upper.pop()
        upper.append(v)
    return np.asarray(lower[:-1]+upper[:-1])


def support(contacts, center, points, tolerance, motion, quiet=True):
    if not len(contacts):return {'supported':False,'kind':'none'}
    xy=np.asarray(contacts)[:,:2];hull=hull2(xy)
    middle=xy.mean(0);_,s,axes=np.linalg.svd(xy-middle,full_matrices=False)
    along=axes[0] if len(axes) else np.array([1.,0.])
    across=np.array([-along[1],along[0]])
    width=float(np.ptp((xy-middle)@across));span=float(np.ptp((xy-middle)@along))
    if width>2*tolerance and len(hull)>=3:
        edge=np.roll(hull,-1,axis=0)-hull
        distance=(edge[:,0]*(center[1]-hull[:,1])-edge[:,1]*(center[0]-hull[:,0]))/np.maximum(np.linalg.norm(edge,axis=1),1e-15)
        return {'supported':bool(np.all(distance>=-tolerance) or np.all(distance<=tolerance)),
                'kind':'area','minimum_com_margin_m':float(distance.min()),'width_m':width}
    if span<=2*tolerance:return {'supported':False,'kind':'point_unverified'}
    axial=(xy-middle)@along;relative=np.asarray(center[:2])-middle
    inside=abs(relative@across)<=tolerance and axial.min()-tolerance<=relative@along<=axial.max()+tolerance
    # A cylinder's line support is not an area polygon. Test gravity potential
    # under tiny rolls, without modifying the evaluated transform.
    origin=np.r_[middle,np.asarray(contacts)[:,2].mean()];axis=np.r_[along,0.]
    changes=[]
    for angle in (-2.,2.):
        a=math.radians(angle);k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        r=np.eye(3)+math.sin(a)*k+(1-math.cos(a))*(k@k)
        rotated=(points-origin)@r.T+origin;com=(center-origin)@r.T+origin
        changes.append(float(com[2]+origin[2]-rotated[:,2].min()-center[2]))
    supported=bool(inside and quiet and min(changes)>=-motion)
    return {'supported':supported,'kind':'line','width_m':width,'roll_probe_deg':2.,
            'roll_potential_changes_m':changes,'com_over_segment':bool(inside)}
