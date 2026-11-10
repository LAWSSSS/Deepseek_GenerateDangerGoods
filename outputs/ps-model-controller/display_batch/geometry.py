"""One projection convention for searching, contact anchors and Blender renders."""
import math
import numpy as np


def material_detail_allowed(name, metallic_linked=False):
    """Mixed PBR atlases retain per-pixel masks even when names mention rubber."""
    import re
    excluded=bool(re.search(r'plastic|rubber|label|nameplate|gasket|seal|glass',name,re.I))
    mixed=metallic_linked and bool(re.search(r'steel|chrome|paint|bronze|metal',name,re.I))
    return not excluded or mixed


def transformed(vertices, matrix):
    return np.asarray(vertices) @ np.asarray(matrix)[:3,:3].T + np.asarray(matrix)[:3,3]


def pose_points(vertices, matrix, yaw):
    p = transformed(vertices, matrix)
    a = math.radians(yaw)
    rot = np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1.]])
    return p @ rot.T, rot


def camera_for(points, elevation):
    center = (points.min(axis=0)+points.max(axis=0))/2
    span = max(np.ptp(points, axis=0))
    a = math.radians(elevation)
    # A distant surveillance camera cropped around the object, not a close product photograph.
    # Keep the same local framing width while reducing artificial perspective exaggeration.
    distance=span*12.
    eye = center + np.array([0,-math.cos(a),math.sin(a)])*distance
    forward = (center-eye); forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0,0,1]); right /= np.linalg.norm(right)
    up = np.cross(right,forward)
    fov=math.degrees(2*math.atan(1.10*span/distance))
    return {"eye":eye.tolist(),"target":center.tolist(),"fov":fov,
            "basis":np.stack([right,up,forward]).tolist()}


def project(points, camera):
    p = (np.asarray(points)-camera['eye']) @ np.asarray(camera['basis']).T
    xy = p[:,:2]/np.maximum(p[:,2,None],1e-8)/(2*math.tan(math.radians(camera['fov']/2)))
    return np.column_stack([.5+xy[:,0],.5-xy[:,1]])


def contact_world(points, faces, band=.008, normal=(0,0,1)):
    # Clip each original triangle to the real ground band. Separate feet remain separate.
    normal=np.asarray(normal,dtype=float);normal/=np.linalg.norm(normal)
    heights=points@normal
    near = heights <= band
    world = []
    for face in faces[near[faces].any(axis=1)]:
        triangle = points[face]
        patch = []
        for first,last in zip(triangle,np.roll(triangle,-1,axis=0)):
            first_z,last_z=float(first@normal),float(last@normal)
            inside,outside = first_z<=band,last_z<=band
            if inside: patch.append(first)
            if inside != outside: patch.append(first+(last-first)*((band-first_z)/(last_z-first_z)))
        if len(patch)<3: continue
        patch = np.array(patch);patch-=(patch@normal)[:,None]*normal
        world.append(patch)
    return world


def contact_geometry(points, faces, camera, band=.008, normal=(0,0,1), patches=None):
    normal=np.asarray(normal,dtype=float);normal/=np.linalg.norm(normal)
    world=contact_world(points,faces,band,normal) if patches is None else patches
    if not world:
        return [], None
    p=np.concatenate(world)
    offsets=np.cumsum([0]+[len(q) for q in world]);uv=project(p,camera)
    polys=[uv[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    anchor=(p.min(axis=0)+p.max(axis=0))/2
    anchor-=normal*float(anchor@normal)
    return polys, project(anchor[None],camera)[0]


def align_up(normal):
    normal=np.asarray(normal,dtype=float);normal/=np.linalg.norm(normal)
    v=np.cross([0,0,1],normal);c=float(normal[2])
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    out=np.eye(4);out[:3,:3]=np.eye(3)+k+k@k/(1+c)
    return out


def terrain_matrix(matrix,yaw,normal):
    a=math.radians(yaw);rot=np.eye(4)
    rot[:3,:3]=[[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]]
    return align_up(normal)@rot@np.asarray(matrix)


def raster(polygons, side, scale=1., offset=(0,0)):
    import cv2
    out=np.zeros((side,side),np.uint8)
    for p in polygons:
        p=np.asarray(p)*scale+offset
        cv2.fillConvexPoly(out,np.rint(p*side*16).astype(np.int32),255,shift=4)
    return out


def horizontal_angle(axis, matrix):
    v=np.asarray(matrix)[:3,:3] @ np.asarray(axis)
    return math.degrees(math.asin(min(1.,abs(v[2])/max(1e-9,np.linalg.norm(v)))))


def contact_support(footprint, support):
    import cv2
    n, labels = cv2.connectedComponents((footprint>0).astype(np.uint8))
    return [float(support[labels==i].mean()) for i in range(1,n)
            if np.count_nonzero(labels==i)>=2]


def scale_from_reference(surface, size_mm, width, height, depth, anchor):
    a,b=np.array(surface['reference'])*[width-1,height-1]
    if depth is not None:
        h,w=depth.shape
        coords=np.array([a,b,anchor])*[(w-1)/(width-1),(h-1)/(height-1)]
        values=[float(depth[round(y),round(x)]) for x,y in coords]
        if max(abs(v-values[2]) for v in values[:2])>.085:
            raise ValueError('cross_depth_reference')
    lo,hi=surface['reference_mm']
    pixels=float(np.linalg.norm(a-b))
    return [pixels*size_mm[0]/hi,pixels*size_mm[1]/lo]
