"""Conservatively carve photographed transparent openings into a cached 3-D mesh."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes
import trimesh


def interior_holes(alpha):
    solid=np.asarray(alpha)>=128
    contours,hierarchy=cv2.findContours(solid.astype("uint8"),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes=np.zeros(solid.shape,np.uint8)
    count=0
    if hierarchy is not None:
        for i,contour in enumerate(contours):
            if hierarchy[0,i,3]<0 or cv2.contourArea(contour)<16:
                continue
            part=np.zeros(solid.shape,np.uint8)
            cv2.drawContours(part,contours,i,1,-1)
            part[(solid)]=0
            if part.sum()>=16:
                holes|=part
                count+=1
    return holes>0,count


def project(points,camera):
    eye=np.asarray(camera["eye"],dtype=np.float64)
    target=np.asarray(camera["target"],dtype=np.float64)
    forward=target-eye
    forward/=np.linalg.norm(forward)
    up=np.array([0.,0.,1.])
    right=np.cross(forward,up)
    right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    delta=points-eye
    depth=delta@forward
    focal=256/np.tan(np.deg2rad(camera["fov_deg"])/2)
    # Blender's image coordinates describe pixel edges; index 0 is the
    # center of the first source pixel, half a pixel inside the frame.
    u=255.5+focal*(delta@right)/np.maximum(depth,1e-6)-512*camera.get("shift_x",0)
    v=255.5-focal*(delta@up)/np.maximum(depth,1e-6)+512*camera.get("shift_y",0)
    return u,v,depth


def opening_clearance(holes):
    """Signed distance in source pixels, with zero on the opening boundary."""
    outside=ndimage.distance_transform_edt(~holes)
    inside=ndimage.distance_transform_edt(holes)
    return np.where(holes,inside-.5,.5-outside)


def cut_opening(field,u,v,depth,pitch,focal,padding_factor=.85):
    # Center-only carving leaves voxel edges protruding into narrow openings.
    # Allow a bounded rim for the projected voxel footprint, then verify the
    # result against the unchanged source-view gates in the actual render.
    near=np.maximum(depth-pitch/2,1e-6)
    if not 0<=padding_factor<=.85:
        raise ValueError('开孔外扩系数超出范围')
    half_span=focal*pitch/near*padding_factor
    padding=np.minimum(half_span,2.5)
    distance=ndimage.map_coordinates(field,[v,u],order=1,mode="constant",cval=-1e6)
    return (depth>0)&(distance>=-padding)


def adaptive_resolution(holes):
    count,_,stats,_=cv2.connectedComponentsWithStats(holes.astype("uint8"),8)
    widths=[min(int(stats[i,cv2.CC_STAT_WIDTH]),int(stats[i,cv2.CC_STAT_HEIGHT]))
            for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=16]
    smallest=min(widths,default=8)
    return 384 if smallest<=4 else (256 if smallest<=6 else 192)


def carve(mesh_path,alpha_path,texture_path,camera,out,resolution=None,padding_factor=.85):
    mesh=trimesh.load(mesh_path,force="mesh",process=False)
    if not mesh.is_watertight:
        raise ValueError("原始重建网格不闭合，无法可靠进行孔洞体积修复")
    alpha=np.asarray(Image.open(alpha_path).convert("L"))
    texture=np.asarray(Image.open(texture_path).convert("RGB"))
    if alpha.shape!=(512,512) or texture.shape!=(512,512,3):
        raise ValueError("孔洞和原纹理必须使用标定相机的 512 像素坐标")
    holes,count=interior_holes(alpha)
    if not count:
        raise ValueError("原素材没有可验证的内部孔洞")
    resolution=resolution or adaptive_resolution(holes)
    extent=float(mesh.extents.max())
    pitch=extent/resolution
    grid=mesh.voxelized(pitch).fill()
    dense=grid.matrix.copy()
    occupied=np.argwhere(dense)
    before=len(occupied)
    if before<1000 or max(dense.shape)>resolution*2:
        raise ValueError("体素化后的网格范围异常")
    field=opening_clearance(holes)
    focal=256/np.tan(np.deg2rad(camera["fov_deg"])/2)
    removed=0
    for start in range(0,before,150000):
        selected=occupied[start:start+150000]
        points=grid.indices_to_points(selected)
        u,v,depth=project(points,camera)
        chosen=selected[cut_opening(field,u,v,depth,pitch,focal,padding_factor)]
        if len(chosen):
            dense[chosen[:,0],chosen[:,1],chosen[:,2]]=False
            removed+=len(chosen)
    fraction=removed/before
    if removed==0 or fraction>.25:
        raise ValueError("孔洞修复没有有效体积，或需要删除超过 25% 的网格")
    # Do not turn a single body into disconnected fragments during repair.
    labels,n=ndimage.label(dense)
    sizes=np.bincount(labels.ravel())[1:]
    substantial=int((sizes>=max(20,before*.005)).sum())
    if substantial>max(1,len(mesh.split(only_watertight=False))):
        raise ValueError("孔洞修复使物品主体分裂")
    vertices,faces,_,_=marching_cubes(np.pad(dense.astype("uint8"),1),.5)
    vertices=grid.indices_to_points(vertices-1)
    repaired=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    trimesh.repair.fix_normals(repaired)
    if not repaired.is_watertight:
        raise ValueError("修复后网格不闭合")
    # Source-view texture is projected only from existing asset pixels. Holes
    # and backfaces are inferred; the later 80% evidence gate still applies.
    valid_color=alpha>=128
    _,nearest=ndimage.distance_transform_edt(~valid_color,return_indices=True)
    u,v,_=project(repaired.vertices,camera)
    u=np.clip(np.rint(u),0,511).astype(np.int32)
    v=np.clip(np.rint(v),0,511).astype(np.int32)
    colors=texture[nearest[0,v,u],nearest[1,v,u]]
    repaired.visual.vertex_colors=np.column_stack((colors,np.full(len(colors),255,dtype=np.uint8)))
    out.mkdir(parents=True,exist_ok=True)
    repaired.export(out/"mesh.glb")
    result={"method":"source_hole_voxel_footprint_carve","resolution":resolution,"source_holes":count,"padding_factor":padding_factor,
            "removed_voxels":removed,"removed_fraction":round(fraction,5),
            "components":substantial,"watertight":bool(repaired.is_watertight),
            "vertices":len(repaired.vertices),"faces":len(repaired.faces),
            "limitation":"hole depth and non-visible surfaces are inferred"}
    (out/"carve-result.json").write_text(json.dumps(result),encoding="utf8")
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","alpha","texture","camera","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument('--padding-factor',type=float,default=.85)
    args=parser.parse_args()
    camera=json.loads(args.camera.read_text(encoding="utf8"))
    carve(args.mesh,args.alpha,args.texture,camera,args.out,padding_factor=args.padding_factor)


if __name__=="__main__":
    main()
