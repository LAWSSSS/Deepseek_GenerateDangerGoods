"""Prepare source-verified higher-detail mesh and material candidates."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh

from mesh_carve import interior_holes, project
from mesh_checks import source_fidelity
from mesh_optimize import simplify
from mesh_runtime import run_bounded


def smooth(mesh, alpha, camera, resolution):
    """Limit smoothing to interior vertices, preserving holes and outer extrema."""
    before=np.asarray(mesh.vertices).copy()
    distance=cv2.distanceTransform((alpha>=128).astype("uint8"),cv2.DIST_L2,5)
    u,v,depth=project(before,camera)
    x=np.clip(np.rint(u).astype(int),0,511)
    y=np.clip(np.rint(v).astype(int),0,511)
    movable=(distance[y,x]>6)&(depth>0)
    for axis in range(3):
        low,high=before[:,axis].min(),before[:,axis].max()
        movable &= (before[:,axis]>low+.015*(high-low)) & (before[:,axis]<high-.015*(high-low))
    laplacian=trimesh.smoothing.laplacian_calculation(mesh,equal_weight=True)
    points=before.copy()
    limit=float(mesh.extents.max())/resolution*.5
    for weight in (.25,-.26,.25,-.26):
        delta=np.asarray(laplacian@points)-points
        points[movable]+=weight*delta[movable]
        diff=points-before
        length=np.linalg.norm(diff,axis=1)
        points=before+diff*np.minimum(1,limit/np.maximum(length,1e-12))[:,None]
    result=mesh.copy()
    result.vertices=points
    return result,{"max_displacement":float(np.linalg.norm(points-before,axis=1).max()),
                   "displacement_limit":limit,"protected_vertices":int((~movable).sum())}


def project_texture(mesh,asset,camera,*,texture_size=2048,detail_strength=.12,detail_gain=.65):
    image=Image.open(asset).convert("RGBA")
    image=image.crop(image.getchannel("A").getbbox())
    if texture_size not in (1024,2048) or not 0<=detail_strength<=.2 or not 0<=detail_gain<=1:
        raise ValueError("不支持的纹理分辨率或细节强度")
    # Match the 435/512 framing used for the reconstruction even when the
    # source must be upsampled, so the calibrated camera still maps to it.
    extent=texture_size*435/512
    factor=extent/max(image.size)
    image=image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))),Image.Resampling.LANCZOS)
    canvas=Image.new("RGBA",(texture_size,texture_size),(0,0,0,0))
    canvas.alpha_composite(image,((texture_size-image.width)//2,(texture_size-image.height)//2))
    rgba=np.asarray(canvas)
    valid=rgba[:,:,3]>=128
    _,nearest=ndimage.distance_transform_edt(~valid,return_indices=True)
    rgb=rgba[nearest[0],nearest[1],:3]
    u,v,_=project(np.asarray(mesh.vertices),camera)
    uv=np.column_stack((np.clip((u+.5)/512,0,1),1-np.clip((v+.5)/512,0,1)))
    holes,_=interior_holes(rgba[:,:,3])
    count,labels,stats,_=cv2.connectedComponentsWithStats(holes.astype("uint8"),8)
    if count<2:
        raise ValueError("素材缺少可确认的内部孔洞")
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    radius=max(4,round(min(stats[index,cv2.CC_STAT_WIDTH],stats[index,cv2.CC_STAT_HEIGHT])*.45))
    distance=ndimage.distance_transform_edt(labels!=index)
    ring=np.clip((radius-distance)/max(2,radius*.15),0,1)
    roughness=np.rint(230-90*ring).astype("uint8")
    metal=np.rint(50*ring).astype("uint8")
    channels=np.stack((np.full_like(roughness,255),roughness,metal),axis=2)
    # Derive only a small, band-limited normal perturbation from observed
    # source luminance. Fade it out around silhouettes and openings.
    luminance=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    scale=texture_size/1024
    detail=cv2.GaussianBlur(luminance,(0,0),2*scale)-cv2.GaussianBlur(luminance,(0,0),9*scale)
    dx=cv2.Sobel(detail,cv2.CV_32F,1,0,ksize=3)/8
    dy=cv2.Sobel(detail,cv2.CV_32F,0,1,ksize=3)/8
    magnitude=np.hypot(dx,dy)
    level=max(float(np.percentile(magnitude[valid],95)),1e-6)
    edge=cv2.distanceTransform(valid.astype("uint8"),cv2.DIST_L2,5)
    interior=np.clip((edge-2*scale)/(4*scale),0,1)
    weight=interior*detail_strength
    nx=-np.clip(dx/level,-1,1)*weight
    ny=np.clip(dy/level,-1,1)*weight
    nz=np.sqrt(np.maximum(0,1-nx*nx-ny*ny))
    normal=np.rint(np.clip(np.stack((nx,ny,nz),axis=2)*.5+.5,0,1)*255).astype("uint8")
    # Restore a bounded amount of the source's medium-frequency detail.  A
    # luminance-only adjustment preserves its hue and avoids creating texture
    # at transparent edges or on unobserved background pixels.
    band=cv2.GaussianBlur(luminance,(0,0),4*scale)-cv2.GaussianBlur(luminance,(0,0),16*scale)
    delta=np.clip(band*detail_gain,-6/255,6/255)*interior
    enhanced=np.rint(np.clip(rgb.astype(np.float32)/255+delta[:,:,None],0,1)*255).astype("uint8")
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(enhanced),
               metallicRoughnessTexture=Image.fromarray(channels),normalTexture=Image.fromarray(normal),
               metallicFactor=1.,roughnessFactor=1.)
    result=mesh.copy()
    result.visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
    return result


def render_source(mesh,folder,camera,blender,device):
    folder.mkdir(parents=True,exist_ok=True)
    mesh_path=folder/"mesh.glb"
    mesh.export(mesh_path)
    request=folder/"request.json"
    request.write_text(json.dumps({"operation":"source","mesh":str(mesh_path),"out":str(folder),
                                   "render_device":device,"camera":camera}),encoding="utf8")
    execution=run_bounded([blender,"--background","--factory-startup","--python-exit-code","1","--python",
                           Path(__file__).with_name("blender_worker.py"),"--",request],
                          log=folder/"blender.log",seconds=180)
    if execution["state"]!="ok":
        raise RuntimeError(f"源视图渲染失败: {execution['state']}")
    return execution


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","alpha","camera","asset","out","blender"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--voxel-resolution",type=int,default=384)
    parser.add_argument("--texture-size",type=int,choices=(1024,2048),default=2048)
    parser.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="CPU")
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    camera=json.loads(args.camera.read_text(encoding="utf-8-sig"))
    alpha=np.asarray(Image.open(args.alpha).convert("L"))
    original=trimesh.load(args.mesh,force="mesh",process=False)
    results=[]
    chosen=None
    for faces in (48000,96000):
        candidate=simplify(original,faces)
        if not candidate.is_watertight:
            trimesh.repair.fill_holes(candidate)
        if not candidate.is_watertight or candidate.euler_number!=original.euler_number or candidate.body_count!=original.body_count:
            results.append({"faces":faces,"status":"topology_failure"})
            continue
        _,nearest=cKDTree(np.asarray(original.vertices)).query(np.asarray(candidate.vertices),workers=1)
        candidate.visual.vertex_colors=np.asarray(original.visual.vertex_colors)[nearest]
        folder=args.out/f"mesh-{faces}"
        execution=render_source(candidate,folder,camera,args.blender,args.render_device)
        with Image.open(folder/"source-view.png") as image:
            fidelity=source_fidelity(alpha,np.asarray(image.getchannel("A")))
        results.append({"faces":faces,"fidelity":fidelity,"execution":execution})
        if fidelity["passed"]:
            chosen=candidate
    if chosen is None:
        raise ValueError("高细节网格均未通过源图检查")
    refined,details=smooth(chosen,alpha,camera,args.voxel_resolution)
    folder=args.out/"mesh-smooth"
    render_source(refined,folder,camera,args.blender,args.render_device)
    with Image.open(folder/"source-view.png") as image:
        fidelity=source_fidelity(alpha,np.asarray(image.getchannel("A")))
    if not fidelity["passed"] or not refined.is_watertight or refined.euler_number!=original.euler_number:
        raise ValueError("平滑网格未通过孔洞或拓扑检查")
    # GLB stores coordinates as float32.  Compare the textured file against
    # the actual exported geometry that the physics worker will load.
    refined=trimesh.load(folder/"mesh.glb",force="mesh",process=False)
    textured=project_texture(refined,args.asset,camera,texture_size=args.texture_size)
    folder=args.out/"mesh-texture"
    render_source(textured,folder,camera,args.blender,args.render_device)
    saved=trimesh.load(folder/"mesh.glb",force="mesh",process=False)
    if not np.array_equal(refined.vertices,saved.vertices) or not np.array_equal(refined.faces,saved.faces):
        raise ValueError("材质导出改变了几何")
    with Image.open(folder/"source-view.png") as image:
        fidelity=source_fidelity(alpha,np.asarray(image.getchannel("A")))
    if not fidelity["passed"]:
        raise ValueError("材质网格未通过源图检查")
    result={"candidates":results,"smoothing":details,"fidelity":fidelity,"mesh":str(folder/"mesh.glb"),
            "source_view":str(folder/"source-view.png"),"texture_size":args.texture_size,
            "unobserved_surface":"projected texture is inferred"}
    (args.out/"detail-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
