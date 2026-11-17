"""Conservatively smooth a 3-D library while retaining evidence and material."""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import trimesh

from batch_v2 import lock_directory
from mesh_carve import project
from mesh_checks import source_fidelity
from mesh_detail import render_source
from mesh_library import write_index
from scene_v2 import atomic_json, digest, key

HERE=Path(__file__).resolve().parent
VERSION="bounded-smooth-v1"


def smooth_mesh(mesh,alpha,camera,*,fraction=.003,iterations=8):
    """Apply bounded Taubin smoothing while protecting source silhouettes."""
    if not 0<fraction<=.003 or not 1<=iterations<=8:
        raise ValueError("平滑强度超出受限范围")
    source=np.asarray(mesh.vertices,dtype=np.float64)
    if len(source)<4 or not len(mesh.faces):
        raise ValueError("网格没有足够的几何")
    distance=cv2.distanceTransform((alpha>=128).astype("uint8"),cv2.DIST_L2,5)
    u,v,depth=project(source,camera)
    px=np.clip(np.rint(u).astype(int),0,alpha.shape[1]-1)
    py=np.clip(np.rint(v).astype(int),0,alpha.shape[0]-1)
    movable=(distance[py,px]>3)&(depth>0)&(u>=0)&(u<alpha.shape[1])&(v>=0)&(v<alpha.shape[0])
    if not movable.any():
        raise ValueError("没有可在源视图中安全平滑的内部顶点")
    span=float(max(mesh.extents))
    if span<=0:
        raise ValueError("网格尺度无效")
    laplacian=trimesh.smoothing.laplacian_calculation(mesh,equal_weight=True)
    normals=np.asarray(mesh.vertex_normals)
    def roughness(vertices):
        displacement=np.asarray(laplacian@vertices)-vertices
        high=np.abs(np.einsum("ij,ij->i",displacement,normals))[movable]/span
        return [float(x) for x in np.percentile(high,[50,90,99])]
    before=roughness(source)
    vertices=source.copy()
    limit=span*fraction
    for _ in range(iterations):
        for weight in (.32,-.33):
            delta=np.asarray(laplacian@vertices)-vertices
            vertices[movable]+=weight*delta[movable]
            diff=vertices-source
            length=np.linalg.norm(diff,axis=1)
            vertices=source+diff*np.minimum(1,limit/np.maximum(length,1e-12))[:,None]
    after=roughness(vertices)
    faces=np.asarray(mesh.faces)
    old_cross=np.cross(source[faces[:,1]]-source[faces[:,0]],source[faces[:,2]]-source[faces[:,0]])
    valid=(np.linalg.norm(old_cross,axis=1)>span*span*1e-12)
    restored=np.zeros(len(vertices),dtype=bool)
    for _ in range(20):
        new_cross=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        inverted=valid & (np.einsum("ij,ij->i",old_cross,new_cross)<=0)
        if not inverted.any(): break
        indices=np.unique(faces[inverted])
        restored[indices]=True
        vertices[indices]=source[indices]
    if inverted.any():
        raise ValueError("平滑造成面翻转或退化")
    after=roughness(vertices)
    result=mesh.copy()
    result.vertices=vertices
    if (result.body_count!=mesh.body_count or result.euler_number!=mesh.euler_number or
            result.is_watertight!=mesh.is_watertight or not np.array_equal(result.faces,mesh.faces)):
        raise ValueError("平滑改变了连通性或网格拓扑")
    volume_change=None
    if mesh.is_watertight and abs(mesh.volume)>1e-12:
        volume_change=abs(result.volume-mesh.volume)/abs(mesh.volume)
        if volume_change>.02:
            raise ValueError("平滑造成过大的体积变化")
    extent_change=float(np.max(np.abs(result.extents-mesh.extents))/span)
    if extent_change>.006:
        raise ValueError("平滑造成过大的外廓变化")
    if after[1]>=before[1]*.99:
        raise ValueError("高频表面起伏没有实质降低")
    return result,{"method":"bounded_taubin_source_silhouette_protected", "iterations":iterations,
                   "max_displacement_fraction":float(np.linalg.norm(vertices-source,axis=1).max()/span),
                   "protected_vertices":int((~movable).sum()),"movable_vertices":int(movable.sum()),
                   "restored_vertices_to_prevent_inversion":int(restored.sum()),
                   "normal_laplacian_before_p50_p90_p99":before,"normal_laplacian_after_p50_p90_p99":after,
                   "extent_change_fraction":extent_change,"volume_change_fraction":volume_change,
                   "topology_preserved":True}


def material_signature(mesh):
    visual=mesh.visual
    if visual.kind=="texture":
        material=visual.material
        textures={}
        for name in ("baseColorTexture","normalTexture","metallicRoughnessTexture"):
            image=getattr(material,name,None)
            if image is not None:
                import hashlib
                textures[name]=hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()
        return {"kind":"texture","uv":np.asarray(visual.uv),"textures":textures}
    if visual.kind in ("vertex","face"):
        return {"kind":visual.kind,"colors":np.asarray(visual.vertex_colors if visual.kind=="vertex" else visual.face_colors)}
    return {"kind":visual.kind}


def same_material(a,b):
    first,second=material_signature(a),material_signature(b)
    if first.keys()!=second.keys(): return False
    for name,value in first.items():
        if isinstance(value,np.ndarray):
            if not np.allclose(value,second[name],atol=1e-6,rtol=0): return False
        elif value!=second[name]: return False
    return True


def process(row,folder,args,reference):
    source=Path(row["source_view"]); alpha_path=source.parent/"source-alpha.png"; mesh_path=Path(row["mesh"])
    if (digest(mesh_path)!=row.get("mesh_sha256") or digest(source)!=row.get("source_view_sha256") or
            digest(alpha_path)!=row.get("source_alpha_sha256") or digest(Path(row["rust"]))!=row.get("rust_sha256")):
        raise ValueError("模型或来源记录的哈希不一致")
    camera=row["reconstruction"]["source_camera"]
    with Image.open(alpha_path) as im: alpha=np.asarray(im.convert("L"))
    with Image.open(source) as im: baseline=source_fidelity(alpha,np.asarray(im.getchannel("A")))
    original=trimesh.load(mesh_path,force="mesh",process=False)
    smoothed,details=smooth_mesh(original,alpha,camera)
    execution=render_source(smoothed,folder,camera,args.blender,args.render_device)
    saved=trimesh.load(folder/"mesh.glb",force="mesh",process=False)
    if (not np.array_equal(smoothed.faces,saved.faces) or not np.allclose(smoothed.vertices,saved.vertices,atol=1e-7,rtol=0)
            or not same_material(original,saved)):
        raise ValueError("网格导出改变了面、顶点或纹理")
    with Image.open(folder/"source-view.png") as image:
        fidelity=source_fidelity(alpha,np.asarray(image.getchannel("A")))
    if (fidelity["iou"]<baseline["iou"]-.005 or fidelity["filled_holes"]>baseline["filled_holes"] or
            fidelity["narrow_recall"]<baseline["narrow_recall"]-.01 or
            baseline["passed"] and not fidelity["passed"]):
        raise ValueError("平滑后轮廓、孔洞或细结构低于原网格的保护门槛")
    shutil.copy2(alpha_path,folder/"source-alpha.png")
    row["mesh"]=str(folder/"mesh.glb")
    row["mesh_sha256"]=digest(folder/"mesh.glb")
    row["source_view"]=str(folder/"source-view.png")
    row["source_view_sha256"]=digest(folder/"source-view.png")
    row["source_alpha_sha256"]=digest(folder/"source-alpha.png")
    row["reconstruction"]={**row["reconstruction"],"fidelity":fidelity}
    row["surface_refinement"]={"state":"accepted","baseline_mesh":str(mesh_path),"baseline_fidelity":baseline,
                               "fidelity":fidelity,"details":details,"source_render":execution,
                               "texture_preserved":True,"reference":reference,
                               "limitation":"六视图未标定，仅作为外观和结构参考；本次未进行多视图重建"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,default=Path("outputs/三维模型库/results.jsonl"))
    parser.add_argument("--six-views",type=Path,default=Path.home()/"Desktop"/"六视图")
    parser.add_argument("--mapping",type=Path,default=HERE/"six-view-map.json")
    parser.add_argument("--out",type=Path,default=Path("outputs/三维模型库-圆润复验"))
    parser.add_argument("--blender",type=Path,default=Path("work/blender45/blender-4.5.3-windows-x64/blender.exe"))
    parser.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="OPTIX")
    parser.add_argument("--numbers",help="只处理逗号分隔的物品编号；省略时处理全部")
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    rows=[json.loads(s) for s in args.library.read_text(encoding="utf8").splitlines() if s.strip()]
    numbers={int(x) for x in args.numbers.split(",")} if args.numbers else {row["number"] for row in rows}
    if not numbers or not numbers<={row["number"] for row in rows}:
        parser.error("物品编号不存在")
    mapping={item["number"]:item for item in json.loads(args.mapping.read_text(encoding="utf8"))["views"]}
    args.out.mkdir(parents=True,exist_ok=True)
    with lock_directory(args.out):
        by_number={row["number"]:row for row in rows}
        existing=args.out/"results.jsonl"
        results={}
        if existing.is_file():
            for line in existing.read_text(encoding="utf8").splitlines():
                if not line.strip(): continue
                old=json.loads(line)
                if old["number"] not in by_number or old["asset_id"]!=by_number[old["number"]]["asset_id"]:
                    raise ValueError("输出目录属于其他模型清单")
                results[old["number"]]=old
        for original in sorted(rows,key=lambda x:x["number"]):
            if original["number"] not in numbers: continue
            row=json.loads(json.dumps(original))
            folder=args.out/"items"/f'{row["number"]:02}-{row["asset_id"]}'
            folder.mkdir(parents=True,exist_ok=True)
            info=mapping.get(row["number"],{"role":"missing","file":None})
            path=args.six_views/info["file"] if info.get("file") else None
            reference={**info,"path":str(path) if path else None,"sha256":digest(path) if path and path.is_file() else None,
                       "use":"uncalibrated_visual_reference"}
            signature=key(VERSION,digest(Path(row["mesh"])),row.get("source_view_sha256"),
                          row.get("source_alpha_sha256"),row["reconstruction"]["source_camera"],reference,
                          digest(HERE/"mesh_smooth_library.py"),digest(HERE/"blender_worker.py"),args.render_device)
            marker=folder/"result.json"
            saved=json.loads(marker.read_text(encoding="utf8")) if marker.is_file() else None
            intact=bool(saved and saved.get("mesh") and Path(saved["mesh"]).is_file()
                        and digest(Path(saved["mesh"]))==saved.get("mesh_sha256"))
            if intact:
                view=Path(saved.get("source_view",""));alpha=view.parent/"source-alpha.png"
                intact=bool(view.is_file() and alpha.is_file() and digest(view)==saved.get("source_view_sha256")
                            and digest(alpha)==saved.get("source_alpha_sha256"))
            if intact and saved.get("surface_refinement",{}).get("signature")==signature:
                row=saved
            else:
                start=time.monotonic()
                try:
                    process(row,folder,args,reference)
                except (ValueError,RuntimeError,OSError) as error:
                    row["surface_refinement"]={"state":"retained_original","reason":f"{type(error).__name__}: {str(error)[:250]}",
                                               "baseline_mesh":original["mesh"],"reference":reference}
                row["surface_refinement"].update(signature=signature,elapsed_seconds=round(time.monotonic()-start,2))
                atomic_json(marker,row)
            results[row["number"]]=row
            ordered=[results[n] for n in sorted(results)]
            temp=args.out/"results.tmp"
            temp.write_text("".join(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n" for x in ordered),encoding="utf8")
            temp.replace(args.out/"results.jsonl")
            write_index(args.out,ordered)
            refinement=row["surface_refinement"]
            print(f'{row["number"]:02} {refinement["state"]} {refinement.get("reason", "")}',flush=True)


if __name__=="__main__":main()
