"""Reduce actual mesh complexity only when source-view geometry stays intact."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh
import pymeshlab

from mesh_checks import source_fidelity
from mesh_visibility import raster


def mask_for(mesh,camera):
    triangles=np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    labels=raster(triangles,np.arange(len(triangles)),camera,size=512)
    return (labels>0).astype("uint8")*255


def simplify(mesh,count):
    collection=pymeshlab.MeshSet()
    collection.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices,dtype=np.float64),
                                      face_matrix=np.asarray(mesh.faces,dtype=np.int32)))
    collection.meshing_decimation_quadric_edge_collapse(targetfacenum=count,preservetopology=True,
                                                        preservenormal=True,preserveboundary=True,
                                                        optimalplacement=False)
    result=collection.current_mesh()
    return trimesh.Trimesh(vertices=result.vertex_matrix(),faces=result.face_matrix(),process=False)


def optimize(mesh_path,alpha_path,render_path,camera,out):
    original=trimesh.load(mesh_path,force="mesh",process=False)
    with Image.open(alpha_path) as image:
        source=np.asarray(image.convert("L"))
    with Image.open(render_path) as image:
        render=np.asarray(image.getchannel("A"))
    baseline=source_fidelity(source,render)
    if not baseline["passed"]:
        raise ValueError("Only a locally accepted mesh may enter physical optimization")
    projected_mask=mask_for(original,camera)
    projected=projected_mask>0
    raster_baseline=source_fidelity(source,projected_mask)
    actual=render>=128
    alignment=float(np.logical_and(projected,actual).sum()/np.logical_or(projected,actual).sum())
    if alignment<.90:
        raise ValueError("简化前的网格投影与 Blender 实际渲染不一致")
    attempts=[]
    chosen=None
    for count in (24000,48000,96000):
        if count>=len(original.faces):
            candidate=original.copy()
        else:
            candidate=simplify(original,count)
            if not candidate.is_watertight:
                # Repair only small open triangle/quad cracks. The checks below
                # must still preserve the original topology and visible holes.
                trimesh.repair.fill_holes(candidate)
        fidelity=source_fidelity(source,mask_for(candidate,camera))
        accepted=(candidate.is_watertight
                  and candidate.euler_number==original.euler_number
                  and candidate.body_count==original.body_count
                  and fidelity["iou"]>=raster_baseline["iou"]-.012
                  and fidelity["filled_holes"]<=raster_baseline["filled_holes"]
                  and fidelity["narrow_recall"]>=raster_baseline["narrow_recall"]-.05)
        attempts.append({"faces":len(candidate.faces),"watertight":bool(candidate.is_watertight),
                         "euler":candidate.euler_number,"components":candidate.body_count,
                         "fidelity":fidelity,"accepted":accepted})
        print(json.dumps({"baseline":baseline,"raster_baseline":raster_baseline,"original_euler":original.euler_number,
                          "original_components":original.body_count,"attempt":attempts[-1]}),flush=True)
        if accepted:
            chosen=candidate
            break
    if chosen is None:
        raise ValueError("减面会破坏来源照片中的孔洞或轮廓")
    if len(original.visual.vertex_colors)==len(original.vertices):
        _,nearest=cKDTree(np.asarray(original.vertices)).query(np.asarray(chosen.vertices),workers=1)
        chosen.visual.vertex_colors=np.asarray(original.visual.vertex_colors)[nearest]
    out.mkdir(parents=True,exist_ok=True)
    chosen.export(out/"mesh.glb")
    result={"method":"validated_quadric_mesh_simplification","original_faces":len(original.faces),
            "faces":len(chosen.faces),"projection_render_iou":round(alignment,4),
            "attempts":attempts}
    (out/"optimize-result.json").write_text(json.dumps(result),encoding="utf8")
    print(json.dumps(result),flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","alpha","source-view","camera","out"):
        cli.add_argument("--"+name,type=Path,required=True)
    args=cli.parse_args()
    optimize(args.mesh,args.alpha,args.source_view,
             json.loads(args.camera.read_text(encoding="utf8")),args.out)


if __name__=="__main__":
    main()
