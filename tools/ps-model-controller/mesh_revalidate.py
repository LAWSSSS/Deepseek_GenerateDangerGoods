"""Revalidate every cached pilot mesh with the corrected canonical camera."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from batch_v2 import Budget, lock_directory
from mesh_build import MODEL_REV, report
from mesh_checks import source_fidelity
from mesh_camera import fit_source_camera
from mesh_runtime import run_bounded, safe_tail
from pose_retest import OPEN_PILOT, assets
from reconstruct_worker import normalized_input
from run import load
from scale_trial import Scene
from scene_v2 import atomic_json, digest, key

HERE=Path(__file__).resolve().parent
VERSION="mesh-calibrated-camera-v3"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("before","scene","out","previous-build","open-from","reposition-from","cache","blender"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--catalog",type=Path,default=HERE/"asset-catalog.json")
    parser.add_argument("--budget",type=int,default=2_000_000)
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.resolve())
    if not 0<=args.budget<=2_000_000:
        parser.error("budget must be between 0 and 2000000")
    args.quiet_from=None
    args.numbers=list(OPEN_PILOT)
    args.all=False
    args.out.mkdir(parents=True,exist_ok=True)
    previous={r["asset_id"]:r for r in
              (json.loads(line) for line in (args.previous_build/"results.jsonl").read_text(encoding="utf8").splitlines())}
    scene=load(Scene,args.scene)
    with lock_directory(args.cache),lock_directory(args.out):
        budget=Budget(args.out/"model-usage.json",args.budget)
        rows=[]
        for folder,row in assets(args,scene):
            prior=previous.get(row["asset_id"])
            if not prior or prior.get("reconstruction",{}).get("model_revision")!=MODEL_REV:
                raise ValueError("Each asset must have a matching cached reconstruction")
            mesh=Path(prior["mesh"]).resolve(strict=True)
            cached_alpha=mesh.parent/"source-alpha.png"
            cached_input=mesh.parent/"model-input.png"
            with tempfile.TemporaryDirectory(prefix="mesh-input-") as temp:
                normalized_input(Path(row["rust"]),Path(temp))
                if (digest(Path(temp)/"source-alpha.png")!=digest(cached_alpha)
                        or digest(Path(temp)/"model-input.png")!=digest(cached_input)):
                    raise ValueError("Cached reconstruction does not match the current input pixels")
            row["signature"]=key(VERSION,row["signature"],digest(mesh),digest(HERE/"blender_worker.py"),
                                 digest(HERE/"mesh_checks.py"),digest(HERE/"mesh_camera.py"))
            marker=folder/"result.json"
            saved=json.loads(marker.read_text(encoding="utf8")) if marker.is_file() else None
            if saved and saved.get("signature")==row["signature"]:
                row=saved
            else:
                row["baseline"]=row["before_png"]
                row["mesh"]=mesh.as_posix()
                row["reconstruction"]={**prior["reconstruction"],"cache_hit":True,
                                       "initial_execution":prior["reconstruction"]["execution"]}
                try:
                    request={"operation":"source","mesh":str(mesh),"out":str(folder)}
                    req=folder/"source-request.json"
                    req.write_text(json.dumps(request),encoding="utf8")
                    elapsed=run_bounded([args.blender,"--background","--factory-startup","--python",
                                         HERE/"blender_worker.py","--",req],
                                        log=folder/"source-render.log",seconds=180)
                    row["reconstruction"]["source_render"]=elapsed
                    if elapsed["state"]!="ok" or not (folder/"source-view.png").is_file():
                        raise RuntimeError("原视角渲染失败: "+safe_tail(folder/"source-render.log",300))
                    with Image.open(cached_alpha) as source, Image.open(folder/"source-view.png") as rendered:
                        camera=fit_source_camera(np.asarray(source),np.asarray(rendered.getchannel("A")))
                    req.write_text(json.dumps({**request,"camera":camera}),encoding="utf8")
                    calibrated=run_bounded([args.blender,"--background","--factory-startup","--python",
                                           HERE/"blender_worker.py","--",req],
                                          log=folder/"source-calibrated.log",seconds=180)
                    row["reconstruction"]["calibrated_render"]=calibrated
                    if calibrated["state"]!="ok":
                        raise RuntimeError("标定相机渲染失败: "+safe_tail(folder/"source-calibrated.log",300))
                    with Image.open(cached_alpha) as source, Image.open(folder/"source-view.png") as rendered:
                        fidelity=source_fidelity(np.asarray(source),np.asarray(rendered.getchannel("A")))
                    if camera["predicted_iou"]-fidelity["iou"]>.04:
                        raise ValueError("标定相机的真实渲染与拟合轮廓不一致")
                    row["reconstruction"]["source_camera"]=camera
                    row["reconstruction"]["fidelity"]=fidelity
                    row["source_view"]=(folder/"source-view.png").as_posix()
                    shutil.copy2(cached_alpha,folder/"source-alpha.png")
                    if fidelity["passed"]:
                        row.update(status="awaiting_structure_review",code="structure_review_required",
                                   reason="原视角几何通过，等待视觉模型检查关键结构")
                    else:
                        row.update(status="review",code="reconstruction_failed",
                                   reason="修正坐标轴后，三维原视角仍未通过轮廓、孔洞或细小结构检查")
                except Exception as error:
                    row.update(status="review",code="reconstruction_error",
                               reason=f"{type(error).__name__}: {str(error)[:220]}")
                atomic_json(marker,row)
            rows.append(row)
            report(args.out,rows,budget)
            print(f"{row['number']:02} {row['status']} {row.get('reason','')}",flush=True)
    report(args.out,rows,budget)


if __name__=="__main__":
    main()
