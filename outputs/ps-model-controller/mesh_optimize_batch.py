"""Optimize every locally valid pilot mesh, then re-render its source view."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from batch_v2 import Budget, lock_directory
from mesh_build import report
from mesh_checks import source_fidelity
from mesh_runtime import run_bounded, safe_tail
from pose_retest import OPEN_PILOT
from scene_v2 import atomic_json, digest, key

HERE=Path(__file__).resolve().parent
VERSION="mesh-actual-simplification-v1"


def process(row,folder,args):
    if not row.get("reconstruction",{}).get("fidelity",{}).get("passed"):
        row.update(status="review",code="reconstruction_failed",reason="原网格未通过结构门槛，不进行减面")
        return
    mesh=Path(row["mesh"]).resolve(strict=True)
    source_view=Path(row["source_view"]).resolve(strict=True)
    source_alpha=source_view.parent/"source-alpha.png"
    camera=row["reconstruction"]["source_camera"]
    signature=key(VERSION,digest(mesh),digest(source_alpha),digest(source_view),camera,
                  digest(HERE/"mesh_optimize.py"),digest(HERE/"mesh_visibility.py"),
                  digest(HERE/"blender_worker.py"))
    cache=args.cache/f"mesh-optimized-{signature}"
    marker=cache/"optimization.json"
    if marker.is_file():
        info=json.loads(marker.read_text(encoding="utf8"))
        info["cache_hit"]=True
    else:
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="optimize-",dir=cache) as temporary:
            stage=Path(temporary)
            camera_file=stage/"camera.json"
            camera_file.write_text(json.dumps(camera),encoding="utf8")
            execution=run_bounded([args.python,HERE/"mesh_optimize.py","--mesh",mesh,"--alpha",source_alpha,
                                   "--source-view",source_view,"--camera",camera_file,"--out",stage],
                                  log=folder/"optimize.log",seconds=300)
            info={"signature":signature,"cache_hit":False,"execution":execution}
            if execution["state"]!="ok" or not (stage/"mesh.glb").is_file():
                info["error"]=safe_tail(folder/"optimize.log")
                row.update(status="review",code="optimization_failed",reason="网格减面无法保留原图结构",optimization=info)
                return
            req=stage/"request.json"
            req.write_text(json.dumps({"operation":"source","mesh":str(stage/"mesh.glb"),
                                       "out":str(stage),"camera":camera}),encoding="utf8")
            result=run_bounded([args.blender,"--background","--factory-startup","--python",
                                HERE/"blender_worker.py","--",req],log=folder/"source-render.log",seconds=180)
            info["source_render"]=result
            if result["state"]!="ok" or not (stage/"source-view.png").is_file():
                row.update(status="review",code="optimization_failed",reason="减面后的真实渲染失败",optimization=info)
                return
            with Image.open(source_alpha) as original,Image.open(stage/"source-view.png") as image:
                info["fidelity"]=source_fidelity(np.asarray(original),np.asarray(image.getchannel("A")))
            info["details"]=json.loads((stage/"optimize-result.json").read_text(encoding="utf8"))
            shutil.copy2(stage/"mesh.glb",cache/"mesh.glb")
            shutil.copy2(stage/"source-view.png",cache/"source-view.png")
            shutil.copy2(source_alpha,cache/"source-alpha.png")
            atomic_json(marker,info)
    row["optimization"]=info
    row["pre_optimization_mesh"]=mesh.as_posix()
    row["mesh"]=(cache/"mesh.glb").as_posix()
    row["source_view"]=(cache/"source-view.png").as_posix()
    row["reconstruction"]["fidelity"]=info["fidelity"]
    if info["fidelity"]["passed"]:
        row.update(status="awaiting_structure_review",code="structure_review_required",
                   reason="减面后的三维网格通过本地结构检查，等待视觉复核")
    else:
        row.update(status="review",code="reconstruction_failed",reason="减面后的 Blender 实际渲染丢失结构")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("build","out","cache","python","blender"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--budget",type=int,default=2_000_000)
    args=cli.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.resolve())
    if not 0<=args.budget<=2_000_000:
        cli.error("budget must be between 0 and 2000000")
    source=[json.loads(line) for line in (args.build/"results.jsonl").read_text(encoding="utf8").splitlines()]
    if {row["number"] for row in source}!=set(OPEN_PILOT):
        raise ValueError("必须同时处理 03、10、27、30 四件")
    args.out.mkdir(parents=True,exist_ok=True)
    with lock_directory(args.cache),lock_directory(args.out):
        budget=Budget(args.out/"model-usage.json",args.budget)
        rows=[]
        for old in sorted(source,key=lambda item:item["number"]):
            row=dict(old)
            folder=args.out/"items"/f"{row['number']:02}-{row['asset_id']}"
            folder.mkdir(parents=True,exist_ok=True)
            row["folder"]=folder.relative_to(args.out).as_posix()
            row["signature"]=key(VERSION,old["signature"],digest(HERE/"mesh_optimize.py"),
                                 digest(HERE/"blender_worker.py"),digest(HERE/"mesh_visibility.py"))
            row["calls"]=[]
            saved_path=folder/"result.json"
            saved=json.loads(saved_path.read_text(encoding="utf8")) if saved_path.is_file() else None
            if saved and saved.get("signature")==row["signature"]:
                row=saved
            else:
                try:
                    process(row,folder,args)
                except (ValueError,RuntimeError,OSError) as issue:
                    row.update(status="review",code="optimization_failed",reason=str(issue)[:220])
                atomic_json(saved_path,row)
            rows.append(row)
            report(args.out,rows,budget)
            print(f"{row['number']:02} {row['status']} {row.get('reason','')}",flush=True)
    report(args.out,rows,budget)


if __name__=="__main__":
    main()
