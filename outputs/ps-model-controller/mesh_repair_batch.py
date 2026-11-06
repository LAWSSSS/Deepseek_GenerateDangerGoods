"""Apply one generic source-photo opening repair to every pilot mesh."""
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
VERSION="source-hole-repair-footprint-v3"


def process(row,folder,args):
    mesh=Path(row["mesh"]).resolve(strict=True)
    source_alpha=Path(row["source_view"]).parent/"source-alpha.png"
    texture=mesh.parent/"model-input.png"
    camera=row["reconstruction"]["source_camera"]
    if not source_alpha.is_file() or not texture.is_file():
        raise ValueError("重建缓存缺少原照片透明度或原纹理")
    signature=key(VERSION,digest(mesh),digest(source_alpha),digest(texture),camera,args.render_device,
                  digest(HERE/"mesh_carve.py"),digest(HERE/"blender_worker.py"),digest(HERE/"mesh_checks.py"))
    cache=args.cache/f"mesh-repair-{signature}"
    marker=cache/"repair.json"
    if marker.is_file():
        info=json.loads(marker.read_text(encoding="utf8"))
        info["cache_hit"]=True
    else:
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="repair-",dir=cache) as tmp:
            stage=Path(tmp)
            camera_path=stage/"camera.json"
            camera_path.write_text(json.dumps(camera),encoding="utf8")
            execution=run_bounded([args.python,HERE/"mesh_carve.py","--mesh",mesh,"--alpha",source_alpha,
                                   "--texture",texture,"--camera",camera_path,"--out",stage],
                                  log=folder/"repair.log",seconds=900)
            info={"cache_hit":False,"signature":signature,"execution":execution}
            if execution["state"]!="ok" or not (stage/"mesh.glb").is_file():
                info["error"]=safe_tail(folder/"repair.log")
                row.update(status="review",code="repair_failed",reason="源图孔洞约束的三维修复未完成",repair=info)
                return
            req=stage/"source-request.json"
            req.write_text(json.dumps({"operation":"source","mesh":str(stage/"mesh.glb"),
                                       "out":str(stage),"camera":camera,
                                       "render_device":args.render_device}),encoding="utf8")
            render=run_bounded([args.blender,"--background","--factory-startup","--python",
                                HERE/"blender_worker.py","--",req],
                               log=folder/"repair-render.log",seconds=180)
            info["source_render"]=render
            info["source_render"]["device"]=args.render_device
            if render["state"]!="ok" or not (stage/"source-view.png").is_file():
                info["error"]=safe_tail(folder/"repair-render.log")
                row.update(status="review",code="repair_failed",reason="修复网格的原视角渲染失败",repair=info)
                return
            with Image.open(source_alpha) as original,Image.open(stage/"source-view.png") as view:
                info["fidelity"]=source_fidelity(np.asarray(original),np.asarray(view.getchannel("A")))
            info["details"]=json.loads((stage/"carve-result.json").read_text(encoding="utf8"))
            shutil.copy2(stage/"mesh.glb",cache/"mesh.glb")
            shutil.copy2(stage/"source-view.png",cache/"source-view.png")
            shutil.copy2(source_alpha,cache/"source-alpha.png")
            shutil.copy2(texture,cache/"model-input.png")
            atomic_json(marker,info)
    row["repair"]=info
    row["original_mesh"]=mesh.as_posix()
    row["mesh"]=(cache/"mesh.glb").as_posix()
    row["source_view"]=(cache/"source-view.png").as_posix()
    row["reconstruction"]["fidelity"]=info["fidelity"]
    if info["fidelity"]["passed"]:
        row.update(status="awaiting_structure_review",code="structure_review_required",
                   reason="源图约束修复通过本地轮廓和孔洞检查，等待视觉结构核验")
    else:
        row.update(status="review",code="reconstruction_failed",reason="修复后仍有轮廓或孔洞缺陷")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("build","out","cache","python","blender"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--budget",type=int,default=2_000_000)
    cli.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="CPU")
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
            row["signature"]=key(VERSION,old["signature"],args.render_device,digest(HERE/"mesh_carve.py"),
                                 digest(HERE/"blender_worker.py"),digest(HERE/"mesh_checks.py"))
            row["calls"]=[]
            saved_path=folder/"result.json"
            saved=json.loads(saved_path.read_text(encoding="utf8")) if saved_path.is_file() else None
            if saved and saved.get("signature")==row["signature"]:
                row=saved
            else:
                try:
                    process(row,folder,args)
                except (ValueError,RuntimeError,OSError) as issue:
                    row.update(status="review",code="repair_failed",reason=str(issue)[:220])
                atomic_json(saved_path,row)
            rows.append(row)
            report(args.out,rows,budget)
            print(f"{row['number']:02} {row['status']} {row.get('reason','')}",flush=True)
    report(args.out,rows,budget)


if __name__=="__main__":
    main()
