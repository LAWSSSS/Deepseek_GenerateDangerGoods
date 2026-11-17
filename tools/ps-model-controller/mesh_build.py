"""Autonomously reconstruct the four fixed-scene pilot assets on CPU."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
from PIL import Image

from batch_v2 import Budget, lock_directory
from mesh_checks import source_fidelity
from mesh_runtime import run_bounded, safe_tail
from pose_retest import OPEN_PILOT, assets
from run import load
from scale_trial import Scene
from scene_v2 import atomic_json, digest, key

HERE = Path(__file__).resolve().parent
CHECKPOINT_SHA = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
MODEL_REV = "5b521936b01fbe1890f6f9baed0254ab6351c04a"
CODE_REV = "mesh-build-v1"


def atomic_rows(path, rows):
    pending = path.with_suffix(".tmp")
    pending.write_text("".join(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n" for row in rows),encoding="utf-8")
    pending.replace(path)


def report(out, rows, budget):
    rows = sorted(rows,key=lambda row:row["number"])
    for row in rows:
        row["token_usage"] = budget.image_usage(row)
    atomic_rows(out/"results.jsonl",rows)
    columns = ("number","image","status","api_calls","total_tokens","prompt_tokens",
               "completion_tokens","prompt_cache_hit_tokens","unreported_calls","budget_tokens",
               "reconstruction_seconds","peak_resident_bytes","repair_seconds",
               "optimization_seconds","source_render_seconds","pipeline_seconds")
    tmp = out/"token-usage.tmp"
    with tmp.open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            usage=row["token_usage"]
            execution=row.get("reconstruction",{}).get("execution",{})
            repair=row.get("repair",{}).get("execution",{})
            optimization=row.get("optimization",{}).get("execution",{})
            rendered=row.get("repair",{}).get("source_render",row.get("reconstruction",{}).get("calibrated_render",{}))
            writer.writerow([row["number"],row["file"],row["status"],
                             *[usage.get(name) for name in columns[3:10]],
                             execution.get("elapsed_seconds"),execution.get("peak_resident_bytes"),
                             repair.get("elapsed_seconds"),optimization.get("elapsed_seconds"),
                             rendered.get("elapsed_seconds"),
                             row.get("elapsed_seconds")])
    tmp.replace(out/"token-usage.csv")
    parts=["<!doctype html><meta charset='utf-8'><title>三维重建复测</title>",
           "<style>body{font-family:Arial,sans-serif;margin:2em;max-width:1100px}"
           "section{border-top:1px solid #ccc;padding:1em 0}img{max-width:480px}"
           "code{white-space:pre-wrap}</style>",
           "<h1>四件三维重建复测</h1><p>没有通过所有几何与视觉核验的样片为待审；不会以旧图冒充新合成。</p>"
           "<p><a href='token-usage.csv'>逐图 token 与耗时</a></p>"]
    for row in rows:
        parts.append(f"<section><h2>{row['number']:02} {html.escape(row['file'])}</h2>")
        parts.append(f"<p>{html.escape(row['status'])} · {html.escape(row.get('reason',''))}</p>")
        usage=row["token_usage"]
        parts.append(f"<p>DeepSeek 调用 {usage['api_calls']} · token {usage['total_tokens'] if usage['total_tokens'] is not None else '未知'}</p>")
        reconstruction=row.get("reconstruction",{})
        if reconstruction.get("fidelity"):
            parts.append("<pre>"+html.escape(json.dumps(reconstruction["fidelity"],ensure_ascii=False,indent=2))+"</pre>")
        if row.get("structure_review"):
            parts.append("<pre>结构复核 "+html.escape(json.dumps(row["structure_review"],ensure_ascii=False,indent=2))+"</pre>")
        if row.get("pose_attempts"):
            parts.append("<ul>")
            for attempt in row["pose_attempts"]:
                physics=attempt.get("physics",{})
                visibility=attempt.get("visibility",{}).get("source_visible_fraction")
                coverage="未知" if visibility is None else f"{visibility:.1%}"
                seconds=attempt["execution"].get("elapsed_seconds","?")
                status=attempt.get("reason","通过本地姿态检查")
                summary=(f"{attempt['rotation_deg']} / 方位 {attempt.get('camera_azimuth_deg',0)}° · {status} · "
                         f"承托部件 {len(physics.get('support_graph',{}).get('anchored',[]))}/"
                         f"{physics.get('component_count','?')} · 源图可见面 {coverage} · {seconds} 秒")
                parts.append("<li>"+html.escape(summary)+"</li>")
            parts.append("</ul>")
        for name,label in (("source_view","三维原视角"),("baseline","上一版全图"),
                           ("preview","新合成全图"),("psd","分层 PSD"),("comparison","前后对照")):
            path=row.get(name)
            if path and Path(path).is_file():
                rel=Path(path).relative_to(out).as_posix() if Path(path).is_relative_to(out) else Path(path).as_uri()
                parts.append(f"<p><a href='{quote(rel,safe='/:')}'>"+label+"</a></p>")
        parts.append("</section>")
    pending=out/"index.tmp"
    pending.write_text("".join(parts),encoding="utf-8")
    pending.replace(out/"index.html")


def build_item(row, folder, args):
    model_folder=args.weights/"TripoSR"
    source=Path(row["rust"])
    signature=key(CODE_REV, MODEL_REV, CHECKPOINT_SHA, digest(source), digest(Path(row["original"])),
                  digest(HERE/"reconstruct_worker.py"),digest(HERE/"blender_worker.py"),
                  digest(HERE/"mesh_checks.py"),128)
    root=args.cache/f"reconstruction-{signature}"
    ready=root/"reconstruction.json"
    if ready.is_file():
        info=json.loads(ready.read_text(encoding="utf-8"))
        info["cache_hit"]=True
    else:
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="build-",dir=root) as temporary:
            stage=Path(temporary)
            execution=run_bounded([args.python,HERE/"reconstruct_worker.py","--vendor",args.vendor,
                                   "--weights",model_folder,"--asset",source,"--out",stage,
                                   "--resolution","128"],log=folder/"reconstruct.log",seconds=900)
            info={"signature":signature,"cache_hit":False,"model_revision":MODEL_REV,
                  "resolution":128,"execution":execution}
            if execution["state"]!="ok" or not (stage/"mesh.glb").is_file():
                info["error"]=safe_tail(folder/"reconstruct.log")
                row.update(status="review",code="reconstruction_failed",
                           reason=f"三维重建未完成：{execution['state']}",reconstruction=info)
                return
            request={"operation":"source","mesh":str(stage/"mesh.glb"),"out":str(stage/"source")}
            request_path=stage/"blender-request.json"
            request_path.write_text(json.dumps(request),encoding="utf-8")
            source_execution=run_bounded([args.blender,"--background","--factory-startup","--python",
                                          HERE/"blender_worker.py","--",request_path],
                                         log=folder/"source-render.log",seconds=180)
            info["source_render"]=source_execution
            if source_execution["state"]!="ok" or not (stage/"source"/"source-view.png").is_file():
                info["error"]=safe_tail(folder/"source-render.log")
                row.update(status="review",code="reconstruction_failed",
                           reason=f"三维原视角渲染未完成：{source_execution['state']}",reconstruction=info)
                return
            with Image.open(stage/"source-alpha.png") as alpha, Image.open(stage/"source"/"source-view.png") as view:
                fidelity=source_fidelity(np.asarray(alpha),np.asarray(view.getchannel("A")))
            info["fidelity"]=fidelity
            info["worker"]=json.loads((stage/"worker-result.json").read_text(encoding="utf-8"))
            for name in ("mesh.glb","model-input.png","source-alpha.png","worker-result.json"):
                shutil.copy2(stage/name,root/name)
            shutil.copy2(stage/"source"/"source-view.png",root/"source-view.png")
            atomic_json(ready,info)
    row["reconstruction"]=info
    row["mesh"]=(root/"mesh.glb").as_posix()
    row["source_view"]=(root/"source-view.png").as_posix()
    if not info["fidelity"]["passed"]:
        row.update(status="review",code="reconstruction_failed",
                   reason="三维原视角未通过轮廓、孔洞或细小结构检查")
    else:
        row.update(status="awaiting_structure_review",code="structure_review_required",
                   reason="本地三维检查通过，等待 DeepSeek 核对关键结构")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("before","scene","out","weights","cache","open-from","reposition-from","vendor","python","blender"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--catalog",type=Path,default=HERE/"asset-catalog.json")
    cli.add_argument("--budget",type=int,default=2_000_000)
    args=cli.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.resolve())
    if not 0<=args.budget<=2_000_000:
        cli.error("budget must be between 0 and 2000000")
    args.quiet_from=None
    args.numbers=list(OPEN_PILOT)
    args.all=False
    args.out.mkdir(parents=True,exist_ok=True)
    args.cache.mkdir(parents=True,exist_ok=True)
    model=args.weights/"TripoSR"/"model.ckpt"
    if not model.is_file() or digest(model)!=CHECKPOINT_SHA:
        raise RuntimeError("TripoSR weights missing or checksum does not match the pinned official model")
    scene=load(Scene,args.scene)
    background=(args.scene.parent/scene.background).resolve(strict=True)
    if digest(background)!=scene.sha256:
        raise RuntimeError("Fixed CCTV background hash mismatch")
    with lock_directory(args.cache),lock_directory(args.out):
        budget=Budget(args.out/"model-usage.json",args.budget)
        rows=[]
        for folder,row in assets(args,scene):
            row["signature"]=key(CODE_REV,row["signature"],MODEL_REV,CHECKPOINT_SHA)
            result=folder/"result.json"
            saved=json.loads(result.read_text(encoding="utf-8")) if result.is_file() else None
            if saved and saved.get("signature")==row["signature"]:
                row=saved
            else:
                row["baseline"]=row["before_png"]
                try:
                    build_item(row,folder,args)
                except Exception as error:
                    row.update(status="review",code="reconstruction_error",reason=f"{type(error).__name__}: {str(error)[:220]}")
                atomic_json(result,row)
            rows.append(row)
            report(args.out,rows,budget)
            print(f"{row['number']:02} {row['status']} {row.get('reason','')}",flush=True)
    report(args.out,rows,budget)


if __name__=="__main__":
    main()
