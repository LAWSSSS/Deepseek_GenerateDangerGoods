"""Build a resumable, provenance-checked model library from the local HD inventory."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import cv2
from PIL import Image

from batch_v2 import lock_directory
from mesh_build import build_item, CHECKPOINT_SHA
from mesh_camera import fit_source_camera
from mesh_carve import interior_holes
from mesh_checks import source_fidelity
from mesh_repair_batch import process as repair_mesh
from mesh_runtime import run_bounded, safe_tail
from reconstruct_worker import normalized_input
from scene_v2 import atomic_json, digest

HERE=Path(__file__).resolve().parent
VERSION="library-v1"


def write_index(out,rows):
    passing=sum(bool(row.get("reconstruction",{}).get("fidelity",{}).get("passed")) for row in rows)
    parts=['<!doctype html><meta charset="utf-8"><title>三维模型库</title>',
           '<style>body{font:16px sans-serif;margin:2em}section{border-top:1px solid #ccc;padding:1em}',
           'img{max-width:240px;max-height:180px;background:#ddd}</style>',
           f'<h1>三维模型库</h1><p>已记录 {len(rows)} 件，原视角几何通过 {passing} 件，其他待审。</p>',
           '<p>背面为推断；仅通过原视角检查不代表结构和场景合成已验收。</p>']
    for row in rows:
        parts.append(f'<section><h2>{row["number"]:02} {html.escape(row["file"])}</h2>')
        status={"review":"待审","awaiting_structure_review":"原视角检查通过，结构待审"}.get(row.get("status"),row.get("status","未知"))
        parts.append(f'<p>{html.escape(status)} · {html.escape(row.get("reason",""))}</p>')
        record=(out/"items"/f'{row["number"]:02}-{row["asset_id"]}'/"result.json").resolve()
        if record.is_file(): parts.append(f'<p><a href="{html.escape(record.as_uri(),quote=True)}">逐件检查记录</a></p>')
        for name,label in (("original","原貌素材"),("rust","做锈素材"),("source_view","原视角渲染"),("mesh","三维网格")):
            path=Path(row.get(name,""))
            if path.is_file():
                uri=html.escape(path.resolve().as_uri(),quote=True)
                parts.append(f'<p><a href="{uri}">{label}</a></p>')
                if name=="source_view": parts.append(f'<img src="{uri}" alt="原视角渲染">')
        parts.append('</section>')
    tmp=out/"index.tmp"
    tmp.write_text(''.join(parts),encoding="utf8")
    tmp.replace(out/"index.html")


def record_source_hashes(row):
    if row.get("source_view"):
        view=Path(row["source_view"])
        alpha=view.parent/"source-alpha.png"
        if view.is_file() and alpha.is_file():
            row["source_view_sha256"]=digest(view)
            row["source_alpha_sha256"]=digest(alpha)
            with Image.open(view) as rendered, Image.open(alpha) as source:
                fidelity=source_fidelity(np.asarray(source.convert("L")),np.asarray(rendered.getchannel("A")))
            reconstruction=row.get("reconstruction",{})
            if reconstruction.get("fidelity")!=fidelity:
                reconstruction["previous_recorded_fidelity"]=reconstruction.get("fidelity")
                reconstruction["fidelity"]=fidelity
            if not fidelity["passed"]:
                row.update(status="review",reason="原视角轮廓、孔洞或细结构未通过")


def holes_corroborated(rust_alpha,clean_alpha):
    rust,_=interior_holes(rust_alpha)
    clean,_=interior_holes(clean_alpha)
    count,labels=cv2.connectedComponents(rust.astype("uint8"))
    support=cv2.dilate(clean.astype("uint8"),np.ones((5,5),np.uint8))>0
    values=[float(support[labels==index].mean()) for index in range(1,count) if np.count_nonzero(labels==index)>=16]
    return {"corroborated":bool(values) and min(values)>=.6,
            "overlap_by_opening":[round(value,4) for value in values],"minimum_overlap":.6}


def build(row,folder,args):
    original=Path(row["original"])
    rust=Path(row["rust"])
    if digest(original)!=row["clean_sha256"] or digest(rust)!=row["rust_sha256"]:
        raise ValueError("素材内容与清单哈希不一致")
    previous=args.previous.get(row["asset_id"])
    if previous and digest(Path(previous["rust"]))==row["rust_sha256"]:
        mesh=Path(previous["mesh"])
        source=Path(previous["source_view"])
        camera=previous["reconstruction"]["source_camera"]
        if row["number"]==3:
            texture=args.bearing/"mesh"
            material=json.loads((texture/"result.json").read_text(encoding="utf8"))
            if material["source_asset_sha256"]!=row["rust_sha256"] or digest(texture/"mesh.glb")!=material["mesh_sha256"]:
                raise ValueError("轴承座细化网格来源不一致")
            mesh,source=texture/"mesh.glb",texture/"source-view.png"
        alpha=Path(previous["source_view"]).parent/"source-alpha.png"
        shutil.copy2(source,folder/"source-view.png")
        shutil.copy2(alpha,folder/"source-alpha.png")
        row.update(mesh=str(mesh),mesh_sha256=digest(mesh),source_view=str(folder/"source-view.png"),
                   reconstruction={**previous["reconstruction"],"source_camera":camera,"reused":True},
                   status="awaiting_structure_review" if previous["reconstruction"]["fidelity"]["passed"] else "review",
                   reason="复用既有模型；关键结构和新场景姿态仍需复核")
        return
    build_item(row,folder,args)
    if not row.get("mesh") or not Path(row["mesh"]).is_file():
        return
    source_alpha=Path(row["mesh"]).parent/"source-alpha.png"
    request={"operation":"source","mesh":row["mesh"],"out":str(folder),"render_device":args.render_device}
    req=folder/"source-request.json"
    def render(name):
        req.write_text(json.dumps(request),encoding="utf8")
        result=run_bounded([args.blender,"--background","--factory-startup","--python-exit-code","1",
                            "--python",HERE/"blender_worker.py","--",req],
                           log=folder/(name+".log"),seconds=180)
        if result["state"]!="ok" or not (folder/"source-view.png").is_file():
            raise RuntimeError(name+": "+safe_tail(folder/(name+".log"),250))
        return result
    initial=render("source")
    with Image.open(source_alpha) as source,Image.open(folder/"source-view.png") as view:
        camera=fit_source_camera(np.asarray(source),np.asarray(view.getchannel("A")))
    request["camera"]=camera
    calibrated=render("calibrated")
    with Image.open(source_alpha) as source,Image.open(folder/"source-view.png") as view:
        fidelity=source_fidelity(np.asarray(source),np.asarray(view.getchannel("A")))
    if camera["predicted_iou"]-fidelity["iou"]>.04:
        raise ValueError("相机标定与实际渲染不一致")
    shutil.copy2(source_alpha,folder/"source-alpha.png")
    row["source_view"]=str(folder/"source-view.png")
    row["mesh_sha256"]=digest(Path(row["mesh"]))
    row["reconstruction"].update(source_camera=camera,fidelity=fidelity,
                                 source_render=initial,calibrated_render=calibrated)
    row.update(status="awaiting_structure_review" if fidelity["passed"] else "review",
               reason="原视角通过，关键结构仍待复核" if fidelity["passed"] else "原视角轮廓、孔洞或细结构未通过")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",type=Path,default=Path("outputs/物品高清输入-39件/素材清单.jsonl"))
    p.add_argument("--out",type=Path,default=Path("outputs/三维模型库"))
    p.add_argument("--weights",type=Path,default=Path("work/v2-models"))
    p.add_argument("--cache",type=Path,default=Path("work/v2-cache"))
    p.add_argument("--vendor",type=Path,default=Path("work/vendor/TripoSR"))
    p.add_argument("--hf-home",type=Path,default=Path("work/huggingface"))
    p.add_argument("--python",type=Path,default=Path("work/triposr-env/Scripts/python.exe"))
    p.add_argument("--blender",type=Path,default=Path("work/blender45/blender-4.5.3-windows-x64/blender.exe"))
    p.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="OPTIX")
    p.add_argument("--previous",type=Path,default=Path("outputs/固定场景-v2最终几何复测/results.jsonl"))
    p.add_argument("--bearing",type=Path,default=Path("outputs/固定场景-轴承座纹理细化-v2"))
    p.add_argument("--retry",action="store_true")
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.resolve())
    if digest(args.weights/"TripoSR/model.ckpt")!=CHECKPOINT_SHA:
        raise ValueError("TripoSR 权重校验失败")
    os.environ["HF_HOME"]=str(args.hf_home)
    args.previous={r["asset_id"]:r for r in (json.loads(line) for line in args.previous.read_text(encoding="utf8").splitlines())}
    records=[json.loads(line) for line in args.inventory.read_text(encoding="utf8").splitlines() if line.strip()]
    if len({r["asset_id"] for r in records})!=len(records) or len({r["number"] for r in records})!=len(records):
        raise ValueError("素材清单包含重复编号或 ID")
    args.out.mkdir(parents=True,exist_ok=True)
    with lock_directory(args.out),lock_directory(args.cache):
        rows=[]
        for record in sorted(records,key=lambda r:r["number"]):
            variants=record["variants"]
            row={"library_version":VERSION,"number":record["number"],"asset_id":record["asset_id"],
                 "file":record["source_file"],"rust":str((args.inventory.parent/variants["rust"]["file"]).resolve()),
                 "original":str((args.inventory.parent/variants["clean"]["file"]).resolve()),
                 "rust_sha256":variants["rust"]["sha256"],"clean_sha256":variants["clean"]["sha256"]}
            folder=args.out/"items"/f'{row["number"]:02}-{row["asset_id"]}'
            folder.mkdir(parents=True,exist_ok=True)
            marker=folder/"result.json"
            saved=json.loads(marker.read_text(encoding="utf8")) if marker.exists() else None
            intact=bool(saved and saved.get("mesh") and Path(saved["mesh"]).is_file()
                        and saved.get("mesh_sha256")==digest(Path(saved["mesh"])))
            if intact and saved.get("source_view_sha256"):
                source=Path(saved.get("source_view",""))
                alpha=source.parent/"source-alpha.png"
                intact=bool(source.is_file() and alpha.is_file() and digest(source)==saved["source_view_sha256"]
                            and digest(alpha)==saved.get("source_alpha_sha256"))
            if (saved and saved.get("library_version")==VERSION and saved.get("rust_sha256")==row["rust_sha256"]
                    and saved.get("clean_sha256")==row["clean_sha256"]
                    and Path(row["rust"]).is_file() and Path(row["original"]).is_file()
                    and digest(Path(row["rust"]))==row["rust_sha256"]
                    and digest(Path(row["original"]))==row["clean_sha256"]
                    and (intact or not args.retry and not saved.get("mesh"))):
                row=saved
            else:
                try:
                    build(row,folder,args)
                except Exception as exc:
                    row.update(status="review",reason=f"{type(exc).__name__}: {str(exc)[:300]}")
            record_source_hashes(row)
            atomic_json(marker,row)
            rows.append(row)
            tmp=args.out/"results.tmp"
            tmp.write_text("".join(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n" for item in rows),encoding="utf8")
            tmp.replace(args.out/"results.jsonl")
            write_index(args.out,rows)
            print(f'{row["number"]:02} {row["status"]} {row["reason"]}',flush=True)
        # Repair only source-photo openings, then keep the original whenever
        # the repaired mesh does not pass the same source-view checks.
        for index,row in enumerate(rows):
            fidelity=row.get("reconstruction",{}).get("fidelity",{})
            if not row.get("mesh") or fidelity.get("passed") or not fidelity.get("filled_holes") or row.get("library_repair"):
                continue
            folder=args.out/"items"/f'{row["number"]:02}-{row["asset_id"]}'
            trial=json.loads(json.dumps(row))
            evidence=None
            try:
                with tempfile.TemporaryDirectory(prefix="clean-alpha-") as tmpdir:
                    normalized_input(Path(row["original"]),Path(tmpdir))
                    with Image.open(Path(tmpdir)/"source-alpha.png") as clean, Image.open(Path(row["source_view"]).parent/"source-alpha.png") as rust:
                        evidence=holes_corroborated(np.asarray(rust),np.asarray(clean))
                if not evidence["corroborated"]:
                    raise ValueError("做锈素材的孔洞未全部得到原貌透明区域支持")
                repair_mesh(trial,folder,args)
                repair=trial.get("repair",{})
                if trial.get("reconstruction",{}).get("fidelity",{}).get("passed"):
                    trial["mesh_sha256"]=digest(Path(trial["mesh"]))
                    trial["library_repair"]={"accepted":True,"source_hole_check":evidence,"evidence":repair}
                    row=trial
                else:
                    row["library_repair"]={"accepted":False,"source_hole_check":evidence,"evidence":repair,"reason":trial.get("reason")}
            except (ValueError,RuntimeError,OSError) as exc:
                row["library_repair"]={"accepted":False,"source_hole_check":evidence,"reason":str(exc)[:300]}
            rows[index]=row
            record_source_hashes(row)
            atomic_json(folder/"result.json",row)
            tmp=args.out/"results.tmp"
            tmp.write_text("".join(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n" for item in rows),encoding="utf8")
            tmp.replace(args.out/"results.jsonl")
            write_index(args.out,rows)
            print(f'{row["number"]:02} repair {row["library_repair"]["accepted"]}',flush=True)


if __name__=="__main__":
    main()
