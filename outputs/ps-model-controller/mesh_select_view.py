"""Choose one scene-conditioned, physically checked render from a model library."""
import argparse
import asyncio
import json
import math
from pathlib import Path
import shutil
import time

import httpx
import numpy as np
from PIL import Image

from batch_v2 import Budget, BudgetReached, Runner, lock_directory
from mesh_carve import interior_holes
from mesh_inspection import scale_projection
from mesh_pipeline import prepare_views, valid_zone
from mesh_placement import mesh_open_candidates, mesh_view, open_fit
from mesh_scene import analyze_scene, candidate_scale
from metric_scale import silhouette_axes
from run import Config, load
from scene_pose import PoseAtlas
from scene_v2 import NeedsReview, Perception, atomic_json, digest, read_image

HERE=Path(__file__).resolve().parent


def overlaps_protected(fit,protected):
    x,y=fit["origin"]
    h,w=fit["pixels"].shape[:2]
    return any(x<r and x+w>l and y<b and y+h>t for l,t,r,b in protected)


async def select(args):
    if (args.out/"result.json").exists():
        raise FileExistsError("请选择新的输出目录")
    args.out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(line) for line in args.library.read_text(encoding="utf8").splitlines() if line.strip()]
    selected=[r for r in records if (args.asset.isdigit() and r["number"]==int(args.asset)) or r["asset_id"]==args.asset]
    if len(selected)!=1:
        raise ValueError("物品尚未入库或编号不唯一")
    row=selected[0].copy()
    row["calls"]=[]
    row["job_id"]=f'view-{row["asset_id"]}'
    model=Path(row.get("mesh",""))
    if not model.is_file() or digest(model)!=row.get("mesh_sha256"):
        raise NeedsReview("网格不存在或与模型库记录不一致")
    source=Path(row.get("source_view",""))
    alpha=source.parent/"source-alpha.png"
    if (not source.is_file() or not alpha.is_file() or digest(source)!=row.get("source_view_sha256")
            or digest(alpha)!=row.get("source_alpha_sha256") or digest(Path(row["rust"]))!=row.get("rust_sha256")):
        raise NeedsReview("源图、原视角或透明度证据与模型库记录不一致")
    if not row.get("reconstruction",{}).get("fidelity",{}).get("passed"):
        raise NeedsReview("原视角轮廓或孔洞未通过，不能选用该模型")
    catalog=json.loads(args.catalog.read_text(encoding="utf8"))
    asset=next((x for x in catalog["assets"] if x["asset_id"]==row["asset_id"]),None)
    row["physical_size"]=asset.get("object_size") if asset else None
    background=args.background.resolve(strict=True)
    frame=read_image(background)
    if args.target_px is None:
        target=round(min(frame.size)*.10,1)
        target_basis="provisional_frame_fraction"
    else:
        target=args.target_px
        target_basis="caller_provisional_pixels"
    if not 8<=target<=min(frame.size)*.8:
        raise ValueError("候选像素尺寸超出可用范围")
    row["approved_size_reference"]={"major_px":target}
    config=load(Config,args.config)
    engine=Perception(args.weights,args.cache)
    async with httpx.AsyncClient(follow_redirects=False,trust_env=False) as http:
        runner=Runner(None,http,config,None,args.cache,Budget(args.out/"model-usage.json",args.budget))
        scene=await analyze_scene(runner,row,background,frame,engine,args.out)
        atlas=PoseAtlas(*engine.scene(background))
        atlas.elevation=scene["camera_elevation_deg"]
        pose_deadline=time.monotonic()+600
        start_scene={**scene,"prefer_openings":True,"orbit_elevation_interval":scene["camera_elevation_interval"],
                     "deadline":pose_deadline}
        start_scene.pop("camera_elevation_interval",None)
        views=await asyncio.to_thread(prepare_views,row,args.out,args,start_scene)
        attempts=list(row["pose_attempts"])
        atomic_json(args.out/"pose-attempts.json",attempts)
        if not views:
            reasons=list(dict.fromkeys(item.get("reason","") for item in attempts if item.get("reason")))
            raise NeedsReview("没有通过刚体与源图可见面检查的姿态："+"；".join(reasons[:4]))
        with Image.open(alpha) as im:
            source_openings=interior_holes(np.asarray(im.convert("L")))[1]
        fits=[]; rejections=[]
        for index,(folder,pose,visibility) in enumerate(views):
            with Image.open(folder/"object.png") as im:
                visible_openings=interior_holes(np.asarray(im.getchannel("A")))[1]
            proposal=args.out/"proposals"/str(index);proposal.mkdir(parents=True,exist_ok=True)
            pixels=mesh_view(folder,target,pose["contacts_px"])
            projection=scale_projection(row,pose,folder) if row["physical_size"] else None
            def at(location):
                if not valid_zone(location,scene):
                    raise NeedsReview("位置不在场景承托区域")
                if not row["physical_size"]:
                    raise NeedsReview("物品物理尺度未知")
                return candidate_scale(row,location,pixels[0][:,:,3],scene,projection=projection)
            def eligible(location):
                if not valid_zone(location,scene): return False
                try: at(location);return True
                except NeedsReview: return not row["physical_size"]
            candidates=mesh_open_candidates(pixels,target,atlas,proposal,eligible)
            for candidate in candidates:
                try:
                    fit=open_fit(candidate,proposal,atlas,frame,pixels,
                                 source_visible_fraction=visibility["source_visible_fraction"],
                                 physics_stable=pose["physics_stable"],
                                 scale_at=at if row["physical_size"] else None,
                                 pixel_size_review=not bool(row["physical_size"]))
                    if overlaps_protected(fit,scene["protected"]):
                        raise NeedsReview("物品范围与保护区相交")
                    fits.append((fit,(folder,pose,visibility),proposal,visible_openings))
                except (NeedsReview,ValueError) as e:
                    rejections.append({"view":index,"candidate":candidate["id"],"reason":str(e)[:180]})
        fits.sort(key=lambda x:(min(x[3],source_openings),x[0]["rank"]),reverse=True)
        if not fits:
            raise NeedsReview("没有同时满足当前场景承托、净空与已知尺度的视角")
        fit,view,proposal,visible_openings=fits[0]
        checked_scene={**scene,"seed_view":view,"deadline":pose_deadline}
        verified=await asyncio.to_thread(prepare_views,row,args.out/"verify",args,checked_scene)
        attempts.extend(row["pose_attempts"])
        atomic_json(args.out/"pose-attempts.json",attempts)
        if not verified:
            raise NeedsReview("所选姿态在俯角区间端点未通过刚体或可见面检查")
        _,pose,visibility=verified[0]
        with Image.open(view[0]/"object.png") as im:
            major=silhouette_axes(np.asarray(im.getchannel("A")))[0]
        endpoints=[]
        for folder,end,endvis in pose["interval_views"]:
            with Image.open(folder/"object.png") as im:
                endmajor=silhouette_axes(np.asarray(im.getchannel("A")))[0]
            endtarget=target*endmajor/major*pose["camera"]["orthographic_scale"]/end["camera"]["orthographic_scale"]
            pixels=mesh_view(folder,endtarget,end["contacts_px"])
            projection=scale_projection(row,end,folder) if row["physical_size"] else None
            def scale_at(location):
                return candidate_scale(row,location,pixels[0][:,:,3],scene,projection=projection)
            candidate={**fit["candidate"],"target_px":endtarget}
            checked=open_fit(candidate,proposal,atlas,frame,pixels,
                             source_visible_fraction=endvis["source_visible_fraction"],physics_stable=end["physics_stable"],
                             fixed_anchor=fit["support_anchor_px"],scale_at=scale_at if row["physical_size"] else None,
                             pixel_size_review=not bool(row["physical_size"]))
            if overlaps_protected(checked,scene["protected"]):
                raise NeedsReview("俯角端点与保护区相交")
            endpoints.append({"elevation_deg":end["camera"]["elevation_deg"],"target_px":endtarget,
                              "support":checked["evidence"]["component_support"],"visibility":endvis})
        if time.monotonic()>=pose_deadline:
            raise NeedsReview("单件姿态诊断达到 600 秒限制")
        shutil.copy2(view[0]/"object.png",args.out/"selected-view.png")
        limitations=["关键结构和视觉质量待复核","目标像素尺寸为候选值，不代表实测"]
        if not row["physical_size"]:
            limitations.append("物品物理尺寸未知，现场尺度待验")
        if source_openings and not visible_openings:
            limitations.append("源图开孔在当前视角未能从透明区域确认")
        result={"status":"view_selected_pending_structure_review","asset_id":row["asset_id"],"number":row["number"],
                "background_sha256":digest(background),"mesh_sha256":row["mesh_sha256"],
                "selected_view":str(args.out/"selected-view.png"),"pose":pose["camera"],
                "settled_matrices":pose["matrices"],"render_source":str(view[0]),
                "rotation_deg":pose["rotation_deg"],"support_anchor_px":fit["support_anchor_px"],
                "target_px":target,"target_basis":target_basis,"scale":fit["scale"],
                "openings":{"source_count":source_openings,"visible_count":visible_openings,
                            "status":"unknown" if source_openings and not visible_openings else "observed",
                            "note":"投影孔洞数量不等于关键结构复核"},
                "evidence":fit["evidence"],"endpoint_checks":endpoints,"attempts":attempts,
                "orbit_estimate":row.get("orbit_estimate"),"rejections":rejections,"scene_analysis":row["scene_analysis"],
                "token_usage":runner.budget.image_usage(row),
                "limitations":limitations}
        atomic_json(args.out/"result.json",result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",required=True,help="物品编号或 ID")
    p.add_argument("--background",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--library",type=Path,default=Path("outputs/三维模型库/results.jsonl"))
    p.add_argument("--catalog",type=Path,default=HERE/"asset-catalog.json")
    p.add_argument("--weights",type=Path,default=Path("work/v2-models"))
    p.add_argument("--cache",type=Path,default=Path("work/v2-cache"))
    p.add_argument("--blender",type=Path,default=Path("work/blender45/blender-4.5.3-windows-x64/blender.exe"))
    p.add_argument("--config",type=Path,default=HERE/"batch-config.example.json")
    p.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="OPTIX")
    p.add_argument("--target-px",type=float)
    p.add_argument("--budget",type=int,default=0)
    args=p.parse_args()
    if not 0<=args.budget<=30000: p.error("budget must be between 0 and 30000")
    for name,value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.resolve())
    args.cache.mkdir(parents=True,exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)
    with lock_directory(args.cache),lock_directory(args.out):
        try:
            result=asyncio.run(select(args))
            print(json.dumps(result,ensure_ascii=False),flush=True)
        except FileExistsError:
            raise
        except (NeedsReview,BudgetReached,ValueError,RuntimeError,OSError,httpx.HTTPError) as error:
            result={"status":"review","asset":args.asset,"reason":str(error)[:500]}
            if (args.out/"pose-attempts.json").is_file():
                result["diagnostic"]=str(args.out/"pose-attempts.json")
            atomic_json(args.out/"result.json",result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            raise SystemExit(2)


if __name__=="__main__": main()
