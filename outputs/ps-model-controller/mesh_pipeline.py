"""Autonomous four-sample 3-D pilot; all uncertain stages fail closed."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import httpx
import numpy as np
from PIL import Image, ImageDraw
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from auto_v3 import save_preview
from batch_v2 import Budget, BudgetReached, Runner, lock_directory
from mesh_ai import check_render, check_structure
from mesh_build import report
from mesh_placement import compose_spec, mesh_view, open_fit
from mesh_runtime import run_bounded, safe_tail
from mesh_scene import MeshSceneAnswer, analyze_scene, candidate_scale
from mesh_visibility import coverage, orbit_estimates
from pose_retest import OPEN_PILOT, comparison, prepare_item, validate
from run import Config, load
from scene_pose import PoseAtlas
from scene_v2 import NeedsReview, Perception, atomic_json, digest, key, read_image

HERE=Path(__file__).resolve().parent
VERSION="mesh-pipeline-v1"
ROTATIONS=((0,0,0),(0,90,0),(0,0,90))


def render_board(row,frame,specs,path):
    board=Image.new("RGB",(1200,760),(35,35,35))
    draw=ImageDraw.Draw(board)
    whole=frame.copy()
    whole.thumbnail((795,400),Image.Resampling.LANCZOS)
    board.paste(whole,((800-whole.width)//2,(400-whole.height)//2))
    draw.text((6,4),"ORIGINAL FULL",fill="white")
    source=read_image(Path(row["original"]),"RGBA")
    source.thumbnail((390,370),Image.Resampling.LANCZOS)
    board.paste(source,(800+(400-source.width)//2,24+(376-source.height)//2),source)
    draw.text((806,4),"SOURCE OBJECT",fill="white")
    for index,spec in enumerate(specs):
        edit=read_image(Path(spec["png"]))
        edit=edit.crop(tuple(spec["crop"]))
        edit.thumbnail((392,328),Image.Resampling.LANCZOS)
        x=400*index
        board.paste(edit,(x+(400-edit.width)//2,430+(330-edit.height)//2))
        draw.text((x+6,406),spec["id"]+" ACTUAL PHOTOSHOP",fill="white")
    save_preview(board,path)


def valid_zone(candidate,scene):
    x,y=candidate["center_px"]
    width,height=scene["size"]
    for ident in scene["answer"].zones:
        center=scene["zones"][ident]["anchors"][0]
        cx=center[0]*(width-1)/1000
        cy=center[1]*(height-1)/1000
        if math.hypot(x-cx,y-cy)<=max(2*candidate["target_px"],.12*width):
            return ident
    return None


def prepare_views(row,folder,args,scene):
    views=[]
    orbit=[]
    row["pose_attempts"]=[]
    reference_alpha=Path(row["source_view"]).parent/"source-alpha.png"
    with Image.open(row["source_view"]) as source,Image.open(reference_alpha) as original:
        source_mask=np.asarray(source.getchannel("A"))
        reference_mask=np.asarray(original.convert("L"))
    source_camera=row["reconstruction"]["source_camera"]

    def check(index,rotation,azimuth=0):
        dest=folder/"views"/str(index)
        dest.mkdir(parents=True,exist_ok=True)
        request=dest/"request.json"
        request.write_text(json.dumps({"mesh":row["mesh"],"out":str(dest),"rotation_deg":rotation,
                                       "render_device":args.render_device,
                                       "camera_azimuth_deg":azimuth,
                                       "camera_elevation_deg":scene["camera_elevation_deg"]}),encoding="utf8")
        execution=run_bounded([args.blender,"--background","--factory-startup","--python-exit-code","1","--python",
                               HERE/"settled_mesh_worker.py","--",request],
                              log=dest/"blender.log",seconds=300)
        attempt={"rotation_deg":rotation,"camera_azimuth_deg":azimuth,"execution":execution}
        row["pose_attempts"].append(attempt)
        if execution["state"]!="ok" or not (dest/"blender-result.json").is_file():
            attempt.update(reason="三维姿态渲染失败",detail=safe_tail(dest/"blender.log",220))
            return
        result=json.loads((dest/"blender-result.json").read_text(encoding="utf8"))
        attempt["physics"]={name:result.get(name) for name in
                            ("component_count","component_min_z","component_collisions",
                             "motion","rotation_motion_deg","span","simulated_frames","stable_windows",
                             "support_graph")}
        if not result["physics_stable"]:
            attempt["reason"]="刚体未落稳或独立部件悬空"
            return
        if not (dest/"triangles.npz").is_file():
            attempt["reason"]="稳定姿态缺少同网格的可见面证据"
            return
        with Image.open(dest/"object.png") as posed:
            visibility=coverage(dest/"triangles.npz",source_camera,result["camera"],source_mask,
                                np.asarray(posed.getchannel("A")),reference_mask)
        attempt["visibility"]=visibility
        if not visibility["known"] or visibility["source_visible_fraction"]<.80:
            attempt["reason"]="新视角露出过多未经原照片验证的表面"
            if azimuth==0 and visibility["known"]:
                orbit.append((dest,result,rotation))
            return
        views.append((dest,{**result,"rotation_deg":rotation},visibility))

    for index,rotation in enumerate(ROTATIONS):
        check(index,rotation)
    if not views and orbit:
        guesses=[(fraction,degrees,rotation) for dest,result,rotation in orbit
                 for degrees,fraction in orbit_estimates(dest/"triangles.npz",source_camera,
                                                          result["camera"],source_mask,reference_mask)]
        if guesses:
            fraction,degrees,rotation=max(guesses,key=lambda guess:guess[0])
            row["orbit_estimate"]={"visible_fraction":round(fraction,4),
                                   "camera_azimuth_deg":degrees,"rotation_deg":rotation}
            if fraction>=.80:
                check(len(ROTATIONS),rotation,degrees)
    return views


async def photoshop_candidates(row,folder,args,background,frame,runner,fits):
    with tempfile.TemporaryDirectory(prefix="mesh-photoshop-",dir=folder) as temp:
        scratch=Path(temp)
        source=scratch/background.name
        shutil.copyfile(background,source)
        specs=[]
        details=[]
        for label,fit in zip("ABC",fits):
            spec,photo=compose_spec(fit,frame,scratch/label,label)
            spec["protected"]=row.get("protected",[])
            specs.append(spec)
            details.append(photo)
        with (folder/"mcp.log").open("a",encoding="utf8") as log:
            params=StdioServerParameters(command=str(args.mcp))
            async with stdio_client(params,errlog=log) as (reader,writer):
                async with ClientSession(reader,writer) as session:
                    await asyncio.wait_for(session.initialize(),timeout=30)
                    runner.session=session
                    reference=read_image(await runner.reference(source.as_posix(),background,scratch))
                    outcomes=await runner.ps({"mode":"compose","source":source.as_posix(),"candidates":specs})
                    measures=[]
                    for spec,outcome,fit in zip(specs,outcomes,fits):
                        _,measured=validate(reference,spec,outcome,fit["candidate"]["target_px"])
                        measures.append(measured)
                    board=folder/"实图复核.jpg"
                    render_board(row,frame,specs,board)
                    answer=await check_render(runner,row,board,[s["id"] for s in specs])
                    row["review"]=answer.model_dump(mode="json")
                    if not answer.passes() and answer.actionable_fix():
                        picked=next(s for s in specs if s["id"]==answer.pick)
                        fixed={**picked,"tone":{**picked["tone"],**answer.fix.model_dump()},
                               "png":(scratch/"fixed.png").as_posix(),"psd":(scratch/"fixed.psd").as_posix(),
                               "object_alpha":(scratch/"fixed-alpha.png").as_posix()}
                        outcome=(await runner.ps({"mode":"compose","source":source.as_posix(),"candidates":[fixed]}))[0]
                        index=next(i for i,s in enumerate(specs) if s["id"]==answer.pick)
                        _,measured=validate(reference,fixed,outcome,fits[index]["candidate"]["target_px"])
                        board=folder/"修正复核.jpg"
                        render_board(row,frame,[fixed],board)
                        answer=await check_render(runner,row,board,[fixed["id"]],recheck=True)
                        row["recheck"]=answer.model_dump(mode="json")
                        specs[index]=fixed
                        measures[index]=measured
                    if not answer.passes():
                        raise NeedsReview("实际合成图未达到视觉复核门槛")
                    index=next(i for i,s in enumerate(specs) if s["id"]==answer.pick)
                    selected=specs[index]
                    rendered=read_image(Path(selected["png"]))
                    comparison(folder,row,rendered,measures[index])
                    shutil.copy2(selected["png"],folder/"三维合成.png")
                    shutil.copy2(selected["psd"],folder/"三维分层.psd")
                    fit=fits[index]
                    row.update(status="pilot_review",code="visual_pass",reason="等待人工核验自然度",
                               preview=(folder/"三维合成.png").as_posix(),
                               psd=(folder/"三维分层.psd").as_posix(),
                               selected=selected["id"],geometry=fit["evidence"],
                               placement_context=fit["context"],projection=measures[index],
                               photography=details[index],actual_pose=fit["pose"],
                               scene_scale=fit["scale"])


async def pipeline(args):
    source_rows=[json.loads(line) for line in (args.build/"results.jsonl").read_text(encoding="utf8").splitlines()]
    if {row["number"] for row in source_rows}!=set(OPEN_PILOT):
        raise ValueError("本次试产必须包含 03、10、27、30 四件")
    scene_cfg=json.loads(args.scene.read_text(encoding="utf-8-sig"))
    background=(args.scene.parent/scene_cfg["background"]).resolve(strict=True)
    if digest(background)!=scene_cfg["sha256"]:
        raise ValueError("固定场景底图内容不匹配")
    cached_angle=None
    if args.geometry_only_from:
        history=(args.geometry_only_from/"results.jsonl").read_text(encoding="utf8").splitlines()
        intervals=set()
        for line in history:
            prior=json.loads(line)
            answer=prior.get("scene_analysis",{}).get("answer")
            if prior.get("background_sha256")==scene_cfg["sha256"] and answer:
                intervals.add(MeshSceneAnswer.model_validate_json(json.dumps(answer)).elevation_deg)
        if len(intervals)!=1:
            raise ValueError("本地诊断需要同一固定底图唯一的历史场景俯角区间")
        low,high=intervals.pop()
        if high-low>25:
            raise ValueError("历史俯角区间不足以进行本地诊断")
        cached_angle=(low+high)/2
    approved={r["asset_id"]:r for r in (json.loads(line) for line in
              (args.approved_from/"results.jsonl").read_text(encoding="utf8").splitlines())}
    with (args.approved_from/"试产验收.csv").open(encoding="utf-8-sig",newline="") as file:
        labels={row["asset_id"]:row for row in csv.DictReader(file)}
    config=load(Config,args.config)
    if config.vision and not config.vision.model.startswith("deepseek-"):
        raise ValueError("本次流程只允许使用已配置的 DeepSeek 视觉模型")
    budget=Budget(args.out/"model-usage.json",args.budget)
    rows=[]
    frame=None
    engine=None
    atlas=None
    async with httpx.AsyncClient(follow_redirects=False,trust_env=False) as http:
        runner=Runner(None,http,config,None,args.cache,budget)
        for source_row in sorted(source_rows,key=lambda item:item["number"]):
            started=time.monotonic()
            row=dict(source_row)
            folder=args.out/"items"/f"{row['number']:02}-{row['asset_id']}"
            folder.mkdir(parents=True,exist_ok=True)
            row["folder"]=folder.relative_to(args.out).as_posix()
            prior=approved.get(row["asset_id"],{})
            if (labels.get(row["asset_id"],{}).get("size_ok")!="合适"
                    or prior.get("background_sha256")!=scene_cfg["sha256"]
                    or not prior.get("bounds_px") or not prior.get("rendered_major_px")):
                raise ValueError("四件样片必须有同一背景下经用户确认的尺寸参照")
            row["approved_size_reference"]={"bounds_px":prior["bounds_px"],
                                             "major_px":prior["rendered_major_px"]}
            row["signature"]=key(VERSION,source_row["signature"],digest(HERE/"mesh_pipeline.py"),
                                 digest(HERE/"mesh_runtime.py"),digest(HERE/"mesh_checks.py"),digest(HERE/"mesh_ai.py"),
                                 digest(HERE/"mesh_scene.py"),digest(HERE/"mesh_placement.py"),
                                 digest(HERE/"settled_mesh_worker.py"),digest(HERE/"mesh_visibility.py"),
                                 digest(HERE/"blender_worker.py"),digest(HERE/"blender_support.py"),
                                 row["approved_size_reference"],args.render_device,cached_angle,digest(args.config))
            row["calls"]=[]
            marker=folder/"result.json"
            saved=json.loads(marker.read_text(encoding="utf8")) if marker.is_file() else None
            if (saved and saved.get("signature")==row["signature"]
                    and (saved.get("status")=="pilot_review" or
                         saved.get("status")=="review" and not args.retry)):
                row=saved
                rows.append(row)
                report(args.out,rows,budget)
                continue
            try:
                fidelity=row.get("reconstruction",{}).get("fidelity",{})
                if cached_angle is not None:
                    row["geometry_only"]={"cached_camera_elevation_deg":cached_angle}
                    if source_row.get("code")=="optimization_failed" or not fidelity.get("passed"):
                        raise NeedsReview("网格未通过结构或减面检查，跳过本地姿态诊断")
                    views=await asyncio.to_thread(prepare_views,row,folder,args,{"camera_elevation_deg":cached_angle})
                    row["geometry_only"]["passing_pose_count"]=len(views)
                    if not views:
                        raise NeedsReview("本地姿态诊断未找到同时通过落稳与可见面检查的姿态")
                    raise NeedsReview("本地姿态诊断通过，仍须进行视觉结构、场景承托及成图复核")
                # The independent model assessment is still useful for diagnosing a
                # failed local mesh, but it can never override the local hard gate.
                try:
                    answer,cached=await check_structure(runner,row,folder)
                    row["structure_review"]={**answer.model_dump(),"cache_hit":cached}
                except (BudgetReached,ValueError,RuntimeError,TimeoutError,OSError,httpx.HTTPError) as issue:
                    row["structure_review"]={"state":"unavailable","reason":str(issue)[:160]}
                    if fidelity.get("passed"):
                        raise
                    answer=None
                if source_row.get("code")=="optimization_failed" or not fidelity.get("passed"):
                    raise NeedsReview("三维结构未通过原视角轮廓或孔洞检查")
                if not answer.preserved:
                    raise NeedsReview("视觉模型发现关键结构缺失："+",".join(answer.missing))
                frame=frame or read_image(background)
                engine=engine or Perception(args.weights,args.cache)
                atlas=atlas or PoseAtlas(*engine.scene(background))
                scene=await analyze_scene(runner,row,background,frame,engine,folder)
                row["scene_analysis"]={"answer":scene["answer"].model_dump(),"cache_hit":scene["cache_hit"]}
                row["protected"]=scene["protected"]
                proposal_folder=folder/"proposals"
                proposal_folder.mkdir(exist_ok=True)
                proposals=prepare_item(proposal_folder,dict(row),atlas,frame,engine)
                views=await asyncio.to_thread(prepare_views,row,folder,args,scene)
                if not views:
                    raise NeedsReview("现有三维模型没有稳定且可验证的姿态")
                fits=[]
                row["placement_rejections"]=[]
                for candidate in proposals.get("candidates",[]):
                    zone=valid_zone(candidate,scene)
                    if not zone:
                        row["placement_rejections"].append({"id":candidate["id"],"reason":"未在模型认可的区域"})
                        continue
                    for view,result,visibility in views:
                        try:
                            pixels=mesh_view(view,candidate["target_px"],result["contacts_px"])
                            scale=candidate_scale(row,candidate,pixels[0][:,:,3],scene)
                            actual={**candidate,"scene_scale_status":"calibrated" if scale["status"]=="calibrated" else "consistent_estimate"}
                            fit=open_fit(actual,proposal_folder,atlas,frame,pixels,
                                         source_visible_fraction=visibility["source_visible_fraction"],
                                         physics_stable=result["physics_stable"])
                            fit.update(pose={"rotation_deg":result["rotation_deg"],"matrices":result["matrices"],
                                             "camera":result["camera"]},scale=scale)
                            fits.append(fit)
                        except (NeedsReview,ValueError) as issue:
                            row["placement_rejections"].append({"id":candidate["id"],"reason":str(issue)[:150]})
                fits.sort(key=lambda value:value["rank"],reverse=True)
                chosen=[]
                for fit in fits:
                    if all(math.dist(fit["origin"],other["origin"])>=.25*fit["candidate"]["target_px"] for other in chosen):
                        chosen.append(fit)
                    if len(chosen)==3:
                        break
                if not chosen:
                    raise NeedsReview("没有同时满足真实三维承托、尺度和净空检查的候选")
                await photoshop_candidates(row,folder,args,background,frame,runner,chosen)
            except (NeedsReview,BudgetReached,ValueError,RuntimeError,TimeoutError,OSError,httpx.HTTPError) as error:
                row.update(status="review",code="pipeline_review",reason=str(error)[:250])
                if source_row.get("code")=="reconstruction_failed":
                    row["code"]="reconstruction_failed"
            row["token_usage"]=budget.image_usage(row)
            row["elapsed_seconds"]=round(time.monotonic()-started,2)
            atomic_json(marker,row)
            rows.append(row)
            report(args.out,rows,budget)
            print(f"{row['number']:02} {row['status']} {row.get('reason','')}",flush=True)
    report(args.out,rows,budget)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("build","scene","out","weights","cache","blender","approved-from"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--config",type=Path,default=HERE/"batch-config.example.json")
    cli.add_argument("--mcp",type=Path,default=Path(os.environ.get("LOCALAPPDATA","."))/
                     "OpenAI/PhotoshopMCP/python/Scripts/photoshop-mcp-server.exe")
    cli.add_argument("--budget",type=int,default=2_000_000)
    cli.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="CPU")
    cli.add_argument("--geometry-only-from",type=Path,help="复用同一底图的历史俯角；只做本地姿态诊断")
    cli.add_argument("--retry",action="store_true",help="重试待审项（如已补齐 API 配置）")
    args=cli.parse_args()
    if not 0<=args.budget<=2_000_000:
        cli.error("budget must be between 0 and 2000000")
    for name,value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.resolve())
    args.out.mkdir(parents=True,exist_ok=True)
    with lock_directory(args.cache),lock_directory(args.out):
        asyncio.run(pipeline(args))


if __name__=="__main__":
    main()
