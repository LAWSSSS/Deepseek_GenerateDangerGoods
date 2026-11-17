"""Choose scene position first, then a verified 3-D view and local color."""
import argparse
import asyncio
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import httpx
import numpy as np
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from batch_v2 import Budget, BudgetReached, Runner, lock_directory
from mesh_carve import interior_holes
from mesh_inspection import scale_projection
from mesh_pipeline import prepare_views, valid_zone
from mesh_placement import compose_spec, mesh_view, open_fit
from mesh_runtime import run_bounded, safe_tail
from mesh_scene import analyze_scene, candidate_scale
from mesh_select_view import overlaps_protected
from mesh_site_selection import select_sites, site_candidate
from mesh_visibility import coverage
from metric_scale import silhouette_axes
from photometry import srgb, statistics
from pose_retest import tile, validate, render_alpha
from run import Config, load
from scene_pose import PoseAtlas
from scene_v2 import NeedsReview, Perception, atomic_json, digest, read_image

HERE=Path(__file__).resolve().parent


def validate_settled(folder,row):
    folder=folder.resolve(strict=True)
    request=json.loads((folder/"request.json").read_text(encoding="utf8"))
    pose=json.loads((folder/"blender-result.json").read_text(encoding="utf8"))
    if digest(Path(request["mesh"]))!=row["mesh_sha256"]:
        raise NeedsReview("缓存落稳姿态的网格与当前物品不一致")
    if not pose.get("physics_stable") or pose.get("component_count")!=1 or not (folder/"triangles.npz").is_file():
        raise NeedsReview("只可复用经过验证的单部件落稳姿态")
    return pose


def rerender(folder,output,row,args,azimuth,elevation,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=1:raise NeedsReview("单件姿态诊断达到 600 秒限制")
    output.mkdir(parents=True,exist_ok=True)
    request={"settled_view":str(folder),"mesh":row["mesh"],"out":str(output),
             "render_device":args.render_device,"render_resolution":1024,
             "camera_azimuth_deg":azimuth,"camera_elevation_deg":elevation}
    path=output/"request.json";path.write_text(json.dumps(request),encoding="utf8")
    execution=run_bounded([args.blender,"--background","--factory-startup","--python-exit-code","1",
                           "--python",HERE/"mesh_render_pose.py","--",path],
                          log=output/"blender.log",seconds=min(180,max(1,remaining)))
    if execution["state"]!="ok":
        raise NeedsReview("三维视角渲染失败："+safe_tail(output/"blender.log",180))
    pose=json.loads((output/"blender-result.json").read_text(encoding="utf8"))
    alpha_path=Path(row["source_view"]).parent/"source-alpha.png"
    with Image.open(row["source_view"]) as source,Image.open(alpha_path) as original,Image.open(output/"object.png") as image:
        visibility=coverage(output/"triangles.npz",row["reconstruction"]["source_camera"],pose["camera"],
                            np.asarray(source.getchannel("A")),np.asarray(image.getchannel("A")),np.asarray(original.convert("L")))
        openings=interior_holes(np.asarray(image.getchannel("A")))[1]
    if not visibility["known"] or visibility["source_visible_fraction"]<.8:
        raise NeedsReview("该视角源图可见面不足或计算不稳定")
    if time.monotonic()>=deadline:raise NeedsReview("单件姿态诊断达到 600 秒限制")
    return output,pose,visibility,openings,execution


def fit_site(site,view,atlas,scene,row,frame,folder,*,target):
    directory,pose,visibility,*_=view
    pixels=mesh_view(directory,target,pose["contacts_px"])
    candidate=site_candidate(site,pixels,atlas,folder,target=target)
    if not valid_zone(candidate,scene):raise NeedsReview("物体轮廓不在认可的场景区域")
    projection=scale_projection(row,pose,directory) if row.get("physical_size") else None
    def at(location):
        if not valid_zone(location,scene):raise NeedsReview("接触位置不在认可区域")
        return candidate_scale(row,location,pixels[0][:,:,3],scene,projection=projection)
    fit=open_fit(candidate,folder,atlas,frame,pixels,
                 source_visible_fraction=visibility["source_visible_fraction"],
                 physics_stable=pose["physics_stable"],fixed_anchor=site["support_anchor_px"],
                 scale_at=at if row.get("physical_size") else None,
                 pixel_size_review=not bool(row.get("physical_size")))
    if overlaps_protected(fit,scene["protected"]):raise NeedsReview("物体范围与保护区相交")
    return fit


def create_board(frame,rendered,previous,bounds,out):
    from auto_v3 import save_preview
    l,t,r,b=bounds
    if previous:
        old=json.loads((previous/"result.json").read_text(encoding="utf8"))["projection"]["bounds_px"]
        l,t,r,b=min(l,old[0]),min(t,old[1]),max(r,old[2]),max(b,old[3])
    pad=max(40,round(.18*max(r-l,b-t)))
    crop=(max(0,l-pad),max(0,t-pad),min(frame.width,r+pad),min(frame.height,b+pad))
    def enlarge(image,width,height):
        scale=min(width/image.width,height/image.height)
        return image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
    board=Image.new("RGB",(1600,1050),(30,33,38))
    tile(board,rendered,0,0,1000,500,"LOCATION FIRST / FULL SCENE")
    tile(board,enlarge(frame.crop(crop),585,470),1000,0,600,500,"ORIGINAL / SAME REGION")
    if previous:
        old=read_image(previous_image(previous))
        tile(board,enlarge(old.crop(crop),785,520),0,500,800,550,"PREVIOUS / REFERENCE")
    else:
        tile(board,enlarge(frame.crop(crop),785,520),0,500,800,550,"ORIGINAL / REFERENCE")
    tile(board,enlarge(rendered.crop(crop),785,520),800,500,800,550,"NEW / POSITION, VIEW AND COLOR")
    save_preview(board,out)


def compare_photometry(photo):
    scene=photo["scene"];before=photo["object_before"];after=photo["object_after"]
    def gaps(obj):
        return {"luma":round(abs(obj["luma_p20_p50_p80"][1]-scene["luma_p20_p50_p80"][1]),2),
                "contrast":round(abs(obj["contrast_p80_p20"]-scene["contrast_p80_p20"]),2),
                "lab_chroma":round(math.dist(obj["lab_ab"],scene["lab_ab"]),2)}
    return {"before":gaps(before),"after":gaps(after)}


def previous_image(folder):
    for name in ("场景合成.png","03-轴承座-检查全图.png"):
        path=folder/name
        if path.is_file():return path
    raise NeedsReview("上一版对照缺少全图 PNG")


def verify_previous(previous,row,frame):
    record=json.loads((previous/"result.json").read_text(encoding="utf8"))
    if record.get("number")!=row["number"]:
        raise NeedsReview("上一版对照不是同一件物品")
    image=read_image(previous_image(previous))
    if image.size!=frame.size:
        raise NeedsReview("上一版对照与场景尺寸不一致")
    l,t,r,b=record["projection"]["bounds_px"]
    pad=max(64,round(.8*max(r-l,b-t)))
    before=np.asarray(frame.convert("RGB"))[::16,::16].astype(np.int16)
    after=np.asarray(image.convert("RGB"))[::16,::16].astype(np.int16)
    yy,xx=np.indices(before.shape[:2]);xx*=16;yy*=16
    outside=(xx<l-pad)|(xx>=r+pad)|(yy<t-pad)|(yy>=b+pad)
    if not outside.any() or np.mean(np.max(np.abs(before-after),axis=2)[outside]>8)>.005:
        raise NeedsReview("上一版对照与当前背景不一致")


async def run(args):
    started=time.monotonic()
    if (args.out/"result.json").exists():raise FileExistsError("请选择新的输出目录")
    args.out.mkdir(parents=True,exist_ok=True)
    library=[json.loads(line) for line in args.library.read_text(encoding="utf8").splitlines() if line.strip()]
    matches=[r for r in library if (args.asset.isdigit() and r["number"]==int(args.asset)) or r["asset_id"]==args.asset]
    if len(matches)!=1:raise NeedsReview("物品不存在或尚未入库")
    from display_batch.inventory import is_storage_tank
    if is_storage_tank(matches[0]['number']):
        raise ValueError('储罐已从可摆放物品中移除')
    row=dict(matches[0]);row["calls"]=[];row["job_id"]=f'location-first-{row["asset_id"]}'
    mesh=Path(row["mesh"]);source=Path(row["source_view"]);alpha=source.parent/"source-alpha.png"
    for file,key in ((mesh,"mesh_sha256"),(source,"source_view_sha256"),(alpha,"source_alpha_sha256"),(Path(row["rust"]),"rust_sha256")):
        if not file.is_file() or digest(file)!=row.get(key):raise NeedsReview("模型或来源文件与记录不一致")
    if not row.get("reconstruction",{}).get("fidelity",{}).get("passed"):
        raise NeedsReview("模型原视角轮廓、孔洞或细结构未通过")
    catalog=json.loads(args.catalog.read_text(encoding="utf8"))
    asset=next((x for x in catalog["assets"] if x["asset_id"]==row["asset_id"]),None)
    row["physical_size"]=asset.get("object_size") if asset else None
    background=args.background.resolve(strict=True);frame=read_image(background)
    if args.previous:verify_previous(args.previous,row,frame)
    target=args.target_px if args.target_px is not None else round(min(frame.size)*.1,1)
    if not 8<=target<=min(frame.size)*.8:raise NeedsReview("目标尺寸不在可用范围")
    row["approved_size_reference"]={"major_px":target}
    config=load(Config,args.config);engine=Perception(args.weights,args.cache)
    events=[];rejected=[];attempts=[]
    async with httpx.AsyncClient(follow_redirects=False,trust_env=False) as http:
        runner=Runner(None,http,config,None,args.cache,Budget(args.out/"model-usage.json",args.budget))
        scene=await analyze_scene(runner,row,background,frame,engine,args.out)
        atlas=PoseAtlas(*engine.scene(background));atlas.elevation=scene["camera_elevation_deg"]
        sites,site_rejections=select_sites(atlas,scene,target,limit=128)
        atomic_json(args.out/"scene-sites.json",{"sites":sites,"rejections":site_rejections,"object_view_used":False})
        events.append("scene_positions_ranked")
        if not sites:raise NeedsReview("场景中没有可用承托点和尺度参照")
        deadline=time.monotonic()+600
        events.append("view_search_started")
        views=[]
        if args.settled_view:
            base=validate_settled(args.settled_view,row)
            yaw=float(base["camera"]["azimuth_deg"])
            for direction in ((yaw-5)%360,yaw,(yaw+5)%360):
                try:
                    view=await asyncio.to_thread(rerender,args.settled_view,args.out/"views"/f"yaw-{direction:g}",
                                                 row,args,direction,scene["camera_elevation_deg"],deadline)
                    views.append(view)
                    attempts.append({"azimuth_deg":direction,"state":"passed","visibility":view[2],"openings":view[3]})
                except NeedsReview as error:
                    attempts.append({"azimuth_deg":direction,"state":"rejected","reason":str(error)})
        else:
            view_scene={**scene,"deadline":deadline,"prefer_openings":True,
                        "orbit_elevation_interval":scene["camera_elevation_interval"]}
            view_scene.pop("camera_elevation_interval",None)
            generated=await asyncio.to_thread(prepare_views,row,args.out,args,view_scene)
            attempts.extend(row["pose_attempts"])
            for folder,pose,visibility in generated:
                with Image.open(folder/"object.png") as image: openings=interior_holes(np.asarray(image.getchannel("A")))[1]
                views.append((folder,pose,visibility,openings,None))
        atomic_json(args.out/"view-attempts.json",attempts)
        if not views:raise NeedsReview("场景已选位，但没有通过落稳与可见面检查的视角")
        with Image.open(alpha) as image:
            source_openings=interior_holes(np.asarray(image.convert("L")))[1]
        chosen=None
        endpoint_cache={}
        for site_index,site in enumerate(sites):
            fits=[]
            for index,view in enumerate(views):
                try:
                    folder=args.out/"proposals"/site["id"]/f"view-{index}"
                    fit=fit_site(site,view,atlas,scene,row,frame,folder,target=target)
                    opening_score=min(view[3],source_openings) if source_openings else 0
                    fits.append((opening_score,view[2]["source_visible_fraction"],fit["rank"],index,fit,view))
                except (NeedsReview,ValueError) as error:
                    rejected.append({"site":site["id"],"view":index,"reason":str(error)[:180]})
            fits.sort(key=lambda item:(bool(item[0]),item[2],item[1]),reverse=True)
            for _,_,_,index,fit,view in fits:
                try:
                    endpoint_checks=[]
                    if args.settled_view:
                        with Image.open(view[0]/"object.png") as im: major=silhouette_axes(np.asarray(im.getchannel("A")))[0]
                        for angle in scene["camera_elevation_interval"]:
                            cache_key=(view[1]["camera"]["azimuth_deg"],angle)
                            if cache_key not in endpoint_cache:
                                try:
                                    endpoint_cache[cache_key]=await asyncio.to_thread(rerender,args.settled_view,
                                        args.out/"endpoints"/f'yaw-{cache_key[0]:g}'/f"elev-{angle:g}",
                                        row,args,cache_key[0],angle,deadline)
                                except NeedsReview as error:
                                    endpoint_cache[cache_key]=error
                            endpoint=endpoint_cache[cache_key]
                            if isinstance(endpoint,NeedsReview):raise endpoint
                            with Image.open(endpoint[0]/"object.png") as im: endmajor=silhouette_axes(np.asarray(im.getchannel("A")))[0]
                            endtarget=target*endmajor/major*view[1]["camera"]["orthographic_scale"]/endpoint[1]["camera"]["orthographic_scale"]
                            endfit=fit_site(site,endpoint,atlas,scene,row,frame,
                                            args.out/"proposals"/site["id"]/f"endpoint-{index}-{angle:g}",target=endtarget)
                            endpoint_checks.append({"elevation_deg":angle,"source_visible_fraction":endpoint[2]["source_visible_fraction"],
                                                    "component_support":endfit["evidence"]["component_support"],"scale":endfit["scale"]})
                    else:
                        verified=await asyncio.to_thread(prepare_views,row,args.out/"verify",args,
                                                          {**scene,"seed_view":view[:3],"deadline":deadline})
                        attempts.extend(row["pose_attempts"])
                        if not verified:raise NeedsReview("俯角端点未通过落稳或可见面检查")
                        for epfolder,eppose,epvis in verified[0][1]["interval_views"]:
                            epview=(epfolder,eppose,epvis,0,None)
                            with Image.open(view[0]/"object.png") as im: major=silhouette_axes(np.asarray(im.getchannel("A")))[0]
                            with Image.open(epfolder/"object.png") as im: endmajor=silhouette_axes(np.asarray(im.getchannel("A")))[0]
                            endtarget=target*endmajor/major*view[1]["camera"]["orthographic_scale"]/eppose["camera"]["orthographic_scale"]
                            endfit=fit_site(site,epview,atlas,scene,row,frame,args.out/"proposals"/site["id"]/f'endpoint-{index}-{eppose["camera"]["elevation_deg"]:g}',target=endtarget)
                            endpoint_checks.append({"elevation_deg":eppose["camera"]["elevation_deg"],
                                "source_visible_fraction":epvis["source_visible_fraction"],
                                "component_support":endfit["evidence"]["component_support"],"scale":endfit["scale"]})
                    chosen=(site_index,site,fit,view,endpoint_checks)
                    break
                except (NeedsReview,ValueError) as error:
                    rejected.append({"site":site["id"],"view":index,"reason":str(error)[:180]})
            if chosen:break
        atomic_json(args.out/"view-attempts.json",attempts)
        if not chosen:raise NeedsReview("已排序的位置均没有同时通过视角、承托、净空和尺度检查的候选")
        site_index,site,fit,view,endpoint_checks=chosen
        events.append("location_and_view_verified")
        if time.monotonic()>=deadline:raise NeedsReview("单件姿态诊断达到 600 秒限制")
        with tempfile.TemporaryDirectory(prefix="photoshop-",dir=args.out) as tmp:
            scratch=Path(tmp);background_copy=scratch/background.name;shutil.copy2(background,background_copy)
            with (args.out/"mcp.log").open("a",encoding="utf8") as log:
                async with stdio_client(StdioServerParameters(command=str(args.mcp)),errlog=log) as (reader,writer):
                    async with ClientSession(reader,writer) as session:
                        await asyncio.wait_for(session.initialize(),timeout=30);runner.session=session
                        reference=read_image(await runner.reference(background_copy.as_posix(),background,scratch))
                        spec,photo=compose_spec(fit,frame,scratch/"candidate",site["id"],protected=scene["protected"])
                        if photo.get("status")!="measured":
                            raise NeedsReview("同一承托面或物品的颜色采样不足")
                        spec["protected"]=scene["protected"]
                        outcome=(await runner.ps({"mode":"compose","source":background_copy.as_posix(),"candidates":[spec]}))[0]
                        rendered,measurement=validate(reference,spec,outcome,target)
                        if fit["scale"] and fit["scale"]["status"]!="unknown":
                            low,high=fit["scale"]["projected_major_px_range"]
                            if not low<=measurement["rendered_major_px"]<=high:
                                raise NeedsReview("实际输出尺寸与局部尺度依据不符")
                        object_alpha=render_alpha(spec,"object_alpha")
                        actual=statistics(np.ascontiguousarray(np.asarray(srgb(rendered.crop(tuple(spec["crop"]))))),object_alpha>=190)
                        if actual is None:raise NeedsReview("Photoshop 输出的物品像素不足")
                        photo["object_after"]=actual[0]
                        photo["applied_tone"]=spec["tone"]
                        photo["comparison"]=compare_photometry(photo)
                        if any(photo["comparison"]["after"][metric]>photo["comparison"]["before"][metric]+4
                               for metric in ("luma","contrast")):
                            raise NeedsReview("实际输出的局部亮度或对比度匹配明显退化")
                        shutil.copy2(spec["png"],args.out/"场景合成.png")
                        shutil.copy2(spec["psd"],args.out/"场景合成.psd")
                        shutil.copy2(view[0]/"object.png",args.out/"选中三维视角.png")
                        create_board(frame,rendered,args.previous,measurement["bounds_px"],args.out/"位置优先-对照.jpg")
        events.extend(["local_color_fitted","photoshop_export_validated"])
        result={"status":"human_inspection","asset_id":row["asset_id"],"number":row["number"],
                "decision_order":events,"location_rank":site_index+1,"location":site,
                "camera":view[1]["camera"],"rotation_deg":view[1].get("rotation_deg"),
                "selected_view":str(args.out/"选中三维视角.png"),
                "mesh_sha256":row["mesh_sha256"],"background_sha256":digest(background),
                "visibility":view[2],"visible_openings":view[3],"geometry":fit["evidence"],
                "support_anchor_px":fit["support_anchor_px"],"scene_scale":fit["scale"],
                "camera_endpoint_checks":endpoint_checks,"projection":measurement,"photography":photo,
                "scene_analysis":row["scene_analysis"],"site_rejections":site_rejections,
                "rejections":rejected,"view_attempts":attempts,
                "target_px":target,"target_basis":"caller_provisional_pixels" if args.target_px else "provisional_frame_fraction",
                "token_usage":runner.budget.image_usage(row),
                "elapsed_seconds":round(time.monotonic()-started,2),
                "review":{"structure":"pending","visual_model":"pending",
                          "scale":fit["scale"]["status"] if fit["scale"] else "unknown"}}
        atomic_json(args.out/"result.json",result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",required=True);p.add_argument("--background",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--library",type=Path,default=Path("outputs/三维模型库-圆润复验/results.jsonl"))
    p.add_argument("--catalog",type=Path,default=HERE/"asset-catalog.json")
    p.add_argument("--weights",type=Path,default=Path("work/v2-models"));p.add_argument("--cache",type=Path,default=Path("work/v2-cache"))
    p.add_argument("--blender",type=Path,default=Path("work/blender45/blender-4.5.3-windows-x64/blender.exe"))
    p.add_argument("--mcp",type=Path,default=Path("work/photoshop-mcp-env/Scripts/photoshop-mcp-server.exe"))
    p.add_argument("--config",type=Path,default=HERE/"batch-config.example.json")
    p.add_argument("--settled-view",type=Path);p.add_argument("--previous",type=Path)
    p.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="OPTIX")
    p.add_argument("--target-px",type=float);p.add_argument("--budget",type=int,default=0)
    args=p.parse_args()
    if not 0<=args.budget<=30000:p.error("budget must be between 0 and 30000")
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    args.cache.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    with lock_directory(args.cache),lock_directory(args.out):
        try:print(json.dumps(asyncio.run(run(args)),ensure_ascii=False),flush=True)
        except FileExistsError:raise
        except Exception as error:
            result={"status":"review","asset":args.asset,"reason":str(error)[:500],
                    "scene_sites":str(args.out/"scene-sites.json") if (args.out/"scene-sites.json").exists() else None,
                    "view_attempts":str(args.out/"view-attempts.json") if (args.out/"view-attempts.json").exists() else None}
            atomic_json(args.out/"result.json",result);print(json.dumps(result,ensure_ascii=False),flush=True);raise SystemExit(2)


if __name__=="__main__":main()
