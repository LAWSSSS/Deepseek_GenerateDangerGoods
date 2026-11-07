"""Produce a human review image from a locally verified 3-D pose."""
import argparse
import asyncio
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import httpx
import numpy as np
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, "outputs/ps-model-controller")
from batch_v2 import Budget, Runner, lock_directory
from mesh_pipeline import valid_zone
from mesh_placement import compose_spec, mesh_open_candidates, mesh_view, open_fit
from mesh_scene import analyze_scene, candidate_scale
from mesh_visibility import coverage
from pose_retest import tile, validate, render_alpha
from metric_scale import silhouette_axes
from mesh_carve import interior_holes
from mesh_checks import source_fidelity
from run import Config, load
from photometry import statistics, srgb
from scene_pose import PoseAtlas
from scene_v2 import NeedsReview, Perception, atomic_json, digest, read_image


def scale_projection(row,pose,folder):
    with Image.open(folder/"object.png") as image:
        projected=silhouette_axes(np.asarray(image.getchannel("A")))[0]
        factor=projected/image.width*pose["camera"]["orthographic_scale"]/max(pose["initial_extents"])
    return {"elevation_deg":pose["camera"]["elevation_deg"],
            "projected_major_mm":[factor*x for x in row["physical_size"]["major_mm"]]}


def mounting_ratio(alpha):
    import cv2
    holes,_=interior_holes(alpha)
    count,_,stats,centers=cv2.connectedComponentsWithStats(holes.astype("uint8"),8)
    if count<4:
        raise NeedsReview("不能识别中央通孔和两个安装孔")
    ids=sorted(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)[:3]
    mounts=ids[1:]
    ys,xs=np.nonzero(alpha>=128)
    if not len(xs) or xs.max()==xs.min():
        raise NeedsReview("不能测量座体外廓")
    return float(np.linalg.norm(centers[mounts[0]]-centers[mounts[1]])/(xs.max()-xs.min()))


async def inspect(args):
    output = args.out.resolve()
    output.mkdir(exist_ok=True)
    if (output/"result.json").exists():
        raise FileExistsError("检查图输出目录已有结果，请使用新目录")
    proposals = output / "proposals"
    proposals.mkdir(exist_ok=True)
    config = load(Config, args.config)
    cache = args.cache.resolve()
    previous = args.history
    history=[json.loads(line) for line in (previous/"results.jsonl").read_text(encoding="utf8").splitlines()]
    selected=[record for record in history if record.get("number")==3]
    if len(selected)!=1:
        raise ValueError("历史结果必须包含唯一的 03 轴承座")
    row=selected[0]
    cfg = args.scene
    cfg_data = json.loads(cfg.read_text(encoding="utf-8-sig"))
    background = (cfg.parent/cfg_data["background"]).resolve()
    if digest(background) != cfg_data["sha256"] or row["background_sha256"] != cfg_data["sha256"]:
        raise ValueError("background provenance mismatch")
    target = row["approved_size_reference"]["major_px"]
    with (args.approved_from/"试产验收.csv").open(encoding="utf-8-sig",newline="") as file:
        labels={record["asset_id"]:record for record in csv.DictReader(file)}
    approved=[json.loads(line) for line in (args.approved_from/"results.jsonl").read_text(encoding="utf8").splitlines()]
    prior=next((record for record in approved if record["asset_id"]==row["asset_id"]),None)
    if (not prior or labels.get(row["asset_id"],{}).get("size_ok")!="合适" or
            prior.get("background_sha256")!=cfg_data["sha256"] or
            prior.get("rendered_major_px")!=target or prior.get("bounds_px")!=row["approved_size_reference"]["bounds_px"] or
            row["number"]!=3 or row["asset_id"]!="asset-928d815bf447" or
            row["targets_px"]!=[target] or not row["reconstruction"]["fidelity"]["passed"]):
        raise ValueError("unverified source silhouette or pixel target")
    view = args.pose.resolve()
    pose = json.loads((view/"blender-result.json").read_text(encoding="utf8"))
    baseline_dir=(args.baseline.parent if args.baseline else Path("outputs/固定场景-轴承座三维检查图"))
    baseline_record=json.loads((args.baseline if args.baseline else baseline_dir/"result.json").read_text(encoding="utf8"))
    if args.baseline:
        if (baseline_record.get("status")!="human_inspection" or baseline_record.get("number")!=3 or
                baseline_record.get("approved_target_px")!=target or
                not isinstance(baseline_record.get("support_anchor_px"),list) or
                len(baseline_record["support_anchor_px"])!=2 or
                not all(isinstance(v,(float,int)) and math.isfinite(v) for v in baseline_record["support_anchor_px"]) or
                baseline_record.get("pose",{}).get("elevation_deg")!=pose["camera"]["elevation_deg"] or
                baseline_record.get("pose",{}).get("azimuth_deg")!=pose["camera"]["azimuth_deg"]):
            raise NeedsReview("上一版检查图的尺寸或姿态与本次不一致")
    fixed_anchor=baseline_record.get("support_anchor_px") if args.baseline else None
    source_alpha = Path(row["source_view"]).parent/"source-alpha.png"
    source_view = args.source_view.resolve()
    pose_request=json.loads((view/"request.json").read_text(encoding="utf8"))
    source_request=json.loads((source_view.parent/"request.json").read_text(encoding="utf8"))
    if digest(Path(pose_request["mesh"]))!=digest(Path(source_request["mesh"])):
        raise NeedsReview("源视图和落稳姿态的网格不一致")
    material=None
    material_result=source_view.parent/"result.json"
    if material_result.exists():
        data=json.loads(material_result.read_text(encoding="utf8"))
        if data.get("mesh_sha256")!=digest(Path(pose_request["mesh"])) or data.get("source_asset_sha256")!=digest(Path(row["rust"])):
            raise NeedsReview("纹理来源或网格记录不匹配")
        material={key:data[key] for key in ("texture_size","detail_strength","detail_gain","normal_detail") if key in data}
    if source_request.get("camera")!=row["reconstruction"]["source_camera"]:
        raise NeedsReview("源视图相机与已标定相机不一致")
    with Image.open(source_view) as source, Image.open(source_alpha) as ref:
        source_mask = np.asarray(source.getchannel("A"))
        reference_mask = np.asarray(ref.convert("L"))
    fidelity=source_fidelity(reference_mask,source_mask)
    if not fidelity["passed"] or fidelity["holes"]<3:
        raise NeedsReview("高细节网格未通过源图轮廓、孔洞或细结构检查")
    source_ratio=mounting_ratio(reference_mask)
    mesh_ratio=mounting_ratio(source_mask)
    factory_ratio=159/206
    if abs(source_ratio-mesh_ratio)>.05 or abs(mesh_ratio-factory_ratio)/factory_ratio>.15:
        raise NeedsReview("安装孔距与源图或目录比例不一致")
    proportion={"source_projected_ratio":source_ratio,"mesh_projected_ratio":mesh_ratio,
                "factory_width_mm":206,"factory_hole_spacing_mm":159,
                "method":"source_view_projection_approximation","projection_tolerance_fraction":.15}
    with Image.open(view/"object.png") as posed:
        visibility = coverage(view/"triangles.npz", row["reconstruction"]["source_camera"],
                              pose["camera"], source_mask, np.asarray(posed.getchannel("A")), reference_mask)
    print("visibility", visibility, flush=True)
    if not pose["physics_stable"] or not visibility["known"] or visibility["source_visible_fraction"] < .80:
        raise NeedsReview("pose failed physical or source coverage gate")
    endpoint_poses=[]
    for angle in (35,55):
        endpoint=args.pose.parent/f"pose-{angle}"
        data=json.loads((endpoint/"blender-result.json").read_text(encoding="utf8"))
        request=json.loads((endpoint/"request.json").read_text(encoding="utf8"))
        if (digest(Path(request["mesh"]))!=digest(Path(pose_request["mesh"])) or
                data["camera"]["azimuth_deg"]!=pose["camera"]["azimuth_deg"]):
            raise NeedsReview("俯角端点与主图网格或方位不一致")
        with Image.open(endpoint/"object.png") as image:
            check=coverage(endpoint/"triangles.npz",row["reconstruction"]["source_camera"],
                           data["camera"],source_mask,np.asarray(image.getchannel("A")),reference_mask)
        if not data["physics_stable"] or not check["known"] or check["source_visible_fraction"] < .80:
            raise NeedsReview(f"俯角 {angle} 的落稳或可见面未通过")
        endpoint_poses.append((endpoint,data,check))
    pixels = mesh_view(view, target, pose["contacts_px"])
    projection=scale_projection(row,pose,view)
    frame = read_image(background)
    engine = Perception(args.weights.resolve(), cache)
    atlas = PoseAtlas(*engine.scene(background))
    row["calls"] = []
    with lock_directory(cache), lock_directory(output):
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as http:
            runner = Runner(None, http, config, None, cache, Budget(output/"model-usage.json", 0))
            scene = await analyze_scene(runner, row, background, frame, engine, output)
            atlas.elevation = scene["camera_elevation_deg"]
            if pose["camera"]["elevation_deg"] != atlas.elevation:
                raise NeedsReview("camera does not match scene")
            records = []
            for allow_missing in (False, True):
                def prefilter(candidate):
                    if not valid_zone(candidate, scene):
                        return False
                    if fixed_anchor is not None:
                        if math.dist(candidate["support_anchor_px"],fixed_anchor)>.15*target:
                            return False
                        candidate={**candidate,"support_anchor_px":fixed_anchor}
                    try:
                        candidate_scale(row, candidate, pixels[0][:,:,3], scene, allow_size_adjustment=True,
                                        projection=projection)
                    except NeedsReview as issue:
                        return (getattr(issue, "code", None) == "insufficient_local_references" and allow_missing)
                    return True

                candidates = mesh_open_candidates(pixels, target, atlas, proposals, prefilter, limit=32)
                fits = []
                for candidate in candidates:
                    candidate = {**candidate, "scene_scale_status":"unknown"}
                    try:
                        scale_candidate=({**candidate,"support_anchor_px":fixed_anchor}
                                         if fixed_anchor is not None else candidate)
                        proposal = candidate_scale(row,scale_candidate,pixels[0][:,:,3],scene,
                                                   allow_size_adjustment=True,projection=projection)
                        actual_target = baseline_record["target_px"] if fixed_anchor is not None else proposal["selected_target_px"]
                    except NeedsReview as issue:
                        if not allow_missing or getattr(issue,"code",None)!="insufficient_local_references":
                            continue
                        actual_target = target
                    candidate = {**candidate,"target_px":actual_target}
                    candidate_pixels = mesh_view(view,actual_target,pose["contacts_px"])
                    def scale_at(location):
                        if not valid_zone(location, scene):
                            raise NeedsReview("actual location outside scene zone")
                        try:
                            return candidate_scale(row,location,candidate_pixels[0][:,:,3],scene,projection=projection)
                        except NeedsReview as issue:
                            if not allow_missing or getattr(issue,"code",None)!="insufficient_local_references":
                                raise
                            return {"status":"unknown","reason":str(issue),"target_px":actual_target}
                    try:
                        fit = open_fit(candidate,proposals,atlas,frame,candidate_pixels,
                                       source_visible_fraction=visibility["source_visible_fraction"],
                                       physics_stable=pose["physics_stable"],pixel_size_review=allow_missing,
                                       scale_at=scale_at,fixed_anchor=fixed_anchor)
                        with Image.open(view/"object.png") as main_image:
                            main_major=silhouette_axes(np.asarray(main_image.getchannel("A")))[0]
                        checks=[]
                        for endpoint,data,check in endpoint_poses:
                            with Image.open(endpoint/"object.png") as endpoint_image:
                                major=silhouette_axes(np.asarray(endpoint_image.getchannel("A")))[0]
                            endpoint_target=(actual_target*major/main_major*pose["camera"]["orthographic_scale"]
                                             /data["camera"]["orthographic_scale"])
                            endpoint_pixels=mesh_view(endpoint,endpoint_target,data["contacts_px"])
                            endpoint_candidate={**candidate,"target_px":endpoint_target}
                            endpoint_projection=scale_projection(row,data,endpoint)
                            def endpoint_scale(location):
                                try:
                                    return candidate_scale(row,location,endpoint_pixels[0][:,:,3],scene,
                                                           projection=endpoint_projection)
                                except NeedsReview as issue:
                                    if not allow_missing or getattr(issue,"code",None)!="insufficient_local_references":
                                        raise
                                    return {"status":"unknown","reason":str(issue),"target_px":endpoint_target}
                            endpoint_fit=open_fit(endpoint_candidate,proposals,atlas,frame,endpoint_pixels,
                                                  source_visible_fraction=check["source_visible_fraction"],
                                                  physics_stable=data["physics_stable"],pixel_size_review=allow_missing,
                                                  fixed_anchor=fit["support_anchor_px"],scale_at=endpoint_scale)
                            checks.append({"elevation_deg":data["camera"]["elevation_deg"],
                                           "source_visible_fraction":check["source_visible_fraction"],
                                           "support_anchor_px":endpoint_fit["support_anchor_px"],
                                           "component_support":endpoint_fit["evidence"]["component_support"],
                                           "scale":endpoint_fit["scale"]})
                        fit["camera_endpoint_checks"]=checks
                        fits.append(fit)
                    except NeedsReview as issue:
                        records.append({"candidate":candidate["id"],"reason":str(issue)})
                print("placement",allow_missing,len(fits),len(candidates),records,flush=True)
                if fits:
                    break
            if not fits:
                raise NeedsReview("no supported and clear inspection location")
            fits.sort(key=lambda fit: (fit["scale"]["status"]!="unknown",fit["rank"]), reverse=True)
            mcp = args.mcp.resolve()
            with tempfile.TemporaryDirectory(prefix="photoshop-", dir=output) as stage:
                scratch = Path(stage)
                source_path = scratch/background.name
                shutil.copy2(background, source_path)
                with (output/"mcp.log").open("a",encoding="utf8") as log:
                    async with stdio_client(StdioServerParameters(command=str(mcp)),errlog=log) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await asyncio.wait_for(session.initialize(), timeout=30)
                            runner.session = session
                            reference = read_image(await runner.reference(source_path.as_posix(), background, scratch))
                            for fit in fits[:3]:
                                spec, photo = compose_spec(fit, frame, scratch/fit["candidate"]["id"], fit["candidate"]["id"])
                                spec["protected"] = scene["protected"]
                                outcomes = await runner.ps({"mode":"compose","source":source_path.as_posix(),"candidates":[spec]})
                                try:
                                    rendered, measured = validate(reference,spec,outcomes[0],fit["candidate"]["target_px"])
                                    actual = measured["rendered_major_px"]
                                    if not .95*target <= actual <= 1.05*target:
                                        raise NeedsReview("actual pixel size exceeds approved range")
                                    if fixed_anchor is not None and abs(actual-baseline_record["projection"]["rendered_major_px"])>1.5:
                                        raise NeedsReview("物品尺寸与上一版相差超过 1.5 px")
                                    if fit["scale"]["status"] != "unknown":
                                        low,high = fit["scale"]["projected_major_px_range"]
                                        if not low <= actual <= high:
                                            raise NeedsReview("actual pixel size exceeds estimated local range")
                                    alpha=render_alpha(spec,"object_alpha")
                                    crop_rgb=np.asarray(srgb(rendered.crop(tuple(spec["crop"]))))
                                    post=statistics(np.ascontiguousarray(crop_rgb),alpha>=190)
                                    if post is None:
                                        raise NeedsReview("成图物品像素不足以测量光色")
                                    photo["object_after"]=post[0]
                                    scene_stats=photo.get("scene")
                                    if scene_stats:
                                        luma_gap=abs(post[0]["luma_p20_p50_p80"][1]-scene_stats["luma_p20_p50_p80"][1])
                                        contrast_gap=abs(post[0]["contrast_p80_p20"]-scene_stats["contrast_p80_p20"])
                                        photo["improvement_vs_previous"]={"luma_gap":luma_gap,"contrast_gap":contrast_gap,
                                            "luma_target_met":luma_gap<=33*.75,
                                            "contrast_target_met":contrast_gap<=38*.75}
                                except (NeedsReview, RuntimeError) as issue:
                                    records.append({"candidate":fit["candidate"]["id"],"reason":str(issue)})
                                    continue
                                break
                            else:
                                raise NeedsReview("Photoshop checks rejected the inspection candidates")
                shutil.copy2(spec["png"], output/"03-轴承座-检查全图.png")
                shutil.copy2(spec["psd"], output/"03-轴承座-检查分层.psd")
            baseline=read_image(baseline_dir/"03-轴承座-检查全图.png")
            prior=baseline_record
            l,t,r,b = measured["bounds_px"]
            old=prior["projection"]["bounds_px"]
            crop=(max(0,min(l,old[0])-56),max(0,min(t,old[1])-56),
                  min(frame.width,max(r,old[2])+56),min(frame.height,max(b,old[3])+56))
            holes,_=interior_holes(reference_mask)
            import cv2
            count,_,stats,_=cv2.connectedComponentsWithStats(holes.astype("uint8"),8)
            if count<=1:
                raise NeedsReview("源图缺少可核验的中央孔")
            largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            x,y,w,h=map(int,stats[largest,:4])
            feature=(max(0,x-40),max(0,y-40),min(512,x+w+40),min(512,y+h+40))
            rust=read_image(Path(row["rust"]),"RGBA")
            rust=rust.crop(rust.getchannel("A").getbbox())
            rust.thumbnail((435,435),Image.Resampling.LANCZOS)
            normalized=Image.new("RGBA",(512,512))
            normalized.alpha_composite(rust,((512-rust.width)//2,(512-rust.height)//2))
            def enlarged(image,width,height):
                factor=min(width/image.width,height/image.height)
                return image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))),Image.Resampling.LANCZOS)
            board = Image.new("RGB",(1800,1530),(29,32,38))
            tile(board,rendered,0,0,900,450,"03 / 3D placement / human review")
            tile(board,enlarged(normalized.crop(feature),290,420),900,0,300,450,"Rust source / bearing detail")
            tile(board,enlarged(read_image(source_view).crop(feature),290,420),1200,0,300,450,"Mesh / bearing detail")
            tile(board,read_image(Path(row["original"])),1500,0,300,450,"Original object")
            tile(board,enlarged(baseline.crop(crop),885,560),0,450,900,600,"Previous / same scene coordinates")
            tile(board,enlarged(rendered.crop(crop),885,560),900,450,900,600,"New / same scene coordinates")
            hole_ids=sorted(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)
            if len(hole_ids)<3:
                raise NeedsReview("源图缺少可确认的三个贯通孔")
            central=hole_ids[0]
            mounts=sorted(hole_ids[1:3],key=lambda i:stats[i,cv2.CC_STAT_LEFT])
            regions=[]
            for label,index in (("Bearing rings",central),("Left mounting hole",mounts[0]),
                                ("Right mounting hole",mounts[1])):
                x,y,w,h=map(int,stats[index,:4])
                pad=max(18,round(max(w,h)*.5))
                regions.append((label,(max(0,x-pad),max(0,y-pad),min(512,x+w+pad),min(512,y+h+pad))))
            yy,xx=np.nonzero(reference_mask>=128)
            top=int(yy.min())
            top_x=xx[yy<=top+8]
            center=int(np.median(top_x))
            regions.append(("Top fitting",(max(0,center-35),max(0,top-8),min(512,center+36),min(512,top+62))))
            source_mesh=read_image(source_view)
            for index,(label,region) in enumerate(regions):
                x=index*450
                tile(board,enlarged(normalized.crop(region),435,210),x,1050,450,240,label+" / source")
                tile(board,enlarged(source_mesh.crop(region),435,210),x,1290,450,240,label+" / mesh")
            board.save(output/"03-轴承座-检查效果图.jpg",quality=94)
            interval_status=("pose_support_and_scale_checked" if all(
                item["scale"]["status"]!="unknown" for item in fit["camera_endpoint_checks"])
                else "pose_support_passed_scale_pending")
            result = {"status":"human_inspection","number":row["number"],
                      "target_px":fit["candidate"]["target_px"],"approved_target_px":target,
                      "mesh":{"sha256":digest(Path(pose_request["mesh"])),"source_fidelity":fidelity,
                              "proportions":proportion,"unobserved_surface":"source-projected texture is inferred",
                              "material":material},
                      "projection":measured,"geometry":fit["evidence"],"visibility":visibility,
                      "support_anchor_px":fit["support_anchor_px"],"camera_endpoint_checks":fit["camera_endpoint_checks"],
                      "baseline_record":str(args.baseline) if args.baseline else None,
                      "pose":pose["camera"],"scene_scale":fit["scale"],"photography":photo,
                      "review":{"structure_v2":"pending","visual_model":"pending",
                                "camera_interval":interval_status,
                                "scale_projection":"orthographic_horizontal_plane_estimate"},
                      "rejections":records,"token_usage":runner.budget.image_usage(row)}
            atomic_json(output/"result.json",result)
            print("review ready",result,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item",choices=["03"],default="03")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--pose",type=Path,required=True)
    parser.add_argument("--source-view",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,help="固定上一版检查图的接触位置和尺寸")
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--mcp",type=Path,required=True)
    parser.add_argument("--history",type=Path,default=Path("outputs/固定场景-v2最终几何复测"))
    parser.add_argument("--approved-from",type=Path,default=Path("outputs/固定场景-空间姿态试产"))
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("batch-config.example.json"))
    args=parser.parse_args()
    asyncio.run(inspect(args))


if __name__ == "__main__":
    main()
