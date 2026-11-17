"""Re-render a fixed-scene sizing review with annotated scale and prepared rust PNGs."""
import argparse
import asyncio
import csv
import html
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from batch_v2 import Runner, lock_directory, render_alpha, verify_background_pixels
from metric_scale import silhouette_axes
from run import Config, load
from scale_trial import Scene, TEMP_ERROR, scene_masks, specification
from scene_v2 import (NeedsReview, atomic_json, boundary_agreement, contact_band, crossing,
                      digest, key, original_edge_neighborhood, Perception, read_image)


REVISION = 2
SCALE = {"偏小": 1.25, "small": 1.25, "偏大": .90, "略大": .90, "large": .90,
         "合适": 1., "恰当": 1., "ok": 1.}
MIN_COVER = .025
MAX_COVER = .48


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def pixel_target(label, before):
    prior = before["rendered_major_px"]
    if not math.isclose(float(label["rendered_major_px"]), prior, abs_tol=.05):
        raise ValueError(f"尺寸标注与前一版渲染不匹配：{label['number']}")
    if label["suggested_major_px"]:
        target = float(label["suggested_major_px"])
        if not 10 <= target <= 5000:
            raise ValueError(f"建议长轴超出范围：{label['number']}")
        return target, "annotated_px"
    return prior*SCALE[label["judgment"]], "judgment_multiplier"


def prepare(args, scene):
    old = {row["asset_id"]: row for row in read_rows(args.before/"results.jsonl")}
    inputs = {row["asset_id"]: row for row in read_rows(args.inputs/"素材清单.jsonl")}
    with (args.before/"人工尺寸标注.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = {row["asset_id"]: row for row in csv.DictReader(stream)}
    if old.keys() != inputs.keys() or old.keys() != labels.keys():
        raise ValueError("原批次、尺寸标注和高清素材必须包含同一组素材编号")
    fixed = args.out/"inputs/27-轮廓校正.json"
    correction = json.loads(fixed.read_text(encoding="utf-8"))
    jobs = []
    for asset_id, prior in old.items():
        number = prior["number"]
        row, label = inputs[asset_id], labels[asset_id]
        if prior["background_sha256"] != scene.sha256 or prior["status"] != "size_review":
            raise ValueError(f"前一版未使用相同背景或未完成渲染：{number}")
        if number != row["number"] or number != int(label["number"]):
            raise ValueError(f"素材编号不一致：{asset_id}")
        if args.only and number not in args.only:
            continue
        rust = row["variants"]["rust"]
        if asset_id == correction["asset_id"]:
            if correction["source_png_sha256"] != rust["sha256"] or correction["after"]["issues"]:
                raise ValueError("27 号校正版与锈蚀源图校验不匹配")
            asset = args.out/correction["file"]
            asset_hash = correction["sha256"]
        else:
            if rust["status"] != "input_ready":
                raise NeedsReview(f"高清锈蚀素材待审：{number}")
            asset = args.inputs/rust["file"]
            asset_hash = rust["sha256"]
        if digest(asset) != asset_hash:
            raise ValueError(f"高清锈蚀素材哈希不匹配：{number}")
        target, basis = pixel_target(label, prior)
        jobs.append((prior, label, asset, asset_hash, target, basis))
    if not jobs or args.only and len(jobs) != len(set(args.only)):
        raise ValueError("选择的编号不在原批次中")
    return sorted(jobs, key=lambda job: job[0]["number"])


def extra_foreground(out, scene, foreground):
    metadata = out/"inputs/前景补充.json"
    if not metadata.is_file():
        return None,set(),None
    entry = json.loads(metadata.read_text(encoding="utf-8"))
    path = out/entry["mask"]
    if entry["background_sha256"]!=scene.sha256 or entry["mask_sha256"]!=digest(path):
        raise ValueError("补充前景蒙版与固定场景校验不符")
    mask = np.asarray(read_image(path,"L"))
    if mask.shape!=foreground.shape or not mask.any():
        raise ValueError("补充前景蒙版的尺寸或透明度无效")
    return np.maximum(foreground,mask),set(entry["numbers"]),entry["mask_sha256"]


class RevisionPlacement:
    def __init__(self, support, foreground, frame):
        self.near = cv2.dilate(support.astype(np.uint8), np.ones((9,9),np.uint8)) > 0
        self.edge = original_edge_neighborhood(np.asarray(frame))
        self.foreground = foreground

    def place(self, silhouette, frame_size, seat, prior):
        h,w = silhouette.shape
        fw,fh = frame_size
        top = math.ceil(.09*fh)+8
        bottom = math.floor(.82*fh)-8
        if w > fw-24 or h > bottom-top:
            raise NeedsReview("调整后的物品轮廓超出可用画面")
        sx,sy = seat
        old_x,_ = prior["center_px"]
        old_bottom = prior["bounds_px"][3]
        xs = {old_x+w*f for f in (0,-.06,.06,-.13,.13,-.22,.22,-.33,.33)}
        xs.update(sx+w*f for f in (0,-.12,.12,-.24,.24,-.40,.40))
        ys = {old_bottom+d for d in (0,-45,45,-90,90,-140,140,210)}
        ys.update(sy+d for d in (0,38,72,115,160,210))
        lefts = {max(12,min(fw-w-12,round(x-w/2))) for x in xs}
        tops = {max(top,min(bottom-h,round(y-h))) for y in ys}
        solid = silhouette>=128
        band = contact_band(solid,w)
        mass = silhouette.astype(np.float32)/255
        step = max(2,round(max(w,h)/185))
        sample = mass[step//2::step,step//2::step]
        size = sample.sum()*255
        by,bx = np.nonzero(band)
        cover_target = max(.10,min(.27,prior["occlusion_ratio"]))
        contact_target = max(.10,min(.35,prior["planned_support"]*.53))
        candidates = []
        for x in sorted(lefts):
            for y in sorted(tops):
                area = np.s_[y+step//2:y+h:step,x+step//2:x+w:step]
                cover = float(np.sum(sample*self.foreground[area])/size)
                if not .019 <= cover <= MAX_COVER+.02:
                    continue
                contact = float(self.near[y+by,x+bx].mean())
                if contact < contact_target:
                    continue
                movement = abs(x+w/2-old_x)/max(110,w*.48) + abs(y+h-old_bottom)/270
                score = .56*contact-2*abs(cover-cover_target)-.085*movement
                candidates.append((score,x,y,contact))
        for _,x,y,contact in sorted(candidates,reverse=True):
            local = self.foreground[y:y+h,x:x+w]
            cover = float(np.sum(mass*local)/(255*mass.sum()))
            edge = self.edge[y:y+h,x:x+w]
            if (MIN_COVER <= cover <= MAX_COVER and boundary_agreement(silhouette,local,edge)>=.48
                    and crossing(silhouette,local)):
                return (x,y,cover,contact),band
        raise NeedsReview("原场景附近缺少同时满足承托和真实前景遮挡的落点")


def validate_render(reference, spec, result, row):
    if "error" in result:
        raise RuntimeError(result["error"])
    png,psd = Path(spec["png"]),Path(spec["psd"])
    if not png.is_file() or not psd.is_file():
        raise RuntimeError("Photoshop PNG 或分层 PSD 未生成")
    if "foreground_bounds" not in result:
        raise NeedsReview("Photoshop 未生成原像素前景层")
    rendered = read_image(png)
    verify_background_pixels(np.asarray(reference),np.asarray(rendered),result["bounds"],spec)
    alpha = render_alpha(spec,"object_alpha")
    front = render_alpha(spec,"foreground_alpha")
    cover = float(np.sum(alpha.astype(np.float32)*(front/255))/alpha.sum())
    if not MIN_COVER-.003 <= cover <= MAX_COVER+.003 or not crossing(alpha,front):
        raise NeedsReview(f"Photoshop 实际遮挡不合格：{cover:.1%}")
    major,minor,_ = silhouette_axes(alpha)
    target = row["planned_major_px"]
    if abs(major-target) > max(3,target*.05):
        raise NeedsReview(f"Photoshop 实测长轴 {major:.1f} px 与目标 {target:.1f} px 不符")
    old = row["before"]["rendered_major_px"]
    if target != old and (major-old)*(target-old) <= 0:
        raise NeedsReview("实测尺寸调整方向与标注不符")
    return rendered, {"rendered_major_px":round(major,1),"rendered_minor_px":round(minor,1),
                      "bounds_px":result["bounds"],"center_px":spec["center"],
                      "clockwise_deg":spec["angle"],"occlusion_ratio":round(cover,4),
                      "foreground_bounds_px":result["foreground_bounds"],
                      "change_percent":round(100*(major/old-1),1),
                      "protected_pixels_verified":True}


def publish(reference, spec, result, row, folder, before_png):
    after, measured = validate_render(reference,spec,result,row)
    before = read_image(before_png)
    if before.size != after.size:
        raise ValueError("前后对照需要相同画布尺寸")
    width,height = after.size
    a,b = row["before"]["bounds_px"],measured["bounds_px"]
    left,top = math.floor(min(a[0],b[0])),math.floor(min(a[1],b[1]))
    right,bottom = math.ceil(max(a[2],b[2])),math.ceil(max(a[3],b[3]))
    pad = max(48,round(.07*max(right-left,bottom-top)))
    crop = [max(0,left-pad),max(0,top-pad),min(width,right+pad),min(height,bottom+pad)]
    for name,image in (("旧版",before),("新版",after)):
        image.crop(crop).save(folder/f"{name}局部.png",compress_level=1)
        image.thumbnail((1280,720),Image.Resampling.LANCZOS)
        image.save(folder/f"{name}全图.jpg",quality=92,subsampling=0)
    Path(spec["png"]).replace(folder/"合成.png")
    Path(spec["psd"]).replace(folder/"分层.psd")
    return {**measured,"comparison_crop_px":crop}


def url(path, base):
    return quote(Path(os.path.relpath(path,base)).as_posix(),safe="/")


def comparison_html(out, rows, canvas, summary):
    cards = []
    jump = []
    width,height = canvas
    for row in rows:
        number = row["number"]
        title = f"{number:02}　{html.escape(row['file'])}"
        jump.append(f'<a href="#item-{number}">{number:02}</a>')
        if row["status"] != "size_review":
            cards.append(f'<article id="item-{number}"><h2>{title}</h2><p>{html.escape(row["reason"])}</p></article>')
            continue
        path = quote(row["folder"],safe="/")
        prior = row["before"]
        x,y,r,b = row["comparison_crop_px"]
        box = f"left:{100*x/width:.4f}%;top:{100*y/height:.4f}%;width:{100*(r-x)/width:.4f}%;height:{100*(b-y)/height:.4f}%"
        before_link = url(row["before_png"],out)
        asset_link = url(row["input_png"],out)
        cards.append(f"""<article id="item-{number}">
<h2>{title}</h2>
<p class="stats"><span>标注 <b>{html.escape(row["judgment"])}</b></span>
<span>完整长轴 <b>{prior["rendered_major_px"]:.1f} → {row["rendered_major_px"]:.1f} px</b>（{row["change_percent"]:+.1f}%）</span>
<span>前景遮挡 <b>{prior["occlusion_ratio"]:.1%} → {row["occlusion_ratio"]:.1%}</b></span></p>
<p class="links"><a href="{before_link}">旧版原尺寸 PNG</a><a href="{path}/合成.png">新版原尺寸 PNG</a>
<a href="{path}/分层.psd">新版分层 PSD</a><a href="{asset_link}">高清锈蚀输入</a></p>
<div class="pair">
<figure><figcaption>旧版全图</figcaption><a class="map" href="{before_link}"><img src="{path}/旧版全图.jpg" loading="lazy" alt="旧版全图"><i style="{box}"></i></a></figure>
<figure><figcaption>新版全图</figcaption><a class="map" href="{path}/合成.png"><img src="{path}/新版全图.jpg" loading="lazy" alt="新版全图"><i style="{box}"></i></a></figure>
</div>
<div class="pair crops">
<figure><figcaption>旧版局部</figcaption><div><a href="{path}/旧版局部.png"><img src="{path}/旧版局部.png" loading="lazy" alt="旧版同坐标局部"></a></div></figure>
<figure><figcaption>新版局部</figcaption><div><a href="{path}/新版局部.png"><img src="{path}/新版局部.png" loading="lazy" alt="新版同坐标局部"></a></div></figure>
</div></article>""")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>固定场景高清复测 · 前后对照</title>
<style>
body{{background:#131920;color:#edf0f4;font:15px system-ui;margin:18px auto;max-width:1540px;padding:0 14px}}
header,article{{background:#202a34;padding:16px;margin-bottom:14px;border-radius:10px}}
h1,h2,p{{margin:.45em 0}}a{{color:#a8d9fc}}.stats,.links,nav{{display:flex;flex-wrap:wrap;gap:8px 17px}}
.stats span{{background:#2c3c49;padding:4px 9px;border-radius:4px}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
figure{{margin:9px 0 0;min-width:0}}figcaption{{font-weight:650;margin:2px 0 5px}}
img{{display:block;width:100%;height:auto}}.map{{position:relative;display:block}}i{{position:absolute;border:2px solid #ffda69;box-sizing:border-box}}
.crops figure>div{{overflow:auto;max-height:710px;background:#151e27}}.crops img{{width:auto;max-width:100%}}
.native .crops img{{max-width:none}}
.hide-box i{{display:none}}.links a{{white-space:nowrap}}label{{margin-right:15px}}
</style>
<header><h1>固定场景高清复测</h1>
<p>{summary["rendered"]}/{len(rows)} 张已渲染。每组全图使用同一画布，局部裁区坐标完全一致，点击局部可看原像素。</p>
<p>未填目标像素时，偏小按完整长轴 +25%，略大 −10%，合适沿用原长轴。前景来自原背景钢件。</p>
<p><label><input id="boxes" type="checkbox" checked>显示局部位置</label><label><input id="native" type="checkbox">局部按 100% 显示</label></p>
<p class="links"><a href="复测标注.csv">复测标注表</a><a href="results.jsonl">结构化结果</a><a href="固定背景.jpg">原场景</a></p>
<nav>{''.join(jump)}</nav></header>
{''.join(cards)}
<script>
document.querySelector("#boxes").onchange=e=>document.body.classList.toggle("hide-box",!e.target.checked);
document.querySelector("#native").onchange=e=>document.body.classList.toggle("native",e.target.checked);
for(const pair of document.querySelectorAll(".crops")){{
  const [before,after]=pair.querySelectorAll("figure > div");
  const link=(from,to)=>from.addEventListener("scroll",()=>{{
    if(from.scrollLeft!==to.scrollLeft||from.scrollTop!==to.scrollTop)
      to.scrollTo(from.scrollLeft,from.scrollTop);
  }});
  link(before,after);
  link(after,before);
}}
</script></html>"""
    pending = out/"index.html.tmp"
    pending.write_text(page,encoding="utf-8")
    pending.replace(out/"index.html")


def write_index(out, rows, scene, canvas):
    rows = sorted(rows,key=lambda r:r["number"])
    completed = [row for row in rows if row["status"]=="size_review"]
    summary = {"background_sha256":scene.sha256,"canvas_px":list(canvas),"items":len(rows),
               "rendered":len(completed),"judgments":dict(Counter(row["judgment"] for row in rows)),
               "all_foreground_verified":bool(completed) and all(r["occlusion_ratio"]>0 for r in completed),
               "model_calls":0}
    atomic_json(out/"summary.json",summary)
    pending=out/"results.jsonl.tmp"
    pending.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    pending.replace(out/"results.jsonl")
    form = out/"复测标注.csv"
    notes = {}
    if form.is_file():
        with form.open(encoding="utf-8-sig",newline="") as stream:
            notes = {r["asset_id"]:r for r in csv.DictReader(stream)}
    fields = ("number","asset_id","file","judgment","old_major_px","planned_major_px",
              "rendered_major_px","change_percent","occlusion_ratio","size_review","sharpness_review","notes")
    with form.with_suffix(".tmp").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for row in completed:
            prev = notes.get(row["asset_id"],{})
            writer.writerow({**row,"old_major_px":row["before"]["rendered_major_px"],
                             **{k:prev.get(k,"") for k in ("size_review","sharpness_review","notes")}})
    form.with_suffix(".tmp").replace(form)
    comparison_html(out,rows,canvas,summary)


async def run(args):
    scene_path = args.before/"场景配置.json"
    scene = load(Scene,scene_path)
    background = (scene_path.parent/scene.background).resolve(strict=True)
    if digest(background) != scene.sha256:
        raise ValueError("固定背景内容与原批次配置不符")
    jobs = prepare(args,scene)
    frame = read_image(background)
    engine = Perception(args.weights.resolve(),args.cache)
    support,foreground = scene_masks(engine,background,frame,scene)
    placement = RevisionPlacement(support,foreground,frame)
    extended,large_items,mask_hash = extra_foreground(args.out,scene,foreground)
    alternate = RevisionPlacement(support,extended,frame) if extended is not None else None
    shutil.copyfile(args.before/"固定背景.jpg",args.out/"固定背景.jpg")
    rows = []
    with tempfile.TemporaryDirectory(prefix="ps-recheck-",dir=args.out) as temp, (args.out/"mcp.log").open("a",encoding="utf-8") as log:
        scratch = Path(temp)
        source = scratch/background.name
        shutil.copyfile(background,source)
        params = StdioServerParameters(command=str(args.mcp.resolve(strict=True)))
        async with stdio_client(params,errlog=log) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await asyncio.wait_for(session.initialize(),timeout=30)
                runner = Runner(session,None,Config(),engine,args.cache,None)
                reference = read_image(await runner.reference(source.as_posix(),background,scratch))
                for start in range(0,len(jobs),3):
                    pending = []
                    for prior,label,asset,asset_hash,target,basis in jobs[start:start+3]:
                        n,asset_id = prior["number"],prior["asset_id"]
                        before_png = args.before/prior["folder"]/"合成.png"
                        extra = n in large_items
                        signature = key(REVISION,scene.sha256,prior["signature"],label,asset_hash,target,
                                        *([mask_hash] if extra else []))
                        folder = args.out/prior["folder"]
                        folder.mkdir(parents=True,exist_ok=True)
                        marker = folder/"result.json"
                        if marker.exists():
                            saved = json.loads(marker.read_text(encoding="utf-8"))
                            files = ("合成.png","分层.psd","旧版全图.jpg","新版全图.jpg","旧版局部.png","新版局部.png")
                            if (saved["signature"]==signature and saved["status"]=="size_review"
                                    and all((folder/name).is_file() for name in files)):
                                rows.append(saved)
                                continue
                        row = {"number":n,"asset_id":asset_id,"file":prior["file"],"status":"failed",
                               "signature":signature,"folder":prior["folder"],"background_sha256":scene.sha256,
                               "judgment":label["judgment"],"planned_major_px":round(target,2),"target_basis":basis,
                               "input_png":asset.as_posix(),"input_sha256":asset_hash,"before_png":before_png.as_posix(),
                               "before":{k:prior[k] for k in ("rendered_major_px","occlusion_ratio","bounds_px","center_px","clockwise_deg")}}
                        work = scratch/f"{n:02}"
                        work.mkdir()
                        try:
                            local_placement = alternate if extra else placement
                            local_foreground = extended if extra else foreground
                            def place(silhouette,size,_support,_foreground,seat):
                                return local_placement.place(silhouette,size,seat,prior)
                            spec,metrics = specification(asset,frame,support,local_foreground,scene,target,work,
                                                         positioner=place,angle=prior["clockwise_deg"],
                                                         blur_limit=0,trim_prepared=True)
                            if not spec["occlusion"]:
                                raise NeedsReview("新轮廓与原背景前景的交集不足")
                            spec.pop("patch_png")
                            ps_asset = work/"object.png"
                            shutil.copyfile(asset,ps_asset)
                            spec.update(asset=ps_asset.as_posix(),png=(work/"final.png").as_posix(),
                                        psd=(work/"final.psd").as_posix())
                            row.update(metrics,tone=spec["tone"],extra_foreground=extra)
                            pending.append((spec,row,folder,before_png))
                        except TEMP_ERROR as error:
                            row["reason"] = str(error)[:300]
                            atomic_json(marker,row)
                            rows.append(row)
                            print(f"{n:02} planning failed {row['reason']}",flush=True)
                    if pending:
                        try:
                            results = await runner.ps(dict(mode="compose",source=source.as_posix(),
                                                           candidates=[p[0] for p in pending]))
                        except TEMP_ERROR as error:
                            results = [{"error":str(error)} for _ in pending]
                        for (spec,row,folder,before_png),result in zip(pending,results,strict=True):
                            try:
                                row.update(await asyncio.to_thread(publish,reference,spec,result,row,folder,before_png))
                                row["status"] = "size_review"
                            except TEMP_ERROR as error:
                                row["reason"] = str(error)[:300]
                            atomic_json(folder/"result.json",row)
                            rows.append(row)
                            print(f'{row["number"]:02} {row["status"]} {row.get("rendered_major_px",row.get("reason",""))} '
                                  f'{row.get("occlusion_ratio","")}',flush=True)
                    write_index(args.out,rows,scene,frame.size)
    return all(row["status"]=="size_review" for row in rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("before","inputs","out","weights","cache"):
        cli.add_argument(f"--{name}",type=Path,required=True)
    cli.add_argument("--mcp",type=Path,default=Path(os.environ.get("LOCALAPPDATA","."))/
                     "OpenAI/PhotoshopMCP/python/Scripts/photoshop-mcp-server.exe")
    cli.add_argument("--only",type=int,action="append")
    args = cli.parse_args()
    for name in ("before","inputs","out","cache"):
        setattr(args,name,getattr(args,name).resolve())
    with lock_directory(args.cache),lock_directory(args.out):
        return 0 if asyncio.run(run(args)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
