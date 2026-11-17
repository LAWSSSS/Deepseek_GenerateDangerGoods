"""Make human sizing samples from one fixed scene and a photographed asset catalog."""
import argparse
import asyncio
import csv
import html
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import Field

from asset_catalog import Catalog
from batch_v2 import Runner, lock_directory, render_alpha, save_mask, verify_background_pixels
from metric_scale import silhouette_axes
from photometry import fit_photo
from run import Config, Strict, load
from scene_v2 import (NeedsReview, Perception, angle_delta, atomic_json, boundary_agreement,
                      contact_band, digest, gap_planes, key, original_edge_neighborhood,
                      posed_alpha, read_image, wear_sample_path)


HERE = Path(__file__).resolve().parent
REVISION = 1
Pixel = Annotated[int,Field(ge=0,le=16000)]
Rect = tuple[Pixel,Pixel,Pixel,Pixel]
Point = tuple[Pixel,Pixel]
TEMP_ERROR = (OSError, RuntimeError, ValueError, TimeoutError, NeedsReview)


class Scene(Strict):
    background: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    slot: Rect
    seat: Point
    front: Point
    px_per_mm: float = Field(gt=0,le=3)
    axis_deg: int = Field(ge=-45,le=45)
    provisional_major_px: dict[str,Annotated[int,Field(ge=32,le=3000)]]


def scene_masks(engine, source, frame, scene):
    w,h=frame.size
    l,t,r,b=scene.slot
    points=(scene.seat,scene.front)
    if not (0<=l<r<=w and 0<=t<b<=h and all(l<=x<r and t<=y<b for x,y in points)):
        raise ValueError("场景裁区和锚点须在同一张原图内")
    cache=engine.cache/f"sizing-site-{key(REVISION,scene.sha256,scene.slot,points,engine.versions)}.npz"
    if cache.is_file():
        with np.load(cache) as saved:
            return saved["support"],saved["foreground"]
    _,depth,_,_=engine.scene(source)
    region=np.s_[t:b,l:r]
    local_depth=cv2.resize(depth,frame.size,interpolation=cv2.INTER_LINEAR)[region]
    local_pts=[(x-l,y-t) for x,y in points]
    options=engine.seeded_masks(np.asarray(frame.crop(scene.slot)),local_pts)
    masks=list(gap_planes(local_depth,options,local_pts,390))
    if not masks:
        raise NeedsReview("固定场景落点的钢件前后关系无法确认")
    seat,front=max(masks,key=lambda pair:np.count_nonzero(pair[0]))
    support=np.zeros((h,w),bool)
    foreground=np.zeros((h,w),np.uint8)
    support[region]=seat
    solid=cv2.morphologyEx(front.astype("uint8")*255,cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    foreground[region]=cv2.GaussianBlur(solid,(3,3),.60)
    temp=cache.with_suffix(".tmp.npz")
    np.savez_compressed(temp,support=support,foreground=foreground)
    temp.replace(cache)
    return support,foreground


async def trial_asset(runner, source, record, scratch):
    if record.object_size is not None:
        try:
            return await runner.asset(source,scratch),None
        except NeedsReview as error:
            note=str(error)
    else:
        note=record.note or "尺寸资料待核"
    cache=runner.cache/f"sizing-asset-{key(REVISION,record.sha256,runner.engine.versions[2],digest(wear_sample_path()))}"
    cache.mkdir(exist_ok=True)
    aged=cache/"aged.png"
    if not aged.is_file():
        cutout,wear,preworn=await asyncio.to_thread(runner.engine.matte,source,allow_partial=True)
        matte=scratch/"manual-matte.png"
        texture=scratch/"manual-wear.png"
        cutout.save(matte)
        wear.save(texture)
        staged=cache/"staged.png"
        await runner.ps(dict(mode="asset",matte=matte.as_posix(),wear=texture.as_posix(),
                             preworn=preworn,png=staged.as_posix(),psd=(cache/"aged.psd").as_posix()))
        staged.replace(aged)
    return aged,note


def major_target(record, scene):
    if record.object_size is None:
        if record.file not in scene.provisional_major_px:
            raise NeedsReview("请为无可靠尺寸的素材给出一个试摆长轴")
        return scene.provisional_major_px[record.file],"manual_pixel_guess"
    a,b=record.object_size.major_mm
    return round(math.sqrt(a*b)*scene.px_per_mm),record.object_size.basis


def choose_position(silhouette, frame_size, support, foreground, seat):
    h,w=silhouette.shape
    fw,fh=frame_size
    safe_top=math.ceil(.09*fh)+8
    safe_bottom=math.floor(.82*fh)-8
    if w>fw-24 or h>safe_bottom-safe_top:
        raise NeedsReview("按试摆尺寸无法完整放入安全画面")
    sx,sy=seat
    reach=min(125,max(28,round(h*.29)))
    solid=silhouette>=128
    band=contact_band(solid,w)
    mass=silhouette.astype("float32")/255
    size=float(mass.sum())
    near=cv2.dilate(support.astype("uint8"),np.ones((9,9),np.uint8))>0
    best=None
    for ox in (-.48,-.32,-.20,-.10,0,.12,.24,.39):
        x=min(fw-w-12,max(12,round(sx-w/2+w*ox)))
        for oy in (-1,-.50,0,.55,1,1.55):
            y=min(safe_bottom-h,max(safe_top,round(sy-h+42+oy*reach)))
            region=np.s_[y:y+h,x:x+w]
            cover=float(np.sum(mass*foreground[region])/(255*size))
            contact=np.count_nonzero(band & near[region])/max(1,np.count_nonzero(band))
            score=.46*contact-1.65*abs(cover-.24)-max(0,cover-.48)*3
            score-=.065*abs(ox)+.022*abs(oy)
            if best is None or score>best[0]:
                best=(score,x,y,cover,contact)
    return best[1:],band


def specification(asset, frame, support, foreground, scene, major, scratch, *,
                  positioner=choose_position, angle=None, blur_limit=None, trim_prepared=False):
    rgba=read_image(asset,"RGBA")
    raw=np.asarray(rgba.getchannel("A"))
    trim_mask=None
    if trim_prepared:
        # Ignore low-alpha haze around generated inputs while keeping real antialiased edges.
        distance=cv2.distanceTransform((raw<128).astype(np.uint8),cv2.DIST_L2,3)
        keep=(raw>=8) & ((raw>=110) | (distance<=max(3,round(max(rgba.size)*.003))))
        if ((raw>0) & ~keep).any():
            raw=np.where(keep,raw,0).astype(np.uint8)
            rgba.putalpha(Image.fromarray(raw))
            marker=Image.new("RGBA",rgba.size,(255,255,255,0))
            marker.putalpha(Image.fromarray(keep.astype(np.uint8)*255))
            trim_mask=scratch/"keep-alpha.png"
            marker.save(trim_mask)
    long,short,direction=silhouette_axes(raw)
    if angle is None:
        angle=round(angle_delta(scene.axis_deg,direction)) if long/short>1.24 else 0
    silhouette,_=posed_alpha(raw,None,angle,(rgba.width//2,rgba.height//2),pixel_scale=major/long)
    h,w=silhouette.shape
    (x,y,cover,contact),band=positioner(silhouette,frame.size,support,foreground,scene.seat)
    fw,fh=frame.size
    padding=max(70,round(max(w,h)*.18))
    l,t,r,b=max(0,x-padding),max(0,y-padding),min(fw,x+w+padding),min(fh,y+h+padding)
    crop=(l,t,r,b)
    mask=support[t:b,l:r]
    occlusion=foreground[t:b,l:r].copy()
    patch=frame.crop(crop)
    placed=np.zeros((b-t,r-l),np.uint8)
    region=np.s_[y-t:y-t+h,x-l:x-l+w]
    placed[region]=silhouette
    # Close gaps in the contact shadow only around supported pixels.
    foot=np.zeros_like(placed)
    foot[region]=band*210
    shadow=cv2.dilate(foot,np.ones((max(3,round(w*.018)),max(3,round(w*.031))),np.uint8))
    shadow=cv2.GaussianBlur(shadow,(0,0),min(9,max(1.1,w*.011)))*mask
    shadow[:max(0,math.ceil(.09*fh)-t)]=0
    shadow[max(0,math.floor(.82*fh)-t):]=0
    target=mask & (placed<128) & (occlusion<128)
    if target.sum()<200:
        target=(placed<128) & (occlusion<128)
    tone,photography,grain=fit_photo(patch,target,rgba,angle,silhouette.shape)
    if blur_limit is not None:
        tone["blur_px"]=min(blur_limit,tone["blur_px"])
    if tone["blur_px"]<.15:
        tone["blur_px"]=0
    edge=original_edge_neighborhood(np.asarray(patch))
    edge_score=boundary_agreement(placed,occlusion,edge)
    use_front=cover>=.025 and edge_score>=.48
    if use_front:
        margin=4
        keep=(max(0,x-l-margin),max(0,y-t-margin),min(r-l,x-l+w+margin),min(b-t,y-t+h+margin))
        kl,kt,kr,kb=keep
        # Restrict the input mask to the area the object and contact shadow can affect.
        local=np.zeros_like(occlusion)
        local[kt:kb,kl:kr]=occlusion[kt:kb,kl:kr]
        occluder=save_mask(local,scratch/"foreground.png",crop,(255,255,255))
    else:
        occluder=None
    spec={"id":scratch.name,"asset":asset.as_posix(),"center":[x+w/2,y+h/2],"width":w,
          "angle":angle,"canvas":[fw,fh],"crop":list(crop),"tone":tone,
          "shade":save_mask(shadow,scratch/"shade.png",crop,(11,11,13)),"occlusion":occluder,
          "patch_png":(scratch/"patch.png").as_posix(),"object_alpha":(scratch/"alpha.png").as_posix()}
    if occluder is not None:
        spec["foreground_alpha"]=(scratch/"front-alpha.png").as_posix()
    if trim_mask is not None:
        spec["asset_trim_mask"]=trim_mask.as_posix()
    if grain is not None:
        grain_path=scratch/"grain.png"
        Image.fromarray(grain).convert("RGB").save(grain_path)
        spec["grain"]=grain_path.as_posix()
    return spec,{"planned_support":round(contact,3),"photography":photography}


def save_render(original, spec, result, folder, source):
    if "error" in result:
        raise RuntimeError(result["error"])
    l,t,r,b=spec["crop"]
    patch=read_image(Path(spec["patch_png"]))
    if patch.size!=(r-l,b-t):
        raise RuntimeError("Photoshop 裁区尺寸与原图坐标不符")
    output=original.copy()
    output.paste(patch,(l,t))
    verify_background_pixels(np.asarray(original),np.asarray(output),result["bounds"],spec)
    alpha=render_alpha(spec,"object_alpha")
    axis,cross,_=silhouette_axes(alpha)
    cover=0.
    if "foreground_alpha" in spec:
        foreground=render_alpha(spec,"foreground_alpha")
        cover=float((alpha.astype("float32")*(foreground/255)).sum()/alpha.sum())
    target=folder/"合成.png"
    pending=folder/"合成.tmp.png"
    output.save(pending,compress_level=1)
    pending.replace(target)
    output.thumbnail((1280,720),Image.Resampling.LANCZOS)
    output.save(folder/"全图.jpg",quality=89)
    patch=patch.resize((round(patch.width/2),round(patch.height/2)),Image.Resampling.LANCZOS)
    patch.save(folder/"局部.jpg",quality=92)
    asset=read_image(source)
    asset.thumbnail((220,145),Image.Resampling.LANCZOS)
    asset.save(folder/"素材.jpg",quality=89)
    return {"rendered_major_px":round(axis,1),"rendered_minor_px":round(cross,1),
            "bounds_px":result["bounds"],"center_px":spec["center"],"clockwise_deg":spec["angle"],
            "occlusion_ratio":round(cover,4),"local_preview_px":list(patch.size)}


def write_index(out, rows, scene, canvas):
    rows=sorted(rows,key=lambda row:row["number"])
    summary={"background_sha256":scene.sha256,"canvas_px":list(canvas),"trial_px_per_mm":scene.px_per_mm,
             "items":len(rows),"rendered":sum(r["status"]=="size_review" for r in rows),"model_calls":0}
    atomic_json(out/"summary.json",summary)
    pending=out/"results.jsonl.tmp"
    pending.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    pending.replace(out/"results.jsonl")
    fields=("number","asset_id","file","status","size_basis","planned_major_px","rendered_major_px",
            "judgment","suggested_major_px","notes")
    form=out/"人工尺寸标注.csv"
    old={}
    if form.exists():
        with form.open(encoding="utf-8-sig",newline="") as stream:
            old={entry["asset_id"]:entry for entry in csv.DictReader(stream)}
    annotated=[]
    for row in rows:
        before=old.get(row["asset_id"],{})
        if before.get("rendered_major_px")==str(row.get("rendered_major_px","")):
            row={**row,**{k:before.get(k,"") for k in ("judgment","suggested_major_px","notes")}}
        annotated.append(row)
    pending=out/"人工尺寸标注.csv.tmp"
    with pending.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(annotated)
    pending.replace(form)
    cards=[]
    jump=[]
    basis={"manual_pixel_guess":"无可信尺寸，暂定像素","similar_catalog":"同类尺寸估算",
           "exact_catalog":"型号有出处，画面尺度待标","measured":"物品实测，画面尺度待标"}
    w,h=canvas
    for row in rows:
        n=row["number"]
        label=f'{n:02} · {html.escape(row["file"])}'
        jump.append(f'<a href="#item-{n}">{label}</a>')
        if row["status"]!="size_review":
            cards.append(f'<article id="item-{n}"><h2>{label}</h2><p>{html.escape(row["reason"])}</p></article>')
            continue
        l,t,r,b=row["bounds_px"]
        box=f"left:{100*l/w:.3f}%;top:{100*t/h:.3f}%;width:{100*(r-l)/w:.3f}%;height:{100*(b-t)/h:.3f}%"
        f=html.escape(row["folder"],quote=True)
        note=html.escape(row.get("material_note") or "")
        occlusion=f'{row["occlusion_ratio"]:.0%}'
        cards.append(f"""<article id="item-{n}" data-number="{n}">
<h2>{label}</h2><p>完整长轴 <b>{row["rendered_major_px"]:.0f} px</b> · 遮挡 {occlusion} · {basis[row["size_basis"]]}
<a href="{f}/合成.png">原尺寸 PNG</a></p><p>{note}</p>
<div class="panels"><a class="view" href="{f}/合成.png"><img loading="lazy" src="{f}/全图.jpg" alt="合成全图"><i style="{box}"></i></a>
<div><img loading="lazy" src="{f}/素材.jpg" alt="输入素材"><div class="zoom"><img loading="lazy" src="{f}/局部.jpg" alt="原图二分之一倍率的局部"></div></div></div>
<label>判断 <select name="judgment"><option value="">未标注</option><option value="ok">合适</option><option value="small">偏小</option><option value="large">偏大</option><option value="unclear">看不清</option></select></label>
<label>建议完整长轴 px <input name="suggested_major_px" type="number" min="1" max="5000"></label>
<label>备注 <input name="notes" size="25"></label></article>""")
    page=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>固定场景 · 异物尺寸标注</title>
<style>body{{font:16px system-ui;margin:24px;background:#13181d;color:#ecf0f4}}article,header{{padding:16px;margin-bottom:16px;background:#1e252c;border-radius:9px}}a{{color:#9bd4ff}}.panels{{display:grid;grid-template-columns:minmax(0,1fr) minmax(250px,1fr);gap:12px}}.view{{position:relative;align-self:start;display:block}}.view img{{width:100%;display:block}}i{{position:absolute;border:2px solid #f5d562;box-sizing:border-box}}.no-box i{{display:none}}.zoom{{max-width:100%;overflow:auto;margin:5px 0}}.zoom img{{display:block;max-width:none}}label{{margin-right:8px;display:inline-block}}p{{margin:.5em 0}}button,input,select{{font:inherit}}@media(max-width:780px){{.panels{{grid-template-columns:1fr}}}}</style>
<header><h1>固定场景尺寸标注</h1><p>{summary["rendered"]}/{len(rows)} 张插入图。全图统一比例，局部统一为原尺寸的 1/2。定位框标完整轮廓范围。</p>
<label><input type="checkbox" id="box" checked>显示定位框</label>
<button id="save">导出已标注 JSONL</button> <a href="人工尺寸标注.csv">CSV 标注表</a> · <a href="固定背景.jpg">查看原背景</a> <span id="count"></span>
<details><summary>按物品跳转</summary><p>{'　'.join(jump)}</p></details></header>
{''.join(cards)}
<script>
const rows={json.dumps([dict(number=r["number"],asset_id=r["asset_id"],rendered_major_px=r.get("rendered_major_px")) for r in rows])};
const scene="{scene.sha256}";
const articles=[...document.querySelectorAll("article[data-number]")];
const storage="scale-trial:"+scene+":{scene.px_per_mm}";
let saved={{}};
try{{saved=JSON.parse(localStorage.getItem(storage)||"{{}}");}}catch{{}}
function count(){{document.querySelector("#count").textContent="已标注 "+Object.values(saved).filter(v=>v.judgment).length+" 张";}}
for(const a of articles){{
 const n=a.dataset.number;
 for(const field of a.querySelectorAll("[name]")){{
  field.value=saved[n]?.[field.name]??"";
  field.addEventListener("input",()=>{{(saved[n]??={{}})[field.name]=field.value;try{{localStorage.setItem(storage,JSON.stringify(saved));}}catch{{}}count();}});
 }}
}}
document.querySelector("#box").onchange=e=>document.body.classList.toggle("no-box",!e.target.checked);
document.querySelector("#save").onclick=()=>{{
 const lines=rows.filter(r=>saved[r.number]?.judgment).map(r=>JSON.stringify({{...r,...saved[r.number],background_sha256:scene,suggested_major_px:Number(saved[r.number].suggested_major_px)||null}}));
 const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([lines.join("\\n")+"\\n"],{{type:"application/x-ndjson"}}));link.download="人工尺寸反馈.jsonl";link.click();
}};
count();
</script>"""
    (out/"index.html").write_text(page,encoding="utf-8")


async def run(args):
    scene=load(Scene,args.scene)
    background=(args.scene.parent/scene.background).resolve(strict=True)
    if digest(background)!=scene.sha256:
        raise ValueError("固定背景的内容和配置哈希不符")
    catalog=Catalog(args.catalog.resolve())
    engine=Perception(args.weights.resolve(),args.cache)
    frame=read_image(background)
    preview=frame.copy()
    preview.thumbnail((1280,720),Image.Resampling.LANCZOS)
    preview.save(args.out/"固定背景.jpg",quality=92)
    support,foreground=scene_masks(engine,background,frame,scene)
    records=list(catalog.by_id.values())
    selected=[(n,r) for n,r in enumerate(records,1) if not args.only or r.file in args.only]
    if not selected or args.only and len(selected)!=len(set(args.only)):
        raise ValueError("指定素材不存在于物品库")
    all_rows=[]
    with tempfile.TemporaryDirectory(prefix="ps-size-") as temp, (args.out/"mcp.log").open("a",encoding="utf-8") as log:
        scratch=Path(temp)
        source=scratch/background.name
        shutil.copyfile(background,source)
        server=StdioServerParameters(command=str(args.mcp.resolve(strict=True)))
        async with stdio_client(server,errlog=log) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await asyncio.wait_for(session.initialize(),timeout=30)
                runner=Runner(session,None,Config(),engine,args.cache,None,catalog=catalog)
                reference=await runner.reference(source.as_posix(),background,scratch)
                original=read_image(reference)
                for start in range(0,len(selected),4):
                    pending=[]
                    for n,record in selected[start:start+4]:
                        signature=key(REVISION,scene.model_dump(),record.model_dump(mode="json"),engine.versions)
                        subdir=f"items/{n:02}-{record.asset_id}"
                        folder=args.out/subdir
                        folder.mkdir(parents=True,exist_ok=True)
                        row={"schema_version":1,"number":n,"asset_id":record.asset_id,"file":record.file,
                             "status":"failed","folder":subdir,"background_sha256":scene.sha256,"signature":signature}
                        marker=folder/"result.json"
                        if marker.exists() and (folder/"合成.png").is_file():
                            old=json.loads(marker.read_text(encoding="utf-8"))
                            if old.get("signature")==signature and old["status"]=="size_review":
                                all_rows.append(old)
                                continue
                        work=scratch/f"{n:02}"
                        work.mkdir()
                        try:
                            path,_,_=catalog.resolve(record.asset_id,Path.cwd())
                            target,basis=major_target(record,scene)
                            row.update(planned_major_px=target,size_basis=basis,
                                       object_size=record.object_size.model_dump(mode="json") if record.object_size else None)
                            asset,note=await trial_asset(runner,path,record,work)
                            spec,metrics=specification(asset,frame,support,foreground,scene,target,work)
                            row.update(metrics,material_note=note)
                            pending.append((spec,row,folder,path))
                        except TEMP_ERROR as error:
                            row["reason"]=str(error)[:260]
                            atomic_json(marker,row)
                            all_rows.append(row)
                            print(f"{n:02} {record.file} FAILED {row['reason']}",flush=True)
                    if pending:
                        payload=dict(mode="compose",source=source.as_posix(),candidates=[p[0] for p in pending])
                        try:
                            results=await runner.ps(payload)
                        except TEMP_ERROR as error:
                            results=[{"error":str(error)} for _ in pending]
                        for (spec,row,folder,path),result in zip(pending,results,strict=True):
                            try:
                                row.update(await asyncio.to_thread(save_render,original,spec,result,folder,path))
                                row["status"]="size_review"
                            except TEMP_ERROR as error:
                                row["reason"]=str(error)[:260]
                            all_rows.append(row)
                            atomic_json(folder/"result.json",row)
                            print(f'{row["number"]:02} {row["file"]} {row["status"]} '
                                  f'{row.get("rendered_major_px",row.get("reason",""))}',flush=True)
                    write_index(args.out,all_rows,scene,frame.size)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scene",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--weights",type=Path,required=True)
    cli.add_argument("--cache",type=Path,required=True)
    cli.add_argument("--catalog",type=Path,default=HERE/"asset-catalog.json")
    cli.add_argument("--mcp",type=Path,default=Path(os.environ.get("LOCALAPPDATA","."))/
                     "OpenAI/PhotoshopMCP/python/Scripts/photoshop-mcp-server.exe")
    cli.add_argument("--only",action="append",help="render only this catalog filename, repeatable")
    args=cli.parse_args()
    args.out=args.out.resolve()
    args.cache=args.cache.resolve()
    with lock_directory(args.cache),lock_directory(args.out):
        asyncio.run(run(args))


if __name__=="__main__":
    main()
