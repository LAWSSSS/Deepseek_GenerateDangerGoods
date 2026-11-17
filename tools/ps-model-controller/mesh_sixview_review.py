"""Make an honest six-direction comparison with an uncalibrated reference sheet."""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from mesh_carve import interior_holes
from mesh_runtime import run_bounded, safe_tail
from scene_v2 import atomic_json, digest

HERE=Path(__file__).resolve().parent
VIEWS=("front","back","left","right","top","bottom")


def fit(image,size):
    image=image.convert("RGBA")
    image.thumbnail(size,Image.Resampling.LANCZOS)
    canvas=Image.new("RGBA",size,(255,255,255,255))
    canvas.alpha_composite(image,((size[0]-image.width)//2,(size[1]-image.height)//2))
    return canvas.convert("RGB")


def rough_reference_overlap(panel,model):
    """Diagnostic only: fit separate bounding boxes without camera calibration."""
    rgb=np.asarray(panel.convert("RGB"))
    mask=(rgb.min(axis=2)<220).astype("uint8")
    mask[:,:4]=0;mask[:,-4:]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count<2:return None
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    region=(labels==index).astype("uint8")
    region=cv2.morphologyEx(region,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    rx,ry,rw,rh,_=stats[index]
    rendered=(np.asarray(model.getchannel("A"))>=128)
    x,y,w,h=cv2.boundingRect(rendered.astype("uint8"))
    if min(w,h,rw,rh)==0:return None
    normalized=cv2.resize(region[ry:ry+rh,rx:rx+rw],(w,h),interpolation=cv2.INTER_NEAREST)>0
    target=np.zeros_like(rendered);target[y:y+h,x:x+w]=normalized
    union=np.logical_or(target,rendered).sum()
    return round(float(np.logical_and(target,rendered).sum()/union),4) if union else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("before","after","pose","reference","out","blender"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="OPTIX")
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"result.json").exists(): raise FileExistsError("请选择新的六视图对照目录")
    pose_request=json.loads((args.pose.parent/"request.json").read_text(encoding="utf8"))
    if digest(Path(pose_request["mesh"]))!=digest(args.after):
        raise ValueError("姿态与平滑后模型不匹配")
    request={"before":str(args.before),"after":str(args.after),"pose":str(args.pose),
             "out":str(args.out),"render_device":args.render_device}
    path=args.out/"request.json";path.write_text(json.dumps(request),encoding="utf8")
    execution=run_bounded([args.blender,"--background","--factory-startup","--python-exit-code","1",
                           "--python",HERE/"mesh_sixview_worker.py","--",path],
                          log=args.out/"blender.log",seconds=240)
    if execution["state"]!="ok":
        raise RuntimeError("六方向渲染失败: "+safe_tail(args.out/"blender.log",300))
    try: font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",18)
    except OSError: font=ImageFont.load_default()
    board=Image.new("RGB",(1800,990),(244,245,246));draw=ImageDraw.Draw(board)
    reference=Image.open(args.reference).convert("RGB")
    w,h=reference.size
    entries=[]
    for index,name in enumerate(VIEWS):
        x=index*300
        draw.text((x+8,7),name.upper(),font=font,fill=(20,25,30))
        col=index%3;row=index//3
        panel=reference.crop((round(col*w/3),round(row*h/2),round((col+1)*w/3),round((row+.86)*h/2)))
        board.paste(fit(panel,(290,265)),(x+5,32))
        images=[Image.open(args.out/f"{which}-{name}.png").convert("RGBA") for which in ("before","after")]
        masks=[np.asarray(image.getchannel("A"))>=128 for image in images]
        union=np.logical_or(*masks).sum()
        iou=float(np.logical_and(*masks).sum()/union) if union else 0.
        holes=[interior_holes(mask.astype("uint8")*255)[1] for mask in masks]
        angle=0.
        if name in ("top","bottom") and union:
            yy,xx=np.nonzero(masks[0]|masks[1])
            _,vectors=np.linalg.eigh(np.cov(np.column_stack((xx,yy)),rowvar=False))
            direction=vectors[:,-1]
            angle=float(np.degrees(np.arctan2(direction[1],direction[0])))
            images=[image.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True) for image in images]
        rough_overlap=rough_reference_overlap(panel,images[1])
        a,b=[image.getchannel("A").getbbox() for image in images]
        if a and b:
            margin=12
            rect=(max(0,min(a[0],b[0])-margin),max(0,min(a[1],b[1])-margin),
                  min(images[0].width,max(a[2],b[2])+margin),min(images[0].height,max(a[3],b[3])+margin))
            images=[image.crop(rect) for image in images]
        for image,y in zip(images,(330,630)):
            board.paste(fit(image,(290,265)),(x+5,y+25))
        entries.append({"direction":name,"before_after_silhouette_iou":round(iou,5),"visible_openings":holes,
                        "display_rotation_deg":round(angle,2),"uncalibrated_reference_bbox_overlap":rough_overlap})
        draw.text((x+8,305),f"原模型  |  IoU {iou:.4f}",font=font,fill=(20,25,30))
        draw.text((x+8,605),"平滑后",font=font,fill=(20,25,30))
    board.save(args.out/"03-六视图前后对照.jpg",quality=94)
    result={"status":"visual_comparison","reference":str(args.reference),"reference_sha256":digest(args.reference),
            "before_mesh_sha256":digest(args.before),"after_mesh_sha256":digest(args.after),
            "pose_record":str(args.pose),"views":entries,"render":execution,
            "limitation":"参考图未标定，模型方向为近似方向；显示的六视角 IoU 比较的是模型平滑前后。另记录的参考图粗略重合率采用各向缩放，仅供发现差异，不能作为几何精确验收或贴图依据。",
            "catalog_reference":{"manufacturer":"ASAHI","model_assumption":"照片标有 P210，按 UCP210 作比例参考，型号未独立确认",
                "housing_length_mm":206,"housing_width_mm":60,"mount_spacing_mm":159,
                "url":"https://www.asahiseiko.co.jp/system/download/datasheet/bearings/ucp.pdf"}}
    atomic_json(args.out/"result.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
