"""Choose and render one verified 03 pose over the scene elevation interval."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mesh_runtime import run_bounded
from mesh_visibility import SourceProjection, _fraction, coverage, fragments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","source-view","alpha","camera","out","blender"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="CPU")
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    args.out.mkdir(parents=True,exist_ok=True)
    camera=json.loads(args.camera.read_text(encoding="utf-8-sig"))
    with Image.open(args.source_view) as image,Image.open(args.alpha) as mask:
        source=np.asarray(image.getchannel("A"))
        reference=np.asarray(mask.convert("L"))
    deadline=time.monotonic()+600
    worker=Path(__file__).with_name("settled_mesh_worker.py")

    def run(elevation,yaw,name,require_coverage):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise TimeoutError("单件姿态诊断超过 600 秒")
        folder=args.out/name
        folder.mkdir(exist_ok=True)
        request=folder/"request.json"
        request.write_text(json.dumps({"mesh":str(args.mesh),"out":str(folder),"rotation_deg":[0,0,0],
                            "render_device":args.render_device,"render_resolution":1024,
                            "camera_azimuth_deg":yaw,"camera_elevation_deg":elevation}),encoding="utf8")
        result=run_bounded([args.blender,"--background","--factory-startup","--python-exit-code","1",
                            "--python",worker,"--",request],log=folder/"blender.log",
                           seconds=min(300,remaining))
        if result["state"]!="ok":
            raise RuntimeError(f"姿态渲染失败: {result['state']}")
        pose=json.loads((folder/"blender-result.json").read_text(encoding="utf8"))
        if not pose["physics_stable"]:
            raise ValueError("刚体未通过落稳检查")
        with Image.open(folder/"object.png") as image:
            visible=coverage(folder/"triangles.npz",camera,pose["camera"],source,
                             np.asarray(image.getchannel("A")),reference)
        print(json.dumps({"view":name,"visibility":visible,"execution":result},ensure_ascii=False),flush=True)
        if require_coverage and (not visible["known"] or visible["source_visible_fraction"]<.80):
            raise ValueError("姿态未通过源图可见面检查")
        return folder,pose

    initial,pose=run(45,90,"initial-45",False)
    with np.load(initial/"triangles.npz") as data:
        original,ids=data["source"],data["source_ids"]
        posed,pids=data["posed"],data["posed_ids"]
    projection=SourceProjection(original,camera,source,reference)
    center=np.asarray(pose["camera"]["target"])
    distance=np.linalg.norm(np.asarray(pose["camera"]["eye"])-center)
    ranked=[]
    for yaw in (75,80,85,90):
        values=[]
        hole=0
        for elevation in (35,45,55):
            if time.monotonic()>=deadline:
                raise TimeoutError("单件姿态诊断超过 600 秒")
            angle=np.deg2rad(yaw)
            tilt=np.deg2rad(elevation)
            horizontal=distance*np.cos(tilt)
            eye=center+np.array([horizontal*np.sin(angle),-horizontal*np.cos(angle),distance*np.sin(tilt)])
            view={**pose["camera"],"eye":eye.tolist(),"azimuth_deg":yaw}
            labels,_,bary=fragments(posed,view,128)
            value=_fraction(original,ids,posed,pids,projection,view,labels,bary)
            values.append(value or 0)
            if elevation==45:
                contours,hierarchy=cv2.findContours((labels>=0).astype("uint8"),
                                                     cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                hole=max([cv2.contourArea(c) for i,c in enumerate(contours)
                          if hierarchy[0,i,3]>=0],default=0)
        print(json.dumps({"yaw":yaw,"estimated_visibility":values,"hole_area":hole}),flush=True)
        if min(values)>=.80:
            ranked.append((hole,min(values),-abs(yaw-90),yaw))
    if not ranked:
        raise ValueError("没有同时满足俯角范围与源图可见面的方位")
    yaw=max(ranked)[-1]
    (args.out/"pose-selection.json").write_text(json.dumps({"yaw":yaw,"ranked":ranked},indent=2),encoding="utf8")
    for elevation in (45,35,55):
        run(elevation,yaw,f"pose-{elevation}",True)


if __name__=="__main__":
    main()
