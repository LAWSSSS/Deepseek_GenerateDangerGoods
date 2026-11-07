"""Reproject source texture onto an existing mesh without changing geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from mesh_checks import source_fidelity
from mesh_detail import project_texture, render_source
from scene_v2 import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","asset","alpha","camera","out","blender"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--render-device",choices=("CPU","CUDA","OPTIX"),default="CPU")
    parser.add_argument("--texture-size",type=int,choices=(1024,2048),default=2048)
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,name,value.resolve())
    if (args.out/"result.json").exists():
        raise FileExistsError("纹理输出目录已有结果，请使用新目录")
    camera=json.loads(args.camera.read_text(encoding="utf-8-sig"))
    mesh=trimesh.load(args.mesh,force="mesh",process=False)
    textured=project_texture(mesh,args.asset,camera,texture_size=args.texture_size)
    execution=render_source(textured,args.out,camera,args.blender,args.render_device)
    saved=trimesh.load(args.out/"mesh.glb",force="mesh",process=False)
    if not np.array_equal(mesh.vertices,saved.vertices) or not np.array_equal(mesh.faces,saved.faces):
        raise ValueError("纹理导出改变了网格几何")
    with Image.open(args.alpha) as source,Image.open(args.out/"source-view.png") as rendered:
        fidelity=source_fidelity(np.asarray(source.convert("L")),np.asarray(rendered.getchannel("A")))
    if not fidelity["passed"]:
        raise ValueError("纹理网格源视图未通过轮廓和孔洞检查")
    result={"mesh":str(args.out/"mesh.glb"),"mesh_sha256":digest(args.out/"mesh.glb"),
            "source_mesh_sha256":digest(args.mesh),"source_asset_sha256":digest(args.asset),
            "source_view":str(args.out/"source-view.png"),"texture_size":args.texture_size,
            "detail_strength":.12,"detail_gain":.65,"normal_detail":"derived from source luminance",
            "unobserved_surface":"source-projected texture is inferred",
            "fidelity":fidelity,"source_render":execution}
    (args.out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
