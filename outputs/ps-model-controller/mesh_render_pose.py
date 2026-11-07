"""Render an already settled single-body pose after a material-only change."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
import blender_worker as worker


def geometry_signature(obj):
    mesh=obj.data
    vertices=np.empty(len(mesh.vertices)*3,np.float32)
    mesh.vertices.foreach_get("co",vertices)
    mesh.calc_loop_triangles()
    faces=np.asarray([list(t.vertices) for t in mesh.loop_triangles],dtype=np.int32)
    digest=hashlib.sha256()
    digest.update(vertices.tobytes())
    digest.update(faces.tobytes())
    digest.update(np.asarray(obj.matrix_world,dtype=np.float64).tobytes())
    return digest.hexdigest()


def main():
    args=sys.argv[sys.argv.index("--")+1:]
    request=json.loads(Path(args[0]).read_text(encoding="utf8"))
    previous=Path(request["settled_view"])
    old_request=json.loads((previous/"request.json").read_text(encoding="utf8"))
    pose=json.loads((previous/"blender-result.json").read_text(encoding="utf8"))
    if not pose.get("physics_stable") or pose.get("component_count")!=1:
        raise ValueError("Only a verified single-body pose can be reused")
    output=Path(request["out"])
    output.mkdir(parents=True,exist_ok=True)
    scene=worker.setup_scene(request.get("render_device","CPU"),request.get("render_resolution",512))
    old=worker.load_mesh(old_request["mesh"])
    expected=geometry_signature(old)
    bpy.data.objects.remove(old,do_unlink=True)
    obj=worker.load_mesh(request["mesh"])
    if geometry_signature(obj)!=expected:
        raise ValueError("Material-only render requires exactly matching vertices and faces")
    parts=worker.split_components(obj)
    if len(parts)!=1:
        raise ValueError("Unexpected component count")
    worker.center_mass(parts[0])
    parts[0].matrix_world=Matrix(pose["matrices"][0])
    span=pose["span"]
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,-.1*span))
    ground=bpy.context.active_object
    ground.name="support"
    ground.scale=(10*span,10*span,.2*span)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    camera=pose["camera"]
    worker.make_camera(scene,camera["eye"],camera["target"],ortho_scale=camera["orthographic_scale"])
    yaw=math.radians(camera["azimuth_deg"])
    center=Vector(camera["target"])
    if yaw:
        light=bpy.data.objects["soft light"]
        turn=Matrix.Rotation(yaw,3,"Z")
        light.location=center+turn@(light.location-center)
        light.rotation_euler=(turn@light.rotation_euler.to_matrix()).to_euler()
    bpy.context.view_layer.update()
    ground.hide_render=True
    worker.render(scene,output/"object.png")
    ground.hide_render=False
    worker.render(scene,output/"with-shadow.png")
    parts[0].hide_render=True
    worker.render(scene,output/"without-shadow.png")
    old_resolution=int(old_request.get("render_resolution",512))
    scale=request.get("render_resolution",512)/old_resolution
    for field in ("contacts_px","all_contacts_px"):
        pose[field]=[[[x*scale,y*scale] for x,y in group] for group in pose[field]]
    pose["material_only_reuse"]={"geometry_sha256":expected,"source_view":str(previous)}
    (output/"blender-result.json").write_text(json.dumps(pose),encoding="utf8")
    shutil.copy2(previous/"triangles.npz",output/"triangles.npz")
    print("MATERIAL_RENDER_OK",flush=True)


if __name__=="__main__":
    main()
