"""Render matched diagnostic directions for a known single-body pose."""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
import blender_worker as worker


def main():
    args=sys.argv[sys.argv.index("--")+1:]
    request=json.loads(Path(args[0]).read_text(encoding="utf8"))
    pose=json.loads(Path(request["pose"]).read_text(encoding="utf8"))
    if not pose.get("physics_stable") or pose.get("component_count")!=1:
        raise ValueError("六方向示意需要已检查的单件姿态")
    out=Path(request["out"]);out.mkdir(parents=True,exist_ok=True)
    scene=worker.setup_scene(request.get("render_device","CPU"),512)
    scene.cycles.samples=16
    scene.world.node_tree.nodes.get("Background").inputs["Strength"].default_value=1.1
    light_data=bpy.data.lights.new("comparison fill","AREA")
    light_data.energy,light_data.shape,light_data.size=220,"DISK",5
    fill=bpy.data.objects.new("comparison fill",light_data)
    scene.collection.objects.link(fill)
    fill.location=(3,2,4)
    fill.rotation_euler=(Vector((0,0,0))-fill.location).to_track_quat("-Z","Y").to_euler()
    center=Vector(pose["camera"]["target"])
    yaw=math.radians(pose["camera"]["azimuth_deg"])
    front=Vector((math.sin(yaw),-math.cos(yaw),0))
    right=Vector((-front.y,front.x,0))
    directions=[("front",front),("back",-front),("left",-right),("right",right),
                ("top",Vector((0,0,1))),("bottom",Vector((0,0,-1)))]
    for label,path in (("before",request["before"]),("after",request["after"])):
        obj=worker.load_mesh(path)
        worker.center_mass(obj)
        obj.matrix_world=Matrix(pose["matrices"][0])
        bpy.context.view_layer.update()
        for name,direction in directions:
            camera=worker.make_camera(scene,center+direction*4,center,
                                      ortho_scale=pose["camera"]["orthographic_scale"])
            worker.render(scene,out/f"{label}-{name}.png")
            bpy.data.objects.remove(camera,do_unlink=True)
        bpy.data.objects.remove(obj,do_unlink=True)
    print("SIX_VIEW_COMPARISON_OK",flush=True)


if __name__=="__main__":main()
