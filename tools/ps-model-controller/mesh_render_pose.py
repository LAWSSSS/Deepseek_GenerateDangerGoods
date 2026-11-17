"""Render an already settled single-body pose after a material-only change."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
import blender_worker as worker
from settled_mesh_worker import triangles


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
    attribute=obj.data.attributes.get("source_face_id") or obj.data.attributes.new("source_face_id","INT","FACE")
    attribute.data.foreach_set("value",np.arange(len(obj.data.polygons),dtype=np.int32))
    original,original_ids=triangles(obj,require_ids=True)
    with np.load(previous/"triangles.npz") as saved:
        reference_source=saved["source"].copy();reference_source_ids=saved["source_ids"].copy()
        reference_pose=saved["posed"].copy();reference_pose_ids=saved["posed_ids"].copy()
    def same_triangles(actual,actual_ids,reference,reference_ids):
        a=np.argsort(actual_ids);b=np.argsort(reference_ids)
        return (np.array_equal(actual_ids[a],reference_ids[b]) and actual.shape==reference.shape and
                np.allclose(actual[a],reference[b],atol=1e-6,rtol=0))
    if not same_triangles(original,original_ids,reference_source,reference_source_ids):
        raise ValueError("Cached source triangle provenance does not match the mesh")
    parts=worker.split_components(obj)
    if len(parts)!=1:
        raise ValueError("Unexpected component count")
    worker.center_mass(parts[0])
    parts[0].matrix_world=Matrix(pose["matrices"][0])
    bpy.context.view_layer.update()
    posed,posed_ids=triangles(parts[0],require_ids=True)
    if not same_triangles(posed,posed_ids,reference_pose,reference_pose_ids):
        raise ValueError("Cached settled triangles do not match the verified pose")
    span=pose["span"]
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,-.1*span))
    ground=bpy.context.active_object
    ground.name="support"
    ground.scale=(10*span,10*span,.2*span)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    previous_camera=pose["camera"]
    center=Vector(previous_camera["target"])
    azimuth=float(request.get("camera_azimuth_deg",previous_camera["azimuth_deg"]))
    elevation=float(request.get("camera_elevation_deg",previous_camera["elevation_deg"]))
    if not 0<=azimuth<360 or not 10<=elevation<=85:
        raise ValueError("Camera orientation is outside the supported range")
    distance=(Vector(previous_camera["eye"])-center).length
    yaw=math.radians(azimuth)
    tilt=math.radians(elevation)
    horizontal=distance*math.cos(tilt)
    eye=center+Vector((horizontal*math.sin(yaw),-horizontal*math.cos(yaw),distance*math.sin(tilt)))
    camera=worker.make_camera(scene,eye,center,ortho_scale=previous_camera["orthographic_scale"])
    if yaw:
        light=bpy.data.objects["soft light"]
        turn=Matrix.Rotation(yaw,3,"Z")
        light.location=center+turn@(light.location-center)
        light.rotation_euler=(turn@light.rotation_euler.to_matrix()).to_euler()
    bpy.context.view_layer.update()
    vertices=worker.world_vertices(parts[0])
    low=min(point.z for point in vertices)
    touching=[point for point in vertices if point.z<=low+.015*span]
    projected=[world_to_camera_view(scene,camera,point) for point in touching]
    contacts=[[[round(point.x*scene.render.resolution_x,2),
                round((1-point.y)*scene.render.resolution_y,2)] for point in projected]]
    ground.hide_render=True
    worker.render(scene,output/"object.png")
    ground.hide_render=False
    worker.render(scene,output/"with-shadow.png")
    parts[0].hide_render=True
    worker.render(scene,output/"without-shadow.png")
    pose["contacts_px"]=contacts
    pose["all_contacts_px"]=contacts
    pose["camera"]={"eye":list(camera.location),"target":list(center),
                    "orthographic_scale":camera.data.ortho_scale,"elevation_deg":elevation,"azimuth_deg":azimuth}
    pose["material_only_reuse"]={"geometry_sha256":expected,"source_view":str(previous),
                                 "contacts_reprojected":True}
    (output/"blender-result.json").write_text(json.dumps(pose),encoding="utf8")
    shutil.copy2(previous/"triangles.npz",output/"triangles.npz")
    print("MATERIAL_RENDER_OK",flush=True)


if __name__=="__main__":
    main()
