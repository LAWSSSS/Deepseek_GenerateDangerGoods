"""Headless Blender 4.5 worker: canonical source view or physical settling."""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0,str(Path(__file__).resolve().parent))
from blender_support import support_graph


def world_vertices(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    matrix = evaluated.matrix_world
    return [matrix @ vertex.co for vertex in evaluated.data.vertices]


def make_camera(scene, eye, target, *, fov=40, ortho_scale=None):
    data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(camera)
    camera.location = eye
    camera.rotation_euler = (Vector(target)-camera.location).to_track_quat("-Z", "Y").to_euler()
    if ortho_scale:
        data.type = "ORTHO"
        data.ortho_scale = ortho_scale
    else:
        data.type = "PERSP"
        data.angle = math.radians(fov)
    scene.camera = camera
    return camera


def setup_scene(render_device="CPU", resolution=512):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if render_device == "CPU":
        scene.cycles.device = "CPU"
    elif render_device in ("CUDA", "OPTIX"):
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = render_device
        preferences.refresh_devices()
        for device in preferences.devices:
            device.use = device.type == render_device
        if not any(device.use for device in preferences.devices):
            raise RuntimeError(f"Requested Cycles backend has no device: {render_device}")
        scene.cycles.device = "GPU"
    else:
        raise ValueError(f"Unsupported Cycles backend: {render_device}")
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    if resolution not in (512,1024,2048):
        raise ValueError("Unsupported render resolution")
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("ambient")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Strength"].default_value = .65
    scene.world = world
    data = bpy.data.lights.new("soft light", "AREA")
    data.energy, data.shape, data.size = 350, "DISK", 5
    light = bpy.data.objects.new("soft light", data)
    scene.collection.objects.link(light)
    light.location = (-3, -2, 5)
    light.rotation_euler = (Vector((0, 0, 0))-light.location).to_track_quat("-Z", "Y").to_euler()
    return scene


def load_mesh(path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise ValueError("GLB contains no mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    # TripoSR exports its Z-up coordinates directly into glTF. Blender's glTF
    # importer converts glTF Y-up to Z-up, so undo that conversion before
    # applying TripoSR's canonical camera and physical poses.
    obj.data.transform(Matrix.Rotation(-math.pi/2, 4, "X"))
    obj.data.update()
    return obj


def render(scene, output):
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def source_view(request, scene, obj, output):
    values=request.get("camera", {})
    camera=make_camera(scene, (1.9, 0, 0), (0, 0, 0), fov=values.get("fov_deg",40))
    camera.data.shift_x=values.get("shift_x",0)
    camera.data.shift_y=values.get("shift_y",0)
    render(scene, output/"source-view.png")
    points = world_vertices(obj)
    bounds = [[min(v[i] for v in points),max(v[i] for v in points)] for i in range(3)]
    return {"camera":{"eye":[1.9, 0, 0], "target":[0, 0, 0],
                      "fov_deg":values.get("fov_deg",40),
                      "shift_x":camera.data.shift_x,"shift_y":camera.data.shift_y},
            "mesh_axis_bounds":bounds,"canonical_axes_restored":True,
            "source_view":"source-view.png"}


def split_components(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not 1 <= len(parts) <= 16:
        raise ValueError("Mesh has an unsupported number of disconnected components")
    return parts


def center_mass(part):
    """Move the origin to the uniform-volume centroid without moving geometry."""
    import bmesh
    mesh = bmesh.new()
    mesh.from_mesh(part.data)
    closed = all(edge.is_manifold for edge in mesh.edges)
    mesh.free()
    if not closed:
        raise ValueError("Component is not closed; center of mass is unknown")
    part.data.calc_loop_triangles()
    volume = 0.0
    moment = Vector((0, 0, 0))
    for triangle in part.data.loop_triangles:
        a, b, c = (part.data.vertices[i].co for i in triangle.vertices)
        v = a.dot(b.cross(c)) / 6
        volume += v
        moment += (a + b + c) * (v / 4)
    if abs(volume) < 1e-10:
        raise ValueError("Component has zero volume; center of mass is unknown")
    centroid = moment / volume
    matrix = part.matrix_world.copy()
    part.data.transform(Matrix.Translation(-centroid))
    part.matrix_world = matrix @ Matrix.Translation(centroid)
    part.data.update()
    return abs(volume), list(centroid)


def settle_view(request, scene, obj, output):
    # Every disconnected component is simulated separately against the plane.
    angles = request.get("rotation_deg", [0, 0, 0])
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = tuple(math.radians(value) for value in angles)
    bpy.context.view_layer.update()
    points = world_vertices(obj)
    initial_extents=[max(p[i] for p in points)-min(p[i] for p in points) for i in range(3)]
    span = max(initial_extents)
    if span <= 0:
        raise ValueError("degenerate mesh")
    obj.location.z += -min(p.z for p in points)+.02*span
    bpy.context.view_layer.update()
    parts = split_components(obj)
    properties = [center_mass(part) for part in parts]
    max_volume = max(value[0] for value in properties)
    bpy.context.view_layer.update()
    initial_matrices = [part.matrix_world.copy() for part in parts]
    for part, (volume, centroid) in zip(parts, properties):
        bpy.ops.object.select_all(action="DESELECT")
        part.select_set(True)
        bpy.context.view_layer.objects.active = part
        bpy.ops.rigidbody.object_add()
        part.rigid_body.type = "ACTIVE"
        part.rigid_body.mass = volume / max_volume
        part.rigid_body.collision_shape = "MESH"
        part.rigid_body.friction = .7
        part.rigid_body.restitution = 0
        part.rigid_body.linear_damping = .35
        part.rigid_body.angular_damping = .55
        part.rigid_body.use_margin = True
        part.rigid_body.collision_margin = .002*span
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,-.1*span))
    ground = bpy.context.active_object
    ground.name = "support"
    ground.scale = (10*span,10*span,.2*span)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = "PASSIVE"
    ground.rigid_body.collision_shape = "BOX"
    ground.rigid_body.friction = .7
    scene.gravity = (0,0,-.98*span)
    scene.rigidbody_world.substeps_per_frame = 10 if request.get("solver_refined") else 5
    scene.rigidbody_world.solver_iterations = 40 if request.get("solver_refined") else 20
    scene.frame_start, scene.frame_end = 1, 400
    previous=None
    current=[]
    displacements=[]
    rotations=[]
    stable_windows=0
    for frame in range(1, 401):
        scene.frame_set(frame)
        if frame%20==0:
            current=[o.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy() for o in parts]
            if previous is not None:
                displacements=[(b.translation-a.translation).length for a,b in zip(previous,current)]
                differences=[math.degrees(a.to_quaternion().rotation_difference(b.to_quaternion()).angle)%360
                             for a,b in zip(previous,current)]
                rotations=[min(angle,360-angle) for angle in differences]
                quiet=(max(displacements,default=999)<=.005*span and max(rotations,default=999)<=1)
                stable_windows=stable_windows+1 if quiet and frame>=80 else 0
                if stable_windows>=2:
                    break
            previous=current
    all_points = [world_vertices(part) for part in parts]
    grounds = [min(p.z for p in vertices) for vertices in all_points]
    support=support_graph(parts,all_points,span)
    assembly_changed = False
    if len(parts)>1:
        initial_relative = [initial_matrices[0].inverted() @ m for m in initial_matrices[1:]]
        final_relative = [current[0].inverted() @ m for m in current[1:]]
        for before, after in zip(initial_relative, final_relative):
            angle = math.degrees(before.to_quaternion().rotation_difference(after.to_quaternion()).angle) % 360
            if (before.translation-after.translation).length > .005*span or min(angle,360-angle)>1:
                assembly_changed = True
    stable = (stable_windows>=2 and max(displacements, default=999) <= .005*span and max(rotations, default=999) <= 1
              and all(z>=-.005*span for z in grounds) and support["all_anchored"]
              and not support["component_collisions"] and not assembly_changed)
    failures=[]
    if stable_windows<2: failures.append("not_converged")
    if any(z < -.005*span for z in grounds): failures.append("ground_penetration")
    if not support["all_anchored"]: failures.append("unsupported")
    if support["component_collisions"]: failures.append("component_collision")
    if assembly_changed: failures.append("assembly_relationship_unknown")
    physics = {"physics_stable":stable, "component_count":len(parts), "component_min_z":grounds,
               "failure_codes":failures,"mass_assumption":"uniform_density",
               "components":[{"volume":v,"centroid_before":c,"mass":v/max_volume} for v,c in properties],
               "initial_extents":initial_extents, "simulated_frames":frame, "stable_windows":stable_windows,
               "component_collisions":support["component_collisions"],"support_graph":support,
               "motion":displacements, "rotation_motion_deg":rotations, "span":span}
    if not stable and request.get("skip_unstable_renders"):
        return physics
    points = [p for vertices in all_points for p in vertices]
    center = sum(points, Vector((0, 0, 0)))/len(points)
    elevation=float(request.get("camera_elevation_deg",56.82))
    if not 10<=elevation<=85:
        raise ValueError("Scene camera elevation is outside the supported range")
    distance=6.2*span
    angle=math.radians(elevation)
    azimuth=float(request.get("camera_azimuth_deg",0))
    if not 0<=azimuth<360:
        raise ValueError("Camera azimuth must be in [0, 360)")
    yaw=math.radians(azimuth)
    horizontal=distance*math.cos(angle)
    camera = make_camera(scene, (center.x+horizontal*math.sin(yaw), center.y-horizontal*math.cos(yaw),
                                 center.z+distance*math.sin(angle)),
                         center, ortho_scale=span*1.8)
    if azimuth:
        light=bpy.data.objects["soft light"]
        turn=Matrix.Rotation(yaw,3,"Z")
        light.location=center+turn@(light.location-center)
        light.rotation_euler=(turn@light.rotation_euler.to_matrix()).to_euler()
    bpy.context.view_layer.update()
    contacts = []
    for vertices in all_points:
        low = min(p.z for p in vertices)
        touching = [p for p in vertices if p.z <= low+.015*span]
        projected = [world_to_camera_view(scene, camera, p) for p in touching]
        contacts.append([[round(p.x*scene.render.resolution_x, 2),
                          round((1-p.y)*scene.render.resolution_y, 2)] for p in projected])
    # Produce separate object and two receiver renders for an actual shadow delta.
    ground.hide_render = True
    render(scene, output/"object.png")
    ground.hide_render = False
    render(scene, output/"with-shadow.png")
    for part in parts:
        part.hide_render = True
    render(scene, output/"without-shadow.png")
    return {**physics,
            "contacts_px":[contacts[i] for i in support["grounded"]],
            "all_contacts_px":contacts,"matrices":[[list(row) for row in matrix] for matrix in current],
            "camera":{"eye":list(camera.location), "target":list(center),
                      "orthographic_scale":camera.data.ortho_scale,"elevation_deg":elevation,"azimuth_deg":azimuth},
            "object":"object.png", "shadow_with":"with-shadow.png", "shadow_without":"without-shadow.png"}


def main():
    args = sys.argv[sys.argv.index("--")+1:]
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    output = Path(request["out"])
    output.mkdir(parents=True, exist_ok=True)
    render_device = request.get("render_device", "CPU")
    scene = setup_scene(render_device, request.get("render_resolution",512))
    obj = load_mesh(request["mesh"])
    if request["operation"] == "source":
        result = source_view(request, scene, obj, output)
    elif request["operation"] == "settle":
        result = settle_view(request, scene, obj, output)
    else:
        raise ValueError("Unknown Blender operation")
    result["render_device"] = render_device
    (output/"blender-result.json").write_text(json.dumps(result), encoding="utf-8")
    print("BLENDER_TASK_OK", flush=True)


if __name__ == "__main__":
    main()
