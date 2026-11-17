"""Blender worker for screening: treat a multi-part revision as one rigid body.

The formal worker splits every disconnected component and intentionally caps the
count at 16.  The feedback revisions are parametric assemblies with hundreds or
thousands of loose pieces, so this worker keeps the whole mesh as one convex
hull.  It is an explicit approximation for support screening only, not a
replacement for the formal per-component settling.
"""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as worker


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    output = Path(request["out"])
    output.mkdir(parents=True, exist_ok=True)
    render_device = request.get("render_device", "CPU")
    scene = worker.setup_scene(render_device, request.get("render_resolution", 512))
    obj = worker.load_mesh(request["mesh"])
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = tuple(math.radians(value) for value in request.get("rotation_deg", [0, 0, 0]))
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.context.view_layer.update()

    points = worker.world_vertices(obj)
    initial_extents = [max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)]
    span = max(initial_extents)
    if span <= 0:
        raise ValueError("degenerate mesh")
    obj.location.z += -min(p.z for p in points) + .02 * span
    bpy.context.view_layer.update()

    bpy.ops.rigidbody.object_add()
    body = obj.rigid_body
    body.type = "ACTIVE"
    body.mass = 1.0
    # A convex hull is deliberately conservative: it keeps single-body
    # settling cheap for assemblies with thousands of loose pieces, but it
    # cannot verify per-part contact or internal clearances.
    body.collision_shape = "CONVEX_HULL"
    body.friction = .7
    body.restitution = 0
    body.linear_damping = .35
    body.angular_damping = .55
    body.use_margin = True
    body.collision_margin = .002 * span

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -.1 * span))
    ground = bpy.context.active_object
    ground.name = "support"
    ground.scale = (10 * span, 10 * span, .2 * span)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = "PASSIVE"
    ground.rigid_body.collision_shape = "BOX"
    ground.rigid_body.friction = .7
    scene.gravity = (0, 0, -.98 * span)
    scene.rigidbody_world.substeps_per_frame = 5
    scene.rigidbody_world.solver_iterations = 20
    scene.frame_start, scene.frame_end = 1, 400

    previous = None
    current = []
    displacements = []
    rotations = []
    stable_windows = 0
    frame = 1
    for frame in range(1, 401):
        scene.frame_set(frame)
        if frame % 20 == 0:
            evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            current = [evaluated.matrix_world.copy()]
            if previous is not None:
                displacements = [(current[0].translation - previous[0].translation).length]
                angle = math.degrees(
                    previous[0].to_quaternion().rotation_difference(current[0].to_quaternion()).angle) % 360
                rotations = [min(angle, 360 - angle)]
                quiet = (max(displacements) <= .005 * span and max(rotations) <= 1)
                stable_windows = stable_windows + 1 if quiet and frame >= 80 else 0
                if stable_windows >= 2:
                    break
            previous = current
    bpy.context.view_layer.update()

    points = worker.world_vertices(obj)
    min_z = min(p.z for p in points)
    stable = (stable_windows >= 2 and max(displacements, default=999) <= .005 * span
              and max(rotations, default=999) <= 1 and min_z >= -.005 * span)
    failures = []
    if stable_windows < 2:
        failures.append("not_converged")
    if min_z < -.005 * span:
        failures.append("ground_penetration")
    if not stable and "not_converged" not in failures:
        failures.append("unsupported")
    physics = {
        "physics_stable": stable,
        "component_count": 1,
        "component_min_z": [min_z],
        "failure_codes": failures,
        "mass_assumption": "single_compound_convex_hull",
        "components": [{"volume": 1.0, "centroid_before": [0, 0, 0], "mass": 1.0}],
        "initial_extents": initial_extents,
        "simulated_frames": frame,
        "stable_windows": stable_windows,
        "component_collisions": False,
        "support_graph": {"all_anchored": stable, "grounded": [0], "component_collisions": False},
        "motion": displacements,
        "rotation_motion_deg": rotations,
        "span": span,
        "screening_approximation": "whole mesh as one convex-hull rigid body; component-level support not verified",
    }
    if not stable and request.get("skip_unstable_renders"):
        (output / "blender-result.json").write_text(json.dumps(physics), encoding="utf-8")
        return

    center = sum(points, Vector((0, 0, 0))) / len(points)
    elevation = float(request.get("camera_elevation_deg", 56.82))
    if not 10 <= elevation <= 85:
        raise ValueError("Scene camera elevation is outside the supported range")
    distance = 6.2 * span
    angle = math.radians(elevation)
    azimuth = float(request.get("camera_azimuth_deg", 0))
    if not 0 <= azimuth < 360:
        raise ValueError("Camera azimuth must be in [0, 360)")
    yaw = math.radians(azimuth)
    horizontal = distance * math.cos(angle)
    camera = worker.make_camera(
        scene,
        (center.x + horizontal * math.sin(yaw), center.y - horizontal * math.cos(yaw),
         center.z + distance * math.sin(angle)),
        center, ortho_scale=span * 1.8)
    if azimuth:
        light = bpy.data.objects["soft light"]
        turn = Matrix.Rotation(yaw, 3, "Z")
        light.location = center + turn @ (light.location - center)
        light.rotation_euler = (turn @ light.rotation_euler.to_matrix()).to_euler()
    bpy.context.view_layer.update()

    low = min(p.z for p in points)
    touching = [p for p in points if p.z <= low + .015 * span]
    projected = [world_to_camera_view(scene, camera, p) for p in touching]
    contacts = [[[round(p.x * scene.render.resolution_x, 2),
                  round((1 - p.y) * scene.render.resolution_y, 2)] for p in projected]]

    ground.hide_render = True
    worker.render(scene, output / "object.png")
    ground.hide_render = False
    worker.render(scene, output / "with-shadow.png")
    obj.hide_render = True
    worker.render(scene, output / "without-shadow.png")
    obj.hide_render = False
    result = {
        **physics,
        "contacts_px": contacts,
        "all_contacts_px": contacts,
        "camera": {"eye": list(camera.location), "target": list(center),
                   "orthographic_scale": camera.data.ortho_scale,
                   "elevation_deg": elevation, "azimuth_deg": azimuth},
        "object": "object.png",
        "shadow_with": "with-shadow.png",
        "shadow_without": "without-shadow.png",
    }
    (output / "blender-result.json").write_text(json.dumps(result), encoding="utf-8")
    print("SCREENING_SETTLE_OK", flush=True)


if __name__ == "__main__":
    main()
