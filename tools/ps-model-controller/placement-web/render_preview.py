"""Render the user's glTF pose and project its near-ground mesh patches."""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import blender_worker as worker


def clip_to_height(points, height):
    clipped = []
    for first, last in zip(points, np.roll(points, -1, axis=0), strict=True):
        first_in, last_in = first[2] <= height, last[2] <= height
        if first_in:
            clipped.append(first)
        if first_in != last_in:
            clipped.append(first + (last - first) * ((height - first[2]) / (last[2] - first[2])))
    return np.asarray(clipped)


def ground_projection(scene, camera, obj, span, elevation):
    points = np.asarray(worker.world_vertices(obj), dtype=np.float64)
    floor = points[:, 2].min()
    height = floor + span * .012
    obj.data.calc_loop_triangles()
    indices = np.empty((len(obj.data.loop_triangles), 3), dtype=np.int32)
    obj.data.loop_triangles.foreach_get('vertices', indices.ravel())
    near = points[:, 2] <= height
    polygons = []
    patches = []
    for face in indices[near[indices].any(axis=1)]:
        patch = clip_to_height(points[face], height)
        if len(patch) < 3:
            continue
        patches.append(patch)
        # Project onto the hypothesized ground plane using the render camera.
        screen = [worker.world_to_camera_view(scene, camera, Vector((x, y, floor))) for x, y, _ in patch]
        if any(point.z <= 0 for point in screen):
            continue
        xy = np.array([(point.x, 1 - point.y) for point in screen])
        area2 = np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))
        if abs(area2) > 1e-9:
            polygons.append(xy.round(6).tolist())
    if not patches:
        return {'polygons': [], 'reason': '没有可定位的网格接触区域'}
    patches = np.concatenate(patches)
    # This is a placement reference between the contact patches, not a claim
    # that the space between separated feet is itself supported.
    anchor = (patches.min(axis=0) + patches.max(axis=0)) / 2
    anchor[2] = floor
    point = worker.world_to_camera_view(scene, camera, Vector(anchor))
    result = {'polygons': polygons, 'band_fraction': .012, 'plane': 'lowest_world_z',
              'anchor_uv': [float(point.x), float(1 - point.y)],
              'anchor_basis': 'near_ground_patch_bounds_center', 'physical_support': 'unknown'}
    if elevation < 8:
        result.update(polygons=[], reason='当前观察角度不足以判断承托投影')
    return result


def ambient_dome(scene, color, strength):
    """Build a smooth hemisphere from colour statistics, without scene texture."""
    width, height = 128, 64
    z = np.sin(np.linspace(-math.pi / 2, math.pi / 2, height, dtype=np.float32))
    brightness = .70 + .90 * (z + 1) / 2
    pixels = np.ones((height, width, 4), dtype=np.float32)
    pixels[:, :, :3] = brightness[:, None, None] * np.asarray(color, dtype=np.float32)
    image = bpy.data.images.new('estimated ambient dome', width=width, height=height, float_buffer=True)
    image.colorspace_settings.name = 'Non-Color'
    image.pixels.foreach_set(pixels.ravel())
    nodes = scene.world.node_tree.nodes
    world = nodes.get('Background')
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = image
    scene.world.node_tree.links.new(env.outputs['Color'], world.inputs['Color'])
    world.inputs['Strength'].default_value = strength


request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
scene = worker.setup_scene('CPU', 1024)
scene.view_settings.view_transform = 'AgX'
if request.get('packed_subject'):
    from blender_display_worker import append_packed_subject
    obj = append_packed_subject(request['render_mesh'], request['packed_subject'], request['render_sha256'])
else:
    obj = worker.load_mesh(request['mesh'])
points = worker.world_vertices(obj)
low = Vector([min(p[i] for p in points) for i in range(3)])
high = Vector([max(p[i] for p in points) for i in range(3)])
center = (low + high) / 2
span = max(high - low)
if span <= 0:
    raise ValueError('模型尺寸为空')
rotation = request.get('rotation_deg') or [0, 0, 0]
if len(rotation) != 3 or not all(math.isfinite(float(v)) for v in rotation):
    raise ValueError('物体旋转角无效')
if any(abs(float(v)) > 1e-6 for v in rotation):
    # Preserve the original normalization while rotating about the same
    # world-space centre as the browser.
    convert = Matrix.Rotation(-math.pi / 2, 4, 'X')
    browser_rotation = (Matrix.Rotation(math.radians(float(rotation[2])), 4, 'Z')
                        @ Matrix.Rotation(math.radians(float(rotation[1])), 4, 'Y')
                        @ Matrix.Rotation(math.radians(float(rotation[0])), 4, 'X'))
    blender_rotation = convert.inverted() @ browser_rotation @ convert
    obj.matrix_world = Matrix.Translation(center) @ blender_rotation @ Matrix.Translation(-center) @ obj.matrix_world

a, e = map(math.radians, (request['azimuth'], request['elevation']))
direction = Vector((math.sin(a)*math.cos(e), -math.cos(a)*math.cos(e), math.sin(e)))
camera = worker.make_camera(scene, center + direction * span * 2.35, center, fov=35)
bpy.context.view_layer.update()

output = Path(request['output'])
contact = ground_projection(scene, camera, obj, span, request['elevation'])
output.with_name('contact.json').write_text(json.dumps(contact, ensure_ascii=False), encoding='utf8')
if request.get('geometry_only'):
    # The server uses these actual projected patches to choose scene samples
    # before lighting. No material render or physical acceptance is implied.
    sys.exit(0)

environment = request.get('environment') or {}
world_color = environment.get('world_color', [.36, .36, .36])
light_color = environment.get('light_color', [1., 1., 1.])
if any(len(color) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in color)
       for color in (world_color, light_color)):
    raise ValueError('环境色估计无效')
ambient_dome(scene, world_color, float(np.clip(environment.get('world_strength', .76), .3, .8)))
# A weak overhead source represents the diffuse sky. Only a consistent
# scene cue can move the light away from overhead.
soft = bpy.data.objects['soft light']
key_world = Vector((0, 0, 1))
key_direction = environment.get('key_direction')
if key_direction is not None:
    if len(key_direction) != 2 or not all(math.isfinite(v) and abs(v) <= 1.01 for v in key_direction):
        raise ValueError('环境光方向估计无效')
    local = Vector((key_direction[0], -key_direction[1], .8)).normalized()
    key_world = camera.matrix_world.to_quaternion() @ local
    key_world.z = max(key_world.z, .32)
soft.data.color = light_color
soft.data.shape = 'DISK'
soft.data.size = 5.8 * span
soft.data.energy = 470 * span * span * float(np.clip(environment.get('key_strength', .18), .1, .65))
soft.location = center + key_world.normalized() * span * 3.5
soft.rotation_euler = (center - soft.location).to_track_quat('-Z', 'Y').to_euler()

worker.render(scene, output)
