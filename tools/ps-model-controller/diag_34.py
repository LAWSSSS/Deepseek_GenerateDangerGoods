"""Temporary item-34 Blender diagnostic: inspect gauge UVs, glass and handles.

Run with:
  blender -b --factory-startup --python diag_34.py -- <model.blend> <out-dir>
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
blend = Path(args[0]).resolve()
out = Path(args[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(blend))
scene = bpy.context.scene

report = {}
for name in ('gauge_dial', 'gauge_glass', 'handle_lower', 'lever_upper',
             'gauge_bezel'):
    obj = bpy.data.objects.get(name)
    if obj is None:
        report[name] = None
        continue
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    report[name] = {
        'bounds_min': [min(p[i] for p in corners) for i in range(3)],
        'bounds_max': [max(p[i] for p in corners) for i in range(3)],
    }

dial = bpy.data.objects.get('gauge_dial')
if dial is not None:
    layer = dial.data.uv_layers.active
    us, vs = [], []
    for poly in dial.data.polygons:
        if poly.normal.x > .9:
            for li in poly.loop_indices:
                uv = layer.data[li].uv
                us.append(uv.x)
                vs.append(uv.y)
    report['dial_front_uv'] = {
        'min_u': min(us), 'max_u': max(us),
        'min_v': min(vs), 'max_v': max(vs),
        'loops': len(us),
    }
    report['dial_material'] = [m.name for m in dial.data.materials]
    report['dial_front_face_count'] = sum(
        1 for p in dial.data.polygons if p.normal.x > .9)

# Camera helper.
def make_camera(eye, target, ortho=None, fov=40):
    data = bpy.data.cameras.new('diag camera')
    cam = bpy.data.objects.new('diag camera', data)
    scene.collection.objects.link(cam)
    cam.location = Vector(eye)
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    if ortho:
        data.type = 'ORTHO'
        data.ortho_scale = ortho
    else:
        data.type = 'PERSP'
        data.angle = math.radians(fov)
    scene.camera = cam
    return cam


def only(names):
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_render = obj.name not in names


def render(path, resolution=512, samples=32):
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def center_of(name):
    obj = bpy.data.objects[name]
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return sum(corners, Vector()) / 8.0


gauge_center = center_of('gauge_glass')
handle_center = (center_of('handle_lower') + center_of('lever_upper')) / 2.0

gauge_names = {'gauge_housing', 'gauge_brass_ring', 'gauge_bezel',
               'gauge_dial', 'gauge_needle', 'gauge_needle_tail',
               'gauge_hub', 'gauge_hub_dark'}
only(gauge_names)
make_camera(gauge_center + Vector((0.24, -0.14, 0.06)), gauge_center,
            ortho=.20)
render(out / 'diag-gauge-noglass.png')

only(gauge_names | {'gauge_glass'})
make_camera(gauge_center + Vector((0.24, -0.14, 0.06)), gauge_center,
            ortho=.20)
render(out / 'diag-gauge-glass.png')

only({'handle_lower', 'lever_upper', 'handle_pivot_boss',
      'handle_pivot_pin', 'handle_pivot_washer_a', 'handle_pivot_washer_b',
      'safety_ring', 'safety_tamper_strap', 'safety_tag', 'safety_tag_wire'})
make_camera(handle_center + Vector((1.5, 0.0, 0.0)), handle_center,
            ortho=.50)
render(out / 'diag-handles-front.png')
make_camera(handle_center + Vector((1.5, -0.3, 0.35)), handle_center,
            ortho=.50)
render(out / 'diag-handles-3q.png')

(out / 'diag-34.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(report, ensure_ascii=False, indent=2))
