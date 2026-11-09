"""Render the unmodified glTF pose using the browser's orthographic camera."""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import blender_worker as worker

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
scene = worker.setup_scene('CPU', 1024)
# Use a bounded local environment and a soft key rather than a pure white
# world. The estimate still comes from a single 2-D image and is not a
# measured 3-D light direction.
environment = request.get('environment') or {}
world_color = environment.get('world_color') or [.32, .32, .32]
if len(world_color) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in world_color):
    raise ValueError('环境色估计无效')
world_strength = float(environment.get('world_strength', .55))
if not math.isfinite(world_strength) or not .1 <= world_strength <= 1.2:
    raise ValueError('环境强度估计无效')
background = scene.world.node_tree.nodes.get('Background')
background.inputs['Color'].default_value = (*world_color, 1)
background.inputs['Strength'].default_value = world_strength
objects = [worker.load_mesh(request['mesh'])]
points = [p for obj in objects for p in worker.world_vertices(obj)]
low = Vector([min(p[i] for p in points) for i in range(3)])
high = Vector([max(p[i] for p in points) for i in range(3)])
center = (low + high) / 2
span = max(high - low)
if span <= 0:
    raise ValueError('模型尺寸为空')
a, e = map(math.radians, (request['azimuth'], request['elevation']))
direction = Vector((math.sin(a)*math.cos(e), -math.cos(a)*math.cos(e), math.sin(e)))
camera=worker.make_camera(scene, center + direction * span * 5, center, ortho_scale=2 * span / 1.35)
key_direction = environment.get('key_direction') or [-.45, -.60]
if len(key_direction) != 2 or not all(math.isfinite(v) and abs(v) <= 1.01 for v in key_direction):
    raise ValueError('环境光方向估计无效')
bpy.context.view_layer.update()
local = Vector((key_direction[0], -key_direction[1], .8)).normalized()
key_world = camera.matrix_world.to_quaternion() @ local
key_strength = float(environment.get('key_strength', .65))
key = bpy.data.objects.get('soft light')
if key is None:
    data = bpy.data.lights.new('soft light', 'AREA')
    key = bpy.data.objects.new('soft light', data)
    scene.collection.objects.link(key)
key.data.shape = 'DISK'
key.data.size = max(span * 4, .2)
key.data.energy = max(30.0, 350.0 * span * span * np.clip(key_strength, .35, 1.3))
key.location = center + key_world * span * 4
key.rotation_euler = (center - key.location).to_track_quat('-Z', 'Y').to_euler()
fill = bpy.data.lights.new('environment fill', 'AREA')
fill.energy = key.data.energy * float(np.clip(environment.get('fill_strength', .25), .1, .4))
fill.shape = 'DISK'
fill.size = max(span * 6, .3)
fill_obj = bpy.data.objects.new('environment fill', fill)
scene.collection.objects.link(fill_obj)
fill_obj.location = center - key_world * span * 4
fill_obj.rotation_euler = (center - fill_obj.location).to_track_quat('-Z', 'Y').to_euler()
worker.render(scene, Path(request['output']))
