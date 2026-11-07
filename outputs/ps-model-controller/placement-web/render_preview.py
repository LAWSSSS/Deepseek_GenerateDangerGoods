"""Render the unmodified glTF pose using the browser's orthographic camera."""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import blender_worker as worker

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
scene = worker.setup_scene('CPU', 1024)
bpy.ops.import_scene.gltf(filepath=request['mesh'])
objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
if not objects:
    raise ValueError('模型中没有网格')
points = [p for obj in objects for p in worker.world_vertices(obj)]
low = Vector([min(p[i] for p in points) for i in range(3)])
high = Vector([max(p[i] for p in points) for i in range(3)])
center = (low + high) / 2
span = max(high - low)
if span <= 0:
    raise ValueError('模型尺寸为空')
a, e = map(math.radians, (request['azimuth'], request['elevation']))
direction = Vector((math.sin(a)*math.cos(e), -math.cos(a)*math.cos(e), math.sin(e)))
worker.make_camera(scene, center + direction * span * 5, center, ortho_scale=2 * span / 1.35)
worker.render(scene, Path(request['output']))
