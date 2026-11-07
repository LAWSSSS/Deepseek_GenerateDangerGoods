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
# Isotropic, neutral illumination keeps the source view free of a baked-in
# scene-specific key light. The scene match is applied after placement.
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = 1.0
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
lighting=request.get('lighting',{})
if lighting.get('status')=='estimated':
    dx,dy=lighting['direction_screen']
    if not all(math.isfinite(v) and abs(v)<=1.01 for v in (dx,dy)):
        raise ValueError('环境光估计无效')
    # A weak, large area source follows only the estimated screen direction.
    # Its elevation is an explicit assumption, not recovered geometry.
    bpy.context.view_layer.update()
    local=Vector((dx,-dy,1.0)).normalized()
    world=camera.matrix_world.to_quaternion() @ local
    data=bpy.data.lights.new('estimated soft scene light','AREA')
    data.energy=25*span*span;data.shape='DISK';data.size=span*3
    light=bpy.data.objects.new('estimated soft scene light',data)
    scene.collection.objects.link(light);light.location=center+world*span*4
    light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
    scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.8
worker.render(scene, Path(request['output']))
