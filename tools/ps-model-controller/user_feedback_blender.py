"""Create single-object and opaque-shell revisions from the existing Blender files."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Quaternion, Vector

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))


def meshes():
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and not obj.hide_render]


def bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    return (Vector([min(p[i] for p in points) for i in range(3)]),
            Vector([max(p[i] for p in points) for i in range(3)]))


def fingerprint(obj):
    h = hashlib.sha256()
    for v in obj.data.vertices:
        h.update(('%0.9f,%0.9f,%0.9f;' % tuple(v.co)).encode('ascii'))
    for face in obj.data.polygons:
        h.update((','.join(str(i) for i in face.vertices) + ';').encode('ascii'))
    return h.hexdigest()


def keep_only_prefix(prefix):
    keep = [obj for obj in meshes() if obj.name.startswith(prefix + ' ')]
    if not keep:
        raise RuntimeError('No objects found for ' + prefix)
    for obj in tuple(bpy.context.scene.objects):
        if obj.type == 'MESH' and obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    lo, hi = bounds(keep)
    shift = Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    for obj in keep:
        matrix = obj.matrix_world.copy()
        matrix.translation += shift
        obj.matrix_world = matrix
    bpy.context.view_layer.update()
    return keep, [float(v) for v in shift]


def full_barrel(dim):
    old = bpy.data.objects.get('barrel')
    if old is None or not old.data.materials:
        raise RuntimeError('The original barrel or its material is missing')
    matrix, material = old.matrix_world.copy(), old.data.materials[0]
    y0, y1 = dim['y0'], dim['y1']
    radius, inner = dim['radius'], dim['inner_radius']
    segments = 192
    verts = []
    for r in (radius, inner):
        for y in (y0, y1):
            for i in range(segments):
                angle = math.tau * i / segments
                verts.append((r * math.cos(angle), y, r * math.sin(angle)))
    o0 = list(range(segments))
    o1 = list(range(segments, segments * 2))
    i0 = list(range(segments * 2, segments * 3))
    i1 = list(range(segments * 3, segments * 4))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(((o0[i], o0[j], o1[j], o1[i]),
                      (i0[i], i1[i], i1[j], i0[j]),
                      (o0[i], i0[i], i0[j], o0[j]),
                      (o1[i], o1[j], i1[j], i1[i])))
    mesh = bpy.data.meshes.new('opaque_full_barrel_mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bad = sum(not edge.is_manifold for edge in bm.edges)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    if bad:
        raise RuntimeError(f'Barrel has {bad} non-manifold edges')
    obj = bpy.data.objects.new('barrel_full_opaque', mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = matrix
    obj.data.materials.append(material)
    uv = mesh.uv_layers.new(name='Atlas')
    margin = .012
    for poly in mesh.polygons:
        center, normal = poly.center, poly.normal
        radial = center.x * normal.x + center.z * normal.z
        planar = abs(normal.y) > .60
        tile = 0 if planar or radial > 0 else 8
        x0, ytile = (tile % 4) * .25, (tile // 4) * .25
        us = []
        for li in poly.loop_indices:
            v = mesh.vertices[mesh.loops[li].vertex_index].co
            u = (math.atan2(v.z, v.x) / math.tau) % 1
            us.append(u)
        wrap = max(us) - min(us) > .5
        for li, u in zip(poly.loop_indices, us):
            v = mesh.vertices[mesh.loops[li].vertex_index].co
            if planar:
                u = (v.x / (radius * 2)) + .5
                vv = (v.z / (radius * 2)) + .5
            else:
                if wrap and u < .5:
                    u = 1.0
                vv = (v.y - y0) / (y1 - y0)
            uv.data[li].uv = (x0 + .25 * (margin + u * (1 - 2 * margin)),
                              ytile + .25 * (margin + vv * (1 - 2 * margin)))
        poly.use_smooth = not planar
    bpy.data.objects.remove(old, do_unlink=True)
    for obj0 in tuple(bpy.context.scene.objects):
        if obj0.type == 'MESH' and (obj0.name == 'piston_body' or
                                    obj0.name.startswith('piston_ring_') or
                                    obj0.name.startswith('gland_ring_')):
            bpy.data.objects.remove(obj0, do_unlink=True)
    bpy.context.view_layer.update()
    total = good = 0
    for f in (.1, .3, .5, .7, .9):
        y = y0 + (y1 - y0) * f
        for i in range(36):
            angle = math.tau * i / 36
            direction = Vector((math.cos(angle), 0, math.sin(angle)))
            hit, position, _, _ = obj.ray_cast(direction * (2 * radius) + Vector((0,y,0)), -direction,
                                               distance=radius * 2.1)
            total += 1
            if hit and abs(math.hypot(position.x, position.z) - radius) < .001:
                good += 1
    if good != total:
        raise RuntimeError(f'Opaque barrel surface failed rays: {good}/{total}')
    return {'outer_shell_rays_passed': good, 'outer_shell_rays_total': total,
            'nonmanifold_edges': bad, 'material': material.name}


def camera_for(scene):
    data = bpy.data.cameras.new('feedback_review_camera')
    cam = bpy.data.objects.new('feedback_review_camera', data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    data.type = 'ORTHO'
    return cam


def render(scene, cam, objects, folder):
    folder.mkdir(exist_ok=True)
    lo, hi = bounds(objects)
    center = (lo + hi) * .5
    span = max(hi - lo)
    directions = {'front':(1,0,.06), 'back':(-1,0,.06), 'left':(0,-1,.06),
                  'right':(0,1,.06), 'top':(0,0,1), 'bottom':(0,0,-1),
                  '3q':(2,-2,1.2)}
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = min(12, os.cpu_count() or 4)
    scene.render.resolution_x = scene.render.resolution_y = 384
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    for name, direction in directions.items():
        direction = Vector(direction).normalized()
        cam.location = center + direction * max(span, .01) * 5
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = (center - cam.location).to_track_quat('-Z', 'Y')
        if abs(direction.z) > .999:
            cam.rotation_quaternion = cam.rotation_quaternion @ Quaternion((0,0,1), math.pi / 2)
        cam.data.ortho_scale = span * 1.28
        scene.render.filepath = str(folder / f'{name}.png')
        bpy.ops.render.render(write_still=True)


def main():
    summaries = []
    for item in request['items']:
        output = Path(item['out'])
        output.mkdir(parents=True, exist_ok=False)
        bpy.ops.wm.open_mainfile(filepath=item['source_blend'], load_ui=False)
        scene = bpy.context.scene
        evidence = {}
        if item['kind'] in ('can', 'bottle'):
            selected, shift = keep_only_prefix(item['prefix'])
            body_name = item['prefix'] + (' body' if item['kind'] == 'can' else ' 筒身')
            if len([obj for obj in selected if obj.name == body_name]) != 1:
                raise RuntimeError('A single body was not found')
            if any(not obj.name.startswith(item['prefix'] + ' ') for obj in selected):
                raise RuntimeError('Unexpected mesh in single-object candidate')
            evidence.update({'single_body_count':1, 'translation':shift, 'prefix':item['prefix']})
        elif item['kind'] == 'jack':
            removed = {'Detached handle bar', 'Handle bar bore liner', 'Handle bar top rim'}
            before = {obj.name:(fingerprint(obj), tuple(tuple(row) for row in obj.matrix_world))
                      for obj in meshes()}
            if not removed.issubset(before):
                raise RuntimeError('The detached handle assembly is incomplete')
            for name in removed:
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
            bpy.context.view_layer.update()
            selected = meshes()
            after = {obj.name:(fingerprint(obj), tuple(tuple(row) for row in obj.matrix_world))
                     for obj in selected}
            if after != {name:signature for name,signature in before.items() if name not in removed}:
                raise RuntimeError('A retained jack component changed')
            required = {'Jack body', 'Cast base', 'Chrome ram', 'Pump handle socket', 'Carry handle loop'}
            if not required.issubset(after):
                raise RuntimeError('An essential jack component is missing')
            evidence.update({'removed_objects':sorted(removed),
                             'retained_geometry_and_transforms_unchanged':True,
                             'retained_object_count':len(after)})
        elif item['kind'] == 'cylinder':
            eye_names = ('rear_clevis','rear_bushing','front_clevis','front_bushing')
            before = {name:fingerprint(bpy.data.objects[name]) for name in eye_names}
            evidence.update(full_barrel(item['barrel']))
            after = {name:fingerprint(bpy.data.objects[name]) for name in eye_names}
            if before != after:
                raise RuntimeError('The external eye geometry changed')
            evidence['external_eye_geometry_unchanged'] = True
            evidence['internal_piston_and_gland_removed'] = not any(
                obj.name == 'piston_body' or obj.name.startswith(('piston_ring_', 'gland_ring_'))
                for obj in bpy.context.scene.objects if obj.type == 'MESH')
            selected = meshes()
        else:
            raise ValueError('Unexpected revision kind')
        if not selected:
            raise RuntimeError('No meshes to export')
        cam = camera_for(scene)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = selected[0]
        bpy.ops.export_scene.gltf(filepath=str(output / 'mesh.glb'), export_format='GLB',
                                  use_selection=True, export_yup=False, export_apply=True)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(output / 'model.blend'))
        evidence.update({'mesh_object_count':len(selected),'mesh_names':[o.name for o in selected],
                         'packed_images':len([i for i in bpy.data.images if i.source == 'FILE' and i.packed_file]),
                         'glb':str(output / 'mesh.glb'),'blend':str(output / 'model.blend')})
        (output / 'geometry-check.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
        summaries.append({'id':item['id'],'out':str(output),'evidence':evidence})
        print('BUILT',item['id'],len(selected),'parts',flush=True)
    for item in request['items']:
        output = Path(item['out'])
        bpy.ops.wm.open_mainfile(filepath=str(output / 'model.blend'), load_ui=False)
        scene = bpy.context.scene
        cam = bpy.data.objects.get('feedback_review_camera') or camera_for(scene)
        scene.camera = cam
        render(scene, cam, meshes(), output / 'renders')
        print('RENDERED',item['id'],flush=True)
    (Path(request['out']) / request.get('summary', 'worker-summary.json')).write_text(
        json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')


main()
