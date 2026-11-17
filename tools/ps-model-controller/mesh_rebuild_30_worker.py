"""Blender worker: part-based rebuild of item 30, a black HCIC hydraulic cylinder.

Evidence: one 1254x1254 clean photograph and its matching rust derivative of a
black-painted hydraulic cylinder with a chrome piston rod, a forged rod-end eye
with a steel bore bushing, an auxiliary single-shear lug, a welded base clevis,
two side port fittings at the base end and one welded elbow port below the rod
gland.  The supplied six-view sheet is only an uncalibrated category reference
for a different black cylinder with a yellow label; it is not used as a
measurement and its yellow label is not copied.

The old TripoSR library mesh is a fused, soft shell: the eye, lugs, ports and
rod are merged, the three through-bores are not true openings and the surface is
grey noise.  It is retained unchanged and only rendered for comparison.

Coordinates: Z up, cylinder axis along Z, rod/eye at +Z, base lug at -Z.  All
three pin bores run along X (the front camera axis), so the front view shows the
holes face-on, as in the photograph.  The candidate is manual-only and is not
accepted, placed automatically or merged into the formal composite.
"""
import json
import math
import random
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
D = request['config']['dimensions']
RR = request['config']['render']
TEX = request['textures']

objects = []
metal_objects = []
nonmetal_objects = []
MATS = {}
CAM = None
METAL_MATERIALS = {'chrome', 'machined', 'dark_steel'}


# ---------------------------------------------------------------------------
# Materials and UV helpers
# ---------------------------------------------------------------------------
def load_image(path, *, noncolor=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    if noncolor:
        image.colorspace_settings.name = 'Non-Color'
    image.pack()
    return image


def make_material(key, spec, *, metallic=1.0, roughness=0.5, coat=0.0,
                  coat_roughness=0.12, alpha=False):
    material = bpy.data.materials.new(spec.get('label', key))
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    principled.inputs['IOR'].default_value = 1.45
    image = load_image(spec['base'])
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = image
    tex.interpolation = 'Linear'
    links.new(tex.outputs['Color'], principled.inputs['Base Color'])
    if spec.get('mr'):
        packed = nodes.new('ShaderNodeTexImage')
        packed.image = load_image(spec['mr'], noncolor=True)
        packed.interpolation = 'Linear'
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(packed.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    else:
        principled.inputs['Metallic'].default_value = metallic
        principled.inputs['Roughness'].default_value = roughness
    if spec.get('normal'):
        normal_tex = nodes.new('ShaderNodeTexImage')
        normal_tex.image = load_image(spec['normal'], noncolor=True)
        normal_tex.interpolation = 'Linear'
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = spec.get('normal_strength', 0.75)
        links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    if coat:
        principled.inputs['Coat Weight'].default_value = coat
        principled.inputs['Coat Roughness'].default_value = coat_roughness
    if alpha:
        # Blender 4.5's glTF exporter detects MASK from an explicit clip
        # node, not from the legacy blend_method.
        clip = nodes.new('ShaderNodeMath')
        clip.operation = 'ROUND'
        clip.use_clamp = False
        links.new(tex.outputs['Alpha'], clip.inputs[0])
        links.new(clip.outputs[0], principled.inputs['Alpha'])
        material.blend_method = 'CLIP'
        material.alpha_threshold = 0.5
        material.use_backface_culling = True
    material['candidate_number'] = 30
    material['texture_role'] = key
    return material


def uv_data(obj):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    return obj.data.uv_layers.active.data


def uv_box(obj, scale=4.0):
    layer = uv_data(obj)
    mesh = obj.data
    for poly in mesh.polygons:
        normal = poly.normal
        axis = max(range(3), key=lambda i: abs(normal[i]))
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            if axis == 0:
                u, v = co.y * scale, co.z * scale
            elif axis == 1:
                u, v = co.x * scale, co.z * scale
            else:
                u, v = co.x * scale, co.y * scale
            layer[li].uv = (u, v)


def uv_cyl_z(obj, z0, z1, radius):
    layer = uv_data(obj)
    mesh = obj.data
    for poly in mesh.polygons:
        normal = poly.normal
        us = []
        if abs(normal.z) > .94:
            for li in poly.loop_indices:
                co = mesh.vertices[mesh.loops[li].vertex_index].co
                layer[li].uv = (.5 + co.x / (2 * radius), .5 + co.y / (2 * radius))
            continue
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            u = (math.atan2(co.y, co.x) / math.tau) % 1.0
            v = (co.z - z0) / max(1e-8, z1 - z0)
            us.append(u)
            layer[li].uv = (u, v)
        if us and max(us) - min(us) > .5:
            for li in poly.loop_indices:
                u, v = layer[li].uv
                if u < .5:
                    layer[li].uv = (u + 1.0, v)


def uv_cyl_x(obj, x0, x1, cy, cz):
    layer = uv_data(obj)
    mesh = obj.data
    for poly in mesh.polygons:
        normal = poly.normal
        us = []
        if abs(normal.x) > .94:
            for li in poly.loop_indices:
                co = mesh.vertices[mesh.loops[li].vertex_index].co
                layer[li].uv = (.5 + (co.y - cy), .5 + (co.z - cz))
            continue
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            u = (math.atan2(co.z - cz, co.y - cy) / math.tau) % 1.0
            v = (co.x - x0) / max(1e-8, x1 - x0)
            us.append(u)
            layer[li].uv = (u, v)
        if us and max(us) - min(us) > .5:
            for li in poly.loop_indices:
                u, v = layer[li].uv
                if u < .5:
                    layer[li].uv = (u + 1.0, v)


def apply_uv(obj, mode):
    if mode == 'box':
        uv_box(obj)
    elif isinstance(mode, tuple) and mode[0] == 'cyl_z':
        uv_cyl_z(obj, mode[1], mode[2], mode[3])
    elif isinstance(mode, tuple) and mode[0] == 'cyl_x':
        uv_cyl_x(obj, mode[1], mode[2], mode[3], mode[4])
    elif mode == 'none':
        return
    else:
        raise ValueError(f'unknown UV mode {mode}')


def apply_bevel(obj, width, segments=2, angle=34):
    if width <= 0:
        return
    mod = obj.modifiers.new('manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.use_clamp_overlap = True
    active(obj)
    if not obj.select_get():
        obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def active(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def smooth_by_angle(obj, angle=34):
    active(obj)
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))
    except Exception:
        for poly in obj.data.polygons:
            poly.use_smooth = True


def classify(obj, key, *, register=True):
    if key in METAL_MATERIALS:
        metal_objects.append(obj.name)
    else:
        nonmetal_objects.append(obj.name)
    if register:
        objects.append(obj)


def finish(obj, name, key, *, bevel=.001, uv='box', smooth=34,
           register=True, material=None):
    if obj.type != 'MESH':
        active(obj)
        bpy.ops.object.convert(target='MESH')
        obj = bpy.context.object
    obj.name = name
    obj.data.name = name
    if bevel:
        apply_bevel(obj, bevel)
    if smooth is not None:
        smooth_by_angle(obj, smooth)
    obj.data.materials.clear()
    obj.data.materials.append(material or MATS[key])
    apply_uv(obj, uv)
    classify(obj, key, register=register)
    return obj


# ---------------------------------------------------------------------------
# Primitive construction
# ---------------------------------------------------------------------------
def mesh_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def cylinder_z(name, radius, z0, z1, key, *, x=0.0, y=0.0, segs=96,
               bevel=.001, uv='box', constant=False, cutter=False):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=z1 - z0, location=(x, y, (z0 + z1) / 2))
    obj = bpy.context.object
    if cutter:
        obj.hide_render = True
        obj.hide_viewport = True
        return obj
    return finish(obj, name, key, bevel=bevel, uv=uv)


def along(name, start, end, radius, key, *, segs=48, bevel=.0008, uv='box'):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv)


def hex_axis(name, radius, thickness, center, axis, key, *, bevel=.0005):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius,
                                        depth=thickness, location=center)
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat('Z', 'Y').to_euler()
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv='box', smooth=8)


def torus(name, major, minor, location, axis, key, *, major_segments=96,
          minor_segments=10, bevel=0.0, uv='box', register=True):
    rotations = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0),
                 'Y': (math.pi / 2, 0, 0)}
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=location, rotation=rotations[axis],
                                     major_radius=major, minor_radius=minor)
    obj = bpy.context.object
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=30,
                  register=register)


def torus_axis(name, major, minor, location, axis, key, *, major_segments=64,
               minor_segments=8):
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=location, major_radius=major,
                                     minor_radius=minor)
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat('Z', 'Y').to_euler()
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=0, uv='box', smooth=30)


def ellipsoid(name, location, radius, key, *, segs=64, rings=32, uv='box'):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings,
                                         radius=1.0, location=location)
    obj = bpy.context.object
    obj.scale = radius
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, key, bevel=0, uv=uv, smooth=34)


def box(name, center, size, key, *, rotation=(0, 0, 0), bevel=.001, uv='box'):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    obj.rotation_euler = rotation
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv)


def annulus_x(name, x0, x1, cy, cz, r_in, r_out, key, *, segs=128, bevel=.0015):
    cx = (x0 + x1) / 2
    half = abs(x1 - x0) / 2
    verts = []
    rings = []
    for x in (-half, half):
        for r in (r_in, r_out):
            start = len(verts)
            for j in range(segs):
                t = math.tau * j / segs
                verts.append((cx + x, cy + r * math.cos(t), cz + r * math.sin(t)))
            rings.append(list(range(start, start + segs)))
    inner0, outer0, inner1, outer1 = rings
    faces = []
    for j in range(segs):
        j2 = (j + 1) % segs
        faces.append((outer0[j], outer0[j2], outer1[j2], outer1[j]))
        faces.append((inner0[j2], inner0[j], inner1[j], inner1[j2]))
        faces.append((inner0[j], inner0[j2], outer0[j2], outer0[j]))
        faces.append((inner1[j2], inner1[j], outer1[j], outer1[j2]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel,
                  uv=('cyl_x', x0, x1, cy, cz), smooth=34)


def plate_stadium_yz(name, x0, x1, half_width, z_top, z_center, key, *,
                     segs=64, bevel=.0012):
    points = [(-half_width, z_top)]
    for k in range(segs + 1):
        a = math.pi + math.pi * k / segs
        points.append((half_width * math.cos(a), z_center + half_width * math.sin(a)))
    points.append((half_width, z_top))
    n = len(points)
    verts = [(x0, y, z) for y, z in points] + [(x1, y, z) for y, z in points]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(n + i for i in range(n)))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel, uv='box', smooth=30)


def apply_boolean(target, cutter, name):
    cutter.hide_viewport = False
    cutter.hide_render = False
    cutter.hide_set(False)
    active(cutter)
    mod = target.modifiers.new(name, 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    mod.material_mode = 'TRANSFER'
    active(target)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    bm = bmesh.new()
    bm.from_mesh(target.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(target.data)
    bm.free()
    target.data.update()


def tube_path(name, points, radius, key, *, sides=12, wobble=0.0, seed=30,
              uv='box'):
    pts = [Vector(p) for p in points]
    rng = random.Random(seed)
    verts = []
    rings = []
    for i, p in enumerate(pts):
        if i == 0:
            tangent = (pts[1] - pts[0]).normalized()
        elif i == len(pts) - 1:
            tangent = (pts[-1] - pts[-2]).normalized()
        else:
            tangent = (pts[i + 1] - pts[i - 1]).normalized()
        ref = Vector((0, 0, 1))
        if abs(tangent.dot(ref)) > .92:
            ref = Vector((0, 1, 0))
        side = tangent.cross(ref).normalized()
        up = tangent.cross(side).normalized()
        r = radius * (1.0 + wobble * (rng.random() - .5) * 2.0)
        start = len(verts)
        for j in range(sides):
            t = math.tau * j / sides
            verts.append(tuple(p + side * (math.cos(t) * r) + up * (math.sin(t) * r)))
        rings.append(list(range(start, start + sides)))
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append((a[j], a[j2], b[j2], b[j]))
    faces.append(tuple(reversed(rings[0])))
    faces.append(tuple(rings[-1]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=0, uv=uv, smooth=40)


def decal_shell(name, z_center, length, arc_width, radius, thickness, material):
    segs = 64
    z0, z1 = z_center - length / 2, z_center + length / 2
    half_angle = (arc_width / 2) / radius
    verts = []
    rings = []
    for r in (radius, radius + thickness):
        for z in (z0, z1):
            start = len(verts)
            for j in range(segs + 1):
                a = -half_angle + 2 * half_angle * j / segs
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(list(range(start, start + segs + 1)))
    r0z0, r0z1, r1z0, r1z1 = rings
    faces = []
    for j in range(segs):
        j2 = j + 1
        faces.append((r0z0[j], r0z0[j2], r0z1[j2], r0z1[j]))
        faces.append((r1z0[j2], r1z0[j], r1z1[j], r1z1[j2]))
        faces.append((r0z0[j], r1z0[j], r1z0[j2], r0z0[j2]))
        faces.append((r0z1[j2], r1z1[j2], r1z1[j], r0z1[j]))
    faces.append((r0z0[0], r0z1[0], r1z1[0], r1z0[0]))
    faces.append((r1z0[segs], r1z1[segs], r0z1[segs], r0z0[segs]))
    obj = mesh_object(name, verts, faces)
    active(obj)
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(material)
    layer = uv_data(obj)
    for poly in obj.data.polygons:
        for li in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[li].vertex_index].co
            u = (co.z - z0) / max(1e-8, z1 - z0)
            a = math.atan2(co.y, co.x)
            v = (half_angle - a) / max(1e-8, 2 * half_angle)
            layer[li].uv = (u, v)
    objects.append(obj)
    return obj


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def build_barrel_and_base():
    b = D['barrel']
    cylinder_z('barrel_tube', b['radius'], b['z0'], b['z1'], 'paint_worn',
               segs=128, bevel=.0009, uv=('cyl_z', b['z0'], b['z1'], b['radius']))
    dome = D['dome']
    ellipsoid('barrel_dome', (0, 0, b['z0'] + .0005),
              (dome['radius'], dome['radius'], b['z0'] - dome['z0']),
              'paint_worn', segs=96, rings=48,
              uv=('cyl_z', dome['z0'], b['z0'], dome['radius']))
    torus('barrel_dome_weld', b['radius'] + .0008, .0034, (0, 0, b['z0'] + .0018),
          'Z', 'rust', major_segments=128, minor_segments=10)

    lug = D['base_lug']
    plate = plate_stadium_yz('base_lug_plate', -lug['x_half'], lug['x_half'],
                             lug['half_width'], lug['z_top'], lug['z_center'],
                             'forged', segs=64, bevel=.0014)
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=lug['bore_radius'],
                                        depth=.2, location=(0, 0, lug['z_center']),
                                        rotation=(0, math.pi / 2, 0))
    cutter = bpy.context.object
    cutter.hide_render = True
    cutter.hide_viewport = True
    apply_boolean(plate, cutter, 'base lug through bore')
    apply_bevel(plate, .001, segments=2, angle=30)
    smooth_by_angle(plate, 30)
    annulus_x('base_lug_bushing', -lug['bush_x_half'], lug['bush_x_half'],
              0, lug['z_center'], lug['bore_radius'], lug['bush_outer'],
              'machined', segs=96, bevel=.0008)
    annulus_x('base_lug_bore_liner', -lug['x_half'] - .0001,
              lug['x_half'] + .0001, 0, lug['z_center'],
              lug['bore_radius'] - .0023, lug['bore_radius'] - .0003,
              'dark_steel', segs=96, bevel=.0003)
    for sign, label in ((1, 'pos'), (-1, 'neg')):
        pts = []
        limit = lug['half_width'] * .93
        for k in range(25):
            y = -limit + 2 * limit * k / 24
            z = -.312 + .008 * (1 - (y / limit) ** 2)
            pts.append((sign * (lug['x_half'] + .0012), y, z))
        tube_path(f'base_lug_weld_{label}', pts, .0030, 'rust', sides=9,
                  wobble=.42, seed=301 + sign)
    # A low welded boss and two hex nipples near the base end.
    p = D['base_ports']
    angle = math.radians(p['azimuth_deg'])
    direction = Vector((math.cos(angle), math.sin(angle), 0))
    pad_center = direction * (b['radius'] - .001)
    for z, suffix in ((p['z_a'], 'a'), (p['z_b'], 'b')):
        box(f'base_port_pad_{suffix}', (pad_center.x, pad_center.y, z),
            (.014, p['pad_width'], p['pad_len']), 'forged',
            rotation=(0, 0, angle), bevel=.003)
        torus_axis(f'base_port_pad_weld_{suffix}', .021, .0022,
                   tuple(direction * (b['radius'] + .0008) + Vector((0, 0, z))),
                   direction, 'rust')
        start = tuple(direction * (b['radius'] + .004) + Vector((0, 0, z)))
        end = tuple(direction * (b['radius'] + .022) + Vector((0, 0, z)))
        hex_axis(f'base_port_hex_{suffix}', p['hex_r'], p['hex_len'],
                 tuple(direction * (b['radius'] + .014) + Vector((0, 0, z))),
                 direction, 'machined', bevel=.0005)
        nipple_start = tuple(direction * (b['radius'] + .024) + Vector((0, 0, z)))
        nipple_end = tuple(direction * (b['radius'] + .038) + Vector((0, 0, z)))
        along(f'base_port_nipple_{suffix}', nipple_start, nipple_end,
              p['nipple_r'], 'machined', segs=32, bevel=.0003)
        along(f'base_port_bore_{suffix}',
              tuple(direction * (b['radius'] + .0355) + Vector((0, 0, z))),
              tuple(direction * (b['radius'] + .0385) + Vector((0, 0, z))),
              p['nipple_r'] * .62, 'dark_steel', segs=24, bevel=0)


def build_head_gland_rod():
    h = D['head']
    cylinder_z('head_collar', h['collar_radius'], h['collar_z0'], h['collar_z1'],
               'paint_clean', segs=112, bevel=.0012,
               uv=('cyl_z', h['collar_z0'], h['collar_z1'], h['collar_radius']))
    torus('head_collar_seam', D['barrel']['radius'] + .0008, .0018,
          (0, 0, h['collar_z0'] + .001), 'Z', 'dark_steel',
          major_segments=112, minor_segments=8)
    torus('head_collar_groove', h['collar_radius'] + .0002, .0014,
          (0, 0, h['collar_z0'] + .024), 'Z', 'dark_steel',
          major_segments=112, minor_segments=8)
    hex_axis('head_set_screw', .0045, .0042, (0, h['collar_radius'] - .0004,
              h['collar_z0'] + .018), (0, 1, 0), 'dark_steel', bevel=.0003)
    cylinder_z('rod_gland', h['gland_radius'], h['gland_z0'], h['gland_z1'],
               'machined', segs=112, bevel=.0008,
               uv=('cyl_z', h['gland_z0'], h['gland_z1'], h['gland_radius']))
    for i, z in enumerate((h['gland_z0'] + .005, h['gland_z0'] + .011)):
        torus(f'gland_groove_{i}', h['gland_radius'] + .0003, .0011,
              (0, 0, z), 'Z', 'dark_steel', major_segments=112, minor_segments=8)
    cylinder_z('rod_wiper', h['wiper_radius'], h['wiper_z0'], h['wiper_z1'],
               'dark_steel', segs=96, bevel=.0005,
               uv=('cyl_z', h['wiper_z0'], h['wiper_z1'], h['wiper_radius']))

    r = D['rod']
    cylinder_z('rod_chrome', r['radius'], r['z0'], r['z1'], 'chrome',
               segs=96, bevel=.0005,
               uv=('cyl_z', r['z0'], r['z1'], r['radius']))
    cylinder_z('rod_step_collar', r['step_radius'], r['step_z0'], r['step_z1'],
               'chrome', segs=96, bevel=.0008,
               uv=('cyl_z', r['step_z0'], r['step_z1'], r['step_radius']))
    torus('rod_step_groove', r['step_radius'] + .0002, .0009,
          (0, 0, (r['step_z0'] + r['step_z1']) / 2), 'Z', 'dark_steel',
          major_segments=96, minor_segments=8)

    s = D['eye_socket']
    bpy.ops.mesh.primitive_cone_add(vertices=96, radius1=s['r0'], radius2=s['r1'],
                                    depth=s['z1'] - s['z0'],
                                    location=(0, D['eye']['y'], (s['z0'] + s['z1']) / 2))
    socket = bpy.context.object
    finish(socket, 'rod_socket', 'forged', bevel=.0012, uv='box', smooth=34)
    n = D['eye_neck']
    bpy.ops.mesh.primitive_cone_add(vertices=96, radius1=n['r0'], radius2=n['r1'],
                                    depth=n['z1'] - n['z0'],
                                    location=(0, D['eye']['y'], (n['z0'] + n['z1']) / 2))
    neck = bpy.context.object
    finish(neck, 'eye_neck', 'forged', bevel=.0012, uv='box', smooth=34)
    ellipsoid('eye_web', (0, D['eye']['y'], .509), (.036, .045, .026), 'forged',
              segs=64, rings=32, uv='box')


def build_eye_and_clevis():
    e = D['eye']
    annulus_x('rod_eye_ring', -e['x_half'], e['x_half'], e['y'], e['z'],
              e['bush_outer'] - .0015, e['outer_radius'], 'forged',
              segs=160, bevel=.0028)
    annulus_x('rod_eye_bushing', -e['bush_x_half'], e['bush_x_half'],
              e['y'], e['z'], e['bore_radius'], e['bush_outer'],
              'machined', segs=128, bevel=.0008)
    annulus_x('rod_eye_bore_liner', -e['x_half'] + .001, e['x_half'] - .001,
              e['y'], e['z'], e['bore_radius'] - .0028,
              e['bore_radius'] - .0003, 'dark_steel', segs=128, bevel=.0003)
    c = D['small_clevis']
    along('small_clevis_boss', (0, c['boss_y0'], c['z']),
          (0, c['y'] + .012, c['z']), c['boss_r'], 'forged',
          segs=48, bevel=.0008)
    annulus_x('small_clevis_lug', -c['x_half'], c['x_half'], c['y'], c['z'],
              c['bore_radius'], c['outer_radius'], 'forged',
              segs=96, bevel=.0012)
    annulus_x('small_clevis_bushing', -c['bush_x_half'], c['bush_x_half'],
              c['y'], c['z'], c['bore_radius'], c['bush_outer'],
              'machined', segs=96, bevel=.0006)
    annulus_x('small_clevis_bore_liner', -c['x_half'] + .001,
              c['x_half'] - .001, c['y'], c['z'],
              c['bore_radius'] - .0022, c['bore_radius'] - .0002,
              'dark_steel', segs=96, bevel=.00025)


def build_elbow_port():
    e = D['elbow_port']
    b = D['barrel']
    angle = math.radians(e['azimuth_deg'])
    direction = Vector((math.cos(angle), math.sin(angle), 0))
    z = e['z']
    box('elbow_pad', tuple(direction * (b['radius'] - .001) + Vector((0, 0, z))),
        (.014, .048, .046), 'forged', rotation=(0, 0, angle), bevel=.003)
    torus_axis('elbow_pad_weld', .022, .0022,
               tuple(direction * (b['radius'] + .0008) + Vector((0, 0, z))),
               direction, 'rust')
    radial_end = direction * (b['radius'] + .042) + Vector((0, 0, z))
    along('elbow_boss', tuple(direction * (b['radius'] + .002) + Vector((0, 0, z))),
          tuple(radial_end), e['tube_r'], 'dark_steel', segs=28, bevel=.0003)
    # quarter bend from radially outward (+dir) to downward (-Z)
    center = direction * (b['radius'] + .042) + Vector((0, 0, z - .012))
    pts = []
    for k in range(13):
        theta = math.pi / 2 * (1 - k / 12)  # 90 deg -> 0 deg
        pts.append(tuple(center + direction * (.012 * math.cos(theta))
                         + Vector((0, 0, .012 * math.sin(theta)))))
    tube_path('elbow_bend', pts, e['tube_r'], 'dark_steel', sides=14, seed=300)
    leg_top = direction * (b['radius'] + .054) + Vector((0, 0, z - .012))
    hex_axis('elbow_hex', e['hex_r'], e['hex_len'],
             tuple(leg_top + Vector((0, 0, -.010))), (0, 0, 1), 'machined',
             bevel=.0005)
    nipple_top = leg_top + Vector((0, 0, -.020))
    along('elbow_nipple', tuple(nipple_top),
          tuple(nipple_top + Vector((0, 0, -e['nipple_len']))),
          e['nipple_r'], 'machined', segs=28, bevel=.0003)
    along('elbow_bore',
          tuple(nipple_top + Vector((0, 0, -e['nipple_len'] + .004))),
          tuple(nipple_top + Vector((0, 0, -e['nipple_len'] + .001))),
          e['nipple_r'] * .62, 'dark_steel', segs=24, bevel=0)


def build_logo():
    l = D['logo']
    decal_shell('HCIC_logo_decal', l['z_center'], l['length'], l['arc_width'],
                l['radius'], .0004, MATS['logo'])


def build_all():
    build_barrel_and_base()
    build_head_gland_rod()
    build_eye_and_clevis()
    build_elbow_port()
    build_logo()


# ---------------------------------------------------------------------------
# Scene, checks, renders, export
# ---------------------------------------------------------------------------
def init_camera(scene):
    data = bpy.data.cameras.new('review camera')
    cam = bpy.data.objects.new('review camera', data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_camera(cam, eye, target, *, ortho_scale=None, fov=40, roll_deg=0.0):
    cam.location = Vector(eye)
    cam.rotation_mode = 'QUATERNION'
    quat = (Vector(target) - cam.location).to_track_quat('-Z', 'Y')
    if roll_deg:
        quat = quat @ Quaternion((0, 0, 1), math.radians(roll_deg))
    cam.rotation_quaternion = quat
    cam.data.type = 'ORTHO' if ortho_scale else 'PERSP'
    if ortho_scale:
        cam.data.ortho_scale = ortho_scale
    else:
        cam.data.angle = math.radians(fov)


def look(direction, center, span, *, ortho_scale=None, roll_deg=0.0):
    direction = Vector(direction).normalized()
    set_camera(CAM, Vector(center) + direction * span * 5, center,
               ortho_scale=ortho_scale or span * 1.27, roll_deg=roll_deg)
    return CAM


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


def object_center(name):
    lo, hi, center, span = bounds([bpy.data.objects[name]])
    return center


def draw(scene, path, resolution, samples, *, transparent=True):
    # All review renders keep alpha: boards composite them on a neutral card
    # and the image-based checks use the alpha mask instead of guessing the
    # background colour.  The world still supplies reflections and light.
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.cycles.use_denoising = samples >= 12
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def set_visible(names):
    for obj in objects:
        obj.hide_render = obj.name not in names


def bvh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    matrix = obj.matrix_world
    verts = [matrix @ v.co for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(verts, faces, epsilon=1e-7)


def ray_hit(name, origin, direction, distance):
    hit = bvh(bpy.data.objects[name]).ray_cast(Vector(origin),
                                               Vector(direction).normalized(), distance)
    return hit[0] is not None


def ray_clear(name, origin, direction, distance):
    return not ray_hit(name, origin, direction, distance)


def ray_depth(name, origin, direction, distance):
    hit = bvh(bpy.data.objects[name]).ray_cast(Vector(origin),
                                               Vector(direction).normalized(), distance)
    return None if hit[0] is None else float(hit[3])


def closure(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bad = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
    bm.free()
    return bad


def intersecting_faces(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm, epsilon=1e-8)
    bad = set()
    for a, b in tree.overlap(tree):
        if a >= b:
            continue
        if set(bm.faces[a].verts).isdisjoint(set(bm.faces[b].verts)):
            bad.add((a, b))
    bm.free()
    return len(bad)


def run_ray_checks():
    e = D['eye']
    c = D['small_clevis']
    lug = D['base_lug']
    checks = {'eyes': {}, 'bushings': {}, 'ports': {}, 'objects': {}}
    eye_origin = (-.20, e['y'], e['z'])
    checks['eyes']['rod_eye'] = {
        'through': ray_clear('rod_eye_ring', eye_origin, (1, 0, 0), .40),
        'ring_solid': ray_hit('rod_eye_ring',
                              (-.20, e['y'] + .054, e['z']), (1, 0, 0), .40),
    }
    checks['bushings']['rod_eye'] = {
        'present': ray_hit('rod_eye_bushing',
                           (-.20, e['y'] + (e['bore_radius'] + e['bush_outer']) / 2,
                            e['z']), (1, 0, 0), .40),
        'liner_clear': ray_clear('rod_eye_bore_liner', eye_origin, (1, 0, 0), .40)}
    clevis_origin = (-.20, c['y'], c['z'])
    checks['eyes']['small_clevis'] = {
        'through': ray_clear('small_clevis_lug', clevis_origin, (1, 0, 0), .40),
        'lug_solid': ray_hit('small_clevis_lug',
                             (-.20, c['y'] + .024, c['z']), (1, 0, 0), .40),
    }
    checks['bushings']['small_clevis'] = {
        'present': ray_hit('small_clevis_bushing',
                           (-.20, c['y'] + (c['bore_radius'] + c['bush_outer']) / 2,
                            c['z']), (1, 0, 0), .40),
        'liner_clear': ray_clear('small_clevis_bore_liner', clevis_origin,
                                 (1, 0, 0), .40)}
    base_origin = (-.20, 0, lug['z_center'])
    checks['eyes']['base_lug'] = {
        'through': ray_clear('base_lug_plate', base_origin, (1, 0, 0), .40),
        'plate_solid': ray_hit('base_lug_plate',
                               (-.20, 0, lug['z_center'] + .036), (1, 0, 0), .40),
    }
    checks['bushings']['base_lug'] = {
        'present': ray_hit('base_lug_bushing',
                           (-.20, 0, lug['z_center'] +
                            (lug['bore_radius'] + lug['bush_outer']) / 2),
                           (1, 0, 0), .40),
        'liner_clear': ray_clear('base_lug_bore_liner', base_origin,
                                 (1, 0, 0), .40)}
    p = D['base_ports']
    angle = math.radians(p['azimuth_deg'])
    d = Vector((math.cos(angle), math.sin(angle), 0))
    z = p['z_a']
    outside = d * .13 + Vector((0, 0, z))
    depth = ray_depth('base_port_nipple_a', outside, -d, .06)
    checks['ports']['base_nipple_a'] = {
        'outer_face_depth_m': round(depth, 5) if depth is not None else None,
        'blind_plug_hit': ray_hit('base_port_bore_a',
                                  d * .070 + Vector((0, 0, z)), d, .04),
    }
    # A ray through the centre of the eye must not hit the neck/web behind it.
    checks['eyes']['rod_eye_web_clear'] = {
        'clear': ray_clear('eye_web', eye_origin, (1, 0, 0), .40)}
    checks['objects'] = {
        'logo': 'HCIC_logo_decal' in bpy.data.objects,
        'logo_texture': 'logo' in MATS and bool(MATS['logo'].node_tree),
        'parts': len(objects),
    }
    return checks


def main():
    global CAM
    scene = legacy.setup_scene('CPU', 512)
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    for key, spec in request['materials'].items():
        MATS[key] = make_material(
            key, spec,
            metallic=spec.get('metallic', 1.0),
            roughness=spec.get('roughness', .5),
            coat=spec.get('coat', 0.0),
            coat_roughness=spec.get('coat_roughness', .12),
            alpha=spec.get('alpha', False))
    CAM = init_camera(scene)
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Strength'].default_value = .58
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 120
        fill.data.shape = 'RECTANGLE'
        fill.data.size = fill.data.size_y = 3.2
        fill.location = (-3.0, 2.4, 2.4)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 220, 3.0),
            ('front reflection', (3.0, 1.4, 1.0), 110, 1.5),
            ('top fill', (0.0, 0.0, 4.0), 80, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 70, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 32, 1.4)):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy = energy
        data.shape = 'RECTANGLE'
        data.size = data.size_y = size
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

    build_all()
    bpy.context.view_layer.update()
    rays = run_ray_checks()
    low, high, center, span = bounds(objects)
    shift = -center
    for obj in objects:
        obj.location += shift
    bpy.context.view_layer.update()
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    out.mkdir(parents=True, exist_ok=True)
    render_dir = out / 'renders'
    render_dir.mkdir(exist_ok=True)

    bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'), export_format='GLB',
                              use_selection=True, export_yup=False,
                              export_apply=True, export_image_format='AUTO',
                              export_materials='EXPORT')
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

    low, high, center, span = bounds(objects)
    parts = [{'name': obj.name, 'vertices': len(obj.data.vertices),
              'faces': len(obj.data.polygons),
              'material': obj.data.materials[0].name if obj.data.materials else None,
              'metal': obj.name in metal_objects}
             for obj in objects]
    closures = {obj.name: closure(obj) for obj in objects}
    intersections = {obj.name: intersecting_faces(obj) for obj in objects}
    directions = {'front': (1, 0, .02), 'back': (-1, 0, .02),
                  'left': (0, -1, .02), 'right': (0, 1, .02),
                  'top': (0, 0, 1), 'bottom': (0, 0, -1)}
    for name, direction in directions.items():
        scale = span * 1.13 if name in ('front', 'back', 'left', 'right') else .30
        look(direction, center, span, ortho_scale=scale)
        draw(scene, render_dir / f'new-{name}.png', RR['six_resolution'],
             RR['six_samples'])

    # Photo-like three-quarter camera (roll makes the Z axis diagonal).
    source_dir = Vector((1.0, -.88, .34)).normalized()
    set_camera(CAM, center + source_dir * 1.90, center, fov=37.5, roll_deg=36)
    draw(scene, render_dir / 'new-source.png', 512, RR['six_samples'])
    draw(scene, render_dir / 'new-source-alpha.png', 512, RR['six_samples'],
         transparent=True)

    base_port_center = (object_center('base_port_hex_a') +
                        object_center('base_port_hex_b')) / 2
    elbow_port_center = object_center('elbow_hex')
    targets = {
        'rod-eye': object_center('rod_eye_bushing'),
        'small-clevis': object_center('small_clevis_bushing'),
        'base-lug': object_center('base_lug_bushing'),
        'head-gland': object_center('rod_gland'),
        'logo': object_center('HCIC_logo_decal'),
        'base-ports': base_port_center,
        'elbow-port': elbow_port_center,
        'overall': center,
    }
    detail_specs = [
        ('rod-eye', (1.0, -.35, .16), .20),
        ('small-clevis', (1.0, -.52, .12), .115),
        ('base-lug', (1.0, -.35, .10), .18),
        ('head-gland', (1.0, -.25, .18), .18),
        ('logo', (1.0, -.02, .02), .32),
        ('base-ports', (.92, .38, .10), .17),
        ('elbow-port', (.55, .82, .12), .19),
        ('overall', (1.0, -.85, .35), span * 1.15),
    ]
    for name, direction, scale in detail_specs:
        look(direction, targets[name], span, ortho_scale=scale)
        draw(scene, render_dir / f'detail-{name}.png', RR['detail_resolution'],
             RR['detail_samples'])

    # Isolation renders: eyes, lugs, fittings and the metal-only pass.
    set_visible({'rod_eye_ring', 'rod_eye_bushing', 'rod_eye_bore_liner',
                 'eye_neck', 'eye_web', 'rod_socket', 'small_clevis_boss',
                 'small_clevis_lug', 'small_clevis_bushing',
                 'small_clevis_bore_liner'})
    look((1.0, -.35, .16), targets['rod-eye'], span, ortho_scale=.20)
    draw(scene, render_dir / 'isolate-rod-eye.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({'small_clevis_boss', 'small_clevis_lug',
                 'small_clevis_bushing', 'small_clevis_bore_liner'})
    look((1.0, -.52, .12), targets['small-clevis'], span, ortho_scale=.115)
    draw(scene, render_dir / 'isolate-small-clevis.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({'base_lug_plate', 'base_lug_bushing', 'base_lug_bore_liner',
                 'base_lug_weld_pos', 'base_lug_weld_neg'})
    look((1.0, -.35, .10), targets['base-lug'], span, ortho_scale=.18)
    draw(scene, render_dir / 'isolate-base-lug.png', RR['detail_resolution'],
         RR['detail_samples'])
    fittings = {obj.name for obj in objects if obj.name.startswith(
        ('base_port_', 'elbow_'))}
    set_visible(fittings)
    look((.75, .65, .15), (base_port_center + elbow_port_center) / 2, span,
         ortho_scale=.50)
    draw(scene, render_dir / 'isolate-ports.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible(set(metal_objects))
    look((1.0, -.7, .28), center, span, ortho_scale=span * 1.15)
    draw(scene, render_dir / 'isolate-metal.png', RR['detail_resolution'],
         RR['detail_samples'])

    set_visible({obj.name for obj in objects})
    look((1.0, -.88, .35), center, span, ortho_scale=span * 1.15)
    draw(scene, render_dir / 'new-material-3q.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({obj.name for obj in objects})
    clay = bpy.data.materials.new('Clay check only')
    clay.use_nodes = True
    cb = clay.node_tree.nodes.get('Principled BSDF')
    cb.inputs['Base Color'].default_value = (.34, .35, .34, 1)
    cb.inputs['Roughness'].default_value = .82
    cb.inputs['Metallic'].default_value = 0
    original_materials = {obj.name: obj.data.materials[0] for obj in objects}
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    look(directions['front'], center, span, ortho_scale=span * 1.13)
    draw(scene, render_dir / 'new-clay-front.png', RR['six_resolution'],
         RR['six_samples'])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(original_materials[obj.name])

    # Two-light metal reflection pass (dielectric paint hidden).
    set_visible(set(metal_objects))
    main_light = bpy.data.objects['main softbox']
    for name, location in (('a', (2.4, -2.8, 3.8)), ('b', (-3.0, 4.0, 5.0))):
        main_light.location = location
        main_light.rotation_euler = (-Vector(location)).to_track_quat('-Z', 'Y').to_euler()
        look((.90, -.40, .35), center, span, ortho_scale=span * 1.15)
        draw(scene, render_dir / f'metal-only-light-{name}.png', 512,
             RR['detail_samples'])
    main_light.location = (2.4, -2.8, 3.8)
    main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
    set_visible({obj.name for obj in objects})
    look(directions['front'], center, span, ortho_scale=span * 1.13)
    draw(scene, render_dir / 'new-light-alternate.png', RR['six_resolution'],
         RR['six_samples'])

    turns = RR['turn_count']
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), center, span,
             ortho_scale=span * 1.18)
        draw(scene, render_dir / f'turn-new-{i:02d}.png', RR['turn_resolution'],
             RR['turn_samples'])
    # Clay turntable for the check page.
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), center, span,
             ortho_scale=span * 1.18)
        draw(scene, render_dir / f'turn-clay-{i:02d}.png', RR['turn_resolution'],
             RR['turn_samples'])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(original_materials[obj.name])

    # Transparent silhouettes for the uncalibrated diagnostics.
    scene.render.image_settings.color_mode = 'RGBA'
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), center, span,
             ortho_scale=span * 1.18)
        draw(scene, render_dir / f'sil-new-{i:02d}.png', 256, 4,
             transparent=True)

    # Old model comparison renders.
    for obj in objects:
        obj.hide_render = True
    old = legacy.load_mesh(request['original_mesh'])
    old_low, old_high, old_center, old_span = bounds([old])
    for name, direction in directions.items():
        scale = old_span * 1.10 if name in ('front', 'back', 'left',
                                            'right') else old_span * .55
        look(direction, old_center, old_span, ortho_scale=scale)
        draw(scene, render_dir / f'old-{name}.png', RR['six_resolution'],
             RR['six_samples'])
    set_camera(CAM, Vector((1.9, 0, 0)) + old_center, old_center,
               fov=37.461451)
    draw(scene, render_dir / 'old-source.png', 512, RR['six_samples'])
    draw(scene, render_dir / 'old-source-alpha.png', 512, RR['six_samples'],
         transparent=True)
    old_vertices = np.array([[v.x, v.y, v.z] for v in legacy.world_vertices(old)],
                            dtype=np.float64)
    old_mean = old_vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((old_vertices - old_mean).T))
    order = np.argsort(values)[::-1]
    major = Vector(vectors[:, order[0]]).normalized()
    side = Vector(vectors[:, order[1]]).normalized()
    projection = (old_vertices - old_mean) @ np.array(major)
    length = float(projection.max() - projection.min())
    old_pc = Vector(tuple(old_mean + np.array(major) *
                          ((projection.max() + projection.min()) / 2)))
    def p(t):
        return old_pc + major * (length * t)
    old_targets = {
        'rod-eye': p(.44),
        'small-clevis': p(.38),
        'base-lug': p(-.44),
        'head-gland': p(.28),
        'logo': old_pc,
        'base-ports': p(-.30) + side * (old_span * .10),
        'elbow-port': p(.28) + side * (old_span * .10),
        'overall': old_pc,
    }
    old_directions = {
        'rod-eye': (side + major * .25 + Vector((0, 0, .10))),
        'small-clevis': (side + major * .15 + Vector((0, 0, .06))),
        'base-lug': (side - major * .20 + Vector((0, 0, -.05))),
        'head-gland': (side + Vector((0, 0, .10))),
        'logo': (side + major * .10),
        'base-ports': (side + Vector((0, 0, .10))),
        'elbow-port': (side + major * .20 + Vector((0, 0, .08))),
        'overall': Vector((1.0, -.88, .35)),
    }
    old_scales = {
        'rod-eye': .30, 'small-clevis': .24, 'base-lug': .30,
        'head-gland': .25, 'logo': .55, 'base-ports': .30,
        'elbow-port': .30, 'overall': 1.15,
    }
    for name in old_targets:
        direction = old_directions[name]
        look(tuple(direction), old_targets[name], old_span,
             ortho_scale=old_span * old_scales[name])
        draw(scene, render_dir / f'old-detail-{name}.png',
             max(320, RR['detail_resolution'] // 2), RR['detail_samples'] // 2)
    look((1.0, -.88, .35), old_center, old_span, ortho_scale=old_span * 1.15)
    draw(scene, render_dir / 'old-material-3q.png', RR['detail_resolution'],
         RR['detail_samples'])
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), old_center, old_span,
             ortho_scale=old_span * 1.18)
        draw(scene, render_dir / f'turn-old-{i:02d}.png', RR['turn_resolution'],
             RR['turn_samples'])
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), old_center, old_span,
             ortho_scale=old_span * 1.18)
        draw(scene, render_dir / f'sil-old-{i:02d}.png', 256, 4,
             transparent=True)

    low, high, center, span = bounds(objects)
    data = {
        'parts': parts,
        'bounds': [list(low), list(high)],
        'centre': list(center),
        'span': span,
        'polygons': sum(len(obj.data.polygons) for obj in objects),
        'triangles': sum(max(0, len(p.vertices) - 2)
                         for obj in objects for p in obj.data.polygons),
        'closures': closures,
        'closed_parts': sum(1 for v in closures.values() if v == 0),
        'self_intersecting_components': {k: v for k, v in intersections.items()
                                          if v},
        'intersection_check': ('BVH 同部件不共享顶点的面交叠；不同零件之间故意装配相交，'
                               '未判定'),
        'metal_objects': metal_objects,
        'nonmetal_objects': nonmetal_objects,
        'ray_checks': rays,
        'old': {'bounds': [list(old_low), list(old_high)], 'span': old_span,
                'centre': list(old_center), 'mesh': request['original_mesh']},
        'materials': {key: {'name': MATS[key].name,
                            'metallic_hint': request['materials'][key].get('metallic'),
                            'roughness_hint': request['materials'][key].get('roughness'),
                            'alpha': request['materials'][key].get('alpha', False)}
                      for key in MATS},
        'render': RR,
        'shift': list(shift),
        'coordinate_system': 'Z up; cylinder axis Z; bores along X; +X front camera',
        'status': 'manual_only_pending_review',
        'reference_role': 'category_reference_uncalibrated',
    }
    (out / 'worker-result.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'ok': True, 'parts': len(objects),
                      'glb': str(out / 'mesh.glb')}, ensure_ascii=False))


main()
