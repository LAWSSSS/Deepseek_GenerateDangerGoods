"""Blender worker: part-based rebuild of item 34, a green 3 L water extinguisher.

Evidence: one clean product photograph of a green Oulunte (欧伦泰) MPZ/3
3-litre water-based fire extinguisher and its matching rust derivative.  The
photograph shows:

* a green painted steel cylinder with a rolled bottom rim and a rounded
  shoulder / short neck,
* a green painted valve with a chrome hose coupling, a round pressure gauge
  (red/green/yellow zones) facing the camera, a lime plastic safety ring and a
  yellow tamper strap / hanging instruction tag,
* an upper squeeze lever and a lower fixed handle with a real gap between them,
* a black rubber hose on the viewer's left that curves down to a black nozzle
  at the lower left, with a body clip and a silver ferrule.

The supplied six-view sheet is an uncalibrated visual reference (role
``visual_match_unverified``): it agrees on the overall layout and the back
label, but its handle angle and some details differ from the photograph.  The
photograph wins wherever the two disagree and the sheet is not used as a
measurement.

The old TripoSR library mesh is a fused shell: handles, gauge and hose merge,
there is no open squeeze gap, the labels are smeared and the bottom is not a
rolled rim.  It is retained unchanged and rendered only for comparison.

Coordinates: Z up, cylinder axis Z, front camera on +X.  The hose is on -Y
(viewer's left from the front), the handles extend +Y and the gauge faces +X.
Every part is a separate closed mesh; openings that are checked by rays are the
handle squeeze gap, the safety-ring hole and the nozzle outlet (blind dark
plug).  The candidate is manual-only and is not accepted, placed automatically
or merged into the formal composite.
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
METAL_MATERIALS = {'chrome', 'brass', 'dark_steel'}


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
    if spec.get('glass'):
        principled.inputs['Base Color'].default_value = (.98, .99, .98, 1)
        principled.inputs['Metallic'].default_value = 0.0
        principled.inputs['Roughness'].default_value = .012
        principled.inputs['Transmission Weight'].default_value = 1.0
        principled.inputs['IOR'].default_value = 1.50
        material['candidate_number'] = 34
        material['texture_role'] = key
        return material
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
        normal_map.inputs['Strength'].default_value = spec.get('normal_strength', 0.55)
        links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    if coat:
        principled.inputs['Coat Weight'].default_value = coat
        principled.inputs['Coat Roughness'].default_value = coat_roughness
    if alpha:
        clip = nodes.new('ShaderNodeMath')
        clip.operation = 'ROUND'
        clip.use_clamp = False
        links.new(tex.outputs['Alpha'], clip.inputs[0])
        links.new(clip.outputs[0], principled.inputs['Alpha'])
        material.blend_method = 'CLIP'
        material.alpha_threshold = 0.5
        material.use_backface_culling = True
    material['candidate_number'] = 34
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


def uv_dial_x(obj, x0, x1, cy, cz, radius):
    # The primitive is created at its world location, so the mesh vertices
    # stay centred on the object origin.  Use those local coordinates here;
    # subtracting the world centre would push the sampled UVs far outside 0..1.
    layer = uv_data(obj)
    mesh = obj.data
    for poly in mesh.polygons:
        normal = poly.normal
        if abs(normal.x) > .9:
            for li in poly.loop_indices:
                co = mesh.vertices[mesh.loops[li].vertex_index].co
                layer[li].uv = (co.y / (2 * radius) + .5,
                                co.z / (2 * radius) + .5)
        else:
            for li in poly.loop_indices:
                co = mesh.vertices[mesh.loops[li].vertex_index].co
                u = (math.atan2(co.z, co.y) / math.tau) % 1.0
                v = (co.x - x0) / max(1e-8, x1 - x0)
                layer[li].uv = (u, v)


def uv_planar_x(obj, y0, y1, z0, z1):
    layer = uv_data(obj)
    for poly in obj.data.polygons:
        for li in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[li].vertex_index].co
            if poly.normal.x > .5:
                u = (co.y - y0) / max(1e-8, y1 - y0)
                v = (co.z - z0) / max(1e-8, z1 - z0)
            else:
                # Border color at the texture edge.
                u, v = .012, .012
            layer[li].uv = (u, v)


def apply_uv(obj, mode):
    if mode == 'box':
        uv_box(obj)
    elif isinstance(mode, tuple) and mode[0] == 'cyl_z':
        uv_cyl_z(obj, mode[1], mode[2], mode[3])
    elif isinstance(mode, tuple) and mode[0] == 'cyl_x':
        uv_cyl_x(obj, mode[1], mode[2], mode[3], mode[4])
    elif isinstance(mode, tuple) and mode[0] == 'dial_x':
        uv_dial_x(obj, mode[1], mode[2], mode[3], mode[4], mode[5])
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
               bevel=.001, uv='box', register=True):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=z1 - z0, location=(x, y, (z0 + z1) / 2))
    obj = bpy.context.object
    return finish(obj, name, key, bevel=bevel, uv=uv, register=register)


def cylinder_x(name, radius, x0, x1, cy, cz, key, *, segs=96,
               bevel=.0008, uv='box', register=True):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=abs(x1 - x0),
                                        location=((x0 + x1) / 2, cy, cz),
                                        rotation=(0, math.pi / 2, 0))
    obj = bpy.context.object
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv, register=register)


def along(name, start, end, radius, key, *, segs=48, bevel=.0008, uv='box',
          register=True):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv, register=register)


def hex_axis(name, radius, thickness, center, axis, key, *, bevel=.0005,
             register=True):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius,
                                        depth=thickness, location=center)
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat('Z', 'Y').to_euler()
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv='box', smooth=8,
                  register=register)


def torus_axis(name, major, minor, location, axis, key, *, major_segments=96,
               minor_segments=10, bevel=0.0, uv='box', register=True):
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=location, major_radius=major,
                                     minor_radius=minor)
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat('Z', 'Y').to_euler()
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=30,
                  register=register)


def torus_z(name, major, minor, location, key, *, major_segments=128,
            minor_segments=10, bevel=0.0, uv='box', register=True):
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=location, major_radius=major,
                                     minor_radius=minor)
    obj = bpy.context.object
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=30,
                  register=register)


def box(name, center, size, key, *, rotation=(0, 0, 0), bevel=.001, uv='box',
        register=True, material=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    obj.rotation_euler = rotation
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv, register=register,
                  material=material)


def lathe_z(name, profile, key, *, segs=160, bevel=.0008,
            uv=('cyl_z', -0.38, 0.442, 0.12), smooth=30):
    verts = []
    rings = []
    for radius, z in profile:
        start = len(verts)
        for j in range(segs):
            t = math.tau * j / segs
            verts.append((radius * math.cos(t), radius * math.sin(t), z))
        rings.append(list(range(start, start + segs)))
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((a[j], a[j2], b[j2], b[j]))
    bottom = len(verts)
    verts.append((0.0, 0.0, profile[0][1]))
    top = len(verts)
    verts.append((0.0, 0.0, profile[-1][1]))
    first, last = rings[0], rings[-1]
    for j in range(segs):
        j2 = (j + 1) % segs
        faces.append((bottom, first[j2], first[j]))
        faces.append((top, last[j], last[j2]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=smooth)


def annulus_x(name, x0, x1, cy, cz, r_in, r_out, key, *, segs=128,
              bevel=.001, register=True):
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
                  uv=('cyl_x', x0, x1, cy, cz), smooth=34, register=register)


def annulus_axis(name, center, axis, length, r_in, r_out, key, *, segs=64,
                 bevel=.0005, register=True):
    d = Vector(axis).normalized()
    ref = Vector((0, 0, 1))
    if abs(d.dot(ref)) > .92:
        ref = Vector((0, 1, 0))
    side = d.cross(ref).normalized()
    up = side.cross(d).normalized()
    center = Vector(center)
    verts = []
    rings = []
    for t in (-length / 2, length / 2):
        for r in (r_in, r_out):
            start = len(verts)
            for j in range(segs):
                a = math.tau * j / segs
                p = center + d * t + side * (r * math.cos(a)) + up * (r * math.sin(a))
                verts.append(tuple(p))
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
    return finish(obj, name, key, bevel=bevel, uv='box', smooth=32,
                  register=register)


def sweep_rounded(name, points, rx, rz, key, *, sides=16, exponent=3.2,
                  bevel=.0008, uv='box', register=True, taper=1.0,
                  taper_start=.55):
    pts = [Vector(p) for p in points]
    verts = []
    rings = []
    tangents = []
    scales = []
    last = max(1, len(pts) - 1)
    for i, p in enumerate(pts):
        t = i / last
        scale = 1.0 + (taper - 1.0) * (t * t * (3 - 2 * t)
                                       if t > taper_start else 0.0)
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == len(pts) - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]
        tangent.normalize()
        tangents.append(tangent.copy())
        scales.append(scale)
        ref = Vector((0, 0, 1))
        if abs(tangent.dot(ref)) > .92:
            ref = Vector((0, 1, 0))
        side = tangent.cross(ref).normalized()
        up = side.cross(tangent).normalized()
        start = len(verts)
        for j in range(sides):
            a = math.tau * j / sides
            ca, sa = math.cos(a), math.sin(a)
            sc = math.copysign(abs(ca) ** (2.0 / exponent), ca)
            ss = math.copysign(abs(sa) ** (2.0 / exponent), sa)
            verts.append(tuple(p + side * (rx * scale * sc) +
                               up * (rz * scale * ss)))
        rings.append(list(range(start, start + sides)))
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append((a[j], a[j2], b[j2], b[j]))
    # Domed end caps: a flat n-gon reads as a cut-off stick at the grip tips.
    cap0 = len(verts)
    verts.append(tuple(pts[0] - tangents[0] * (rz * scales[0] * .55)))
    cap1 = len(verts)
    verts.append(tuple(pts[-1] + tangents[-1] * (rz * scales[-1] * .55)))
    first, last_ring = rings[0], rings[-1]
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append((cap0, first[j2], first[j]))
        faces.append((cap1, last_ring[j], last_ring[j2]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=35,
                  register=register)


def catmull_rom(points, per_seg=6):
    pts = [Vector(p) for p in points]
    padded = [pts[0]] + pts + [pts[-1]]
    sampled = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i:i + 4]
        for k in range(per_seg):
            t = k / per_seg
            sampled.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                                  (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                                  (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    sampled.append(pts[-1])
    return sampled


def tube_path(name, points, radius, key, *, sides=16, wobble=0.0, seed=34,
              uv='box', register=True):
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
            verts.append(tuple(p + side * (math.cos(t) * r) +
                               up * (math.sin(t) * r)))
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
    return finish(obj, name, key, bevel=0, uv=uv, smooth=40,
                  register=register)


def decal_shell(name, center_angle, half_angle, z0, z1, radius, thickness,
                material, *, segs=144, register=True):
    verts = []
    rings = []
    for r in (radius, radius + thickness):
        for z in (z0, z1):
            start = len(verts)
            for j in range(segs + 1):
                a = center_angle - half_angle + 2 * half_angle * j / segs
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
            angle = math.atan2(co.y, co.x)
            delta = math.atan2(math.sin(angle - center_angle),
                               math.cos(angle - center_angle))
            u = min(1.0, max(0.0, delta / (2 * half_angle) + .5))
            v = (co.z - z0) / max(1e-8, z1 - z0)
            layer[li].uv = (u, v)
    objects.append(obj)
    return obj


def disc_x(name, x0, x1, cy, cz, radius, key, *, segs=128, bevel=.0003,
           material=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=abs(x1 - x0),
                                        location=((x0 + x1) / 2, cy, cz),
                                        rotation=(0, math.pi / 2, 0))
    obj = bpy.context.object
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel,
                  uv=('dial_x', x0, x1, cy, cz, radius), smooth=30,
                  material=material)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def build_body():
    body = D['body']
    profile = [
        (0.007, -0.351), (0.037, -0.352), (0.065, -0.355), (0.087, -0.359),
        (0.105, -0.364), (0.119, -0.370), (0.130, -0.374), (0.1375, -0.374),
        (0.1438, -0.371), (0.1475, -0.366), (0.150, -0.358), (0.1506, -0.348),
        (0.150, -0.338), (0.150, body['z0']), (0.150, body['z1']),
        (0.1506, 0.286), (0.1506, 0.292), (0.149, 0.300), (0.145, 0.310),
        (0.139, 0.320), (0.131, 0.330), (0.121, 0.340), (0.110, 0.350),
        (0.099, 0.360), (0.088, 0.370), (0.078, 0.380), (0.068, 0.389),
        (0.058, 0.397), (0.050, 0.406), (0.0465, 0.420), (0.046, 0.432),
        (0.047, 0.440),
    ]
    lathe_z('body_shell', profile, 'paint_green_worn', segs=192,
            bevel=.0007, uv=('cyl_z', -0.380, 0.442, 0.150))
    torus_z('body_top_seam', 0.1503, .0038, (0, 0, 0.289),
            'paint_green_worn', major_segments=192, minor_segments=10)
    torus_z('body_bottom_rim', 0.1406, .0058, (0, 0, -0.367),
            'paint_green_worn', major_segments=192, minor_segments=12)
    torus_z('body_bottom_inner_ring', 0.0725, .0026, (0, 0, -0.357),
            'paint_green_worn', major_segments=128, minor_segments=8)
    torus_z('body_neck_roll', 0.0465, .0026, (0, 0, 0.430),
            'paint_green_worn', major_segments=96, minor_segments=8)
    # A subtle vertical manufacturing seam on the back-right quadrant.
    pts = []
    for k in range(25):
        z = -0.300 + 0.560 * k / 24
        a = math.radians(128)
        pts.append((0.1501 * math.cos(a), 0.1501 * math.sin(a), z))
    tube_path('body_vertical_seam', pts, .0011, 'paint_green_worn',
              sides=7, wobble=.18, seed=341)
    emb = D['label']['emboss']
    decal_shell('body_emboss_decal', emb['center_angle'], emb['half_angle'],
                emb['z0'], emb['z1'], emb['radius'], emb['thickness'],
                MATS['emboss'])


def build_neck_and_valve():
    v = D['valve']
    cylinder_z('neck_brass_collar', 0.041, 0.438, 0.449, 'brass',
               segs=128, bevel=.0007,
               uv=('cyl_z', 0.438, 0.449, 0.041))
    cylinder_z('valve_body', v['r'], v['z0'], v['z1'], 'paint_green_clean',
               segs=128, bevel=.0012,
               uv=('cyl_z', v['z0'], v['z1'], v['r']))
    cylinder_z('valve_top', v['top_r'], v['z1'], v['top_z1'],
               'paint_green_clean', segs=96, bevel=.001,
               uv=('cyl_z', v['z1'], v['top_z1'], v['top_r']))
    px, py, pz = v['pivot']
    cylinder_x('handle_pivot_boss', v['pivot_r'], px - v['pivot_len'] / 2,
               px + v['pivot_len'] / 2, py, pz, 'paint_green_clean',
               segs=96, bevel=.001)
    # Gauge neck and hose port.
    g = D['gauge']
    cylinder_x('gauge_boss', g['housing_r'], 0.010, 0.040, g['center'][1],
               g['center'][2], 'paint_green_clean', segs=96, bevel=.001)
    along('hose_port_boss', (0.020, -0.020, 0.512), (0.020, -0.058, 0.514),
          0.020, 'paint_green_clean', segs=48, bevel=.001)
    hex_axis('hose_port_nut', 0.023, .012, (0.020, -0.066, 0.514),
             (0, 1, 0), 'brass', bevel=.0005)
    cylinder_x('valve_screw_a', .0042, 0.034, 0.040, 0.012, 0.470,
               'dark_steel', segs=20, bevel=.0003)
    cylinder_x('valve_screw_b', .0042, 0.034, 0.040, 0.012, 0.492,
               'dark_steel', segs=20, bevel=.0003)


def build_gauge():
    g = D['gauge']
    cx, cy, cz = g['center']
    front_x = g['front_x']
    cylinder_x('gauge_housing', g['housing_r'], 0.040, 0.058, cy, cz,
               'paint_green_clean', segs=128, bevel=.0012)
    cylinder_x('gauge_brass_ring', 0.0415, 0.056, 0.061, cy, cz, 'brass',
               segs=128, bevel=.0006)
    annulus_x('gauge_bezel', 0.060, front_x, cy, cz, g['bezel_in'],
              g['bezel_r'], 'chrome', segs=160, bevel=.0010)
    disc_x('gauge_dial', 0.0615, 0.0632, cy, cz, g['dial_r'], 'gauge_face',
           segs=128, bevel=.00025)
    along('gauge_needle', (0.0645, cy, cz), (0.0645, cy - .012, cz + .022),
          .0013, 'dark_steel', segs=12, bevel=.0002)
    along('gauge_needle_tail', (0.0645, cy, cz), (0.0645, cy + .007, cz - .009),
          .0011, 'dark_steel', segs=12, bevel=.0002)
    cylinder_x('gauge_hub', .0062, 0.0635, 0.0678, cy, cz, 'chrome',
               segs=48, bevel=.0005)
    cylinder_x('gauge_hub_dark', .0026, 0.0678, 0.0688, cy, cz, 'dark_steel',
               segs=24, bevel=.0002)
    disc_x('gauge_glass', 0.0680, 0.0696, cy, cz, 0.0367, 'glass',
           segs=128, bevel=.0002)


def build_handles():
    h = D['handles']
    sweep_rounded('handle_lower', h['lower'], h['lower_rx'], h['lower_rz'],
                  'paint_green_clean', sides=20, exponent=3.4, bevel=.0011,
                  taper=1.22, taper_start=.62)
    sweep_rounded('lever_upper', h['upper'], h['upper_rx'], h['upper_rz'],
                  'paint_green_clean', sides=20, exponent=3.4, bevel=.0011,
                  taper=1.32, taper_start=.58)
    px, py, pz = D['valve']['pivot']
    cylinder_x('handle_pivot_pin', .0068, px - .046, px + .046, py, pz,
               'chrome', segs=32, bevel=.0004)
    cylinder_x('handle_pivot_washer_a', .0115, px - .052, px - .045, py, pz,
               'dark_steel', segs=32, bevel=.0004)
    cylinder_x('handle_pivot_washer_b', .0115, px + .045, px + .052, py, pz,
               'dark_steel', segs=32, bevel=.0004)


def build_safety():
    s = D['safety']
    torus_axis('safety_ring', s['ring_major'], s['ring_minor'],
               s['ring_center'], (1, 0, 0), 'lime_plastic',
               major_segments=64, minor_segments=10)
    torus_axis('safety_tamper_strap', s['strap_major'], s['strap_minor'],
               s['strap_center'], (1, 0, 0), 'yellow_plastic',
               major_segments=64, minor_segments=10)
    tag = box('safety_tag', s['tag_center'], s['tag_size'], 'yellow_tag',
              rotation=(s['tag_rot'], 0, 0), bevel=.0012)
    lo, hi, _, _ = bounds([tag])
    uv_planar_x(tag, lo.y, hi.y, lo.z, hi.z)
    pts = catmull_rom([(0.030, 0.059, 0.558), (0.022, 0.064, 0.545),
                       (0.016, 0.072, 0.530), (0.010, 0.078, 0.514)], 4)
    tube_path('safety_tag_wire', pts, .0012, 'dark_steel', sides=6, seed=343)


def build_hose():
    h = D['hose']
    path = catmull_rom(h['path'], 7)
    tube_path('hose', path, h['radius'], 'rubber', sides=18, wobble=0.0,
              seed=344)
    along('hose_coupling', (0.020, -0.058, 0.516), (0.020, -0.112, 0.516),
          h['coupling_r'], 'chrome', segs=64, bevel=.0008)
    torus_axis('hose_coupling_ferrule', .0178, .0032, (0.020, -0.104, 0.516),
               (0, -1, 0), 'chrome', major_segments=64, minor_segments=8)
    torus_axis('hose_top_band', .0142, .0028, (0.020, -0.126, 0.514),
               (0, -1, -0.15), 'chrome', major_segments=48, minor_segments=8)
    # Lower hose clip and bracket.
    clip = (0.055, -0.184, -0.155)
    torus_axis('hose_clip_ring', .0152, .0046, clip, (0, 0, 1),
               'rubber', major_segments=48, minor_segments=10)
    box('hose_clip_bracket', (0.050, -0.153, -0.155),
        (.012, .060, .020), 'rubber', rotation=(0, 0, 0), bevel=.0025)
    box('hose_clip_buckle', (0.050, -0.125, -0.155),
        (.016, .008, .026), 'chrome', bevel=.0012)
    # Nozzle sleeve, band, body, outlet ring and blind dark plug.
    cuff = (0.042, -0.180, -0.244)
    along('nozzle_sleeve', cuff, (0.040, -0.180, -0.264), .0155, 'rubber',
          segs=40, bevel=.0008)
    torus_axis('nozzle_band', .0160, .0030, (0.041, -0.180, -0.256),
               (0, 0, 1), 'chrome', major_segments=48, minor_segments=8)
    box('nozzle_body', (0.040, -0.180, -0.276), (.052, .052, .064),
        'rubber', bevel=.006)
    annulus_axis('nozzle_outlet_ring', (0.040, -0.180, -0.316),
                 (0, 0, 1), .024, .010, .017, 'rubber', segs=64,
                 bevel=.0012)
    cylinder_z('nozzle_dark_plug', .0120, -0.318, -0.306, 'dark_steel',
               x=0.040, y=-0.180, segs=48, bevel=.0005)
    box('nozzle_retention_lug', (0.014, -0.180, -0.256),
        (.016, .014, .030), 'rubber', bevel=.0022)
    box('nozzle_retention_spring', (0.014, -0.180, -0.272),
        (.008, .026, .004), 'chrome', bevel=.0008)


def build_labels():
    l = D['label']
    for key, name in (('front', 'label_front_decal'),
                      ('back', 'label_back_decal')):
        spec = l[key]
        decal_shell(name, spec['center_angle'], spec['half_angle'],
                    spec['z0'], spec['z1'], spec['radius'],
                    spec['thickness'], MATS[f'label_{key}'])


def build_all():
    build_body()
    build_neck_and_valve()
    build_gauge()
    build_handles()
    build_safety()
    build_hose()
    build_labels()


# ---------------------------------------------------------------------------
# Scene, checks, renders, export
# ---------------------------------------------------------------------------
def init_camera(scene):
    data = bpy.data.cameras.new('review camera')
    cam = bpy.data.objects.new('review camera', data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_camera(cam, eye, target, *, ortho_scale=None, fov=40, roll_deg=0.0,
               shift_x=0.0, shift_y=0.0):
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
    cam.data.shift_x = shift_x
    cam.data.shift_y = shift_y


def look(direction, center, span, *, ortho_scale=None, roll_deg=0.0):
    direction = Vector(direction).normalized()
    set_camera(CAM, Vector(center) + direction * span * 5, center,
               ortho_scale=ortho_scale or span * 1.27, roll_deg=roll_deg)


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


def object_center(name):
    lo, hi, center, span = bounds([bpy.data.objects[name]])
    return center


def draw(scene, path, resolution, samples, *, transparent=True):
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
                                               Vector(direction).normalized(),
                                               distance)
    return hit[0] is not None


def ray_clear(name, origin, direction, distance):
    return not ray_hit(name, origin, direction, distance)


def ray_clear_scene(origin, direction, distance):
    for obj in objects:
        if ray_hit(obj.name, origin, direction, distance):
            return False, obj.name
    return True, None


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
    checks = {'handle': {}, 'safety': {}, 'hose': {}, 'gauge': {},
              'objects': {}}
    gap_y, gap_z = 0.200, 0.565
    clear, blocker = ray_clear_scene((-0.30, gap_y, gap_z), (1, 0, 0), .60)
    checks['handle'] = {
        'gap_clear': bool(clear),
        'gap_blocker': blocker,
        'lower_solid': ray_hit('handle_lower', (-0.30, 0.200, 0.498),
                               (1, 0, 0), .60),
        'upper_solid': ray_hit('lever_upper', (-0.30, 0.200, 0.633),
                               (1, 0, 0), .60),
        'pivot_solid': ray_hit('handle_pivot_pin', (-0.30, 0.018, 0.545),
                               (1, 0, 0), .60),
    }
    ring = D['safety']['ring_center']
    checks['safety'] = {
        'hole_clear': ray_clear('safety_ring', (-0.30, ring[1], ring[2]),
                                (1, 0, 0), .60),
        'ring_solid': ray_hit('safety_ring', (-0.30, ring[1] +
                                              D['safety']['ring_major'],
                                              ring[2]), (1, 0, 0), .60),
        'tamper_present': ray_hit('safety_tamper_strap', (-0.30, 0.044,
                                                          0.545),
                                  (1, 0, 0), .60),
        'tag_present': 'safety_tag' in bpy.data.objects,
    }
    checks['hose'] = {
        'hose_solid': ray_hit('hose', (-0.30, -0.198, 0.095),
                              (1, 0, 0), .60),
        'coupling_solid': ray_hit('hose_coupling', (-0.30, -0.085, 0.516),
                                  (1, 0, 0), .60),
        'nozzle_body_solid': ray_hit('nozzle_body', (-0.30, -0.180, -0.282),
                                     (1, 0, 0), .60),
        'outlet_ring_solid': ray_hit('nozzle_outlet_ring', (-0.30, -0.166,
                                                            -0.309),
                                     (1, 0, 0), .60),
        'dark_plug_blind': ray_hit('nozzle_dark_plug', (0.040, -0.180, -0.45),
                                   (0, 0, 1), .20),
        'plug_ray_clear_outlet': not ray_hit('nozzle_outlet_ring',
                                             (0.040, -0.180, -0.45),
                                             (0, 0, 1), .20),
    }
    g = D['gauge']['center']
    checks['gauge'] = {
        'glass_hit': ray_hit('gauge_glass', (0.30, g[1], g[2]),
                             (-1, 0, 0), .40),
        'bezel_hit': ray_hit('gauge_bezel', (0.30, g[1] + .041, g[2]),
                             (-1, 0, 0), .40),
        'dial_present': 'gauge_dial' in bpy.data.objects,
        'needle_present': 'gauge_needle' in bpy.data.objects,
    }
    checks['objects'] = {
        'front_label': 'label_front_decal' in bpy.data.objects,
        'back_label': 'label_back_decal' in bpy.data.objects,
        'emboss_label': 'body_emboss_decal' in bpy.data.objects,
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
    background.inputs['Strength'].default_value = .62
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 110
        fill.data.shape = 'RECTANGLE'
        fill.data.size = fill.data.size_y = 3.2
        fill.location = (-3.0, 2.4, 2.4)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 210, 3.0),
            ('front reflection', (3.0, 1.4, 1.0), 105, 1.5),
            ('top fill', (0.0, 0.0, 4.0), 80, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 72, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 34, 1.4)):
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
        scale = span * 1.13 if name in ('front', 'back', 'left', 'right') else .32
        look(direction, center, span, ortho_scale=scale)
        draw(scene, render_dir / f'new-{name}.png', RR['six_resolution'],
             RR['six_samples'])

    # Photo-like three-quarter camera, and the library's calibrated camera.
    source_dir = Vector((1.0, -.62, .20)).normalized()
    set_camera(CAM, center + source_dir * 1.90, center, fov=37.5, roll_deg=0)
    draw(scene, render_dir / 'new-source-3q.png', 512, RR['six_samples'])
    set_camera(CAM, center + Vector((1.9, 0, 0)), center, fov=38.102696,
               shift_x=-0.002019, shift_y=-0.002529)
    draw(scene, render_dir / 'new-source.png', 512, RR['six_samples'])
    draw(scene, render_dir / 'new-source-alpha.png', 512, RR['six_samples'],
         transparent=True)

    handle_center = (object_center('handle_lower') +
                     object_center('lever_upper')) / 2
    label_front_center = object_center('label_front_decal')
    label_back_center = object_center('label_back_decal')
    label_front_center.x = .120
    label_back_center.x = -.120
    targets = {
        'gauge': object_center('gauge_glass'),
        'handles': handle_center,
        'hose-top': object_center('hose_coupling'),
        'nozzle': object_center('nozzle_body'),
        'label-front': label_front_center,
        'label-back': label_back_center,
        'neck': object_center('neck_brass_collar'),
        'bottom': object_center('body_bottom_rim'),
        'overall': center,
    }
    detail_specs = [
        ('gauge', (1.0, -.45, .10), .21),
        ('handles', (1.0, -.25, .10), .40),
        ('hose-top', (1.0, -.65, .20), .18),
        ('nozzle', (1.0, -.50, -.10), .25),
        ('label-front', (1.0, -.03, .02), .36),
        ('label-back', (-1.0, .03, .02), .36),
        ('neck', (1.0, -.35, .30), .22),
        ('bottom', (1.0, -.35, -.35), .30),
        ('overall', (1.0, -.62, .22), span * 1.15),
    ]
    for name, direction, scale in detail_specs:
        look(direction, targets[name], span, ortho_scale=scale)
        draw(scene, render_dir / f'detail-{name}.png', RR['detail_resolution'],
             RR['detail_samples'])

    set_visible({'gauge_housing', 'gauge_brass_ring', 'gauge_bezel',
                 'gauge_dial', 'gauge_glass', 'gauge_needle',
                 'gauge_needle_tail', 'gauge_hub', 'gauge_hub_dark'})
    look((1.0, -.45, .10), targets['gauge'], span, ortho_scale=.21)
    draw(scene, render_dir / 'isolate-gauge.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({'handle_lower', 'lever_upper', 'handle_pivot_boss',
                 'handle_pivot_pin', 'handle_pivot_washer_a',
                 'handle_pivot_washer_b', 'safety_ring',
                 'safety_tamper_strap', 'safety_tag', 'safety_tag_wire'})
    look((1.0, -.30, .10), targets['handles'], span, ortho_scale=.40)
    draw(scene, render_dir / 'isolate-handles.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({obj.name for obj in objects if obj.name.startswith(
        ('hose', 'nozzle'))})
    look((1.0, -.55, -.05), targets['nozzle'], span, ortho_scale=.42)
    draw(scene, render_dir / 'isolate-hose.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({'label_front_decal', 'label_back_decal', 'body_emboss_decal'})
    look((1.0, -.45, .10), center, span, ortho_scale=span * 1.1)
    draw(scene, render_dir / 'isolate-labels.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible(set(metal_objects))
    look((1.0, -.7, .28), center, span, ortho_scale=span * 1.15)
    draw(scene, render_dir / 'isolate-metal.png', RR['detail_resolution'],
         RR['detail_samples'])

    set_visible({obj.name for obj in objects})
    look((1.0, -.62, .28), center, span, ortho_scale=span * 1.15)
    draw(scene, render_dir / 'new-material-3q.png', RR['detail_resolution'],
         RR['detail_samples'])
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
               fov=38.102696, shift_x=-0.002019, shift_y=-0.002529)
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
        'gauge': p(.40), 'handles': p(.45), 'hose-top': p(.34),
        'nozzle': p(-.42) + side * (old_span * .12), 'label-front': old_pc,
        'label-back': old_pc, 'neck': p(.32), 'bottom': p(-.48),
        'overall': old_pc,
    }
    old_directions = {
        'gauge': (side + major * .10 + Vector((0, 0, .06))),
        'handles': (side + major * .18 + Vector((0, 0, .08))),
        'hose-top': (side + Vector((0, 0, .10))),
        'nozzle': (side - major * .25 + Vector((0, 0, -.05))),
        'label-front': (side + major * .02),
        'label-back': (-side + major * .02),
        'neck': (side + Vector((0, 0, .12))),
        'bottom': (side - major * .40 + Vector((0, 0, -.06))),
        'overall': Vector((1.0, -.62, .25)),
    }
    old_scales = {
        'gauge': .22, 'handles': .45, 'hose-top': .28, 'nozzle': .35,
        'label-front': .52, 'label-back': .52, 'neck': .28,
        'bottom': .32, 'overall': 1.15,
    }
    for name in old_targets:
        look(tuple(old_directions[name]), old_targets[name], old_span,
             ortho_scale=old_span * old_scales[name])
        draw(scene, render_dir / f'old-detail-{name}.png',
             max(320, RR['detail_resolution'] // 2), RR['detail_samples'] // 2)
    look((1.0, -.62, .28), old_center, old_span, ortho_scale=old_span * 1.15)
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
        'coordinate_system': 'Z up; cylinder axis Z; front camera +X; hose -Y; '
                             'handles +Y; gauge faces +X',
        'status': 'manual_only_pending_review',
        'reference_role': 'visual_match_unverified',
    }
    (out / 'worker-result.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'ok': True, 'parts': len(objects),
                      'glb': str(out / 'mesh.glb')}, ensure_ascii=False))


main()
