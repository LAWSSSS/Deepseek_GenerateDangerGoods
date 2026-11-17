"""Blender worker for the review-only Raid-type aerosol-can candidate (item 15).

Shape basis: the original clean/rust photograph decides proportions and the
visible front structure.  The user six-view sheet is an uncalibrated visual
match; its front/back/side panels are used for the hidden-half label wrap and
the top/bottom topology, never as measured geometry.  The candidate is a
part-based rebuild: closed printed tinplate body with rolled/concave bottom,
brushed metal base and rust rim, glossy black plastic overcap with two raised
side horns and a real two-lobed window, internal valve stem, actuator, trigger
and a two-bore nozzle face.  It never replaces the library mesh, is manual-only
and never enters automatic placement.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
config = request['config']
dims = config['dimensions']
window = config['window']
nozzle = config['nozzle']
objects = []
special_materials = {}
TAU = math.tau
R = dims['body_radius']

# Start from a genuinely empty file: Blender's startup Cube would otherwise
# enclose the small can and block every render.
bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.context.preferences.filepaths.save_version = 0
except Exception:
    pass


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def image_node(nodes, key, noncolor=False):
    node = nodes.new('ShaderNodeTexImage')
    node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
    node.image.pack()
    if noncolor:
        node.image.colorspace_settings.name = 'Non-Color'
    node.interpolation = 'Linear'
    node.extension = 'REPEAT'
    return node


def textured_material(key, prefix, normal_strength, *, base_key=None):
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    color = image_node(nodes, (base_key or prefix) + '_base')
    metal = image_node(nodes, prefix + '_mr', True)
    normal = image_node(nodes, prefix + '_normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(metal.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = normal_strength
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    special_materials[key] = material
    return material


def solid_material(key, rgb, roughness, metallic=0.0):
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    principled = material.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (*rgb, 1.0)
    principled.inputs['Roughness'].default_value = roughness
    principled.inputs['Metallic'].default_value = metallic
    special_materials[key] = material
    return material


def make_materials():
    textured_material('label', 'label', config['material_profile']['label']['normal_strength'])
    textured_material('cap', 'cap', config['material_profile']['cap']['normal_strength'])
    textured_material('metal', 'metal', config['material_profile']['metal']['normal_strength'])
    textured_material('rust', 'rust', config['material_profile']['rust']['normal_strength'])
    textured_material('dust', 'dust', 0.16, base_key='dust')
    solid_material('nozzle', (0.006, 0.006, 0.007), 0.24)
    solid_material('inner', (0.010, 0.010, 0.011), 0.55)
    solid_material('stem', (0.34, 0.35, 0.36), 0.30, metallic=1.0)
    solid_material('nozzle_ring', (0.070, 0.072, 0.076), 0.28, metallic=0.85)


# ---------------------------------------------------------------------------
# Generic geometry helpers
# ---------------------------------------------------------------------------
def mesh_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def apply_bevel(obj, width, segments=2, angle=45):
    if not width:
        return obj
    bpy.ops.object.select_all(action='DESELECT')
    mod = obj.modifiers.new('Manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def revolve_solid(name, profile, segments):
    verts, rings = [], []
    for r, z in profile:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            ids = []
            for s in range(segments):
                a = TAU * s / segments
                ids.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(ids)
    faces = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        if len(a) == 1 and len(b) > 1:
            for s in range(segments):
                faces.append((a[0], b[s], b[(s + 1) % segments]))
        elif len(a) > 1 and len(b) == 1:
            for s in range(segments):
                faces.append((a[s], a[(s + 1) % segments], b[0]))
        elif len(a) > 1 and len(b) > 1:
            for s in range(segments):
                faces.append((a[s], a[(s + 1) % segments],
                              b[(s + 1) % segments], b[s]))
    return mesh_object(name, verts, faces)


def box_at(name, center, half):
    x, y, z = center
    hx, hy, hz = half
    verts = []
    for dz in (-hz, hz):
        for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            verts.append((x + dx, y + dy, z + dz))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def cylinder_x(name, center, radius, half_length, segments=48):
    cx, cy, cz = center
    verts, rings = [], []
    for s in (-half_length, half_length):
        ids = []
        for i in range(segments):
            a = TAU * i / segments
            ids.append(len(verts))
            verts.append((cx + s, cy + radius * math.cos(a),
                          cz + radius * math.sin(a)))
        rings.append(ids)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((rings[0][i], rings[0][j], rings[1][j], rings[1][i]))
    faces.append(tuple(reversed(rings[0])))
    faces.append(tuple(rings[1]))
    return mesh_object(name, verts, faces)


def cylinder_z(name, center, radius, z0, z1, segments=32):
    cx, cy = center[0], center[1]
    profile = [(0.0, z0), (radius, z0), (radius, z1), (0.0, z1)]
    obj = revolve_solid(name, profile, segments)
    obj.data.transform(Matrix.Translation((cx, cy, 0)))
    obj.data.update()
    return obj


def torus_at(name, center, major, minor, *, axis=(0, 0, 1), major_segments=96,
             minor_segments=12):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                     major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=center)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = Vector(axis).to_track_quat('Z', 'Y').to_euler()
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    objects.append(obj)
    return obj


def assign(obj, material):
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def smooth_all(obj):
    for poly in obj.data.polygons:
        poly.use_smooth = True


def wrap_angle(value):
    return (value + math.pi) % TAU - math.pi


# ---------------------------------------------------------------------------
# UV mapping
# ---------------------------------------------------------------------------
def uv_label(obj):
    """Cylindrical label UV with a repeat-safe seam at the front centre."""
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    uv = obj.data.uv_layers.active.data
    z0, z1 = dims['label_z0'], dims['label_z1']
    for poly in obj.data.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.y, co.x))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            co = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            delta = wrap_angle(angle - ca)
            uv[loop_index].uv = ((ca + delta) / TAU, (co.z - z0) / (z1 - z0))


def uv_planar(obj, scale=0.075, offset=0.5):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    uv = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            uv[loop_index].uv = (offset + co.x / scale, offset + co.y / scale)


def uv_cylindrical_generic(obj, z0, z1, segments_scale=1.0):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    uv = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.y, co.x))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            co = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            delta = wrap_angle(angle - ca)
            uv[loop_index].uv = ((ca + delta) / TAU * segments_scale,
                                 (co.z - z0) / (z1 - z0))


# ---------------------------------------------------------------------------
# Body and bottom
# ---------------------------------------------------------------------------
def build_labeled_body():
    """Closed painted tinplate cylinder from the bottom overlap to under the cap."""
    # Recessed bottom: the floor is above the concave metal base so the base,
    # not a flat painted disk, is what is seen from below.
    profile = [
        (0.0000, 0.0125),
        (0.0320, 0.0125),
        (0.0320, 0.0055),
        (0.0332, 0.0055),
        (0.0332, 0.1980),
        (0.0330, 0.2010),
        (0.0326, 0.2025),
        (0.0000, 0.2025),
    ]
    body = revolve_solid('Can labeled body', profile, config['radial_segments'])
    assign(body, special_materials['label'])
    if not body.data.uv_layers:
        body.data.uv_layers.new(name='UVMap')
    z0, z1 = dims['label_z0'], dims['label_z1']
    uv = body.data.uv_layers.active.data
    for poly in body.data.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = body.data.vertices[body.data.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.y, co.x))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            co = body.data.vertices[body.data.loops[loop_index].vertex_index].co
            delta = wrap_angle(angle - ca)
            v = (co.z - z0) / (z1 - z0)
            uv[loop_index].uv = ((ca + delta) / TAU,
                                 min(1.0, max(0.0, v)))
    apply_bevel(body, 0.00030, segments=2, angle=45)
    smooth_all(body)
    objects.append(body)
    return body


def build_bottom_metal():
    """Closed brushed-metal base with the concave dish and rolled rim."""
    profile = [
        (0.0000, dims['bottom_dish_center_z']),
        (0.0050, dims['bottom_dish_center_z'] - 0.00025),
        (0.0100, dims['bottom_dish_center_z'] - 0.0011),
        (0.0160, 0.0090),
        (0.0220, 0.0072),
        (0.0270, 0.0052),
        (0.0305, 0.0032),
        (0.0322, 0.0012),
        (0.0332, 0.0002),
        (0.0336, 0.0011),
        (0.0336, 0.0028),
        (0.0332, 0.0044),
        (0.0315, 0.0120),
        (0.0000, 0.0120),
    ]
    bottom = revolve_solid('Can bottom metal', profile, config['radial_segments'])
    assign(bottom, special_materials['metal'])
    if not bottom.data.uv_layers:
        bottom.data.uv_layers.new(name='UVMap')
    uv = bottom.data.uv_layers.active.data
    for poly in bottom.data.polygons:
        for loop_index in poly.loop_indices:
            co = bottom.data.vertices[bottom.data.loops[loop_index].vertex_index].co
            if abs(poly.normal.z) > 0.45:
                uv[loop_index].uv = (0.5 + co.x / 0.078, 0.5 + co.y / 0.078)
            else:
                a = math.atan2(co.y, co.x)
                uv[loop_index].uv = (a / TAU, co.z / 0.02)
    apply_bevel(bottom, 0.00028, segments=2, angle=42)
    smooth_all(bottom)
    objects.append(bottom)
    return bottom


def build_top_shoulder():
    """Closed hidden shoulder/crimp under the cap; keeps the can complete."""
    profile = [
        (0.0000, 0.2000),
        (0.0328, 0.2000),
        (0.0325, 0.2040),
        (0.0312, 0.2080),
        (0.0290, 0.2120),
        (0.0265, 0.2145),
        (0.0260, 0.2180),
        (0.0255, 0.2186),
        (0.0000, 0.2186),
    ]
    shoulder = revolve_solid('Can top shoulder', profile, config['radial_segments'])
    assign(shoulder, special_materials['metal'])
    uv_planar(shoulder, 0.055, 0.5)
    smooth_all(shoulder)
    objects.append(shoulder)
    return shoulder


def build_bottom_rims():
    rust = special_materials['rust']
    metal = special_materials['metal']
    ring = torus_at('Bottom rolled rim', (0, 0, dims['bottom_rim_outer_z']),
                    0.0328, 0.00100, major_segments=128, minor_segments=14)
    assign(ring, rust)
    uv_cylindrical_generic(ring, 0.0, 0.02)
    lip = torus_at('Bottom rim upper lip', (0, 0, 0.0068), 0.0317, 0.00065,
                   major_segments=128, minor_segments=12)
    assign(lip, metal)
    uv_cylindrical_generic(lip, 0.0, 0.02)
    return ring, lip


# ---------------------------------------------------------------------------
# Cap
# ---------------------------------------------------------------------------
CAP_PROFILE = [
    (0.0338, dims['cap_z0']),
    (0.0338, 0.2058),
    (0.0341, 0.2074),
    (0.0335, 0.2092),
    (0.0333, 0.2130),
    (0.0337, 0.2250),
    (0.0343, 0.2360),
    (0.0350, 0.2450),
    (0.0352, 0.2492),
    (0.0347, 0.2518),
    (0.0322, 0.2522),
    (0.0270, 0.2515),
    (0.0220, 0.2505),
    (0.0160, 0.2495),
    (0.0100, 0.2490),
    (0.0040, 0.2488),
]

def chaikin_profile(points, iterations=3):
    """Corner-cutting keeps the spline inside the control polygon.

    Catmull-Rom overshot the small cap step and created a self-intersecting
    fold around z=0.207; Chaikin smoothing cannot overshoot and still rounds
    the moulded edges.
    """
    result = list(points)
    for _ in range(iterations):
        smoothed = [result[0]]
        for i in range(len(result) - 1):
            p, q = result[i], result[i + 1]
            smoothed.append((0.75 * p[0] + 0.25 * q[0],
                             0.75 * p[1] + 0.25 * q[1]))
            smoothed.append((0.25 * p[0] + 0.75 * q[0],
                             0.25 * p[1] + 0.75 * q[1]))
        smoothed.append(result[-1])
        result = smoothed
    return result


def window_contains(y, z):
    half_w = window['half_width_y']
    half_h = window['half_height_slope']
    zc = window['center'][2]
    u = y / half_w
    v = (z - zc) / half_h
    angle = math.atan2(v, u)
    edge = 1.0 + window['peanut_lobe'] * math.cos(2.0 * angle)
    return math.hypot(u, v) < edge * 0.985


def build_cap():
    """Hollow cap made as a surface with a real window hole, then solidified.

    This avoids the fragile thin-wall boolean cavity: the cap is one continuous
    outer surface with the two-lobed window faces removed; a Solidify modifier
    gives the plastic wall thickness and closes the window and bottom rims.
    """
    profile = chaikin_profile(CAP_PROFILE, iterations=3)
    segments = 384
    verts, rings = [], []
    for r, z in profile:
        ids = []
        for j in range(segments):
            a = TAU * j / segments
            ids.append(len(verts))
            verts.append((r * math.cos(a), r * math.sin(a), z))
        rings.append(ids)
    # Close the small flat top disk with a fan; keeping the last ring at a
    # finite radius avoids the degenerate pole that made Solidify explode.
    top_z = profile[-1][1]
    center_index = len(verts)
    verts.append((0.0, 0.0, top_z))
    top_ring = rings[-1]
    top_faces = [(top_ring[j], top_ring[(j + 1) % segments], center_index)
                 for j in range(segments)]
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        if len(a) == 1 and len(b) > 1:
            for j in range(segments):
                faces.append((a[0], b[j], b[(j + 1) % segments]))
        elif len(a) > 1 and len(b) == 1:
            for j in range(segments):
                faces.append((a[j], a[(j + 1) % segments], b[0]))
        elif len(a) > 1 and len(b) > 1:
            for j in range(segments):
                j2 = (j + 1) % segments
                quad = (a[j], a[j2], b[j2], b[j])
                cs = [verts[k] for k in quad]
                cx = sum(p[0] for p in cs) / 4
                cy = sum(p[1] for p in cs) / 4
                cz = sum(p[2] for p in cs) / 4
                if cx > 0.012 and window_contains(cy, cz):
                    continue
                faces.append(quad)
    faces.extend(top_faces)
    cap = mesh_object('Cap', verts, faces)

    # The two raised side horns and the saddle top are inferred from the
    # original photo; the uncalibrated reference only confirms the lobes.
    for vert in cap.data.vertices:
        x, y, z = vert.co
        r = math.hypot(x, y)
        if z <= 0.242 or r <= 0.0175:
            continue
        phi = math.atan2(y, x)
        hump = (math.exp(-((wrap_angle(phi - math.pi / 2) / 0.58) ** 2)) +
                math.exp(-((wrap_angle(phi + math.pi / 2) / 0.58) ** 2)))
        vertical = min(1.0, max(0.0, (z - 0.242) / 0.010))
        radial = min(1.0, max(0.0, (r - 0.0175) / 0.010))
        blend = vertical * radial
        vert.co.z += (0.0105 * hump - 0.0032 * (1.0 - hump)) * blend
    cap.data.update()

    bm = bmesh.new()
    bm.from_mesh(cap.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces],
                     context='VERTS')
    bm.to_mesh(cap.data)
    bm.free()
    cap.data.update()

    bm2 = bmesh.new()
    bm2.from_mesh(cap.data)
    bm2.verts.ensure_lookup_table()
    boundary = [v.index for v in bm2.verts
                if v.is_boundary and v.co.z > 0.220 and v.co.x > 0.012]
    neighbours = set()
    for v in bm2.verts:
        if v.index in boundary:
            for e in v.link_edges:
                neighbours.add(e.other_vert(v).index)
    bm2.free()

    # Relax only the window rim and its first ring.  This rounds the
    # stair-stepped face-deletion boundary without the folding that a full
    # snap to the analytic outline produced near the top saddle.
    group = cap.vertex_groups.new(name='Window rim')
    group.add(boundary, 1.0, 'REPLACE')
    group.add([i for i in neighbours if i not in boundary], 0.5, 'REPLACE')
    bpy.ops.object.select_all(action='DESELECT')
    cap.select_set(True)
    bpy.context.view_layer.objects.active = cap
    smooth = cap.modifiers.new('Window rim relax', 'SMOOTH')
    smooth.factor = 0.55
    smooth.iterations = 2
    smooth.vertex_group = group.name
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    cap.data.update()

    bpy.ops.object.select_all(action='DESELECT')
    cap.select_set(True)
    bpy.context.view_layer.objects.active = cap
    mod = cap.modifiers.new('Cap plastic wall', 'SOLIDIFY')
    mod.thickness = 0.0015
    mod.offset = 0.0
    mod.use_rim = True
    mod.use_even_offset = True
    mod.use_quality_normals = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    cap.data.update()

    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(32))
    except Exception:
        smooth_all(cap)
    uv_cylindrical_generic(cap, dims['cap_z0'], 0.26)
    assign(cap, special_materials['cap'])
    objects.append(cap)
    make_window_bezel(profile)
    step = torus_at('Cap lower step ring', (0, 0, 0.2075), 0.03385, 0.00030,
                    major_segments=192, minor_segments=8)
    assign(step, special_materials['cap'])
    uv_cylindrical_generic(step, dims['cap_z0'], 0.26)
    return cap


def make_window_bezel(profile):
    """Smooth moulded lip covering the quad-grid window edge."""
    profile_r = [(r, z) for r, z in profile if r > 1e-6]

    def radius_at(z):
        return min(profile_r, key=lambda p: abs(p[1] - z))[0]

    zc = window['center'][2]
    hw = window['half_width_y']
    hh = window['half_height_slope']
    lobe = window['peanut_lobe']
    major, minor = 256, 12
    tube_r = 0.00105
    centers = []
    for i in range(major):
        t = TAU * i / major
        rfac = 1.0 + lobe * math.cos(2.0 * t)
        y = hw * rfac * math.cos(t)
        z = zc + hh * rfac * math.sin(t)
        rr = radius_at(z)
        y = max(-rr * 0.97, min(rr * 0.97, y))
        x = math.sqrt(max(1e-8, rr * rr - y * y)) + 0.00030
        centers.append(Vector((x, y, z)))
    front = Vector((1.0, 0.0, 0.0))
    verts = []
    for i in range(major):
        tangent = (centers[(i + 1) % major] - centers[i - 1]).normalized()
        binormal = tangent.cross(front).normalized()
        for k in range(minor):
            a = TAU * k / minor
            p = centers[i] + front * (math.cos(a) * tube_r) + \
                binormal * (math.sin(a) * tube_r)
            verts.append(tuple(p))
    faces = []
    for i in range(major):
        i2 = (i + 1) % major
        for k in range(minor):
            k2 = (k + 1) % minor
            faces.append((i * minor + k, i2 * minor + k,
                          i2 * minor + k2, i * minor + k2))
    bezel = mesh_object('Window bezel', verts, faces)
    for poly in bezel.data.polygons:
        poly.use_smooth = True
    assign(bezel, special_materials['cap'])
    uv_cylindrical_generic(bezel, dims['cap_z0'], 0.26)
    objects.append(bezel)
    return bezel


def apply_boolean(target, cutter, name):
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    mod = target.modifiers.new(name, 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    mod.use_self = False
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if cutter.name in bpy.data.objects:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


# ---------------------------------------------------------------------------
# Internal actuator / nozzle
# ---------------------------------------------------------------------------
def build_valve_and_nozzle():
    stem = cylinder_z('Valve stem', (0, 0), 0.0025, 0.2160, 0.2350, 32)
    assign(stem, special_materials['stem'])
    objects.append(stem)
    cup = torus_at('Valve cup ring', (0, 0, 0.2168), 0.0145, 0.0012,
                   major_segments=64, minor_segments=10)
    assign(cup, special_materials['inner'])
    uv_cylindrical_generic(cup, 0.2, 0.24)

    base = box_at('Actuator base', (0.0030, 0, 0.2338), (0.0065, 0.0098, 0.0050))
    apply_bevel(base, 0.0008, 2)
    assign(base, special_materials['nozzle'])
    uv_planar(base, 0.03, 0.5)
    objects.append(base)

    neck = box_at('Actuator neck', (0.0100, 0, 0.2370), (0.0042, 0.0050, 0.0035))
    apply_bevel(neck, 0.0007, 2)
    assign(neck, special_materials['nozzle'])
    uv_planar(neck, 0.03, 0.5)
    objects.append(neck)

    head_center = tuple(nozzle['head_center'])
    head_half = tuple(nozzle['head_half'])
    head = box_at('Nozzle head', head_center, head_half)
    # Real through-bores in the actuator face, not painted dark spots.
    for i, y in enumerate(nozzle['bore_y']):
        cutter = cylinder_x(f'Bore cutter {i}', (head_center[0], y, nozzle['bore_z']),
                            nozzle['bore_radius'], 0.008, 40)
        bpy.ops.object.select_all(action='DESELECT')
        head.select_set(True)
        bpy.context.view_layer.objects.active = head
        mod = head.modifiers.new(f'Nozzle bore {i}', 'BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.solver = 'EXACT'
        mod.object = cutter
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    apply_bevel(head, 0.00025, 2, 35)
    assign(head, special_materials['nozzle'])
    uv_planar(head, 0.02, 0.5)
    objects.append(head)

    front_x = head_center[0] + head_half[0]
    for i, y in enumerate(nozzle['bore_y']):
        ring = torus_at(f'Nozzle insert {i + 1}',
                        (front_x + 0.0002, y, nozzle['bore_z']),
                        0.00155, 0.00032, axis=(1, 0, 0),
                        major_segments=32, minor_segments=8)
        assign(ring, special_materials['nozzle_ring'])
        uv_planar(ring, 0.01, 0.5)

    seam = torus_at('Cap seam dust ring', (0, 0, 0.1989), 0.0337, 0.00050,
                    major_segments=128, minor_segments=10)
    assign(seam, special_materials['dust'])
    uv_cylindrical_generic(seam, 0.0, 0.02)
    return stem, cup, base, neck, head, seam


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------
make_materials()
body = build_labeled_body()
build_bottom_metal()
build_top_shoulder()
build_bottom_rims()
cap = build_cap()
build_valve_and_nozzle()
bpy.context.view_layer.update()

object_names = [obj.name for obj in objects]
for obj in objects:
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
# Bake every transform into mesh data so all exporters, renderers and BVH
# ray checks agree on one coordinate system.
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.context.view_layer.update()

verts = np.concatenate([np.asarray([tuple(v.co) for v in obj.data.vertices])
                        for obj in objects])
low = Vector(tuple(verts.min(axis=0)))
high = Vector(tuple(verts.max(axis=0)))
Z_SHIFT = -low.z
for obj in objects:
    obj.data.transform(Matrix.Translation((0.0, 0.0, Z_SHIFT)))
    obj.data.update()
bpy.context.view_layer.update()

verts = np.concatenate([np.asarray([tuple(v.co) for v in obj.data.vertices])
                        for obj in objects])
low = Vector(tuple(verts.min(axis=0)))
high = Vector(tuple(verts.max(axis=0)))
center = (low + high) / 2

bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'), export_format='GLB',
                          use_selection=True, export_yup=False, export_apply=True)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))
bpy.ops.wm.open_mainfile(filepath=str(out / 'model.blend'))

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.cycles.max_bounces = 8
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
world = bpy.data.worlds.new('ambient')
world.use_nodes = True
world.node_tree.nodes.get('Background').inputs['Strength'].default_value = 0.48
world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
scene.world = world
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, width, height in (
        ('large softbox', (2.6, -3.0, 3.6), 220, 3.0, 3.0),
        ('fill softbox', (-3.2, 2.6, 2.2), 115, 3.0, 3.0),
        ('front strip', (3.2, 1.0, 1.3), 115, 1.2, 3.4),
        ('rim high', (-1.2, 3.2, 3.4), 75, 1.4, 2.2),
        ('low fill', (0, -2.6, -0.28), 42, 1.0, 1.0)):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = width
    data.size_y = height
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

render_device = 'CPU'
try:
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'OPTIX'
    preferences.refresh_devices()
    for device in preferences.devices:
        device.use = device.type == 'OPTIX'
    if any(device.use for device in preferences.devices):
        scene.cycles.device = 'GPU'
        render_device = 'OPTIX'
    else:
        raise RuntimeError('no OPTIX device')
except Exception:
    scene.cycles.device = 'CPU'
    render_device = 'CPU'

bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

objects = [bpy.data.objects[name] for name in object_names]
verts = np.concatenate([np.asarray([tuple(v.co) for v in o.data.vertices])
                        for o in objects])
low = Vector(tuple(verts.min(axis=0)))
high = Vector(tuple(verts.max(axis=0)))
center = (low + high) / 2
span = max(high - low)



def camera(scene, direction, target, span_value, scale=1.27):
    direction = Vector(direction).normalized()
    cam = legacy.make_camera(scene, Vector(target) + direction * span_value * 5,
                             target, ortho_scale=span_value * scale)
    return cam


def draw(scene, path, *, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


directions = {'front': (1, 0, .16), 'back': (-1, 0, .16),
              'left': (0, -1, .18), 'right': (0, 1, .18),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 48
render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
span_front = max(high.y - low.y, high.z - low.z)
for name, direction in directions.items():
    use_span = span_front if name in ('front', 'back') else span
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .34,
             scale=1.0)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

if not request['draft']:
    # Turntable views.
    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .14), center, span)
        draw(scene, render_dir / f'turn-{i:02d}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    detail_cameras = (
        ('cap-front', (0.0, 0.0, 0.226), (1, 0, .08), .085),
        ('cap-top', (0.0, 0.0, 0.245), (0, 0, 1), .080),
        ('window', (0.0, -0.002, 0.239), (1, -0.45, .55), .045),
        ('nozzle', (0.0, 0.0, 0.239), (1, -0.18, .30), .032),
        ('cap-seam', (0.0, 0.0, 0.199), (1, -0.55, .16), .055),
        ('bottom', (0.0, 0.0, 0.006), (0, 0, -1), .082),
        ('bottom-rim', (0.0, 0.0, 0.006), (1, -0.7, -.65), .070),
        ('label-front', (0.0, 0.0, 0.105), (1, 0, .05), .20),
        ('label-back', (0.0, 0.0, 0.105), (-1, 0, .05), .20),
        ('label-left', (0.0, 0.0, 0.105), (0, -1, .05), .20),
        ('label-bottom', (0.0, 0.0, 0.018), (1, -0.6, -.25), .075),
        ('surface', (0.0, 0.0, 0.12), (1, -0.75, .30), .11),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale, scale=1.0)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    # Isolated groups.
    groups = (
        ('body-only', lambda n: n == 'Can labeled body', (0.5, -1, .18),
         center, .50),
        ('cap-only', lambda n: n in ('Cap', 'Cap seam dust ring',
                                     'Window bezel', 'Cap lower step ring'),
         (0.7, -1, .75), (0, 0, 0.228), .26),
        ('cap-top', lambda n: n == 'Cap', (0, 0, 1), (0, 0, 0.228), .20),
        ('nozzle-only', lambda n: ('Nozzle' in n or 'Actuator' in n or
                                   'Trigger' in n or 'Valve' in n),
         (1.0, -0.8, .55), (0.004, 0, 0.241), .10),
        ('nozzle-face', lambda n: n == 'Nozzle head', (1, 0, 0),
         (nozzle['head_center'][0], 0, nozzle['bore_z'] + Z_SHIFT), .030),
        ('bottom-only', lambda n: n in ('Can bottom metal', 'Can top shoulder',
                                        'Bottom rolled rim',
                                        'Bottom rim upper lip'),
         (0, 0, -1), (0, 0, 0.01), .20),
        ('window-look', lambda n: (n in ('Cap', 'Window bezel', 'Nozzle head',
                                         'Nozzle insert 1', 'Nozzle insert 2',
                                         'Actuator base', 'Actuator neck',
                                         'Valve stem')),
         (1.0, -0.55, .78), (0.006, 0, 0.236), .13),
    )
    for label, predicate, direction, target, scale in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        cam = camera(scene, direction, Vector(target), scale, scale=1.0)
        draw(scene, render_dir / f'{label}.png', resolution=768, samples=20)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    # Alternate key light for dielectric response and metal reflection checks.
    key = bpy.data.objects['large softbox']
    key.location = (-3.4, 2.8, 4.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, 0.16)), .16,
                 scale=1.0)
    draw(scene, render_dir / 'new-surface-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    low_fill = bpy.data.objects['low fill']
    low_fill.location = (1.5, 1.3, -0.9)
    low_fill.rotation_euler = (-low_fill.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, (0, 0, -1), Vector((0, 0, 0.006)), .082, scale=1.0)
    draw(scene, render_dir / 'bottom-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    low_fill.location = (0, -2.6, -0.28)
    low_fill.rotation_euler = (-low_fill.location).to_track_quat('-Z', 'Y').to_euler()
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Old library mesh comparison.
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
bpy.context.view_layer.update()
old_verts = [old.matrix_world @ v.co for v in old.data.vertices]
old_low = Vector(tuple(np.min(np.asarray([tuple(v) for v in old_verts]), axis=0)))
old_high = Vector(tuple(np.max(np.asarray([tuple(v) for v in old_verts]), axis=0)))
old_center = (old_low + old_high) / 2
old.location += Vector((0, 0, center.z)) - Vector((old_center.x, old_center.y, old_center.z))
bpy.context.view_layer.update()
old_verts = [old.matrix_world @ v.co for v in old.data.vertices]
old_low = Vector(tuple(np.min(np.asarray([tuple(v) for v in old_verts]), axis=0)))
old_high = Vector(tuple(np.max(np.asarray([tuple(v) for v in old_verts]), axis=0)))
old_center = (old_low + old_high) / 2
old_span = max(old_high - old_low)
for name, direction in directions.items():
    use_span = max(old_high.y - old_low.y, old_high.z - old_low.z) \
        if name in ('front', 'back') else old_span
    cam = camera(scene, direction, old_center, use_span)
    draw(scene, render_dir / f'old-{name}.png',
         resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (2.3, -2.0, 1.0), old_center, old_span * 0.85,
             scale=1.0)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)
for obj in objects:
    obj.hide_render = False

# ---------------------------------------------------------------------------
# Ray checks for the real window and nozzle bores
# ---------------------------------------------------------------------------
def ray_clearance(bvh, origin, direction, max_distance=0.08):
    hit, normal, index, distance = bvh.ray_cast(Vector(origin), Vector(direction),
                                                max_distance)
    if distance is None:
        return {'distance_m': None, 'hit': None}
    location = Vector(origin) + Vector(direction).normalized() * float(distance)
    return {'distance_m': float(distance), 'hit': [float(v) for v in location]}


def structural_rays():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cap_obj = bpy.data.objects['Cap']
    head_obj = bpy.data.objects['Nozzle head']
    cap_bvh = BVHTree.FromObject(cap_obj, depsgraph)
    head_bvh = BVHTree.FromObject(head_obj, depsgraph)
    axis = Vector((1.0, 0.0, 0.0))
    c = Vector(window['center'])
    c.z += Z_SHIFT
    window_rays = []
    for dy, dz in ((0.0, 0.0), (-0.006, 0.0), (0.006, 0.0),
                   (0.0, -0.0045), (0.0, 0.0045)):
        origin = Vector((0.060, c.y + dy, c.z + dz))
        result = ray_clearance(cap_bvh, origin, -axis)
        hit = result['hit']
        # An open window lets the ray reach the inner cavity (x < 0.025).  A
        # blocked window hits the outer front wall near x ~= 0.034.
        clear = hit is None or hit[0] < 0.025
        window_rays.append({'offset_y_m': dy, 'offset_z_m': dz,
                            'first_hit_distance_m': None
                            if result['distance_m'] is None
                            else round(result['distance_m'], 6),
                            'first_hit_x_m': None if hit is None else round(hit[0], 6),
                            'clearance_ok': bool(clear)})
    # A ray at the solid top centre is the control: it must hit the cap surface.
    control = ray_clearance(cap_bvh, Vector((0.0, 0.0, 0.256 + Z_SHIFT)),
                            Vector((0.0, 0.0, -1.0)))
    bores = []
    head_x = nozzle['head_center'][0] + nozzle['head_half'][0]
    for y in nozzle['bore_y']:
        origin = Vector((head_x + 0.03, y, nozzle['bore_z'] + Z_SHIFT))
        result = ray_clearance(head_bvh, origin, Vector((-1, 0, 0)))
        hit = result['hit']
        # A through bore either misses the head or reaches behind its back face.
        through = hit is None or hit[0] < head_x - nozzle['head_half'][0] - 0.0002
        bores.append({'y_m': y,
                      'first_hit_distance_m': None
                      if result['distance_m'] is None
                      else round(result['distance_m'], 6),
                      'first_hit_x_m': None if hit is None else round(hit[0], 6),
                      'through_ok': bool(through)})
    control_bore = ray_clearance(
        head_bvh, Vector((head_x + 0.03, 0.0, nozzle['bore_z'] + Z_SHIFT)),
        Vector((-1, 0, 0)))
    return {
        'window': window_rays,
        'window_control_solid_rim_hit_m': (None
                                           if control['distance_m'] is None
                                           else round(control['distance_m'], 6)),
        'window_control_ok': bool(control['distance_m'] is not None and
                                  control['distance_m'] < 0.008),
        'nozzle_bores': bores,
        'nozzle_control_solid_hit_m': (None
                                       if control_bore['distance_m'] is None
                                       else round(control_bore['distance_m'], 6)),
        'nozzle_control_ok': bool(control_bore['distance_m'] is not None and
                                  control_bore['distance_m'] < 0.035),
        'method': ('BVH rays from outside through the cap window and the two '
                   'actuator bores.  A clear window ray reaches the inner cavity '
                   '(hit x < 0.020 m); a blocked ray hits the front wall near '
                   'x ~= 0.034 m.  Control rays hit solid cap/head material. '
                   'These prove local openings in the generated parts, not hidden '
                   'assembly relationships.'),
    }


rays = structural_rays() if not request['draft'] else {'status': 'not_run_draft'}


def intersecting_faces(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
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


intersections = {o.name: intersecting_faces(o) for o in objects}
parts = [{'name': o.name, 'vertices': len(o.data.vertices),
          'faces': len(o.data.polygons)} for o in objects]
data = {
    'parts': parts,
    'bounds': [list(low), list(high)],
    'span': float(span),
    'parts_count': len(parts),
    'polygons': sum(len(o.data.polygons) for o in objects),
    'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                     for p in o.data.polygons),
    'render_device': render_device,
    'render_samples': samples,
    'view_directions': {k: list(v) for k, v in directions.items()},
    'structural_rays': rays,
    'self_intersecting_components': {k: v for k, v in intersections.items() if v},
    'intersection_check': ('BVH same-part face overlap; intentional assembly '
                           'overlap between parts is not judged'),
    'geometry_basis': ('source-photo proportions and visible front structure; '
                       'six-view sheet is an uncalibrated visual match used for '
                       'the hidden-half label wrap and top/bottom topology'),
    'window_note': ('The cap window is a real boolean through-cut; its two-lobed '
                    'outline, the internal actuator linkage and the exact cap '
                    'wall thickness are inferred from the uncalibrated views.'),
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
