"""Blender worker for the review-only gas-cylinder candidate (item 21).

Shape basis: the clean/rust photographs show a tall brushed-metal cylinder with
a rounded shoulder, a narrow neck with thin collar rings, a silver valve body,
a threaded side outlet, a small knurled safety plug, a front nameplate and a
black scalloped handwheel with a metal centre cap.  The user six-view sheet is
an uncalibrated visual-match atlas and is used for part topology/appearance
only, never as measured geometry.  The candidate is a parametric part-based
rebuild; it never replaces the library mesh, stays manual-only and never enters
automatic placement.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
config = request['config']
geo = config['geometry']
profile = config['material_profile']
texture_cfg = config['texture']
render_cfg = config['render']
objects = []
materials = {}
TAU = math.tau


# ---------------------------------------------------------------------------
# Materials and UVs
# ---------------------------------------------------------------------------
def image_node(nodes, key, noncolor=False):
    node = nodes.new('ShaderNodeTexImage')
    node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
    node.image.pack()
    if noncolor:
        node.image.colorspace_settings.name = 'Non-Color'
    node.interpolation = 'Linear'
    return node


def _srgb_to_linear(value):
    value = value / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def textured_material(name, prefix, normal_strength):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    color = image_node(nodes, prefix + '_base')
    metal = image_node(nodes, prefix + '_mr', True)
    normal = image_node(nodes, prefix + '_normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(metal.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], bsdf.inputs['Roughness'])
    links.new(split.outputs['Blue'], bsdf.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = normal_strength
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    return material


def uniform_material(name, rgb, metallic, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (
        _srgb_to_linear(rgb[0]), _srgb_to_linear(rgb[1]), _srgb_to_linear(rgb[2]), 1.0)
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    return material


def make_materials():
    materials['body'] = textured_material('21 brushed aluminium body + rust', 'body', .055)
    materials['steel'] = textured_material('21 valve steel', 'steel', .035)
    materials['black'] = textured_material('21 black handwheel plastic', 'black', .045)
    materials['cap'] = textured_material('21 radial brushed cap', 'cap', .25)
    materials['bottom'] = textured_material('21 bottom brushed plate', 'bottom', .035)
    materials['bore'] = uniform_material('21 dark bore steel',
                                         profile['bore']['base_rgb'],
                                         profile['bore']['metallic'],
                                         profile['bore']['base_roughness'])
    materials['clay'] = uniform_material('21 clay grey', [148, 148, 148], 0.0, .62)
    materials['old_display'] = uniform_material('21 old display neutral',
                                               [168, 164, 155], .25, .55)


def ensure_uv(obj):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')


def box_uv(obj, scale):
    ensure_uv(obj)
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    for poly in mesh.polygons:
        normal = poly.normal
        axis = max(range(3), key=lambda i: abs(normal[i]))
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            if axis == 0:
                u, v = co.y / scale, co.z / scale
            elif axis == 1:
                u, v = co.x / scale, co.z / scale
            else:
                u, v = co.x / scale, co.y / scale
            uv[loop_index].uv = (u % 1.0, v % 1.0)


def cyl_uv(obj, z0, z1):
    ensure_uv(obj)
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    height = max(1e-9, z1 - z0)
    for poly in mesh.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.y, co.x))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            delta = (angle - ca + math.pi) % TAU - math.pi
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv[loop_index].uv = ((ca + delta) / TAU + .5,
                                 (co.z - z0) / height)


def axis_y_uv(obj, cx, cz, y0, y1):
    """Cylindrical UV around a Y-parallel axis (for the outlet and safety plug)."""
    ensure_uv(obj)
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    span = y1 - y0
    if abs(span) < 1e-9:
        span = 1e-9
    for poly in mesh.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.z - cz, co.x - cx))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            delta = (angle - ca + math.pi) % TAU - math.pi
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv[loop_index].uv = ((ca + delta) / TAU + .5,
                                 (co.y - y0) / span)


def disc_uv(obj, cx, cy, radius):
    ensure_uv(obj)
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv[loop_index].uv = ((co.x - cx) / (2 * radius) + .5,
                                 (co.y - cy) / (2 * radius) + .5)


def set_smooth_threads(obj, cfg):
    """Smooth the collar/bore but keep the annular thread ridges crisp."""
    start = cfg['thread_start'] - 0.0005
    end = cfg['thread_end'] + 0.0005
    for poly in obj.data.polygons:
        y = poly.center.y
        poly.use_smooth = not (start <= y <= end)


def set_smooth(obj, mode='all', axis=None):
    if mode == 'flat':
        for poly in obj.data.polygons:
            poly.use_smooth = False
        return
    if mode == 'side' and axis is not None:
        axis = Vector(axis).normalized()
        for poly in obj.data.polygons:
            poly.use_smooth = abs(poly.normal.dot(axis)) < .80
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True


def assign(obj, key, *, smooth='all', axis=None, uv='box', scale=.08, z0=None, z1=None):
    obj.data.materials.clear()
    obj.data.materials.append(materials[key])
    if uv == 'cyl':
        cyl_uv(obj, z0, z1)
    elif uv == 'disc':
        disc_uv(obj, *scale)
    elif uv == 'yaxis':
        axis_y_uv(obj, *scale)
    elif uv == 'none':
        ensure_uv(obj)
    else:
        box_uv(obj, scale)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def mesh_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def apply_bevel(obj, width, segments=2):
    if not width:
        return obj
    bpy.ops.object.select_all(action='DESELECT')
    mod = obj.modifiers.new('Manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        obj.modifiers.remove(mod)
    return obj


def revolve_solid(name, profile_points, segments):
    verts, rings = [], []
    for r, z in profile_points:
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


def torus(name, center, normal, major, minor, major_segments=96, minor_segments=12):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                     major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=center)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = Vector(normal).to_track_quat('Z', 'Y').to_euler()
    return obj


def cylinder_z(name, center_xy, radius, z0, z1, segments=64):
    cx, cy = center_xy
    obj = revolve_solid(name, [(0.0, z0), (radius, z0), (radius, z1), (0.0, z1)],
                        segments)
    obj.data.transform(Matrix.Translation((cx, cy, 0.0)))
    obj.data.update()
    return obj


def revolve_profile_y(name, profile_points, segments, center, reverse=False):
    obj = revolve_solid(name, profile_points, segments)
    obj.data.transform(Matrix.Rotation(-math.pi / 2, 4, 'X'))
    if reverse:
        obj.data.transform(Matrix.Rotation(math.pi, 4, 'Z'))
    obj.data.transform(Matrix.Translation(center))
    obj.data.update()
    return obj


def knurled_cylinder_z(name, center_xy, radius, z0, z1, teeth, depth):
    cx, cy = center_xy
    n = teeth * 2
    verts = []
    bottom, top = [], []
    for s in range(n):
        a = TAU * s / n
        rr = radius if s % 2 == 0 else radius - depth
        bottom.append(len(verts))
        verts.append((cx + rr * math.cos(a), cy + rr * math.sin(a), z0))
    for s in range(n):
        a = TAU * s / n
        rr = radius if s % 2 == 0 else radius - depth
        top.append(len(verts))
        verts.append((cx + rr * math.cos(a), cy + rr * math.sin(a), z1))
    cb = len(verts)
    verts.append((cx, cy, z0))
    ct = len(verts)
    verts.append((cx, cy, z1))
    faces = []
    for s in range(n):
        s2 = (s + 1) % n
        faces.append((bottom[s], bottom[s2], top[s2], top[s]))
        faces.append((cb, bottom[s2], bottom[s]))
        faces.append((ct, top[s], top[s2]))
    return mesh_object(name, verts, faces)


def fluted_wheel(name, cfg, per_tooth=10):
    z0 = cfg['z0']
    flutes = cfg['flutes']
    depth = cfg['flute_depth']
    n = flutes * per_tooth
    verts = []
    rings = []
    for off, base_r in cfg['rings']:
        ids = []
        z = z0 + off
        for i in range(n):
            a = TAU * i / n
            groove = (0.5 - 0.5 * math.cos(flutes * a)) ** 1.35
            r = max(0.0015, base_r - depth * groove)
            ids.append(len(verts))
            verts.append((r * math.cos(a), r * math.sin(a), z))
        rings.append(ids)
    faces = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        for s in range(n):
            s2 = (s + 1) % n
            faces.append((a[s], a[s2], b[s2], b[s]))
    # Bottom cap.
    cb = len(verts)
    verts.append((0.0, 0.0, z0))
    for s in range(n):
        faces.append((cb, rings[0][(s + 1) % n], rings[0][s]))
    # Shallow dished top with an inner ring and centre vertex.
    top_z = z0 + cfg['rings'][-1][0]
    inner_z = top_z - cfg['top_inner_drop']
    inner = []
    for i in range(n):
        a = TAU * i / n
        inner.append(len(verts))
        verts.append((cfg['top_inner_r'] * math.cos(a),
                      cfg['top_inner_r'] * math.sin(a), inner_z))
    for s in range(n):
        s2 = (s + 1) % n
        faces.append((rings[-1][s], rings[-1][s2], inner[s2], inner[s]))
    ct = len(verts)
    verts.append((0.0, 0.0, inner_z - cfg['top_center_drop']))
    for s in range(n):
        faces.append((ct, inner[s], inner[(s + 1) % n]))
    return mesh_object(name, verts, faces)


def cut_box(obj, center, half):
    cutter = box_at(obj.name + ' cutter', center, half)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mod = obj.modifiers.new('Slot', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    finally:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return obj


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


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


# ---------------------------------------------------------------------------
# Spout / profile helpers
# ---------------------------------------------------------------------------
def spout_profile(cfg):
    boss = cfg['boss_r']
    outer = cfg['outer_r']
    bore = cfg['bore_r']
    tip = cfg['y_tip']
    start = cfg['thread_start']
    end = cfg['thread_end']
    depth = cfg['thread_depth']
    grooves = cfg['thread_grooves']
    blind = cfg['blind_end']
    pts = [(boss, 0.0), (boss, 0.0060), (0.0095, 0.0080), (0.0081, 0.0100),
           (outer, start)]
    pitch = (end - start) / grooves
    for k in range(grooves):
        pts.append((outer - depth, start + (k + 0.5) * pitch))
        pts.append((outer, start + (k + 1) * pitch))
    pts += [(0.0078, 0.0328), (0.0070, tip), (bore, tip),
            (bore, blind), (0.0, blind)]
    return [(0.0, 0.0)] + pts


def safety_plug_profile(cfg):
    length = cfg['length']
    body = cfg['body_r']
    neck = cfg['neck_r']
    socket = cfg['socket_r']
    socket_depth = cfg['socket_depth']
    knurl_from = cfg['knurl_from']
    return [(0.0, 0.0), (neck, 0.0), (neck, knurl_from),
            (body, knurl_from), (body, length), (socket, length),
            (socket, length - socket_depth), (0.0, length - socket_depth)]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build():
    seg = geo['radial_segments']
    body_cfg = geo['body']
    body = revolve_solid('Gas cylinder body', body_cfg['profile'], seg)
    assign(body, 'body', uv='cyl', z0=body_cfg['z0'], z1=body_cfg['z1'],
           smooth='side', axis=(0, 0, 1))

    plate_cfg = geo['bottom_plate']
    plate = revolve_solid('Bottom brushed plate', [
        (0.0, plate_cfg['z0']), (plate_cfg['r'], plate_cfg['z0']),
        (plate_cfg['r'], plate_cfg['z1']), (0.0, plate_cfg['z1'])], seg)
    assign(plate, 'bottom', uv='disc', scale=(0.0, 0.0, plate_cfg['r']),
           smooth='side', axis=(0, 0, 1))

    for i, ring in enumerate(geo['neck_rings'], start=1):
        obj = torus(f'Neck ring {i}', (0, 0, ring['z']), (0, 0, 1),
                    ring['major'], ring['minor'], 128, 10)
        assign(obj, 'steel', uv='box', scale=.03, smooth='all')

    valve_cfg = geo['valve_body']
    valve = revolve_solid('Valve body', valve_cfg['profile'], seg)
    assign(valve, 'steel', uv='cyl', z0=valve_cfg['z0'], z1=valve_cfg['z1'],
           smooth='side', axis=(0, 0, 1))

    spout_cfg = geo['spout']
    spout = revolve_profile_y('Outlet spout', spout_profile(spout_cfg), seg,
                              (0.0, 0.0, spout_cfg['z']))
    assign(spout, 'steel', uv='yaxis',
           scale=(0.0, spout_cfg['z'], spout_cfg['y_start'], spout_cfg['y_tip']),
           smooth='side', axis=(0, 1, 0))
    set_smooth_threads(spout, spout_cfg)

    plug_cfg = geo['safety_plug']
    plug_body = revolve_profile_y('Safety plug body', safety_plug_profile(plug_cfg),
                                  seg, (0.0, plug_cfg['mount_y'],
                                        plug_cfg['center_z']), reverse=True)
    assign(plug_body, 'steel', uv='yaxis',
           scale=(0.0, plug_cfg['center_z'], plug_cfg['mount_y'],
                  plug_cfg['mount_y'] - plug_cfg['length']),
           smooth='side', axis=(0, 1, 0))
    plug_knurl = knurled_cylinder_z('Safety plug knurl', (0.0, 0.0),
                                    plug_cfg.get('knurl_r', plug_cfg['body_r'] + 0.0011),
                                    plug_cfg['knurl_from'], plug_cfg['knurl_to'],
                                    plug_cfg['knurl_teeth'], plug_cfg['knurl_depth'])
    plug_knurl.data.transform(Matrix.Rotation(-math.pi / 2, 4, 'X'))
    plug_knurl.data.transform(Matrix.Rotation(math.pi, 4, 'Z'))
    plug_knurl.data.transform(Matrix.Translation(
        (0.0, plug_cfg['mount_y'], plug_cfg['center_z'])))
    plug_knurl.data.update()
    assign(plug_knurl, 'steel', uv='box', scale=.02, smooth='flat')

    plaque_cfg = geo['plaque']
    pc = plaque_cfg['center']
    plaque = box_at('Valve nameplate',
                    ((plaque_cfg['x0'] + plaque_cfg['x1']) / 2, pc[1], pc[2]),
                    ((plaque_cfg['x1'] - plaque_cfg['x0']) / 2,
                     plaque_cfg['width_y'] / 2, plaque_cfg['height_z'] / 2))
    apply_bevel(plaque, plaque_cfg['bevel'], 3)
    assign(plaque, 'steel', uv='box', scale=.03, smooth='flat')

    nut_cfg = geo['packing_nut']
    nut = revolve_solid('Valve packing nut', [
        (0.0, nut_cfg['z0']), (nut_cfg['r'] * .88, nut_cfg['z0']),
        (nut_cfg['r'], nut_cfg['z0'] + 0.0009),
        (nut_cfg['r'], nut_cfg['z1'] - 0.0009),
        (nut_cfg['r'] * .88, nut_cfg['z1']), (0.0, nut_cfg['z1'])], 96)
    assign(nut, 'steel', uv='cyl', z0=nut_cfg['z0'], z1=nut_cfg['z1'],
           smooth='side', axis=(0, 0, 1))

    spindle_cfg = geo['spindle']
    spindle = revolve_solid('Valve spindle', [
        (0.0, spindle_cfg['z0']), (spindle_cfg['r'], spindle_cfg['z0']),
        (spindle_cfg['r'], spindle_cfg['z1']), (0.0, spindle_cfg['z1'])], 64)
    assign(spindle, 'steel', uv='box', scale=.02, smooth='side', axis=(0, 0, 1))

    wheel_cfg = geo['handwheel']
    wheel = fluted_wheel('Black scalloped handwheel', wheel_cfg)
    assign(wheel, 'black', uv='box', scale=.06, smooth='all')

    cap_cfg = geo['cap']
    cap = revolve_solid('Handwheel metal cap', [
        (0.0, cap_cfg['z0']), (cap_cfg['r'] * .94, cap_cfg['z0']),
        (cap_cfg['r'], cap_cfg['z0'] + 0.0004),
        (cap_cfg['r'], cap_cfg['z1'] - 0.0004),
        (cap_cfg['r'] * .96, cap_cfg['z1']), (0.0, cap_cfg['z1'])], 96)
    assign(cap, 'cap', uv='disc', scale=(0.0, 0.0, cap_cfg['r']),
           smooth='side', axis=(0, 0, 1))

    screw_cfg = geo['screw']
    screw = cylinder_z('Handwheel centre screw', (0.0, 0.0), screw_cfg['r'],
                       screw_cfg['z0'], screw_cfg['z1'], 64)
    cut_box(screw, (0.0, 0.0, screw_cfg['z1'] - screw_cfg['slot_depth'] / 2),
            (screw_cfg['slot_width'] / 2, screw_cfg['r'] * 1.25,
             screw_cfg['slot_depth'] / 2))
    assign(screw, 'steel', uv='box', scale=.012, smooth='flat')


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def camera(scene, direction, center, span, scale=1.10):
    direction = Vector(direction).normalized()
    cam = legacy.make_camera(scene, Vector(center) + direction * span * 5, center,
                             ortho_scale=span * scale)
    return cam


def camera_axis(scene, axis, center, span):
    """Orthographic camera along axis with explicit screen orientation."""
    span = float(span)
    data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera', data)
    scene.collection.objects.link(cam)
    data.type = 'ORTHO'
    data.ortho_scale = span
    direction = Vector(axis).normalized()
    eye = Vector(center) + direction * span * 5
    if abs(direction.z) > .999:
        if direction.z > 0:            # top view: screen right +Y, up -X
            right = Vector((0.0, 1.0, 0.0))
            up = Vector((-1.0, 0.0, 0.0))
        else:                           # bottom view: screen right +Y, up +X
            right = Vector((0.0, 1.0, 0.0))
            up = Vector((1.0, 0.0, 0.0))
        z_local = direction
        rotation = Matrix((right, up, z_local)).transposed().to_4x4()
        cam.matrix_world = Matrix.Translation(eye) @ rotation
    else:
        cam.location = eye
        cam.rotation_euler = (Vector(center) - eye).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam
    return cam


def draw(scene, path, *, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def set_key_light(scene, position):
    key = bpy.data.objects.get('large softbox')
    if key is None:
        return
    key.location = position
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()


def hide_all(obs, hidden=True):
    for obj in obs:
        obj.hide_render = hidden


def clay_override(obs):
    clay = bpy.data.materials.get('21 clay grey')
    if clay is None:
        clay = uniform_material('21 clay grey', [148, 148, 148], 0.0, .62)
    saved = {}
    for obj in obs:
        saved[obj.name] = [slot.material for slot in obj.material_slots]
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    return saved


def restore_materials(obs, saved):
    for obj in obs:
        obj.data.materials.clear()
        for material in saved[obj.name]:
            obj.data.materials.append(material)


try:
    scene = legacy.setup_scene('OPTIX', 512)
    render_device = 'OPTIX'
except Exception:
    scene = legacy.setup_scene('CPU', 512)
    render_device = 'CPU'
scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = \
    render_cfg['world_strength']
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, width, height in (
        ('large softbox', tuple(render_cfg['key_light_a']), 220, 3.0, 3.0),
        ('fill softbox', (-3.2, 2.6, 2.2), 115, 3.0, 3.0),
        ('front strip', (3.2, 1.0, 1.3), 115, 1.2, 3.4),
        ('rim high', (-1.2, 3.2, 3.4), 75, 1.4, 2.2),
        ('top fill', (0.0, 0.5, 4.0), 100, 2.5, 2.5),
        ('low fill', (0, -2.6, -.28), 42, 1.0, 1.0)):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = width
    data.size_y = height
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

make_materials()
build()
sc = geo['spout']
plug = geo['safety_plug']
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
# The model is already built with its bottom on z=0; no shift is applied.
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
scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
objects = [bpy.data.objects[name] for name in object_names]
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)

parts = [{'name': o.name, 'vertices': len(o.data.vertices),
          'faces': len(o.data.polygons)} for o in objects]
intersections = {o.name: intersecting_faces(o) for o in objects}
render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)

directions = {'front': (1, 0, 0), 'back': (-1, 0, 0),
              'left': (0, -1, 0), 'right': (0, 1, 0),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else render_cfg['six_view_resolution']
samples = render_cfg['draft_samples'] if request['draft'] else render_cfg['samples']
height = high.z - low.z
width = max(high.x - low.x, high.y - low.y)
span_side = height * 1.10
span_cap = width * 1.42
for name, direction in directions.items():
    if name in ('top', 'bottom'):
        cam = camera_axis(scene, direction, center, span_cap)
    else:
        cam = camera(scene, direction, center, span_side)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

# Close-up: body shoulder + valve + handwheel.
close_target = Vector((0.0, 0.0, 0.352))
close_span = 0.155
cam = camera(scene, (0.78, -0.62, 0.16), close_target, close_span)
draw(scene, render_dir / 'new-material-close.png',
     resolution=1024 if not request['draft'] else 640,
     samples=32 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
set_key_light(scene, tuple(render_cfg['key_light_b']))
cam = camera(scene, (0.78, -0.62, 0.16), close_target, close_span)
draw(scene, render_dir / 'new-surface-alternate.png',
     resolution=1024 if not request['draft'] else 640,
     samples=32 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
set_key_light(scene, tuple(render_cfg['key_light_a']))

if not request['draft']:
    # Clay geometry views.
    saved = clay_override(objects)
    cam = camera(scene, directions['front'], center, span_side)
    draw(scene, render_dir / 'new-clay-front.png',
         resolution=render_cfg['clay_resolution'], samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (0.78, -0.62, 0.16), close_target, close_span)
    draw(scene, render_dir / 'new-clay-close.png', resolution=1024, samples=20)
    bpy.data.objects.remove(cam, do_unlink=True)
    restore_materials(objects, saved)

    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .12), center, span_side)
        draw(scene, render_dir / f'turn-{i:02d}.png',
             resolution=render_cfg['turntable_resolution'], samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    sc = geo['spout']
    plug = geo['safety_plug']
    wheel_cfg = geo['handwheel']
    detail_cameras = (
        ('front', center, (1, 0, 0), span_side, 'plain'),
        ('back', center, (-1, 0, 0), span_side, 'plain'),
        ('side', center, (0, -1, 0), span_side, 'plain'),
        ('top', center, (0, 0, 1), span_cap, 'axis'),
        ('bottom', center, (0, 0, -1), span_cap, 'axis'),
        ('valve', (0.0, 0.0, 0.356), (0.78, -0.62, 0.20), 0.075, 'plain'),
        ('handwheel', (0.0, 0.0, 0.397), (0.66, -0.72, 0.30), 0.060, 'plain'),
        ('handwheel-top', (0.0, 0.0, 0.397), (0, 0, 1), 0.058, 'axis'),
        ('flutes', (0.0, 0.0, 0.396), (0.35, -0.55, 0.76), 0.047, 'plain'),
        ('spout', (0.0, 0.017, 0.356), (0.72, -0.62, 0.28), 0.048, 'plain'),
        ('spout-end', (0.0, 0.020, 0.356), (0, 1, 0), 0.030, 'plain'),
        ('spout-side', (0.0, 0.017, 0.356), (1, 0, 0.12), 0.043, 'plain'),
        ('threads', (0.0, 0.027, 0.356), (0.55, -0.72, 0.22), 0.026, 'plain'),
        ('safety-plug', (0.0, -0.018, 0.3452), (-0.55, -0.78, 0.22), 0.024, 'plain'),
        ('plaque', (0.016, 0.0, 0.354), (1, -0.18, 0.10), 0.034, 'plain'),
        ('neck-rings', (0.0, 0.0, 0.330), (0.72, -0.66, 0.12), 0.043, 'plain'),
        ('shoulder', (0.0, 0.0, 0.292), (0.70, -0.68, 0.12), 0.098, 'plain'),
        ('bottom-edge', (0.0, 0.0, 0.012), (0.58, -0.78, 0.10), 0.060, 'plain'),
        ('surface', (0.0, 0.0, 0.180), (1, 0, 0.05), 0.055, 'plain'),
        ('rust-shoulder', (0.0, 0.0, 0.303), (0.62, -0.76, 0.16), 0.070, 'plain'),
        ('rust-bottom', (0.0, 0.0, 0.040), (0.58, -0.80, 0.10), 0.075, 'plain'),
    )
    for item in detail_cameras:
        name, target, direction, dspan, mode = item
        if mode == 'axis':
            cam = camera_axis(scene, direction, Vector(target), dspan)
        else:
            cam = camera(scene, direction, Vector(target), dspan)
        draw(scene, render_dir / f'detail-{name}.png',
             resolution=render_cfg['detail_resolution'], samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('spout-only', lambda n: n == 'Outlet spout',
         (0.72, -0.62, 0.28), (0, 0.017, sc['z']), 0.048),
        ('spout-only-end', lambda n: n == 'Outlet spout',
         (0, 1, 0), (0, 0.020, sc['z']), 0.030),
        ('valve-spout-only',
         lambda n: n in ('Valve body', 'Outlet spout', 'Safety plug body',
                         'Safety plug knurl', 'Valve nameplate', 'Valve packing nut'),
         (0.70, -0.66, 0.22), (0, 0, 0.356), 0.085),
        ('handwheel-only',
         lambda n: n in ('Black scalloped handwheel', 'Handwheel metal cap',
                         'Handwheel centre screw'),
         (0.60, -0.76, 0.28), (0, 0, 0.397), 0.060),
        ('safety-plug-only',
         lambda n: n in ('Safety plug body', 'Safety plug knurl'),
         (-0.55, -0.78, 0.22), (0, -0.018, plug['center_z']), 0.024),
        ('neck-only',
         lambda n: n in ('Gas cylinder body', 'Neck ring 1', 'Neck ring 2',
                         'Neck ring 3', 'Valve body'),
         (0.72, -0.66, 0.12), (0, 0, 0.330), 0.085),
        ('top-view-all', lambda n: True, (0, 0, 1), center, span_cap),
    )
    for label, predicate, direction, target, dspan in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        if label == 'top-view-all':
            cam = camera_axis(scene, direction, Vector(target), dspan)
        else:
            cam = camera(scene, direction, Vector(target), dspan)
        draw(scene, render_dir / f'{label}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    # Alternate key light on the same front camera: highlights must move.
    set_key_light(scene, tuple(render_cfg['key_light_b']))
    cam = camera(scene, directions['front'], center, span_side)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (0.78, -0.62, 0.16), close_target, close_span)
    draw(scene, render_dir / 'new-surface-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    set_key_light(scene, tuple(render_cfg['key_light_a']))

# ---------------------------------------------------------------------------
# Old mesh: display alignment, then six views and close-up
# ---------------------------------------------------------------------------
hide_all(objects)
old = legacy.load_mesh(request['original_mesh'])
if not old.data.materials:
    old.data.materials.append(materials['old_display'])
bpy.context.view_layer.update()

def old_points():
    return np.array([tuple(p) for p in legacy.world_vertices(old)], dtype=np.float64)

pts = old_points()
centroid = pts.mean(axis=0)
cov = np.cov((pts - centroid).T)
eigvals, eigvecs = np.linalg.eigh(cov)
axis = eigvecs[:, int(np.argmax(eigvals))]
proj = (pts - centroid) @ axis
lo_pts = pts[proj < np.percentile(proj, 15)]
hi_pts = pts[proj > np.percentile(proj, 85)]

def radial_spread(q):
    c = q.mean(axis=0)
    d = q - c
    return float(np.linalg.norm(d - np.outer(d @ axis, axis), axis=1).mean())

if radial_spread(hi_pts) > radial_spread(lo_pts):
    axis = -axis
z_axis = axis / np.linalg.norm(axis)
tmp = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
x_axis = np.cross(tmp, z_axis)
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
R = np.stack([x_axis, y_axis, z_axis], axis=0)
old.data.transform(Matrix(R.tolist()).to_4x4())
old.data.update()
bpy.context.view_layer.update()
pts = old_points()
zmax = pts[:, 2].max()
zmin = pts[:, 2].min()
zspan = max(1e-9, zmax - zmin)
top = pts[pts[:, 2] > zmax - 0.15 * zspan]
axis_xy = np.median(pts[pts[:, 2] < zmin + 0.35 * zspan, :2], axis=0)
r = np.linalg.norm(top[:, :2] - axis_xy, axis=1)
far = top[r > np.percentile(r, 95)]
if len(far):
    spout_angle = math.atan2(float(far[:, 1].mean() - axis_xy[1]),
                             float(far[:, 0].mean() - axis_xy[0]))
else:
    spout_angle = 0.0
rotate_z = math.pi / 2 - spout_angle
old.data.transform(Matrix.Rotation(rotate_z, 4, 'Z'))
old.data.update()
bpy.context.view_layer.update()
points = old_points()
zmin, zmax = points[:, 2].min(), points[:, 2].max()
old_height = max(1e-9, zmax - zmin)
scale_factor = height / old_height
old.data.transform(Matrix.Scale(scale_factor, 4))
old.data.update()
bpy.context.view_layer.update()
points = old_points()
lo_p = points.min(axis=0)
hi_p = points.max(axis=0)
old_center = (lo_p + hi_p) / 2
old.data.transform(Matrix.Translation((-old_center[0], -old_center[1], -lo_p[2])))
old.data.update()
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old_span_front = max(oldhigh.y - oldlow.y, oldhigh.z - oldlow.z)

for name, direction in directions.items():
    if name in ('top', 'bottom'):
        cam = camera_axis(scene, direction, center, span_cap)
    else:
        cam = camera(scene, direction, center, span_side)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (0.78, -0.62, 0.16), close_target, close_span)
draw(scene, render_dir / 'old-material-close.png', resolution=1024, samples=32)
bpy.data.objects.remove(cam, do_unlink=True)

# ---------------------------------------------------------------------------
# Source-photo framing diagnostics on the new candidate
# ---------------------------------------------------------------------------
old.hide_render = True
for obj in objects:
    obj.hide_render = False
source = request['source_camera']
source_bbox = request.get('source_alpha_bbox') or [0, 0, 511, 511]
factor = span / request['old_span_raw']
result_diag = {}


def source_camera():
    cam = legacy.make_camera(scene, (1.9 * factor, 0, center.z),
                             (0, 0, center.z), fov=source['fov_deg'])
    cam.data.shift_x = source.get('shift_x', 0)
    cam.data.shift_y = source.get('shift_y', 0)
    return cam


def fit_source_camera(cam):
    from bpy_extras.object_utils import world_to_camera_view
    bpy.context.view_layer.update()
    corners = [Vector((x, y, z)) for x in (low.x, high.x)
               for y in (low.y, high.y) for z in (low.z, high.z)]
    projected = [world_to_camera_view(scene, cam, p) for p in corners]
    height_px = (max(v.y for v in projected) - min(v.y for v in projected)) * 512
    source_h = max(1, source_bbox[3] - source_bbox[1] + 1)
    source_fit = height_px / source_h
    cam.location.x *= source_fit
    bpy.context.view_layer.update()
    trial = render_dir / 'new-source-trial.png'
    draw(scene, trial, resolution=512, samples=8)
    img = bpy.data.images.load(str(trial))
    w, h = img.size
    pixels = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(pixels)
    mask = pixels.reshape(h, w, 4)[..., 3] >= .5
    rows = np.nonzero(mask.any(axis=1))[0]
    trial_height = int(rows.max() - rows.min() + 1) if len(rows) else 1
    correction = trial_height / source_h
    cam.location.x *= correction
    bpy.context.view_layer.update()
    columns = np.nonzero(mask.any(axis=0))[0]
    trial_cx = ((columns.min() + columns.max() + 1) / 2) if len(columns) else w / 2
    source_cx = (source_bbox[0] + source_bbox[2] + 1) / 2
    shift_px = (trial_cx - source_cx) / max(1e-9, correction)
    visible_width = 2 * abs(cam.location.x) * math.tan(math.radians(source['fov_deg']) / 2)
    shift_world = shift_px / w * visible_width
    cam.location.y += shift_world
    trial_cy = ((rows.min() + rows.max() + 1) / 2) if len(rows) else h / 2
    source_cy = h - (source_bbox[1] + source_bbox[3] + 1) / 2
    vertical_px = (trial_cy - source_cy) / max(1e-9, correction)
    vertical_world = vertical_px / h * visible_width
    cam.location.z += vertical_world
    bpy.context.view_layer.update()
    bpy.data.images.remove(img)
    return {'height_px_initial': round(float(height_px), 3),
            'source_fit': round(float(source_fit), 5),
            'trial_height_px': trial_height,
            'correction': round(float(correction), 5),
            'shift_px': round(float(shift_px), 3),
            'vertical_shift_px': round(float(vertical_px), 3)}

cam = source_camera()
fit = fit_source_camera(cam)
draw(scene, render_dir / 'new-source.png', resolution=512, samples=samples)
bpy.data.objects.remove(cam, do_unlink=True)
result_diag['strict'] = fit
for roll in (-12.0, 12.0):
    cam = source_camera()
    base = cam.rotation_euler.to_quaternion()
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = base @ Quaternion((0, 0, 1), math.radians(roll))
    fit2 = fit_source_camera(cam)
    name = f'new-source-roll{roll:+.1f}.png'
    draw(scene, render_dir / name, resolution=512, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
    result_diag[f'{roll:+.1f}'] = fit2
bpy.data.objects.remove(old, do_unlink=True)

data = {
    'parts': parts,
    'bounds': [list(low), list(high)],
    'span': span,
    'height': height,
    'width': width,
    'parts_count': len(parts),
    'polygons': sum(len(o.data.polygons) for o in objects),
    'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                     for p in o.data.polygons),
    'materials': ['body', 'steel', 'black', 'cap', 'bottom'],
    'render_device': render_device,
    'self_intersecting_components': {k: v for k, v in intersections.items() if v},
    'intersection_check': ('BVH same-part face overlap; assembly overlap between '
                           'parts is not judged'),
    'geometry_basis': ('original photo proportions; six-view sheet is an '
                       'uncalibrated visual-match reference'),
    'render_samples': samples,
    'view_directions': {k: list(v) for k, v in directions.items()},
    'view_note': ('Front camera is at +X, so the +Y outlet appears on the right, '
                  'matching the source photo and the front reference panel. The '
                  'top/bottom cameras are rolled so +Y is screen-right. The '
                  'reference left/right panels both draw the outlet on the same '
                  'side and are not a consistent turntable.'),
    'source_camera_note': ('source-photo framing distance scaled by new/old span, '
                           'then fitted to the source-alpha height; diagnostic only'),
    'source_camera_factor': factor,
    'source_camera_fits': result_diag,
    'spout': {'z': sc['z'], 'outer_r': sc['outer_r'], 'bore_r': sc['bore_r'],
              'y_tip': sc['y_tip'], 'blind_end': sc['blind_end'],
              'thread_grooves': sc['thread_grooves']},
    'safety_plug': {'center_z': plug['center_z'], 'mount_y': plug['mount_y'],
                    'length': plug['length'], 'socket_r': plug['socket_r'],
                    'socket_depth': plug['socket_depth']},
    'old_alignment': {
        'pca_axis': [round(float(v), 6) for v in z_axis],
        'rotation_about_z_deg': round(math.degrees(rotate_z), 3),
        'scale_factor': round(float(scale_factor), 6),
        'note': ('The old library mesh was rotated/scaled/translated for display '
                 'comparison only; the library file and its hash were not modified.')},
    'old_span_raw': float(request['old_span_raw']),
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
