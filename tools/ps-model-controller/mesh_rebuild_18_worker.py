"""Blender worker for the review-only mountain-bike rear air-shock candidate (item 18).

Shape basis: the original clean/rust photograph shows one black DNM-style
mountain-bike rear air shock at a three-quarter angle.  The user six-view sheet
is an uncalibrated visual-match atlas of a closely related DNM shock; it is used
for part topology and appearance only, never as measured geometry.  The
candidate is a parametric, part-based rebuild: air can, top cap, rebound knob,
three-position blue lever, upper/lower clevis eyelets with real bores and steel
bushings, seal head, sag O-ring and damper shaft.  It never replaces the library
mesh, stays manual-only and never enters automatic placement.
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
    materials['can'] = textured_material('18 can black anodised + decal', 'can', .045)
    materials['cap'] = textured_material('18 top cap black anodised + print', 'cap', .035)
    materials['black'] = textured_material('18 black anodised metal', 'black', .045)
    materials['steel'] = textured_material('18 bushing steel', 'steel', .035)
    materials['lever'] = textured_material('18 blue lever top print', 'lever', .025)
    materials['red'] = uniform_material('18 red anodised', profile['red']['base_rgb'],
                                        profile['red']['metallic'],
                                        profile['red']['base_roughness'])
    materials['blue'] = uniform_material('18 blue anodised', profile['blue']['base_rgb'],
                                         profile['blue']['metallic'],
                                         profile['blue']['base_roughness'])
    materials['rubber'] = uniform_material('18 blue rubber O-ring',
                                           profile['blue_rubber']['base_rgb'],
                                           profile['blue_rubber']['metallic'],
                                           profile['blue_rubber']['base_roughness'])
    materials['bore'] = uniform_material('18 dark bore steel', [42, 43, 46], 1.0, .48)
    materials['clay'] = uniform_material('18 clay grey', [148, 148, 148], 0.0, .62)


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


def disc_uv(obj, cx, cy, radius):
    ensure_uv(obj)
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv[loop_index].uv = ((co.x - cx) / (2 * radius) + .5,
                                 (co.y - cy) / (2 * radius) + .5)


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
    if mode == 'xside':
        for poly in obj.data.polygons:
            poly.use_smooth = abs(poly.normal.x) < .80
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


def tube_x(name, center, outer_r, inner_r, half_len, segments=64):
    cx, cy, cz = center
    verts = []
    rings = {}
    for x, key in ((-half_len, 'lo'), (half_len, 'hi')):
        for r, rkey in ((outer_r, 'outer'), (inner_r, 'inner')):
            ids = []
            for s in range(segments):
                a = TAU * s / segments
                ids.append(len(verts))
                verts.append((cx + x, cy + r * math.cos(a), cz + r * math.sin(a)))
            rings[(key, rkey)] = ids
    lo_o, hi_o = rings[('lo', 'outer')], rings[('hi', 'outer')]
    lo_i, hi_i = rings[('lo', 'inner')], rings[('hi', 'inner')]
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        faces.append((lo_o[s], hi_o[s], hi_o[s2], lo_o[s2]))
        faces.append((lo_i[s], lo_i[s2], hi_i[s2], hi_i[s]))
        faces.append((lo_o[s], lo_i[s], lo_i[s2], lo_o[s2]))
        faces.append((hi_o[s], hi_o[s2], hi_i[s2], hi_i[s]))
    return mesh_object(name, verts, faces)


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


def plate_with_hole(name, x_center, half_x, zc, outer_r, hole_r, y_half,
                    bottom_z=None, top_z=None, upper=True, segments=72):
    pts = []
    if upper:
        for k in range(segments + 1):
            th = math.pi * k / segments
            pts.append((outer_r * math.cos(th), zc + outer_r * math.sin(th)))
        pts.append((-y_half, bottom_z))
        pts.append((y_half, bottom_z))
    else:
        for k in range(segments + 1):
            th = math.pi + math.pi * k / segments
            pts.append((outer_r * math.cos(th), zc + outer_r * math.sin(th)))
        pts.append((y_half, top_z))
        pts.append((-y_half, top_z))
    cleaned = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - cleaned[-1][0]) > 1e-9 or abs(p[1] - cleaned[-1][1]) > 1e-9:
            cleaned.append(p)
    if abs(cleaned[0][0] - cleaned[-1][0]) < 1e-9 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-9:
        cleaned.pop()
    pts = cleaned
    n = len(pts)
    verts = [(x_center - half_x, y, z) for y, z in pts] + \
            [(x_center + half_x, y, z) for y, z in pts]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    obj = mesh_object(name, verts, faces)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=72, radius=hole_r, depth=4 * half_x,
        location=(x_center, 0.0, zc), rotation=(0.0, math.pi / 2, 0.0))
    cutter = bpy.context.object
    cutter.name = name + ' bore cutter'
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mod = obj.modifiers.new('Through bore', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    finally:
        bpy.data.objects.remove(cutter, do_unlink=True)
    apply_bevel(obj, 0.00045, 2)
    return obj


def apply_transform(obj, matrix):
    obj.data.transform(matrix)
    obj.data.update()
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
# Model
# ---------------------------------------------------------------------------
def air_can_profile(cfg):
    z0, z1, r = cfg['z0'], cfg['z1'], cfg['r']
    neck = cfg['bottom_neck_r']
    gd = cfg['groove_depth']
    points = [(0.0, z0), (neck, z0), (neck, z0 + 0.0015),
              (r * .985, z0 + 0.0028), (r, z0 + 0.0045)]
    g0, g1, g2 = cfg['grooves_z']
    for z in (g0, g1, g2):
        points += [(r, z - 0.0016), (r - gd, z), (r, z + 0.0016)]
    points += [(r, z1 - 0.0040), (r * .988, z1 - 0.0018),
               (r * .92, z1), (0.0, z1)]
    return points


def cap_profile(cfg):
    z0, z1 = cfg['z0'], cfg['z1']
    fr = cfg['flange_r']
    br = cfg['body_r']
    tr = cfg['top_r']
    f0, f1 = cfg['flange_z0'], cfg['flange_z1']
    return [(0.0, z0), (fr * .93, z0), (fr, z0 + 0.0010), (fr, f1),
            (br, f1 + 0.0015), (br, z1 - 0.0020), (tr, z1 - 0.0006),
            (tr, z1), (0.0, z1)]


def seal_profile(cfg):
    z0, z1, r = cfg['z0'], cfg['z1'], cfg['r']
    tr = cfg['top_r']
    return [(0.0, z0), (r * .74, z0), (r, z0 + 0.0018), (r, z0 + 0.0040),
            (r * .93, z0 + 0.0060), (r, z0 + 0.0082), (r, z1 - 0.0020),
            (tr, z1), (0.0, z1)]


def shaft_profile(cfg):
    z0, z1, r = cfg['z0'], cfg['z1'], cfg['r']
    return [(0.0, z0), (r * .82, z0), (r, z0 + 0.0015),
            (r, z1 - 0.0025), (r * .96, z1 - 0.0010), (r * .88, z1), (0.0, z1)]


def build():
    seg = geo['radial_segments']
    can_cfg = geo['air_can']
    can = revolve_solid('Air can', air_can_profile(can_cfg), seg)
    assign(can, 'can', uv='cyl', z0=can_cfg['z0'], z1=can_cfg['z1'],
           smooth='side', axis=(0, 0, 1))

    cap_cfg = geo['top_cap']
    cap = revolve_solid('Top cap', cap_profile(cap_cfg), seg)
    assign(cap, 'cap', uv='cyl', z0=cap_cfg['z0'], z1=cap_cfg['z1'],
           smooth='side', axis=(0, 0, 1))
    seam = geo['top_seam_ring']
    ring = torus('Top cap seam', (0, 0, seam['z']), (0, 0, 1),
                 seam['major'], seam['minor'], 128, 10)
    assign(ring, 'black', uv='box', scale=.05, smooth='all')

    yoke = geo['top_yoke']
    top_yoke = box_at('Top yoke', yoke['center'], yoke['half'])
    apply_bevel(top_yoke, yoke['bevel'], 3)
    assign(top_yoke, 'black', uv='box', scale=.06, smooth='flat')

    eye = geo['top_eyelet']
    for side, xc in (('L', -eye['plate_x']), ('R', eye['plate_x'])):
        plate = plate_with_hole('Top eyelet plate ' + side, xc, eye['plate_half_x'],
                                eye['center_z'], eye['outer_r'], eye['hole_r'],
                                eye['y_half'], bottom_z=eye['bottom_z'], upper=True)
        assign(plate, 'black', uv='box', scale=.05, smooth='xside')
    top_bush = tube_x('Top eyelet bushing', (0, 0, eye['center_z']),
                      eye['bushing_outer_r'], eye['bushing_inner_r'],
                      eye['bushing_half_len'], 72)
    assign(top_bush, 'steel', uv='box', scale=.03, smooth='all')

    byoke = geo['bottom_yoke']
    bottom_yoke = box_at('Bottom yoke', byoke['center'], byoke['half'])
    apply_bevel(bottom_yoke, byoke['bevel'], 3)
    assign(bottom_yoke, 'black', uv='box', scale=.06, smooth='flat')

    beye = geo['bottom_eyelet']
    for side, xc in (('L', -beye['plate_x']), ('R', beye['plate_x'])):
        plate = plate_with_hole('Bottom eyelet plate ' + side, xc,
                                beye['plate_half_x'], beye['center_z'],
                                beye['outer_r'], beye['hole_r'], beye['y_half'],
                                top_z=beye['top_z'], upper=False)
        assign(plate, 'black', uv='box', scale=.045, smooth='xside')
    bottom_bush = tube_x('Bottom eyelet bushing', (0, 0, beye['center_z']),
                         beye['bushing_outer_r'], beye['bushing_inner_r'],
                         beye['bushing_half_len'], 64)
    assign(bottom_bush, 'steel', uv='box', scale=.025, smooth='all')

    shaft_cfg = geo['shaft']
    shaft = revolve_solid('Damper shaft', shaft_profile(shaft_cfg), 96)
    assign(shaft, 'black', uv='cyl', z0=shaft_cfg['z0'], z1=shaft_cfg['z1'],
           smooth='side', axis=(0, 0, 1))
    collar_cfg = geo['shaft_collar']
    collar = revolve_solid('Shaft collar', [
        (0.0, collar_cfg['z0']), (collar_cfg['r'] * .88, collar_cfg['z0']),
        (collar_cfg['r'], collar_cfg['z0'] + 0.0010),
        (collar_cfg['r'], collar_cfg['z1'] - 0.0010),
        (collar_cfg['r'] * .88, collar_cfg['z1']), (0.0, collar_cfg['z1'])], 64)
    assign(collar, 'black', uv='box', scale=.03, smooth='side', axis=(0, 0, 1))

    seal_cfg = geo['seal_head']
    seal = revolve_solid('Seal head', seal_profile(seal_cfg), 96)
    assign(seal, 'black', uv='cyl', z0=seal_cfg['z0'], z1=seal_cfg['z1'],
           smooth='side', axis=(0, 0, 1))
    for i, rib in enumerate(geo['seal_ribs'], start=1):
        rib_obj = torus(f'Seal rib {i}', (0, 0, rib['z']), (0, 0, 1),
                        rib['major'], rib['minor'], 96, 8)
        assign(rib_obj, 'black', uv='box', scale=.035, smooth='all')

    sag = geo['sag_ring']
    sag_ring = torus('Sag indicator ring', (0, 0, sag['z']), (0, 0, 1),
                     sag['major'], sag['minor'], 96, 14)
    assign(sag_ring, 'rubber', uv='box', scale=.025, smooth='all')
    screw = geo['sag_screw']
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=screw['r'], depth=screw['half_len'] * 2,
        location=screw['center'], rotation=(math.pi / 2, 0, 0))
    sag_screw = bpy.context.object
    sag_screw.name = 'Sag ring screw'
    assign(sag_screw, 'blue', uv='box', scale=.015, smooth='all')
    apply_bevel(sag_screw, 0.0003, 2)

    rk = geo['red_knob']
    boss = cylinder_z('Red knob boss', rk['center'], rk['boss_r'],
                      rk['boss_z0'], rk['boss_z1'], 64)
    assign(boss, 'black', uv='box', scale=.03, smooth='side', axis=(0, 0, 1))
    knob = knurled_cylinder_z('Red rebound knob', rk['center'], rk['r'],
                              rk['z0'], rk['z1'], rk['teeth'], rk['tooth_depth'])
    assign(knob, 'red', uv='box', scale=.03, smooth='flat')
    top_disc = cylinder_z('Red knob top', rk['center'], rk['r'] * .93,
                          rk['z1'] - 0.0002, rk['z1'] + 0.0008, 64)
    assign(top_disc, 'red', uv='box', scale=.03, smooth='flat')
    groove = torus('Red knob top groove', (rk['center'][0], rk['center'][1], rk['z1'] + 0.0008),
                   (0, 0, 1), rk['r'] * .62, 0.00025, 64, 6)
    assign(groove, 'red', uv='box', scale=.02, smooth='all')

    bl = geo['blue_lever']
    collar = knurled_cylinder_z('Blue lever red collar', bl['center'],
                                bl['collar_r'], bl['collar_z0'], bl['collar_z1'],
                                bl['teeth'], bl['tooth_depth'])
    assign(collar, 'red', uv='box', scale=.03, smooth='flat')
    head = cylinder_z('Blue lever head', bl['center'], bl['head_r'],
                      bl['head_z0'], bl['head_z1'], 64)
    apply_bevel(head, 0.0005, 2)
    assign(head, 'blue', uv='box', scale=.02, smooth='side', axis=(0, 0, 1))
    cy = bl['center'][1]
    hz = (bl['arm_z_top'] + bl['arm_z_bottom']) / 2
    horizontal = box_at('Blue lever arm horizontal',
                        (bl['center'][0], (cy + bl['arm_y1']) / 2,
                         bl['arm_z_top'] - bl['arm_thickness'] / 2),
                        (bl['arm_x_half'], (bl['arm_y1'] - cy) / 2,
                         bl['arm_thickness'] / 2))
    apply_bevel(horizontal, 0.0006, 2)
    assign(horizontal, 'blue', uv='box', scale=.02, smooth='flat')
    vertical = box_at('Blue lever arm vertical',
                      (bl['center'][0], bl['arm_y1'], hz),
                      (bl['arm_x_half'], bl['arm_thickness'] / 2,
                       (bl['arm_z_top'] - bl['arm_z_bottom']) / 2))
    apply_bevel(vertical, 0.0008, 3)
    assign(vertical, 'blue', uv='box', scale=.02, smooth='flat')
    mark = cylinder_z('Blue lever top mark', bl['center'],
                      bl['mark_r'], bl['head_z1'] + 0.0002,
                      bl['head_z1'] + 0.0008, 64)
    assign(mark, 'lever', uv='none', smooth='flat')
    disc_uv(mark, bl['center'][0], bl['center'][1], bl['mark_r'])


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def camera(scene, direction, center, span, scale=1.27):
    direction = Vector(direction).normalized()
    cam = legacy.make_camera(scene, Vector(center) + direction * span * 5, center,
                             ortho_scale=span * scale)
    if abs(direction.z) > .999:
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.rotation_quaternion @ Quaternion((0, 0, 1), math.pi / 2)
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
    # The blend is re-opened before rendering, so the pre-save material
    # references are invalid.  Look the clay material up by name (or build it).
    clay = bpy.data.materials.get('18 clay grey')
    if clay is None:
        clay = uniform_material('18 clay grey', [148, 148, 148], 0.0, .62)
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
eye = geo['top_eyelet']
beye = geo['bottom_eyelet']
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
Z_SHIFT = -low.z
for obj in objects:
    obj.data.transform(obj.matrix_basis)
    obj.matrix_basis = Matrix.Identity(4)
    apply_transform(obj, Matrix.Translation((0, 0, Z_SHIFT)))
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)

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
directions = {'front': (1, 0, .16), 'back': (-1, 0, .16),
              'left': (0, -1, .18), 'right': (0, 1, .18),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else render_cfg['six_view_resolution']
samples = render_cfg['draft_samples'] if request['draft'] else render_cfg['samples']
span_len = span
span_front = max(high.y - low.y, high.z - low.z)
span_cap = max(high.x - low.x, high.y - low.y)
for name, direction in directions.items():
    if name in ('top', 'bottom'):
        use_span = span_cap * 1.18
    elif name in ('front', 'back'):
        use_span = span_front
    else:
        use_span = span_len
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

# Renders used only for silhouette / opening checks.
hide_all(objects)
for keep in objects:
    if keep.name.startswith('Top eyelet'):
        keep.hide_render = False
cam = camera(scene, directions['front'], (0, 0, eye['center_z'] + Z_SHIFT),
             eye['outer_r'] * 2.4)
draw(scene, render_dir / 'top-eyelet-only.png', resolution=512,
     samples=max(8, samples // 4))
bpy.data.objects.remove(cam, do_unlink=True)
hide_all(objects)
for keep in objects:
    if keep.name.startswith('Bottom eyelet'):
        keep.hide_render = False
cam = camera(scene, directions['front'], (0, 0, beye['center_z'] + Z_SHIFT),
             beye['outer_r'] * 2.4)
draw(scene, render_dir / 'bottom-eyelet-only.png', resolution=512,
     samples=max(8, samples // 4))
bpy.data.objects.remove(cam, do_unlink=True)
hide_all(objects)
for keep in objects:
    if keep.name.startswith('Top eyelet') or keep.name.startswith('Bottom eyelet'):
        keep.hide_render = False
cam = camera(scene, directions['front'], (0, 0, 0.096 + Z_SHIFT), span)
draw(scene, render_dir / 'eyelet-only.png', resolution=512,
     samples=max(8, samples // 4))
bpy.data.objects.remove(cam, do_unlink=True)
hide_all(objects, False)

cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .14)
draw(scene, render_dir / 'new-material-close.png',
     resolution=1024 if not request['draft'] else 640,
     samples=32 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
# Alternate key light, same close-up camera: needed even in draft for the
# metal-response check.
set_key_light(scene, tuple(render_cfg['key_light_b']))
cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .14)
draw(scene, render_dir / 'new-surface-alternate.png',
     resolution=1024 if not request['draft'] else 640,
     samples=32 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
set_key_light(scene, tuple(render_cfg['key_light_a']))

if not request['draft']:
    # Clay geometry views.
    saved = clay_override(objects)
    cam = camera(scene, directions['front'], center, span_front)
    draw(scene, render_dir / 'new-clay-front.png', resolution=render_cfg['clay_resolution'],
         samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .14)
    draw(scene, render_dir / 'new-clay-close.png', resolution=1024, samples=20)
    bpy.data.objects.remove(cam, do_unlink=True)
    restore_materials(objects, saved)

    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .14), center, span)
        draw(scene, render_dir / f'turn-{i:02d}.png',
             resolution=render_cfg['turntable_resolution'], samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    bl = geo['blue_lever']
    detail_cameras = (
        ('front', (0, 0, center.z), (1, 0, 0), .24),
        ('back', (0, 0, center.z), (-1, 0, 0), .24),
        ('side', (0, 0, center.z), (0, -1, .05), .24),
        ('top', (0, 0, center.z), (0, 0, 1), .075),
        ('bottom', (0, 0, center.z), (0, 0, -1), .075),
        ('top-eyelet', (0, 0, eye['center_z'] + Z_SHIFT), (0.75, -0.6, .25), .055),
        ('top-eyelet-front', (0, 0, eye['center_z'] + Z_SHIFT), (1, 0, .02), .043),
        ('top-eyelet-top', (0, 0, eye['center_z'] + Z_SHIFT), (0, 0, 1), .045),
        ('bottom-eyelet', (0, 0, beye['center_z'] + Z_SHIFT), (0.75, -0.6, .22), .045),
        ('bottom-eyelet-front', (0, 0, beye['center_z'] + Z_SHIFT), (1, 0, 0), .038),
        ('adjusters', (0, 0, 0.174 + Z_SHIFT), (0.72, -0.66, .34), .062),
        ('blue-lever', (0, bl['center'][1], 0.169 + Z_SHIFT), (0.50, -0.82, .26), .050),
        ('red-knob', (0, -0.0152, 0.174 + Z_SHIFT), (0.50, -0.82, .34), .045),
        ('cap', (0, 0, 0.166 + Z_SHIFT), (0.70, -0.68, .30), .060),
        ('label-front', (0, 0, 0.112 + Z_SHIFT), (1, 0, 0), .110),
        ('label-back', (0, 0, 0.112 + Z_SHIFT), (-1, 0, 0), .110),
        ('sag-ring', (0, 0, 0.055 + Z_SHIFT), (0.70, -0.68, .26), .058),
        ('shaft-eyelet', (0, 0, 0.020 + Z_SHIFT), (0.80, -0.58, .18), .060),
        ('can-grooves', (0, 0, 0.080 + Z_SHIFT), (0.60, -0.78, .20), .048),
        ('surface', (0, 0, 0.120 + Z_SHIFT), (0.88, -0.46, .30), .036),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png',
             resolution=render_cfg['detail_resolution'], samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('top-eyelet-only-3q',
         lambda n: n.startswith('Top eyelet'), (0.75, -0.6, .25),
         (0, 0, eye['center_z'] + Z_SHIFT), .050),
        ('bottom-eyelet-only-3q',
         lambda n: n.startswith('Bottom eyelet'), (0.75, -0.6, .22),
         (0, 0, beye['center_z'] + Z_SHIFT), .045),
        ('adjusters-only',
         lambda n: ('Red ' in n or 'Blue lever' in n), (0.65, -0.78, .32),
         (0, 0, 0.174 + Z_SHIFT), .060),
        ('can-label-only',
         lambda n: n == 'Air can', (1, 0, 0), (0, 0, 0.112 + Z_SHIFT), .115),
        ('seal-shaft-only',
         lambda n: ('Seal' in n or 'Damper shaft' in n or 'Shaft collar' in n
                    or 'Sag' in n), (0.75, -0.62, .22),
         (0, 0, 0.040 + Z_SHIFT), .080),
        ('top-view-all', lambda n: True, (0, 0, 1), (0, 0, center.z), .075),
    )
    for label, predicate, direction, target, scale in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'{label}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    # Alternate key light on the same camera: highlight must move, not be baked.
    set_key_light(scene, tuple(render_cfg['key_light_b']))
    cam = camera(scene, directions['front'], center, span_front)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .14)
    draw(scene, render_dir / 'new-surface-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    set_key_light(scene, tuple(render_cfg['key_light_a']))

# ---------------------------------------------------------------------------
# Old mesh: display alignment, then six views and closeup
# ---------------------------------------------------------------------------
hide_all(objects)
old = legacy.load_mesh(request['original_mesh'])
bpy.context.view_layer.update()
old_points = np.array([tuple(p) for p in legacy.world_vertices(old)], dtype=np.float64)
old_center = old_points.mean(axis=0)
cov = np.cov((old_points - old_center).T)
eigvals, eigvecs = np.linalg.eigh(cov)
major = eigvecs[:, int(np.argmax(eigvals))]
major_yz = Vector((0.0, float(major[1]), float(major[2]))).normalized()
angle = math.atan2(major_yz.y, major_yz.z)
old.data.transform(Matrix.Rotation(angle, 4, 'X'))
old.data.update()
bpy.context.view_layer.update()
points = np.array([tuple(p) for p in legacy.world_vertices(old)], dtype=np.float64)
zmin, zmax = points[:, 2].min(), points[:, 2].max()
zspan = max(1e-9, zmax - zmin)
top = points[points[:, 2] > zmax - .18 * zspan]
bottom = points[points[:, 2] < zmin + .18 * zspan]
top_radial = float(np.sqrt((top[:, 0] - top[:, 0].mean()) ** 2 +
                           (top[:, 1] - top[:, 1].mean()) ** 2).max())
bottom_radial = float(np.sqrt((bottom[:, 0] - bottom[:, 0].mean()) ** 2 +
                              (bottom[:, 1] - bottom[:, 1].mean()) ** 2).max())
flipped = False
if top_radial < bottom_radial:
    old.data.transform(Matrix.Rotation(math.pi, 4, 'X'))
    old.data.update()
    flipped = True
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old_span_front = max(oldhigh.y - oldlow.y, oldhigh.z - oldlow.z)
old_span_raw = max(1e-9, oldspan)
scale_factor = span / old_span_raw
old.data.transform(Matrix.Scale(scale_factor, 4))
old.data.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old.data.transform(Matrix.Translation((center.x - oldcenter.x,
                                       center.y - oldcenter.y,
                                       low.z - oldlow.z)))
old.data.update()
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old_span_front = max(oldhigh.y - oldlow.y, oldhigh.z - oldlow.z)
for name, direction in directions.items():
    if name in ('top', 'bottom'):
        use_span = span_cap * 1.18
    elif name in ('front', 'back'):
        use_span = old_span_front
    else:
        use_span = oldspan
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .14)
draw(scene, render_dir / 'old-material-close.png', resolution=1024, samples=32)
bpy.data.objects.remove(cam, do_unlink=True)

# ---------------------------------------------------------------------------
# Source-photo framing diagnostics on the new candidate
# ---------------------------------------------------------------------------
result_diag = {}
old.hide_render = True
for obj in objects:
    obj.hide_render = False
source = request['source_camera']
source_bbox = request.get('source_alpha_bbox') or [0, 0, 511, 511]
# The factor is relative to the raw old span recorded before display scaling.
factor = span / old_span_raw


def source_camera(roll_deg):
    cam = legacy.make_camera(scene, (1.9 * factor, 0, center.z),
                             (0, 0, center.z), fov=source['fov_deg'])
    cam.data.shift_x = source.get('shift_x', 0)
    cam.data.shift_y = source.get('shift_y', 0)
    if abs(roll_deg) > 1e-9:
        base = cam.rotation_euler.to_quaternion()
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = base @ Quaternion((0, 0, 1), math.radians(roll_deg))
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


for roll in (0.0, -38.6, 38.6):
    cam = source_camera(roll)
    fit = fit_source_camera(cam)
    name = f'new-source-roll{roll:+.1f}.png'
    draw(scene, render_dir / name, resolution=512, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
    result_diag[f'{roll:+.1f}'] = fit
# Keep the strict camera image under the canonical name.
import shutil
shutil.copyfile(render_dir / 'new-source-roll+0.0.png', render_dir / 'new-source.png')
bpy.data.objects.remove(old, do_unlink=True)


data = {
    'parts': parts,
    'bounds': [list(low), list(high)],
    'z_shift': Z_SHIFT,
    'top_eyelet_center_z': eye['center_z'] + Z_SHIFT,
    'bottom_eyelet_center_z': beye['center_z'] + Z_SHIFT,
    'span': span,
    'parts_count': len(parts),
    'polygons': sum(len(o.data.polygons) for o in objects),
    'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                     for p in o.data.polygons),
    'texture_size': texture_cfg['can_size'],
    'materials': ['can', 'cap', 'black', 'steel', 'lever', 'red', 'blue',
                  'rubber', 'bore'],
    'render_device': render_device,
    'self_intersecting_components': {k: v for k, v in intersections.items() if v},
    'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
    'geometry_basis': ('original photo proportions; six-view sheet is an uncalibrated '
                       'visual-match reference'),
    'render_samples': samples,
    'view_directions': {k: list(v) for k, v in directions.items()},
    'view_note': ('Camera at +X is the source-photo view; the upper/lower eyelet '
                  'bores are along X and appear as true circular openings in the '
                  'front view. The six-view bottom panel is internally inconsistent '
                  'about the lower eyelet axis and is not used as a strict projection.'),
    'source_camera_note': ('source-photo framing distance scaled by new/old span, '
                           'then fitted to the source-alpha height; diagnostic only'),
    'source_camera_factor': factor,
    'source_camera_fits': result_diag,
    'old_alignment': {
        'rotation_about_x_deg': round(math.degrees(-angle), 3),
        'flipped_180_x': flipped,
        'scale_factor': round(float(scale_factor), 6),
        'note': ('The old library mesh was rotated/scaled/translated for display '
                 'comparison only; the library file and its hash were not modified.')},
    'source_is_diagonal': True,
    'source_major_axis_below_horizontal_deg': 51.43,
    'source_roll_diagnostic_deg': [-38.6, 38.6],
    'isolated_checks': {'top_eyelet_bore_expected': 1,
                        'bottom_eyelet_bore_expected': 1,
                        'eyelet_pair_expected': 2,
                        'clevis_slot_is_open_not_enclosed': True},
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
