"""Blender worker for the review-only item-39 oil-receiver candidate.

Shape basis: the clean/rust photograph of the same upright teal HPEOK PHO-10
oil receiver plus an uncalibrated six-view visual-match sheet.  The candidate is
a part-based parametric rebuild: torispherical body, two weld seams, two front
sight/plug bosses, the visible HPEOK label as a curved decal, a top stainless
valve stack with brass plug, quick-connect and black ribbed knob, a matching
bottom drain stack and the small angled service fitting.  It never replaces the
library mesh, is manual-only and never enters automatic placement.
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
objects = []
material_key = {}
TAU = math.tau
TOP = 0.0
BOTTOM = 0.0


# ---------------------------------------------------------------------------
# Materials and UVs
# ---------------------------------------------------------------------------
def texture_node(nodes, key, noncolor=False, extend='REPEAT'):
    node = nodes.new('ShaderNodeTexImage')
    node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
    node.image.pack()
    if noncolor:
        node.image.colorspace_settings.name = 'Non-Color'
    node.interpolation = 'Linear'
    node.extension = extend
    return node


def textured_material(name, base_key, mr_key, normal_key, normal_strength):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    color = texture_node(nodes, base_key)
    metal = texture_node(nodes, mr_key, True)
    normal = texture_node(nodes, normal_key, True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(metal.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = normal_strength
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    return material


def plain_material(name, color, roughness, metallic, coat=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (*color, 1)
    principled.inputs['Roughness'].default_value = roughness
    principled.inputs['Metallic'].default_value = metallic
    if coat:
        principled.inputs['Coat Weight'].default_value = coat
        principled.inputs['Coat Roughness'].default_value = .08
    return material


MATERIALS = {}


def make_materials():
    return {
        'body': textured_material('39 teal painted body / 青绿漆筒体',
                                  'body_base', 'body_mr', 'body_normal', .28),
        'paint': textured_material('39 teal painted fittings / 青绿漆小件',
                                   'paint_base', 'paint_mr', 'paint_normal', .16),
        'label': textured_material('39 HPEOK label paper / 标签纸',
                                   'label_base', 'label_mr', 'label_normal', .10),
        'steel': textured_material('39 stainless steel / 不锈钢',
                                   'metal_base', 'metal_mr', 'metal_normal', .22),
        'brass': textured_material('39 brass / 黄铜',
                                   'brass_base', 'brass_mr', 'brass_normal', .18),
        'black': textured_material('39 black engineering plastic / 黑色工程塑料',
                                   'black_base', 'black_mr', 'black_normal', .20),
        'red': textured_material('39 red plastic / 红色塑料',
                                 'red_base', 'red_mr', 'red_normal', .12),
        'glass': plain_material('39 dark sight glass / 深色视镜',
                                (0.012, 0.018, 0.017), .14, 0, .65),
        'bronze': plain_material('39 bronze service cap / 青铜服务帽',
                                 (0.23, 0.10, 0.035), .48, 1.0),
        'slot': plain_material('39 dark valve slot / 阀芯槽',
                               (0.025, 0.025, 0.022), .52, .55),
    }


def box_uv(obj, scale=0.50):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='UVMap')
    uv = mesh.uv_layers.active.data
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
            uv[loop_index].uv = (u, v)


def tile_uv(tile, u, v):
    col, top_row = tile % 2, tile // 2
    return (col * .5 + .010 + (u % 1.0) * .480,
            (1 - top_row) * .5 + .010 + (v % 1.0) * .480)


def remap_uv(obj, tile):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    for loop in obj.data.uv_layers.active.data:
        u, v = loop.uv
        loop.uv = tile_uv(tile, u, v)


def cyl_uv_z(obj, scale=1.0):
    """Cylindrical UV on the full body texture; u=0.5 faces +X."""
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    zs = [v.co.z for v in mesh.vertices]
    z0, z1 = min(zs), max(zs)
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
            uv[loop_index].uv = ((0.5 + (ca + delta) / TAU) % 1.0,
                                 (co.z - z0) / max(1e-9, z1 - z0))


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


def assign(obj, key, *, smooth='all', axis=None, scale=0.40, uv='box', tile=None,
           label_uv=False):
    obj.data.materials.clear()
    obj.data.materials.append(MATERIALS[key])
    material_key[obj.name] = key
    if label_uv:
        pass
    elif uv == 'z':
        cyl_uv_z(obj, scale)
    else:
        box_uv(obj, scale)
        if tile is not None:
            remap_uv(obj, tile)
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


def bevel(obj, width, segments=2):
    if not width:
        return obj
    bpy.ops.object.select_all(action='DESELECT')
    modifier = obj.modifiers.new('Manufactured edge radius', 'BEVEL')
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = 'ANGLE'
    modifier.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def basis(axis):
    n = Vector(axis).normalized()
    helper = Vector((0, 0, 1)) if abs(n.z) < .9 else Vector((0, 1, 0))
    u = n.cross(helper).normalized()
    v = n.cross(u).normalized()
    return u, v, n


def revolve_axis(name, profile, center, axis, segments=96):
    u, v, n = basis(axis)
    center = Vector(center)
    verts, rings = [], []
    for radius, height in profile:
        if radius <= 1e-9:
            rings.append([len(verts)])
            verts.append(tuple(center + n * height))
        else:
            ids = []
            for s in range(segments):
                angle = TAU * s / segments
                ids.append(len(verts))
                verts.append(tuple(center + n * height +
                                   (u * math.cos(angle) + v * math.sin(angle)) * radius))
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


def torch_ring(name, center, normal, major, minor, segments=96, minor_segments=10):
    u, v, n = basis(normal)
    center = Vector(center)
    verts, rings = [], []
    for i in range(segments):
        theta = TAU * i / segments
        radial = u * math.cos(theta) + v * math.sin(theta)
        ring = []
        for j in range(minor_segments):
            phi = TAU * j / minor_segments
            ring.append(len(verts))
            verts.append(tuple(center + radial * (major + minor * math.cos(phi)) +
                               n * (minor * math.sin(phi))))
        rings.append(ring)
    faces = []
    for i in range(segments):
        a, b = rings[i], rings[(i + 1) % segments]
        for j in range(minor_segments):
            faces.append((a[j], a[(j + 1) % minor_segments],
                          b[(j + 1) % minor_segments], b[j]))
    return mesh_object(name, verts, faces)


def torus(name, center, normal, major, minor, segments=96, minor_segments=10):
    return torch_ring(name, center, normal, major, minor, segments, minor_segments)


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


def hex_prism(name, center, axis, across_flats, length, phase=math.pi / 6, bevel_width=0.0):
    u, v, n = basis(axis)
    center = Vector(center)
    radius = (across_flats / 2) / math.cos(math.pi / 6)
    verts = []
    for h in (-length / 2, length / 2):
        for s in range(6):
            angle = TAU * s / 6 + phase
            verts.append(tuple(center + n * h +
                               (u * math.cos(angle) + v * math.sin(angle)) * radius))
    verts.append(tuple(center - n * (length / 2)))
    verts.append(tuple(center + n * (length / 2)))
    bottom_center, top_center = 12, 13
    faces = []
    for s in range(6):
        faces.append((s, (s + 1) % 6, 6 + (s + 1) % 6, 6 + s))
    for s in range(6):
        faces.append((bottom_center, (s + 1) % 6, s))
        faces.append((top_center, 6 + s, 6 + (s + 1) % 6))
    obj = mesh_object(name, verts, faces)
    if bevel_width:
        bevel(obj, bevel_width, 2)
    return obj


def fluted_knob(name, center, axis, y0, y1, radius, flutes=16, segments=96):
    u, v, n = basis(axis)
    center = Vector(center)
    cuts = [(y0, .86), (y0 + .0015, 1.0), (y1 - .0015, 1.0), (y1, .86)]
    verts, rings = [], []
    for height, factor in cuts:
        ring = []
        for s in range(segments):
            angle = TAU * s / segments
            ripple = 1.0 + .075 * math.cos(flutes * angle) if factor > .9 else 1.0
            r = radius * factor * ripple
            ring.append(len(verts))
            verts.append(tuple(center + n * height +
                               (u * math.cos(angle) + v * math.sin(angle)) * r))
        rings.append(ring)
    faces = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        for s in range(segments):
            faces.append((a[s], a[(s + 1) % segments],
                          b[(s + 1) % segments], b[s]))
    v0 = len(verts)
    verts.append(tuple(center + n * y0))
    v1 = len(verts)
    verts.append(tuple(center + n * y1))
    for s in range(segments):
        faces.append((v0, (s + 1) % segments, s))
        faces.append((v1, rings[-1][s], rings[-1][(s + 1) % segments]))
    return mesh_object(name, verts, faces)


def curved_label(name, radius, theta0, theta1, z0, z1, thickness, nu=56, nv=40):
    verts, uvs = [], []
    layers = []
    for side in (1, -1):
        r = radius + side * thickness / 2
        layer = []
        for iv in range(nv + 1):
            v = iv / nv
            z = z0 + (z1 - z0) * v
            row = []
            for iu in range(nu + 1):
                u = iu / nu
                theta = theta0 + (theta1 - theta0) * u
                row.append(len(verts))
                verts.append((r * math.cos(theta), r * math.sin(theta), z))
                uvs.append((u, v))
            layer.append(row)
        layers.append(layer)
    outer, inner = layers
    faces = []
    for iv in range(nv):
        for iu in range(nu):
            a = outer[iv][iu]
            b = outer[iv][iu + 1]
            c = outer[iv + 1][iu + 1]
            d = outer[iv + 1][iu]
            faces.append((a, b, c, d))
            a = inner[iv][iu]
            b = inner[iv][iu + 1]
            c = inner[iv + 1][iu + 1]
            d = inner[iv + 1][iu]
            faces.append((a, d, c, b))
    for iu in range(nu):
        a = outer[0][iu]
        b = outer[0][iu + 1]
        c = inner[0][iu + 1]
        d = inner[0][iu]
        faces.append((a, b, c, d))
        a = outer[nv][iu]
        b = outer[nv][iu + 1]
        c = inner[nv][iu + 1]
        d = inner[nv][iu]
        faces.append((a, d, c, b))
    for iv in range(nv):
        a = outer[iv][0]
        b = outer[iv + 1][0]
        c = inner[iv + 1][0]
        d = inner[iv][0]
        faces.append((a, b, c, d))
        a = outer[iv][nu]
        b = outer[iv + 1][nu]
        c = inner[iv + 1][nu]
        d = inner[iv][nu]
        faces.append((a, d, c, b))
    obj = mesh_object(name, verts, faces)
    mesh = obj.data
    mesh.uv_layers.new(name='UVMap')
    # Only the outer surface carries meaningful UVs; side/inner faces reuse the
    # nearest outer coordinate so the label never samples outside its image.
    outer_uv = {vid: uvs[vid] for row in outer for vid in row}
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vid = mesh.loops[loop_index].vertex_index
            u, v = outer_uv.get(vid, (0.5, 0.5))
            mesh.uv_layers.active.data[loop_index].uv = (u, v)
    return obj


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


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


def non_manifold_edges(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    count = sum(1 for edge in bm.edges if not edge.is_manifold)
    bm.free()
    return count


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_body():
    body_cfg = config['body']
    body = revolve_axis('Body', body_cfg['profile'], (0, 0, 0), (0, 0, 1),
                        config['radial_segments'])
    assign(body, 'body', smooth='all', uv='z')
    for seam in body_cfg['seams']:
        ring = torus(seam['name'], (0, 0, seam['z']), (0, 0, 1),
                     seam['major'], seam['minor'], 128, 10)
        assign(ring, 'paint', smooth='all', scale=.09, uv='box', tile=1)


def build_ports():
    port_cfg = config['side_ports']['boss']
    for label, z in zip(('Upper', 'Lower'), config['side_ports']['centers_z_m']):
        prefix = f'Port {label}'
        center = (port_cfg['center_radius_m'], 0.0, z)
        boss = hex_prism(f'{prefix} boss', center, (1, 0, 0),
                         port_cfg['across_flats_m'], port_cfg['depth_m'],
                         phase=math.pi / 6, bevel_width=.0012)
        assign(boss, 'paint', smooth='side', axis=(1, 0, 0), scale=.12,
               uv='box', tile=1)
        ring = torus(f'{prefix} outer ring', (0.0910, 0, z), (1, 0, 0),
                     port_cfg['outer_ring_major_m'], port_cfg['outer_ring_minor_m'],
                     96, 10)
        assign(ring, 'paint', smooth='all', scale=.07, uv='box', tile=1)
        face = revolve_axis(f'{prefix} face', [
            (0.0, 0.0892), (port_cfg['face_radius_m'], 0.0892),
            (port_cfg['face_radius_m'], 0.0922), (0.0, 0.0922)],
            (0, 0, z), (1, 0, 0), 96)
        assign(face, 'paint', smooth='side', axis=(1, 0, 0), scale=.06,
               uv='box', tile=1)
        inner = torus(f'{prefix} inner ring', (0.0924, 0, z), (1, 0, 0),
                      port_cfg['inner_ring_major_m'], port_cfg['inner_ring_minor_m'],
                      80, 8)
        assign(inner, 'steel', smooth='all', scale=.04, uv='box')
        glass = revolve_axis(f'{prefix} glass', [
            (0.0, 0.0918), (port_cfg['glass_radius_m'], 0.0918),
            (port_cfg['glass_radius_m'], 0.0944), (0.0, 0.0944)],
            (0, 0, z), (1, 0, 0), 64)
        assign(glass, 'glass', smooth='all', scale=.04, uv='box')
        dot = revolve_axis(f'{prefix} red dot', [
            (0.0, 0.0943), (port_cfg['red_dot_radius_m'], 0.0943),
            (port_cfg['red_dot_radius_m'], 0.0954), (0.0, 0.0954)],
            (0, 0.0033, z + 0.0033), (1, 0, 0), 32)
        assign(dot, 'red', smooth='all', scale=.02, uv='box')
        for i, angle in enumerate((90, 210, 330), start=1):
            rad = math.radians(angle)
            screw = revolve_axis(f'{prefix} screw {i}', [
                (0.0, 0.0942), (port_cfg['screw_radius_m'], 0.0942),
                (port_cfg['screw_radius_m'], 0.0953), (0.0, 0.0953)],
                (0, port_cfg['screw_ring_radius_m'] * math.cos(rad),
                 z + port_cfg['screw_ring_radius_m'] * math.sin(rad)),
                (1, 0, 0), 20)
            assign(screw, 'slot', smooth='side', axis=(1, 0, 0), scale=.02,
                   uv='box')


def build_label():
    label = config['label']
    half = math.radians(label['arc_half_angle_deg'])
    z0 = label['z_center_m'] - label['height_m'] / 2
    z1 = label['z_center_m'] + label['height_m'] / 2
    obj = curved_label('HPEOK PHO-10 label', config['body']['outer_radius_m'] +
                       label['surface_offset_m'], -half, half, z0, z1,
                       label['thickness_m'])
    assign(obj, 'label', smooth='all', label_uv=True)


def build_coupler(prefix, y0, y1, base_radius, tip_radius, grooves, center_z):
    length = y0 - y1
    profile = [
        (0.0, 0.0),
        (base_radius, 0.0),
        (base_radius, .004),
        (base_radius * .90, .006),
        (base_radius * 1.06, .009),
        (base_radius * .90, .012),
        (base_radius * 1.06, .015),
        (tip_radius, .020),
        (tip_radius, length - .002),
        (tip_radius * .88, length),
        (0.0, length),
    ]
    obj = revolve_axis(f'{prefix} coupler', profile, (0, y0, center_z), (0, -1, 0), 72)
    assign(obj, 'steel', smooth='side', axis=(0, 1, 0), scale=.08, uv='box')
    for i, slot in enumerate(grooves, start=1):
        ring = torus(f'{prefix} coupler groove {i}', (0, y0 - slot, center_z), (0, 1, 0),
                     tip_radius * 1.04, .0007, 64, 8)
        assign(ring, 'slot', smooth='all', scale=.03, uv='box')


def build_top():
    z = config['top_assembly']
    neck = revolve_axis('Top neck', [
        (0.0, z['neck']['z0']), (z['neck']['radius'], z['neck']['z0']),
        (z['neck']['radius'], z['neck']['z1']), (0.0, z['neck']['z1'])],
        (0, 0, 0), (0, 0, 1), 72)
    assign(neck, 'paint', smooth='side', axis=(0, 0, 1), scale=.08,
           uv='box', tile=1)
    ring = revolve_axis('Top brass ring', [
        (0.0, z['brass_ring']['z0']), (z['brass_ring']['radius'], z['brass_ring']['z0']),
        (z['brass_ring']['radius'], z['brass_ring']['z1']), (0.0, z['brass_ring']['z1'])],
        (0, 0, 0), (0, 0, 1), 64)
    assign(ring, 'brass', smooth='side', axis=(0, 0, 1), scale=.04, uv='box')
    nut = hex_prism('Top hex nut', (0, 0, (z['hex_nut']['z0'] + z['hex_nut']['z1']) / 2),
                    (0, 0, 1), z['hex_nut']['across_flats'],
                    z['hex_nut']['z1'] - z['hex_nut']['z0'], bevel_width=.0008)
    assign(nut, 'steel', smooth='side', axis=(0, 0, 1), scale=.06, uv='box')
    block = box_at('Top valve block', (0, 0, z['block']['center_z']), z['block']['half'])
    bevel(block, .0032, 3)
    assign(block, 'steel', smooth='all', scale=.06, uv='box')
    plug = hex_prism('Top brass plug', (0, 0, (z['brass_plug']['z0'] + z['brass_plug']['z1']) / 2),
                     (0, 0, 1), z['brass_plug']['across_flats'],
                     z['brass_plug']['z1'] - z['brass_plug']['z0'], bevel_width=.0007)
    assign(plug, 'brass', smooth='side', axis=(0, 0, 1), scale=.03, uv='box')
    slot = box_at('Top brass slot', tuple(z['slot']['center']), tuple(z['slot']['half']))
    assign(slot, 'slot', smooth='flat', scale=.02, uv='box')
    red = revolve_axis('Top red cap', [
        (0.0, z['red_cap']['center'][2] - z['red_cap']['height'] / 2),
        (z['red_cap']['radius'], z['red_cap']['center'][2] - z['red_cap']['height'] / 2),
        (z['red_cap']['radius'], z['red_cap']['center'][2] + z['red_cap']['height'] / 2),
        (0.0, z['red_cap']['center'][2] + z['red_cap']['height'] / 2)],
        (z['red_cap']['center'][0], z['red_cap']['center'][1], 0), (0, 0, 1), 32)
    assign(red, 'red', smooth='side', axis=(0, 0, 1), scale=.02, uv='box')
    build_coupler('Top', z['left_coupler']['y0'], z['left_coupler']['y1'],
                  z['left_coupler']['base_radius'], z['left_coupler']['tip_radius'],
                  (.010, .015, .021), z['block']['center_z'])
    nipple = revolve_axis('Top nipple', [
        (0.0, z['right_nipple']['y0']), (z['right_nipple']['radius'], z['right_nipple']['y0']),
        (z['right_nipple']['radius'], z['right_nipple']['y1']),
        (0.0, z['right_nipple']['y1'])], (0, 0, z['block']['center_z']), (0, 1, 0), 48)
    assign(nipple, 'steel', smooth='side', axis=(0, 1, 0), scale=.04, uv='box')
    knob = fluted_knob('Top black knob', (0, 0, z['block']['center_z']), (0, 1, 0),
                       z['knob']['y0'], z['knob']['y1'], z['knob']['radius'],
                       z['knob']['flutes'], 96)
    assign(knob, 'black', smooth='all', scale=.06, uv='box')
    cap = revolve_axis('Top knob cap', [
        (0.0, z['knob_cap']['y0']), (z['knob_cap']['radius'], z['knob_cap']['y0']),
        (z['knob_cap']['radius'], z['knob_cap']['y1']), (0.0, z['knob_cap']['y1'])],
        (0, 0, z['block']['center_z']), (0, 1, 0), 48)
    assign(cap, 'black', smooth='side', axis=(0, 1, 0), scale=.04, uv='box')


def build_bottom():
    z = config['bottom_assembly']
    neck = revolve_axis('Bottom neck', [
        (0.0, z['neck']['z1']), (z['neck']['radius'], z['neck']['z1']),
        (z['neck']['radius'], z['neck']['z0']), (0.0, z['neck']['z0'])],
        (0, 0, 0), (0, 0, 1), 72)
    assign(neck, 'paint', smooth='side', axis=(0, 0, 1), scale=.08,
           uv='box', tile=1)
    ring = revolve_axis('Bottom brass ring', [
        (0.0, z['brass_ring']['z1']), (z['brass_ring']['radius'], z['brass_ring']['z1']),
        (z['brass_ring']['radius'], z['brass_ring']['z0']), (0.0, z['brass_ring']['z0'])],
        (0, 0, 0), (0, 0, 1), 64)
    assign(ring, 'brass', smooth='side', axis=(0, 0, 1), scale=.04, uv='box')
    nut = hex_prism('Bottom hex nut', (0, 0, (z['hex_nut']['z0'] + z['hex_nut']['z1']) / 2),
                    (0, 0, 1), z['hex_nut']['across_flats'],
                    z['hex_nut']['z0'] - z['hex_nut']['z1'], bevel_width=.0008)
    assign(nut, 'steel', smooth='side', axis=(0, 0, 1), scale=.06, uv='box')
    block = box_at('Bottom valve block', (0, 0, z['block']['center_z']), z['block']['half'])
    bevel(block, .0022, 2)
    assign(block, 'steel', smooth='all', scale=.06, uv='box')
    plug = hex_prism('Bottom brass plug',
                     ((z['brass_plug']['x0'] + z['brass_plug']['x1']) / 2, 0, z['block']['center_z']),
                     (1, 0, 0), z['brass_plug']['across_flats'],
                     z['brass_plug']['x1'] - z['brass_plug']['x0'], bevel_width=.0007)
    assign(plug, 'brass', smooth='side', axis=(1, 0, 0), scale=.03, uv='box')
    build_coupler('Bottom', z['left_coupler']['y0'], z['left_coupler']['y1'],
                  z['left_coupler']['base_radius'], z['left_coupler']['tip_radius'],
                  (.009, .014, .019), z['block']['center_z'])
    nipple = revolve_axis('Bottom nipple', [
        (0.0, z['right_nipple']['y0']), (z['right_nipple']['radius'], z['right_nipple']['y0']),
        (z['right_nipple']['radius'], z['right_nipple']['y1']),
        (0.0, z['right_nipple']['y1'])], (0, 0, z['block']['center_z']), (0, 1, 0), 48)
    assign(nipple, 'steel', smooth='side', axis=(0, 1, 0), scale=.04, uv='box')
    knob = fluted_knob('Bottom black knob', (0, 0, z['block']['center_z']), (0, 1, 0),
                       z['knob']['y0'], z['knob']['y1'], z['knob']['radius'],
                       z['knob']['flutes'], 96)
    assign(knob, 'black', smooth='all', scale=.06, uv='box')
    cap = revolve_axis('Bottom knob cap', [
        (0.0, z['knob_cap']['y0']), (z['knob_cap']['radius'], z['knob_cap']['y0']),
        (z['knob_cap']['radius'], z['knob_cap']['y1']), (0.0, z['knob_cap']['y1'])],
        (0, 0, z['block']['center_z']), (0, 1, 0), 48)
    assign(cap, 'black', smooth='side', axis=(0, 1, 0), scale=.04, uv='box')


def build_side_fitting():
    z = config['side_fitting']
    angle = math.radians(z['azimuth_deg'])
    # Surface radius from the body profile at the fitting height.
    profile = config['body']['profile']
    radius = profile[0][0]
    for (r0, h0), (r1, h1) in zip(profile, profile[1:]):
        if h0 <= z['surface_z_m'] <= h1:
            t = (z['surface_z_m'] - h0) / max(1e-9, h1 - h0)
            radius = r0 + (r1 - r0) * t
            break
    point = Vector((radius * math.cos(angle), radius * math.sin(angle),
                    z['surface_z_m']))
    axis = Vector(z['axis']).normalized()
    boss = revolve_axis('Side fitting boss', [
        (0.0, -0.003), (z['boss_radius_m'], -0.003),
        (z['boss_radius_m'], z['boss_length_m']), (0.0, z['boss_length_m'])],
        point, axis, 48)
    assign(boss, 'paint', smooth='side', axis=tuple(axis), scale=.04,
           uv='box', tile=1)
    hexnut = hex_prism('Side fitting brass hex',
                       point + axis * (z['boss_length_m'] + z['brass_hex_length_m'] / 2),
                       axis, z['brass_hex_across_flats_m'], z['brass_hex_length_m'],
                       bevel_width=.0006)
    assign(hexnut, 'brass', smooth='side', axis=tuple(axis), scale=.03, uv='box')
    cap = revolve_axis('Side fitting bronze cap', [
        (0.0, z['boss_length_m'] + z['brass_hex_length_m']),
        (z['cap_radius_m'], z['boss_length_m'] + z['brass_hex_length_m']),
        (z['cap_radius_m'], z['boss_length_m'] + z['brass_hex_length_m'] + z['cap_length_m']),
        (0.0, z['boss_length_m'] + z['brass_hex_length_m'] + z['cap_length_m'])],
        point, axis, 40)
    assign(cap, 'bronze', smooth='side', axis=tuple(axis), scale=.03, uv='box')


def build():
    build_body()
    build_ports()
    build_label()
    build_top()
    build_bottom()
    build_side_fitting()


# ---------------------------------------------------------------------------
# Render setup
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


def show_only(predicate):
    for obj in objects:
        obj.hide_render = not predicate(obj)


def all_visible():
    for obj in objects:
        obj.hide_render = False


try:
    scene = legacy.setup_scene('OPTIX', 512)
    render_device = 'OPTIX'
except Exception:
    scene = legacy.setup_scene('CPU', 512)
    render_device = 'CPU'
scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .50
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
LIGHT_SPECS = (
    ('large softbox', (2.6, -3.0, 3.6), 210, 3.0, 3.0),
    ('fill softbox', (-3.2, 2.6, 2.2), 105, 3.0, 3.0),
    ('front strip', (3.2, 1.0, 1.3), 110, 1.2, 3.4),
    ('rim high', (-1.2, 3.2, 3.4), 70, 1.4, 2.2),
    ('low fill', (0, -2.6, -.28), 38, 1.0, 1.0),
)
for name, location, energy, width, height in LIGHT_SPECS:
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = width
    data.size_y = height
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

MATERIALS = make_materials()
build()
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
Z_SHIFT = -low.z
TOP = high.z - low.z
for obj in objects:
    obj.location.z += Z_SHIFT
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
BOTTOM = low.z

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
          'faces': len(o.data.polygons),
          'material': material_key.get(o.name, '')} for o in objects]
closures = {o.name: non_manifold_edges(o) for o in objects}
intersections = {o.name: intersecting_faces(o) for o in objects}
render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
directions = {'front': (1, 0, .06), 'back': (-1, 0, .06),
              'left': (0, -1, .08), 'right': (0, 1, .08),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
r = config['render']
resolution = max(384, r['view_resolution'] // 2) if request['draft'] else r['view_resolution']
samples = max(8, r['view_samples'] // 4) if request['draft'] else r['view_samples']
span_front = high.z - low.z
span_side = high.z - low.z
span_top = max(high.x - low.x, high.y - low.y) * 1.45


def view_camera(name, center_point, extra=1.0):
    local = span_top if name in ('top', 'bottom') else (span_front if name in ('front', 'back') else span_side)
    return camera(scene, directions[name], center_point, local * extra)


for name in directions:
    cam = view_camera(name, center)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

detail_res = max(384, r['detail_resolution'] // 2) if request['draft'] else r['detail_resolution']
detail_samples = max(8, r['detail_samples'] // 3) if request['draft'] else r['detail_samples']
material_res = max(512, r['material_resolution'] // 2) if request['draft'] else r['material_resolution']
material_samples = max(8, r['material_samples'] // 3) if request['draft'] else r['material_samples']


def zadd(value):
    return value + Z_SHIFT


detail_cameras = (
    ('overall-front', (0, 0, zadd(.34)), (1, -.25, .10), .70),
    ('top-manifold', (0, 0, zadd(.615)), (.65, -.65, .85), .16),
    ('top-block', (0, 0, zadd(.615)), (1, -.30, .18), .12),
    ('top-knob', (0, .048, zadd(.611)), (1, .55, .22), .075),
    ('top-coupler', (0, -.045, zadd(.611)), (1, -.45, .18), .075),
    ('upper-port', (config['body']['outer_radius_m'] + .006, 0, zadd(.4655)),
     (1, 0, .05), .075),
    ('lower-port', (config['body']['outer_radius_m'] + .006, 0, zadd(.1876)),
     (1, 0, .05), .075),
    ('label', (config['body']['outer_radius_m'] + .004, 0, zadd(.3290)),
     (1, 0, 0), .135),
    ('weld-upper', (config['body']['outer_radius_m'], .012, zadd(.5300)),
     (.7, -1, .22), .075),
    ('weld-lower', (config['body']['outer_radius_m'], .012, zadd(.1080)),
     (.7, -1, .22), .075),
    ('side-fitting', (0, 0, zadd(.556)), (.55, -1, .45), .085),
    ('bottom-manifold', (0, 0, zadd(-.006)), (.65, -.65, -.25), .16),
    ('bottom-block', (0, 0, zadd(-.006)), (.8, -.45, -.35), .12),
    ('body-surface', (config['body']['outer_radius_m'], -.03, zadd(.26)),
     (.9, -1, .22), .22),
)
for name, target, direction, scale in detail_cameras:
    cam = camera(scene, direction, Vector(target), scale)
    draw(scene, render_dir / f'detail-{name}.png',
         resolution=detail_res, samples=detail_samples)
    bpy.data.objects.remove(cam, do_unlink=True)

isolated_groups = (
    ('body-only', lambda o: o.name == 'Body' or o.name.startswith('Weld seam'),
     (0.6, -1, .16), (0, 0, zadd(.32)), .58),
    ('ports-only', lambda o: o.name.startswith('Port '), (1, 0, .06),
     (config['body']['outer_radius_m'], 0, zadd(.33)), .33),
    ('label-only', lambda o: o.name.startswith('HPEOK'), (1, 0, 0),
     (config['body']['outer_radius_m'], 0, zadd(.329)), .16),
    ('top-valve-only', lambda o: o.name.startswith('Top '), (.65, -.65, .85),
     (0, 0, zadd(.615)), .16),
    ('bottom-valve-only', lambda o: o.name.startswith('Bottom '), (.65, -.65, -.25),
     (0, 0, zadd(-.006)), .16),
    ('side-fitting-only', lambda o: o.name.startswith('Side fitting'),
     (.55, -1, .45), (0, 0, zadd(.556)), .09),
    ('silhouette-front', lambda o: True, (1, 0, .06), center, span_front * 1.25),
    ('silhouette-top', lambda o: True, (0, 0, 1), center, span_top * 1.20),
)
for label, predicate, direction, target, scale in isolated_groups:
    show_only(predicate)
    cam = camera(scene, direction, Vector(target), scale)
    draw(scene, render_dir / f'{label}.png',
         resolution=512 if request['draft'] else 640,
         samples=8 if request['draft'] else 18)
    bpy.data.objects.remove(cam, do_unlink=True)
all_visible()

# Material and light-response renders.
close_target = Vector((0.03, 0, zadd(.34)))
cam = camera(scene, (1, -1, .30), close_target, .30)
draw(scene, render_dir / 'new-material-close.png',
     resolution=material_res, samples=material_samples)
bpy.data.objects.remove(cam, do_unlink=True)

show_only(lambda o: material_key.get(o.name) in ('steel', 'brass', 'bronze', 'slot'))
metal_target = Vector((0, 0, zadd(.615)))
cam = camera(scene, (.8, -.9, .55), metal_target, .17)
draw(scene, render_dir / 'new-metal-light-a.png',
     resolution=material_res // 2 if request['draft'] else 640,
     samples=8 if request['draft'] else 24)
bpy.data.objects.remove(cam, do_unlink=True)
key = bpy.data.objects['large softbox']
key.location = (-3.4, 2.8, 4.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
cam = camera(scene, (.8, -.9, .55), metal_target, .17)
draw(scene, render_dir / 'new-metal-light-b.png',
     resolution=material_res // 2 if request['draft'] else 640,
     samples=8 if request['draft'] else 24)
bpy.data.objects.remove(cam, do_unlink=True)

show_only(lambda o: material_key.get(o.name) in ('body', 'paint'))
paint_target = Vector((0.02, 0, zadd(.34)))
cam = camera(scene, (1, -.95, .28), paint_target, .30)
draw(scene, render_dir / 'new-paint-light-b.png',
     resolution=material_res // 2 if request['draft'] else 640,
     samples=8 if request['draft'] else 24)
bpy.data.objects.remove(cam, do_unlink=True)
key.location = (2.6, -3.0, 3.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
cam = camera(scene, (1, -.95, .28), paint_target, .30)
draw(scene, render_dir / 'new-paint-light-a.png',
     resolution=material_res // 2 if request['draft'] else 640,
     samples=8 if request['draft'] else 24)
bpy.data.objects.remove(cam, do_unlink=True)
all_visible()

cam = camera(scene, (1, -.95, .28), paint_target, .30)
draw(scene, render_dir / 'new-material-close-b.png',
     resolution=material_res, samples=material_samples)
bpy.data.objects.remove(cam, do_unlink=True)
key.location = (-3.4, 2.8, 4.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
cam = camera(scene, (1, -.95, .28), paint_target, .30)
draw(scene, render_dir / 'new-surface-alternate.png',
     resolution=material_res, samples=material_samples)
bpy.data.objects.remove(cam, do_unlink=True)
key.location = (2.6, -3.0, 3.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

if not request['draft']:
    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .10), center, span_front)
        draw(scene, render_dir / f'turn-{i:02d}.png',
             resolution=r['turntable_resolution'], samples=r['turntable_samples'])
        bpy.data.objects.remove(cam, do_unlink=True)

# Source-photo framing diagnostic (not an acceptance gate).
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old_span_front = oldhigh.z - oldlow.z
old.location += center - oldcenter
bpy.context.view_layer.update()
for obj in objects:
    obj.hide_render = False
old.hide_render = True
source = request['source_camera']
factor = span / max(1e-6, oldspan)
cam = legacy.make_camera(scene, (1.9 * factor, 0, center.z),
                         (0, 0, center.z), fov=source['fov_deg'])
cam.data.shift_x = source.get('shift_x', 0)
cam.data.shift_y = source.get('shift_y', 0)
from bpy_extras.object_utils import world_to_camera_view
bpy.context.view_layer.update()
corners = [Vector((x, y, z)) for x in (low.x, high.x)
           for y in (low.y, high.y) for z in (low.z, high.z)]
projected = [world_to_camera_view(scene, cam, p) for p in corners]
height_px = (max(v.y for v in projected) - min(v.y for v in projected)) * 512
source_bbox = request.get('source_alpha_bbox')
source_fit = 1.0
source_correction = 1.0
source_shift_px = 0.0
source_shift_world = 0.0
source_vertical_shift_px = 0.0
source_vertical_shift_world = 0.0
trial_height = height_px
if source_bbox:
    source_h = max(1, source_bbox[3] - source_bbox[1] + 1)
    source_fit = height_px / source_h
    cam.location.x *= source_fit
    bpy.context.view_layer.update()
    trial = render_dir / 'new-source-trial.png'
    draw(scene, trial, resolution=512, samples=max(8, samples // 4))
    img = bpy.data.images.load(str(trial))
    w, h = img.size
    pixels = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(pixels)
    mask = pixels.reshape(h, w, 4)[..., 3] >= .5
    rows = np.nonzero(mask.any(axis=1))[0]
    trial_height = int(rows.max() - rows.min() + 1) if len(rows) else 1
    source_correction = trial_height / source_h
    cam.location.x *= source_correction
    bpy.context.view_layer.update()
    columns = np.nonzero(mask.any(axis=0))[0]
    trial_cx = ((columns.min() + columns.max() + 1) / 2) if len(columns) else w / 2
    source_cx = (source_bbox[0] + source_bbox[2] + 1) / 2
    source_shift_px = (trial_cx - source_cx) / max(1e-6, source_correction)
    visible_width = 2 * abs(cam.location.x) * math.tan(math.radians(source['fov_deg']) / 2)
    source_shift_world = source_shift_px / w * visible_width
    cam.location.y += source_shift_world
    trial_cy = (rows.min() + rows.max() + 1) / 2
    source_cy = h - (source_bbox[1] + source_bbox[3] + 1) / 2
    source_vertical_shift_px = (trial_cy - source_cy) / max(1e-6, source_correction)
    source_vertical_shift_world = source_vertical_shift_px / h * visible_width
    cam.location.z += source_vertical_shift_world
    bpy.context.view_layer.update()
    bpy.data.images.remove(img)
source_eye = tuple(cam.location)
source_shift = (cam.data.shift_x, cam.data.shift_y)
draw(scene, render_dir / 'new-source.png', resolution=512, samples=samples)
bpy.data.objects.remove(cam, do_unlink=True)

for obj in objects:
    obj.hide_render = True
old.hide_render = False
for name in directions:
    local = span_top if name in ('top', 'bottom') else (span_front if name in ('front', 'back') else span_side)
    old_local = (max(oldhigh.x - oldlow.x, oldhigh.y - oldlow.y) * 1.45
                 if name in ('top', 'bottom') else old_span_front)
    cam = camera(scene, directions[name], center,
                 old_local if name in ('top', 'bottom') else max(old_span_front, oldspan))
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = legacy.make_camera(scene, source_eye, (0, 0, center.z), fov=source['fov_deg'])
cam.data.shift_x, cam.data.shift_y = source_shift
draw(scene, render_dir / 'old-source.png', resolution=512, samples=samples)
bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (1, -1, .30), close_target, .30)
draw(scene, render_dir / 'old-material-close.png',
     resolution=material_res, samples=material_samples)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)
all_visible()

data = {
    'parts': parts,
    'part_count': len(parts),
    'closures': closures,
    'closed_parts': sum(1 for v in closures.values() if v == 0),
    'self_intersecting_components': {k: v for k, v in intersections.items() if v},
    'bounds': [list(low), list(high)],
    'span': span,
    'overall_height_m': high.z - low.z,
    'body_diameter_m': 2 * config['body']['outer_radius_m'],
    'polygons': sum(len(o.data.polygons) for o in objects),
    'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                     for p in o.data.polygons),
    'materials': sorted(set(material_key.values())),
    'material_parts': {key: sorted(n for n, k in material_key.items() if k == key)
                       for key in sorted(set(material_key.values()))},
    'render_device': render_device,
    'z_shift': Z_SHIFT,
    'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
    'geometry_basis': ('clean/rust photo pixel ratios plus the uncalibrated six-view '
                       'visual match; label volume 10 L used only for working scale'),
    'render_samples': samples,
    'view_directions': {k: list(v) for k, v in directions.items()},
    'source_camera_note': ('source-photo framing distance scaled by new/old span, '
                           'then fitted to the source-alpha height; diagnostic only'),
    'source_camera_factor': factor,
    'source_framing_fit': source_fit,
    'source_framing_trial_height_px': trial_height,
    'source_framing_correction': source_correction,
    'source_framing_shift_px': round(float(source_shift_px), 3),
    'source_framing_shift_world': round(float(source_shift_world), 5),
    'source_framing_vertical_shift_px': round(float(source_vertical_shift_px), 3),
    'source_framing_vertical_shift_world': round(float(source_vertical_shift_world), 5),
    'source_framed_height_px': height_px,
    'overall_height_px_source': 1233,
    'body_diameter_px_source': 311,
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
print('WORKER39_OK', json.dumps({'parts': len(parts), 'span': span,
                                 'device': render_device}, ensure_ascii=False))
