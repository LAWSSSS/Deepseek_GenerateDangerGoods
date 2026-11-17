"""Blender worker for the review-only 10 t bottle-jack candidate (item 10).

Shape basis: the original clean/rust photograph for proportions and visible
structure, with the user six-view sheet as an uncalibrated visual-match
reference for hidden/back-side topology.  The candidate is a part-based
rebuild of the red cast base, body, chrome ram, checker saddle, pump/handle
socket, carry loop, label and the detached handle tube.  It never replaces the
library mesh, is manual-only and is never used by the automatic placement
pipeline.
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
material_for_tile = {}
special_materials = {}
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


def tile_materials():
    names = {0: '10 clean red enamel', 1: '10 weathered red enamel',
             2: '10 worn red enamel', 3: '10 rusted red enamel'}
    for tile, name in names.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')
        color = image_node(nodes, 'red_base')
        metal = image_node(nodes, 'red_mr', True)
        normal = image_node(nodes, 'red_normal', True)
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        links.new(metal.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = {0: .10, 1: .13, 2: .12, 3: .20}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        material_for_tile[tile] = material

    def set_material(key, prefix, normal_strength):
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')
        color = image_node(nodes, prefix + '_base')
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

    set_material('chrome', 'chrome', .12)
    set_material('steel', 'steel', .16)
    set_material('saddle', 'saddle', .55)
    set_material('label', 'label', .04)

    def plain(key, color, metallic, roughness):
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        node = material.node_tree.nodes.get('Principled BSDF')
        node.inputs['Base Color'].default_value = (*color, 1)
        node.inputs['Metallic'].default_value = metallic
        node.inputs['Roughness'].default_value = roughness
        special_materials[key] = material

    plain('dark', (.010, .012, .013), .45, .52)
    plain('bore', (.060, .064, .068), .85, .34)


def tile_uv(tile, u, v):
    col, top_row = tile % 2, tile // 2
    return (col * .5 + .012 + (u % 1.0) * .476,
            (1 - top_row) * .5 + .012 + (v % 1.0) * .476)


def box_uv(obj, scale=0.55):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='Atlas')
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


def remap_uv(obj, tile):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    for loop in obj.data.uv_layers.active.data:
        u, v = loop.uv
        loop.uv = tile_uv(tile, u, v)


def cyl_uv_z(obj, tile, scale=0.55):
    """Cylindrical UV about Z with an atlas tile."""
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
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
            uv[loop_index].uv = tile_uv(tile, (ca + delta) / TAU + .5,
                                        co.z / scale)


def cyl_uv_z_full(obj, scale=0.55):
    """Cylindrical UV about Z using the whole texture (chrome ram)."""
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
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
            uv[loop_index].uv = ((ca + delta) / TAU + .5, co.z / scale)


def label_uv(obj, a_half, z0, z1):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='LabelUV')
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            angle = math.atan2(co.y, co.x)
            u = (angle + a_half) / (2 * a_half)
            v = (co.z - z0) / max(1e-9, z1 - z0)
            uv[loop_index].uv = (min(1.0, max(0.0, u)), min(1.0, max(0.0, v)))


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


def assign_tile(obj, tile, *, smooth='all', axis=None, scale=0.55, uv='box'):
    obj.data.materials.clear()
    obj.data.materials.append(material_for_tile[tile])
    if uv == 'z':
        cyl_uv_z(obj, tile, scale)
    else:
        box_uv(obj, scale)
        remap_uv(obj, tile)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def assign_special(obj, key, *, smooth='all', axis=None, scale=0.45, boxed=True):
    obj.data.materials.clear()
    obj.data.materials.append(special_materials[key])
    if boxed:
        box_uv(obj, scale)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def assign_two(obj, outer_key, inner_key, inner_faces, *, smooth='all', scale=0.30):
    """Outer atlas/special material plus a different inner-wall material."""
    box_uv(obj, scale)
    if outer_key in special_materials:
        outer = special_materials[outer_key]
    else:
        outer = material_for_tile[outer_key]
        remap_uv(obj, int(outer_key))
    obj.data.materials.clear()
    obj.data.materials.append(outer)
    obj.data.materials.append(special_materials[inner_key])
    inner = set(inner_faces)
    for index, poly in enumerate(obj.data.polygons):
        poly.material_index = 1 if index in inner else 0
    set_smooth(obj, smooth)
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


def revolve_ring(name, profile, segments):
    verts, rings = [], []
    for r, z in profile:
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            ids.append(len(verts))
            verts.append((r * math.cos(a), r * math.sin(a), z))
        rings.append(ids)
    faces = []
    n = len(rings)
    for k in range(n):
        j = (k + 1) % n
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((rings[k][s], rings[k][s1], rings[j][s1], rings[j][s]))
    return mesh_object(name, verts, faces)


def superellipse_loft(name, levels, segments):
    """Closed loft of superellipse rings; levels are (z, ax, ay, exponent)."""
    verts, rings = [], []
    for level in levels:
        if isinstance(level, dict):
            z, ax, ay, exponent = (level['z'], level['ax'], level['ay'],
                                   level['exponent'])
        else:
            z, ax, ay, exponent = level
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            ca, sa = math.cos(a), math.sin(a)
            x = ax * math.copysign(abs(ca) ** exponent, ca)
            y = ay * math.copysign(abs(sa) ** exponent, sa)
            ids.append(len(verts))
            verts.append((x, y, z))
        rings.append(ids)
    faces = [tuple(rings[0][::-1]), tuple(rings[-1])]
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((a[s], a[s1], b[s1], b[s]))
    return mesh_object(name, verts, faces)


def tube(name, start, end, r_out, r_in, segments=64):
    """Closed thick-walled pipe; returns the object and inner-wall face ids."""
    start, end = Vector(start), Vector(end)
    axis = (end - start).normalized()
    u = axis.orthogonal().normalized()
    v = axis.cross(u).normalized()
    verts = []
    rings = []
    for center, radius in ((start, r_out), (end, r_out), (start, r_in), (end, r_in)):
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            p = center + u * (radius * math.cos(a)) + v * (radius * math.sin(a))
            ids.append(len(verts))
            verts.append(tuple(p))
        rings.append(ids)
    os_, oe, ins, ie = rings
    faces, inner_faces = [], []
    for s in range(segments):
        s1 = (s + 1) % segments
        faces.append((os_[s], os_[s1], oe[s1], oe[s]))
        inner_faces.append(len(faces))
        faces.append((ins[s], ie[s], ie[s1], ins[s1]))
        faces.append((os_[s], ins[s], ins[s1], os_[s1]))
        faces.append((oe[s], oe[s1], ie[s1], ie[s]))
    return mesh_object(name, verts, faces), inner_faces


def along(name, start, end, radius, segments=32, vertices=None):
    start, end = Vector(start), Vector(end)
    count = vertices or segments
    bpy.ops.mesh.primitive_cylinder_add(vertices=count, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    return obj


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


def torus(name, center, normal, major, minor, major_segments=48, minor_segments=12):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                     major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=center)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = Vector(normal).to_track_quat('Z', 'Y').to_euler()
    return obj


def cutter_cylinder(radius, start, end, segments=64):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    return obj


def boolean_cut(target, cutters, name):
    bpy.ops.object.select_all(action='DESELECT')
    for i, cutter in enumerate(cutters):
        mod = target.modifiers.new(f'{name} {i + 1:02}', 'BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.solver = 'EXACT'
        mod.object = cutter
        bpy.context.view_layer.objects.active = target
        target.select_set(True)
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def curved_sector(name, r0, r1, z0, z1, a0, a1, segments=16):
    """Closed curved plate sector about Z (label plate)."""
    verts, sections = [], []
    for i in range(segments + 1):
        a = a0 + (a1 - a0) * i / segments
        ca, sa = math.cos(a), math.sin(a)
        ids = []
        for r, z in ((r0, z0), (r0, z1), (r1, z1), (r1, z0)):
            ids.append(len(verts))
            verts.append((r * ca, r * sa, z))
        sections.append(ids)
    faces = []
    for i in range(segments):
        a, b = sections[i], sections[i + 1]
        for k in range(4):
            k1 = (k + 1) % 4
            faces.append((a[k], a[k1], b[k1], b[k]))
    faces.append(tuple(sections[0][::-1]))
    faces.append(tuple(sections[-1]))
    return mesh_object(name, verts, faces)


def rounded_rect_loop(cy, cz, hy, hz, radius, samples=8):
    points = []
    corners = ((cy + hy - radius, cz + hz - radius, 0, 90),
               (cy - hy + radius, cz + hz - radius, 90, 180),
               (cy - hy + radius, cz - hz + radius, 180, 270),
               (cy + hy - radius, cz - hz + radius, 270, 360))
    for ccy, ccz, a0, a1 in corners:
        for k in range(samples):
            angle = math.radians(a0 + (a1 - a0) * k / samples)
            points.append((ccy + radius * math.cos(angle),
                           ccz + radius * math.sin(angle)))
    return points


def sweep_loop(name, points, x0, radius, segments=14):
    """Closed circular sweep along a closed (y,z) path in a plane at x=x0."""
    n = len(points)
    verts = []
    for i, (y, z) in enumerate(points):
        y0, z0 = points[(i - 1) % n]
        y1, z1 = points[(i + 1) % n]
        ty, tz = y1 - y0, z1 - z0
        length = math.hypot(ty, tz) or 1.0
        ty, tz = ty / length, tz / length
        ny, nz = -tz, ty
        for j in range(segments):
            a = TAU * j / segments
            verts.append((x0 + radius * math.cos(a),
                          y + ny * radius * math.sin(a),
                          z + nz * radius * math.sin(a)))
    faces = []
    for i in range(n):
        i1 = (i + 1) % n
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((i * segments + j, i * segments + j1,
                          i1 * segments + j1, i1 * segments + j))
    return mesh_object(name, verts, faces)


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build():
    seg = config['radial_segments']

    # --- Cast base -------------------------------------------------------
    base = superellipse_loft('Cast base', config['base']['levels'], 128)
    apply_bevel(base, .0035, 3)
    b = config['base']
    cutters = [cutter_cylinder(b['corner_holes']['radius'],
                               (x, y, -0.03), (x, y, 0.26), 32)
               for x, y in b['corner_holes']['centers']]
    boolean_cut(base, cutters, 'Base corner holes')
    assign_tile(base, 1, smooth='side', axis=(0, 0, 1), scale=0.45)

    bottom = b['bottom']
    ring = revolve_ring('Base bottom ring',
                        [(bottom['ring_major'] - bottom['ring_minor'], 0.0),
                         (bottom['ring_major'] + bottom['ring_minor'], 0.0),
                         (bottom['ring_major'] + bottom['ring_minor'], bottom['rib_top']),
                         (bottom['ring_major'] - bottom['ring_minor'], bottom['rib_top'])],
                        96)
    assign_tile(ring, 3, smooth='all', scale=0.16)
    pad = along('Base bottom pad', (0, 0, 0.0), (0, 0, 0.020),
                bottom['pad_radius'], 48)
    apply_bevel(pad, .0025, 2)
    assign_tile(pad, 2, smooth='side', axis=(0, 0, 1), scale=0.12)
    rib1 = box_at('Base bottom rib X', (0, 0, bottom['rib_top'] / 2),
                  (bottom['rib_half_x'], bottom['rib_half_width'],
                   bottom['rib_top'] / 2))
    apply_bevel(rib1, .0015, 2)
    assign_tile(rib1, 3, smooth='side', axis=(0, 0, 1), scale=0.14)
    rib2 = box_at('Base bottom rib Y', (0, 0, bottom['rib_top'] / 2),
                  (bottom['rib_half_width'], bottom['rib_half_y'],
                   bottom['rib_top'] / 2))
    apply_bevel(rib2, .0015, 2)
    assign_tile(rib2, 3, smooth='side', axis=(0, 0, 1), scale=0.14)

    collar = torus('Base body collar', (0, 0, 0.172), (0, 0, 1),
                   0.126, .012, 96, 12)
    assign_tile(collar, 1, smooth='all', scale=0.16)

    bosses = b['bosses']
    for index, sy in enumerate(bosses['centers_y'], start=1):
        for tag, sx in (('front', 1), ('back', -1)):
            cx = sx * bosses['outer_x']
            start = (cx - sx * bosses['length'] / 2, sy, bosses['z'])
            end = (cx + sx * bosses['length'] / 2, sy, bosses['z'])
            boss = along(f'Base boss {tag} {index}', start, end,
                         bosses['radius'], 32)
            apply_bevel(boss, .003, 2)
            assign_tile(boss, 2, smooth='side', axis=(1, 0, 0), scale=0.12)
            bolt = along(f'Base bolt {tag} {index}',
                         (end[0] - sx * .004, sy, bosses['z']),
                         (end[0] + sx * .012, sy, bosses['z']),
                         bosses['bolt_radius'], 6)
            assign_tile(bolt, 2, smooth='flat', scale=0.06)

    rv = b['release_valve']
    vx, vy, vz = rv['center']
    valve = along('Release valve hex', (vx, vy, vz),
                  (vx, vy + rv['length'], vz), rv['hex_radius'], 6)
    assign_special(valve, 'steel', smooth='flat', scale=0.05)
    stem = along('Release valve stem', (vx, vy + rv['length'], vz),
                 (vx, vy + rv['length'] + .012, vz), rv['stem_radius'], 16)
    assign_special(stem, 'steel', smooth='side', axis=(0, 1, 0), scale=0.04)

    # --- Painted reservoir body -----------------------------------------
    body = revolve_solid('Jack body', config['body_profile'], seg)
    assign_tile(body, 1, smooth='all', scale=0.55, uv='z')

    top = config['top']
    hx = top['hex_collar']
    collar = along('Hex collar', (0, 0, hx['z0']), (0, 0, hx['z1']),
                   hx['radius'], 64, vertices=hx['vertices'])
    apply_bevel(collar, .004, 2)
    assign_tile(collar, 0, smooth='flat', scale=0.35)
    up = top['upper_collar']
    upper = along('Upper hex collar', (0, 0, up['z0']), (0, 0, up['z1']),
                  up['radius'], 48, vertices=up['vertices'])
    apply_bevel(upper, .003, 2)
    assign_tile(upper, 0, smooth='flat', scale=0.25)
    rn = top['ram_neck']
    neck = along('Ram neck', (0, 0, rn['z0']), (0, 0, rn['z1']),
                 rn['radius'], 96)
    assign_tile(neck, 1, smooth='all', scale=0.25)
    gasket = torus('Ram gasket', (0, 0, rn['z1'] - .004), (0, 0, 1),
                   rn['radius'] - .004, .0025, 64, 8)
    assign_special(gasket, 'dark', smooth='all', scale=0.03)

    ram_cfg = top['ram']
    ram = along('Chrome ram', (0, 0, ram_cfg['z0']), (0, 0, ram_cfg['z1']),
                ram_cfg['radius'], 96)
    apply_bevel(ram, .0015, 2)
    ram.data.materials.clear()
    ram.data.materials.append(special_materials['chrome'])
    cyl_uv_z_full(ram, .30)
    set_smooth(ram, 'side', axis=(0, 0, 1))
    objects.append(ram)

    skirt_cfg = top['saddle_skirt']
    skirt = along('Saddle skirt', (0, 0, skirt_cfg['z0']),
                  (0, 0, skirt_cfg['z1']), skirt_cfg['radius'], 96)
    apply_bevel(skirt, .003, 2)
    assign_tile(skirt, 0, smooth='side', axis=(0, 0, 1), scale=0.18)

    saddle_cfg = top['saddle']
    saddle = along('Saddle cap', (0, 0, saddle_cfg['z0']),
                   (0, 0, saddle_cfg['z1']), saddle_cfg['radius'], 96)
    apply_bevel(saddle, .005, 3)
    saddle.data.materials.clear()
    saddle.data.materials.append(special_materials['saddle'])
    box_uv(saddle, .12)
    set_smooth(saddle, 'side', axis=(0, 0, 1))
    objects.append(saddle)
    top_rim = torus('Saddle top rim', (0, 0, saddle_cfg['z1'] - .004), (0, 0, 1),
                    saddle_cfg['radius'] - .004, .0035, 96, 8)
    assign_tile(top_rim, 0, smooth='all', scale=0.10)

    # --- Pump, socket and linkage ---------------------------------------
    pump = config['pump']
    py = pump['y']
    pump_body = along('Pump body', (0, py, pump['body_z0']),
                      (0, py, pump['body_z1']), pump['body_radius'], 64)
    apply_bevel(pump_body, .002, 2)
    assign_tile(pump_body, 1, smooth='side', axis=(0, 0, 1), scale=0.25)
    lc = pump['lower_collar']
    lower_collar = along('Pump lower collar', (0, py, lc['z0']),
                         (0, py, lc['z1']), lc['radius'], 64)
    apply_bevel(lower_collar, .003, 2)
    assign_tile(lower_collar, 2, smooth='side', axis=(0, 0, 1), scale=0.16)

    yoke = box_at('Pump yoke', tuple(pump['yoke_center']), tuple(pump['yoke_half']))
    apply_bevel(yoke, .006, 3)
    assign_tile(yoke, 1, smooth='side', axis=(0, 0, 1), scale=0.18)
    pivot = along('Pump pivot pin',
                  (-pump['pivot_length'] / 2, py, pump['yoke_center'][2]),
                  (pump['pivot_length'] / 2, py, pump['yoke_center'][2]),
                  pump['pivot_radius'], 32)
    assign_special(pivot, 'steel', smooth='side', axis=(1, 0, 0), scale=0.06)

    center = Vector(pump['socket_center'])
    axis = Vector(pump['socket_axis']).normalized()
    start = center - axis * pump['socket_length'] / 2
    end = center + axis * pump['socket_length'] / 2
    socket, _ = tube('Pump handle socket', start, end,
                     pump['socket_outer_radius'],
                     pump['socket_inner_radius'], 72)
    assign_tile(socket, 0, smooth='all', scale=0.28)
    liner_start = start + axis * .002
    liner_end = end - axis * .002
    liner, _ = tube('Socket bore liner', liner_start, liner_end,
                    pump['socket_inner_radius'] - .0004,
                    pump['socket_inner_radius'] - .0018, 72)
    assign_special(liner, 'bore', smooth='all', scale=0.10)
    for tag, point in (('open', end), ('back', start)):
        rim = torus(f'Socket rim {tag}', point, tuple(axis),
                    (pump['socket_outer_radius'] + pump['socket_inner_radius']) / 2,
                    (pump['socket_outer_radius'] - pump['socket_inner_radius']) / 2,
                    72, 8)
        assign_tile(rim, 2, smooth='all', scale=0.10)

    link = box_at('Pump link plate', tuple(pump['link_center']),
                  tuple(pump['link_half']))
    apply_bevel(link, .002, 2)
    assign_tile(link, 1, smooth='side', axis=(1, 0, 0), scale=0.12)
    for index, bz in enumerate(pump['link_bolt_z'], start=1):
        bolt = along(f'Pump link bolt {index}',
                     (pump['link_center'][0] - .006, py, bz),
                     (pump['link_center'][0] + .014, py, bz),
                     pump['link_bolt_radius'], 6)
        assign_special(bolt, 'steel', smooth='flat', scale=0.05)
    lower_pin = along('Pump lower pin',
                      (pump['link_center'][0] - .006, py, pump['lower_pin_z']),
                      (pump['link_center'][0] + .016, py, pump['lower_pin_z']),
                      pump['lower_pin_radius'], 24)
    assign_special(lower_pin, 'steel', smooth='side', axis=(1, 0, 0), scale=0.05)

    # --- Carry loop ------------------------------------------------------
    hl = config['handle_loop']
    path = rounded_rect_loop(hl['center_y'], hl['center_z'],
                             hl['half_y'], hl['half_z'], hl['corner_radius'], 8)
    loop = sweep_loop('Carry handle loop', path, hl['plane_x'],
                      hl['rod_radius'], 14)
    assign_tile(loop, 0, smooth='all', scale=0.16)
    for index, mz in enumerate(hl['mount_z'], start=1):
        mount = along(f'Handle loop mount {index}',
                      (0, -0.140, mz), (0, -0.072, mz),
                      hl['mount_radius'], 32)
        apply_bevel(mount, .002, 2)
        assign_tile(mount, 2, smooth='side', axis=(0, 1, 0), scale=0.09)

    # --- Detached handle tube -------------------------------------------
    hb = config['handle_bar']
    bar, _ = tube('Detached handle bar', hb['bottom'], hb['top'],
                  hb['outer_radius'], hb['inner_radius'], 64)
    assign_tile(bar, 0, smooth='all', scale=0.16)
    bar_axis = (Vector(hb['top']) - Vector(hb['bottom'])).normalized()
    bar_liner, _ = tube('Handle bar bore liner',
                        Vector(hb['bottom']) + bar_axis * .002,
                        Vector(hb['top']) - bar_axis * .002,
                        hb['inner_radius'] - .0004,
                        hb['inner_radius'] - .0016, 64)
    assign_special(bar_liner, 'bore', smooth='all', scale=0.08)
    rim = torus('Handle bar top rim', hb['top'], tuple(bar_axis),
                (hb['outer_radius'] + hb['inner_radius']) / 2,
                (hb['outer_radius'] - hb['inner_radius']) / 2, 64, 8)
    assign_tile(rim, 0, smooth='all', scale=0.08)

    # --- Recreated label plate ------------------------------------------
    lb = config['label']
    a_half = math.radians(lb['half_angle_deg'])
    label = curved_sector('Nameplate', lb['r_in'], lb['r_out'],
                          lb['z0'], lb['z1'], -a_half, a_half, 64)
    label.data.materials.clear()
    label.data.materials.append(special_materials['label'])
    label_uv(label, a_half, lb['z0'], lb['z1'])
    set_smooth(label, 'all')
    objects.append(label)


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


try:
    scene = legacy.setup_scene('OPTIX', 512)
    render_device = 'OPTIX'
except Exception:
    scene = legacy.setup_scene('CPU', 512)
    render_device = 'CPU'
scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .45
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, width, height in (
        ('large softbox', (2.6, -3.0, 3.6), 210, 3.0, 3.0),
        ('fill softbox', (-3.2, 2.6, 2.2), 110, 3.0, 3.0),
        ('front strip', (3.2, 1.0, 1.3), 110, 1.2, 3.4),
        ('rim high', (-1.2, 3.2, 3.4), 70, 1.4, 2.2),
        ('low fill', (0, -2.6, -.28), 40, 1.0, 1.0)):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = width
    data.size_y = height
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

tile_materials()
build()
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
Z_SHIFT = -low.z
for obj in objects:
    obj.location.z += Z_SHIFT
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
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 72
for name, direction in directions.items():
    cam = camera(scene, direction, center, span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .70)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

zs = Z_SHIFT
pump_center = Vector(config['pump']['socket_center'])
pump_axis = Vector(config['pump']['socket_axis']).normalized()
bar_center = (Vector(config['handle_bar']['bottom']) +
              Vector(config['handle_bar']['top'])) / 2
bar_axis = (Vector(config['handle_bar']['top']) -
            Vector(config['handle_bar']['bottom'])).normalized()
groups = (
    ('base-holes-only', lambda n: n == 'Cast base', (0, 0, -1),
     (0, 0, 0.10 + zs), .52),
    # Only the socket tube itself; the liner/rim are inside the isolated
    # silhouette and would add cosmetic contour fragments.
    ('socket-bore-only', lambda n: n == 'Pump handle socket',
     tuple(pump_axis), tuple(pump_center), .14),
    ('handle-loop-only', lambda n: n == 'Carry handle loop', (1, 0, .03),
     (0, -0.190, 0.468 + zs), .33),
    ('handle-bar-only', lambda n: n == 'Detached handle bar',
     tuple(-bar_axis), tuple(bar_center), .16),
    ('body-only', lambda n: n == 'Jack body', (1, -1, .25), (0, 0, 0), 1.05),
)
if request['draft']:
    for label, predicate, direction, target, scale in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'{label}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

if not request['draft']:
    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .15), center, span)
        draw(scene, render_dir / f'turn-{i:02d}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    detail_cameras = (
        ('saddle', (0, 0, 0.964 + zs), (0.75, -0.62, 0.55), .17),
        ('ram-hex', (0, 0, 0.858 + zs), (0.85, -0.65, 0.18), .25),
        ('shoulder', (0, 0, 0.745 + zs), (0.85, -0.55, 0.22), .25),
        ('label', (0.108, 0, 0.455 + zs), (1, -0.22, 0.03), .35),
        ('loop', (-0.01, -0.190, 0.468 + zs), (1, 0.02, 0.05), .33),
        ('pump-head', (0, 0.173, 0.545 + zs), (1, -1, 0.55), .24),
        ('pump-socket', (0.045, 0.137, 0.614 + zs), (0.50, -0.40, 0.768), .11),
        ('pump-link', (0.048, 0.173, 0.330 + zs), (1, -0.20, 0.05), .22),
        ('base-front', (0.150, 0.000, 0.100 + zs), (1, -0.55, 0.16), .55),
        ('base-bottom', (0, 0, 0.020 + zs), (0, 0, -1), .50),
        ('release-valve', (0.040, 0.225, 0.155 + zs), (0.9, 0.5, 0.20), .075),
        ('handle-bar', (0.080, 0.400, 0.350 + zs), (1, 0.35, 0.15), .62),
        ('surface', (0.060, -0.100, 0.320 + zs), (1, -0.80, 0.20), .18),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    for label, predicate, direction, target, scale in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'{label}.png', resolution=768, samples=20)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    key = bpy.data.objects['large softbox']
    key.location = (-3.4, 2.8, 4.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .70)
    draw(scene, render_dir / 'new-metal-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Source-photo framing diagnostic (not an acceptance gate).
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
_, _, oldcenter, oldspan = bounds([old])
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
draw(scene, render_dir / 'new-source.png', resolution=512, samples=samples)
bpy.data.objects.remove(cam, do_unlink=True)

for obj in objects:
    obj.hide_render = True
old.hide_render = False
for name, direction in directions.items():
    cam = camera(scene, direction, center, oldspan)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .70)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                         for p in o.data.polygons),
        'texture_size': 2048, 'chrome_texture_size': 1024,
        'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'render_device': render_device,
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': 'original photo proportions; six-view sheet is an uncalibrated visual-match reference',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': ('Front camera at +X (label side); left at -Y, right at +Y. '
                      'The pump/handle socket is on +Y, the carry loop on -Y and the '
                      'detached handle bar stands on the +Y side with an inferred pose.'),
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
        'isolated_checks': {'base_corner_holes_expected': 4,
                            'socket_openings_expected': 1,
                            'handle_loop_opening_expected': 1,
                            'handle_bar_bore_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
