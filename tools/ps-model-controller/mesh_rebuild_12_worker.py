"""Blender worker for the review-only 2 t trolley-jack candidate (item 12).

Shape basis: the original clean/rust photograph for proportions and visible
structure, with the user six-view sheet as an uncalibrated visual-match
reference for hidden/back-side topology.  The candidate is a part-based
rebuild: welded red frame, two large front wheels on a through axle, two
small rear casters, tapered lifting arm and head, knurled saddle, black pump
cover with recreated 2 TON label, yellow carry loop and a horizontal tubular
handle.  It never replaces the library mesh, is manual-only and is never used
by the automatic placement pipeline.
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
    names = {0: '12 clean red enamel', 1: '12 weathered red enamel',
             2: '12 worn red enamel', 3: '12 rusted red enamel'}
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
        normal_map.inputs['Strength'].default_value = {0: .10, 1: .14, 2: .10, 3: .22}[tile]
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

    for key, prefix, strength in (
            ('chrome', 'chrome', .16), ('steel', 'steel', .14),
            ('gold', 'gold', .10), ('black', 'black', .16),
            ('yellow', 'yellow', .10), ('rubber', 'rubber', .40),
            ('saddle', 'saddle', .55), ('ton_label', 'ton_label', .02),
            ('warning_label', 'warning_label', .03)):
        set_material(key, prefix, strength)


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


def tube(name, start, end, r_out, r_in, segments=64):
    start, end = Vector(start), Vector(end)
    axis = (end - start).normalized()
    u = axis.orthogonal().normalized()
    v = axis.cross(u).normalized()
    verts, rings = [], []
    for center, radius in ((start, r_out), (end, r_out), (start, r_in), (end, r_in)):
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            p = center + u * (radius * math.cos(a)) + v * (radius * math.sin(a))
            ids.append(len(verts))
            verts.append(tuple(p))
        rings.append(ids)
    os_, oe, ins, ie = rings
    faces = []
    for s in range(segments):
        s1 = (s + 1) % segments
        faces.append((os_[s], os_[s1], oe[s1], oe[s]))
        faces.append((ins[s], ie[s], ie[s1], ins[s1]))
        faces.append((os_[s], ins[s], ins[s1], os_[s1]))
        faces.append((oe[s], oe[s1], ie[s1], ie[s]))
    return mesh_object(name, verts, faces)


def along(name, start, end, radius, segments=32):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius,
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


def extrude_profile(name, profile, y0, y1):
    n = len(profile)
    verts = [(x, y0, z) for x, z in profile] + [(x, y1, z) for x, z in profile]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def tapered_beam(name, p0, p1, half_width, h0, h1):
    p0, p1 = Vector(p0), Vector(p1)
    axis = (p1 - p0).normalized()
    side = Vector((0, 1, 0))
    up = Vector((axis.z, 0, -axis.x)).normalized()
    verts = []
    for center, h in ((p0, h0), (p1, h1)):
        for ss, uu in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(tuple(center + side * (half_width * ss) + up * (h * uu)))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def plate_link(name, p0, p1, half_width_y, half_thickness):
    return tapered_beam(name, p0, p1, half_width_y, half_thickness, half_thickness)


def sweep_tube(name, points, radius, segments=14):
    pts = [Vector(p) for p in points]
    n = len(pts)
    verts, rings = [], []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        ref = Vector((1, 0, 0))
        if abs(t.dot(ref)) > .95:
            ref = Vector((0, 0, 1))
        n1 = t.cross(ref).normalized()
        n2 = t.cross(n1).normalized()
        ids = []
        for j in range(segments):
            a = TAU * j / segments
            verts.append(tuple(p + n1 * (radius * math.cos(a)) +
                               n2 * (radius * math.sin(a))))
            ids.append(len(verts) - 1)
        rings.append(ids)
    faces = []
    for i in range(n - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((a[j], a[j1], b[j1], b[j]))
    faces.append(tuple(reversed(rings[0])))
    faces.append(tuple(rings[-1]))
    return mesh_object(name, verts, faces)


def hex_nut(name, center, axis, radius, length, segments=6):
    return along(name, Vector(center) - Vector(axis) * (length / 2),
                 Vector(center) + Vector(axis) * (length / 2), radius, segments)


def oriented_plate(name, center, u_dir, v_dir, half_u, half_v, thickness):
    center = Vector(center)
    u = Vector(u_dir).normalized()
    v = Vector(v_dir).normalized()
    normal = u.cross(v).normalized()
    verts = []
    for sn in (-1, 1):
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(tuple(center + u * (half_u * su) + v * (half_v * sv) +
                               normal * (thickness * sn)))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def plate_uv(obj, center, u_dir, v_dir, half_u, half_v):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='LabelUV')
    center = Vector(center)
    u = Vector(u_dir).normalized()
    v = Vector(v_dir).normalized()
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co - center
            uv[loop_index].uv = (min(1, max(0, co.dot(u) / (2 * half_u) + .5)),
                                 min(1, max(0, co.dot(v) / (2 * half_v) + .5)))


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
    frame = config['frame']
    wheels = config['wheels']
    arm_cfg = config['arm']
    head = config['head']
    saddle = config['saddle']
    pump = config['pump']
    socket = config['socket']
    handle = config['handle']
    loop = config['loop']
    labels_cfg = config['labels']

    # --- Welded red frame ------------------------------------------------
    y = frame['side_y']
    for side, sy in (('left', -1), ('right', 1)):
        outer = sy * (y + frame['side_thickness'] / 2)
        inner = sy * (y - frame['side_thickness'] / 2)
        plate = extrude_profile(f'Frame {side} plate', frame['side_profile'],
                                min(outer, inner), max(outer, inner))
        apply_bevel(plate, .0045, 2)
        assign_tile(plate, 1, smooth='side', axis=(0, 1, 0), scale=0.42)
        rib = box_at(f'Frame {side} rib',
                     ((frame['rib_x0'] + frame['rib_x1']) / 2,
                      sy * (y + frame['side_thickness'] / 2 + frame['rib_protrusion'] / 2),
                      frame['rib_z']),
                     ((frame['rib_x1'] - frame['rib_x0']) / 2, frame['rib_protrusion'] / 2,
                      frame['rib_half_height']))
        assign_tile(rib, 2, smooth='side', axis=(0, 1, 0), scale=0.16)
        front_bolt = hex_nut(f'Frame front bolt {side}',
                             (-0.252, sy * (y + .006), 0.070), (0, 1, 0),
                             0.0105, 0.014)
        assign_special(front_bolt, 'gold', smooth='flat', scale=0.05)
        rear_bolt = hex_nut(f'Frame rear bolt {side}',
                            (0.235, sy * (y + .006), 0.128), (0, 1, 0),
                            0.012, 0.016)
        assign_special(rear_bolt, 'gold', smooth='flat', scale=0.05)
    floor = box_at('Frame floor', ((frame['floor_x0'] + frame['floor_x1']) / 2, 0,
                                   (frame['floor_z0'] + frame['floor_z1']) / 2),
                   ((frame['floor_x1'] - frame['floor_x0']) / 2, frame['floor_half_y'],
                    (frame['floor_z1'] - frame['floor_z0']) / 2))
    apply_bevel(floor, .003, 2)
    assign_tile(floor, 1, smooth='flat', scale=0.40)
    rear_cross = box_at('Frame rear cross', (0.294, 0, 0.070), (0.010, 0.104, 0.022))
    apply_bevel(rear_cross, .003, 2)
    assign_tile(rear_cross, 1, smooth='side', axis=(1, 0, 0), scale=0.20)
    pump_cross = box_at('Frame pump crossmember', (0.191, 0, 0.153),
                        (0.072, 0.104, 0.012))
    apply_bevel(pump_cross, .003, 2)
    assign_tile(pump_cross, 1, smooth='side', axis=(1, 0, 0), scale=0.22)

    # --- Front wheels, hubs and through axle -----------------------------
    for side, sy in (('left', -1), ('right', 1)):
        cy = sy * wheels['front_y']
        tire = along(f'Front wheel {side}',
                     (wheels['front_x'], cy - wheels['front_width'] / 2, wheels['front_radius']),
                     (wheels['front_x'], cy + wheels['front_width'] / 2, wheels['front_radius']),
                     wheels['front_radius'], 72)
        apply_bevel(tire, .007, 3)
        assign_special(tire, 'rubber', smooth='side', axis=(0, 1, 0), scale=0.30)
        hub = tube(f'Front hub {side}',
                   (wheels['front_x'], cy - wheels['front_hub_width'] / 2, wheels['front_radius']),
                   (wheels['front_x'], cy + wheels['front_hub_width'] / 2, wheels['front_radius']),
                   wheels['front_hub_radius'], wheels['front_hub_bore'], 48)
        apply_bevel(hub, .004, 2)
        assign_special(hub, 'black', smooth='side', axis=(0, 1, 0), scale=0.12)
        nut = hex_nut(f'Front hub nut {side}',
                      (wheels['front_x'], sy * (wheels['front_y'] + wheels['front_hub_width'] / 2 + .006),
                       wheels['front_radius']), (0, 1, 0), 0.0105, 0.014)
        assign_special(nut, 'gold', smooth='flat', scale=0.04)
        hub_rim = torus(f'Front hub rim {side}',
                        (wheels['front_x'], sy * (wheels['front_y'] + wheels['front_hub_width'] / 2 - .002),
                         wheels['front_radius']), (0, 1, 0),
                        wheels['front_hub_radius'] - .004, .004, 48, 8)
        assign_special(hub_rim, 'steel', smooth='all', scale=0.10)
    axle = along('Front axle', (wheels['front_x'], -wheels['axle_half_length'], wheels['front_radius']),
                 (wheels['front_x'], wheels['axle_half_length'], wheels['front_radius']),
                 wheels['axle_radius'], 32)
    assign_special(axle, 'chrome', smooth='side', axis=(0, 1, 0), scale=0.16)

    # --- Rear casters ----------------------------------------------------
    for side, sy in (('left', -1), ('right', 1)):
        cy = sy * wheels['rear_y']
        ear = box_at(f'Caster ear {side}',
                     (wheels['ear_center_x'], sy * (wheels['rear_y'] + .010),
                      wheels['ear_center_z']),
                     (wheels['ear_half_x'], wheels['ear_half_y'], wheels['ear_half_z']))
        apply_bevel(ear, .008, 3)
        assign_tile(ear, 1, smooth='side', axis=(0, 1, 0), scale=0.16)
        ear_bolt = hex_nut(f'Caster top bolt {side}',
                           (wheels['ear_center_x'], sy * (wheels['rear_y'] + .010),
                            wheels['ear_center_z'] + wheels['ear_half_z'] + .004),
                           (0, 0, 1), 0.011, 0.010)
        assign_special(ear_bolt, 'gold', smooth='flat', scale=0.04)
        rear_tire = along(f'Rear wheel {side}',
                          (wheels['rear_x'], cy - wheels['rear_width'] / 2, wheels['rear_radius']),
                          (wheels['rear_x'], cy + wheels['rear_width'] / 2, wheels['rear_radius']),
                          wheels['rear_radius'], 48)
        apply_bevel(rear_tire, .004, 2)
        assign_special(rear_tire, 'rubber', smooth='side', axis=(0, 1, 0), scale=0.10)
        rear_hub = tube(f'Rear hub {side}',
                        (wheels['rear_x'], cy - wheels['rear_width'] / 2, wheels['rear_radius']),
                        (wheels['rear_x'], cy + wheels['rear_width'] / 2, wheels['rear_radius']),
                        wheels['rear_hub_radius'], wheels['rear_hub_bore'], 32)
        assign_special(rear_hub, 'gold', smooth='side', axis=(0, 1, 0), scale=0.05)
        for k, fy in enumerate((cy - wheels['fork_gap'] / 2 - .003,
                                cy + wheels['fork_gap'] / 2 + .003), start=1):
            fork = box_at(f'Rear fork plate {side} {k}',
                          (wheels['rear_x'] - .004, fy,
                           (wheels['fork_z0'] + wheels['fork_z1']) / 2),
                          (wheels['fork_half_x'], .0035,
                           (wheels['fork_z1'] - wheels['fork_z0']) / 2))
            apply_bevel(fork, .002, 2)
            assign_special(fork, 'black', smooth='flat', scale=0.07)
        top = box_at(f'Rear fork top {side}',
                     (wheels['rear_x'] - .004, cy, wheels['fork_z1'] - .006),
                     (wheels['fork_half_x'], wheels['fork_gap'] / 2 + .008, .006))
        apply_bevel(top, .002, 2)
        assign_special(top, 'black', smooth='flat', scale=0.06)
        stem = along(f'Caster stem {side}',
                     (wheels['rear_x'] - .009, cy, wheels['fork_z1']),
                     (wheels['rear_x'] - .009, cy, wheels['ear_center_z'] + .010),
                     0.010, 24)
        assign_special(stem, 'steel', smooth='side', axis=(0, 0, 1), scale=0.05)
        rear_axle = along(f'Rear axle {side}',
                          (wheels['rear_x'], cy - wheels['fork_gap'] / 2 - .010, wheels['rear_radius']),
                          (wheels['rear_x'], cy + wheels['fork_gap'] / 2 + .010, wheels['rear_radius']),
                          wheels['rear_axle_radius'], 16)
        assign_special(rear_axle, 'chrome', smooth='side', axis=(0, 1, 0), scale=0.04)
    rear_rod = along('Rear cross rod', (0.300, -0.126, 0.085), (0.300, 0.126, 0.085),
                     0.0065, 24)
    assign_special(rear_rod, 'chrome', smooth='side', axis=(0, 1, 0), scale=0.12)

    # --- Lifting arm, pivot, head and link --------------------------------
    pivot = Vector(arm_cfg['pivot'])
    front = Vector(arm_cfg['front'])
    arm = tapered_beam('Lifting arm', pivot, front, arm_cfg['half_width'],
                       arm_cfg['half_height_rear'], arm_cfg['half_height_front'])
    apply_bevel(arm, .008, 3)
    assign_tile(arm, 1, smooth='side', axis=(0, 1, 0), scale=0.36)
    boss = along('Arm pivot boss', pivot - Vector((0, arm_cfg['pivot_boss_half_length'], 0)),
                 pivot + Vector((0, arm_cfg['pivot_boss_half_length'], 0)),
                 arm_cfg['pivot_boss_radius'], 48)
    apply_bevel(boss, .004, 2)
    assign_special(boss, 'black', smooth='side', axis=(0, 1, 0), scale=0.18)
    boss_pin = along('Arm pivot pin',
                     pivot - Vector((0, arm_cfg['pivot_boss_half_length'] + .006, 0)),
                     pivot + Vector((0, arm_cfg['pivot_boss_half_length'] + .006, 0)),
                     0.011, 20)
    assign_special(boss_pin, 'steel', smooth='side', axis=(0, 1, 0), scale=0.06)

    head_box = box_at('Lifting head', head['center'], head['half'])
    apply_bevel(head_box, .006, 3)
    assign_special(head_box, 'black', smooth='side', axis=(0, 1, 0), scale=0.14)
    post = along('Saddle post', (saddle['center_x'], 0, head['post_z0']),
                 (saddle['center_x'], 0, head['post_z1']), head['post_radius'], 48)
    apply_bevel(post, .003, 2)
    assign_special(post, 'black', smooth='side', axis=(0, 0, 1), scale=0.10)
    pin = along('Head pin', (head['center'][0], -head['pin_half_length'], head['pin_z']),
                (head['center'][0], head['pin_half_length'], head['pin_z']),
                head['pin_radius'], 20)
    assign_special(pin, 'steel', smooth='side', axis=(0, 1, 0), scale=0.05)
    cotter = torus('Head cotter pin', (head['center'][0], -head['pin_half_length'] - .002,
                                      head['pin_z'] - .012), (0, 1, 0), .010, .0018, 20, 6)
    assign_special(cotter, 'steel', smooth='all', scale=0.03)

    for side, sy in (('left', -1), ('right', 1)):
        link = plate_link(f'Link plate {side}', head['link_top'], head['link_bottom'],
                          head['link_half_width_y'], head['link_half_thickness'])
        link.location.y = sy * 0.088
        apply_bevel(link, .0025, 2)
        assign_special(link, 'black', smooth='flat', scale=0.12)
        for label, at in (('upper', head['link_top']), ('lower', head['link_bottom'])):
            link_pin = along(f'Link pin {label} {side}',
                             (at[0], sy * 0.088 - .020, at[2]),
                             (at[0], sy * 0.088 + .020, at[2]),
                             head['link_pin_radius'], 16)
            assign_special(link_pin, 'steel', smooth='side', axis=(0, 1, 0), scale=0.04)

    # --- Saddle ----------------------------------------------------------
    pad = revolve_solid('Saddle pad', [
        (0.0, saddle['z_top'] - saddle['dish_depth']),
        (0.030, saddle['z_top'] - saddle['dish_depth'] + .002),
        (0.050, saddle['z_top']),
        (0.059, saddle['z_top']),
        (saddle['radius'], saddle['z_top'] - .004),
        (saddle['radius'], saddle['z_bottom'] + .006),
        (saddle['radius'] - .007, saddle['z_bottom']),
        (0.030, saddle['z_bottom'] - .001),
        (0.0, saddle['z_bottom'] - .001),
    ], seg)
    pad.location.x = saddle['center_x']
    apply_bevel(pad, .002, 2)
    assign_special(pad, 'saddle', smooth='all', scale=0.16)
    saddle_rim = torus('Saddle rim',
                       (saddle['center_x'], 0, saddle['z_top'] - .002),
                       (0, 0, 1), saddle['radius'] - .003, saddle['rim_minor'], 64, 8)
    assign_special(saddle_rim, 'saddle', smooth='all', scale=0.10)

    # --- Pump cover and recreated labels ---------------------------------
    cover = box_at('Pump cover', pump['center'], pump['half'])
    apply_bevel(cover, .008, 3)
    assign_special(cover, 'black', smooth='flat', scale=0.35)
    for side, sy in (('left', -1), ('right', 1)):
        lab = oriented_plate(f'Pump side label {side}',
                             (pump['center'][0], sy * (pump['half'][1] + .0015),
                              pump['center'][2]),
                             (1, 0, 0), (0, 0, 1),
                             pump['label_half_u'], pump['label_half_v'], .0012)
        plate_uv(lab, (pump['center'][0], sy * (pump['half'][1] + .0015),
                       pump['center'][2]), (1, 0, 0), (0, 0, 1),
                 pump['label_half_u'], pump['label_half_v'])
        lab.data.materials.clear()
        lab.data.materials.append(special_materials['ton_label'])
        set_smooth(lab, 'flat')
        objects.append(lab)
    top_lab = oriented_plate('Pump top label',
                             (pump['center'][0], 0, pump['center'][2] + pump['half'][2] + .0015),
                             (1, 0, 0), (0, 1, 0),
                             pump['top_label_half_u'], pump['top_label_half_v'], .0012)
    plate_uv(top_lab, (pump['center'][0], 0,
                       pump['center'][2] + pump['half'][2] + .0015),
             (1, 0, 0), (0, 1, 0),
             pump['top_label_half_u'], pump['top_label_half_v'])
    top_lab.data.materials.clear()
    top_lab.data.materials.append(special_materials['ton_label'])
    set_smooth(top_lab, 'flat')
    objects.append(top_lab)
    for i, (bx, by) in enumerate(((pump['center'][0] + .055, -.070),
                                  (pump['center'][0] + .055, .070),
                                  (pump['center'][0] - .058, -.070)), start=1):
        bolt = hex_nut(f'Pump bolt {i}',
                       (bx, by, pump['center'][2] + pump['half'][2] + .004),
                       (0, 0, 1), pump['bolt_radius'], 0.010)
        assign_special(bolt, 'gold', smooth='flat', scale=0.04)
    filler = along('Pump filler cap',
                   (pump['center'][0] - .058, .070, pump['center'][2] + pump['half'][2]),
                   (pump['center'][0] - .058, .070, pump['center'][2] + pump['half'][2] + .016),
                   0.012, 24)
    apply_bevel(filler, .002, 2)
    assign_special(filler, 'gold', smooth='side', axis=(0, 0, 1), scale=0.05)

    front_lab = oriented_plate('Front rating label',
                               (head['center'][0] - head['half'][0] - .0015, 0,
                                head['center'][2] + .004),
                               (0, 1, 0), (0, 0, 1),
                               labels_cfg['front_label_half_u'],
                               labels_cfg['front_label_half_v'], .0012)
    plate_uv(front_lab, (head['center'][0] - head['half'][0] - .0015, 0,
                         head['center'][2] + .004),
             (0, 1, 0), (0, 0, 1),
             labels_cfg['front_label_half_u'], labels_cfg['front_label_half_v'])
    front_lab.data.materials.clear()
    front_lab.data.materials.append(special_materials['ton_label'])
    set_smooth(front_lab, 'flat')
    objects.append(front_lab)

    # Warning strip along the lifting-arm sides.
    arm_dir = (front - pivot).normalized()
    arm_up = Vector((arm_dir.z, 0, -arm_dir.x)).normalized()
    t = labels_cfg['warning_center_t']
    wc = pivot + (front - pivot) * t
    for side, sy in (('left', -1), ('right', 1)):
        center = (wc.x, sy * (arm_cfg['half_width'] + .0022), wc.z)
        lab = oriented_plate(f'Warning label {side}', center, arm_dir, arm_up,
                             labels_cfg['warning_half_u'],
                             labels_cfg['warning_half_v'], .0010)
        plate_uv(lab, center, arm_dir, arm_up, labels_cfg['warning_half_u'],
                 labels_cfg['warning_half_v'])
        lab.data.materials.clear()
        lab.data.materials.append(special_materials['warning_label'])
        set_smooth(lab, 'flat')
        objects.append(lab)

    # --- Handle socket and handle ---------------------------------------
    sx = socket['center'][0]
    socket_tube = tube('Handle socket',
                       (sx - socket['length'] / 2, 0, socket['center'][2]),
                       (sx + socket['length'] / 2, 0, socket['center'][2]),
                       socket['outer_radius'], socket['inner_radius'], 48)
    apply_bevel(socket_tube, .003, 2)
    assign_special(socket_tube, 'black', smooth='side', axis=(1, 0, 0), scale=0.10)
    for side, sy in (('left', -1), ('right', 1)):
        ear = box_at(f'Socket ear {side}',
                     (sx - .030, sy * socket['ear_y'], socket['center'][2]),
                     (socket['ear_half'][0], socket['ear_half'][1], socket['ear_half'][2]))
        apply_bevel(ear, .003, 2)
        assign_special(ear, 'black', smooth='flat', scale=0.08)
        bolt = hex_nut(f'Socket bolt {side}',
                       (sx - .030, sy * (socket['ear_y'] + .012), socket['center'][2]),
                       (0, 1, 0), socket['bolt_radius'], 0.016)
        assign_special(bolt, 'gold', smooth='flat', scale=0.04)
    socket_bracket = box_at('Socket bracket', (sx - .055, 0, 0.135),
                            (0.024, 0.052, 0.045))
    apply_bevel(socket_bracket, .004, 2)
    assign_special(socket_bracket, 'black', smooth='flat', scale=0.10)

    hstart, hend = Vector(handle['start']), Vector(handle['end'])
    hdir = (hend - hstart).normalized()
    pipe = along('Handle pipe', hstart, hend, handle['radius'], 48)
    apply_bevel(pipe, .0018, 2)
    assign_tile(pipe, 0, smooth='side', axis=tuple(hdir), scale=0.40)
    frac_start = ((handle['grip_start'] - hstart.x) /
                  max(1e-9, hend.x - hstart.x))
    frac_end = ((handle['grip_end'] - hstart.x) /
                max(1e-9, hend.x - hstart.x))
    gs = hstart + (hend - hstart) * frac_start
    ge = hstart + (hend - hstart) * frac_end
    grip = along('Handle grip', gs, ge, handle['grip_radius'], 48)
    apply_bevel(grip, .006, 3)
    assign_special(grip, 'rubber', smooth='side', axis=tuple(hdir), scale=0.12)
    end_cap = along('Grip end cap', ge, ge + hdir * .012, handle['grip_radius'] - .004, 32)
    apply_bevel(end_cap, .004, 2)
    assign_special(end_cap, 'black', smooth='side', axis=tuple(hdir), scale=0.04)

    # --- Yellow carry loop ----------------------------------------------
    left = []
    for i in range(7):
        s = i / 6
        left.append((loop['mount_x'] - .029 * s,
                     -(loop['mount_y'] - .022 * s ** 1.1),
                     loop['mount_z'] + .152 * s))
    arc = []
    for i in range(1, 12):
        a = math.pi - math.pi * i / 12
        arc.append((loop['top_x'], (loop['top_half_y'] + .002) * math.cos(a),
                    .308 + .032 * math.sin(a)))
    points = left + arc + list(reversed([(x, -yy, z) for x, yy, z in left]))
    carry = sweep_tube('Carry loop', points, loop['radius'], 12)
    assign_special(carry, 'yellow', smooth='all', scale=0.16)
    for side, sy in (('left', -1), ('right', 1)):
        bush = along(f'Loop bush {side}',
                     (loop['mount_x'], sy * (loop['mount_y'] - .020), loop['mount_z']),
                     (loop['mount_x'], sy * (loop['mount_y'] + .008), loop['mount_z']),
                     0.016, 24)
        assign_special(bush, 'black', smooth='side', axis=(0, 1, 0), scale=0.05)
        bolt = hex_nut(f'Loop bolt {side}',
                       (loop['mount_x'], sy * (loop['mount_y'] + .018), loop['mount_z']),
                       (0, 1, 0), loop['bolt_radius'], loop['bolt_depth'])
        assign_special(bolt, 'steel', smooth='flat', scale=0.04)


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
directions = {'front': (-1, 0, .16), 'back': (1, 0, .16),
              'left': (0, -1, .18), 'right': (0, 1, .18),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 64
span_len = span
span_front = max(high.y - low.y, high.z - low.z)
for name, direction in directions.items():
    use_span = span_front if name in ('front', 'back') else span_len
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (-2.3, -2.0, 1.0), Vector((0, 0, center.z)), .78)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

zs = Z_SHIFT
wheel_cfg = config['wheels']
groups = (
    ('loop-only', lambda n: n == 'Carry loop', (-1, 0, .08),
     (config['loop']['mount_x'], 0, config['loop']['top_z'] + zs), .40),
    ('hub-bores-only', lambda n: n in ('Front hub left', 'Front hub right'),
     (0, -1, 0), (wheel_cfg['front_x'], 0, wheel_cfg['front_radius'] + zs), .32),
    ('socket-bore-only', lambda n: n == 'Handle socket', (-1, 0, 0),
     (config['socket']['center'][0], 0, config['socket']['center'][2] + zs), .28),
    ('frame-only', lambda n: n.startswith('Frame ') or n.startswith('Lifting arm'),
     (0.7, -1, .35), (0, 0, .14 + zs), 1.05),
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
        cam = camera(scene, (math.cos(angle), math.sin(angle), .14), center, span)
        draw(scene, render_dir / f'turn-{i:02d}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    arm_cfg = config['arm']
    pivot = Vector(arm_cfg['pivot'])
    front = Vector(arm_cfg['front'])
    t = config['labels']['warning_center_t']
    wc = pivot + (front - pivot) * t
    arm_dir = (front - pivot).normalized()
    detail_cameras = (
        ('saddle', (config['saddle']['center_x'], 0,
                    config['saddle']['z_top'] + zs), (0.4, -1, .55), .22),
        ('saddle-top', (config['saddle']['center_x'], 0,
                        config['saddle']['z_top'] + zs), (0.1, -0.3, 1), .18),
        ('arm-label', (wc.x, -0.09, wc.z + zs), (0.15, -1, .18), .34),
        ('carry-loop', (config['loop']['mount_x'], -0.02,
                        (config['loop']['top_z'] + config['loop']['mount_z']) / 2 + zs),
         (0.4, -1, .25), .30),
        ('loop-mounts', (config['loop']['mount_x'], -config['loop']['mount_y'],
                         config['loop']['mount_z'] + zs), (0.35, -1, .15), .17),
        ('pump-cover', (config['pump']['center'][0], -0.10,
                        config['pump']['center'][2] + zs), (0.4, -1, .35), .30),
        ('pump-label', (config['pump']['center'][0], -0.095,
                        config['pump']['center'][2] + zs), (0.15, -1, .10), .20),
        ('handle-socket', (config['socket']['center'][0], -0.055,
                           config['socket']['center'][2] + zs), (0.5, -1, .35), .22),
        ('handle-grip', (0.81, 0, 0.223 + zs), (0.6, -1, .25), .26),
        ('front-wheel', (wheel_cfg['front_x'], -wheel_cfg['front_y'] - .02,
                         wheel_cfg['front_radius'] + zs), (0.5, -1, .30), .24),
        ('rear-caster', (wheel_cfg['rear_x'], -wheel_cfg['rear_y'] - .03,
                         wheel_cfg['rear_radius'] + zs), (0.4, -1, .35), .17),
        ('frame-side', (0.02, -0.125, 0.10 + zs), (0.2, -1, .20), .50),
        ('arm-pivot', (-0.075, -0.10, 0.215 + zs), (0.3, -1, .40), .25),
        ('arm-link', (config['head']['link_top'][0], -0.10,
                      (config['head']['link_top'][2] + config['head']['link_bottom'][2]) / 2 + zs),
         (0.4, -1, .30), .28),
        ('base-bottom', (0, 0, 0.05 + zs), (0, 0, -1), .95),
        ('rear-frame', (0.25, -0.12, 0.12 + zs), (0.5, -1, .30), .28),
        ('surface', (0.02, -0.11, 0.16 + zs), (0.9, -1, .25), .20),
        ('front', (0, 0, 0.20 + zs), (-1, 0, .10), .48),
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
    cam = camera(scene, (-2.3, -2.0, 1.0), Vector((0, 0, center.z)), .78)
    draw(scene, render_dir / 'new-metal-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Source-photo framing diagnostic (not an acceptance gate).
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
# The loaded library mesh keeps its long axis along Y (saddle at -Y). Bake a
# -90 deg Z rotation into this comparison copy so its long axis matches this
# candidate's X and the saddle faces -X; the library file itself is untouched.
old.data.transform(Matrix.Rotation(-math.pi / 2, 4, 'Z'))
old.data.update()
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old_span_front = max(oldhigh.y - oldlow.y, oldhigh.z - oldlow.z)
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
    use_span = old_span_front if name in ('front', 'back') else oldspan
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (-2.3, -2.0, 1.0), Vector((0, 0, center.z)), .78)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                         for p in o.data.polygons),
        'texture_size': 2048, 'detail_texture_size': 1024,
        'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'render_device': render_device,
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': 'original photo proportions; six-view sheet is an uncalibrated visual-match reference',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': ('Front camera at -X (saddle end); back at +X (handle end); '
                      'left at -Y (photo side), right at +Y. The handle is modelled '
                      'horizontally as in the photo and side reference panels; the '
                      'front/back reference panels show it upright, which conflicts.'),
        'old_view_rotation_deg_z': -90.0,
        'old_view_note': ('The library mesh long axis is Y; the comparison copy is rotated '
                          '-90 deg about Z and baked into its mesh data for the old/new '
                          'renders only. The library file is unchanged.'),
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
        'isolated_checks': {'carry_loop_opening_expected': 1,
                            'front_hub_bore_expected': 1,
                            'handle_socket_bore_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
