"""Blender worker for the review-only QL5 screw-jack candidate (item 11).

Shape basis: the original clean/rust photograph for proportions and visible
structure, with the user six-view sheet as an uncalibrated visual-match
reference for hidden/back-side topology.  The candidate is a part-based
rebuild of the green cast body/base, chrome lifting screw, faceted collar,
rectangular operating lever, crank/gear housing, carry loop and recreated
label.  It never replaces the library mesh, is manual-only and is never used
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
    names = {0: '11 clean green enamel', 1: '11 weathered green enamel',
             2: '11 worn green enamel', 3: '11 rusted green enamel'}
    for tile, name in names.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')
        color = image_node(nodes, 'green_base')
        metal = image_node(nodes, 'green_mr', True)
        normal = image_node(nodes, 'green_normal', True)
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        links.new(metal.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = {0: .12, 1: .16, 2: .12, 3: .24}[tile]
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

    set_material('chrome', 'chrome', .18)
    set_material('steel', 'steel', .16)
    set_material('label', 'label', .04)

    def plain(key, color, metallic, roughness):
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        node = material.node_tree.nodes.get('Principled BSDF')
        node.inputs['Base Color'].default_value = (*color, 1)
        node.inputs['Metallic'].default_value = metallic
        node.inputs['Roughness'].default_value = roughness
        special_materials[key] = material

    bright = config['material_profile']['bright_green']['base_rgb']
    plain('bright_green', tuple(c / 255 for c in bright), 0.0, 0.40)
    plain('darksteel', (.055, .060, .065), .90, .32)
    plain('bore', (.012, .014, .016), .55, .48)


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


def cyl_uv_z_full(obj, scale=0.55):
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


def assign_two(obj, outer_key, inner_key, inner_faces, *, smooth='all', axis=None,
               scale=0.30):
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


def tube(name, start, end, r_out, r_in, segments=64):
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


def square_tube(name, start, end, outer_half, inner_half, up=(0, 0, 1)):
    """Closed hollow square/rectangular tube with an open far end.

    The end faces only cover the wall thickness, so each end is a real
    opening.  The returned face list also identifies the inner-wall faces so
    a dark bore material can be assigned.
    """
    start, end = Vector(start), Vector(end)
    axis = (end - start).normalized()
    up = Vector(up)
    if abs(axis.dot(up.normalized())) > .95:
        up = Vector((0, 1, 0))
    u = (up - axis * up.dot(axis)).normalized()
    v = axis.cross(u).normalized()
    verts, rings = [], []
    for center in (start, end):
        for sign_u, sign_v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            rings.append(center + u * (outer_half * sign_u) + v * (outer_half * sign_v))
    verts = [tuple(p) for p in rings]
    rings = [[0, 1, 2, 3], [4, 5, 6, 7]]
    inner_start = len(verts)
    for center in (start, end):
        for sign_u, sign_v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(tuple(center + u * (inner_half * sign_u) +
                               v * (inner_half * sign_v)))
    inner_rings = [[inner_start, inner_start + 1, inner_start + 2, inner_start + 3],
                   [inner_start + 4, inner_start + 5, inner_start + 6, inner_start + 7]]
    faces, inner_faces = [], []
    os_, oe = rings
    for i in range(4):
        j = (i + 1) % 4
        faces.append((os_[i], os_[j], oe[j], oe[i]))
    for i in range(4):
        j = (i + 1) % 4
        faces.append((inner_rings[0][i], inner_rings[1][i],
                      inner_rings[1][j], inner_rings[0][j]))
        inner_faces.append(len(faces) - 1)
    for i in range(4):
        j = (i + 1) % 4
        faces.append((os_[i], inner_rings[0][i], inner_rings[0][j], os_[j]))
        faces.append((oe[i], oe[j], inner_rings[1][j], inner_rings[1][i]))
    return mesh_object(name, verts, faces), inner_faces


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


def oriented_box(name, center, half_tangent, half_radial, half_z, azimuth):
    center = Vector(center)
    dr = Vector((math.cos(azimuth), math.sin(azimuth), 0))
    dt = Vector((-math.sin(azimuth), math.cos(azimuth), 0))
    dz = Vector((0, 0, 1))
    verts = []
    for sz in (-1, 1):
        for st, sr in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = center + dt * (half_tangent * st) + dr * (half_radial * sr) + dz * (half_z * sz)
            verts.append(tuple(p))
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


def sweep_loop_plane(name, points, azimuth, radius, segments=14):
    """Closed circular sweep along a closed (radial, z) path.

    The loop plane contains the radial direction at ``azimuth`` and world Z,
    so the loop stands off the cast body and is seen at an angle from the
    label side, matching the clean photo.
    """
    phi = math.radians(azimuth)
    rvec = Vector((math.cos(phi), math.sin(phi), 0))
    tvec = Vector((-math.sin(phi), math.cos(phi), 0))
    zvec = Vector((0, 0, 1))
    n = len(points)
    verts = []
    for i, (rr, zz) in enumerate(points):
        r0, z0 = points[(i - 1) % n]
        r1, z1 = points[(i + 1) % n]
        tr, tz = r1 - r0, z1 - z0
        length = math.hypot(tr, tz) or 1.0
        tr, tz = tr / length, tz / length
        tangent = (rvec * tr + zvec * tz).normalized()
        normal = tangent.cross(tvec).normalized()
        base = rvec * rr + zvec * zz
        for j in range(segments):
            a = TAU * j / segments
            verts.append(tuple(base + tvec * (radius * math.cos(a)) +
                               normal * (radius * math.sin(a))))
    faces = []
    for i in range(n):
        i1 = (i + 1) % n
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((i * segments + j, i * segments + j1,
                          i1 * segments + j1, i1 * segments + j))
    return mesh_object(name, verts, faces)


def body_radius(z):
    points = config['body_profile']
    for (r0, z0), (r1, z1) in zip(points, points[1:]):
        if z0 <= z <= z1 and abs(z1 - z0) > 1e-9:
            t = (z - z0) / (z1 - z0)
            return r0 + (r1 - r0) * t
    return points[-1][0]


def tapered_sector(name, a0, a1, z0, z1, offset, thickness, segments=64, steps=10):
    """Curved plate whose radius follows the cast body taper."""
    verts, sections = [], []
    for i in range(segments + 1):
        a = a0 + (a1 - a0) * i / segments
        ca, sa = math.cos(a), math.sin(a)
        ids = []
        for k in range(steps + 1):
            z = z0 + (z1 - z0) * k / steps
            r = body_radius(z) + offset
            ids.append(len(verts))
            verts.append((r * ca, r * sa, z))
        sections.append(ids)
    faces = []
    # Front surface.
    for i in range(segments):
        a, b = sections[i], sections[i + 1]
        for k in range(steps):
            faces.append((a[k], b[k], b[k + 1], a[k + 1]))
    # Back surface (offset by thickness in the radial direction).
    back_start = len(verts)
    for section in sections:
        for index in section:
            x, y, z = verts[index]
            rad = math.hypot(x, y) or 1.0
            verts.append((x * (1 + thickness / rad), y * (1 + thickness / rad), z))
    back = [[index + back_start for index in section] for section in sections]
    for i in range(segments):
        a, b = back[i], back[i + 1]
        for k in range(steps):
            faces.append((a[k], a[k + 1], b[k + 1], b[k]))
    # Side walls and top/bottom rims.
    for i in range(segments):
        for k in (0, steps):
            faces.append((sections[i][k], sections[i + 1][k],
                          back[i + 1][k], back[i][k]))
    for k in range(steps):
        faces.append((sections[0][k], back[0][k], back[0][k + 1], sections[0][k + 1]))
        faces.append((sections[-1][k], sections[-1][k + 1],
                      back[-1][k + 1], back[-1][k]))
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

    # --- Base: stepped rim, recessed ribbed underside and centre boss -----
    base = revolve_solid('Cast base', config['base']['profile'], 128)
    apply_bevel(base, .0035, 3)
    assign_tile(base, 1, smooth='side', axis=(0, 0, 1), scale=0.45)

    collar = torus('Base body collar', (0, 0, config['base']['collar']['z']),
                   (0, 0, 1), config['base']['collar']['major'],
                   config['base']['collar']['minor'], 96, 12)
    assign_tile(collar, 1, smooth='all', scale=0.16)
    base_rim = torus('Base outer rim', (0, 0, .0185), (0, 0, 1), .1130, .0040, 96, 8)
    assign_tile(base_rim, 1, smooth='all', scale=0.10)
    base_lower_rim = torus('Base lower rim', (0, 0, .0015), (0, 0, 1), .1120, .0040, 96, 8)
    assign_tile(base_lower_rim, 3, smooth='all', scale=0.08)

    bottom = config['base']['bottom']
    for i in range(bottom['rib_count']):
        angle = TAU * i / bottom['rib_count']
        radius = (bottom['rib_inner_radius'] + bottom['rib_outer_radius']) / 2
        rib = box_at(f'Base bottom rib {i + 1}',
                     (radius, 0, (bottom['rib_z0'] + bottom['rib_z1']) / 2),
                     ((bottom['rib_outer_radius'] - bottom['rib_inner_radius']) / 2,
                      bottom['rib_half_width'],
                      (bottom['rib_z1'] - bottom['rib_z0']) / 2))
        apply_bevel(rib, .0018, 2)
        rib.rotation_euler.z = angle
        assign_tile(rib, 2, smooth='side', axis=(0, 0, 1), scale=0.14)
    boss = along('Base center boss', (0, 0, bottom['center_boss_z0']),
                 (0, 0, bottom['center_boss_z1']),
                 bottom['center_boss_radius'], 48)
    apply_bevel(boss, .0018, 2)
    assign_tile(boss, 2, smooth='side', axis=(0, 0, 1), scale=0.10)
    recess = along('Base center recess', (0, 0, bottom['center_boss_z0'] - .0003),
                   (0, 0, bottom['center_boss_z0'] + .0012),
                   bottom['center_recess_radius'], 32)
    assign_special(recess, 'bore', smooth='side', axis=(0, 0, 1), scale=0.05)

    # --- Cast green body -------------------------------------------------
    body = revolve_solid('Jack body', config['body_profile'], seg)
    assign_tile(body, 1, smooth='all', scale=0.34, uv='z')

    # --- Top collar, screw and cap --------------------------------------
    top = config['top']
    collar_cfg = top['collar']
    collar = revolve_ring('Top collar', collar_cfg['profile'],
                          collar_cfg['segments'])
    apply_bevel(collar, .0015, 2)
    assign_tile(collar, 1, smooth='flat', scale=0.14)

    for index, azimuth_deg in enumerate(top['lugs']['azimuth_deg'], start=1):
        az = math.radians(azimuth_deg)
        lug_cfg = top['lugs']
        center = (math.cos(az) * lug_cfg['radius'],
                  math.sin(az) * lug_cfg['radius'], (lug_cfg['z0'] + lug_cfg['z1']) / 2)
        lug = oriented_box(f'Collar lug {index}', center,
                           lug_cfg['half_tangent'], lug_cfg['half_radial'],
                           (lug_cfg['z1'] - lug_cfg['z0']) / 2, az)
        apply_bevel(lug, .0012, 2)
        assign_tile(lug, 2, smooth='side', axis=(0, 0, 1), scale=0.06)
        pin_start = (math.cos(az) * (lug_cfg['radius'] - .006),
                     math.sin(az) * (lug_cfg['radius'] - .006), lug_cfg['z1'] - .010)
        pin_end = (math.cos(az) * (lug_cfg['radius'] + .006),
                   math.sin(az) * (lug_cfg['radius'] + .006), lug_cfg['z1'] - .010)
        pin = along(f'Collar lug pin {index}', pin_start, pin_end,
                    lug_cfg['pin_radius'], 16)
        assign_special(pin, 'steel', smooth='side', scale=0.03)

    screw_cfg = top['screw']
    screw = along('Chrome screw', (0, 0, screw_cfg['z0']),
                  (0, 0, screw_cfg['z1']), screw_cfg['radius'], 96)
    apply_bevel(screw, .0018, 2)
    screw.data.materials.clear()
    screw.data.materials.append(special_materials['chrome'])
    cyl_uv_z_full(screw, .30)
    set_smooth(screw, 'side', axis=(0, 0, 1))
    objects.append(screw)

    cap = top['cap']
    lip = along('Screw cap lip', (0, 0, cap['lip_z0']), (0, 0, cap['lip_z1']),
                cap['lip_radius'], 64)
    apply_bevel(lip, .0025, 2)
    assign_special(lip, 'darksteel', smooth='side', axis=(0, 0, 1), scale=0.10)
    cap_top = along('Screw cap top', (0, 0, cap['lip_z1']), (0, 0, cap['top_z1']),
                    cap['top_radius'], 64)
    apply_bevel(cap_top, .002, 2)
    assign_special(cap_top, 'darksteel', smooth='side', axis=(0, 0, 1), scale=0.12)
    top_rim = torus('Screw cap top rim', (0, 0, cap['top_z1'] - .0015),
                    (0, 0, 1), cap['top_rim_radius'], .0025, 64, 8)
    assign_special(top_rim, 'steel', smooth='all', scale=0.06)

    # --- Recreated label plate ------------------------------------------
    lb = config['label']
    a_half = math.radians(lb['half_angle_deg'])
    label = tapered_sector('Nameplate', -a_half, a_half, lb['z0'], lb['z1'],
                           .0006, lb['thickness'], 64, 12)
    label.data.materials.clear()
    label.data.materials.append(special_materials['label'])
    label_uv(label, a_half, lb['z0'], lb['z1'])
    set_smooth(label, 'flat')
    objects.append(label)

    # --- Raised 5T cast mark --------------------------------------------
    em = config['emboss']
    em_az = math.radians(em['azimuth_deg'])
    em_half = math.radians(em['half_angle_deg'])
    pad = tapered_sector('Emboss pad', em_az - em_half, em_az + em_half,
                         em['pad_z0'], em['pad_z1'], .0004, em['pad_thickness'], 32, 8)
    assign_tile(pad, 2, smooth='flat', scale=0.08)
    padding = body_radius(em['z']) + .0004 + em['pad_thickness']
    font = bpy.data.fonts.load('C:/Windows/Fonts/arialbd.ttf')
    curve = bpy.data.curves.new('Emboss text', 'FONT')
    curve.body = em['text']
    curve.font = font
    curve.size = em['size']
    curve.extrude = .0018
    curve.align_x = 'CENTER'
    curve.align_y = 'CENTER'
    text_obj = bpy.data.objects.new('Emboss text', curve)
    bpy.context.scene.collection.objects.link(text_obj)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = text_obj
    text_obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    text_obj = bpy.context.object
    transform = Matrix(((0, 0, 1, padding + .0012),
                        (1, 0, 0, 0.0),
                        (0, 1, 0, em['z']),
                        (0, 0, 0, 1)))
    text_obj.data.transform(transform)
    bm = bmesh.new()
    bm.from_mesh(text_obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(text_obj.data)
    bm.free()
    assign_tile(text_obj, 2, smooth='flat', scale=0.08)

    # --- Crank/gear housing, lever socket, lever and fittings -----------
    mech = config['mechanism']
    az = math.radians(mech['azimuth_deg'])
    dr = Vector((math.cos(az), math.sin(az), 0))
    dt = Vector((-math.sin(az), math.cos(az), 0))
    dz = Vector((0, 0, 1))
    socket_center = dr * mech['socket_center_radius'] + dz * mech['socket_z']
    lever_angle = math.radians(mech['lever_angle_deg'])
    lever_dir = (dr * math.cos(lever_angle) + dz * math.sin(lever_angle)).normalized()

    housing_cfg = mech['housing']
    housing_center = dr * housing_cfg['offset_radius'] + dz * housing_cfg['z']
    housing = along('Crank housing',
                    housing_center - dr * housing_cfg['length'] / 2,
                    housing_center + dr * housing_cfg['length'] / 2,
                    housing_cfg['radius'], 64)
    apply_bevel(housing, .004, 3)
    assign_tile(housing, 1, smooth='side', axis=tuple(dr), scale=0.20)

    sock = mech['socket']
    socket, _ = tube('Lever socket',
                     socket_center - lever_dir * sock['length'] / 2,
                     socket_center + lever_dir * sock['length'] / 2,
                     sock['outer_radius'], sock['inner_radius'], 64)
    assign_tile(socket, 2, smooth='all', scale=0.30)
    socket_liner, _ = tube('Socket bore liner',
                           socket_center - lever_dir * (sock['length'] / 2 - .002),
                           socket_center + lever_dir * (sock['length'] / 2 - .002),
                           sock['inner_radius'] - .0003,
                           sock['inner_radius'] - .0021, 64)
    assign_special(socket_liner, 'bore', smooth='all', scale=0.10)
    socket_rim = torus('Socket rim', socket_center + lever_dir * sock['length'] / 2,
                       tuple(lever_dir),
                       (sock['outer_radius'] + sock['inner_radius']) / 2,
                       (sock['outer_radius'] - sock['inner_radius']) / 2, 64, 8)
    assign_tile(socket_rim, 2, smooth='all', scale=0.06)

    lever_start = socket_center - lever_dir * .045
    lever_end = socket_center + lever_dir * mech['lever_length']
    lever, _ = square_tube('Lever square tube', lever_start, lever_end,
                           mech['lever_outer_half'],
                           mech['lever_inner_half'])
    assign_tile(lever, 1, smooth='side', axis=tuple(lever_dir), scale=0.32)
    lever_liner, _ = square_tube('Lever bore liner',
                                 lever_start + lever_dir * .003,
                                 lever_end - lever_dir * .0015,
                                 mech['lever_inner_half'] - .0003,
                                 mech['lever_inner_half'] - .0015)
    assign_special(lever_liner, 'bore', smooth='side', axis=tuple(lever_dir), scale=0.10)

    for index, off in enumerate(mech['socket_bolts']['tangent_offsets'], start=1):
        c = socket_center + dt * (sock['outer_radius'] + .002) + dz * off
        bolt = along(f'Socket bolt {index}', c - dt * .004, c + dt * .016,
                     mech['socket_bolts']['radius'], 6)
        assign_special(bolt, 'steel', smooth='flat', scale=0.04)

    plate_cfg = mech['side_plate']
    plate_center = (dr * housing_cfg['offset_radius'] + dt * (housing_cfg['radius'] - .002)
                    + dz * housing_cfg['z'])
    plate = along('Mechanism side plate',
                  plate_center, plate_center + dt * plate_cfg['thickness'],
                  plate_cfg['radius'], 64)
    apply_bevel(plate, .0025, 2)
    assign_tile(plate, 2, smooth='side', axis=tuple(dt), scale=0.14)
    side_bolt = along('Mechanism side bolt',
                      plate_center + dt * plate_cfg['thickness'],
                      plate_center + dt * (plate_cfg['thickness'] + plate_cfg['bolt_length']),
                      plate_cfg['bolt_radius'], 6)
    assign_special(side_bolt, 'steel', smooth='flat', scale=0.05)

    bracket_cfg = mech['lower_bracket']
    bracket_center = (dr * bracket_cfg['offset_radius'] + dz * bracket_cfg['z'])
    bracket = oriented_box('Lower bracket', bracket_center,
                           bracket_cfg['half_tangent'], bracket_cfg['half_radial'],
                           bracket_cfg['half_z'], az)
    apply_bevel(bracket, .002, 2)
    assign_tile(bracket, 2, smooth='side', axis=(0, 0, 1), scale=0.12)
    bracket_bolt = along('Lower bracket bolt',
                         bracket_center + dt * bracket_cfg['half_tangent'],
                         bracket_center + dt * (bracket_cfg['half_tangent'] + .016),
                         bracket_cfg['bolt_radius'], 6)
    assign_special(bracket_bolt, 'steel', smooth='flat', scale=0.04)

    link_cfg = mech['link_plate']
    link_center = (dr * link_cfg['offset_radius'] +
                   dz * (link_cfg['z0'] + link_cfg['z1']) / 2)
    link = oriented_box('Link plate', link_center,
                        link_cfg['half_tangent'], link_cfg['half_radial'],
                        (link_cfg['z1'] - link_cfg['z0']) / 2, az)
    apply_bevel(link, .0015, 2)
    assign_tile(link, 2, smooth='side', axis=(0, 0, 1), scale=0.08)
    pin_center = dr * link_cfg['offset_radius'] + dz * link_cfg['z0'] + dt * 0
    pin = along('Link pin',
                pin_center - dt * link_cfg['pin_length'] / 2,
                pin_center + dt * link_cfg['pin_length'] / 2,
                link_cfg['pin_radius'], 24)
    assign_special(pin, 'steel', smooth='side', axis=tuple(dt), scale=0.04)

    knob_cfg = mech['knob']
    knob_center = (dr * knob_cfg['offset_radius'] + dt * knob_cfg['tangent_offset']
                   + dz * knob_cfg['z'])
    stem = along('Release knob stem', knob_center,
                 knob_center + dt * knob_cfg['stem_length'],
                 knob_cfg['stem_radius'], 24)
    assign_special(stem, 'bright_green', smooth='side', axis=tuple(dt), scale=0.03)
    cap_start = knob_center + dt * knob_cfg['stem_length']
    cap_end = cap_start + dt * knob_cfg['cap_length']
    knob, _ = tube('Release knob cap', cap_start, cap_end,
                   knob_cfg['cap_radius'], knob_cfg['bore_radius'], 48)
    assign_special(knob, 'bright_green', smooth='all', scale=0.05)
    knob_liner, _ = tube('Release knob bore liner',
                         cap_start + dt * .001, cap_end - dt * .001,
                         knob_cfg['bore_radius'] - .0002,
                         knob_cfg['bore_radius'] - .0014, 48)
    assign_special(knob_liner, 'bore', smooth='all', scale=0.04)
    knob_rim = torus('Release knob rim', cap_end, tuple(dt),
                     (knob_cfg['cap_radius'] + knob_cfg['bore_radius']) / 2,
                     (knob_cfg['cap_radius'] - knob_cfg['bore_radius']) / 2, 48, 8)
    assign_special(knob_rim, 'bright_green', smooth='all', scale=0.04)

    # --- Carry loop and mounts ------------------------------------------
    hl = config['handle_loop']
    path = rounded_rect_loop(hl['center_radius'], hl['center_z'],
                             hl['half_radius'], hl['half_z'],
                             hl['corner_radius'], 8)
    loop = sweep_loop_plane('Carry handle loop', path, hl['azimuth_deg'],
                            hl['rod_radius'], 14)
    assign_tile(loop, 1, smooth='all', scale=0.16)
    loop_phi = math.radians(hl['azimuth_deg'])
    loop_r = Vector((math.cos(loop_phi), math.sin(loop_phi), 0))
    loop_t = Vector((-math.sin(loop_phi), math.cos(loop_phi), 0))
    loop_pad = tapered_sector('Loop backplate', loop_phi - .10, loop_phi + .10,
                              .090, .315, .0003, .0030, 16, 10)
    assign_tile(loop_pad, 2, smooth='flat', scale=0.10)
    back_az = math.radians(125)
    body_rib = tapered_sector('Body back rib', back_az - .10, back_az + .10,
                              .070, .300, .0003, .0028, 12, 10)
    assign_tile(body_rib, 2, smooth='flat', scale=0.10)
    for index, mz in enumerate(hl['mount_z'], start=1):
        mount = along(f'Loop mount {index}',
                      loop_r * hl['mount_r0'] + dz * mz,
                      loop_r * hl['mount_r1'] + dz * mz,
                      hl['mount_radius'], 32)
        apply_bevel(mount, .0018, 2)
        assign_tile(mount, 2, smooth='side', axis=tuple(loop_r), scale=0.09)
        pin_center = loop_r * (hl['mount_r1'] - .010) + dz * mz
        pin = along(f'Loop pin {index}',
                    pin_center - loop_t * hl['pin_length'] / 2,
                    pin_center + loop_t * hl['pin_length'] / 2,
                    hl['pin_radius'], 16)
        assign_special(pin, 'steel', smooth='side', axis=tuple(loop_t), scale=0.04)


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

cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .72)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

zs = Z_SHIFT
mech = config['mechanism']
az = math.radians(mech['azimuth_deg'])
dr = Vector((math.cos(az), math.sin(az), 0))
dt = Vector((-math.sin(az), math.cos(az), 0))
dz = Vector((0, 0, 1))
socket_center = dr * mech['socket_center_radius'] + dz * (mech['socket_z'] + zs)
lever_angle = math.radians(mech['lever_angle_deg'])
lever_dir = (dr * math.cos(lever_angle) + dz * math.sin(lever_angle)).normalized()
lever_end = socket_center + lever_dir * mech['lever_length']
hl = config['handle_loop']
loop_phi = math.radians(hl['azimuth_deg'])
loop_r = Vector((math.cos(loop_phi), math.sin(loop_phi), 0))
loop_t = Vector((-math.sin(loop_phi), math.cos(loop_phi), 0))
loop_center = loop_r * hl['center_radius'] + dz * (hl['center_z'] + zs)
loop_mount_center = loop_r * hl['mount_r1'] + dz * (hl['mount_z'][1] + zs)
groups = (
    ('lever-bore-only', lambda n: n == 'Lever square tube',
     tuple(lever_dir), tuple(lever_end), .16),
    ('handle-loop-only', lambda n: n == 'Carry handle loop', tuple(loop_t),
     tuple(loop_center), .40),
    ('base-bottom-only', lambda n: n == 'Cast base' or n.startswith('Base bottom rib')
     or n.startswith('Base center'), (0, 0, -1), (0, 0, .01 + zs), .34),
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
        ('lever-tip', tuple(lever_end), (0.9, -0.6, 0.55), .20),
        ('lever-socket', tuple(socket_center), tuple(dt + Vector((0, 0, .15))), .28),
        ('knob', tuple(dr * (mech['knob']['offset_radius'] + .05)
                      + dt * (mech['knob']['tangent_offset'] + .03)
                      + dz * (mech['knob']['z'] + zs)), (0.8, -0.7, 0.35), .14),
        ('mechanism-side', tuple(dr * mech['housing']['offset_radius']
                                 + dt * (mech['housing']['radius'] + .006)
                                 + dz * (mech['housing']['z'] + zs)),
         tuple(dt + Vector((0, 0, .20))), .20),
        ('loop', tuple(loop_center), tuple(loop_t + Vector((0, 0, .05))), .30),
        ('loop-mounts', tuple(loop_mount_center), tuple(loop_t + Vector((0, 0, .12))), .18),
        ('collar', (0, 0, config['top']['collar']['profile'][3][1] + zs),
         (0.9, -0.7, 0.35), .25),
        ('screw', (0, 0, (config['top']['screw']['z0'] + config['top']['screw']['z1']) / 2 + zs),
         (1, -0.4, 0.15), .32),
        ('cap', (0, 0, config['top']['cap']['top_z1'] - .01 + zs),
         (0.8, -0.6, 0.55), .18),
        ('label', (body_radius(config['label']['z0'] + .02),
                   0, (config['label']['z0'] + config['label']['z1']) / 2 + zs),
         (1, -0.25, 0.05), .36),
        ('emboss', (body_radius(config['emboss']['z']) + .004, 0,
                    config['emboss']['z'] + zs), (1, -0.25, 0.05), .22),
        ('base-front', (0.12, 0, .07 + zs), (1, -0.5, 0.15), .50),
        ('base-bottom', (0, 0, .015 + zs), (0, 0, -1), .34),
        ('surface', (0.06, -0.09, .30 + zs), (1, -0.8, 0.2), .18),
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
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .72)
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
cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .72)
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
                      'The lever/crank assembly and the carry loop are on the -Y side, '
                      'matching the clean photo; the uncalibrated reference panels draw '
                      'the loop on the opposite side and are treated as mirrored.'),
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
        'isolated_checks': {'lever_bore_expected': 1,
                            'handle_loop_opening_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
