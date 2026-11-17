"""Blender worker for the review-only fire-extinguisher candidate (item 35).

Shape basis: the clean/rust photograph of a red MFZ/ABC3 dry-powder
extinguisher.  The user six-view sheet is an uncalibrated visual-match
reference of the same family; it is used for topology/back-side orientation
only and is never treated as measured geometry.

The candidate is a part-based rebuild: rolled body shell with a concave bottom,
base band, lower/upper rolled seams, green valve collar, valve body, chrome
gauge bezel with a rectified photographic dial and a 3D needle, glass cover,
two handle bars, pivot bracket and pin, safety pin with red tamper seal, flared
white nozzle horn with a real through-bore, and a thin curved label shell using
a rectified label texture.  It never replaces the library mesh, is manual-only
and never enters automatic placement.
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
P = request['dimensions']
objects = []
tile_materials = {}
special_materials = {}
TAU = math.tau


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def image_node(nodes, path, noncolor=False):
    node = nodes.new('ShaderNodeTexImage')
    node.image = bpy.data.images.load(str(path), check_existing=True)
    node.image.pack()
    if noncolor:
        node.image.colorspace_settings.name = 'Non-Color'
    node.interpolation = 'Linear'
    return node


def make_atlas_materials():
    names = {0: '35 red enamel', 1: '35 black enamel',
             2: '35 white plastic', 3: '35 olive collar'}
    for tile, name in names.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')
        color = image_node(nodes, request['textures']['base_color'])
        metal = image_node(nodes, request['textures']['metallic_roughness'], True)
        normal = image_node(nodes, request['textures']['normal'], True)
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        links.new(metal.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = {0: .18, 1: .12,
                                                       2: .10, 3: .10}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        if tile in (0, 1):
            try:
                principled.inputs['Coat Weight'].default_value = .55 if tile == 0 else .45
                principled.inputs['Coat Roughness'].default_value = .05
            except KeyError:
                pass
        tile_materials[tile] = material


def make_special_materials():
    def textured(key, prefix, roughness, normal_strength, coat=0.0):
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')
        color = image_node(nodes, request['textures'][prefix + '_base'])
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        principled.inputs['Roughness'].default_value = roughness
        principled.inputs['Metallic'].default_value = 0.0
        try:
            normal = image_node(nodes, request['textures'][prefix + '_normal'], True)
            normal_map = nodes.new('ShaderNodeNormalMap')
            normal_map.inputs['Strength'].default_value = normal_strength
            links.new(normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        except KeyError:
            pass
        if coat:
            try:
                principled.inputs['Coat Weight'].default_value = coat
                principled.inputs['Coat Roughness'].default_value = .06
            except KeyError:
                pass
        special_materials[key] = material

    textured('label', 'label', .38, .06, coat=.20)
    textured('dial', 'dial', .28, .04, coat=.55)

    def plain(key, color, metallic, roughness, alpha=None, transmission=0.0):
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        principled = nodes.get('Principled BSDF')
        principled.inputs['Base Color'].default_value = (*color, 1)
        principled.inputs['Metallic'].default_value = metallic
        principled.inputs['Roughness'].default_value = roughness
        if transmission:
            try:
                principled.inputs['Transmission Weight'].default_value = transmission
            except KeyError:
                pass
            try:
                principled.inputs['IOR'].default_value = 1.45
            except KeyError:
                pass
        if alpha is not None:
            principled.inputs['Alpha'].default_value = alpha
            try:
                material.blend_method = 'BLEND'
            except Exception:
                pass
            try:
                material.show_transparent_back = False
            except Exception:
                pass
        special_materials[key] = material

    plain('chrome', (.78, .80, .83), 1.0, .10)
    plain('dark_steel', (.16, .17, .18), .85, .28)
    plain('glass', (.92, .95, 1.0), 0.0, .04, alpha=.13, transmission=.55)
    plain('red_seal', (.72, .03, .04), 0.0, .30)
    plain('needle_white', (.93, .94, .93), 0.0, .22)
    plain('needle_black', (.015, .015, .018), 0.0, .30)


def tile_uv(tile, u, v):
    col, row = tile % 2, tile // 2
    return (col * .5 + .012 + (u % 1.0) * .476,
            (1 - row) * .5 + .012 + (v % 1.0) * .476)


def box_uv(obj, scale=0.08):
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


def cyl_uv_z(obj, tile, scale=0.30):
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


def assign_tile(obj, tile, *, smooth='all', axis=None, scale=0.08, uv='box'):
    obj.data.materials.clear()
    obj.data.materials.append(tile_materials[tile])
    if uv == 'z':
        cyl_uv_z(obj, tile, scale)
    else:
        box_uv(obj, scale)
        remap_uv(obj, tile)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def assign_special(obj, key, *, smooth='all', axis=None):
    obj.data.materials.clear()
    obj.data.materials.append(special_materials[key])
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


# ---------------------------------------------------------------------------
# Geometry helpers
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
    if width <= 0:
        return obj
    mod = obj.modifiers.new('bevel', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(40)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def revolve_profile(name, profile, segments=96):
    """Revolve a closed (r,z) profile about Z.  r=0 entries become poles."""
    verts = []
    rings = []
    for r, z in profile:
        if r < 1e-7:
            rings.append(('pole', len(verts)))
            verts.append((0.0, 0.0, z))
        else:
            ids = []
            for k in range(segments):
                a = TAU * k / segments
                ids.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(('ring', ids))
    faces = []
    n = len(profile)
    for i in range(n):
        j = (i + 1) % n
        ti, ii = rings[i]
        tj, jj = rings[j]
        if ti == 'pole' and tj == 'pole':
            continue
        if ti == 'pole':
            for k in range(segments):
                faces.append((ii, jj[k], jj[(k + 1) % segments]))
        elif tj == 'pole':
            for k in range(segments):
                faces.append((ii[k], ii[(k + 1) % segments], jj))
        else:
            for k in range(segments):
                faces.append((ii[k], ii[(k + 1) % segments],
                              jj[(k + 1) % segments], jj[k]))
    return mesh_object(name, verts, faces)


def cylinder_axis(name, center, axis, radius, length, segments=64, radius2=None):
    axis = Vector(axis).normalized()
    center = Vector(center)
    radius2 = radius if radius2 is None else radius2
    helper = Vector((0, 0, 1)) if abs(axis.z) < .9 else Vector((0, 1, 0))
    u = axis.cross(helper).normalized()
    v = axis.cross(u).normalized()
    a = center - axis * length / 2
    b = center + axis * length / 2
    verts, ring_a, ring_b = [], [], []
    for k in range(segments):
        ang = TAU * k / segments
        d = u * math.cos(ang) + v * math.sin(ang)
        ring_a.append(len(verts)); verts.append(tuple(a + d * radius))
        ring_b.append(len(verts)); verts.append(tuple(b + d * radius2))
    ca = len(verts); verts.append(tuple(a))
    cb = len(verts); verts.append(tuple(b))
    faces = []
    for k in range(segments):
        k2 = (k + 1) % segments
        faces.append((ring_a[k], ring_a[k2], ring_b[k2], ring_b[k]))
        faces.append((ca, ring_a[k2], ring_a[k]))
        faces.append((cb, ring_b[k], ring_b[k2]))
    return mesh_object(name, verts, faces)


def annulus_axis(name, center, axis, outer, inner, length, segments=72):
    axis = Vector(axis).normalized()
    center = Vector(center)
    helper = Vector((0, 0, 1)) if abs(axis.z) < .9 else Vector((0, 1, 0))
    u = axis.cross(helper).normalized()
    v = axis.cross(u).normalized()
    a = center - axis * length / 2
    b = center + axis * length / 2
    verts = []
    rings = {}
    for side, origin in (('a', a), ('b', b)):
        for kind, radius in (('o', outer), ('i', inner)):
            ids = []
            for k in range(segments):
                ang = TAU * k / segments
                d = u * math.cos(ang) + v * math.sin(ang)
                ids.append(len(verts)); verts.append(tuple(origin + d * radius))
            rings[(side, kind)] = ids
    faces = []
    for k in range(segments):
        k2 = (k + 1) % segments
        faces.append((rings[('a', 'o')][k], rings[('a', 'o')][k2],
                      rings[('b', 'o')][k2], rings[('b', 'o')][k]))
        faces.append((rings[('b', 'i')][k], rings[('b', 'i')][k2],
                      rings[('a', 'i')][k2], rings[('a', 'i')][k]))
        faces.append((rings[('a', 'i')][k], rings[('a', 'i')][k2],
                      rings[('a', 'o')][k2], rings[('a', 'o')][k]))
        faces.append((rings[('b', 'o')][k], rings[('b', 'o')][k2],
                      rings[('b', 'i')][k2], rings[('b', 'i')][k]))
    return mesh_object(name, verts, faces)


def torus_axis(name, center, axis, major, minor, major_segments=96, minor_segments=16):
    axis = Vector(axis).normalized()
    center = Vector(center)
    helper = Vector((0, 0, 1)) if abs(axis.z) < .9 else Vector((0, 1, 0))
    u = axis.cross(helper).normalized()
    v = axis.cross(u).normalized()
    verts, rings = [], []
    for i in range(major_segments):
        a = TAU * i / major_segments
        radial = u * math.cos(a) + v * math.sin(a)
        ring = []
        for j in range(minor_segments):
            b = TAU * j / minor_segments
            p = center + radial * (major + minor * math.cos(b)) + axis * (minor * math.sin(b))
            ring.append(len(verts)); verts.append(tuple(p))
        rings.append(ring)
    faces = []
    for i in range(major_segments):
        i2 = (i + 1) % major_segments
        for j in range(minor_segments):
            j2 = (j + 1) % minor_segments
            faces.append((rings[i][j], rings[i2][j], rings[i2][j2], rings[i][j2]))
    return mesh_object(name, verts, faces)


def cone_shell(name, base, tip, r_base, r_tip, r_base_in, r_tip_in, segments=64):
    base, tip = Vector(base), Vector(tip)
    axis = (tip - base).normalized()
    helper = Vector((0, 0, 1)) if abs(axis.z) < .9 else Vector((0, 1, 0))
    u = axis.cross(helper).normalized()
    v = axis.cross(u).normalized()
    verts = []
    rings = {}
    for side, origin, ro, ri in (('base', base, r_base, r_base_in),
                                 ('tip', tip, r_tip, r_tip_in)):
        for kind, radius in (('o', ro), ('i', ri)):
            ids = []
            for k in range(segments):
                ang = TAU * k / segments
                d = u * math.cos(ang) + v * math.sin(ang)
                ids.append(len(verts)); verts.append(tuple(origin + d * radius))
            rings[(side, kind)] = ids
    faces = []
    for k in range(segments):
        k2 = (k + 1) % segments
        faces.append((rings[('base', 'o')][k], rings[('base', 'o')][k2],
                      rings[('tip', 'o')][k2], rings[('tip', 'o')][k]))
        faces.append((rings[('tip', 'i')][k], rings[('tip', 'i')][k2],
                      rings[('base', 'i')][k2], rings[('base', 'i')][k]))
        faces.append((rings[('tip', 'o')][k], rings[('tip', 'o')][k2],
                      rings[('tip', 'i')][k2], rings[('tip', 'i')][k]))
        faces.append((rings[('base', 'i')][k], rings[('base', 'i')][k2],
                      rings[('base', 'o')][k2], rings[('base', 'o')][k]))
    return mesh_object(name, verts, faces)


def swept_bar(name, points, half_width, half_thickness, segments=8):
    """Sweep a chamfered rectangular section along a polyline in the YZ plane."""
    pts = [Vector((0.0, y, z)) for y, z in points]
    verts, rings = [], []
    for i, p in enumerate(pts):
        if i == 0:
            tangent = (pts[1] - pts[0]).normalized()
        elif i == len(pts) - 1:
            tangent = (pts[-1] - pts[-2]).normalized()
        else:
            tangent = (pts[i + 1] - pts[i - 1]).normalized()
        normal = Vector((0.0, -tangent.z, tangent.y)).normalized()
        binormal = Vector((1.0, 0.0, 0.0))
        ring = []
        for j in range(segments):
            a = TAU * j / segments
            # Superellipse-like chamfered rectangle.
            cx = math.cos(a); sy = math.sin(a)
            du = half_width * math.copysign(abs(cx) ** .55, cx)
            dv = half_thickness * math.copysign(abs(sy) ** .55, sy)
            q = p + binormal * du + normal * dv
            ring.append(len(verts)); verts.append(tuple(q))
        rings.append(ring)
    faces = []
    for i in range(len(rings) - 1):
        for j in range(segments):
            j2 = (j + 1) % segments
            faces.append((rings[i][j], rings[i + 1][j],
                          rings[i + 1][j2], rings[i][j2]))
    c0 = len(verts); verts.append(tuple(pts[0]))
    c1 = len(verts); verts.append(tuple(pts[-1]))
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append((c0, rings[0][j2], rings[0][j]))
        faces.append((c1, rings[-1][j], rings[-1][j2]))
    return mesh_object(name, verts, faces)


def oriented_plate(name, center, u_dir, v_dir, half_u, half_v, thickness):
    center = Vector(center)
    u = Vector(u_dir).normalized() * half_u
    v = Vector(v_dir).normalized() * half_v
    n = u.cross(v).normalized() * (thickness / 2)
    verts = [tuple(center - u - v - n), tuple(center + u - v - n),
             tuple(center + u + v - n), tuple(center - u + v - n),
             tuple(center - u - v + n), tuple(center + u - v + n),
             tuple(center + u + v + n), tuple(center - u + v + n)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def curved_label(name, angle0, angle1, radius, thickness, z0, z1,
                 radial_segments=64, vertical_segments=2):
    verts = []
    grid = {}
    for layer, rad in (('outer', radius + thickness), ('inner', radius)):
        for iv in range(vertical_segments + 1):
            z = z0 + (z1 - z0) * iv / vertical_segments
            for iu in range(radial_segments + 1):
                a = angle0 + (angle1 - angle0) * iu / radial_segments
                grid[(layer, iv, iu)] = len(verts)
                verts.append((rad * math.cos(a), rad * math.sin(a), z))
    faces = []
    for iv in range(vertical_segments):
        for iu in range(radial_segments):
            faces.append((grid[('outer', iv, iu)], grid[('outer', iv, iu + 1)],
                          grid[('outer', iv + 1, iu + 1)], grid[('outer', iv + 1, iu)]))
            faces.append((grid[('inner', iv, iu)], grid[('inner', iv + 1, iu)],
                          grid[('inner', iv + 1, iu + 1)], grid[('inner', iv, iu + 1)]))
    for iu in range(radial_segments):
        faces.append((grid[('inner', 0, iu)], grid[('inner', 0, iu + 1)],
                      grid[('outer', 0, iu + 1)], grid[('outer', 0, iu)]))
        faces.append((grid[('inner', vertical_segments, iu)],
                      grid[('outer', vertical_segments, iu)],
                      grid[('outer', vertical_segments, iu + 1)],
                      grid[('inner', vertical_segments, iu + 1)]))
    for iv in range(vertical_segments):
        faces.append((grid[('inner', iv, 0)], grid[('outer', iv, 0)],
                      grid[('outer', iv + 1, 0)], grid[('inner', iv + 1, 0)]))
        faces.append((grid[('inner', iv, radial_segments)],
                      grid[('inner', iv + 1, radial_segments)],
                      grid[('outer', iv + 1, radial_segments)],
                      grid[('outer', iv, radial_segments)]))
    obj = mesh_object(name, verts, faces)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Label')
    uv = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            vi = obj.data.loops[loop_index].vertex_index
            # Recover grid indices from stored order.
            q = vi
            layer_idx = 0 if q < (radial_segments + 1) * (vertical_segments + 1) else 1
            local = q - layer_idx * (radial_segments + 1) * (vertical_segments + 1)
            iv, iu = divmod(local, radial_segments + 1)
            u = iu / radial_segments
            v = iv / vertical_segments
            # Outer layer faces map directly; inner/edge faces clamp to the border.
            if layer_idx == 1:
                u = min(max(u, 0.0), 1.0)
            uv[loop_index].uv = (u, v)
    return obj


def dial_disc(name, center, axis, radius, depth, segments=96):
    center = Vector(center)
    axis = Vector(axis).normalized()
    u = Vector((0, 1, 0)) if abs(axis.y) < .9 else Vector((1, 0, 0))
    u = (u - axis * u.dot(axis)).normalized()
    v = axis.cross(u).normalized()
    front = center + axis * depth / 2
    back = center - axis * depth / 2
    verts = []
    rf, rb = [], []
    for k in range(segments):
        a = TAU * k / segments
        d = u * math.cos(a) + v * math.sin(a)
        rf.append(len(verts)); verts.append(tuple(front + d * radius))
        rb.append(len(verts)); verts.append(tuple(back + d * radius))
    cf = len(verts); verts.append(tuple(front))
    cb = len(verts); verts.append(tuple(back))
    faces = []
    for k in range(segments):
        k2 = (k + 1) % segments
        faces.append((cf, rf[k], rf[k2]))
        faces.append((cb, rb[k2], rb[k]))
        faces.append((rf[k], rb[k], rb[k2], rf[k2]))
    obj = mesh_object(name, verts, faces)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Dial')
    uv = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            rel = Vector(co) - center
            uu = rel.dot(u) / (2 * radius) + .5
            vv = rel.dot(v) / (2 * radius) + .5
            uv[loop_index].uv = (min(max(uu, 0), 1), min(max(vv, 0), 1))
    return obj


# ---------------------------------------------------------------------------
# Build the extinguisher
# ---------------------------------------------------------------------------
def build():
    body_r = P['body_radius']
    # Body shell: concave bottom, base band, straight barrel, rolled top,
    # shallow dome and a short neck (the valve covers the top face).
    profile = [
        (0.0, P['bottom_center_z'] * 2.33),
        (0.022, P['bottom_center_z'] * 1.82),
        (0.042, P['bottom_center_z'] * .84),
        (0.058, P['bottom_center_z'] * .22),
        (P['base_radius'], 0.0),
        (P['base_radius'], 0.0055),
        (body_r + 0.0007, 0.0080),
        (body_r, 0.0105),
        (body_r, 0.0210),
        (body_r, P['body_top_z']),
        (body_r - 0.0003, P['body_top_z'] + 0.0055),
        (body_r - 0.0015, P['body_top_z'] + 0.0085),
        (body_r - 0.0035, P['body_top_z'] + 0.0115),
        (body_r - 0.0075, P['body_top_z'] + 0.0165),
        (body_r - 0.0135, P['body_top_z'] + 0.0225),
        (body_r - 0.0215, P['body_top_z'] + 0.0270),
        (body_r - 0.0315, P['body_top_z'] + 0.0305),
        (body_r - 0.0415, P['body_top_z'] + 0.0330),
        (P['neck_radius'], P['body_top_z'] + 0.0340),
        (P['neck_radius'], P['neck_top_z']),
        (0.0, P['neck_top_z']),
    ]
    body = revolve_profile('35 body', profile, 128)
    assign_tile(body, 0, smooth='side', axis=(0, 0, 1), scale=0.34, uv='z')
    assign_tile(revolve_profile('35 lower rolled seam',
                                [(body_r - 0.0020, 0.0185), (body_r + 0.0006, 0.0198),
                                 (body_r + 0.0016, 0.0212), (body_r + 0.0004, 0.0228),
                                 (body_r - 0.0020, 0.0238)], 128),
                0, smooth='all', scale=0.10)
    assign_tile(torus_axis('35 upper rolled seam', (0, 0, P['body_top_z'] + 0.0032),
                           (0, 0, 1), body_r, 0.0024, 128, 16),
                0, smooth='all', scale=0.10)
    assign_tile(torus_axis('35 bottom lip', (0, 0, 0.0016), (0, 0, 1),
                           P['base_radius'] - 0.0006, 0.0017, 128, 12),
                0, smooth='all', scale=0.08)

    # Green valve collar.
    collar = revolve_profile('35 valve collar',
                             [(0.0, P['collar_bottom_z']),
                              (P['collar_radius'] + 0.0025, P['collar_bottom_z']),
                              (P['collar_radius'] + 0.0028, P['collar_bottom_z'] + 0.0018),
                              (P['collar_radius'] + 0.0018, P['collar_bottom_z'] + 0.0038),
                              (P['collar_radius'], P['collar_bottom_z'] + 0.0050),
                              (P['collar_radius'], P['collar_top_z']),
                              (0.0, P['collar_top_z'])], 96)
    assign_tile(collar, 3, smooth='side', axis=(0, 0, 1), scale=0.10, uv='z')

    # Valve body (dark painted metal) and its bosses.
    valve = cylinder_axis('35 valve body', (0, 0, P['valve_center_z']),
                          (0, 0, 1), 0.0192, P['valve_height'], 64)
    apply_bevel(valve, 0.0015, 2)
    assign_tile(valve, 1, smooth='all', scale=0.07)
    flange = cylinder_axis('35 valve bottom flange', (0, 0, P['valve_bottom_z'] + 0.002),
                           (0, 0, 1), 0.0245, 0.004, 64)
    assign_tile(flange, 1, smooth='all', scale=0.06)
    top_nut = cylinder_axis('35 valve top nut', (0, 0, P['valve_top_z'] + 0.0045),
                            (0, 0, 1), 0.0135, 0.007, 6)
    assign_tile(top_nut, 1, smooth='flat', scale=0.05)
    gauge_housing = cylinder_axis('35 gauge housing',
                                  (0.024, 0, P['gauge_center_z']), (1, 0, 0),
                                  0.0218, 0.016, 72)
    assign_tile(gauge_housing, 1, smooth='side', axis=(1, 0, 0), scale=0.07)
    nozzle_boss = cylinder_axis('35 nozzle boss', (0, -0.020, P['nozzle_axis_z']),
                                (0, 1, 0), 0.0140, 0.020, 48)
    assign_tile(nozzle_boss, 1, smooth='side', axis=(0, 1, 0), scale=0.06)

    # Chrome gauge bezel, rectified photographic dial, 3D needle, glass and screw.
    bezel = annulus_axis('35 chrome gauge bezel',
                         (P['gauge_face_x'] - 0.0025, 0, P['gauge_center_z']),
                         (1, 0, 0), 0.0217, 0.0181, 0.007, 96)
    assign_special(bezel, 'chrome', smooth='side', axis=(1, 0, 0))
    face = dial_disc('35 gauge dial', (P['gauge_face_x'], 0, P['gauge_center_z']),
                     (1, 0, 0), 0.0179, 0.0009, 96)
    assign_special(face, 'dial', smooth='all')
    # Needle: black outline plate with a white plate on top, pointing into the red zone.
    angle = math.radians(-48)
    du = Vector((0, math.cos(angle), math.sin(angle)))
    dv = Vector((0, -math.sin(angle), math.cos(angle)))
    needle_len = 0.0155
    tip = Vector((P['gauge_face_x'] + 0.0006, 0, P['gauge_center_z'])) + du * needle_len
    base = Vector((P['gauge_face_x'] + 0.0006, 0, P['gauge_center_z'])) + du * 0.0022
    mid = (tip + base) / 2
    outline = oriented_plate('35 gauge needle outline', mid, du, dv,
                             needle_len / 2, 0.0024, 0.00035)
    assign_special(outline, 'needle_black', smooth='flat')
    needle = oriented_plate('35 gauge needle',
                            Vector((P['gauge_face_x'] + 0.00085, 0,
                                    P['gauge_center_z'])) + du * (needle_len / 2 + 0.0022),
                            du, dv, needle_len / 2, 0.0017, 0.00035)
    assign_special(needle, 'needle_white', smooth='flat')
    glass = dial_disc('35 gauge glass', (P['gauge_face_x'] + 0.00115, 0, P['gauge_center_z']),
                      (1, 0, 0), 0.0180, 0.0005, 96)
    assign_special(glass, 'glass', smooth='all')
    hub = annulus_axis('35 gauge chrome centre hub',
                       (P['gauge_face_x'] + 0.00055, 0, P['gauge_center_z']),
                       (1, 0, 0), 0.0058, 0.0024, 0.0011, 56)
    assign_special(hub, 'chrome', smooth='side', axis=(1, 0, 0))
    screw = cylinder_axis('35 gauge centre screw',
                          (P['gauge_face_x'] + 0.00045, 0, P['gauge_center_z']),
                          (1, 0, 0), 0.0024, 0.0012, 32)
    assign_special(screw, 'chrome', smooth='side', axis=(1, 0, 0))
    slot = oriented_plate('35 gauge screw slot',
                          (P['gauge_face_x'] + 0.00095, 0, P['gauge_center_z']),
                          (0, 1, 0), (0, 0, 1), 0.0019, 0.00022, 0.0003)
    assign_special(slot, 'needle_black', smooth='flat')
    # Tilt the whole gauge assembly slightly upward so the dial is visible in
    # the top view, as in the reference sheet; the source photo is not calibrated.
    pivot = Vector((0.019, 0, P['gauge_center_z']))
    tilt = (Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(-18), 4, 'Y')
            @ Matrix.Translation(-pivot))
    for obj in [o for o in objects if 'gauge' in o.name]:
        obj.data.transform(tilt)

    # Handle bracket, pivot and the two enamel bars.
    bracket = mesh_object('35 handle bracket',
                          [(-0.007, 0.011, 0.3915), (0.007, 0.011, 0.3915),
                           (0.007, 0.047, 0.3915), (-0.007, 0.047, 0.3915),
                           (-0.007, 0.011, 0.4315), (0.007, 0.011, 0.4315),
                           (0.007, 0.047, 0.4315), (-0.007, 0.047, 0.4315)],
                          [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                           (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    apply_bevel(bracket, 0.0016, 2)
    assign_tile(bracket, 1, smooth='flat', scale=0.07)
    pivot = cylinder_axis('35 pivot boss', (0.0, 0.040, P['pivot_z']), (1, 0, 0),
                          0.0080, 0.018, 48)
    assign_tile(pivot, 1, smooth='side', axis=(1, 0, 0), scale=0.05)
    pivot_pin = cylinder_axis('35 chrome pivot pin', (0.006, 0.040, P['pivot_z']),
                              (1, 0, 0), 0.0043, 0.010, 32)
    assign_special(pivot_pin, 'chrome', smooth='side', axis=(1, 0, 0))
    lower = swept_bar('35 lower handle',
                      [(0.028, 0.408), (0.042, 0.406), (0.058, 0.403),
                       (0.075, 0.400), (0.090, 0.398), (0.100, 0.397)],
                      0.0060, 0.0030, 10)
    assign_tile(lower, 1, smooth='all', scale=0.06)
    upper = swept_bar('35 upper handle lever',
                      [(0.030, 0.418), (0.044, 0.428), (0.060, 0.442),
                       (0.076, 0.457), (0.090, 0.470), (0.100, 0.478)],
                      0.0058, 0.0028, 10)
    assign_tile(upper, 1, smooth='all', scale=0.06)

    # Safety pin (real torus opening) and red tamper seal.
    pin = torus_axis('35 safety pin', (0.005, 0.050, 0.4245), (1, 0, 0),
                     0.0046, 0.0009, 48, 10)
    assign_special(pin, 'chrome', smooth='all')
    seal = torus_axis('35 red tamper seal', (0.004, 0.0565, 0.4175), (1, 0, 0),
                      0.0030, 0.0007, 40, 8)
    assign_special(seal, 'red_seal', smooth='all')
    seal_tag = oriented_plate('35 red seal tag', (0.004, 0.0595, 0.4115),
                              (0, 0, 1), (1, 0, 0), 0.0038, 0.0007, 0.0005)
    assign_special(seal_tag, 'red_seal', smooth='flat')

    # Flared white nozzle horn with a real through-bore, plus its base collar.
    horn = cone_shell('35 nozzle horn',
                      (0.0, -0.0180, P['nozzle_axis_z'] - 0.0020),
                      (0.0, -0.0600, P['nozzle_axis_z'] + 0.0050),
                      P['horn_base_outer'], P['horn_tip_outer'],
                      P['horn_base_inner'], P['horn_tip_inner'], 72)
    assign_tile(horn, 2, smooth='side', axis=(0, 1, 0), scale=0.06)
    horn_collar = cylinder_axis('35 nozzle collar',
                                (0.0, -0.0165, P['nozzle_axis_z'] - 0.0022),
                                (0, 1, 0), 0.0175, 0.010, 48)
    assign_tile(horn_collar, 2, smooth='side', axis=(0, 1, 0), scale=0.05)

    # Curved printed label shell.
    label = curved_label('35 label', math.radians(-59.0), math.radians(59.0),
                         body_r + 0.0006, 0.0010,
                         P['label_bottom_z'], P['label_top_z'], 96, 3)
    assign_special(label, 'label', smooth='all')


def bounds(obs):
    pts = []
    for obj in obs:
        pts.extend(obj.matrix_world @ v.co for v in obj.data.vertices)
    low = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    high = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (low + high) / 2
    span = max(high - low)
    return low, high, center, span


try:
    scene = legacy.setup_scene('OPTIX', 512)
    render_device = 'OPTIX'
except Exception:
    scene = legacy.setup_scene('CPU', 512)
    render_device = 'CPU'
make_atlas_materials()
make_special_materials()
build()
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
Z_SHIFT = -low.z
for obj in objects:
    obj.location.z += Z_SHIFT
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)


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
# Export and save
# ---------------------------------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'), export_format='GLB',
                          use_selection=True, export_yup=False, export_apply=True)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

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


scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .48
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
light_specs = (
    ('large softbox', (2.6, -3.0, 3.6), 220, 3.0, 3.0),
    ('fill softbox', (-3.2, 2.6, 2.2), 115, 3.0, 3.0),
    ('front strip', (3.2, 1.0, 1.3), 115, 1.2, 3.4),
    ('rim high', (-1.2, 3.2, 3.4), 75, 1.4, 2.2),
    ('low fill', (0, -2.6, -.28), 42, 1.0, 1.0),
    ('bottom fill', (0, 0.5, -2.8), 24, 2.6, 2.6),
)
lights = {}
for name, location, energy, width, height in light_specs:
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = width
    data.size_y = height
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()
    lights[name] = light

render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
directions = {'front': (1, 0, .16), 'back': (-1, 0, .16),
              'left': (0, -1, .18), 'right': (0, 1, .18),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 768
samples = 10 if request['draft'] else 44
span_len = span
span_front = max(high.y - low.y, high.z - low.z)
parts = [{'name': o.name, 'vertices': len(o.data.vertices),
          'faces': len(o.data.polygons)} for o in objects]
intersections = {o.name: intersecting_faces(o) for o in objects}

for name, direction in directions.items():
    use_span = span_front if name in ('front', 'back') else span_len
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

# Detail views.
detail_cameras = (
    ('gauge', (P['gauge_face_x'] + 0.10, -0.055, P['gauge_center_z'] + 0.020),
     (0.030, 0.0, P['gauge_center_z']), 0.075),
    ('handle', (0.10, -0.10, 0.44), (0.0, 0.045, 0.425), 0.12),
    ('nozzle', (0.055, -0.13, 0.415), (0.0, -0.035, 0.410), 0.10),
    ('label', (0.16, -0.05, 0.20), (0.0, 0.0, 0.18), 0.22),
    ('bottom', (0.05, -0.05, -0.16), (0.0, 0.0, 0.004), 0.22),
    ('top', (0.06, 0.02, 0.60), (0.0, 0.0, 0.41), 0.18),
    ('seam', (0.13, -0.05, 0.36), (0.0, 0.0, 0.345), 0.10),
    ('overall-3q', (0.28, -0.24, 0.16), (0.0, 0.0, 0.24), 0.55),
)
for name, eye, target, scale in detail_cameras:
    cam = camera(scene, Vector(eye) - Vector(target), Vector(target), scale, 1.0)
    draw(scene, render_dir / f'detail-{name}.png',
         resolution=768 if not request['draft'] else 480,
         samples=26 if not request['draft'] else 8)
    bpy.data.objects.remove(cam, do_unlink=True)

# Isolation renders.
groups = (
    ('body-only', lambda n: n.startswith('35 body') or 'seam' in n or 'lip' in n,
     (1, -0.25, .12), (0, 0, 0.18), 0.55),
    ('label-only', lambda n: n == '35 label', (1, -0.25, .12), (0, 0, 0.18), 0.55),
    ('valve-only', lambda n: ('valve' in n or 'gauge housing' in n or 'nozzle boss' in n)
     and 'gauge' not in n, (0.7, -1, .45), (0, 0, 0.405), 0.16),
    ('gauge-only', lambda n: 'gauge' in n, (1, -0.35, .25), (0.02, 0, 0.404), 0.11),
    ('handle-only', lambda n: 'handle' in n or 'pivot' in n or 'safety' in n or 'seal' in n,
     (0.6, -1, .35), (0, 0.04, 0.42), 0.18),
    ('nozzle-only', lambda n: 'nozzle horn' in n, (0.15, -1, .18), (0, -0.04, 0.408), 0.12),
    ('nozzle-through', lambda n: 'nozzle horn' in n, (0, -1, 0), (0, -0.04, 0.408), 0.12),
    ('safety-ring', lambda n: n == '35 safety pin', (1, 0, 0), (0.0, 0.05, 0.4245), 0.05),
    ('top-parts', lambda n: not ('body' in n or 'seam' in n or 'lip' in n or n == '35 label'),
     (0, 0, 1), (0, 0, 0.40), 0.16),
)
for label, predicate, direction, target, scale in groups:
    visible = [o.name for o in objects if predicate(o.name)]
    for obj in objects:
        obj.hide_render = obj.name not in visible
    cam = camera(scene, direction, Vector(target), scale, 1.0)
    draw(scene, render_dir / f'{label}.png',
         resolution=768 if not request['draft'] else 480,
         samples=20 if not request['draft'] else 8)
    bpy.data.objects.remove(cam, do_unlink=True)
for obj in objects:
    obj.hide_render = False

# Material / clay comparison.
clay_saved = {}


def set_clay(enable):
    if enable:
        clay = bpy.data.materials.new('35 clay check')
        clay.use_nodes = True
        p = clay.node_tree.nodes.get('Principled BSDF')
        p.inputs['Base Color'].default_value = (.55, .56, .58, 1)
        p.inputs['Roughness'].default_value = .55
        p.inputs['Metallic'].default_value = 0.0
        for obj in objects:
            clay_saved[obj.name] = [slot.material for slot in obj.material_slots]
            for slot in obj.material_slots:
                slot.material = clay
    else:
        for obj in objects:
            mats = clay_saved.get(obj.name)
            if not mats:
                continue
            for slot, material in zip(obj.material_slots, mats):
                slot.material = material


cam = camera(scene, (1, -0.55, 0.20), center, span, 1.18)
draw(scene, render_dir / 'new-material-front.png', resolution=1024,
     samples=32 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
set_clay(True)
cam = camera(scene, (1, -0.55, 0.20), center, span, 1.18)
draw(scene, render_dir / 'new-clay-front.png', resolution=1024,
     samples=24 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
set_clay(False)

# Alternate key light for enamel response.
lights['large softbox'].location = (-3.4, 2.8, 4.6)
lights['large softbox'].rotation_euler = (-lights['large softbox'].location).to_track_quat('-Z', 'Y').to_euler()
cam = camera(scene, (1, -0.55, 0.20), center, span, 1.18)
draw(scene, render_dir / 'new-material-alternate.png', resolution=1024,
     samples=32 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
lights['large softbox'].location = (2.6, -3.0, 3.6)
lights['large softbox'].rotation_euler = (-lights['large softbox'].location).to_track_quat('-Z', 'Y').to_euler()

# Chrome-only renders under two light directions (metal reflection evidence).
# A dark neutral environment keeps the reflection directional instead of
# flooding every chrome face with a uniform white studio reflection.
chrome_names = {o.name for o in objects
                if any(material is not None and material.name == 'chrome'
                       for material in o.data.materials)}
background = scene.world.node_tree.nodes.get('Background')
old_strength = background.inputs['Strength'].default_value
old_color = tuple(background.inputs['Color'].default_value)
background.inputs['Strength'].default_value = .06
background.inputs['Color'].default_value = (.06, .07, .08, 1)
for name, light in lights.items():
    light.hide_render = name != 'large softbox'
key_energy = lights['large softbox'].data.energy
key_size = lights['large softbox'].data.size
key_size_y = lights['large softbox'].data.size_y
lights['large softbox'].data.energy = 520
lights['large softbox'].data.size = 1.0
lights['large softbox'].data.size_y = 1.0
for label, location in (('a', (2.6, -3.0, 3.6)), ('b', (-3.2, 2.4, 3.0))):
    for obj in objects:
        obj.hide_render = obj.name not in chrome_names
    lights['large softbox'].location = location
    lights['large softbox'].rotation_euler = (-lights['large softbox'].location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, (1, -0.55, 0.25), Vector((0, 0, 0.41)), 0.14, 1.0)
    draw(scene, render_dir / f'metal-light-{label}.png', resolution=640,
         samples=28 if not request['draft'] else 8)
    bpy.data.objects.remove(cam, do_unlink=True)
for obj in objects:
    obj.hide_render = False
for light in lights.values():
    light.hide_render = False
lights['large softbox'].data.energy = key_energy
lights['large softbox'].data.size = key_size
lights['large softbox'].data.size_y = key_size_y
background.inputs['Strength'].default_value = old_strength
background.inputs['Color'].default_value = old_color
lights['large softbox'].location = (2.6, -3.0, 3.6)
lights['large softbox'].rotation_euler = (-lights['large softbox'].location).to_track_quat('-Z', 'Y').to_euler()

# 24-direction turntable.
for i in range(24):
    angle = TAU * i / 24
    cam = camera(scene, (math.cos(angle), math.sin(angle), .14), center, span, 1.20)
    draw(scene, render_dir / f'turn-{i:02d}.png', resolution=512,
         samples=10 if not request['draft'] else 6)
    bpy.data.objects.remove(cam, do_unlink=True)

# Source-photo framing diagnostic (not an acceptance gate).
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
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
source_fit = source_correction = 1.0
source_shift_px = source_shift_world = 0.0
source_vertical_shift_px = source_vertical_shift_world = 0.0
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

# Old model six-view renders.
for obj in objects:
    obj.hide_render = True
old.hide_render = False
for name, direction in directions.items():
    use_span = old_span_front if name in ('front', 'back') else oldspan
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (1, -0.55, 0.20), center, oldspan, 1.18)
draw(scene, render_dir / 'old-material-front.png', resolution=1024,
     samples=24 if not request['draft'] else 8)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                         for p in o.data.polygons),
        'materials': sorted(tile_materials),
        'special_materials': sorted(special_materials),
        'render_device': render_device,
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': ('original clean/rust photo proportions; six-view sheet is an '
                           'uncalibrated visual-match reference, not a measurement'),
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': ('Camera at +X is the source-photo view and faces the label; '
                      'left at -Y is the nozzle side and right at +Y the handle side. '
                      'The source photo and the reference sheet are not a calibrated turntable.'),
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
        'isolated_checks': {'nozzle_bore_expected_holes': 1,
                            'safety_pin_expected_holes': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
