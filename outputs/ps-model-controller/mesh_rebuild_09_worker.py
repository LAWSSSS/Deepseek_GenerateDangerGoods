"""Blender worker for the review-only separator-filter candidate (item 09).

Shape basis: the original rust photograph for proportions and visible
structure.  The user-supplied six-view sheet is an uncalibrated visual-match
reference of the same visible design: a vertical separator with a top flange,
one side nozzle, a lower skirt on four legs and a dark automatic drain.  The
sheet is used for the hidden/back-side topology and for comparison, never as
measured geometry.  The candidate never replaces the library mesh, is
manual-only and is never used by the automatic placement pipeline.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector, Quaternion
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
def atlas_materials():
    names = {0: '09 clean brushed steel', 1: '09 weathered steel',
             2: '09 weld steel', 3: '09 rust-stained steel'}
    for tile, name in names.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')

        def texture(key, noncolor=False):
            node = nodes.new('ShaderNodeTexImage')
            node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
            node.image.pack()
            if noncolor:
                node.image.colorspace_settings.name = 'Non-Color'
            node.interpolation = 'Linear'
            return node

        color = texture('steel_base')
        metal = texture('steel_mr', True)
        normal = texture('steel_normal', True)
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        links.new(metal.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = {0: .06, 1: .08, 2: .10, 3: .12}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        material_for_tile[tile] = material

    brass = bpy.data.materials.new('09 brass')
    brass.use_nodes = True
    nodes = brass.node_tree.nodes
    links = brass.node_tree.links
    principled = nodes.get('Principled BSDF')

    def brass_texture(key, noncolor=False):
        node = nodes.new('ShaderNodeTexImage')
        node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
        node.image.pack()
        if noncolor:
            node.image.colorspace_settings.name = 'Non-Color'
        node.interpolation = 'Linear'
        return node

    color = brass_texture('brass_base')
    metal = brass_texture('brass_mr', True)
    normal = brass_texture('brass_normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(metal.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = .18
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    special_materials['brass'] = brass

    def plain(key, color, metallic, roughness):
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        node = material.node_tree.nodes.get('Principled BSDF')
        node.inputs['Base Color'].default_value = (*color, 1)
        node.inputs['Metallic'].default_value = metallic
        node.inputs['Roughness'].default_value = roughness
        special_materials[key] = material

    plain('dark', (.008, .009, .009), .30, .58)
    plain('gasket', (.012, .013, .012), .00, .78)
    plain('glass', (.010, .040, .030), .25, .09)
    plain('polished', (.56, .57, .58), 1.00, .22)
    plain('inset', (.085, .090, .095), .45, .44)
    plain('drain', (.155, .160, .165), .45, .38)
    plain('blue', (.015, .080, .500), .00, .34)
    plain('red', (.580, .025, .015), .00, .46)


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
    uv = obj.data.uv_layers.active.data
    for loop in uv:
        u, v = loop.uv
        loop.uv = tile_uv(tile, u, v)


def cyl_uv_z(obj, tile, scale=0.55):
    """Cylindrical UV about the Z axis; avoids box-UV smearing on shells."""
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


def cyl_uv_y(obj, tile, z_center, scale=0.45):
    """Cylindrical UV about the Y axis for the side nozzle."""
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    for poly in mesh.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.z - z_center, co.x))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            delta = (angle - ca + math.pi) % TAU - math.pi
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv[loop_index].uv = tile_uv(tile, (ca + delta) / TAU + .5,
                                        co.y / scale)


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


def assign_tile(obj, tile, *, boxed=True, smooth='all', axis=None, scale=0.55,
                uv='box', z_center=None):
    obj.data.materials.clear()
    obj.data.materials.append(material_for_tile[tile])
    if uv == 'z':
        cyl_uv_z(obj, tile, scale)
    elif uv == 'y':
        cyl_uv_y(obj, tile, 0.0 if z_center is None else z_center, scale)
    elif boxed:
        box_uv(obj, scale)
        remap_uv(obj, tile)
    else:
        remap_uv(obj, tile)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def assign_special(obj, key, *, boxed=True, smooth='all', axis=None, scale=0.45):
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
    """Revolve a pole-to-pole (r,z) profile about Z into a closed solid."""
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
    """Revolve a closed (r,z) profile about Z into a closed ring solid."""
    verts, rings = [], []
    for r, z in profile:
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            ids.append(len(verts))
            verts.append((r * math.cos(a), r * math.sin(a), z))
        rings.append(ids)
    n = len(rings)
    faces = []
    for k in range(n):
        j = (k + 1) % n
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((rings[k][s], rings[k][s1], rings[j][s1], rings[j][s]))
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


def rope_ring_z(name, z, major, minor, twists=64, major_segments=384,
                minor_segments=10, wobble=.0012):
    """Rope-like weld bead revolved about Z; radius and height are modulated."""
    verts = []
    for i in range(major_segments):
        t = TAU * i / major_segments
        ripple = math.sin(twists * t)
        r = minor * (1 + .22 * ripple)
        dz = wobble * math.sin(twists * t + .6)
        c = Vector((major * math.cos(t), major * math.sin(t), z + dz))
        radial = Vector((math.cos(t), math.sin(t), 0.0))
        axial = Vector((0.0, 0.0, 1.0))
        for j in range(minor_segments):
            p = TAU * j / minor_segments
            q = c + radial * (r * math.cos(p)) + axial * (r * math.sin(p))
            verts.append(tuple(q))
    faces = []
    for i in range(major_segments):
        i1 = (i + 1) % major_segments
        for j in range(minor_segments):
            j1 = (j + 1) % minor_segments
            faces.append((i * minor_segments + j, i * minor_segments + j1,
                          i1 * minor_segments + j1, i1 * minor_segments + j))
    return mesh_object(name, verts, faces)


def rope_ring_y(name, y, z_center, major, minor, twists=24, major_segments=256,
                minor_segments=10, wobble=.0009):
    """Rope-like weld bead around a Y-axis nozzle."""
    verts = []
    for i in range(major_segments):
        t = TAU * i / major_segments
        ripple = math.sin(twists * t)
        r = minor * (1 + .22 * ripple)
        dy = wobble * math.sin(twists * t + .6)
        c = Vector((major * math.cos(t), y + dy,
                    z_center + major * math.sin(t)))
        radial = Vector((math.cos(t), 0.0, math.sin(t)))
        axial = Vector((0.0, 1.0, 0.0))
        for j in range(minor_segments):
            p = TAU * j / minor_segments
            q = c + radial * (r * math.cos(p)) + axial * (r * math.sin(p))
            verts.append(tuple(q))
    faces = []
    for i in range(major_segments):
        i1 = (i + 1) % major_segments
        for j in range(minor_segments):
            j1 = (j + 1) % minor_segments
            faces.append((i * minor_segments + j, i * minor_segments + j1,
                          i1 * minor_segments + j1, i1 * minor_segments + j))
    return mesh_object(name, verts, faces)


def disc_y(name, y0, y1, radius, segments=96, z_center=0.0):
    verts, rings = [], []
    for y in (y0, y1):
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            ids.append(len(verts))
            verts.append((radius * math.cos(a), y,
                          z_center + radius * math.sin(a)))
        rings.append(ids)
    a, b = rings
    faces = [tuple(a), tuple(b)[::-1]]
    for s in range(segments):
        s1 = (s + 1) % segments
        faces.append((a[s], a[s1], b[s1], b[s]))
    return mesh_object(name, verts, faces)


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
    """Closed curved plate sector about Z (used for the nameplate)."""
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


def radial_prism(name, points_uz, angles, r0, r1, azimuth):
    """Prism in the radial plane with a (u,z) polygon, then rotate about Z."""
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    verts = []
    for r in (r0, r1):
        for u, z in points_uz:
            verts.append((r * ca - u * sa, r * sa + u * ca, z))
    n = len(points_uz)
    faces = [tuple(range(n)), tuple(range(n, 2 * n))[::-1]]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def arrow_cylinder_x(name, points_yz, surface_r, z_ref, offset0, offset1,
                     segments=1):
    """Arrow prism following a Z-axis cylinder, on the +X side."""
    verts = []
    for offset in (offset0, offset1):
        for y, z in points_yz:
            x = math.sqrt(max(0.0, surface_r * surface_r - y * y)) + offset
            verts.append((x, y, z))
    n = len(points_yz)
    faces = [tuple(range(n)), tuple(range(n, 2 * n))[::-1]]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def arrow_cylinder_x_side(name, points_yz, surface_r, z_ref, offset0, offset1):
    """Arrow prism following a Y-axis side pipe, on the +X side."""
    verts = []
    for offset in (offset0, offset1):
        for y, z in points_yz:
            x = math.sqrt(max(0.0, surface_r * surface_r - (z - z_ref) ** 2)) + offset
            verts.append((x, y, z))
    n = len(points_yz)
    faces = [tuple(range(n)), tuple(range(n, 2 * n))[::-1]]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
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
    body = revolve_solid('Separator body', config['body_profile'], seg)
    assign_tile(body, 1, smooth='all', scale=0.62, uv='z')

    for seam in config['seam_welds']:
        bead = rope_ring_z(seam['name'], seam['z'], seam['major'], seam['minor'],
                           twists=seam['twists'], major_segments=480)
        assign_tile(bead, 2, smooth='all', scale=0.14, uv='z')

    # Top pipe and eight-bolt flange with real through bores.
    tp = config['top_pipe']
    pipe, _ = tube('Top pipe', (0, 0, tp['z0']), (0, 0, tp['z1']),
                   tp['outer_r'], tp['bore_r'], 64)
    assign_tile(pipe, 0, smooth='all', scale=0.40, uv='z')
    tf = config['top_flange']
    weld = torus('Top pipe weld', (0, 0, tf['weld_z']), (0, 0, 1),
                 tf['weld_major'], tf['weld_minor'], 64, 10)
    assign_tile(weld, 2, smooth='all', scale=0.10, uv='z')
    liner_top, _ = tube('Bore liner top', (0, 0, tp['z0']), (0, 0, tp['z1']),
                        tp['bore_r'], tp['bore_r'] - .0008, 96)
    assign_special(liner_top, 'dark', boxed=True, smooth='all', scale=0.10)

    zc = (tf['z0'] + tf['z1']) / 2
    flange = along('Top flange', (0, 0, tf['z0']), (0, 0, tf['z1']),
                   tf['outer_r'], 128)
    apply_bevel(flange, .0012, 2)
    cutters = [cutter_cylinder(tf['bore_r'], (0, 0, tf['z0'] - .02),
                               (0, 0, tf['z1'] + .02), 64)]
    for i in range(tf['bolt_holes']):
        a = math.radians(tf['phase_deg'] + i * 360 / tf['bolt_holes'])
        x = tf['bolt_circle_r'] * math.cos(a)
        y = tf['bolt_circle_r'] * math.sin(a)
        cutters.append(cutter_cylinder(tf['bolt_hole_r'],
                                       (x, y, tf['z0'] - .02), (x, y, tf['z1'] + .02), 32))
    boolean_cut(flange, cutters, 'Top flange bore and bolt holes')
    assign_tile(flange, 0, boxed=True, smooth='side', axis=(0, 0, 1), scale=0.35)

    for tag, z0f, z1f in (('top', tf['z1'], tf['z1'] + tf['raised_face_thickness']),
                          ('bottom', tf['z0'] - tf['raised_face_thickness'], tf['z0'])):
        raised, _ = tube(f'Top raised face {tag}', (0, 0, z0f), (0, 0, z1f),
                         tf['raised_face_outer_r'], tf['bore_r'], 96)
        apply_bevel(raised, .0006, 2)
        assign_tile(raised, 0, boxed=True, smooth='all', scale=0.25)
    hub, _ = tube('Top flange hub', (0, 0, tf['hub_z0']), (0, 0, tf['hub_z1']),
                  tf['hub_r'], tp['outer_r'], 96)
    assign_tile(hub, 0, smooth='all', scale=0.30, uv='z')

    # Small vent/plug boss on the upper dome shoulder.
    ve = config['vent']
    va = math.radians(ve['azimuth_deg'])
    vx, vy = ve['radius'] * math.cos(va), ve['radius'] * math.sin(va)
    collar = torus('Vent collar', (vx, vy, ve['collar_z']), (0, 0, 1),
                   ve['collar_r'], ve['collar_minor'], 40, 8)
    assign_tile(collar, 2, boxed=True, smooth='all')
    stub = along('Vent stub', (vx, vy, ve['z0']), (vx, vy, ve['z1']),
                 ve['stub_r'], 32)
    assign_tile(stub, 0, boxed=True, smooth='side', axis=(0, 0, 1), scale=0.20)
    hexcap = along('Vent hex cap', (vx, vy, ve['hex_z0']), (vx, vy, ve['hex_z1']),
                   ve['hex_r'], 6, vertices=6)
    assign_tile(hexcap, 0, boxed=True, smooth='flat', scale=0.12)

    # Side nozzle along +Y with an eight-bolt flange and a small round port.
    sn = config['side_nozzle']
    z_noz = sn['z']
    pipe, _ = tube('Side pipe', (0, sn['pipe_y0'], z_noz), (0, sn['pipe_y1'], z_noz),
                   sn['pipe_outer_r'], sn['pipe_bore_r'], 64)
    assign_tile(pipe, 0, smooth='all', scale=0.42, uv='y', z_center=z_noz)
    collar, _ = tube('Side nozzle root collar',
                     (0, sn['root_collar_y0'], z_noz),
                     (0, sn['root_collar_y1'], z_noz),
                     sn['root_collar_outer_r'], sn['root_collar_inner_r'], 64)
    assign_tile(collar, 2, smooth='all', scale=0.25, uv='y', z_center=z_noz)
    weld = rope_ring_y('Side nozzle weld', sn['weld_y'], z_noz,
                       sn['weld_major'], sn['weld_minor'],
                       twists=sn['weld_twists'], major_segments=384)
    assign_tile(weld, 2, smooth='all', scale=0.10, uv='y', z_center=z_noz)
    liner_side, _ = tube('Bore liner side', (0, sn['pipe_y0'], z_noz),
                         (0, sn['pipe_y1'], z_noz),
                         sn['pipe_bore_r'], sn['pipe_bore_r'] - .0008, 96)
    assign_special(liner_side, 'dark', boxed=True, smooth='all', scale=0.10)

    flange = disc_y('Side flange', sn['flange_y0'], sn['flange_y1'],
                    sn['flange_outer_r'], 128, z_center=z_noz)
    apply_bevel(flange, .0012, 2)
    cutters = [cutter_cylinder(sn['flange_bore_r'],
                               (0, sn['flange_y0'] - .03, z_noz),
                               (0, sn['flange_y1'] + .03, z_noz), 64)]
    for i in range(sn['bolt_holes']):
        a = math.radians(sn['phase_deg'] + i * 360 / sn['bolt_holes'])
        x = sn['bolt_circle_r'] * math.cos(a)
        z = z_noz + sn['bolt_circle_r'] * math.sin(a)
        cutters.append(cutter_cylinder(sn['bolt_hole_r'],
                                       (x, sn['flange_y0'] - .03, z),
                                       (x, sn['flange_y1'] + .03, z), 32))
    boolean_cut(flange, cutters, 'Side flange bore and bolt holes')
    assign_tile(flange, 0, boxed=True, smooth='side', axis=(0, 1, 0), scale=0.35)

    for tag, y0f, y1f in (('inner', sn['flange_y0'] - sn['raised_face_thickness'],
                           sn['flange_y0']),
                          ('outer', sn['flange_y1'],
                           sn['flange_y1'] + sn['raised_face_thickness'])):
        raised, _ = tube(f'Side raised face {tag}', (0, y0f, z_noz), (0, y1f, z_noz),
                         sn['raised_face_outer_r'], sn['raised_face_inner_r'], 96)
        apply_bevel(raised, .0006, 2)
        assign_tile(raised, 0, boxed=True, smooth='all', scale=0.25)

    port = along('Side port boss', (sn['port_x0'], sn['port_y'], z_noz),
                 (sn['port_x1'], sn['port_y'], z_noz), sn['port_r'], 40)
    apply_bevel(port, .0010, 2)
    boolean_cut(port, [cutter_cylinder(sn['port_socket_r'],
                                       (sn['port_x1'] - .002, sn['port_y'], z_noz),
                                       (sn['port_x1'] + .010, sn['port_y'], z_noz), 24)],
                'Side port socket')
    assign_tile(port, 0, boxed=True, smooth='side', axis=(1, 0, 0), scale=0.15)
    ring = torus('Side port ring', (sn['port_x1'] - .0005, sn['port_y'], z_noz),
                 (1, 0, 0), sn['port_ring_major'], sn['port_ring_minor'], 40, 8)
    assign_tile(ring, 0, boxed=True, smooth='all', scale=0.10)

    # Curved nameplate with an etched inset and rivets (text deliberately omitted).
    np_ = config['nameplate']
    a_half = math.radians(np_['half_angle_deg'])
    plate = curved_sector('Nameplate', np_['r_in'], np_['r_out'],
                          np_['z0'], np_['z1'], -a_half, a_half, 18)
    assign_special(plate, 'polished', boxed=True, smooth='all', scale=0.20)
    inset = curved_sector('Nameplate inset', np_['inset_r0'], np_['inset_r1'],
                          np_['inset_z0'], np_['inset_z1'],
                          -math.radians(np_['inset_half_angle_deg']),
                          math.radians(np_['inset_half_angle_deg']), 16)
    assign_special(inset, 'inset', boxed=True, smooth='all', scale=0.12)
    for i, (ra, rz) in enumerate(((np_['rivet_angle_deg'], np_['rivet_z'][0]),
                                  (-np_['rivet_angle_deg'], np_['rivet_z'][0]),
                                  (np_['rivet_angle_deg'], np_['rivet_z'][1]),
                                  (-np_['rivet_angle_deg'], np_['rivet_z'][1]))):
        a = math.radians(ra)
        start = (np_['rivet_r0'] * math.cos(a), np_['rivet_r0'] * math.sin(a), rz)
        end = (np_['rivet_r1'] * math.cos(a), np_['rivet_r1'] * math.sin(a), rz)
        rivet = along(f'Nameplate rivet {i + 1}', start, end,
                      np_['rivet_radius'], 16)
        assign_special(rivet, 'polished', boxed=True, smooth='all', scale=0.05)

    # Observed red painted flow arrows, conforming to the pipe surfaces.
    ar = config['arrows']['top']
    top_points = [(-ar['y_half'], ar['z0']), (ar['y_half'], ar['z0']),
                  (ar['y_half'], ar['z_shaft']), (ar['y_head_half'], ar['z_shaft']),
                  (0.0, ar['z_tip']), (-ar['y_head_half'], ar['z_shaft']),
                  (-ar['y_half'], ar['z_shaft'])]
    top_arrow = arrow_cylinder_x('Top red arrow', top_points, tp['outer_r'], 0.0,
                                 0.0002, 0.0002 + ar['thickness'])
    assign_special(top_arrow, 'red', boxed=True, smooth='flat', scale=0.08)
    sa = config['arrows']['side']
    side_points = [(sa['y_tail'], z_noz - sa['z_half']),
                   (sa['y_tail'], z_noz + sa['z_half']),
                   (sa['y_shaft'], z_noz + sa['z_half']),
                   (sa['y_shaft'], z_noz + sa['z_head_half']),
                   (sa['y_tip'], z_noz),
                   (sa['y_shaft'], z_noz - sa['z_head_half']),
                   (sa['y_shaft'], z_noz - sa['z_half'])]
    side_arrow = arrow_cylinder_x_side('Side red arrow', side_points,
                                       sn['pipe_outer_r'], z_noz,
                                       0.0002, 0.0002 + sa['thickness'])
    assign_special(side_arrow, 'red', boxed=True, smooth='flat', scale=0.08)

    # Lower skirt and dished bottom head.
    sk = config['skirt']
    skirt = revolve_ring('Support skirt',
                         [(sk['r_in'], sk['z0']), (sk['r_out'], sk['z0']),
                          (sk['r_out'], sk['z1']), (sk['r_in'], sk['z1'])], seg)
    assign_tile(skirt, 1, smooth='all', scale=0.45, uv='z')
    bh = config['bottom_head']
    head_profile = [(0.0, bh['z_center'])]
    for i in range(1, 25):
        a = (math.pi / 2) * i / 24
        head_profile.append((bh['r'] * math.sin(a),
                             bh['z_center'] + (bh['z_edge'] - bh['z_center']) * (1 - math.cos(a))))
    head_profile.append((0.0, bh['z_edge']))
    head = revolve_solid('Bottom head', head_profile, seg)
    assign_tile(head, 1, smooth='all', scale=0.40, uv='z')

    # Four radial legs with gussets, beveled foot pads and real slotted holes.
    lg = config['legs']
    for index, az in enumerate(lg['azimuths_deg'], start=1):
        azimuth = math.radians(az)
        leg = box_at(f'Support leg {index}',
                     (lg['r_center'], 0, (lg['z0'] + lg['z1']) / 2),
                     (lg['thickness'] / 2, lg['width'] / 2,
                      (lg['z1'] - lg['z0']) / 2))
        apply_bevel(leg, .0012, 2)
        assign_tile(leg, 1, boxed=True, smooth='side', axis=(1, 0, 0), scale=0.30)
        leg.rotation_euler = (0, 0, azimuth)
        gu = lg['gusset']
        prism = radial_prism(f'Leg gusset {index}',
                             [(-gu['u_half_bottom'], gu['z0']),
                              (gu['u_half_bottom'], gu['z0']),
                              (gu['u_half_top'], gu['z1']),
                              (-gu['u_half_top'], gu['z1'])],
                             None, gu['r0'], gu['r1'], azimuth)
        assign_tile(prism, 2, boxed=True, smooth='flat', scale=0.25)
        pad = lg['pad']
        pad_obj = box_at(f'Foot pad {index}',
                         (lg['r_center'], 0, (pad['z0'] + pad['z1']) / 2),
                         (pad['radial'] / 2, pad['tangential'] / 2,
                          (pad['z1'] - pad['z0']) / 2))
        apply_bevel(pad_obj, pad['fillet'], 3)
        slot_r = lg['r_center'] + pad.get('slot_offset_radial', 0.0)
        slot = cutter_cylinder(1.0, (slot_r, 0, pad['z0'] - .02),
                               (slot_r, 0, pad['z1'] + .02), 48)
        slot.scale = (pad['slot_radial'] / 2, pad['slot_tangential'] / 2, 1.0)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = slot
        slot.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.context.view_layer.update()
        boolean_cut(pad_obj, [slot], f'Foot pad {index} slot')
        assign_tile(pad_obj, 3, boxed=True, smooth='side', axis=(0, 0, 1), scale=0.18)
        pad_obj.rotation_euler = (0, 0, azimuth)

    # Dark automatic drain under the bottom head.
    dr = config['drain']
    neck, _ = tube('Drain neck', (0, 0, dr['neck_z0']), (0, 0, dr['neck_z1']),
                   dr['neck_r_out'], dr['neck_r_in'], 48)
    assign_tile(neck, 1, smooth='all', scale=0.16, uv='z')
    mount = along('Drain mount flange', (0, 0, dr['mount_flange_z0']),
                  (0, 0, dr['mount_flange_z1']), dr['mount_flange_r'], 64)
    apply_bevel(mount, .0010, 2)
    assign_tile(mount, 1, boxed=True, smooth='side', axis=(0, 0, 1), scale=0.15)
    for i, sy in enumerate((-1, 1), start=1):
        nut = along(f'Drain mount nut {i}', (0, sy * dr['mount_nut_y'], dr['mount_nut_z0']),
                    (0, sy * dr['mount_nut_y'], dr['mount_nut_z1']),
                    dr['mount_nut_r'], 6, vertices=6)
        assign_tile(nut, 0, boxed=True, smooth='flat', scale=0.06)

    body_center = (0, 0, dr['body_center_z'])
    body_half = (dr['body_half_x'], dr['body_half_y'], dr['body_half_z'])
    drain_body = box_at('Drain body', body_center, body_half)
    apply_bevel(drain_body, dr['body_bevel'], 4)
    assign_special(drain_body, 'drain', boxed=True, smooth='all', scale=0.14)
    for i, sy in enumerate((-1, 1), start=1):
        cap = box_at(f'Drain end cap {i}',
                     (0, sy * dr['end_cap_y'], dr['body_center_z']),
                     (dr['end_cap_half_x'], dr['end_cap_half_y'], dr['end_cap_half_z']))
        apply_bevel(cap, .0012, 2)
        assign_tile(cap, 1, boxed=True, smooth='all', scale=0.10)
    for i, bz in enumerate(dr['band_z'], start=1):
        band = box_at(f'Drain band {i}', (dr['band_x'], 0, bz),
                      (.0008, dr['band_half_y'], .0022))
        assign_special(band, 'dark', boxed=True, smooth='flat', scale=0.08)

    ring, _ = tube('Drain sight ring', (dr['sight_ring_x0'], 0, dr['body_center_z']),
                   (dr['sight_ring_x1'], 0, dr['body_center_z']),
                   dr['sight_ring_outer_r'], dr['sight_ring_inner_r'], 64)
    apply_bevel(ring, .0006, 2)
    assign_special(ring, 'gasket', boxed=True, smooth='all', scale=0.08)
    glass = along('Drain sight glass', (dr['sight_glass_x0'], 0, dr['body_center_z']),
                  (dr['sight_glass_x1'], 0, dr['body_center_z']),
                  dr['sight_glass_r'], 48)
    assign_special(glass, 'glass', boxed=True, smooth='all', scale=0.06)
    center = along('Drain brass center', (dr['brass_center_x0'], 0, dr['body_center_z']),
                   (dr['brass_center_x1'], 0, dr['body_center_z']),
                   dr['brass_center_r'], 32)
    apply_bevel(center, .0008, 2)
    boolean_cut(center, [cutter_cylinder(dr['brass_socket_r'],
                                         (dr['brass_center_x1'] - dr['brass_socket_depth'], 0,
                                          dr['body_center_z']),
                                         (dr['brass_center_x1'] + .01, 0, dr['body_center_z']), 24)],
                'Drain brass socket')
    assign_special(center, 'brass', boxed=True, smooth='side', axis=(1, 0, 0), scale=0.06)
    lever = box_at('Drain brass lever', dr['lever_center'], dr['lever_half'])
    apply_bevel(lever, .0006, 2)
    assign_special(lever, 'brass', boxed=True, smooth='flat', scale=0.04)

    # Brass/blue spigot below the sight glass with a real through bore.
    sx = dr['spigot_x']
    nut = along('Drain spigot nut', (sx, 0, dr['spigot_nut_z0']),
                (sx, 0, dr['spigot_nut_z1']), dr['spigot_nut_r'], 6, vertices=6)
    boolean_cut(nut, [cutter_cylinder(dr['blue_tube_bore_r'],
                                      (sx, 0, dr['spigot_nut_z0'] - .01),
                                      (sx, 0, dr['spigot_nut_z1'] + .01), 24)],
                'Drain spigot nut bore')
    assign_special(nut, 'brass', boxed=True, smooth='flat', scale=0.05)
    cone = revolve_ring('Drain spigot cone',
                        [(dr['blue_tube_bore_r'], dr['spigot_cone_z0']),
                         (dr['spigot_cone_r1'], dr['spigot_cone_z0']),
                         (dr['spigot_cone_r0'], dr['spigot_cone_z1']),
                         (dr['blue_tube_bore_r'], dr['spigot_cone_z1'])], 32)
    cone.location = (sx, 0, 0)
    assign_special(cone, 'brass', boxed=True, smooth='all', scale=0.05)
    collar = along('Drain blue collar', (sx, 0, dr['blue_collar_z0']),
                   (sx, 0, dr['blue_collar_z1']), dr['blue_collar_r'], 32)
    boolean_cut(collar, [cutter_cylinder(dr['blue_tube_bore_r'],
                                         (sx, 0, dr['blue_collar_z0'] - .01),
                                         (sx, 0, dr['blue_collar_z1'] + .01), 24)],
                'Drain blue collar bore')
    assign_special(collar, 'blue', boxed=True, smooth='side', axis=(0, 0, 1), scale=0.05)
    blue_tube, _ = tube('Drain blue tube', (sx, 0, dr['blue_tube_z0']),
                        (sx, 0, dr['blue_tube_z1']),
                        dr['blue_tube_r'], dr['blue_tube_bore_r'], 32)
    assign_special(blue_tube, 'blue', boxed=True, smooth='all', scale=0.05)

    ex, ey, ez = dr['elbow_center']
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=dr['elbow_r'],
                                         location=(ex, ey, ez))
    elbow = bpy.context.object
    elbow.name = 'Drain elbow joint'
    assign_special(elbow, 'gasket', boxed=True, smooth='all', scale=0.05)
    hose = along('Drain elbow hose', (ex, ey, ez), (ex, dr['elbow_hose_y'], ez),
                 dr['elbow_r'] * .80, 24)
    assign_special(hose, 'gasket', boxed=True, smooth='side', axis=(0, 1, 0), scale=0.05)
    drop = along('Drain elbow drop', (ex, ey, ez), (ex, ey, dr['elbow_drop_z']),
                 dr['elbow_r'] * .80, 24)
    assign_special(drop, 'gasket', boxed=True, smooth='side', axis=(0, 0, 1), scale=0.05)
    st = along('Drain steel tube', (dr['steel_tube_center'][0], dr['steel_tube_center'][1],
                                    dr['steel_tube_z0']),
               (dr['steel_tube_center'][0], dr['steel_tube_center'][1],
                dr['steel_tube_z1']), dr['steel_tube_r'], 16)
    assign_special(st, 'polished', boxed=True, smooth='side', axis=(0, 0, 1), scale=0.04)

    # Thin square rod cage around the drain.
    fr = config['frame']
    cx, cy = fr['corner_x'], fr['corner_y']
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)), start=1):
        post = along(f'Drain frame post {i}', (sx * cx, sy * cy, fr['post_z0']),
                     (sx * cx, sy * cy, fr['post_z1']), fr['post_r'], 12)
        assign_special(post, 'polished', boxed=True, smooth='side', axis=(0, 0, 1), scale=0.05)
    for i, rz in enumerate(fr['rail_z'], start=1):
        for j, sy in enumerate((-1, 1), start=1):
            rail = along(f'Drain frame X rail {i}-{j}', (-cx, sy * cy, rz),
                         (cx, sy * cy, rz), fr['rail_r'], 12)
            assign_special(rail, 'polished', boxed=True, smooth='side', axis=(1, 0, 0), scale=0.05)
        for j, sx in enumerate((-1, 1), start=1):
            rail = along(f'Drain frame Y rail {i}-{j}', (sx * cx, -cy, rz),
                         (sx * cx, cy, rz), fr['rail_r'], 12)
            assign_special(rail, 'polished', boxed=True, smooth='side', axis=(0, 1, 0), scale=0.05)


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

atlas_materials()
build()
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
Z_SHIFT = -center.z
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

if not request['draft']:
    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .15), center, span)
        draw(scene, render_dir / f'turn-{i:02d}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    zs = Z_SHIFT
    detail_cameras = (
        ('top-flange', (0, 0, 0.984 + zs), (1, -0.65, -0.28), .21),
        ('top-face', (0, 0, 0.990 + zs), (0.3, -0.3, 1), .22),
        ('vent', (0.0368, 0.0368, 0.902 + zs), (1, -1, 0.6), .05),
        ('side-flange', (0, 0.215, 0.8107 + zs), (0.35, 1, 0.02), .22),
        ('side-port', (0.052, 0.132, 0.8107 + zs), (1, 0.25, 0.05), .075),
        ('quiver', (0, 0, 0.0), (0, 0, 0), .01),
        ('nameplate', (0.081, 0, 0.590 + zs), (1, -0.35, 0.10), .13),
        ('upper-seams', (0, -0.095, 0.815 + zs), (0.8, -0.8, 0.22), .32),
        ('lower-seams', (0, -0.095, 0.245 + zs), (0.8, -0.8, 0.25), .30),
        ('leg-gusset', (0.075, -0.075, 0.205 + zs), (1, -1, 0.18), .13),
        ('foot', (0.0912, 0.0912, 0.026 + zs), (1, 0.2, 0.8), .10),
        ('drain', (0, 0, 0.089 + zs), (1, -0.8, 0.05), .17),
        ('drain-mount', (0, 0, 0.130 + zs), (1, -0.55, 0.18), .09),
        ('drain-cage', (0.033, -0.015, 0.110 + zs), (1, -0.75, 0.10), .13),
        ('drain-bottom', (0.030, 0, 0.050 + zs), (1, -0.4, -0.25), .07),
        ('surface', (0.06, -0.04, 0.400 + zs), (1, -0.55, 0.25), .20),
    )
    for name, target, direction, scale in detail_cameras:
        if name == 'quiver':
            continue
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('top-flange-only', lambda n: n.startswith('Top flange')
         or n.startswith('Top pipe') or n.startswith('Top raised face'),
         (0, 0, 1), (0, 0, 0.984 + zs), .36),
        ('side-flange-only', lambda n: n.startswith('Side flange')
         or n.startswith('Side pipe') or n.startswith('Side raised face'),
         (0, 1, .02), (0, 0.22, 0.8107 + zs), .38),
        ('foot-slot-only', lambda n: n == 'Foot pad 1',
         (.45, .45, -1), (0.0912, 0.0912, 0.007 + zs), .085),
        ('drain-bore-only', lambda n: n in {'Drain spigot nut', 'Drain spigot cone',
                                            'Drain blue collar', 'Drain blue tube'},
         (0, 0, -1), (0.030, 0, 0.040 + zs), .065),
        ('body-only', lambda n: n == 'Separator body',
         (1, -1, .25), (0, 0, 0), 0.95),
    )
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
        'texture_size': 2048, 'brass_texture_size': 1024,
        'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'render_device': render_device,
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': 'original photo proportions; six-view sheet is an uncalibrated visual-match reference',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': ('Front camera at +X (the source-photo direction), left at -Y, '
                      'right at +Y; the side nozzle is along +Y and the drain face is +X.'),
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
        'isolated_checks': {'top_flange_interior_holes_expected': 9,
                            'side_flange_interior_holes_expected': 9,
                            'foot_slot_expected': 1, 'drain_bore_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
