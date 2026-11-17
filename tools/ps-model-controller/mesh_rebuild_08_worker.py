"""Blender worker for the review-only separator-filter candidate (item 08).

Shape basis: the original rust photograph for proportions and visible
structure.  The user-supplied six-view sheet is an uncalibrated visual-match
reference of the same visible design; it is used for the hidden/back-side
topology and for comparison, never as measured geometry.  Inferred parts are
listed in the record.  The candidate never replaces the library mesh, is
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
    names = {0: '08 blue paint', 1: '08 weathered blue',
             2: '08 rusted paint', 3: '08 chipped blue'}
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

        color = texture('paint_base')
        metal = texture('paint_mr', True)
        normal = texture('paint_normal', True)
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        links.new(metal.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = {0: .14, 1: .18, 2: .28, 3: .20}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        material_for_tile[tile] = material

    brass = bpy.data.materials.new('08 brass drain')
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
    normal_map.inputs['Strength'].default_value = .20
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    special_materials['brass'] = brass

    dark = bpy.data.materials.new('08 dark bore')
    dark.use_nodes = True
    principled = dark.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.008, .009, .009, 1)
    principled.inputs['Metallic'].default_value = .30
    principled.inputs['Roughness'].default_value = .58
    special_materials['dark'] = dark

    gasket = bpy.data.materials.new('08 black washer')
    gasket.use_nodes = True
    principled = gasket.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.012, .013, .012, 1)
    principled.inputs['Metallic'].default_value = .00
    principled.inputs['Roughness'].default_value = .78
    special_materials['gasket'] = gasket

    glass = bpy.data.materials.new('08 green sight glass')
    glass.use_nodes = True
    principled = glass.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.012, .085, .045, 1)
    principled.inputs['Metallic'].default_value = .28
    principled.inputs['Roughness'].default_value = .09
    special_materials['glass'] = glass


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


def assign_tile(obj, tile, *, boxed=True, smooth='all', axis=None, scale=0.55):
    obj.data.materials.clear()
    obj.data.materials.append(material_for_tile[tile])
    if boxed:
        box_uv(obj, scale)
    remap_uv(obj, tile)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def assign_brass(obj, *, boxed=True, smooth='all', axis=None, scale=0.30):
    obj.data.materials.clear()
    obj.data.materials.append(special_materials['brass'])
    if boxed:
        box_uv(obj, scale)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def assign_special(obj, key, *, boxed=True, smooth='all', axis=None):
    obj.data.materials.clear()
    obj.data.materials.append(special_materials[key])
    if boxed:
        box_uv(obj, .45 if key == 'paint' else .55)
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


def oval_torus(name, center, half_width, half_height, tube_radius,
               major_segments=56, minor_segments=12, axis='X'):
    """Elliptical tube ring in the plane normal to X (windows face +/-X)."""
    center = Vector(center)
    if axis != 'X':
        raise ValueError('only X-axis oval rings are used')
    verts = []
    for i in range(major_segments):
        t = TAU * i / major_segments
        cy, cz = half_width * math.cos(t), half_height * math.sin(t)
        radial = Vector((0.0, math.cos(t) / half_width,
                         math.sin(t) / half_height)).normalized()
        for j in range(minor_segments):
            p = TAU * j / minor_segments
            q = center + Vector((0.0, cy, cz)) \
                + radial * (tube_radius * math.cos(p)) \
                + Vector((tube_radius * math.sin(p), 0.0, 0.0))
            verts.append(tuple(q))
    faces = []
    for i in range(major_segments):
        i1 = (i + 1) % major_segments
        for j in range(minor_segments):
            j1 = (j + 1) % minor_segments
            faces.append((i * minor_segments + j, i * minor_segments + j1,
                          i1 * minor_segments + j1, i1 * minor_segments + j))
    return mesh_object(name, verts, faces)


def elliptic_disc_x(name, x0, x1, half_width, half_height, z_center=0.0,
                    segments=64):
    verts, rings = [], []
    for x in (x0, x1):
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            ids.append(len(verts))
            verts.append((x, half_width * math.cos(a),
                          z_center + half_height * math.sin(a)))
        rings.append(ids)
    a, b = rings
    faces = [tuple(a), tuple(b)[::-1]]
    for s in range(segments):
        s1 = (s + 1) % segments
        faces.append((a[s], a[s1], b[s1], b[s]))
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
    R = config['radius']
    z_top = config['shell_top_z']
    z_bottom = config['shell_bottom_z']
    top_h = config['top_head_depth']
    bottom_h = config['bottom_head_depth']
    seg = config['radial_segments']
    nz = config['nozzle']
    nz_z = nz['z']

    # Main pressure body: shallow dished top head, cylindrical shell and
    # shallow dished bottom head.  It is one closed solid.
    profile = [(0.0, z_top + top_h)]
    for i in range(1, 33):
        theta = (math.pi / 2) * i / 32
        profile.append((R * math.sin(theta), z_top + top_h * math.cos(theta)))
    for i in range(1, 33):
        profile.append((R, z_top + (z_bottom - z_top) * i / 32))
    # The shell loop already ends at (R, z_bottom); start at i=31 so the
    # bottom head does not duplicate that ring (duplicate rings create
    # non-manifold edges after glTF splitting and vertex welding).
    for i in range(31, 0, -1):
        theta = (math.pi / 2) * i / 32
        profile.append((R * math.sin(theta), z_bottom - bottom_h * math.cos(theta)))
    profile.append((0.0, z_bottom - bottom_h))
    body = revolve_solid('Separator body', profile, seg)
    # One material keeps the exported body a single watertight GLB primitive.
    assign_tile(body, 1, boxed=True, smooth='all')

    # Two circumferential rope-like weld beads near the head tangents.
    for name, z in (('Top seam weld', config['top_seam_z']),
                    ('Bottom seam weld', config['bottom_seam_z'])):
        bead = rope_ring_z(name, z, R - .0015, config['seam_minor'],
                           twists=72, major_segments=480)
        assign_tile(bead, 2, boxed=True, smooth='all')

    # Opposed side nozzles with real bored flanges.  Nozzles are along Y;
    # the source photograph is the +X view (as in the library reconstruction).
    for side, tag in ((-1, '-Y'), (1, '+Y')):
        shell_y = side * R
        pipe_end = side * (R + nz['length'] + .002)
        pipe, _ = tube(f'Nozzle {tag} pipe',
                       (0, side * (R - .018), nz_z), (0, pipe_end, nz_z),
                       nz['outer'], nz['bore'], 64)
        assign_tile(pipe, 0, boxed=True, smooth='all')
        sleeve, _ = tube(f'Nozzle {tag} pipe sleeve',
                         (0, side * (R - .016), nz_z), (0, pipe_end, nz_z),
                         nz['sleeve_outer'], nz['sleeve_inner'], 64)
        assign_special(sleeve, 'dark', boxed=True, smooth='all')

        y0 = side * (R + nz['length'])
        y1 = side * (R + nz['length'] + nz['flange_thickness'])
        flange = disc_y(f'Nozzle {tag} flange', y0, y1, nz['flange_outer'], 128,
                        z_center=nz_z)
        apply_bevel(flange, .0012, 2)
        cutters = [cutter_cylinder(nz['flange_bore'],
                                   (0, y0 - .03, nz_z), (0, y1 + .03, nz_z), 64)]
        for i, a in enumerate(nz['bolt_azimuths_deg'], start=1):
            angle = math.radians(a)
            x = nz['bolt_circle_radius'] * math.cos(angle)
            z = nz_z + nz['bolt_circle_radius'] * math.sin(angle)
            cutters.append(cutter_cylinder(nz['bolt_hole_radius'],
                                           (x, y0 - .03, z), (x, y1 + .03, z), 40))
        boolean_cut(flange, cutters, f'Nozzle {tag} flange bore and bolt holes')
        assign_tile(flange, 1, boxed=True, smooth='side', axis=(0, side, 0))

        for face, y0f, y1f in (('inner', min(y0, y1) - nz['raised_face_thickness'],
                                min(y0, y1)),
                               ('outer', max(y0, y1),
                                max(y0, y1) + nz['raised_face_thickness'])):
            raised, _ = tube(f'Nozzle {tag} raised face {face}',
                             (0, y0f, nz_z), (0, y1f, nz_z),
                             nz['raised_face_outer'], nz['raised_face_inner'], 96)
            apply_bevel(raised, .0006, 2)
            assign_tile(raised, 0, boxed=True, smooth='all')

        weld = rope_ring_y(f'Nozzle {tag} weld', shell_y, nz_z,
                           nz['weld_major'], nz['weld_minor'],
                           twists=28, major_segments=384)
        assign_tile(weld, 2, boxed=True, smooth='all')

    # Forged lifting eye with a real through hole, sitting on a saddle pad.
    lug = config['lift_lug']
    zc = lug['eye_center_z']
    pad = revolve_solid('Lift lug pad',
                        [(0.0, lug['pad_z0']), (lug['pad_r0'], lug['pad_z0']),
                         (lug['pad_r0'] - .002, lug['pad_z0'] + .006),
                         (lug['pad_r1'] + .006, lug['pad_z1'] - .004),
                         (lug['pad_r1'], lug['pad_z1']), (0.0, lug['pad_z1'])], 96)
    apply_bevel(pad, .0012, 2)
    assign_tile(pad, 0, boxed=True, smooth='all')
    neck = revolve_solid('Lift lug neck',
                         [(0.0, lug['neck_z0']), (lug['neck_r0'], lug['neck_z0']),
                          (lug['neck_r1'], lug['neck_z1']), (0.0, lug['neck_z1'])], 64)
    assign_tile(neck, 1, boxed=True, smooth='all')
    collar = torus('Lift lug collar', (0, 0, lug['collar_z']), (0, 0, 1),
                   lug['collar_major'], lug['collar_minor'], 48, 10)
    assign_tile(collar, 0, boxed=True, smooth='all')
    eye = torus('Lift lug eye', (0, 0, zc), (1, 0, 0),
                lug['eye_major'], lug['eye_minor'], 56, 16)
    assign_tile(eye, 0, boxed=True, smooth='all')
    band = box_at('Lift lug top band',
                  (0, 0, zc + lug['eye_major'] - .001),
                  (lug['band_half_x'], lug['band_half_y'], lug['band_half_z']))
    apply_bevel(band, .0010, 2)
    assign_tile(band, 1, boxed=True, smooth='flat')

    # Brass automatic drain / sight-glass assembly under the bottom head.
    dr = config['drain']
    boss = along('Drain blue boss', (0, 0, dr['blue_boss_z0']),
                 (0, 0, dr['blue_boss_z1']), dr['blue_boss_r'], 40)
    assign_tile(boss, 3, boxed=True, smooth='side', axis=(0, 0, 1))
    for i in range(dr['thread_rings']):
        z = dr['blue_boss_z0'] - .013 - i * .008
        ring = torus(f'Drain thread ring {i + 1}', (0, 0, z), (0, 0, 1),
                     dr['thread_major'], dr['thread_minor'], 48, 8)
        assign_tile(ring, 3, boxed=True, smooth='all')
    head_weld = torus('Drain head weld', (0, 0, dr['head_apex_z']), (0, 0, 1),
                      dr['weld_major'], dr['weld_minor'], 48, 10)
    assign_tile(head_weld, 2, boxed=True, smooth='all')

    nut = along('Drain brass nut', (0, 0, dr['brass_nut_z0']),
                (0, 0, dr['brass_nut_z1']), dr['brass_nut_r'], 6)
    apply_bevel(nut, .0008, 2)
    assign_brass(nut, boxed=True, smooth='flat')
    washer = along('Drain washer', (0, 0, dr['washer_z0']),
                   (0, 0, dr['washer_z1']), dr['washer_r'], 64)
    assign_special(washer, 'gasket', boxed=True, smooth='side', axis=(0, 0, 1))
    collar = along('Drain knurled collar', (0, 0, dr['collar_z0']),
                   (0, 0, dr['collar_z1']), dr['collar_r'], 96)
    assign_brass(collar, boxed=True, smooth='side', axis=(0, 0, 1))
    zc_rib = (dr['collar_z0'] + dr['collar_z1']) / 2
    for i in range(dr['collar_ribs']):
        a = TAU * i / dr['collar_ribs']
        rib = box_at(f'Drain collar rib {i + 1:02d}', (0, 0, 0),
                     (dr['rib_depth'] / 2, dr['rib_half_tangent'],
                      (dr['collar_z1'] - dr['collar_z0']) / 2 - .0015))
        rib.rotation_euler = (0, 0, a)
        rib.location = (math.cos(a) * (dr['collar_r'] + .0004),
                        math.sin(a) * (dr['collar_r'] + .0004), zc_rib)
        assign_brass(rib, boxed=True, smooth='flat')

    body = revolve_solid('Drain body',
                         [(0.0, dr['body_z0']), (dr['body_top_r'], dr['body_z0']),
                          (dr['body_top_r'], dr['body_z1']),
                          (dr['body_top_r'] - .002, dr['body_round_z'] + .004),
                          (dr['body_neck_r'] + .006, dr['body_round_z']),
                          (dr['body_neck_r'], dr['body_neck_z']),
                          (0.0, dr['body_neck_z'])], 96)
    # The body stays a closed solid.  The spigot/nut bores are open; a dark
    # cap marks where the visible bore ends at the body.  The internal drain
    # passage is not modelled (recorded as unverified).
    assign_brass(body, boxed=True, smooth='all')

    spigot_nut = along('Drain spigot nut', (0, 0, dr['spigot_nut_z0']),
                       (0, 0, dr['spigot_nut_z1']), dr['spigot_nut_r'], 6)
    apply_bevel(spigot_nut, .0005, 2)
    assign_brass(spigot_nut, boxed=True, smooth='flat')
    spigot = along('Drain spigot', (0, 0, dr['spigot_z0']),
                   (0, 0, dr['spigot_z1']), dr['spigot_r'], 40)
    assign_brass(spigot, boxed=True, smooth='side', axis=(0, 0, 1))
    lip = torus('Drain spigot lip', (0, 0, dr['spigot_z1'] + .001), (0, 0, 1),
                dr['spigot_r'], .0016, 40, 8)
    assign_brass(lip, boxed=True, smooth='all')
    bore_cutters = [
        cutter_cylinder(dr['bore_radius'], (0, 0, dr['spigot_z1'] - .03),
                        (0, 0, dr['spigot_nut_z1'] + .01), 32)
        for _ in range(2)]
    boolean_cut(spigot_nut, [bore_cutters[0]], 'Drain spigot nut bore')
    boolean_cut(spigot, [bore_cutters[1]], 'Drain spigot bore')
    bore_end = along('Drain bore end', (0, 0, dr['body_neck_z'] - .0015),
                     (0, 0, dr['body_neck_z'] + .001),
                     dr['bore_radius'] * .98, 32)
    assign_special(bore_end, 'dark', boxed=True, smooth='side', axis=(0, 0, 1))

    # Two oval sight windows (front and back), following the reference image.
    for side, tag in ((1, '+X'), (-1, '-X')):
        bezel = oval_torus(f'Drain window bezel {tag}',
                           (side * .0398, 0, dr['window_z_center']),
                           dr['window_half_width'], dr['window_half_height'],
                           dr['window_tube'], 56, 10)
        assign_brass(bezel, boxed=True, smooth='all')
        backing = elliptic_disc_x(f'Drain window backing {tag}',
                                  side * .0388, side * .0396,
                                  dr['window_half_width'] - .001,
                                  dr['window_half_height'] - .001,
                                  dr['window_z_center'], 64)
        assign_special(backing, 'dark', boxed=True, smooth='flat')
        lens = elliptic_disc_x(f'Drain window lens {tag}',
                               side * .0394, side * .0408,
                               dr['lens_half_width'], dr['lens_half_height'],
                               dr['window_z_center'], 64)
        assign_special(lens, 'glass', boxed=True, smooth='all')


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
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .60
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, width, height in (
        ('large softbox', (2.6, -3.0, 3.6), 230, 3.2, 3.2),
        ('fill softbox', (-3.2, 2.6, 2.2), 120, 3.0, 3.0),
        ('front strip', (3.2, 1.0, 1.3), 120, 1.2, 3.4),
        ('rim high', (-1.2, 3.2, 3.4), 90, 1.4, 2.2),
        ('low fill', (0, -2.6, -.28), 50, 1.0, 1.0)):
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
        draw(scene, render_dir / f'turn-{i:02}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    detail_cameras = (
        ('lift-eye', (0, 0, 0.459 + Z_SHIFT), (1, -0.35, 0.22), .18),
        ('lift-eye-side', (0, 0, 0.459 + Z_SHIFT), (0.25, -1, 0.18), .18),
        ('flange-left', (0, -0.17, 0.22 + Z_SHIFT), (0, -1, 0.10), .34),
        ('flange-right', (0, 0.17, 0.22 + Z_SHIFT), (0, 1, 0.10), .34),
        ('nozzle-weld', (0, -0.105, 0.22 + Z_SHIFT), (0.8, -0.75, 0.25), .18),
        ('top-seam', (0, -0.10, 0.366 + Z_SHIFT), (0.8, -0.8, 0.30), .24),
        ('bottom-seam', (0, -0.10, -0.256 + Z_SHIFT), (0.8, -0.8, 0.30), .24),
        ('drain', (0, 0, -0.415 + Z_SHIFT), (1, -0.8, 0.12), .27),
        ('drain-window', (0.040, 0, -0.415 + Z_SHIFT), (1, -0.18, 0.10), .13),
        ('drain-knurl', (0, 0, -0.362 + Z_SHIFT), (1, -0.55, 0.45), .15),
        ('drain-bottom', (0, 0, -0.478 + Z_SHIFT), (1, -0.55, 0.18), .10),
        ('surface', (0.10, -0.05, 0.05 + Z_SHIFT), (1, -0.55, 0.25), .26),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('flange-bores-only', lambda n: n.startswith('Nozzle '),
         (0, 1, .03), (0, 0, 0.22 + Z_SHIFT), .40),
        ('lift-eye-only', lambda n: n.startswith('Lift lug eye')
         or n.startswith('Lift lug top band'),
         (1, 0, .05), (0, 0, 0.459 + Z_SHIFT), .14),
        # The body and collar are solid (the visible bore is capped at the
        # body), so the isolated check covers the open spigot/nut passage.
        ('drain-bore-only', lambda n: n in {'Drain spigot', 'Drain spigot nut',
                                            'Drain spigot lip'},
         (0, 0, -1), (0, 0, -0.48 + Z_SHIFT), .055),
        ('body-only', lambda n: n == 'Separator body',
         (1, -1, .25), (0, 0, 0), 0.95),
    )
    for label, predicate, direction, target, scale in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'{label}.png', resolution=512, samples=16)
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
# Diagnostic framing: fit the perspective distance to the source-alpha height
# so the silhouette comparison is not dominated by a framing mismatch.
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
    # One measured correction: the projected bbox corners overestimate the
    # rendered silhouette, so render a trial and match the actual alpha height.
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
    # Move the camera laterally (without rotating) to centre the diagnostic
    # silhouette on the source bbox.  This is framing alignment only.
    cam.location.y += source_shift_world
    # The Blender mask rows run bottom-up; the source bbox is top-down.
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
                      'right at +Y; nozzles are along Y and the drain windows face +/-X.'),
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
        'isolated_checks': {'flange_interior_holes_expected': 5,
                            'lift_eye_expected': 1, 'drain_bore_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
