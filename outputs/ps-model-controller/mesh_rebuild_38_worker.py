"""Blender worker for the review-only horizontal pressure-tank candidate (item 38).

Shape basis: the original clean/rust photograph is the primary source for
proportions, fittings and local paint/rust appearance.  The user six-view
collage is an uncalibrated visual-match product render of the same horizontal
red pressure-tank family; it is used for hidden/back-side topology and as a
diagnostic comparison only.

The candidate is a part-based rebuild instead of a smoothed single-mesh repair:
three shell pieces (cylinder + two dished heads), two circumferential weld
beads, four folded support brackets with real bolt holes, top regulator/filter
assembly with a black adjusting knob and open coupler socket, top brass safety
valve, two capped ports on the right head, bottom drain valve with an inferred
outlet bore and black handle, a recreated yellow label, and separate brass,
steel and black-rubber PBR materials.  It never replaces the library mesh, is
manual-only and never enters automatic placement or final compositing.
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


def tile_materials():
    names = {0: '38 clean red enamel', 1: '38 weathered red enamel',
             2: '38 stained red enamel', 3: '38 rusted red enamel'}
    for tile, name in names.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')
        color = image_node(nodes, 'paint_base')
        metal = image_node(nodes, 'paint_mr', True)
        normal = image_node(nodes, 'paint_normal', True)
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        links.new(metal.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = {0: .10, 1: .13, 2: .15, 3: .22}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        material_for_tile[tile] = material

    set_material('brass', 'brass', .10)
    set_material('steel', 'steel', .12)
    set_material('black', 'black', .20)
    set_material('label', 'label', .03)

    def plain(key, color, metallic, roughness):
        material = bpy.data.materials.new(key)
        material.use_nodes = True
        node = material.node_tree.nodes.get('Principled BSDF')
        node.inputs['Base Color'].default_value = (*color, 1)
        node.inputs['Metallic'].default_value = metallic
        node.inputs['Roughness'].default_value = roughness
        special_materials[key] = material

    plain('bore', (.040, .043, .046), .75, .42)


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


def cyl_uv_y(obj, tile, y0, y1, center_z):
    """Cylindrical UV about the Y tank axis, using a paint atlas tile."""
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    length = max(1e-9, y1 - y0)
    for poly in mesh.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.z - center_z, co.x))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            delta = (angle - ca + math.pi) % TAU - math.pi
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv[loop_index].uv = tile_uv(tile, (ca + delta) / TAU + .5,
                                        (co.y - y0) / length)


def label_uv_y(obj, y0, y1, a0, a1, center_z):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='LabelUV')
    uv = obj.data.uv_layers.active.data
    mesh = obj.data
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            angle = math.atan2(co.z - center_z, co.x)
            u = (co.y - y0) / max(1e-9, y1 - y0)
            v = (angle - a0) / max(1e-9, a1 - a0)
            uv[loop_index].uv = (min(1.0, max(0.0, u)),
                                min(1.0, max(0.0, v)))


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


def assign_tile(obj, tile, *, smooth='all', axis=None, scale=0.55, uv='box',
                y0=None, y1=None, center_z=None):
    obj.data.materials.clear()
    obj.data.materials.append(material_for_tile[tile])
    if uv == 'y':
        cyl_uv_y(obj, tile, y0, y1, center_z)
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


def assign_two(obj, outer_key, inner_key, inner_faces, *, smooth='all',
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


def revolve_solid_y(name, profile, segments, center_z):
    """Closed solid revolved about the Y axis at x=0, z=center_z.

    profile entries are (r, y); r=0 entries create flat cap fans.
    """
    verts, rings = [], []
    for r, y in profile:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, y, center_z))
        else:
            ids = []
            for s in range(segments):
                a = TAU * s / segments
                ids.append(len(verts))
                verts.append((r * math.cos(a), y, center_z + r * math.sin(a)))
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


def revolve_axis(name, profile, start, direction, segments):
    """Closed solid revolved about an arbitrary axis; profile entries (r,t)."""
    start = Vector(start)
    direction = Vector(direction).normalized()
    u = direction.orthogonal().normalized()
    v = direction.cross(u).normalized()
    verts, rings = [], []
    for r, t in profile:
        center = start + direction * t
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append(tuple(center))
        else:
            ids = []
            for s in range(segments):
                a = TAU * s / segments
                ids.append(len(verts))
                verts.append(tuple(center + u * (r * math.cos(a)) +
                                   v * (r * math.sin(a))))
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


def plate_x(name, yz_profile, x_center, thickness):
    """Closed plate thin in X, extruded between x_center +/- thickness/2."""
    xs = (x_center - thickness / 2, x_center + thickness / 2)
    verts = []
    for x in xs:
        for y, z in yz_profile:
            verts.append((x, y, z))
    n = len(yz_profile)
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def curved_strap(name, y_center, width, r_in, r_out, th0, th1, center_z,
                 segments):
    y0, y1 = y_center - width / 2, y_center + width / 2
    verts, sections = [], []
    for i in range(segments + 1):
        a = th0 + (th1 - th0) * i / segments
        ca, sa = math.cos(a), math.sin(a)
        ids = []
        for r, y in ((r_in, y0), (r_out, y0), (r_out, y1), (r_in, y1)):
            ids.append(len(verts))
            verts.append((r * ca, y, center_z + r * sa))
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


def curved_label(name, y0, y1, r0, r1, a0, a1, center_z, segments):
    verts, sections = [], []
    for i in range(segments + 1):
        a = a0 + (a1 - a0) * i / segments
        ca, sa = math.cos(a), math.sin(a)
        ids = []
        for r, y in ((r0, y0), (r0, y1), (r1, y1), (r1, y0)):
            ids.append(len(verts))
            verts.append((r * ca, y, center_z + r * sa))
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
def head_profile(radius, apex_y, junction_y, depth, samples=26):
    """Elongated-ellipsoidal dished head profile from apex to junction."""
    sign = 1 if apex_y > junction_y else -1
    pts = []
    for i in range(samples + 1):
        t = i / samples
        y = apex_y + (junction_y - apex_y) * t
        q = (y - junction_y) / (depth * sign)
        q = max(-1.0, min(1.0, q))
        r = radius * math.sqrt(max(0.0, 1.0 - q * q))
        pts.append((r, y))
    return pts


def build():
    seg = config['radial_segments']
    tank = config['tank']
    R = tank['radius']
    cz = tank['center_z']
    junction = tank['head_junction_y']
    apex = tank['total_half_length']
    depth = apex - junction

    # Shell: cylinder plus two separate dished heads, split for material
    # variation and crisp weld junctions.  The parts overlap slightly and are
    # closed solids; the internal flat caps are hidden.
    cylinder_start = -tank['cylinder_end_y']
    cylinder_end = tank['cylinder_end_y']
    cylinder = revolve_solid_y('Tank cylinder', [
        (0.0, cylinder_start), (R, cylinder_start),
        (R, cylinder_end), (0.0, cylinder_end)], seg, cz)
    assign_tile(cylinder, 1, smooth='all', uv='y',
                y0=-apex, y1=apex, center_z=cz)
    left_head = revolve_solid_y('Tank head left', head_profile(
        R, -apex, -junction, depth) + [(0.0, -junction)], seg, cz)
    assign_tile(left_head, 0, smooth='all', uv='y',
                y0=-apex, y1=apex, center_z=cz)
    right_head = revolve_solid_y('Tank head right', [(0.0, junction)] +
                                 list(reversed(head_profile(R, apex, junction,
                                                            depth))), seg, cz)
    assign_tile(right_head, 0, smooth='all', uv='y',
                y0=-apex, y1=apex, center_z=cz)

    for tag, sy in (('left', -1), ('right', 1)):
        seam = torus(f'Tank weld seam {tag}',
                     (0, sy * tank['weld_seam_y'], cz), (0, 1, 0),
                     tank['weld_seam_major'], tank['weld_seam_minor'], 96, 10)
        assign_tile(seam, 1, smooth='all', scale=0.18)

    # --- Folded supports: curved saddle strap, two legs and two flanged feet
    sup = config['supports']
    for tag, sy in (('left', -1), ('right', 1)):
        y_center = sy * abs(sup['y_positions'][0])
        strap = curved_strap(
            f'Support {tag} strap', y_center, sup['strap_width'],
            R + sup['strap_inner_offset'], R + sup['strap_outer_offset'],
            math.radians(sup['strap_theta0_deg']),
            math.radians(sup['strap_theta1_deg']), cz,
            sup['strap_segments'])
        apply_bevel(strap, .0014, 2)
        assign_tile(strap, 1, smooth='all', scale=0.20)

        profile = [(y_center + sup['leg_top_half_y'], sup['leg_top_z']),
                   (y_center - sup['leg_top_half_y'], sup['leg_top_z']),
                   (y_center - sup['leg_bottom_half_y'], sup['leg_bottom_z']),
                   (y_center + sup['leg_bottom_half_y'], sup['leg_bottom_z'])]
        for face_tag, sx in (('front', 1), ('back', -1)):
            leg = plate_x(f'Support {tag} {face_tag} leg', profile,
                          sx * sup['leg_x'], sup['leg_thickness'])
            apply_bevel(leg, .0012, 2)
            assign_tile(leg, 1, smooth='flat', scale=0.12)

            foot = box_at(f'Support {tag} {face_tag} foot',
                          (sx * sup['foot_center_x'], y_center,
                           (sup['foot_z0'] + sup['foot_z1']) / 2),
                          (sup['foot_half_x'], sup['foot_half_y'],
                           (sup['foot_z1'] - sup['foot_z0']) / 2))
            apply_bevel(foot, .0015, 2)
            cutter = cutter_cylinder(
                sup['hole_radius'],
                (sx * sup['hole_x'], y_center, sup['foot_z0'] - .010),
                (sx * sup['hole_x'], y_center, sup['foot_z1'] + .010), 40)
            boolean_cut(foot, [cutter], f'Support {tag} {face_tag} foot hole')
            assign_tile(foot, 3, smooth='flat', scale=0.10)

    # --- Top regulator / filter assembly
    reg = config['regulator']
    ry = reg['y']
    boss = along('Regulator boss', (0, ry, reg['boss_z0']),
                 (0, ry, reg['boss_z1']), reg['boss_radius'], 64)
    apply_bevel(boss, .0025, 2)
    assign_tile(boss, 2, smooth='side', axis=(0, 0, 1), scale=0.12)
    bead = torus('Weld bead regulator', (0, ry, 0.525),
                 (0, 0, 1), reg['boss_radius'] + .0015, .0026, 64, 8)
    assign_tile(bead, 3, smooth='all', scale=0.06)
    riser, _ = tube('Regulator riser',
                    (0, ry, reg['riser_z0']),
                    (0, ry, reg['riser_z1']),
                    reg['riser_outer'], reg['riser_inner'], 48)
    assign_tile(riser, 2, smooth='all', scale=0.05)
    union = along('Regulator union', (0, ry, reg['union_z0']),
                  (0, ry, reg['union_z1']), reg['union_radius'], 48,
                  vertices=6)
    apply_bevel(union, .0015, 2)
    assign_special(union, 'steel', smooth='flat', scale=0.05)
    body = box_at('Regulator body', tuple(reg['body_center']),
                  tuple(reg['body_half']))
    apply_bevel(body, reg['body_bevel'], 3)
    assign_special(body, 'steel', smooth='all', scale=0.12)
    front_boss = along('Regulator front boss',
                       (reg['front_boss_x0'], ry, reg['body_center'][2]),
                       (reg['front_boss_x1'], ry, reg['body_center'][2]),
                       reg['front_boss_radius'], 40)
    apply_bevel(front_boss, .0012, 2)
    assign_special(front_boss, 'steel', smooth='side', axis=(1, 0, 0),
                   scale=0.04)
    plug = along('Regulator brass plug',
                 (reg['brass_plug_x0'], ry, reg['body_center'][2]),
                 (reg['brass_plug_x1'], ry, reg['body_center'][2]),
                 reg['brass_plug_radius'], 32)
    apply_bevel(plug, .0008, 2)
    assign_special(plug, 'brass', smooth='side', axis=(1, 0, 0), scale=0.025)

    knob_len = reg['knob_y1'] - reg['knob_y0']
    knob_profile = [(0.0, 0.0), (reg['knob_radius'], 0.0)]
    ribs = 4
    for k in range(ribs):
        t0 = knob_len * (.10 + .78 * k / ribs)
        t1 = knob_len * (.10 + .78 * (k + .55) / ribs)
        knob_profile += [(reg['knob_rib_radius'], t0),
                         (reg['knob_radius'], t1)]
    knob_profile += [(reg['knob_radius'], knob_len), (0.0, knob_len)]
    knob = revolve_axis('Regulator knob', knob_profile,
                        (0, reg['knob_y0'], reg['body_center'][2]),
                        (0, 1, 0), 48)
    assign_special(knob, 'black', smooth='all', scale=0.05)
    stem_len = reg['knob_stem_y1'] - reg['knob_stem_y0']
    stem = revolve_axis('Regulator knob stem', [
        (0.0, 0.0), (reg['knob_stem_radius'], 0.0),
        (reg['knob_stem_radius'], stem_len * .82),
        (reg['knob_stem_radius'] * .72, stem_len), (0.0, stem_len)],
        (0, reg['knob_stem_y0'], reg['body_center'][2]), (0, 1, 0), 36)
    assign_special(stem, 'black', smooth='all', scale=0.04)
    pipe = along('Regulator coupler pipe',
                 (0, reg['coupler_pipe_y0'], reg['body_center'][2]),
                 (0, reg['coupler_pipe_y1'], reg['body_center'][2]),
                 reg['coupler_pipe_radius'], 40)
    apply_bevel(pipe, .0012, 2)
    assign_special(pipe, 'steel', smooth='side', axis=(0, 1, 0), scale=0.04)
    coupler, _ = tube(
        'Regulator coupler',
        (0, reg['coupler_y0'], reg['body_center'][2]),
        (0, reg['coupler_y1'], reg['body_center'][2]),
        reg['coupler_outer'], reg['coupler_inner'], 56)
    assign_special(coupler, 'steel', smooth='all', scale=0.04)
    coupler_liner, _ = tube(
        'Regulator coupler liner',
        (0, reg['coupler_y0'] + .0004, reg['body_center'][2]),
        (0, reg['coupler_y1'] - .0004, reg['body_center'][2]),
        reg['coupler_inner'] - .0002, reg['coupler_inner'] - .0012, 56)
    assign_special(coupler_liner, 'bore', smooth='all', scale=0.03)
    collar = torus('Regulator coupler collar',
                   (0, reg['coupler_collar_y'], reg['body_center'][2]),
                   (0, 1, 0), reg['coupler_collar_major'],
                   reg['coupler_collar_minor'], 48, 8)
    assign_special(collar, 'steel', smooth='all', scale=0.04)

    # --- Top brass safety valve
    saf = config['safety']
    sy = saf['y']
    sboss = along('Safety boss', (0, sy, saf['boss_z0']),
                  (0, sy, saf['boss_z1']), saf['boss_radius'], 48)
    apply_bevel(sboss, .0022, 2)
    assign_tile(sboss, 2, smooth='side', axis=(0, 0, 1), scale=0.10)
    sbead = torus('Weld bead safety', (0, sy, 0.525), (0, 0, 1),
                  saf['boss_radius'] + .0015, .0026, 48, 8)
    assign_tile(sbead, 3, smooth='all', scale=0.05)
    sbody = along('Safety body', (0, sy, saf['body_z0']),
                  (0, sy, saf['body_z1']), saf['body_radius'], 32,
                  vertices=6)
    apply_bevel(sbody, .0012, 2)
    assign_special(sbody, 'brass', smooth='flat', scale=0.04)
    cap_len = saf['cap_z1'] - saf['cap_z0']
    cap_profile = [(0.0, 0.0), (saf['cap_radius'] * .72, 0.0)]
    for k in range(3):
        t0 = cap_len * (.08 + .72 * k / 3)
        t1 = cap_len * (.08 + .72 * (k + .55) / 3)
        cap_profile += [(saf['cap_radius'], t0),
                        (saf['cap_radius'] * .86, t1)]
    cap_profile += [(saf['cap_radius'] * .86, cap_len * .96),
                    (saf['cap_radius'] * .45, cap_len), (0.0, cap_len)]
    scap = revolve_axis('Safety cap', cap_profile,
                        (0, sy, saf['cap_z0']), (0, 0, 1), 48)
    assign_special(scap, 'brass', smooth='all', scale=0.03)
    sni = along('Safety nipple', (0, sy, saf['cap_z1']),
                (0, sy, saf['nipple_z1']), saf['nipple_radius'], 24)
    apply_bevel(sni, .0008, 2)
    assign_special(sni, 'brass', smooth='side', axis=(0, 0, 1), scale=0.02)

    # --- Two capped ports on the right head
    ports = config['ports']
    for tag, pz in (('upper', ports['upper_z']), ('lower', ports['lower_z'])):
        pboss = along(f'Port {tag} boss', (0, ports['boss_y0'], pz),
                      (0, ports['boss_y1'], pz), ports['boss_radius'], 48)
        apply_bevel(pboss, .0018, 2)
        assign_tile(pboss, 2, smooth='side', axis=(0, 1, 0), scale=0.08)
        pbead = torus(f'Weld bead port {tag}',
                      (0, ports['weld_y'], pz), (0, 1, 0),
                      ports['boss_radius'] + .0015, .0024, 48, 8)
        assign_tile(pbead, 3, smooth='all', scale=0.05)
        pcap = along(f'Port {tag} cap', (0, ports['cap_y0'], pz),
                     (0, ports['cap_y1'], pz), ports['cap_radius'], 32,
                     vertices=6)
        apply_bevel(pcap, .0015, 2)
        assign_tile(pcap, 2, smooth='flat', scale=0.05)

    # --- Bottom drain valve
    dr = config['drain']
    dy = dr['y']
    dboss = along('Drain boss', (0, dy, dr['boss_z0']),
                  (0, dy, dr['boss_z1']), dr['boss_radius'], 48)
    apply_bevel(dboss, .0022, 2)
    assign_tile(dboss, 2, smooth='side', axis=(0, 0, 1), scale=0.10)
    dbead = torus('Weld bead drain', (0, dy, 0.149), (0, 0, 1),
                  dr['boss_radius'] + .0015, .0026, 48, 8)
    assign_tile(dbead, 3, smooth='all', scale=0.05)
    dnut = along('Drain nut', (0, dy, dr['nut_z0']), (0, dy, dr['nut_z1']),
                 dr['nut_radius'], 40, vertices=6)
    apply_bevel(dnut, .0013, 2)
    assign_special(dnut, 'steel', smooth='flat', scale=0.04)
    dbody = along('Drain body', (0, dy, dr['body_z0']),
                  (0, dy, dr['body_z1']), dr['body_radius'], 40)
    apply_bevel(dbody, .0015, 2)
    assign_special(dbody, 'brass', smooth='side', axis=(0, 0, 1), scale=0.04)
    for tag, sy2 in (('left', -1), ('right', 1)):
        nipple = along(f'Drain nipple {tag}',
                       (0, dy + sy2 * .016, dr['nipple_z']),
                       (0, dy + sy2 * .034, dr['nipple_z']),
                       0.009, 24, vertices=6)
        apply_bevel(nipple, .0010, 2)
        assign_special(nipple, 'brass', smooth='flat', scale=0.03)
    dout, _ = tube('Drain outlet', (0, dy, dr['outlet_z0']),
                   (0, dy, dr['outlet_z1']),
                   dr['outlet_outer'], dr['outlet_inner'], 32)
    assign_special(dout, 'brass', smooth='all', scale=0.025)
    dliner, _ = tube('Drain outlet liner',
                     (0, dy, dr['outlet_z0'] + .0002),
                     (0, dy, dr['outlet_z1'] - .0002),
                     dr['outlet_inner'] - .00015,
                     dr['outlet_inner'] - .0008, 32)
    assign_special(dliner, 'bore', smooth='all', scale=0.02)
    handle_start = Vector(dr['handle_start'])
    handle_end = Vector(dr['handle_end'])
    handle_dir = (handle_end - handle_start).normalized()
    handle_len = (handle_end - handle_start).length
    handle_profile = [(0.0, 0.0), (dr['handle_radius'], 0.0)]
    for k in range(3):
        t0 = handle_len * (.10 + .72 * k / 3)
        t1 = handle_len * (.10 + .72 * (k + .55) / 3)
        handle_profile += [(dr['handle_rib_radius'], t0),
                           (dr['handle_radius'], t1)]
    handle_profile += [(dr['handle_radius'], handle_len * .92),
                       (dr['handle_radius'] * .55, handle_len),
                       (0.0, handle_len)]
    handle = revolve_axis('Drain handle', handle_profile, handle_start,
                          handle_dir, 36)
    assign_special(handle, 'black', smooth='all', scale=0.04)

    # --- Recreated label on the +X cylinder face
    lb = config['label']
    a_half = math.radians(lb['angle_half_deg'])
    label = curved_label('Nameplate', lb['y0'], lb['y1'], lb['r0'], lb['r1'],
                         -a_half, a_half, lb['center_z'], lb['segments'])
    label.data.materials.clear()
    label.data.materials.append(special_materials['label'])
    label_uv_y(label, lb['y0'], lb['y1'], -a_half, a_half, lb['center_z'])
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
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .48
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, width, height in (
        ('large softbox', (2.6, -3.0, 3.6), 220, 3.0, 3.0),
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

tile_materials()
build()
bpy.context.view_layer.update()

# Bake primitive object transforms into mesh data before export.  Otherwise a
# torus/cylinder rotation would be applied twice when the GLB is re-imported.
for obj in objects:
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
Z_SHIFT = -low.z
for obj in objects:
    obj.data.transform(Matrix.Translation((0, 0, Z_SHIFT)))
    obj.data.update()
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
directions = {'front': (1, 0, .22), 'back': (-1, 0, .22),
              'left': (0, -1, .18), 'right': (0, 1, .18),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 704
samples = 8 if request['draft'] else 56
for name, direction in directions.items():
    cam = camera(scene, direction, center, span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .80)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

zs = Z_SHIFT
groups = (
    ('feet-holes-only', lambda n: n.endswith(' foot'), (0, 0, 1),
     (0, 0, 0.004 + zs), .60),
    ('coupler-bore-only', lambda n: n == 'Regulator coupler', (0, 1, 0),
     (0, -0.250, 0.601 + zs), .085),
    ('drain-outlet-only', lambda n: n == 'Drain outlet', (0, 0, -1),
     (0, 0.200, 0.089 + zs), .030),
    ('support-only', lambda n: n.startswith('Support'), (1, -0.6, .25),
     (0, 0, 0.15 + zs), .80),
    ('regulator-only', lambda n: n.startswith('Regulator'), (2.2, -1.2, .9),
     (0, -0.32, 0.60 + zs), .30),
    ('safety-only', lambda n: n.startswith('Safety'), (2.0, -0.5, 1.0),
     (0, 0.02, 0.57 + zs), .20),
    ('ports-only', lambda n: n.startswith('Port '), (0.8, 2.0, .4),
     (0, 0.50, 0.345 + zs), .26),
    ('drain-only', lambda n: n.startswith('Drain'), (2.0, 1.0, -.4),
     (0, 0.20, 0.11 + zs), .22),
    ('tank-only', lambda n: n.startswith('Tank'), (1, -0.6, .2),
     (0, 0, 0.33 + zs), 1.12),
    ('label-only', lambda n: n == 'Nameplate', (1, 0, .1),
     (0.188, -0.02, 0.35 + zs), .22),
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
        ('regulator', (0, -0.32, 0.60 + zs), (2.2, -1.2, .9), .30),
        ('safety', (0, 0.02, 0.57 + zs), (2.0, -0.5, 1.0), .20),
        ('ports', (0, 0.50, 0.345 + zs), (0.8, 2.0, .4), .26),
        ('drain', (0, 0.20, 0.11 + zs), (2.0, 1.0, -.4), .22),
        ('foot-left', (-0.215, -0.31, 0.05 + zs), (-2.0, -1.0, -.3), .26),
        ('foot-right', (0.215, 0.31, 0.05 + zs), (2.0, 1.0, -.3), .26),
        ('label', (0.188, -0.02, 0.35 + zs), (1, 0, .08), .22),
        ('seam', (0, -0.39, 0.33 + zs), (1, -0.8, .4), .28),
        ('top', (0, -0.05, 0.55 + zs), (0, 0, 1), .55),
        ('bottom', (0, 0, 0.18 + zs), (0, 0, -1), .55),
        ('surface', (0.188, -0.20, 0.35 + zs), (1, -0.6, .2), .18),
        ('front', (0, 0, 0.33 + zs), (1, -0.3, .15), 1.12),
        ('left', (0, 0, 0.33 + zs), (0, -1, .18), .95),
        ('right', (0, 0, 0.33 + zs), (0, 1, .18), .95),
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

    metal_parts = {
        'Safety body', 'Safety cap', 'Safety nipple',
        'Regulator brass plug', 'Regulator union', 'Regulator body',
        'Regulator front boss', 'Regulator coupler pipe',
        'Regulator coupler', 'Regulator coupler collar',
        'Drain nut', 'Drain body', 'Drain nipple left',
        'Drain nipple right', 'Drain outlet',
    }
    def draw_metal(prefix, alt):
        for obj in objects:
            obj.hide_render = obj.name not in metal_parts
        for tag, target, direction, scale in (
                ('reg', (0, -0.32, 0.60 + zs), (2.2, -1.2, .9), .30),
                ('drain', (0, 0.20, 0.11 + zs), (2.0, 1.0, -.4), .22)):
            cam = camera(scene, direction, Vector(target), scale)
            draw(scene, render_dir / f'metal-{tag}-{prefix}.png',
                 resolution=768, samples=24)
            bpy.data.objects.remove(cam, do_unlink=True)
        for obj in objects:
            obj.hide_render = False

    draw_metal('a', False)
    key = bpy.data.objects['large softbox']
    key.location = (-3.4, 2.8, 4.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    draw_metal('b', True)
    cam = camera(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .80)
    draw(scene, render_dir / 'new-metal-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Source-photo framing diagnostic (not an acceptance gate).
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
bpy.context.view_layer.update()
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
    cam = camera(scene, direction, center, span)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .80)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(poly.vertices) - 2)
                         for o in objects for poly in o.data.polygons),
        'texture_size': 2048, 'detail_texture_size': 512,
        'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'render_device': render_device,
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': ('original clean/rust photograph proportions; six-view collage '
                           'is an uncalibrated visual-match product render'),
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': ('Front camera at +X (label/regulator side). The tank axis is Y. '
                      'Left camera at -Y sees the left head and regulator stem; right '
                      'camera at +Y sees the two capped ports and drain.'),
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
        'isolated_checks': {'foot_holes_expected': 4,
                            'coupler_socket_expected': 1,
                            'drain_outlet_bore_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
