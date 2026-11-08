"""Blender worker for the review-only blue aerosol-can candidate (item 14).

Shape basis: the clean/rust photographs show one used blue Lanju insecticide
aerosol can.  The user six-view sheet is an uncalibrated visual-match sheet of
the same product; it supplies the printed label wrap, the blue over-cap with a
U-shaped spray well and the stamped metal base.  The candidate is a part-based
rebuild (metal body, printed label sleeve, hollow blue cap with a real spray
well and actuator bore, stamped concave base).  It never replaces the library
mesh, is manual-only and never enters automatic placement.
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
can_cfg = config['can']
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


def textured_material(key, prefix, normal_strength):
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    color = image_node(nodes, prefix + '_base')
    mr = image_node(nodes, prefix + '_mr', True)
    normal = image_node(nodes, prefix + '_normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(mr.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = normal_strength
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    materials[key] = material


def make_materials():
    textured_material('label', 'label', .24)
    textured_material('cap', 'cap', .30)
    textured_material('metal', 'metal', .22)
    textured_material('bottom', 'bottom', .18)
    textured_material('dark', 'dark', .20)


def box_uv(obj, scale=0.55):
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


def cyl_uv(obj, z0, z1, u_center=0.5):
    """Cylindrical UV; the wrap seam sits at -X (u=0/1), front +X at u=.5."""
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='UVMap')
    uv = mesh.uv_layers.active.data
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            u = (math.atan2(co.y, co.x) / TAU + u_center) % 1.0
            v = (co.z - z0) / max(1e-9, (z1 - z0))
            uv[loop_index].uv = (u, max(0.0, min(1.0, v)))


def planar_uv_xy(obj, radius):
    """Planar UV for the bottom-facing dish.

    The bottom camera looks from -Z, so its screen right is -X.  Mirroring U
    keeps the stamped code readable in the bottom render.
    """
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='UVMap')
    uv = mesh.uv_layers.active.data
    for loop_index, loop in enumerate(mesh.loops):
        co = mesh.vertices[loop.vertex_index].co
        uv[loop_index].uv = (.5 - co.x / (2 * radius),
                             .5 + co.y / (2 * radius))


def set_smooth(obj, smooth=True, axis_flat=None):
    if axis_flat is None:
        for poly in obj.data.polygons:
            poly.use_smooth = smooth
        return
    axis = Vector(axis_flat).normalized()
    for poly in obj.data.polygons:
        poly.use_smooth = smooth and abs(poly.normal.dot(axis)) < .94


def assign(obj, key, *, uv='box', z0=None, z1=None, smooth=True, axis_flat=None,
           scale=.55):
    obj.data.materials.clear()
    obj.data.materials.append(materials[key])
    if uv == 'label':
        cyl_uv(obj, z0, z1, .5)
    elif uv == 'cyl':
        cyl_uv(obj, z0, z1, .5)
    elif uv == 'planar':
        planar_uv_xy(obj, can_cfg['max_radius'] * 1.04)
    else:
        box_uv(obj, scale)
    set_smooth(obj, smooth, axis_flat)
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


def apply_bevel(obj, width, segments=2, angle=32.0):
    if not width:
        return obj
    bpy.ops.object.select_all(action='DESELECT')
    mod = obj.modifiers.new('Edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def weld(obj, dist=1e-7):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=dist)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def boolean_difference(obj, cutter):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    mod = obj.modifiers.new('Boolean cut', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    mod.use_self = False
    bpy.ops.object.modifier_apply(modifier=mod.name)
    remove_objects([cutter])
    weld(obj)
    obj.data.validate()
    obj.data.update()
    return obj


def remove_objects(objs):
    for obj in objs:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)


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


def torus_mesh(name, center, normal, major, minor, major_segments=160,
               minor_segments=12):
    center = Vector(center)
    normal = Vector(normal).normalized()
    u = normal.orthogonal().normalized()
    v = normal.cross(u)
    verts = []
    for i in range(major_segments):
        a = TAU * i / major_segments
        ring_center = center + (u * math.cos(a) + v * math.sin(a)) * major
        radial = u * math.cos(a) + v * math.sin(a)
        for j in range(minor_segments):
            b = TAU * j / minor_segments
            verts.append(tuple(ring_center + radial * (minor * math.cos(b)) +
                               normal * (minor * math.sin(b))))
    faces = []
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = i * minor_segments + (j + 1) % minor_segments
            c = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            d = ((i + 1) % major_segments) * minor_segments + j
            faces.append((a, b, c, d))
    return mesh_object(name, verts, faces)


def tube_shell(name, r_out, r_in, z0, z1, segments):
    verts = []
    for r in (r_out, r_in):
        for z in (z0, z1):
            for s in range(segments):
                a = TAU * s / segments
                verts.append((r * math.cos(a), r * math.sin(a), z))
    # indices: ring(r,z,s) = (r_idx*2+z_idx)*segments+s
    def idx(ri, zi, s):
        return (ri * 2 + zi) * segments + (s % segments)
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        faces.append((idx(0, 0, s), idx(0, 0, s2), idx(0, 1, s2), idx(0, 1, s)))
        faces.append((idx(1, 0, s), idx(1, 1, s), idx(1, 1, s2), idx(1, 0, s2)))
        faces.append((idx(0, 1, s), idx(1, 1, s), idx(1, 1, s2), idx(0, 1, s2)))
        faces.append((idx(0, 0, s), idx(1, 0, s), idx(1, 0, s2), idx(0, 0, s2)))
    return mesh_object(name, verts, faces)


def cylinder_z(name, cx, cy, z0, z1, radius, segments):
    return revolve_solid(name, [(0.0, z0), (radius, z0), (radius, z1),
                                (0.0, z1)], segments)


def cylinder_x(name, x0, x1, radius, cy, cz, segments):
    verts = []
    for x in (x0, x1):
        for s in range(segments):
            a = TAU * s / segments
            verts.append((x, cy + radius * math.cos(a),
                          cz + radius * math.sin(a)))
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        faces.append((s, s2, segments + s2, segments + s))
    faces.append(tuple(range(segments - 1, -1, -1)))
    faces.append(tuple(range(segments, 2 * segments)))
    return mesh_object(name, verts, faces)


def prism_xy(name, points, z0, z1):
    n = len(points)
    verts = [(x, y, z0) for x, y in points] + [(x, y, z1) for x, y in points]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def u_outline(back_x, half_width, front_x, arc_steps=40):
    """U-shaped slot opening toward +X with its rounded end toward -X."""
    points = [(front_x, -half_width), (back_x, -half_width)]
    for k in range(1, arc_steps):
        a = -math.pi / 2 - math.pi * k / arc_steps
        points.append((back_x + half_width * math.cos(a),
                       half_width * math.sin(a)))
    points.append((back_x, half_width))
    points.append((front_x, half_width))
    return points


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


def bvh_from(obj):
    verts = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    polys = [tuple(poly.vertices) for poly in obj.data.polygons]
    return BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=1e-8)


def ray_cast(tree, origin, direction):
    location, normal, index, distance = tree.ray_cast(
        Vector(origin), Vector(direction).normalized(), 10.0)
    if location is None:
        return None
    return {'location': tuple(float(v) for v in location),
            'distance': float(distance)}


def opening_checks_world(shift):
    cap = bpy.data.objects.get('Can cap')
    actuator = bpy.data.objects.get('Can actuator')
    insert = bpy.data.objects.get('Can nozzle insert')
    if cap is None or actuator is None or insert is None:
        return {'passed': False, 'error': 'cap/actuator/insert parts missing'}
    cap_tree = bvh_from(cap)
    act_tree = bvh_from(actuator)
    insert_tree = bvh_from(insert)
    # Sample the rim just behind the well, where the cap top is the local lip;
    # a far-side sample would sit low on the dome and understate the depth.
    top_side = ray_cast(cap_tree, (-0.019, 0.0, 0.40), (0, 0, -1))
    floor = ray_cast(cap_tree, (0.010, 0.010, 0.40), (0, 0, -1))
    cavity = ray_cast(cap_tree, (0.0, 0.0, -0.05), (0, 0, 1))
    annulus = ray_cast(cap_tree, (0.0325, 0.0, -0.05), (0, 0, 1))
    bore_z = can_cfg['cap']['actuator']['bore']['z'] + shift
    bore_a = ray_cast(act_tree, (0.03, 0.0, bore_z), (-1, 0, 0))
    bore_b = ray_cast(insert_tree, (0.03, 0.0, bore_z), (-1, 0, 0))
    candidates = [hit for hit in (bore_a, bore_b) if hit is not None]
    bore_hit = min(candidates, key=lambda hit: hit['distance']) if candidates else None
    wall_hit = ray_cast(act_tree, (0.03, 0.0032, bore_z), (-1, 0, 0))
    front_x = can_cfg['cap']['actuator']['center'][0] + \
        can_cfg['cap']['actuator']['half'][0]
    well_depth = None if top_side is None or floor is None else \
        float(top_side['location'][2] - floor['location'][2])
    hollow = None if cavity is None or annulus is None else \
        float(cavity['location'][2] - annulus['location'][2])
    bore_x = None if bore_hit is None else float(bore_hit['location'][0])
    wall_x = None if wall_hit is None else float(wall_hit['location'][0])
    result = {
        'well_floor_z': None if floor is None else round(float(floor['location'][2]), 6),
        'cap_rim_sample_z': None if top_side is None else round(float(top_side['location'][2]), 6),
        'well_depth_m': None if well_depth is None else round(well_depth, 6),
        'well_recess_passed': bool(well_depth is not None and well_depth >= .006),
        'cavity_ceiling_z': None if cavity is None else round(float(cavity['location'][2]), 6),
        'bottom_annulus_z': None if annulus is None else round(float(annulus['location'][2]), 6),
        'cap_hollow_difference_m': None if hollow is None else round(hollow, 6),
        'cap_bottom_opening_passed': bool(hollow is not None and hollow >= .02),
        'actuator_front_x': round(float(front_x), 6),
        'bore_hit_x': None if bore_x is None else round(bore_x, 6),
        'bore_wall_hit_x': None if wall_x is None else round(wall_x, 6),
        'nozzle_bore_passed': bool(bore_x is not None and wall_x is not None and
                                   bore_x < front_x - .0005 and
                                   abs(wall_x - front_x) < .002),
        'shift_z': round(float(shift), 6),
        'method': ('Blender BVH ray casts on the assembled world-space meshes: the '
                   'well ray enters the recess and hits a floor below the cap rim; '
                   'the centre bottom ray reaches the cavity ceiling while an outer '
                   'ray hits the bottom annulus; the bore ray passes through the '
                   'actuator front wall and stops on the dark insert, while a parallel '
                   'off-axis ray hits the front wall.'),
    }
    result['passed'] = bool(result['well_recess_passed'] and
                            result['cap_bottom_opening_passed'] and
                            result['nozzle_bore_passed'])
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build():
    seg = config['radial_segments']
    zlab0 = can_cfg['label']['z0']
    zlab1 = can_cfg['label']['z1']
    zcap0 = can_cfg['cap']['z0']
    zcap1 = can_cfg['cap']['z1']

    body = revolve_solid('Can body', can_cfg['body']['profile'], seg)
    assign(body, 'metal', uv='cyl', z0=can_cfg['body']['z0'],
           z1=can_cfg['body']['z1'], smooth=True, axis_flat=(0, 0, 1))

    label = tube_shell('Can label', can_cfg['label']['outer_radius'],
                       can_cfg['label']['inner_radius'], zlab0, zlab1, seg)
    assign(label, 'label', uv='label', z0=zlab0, z1=zlab1, smooth=True,
           axis_flat=(0, 0, 1))

    bottom = revolve_solid('Can bottom cup', can_cfg['bottom']['profile'], seg)
    assign(bottom, 'bottom', uv='planar', smooth=True, axis_flat=(0, 0, 1))

    rim_cfg = can_cfg['bottom']['rim_torus']
    rim = torus_mesh('Can bottom rim', (0, 0, rim_cfg['z']), (0, 0, 1),
                     rim_cfg['major'], rim_cfg['minor'], seg, 12)
    assign(rim, 'metal', uv='box', smooth=True, scale=.25)
    step_cfg = can_cfg['bottom']['step_torus']
    step = torus_mesh('Can bottom step', (0, 0, step_cfg['z']), (0, 0, 1),
                      step_cfg['major'], step_cfg['minor'], seg, 10)
    assign(step, 'metal', uv='box', smooth=True, scale=.20)
    ridge_cfg = can_cfg['bottom']['ridge_torus']
    ridge = torus_mesh('Can bottom ridge', (0, 0, ridge_cfg['z']), (0, 0, 1),
                       ridge_cfg['major'], ridge_cfg['minor'], seg, 10)
    assign(ridge, 'metal', uv='box', smooth=True, scale=.15)

    cap = revolve_solid('Can cap', can_cfg['cap']['profile'], seg)
    cavity = can_cfg['cap']['cavity']
    cavity_cutter = revolve_solid('Cap cavity cutter', [
        (0.0, cavity['z0']), (cavity['radius'], cavity['z0']),
        (cavity['radius'], cavity['z1']), (0.0, cavity['z1'])], seg)
    boolean_difference(cap, cavity_cutter)
    well = can_cfg['cap']['well']
    well_cutter = prism_xy('Well cutter', u_outline(well['back_center_x'],
                                                    well['half_width'],
                                                    well['front_x']),
                            well['floor_z'], well['top_z'])
    apply_bevel(well_cutter, well['bevel'], 3, 40.0)
    boolean_difference(cap, well_cutter)
    lip = well.get('front_lip')
    if lip:
        # The shallow front lip keeps a raised rim around the spray well while
        # still making the well visible from the front, as in the reference.
        lip_cutter = box_at(
            'Front lip cutter',
            ((lip['x0'] + lip['x1']) / 2, 0,
             (lip['floor_z'] + lip['top_z']) / 2),
            ((lip['x1'] - lip['x0']) / 2, lip['half_width'],
             (lip['top_z'] - lip['floor_z']) / 2))
        apply_bevel(lip_cutter, lip['bevel'], 2, 40.0)
        boolean_difference(cap, lip_cutter)
    apply_bevel(cap, .0012, 3, 34.0)
    assign(cap, 'cap', uv='cyl', z0=zcap0, z1=zcap1, smooth=True)

    seam_cfg = can_cfg['cap']['seam_ring']
    seam = torus_mesh('Can cap seam', (0, 0, seam_cfg['z']), (0, 0, 1),
                      seam_cfg['major'], seam_cfg['minor'], seg, 10)
    assign(seam, 'dark', uv='box', smooth=True, scale=.06)

    boss_cfg = can_cfg['cap']['boss']
    boss = cylinder_z('Can nozzle boss', boss_cfg['center'][0],
                      boss_cfg['center'][1], boss_cfg['z0'], boss_cfg['z1'],
                      boss_cfg['radius'], 64)
    assign(boss, 'cap', uv='cyl', z0=boss_cfg['z0'], z1=boss_cfg['z1'],
           smooth=True, axis_flat=(0, 0, 1))
    collar_cfg = can_cfg['cap']['boss_collar']
    collar = torus_mesh('Can nozzle collar',
                        (boss_cfg['center'][0], boss_cfg['center'][1],
                         collar_cfg['z']), (0, 0, 1),
                        collar_cfg['major'], collar_cfg['minor'], 96, 10)
    assign(collar, 'cap', uv='box', smooth=True, scale=.08)

    act = can_cfg['cap']['actuator']
    button = box_at('Can actuator', (act['center'][0], act['center'][1],
                                     (act['z0'] + act['z1']) / 2), act['half'])
    apply_bevel(button, .0014, 3, 35.0)
    bore = cylinder_x('Actuator bore cutter', act['center'][0] - .03,
                      act['center'][0] + .03, act['bore']['radius'],
                      act['center'][1], act['bore']['z'], 48)
    boolean_difference(button, bore)
    assign(button, 'dark', uv='box', smooth=True, scale=.035)

    ribs = act['ribs']
    for i, dx in enumerate(ribs['x_offsets'], start=1):
        rib = box_at(f'Can actuator rib {i}',
                     (act['center'][0] + dx, act['center'][1], ribs['z']),
                     ribs['half'])
        apply_bevel(rib, .00025, 2, 35.0)
        assign(rib, 'dark', uv='box', smooth=False, scale=.02)

    insert_cfg = can_cfg['cap']['nozzle_insert']
    insert = cylinder_x('Can nozzle insert', insert_cfg['x0'],
                        insert_cfg['x1'], insert_cfg['radius'],
                        act['center'][1], act['bore']['z'], 48)
    assign(insert, 'dark', uv='box', smooth=True, scale=.02)


# ---------------------------------------------------------------------------
# Render setup
# ---------------------------------------------------------------------------
def camera(scene, direction, center, span, scale=1.16, roll_top=False):
    direction = Vector(direction).normalized()
    cam = legacy.make_camera(scene, Vector(center) + direction * span * 5, center,
                             ortho_scale=span * scale)
    if roll_top and abs(direction.z) > .999:
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.rotation_quaternion @ Quaternion((0, 0, 1), -math.pi / 2)
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
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .55
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

make_materials()
build()
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
Z_SHIFT = -low.z
for obj in objects:
    obj.location.z += Z_SHIFT
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
opening = opening_checks_world(Z_SHIFT)
bpy.context.view_layer.update()

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
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 44
front_span = config['render']['front_span']
top_span = config['render']['top_span']
new_render_bounds = {'low': list(low), 'high': list(high), 'center': list(center)}

for name, direction in directions.items():
    use_span = top_span if name in ('top', 'bottom') else front_span
    # Roll the top camera so +X (spray/front) is at the bottom of the frame.
    # The bottom reference shows the stamped code horizontally, so the bottom
    # camera stays unrolled.
    cam = camera(scene, direction, center, use_span, roll_top=name == 'top')
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution,
         samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (2.2, -2.0, 1.0), Vector((0, 0, center.z)), .30)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

# Clay material comparison (front + cap close).
clay = bpy.data.materials.new('14 clay')
clay.use_nodes = True
clay_bsdf = clay.node_tree.nodes.get('Principled BSDF')
clay_bsdf.inputs['Base Color'].default_value = (.52, .52, .52, 1)
clay_bsdf.inputs['Roughness'].default_value = .55
clay_bsdf.inputs['Metallic'].default_value = 0.0
saved_materials = {}
for obj in objects:
    saved_materials[obj.name] = [m for m in obj.data.materials]
    obj.data.materials.clear()
    obj.data.materials.append(clay)
cam = camera(scene, directions['front'], center, front_span)
draw(scene, render_dir / 'new-clay-front.png', resolution=resolution,
     samples=samples)
bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (2.2, -2.0, 1.0), Vector((0, 0, center.z)), .30)
draw(scene, render_dir / 'new-clay-close.png',
     resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
for obj in objects:
    obj.data.materials.clear()
    for material in saved_materials[obj.name]:
        obj.data.materials.append(material)

if request['draft']:
    well_cfg = can_cfg['cap']['well']
    act_cfg = can_cfg['cap']['actuator']
    draft_details = (
        ('cap-front', (0, 0, 0.232 + Z_SHIFT), (1, 0, .04), .105),
        ('cap-top', (0, 0, 0.243 + Z_SHIFT), (.18, -.25, 1), .082),
        ('cap-well', (well_cfg['back_center_x'] + .014, 0, .245 + Z_SHIFT),
         (.55, -.55, .82), .062),
        ('nozzle', (act_cfg['center'][0] + .006, 0, .247 + Z_SHIFT),
         (1, -.06, .10), .034),
        ('bottom', (0, 0, 0.006 + Z_SHIFT), (0, 0, -1), .085),
        ('bottom-code', (0, 0, 0.006 + Z_SHIFT), (0, 0, -1), .044),
    )
    for name, target, direction, scale in draft_details:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=640,
             samples=16)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = not (obj.name in ('Can cap', 'Can cap seam',
                                            'Can nozzle boss', 'Can nozzle collar',
                                            'Can actuator', 'Can actuator rib 1',
                                            'Can actuator rib 2', 'Can actuator rib 3',
                                            'Can actuator rib 4', 'Can nozzle insert'))
    cam = camera(scene, (0, 0, -1), Vector((0, 0, .215 + Z_SHIFT)), .082)
    draw(scene, render_dir / 'cap-bottom-isolated.png', resolution=640,
         samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

if not request['draft']:
    for i in range(24):
        angle = TAU * i / 24
        elev = config['render']['turntable_elevation']
        cam = camera(scene, (math.cos(angle), math.sin(angle), elev),
                     center, front_span)
        draw(scene, render_dir / f'turn-{i:02d}.png', resolution=512, samples=12)
        bpy.data.objects.remove(cam, do_unlink=True)

    cap_cfg = can_cfg['cap']
    well_cfg = cap_cfg['well']
    act_cfg = cap_cfg['actuator']
    detail_cameras = (
        ('front', (0, 0, .0), (1, 0, 0), front_span),
        ('cap-front', (0, 0, 0.232 + Z_SHIFT), (1, 0, .04), .105),
        ('cap-top', (0, 0, 0.243 + Z_SHIFT), (.18, -.25, 1), .082),
        ('cap-well', (well_cfg['back_center_x'] + .014, 0, .245 + Z_SHIFT),
         (.55, -.55, .82), .062),
        ('nozzle', (act_cfg['center'][0] + .006, 0, .247 + Z_SHIFT),
         (1, -.06, .10), .034),
        ('label-logo', (0, 0, .158 + Z_SHIFT), (1, 0, 0), .115),
        ('label-lower', (0, 0, .055 + Z_SHIFT), (1, 0, 0), .105),
        ('label-side', (0, 0, .105 + Z_SHIFT), (.12, -1, .03), .245),
        ('cap-seam', (0, 0, .200 + Z_SHIFT), (.55, -.75, .28), .062),
        ('bottom', (0, 0, 0.006 + Z_SHIFT), (0, 0, -1), .085),
        # detail-bottom and new-bottom-alternate must share the same camera roll
        # so the metal reflection comparison is pixel-aligned.
        ('bottom-code', (0, 0, 0.006 + Z_SHIFT), (0, 0, -1), .044),
        ('bottom-rim', (0.025, 0, 0.004 + Z_SHIFT), (.2, -.3, -1), .050),
        ('body-side', (0, 0, .105 + Z_SHIFT), (.32, -1, .10), .215),
        ('surface', (0.0, -0.028, .130 + Z_SHIFT), (.55, -1, .24), .070),
        ('cap-scratch', (0.0, -0.030, .218 + Z_SHIFT), (.45, -1, .30), .050),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768,
             samples=28)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('cap-only', lambda n: n in ('Can cap', 'Can cap seam', 'Can nozzle boss',
                                     'Can nozzle collar', 'Can actuator',
                                     'Can actuator rib 1', 'Can actuator rib 2',
                                     'Can actuator rib 3', 'Can actuator rib 4',
                                     'Can nozzle insert'),
         (0.4, -1, .35), (0, 0, .225 + Z_SHIFT), .095),
        ('label-only', lambda n: n == 'Can label', (1, 0, 0),
         (0, 0, .10 + Z_SHIFT), .235),
        ('metal-only', lambda n: n in ('Can body', 'Can bottom cup', 'Can bottom rim',
                                       'Can bottom step', 'Can bottom ridge'),
         (0.3, -1, .18), (0, 0, .105 + Z_SHIFT), .235),
        ('actuator-only', lambda n: n in ('Can nozzle boss', 'Can nozzle collar',
                                          'Can actuator', 'Can actuator rib 1',
                                          'Can actuator rib 2', 'Can actuator rib 3',
                                          'Can actuator rib 4', 'Can nozzle insert'),
         (1, -.2, .35), (act_cfg['center'][0], 0, .246 + Z_SHIFT), .050),
        ('cap-top-isolated', lambda n: n in ('Can cap', 'Can cap seam',
                                              'Can nozzle boss', 'Can nozzle collar',
                                              'Can actuator', 'Can actuator rib 1',
                                              'Can actuator rib 2', 'Can actuator rib 3',
                                              'Can actuator rib 4', 'Can nozzle insert'),
         (0.12, -0.2, 1), (0, 0, .240 + Z_SHIFT), .082),
        ('cap-bottom-isolated', lambda n: n in ('Can cap', 'Can cap seam'),
         (0.35, -0.45, -1), (0, 0, .215 + Z_SHIFT), .090),
    )
    for label, predicate, direction, target, scale in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        cam = camera(scene, direction, Vector(target), scale,
                     roll_top=label == 'cap-bottom-isolated')
        draw(scene, render_dir / f'{label}.png', resolution=768, samples=22)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    key = bpy.data.objects['large softbox']
    key.location = (-3.4, 2.8, 4.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, directions['front'], center, front_span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768,
         samples=28)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (0, 0, -1), Vector((0, 0, 0.006 + Z_SHIFT)), .085)
    draw(scene, render_dir / 'new-bottom-alternate.png', resolution=768,
         samples=28)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Source-photo framing diagnostic (not an acceptance gate).
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old_height = max(1e-6, oldhigh.z - oldlow.z)
new_height = max(1e-6, high.z - low.z)
# The library mesh uses an arbitrary reconstruction scale.  The source camera
# was calibrated for that original scale, so the perspective diagnostic must
# move the camera by the same uniform scale factor.
old_scale = new_height / old_height
old.data.transform(Matrix.Scale(old_scale, 4))
old.data.update()
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old.data.transform(Matrix.Translation((center.x - oldcenter.x,
                                       center.y - oldcenter.y,
                                       low.z - oldlow.z)))
old.data.update()
bpy.context.view_layer.update()
oldlow, oldhigh, oldcenter, oldspan = bounds([old])
old.location = (0, 0, 0)
old.hide_render = True
for obj in objects:
    obj.hide_render = False
source = request['source_camera']
factor = old_scale
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
    use_span = top_span if name in ('top', 'bottom') else front_span
    cam = camera(scene, direction, center, use_span, roll_top=name == 'top')
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution,
         samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (2.2, -2.0, 1.0), Vector((0, 0, center.z)), .30)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {
    'parts': parts,
    'bounds': [list(low), list(high)],
    'span': span,
    'parts_count': len(parts),
    'polygons': sum(len(o.data.polygons) for o in objects),
    'triangles': sum(max(0, len(p.vertices) - 2)
                     for o in objects for p in o.data.polygons),
    'texture_size': 2048,
    'detail_texture_size': 1024,
    'materials': sorted(materials),
    'render_device': render_device,
    'self_intersecting_components': {k: v for k, v in intersections.items() if v},
    'intersection_check': ('BVH same-part face overlap; intentional assembly overlap '
                           'between label/body and cap/body is not judged'),
    'geometry_basis': ('original used-can photographs plus the uncalibrated six-view '
                       'visual-match sheet'),
    'render_samples': samples,
    'view_directions': {k: list(v) for k, v in directions.items()},
    'view_note': ('Front camera is +X, the source-photo view. The six-view sheet is '
                  'not a calibrated turntable. Top/bottom cameras are rolled so +X '
                  '(the spray/front direction) is at the bottom of the frame.'),
    'old_mesh_comparison_scale': {
        'old_height_m': round(float(old_height), 6),
        'new_height_m': round(float(new_height), 6),
        'uniform_scale_applied': round(float(old_scale), 6),
        'note': ('The library mesh has an arbitrary reconstruction scale and was '
                 'uniformly scaled for visual comparison only; the library file was '
                 'not modified.')},
    'source_camera_note': ('source-photo framing distance scaled by old/new span, '
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
    'z_shift': Z_SHIFT,
    'new_bounds_before_export': new_render_bounds,
    'opening_checks': opening,
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
