"""Blender worker for the review-only sealant-can candidate (item 13).

Shape basis: the original clean/rust photograph shows two upright used white
plastic cans.  The user six-view sheet is an uncalibrated category reference of
one clean can; it is used for lid/body/bottom topology only and is never treated
as a measured or identical-unit reference.  The candidate is a part-based
rebuild of two separate cans: tapered body, overhanging lid with raised rim and
shallow recess, two seam rings, a two-block lift latch, a side clip, two base
tabs, a centre gate and a small moulded bottom mark.  It never replaces the
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
    names = {0: '13 clean white plastic', 1: '13 weathered white plastic',
             2: '13 stained white plastic', 3: '13 worn white plastic'}
    for tile, name in names.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')
        color = image_node(nodes, 'white_base')
        metal = image_node(nodes, 'white_mr', True)
        normal = image_node(nodes, 'white_normal', True)
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(color.outputs['Color'], principled.inputs['Base Color'])
        links.new(metal.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = {0: .08, 1: .14, 2: .17, 3: .10}[tile]
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

    set_material('mark', 'mark', .08)


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
def build_can(prefix, center, rotation_deg):
    """Build one can at the origin, then bake rotation + translation."""
    seg = config['radial_segments']
    can_cfg = config['can']
    start = len(objects)

    body = revolve_solid(f'{prefix} body', can_cfg['body']['profile'], seg)
    assign_tile(body, 1, smooth='side', axis=(0, 0, 1), scale=0.42, uv='z')

    lid = revolve_solid(f'{prefix} lid', can_cfg['lid']['profile'], seg)
    assign_tile(lid, 0, smooth='side', axis=(0, 0, 1), scale=0.42, uv='z')

    for seam in can_cfg['seam_rings']:
        ring = torus(f"{prefix} {seam['name']}", (0, 0, seam['z']),
                     (0, 0, 1), seam['major'], seam['minor'], 96, 10)
        assign_tile(ring, 2, smooth='all', scale=0.20)
    for rim in can_cfg.get('recess_rings', []):
        ring = torus(f"{prefix} {rim['name']}", (0, 0, rim['z']),
                     (0, 0, 1), rim['major'], rim['minor'], 96, 8)
        assign_tile(ring, 2, smooth='all', scale=0.12)

    latch = can_cfg['latch']
    upper = box_at(f'{prefix} latch upper', latch['upper_center'], latch['upper_half'])
    apply_bevel(upper, .0022, 3)
    assign_tile(upper, 0, smooth='all', scale=0.10)
    lower = box_at(f'{prefix} latch lower', latch['lower_center'], latch['lower_half'])
    apply_bevel(lower, .0018, 3)
    assign_tile(lower, 1, smooth='all', scale=0.08)

    bottom = can_cfg['bottom_tabs']
    for i, angle in enumerate(bottom['angles_deg'], start=1):
        rad = math.radians(angle)
        tab = box_at(f'{prefix} bottom tab {i}',
                     (bottom['radius'] * math.cos(rad),
                      bottom['radius'] * math.sin(rad), bottom['z']),
                     bottom['half'])
        apply_bevel(tab, .0012, 2)
        assign_tile(tab, 2, smooth='all', scale=0.06)

    gate = revolve_solid(f'{prefix} bottom gate', [
        (0.0, can_cfg['bottom_gate']['center'][2] - can_cfg['bottom_gate']['depth'] / 2),
        (can_cfg['bottom_gate']['radius'],
         can_cfg['bottom_gate']['center'][2] - can_cfg['bottom_gate']['depth'] / 2),
        (can_cfg['bottom_gate']['radius'],
         can_cfg['bottom_gate']['center'][2] + can_cfg['bottom_gate']['depth'] / 2),
        (0.0, can_cfg['bottom_gate']['center'][2] + can_cfg['bottom_gate']['depth'] / 2),
    ], 48)
    assign_tile(gate, 2, smooth='side', axis=(0, 0, 1), scale=0.04)

    mark = can_cfg['bottom_mark']
    cx, cy = mark['center']
    z = mark['z']
    tri = []
    for k in range(3):
        a = math.radians(90 + 120 * k)
        tri.append(Vector((cx + mark['circumradius'] * math.cos(a),
                           cy + mark['circumradius'] * math.sin(a), z)))
    for k in range(3):
        p0, p1 = tri[k], tri[(k + 1) % 3]
        u = (p1 - p0).normalized()
        v = Vector((-u.y, u.x, 0))
        bar = oriented_plate(f'{prefix} bottom mark bar {k + 1}',
                             (p0 + p1) / 2, u, v,
                             (p1 - p0).length / 2, mark['bar_half_width'],
                             mark['thickness'])
        assign_special(bar, 'mark', smooth='flat', scale=0.05)
    for k, dx in enumerate((-0.0040, 0.0040), start=1):
        bar = oriented_plate(f'{prefix} bottom mark letter {k}',
                             (cx + dx, cy - 0.0130, z), (1, 0, 0), (0, 1, 0),
                             0.0011, 0.0026, mark['thickness'])
        assign_special(bar, 'mark', smooth='flat', scale=0.05)

    rotation = Matrix.Rotation(math.radians(rotation_deg), 4, 'Z')
    translation = Matrix.Translation((center[0], center[1], 0.0))
    transform = translation @ rotation
    made = objects[start:]
    for obj in made:
        # Some primitives (the seam torus) keep a non-identity object rotation
        # and location.  Bake that into the mesh data before applying the can
        # transform; otherwise the object rotation is applied a second time on
        # export and the ring lands on the other can.
        obj.data.transform(obj.matrix_basis)
        obj.matrix_basis = Matrix.Identity(4)
        apply_transform(obj, transform)
    return made


def build():
    layout = config['layout']
    build_can('Can front', layout['front_center'],
              layout['front_rotation_deg'])
    build_can('Can rear', layout['rear_center'],
              layout['rear_rotation_deg'])


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
samples = 8 if request['draft'] else 48
span_len = span
span_front = max(high.y - low.y, high.z - low.z)
for name, direction in directions.items():
    use_span = span_front if name in ('front', 'back') else span_len
    cam = camera(scene, direction, center, use_span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .78)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

# Single-can diagnostic using the rear can, whose latch is closest to +X.
front_prefix, rear_prefix = 'Can front ', 'Can rear '
single_center = None
for obj in objects:
    if obj.name.startswith(rear_prefix):
        lo, hi, c, s = bounds([obj])
for obj in objects:
    if obj.name == rear_prefix + 'body':
        lo, hi, c, s = bounds([obj])
        single_center = c
if single_center is None:
    single_center = center
for i, name in enumerate(('front', 'back', 'left', 'right', 'top', 'bottom')):
    for obj in objects:
        obj.hide_render = not obj.name.startswith(rear_prefix)
    direction = directions[name]
    cam = camera(scene, direction, single_center, .36)
    draw(scene, render_dir / f'single-{name}.png',
         resolution=512 if request['draft'] else 640,
         samples=8 if request['draft'] else 32)
    bpy.data.objects.remove(cam, do_unlink=True)
for obj in objects:
    obj.hide_render = False

if not request['draft']:
    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .14), center, span)
        draw(scene, render_dir / f'turn-{i:02d}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    can_cfg = config['can']
    layout = config['layout']
    rear = layout['rear_center']
    front = layout['front_center']
    detail_cameras = (
        ('front', (0, 0, 0.16 + Z_SHIFT), (1, 0, .10), .52),
        ('top', (0, 0, 0.28 + Z_SHIFT), (0, 0, 1), .52),
        ('lid-top', (rear[0], rear[1], 0.28 + Z_SHIFT), (0.2, -0.45, 1), .19),
        ('lid-rim', (rear[0], rear[1], 0.278 + Z_SHIFT), (0.4, -1, .55), .16),
        ('seam-latch', (rear[0] + 0.105, rear[1] - 0.015, 0.250 + Z_SHIFT),
         (0.65, -1, .22), .13),
        ('front-latch', (front[0], front[1] - 0.105, 0.250 + Z_SHIFT),
         (0.20, -1, .12), .11),
        ('bottom', (rear[0], rear[1], 0.02 + Z_SHIFT), (0, 0, -1), .28),
        ('bottom-mark', (rear[0] - 0.030, rear[1] + 0.020, 0.002 + Z_SHIFT),
         (0.15, -0.25, -1), .075),
        ('can-gap', (0, 0, 0.26 + Z_SHIFT), (0, 0.2, 1), .38),
        ('body', (front[0], front[1], 0.12 + Z_SHIFT), (0.4, -1, .18), .30),
        ('surface', (front[0], front[1] - 0.098, 0.10 + Z_SHIFT),
         (0.9, -1, .30), .22),
        ('lid-recess', (rear[0], rear[1], 0.274 + Z_SHIFT), (0.15, -0.2, 1), .11),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('body-only', lambda n: n.endswith(' body'), (0.5, -1, .18), center, .50),
        ('lid-only', lambda n: (n.endswith(' lid') or 'seam' in n or 'latch' in n),
         (0.4, -1, .35), (0, 0, 0.25 + Z_SHIFT), .48),
        ('latch-only', lambda n: 'latch' in n, (0.6, -1, .18),
         (rear[0] + 0.10, rear[1], 0.25 + Z_SHIFT), .24),
        ('bottom-parts-only', lambda n: ('bottom' in n), (0, 0, -1),
         (rear[0], rear[1], 0.01 + Z_SHIFT), .25),
        ('two-cans-top', lambda n: True, (0, 0, 1),
         (0, 0, 0.28 + Z_SHIFT), .52),
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
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .78)
    draw(scene, render_dir / 'new-surface-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

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
cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .78)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                         for p in o.data.polygons),
        'texture_size': 2048, 'detail_texture_size': 512,
        'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'render_device': render_device,
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': 'original two-can photo proportions; six-view sheet is an uncalibrated single-can category reference',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': ('Camera at +X is the source-photo view; back at -X. The front can '
                      'is nearer +X and offset to -Y with its latch facing -Y (side profile); '
                      'the rear can is offset to +Y with its latch near +X. The source photo and '
                      'the category reference are not a calibrated turntable.'),
        'single_can_source': 'Can rear (rotation -8 deg, latch near +X)',
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
        'isolated_checks': {'single_can_enclosed_holes_expected': 0,
                            'two_can_top_gap_expected': True}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
