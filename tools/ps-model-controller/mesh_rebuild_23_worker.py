"""Blender worker: part-based rebuild of item 23, an ARK KC S1BN125-145 cylinder.

The source photograph is a three-quarter view of a black-capped, light-barrelled
tie-rod pneumatic cylinder with four exposed tie rods and hex nuts, a black
front gland, a chrome piston rod with one large hex nut, brass hex-socket plugs
and threaded side ports, and a printed ARK label.  The six-view sheet is an
uncalibrated visual match sheet; it duplicates the label on the back panel and
shows side ports on both sides, so those choices are marked as inferred.

Coordinates: Z up, cylinder axis along Y, rod end at -Y (front), rear at +Y,
label on the observed +X side.  All visible parts are separate watertight
meshes.  Metals keep a metallic PBR response; the label and rubber wiper are
dielectric.  No lighting or shadow is baked into the atlases.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
cfg = request['config']
D = cfg['dimensions']
TEX = request['textures']
TILE = cfg['texture']['tiles']

objects = []
tiles = {}
metal_objects = []
nonmetal_objects = []
MATERIAL = None
CAM = None
DYNAMIC_TILES = {'aluminum', 'cap', 'chrome', 'steel', 'brass', 'thread',
                 'dark_steel', 'port_dark'}
DIELECTRIC_TILES = {'label', 'rubber'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed black cap, aluminium, steel, brass and label PBR atlas')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    principled.inputs['IOR'].default_value = 1.45

    def texture(name, noncolor=False):
        node = nodes.new('ShaderNodeTexImage')
        node.image = bpy.data.images.load(TEX[name], check_existing=True)
        node.image.pack()
        if noncolor:
            node.image.colorspace_settings.name = 'Non-Color'
        node.interpolation = 'Linear'
        return node

    color = texture('base_color')
    metal = texture('metallic_roughness', True)
    normal = texture('normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(metal.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = .30
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    return material


def tile_bounds(idx):
    col, row = idx % 4, idx // 4
    return (col * .25, row * .25, (col + 1) * .25, (row + 1) * .25)


def remap_uv(obj, tile, *, v_lo=.01, v_hi=.99, u_lo=.01, u_hi=.99, margin=.012):
    x0, y0, x1, y1 = tile_bounds(tile)
    w, h = x1 - x0, y1 - y0
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    for loop in obj.data.uv_layers.active.data:
        u, v = loop.uv
        u = u_lo + (u_hi - u_lo) * (u - math.floor(u))
        v = v_lo + (v_hi - v_lo) * (v - math.floor(v))
        loop.uv = (x0 + margin * w + u * (w - 2 * margin * w),
                   y0 + margin * h + v * (h - 2 * margin * h))


def assign(obj, tile, *, mode='default', v_lo=.01, v_hi=.99, u_lo=.01, u_hi=.99,
           margin=.012, register=True, cutter=False):
    if cutter:
        obj.data.materials.clear()
        obj.hide_render = True
        obj.hide_viewport = True
        return obj
    obj.data.materials.clear()
    obj.data.materials.append(MATERIAL)
    tiles[obj.name] = tile
    name = next((key for key, value in TILE.items() if value == tile), f'tile{tile}')
    if mode == 'constant':
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name='Atlas')
        x0, y0, x1, y1 = tile_bounds(tile)
        center = (x0 + .5 * (x1 - x0), y0 + .5 * (y1 - y0))
        for loop in obj.data.uv_layers.active.data:
            loop.uv = center
    else:
        remap_uv(obj, tile, v_lo=v_lo, v_hi=v_hi, u_lo=u_lo, u_hi=u_hi, margin=margin)
    if register:
        objects.append(obj)
        if name in DYNAMIC_TILES:
            metal_objects.append(obj.name)
        elif name in DIELECTRIC_TILES:
            nonmetal_objects.append(obj.name)
    return obj


def apply_bevel(obj, width, segments=2, angle=40):
    if width <= 0:
        return
    mod = obj.modifiers.new('Small manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def finish(obj, name, tile, *, bevel=.0012, smooth=True, constant=False,
           v_lo=.01, v_hi=.99, register=True, cutter=False):
    obj.name = name
    obj.data.name = name
    if obj.type != 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH')
        obj = bpy.context.object
    apply_bevel(obj, bevel)
    for poly in obj.data.polygons:
        poly.use_smooth = smooth
    if smooth:
        for poly in obj.data.polygons:
            if len(poly.vertices) == 3:
                poly.use_smooth = False
    return assign(obj, tile, mode='constant' if constant else 'default',
                  v_lo=v_lo, v_hi=v_hi, register=register, cutter=cutter)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
def mesh_object(name, verts, faces, loop_uvs=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if loop_uvs is not None:
        layer = mesh.uv_layers.new(name='Atlas')
        if len(loop_uvs) != len(mesh.polygons):
            raise ValueError('UV face count differs from mesh face count')
        for poly, face_uvs in zip(mesh.polygons, loop_uvs):
            for li, uv in zip(poly.loop_indices, face_uvs):
                layer.data[li].uv = uv
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def along(name, start, end, radius, tile, *, segs=32, bevel=.001, constant=False,
          cutter=False):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, constant=constant, cutter=cutter)


def box(name, center, size, tile, *, bevel=.001, constant=False, cutter=False):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, tile, bevel=bevel, constant=constant, cutter=cutter)


def rotated_box(name, center, size, angle_y, tile, *, bevel=.0008, constant=False):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    obj.rotation_euler = (0, angle_y, 0)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, constant=constant)


def torus(name, major, minor, location, axis, tile, *, major_segments=64,
          minor_segments=12, cutter=False):
    rotations = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0)}
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=location, rotation=rotations[axis],
                                     major_radius=major, minor_radius=minor)
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=0, constant=True, cutter=cutter)


def hex_prism_axis(name, radius, thickness, center, axis, tile, *, bevel=.0008,
                   constant=False, cutter=False):
    rotations = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0),
                 'Y': (math.pi / 2, 0, 0)}
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius, depth=thickness,
                                        location=center, rotation=rotations[axis])
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, smooth=False, constant=constant,
                  cutter=cutter)


def helix(name, axis, center, radius, tube_radius, pitch, h0, h1, tile, *,
          seg_per_turn=14, tube_segments=8, bevel=0.0):
    """Closed helical tube used as a visual thread ridge.

    center is (y, z) for the X axis, (x, z) for the Y axis and (x, y) for the
    Z axis.  h0/h1 are coordinates along the thread axis.
    """
    length = abs(h1 - h0)
    turns = length / pitch
    rings = max(4, int(math.ceil(turns * seg_per_turn)))
    verts, faces = [], []
    if axis == 'Y':
        rad = lambda a: Vector((math.cos(a), 0.0, math.sin(a)))
        axis_dir = Vector((0.0, 1.0, 0.0))
        point = lambda a, h: Vector((center[0] + radius * math.cos(a), h,
                                     center[1] + radius * math.sin(a)))
        end = lambda a, h: Vector((center[0] + radius * math.cos(a), h,
                                   center[1] + radius * math.sin(a)))
    elif axis == 'Z':
        rad = lambda a: Vector((math.cos(a), math.sin(a), 0.0))
        axis_dir = Vector((0.0, 0.0, 1.0))
        point = lambda a, h: Vector((center[0] + radius * math.cos(a),
                                     center[1] + radius * math.sin(a), h))
        end = point
    elif axis == 'X':
        rad = lambda a: Vector((0.0, math.cos(a), math.sin(a)))
        axis_dir = Vector((1.0, 0.0, 0.0))
        point = lambda a, h: Vector((h, center[0] + radius * math.cos(a),
                                     center[1] + radius * math.sin(a)))
        end = point
    else:
        raise ValueError('unsupported helix axis')
    for i in range(rings + 1):
        theta = math.tau * turns * i / rings
        h = h0 + (h1 - h0) * i / rings
        rdir = rad(theta)
        c = point(theta, h)
        ring = []
        for k in range(tube_segments):
            phi = math.tau * k / tube_segments
            p = c + tube_radius * (math.cos(phi) * rdir + math.sin(phi) * axis_dir)
            ring.append(len(verts))
            verts.append(tuple(p))
        if i:
            for k in range(tube_segments):
                k2 = (k + 1) % tube_segments
                a = ring[k] - tube_segments
                b = ring[k2] - tube_segments
                faces.append((a, b, ring[k2], ring[k]))
    s0 = len(verts)
    verts.append(tuple(end(0.0, h0)))
    s1 = len(verts)
    theta = math.tau * turns
    verts.append(tuple(end(theta, h1)))
    for k in range(tube_segments):
        k2 = (k + 1) % tube_segments
        faces.append((s0, (k + 1) % tube_segments, k))
        faces.append((s1, rings * tube_segments + k,
                      rings * tube_segments + k2))
    obj = mesh_object(name, verts, faces)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return finish(obj, name, tile, bevel=bevel, smooth=True)


def curved_label_patch(name, radius, thickness, y_center, y_half,
                       angle_center, angle_half, tile, *, seg_a=36, seg_y=6):
    """Closed thin curved plate carrying the printed label on the +X side."""
    verts = []

    def index(layer, iy, ia):
        return layer * (seg_y + 1) * (seg_a + 1) + iy * (seg_a + 1) + ia

    for layer, r in ((0, radius), (1, radius + thickness)):
        for iy in range(seg_y + 1):
            y = y_center - y_half + 2 * y_half * iy / seg_y
            for ia in range(seg_a + 1):
                a = angle_center - angle_half + 2 * angle_half * ia / seg_a
                verts.append((r * math.cos(a), y, r * math.sin(a)))
    faces, uvs = [], []

    def uv(iy, ia):
        # u runs along the cylinder axis (label width); v runs around the
        # cross-section (label height), so the drawn label keeps its aspect.
        return ((iy / seg_y), (ia / seg_a))

    for iy in range(seg_y):
        for ia in range(seg_a):
            # outer surface
            q = (index(1, iy, ia), index(1, iy, ia + 1),
                 index(1, iy + 1, ia + 1), index(1, iy + 1, ia))
            faces.append(q)
            uvs.append([uv(iy, ia), uv(iy, ia + 1),
                        uv(iy + 1, ia + 1), uv(iy + 1, ia)])
            # inner surface
            q = (index(0, iy, ia), index(0, iy + 1, ia),
                 index(0, iy + 1, ia + 1), index(0, iy, ia + 1))
            faces.append(q)
            uvs.append([uv(iy, ia), uv(iy + 1, ia),
                        uv(iy + 1, ia + 1), uv(iy, ia + 1)])
    for ia in range(seg_a):
        q = (index(0, 0, ia), index(1, 0, ia), index(1, 0, ia + 1),
             index(0, 0, ia + 1))
        faces.append(q)
        uvs.append([uv(0, ia), uv(0, ia), uv(0, ia + 1), uv(0, ia + 1)])
        q = (index(0, seg_y, ia), index(0, seg_y, ia + 1),
             index(1, seg_y, ia + 1), index(1, seg_y, ia))
        faces.append(q)
        uvs.append([uv(seg_y, ia), uv(seg_y, ia + 1),
                    uv(seg_y, ia + 1), uv(seg_y, ia)])
    for iy in range(seg_y):
        q = (index(0, iy, 0), index(0, iy + 1, 0), index(1, iy + 1, 0),
             index(1, iy, 0))
        faces.append(q)
        uvs.append([uv(iy, 0), uv(iy + 1, 0), uv(iy + 1, 0), uv(iy, 0)])
        q = (index(0, iy, seg_a), index(1, iy, seg_a),
             index(1, iy + 1, seg_a), index(0, iy + 1, seg_a))
        faces.append(q)
        uvs.append([uv(iy, seg_a), uv(iy, seg_a),
                    uv(iy + 1, seg_a), uv(iy + 1, seg_a)])
    obj = mesh_object(name, verts, faces, uvs)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return finish(obj, name, tile, bevel=0, smooth=True)


# ---------------------------------------------------------------------------
# Boolean helper
# ---------------------------------------------------------------------------
def apply_boolean(target, cutters, name):
    for cutter in cutters:
        cutter.hide_viewport = False
        cutter.hide_render = False
        cutter.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    for cutter in cutters:
        cutter.select_set(True)
    bpy.context.view_layer.objects.active = cutters[0]
    if len(cutters) > 1:
        bpy.ops.object.join()
    cutter = bpy.context.active_object
    mod = target.modifiers.new(name, 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    mod.material_mode = 'TRANSFER'
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    bm = bmesh.new()
    bm.from_mesh(target.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(target.data)
    bm.free()
    target.data.update()


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def build_cap(side):
    front = side == 'front'
    cap = D['cap']
    if front:
        y0, y1 = cap['front_y0'], cap['front_y1']
        cy = (y0 + y1) / 2
        outer_y, inner_y = y0, y1
        direction = 1          # outward is -Y
    else:
        y0, y1 = cap['rear_y0'], cap['rear_y1']
        cy = (y0 + y1) / 2
        outer_y, inner_y = y1, y0
        direction = -1         # outward is +Y
    face_sign = -1 if front else 1
    cap_obj = box(f'cap_{side}', (0, cy, 0),
                  (cap['size_x'], cap['thickness'], cap['size_z']),
                  TILE['cap'], bevel=cap['bevel'])
    cutters = []
    ox, oz = cap['corner_offset_x'], cap['corner_offset_z']
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * ox, sz * oz
            cutters.append(along(f'cut_corner_{side}_{sx}_{sz}',
                                 (x, y0 - .01, z), (x, y1 + .01, z),
                                 cap['corner_hole_radius'], TILE['rubber'],
                                 segs=40, bevel=0, cutter=True))
    if front:
        cutters.append(along(f'cut_rod_hole_{side}', (0, y0 - .01, 0),
                             (0, y1 + .01, 0), cap['rod_hole_radius'],
                             TILE['rubber'], segs=64, bevel=0, cutter=True))
    p = D['ports']
    for sx in (-1, 1):
        # large threaded air port, blind, on both side faces
        cutters.append(along(f'cut_port_{side}_{sx}',
                             (sx * (p['x_face'] + .01), cy, p['port_z']),
                             (sx * (p['x_face'] - p['bore_depth']), cy, p['port_z']),
                             p['bore_radius'], TILE['rubber'], segs=48,
                             bevel=0, cutter=True))
        # shallow brass-plug seat
        cutters.append(along(f'cut_brass_{side}_{sx}',
                             (sx * (p['x_face'] + .01), cy, p['brass_z']),
                             (sx * (p['x_face'] - p['brass_recess_depth']),
                              cy, p['brass_z']),
                             p['brass_recess_radius'], TILE['rubber'], segs=40,
                             bevel=0, cutter=True))
    apply_boolean(cap_obj, cutters, f'Cut {side} cap tie holes, rod and side ports')
    for poly in cap_obj.data.polygons:
        poly.use_smooth = False

    prefix = 'front' if front else 'rear'
    rib_y = outer_y + face_sign * (cap['rib_depth'] / 2)
    frame_y = rib_y
    frame_outer = cap['frame_outer']
    bar = cap['frame_width']
    # square raised rim: four bars
    box(f'cap_frame_{prefix}_top', (0, frame_y, frame_outer - bar / 2),
        (2 * frame_outer, cap['frame_depth'], bar), TILE['cap'], bevel=.0009)
    box(f'cap_frame_{prefix}_bottom', (0, frame_y, -frame_outer + bar / 2),
        (2 * frame_outer, cap['frame_depth'], bar), TILE['cap'], bevel=.0009)
    box(f'cap_frame_{prefix}_left', (-frame_outer + bar / 2, frame_y, 0),
        (bar, cap['frame_depth'], 2 * (frame_outer - bar)), TILE['cap'],
        bevel=.0009)
    box(f'cap_frame_{prefix}_right', (frame_outer - bar / 2, frame_y, 0),
        (bar, cap['frame_depth'], 2 * (frame_outer - bar)), TILE['cap'],
        bevel=.0009)
    # six radial ribs: vertical up/down plus four diagonals
    rib = cap
    rib_len = rib['rib_outer_radius'] - rib['rib_inner_radius']
    rc = (rib['rib_inner_radius'] + rib['rib_outer_radius']) / 2
    for i, a in enumerate((math.pi / 2, 3 * math.pi / 2,
                           math.pi / 4, 3 * math.pi / 4,
                           5 * math.pi / 4, 7 * math.pi / 4), 1):
        center = (rc * math.cos(a), rib_y, rc * math.sin(a))
        rotated_box(f'cap_rib_{prefix}_{i}', center,
                    (rib_len + .006, cap['rib_depth'], cap['rib_width']),
                    -a, TILE['cap'], bevel=.0008)
    # raised corner bosses under the washers
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * ox, sz * oz
            along(f'cap_boss_{prefix}_{sx}_{sz}',
                  (x, outer_y, z), (x, outer_y + face_sign * cap['corner_boss_depth'], z),
                  cap['corner_boss_radius'], TILE['cap'], segs=40, bevel=.0008)
    if front:
        g = D['gland']
        base = along('gland_base', (0, g['base_y0'], 0), (0, g['base_y1'], 0),
                     g['base_radius'], TILE['cap'], segs=64, bevel=g['bevel'])
        cut = along('cut_gland_base_rod', (0, g['base_y0'] - .01, 0),
                    (0, g['base_y1'] + .01, 0), cap['rod_hole_radius'],
                    TILE['rubber'], segs=64, bevel=0, cutter=True)
        apply_boolean(base, [cut], 'Drill gland base rod clearance')
        main = along('gland', (0, g['y0'], 0), (0, g['y1'], 0),
                     g['radius'], TILE['cap'], segs=64, bevel=g['bevel'])
        # g.y0 is nearer the cap and g.y1 is the outer tip, so use min/max
        # rather than y0-.01/y1+.01 (which would collapse to zero length).
        cut = along('cut_gland_rod',
                    (0, min(g['y0'], g['y1']) - .01, 0),
                    (0, max(g['y0'], g['y1']) + .01, 0),
                    cap['rod_hole_radius'], TILE['rubber'], segs=64, bevel=0,
                    cutter=True)
        apply_boolean(main, [cut], 'Drill gland rod clearance')
        torus('gland_wiper', g['wiper_major'], g['wiper_minor'],
              (0, g['wiper_y'], 0), 'Y', TILE['rubber'],
              major_segments=64, minor_segments=10)
    else:
        b = D['rear_boss']
        boss = along('rear_boss', (0, b['y0'], 0), (0, b['y1'], 0),
                     b['radius'], TILE['cap'], segs=64, bevel=.0015)
        cut = along('cut_rear_boss_hole',
                    (0, b['y1'] + .01, 0),
                    (0, b['y1'] - cap['rear_center_hole_depth'], 0),
                    cap['rear_center_hole_radius'], TILE['rubber'], segs=32,
                    bevel=0, cutter=True)
        apply_boolean(boss, [cut], 'Drill rear boss centre recess')
        torus('rear_boss_groove', b['groove_major'], b['groove_minor'],
              (0, b['groove_y'], 0), 'Y', TILE['dark_steel'],
              major_segments=64, minor_segments=8)

    # side ports: brass hex-socket plug above, dark threaded bore below
    for sx in (-1, 1):
        brass = along(f'brass_{prefix}_{sx}',
                      (sx * (p['x_face'] - p['brass_thickness']), cy, p['brass_z']),
                      (sx * (p['x_face'] + .0002), cy, p['brass_z']),
                      p['brass_radius'], TILE['brass'], segs=40, bevel=.0006)
        # real hex socket cut into the brass plug, with a dark bottom insert
        socket_cut = hex_prism_axis(
            f'cut_brass_socket_{prefix}_{sx}', p['brass_socket_radius'] + .0003,
            .012, (sx * (p['x_face'] + .004), cy, p['brass_z']), 'X',
            TILE['rubber'], bevel=0, cutter=True)
        apply_boolean(brass, [socket_cut], 'Cut brass hex socket')
        hex_prism_axis(f'brass_socket_{prefix}_{sx}', p['brass_socket_radius'] - .0002,
                       .0005, (sx * (p['x_face'] - .0021), cy, p['brass_z']),
                       'X', TILE['port_dark'], bevel=.0001)
        helix(f'port_thread_{prefix}_{sx}', 'X', (cy, p['port_z']),
              p['thread_major'], p['thread_tube'], p['thread_pitch'],
              sx * (p['x_face'] - p['thread_start_inset']),
              sx * (p['x_face'] - p['thread_end_inset']), TILE['thread'],
              seg_per_turn=14, tube_segments=8)
        along(f'port_bottom_{prefix}_{sx}',
              (sx * (p['x_face'] - p['bottom_disc_inset'] - .0006), cy, p['port_z']),
              (sx * (p['x_face'] - p['bottom_disc_inset'] + .0006), cy, p['port_z']),
              p['bore_radius'] - .0002, TILE['port_dark'], segs=40, bevel=0)
    return cap_obj


def build_barrel():
    b = D['barrel']
    along('barrel', (0, b['y0'], 0), (0, b['y1'], 0), b['radius'],
          TILE['aluminum'], segs=128, bevel=.0015)
    for sign, name in ((-1, 'front'), (1, 'rear')):
        torus(f'barrel_seam_{name}', b['seam_radius'], b['seam_minor'],
              (0, sign * b['seam_y'], 0), 'Y', TILE['rubber'],
              major_segments=128, minor_segments=10)
    lab = D['barrel_label']
    curved_label_patch('label', lab['radius'], lab['thickness'],
                       lab['y_center'], lab['y_width'] / 2,
                       math.radians(lab['arc_center_deg']),
                       math.radians(lab['arc_span_deg']) / 2, TILE['label'],
                       seg_a=40, seg_y=8)


def build_tie_rods():
    t = D['tie_rods']
    index = 0
    for sx in (-1, 1):
        for sz in (-1, 1):
            index += 1
            x, z = sx * t['offset_x'], sz * t['offset_z']
            along(f'tie_rod_{index}', (x, t['y0'], z), (x, t['y1'], z),
                  t['radius'], TILE['steel'], segs=40, bevel=.0003)
            for sign, name in ((-1, 'front'), (1, 'rear')):
                if name == 'front':
                    h0, h1 = t['thread_y_front'], t['thread_y_front_end']
                else:
                    h0, h1 = t['thread_y_rear_end'], t['thread_y_rear']
                helix(f'tie_thread_{index}_{name}', 'Y', (x, z),
                      t['thread_major'], t['thread_tube'], t['thread_pitch'],
                      h0, h1, TILE['thread'], seg_per_turn=12,
                      tube_segments=8)
                wy = t['washer_y_front'] if name == 'front' else t['washer_y_rear']
                torus(f'tie_washer_{name}_{sx}_{sz}', t['washer_major'],
                      t['washer_minor'], (x, wy, z), 'Y', TILE['dark_steel'],
                      major_segments=40, minor_segments=8)
                ny = t['nut_y_front'] if name == 'front' else t['nut_y_rear']
                hex_prism_axis(f'tie_nut_{name}_{sx}_{sz}', t['nut_radius'],
                               t['nut_thickness'], (x, ny, z), 'Y',
                               TILE['steel'], bevel=.0007)
                ey = t['y0'] if name == 'front' else t['y1']
                along(f'tie_end_dot_{index}_{name}',
                      (x, ey - .0002 * sign, z), (x, ey + .0002 * sign, z),
                      t['end_dot_radius'], TILE['port_dark'], segs=20, bevel=0)


def build_rod():
    r = D['rod']
    along('rod', (0, r['shaft_y0'], 0), (0, r['shaft_y1'], 0), r['radius'],
          TILE['chrome'], segs=96, bevel=.0012)
    along('rod_thread_core', (0, r['thread_y0'], 0), (0, r['thread_y1'], 0),
          r['thread_core_radius'], TILE['thread'], segs=96, bevel=.0006)
    helix('rod_thread', 'Y', (0, 0), r['thread_major'], r['thread_tube'],
          r['thread_pitch'], r['thread_y0'], r['thread_y1'], TILE['thread'],
          seg_per_turn=20, tube_segments=8)
    hex_prism_axis('rod_nut', r['nut_radius'], r['nut_thickness'],
                   (0, r['nut_y'], 0), 'Y', TILE['steel'], bevel=.0010)
    along('rod_end_dot', (0, r['end_y'] - .0002, 0), (0, r['end_y'] + .0003, 0),
          r['end_dot_radius'], TILE['port_dark'], segs=32, bevel=.0002)


def build_all():
    build_cap('front')
    build_cap('rear')
    build_barrel()
    build_tie_rods()
    build_rod()


# ---------------------------------------------------------------------------
# Scene, camera, rays, export and renders
# ---------------------------------------------------------------------------
def init_camera(scene):
    data = bpy.data.cameras.new('review camera')
    cam = bpy.data.objects.new('review camera', data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_camera(cam, eye, target, *, ortho_scale=None, fov=40, shift_x=0, shift_y=0,
               top_orientation=False):
    cam.location = Vector(eye)
    cam.rotation_mode = 'QUATERNION'
    quat = (Vector(target) - cam.location).to_track_quat('-Z', 'Y')
    if top_orientation:
        quat = quat @ Quaternion((0, 0, 1), math.pi / 2)
    cam.rotation_quaternion = quat
    cam.data.type = 'ORTHO' if ortho_scale else 'PERSP'
    if ortho_scale:
        cam.data.ortho_scale = ortho_scale
    else:
        cam.data.angle = math.radians(fov)
    cam.data.shift_x = shift_x
    cam.data.shift_y = shift_y


def look(scene, direction, center, span, *, ortho_scale=None):
    direction = Vector(direction).normalized()
    set_camera(CAM, Vector(center) + direction * span * 5, center,
               ortho_scale=ortho_scale or span * 1.27,
               top_orientation=abs(direction.z) > .999)
    return CAM


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


def object_center(name):
    obj = bpy.data.objects[name]
    lo, hi, center, span = bounds([obj])
    return center


def intersecting_faces(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
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


def draw(scene, path, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def set_visible(names):
    for obj in objects:
        obj.hide_render = obj.name not in names


def make_bvh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    matrix = obj.matrix_world
    verts = [matrix @ v.co for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(verts, faces, epsilon=1e-7)


def ray_checks():
    """Measure the real side-port depths, tie-rod holes and rod opening."""
    cap = D['cap']
    p = D['ports']
    checks = {'ports': {}, 'tie_holes': {}, 'brass_seats': {}, 'rod_hole': {},
              'rear_center_recess': {}}
    for side, cy in (('front', p['front_cap_y']), ('rear', p['rear_cap_y'])):
        tree = make_bvh(bpy.data.objects[f'cap_{side}'])
        for sx in (-1, 1):
            # big threaded port at z=port_z
            origin = Vector((sx * .20, cy, p['port_z']))
            direction = Vector((-sx, 0, 0))
            hit = tree.ray_cast(origin, direction, 1.0)
            depth = None if hit[0] is None else round(
                abs(p['x_face'] - abs(hit[0].x)), 5)
            checks['ports'][f'{side}_{sx}'] = {
                'hit': bool(hit[0] is not None), 'bore_depth_m': depth,
                'real_recess': bool(hit[0] is not None and depth is not None
                                    and depth >= .020)}
            # shallow brass plug seat at z=brass_z
            origin = Vector((sx * .20, cy, p['brass_z']))
            hit = tree.ray_cast(origin, direction, 1.0)
            seat_depth = None if hit[0] is None else round(
                abs(p['x_face'] - abs(hit[0].x)), 5)
            checks['brass_seats'][f'{side}_{sx}'] = {
                'hit': bool(hit[0] is not None), 'seat_depth_m': seat_depth,
                'present': bool(hit[0] is not None and seat_depth is not None
                                and .0015 <= seat_depth <= .010)}
    for side in ('front', 'rear'):
        tree = make_bvh(bpy.data.objects[f'cap_{side}'])
        y_out = (-.20 if side == 'front' else .20)
        direction = Vector((0, 1, 0)) if side == 'front' else Vector((0, -1, 0))
        holes = []
        for sx in (-1, 1):
            for sz in (-1, 1):
                x, z = sx * cap['corner_offset_x'], sz * cap['corner_offset_z']
                hit = tree.ray_cast(Vector((x, y_out, z)), direction, 1.0)
                holes.append({'corner': [sx, sz], 'hit': hit[0] is not None})
        checks['tie_holes'][side] = {
            'all_through': all(not h['hit'] for h in holes), 'rays': holes}
    tree = make_bvh(bpy.data.objects['cap_front'])
    hit = tree.ray_cast(Vector((0, -.20, 0)), Vector((0, 1, 0)), 1.0)
    checks['rod_hole'] = {'through': hit[0] is None, 'hit': hit[0] is not None}
    tree = make_bvh(bpy.data.objects['rear_boss'])
    hit = tree.ray_cast(Vector((0, .20, 0)), Vector((0, -1, 0)), 1.0)
    recess = None if hit[0] is None else round(.1535 - hit[0].y, 5)
    checks['rear_center_recess'] = {
        'hit': bool(hit[0] is not None), 'depth_m': recess,
        'real_recess': bool(hit[0] is not None and recess is not None
                            and recess >= .003)}
    depths = [v['bore_depth_m'] for v in checks['ports'].values()
              if v['bore_depth_m'] is not None]
    checks['port_bore_depth_min_m'] = min(depths) if depths else None
    checks['passed'] = bool(
        all(v['real_recess'] for v in checks['ports'].values())
        and all(v['present'] for v in checks['brass_seats'].values())
        and all(v['all_through'] for v in checks['tie_holes'].values())
        and checks['rod_hole']['through']
        and checks['rear_center_recess']['real_recess'])
    return checks


def main():
    global MATERIAL, CAM
    scene = legacy.setup_scene('CPU', 512)
    MATERIAL = atlas_material()
    CAM = init_camera(scene)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .42
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 90
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 3.0
        fill.data.size_y = 3.0
        fill.location = (-3.0, 2.5, 2.0)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 180, 3.0),
            ('front reflection', (3.0, 1.4, 1.0), 90, 1.4),
            ('top fill', (0.0, 0.0, 4.0), 55, 2.2)):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy = energy
        data.shape = 'RECTANGLE'
        data.size = size
        data.size_y = size
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

    build_all()
    bpy.context.view_layer.update()
    ray = ray_checks()
    low, high, center, span = bounds(objects)
    shift = -center
    for obj in objects:
        obj.location += shift
    bpy.context.view_layer.update()
    low, high, center, span = bounds(objects)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'), export_format='GLB',
                              use_selection=True, export_yup=False, export_apply=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

    parts = [{'name': obj.name, 'vertices': len(obj.data.vertices),
              'faces': len(obj.data.polygons), 'tile': tiles.get(obj.name)}
             for obj in objects]
    intersections = {obj.name: intersecting_faces(obj) for obj in objects}
    render_dir = out / 'renders'
    render_dir.mkdir(exist_ok=True)
    # The reference sheet labels: Front = observed label side (+X),
    # Back = opposite side (-X), Left = rod end (-Y), Right = rear end (+Y).
    directions = {'front': (1, 0, .18), 'back': (-1, 0, .18),
                  'left': (0, -1, .18), 'right': (0, 1, .18),
                  'top': (0, 0, 1), 'bottom': (0, 0, -1)}
    rr = cfg['render']
    for name, direction in directions.items():
        look(scene, direction, center, span)
        draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    source = request['source_camera']
    set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
               shift_x=source.get('shift_x', 0), shift_y=source.get('shift_y', 0))
    draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])
    # uncalibrated three-quarter view comparable to the source photograph;
    # orthographic so the object fills the frame for the side-by-side board
    look(scene, (0.95, -1.0, 0.38), center, span, ortho_scale=span * 1.22)
    draw(scene, render_dir / 'new-3q.png', 512, rr['six_view_samples'])

    if not request.get('draft'):
        shift_v = Vector(shift)
        target = {
            'port-front': Vector((.09, D['ports']['front_cap_y'], 0)) + shift_v,
            'port-rear': Vector((.09, D['ports']['rear_cap_y'], 0)) + shift_v,
            'front-face': object_center('cap_front'),
            'rod-end': object_center('rod_nut'),
            'tie-rod-end': object_center('tie_nut_front_1_1'),
            'rear-boss': object_center('rear_boss'),
            'label': object_center('label'),
            'barrel-seam': object_center('barrel_seam_front'),
        }
        detail_specs = [
            ('port-front', (1.0, -0.55, 0.35), target['port-front'], .160),
            ('port-rear', (1.0, 0.55, 0.35), target['port-rear'], .160),
            ('front-face', (1.0, -0.85, 0.30), target['front-face'], .235),
            ('rod-end', (0.9, -1.0, 0.30), target['rod-end'], .130),
            ('tie-rod-end', (0.8, -1.0, 0.45), target['tie-rod-end'], .065),
            ('rear-boss', (0.9, 1.0, 0.32), target['rear-boss'], .130),
            ('label', (1.0, -0.30, 0.22), target['label'], .105),
            ('barrel-seam', (0.8, -0.55, 0.55), target['barrel-seam'], .120),
            ('overall-3q', (1.0, -1.0, 0.55), center, span * 1.14),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        cap_front_names = {'cap_front', 'gland_base', 'gland', 'gland_wiper'}
        cap_rear_names = {'cap_rear', 'rear_boss', 'rear_boss_groove'}
        for prefix in ('front', 'rear'):
            dest = cap_front_names if prefix == 'front' else cap_rear_names
            dest |= {f'cap_frame_{prefix}_{n}' for n in ('top', 'bottom', 'left', 'right')}
            dest |= {f'cap_rib_{prefix}_{i}' for i in range(1, 7)}
            dest |= {f'cap_boss_{prefix}_{sx}_{sz}' for sx in (-1, 1)
                     for sz in (-1, 1)}
            dest |= {f'tie_nut_{prefix}_{sx}_{sz}' for sx in (-1, 1)
                     for sz in (-1, 1)}
            dest |= {f'tie_washer_{prefix}_{sx}_{sz}' for sx in (-1, 1)
                     for sz in (-1, 1)}
            dest |= {f'brass_{prefix}_{sx}' for sx in (-1, 1)}
            dest |= {f'brass_socket_{prefix}_{sx}' for sx in (-1, 1)}
            dest |= {f'port_thread_{prefix}_{sx}' for sx in (-1, 1)}
            dest |= {f'port_bottom_{prefix}_{sx}' for sx in (-1, 1)}
        cap_all = cap_front_names | cap_rear_names
        set_visible(cap_all)
        look(scene, (1.0, 0.0, 0.12), center, span, ortho_scale=.40)
        draw(scene, render_dir / 'caps-only-side.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible(cap_all)
        look(scene, (0, -1.0, 0.10), center, span, ortho_scale=.235)
        draw(scene, render_dir / 'caps-only-ends.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible(cap_front_names)
        look(scene, (0.9, -1.0, 0.28), object_center('cap_front'), span,
             ortho_scale=.235)
        draw(scene, render_dir / 'front-cap-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible(cap_rear_names)
        look(scene, (0.9, 1.0, 0.30), object_center('cap_rear'), span,
             ortho_scale=.235)
        draw(scene, render_dir / 'rear-cap-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'rod', 'rod_thread_core', 'rod_thread', 'rod_nut',
                     'rod_end_dot', 'gland_base', 'gland', 'gland_wiper'})
        look(scene, (0.85, -1.0, 0.35), object_center('rod_nut'), span,
             ortho_scale=.235)
        draw(scene, render_dir / 'rod-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        barrel_names = {'barrel', 'label', 'barrel_seam_front',
                        'barrel_seam_rear'} | {f'tie_rod_{i}' for i in range(1, 5)}
        barrel_names |= {f'tie_thread_{i}_{n}' for i in range(1, 5)
                         for n in ('front', 'rear')}
        barrel_names |= {f'tie_nut_{n}_{sx}_{sz}' for n in ('front', 'rear')
                         for sx in (-1, 1) for sz in (-1, 1)}
        barrel_names |= {f'tie_washer_{n}_{sx}_{sz}' for n in ('front', 'rear')
                         for sx in (-1, 1) for sz in (-1, 1)}
        barrel_names |= {f'tie_end_dot_{i}_{n}' for i in range(1, 5)
                         for n in ('front', 'rear')}
        set_visible(barrel_names)
        look(scene, (0.55, -0.75, 0.55), center, span, ortho_scale=.36)
        draw(scene, render_dir / 'barrel-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({obj.name for obj in objects})
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-material-front.png', rr['six_view_resolution'],
             rr['six_view_samples'])

        clay = bpy.data.materials.new('Clay')
        clay.use_nodes = True
        clay_b = clay.node_tree.nodes.get('Principled BSDF')
        clay_b.inputs['Base Color'].default_value = (.55, .56, .54, 1)
        clay_b.inputs['Roughness'].default_value = .82
        clay_b.inputs['Metallic'].default_value = 0
        for obj in objects:
            obj.data.materials.clear()
            obj.data.materials.append(clay)
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-clay-front.png', rr['six_view_resolution'],
             rr['six_view_samples'])
        for obj in objects:
            obj.data.materials.clear()
            obj.data.materials.append(MATERIAL)

        main_light = bpy.data.objects['main softbox']
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-light-alternate.png', rr['six_view_resolution'],
             rr['six_view_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()

        set_visible(set(metal_objects))
        fill_light = bpy.data.objects.get('soft light')
        front_light = bpy.data.objects.get('front reflection')
        top_light = bpy.data.objects.get('top fill')
        for light in (fill_light, front_light, top_light):
            if light is not None:
                light.hide_render = True
        metal_target = object_center('rod_nut')
        look(scene, (0.9, -1.0, 0.55), metal_target, span, ortho_scale=.31)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (0.9, -1.0, 0.55), metal_target, span, ortho_scale=.31)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for light in (fill_light, front_light, top_light):
            if light is not None:
                light.hide_render = False
        set_visible({obj.name for obj in objects})

        for i in range(24):
            a = math.tau * i / 24
            look(scene, (math.cos(a), math.sin(a), .24), center, span)
            draw(scene, render_dir / f'turn-{i:02d}.png', rr['turntable_resolution'],
                 rr['turntable_samples'])

    for obj in objects:
        obj.hide_render = True
    old = legacy.load_mesh(request['original_mesh'])
    _, _, oldcenter, oldspan = bounds([old])
    for name, direction in directions.items():
        look(scene, direction, oldcenter, oldspan)
        draw(scene, render_dir / f'old-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    data = {
        'parts': parts,
        'bounds': [list(low), list(high)],
        'centre': list(center),
        'span': span,
        'polygons': sum(len(obj.data.polygons) for obj in objects),
        'triangles': sum(max(0, len(p.vertices) - 2)
                         for obj in objects for p in obj.data.polygons),
        'texture_size': cfg['texture']['size'],
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH 同部件不共享顶点的面交叠；零件之间故意装配相交，未判定',
        'metal_objects': metal_objects,
        'nonmetal_objects': nonmetal_objects,
        'ray_checks': ray,
        'tiles': tiles,
        'geometry_basis': '原照可见结构 + 未标定六视图（ARK 拉杆式气缸）',
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'shift': list(shift),
    }
    (out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                            encoding='utf8')


if __name__ == '__main__':
    main()
