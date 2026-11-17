"""Blender worker: part-based rebuild of item 22, a tie-rod pneumatic cylinder.

The source photograph is a three-quarter view of a grey-capped, silver-barrel
ISO-15552-style cylinder with four tie rods, a chrome piston rod with two hex
nuts, brass-coloured corner nuts, and green G3/8 port markings on the top face
of both end caps.  The six-view sheet is an uncalibrated visual match sheet;
its bottom panel repeats the green ring on the bottom face, which contradicts
the photograph and the front/back/top panels, so the model keeps a real top-face
port only.  Dimensions are a plausible SC80-style working scale, not measured.

Coordinates: Z up, cylinder axis along Y, rod end at -Y (front), rear at +Y.
All visible parts are separate watertight meshes.  Metal parts keep metallic
PBR response; the green print, rubber seals and the green port ring are
dielectric.  No lighting or shadow is baked into the atlas.
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
DYNAMIC_TILES = {'aluminum', 'cast', 'chrome', 'steel', 'brass', 'thread', 'dark_steel'}
DIELECTRIC_TILES = {'green', 'rubber', 'port_ring'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed aluminium, steel, brass and green print PBR atlas')
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
    normal_map.inputs['Strength'].default_value = .32
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


def cylinder_z(name, radius, z0, z1, tile, *, x=0.0, y=0.0, segs=64, bevel=.001,
               constant=False, cutter=False):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius, depth=z1 - z0,
                                        location=(x, y, (z0 + z1) / 2))
    return finish(bpy.context.object, name, tile, bevel=bevel, constant=constant,
                  cutter=cutter)


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


def hex_prism(name, radius, thickness, center_y, tile, *, x=0.0, z=0.0, bevel=.0008):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius, depth=thickness,
                                        location=(x, center_y, z),
                                        rotation=(math.pi / 2, 0, 0))
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, smooth=False)


def helix(name, axis, center, radius, tube_radius, pitch, h0, h1, tile, *,
          seg_per_turn=16, tube_segments=8, direction=1, bevel=0.0):
    """Closed helical tube used as a visual thread ridge."""
    if axis == 'Y':
        length = abs(h1 - h0)
        turns = length / pitch
        rings = max(4, int(math.ceil(turns * seg_per_turn)))
        verts, faces = [], []
        for i in range(rings + 1):
            theta = math.tau * turns * i / rings
            cy = h0 + (h1 - h0) * i / rings
            rad = Vector((math.cos(theta), 0.0, math.sin(theta)))
            axis_dir = Vector((0.0, 1.0, 0.0))
            c = Vector((center[0] + radius * rad.x, cy, center[1] + radius * rad.z))
            ring = []
            for k in range(tube_segments):
                phi = math.tau * k / tube_segments
                p = c + tube_radius * (math.cos(phi) * rad + math.sin(phi) * axis_dir)
                ring.append(len(verts))
                verts.append(tuple(p))
            if i:
                for k in range(tube_segments):
                    k2 = (k + 1) % tube_segments
                    a = ring[k] - tube_segments
                    b = ring[k2] - tube_segments
                    faces.append((a, b, ring[k2], ring[k]))
        # cap both ends
        s0 = len(verts)
        verts.append((center[0] + radius, h0, center[1]))
        s1 = len(verts)
        verts.append((center[0] + radius * math.cos(math.tau * turns),
                      h1, center[1] + radius * math.sin(math.tau * turns)))
        for k in range(tube_segments):
            k2 = (k + 1) % tube_segments
            faces.append((s0, (k + 1) % tube_segments, k))
            faces.append((s1, rings * tube_segments + k, rings * tube_segments + k2))
    else:  # Z axis, used for the tapped top-face ports
        length = abs(h1 - h0)
        turns = length / pitch
        rings = max(4, int(math.ceil(turns * seg_per_turn)))
        verts, faces = [], []
        for i in range(rings + 1):
            theta = math.tau * turns * i / rings
            cz = h0 + (h1 - h0) * i / rings
            rad = Vector((math.cos(theta), math.sin(theta), 0.0))
            axis_dir = Vector((0.0, 0.0, 1.0))
            c = Vector((center[0] + radius * rad.x, center[1] + radius * rad.y, cz))
            ring = []
            for k in range(tube_segments):
                phi = math.tau * k / tube_segments
                p = c + tube_radius * (math.cos(phi) * rad + math.sin(phi) * axis_dir)
                ring.append(len(verts))
                verts.append(tuple(p))
            if i:
                for k in range(tube_segments):
                    k2 = (k + 1) % tube_segments
                    a = ring[k] - tube_segments
                    b = ring[k2] - tube_segments
                    faces.append((a, b, ring[k2], ring[k]))
        s0 = len(verts)
        verts.append((center[0] + radius, center[1], h0))
        s1 = len(verts)
        verts.append((center[0] + radius * math.cos(math.tau * turns),
                      center[1] + radius * math.sin(math.tau * turns), h1))
        for k in range(tube_segments):
            k2 = (k + 1) % tube_segments
            faces.append((s0, (k + 1) % tube_segments, k))
            faces.append((s1, rings * tube_segments + k, rings * tube_segments + k2))
    obj = mesh_object(name, verts, faces)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return finish(obj, name, tile, bevel=bevel, smooth=True)


def revolve_y(name, profile, tile, *, segs=64, cutter=False):
    """Closed solid of revolution around Y; profile is [(y, radius), ...]."""
    verts, rings = [], []
    for y, r in profile:
        start = len(verts)
        for j in range(segs):
            a = math.tau * j / segs
            verts.append((r * math.cos(a), y, r * math.sin(a)))
        rings.append(list(range(start, start + segs)))
    faces = []
    for i in range(len(profile) - 1):
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((rings[i][j], rings[i][j2], rings[i + 1][j2],
                          rings[i + 1][j]))
    c0 = len(verts)
    verts.append((0, profile[0][0], 0))
    c1 = len(verts)
    verts.append((0, profile[-1][0], 0))
    for j in range(segs):
        j2 = (j + 1) % segs
        faces.append((c0, rings[0][j2], rings[0][j]))
        faces.append((c1, rings[-1][j], rings[-1][j2]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, tile, bevel=0, smooth=True, cutter=cutter)


def stepped_hole(name, outer_y, inner_y, r_large, r_small, depth, tile, *,
                 segs=56, direction=1, cutter=True):
    """One closed cutter with a large outer counterbore and a smaller through hole."""
    y_out = outer_y - direction * .010
    y_shoulder = outer_y + direction * depth
    y_in = inner_y + direction * .010
    profile = [(y_out, r_large), (y_shoulder, r_large),
               (y_shoulder, r_small), (y_in, r_small)]
    return revolve_y(name, profile, tile, segs=segs, cutter=cutter)


def annulus(name, center, z0, z1, r_in, r_out, tile, *, segs=96, uv_rotate=0.0):
    """Closed thin annular prism; planar UVs sample the printed port ring."""
    cx, cy = center
    verts, faces, loop_uvs = [], [], []
    rings = []
    ca, sa = math.cos(uv_rotate), math.sin(uv_rotate)

    def uv(x, y):
        u = (x - cx) / (2 * r_out) + .5
        v = (y - cy) / (2 * r_out) + .5
        du, dv = u - .5, v - .5
        return (.5 + ca * du - sa * dv, .5 + sa * du + ca * dv)

    for z in (z0, z1):
        for r in (r_in, r_out):
            start = len(verts)
            for j in range(segs):
                a = math.tau * j / segs
                verts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
            rings.append(list(range(start, start + segs)))
    bi, bo, ti, to = rings
    for j in range(segs):
        j2 = (j + 1) % segs
        # outer wall
        q = (bo[j], bo[j2], to[j2], to[j])
        faces.append(q)
        loop_uvs.append([uv(*verts[i][:2]) for i in q])
        # inner wall
        q = (bi[j2], bi[j], ti[j], ti[j2])
        faces.append(q)
        loop_uvs.append([uv(*verts[i][:2]) for i in q])
        # top face
        q = (ti[j], ti[j2], to[j2], to[j])
        faces.append(q)
        loop_uvs.append([uv(*verts[i][:2]) for i in q])
        # bottom face
        q = (bo[j2], bo[j], bi[j], bi[j2])
        faces.append(q)
        loop_uvs.append([uv(*verts[i][:2]) for i in q])
    obj = mesh_object(name, verts, faces, loop_uvs)
    for poly in obj.data.polygons:
        poly.use_smooth = False
    return assign(obj, tile)


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
    # Booleans can leave duplicate vertices along a seam; merge them before
    # recalculating normals so the cap stays a closed two-manifold solid.
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
        port_y = D['ports']['front_y']
    else:
        y0, y1 = cap['rear_y0'], cap['rear_y1']
        port_y = D['ports']['rear_y']
    cy = (y0 + y1) / 2
    outer_y = y0 if front else y1
    inner_y = y1 if front else y0
    direction = 1 if front else -1
    cap_obj = box(f'cap_{side}', (0, cy, 0),
                  (cap['size_x'], cap['thickness'], cap['size_z']),
                  TILE['cast'], bevel=cap['bevel'], constant=False)
    cap_obj.name = f'cap_{side}'
    cap_obj.data.name = f'cap_{side}'
    cutters = []
    ox, oz = cap['corner_offset_x'], cap['corner_offset_z']
    # One stepped cutter per corner: a real through hole plus the outer
    # counterbore, built as a single watertight solid so the cutters do not
    # overlap and the boolean stays manifold.
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * ox, sz * oz
            hole = stepped_hole(f'cut_corner_{side}_{sx}_{sz}', outer_y,
                                inner_y, cap['counterbore_radius'],
                                cap['corner_hole_radius'],
                                cap['counterbore_depth'], TILE['rubber'],
                                direction=direction)
            hole.location = (x, 0, z)
            cutters.append(hole)
    # real rod through-hole in the front cap
    if front:
        cutters.append(along(f'cut_rod_hole_{side}', (0, y0 - .01, 0),
                             (0, y1 + .01, 0), cap['rod_hole_radius'],
                             TILE['rubber'], segs=64, bevel=0, cutter=True))
    # real blind port bore in the cap top face
    cutters.append(cylinder_z(f'cut_port_{side}', D['ports']['bore_radius'],
                              D['ports']['top_z'] - D['ports']['bore_depth'],
                              D['ports']['top_z'] + .002,
                              TILE['rubber'], x=0, y=port_y, segs=64,
                              bevel=0, cutter=True))
    # small side grub-screw recesses on both X faces of both caps
    g = D['side_grub']
    for sx in (-1, 1):
        cutters.append(along(f'cut_grub_{side}_{sx}',
                             (sx * (g['x_face'] + .002), cy, g['z']),
                             (sx * (g['x_face'] - g['depth']), cy, g['z']),
                             g['radius'], TILE['rubber'], segs=32, bevel=0,
                             cutter=True))
    apply_boolean(cap_obj, cutters, f'Cut {side} cap ports, rod and tie holes')
    for poly in cap_obj.data.polygons:
        poly.use_smooth = False

    # ribs, boss, gland, port marking and corner nuts
    prefix = 'front' if front else 'rear'
    rib = D['front_ribs'] if front else D['rear_ribs']
    for i, a in enumerate((math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4,
                           7 * math.pi / 4), 1):
        radius = (rib['inner_radius'] + rib['outer_radius']) / 2
        center = (radius * math.cos(a), rib['center_y'], radius * math.sin(a))
        rotated_box(f'{prefix}_rib_{i}', center,
                    (rib['outer_radius'] - rib['inner_radius'] + .006,
                     rib['thickness'], rib['half_width'] * 2),
                    -a, TILE['cast'], bevel=.0008)
    if front:
        gland = D['gland']
        gland_obj = along('gland', (0, gland['y0'], 0), (0, gland['y1'], 0),
                          gland['radius'], TILE['cast'], segs=64,
                          bevel=gland['bevel'])
        # real rod clearance through the gland
        cut = along('cut_gland_rod', (0, gland['y0'] - .01, 0),
                    (0, gland['y1'] + .01, 0), cap['rod_hole_radius'],
                    TILE['rubber'], segs=64, bevel=0, cutter=True)
        apply_boolean(gland_obj, [cut], 'Drill gland rod clearance')
        torus('gland_wiper', gland['seal_major'], gland['seal_minor'],
              (0, gland['seal_y'], 0), 'Y', TILE['rubber'],
              major_segments=64, minor_segments=10)
        torus('gland_base_seam', gland['radius'] - .001, .0009,
              (0, gland['y0'] - .0002, 0), 'Y', TILE['rubber'],
              major_segments=64, minor_segments=8)
    else:
        boss = D['rear_boss']
        along('rear_boss', (0, boss['y0'], 0), (0, boss['y1'], 0),
              boss['radius'], TILE['cast'], segs=64, bevel=.0015)
        torus('rear_boss_groove', boss['groove_major'], boss['groove_minor'],
              (0, boss['groove_y'], 0), 'Y', TILE['dark_steel'],
              major_segments=64, minor_segments=8)
    # corner brass nuts and dark seating rings
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * ox, sz * oz
            nut_y = (D['tie_rods']['nut_y_front'] if front
                     else D['tie_rods']['nut_y_rear'])
            hex_prism(f'tie_nut_{prefix}_{sx}_{sz}', D['tie_rods']['nut_radius'],
                      D['tie_rods']['nut_thickness'], nut_y, TILE['brass'],
                      x=x, z=z, bevel=.0007)
            seat_y = outer_y + (.0042 if front else -.0042)
            torus(f'tie_seat_{prefix}_{sx}_{sz}', .0097, .0007,
                  (x, seat_y, z), 'Y', TILE['dark_steel'],
                  major_segments=40, minor_segments=8)
    # side grub screws and slots
    for sx in (-1, 1):
        along(f'grub_{prefix}_{sx}',
              (sx * (g['x_face'] - g['depth'] + .0002), cy, g['z']),
              (sx * (g['x_face'] - .0001), cy, g['z']),
              g['screw_radius'], TILE['dark_steel'], segs=32, bevel=.0003)
        box(f'grub_slot_{prefix}_{sx}',
            (sx * (g['x_face'] - .0001), cy, g['z']),
            (.0005, .0058, .0011), TILE['rubber'], bevel=.00015)
    # printed green port ring and short pointer tab
    p = D['ports']
    annulus(f'port_ring_{side}', (0, port_y), p['ring_z'],
            p['ring_z'] + .0004, p['ring_inner'], p['ring_outer'],
            TILE['port_ring'], segs=128,
            uv_rotate=0.0 if front else math.pi)
    tab_y = p['tab_center_y_front'] if front else p['tab_center_y_rear']
    box(f'port_tab_{side}', (0, tab_y, p['ring_z'] + .0002),
        tuple(p['tab_size']), TILE['green'], bevel=0, constant=True)
    # real tapped thread inside the port and a dark blind bottom
    helix(f'port_thread_{side}', 'Z', (0, port_y),
          p['thread_major'], p['thread_tube'], p['thread_pitch'],
          p['thread_top_z'], p['thread_bottom_z'], TILE['thread'],
          seg_per_turn=18, tube_segments=8, direction=-1)
    cylinder_z(f'port_bottom_{side}', p['bore_radius'] - .0002,
               p['bottom_disc_z'] - .0012, p['bottom_disc_z'],
               TILE['dark_steel'], x=0, y=port_y, segs=48, bevel=0)
    return cap_obj


def build_barrel():
    b = D['barrel']
    along('barrel', (0, b['y0'], 0), (0, b['y1'], 0), b['radius'],
          TILE['aluminum'], segs=128, bevel=.0015)
    for sign, name in ((-1, 'front'), (1, 'rear')):
        torus(f'barrel_seam_{name}', b['seam_radius'], b['seam_minor'],
              (0, sign * b['seam_y'], 0), 'Y', TILE['rubber'],
              major_segments=128, minor_segments=10)


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
                torus(f'tie_collar_{index}_{name}', t['collar_major'],
                      t['collar_minor'], (x, sign * t['collar_y'], z), 'Y',
                      TILE['rubber'], major_segments=40, minor_segments=8)


def build_rod():
    r = D['rod']
    along('rod', (0, r['y0'], 0), (0, r['thread_y0'], 0), r['radius'],
          TILE['chrome'], segs=96, bevel=.0012)
    along('rod_thread_core', (0, r['thread_y0'], 0), (0, r['thread_y1'], 0),
          r['thread_core_radius'], TILE['thread'], segs=96, bevel=.0006)
    helix('rod_thread', 'Y', (0, 0), r['thread_major'], r['thread_tube'],
          r['thread_pitch'], r['thread_y0'], r['thread_y1'], TILE['thread'],
          seg_per_turn=20, tube_segments=8, direction=-1)
    hex_prism('rod_nut_1', r['nut_radius'], r['nut_thickness'], r['nut1_y'],
              TILE['steel'], bevel=.0010)
    hex_prism('rod_nut_2', r['nut_radius'], r['nut_thickness'], r['nut2_y'],
              TILE['steel'], bevel=.0010)
    torus('rod_washer', r['washer_major'], r['washer_minor'],
          (0, r['washer_y'], 0), 'Y', TILE['steel'],
          major_segments=64, minor_segments=8)
    along('rod_end_dot', (0, r['end_y'] - .0002, 0), (0, r['end_y'] + .0003, 0),
          r['end_dot_radius'], TILE['dark_steel'], segs=32, bevel=.0002)


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
    """Measure the real port depths and tie-rod/rod through-holes."""
    cap = D['cap']
    p = D['ports']
    checks = {'ports': {}, 'tie_holes': {}, 'counterbores': {}, 'rod_hole': {}}
    for side, y_port in (('front', p['front_y']), ('rear', p['rear_y'])):
        tree = make_bvh(bpy.data.objects[f'cap_{side}'])
        z_top = p['top_z'] + .06
        hit = tree.ray_cast(Vector((0, y_port, z_top)), Vector((0, 0, -1)), 1.0)
        offset = tree.ray_cast(Vector((.020, y_port, z_top)),
                               Vector((0, 0, -1)), 1.0)
        depth = None if hit[0] is None else round(p['top_z'] - hit[0].z, 5)
        face_depth = None if offset[0] is None else round(p['top_z'] - offset[0].z, 5)
        checks['ports'][side] = {
            'bore_hit': bool(hit[0] is not None),
            'bore_depth_m': depth,
            'outside_hit': bool(offset[0] is not None),
            'outside_depth_m': face_depth,
            'real_recess': bool(hit[0] is not None and depth is not None
                                and depth >= .010),
        }
    for side in ('front', 'rear'):
        tree = make_bvh(bpy.data.objects[f'cap_{side}'])
        y_out = (-.19 if side == 'front' else .19)
        direction = Vector((0, 1, 0)) if side == 'front' else Vector((0, -1, 0))
        holes, cbs = [], []
        for sx in (-1, 1):
            for sz in (-1, 1):
                x, z = sx * cap['corner_offset_x'], sz * cap['corner_offset_z']
                hit = tree.ray_cast(Vector((x, y_out, z)), direction, 1.0)
                holes.append({'corner': [sx, sz], 'hit': hit[0] is not None})
                face_y = -.145 if side == 'front' else .145
                # probe between the small through hole and the large
                # counterbore wall, where the shoulder is real material
                probe_x = x + .008
                cb_hit = tree.ray_cast(Vector((probe_x, face_y - .0001
                                               if side == 'front'
                                               else face_y + .0001, z)),
                                       direction, 1.0)
                cb_depth = (None if cb_hit[0] is None else
                            round(abs(abs(cb_hit[0].y) - .145), 5))
                cbs.append({'corner': [sx, sz], 'probe_x_offset_m': .008,
                            'hit': cb_hit[0] is not None,
                            'depth_m': cb_depth})
        checks['tie_holes'][side] = {
            'all_through': all(not h['hit'] for h in holes),
            'rays': holes,
        }
        checks['counterbores'][side] = {
            'all_present': all(c['hit'] for c in cbs),
            'rays': cbs,
        }
    tree = make_bvh(bpy.data.objects['cap_front'])
    hit = tree.ray_cast(Vector((0, -.19, 0)), Vector((0, 1, 0)), 1.0)
    checks['rod_hole'] = {'through': hit[0] is None, 'hit': hit[0] is not None}
    depths = [v['bore_depth_m'] for v in checks['ports'].values()
              if v['bore_depth_m'] is not None]
    checks['port_bore_depth_min_m'] = min(depths) if depths else None
    checks['passed'] = bool(
        all(v['real_recess'] for v in checks['ports'].values())
        and all(v['all_through'] for v in checks['tie_holes'].values())
        and all(v['all_present'] for v in checks['counterbores'].values())
        and checks['rod_hole']['through'])
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
              'faces': len(obj.data.polygons), 'tile': tiles.get(obj.name)} for obj in objects]
    intersections = {obj.name: intersecting_faces(obj) for obj in objects}
    render_dir = out / 'renders'
    render_dir.mkdir(exist_ok=True)
    directions = {'front': (0, -1, .18), 'back': (0, 1, .18),
                  'left': (-1, 0, .18), 'right': (1, 0, .18),
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

    if not request.get('draft'):
        # detail targets use the actual shifted part positions
        target = {
            'port-front': object_center('port_ring_front'),
            'port-rear': object_center('port_ring_rear'),
            'front-face': object_center('cap_front'),
            'rod-gland': object_center('rod_nut_1'),
            'tie-rod-end': object_center('tie_nut_front_1_1'),
            'rear-boss': object_center('rear_boss'),
            'side-grub': object_center('grub_front_1'),
            'barrel-seam': object_center('barrel_seam_front'),
        }
        detail_specs = [
            ('port-front', (0.55, -0.80, 0.95), target['port-front'], .062),
            ('port-rear', (0.55, 0.80, 0.95), target['port-rear'], .062),
            ('front-face', (0.45, -1.0, 0.30), target['front-face'], .155),
            ('rod-gland', (0.70, -1.0, 0.36), target['rod-gland'], .115),
            ('tie-rod-end', (0.60, -1.0, 0.36), target['tie-rod-end'], .058),
            ('rear-boss', (0.50, 1.0, 0.34), target['rear-boss'], .115),
            ('side-grub', (1.0, -0.32, 0.28), target['side-grub'], .058),
            ('barrel-seam', (0.85, -0.60, 0.50), target['barrel-seam'], .105),
            ('overall-3q', (1.0, -1.0, 0.55), center, span * 1.14),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        # isolation renders for real openings / cap faces / subassemblies
        port_front = {'port_ring_front', 'port_tab_front', 'port_thread_front',
                      'port_bottom_front'}
        port_rear = {'port_ring_rear', 'port_tab_rear', 'port_thread_rear',
                     'port_bottom_rear'}
        set_visible({'cap_front', 'cap_rear'} | port_front | port_rear)
        look(scene, (0, 0, 1), center, span, ortho_scale=.27)
        draw(scene, render_dir / 'caps-only-top.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'cap_front', 'cap_rear'})
        look(scene, (0, 0, -1), center, span, ortho_scale=.27)
        draw(scene, render_dir / 'caps-only-bottom.png', rr['detail_resolution'],
             rr['detail_samples'])
        front_iso = ({'cap_front', 'front_rib_1', 'front_rib_2', 'front_rib_3',
                      'front_rib_4', 'gland', 'gland_wiper', 'gland_base_seam',
                      'grub_front_1', 'grub_front_-1', 'grub_slot_front_1',
                      'grub_slot_front_-1'} | port_front |
                     {f'tie_nut_front_{sx}_{sz}' for sx in (-1, 1)
                      for sz in (-1, 1)} |
                     {f'tie_seat_front_{sx}_{sz}' for sx in (-1, 1)
                      for sz in (-1, 1)})
        set_visible(front_iso)
        look(scene, (0.55, -1.0, 0.25), object_center('cap_front'), span,
             ortho_scale=.19)
        draw(scene, render_dir / 'front-cap-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        rear_iso = ({'cap_rear', 'rear_rib_1', 'rear_rib_2', 'rear_rib_3',
                     'rear_rib_4', 'rear_boss', 'rear_boss_groove',
                     'grub_rear_1', 'grub_rear_-1', 'grub_slot_rear_1',
                     'grub_slot_rear_-1'} | port_rear |
                    {f'tie_nut_rear_{sx}_{sz}' for sx in (-1, 1)
                     for sz in (-1, 1)} |
                    {f'tie_seat_rear_{sx}_{sz}' for sx in (-1, 1)
                     for sz in (-1, 1)})
        set_visible(rear_iso)
        look(scene, (0.55, 1.0, 0.30), object_center('cap_rear'), span,
             ortho_scale=.19)
        draw(scene, render_dir / 'rear-cap-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'rod', 'rod_thread', 'rod_nut_1', 'rod_nut_2',
                     'rod_washer', 'rod_end_dot', 'gland', 'gland_wiper',
                     'gland_base_seam'})
        look(scene, (0.8, -1.0, 0.35), object_center('rod_nut_1'), span,
             ortho_scale=.22)
        draw(scene, render_dir / 'rod-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'barrel', 'tie_rod_1', 'tie_rod_2', 'tie_rod_3', 'tie_rod_4',
                     'barrel_seam_front', 'barrel_seam_rear'}
                    | {f'tie_collar_{i}_{n}' for i in range(1, 5)
                       for n in ('front', 'rear')})
        look(scene, (0.25, -0.55, 0.80), center, span, ortho_scale=.26)
        draw(scene, render_dir / 'barrel-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({obj.name for obj in objects})
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-material-front.png', rr['six_view_resolution'],
             rr['six_view_samples'])

        # clay
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

        # alternate light for the same front camera
        main_light = bpy.data.objects['main softbox']
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-light-alternate.png', rr['six_view_resolution'],
             rr['six_view_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()

        # metal-only brightness comparison with a moving main light
        set_visible(set(metal_objects))
        fill_light = bpy.data.objects.get('soft light')
        front_light = bpy.data.objects.get('front reflection')
        top_light = bpy.data.objects.get('top fill')
        for light in (fill_light, front_light, top_light):
            if light is not None:
                light.hide_render = True
        metal_target = object_center('rod_nut_1')
        look(scene, (0.9, -1.0, 0.55), metal_target, span, ortho_scale=.30)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (0.9, -1.0, 0.55), metal_target, span, ortho_scale=.30)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for light in (fill_light, front_light, top_light):
            if light is not None:
                light.hide_render = False
        set_visible({obj.name for obj in objects})

        # turntable
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
        'geometry_basis': '原照可见结构 + 未标定六视图（拉杆式气缸）',
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'shift': list(shift),
    }
    (out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                            encoding='utf8')


main()
