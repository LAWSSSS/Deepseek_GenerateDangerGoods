"""Blender worker: part-based rebuild of item 24, a square-profile air cylinder.

Source photograph: SHAKO-badged square-profile double-acting cylinder, read as
SUS63-100 (63 mm bore, 100 mm stroke), with black end caps, a ribbed extruded
aluminium barrel, end-face threaded ports, a chrome piston rod, two hex nuts
and a white/red label.  The six-view sheet is an uncalibrated AI-generated
visual reference that calls the same object SMC SU63-100 and duplicates the
end-face port on the side faces; the candidate follows the photograph for the
port and fastener faces and uses the sheet only for hidden-side topology.

Coordinates: Z up, cylinder axis along Y, rod end at -Y, label side +X.
All parts are separate watertight meshes.  Metals keep metallic PBR response;
rubber and the printed label are dielectric.  No light or shadow is baked into
the textures.  Candidate is manual-only and never replaces the library mesh.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
cfg = request['config']
D = cfg['dimensions']
TEX = request['textures']
TILE = cfg['texture']['tiles']
DRAFT = bool(request.get('draft'))

objects = []
tiles = {}
metal_objects = []
nonmetal_objects = []
MATERIAL = None
LABEL_MAT = None
CAM = None
DYNAMIC_TILES = {'aluminum', 'cap', 'chrome', 'steel', 'thread', 'dark'}
DIELECTRIC_TILES = {'rubber'}


# ---------------------------------------------------------------------------
# Materials and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed aluminium, black cap, steel and rubber PBR atlas')
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
    normal_map.inputs['Strength'].default_value = .18
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    return material


def label_material():
    material = bpy.data.materials.new('Redrawn SHAKO SUS63-100 label, printed vinyl')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(TEX['label'], check_existing=True)
    tex.image.pack()
    tex.interpolation = 'Linear'
    links.new(tex.outputs['Color'], principled.inputs['Base Color'])
    principled.inputs['Roughness'].default_value = .52
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['IOR'].default_value = 1.45
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
# Primitive helpers
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
    """Closed helical tube used as a visual thread ridge (Y-axis only here)."""
    if axis != 'Y':
        raise ValueError('item 24 uses only Y-axis threads')
    length = abs(h1 - h0)
    turns = length / pitch
    rings = max(4, int(math.ceil(turns * seg_per_turn)))
    verts, faces = [], []
    for i in range(rings + 1):
        theta = direction * math.tau * turns * i / rings
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
    s0 = len(verts)
    verts.append((center[0] + radius, h0, center[1]))
    s1 = len(verts)
    theta_end = direction * math.tau * turns
    verts.append((center[0] + radius * math.cos(theta_end),
                  h1, center[1] + radius * math.sin(theta_end)))
    start = (rings * tube_segments)
    for k in range(tube_segments):
        k2 = (k + 1) % tube_segments
        faces.append((s0, k2, k))
        faces.append((s1, start + k, start + k2))
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
                 segs=56, direction=1, cutter=True, center=(0.0, 0.0)):
    y_out = outer_y - direction * .010
    y_shoulder = outer_y + direction * depth
    y_in = inner_y + direction * .010
    profile = [(y_out, r_large), (y_shoulder, r_large),
               (y_shoulder, r_small), (y_in, r_small)]
    obj = revolve_y(name, profile, tile, segs=segs, cutter=cutter)
    obj.location.x = center[0]
    obj.location.z = center[1]
    return obj


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
def cross_recess(prefix, center, y_or_x, axis, tile):
    """Two shallow dark bars reading as a cross drive on a fastener head."""
    x, y, z = center
    if axis == 'Y':
        box(f'{prefix}_cross1', (x, y, z), (0.0011, 0.0005, 0.0080),
            tile, bevel=0.0001, constant=True)
        box(f'{prefix}_cross2', (x, y, z), (0.0080, 0.0005, 0.0011),
            tile, bevel=0.0001, constant=True)
    else:  # X axis
        box(f'{prefix}_cross1', (x, y, z), (0.0005, 0.0011, 0.0080),
            tile, bevel=0.0001, constant=True)
        box(f'{prefix}_cross2', (x, y, z), (0.0005, 0.0080, 0.0011),
            tile, bevel=0.0001, constant=True)


def fastener(name, face_y, side, x, z, radius, washer_radius, height, tile,
             *, washer=True):
    """Cross-recessed screw head on a cap end face; outward direction = side."""
    if washer:
        along(f'{name}_washer', (x, face_y, z), (x, face_y + side * .0012, z),
              washer_radius, tile, segs=40, bevel=.0002)
    base = face_y + side * (.0012 if washer else 0.0)
    along(f'{name}_head', (x, base, z), (x, base + side * height, z),
          radius, tile, segs=40, bevel=.0003)
    cross_recess(name, (x, base + side * (height + .00025), z), None, 'Y',
                 TILE['dark'])


def side_screw(name, x_face, y, z, radius, height, tile):
    along(f'{name}_washer', (x_face, y, z), (x_face + .0010, y, z),
          radius * 1.25, tile, segs=32, bevel=.0002)
    along(f'{name}_head', (x_face + .0010, y, z),
          (x_face + .0010 + height, y, z), radius, tile, segs=32, bevel=.0003)
    cross_recess(name, (x_face + .0010 + height + .00025, y, z), None, 'X',
                 TILE['dark'])


def add_text(name, body, size, extrude, location, tile):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.data.body = body
    obj.data.size = size
    obj.data.extrude = extrude
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    boundary = [edge for edge in bm.edges if edge.is_boundary]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    for poly in obj.data.polygons:
        poly.use_smooth = False
    assign(obj, tile, mode='constant')
    return obj


def build_cap(side_tag):
    cap = D['cap']
    front = side_tag == 'front'
    side = -1 if front else 1
    y_face = cap['y_front_outer'] if front else cap['y_rear_outer']
    y_inner = cap['y_front_inner'] if front else cap['y_rear_inner']
    cy = (y_face + y_inner) / 2
    half_x, half_z = cap['half_x'], cap['half_z']
    name = f'cap_{side_tag}'
    cap_obj = box(name, (0, cy, 0), (2 * half_x, abs(y_face - y_inner), 2 * half_z),
                  TILE['cap'], bevel=cap['bevel'])
    # Real end-face port: stepped blind hole with internal thread.
    direction = -side  # inward along Y
    inner_y = y_face + direction * (cap['port_depth'] - .010)
    port = stepped_hole(f'cap_{side_tag}_port_cutter', y_face, inner_y,
                        cap['port_counterbore_radius'], cap['port_bore_radius'],
                        cap['port_counterbore_depth'], TILE['dark'],
                        direction=direction,
                        center=(cap['port_center_x'], cap['port_center_z']))
    cutters = [port]
    if front:
        # Real rod through-hole in the front end cap.
        hole = along(f'cap_{side_tag}_rod_cutter',
                     (0, y_face + side * .012, 0),
                     (0, y_inner - side * .012, 0),
                     D['rod']['radius'] + .0006, TILE['dark'], cutter=True)
        cutters.append(hole)
    apply_boolean(cap_obj, cutters, f'cap_{side_tag}_holes')
    apply_bevel(cap_obj, .00025, segments=1, angle=35)
    # Re-assert material/UV after boolean (material slots were transferred).
    cap_obj.data.materials.clear()
    cap_obj.data.materials.append(MATERIAL)
    remap_uv(cap_obj, TILE['cap'])
    tiles[cap_obj.name] = TILE['cap']

    # Port thread inside the blind hole.
    y_start = y_face + direction * cap['port_counterbore_depth']
    y_end = y_face + direction * (cap['port_depth'] - .0010)
    helix(f'cap_{side_tag}_port_thread', 'Y',
          (cap['port_center_x'], cap['port_center_z']),
          cap['port_thread_radius'], cap['port_thread_tube'],
          cap['port_thread_pitch'], y_start, y_end, TILE['thread'],
          direction=direction, seg_per_turn=12, tube_segments=8)
    along(f'cap_{side_tag}_port_bottom',
          (cap['port_center_x'], y_face + direction * cap['port_depth'],
           cap['port_center_z']),
          (cap['port_center_x'],
           y_face + direction * (cap['port_depth'] - .0006),
           cap['port_center_z']),
          cap['port_bore_radius'] - .0001, TILE['dark'], segs=40, bevel=0,
          constant=True)

    # Four corner cross-recessed screws with silver washers.
    for sx in (-1, 1):
        for sz in (-1, 1):
            fastener(f'cap_{side_tag}_corner_{sx}_{sz}', y_face, side,
                     sx * cap['corner_screw_x'], sz * cap['corner_screw_x'],
                     cap['corner_screw_head_radius'],
                     cap['corner_screw_washer_radius'],
                     cap['corner_screw_head_height'], TILE['steel'])

    # Small cross-recessed screw below/right of the port.
    fastener(f'cap_{side_tag}_small', y_face, side,
             cap['small_screw_x'], cap['small_screw_z'],
             cap['small_screw_radius'], cap['small_screw_radius'] * 1.15,
             cap['small_screw_height'], TILE['steel'], washer=False)

    # Four diagonal ribs on the end face (X-shaped, as in photo/reference).
    r0, r1 = cap['rib_start_radius'], cap['rib_end_radius']
    for k, phi in enumerate((45, 135, 225, 315), start=1):
        a = math.radians(phi)
        r = (r0 + r1) / 2
        rotated_box(f'cap_{side_tag}_rib_{k}',
                    (r * math.cos(a), y_face + side * (cap['rib_height'] / 2),
                     r * math.sin(a)),
                    (r1 - r0, cap['rib_height'], cap['rib_width']), -a,
                    TILE['cap'], bevel=.0005)

    # Rear central boss and ring groove; front gland collar and wiper.
    if front:
        rod = D['rod']
        along('gland_front', (0, y_face, 0),
              (0, rod['gland_y1'], 0), rod['gland_radius'], TILE['cap'],
              segs=64, bevel=.0012)
        torus('gland_wiper_front', rod['wiper_major'], rod['wiper_minor'],
              (0, rod['wiper_y'], 0), 'Y', TILE['rubber'])
    else:
        along('rear_boss', (0, y_face, 0),
              (0, y_face + side * cap['rear_boss_height'], 0),
              cap['rear_boss_radius'], TILE['cap'], segs=64, bevel=.0008)
        torus('rear_boss_groove', cap['rear_boss_groove_radius'],
              cap['rear_boss_groove_minor'],
              (0, y_face + side * (cap['rear_boss_height'] + .0002), 0),
              'Y', TILE['dark'])

    # EA / EB embossed letters on the top face near the outer end.
    ty = side * (cap['y_rear_outer'] - cap['text_y_inset'])
    for body, tx in (('EB', cap['text_x_eb']), ('EA', cap['text_x_ea'])):
        add_text(f'cap_{side_tag}_text_{body}', body, cap['text_size'],
                 cap['text_extrude'],
                 (tx, ty, cap['half_z'] + cap['text_extrude'] * .5 + .0001),
                 TILE['cap'])

    # Side-face set screw (visible on the +X face in the photograph).
    side_screw(f'cap_{side_tag}_side', half_x, side * cap['side_screw_y_offset'],
               cap['side_screw_z'], cap['side_screw_radius'],
               cap['side_screw_height'], TILE['steel'])
    return cap_obj


def build_barrel():
    b = D['barrel']
    half = b['half']
    length = b['y1'] - b['y0']
    cy = (b['y0'] + b['y1']) / 2
    barrel = box('barrel', (0, cy, 0), (2 * half, length, 2 * half),
                 TILE['aluminum'], bevel=b['corner_radius'])
    cutters = []
    c = b['groove_radius']
    depth = b['groove_depth']
    centre = half + c - depth
    for i, off in enumerate(b['groove_centers']):
        for axis in ('X', 'Z'):
            for sign in (-1, 1):
                if axis == 'Z':
                    x, z = off, sign * centre
                    start = (x, b['y0'] - .004, z)
                    end = (x, b['y1'] + .004, z)
                else:
                    x, z = sign * centre, off
                    start = (x, b['y0'] - .004, z)
                    end = (x, b['y1'] + .004, z)
                cutters.append(along(f'barrel_groove_cutter_{axis}_{sign}_{i}',
                                     start, end, c, TILE['dark'],
                                     segs=24, bevel=0, cutter=True))
    apply_boolean(barrel, cutters, 'barrel_grooves')
    apply_bevel(barrel, .0003, segments=1, angle=35)
    barrel.data.materials.clear()
    barrel.data.materials.append(MATERIAL)
    # Longitudinal brushed UVs: U along Y, V around the profile.
    mesh = barrel.data
    if mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers.active)
    uv = mesh.uv_layers.new(name='Atlas')
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            v = mesh.vertices[mesh.loops[li].vertex_index].co
            u = (v.y - b['y0']) / (b['y1'] - b['y0'])
            w = math.atan2(v.z, v.x) / math.tau + .5
            uv.data[li].uv = (u, w)
    remap_uv(barrel, TILE['aluminum'])
    tiles[barrel.name] = TILE['aluminum']
    # Dark gasket lines at both cap joints.
    for tag, y in (('front', -b['seam_y']), ('rear', b['seam_y'])):
        box(f'barrel_seam_{tag}', (0, y, 0),
            (b['seam_size'], b['seam_thickness'], b['seam_size']),
            TILE['rubber'], bevel=.0008)
    return barrel


def build_rod():
    r = D['rod']
    along('rod_shank', (0, r['y_inside'], 0), (0, r['y_shank_end'], 0),
          r['radius'], TILE['chrome'], segs=64, bevel=.0008)
    along('rod_thread_core', (0, r['thread_y0'], 0),
          (0, r['thread_y1'] - .0008, 0), r['core_radius'], TILE['thread'],
          segs=48, bevel=.0006)
    helix('rod_thread', 'Y', (0, 0), r['thread_radius'], r['thread_tube'],
          r['thread_pitch'], r['thread_y0'], r['thread_y1'], TILE['thread'],
          direction=1, seg_per_turn=16, tube_segments=8)
    along('rod_washer', (0, r['washer_y'] - r['washer_thickness'] / 2, 0),
          (0, r['washer_y'] + r['washer_thickness'] / 2, 0),
          r['washer_radius'], TILE['steel'], segs=48, bevel=.0004)
    for tag, y in (('1', r['nut1_y']), ('2', r['nut2_y'])):
        nut = hex_prism(f'rod_nut_{tag}', r['nut_radius'], r['nut_thickness'], y,
                        TILE['steel'])
        cutter = along(f'rod_nut_{tag}_bore_cutter',
                       (0, y - r['nut_thickness'], 0),
                       (0, y + r['nut_thickness'], 0),
                       r['radius'] + .0003, TILE['dark'], cutter=True)
        apply_boolean(nut, [cutter], f'rod_nut_{tag}_bore')
        assign(nut, TILE['steel'], mode='constant', register=False)
        tiles[nut.name] = TILE['steel']


def build_label():
    lab = D['label']
    b = D['barrel']
    x0 = b['half'] + lab['x_offset']
    x1 = x0 + .0008
    y0, y1 = lab['y_center'] - lab['u_len'] / 2, lab['y_center'] + lab['u_len'] / 2
    z0, z1 = lab['z_center'] - lab['v_len'] / 2, lab['z_center'] + lab['v_len'] / 2
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
             (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 7, 4), (1, 5, 6, 2), (0, 1, 2, 3),
             (4, 7, 6, 5), (0, 4, 5, 1), (3, 2, 6, 7)]
    center_uv = (0.5, 0.5)
    uvs = [[center_uv] * 4, [(0, 0), (1, 0), (1, 1), (0, 1)],
           [center_uv] * 4, [center_uv] * 4, [center_uv] * 4, [center_uv] * 4]
    obj = mesh_object('label', verts, faces, uvs)
    apply_bevel(obj, .00015, segments=1, angle=40)
    obj.data.materials.clear()
    obj.data.materials.append(LABEL_MAT)
    objects.append(obj)
    nonmetal_objects.append(obj.name)
    return obj


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
               ortho_scale=ortho_scale or span * 1.15,
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
    cap = D['cap']
    checks = {'ports': {}, 'rod_hole_through': None, 'barrel_grooves': {}}
    for tag, side, y_face in (('front', -1, cap['y_front_outer']),
                              ('rear', 1, cap['y_rear_outer'])):
        bvh = make_bvh(bpy.data.objects[f'cap_{tag}'])
        origin = Vector((cap['port_center_x'], y_face + side * .03,
                         cap['port_center_z']))
        direction = Vector((0, -side, 0))
        hit = bvh.ray_cast(origin, direction, 0.08)
        if hit[0] is None:
            checks['ports'][tag] = {'blind': False, 'depth_m': None,
                                    'first_hit': None}
        else:
            checks['ports'][tag] = {'blind': True,
                                    'depth_m': round(hit[3] - .03, 6),
                                    'first_hit': [round(v, 6) for v in hit[0]]}
    front = make_bvh(bpy.data.objects['cap_front'])
    hit = front.ray_cast(Vector((0, -0.20, 0)), Vector((0, 1, 0)), 0.3)
    checks['rod_hole_through'] = hit[0] is None
    barrel = make_bvh(bpy.data.objects['barrel'])
    for label, z in (('groove_crest', 0.0), ('rib_crest', 0.00875)):
        hit = barrel.ray_cast(Vector((0.2, 0.0, z)), Vector((-1, 0, 0)), 0.3)
        checks['barrel_grooves'][label] = (
            None if hit[0] is None else round(hit[0].x, 6))
    return checks


def main():
    global MATERIAL, LABEL_MAT, CAM
    scene = legacy.setup_scene('CPU', 512)
    MATERIAL = atlas_material()
    LABEL_MAT = label_material()
    CAM = init_camera(scene)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .42
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    scene.cycles.use_denoising = bool(cfg['render'].get('denoise', True))
    try:
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
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

    build_barrel()
    build_cap('front')
    build_cap('rear')
    build_rod()
    build_label()
    bpy.context.view_layer.update()
    checks = ray_checks()
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
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

    def closure(obj):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        nonmanifold = sum(1 for edge in bm.edges if not edge.is_manifold)
        boundary = sum(1 for edge in bm.edges if edge.is_boundary)
        bm.free()
        return nonmanifold, boundary

    parts = []
    for obj in objects:
        nonmanifold, boundary = closure(obj)
        parts.append({'name': obj.name, 'vertices': len(obj.data.vertices),
                      'faces': len(obj.data.polygons), 'tile': tiles.get(obj.name),
                      'metal': obj.name in metal_objects,
                      'nonmanifold_edges': nonmanifold,
                      'boundary_edges': boundary,
                      'closed_blender': bool(nonmanifold == 0 and boundary == 0)})
    intersections = {obj.name: intersecting_faces(obj) for obj in objects}
    render_dir = out / 'renders'
    render_dir.mkdir(exist_ok=True)
    directions = {'front': (1, 0, .18), 'back': (-1, 0, .18),
                  'left': (0, -1, .18), 'right': (0, 1, .18),
                  'top': (0, 0, 1), 'bottom': (0, 0, -1)}
    rr = cfg['render']
    for name, direction in directions.items():
        look(scene, direction, center, span)
        draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    if not DRAFT:
        # Detail targets.
        target = {
            'front-face': object_center('cap_front'),
            'rear-face': object_center('cap_rear'),
            'rod-gland': object_center('rod_nut_1'),
            'label': object_center('label'),
            'barrel-profile': object_center('barrel'),
            'ea-eb': object_center('cap_rear'),
            'side-screw': object_center('cap_front_side_head'),
        }
        specs = [
            ('front-face', (0.55, -1.0, 0.35), target['front-face'], .175),
            ('rear-face', (0.55, 1.0, 0.35), target['rear-face'], .175),
            ('port-front', (0.30, -1.0, 0.75),
             Vector((0, -0.091, D['cap']['port_center_z'])), .065),
            ('port-rear', (0.30, 1.0, 0.75),
             Vector((0, 0.091, D['cap']['port_center_z'])), .065),
            ('rod-gland', (0.80, -1.0, 0.35), target['rod-gland'], .130),
            ('rod-thread', (0.85, -1.0, 0.30), Vector((0, -0.155, 0)), .090),
            ('label', (1.0, -0.20, 0.18), target['label'], .095),
            ('barrel-profile', (1.0, -0.45, 0.55), target['barrel-profile'], .150),
            ('ea-eb', (0.35, 0.25, 1.2), target['ea-eb'], .090),
            ('side-screw', (1.0, -0.35, 0.25), target['side-screw'], .060),
            ('overall-3q', (1.0, -1.0, 0.55), center, span * 1.08),
        ]
        for name, direction, tgt, scale in specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png',
                 rr['detail_resolution'], rr['detail_samples'])

        caps = {'cap_front', 'cap_rear', 'gland_front', 'gland_wiper_front',
                'rear_boss', 'rear_boss_groove'}
        caps |= {name for name in bpy.data.objects.keys()
                 if name.startswith('cap_front_') or name.startswith('cap_rear_')}
        set_visible(caps)
        cap_objects = [bpy.data.objects[n] for n in caps
                       if n in bpy.data.objects and bpy.data.objects[n].type == 'MESH']
        _, _, cap_center, cap_span = bounds(cap_objects)
        look(scene, (0, 0, 1), cap_center, cap_span,
             ortho_scale=cap_span * 1.15)
        draw(scene, render_dir / 'caps-only-top.png', rr['detail_resolution'],
             rr['detail_samples'])
        look(scene, (0, 0, -1), cap_center, cap_span,
             ortho_scale=cap_span * 1.15)
        draw(scene, render_dir / 'caps-only-bottom.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({name for name in caps
                     if not name.startswith('cap_rear') and name != 'rear_boss'
                     and name != 'rear_boss_groove'})
        look(scene, (0.55, -1.0, 0.30), object_center('cap_front'), span,
             ortho_scale=.185)
        draw(scene, render_dir / 'front-cap-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({name for name in caps
                     if not name.startswith('cap_front') and name != 'gland_front'
                     and name != 'gland_wiper_front'})
        look(scene, (0.55, 1.0, 0.30), object_center('cap_rear'), span,
             ortho_scale=.185)
        draw(scene, render_dir / 'rear-cap-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'rod_shank', 'rod_thread_core', 'rod_thread', 'rod_washer',
                     'rod_nut_1', 'rod_nut_2', 'gland_front', 'gland_wiper_front'})
        look(scene, (0.8, -1.0, 0.35), object_center('rod_nut_1'), span,
             ortho_scale=.220)
        draw(scene, render_dir / 'rod-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'barrel', 'barrel_seam_front', 'barrel_seam_rear', 'label'})
        look(scene, (0.35, -0.55, 0.80), center, span, ortho_scale=.255)
        draw(scene, render_dir / 'barrel-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'barrel', 'barrel_seam_front', 'barrel_seam_rear', 'label'})
        look(scene, (1.0, -0.25, 0.20), center, span, ortho_scale=.220)
        draw(scene, render_dir / 'barrel-label.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({obj.name for obj in objects})
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-material-front.png',
             rr['six_view_resolution'], rr['six_view_samples'])

        # Clay render for shape audit.
        clay = bpy.data.materials.new('Clay')
        clay.use_nodes = True
        clay_b = clay.node_tree.nodes.get('Principled BSDF')
        clay_b.inputs['Base Color'].default_value = (.55, .56, .54, 1)
        clay_b.inputs['Roughness'].default_value = .82
        clay_b.inputs['Metallic'].default_value = 0
        saved = {}
        for obj in objects:
            saved[obj.name] = [slot.material for slot in obj.material_slots]
            obj.data.materials.clear()
            obj.data.materials.append(clay)
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-clay-front.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        look(scene, (1.0, -1.0, 0.55), center, span)
        draw(scene, render_dir / 'new-clay-3q.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        for obj in objects:
            obj.data.materials.clear()
            for mat in saved[obj.name]:
                obj.data.materials.append(mat)

        # Alternate light on the same front camera.
        main_light = bpy.data.objects['main softbox']
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-light-alternate.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()

        # Metal-only brightness comparison with a moving main light.
        set_visible(set(metal_objects))
        for name in ('soft light', 'front reflection', 'top fill'):
            light = bpy.data.objects.get(name)
            if light is not None:
                light.hide_render = True
        metal_target = object_center('rod_nut_1')
        look(scene, (0.9, -1.0, 0.55), metal_target, span, ortho_scale=.22)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (0.9, -1.0, 0.55), metal_target, span, ortho_scale=.22)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for name in ('soft light', 'front reflection', 'top fill'):
            light = bpy.data.objects.get(name)
            if light is not None:
                light.hide_render = False
        set_visible({obj.name for obj in objects})

        # 24-direction elevated turntable (RGBA, used for silhouette diagnostics).
        for i in range(24):
            a = math.tau * i / 24
            look(scene, (math.cos(a), math.sin(a), .35), center, span)
            draw(scene, render_dir / f'turn-{i:02d}.png',
                 rr['turntable_resolution'], rr['turntable_samples'])

    # Old library mesh rendered with the same rig, framed to its own bounds.
    for obj in objects:
        obj.hide_render = True
    old = legacy.load_mesh(request['original_mesh'])
    _, _, oldcenter, oldspan = bounds([old])
    old.location -= oldcenter
    bpy.context.view_layer.update()
    _, _, oldcenter, oldspan = bounds([old])
    for name, direction in directions.items():
        look(scene, direction, oldcenter, oldspan)
        draw(scene, render_dir / f'old-{name}.png', rr['old_resolution'],
             rr['six_view_samples'])
    look(scene, (1.0, -1.0, 0.55), oldcenter, oldspan)
    draw(scene, render_dir / 'old-overall-3q.png', rr['old_resolution'],
         rr['six_view_samples'])

    data = {
        'parts': parts,
        'bounds': [list(low), list(high)],
        'centre': list(center),
        'span': span,
        'polygons': sum(len(obj.data.polygons) for obj in objects),
        'triangles': sum(max(0, len(p.vertices) - 2)
                         for obj in objects for p in obj.data.polygons),
        'metal_objects': metal_objects,
        'nonmetal_objects': nonmetal_objects,
        'tiles': tiles,
        'ray_checks': checks,
        'self_intersections': intersections,
        'self_intersection_total': sum(intersections.values()),
        'closed_blender_parts': sum(1 for part in parts if part['closed_blender']),
        'open_blender_parts': [part['name'] for part in parts
                               if not part['closed_blender']],
        'render_samples': rr['six_view_samples'],
        'render_resolution': rr['six_view_resolution'],
        'old_render_resolution': rr['old_resolution'],
        'draft': DRAFT,
        'note': ('端面进气口、端面螺纹、前端杆通孔、四角螺钉、EA/EB 铸字和端面加强筋'
                 '按原照与未标定六视图建立；孔深、螺纹和背面为推断。'),
    }
    (out / 'worker-result.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'ok', 'parts': len(objects),
                      'self_intersections': data['self_intersection_total'],
                      'port_checks': checks['ports']}, ensure_ascii=False))


main()
