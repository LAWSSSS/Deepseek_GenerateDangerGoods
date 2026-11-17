"""Blender worker: part-based rebuild of item 32, a long black hydraulic cylinder.

Evidence:
- one 1372x1147 clean photograph and its matching 1371x1147 rust derivative of
  a long black welded hydraulic cylinder: forged/welded eye at each end, a long
  chrome piston rod, a stepped black/steel gland and seal stack, a yellow label
  band with red Chinese characters near the rod end, and two same-side radial
  ports with steel hex nuts and yellow plastic caps;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified).  The mapping points both item 30 and item 32 at
  the same vertical product sheet.  It confirms two same-side ports, two eyes
  and the yellow label, but its shorter/retracted pose and proportions do not
  match the long extended source photograph.  The photograph is authoritative
  for extension, length and the long chrome rod; the sheet is used only for
  topology.

The old TripoSR library mesh is a soft 4000-vertex blob: both eye holes are
filled and the gland stack, label band, ports and weld beads are fused into an
undifferentiated surface.  It is retained unchanged and is not used as the
candidate's base geometry.  The candidate is a parametric part rebuild with
true through-bored eyes and steel bushings, a long chrome rod, a stepped
gland/seal stack, a label sleeve with an approximate red-letter texture, two
welded ports with hex nuts and knurled yellow caps, a welded rear cap and a
closed long barrel.  All candidates remain manual-only and pending human
review; nothing replaces the library mesh or enters automatic placement.

Coordinates: local cylinder axis along Y, rear eye at -Y, rod end at +Y, +Z up,
ports on +Z.  The built assembly is then rotated about X by the old mesh's PCA
pose so the library-space pose matches the source photograph (rod end toward
-Y/+Z, rear eye toward +Y/-Z).  Export stays Z-up (export_yup=False).
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
cfg = request['config']
D = cfg['dimensions']
TEX = request['textures']
TILE = cfg['texture']['tiles']
TILE_NAME = {value: key for key, value in TILE.items()}
POSE_ROTATION_X = math.radians(float(cfg['orientation'].get('pose_rotation_deg_x', 133.0)))

objects = []
tiles = {}
metal_objects = []
nonmetal_objects = []
MATERIAL = None
CAM = None
METAL_TILES = {'chrome', 'steel', 'dark_steel', 'weld'}
DIELECTRIC_TILES = {'black_paint', 'black_edge', 'label', 'yellow_plastic',
                    'rubber', 'rust', 'spare10', 'spare11', 'spare12',
                    'spare13', 'spare14', 'spare15'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed black paint, chrome, steel, label and plastic PBR atlas')
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


def ensure_auto_uv(obj, scale=3.5):
    if obj.data.uv_layers:
        return
    layer = obj.data.uv_layers.new(name='Atlas')
    mesh = obj.data
    for poly in mesh.polygons:
        normal = poly.normal
        axis = max(range(3), key=lambda i: abs(normal[i]))
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            if axis == 0:
                u, v = co.y * scale, co.z * scale
            elif axis == 1:
                u, v = co.x * scale, co.z * scale
            else:
                u, v = co.x * scale, co.y * scale
            layer.data[li].uv = (u, v)


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
    name = TILE_NAME.get(tile, f'tile{tile}')
    if mode == 'constant':
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name='Atlas')
        x0, y0, x1, y1 = tile_bounds(tile)
        center = (x0 + .5 * (x1 - x0), y0 + .5 * (y1 - y0))
        for loop in obj.data.uv_layers.active.data:
            loop.uv = center
    else:
        ensure_auto_uv(obj)
        remap_uv(obj, tile, v_lo=v_lo, v_hi=v_hi, u_lo=u_lo, u_hi=u_hi, margin=margin)
    if register:
        objects.append(obj)
        if name in METAL_TILES:
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
    return assign(obj, tile, mode='constant' if constant else 'default',
                  v_lo=v_lo, v_hi=v_hi, register=register, cutter=cutter)


def smooth_side(obj, axis):
    axis = Vector(axis).normalized()
    for poly in obj.data.polygons:
        poly.use_smooth = abs(poly.normal.dot(axis)) < .70


# ---------------------------------------------------------------------------
# Mesh construction helpers
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


def along(name, start, end, radius, tile, *, segs=48, bevel=.001, constant=False,
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


def torus(name, major, minor, location, axis, tile, *, major_segments=96,
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


def box(name, center, size, tile, *, bevel=.001, constant=False, cutter=False,
        rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    obj.rotation_euler = rotation
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, constant=constant, cutter=cutter)


def hex_prism_axis(name, radius, thickness, center, axis, tile, *, bevel=.0008,
                   smooth=False):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius, depth=thickness,
                                        location=center)
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat('Z', 'Y').to_euler()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, smooth=smooth)


def annulus_axis(name, axis, center, a0, a1, r_in, r_out, tile, *, segs=128,
                 bevel=0.0):
    """Closed washer/ring extruded along X or Y."""
    cx, cy, cz = center
    rings = []
    verts = []
    order = ((a0, r_in), (a1, r_in), (a1, r_out), (a0, r_out))
    for a, r in order:
        start = len(verts)
        for j in range(segs):
            t = math.tau * j / segs
            if axis == 'Y':
                verts.append((cx + r * math.cos(t), a, cz + r * math.sin(t)))
            elif axis == 'X':
                verts.append((a, cy + r * math.cos(t), cz + r * math.sin(t)))
            else:
                raise ValueError('annulus_axis supports X or Y')
        rings.append(list(range(start, start + segs)))
    faces = []
    for k in range(4):
        r0, r1 = rings[k], rings[(k + 1) % 4]
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((r0[j], r0[j2], r1[j2], r1[j]))
    obj = mesh_object(name, verts, faces)
    if bevel:
        apply_bevel(obj, bevel, segments=2, angle=35)
    else:
        for poly in obj.data.polygons:
            poly.use_smooth = False
    return assign(obj, tile)


def plate_stadium_x(name, yc, radius, y_flat, half_height, x0, x1, tile,
                    *, mirrored=False, bevel=.003, segs=40):
    """Extruded thick plate with a round eye end and a flat neck end."""
    pts = []

    def add(point):
        if not pts or (abs(point[0] - pts[-1][0]) > 1e-9 or
                       abs(point[1] - pts[-1][1]) > 1e-9):
            pts.append(point)

    if not mirrored:
        for k in range(segs + 1):
            a = math.pi / 2 + math.pi * k / segs
            add((yc + radius * math.cos(a), radius * math.sin(a)))
        add((y_flat, -half_height))
        add((y_flat, half_height))
    else:
        add((y_flat, half_height))
        add((yc, half_height))
        for k in range(1, segs + 1):
            a = math.pi / 2 - math.pi * k / segs
            add((yc + radius * math.cos(a), radius * math.sin(a)))
        add((yc, -half_height))
        add((y_flat, -half_height))
    n = len(pts)
    verts = [(x0, y, z) for y, z in pts] + [(x1, y, z) for y, z in pts]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(n + i for i in range(n)))
    obj = mesh_object(name, verts, faces)
    if bevel:
        apply_bevel(obj, bevel, segments=2, angle=35)
    for poly in obj.data.polygons:
        poly.use_smooth = False
    return assign(obj, tile)


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
# Label band: wrap a red-letter yellow label around the sleeve
# ---------------------------------------------------------------------------
def label_uvs(obj, tile, y0, y1, r_mid):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='Atlas')
    layer = mesh.uv_layers.active.data
    x0, y0t, x1, y1t = tile_bounds(tile)
    margin = .012
    w, h = x1 - x0, y1t - y0t
    px0, py0, px1, py1 = tile_bounds(TILE['yellow_plastic'])
    plastic_center = ((px0 + px1) / 2, (py0 + py1) / 2)
    # Start every loop on the clean yellow-plastic texel, then override the
    # outer label surface.  This also covers any bevel-generated loops that are
    # not reached by the per-polygon pass below.
    for d in layer:
        d.uv = plastic_center
    for poly in mesh.polygons:
        c = poly.center
        radial = math.hypot(c.x, c.z)
        if radial > r_mid:
            us = []
            for li in poly.loop_indices:
                v = mesh.vertices[mesh.loops[li].vertex_index].co
                us.append(math.atan2(v.z, v.x) / math.tau + .5)
            for li, u in zip(poly.loop_indices, us):
                v = mesh.vertices[mesh.loops[li].vertex_index].co
                vv = min(1.0, max(0.0, (v.y - y0) / (y1 - y0)))
                uu = u % 1.0
                layer[li].uv = (x0 + margin * w + (1 - uu) * (w - 2 * margin * w),
                                y0t + margin * h + vv * (h - 2 * margin * h))
            poly.use_smooth = True
        else:
            for li in poly.loop_indices:
                layer[li].uv = plastic_center
    print('DEBUG post', min(d.uv[1] for d in layer), max(d.uv[1] for d in layer),
          'bad', sum(1 for d in layer if d.uv[1] < .252))


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def build_rear_eye():
    c = D['rear_clevis']
    plate = plate_stadium_x('rear_clevis', c['eye_center_y'], c['eye_outer_radius'],
                            c['neck_end_y'], c['plate_half_height'],
                            c['plate_x0'], c['plate_x1'], TILE['black_paint'],
                            mirrored=False, bevel=.0025)
    eye_y = c['eye_center_y']
    cutter = along('cut_rear_eye', (c['plate_x0'] - .02, eye_y, 0),
                   (c['plate_x1'] + .02, eye_y, 0), c['eye_bore_radius'],
                   TILE['rubber'], segs=96, bevel=0, cutter=True)
    apply_boolean(plate, [cutter], 'Bore rear eye')
    apply_bevel(plate, .0010, segments=1, angle=35)
    for poly in plate.data.polygons:
        poly.use_smooth = False
    assign(plate, TILE['black_paint'], register=False)
    bushing = annulus_axis('rear_bushing', 'X', (0, eye_y, 0),
                           -c['bushing_half_x'], c['bushing_half_x'],
                           c['bushing_inner_radius'], c['bushing_outer_radius'],
                           TILE['steel'], segs=128, bevel=.0008)
    smooth_side(bushing, (1, 0, 0))
    for side, a0, a1 in (('neg', c['plate_x0'] - .0015, c['plate_x0']),
                         ('pos', c['plate_x1'], c['plate_x1'] + .0015)):
        annulus_axis(f'rear_eye_rim_{side}', 'X', (0, eye_y, 0), a0, a1,
                     c['bushing_outer_radius'], c['bushing_outer_radius'] + .005,
                     TILE['black_paint'], segs=128, bevel=.0004)


def build_rear_cap():
    r = D['rear_cap']
    cap = along('rear_cap', (0, r['y0'], 0), (0, r['y1'], 0), r['radius'],
                TILE['black_paint'], segs=160, bevel=.0020)
    smooth_side(cap, (0, 1, 0))
    torus('rear_cap_weld', r['radius'] + .0004, .0018,
          (0, r['weld_y'], 0), 'Y', TILE['weld'], major_segments=160,
          minor_segments=8)


def build_barrel():
    b = D['barrel']
    obj = along('barrel', (0, b['y0'], 0), (0, b['y1'], 0), b['radius'],
                TILE['black_paint'], segs=192, bevel=.0016)
    smooth_side(obj, (0, 1, 0))
    for name, y in (('rear', b['seam_y0']), ('front', b['seam_y1'])):
        torus(f'barrel_seam_{name}', b['radius'] - .0012, .0012,
              (0, y, 0), 'Y', TILE['weld'], major_segments=192, minor_segments=8)


def build_label_band():
    b = D['label_band']
    band = annulus_axis('yellow_label_band', 'Y', (0, 0, 0), b['y0'], b['y1'],
                        b['r_in'], b['r_out'], TILE['label'], segs=192,
                        bevel=.0004)
    label_uvs(band, TILE['label'], b['y0'], b['y1'],
              (b['r_in'] + b['r_out']) / 2)
    tiles['yellow_label_band'] = 'label+yellow_plastic'
    nonmetal_objects.append(band.name)
    for side, y0, y1, tile in (
            ('lower', b['y0'] - .0018, b['y0'], TILE['steel']),
            ('upper', b['y1'], b['y1'] + .0018, TILE['black_edge'])):
        annulus_axis(f'label_rim_{side}', 'Y', (0, 0, 0), y0, y1,
                     b['r_in'], b['r_out'] + .0004, tile, segs=192,
                     bevel=.0003)


def build_front_gland():
    g = D['front_gland']
    cap = annulus_axis('front_cap', 'Y', (0, 0, 0), g['cap_y0'], g['cap_y1'],
                       g['cap_r_in'], g['cap_r_out'], TILE['black_paint'],
                       segs=160, bevel=.0018)
    smooth_side(cap, (0, 1, 0))
    collar = annulus_axis('front_collar', 'Y', (0, 0, 0), g['collar_y0'],
                          g['collar_y1'], g['collar_r_in'], g['collar_r_out'],
                          TILE['black_paint'], segs=160, bevel=.0016)
    smooth_side(collar, (0, 1, 0))
    torus('front_collar_weld', g['collar_r_out'] - .002, .0016,
          (0, g['collar_y0'] + .001, 0), 'Y', TILE['weld'],
          major_segments=160, minor_segments=8)
    wiper = annulus_axis('front_wiper', 'Y', (0, 0, 0), g['wiper_y0'],
                         g['wiper_y1'], g['wiper_r_in'], g['wiper_r_out'],
                         TILE['steel'], segs=96, bevel=.0005)
    smooth_side(wiper, (0, 1, 0))
    for i, ring in enumerate(D['gland_rings']):
        item = annulus_axis(f'gland_ring_{i:02d}', 'Y', (0, 0, 0),
                            ring['y0'], ring['y1'], ring['r_in'], ring['r_out'],
                            TILE[ring['tile']], segs=128, bevel=.0006)
        smooth_side(item, (0, 1, 0))


def build_rod_and_front_eye():
    r = D['rod']
    rod = along('rod', (0, r['y0'], 0), (0, r['y1'], 0), r['radius'],
                TILE['chrome'], segs=128, bevel=.0012)
    smooth_side(rod, (0, 1, 0))
    b = D['rod_end_boss']
    boss = along('rod_end_boss', (0, b['y0'], 0), (0, b['y1'], 0), b['radius'],
                 TILE['black_paint'], segs=96, bevel=.0020)
    smooth_side(boss, (0, 1, 0))
    c = D['rod_clevis']
    plate = plate_stadium_x('front_clevis', c['eye_center_y'], c['eye_outer_radius'],
                            c['neck_end_y'], c['plate_half_height'],
                            c['plate_x0'], c['plate_x1'], TILE['black_paint'],
                            mirrored=True, bevel=.0015)
    eye_y = c['eye_center_y']
    cutter = along('cut_front_eye', (c['plate_x0'] - .02, eye_y, 0),
                   (c['plate_x1'] + .02, eye_y, 0), c['eye_bore_radius'],
                   TILE['rubber'], segs=96, bevel=0, cutter=True)
    apply_boolean(plate, [cutter], 'Bore front eye')
    # The thin forged eye plate plus a Boolean hole rim produced overlapping
    # bevel faces at the hole edge; keep that rim sharp rather than leave
    # self-intersections in the exported mesh.
    for poly in plate.data.polygons:
        poly.use_smooth = False
    assign(plate, TILE['black_paint'], register=False)
    bushing = annulus_axis('front_bushing', 'X', (0, eye_y, 0),
                           -c['bushing_half_x'], c['bushing_half_x'],
                           c['bushing_inner_radius'], c['bushing_outer_radius'],
                           TILE['steel'], segs=128, bevel=.0008)
    smooth_side(bushing, (1, 0, 0))
    for side, a0, a1 in (('neg', c['plate_x0'] - .0012, c['plate_x0']),
                         ('pos', c['plate_x1'], c['plate_x1'] + .0012)):
        annulus_axis(f'front_eye_rim_{side}', 'X', (0, eye_y, 0), a0, a1,
                     c['bushing_outer_radius'], c['bushing_outer_radius'] + .0045,
                     TILE['black_paint'], segs=128, bevel=.0004)


def build_port(p):
    name = p['name']
    y = p['center_y']
    pad = cylinder_z(f'{name}_port_pad', p['pad_radius'], p['pad_z0'],
                     p['pad_z1'], TILE['black_paint'], x=0, y=y, segs=64,
                     bevel=.0015)
    smooth_side(pad, (0, 0, 1))
    torus(f'{name}_port_weld', p['pad_radius'] + .0005, .0016,
          (0, y, 0.0555), 'Z', TILE['weld'], major_segments=64, minor_segments=6)
    stem = cylinder_z(f'{name}_port_stem', p['stem_radius'], p['stem_z0'],
                      p['stem_z1'], TILE['dark_steel'], x=0, y=y, segs=48,
                      bevel=.0006)
    smooth_side(stem, (0, 0, 1))
    washer = cylinder_z(f'{name}_port_washer', p['nut_radius'] + .0008,
                        p['nut_z0'] - .0016, p['nut_z0'], TILE['steel'],
                        x=0, y=y, segs=64, bevel=.0004)
    smooth_side(washer, (0, 0, 1))
    hex_prism_axis(f'{name}_port_nut', p['nut_radius'],
                   p['nut_z1'] - p['nut_z0'],
                   (0, y, (p['nut_z0'] + p['nut_z1']) / 2), (0, 0, 1),
                   TILE['steel'], bevel=.0008)
    cap = cylinder_z(f'{name}_port_cap', p['cap_radius'], p['cap_z0'],
                     p['cap_z1'], TILE['yellow_plastic'], x=0, y=y, segs=64,
                     bevel=.0014)
    smooth_side(cap, (0, 0, 1))
    for i, z in enumerate((p['cap_z0'] + .004, p['cap_z0'] + .008,
                           p['cap_z0'] + .012)):
        torus(f'{name}_port_groove_{i}', p['cap_radius'] - .0005, .00055,
              (0, y, z), 'Z', TILE['yellow_plastic'], major_segments=64,
              minor_segments=6)
    torus(f'{name}_port_cap_rim', p['cap_radius'] - .0018, .0010,
          (0, y, p['cap_z1'] - .0014), 'Z', TILE['yellow_plastic'],
          major_segments=64, minor_segments=6)


def build_ports():
    for p in D['ports']:
        build_port(p)


def build_all():
    build_rear_eye()
    build_rear_cap()
    build_barrel()
    build_label_band()
    build_front_gland()
    build_rod_and_front_eye()
    build_ports()


# ---------------------------------------------------------------------------
# Scene, cameras, export and renders
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
    """Verify both eye through-holes, bushings, ports and required part counts."""
    checks = {'eye_holes': {}, 'bushings': {}, 'ports': {}, 'label': {},
              'barrel': {}, 'counts': {}}
    for which, center_name, item in (('rear', 'rear_clevis', D['rear_clevis']),
                                     ('front', 'front_clevis', D['rod_clevis'])):
        plate = make_bvh(bpy.data.objects[center_name])
        eye_y = item['eye_center_y']
        rays = []
        for side in (-1, 1):
            origin = Vector((side * .20, eye_y, 0))
            hit = plate.ray_cast(origin, Vector((-side, 0, 0)), 1.0)
            rays.append({'side': side, 'hit': hit[0] is not None})
        checks['eye_holes'][which] = {'through': all(not r['hit'] for r in rays),
                                      'rays': rays}
        bushing = make_bvh(bpy.data.objects[f'{which}_bushing'])
        mid = (item['bushing_outer_radius'] + item['bushing_inner_radius']) / 2
        bushing_rays = []
        for side in (-1, 1):
            origin = Vector((side * .20, eye_y + mid, 0))
            hit = bushing.ray_cast(origin, Vector((-side, 0, 0)), 1.0)
            bushing_rays.append({'side': side, 'hit': hit[0] is not None})
        checks['bushings'][which] = {'present': all(r['hit'] for r in bushing_rays),
                                     'rays': bushing_rays}

    port_rows = []
    for p in D['ports']:
        name = p['name']
        cap = make_bvh(bpy.data.objects[f'{name}_port_cap'])
        origin_top = Vector((0, p['center_y'], .25))
        hit = cap.ray_cast(origin_top, Vector((0, 0, -1)), 1.0)
        z_top = None
        if hit[0] is not None:
            z_top = .25 - float(hit[3])
        nut = make_bvh(bpy.data.objects[f'{name}_port_nut'])
        nut_hit = nut.ray_cast(Vector((0.03, p['center_y'], (p['nut_z0'] + p['nut_z1']) / 2)),
                               Vector((-1, 0, 0)), 1.0)
        port_rows.append({
            'name': name,
            'cap_hit': hit[0] is not None,
            'cap_top_z': round(z_top, 5) if z_top is not None else None,
            'protrudes_above_barrel': bool(z_top is not None and z_top > D['barrel']['radius'] + .025),
            'hex_hit_from_side': nut_hit[0] is not None,
            'y_m': p['center_y'],
        })
    checks['ports'] = {'rows': port_rows,
                       'count': len(port_rows),
                       'same_side': all(p['center_y'] is not None for p in D['ports']),
                       'all_caps_present': all(r['cap_hit'] for r in port_rows),
                       'all_protrude': all(r['protrudes_above_barrel'] for r in port_rows),
                       'hex_nuts_present': sum(1 for r in port_rows if r['hex_hit_from_side'])}

    band = make_bvh(bpy.data.objects['yellow_label_band'])
    band_hit = band.ray_cast(Vector((0.02, .20, .20)), Vector((0, 0, -1)), 1.0)
    checks['label'] = {
        'band_present': band_hit[0] is not None,
        'text': cfg['dimensions']['label_band']['text'],
        'y0': cfg['dimensions']['label_band']['y0'],
        'y1': cfg['dimensions']['label_band']['y1'],
        'note': '环带存在与位置已由射线确认；红字纹理为近似重绘，不能证明原厂字稿。',
    }

    b = D['barrel']
    barrel = make_bvh(bpy.data.objects['barrel'])
    top = barrel.ray_cast(Vector((0, 0, .2)), Vector((0, 0, -1)), 1.0)
    bottom = barrel.ray_cast(Vector((0, 0, -.2)), Vector((0, 0, 1)), 1.0)
    top_z = .2 - float(top[3]) if top[0] is not None else None
    bottom_z = -.2 + float(bottom[3]) if bottom[0] is not None else None
    checks['barrel'] = {
        'top_hit': top[0] is not None,
        'bottom_hit': bottom[0] is not None,
        'top_radius_m': round(top_z, 5) if top_z is not None else None,
        'bottom_radius_m': round(abs(bottom_z), 5) if bottom_z is not None else None,
        'solid_closed_barrel': bool(top[0] is not None and bottom[0] is not None),
    }

    checks['counts'] = {
        'gland_rings': len([n for n in bpy.data.objects.keys()
                            if n.startswith('gland_ring_')]),
        'ports': len(D['ports']),
        'port_caps': len([n for n in bpy.data.objects.keys()
                          if n.endswith('_port_cap')]),
        'eye_bushings': len([n for n in bpy.data.objects.keys()
                             if n.endswith('_bushing')]),
        'label_rims': len([n for n in bpy.data.objects.keys()
                           if n.startswith('label_rim_')]),
    }
    checks['passed'] = bool(
        all(v['through'] for v in checks['eye_holes'].values())
        and all(v['present'] for v in checks['bushings'].values())
        and checks['ports']['count'] == 2
        and checks['ports']['all_caps_present']
        and checks['ports']['all_protrude']
        and checks['ports']['hex_nuts_present'] == 2
        and checks['label']['band_present']
        and checks['barrel']['solid_closed_barrel']
        and checks['counts']['gland_rings'] == 6)
    return checks


def pose_assembly():
    """Rotate the locally-built assembly into the old TripoSR PCA pose."""
    low, high, center, span = bounds(objects)
    for obj in objects:
        obj.location += -center
    bpy.context.view_layer.update()
    rotation = Matrix.Rotation(POSE_ROTATION_X, 4, 'X')
    for obj in objects:
        obj.matrix_world = rotation @ obj.matrix_world
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.update()


def main():
    global MATERIAL, CAM
    scene = legacy.setup_scene('CPU', 512)
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, 'denoiser'):
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    MATERIAL = atlas_material()
    CAM = init_camera(scene)
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Strength'].default_value = .70
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 130
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 3.6
        fill.data.size_y = 3.6
        fill.location = (-3.0, 2.6, 2.2)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.6, -2.8, 3.9), 260, 3.2),
            ('front reflection', (3.0, 1.5, 1.0), 120, 1.5),
            ('top fill', (0.0, 0.0, 4.0), 90, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 80, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 45, 1.4),
            ('side fill', (0.4, 3.0, 0.3), 60, 1.8)):
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
    # Ray checks are performed in the local build frame, where the eye bores
    # are along X and the ports/label are radial +Z.  The assembly is rotated
    # into the library pose afterwards, so world-space rays would no longer be
    # axis-aligned.
    ray = ray_checks()
    pose_assembly()
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
    directions = {'front': (1, 0, .05), 'back': (-1, 0, .05),
                  'left': (0, -1, .05), 'right': (0, 1, .05),
                  'top': (0, 0, 1), 'bottom': (0, 0, -1)}
    rr = cfg['render']
    for name, direction in directions.items():
        view_scale = span * (1.20 if name in ('left', 'right', 'top', 'bottom') else 1.08)
        look(scene, direction, center, span, ortho_scale=view_scale)
        draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    source = request['source_camera']
    set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
               shift_x=source.get('shift_x', 0), shift_y=source.get('shift_y', 0))
    draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])

    if not request.get('draft'):
        target = {
            'rear-eye': object_center('rear_bushing'),
            'front-eye': object_center('front_bushing'),
            'upper-port': object_center('upper_port_cap'),
            'lower-port': object_center('lower_port_cap'),
            'gland-stack': object_center('gland_ring_02'),
            'label-band': object_center('yellow_label_band'),
            'rear-cap': object_center('rear_cap'),
            'rod': object_center('rod'),
            'barrel': object_center('barrel'),
            'overall-3q': center,
        }
        detail_specs = [
            ('rear-eye', (1.0, -.35, .20), target['rear-eye'], .175),
            ('front-eye', (1.0, .25, .18), target['front-eye'], .150),
            ('upper-port', (.55, -.55, .80), target['upper-port'], .080),
            ('lower-port', (.55, .55, .80), target['lower-port'], .080),
            ('gland-stack', (.75, .55, .30), target['gland-stack'], .110),
            ('label-band', (.70, -.50, .35), target['label-band'], .230),
            ('rear-cap', (.55, -.80, .30), target['rear-cap'], .130),
            ('rod', (.65, .60, .30), target['rod'], .200),
            ('barrel', (.70, -.55, .38), target['barrel'], .540),
            ('overall-3q', (1.0, -1.0, .55), center, span * 1.14),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        set_visible({'rear_clevis', 'rear_bushing',
                     'rear_eye_rim_neg', 'rear_eye_rim_pos'})
        look(scene, (1, -.35, .15), object_center('rear_bushing'), span,
             ortho_scale=.185)
        draw(scene, render_dir / 'rear-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'front_clevis', 'front_bushing', 'front_eye_rim_neg',
                     'front_eye_rim_pos', 'rod', 'rod_end_boss'})
        look(scene, (1, .30, .16), object_center('front_bushing'), span,
             ortho_scale=.170)
        draw(scene, render_dir / 'front-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        gland_names = {'rod', 'front_cap', 'front_collar', 'front_wiper',
                       'front_collar_weld', 'rod_end_boss'}
        gland_names |= {n for n in bpy.data.objects.keys()
                        if n.startswith('gland_ring_')}
        set_visible(gland_names)
        look(scene, (.80, .55, .28), object_center('gland_ring_02'), span,
             ortho_scale=.150)
        draw(scene, render_dir / 'gland-rod-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        port_names = set()
        for p in D['ports']:
            prefix = f"{p['name']}_port_"
            port_names |= {n for n in bpy.data.objects.keys()
                           if n.startswith(prefix)}
        set_visible(port_names)
        look(scene, (.55, -.75, .80), object_center('upper_port_cap'), span,
             ortho_scale=.220)
        draw(scene, render_dir / 'ports-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'yellow_label_band', 'label_rim_lower', 'label_rim_upper',
                     'barrel'})
        look(scene, (.70, -.55, .38), object_center('yellow_label_band'), span,
             ortho_scale=.260)
        draw(scene, render_dir / 'label-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'barrel', 'barrel_seam_rear', 'barrel_seam_front',
                     'rear_cap', 'rear_cap_weld', 'front_cap', 'front_collar',
                     'front_wiper', 'front_collar_weld', 'rear_clevis'})
        look(scene, (.70, -.55, .38), center, span, ortho_scale=.560)
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
        main_light.location = (2.6, -2.8, 3.9)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()

        set_visible(set(metal_objects))
        for name in ('soft light', 'front reflection', 'top fill', 'rim',
                     'low fill', 'side fill'):
            light = bpy.data.objects.get(name)
            if light is not None:
                light.hide_render = True
        metal_target = object_center('rod')
        look(scene, (.85, -.45, .35), metal_target, span, ortho_scale=.260)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (.85, -.45, .35), metal_target, span, ortho_scale=.260)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.6, -2.8, 3.9)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for name in ('soft light', 'front reflection', 'top fill', 'rim',
                     'low fill', 'side fill'):
            light = bpy.data.objects.get(name)
            if light is not None:
                light.hide_render = False
        set_visible({obj.name for obj in objects})

        for i in range(24):
            a = math.tau * i / 24
            look(scene, (math.cos(a), math.sin(a), .20), center, span)
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
        'geometry_basis': '原照可见结构 + 未标定六视图同侧双油口拓扑（长行程黑色双耳环液压缸）',
        'pose': {'rotation_x_deg': round(math.degrees(POSE_ROTATION_X), 3),
                 'old_mesh_pca_axis_zyx_reference': [0.051, 0.681, -0.731],
                 'note': '按旧网格 PCA 主轴恢复原照斜置姿态；不是摄影测量。'},
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'config_version': cfg['version'],
        'mixed_tile_parts': {'yellow_label_band': '外表面 label 环带，内/端面为黄色塑料；按非金属登记'},
    }
    (out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                            encoding='utf8')


main()
