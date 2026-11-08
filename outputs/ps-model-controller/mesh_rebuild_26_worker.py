"""Blender worker: part-based rebuild of item 26, a red welded hydraulic cylinder.

Evidence:
- one 1254x1254 clean photograph and its matching rust derivative of a red
  welded hydraulic cylinder with a single eye clevis and steel bushing at the
  top, a double-groove collar, a barrel with a large barcode label, two welded
  side ports at different axial stations on the same visible side, and a bottom
  cap with a short painted rod stub containing a small central blind hole;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified).  It is used only for topology and the unseen back.
  Its panels contradict each other and the photograph (the top panel shows the
  eye bore along the barrel axis; the front/bottom panels show the two ports
  diametrically opposite at one station, while the photograph shows both on the
  same side at different axial stations).  The photograph is authoritative.

The old TripoSR library mesh has only about 4.8k vertices and 9.7k faces; the
six-view audit and direct inspection show a melted surface, fused ports and an
unreadable label.  Its source-view contour passed the old single-image gate
(IoU about 0.927, no filled holes), but that one contour cannot prove the
structure.  It is retained unchanged as a comparison only.  The candidate is a
parametric part rebuild with a true eye through-bore and steel bushing, a
separate collar, barrel, two welded port bosses with threaded spigots and real
blind bores, a bottom cap/cone/stub with a real blind hole, a curved printed
label patch, and red/steel/dark-steel/rust/label PBR materials.

Coordinates: Z up, cylinder axis along Y, eye at +Y, bottom stub at -Y, eye
bore along X, both ports on the +Z side.  All candidates remain manual-only
and pending human review; nothing replaces the library mesh or enters
automatic placement or the formal composite.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector
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

objects = []
tiles = {}
metal_objects = []
nonmetal_objects = []
MATERIAL = None
CAM = None
METAL_TILES = {'steel', 'dark_steel'}
DIELECTRIC_TILES = {'red', 'red_dark', 'weld', 'rust', 'rubber', 'label',
                    'spare8', 'spare9', 'spare10', 'spare11', 'spare12',
                    'spare13', 'spare14', 'spare15'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed red paint, steel, dark steel and label PBR atlas')
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


def tile_bounds(idx):
    """UV bounds for atlas tile idx.

    The driver writes tile idx at image row idx//4 counted from the top.  Blender
    UV v=0 is the image bottom, so the v range is mirrored.  This is the
    corrected convention; it keeps the sampled tile identical to the layout
    sheet.
    """
    col, row = idx % 4, idx // 4
    v0 = (3 - row) * .25
    return (col * .25, v0, (col + 1) * .25, v0 + .25)


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
    elif mode == 'keep':
        pass
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


def cone_along(name, start, end, r_start, r_end, tile, *, segs=96, bevel=.001):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cone_add(vertices=segs, radius1=r_start,
                                    radius2=r_end, depth=(end - start).length,
                                    location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, smooth=True)


def torus(name, major, minor, location, axis, tile, *, major_segments=96,
          minor_segments=10, cutter=False):
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


def torus_axis(name, major, minor, location, axis, tile, *, major_segments=64,
               minor_segments=8):
    axis = Vector(axis).normalized()
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=location,
                                     major_radius=major, minor_radius=minor)
    obj = bpy.context.object
    obj.rotation_euler = axis.to_track_quat('Z', 'Y').to_euler()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=0, constant=True)


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


def annulus_along(name, start, end, r_in, r_out, tile, *, segs=96, bevel=0.0):
    """Closed washer/tube between two points on an arbitrary axis."""
    start, end = Vector(start), Vector(end)
    d = end - start
    if d.length < 1e-9:
        raise ValueError('annulus_along needs distinct endpoints')
    d.normalize()
    up = Vector((0, 0, 1))
    if abs(d.dot(up)) > .95:
        up = Vector((1, 0, 0))
    u = d.cross(up).normalized()
    v = d.cross(u).normalized()
    verts, rings = [], []
    for point, radius in ((start, r_in), (end, r_in),
                          (end, r_out), (start, r_out)):
        ids = []
        for j in range(segs):
            t = math.tau * j / segs
            p = point + u * (radius * math.cos(t)) + v * (radius * math.sin(t))
            ids.append(len(verts))
            verts.append(tuple(p))
        rings.append(ids)
    faces = []
    for k in range(4):
        a, b = rings[k], rings[(k + 1) % 4]
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((a[j], a[j2], b[j2], b[j]))
    obj = mesh_object(name, verts, faces)
    if bevel:
        apply_bevel(obj, bevel, segments=2, angle=35)
    else:
        for poly in obj.data.polygons:
            poly.use_smooth = False
    obj = assign(obj, tile)
    smooth_side(obj, d)
    return obj


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


def curved_patch(name, y0, y1, angle_deg, r_in, r_out, tile, *, steps=28):
    """Closed curved slab on the +X side of the barrel, with custom label UVs."""
    a = math.radians(angle_deg)
    verts, faces, uvs = [], [], []
    rings = []
    for radius in (r_in, r_out):
        rows = []
        for iy, y in enumerate((y0, y1)):
            row = []
            for j in range(steps + 1):
                t = -a + 2 * a * j / steps
                x = radius * math.cos(t)
                z = radius * math.sin(t)
                row.append(len(verts))
                verts.append((x, y, z))
            rows.append(row)
        rings.append(rows)
    # outer and inner surfaces
    for layer, flip in ((1, False), (0, True)):
        rows = rings[layer]
        for iy in range(1):
            for j in range(steps):
                p0, p1 = rows[iy][j], rows[iy][j + 1]
                p2, p3 = rows[iy + 1][j + 1], rows[iy + 1][j]
                face = (p0, p1, p2, p3) if not flip else (p3, p2, p1, p0)
                faces.append(face)
    # side walls: y ends and angular ends
    for j in range(steps):
        # y0 and y1 end walls
        faces.append((rings[0][0][j], rings[0][0][j + 1],
                      rings[1][0][j + 1], rings[1][0][j]))
        faces.append((rings[0][1][j + 1], rings[0][1][j],
                      rings[1][1][j], rings[1][1][j + 1]))
    for j in (0, steps):
        for iy in range(1):
            p0 = rings[0][iy][j]
            p1 = rings[0][iy + 1][j]
            p2 = rings[1][iy + 1][j]
            p3 = rings[1][iy][j]
            if j == 0:
                faces.append((p0, p1, p2, p3))
            else:
                faces.append((p3, p2, p1, p0))
    obj = mesh_object(name, verts, faces)
    # custom UVs: u along the barrel (Y), v around the circumference.  The
    # central half of the label atlas tile holds the 2:1 printed artwork.
    layer = obj.data.uv_layers.new(name='Atlas')
    x0, y0uv, x1, y1uv = tile_bounds(tile)
    w, h = x1 - x0, y1uv - y0uv
    for poly in obj.data.polygons:
        for li in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[li].vertex_index].co
            if co.x < 0:
                # inner/back faces are hidden; keep them in the label tile
                u = 0.02
                v = 0.5
            else:
                u = (co.y - y0) / (y1 - y0) if abs(y1 - y0) > 1e-9 else .5
                theta = math.atan2(co.z, co.x)
                v = (theta + a) / (2 * a) if a > 0 else .5
            layer.data[li].uv = (x0 + .012 * w + u * .976 * w,
                                 y0uv + .26 * h + v * .48 * h)
    return assign(obj, tile, mode='keep')


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def build_eye():
    e = D['eye']
    yc = e['center_y']
    half_x = e['plate_half_x']
    # A true washer plate: the eye bore is the annulus inner hole, so no
    # boolean seam is needed and the mesh stays watertight.
    plate = annulus_axis('eye_plate', 'X', (0, yc, 0), -half_x, half_x,
                         e['bore_radius'], e['outer_radius'], TILE['red'],
                         segs=112, bevel=.0018)
    smooth_side(plate, (1, 0, 0))
    for side, name in ((-1, 'eye_boss_neg'), (1, 'eye_boss_pos')):
        a0 = side * half_x
        a1 = side * (half_x + e['boss_depth'])
        annulus_axis(name, 'X', (0, yc, 0), min(a0, a1), max(a0, a1),
                     e['bore_radius'], e['boss_outer_radius'], TILE['red'],
                     segs=96, bevel=.001)
    annulus_axis('eye_bushing', 'X', (0, yc, 0), -e['bushing_half_x'],
                 e['bushing_half_x'], e['bushing_inner_radius'],
                 e['bushing_outer_radius'], TILE['steel'], segs=112,
                 bevel=.0007)
    neck_y0, neck_y1 = e['neck_y0'], e['neck_y1']
    box('eye_neck', (0, (neck_y0 + neck_y1) / 2, 0),
        (2 * e['neck_half_x'], neck_y1 - neck_y0, 2 * e['neck_half_z']),
        TILE['red'], bevel=.004)


def build_collar():
    c = D['collar']
    collar = along('collar', (0, c['y0'], 0), (0, c['y1'], 0), c['radius'],
                   TILE['red'], segs=128, bevel=.002)
    smooth_side(collar, (0, 1, 0))
    torus('collar_ring_top', c['radius'] + .0008, c['ring_minor'],
          (0, c['top_ring_y'], 0), 'Y', TILE['red_dark'],
          major_segments=128, minor_segments=8)
    torus('collar_ring_bottom', c['radius'] + .0008, c['ring_minor'],
          (0, c['bottom_ring_y'], 0), 'Y', TILE['red_dark'],
          major_segments=128, minor_segments=8)
    torus('collar_groove', c['radius'] + .0004, c['groove_minor'],
          (0, c['groove_y'], 0), 'Y', TILE['rubber'],
          major_segments=128, minor_segments=8)
    torus('collar_weld', c['radius'] + .001, .0022,
          (0, c['y0'] + .003, 0), 'Y', TILE['weld'],
          major_segments=128, minor_segments=8)


def build_barrel():
    b = D['barrel']
    barrel = along('barrel', (0, b['y0'], 0), (0, b['y1'], 0), b['radius'],
                   TILE['red'], segs=160, bevel=.0018)
    smooth_side(barrel, (0, 1, 0))
    for name, y in (('top', b['seam_top']), ('bottom', b['seam_bottom'])):
        torus(f'barrel_seam_{name}', b['radius'] + .0011, .0018,
              (0, y, 0), 'Y', TILE['weld'], major_segments=160,
              minor_segments=8)


def build_bottom_end():
    b = D['bottom_end']
    band = along('bottom_band', (0, b['band_y0'], 0), (0, b['band_y1'], 0),
                 b['band_radius'], TILE['red'], segs=128, bevel=.003)
    smooth_side(band, (0, 1, 0))
    cap = along('bottom_cap', (0, b['cap_y0'], 0), (0, b['cap_y1'], 0),
                b['cap_radius'], TILE['red'], segs=128, bevel=.003)
    smooth_side(cap, (0, 1, 0))
    cone_along('bottom_cone', (0, b['cone_y0'], 0), (0, b['cone_y1'], 0),
               b['cone_r0'], b['cone_r1'], TILE['red'], segs=112, bevel=.0015)
    base = along('stub_base', (0, b['stub_base_y0'], 0),
                 (0, b['stub_base_y1'], 0), b['stub_base_radius'],
                 TILE['red'], segs=96, bevel=.002)
    smooth_side(base, (0, 1, 0))
    # The blind hole is built as a tube plus a solid rear section instead of a
    # boolean, so every stub part remains watertight.
    stub_tube = annulus_axis('stub_tube', 'Y', (0, 0, 0),
                             b['stub_hole_y0'], b['stub_hole_y1'],
                             b['stub_hole_radius'], b['stub_radius'],
                             TILE['red'], segs=96, bevel=.0015)
    smooth_side(stub_tube, (0, 1, 0))
    stub_solid = along('stub_solid', (0, b['stub_hole_y1'] - .001, 0),
                       (0, b['stub_y1'], 0), b['stub_radius'], TILE['red'],
                       segs=96, bevel=.0025)
    smooth_side(stub_solid, (0, 1, 0))
    annulus_axis('stub_liner', 'Y', (0, 0, 0), b['stub_hole_y0'],
                 b['stub_hole_y1'], b['stub_liner_r_in'],
                 b['stub_liner_r_out'], TILE['dark_steel'], segs=64,
                 bevel=.0004)
    annulus_axis('stub_mouth_ring', 'Y', (0, 0, 0), b['stub_hole_y0'],
                 b['stub_hole_y0'] + b['stub_mouth_thickness'],
                 b['stub_liner_r_in'], b['stub_mouth_r_out'], TILE['steel'],
                 segs=64, bevel=.00035)
    along('stub_hole_bottom', (0, b['stub_hole_y1'] - .0015, 0),
          (0, b['stub_hole_y1'], 0), b['stub_liner_r_out'], TILE['dark_steel'],
          segs=64, bevel=.0003)


def build_port(which):
    p = D['ports']
    spec = p[which]
    y = spec['y']
    tilt = math.radians(spec['tilt_deg'])
    axis = Vector((0.0, -math.sin(tilt), math.cos(tilt))).normalized()
    base = Vector((0.0, y, p['base_z']))
    boss_end = base + axis * p['boss_length']
    spigot_start = boss_end - axis * p['spigot_overlap']
    spigot_end = boss_end + axis * p['spigot_length']
    bore_end = spigot_end - axis * p['bore_depth']
    name = f'port_{which}'
    # Tubes keep the bore real and the mesh watertight without boolean seams.
    boss = annulus_along(f'{name}_boss', base, boss_end,
                         p['bore_radius'], p['boss_radius'], TILE['red'],
                         segs=64, bevel=.0018)
    spigot = annulus_along(f'{name}_spigot', spigot_start, spigot_end,
                           p['bore_radius'], p['spigot_radius'], TILE['red'],
                           segs=64, bevel=.0012)
    # dark liner, mouth ring and blind bottom
    liner = annulus_along(f'{name}_liner', bore_end, spigot_end,
                          p['bore_liner_r_in'], p['bore_radius'],
                          TILE['dark_steel'], segs=64, bevel=.0004)
    ring = annulus_along(f'{name}_mouth_ring',
                         spigot_end - axis * p['mouth_ring_thickness'],
                         spigot_end, p['bore_liner_r_in'],
                         p['mouth_ring_r_out'], TILE['steel'], segs=64,
                         bevel=.00035)
    along(f'{name}_hole_bottom', bore_end - axis * .0015, bore_end,
          p['bore_radius'], TILE['dark_steel'], segs=64, bevel=.0003)
    torus_axis(f'{name}_weld', p['boss_radius'] + .001, p['weld_minor'],
               base + axis * p['weld_offset'], axis, TILE['weld'],
               major_segments=64, minor_segments=8)
    for i in range(p['thread_count']):
        t = spigot_end - axis * (.004 + i * p['thread_spacing'])
        torus_axis(f'{name}_thread_{i:02d}', p['spigot_radius'] + .0002,
                   p['thread_minor'], t, axis, TILE['red_dark'],
                   major_segments=64, minor_segments=6)
    return boss, spigot, liner, ring


def build_label():
    l = D['label']
    curved_patch('label_patch', l['y0'], l['y1'], l['angle_deg'],
                 l['r_in'], l['r_out'], TILE['label'])


def build_all():
    build_eye()
    build_collar()
    build_barrel()
    build_bottom_end()
    build_port('upper')
    build_port('lower')
    build_label()


# ---------------------------------------------------------------------------
# Scene, rays, export and renders
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
        from mathutils import Quaternion
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


def port_axis(which):
    tilt = math.radians(D['ports'][which]['tilt_deg'])
    return Vector((0.0, -math.sin(tilt), math.cos(tilt))).normalized()


def port_points(which):
    p = D['ports']
    y = p[which]['y']
    axis = port_axis(which)
    base = Vector((0.0, y, p['base_z']))
    boss_end = base + axis * p['boss_length']
    spigot_end = boss_end + axis * p['spigot_length']
    bore_end = spigot_end - axis * p['bore_depth']
    return base, boss_end, spigot_end, bore_end, axis


def ray_checks():
    checks = {'eye': {}, 'bushing': {}, 'ports': {}, 'stub': {},
              'label': {}, 'barrel': {}}
    e = D['eye']
    plate = make_bvh(bpy.data.objects['eye_plate'])
    eye_rays = []
    for side in (-1, 1):
        origin = Vector((side * .25, e['center_y'], 0))
        hit = plate.ray_cast(origin, Vector((-side, 0, 0)), 1.0)
        eye_rays.append({'side': side, 'hit': hit[0] is not None})
    checks['eye'] = {'through': all(not r['hit'] for r in eye_rays),
                     'rays': eye_rays}
    bushing = make_bvh(bpy.data.objects['eye_bushing'])
    bushing_rays = []
    for side in (-1, 1):
        r = (e['bushing_outer_radius'] + e['bushing_inner_radius']) / 2
        origin = Vector((side * .25, e['center_y'] + r, 0))
        hit = bushing.ray_cast(origin, Vector((-side, 0, 0)), 1.0)
        bushing_rays.append({'side': side, 'hit': hit[0] is not None})
    checks['bushing'] = {'present': all(r['hit'] for r in bushing_rays),
                         'rays': bushing_rays}
    p = D['ports']
    for which in ('upper', 'lower'):
        base, boss_end, spigot_end, bore_end, axis = port_points(which)
        boss = make_bvh(bpy.data.objects[f'port_{which}_boss'])
        spigot = make_bvh(bpy.data.objects[f'port_{which}_spigot'])
        origin = spigot_end + axis * .20
        hit_boss = boss.ray_cast(origin, -axis, .40)
        hit_spigot = spigot.ray_cast(origin, -axis, .40)
        cap = make_bvh(bpy.data.objects[f'port_{which}_hole_bottom'])
        cap_hit = cap.ray_cast(origin, -axis, .40)
        checks['ports'][which] = {
            'mouth_open': hit_boss[0] is None and hit_spigot[0] is None,
            'blind_bottom_hit': cap_hit[0] is not None,
            'spigot_end': list(spigot_end),
            'bore_end': list(bore_end),
            'note': '沿油口轴线从外端向内的中心射线；孔口环/衬套为独立部件，不参与此命中判定。',
        }
    b = D['bottom_end']
    stub_tube = make_bvh(bpy.data.objects['stub_tube'])
    stub_solid = make_bvh(bpy.data.objects['stub_solid'])
    stub_tube_hit = stub_tube.ray_cast(Vector((0, -0.70, 0)), Vector((0, 1, 0)), .50)
    stub_solid_hit = stub_solid.ray_cast(Vector((0, -0.70, 0)), Vector((0, 1, 0)), .50)
    checks['stub'] = {
        'hole_open': stub_tube_hit[0] is None,
        'blind_bottom_hit': stub_solid_hit[0] is not None,
        'hole_y0': b['stub_hole_y0'], 'hole_y1': b['stub_hole_y1'],
    }
    label = make_bvh(bpy.data.objects['label_patch'])
    label_hit = label.ray_cast(Vector((.30, (D['label']['y0'] + D['label']['y1']) / 2,
                                       0)), Vector((-1, 0, 0)), .50)
    checks['label'] = {'present': label_hit[0] is not None}
    barrel = bpy.data.objects['barrel']
    radii = [math.hypot(v.co.x, v.co.z) for v in barrel.data.vertices
             if abs(v.co.y) < D['barrel']['y1'] - .01]
    checks['barrel'] = {
        'radial_deviation_m': round(float(max(abs(r - D['barrel']['radius'])
                                              for r in radii)), 6) if radii else None,
    }
    checks['port_count'] = len([n for n in bpy.data.objects.keys()
                                if n.startswith('port_') and n.endswith('_boss')])
    checks['passed'] = bool(
        checks['eye']['through']
        and checks['bushing']['present']
        and all(v['mouth_open'] and v['blind_bottom_hit']
                for v in checks['ports'].values())
        and checks['stub']['hole_open'] and checks['stub']['blind_bottom_hit']
        and checks['label']['present']
        and checks['port_count'] == 2)
    return checks


def main():
    global MATERIAL, CAM
    scene = legacy.setup_scene('CPU', 512)
    scene.render.threads_mode = 'AUTO'
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, 'denoiser'):
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    MATERIAL = atlas_material()
    CAM = init_camera(scene)
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Strength'].default_value = .42
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 95
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 3.4
        fill.data.size_y = 3.4
        fill.location = (-3.0, 2.6, 2.2)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 200, 3.2),
            ('front reflection', (3.0, 1.4, 1.0), 95, 1.5),
            ('top fill', (0.0, 0.0, 4.0), 70, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 60, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 28, 1.4)):
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
    directions = {'front': (1, 0, .05), 'back': (-1, 0, .05),
                  'left': (0, -1, .05), 'right': (0, 1, .05),
                  'top': (0, 0, 1), 'bottom': (0, 0, -1)}
    rr = cfg['render']
    for name, direction in directions.items():
        view_scale = span * (0.46 if name in ('left', 'right') else
                             (0.62 if name in ('top', 'bottom') else 1.12))
        look(scene, direction, center, span, ortho_scale=view_scale)
        draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    source = request['source_camera']
    set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
               shift_x=source.get('shift_x', 0), shift_y=source.get('shift_y', 0))
    draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])

    if not request.get('draft'):
        target = {
            'eye-bushing': object_center('eye_bushing'),
            'eye-bore': object_center('eye_plate'),
            'collar': object_center('collar'),
            'label': object_center('label_patch'),
            'port-upper': object_center('port_upper_spigot'),
            'port-lower': object_center('port_lower_spigot'),
            'bottom-cap': object_center('bottom_band'),
            'bottom-stub': object_center('stub_tube'),
            'barrel-seam': object_center('barrel_seam_top'),
        }
        detail_specs = [
            ('eye-bushing', (1.0, -.10, .16), target['eye-bushing'], .235),
            ('eye-bore', (.92, -.30, .22), target['eye-bore'], .205),
            ('collar', (1.0, -.18, .48), target['collar'], .240),
            ('label', (1.0, .04, .06), target['label'], .225),
            ('port-upper', (.55, -.35, .76), target['port-upper'], .230),
            ('port-lower', (.45, -.62, .65), target['port-lower'], .230),
            ('bottom-cap', (.80, -.55, .30), target['bottom-cap'], .260),
            ('bottom-stub', (1.0, -.28, .20), target['bottom-stub'], .220),
            ('barrel-seam', (.72, .52, .44), target['barrel-seam'], .240),
            ('overall-3q', (1.0, -1.0, .55), center, span * 1.14),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        # Isolated subassemblies.
        set_visible({'eye_plate', 'eye_boss_neg', 'eye_boss_pos', 'eye_neck',
                     'eye_bushing'})
        look(scene, (1, -.10, .16), object_center('eye_bushing'), span,
             ortho_scale=.235)
        draw(scene, render_dir / 'eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'collar', 'collar_ring_top', 'collar_ring_bottom',
                     'collar_groove', 'collar_weld'})
        look(scene, (1, -.18, .48), object_center('collar'), span,
             ortho_scale=.240)
        draw(scene, render_dir / 'collar-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'barrel', 'barrel_seam_top', 'barrel_seam_bottom'})
        look(scene, (.35, -.60, .72), center, span, ortho_scale=.58)
        draw(scene, render_dir / 'barrel-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        for which in ('upper', 'lower'):
            names = {n for n in bpy.data.objects.keys()
                     if n.startswith(f'port_{which}_')}
            set_visible(names)
            look(scene, (.55, -.35, .76) if which == 'upper' else (.45, -.62, .65),
                 object_center(f'port_{which}_spigot'), span, ortho_scale=.235)
            draw(scene, render_dir / f'port-{which}-only.png',
                 rr['detail_resolution'], rr['detail_samples'])
        set_visible({'bottom_band', 'bottom_cap', 'bottom_cone', 'stub_base',
                     'stub_tube', 'stub_solid', 'stub_liner', 'stub_mouth_ring',
                     'stub_hole_bottom'})
        look(scene, (.80, -.55, .30), object_center('bottom_band'), span,
             ortho_scale=.275)
        draw(scene, render_dir / 'bottom-end-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({obj.name for obj in objects})
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-material-front.png', rr['six_view_resolution'],
             rr['six_view_samples'])

        # Clay.
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

        # Alternate light for the same front camera.
        main_light = bpy.data.objects['main softbox']
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-light-alternate.png', rr['six_view_resolution'],
             rr['six_view_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()

        # Metal-only brightness comparison with a moving main light.
        set_visible(set(metal_objects))
        fill_light = bpy.data.objects.get('soft light')
        front_light = bpy.data.objects.get('front reflection')
        top_light = bpy.data.objects.get('top fill')
        rim = bpy.data.objects.get('rim')
        low_light = bpy.data.objects.get('low fill')
        for light in (fill_light, front_light, top_light, rim, low_light):
            if light is not None:
                light.hide_render = True
        metal_target = object_center('eye_bushing')
        look(scene, (1.0, -.08, .10), metal_target, span, ortho_scale=.225)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (1.0, -.08, .10), metal_target, span, ortho_scale=.225)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for light in (fill_light, front_light, top_light, rim, low_light):
            if light is not None:
                light.hide_render = False
        set_visible({obj.name for obj in objects})

        # Turntable.
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
        'geometry_basis': '原照可见结构 + 未标定六视图（红色焊接式单耳环液压缸）',
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'shift': list(shift),
        'config_version': cfg['version'],
    }
    (out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                            encoding='utf8')


main()
