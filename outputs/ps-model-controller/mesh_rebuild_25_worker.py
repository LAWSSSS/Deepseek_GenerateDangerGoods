"""Blender worker: part-based rebuild of item 25, a blue welded hydraulic cylinder.

Evidence:
- one 1774x887 clean photograph and its matching rust derivative of a blue
  welded hydraulic cylinder with a large base-end eye, a bolted rod-end flange,
  a chrome piston rod, a rod-end eye, a top valve/manifold block and two bent
  steel pipes;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified).  It is used for topology and the unseen back, not
  for measurements.

The old TripoSR library mesh is a soft low-poly blob: the two eye holes are
filled, the flange bolts and pipes are fused, and the surface is a rough
photo-projected shell.  It is retained unchanged and is not used as the
candidate's base geometry.  The candidate is a parametric part rebuild with a
true barrel, real through-bored eye rings with steel bushings, a 16-bolt
flange, a chrome rod, a separate valve block, hex fittings, bent pipes, grease
nipples and blue/steel/bronze PBR materials.  All candidates remain
manual-only and pending human review; nothing replaces the library mesh or
enters automatic placement.

Coordinates: Z up, cylinder axis along Y, base eye at -Y, rod end at +Y.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
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
TILE_NAME = {value: key for key, value in TILE.items()}

objects = []
tiles = {}
metal_objects = []
nonmetal_objects = []
MATERIAL = None
CAM = None
METAL_TILES = {'chrome', 'steel', 'brass', 'dark_steel'}
DIELECTRIC_TILES = {'blue', 'blue_dark', 'rubber', 'weld', 'rust', 'blue_pipe',
                    'spare10', 'spare11', 'spare12', 'spare13', 'spare14', 'spare15'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed blue paint, chrome, steel and bronze PBR atlas')
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


def sphere(name, radius, location, tile, *, segments=24, rings=12):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings,
                                         radius=radius, location=location)
    return finish(bpy.context.object, name, tile, bevel=0, smooth=True)


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
    """Extruded thick plate with a round eye end and a flat neck end.

    mirrored=False: round eye on the -Y side (base clevis), flat neck at +Y.
    mirrored=True: round eye on the +Y side (rod clevis), flat neck at -Y.
    """
    pts = []

    def add(point):
        if not pts or (abs(point[0] - pts[-1][0]) > 1e-9 or
                       abs(point[1] - pts[-1][1]) > 1e-9):
            pts.append(point)

    if not mirrored:
        # round eye on -Y, flat neck on +Y: left semicircle, bottom, right edge, top
        for k in range(segs + 1):
            a = math.pi / 2 + math.pi * k / segs
            add((yc + radius * math.cos(a), radius * math.sin(a)))
        add((y_flat, -half_height))
        add((y_flat, half_height))
    else:
        # flat neck on -Y, round eye on +Y: top, right semicircle, bottom
        add((y_flat, half_height))
        add((yc, half_height))
        for k in range(1, segs + 1):
            a = math.pi / 2 - math.pi * k / segs
            add((yc + radius * math.cos(a), radius * math.sin(a)))
        add((yc, -half_height))
        add((y_flat, -half_height))
    # the closing edge is implicit; add() drops duplicate start/end points
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
# Pipes
# ---------------------------------------------------------------------------
def rounded_path(points, corner_radius, steps=10, sample=0.006):
    pts = [Vector(p) for p in points]
    dense = [pts[0]]
    previous = pts[0]
    for i in range(1, len(pts) - 1):
        p, a, b = pts[i], pts[i - 1], pts[i + 1]
        da, db = a - p, b - p
        la, lb = da.length, db.length
        r = min(corner_radius, .45 * la, .45 * lb)
        start = p + da.normalized() * r
        end = p + db.normalized() * r
        length = (start - previous).length
        count = max(1, int(math.ceil(length / sample)))
        for j in range(1, count + 1):
            dense.append(previous + (start - previous) * (j / count))
        for j in range(1, steps + 1):
            t = j / steps
            dense.append((1 - t) ** 2 * start + 2 * (1 - t) * t * p + t * t * end)
        previous = end
    last = pts[-1]
    length = (last - previous).length
    count = max(1, int(math.ceil(length / sample)))
    for j in range(1, count + 1):
        dense.append(previous + (last - previous) * (j / count))
    # drop consecutive duplicates
    cleaned = [dense[0]]
    for point in dense[1:]:
        if (point - cleaned[-1]).length > 1e-7:
            cleaned.append(point)
    return cleaned


def tube_path(name, points, radius, tile, *, corner_radius=.025, radial=16,
              bevel=0.0):
    path = rounded_path(points, corner_radius)
    tangents = []
    for i in range(len(path)):
        if i == 0:
            t = path[1] - path[0]
        elif i == len(path) - 1:
            t = path[-1] - path[-2]
        else:
            t = path[i + 1] - path[i - 1]
        tangents.append(t.normalized())
    # initial normal: any vector perpendicular to the first tangent
    up = Vector((0, 0, 1))
    if abs(tangents[0].dot(up)) > .95:
        up = Vector((1, 0, 0))
    normal = tangents[0].cross(up).normalized()
    normals = [normal]
    for i in range(1, len(path)):
        q = tangents[i - 1].rotation_difference(tangents[i])
        normal = (q @ normals[-1]).normalized()
        normals.append(normal)
    verts, faces = [], []
    ring_ids = []
    for point, tan, n in zip(path, tangents, normals):
        b = tan.cross(n).normalized()
        ids = []
        for k in range(radial):
            a = math.tau * k / radial
            p = point + radius * (math.cos(a) * n + math.sin(a) * b)
            ids.append(len(verts))
            verts.append(tuple(p))
        ring_ids.append(ids)
    for i in range(len(ring_ids) - 1):
        a, b = ring_ids[i], ring_ids[i + 1]
        for k in range(radial):
            k2 = (k + 1) % radial
            faces.append((a[k], a[k2], b[k2], b[k]))
    c0 = len(verts)
    verts.append(tuple(path[0]))
    c1 = len(verts)
    verts.append(tuple(path[-1]))
    for k in range(radial):
        k2 = (k + 1) % radial
        faces.append((c0, ring_ids[0][k2], ring_ids[0][k]))
        faces.append((c1, ring_ids[-1][k], ring_ids[-1][k2]))
    obj = mesh_object(name, verts, faces)
    if bevel:
        apply_bevel(obj, bevel, segments=2, angle=35)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return assign(obj, tile)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def build_eye_plate(which):
    if which == 'base':
        c = D['base_clevis']
        plate = plate_stadium_x('base_clevis', c['eye_center_y'],
                                c['eye_outer_radius'], c['neck_end_y'],
                                c['plate_half_height'], c['plate_x0'],
                                c['plate_x1'], TILE['blue'], mirrored=False)
        eye_y = c['eye_center_y']
        bore = c['eye_bore_radius']
        boss_r = c['boss_outer_radius']
        depth = c['boss_depth']
        bushing_outer = c['bushing_outer_radius']
        bushing_inner = c['bushing_inner_radius']
        half_x = c['bushing_half_x']
        x0, x1 = c['plate_x0'], c['plate_x1']
        grease = c['grease_center']
    else:
        c = D['rod_clevis']
        plate = plate_stadium_x('rod_clevis', c['eye_center_y'],
                                c['eye_outer_radius'], c['neck_end_y'],
                                c['plate_half_height'], c['plate_x0'],
                                c['plate_x1'], TILE['blue'], mirrored=True)
        eye_y = c['eye_center_y']
        bore = c['eye_bore_radius']
        boss_r = c['boss_outer_radius']
        depth = c['boss_depth']
        bushing_outer = c['bushing_outer_radius']
        bushing_inner = c['bushing_inner_radius']
        half_x = c['bushing_half_x']
        x0, x1 = c['plate_x0'], c['plate_x1']
        grease = c['grease_center']
    # real through bore along X
    cutter = along(f'cut_{which}_eye', (x0 - .02, eye_y, 0),
                   (x1 + .02, eye_y, 0), bore, TILE['rubber'], segs=72,
                   bevel=0, cutter=True)
    apply_boolean(plate, [cutter], f'Bore {which} eye')
    apply_bevel(plate, .0016, segments=2, angle=35)
    for poly in plate.data.polygons:
        poly.use_smooth = False
    # raised machined boss rings on both faces
    annulus_axis(f'{which}_boss_neg', 'X', (0, eye_y, 0),
                 x0 - depth, x0, bore, boss_r, TILE['blue'], segs=96,
                 bevel=.0012)
    annulus_axis(f'{which}_boss_pos', 'X', (0, eye_y, 0),
                 x1, x1 + depth, bore, boss_r, TILE['blue'], segs=96,
                 bevel=.0012)
    # steel bushing tube through the bore
    bushing = annulus_axis(f'{which}_bushing', 'X', (0, eye_y, 0),
                           -half_x, half_x, bushing_inner, bushing_outer,
                           TILE['steel'], segs=96, bevel=.0008)
    smooth_side(bushing, (1, 0, 0))
    # grease fitting on the top edge
    gx, gy, gz = grease
    hex_prism_axis(f'{which}_grease_base', .0085, .007,
                   (gx, gy, gz + .0035), (0, 0, 1), TILE['steel'],
                   bevel=.0006)
    along(f'{which}_grease_stem', (gx, gy, gz + .007),
          (gx, gy, gz + .013), .0032, TILE['steel'], segs=24, bevel=.0003)
    sphere(f'{which}_grease_tip', .0044, (gx, gy, gz + .0145), TILE['steel'],
           segments=20, rings=10)
    return plate


def build_barrel():
    r = D['rear_cap']
    cylinder = along('rear_cap', (0, r['y0'], 0), (0, r['y1'], 0),
                     r['radius'], TILE['blue'], segs=128, bevel=.0022)
    smooth_side(cylinder, (0, 1, 0))
    b = D['barrel']
    barrel = along('barrel', (0, b['y0'], 0), (0, b['y1'], 0), b['radius'],
                   TILE['blue'], segs=160, bevel=.0018)
    smooth_side(barrel, (0, 1, 0))
    for name, y in (('rear', b['seam_y0']), ('front', b['seam_y1'])):
        torus(f'barrel_seam_{name}', b['radius'] + .0012, .0016,
              (0, y, 0), 'Y', TILE['weld'], major_segments=160,
              minor_segments=8)
    torus('rear_cap_seam', r['radius'] + .001, .0020,
          (0, r['y1'] - .002, 0), 'Y', TILE['weld'],
          major_segments=160, minor_segments=8)
    torus('rear_cap_band', r['radius'] - .001, .0030,
          (0, r['y1'] - .010, 0), 'Y', TILE['blue_dark'],
          major_segments=160, minor_segments=8)


def build_flange():
    f = D['flange']
    flange = annulus_axis('rod_flange', 'Y', (0, 0, 0), f['y0'], f['y1'],
                          f['r_in'], f['r_out'], TILE['blue'], segs=160,
                          bevel=.0022)
    smooth_side(flange, (0, 1, 0))
    torus('flange_outer_rim', f['r_out'] - .004, .0025,
          (0, f['y1'] + .001, 0), 'Y', TILE['blue_dark'],
          major_segments=160, minor_segments=8)
    torus('flange_barrel_collar', f['r_out'] - .012, .0035,
          (0, f['y0'] + .001, 0), 'Y', TILE['weld'],
          major_segments=160, minor_segments=8)
    count = f['bolt_count']
    phase = math.radians(f['bolt_phase_deg'])
    for i in range(count):
        a = phase + math.tau * i / count
        x = f['bolt_circle_radius'] * math.cos(a)
        z = f['bolt_circle_radius'] * math.sin(a)
        along(f'flange_bolt_{i:02d}', (x, f['bolt_head_y0'], z),
              (x, f['bolt_head_y1'], z), f['bolt_head_radius'], TILE['blue'],
              segs=32, bevel=.0008)
        # dark hex socket on the rod-side face
        hex_prism_axis(f'flange_socket_{i:02d}', f['socket_radius'], .0035,
                       (x, f['bolt_head_y1'] - .0005, z), (0, 1, 0),
                       TILE['dark_steel'], bevel=.00025)
    # bronze wear ring and dark segmented wiper on the rod side
    g = D['gland_ring']
    annulus_axis('gland_ring', 'Y', (0, 0, 0), g['y0'], g['y1'],
                 g['r_in'], g['r_out'], TILE['brass'], segs=128,
                 bevel=.0008)
    w = D['wiper']
    wiper = annulus_axis('wiper_ring', 'Y', (0, 0, 0), w['y0'], w['y1'],
                         w['r_in'], w['r_out'], TILE['dark_steel'], segs=128,
                         bevel=.0006)
    teeth = w['teeth']
    for i in range(teeth):
        a = math.tau * i / teeth
        radius = (w['r_out'] + w['r_in']) / 2
        box(f'wiper_tooth_{i:02d}',
            (radius * math.cos(a), (w['y1'] + w['tooth_y1']) / 2,
             radius * math.sin(a)),
            (.0075, w['tooth_y1'] - w['y1'], .0085), TILE['dark_steel'],
            bevel=.0004, rotation=(0, -a, 0))


def build_rod():
    r = D['rod']
    rod = along('rod', (0, r['y0'], 0), (0, r['y1'], 0), r['radius'],
                TILE['chrome'], segs=128, bevel=.0016)
    smooth_side(rod, (0, 1, 0))
    c = D['rod_collar']
    collar = along('rod_collar', (0, c['y0'], 0), (0, c['y1'], 0),
                   c['radius'], TILE['blue'], segs=96, bevel=.0022)
    smooth_side(collar, (0, 1, 0))
    torus('rod_collar_seal', c['radius'] + .0008, .0013,
          (0, c['seal_y'], 0), 'Y', TILE['rubber'],
          major_segments=96, minor_segments=8)


def build_valve_block():
    v = D['valve_block']
    cx, cy, cz = v['center']
    sx, sy, sz = v['size']
    px, py, pz = v['pad_size']
    box('valve_pad', (cx, cy, v['pad_center_z']), (px, py, pz),
        TILE['blue'], bevel=.0020)
    block = box('valve_block', (cx, cy, cz), (sx, sy, sz), TILE['blue'],
                bevel=.0026)
    top = cz + sz / 2
    for i, (x, y) in enumerate(v['top_bolt_xy'], 1):
        along(f'valve_bolt_{i}', (x, y, top), (x, y, top + v['top_bolt_height']),
              v['top_bolt_radius'], TILE['blue'], segs=24, bevel=.0007)
        hex_prism_axis(f'valve_socket_{i}', v['top_socket_radius'], .0032,
                       (x, y, top + v['top_bolt_height'] - .0004), (0, 0, 1),
                       TILE['dark_steel'], bevel=.00025)
    hex_prism_axis('valve_center_plug', v['center_plug_radius'],
                   v['center_plug_height'], (cx, cy, top + v['center_plug_height'] / 2),
                   (0, 0, 1), TILE['blue'], bevel=.0007)
    hex_prism_axis('valve_center_socket', .010, .0032,
                   (cx, cy, top + v['center_plug_height'] - .0004), (0, 0, 1),
                   TILE['dark_steel'], bevel=.00025)
    front_x = cx + sx / 2
    for i, (y, z) in enumerate(v['front_plugs'], 1):
        annulus_axis(f'valve_front_rim_{i}', 'X', (0, y, z),
                     front_x, front_x + .004, .006, v['front_plug_rim_radius'],
                     TILE['steel'], segs=48, bevel=.0005)
        hex_prism_axis(f'valve_front_socket_{i}', v['front_plug_socket_radius'],
                       .0035, (front_x + .0042, y, z), (1, 0, 0),
                       TILE['dark_steel'], bevel=.00025)
    # large hex fittings on both ends; pipes enter the small nipples
    for name, y in (('left', v['left_fitting_y']), ('right', v['right_fitting_y'])):
        direction = -1 if name == 'left' else 1
        hex_prism_axis(f'valve_{name}_fitting', v['fitting_hex_radius'],
                       v['fitting_length'],
                       (cx, y + direction * v['fitting_length'] / 2, cz),
                       (0, 1, 0), TILE['blue'], bevel=.0007)
        along(f'valve_{name}_nipple',
              (cx, y + direction * v['fitting_length'], cz),
              (cx, y + direction * (v['fitting_length'] + .014), cz),
              .0105, TILE['steel'], segs=40, bevel=.0005)


def build_pipes():
    p = D['pipes']
    tube_path('pipe_left', p['left_path'], p['radius'], TILE['blue_pipe'],
              corner_radius=p['corner_radius'], radial=20)
    tube_path('pipe_right', p['right_path'], p['radius'], TILE['blue_pipe'],
              corner_radius=p['corner_radius'], radial=20)
    for i, y in enumerate(p['barrel_entry_y']):
        torus(f'pipe_weld_{i}', .0115, .0018, (p['left_path'][0][0], y, .086),
              'Z', TILE['weld'], major_segments=40, minor_segments=8)


def build_small_details():
    b = D['barrel_plug']
    x, y, z = b['center']
    hex_prism_axis('barrel_plug_hex', b['hex_radius'], b['height'],
                   (x, y, z + b['height'] / 2), (0, 0, 1), TILE['steel'],
                   bevel=.0006)
    along('barrel_plug_stem', (x, y, z + b['height']),
          (x, y, z + b['height'] + .009), .0034, TILE['steel'], segs=24,
          bevel=.0003)
    sphere('barrel_plug_tip', .0046,
           (x, y, z + b['height'] + .0105), TILE['steel'],
           segments=20, rings=10)


def build_all():
    build_eye_plate('base')
    build_eye_plate('rod')
    build_barrel()
    build_flange()
    build_rod()
    build_valve_block()
    build_pipes()
    build_small_details()


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
    """Verify the real eye through-holes, bushings, flange annulus and bolts."""
    checks = {'eye_holes': {}, 'bushings': {}, 'bosses': {}, 'flange': {}}
    for which in ('base', 'rod'):
        c = D['base_clevis'] if which == 'base' else D['rod_clevis']
        plate = make_bvh(bpy.data.objects[f'{which}_clevis'])
        eye_y = c['eye_center_y']
        rays = []
        for side in (-1, 1):
            origin = Vector((side * .20, eye_y, 0))
            direction = Vector((-side, 0, 0))
            hit = plate.ray_cast(origin, direction, 1.0)
            rays.append({'side': side, 'hit': hit[0] is not None})
        checks['eye_holes'][which] = {
            'through': all(not r['hit'] for r in rays), 'rays': rays}
        bushing = make_bvh(bpy.data.objects[f'{which}_bushing'])
        bushing_rays = []
        for side in (-1, 1):
            origin = Vector((side * .20, eye_y + (c['bushing_outer_radius'] +
                                                  c['bushing_inner_radius']) / 2, 0))
            hit = bushing.ray_cast(origin, Vector((-side, 0, 0)), 1.0)
            bushing_rays.append({'side': side, 'hit': hit[0] is not None})
        checks['bushings'][which] = {
            'present': all(r['hit'] for r in bushing_rays),
            'rays': bushing_rays}
        boss = make_bvh(bpy.data.objects[f'{which}_boss_pos'])
        origin = Vector((.20, eye_y + c['boss_outer_radius'] * .8, 0))
        hit = boss.ray_cast(origin, Vector((-1, 0, 0)), 1.0)
        checks['bosses'][which] = {'hit': hit[0] is not None}
    f = D['flange']
    flange = make_bvh(bpy.data.objects['rod_flange'])
    # clearance between rod and flange bore should be open
    clearance = flange.ray_cast(Vector((0, -.30, .0285)), Vector((0, 1, 0)), 1.0)
    solid = flange.ray_cast(Vector((0, -.30, .070)), Vector((0, 1, 0)), 1.0)
    checks['flange'] = {
        'central_clearance_open': clearance[0] is None,
        'solid_ring_hit': solid[0] is not None,
        'bolt_count': len([n for n in bpy.data.objects.keys()
                           if n.startswith('flange_bolt_')]),
        'socket_count': len([n for n in bpy.data.objects.keys()
                             if n.startswith('flange_socket_')]),
    }
    checks['valve'] = {
        'top_bolts': len([n for n in bpy.data.objects.keys()
                          if n.startswith('valve_bolt_')]),
        'front_plugs': len([n for n in bpy.data.objects.keys()
                            if n.startswith('valve_front_socket_')]),
        'pipes': len([n for n in bpy.data.objects.keys()
                      if n.startswith('pipe_')]),
    }
    # barrel roundness
    barrel = bpy.data.objects['barrel']
    radii = [math.hypot(v.co.x, v.co.z) for v in barrel.data.vertices
             if .02 < abs(v.co.y) and abs(v.co.y) < .44 and v.co.y < .08]
    checks['barrel'] = {
        'radial_deviation_m': round(float(max(abs(r - D['barrel']['radius'])
                                              for r in radii)), 6) if radii else None,
    }
    checks['passed'] = bool(
        all(v['through'] for v in checks['eye_holes'].values())
        and all(v['present'] for v in checks['bushings'].values())
        and checks['flange']['central_clearance_open']
        and checks['flange']['solid_ring_hit']
        and checks['flange']['bolt_count'] == D['flange']['bolt_count']
        and checks['valve']['top_bolts'] == 4
        and checks['valve']['front_plugs'] == 2)
    return checks


def main():
    global MATERIAL, CAM
    scene = legacy.setup_scene('CPU', 512)
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, 'denoiser'):
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    MATERIAL = atlas_material()
    CAM = init_camera(scene)
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Strength'].default_value = .40
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 90
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 3.4
        fill.data.size_y = 3.4
        fill.location = (-3.0, 2.6, 2.2)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 180, 3.2),
            ('front reflection', (3.0, 1.4, 1.0), 90, 1.5),
            ('top fill', (0.0, 0.0, 4.0), 65, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 55, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 25, 1.4)):
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
        # End-on views need a tighter framing than the long side views so the
        # eye/flange detail is readable in the six-view board.
        view_scale = span * (0.52 if name in ('left', 'right') else 1.12)
        look(scene, direction, center, span, ortho_scale=view_scale)
        draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    source = request['source_camera']
    set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
               shift_x=source.get('shift_x', 0), shift_y=source.get('shift_y', 0))
    draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])

    if not request.get('draft'):
        target = {
            'base-eye': object_center('base_bushing'),
            'base-grease': object_center('base_grease_tip'),
            'valve-block': object_center('valve_block'),
            'valve-ports': (object_center('valve_front_socket_1') +
                            object_center('valve_front_socket_2')) / 2,
            'pipe-loop': object_center('pipe_left'),
            'rod-flange': object_center('gland_ring'),
            'rod-wiper': object_center('wiper_ring'),
            'rod-clevis': object_center('rod_bushing'),
            'barrel-seam': object_center('barrel_seam_front'),
            'valve-fitting': object_center('valve_right_fitting'),
        }
        detail_specs = [
            ('base-eye', (1.0, -.25, .18), target['base-eye'], .205),
            ('base-grease', (.55, -.55, .75), target['base-grease'], .060),
            ('valve-block', (.75, -.55, .55), target['valve-block'], .175),
            ('valve-ports', (1.0, -.15, .15), target['valve-ports'], .150),
            ('pipe-loop', (.65, -.55, .65), target['pipe-loop'], .240),
            ('rod-flange', (.45, .90, .25), target['rod-flange'], .205),
            ('rod-wiper', (.25, 1.0, .12), target['rod-wiper'], .095),
            ('rod-clevis', (1.0, .25, .18), target['rod-clevis'], .190),
            ('barrel-seam', (.85, -.45, .35), target['barrel-seam'], .160),
            ('valve-fitting', (.80, .30, .45), target['valve-fitting'], .085),
            ('overall-3q', (1.0, -1.0, .55), center, span * 1.14),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        # Isolated subassemblies and openings.
        set_visible({'base_clevis', 'base_boss_neg', 'base_boss_pos',
                     'base_bushing', 'base_grease_base', 'base_grease_stem',
                     'base_grease_tip'})
        look(scene, (1, -.25, .15), object_center('base_bushing'), span,
             ortho_scale=.225)
        draw(scene, render_dir / 'base-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'rod_clevis', 'rod_boss_neg', 'rod_boss_pos',
                     'rod_bushing', 'rod_grease_base', 'rod_grease_stem',
                     'rod_grease_tip', 'rod', 'rod_collar', 'rod_collar_seal'})
        look(scene, (1, .25, .15), object_center('rod_bushing'), span,
             ortho_scale=.215)
        draw(scene, render_dir / 'rod-clevis-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'rod_flange', 'gland_ring', 'wiper_ring'}
                    | {n for n in bpy.data.objects.keys()
                       if n.startswith(('flange_bolt_', 'flange_socket_',
                                        'wiper_tooth_'))})
        look(scene, (1, .55, .18), object_center('rod_flange'), span,
             ortho_scale=.215)
        draw(scene, render_dir / 'flange-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'valve_block', 'valve_pad', 'valve_center_plug',
                     'valve_center_socket', 'valve_left_fitting',
                     'valve_right_fitting', 'valve_left_nipple',
                     'valve_right_nipple'}
                    | {n for n in bpy.data.objects.keys()
                       if n.startswith(('valve_bolt_', 'valve_socket_',
                                        'valve_front_'))})
        look(scene, (.8, -.8, .8), object_center('valve_block'), span,
             ortho_scale=.225)
        draw(scene, render_dir / 'valve-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'pipe_left', 'pipe_right', 'pipe_weld_0', 'pipe_weld_1'})
        look(scene, (.7, -.7, .7), center, span, ortho_scale=.55)
        draw(scene, render_dir / 'pipes-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'barrel', 'rear_cap', 'barrel_seam_front',
                     'barrel_seam_rear', 'rear_cap_seam'})
        look(scene, (.3, -.6, .75), center, span, ortho_scale=.55)
        draw(scene, render_dir / 'barrel-only.png', rr['detail_resolution'],
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
        metal_target = object_center('gland_ring')
        look(scene, (.85, -.45, .35), metal_target, span, ortho_scale=.245)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (.85, -.45, .35), metal_target, span, ortho_scale=.245)
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
        'geometry_basis': '原照可见结构 + 未标定六视图（蓝色焊接式双耳环液压缸）',
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'shift': list(shift),
        'config_version': cfg['version'],
    }
    (out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                            encoding='utf8')


main()
