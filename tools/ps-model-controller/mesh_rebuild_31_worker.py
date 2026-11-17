"""Blender worker: part-based rebuild of item 31, a yellow cutaway hydraulic cylinder.

Evidence:
- one 1475x1066 clean photograph and its matching rust derivative of a yellow
  welded hydraulic cylinder with rear/rod eyes, a single longitudinal cutaway
  window, a chrome piston rod, a rear piston seal stack, a front rod gland seal
  stack, a rear vertical filler/breather cap and a tee-shaped twin port block;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified).  Its front/back panels agree with the photo on the
  window side; its left/right/top/bottom panels disagree about the eye-bore
  orientation and are not used as a consistent turntable.  The photo plus the
  front/back panels are the basis for both eye bores being along X.

The old TripoSR library mesh is a soft low-poly blob: the two eye holes are
filled, the cutaway window is fused shut and the seal stacks are lost.  It is
retained unchanged and is not used as the candidate's base geometry.  The
candidate is a parametric part rebuild with a real thick-walled barrel and a
rounded longitudinal window, a dark honed-bore surface, true through-bored eyes
with steel bushings (and a rubber rear sleeve), a chrome rod, piston rings, a
gland seal stack, a rear cap/eye cone, a knurled rear port cap and a tee port
with real through bores.  All candidates remain manual-only and pending human
review; nothing replaces the library mesh or enters automatic placement.

Coordinates: Z up, cylinder axis along Y, rear eye at -Y, rod end at +Y, and
the cutaway window faces +X.
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
METAL_TILES = {'chrome', 'steel', 'bronze', 'dark_steel'}
DIELECTRIC_TILES = {'yellow', 'yellow_dark', 'rubber', 'weld', 'rust',
                    'spare10', 'spare11', 'spare12', 'spare13', 'spare14', 'spare15'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed yellow paint, chrome, steel, bronze and rubber PBR atlas')
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


def cone_axis_y(name, y0, y1, r0, r1, tile, *, segs=96, bevel=.0015):
    bpy.ops.mesh.primitive_cone_add(vertices=segs, radius1=r0, radius2=r1,
                                    depth=y1 - y0, location=(0, (y0 + y1) / 2, 0),
                                    rotation=(-math.pi / 2, 0, 0))
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel)


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
# Barrel with a rounded longitudinal window
# ---------------------------------------------------------------------------
def barrel_tube(name, y0, y1, radius, inner_radius, segs=160):
    verts = []
    for y in (y0, y1):
        for j in range(segs):
            t = math.tau * j / segs
            verts.append((radius * math.cos(t), y, radius * math.sin(t)))
    for y in (y0, y1):
        for j in range(segs):
            t = math.tau * j / segs
            verts.append((inner_radius * math.cos(t), y, inner_radius * math.sin(t)))
    o0 = list(range(0, segs))
    o1 = list(range(segs, 2 * segs))
    i0 = list(range(2 * segs, 3 * segs))
    i1 = list(range(3 * segs, 4 * segs))
    faces = []
    for j in range(segs):
        j2 = (j + 1) % segs
        faces.append((o0[j], o0[j2], o1[j2], o1[j]))
        faces.append((i0[j], i1[j], i1[j2], i0[j2]))
        faces.append((o0[j], i0[j], i0[j2], o0[j2]))
        faces.append((o1[j], o1[j2], i1[j2], i1[j]))
    return mesh_object(name, verts, faces)


def stadium_prism_x(name, yc, half_straight, rz, x0, x1, segs=64):
    points = []
    points.append((yc - half_straight, rz))
    points.append((yc + half_straight, rz))
    for k in range(1, segs + 1):
        a = math.pi / 2 - math.pi * k / segs
        points.append((yc + half_straight + rz * math.cos(a), rz * math.sin(a)))
    points.append((yc + half_straight, -rz))
    points.append((yc - half_straight, -rz))
    for k in range(1, segs + 1):
        a = -math.pi / 2 - math.pi * k / segs
        points.append((yc - half_straight + rz * math.cos(a), rz * math.sin(a)))
    cleaned = []
    for p in points:
        if not cleaned or (abs(p[0] - cleaned[-1][0]) > 1e-9 or
                           abs(p[1] - cleaned[-1][1]) > 1e-9):
            cleaned.append(p)
    if (abs(cleaned[0][0] - cleaned[-1][0]) < 1e-9 and
            abs(cleaned[0][1] - cleaned[-1][1]) < 1e-9):
        cleaned.pop()
    n = len(cleaned)
    verts = [(x0, y, z) for y, z in cleaned] + [(x1, y, z) for y, z in cleaned]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(n + i for i in range(n)))
    return mesh_object(name, verts, faces)


def assign_barrel_uvs(obj, radius, inner_radius, y0, y1):
    """Yellow outer/cut/cap faces, dark honed-bore inner surface."""
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='Atlas')
    layer = mesh.uv_layers.active.data
    margin = .012
    for poly in mesh.polygons:
        c = poly.center
        n = poly.normal
        dist = math.hypot(c.x, c.z)
        radial = 0.0 if dist < 1e-9 else (c.x * n.x + c.z * n.z) / dist
        if abs(n.y) > .60:
            tile = TILE['yellow']
            planar = True
            smooth = False
        elif dist > (radius + inner_radius) * .5:
            tile = TILE['yellow']
            planar = False
            smooth = True
        elif dist < inner_radius + .0025 and radial < -.15:
            tile = TILE['bore']
            planar = False
            smooth = True
        else:
            tile = TILE['yellow_dark']
            planar = True
            smooth = False
        poly.use_smooth = smooth
        xt0, yt0, xt1, yt1 = tile_bounds(tile)
        w, h = xt1 - xt0, yt1 - yt0
        for li in poly.loop_indices:
            v = mesh.vertices[mesh.loops[li].vertex_index].co
            if planar:
                uu = (v.y - y0) / (y1 - y0)
                vv = (math.hypot(v.x, v.z) - inner_radius) / (radius - inner_radius)
            else:
                ang = math.atan2(v.z, v.x) / math.tau + .5
                uu = ang - math.floor(ang)
                vv = (v.y - y0) / (y1 - y0)
            layer[li].uv = (xt0 + margin * w + uu * (w - 2 * margin * w),
                            yt0 + margin * h + vv * (h - 2 * margin * h))


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def build_rear_eye():
    c = D['rear_clevis']
    plate = plate_stadium_x('rear_clevis', c['eye_center_y'], c['eye_outer_radius'],
                            c['neck_end_y'], c['plate_half_height'],
                            c['plate_x0'], c['plate_x1'], TILE['yellow'],
                            mirrored=False, bevel=.0025)
    eye_y = c['eye_center_y']
    cutter = along('cut_rear_eye', (c['plate_x0'] - .02, eye_y, 0),
                   (c['plate_x1'] + .02, eye_y, 0), c['eye_bore_radius'],
                   TILE['rubber'], segs=96, bevel=0, cutter=True)
    apply_boolean(plate, [cutter], 'Bore rear eye')
    apply_bevel(plate, .0010, segments=1, angle=35)
    for poly in plate.data.polygons:
        poly.use_smooth = False
    assign(plate, TILE['yellow'], register=False)
    bushing = annulus_axis('rear_bushing', 'X', (0, eye_y, 0),
                           -c['bushing_half_x'], c['bushing_half_x'],
                           c['bushing_inner_radius'], c['bushing_outer_radius'],
                           TILE['steel'], segs=128, bevel=.0008)
    smooth_side(bushing, (1, 0, 0))
    rubber = annulus_axis('rear_rubber', 'X', (0, eye_y, 0),
                          -c['rubber_half_x'], c['rubber_half_x'],
                          c['rubber_inner_radius'], c['rubber_outer_radius'],
                          TILE['rubber'], segs=128, bevel=.0006)
    smooth_side(rubber, (1, 0, 0))
    # A shallow raised painted rim behind each bushing face.
    for side, a0, a1 in (('neg', c['plate_x0'] - .0015, c['plate_x0']),
                         ('pos', c['plate_x1'], c['plate_x1'] + .0015)):
        annulus_axis(f'rear_eye_rim_{side}', 'X', (0, eye_y, 0), a0, a1,
                     c['bushing_outer_radius'], c['bushing_outer_radius'] + .006,
                     TILE['yellow'], segs=128, bevel=.0004)


def build_rear_cap():
    r = D['rear_cap']
    cap = along('rear_cap', (0, r['y0'], 0), (0, r['y1'], 0), r['radius'],
                TILE['yellow'], segs=160, bevel=.0020)
    smooth_side(cap, (0, 1, 0))
    cone = cone_axis_y('rear_cone', r['cone_y0'], r['cone_y1'],
                       r['cone_r0'], r['cone_r1'], TILE['yellow'], segs=128)
    smooth_side(cone, (0, 1, 0))
    torus('rear_cap_weld', r['radius'] + .0008, .0019,
          (0, r['y1'] - .002, 0), 'Y', TILE['weld'], major_segments=160,
          minor_segments=8)
    torus('rear_cone_weld', (r['cone_r0'] + r['cone_r1']) * .5, .0016,
          (0, r['cone_y0'] + .004, 0), 'Y', TILE['weld'], major_segments=128,
          minor_segments=8)


def build_barrel():
    b = D['barrel']
    obj = barrel_tube('barrel', b['y0'], b['y1'], b['radius'], b['inner_radius'],
                      segs=192)
    cutter = stadium_prism_x('cut_window', b['window_center_y'],
                             b['window_half_straight'], b['window_radius_z'],
                             b['window_cutter_x0'], b['window_cutter_x1'],
                             segs=72)
    cutter.data.materials.append(MATERIAL)
    apply_boolean(obj, [cutter], 'Longitudinal cutaway window')
    apply_bevel(obj, .0009, segments=1, angle=35)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    obj.name = 'barrel'
    obj.data.name = 'barrel'
    obj.data.materials.clear()
    obj.data.materials.append(MATERIAL)
    assign_barrel_uvs(obj, b['radius'], b['inner_radius'], b['y0'], b['y1'])
    tiles[obj.name] = 'yellow+bore+edge'
    objects.append(obj)
    nonmetal_objects.append(obj.name)
    for name, y in (('rear', b['seam_y0']), ('front', b['seam_y1'])):
        torus(f'barrel_seam_{name}', b['radius'] + .0008, .0017,
              (0, y, 0), 'Y', TILE['weld'], major_segments=192, minor_segments=8)


def build_front_gland():
    g = D['front_gland']
    cap = annulus_axis('front_gland_cap', 'Y', (0, 0, 0), g['cap_y0'], g['cap_y1'],
                       g['cap_r_in'], g['cap_r_out'], TILE['yellow'], segs=160,
                       bevel=.0018)
    smooth_side(cap, (0, 1, 0))
    # Small machined caps sample one clean yellow texel so a procedural rust
    # blotch cannot turn the whole front ring into a brown band.
    assign(cap, TILE['yellow'], mode='constant', register=False)
    collar = annulus_axis('front_collar', 'Y', (0, 0, 0), g['collar_y0'],
                          g['collar_y1'], g['collar_r_in'], g['collar_r_out'],
                          TILE['yellow'], segs=160, bevel=.0016)
    smooth_side(collar, (0, 1, 0))
    assign(collar, TILE['yellow'], mode='constant', register=False)
    torus('front_collar_weld', g['collar_r_out'] - .0025, .0017,
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


def build_piston():
    p = D['piston']
    body = along('piston_body', (0, p['y0'], 0), (0, p['y1'], 0), p['radius'],
                 TILE['dark_steel'], segs=128, bevel=.0012)
    smooth_side(body, (0, 1, 0))
    for i, yc in enumerate(p['ring_ys']):
        ring = annulus_axis(f'piston_ring_{i:02d}', 'Y', (0, 0, 0),
                            yc - p['ring_thickness'] / 2, yc + p['ring_thickness'] / 2,
                            p['ring_r_in'], p['ring_r_out'], TILE['steel'],
                            segs=128, bevel=.0005)
        smooth_side(ring, (0, 1, 0))


def build_rod_and_front_eye():
    r = D['rod']
    rod = along('rod', (0, r['y0'], 0), (0, r['y1'], 0), r['radius'],
                TILE['chrome'], segs=128, bevel=.0012)
    smooth_side(rod, (0, 1, 0))
    c = D['rod_clevis']
    plate = plate_stadium_x('front_clevis', c['eye_center_y'], c['eye_outer_radius'],
                            c['neck_end_y'], c['plate_half_height'],
                            c['plate_x0'], c['plate_x1'], TILE['yellow'],
                            mirrored=True, bevel=.0025)
    eye_y = c['eye_center_y']
    cutter = along('cut_front_eye', (c['plate_x0'] - .02, eye_y, 0),
                   (c['plate_x1'] + .02, eye_y, 0), c['eye_bore_radius'],
                   TILE['rubber'], segs=96, bevel=0, cutter=True)
    apply_boolean(plate, [cutter], 'Bore front eye')
    apply_bevel(plate, .0010, segments=1, angle=35)
    for poly in plate.data.polygons:
        poly.use_smooth = False
    assign(plate, TILE['yellow'], register=False)
    bushing = annulus_axis('front_bushing', 'X', (0, eye_y, 0),
                           -c['bushing_half_x'], c['bushing_half_x'],
                           c['bushing_inner_radius'], c['bushing_outer_radius'],
                           TILE['steel'], segs=128, bevel=.0008)
    smooth_side(bushing, (1, 0, 0))
    for side, a0, a1 in (('neg', c['plate_x0'] - .0012, c['plate_x0']),
                         ('pos', c['plate_x1'], c['plate_x1'] + .0012)):
        annulus_axis(f'front_eye_rim_{side}', 'X', (0, eye_y, 0), a0, a1,
                     c['bushing_outer_radius'], c['bushing_outer_radius'] + .005,
                     TILE['yellow'], segs=128, bevel=.0004)
    b = D['rod_end_boss']
    boss = along('rod_end_boss', (0, b['y0'], 0), (0, b['y1'], 0), b['radius'],
                 TILE['yellow'], segs=96, bevel=.0020)
    smooth_side(boss, (0, 1, 0))
    torus('rod_boss_weld', b['radius'] + .0008, .0015, (0, b['seal_y'], 0), 'Y',
          TILE['weld'], major_segments=96, minor_segments=8)


def build_rear_port():
    p = D['rear_port']
    y = p['center_y']
    cylinder_z('rear_port_pad', p['pad_radius'], p['pad_z0'], p['pad_z1'],
               TILE['yellow'], x=0, y=y, segs=64, bevel=.0015)
    hex_prism_axis('rear_port_hex', p['hex_radius'], p['hex_z1'] - p['hex_z0'],
                   (0, y, (p['hex_z0'] + p['hex_z1']) / 2), (0, 0, 1),
                   TILE['yellow'], bevel=.0008)
    cylinder_z('rear_port_neck', p['neck_radius'], p['neck_z0'], p['neck_z1'],
               TILE['yellow'], x=0, y=y, segs=48, bevel=.0006)
    cap = cylinder_z('rear_port_cap', p['cap_radius'], p['cap_z0'], p['cap_z1'],
                     TILE['yellow'], x=0, y=y, segs=64, bevel=.0016)
    smooth_side(cap, (0, 0, 1))
    for i, z in enumerate(p['groove_zs']):
        torus(f'rear_port_groove_{i}', p['cap_radius'] - .0006, .0007,
              (0, y, z), 'Z', TILE['yellow_dark'], major_segments=64,
              minor_segments=6)
    torus('rear_port_cap_top_rim', p['cap_radius'] - .002, .0012,
          (0, y, p['cap_z1'] - .0015), 'Z', TILE['yellow_dark'],
          major_segments=64, minor_segments=6)
    torus('rear_port_weld', p['pad_radius'] + .001, .0016,
          (0, y, p['pad_z0'] + .002), 'Z', TILE['weld'],
          major_segments=64, minor_segments=6)


def build_tee_port():
    p = D['tee_port']
    y = p['center_y']
    cyl = cylinder_z('tee_pad', .017, p['stem_z0'] - .004, p['stem_z0'] + .006,
                     TILE['yellow'], x=0, y=y, segs=64, bevel=.0014)
    smooth_side(cyl, (0, 0, 1))
    stem = cylinder_z('tee_stem', p['stem_radius'], p['stem_z0'], p['stem_z1'],
                      TILE['yellow'], x=0, y=y, segs=48, bevel=.0010)
    smooth_side(stem, (0, 0, 1))
    torus('tee_stem_weld', p['stem_radius'] + .001, .0017,
          (0, y, p['stem_z0'] + .004), 'Z', TILE['weld'],
          major_segments=48, minor_segments=6)
    z = p['bar_z']
    z0, z1 = z - p['bar_radius'], z + p['bar_radius']
    # Stepped thick-walled crossbar: larger counterbores at both mouths,
    # smaller true through bore in the middle.
    annulus_axis('tee_bar_neg', 'Y', (0, 0, z), p['bar_y0'],
                 p['bar_y0'] + p['counterbore_len'], p['counterbore_r'],
                 p['bar_radius'], TILE['yellow'], segs=96, bevel=.0008)
    annulus_axis('tee_bar_center', 'Y', (0, 0, z),
                 p['bar_y0'] + p['counterbore_len'],
                 p['bar_y1'] - p['counterbore_len'], p['bore_r'],
                 p['bar_radius'], TILE['yellow'], segs=96, bevel=.0008)
    annulus_axis('tee_bar_pos', 'Y', (0, 0, z),
                 p['bar_y1'] - p['counterbore_len'], p['bar_y1'],
                 p['counterbore_r'], p['bar_radius'], TILE['yellow'], segs=96,
                 bevel=.0008)
    # Dark internal liner so the bore does not read as solid yellow paint.
    liner = annulus_axis('tee_port_liner', 'Y', (0, 0, z),
                         p['bar_y0'] - .001, p['bar_y1'] + .001,
                         p['liner_r_in'], p['liner_r_out'], TILE['dark_steel'],
                         segs=96, bevel=0)
    smooth_side(liner, (0, 1, 0))
    # End rims make the two counterbore mouths legible.
    for side, a0, a1 in (('neg', p['bar_y0'] - .0018, p['bar_y0']),
                         ('pos', p['bar_y1'], p['bar_y1'] + .0018)):
        annulus_axis(f'tee_port_rim_{side}', 'Y', (0, 0, z), a0, a1,
                     p['counterbore_r'], p['bar_radius'] + .0025,
                     TILE['yellow'], segs=96, bevel=.0004)


def build_all():
    build_rear_eye()
    build_rear_cap()
    build_barrel()
    build_front_gland()
    build_piston()
    build_rod_and_front_eye()
    build_rear_port()
    build_tee_port()


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
    """Verify eye through-holes, bushings, window, tee ports and ring counts."""
    checks = {'eye_holes': {}, 'bushings': {}, 'window': {}, 'ports': {},
              'counts': {}}
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
    rear_eye_y = D['rear_clevis']['eye_center_y']
    rear_mid = (D['rear_clevis']['rubber_outer_radius'] +
                D['rear_clevis']['rubber_inner_radius']) / 2
    rubber = make_bvh(bpy.data.objects['rear_rubber'])
    hit = rubber.ray_cast(Vector((.20, rear_eye_y + rear_mid, 0)),
                          Vector((-1, 0, 0)), 1.0)
    checks['bushings']['rear_rubber_present'] = hit[0] is not None

    b = D['barrel']
    barrel = make_bvh(bpy.data.objects['barrel'])
    y_mid = b['window_center_y']
    # Through the window the ray crosses the near wall opening but still hits
    # the far inner surface, so compare distances against the solid margin.
    window_front = barrel.ray_cast(Vector((.20, y_mid, .025)), Vector((-1, 0, 0)), 1.0)
    back_solid = barrel.ray_cast(Vector((-.20, y_mid, .025)), Vector((1, 0, 0)), 1.0)
    margin_solid = barrel.ray_cast(Vector((.20, y_mid, .045)), Vector((-1, 0, 0)), 1.0)
    front_distance = window_front[3] if window_front[0] is not None else None
    back_distance = back_solid[3] if back_solid[0] is not None else None
    margin_distance = margin_solid[3] if margin_solid[0] is not None else None
    checks['window'] = {
        'front_near_wall': {
            'hit': window_front[0] is not None,
            'distance_m': round(float(front_distance), 6) if front_distance else None,
        },
        'back_wall': {
            'hit': back_solid[0] is not None,
            'distance_m': round(float(back_distance), 6) if back_distance else None,
        },
        'solid_margin': {
            'hit': margin_solid[0] is not None,
            'distance_m': round(float(margin_distance), 6) if margin_distance else None,
        },
        'open_through_near_wall': bool(
            front_distance is not None and margin_distance is not None
            and front_distance > margin_distance + .02),
        'back_wall_intact': bool(back_distance is not None),
    }

    p = D['tee_port']
    bar_names = ('tee_bar_neg', 'tee_bar_center', 'tee_bar_pos')
    bar_bvh = [make_bvh(bpy.data.objects[n]) for n in bar_names]
    liner = make_bvh(bpy.data.objects['tee_port_liner'])
    port_rays = []
    for side, y0 in (('neg', p['bar_y0'] - .02), ('pos', p['bar_y1'] + .02)):
        direction = Vector((0, 1 if side == 'neg' else -1, 0))
        origin = Vector((0, y0, p['bar_z']))
        hits = [tree.ray_cast(origin, direction, 1.0)[0] is not None for tree in bar_bvh]
        liner_hit = liner.ray_cast(origin, direction, 1.0)[0] is not None
        port_rays.append({'side': side, 'bar_hit': any(hits),
                          'liner_hit': liner_hit,
                          'through': not any(hits) and not liner_hit})
    checks['ports'] = {
        'mouths': port_rays,
        'mouths_through': sum(1 for r in port_rays if r['through']),
        'counterbore_r': p['counterbore_r'],
        'bore_r': p['bore_r'],
    }

    checks['counts'] = {
        'piston_rings': len([n for n in bpy.data.objects.keys()
                             if n.startswith('piston_ring_')]),
        'gland_rings': len([n for n in bpy.data.objects.keys()
                            if n.startswith('gland_ring_')]),
        'tee_bars': len(bar_names),
        'rear_port_grooves': len([n for n in bpy.data.objects.keys()
                                  if n.startswith('rear_port_groove_')]),
    }
    barrel = bpy.data.objects['barrel']
    radii = [math.hypot(v.co.x, v.co.z) for v in barrel.data.vertices
             if math.hypot(v.co.x, v.co.z) > (b['radius'] + b['inner_radius']) * .55]
    checks['barrel'] = {
        'outer_radial_deviation_m': round(float(max(abs(r - b['radius'])
                                                    for r in radii)), 6) if radii else None,
    }
    checks['passed'] = bool(
        all(v['through'] for v in checks['eye_holes'].values())
        and all(v['present'] for v in checks['bushings'].values()
                if isinstance(v, dict) and 'present' in v)
        and checks['bushings'].get('rear_rubber_present')
        and checks['window']['open_through_near_wall']
        and checks['window']['back_wall_intact']
        and checks['ports']['mouths_through'] == 2
        and checks['counts']['piston_rings'] == 4
        and checks['counts']['gland_rings'] >= 5)
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
    background.inputs['Strength'].default_value = .55
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 110
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 3.4
        fill.data.size_y = 3.4
        fill.location = (-3.0, 2.6, 2.2)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 200, 3.2),
            ('front reflection', (3.0, 1.4, 1.0), 95, 1.5),
            ('window fill', (3.0, -1.2, 1.2), 130, 2.2),
            ('top fill', (0.0, 0.0, 4.0), 70, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 60, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 30, 1.4)):
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
        # End-on views need a tighter framing; the long side and top/bottom
        # views must keep the full length in frame.
        view_scale = span * (0.42 if name in ('left', 'right') else 1.10)
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
            'rear-port': object_center('rear_port_cap'),
            'front-eye': object_center('front_bushing'),
            'gland-stack': object_center('gland_ring_02'),
            'piston-rings': object_center('piston_ring_02'),
            'window': object_center('barrel'),
            'tee-port': object_center('tee_bar_center'),
            'barrel-seam': object_center('barrel_seam_front'),
            'rear-cap': object_center('rear_cap'),
            'rod': object_center('rod'),
        }
        detail_specs = [
            ('rear-eye', (1.0, -.35, .20), target['rear-eye'], .195),
            ('rear-port', (.55, -.55, .80), target['rear-port'], .090),
            ('front-eye', (1.0, .25, .18), target['front-eye'], .170),
            ('gland-stack', (.75, .55, .30), target['gland-stack'], .115),
            ('piston-rings', (.90, -.35, .25), target['piston-rings'], .135),
            ('window', (.70, -.55, .38), target['window'], .360),
            ('tee-port', (.65, -.70, .55), target['tee-port'], .130),
            ('barrel-seam', (.85, .45, .30), target['barrel-seam'], .160),
            ('rear-cap', (.55, -.80, .30), target['rear-cap'], .130),
            ('rod', (.65, .60, .30), target['rod'], .200),
            ('overall-3q', (1.0, -1.0, .55), center, span * 1.14),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        set_visible({'rear_clevis', 'rear_bushing', 'rear_rubber',
                     'rear_eye_rim_neg', 'rear_eye_rim_pos'})
        look(scene, (1, -.35, .15), object_center('rear_bushing'), span,
             ortho_scale=.205)
        draw(scene, render_dir / 'rear-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'front_clevis', 'front_bushing', 'front_eye_rim_neg',
                     'front_eye_rim_pos', 'rod', 'rod_end_boss', 'rod_boss_weld'})
        look(scene, (1, .30, .16), object_center('front_bushing'), span,
             ortho_scale=.185)
        draw(scene, render_dir / 'front-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'rod', 'piston_body'}
                    | {n for n in bpy.data.objects.keys()
                       if n.startswith('piston_ring_')})
        look(scene, (.9, -.30, .25), object_center('piston_ring_02'), span,
             ortho_scale=.165)
        draw(scene, render_dir / 'piston-rod-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'front_gland_cap', 'front_collar', 'front_wiper',
                     'front_collar_weld'}
                    | {n for n in bpy.data.objects.keys()
                       if n.startswith('gland_ring_')})
        look(scene, (.80, .55, .28), object_center('gland_ring_02'), span,
             ortho_scale=.145)
        draw(scene, render_dir / 'gland-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'tee_pad', 'tee_stem', 'tee_stem_weld', 'tee_bar_neg',
                     'tee_bar_center', 'tee_bar_pos', 'tee_port_liner',
                     'tee_port_rim_neg', 'tee_port_rim_pos'})
        look(scene, (.65, -.75, .55), object_center('tee_bar_center'), span,
             ortho_scale=.135)
        draw(scene, render_dir / 'tee-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'barrel', 'barrel_seam_rear', 'barrel_seam_front',
                     'rear_cap', 'rear_cone', 'rear_cap_weld', 'rear_cone_weld',
                     'front_gland_cap', 'front_collar', 'front_collar_weld'})
        look(scene, (.70, -.55, .38), center, span, ortho_scale=.520)
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
        for name in ('soft light', 'front reflection', 'window fill', 'top fill',
                     'rim', 'low fill'):
            light = bpy.data.objects.get(name)
            if light is not None:
                light.hide_render = True
        metal_target = object_center('gland_ring_02')
        look(scene, (.85, -.45, .35), metal_target, span, ortho_scale=.175)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (.85, -.45, .35), metal_target, span, ortho_scale=.175)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for name in ('soft light', 'front reflection', 'window fill', 'top fill',
                     'rim', 'low fill'):
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
        'geometry_basis': '原照可见结构 + 未标定六视图正面/背面拓扑（黄色剖视双耳环液压缸）',
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'shift': list(shift),
        'config_version': cfg['version'],
        'mixed_tile_parts': {'barrel': '外壁/切边为黄漆，内孔为深色 bore；按非金属主体登记'},
    }
    (out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                            encoding='utf8')


main()
