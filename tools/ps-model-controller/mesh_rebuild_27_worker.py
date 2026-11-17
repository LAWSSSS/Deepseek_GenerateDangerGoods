"""Blender worker: part-based rebuild of item 27, a black/blue motorcycle rear shock.

Evidence:
- one 1254x1254 clean photograph and its matching rust derivative of a
  YU-AN-style rear shock absorber with a blue coil spring, black anodized
  damper body, braided stainless hose and a piggyback reservoir;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified).  Its panels disagree about the top clevis and
  hose side, so the photograph wins for the visible front and the sheet is
  used for the unseen back/topology only.

The old TripoSR library mesh is a soft shell: the spring coils merge into the
body at the back, the reservoir is lumpy and the small fittings are lost.  It
is retained unchanged and is not used as the candidate's base geometry.  The
candidate is a parametric part rebuild with a real helical spring, true
through-bored eye/clevis, steel bushings, chrome shaft, braided hose,
knurled reservoir and PBR materials.  All candidates remain manual-only and
pending review; nothing replaces the library mesh or enters automatic
placement.

Coordinates: Z up, main axis along Z, top eye at +Z, bottom clevis at -Z,
front camera at +X, hose/reservoir on +Y.
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
METAL_TILES = {'black_ano', 'blue_ano', 'chrome', 'steel', 'dark_steel',
               'bronze', 'silver_band', 'braid'}
DIELECTRIC_TILES = {'blue_spring', 'rubber', 'rust', 'spare11', 'spare12',
                    'spare13', 'spare14', 'spare15'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed black anodized, blue paint, '
                                      'chrome, steel and reservoir PBR atlas')
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
    normal_map.inputs['Strength'].default_value = .34
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    return material


def decal_material(name, path, *, metallic, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    image = bpy.data.images.load(path, check_existing=True)
    image.pack()
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = image
    tex.interpolation = 'Linear'
    links.new(tex.outputs['Color'], principled.inputs['Base Color'])
    principled.inputs['Metallic'].default_value = metallic
    principled.inputs['Roughness'].default_value = roughness
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


def cone_z(name, r0, r1, z0, z1, tile, *, x=0.0, y=0.0, segs=96, bevel=0.0):
    bpy.ops.mesh.primitive_cone_add(vertices=segs, radius1=r0, radius2=r1,
                                    depth=z1 - z0, location=(x, y, (z0 + z1) / 2))
    return finish(bpy.context.object, name, tile, bevel=bevel)


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
    verts = []
    rings = []
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


def plate_stadium_xz(name, eye_z, eye_r, bottom_z, half_x, y0, y1, tile,
                     *, segs=56, bevel=.0009):
    """Plate in the XZ plane (thin along Y) with a round eye at the top."""
    pts = []
    for k in range(segs + 1):
        a = math.pi * k / segs
        pts.append((eye_r * math.cos(a), eye_z + eye_r * math.sin(a)))
    pts.append((-half_x, bottom_z))
    pts.append((half_x, bottom_z))
    n = len(pts)
    verts = [(x, y0, z) for x, z in pts] + [(x, y1, z) for x, z in pts]
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


def plate_stadium_xz_bottom(name, eye_z, eye_r, top_z, half_x, y0, y1, tile,
                            *, segs=56, bevel=.0009):
    """Plate in the XZ plane (thin along Y) with a round eye at the bottom."""
    pts = []
    for k in range(segs + 1):
        a = math.pi + math.pi * k / segs
        pts.append((eye_r * math.cos(a), eye_z + eye_r * math.sin(a)))
    pts.append((half_x, top_z))
    pts.append((-half_x, top_z))
    n = len(pts)
    verts = [(x, y0, z) for x, z in pts] + [(x, y1, z) for x, z in pts]
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


def tube_points(name, path, radius, tile, *, radial=12):
    """Capped tube through an explicit polyline (no corner rounding)."""
    path = [Vector(p) for p in path]
    tangents = []
    for i in range(len(path)):
        if i == 0:
            t = path[1] - path[0]
        elif i == len(path) - 1:
            t = path[-1] - path[-2]
        else:
            t = path[i + 1] - path[i - 1]
        tangents.append(t.normalized())
    up = Vector((0, 0, 1))
    if abs(tangents[0].dot(up)) > .95:
        up = Vector((1, 0, 0))
    normal = tangents[0].cross(up).normalized()
    normals = [normal]
    for i in range(1, len(path)):
        q = tangents[i - 1].rotation_difference(tangents[i])
        normal = (q @ normals[-1]).normalized()
        normals.append(normal)
    verts, faces, ring_ids = [], [], []
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
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return assign(obj, tile)


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
    cleaned = [dense[0]]
    for point in dense[1:]:
        if (point - cleaned[-1]).length > 1e-7:
            cleaned.append(point)
    return cleaned


def tube_path(name, points, radius, tile, *, corner_radius=.025, radial=16,
              bevel=0.0):
    path = rounded_path(points, corner_radius)
    obj = tube_points(name, path, radius, tile, radial=radial)
    return obj


def axis_basis(axis):
    ez = Vector(axis).normalized()
    ref = Vector((1, 0, 0))
    ex = ref - ez * ref.dot(ez)
    if ex.length < 1e-6:
        ref = Vector((1, 0, 0)) if abs(ez.x) < .9 else Vector((0, 1, 0))
        ex = ref - ez * ref.dot(ez)
    ex.normalize()
    ey = ez.cross(ex)
    return ex, ey, ez


def segment_shell(name, center, axis, radius, t0, t1, half_angle, tile,
                  material, *, segs=56, thickness=0.0006):
    """Closed curved label strip wrapped around an axis, with readable UVs."""
    center = Vector(center)
    ex, ey, ez = axis_basis(axis)
    verts, faces, uvs = [], [], []
    a0, a1 = -half_angle, half_angle
    for i in range(segs + 1):
        a = a0 + (a1 - a0) * i / segs
        c, s = math.cos(a), math.sin(a)
        for r, t in ((radius + thickness, t0), (radius + thickness, t1),
                     (radius, t1), (radius, t0)):
            p = center + ex * (r * c) + ey * (r * s) + ez * t
            verts.append(tuple(p))
    for i in range(segs):
        base, nxt = i * 4, (i + 1) * 4
        faces.append((base + 0, nxt + 0, nxt + 1, base + 1))
        uvs.append(((i / segs, 0), ((i + 1) / segs, 0),
                    ((i + 1) / segs, 1), (i / segs, 1)))
        faces.append((base + 1, nxt + 1, nxt + 2, base + 2))
        uvs.append(((i / segs, 1), ((i + 1) / segs, 1),
                    ((i + 1) / segs, 0), (i / segs, 0)))
        faces.append((base + 2, nxt + 2, nxt + 3, base + 3))
        uvs.append(((i / segs, 0), ((i + 1) / segs, 0),
                    ((i + 1) / segs, 1), (i / segs, 1)))
        faces.append((base + 3, nxt + 3, nxt + 0, base + 0))
        uvs.append(((i / segs, 1), ((i + 1) / segs, 1),
                    ((i + 1) / segs, 0), (i / segs, 0)))
    faces.append((0, 1, 2, 3))
    uvs.append(((0, 0), (0, 1), (0, 0), (0, 1)))
    end = segs * 4
    faces.append((end + 3, end + 2, end + 1, end + 0))
    uvs.append(((1, 0), (1, 1), (1, 0), (1, 1)))
    obj = mesh_object(name, verts, faces, loop_uvs=uvs)
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    objects.append(obj)
    tiles[obj.name] = None
    return obj


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def build_top_clevis():
    c = D['top_clevis']
    eye_z, r, bore = c['eye_z'], c['eye_outer_radius'], c['bore_radius']
    thickness, gap = c['plate_thickness'], c['plate_gap']
    half_x, bottom = c['plate_x_half'], c['plate_bottom_z']
    centers = (-(gap / 2 + thickness / 2), gap / 2 + thickness / 2)
    for name, yc in zip(('top_plate_neg', 'top_plate_pos'), centers):
        y0, y1 = yc - thickness / 2, yc + thickness / 2
        plate = plate_stadium_xz(name, eye_z, r, bottom, half_x, y0, y1,
                                 TILE['black_ano'])
        cutter = along(f'cut_{name}', (0, y0 - .02, eye_z), (0, y1 + .02, eye_z),
                       bore, TILE['rubber'], segs=64, bevel=0, cutter=True)
        apply_boolean(plate, [cutter], f'Bore {name}')
        apply_bevel(plate, .0008, segments=2, angle=35)
        for poly in plate.data.polygons:
            poly.use_smooth = False
    annulus_axis('top_bushing', 'Y', (0, 0, eye_z),
                 -(.016), .016, c['bushing_inner_radius'],
                 c['bushing_outer_radius'], TILE['steel'], segs=96, bevel=.0004)
    box('top_bridge', (0, 0, 0.1605), (0.026, 0.030, 0.013),
        TILE['black_ano'], bevel=.0018)
    n = c['neck']
    box('top_neck', n['center'], n['size'], TILE['black_ano'], bevel=.0018)
    b = c['clamp_bolt']
    hex_prism_axis('top_clamp_washer', b['washer_radius'], .0018,
                   (b['center'][0] - .0007, b['center'][1], b['center'][2]),
                   b['axis'], TILE['steel'], bevel=.0003)
    hex_prism_axis('top_clamp_bolt', b['radius'], b['thickness'],
                   b['center'], b['axis'], TILE['dark_steel'], bevel=.0004)
    hex_prism_axis('top_clamp_socket', b['radius'] * .55, .0016,
                   (b['center'][0] + b['thickness'] / 2 - .0002,
                    b['center'][1], b['center'][2]),
                   b['axis'], TILE['steel'], bevel=.0002)


def build_cap_label():
    c = D['top_cap']
    cylinder_z('top_cap', c['radius'], c['z0'], c['z1'], TILE['black_ano'],
               segs=128, bevel=.0018)
    cone_z('top_cap_shoulder', c['radius'], c['shoulder_radius_top'],
           c['shoulder_z0'], c['shoulder_z1'], TILE['black_ano'], segs=128,
           bevel=.0012)
    for i, z in enumerate(c['groove_z']):
        torus(f'top_cap_groove_{i}', c['radius'] - .0005, .0008, (0, 0, z), 'Z',
              TILE['dark_steel'], major_segments=128, minor_segments=8)
    l = D['label']
    mat = decal_material('YU-AN cyan/white label', TEX['label_cap'],
                         metallic=0.0, roughness=0.36)
    strip = segment_shell('cap_label', (0, 0, (l['z0'] + l['z1']) / 2),
                          (0, 0, 1), l['radius'], -(l['z1'] - l['z0']) / 2,
                          (l['z1'] - l['z0']) / 2, l['half_angle_rad'],
                          None, mat, segs=64, thickness=0.0006)
    nonmetal_objects.append(strip.name)


def build_seats_and_spring():
    s = D['upper_seat']
    cylinder_z('upper_seat', s['radius'], s['z0'], s['z1'], TILE['black_ano'],
               segs=128, bevel=.0016)
    cylinder_z('upper_seat_collar', s['collar_radius'], s['z0'] - .004,
               s['z0'] + .004, TILE['black_ano'], segs=128, bevel=.0010)
    torus('upper_seat_lip', s['radius'] - .002, .0022, (0, 0, s['z1'] - .001),
          'Z', TILE['black_ano'], major_segments=128, minor_segments=8)
    l = D['lower_seat']
    cylinder_z('lower_seat', l['radius'], l['z0'], l['z1'], TILE['black_ano'],
               segs=128, bevel=.0016)
    torus('lower_seat_lip', l['radius'] - .002, .0022, (0, 0, l['z0'] + .001),
          'Z', TILE['black_ano'], major_segments=128, minor_segments=8)
    sp = D['spring']
    turns = sp['turns']
    n = max(120, int(turns * 64))
    pts = []
    for i in range(n + 1):
        t = i / n
        a = math.tau * turns * t
        z = sp['z0'] + (sp['z1'] - sp['z0']) * t
        pts.append((sp['mean_radius'] * math.cos(a),
                    sp['mean_radius'] * math.sin(a), z))
    tube_points('spring', pts, sp['wire_radius'], TILE['blue_spring'], radial=14)
    # small closed-end pads at both spring seats, read as ground coil ends
    torus('spring_end_lower', sp['mean_radius'], sp['wire_radius'] * .96,
          (0, 0, sp['z0'] + sp['wire_radius'] * .55), 'Z',
          TILE['blue_spring'], major_segments=128, minor_segments=10)
    torus('spring_end_upper', sp['mean_radius'], sp['wire_radius'] * .96,
          (0, 0, sp['z1'] - sp['wire_radius'] * .55), 'Z',
          TILE['blue_spring'], major_segments=128, minor_segments=10)


def build_damper():
    d = D['damper']
    cylinder_z('damper_upper', d['upper_body_radius'], d['upper_body_z0'],
               d['upper_body_z1'], TILE['black_ano'], segs=128, bevel=.0012)
    cylinder_z('chrome_shaft', d['chrome_radius'], d['chrome_z0'],
               d['chrome_z1'], TILE['chrome'], segs=96, bevel=.0008)
    smooth_side(bpy.data.objects['chrome_shaft'], (0, 0, 1))
    torus('bronze_guide', d['bronze_guide_radius'], .0018,
          (0, 0, d['bronze_guide_z']), 'Z', TILE['bronze'],
          major_segments=96, minor_segments=8)
    cylinder_z('rubber_wiper', d['rubber_wiper_radius'], d['rubber_wiper_z0'],
               d['rubber_wiper_z1'], TILE['rubber'], segs=96, bevel=.0006)
    b = D['lower_body']
    cylinder_z('lower_body', b['radius'], b['z0'], b['z1'], TILE['black_ano'],
               segs=128, bevel=.0014)
    cone_z('lower_body_cone', b['cone_radius_bottom'], b['cone_radius_top'],
           b['cone_z'], b['z0'], TILE['black_ano'], segs=96, bevel=.0010)
    p = b['front_plug']
    hex_prism_axis('lower_body_plug', p['radius'], p['thickness'],
                   p['center'], p['axis'], TILE['dark_steel'], bevel=.0003)


def build_bottom_clevis():
    c = D['bottom_clevis']
    eye_z, r, bore = c['eye_z'], c['eye_outer_radius'], c['bore_radius']
    thickness, gap = c['plate_thickness'], c['plate_gap']
    half_x, top = c['plate_x_half'], c['plate_top_z']
    centers = (-(gap / 2 + thickness / 2), gap / 2 + thickness / 2)
    for name, yc in zip(('bottom_plate_neg', 'bottom_plate_pos'), centers):
        y0, y1 = yc - thickness / 2, yc + thickness / 2
        plate = plate_stadium_xz_bottom(name, eye_z, r, top, half_x, y0, y1,
                                        TILE['black_ano'])
        cutter = along(f'cut_{name}', (0, y0 - .02, eye_z), (0, y1 + .02, eye_z),
                       bore, TILE['rubber'], segs=64, bevel=0, cutter=True)
        apply_boolean(plate, [cutter], f'Bore {name}')
        apply_bevel(plate, .0008, segments=2, angle=35)
        for poly in plate.data.polygons:
            poly.use_smooth = False
    annulus_axis('bottom_bushing', 'Y', (0, 0, eye_z), -.016, .016,
                 c['bushing_inner_radius'], c['bushing_outer_radius'],
                 TILE['steel'], segs=96, bevel=.0004)
    box('bottom_neck', (0, 0, -0.104), (0.020, 0.026, 0.020),
        TILE['black_ano'], bevel=.0016)
    for name, yc in (('bottom_bolt_neg', -0.0165), ('bottom_bolt_pos', 0.0165)):
        hex_prism_axis(name, c['bolt_head_radius'], c['bolt_head_thickness'],
                       (0, yc, eye_z), (0, 1, 0), TILE['steel'], bevel=.0005)
        hex_prism_axis(f'{name}_socket', c['bolt_head_radius'] * .52, .0014,
                       (0, yc + (0.0032 if yc > 0 else -0.0032), eye_z),
                       (0, 1, 0), TILE['dark_steel'], bevel=.0002)


def build_hose_and_reservoir():
    h = D['hose']
    tube_path('braided_hose', h['path'], h['radius'], TILE['braid'],
              corner_radius=h['corner_radius'], radial=16)
    pts = [Vector(p) for p in h['path']]
    d0 = (pts[1] - pts[0]).normalized()
    d1 = (pts[-1] - pts[-2]).normalized()
    f = h['ferrule_length']
    along('hose_ferrule_top', pts[0] - d0 * .001, pts[0] + d0 * f,
          h['ferrule_radius'], TILE['steel'], segs=32, bevel=.0004)
    along('hose_ferrule_bottom', pts[-1] - d1 * f, pts[-1] + d1 * .001,
          h['ferrule_radius'], TILE['steel'], segs=32, bevel=.0004)
    b = h['top_banjo']
    hex_prism_axis('hose_top_banjo', b['radius'], b['thickness'], b['center'],
                   b['axis'], TILE['steel'], bevel=.0004)
    hex_prism_axis('hose_top_nut', b['radius'] * .82, .005,
                   (b['center'][0], b['center'][1] + .006, b['center'][2]),
                   b['axis'], TILE['dark_steel'], bevel=.0003)
    r = D['reservoir']
    origin = Vector(r['origin'])
    ex, ey, ez = axis_basis(r['axis'])

    def p(t):
        return origin + ez * t

    along('reservoir_body', p(r['body_t0']), p(r['body_t1']),
          r['body_radius'], TILE['blue_ano'], segs=96, bevel=.0011)
    along('reservoir_band', p(r['band_t0']), p(r['band_t1']),
          r['band_radius'], TILE['silver_band'], segs=96, bevel=.0006)
    along('reservoir_cap', p(r['cap_t0']), p(r['cap_t1']),
          r['cap_radius'], TILE['blue_ano'], segs=96, bevel=.0011)
    along('reservoir_end', p(r['cap_t1']), p(r['cap_t1'] + .0012),
          r['cap_radius'] * .92, TILE['blue_ano'], segs=96, bevel=.0003)
    hex_prism_axis('reservoir_end_socket', r['end_socket_radius'],
                   r['end_socket_depth'], p(r['cap_t1'] + .0011),
                   r['axis'], TILE['dark_steel'], bevel=.0002)
    # knurled ribs around the end cap
    mid = (r['cap_t0'] + r['cap_t1']) / 2
    for i in range(r['knurl_count']):
        a = math.tau * i / r['knurl_count']
        radial = ex * (math.cos(a) * r['cap_radius']) + \
            ey * (math.sin(a) * r['cap_radius'])
        c = p(mid) + radial
        along(f'reservoir_knurl_{i:02d}', c - ez * (r['knurl_length'] / 2),
              c + ez * (r['knurl_length'] / 2), r['knurl_radius'],
              TILE['blue_ano'], segs=16, bevel=.0002)
    # near-end 90 degree banjo and hex nut
    banjo_center = p(-.002)
    along('reservoir_banjo_block', banjo_center - ex * .007,
          banjo_center + ex * .007, r['banjo_radius'], TILE['steel'],
          segs=32, bevel=.0004)
    hex_prism_axis('reservoir_hex_nut', r['hex_nut_radius'],
                   r['hex_nut_t1'] - r['hex_nut_t0'],
                   p((r['hex_nut_t0'] + r['hex_nut_t1']) / 2),
                   r['axis'], TILE['steel'], bevel=.0004)
    torus('reservoir_banjo_ring', r['banjo_radius'] * 1.02, .0012,
          tuple(p(0.0)), 'X', TILE['dark_steel'],
          major_segments=40, minor_segments=8)
    # small etched band label, metallic silver with dark lettering
    mat = decal_material('Silver band YU-AN etching', TEX['label_band'],
                         metallic=0.82, roughness=0.24)
    strip = segment_shell('reservoir_band_label', p((r['band_label_t0'] +
                                                     r['band_label_t1']) / 2),
                          r['axis'], r['band_radius'], -(r['band_label_t1'] -
                                                        r['band_label_t0']) / 2,
                          (r['band_label_t1'] - r['band_label_t0']) / 2,
                          r['band_label_half_angle_rad'], None, mat,
                          segs=48, thickness=0.0004)
    metal_objects.append(strip.name)


def build_all():
    build_top_clevis()
    build_cap_label()
    build_seats_and_spring()
    build_damper()
    build_bottom_clevis()
    build_hose_and_reservoir()


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
    checks = {'eyes': {}, 'bushings': {}, 'bolts': {}, 'spring': {},
              'reservoir': {}, 'objects': {}}
    tc = D['top_clevis']
    eye_z = tc['eye_z']
    # Top eye: bore along Y.  Center ray must pass between the two plates;
    # an off-centre ray must hit a plate; a ring offset must hit the bushing.
    plates = [bpy.data.objects['top_plate_neg'], bpy.data.objects['top_plate_pos']]
    centre_clear = []
    solid_hits = []
    for plate in plates:
        tree = make_bvh(plate)
        hit = tree.ray_cast(Vector((0, -.06, eye_z)), Vector((0, 1, 0)), .2)
        centre_clear.append(hit[0] is None)
        hit2 = tree.ray_cast(Vector((tc['plate_x_half'] * .72, -.06, eye_z)),
                             Vector((0, 1, 0)), .2)
        solid_hits.append(hit2[0] is not None)
    bushing = make_bvh(bpy.data.objects['top_bushing'])
    ring_offset = (tc['bushing_inner_radius'] + tc['bushing_outer_radius']) / 2
    ring_hit = bushing.ray_cast(Vector((0, -.06, eye_z + ring_offset)),
                                Vector((0, 1, 0)), .2)
    checks['eyes']['top'] = {
        'through': all(centre_clear), 'plate_solid': all(solid_hits),
        'centre_rays_clear': centre_clear, 'solid_side_hits': solid_hits}
    checks['bushings']['top'] = {'present': ring_hit[0] is not None}
    bc = D['bottom_clevis']
    eye_z = bc['eye_z']
    plates = [bpy.data.objects['bottom_plate_neg'], bpy.data.objects['bottom_plate_pos']]
    centre_clear = []
    solid_hits = []
    for plate in plates:
        tree = make_bvh(plate)
        hit = tree.ray_cast(Vector((0, -.06, eye_z)), Vector((0, 1, 0)), .2)
        centre_clear.append(hit[0] is None)
        hit2 = tree.ray_cast(Vector((0, -.06, eye_z + bc['eye_outer_radius'] * .72)),
                             Vector((0, 1, 0)), .2)
        solid_hits.append(hit2[0] is not None)
    bushing = make_bvh(bpy.data.objects['bottom_bushing'])
    ring_offset = (bc['bushing_inner_radius'] + bc['bushing_outer_radius']) / 2
    ring_hit = bushing.ray_cast(Vector((0, -.06, eye_z + ring_offset)),
                                Vector((0, 1, 0)), .2)
    checks['eyes']['bottom'] = {
        'through': all(centre_clear), 'plate_solid': all(solid_hits),
        'centre_rays_clear': centre_clear, 'solid_side_hits': solid_hits}
    checks['bushings']['bottom'] = {'present': ring_hit[0] is not None}
    checks['bolts']['top_clamp'] = bpy.data.objects.get('top_clamp_bolt') is not None
    checks['bolts']['bottom_head_count'] = len(
        [n for n in bpy.data.objects.keys() if n.startswith('bottom_bolt_') and
         not n.endswith('_socket')])
    checks['bolts']['reservoir_socket'] = bpy.data.objects.get(
        'reservoir_end_socket') is not None
    checks['bolts']['knurl_count'] = len(
        [n for n in bpy.data.objects.keys() if n.startswith('reservoir_knurl_')])
    # Spring: open helix, not a filled cylinder.
    spring = bpy.data.objects['spring']
    tree = make_bvh(spring)
    z0, z1 = D['spring']['z0'], D['spring']['z1']
    hits = 0
    rays = 24
    for i in range(rays):
        z = z0 + (z1 - z0) * (i + .5) / rays
        hit = tree.ray_cast(Vector((0, 0, z)), Vector((1, 0, 0)), .08)
        hits += int(hit[0] is not None)
    checks['spring'] = {
        'turns': D['spring']['turns'],
        'wire_radius_m': D['spring']['wire_radius'],
        'radial_rays_hit': hits, 'radial_rays': rays,
        'open_helix': 0 < hits < rays,
        'note': 'from-axis outward rays intersect the wire on some z slices and pass through gaps on others'}
    # Reservoir end socket reachable along the axis.
    r = D['reservoir']
    origin = Vector(r['origin'])
    ez = Vector(r['axis']).normalized()
    tip = origin + ez * (r['cap_t1'] + .05)
    socket = make_bvh(bpy.data.objects['reservoir_end_socket'])
    hit = socket.ray_cast(tip, -ez, .2)
    checks['reservoir'] = {
        'end_socket_hit': hit[0] is not None,
        'cap': bpy.data.objects.get('reservoir_cap') is not None,
        'band': bpy.data.objects.get('reservoir_band') is not None,
        'knurls': checks['bolts']['knurl_count'],
    }
    checks['objects'] = {
        'parts': len(objects),
        'top_plates': len([n for n in bpy.data.objects.keys()
                           if n.startswith('top_plate_')]),
        'bottom_plates': len([n for n in bpy.data.objects.keys()
                              if n.startswith('bottom_plate_')]),
        'hose': bpy.data.objects.get('braided_hose') is not None,
    }
    checks['passed'] = bool(
        checks['eyes']['top']['through'] and checks['eyes']['bottom']['through']
        and checks['eyes']['top']['plate_solid']
        and checks['eyes']['bottom']['plate_solid']
        and checks['bushings']['top']['present']
        and checks['bushings']['bottom']['present']
        and checks['spring']['open_helix']
        and checks['reservoir']['end_socket_hit'])
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
    background.inputs['Strength'].default_value = .46
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
            ('main softbox', (2.4, -2.8, 3.8), 230, 3.2),
            ('front reflection', (3.0, 1.4, 1.0), 110, 1.5),
            ('top fill', (0.0, 0.0, 4.0), 85, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 70, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 35, 1.4)):
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
        view_scale = span * (1.12 if name in ('front', 'back') else
                             (.80 if name in ('left', 'right') else 1.02))
        look(scene, direction, center, span, ortho_scale=view_scale)
        draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    source = request['source_camera']
    set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
               shift_x=source.get('shift_x', 0), shift_y=source.get('shift_y', 0))
    draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])

    if not request.get('draft'):
        target = {
            'top-eye': object_center('top_bushing'),
            'top-bolt': object_center('top_clamp_bolt'),
            'label': object_center('cap_label'),
            'spring-upper': Vector((0, 0, 0.030)),
            'spring-lower': Vector((0, 0, -0.040)),
            'bottom-clevis': object_center('bottom_bushing'),
            'hose-top': object_center('hose_top_banjo'),
            'hose-run': Vector((0.007, 0.085, 0.000)),
            'reservoir-banjo': object_center('reservoir_banjo_block'),
            'reservoir-cap': object_center('reservoir_cap'),
            'overall-3q': center,
        }
        detail_specs = [
            ('top-eye', (1.0, -.22, .18), target['top-eye'], .095),
            ('top-bolt', (1.0, -.08, .12), target['top-bolt'], .050),
            ('label', (1.0, -.16, .06), target['label'], .105),
            ('spring-upper', (.75, -.55, .18), target['spring-upper'], .115),
            ('spring-lower', (.70, -.60, .10), target['spring-lower'], .105),
            ('bottom-clevis', (1.0, -.20, .10), target['bottom-clevis'], .080),
            ('hose-top', (.85, .35, .25), target['hose-top'], .075),
            ('hose-run', (.45, .78, .12), target['hose-run'], .170),
            ('reservoir-banjo', (.60, .72, .18), target['reservoir-banjo'], .080),
            ('reservoir-cap', (.72, .62, .18), target['reservoir-cap'], .095),
            ('overall-3q', (1.0, -.85, .45), target['overall-3q'], span * 1.14),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        set_visible({'top_plate_neg', 'top_plate_pos', 'top_bushing',
                     'top_bridge', 'top_neck', 'top_clamp_washer',
                     'top_clamp_bolt', 'top_clamp_socket'})
        look(scene, (1, -.25, .18), object_center('top_bushing'), span,
             ortho_scale=.100)
        draw(scene, render_dir / 'top-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'bottom_plate_neg', 'bottom_plate_pos', 'bottom_bushing',
                     'bottom_neck', 'bottom_bolt_neg', 'bottom_bolt_pos',
                     'bottom_bolt_neg_socket', 'bottom_bolt_pos_socket'})
        look(scene, (1, -.20, .12), object_center('bottom_bushing'), span,
             ortho_scale=.090)
        draw(scene, render_dir / 'bottom-clevis-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'spring', 'spring_end_lower', 'spring_end_upper',
                     'upper_seat', 'upper_seat_collar', 'upper_seat_lip',
                     'lower_seat', 'lower_seat_lip'})
        look(scene, (.7, -.7, .15), Vector((0, 0, 0.0)), span, ortho_scale=.245)
        draw(scene, render_dir / 'spring-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        reservoir = {'reservoir_body', 'reservoir_band', 'reservoir_cap',
                     'reservoir_end', 'reservoir_end_socket',
                     'reservoir_banjo_block', 'reservoir_hex_nut',
                     'reservoir_banjo_ring', 'reservoir_band_label'} | {
            n for n in bpy.data.objects.keys()
            if n.startswith('reservoir_knurl_')}
        set_visible(reservoir)
        look(scene, (.55, .78, .30), object_center('reservoir_band'), span,
             ortho_scale=.150)
        draw(scene, render_dir / 'reservoir-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'braided_hose', 'hose_ferrule_top', 'hose_ferrule_bottom',
                     'hose_top_banjo', 'hose_top_nut', 'reservoir_banjo_block',
                     'reservoir_hex_nut', 'reservoir_banjo_ring'})
        look(scene, (1.0, -.55, .25), Vector((0.01, 0.045, 0.02)), span,
             ortho_scale=.240)
        draw(scene, render_dir / 'hose-only.png', rr['detail_resolution'],
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
            if obj.name in ('cap_label',):
                obj.data.materials.clear()
                obj.data.materials.append(
                    decal_material('YU-AN cyan/white label', TEX['label_cap'],
                                   metallic=0.0, roughness=0.36))
            elif obj.name == 'reservoir_band_label':
                obj.data.materials.clear()
                obj.data.materials.append(
                    decal_material('Silver band YU-AN etching', TEX['label_band'],
                                   metallic=0.82, roughness=0.24))
            else:
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
        rim = bpy.data.objects.get('rim')
        low_light = bpy.data.objects.get('low fill')
        for light in (fill_light, front_light, top_light, rim, low_light):
            if light is not None:
                light.hide_render = True
        metal_target = Vector((0.02, 0.070, 0.010))
        look(scene, (.90, -.40, .35), metal_target, span, ortho_scale=.360)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (.90, -.40, .35), metal_target, span, ortho_scale=.360)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for light in (fill_light, front_light, top_light, rim, low_light):
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
        'geometry_basis': '原照可见结构 + 未标定六视图（黑蓝摩托车后减震器）',
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'shift': list(shift),
        'config_version': cfg['version'],
    }
    (out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                            encoding='utf8')


main()
