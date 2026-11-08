"""Blender worker: part-based rebuild of item 29, a red/black motorcycle rear shock.

Evidence:
- one 2164x727 clean side photograph and its matching rust derivative of a
  red-anodized shock with a gold damper body, black coil spring and large
  red preload adjuster ring;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified).  It is a generated catalogue-style sheet, so the
  photograph wins for the visible side; the sheet is used only for the
  unseen back/topology and for the general part inventory.

The old TripoSR library mesh is a low-resolution shell: the eye holes are
filled, the spring coils fuse into the body and the adjuster ring is lost.
It is retained unchanged and is not used as the candidate's base geometry.
The candidate is a parametric part rebuild with a real helical spring, true
through-bored eyes with steel/brass bushings, a threaded preload collar, a
lugged adjuster ring with a set screw, a red right perch, a chrome shaft and
an atlas PBR material.  All candidates remain manual-only and pending review;
nothing replaces the library mesh or enters automatic placement.

Coordinates: Z up, main axis along Y, left eye at -Y, right eye at +Y,
front camera at +X (so the rendered side matches the source photograph),
eye bores along X.
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
METAL_TILES = {'red_ano', 'red_dark', 'gold_ano', 'gold_thread', 'chrome',
               'steel', 'brass', 'black_steel'}
DIELECTRIC_TILES = {'spring_black', 'rubber', 'rust', 'spare11', 'spare12',
                    'spare13', 'spare14', 'spare15'}


# ---------------------------------------------------------------------------
# Atlas material and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed red anodized, gold, chrome, '
                                      'steel and brass PBR atlas')
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
    normal_map.inputs['Strength'].default_value = .22
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


def ensure_auto_uv(obj, scale=6.0):
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
           margin=.012, register=True):
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
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def finish(obj, name, tile, *, bevel=.0012, smooth=True, constant=False,
           v_lo=.01, v_hi=.99):
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
                  v_lo=v_lo, v_hi=v_hi)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
def mesh_object(name, verts, faces, loop_uvs=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    if loop_uvs is not None:
        layer = mesh.uv_layers.new(name='Atlas')
        for poly, face_uvs in zip(mesh.polygons, loop_uvs):
            for li, uv in zip(poly.loop_indices, face_uvs):
                layer.data[li].uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def axis_basis(axis):
    axis = Vector(axis).normalized()
    if abs(axis.y) > .9:
        u, v = Vector((1, 0, 0)), Vector((0, 0, 1))
    elif abs(axis.x) > .9:
        u, v = Vector((0, 1, 0)), Vector((0, 0, 1))
    else:
        u = axis.cross(Vector((0, 0, 1))).normalized()
        v = axis.cross(u).normalized()
    return axis, u, v


def lathe_axis(name, axis, origin, profile, tile, *, closed=False, segs=96,
               bevel=0.0, smooth=True):
    """Revolve a (axial position, radius) profile around an arbitrary axis."""
    axis, u, v = axis_basis(axis)
    origin = Vector(origin)
    count = len(profile)
    verts = []
    for t, radius in profile:
        for i in range(segs):
            ang = math.tau * i / segs
            p = origin + axis * t + u * (radius * math.cos(ang)) + \
                v * (radius * math.sin(ang))
            verts.append(tuple(p))
    faces, loop_uvs = [], []

    def vfrac(j):
        return j / count if closed else j / (count - 1)

    segments = count if closed else count - 1
    for j in range(segments):
        j1 = (j + 1) % count
        v0 = vfrac(j)
        v1 = (j + 1) / count if closed else (j + 1) / (count - 1)
        for i in range(segs):
            i1 = (i + 1) % segs
            faces.append((j * segs + i, j * segs + i1,
                          j1 * segs + i1, j1 * segs + i))
            loop_uvs.append(((i / segs, v0), (i1 / segs, v0),
                             (i1 / segs, v1), (i / segs, v1)))
    if not closed:
        # flat caps close the solid without hiding the through-hole profiles
        for end, j in ((0, 0), (1, count - 1)):
            t = profile[j][0]
            center = len(verts)
            verts.append(tuple(origin + axis * t))
            for i in range(segs):
                i1 = (i + 1) % segs
                a0 = math.tau * i / segs
                a1 = math.tau * i1 / segs
                if end == 0:
                    faces.append((center, j * segs + i1, j * segs + i))
                else:
                    faces.append((center, j * segs + i, j * segs + i1))
                loop_uvs.append((
                    (.5, .5),
                    (.5 + .45 * math.cos(a1), .5 + .45 * math.sin(a1)),
                    (.5 + .45 * math.cos(a0), .5 + .45 * math.sin(a0))))
    obj = mesh_object(name, verts, faces, loop_uvs)
    return finish(obj, name, tile, bevel=bevel, smooth=smooth)


def annulus_axis(name, axis, center, t0, t1, r_in, r_out, tile, *, segs=96,
                 bevel=0.0):
    """Closed washer/tube with a real axial through-hole."""
    profile = [(t0, r_out), (t1, r_out), (t1, r_in), (t0, r_in)]
    return lathe_axis(name, axis, center, profile, tile, closed=True, segs=segs,
                      bevel=bevel)


def helix_spring(name, y0, y1, mean_r, wire_r, turns, tile, *, phase=0.0,
                 steps_per_turn=44, radial=14, bevel=0.0):
    steps = max(8, int(round(turns * steps_per_turn)))
    verts, faces, loop_uvs = [], [], []
    centers = []
    frames = []
    for k in range(steps + 1):
        t = k / steps
        y = y0 + (y1 - y0) * t
        phi = phase + math.tau * turns * t
        c = Vector((mean_r * math.cos(phi), y, mean_r * math.sin(phi)))
        tangent = Vector((-mean_r * math.tau * turns * math.sin(phi),
                          (y1 - y0),
                          mean_r * math.tau * turns * math.cos(phi))).normalized()
        n = Vector((math.cos(phi), 0.0, math.sin(phi)))
        n = (n - tangent * n.dot(tangent)).normalized()
        b = tangent.cross(n).normalized()
        centers.append(c)
        frames.append((n, b, t))
        for j in range(radial):
            psi = math.tau * j / radial
            p = c + n * (wire_r * math.cos(psi)) + b * (wire_r * math.sin(psi))
            verts.append(tuple(p))
    for k in range(steps):
        for j in range(radial):
            j1 = (j + 1) % radial
            a = k * radial + j
            b = k * radial + j1
            c = (k + 1) * radial + j1
            d = (k + 1) * radial + j
            faces.append((a, b, c, d))
            u0 = frames[k][2] * turns * 4.
            u1 = frames[k + 1][2] * turns * 4.
            loop_uvs.append(((u0, j / radial), (u0, j1 / radial),
                             (u1, j1 / radial), (u1, j / radial)))
    # end caps make the wire a closed solid; the cut ends read as ground wire
    for end, k in ((0, 0), (1, steps)):
        center = len(verts)
        verts.append(tuple(centers[k]))
        for j in range(radial):
            j1 = (j + 1) % radial
            if end == 0:
                faces.append((center, k * radial + j1, k * radial + j))
            else:
                faces.append((center, k * radial + j, k * radial + j1))
            a0 = math.tau * j / radial
            a1 = math.tau * j1 / radial
            loop_uvs.append((
                (.5, .5),
                (.5 + .45 * math.cos(a1), .5 + .45 * math.sin(a1)),
                (.5 + .45 * math.cos(a0), .5 + .45 * math.sin(a0))))
    obj = mesh_object(name, verts, faces, loop_uvs)
    return finish(obj, name, tile, bevel=bevel, smooth=True)


def box(name, center, size, tile, *, rotation_y=0.0, bevel=0.001, smooth=False):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    if rotation_y:
        obj.rotation_euler[1] = rotation_y
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, smooth=smooth)


def cylinder_axis(name, center, axis, radius, length, tile, *, verts=32,
                  bevel=0.0006, smooth=True):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius,
                                        depth=length, location=center)
    obj = bpy.context.object
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(
        Vector(axis).normalized())
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, smooth=smooth)


def plate_prism(name, y0, y1, h0, h1, thickness, tile, *, bevel=0.0012):
    x0, x1 = -thickness / 2, thickness / 2
    verts = [
        (x0, y0, -h0), (x0, y0, h0), (x0, y1, h1), (x0, y1, -h1),
        (x1, y0, -h0), (x1, y0, h0), (x1, y1, h1), (x1, y1, -h1),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5),
        (0, 3, 2, 1), (4, 5, 6, 7),
        (0, 1, 5, 4), (3, 7, 6, 2),
        (0, 4, 7, 3), (1, 2, 6, 5),
    ]
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, tile, bevel=bevel, smooth=False)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def build_eye(prefix, spec, collar_x):
    y = spec['y']
    boss = annulus_axis(f'{prefix}_red_boss', (1, 0, 0), (0, y, 0),
                        -spec['boss_thickness'] / 2,
                        spec['boss_thickness'] / 2,
                        spec['boss_inner_radius'], spec['boss_outer_radius'],
                        TILE['red_ano'], bevel=.0005)
    collar = annulus_axis(f'{prefix}_red_collar', (1, 0, 0), (0, y, 0),
                          collar_x - spec['collar_thickness'] / 2,
                          collar_x + spec['collar_thickness'] / 2,
                          spec['collar_inner_radius'], spec['collar_outer_radius'],
                          TILE['red_dark'], bevel=.0004)
    bushing = annulus_axis(f'{prefix}_steel_bushing', (1, 0, 0), (0, y, 0),
                           -spec['bushing_thickness'] / 2,
                           spec['bushing_thickness'] / 2,
                           spec['bushing_inner_radius'],
                           spec['bushing_outer_radius'],
                           TILE['steel'], bevel=.0004)
    sleeve = annulus_axis(f'{prefix}_brass_sleeve', (1, 0, 0), (0, y, 0),
                          -spec['sleeve_thickness'] / 2,
                          spec['sleeve_thickness'] / 2,
                          spec['sleeve_inner_radius'],
                          spec['sleeve_outer_radius'],
                          TILE['brass'], bevel=.0003)
    return [boss, collar, bushing, sleeve]


def thread_profile(y0, y1, radius, depth, pitch):
    count = max(8, int(round((y1 - y0) / pitch)))
    span = (y1 - y0 - .0010) / count
    pts = [(y0, radius * .985)]
    for k in range(count):
        y = y0 + .0005 + span * k
        pts.append((y, radius))
        pts.append((y + span * .5, radius - depth))
    pts.append((y1 - .0005, radius))
    pts.append((y1, radius * .985))
    return pts


def build_left_end():
    a = D['left_arm']
    plate_prism('left_eye_arm', a['y0'], a['y1'], a['half_height_eye'],
                a['half_height_body'], a['thickness_x'], TILE['red_ano'],
                bevel=.0014)
    lath = D['left_neck']['profile']
    lathe_axis('left_neck', (0, 1, 0), (0, 0, 0), lath, TILE['red_ano'],
               bevel=.0006)
    lathe_axis('left_body', (0, 1, 0), (0, 0, 0),
               D['left_body']['profile'], TILE['red_ano'], bevel=0.0)
    # two dark machined grooves follow the red body profile
    for idx, y in enumerate((-0.1380, -0.1196)):
        annulus_axis(f'left_body_groove_{idx}', (0, 1, 0), (0, 0, 0),
                     y - .0006, y + .0006, .0305, .0332, TILE['red_dark'],
                     segs=96, bevel=0)
    build_eye('left_eye', D['left_eye'], D['left_eye']['collar_x'])


def build_body_and_adjuster():
    g = D['gold_thread']
    lathe_axis('gold_thread_collar', (0, 1, 0), (0, 0, 0),
               thread_profile(g['y0'], g['y1'], g['base_radius'],
                              g['thread_depth'], g['pitch']),
               TILE['gold_thread'], bevel=0.0, smooth=False)
    r = D['adjuster_ring']
    annulus_axis('adjuster_ring_base', (0, 1, 0), (0, 0, 0),
                 r['y0'], r['y1'], r['inner_radius'], r['outer_radius'],
                 TILE['red_ano'], bevel=.0007)
    ymid = (r['y0'] + r['y1']) / 2
    rmid = (r['outer_radius'] + r['lug_outer_radius']) / 2
    for i in range(r['lug_count']):
        ang = math.tau * i / r['lug_count']
        box(f'adjuster_lug_{i:02d}',
            (rmid * math.cos(ang), ymid, rmid * math.sin(ang)),
            (r['lug_outer_radius'] - r['outer_radius'] + .004,
             r['lug_y_length'], r['lug_tangential_width']),
            TILE['red_ano'], rotation_y=-ang, bevel=.0010)
    ang = r['set_screw_angle_rad']
    radial = Vector((math.cos(ang), 0, math.sin(ang)))
    center = radial * r['set_screw_radius_position']
    center.y = (r['y0'] + r['y1']) / 2
    cylinder_axis('adjuster_set_screw', center, radial,
                  r['set_screw_radius'], r['set_screw_length'],
                  TILE['black_steel'], verts=24, bevel=.0004)
    socket_center = radial * (r['set_screw_radius_position'] +
                              r['set_screw_length'] / 2 - .0012)
    socket_center.y = center.y
    cylinder_axis('adjuster_set_socket', socket_center, radial, .0034, .0018,
                  TILE['rubber'], verts=6, bevel=0, smooth=False)


def build_spring_and_core():
    s = D['spring']
    helix_spring('spring', s['y0'], s['y1'], s['mean_radius'],
                 s['wire_radius'], s['turns'], TILE['spring_black'],
                 phase=s['phase_rad'], steps_per_turn=s['steps_per_turn'],
                 radial=s['radial_segments'])
    lathe_axis('gold_damper_barrel', (0, 1, 0), (0, 0, 0),
               D['gold_barrel']['profile'], TILE['gold_ano'], bevel=.0006)
    lathe_axis('black_spacer', (0, 1, 0), (0, 0, 0),
               D['black_spacer']['profile'], TILE['black_steel'], bevel=.0005)
    w = D['shaft_wiper']
    annulus_axis('shaft_wiper', (0, 1, 0), (0, 0, 0), w['y0'], w['y1'],
                 w['inner_radius'], w['outer_radius'], TILE['rubber'],
                 bevel=.0005)
    lathe_axis('chrome_shaft', (0, 1, 0), (0, 0, 0),
               D['chrome_shaft']['profile'], TILE['chrome'], bevel=.0003)


def build_right_end():
    p = D['right_perch']
    outer = p['outer_profile']
    prof = [(t, r) for t, r in outer] + \
           [(p['y1'], p['inner_radius']), (p['y0'], p['inner_radius'])]
    lathe_axis('right_perch', (0, 1, 0), (0, 0, 0), prof, TILE['red_ano'],
               closed=True, bevel=.0008)
    lathe_axis('right_hub', (0, 1, 0), (0, 0, 0),
               D['right_hub']['profile'], TILE['red_ano'], bevel=.0006)
    a = D['right_arm']
    plate_prism('right_eye_arm', a['y0'], a['y1'], a['half_height_perch'],
                a['half_height_eye'], a['thickness_x'], TILE['red_ano'],
                bevel=.0014)
    build_eye('right_eye', D['right_eye'], D['right_eye']['collar_x'])


def build_all():
    build_left_end()
    build_body_and_adjuster()
    build_spring_and_core()
    build_right_end()


# ---------------------------------------------------------------------------
# Scene, rays, export and renders
# ---------------------------------------------------------------------------
def init_camera(scene):
    data = bpy.data.cameras.new('review camera')
    cam = bpy.data.objects.new('review camera', data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_camera(cam, eye, target, *, ortho_scale=None, fov=40, shift_x=0,
               shift_y=0, top_orientation=False):
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
    return bounds([bpy.data.objects[name]])[2]


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
    checks = {'eyes': {}, 'spring': {}, 'adjuster': {}, 'objects': {}}
    for side, spec, prefix in (
            ('left', D['left_eye'], 'left_eye'),
            ('right', D['right_eye'], 'right_eye')):
        y = spec['y']
        names = [f'{prefix}_red_boss', f'{prefix}_red_collar',
                 f'{prefix}_steel_bushing', f'{prefix}_brass_sleeve']
        clear = []
        for name in names:
            tree = make_bvh(bpy.data.objects[name])
            hit = tree.ray_cast(Vector((-.10, y, 0)), Vector((1, 0, 0)), .2)
            clear.append(hit[0] is None)
        steel = make_bvh(bpy.data.objects[f'{prefix}_steel_bushing'])
        steel_r = (spec['bushing_inner_radius'] +
                   spec['bushing_outer_radius']) / 2
        hit_steel = steel.ray_cast(Vector((-.10, y, steel_r)),
                                   Vector((1, 0, 0)), .2)
        brass = make_bvh(bpy.data.objects[f'{prefix}_brass_sleeve'])
        brass_r = (spec['sleeve_inner_radius'] +
                   spec['sleeve_outer_radius']) / 2
        hit_brass = brass.ray_cast(Vector((-.10, y, brass_r)),
                                   Vector((1, 0, 0)), .2)
        checks['eyes'][side] = {
            'center_clear': clear,
            'through': all(clear),
            'steel_bushing_hit': hit_steel[0] is not None,
            'brass_sleeve_hit': hit_brass[0] is not None,
            'bore_diameter_m': 2 * spec['sleeve_inner_radius'],
        }
    spring = bpy.data.objects['spring']
    tree = make_bvh(spring)
    s = D['spring']
    hits = 0
    rays = 24
    for i in range(rays):
        y = s['y0'] + (s['y1'] - s['y0']) * (i + .5) / rays
        hit = tree.ray_cast(Vector((0, y, 0)), Vector((0, 0, 1)), .08)
        hits += int(hit[0] is not None)
    checks['spring'] = {
        'turns': s['turns'],
        'wire_radius_m': s['wire_radius'],
        'radial_rays_hit': hits,
        'radial_rays': rays,
        'open_helix': 0 < hits < rays,
        'note': ('轴线向外射线在部分位置命中钢丝、在圈间空隙穿过，'
                 '证明弹簧是开放螺旋而不是实心圆柱'),
    }
    checks['adjuster'] = {
        'lug_count': len([n for n in bpy.data.objects.keys()
                          if n.startswith('adjuster_lug_')]),
        'set_screw': bpy.data.objects.get('adjuster_set_screw') is not None,
        'set_socket': bpy.data.objects.get('adjuster_set_socket') is not None,
        'thread_ridges': int(round(
            (D['gold_thread']['y1'] - D['gold_thread']['y0']) /
            D['gold_thread']['pitch'])),
    }
    checks['objects'] = {
        'parts': len(objects),
        'steel_bushings': len([n for n in bpy.data.objects.keys()
                               if n.endswith('_steel_bushing')]),
        'brass_sleeves': len([n for n in bpy.data.objects.keys()
                              if n.endswith('_brass_sleeve')]),
    }
    checks['passed'] = bool(
        checks['eyes']['left']['through'] and checks['eyes']['right']['through']
        and checks['eyes']['left']['steel_bushing_hit']
        and checks['eyes']['right']['steel_bushing_hit']
        and checks['eyes']['left']['brass_sleeve_hit']
        and checks['eyes']['right']['brass_sleeve_hit']
        and checks['spring']['open_helix']
        and checks['adjuster']['lug_count'] == D['adjuster_ring']['lug_count']
        and checks['adjuster']['set_screw'])
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
    view_names = ['front'] if request.get('draft') else list(directions)
    for name in view_names:
        direction = directions[name]
        view_scale = span * (1.10 if name in ('front', 'back', 'top', 'bottom')
                             else .32)
        look(scene, direction, center, span, ortho_scale=view_scale)
        draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
             rr['six_view_samples'])

    source = request['source_camera']
    set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
               shift_x=source.get('shift_x', 0), shift_y=source.get('shift_y', 0))
    draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])

    if not request.get('draft'):
        target = {
            'left-eye': object_center('left_eye_steel_bushing'),
            'right-eye': object_center('right_eye_steel_bushing'),
            'adjuster-ring': object_center('adjuster_ring_base'),
            'thread': Vector((0, D['gold_thread']['y0'] + .015, 0)),
            'spring-center': Vector((0, .0, 0)),
            'barrel-spacer': Vector((0, .02, 0)),
            'right-perch': object_center('right_perch'),
            'overall-3q': center,
        }
        detail_specs = [
            ('left-eye', (1.0, -.30, .22), target['left-eye'], .080),
            ('right-eye', (1.0, .30, .18), target['right-eye'], .085),
            ('adjuster-ring', (.95, -.62, .18), target['adjuster-ring'], .115),
            ('thread', (.80, -.62, .16), target['thread'], .062),
            ('spring-center', (.92, -.42, .22), target['spring-center'], .150),
            ('barrel-spacer', (.88, -.48, .20), target['barrel-spacer'], .120),
            ('right-perch', (.90, .35, .18), target['right-perch'], .110),
            ('overall-3q', (1.0, -.78, .34), target['overall-3q'], span * 1.12),
        ]
        for name, direction, tgt, scale in detail_specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
                 rr['detail_samples'])

        set_visible({'left_eye_red_boss', 'left_eye_red_collar',
                     'left_eye_steel_bushing', 'left_eye_brass_sleeve',
                     'left_eye_arm', 'left_neck'})
        look(scene, (1, -.32, .20), object_center('left_eye_steel_bushing'),
             span, ortho_scale=.095)
        draw(scene, render_dir / 'left-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'right_eye_red_boss', 'right_eye_red_collar',
                     'right_eye_steel_bushing', 'right_eye_brass_sleeve',
                     'right_eye_arm', 'right_hub'})
        look(scene, (1, .32, .18), object_center('right_eye_steel_bushing'),
             span, ortho_scale=.100)
        draw(scene, render_dir / 'right-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'spring'})
        look(scene, (.85, -.50, .20), Vector((0, .0, 0)), span, ortho_scale=.240)
        draw(scene, render_dir / 'spring-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        adjuster = {'adjuster_ring_base', 'adjuster_set_screw',
                    'adjuster_set_socket', 'gold_thread_collar'} | {
            n for n in bpy.data.objects.keys()
            if n.startswith('adjuster_lug_')}
        set_visible(adjuster)
        look(scene, (.95, -.70, .16), object_center('adjuster_ring_base'),
             span, ortho_scale=.125)
        draw(scene, render_dir / 'adjuster-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        set_visible({'gold_damper_barrel', 'black_spacer', 'shaft_wiper',
                     'chrome_shaft', 'right_perch', 'right_hub'})
        look(scene, (.85, -.35, .18), Vector((0, .04, 0)), span,
             ortho_scale=.180)
        draw(scene, render_dir / 'damper-core-only.png', rr['detail_resolution'],
             rr['detail_samples'])

        set_visible({obj.name for obj in objects})
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-material-front.png',
             rr['six_view_resolution'], rr['six_view_samples'])

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
        draw(scene, render_dir / 'new-clay-front.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        for obj in objects:
            obj.data.materials.clear()
            obj.data.materials.append(MATERIAL)

        main_light = bpy.data.objects['main softbox']
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat(
            '-Z', 'Y').to_euler()
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-light-alternate.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat(
            '-Z', 'Y').to_euler()

        set_visible(set(metal_objects))
        for light_name in ('soft light', 'front reflection', 'top fill',
                           'rim', 'low fill'):
            light = bpy.data.objects.get(light_name)
            if light is not None:
                light.hide_render = True
        metal_target = Vector((0, .01, 0))
        look(scene, (.90, -.42, .32), metal_target, span, ortho_scale=.330)
        draw(scene, render_dir / 'metal-only-light-a.png',
             rr['metal_resolution'], rr['metal_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat(
            '-Z', 'Y').to_euler()
        look(scene, (.90, -.42, .32), metal_target, span, ortho_scale=.330)
        draw(scene, render_dir / 'metal-only-light-b.png',
             rr['metal_resolution'], rr['metal_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat(
            '-Z', 'Y').to_euler()
        for light_name in ('soft light', 'front reflection', 'top fill',
                           'rim', 'low fill'):
            light = bpy.data.objects.get(light_name)
            if light is not None:
                light.hide_render = False
        set_visible({obj.name for obj in objects})

        for i in range(24):
            a = math.tau * i / 24
            look(scene, (math.cos(a), math.sin(a), .20), center, span)
            draw(scene, render_dir / f'turn-{i:02d}.png',
                 rr['turntable_resolution'], rr['turntable_samples'])

    for obj in objects:
        obj.hide_render = True
    old = legacy.load_mesh(request['original_mesh'])
    oldlo, oldhi, oldcenter, oldspan = bounds([old])
    for name in (['front'] if request.get('draft') else list(directions)):
        direction = directions[name]
        look(scene, direction, oldcenter, oldspan,
             ortho_scale=oldspan * (1.10 if name in
                                    ('front', 'back', 'top', 'bottom') else .32))
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
        'geometry_basis': '原照可见结构 + 未标定六视图（红黑摩托车后减震器）',
        'render_samples': rr['six_view_samples'],
        'source_camera': source,
        'shift': list(shift),
        'old_bounds': [list(oldlo), list(oldhi)],
        'config_version': cfg['version'],
    }
    (out / 'worker-result.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


main()
