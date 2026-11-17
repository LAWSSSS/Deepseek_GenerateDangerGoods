"""Blender worker: part-based rebuild of item 28, a black coil-over shock.

Source photograph: a black motorcycle-style rear shock with a 12-turn black
coil spring, a black upper damper body, a chrome piston rod at the lower third
and two eye mounts with steel bushings.  The supplied six-view sheet is an
uncalibrated AI-generated sheet that agrees on the visible front/side layout
but contradicts the photograph in its top/bottom panels (it draws the eye bore
facing the long axis).  The candidate follows the photograph: long axis +Z,
eye bore axis +X.

Coordinates: Z up, eye bore axis along X, front side +X.  All parts are
separate watertight meshes.  Painted steel and rubber are dielectric; the two
bushings, inner sleeves and chrome rod are metallic.  No light or shadow is
baked into the textures.  Candidate is manual-only and never replaces the
library mesh.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Matrix, Quaternion, Vector
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
CAM = None
METAL_TILES = {'steel', 'chrome', 'dark', 'steel_ground'}
NONMETAL_TILES = {'paint', 'seat', 'rubber'}


# ---------------------------------------------------------------------------
# Materials and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Black paint, rusted steel, chrome and rubber PBR atlas')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    principled.inputs['IOR'].default_value = 1.48

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


def assign(obj, tile, *, mode='default', register=True):
    obj.data.materials.clear()
    obj.data.materials.append(MATERIAL)
    tiles[obj.name] = tile
    if mode == 'constant':
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name='Atlas')
        x0, y0, x1, y1 = tile_bounds(tile)
        center = (x0 + .5 * (x1 - x0), y0 + .5 * (y1 - y0))
        for loop in obj.data.uv_layers.active.data:
            loop.uv = center
    else:
        remap_uv(obj, tile)
    if register:
        objects.append(obj)
        name = next((key for key, value in TILE.items() if value == tile), None)
        if name in METAL_TILES:
            metal_objects.append(obj.name)
        elif name in NONMETAL_TILES:
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


def finish(obj, name, tile, *, bevel=.0012, smooth=True, constant=False, register=True):
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
                  register=register)


def cylindrical_uv(obj, axis):
    mesh = obj.data
    if mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers.active)
    values = []
    for v in mesh.vertices:
        co = v.co
        if axis == 'X':
            along = co.x
        elif axis == 'Y':
            along = co.y
        else:
            along = co.z
        values.append(along)
    lo, hi = min(values), max(values)
    span = max(1e-9, hi - lo)
    uv = mesh.uv_layers.new(name='Atlas')
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            if axis == 'X':
                theta = math.atan2(co.z, co.y)
            elif axis == 'Y':
                theta = math.atan2(co.x, co.z)
            else:
                theta = math.atan2(co.y, co.x)
            along = {'X': co.x, 'Y': co.y, 'Z': co.z}[axis]
            uv.data[li].uv = (theta / math.tau + .5, (along - lo) / span)


def radial_uv(obj):
    mesh = obj.data
    if mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers.active)
    radii = [math.hypot(v.co.x, v.co.y) for v in mesh.vertices]
    hi = max(1e-9, max(radii))
    uv = mesh.uv_layers.new(name='Atlas')
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            theta = math.atan2(co.y, co.x)
            uv.data[li].uv = (theta / math.tau + .5, math.hypot(co.x, co.y) / hi)


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


def axis_cylinder(name, radius, length, center, axis, tile, *, segs=96, bevel=.0012,
                  register=True):
    rotation = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius, depth=length,
                                        location=center, rotation=rotation)
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj = finish(obj, name, tile, bevel=bevel, smooth=True, register=register)
    cylindrical_uv(obj, axis)
    remap_uv(obj, tile)
    return obj


def z_cone(name, radius_bottom, radius_top, z0, z1, tile, *, segs=96, bevel=.0012):
    bpy.ops.mesh.primitive_cone_add(vertices=segs, radius1=radius_bottom,
                                    radius2=radius_top, depth=z1 - z0,
                                    location=(0, 0, (z0 + z1) / 2))
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj = finish(obj, name, tile, bevel=bevel, smooth=True)
    cylindrical_uv(obj, 'Z')
    remap_uv(obj, tile)
    return obj


def torus(name, major, minor, z, axis, tile, *, major_segments=96, minor_segments=20):
    rotations = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0)}
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=(0, 0, z), rotation=rotations[axis],
                                     major_radius=major, minor_radius=minor)
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=0, smooth=True)


def raw_cylinder(radius, length, center, axis, segs=96):
    rotation = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius, depth=length,
                                        location=center, rotation=rotation)
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return obj


def raw_box(center, size):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def apply_boolean(target, cutters, name):
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


def ring_along_x(name, outer, inner, length, center_z, tile, *, segs=128,
                 bevel=.0008):
    """Closed annulus around X; the central bore stays empty."""
    obj = raw_cylinder(outer, length, (0, 0, center_z), 'X', segs)
    cutter = raw_cylinder(inner, length + .04, (0, 0, center_z), 'X', segs)
    apply_boolean(obj, [cutter], name + '_bore')
    obj.name = name
    obj.data.name = name
    apply_bevel(obj, bevel, segments=2, angle=35)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj = finish(obj, name, tile, bevel=0, smooth=True)
    cylindrical_uv(obj, 'X')
    remap_uv(obj, tile)
    return obj


def disc_with_notch(name, radius, z0, z1, tile, notch=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=radius, depth=z1 - z0,
                                        location=(0, 0, (z0 + z1) / 2))
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if notch:
        depth_x, width_y, wz0, wz1 = notch
        cutter = raw_box((radius - depth_x / 2 + .002, 0, (wz0 + wz1) / 2),
                         (depth_x + .006, width_y, wz1 - wz0))
        apply_boolean(obj, [cutter], name + '_notch')
    obj = finish(obj, name, tile, bevel=.0006, smooth=True)
    radial_uv(obj)
    remap_uv(obj, tile)
    return obj


def seat_ring(name, outer, inner, z0, z1, tile):
    obj = raw_cylinder(outer, z1 - z0, (0, 0, (z0 + z1) / 2), 'Z', 128)
    cutter = raw_cylinder(inner, (z1 - z0) + .04, (0, 0, (z0 + z1) / 2), 'Z', 128)
    apply_boolean(obj, [cutter], name + '_bore')
    obj.name = name
    obj.data.name = name
    obj = finish(obj, name, tile, bevel=.0006, smooth=True)
    radial_uv(obj)
    remap_uv(obj, tile)
    return obj


def helix_spring(name, mean_radius, tube_radius, z_top, z_bottom, turns, tile, *,
                 direction=1, segments_per_turn=48, tube_segments=16, phase_deg=0.0):
    dz = z_top - z_bottom
    rings = max(8, int(round(turns * segments_per_turn)))
    phase = math.radians(phase_deg)
    verts, faces, uvs = [], [], []

    def centre(i):
        t = i / rings
        theta = direction * math.tau * turns * t + phase
        return (Vector((mean_radius * math.cos(theta), mean_radius * math.sin(theta),
                        z_top - dz * t)), theta)

    ring_sets = []
    for i in range(rings + 1):
        c, theta = centre(i)
        tangent = Vector((-math.sin(theta) * direction, math.cos(theta) * direction, 0.0))
        radial = Vector((math.cos(theta), math.sin(theta), 0.0))
        normal = tangent.cross(radial).normalized()
        ids = []
        for k in range(tube_segments):
            phi = math.tau * k / tube_segments
            p = c + tube_radius * (math.cos(phi) * radial + math.sin(phi) * normal)
            ids.append(len(verts))
            verts.append(tuple(p))
        ring_sets.append(ids)

    for i in range(rings):
        for k in range(tube_segments):
            k2 = (k + 1) % tube_segments
            a = ring_sets[i][k]
            b = ring_sets[i][k2]
            c = ring_sets[i + 1][k2]
            d = ring_sets[i + 1][k]
            faces.append((a, b, c, d))
            uvs.append([(i / rings, k / tube_segments),
                        (i / rings, (k + 1) / tube_segments),
                        ((i + 1) / rings, (k + 1) / tube_segments),
                        ((i + 1) / rings, k / tube_segments)])

    c0 = len(verts)
    verts.append(tuple(centre(0)[0]))
    c1 = len(verts)
    verts.append(tuple(centre(rings)[0]))
    for k in range(tube_segments):
        k2 = (k + 1) % tube_segments
        faces.append((c0, ring_sets[0][k2], ring_sets[0][k]))
        uvs.append([(.5, .5), (.0, k2 / tube_segments + .5 / tube_segments),
                    (.0, k / tube_segments + .5 / tube_segments)])
        faces.append((c1, ring_sets[rings][k], ring_sets[rings][k2]))
        uvs.append([(.5, .5), (.1, k / tube_segments + .5 / tube_segments),
                    (.1, k2 / tube_segments + .5 / tube_segments)])

    obj = mesh_object(name, verts, faces, uvs)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return finish(obj, name, tile, bevel=0, smooth=True)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def eye_assembly(tag, spec):
    z = spec['center_z']
    ring_along_x(f'{tag}_black_eye', spec['outer_radius'], spec['boss_inner_radius'],
                 spec['axial_length'], z, TILE['paint'], bevel=.0040)
    ring_along_x(f'{tag}_steel_bushing', spec['bushing_outer_radius'],
                 spec['bushing_inner_radius'], spec['axial_length'] + .004, z,
                 TILE['steel'], bevel=.0015)
    ring_along_x(f'{tag}_inner_sleeve', spec['sleeve_outer_radius'],
                 spec['sleeve_inner_radius'], spec['axial_length'] + .008, z,
                 TILE['steel_ground'], bevel=.0010)
    axis_cylinder(f'{tag}_saddle', spec['saddle_radius'], spec['saddle_length'],
                  (0, 0, spec['saddle_center_z']), 'X', TILE['paint'],
                  segs=96, bevel=.0016)
    z0, z1 = spec['neck_z0'], spec['neck_z1']
    if z0 < z1:
        z_cone(f'{tag}_neck', spec['neck_radius_seat'], spec['neck_radius_eye'],
               z0, z1, TILE['paint'], segs=64, bevel=.0012)
    else:
        z_cone(f'{tag}_neck', spec['neck_radius_eye'], spec['neck_radius_seat'],
               z1, z0, TILE['paint'], segs=64, bevel=.0012)


def build_top_seat():
    s = D['top_seat']
    disc_with_notch('top_seat_upper', s['upper_radius'], s['z_upper_mid'],
                    s['z_top'], TILE['seat'],
                    notch=(s['notch_depth_x'], s['notch_width_y'],
                           s['notch_z0'], s['notch_z1']))
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=s['middle_radius'],
                                        depth=s['z_upper_mid'] - s['z_middle_mid'],
                                        location=(0, 0, (s['z_upper_mid'] + s['z_middle_mid']) / 2))
    mid = bpy.context.object
    bpy.context.view_layer.objects.active = mid
    mid.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    finish(mid, 'top_seat_middle', TILE['seat'], bevel=.0008, smooth=True)
    radial_uv(mid)
    remap_uv(mid, TILE['seat'])
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=s['perch_radius'],
                                        depth=s['z_middle_mid'] - s['z_perch'],
                                        location=(0, 0, (s['z_middle_mid'] + s['z_perch']) / 2))
    perch = bpy.context.object
    bpy.context.view_layer.objects.active = perch
    perch.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    finish(perch, 'top_seat_perch', TILE['seat'], bevel=.0008, smooth=True)
    radial_uv(perch)
    remap_uv(perch, TILE['seat'])
    torus('top_seat_lip', s['lip_major'], s['lip_minor'], s['z_top'] - .0005,
          'Z', TILE['seat'])


def build_body():
    b = D['body']
    z_cone('damper_body', b['radius_bottom'], b['radius_top'],
           b['z_bottom'], b['z_top'], TILE['paint'], segs=128, bevel=.0012)
    z_cone('damper_gland', b['gland_radius_bottom'], b['radius_bottom'],
           b['gland_z0'], b['gland_z1'], TILE['paint'], segs=96, bevel=.0010)
    # Wiper ring: a closed torus around the rod, slightly clear of the chrome.
    torus('damper_wiper', b['wiper_major'], b['wiper_minor'], b['wiper_z'],
          'Z', TILE['rubber'], major_segments=96, minor_segments=16)


def build_lower_seat():
    s = D['lower_seat']
    seat_ring('lower_seat_perch', s['perch_radius'], s['inner_radius'],
              s['z_perch_bottom'], s['z_top'], TILE['seat'])
    seat_ring('lower_seat_lower', s['lower_radius'], s['inner_radius'],
              s['z_bottom'], s['z_perch_bottom'], TILE['seat'])
    torus('lower_seat_lip', s['lip_major'], s['lip_minor'], s['z_top'] - .0006,
          'Z', TILE['seat'])
    torus('lower_seat_bump', s['bump_major'], s['bump_minor'], s['bump_z'],
          'Z', TILE['rubber'], major_segments=96, minor_segments=16)


def build_rod():
    r = D['rod']
    axis_cylinder('chrome_rod', r['radius'], r['z_top'] - r['z_bottom'],
                  (0, 0, (r['z_top'] + r['z_bottom']) / 2), 'Z', TILE['chrome'],
                  segs=96, bevel=.0008)


def build_spring():
    s = D['spring']
    helix_spring('coil_spring', s['mean_radius'], s['wire_radius'],
                 s['z_top'], s['z_bottom'], s['turns'], TILE['paint'],
                 direction=s.get('direction', 1),
                 segments_per_turn=s.get('segments_per_turn', 48),
                 tube_segments=s.get('tube_segments', 16),
                 phase_deg=s.get('phase_deg', 0.0))


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
    lo, hi, center, span = bounds([bpy.data.objects[name]])
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
    checks = {'eye_bores': {}, 'spring_axis_open': None}
    all_bvh = {obj.name: make_bvh(obj) for obj in objects}
    for tag, spec in (('top', D['top_eye']), ('bottom', D['bottom_eye'])):
        z = spec['center_z']
        rows = {}
        for offset in (0.0, .010):
            blockers = []
            for name, bvh in all_bvh.items():
                hit = bvh.ray_cast(Vector((-.30, offset, z)),
                                   Vector((1, 0, 0)), .60)
                if hit[0] is not None:
                    blockers.append({'part': name,
                                     'hit': [round(v, 6) for v in hit[0]]})
            rows[f'offset_{int(round(offset * 1000)):02d}mm'] = {
                'blockers': blockers, 'through': not blockers}
        checks['eye_bores'][tag] = rows
    spring = all_bvh['coil_spring']
    hit = spring.ray_cast(Vector((0, 0, -.40)), Vector((0, 0, 1)), .90)
    checks['spring_axis_open'] = hit[0] is None
    return checks


def main():
    global MATERIAL, CAM
    scene = legacy.setup_scene('CPU', 512)
    MATERIAL = atlas_material()
    CAM = init_camera(scene)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .55
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    scene.cycles.use_denoising = bool(cfg['render'].get('denoise', True))
    try:
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 110
        fill.data.shape = 'RECTANGLE'
        fill.data.size = 3.0
        fill.data.size_y = 3.0
        fill.location = (-3.0, 2.5, 2.0)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 220, 3.0),
            ('front reflection', (3.0, 1.4, 1.0), 110, 1.4),
            ('top fill', (0.0, 0.0, 4.0), 65, 2.2),
            ('back rim', (-2.4, 2.8, 2.6), 75, 2.4)):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy = energy
        data.shape = 'RECTANGLE'
        data.size = size
        data.size_y = size
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

    build_top_seat()
    build_body()
    build_spring()
    build_rod()
    build_lower_seat()
    eye_assembly('top', D['top_eye'])
    eye_assembly('bottom', D['bottom_eye'])
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
        # Source-view camera: strict upright and a tilt-compensated pose that
        # follows the uncalibrated photograph's ~8.7 degree lean.
        source = request['source_camera']
        set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
                   shift_x=source.get('shift_x', 0),
                   shift_y=source.get('shift_y', 0))
        draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])
        # Rotate the whole assembly about X for this diagnostic only.
        saved = [(obj, obj.matrix_world.copy()) for obj in objects]
        rot = Matrix.Rotation(math.radians(-8.7), 4, 'X')
        for obj in objects:
            obj.matrix_world = rot @ obj.matrix_world
        bpy.context.view_layer.update()
        set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
                   shift_x=source.get('shift_x', 0),
                   shift_y=source.get('shift_y', 0))
        draw(scene, render_dir / 'new-source-tilt.png', 512, rr['six_view_samples'])
        for obj, matrix in saved:
            obj.matrix_world = matrix
        bpy.context.view_layer.update()

        target = {
            'top-eye': Vector((0, 0, D['top_eye']['center_z'] - center.z)),
            'top-seat': Vector((0, 0, 0.322 - center.z)),
            'spring': Vector((0, 0, 0.10 - center.z)),
            'body-rod': Vector((0, 0, -0.20 - center.z)),
            'lower-seat': Vector((0, 0, -0.295 - center.z)),
            'bottom-eye': Vector((0, 0, D['bottom_eye']['center_z'] - center.z)),
            'overall': center,
        }
        specs = [
            ('top-eye', (1.0, -0.28, 0.22), target['top-eye'], .185),
            ('top-eye-bore', (0.30, -1.0, 0.20), target['top-eye'], .145),
            ('top-seat-notch', (0.45, -1.0, 0.55), target['top-seat'], .245),
            ('spring-coils', (0.30, -1.0, 0.20), target['spring'], .360),
            ('body-rod', (0.55, -1.0, 0.15), target['body-rod'], .280),
            ('lower-seat', (0.45, -1.0, 0.30), target['lower-seat'], .250),
            ('bottom-eye', (1.0, -0.28, -0.22), target['bottom-eye'], .205),
            ('bottom-eye-bore', (0.30, -1.0, -0.20), target['bottom-eye'], .160),
            ('overall-3q', (1.0, -1.0, 0.45), center, span * 1.08),
            ('overall-side', (0.15, -1.0, 0.10), center, span * 1.10),
        ]
        for name, direction, tgt, scale in specs:
            look(scene, direction, tgt, span, ortho_scale=scale)
            draw(scene, render_dir / f'detail-{name}.png',
                 rr['detail_resolution'], rr['detail_samples'])

        seat_names = {'top_seat_upper', 'top_seat_middle', 'top_seat_perch',
                      'top_seat_lip', 'lower_seat_perch', 'lower_seat_lower',
                      'lower_seat_lip', 'lower_seat_bump'}
        set_visible(seat_names)
        seat_objects = [bpy.data.objects[n] for n in seat_names]
        _, _, seat_center, seat_span = bounds(seat_objects)
        look(scene, (0.55, -1.0, 0.85), seat_center, seat_span,
             ortho_scale=seat_span * 1.18)
        draw(scene, render_dir / 'seats-only-top.png', rr['detail_resolution'],
             rr['detail_samples'])
        look(scene, (0.55, -1.0, -0.85), seat_center, seat_span,
             ortho_scale=seat_span * 1.18)
        draw(scene, render_dir / 'seats-only-bottom.png', rr['detail_resolution'],
             rr['detail_samples'])

        registered = {obj.name for obj in objects}
        eye_names = {name for name in bpy.data.objects.keys()
                     if (name.startswith('top_') or name.startswith('bottom_'))
                     and name in registered and '_seat_' not in name
                     and bpy.data.objects[name].type == 'MESH'}
        set_visible(eye_names)
        look(scene, (1.0, -0.30, 0.15), object_center('top_black_eye'), span,
             ortho_scale=.185)
        draw(scene, render_dir / 'top-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])
        look(scene, (1.0, -0.30, -0.15), object_center('bottom_black_eye'), span,
             ortho_scale=.205)
        draw(scene, render_dir / 'bottom-eye-only.png', rr['detail_resolution'],
             rr['detail_samples'])

        set_visible({'coil_spring'})
        look(scene, (0.35, -1.0, 0.25), center, span, ortho_scale=.520)
        draw(scene, render_dir / 'spring-only.png', rr['detail_resolution'],
             rr['detail_samples'])

        set_visible({'damper_body', 'damper_gland', 'damper_wiper', 'chrome_rod'})
        look(scene, (0.55, -1.0, 0.18), center, span, ortho_scale=.420)
        draw(scene, render_dir / 'body-rod-only.png', rr['detail_resolution'],
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
        saved_mats = {}
        for obj in objects:
            saved_mats[obj.name] = [slot.material for slot in obj.material_slots]
            obj.data.materials.clear()
            obj.data.materials.append(clay)
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-clay-front.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        look(scene, (1.0, -1.0, 0.45), center, span)
        draw(scene, render_dir / 'new-clay-3q.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        for obj in objects:
            obj.data.materials.clear()
            for mat in saved_mats[obj.name]:
                obj.data.materials.append(mat)

        main_light = bpy.data.objects['main softbox']
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, directions['front'], center, span)
        draw(scene, render_dir / 'new-light-alternate.png',
             rr['six_view_resolution'], rr['six_view_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()

        set_visible(set(metal_objects))
        for name in ('soft light', 'front reflection', 'top fill', 'back rim'):
            light = bpy.data.objects.get(name)
            if light is not None:
                light.hide_render = True
        metal_target = center
        look(scene, (1.0, -0.55, 0.20), metal_target, span, ortho_scale=span * 1.12)
        draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
        main_light.location = (-3.0, 4.0, 5.0)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        look(scene, (1.0, -0.55, 0.20), metal_target, span, ortho_scale=span * 1.12)
        draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
        main_light.location = (2.4, -2.8, 3.8)
        main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
        for name in ('soft light', 'front reflection', 'top fill', 'back rim'):
            light = bpy.data.objects.get(name)
            if light is not None:
                light.hide_render = False
        set_visible({obj.name for obj in objects})

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
    look(scene, (1.0, -1.0, 0.45), oldcenter, oldspan)
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
        'spring_turns': D['spring']['turns'],
        'render_samples': rr['six_view_samples'],
        'render_resolution': rr['six_view_resolution'],
        'old_render_resolution': rr['old_resolution'],
        'draft': DRAFT,
        'note': ('两个安装环孔、弹簧圈间空隙和弹簧中心轴按真实几何保留；'
                 '衬套分层、座盘缺口、橡胶环和内部结构为按原照推断。'),
    }
    (out / 'worker-result.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'ok', 'parts': len(objects),
                      'self_intersections': data['self_intersection_total'],
                      'eye_bores': checks['eye_bores'],
                      'spring_axis_open': checks['spring_axis_open']},
                     ensure_ascii=False))


main()
