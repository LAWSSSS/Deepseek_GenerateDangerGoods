"""Blender worker for the review-only parametric DN200 ball-valve reconstruction.

The model is assembled from separate, mostly closed mechanical parts.  Reference
photographs are uncalibrated, so the dimensions below are relative proportions
from the front panel; they are not claimed to be measured dimensions.
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
config = request['config']
objects = []
material_for_tile = {}
special_materials = {}


def atlas_materials():
    """Create four relightable PBR materials sharing one observed detail atlas."""
    names = {0: '02 cast steel', 1: '02 machined steel',
             2: '02 blue paint', 3: '02 rusted steel'}
    materials = {}
    for tile, name in names.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        principled = nodes.get('Principled BSDF')

        def texture(key, noncolor=False):
            node = nodes.new('ShaderNodeTexImage')
            node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
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
        normal_map.inputs['Strength'].default_value = .28
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        materials[tile] = material
        material_for_tile[tile] = material
    ball = bpy.data.materials.new('02 ball steel')
    ball.use_nodes = True
    nodes = ball.node_tree.nodes
    links = ball.node_tree.links
    principled = nodes.get('Principled BSDF')
    color = nodes.new('ShaderNodeTexImage')
    color.image = bpy.data.images.load(request['textures']['base_color'], check_existing=True)
    color.image.pack()
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    principled.inputs['Roughness'].default_value = .50
    principled.inputs['Metallic'].default_value = .70
    normal = nodes.new('ShaderNodeTexImage')
    normal.image = bpy.data.images.load(request['textures']['normal'], check_existing=True)
    normal.image.pack()
    normal.image.colorspace_settings.name = 'Non-Color'
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = .20
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    special_materials['ball'] = ball
    return materials


def assign(obj, tile, *, constant_uv=False):
    material = material_for_tile[tile]
    obj.data.materials.clear()
    obj.data.materials.append(material)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    col, top_row = tile % 2, tile // 2
    for loop in uv:
        u, v = loop.uv if not constant_uv else (.5, .5)
        loop.uv = (col * .5 + .01 + (u % 1.0) * .48,
                   (1 - top_row) * .5 + .01 + (v % 1.0) * .48)
    objects.append(obj)
    return obj


def remap_uv(obj, tile, *, constant_uv=False):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    col, top_row = tile % 2, tile // 2
    for loop in uv:
        u, v = loop.uv if not constant_uv else (.5, .5)
        loop.uv = (col * .5 + .01 + (u % 1.0) * .48,
                   (1 - top_row) * .5 + .01 + (v % 1.0) * .48)
    return obj


def finish(obj, name, tile, bevel=.002, smooth='quads', constant_uv=False):
    obj.name = name
    if obj.type != 'MESH':
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.convert(target='MESH')
        obj = bpy.context.object
    if bevel:
        mod = obj.modifiers.new('Small manufactured edge radius', 'BEVEL')
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = 'ANGLE'
        mod.use_clamp_overlap = True
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    for poly in obj.data.polygons:
        poly.use_smooth = smooth == 'all' or (smooth == 'quads' and len(poly.vertices) == 4)
    return assign(obj, tile, constant_uv=constant_uv)


def cylinder(name, radius, z0, z1, tile, *, xy=(0, 0), segments=None, bevel=.0018, smooth='quads'):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments or config['radial_segments'],
                                        radius=radius, depth=z1 - z0,
                                        location=(xy[0], xy[1], (z0 + z1) / 2))
    return finish(bpy.context.object, name, tile, bevel, smooth)


def along(name, start, end, radius, tile, segments=32, bevel=.001, smooth='quads'):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    return finish(obj, name, tile, bevel, smooth)


def hex_along(name, start, end, radius, tile):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    return finish(obj, name, tile, bevel=.0005, smooth=False)


def box(name, center, size, tile, bevel=.002, smooth=False):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, tile, bevel, smooth)


def boolean_cut(target, cutters, name):
    """Apply each cutter separately: joined cutters can fail in the exact solver."""
    for i, cutter in enumerate(cutters):
        mod = target.modifiers.new(f'{name} {i + 1:02}', 'BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.solver = 'EXACT'
        mod.object = cutter
        bpy.context.view_layer.objects.active = target
        target.select_set(True)
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def cutter_cylinder(radius, start, end, segments=64):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    return obj


def ring(name, outer, inner, y0, y1, tile, z_center=None, bevel=.0008):
    z = config['body_center_z'] if z_center is None else z_center
    bpy.ops.mesh.primitive_cylinder_add(vertices=config['radial_segments'], radius=outer,
                                        depth=y1 - y0, rotation=(math.pi / 2, 0, 0),
                                        location=(0, (y0 + y1) / 2, z))
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    cut = cutter_cylinder(inner, (0, y0 - .01, z), (0, y1 + .01, z))
    boolean_cut(obj, [cut], 'Central bore')
    return finish(obj, name, tile, bevel)


def flange(name, y0, y1, raised_name, z_center):
    """End flange with eight through bolt holes and a raised face."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=config['radial_segments'], radius=config['flange_radius'],
                                        depth=y1 - y0, rotation=(math.pi / 2, 0, 0),
                                        location=(0, (y0 + y1) / 2, z_center))
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    cutters = [cutter_cylinder(config['bore_radius'], (0, y0 - .02, z_center), (0, y1 + .02, z_center))]
    for i in range(config['flange_bolt_holes']):
        angle = math.tau * i / config['flange_bolt_holes']
        x = config['flange_bolt_circle'] * math.cos(angle)
        z = z_center + config['flange_bolt_circle'] * math.sin(angle)
        cutters.append(cutter_cylinder(config['flange_hole_radius'],
                                       (x, y0 - .02, z), (x, y1 + .02, z), 48))
    boolean_cut(obj, cutters, 'Eight flange bolt holes')
    finish(obj, name, 1, .0012)
    # Raised machined face on the outside.
    outward = 1 if y1 > 0 else -1
    face0, face1 = (y1, y1 + .006) if outward > 0 else (y1 - .006, y1)
    ring(raised_name, .132, config['bore_radius'], face0, face1, 1, z_center, .0006)
    return obj


def sphere(name, radius, center, tile, scale=(1, 1, 1), segments=128, rings=64, smooth='all'):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings,
                                        radius=radius, location=center)
    obj = bpy.context.object
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, tile, bevel=0, smooth=smooth)


def torus(name, major, minor, center, rotation, tile):
    bpy.ops.mesh.primitive_torus_add(major_segments=config['radial_segments'], minor_segments=24,
                                     major_radius=major, minor_radius=minor,
                                     location=center, rotation=rotation)
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return finish(obj, name, tile, bevel=0, smooth='all')


def prism_xz(name, points, y_center, thickness, tile):
    verts = []
    faces = []
    for y in (y_center - thickness / 2, y_center + thickness / 2):
        verts.extend((x, y, z) for x, z in points)
    n = len(points)
    faces.append(tuple(range(n)))
    faces.append(tuple(range(n, 2 * n))[::-1])
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return finish(obj, name, tile, bevel=.001, smooth=False, constant_uv=True)


def label_font():
    for path in ('C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return bpy.data.fonts.load(path, check_existing=True)
        except Exception:
            pass
    return None


FONT = None


def cast_label(text, z_center, size, tile=0, name=None):
    global FONT
    bpy.ops.object.text_add(location=(0, 0, 0), rotation=(math.pi / 2, 0, math.pi / 2))
    obj = bpy.context.object
    if FONT is None:
        FONT = label_font()
    if FONT:
        obj.data.font = FONT
    obj.data.body = text
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.size = size
    obj.data.extrude = .0035
    obj.data.bevel_depth = 0.0
    obj.data.bevel_resolution = 0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    rx, ry, rz = config['body_radius'], config['body_radius'] * config['body_scale_y'], config['body_radius']
    zc = config['body_center_z']
    for vertex in obj.data.vertices:
        y = vertex.co.y
        z = vertex.co.z + z_center
        term = 1 - (y / ry) ** 2 - ((z - zc) / rz) ** 2
        x = rx * math.sqrt(max(.0004, term))
        vertex.co.x = x + .0015 + vertex.co.x
        vertex.co.z = z
    obj.name = name or ('Cast label ' + text)
    return assign(obj, tile, constant_uv=True)


def build():
    zc = config['body_center_z']
    # Main cast body and the full-length bored neck.
    body = sphere('Cast valve body', config['body_radius'], (0, 0, zc), 0,
                  scale=(1, config['body_scale_y'], 1))
    cut = cutter_cylinder(config['bore_radius'] + .002,
                          (0, -config['flange_outer_y'] - .05, zc),
                          (0, config['flange_outer_y'] + .05, zc))
    boolean_cut(body, [cut], 'Bored body')
    remap_uv(body, 0)
    for sy, suffix in ((1, 'plus'), (-1, 'minus')):
        neck = along(f'Bored neck {suffix}', (0, sy * .140, zc),
                     (0, sy * config['flange_inner_y'], zc),
                     config['neck_radius'], 0, 128, .0015)
        cut = cutter_cylinder(config['bore_radius'],
                              (0, sy * .140, zc), (0, sy * config['flange_outer_y'], zc))
        boolean_cut(neck, [cut], 'Neck bore')
        remap_uv(neck, 0)
        neck.data.polygons.foreach_set('use_smooth', [True] * len(neck.data.polygons))

    # Split body flange and its twelve through bolts.
    split_y0, split_y1 = config['split_flange_major_y'], config['split_flange_minor_y']
    bpy.ops.mesh.primitive_cylinder_add(vertices=config['radial_segments'],
                                        radius=config['split_flange_radius'],
                                        depth=split_y1 - split_y0, rotation=(math.pi / 2, 0, 0),
                                        location=(0, (split_y0 + split_y1) / 2, zc))
    split = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    cutters = [cutter_cylinder(.138, (0, split_y0 - .02, zc), (0, split_y1 + .02, zc))]
    for i in range(config['split_flange_bolts']):
        angle = math.tau * i / config['split_flange_bolts']
        x = config['split_bolt_circle'] * math.cos(angle)
        z = zc + config['split_bolt_circle'] * math.sin(angle)
        cutters.append(cutter_cylinder(.0095, (x, split_y0 - .02, z), (x, split_y1 + .02, z), 32))
    boolean_cut(split, cutters, 'Split flange bolt holes')
    finish(split, 'Body split flange', 0, .0015)
    for i in range(config['split_flange_bolts']):
        angle = math.tau * i / config['split_flange_bolts']
        x = config['split_bolt_circle'] * math.cos(angle)
        z = zc + config['split_bolt_circle'] * math.sin(angle)
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=.014, depth=.016,
                                            rotation=(math.pi / 2, 0, 0),
                                            location=(x, split_y0 - .008, z))
        finish(bpy.context.object, f'Split flange bolt head {i + 1:02}', 0,
               bevel=.0004, smooth=False)
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=.0075, depth=.052,
                                            rotation=(math.pi / 2, 0, 0),
                                            location=(x, (split_y0 + split_y1) / 2, z))
        finish(bpy.context.object, f'Split flange bolt shank {i + 1:02}', 3,
               bevel=.0003, smooth='all')
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=.013, depth=.016,
                                            rotation=(math.pi / 2, 0, 0),
                                            location=(x, split_y1 + .008, z))
        finish(bpy.context.object, f'Split flange nut {i + 1:02}', 0,
               bevel=.0004, smooth=False)

    # End flanges with real through holes.
    flange('End flange plus', config['flange_inner_y'], config['flange_outer_y'],
           'Raised face plus', zc)
    flange('End flange minus', -config['flange_outer_y'], -config['flange_inner_y'],
           'Raised face minus', zc)
    ring('Seat ring plus', .112, .100, .268, .280, 1, zc, .0006)
    ring('Seat ring minus', .112, .100, -.280, -.268, 1, zc, .0006)

    # Closed ball: visible as a dome through each bored end.
    ball = sphere('Ball', config['ball_radius'], (0, 0, zc), 1, segments=96, rings=48)
    ball.data.materials.clear()
    ball.data.materials.append(special_materials['ball'])

    # Lower cast boss with a blind hex socket.
    cylinder('Bottom boss', .075, config['bottom_boss_bottom'], config['bottom_boss_top'], 0,
             xy=(0, 0), bevel=.002)
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=.034, depth=.034,
                                        location=(0, 0, .035))
    cut = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    boss = bpy.data.objects.get('Bottom boss')
    boolean_cut(boss, [cut], 'Blind hex socket')
    remap_uv(boss, 0)
    boss.data.polygons.foreach_set('use_smooth', [False] * len(boss.data.polygons))

    # Stem stack and blue gearbox.
    along('Stem', (0, 0, .44), (0, 0, config['stem_top']), .038, 1, 96, .0015)
    cylinder('Stem collar', .048, .470, .492, 1, bevel=.001)
    cylinder('Blue adapter ring', .075, config['stem_top'], config['adapter_top'], 2, bevel=.002)
    for i in range(6):
        angle = math.tau * i / 6
        x, y = .060 * math.cos(angle), .060 * math.sin(angle)
        hex_along(f'Adapter bolt {i + 1:02}', (x, y, config['adapter_top'] - .002),
                  (x, y, config['adapter_top'] + .012), .007, 1)
    cylinder('Blue pedestal', .060, config['adapter_top'], config['pedestal_top'], 2, bevel=.003)
    box('Gearbox housing', (0, 0, (config['gear_bottom'] + config['gear_top']) / 2),
        (2 * config['gear_half_x'], 2 * config['gear_half_y'],
         config['gear_top'] - config['gear_bottom']), 2, bevel=.012)
    along('Gearbox round cover', (config['gear_half_x'], 0, config['gear_center_z']),
          (config['gear_half_x'] + .018, 0, config['gear_center_z']), .066, 2, 96, .0015)
    for i in range(6):
        angle = math.tau * i / 6
        y, z = .047 * math.cos(angle), config['gear_center_z'] + .047 * math.sin(angle)
        hex_along(f'Cover bolt {i + 1:02}',
                  (config['gear_half_x'] + .017, y, z),
                  (config['gear_half_x'] + .028, y, z), .0065, 1)
    for i, (x, y) in enumerate(((-.065, -.060), (.065, -.060), (-.065, .060), (.065, .060))):
        hex_along(f'Gearbox top bolt {i + 1:02}', (x, y, config['gear_top'] - .002),
                  (x, y, config['gear_top'] + .011), .009, 1)
    along('Gearbox side shaft', (0, -.090, config['gear_center_z']),
          (0, -.155, config['gear_center_z']), .020, 2, 64, .001)
    along('Gearbox side shaft tip', (0, -.155, config['gear_center_z']),
          (0, -.170, config['gear_center_z']), .012, 1, 32, .0008)

    # Three-spoke handwheel in the YZ plane, offset toward +X.
    wx = config['wheel_center_x']
    wz = config['gear_center_z']
    torus('Handwheel rim', config['wheel_radius'], config['wheel_tube'], (wx, 0, wz),
          (0, math.pi / 2, 0), 2)
    along('Handwheel hub', (.155, 0, wz), (.212, 0, wz), .034, 2, 96, .0015)
    for i, angle in enumerate((math.radians(30), math.radians(150), math.radians(270))):
        y0, z0 = .026 * math.cos(angle), .026 * math.sin(angle)
        y1, z1 = .196 * math.cos(angle), .196 * math.sin(angle)
        along(f'Handwheel spoke {i + 1}', (wx, y0, wz + z0), (wx, y1, wz + z1),
              .0145, 2, 48, .001)
    along('Wheel drive shaft', (config['gear_half_x'] + .018, 0, wz),
          (0.160, 0, wz), .022, 1, 64, .001)
    along('Wheel center cap', (0.208, 0, wz), (0.222, 0, wz), .020, 1, 64, .001)
    along('Wheel center bolt', (0.221, 0, wz), (0.228, 0, wz), .010, 1, 6, .0005)

    # Cast lettering and the small upper-front cast lug.
    cast_label('DN200', .315, .045)
    cast_label('25', .250, .052)
    cast_label('WCB', .185, .048)
    for i, y in enumerate((-.036, -.012, .012, .036), 1):
        z = .140
        term = 1 - (y / (config['body_radius'] * config['body_scale_y'])) ** 2 - ((z - zc) / config['body_radius']) ** 2
        x = config['body_radius'] * math.sqrt(max(.0004, term)) - .002
        box(f'Cast square {i}', (x, y, z), (.014, .016, .016), 0, bevel=.002)
    box('Upper front cast lug', (config['body_radius'] - .010, 0, .405),
        (.032, .030, .052), 0, bevel=.006)

    # Two support saddles with real base mounting holes.
    for side, label in ((1, 'plus'), (-1, 'minus')):
        fy = side * config['foot_y']
        bpy.ops.mesh.primitive_cube_add(size=1, location=(.020, fy, .008))
        plate = bpy.context.object
        plate.scale = (.150, .170, .016)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        cutters = []
        for dy in (-.055, .055):
            cutters.append(cutter_cylinder(.011, (.020, fy + dy, -.01), (.020, fy + dy, .03), 32))
        boolean_cut(plate, cutters, 'Foot plate mounting holes')
        finish(plate, f'Foot base {label}', 1, .0015)
        for ux, uname in ((-.055, 'inner'), (.095, 'outer')):
            box(f'Foot upright {label} {uname}', (ux, fy, .076), (.014, .070, .120), 1, bevel=.001)
            prism_xz(f'Foot gusset {label} {uname}',
                     [(ux + (-.001 if uname == 'inner' else .001), .020),
                      (ux + (-.001 if uname == 'inner' else .001), .102),
                      (ux + (.035 if uname == 'inner' else -.035), .020)],
                     fy, .012, 1)
            hex_along(f'Foot top bolt {label} {uname}', (ux - (0.02 if uname == 'inner' else -.02), fy, .120),
                      (ux + (.02 if uname == 'inner' else -.02), fy, .120), .011, 3)
        box(f'Foot top strap {label}', (.020, fy, .120), (.190, .060, .030), 1, bevel=.002)

    # Match the existing library's centred Z range.
    for obj in objects:
        obj.location.z -= .5


def camera(scene, direction, center, span, scale=1.27):
    direction = Vector(direction).normalized()
    cam = legacy.make_camera(scene, Vector(center) + direction * span * 5, center,
                             ortho_scale=span * scale)
    if abs(direction.z) > .999:
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.rotation_quaternion @ Quaternion((0, 0, 1), math.pi / 2)
    return cam


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


def intersecting_faces(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
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


def draw(scene, path, *, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


scene = legacy.setup_scene('OPTIX', 512)
scene.render.threads_mode = 'AUTO'
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .60
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, width, height in (
        ('large softbox', (2.4, -2.8, 3.8), 170, 3, 3),
        ('fill softbox', (-3, 2.5, 2), 95, 3, 3),
        ('front reflection', (3.0, 1.4, 1), 95, 1.4, 3.2),
        ('side fill minus Y', (0, -2.5, -.245), 40, .6, .6),
        ('side fill plus Y', (0, 2.5, -.245), 40, .6, .6)):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = width
    data.size_y = height
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()
atlas_materials()
build()
bpy.context.view_layer.update()
object_names = [obj.name for obj in objects]
low, high, center, span = bounds(objects)
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'), export_format='GLB', use_selection=True,
                          export_yup=False, export_apply=True)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))
# Rendering from the saved, reloaded file makes the delivered .blend the single
# source of truth for lighting and materials.  Without this reload, the
# in-memory scene rendered the deep bore as an unlit disc even though the saved
# file rendered correctly; the cause was not reproduced in an isolated reload.
bpy.ops.wm.open_mainfile(filepath=str(out / 'model.blend'))
scene = bpy.context.scene
scene.render.threads_mode = 'AUTO'
objects = [bpy.data.objects[name] for name in object_names]
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
parts = [{'name': o.name, 'vertices': len(o.data.vertices), 'faces': len(o.data.polygons)} for o in objects]
intersections = {o.name: intersecting_faces(o) for o in objects}
render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
directions = {'front': (1, 0, .20), 'back': (-1, 0, .20), 'left': (0, -1, 0.0),
              'right': (0, 1, 0.0), 'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 96
for name, direction in directions.items():
    cam = camera(scene, direction, center, span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
source = request['source_camera']
cam = legacy.make_camera(scene, (1.9, 0, .30), (0, 0, 0), fov=source['fov_deg'])
cam.data.shift_x = source.get('shift_x', 0)
cam.data.shift_y = source.get('shift_y', 0)
draw(scene, render_dir / 'new-source.png', resolution=512, samples=samples)
bpy.data.objects.remove(cam, do_unlink=True)

# Same-light material close-up of the new model.
cam = camera(scene, (3, -.4, .25), Vector((.18, 0, -.20)), .32)
draw(scene, render_dir / 'new-material-close.png', resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

if not request['draft']:
    for i in range(24):
        angle = math.tau * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .15), center, span)
        draw(scene, render_dir / f'turn-{i:02}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)
    for name, target, direction, scale in (
            ('wheel', (config['wheel_center_x'], 0, config['gear_center_z'] - .5), (1.4, -1.1, .6), .58),
            ('flange', (0, config['flange_outer_y'], config['body_center_z'] - .5), (0, 1, 0), .50),
            ('body', (.17, 0, config['body_center_z'] - .5 + .02), (1, -.18, .10), .48),
            ('foot', (.03, config['foot_y'], .008 - .5), (1.2, -1.4, .8), .55),
            ('bottom', (0, 0, config['bottom_boss_bottom'] - .5), (.5, -.8, -1), .46),
            ('surface', (.18, 0, -.20), (3, -.4, .25), .32)):
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    # Isolated opening checks: three wheel windows, eight flange bolt holes
    # plus the central bore, and two holes in each foot base plate.
    wheel_names = {'Handwheel rim', 'Handwheel hub', 'Wheel center cap', 'Wheel center bolt'} | {
        f'Handwheel spoke {i}' for i in range(1, 4)}
    for obj in objects:
        obj.hide_render = obj.name not in wheel_names
    cam = camera(scene, (1, 0, 0), Vector((config['wheel_center_x'], 0, config['gear_center_z'] - .5)),
                 .55, 1.12)
    draw(scene, render_dir / 'wheel-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    flange_names = {'End flange plus', 'Raised face plus'}
    for obj in objects:
        obj.hide_render = obj.name not in flange_names
    cam = camera(scene, (0, 1, 0), Vector((0, config['flange_outer_y'], config['body_center_z'] - .5)),
                 .48, 1.10)
    draw(scene, render_dir / 'flange-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = obj.name != 'Foot base plus'
    cam = camera(scene, (0, 0, -1), Vector((.030, config['foot_y'], .008 - .5)), .25, 1.15)
    draw(scene, render_dir / 'foot-plate-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    # A second orientation of the broad key light checks that highlights move.
    key = bpy.data.objects['large softbox']
    key.location = (-3, 4, 5)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.4, -2.8, 3.8)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
_, _, oldcenter, oldspan = bounds([old])
for name, direction in directions.items():
    cam = camera(scene, direction, oldcenter, oldspan)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
old.location += center - oldcenter
bpy.context.view_layer.update()
cam = camera(scene, (3, -.4, .25), Vector((.18, 0, -.20)), .32)
draw(scene, render_dir / 'old-material-close.png', resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects for p in o.data.polygons),
        'texture_size': 2048, 'materials': sorted(material_for_tile),
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH 同部件不共享顶点的面交叠；部件之间的装配交叠未判定',
        'geometry_basis': 'six-view panels, uncalibrated', 'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': '左视相机在 -Y、右视在 +Y，以匹配参考面板中手轮的左右位置；参考图未标定',
        'isolated_checks': {'handwheel_openings_expected': 3, 'end_flange_interior_holes_expected': 9,
                            'foot_plate_holes_expected': 2}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
