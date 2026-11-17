"""Blender worker for the review-only parametric ASAHI P210 pillow-block candidate.

The reference photographs are uncalibrated.  Public UCP210/P210/UC210 catalog
dimensions are used only as a proportion sanity check; no dimension here is
claimed to be a measurement of the photographed object.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Matrix, Vector, Quaternion
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
    names = {0: '03 blue paint', 1: '03 bearing steel',
             2: '03 dark steel', 3: '03 rusted cast'}
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
        normal_map.inputs['Strength'].default_value = {0: .22, 1: .16, 2: .24, 3: .35}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        material_for_tile[tile] = material

    chrome = bpy.data.materials.new('03 chrome zerk')
    chrome.use_nodes = True
    principled = chrome.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.72, .74, .77, 1)
    principled.inputs['Metallic'].default_value = 1.0
    principled.inputs['Roughness'].default_value = .14
    special_materials['chrome'] = chrome

    rubber = bpy.data.materials.new('03 black rubber seal')
    rubber.use_nodes = True
    principled = rubber.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.012, .012, .012, 1)
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = .72
    special_materials['rubber'] = rubber

    screw = bpy.data.materials.new('03 black set screw')
    screw.use_nodes = True
    principled = screw.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.010, .010, .011, 1)
    principled.inputs['Metallic'].default_value = .72
    principled.inputs['Roughness'].default_value = .30
    special_materials['screw'] = screw

    label = bpy.data.materials.new('03 label paint')
    label.use_nodes = True
    principled = label.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.075, .150, .250, 1)
    principled.inputs['Metallic'].default_value = .03
    principled.inputs['Roughness'].default_value = .60
    special_materials['label'] = label

    bare = bpy.data.materials.new('03 bare machined pad')
    bare.use_nodes = True
    principled = bare.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.36, .36, .35, 1)
    principled.inputs['Metallic'].default_value = .82
    principled.inputs['Roughness'].default_value = .48
    special_materials['bare'] = bare


def remap_uv(obj, tile, constant_uv=False):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    col, top_row = tile % 2, tile // 2
    for loop in uv:
        u, v = loop.uv if not constant_uv else (.5, .5)
        loop.uv = (col * .5 + .01 + (u % 1.0) * .48,
                   (1 - top_row) * .5 + .01 + (v % 1.0) * .48)


def box_uv(obj, scale=1.1):
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name='Atlas')
    uv = mesh.uv_layers.active.data
    for poly in mesh.polygons:
        normal = poly.normal
        axis = max(range(3), key=lambda i: abs(normal[i]))
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            if axis == 0:
                u, v = co.y / scale, co.z / scale
            elif axis == 1:
                u, v = co.x / scale, co.z / scale
            else:
                u, v = co.x / scale, co.y / scale
            uv[loop_index].uv = (u, v)


def assign(obj, tile, *, constant_uv=False, boxed=False):
    obj.data.materials.clear()
    obj.data.materials.append(material_for_tile[tile])
    if boxed:
        box_uv(obj)
    remap_uv(obj, tile, constant_uv=constant_uv)
    objects.append(obj)
    return obj


def assign_special(obj, key):
    obj.data.materials.clear()
    obj.data.materials.append(special_materials[key])
    objects.append(obj)
    return obj


def finish(obj, name, tile, bevel=.002, smooth='quads', constant_uv=False, boxed=False):
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
    return assign(obj, tile, constant_uv=constant_uv, boxed=boxed)


def finish_special(obj, name, key, bevel=.002, smooth='all'):
    obj.name = name
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
    return assign_special(obj, key)


def cylinder(name, radius, z0, z1, tile, *, xy=(0, 0), segments=None, bevel=.0018, smooth='quads'):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments or config['radial_segments'],
                                        radius=radius, depth=z1 - z0,
                                        location=(xy[0], xy[1], (z0 + z1) / 2))
    return finish(bpy.context.object, name, tile, bevel, smooth)


def along(name, start, end, radius, tile, segments=32, bevel=.001, smooth='quads', special=None):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    if special:
        return finish_special(obj, name, special, bevel, smooth)
    return finish(obj, name, tile, bevel, smooth)


def hex_along(name, start, end, radius, tile, bevel=.0005, special=None):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    if special:
        return finish_special(obj, name, special, bevel, smooth=False)
    return finish(obj, name, tile, bevel, smooth=False)


def box(name, center, size, tile, bevel=.002, smooth=False, special=None, boxed=True):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if special:
        return finish_special(obj, name, special, bevel, 'quads' if smooth else False)
    return finish(obj, name, tile, bevel, smooth, boxed=boxed)


def boolean_cut(target, cutters, name):
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


def prism_yz(name, points, x_center, thickness, tile, bevel=.002):
    verts = []
    faces = []
    n = len(points)
    x0, x1 = x_center - thickness / 2, x_center + thickness / 2
    for x in (x0, x1):
        verts.extend((x, y, z) for y, z in points)
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
    return finish(obj, name, tile, bevel=bevel, smooth=False, boxed=True)


def capsule_prism(name, width_x, length_y, z0, z1, center=(0, 0), segments=40):
    r = width_x / 2
    half = max(0.0, (length_y - width_x) / 2)
    outline = []
    for i in range(segments + 1):
        angle = math.pi * i / segments
        outline.append((r * math.cos(angle), half + r * math.sin(angle)))
    for i in range(segments + 1):
        angle = math.pi + math.pi * i / segments
        outline.append((r * math.cos(angle), -half + r * math.sin(angle)))
    n = len(outline)
    verts = []
    for z in (z0, z1):
        verts.extend((center[0] + x, center[1] + y, z) for x, y in outline)
    faces = [tuple(range(n)), tuple(range(n, 2 * n))[::-1]]
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
    return obj


def label_font():
    for path in ('C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return bpy.data.fonts.load(path, check_existing=True)
        except Exception:
            pass
    return None


FONT = None


def cast_label(text, y, z, size, name):
    global FONT
    bpy.ops.object.text_add(location=(0, 0, 0))
    obj = bpy.context.object
    if FONT is None:
        FONT = label_font()
    if FONT:
        obj.data.font = FONT
    obj.data.body = text
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.size = size
    obj.data.extrude = .007
    obj.data.bevel_depth = 0.0
    obj.data.bevel_resolution = 0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.object
    # local x -> world +Y, local y -> world +Z, local z (extrude) -> world +X
    obj.data.transform(Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0))).to_4x4())
    obj.data.update()
    obj.location = (config['base_depth'] / 2 + .002, y, z)
    obj.name = name
    return assign_special(obj, 'label')


def build():
    axis_z = config['axis_height']
    base_t = config['base_thickness']
    base_depth = config['base_depth']
    base_length = config['base_length']
    half = base_length / 2
    bolt_y = config['bolt_spacing'] / 2
    slot_w = config['slot_width']
    slot_l = config['slot_length']

    # Base plate with a cast underside arch and two elongated mounting slots.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, base_t / 2))
    base = bpy.context.object
    base.scale = (base_depth, base_length, base_t)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    bevel = base.modifiers.new('Base edge radius', 'BEVEL')
    bevel.width = .030
    bevel.segments = 3
    bevel.limit_method = 'ANGLE'
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    cutters = [cutter_cylinder(.55, (-base_depth / 2 - .05, 0, -.43),
                               (base_depth / 2 + .05, 0, -.43), 80)]
    for side in (-1, 1):
        cutters.append(capsule_prism('Slot cutter', slot_w, slot_l, -.08, base_t + .08,
                                     center=(0, side * bolt_y)))
    boolean_cut(base, cutters, 'Base slot and underside arch')
    finish(base, 'Base plate', 0, bevel=.0015, boxed=True)

    # Raised cast collars around the two mounting slots.
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        collar = capsule_prism(f'Foot collar {suffix}', .30, .35, base_t - .004, base_t + .020,
                               center=(0, side * bolt_y))
        bpy.context.view_layer.objects.active = collar
        collar.select_set(True)
        mod = collar.modifiers.new('Collar edge radius', 'BEVEL')
        mod.width = .006
        mod.segments = 2
        mod.limit_method = 'ANGLE'
        bpy.ops.object.modifier_apply(modifier=mod.name)
        cutter = capsule_prism('Collar slot cutter', slot_w, slot_l, -.08, base_t + .08,
                               center=(0, side * bolt_y))
        boolean_cut(collar, [cutter], 'Collar slot')
        finish(collar, f'Foot collar {suffix}', 0, bevel=0, smooth=False, boxed=True)

        pad = box(f'Bottom machined pad {suffix}', (0, side * bolt_y, -.003),
                  (.42, .34, .010), 0, bevel=.002, smooth=False, special='bare')
        cutter = capsule_prism('Pad slot cutter', slot_w, slot_l, -.08, .08,
                               center=(0, side * bolt_y))
        boolean_cut(pad, [cutter], 'Pad slot')
        # Re-assign after the boolean rebuilds the mesh.
        pad.data.materials.clear()
        pad.data.materials.append(special_materials['bare'])

    # Two low saddles blend the cast feet into the housing.
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        box(f'Foot saddle {suffix}', (0, side * .43, base_t + .012),
            (.36, .20, .030), 0, bevel=.008, smooth=False, boxed=True)

    # Housing arch profile in YZ, extruded along the bearing (X) axis.
    r = config['dome_radius']
    points = [(-config['body_half_width'], base_t - .004),
              (-.575, .30), (-.535, .41)]
    for i in range(81):
        angle = math.pi - math.pi * i / 80
        points.append((r * math.cos(angle), axis_z + r * math.sin(angle)))
    points += [(.535, .41), (.575, .30), (config['body_half_width'], base_t - .004)]
    body = prism_yz('Housing body', points, 0, config['body_thickness'], 0, bevel=.070)
    cut = cutter_cylinder(config['housing_bore'],
                          (-config['body_thickness'], 0, axis_z),
                          (config['body_thickness'], 0, axis_z), 96)
    boolean_cut(body, [cut], 'Housing bore')
    # Keep the blue atlas after the boolean.
    body.data.materials.clear()
    body.data.materials.append(material_for_tile[0])
    box_uv(body)
    remap_uv(body, 0)

    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=config['boss_outer'],
                                        depth=config['boss_thickness'],
                                        rotation=(0, math.pi / 2, 0),
                                        location=(0, 0, axis_z))
    boss = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    cut = cutter_cylinder(config['housing_bore'], (-.3, 0, axis_z), (.3, 0, axis_z), 96)
    boolean_cut(boss, [cut], 'Boss bore')
    finish(boss, 'Housing bearing boss', 0, bevel=.004, boxed=True)

    # Bearing insert: dark outer ring, polished wide inner ring, rubber seals.
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=config['outer_ring_outer'],
                                        depth=config['outer_ring_width'],
                                        rotation=(0, math.pi / 2, 0),
                                        location=(0, 0, axis_z))
    outer = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    cut = cutter_cylinder(config['outer_ring_inner'], (-.2, 0, axis_z), (.2, 0, axis_z), 96)
    boolean_cut(outer, [cut], 'Outer ring bore')
    finish(outer, 'Bearing outer ring', 2, bevel=.004, boxed=True)

    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=config['inner_ring_outer'],
                                        depth=config['inner_ring_width'],
                                        rotation=(0, math.pi / 2, 0),
                                        location=(0, 0, axis_z))
    inner = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bevel = inner.modifiers.new('Inner ring chamfer', 'BEVEL')
    bevel.width = .025
    bevel.segments = 2
    bevel.limit_method = 'ANGLE'
    bpy.context.view_layer.objects.active = inner
    inner.select_set(True)
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    cutters = [cutter_cylinder(config['inner_ring_inner'], (-.3, 0, axis_z), (.3, 0, axis_z), 96)]
    for angle_deg in config['set_screw_angles_deg']:
        angle = math.radians(angle_deg)
        direction = Vector((0, math.cos(angle), math.sin(angle)))
        cutters.append(cutter_cylinder(config['set_screw_hole_radius'],
                                       Vector((config['set_screw_x'], 0, axis_z)) + direction * .17,
                                       Vector((config['set_screw_x'], 0, axis_z)) + direction * .345,
                                       40))
    boolean_cut(inner, cutters, 'Bearing bore and set-screw holes')
    finish(inner, 'Bearing inner ring', 1, bevel=0, boxed=True)

    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        x = side * (config['outer_ring_width'] / 2 + .004)
        bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=.338, depth=.014,
                                            rotation=(0, math.pi / 2, 0),
                                            location=(x, 0, axis_z))
        seal = bpy.context.object
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        cut = cutter_cylinder(.315, (x - .03, 0, axis_z), (x + .03, 0, axis_z), 64)
        boolean_cut(seal, [cut], 'Seal bore')
        finish_special(seal, f'Bearing seal {suffix}', 'rubber', bevel=0, smooth='all')

    # Two black set-screw socket caps sit on the front chamfer of the inner
    # ring, matching the visible sockets in the reference.  The threaded holes
    # themselves are the radial cuts above.  The chamfer placement is inferred.
    for i, angle_deg in enumerate(config['set_screw_angles_deg'], 1):
        angle = math.radians(angle_deg)
        direction = Vector((0, math.cos(angle), math.sin(angle)))
        surface = Vector((.2138, 0, axis_z)) + direction * .3025
        normal = (Vector((1, 0, 0)) + direction).normalized()
        head = along(f'Set screw {i}', surface - normal * .004, surface + normal * .006,
                     config['set_screw_head_radius'], 0, segments=48, bevel=.0008,
                     smooth='quads', special='screw')
        socket_start = surface + normal * .004
        socket_end = surface + normal * .009
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=.0105,
                                            depth=(socket_end - socket_start).length,
                                            location=(socket_start + socket_end) / 2)
        socket = bpy.context.object
        socket.rotation_euler = (socket_end - socket_start).to_track_quat('Z', 'Y').to_euler()
        boolean_cut(head, [socket], 'Set screw hex socket')
        head.data.materials.clear()
        head.data.materials.append(special_materials['screw'])

    # Grease nipple: cast boss, hex body, small stem and domed tip.
    cylinder('Zerk boss', .055, 1.000 - .006, 1.016, 0, xy=(config['zerk_x'], 0),
             segments=64, bevel=.002)
    hex_along('Zerk hex body', (config['zerk_x'], 0, 1.014), (config['zerk_x'], 0, 1.046),
              .042, 0, bevel=.0008, special='chrome')
    along('Zerk stem', (config['zerk_x'], 0, 1.044), (config['zerk_x'], 0, 1.078),
          .016, 0, segments=32, bevel=.0008, smooth='all', special='chrome')
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=.022,
                                        location=(config['zerk_x'], 0, 1.078))
    tip = bpy.context.object
    tip.scale = (1, 1, 1.1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    finish_special(tip, 'Zerk domed tip', 'chrome', bevel=0, smooth='all')

    # Embossed brand and model on the front faces of the base wings.
    cast_label('ASAHI', -config['label_y'], config['label_height'], config['label_size'],
               'Cast label ASAHI')
    cast_label('P210', config['label_y'], config['label_height'], config['label_size'],
               'Cast label P210')

    # Match the library's centred Z range.
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
bpy.ops.wm.open_mainfile(filepath=str(out / 'model.blend'))
scene = bpy.context.scene
scene.render.threads_mode = 'AUTO'
objects = [bpy.data.objects[name] for name in object_names]
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
parts = [{'name': o.name, 'vertices': len(o.data.vertices), 'faces': len(o.data.polygons)}
         for o in objects]
intersections = {o.name: intersecting_faces(o) for o in objects}
render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
directions = {'front': (1, 0, .30), 'back': (-1, 0, .30), 'left': (0, -1, .35),
              'right': (0, 1, .35), 'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 96
for name, direction in directions.items():
    cam = camera(scene, direction, center, span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

# Same-light material close-up of the new model.
cam = camera(scene, (3, -.4, .25), Vector((0, 0, .0)), .55)
draw(scene, render_dir / 'new-material-close.png', resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

if not request['draft']:
    for i in range(24):
        angle = math.tau * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .15), center, span)
        draw(scene, render_dir / f'turn-{i:02}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)
    detail_cameras = (
        ('bearing', (0, 0, .0), (1, 0, 0), .95),
        ('set-screw', (config['set_screw_x'], 0, .0), (.9, -1, .4), .45),
        ('zerk', (config['zerk_x'], 0, config['zerk_top'] - .5), (1, -1, 1.2), .32),
        ('label-asahi', (0, -config['label_y'], config['label_height'] - .5),
         (1, -.15, .05), .55),
        ('label-p210', (0, config['label_y'], config['label_height'] - .5),
         (1, -.15, .05), .45),
        ('slot', (0, -config['bolt_spacing'] / 2, config['base_thickness'] - .5),
         (1, -.4, .8), .50),
        ('bottom', (0, 0, -.5), (.6, -.5, -1), 1.55),
        ('surface', (.15, -.35, .28 - .5), (2, -1, 1), .48),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    # Isolated opening checks.
    bearing_names = {'Bearing outer ring', 'Bearing inner ring',
                     'Bearing seal minus', 'Bearing seal plus'}
    for obj in objects:
        obj.hide_render = obj.name not in bearing_names
    cam = camera(scene, (1, 0, 0), Vector((0, 0, 0)), .92, 1.12)
    draw(scene, render_dir / 'bearing-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)

    base_names = {'Base plate', 'Foot collar minus', 'Foot collar plus',
                  'Bottom machined pad minus', 'Bottom machined pad plus'}
    for obj in objects:
        obj.hide_render = obj.name not in base_names
    cam = camera(scene, (0, 0, 1), Vector((0, 0, config['base_thickness'] - .5)),
                 config['base_length'], 1.08)
    draw(scene, render_dir / 'base-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)

    housing_names = {'Housing body', 'Housing bearing boss', 'Zerk boss'}
    for obj in objects:
        obj.hide_render = obj.name not in housing_names
    cam = camera(scene, (1, 0, 0), Vector((0, 0, 0)), 1.0, 1.12)
    draw(scene, render_dir / 'housing-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    # A second key-light orientation checks that metal highlights move.
    key = bpy.data.objects['large softbox']
    key.location = (-3, 4, 5)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (3, -.4, .25), Vector((0, 0, .0)), .55)
    draw(scene, render_dir / 'new-metal-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.4, -2.8, 3.8)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Render the new model into the source-photo framing (diagnostic only).
for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
_, _, oldcenter, oldspan = bounds([old])
old.location += center - oldcenter
bpy.context.view_layer.update()
for obj in objects:
    obj.hide_render = False
old.hide_render = True
source = request['source_camera']
factor = span / max(1e-6, oldspan)
cam = legacy.make_camera(scene, (1.9 * factor, 0, center.z),
                         (0, 0, center.z), fov=source['fov_deg'])
cam.data.shift_x = source.get('shift_x', 0)
cam.data.shift_y = source.get('shift_y', 0)
draw(scene, render_dir / 'new-source.png', resolution=512, samples=samples)
bpy.data.objects.remove(cam, do_unlink=True)

for obj in objects:
    obj.hide_render = True
old.hide_render = False
for name, direction in directions.items():
    cam = camera(scene, direction, center, oldspan)
    draw(scene, render_dir / f'old-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (3, -.4, .25), Vector((0, 0, .0)), .55)
draw(scene, render_dir / 'old-material-close.png', resolution=640 if request['draft'] else 1024,
     samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects for p in o.data.polygons),
        'texture_size': 2048, 'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': 'six-view panels plus public catalog dimension ratios, uncalibrated',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': 'Left camera at -Y and right at +Y; the reference is uncalibrated.',
        'source_camera_note': 'source-photo framing distance scaled by new/old span; diagnostic only',
        'source_camera_factor': factor,
        'isolated_checks': {'bearing_bore_expected': 1, 'base_slots_expected': 2,
                            'housing_bore_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
