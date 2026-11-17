"""Blender worker for the review-only SN513 split plummer-block candidate (item 04).

The user six-view sheet is uncalibrated.  Public SN 513 / SNL 513-611 catalog
proportions are used only as a plausibility check.  No dimension in this file
is presented as a measurement of the photographed housing.
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
    names = {0: '04 green cast paint', 1: '04 bearing steel',
             2: '04 dark steel', 3: '04 rusted cast'}
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
        normal_map.inputs['Strength'].default_value = {0: .20, 1: .12, 2: .18, 3: .28}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        material_for_tile[tile] = material

    red = bpy.data.materials.new('04 red grease fitting')
    red.use_nodes = True
    principled = red.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.54, .035, .022, 1)
    principled.inputs['Metallic'].default_value = .02
    principled.inputs['Roughness'].default_value = .34
    special_materials['red'] = red

    black = bpy.data.materials.new('04 fitting bore')
    black.use_nodes = True
    principled = black.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.008, .009, .009, 1)
    principled.inputs['Metallic'].default_value = .35
    principled.inputs['Roughness'].default_value = .52
    special_materials['black'] = black

    label = bpy.data.materials.new('04 cast label paint')
    label.use_nodes = True
    principled = label.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.045, .092, .082, 1)
    principled.inputs['Metallic'].default_value = .02
    principled.inputs['Roughness'].default_value = .62
    special_materials['label'] = label

    marking = bpy.data.materials.new('04 bolt grade marking')
    marking.use_nodes = True
    principled = marking.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.46, .47, .46, 1)
    principled.inputs['Metallic'].default_value = .78
    principled.inputs['Roughness'].default_value = .42
    special_materials['marking'] = marking

    # A slight clear-coat-like response keeps the red fitting readable without
    # baking any fixed highlight into the base colour.
    for key in ('red', 'label', 'marking', 'black'):
        special_materials[key].node_tree.nodes.get('Principled BSDF').inputs[
            'Coat Weight'].default_value = .12 if key == 'red' else 0.0


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


def bevel_object(obj, width, segments=3):
    mod = obj.modifiers.new('Manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


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


def mesh_object(name, verts, faces):
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


def prism_x(name, points, x0, x1):
    """Extrude a closed YZ outline along X."""
    n = len(points)
    verts = [(x0, y, z) for y, z in points] + [(x1, y, z) for y, z in points]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def prism_z(name, points_xy, z0, z1):
    """Extrude a closed XY outline along Z."""
    n = len(points_xy)
    verts = [(x, y, z0) for x, y in points_xy] + [(x, y, z1) for x, y in points_xy]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def frustum_z(name, lower_xy, upper_xy, z0, z1):
    n = len(lower_xy)
    verts = [(x, y, z0) for x, y in lower_xy] + [(x, y, z1) for x, y in upper_xy]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def rounded_rect(hx, hy, radius, per_corner=8):
    radius = min(radius, hx * .98, hy * .98)
    corners = ((hx - radius, -hy + radius, -math.pi / 2, 0.0),
               (hx - radius, hy - radius, 0.0, math.pi / 2),
               (-hx + radius, hy - radius, math.pi / 2, math.pi),
               (-hx + radius, -hy + radius, math.pi, 1.5 * math.pi))
    points = []
    for cx, cy, a0, a1 in corners:
        for i in range(per_corner + 1):
            angle = a0 + (a1 - a0) * i / per_corner
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def capsule_x(width_y, length_x, center_y, center_x=0.0):
    """Closed XY capsule with its long axis along X."""
    r = width_y / 2
    half = max(0.0, (length_x - width_y) / 2)
    points = []
    for i in range(17):
        angle = -math.pi / 2 + math.pi * i / 16
        points.append((center_x + half + r * math.cos(angle),
                       center_y + r * math.sin(angle)))
    for i in range(17):
        angle = math.pi / 2 + math.pi * i / 16
        points.append((center_x - half + r * math.cos(angle),
                       center_y + r * math.sin(angle)))
    return points


def font():
    for path in ('C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return bpy.data.fonts.load(path, check_existing=True)
        except Exception:
            pass
    return None


FONT = None


def place_text(text, size, center, right, up, normal, name, material, extrude=.0018):
    """Create a closed extruded text mesh on an arbitrary oriented plane."""
    global FONT
    bpy.ops.object.text_add(location=(0, 0, 0))
    obj = bpy.context.object
    if FONT is None:
        FONT = font()
    if FONT:
        obj.data.font = FONT
    obj.data.body = text
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.size = size
    obj.data.extrude = extrude
    obj.data.bevel_depth = 0.0
    obj.data.bevel_resolution = 0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.object
    right, up, normal = (Vector(v).normalized() for v in (right, up, normal))
    matrix = Matrix(((right.x, up.x, normal.x, 0.0),
                     (right.y, up.y, normal.y, 0.0),
                     (right.z, up.z, normal.z, 0.0),
                     (0.0, 0.0, 0.0, 1.0)))
    obj.data.transform(matrix)
    obj.data.update()
    obj.location = Vector(center) + normal * .0008
    obj.name = name
    return assign_special(obj, material)


def build():
    axis = config['axis_height']
    gap = config['split_gap']
    split_b = axis - gap / 2
    split_c = axis + gap / 2
    base_t = config['base_thickness']
    half_len = config['base_length'] / 2
    half_depth = config['base_depth'] / 2
    body_half = config['body_depth'] / 2
    r_bore = config['seat_radius']
    outer_y = config['dome_radius_y']
    outer_z = config['dome_radius_z']
    lug_outer = config['lug_outer_y']
    lug_top = config['lug_top_z']
    boss_r = config['boss_outer_radius']
    boss_x = config['boss_face_x']
    boss_t = config['boss_thickness']

    # Base plate: rounded plan, manufacturing edge radius, two mounting slots,
    # a tapered underside pocket and four small cast marks.
    base = prism_z('Base plate', rounded_rect(half_depth, half_len, .045, 10), 0, base_t)
    bpy.context.view_layer.objects.active = base
    base.select_set(True)
    mod = base.modifiers.new('Base edge radius', 'BEVEL')
    mod.width = .010
    mod.segments = 3
    mod.limit_method = 'ANGLE'
    bpy.ops.object.modifier_apply(modifier=mod.name)
    cutters = []
    for side in (-1, 1):
        y = side * config['mount_bolt_spacing'] / 2
        cutters.append(prism_z('Slot cutter', capsule_x(config['mount_slot_width'],
                                                        config['mount_slot_length'], y),
                               -.03, base_t + .03))
    lower = rounded_rect(.120, half_len - .060, .030, 10)
    upper = rounded_rect(.085, half_len - .085, .025, 10)
    cutters.append(frustum_z('Underside pocket cutter', lower, upper, -.03, base_t - .040))
    for side in (-1, 1):
        y = side * config['mount_bolt_spacing'] / 2
        for dy in (-.065, .065):
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=.010, depth=.03,
                                                location=(0, y + dy, .002))
            cutters.append(bpy.context.object)
    boolean_cut(base, cutters, 'Base slot and underside pocket')
    # Do not bevel the boolean result: a second bevel on the slot/pocket rims
    # produced tiny overlapping faces in the draft audit.
    finish(base, 'Base plate', 0, bevel=0, boxed=True)

    # Raised machined-looking pads around the mounting slots.
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        y = side * config['mount_bolt_spacing'] / 2
        pad_outline = [(px, py + y) for px, py in
                       capsule_x(config['mount_pad_width_y'],
                                 config['mount_pad_length_x'], 0.0)]
        pad = prism_z(f'Mounting pad {suffix}', pad_outline,
                      base_t - .004, base_t + config['mount_pad_height'])
        cutter = prism_z('Pad slot cutter',
                         capsule_x(config['mount_slot_width'], config['mount_slot_length'], y),
                         -.03, base_t + .06)
        boolean_cut(pad, [cutter], 'Pad slot')
        finish(pad, f'Mounting pad {suffix}', 0, bevel=.003, boxed=True)

        # The underside rib is visible through the pocket in the bottom view;
        # its exact depth is inferred.
    rib = prism_z('Underside central rib',
                  rounded_rect(.055, .175, .030, 8), .006, base_t - .040)
    finish(rib, 'Underside central rib', 0, bevel=.006, boxed=True)

    saddle = prism_z('Housing saddle',
                     rounded_rect(.185, .345, .075, 10), base_t - .006, base_t + .024)
    finish(saddle, 'Housing saddle', 0, bevel=.010, boxed=True)

    # Lower housing half: a simple cast pedestal, with the lower half-bore cut
    # by a cylinder.  Making the bore a boolean cut keeps the extruded end faces
    # convex and closed; the draft's concave n-gon and post-boolean bevel were
    # the sources of the open/non-manifold lower housing.
    pts = [(-.340, base_t - .008), (-.332, base_t + .025), (-.320, base_t + .070),
           (-.309, base_t + .116), (-.302, base_t + .153), (-.300, split_b),
           (.300, split_b), (.302, base_t + .153), (.309, base_t + .116),
           (.320, base_t + .070), (.332, base_t + .025), (.340, base_t - .008)]
    lower = prism_x('Housing lower half', pts, -body_half, body_half)
    bevel_object(lower, .016, 3)
    cut = cutter_cylinder(r_bore, (-.3, 0, axis), (.3, 0, axis), 128)
    boolean_cut(lower, [cut], 'Housing lower bore')
    finish(lower, 'Housing lower half', 0, bevel=0, boxed=True)

    # Upper cap: an elliptical barrel vault, a separate tall bolt lug at each
    # side, and a boolean upper half-bore.  The lugs stand in the valleys seen
    # in the reference; they intersect the vault only near the split so they
    # remain closed solids and keep the arched outer silhouette.
    arch = [(outer_y * math.sin(-math.pi / 2 + math.pi * i / 96),
             split_c + (config['total_height'] - split_c) * math.cos(-math.pi / 2 + math.pi * i / 96))
            for i in range(97)]
    cap = prism_x('Housing cap', arch, -body_half, body_half)
    bevel_object(cap, .020, 4)
    cut = cutter_cylinder(r_bore, (-.3, 0, axis), (.3, 0, axis), 128)
    boolean_cut(cap, [cut], 'Housing cap bore')
    finish(cap, 'Housing cap', 0, bevel=0, boxed=True)

    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        y = side * config['cap_bolt_y']
        # The lug reaches inward until it overlaps the vault low down, so the
        # bolted ear is a connected cap feature rather than a floating column.
        # A small taper and a generous edge radius read as a cast ear.
        lug_base = rounded_rect(.130, .0475, .018, 8)
        lug_lower = [(px, py + y) for px, py in lug_base]
        lug_upper = [(px * .92, py * .92 + y) for px, py in lug_base]
        lug = frustum_z(f'Cap lug {suffix}', lug_lower, lug_upper, split_c, lug_top)
        bevel_object(lug, .008, 3)
        finish(lug, f'Cap lug {suffix}', 0, bevel=0, boxed=True)

    # Split bearing bosses on both ends.  They are separate upper and lower
    # half-annuli so the shaft-centre split line remains visible.
    def boss_half(name, side, upper, front):
        r_out, r_in = boss_r, r_bore
        if upper:
            z_side = split_c
            a_o = math.asin((z_side - axis) / r_out)
            a_i = math.asin((z_side - axis) / r_in)
            outer = [(r_out * math.cos(a_o + (math.pi - 2 * a_o) * i / 64),
                      axis + r_out * math.sin(a_o + (math.pi - 2 * a_o) * i / 64))
                     for i in range(65)]
            inner = [(r_in * math.cos(a_i + (math.pi - 2 * a_i) * i / 64),
                      axis + r_in * math.sin(a_i + (math.pi - 2 * a_i) * i / 64))
                     for i in range(64, -1, -1)]
        else:
            z_side = split_b
            a_o = math.asin((axis - z_side) / r_out)
            a_i = math.asin((axis - z_side) / r_in)
            outer = [(r_out * math.cos(math.pi + a_o + (math.pi - 2 * a_o) * i / 64),
                      axis + r_out * math.sin(math.pi + a_o + (math.pi - 2 * a_o) * i / 64))
                     for i in range(65)]
            inner = [(r_in * math.cos(math.pi + a_i + (math.pi - 2 * a_i) * i / 64),
                      axis + r_in * math.sin(math.pi + a_i + (math.pi - 2 * a_i) * i / 64))
                     for i in range(64, -1, -1)]
        outline = outer + inner
        x1 = boss_x if front else -boss_x
        x0 = x1 - boss_t if front else x1 + boss_t
        obj = prism_x(name, outline, min(x0, x1), max(x0, x1))
        return finish(obj, name, 0, bevel=.0035, boxed=True)

    boss_half('Front bearing boss upper', 1, True, True)
    boss_half('Front bearing boss lower', 1, False, True)
    boss_half('Back bearing boss upper', -1, True, False)
    boss_half('Back bearing boss lower', -1, False, False)

    # Small cast tabs at the bottom centre of the front and back bosses.
    for side, suffix in ((1, 'front'), (-1, 'back')):
        x = side * (boss_x + .001)
        tab = prism_x(f'Seat tab {suffix}',
                      [(-.035, base_t + .001), (-.030, base_t + .035),
                       (.030, base_t + .035), (.035, base_t + .001)],
                      x - .006, x + .006)
        finish(tab, f'Seat tab {suffix}', 0, bevel=.004, boxed=True)

    # Bearing: heavy outer ring and a wide, chamfered inner ring.  The visible
    # bearing proportions follow the drawing; catalog dimensions are not used
    # to claim a measured fit.
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=config['bearing_outer_radius'],
                                        depth=config['bearing_outer_width'],
                                        rotation=(0, math.pi / 2, 0),
                                        location=(-config['bearing_outer_front_x'] +
                                                  config['bearing_outer_width'] / 2, 0, axis))
    outer = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    cut = cutter_cylinder(config['bearing_outer_inner'], (-.3, 0, axis), (.3, 0, axis), 128)
    boolean_cut(outer, [cut], 'Bearing outer bore')
    finish(outer, 'Bearing outer ring', 2, bevel=.003, boxed=True)

    # Opaque cage/roller band between the two rings.  Without it the empty
    # annular gap is a second see-through opening, which a real bearing does
    # not have and which would make the isolated opening check ambiguous.
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=.1125, depth=.140,
                                        rotation=(0, math.pi / 2, 0),
                                        location=(.030, 0, axis))
    cage = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    cut = cutter_cylinder(.0965, (-.3, 0, axis), (.3, 0, axis), 128)
    boolean_cut(cage, [cut], 'Bearing cage bore')
    finish(cage, 'Bearing cage ring', 2, bevel=.002, boxed=True)

    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=config['bearing_inner_outer'],
                                        depth=config['bearing_inner_width'],
                                        rotation=(0, math.pi / 2, 0),
                                        location=(-config['bearing_inner_front_x'] +
                                                  config['bearing_inner_width'] / 2, 0, axis))
    inner = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bevel = inner.modifiers.new('Inner ring chamfer', 'BEVEL')
    bevel.width = .014
    bevel.segments = 3
    bevel.limit_method = 'ANGLE'
    bpy.context.view_layer.objects.active = inner
    inner.select_set(True)
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    cut = cutter_cylinder(config['bearing_inner_bore'], (-.3, 0, axis), (.3, 0, axis), 128)
    boolean_cut(inner, [cut], 'Bearing inner bore')
    finish(inner, 'Bearing inner ring', 1, bevel=0, boxed=True)

    # Two cap bolts: machined washer faces, hex heads and tiny grade marks.
    z0 = lug_top
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        y = side * config['cap_bolt_y']
        cylinder(f'Bolt washer {suffix}', config['cap_bolt_washer_radius'],
                 z0 - .002, z0 + .010, 2, xy=(0, y), segments=48, bevel=.002)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=6,
            radius=config['cap_bolt_head_across_flats'] / math.sqrt(3),
            depth=config['cap_bolt_head_height'],
            location=(0, y, z0 + .010 + config['cap_bolt_head_height'] / 2))
        head = bpy.context.object
        finish(head, f'Cap bolt head {suffix}', 2, bevel=.0035, smooth=False)
        top = z0 + .010 + config['cap_bolt_head_height']
        place_text('8.8', .014, (0, y, top + .0008), (0, 1, 0), (-1, 0, 0), (0, 0, 1),
                   f'Bolt grade mark {suffix}', 'marking', extrude=.0006)

    # Cast model designation on the front and back cap faces.  The reference
    # sheet repeats marks inconsistently; two centred marks are the conservative
    # reading of the source 3/4 photograph.
    z_label = .478
    place_text(config['label_text'], config['label_size'], (body_half + .002, 0, z_label),
               (0, 1, 0), (0, 0, 1), (1, 0, 0), 'Cast label front', 'label', extrude=.0015)
    place_text(config['label_text'], config['label_size'], (-body_half - .002, 0, z_label),
               (0, -1, 0), (0, 0, 1), (-1, 0, 0), 'Cast label back', 'label', extrude=.0015)
    # The top panel of the reference shows the mark on the crown, behind the
    # grease fitting.  It is a separate inferred mark, not a second view of the
    # flat front text.
    place_text(config['label_text'], config['label_size'] * .92,
               (config['label_top_x'], config['label_top_y'], config['label_top_z']),
               (0, 1, 0), (-1, 0, 0), (0, 0, 1), 'Cast label top', 'label', extrude=.0015)

    # Red grease fitting on the front crown, with a cast boss and a dark bore.
    normal = Vector(config['zerk_normal']).normalized()
    center = Vector((config['zerk_x'], 0, config['zerk_z']))
    boss = along('Zerk boss', center - normal * .030, center + normal * .006,
                 config['zerk_boss_radius'], 0, segments=48, bevel=.003, smooth='quads')
    along('Zerk red body', center + normal * .003, center + normal * .034, .035, 0,
          segments=40, bevel=.0015, smooth='quads', special='red')
    bpy.ops.mesh.primitive_torus_add(major_radius=.034, minor_radius=.010,
                                     major_segments=48, minor_segments=12,
                                     location=center + normal * .039)
    ring = bpy.context.object
    ring.rotation_euler = normal.to_track_quat('Z', 'Y').to_euler()
    finish_special(ring, 'Zerk red ring', 'red', bevel=0, smooth='all')
    along('Zerk dark bore', center + normal * .034, center + normal * .043, .017, 0,
          segments=32, bevel=0, smooth='all', special='black')

    # Match the centred Z convention used by the earlier review candidates.
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
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .58
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, width, height in (
        ('large softbox', (2.4, -2.8, 3.8), 185, 3, 3),
        ('fill softbox', (-3, 2.5, 2), 100, 3, 3),
        ('front reflection', (3.0, 1.4, 1), 100, 1.4, 3.2),
        ('side fill minus Y', (0, -2.5, -.245), 45, .6, .6),
        ('side fill plus Y', (0, 2.5, -.245), 45, .6, .6)):
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

cam = camera(scene, (3, -.4, .25), Vector((0, 0, center.z)), .62)
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
        ('bearing', (0, 0, config['axis_height'] - .5), (1, 0, .02), .92),
        ('bolt', (0, config['cap_bolt_y'], config['lug_top_z'] - .5), (1.2, -.8, 1.0), .28),
        ('zerk', (config['zerk_x'], 0, config['zerk_z'] - .5), (1.0, -.4, 1.1), .30),
        ('label', (config['body_depth'] / 2, 0, .478 - .5), (1, -.10, .10), .49),
        ('slot', (0, -config['mount_bolt_spacing'] / 2, config['base_thickness'] - .5), (1, -.35, .9), .47),
        ('split', (config['body_depth'] / 2, 0, config['axis_height'] - .5), (1, -.35, .18), .47),
        ('bottom', (0, 0, -.5), (.7, -.55, -1), 1.55),
        ('surface', (.12, -.30, .34 - .5), (2, -1, 1.2), .46),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    bearing_names = {'Bearing outer ring', 'Bearing inner ring', 'Bearing cage ring'}
    for obj in objects:
        obj.hide_render = obj.name not in bearing_names
    cam = camera(scene, (1, 0, 0), Vector((0, 0, center.z)), .90, 1.10)
    draw(scene, render_dir / 'bearing-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)

    base_names = {'Base plate', 'Mounting pad minus', 'Mounting pad plus',
                  'Underside central rib', 'Housing saddle'}
    for obj in objects:
        obj.hide_render = obj.name not in base_names
    cam = camera(scene, (0, 0, 1), Vector((0, 0, config['base_thickness'] - .5)), config['base_length'], 1.08)
    draw(scene, render_dir / 'base-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)

    housing_names = {'Housing lower half', 'Housing cap', 'Cap lug minus',
                     'Cap lug plus', 'Front bearing boss upper',
                     'Front bearing boss lower', 'Back bearing boss upper',
                     'Back bearing boss lower', 'Housing saddle', 'Zerk boss'}
    for obj in objects:
        obj.hide_render = obj.name not in housing_names
    cam = camera(scene, (1, 0, 0), Vector((0, 0, center.z)), 1.02, 1.12)
    draw(scene, render_dir / 'housing-only.png', resolution=512, samples=16)
    bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    key = bpy.data.objects['large softbox']
    key.location = (-3, 4, 5)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (3, -.4, .25), Vector((0, 0, center.z)), .62)
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
cam = camera(scene, (3, -.4, .25), Vector((0, 0, center.z)), .62)
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
        'geometry_basis': 'six-view panels plus public SN 513 catalog ratios, uncalibrated',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': 'Left camera at -Y and right at +Y; the reference is uncalibrated.',
        'source_camera_note': 'source-photo framing distance scaled by new/old span; diagnostic only',
        'source_camera_factor': factor,
        'isolated_checks': {'bearing_bore_expected': 1, 'base_slots_expected': 2,
                            'housing_bore_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
