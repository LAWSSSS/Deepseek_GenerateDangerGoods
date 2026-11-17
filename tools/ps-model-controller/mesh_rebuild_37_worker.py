"""Blender worker: part-based rebuild of item 37, a black oil-collector/filter.

Evidence: one clean product photograph (1071 x 1468, alpha channel) of a black
inline pneumatic oil collector / filter and its matching rust derivative.  The
photograph shows, from top to bottom:

* a rectangular black mounting plate with two true stadium-shaped through
  holes and a ridged rectangular neck below it,
* a round overhanging cap with a shallow stepped top,
* a black plastic cage with four vertical posts and three rows of large
  windows; the front pair of windows is split by a narrow centre post and a
  white woven filter element is visible behind every window.  Two shallow
  dark slots sit on the upper band,
* a wide black collar ring, then a smoke-grey translucent bowl with two
  visible internal ribs (four are inferred around the circumference),
* a short black neck, a knurled black drain knob and a small brass drain pin.

The six-view map has no reference for item 37 (role ``missing``), so the
photograph and its visible proportions are the only design evidence.  Hidden
back, bowl interior, mounting-thread depth and true dimensions are inferred.
The old TripoSR library mesh is a fused, smoothed shell: the windows are
painted grooves, the bracket is lumpy, the translucent bowl is opaque and the
drain parts are merged.  It is kept unchanged and rendered only for
comparison.

Coordinates: Z up, cylinder axis Z, front camera on +X.  The mounting plate is
normal to X; the front cage windows are at +/-35 degrees.  Every part is a
separate closed mesh; openings checked by rays are the two mounting holes, all
front cage windows, the filter visibility and the shallow brass drain recess.
The candidate is manual-only and is not accepted, placed automatically or
merged into the formal composite.
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
D = request['config']['dimensions']
RR = request['config']['render']
TEX = request['textures']

objects = []
metal_objects = []
nonmetal_objects = []
MATS = {}
CAM = None
METAL_MATERIALS = {'brass'}
CAGE_PARTS = []


# ---------------------------------------------------------------------------
# Materials and UV helpers
# ---------------------------------------------------------------------------
def load_image(path, *, noncolor=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    if noncolor:
        image.colorspace_settings.name = 'Non-Color'
    image.pack()
    return image


def make_material(key, spec, *, metallic=1.0, roughness=0.5, coat=0.0,
                  coat_roughness=0.12, alpha=False):
    material = bpy.data.materials.new(spec.get('label', key))
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    principled.inputs['IOR'].default_value = 1.45
    image = load_image(spec['base'])
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = image
    tex.interpolation = 'Linear'
    links.new(tex.outputs['Color'], principled.inputs['Base Color'])
    if spec.get('mr'):
        packed = nodes.new('ShaderNodeTexImage')
        packed.image = load_image(spec['mr'], noncolor=True)
        packed.interpolation = 'Linear'
        split = nodes.new('ShaderNodeSeparateColor')
        links.new(packed.outputs['Color'], split.inputs['Color'])
        links.new(split.outputs['Green'], principled.inputs['Roughness'])
        links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    else:
        principled.inputs['Metallic'].default_value = metallic
        principled.inputs['Roughness'].default_value = roughness
    if spec.get('normal'):
        normal_tex = nodes.new('ShaderNodeTexImage')
        normal_tex.image = load_image(spec['normal'], noncolor=True)
        normal_tex.interpolation = 'Linear'
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.inputs['Strength'].default_value = spec.get('normal_strength', 0.55)
        links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    if spec.get('transmission'):
        principled.inputs['Transmission Weight'].default_value = float(
            spec['transmission'])
        principled.inputs['IOR'].default_value = float(spec.get('ior', 1.49))
        material.use_backface_culling = False
        material.blend_method = 'BLEND'
    if coat:
        principled.inputs['Coat Weight'].default_value = coat
        principled.inputs['Coat Roughness'].default_value = coat_roughness
    if alpha:
        clip = nodes.new('ShaderNodeMath')
        clip.operation = 'ROUND'
        clip.use_clamp = False
        links.new(tex.outputs['Alpha'], clip.inputs[0])
        links.new(clip.outputs[0], principled.inputs['Alpha'])
        material.blend_method = 'CLIP'
        material.alpha_threshold = 0.5
        material.use_backface_culling = True
    material['candidate_number'] = 37
    material['texture_role'] = key
    return material


def uv_data(obj):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='UVMap')
    return obj.data.uv_layers.active.data


def uv_box(obj, scale=4.0):
    layer = uv_data(obj)
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
            layer[li].uv = (u, v)


def uv_cyl_z(obj, z0, z1, radius):
    layer = uv_data(obj)
    mesh = obj.data
    for poly in mesh.polygons:
        normal = poly.normal
        us = []
        if abs(normal.z) > .94:
            for li in poly.loop_indices:
                co = mesh.vertices[mesh.loops[li].vertex_index].co
                layer[li].uv = (.5 + co.x / (2 * radius), .5 + co.y / (2 * radius))
            continue
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            u = (math.atan2(co.y, co.x) / math.tau) % 1.0
            v = (co.z - z0) / max(1e-8, z1 - z0)
            us.append(u)
            layer[li].uv = (u, v)
        if us and max(us) - min(us) > .5:
            for li in poly.loop_indices:
                u, v = layer[li].uv
                if u < .5:
                    layer[li].uv = (u + 1.0, v)


def uv_scale(obj, su, sv):
    for loop in obj.data.loops:
        u, v = obj.data.uv_layers.active.data[loop.index].uv
        obj.data.uv_layers.active.data[loop.index].uv = (u * su, v * sv)


def apply_uv(obj, mode):
    if mode == 'box':
        uv_box(obj)
    elif isinstance(mode, tuple) and mode[0] == 'cyl_z':
        uv_cyl_z(obj, mode[1], mode[2], mode[3])
    elif mode == 'none':
        return
    else:
        raise ValueError(f'unknown UV mode {mode}')


def apply_bevel(obj, width, segments=2, angle=34):
    if width <= 0:
        return
    mod = obj.modifiers.new('manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.use_clamp_overlap = True
    active(obj)
    if not obj.select_get():
        obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def active(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def smooth_by_angle(obj, angle=34):
    active(obj)
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))
    except Exception:
        for poly in obj.data.polygons:
            poly.use_smooth = True


def classify(obj, key, *, register=True):
    if key in METAL_MATERIALS:
        metal_objects.append(obj.name)
    else:
        nonmetal_objects.append(obj.name)
    if obj.name.startswith('cage_'):
        CAGE_PARTS.append(obj.name)
    if register:
        objects.append(obj)


def finish(obj, name, key, *, bevel=.001, uv='box', smooth=34,
           register=True, material=None):
    if obj.type != 'MESH':
        active(obj)
        bpy.ops.object.convert(target='MESH')
        obj = bpy.context.object
    obj.name = name
    obj.data.name = name
    if bevel:
        apply_bevel(obj, bevel)
    if smooth is not None:
        smooth_by_angle(obj, smooth)
    obj.data.materials.clear()
    obj.data.materials.append(material or MATS[key])
    apply_uv(obj, uv)
    classify(obj, key, register=register)
    return obj


# ---------------------------------------------------------------------------
# Primitive construction
# ---------------------------------------------------------------------------
def mesh_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def cylinder_z(name, radius, z0, z1, key, *, x=0.0, y=0.0, segs=96,
               bevel=.001, uv='box', register=True):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=z1 - z0, location=(x, y, (z0 + z1) / 2))
    obj = bpy.context.object
    return finish(obj, name, key, bevel=bevel, uv=uv, register=register)


def box(name, center, size, key, *, rotation=(0, 0, 0), bevel=.001, uv='box',
        register=True, material=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    obj.rotation_euler = rotation
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, key, bevel=bevel, uv=uv, register=register,
                  material=material)


def lathe_z(name, profile, key, *, segs=160, bevel=.0008, uv='box', smooth=30,
            register=True):
    """Revolve an open profile (radius, z) around Z and cap it at both ends."""
    verts = []
    rings = []
    for radius, z in profile:
        start = len(verts)
        for j in range(segs):
            t = math.tau * j / segs
            verts.append((radius * math.cos(t), radius * math.sin(t), z))
        rings.append(list(range(start, start + segs)))
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((a[j], a[j2], b[j2], b[j]))
    bottom = len(verts)
    verts.append((0.0, 0.0, profile[0][1]))
    top = len(verts)
    verts.append((0.0, 0.0, profile[-1][1]))
    first, last = rings[0], rings[-1]
    for j in range(segs):
        j2 = (j + 1) % segs
        faces.append((bottom, first[j2], first[j]))
        faces.append((top, last[j], last[j2]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=smooth,
                  register=register)


def lathe_ring(name, profile, key, *, segs=160, bevel=0.0, uv='box', smooth=32,
               register=True):
    """Revolve a *closed* cross-section loop (radius, z) into a ring solid."""
    verts = []
    rings = []
    for radius, z in profile:
        start = len(verts)
        for j in range(segs):
            t = math.tau * j / segs
            verts.append((radius * math.cos(t), radius * math.sin(t), z))
        rings.append(list(range(start, start + segs)))
    faces = []
    count = len(rings)
    for i in range(count):
        a, b = rings[i], rings[(i + 1) % count]
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append((a[j], a[j2], b[j2], b[j]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=smooth,
                  register=register)


def arc_wall(name, a0_deg, a1_deg, z0, z1, r_out_top, r_out_bottom,
             thickness, key, *, segs=16, bevel=0.0, uv='box', smooth=32,
             register=True):
    """A closed annular-sector wall between two angles (degrees)."""
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    verts = []
    rings = []
    for z, r_out in ((z0, r_out_bottom), (z1, r_out_top)):
        for r in (r_out, r_out - thickness):
            start = len(verts)
            for j in range(segs + 1):
                a = a0 + (a1 - a0) * j / segs
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(list(range(start, start + segs + 1)))
    ob0, ib0, ot0, it0 = rings
    faces = []
    for j in range(segs):
        j2 = j + 1
        faces.append((ob0[j], ob0[j2], ot0[j2], ot0[j]))
        faces.append((ib0[j2], ib0[j], it0[j], it0[j2]))
        faces.append((ot0[j], ot0[j2], it0[j2], it0[j]))
        faces.append((ob0[j2], ob0[j], ib0[j], ib0[j2]))
    faces.append((ob0[0], ib0[0], it0[0], ot0[0]))
    faces.append((ob0[segs], ot0[segs], it0[segs], ib0[segs]))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel, uv=uv, smooth=smooth,
                  register=register)


def quarter_cylinder_x(name, y_corner, z_corner, radius, x0, x1, key, *,
                       segs=20, bevel=.0004):
    """A quarter-cylinder gusset in the concave corner, extruded along X."""
    sign = 1.0 if y_corner >= 0 else -1.0
    points = [(y_corner, z_corner), (y_corner, z_corner - radius)]
    for j in range(1, segs):
        a = -math.pi / 2 + (math.pi / 2) * j / segs
        points.append((y_corner + sign * radius * math.cos(a),
                       z_corner + radius * math.sin(a)))
    points.append((y_corner + sign * radius, z_corner))
    n = len(points)
    verts = [(x0, py, pz) for py, pz in points]
    verts += [(x1, py, pz) for py, pz in points]
    faces = [tuple(range(n)), tuple(reversed(range(n, 2 * n)))]
    for j in range(n):
        j2 = (j + 1) % n
        faces.append((j, j2, n + j2, n + j))
    obj = mesh_object(name, verts, faces)
    return finish(obj, name, key, bevel=bevel, uv='box', smooth=30)


def stadium_prism(name, cy, cz, width, height, x0, x1, *, segs=28):
    """A convex stadium prism along X, used only as a boolean cutter."""
    half = height / 2
    straight = max(0.0, (width - height) / 2)
    points = []
    for j in range(segs + 1):
        t = -math.pi / 2 + math.pi * j / segs
        points.append((straight + half * math.cos(t), half * math.sin(t)))
    for j in range(segs + 1):
        t = math.pi / 2 + math.pi * j / segs
        points.append((-straight + half * math.cos(t), half * math.sin(t)))
    n = len(points)
    verts = [(x0, cy + py, cz + pz) for py, pz in points]
    verts += [(x1, cy + py, cz + pz) for py, pz in points]
    faces = [tuple(range(n)), tuple(reversed(range(n, 2 * n)))]
    for j in range(n):
        j2 = (j + 1) % n
        faces.append((j, j2, n + j2, n + j))
    obj = mesh_object(name, verts, faces)
    obj.hide_render = True
    return obj


def clean_mesh(obj, dist=1e-6):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=dist)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def boolean_difference(target, cutter):
    mod = target.modifiers.new('true opening', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    active(target)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    clean_mesh(target)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def build_bracket():
    plate = D['plate']
    center_z = (plate['z0'] + plate['z1']) / 2
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, center_z))
    obj = bpy.context.object
    obj.scale = (plate['thickness'], plate['width'], plate['height'])
    active(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    apply_bevel(obj, .008, segments=3, angle=30)
    for index, hole in enumerate(plate['holes']):
        cutter = stadium_prism(
            f'bracket_hole_cutter_{index}', hole['y'], hole['z'],
            hole['w'], hole['h'], -plate['thickness'] * 2.6,
            plate['thickness'] * 2.6)
        boolean_difference(obj, cutter)
    clean_mesh(obj)
    finish(obj, 'bracket_plate', 'black_plastic', bevel=0, uv='box', smooth=30)

    # Quarter-round gussets in the concave corner between the plate
    # underside and the neck sides.  Their flat faces stay flush with the
    # plate and neck, so they read as fillets rather than protruding balls.
    for side in (-1, 1):
        quarter_cylinder_x(
            f'bracket_fillet_{"l" if side < 0 else "r"}',
            side * D['neck']['width'] / 2, plate['z0'], .012,
            -plate['thickness'] / 2, plate['thickness'] / 2,
            'black_plastic', bevel=.0004)


def build_neck():
    neck = D['neck']
    center_z = (neck['z0'] + neck['z1']) / 2
    box('neck_core', (0, 0, center_z),
        (neck['depth'], neck['width'], neck['z1'] - neck['z0']),
        'black_plastic', bevel=.007, uv='box')
    positions = neck['rib_y']
    for face, x in (('front', neck['depth'] / 2 - .001),
                    ('back', -neck['depth'] / 2 + .001)):
        for index, y in enumerate(positions):
            cylinder_z(f'neck_rib_{face}_{index}', neck['rib_radius'],
                       neck['z0'] + .002, neck['z1'] - .002,
                       'black_plastic', x=x, y=y, segs=28, bevel=.0028,
                       uv='box')


def build_cap():
    cap = D['cap']
    profile = [
        (0.200, cap['z0']), (0.236, cap['z0']),
        (0.2445, cap['z0'] + .0024), (0.2477, cap['z0'] + .007),
        (0.2477, cap['z1'] - .007), (0.2462, cap['z1'] - .0024),
        (0.2410, cap['z1']), (0.2290, cap['z1'] + .0008),
        (0.2240, cap['z1'] - .0014), (0.1500, cap['z1'] - .0014),
    ]
    obj = lathe_z('cap', profile, 'black_plastic', segs=192, bevel=.0007,
                  uv=('cyl_z', cap['z0'], cap['z1'], cap['radius']))
    # A fine moulded bead near the outer edge of the cap.
    bpy.ops.mesh.primitive_torus_add(major_segments=192, minor_segments=8,
                                     location=(0, 0, cap['z1'] - .0045),
                                     major_radius=.2400, minor_radius=.0018)
    bead = bpy.context.object
    finish(bead, 'cap_edge_bead', 'black_plastic', bevel=0, uv='box',
           smooth=32)


def build_cage():
    cage = D['cage']
    r_top = cage['r_top']
    r_bottom = cage['r_bottom']
    z_top = cage['z_top']
    z_bottom = cage['z_bottom']
    wall = cage['wall']

    def radius_at(z):
        t = (z_top - z) / (z_top - z_bottom)
        return r_top + (r_bottom - r_top) * t

    for name, band in cage['bands'].items():
        z0, z1 = band['z0'], band['z1']
        r0, r1 = radius_at(z0), radius_at(z1)
        profile = [
            (r0 - wall + .0012, z0), (r0 - .0012, z0),
            (r0, z0 + .0012), (r1, z1 - .0012),
            (r1 - .0012, z1), (r1 - wall + .0012, z1),
            (r1 - wall, z1 - .0012), (r0 - wall, z0 + .0012),
        ]
        lathe_ring(f'cage_band_{name}', profile, 'black_plastic',
                   segs=192, bevel=.0008, uv=('cyl_z', z_bottom, z_top, r_top))

    posts = cage['posts']
    for index, post in enumerate(posts):
        arc_wall(f'cage_post_{post["name"]}', post['a0'], post['a1'],
                 z_bottom, z_top, r_top + .0008, r_bottom + .0008,
                 wall, 'black_plastic', segs=post.get('segs', 16),
                 bevel=.0008, uv='box', smooth=32)

    # White woven filter element.  Its UV is tiled so the weave stays fine.
    filt = D['filter']
    profile = [
        (0.1700, filt['z0']),
        (0.1880, filt['z0'] + .0022), (0.1900, filt['z0'] + .0060),
        (0.1900, filt['z1'] - .0060), (0.1880, filt['z1'] - .0022),
        (0.1700, filt['z1']),
    ]
    core = lathe_z('filter_core', profile, 'filter_mesh', segs=192,
                   bevel=0, uv=('cyl_z', filt['z0'], filt['z1'], filt['r']))
    uv_scale(core, 2.0, 1.5)

    # Shallow decorative slots on the upper band (not through holes).
    slot = D['slots']
    for index, angle in enumerate(slot['angles']):
        a0, a1 = angle - slot['half_angle'], angle + slot['half_angle']
        arc_wall(f'slot_inset_{index}', a0, a1, slot['z0'], slot['z1'],
                 slot['r_inset'] + .0010, slot['r_inset'] + .0010,
                 .0008, 'black_matte', segs=8, bevel=0, uv='box',
                 smooth=30)
        # Frame bars: two horizontal and two vertical.
        arc_wall(f'slot_frame_{index}_top', a0, a1, slot['z1'] - .0030,
                 slot['z1'], slot['r_frame'], slot['r_frame'],
                 slot['r_frame'] - slot['r_inset'], 'black_plastic',
                 segs=8, bevel=0, uv='box', smooth=30)
        arc_wall(f'slot_frame_{index}_bottom', a0, a1, slot['z0'],
                 slot['z0'] + .0030, slot['r_frame'], slot['r_frame'],
                 slot['r_frame'] - slot['r_inset'], 'black_plastic',
                 segs=8, bevel=0, uv='box', smooth=30)
        arc_wall(f'slot_frame_{index}_left', a0, a0 + 2.0,
                 slot['z0'], slot['z1'], slot['r_frame'], slot['r_frame'],
                 slot['r_frame'] - slot['r_inset'], 'black_plastic',
                 segs=3, bevel=0, uv='box', smooth=30)
        arc_wall(f'slot_frame_{index}_right', a1 - 2.0, a1,
                 slot['z0'], slot['z1'], slot['r_frame'], slot['r_frame'],
                 slot['r_frame'] - slot['r_inset'], 'black_plastic',
                 segs=3, bevel=0, uv='box', smooth=30)


def build_collar():
    collar = D['collar']
    z0, z1 = collar['z0'], collar['z1']
    r_in, r_out = collar['r_in'], collar['r_out']
    profile = [
        (r_in, z1 - .0015), (r_in + .0015, z1),
        (r_out - .0040, z1), (r_out - .0010, z1 - .0018),
        (r_out, z1 - .0060), (r_out - .0008, z0 + .0080),
        (r_out - .0030, z0 + .0020), (r_out - .0080, z0),
        (r_in + .0060, z0), (r_in + .0015, z0 + .0018),
        (r_in, z0 + .0060),
    ]
    lathe_ring('collar', profile, 'black_plastic', segs=192, bevel=0,
               uv=('cyl_z', z0, z1, r_out))
    # Fine moulded seam line around the collar.
    bpy.ops.mesh.primitive_torus_add(major_segments=192, minor_segments=8,
                                     location=(0, 0, z0 + .022),
                                     major_radius=r_out - .0007,
                                     minor_radius=.0010)
    seam = bpy.context.object
    finish(seam, 'collar_seam', 'black_plastic', bevel=0, uv='box',
           smooth=32)


def build_bowl():
    bowl = D['bowl']
    profile = [
        (0.2420, -0.1587), (0.2405, -0.2000), (0.2360, -0.2800),
        (0.2290, -0.3350), (0.2210, -0.3520), (0.2060, -0.3620),
        (0.1800, -0.3690), (0.1350, -0.3735), (0.0850, -0.3745),
        (0.0750, -0.3740), (0.0700, -0.3700),
        (0.0800, -0.3680), (0.1300, -0.3650), (0.1740, -0.3600),
        (0.2000, -0.3520), (0.2140, -0.3400), (0.2220, -0.3200),
        (0.2280, -0.2800), (0.2330, -0.2000), (0.2350, -0.1587),
    ]
    lathe_ring('bowl_shell', profile, 'smoke_bowl', segs=192, bevel=0,
               uv=('cyl_z', bowl['z0'], bowl['z1'], bowl['r_top']),
               smooth=36)
    for index, angle in enumerate(bowl['rib_angles']):
        inner_top = .2350
        inner_bottom = .2240
        arc_wall(f'bowl_rib_{index}', angle - bowl['rib_half_angle'],
                 angle + bowl['rib_half_angle'], bowl['rib_z0'],
                 bowl['rib_z1'], inner_top - .0015, inner_bottom - .0015,
                 .0055, 'smoke_bowl', segs=4, bevel=0, uv='box',
                 smooth=30)


def build_drain():
    drain = D['drain']
    cylinder_z('drain_neck', drain['neck_radius'], drain['neck_z0'],
               drain['neck_z1'], 'black_plastic', segs=96, bevel=.0016,
               uv='box')

    # Knurled solid knob.
    ridges = drain['ridges']
    samples = ridges * 6
    levels = [
        (drain['knob_z0'], drain['knob_radius'] * .78),
        (drain['knob_z0'] + .004, drain['knob_radius'] * .94),
        (drain['knob_z0'] + .009, drain['knob_radius']),
        (drain['knob_z1'] - .010, drain['knob_radius']),
        (drain['knob_z1'] - .004, drain['knob_radius'] * .96),
        (drain['knob_z1'], drain['knob_radius'] * .84),
    ]
    verts = []
    rings = []
    for z, radius in levels:
        start = len(verts)
        for j in range(samples):
            phase = (j % 6) / 6.0
            valley = 0.0 if (0.28 <= phase <= 0.72) else 1.0
            # Rounded transitions so the knurl does not look like a comb.
            if phase < 0.28:
                valley = 0.5 - 0.5 * math.cos(math.pi * phase / .28)
            elif phase > 0.72:
                valley = 0.5 - 0.5 * math.cos(math.pi * (1.0 - phase) / .28)
            r = radius - drain['ridge_depth'] * valley
            a = math.tau * j / samples
            verts.append((r * math.cos(a), r * math.sin(a), z))
        rings.append(list(range(start, start + samples)))
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(samples):
            j2 = (j + 1) % samples
            faces.append((a[j], a[j2], b[j2], b[j]))
    bottom = len(verts)
    verts.append((0, 0, levels[0][0]))
    top = len(verts)
    verts.append((0, 0, levels[-1][0]))
    for j in range(samples):
        j2 = (j + 1) % samples
        faces.append((bottom, rings[0][j2], rings[0][j]))
        faces.append((top, rings[-1][j], rings[-1][j2]))
    knob = mesh_object('drain_knob', verts, faces)
    finish(knob, 'drain_knob', 'black_plastic', bevel=0, uv='box', smooth=28)

    brass = drain['brass']
    profile = [
        (0.0060, brass['recess_z']),
        (0.0100, brass['recess_z'] - .0045), (0.0130, -0.4980),
        (0.0145, -0.5000), (0.0165, -0.4990), (0.0175, -0.4950),
        (0.0180, -0.4850), (0.0180, -0.4740), (0.0165, -0.4660),
        (0.0140, brass['top_z']),
    ]
    lathe_z('brass_nipple', profile, 'brass', segs=96, bevel=.0004,
            uv='box', smooth=24)


def refine_uvs():
    """Tile the dark plastic/brass grain so it stays fine at this model scale."""
    for obj in objects:
        if not obj.data.materials:
            continue
        role = obj.data.materials[0].get('texture_role')
        if role in ('black_plastic', 'black_matte', 'brass'):
            uv_scale(obj, 3.0, 2.0)


def build_all():
    build_bracket()
    build_neck()
    build_cap()
    build_cage()
    build_collar()
    build_bowl()
    build_drain()
    refine_uvs()


# ---------------------------------------------------------------------------
# Scene, checks, renders, export
# ---------------------------------------------------------------------------
def init_camera(scene):
    data = bpy.data.cameras.new('review camera')
    cam = bpy.data.objects.new('review camera', data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_camera(cam, eye, target, *, ortho_scale=None, fov=40, roll_deg=0.0,
               shift_x=0.0, shift_y=0.0):
    cam.location = Vector(eye)
    cam.rotation_mode = 'QUATERNION'
    quat = (Vector(target) - cam.location).to_track_quat('-Z', 'Y')
    if roll_deg:
        quat = quat @ Quaternion((0, 0, 1), math.radians(roll_deg))
    cam.rotation_quaternion = quat
    cam.data.type = 'ORTHO' if ortho_scale else 'PERSP'
    if ortho_scale:
        cam.data.ortho_scale = ortho_scale
    else:
        cam.data.angle = math.radians(fov)
    cam.data.shift_x = shift_x
    cam.data.shift_y = shift_y


def look(direction, center, span, *, ortho_scale=None, roll_deg=0.0):
    direction = Vector(direction).normalized()
    set_camera(CAM, Vector(center) + direction * span * 5, center,
               ortho_scale=ortho_scale or span * 1.27, roll_deg=roll_deg)


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


def object_center(name):
    lo, hi, center, span = bounds([bpy.data.objects[name]])
    return center


def draw(scene, path, resolution, samples, *, transparent=True):
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.cycles.use_denoising = samples >= 12
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def set_visible(names):
    for obj in objects:
        obj.hide_render = obj.name not in names


def bvh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    matrix = obj.matrix_world
    verts = [matrix @ v.co for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(verts, faces, epsilon=1e-7)


def ray_hit(name, origin, direction, distance):
    hit = bvh(bpy.data.objects[name]).ray_cast(Vector(origin),
                                               Vector(direction).normalized(),
                                               distance)
    return hit[0] is not None


def ray_hit_any(names, origin, direction, distance):
    for name in names:
        if ray_hit(name, origin, direction, distance):
            return name
    return None


def ray_clear_any(names, origin, direction, distance):
    return ray_hit_any(names, origin, direction, distance) is None


def ray_hit_point(name, origin, direction, distance):
    hit = bvh(bpy.data.objects[name]).ray_cast(Vector(origin),
                                               Vector(direction).normalized(),
                                               distance)
    if hit[0] is None:
        return None
    return hit[0]


def closure(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bad = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
    bm.free()
    return bad


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


def run_ray_checks():
    plate = D['plate']
    cage = D['cage']
    drain = D['drain']
    checks = {'bracket': {}, 'cap': {}, 'neck': {}, 'cage': {}, 'filter': {},
              'collar': {}, 'bowl': {}, 'drain': {}, 'slots': {},
              'objects': {}}
    origins = [(-.70, h['y'], h['z']) for h in plate['holes']]
    checks['bracket'] = {
        'hole_0_clear': ray_clear_any(['bracket_plate'], origins[0], (1, 0, 0), 1.4),
        'hole_1_clear': ray_clear_any(['bracket_plate'], origins[1], (1, 0, 0), 1.4),
        'plate_solid': ray_hit('bracket_plate', (-.70, .0, .450), (1, 0, 0), 1.4),
        'fillet_l_present': 'bracket_fillet_l' in bpy.data.objects,
        'fillet_r_present': 'bracket_fillet_r' in bpy.data.objects,
    }
    cap = D['cap']
    checks['cap'] = {
        'solid': ray_hit('cap', (.70, .150, .2160), (-1, 0, 0), 1.4),
        'bead_present': 'cap_edge_bead' in bpy.data.objects,
    }
    checks['neck'] = {
        'core_solid': ray_hit('neck_core', (.70, .105, .2693), (-1, 0, 0), 1.4),
        'front_rib_solid': ray_hit('neck_rib_front_2', (.70, .0, .2693),
                                   (-1, 0, 0), 1.4),
        'back_rib_present': 'neck_rib_back_0' in bpy.data.objects,
    }
    # Cage windows: front pair x three rows.  The ray must miss every cage
    # part and hit the white filter element behind it.
    window_results = {}
    for column, angle in (('l', -35.0), ('r', 35.0)):
        for row, z in enumerate((.0945, .0130, -.0650)):
            a = math.radians(angle)
            origin = (1.4 * math.cos(a), 1.4 * math.sin(a), z)
            direction = (-math.cos(a), -math.sin(a), 0)
            # Stop just before the filter element so the far-side cage cannot
            # be mistaken for a blocker of the front window.
            blocker = ray_hit_any(CAGE_PARTS, origin, direction, 1.225)
            window_results[f'{column}{row}'] = {
                'clear': blocker is None,
                'cage_blocker': blocker,
                'filter_hit': ray_hit('filter_core', origin, direction, 1.225),
            }
    checks['cage'] = {
        'windows': window_results,
        'all_front_windows_open': all(v['clear'] for v in window_results.values()),
        'front_post_solid': ray_hit('cage_post_front', (1.4, 0, .100),
                                    (-1, 0, 0), 1.225),
        'side_post_solid': ray_hit('cage_post_side_r', (0, 1.4, .100),
                                   (0, -1, 0), 1.225),
        'top_band_solid': ray_hit('cage_band_top',
                                  (1.4 * math.cos(math.radians(35)),
                                   1.4 * math.sin(math.radians(35)), .150),
                                  (-math.cos(math.radians(35)),
                                   -math.sin(math.radians(35)), 0), 1.225),
        'mid_band_solid': ray_hit('cage_band_mid1',
                                  (1.4 * math.cos(math.radians(35)),
                                   1.4 * math.sin(math.radians(35)), .0540),
                                  (-math.cos(math.radians(35)),
                                   -math.sin(math.radians(35)), 0), 1.225),
    }
    checks['filter'] = {
        'core_solid': ray_hit('filter_core', (1.4, 0, .150), (-1, 0, 0), 1.225),
    }
    checks['collar'] = {
        'solid': ray_hit('collar', (.70, .130, -.1100), (-1, 0, 0), 1.4),
        'seam_present': 'collar_seam' in bpy.data.objects,
    }
    checks['bowl'] = {
        'wall_hit': ray_hit('bowl_shell', (.70, .130, -.2500), (-1, 0, 0), 1.4),
        'rib_count': len([n for n in bpy.data.objects.keys()
                          if n.startswith('bowl_rib_')]),
        'ribs_present': len([n for n in bpy.data.objects.keys()
                             if n.startswith('bowl_rib_')]) == 4,
    }
    center_hit = ray_hit_point('brass_nipple', (0, 0, -.60), (0, 0, 1), .30)
    side_hit = ray_hit_point('brass_nipple', (0, .012, -.60), (0, 0, 1), .30)
    recess_depth = (float(center_hit.z - side_hit.z)
                    if center_hit is not None and side_hit is not None else None)
    checks['drain'] = {
        'neck_solid': ray_hit('drain_neck', (.70, .030, -.3850), (-1, 0, 0), 1.4),
        'knob_solid': ray_hit('drain_knob', (.70, .030, -.4300), (-1, 0, 0), 1.4),
        'brass_solid': ray_hit('brass_nipple', (.70, .008, -.4800), (-1, 0, 0), 1.4),
        'brass_center_hit_z': round(float(center_hit.z), 5) if center_hit else None,
        'brass_side_hit_z': round(float(side_hit.z), 5) if side_hit else None,
        'brass_recess_depth_m': round(recess_depth, 5) if recess_depth else None,
        'brass_blind_recess': bool(recess_depth is not None and recess_depth > .005),
    }
    checks['slots'] = {
        'inset_count': len([n for n in bpy.data.objects.keys()
                            if n.startswith('slot_inset_')]),
        'frame_count': len([n for n in bpy.data.objects.keys()
                            if n.startswith('slot_frame_')]),
        'present': len([n for n in bpy.data.objects.keys()
                        if n.startswith('slot_inset_')]) == 4,
    }
    checks['objects'] = {
        'bracket': 'bracket_plate' in bpy.data.objects,
        'cap': 'cap' in bpy.data.objects,
        'cage_bands': len([n for n in bpy.data.objects.keys()
                           if n.startswith('cage_band_')]) == 4,
        'cage_posts': len([n for n in bpy.data.objects.keys()
                           if n.startswith('cage_post_')]) == 4,
        'filter': 'filter_core' in bpy.data.objects,
        'collar': 'collar' in bpy.data.objects,
        'bowl': 'bowl_shell' in bpy.data.objects,
        'drain': 'drain_knob' in bpy.data.objects,
        'brass': 'brass_nipple' in bpy.data.objects,
        'parts': len(objects),
    }
    return checks


def main():
    global CAM
    device = RR.get('device', 'CPU')
    try:
        scene = legacy.setup_scene(device, 512)
    except Exception as error:
        print(f'render device {device} unavailable ({error}); using CPU', flush=True)
        scene = legacy.setup_scene('CPU', 512)
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    for key, spec in request['materials'].items():
        MATS[key] = make_material(
            key, spec,
            metallic=spec.get('metallic', 1.0),
            roughness=spec.get('roughness', .5),
            coat=spec.get('coat', 0.0),
            coat_roughness=spec.get('coat_roughness', .12),
            alpha=spec.get('alpha', False))
    CAM = init_camera(scene)
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Strength'].default_value = .62
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    fill = bpy.data.objects.get('soft light')
    if fill is not None:
        fill.data.energy = 110
        fill.data.shape = 'RECTANGLE'
        fill.data.size = fill.data.size_y = 3.2
        fill.location = (-3.0, 2.4, 2.4)
        fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
    for name, location, energy, size in (
            ('main softbox', (2.4, -2.8, 3.8), 210, 3.0),
            ('front reflection', (3.0, 1.4, 1.0), 105, 1.5),
            ('top fill', (0.0, 0.0, 4.0), 80, 2.4),
            ('rim', (-3.0, 3.0, 3.2), 72, 1.6),
            ('low fill', (0.0, -3.0, -0.2), 34, 1.4)):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy = energy
        data.shape = 'RECTANGLE'
        data.size = data.size_y = size
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

    build_all()
    bpy.context.view_layer.update()
    rays = run_ray_checks()
    low, high, center, span = bounds(objects)
    shift = -center
    for obj in objects:
        obj.location += shift
    bpy.context.view_layer.update()
    if request.get('geometry_only'):
        closures = {obj.name: closure(obj) for obj in objects}
        intersections = {obj.name: intersecting_faces(obj) for obj in objects}
        debug = {
            'parts': len(objects),
            'names': [obj.name for obj in objects],
            'closures': closures,
            'closed_parts': sum(1 for v in closures.values() if v == 0),
            'self_intersecting_components': {k: v for k, v in intersections.items()
                                              if v},
            'ray_checks': rays,
            'bounds': [list(x) for x in (low, high)],
            'span': span,
            'metal_objects': metal_objects,
            'nonmetal_objects': nonmetal_objects,
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / 'geometry-debug.json').write_text(
            json.dumps(debug, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'ok': True, 'geometry_only': True,
                          'closed': debug['closed_parts'],
                          'parts': debug['parts']}, ensure_ascii=False))
        return
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    out.mkdir(parents=True, exist_ok=True)
    render_dir = out / 'renders'
    render_dir.mkdir(exist_ok=True)

    bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'),
                              export_format='GLB', use_selection=True,
                              export_yup=False, export_apply=True,
                              export_image_format='AUTO',
                              export_materials='EXPORT')
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

    low, high, center, span = bounds(objects)
    parts = [{'name': obj.name, 'vertices': len(obj.data.vertices),
              'faces': len(obj.data.polygons),
              'material': obj.data.materials[0].name if obj.data.materials else None,
              'metal': obj.name in metal_objects}
             for obj in objects]
    closures = {obj.name: closure(obj) for obj in objects}
    intersections = {obj.name: intersecting_faces(obj) for obj in objects}
    directions = {'front': (1, 0, .02), 'back': (-1, 0, .02),
                  'left': (0, -1, .02), 'right': (0, 1, .02),
                  'top': (0, 0, 1), 'bottom': (0, 0, -1)}
    for name, direction in directions.items():
        scale = (span * 1.13 if name in ('front', 'back', 'left', 'right')
                 else span * .62)
        look(direction, center, span, ortho_scale=scale)
        draw(scene, render_dir / f'new-{name}.png', RR['six_resolution'],
             RR['six_samples'])

    set_camera(CAM, center + Vector((1.9, 0, 0)), center, fov=38.05401,
               shift_x=-.000224, shift_y=.002531)
    draw(scene, render_dir / 'new-source.png', 512, RR['six_samples'])
    draw(scene, render_dir / 'new-source-alpha.png', 512, RR['six_samples'],
         transparent=True)
    source_dir = Vector((1.0, -.55, .20)).normalized()
    set_camera(CAM, center + source_dir * 1.90, center, fov=37.5)
    draw(scene, render_dir / 'new-source-3q.png', 512, RR['six_samples'])

    targets = {
        'bracket': object_center('bracket_plate'),
        'neck': object_center('neck_core'),
        'cap': object_center('cap'),
        'cage': Vector((0, 0, .055)),
        'windows': Vector((0, 0, .020)),
        'slots': Vector((0, 0, .150)),
        'collar': object_center('collar'),
        'bowl': Vector((0, 0, -.265)),
        'drain': Vector((0, 0, -.435)),
        'overall': center,
    }
    detail_specs = [
        ('bracket', (1.0, -.14, .06), .42),
        ('neck', (1.0, -.24, .15), .33),
        ('cap', (1.0, -.28, .18), .56),
        ('cage', (1.0, -.18, .04), .54),
        ('windows', (1.0, -.05, .01), .50),
        ('slots', (1.0, -.26, .10), .38),
        ('collar', (1.0, -.30, .15), .58),
        ('bowl', (1.0, -.22, .08), .56),
        ('drain', (1.0, -.25, -.08), .24),
        ('overall', (1.0, -.62, .22), span * 1.15),
    ]
    for name, direction, scale in detail_specs:
        look(direction, targets[name], span, ortho_scale=scale)
        draw(scene, render_dir / f'detail-{name}.png', RR['detail_resolution'],
             RR['detail_samples'])

    bracket_parts = {'bracket_plate', 'bracket_fillet_l', 'bracket_fillet_r'}
    set_visible(bracket_parts)
    look((1.0, -.14, .06), targets['bracket'], span, ortho_scale=.42)
    draw(scene, render_dir / 'isolate-bracket.png', RR['detail_resolution'],
         RR['detail_samples'])
    cage_parts = set(CAGE_PARTS) | {'filter_core'}
    set_visible(cage_parts)
    look((1.0, -.18, .04), targets['cage'], span, ortho_scale=.54)
    draw(scene, render_dir / 'isolate-cage.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({'cap', 'cap_edge_bead', 'neck_core'} |
                {n for n in bpy.data.objects.keys() if n.startswith('neck_rib_')} |
                {'collar', 'collar_seam'})
    look((1.0, -.28, .16), Vector((0, 0, .120)), span, ortho_scale=.62)
    draw(scene, render_dir / 'isolate-cap-collar.png',
         RR['detail_resolution'], RR['detail_samples'])
    set_visible({'bowl_shell'} |
                {n for n in bpy.data.objects.keys() if n.startswith('bowl_rib_')})
    look((1.0, -.22, .08), targets['bowl'], span, ortho_scale=.56)
    draw(scene, render_dir / 'isolate-bowl.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({'drain_neck', 'drain_knob', 'brass_nipple'})
    look((1.0, -.25, -.08), targets['drain'], span, ortho_scale=.24)
    draw(scene, render_dir / 'isolate-drain.png', RR['detail_resolution'],
         RR['detail_samples'])
    set_visible({'brass_nipple'})
    look((1.0, -.25, -.08), targets['drain'], span, ortho_scale=.12)
    draw(scene, render_dir / 'isolate-brass.png', RR['detail_resolution'],
         RR['detail_samples'])

    set_visible({obj.name for obj in objects})
    look((1.0, -.62, .28), center, span, ortho_scale=span * 1.15)
    draw(scene, render_dir / 'new-material-3q.png', RR['detail_resolution'],
         RR['detail_samples'])
    clay = bpy.data.materials.new('Clay check only')
    clay.use_nodes = True
    cb = clay.node_tree.nodes.get('Principled BSDF')
    cb.inputs['Base Color'].default_value = (.34, .35, .34, 1)
    cb.inputs['Roughness'].default_value = .82
    cb.inputs['Metallic'].default_value = 0
    original_materials = {obj.name: obj.data.materials[0] for obj in objects}
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    look(directions['front'], center, span, ortho_scale=span * 1.13)
    draw(scene, render_dir / 'new-clay-front.png', RR['six_resolution'],
         RR['six_samples'])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(original_materials[obj.name])

    set_visible({'brass_nipple'})
    main_light = bpy.data.objects['main softbox']
    for name, location in (('a', (2.4, -2.8, 3.8)), ('b', (-3.0, 4.0, 5.0))):
        main_light.location = location
        main_light.rotation_euler = (-Vector(location)).to_track_quat('-Z', 'Y').to_euler()
        look((1.0, -.25, -.08), targets['drain'], span, ortho_scale=.12)
        draw(scene, render_dir / f'metal-only-light-{name}.png', 512,
             RR['detail_samples'])
    main_light.location = (2.4, -2.8, 3.8)
    main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
    set_visible({obj.name for obj in objects})
    look(directions['front'], center, span, ortho_scale=span * 1.13)
    draw(scene, render_dir / 'new-light-alternate.png', RR['six_resolution'],
         RR['six_samples'])

    turns = RR['turn_count']
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), center, span,
             ortho_scale=span * 1.18)
        draw(scene, render_dir / f'turn-new-{i:02d}.png', RR['turn_resolution'],
             RR['turn_samples'])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), center, span,
             ortho_scale=span * 1.18)
        draw(scene, render_dir / f'turn-clay-{i:02d}.png', RR['turn_resolution'],
             RR['turn_samples'])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(original_materials[obj.name])
    scene.render.image_settings.color_mode = 'RGBA'
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), center, span,
             ortho_scale=span * 1.18)
        draw(scene, render_dir / f'sil-new-{i:02d}.png', 256, 4,
             transparent=True)

    # Old model comparison renders.
    for obj in objects:
        obj.hide_render = True
    old = legacy.load_mesh(request['original_mesh'])
    old_low, old_high, old_center, old_span = bounds([old])
    for name, direction in directions.items():
        scale = (old_span * 1.10 if name in ('front', 'back', 'left', 'right')
                 else old_span * .55)
        look(direction, old_center, old_span, ortho_scale=scale)
        draw(scene, render_dir / f'old-{name}.png', RR['six_resolution'],
             RR['six_samples'])
    set_camera(CAM, old_center + Vector((1.9, 0, 0)), old_center,
               fov=38.05401, shift_x=-.000224, shift_y=.002531)
    draw(scene, render_dir / 'old-source.png', 512, RR['six_samples'])
    draw(scene, render_dir / 'old-source-alpha.png', 512, RR['six_samples'],
         transparent=True)

    old_targets = {
        'bracket': Vector((0, 0, .430)),
        'neck': Vector((0, 0, .300)),
        'cap': Vector((0, 0, .235)),
        'cage': Vector((0, 0, .120)),
        'windows': Vector((0, 0, .100)),
        'slots': Vector((0, 0, .175)),
        'collar': Vector((0, 0, -.020)),
        'bowl': Vector((0, 0, -.230)),
        'drain': Vector((0, 0, -.430)),
        'overall': old_center,
    }
    old_scales = {
        'bracket': .42, 'neck': .33, 'cap': .56, 'cage': .54,
        'windows': .50, 'slots': .38, 'collar': .58, 'bowl': .56,
        'drain': .24, 'overall': 1.15,
    }
    for name, direction, scale in detail_specs:
        look(direction, old_targets[name] + old_center,
             old_span, ortho_scale=scale)
        draw(scene, render_dir / f'old-detail-{name}.png',
             max(320, RR['detail_resolution'] // 2),
             RR['detail_samples'] // 2)
    look((1.0, -.62, .28), old_center, old_span, ortho_scale=old_span * 1.15)
    draw(scene, render_dir / 'old-material-3q.png', RR['detail_resolution'],
         RR['detail_samples'])
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), old_center, old_span,
             ortho_scale=old_span * 1.18)
        draw(scene, render_dir / f'turn-old-{i:02d}.png', RR['turn_resolution'],
             RR['turn_samples'])
    for i in range(turns):
        a = math.tau * i / turns
        look((math.cos(a), math.sin(a), .16), old_center, old_span,
             ortho_scale=old_span * 1.18)
        draw(scene, render_dir / f'sil-old-{i:02d}.png', 256, 4,
             transparent=True)

    low, high, center, span = bounds(objects)
    data = {
        'parts': parts,
        'bounds': [list(low), list(high)],
        'centre': list(center),
        'span': span,
        'polygons': sum(len(obj.data.polygons) for obj in objects),
        'triangles': sum(max(0, len(p.vertices) - 2)
                         for obj in objects for p in obj.data.polygons),
        'closures': closures,
        'closed_parts': sum(1 for v in closures.values() if v == 0),
        'self_intersecting_components': {k: v for k, v in intersections.items()
                                          if v},
        'intersection_check': ('BVH 同部件不共享顶点的面交叠；不同零件之间故意装配相交，'
                               '未判定'),
        'metal_objects': metal_objects,
        'nonmetal_objects': nonmetal_objects,
        'ray_checks': rays,
        'old': {'bounds': [list(old_low), list(old_high)], 'span': old_span,
                'centre': list(old_center), 'mesh': request['original_mesh']},
        'materials': {key: {'name': MATS[key].name,
                            'metallic_hint': request['materials'][key].get('metallic'),
                            'roughness_hint': request['materials'][key].get('roughness'),
                            'transmission': request['materials'][key].get('transmission')}
                      for key in MATS},
        'render': RR,
        'shift': list(shift),
        'coordinate_system': 'Z up; cylinder axis Z; front camera +X; '
                             'mounting plate normal X; front cage windows at +/-35 deg',
        'status': 'manual_only_pending_review',
        'reference_role': 'missing',
    }
    (out / 'worker-result.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'ok': True, 'parts': len(objects),
                      'glb': str(out / 'mesh.glb')}, ensure_ascii=False))


main()
