"""Blender worker for the review-only black AERO air-shock candidate (item 17).

Evidence:
- one clean/rust photograph of a used black air shock with a red radial
  adjuster, a silver separator band and a bottom eye;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified), used for topology and the unseen back, never for
  measurements.

The old TripoSR library mesh is a soft blob: the top and bottom eyes are fused,
the adjuster is merged into the cap and the cylindrical can has no true holes.
It is retained unchanged and is not used as the candidate's base geometry.
This part-based rebuild models the can, a real silver separator band, the
pedestal with a V relief, a clevis-style top eye with a real through-hole, a
radial red knob with a knurled grip and domed face, a tapered lower stem and a
bottom eye with a real through-hole and dark bushing.  All candidates remain
manual-only and pending human review; nothing replaces the library mesh or
enters automatic placement.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
config = request['config']
draft = bool(request.get('draft'))
objects = []
material_by_key = {}
TAU = math.tau


# The scene reset must happen before materials and geometry are created.
try:
    scene = legacy.setup_scene('OPTIX', 512)
    render_device = 'OPTIX'
except Exception:
    scene = legacy.setup_scene('CPU', 512)
    render_device = 'CPU'
scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def image_node(nodes, key, noncolor=False):
    node = nodes.new('ShaderNodeTexImage')
    node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
    node.image.pack()
    if noncolor:
        node.image.colorspace_settings.name = 'Non-Color'
    node.interpolation = 'Linear'
    node.extension = 'REPEAT'
    return node


def make_textured_material(key, prefix, normal_strength):
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    color = image_node(nodes, prefix + '_base')
    orm = image_node(nodes, prefix + '_mr', True)
    normal = image_node(nodes, prefix + '_normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(orm.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = normal_strength
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    material_by_key[key] = material
    return material


def srgb_to_linear(value):
    channel = value / 255.0
    return channel / 12.92 if channel <= 0.04045 else \
        ((channel + 0.055) / 1.055) ** 2.4


def make_solid_material(key, rgb, roughness, metallic, srgb=False):
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    principled = material.node_tree.nodes.get('Principled BSDF')
    color = [srgb_to_linear(v) for v in rgb] if srgb else list(rgb)
    principled.inputs['Base Color'].default_value = (*color, 1.0)
    principled.inputs['Roughness'].default_value = roughness
    principled.inputs['Metallic'].default_value = metallic
    material_by_key[key] = material
    return material


make_textured_material('body', 'body', 0.24)
make_textured_material('frame', 'frame', 0.28)
make_textured_material('machined', 'machined', 0.24)
make_textured_material('grip', 'grip', 0.55)
profile = config['material_profile']
make_solid_material('knob_cap', profile['knob_cap']['base_rgb'],
                    profile['knob_cap']['roughness'], profile['knob_cap']['metallic'],
                    srgb=True)
make_solid_material('bushing', profile['bushing']['base_rgb'],
                    profile['bushing']['roughness'], profile['bushing']['metallic'],
                    srgb=True)
clay = make_solid_material('clay', (0.55, 0.56, 0.57), 0.55, 0.0)


# ---------------------------------------------------------------------------
# Mesh helpers
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
    return obj


def clean_object(obj, merge=1e-6):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if merge:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(verbose=False)
    mesh.update()


def boundary_edges(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    count = sum(1 for edge in bm.edges if not edge.is_manifold)
    bm.free()
    return count


def bevel_object(obj, width, segments=2, angle_deg=30):
    if width <= 0:
        return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new('Edge bevel', 'BEVEL')
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = 'ANGLE'
    modifier.angle_limit = math.radians(angle_deg)
    modifier.use_clamp_overlap = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    clean_object(obj, merge=1e-7)


def boolean_difference(target, cutter):
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new('Subtract hole', 'BOOLEAN')
    modifier.operation = 'DIFFERENCE'
    modifier.solver = 'EXACT'
    modifier.object = cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    clean_object(target, merge=1e-7)


def revolve_solid(name, profile_points, segments):
    verts, rings = [], []
    for r, z in profile_points:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            ids = []
            for s in range(segments):
                a = TAU * s / segments
                ids.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(ids)
    faces = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        if len(a) == 1 and len(b) > 1:
            for s in range(segments):
                faces.append((a[0], b[s], b[(s + 1) % segments]))
        elif len(a) > 1 and len(b) == 1:
            for s in range(segments):
                faces.append((a[s], a[(s + 1) % segments], b[0]))
        elif len(a) > 1 and len(b) > 1:
            for s in range(segments):
                faces.append((a[s], a[(s + 1) % segments],
                              b[(s + 1) % segments], b[s]))
    return mesh_object(name, verts, faces)


def torus_z(name, center_z, major, minor, major_segments=120, minor_segments=10):
    verts, faces = [], []
    for i in range(major_segments):
        phi = TAU * i / major_segments
        cx, cy = major * math.cos(phi), major * math.sin(phi)
        nx, ny = math.cos(phi), math.sin(phi)
        for j in range(minor_segments):
            psi = TAU * j / minor_segments
            radial = minor * math.cos(psi)
            verts.append((cx + nx * radial, cy + ny * radial,
                          center_z + minor * math.sin(psi)))
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = i * minor_segments + (j + 1) % minor_segments
            c = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            d = ((i + 1) % major_segments) * minor_segments + j
            faces.append((a, b, c, d))
    return mesh_object(name, verts, faces)


def annular_tube_x(name, x0, x1, y, z, outer, inner, segments):
    verts, faces = [], []
    for x in (x0, x1):
        for r in (outer, inner):
            for s in range(segments):
                a = TAU * s / segments
                verts.append((x, y + r * math.cos(a), z + r * math.sin(a)))
    o0, h0, o1, h1 = 0, segments, 2 * segments, 3 * segments
    for s in range(segments):
        ns = (s + 1) % segments
        faces.append((o0 + s, o0 + ns, o1 + ns, o1 + s))
        faces.append((h0 + ns, h0 + s, h1 + s, h1 + ns))
        faces.append((o0 + ns, o0 + s, h0 + s, h0 + ns))
        faces.append((o1 + s, o1 + ns, h1 + ns, h1 + s))
    return mesh_object(name, verts, faces)


def extrude_profile_x(name, points_yz, x0, x1):
    count = len(points_yz)
    verts = [(x0, y, z) for y, z in points_yz]
    verts += [(x1, y, z) for y, z in points_yz]
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, count + j, count + i))
    faces.append(tuple(range(count - 1, -1, -1)))
    faces.append(tuple(count + i for i in range(count)))
    return mesh_object(name, verts, faces)


def cylinder_x(name, x0, x1, y, z, radius, segments=96):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius,
                                        depth=(x1 - x0), end_fill_type='NGON',
                                        location=((x0 + x1) / 2, y, z),
                                        rotation=(0, math.pi / 2, 0))
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    clean_object(obj, merge=1e-7)
    return obj


def box_mesh(name, center, size):
    x, y, z = center
    sx, sy, sz = (s / 2 for s in size)
    verts = [(x - sx, y - sy, z - sz), (x + sx, y - sy, z - sz),
             (x + sx, y + sy, z - sz), (x - sx, y + sy, z - sz),
             (x - sx, y - sy, z + sz), (x + sx, y - sy, z + sz),
             (x + sx, y + sy, z + sz), (x - sx, y + sy, z + sz)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def lofted_box(name, z0, z1, half_x0, half_y0, half_x1, half_y1):
    verts = [(-half_x0, -half_y0, z0), (half_x0, -half_y0, z0),
             (half_x0, half_y0, z0), (-half_x0, half_y0, z0),
             (-half_x1, -half_y1, z1), (half_x1, -half_y1, z1),
             (half_x1, half_y1, z1), (-half_x1, half_y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def transform_mesh(obj, matrix):
    obj.data.transform(matrix)
    obj.data.update()


# ---------------------------------------------------------------------------
# UV / shading
# ---------------------------------------------------------------------------
def ensure_uv(obj, name='UVMap'):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name=name)
    return obj.data.uv_layers.active.data


def cylindrical_uv(obj, z0, z1, u_offset=0.5, clockwise=True):
    """Wrap the texture around Z.

    clockwise=True gives u = -atan2(y,x)/2pi + u_offset.  Viewed from +X, the
    texture then reads left-to-right, and the seam sits at -X (back) when
    u_offset is 0.5.
    """
    uv = ensure_uv(obj)
    mesh = obj.data
    for poly in mesh.polygons:
        angles = []
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            angles.append(math.atan2(co.y, co.x))
        ca = math.atan2(sum(math.sin(a) for a in angles),
                        sum(math.cos(a) for a in angles))
        for loop_index, angle in zip(poly.loop_indices, angles):
            delta = (angle - ca + math.pi) % TAU - math.pi
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            if clockwise:
                u = -(ca + delta) / TAU + u_offset
            else:
                u = (ca + delta) / TAU + u_offset
            v = (co.z - z0) / max(1e-9, z1 - z0)
            uv[loop_index].uv = (u, v)


def box_uv(obj, scale=0.12):
    uv = ensure_uv(obj)
    mesh = obj.data
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


def set_smooth(obj, mode='all', axis=None):
    if mode == 'flat':
        for poly in obj.data.polygons:
            poly.use_smooth = False
        return
    if mode == 'side' and axis is not None:
        axis = Vector(axis).normalized()
        for poly in obj.data.polygons:
            poly.use_smooth = abs(poly.normal.dot(axis)) < 0.80
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True


def assign(obj, key, *, uv='box', z0=None, z1=None, smooth='all', axis=None,
           box_scale=0.12, u_offset=0.5, clockwise=True):
    obj.data.materials.clear()
    obj.data.materials.append(material_by_key[key])
    if uv == 'cylinder':
        if z0 is None or z1 is None:
            lo = min(v.co.z for v in obj.data.vertices)
            hi = max(v.co.z for v in obj.data.vertices)
            z0, z1 = lo, hi
        cylindrical_uv(obj, z0, z1, u_offset=u_offset, clockwise=clockwise)
    elif uv == 'box':
        box_uv(obj, box_scale)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
D = config['dimensions']
SEG = config['radial_segments']
F = config['features']
BODY_Z0, BODY_Z1 = D['body_z0_m'], D['body_z1_m']

body = revolve_solid('Canister body', config['profiles']['body'], SEG)
assign(body, 'body', uv='cylinder', z0=BODY_Z0, z1=BODY_Z1,
       smooth='side', axis=(0, 0, 1), clockwise=False)

upper = revolve_solid('Top cap', config['profiles']['upper'], SEG)
assign(upper, 'frame', uv='box', smooth='side', axis=(0, 0, 1))

silver = revolve_solid('Silver separator band',
                       [[0.0, D['silver_ring_z0_m']],
                        [0.0305, D['silver_ring_z0_m']],
                        [0.0305, D['silver_ring_z1_m']],
                        [0.0, D['silver_ring_z1_m']]], SEG)
assign(silver, 'machined', uv='cylinder', z0=D['silver_ring_z0_m'],
       z1=D['silver_ring_z1_m'], smooth='side', axis=(0, 0, 1))

lip = revolve_solid('Bottom machined lip',
                    [[0.0, D['bottom_lip_z0_m']],
                     [0.0305, D['bottom_lip_z0_m']],
                     [0.0305, D['bottom_lip_z1_m']],
                     [0.0, D['bottom_lip_z1_m']]], SEG)
assign(lip, 'machined', uv='cylinder', z0=D['bottom_lip_z0_m'],
       z1=D['bottom_lip_z1_m'], smooth='side', axis=(0, 0, 1))

stem = revolve_solid('Lower stem', config['profiles']['lower_stem'], SEG)
assign(stem, 'frame', uv='cylinder', z0=0.0245, z1=0.0658,
       smooth='side', axis=(0, 0, 1), box_scale=0.12)

# Top pedestal: base plate, tapered neck, real V relief on the +X face.
P = F['top_pedestal']
plate = box_mesh('Top pedestal base',
                 (0, 0, (P['base_z0_m'] + P['base_z1_m']) / 2),
                 (P['base_depth_x_m'], P['base_width_y_m'],
                  P['base_z1_m'] - P['base_z0_m']))
bevel_object(plate, P['bevel_m'], segments=3)

neck = lofted_box('Top pedestal neck', P['neck_z0_m'], P['neck_z1_m'],
                  P['neck_bottom_depth_x_m'] / 2, P['neck_bottom_width_y_m'] / 2,
                  P['neck_top_depth_x_m'] / 2, P['neck_top_width_y_m'] / 2)
bevel_object(neck, P['bevel_m'], segments=2)
V = P['v_notch']
wedge = extrude_profile_x('V notch cutter',
                          [(-V['top_half_width_y_m'], V['top_z_m']),
                           (V['top_half_width_y_m'], V['top_z_m']),
                           (0.0, V['apex_z_m'])],
                          V['front_x_m'], V['back_x_m'])
boolean_difference(neck, wedge)
assign(neck, 'frame', uv='box', smooth='all', box_scale=0.03)
assign(plate, 'frame', uv='box', smooth='all', box_scale=0.03)

# Top eye: thick clevis-style lug with a real through-hole along X.
E = F['top_eye']
arc = []
arc_center_z = E['arc_center_z_m']
for step in range(49):
    a = math.pi - math.pi * step / 48
    arc.append((E['half_width_m'] * math.cos(a),
                arc_center_z + E['half_width_m'] * math.sin(a)))
eye_points = [(-E['half_width_m'], E['bottom_z_m']),
              (-E['half_width_m'], arc_center_z)] + arc[1:] + \
             [(E['half_width_m'], E['bottom_z_m'])]
top_eye = extrude_profile_x('Top eye lug', eye_points,
                            -E['x_half_thickness_m'], E['x_half_thickness_m'])
top_eye_bevel = E['bevel_m']
bevel_object(top_eye, top_eye_bevel, segments=2, angle_deg=28)
cutter = cylinder_x('Top eye hole cutter',
                    -E['x_half_thickness_m'] - 0.005, E['x_half_thickness_m'] + 0.005,
                    0.0, D['top_hole_z_m'], E['hole_radius_m'], 128)
boolean_difference(top_eye, cutter)
top_eye_boundary = boundary_edges(top_eye)
assign(top_eye, 'frame', uv='box', smooth='all', box_scale=0.03)

top_rim_r = E['rim']
top_rims = []
for label, x0, x1 in (('Top eye rim +X', top_rim_r['x0_m'], top_rim_r['x1_m']),
                      ('Top eye rim -X', -top_rim_r['x1_m'], -top_rim_r['x0_m'])):
    rim = annular_tube_x(label, x0, x1, 0.0, D['top_hole_z_m'],
                         top_rim_r['outer_radius_m'], top_rim_r['inner_radius_m'],
                         top_rim_r['segments'])
    assign(rim, 'machined', uv='box', smooth='side', axis=(1, 0, 0),
           box_scale=0.02)
    top_rims.append(rim)

TBUS = E['bushing']
top_bushing = annular_tube_x('Top eye bushing', TBUS['x0_m'], TBUS['x1_m'],
                             0.0, D['top_hole_z_m'], TBUS['outer_radius_m'],
                             TBUS['inner_radius_m'], TBUS['segments'])
assign(top_bushing, 'bushing', uv='box', smooth='side', axis=(1, 0, 0),
       box_scale=0.02)

# Radial adjuster knob: black boss, ribbed collar, red knurled grip, red domed cap.
K = F['knob']
azimuth = math.radians(K['azimuth_deg'])
knob_dir = Vector((math.cos(azimuth), math.sin(azimuth), 0.0))
knob_origin = Vector((0.0, 0.0, K['center_z_m']))
knob_matrix = Matrix.Translation(knob_origin) @ \
    Vector((0, 0, 1)).rotation_difference(knob_dir).to_matrix().to_4x4()

boss = revolve_solid('Knob boss', config['profiles']['knob_boss'], 96)
assign(boss, 'frame', uv='cylinder', z0=0.006, z1=0.030,
       smooth='side', axis=(0, 0, 1), box_scale=0.02)
transform_mesh(boss, knob_matrix)

collar = torus_z('Knob collar ring', K['collar_z_m'],
                 K['collar_major_radius_m'], K['collar_minor_radius_m'],
                 96, 10)
assign(collar, 'frame', uv='box', smooth='all', box_scale=0.02)
transform_mesh(collar, knob_matrix)

grip = revolve_solid('Knob grip', config['profiles']['knob_grip'], 128)
assign(grip, 'grip', uv='cylinder', z0=0.030, z1=0.039,
       smooth='side', axis=(0, 0, 1), box_scale=0.02)
transform_mesh(grip, knob_matrix)

knob_seam = torus_z('Knob seam ring', K['seam_z_m'],
                    K['seam_major_radius_m'], K['seam_minor_radius_m'],
                    96, 8)
assign(knob_seam, 'frame', uv='box', smooth='all', box_scale=0.02)
transform_mesh(knob_seam, knob_matrix)

knob_cap = revolve_solid('Knob domed cap', config['profiles']['knob_cap'], 128)
assign(knob_cap, 'knob_cap', uv='box', smooth='side', axis=(0, 0, 1),
       box_scale=0.02)
transform_mesh(knob_cap, knob_matrix)

# Bottom eye: capsule profile with a real through-hole and a dark bushing.
B = F['bottom_eye']
bottom_points = [(-B['capsule_half_width_m'], B['capsule_center_z_m'])]
for step in range(1, 48):
    a = math.pi + math.pi * step / 48
    bottom_points.append((B['capsule_half_width_m'] * math.cos(a),
                          B['capsule_center_z_m'] +
                          B['capsule_half_width_m'] * math.sin(a)))
bottom_points += [(B['capsule_half_width_m'], B['capsule_center_z_m']),
                  (0.0120, 0.0230), (B['flare_half_width_m'], B['top_z_m']),
                  (-B['flare_half_width_m'], B['top_z_m']),
                  (-0.0120, 0.0230)]
bottom_eye = extrude_profile_x('Bottom eye lug', bottom_points,
                               -B['x_half_thickness_m'], B['x_half_thickness_m'])
bevel_object(bottom_eye, B['bevel_m'], segments=2, angle_deg=28)
cutter = cylinder_x('Bottom eye hole cutter',
                    -B['x_half_thickness_m'] - 0.005, B['x_half_thickness_m'] + 0.005,
                    0.0, D['bottom_hole_z_m'], B['hole_radius_m'], 128)
boolean_difference(bottom_eye, cutter)
bottom_eye_boundary = boundary_edges(bottom_eye)
assign(bottom_eye, 'frame', uv='box', smooth='all', box_scale=0.03)

bottom_rim_r = B['rim']
for label, x0, x1 in (('Bottom eye rim +X', bottom_rim_r['x0_m'], bottom_rim_r['x1_m']),
                      ('Bottom eye rim -X', -bottom_rim_r['x1_m'], -bottom_rim_r['x0_m'])):
    rim = annular_tube_x(label, x0, x1, 0.0, D['bottom_hole_z_m'],
                         bottom_rim_r['outer_radius_m'], bottom_rim_r['inner_radius_m'],
                         bottom_rim_r['segments'])
    assign(rim, 'machined', uv='box', smooth='side', axis=(1, 0, 0),
           box_scale=0.02)

BUS = B['bushing']
bushing = annular_tube_x('Bottom bushing', BUS['x0_m'], BUS['x1_m'],
                         0.0, D['bottom_hole_z_m'], BUS['outer_radius_m'],
                         BUS['inner_radius_m'], BUS['segments'])
assign(bushing, 'bushing', uv='box', smooth='side', axis=(1, 0, 0),
       box_scale=0.02)

bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Self-intersection / geometry records
# ---------------------------------------------------------------------------
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


def world_bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


body_radii = []
for vertex in body.data.vertices:
    radius = math.hypot(vertex.co.x, vertex.co.y)
    if radius > 0.02:
        body_radii.append(radius)
body_radii = np.asarray(body_radii, dtype=np.float64)
radial_deviation = float(np.max(np.abs(body_radii - D['body_radius_m'])))
ring_map = {}
for vertex in body.data.vertices:
    radius = math.hypot(vertex.co.x, vertex.co.y)
    if radius <= 0.02:
        continue
    key = round(vertex.co.z, 6)
    ring_map.setdefault(key, []).append(radius)
ring_variation = max((max(values) - min(values) for key, values in ring_map.items()
                      if 0.0675 < key < 0.1031), default=0.0)

low, high, center, span = world_bounds(objects)
Z_SHIFT = -low.z
for obj in objects:
    obj.location.z += Z_SHIFT
bpy.context.view_layer.update()
low, high, center, span = world_bounds(objects)

parts = []
for obj in objects:
    parts.append({'name': obj.name,
                  'vertices': len(obj.data.vertices),
                  'faces': len(obj.data.polygons),
                  'boundary_edges': boundary_edges(obj),
                  'self_intersections': intersecting_faces(obj)})
intersections = {item['name']: item['self_intersections']
                 for item in parts if item['self_intersections']}

# ---------------------------------------------------------------------------
# Export and save
# ---------------------------------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
out.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'), export_format='GLB',
                          use_selection=True, export_yup=False, export_apply=True,
                          export_materials='EXPORT')
bpy.ops.file.pack_all()


# ---------------------------------------------------------------------------
# Render setup
# ---------------------------------------------------------------------------
R = config['render']
background = scene.world.node_tree.nodes.get('Background')
background.inputs['Strength'].default_value = R['world_strength']
background.inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
light_specs = (
    ('large softbox', (2.6, -3.0, 3.6), R['key_energy'], 3.0, 3.0),
    ('fill softbox', (-3.2, 2.6, 2.2), R['fill_energy'], 3.0, 3.0),
    ('front strip', (3.2, 1.0, 1.3), R['front_strip_energy'], 1.2, 3.4),
    ('rim high', (-1.2, 3.2, 3.4), R['rim_energy'], 1.4, 2.2),
    ('low fill', (0, -2.6, -0.28), R['low_fill_energy'], 1.0, 1.0),
    ('bottom fill', (0.4, 0.2, -2.8), R['bottom_fill_energy'], 2.0, 2.0),
)
for name, location, energy, width, height in light_specs:
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = width
    data.size_y = height
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))


# ---------------------------------------------------------------------------
# Camera and render helpers
# ---------------------------------------------------------------------------
def camera(scene, direction, target, ortho_scale):
    direction = Vector(direction).normalized()
    cam = legacy.make_camera(scene, Vector(target) + direction * 3.0,
                             target, ortho_scale=ortho_scale)
    if abs(direction.z) > 0.999:
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.rotation_quaternion @ Quaternion((0, 0, 1), math.pi / 2)
    return cam


def draw(scene, path, *, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_view(name, direction, target, scale, *, resolution=768, samples=32,
                path=None):
    cam = camera(scene, direction, target, scale)
    draw(scene, path or (render_dir / f'{name}.png'),
         resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)


render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
directions = {'front': (1, 0, 0.16), 'back': (-1, 0, 0.16),
              'left': (0, -1, 0.18), 'right': (0, 1, 0.18),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
height_scale = D['total_height_m'] * 1.18
diameter_scale = D['body_diameter_m'] * 1.90
center_z = center.z

for name, direction in directions.items():
    scale = diameter_scale if name in ('top', 'bottom') else height_scale
    render_view(f'new-{name}', direction, center, scale, resolution=768, samples=36)

factor = None
source_fit = 1.0
source_correction = 1.0
source_shift_px = 0.0
source_shift_world = 0.0
source_vertical_shift_px = 0.0
source_vertical_shift_world = 0.0
trial_height = None
height_px = None

if draft:
    render_view('new-source', (1.0, 0.0, 0.05), center, height_scale,
                resolution=512, samples=12)
else:
    # Detail renders.
    knob_xy = (math.cos(azimuth) * 0.037, math.sin(azimuth) * 0.037)
    details = (
        ('top-eye', (0.6, -0.7, 0.45), (0, 0, 0.170), 0.062),
        ('top-eye-hole', (0.75, -0.45, 0.20), (0, 0, D['top_hole_z_m']), 0.040),
        ('top-pedestal', (0.5, -0.8, 0.20), (0, 0, 0.145), 0.080),
        ('knob', (0.9, -0.35, 0.30), (knob_xy[0], knob_xy[1], K['center_z_m']), 0.055),
        ('knob-knurl', (0.8, -0.5, 0.20), (knob_xy[0], knob_xy[1], K['center_z_m']), 0.028),
        ('silver-ring', (0.7, -0.6, 0.25), (0, 0, 0.1046), 0.080),
        ('aero-logo', (0.707, 0.707, 0.10), (0.0219, 0.0219, 0.086), 0.060),
        ('bottom-lip', (0.6, -0.7, 0.20), (0, 0, 0.064), 0.070),
        ('lower-stem', (0.5, -0.8, 0.15), (0, 0, 0.040), 0.090),
        ('bottom-eye', (0.7, -0.7, 0.0), (0, 0, 0.012), 0.055),
        ('bottom-eye-hole', (0.75, -0.50, 0.10), (0, 0, D['bottom_hole_z_m']), 0.035),
        ('surface', (0.9, -0.5, 0.10), (0.024, -0.014, 0.080), 0.035),
        ('top-detail', (0.2, -0.3, 1.0), (0, 0, 0.160), 0.105),
    )
    for name, direction, target, scale in details:
        render_view(f'detail-{name}', direction, target, scale,
                    resolution=768, samples=32)

    # Isolated structure renders.
    top_mount_names = {'Top pedestal base', 'Top pedestal neck', 'Top eye lug',
                       'Top eye rim +X', 'Top eye rim -X', 'Top eye bushing',
                       'Knob boss', 'Knob collar ring', 'Knob grip',
                       'Knob seam ring', 'Knob domed cap'}
    bottom_eye_names = {'Lower stem', 'Bottom eye lug', 'Bottom eye rim +X',
                        'Bottom eye rim -X', 'Bottom bushing'}
    body_names = {'Canister body', 'Silver separator band', 'Top cap',
                  'Bottom machined lip'}
    isolated_groups = {
        'top-mount-only': (top_mount_names, (0.7, -0.7, 0.25), (0, 0, 0.155), 0.100),
        'bottom-eye-only': (bottom_eye_names, (0.7, -0.7, 0.10), (0, 0, 0.032), 0.085),
        'body-only': (body_names, (0.7, -0.7, 0.25), (0, 0, 0.100), 0.120),
    }
    for label, (names, direction, target, scale) in isolated_groups.items():
        for obj in objects:
            obj.hide_render = obj.name not in names
        render_view(label, direction, target, scale, resolution=640, samples=24)
        if label == 'body-only':
            render_view('body-only-top', (0.0, 0.0, 1.0), (0, 0, 0.100), 0.085,
                        resolution=640, samples=24)
        if label == 'bottom-eye-only':
            render_view('bottom-eye-bottom', (0.0, 0.0, -1.0),
                        (0, 0, 0.015), 0.070, resolution=640, samples=24)
    for obj in objects:
        obj.hide_render = False

    # Material / lighting comparisons.
    render_view('new-material-close', (2.3, -2.0, 1.0), (0, 0, 0.090), 0.120,
                resolution=1024, samples=40)
    key = bpy.data.objects['large softbox']
    key.location = (-3.4, 2.8, 4.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    render_view('new-light-alternate', directions['front'], center, height_scale,
                resolution=768, samples=32)
    render_view('new-surface-alternate', (2.3, -2.0, 1.0), (0, 0, 0.090), 0.120,
                resolution=1024, samples=40)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

    # Metal reflection response under a low-ambient, strong, moved key light.
    # This is a dedicated reflectivity test, not the appearance reference.
    key = bpy.data.objects['large softbox']
    lights = {name: bpy.data.objects[name] for name, *_ in light_specs}
    saved_world = background.inputs['Strength'].default_value
    saved_lights = {name: (obj.data.energy, obj.data.size, obj.data.size_y)
                    for name, obj in lights.items()}
    saved_key_location = key.location.copy()
    saved_key_rotation = key.rotation_euler.copy()

    def low_ambient():
        background.inputs['Strength'].default_value = 0.08
        for name, obj in lights.items():
            obj.data.energy = 0.0 if name != 'large softbox' else 1500.0
        key.data.size = 0.8
        key.data.size_y = 0.8
        bpy.context.view_layer.update()

    def place_key(position):
        key.location = position
        key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()

    metal_names = {'Silver separator band', 'Bottom machined lip',
                   'Top eye rim +X', 'Top eye rim -X',
                   'Bottom eye rim +X', 'Bottom eye rim -X'}
    for obj in objects:
        obj.hide_render = obj.name not in metal_names
    low_ambient()
    place_key((2.4, -2.2, 2.6))
    render_view('metal-ring-key', (1.0, -0.2, 0.06), (0, 0, 0.1046), 0.060,
                resolution=768, samples=36)
    place_key((-2.4, 2.2, 2.6))
    render_view('metal-ring-alt', (1.0, -0.2, 0.06), (0, 0, 0.1046), 0.060,
                resolution=768, samples=36)

    for obj in objects:
        obj.hide_render = obj.name not in {'Knob grip', 'Knob domed cap'}
    place_key((2.4, -2.2, 2.6))
    render_view('knob-key', (0.6, -0.8, 0.30), (knob_xy[0], knob_xy[1], K['center_z_m']),
                0.050, resolution=768, samples=36)
    place_key((-2.4, 2.2, 2.6))
    render_view('knob-alt', (0.6, -0.8, 0.30), (knob_xy[0], knob_xy[1], K['center_z_m']),
                0.050, resolution=768, samples=36)
    for obj in objects:
        obj.hide_render = False
    background.inputs['Strength'].default_value = saved_world
    for name, (energy, size, size_y) in saved_lights.items():
        lights[name].data.energy = energy
        lights[name].data.size = size
        lights[name].data.size_y = size_y
    key.location = saved_key_location
    key.rotation_euler = saved_key_rotation
    bpy.context.view_layer.update()

    # Clay six views.
    original_materials = {}
    for obj in objects:
        original_materials[obj.name] = [slot.material for slot in obj.material_slots]
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    for name, direction in directions.items():
        scale = diameter_scale if name in ('top', 'bottom') else height_scale
        render_view(f'clay-{name}', direction, center, scale,
                    resolution=640, samples=20)
    for obj in objects:
        obj.data.materials.clear()
        for material in original_materials[obj.name]:
            obj.data.materials.append(material)
    bpy.context.view_layer.update()

    # Turntable.
    for i in range(24):
        angle = TAU * i / 24
        render_view(f'turn-{i:02d}', (math.cos(angle), math.sin(angle), 0.14),
                    center, height_scale, resolution=512, samples=10)

    # Source-photo framing diagnostic (not an acceptance gate).
    for obj in objects:
        obj.hide_render = True
    old = legacy.load_mesh(request['original_mesh'])
    bpy.context.view_layer.update()
    oldlow, oldhigh, oldcenter, oldspan = world_bounds([old])
    old_span_front = max(oldhigh.y - oldlow.y, oldhigh.z - oldlow.z)
    old.location += center - oldcenter
    bpy.context.view_layer.update()
    for obj in objects:
        obj.hide_render = False
    old.hide_render = True
    source = request['source_camera']
    factor = span / max(1e-6, oldspan)
    from bpy_extras.object_utils import world_to_camera_view
    cam = legacy.make_camera(scene, (1.9 * factor, 0, center.z),
                             (0, 0, center.z), fov=source['fov_deg'])
    cam.data.shift_x = source.get('shift_x', 0)
    cam.data.shift_y = source.get('shift_y', 0)
    bpy.context.view_layer.update()
    corners = [Vector((x, y, z)) for x in (low.x, high.x)
               for y in (low.y, high.y) for z in (low.z, high.z)]
    projected = [world_to_camera_view(scene, cam, p) for p in corners]
    height_px = (max(v.y for v in projected) - min(v.y for v in projected)) * 512
    source_bbox = request.get('source_alpha_bbox')
    source_fit = 1.0
    source_correction = 1.0
    source_shift_px = 0.0
    source_shift_world = 0.0
    source_vertical_shift_px = 0.0
    source_vertical_shift_world = 0.0
    trial_height = height_px
    if source_bbox:
        source_h = max(1, source_bbox[3] - source_bbox[1] + 1)
        source_fit = height_px / source_h
        cam.location.x *= source_fit
        bpy.context.view_layer.update()
        trial = render_dir / 'new-source-trial.png'
        draw(scene, trial, resolution=512, samples=10)
        image = bpy.data.images.load(str(trial))
        w, h = image.size
        pixels = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        mask = pixels.reshape(h, w, 4)[..., 3] >= 0.5
        rows = np.nonzero(mask.any(axis=1))[0]
        trial_height = int(rows.max() - rows.min() + 1) if len(rows) else 1
        source_correction = trial_height / source_h
        cam.location.x *= source_correction
        bpy.context.view_layer.update()
        columns = np.nonzero(mask.any(axis=0))[0]
        trial_cx = ((columns.min() + columns.max() + 1) / 2) if len(columns) else w / 2
        source_cx = (source_bbox[0] + source_bbox[2] + 1) / 2
        source_shift_px = (trial_cx - source_cx) / max(1e-6, source_correction)
        visible_width = 2 * abs(cam.location.x) * math.tan(math.radians(source['fov_deg']) / 2)
        source_shift_world = source_shift_px / w * visible_width
        cam.location.y += source_shift_world
        trial_cy = (rows.min() + rows.max() + 1) / 2
        source_cy = h - (source_bbox[1] + source_bbox[3] + 1) / 2
        source_vertical_shift_px = (trial_cy - source_cy) / max(1e-6, source_correction)
        source_vertical_shift_world = source_vertical_shift_px / h * visible_width
        cam.location.z += source_vertical_shift_world
        bpy.context.view_layer.update()
        bpy.data.images.remove(image)
    draw(scene, render_dir / 'new-source.png', resolution=512, samples=28)
    bpy.data.objects.remove(cam, do_unlink=True)

    # Old model at the same neutral lighting.
    for obj in objects:
        obj.hide_render = True
    old.hide_render = False
    for name, direction in directions.items():
        old_scale = old_span_front * 1.18 if name in ('front', 'back') else oldspan * 1.18
        if name in ('top', 'bottom'):
            old_scale = max(oldhigh.x - oldlow.x, oldhigh.y - oldlow.y) * 1.70
        cam = camera(scene, direction, center, old_scale)
        draw(scene, render_dir / f'old-{name}.png', resolution=768, samples=28)
        bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), center, oldspan * 0.42)
    draw(scene, render_dir / 'old-material-close.png', resolution=1024, samples=28)
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.objects.remove(old, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

data = {
    'parts': parts,
    'bounds': [list(low), list(high)],
    'span': span,
    'parts_count': len(parts),
    'polygons': sum(len(obj.data.polygons) for obj in objects),
    'triangles': sum(max(0, len(poly.vertices) - 2) for obj in objects
                     for poly in obj.data.polygons),
    'materials': sorted(material_by_key),
    'render_device': render_device,
    'self_intersecting_components': intersections,
    'intersection_check': 'BVH same-part face overlap; deliberate assembly overlap between parts is not judged',
    'geometry_basis': 'source-photo proportions with an uncalibrated visual reference for the back and hidden topology',
    'render_samples': {'views': 36, 'details': 32, 'turntable': 10},
    'view_directions': {k: list(v) for k, v in directions.items()},
    'view_note': ('Camera at +X is the source-photo front; the AERO print is centred at -45 deg '
                  'azimuth so it appears on the right of the front view as in the photograph. '
                  'The old model is rotated/centred for comparison only; its file is unchanged.'),
    'top_eye_boundary_edges_after_boolean': top_eye_boundary,
    'bottom_eye_boundary_edges_after_boolean': bottom_eye_boundary,
    'body_radius_max_abs_deviation_m': radial_deviation,
    'body_ring_radius_variation_m': ring_variation,
    'source_camera_note': ('source-photo framing distance scaled by new/old span, '
                           'then fitted to the source-alpha height; diagnostic only'),
    'source_camera_factor': factor,
    'source_framing_fit': source_fit,
    'source_framing_trial_height_px': trial_height,
    'source_framing_correction': source_correction,
    'source_framing_shift_px': round(float(source_shift_px), 3),
    'source_framing_shift_world': round(float(source_shift_world), 5),
    'source_framing_vertical_shift_px': round(float(source_vertical_shift_px), 3),
    'source_framing_vertical_shift_world': round(float(source_vertical_shift_world), 5),
    'source_framed_height_px': height_px,
    'z_shift_m': round(float(Z_SHIFT), 9),
    'final_hole_z_m': {'top': round(float(D['top_hole_z_m'] + Z_SHIFT), 9),
                       'bottom': round(float(D['bottom_hole_z_m'] + Z_SHIFT), 9)},
    'knob_extent_m': 0.0462,
    'dimensions': {key: D[key] for key in
                   ('total_height_m', 'body_radius_m', 'body_diameter_m',
                    'body_z0_m', 'body_z1_m', 'upper_z0_m', 'upper_z1_m',
                    'silver_ring_z0_m', 'silver_ring_z1_m',
                    'bottom_lip_z0_m', 'bottom_lip_z1_m',
                    'top_hole_z_m', 'top_hole_radius_m',
                    'bottom_hole_z_m', 'bottom_hole_radius_m',
                    'knob_center_z_m', 'knob_azimuth_deg', 'eye_to_eye_m')},
    'draft': draft,
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
print(json.dumps({'status': 'rendered', 'parts': len(parts),
                  'render_device': render_device, 'draft': draft},
                 ensure_ascii=False), flush=True)
