"""Blender worker for the review-only red gas-cylinder candidate (item 20).

Evidence:
- one straight-on 1254 px clean photograph and its matching rust derivative of
  a tall red painted steel gas cylinder with a vertically stacked white CCS
  mark, a domed valve guard with a real oval hand-hole, a collar/neck, a
  rounded bottom and localised rust;
- one uncalibrated six-view sheet matched by appearance only
  (visual_match_unverified).  It is used for topology and the unseen back, never
  for measurements, and its open-looking guard and narrower proportions
  conflict with the photograph.

The old TripoSR library mesh is a soft low-poly blob: the guard is fused, the
oval hand-hole is filled (the source-fidelity audit records one filled hole),
and the surface is a rough photo-projected shell.  It is retained unchanged and
is not used as the candidate's base geometry.  The candidate is a parametric
part rebuild with a true body of revolution, a hollow guard with a real oval
front hand-hole and a solid far wall, a separate brass valve and dark steel neck
seal, a rounded convex bottom with a small contact pad, and red enamel PBR
textures that keep the CCS print and local rust.  All candidates remain
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


def make_textured_material(key, prefix, normal_strength, coat=0.0, coat_roughness=0.1):
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
    if 'Coat Weight' in principled.inputs:
        principled.inputs['Coat Weight'].default_value = coat
    if 'Coat Roughness' in principled.inputs:
        principled.inputs['Coat Roughness'].default_value = coat_roughness
    if 'Specular IOR Level' in principled.inputs:
        principled.inputs['Specular IOR Level'].default_value = 0.22
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


profile = config['material_profile']
make_textured_material('body', 'body', 0.18,
                       profile['body']['coat'], profile['body']['coat_roughness'])
make_textured_material('guard', 'guard', 0.18,
                       profile['guard']['coat'], profile['guard']['coat_roughness'])
make_textured_material('steel', 'steel', 0.30)
make_solid_material('brass', profile['brass']['base_rgb'],
                    profile['brass']['roughness'], profile['brass']['metallic'],
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


def boolean_difference(target, cutter):
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new('Subtract structure', 'BOOLEAN')
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


def cylinder_z(name, z0, z1, radius, segments=64, sides=None):
    vertices = sides or segments
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius,
                                        depth=(z1 - z0), end_fill_type='NGON',
                                        location=(0, 0, (z0 + z1) / 2))
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    clean_object(obj, merge=1e-7)
    return obj


def stadium_prism(name, center_y, center_z, half_width, half_height,
                  corner_radius, x0, x1, arc_steps=20):
    """Rounded-slot (stadium) cross-section in YZ, extruded along X."""
    points = []
    straight = max(0.0, half_height - corner_radius)
    # right straight side, bottom -> top
    for k in range(9):
        z = center_z - straight + 2 * straight * k / 8
        points.append((center_y + half_width, z))
    # top semicircle, +Y -> -Y
    for k in range(1, arc_steps):
        a = math.pi * k / arc_steps
        points.append((center_y + corner_radius * math.cos(a),
                       center_z + straight + corner_radius * math.sin(a)))
    # left straight side, top -> bottom
    for k in range(9):
        z = center_z + straight - 2 * straight * k / 8
        points.append((center_y - half_width, z))
    # bottom semicircle, -Y -> +Y
    for k in range(1, arc_steps):
        a = math.pi + math.pi * k / arc_steps
        points.append((center_y + corner_radius * math.cos(a),
                       center_z - straight + corner_radius * math.sin(a)))
    n = len(points)
    verts = [(x0, y, z) for y, z in points] + [(x1, y, z) for y, z in points]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(n + i for i in range(n)))
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
P = config['profiles']
BODY_Z1 = D['neck_z1_m']

body = revolve_solid('Cylinder body', P['body'], SEG)
assign(body, 'body', uv='cylinder', z0=0.0, z1=BODY_Z1,
       smooth='side', axis=(0, 0, 1), clockwise=False)

# Hollow valve guard: solid outer dome minus an inner cavity minus a real oval
# hand-hole in the photograph-facing wall only.  The photograph shows a dark
# interior behind the slot, so the far wall is kept solid; the uncalibrated
# six-view sheet repeats a slot in several panels, which is internally
# inconsistent (left/right cannot both show the same front slot) and is not
# followed without photographic support.
guard = revolve_solid('Valve guard', [[0.0, D['guard_z0_m']]] + P['guard_outer'], SEG)
cavity = revolve_solid('Guard cavity cutter', P['guard_cavity'], SEG)
boolean_difference(guard, cavity)
window = stadium_prism('Guard hand-hole cutter',
                       D['window_center_y_m'], D['window_center_z_m'],
                       D['window_half_width_y_m'], D['window_half_height_z_m'],
                       D['window_corner_radius_m'], 0.018, 0.080)
boolean_difference(guard, window)
guard_boundary = boundary_edges(guard)
assign(guard, 'guard', uv='cylinder', z0=D['guard_z0_m'], z1=D['guard_z1_m'],
       smooth='all', clockwise=False)

# Dark steel seal ring at the guard/neck joint.
seal = torus_z('Neck seal ring', D['neck_seal_z_m'],
               D['neck_seal_major_radius_m'], D['neck_seal_minor_radius_m'],
               160, 10)
assign(seal, 'steel', uv='box', smooth='all', box_scale=0.03)

# Valve visible through the guard window.  The exact valve make and its internal
# mechanism are not visible; this is a conservative service valve silhouette.
valve_body = cylinder_z('Valve body', D['valve_body_z0_m'], D['valve_body_z1_m'],
                        D['valve_body_radius_m'], segments=64)
assign(valve_body, 'steel', uv='box', smooth='side', axis=(0, 0, 1),
       box_scale=0.03)
valve_hex = cylinder_z('Valve hex nut', D['valve_hex_z0_m'], D['valve_hex_z1_m'],
                       D['valve_hex_radius_m'], sides=6)
assign(valve_hex, 'steel', uv='box', smooth='flat', box_scale=0.03)
valve_stem = revolve_solid('Valve stem', [
    [0.0, D['valve_stem_z0_m']],
    [D['valve_stem_radius_m'], D['valve_stem_z0_m']],
    [D['valve_stem_radius_m'], D['valve_stem_z1_m'] - 0.0020],
    [D['valve_stem_radius_m'] * 0.8, D['valve_stem_z1_m']],
    [0.0, D['valve_stem_z1_m']],
], 64)
assign(valve_stem, 'brass', uv='box', smooth='side', axis=(0, 0, 1),
       box_scale=0.03)

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
                      if 0.0850 < key < 0.6805), default=0.0)

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
directions = {'front': (1, 0, 0.035), 'back': (-1, 0, 0.035),
              'left': (0, -1, 0.035), 'right': (0, 1, 0.035),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
height_scale = D['total_height_m'] * 1.18
diameter_scale = D['body_diameter_m'] * 1.90
center_z = center.z

for name, direction in directions.items():
    scale = diameter_scale if name in ('top', 'bottom') else height_scale
    render_view(f'new-{name}', direction, center, scale, resolution=768, samples=32)

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
    render_view('new-source', (1.0, 0.0, 0.02), center, height_scale,
                resolution=512, samples=12)
else:
    details = (
        ('guard-window', (1.0, -0.45, 0.10), (0, 0, 0.8036 + Z_SHIFT), 0.085),
        ('guard-window-close', (1.0, -0.20, 0.05), (0, 0, 0.8036 + Z_SHIFT), 0.045),
        ('guard-top', (0.25, -0.45, 1.0), (0, 0, 0.828 + Z_SHIFT), 0.095),
        ('neck-collar', (0.9, -0.4, 0.10), (0, 0, 0.760 + Z_SHIFT), 0.070),
        ('shoulder', (0.8, -0.6, 0.20), (0, 0, 0.716 + Z_SHIFT), 0.130),
        ('ccs-logo', (0.90, -0.35, 0.05), (0, 0, 0.494 + Z_SHIFT), 0.210),
        ('body-surface', (0.95, -0.32, 0.08), (0.062, -0.012, 0.470 + Z_SHIFT), 0.120),
        ('bottom-foot', (0.8, -0.6, 0.05), (0, 0, 0.026 + Z_SHIFT), 0.100),
        ('bottom-underside', (0.35, -0.45, -1.0), (0, 0, 0.010 + Z_SHIFT), 0.100),
        ('valve-window', (1.0, -0.08, 0.03), (0, 0, 0.8036 + Z_SHIFT), 0.035),
        ('top-detail', (0.1, -0.2, 1.0), (0, 0, 0.840 + Z_SHIFT), 0.110),
    )
    for name, direction, target, scale in details:
        render_view(f'detail-{name}', direction, target, scale,
                    resolution=640, samples=24)

    isolated_groups = {
        'guard-only': ({'Valve guard'}, (1.0, -0.4, 0.10),
                       (0, 0, 0.812 + Z_SHIFT), 0.110),
        'guard-top-only': ({'Valve guard'}, (0.2, -0.3, 1.0),
                           (0, 0, 0.812 + Z_SHIFT), 0.085),
        'valve-only': ({'Valve body', 'Valve hex nut', 'Valve stem'},
                       (1.0, -0.25, 0.10), (0, 0, 0.797 + Z_SHIFT), 0.070),
        'body-only': ({'Cylinder body'}, (1.0, -0.35, 0.10),
                      (0, 0, 0.385 + Z_SHIFT), 0.930),
        'body-top-only': ({'Cylinder body'}, (0.0, 0.0, 1.0),
                          (0, 0, 0.700 + Z_SHIFT), 0.180),
        'body-bottom-only': ({'Cylinder body'}, (0.0, 0.0, -1.0),
                             (0, 0, 0.070 + Z_SHIFT), 0.180),
    }
    for label, (names, direction, target, scale) in isolated_groups.items():
        for obj in objects:
            obj.hide_render = obj.name not in names
        render_view(label, direction, target, scale, resolution=640, samples=20)
    for obj in objects:
        obj.hide_render = False

    # Material / lighting comparisons.
    render_view('new-material-close', (2.3, -2.0, 1.0),
                (0, 0, 0.500 + Z_SHIFT), 0.330,
                resolution=1024, samples=32)
    key = bpy.data.objects['large softbox']
    key.location = (-3.4, 2.8, 4.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    render_view('new-light-alternate', directions['front'], center, height_scale,
                resolution=768, samples=28)
    render_view('new-surface-alternate', (2.3, -2.0, 1.0),
                (0, 0, 0.500 + Z_SHIFT), 0.330,
                resolution=1024, samples=32)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

    # Dedicated reflection response: low ambient, one strong moved key.
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

    metal_names = {'Neck seal ring', 'Valve body', 'Valve hex nut', 'Valve stem'}
    for obj in objects:
        obj.hide_render = obj.name not in metal_names
    low_ambient()
    place_key((2.4, -2.2, 2.6))
    render_view('metal-key', (1.0, -0.15, 0.05),
                (0, 0, 0.797 + Z_SHIFT), 0.055,
                resolution=640, samples=28)
    place_key((-2.4, 2.2, 2.6))
    render_view('metal-alt', (1.0, -0.15, 0.05),
                (0, 0, 0.797 + Z_SHIFT), 0.055,
                resolution=640, samples=28)

    # Paint gloss response on the dielectric enamel (not counted as metal).
    # Use a weaker key than the metal test so the large painted area is not
    # blown out in the review board.
    key.data.energy = 420.0
    key.data.size = 1.2
    key.data.size_y = 1.2
    bpy.context.view_layer.update()
    for obj in objects:
        obj.hide_render = obj.name != 'Cylinder body'
    place_key((2.4, -2.2, 2.6))
    render_view('paint-key', (0.95, -0.32, 0.08),
                (0.062, -0.012, 0.470 + Z_SHIFT), 0.120,
                resolution=640, samples=28)
    place_key((-2.4, 2.2, 2.6))
    render_view('paint-alt', (0.95, -0.32, 0.08),
                (0.062, -0.012, 0.470 + Z_SHIFT), 0.120,
                resolution=640, samples=28)
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
                    resolution=640, samples=16)
    for obj in objects:
        obj.data.materials.clear()
        for material in original_materials[obj.name]:
            obj.data.materials.append(material)
    bpy.context.view_layer.update()

    # Turntable.
    for i in range(24):
        angle = TAU * i / 24
        render_view(f'turn-{i:02d}', (math.cos(angle), math.sin(angle), 0.14),
                    center, height_scale, resolution=512, samples=8)

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
    draw(scene, render_dir / 'new-source.png', resolution=512, samples=24)
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
        draw(scene, render_dir / f'old-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), center, oldspan * 0.42)
    draw(scene, render_dir / 'old-material-close.png', resolution=1024, samples=24)
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
    'view_note': ('Camera at +X is the source-photo front. The guard hand-hole is centred at '
                  'Y=+4.2 mm and the CCS print at U=0.4844 to follow the small rightward '
                  'offset visible in the photograph. The hand-hole is cut through the front '
                  'wall only; the photograph shows a dark interior behind the slot, so the far '
                  'wall stays solid. The old model is rotated/centred for comparison only; '
                  'its file is unchanged.'),
    'guard_boundary_edges_after_boolean': guard_boundary,
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
    'final_window_center_z_m': round(float(D['window_center_z_m'] + Z_SHIFT), 9),
    'dimensions': {key: D[key] for key in
                   ('total_height_m', 'body_radius_m', 'body_diameter_m',
                    'body_bottom_round_z1_m', 'body_cylinder_z1_m',
                    'shoulder_z1_m', 'neck_z0_m', 'neck_z1_m',
                    'guard_z0_m', 'guard_z1_m', 'guard_radius_m',
                    'window_center_z_m', 'window_half_width_y_m',
                    'window_half_height_z_m', 'window_center_y_m',
                    'bottom_contact_radius_m')},
    'draft': draft,
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
print(json.dumps({'status': 'rendered', 'parts': len(parts),
                  'render_device': render_device, 'draft': draft},
                 ensure_ascii=False), flush=True)
