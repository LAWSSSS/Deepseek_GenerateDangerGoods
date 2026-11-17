"""Blender worker for the review-only Botny spray-paint can (item 36).

The primary shape evidence is the single clean/rust HD photograph of an upright
grey-cap aerosol spray-paint can.  The supplied six-view sheet is an
uncalibrated visual reference of the same product family; its front panel is
used for silhouette/topology comparison and its back/side panels only for the
hidden half of the label.  It is never treated as a measured turntable.

The old TripoSR library mesh has a visibly elliptical cross-section, an axis
tilted about 11 degrees, a fused cap/body silhouette and a smeared label.  It
is kept unchanged.  This candidate is a parametric part rebuild with:

* a true straight cylindrical printed body;
* a hollow grey over-cap (real downward-open cavity) with a rounded top edge
  and a shallow central sprue dimple;
* a rolled silver base rim and a concave stamped bottom dish with a raised
  inner ring and an inferred batch/expiry stamp;
* a hidden valve cup/stem under the cap;
* separate flat-albedo PBR maps (label print, grey plastic, aluminium) whose
  photographed lighting has been flattened out, so highlights move with live
  lighting instead of being baked into the colour.

It never replaces the library mesh, stays manual-only and never enters
automatic placement or final compositing.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Quaternion, Matrix
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
config = request['config']
objects = []
material_by_key = {}
TAU = math.tau

try:
    scene = legacy.setup_scene('OPTIX', 512)
    render_device = 'OPTIX'
except Exception:
    scene = legacy.setup_scene('CPU', 512)
    render_device = 'CPU'
scene.render.threads_mode = 'AUTO'
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.view_settings.exposure = -0.6


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


def make_solid_material(key, rgb, roughness, metallic):
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    principled = material.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (*rgb, 1.0)
    principled.inputs['Roughness'].default_value = roughness
    principled.inputs['Metallic'].default_value = metallic
    material_by_key[key] = material
    return material


make_textured_material('body', 'body', 0.18)
make_textured_material('cap', 'cap', 0.22)
make_textured_material('base', 'base', 0.24)
profile = config['material_profile']
make_solid_material('valve', [v / 255 for v in profile['valve']['base_rgb_srgb']],
                    profile['valve']['roughness'], profile['valve']['metallic'])
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


def revolve_solid(name, profile, segments):
    verts, rings = [], []
    for r, z in profile:
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


def boolean_difference(target, cutter):
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new('Cap cavity subtraction', 'BOOLEAN')
    modifier.operation = 'DIFFERENCE'
    modifier.solver = 'EXACT'
    modifier.object = cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    clean_object(target, merge=1e-6)


def cylinder_cutter(name, radius, z0, z1, segments):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segments, radius=radius, depth=z1 - z0,
        end_fill_type='NGON', location=(0.0, 0.0, (z0 + z1) / 2.0))
    obj = bpy.context.object
    obj.name = name
    return obj


# ---------------------------------------------------------------------------
# UVs
# ---------------------------------------------------------------------------
def ensure_uv(obj, name='UVMap'):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name=name)
    return obj.data.uv_layers.active.data


def cylindrical_uv(obj, z0, z1, u_offset=0.25):
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
            u = (ca + delta) / TAU + u_offset
            v = (co.z - z0) / max(1e-9, z1 - z0)
            uv[loop_index].uv = (u, v)


def planar_uv_rect(obj, u0, u1, v0, v1, radius, flip_x=False, z_limit=None):
    """Planar X/Y map into a rectangle; optionally only faces below z_limit."""
    uv = ensure_uv(obj)
    mesh = obj.data
    for poly in mesh.polygons:
        if z_limit is not None and poly.center.z > z_limit:
            continue
        for loop_index in poly.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            fx = (-co.x if flip_x else co.x) / (2.0 * radius) + 0.5
            fy = co.y / (2.0 * radius) + 0.5
            uv[loop_index].uv = (u0 + (u1 - u0) * fx, v0 + (v1 - v0) * fy)


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


def assign(obj, key, *, uv='cylinder', z0=None, z1=None, smooth='all', axis=None):
    obj.data.materials.clear()
    obj.data.materials.append(material_by_key[key])
    if uv == 'cylinder':
        if z0 is None or z1 is None:
            lo = min(v.co.z for v in obj.data.vertices)
            hi = max(v.co.z for v in obj.data.vertices)
            z0, z1 = lo, hi
        cylindrical_uv(obj, z0, z1)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
D = config['dimensions']
SEG = config['radial_segments']
BODY_Z1 = D['body_z1_m']
LABEL_Z0, LABEL_Z1 = D['label_z0_m'], D['label_z1_m']
CAP_Z0, CAP_Z1 = D['cap_z0_m'], D['cap_z1_m']
BODY_R = D['body_radius_m']
CAP_R = D['cap_outer_radius_m']
BASE_R = D['base_outer_radius_m']
BASE_TOP = D['base_inner_top_z_m']

body = revolve_solid('Can body', config['profiles']['body'], SEG)
assign(body, 'body', uv='cylinder', z0=LABEL_Z0, z1=LABEL_Z1,
       smooth='side', axis=(0, 0, 1))

base = revolve_solid('Can base', config['profiles']['base'], SEG)
# One metallic material for the whole base, with a two-tile atlas: left half is
# the brushed side/rim, right half is the stamped concave bottom.  This keeps
# the object single-material so its watertight topology is easy to audit.
base.data.materials.append(material_by_key['base'])
ensure_uv(base)
cylindrical_uv(base, 0.0, BASE_TOP)
for poly in base.data.polygons:
    if poly.normal.z < -0.85 and poly.center.z < 0.0049:
        for loop_index in poly.loop_indices:
            co = base.data.vertices[base.data.loops[loop_index].vertex_index].co
            # The right-hand tile holds the stamped bottom.  The U flip makes
            # the batch code read correctly in a camera looking from -Z.
            u = 0.75 - co.x / (4.0 * BASE_R)
            v = 0.5 + co.y / (2.0 * BASE_R)
            base.data.uv_layers.active.data[loop_index].uv = (u, v)
        poly.use_smooth = False
    else:
        poly.use_smooth = abs(poly.normal.z) < 0.90
objects.append(base)

cap = revolve_solid('Cap shell', config['profiles']['cap'], SEG)
cavity = config['features']['cap_cavity_cutter']
cutter = cylinder_cutter('Cap cavity cutter', cavity['radius_m'],
                         cavity['z0_m'], cavity['z1_m'], cavity['segments'])
boolean_difference(cap, cutter)
cap_boundary = boundary_edges(cap)
cap.data.materials.append(material_by_key['cap'])
ensure_uv(cap)
cylindrical_uv(cap, CAP_Z0, CAP_Z1)
for poly in cap.data.polygons:
    if poly.normal.z > 0.45 and poly.center.z > CAP_Z1 - 0.006:
        for loop_index in poly.loop_indices:
            co = cap.data.vertices[cap.data.loops[loop_index].vertex_index].co
            # Right-hand tile holds the planar cap top.
            u = 0.75 + co.x / (4.0 * CAP_R)
            v = 0.5 + co.y / (2.0 * CAP_R)
            cap.data.uv_layers.active.data[loop_index].uv = (u, v)
        poly.use_smooth = True
    else:
        # Left-hand tile holds the cylindrical cap side.  The side tile is
        # addressed as u_atlas = 0.5 * u_side.
        for loop_index in poly.loop_indices:
            co = cap.data.vertices[cap.data.loops[loop_index].vertex_index].co
            angle = math.atan2(co.y, co.x)
            u_side = angle / TAU + 0.25
            v = (co.z - CAP_Z0) / max(1e-9, CAP_Z1 - CAP_Z0)
            cap.data.uv_layers.active.data[loop_index].uv = (0.5 * u_side, v)
        poly.use_smooth = abs(poly.normal.z) < 0.90
objects.append(cap)

valve = revolve_solid('Valve stem', config['profiles']['valve'], 64)
assign(valve, 'valve', uv='cylinder', z0=0.184, z1=0.1905, smooth='all')

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
    if 0.0063 < vertex.co.z < 0.1500:
        body_radii.append(math.hypot(vertex.co.x, vertex.co.y))
body_radii = np.asarray(body_radii, dtype=np.float64)
radial_deviation = float(np.max(np.abs(body_radii - BODY_R))) if len(body_radii) else 0.0
rings = {}
for vertex in body.data.vertices:
    if 0.0063 < vertex.co.z < 0.1500:
        key = round(vertex.co.z, 6)
        rings.setdefault(key, []).append(math.hypot(vertex.co.x, vertex.co.y))
ring_variation = max((max(values) - min(values) for values in rings.values()),
                     default=0.0)

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
background = scene.world.node_tree.nodes.get('Background')
background.inputs['Strength'].default_value = 0.55
background.inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
light_specs = (
    ('large softbox', (2.6, -3.0, 3.6), 250, 3.0, 3.0),
    ('fill softbox', (-3.2, 2.6, 2.2), 125, 3.0, 3.0),
    ('front strip', (3.2, 1.0, 1.3), 125, 1.2, 3.4),
    ('rim high', (-1.2, 3.2, 3.4), 85, 1.4, 2.2),
    ('low fill', (0, -2.6, -0.28), 45, 1.0, 1.0),
    ('bottom fill', (0.4, 0.2, -2.8), 35, 2.0, 2.0),
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
        # Explicit top/bottom orientation so the bottom stamp is readable and
        # repeated renders keep the same image axes.
        if direction.z > 0:
            right, up, back = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
        else:
            right, up, back = Vector((-1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, -1))
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = Matrix((right, up, back)).transposed().to_quaternion()
    return cam


def draw(scene, path, *, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_view(name, direction, target, scale, *, resolution=768, samples=40,
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
diameter_scale = D['body_diameter_m'] * 1.70
center_z = center.z

for name, direction in directions.items():
    scale = diameter_scale if name in ('top', 'bottom') else height_scale
    render_view(f'new-{name}', direction, center, scale,
                resolution=768, samples=40)

# Detail renders.
details = (
    ('cap-top', (0.35, -0.8, 1.0), (0, 0, 0.196), 0.080, 720, 36),
    ('cap-side', (0.6, -1.0, 0.15), (0, 0, 0.176), 0.090, 720, 36),
    ('cap-seam', (0.4, -1.0, 0.25), (0, -0.030, 0.1508), 0.075, 720, 36),
    ('cap-dimples', (0.2, -0.5, 1.0), (0, 0.0, 0.1995), 0.035, 720, 36),
    ('front-label', (1.0, -0.06, 0.04), (0.0322, 0.0, 0.090), 0.180, 720, 40),
    ('back-label', (-1.0, 0.06, 0.04), (-0.0322, 0.0, 0.090), 0.190, 720, 40),
    ('body', (0.35, -1.0, 0.12), (0, 0, 0.080), 0.290, 720, 36),
    ('body-surface', (1.0, -0.35, 0.12), (0.0322, -0.008, 0.080), 0.080, 720, 40),
    ('bottom', (0.2, -0.3, -1.0), (0, 0, 0.004), 0.085, 720, 40),
    ('bottom-dish', (0.0, 0.0, -1.0), (0, 0, 0.004), 0.068, 720, 40),
    ('bottom-stamp', (0.0, 0.0, -1.0), (0, 0, 0.0045), 0.045, 720, 40),
    ('metal-rim', (0.6, -0.8, -0.2), (0.025, -0.012, 0.003), 0.030, 720, 40),
    ('valve', (0.4, -0.6, 0.8), (0, 0, 0.187), 0.035, 720, 36),
    ('surface', (1.0, -0.15, 0.08), (0.0322, -0.005, 0.120), 0.065, 720, 40),
)
for name, direction, target, scale, resolution, samples in details:
    render_view(f'detail-{name}', direction, target, scale,
                resolution=resolution, samples=samples)

# Isolated structure renders.
isolated_groups = {
    'body-only': ['Can body'],
    'cap-only': ['Cap shell'],
    'base-only': ['Can base'],
    'valve-only': ['Valve stem'],
}
for label, names in isolated_groups.items():
    for obj in objects:
        obj.hide_render = obj.name not in names
    render_view(label, (0.35, -1.0, 0.20), (0, 0, 0.080), 0.30,
                resolution=640, samples=28)
    if label == 'cap-only':
        render_view('cap-only-top', (0.0, 0.0, 1.0), (0, 0, 0.176), 0.080,
                    resolution=640, samples=28)
        render_view('cap-only-bottom', (0.0, 0.0, -1.0), (0, 0, 0.176), 0.080,
                    resolution=640, samples=28)
        render_view('cap-only-bottom-3q', (0.55, -0.70, -0.85), (0, 0, 0.172),
                    0.085, resolution=640, samples=28)
    if label == 'base-only':
        render_view('base-only-bottom', (0.0, 0.0, -1.0), (0, 0, 0.004), 0.072,
                    resolution=640, samples=28)
for obj in objects:
    obj.hide_render = False

# Material / lighting comparisons.
render_view('new-material-close', (2.3, -2.0, 1.0), (0, 0, 0.090), 0.105,
            resolution=1024, samples=40)
key = bpy.data.objects['large softbox']
key.location = (-3.4, 2.8, 4.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
render_view('new-light-alternate', directions['front'], center, height_scale,
            resolution=768, samples=36)
render_view('new-surface-alternate', (2.3, -2.0, 1.0), (0, 0, 0.090), 0.105,
            resolution=1024, samples=40)
key.location = (2.6, -3.0, 3.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Metal base under two key-light directions (base only).
for obj in objects:
    obj.hide_render = obj.name != 'Can base'
render_view('metal-base-key', (0.4, -0.5, -1.0), (0, 0, 0.004), 0.078,
            resolution=768, samples=40)
key.location = (-2.6, 2.2, -3.2)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
render_view('metal-base-alt', (0.4, -0.5, -1.0), (0, 0, 0.004), 0.078,
            resolution=768, samples=40)
key.location = (2.6, -3.0, 3.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
render_view('metal-rim-key', (0.6, -0.8, -0.15), (0.025, -0.012, 0.003), 0.040,
            resolution=768, samples=40)
key.location = (-2.6, 2.2, 3.2)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
render_view('metal-rim-alt', (0.6, -0.8, -0.15), (0.025, -0.012, 0.003), 0.040,
            resolution=768, samples=40)
key.location = (2.6, -3.0, 3.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
for obj in objects:
    obj.hide_render = False

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
                resolution=640, samples=24)
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
draw(scene, render_dir / 'new-source.png', resolution=512, samples=36)
bpy.data.objects.remove(cam, do_unlink=True)

# Old model at the same neutral lighting.  The old vertex-colour material is
# much brighter than the candidate's flattened albedo; a one-stop lower
# exposure makes its smeared label and lumpy silhouette visible.  Lighting
# directions and the background stay identical.
scene.view_settings.exposure = -1.6
for obj in objects:
    obj.hide_render = True
old.hide_render = False
for name, direction in directions.items():
    old_scale = old_span_front * 1.18 if name in ('front', 'back') else oldspan * 1.18
    if name in ('top', 'bottom'):
        old_scale = max(oldhigh.x - oldlow.x, oldhigh.y - oldlow.y) * 1.70
    cam = camera(scene, direction, oldcenter, old_scale)
    draw(scene, render_dir / f'old-{name}.png', resolution=768, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
cam = camera(scene, (2.3, -2.0, 1.0), oldcenter, oldspan * 0.42)
draw(scene, render_dir / 'old-material-close.png', resolution=1024, samples=32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)
scene.view_settings.exposure = -0.6
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
    'geometry_basis': 'source-photo proportions with an uncalibrated visual reference for the hidden label half, cap cavity and bottom topology',
    'render_samples': {'views': 40, 'details': 36, 'turntable': 10},
    'new_render_exposure_ev': -0.6,
    'old_render_exposure_ev': -1.6,
    'old_render_relative_stops': -1.0,
    'view_directions': {k: list(v) for k, v in directions.items()},
    'view_note': ('Camera at +X is the source-photo front; the label centre is +X. '
                  'The old model is centred for comparison only; its file is unchanged.'),
    'cap_boundary_edges_after_boolean': cap_boundary,
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
    'dimensions': {key: D[key] for key in
                   ('total_height_m', 'body_radius_m', 'body_diameter_m',
                    'body_z0_m', 'body_z1_m', 'label_z0_m', 'label_z1_m',
                    'cap_outer_radius_m', 'cap_inner_radius_m', 'cap_z0_m',
                    'cap_z1_m', 'base_outer_radius_m', 'base_bottom_center_z_m',
                    'valve_top_z_m')},
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
print(json.dumps({'status': 'rendered', 'parts': len(parts),
                  'render_device': render_device}, ensure_ascii=False), flush=True)
