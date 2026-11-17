"""Blender worker for the review-only red aerosol-can candidate (item 16).

The primary shape evidence is the single clean/rust photograph of one used red
slim aerosol can.  The user six-view sheet is an uncalibrated generated visual
reference: it supplies the cap scoop/recess, rim, bottom-dish topology and the
layout of the unseen back label, but it is never treated as a measured
turntable.  The candidate is a part-based rebuild with a genuine cylindrical
body, a closed cap solid with a subtracted oval scoop, a raised elliptical rim,
a low actuator with a real nozzle through-hole, a dark cap/body seam and a
stamped aluminium base.  It never replaces the library mesh, stays manual-only
and never enters automatic placement.
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
objects = []
material_by_key = {}
TAU = math.tau


# The scene reset must happen before materials and geometry are created;
# setup_scene() loads factory settings and would otherwise remove them.
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


def make_solid_material(key, rgb, roughness, metallic):
    material = bpy.data.materials.new(key)
    material.use_nodes = True
    principled = material.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (*rgb, 1.0)
    principled.inputs['Roughness'].default_value = roughness
    principled.inputs['Metallic'].default_value = metallic
    material_by_key[key] = material
    return material


make_textured_material('body', 'body', 0.20)
make_textured_material('cap', 'cap', 0.30)
make_textured_material('base', 'base', 0.28)
profile = config['material_profile']
make_solid_material('nozzle', profile['nozzle']['base_rgb'],
                    profile['nozzle']['roughness'], profile['nozzle']['metallic'])
make_solid_material('seam', profile['seam']['base_rgb'],
                    profile['seam']['roughness'], profile['seam']['metallic'])
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


def elliptic_torus(name, center_z, major_x, major_y, minor,
                   major_segments=160, minor_segments=12):
    verts, faces = [], []
    for i in range(major_segments):
        phi = TAU * i / major_segments
        cx, cy = major_x * math.cos(phi), major_y * math.sin(phi)
        # Ellipse normal in the horizontal plane.
        nx, ny = math.cos(phi) / major_x, math.sin(phi) / major_y
        length = math.hypot(nx, ny)
        nx, ny = nx / length, ny / length
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


def torus_z(name, center_z, major, minor, major_segments=192, minor_segments=8):
    return elliptic_torus(name, center_z, major, major, minor,
                          major_segments, minor_segments)


def nozzle_tube(name, x0, x1, y, z, outer, hole, segments):
    verts, faces = [], []
    for x in (x0, x1):
        for r in (outer, hole):
            for s in range(segments):
                a = TAU * s / segments
                verts.append((x, y + r * math.cos(a), z + r * math.sin(a)))
    # vertex block order: x0 outer, x0 hole, x1 outer, x1 hole
    o0, h0, o1, h1 = 0, segments, 2 * segments, 3 * segments
    for s in range(segments):
        ns = (s + 1) % segments
        faces.append((o0 + s, o0 + ns, o1 + ns, o1 + s))
        faces.append((h0 + ns, h0 + s, h1 + s, h1 + ns))
        faces.append((o0 + ns, o0 + s, h0 + s, h0 + ns))
        faces.append((o1 + s, o1 + ns, h1 + ns, h1 + s))
    return mesh_object(name, verts, faces)


def scaled_uv_sphere(name, center, radii, segments=64, rings=32):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings,
                                         radius=1.0, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = radii
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def boolean_difference(target, cutter):
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new('Oval scoop subtraction', 'BOOLEAN')
    modifier.operation = 'DIFFERENCE'
    modifier.solver = 'EXACT'
    modifier.object = cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    clean_object(target, merge=1e-6)


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


def box_uv(obj, scale=0.05):
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


def assign(obj, key, *, uv='cylinder', z0=None, z1=None, smooth='all', axis=None,
           box_scale=0.05):
    obj.data.materials.clear()
    obj.data.materials.append(material_by_key[key])
    if uv == 'cylinder':
        if z0 is None or z1 is None:
            lo = min(v.co.z for v in obj.data.vertices)
            hi = max(v.co.z for v in obj.data.vertices)
            z0, z1 = lo, hi
        cylindrical_uv(obj, z0, z1)
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
BODY_Z0, BODY_Z1 = D['body_z0_m'], D['body_z1_m']
CAP_Z0, CAP_Z1 = D['cap_z0_m'], D['cap_z1_m']

body = revolve_solid('Can body', config['profiles']['body'], SEG)
assign(body, 'body', uv='cylinder', z0=BODY_Z0, z1=BODY_Z1,
       smooth='side', axis=(0, 0, 1))

base = revolve_solid('Can base', config['profiles']['base'], SEG)
assign(base, 'base', uv='cylinder', z0=0.0, z1=D['base_top_z_m'],
       smooth='side', axis=(0, 0, 1))

cap = revolve_solid('Cap shell', config['profiles']['cap'], SEG)
recess = config['features']['recess_cutter']
cutter = scaled_uv_sphere('Recess cutter', recess['center'], recess['radii'],
                          recess['sphere_segments'], recess['sphere_rings'])
boolean_difference(cap, cutter)
cap_boundary = boundary_edges(cap)
assign(cap, 'cap', uv='cylinder', z0=CAP_Z0, z1=CAP_Z1, smooth='all')

rim_cfg = config['features']['recess_rim']
rim = elliptic_torus('Cap recess rim', rim_cfg['center_z'],
                     rim_cfg['major_x'], rim_cfg['major_y'], rim_cfg['minor'],
                     rim_cfg['major_segments'], rim_cfg['minor_segments'])
assign(rim, 'cap', uv='cylinder', z0=rim_cfg['center_z'] - rim_cfg['minor'],
       z1=rim_cfg['center_z'] + rim_cfg['minor'], smooth='all')

act_cfg = config['features']['actuator']
actuator = scaled_uv_sphere('Actuator button', act_cfg['center'], act_cfg['radii'],
                            segments=64, rings=32)
assign(actuator, 'cap', uv='box', smooth='all', box_scale=0.02)

noz_cfg = config['features']['nozzle']
nozzle = nozzle_tube('Nozzle insert', noz_cfg['x0'], noz_cfg['x1'],
                     noz_cfg['y'], noz_cfg['z'], noz_cfg['outer_radius'],
                     noz_cfg['hole_radius'], noz_cfg['segments'])
assign(nozzle, 'nozzle', uv='box', smooth='all', box_scale=0.006)

seam_cfg = config['features']['cap_seam_ring']
seam = torus_z('Cap seam ring', seam_cfg['z'], seam_cfg['major'],
               seam_cfg['minor'], seam_cfg['major_segments'],
               seam_cfg['minor_segments'])
assign(seam, 'seam', uv='box', smooth='all', box_scale=0.02)

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
    body_radii.append(math.hypot(vertex.co.x, vertex.co.y))
body_radii = np.asarray(body_radii, dtype=np.float64)
radial_deviation = float(np.max(np.abs(body_radii - D['body_radius_m'])))
# Radial variation within horizontal rings (excluding the flat caps).
rings = {}
for vertex in body.data.vertices:
    key = round(vertex.co.z, 6)
    rings.setdefault(key, []).append(math.hypot(vertex.co.x, vertex.co.y))
ring_variation = max((max(values) - min(values) for key, values in rings.items()
                      if 0.02 < key < 0.23), default=0.0)

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
background.inputs['Strength'].default_value = 0.52
background.inputs['Color'].default_value = (1, 1, 1, 1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)
light_specs = (
    ('large softbox', (2.6, -3.0, 3.6), 240, 3.0, 3.0),
    ('fill softbox', (-3.2, 2.6, 2.2), 120, 3.0, 3.0),
    ('front strip', (3.2, 1.0, 1.3), 120, 1.2, 3.4),
    ('rim high', (-1.2, 3.2, 3.4), 80, 1.4, 2.2),
    ('low fill', (0, -2.6, -0.28), 45, 1.0, 1.0),
    ('bottom fill', (0.4, 0.2, -2.8), 30, 2.0, 2.0),
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


def render_view(name, direction, target, scale, *, resolution=768, samples=36,
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
    ('cap-top', (0.35, -0.8, 1.0), (0, 0, 0.2955), 0.075, 768, 40),
    ('cap-recess', (0.5, -0.6, 1.0), (0, 0.002, 0.2952), 0.052, 768, 40),
    ('recess-rim', (0.2, -1.0, 0.75), (0, 0.002, 0.2978), 0.045, 768, 40),
    ('nozzle', (0.55, -0.35, 1.0), (0.0065, 0.0025, 0.2952), 0.030, 768, 40),
    ('cap-seam', (0.4, -1.0, 0.25), (0, -0.028, 0.2474), 0.070, 768, 36),
    ('front-label', (1.0, -0.06, 0.04), (0.0296, 0.0, 0.170), 0.180, 768, 40),
    ('back-label', (-1.0, 0.06, 0.04), (-0.0296, 0.0, 0.155), 0.200, 768, 40),
    ('body', (0.35, -1.0, 0.12), (0, 0, 0.140), 0.290, 768, 36),
    ('body-surface', (1.0, -0.35, 0.12), (0.0296, -0.008, 0.150), 0.085, 768, 40),
    ('bottom', (0.2, -0.3, -1.0), (0, 0, 0.004), 0.080, 768, 40),
    ('bottom-dish', (0.0, 0.0, -1.0), (0, 0, 0.004), 0.062, 768, 40),
    ('metal-rim', (0.6, -0.8, -0.2), (0.024, -0.012, 0.006), 0.032, 768, 40),
    ('surface', (1.0, -0.15, 0.08), (0.0296, -0.005, 0.205), 0.060, 768, 40),
)
for name, direction, target, scale, resolution, samples in details:
    render_view(f'detail-{name}', direction, target, scale,
                resolution=resolution, samples=samples)

# Isolated structure renders.
isolated_groups = {
    'body-only': ['Can body'],
    'cap-only': ['Cap shell', 'Cap recess rim', 'Actuator button', 'Nozzle insert',
                 'Cap seam ring'],
    'base-only': ['Can base'],
}
for label, names in isolated_groups.items():
    for obj in objects:
        obj.hide_render = obj.name not in names
    render_view(label, (0.35, -1.0, 0.20), (0, 0, 0.145), 0.32,
                resolution=640, samples=28)
    if label == 'cap-only':
        render_view('cap-only-top', (0.0, 0.0, 1.0), (0, 0, 0.273), 0.080,
                    resolution=640, samples=28)
    if label == 'base-only':
        render_view('base-only-bottom', (0.0, 0.0, -1.0), (0, 0, 0.004), 0.070,
                    resolution=640, samples=28)
for obj in objects:
    obj.hide_render = False

# Material / lighting comparisons.
render_view('new-material-close', (2.3, -2.0, 1.0), (0, 0, 0.170), 0.115,
            resolution=1024, samples=40)
key = bpy.data.objects['large softbox']
key.location = (-3.4, 2.8, 4.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
render_view('new-light-alternate', directions['front'], center, height_scale,
            resolution=768, samples=36)
render_view('new-surface-alternate', (2.3, -2.0, 1.0), (0, 0, 0.170), 0.115,
            resolution=1024, samples=40)
key.location = (2.6, -3.0, 3.6)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Metal base under two key-light directions (base only).
for obj in objects:
    obj.hide_render = obj.name != 'Can base'
render_view('metal-base-key', (0.4, -0.5, -1.0), (0, 0, 0.004), 0.075,
            resolution=768, samples=40)
key.location = (-2.6, 2.2, -3.2)
key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
render_view('metal-base-alt', (0.4, -0.5, -1.0), (0, 0, 0.004), 0.075,
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

# Old model at the same neutral lighting.
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
    'geometry_basis': 'source-photo proportions with an uncalibrated visual reference for the back and cap/bottom topology',
    'render_samples': {'views': 40, 'details': 40, 'turntable': 10},
    'view_directions': {k: list(v) for k, v in directions.items()},
    'view_note': ('Camera at +X is the source-photo front; the label centre is +X. '
                  'The old model is rotated/centred for comparison only; its file is unchanged.'),
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
                    'body_z0_m', 'body_z1_m', 'cap_z0_m', 'cap_z1_m',
                    'base_top_z_m')},
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
print(json.dumps({'status': 'rendered', 'parts': len(parts),
                  'render_device': render_device}, ensure_ascii=False), flush=True)
