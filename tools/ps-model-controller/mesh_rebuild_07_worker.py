"""Blender worker for the review-only vertical stainless mixing-tank candidate (item 07).

Shape basis: the original rust photograph for proportions and visible
structure; the user-supplied six-view sheet is an uncalibrated *category*
reference of a similar vertical tank (its centre manway and top nozzles are
not present on the photographed unit).  Hidden/back-side details and the
repetition of off-frame features are marked as inferred in the record.

The candidate never replaces the library mesh, is manual-only and is never
used by the automatic placement pipeline.
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
TAU = math.tau


# ---------------------------------------------------------------------------
# Materials and UVs
# ---------------------------------------------------------------------------
def atlas_materials():
    names = {0: '07 bright stainless', 1: '07 brushed stainless',
             2: '07 weathered stainless', 3: '07 stained steel'}
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
        normal_map.inputs['Strength'].default_value = {0: .16, 1: .20, 2: .26, 3: .30}[tile]
        links.new(normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
        material_for_tile[tile] = material

    paint = bpy.data.materials.new('07 blue motor paint')
    paint.use_nodes = True
    nodes = paint.node_tree.nodes
    links = paint.node_tree.links
    principled = nodes.get('Principled BSDF')

    def paint_texture(key, noncolor=False):
        node = nodes.new('ShaderNodeTexImage')
        node.image = bpy.data.images.load(request['textures'][key], check_existing=True)
        node.image.pack()
        if noncolor:
            node.image.colorspace_settings.name = 'Non-Color'
        node.interpolation = 'Linear'
        return node

    color = paint_texture('paint_base')
    metal = paint_texture('paint_mr', True)
    normal = paint_texture('paint_normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(metal.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = .18
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    special_materials['paint'] = paint

    dark = bpy.data.materials.new('07 dark bore')
    dark.use_nodes = True
    principled = dark.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.008, .009, .009, 1)
    principled.inputs['Metallic'].default_value = .30
    principled.inputs['Roughness'].default_value = .58
    special_materials['dark'] = dark

    gasket = bpy.data.materials.new('07 black gasket')
    gasket.use_nodes = True
    principled = gasket.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.012, .013, .012, 1)
    principled.inputs['Metallic'].default_value = .00
    principled.inputs['Roughness'].default_value = .78
    special_materials['gasket'] = gasket

    glass = bpy.data.materials.new('07 sight glass')
    glass.use_nodes = True
    principled = glass.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.010, .045, .16, 1)
    principled.inputs['Metallic'].default_value = .45
    principled.inputs['Roughness'].default_value = .07
    special_materials['glass'] = glass


def box_uv(obj, scale=0.80):
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


def remap_uv(obj, tile):
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    col, top_row = tile % 2, tile // 2
    for loop in uv:
        u, v = loop.uv
        loop.uv = (col * .5 + .012 + (u % 1.0) * .476,
                   (1 - top_row) * .5 + .012 + (v % 1.0) * .476)


def set_smooth(obj, mode='all', axis=None):
    if mode == 'flat':
        for poly in obj.data.polygons:
            poly.use_smooth = False
        return
    if mode == 'side' and axis is not None:
        axis = Vector(axis).normalized()
        for poly in obj.data.polygons:
            poly.use_smooth = abs(poly.normal.dot(axis)) < .80
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True


def assign_tile(obj, tile, *, boxed=True, smooth='all', axis=None):
    obj.data.materials.clear()
    obj.data.materials.append(material_for_tile[tile])
    if boxed:
        box_uv(obj)
    remap_uv(obj, tile)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def assign_special(obj, key, *, boxed=True, smooth='all', axis=None):
    obj.data.materials.clear()
    obj.data.materials.append(special_materials[key])
    if boxed:
        box_uv(obj, .45 if key == 'paint' else .80)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def hollow_pipe(name, start, end, r_out, r_in, tile, segments=64):
    """One steel thick-walled pipe plus a separate dark inner sleeve."""
    pipe, _ = tube(name, start, end, r_out, r_in, segments)
    assign_tile(pipe, tile, boxed=True, smooth='all')
    sleeve, _ = tube(name + ' sleeve', start, end, r_in - .0002, r_in - .0014,
                     segments)
    assign_special(sleeve, 'dark', boxed=True, smooth='all')
    return pipe, sleeve


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
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


def apply_bevel(obj, width, segments=2):
    if not width:
        return obj
    mod = obj.modifiers.new('Manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return obj


def revolve_solid(name, profile, segments):
    """Revolve a pole-to-pole (r,z) profile about Z into a closed solid."""
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


def revolve_ring(name, profile, segments):
    """Revolve a closed (r,z) profile about Z into a closed ring solid."""
    verts, rings = [], []
    for r, z in profile:
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            ids.append(len(verts))
            verts.append((r * math.cos(a), r * math.sin(a), z))
        rings.append(ids)
    n = len(rings)
    faces = []
    for k in range(n):
        j = (k + 1) % n
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((rings[k][s], rings[k][s1], rings[j][s1], rings[j][s]))
    return mesh_object(name, verts, faces)


def tube(name, start, end, r_out, r_in, segments=64):
    """Closed thick-walled pipe; returns the object and inner-wall face ids."""
    start, end = Vector(start), Vector(end)
    axis = (end - start).normalized()
    u = axis.orthogonal().normalized()
    v = axis.cross(u).normalized()
    verts = []
    rings = []
    for center, radius in ((start, r_out), (end, r_out), (start, r_in), (end, r_in)):
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            p = center + u * (radius * math.cos(a)) + v * (radius * math.sin(a))
            ids.append(len(verts))
            verts.append(tuple(p))
        rings.append(ids)
    os_, oe, ins, ie = rings
    faces, inner_faces = [], []
    for s in range(segments):
        s1 = (s + 1) % segments
        faces.append((os_[s], os_[s1], oe[s1], oe[s]))
        inner_faces.append(len(faces))
        faces.append((ins[s], ie[s], ie[s1], ins[s1]))
        faces.append((os_[s], ins[s], ins[s1], os_[s1]))
        faces.append((oe[s], oe[s1], ie[s1], ie[s]))
    return mesh_object(name, verts, faces), inner_faces


def along(name, start, end, radius, segments=32):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    return obj


def box_at(name, center, half):
    x, y, z = center
    hx, hy, hz = half
    verts = []
    for dz in (-hz, hz):
        for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            verts.append((x + dx, y + dy, z + dz))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def torus(name, center, normal, major, minor, major_segments=32, minor_segments=8):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                     major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=center)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = Vector(normal).to_track_quat('Z', 'Y').to_euler()
    return obj


def bolted_flange(name, start, end, r_out, r_in, segments=64):
    obj, _ = tube(name, start, end, r_out, r_in, segments)
    return obj


def folded_leg(name, azimuth_deg, spec):
    az = math.radians(azimuth_deg)
    radial = Vector((math.cos(az), math.sin(az), 0))
    tangent = Vector((-math.sin(az), math.cos(az), 0))
    z_top, z_bottom = spec['leg_top_z'], spec['leg_bottom_z']
    w_top, w_bottom = spec['leg_top_half_width'], spec['leg_bottom_half_width']
    rb_top, rb_bottom = spec['leg_top_base_radius'], spec['leg_bottom_base_radius']
    fold, t = spec['leg_fold'], spec['leg_thickness']
    levels = (z_top, (z_top + z_bottom) / 2, z_bottom)
    verts, rings = [], []
    for z in levels:
        f = (z - z_top) / (z_bottom - z_top)
        w = w_top + (w_bottom - w_top) * f
        # Creased front, flat back: a folded sheet-metal blade.  A mirrored
        # back fold puts two large faces 5 mm apart and the BVH audit reports
        # them as overlapping, so the hidden back is left flat.
        offsets = [(-w, t / 2), (0.0, fold + t / 2), (w, t / 2),
                   (w, -t / 2), (-w, -t / 2)]
        ring = []
        for u, dro in offsets:
            # The top edge follows the shell curvature so the wide blade does
            # not float off the vessel; it straightens towards the foot.
            wrap = math.sqrt(max(0.0, (rb_top + .002) ** 2 - u * u))
            rb = (1 - f) * wrap + f * rb_bottom
            p = radial * (rb + dro) + tangent * u
            ring.append(len(verts))
            verts.append((p.x, p.y, z))
        rings.append(ring)
    faces = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        for s in range(5):
            s1 = (s + 1) % 5
            faces.append((a[s], a[s1], b[s1], b[s]))
    top, bottom = rings[0], rings[-1]
    faces += [(top[0], top[1], top[4]),
              (top[1], top[2], top[3], top[4]),
              (bottom[4], bottom[1], bottom[0]),
              (bottom[4], bottom[3], bottom[2], bottom[1])]
    return mesh_object(name, verts, faces)


def prism_yz(name, profile, x0, x1):
    """Extrude a closed (y,z) profile along X into a closed prism."""
    n = len(profile)
    verts = []
    for x in (x0, x1):
        for y, z in profile:
            verts.append((x, y, z))
    faces = [tuple(range(n)), tuple(range(n, 2 * n))[::-1]]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return mesh_object(name, verts, faces)


def rounded_path(points, radius, segments):
    pts = [Vector(p) for p in points]
    if len(pts) < 3:
        return pts
    result = [pts[0]]
    for i in range(1, len(pts) - 1):
        p, a, b = pts[i], pts[i - 1], pts[i + 1]
        da = (a - p)
        db = (b - p)
        ra = min(radius, da.length * .45)
        rb = min(radius, db.length * .45)
        pa = p + da.normalized() * ra
        pb = p + db.normalized() * rb
        result.append(pa)
        for s in range(1, segments):
            u = s / segments
            q = (1 - u) ** 2 * pa + 2 * (1 - u) * u * p + u * u * pb
            result.append(q)
        result.append(pb)
    result.append(pts[-1])
    return result


def swept_tube(name, path, r_out, r_in, segments=48):
    pts = [Vector(p) for p in path]
    n = len(pts)
    tangents = []
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        tangents.append(t.normalized())
    frames = []
    u = tangents[0].orthogonal().normalized()
    for i, t in enumerate(tangents):
        if i:
            q = tangents[i - 1].rotation_difference(t)
            u = (q @ u).normalized()
        v = t.cross(u).normalized()
        frames.append((u.copy(), v.copy()))
    verts = []
    rings = []
    for center, radius in ((None, r_out), (None, r_in)):
        ids = []
        for i, p in enumerate(pts):
            uu, vv = frames[i]
            for s in range(segments):
                a = TAU * s / segments
                q = p + uu * (radius * math.cos(a)) + vv * (radius * math.sin(a))
                ids.append(len(verts))
                verts.append(tuple(q))
        rings.append(ids)
    outer, inner = rings
    faces = []
    for i in range(n - 1):
        a0 = outer[i * segments:(i + 1) * segments]
        a1 = outer[(i + 1) * segments:(i + 2) * segments]
        b0 = inner[i * segments:(i + 1) * segments]
        b1 = inner[(i + 1) * segments:(i + 2) * segments]
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((a0[s], a0[s1], a1[s1], a1[s]))
            faces.append((b0[s], b1[s], b1[s1], b0[s1]))
    a0 = outer[:segments]
    b0 = inner[:segments]
    a1 = outer[(n - 1) * segments:n * segments]
    b1 = inner[(n - 1) * segments:n * segments]
    for s in range(segments):
        s1 = (s + 1) % segments
        faces.append((a0[s], b0[s], b0[s1], a0[s1]))
        faces.append((a1[s], a1[s1], b1[s1], b1[s]))
    return mesh_object(name, verts, faces)


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build():
    R = config['radius']
    z_top = config['shell_top_z']
    h = config['head_depth']
    z_bottom = config['shell_bottom_z']
    z_apex = z_top + h
    seg = config['radial_segments']

    # Main pressure body: shallow dished top, cylindrical shell, shallow upper
    # cone.  It is a closed solid; bores are separate thick-walled parts.
    profile = [(0.0, z_apex)]
    for i in range(1, 33):
        theta = (math.pi / 2) * i / 32
        profile.append((R * math.sin(theta), z_top + h * math.cos(theta)))
    for i in range(1, 33):
        profile.append((R, z_top + (z_bottom - z_top) * i / 32))
    for i in range(1, 25):
        f = i / 24
        profile.append((R + (config['cone_flange_radius'] - R) * f,
                        z_bottom + (config['cone_flange_top_z'] - z_bottom) * f))
    profile.append((0.0, config['cone_flange_top_z']))
    body = revolve_solid('Tank body', profile, seg)
    assign_tile(body, 0, boxed=True, smooth='all')

    # Three circumferential weld bands: top tangent, mid shell, bottom tangent.
    for name, z, half, raise_ in (('Top head seam band', z_top, .006, .0035),
                                  ('Mid shell seam band', config['shell_mid_seam_z'], .006, .0025),
                                  ('Bottom seam band', z_bottom, .007, .0045)):
        ring = revolve_ring(name,
                            [(R - .002, z - half), (R + raise_, z - half + .0015),
                             (R + raise_, z + half - .0015), (R - .002, z + half)], seg)
        assign_tile(ring, 1, boxed=True, smooth='all')

    # Bolted lower cone and drain.
    ring = revolve_ring('Cone flange ring',
                        [(config['cone_flange_inner'], config['cone_flange_top_z']),
                         (config['cone_flange_outer'], config['cone_flange_top_z']),
                         (config['cone_flange_outer'], config['cone_flange_bottom_z']),
                         (config['cone_flange_inner'], config['cone_flange_bottom_z'])], seg)
    apply_bevel(ring, .0012, 2)
    assign_tile(ring, 1, boxed=True, smooth='all')
    for i in range(config['cone_flange_bolts']):
        a = TAU * i / config['cone_flange_bolts']
        x = config['cone_flange_bolt_radius'] * math.cos(a)
        y = config['cone_flange_bolt_radius'] * math.sin(a)
        bolt = along(f'Cone flange bolt {i + 1:02d}',
                     (x, y, config['cone_flange_bottom_z'] - .004),
                     (x, y, config['cone_flange_top_z'] + .003),
                     config['cone_flange_bolt_head_radius'], 6)
        assign_tile(bolt, 3, boxed=True, smooth='flat')
    for i, az in enumerate(config['cone_clamp_azimuths_deg'], start=1):
        a = math.radians(az)
        cx, cy = .052 * math.cos(a), .052 * math.sin(a)
        lug = box_at(f'Cone clamp lug {i}', (cx, cy, -.286), (.009, .009, .030))
        apply_bevel(lug, .0010, 2)
        assign_tile(lug, 1, boxed=True, smooth='flat')
        pin = along(f'Cone clamp pin {i}', (cx, cy - .018, -.296),
                    (cx, cy + .018, -.296), .0045, 16)
        assign_tile(pin, 3, boxed=True, smooth='side', axis=(0, 1, 0))
    lower = revolve_solid('Lower cone',
                          [(0.0, config['lower_cone_top_z']),
                           (config['lower_cone_top_radius'], config['lower_cone_top_z']),
                           (0.0, config['lower_cone_tip_z'])], 96)
    assign_tile(lower, 2, boxed=True, smooth='all')
    hollow_pipe('Drain pipe', (0, 0, config['drain_z_top']),
                (0, 0, config['drain_z_bottom']),
                config['drain_outer'], config['drain_bore'], 1, segments=40)
    lip = torus('Drain lip', (0, 0, config['drain_z_bottom']), (0, 0, 1),
                config['drain_outer'] + .0015, .0035, 36, 8)
    assign_tile(lip, 1, boxed=True, smooth='all')

    # Three folded-plate legs with leveling feet.
    for i, az in enumerate(config['leg_azimuths_deg'], start=1):
        leg = folded_leg(f'Leg {i}', az, config)
        assign_tile(leg, 1, boxed=True, smooth='flat')
        a = math.radians(az)
        radial = Vector((math.cos(a), math.sin(a), 0))
        foot_r = config['leg_bottom_base_radius'] + config['leg_fold']
        cx, cy = radial.x * foot_r, radial.y * foot_r
        stem = along(f'Leg {i} foot stem', (cx, cy, config['foot_stem_top_z']),
                     (cx, cy, config['foot_stem_bottom_z']),
                     config['foot_stem_radius'], 24)
        assign_tile(stem, 1, boxed=True, smooth='side', axis=(0, 0, 1))
        collar = along(f'Leg {i} foot collar', (cx, cy, config['foot_collar_bottom_z']),
                       (cx, cy, config['foot_collar_top_z']),
                       config['foot_collar_radius'], 24)
        assign_tile(collar, 1, boxed=True, smooth='side', axis=(0, 0, 1))
        pad = along(f'Leg {i} foot pad', (cx, cy, config['foot_pad_bottom_z']),
                    (cx, cy, config['foot_pad_top_z']),
                    config['foot_pad_radius'], 40)
        assign_tile(pad, 1, boxed=True, smooth='side', axis=(0, 0, 1))
        apply_bevel(pad, .0012, 2)

    # Large flanged top port with a domed blind cover and four swing bolts.
    port = config['top_port']
    py = port['y']
    head_z = z_top + h * math.sqrt(max(0.0, 1 - (abs(py) / R) ** 2))
    collar = torus('Top port collar', (0, py, head_z + .001), (0, 0, 1),
                   port['neck_outer'] + .003, .0045, 64, 8)
    assign_tile(collar, 1, boxed=True, smooth='all')
    hollow_pipe('Top port neck', (0, py, port['neck_z0']),
                (0, py, port['neck_z1']),
                port['neck_outer'], port['neck_bore'], 1, segments=72)
    flange = revolve_ring('Top port flange',
                          [(port['neck_bore'], port['flange_z0']),
                           (port['flange_outer'], port['flange_z0']),
                           (port['flange_outer'], port['flange_z1']),
                           (port['neck_bore'], port['flange_z1'])], 96)
    flange.location.y = py
    apply_bevel(flange, .0015, 2)
    assign_tile(flange, 1, boxed=True, smooth='all')
    gasket = revolve_ring('Top port gasket',
                          [(port['gasket_inner'], port['gasket_z0']),
                           (port['gasket_outer'], port['gasket_z0']),
                           (port['gasket_outer'], port['gasket_z1']),
                           (port['gasket_inner'], port['gasket_z1'])], 96)
    gasket.location.y = py
    assign_special(gasket, 'gasket', boxed=True, smooth='all')
    z0, z1 = port['lid_z0'], port['lid_z1']
    lid_profile = [(0.0, z0), (port['lid_radius'], z0),
                   (port['lid_radius'] + .0015, z0 + .004),
                   (port['lid_radius'], z0 + .010),
                   (port['lid_radius'] - .003, z0 + .018),
                   (port['lid_radius'] - .010, z0 + .028),
                   (port['lid_radius'] - .020, z0 + .038),
                   (port['lid_radius'] - .032, z0 + .046),
                   (port['lid_radius'] - .042, z0 + .053),
                   (0.0, z1)]
    lid = revolve_solid('Top port lid', lid_profile, 96)
    lid.location.y = py
    assign_tile(lid, 0, boxed=True, smooth='all')
    boss = along('Top port lid boss', (0, py, z1 - .002),
                 (0, py, z1 + .006), .010, 32)
    assign_tile(boss, 1, boxed=True, smooth='side', axis=(0, 0, 1))
    for i, az in enumerate(port['swing_bolt_azimuths_deg'], start=1):
        a = math.radians(az)
        bx = port['swing_bolt_radius'] * math.cos(a)
        by = py + port['swing_bolt_radius'] * math.sin(a)
        rod = along(f'Top port swing bolt {i}',
                    (bx, by, port['swing_bolt_z0']),
                    (bx, by, port['swing_bolt_z1']),
                    port['swing_bolt_radius_rod'], 16)
        assign_tile(rod, 3, boxed=True, smooth='side', axis=(0, 0, 1))
        nut = along(f'Top port nut {i}', (bx, by, port['nut_z0']),
                    (bx, by, port['nut_z1']), .0072, 6)
        assign_tile(nut, 3, boxed=True, smooth='flat')

    # Lifting lug: base pad, two upright bars and a torus eye (real through hole).
    lug = config['lift_lug']
    lx, ly = lug['x'], lug['y']
    base = box_at('Lift lug base', (lx, ly, (lug['base_z0'] + lug['base_z1']) / 2),
                  (.028, .026, (lug['base_z1'] - lug['base_z0']) / 2))
    apply_bevel(base, .0015, 2)
    assign_tile(base, 1, boxed=True, smooth='flat')
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        bar = box_at(f'Lift lug bar {suffix}',
                     (lx, ly + side * lug['bar_half_gap'],
                      (lug['bar_z0'] + lug['bar_z1']) / 2),
                     (lug['bar_thickness_x'] / 2, lug['bar_width_y'] / 2,
                      (lug['bar_z1'] - lug['bar_z0']) / 2))
        apply_bevel(bar, .0010, 2)
        assign_tile(bar, 1, boxed=True, smooth='flat')
    web = box_at('Lift lug web',
                 (lx, ly, (lug['bar_z0'] + lug['web_top']) / 2),
                 (.0025, lug['bar_half_gap'],
                  (lug['web_top'] - lug['bar_z0']) / 2))
    assign_tile(web, 1, boxed=True, smooth='flat')
    eye = torus('Lift lug eye', (lx, ly, lug['eye_z']), (1, 0, 0),
                lug['eye_major'], lug['eye_minor'], 36, 10)
    assign_tile(eye, 1, boxed=True, smooth='all')

    # Recirculation pipe: top flange, swept pipe, wall flange and small valve.
    pipe = config['top_pipe']
    hollow_pipe('Pipe top stub', (pipe['stub_x'], pipe['stub_y'], pipe['stub_z0']),
                (pipe['stub_x'], pipe['stub_y'], pipe['stub_z1']),
                pipe['stub_outer'], pipe['stub_bore'], 1, segments=48)
    top_flange = revolve_ring('Pipe top flange',
                              [(pipe['stub_bore'], pipe['flange_z0']),
                               (pipe['flange_outer'], pipe['flange_z0']),
                               (pipe['flange_outer'], pipe['flange_z1']),
                               (pipe['stub_bore'], pipe['flange_z1'])], 48)
    top_flange.location.x = pipe['stub_x']
    top_flange.location.y = pipe['stub_y']
    apply_bevel(top_flange, .0012, 2)
    assign_tile(top_flange, 1, boxed=True, smooth='all')
    for i in range(4):
        a = TAU * i / 4 + .4
        bx = pipe['stub_x'] + .018 * math.cos(a)
        by = pipe['stub_y'] + .018 * math.sin(a)
        bolt = along(f'Pipe top bolt {i + 1}', (bx, by, pipe['flange_z1'] - .002),
                     (bx, by, pipe['flange_z1'] + .006), .0032, 6)
        assign_tile(bolt, 3, boxed=True, smooth='flat')
    path = rounded_path(pipe['path'], pipe['corner_radius'], pipe['corner_segments'])
    sweep = swept_tube('Top pipe', path, pipe['outer'], pipe['bore'], 40)
    assign_tile(sweep, 1, boxed=True, smooth='all')
    sleeve = swept_tube('Top pipe sleeve', path, pipe['bore'] - .0002,
                        pipe['bore'] - .0014, 40)
    assign_special(sleeve, 'dark', boxed=True, smooth='all')

    az = math.radians(pipe['wall_azimuth_deg'])
    radial = Vector((math.cos(az), math.sin(az), 0))
    tangent = Vector((-math.sin(az), math.cos(az), 0))
    wz = pipe['wall_flange_z']
    wall_r = R - .004
    wall_center = Vector((radial.x * wall_r, radial.y * wall_r, wz))
    wall_flange = along('Pipe wall flange',
                        wall_center - radial * .006, wall_center + radial * .006,
                        pipe['wall_flange_outer'], 48)
    assign_tile(wall_flange, 1, boxed=True, smooth='side', axis=tuple(radial))
    for i in range(pipe['wall_bolt_count']):
        a = TAU * i / pipe['wall_bolt_count'] + .35
        offset = radial * .008 + tangent * (pipe['wall_bolt_radius'] * math.cos(a))
        offset += Vector((0, 0, pipe['wall_bolt_radius'] * math.sin(a)))
        p0 = wall_center + offset - radial * .004
        p1 = wall_center + offset + radial * .006
        bolt = along(f'Pipe wall bolt {i + 1}', p0, p1, .0030, 6)
        assign_tile(bolt, 3, boxed=True, smooth='flat')
    valve_center = wall_center + radial * .020
    valve = along('Pipe valve body', valve_center - radial * (pipe['valve_length'] / 2),
                  valve_center + radial * (pipe['valve_length'] / 2),
                  pipe['valve_radius'], 32)
    assign_tile(valve, 1, boxed=True, smooth='side', axis=tuple(radial))
    for side, nm in ((-1, 'inlet'), (1, 'outlet')):
        fl = along(f'Pipe valve flange {nm}',
                   valve_center + radial * (side * (pipe['valve_length'] / 2 + .001)),
                   valve_center + radial * (side * (pipe['valve_length'] / 2 + pipe['valve_flange_thickness'] + .001)),
                   pipe['valve_flange_outer'], 40)
        assign_tile(fl, 1, boxed=True, smooth='side', axis=tuple(radial))
    stem_dir = -tangent
    stem_base = valve_center + stem_dir * .012
    stem_tip = valve_center + stem_dir * (pipe['valve_stem_length'] + .012)
    stem = along('Pipe valve stem', stem_base, stem_tip, pipe['valve_stem_radius'], 16)
    assign_tile(stem, 1, boxed=True, smooth='side', axis=tuple(stem_dir))
    handle = along('Pipe valve handle',
                   stem_tip - Vector((0, 0, pipe['valve_handle_length'] / 2)),
                   stem_tip + Vector((0, 0, pipe['valve_handle_length'] / 2)),
                   pipe['valve_handle_radius'], 16)
    assign_tile(handle, 1, boxed=True, smooth='side', axis=(0, 0, 1))

    # Capsule sight glass on the lower front shell.
    sg = config['sight_glass']
    y0 = sg['half_width']
    zt, zb = sg['z_top'], sg['z_bottom']
    arch_center = zt - y0
    top_profile = []
    for i in range(25):
        a = math.pi * i / 24
        top_profile.append((y0 * math.cos(a), arch_center + y0 * math.sin(a)))
    profile_yz = [(-y0, zb)] + top_profile + [(y0, zb)]
    # Ensure the closed loop runs around the outside without duplicate points.
    profile_yz = [(-y0, zb), (y0, zb)] + top_profile[0:]
    pad = prism_yz('Sight glass pad', profile_yz, R - .002,
                   R - .002 + sg['pad_depth'])
    apply_bevel(pad, .0012, 2)
    assign_tile(pad, 1, boxed=True, smooth='all')
    bezel = torus('Sight glass bezel', (R - .002 + sg['pad_depth'], 0, sg['z_center']),
                  (1, 0, 0), sg['bezel_major'], sg['bezel_minor'], 48, 10)
    assign_tile(bezel, 1, boxed=True, smooth='all')
    lens = along('Sight glass lens', (R + sg['pad_depth'] - .004, 0, sg['z_center']),
                 (R + sg['pad_depth'] + .0015, 0, sg['z_center']),
                 sg['lens_radius'], 48)
    assign_special(lens, 'glass', boxed=True, smooth='all')
    plug = along('Sight glass plug', (R + sg['pad_depth'] - .001, 0, sg['plug_z']),
                 (R + sg['pad_depth'] + .007, 0, sg['plug_z']),
                 sg['plug_radius'], 6)
    assign_tile(plug, 1, boxed=True, smooth='flat')
    socket = along('Sight glass plug socket', (R + sg['pad_depth'] + .0065, 0, sg['plug_z']),
                   (R + sg['pad_depth'] + .0085, 0, sg['plug_z']), .0035, 6)
    assign_special(socket, 'dark', boxed=True, smooth='flat')

    # Stainless skid frame and agitator drive.
    sk = config['skid']
    frame_z = sk['frame_z']
    ymid = (sk['y_min'] + sk['y_max']) / 2
    yhalf = (sk['y_max'] - sk['y_min']) / 2
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        rail = box_at(f'Skid rail {suffix}', (side * sk['rail_x'], ymid, frame_z),
                      (sk['rail_cross'] / 2, yhalf, sk['rail_cross'] / 2))
        apply_bevel(rail, .0010, 2)
        assign_tile(rail, 1, boxed=True, smooth='flat')
    for y, suffix in ((sk['y_min'], 'front'), (sk['y_max'], 'back')):
        rail = box_at(f'Skid cross rail {suffix}', (0, y, frame_z),
                      (sk['rail_x'] + sk['rail_cross'] / 2, sk['rail_cross'] / 2,
                       sk['rail_cross'] / 2))
        apply_bevel(rail, .0010, 2)
        assign_tile(rail, 1, boxed=True, smooth='flat')
    head_at = lambda x, y: z_top + h * math.sqrt(max(0.0, 1 - (x * x + y * y) / (R * R)))
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        px = side * sk['post_x']
        pz0 = head_at(px, sk['post_y'])
        post = box_at(f'Skid post {suffix}', (px, sk['post_y'],
                                              (pz0 + frame_z - sk['rail_cross'] / 2) / 2),
                      (sk['post_section'] / 2, sk['post_section'] / 2,
                       (frame_z - sk['rail_cross'] / 2 - pz0) / 2))
        apply_bevel(post, .0010, 2)
        assign_tile(post, 1, boxed=True, smooth='flat')
        pad = along(f'Skid post pad {suffix}', (px, sk['post_y'], pz0 - .002),
                    (px, sk['post_y'], pz0 + .004), .020, 32)
        assign_tile(pad, 1, boxed=True, smooth='side', axis=(0, 0, 1))
    for i, (cx, cy) in enumerate(sk['column_xy'], start=1):
        cz0 = head_at(cx, cy)
        col = along(f'Skid column {i}', (cx, cy, cz0 - .002),
                    (cx, cy, frame_z - sk['rail_cross'] / 2),
                    sk['column_radius'], 32)
        assign_tile(col, 1, boxed=True, smooth='side', axis=(0, 0, 1))
        for z, nm in ((cz0 + .003, 'base'), (frame_z - sk['rail_cross'] / 2 - .003, 'top')):
            fl = along(f'Skid column {i} flange {nm}', (cx, cy, z - sk['column_flange_thickness'] / 2),
                       (cx, cy, z + sk['column_flange_thickness'] / 2),
                       sk['column_flange_radius'], 40)
            assign_tile(fl, 1, boxed=True, smooth='side', axis=(0, 0, 1))
    plate = box_at('Skid base plate',
                   (0, (sk['base_plate_y0'] + sk['base_plate_y1']) / 2,
                    frame_z + sk['rail_cross'] / 2 + sk['base_plate_thickness'] / 2),
                   (sk['base_plate_x'] / 2,
                    (sk['base_plate_y1'] - sk['base_plate_y0']) / 2,
                    sk['base_plate_thickness'] / 2))
    apply_bevel(plate, .0010, 2)
    assign_tile(plate, 1, boxed=True, smooth='flat')

    mo = config['motor']
    shaft_x, shaft_y = mo['drive_shaft_x'], mo['drive_shaft_y']
    shaft_head_z = head_at(shaft_x, shaft_y)
    shaft = along('Drive shaft', (shaft_x, shaft_y, mo['drive_shaft_z0']),
                  (shaft_x, shaft_y, mo['drive_shaft_z1']),
                  mo['drive_shaft_radius'], 24)
    assign_tile(shaft, 1, boxed=True, smooth='side', axis=(0, 0, 1))
    coupling = along('Drive shaft coupling', (shaft_x, shaft_y, mo['drive_coupling_z0']),
                     (shaft_x, shaft_y, mo['drive_coupling_z1']),
                     mo['drive_coupling_radius'], 32)
    assign_tile(coupling, 1, boxed=True, smooth='side', axis=(0, 0, 1))
    head_flange = along('Drive shaft head flange',
                        (shaft_x, shaft_y, shaft_head_z - .002),
                        (shaft_x, shaft_y, shaft_head_z + .006),
                        mo['head_flange_radius'], 40)
    assign_tile(head_flange, 1, boxed=True, smooth='side', axis=(0, 0, 1))

    # Motor and gearbox (blue paint), with cooling fins and a fan cowl.
    for fy in mo['foot_y']:
        foot = box_at(f'Motor foot {fy:.3f}', (0, fy, (mo['foot_z0'] + mo['foot_z1']) / 2),
                      (mo['foot_half_x'], .012, (mo['foot_z1'] - mo['foot_z0']) / 2))
        assign_special(foot, 'paint', boxed=True, smooth='flat')
    body = along('Motor body', (0, mo['body_y0'], mo['axis_z']),
                 (0, mo['body_y1'], mo['axis_z']), mo['body_radius'], 48)
    assign_special(body, 'paint', boxed=True, smooth='all')
    for i in range(mo['fin_count']):
        a = TAU * i / mo['fin_count']
        fin = box_at(f'Motor fin {i + 1}', (0, 0, 0),
                     (mo['fin_thickness'] / 2, mo['fin_length'] / 2, mo['fin_height'] / 2))
        fin.rotation_euler = (0, a, 0)
        fin.location = (math.sin(a) * (mo['body_radius'] + .0015),
                        (mo['body_y0'] + mo['body_y1']) / 2,
                        mo['axis_z'] + math.cos(a) * (mo['body_radius'] + .0015))
        assign_special(fin, 'paint', boxed=True, smooth='flat')
    gb = box_at('Gearbox', (0, (mo['gearbox_y0'] + mo['gearbox_y1']) / 2, mo['axis_z']),
                (mo['gearbox_half_x'], (mo['gearbox_y1'] - mo['gearbox_y0']) / 2,
                 mo['gearbox_half_z']))
    apply_bevel(gb, .004, 3)
    assign_special(gb, 'paint', boxed=True, smooth='flat')
    for i in range(mo['gearbox_rib_count']):
        z = mo['axis_z'] - mo['gearbox_half_z'] + .012 + i * (.018)
        rib = box_at(f'Gearbox rib {i + 1}',
                     (0, (mo['gearbox_y0'] + mo['gearbox_y1']) / 2, z),
                     (mo['gearbox_half_x'] + .002, .020, .003))
        assign_special(rib, 'paint', boxed=True, smooth='flat')
    fan = along('Fan cowl', (0, mo['fan_y0'], mo['axis_z']),
                (0, mo['fan_y1'], mo['axis_z']), mo['fan_radius'], 48)
    assign_special(fan, 'paint', boxed=True, smooth='all')
    rim = torus('Fan cowl rim', (0, mo['fan_y1'], mo['axis_z']), (0, 1, 0),
                mo['fan_rim_major'], mo['fan_rim_minor'], 48, 8)
    assign_special(rim, 'paint', boxed=True, smooth='all')
    grille = along('Fan grille', (0, mo['fan_y1'] + .0005, mo['axis_z']),
                   (0, mo['fan_y1'] + .0025, mo['axis_z']), mo['grille_radius'], 48)
    assign_special(grille, 'dark', boxed=True, smooth='all')
    for i in range(mo['grille_bars']):
        a = TAU * i / mo['grille_bars']
        bar = box_at(f'Fan grille bar {i + 1}', (0, 0, 0),
                     (.0035, .002, mo['grille_radius'] * .9))
        bar.rotation_euler = (0, a, 0)
        bar.location = (0, mo['fan_y1'] + .004, mo['axis_z'])
        assign_tile(bar, 1, boxed=True, smooth='flat')
    terminal = box_at('Motor terminal box',
                      (0, mo['terminal_y'], (mo['terminal_z0'] + mo['terminal_z1']) / 2),
                      (mo['terminal_half_x'], mo['terminal_half_y'],
                       (mo['terminal_z1'] - mo['terminal_z0']) / 2))
    apply_bevel(terminal, .002, 2)
    assign_special(terminal, 'paint', boxed=True, smooth='flat')
    lid = box_at('Motor terminal lid',
                 (0, mo['terminal_y'], mo['terminal_z1'] + .0015),
                 (mo['terminal_half_x'] + .002, mo['terminal_half_y'] + .002, .0015))
    assign_special(lid, 'paint', boxed=True, smooth='flat')
    eyebolt = along('Motor eye bolt', (0, mo['terminal_y'], mo['terminal_z1'] + .003),
                    (0, mo['terminal_y'], mo['terminal_z1'] + .012), .004, 12)
    assign_tile(eyebolt, 3, boxed=True, smooth='side', axis=(0, 0, 1))


# ---------------------------------------------------------------------------
# Render setup
# ---------------------------------------------------------------------------
def camera(scene, direction, center, span, scale=1.27):
    direction = Vector(direction).normalized()
    cam = legacy.make_camera(scene, Vector(center) + direction * span * 5, center,
                             ortho_scale=span * scale)
    if abs(direction.z) > .999:
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.rotation_quaternion @ Quaternion((0, 0, 1), math.pi / 2)
    return cam


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
        ('large softbox', (2.6, -3.0, 3.6), 230, 3.2, 3.2),
        ('fill softbox', (-3.2, 2.6, 2.2), 120, 3.0, 3.0),
        ('front strip', (3.2, 1.0, 1.3), 120, 1.2, 3.4),
        ('rim high', (-1.2, 3.2, 3.4), 90, 1.4, 2.2),
        ('low fill', (0, -2.6, -.28), 50, 1.0, 1.0)):
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
Z_SHIFT = -center.z
for obj in objects:
    obj.location.z += Z_SHIFT
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
bpy.ops.wm.open_mainfile(filepath=str(out / 'model.blend'))
scene = bpy.context.scene
scene.render.threads_mode = 'AUTO'
objects = [bpy.data.objects[name] for name in object_names]
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
parts = [{'name': o.name, 'vertices': len(o.data.vertices),
          'faces': len(o.data.polygons)} for o in objects]
intersections = {o.name: intersecting_faces(o) for o in objects}
render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
directions = {'front': (1, 0, .16), 'back': (-1, 0, .16),
              'left': (0, -1, .18), 'right': (0, 1, .18),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 72
for name, direction in directions.items():
    cam = camera(scene, direction, center, span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .70)
draw(scene, render_dir / 'new-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)

if not request['draft']:
    for i in range(24):
        angle = TAU * i / 24
        cam = camera(scene, (math.cos(angle), math.sin(angle), .15), center, span)
        draw(scene, render_dir / f'turn-{i:02}.png', resolution=512, samples=8)
        bpy.data.objects.remove(cam, do_unlink=True)

    detail_cameras = (
        ('top-port', (0, -0.202, 0.40 + Z_SHIFT), (1.0, -0.8, 0.9), .32),
        ('lift-lug', (0, 0.005, 0.405 + Z_SHIFT), (1.0, 0.2, 0.35), .17),
        ('motor', (0, 0.176, 0.46 + Z_SHIFT), (1.0, -0.7, 0.6), .36),
        ('skid', (0, 0.176, 0.39 + Z_SHIFT), (1.0, -0.2, 0.2), .40),
        ('side-pipe', (0.18, -0.20, 0.20 + Z_SHIFT), (1.0, -1.0, 0.2), .55),
        ('pipe-valve', (0.18, -0.196, -0.13 + Z_SHIFT), (1.0, -0.8, -0.2), .24),
        ('sight-glass', (0.25, 0, -0.175 + Z_SHIFT), (1.0, -0.25, 0.1), .21),
        ('cone-flange', (0, 0, -0.26 + Z_SHIFT), (1.0, -0.8, 0.25), .36),
        ('legs', (0.20, -0.16, -0.33 + Z_SHIFT), (1.0, -0.4, 0.3), .55),
        ('foot', (0.20, -0.15, -0.48 + Z_SHIFT), (1.0, -1.0, 0.4), .20),
        ('seam', (0, -0.20, 0.05 + Z_SHIFT), (1.0, -0.6, 0.3), .30),
        ('surface', (0.23, -0.08, -0.05 + Z_SHIFT), (1.0, -0.5, 0.3), .30),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('top-port-only', lambda n: n.startswith('Top port neck')
         or n.startswith('Top port flange') or n.startswith('Top port neck sleeve'),
         (0, 0, 1), (0, -0.202, 0.40 + Z_SHIFT), .22),
        ('lift-lug-only', lambda n: n.startswith('Lift lug'),
         (1, 0, .05), (0, 0.005, 0.402 + Z_SHIFT), .15),
        ('drain-only', lambda n: n.startswith('Drain'),
         (0, 0, -1), (0, 0, -0.41 + Z_SHIFT), .10),
        ('legs-only', lambda n: n.startswith('Leg '),
         (1, -1, .25), (0, 0, -0.33 + Z_SHIFT), .90),
        ('motor-only', lambda n: n.startswith(('Motor ', 'Gearbox', 'Fan ', 'Skid ')),
         (1, -0.6, .5), (0, 0.175, 0.42 + Z_SHIFT), .55),
    )
    for label, predicate, direction, target, scale in groups:
        visible = [o.name for o in objects if predicate(o.name)]
        for obj in objects:
            obj.hide_render = obj.name not in visible
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'{label}.png', resolution=512, samples=16)
        bpy.data.objects.remove(cam, do_unlink=True)
    for obj in objects:
        obj.hide_render = False

    key = bpy.data.objects['large softbox']
    key.location = (-3.4, 2.8, 4.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()
    cam = camera(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', resolution=768, samples=24)
    bpy.data.objects.remove(cam, do_unlink=True)
    cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .70)
    draw(scene, render_dir / 'new-metal-alternate.png', resolution=1024, samples=32)
    bpy.data.objects.remove(cam, do_unlink=True)
    key.location = (2.6, -3.0, 3.6)
    key.rotation_euler = (-key.location).to_track_quat('-Z', 'Y').to_euler()

# Source-photo framing diagnostic (not an acceptance gate).
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
cam = camera(scene, (2.3, -2.0, 1.0), Vector((0, 0, center.z)), .70)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                         for p in o.data.polygons),
        'texture_size': 2048, 'paint_texture_size': 1024,
        'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': 'original photo proportions; six-view sheet is an uncalibrated category reference',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': 'Front camera at +X, left at -Y, right at +Y; the sheet is uncalibrated and its top stirring parts differ.',
        'source_camera_note': 'source-photo framing distance scaled by new/old span; diagnostic only',
        'source_camera_factor': factor,
        'isolated_checks': {'top_port_expected': 1, 'lift_lug_expected': 1,
                            'drain_expected': 1}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
