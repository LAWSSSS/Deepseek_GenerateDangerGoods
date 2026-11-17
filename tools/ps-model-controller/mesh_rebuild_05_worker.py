"""Blender worker for the review-only vertical stainless storage-tank candidate (item 05).

Shape basis: the original 3/4 rust photograph plus the user-supplied, uncalibrated
six-view sheet.  The working diameter follows the model-library visual scale; no
dimension here is presented as a measurement of the photographed unit.  The
candidate never replaces the library mesh and is never used by the automatic
placement pipeline.
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
    names = {0: '05 bright stainless', 1: '05 brushed stainless',
             2: '05 weathered stainless', 3: '05 stained steel'}
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

    dark = bpy.data.materials.new('05 dark bore')
    dark.use_nodes = True
    principled = dark.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.008, .009, .009, 1)
    principled.inputs['Metallic'].default_value = .30
    principled.inputs['Roughness'].default_value = .58
    special_materials['dark'] = dark

    gasket = bpy.data.materials.new('05 manway gasket')
    gasket.use_nodes = True
    principled = gasket.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.012, .013, .012, 1)
    principled.inputs['Metallic'].default_value = .00
    principled.inputs['Roughness'].default_value = .78
    special_materials['gasket'] = gasket


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
        box_uv(obj)
    set_smooth(obj, smooth, axis)
    objects.append(obj)
    return obj


def hollow_pipe(name, start, end, r_out, r_in, tile, segments=64):
    """One steel thick-walled pipe plus a separate dark liner sleeve.

    Keeping the liner as its own watertight object avoids a material seam that
    the glTF exporter would split into two open primitives (and it keeps the
    isolated opening render unambiguous).
    """
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


def boolean_cut(target, cutter, label):
    mod = target.modifiers.new(label, 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def revolve_solid(name, profile, segments):
    """Revolve a pole-to-pole (r,z) profile into a closed solid."""
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
    """Revolve a closed (r,z) profile into a closed ring solid."""
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


def rotate_rt(r, t, angle_deg):
    a = math.radians(angle_deg)
    return (r * math.cos(a) - t * math.sin(a), r * math.sin(a) + t * math.cos(a))


def box_world(name, r_center, z_center, half_r, half_t, half_z, angle_deg):
    pts = []
    for z in (z_center - half_z, z_center + half_z):
        for sr, st in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = rotate_rt(r_center + sr * half_r, st * half_t, angle_deg)
            pts.append((x, y, z))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, pts, faces)


def tapered_leg(name, r_top, z_top, w_top, r_bot, z_bot, w_bot, thickness, angle_deg):
    pts = []
    for r, z, w in ((r_top, z_top, w_top), (r_bot, z_bot, w_bot)):
        for sr, st in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = rotate_rt(r + sr * thickness / 2, st * w / 2, angle_deg)
            pts.append((x, y, z))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, pts, faces)


def slanted_slab(name, p0, p1, half_thick, half_width, angle_deg):
    """A slab following a line in the local radial-vertical plane."""
    (r0, z0), (r1, z1) = p0, p1
    dr, dz = r1 - r0, z1 - z0
    length = math.hypot(dr, dz)
    nr, nz = -dz / length, dr / length
    corners = []
    for (r, z) in ((r0, z0), (r1, z1)):
        for sign in (-1, 1):
            corners.append((r + sign * nr * half_thick, z + sign * nz * half_thick))
    # corners order: p0-, p0+, p1-, p1+; build two tangential layers
    verts = []
    for t in (-half_width, half_width):
        for (r, z) in corners:
            x, y = rotate_rt(r, t, angle_deg)
            verts.append((x, y, z))
    faces = [(0, 1, 3, 2), (4, 6, 7, 5),
             (0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6)]
    return mesh_object(name, verts, faces)


def capsule_prism(name, r_center, half_len, radius, z0, z1, angle_deg, segments=16):
    pts = []
    for i in range(segments + 1):
        a = -math.pi / 2 + math.pi * i / segments
        pts.append((r_center + half_len + radius * math.cos(a),
                    radius * math.sin(a)))
    for i in range(segments + 1):
        a = math.pi / 2 + math.pi * i / segments
        pts.append((r_center - half_len + radius * math.cos(a),
                    radius * math.sin(a)))
    n = len(pts)
    verts = []
    for z in (z0, z1):
        for r, t in pts:
            x, y = rotate_rt(r, t, angle_deg)
            verts.append((x, y, z))
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
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
    R = config['diameter'] / 2
    zb, zt = config['shell_bottom_z'], config['shell_top_z']
    db, dt = config['bottom_head_depth'], config['top_head_depth']

    # Tank body: dished bottom + cylindrical shell + flatter dished top.
    profile = [(0.0, zb - db)]
    for i in range(1, 33):
        f = i / 32
        profile.append((R * f, zb - db * math.sqrt(max(0.0, 1 - f * f))))
    profile.append((R, zt))
    for i in range(1, 33):
        f = 1 - i / 32
        profile.append((R * f, zt + dt * math.sqrt(max(0.0, 1 - f * f))))
    body = revolve_solid('Tank body', profile, config['radial_segments'])
    assign_tile(body, 0, boxed=True, smooth='all')

    # Raised weld bands at the two head/shell tangents.
    for name, z in (('Shell seam lower', zb), ('Shell seam upper', zt)):
        ring = revolve_ring(name,
                            [(R - .002, z - .006), (R + .0035, z - .006),
                             (R + .0035, z + .006), (R - .002, z + .006)],
                            config['radial_segments'])
        assign_tile(ring, 1, boxed=True, smooth='all')

    # Three plate legs, slanted brackets and slotted foot plates.
    for i, angle in enumerate(config['leg_angles_deg'], start=1):
        leg = tapered_leg(f'Leg {i} plate',
                          config['leg_top_radius'], config['leg_top_z'],
                          config['leg_top_width'], config['leg_bottom_radius'],
                          config['leg_bottom_z'], config['leg_bottom_width'],
                          config['leg_thickness'], angle)
        apply_bevel(leg, .0018, 2)
        assign_tile(leg, 1, boxed=True, smooth='flat')

        bracket = slanted_slab(f'Leg {i} bracket', (.165, .178), (.252, .246),
                               .0065, .036, angle)
        apply_bevel(bracket, .0018, 2)
        assign_tile(bracket, 2, boxed=True, smooth='flat')

        foot = box_world(f'Foot plate {i}', config['foot_radius'],
                         config['foot_thickness'] / 2,
                         config['foot_radial_length'] / 2,
                         config['foot_width'] / 2,
                         config['foot_thickness'] / 2, angle)
        cutter = capsule_prism(f'Foot slot cutter {i}', config['foot_slot_radius'],
                               (config['foot_slot_length'] - config['foot_slot_width']) / 2,
                               config['foot_slot_width'] / 2,
                               -.02, config['foot_thickness'] + .02, angle)
        boolean_cut(foot, cutter, f'Foot slot {i}')
        assign_tile(foot, 3, boxed=True, smooth='flat')

    # Manway: neck, flange, gasket, domed lid, two hinge lugs, U handle.
    neck, _ = tube('Manway neck',
                   (0, 0, config['manway_neck_z0']),
                   (0, 0, config['manway_neck_z1']),
                   config['manway_neck_outer'], config['manway_neck_inner'],
                   segments=96)
    assign_tile(neck, 0, boxed=True, smooth='all')
    flange = revolve_ring('Manway flange',
                          [(config['manway_neck_inner'], config['manway_flange_z0']),
                           (config['manway_flange_outer'], config['manway_flange_z0']),
                           (config['manway_flange_outer'], config['manway_flange_z1']),
                           (config['manway_neck_inner'], config['manway_flange_z1'])],
                          96)
    assign_tile(flange, 0, boxed=True, smooth='all')
    gasket = revolve_ring('Manway gasket',
                          [(.0615, 1.079), (.0785, 1.079),
                           (.0785, 1.0835), (.0615, 1.0835)], 96)
    assign_special(gasket, 'gasket', boxed=True, smooth='all')

    lid_profile = [(0.0, config['manway_lid_z0']),
                   (.080, config['manway_lid_z0']),
                   (config['manway_lid_radius'], config['manway_lid_z0'] + .004),
                   (.079, 1.095), (.055, 1.099), (0.0, config['manway_lid_z1'])]
    lid = revolve_solid('Manway lid', lid_profile, 96)
    assign_tile(lid, 0, boxed=True, smooth='all')

    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        hinge = box_world(f'Manway hinge {suffix}', .088, 1.080,
                          .018, .017, .010, 90 if side > 0 else 270)
        # Hinge lugs sit at +/-Y; rotate the local radial axis accordingly.
        # The box above is built with its long axis radial to the manway.
        apply_bevel(hinge, .0015, 2)
        assign_tile(hinge, 2, boxed=True, smooth='flat')

    for side, suffix in ((-1, 'left'), (1, 'right')):
        post = along(f'Manway handle post {suffix}', (side * .038, 0, 1.097),
                     (side * .038, 0, 1.115), .0045, 32)
        assign_tile(post, 1, boxed=True, smooth='side', axis=(0, 0, 1))
    bar = along('Manway handle bar', (-.038, 0, 1.115), (.038, 0, 1.115), .0045, 32)
    assign_tile(bar, 1, boxed=True, smooth='side', axis=(1, 0, 0))

    # Six swing clamps: welded lug, slanted bolt and an eye ring visible from above.
    for i, angle in enumerate(config['manway_clamp_angles_deg'], start=1):
        a = math.radians(angle)
        radial = Vector((math.cos(a), math.sin(a), 0))
        tangent = Vector((-math.sin(a), math.cos(a), 0))
        lug = box_world(f'Manway clamp {i} lug', .086, 1.082, .011, .008, .008, angle)
        apply_bevel(lug, .0012, 2)
        assign_tile(lug, 2, boxed=True, smooth='flat')
        p0 = radial * .085 + Vector((0, 0, 1.082))
        p1 = radial * .098 + Vector((0, 0, 1.107))
        bolt = along(f'Manway clamp {i} bolt', p0, p1, .0042, 24)
        assign_tile(bolt, 1, boxed=True, smooth='side',
                    axis=(p1 - p0).normalized())
        eye_normal = (radial * .50 + Vector((0, 0, .866))).normalized()
        eye = torus(f'Manway clamp {i} eye', p1, eye_normal, .011, .0035, 32, 8)
        assign_tile(eye, 1, boxed=True, smooth='all')

    # Four top nozzles in a diagonal ring around the manway.
    for spec in config['top_nozzles']:
        a = math.radians(spec['angle_deg'])
        x, y = math.cos(a) * config['top_nozzle_radius'], math.sin(a) * config['top_nozzle_radius']
        z0 = 1.028
        z1 = spec['top_z']
        hollow_pipe(spec['name'], (x, y, z0), (x, y, z1),
                    spec['outer'], spec['bore'], 1, segments=48)
        if spec['kind'] == 'flange':
            ring = tube(spec['name'] + ' flange', (x, y, z1 - .008), (x, y, z1),
                        config['top_flange_outer'], spec['bore'] + .002, segments=64)
            assign_tile(ring[0], 1, boxed=True, smooth='side', axis=(0, 0, 1))
            for b in range(config['top_bolt_count']):
                ba = TAU * b / config['top_bolt_count'] + .26
                bx, by = x + math.cos(ba) * .033, y + math.sin(ba) * .033
                bolt = along(f"{spec['name']} bolt {b + 1}", (bx, by, z1),
                             (bx, by, z1 + .007), .0045, 16)
                assign_tile(bolt, 2, boxed=True, smooth='side', axis=(0, 0, 1))
        else:
            lip = torus(spec['name'] + ' lip', (x, y, z1), (0, 0, 1),
                        .024, .005, 40, 8)
            assign_tile(lip, 1, boxed=True, smooth='all')

    # Two inferred small side ferrules near the upper shell, capped.
    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        y0 = side * (R - .008)
        y1 = side * (R + .030)
        pipe, _ = tube(f'Side ferrule {suffix}', (0, y0, config['side_ferrule_z']),
                       (0, y1, config['side_ferrule_z']),
                       config['side_ferrule_outer'], config['side_ferrule_bore'],
                       segments=48)
        assign_tile(pipe, 1, boxed=True, smooth='all')
        cap = along(f'Side ferrule {suffix} cap', (0, y1, config['side_ferrule_z']),
                    (0, y1 + side * .012, config['side_ferrule_z']), .026, 32)
        assign_tile(cap, 2, boxed=True, smooth='side', axis=(0, 1, 0))

    # Two front ports: open bores with hex boss and flange.
    port_specs = (('Front port upper', config['upper_front_port']),
                  ('Front port lower', config['lower_front_port']))
    for name, spec in port_specs:
        z = spec['z']
        if name.endswith('upper'):
            x0, x1 = R - .008, R + .050
        else:
            x0, x1 = R - .030, R + .040
        hollow_pipe(name, (x0, 0, z), (x1, 0, z),
                    spec['outer'], spec['bore'], 1, segments=48)
        boss = tube(name + ' hex boss', (x0 + .020, 0, z), (x0 + .034, 0, z),
                    .036, spec['bore'] + .002, segments=6)
        assign_tile(boss[0], 1, boxed=True, smooth='side', axis=(1, 0, 0))
        flange = tube(name + ' flange', (x1 - .008, 0, z), (x1, 0, z),
                      spec['flange'], spec['bore'] + .002, segments=64)
        assign_tile(flange[0], 1, boxed=True, smooth='side', axis=(1, 0, 0))

    # Bottom drain with an inferred ball-valve body and horizontal T handle.
    dz0, dz1 = config['drain_z_bottom'], config['drain_z_top']
    hollow_pipe('Drain pipe', (0, 0, dz0), (0, 0, dz1),
                config['drain_outer'], config['drain_bore'], 2, segments=48)
    flange = tube('Drain flange', (0, 0, dz1 - .008), (0, 0, dz1),
                  config['drain_flange'], config['drain_bore'] + .002, segments=64)
    assign_tile(flange[0], 2, boxed=True, smooth='side', axis=(0, 0, 1))
    valve = tube('Drain valve body', (0, 0, .115), (0, 0, .146),
                 .043, config['drain_bore'] + .002, segments=48)
    assign_tile(valve[0], 2, boxed=True, smooth='side', axis=(0, 0, 1))
    collar = tube('Drain collar', (0, 0, dz0), (0, 0, dz0 + .009),
                  .036, config['drain_bore'] + .002, segments=48)
    assign_tile(collar[0], 2, boxed=True, smooth='side', axis=(0, 0, 1))
    stem = along('Drain valve stem', (.040, 0, .130), (.082, 0, .130), .0045, 16)
    assign_tile(stem, 1, boxed=True, smooth='side', axis=(1, 0, 0))
    bar = along('Drain valve bar', (.082, -.035, .130), (.082, .035, .130), .005, 16)
    assign_tile(bar, 1, boxed=True, smooth='side', axis=(0, 1, 0))


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
# Keep the build-space centre for detail-camera targets, then centre the
# exported model vertically at z=0 (the x/y centre is already 0 by symmetry).
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
directions = {'front': (1, 0, .18), 'back': (-1, 0, .18),
              'left': (0, -1, .20), 'right': (0, 1, .20),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
resolution = 512 if request['draft'] else 768
samples = 8 if request['draft'] else 96
for name, direction in directions.items():
    cam = camera(scene, direction, center, span)
    draw(scene, render_dir / f'new-{name}.png', resolution=resolution, samples=samples)
    bpy.data.objects.remove(cam, do_unlink=True)

cam = camera(scene, (3, -.55, .30), Vector((0, 0, center.z)), .62)
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
        ('manway', (0, 0, 1.090 + Z_SHIFT), (1.2, -1.0, 1.1), .34),
        ('clamps', (.070, 0, 1.083 + Z_SHIFT), (1.4, -.6, .55), .30),
        ('top-nozzles', (0, 0, 1.055 + Z_SHIFT), (1.0, -.7, 1.4), .46),
        ('front-port', (.295, 0, .900 + Z_SHIFT), (1.2, -.5, .20), .23),
        ('lower-port', (.285, 0, .250 + Z_SHIFT), (1.2, -.5, .12), .25),
        ('legs', (.240, .110, .155 + Z_SHIFT), (1.45, -.9, .35), .58),
        ('foot', (.330, .055, .020 + Z_SHIFT), (1.3, -1.0, .75), .26),
        ('drain', (0, 0, .130 + Z_SHIFT), (.9, -.7, -.65), .27),
        ('seam', (.200, -.150, .945 + Z_SHIFT), (1.2, -.7, .45), .30),
        ('surface', (.180, -.200, .560 + Z_SHIFT), (1.6, -1.0, .45), .32),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('top-nozzles-only', lambda n: n.startswith('Top nozzle'), (0, 0, 1),
         (0, 0, 1.055 + Z_SHIFT), .60),
        ('front-ports-only', lambda n: n.startswith('Front port'), (1, 0, 0),
         (0, 0, .575 + Z_SHIFT), .95),
        ('drain-only', lambda n: n.startswith('Drain'), (0, 0, -1),
         (0, 0, .125 + Z_SHIFT), .38),
        ('feet-only', lambda n: n.startswith('Foot plate'), (0, 0, -1),
         (0, 0, .020 + Z_SHIFT), .90),
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
    cam = camera(scene, (3, -.55, .30), Vector((0, 0, center.z)), .62)
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
cam = camera(scene, (3, -.55, .30), Vector((0, 0, center.z)), .62)
draw(scene, render_dir / 'old-material-close.png',
     resolution=640 if request['draft'] else 1024, samples=8 if request['draft'] else 32)
bpy.data.objects.remove(cam, do_unlink=True)
bpy.data.objects.remove(old, do_unlink=True)

data = {'parts': parts, 'bounds': [list(low), list(high)], 'span': span,
        'parts_count': len(parts),
        'polygons': sum(len(o.data.polygons) for o in objects),
        'triangles': sum(max(0, len(p.vertices) - 2) for o in objects
                         for p in o.data.polygons),
        'texture_size': 2048, 'materials': sorted(material_for_tile),
        'special_materials': sorted(special_materials),
        'self_intersecting_components': {k: v for k, v in intersections.items() if v},
        'intersection_check': 'BVH same-part face overlap; assembly overlap between parts is not judged',
        'geometry_basis': 'original 3/4 photo proportions plus uncalibrated six-view construction details',
        'render_samples': samples,
        'view_directions': {k: list(v) for k, v in directions.items()},
        'view_note': 'Left camera at -Y and right at +Y; the six-view sheet is uncalibrated and partially mirror-inconsistent.',
        'source_camera_note': 'source-photo framing distance scaled by new/old span; diagnostic only',
        'source_camera_factor': factor,
        'isolated_checks': {'top_nozzles_expected': 4, 'front_ports_expected': 2,
                            'drain_expected': 1, 'foot_slots_expected': 3}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
