"""Blender worker for the review-only horizontal stainless pressure-tank candidate (item 06).

Shape basis: the original 3/4 rust photograph for proportions and visible
features; the user-supplied, uncalibrated six-view sheet for hidden/extra
construction details.  The working diameter follows the model-library visual
scale; no dimension here is presented as a measurement of the photographed
unit.  The candidate never replaces the library mesh and is never used by the
automatic placement pipeline.
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
    names = {0: '06 bright stainless', 1: '06 brushed stainless',
             2: '06 weathered stainless', 3: '06 stained steel'}
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

    dark = bpy.data.materials.new('06 dark bore')
    dark.use_nodes = True
    principled = dark.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value = (.008, .009, .009, 1)
    principled.inputs['Metallic'].default_value = .30
    principled.inputs['Roughness'].default_value = .58
    special_materials['dark'] = dark

    gasket = bpy.data.materials.new('06 manway gasket')
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
    """One steel thick-walled pipe plus a separate dark liner sleeve."""
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


def revolve_solid_y(name, profile, segments):
    """Revolve a pole-to-pole (r,y) profile about the Y axis into a closed solid."""
    verts, rings = [], []
    for r, y in profile:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, y, 0.0))
        else:
            ids = []
            for s in range(segments):
                a = TAU * s / segments
                ids.append(len(verts))
                verts.append((r * math.cos(a), y, r * math.sin(a)))
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


def revolve_ring_y(name, profile, segments):
    """Revolve a closed (r,y) profile about the Y axis into a closed ring solid."""
    verts, rings = [], []
    for r, y in profile:
        ids = []
        for s in range(segments):
            a = TAU * s / segments
            ids.append(len(verts))
            verts.append((r * math.cos(a), y, r * math.sin(a)))
        rings.append(ids)
    n = len(rings)
    faces = []
    for k in range(n):
        j = (k + 1) % n
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((rings[k][s], rings[k][s1], rings[j][s1], rings[j][s]))
    return mesh_object(name, verts, faces)


def revolve_solid(name, profile, segments):
    """Revolve a pole-to-pole (r,z) profile about Z; used for the manway lid."""
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


def tapered_leg_y(name, y_center, z_top, z_bottom, w_top, w_bottom,
                  d_top, d_bottom):
    verts = []
    for z, w, d in ((z_top, w_top, d_top), (z_bottom, w_bottom, d_bottom)):
        for dx, dy in ((-d / 2, -w / 2), (d / 2, -w / 2),
                       (d / 2, w / 2), (-d / 2, w / 2)):
            verts.append((dx, y_center + dy, z))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7),
             (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def annular_sector_y(name, y0, y1, r_in, r_out, angle0, angle1, segments=24):
    verts = []
    for y in (y0, y1):
        for r in (r_in, r_out):
            for s in range(segments + 1):
                a = math.radians(angle0 + (angle1 - angle0) * s / segments)
                verts.append((r * math.cos(a), y, r * math.sin(a)))
    # vertex index = (y_index*2 + r_index)*(segments+1) + s
    n = segments + 1

    def idx(yi, ri, s):
        return (yi * 2 + ri) * n + s

    faces = []
    for s in range(segments):
        # outer and inner cylindrical surfaces
        faces.append((idx(0, 1, s), idx(0, 1, s + 1), idx(1, 1, s + 1), idx(1, 1, s)))
        faces.append((idx(0, 0, s), idx(1, 0, s), idx(1, 0, s + 1), idx(0, 0, s + 1)))
        # side caps at y0 and y1
        faces.append((idx(0, 0, s), idx(0, 0, s + 1), idx(0, 1, s + 1), idx(0, 1, s)))
        faces.append((idx(1, 0, s), idx(1, 1, s), idx(1, 1, s + 1), idx(1, 0, s + 1)))
    # end caps at the two angles
    faces.append((idx(0, 0, 0), idx(0, 1, 0), idx(1, 1, 0), idx(1, 0, 0)))
    faces.append((idx(0, 0, segments), idx(1, 0, segments),
                  idx(1, 1, segments), idx(0, 1, segments)))
    return mesh_object(name, verts, faces)


def triangular_prism_y(name, y_center, thickness, p0, p1, p2):
    y0, y1 = y_center - thickness / 2, y_center + thickness / 2
    verts = []
    for y in (y0, y1):
        for x, z in (p0, p1, p2):
            verts.append((x, y, z))
    faces = [(0, 2, 1), (3, 4, 5),
             (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
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
    R = config['radius']
    h = config['head_depth']
    ys = config['shell_half_length']
    y_end = ys + h

    # Horizontal body: elliptical dished head + cylindrical shell + head.
    profile = [(0.0, -y_end)]
    for i in range(1, 33):
        theta = (math.pi / 2) * i / 32
        profile.append((R * math.sin(theta), -y_end + h * (1 - math.cos(theta))))
    for i in range(1, 33):
        profile.append((R, -ys + 2 * ys * i / 32))
    for i in range(1, 33):
        theta = (math.pi / 2) * i / 32
        profile.append((R * math.cos(theta), ys + h * math.sin(theta)))
    body = revolve_solid_y('Tank body', profile, config['radial_segments'])
    assign_tile(body, 0, boxed=True, smooth='all')

    # Raised weld bands at the two head/shell tangents.
    for name, y in (('Shell seam minus', -ys), ('Shell seam plus', ys)):
        ring = revolve_ring_y(name,
                              [(R - .002, y - .006), (R + .0035, y - .006),
                               (R + .0035, y + .006), (R - .002, y + .006)],
                              config['radial_segments'])
        assign_tile(ring, 1, boxed=True, smooth='all')

    # Two saddle supports: curved band, tapered leg, foot plate with two slots,
    # and two triangular gussets per support (gussets inferred from the sheet).
    slot_z0, slot_z1 = config['floor_z'] - .02, config['leg_bottom_z'] + .04
    for i, yc in enumerate(config['support_y'], start=1):
        a0, a1 = config['support_band_angles_deg']
        band = annular_sector_y(f'Support {i} saddle band',
                                yc - config['support_band_width_y'] / 2,
                                yc + config['support_band_width_y'] / 2,
                                config['support_band_radius_inner'],
                                config['support_band_radius_outer'],
                                a0, a1, 32)
        apply_bevel(band, .0015, 2)
        assign_tile(band, 0, boxed=True, smooth='all')

        leg = tapered_leg_y(f'Support {i} leg', yc,
                            config['leg_top_z'], config['leg_bottom_z'],
                            config['leg_top_width_y'], config['leg_bottom_width_y'],
                            config['leg_top_depth_x'], config['leg_bottom_depth_x'])
        apply_bevel(leg, .0020, 2)
        assign_tile(leg, 0, boxed=True, smooth='flat')

        foot_z = config['floor_z'] + config['foot_thickness'] / 2
        foot = box_at(f'Support {i} foot plate', (0, yc, foot_z),
                      (config['foot_size_x'] / 2, config['foot_size_y'] / 2,
                       config['foot_thickness'] / 2))
        for j, sy in enumerate(config['foot_slot_y'], start=1):
            cutter = capsule_prism(f'Support {i} slot cutter {j}', 0,
                                   config['foot_slot_half_length_x'],
                                   config['foot_slot_radius'],
                                   slot_z0, slot_z1, 0)
            cutter.location.y = yc + sy
            bpy.context.view_layer.update()
            boolean_cut(foot, cutter, f'Foot slot {i}-{j}')
        apply_bevel(foot, .0012, 2)
        assign_tile(foot, 1, boxed=True, smooth='flat')

        gx0 = config['gusset_x_inner']
        gx1 = config['gusset_x_outer']
        gz0 = config['leg_bottom_z']
        gz1 = config['gusset_z_top']
        for side, suffix in ((-1, 'minus'), (1, 'plus')):
            g = triangular_prism_y(f'Support {i} gusset {suffix}',
                                   yc + side * config['gusset_offset_y'],
                                   config['gusset_thickness'],
                                   (gx0, gz0), (gx1, gz0), (gx0, gz1))
            apply_bevel(g, .0010, 2)
            assign_tile(g, 0, boxed=True, smooth='flat')

    # Manway: weld collar, neck, flange, gasket, domed lid, handle, hinges and
    # six swing bolts with eye rings (count/arrangement partly inferred).
    my = config['manway_y']
    neck, _ = tube('Manway neck', (0, my, config['manway_neck_z0']),
                   (0, my, config['manway_neck_z1']),
                   config['manway_neck_outer'], config['manway_neck_inner'],
                   segments=96)
    assign_tile(neck, 0, boxed=True, smooth='all')
    neck_sleeve, _ = tube('Manway neck sleeve',
                          (0, my, config['manway_neck_z0'] + .001),
                          (0, my, config['manway_neck_z1'] - .001),
                          config['manway_neck_inner'] - .0004,
                          config['manway_neck_inner'] - .0015, 96)
    assign_special(neck_sleeve, 'dark', boxed=True, smooth='all')
    collar = torus('Manway neck weld collar', (0, my, R + .0005), (0, 0, 1),
                   .126, .006, 64, 8)
    assign_tile(collar, 1, boxed=True, smooth='all')

    flange = revolve_ring('Manway flange',
                          [(config['manway_neck_inner'], config['manway_flange_z0']),
                           (config['manway_flange_outer'], config['manway_flange_z0']),
                           (config['manway_flange_outer'], config['manway_flange_z1']),
                           (config['manway_neck_inner'], config['manway_flange_z1'])],
                          96)
    flange.location.y = my
    assign_tile(flange, 0, boxed=True, smooth='all')

    gasket = revolve_ring('Manway gasket',
                          [(.086, config['manway_gasket_z0']),
                           (.136, config['manway_gasket_z0']),
                           (.136, config['manway_gasket_z1']),
                           (.086, config['manway_gasket_z1'])], 96)
    gasket.location.y = my
    assign_special(gasket, 'gasket', boxed=True, smooth='all')

    z0 = config['manway_lid_z0']
    lid_profile = [(0.0, z0), (config['manway_lid_radius'], z0),
                   (config['manway_lid_radius'], z0 + .006),
                   (config['manway_lid_radius'] - .010, z0 + .018),
                   (config['manway_lid_radius'] - .035, z0 + .034),
                   (config['manway_lid_radius'] - .075, z0 + .046),
                   (0.0, config['manway_lid_z1'])]
    lid = revolve_solid('Manway lid', lid_profile, 96)
    lid.location.y = my
    assign_tile(lid, 0, boxed=True, smooth='all')

    boss = along('Manway handle boss', (0, my, config['manway_lid_z1'] - .001),
                 (0, my, config['manway_lid_z1'] + .006), .015, 32)
    assign_tile(boss, 1, boxed=True, smooth='side', axis=(0, 0, 1))
    ring = torus('Manway handle ring', (0, my, config['manway_lid_z1'] + .006),
                 (0, 0, 1), .013, .0032, 28, 8)
    assign_tile(ring, 1, boxed=True, smooth='all')

    for side, suffix in ((-1, 'minus'), (1, 'plus')):
        hinge = box_at(f'Manway hinge {suffix}',
                       (side * .052, my - .152, .318), (.019, .022, .013))
        apply_bevel(hinge, .0012, 2)
        assign_tile(hinge, 1, boxed=True, smooth='flat')
    pin = along('Manway hinge pin', (-.080, my - .152, .318),
                (.080, my - .152, .318), .0055, 24)
    assign_tile(pin, 1, boxed=True, smooth='side', axis=(1, 0, 0))

    for i, angle in enumerate(config['manway_clamp_angles_deg'], start=1):
        a = math.radians(angle)
        radial = Vector((math.cos(a), math.sin(a), 0))
        tangent = Vector((-math.sin(a), math.cos(a), 0))
        lug = box_world(f'Manway clamp {i} lug',
                        config['manway_clamp_lug_radius'], .310,
                        .016, .016, .014, angle)
        apply_bevel(lug, .0012, 2)
        lug.location.y = my
        assign_tile(lug, 1, boxed=True, smooth='flat')
        p0 = Vector((config['manway_clamp_bolt_radius'] * math.cos(a),
                     my + config['manway_clamp_bolt_radius'] * math.sin(a), .30))
        p1 = Vector((p0.x, p0.y, config['manway_clamp_bolt_top_z']))
        bolt = along(f'Manway clamp {i} bolt', p0, p1, .0050, 20)
        assign_tile(bolt, 1, boxed=True, smooth='side',
                    axis=(p1 - p0).normalized())
        eye = torus(f'Manway clamp {i} eye', p1, tangent, .0135, .0036, 28, 8)
        assign_tile(eye, 1, boxed=True, smooth='all')
        nut = along(f'Manway clamp {i} nut', (p0.x, p0.y, .332),
                    (p0.x, p0.y, .344), .0072, 6)
        assign_tile(nut, 1, boxed=True, smooth='side', axis=(0, 0, 1))

    # Three short top ferrules with rolled lips and weld collars.
    for spec in config['top_nozzles']:
        y = spec['y']
        hollow_pipe(spec['name'], (0, y, R - .02), (0, y, spec['top_z']),
                    spec['outer'], spec['bore'], 1, segments=48)
        lip = torus(spec['name'] + ' lip', (0, y, spec['top_z']), (0, 0, 1),
                    spec['outer'] + .002, .0042, 40, 8)
        assign_tile(lip, 1, boxed=True, smooth='all')
        collar = torus(spec['name'] + ' collar', (0, y, R + .0015), (0, 0, 1),
                       spec['outer'] + .006, .0045, 40, 8)
        assign_tile(collar, 1, boxed=True, smooth='all')

    # Open flanged ports on the +Y head.  The upper port is visible in the
    # photograph; the lower one is inferred from the six-view right/front panels.
    for spec in config['right_ports']:
        z = spec['z']
        hollow_pipe(spec['name'], (0, spec['y_start'], z), (0, spec['y_end'], z),
                    spec['outer'], spec['bore'], 1, segments=48)
        flange = tube(spec['name'] + ' flange', (0, spec['y_end'] - .018, z),
                      (0, spec['y_end'], z), spec['flange_outer'],
                      spec['bore'] + .002, segments=64)
        assign_tile(flange[0], 1, boxed=True, smooth='side', axis=(0, 1, 0))
        collar = torus(spec['name'] + ' collar', (0, .44, z), (0, 1, 0),
                       spec['outer'] + .006, .0045, 36, 8)
        assign_tile(collar, 1, boxed=True, smooth='all')
        for b in range(config['right_port_bolt_count']):
            ba = TAU * b / config['right_port_bolt_count'] + .26
            bx = math.cos(ba) * config['right_port_bolt_radius']
            bz = z + math.sin(ba) * config['right_port_bolt_radius']
            stud = along(f'{spec["name"]} stud {b + 1}',
                         (bx, spec['y_end'], bz), (bx, spec['y_end'] + .010, bz),
                         .0048, 6)
            assign_tile(stud, 1, boxed=True, smooth='side', axis=(0, 1, 0))

    # Small bottom drain near the +Y end (the six-view bottom panel instead
    # shows a central flanged port; the photographed unit is followed here).
    dy = config['drain_y']
    hollow_pipe('Drain pipe', (0, dy, config['drain_z_top']),
                (0, dy, config['drain_z_bottom']),
                config['drain_outer'], config['drain_bore'], 1, segments=48)
    lip = torus('Drain lip', (0, dy, config['drain_z_bottom']), (0, 0, 1),
                config['drain_outer'] + .002, .0042, 36, 8)
    assign_tile(lip, 1, boxed=True, smooth='all')
    collar = torus('Drain collar', (0, dy, -R + .002), (0, 0, 1),
                   config['drain_outer'] + .007, .0045, 36, 8)
    assign_tile(collar, 1, boxed=True, smooth='all')


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
# Centre the exported model vertically at z=0 (the x/y centre is already
# near zero; small asymmetry from the manway, ports and drain is preserved).
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
        ('manway', (0, config['manway_y'], .36 + Z_SHIFT), (1.2, -0.9, 1.0), .40),
        ('clamps', (.09, config['manway_y'] + .09, .35 + Z_SHIFT), (1.4, -0.7, .7), .37),
        ('top-nozzles', (0, -.12, .30 + Z_SHIFT), (1.0, -0.7, 1.4), .55),
        ('right-ports', (0, .57, .0 + Z_SHIFT), (0.8, 1.0, .30), .55),
        ('right-head', (0, .57, .0 + Z_SHIFT), (0.1, 1.0, .16), .85),
        ('support', (0, .255, -.35 + Z_SHIFT), (1.4, -0.8, .35), .46),
        ('foot', (0, .255, -.40 + Z_SHIFT), (1.2, -1.0, .65), .30),
        ('drain', (0, config['drain_y'], -.30 + Z_SHIFT), (0.8, -0.7, -0.65), .26),
        ('seam', (0, .40, -.10 + Z_SHIFT), (1.2, -0.8, .45), .32),
        ('surface', (.16, -.10, .02 + Z_SHIFT), (1.6, -0.9, .40), .38),
    )
    for name, target, direction, scale in detail_cameras:
        cam = camera(scene, direction, Vector(target), scale)
        draw(scene, render_dir / f'detail-{name}.png', resolution=768, samples=24)
        bpy.data.objects.remove(cam, do_unlink=True)

    groups = (
        ('top-nozzles-only', lambda n: n.startswith('Top nozzle'), (0, 0, 1),
         (0, -.12, .30 + Z_SHIFT), .62),
        ('right-ports-only', lambda n: n.startswith('Right port'), (0, 1, .15),
         (0, .57, .0 + Z_SHIFT), .48),
        ('manway-only', lambda n: n.startswith('Manway neck')
         or n.startswith('Manway flange'), (0, 0, 1),
         (0, config['manway_y'], .31 + Z_SHIFT), .36),
        ('drain-only', lambda n: n.startswith('Drain'), (0, 0, -1),
         (0, config['drain_y'], -.30 + Z_SHIFT), .24),
        ('feet-only', lambda n: n.startswith('Support') and 'foot plate' in n,
         (0, 0, -1), (0, .0, -.41 + Z_SHIFT), .95),
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
        'isolated_checks': {'top_nozzles_expected': 3, 'right_ports_expected': 2,
                            'manway_expected': 1, 'drain_expected': 1,
                            'foot_slots_expected': 4}}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
