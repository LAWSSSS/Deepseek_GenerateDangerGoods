"""Blender worker: part-based rebuild of the six-cylinder gas-bottle group (19).

Coordinates follow the existing library: the row runs along +Y, the source
camera is on +X, and +Z is up.  The model is built with the measured photo
silhouette proportions and then centred at the origin.  Every visible part is
separate; the two green neck guards carry a real oval through-hole.  Painted
steel is deliberately a dielectric coating (metallic 0), while the brass valve,
regulator bodies, gauge bezels and outlets are metallic.  No lighting or
shadow is baked into the texture.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree
from mathutils import kdtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf8'))
out = Path(request['out'])
cfg = request['config']
TEX = request['textures']
TILE = cfg['texture']['tiles']
objects = []
tiles = {}
metal_objects = []
guard_objects = []
body_objects = []
clay = None
CAM = None


# ---------------------------------------------------------------------------
# Materials and UV helpers
# ---------------------------------------------------------------------------
def atlas_material():
    material = bpy.data.materials.new('Observed paint, rust, brass and steel PBR atlas')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    principled.inputs['IOR'].default_value = 1.45

    def texture(name, noncolor=False):
        node = nodes.new('ShaderNodeTexImage')
        node.image = bpy.data.images.load(TEX[name], check_existing=True)
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
    normal_map.inputs['Strength'].default_value = .38
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    return material


# created after legacy.setup_scene(), which resets the file and invalidates any
# material created before it
MATERIAL = None


def tile_bounds(idx):
    # The atlas PNG has tile row 0 at the bottom of the image, and UV v=0 maps
    # to the image bottom.  Keep the same row order here so a tile name always
    # samples the matching painted tile.
    col, row = idx % 4, idx // 4
    return (col * .25, row * .25, (col + 1) * .25, (row + 1) * .25)


def remap_uv(obj, tile, *, v_lo=.01, v_hi=.99, u_lo=.01, u_hi=.99, margin=.012):
    """Map the object's 0..1 UVs into one atlas tile.

    v_lo/v_hi select a band inside the tile; bodies use the full tile so the
    baked bottom rust band follows the physical bottom, while guards and caps
    use the upper, cleaner band.
    """
    x0, y0, x1, y1 = tile_bounds(tile)
    w, h = x1 - x0, y1 - y0
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    for loop in uv:
        u, v = loop.uv
        u = u_lo + (u_hi - u_lo) * u
        v = v_lo + (v_hi - v_lo) * v
        loop.uv = (x0 + margin * w + u * (w - 2 * margin * w),
                   y0 + margin * h + v * (h - 2 * margin * h))


def assign(obj, tile, *, mode='default', v_lo=.01, v_hi=.99, u_lo=.01, u_hi=.99,
           margin=.012, register=True):
    obj.data.materials.clear()
    obj.data.materials.append(MATERIAL)
    tiles[obj.name] = tile
    if mode == 'constant':
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name='Atlas')
        x0, y0, x1, y1 = tile_bounds(tile)
        center = (x0 + .5 * (x1 - x0), y0 + .5 * (y1 - y0))
        for loop in obj.data.uv_layers.active.data:
            loop.uv = center
    else:
        remap_uv(obj, tile, v_lo=v_lo, v_hi=v_hi, u_lo=u_lo, u_hi=u_hi, margin=margin)
    if register:
        objects.append(obj)
        if tile in (TILE['brass'], TILE['chrome']):
            metal_objects.append(obj.name)
    return obj


def apply_bevel(obj, width, segments=2, angle=40):
    if width <= 0:
        return
    mod = obj.modifiers.new('Small manufactured edge radius', 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(angle)
    mod.use_clamp_overlap = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def finish(obj, name, tile, *, bevel=.0012, smooth=True, constant=False,
           v_lo=.01, v_hi=.99, register=True):
    obj.name = name
    obj.data.name = name
    if obj.type != 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH')
        obj = bpy.context.object
    apply_bevel(obj, bevel)
    for poly in obj.data.polygons:
        poly.use_smooth = smooth
    return assign(obj, tile, mode='constant' if constant else 'default',
                  v_lo=v_lo, v_hi=v_hi, register=register)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
def mesh_object(name, verts, faces, loop_uvs=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if loop_uvs is not None:
        layer = mesh.uv_layers.new(name='Atlas')
        for poly, face_uvs in zip(mesh.polygons, loop_uvs):
            for li, uv in zip(poly.loop_indices, face_uvs):
                layer.data[li].uv = uv
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def revolve(name, profile, tile, *, x=0.0, y=0.0, segs=None, v_lo=.01, v_hi=.99,
            bevel=.0, smooth=True):
    """Closed solid of revolution from a bottom-to-top (radius, z) profile."""
    segs = segs or cfg['radial_segments']
    zmin, zmax = profile[0][1], profile[-1][1]
    verts, ring_start = [], []
    for r, z in profile:
        ring_start.append(len(verts))
        for j in range(segs):
            a = math.tau * j / segs
            verts.append((x + r * math.cos(a), y + r * math.sin(a), z))
    faces, loop_uvs = [], []
    vnorm = lambda z: (z - zmin) / max(1e-9, zmax - zmin)

    def side_uvs(i, j, i2, j2):
        u0, u1 = j / segs, (j + 1) / segs
        return [(u0, vnorm(profile[i][1])), (u1, vnorm(profile[i][1])),
                (u1, vnorm(profile[i2][1])), (u0, vnorm(profile[i2][1]))]

    for i in range(len(profile) - 1):
        for j in range(segs):
            j2 = (j + 1) % segs
            a, b = ring_start[i] + j, ring_start[i] + j2
            c, d = ring_start[i + 1] + j2, ring_start[i + 1] + j
            faces.append((a, b, c, d))
            loop_uvs.append(side_uvs(i, j, i + 1, j2))
    # centre-cap vertices keep the solid closed and the caps flat
    bottom_center = len(verts)
    verts.append((x, y, profile[0][1]))
    top_center = len(verts)
    verts.append((x, y, profile[-1][1]))
    for j in range(segs):
        j2 = (j + 1) % segs
        a, b = ring_start[0] + j, ring_start[0] + j2
        faces.append((bottom_center, b, a))
        loop_uvs.append([(.5, vnorm(profile[0][1])), ((j + 1) / segs, vnorm(profile[0][1])),
                         (j / segs, vnorm(profile[0][1]))])
        c, d = ring_start[-1] + j, ring_start[-1] + j2
        faces.append((top_center, c, d))
        loop_uvs.append([(.5, vnorm(profile[-1][1])), (j / segs, vnorm(profile[-1][1])),
                         ((j + 1) / segs, vnorm(profile[-1][1]))])
    obj = mesh_object(name, verts, faces, loop_uvs)
    for poly in obj.data.polygons:
        poly.use_smooth = smooth
    for poly in obj.data.polygons:
        if len(poly.vertices) == 3:
            poly.use_smooth = False
    apply_bevel(obj, bevel)
    return assign(obj, tile, v_lo=v_lo, v_hi=v_hi)


def bezier(p0, p1, p2, p3, n):
    points = []
    for i in range(n + 1):
        t = i / n
        s = 1 - t
        points.append(tuple(s * s * s * a + 3 * s * s * t * b + 3 * s * t * t * c +
                            t * t * t * d for a, b, c, d in zip(p0, p1, p2, p3)))
    return points


def cylinder(name, radius, z0, z1, tile, *, x=0.0, y=0.0, segs=64, bevel=.001,
             constant=False):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius, depth=z1 - z0,
                                        location=(x, y, (z0 + z1) / 2))
    return finish(bpy.context.object, name, tile, bevel=bevel, constant=constant)


def along(name, start, end, radius, tile, *, segs=32, bevel=.001, constant=False):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=radius,
                                        depth=(end - start).length,
                                        location=(start + end) / 2)
    obj = bpy.context.object
    obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, constant=constant)


def torus(name, major, minor, location, axis, tile, *, major_segments=96, minor_segments=16):
    rotations = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0)}
    bpy.ops.mesh.primitive_torus_add(major_segments=major_segments,
                                     minor_segments=minor_segments,
                                     location=location, rotation=rotations[axis],
                                     major_radius=major, minor_radius=minor)
    obj = bpy.context.object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=0, constant=True)


def box(name, center, size, tile, *, bevel=.001, constant=False):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.scale = size
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, tile, bevel=bevel, constant=constant)


def scalloped_disc(name, radius, thickness, lobes, tile, center, axis='Z',
                   *, segs=96, amp_factor=.13, bevel=0.0):
    """A flower/star hand-wheel with rounded lobes, built as a closed solid."""
    amp = radius * amp_factor
    mid = radius - amp
    z0, z1 = -thickness / 2, thickness / 2
    verts, loop_uvs, faces = [], [], []
    ring_top, ring_bot = [], []
    for z in (z1, z0):
        start = len(verts)
        for j in range(segs):
            a = math.tau * j / segs
            r = mid + amp * math.cos(lobes * a)
            verts.append((r * math.cos(a), r * math.sin(a), z))
        if z == z1:
            ring_top = list(range(start, start + segs))
        else:
            ring_bot = list(range(start, start + segs))
    top_center = len(verts)
    verts.append((0, 0, z1))
    bot_center = len(verts)
    verts.append((0, 0, z0))
    for j in range(segs):
        j2 = (j + 1) % segs
        a, b = ring_top[j], ring_top[j2]
        c, d = ring_bot[j2], ring_bot[j]
        faces.append((a, b, c, d))
        loop_uvs.append([(j / segs, 1), ((j + 1) / segs, 1),
                         ((j + 1) / segs, 0), (j / segs, 0)])
        faces.append((top_center, a, b))
        loop_uvs.append([(.5, .5), (j / segs, 1), ((j + 1) / segs, 1)])
        faces.append((bot_center, d, c))
        loop_uvs.append([(.5, .5), (j / segs, 0), ((j + 1) / segs, 0)])
    obj = mesh_object(name, verts, faces, loop_uvs)
    obj.rotation_euler = {'Z': (0, 0, 0), 'X': (0, math.pi / 2, 0),
                          'Y': (math.pi / 2, 0, 0)}[axis]
    obj.location = center
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return finish(obj, name, tile, bevel=bevel, constant=True)


def cut_oval(obj, y0, z0, semi_y, semi_z, depth=.24):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1, depth=depth,
                                        location=(0, y0, z0),
                                        rotation=(0, math.pi / 2, 0))
    cutter = bpy.context.object
    cutter.scale = (semi_z, semi_y, 1)
    bpy.context.view_layer.objects.active = cutter
    cutter.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # construction helper: keep it out of the exported selection and hidden
    assign(cutter, TILE['dark'], mode='constant', register=False)
    mod = obj.modifiers.new('Oval lifting hole', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    mod.material_mode = 'TRANSFER'
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    cutter.hide_render = True
    cutter.hide_viewport = True
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


# ---------------------------------------------------------------------------
# Cylinder parts
# ---------------------------------------------------------------------------
def body_profile(c):
    r = c['radius']
    bh = r * cfg['bottom']['height_factor']
    r0 = r * cfg['bottom']['flat_radius_factor']
    profile = [(r0, 0.0)]
    profile += bezier((r0, 0.0), (r0 + .78 * (r - r0), 0.0),
                      (r, .12 * bh), (r, bh), 8)[1:]
    profile.append((r, c['body_top']))
    shoulder_h = c['neck_top'] - c['body_top']
    c1, c2 = cfg['shoulder']['control_1'], cfg['shoulder']['control_2']
    profile += bezier((r, c['body_top']),
                      (r, c['body_top'] + c1 * shoulder_h),
                      (c['neck_r'], c['neck_top'] - c2 * shoulder_h),
                      (c['neck_r'], c['neck_top']),
                      cfg['shoulder']['samples'])[1:]
    return profile


def build_body(c):
    obj = revolve(f"{c['label']} 筒身", body_profile(c), c['tile'],
                  x=0, y=c['y'], bevel=cfg['bevel_m'])
    body_objects.append(obj.name)
    torus(f"{c['label']} 底部卷边", c['radius'] * .965, .0018,
          (0, c['y'], .004), 'Z', TILE['dark'], major_segments=96, minor_segments=8)
    return obj


def build_cap(c):
    cap = c['cap']
    r = cap['radius']
    z0, z1 = cap['base_z'], cap['top_z']
    profile = [(cap['waist_r'], z0), (cap['waist_r'] + .0012, z0 + .002),
               (r, z0 + .006), (r + .0004, z0 + .014), (r, z0 + .026),
               (r, z0 + .050), (r * .985, z0 + .062), (r * .93, z0 + .072),
               (r * .81, z0 + .081), (r * .60, z0 + .089), (r * .33, z0 + .095),
               (r * .12, z0 + .098), (.0025, z1)]
    revolve(f"{c['label']} 阀帽", profile, c['tile'], x=0, y=c['y'],
            bevel=0.0, v_lo=.25, v_hi=.98)
    torus(f"{c['label']} 阀帽底圈", cap['waist_r'] + .0006, .0016,
          (0, c['y'], z0 + .004), 'Z', c['tile'], major_segments=72,
          minor_segments=10)


def build_guard(c):
    g = c['guard']
    base_r, body_r, top = g['base_r'], g['body_r'], g['top_z']
    z0 = g['base_z']
    s = (top - z0) / .166
    profile = [(base_r * .96, z0), (base_r * 1.02, z0 + .006 * s),
               (base_r * 1.06, z0 + .012 * s),
               (body_r * .985, z0 + .022 * s),
               (body_r, z0 + .040 * s),
               (body_r * 1.004, z0 + .100 * s),
               (body_r, z0 + .120 * s),
               (body_r * .975, z0 + .136 * s),
               (body_r * .90, z0 + .148 * s),
               (body_r * .77, z0 + .157 * s),
               (body_r * .56, z0 + .164 * s),
               (body_r * .30, z0 + .170 * s),
               (body_r * .09, z0 + .173 * s),
               (.0025, top)]
    obj = revolve(f"{c['label']} 护圈", profile, c['tile'], x=0, y=c['y'],
                  bevel=0.0, v_lo=.25, v_hi=.98)
    guard_objects.append(obj.name)
    # the photographed oval is offset in the guard's local Y; add the guard's
    # row position so the cutter actually passes through the right guard
    cut_oval(obj, c['y'] + g['hole']['y'], g['hole']['z'],
             g['hole']['semi_y'], g['hole']['semi_z'])
    torus(f"{c['label']} 护圈底圈", base_r * .99, .0022,
          (0, c['y'], z0 + .004), 'Z', c['tile'], major_segments=96,
          minor_segments=12)
    # Inferred hidden valve stem, visible only through the oval hand-hole.  It
    # keeps the opening from reading as an empty hollow tube; its shape and
    # presence are not measured.
    cylinder(f"{c['label']} 护圈内阀杆", .018, c['neck_top'] - .012, top - .022,
             TILE['dark'], x=0, y=c['y'] + g['hole']['y'] * .36,
             bevel=.0008, constant=True)
    cylinder(f"{c['label']} 护圈内阀杆黄铜肩", .0205, g['hole']['z'] - .030,
             g['hole']['z'] - .016, TILE['brass'], x=0,
             y=c['y'] + g['hole']['y'] * .36, bevel=.0006, constant=True)
    lab = g['label']
    x_face = math.sqrt(max(0.0, body_r * body_r - lab['y'] * lab['y'])) + .001
    box(f"{c['label']} 白色检验标牌", (x_face, c['y'] + lab['y'], lab['z']),
        (.0016, lab['width_y'], lab['height_z']), TILE['label_white'],
        bevel=.0004, constant=False)
    return obj


def blue1_valve(c):
    v = c['valve']
    y = c['y']
    brass = TILE['brass']
    chrome = TILE['chrome']
    plastic = TILE['plastic_blue']
    torus(f"{c['label']} 阀座钢圈", c['neck_r'] + .0025, .0016,
          (0, y, c['neck_top'] + .001), 'Z', chrome, major_segments=72,
          minor_segments=10)
    cylinder(f"{c['label']} 黄铜阀体", v['body_r'], v['body_z0'], v['body_z1'],
             brass, x=0, y=y, bevel=.0012, constant=True)
    cylinder(f"{c['label']} 黄铜六角锁母", v['hex_r'], v['hex_z0'], v['hex_z1'],
             brass, x=0, y=y, segs=6, bevel=.0007, constant=True)
    b = v['block']
    box(f"{c['label']} 黄铜阀块", (b['center_x'], y + b['center_y'], b['center_z']),
        (b['size_x'], b['size_y'], b['size_z']), brass, bevel=.0025, constant=True)
    box(f"{c['label']} 阀块正面铭牌", (b['size_x'] / 2 + .0004, y, b['center_z']),
        (.0014, .022, .016), brass, bevel=.0004, constant=True)
    for i in (-1, 0, 1):
        box(f"{c['label']} 阀块凸棱 {i+2}", (b['size_x'] / 2 + .0012, y + i * .0045,
                                          b['center_z'] + .002), (.0006, .0024, .0012),
            TILE['dark'], bevel=.00015, constant=True)
    o = v['outlet']
    along(f"{c['label']} 黄铜出气口", (0, y + o['y0'], o['z']),
          (0, y + o['y1'], o['z']), o['radius'], brass, segs=48, constant=True)
    along(f"{c['label']} 出气口六角螺母", (0, y + o['nut_y0'], o['z']),
          (0, y + o['nut_y1'], o['z']), o['nut_r'], brass, segs=6,
          bevel=.0006, constant=True)
    along(f"{c['label']} 出气口螺纹嘴", (0, y + o['nipple_y0'], o['z']),
          (0, y + o['nipple_y1'], o['z']), o['nipple_r'], chrome, segs=32,
          constant=True)
    hw = v['handwheel']
    scalloped_disc(f"{c['label']} 蓝色手轮", hw['radius'],
                   hw['z1'] - hw['z0'], hw['lobes'], plastic,
                   (0, y, (hw['z0'] + hw['z1']) / 2), 'Z', amp_factor=.125)
    cylinder(f"{c['label']} 手轮轮毂", hw['hub_r'], hw['z0'] - .002,
             hw['z1'] + .005, plastic, x=0, y=y, bevel=.0008, constant=True)
    cylinder(f"{c['label']} 手轮铭牌螺钉", .0065, hw['z1'] + .004,
             hw['z1'] + .008, chrome, x=0, y=y, segs=24, bevel=.0005, constant=True)
    along(f"{c['label']} 阀杆", (0, y, hw['z0'] - .004), (0, y, v['body_z1']),
          .0085, brass, segs=32, constant=True)


def regulator(c):
    v = c['regulator']
    y = c['y']
    zc = v['center_z']
    chrome = TILE['chrome']
    brass = TILE['brass']
    plastic = TILE['plastic_blue']
    along(f"{c['label']} 减压器进气杆", (0, y, v['stem_z0']), (0, y, v['stem_z1']),
          v['stem_r'], chrome, segs=40, constant=True)
    along(f"{c['label']} 减压器进气六角", (0, y, v['stem_z1'] - .020),
          (0, y, v['stem_z1'] - .010), v['stem_r'] + .0035, brass, segs=6,
          bevel=.0005, constant=True)
    along(f"{c['label']} 减压器主体", (v['body_x0'], y, zc),
          (v['body_x1'], y, zc), v['body_r'], chrome, segs=96,
          bevel=.0025, constant=True)
    torus(f"{c['label']} 减压器前缘", v['body_r'] - .002, .0018,
          (v['body_x1'] - .0015, y, zc), 'X', chrome, major_segments=96,
          minor_segments=12)
    along(f"{c['label']} 减压器后盖", (v['body_x0'] - .004, y, zc),
          (v['body_x0'] + .001, y, zc), v['body_r'] * .93, chrome, segs=64,
          bevel=.0015, constant=True)
    g = v['gauge']
    along(f"{c['label']} 压力表表圈", (g['front_x'] - .0035, y, g['center_z']),
          (g['front_x'] - .0010, y, g['center_z']), g['bezel_major'], chrome,
          segs=96, bevel=.0012, constant=True)
    torus(f"{c['label']} 压力表滚边", g['bezel_major'] - .001, g['bezel_minor'],
          (g['front_x'] - .0008, y, g['center_z']), 'X', chrome,
          major_segments=96, minor_segments=14)
    # the dial is a thin cylinder rotated so its face points to +X
    dial = cylinder(f"{c['label']} 压力表表盘", g['radius'], -.0006, .0006,
                    TILE['gauge'], x=0, y=0, segs=96, bevel=0)
    dial.rotation_euler = (0, math.pi / 2, 0)
    dial.location = (g['front_x'], y, g['center_z'])
    bpy.context.view_layer.objects.active = dial
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    a1 = math.radians(58)
    r_needle = g['radius'] * .78
    n0 = (g['front_x'] + .0012, y - .001, g['center_z'])
    n1 = (g['front_x'] + .0012, y - .001 + r_needle * math.sin(a1),
          g['center_z'] + r_needle * math.cos(a1))
    along(f"{c['label']} 压力表指针", n0, n1, .00065, TILE['red'], segs=12,
          bevel=0, constant=True)
    cylinder(f"{c['label']} 压力表中心轴", .0022, g['front_x'] + .0004,
             g['front_x'] + .0020, chrome, x=0, y=y, segs=24,
             bevel=.0003, constant=True)
    t = v['top_knob']
    scalloped_disc(f"{c['label']} 顶部蓝色旋钮", t['radius'], t['thickness'],
                   t['lobes'], plastic, (0, y, t['center_z']), 'Z',
                   amp_factor=.14)
    cylinder(f"{c['label']} 顶部旋钮轮毂", t['hub_r'], t['center_z'] + .004,
             t['center_z'] + .010, plastic, x=0, y=y, bevel=.0006, constant=True)
    cylinder(f"{c['label']} 顶部旋钮螺钉", t['hub_r'] * .52,
             t['center_z'] + .010, t['center_z'] + .013, chrome, x=0, y=y,
             segs=24, bevel=.0003, constant=True)
    lk = v['left_knob']
    scalloped_disc(f"{c['label']} 侧向蓝色旋钮", lk['radius'], lk['thickness'],
                   lk['lobes'], plastic, (0, y + lk['center_y'], lk['center_z']),
                   'Y', amp_factor=.15)
    along(f"{c['label']} 侧向旋钮轴", (0, y + lk['center_y'] + .004, lk['center_z']),
          (0, y + lk['center_y'] + .014, lk['center_z']), lk['hub_r'], plastic,
          segs=24, constant=True)
    o = v['outlet']
    along(f"{c['label']} 减压器出气嘴", (0, y + o['y0'], o['z']),
          (0, y + o['y1'], o['z']), o['radius'], chrome, segs=40,
          constant=True)
    along(f"{c['label']} 出气嘴六角", (0, y + o['nut_y0'], o['z']),
          (0, y + o['nut_y1'], o['z']), o['nut_r'], brass, segs=6,
          bevel=.0005, constant=True)
    torus(f"{c['label']} 进气密封圈", v['stem_r'] + .0012, .0011,
          (0, y, v['stem_z1'] - .002), 'Z', TILE['dark'],
          major_segments=64, minor_segments=8)


def build_cylinder(c):
    build_body(c)
    if c['top'] == 'cap':
        build_cap(c)
    elif c['top'] == 'guard':
        build_guard(c)
    elif c['top'] == 'handwheel':
        blue1_valve(c)
    elif c['top'] == 'regulator':
        regulator(c)


# ---------------------------------------------------------------------------
# Scene, camera, export and renders
# ---------------------------------------------------------------------------
def init_camera(scene):
    data = bpy.data.cameras.new('review camera')
    cam = bpy.data.objects.new('review camera', data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_camera(cam, eye, target, *, ortho_scale=None, fov=40, shift_x=0, shift_y=0,
               top_orientation=False):
    cam.location = Vector(eye)
    cam.rotation_mode = 'QUATERNION'
    quat = (Vector(target) - cam.location).to_track_quat('-Z', 'Y')
    if top_orientation:
        quat = quat @ Quaternion((0, 0, 1), math.pi / 2)
    cam.rotation_quaternion = quat
    cam.data.type = 'ORTHO' if ortho_scale else 'PERSP'
    if ortho_scale:
        cam.data.ortho_scale = ortho_scale
    else:
        cam.data.angle = math.radians(fov)
    cam.data.shift_x = shift_x
    cam.data.shift_y = shift_y


def look(scene, direction, center, span, *, ortho_scale=None):
    direction = Vector(direction).normalized()
    set_camera(CAM, Vector(center) + direction * span * 5, center,
               ortho_scale=ortho_scale or span * 1.27,
               top_orientation=abs(direction.z) > .999)
    return CAM


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo + hi) / 2, max(hi - lo)


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


def draw(scene, path, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def set_visible(names):
    for obj in objects:
        obj.hide_render = obj.name not in names


def body_gaps():
    by_name = {obj.name: obj for obj in objects}
    gaps = []
    for left, right in zip(body_objects, body_objects[1:]):
        a = [tuple(v) for v in legacy.world_vertices(by_name[left])]
        b = [tuple(v) for v in legacy.world_vertices(by_name[right])]
        tree = kdtree.KDTree(len(a))
        for i, point in enumerate(a):
            tree.insert(point, i)
        tree.balance()
        nearest = min(tree.find(point)[2] for point in b)
        gaps.append({'pair': [left, right], 'min_gap_m': round(float(nearest), 6)})
    return gaps


scene = legacy.setup_scene('CPU', 512)
MATERIAL = atlas_material()
CAM = init_camera(scene)
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .35
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value = (1, 1, 1, 1)
# reuse the default light as the fill; add a main softbox and a front reflector
fill = bpy.data.objects.get('soft light')
if fill is not None:
    fill.data.energy = 95
    fill.data.shape = 'RECTANGLE'
    fill.data.size = 3.0
    fill.data.size_y = 3.0
    fill.location = (-3.0, 2.5, 2.0)
    fill.rotation_euler = (-fill.location).to_track_quat('-Z', 'Y').to_euler()
for name, location, energy, size in (
        ('main softbox', (2.4, -2.8, 3.8), 170, 3.0),
        ('front reflection', (3.0, 1.4, 1.0), 95, 1.4)):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'RECTANGLE'
    data.size = size
    data.size_y = size
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()

for c in cfg['cylinders']:
    build_cylinder(c)

bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
shift = -center
for obj in objects:
    obj.location += shift
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)

bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.export_scene.gltf(filepath=str(out / 'mesh.glb'), export_format='GLB',
                          use_selection=True, export_yup=False, export_apply=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'model.blend'))

parts = [{'name': obj.name, 'vertices': len(obj.data.vertices),
          'faces': len(obj.data.polygons), 'tile': tiles.get(obj.name)} for obj in objects]
intersections = {obj.name: intersecting_faces(obj) for obj in objects}
render_dir = out / 'renders'
render_dir.mkdir(exist_ok=True)
directions = {'front': (1, 0, .24), 'back': (-1, 0, .24),
              'left': (0, 1, .24), 'right': (0, -1, .24),
              'top': (0, 0, 1), 'bottom': (0, 0, -1)}
rr = cfg['render']
for name, direction in directions.items():
    look(scene, direction, center, span)
    draw(scene, render_dir / f'new-{name}.png', rr['six_view_resolution'],
         rr['six_view_samples'])

source = request['source_camera']
set_camera(CAM, (1.9, 0, 0), (0, 0, 0), fov=source['fov_deg'],
           shift_x=source.get('shift_x', 0), shift_y=source.get('shift_y', 0))
draw(scene, render_dir / 'new-source.png', 512, rr['six_view_samples'])

zc = shift.z
yd = {c['id']: c['y'] + shift.y for c in cfg['cylinders']}
zd = lambda z: z + zc
detail_specs = [
    ('red-cap', (2.2, -2.8, 2.2), (0, yd['red'], zd(.965)), .30),
    ('green1-hole', (1.6, -1.0, .70), (0, yd['green1'], zd(.920)), .24),
    ('green2-hole', (1.6, -1.0, .70), (0, yd['green2'], zd(.786)), .23),
    ('blue1-valve', (2.2, -2.6, 1.8), (0, yd['blue1'], zd(.660)), .31),
    ('blue2-gauge', (2.2, -2.2, 1.1), (0, yd['blue2'], zd(.548)), .27),
    ('blue3-gauge', (2.2, -2.2, 1.0), (0, yd['blue3'], zd(.410)), .24),
    ('bottom-rust', (2.0, -2.8, -.8), (0, yd['red'], zd(.045)), .34),
    ('row-gaps', (1.2, -1.2, 1.6), (0, 0, zd(.30)), .85),
    ('labels', (2.0, -1.2, .8), (0, yd['green1'], zd(.875)), .20),
]
guard_target = (0, (yd['green1'] + yd['green2']) / 2, zd(.885))
set_visible(set(guard_objects))
for name, direction in (('front', (1, 0, .10)), ('back', (-1, 0, .10))):
    look(scene, direction, Vector(guard_target), span, ortho_scale=.62)
    draw(scene, render_dir / f'guards-only-{name}.png', rr['detail_resolution'],
         rr['detail_samples'])
set_visible(set(body_objects))
look(scene, (0, 0, 1), center, span, ortho_scale=.95)
draw(scene, render_dir / 'bodies-only-top.png', rr['detail_resolution'],
     rr['detail_samples'])
# metal-only focus on the blue-1 valve, not the whole row; the fill and front
# lights are switched off so the moving main light produces a clear highlight
metal_focus = [name for name in metal_objects if name.startswith('蓝色气瓶一')]
set_visible(set(metal_focus))
fill_light = bpy.data.objects.get('soft light')
front_light = bpy.data.objects.get('front reflection')
for light in (fill_light, front_light):
    if light is not None:
        light.hide_render = True
metal_target = (0, yd['blue1'], zd(.655))
look(scene, (2.4, -2.2, 1.4), Vector(metal_target), span, ortho_scale=.22)
draw(scene, render_dir / 'metal-only-light-a.png', 512, rr['detail_samples'])
main_light_for_metal = bpy.data.objects['main softbox']
main_light_for_metal.location = (-3.0, 4.0, 5.0)
main_light_for_metal.rotation_euler = (-main_light_for_metal.location).to_track_quat('-Z', 'Y').to_euler()
look(scene, (2.4, -2.2, 1.4), Vector(metal_target), span, ortho_scale=.22)
draw(scene, render_dir / 'metal-only-light-b.png', 512, rr['detail_samples'])
main_light_for_metal.location = (2.4, -2.8, 3.8)
main_light_for_metal.rotation_euler = (-main_light_for_metal.location).to_track_quat('-Z', 'Y').to_euler()
for light in (fill_light, front_light):
    if light is not None:
        light.hide_render = False
set_visible(set(obj.name for obj in objects))

if not request.get('draft'):
    for name, direction, target, scale in detail_specs:
        look(scene, direction, Vector(target), span, ortho_scale=scale)
        draw(scene, render_dir / f'detail-{name}.png', rr['detail_resolution'],
             rr['detail_samples'])

    for i in range(24):
        a = math.tau * i / 24
        look(scene, (math.cos(a), math.sin(a), .24), center, span)
        draw(scene, render_dir / f'turn-{i:02d}.png', rr['turntable_resolution'],
             rr['turntable_samples'])

    look(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-material-front.png', rr['six_view_resolution'],
         rr['six_view_samples'])
    clay = bpy.data.materials.new('Clay')
    clay.use_nodes = True
    clay_principled = clay.node_tree.nodes.get('Principled BSDF')
    clay_principled.inputs['Base Color'].default_value = (.55, .56, .54, 1)
    clay_principled.inputs['Roughness'].default_value = .82
    clay_principled.inputs['Metallic'].default_value = 0
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
    look(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-clay-front.png', rr['six_view_resolution'],
         rr['six_view_samples'])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(MATERIAL)

    main_light = bpy.data.objects['main softbox']
    main_light.location = (-3.0, 4.0, 5.0)
    main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()
    look(scene, directions['front'], center, span)
    draw(scene, render_dir / 'new-light-alternate.png', rr['six_view_resolution'],
         rr['six_view_samples'])
    main_light.location = (2.4, -2.8, 3.8)
    main_light.rotation_euler = (-main_light.location).to_track_quat('-Z', 'Y').to_euler()

for obj in objects:
    obj.hide_render = True
old = legacy.load_mesh(request['original_mesh'])
_, _, oldcenter, oldspan = bounds([old])
for name, direction in directions.items():
    look(scene, direction, oldcenter, oldspan)
    draw(scene, render_dir / f'old-{name}.png', rr['six_view_resolution'],
         rr['six_view_samples'])
old.hide_render = True

data = {
    'parts': parts,
    'bounds': [list(low), list(high)],
    'centre': list(center),
    'span': span,
    'polygons': sum(len(obj.data.polygons) for obj in objects),
    'triangles': sum(max(0, len(p.vertices) - 2) for obj in objects for p in obj.data.polygons),
    'texture_size': cfg['texture']['size'],
    'self_intersecting_components': {k: v for k, v in intersections.items() if v},
    'intersection_check': 'BVH 同部件不共享顶点的面交叠；零件之间故意装配相交，未判定',
    'body_objects': body_objects,
    'body_gaps': body_gaps(),
    'guard_objects': guard_objects,
    'metal_objects': metal_objects,
    'tiles': tiles,
    'geometry_basis': '原照正投影轮廓测量 + 未标定单只蓝色气瓶类别参考',
    'render_samples': rr['six_view_samples'],
    'source_camera': source,
    'shift': list(shift),
}
(out / 'worker-result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                        encoding='utf8')
