"""Blender surface finish. All added layers are bounded and leave geometry intact."""
import hashlib
import re

import bpy
import numpy as np
from mathutils import Vector
from mathutils.noise import noise_vector, seed_set

from .scanned_shader import assign, math_node, mix, opaque_surface_output, original, texture


def mul(tree, a, b):
    return math_node(tree, 'MULTIPLY', a, b)


def add(tree, a, b):
    return math_node(tree, 'ADD', a, b)


def ramp(tree, value, start, slope):
    return math_node(tree, 'MULTIPLY', math_node(tree, 'SUBTRACT', value, start), slope, clamp=True)


def channels(tree, color):
    node = tree.nodes.new('ShaderNodeSeparateColor')
    assign(tree, color, node.inputs[0])
    return tuple(node.outputs[name] for name in ('Red', 'Green', 'Blue'))


def tile(tree, coordinate, entry, name, scale, offset):
    size = tree.nodes.new('ShaderNodeVectorMath')
    size.operation = 'SCALE'
    assign(tree, coordinate, size.inputs[0])
    size.inputs['Scale'].default_value = scale
    shift = tree.nodes.new('ShaderNodeVectorMath')
    shift.operation = 'ADD'
    assign(tree, size.outputs[0], shift.inputs[0])
    shift.inputs[1].default_value = offset
    node = texture(tree, entry['path'], shift.outputs[0], name)
    node.name = name
    node.projection = 'BOX'
    node.projection_blend = .28
    node.extension = 'REPEAT'
    return channels(tree, node.outputs['Color'])


def relief(tree, name, height, normal, distance, strength):
    bump = tree.nodes.new('ShaderNodeBump')
    bump.name = name
    bump.inputs['Distance'].default_value = distance
    assign(tree, strength, bump.inputs['Strength'])
    assign(tree, height, bump.inputs['Height'])
    if normal is not None:
        assign(tree, normal, bump.inputs['Normal'])
    return bump.outputs['Normal']


def dust_color(environment):
    color = environment['dust_color']
    luma = max(.005, sum(weight * value for weight, value in zip((.2126, .7152, .0722), color)))
    return tuple(.22 * min(1.12, max(.88, value / luma)) for value in color) + (1.,)


def surface_noise(tree, coordinates, scale, detail):
    node = tree.nodes.new('ShaderNodeTexNoise')
    assign(tree, coordinates, node.inputs['Vector'])
    node.inputs['Scale'].default_value = scale
    node.inputs['Detail'].default_value = detail
    return node.outputs['Fac']


def update_surface_palette(tree, environment):
    """Keep mineral colour under the measured local camera cast."""
    observed = np.asarray(environment['dust_color'])
    luma = max(.015, float(observed @ [.2126, .7152, .0722]))
    tint = np.clip(observed / luma, .84, 1.16)
    value = float(np.clip(.91 + (luma - .12) * .43, .86, 1.08))
    for name, pigment in (
        ('Local dark oxide', (.085, .059, .044)),
        ('Local light oxide', (.202, .154, .116)),
        ('Local nonferrous patina', (.24, .235, .220)),
        ('Local ground-in grime', (.042, .035, .028)),
        ('Local dull nonferrous oxide', (.175, .170, .153)),
        ('Local wet sediment', (.025, .021, .016)),
        ('Local dry sediment', (.21, .174, .125)),
        ('Local caked oil dirt', (.024, .020, .016)),
        ('Local caked mineral dirt', (.18, .158, .126)),
        ('Local aged amber film', (.270, .172, .069)),
        ('Local ochre in grime', (.206, .144, .074)),
    ):
        node = tree.nodes.get(name)
        if node is not None:
            node.outputs[0].default_value = tuple(pigment * tint * value) + (1.,)


def palette_socket(tree, name):
    node = tree.nodes.new('ShaderNodeRGB')
    node.name = name
    return node.outputs[0]


def ground_sediment(tree, coordinates, entry, paint, grime):
    """Mud patches with runs along the normal service axis of the painted bottle."""
    stretch = tree.nodes.new('ShaderNodeVectorMath')
    stretch.operation = 'MULTIPLY'
    assign(tree, coordinates, stretch.inputs[0])
    stretch.inputs[1].default_value = (2.8, .62, 2.8)
    _, runoff, grains = tile(tree, stretch.outputs[0], entry, 'Axial stains and sediment', 1.6, (.46, .28, .39))
    burden = add(tree, mul(tree, runoff, .78), mul(tree, grains, .34))
    mud = mul(tree, paint, ramp(tree, burden, .035, 2.35))
    return math_node(tree, 'MAXIMUM', grime, mud), ramp(tree, grains, .18, 1.9)


def dull_nonferrous_layer(tree, substrate, coverage, fine, normal):
    oxide = tree.nodes.new('ShaderNodeBsdfPrincipled')
    oxide.name = 'Dull nonferrous oxide islands'
    oxide['display_deposit'] = True
    oxide.inputs['Metallic'].default_value = 0.
    oxide.inputs['Specular IOR Level'].default_value = .11
    tint = palette_socket(tree, 'Local dull nonferrous oxide')
    assign(tree, mix(tree, mul(tree, fine, .72), tint, .57, blend='MULTIPLY'), oxide.inputs['Base Color'])
    assign(tree, add(tree, .78, mul(tree, fine, .17)), oxide.inputs['Roughness'])
    assign(tree, normal, oxide.inputs['Normal'])
    layer = tree.nodes.new('ShaderNodeMixShader')
    assign(tree, coverage, layer.inputs[0])
    assign(tree, substrate, layer.inputs[1])
    assign(tree, oxide.outputs[0], layer.inputs[2])
    return layer.outputs[0]


def caked_dirt(tree, coords, texture_entry, surface, label, native_color):
    """Separate packed wet dirt, dry crust and a thin splatter on bottle print."""
    _, wet, grains = tile(tree, coords, texture_entry, 'Caked dirt and splatter', 2.36, (.61, .43, .82))
    blotches = surface_noise(tree, coords, 8.5, 3.)
    field = add(tree, mul(tree, blotches, .47), add(tree, mul(tree, wet, .42), mul(tree, grains, .29)))
    packed = ramp(tree, field, .255, 3.8)
    gain = mul(tree, surface, mul(tree, packed, .92))
    if label is not None:
        luma = tree.nodes.new('ShaderNodeRGBToBW')
        assign(tree, native_color, luma.inputs[0])
        # The audited print rectangle is part of the bottle material. Keep the
        # brightest lettering under at most eight percent dirt.
        visible_ink = ramp(tree, luma.outputs[0], .13, 5.)
        allowance = add(tree, .08, mul(tree, math_node(tree, 'SUBTRACT', 1., visible_ink), .18))
        splatter = mul(tree, label, mul(tree, packed, allowance))
        gain = math_node(tree, 'ADD', gain, splatter, clamp=True)
    mineral = ramp(tree, add(tree, mul(tree, grains, .62), mul(tree, blotches, .54)), .30, 2.8)
    pigment = mix(tree, mineral, palette_socket(tree, 'Local caked oil dirt'),
                  palette_socket(tree, 'Local caked mineral dirt'))
    return gain, pigment, mul(tree, grains, packed)


def amber_film(tree, coords, substrate, nonferrous, oil, fine, scuffs, normal):
    blotches = surface_noise(tree, coords, 7.2, 3.)
    specks = surface_noise(tree, coords, 37., 2.)
    deposition = add(tree, mul(tree, blotches, .65), add(tree, mul(tree, oil, .32), mul(tree, specks, .25)))
    coverage = ramp(tree, deposition, .393, 4.5)
    worn = math_node(tree, 'SUBTRACT', 1., mul(tree, scuffs, .56))
    residue = mul(tree, nonferrous, mul(tree, coverage, worn))
    film = tree.nodes.new('ShaderNodeBsdfPrincipled')
    film.name = 'Old amber oil film'
    film['display_deposit'] = True
    film.inputs['Metallic'].default_value = 0.
    film.inputs['Specular IOR Level'].default_value = .22
    tint = palette_socket(tree, 'Local aged amber film')
    assign(tree, mix(tree, mul(tree, fine, .68), tint, .62, blend='MULTIPLY'), film.inputs['Base Color'])
    assign(tree, add(tree, .53, mul(tree, fine, .23)), film.inputs['Roughness'])
    assign(tree, normal, film.inputs['Normal'])
    layer = tree.nodes.new('ShaderNodeMixShader')
    assign(tree, mul(tree, residue, .82), layer.inputs[0])
    assign(tree, substrate, layer.inputs[1])
    assign(tree, film.outputs[0], layer.inputs[2])
    return layer.outputs[0], residue


def shape_weld_beads(mesh, finish):
    """Deform only vertices whose incident corners belong to documented weld tiles."""
    tiles = {tile for entry in finish['regions']['materials'].values() for tile in entry.get('weld_tiles', ())}
    if not tiles or not mesh.uv_layers.active:
        return {'vertices': 0, 'max_offset': 0.}
    uv = np.empty((len(mesh.loops), 2), np.float32)
    mesh.uv_layers.active.uv.foreach_get('vector', uv.ravel())
    corner_ids = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get('vertex_index', corner_ids)
    cells = np.floor(uv * 4).astype(np.int32)
    weld_corners = np.isin(cells[:, 0] + cells[:, 1] * 4, list(tiles)) & ((cells >= 0) & (cells < 4)).all(axis=1)
    selected = np.unique(corner_ids[weld_corners])
    selected = np.setdiff1d(selected, corner_ids[~weld_corners])
    if not len(selected):
        return {'vertices': 0, 'max_offset': 0.}
    vertices = np.empty((len(mesh.vertices), 3), np.float32)
    normals = np.empty_like(vertices)
    mesh.vertices.foreach_get('co', vertices.ravel())
    mesh.vertex_normals.foreach_get('vector', normals.ravel())
    # Source coordinates and normals keep the perturbation continuous across
    # coincident UV corners. A sub-millimetric relief leaves the parts intact.
    offsets = np.empty(len(selected), np.float32)
    # Blender otherwise chooses a noise seed for each process. Replays that
    # only change appearance must use identical displaced weld vertices.
    seed_set(527)
    for index, point in enumerate(vertices[selected]):
        position = Vector(point)
        coarse = noise_vector(position * 51.)[0]
        detail = noise_vector(position * 123.)[1]
        offsets[index] = .00078 * (.72 * coarse + .28 * detail)
    vertices[selected] += offsets[:, None] * normals[selected]
    mesh.vertices.foreach_set('co', vertices.ravel())
    mesh.update()
    return {'vertices': int(len(selected)), 'max_offset': float(np.max(np.abs(offsets))),
            'vertex_sha256': hashlib.sha256(vertices.tobytes()).hexdigest(),
            'basis': 'authored_weld_uv_tiles_only_in_unit_length_mesh'}


def cylinder_channels(tree, coordinates, entry, texture_entry):
    """Sample axial scrapes and circumferential scuffs without box projection seams."""
    mode = entry['cylinder']
    split = tree.nodes.new('ShaderNodeSeparateXYZ')
    assign(tree, coordinates.outputs['UV' if mode == 'source_uv' else 'Object'], split.inputs[0])
    x, y, z = (split.outputs[axis] for axis in 'XYZ')
    if mode == 'source_uv':
        axial, angle, domain = y, x, 1.
    else:
        low, high, center_x, center_z, radius = entry['barrel']
        axial = mul(tree, add(tree, y, -low), 1. / (high - low))
        radial_x, radial_z = add(tree, x, -center_x), add(tree, z, -center_z)
        angle = add(tree, mul(tree, math_node(tree, 'ARCTAN2', radial_z, radial_x), 1. / (2. * np.pi)), .5)
        radial = math_node(tree, 'SQRT', add(tree, mul(tree, radial_x, radial_x), mul(tree, radial_z, radial_z)), 0.)
        wall = math_node(tree, 'ABSOLUTE', math_node(tree, 'SUBTRACT', radial, radius), 0.)
        domain = mul(tree, ramp(tree, math_node(tree, 'SUBTRACT', .125 * radius, wall), 0., 11. / radius),
                     mul(tree, ramp(tree, axial, -.012, 55.),
                         ramp(tree, math_node(tree, 'SUBTRACT', 1.012, axial), 0., 55.)))
    vector = tree.nodes.new('ShaderNodeCombineXYZ')
    assign(tree, axial, vector.inputs['X'])
    assign(tree, angle, vector.inputs['Y'])
    wear = texture(tree, texture_entry['path'], vector.outputs[0], 'Cylinder wear · axial and angular')
    return (*channels(tree, wear.outputs['Color']), domain)


def corroded_metal(tree, coords, substrate, normal, fine, pits, oxidation, metal, paint,
                   native_metalness, rust, exposed, chip_area, cylindrical, *, local_weather=False,
                   handling=False, patina=0.):
    """Layer oxidized islands and worn steel using a shared ferrous mask."""
    eligible = math_node(tree, 'ADD', metal, paint, clamp=True)
    islands = surface_noise(tree, coords, 11., 4.)
    start = .477 if handling else .495 if local_weather else .54
    islands = ramp(tree, add(tree, islands, mul(tree, oxidation, .32)), start, 4.3)
    scoured = math_node(tree, 'MAXIMUM', exposed, mul(tree, chip_area, .70))
    islands = mul(tree, islands, math_node(tree, 'SUBTRACT', 1., mul(tree, ramp(tree, scoured, .12, 3.5), .88)))
    rust = mul(tree, math_node(tree, 'MAXIMUM', mul(tree, eligible, islands), rust), .85)
    steel_weight = math_node(tree, 'MAXIMUM', ramp(tree, exposed, 0., 1.30), mul(tree, chip_area, .83))
    intact_metal = mul(tree, metal, ramp(tree, native_metalness, .35, 1.55))
    steel_weight = math_node(tree, 'MAXIMUM', steel_weight, mul(tree, intact_metal, .83))
    steel = tree.nodes.new('ShaderNodeBsdfPrincipled')
    steel.name = 'Weathered exposed steel'
    steel['display_deposit'] = True
    assign(tree, mix(tree, fine, (.31, .306, .298, 1.), (.49, .484, .469, 1.)), steel.inputs['Base Color'])
    steel.inputs['Metallic'].default_value = 1.
    assign(tree, add(tree, .40 if local_weather else .34, mul(tree, fine, .17)), steel.inputs['Roughness'])
    if cylindrical == 'normalized_y':
        steel.inputs['Anisotropic'].default_value = .15
        direction = tree.nodes.new('ShaderNodeVectorTransform')
        direction.vector_type = 'VECTOR'
        direction.convert_from, direction.convert_to = 'OBJECT', 'WORLD'
        direction.inputs['Vector'].default_value = (0., 1., 0.)
        assign(tree, direction.outputs[0], steel.inputs['Tangent'])
    assign(tree, normal, steel.inputs['Normal'])
    worn = tree.nodes.new('ShaderNodeMixShader')
    assign(tree, steel_weight, worn.inputs[0])
    assign(tree, substrate, worn.inputs[1])
    assign(tree, steel.outputs[0], worn.inputs[2])

    oxide = tree.nodes.new('ShaderNodeBsdfPrincipled')
    oxide.name = 'Matte iron oxide'
    oxide['display_deposit'] = True
    oxide.inputs['Metallic'].default_value = 0.
    oxide.inputs['Specular IOR Level'].default_value = .14
    colors = (palette_socket(tree, 'Local dark oxide'), palette_socket(tree, 'Local light oxide')) if local_weather else (
        (.078, .037, .022, 1.), (.195, .113, .072, 1.))
    assign(tree, mix(tree, fine, *colors), oxide.inputs['Base Color'])
    assign(tree, add(tree, .75, mul(tree, pits, .20)), oxide.inputs['Roughness'])
    assign(tree, relief(tree, 'Rust-pit relief', mul(tree, pits, -1.), normal, .00028, .24), oxide.inputs['Normal'])
    top = tree.nodes.new('ShaderNodeMixShader')
    assign(tree, rust, top.inputs[0])
    assign(tree, worn.outputs[0], top.inputs[1])
    assign(tree, oxide.outputs[0], top.inputs[2])
    visible_steel = mul(tree, steel_weight, math_node(tree, 'SUBTRACT', 1., rust))
    write_surface_pass(tree, 'corrosion_finish', (rust, visible_steel, patina if local_weather else eligible))
    return top.outputs[0], visible_steel


def finish_surface(material, shader, finish, environment, strength):
    tree = material.node_tree
    if shader.get('generated_finish'):
        color = tree.nodes.get('Generated dust tint')
        if color is not None:
            color.outputs[0].default_value = dust_color(environment)
        if finish.get('version') in ('generated-pbr-v8', 'generated-pbr-v9', 'generated-pbr-v10',
                                     'generated-pbr-v11', 'generated-pbr-v12'):
            update_surface_palette(tree, environment)
        return True
    name = re.sub(r'\.\d{3}$', '', material.name)
    entry = finish['regions']['materials'].get(name)
    if not entry or not any(entry['coverage'][:3]) or (
            not entry.get('authored_atlas') and re.search(
                r'\b(?:label|mark|decal|print|chrome|machined|bearing|polished|zerk)\b', name, re.I)):
        return False
    output = opaque_surface_output(tree, shader)
    if output is None:
        return False
    if not 0 <= strength <= .5:
        raise ValueError('generated_pbr surface dust exceeds the supported range')
    coordinates = tree.nodes.new('ShaderNodeTexCoord')
    weld = 0.
    if 'mask' in entry:
        mask = texture(tree, entry['mask'], coordinates.outputs['UV'], 'Known surface regions')
        mask.interpolation = 'Closest'
        metal, paint, third = channels(tree, mask.outputs['Color'])
        if entry.get('weld_tiles'):
            mask.image.alpha_mode = 'CHANNEL_PACKED'
            weld = mask.outputs['Alpha']
    else:
        metal, paint, third = entry['constant'][:3]
    rough = original(shader.inputs['Roughness'])
    # The authored regions exclude hard chrome, inserts, seals and marks. With
    # inferred masks, a very low source roughness is evidence of a finished surface.
    if entry.get('authored_atlas'):
        nonferrous = third
    else:
        metal = math_node(tree, 'ADD', mul(tree, metal, ramp(tree, rough, .17, 7.)), third, clamp=True)
        nonferrous = 0.
    known = math_node(tree, 'ADD', math_node(tree, 'ADD', metal, paint, clamp=True), nonferrous, clamp=True)
    version = finish.get('version')
    local_weather = version in ('generated-pbr-v8', 'generated-pbr-v9', 'generated-pbr-v10',
                                'generated-pbr-v11', 'generated-pbr-v12')
    aged_surface = version in ('generated-pbr-v10', 'generated-pbr-v11', 'generated-pbr-v12')
    handling = 'handling' in finish['maps']
    corrosion = finish.get('corrosion', False)
    ferrous = corrosion and (entry.get('authored_atlas') or (
        finish['regions'].get('ferrous_source') and not re.search(
            r'\b(?:alumin\w*|brass|bronze|copper|stainless)\b', name, re.I)))
    cylinder_pack = 'cylinder' in finish['maps']
    cylindrical = cylinder_pack and entry.get('cylinder')
    caked = version in ('generated-pbr-v11', 'generated-pbr-v12') and entry.get('authored_atlas') and (
        cylindrical == 'source_uv' or entry['coverage'][2] > 0)
    amber = finish.get('amber_age') and entry.get('authored_atlas') and entry['coverage'][2] > 0
    hydraulic = 'hydraulic' in finish['maps'] and cylindrical == 'normalized_y'
    formed = cylinder_pack or version == 'generated-pbr-v3'
    layered_wear = formed or finish.get('version') == 'generated-pbr-v2'
    coords = coordinates.outputs['Object']  # Canonical mesh is scaled uniformly by longest axis.
    oxidation, pits, fine = tile(tree, coords, finish['maps']['oxide'], 'Generated oxidation', 1.9, (.17, .39, .72))
    chips, lines, chip_grain = tile(tree, coords, finish['maps']['paint'], 'Generated chipped paint', 2.4, (.57, .12, .29))
    chip_area = mul(tree, paint, ramp(tree, chips, .06 if layered_wear else .14, 1.35 if layered_wear else 1.22))
    rust_area = mul(tree, metal, ramp(tree, oxidation, .08, 1.08)) if ferrous or not corrosion else 0.
    scratches = mul(tree, known, lines)
    if handling:
        scratch_lines, oil, dry_scale = tile(tree, coords, finish['maps']['handling'],
            'Grouped handling marks', 1.55, (.34, .67, .29))
        inspection_wear = add(tree, mul(tree, known, .67), mul(tree, nonferrous, .33))
        scuff_lines = mul(tree, scratch_lines, inspection_wear)
        grime = mul(tree, oil, add(tree, mul(tree, known, .51), mul(tree, nonferrous, .24)))
        scratches = math_node(tree, 'MAXIMUM', scratches, scuff_lines)
        if aged_surface and cylindrical == 'source_uv':
            grime, sediment = ground_sediment(tree, coords, finish['maps']['handling'], paint, grime)
    worn = math_node(tree, 'ADD', rust_area, chip_area, clamp=True)
    paint_wear = 0.
    if layered_wear:
        # Patchy faded or stained paint remains dielectric until it chips.
        # Give it a separate projection and scale from the exposed steel.
        mottling, _, _ = tile(tree, coords, finish['maps']['oxide'], 'Generated paint weathering',
                             2.8, (.73, .17, .51))
        if formed:
            # Retain the irregular islands but break up the uniform dark fill.
            mottling = mul(tree, mottling, add(tree, .46, mul(tree, surface_noise(tree, coords, 64., 2.), 1.0)))
        paint_wear = mul(tree, paint, ramp(tree, mottling, .15, 1.55))
        worn = math_node(tree, 'ADD', worn, paint_wear, clamp=True)
    body_rust, scratch_steel = 0., 0.
    if cylindrical:
        scratch_map = finish['maps']['hydraulic' if hydraulic else 'cylinder']
        scrape, exposed, nuclei, domain = cylinder_channels(tree, coordinates, entry, scratch_map)
        surface = mul(tree, paint, domain)
        body_rust = mul(tree, surface, ramp(tree, add(tree, oxidation, mul(tree, nuclei, 1.05 if hydraulic else .74)), .18, 1.7))
        if hydraulic:
            exposed = ramp(tree, exposed, .12, 1.75)
        clean = math_node(tree, 'SUBTRACT', 1., mul(tree, body_rust, .87))
        scratch_steel = mul(tree, mul(tree, surface, exposed), clean)
        scuffs = mul(tree, surface, scrape)
        worn = math_node(tree, 'MAXIMUM', worn, body_rust)
        scratches = math_node(tree, 'MAXIMUM', scratches, scuffs)
    base = original(shader.inputs['Base Color'])
    base = mix(tree, mul(tree, known, .58), base, add(tree, .89, mul(tree, fine, .21)), blend='MULTIPLY')
    base = mix(tree, mul(tree, nonferrous, oxidation), base, .67, blend='MULTIPLY')
    patina = 0.
    if local_weather and entry.get('authored_atlas') and entry['coverage'][2] > 0:
        field = add(tree, surface_noise(tree, coords, 16., 3.), mul(tree, oxidation, .29))
        patina = mul(tree, nonferrous, ramp(tree, field, .49, 4.3))
        if handling:
            patina = math_node(tree, 'MAXIMUM', patina, mul(tree, nonferrous, dry_scale))
        base = mix(tree, mul(tree, patina, .56 if handling else .36), base, palette_socket(tree, 'Local nonferrous patina'))
    oxide_tint = mix(tree, fine, (.065, .052, .042, 1.), (.16, .125, .096, 1.))
    chip_tint = mix(tree, oxidation, (.24, .224, .20, 1.), oxide_tint)
    if not ferrous:
        base = mix(tree, mul(tree, rust_area, .78), base, oxide_tint)
    if layered_wear:
        base = mix(tree, mul(tree, paint_wear, .58), base, oxide_tint)
    if formed:
        bead = surface_noise(tree, coords, 102., 2.)
        base = mix(tree, mul(tree, weld, .72), base, add(tree, .62, mul(tree, bead, .34)), blend='MULTIPLY')
    base = mix(tree, mul(tree, chip_area, .79), base, chip_tint)
    base = mix(tree, mul(tree, scratches, .42 if hydraulic else .32), base, .55, blend='MULTIPLY')
    if handling:
        base = mix(tree, mul(tree, scuff_lines, .37), base, .38, blend='MULTIPLY')
    if cylindrical and not ferrous:
        # The surrounding rust is dark and rough; only a few deep scrape
        # centers expose the cooler steel response.
        rust_color = mix(tree, fine, (.074, .035, .020, 1.), (.18, .093, .045, 1.))
        base = mix(tree, mul(tree, body_rust, .78), base, rust_color)
        if not hydraulic:
            base = mix(tree, scratch_steel, base, (.30, .287, .265, 1.))
    assign(tree, base, shader.inputs['Base Color'])
    rough_area = math_node(tree, 'ADD', mul(tree, nonferrous, .64), mul(tree, metal, .31), clamp=True)
    rough = mix(tree, rough_area, rough, math_node(tree, 'MAXIMUM', rough, add(tree, .50, mul(tree, pits, .20))))
    rough = mix(tree, worn, rough, math_node(tree, 'MAXIMUM', rough, add(tree, .61, mul(tree, fine, .25))))
    if layered_wear:
        # Break the long highlight over scuffed aluminum without turning it
        # into iron oxide or changing the untouched areas' metalness.
        abrasion = mul(tree, nonferrous, ramp(tree, oxidation, .18, 1.35))
        rough = mix(tree, mul(tree, abrasion, .5), rough, math_node(tree, 'MAXIMUM', rough, .68))
    if formed:
        # Different wear levels interrupt the large continuous highlight on
        # exposed metal and paint, while protected chrome is still excluded.
        glancing = surface_noise(tree, coords, 19., 3.)
        scuff = mul(tree, math_node(tree, 'ADD', nonferrous, paint, clamp=True), ramp(tree, glancing, .40, 2.8))
        rough = mix(tree, mul(tree, scuff, .40), rough, math_node(tree, 'MAXIMUM', rough, .63))
        rough = mix(tree, mul(tree, weld, .64), rough, add(tree, .55, mul(tree, bead, .30)))
    if corrosion and formed:
        rough = mix(tree, mul(tree, nonferrous, .68), rough, add(tree, .34, mul(tree, glancing, .27)))
    if local_weather:
        rough = mix(tree, mul(tree, patina, .66), rough, add(tree, .68, mul(tree, fine, .15)))
    rough = mix(tree, mul(tree, scratches, .35), rough, .72)
    if cylindrical:
        rough = mix(tree, body_rust, rough, add(tree, .68, mul(tree, pits, .21)))
        if not hydraulic:
            rough = mix(tree, scratch_steel, rough, add(tree, .28, mul(tree, fine, .16)))
        coat = shader.inputs['Coat Weight']
        damage = math_node(tree, 'MAXIMUM', body_rust, scuffs)
        assign(tree, mul(tree, original(coat), math_node(tree, 'SUBTRACT', 1., mul(tree, damage, .85))), coat)
    assign(tree, rough, shader.inputs['Roughness'])
    native_metalness = original(shader.inputs['Metallic'])
    metalness = native_metalness
    metalness = mix(tree, mul(tree, rust_area, .80), metalness, .06)
    metalness = mix(tree, mul(tree, chip_area, .72), metalness, mix(tree, oxidation, .72, .12))
    if cylindrical:
        metalness = mix(tree, body_rust, metalness, .04)
        if not hydraulic:
            metalness = mix(tree, scratch_steel, metalness, .88)
    if local_weather:
        metalness = mix(tree, mul(tree, patina, .45), metalness, .38)
    assign(tree, metalness, shader.inputs['Metallic'])
    source_normal = original(shader.inputs['Normal']) if shader.inputs['Normal'].is_linked else None
    steel_height = mul(tree, metal, add(tree, mul(tree, fine, .32), mul(tree, pits, -.45)))
    paint_height = mul(tree, paint, add(tree, mul(tree, chip_grain, .20), mul(tree, chips, -.63)))
    height = add(tree, add(tree, steel_height, paint_height),
                 add(tree, mul(tree, nonferrous, mul(tree, fine, .20)), mul(tree, scratches, -.38)))
    if formed:
        height = add(tree, height, add(tree, mul(tree, scuff, mul(tree, glancing, .48)), mul(tree, weld, bead)))
    if cylindrical:
        height = add(tree, height, add(tree, mul(tree, scuffs, -1.12 if hydraulic else -.80),
                                      mul(tree, mul(tree, body_rust, pits), -.68 if hydraulic else -.54)))
    if handling:
        height = add(tree, height, add(tree, mul(tree, scuff_lines, -.48), mul(tree, patina, .10)))
    normal = relief(tree, 'Generated shallow surface relief', height, source_normal, .00048, mul(tree, known, .34))
    assign(tree, normal, shader.inputs['Normal'])
    substrate = shader.outputs[0]
    if ferrous:
        rust = math_node(tree, 'MAXIMUM', rust_area, body_rust)
        substrate, exposed_steel = corroded_metal(tree, coords, substrate, normal, fine, pits, oxidation,
            metal, paint, native_metalness, rust, scratch_steel, chip_area, cylindrical,
            local_weather=local_weather, handling=handling, patina=patina)
        assign(tree, substrate, output.inputs['Surface'])
    elif hydraulic:
        steel = tree.nodes.new('ShaderNodeBsdfPrincipled')
        steel.name = 'Hydraulic exposed steel'
        steel['display_deposit'] = True
        assign(tree, mix(tree, fine, (.26, .255, .245, 1.), (.43, .421, .406, 1.)), steel.inputs['Base Color'])
        steel.inputs['Metallic'].default_value = 1.
        assign(tree, add(tree, .22, mul(tree, fine, .23)), steel.inputs['Roughness'])
        steel.inputs['Anisotropic'].default_value = .20
        direction = tree.nodes.new('ShaderNodeVectorTransform')
        direction.vector_type = 'VECTOR'
        direction.convert_from, direction.convert_to = 'OBJECT', 'WORLD'
        direction.inputs['Vector'].default_value = (0., 1., 0.)
        assign(tree, direction.outputs[0], steel.inputs['Tangent'])
        assign(tree, normal, steel.inputs['Normal'])
        layer = tree.nodes.new('ShaderNodeMixShader')
        assign(tree, scratch_steel, layer.inputs[0])
        assign(tree, substrate, layer.inputs[1])
        assign(tree, steel.outputs[0], layer.inputs[2])
        substrate = layer.outputs[0]
        assign(tree, substrate, output.inputs['Surface'])
    if hydraulic:
        write_surface_pass(tree, 'hydraulic_finish', (body_rust, scratch_steel, scuffs))
    if aged_surface and entry.get('authored_atlas') and entry['coverage'][2] > 0:
        veil = ramp(tree, patina, .11, 1.7)
        rubbed = math_node(tree, 'SUBTRACT', 1., mul(tree, scuff_lines, .33))
        substrate = dull_nonferrous_layer(tree, substrate, mul(tree, mul(tree, veil, rubbed), .87), fine, normal)
        assign(tree, substrate, output.inputs['Surface'])
    if amber:
        substrate, residue = amber_film(tree, coords, substrate, nonferrous, oil, fine, scuff_lines, normal)
        assign(tree, substrate, output.inputs['Surface'])
        write_surface_pass(tree, 'amber_film', (residue, nonferrous, 0.))
    if handling:
        stain = tree.nodes.new('ShaderNodeBsdfPrincipled')
        stain.name = 'Ground-in nonmetallic grime'
        stain['display_deposit'] = True
        stain.inputs['Specular IOR Level'].default_value = .16
        stain_normal = normal
        if caked:
            bottle = cylindrical == 'source_uv'
            surface = paint if bottle else math_node(tree, 'ADD', metal, nonferrous, clamp=True)
            print_region = math_node(tree, 'SUBTRACT', 1., paint) if bottle else None
            crust, color, texture_height = caked_dirt(tree, coords, finish['maps']['handling'], surface, print_region,
                                                    original(shader.inputs['Base Color']))
            grime = math_node(tree, 'MAXIMUM', grime, crust)
            stain_normal = relief(tree, 'Packed dirt relief', texture_height, normal, .00074, .26)
        elif aged_surface and cylindrical == 'source_uv':
            color = mix(tree, sediment, palette_socket(tree, 'Local wet sediment'),
                        palette_socket(tree, 'Local dry sediment'))
        else:
            color = palette_socket(tree, 'Local ground-in grime')
        if amber:
            color = mix(tree, mul(tree, residue, .70), color, palette_socket(tree, 'Local ochre in grime'))
        assign(tree, color, stain.inputs['Base Color'])
        assign(tree, add(tree, .69, mul(tree, fine, .19)), stain.inputs['Roughness'])
        assign(tree, stain_normal, stain.inputs['Normal'])
        layer = tree.nodes.new('ShaderNodeMixShader')
        assign(tree, grime, layer.inputs[0])
        assign(tree, substrate, layer.inputs[1])
        assign(tree, stain.outputs[0], layer.inputs[2])
        substrate = layer.outputs[0]
        assign(tree, substrate, output.inputs['Surface'])
        write_surface_pass(tree, 'handling_finish', (scuff_lines, grime, patina))
    powder = 0.
    if strength:
        geometry = tree.nodes.new('ShaderNodeNewGeometry')
        upward = tree.nodes.new('ShaderNodeSeparateXYZ')
        assign(tree, geometry.outputs['Normal'], upward.inputs[0])
        up = math_node(tree, 'MAXIMUM', upward.outputs['Z'], 0.)
        ao = tree.nodes.new('ShaderNodeAmbientOcclusion')
        ao.only_local = True
        ao.samples = 4
        ao.inputs['Distance'].default_value = .046
        cavity = math_node(tree, 'SUBTRACT', 1., ao.outputs['AO'], clamp=True)
        edge = ramp(tree, geometry.outputs['Pointiness'], .505, 14.)
        deposit, grains, powder_detail = tile(tree, coords, finish['maps']['dust'], 'Generated dust deposits',
                                               3.1, (.83, .24, .48))
        deposition = math_node(tree, 'ADD', add(tree, .22, mul(tree, up, .52)), mul(tree, cavity, .50), clamp=True)
        if layered_wear:
            deposition = math_node(tree, 'ADD', add(tree, .32, mul(tree, up, .55)), mul(tree, cavity, .43), clamp=True)
            deposit = ramp(tree, deposit, .10, 1.5)
        distribution = mul(tree, add(tree, .04, mul(tree, deposit, .96)),
                           math_node(tree, 'SUBTRACT', 1., mul(tree, edge, .64)))
        dust_region = math_node(tree, 'ADD', add(tree, metal, nonferrous), mul(tree, paint, .5), clamp=True)
        powder = mul(tree, mul(tree, strength, dust_region), mul(tree, deposition, distribution))
        if hydraulic:
            powder = mul(tree, powder, math_node(tree, 'SUBTRACT', 1., mul(tree, scratch_steel, .60)))
        if ferrous:
            powder = mul(tree, powder, math_node(tree, 'SUBTRACT', 1., mul(tree, exposed_steel, .34)))
        powder.node.name = 'Generated dust coverage'
        tint = tree.nodes.new('ShaderNodeRGB')
        tint.name = 'Generated dust tint'
        tint.outputs[0].default_value = dust_color(environment)
        dust = tree.nodes.new('ShaderNodeBsdfPrincipled')
        dust.name = 'Generated dielectric dust'
        dust['display_deposit'] = True
        assign(tree, mix(tree, .7, tint.outputs[0], add(tree, .86, mul(tree, grains, .20)), blend='MULTIPLY'),
               dust.inputs['Base Color'])
        dust.inputs['Metallic'].default_value = 0.
        dust.inputs['Specular IOR Level'].default_value = .10
        assign(tree, add(tree, .82, mul(tree, grains, .14)), dust.inputs['Roughness'])
        assign(tree, relief(tree, 'Generated dust micro-relief', add(tree, grains, mul(tree, powder_detail, .35)),
                            normal, .00024, .18), dust.inputs['Normal'])
        layer = tree.nodes.new('ShaderNodeMixShader')
        assign(tree, powder, layer.inputs[0])
        assign(tree, substrate, layer.inputs[1])
        assign(tree, dust.outputs[0], layer.inputs[2])
        assign(tree, layer.outputs[0], output.inputs['Surface'])
    write_surface_pass(tree, 'generated_finish', (worn, scratches, powder))
    if local_weather:
        update_surface_palette(tree, environment)
    shader['generated_finish'] = True
    return True


def write_surface_pass(tree, name, values):
    aov = tree.nodes.new('ShaderNodeOutputAOV')
    aov.aov_name = name
    rgb = tree.nodes.new('ShaderNodeCombineColor')
    for label, value in zip(('Red', 'Green', 'Blue'), values):
        assign(tree, value, rgb.inputs[label])
    assign(tree, rgb.outputs[0], aov.inputs['Color'])
