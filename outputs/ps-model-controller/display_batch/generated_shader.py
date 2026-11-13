"""Blender surface finish. All added layers are bounded and leave geometry intact."""
import re

import bpy

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


def finish_surface(material, shader, finish, environment, strength):
    tree = material.node_tree
    if shader.get('generated_finish'):
        color = tree.nodes.get('Generated dust tint')
        if color is not None:
            color.outputs[0].default_value = dust_color(environment)
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
    if 'mask' in entry:
        mask = texture(tree, entry['mask'], coordinates.outputs['UV'], 'Known surface regions')
        mask.interpolation = 'Closest'
        metal, paint, third = channels(tree, mask.outputs['Color'])
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
    layered_wear = finish.get('version') == 'generated-pbr-v2'
    coords = coordinates.outputs['Object']  # Canonical mesh is scaled uniformly by longest axis.
    oxidation, pits, fine = tile(tree, coords, finish['maps']['oxide'], 'Generated oxidation', 1.9, (.17, .39, .72))
    chips, lines, chip_grain = tile(tree, coords, finish['maps']['paint'], 'Generated chipped paint', 2.4, (.57, .12, .29))
    chip_area = mul(tree, paint, ramp(tree, chips, .06 if layered_wear else .14, 1.35 if layered_wear else 1.22))
    rust_area = mul(tree, metal, ramp(tree, oxidation, .08, 1.08))
    scratches = mul(tree, known, lines)
    worn = math_node(tree, 'ADD', rust_area, chip_area, clamp=True)
    paint_wear = 0.
    if layered_wear:
        # Patchy faded or stained paint remains dielectric until it chips.
        # Give it a separate projection and scale from the exposed steel.
        mottling, _, _ = tile(tree, coords, finish['maps']['oxide'], 'Generated paint weathering',
                             2.8, (.73, .17, .51))
        paint_wear = mul(tree, paint, ramp(tree, mottling, .15, 1.55))
        worn = math_node(tree, 'ADD', worn, paint_wear, clamp=True)
    base = original(shader.inputs['Base Color'])
    base = mix(tree, mul(tree, known, .58), base, add(tree, .89, mul(tree, fine, .21)), blend='MULTIPLY')
    base = mix(tree, mul(tree, nonferrous, oxidation), base, .67, blend='MULTIPLY')
    oxide_tint = mix(tree, fine, (.065, .052, .042, 1.), (.16, .125, .096, 1.))
    chip_tint = mix(tree, oxidation, (.24, .224, .20, 1.), oxide_tint)
    base = mix(tree, mul(tree, rust_area, .78), base, oxide_tint)
    if layered_wear:
        base = mix(tree, mul(tree, paint_wear, .58), base, oxide_tint)
    base = mix(tree, mul(tree, chip_area, .79), base, chip_tint)
    base = mix(tree, mul(tree, scratches, .32), base, .55, blend='MULTIPLY')
    assign(tree, base, shader.inputs['Base Color'])
    rough_area = math_node(tree, 'ADD', mul(tree, nonferrous, .64), mul(tree, metal, .31), clamp=True)
    rough = mix(tree, rough_area, rough, math_node(tree, 'MAXIMUM', rough, add(tree, .50, mul(tree, pits, .20))))
    rough = mix(tree, worn, rough, math_node(tree, 'MAXIMUM', rough, add(tree, .61, mul(tree, fine, .25))))
    if layered_wear:
        # Break the long highlight over scuffed aluminum without turning it
        # into iron oxide or changing the untouched areas' metalness.
        abrasion = mul(tree, nonferrous, ramp(tree, oxidation, .18, 1.35))
        rough = mix(tree, mul(tree, abrasion, .5), rough, math_node(tree, 'MAXIMUM', rough, .68))
    assign(tree, mix(tree, mul(tree, scratches, .35), rough, .72), shader.inputs['Roughness'])
    metalness = original(shader.inputs['Metallic'])
    metalness = mix(tree, mul(tree, rust_area, .80), metalness, .06)
    assign(tree, mix(tree, mul(tree, chip_area, .72), metalness, mix(tree, oxidation, .72, .12)),
           shader.inputs['Metallic'])
    source_normal = original(shader.inputs['Normal']) if shader.inputs['Normal'].is_linked else None
    steel_height = mul(tree, metal, add(tree, mul(tree, fine, .32), mul(tree, pits, -.45)))
    paint_height = mul(tree, paint, add(tree, mul(tree, chip_grain, .20), mul(tree, chips, -.63)))
    height = add(tree, add(tree, steel_height, paint_height),
                 add(tree, mul(tree, nonferrous, mul(tree, fine, .20)), mul(tree, scratches, -.38)))
    normal = relief(tree, 'Generated shallow surface relief', height, source_normal, .00048, mul(tree, known, .34))
    assign(tree, normal, shader.inputs['Normal'])
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
        assign(tree, shader.outputs[0], layer.inputs[1])
        assign(tree, dust.outputs[0], layer.inputs[2])
        assign(tree, layer.outputs[0], output.inputs['Surface'])
    aov = tree.nodes.new('ShaderNodeOutputAOV')
    aov.aov_name = 'generated_finish'
    rgb = tree.nodes.new('ShaderNodeCombineColor')
    for label, value in zip(('Red', 'Green', 'Blue'), (worn, scratches, powder)):
        assign(tree, value, rgb.inputs[label])
    assign(tree, rgb.outputs[0], aov.inputs['Color'])
    shader['generated_finish'] = True
    return True
