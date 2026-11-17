"""Three bounded layers: unchanged source, material wear, one dust deposit.

Environment inputs intentionally do not modify material parameters. Lighting and
camera adaptation belong to their respective pipeline stages.
"""
import math
import re

import bpy

from .generated_shader import (add, channels, mul, ramp, relief, tile, write_surface_pass)
from .scanned_shader import assign, math_node, mix, opaque_surface_output, original, texture


VERSION = 'physical-layers-v1'


def layer_mix(tree, name, below, above, coverage):
    node = tree.nodes.new('ShaderNodeMixShader')
    node.name = name
    assign(tree, coverage, node.inputs[0])
    assign(tree, below, node.inputs[1])
    assign(tree, above, node.inputs[2])
    return node.outputs[0]


def clone_source(tree, shader):
    """Abrasion keeps each source material's colour and metalness."""
    worn = tree.nodes.new('ShaderNodeBsdfPrincipled')
    worn.name = 'Physical abraded source'
    worn['display_deposit'] = True
    for socket in shader.inputs:
        if socket.name in worn.inputs and hasattr(socket, 'default_value'):
            assign(tree, original(socket), worn.inputs[socket.name])
    return worn


def finish_surface(material, shader, finish, environment, strength):
    if finish.get('version') != VERSION:
        raise ValueError('unsupported physical layers version')
    if not math.isfinite(float(strength)) or not 0 <= float(strength) <= .5:
        raise ValueError('physical_layers surface dust must be between zero and .5')
    size = float(finish['size_m'])
    if not math.isfinite(size) or size <= 0:
        raise ValueError('physical_layers requires a positive physical size in metres')
    stage = finish.get('layer_stage', 'full')
    if stage not in ('source', 'wear', 'full'):
        raise ValueError('unknown physical material layer stage')
    tree = material.node_tree
    if shader.get('physical_layers'):
        if shader.get('physical_layers_signature') != finish['signature'] or shader.get('physical_layers_stage') != stage:
            raise ValueError('physical material or stage changed; reimport source mesh')
        tree.nodes['Physical dust strength'].outputs[0].default_value = float(strength)
        return True
    name = re.sub(r'\.\d{3}$', '', material.name)
    entry = finish['regions']['materials'].get(name)
    if stage == 'source' or not entry or not entry.get('eligible'):
        return False
    output = opaque_surface_output(tree, shader)
    if output is None:
        return False
    # Never stack this mode on generated/source-weathering modifications.
    if shader.get('generated_finish'):
        raise ValueError('physical layers must start from the source material')
    recipe = finish['recipe']
    coordinate = tree.nodes.new('ShaderNodeTexCoord')
    metric = tree.nodes.new('ShaderNodeVectorMath')
    metric.name = 'Physical coordinates in metres'
    metric.operation = 'SCALE'
    assign(tree, coordinate.outputs['Object'], metric.inputs[0])
    metric.inputs['Scale'].default_value = size
    coord = metric.outputs[0]
    if 'mask' in entry:
        mask = texture(tree, entry['mask'], coordinate.outputs['UV'], 'Physical known source regions')
        mask.name = 'Physical known source regions'
        mask.interpolation = 'Closest'
        metal, paint, third = channels(tree, mask.outputs['Color'])
    else:
        metal, paint, third = entry['constant'][:3]
    authored = entry.get('authored_atlas', False)
    if authored:
        ferrous, nonferrous = metal, third
    else:
        ferrous = metal if entry['metal_family'] == 'ferrous' else 0.
        nonferrous = 0.
    known = math_node(tree, 'ADD', math_node(tree, 'ADD', metal, paint, clamp=True), third, clamp=True)
    oxidizable_paint = paint if authored or entry['metal_family'] == 'ferrous' else 0.
    ferrous_domain = math_node(tree, 'ADD', ferrous, oxidizable_paint, clamp=True)
    # Same asset has the same physical defects in every environment.
    seed = finish.get('source_mesh_hash', '0' * 64)
    offset = tuple(int(seed[i:i+4], 16) / 65535. for i in (0, 4, 8))
    oxide, pits, fine = tile(tree, coord, finish['maps']['oxide'], 'Physical oxide masks',
                              1. / recipe['oxide_period_m'], offset)
    chips, lines, grain = tile(tree, coord, finish['maps']['paint'], 'Physical abrasion masks',
                               1. / recipe['scratch_period_m'], (.31, .17, .59))
    rust = mul(tree, ferrous_domain, mul(tree, ramp(tree, oxide, .35, 1.5), recipe['oxide_max']))
    # Roughness changes remain bounded; unchanged source governs intact regions.
    abrasion = mul(tree, known, mul(tree, ramp(tree, math_node(tree, 'MAXIMUM', chips, lines), .18, 1.4),
                                   recipe['abrasion_max']))
    original_normal = original(shader.inputs['Normal']) if shader.inputs['Normal'].is_linked else None
    normal = relief(tree, 'Physical wear depth in metres', add(tree, mul(tree, pits, -.5), mul(tree, lines, -.35)),
                    original_normal, recipe['relief_depth_m'] / size, .22)
    worn = clone_source(tree, shader)
    source_rough = original(shader.inputs['Roughness'])
    assign(tree, math_node(tree, 'MINIMUM', add(tree, source_rough, mul(tree, grain, .12)), .85), worn.inputs['Roughness'])
    assign(tree, normal, worn.inputs['Normal'])
    # Only confidently painted iron can reveal steel; unknown paint keeps its source.
    bare = tree.nodes.new('ShaderNodeBsdfPrincipled')
    bare.name = 'Physical exposed steel'
    bare['display_deposit'] = True
    bare.inputs['Base Color'].default_value = (.37, .37, .36, 1.)
    bare.inputs['Metallic'].default_value = 1.
    assign(tree, add(tree, .34, mul(tree, fine, .16)), bare.inputs['Roughness'])
    assign(tree, normal, bare.inputs['Normal'])
    exposed = mul(tree, oxidizable_paint, ramp(tree, chips, .55, 2.))
    abraded = layer_mix(tree, 'Physical chipped coating response', worn.outputs[0], bare.outputs[0], exposed)
    surface = layer_mix(tree, 'Physical bounded abrasion', shader.outputs[0], abraded, abrasion)
    rust_shader = tree.nodes.new('ShaderNodeBsdfPrincipled')
    rust_shader.name = 'Physical dielectric oxide'
    rust_shader['display_deposit'] = True
    rust_shader.inputs['Metallic'].default_value = 0.
    rust_shader.inputs['Specular IOR Level'].default_value = .35
    assign(tree, mix(tree, fine, (.07, .046, .030, 1.), (.16, .110, .074, 1.)), rust_shader.inputs['Base Color'])
    assign(tree, add(tree, .76, mul(tree, pits, .12)), rust_shader.inputs['Roughness'])
    assign(tree, normal, rust_shader.inputs['Normal'])
    surface = layer_mix(tree, 'Physical bounded oxidation', surface, rust_shader.outputs[0], rust)
    dust_strength = tree.nodes.new('ShaderNodeValue')
    dust_strength.name = 'Physical dust strength'
    dust_strength.outputs[0].default_value = float(strength)
    powder = 0.
    if stage == 'full':
        geometry = tree.nodes.new('ShaderNodeNewGeometry')
        up_split = tree.nodes.new('ShaderNodeSeparateXYZ')
        assign(tree, geometry.outputs['Normal'], up_split.inputs[0])
        up = math_node(tree, 'MAXIMUM', up_split.outputs['Z'], 0.)
        ao = tree.nodes.new('ShaderNodeAmbientOcclusion')
        ao.name = 'Physical local cavity'
        ao.only_local = True
        ao.samples = 4
        ao.inputs['Distance'].default_value = .008 / size
        cavity = math_node(tree, 'SUBTRACT', 1., ao.outputs['AO'], clamp=True)
        coverage, grains, _ = tile(tree, coord, finish['maps']['dust'], 'Physical dust patches',
                                   1. / recipe['dust_period_m'], (.53, .83, .29))
        deposition = math_node(tree, 'ADD', add(tree, .10, mul(tree, up, .55)), mul(tree, cavity, .35), clamp=True)
        domain = math_node(tree, 'ADD', add(tree, metal, third), mul(tree, paint, .5), clamp=True)
        powder = mul(tree, mul(tree, dust_strength.outputs[0], domain),
                     mul(tree, deposition, ramp(tree, coverage, .18, 1.22)))
        dust = tree.nodes.new('ShaderNodeBsdfPrincipled')
        dust.name = 'Physical single dielectric deposit'
        dust['display_deposit'] = True
        dust.inputs['Base Color'].default_value = tuple(recipe['dust_albedo_linear']) + (1.,)
        dust.inputs['Metallic'].default_value = 0.
        dust.inputs['Roughness'].default_value = .86
        dust.inputs['Specular IOR Level'].default_value = .3
        assign(tree, relief(tree, 'Physical dust depth in metres', grains, original_normal,
                           recipe['dust_depth_m'] / size, .12), dust.inputs['Normal'])
        surface = layer_mix(tree, 'Physical one deposition layer', surface, dust.outputs[0], powder)
    assign(tree, surface, output.inputs['Surface'])
    write_surface_pass(tree, 'physical_layers', (rust, abrasion, powder))
    write_surface_pass(tree, 'generated_finish', (rust, abrasion, powder))
    metal_region = math_node(tree, 'ADD', metal, nonferrous, clamp=True)
    classified = math_node(tree, 'ADD', metal_region, paint, clamp=True)
    unknown_region = math_node(tree, 'SUBTRACT', 1., classified, clamp=True)
    write_surface_pass(tree, 'physical_regions', (metal_region, paint, unknown_region))
    shader['physical_layers'] = True
    shader['physical_layers_signature'] = finish['signature']
    shader['physical_layers_stage'] = stage
    return True
