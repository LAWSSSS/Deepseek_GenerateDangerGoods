"""Object-space surface stacks on exact named parts, with the source UV retained."""
import math
import re

import bpy

from .component_pbr import AOV, VERSION, slot_name
from .generated_shader import add, channels, mul, ramp, relief, surface_noise, write_surface_pass
from .physical_layers import layer_mix
from .reference_shader import bounded, reference_texture
from .scanned_shader import assign, math_node, mix, opaque_surface_output, original, texture


def assign_slots(objects, finish):
    """Set new slots before joining; faces, UVs and authored normals stay intact."""
    regions = [region for row in finish['regions']['materials'].values() for region in row['components']]
    by_name = {obj.name: obj for obj in objects}
    if len(by_name) != len(objects):
        raise ValueError('duplicate imported part names')
    for region in regions:
        expected = region['material']
        copy = None
        for name in region['parts']:
            obj = by_name.get(name)
            if (obj is None or len(obj.data.materials) != 1 or obj.data.materials[0] is None
                    or re.sub(r'\.\d{3}$', '', obj.data.materials[0].name) != expected):
                raise ValueError(f'Imported part differs from the audited material: {name} / {expected}')
            if copy is None:
                copy = obj.data.materials[0].copy()
                copy.name = slot_name(region)
                copy['component_pbr_region'] = region['id']
            if obj.data.users > 1:
                obj.data = obj.data.copy()
            obj.data.materials[0] = copy


def package_fields(tree, coordinate, spec, seed, tag):
    period = tree.nodes.new('ShaderNodeVectorMath')
    period.operation = 'SCALE'
    assign(tree, coordinate, period.inputs[0])
    period.inputs['Scale'].default_value = 1. / spec['tile_m']
    shifted = tree.nodes.new('ShaderNodeVectorMath')
    shifted.operation = 'ADD'
    assign(tree, period.outputs[0], shifted.inputs[0])
    shifted.inputs[1].default_value = seed
    base = reference_texture(tree, spec['maps']['albedo'], shifted.outputs[0],
                             'Component ' + tag + ' base color', color=True)
    # Use all three data channels from the same aligned texture lookup.
    data = texture(tree, spec['maps']['surface']['path'], shifted.outputs[0], 'Component ' + tag + ' surface')
    data.name = 'Component ' + tag + ' surface'
    data['reference_sha256'] = spec['maps']['surface']['sha256']
    data.projection = 'BOX'
    data.projection_blend = .24
    roughness, height, wear = channels(tree, data.outputs['Color'])
    luminance = tree.nodes.new('ShaderNodeRGBToBW')
    assign(tree, base, luminance.inputs[0])
    ratio = bounded(tree, math_node(tree, 'DIVIDE', luminance.outputs[0], spec['albedo_median_luma']), .45, 1.65)
    variant = mix(tree, 1., tuple(spec['anchor_linear']) + (1.,), ratio, blend='MULTIPLY')
    colour = mix(tree, spec['albedo_variation'], tuple(spec['anchor_linear']) + (1.,), variant)
    if spec['role'] == 'oxide':
        # Retain colour diversity inside oxide, after the preparation step
        # samples the scan's brown pixels and removes non-rust donor colour.
        colour = base
    return {'colour': colour, 'rough': roughness, 'height': height, 'wear': wear}


def machining_highlights(tree, principled, profile):
    """Align conductive reflection with the part. Taper to isotropic on end caps."""
    transform = tree.nodes.new('ShaderNodeVectorTransform')
    transform.name = 'Component machining axis'
    transform.vector_type, transform.convert_from, transform.convert_to = 'VECTOR', 'OBJECT', 'WORLD'
    transform.inputs['Vector'].default_value = profile['axis']
    axis = tree.nodes.new('ShaderNodeVectorMath')
    axis.operation = 'NORMALIZE'
    assign(tree, transform.outputs[0], axis.inputs[0])
    geometry = tree.nodes.new('ShaderNodeNewGeometry')
    tangent = tree.nodes.new('ShaderNodeVectorMath')
    tangent.name = 'Component circumferential tangent'
    tangent.operation = 'CROSS_PRODUCT'
    assign(tree, axis.outputs[0], tangent.inputs[0])
    assign(tree, geometry.outputs['Normal'], tangent.inputs[1])
    length = tree.nodes.new('ShaderNodeVectorMath')
    length.operation = 'LENGTH'
    assign(tree, tangent.outputs[0], length.inputs[0])
    taper = ramp(tree, length.outputs['Value'], .12, 3.)
    assign(tree, mul(tree, taper, profile['anisotropy']), principled.inputs['Anisotropic'])
    assign(tree, tangent.outputs[0], principled.inputs['Tangent'])


def oxide_spread_field(tree, donor_mask, coarse, cavity, coating_wear):
    """Extend oxide in mottled patches, with paint breach and recess bias."""
    brown = ramp(tree, donor_mask, .045, 3.6)
    clusters = ramp(tree, add(tree, coarse, mul(tree, cavity, .47)), .34, 3.9)
    spread = mul(tree, brown, clusters)
    if coating_wear is not None:
        breaches = mul(tree, ramp(tree, coating_wear, .025, 3.8), 1.34)
        recesses = mul(tree, cavity, .80)
        spots = mul(tree, ramp(tree, coarse, .445, 4.), .80)
        spread = mul(tree, spread, bounded(tree, add(tree, add(tree, breaches, recesses), spots)))
    return spread


def response(tree, tag, fields, spec, size_m, source_normal, controls, *, region=None, film_loss=None,
             scuff=None):
    region = region or {}
    name = f'Component PBR {tag}'
    principled = tree.nodes.new('ShaderNodeBsdfPrincipled')
    principled.name = name
    principled['display_deposit'] = True
    principled['component_pbr_role'] = spec['role']
    metallic = 1. if spec['role'] in ('steel', 'nonferrous') else 0.
    principled.inputs['Metallic'].default_value = metallic
    assign(tree, fields['colour'], principled.inputs['Base Color'])
    roughness = fields['rough']
    if 'parent_roughness' in region:
        roughness = add(tree, mul(tree, roughness, .4), .6 * region['parent_roughness'])
    assign(tree, bounded(tree, add(tree, roughness, controls.get('roughness_shift', 0.)), .08, .95),
           principled.inputs['Roughness'])
    height = math_node(tree, 'MULTIPLY', math_node(tree, 'SUBTRACT', fields['height'], .5), 2.)
    gain = min(1., .85 * region.get('relief_gain', 1.) * controls.get('relief_gain', 1.))
    normal = relief(tree, 'Physical ' + tag + ' library microheight', height, source_normal,
                    spec['relief_um'] * 1e-6 / size_m, gain)
    if film_loss is not None:
        chip, depth = film_loss
        normal = relief(tree, 'Physical coating edge relief', chip, normal, depth * 1e-6 / size_m, .73)
    assign(tree, normal, principled.inputs['Normal'])
    principled.inputs['IOR'].default_value = {'paint': 1.49, 'rubber': 1.48, 'plastic': 1.47, 'oxide': 1.48}.get(spec['role'], 1.5)
    if spec['role'] == 'paint':
        coat = .055 if scuff is None else mul(tree, .055, math_node(tree, 'SUBTRACT', 1., mul(tree, scuff, .68)))
        assign(tree, coat, principled.inputs['Coat Weight'])
        assign(tree, bounded(tree, add(tree, roughness, .04), .22, .8), principled.inputs['Coat Roughness'])
        assign(tree, normal, principled.inputs['Coat Normal'])
    if profile := region.get('machining'):
        if profile['anisotropy'] > 0.:
            machining_highlights(tree, principled, profile)
    return principled.outputs[0], metallic, normal


def finish_surface(material, shader, finish, strength):
    region_id = material.get('component_pbr_region')
    stage = finish.get('layer_stage', 'full')
    if stage not in ('source', 'wear', 'full'):
        raise ValueError('Unknown component material layer stage')
    if not region_id or stage == 'source':
        return False
    tree = material.node_tree
    if shader.get('component_pbr_version'):
        if shader.get('physical_layers_signature') != finish['signature']:
            raise ValueError('Component package changed on an already configured asset')
        return True
    output = opaque_surface_output(tree, shader)
    if output is None or shader.get('physical_layers') or shader.get('generated_finish'):
        raise ValueError('component packages require the original opaque source shader')
    region = next(row for entry in finish['regions']['materials'].values() for row in entry['components']
                  if row['id'] == region_id)
    packs = finish['reference']['packages']
    size_m = float(finish['size_m'])
    controls = finish.get('controls', {})
    if not math.isfinite(size_m) or size_m <= 0 or not 0 <= float(strength) <= .5:
        raise ValueError('component material size or dust limit is invalid')
    coords = tree.nodes.new('ShaderNodeTexCoord')
    metric = tree.nodes.new('ShaderNodeVectorMath')
    metric.operation = 'SCALE'
    assign(tree, coords.outputs['Object'], metric.inputs[0])
    metric.inputs['Scale'].default_value = size_m
    point = metric.outputs[0]
    mesh_hash = finish['source_mesh_hash']
    offset = tuple(int(mesh_hash[i:i+4], 16) / 65535. + region_id * .13 for i in (0, 4, 8))
    spec = packs[region['package']]
    pack = package_fields(tree, point, spec, offset, 'body')
    keeps_normal = spec['role'] in ('rubber', 'plastic') or region.get('keep_source_normal', False)
    source_normal = original(shader.inputs['Normal']) if keeps_normal and shader.inputs['Normal'].is_linked else None

    paint, rust = 0., 0.
    # Independent weathering fields. A coarse asset-fixed field keeps the wear
    # clustered and exposes intact areas between donor scratches and oxide.
    coarse = surface_noise(tree, point, 19., 2.)
    wear = region.get('wear', 1.)
    cavity = 0.
    if 'oxide' in region or (strength and spec['role'] in ('steel', 'paint')):
        ao = tree.nodes.new('ShaderNodeAmbientOcclusion')
        ao.only_local = True
        ao.samples = 4
        ao.inputs['Distance'].default_value = .008 / size_m
        cavity = math_node(tree, 'SUBTRACT', 1., ao.outputs['AO'])
    film, scuff = None, None
    if 'substrate' in region:
        scratches = surface_noise(tree, point, 34., 2.)
        distribution = ramp(tree, scratches, .28, 2.3)
        paint = mul(tree, ramp(tree, pack['wear'], .38, 2.3), distribution)
        paint = bounded(tree, mul(tree, paint, wear * controls.get('scratch_gain', 1.)), 0., .94)
        if scuff_strength := region.get('scuff_strength', 0.):
            # Shallow handling marks change the surviving coating's response.
            # The existing chip threshold still controls full paint loss.
            trace = mul(tree, ramp(tree, pack['wear'], .025, 3.1), distribution)
            coating = math_node(tree, 'SUBTRACT', 1., paint)
            scuff = mul(tree, mul(tree, trace, coating), scuff_strength)
            scuff.node.name = 'Component shallow paint scuff'
            pack['rough'] = add(tree, pack['rough'], mul(tree, scuff, .26))
        if region.get('paint_loss_um', 0.):
            film = (math_node(tree, 'SUBTRACT', 1., paint), region['paint_loss_um'])
    if spec['role'] in ('paint', 'rubber', 'plastic'):
        # Intact enamel and polymers still have variation across larger areas
        # than the fine donor grain. Only specular roughness is varied here.
        change = mul(tree, math_node(tree, 'SUBTRACT', coarse, .5), .15)
        pack['rough'] = add(tree, pack['rough'], change)
    if source_mix := region.get('source_roughness_mix', 0.):
        authored = bounded(tree, original(shader.inputs['Roughness']), .39, .86)
        pack['rough'] = add(tree, mul(tree, pack['rough'], 1. - source_mix), mul(tree, authored, source_mix))
        pack['rough'].node.name = 'Component source polymer roughness blend'
    surface, metallic, normal = response(tree, spec['role'], pack, spec, size_m, source_normal, controls,
                                         region=region, film_loss=film, scuff=scuff)
    if 'substrate' in region:
        underneath = packs[region['substrate']]
        under_pack = package_fields(tree, point, underneath, offset, 'substrate')
        under_surface, _, _ = response(tree, 'exposed steel', under_pack, underneath, size_m, None, controls,
                                       film_loss=film)
        surface = layer_mix(tree, 'Component paint loss and exposed steel', surface, under_surface, paint)
        metallic = paint
    if 'oxide' in region:
        oxidation = packs[region['oxide']]
        oxide_pack = package_fields(tree, point, oxidation, (offset[2] + .41, offset[0], -offset[1]), 'oxide')
        rust_surface, _, _ = response(tree, 'iron oxide', oxide_pack, oxidation, size_m, None, controls)
        active = ramp(tree, add(tree, coarse, mul(tree, cavity, .24)), .41, 3.5)
        oxide_cue = mul(tree, oxide_pack['wear'], active)
        if 'substrate' in region:
            # Old rust grows mostly around breaches; an extra sparse field
            # allows small under-film spots without treating paint as steel.
            edge = add(tree, mul(tree, paint, 1.8), mul(tree, cavity, .32))
            rim = mul(tree, ramp(tree, coarse, .57, 4.5), .94)
            oxide_cue = mul(tree, oxide_cue, bounded(tree, add(tree, edge, rim)))
        if amount := region.get('oxide_spread', 0.):
            coating_wear = pack['wear'] if 'substrate' in region else None
            expanded = oxide_spread_field(tree, oxide_pack['wear'], coarse, cavity, coating_wear)
            oxide_cue = math_node(tree, 'MAXIMUM', oxide_cue, mul(tree, expanded, amount))
        rust = bounded(tree, mul(tree, oxide_cue, wear * controls.get('oxide_gain', 1.)), 0., .98)
        surface = layer_mix(tree, 'Component dielectric iron oxide', surface, rust_surface, rust)
        metallic = mul(tree, metallic, math_node(tree, 'SUBTRACT', 1., rust))

    # One physically separate deposit. The local normal belongs to the asset;
    # turning the model or moving the lights cannot move its wear pattern.
    dust = 0.
    if strength and stage == 'full' and spec['role'] in ('steel', 'paint'):
        geo = tree.nodes.new('ShaderNodeNewGeometry')
        local = tree.nodes.new('ShaderNodeVectorTransform')
        local.vector_type, local.convert_from, local.convert_to = 'NORMAL', 'WORLD', 'OBJECT'
        assign(tree, geo.outputs['Normal'], local.inputs['Vector'])
        axes = tree.nodes.new('ShaderNodeSeparateXYZ')
        assign(tree, local.outputs[0], axes.inputs[0])
        shelter = add(tree, mul(tree, cavity, .54), mul(tree, bounded(tree, axes.outputs['Z']), .30))
        dust = mul(tree, mul(tree, shelter, ramp(tree, coarse, .36, 2.3)), float(strength) * (.5 if spec['role'] == 'paint' else 1.))
        deposit = tree.nodes.new('ShaderNodeBsdfPrincipled')
        deposit.name = 'Component fine mineral deposit'
        deposit['display_deposit'] = True
        deposit['component_pbr_role'] = 'dust'
        deposit.inputs['Base Color'].default_value = (.16, .145, .126, 1.)
        deposit.inputs['Roughness'].default_value = .87
        assign(tree, normal, deposit.inputs['Normal'])
        surface = layer_mix(tree, 'Component local powder', surface, deposit.outputs[0], dust)
        metallic = mul(tree, metallic, math_node(tree, 'SUBTRACT', 1., dust))

    coverage = 1.
    if 'print_mask' in region:
        mask = texture(tree, region['print_mask']['path'], coords.outputs['UV'], 'Component protected printing')
        mask.name = 'Component protected printing'
        mask.extension = 'REPEAT'
        coverage = math_node(tree, 'SUBTRACT', 1., channels(tree, mask.outputs['Color'])[0])
    surface = layer_mix(tree, 'Component preserve source printing', shader.outputs[0], surface, coverage)
    assign(tree, surface, output.inputs['Surface'])
    metal_aov = tree.nodes.get('batch_metallic_aov')
    if metal_aov is not None:
        original_part = mul(tree, original(shader.inputs['Metallic']), math_node(tree, 'SUBTRACT', 1., coverage))
        assign(tree, bounded(tree, add(tree, original_part, mul(tree, coverage, metallic))), metal_aov.inputs['Value'])
    wear_fields = tuple(mul(tree, coverage, factor) for factor in (rust, paint, dust))
    write_surface_pass(tree, 'generated_finish', wear_fields)
    write_surface_pass(tree, 'physical_layers', wear_fields)
    write_surface_pass(tree, 'physical_regions', (
        mul(tree, coverage, 1. if spec['role'] in ('steel', 'nonferrous') else 0.),
        mul(tree, coverage, 1. if spec['role'] == 'paint' else 0.),
        math_node(tree, 'SUBTRACT', 1., coverage)))
    write_surface_pass(tree, AOV, (
        mul(tree, coverage, 1. if spec['role'] == 'rubber' else 0.),
        mul(tree, coverage, 1. if spec['role'] == 'plastic' else 0.),
        math_node(tree, 'SUBTRACT', 1., coverage)))
    shader['component_pbr_version'] = VERSION
    shader['component_pbr_region'] = region_id
    shader['physical_layers'] = True
    shader['physical_layers_signature'] = finish['signature']
    return True
