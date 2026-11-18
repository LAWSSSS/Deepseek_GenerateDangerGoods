"""Asset-fixed material reconstruction from estimated reference texture channels.

The source shader remains untouched and is used for protected/unknown UV regions.
Known body regions receive a new base response: source identity colour, reference
texture variation and estimated roughness. Photograph highlights are not normals.
No background, camera or atmosphere state is accepted by this module.
"""
import math
import re

import bpy

from .generated_shader import add, channels, mul, relief, write_surface_pass
from .physical_layers import clone_source, layer_mix
from .scanned_shader import assign, math_node, mix, opaque_surface_output, original, texture


VERSION = 'reference-shader-v1'


def bounded(tree, value, low=0., high=1.):
    return math_node(tree, 'MAXIMUM', math_node(tree, 'MINIMUM', value, high), low)


def _linear_color(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('reference identity requires three linear RGB components')
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in result):
        raise ValueError('reference identity is outside linear RGB range')
    return result + (1.,)


def identity_color(tree, shader, entry, metal, paint, third):
    """Use measured identity pigments, never the source's baked shading field."""
    regions = entry.get('source_identity_by_region')
    if regions:
        names = ('metal', 'paint', 'nonferrous' if entry.get('authored_atlas') else 'existing_oxide')
        fallback = entry.get('source_identity_linear', next(iter(regions.values())))
        coverage = entry.get('coverage', [0., 0., 0.])
        for index, name in enumerate(names):
            if index < len(coverage) and coverage[index] > 0 and name not in regions:
                raise ValueError('reference source identity missing known region: ' + name)
        colors = [_linear_color(regions.get(name, fallback)) for name in names]
        result = colors[0]
        result = mix(tree, paint, result, colors[1])
        return mix(tree, third, result, colors[2])
    if entry.get('source_identity_linear') is not None:
        return _linear_color(entry['source_identity_linear'])
    if not shader.inputs['Base Color'].is_linked:
        return tuple(shader.inputs['Base Color'].default_value)
    raise ValueError('reference body requires measured source identity; linked source colour is not a pigment')


def reference_texture(tree, entry, coordinate, name, *, color=False):
    """Separate colour/data image datablocks prevent shared-path colour-space edits."""
    node = tree.nodes.new('ShaderNodeTexImage')
    node.name = name
    path = bpy.path.abspath(entry['path'])
    space = 'sRGB' if color else 'Non-Color'
    image = next((image for image in bpy.data.images
                  if bpy.path.abspath(image.filepath) == path and image.colorspace_settings.name == space), None)
    if image is None:
        image = bpy.data.images.load(path, check_existing=False)
        image.colorspace_settings.name = space
    node.image = image
    node.projection = 'BOX'
    node.projection_blend = .28
    node.extension = 'REPEAT'
    node.interpolation = 'Linear'
    node['reference_sha256'] = entry.get('sha256', '')
    assign(tree, coordinate, node.inputs['Vector'])
    return node.outputs['Color'] if color else channels(tree, node.outputs['Color'])[0]


def finish_surface(material, shader, finish, strength):
    reference = finish['reference']
    if finish.get('version') != 'physical-layers-v1' or reference.get('version') != 'reference-pbr-v1':
        raise ValueError('unsupported reference material protocol')
    if not reference.get('estimated'):
        raise ValueError('image-derived reference material must be marked estimated')
    size, scale = float(finish['size_m']), float(reference['scale_m'])
    amplitude = float(reference.get('height_amplitude_m', .00008))
    if not all(math.isfinite(value) and value > 0 for value in (size, scale)):
        raise ValueError('reference material requires positive metric dimensions')
    if not math.isfinite(amplitude) or not 0 <= amplitude <= .00008:
        raise ValueError('reference estimated height exceeds 80 micrometres')
    if not math.isfinite(float(strength)) or not 0 <= float(strength) <= .5:
        raise ValueError('reference dust must be between zero and .5')
    stage = finish.get('layer_stage', 'full')
    if stage not in ('source', 'wear', 'full'):
        raise ValueError('unknown reference material stage')
    tree = material.node_tree
    if shader.get('physical_layers'):
        if (shader.get('physical_layers_signature') != finish['signature']
                or shader.get('physical_layers_stage') != stage
                or shader.get('reference_material_version') != VERSION):
            raise ValueError('reference material changed; reimport source mesh')
        tree.nodes['Physical dust strength'].outputs[0].default_value = float(strength)
        return True
    name = re.sub(r'\.\d{3}$', '', material.name)
    entry = finish['regions']['materials'].get(name)
    if stage == 'source' or not entry or not entry.get('eligible') or entry.get('protected'):
        return False
    output = opaque_surface_output(tree, shader)
    if output is None:
        return False
    if shader.get('generated_finish'):
        raise ValueError('reference rebuild must start from the source material')
    required = {'albedo', 'roughness', 'oxide', 'abrasion', 'height', 'dust'}
    if set(reference['maps']) != required:
        raise ValueError('reference material channel set mismatch')
    controls = finish.get('controls', {})
    oxide_gain = float(controls.get('oxide_gain', 1.))
    scratch_gain = float(controls.get('scratch_gain', 1.))
    roughness_shift = float(controls.get('roughness_shift', 0.))
    relief_gain = float(controls.get('relief_gain', 1.))
    coordinate = tree.nodes.new('ShaderNodeTexCoord')
    if 'mask' in entry:
        mask = texture(tree, entry['mask'], coordinate.outputs['UV'], 'Physical known source regions')
        mask.name = 'Physical known source regions'
        mask.interpolation = 'Closest'
        metal, paint, third = channels(tree, mask.outputs['Color'])
    else:
        metal, paint, third = entry['constant'][:3]
    # Authored RGB is iron/paint/nonferrous; inferred RGB is metal/paint/rust.
    # These protocols cannot be interchanged, especially on aluminium cylinders.
    authored = bool(entry.get('authored_atlas'))
    ferrous = metal if authored or entry.get('metal_family') == 'ferrous' else 0.
    oxidizable_paint = paint if authored or entry.get('metal_family') == 'ferrous' else 0.
    existing_oxide = third if not authored and entry.get('metal_family') == 'ferrous' else 0.
    known = bounded(tree, add(tree, add(tree, metal, paint), third))
    ferrous_domain = bounded(tree, add(tree, add(tree, ferrous, oxidizable_paint), existing_oxide))
    identity = identity_color(tree, shader, entry, metal, paint, third)
    metric = tree.nodes.new('ShaderNodeVectorMath')
    metric.name = 'Physical coordinates in metres'
    metric.operation = 'SCALE'
    assign(tree, coordinate.outputs['Object'], metric.inputs[0])
    metric.inputs['Scale'].default_value = size
    repeat = tree.nodes.new('ShaderNodeVectorMath')
    repeat.name = 'Reference physical texture period'
    repeat.operation = 'SCALE'
    assign(tree, metric.outputs[0], repeat.inputs[0])
    repeat.inputs['Scale'].default_value = 1. / scale
    shift = tree.nodes.new('ShaderNodeVectorMath')
    shift.name = 'Reference asset-fixed texture offset'
    shift.operation = 'ADD'
    assign(tree, repeat.outputs[0], shift.inputs[0])
    seed = finish['source_mesh_hash']
    shift.inputs[1].default_value = tuple(int(seed[i:i+4], 16) / 65535. for i in (0, 4, 8))
    maps = {key: reference_texture(tree, entry, shift.outputs[0], 'Reference ' + key, color=key == 'albedo')
            for key, entry in reference['maps'].items()}
    luminance = tree.nodes.new('ShaderNodeRGBToBW')
    assign(tree, maps['albedo'], luminance.inputs[0])
    mean = reference.get('albedo_mean_linear', [.18, .18, .18])
    mean_luma = max(.01, sum(float(v)*w for v, w in zip(mean, (.2126, .7152, .0722))))
    variation = bounded(tree, math_node(tree, 'DIVIDE', luminance.outputs[0], mean_luma), .45, 1.65)
    # Remove old baked highlights from the known body, preserving its colour class.
    body = clone_source(tree, shader)
    body.name = 'Reference body response'
    assign(tree, mix(tree, 1., identity, variation, blend='MULTIPLY'), body.inputs['Base Color'])
    assign(tree, bounded(tree, add(tree, maps['roughness'], roughness_shift), .08, .95), body.inputs['Roughness'])
    source_normal = original(shader.inputs['Normal']) if entry.get('source_normal_trusted') and shader.inputs['Normal'].is_linked else None
    height = math_node(tree, 'MULTIPLY', math_node(tree, 'SUBTRACT', maps['height'], .5), 2.)
    normal = relief(tree, 'Physical reference estimated microheight', height, source_normal,
                    amplitude / size, min(1., .35 * relief_gain))
    assign(tree, normal, body.inputs['Normal'])
    surface = layer_mix(tree, 'Reference known body replacement', shader.outputs[0], body.outputs[0], known)
    # Scratches expose iron only on confirmed painted iron. Nonferrous regions
    # retain their original metalness; plastic/labels are outside the known mask.
    abrasion_field = bounded(tree, mul(tree, maps['abrasion'], scratch_gain), 0., .95)
    abrasion = mul(tree, known, abrasion_field)
    scratched = clone_source(tree, body)
    scratched.name = 'Reference scratched response'
    assign(tree, bounded(tree, mul(tree, maps['roughness'], .72), .12, .72), scratched.inputs['Roughness'])
    bare = tree.nodes.new('ShaderNodeBsdfPrincipled')
    bare.name = 'Physical exposed steel'
    bare['display_deposit'] = True
    assign(tree, mix(tree, 1., (.37, .37, .36, 1.), variation, blend='MULTIPLY'), bare.inputs['Base Color'])
    bare.inputs['Metallic'].default_value = 1.
    assign(tree, bounded(tree, mul(tree, maps['roughness'], .65), .18, .55), bare.inputs['Roughness'])
    assign(tree, normal, bare.inputs['Normal'])
    abraded = layer_mix(tree, 'Reference coating abrasion', scratched.outputs[0], bare.outputs[0], oxidizable_paint)
    surface = layer_mix(tree, 'Physical bounded abrasion', surface, abraded, abrasion)
    oxide = mul(tree, ferrous_domain, bounded(tree, mul(tree, maps['oxide'], oxide_gain), 0., .95))
    rust = tree.nodes.new('ShaderNodeBsdfPrincipled')
    rust.name = 'Physical dielectric oxide'
    rust['display_deposit'] = True
    rust.inputs['Metallic'].default_value = 0.
    rust.inputs['Specular IOR Level'].default_value = .35
    assign(tree, maps['albedo'], rust.inputs['Base Color'])
    assign(tree, bounded(tree, add(tree, maps['roughness'], .08), .68, .96), rust.inputs['Roughness'])
    assign(tree, normal, rust.inputs['Normal'])
    surface = layer_mix(tree, 'Physical bounded oxidation', surface, rust.outputs[0], oxide)
    dust_strength = tree.nodes.new('ShaderNodeValue')
    dust_strength.name = 'Physical dust strength'
    dust_strength.outputs[0].default_value = float(strength)
    powder = 0.
    if stage == 'full':
        geometry = tree.nodes.new('ShaderNodeNewGeometry')
        local = tree.nodes.new('ShaderNodeVectorTransform')
        local.name = 'Physical asset-fixed deposition normal'
        local.vector_type = 'NORMAL'; local.convert_from = 'WORLD'; local.convert_to = 'OBJECT'
        assign(tree, geometry.outputs['Normal'], local.inputs['Vector'])
        up = tree.nodes.new('ShaderNodeSeparateXYZ')
        assign(tree, local.outputs['Vector'], up.inputs[0])
        ao = tree.nodes.new('ShaderNodeAmbientOcclusion')
        ao.name = 'Physical local cavity'; ao.only_local = True; ao.samples = 4
        ao.inputs['Distance'].default_value = .008 / size
        cavity = bounded(tree, math_node(tree, 'SUBTRACT', 1., ao.outputs['AO']))
        deposition = bounded(tree, add(tree, .10, add(tree, mul(tree, bounded(tree, up.outputs['Z']), .55), mul(tree, cavity, .35))))
        domain = bounded(tree, add(tree, add(tree, metal, third), mul(tree, paint, .5)))
        powder = mul(tree, mul(tree, dust_strength.outputs[0], domain), mul(tree, deposition, maps['dust']))
        dust = tree.nodes.new('ShaderNodeBsdfPrincipled')
        dust.name = 'Physical single dielectric deposit'; dust['display_deposit'] = True
        dust.inputs['Base Color'].default_value = _linear_color(reference.get('dust_color_linear', [.20, .185, .165]))
        dust.inputs['Metallic'].default_value = 0.
        dust.inputs['Roughness'].default_value = .86
        dust.inputs['Specular IOR Level'].default_value = .3
        assign(tree, normal, dust.inputs['Normal'])
        surface = layer_mix(tree, 'Physical one deposition layer', surface, dust.outputs[0], powder)
    assign(tree, surface, output.inputs['Surface'])
    # The worker creates this AOV from the source before material adaptation.
    # Rebind it to the actual layer mixture so rust/dust are not reported as
    # bare metal to later photographic processing.
    metallic_aov = tree.nodes.get('batch_metallic_aov')
    if metallic_aov is not None:
        source_metal = original(shader.inputs['Metallic'])
        scratched_metal = add(tree, source_metal,
                              mul(tree, oxidizable_paint, math_node(tree, 'SUBTRACT', 1., source_metal)))
        effective_metal = add(tree, mul(tree, source_metal, math_node(tree, 'SUBTRACT', 1., abrasion)),
                              mul(tree, scratched_metal, abrasion))
        effective_metal = mul(tree, mul(tree, effective_metal, math_node(tree, 'SUBTRACT', 1., oxide)),
                              math_node(tree, 'SUBTRACT', 1., powder))
        assign(tree, bounded(tree, effective_metal), metallic_aov.inputs['Value'])
        metallic_aov['basis'] = 'reference_layer_weighted_material_metalness'
    write_surface_pass(tree, 'physical_layers', (oxide, abrasion, powder))
    write_surface_pass(tree, 'generated_finish', (oxide, abrasion, powder))
    metal_region = bounded(tree, add(tree, metal, third if authored else 0.))
    write_surface_pass(tree, 'physical_regions', (metal_region, paint, bounded(tree, math_node(tree, 'SUBTRACT', 1., known))))
    shader['physical_layers'] = True
    shader['physical_layers_signature'] = finish['signature']
    shader['physical_layers_stage'] = stage
    shader['reference_material_version'] = VERSION
    shader['reference_estimated_channels'] = True
    shader['source_normal_preserved_on_body'] = bool(source_normal)
    return True
