"""Blender-only surface layers for a cached local scrap material variant."""
import re

import bpy


REGION_COLORS = ((.10,.38,.82),(.78,.32,.07),(.55,.16,.06),(.13,.64,.25))


def assign(tree,source,destination):
    if isinstance(source,bpy.types.NodeSocket):
        tree.links.new(source,destination)
    else:
        destination.default_value = source


def original(socket):
    return socket.links[0].from_socket if socket.is_linked else socket.default_value


def math_node(tree,operation,a,b,clamp=False):
    node = tree.nodes.new('ShaderNodeMath')
    node.operation = operation
    node.use_clamp = clamp
    assign(tree,a,node.inputs[0])
    assign(tree,b,node.inputs[1])
    return node.outputs[0]


def color_mix(tree,factor,a,b,mode='MIX'):
    node = tree.nodes.new('ShaderNodeMixRGB')
    node.blend_type = mode
    assign(tree,factor,node.inputs[0])
    for source,socket in zip((a,b),node.inputs[1:3]):
        # A source roughness may be a scalar instead of a texture connection.
        value = (source,source,source,1.) if isinstance(source,(int,float)) else source
        assign(tree,value,socket)
    return node.outputs[0]


def image_node(tree,path,name):
    image = bpy.data.images.load(path,check_existing=True)
    image.colorspace_settings.name = 'Non-Color'
    image.alpha_mode = 'CHANNEL_PACKED'
    image['scrap_texture'] = True
    node = tree.nodes.new('ShaderNodeTexImage')
    node.image = image
    node.label = name
    return node


def aov(tree,name,source,channel='Value'):
    node = tree.nodes.new('ShaderNodeOutputAOV')
    node.aov_name = name
    assign(tree,source,node.inputs[channel])


def apply_surface(material,entry,profile,request):
    if not material.use_nodes:
        return
    tree = material.node_tree
    output = next((n for n in tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
    if output is None or not output.inputs['Surface'].is_linked:
        return
    shader = output.inputs['Surface'].links[0].from_node
    if shader.type!='BSDF_PRINCIPLED':
        return
    geometry = tree.nodes.new('ShaderNodeNewGeometry')
    up = tree.nodes.new('ShaderNodeVectorMath')
    up.operation = 'DOT_PRODUCT'
    up.inputs[1].default_value = (0,0,1)
    tree.links.new(geometry.outputs['True Normal'],up.inputs[0])
    upward = up.outputs['Value']
    aov(tree,'upward',upward)
    aov(tree,'metallic',original(shader.inputs['Metallic']))
    alpha = shader.inputs['Alpha']
    transmission = shader.inputs['Transmission Weight']
    if alpha.is_linked or alpha.default_value<.999 or transmission.is_linked or transmission.default_value>.02:
        return
    if not entry.get('mask') and not entry.get('constant'):
        return
    if entry.get('mask'):
        mask = image_node(tree,entry['mask'],'Original UV material partitions')
        rgb = tree.nodes.new('ShaderNodeSeparateColor')
        tree.links.new(mask.outputs['Color'],rgb.inputs['Color'])
        weights = [rgb.outputs[name] for name in ('Red','Green','Blue')]+[mask.outputs['Alpha']]
    else:
        weights = entry['constant']

    tile = image_node(tree,profile['texture'],'Texture from nearby scrap statistics')
    tile.projection = 'BOX'
    tile.projection_blend = .28
    coordinates = tree.nodes.new('ShaderNodeTexCoord')
    scale = tree.nodes.new('ShaderNodeVectorMath')
    scale.operation = 'SCALE'
    scale.inputs['Scale'].default_value = request['surface_texture_scale']
    tree.links.new(coordinates.outputs['Object'],scale.inputs[0])
    tree.links.new(scale.outputs['Vector'],tile.inputs['Vector'])
    bands = tree.nodes.new('ShaderNodeSeparateColor')
    tree.links.new(tile.outputs['Color'],bands.inputs['Color'])

    confirmed = profile['status']=='estimated'
    detail = profile['detail_weight']
    shade = math_node(tree,'ADD',math_node(tree,'MULTIPLY',bands.outputs['Red'],detail*2),1-detail)
    base = original(shader.inputs['Base Color'])
    roughness = original(shader.inputs['Roughness'])
    region = (.06,.06,.06,1.)
    coverage = 0.
    exposed = 0.
    dust_weights = (1.,1.12,.79,.62)
    colors = ('metal','paint','rust')
    base_blend = (.62,.81,.52) if confirmed else (.18,.42,.20)
    rough_blend = (.69,.46,.43) if confirmed else (.26,.24,.24)
    roughness_targets = (profile['metal_roughness'],.61,.77)
    for index,weight in enumerate(weights):
        region = color_mix(tree,weight,region,(*REGION_COLORS[index],1.))
        coverage = math_node(tree,'ADD',coverage,math_node(tree,'MULTIPLY',weight,dust_weights[index]))
        if index==3:
            continue
        tint = color_mix(tree,1.,(*profile['palette'][colors[index]],1.),shade,'MULTIPLY')
        base = color_mix(tree,math_node(tree,'MULTIPLY',weight,base_blend[index]),base,tint)
        texture_roughness = math_node(tree,'ADD',math_node(tree,'MULTIPLY',bands.outputs['Green'],.12),
            roughness_targets[index]-.06+request['roughness_delta'],clamp=True)
        roughness = color_mix(tree,math_node(tree,'MULTIPLY',weight,rough_blend[index]),roughness,texture_roughness)
        exposed = math_node(tree,'ADD',exposed,weight,clamp=True)
    assign(tree,base,shader.inputs['Base Color'])
    assign(tree,roughness,shader.inputs['Roughness'])
    micro = tree.nodes.new('ShaderNodeBump')
    micro.inputs['Distance'].default_value = .0014
    assign(tree,math_node(tree,'MULTIPLY',exposed,detail*.92),micro.inputs['Strength'])
    tree.links.new(bands.outputs['Green'],micro.inputs['Height'])
    if shader.inputs['Normal'].is_linked:
        assign(tree,original(shader.inputs['Normal']),micro.inputs['Normal'])
    tree.links.new(micro.outputs['Normal'],shader.inputs['Normal'])

    cavity = tree.nodes.new('ShaderNodeAmbientOcclusion')
    cavity.samples = 4
    cavity.only_local = True
    cavity.inputs['Distance'].default_value = .022
    recess = math_node(tree,'SUBTRACT',1.,cavity.outputs['AO'],clamp=True)
    top = math_node(tree,'MAXIMUM',upward,0.)
    settle = math_node(tree,'ADD',math_node(tree,'MULTIPLY',top,.66),
        math_node(tree,'ADD',.24,math_node(tree,'MULTIPLY',recess,.48)),clamp=True)
    mottled = math_node(tree,'ADD',math_node(tree,'MULTIPLY',bands.outputs['Blue'],.88),.50)
    amount = min(.83,profile['coating_strength']*request['dust_control']*request['dust_gain'])
    dust = math_node(tree,'MULTIPLY',math_node(tree,'MULTIPLY',settle,mottled),
        math_node(tree,'MULTIPLY',coverage,amount),clamp=True)
    powder = tree.nodes.new('ShaderNodeBsdfPrincipled')
    powder.label = 'Deposited dust'
    powder.inputs['Metallic'].default_value = 0.
    powder.inputs['Roughness'].default_value = .89
    powder.inputs['Specular IOR Level'].default_value = .20
    powder.inputs['Emission Strength'].default_value = 0.
    dust_tint = color_mix(tree,1.,(*profile['palette']['dust'],1.),shade,'MULTIPLY')
    tree.links.new(dust_tint,powder.inputs['Base Color'])
    tree.links.new(micro.outputs['Normal'],powder.inputs['Normal'])
    layer = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(dust,layer.inputs[0])
    tree.links.new(shader.outputs['BSDF'],layer.inputs[1])
    tree.links.new(powder.outputs['BSDF'],layer.inputs[2])
    tree.links.new(layer.outputs[0],output.inputs['Surface'])
    aov(tree,'surface_dust',dust)
    aov(tree,'material_region',region,'Color')


def configure_scrap_materials(obj,request,state):
    profile = request['environment']['scrap_surface']
    key = (profile['variant_signature'],request['surface_texture_scale'],request['roughness_delta'],
           request['dust_control'],request['dust_gain'])
    if state.get('scrap_material_key')==key:
        return
    retired = {slot.material for slot in obj.material_slots if slot.link=='OBJECT' and slot.material}
    for slot in obj.material_slots:
        slot.link = 'DATA'
    for material in retired:
        if material.users==0 and material.get('scrap_variant'):
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users==0 and image.get('scrap_texture'):
            bpy.data.images.remove(image)
    copies = {}
    for slot in obj.material_slots:
        material = slot.material
        if material is None:
            continue
        if material not in copies:
            variant = material.copy()
            variant.use_fake_user = False
            variant['scrap_variant'] = True
            entries = profile['materials']
            entry = entries.get(material.name,entries.get(re.sub(r'\.\d{3}$','',material.name),{}))
            apply_surface(variant,entry,profile,request)
            copies[material] = variant
        slot.link = 'OBJECT'
        slot.material = copies[material]
    state['scrap_material_key'] = key
