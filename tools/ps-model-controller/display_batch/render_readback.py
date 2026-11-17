"""Read the constructed Blender graph and lamps, never echo requested controls.

Texture-domain fractions describe UV texels, not screen or world surface area.
Linked BRDF values remain fields; they are not collapsed into a fictitious scalar.
"""
import hashlib
import json
import math

import numpy as np

VERSION = 'blender-node-readback-v1'


def _value(value):
    if isinstance(value,(float,int,bool,str)): return value
    try: return [float(v) for v in value]
    except TypeError: return None


def socket_state(socket, depth=2):
    if not socket.is_linked:
        return {'kind':'constant','value':_value(socket.default_value)}
    link=socket.links[0];node=link.from_node
    record={'kind':'field','node':node.name,'type':node.type,'output':link.from_socket.name}
    if node.type=='MATH': record['operation']=node.operation
    if node.type in ('TEX_IMAGE','TEX_ENVIRONMENT') and node.image:
        record.update(image=node.image.name,size=list(node.image.size),color_space=node.image.colorspace_settings.name)
        if node.type=='TEX_ENVIRONMENT':
            pixels=np.empty(len(node.image.pixels),np.float32)
            node.image.pixels.foreach_get(pixels)
            record['pixel_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
            record['mean_rgb_linear']=[float(v) for v in pixels.reshape(-1,4)[:,:3].mean(axis=0)]
    if depth>0:
        record['inputs']=[{'socket':item.name,**socket_state(item,depth-1)} for item in node.inputs
                          if getattr(item,'enabled',True) and hasattr(item,'default_value')]
    return record


def _domain(socket, images, visiting=None):
    """Evaluate classification-only socket trees; procedural/noise nodes return unknown."""
    visiting=set() if visiting is None else set(visiting)
    if not socket.is_linked:
        return float(socket.default_value) if isinstance(socket.default_value,(float,int)) else np.asarray(socket.default_value)
    link=socket.links[0];node=link.from_node
    if node.name in visiting:return None
    visiting.add(node.name)
    if node.type=='TEX_IMAGE' and node.image:
        key=node.image.as_pointer()
        if key not in images:
            pixels=np.empty(len(node.image.pixels),np.float32)
            node.image.pixels.foreach_get(pixels)
            images[key]=pixels.reshape(-1,4)[:,:3]
        return images[key]
    if node.type in ('SEPARATE_COLOR','SEPRGB'):
        color=_domain(node.inputs[0],images,visiting)
        if color is None:return None
        channel=next((i for i,s in enumerate(node.outputs) if s==link.from_socket),None)
        return np.asarray(color)[...,channel] if channel is not None else None
    if node.type=='VALUE':return float(link.from_socket.default_value)
    if node.type!='MATH':return None
    a=_domain(node.inputs[0],images,visiting);b=_domain(node.inputs[1],images,visiting)
    if a is None or b is None:return None
    functions={'ADD':np.add,'SUBTRACT':np.subtract,'MULTIPLY':np.multiply,'MINIMUM':np.minimum,'MAXIMUM':np.maximum}
    if node.operation not in functions:return None
    result=functions[node.operation](a,b)
    return np.clip(result,0,1) if node.use_clamp else result


def _factor_domain(tree, layer_name, images):
    layer=tree.nodes.get(layer_name)
    if not layer or not layer.inputs[0].is_linked:return None
    # Both physical coverage formulas multiply a known material domain first.
    root=layer.inputs[0].links[0].from_node
    if root.type!='MATH' or root.operation!='MULTIPLY':return None
    domain=_domain(root.inputs[0],images)
    if domain is None:return None
    values=np.asarray(domain)
    return {'mean':float(np.clip(values,0,1).mean()),'nonzero_fraction':float((values>.001).mean()),
            'basis':'actual_shader_classification_socket_values; UV_texel_fraction_not_surface_area'}


def material_state(obj):
    rows=[];domains={key:False for key in ('oxide_gain','scratch_gain','roughness_shift','relief_gain')}
    images={}
    for material in obj.data.materials:
        if not material or not material.use_nodes:continue
        tree=material.node_tree
        source=[node for node in tree.nodes if node.type=='BSDF_PRINCIPLED' and not node.get('display_deposit')]
        applied=any(node.get('physical_layers') for node in source)
        known=_factor_domain(tree,'Physical bounded abrasion',images) if applied else None
        ferrous=_factor_domain(tree,'Physical bounded oxidation',images) if applied else None
        targets={key:bool(known and known['nonzero_fraction']>0) for key in domains}
        targets['oxide_gain']=bool(ferrous and ferrous['nonzero_fraction']>0)
        for key,value in targets.items():domains[key] |= value
        row={'material':material.name,'physical_layers_applied':applied,'known_domain':known,'oxidizable_domain':ferrous,
             'control_domains':targets,'source_brdf':[{ 'node':node.name,
                 **{key:socket_state(node.inputs[key],1) for key in ('Base Color','Metallic','Roughness','Normal')}} for node in source],
             'layer_factors':{},'response_brdf':[],'relief':[]}
        for name in ('Physical bounded oxidation','Physical bounded abrasion','Physical one deposition layer','Physical known-region roughness correction'):
            node=tree.nodes.get(name)
            if node:row['layer_factors'][name]=socket_state(node.inputs[0],4)
        for node in tree.nodes:
            if node.type=='BSDF_PRINCIPLED' and node.get('display_deposit'):
                row['response_brdf'].append({'node':node.name,
                    **{key:socket_state(node.inputs[key],2) for key in ('Metallic','Roughness','Normal')}})
            if node.type=='BUMP' and node.name.startswith('Physical '):
                row['relief'].append({'node':node.name,'distance_mesh_units':socket_state(node.inputs['Distance'],0),
                                      'strength':socket_state(node.inputs['Strength'],0)})
        rows.append(row)
    result={'version':VERSION,'basis':'actual_Blender_material_nodes_after_configuration',
            'materials':rows,'control_domains':domains,'scope':'whole_classified_regions_only',
            'limitations':['linked_BRDF_is_a_field_not_a_scalar','UV_domain_coverage_is_not_visible_coverage',
                           'part_specific_shader_controls_are_not_registered']}
    result['state_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def lighting_state(scene):
    lights=[]
    for obj in scene.objects:
        if obj.type!='LIGHT':continue
        lamp=obj.data
        lights.append({'name':obj.name,'type':lamp.type,'energy_W':float(lamp.energy),
                       'size_mesh_units':float(lamp.size) if lamp.type=='AREA' else None,
                       'color_linear':list(lamp.color),'position':list(obj.location),
                       'rotation_euler_rad':list(obj.rotation_euler),'hidden_render':bool(obj.hide_render)})
    world=[]
    if scene.world and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.type=='BACKGROUND':world.append({'color':socket_state(node.inputs['Color'],1),'strength':socket_state(node.inputs['Strength'],0)})
    active=[light for light in lights if light['type']=='AREA' and not light['hidden_render']
            and light['energy_W']>0 and max(light['color_linear'])>0]
    result={'version':VERSION,'basis':'actual_Blender_lamp_and_world_nodes_after_configuration',
            'lights':lights,'world':world,'control_domains':{'light_width_scale':len(active)>=2,'fill_gain':len(active)>=2},
            'limitations':['virtual_lamp_power_is_not_measured_scene_illumination','size_uses_normalized_scene_units']}
    result['state_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def compact_readback(record):
    """Keep measured values and confidence in prompts; retain full graphs in cache."""
    summary={key:record[key] for key in ('version','basis','state_sha256','control_domains','limitations') if key in record}
    if 'materials' in record:
        summary['scope']=record['scope']
        summary['materials']=[{
            'material':row['material'],'physical_layers_applied':row['physical_layers_applied'],
            'known_domain':row['known_domain'],'oxidizable_domain':row['oxidizable_domain'],
            'source_brdf':[{key:node[key] if node[key].get('kind')=='constant' else
                           {name:node[key][name] for name in ('kind','type','node') if name in node[key]}
                           for key in ('Metallic','Roughness','Normal')} for node in row['source_brdf']],
            'relief':row['relief'], 'layer_factors':{name:value['kind'] for name,value in row['layer_factors'].items()},
            'response_brdf':[{'node':node['node'],
                              'metallic':node['Metallic'],
                              'roughness':node['Roughness'] if node['Roughness']['kind']=='constant' else
                                  {'kind':'field','node':node['Roughness']['node']}}
                             for node in row['response_brdf']]}
            for row in record['materials']]
    else:
        summary.update(lights=record['lights'],world=record['world'])
    return summary
