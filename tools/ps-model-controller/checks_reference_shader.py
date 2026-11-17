"""Real Blender graph/round-trip and render-cache tests; no API or GPU render."""
import copy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from display_batch.common import BLENDER, HERE
from display_batch.rendering import part_keys


class ReferenceShaderTests(unittest.TestCase):
    def test_reference_only_invalidates_object_and_object_shadow(self):
        request = dict(windows={'object': {}, 'shadow': {}}, environment={}, appearance='native', camera={},
                       plane_normal=[0, 0, 1], light_gain=1., mesh_hash='mesh', matrix=[], yaw=0.,
                       roughness_delta=0., dust_gain=1., dust_control=.5, material_classes=['steel'],
                       generated_finish={'version': 'physical-layers-v1', 'signature': 'asset',
                                         'regions': {'signature': 'regions'},
                                         'reference': {'maps': {'albedo': {'sha256': 'first'}}}})
        initial = part_keys(request, 'CPU', 'fixture')
        request['generated_finish']['reference']['maps']['albedo']['sha256'] = 'second'
        changed = part_keys(request, 'CPU', 'fixture')
        self.assertNotEqual(initial['object'], changed['object'])
        self.assertNotEqual(initial['with_shadow'], changed['with_shadow'])
        self.assertEqual(initial['without_shadow'], changed['without_shadow'])

    @unittest.skipUnless(BLENDER.is_file(), 'Blender is not installed')
    def test_actual_reference_graph_protection_domains_and_roundtrip(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tile = folder / 'tile.png'
            Image.new('RGB', (16, 16), (100, 80, 130)).save(tile)
            script = folder / 'check.py'
            script.write_text('''import sys, json, copy
from pathlib import Path
import bpy
sys.path.insert(0, HERE)
from display_batch.physical_layers import finish_surface
from display_batch.render_readback import material_state,compact_readback
bpy.ops.wm.read_factory_settings(use_empty=True)

def source(name, metallic=.8, linked=False):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    node=mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Metallic'].default_value=metallic
    node.inputs['Base Color'].default_value=(.31,.12,.08,1.)
    node.inputs['Roughness'].default_value=.42
    image=mat.node_tree.nodes.new('ShaderNodeTexImage')
    image.image=bpy.data.images.load(TILE,check_existing=True)
    if linked:
        mat.node_tree.links.new(image.outputs['Color'],node.inputs['Base Color'])
        mat.node_tree.links.new(image.outputs['Color'],node.inputs['Roughness'])
    normal=mat.node_tree.nodes.new('ShaderNodeBump')
    mat.node_tree.links.new(normal.outputs['Normal'],node.inputs['Normal'])
    aov=mat.node_tree.nodes.new('ShaderNodeOutputAOV');aov.name='batch_metallic_aov';aov.aov_name='metallic'
    aov.inputs['Value'].default_value=metallic
    return mat,node,image,normal

def profile(name,weights,stage='full',size=.2,eligible=True,authored=True,family='partitioned'):
    return {'version':'physical-layers-v1','signature':'fixture-'+str(size)+'-'+stage,'source_mesh_hash':'abcdef12'*8,
      'size_m':size,'recipe':{},'layer_stage':stage,
      'reference':{'version':'reference-pbr-v1','estimated':True,'scale_m':.06,
        'height_amplitude_m':.00008,'maps':{key:{'path':TILE,'sha256':'fixture'}
         for key in ('albedo','roughness','oxide','abrasion','height','dust')}},
      'regions':{'materials':{name:{'coverage':weights,'constant':weights,'eligible':eligible,
       'source_identity_by_region':{key:[.31,.12,.08] for key in ('metal','paint','nonferrous','existing_oxide')},
       'authored_atlas':authored,'metal_family':family}}}}

mat,node,image,old_normal=source('paint',0.,True)
finish=profile('paint',[0,1,0])
old_links=[(link.from_node.name,link.from_socket.name,link.to_node.name,link.to_socket.name) for link in mat.node_tree.links]
old_image=image.image;old_space=old_image.colorspace_settings.name
assert finish_surface(mat,node,finish,{'dust_color':[1.,0.,0.]},.5)
tree=mat.node_tree
assert image.image==old_image and image.image.colorspace_settings.name==old_space
assert node.inputs['Base Color'].links[0].from_node==image
assert node.inputs['Roughness'].links[0].from_node==image
assert node.inputs['Normal'].links[0].from_node==old_normal
assert tree.nodes['Reference body response'].inputs['Base Color'].links[0].from_node!=image
assert not tree.nodes['Physical reference estimated microheight'].inputs['Normal'].is_linked
assert tree.nodes['Reference albedo'].image.colorspace_settings.name=='sRGB'
assert all(tree.nodes['Reference '+kind].image.colorspace_settings.name=='Non-Color' for kind in ('roughness','oxide','abrasion','height','dust'))
assert len({tree.nodes['Reference '+kind].inputs['Vector'].links[0].from_node.name for kind in ('albedo','roughness','oxide','abrasion','height','dust')})==1
assert abs(tree.nodes['Physical coordinates in metres'].inputs['Scale'].default_value-.2)<1e-6
assert abs(tree.nodes['Reference physical texture period'].inputs['Scale'].default_value-1/.06)<1e-6
assert abs(tree.nodes['Physical reference estimated microheight'].inputs['Distance'].default_value-.00008/.2)<1e-7
assert tree.nodes['Physical exposed steel'].inputs['Metallic'].default_value==1.
assert tree.nodes['Physical dielectric oxide'].inputs['Metallic'].default_value==0.
assert tree.nodes['Physical single dielectric deposit'].inputs['Metallic'].default_value==0.
assert len([n for n in tree.nodes if n.name=='Physical single dielectric deposit'])==1
bpy.ops.mesh.primitive_cube_add()
obj=bpy.context.object;obj.data.materials.clear();obj.data.materials.append(mat)
state=material_state(obj)
assert all(state['control_domains'].values()),state
assert state['materials'][0]['known_domain']['mean']==1.
assert state['materials'][0]['oxidizable_domain']['mean']==1.
reference_read=state['materials'][0]['reference_response']
assert reference_read['status']=='measured'
assert {row['node'] for row in reference_read['responses']}=={'Reference body response','Reference scratched response'}
assert all(row['connected_to_surface'] and row['Base Color']['kind']=='field' and row['Normal']['kind']=='field' for row in reference_read['responses'])
compact=compact_readback(state)['materials'][0]['reference_response']
assert compact['source_brdf_role']=='retained_original_source_shader_not_visible_reference_body'
assert compact['responses'][0]['Normal']['node']=='Physical reference estimated microheight'
assert any(item['socket']=='Distance' and abs(item['value']-.00008/.2)<1e-7 for item in compact['responses'][0]['Normal']['inputs'])

# A control must still work at the maximum declared height: gain changes bump
# strength rather than silently clipping the distance to the same value.
m,n,_,_=source('reliefgain');p=profile('reliefgain',[1,0,0]);p['controls']={'relief_gain':1.5}
assert finish_surface(m,n,p,{},.5)
assert abs(m.node_tree.nodes['Physical reference estimated microheight'].inputs['Strength'].default_value-.525)<1e-6

# Numerically evaluate connected deposition coverage with all input signals at
# their maximum. Paint gets half the metal dose and never a second dust layer.
def ev_input(socket):
    return ev(socket.links[0].from_socket) if socket.is_linked else socket.default_value
def ev(socket):
    n=socket.node
    if n.type=='VALUE':return socket.default_value
    if n.type=='TEX_IMAGE':return (1.,1.,1.,1.)
    if n.type=='NEW_GEOMETRY':return (0.,0.,1.)
    if n.type=='VECT_TRANSFORM':return ev_input(n.inputs['Vector'])
    if n.type=='AMBIENT_OCCLUSION':return 0.
    if n.type=='SEPXYZ':return ev_input(n.inputs[0])['XYZ'.index(socket.name)]
    if n.type=='SEPARATE_COLOR':return ev_input(n.inputs[0])[('Red','Green','Blue').index(socket.name)]
    if n.type=='MATH':
        a,b=ev_input(n.inputs[0]),ev_input(n.inputs[1])
        value={'ADD':lambda:a+b,'SUBTRACT':lambda:a-b,'MULTIPLY':lambda:a*b,
               'MINIMUM':lambda:min(a,b),'MAXIMUM':lambda:max(a,b)}[n.operation]()
        return max(0.,min(1.,value)) if n.use_clamp else value
    raise AssertionError(n.type)
assert abs(ev_input(tree.nodes['Physical one deposition layer'].inputs[0])-.25)<1e-6
assert abs(ev_input(m.node_tree.nodes['Physical one deposition layer'].inputs[0])-.5)<1e-6
assert tree.nodes['batch_metallic_aov']['basis']=='reference_layer_weighted_material_metalness'
assert ev_input(m.node_tree.nodes['batch_metallic_aov'].inputs['Value'])<.8

# Changes to the background do not alter pigments, texture alignment or layers.
before=[(n.name,n.type) for n in tree.nodes]
assert finish_surface(mat,node,finish,{'dust_color':[0.,0.,1.],'haze':.9},0.)
assert before==[(n.name,n.type) for n in tree.nodes]
assert tree.nodes['Physical dust strength'].outputs[0].default_value==0.

# Aluminium and unknown alloys never receive orange iron corrosion.
for family in ('nonferrous_or_stainless','unknown'):
    m,n,_,_=source(family);p=profile(family,[1,0,0],authored=False,family=family)
    assert finish_surface(m,n,p,{},.5)
    obj.data.materials.clear();obj.data.materials.append(m)
    read=material_state(obj)
    assert read['control_domains']['oxide_gain'] is False
    assert read['control_domains']['scratch_gain'] is True

# The same protocol's third channel is authored nonferrous, not inferred rust.
m,n,_,_=source('third');p=profile('third',[0,0,1])
assert finish_surface(m,n,p,{},.5)
obj.data.materials.clear();obj.data.materials.append(m)
assert material_state(obj)['control_domains']['oxide_gain'] is False

# Protected and source-stage materials retain their exact source node trees.
for name,eligible,stage in [('chrome',False,'full'),('plastic',False,'full'),('label',False,'full'),('source',True,'source')]:
    m,n,_,_=source(name);p=profile(name,[1,0,0],stage=stage,eligible=eligible)
    count=len(m.node_tree.nodes);assert not finish_surface(m,n,p,{},.5)
    assert len(m.node_tree.nodes)==count

# Absolute dust changes do not stack; material changes require a clean source.
bad=copy.deepcopy(finish);bad['signature']='new'
try:
    finish_surface(mat,node,bad,{},.5);raise AssertionError('stale graph accepted')
except ValueError:pass
for amount in (.75,float('nan')):
    try:
        finish_surface(mat,node,finish,{},amount);raise AssertionError('dust accepted')
    except ValueError:pass
bad=copy.deepcopy(finish);bad['reference']['height_amplitude_m']=.001
try:
    finish_surface(mat,node,bad,{},.5);raise AssertionError('height accepted')
except ValueError:pass

# A material with only a paint region needs no invented metal identity sample.
m,n,_,_=source('onlypaint',0.,True);p=profile('onlypaint',[0,1,0])
p['regions']['materials']['onlypaint']['source_identity_by_region']={'paint':[.31,.12,.08]}
assert finish_surface(m,n,p,{},.5)

# Readback follows actual body sockets, not retained original parameters or the
# request. A disconnected reference graph must not advertise working controls.
obj.data.materials.clear();obj.data.materials.append(m)
old_state=material_state(obj);body=m.node_tree.nodes['Reference body response']
for name,value in [('Base Color',(.02,.04,.06,1.)),('Roughness',.71)]:
    for link in list(body.inputs[name].links):m.node_tree.links.remove(link)
    body.inputs[name].default_value=value
new_state=material_state(obj)
assert new_state['state_sha256']!=old_state['state_sha256']
ref=compact_readback(new_state)['materials'][0]['reference_response']['responses'][0]
assert abs(ref['Roughness']['value']-.71)<1e-6
assert abs(ref['Base Color']['value'][0]-.02)<1e-6
assert n.inputs['Roughness'].is_linked and n.inputs['Roughness'].links[0].from_node!=body
output=next(node for node in m.node_tree.nodes if node.type=='OUTPUT_MATERIAL')
m.node_tree.links.new(n.outputs[0],output.inputs['Surface'])
disconnected=material_state(obj)
assert disconnected['materials'][0]['reference_response']['status']=='incomplete'
assert not any(disconnected['control_domains'].values())

mat.use_fake_user=True
bpy.ops.wm.save_as_mainfile(filepath=BLEND)
bpy.ops.wm.open_mainfile(filepath=BLEND)
mat=bpy.data.materials['paint']
assert mat.node_tree.nodes['Physical dust strength'].outputs[0].default_value==0.
assert mat.node_tree.nodes['Principled BSDF']['reference_material_version']=='reference-shader-v1'
Path(RESULT).write_text(json.dumps({'passed':True,'domains':state['control_domains'],'source_shader_intact':True}),encoding='utf8')
'''.replace('HERE', repr(str(HERE))).replace('TILE', repr(str(tile))).replace('BLEND', repr(str(folder/'node-check.blend'))).replace('RESULT', repr(str(folder/'result.json'))), encoding='utf8')
            process = subprocess.run([str(BLENDER), '--background', '--factory-startup', '--python', str(script)],
                                     capture_output=True, text=True, encoding='utf8', errors='replace', timeout=75)
            self.assertEqual(process.returncode, 0, process.stdout[-7000:]+process.stderr[-7000:])
            self.assertTrue((folder/'result.json').is_file(), process.stdout[-7000:]+process.stderr[-7000:])
            self.assertTrue(json.loads((folder/'result.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
