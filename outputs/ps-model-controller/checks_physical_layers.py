"""Pure configuration and real Blender node checks; no image-generation API."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from display_batch.common import BLENDER, HERE, file_hash
from display_batch.physical_material import RECIPE, VERSION, metal_family, metric_size, prepare_finish


class PhysicalMaterialTests(unittest.TestCase):
    def test_invalid_physical_sizes_rejected(self):
        for value in (None, True, 0, -1, float('nan'), float('inf'), 'unknown'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                metric_size(value)
        self.assertEqual(metric_size(.206), .206)

    def test_metalness_is_not_evidence_for_iron(self):
        self.assertEqual(metal_family('unknown', ['steel'], False), 'unknown')
        self.assertEqual(metal_family('brass steel mixed', ['steel'], False), 'nonferrous_or_stainless')
        self.assertEqual(metal_family('aged steel', [], False), 'ferrous')

    def test_payload_metric_hash_preservation_and_protected_mapping(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            mesh = path/'mesh.glb'
            mesh.write_bytes(b'fixture mesh identity')
            image = path/'tile.png'
            Image.new('RGB', (8, 8), (80, 90, 120)).save(image)
            pack = {'maps': {k: {'path': str(image), 'sha256': file_hash(image)} for k in ('oxide','paint','dust')},
                    'manifest': 'fixture', 'atlases': {}}
            regions = {'signature':'fixture', 'materials': {
                'steel body': {'coverage':[1,0,0,0], 'constant':[1,0,0,0]},
                'chrome rod': {'coverage':[1,0,0,0], 'constant':[1,0,0,0]},
                'unknown': {'coverage':[0,0,0,0]},
                'Mixed chrome paint atlas': {'coverage':[.3,.3,.3], 'constant':[1,0,0], 'authored_atlas':True},
            }}
            with patch('display_batch.physical_material.finish_regions', return_value=regions):
                a = prepare_finish(mesh,path,pack,.2)
                b = prepare_finish(mesh,path,pack,.2)
                c = prepare_finish(mesh,path,pack,1.)
            self.assertEqual(a['signature'], b['signature'])
            self.assertNotEqual(a['signature'], c['signature'])
            self.assertFalse(a['geometry_modified'])
            self.assertEqual(set(a['maps']), {'oxide','paint','dust'})
            self.assertFalse(a['regions']['materials']['chrome rod']['eligible'])
            self.assertFalse(a['regions']['materials']['unknown']['eligible'])
            self.assertTrue(a['regions']['materials']['Mixed chrome paint atlas']['eligible'])
            self.assertEqual(mesh.read_bytes(), b'fixture mesh identity')

    @unittest.skipUnless(BLENDER.is_file(), 'Blender is not installed')
    def test_actual_blender_node_layers_and_roundtrip(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            tile=folder/'tile.png'
            Image.new('RGB',(16,16),(100,80,130)).save(tile)
            script=folder/'check.py'
            script.write_text('''import sys, json, math
from pathlib import Path
import bpy
sys.path.insert(0, HERE)
from display_batch.physical_layers import finish_surface
bpy.ops.wm.read_factory_settings(use_empty=True)

def source(name, metallic=.8):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    node=mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Metallic'].default_value=metallic
    node.inputs['Base Color'].default_value=(.31,.12,.08,1.)
    node.inputs['Roughness'].default_value=.42
    return mat,node

def profile(name,weights,stage='full',size=.2,eligible=True):
    return {'version':VERSION,'signature':'fixture-'+str(size)+'-'+stage,'source_mesh_hash':'abcdef12'*8,
      'maps':{key:{'path':TILE} for key in ('oxide','paint','dust')},'size_m':size,'recipe':RECIPE,'layer_stage':stage,
      'regions':{'materials':{name:{'coverage':weights,'constant':weights,'eligible':eligible,
       'authored_atlas':True,'metal_family':'partitioned'}}}}

mat,node=source('paint',0.)
finish=profile('paint',[0,1,0]); before=[tuple(node.inputs['Base Color'].default_value),node.inputs['Roughness'].default_value,node.inputs['Metallic'].default_value]
assert finish_surface(mat,node,finish,{'dust_color':[1.,0.,0.]},.5)
tree=mat.node_tree
assert [tuple(node.inputs['Base Color'].default_value),node.inputs['Roughness'].default_value,node.inputs['Metallic'].default_value]==before
assert len([n for n in tree.nodes if n.name=='Physical single dielectric deposit'])==1
assert not any('Even ' in n.name or 'amber' in n.name.lower() or 'caked' in n.name.lower() for n in tree.nodes)
assert tree.nodes['Physical single dielectric deposit'].inputs['Metallic'].default_value==0.
assert tree.nodes['Physical dielectric oxide'].inputs['Metallic'].default_value==0.
assert tree.nodes['Physical exposed steel'].inputs['Metallic'].default_value==1.
assert abs(tree.nodes['Physical coordinates in metres'].inputs['Scale'].default_value-.2)<1e-6
assert abs(tree.nodes['Physical wear depth in metres'].inputs['Distance'].default_value-RECIPE['relief_depth_m']/.2)<1e-7
assert {'physical_layers','physical_regions','generated_finish'} <= {n.aov_name for n in tree.nodes if n.type=='OUTPUT_AOV'}
# Evaluate the coverage arithmetic at worst-case texture/AO/up values. This
# checks the full connected graph, including paint attenuation and clamps.
def evaluate_input(socket):
    return evaluate_output(socket.links[0].from_socket) if socket.is_linked else socket.default_value
def evaluate_output(socket):
    n=socket.node;kind=n.bl_idname
    if kind=='ShaderNodeValue':return socket.default_value
    if kind=='ShaderNodeNewGeometry':return (0.,0.,1.)
    if kind=='ShaderNodeAmbientOcclusion':return 0.
    if kind=='ShaderNodeTexImage':return (1.,1.,1.,1.)
    if kind=='ShaderNodeSeparateXYZ':return evaluate_input(n.inputs[0])['XYZ'.index(socket.name)]
    if kind=='ShaderNodeSeparateColor':return evaluate_input(n.inputs[0])[('Red','Green','Blue').index(socket.name)]
    if kind=='ShaderNodeMath':
        a,b=evaluate_input(n.inputs[0]),evaluate_input(n.inputs[1])
        result={'ADD':lambda:a+b,'SUBTRACT':lambda:a-b,'MULTIPLY':lambda:a*b,
                'MAXIMUM':lambda:max(a,b),'MINIMUM':lambda:min(a,b)}[n.operation]()
        return min(1.,max(0.,result)) if n.use_clamp else result
    raise AssertionError(kind)
assert abs(evaluate_input(tree.nodes['Physical one deposition layer'].inputs[0])-.25)<1e-6
steel_m,steel_s=source('steel');steel_p=profile('steel',[1,0,0])
assert finish_surface(steel_m,steel_s,steel_p,{},.5)
assert abs(evaluate_input(steel_m.node_tree.nodes['Physical one deposition layer'].inputs[0])-.5)<1e-6
count=len(tree.nodes)
color=tuple(tree.nodes['Physical single dielectric deposit'].inputs['Base Color'].default_value)
assert finish_surface(mat,node,finish,{'dust_color':[0.,0.,1.]},0.)
assert len(tree.nodes)==count and tree.nodes['Physical dust strength'].outputs[0].default_value==0.
assert tuple(tree.nodes['Physical single dielectric deposit'].inputs['Base Color'].default_value)==color
try:
    finish_surface(mat,node,finish,{},.75)
    raise AssertionError('dust range accepted')
except ValueError: pass
for name,eligible,stage in [('chrome',False,'full'),('source',True,'source'),('plastic',False,'full')]:
    m,n=source(name);p=profile(name,[1,0,0],stage=stage,eligible=eligible)
    count=len(m.node_tree.nodes);assert not finish_surface(m,n,p,{},.5)
    assert len(m.node_tree.nodes)==count
m,n=source('wear');p=profile('wear',[1,0,0],stage='wear')
assert finish_surface(m,n,p,{},.5)
assert m.node_tree.nodes.get('Physical single dielectric deposit') is None
mat.use_fake_user=True
bpy.ops.wm.save_as_mainfile(filepath=BLEND)
bpy.ops.wm.open_mainfile(filepath=BLEND)
assert bpy.data.materials['paint'].node_tree.nodes['Physical dust strength'].outputs[0].default_value==0.
Path(RESULT).write_text(json.dumps({'passed':True,'nodes':count}),encoding='utf8')
'''.replace('HERE',repr(str(HERE))).replace('TILE',repr(str(tile))).replace('VERSION',repr(VERSION)).replace('RECIPE',repr(RECIPE)).replace('BLEND',repr(str(folder/'node-check.blend'))).replace('RESULT',repr(str(folder/'result.json'))),encoding='utf8')
            process=subprocess.run([str(BLENDER),'--background','--factory-startup','--python',str(script)],
                                   capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            self.assertEqual(process.returncode,0,process.stdout[-6000:]+process.stderr[-6000:])
            self.assertTrue((folder/'result.json').is_file(),process.stdout[-6000:]+process.stderr[-6000:])
            self.assertTrue(json.loads((folder/'result.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
