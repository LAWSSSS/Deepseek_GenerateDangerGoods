"""Real Blender shader graph checks; no model-service calls or GPU rendering."""
import subprocess
import tempfile
from pathlib import Path
import unittest
from display_batch.common import HERE, BLENDER


class SourceShaderTests(unittest.TestCase):
    def test_coherent_deposit_preserves_source_and_has_bounded_surface_variation(self):
        script=f'''
import sys,json
sys.path.insert(0,{str(HERE)!r})
import bpy
from display_batch.scanned_shader import matte_dust_surface
sample={{'noise':.5,'up':1.,'ao':.7,'edge':.5}}
def evaluate(socket):
 if not socket.is_linked:return float(socket.default_value)
 node=socket.links[0].from_node
 if node.type=='VALUE':return node.outputs[0].default_value
 if node.type=='TEX_NOISE':return sample['noise']
 if node.type=='AMBIENT_OCCLUSION':return sample['ao']
 if node.bl_idname=='ShaderNodeSeparateXYZ':return sample['up']
 if node.type=='NEW_GEOMETRY':return sample['edge']
 a,b=[evaluate(s) for s in node.inputs[:2]]
 result={{'ADD':lambda:a+b,'SUBTRACT':lambda:a-b,'MULTIPLY':lambda:a*b,'MAXIMUM':lambda:max(a,b)}}[node.operation]()
 return min(1,max(0,result)) if node.use_clamp else result
results={{}}
for name,weights in [('steel',[1,0,0,0]),('paint',[0,1,0,0]),('plastic',[0,0,0,1])]:
 m=bpy.data.materials.new(name);m.use_nodes=True;t=m.node_tree;p=t.nodes.get('Principled BSDF')
 rough=t.nodes.new('ShaderNodeValue');rough.outputs[0].default_value=.27
 normal=t.nodes.new('ShaderNodeNormalMap')
 t.links.new(rough.outputs[0],p.inputs['Roughness']);t.links.new(normal.outputs[0],p.inputs['Normal'])
 p.inputs['Base Color'].default_value=(.32,.07,.03,1.);p.inputs['Metallic'].default_value=1. if name=='steel' else 0.
 base=tuple(p.inputs['Base Color'].default_value);alpha=p.inputs['Alpha'].default_value
 parts={{'materials':{{name:{{'constant':weights,'coverage':weights}}}}}}
 env={{'dust_color':[.12,.13,.14],'integration_mode':'coherent'}}
 profile={{'mode':'preserve_source','paint_dust_scale':.5,'dust_scale':.5}}
 applied=matte_dust_surface(m,p,parts,env,['steel','paint','polymer'],profile)
 if name=='plastic':
  assert not applied and p.inputs['Roughness'].links[0].from_node==rough
  continue
 count=len(t.nodes);out=next(n for n in t.nodes if n.type=='OUTPUT_MATERIAL');layer=out.inputs['Surface'].links[0].from_node
 assert p.inputs['Normal'].links[0].from_node==normal
 assert tuple(p.inputs['Base Color'].default_value)==base and p.inputs['Alpha'].default_value==alpha
 assert p.inputs['Metallic'].default_value==(1. if name=='steel' else 0.)
 for key in ('Source weathering patches','Source weathering fine relief'):
  assert t.nodes[key].inputs['Vector'].links[0].from_socket.name=='Object'
 values=[]
 for up in (0.,.5,1.):
  for noise in (.2,.5,.8):
   sample.update(up=up,noise=noise)
   values.append(evaluate(layer.inputs[0]));r=evaluate(p.inputs['Roughness'])
   assert .27-1e-6<=r<=.47+1e-6
   if name=='steel':assert r>.34
   if name=='paint':
    assert abs(r-.27)<1e-6
    assert p.inputs['Roughness'].links[0].from_node==rough
 assert 0<=min(values)<max(values)<=.5 and max(values)>min(values)*2
 results[name]=values
 if name=='paint':
  worn=layer.inputs[1].links[0].from_node
  assert worn.name=='Source paint wear mix'
  assert worn.inputs[1].links[0].from_node==p
  sample.update(noise=.8,edge=.5)
  assert abs(evaluate(worn.inputs[0])-.175)<1e-6
  sample['edge']=.7
  assert abs(evaluate(worn.inputs[0])-.35)<1e-6
  sample['noise']=.2
  assert evaluate(worn.inputs[0])==0
  assert t.nodes['Source paint wear patches'].inputs['Vector'].links[0].from_socket.name=='Object'
  assert t.nodes['Source paint worn metal'].inputs['Metallic'].default_value>.7
  sample.update(noise=.5,edge=.5)
 profile['dust_scale']=0.
 assert not matte_dust_surface(m,p,parts,env,['steel','paint'],profile)
 assert evaluate(layer.inputs[0])==0 and abs(evaluate(p.inputs['Roughness'])-.27)<1e-6
 if name=='paint':assert evaluate(worn.inputs[0])==0
 profile['dust_scale']=.5
 matte_dust_surface(m,p,parts,env,['steel','paint'],profile)
 assert len(t.nodes)==count
assert all(abs(p/m-.5)<1e-6 for p,m in zip(results['paint'],results['steel']))
print('COHERENT_SHADER_CHECKS_OK',json.dumps({{k:[min(v),max(v)] for k,v in results.items()}}))
'''
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'coherent-shader.py';path.write_text(script,encoding='utf8')
            result=subprocess.run([str(BLENDER),'--background','--factory-startup','--python',str(path)],
                capture_output=True,text=True,encoding='utf8',timeout=60,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertIn('COHERENT_SHADER_CHECKS_OK',result.stdout,result.stdout+result.stderr)

    def test_zero_dust_paint_half_polymer_and_background_switch(self):
        script=f'''
import sys
sys.path.insert(0,{str(HERE)!r})
import bpy
from display_batch.scanned_shader import matte_dust_surface
def evaluate(socket):
 if not socket.is_linked:return float(socket.default_value)
 link=socket.links[0];node=link.from_node
 if node.type=='TEX_NOISE':return .5
 if node.type=='AMBIENT_OCCLUSION':return .7
 if node.bl_idname=='ShaderNodeSeparateXYZ':return 1.
 if node.type=='NEW_GEOMETRY':return .5
 a,b=[evaluate(s) for s in node.inputs[:2]]
 result={{'ADD':lambda:a+b,'SUBTRACT':lambda:a-b,'MULTIPLY':lambda:a*b,'MAXIMUM':lambda:max(a,b)}}[node.operation]()
 return min(1,max(0,result)) if node.use_clamp else result
amounts=[]
for name,weights in [('steel',[1,0,0,0]),('paint',[0,1,0,0]),('plastic',[0,0,0,1])]:
 m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF')
 p.inputs['Roughness'].default_value=.31;p.inputs['Metallic'].default_value=1. if name=='steel' else 0.
 p.inputs['Base Color'].default_value=(.4,.05,.02,1.)
 parts={{'materials':{{name:{{'constant':weights,'coverage':weights}}}}}}
 env={{'dust_color':[.12,.13,.14]}};profile={{'mode':'preserve_source','paint_dust_scale':.5,'dust_scale':0.}}
 assert not matte_dust_surface(m,p,parts,env,['steel','paint'],profile)
 assert not p.inputs['Roughness'].is_linked
 profile['dust_scale']=.5
 applied=matte_dust_surface(m,p,parts,env,['steel','paint'],profile)
 if name=='plastic':
  assert not applied and p.inputs['Metallic'].default_value==0.
  continue
 count=len(m.node_tree.nodes)
 out=next(n for n in m.node_tree.nodes if n.type=='OUTPUT_MATERIAL')
 layer=out.inputs['Surface'].links[0].from_node
 amounts.append(evaluate(layer.inputs[0]))
 assert tuple(round(x,3) for x in p.inputs['Base Color'].default_value)==(.4,.05,.02,1.)
 color=m.node_tree.nodes['Source weathering color'];first=tuple(color.outputs[0].default_value)
 matte_dust_surface(m,p,parts,{{'dust_color':[.18,.15,.12]}},['steel','paint'],profile)
 assert len(m.node_tree.nodes)==count and tuple(color.outputs[0].default_value)!=first
 profile['dust_scale']=0.
 assert not matte_dust_surface(m,p,parts,env,['steel','paint'],profile)
 assert abs(evaluate(layer.inputs[0]))<1e-7
 assert abs(evaluate(p.inputs['Roughness'])-.31)<1e-5
assert abs(amounts[1]/amounts[0]-.5)<1e-6
print('SOURCE_SHADER_CHECKS_OK')
'''
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'source-shader.py';p.write_text(script,encoding='utf8')
            result=subprocess.run([str(BLENDER),'--background','--factory-startup','--python',str(p)],
                capture_output=True,text=True,encoding='utf8',timeout=60,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertIn('SOURCE_SHADER_CHECKS_OK',result.stdout,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
