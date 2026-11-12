"""Real Blender shader graph checks; no model-service calls or GPU rendering."""
import subprocess
import tempfile
from pathlib import Path
import unittest
from display_batch.common import HERE, BLENDER


class SourceShaderTests(unittest.TestCase):
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
