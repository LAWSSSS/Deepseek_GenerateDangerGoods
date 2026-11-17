"""Construct real Blender nodes and lamps, verify readback without any render."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from display_batch.common import BLENDER,HERE


class RenderReadbackTests(unittest.TestCase):
    @unittest.skipUnless(BLENDER.is_file(),'Blender is not installed')
    def test_actual_nodes_and_lamps_not_requested_parameter_echo(self):
        with TemporaryDirectory(prefix='scrap-render-readback-') as temporary:
            root=Path(temporary);script=root/'smoke.py';result=root/'readback.json'
            script.write_text('''import sys,json
from pathlib import Path
import bpy
sys.path.insert(0,CONTROLLER)
from display_batch.physical_layers import finish_surface
from display_batch.render_readback import material_state,lighting_state,compact_readback
root=Path(ROOT)
image=bpy.data.images.new('fixture-pattern',width=4,height=4)
image.pixels[:]=[.5,.5,.5,1.]*16
image.filepath_raw=str(root/'pattern.png');image.file_format='PNG';image.save()
mask=bpy.data.images.new('nonferrous-mask',width=4,height=4)
mask.pixels[:]=[0.,0.,1.,1.]*16
mask.filepath_raw=str(root/'mask.png');mask.file_format='PNG';mask.save()
bpy.ops.mesh.primitive_cube_add()
obj=bpy.context.object
recipe={'oxide_period_m':.12,'scratch_period_m':.08,'dust_period_m':.09,
 'relief_depth_m':.00008,'dust_depth_m':.00004,'oxide_max':.3,'abrasion_max':.22,
 'dust_albedo_linear':[.2,.185,.165],'max_dust':.5,'paint_dust_multiplier':.5}
maps={name:{'path':str(root/'pattern.png')} for name in ('oxide','paint','dust')}
for name,entry in [('steel',{'constant':[1,0,0],'eligible':True,'metal_family':'ferrous'}),
                   ('aluminum',{'mask':str(root/'mask.png'),'authored_atlas':True,'eligible':True,'metal_family':'partitioned'}),
                   ('plastic',{'constant':[0,0,0],'eligible':False,'metal_family':'unknown'})]:
 material=bpy.data.materials.new(name);material.use_nodes=True;obj.data.materials.append(material)
 shader=material.node_tree.nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=.31
 finish={'version':'physical-layers-v1','signature':'fixture-'+name,'size_m':.8,'recipe':recipe,
         'regions':{'materials':{name:entry}},'maps':maps,'source_mesh_hash':'a'*64}
 finish_surface(material,shader,finish,{},.5)
actual=material_state(obj)
rows={row['material']:row for row in actual['materials']}
assert rows['steel']['control_domains']['oxide_gain'],actual
assert not rows['aluminum']['control_domains']['oxide_gain'],actual
assert rows['aluminum']['control_domains']['scratch_gain'],actual
assert not any(rows['plastic']['control_domains'].values()),actual
assert abs(rows['steel']['source_brdf'][0]['Roughness']['value']-.31)<1e-6
before=actual['state_sha256']
obj.data.materials[0].node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.57
assert material_state(obj)['state_sha256']!=before
for old in list(bpy.context.scene.objects):
 if old.type=='LIGHT':bpy.data.objects.remove(old,do_unlink=True)
for name,energy,size in [('key',120.,4.8),('fill',35.,5.8)]:
 data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.size=size
 lamp=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(lamp)
light=lighting_state(bpy.context.scene)
assert sorted(row['energy_W'] for row in light['lights'])==[35.,120.]
assert all(light['control_domains'].values())
old=light['state_sha256'];bpy.data.lights['fill'].energy=42
assert lighting_state(bpy.context.scene)['state_sha256']!=old
Path(RESULT).write_text(json.dumps({'passed':True,'render_calls':0,'gpu_used':False,
 'material':compact_readback(actual),'lighting':compact_readback(light)}),encoding='utf8')
'''.replace('CONTROLLER',repr(str(HERE))).replace('ROOT',repr(str(root))).replace('RESULT',repr(str(result))),encoding='utf8')
            process=subprocess.run([str(BLENDER),'--background','--factory-startup','--python-exit-code','1','--python',str(script)],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            self.assertEqual(process.returncode,0,process.stdout[-5000:]+process.stderr[-5000:])
            evidence=json.loads(result.read_text(encoding='utf8'))
            self.assertTrue(evidence['passed'])
            self.assertEqual(evidence['render_calls'],0)


if __name__=='__main__':unittest.main(verbosity=2)
