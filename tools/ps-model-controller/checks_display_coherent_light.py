"""Offline tests for local reflection contrast and source-finish preservation."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from checks_display_local_light import LocalLightTests
from display_batch.common import BLENDER, HERE
from display_batch.geometry import camera_for
from display_batch.local import neighbourhood
from display_batch.local_light import clear_reflection_samples, estimate_local_light, light_rig


class CoherentLightTests(unittest.TestCase):
    def fixture(self):
        scene,candidate=LocalLightTests().fixture()
        candidate.update(camera=camera_for(np.array([[-.5,-.4,0],[.5,.4,.6]]),45),anchor_px=[215,190])
        env,_=neighbourhood(scene,candidate)
        return scene,candidate,{**env,'integration_mode':'coherent'}

    def test_local_field_has_upper_lower_contrast_without_remote_sampling(self):
        scene,candidate,env=self.fixture()
        result,mask=estimate_local_light(scene,candidate,env)
        field=np.asarray(result['reflection_probe'])
        self.assertGreater(field[48:,:,:].mean()/field[:16,:,:].mean(),3.)
        self.assertGreater(field.min(),0)
        before=copy.deepcopy(result)
        x,y,r,b=result['local_irradiance']['bounds']
        outside=np.ones(mask.shape,bool);outside[y:b,x:r]=False
        scene['frame'][outside]=[255,0,0]
        after,_=estimate_local_light(scene,candidate,env)
        self.assertEqual(before,after)
        self.assertLessEqual(result['local_irradiance']['reflection_field']['confidence'],.4)
        for key in ('airborne_dust','haze','dust_color','sensor'):
            self.assertEqual(result[key],env[key])

    def test_fill_cannot_cancel_key_shape_and_unknown_has_no_strip(self):
        env={'integration_mode':'coherent','local_irradiance':{
            'direction':{'direction':None,'confidence':0.},'level':1.,'fill_ratio':.85}}
        sky,lights=light_rig(env,{'eye':[0,-6,6],'target':[0,0,0]})
        self.assertLess(float(np.max(np.ptp(sky,axis=1))),1e-6)
        self.assertLess(lights[1][1]/lights[0][1],.3)
        self.assertGreaterEqual(lights[0][2],3.)
        self.assertLess(lights[0][2],5.)
        legacy=copy.deepcopy(env);legacy.pop('integration_mode')
        _,old=light_rig(legacy,{'eye':[0,-6,6],'target':[0,0,0]})
        self.assertGreater(old[1][1]/old[0][1],1.)

    def test_reflection_veil_uses_same_local_field_and_bounded_extension(self):
        from display_batch.atmosphere import field_crop
        with tempfile.TemporaryDirectory() as temp:
            transmission=np.tile(np.linspace(.8,1.,4,dtype=np.float32),(2,1))
            air=np.full((2,4,3),.4,np.float32)
            np.savez(Path(temp)/'fields.npz',transmission=transmission,airlight=air)
            env={'integration_mode':'coherent','dust_color':[.2]*3,
                 'airborne_dust':{'amount':.28,'color_linear':[.8]*3},
                 'atmospheric_context':{'folder':temp,'signature':'local-test',
                    'summary':{'bounds':[20,10,60,30],'confidence':'estimated','version':'test'}}}
            observed=np.full((40,80,3),.3,np.float32);selected=np.ones((40,80),bool)
            clear,evidence=clear_reflection_samples(observed,env,[0,0,80,40],selected)
            t=field_crop(transmission,[0,0,40,20],(40,20))
            expected=(.3-(1-t[...,None])*.4)/t[...,None]
            np.testing.assert_allclose(clear[10:30,20:60],np.broadcast_to(expected,(20,40,3)),atol=1e-7)
            np.testing.assert_array_equal(clear[:,:10],observed[:,:10])
            self.assertEqual(evidence['covered_sample_fraction'],.25)
            self.assertFalse(evidence['global_fallback'])
            env['airborne_dust']={'amount':0.,'color_linear':[0.]*3}
            repeated,_=clear_reflection_samples(observed,env,[0,0,80,40],selected)
            np.testing.assert_array_equal(clear,repeated)
            np.savez(Path(temp)/'fields.npz',transmission=np.ones((2,4),np.float32),airlight=air)
            clear,_=clear_reflection_samples(observed,env,[0,0,80,40],selected)
            np.testing.assert_array_equal(clear,observed)

    def test_missing_local_field_does_not_borrow_legacy_haze(self):
        observed=np.full((10,10,3),.3,np.float32)
        env={'integration_mode':'coherent','dust_color':[.2]*3,
             'airborne_dust':{'amount':.28,'color_linear':[.8]*3}}
        clear,info=clear_reflection_samples(observed,env,[0,0,10,10],np.ones((10,10),bool))
        np.testing.assert_array_equal(clear,observed)
        self.assertEqual(info['confidence'],'unknown')

    def test_coherent_dust_preserves_original_roughness_graph(self):
        script=f'''
import sys
sys.path.insert(0,{str(HERE)!r})
import bpy
from display_batch.scanned_shader import matte_dust_surface
for linked in (False,True):
 material=bpy.data.materials.new('paint');material.use_nodes=True
 shader=material.node_tree.nodes.get('Principled BSDF');rough=shader.inputs['Roughness']
 rough.default_value=.33
 if linked:
  texture=material.node_tree.nodes.new('ShaderNodeValue');texture.outputs[0].default_value=.41
  material.node_tree.links.new(texture.outputs[0],rough)
 before=rough.links[0].from_socket if linked else rough.default_value
 parts={{'materials':{{'paint':{{'constant':[0,1,0,0],'coverage':[0,1,0,0]}}}}}}
 env={{'integration_mode':'coherent','dust_color':[.12,.13,.14]}}
 profile={{'mode':'preserve_source','dust_scale':.5,'paint_dust_scale':.5}}
 assert matte_dust_surface(material,shader,parts,env,['paint'],profile)
 assert (rough.links[0].from_socket if linked else rough.default_value)==before
 count=len(material.node_tree.nodes)
 profile['dust_scale']=0.
 assert not matte_dust_surface(material,shader,parts,env,['paint'],profile)
 assert len(material.node_tree.nodes)==count
 assert material.node_tree.nodes['Source weathering strength'].inputs[1].default_value==0
 assert (rough.links[0].from_socket if linked else rough.default_value)==before
print('COHERENT_SOURCE_FINISH_OK')
'''
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'shader-check.py';path.write_text(script,encoding='utf8')
            result=subprocess.run([str(BLENDER),'--background','--factory-startup','--python',str(path)],
                                  capture_output=True,text=True,encoding='utf8',timeout=60,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('COHERENT_SOURCE_FINISH_OK',result.stdout,result.stdout+result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
