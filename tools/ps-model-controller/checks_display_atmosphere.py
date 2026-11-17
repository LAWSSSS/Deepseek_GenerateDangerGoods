"""Regression checks for atmospheric compositing and material independence."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest

import numpy as np

from display_batch.atmosphere import (Atmospheres, estimate, transmission_fields, fogged_radiance,
                                     shadowed_background, calibrate_environment, regions_mask)
from display_batch.common import BLENDER, HERE
from display_batch.photography import match_linear
from display_batch.rendering import part_keys
from display_batch.review import fusion_actions
from display_batch.schema import BatchConfig, FusionObservation
from display_batch.geometry import camera_for, illumination_axis
from display_batch.fusion import optical_layer, topology


class AtmosphereTests(unittest.TestCase):
    def test_excluded_text_cannot_lower_neighbouring_dark_channel(self):
        rng=np.random.default_rng(12)
        dark=np.repeat(rng.integers(150,230,(100,160,1),dtype=np.uint8),3,axis=2)
        bright=dark.copy()
        rect=[[.375,0,.625,1]]
        mask=regions_mask(rect,(100,160));dark[mask]=0;bright[mask]=255
        answer={'kind':'veil','confidence':.9,'haze_regions':[],'exclude_regions':rect,
                'light_regions':[],'light_direction':None,'evidence':'text excluded'}
        first,_=estimate(dark,np.ones((100,160)),np.ones((100,160),bool),answer)
        second,_=estimate(bright,np.ones((100,160)),np.ones((100,160),bool),answer)
        np.testing.assert_array_equal(first['transmission'],second['transmission'])

    def test_small_hole_limits_blur_instead_of_disabling_the_camera(self):
        alpha=np.zeros((60,60),np.float32);alpha[10:50,10:50]=1
        alpha[29:31,29:31]=0
        rgb=np.full((60,60,3),.65,np.float32)
        _,coverage,sigma,iou=optical_layer(rgb,alpha,1.5,1.)
        self.assertGreater(sigma,.18)
        self.assertLess(sigma,1.4)
        self.assertGreaterEqual(iou,.995)
        self.assertEqual(topology(coverage>.5),topology(alpha>.5))

    def test_light_direction_preserves_both_screen_coordinates(self):
        for elevation in (35,45,55):
            camera=camera_for(np.array([[-1,-1,0],[1,1,1]]),elevation)
            right,up,_=np.asarray(camera['basis'])
            for x,y in ((.3,-.5),(-.6,-.2),(.5,.4)):
                axis=illumination_axis(camera,(x,y))
                projected=np.array([axis@right,-axis@up])
                np.testing.assert_allclose(projected/np.linalg.norm(projected),
                    np.array([x,y])/np.linalg.norm([x,y]),atol=1e-8)
                self.assertGreater(axis[2],0)

    def test_airlight_calibration_never_changes_the_deposit_material(self):
        fields={'transmission':np.full((24,32),.7,np.float32),
                'airlight':np.full((24,32,3),.4,np.float32),'sample_mask':np.ones((24,32)),
                'signature':'test','summary':{'visual_evidence':{'light_direction':None,'confidence':.7}}}
        scene={'atmosphere':fields,'frame':np.full((24,32,3),150,np.uint8)}
        environment={'dust_color':[.1,.15,.2]}
        calibrate_environment(scene,{},environment,np.ones((24,32),bool))
        self.assertEqual(environment['dust_color'],[.1,.15,.2])
        fields['transmission'][:]=.95
        calibrate_environment(scene,{},environment,np.ones((24,32),bool))
        self.assertEqual(environment['dust_color'],[.1,.15,.2])

    def test_malformed_scene_analysis_is_cached_as_unknown_once(self):
        class Vision:
            endpoint=SimpleNamespace(model_dump=lambda:{'model':'fake'})
            calls=0
            async def ask(self,*args,**kwargs):
                self.calls+=1
                raise RuntimeError('bounded schema retry exhausted')
        async def run(directory):
            scene={'key':'scene','folder':directory,'frame':np.full((40,60,3),100,np.uint8),
                   'depth_native':np.ones((40,60)),'allowed':np.ones((40,60),bool),
                   'protected':np.zeros((40,60),bool)}
            vision=Vision()
            first=await Atmospheres(directory,vision).prepare(scene)
            second=await Atmospheres(directory,vision).prepare(scene)
            self.assertEqual(vision.calls,1)
            self.assertEqual(first['summary']['confidence'],'unknown')
            np.testing.assert_array_equal(first['transmission'],second['transmission'])
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory))

    def test_clear_frame_and_bright_surfaces_do_not_gain_fog(self):
        frame=np.full((72,128,3),220,np.uint8)
        frame[::4]=110
        answer={'kind':'clear','confidence':.9,'haze_regions':[], 'exclude_regions':[],
                'light_regions':[],'light_direction':None,'evidence':'clear'}
        fields,info=estimate(frame,np.ones((72,128)),np.ones((72,128),bool),answer)
        np.testing.assert_array_equal(fields['transmission'],1.)
        self.assertEqual(info['depth_coefficient'],0.)

    def test_spatial_veil_is_local_and_repeatable(self):
        rng=np.random.default_rng(4)
        clear=np.repeat(rng.uniform(.04,.65,(90,160,1)),3,axis=2)
        t=np.tile(np.linspace(.95,.6,160),(90,1))
        fog=clear*t[...,None]+.85*(1-t[...,None])
        frame=np.uint8(np.clip(fog*255,0,255))
        answer={'kind':'veil','confidence':.9,'haze_regions':[], 'exclude_regions':[],
                'light_regions':[],'light_direction':None,'evidence':'right side veil'}
        fields,_=estimate(frame,np.full((90,160),.5),np.ones((90,160),bool),answer)
        other,_=estimate(frame,np.full((90,160),.5),np.ones((90,160),bool),answer)
        self.assertGreater(float(fields['transmission'][:,:30].mean()),float(fields['transmission'][:,-30:].mean())+.03)
        np.testing.assert_array_equal(fields['transmission'],other['transmission'])

    def test_object_depth_uses_the_same_inverse_depth_frame(self):
        fields={'summary':{'depth_near':1.,'depth_far':0.,'depth_coefficient':.5},
                'screen_tau':np.full((10,10),.05,np.float32),'transmission':np.full((10,10),.8,np.float32),
                'airlight':np.full((10,10,3),.7,np.float32),'inverse_depth':np.full((10,10),.4,np.float32)}
        scene={'atmosphere':fields,'size':(10,10)}
        order={'confidence':'estimated','scale':1.,'offset':0.}
        near=transmission_fields(scene,(0,0,10,10),np.full((10,10),1.2),order)[0]
        far=transmission_fields(scene,(0,0,10,10),np.full((10,10),3.),order)[0]
        self.assertTrue(np.all(near>far))
        off=transmission_fields(scene,(0,0,10,10),np.full((10,10),3.),order,enabled=False)[0]
        np.testing.assert_array_equal(off,1.)

    def test_shadow_changes_only_the_transmitted_surface_light(self):
        clear=np.full((3,4,3),.4,np.float32)
        air=np.full_like(clear,.7)
        t=np.full((3,4),.6,np.float32)
        observed=fogged_radiance(clear,t,air)
        shaded=shadowed_background(observed,np.full((3,4),.5),t,air)
        expected=fogged_radiance(clear*.5,t,air)
        np.testing.assert_allclose(shaded,expected,atol=1e-7)
        np.testing.assert_array_equal(shadowed_background(observed,np.zeros((3,4)),t,air),observed)

    def test_haze_attenuates_highlights_without_changing_coverage(self):
        clear=np.zeros((9,9,3),np.float32)
        clear[:,4]=.9
        alpha=np.ones((9,9),np.float32);alpha[2:5,2:5]=0
        original=alpha.copy()
        result=fogged_radiance(clear,np.full((9,9),.7),np.full_like(clear,.5))
        self.assertAlmostEqual(float(result[0,4,0]-result[0,0,0]),.63,places=6)
        premult=result*alpha[...,None]
        self.assertFalse(premult[alpha==0].any())
        np.testing.assert_array_equal(alpha,original)

    def test_calibration_preserves_material_brightness_differences(self):
        alpha=np.ones((32,32),np.float32)
        pixels=np.full((800,3),128,np.float32)
        environment={'calibration':{'exposure_ev':0.,'white_balance':[1.,1.,1.],'limited':False},'haze':.3}
        def apply(value):
            return match_linear(np.full((32,32,3),value,np.float32),alpha,pixels,np.zeros_like(alpha),
                {},17,environment,reference=False,camera_capture=False)
        dark,d=apply(.04);bright,b=apply(.5)
        self.assertGreater(float(bright.mean()-dark.mean()),.3)
        self.assertEqual(d['exposure_ev'],b['exposure_ev'])
        self.assertEqual(d['haze'],0.)
        self.assertEqual(d['contrast_gain'],1.)

    def test_fog_correction_reuses_render_dependencies(self):
        request={'windows':{'object':1,'shadow':2},'environment':{'world_color':[1,1,1],'haze':.1},
                 'appearance':'native','camera':{},'plane_normal':[0,0,1],'light_gain':1,
                 'mesh_hash':'mesh','matrix':[],'yaw':0,'roughness_delta':0,'dust_gain':1,
                 'dust_control':1,'material_classes':['steel']}
        original=part_keys(request,'CPU','v1')
        request['environment']['haze']=.3
        self.assertEqual(original,part_keys(request,'CPU','v1'))
        request['dust_control']=1.3
        changed=part_keys(request,'CPU','v1')
        self.assertNotEqual(original['object'],changed['object'])
        self.assertNotEqual(original['with_shadow'],changed['with_shadow'])
        self.assertEqual(original['without_shadow'],changed['without_shadow'])

    def test_recognizer_routes_airlight_to_compositing_not_weathering(self):
        scores=dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),3)
        obs=FusionObservation(trace='visible',scores=scores,defects=[
            {'code':'haze_clear','severity':'major','evidence':'物体暗部比相邻钢片过黑'}],evidence='薄雾不匹配')
        actions=fusion_actions(obs.review(),{'photo':{},'checks':{}},{'haze':['more'],'dust':['more']})
        self.assertEqual([(a.kind,a.value) for a in actions],[('haze','more')])
        obs.defects[0].code='material'
        self.assertFalse(fusion_actions(obs.review(),{'photo':{},'checks':{}},{'material':['weathered']}))
        self.assertEqual(BatchConfig(backgrounds='x',output='y').atmosphere,'auto')

    def test_blender_material_retains_original_metal_under_independent_dust(self):
        code=f'''import sys
sys.path.insert(0,{str(HERE)!r})
import bpy
from blender_display_worker import native_surface
mat=bpy.data.materials.new('brushed stainless')
mat.use_nodes=True
shader=mat.node_tree.nodes.get('Principled BSDF')
shader.inputs['Metallic'].default_value=1.
shader.inputs['Roughness'].default_value=.23
r={{'material_classes':['steel'],'environment':{{'dust_color':[.1,.12,.13],'haze':0.}},'dust_gain':1.,'dust_control':1.}}
native_surface(mat,shader,r)
factor=mat.node_tree.nodes['native deposit amount'].inputs[1].default_value
nodes=len(mat.node_tree.nodes)
r['environment']['haze']=.9
native_surface(mat,shader,r)
assert shader.inputs['Metallic'].default_value==1.
assert mat.node_tree.nodes['native dielectric deposit'].inputs['Metallic'].default_value==0.
assert mat.node_tree.nodes['native roughness adjustment'].inputs[1].default_value==0.
assert mat.node_tree.nodes['native deposit amount'].inputs[1].default_value==factor
assert len(mat.node_tree.nodes)==nodes
print('ATMOSPHERE_SHADER_PASS')
'''
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'check.py';script.write_text(code,encoding='utf8')
            result=subprocess.run([str(BLENDER),'--background','--factory-startup','--python-exit-code','1',
                '--python',str(script)],capture_output=True,text=True,encoding='utf8',timeout=60,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertEqual(result.returncode,0,result.stdout[-1800:]+result.stderr[-600:])
        self.assertIn('ATMOSPHERE_SHADER_PASS',result.stdout)


if __name__=='__main__':
    unittest.main(verbosity=2)
