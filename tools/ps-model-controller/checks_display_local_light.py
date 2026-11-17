"""Regression checks for stronger optics, paint masks and bounded local illumination."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import checks_display_local
from display_batch.common import BLENDER, HERE
from display_batch.geometry import camera_for
from display_batch.local import camera_residual, foreground_dust, local_veil, neighbourhood, pool_object_rgb
from display_batch.local_light import estimate_local_light, fit_direction, light_rig, receiver_calibration
from display_batch.rendering import part_keys


class LocalLightTests(unittest.TestCase):
    def fixture(self):
        scene,candidate = checks_display_local.LocalTests().fixture()
        # Neutral, moderately varying material avoids an extreme pixel-noise mask.
        yy,xx = np.mgrid[:300,:440]
        value = np.uint8(np.clip(110+xx*.04+yy*.03,0,255))
        scene['frame'] = np.repeat(value[...,None],3,axis=2)
        return scene,candidate

    def test_remote_picture_and_global_light_hints_cannot_affect_estimate(self):
        scene,candidate = self.fixture()
        env,_ = neighbourhood(scene,candidate)
        before,samples = estimate_local_light(scene,candidate,env)
        x,y,r,b = env['local_context']['bounds']
        remote = np.ones(samples.shape,bool);remote[y:b,x:r]=False
        scene['frame'][remote]=[255,0,20]
        scene['answer']={'light_direction':[-1,-1],'haze':1}
        after,other = estimate_local_light(scene,candidate,env)
        self.assertEqual(before,after)
        np.testing.assert_array_equal(samples,other)
        self.assertFalse(samples[remote].any())

    def test_foreground_colour_reflections_and_text_are_not_light_samples(self):
        scene,candidate = self.fixture()
        scene['protected'][110:130,160:170]=True
        scene['frame'][220:230,170:190]=[250,250,250]
        scene['frame'][220:230,190:210]=[160,30,20]
        scene['depth_native'][210:230,220:245]=.9
        env,_ = neighbourhood(scene,candidate)
        light,mask = estimate_local_light(scene,candidate,env)
        self.assertFalse(mask[scene['protected']].any())
        self.assertFalse(mask[scene['depth_native']>.8].any())
        self.assertFalse(mask[220:230,170:210].any())
        self.assertFalse(light['local_irradiance']['global_fallback'])

    def test_direction_requires_consistent_spatial_evidence(self):
        yy,xx = np.mgrid[-1:1:6j,-1:1:6j]
        xy = np.column_stack((xx.ravel(),yy.ravel()))
        fitted = fit_direction(xy,np.exp(xy[:,0]*.3+xy[:,1]*.1)*.18)
        self.assertGreater(np.dot(fitted['direction'],np.array([3,1])/np.sqrt(10)),.99)
        self.assertIsNone(fit_direction(xy,np.full(len(xy),.18))['direction'])
        self.assertIsNone(fit_direction(xy[:4],np.full(4,.18))['direction'])

    def test_air_and_dust_are_unchanged_by_lighting_fit(self):
        scene,candidate = self.fixture();env,_ = neighbourhood(scene,candidate)
        original = copy.deepcopy(env)
        new,_ = estimate_local_light(scene,candidate,env)
        self.assertEqual(env,original)
        for key in ('dust_color','airborne_dust','haze','sensor','camera_psf'):
            self.assertEqual(env[key],new[key])

    def test_unknown_direction_produces_broad_dome_without_reflection_strip(self):
        env={'local_irradiance':{'direction':{'direction':None,'confidence':0},'level':.7,'fill_ratio':.6}}
        sky,lights = light_rig(env,{},1.)
        self.assertLess(float(np.max(np.ptp(sky,axis=1))),1e-6)
        self.assertTrue(all(size>=7.5 for _,_,size in lights))
        self.assertTrue(all(power>0 for _,power,_ in lights))

    def test_receiver_calibration_uses_only_shared_neutral_reference(self):
        raw = np.ones((20,30,3),np.float32)*.4
        env={'local_irradiance':{'target_clear_rgb_linear':[.16]*3}}
        meta={'part_paths':{'without_shadow':'receiver'}}
        with patch('cv2.imread',return_value=raw) as reader:
            value=receiver_calibration('object-A',meta,env)
            other=receiver_calibration('object-B',meta,env)
        self.assertEqual(value,other)
        self.assertLess(abs(value['fitted_luma']-value['target_luma']),.002)
        self.assertIn('without-shadow',reader.call_args[0][0])

    def test_fill_uses_observed_camera_hemisphere(self):
        env={'local_irradiance':{'direction':{'direction':None,'confidence':0},'level':1.,'fill_ratio':.8}}
        _,front=light_rig(env,{'eye':[0,-6,6],'target':[0,0,0]})
        _,side=light_rig(env,{'eye':[6,0,6],'target':[0,0,0]})
        self.assertLess(front[1][0][1],0)
        self.assertGreater(side[1][0][0],0)
        self.assertAlmostEqual(front[1][1],side[1][1])

    def test_render_keys_include_lighting_and_paint_coverage(self):
        request={'windows':{'object':{},'shadow':{}},'camera':{},'environment':{},'appearance':'local',
                 'plane_normal':[0,0,1],'light_gain':1.,'mesh_hash':'m','matrix':[],'yaw':0,
                 'roughness_delta':0.,'dust_gain':1.,'dust_control':1.,'material_classes':['steel']}
        old=part_keys(request,'OPTIX','test')
        request['matte_profile']={'paint_dust_scale':.5}
        paint=part_keys(request,'OPTIX','test')
        self.assertNotEqual(old['object'],paint['object'])
        self.assertEqual(old['without_shadow'],paint['without_shadow'])
        request['environment']['local_irradiance']={'level':.8}
        light=part_keys(request,'OPTIX','test')
        self.assertTrue(all(paint[k]!=light[k] for k in light))


class StrongOpticsTests(unittest.TestCase):
    def test_changed_shadow_cannot_change_frozen_camera_noise(self):
        rgb,alpha,env=checks_display_local.ForegroundDustTests().fixture()
        background=np.full_like(rgb,.18);writable=np.ones_like(alpha,bool)
        _,_,before=camera_residual(background,rgb,alpha,background,env,{},123,writable)
        _,_,after=camera_residual(background*.7,rgb,alpha,background,env,{},123,writable,fixed_sensor=before)
        self.assertEqual(before,after)

    def test_large_pool_preserves_constant_colour_across_empty_and_odd_cells(self):
        alpha=np.zeros((31,37),np.float32);alpha[3:28,4:34]=.7;alpha[14:19,15:20]=0
        rgb=np.ones((31,37,3),np.float32)*.99;rgb[alpha>0]=[.2,.3,.4]
        for kernel in (2,4,6):
            pooled=pool_object_rgb(rgb,alpha,kernel)
            np.testing.assert_allclose(pooled[alpha>0],rgb[alpha>0],atol=1e-6)

    def test_strong_effects_are_independent_and_do_not_touch_holes(self):
        rgb,alpha,env=checks_display_local.ForegroundDustTests().fixture()
        veiled,air=local_veil(rgb,{**env,'haze':.24},{'photo_style':'dusty_scrap'})
        original=alpha.copy()
        for kernel in (2,4,6):
            delta,info=foreground_dust(rgb,veiled,alpha,env,air,300,pool_weight=1.,pool_kernel=kernel,fog_weight=0.)
            self.assertEqual(info['fog_extra_max'],0)
            self.assertEqual(info['pool_weight_max'],1)
            self.assertFalse(delta[alpha==0].any())
        previous=0
        for fog in (.12,.24,.36):
            delta,info=foreground_dust(rgb,veiled,alpha,env,air,300,pool_weight=0.,fog_weight=fog)
            self.assertEqual(info['pool_weight_max'],0)
            self.assertGreater(info['fog_extra_max'],previous)
            previous=info['fog_extra_max']
            self.assertLessEqual(info['fog_total_max'],.550001)
            self.assertFalse(delta[alpha==0].any())
        np.testing.assert_array_equal(alpha,original)

    def test_clear_neighbourhood_is_not_forced_to_have_haze(self):
        rgb,alpha,env=checks_display_local.ForegroundDustTests().fixture()
        delta,info=foreground_dust(rgb,rgb,alpha,env,{'amount':0.,'color_linear':[.3]*3},300,
                                  pool_weight=0.,fog_weight=.36)
        self.assertFalse(delta.any());self.assertEqual(info['fog_extra_max'],0)

    def test_blender_paint_dust_is_half_and_layers_do_not_accumulate(self):
        source = f'''
import sys,json
sys.path.insert(0,{str(HERE)!r})
import bpy
from display_batch.scanned_shader import matte_dust_surface
def evaluate(socket):
 if not socket.is_linked:return float(socket.default_value)
 node=socket.links[0].from_node
 if node.type=='TEX_NOISE':return .5
 a,b=[evaluate(s) for s in node.inputs[:2]]
 result={{'ADD':lambda:a+b,'MULTIPLY':lambda:a*b,'MAXIMUM':lambda:max(a,b)}}[node.operation]()
 return min(1,max(0,result)) if node.use_clamp else result
for name,weights,expected in [('paint',[0,1,0,0],.47),('metal',[1,0,0,0],.94)]:
 material=bpy.data.materials.new(name);material.use_nodes=True
 shader=material.node_tree.nodes.get('Principled BSDF')
 partitions={{'materials':{{name:{{'constant':weights,'coverage':weights}}}}}}
 env={{'dust_color':[.13,.14,.15]}}
 args=(material,shader,partitions,env,['paint','steel'],{{'paint_dust_scale':.5}})
 assert matte_dust_surface(*args)
 nodes=len(material.node_tree.nodes)
 assert matte_dust_surface(*args) and len(material.node_tree.nodes)==nodes
 coverage=material.node_tree.nodes['Matte dust coverage']
 value=evaluate(coverage.inputs[0])*evaluate(coverage.inputs[1])
 assert abs(value-expected)<1e-5,(name,value)
 # Half the approved surface coverage, without adding a second dust layer.
 reduced=(material,shader,partitions,env,['paint','steel'],{{'paint_dust_scale':.5,'dust_scale':.5}})
 assert matte_dust_surface(*reduced) and len(material.node_tree.nodes)==nodes
 half=evaluate(coverage.inputs[0])*evaluate(coverage.inputs[1])
 assert abs(half-expected*.5)<1e-5,(name,half)
 assert matte_dust_surface(*args) and len(material.node_tree.nodes)==nodes
 assert abs(evaluate(coverage.inputs[0])*evaluate(coverage.inputs[1])-expected)<1e-5
 dust=material.node_tree.nodes['Matte dust surface']
 assert dust.inputs['Metallic'].default_value==0 and dust.inputs['Specular IOR Level'].default_value==0
print('PAINT_HALF_AND_IDEMPOTENCE_OK')
'''
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'check.py';path.write_text(source,encoding='utf8')
            result=subprocess.run([str(BLENDER),'--background','--factory-startup','--python',str(path)],
                                  capture_output=True,text=True,encoding='utf8',timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PAINT_HALF_AND_IDEMPOTENCE_OK',result.stdout)


if __name__=='__main__':unittest.main(verbosity=2)
