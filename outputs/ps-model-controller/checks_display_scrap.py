"""Regressions for local sampling, UV protection, and material-stage actions."""
import copy
import unittest

import numpy as np

from display_batch.local_scrap import LocalReference, reference_regions, summarize_reference, texture_field
from display_batch.photography import capture_scrap
from display_batch.rendering import part_keys
from display_batch.review import allowed_actions, fusion_actions, repair
from display_batch.schema import BatchConfig, FusionObservation
from display_batch.surface_maps import partition


class ScrapTests(unittest.TestCase):
    def test_mixed_map_preserves_marking_and_uses_linked_metal_values(self):
        rgb = np.float32([[[.56,.57,.58],[.62,.19,.14],[.49,.30,.18],[.86,.86,.85],[.06,.07,.07]]])
        metal = np.float32([[.91,0,0,0,0]])
        roughness = np.float32([[.22,.55,.83,.61,.58]])
        mask = partition('steel, paint, rust, rubber PBR atl',rgb,metal,roughness,['steel','paint','rust','rubber'])
        np.testing.assert_array_equal(mask[0],np.uint8([
            [255,0,0,0],[0,255,0,0],[0,0,255,0],[0,0,0,0],[0,0,0,0]]))
        plastic = partition('black polypropylene cap',rgb,metal*0,roughness,['plastic','steel'])
        self.assertTrue(np.all(plastic[...,:3]==0))
        self.assertTrue(np.all(plastic[...,3]==255))
        self.assertFalse(partition('glass sight',rgb,metal,roughness,['steel']).any())

    def test_candidate_tiles_exclude_foreground_and_protection(self):
        rng = np.random.default_rng(6)
        frame = rng.integers(108,141,(330,500,3),dtype=np.uint8)
        protected = np.zeros((330,500),bool)
        protected[110:139,184:233] = True
        depth = np.full((330,500),.4)
        depth[170:210,130:300] = .7
        scene = {'frame':frame,'allowed':np.ones((330,500),bool),'protected':protected,
                 'depth_native':depth,'supports':[{'id':'S1','depth':.4}]}
        candidate = {'site_id':'S1','target_px':190,'anchor_px':[230,170]}
        bounds = [96,60,398,277]
        rgb,valid,windows = reference_regions(scene,candidate,bounds)
        self.assertGreater(len(windows),0)
        for l,t,r,b in windows:
            area = np.s_[t+60:b+60,l+96:r+96]
            self.assertFalse(protected[area].any())
            self.assertTrue(np.all(depth[area]==.4))
        frame[:60] = (255,0,255)
        _,other_valid,other_windows = reference_regions(scene,candidate,bounds)
        self.assertEqual(windows,other_windows)
        np.testing.assert_array_equal(valid,other_valid)
        answer = LocalReference(patches=[],light_direction=None,light_confidence=0.,evidence='无可比较废钢')
        profile = summarize_reference(rgb,valid,windows,answer,{'dust_color':[.2,.18,.16]})
        self.assertEqual(profile['status'],'unknown')
        tile = texture_field('1234567812345678',profile['statistics'])
        np.testing.assert_array_equal(tile,texture_field('1234567812345678',profile['statistics']))
        self.assertGreater(tile[...,0].std(),10.)

    def test_capture_keeps_curved_surface_tones_when_reference_brightness_changes(self):
        raw = np.full((32,32,3),.12,np.float32)
        raw[:,16:] = .65
        alpha = np.ones((32,32),np.float32)
        alpha[11:16,6:11] = 0
        saved = alpha.copy()
        env = {'sensor':{'chroma':2.},'calibration':{'limited':False},'material_classes':['steel']}
        low,info = capture_scrap(raw,alpha,np.full((50,3),74,np.float32),{},env)
        high,_ = capture_scrap(raw,alpha,np.full((50,3),181,np.float32),{},env)
        np.testing.assert_array_equal(low,high)
        np.testing.assert_array_equal(alpha,saved)
        self.assertGreater(float(low[4,24].mean()-low[4,4].mean()),.2)
        self.assertEqual(info['surface_wear_rms'],0.)
        self.assertEqual(info['contrast_gain'],1.)

    def test_render_dependencies_follow_material_lighting_and_camera_stages(self):
        request = {'appearance':'local_scrap','mesh_hash':'model','matrix':[],'yaw':0,'camera':{},
                   'plane_normal':[0,0,1],'roughness_delta':0.,'dust_gain':1.,'dust_control':1.,
                   'material_classes':['steel'],'surface_texture_scale':2.,'light_gain':1.,
                   'windows':{'object':{'frame':512},'shadow':{'frame':512}},
                   'environment':{'world_color':[.4,.4,.4],'haze':0.,'scrap_surface':{'variant_signature':'A'}}}
        base = part_keys(request,'OPTIX','worker')
        changed = part_keys({**request,'dust_control':1.3},'OPTIX','worker')
        self.assertNotEqual(changed['object'],base['object'])
        self.assertNotEqual(changed['with_shadow'],base['with_shadow'])
        self.assertEqual(changed['without_shadow'],base['without_shadow'])
        variant = copy.deepcopy(request)
        variant['environment']['scrap_surface']['variant_signature'] = 'B'
        self.assertNotEqual(part_keys(variant,'OPTIX','worker')['object'],base['object'])
        self.assertEqual(part_keys({**request,'haze':'more','tone':'darker'},'OPTIX','worker'),base)
        lights = part_keys({**request,'light_gain':.65},'OPTIX','worker')
        self.assertTrue(all(lights[k]!=base[k] for k in base))

    def test_excess_surface_dust_triggers_a_material_change(self):
        config = BatchConfig(backgrounds='x',output='y',material_adaptation='local_scrap')
        self.assertEqual(config.appearance,'local')
        self.assertEqual(config.asset_scale_factors,{'31':.375,'33':.25})
        historical = BatchConfig(backgrounds='x',output='y',asset_scale_factors={'31':.5})
        self.assertEqual(historical.asset_scale_factors,{'31':.5})
        scores = dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),3)
        review = FusionObservation(trace='visible',scores=scores,evidence='粉尘偏厚',
            defects=[{'code':'dust_heavy','severity':'major','evidence':'缸体积灰过厚'}]).review()
        params = {'candidate':'A','photo_model':'local','material_adaptation':'local_scrap',
                  'material':'original','light':'scene','tone':'scene',
                  **dict.fromkeys(('chroma','dust','contrast','haze','sharpness'),'auto')}
        comp = {'photo':{'surface_dust_percentiles':[.1,.3,.6]},'checks':{}}
        allowed = allowed_actions(params,comp,['A'],render_style='local_scrap')
        actions = fusion_actions(review,comp,allowed)
        updated,log = repair(params,review,allowed,['A'],lambda p:False,dust_in_shader=True,suggestions=actions)
        self.assertEqual(updated['dust'],'less')
        self.assertEqual(updated['haze'],'auto')
        self.assertEqual(log[0]['stage'],'material')
        self.assertNotIn('contrast',allowed)
        self.assertFalse(review.photo_passes())


if __name__=='__main__':
    unittest.main(verbosity=2)
