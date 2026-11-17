"""Cause-specific material/light controls and compatibility at public boundaries."""
import copy
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/ps-model-controller'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from scrap_composite.render_controls import DEFAULTS, validated, RENDER_CONTROLS
from scrap_composite.photography_controls import composite_changes
from display_batch.effects import parameters
from display_batch.schema import EffectSettings, Action, apply_actions, TONE_PROFILES, LIGHT_PROFILES
from display_batch.physical_material import refine_finish
from display_batch.local_light import coherent_rig
from display_batch.rendering import render_request, part_keys
from display_batch.review import allowed_actions, repair
from display_batch.schema import EffectsReview


class RegisteredRenderTests(unittest.TestCase):
    def test_opt_in_and_absolute_replay(self):
        self.assertNotIn('oxide_gain', parameters(EffectSettings()))
        effects=EffectSettings(surface_finish='physical_layers',surface_finish_manifest='fixture',
                               asset_refinement='component_v1',render_controls={'oxide_gain':1.5})
        params=parameters(effects)
        self.assertEqual(params['oxide_gain'],'1.5')
        changed=apply_actions(params,[Action(kind='oxide_gain',value='0.5')],[])
        self.assertEqual(changed['oxide_gain'],'0.5')
        with self.assertRaises(ValueError):
            apply_actions(parameters(EffectSettings()),[Action(kind='oxide_gain',value='0.5')],[])
        for bad in ({'oxide_gain':True},{'oxide_gain':float('nan')},{'oxide_gain':2}, {'fake':1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):validated(bad)

    def test_material_overrides_do_not_compound_or_mutate_source(self):
        source={'signature':'source','recipe':{'oxide_max':.3},'regions':{},'maps':{}}
        original=copy.deepcopy(source)
        one=refine_finish(source,{'oxide_gain':1.5})
        two=refine_finish(one,{'oxide_gain':1.5})
        self.assertEqual(one,two)
        self.assertEqual(source,original)
        self.assertEqual(one['recipe'],source['recipe'])
        self.assertNotEqual(one['signature'],source['signature'])

    def test_light_width_and_fill_are_independent_actual_rig_changes(self):
        env={'local_irradiance':{'direction':{'direction':None,'confidence':0},'level':1.,'fill_ratio':.3}}
        camera={'eye':[0,-6,6],'target':[0,0,0]}
        sky,lights=coherent_rig(env,camera)
        sky2,wide=coherent_rig({**env,'photographic_light':{'light_width_scale':1.25,'fill_gain':1.}},camera)
        _,filled=coherent_rig({**env,'photographic_light':{'light_width_scale':1.,'fill_gain':1.25}},camera)
        np.testing.assert_array_equal(sky,sky2)
        self.assertEqual(lights[0][1],wide[0][1])
        self.assertAlmostEqual(lights[0][2]*1.25,wide[0][2])
        self.assertEqual(lights[0],filled[0])
        self.assertAlmostEqual(lights[1][1]*1.25,filled[1][1])

    def test_typed_operations_have_physical_cause_and_unit(self):
        params={'asset_refinement':'component_v1',**DEFAULTS}
        changes=composite_changes(params,RENDER_CONTROLS,{},tone_profiles=TONE_PROFILES,light_profiles=LIGHT_PROFILES)
        by_key={c.control:c for c in changes}
        self.assertEqual(set(by_key),set(RENDER_CONTROLS))
        self.assertEqual(by_key['roughness_shift'].cause,'surface_roughness')
        self.assertEqual(by_key['light_width_scale'].cause,'light_size')
        self.assertEqual(by_key['roughness_shift'].unit,'roughness')
        fill=next(c for c in changes if c.control=='fill_gain' and c.direction=='increase')
        self.assertLess(float(fill.value),float(fill.current))
        self.assertEqual(fill.factor,'key_fill_ratio')

    def test_material_change_keeps_empty_receiver_light_change_invalidates_it(self):
        request={'windows':{'object':{'x':1},'shadow':{'x':2}},'environment':{},'appearance':'local',
                 'camera':{},'plane_normal':[0,0,1],'light_gain':1,'mesh_hash':'mesh','matrix':[],
                 'yaw':0,'roughness_delta':0,'dust_gain':1,'dust_control':1,'material_classes':['steel']}
        base=part_keys(request,'CPU','dep')
        material=part_keys({**request,'roughness_delta':.08},'CPU','dep')
        light=part_keys({**request,'environment':{'photographic_light':{'light_width_scale':1.25}}},'CPU','dep')
        self.assertEqual(base['without_shadow'],material['without_shadow'])
        self.assertNotEqual(base['with_shadow'],material['with_shadow'])
        self.assertNotEqual(base['without_shadow'],light['without_shadow'])

    def test_nonferrous_oxide_and_legacy_paths_do_not_offer_ineffective_controls(self):
        from display_batch.pipeline import Pipeline
        from display_batch.schema import BatchConfig
        pipe=Pipeline.__new__(Pipeline)
        pipe.config=BatchConfig(preset='realism_v1',backgrounds='fixture',output='fixture',
                                effects={'asset_refinement':'component_v1'})
        params=pipe.default_params('S1')
        photo={'material_readback':{'control_domains':{'oxide_gain':False,'scratch_gain':True}},
               'lighting_readback':{'control_domains':{'fill_gain':True}}}
        opts=allowed_actions(params,{'photo':photo},['S1'],render_style='local')
        self.assertNotIn('oxide_gain',opts)
        self.assertIn('scratch_gain',opts)
        self.assertIn('fill_gain',opts)
        legacy=allowed_actions({**params,'review_profile':'legacy'},{'photo':photo},['S1'],render_style='local')
        self.assertFalse(set(RENDER_CONTROLS)&set(legacy))
        review=EffectsReview(trace='visible',scores=dict.fromkeys(
            ('support','occlusion','lighting','edge','texture','material','structure'),2),
            defects=[],actions=[Action(kind='oxide_gain',value='1.5')],evidence='fixture')
        updated,log=repair(params,review,{'oxide_gain':['1.5']},['S1'],lambda _:False)
        self.assertIsNone(updated)
        self.assertEqual(log[0]['disposition'],'requires_typed_photographic_diagnosis')


if __name__=='__main__':unittest.main()
