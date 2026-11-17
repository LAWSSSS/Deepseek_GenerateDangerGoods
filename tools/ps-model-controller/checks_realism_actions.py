"""Realism controls must describe actual pixels and cause-specific actuators."""
import json
import unittest
from unittest.mock import AsyncMock, patch

import batch_display
import numpy as np
from display_batch.photography import match_linear
from display_batch.pipeline import Pipeline
from display_batch.review import allowed_actions
from display_batch.schema import BatchConfig, LIGHT_PROFILES, PHOTO_CONTROLS, TONE_PROFILES
from scrap_composite.photography_agent import run_photography, scope_plan
from scrap_composite.photography_controls import composite_changes
from scrap_composite.photography_protocol import PhotoBriefV2, PhotoContext, PhotoReport


def fixture():
    pipe=Pipeline.__new__(Pipeline)
    pipe.config=BatchConfig(preset='realism_v1',backgrounds='fixture',output='fixture',effects={'ai_adjust':True})
    params=pipe.default_params('S1')
    photo={'lighting_basis':'bounded-broad-light-v1','calibration':{'version':'bounded-broad-light-v1'},
        'effects':{'edge_samples':30,'sample_pixels':200,'fog_base':.2,'fog_total_median':.2,
                   'surface_dust_applied':True,'generated_finish':{'materials':['steel']}},
        'local_context':{'confidence':'estimated'},'atmosphere':{'confidence':'estimated'},'blur_sigma':.4,
        'effective_controls':{label:{'chroma':1/v,'contrast':1+.2*v,'sharpness':.4/v,'haze':.2*v}
                              for label,v in PHOTO_CONTROLS['sharpness'].items()}}
    allowed=allowed_actions(params,{'photo':photo},['S1','S2'],render_style='local')
    return params,photo,allowed


def report(factor,direction='decrease',state='too_high'):
    value={'trace':'visible','scores':dict.fromkeys(
        ('support','occlusion','lighting','edge','texture','material','structure'),2),
        'findings':[{'factor':factor,'direction':direction,'observed_state':state,
            'region':[.3,.3,.6,.6],'severity':'major','visibility':'visible','confidence':.9,
            'cue':'目标对应摄影量高于同层参照','reference':'左侧同层表面',
            'reference_role':'scene','reference_region':[.1,.3,.25,.6],'physical_cue':None}]}
    context=PhotoContext(images=['frame'])
    return PhotoReport.bind(PhotoBriefV2.model_validate_json(json.dumps(value)),'a'*64,'object',context)


class RealismActionTests(unittest.IsolatedAsyncioTestCase):
    def test_real_pixels_and_report_show_bypassed_contrast_chroma(self):
        params,_,_=fixture()
        yy,xx=np.mgrid[:40,:48]
        raw=np.stack((.1+xx*.005,.16+yy*.002,.12+xx*.003),axis=-1).astype(np.float32)
        alpha=np.ones(raw.shape[:2],np.float32);highlight=np.zeros_like(alpha)
        pixels=np.tile(np.float32([[90,87,84],[130,126,120]]),(100,1))
        environment={'material_classes':['steel'],'calibration':{'white_balance':[1.,1.,1.],
            'exposure_ev':0.,'limited':False},'sensor':{'noise_sigma':0.,'edge_energy':.05}}
        with patch('display_batch.photography.display',side_effect=lambda value:np.clip(value,0,1)):
            first,record=match_linear(raw,alpha,pixels,highlight,params,1,environment,
                                      camera_capture=False,surface_weathering=False)
            second,changed=match_linear(raw,alpha,pixels,highlight,
                {**params,'chroma':'more','contrast':'less'},1,environment,
                camera_capture=False,surface_weathering=False)
        np.testing.assert_array_equal(first,second)
        for measured in (record,changed):
            for entry in measured['effective_controls'].values():
                self.assertEqual(entry['chroma'],1.)
                self.assertEqual(entry['contrast'],1.)

    def test_ignored_controls_are_removed_even_with_stale_measurements(self):
        params,photo,allowed=fixture()
        ignored={'chroma','contrast','dust','haze','pool_kernel','material'}
        self.assertFalse(ignored & set(allowed))
        self.assertTrue({'light','tone','surface_dust','fog_gain','sharpness'} <= set(allowed))
        unsafe={**allowed,**{key:['less','more'] for key in ('chroma','contrast','dust','haze')}}
        changes=composite_changes(params,unsafe,photo,tone_profiles=TONE_PROFILES,light_profiles=LIGHT_PROFILES)
        self.assertFalse(ignored & {change.control for change in changes})

    def test_separate_light_dust_air_and_no_reflection_to_dust_fallback(self):
        params,photo,allowed=fixture()
        changes=composite_changes(params,allowed,photo,tone_profiles=TONE_PROFILES,light_profiles=LIGHT_PROFILES)
        for factor,control,stage in [('light_ratio','light','render'),('dust_coverage','surface_dust','render'),
                                     ('haze_amount','fog_gain','composite')]:
            value=report(factor)
            scope=scope_plan(value,value.context,changes,parameters=params)
            self.assertTrue(scope.options)
            self.assertEqual({o.control for o in scope.options},{control})
            self.assertEqual({o.stage for o in scope.options},{stage})
        for factor in ('specular_width','specular_shape','material_response','key_direction'):
            value=report(factor,'unknown','mismatch')
            scope=scope_plan(value,value.context,changes,parameters=params)
            self.assertEqual(scope.reason,'no_supported_control')
            self.assertFalse(scope.options)

    async def test_disabled_contrast_never_calls_executor_or_schedules_review(self):
        params,photo,allowed=fixture()
        changes=composite_changes(params,allowed,photo,tone_profiles=TONE_PROFILES,light_profiles=LIGHT_PROFILES)
        value=report('tonal_contrast')
        client=type('Client',(),{'ask':AsyncMock()})()
        decision=await run_photography(client,value.context,'a'*64,report=value,
                                      plan=True,changes=changes,parameters=params)
        self.assertEqual(decision.plan.reason,'no_supported_control')
        self.assertEqual(decision.updates,{})
        client.ask.assert_not_called()

    def test_zero_actual_blur_cannot_offer_more_sharpness(self):
        params,photo,allowed=fixture();photo['blur_sigma']=0.
        changes=composite_changes(params,allowed,photo,tone_profiles=TONE_PROFILES,light_profiles=LIGHT_PROFILES)
        self.assertFalse(any(c.factor=='sharpness' and c.direction=='increase' for c in changes))


if __name__=='__main__':unittest.main(verbosity=2)
