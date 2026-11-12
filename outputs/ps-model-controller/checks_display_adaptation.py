"""Scale/air/material-control regressions independent of online aesthetic scores."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pydantic import ValidationError
from display_batch.schema import EffectSettings, BatchConfig, EffectsReview, Scores, Defect, Action
from display_batch.consistent_scale import asset_profile, scaled_asset, local_scale, calibration_to_native, scene_calibration
from display_batch.local_atmosphere import estimate, prepare, fields_for
from display_batch.effects import parameters, optical_delta, render_identity, dust_profile
from display_batch.local import neighbourhood
from display_batch.review import allowed_actions, repair
from display_batch.seams import audit_boundary
import checks_display_local_light


class AdaptationTests(unittest.TestCase):
    def test_one_length_and_user_factor_are_idempotent(self):
        asset={'semantic':{'major_mm':[450.,900.]}}
        first=scaled_asset(asset,.375);second=scaled_asset(first,.375)
        self.assertEqual(first['physical_size'],second['physical_size'])
        self.assertAlmostEqual(first['physical_size']['nominal_mm'],np.sqrt(450*900)*.375)
        self.assertNotIn('physical_size',asset)

    def test_shared_reference_scale_changes_projection_not_asset_size(self):
        s={'size':[1000,700],'answer':{'hopper':[[0,0],[1,0],[1,1]],'elevation':[45,45]},
           'supports':[{'id':'S1','reference':[[.1,.5],[.3,.5]],'reference_mm':[200,200],
                        'depth':.4,'confidence':1.},{'id':'S2','reference':[[.5,.2],[.6,.2]],
                        'reference_mm':[200,200],'depth':.7,'confidence':1.}]}
        profile=asset_profile({'semantic':{'major_mm':[100,400]}})
        near,e=local_scale(s,s['supports'][0],[200,350],profile)
        far,f=local_scale(s,s['supports'][1],[550,140],profile)
        self.assertAlmostEqual(near[0]/far[0],2.)
        self.assertEqual(near[0],near[1]);self.assertEqual(e['physical_size'],f['physical_size'])
        self.assertEqual(e['rejected_references'][0]['reason'],'cross_depth_reference')
        self.assertEqual(e['status'],'unknown')
        original=copy.deepcopy(e)
        native=calibration_to_native(e,.25,[4000,2800])
        self.assertEqual(e,original)
        self.assertAlmostEqual(native['pixels_per_mm'],e['pixels_per_mm']*4)
        self.assertEqual(native['references'][0]['anchor_px'],[v*4 for v in e['references'][0]['anchor_px']])
        self.assertEqual(native['physical_size'],e['physical_size'])

    def test_config_backward_compatibility_and_dust_cap(self):
        self.assertEqual(EffectSettings().fog_mode,'manual_extra')
        self.assertEqual(BatchConfig(backgrounds='.',output='x').scale_mode,'legacy')
        with self.assertRaises(ValidationError):EffectSettings(weathering='preserve_source',surface_dust=.75)
        self.assertEqual(dust_profile(.5,'preserve_source')['paint_dust_scale'],.5)

    def test_inherited_and_duplicate_references_do_not_become_multiple_depth_rulers(self):
        source={'id':'S','reference':[[.1,.2],[.2,.2]],'reference_mm':[100,100],
                'depth':.7,'confidence':1.,'reference_status':'measured'}
        derived={**source,'id':'D','depth':.4,'plane_basis':'segmentation_only_hidden_height_unknown'}
        duplicate={**source,'id':'Q','reference':list(reversed(source['reference']))}
        scene={'size':[1000,700],'answer':{'hopper':[[0,0],[1,0],[1,1]],'elevation':[45,45]},
               'supports':[source,derived,duplicate]}
        calibration=scene_calibration(scene)
        self.assertEqual(len(calibration['references']),1)
        self.assertEqual({r['reason'] for r in calibration['rejected_references']},
                         {'inherited_reference_has_no_local_measurement','duplicate_physical_reference'})
        profile=asset_profile({'semantic':{'major_mm':[200,200]},'measured_major_mm':200})
        _,evidence=local_scale(scene,derived,[200,300],profile)
        self.assertEqual(evidence['status'],'unknown')
        self.assertEqual(evidence['basis'],'shared_uncalibrated_scene_estimate')

    def fixture(self):
        scene,pose=checks_display_local_light.LocalLightTests().fixture();scene['key']='synthetic'
        env,_=neighbourhood(scene,pose)
        return scene,pose,env

    def test_remote_blue_highlight_cannot_tint_local_air(self):
        s,p,e=self.fixture();a,summary=estimate(s,p,e)
        l,t,r,b=e['local_context']['bounds'];mask=np.ones(s['allowed'].shape,bool);mask[t:b,l:r]=False
        s['frame'][mask]=[30,100,255]
        aa,ss=estimate(s,p,e)
        np.testing.assert_array_equal(a['airlight'],aa['airlight']);self.assertEqual(summary,ss)
        self.assertLess(np.max(a['airlight'])-np.min(a['airlight']),.03)

    def test_clear_and_off_air_leave_geometric_alpha(self):
        s,p,e=self.fixture();s['atmosphere']={'summary':{'visual_evidence':{'kind':'clear','confidence':1.}}}
        f,q=estimate(s,p,e);np.testing.assert_array_equal(f['transmission'],1.)
        with tempfile.TemporaryDirectory() as cache:
            context=prepare(s,p,e,cache);bounds=p['bounds'];x,y,r,b=bounds
            shape=(b-y,r-x)
            ot,bt,air,_=fields_for(context,bounds,np.zeros(shape),{},p,s,1.5,False)
            np.testing.assert_array_equal(ot,1.);np.testing.assert_array_equal(bt,1.)

    def test_auto_fog_is_not_an_additional_35_percent_layer(self):
        s,p,e=self.fixture();e['camera_psf']['edge_samples']=30
        rgb=np.full((20,20,3),.5,np.float32);alpha=np.ones((20,20),np.float32);alpha[8:12,8:12]=0
        old=alpha.copy();params=parameters(EffectSettings(fog_mode='local_auto'))
        air={'transmission_field':np.full(alpha.shape,.8,np.float32),'color_linear':[.3]*3}
        delta,measured=optical_delta(rgb,rgb,alpha,e,air,100,params)
        self.assertAlmostEqual(measured['fog_total_median'],.2,places=5)
        self.assertEqual(measured['fog_extra_max'],0.);np.testing.assert_array_equal(alpha,old)
        self.assertLess(float(abs(delta).max()),1e-6)

    def test_fog_and_pool_do_not_invalidate_render_but_weathering_does(self):
        p={'candidate':'a','material':'original','light':'scene',**parameters(EffectSettings())}
        key=render_identity(p,'local')
        self.assertEqual(key,render_identity({**p,'fog_gain':'1.5','pool_kernel':'6'},'local'))
        self.assertNotEqual(key,render_identity({**p,'weathering':'preserve_source'},'local'))

    def test_auto_fog_after_camera_blur_stays_inside_visible_alpha(self):
        from display_batch.local import camera_residual
        from display_batch.photography import linear
        from display_batch.compose import srgb
        _,_,env=self.fixture();env['camera_psf'].update(edge_samples=50,sigma_px=1.2)
        clear=np.full((80,90,3),.4,np.float32);alpha=np.zeros(clear.shape[:2],np.float32)
        alpha[15:65,20:70]=1.;alpha[30:45,35:50]=0.
        writable=np.ones(alpha.shape,bool);writable[50:,:30]=False
        params=parameters(EffectSettings(fog_mode='local_auto',pool_kernel=1))
        air={'transmission_field':np.full(alpha.shape,.7,np.float32),'color_linear':[.5]*3}
        fogged=srgb(linear(clear)*.7+.5*.3)
        delta,_=optical_delta(clear,fogged,alpha,env,air,100,params)
        np.testing.assert_allclose(delta[alpha>0],(linear(fogged)-linear(clear))[alpha>0],atol=1e-6)
        before,_,a=camera_residual(linear(clear),clear,alpha,linear(clear),env,params,3,writable)
        after,_,b=camera_residual(linear(clear),clear,alpha,linear(clear),env,params,3,writable,object_delta=delta)
        protected=(alpha==0)|~writable
        np.testing.assert_array_equal(after[protected],before[protected])
        self.assertGreater(float(abs(after-before).max()),.03)
        self.assertEqual(a['added_noise_sigma'],b['added_noise_sigma'])

    def test_inner_rectangular_cutoff_is_detected(self):
        before=np.full((120,160,3),130,np.uint8);after=before.copy();after[8:112,8:152]-=50
        receiver=np.ones((120,160),bool);alpha=np.zeros(receiver.shape,np.float32)
        self.assertFalse(audit_boundary(before,after,alpha,receiver)['pass'])
        self.assertTrue(audit_boundary(before,before,alpha,receiver)['pass'])

    def test_review_does_not_offer_heavy_dust_or_wrong_fog_mode(self):
        from checks_display_effects import params,comp
        p=params(**parameters(EffectSettings(weathering='preserve_source',fog_mode='local_auto')))
        choices=allowed_actions(p,comp(),['C1'],render_style='local')
        self.assertNotIn('fog_extra',choices);self.assertIn('fog_gain',choices)
        self.assertTrue(all(float(x)<=.5 for x in choices['surface_dust']))
        review=EffectsReview(scores=Scores(**dict.fromkeys(['support','occlusion','lighting','edge','texture','material','structure'],3)),
            trace='visible',defects=[Defect(code='identity_loss',severity='major',region=(0.,0.,1.,1.),evidence='灰层遮盖原色')],
            actions=[Action(kind='surface_dust',value='0.25')],evidence='原貌被灰层遮盖')
        updated,_=repair(p,review,choices,['C1'],lambda _:False)
        self.assertEqual(updated['surface_dust'],'0.25')

    def test_severe_visual_defect_cannot_win_by_fewer_unknowns(self):
        from display_batch.schema import version_rank
        scores=Scores(**dict.fromkeys(['support','occlusion','lighting','edge','texture','material','structure'],3))
        good=EffectsReview(scores=scores,trace='none',defects=[],actions=[],evidence='可用')
        bad=good.model_copy(update={'defects':[Defect(code='identity_loss',severity='major',region=(0.,0.,1.,1.),evidence='原貌丢失')]})
        self.assertGreater(version_rank({'scale':{'status':'unknown'}},good),version_rank({'scale':{'status':'pass'}},bad))
        self.assertGreater(version_rank({'scale':{'status':'unknown'}},bad),version_rank({'scale':{'status':'fail'}},good))

    def test_insufficient_local_light_has_a_renderable_soft_fallback(self):
        from display_batch.effects import lighting_for
        from display_batch.local_light import light_rig
        s,p,e=self.fixture()
        with patch('display_batch.local_light.estimate_local_light',side_effect=ValueError('insufficient_local_neutral_light_evidence')):
            fallback=lighting_for(s,p,e,'local',preserve=True)
        sky,lights=light_rig(fallback,{},1.)
        self.assertTrue(np.isfinite(sky).all());self.assertEqual(len(lights),2)


if __name__=='__main__':unittest.main()
