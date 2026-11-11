"""Regressions for local-only matching and rectangular receiver seams."""
import copy
import unittest
import cv2
import numpy as np

from display_batch.local import neighbourhood, camera_residual, seam_measurement, local_veil, foreground_dust, pool_object_rgb
from display_batch.compose import taper_shadow
from display_batch.photography import linear, match_linear
from display_batch.schema import BatchConfig
from display_batch.geometry import camera_for
from display_batch.rendering import part_keys


class LocalTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(2)
        gray=rng.integers(90,170,(300,440),dtype=np.uint8)
        scene={'size':[440,300],'frame':np.repeat(gray[...,None],3,axis=2),
            'allowed':np.ones((300,440),bool),'protected':np.zeros((300,440),bool),
            'depth_native':np.full((300,440),.5),'answer':{'haze':.8,'light_direction':[1,0]},
            'supports':[{'id':'S1','depth':.5,'mask':np.ones((300,440),bool)}]}
        candidate={'bounds':[170,120,260,210],'target_px':90,'site_id':'S1','support_ids':['S1']}
        return scene,candidate

    def test_remote_light_colour_and_haze_do_not_change_local_match(self):
        scene,candidate=self.fixture()
        candidate.update(camera=camera_for(np.array([[-.5,-.4,0],[.5,.4,.6]]),45),full_px=160,offset=[130,80])
        env,mask=neighbourhood(scene,candidate)
        x,y,r,b=env['local_context']['bounds']
        remote=np.ones(mask.shape,bool);remote[y:b,x:r]=False
        scene['frame'][remote]=[255,10,240]
        scene['answer']={'haze':0,'light_direction':[-1,1]}
        other,othermask=neighbourhood(scene,candidate)
        self.assertEqual(env,other)
        np.testing.assert_array_equal(mask,othermask)
        self.assertFalse(mask[remote].any())

    def test_insufficient_support_never_expands_to_whole_frame(self):
        scene,candidate=self.fixture()
        scene['supports'][0]['mask'][:]=False
        env,mask=neighbourhood(scene,candidate)
        self.assertFalse(env['local_context']['global_fallback'])
        x,y,r,b=env['local_context']['bounds']
        outside=mask.copy();outside[y:b,x:r]=False
        self.assertFalse(outside.any())
        self.assertLessEqual(r-x,round((candidate['bounds'][2]-candidate['bounds'][0])+candidate['target_px']*.6)+2)
        self.assertEqual(env['basis'],'nearby_same_depth_surface_estimate')

    def test_foreground_and_protection_are_excluded_from_sampling(self):
        scene,candidate=self.fixture()
        scene['protected'][100:130,160:210]=True
        scene['depth_native'][210:230,160:270]=.9
        _,mask=neighbourhood(scene,candidate)
        self.assertFalse(mask[scene['protected']].any())
        self.assertFalse(mask[scene['depth_native']>.8].any())

    def test_receiver_cutoff_is_softened_without_touching_object_alpha(self):
        receiver=np.zeros((120,150),bool);receiver[15:105,20:135]=True
        loss=receiver.astype(np.float32)*.35
        alpha=np.zeros_like(loss);alpha[45:80,60:105]=1
        original=alpha.copy()
        before=seam_measurement(loss,receiver,alpha)
        after=taper_shadow(loss,24,receiver)
        self.assertLess(seam_measurement(after,receiver,alpha)['loss_p95'],before['loss_p95']*.1)
        self.assertAlmostEqual(float(after[60,80]),.35,places=6)
        self.assertFalse(after[~receiver].any())
        np.testing.assert_array_equal(alpha,original)

    def test_camera_filter_never_pastes_a_rectangular_background(self):
        rgb=np.full((90,120,3),.7,np.float32)
        background=linear(np.full_like(rgb,.4))
        alpha=np.zeros((90,120),np.float32);alpha[30:65,45:85]=1
        alpha[40:52,55:65]=0
        original=alpha.copy();writable=np.ones_like(alpha,bool);writable[:,:30]=False
        env={'camera_psf':{'sigma_px':.85},'sensor':{'noise_sigma':.003}}
        out,edited,info=camera_residual(background,rgb,alpha,background,env,{},12,writable)
        np.testing.assert_array_equal(alpha,original)
        self.assertFalse(edited[:10].any());self.assertFalse(edited[:,:30].any())
        np.testing.assert_allclose(out[:10],.4,atol=1e-6)
        self.assertGreater(info['blur_sigma'],.2)
        self.assertGreaterEqual(info['optical_iou'],.995)
        self.assertFalse(edited[45:48,59:61].any())
        repeated=camera_residual(background,rgb,alpha,background,env,{},12,writable)
        np.testing.assert_array_equal(out,repeated[0])

    def test_empty_insert_has_no_camera_or_noise_rectangle(self):
        image=np.full((40,60,3),.4,np.float32);bg=linear(image)
        env={'camera_psf':{'sigma_px':1.},'sensor':{'noise_sigma':.01}}
        out,edited,_=camera_residual(bg,image,np.zeros((40,60),np.float32),bg,env,{},1,np.ones((40,60),bool))
        self.assertFalse(edited.any())
        np.testing.assert_allclose(out,image,atol=1e-6)

    def test_local_veil_off_and_material_independence(self):
        source=np.full((30,40,3),.25,np.float32);alpha=np.ones((30,40),np.float32)
        env={'material_classes':['steel'],'sensor':{'surface_detail_rms':.018},
             'local_context':{'version':'test'},'haze':.1,'dust_color':[.2,.2,.2]}
        args=(source,alpha,np.full((100,3),140,np.float32),np.zeros_like(alpha),{},1)
        a,ai=match_linear(*args,env,reference=True,camera_capture=False)
        b,bi=match_linear(*args,{**env,'haze':0},reference=True,camera_capture=False)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ai['dust_mean'],bi['dust_mean'])
        off,stats=local_veil(a,env,{'atmosphere':'off'})
        np.testing.assert_allclose(a,off,atol=1e-6)
        self.assertEqual(stats['amount'],0)
        on,stats=local_veil(a,env,{'atmosphere':'auto'})
        self.assertGreater(float(np.max(abs(on-off))),.001)
        self.assertGreater(stats['amount'],0)

    def test_texture_changes_do_not_recompute_empty_receiver(self):
        request={'appearance':'local','camera':{},'plane_normal':[0,0,1],'mesh_hash':'g','matrix':[],
            'yaw':0,'roughness_delta':0.,'dust_gain':1.,'dust_control':1.,'material_classes':['steel'],
            'environment':{'world_color':[.2,.2,.2],'haze':0.,'local_context':{'surface_texture_rms':.01}},
            'light_gain':1.,'windows':{'object':{'frame':512,'rect':[100,100,400,400]},
                                     'shadow':{'frame':512,'rect':[50,50,450,450]}}}
        first=part_keys(request,'OPTIX','test')
        request['environment']['local_context']['surface_texture_rms']=.03
        second=part_keys(request,'OPTIX','test')
        self.assertNotEqual(first['object'],second['object'])
        self.assertNotEqual(first['with_shadow'],second['with_shadow'])
        self.assertEqual(first['without_shadow'],second['without_shadow'])

    def test_sensor_noise_only_adds_the_measured_variance_deficit(self):
        rng=np.random.default_rng(5)
        rgb=np.clip(.5+rng.normal(0,.035,(100,100,3)),0,1).astype(np.float32)
        alpha=np.ones((100,100),np.float32)
        background=linear(np.full_like(rgb,.4))
        env={'camera_psf':{'sigma_px':.38},'sensor':{'noise_sigma':.001}}
        _,_,info=camera_residual(background,rgb,alpha,background,env,{},12,np.ones_like(alpha,bool))
        self.assertEqual(info['added_noise_sigma'],0.)
        self.assertEqual(info['blur_sigma'],0.)

    def test_local_surface_supplement_is_bounded_and_alpha_stays_fixed(self):
        alpha=np.ones((48,60),np.float32);alpha[15:25,20:30]=0
        source=np.full((48,60,3),.25,np.float32);source[30:33,10:50]=1.2
        env={'material_classes':['steel'],'sensor':{'surface_detail_rms':.008},'local_context':{},'haze':0}
        # A nonempty descriptor activates locally measured texture strength.
        env['local_context']={'version':'test'}
        result,info=match_linear(source,alpha,np.full((400,3),140,np.float32),
            np.zeros_like(alpha),{},2,env,metallic=np.ones_like(alpha),reference=True,camera_capture=False)
        self.assertLess(info['surface_wear_rms'],.01)
        self.assertGreater(float(result[31,20,0]-result[10,20,0]),.04)
        self.assertTrue(info['alpha_unchanged'])
        self.assertEqual(BatchConfig(backgrounds='x',output='y',appearance='local').appearance,'local')


class ForegroundDustTests(unittest.TestCase):
    def test_pool_normalizes_partial_coverage_on_odd_grid(self):
        alpha = np.zeros((13, 17), np.float32)
        alpha[2:11, 3:14] = .8
        rgb = np.full((*alpha.shape, 3), .95, np.float32)
        rgb[alpha > 0] = (.2, .4, .6)
        result = pool_object_rgb(rgb, alpha)
        self.assertEqual(result.shape, rgb.shape)
        np.testing.assert_allclose(result[alpha > 0], rgb[alpha > 0], atol=1e-6)

    def fixture(self):
        alpha = np.zeros((41, 47), np.float32)
        alpha[3:36, 5:40] = 1
        alpha[16:25, 17:26] = 0
        alpha[37:40, 7:24] = 1
        alpha[38:40, 7:24] = 0
        rgb = np.full((*alpha.shape, 3), .4, np.float32)
        rgb[7:33:2, 8:37:2] = .65
        env = {'local_context': {'confidence': 'estimated', 'sample_pixels': 120,
                                 'basis': 'nearby_same_support_and_depth'},
               'camera_psf': {'sigma_px': .85, 'edge_samples': 24},
               'sensor': {'noise_sigma': .008}, 'dust_color': [.5, .5, .5]}
        return rgb, alpha, env

    def test_fog_and_pool_fade_around_holes_and_thin_parts(self):
        rgb, alpha, env = self.fixture()
        original_alpha = alpha.copy()
        haze_rgb, local_air = local_veil(rgb, {**env, 'haze': .28}, {'photo_style': 'dusty_scrap'})
        delta, info = foreground_dust(rgb, haze_rgb, alpha, env, local_air, 300)
        self.assertAlmostEqual(info['fog_total_max'], .32, places=6)
        self.assertLessEqual(info['fog_extra_max'], .09)
        self.assertGreaterEqual(info['pool_weight'], .15)
        self.assertLessEqual(info['pool_weight'], .35)
        self.assertFalse(delta[alpha == 0].any())
        self.assertFalse(delta[37, 7:24].any())
        self.assertFalse(delta[15, 19].any())  # Near a transparent hole.
        self.assertTrue(delta[9:13, 11:15].any())
        np.testing.assert_array_equal(alpha, original_alpha)

    def test_unknown_local_evidence_preserves_clear_object(self):
        rgb, alpha, env = self.fixture()
        env['local_context']['confidence'] = 'unknown'
        delta, info = foreground_dust(rgb, rgb, alpha, env, {'amount': .2, 'color_linear': [.5] * 3}, 200)
        self.assertFalse(delta.any())
        self.assertEqual(info['pool_weight'], 0.)
        self.assertEqual(info['fog_extra_max'], 0.)

    def test_pool_and_fog_levels_change_independently(self):
        rgb, alpha, env = self.fixture()
        veiled, air = local_veil(rgb, {**env, 'haze': .15}, {'photo_style': 'dusty_scrap'})
        args = (rgb, veiled, alpha, env, air, 300)
        zero, _ = foreground_dust(*args, pool_weight=0., fog_fraction=0.)
        light_pool, p = foreground_dust(*args, pool_weight=.15, fog_fraction=0.)
        strong_pool, _ = foreground_dust(*args, pool_weight=.45, fog_fraction=0.)
        light_fog, f = foreground_dust(*args, pool_weight=0., fog_fraction=1 / 3)
        strong_fog, _ = foreground_dust(*args, pool_weight=0., fog_fraction=1.)
        self.assertFalse(zero.any())
        self.assertEqual(p['fog_extra_max'], 0.)
        self.assertEqual(f['pool_weight'], 0.)
        self.assertTrue(light_pool.any() and light_fog.any())
        np.testing.assert_allclose(strong_pool, 3 * light_pool, atol=1e-6)
        np.testing.assert_allclose(strong_fog, 3 * light_fog, atol=1e-6)

    def test_fixed_tone_keeps_exposure_and_contrast_in_material_comparison(self):
        alpha = np.ones((30, 40), np.float32)
        raw = np.full((30, 40, 3), .4, np.float32)
        samples = np.full((120, 3), 135, np.float32)
        highlight = np.zeros_like(alpha)
        env = {'material_classes': ['steel']}
        _, before = match_linear(raw, alpha, samples, highlight, {}, 2, env, reference=True)
        _, after = match_linear(raw * .65, alpha, samples, highlight, {}, 2, env,
                                reference=True, fixed_tone=before)
        for k in ('exposure_ev', 'contrast_gain', 'chroma_gain', 'white_balance'):
            self.assertEqual(before[k], after[k])

    def test_added_residual_cannot_bleed_past_visible_alpha_or_change_noise(self):
        rgb, alpha, env = self.fixture()
        writable = np.ones_like(alpha, bool)
        writable[:, 32:] = False
        bg = linear(np.full_like(rgb, .5))
        shadow = bg.copy()
        shadow[33:38, 3:31] *= .8
        args = (shadow, rgb, alpha, bg, env, {}, 19, writable)
        base, edited, optics = camera_residual(*args)
        trial, trial_edit, trial_optics = camera_residual(*args, object_delta=np.full_like(rgb, -.03))
        outside = (alpha == 0) | ~writable
        np.testing.assert_array_equal(base[outside], trial[outside])
        np.testing.assert_array_equal(trial_edit, edited)
        self.assertEqual(optics, trial_optics)
        self.assertTrue(np.any(base[~outside] != trial[~outside]))


if __name__=='__main__':unittest.main(verbosity=2)
