"""Whole-frame mean control and the requested fine optical ladder."""
import copy
import unittest

import numpy as np

import checks_display_local
from display_batch.global_light import estimate_global_mean
from display_batch.local import foreground_dust, local_veil, pool_object_rgb
from display_batch.photography import linear
from validate_display_local_light import FINE_FOG, FINE_POOL


class FullMeanTests(unittest.TestCase):
    def environment(self):
        return {'local_irradiance': {'bounds': [1, 1, 3, 3],
                    'direction': {'direction': [1., 0.], 'confidence': .5}, 'fill_ratio': .6},
                'airborne_dust': {'amount': .2}, 'dust_color': [.3]*3,
                'sensor': {'noise': .02}, 'camera_psf': {'sigma': .5},
                'reflection_probe': [[[.1]*3]]}

    def test_full_mean_counts_every_pixel_including_remote_dark_and_saturated(self):
        pixels=np.zeros((4, 4, 3), np.uint8)
        pixels[1:3, 1:3]=80
        pixels[0, 0]=[255, 255, 255]
        pixels[3, 3]=[255, 0, 0]
        scene={'frame': pixels, 'size': [4, 4]}
        env,mask=estimate_global_mean(scene,self.environment())
        expected=linear(pixels.astype(np.float32)/255).reshape(-1,3).sum(axis=0,dtype=np.float64)/16
        np.testing.assert_allclose(env['local_irradiance']['observed_mean_rgb_linear'],expected,atol=1e-12)
        self.assertTrue(mask.all())
        self.assertEqual(env['local_irradiance']['sample_count'],16)
        changed=pixels.copy();changed[0,0]=0
        other,_=estimate_global_mean({**scene,'frame':changed},self.environment())
        self.assertNotEqual(env['world_color'],other['world_color'])

    def test_global_mean_does_not_change_air_surface_dust_camera_or_direction(self):
        before=self.environment();saved=copy.deepcopy(before)
        env,_=estimate_global_mean({'frame':np.full((4,4,3),110,np.uint8),'size':[4,4]},before)
        self.assertEqual(saved,before)
        for key in ('airborne_dust','dust_color','sensor','camera_psf'):
            self.assertEqual(env[key],before[key])
        for key in ('direction','fill_ratio'):
            self.assertEqual(env['local_irradiance'][key],before['local_irradiance'][key])
        self.assertNotIn('reflection_probe',env)
        self.assertNotIn('reconstruction',env['local_irradiance'])

    def test_global_mean_is_independent_of_local_dehaze_or_bounds(self):
        scene={'frame':np.full((4,4,3),[70,100,150],np.uint8),'size':[4,4]}
        local=self.environment();other=copy.deepcopy(local)
        other['airborne_dust']['amount']=.7
        other['local_irradiance']['bounds']=[0,0,2,2]
        a,_=estimate_global_mean(scene,local);b,_=estimate_global_mean(scene,other)
        for key in ('world_color','light_color','light_level'):
            self.assertEqual(a[key],b[key])


class FineOpticsTests(unittest.TestCase):
    def test_twelve_pool_and_five_percent_fog_steps_keep_alpha(self):
        self.assertEqual(FINE_POOL,(12,))
        self.assertEqual(FINE_FOG,(.25,.30,.35,.40,.45,.50,.55))
        rgb,alpha,env=checks_display_local.ForegroundDustTests().fixture()
        before=alpha.copy()
        veiled,air=local_veil(rgb,{**env,'haze':.24},{'photo_style':'dusty_scrap'})
        effect,info=foreground_dust(rgb,veiled,alpha,env,air,300,pool_weight=1.,pool_kernel=12,
                                    fog_weight=0.,uniform=True,fog_profile='wide')
        expected=(pool_object_rgb(linear(rgb),alpha,12)-linear(rgb))*(1-air['amount'])*(alpha>0)[...,None]
        np.testing.assert_allclose(effect,expected,atol=1e-6)
        self.assertEqual(info['pool_weight_max'],1.)
        last=None
        for amount in FINE_FOG:
            effect,info=foreground_dust(rgb,veiled,alpha,env,air,300,pool_weight=0.,
                                        fog_weight=amount,uniform=True,fog_profile='wide')
            expected=(np.float32(air['color_linear'])-linear(veiled))*amount*(alpha>0)[...,None]
            np.testing.assert_allclose(effect,expected,atol=1e-6)
            self.assertAlmostEqual(info['fog_extra_max'],amount,places=6)
            if last is not None:self.assertGreater(float(np.abs(effect).sum()),last)
            last=float(np.abs(effect).sum())
        np.testing.assert_array_equal(alpha,before)


if __name__=='__main__':unittest.main(verbosity=2)
