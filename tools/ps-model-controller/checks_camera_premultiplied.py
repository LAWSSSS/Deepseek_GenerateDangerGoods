"""Opaque compositing, camera-edge and legacy replay regressions; CPU only."""
import unittest
import cv2
import numpy as np

from display_batch.local import camera_residual
from display_batch.photography import linear


class PremultipliedCameraTests(unittest.TestCase):
    def fixture(self):
        alpha=np.zeros((120,160),np.float32);alpha[18:102,25:135]=1.
        rgb=np.full((120,160,3),.36,np.float32)
        background=np.full_like(rgb,.16)
        environment={'camera_psf':{'sigma_px':.95},'sensor':{'noise_sigma':0.}}
        writable=np.ones_like(alpha,bool)
        return rgb,alpha,background,environment,writable

    def test_opaque_object_is_independent_of_hidden_background_texture(self):
        rgb,alpha,background,env,writable=self.fixture()
        yy,xx=np.indices(alpha.shape)
        other=np.repeat(np.where((xx+yy)%2,.30,.02)[...,None],3,axis=2).astype(np.float32)
        outputs={}
        for mode in ('legacy','premultiplied'):
            a=camera_residual(background,rgb,alpha,background,env,{'camera_pipeline':mode},42,writable)[0]
            b=camera_residual(other,rgb,alpha,other,env,{'camera_pipeline':mode},42,writable)[0]
            outputs[mode]=float(np.max(abs(a[35:85,45:115]-b[35:85,45:115])))
        self.assertGreater(outputs['legacy'],.1)
        self.assertLess(outputs['premultiplied'],1e-6)

    def test_all_rgb_effects_use_one_premultiplied_psf_including_outer_edge(self):
        rgb,alpha,background,env,writable=self.fixture()
        rgb[:]=.12
        delta=np.full_like(rgb,.08)
        settings={'camera_pipeline':'premultiplied'}
        result,_,stats=camera_residual(background,rgb,alpha,background,env,settings,42,writable,object_delta=delta)
        sigma=stats['blur_sigma'];self.assertGreater(sigma,.2)
        coverage=cv2.GaussianBlur(alpha,(0,0),sigma,borderType=cv2.BORDER_CONSTANT)
        p=cv2.GaussianBlur((linear(rgb)+delta)*alpha[...,None],(0,0),sigma,borderType=cv2.BORDER_CONSTANT)
        expected=background*(1-coverage[...,None])+p
        np.testing.assert_allclose(linear(result),expected,rtol=0,atol=2e-7)
        outer=(alpha==0)&(coverage>.01)
        self.assertTrue(outer.any())

    def test_hole_centers_foreground_and_protected_pixels_stay_background(self):
        rgb,alpha,background,env,writable=self.fixture()
        alpha[45:75,65:95]=0
        original=alpha.copy();writable[:,105:112]=False
        settings={'camera_pipeline':'premultiplied'}
        result,edited,stats=camera_residual(background*.9,rgb,alpha,background,env,settings,42,writable,
                                          object_delta=np.full_like(rgb,.07))
        np.testing.assert_array_equal(alpha,original)
        np.testing.assert_allclose(linear(result[~writable]),background[~writable],atol=1e-7)
        self.assertFalse(edited[~writable].any())
        # A transparent opening reveals the same shadowed background. Camera
        # convolution may soften its rim, but cannot fill its clear center.
        np.testing.assert_allclose(linear(result[54:66,74:86]),background[54:66,74:86]*.9,atol=1e-7)
        self.assertTrue(stats['geometric_alpha_unchanged'])

    def test_absent_option_replays_explicit_legacy_and_noise_is_deterministic(self):
        rgb,alpha,background,env,writable=self.fixture();env['sensor']['noise_sigma']=.008
        args=(background,rgb,alpha,background,env)
        implicit=camera_residual(*args,{},8,writable)
        explicit=camera_residual(*args,{'camera_pipeline':'legacy'},8,writable)
        np.testing.assert_array_equal(implicit[0],explicit[0])
        np.testing.assert_array_equal(implicit[1],explicit[1])
        first=camera_residual(*args,{'camera_pipeline':'premultiplied'},8,writable)
        second=camera_residual(*args,{'camera_pipeline':'premultiplied'},8,writable)
        np.testing.assert_array_equal(first[0],second[0])

    def test_hole_center_preserves_existing_nonuniform_shadow_image(self):
        rgb,alpha,background,env,writable=self.fixture()
        alpha[35:85,55:105]=0
        yy,xx=np.indices(alpha.shape)
        shaded=background*np.where((xx+yy)%2,.68,1.)[...,None]
        old=camera_residual(shaded,rgb,alpha,background,env,{},3,writable)[0]
        new=camera_residual(shaded,rgb,alpha,background,env,{'camera_pipeline':'premultiplied'},3,writable)[0]
        np.testing.assert_allclose(new[44:76,64:96],old[44:76,64:96],atol=1e-7)


if __name__=='__main__':unittest.main(verbosity=2)
