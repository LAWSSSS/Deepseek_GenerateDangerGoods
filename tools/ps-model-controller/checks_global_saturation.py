"""Numerical guards for a global colour experiment, not evidence of visual fusion."""
import unittest
import cv2
import numpy as np
from display_batch.global_saturation import apply, background_statistics, colour_mask, measure, hsv
from display_batch.schema import BatchConfig


class GlobalSaturationTests(unittest.TestCase):
    def test_literal_full_frame_mean_includes_remote_region(self):
        image=np.full((20,30,3),128,np.uint8);image[:10]=[255,0,0]
        stats=background_statistics(image)
        self.assertAlmostEqual(stats['mean_s'],.5,places=5)
        self.assertEqual(stats['pixels'],600)

    def test_red_yellow_blue_keep_lab_lightness_and_chroma_direction(self):
        for color in ([.65,.08,.03],[.75,.60,.03],[.04,.16,.55]):
            rgb=np.full((32,32,3),color,np.float32);alpha=np.ones((32,32),np.float32)
            stats=background_statistics(np.full((40,40,3),[128,120,116],np.uint8))
            out,report,mask=apply(rgb,alpha,stats)
            self.assertTrue(report['applied']);self.assertEqual(report['after']['status'],'pass')
            before=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB);after=cv2.cvtColor(out,cv2.COLOR_RGB2LAB)
            # OpenCV's RGB->Lab LUT has small round-trip errors. At very low
            # chroma an angular error exaggerates them; check Lab components.
            self.assertLess(float(np.abs(before[...,0]-after[...,0]).max()),.25)
            self.assertLess(float(np.abs(after[...,1:]-before[...,1:]*report['gain']).max()),.25)
            self.assertLess(report['after']['mean_s'],report['before']['mean_s'])

    def test_holes_foreground_and_alpha_stay_identical(self):
        rgb=np.full((24,24,3),[.8,.1,.1],np.float32)
        alpha=np.ones((24,24),np.float32);alpha[8:16,8:16]=0;alpha[:3]=0;alpha[-1]=.4
        original_alpha=alpha.copy();stats=background_statistics(np.full((16,16,3),128,np.uint8))
        out,report,_=apply(rgb,alpha,stats)
        np.testing.assert_array_equal(alpha,original_alpha)
        np.testing.assert_array_equal(out[alpha==0],rgb[alpha==0])
        self.assertEqual(report['after']['status'],'pass')

    def test_neutral_labels_cannot_dilute_coloured_region(self):
        rgb=np.ones((50,50,3),np.float32)*.85;rgb[10:20,10:20]=[.8,.1,.1]
        alpha=np.ones((50,50),np.float32)
        stats=background_statistics(np.full((10,10,3),[140,130,125],np.uint8))
        out,report,mask=apply(rgb,alpha,stats)
        self.assertEqual(int(mask.sum()),100)
        np.testing.assert_array_equal(out[~mask],rgb[~mask])
        self.assertEqual(report['after']['status'],'pass')

    def test_lower_saturation_is_never_increased(self):
        rgb=np.full((20,20,3),[.5,.47,.45],np.float32)
        stats=background_statistics(np.full((10,10,3),[255,0,0],np.uint8))
        out,report,_=apply(rgb,np.ones((20,20),np.float32),stats)
        np.testing.assert_array_equal(out,rgb);self.assertEqual(report['gain'],1.)

    def test_neutral_or_tiny_region_remains_unassessable(self):
        rgb=np.full((20,20,3),.5,np.float32);alpha=np.ones((20,20),np.float32)
        stats=background_statistics(np.full((10,10,3),128,np.uint8))
        out,report,_=apply(rgb,alpha,stats)
        np.testing.assert_array_equal(out,rgb)
        self.assertEqual(report['after']['status'],'unknown')

    def test_frozen_mask_detects_colour_loss_without_changing_denominator(self):
        rgb=np.full((20,20,3),[.7,.06,.06],np.float32)
        stats=background_statistics(np.full((20,20,3),128,np.uint8))
        out,report,mask=apply(rgb,np.ones((20,20),np.float32),stats)
        self.assertEqual(measure(out,mask,stats)['pixels'],400)
        self.assertEqual(report['after']['pixels'],report['before']['pixels'])

    def test_old_default_and_independent_api_option(self):
        config={'backgrounds':'example','output':'example'}
        self.assertEqual(BatchConfig(**config).chroma_mode,'legacy')
        self.assertEqual(BatchConfig(**config,chroma_mode='global_mean').chroma_mode,'global_mean')

    def test_dark_saturated_paint_is_not_exempted_by_low_lab_chroma(self):
        rgb=np.full((20,20,3),[.12,.095,.09],np.float32)
        stats=background_statistics(np.full((20,20,3),[128,120,116],np.uint8))
        out,report,mask=apply(rgb,np.ones((20,20),np.float32),stats)
        self.assertEqual(int(mask.sum()),400)
        self.assertTrue(report['applied']);self.assertEqual(report['after']['status'],'pass')


if __name__=='__main__':unittest.main(verbosity=2)
