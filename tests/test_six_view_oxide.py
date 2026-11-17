"""Guard reference azimuth, excluded markings and coated-material separation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ps-model-controller'))
from prepare_six_view_oxide import bake, material_cues


class SixViewOxideTests(unittest.TestCase):
    def test_registered_front_and_lower_rust_keep_angular_and_axial_positions(self):
        image = np.full((120, 120, 3), .39, np.float32)
        image[68:82, 49:71] = (.56,.335,.14)
        view = {'angle':0,'rect':[10,10,110,110]}
        spec = {'material':'steel','mode':'cylinder','direction':'vertical','views':[view]}
        _, channels = bake(image, spec, np.array([120,120]))
        self.assertGreater(channels[335,512,0], .8)
        self.assertGreater(channels[335,512,2], .95)
        self.assertLess(channels[120,512,0], .02)
        self.assertLess(channels[335,0,2], .02)
        view['exclude'] = [[45,64,75,86]]
        _, excluded = bake(image, spec, np.array([120,120]))
        self.assertEqual(excluded[335,512,0], 0.)
        self.assertEqual(excluded[335,512,2], 0.)

    def test_circular_exclusion_keeps_the_adjacent_flange_colour(self):
        image = np.broadcast_to(np.float32([.57,.35,.14]), (100,100,3)).copy()
        spec = {'material':'steel','mode':'plane','views':[
            {'rect':[0,0,100,100],'exclude_ellipses':[[50,50,17,20]]}]}
        _, field = bake(image,spec,np.array([100,100]))
        self.assertEqual(float(field[256,256,2]),0.)
        self.assertEqual(float(field[256,256,0]),0.)
        self.assertGreater(float(field[256,387,2]),.99)
        self.assertGreater(float(field[256,387,0]),.8)

    def test_layered_rust_deepens_observed_oxide_without_marking_clear_metal(self):
        image = np.full((100,100,3), .54, np.float32)
        image[20:80,20:80] = (.55,.34,.16)
        spec = {'material':'steel','mode':'plane','views':[{'rect':[0,0,100,100]}]}
        old_colour, old = bake(image,spec,np.array([100,100]))
        colour, layered = bake(image,{**spec,'oxide_profile':'layered'},np.array([100,100]))
        self.assertLess(float(colour[256,256,0]),float(old_colour[256,256,0]))
        self.assertGreater(float(layered[256,256,0]),.85)
        self.assertEqual(float(layered[20,20,0]),0.)
        self.assertTrue(np.array_equal(layered[..., 2],old[..., 2]))

    def test_paint_pigment_is_not_oxide_and_silver_chips_stay_distinct(self):
        blue = np.asarray([[[.12,.28,.48],[.62,.63,.65],[.63,.39,.20]]],dtype=np.float32)
        oxide, chips = material_cues(blue, 'blue')
        self.assertLess(oxide[0,0], .02)
        self.assertLess(chips[0,0], .02)
        self.assertLess(oxide[0,1], .02)
        self.assertGreater(chips[0,1], .9)
        self.assertGreater(oxide[0,2], .8)
        red = np.asarray([[[.58,.15,.13],[.60,.37,.20]]],dtype=np.float32)
        rust, _ = material_cues(red, 'red')
        self.assertLess(rust[0,0], .02)
        self.assertGreater(rust[0,1], .65)


if __name__ == '__main__':
    unittest.main()
