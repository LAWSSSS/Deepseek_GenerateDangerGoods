"""Regressions for scene-only veil and uncertainty-bounded broad illumination."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from display_batch import scene_atmosphere, realism_light
from display_batch.atmosphere import shadowed_background
from display_batch.geometry import camera_for
from display_batch.local import neighbourhood


def srgb(x):
    return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-.055)


def fixture(fog=.25,kind='haze',size=(384,256)):
    w,h=size
    y,x=np.mgrid[:h,:w]
    # Repeated dark to medium neutral steel features have a known synthetic veil.
    value=.008+.40*(.5+.5*np.sin(x/8)*np.cos(y/11))**2
    clear=np.repeat(value[...,None],3,axis=2)
    air=np.float32([.50,.48,.46])
    observed=clear*(1-fog)+air*fog
    scene={'key':'synthetic','size':[w,h],'frame':Image.fromarray(np.uint8(np.rint(srgb(observed)*255))),
        'depth_native':np.full((h,w),.5,np.float32),'allowed':np.ones((h,w),bool),
        'protected':np.zeros((h,w),bool),
        'supports':[{'id':'S1','depth':.5,'mask':np.ones((h,w),bool)}],
        'atmosphere':{'summary':{'visual_evidence':{'kind':kind,'confidence':.8,
            'haze_regions':[[0,0,1,1]] if kind!='clear' else [],'exclude_regions':[]}}}}
    return scene


class SceneAtmosphereTests(unittest.TestCase):
    def test_clear_scene_is_not_forced_to_have_haze(self):
        fields,summary=scene_atmosphere.estimate(fixture(0.,'clear'))
        np.testing.assert_array_equal(fields['transmission'],1.)
        self.assertEqual(summary['depth_coefficient'],0.)

    def test_raised_dark_structure_gives_more_haze_than_clear(self):
        clear,_=scene_atmosphere.estimate(fixture(0.,'haze'))
        veiled,summary=scene_atmosphere.estimate(fixture(.30,'haze'))
        self.assertGreater(float(np.median(clear['transmission'])-np.median(veiled['transmission'])),.12)
        self.assertTrue(summary['material_and_candidate_independent'])
        self.assertFalse(summary['metric_air_density_known'])
        self.assertGreaterEqual(float(veiled['transmission'].min()),.5)

    def test_uniform_white_surface_is_not_a_fog_measurement(self):
        scene=fixture(.3)
        scene['frame']=Image.new('RGB',tuple(scene['size']),(205,205,205))
        fields,summary=scene_atmosphere.estimate(scene)
        self.assertEqual(summary['supported_tiles'],0)
        np.testing.assert_array_equal(fields['transmission'],1.)
        self.assertEqual(summary['confidence'],'unknown')

    def test_protected_regions_cannot_bias_neighbours(self):
        scene=fixture(.2)
        scene['protected'][70:110,100:150]=True
        before,_=scene_atmosphere.estimate(scene)
        array=np.asarray(scene['frame']).copy();array[scene['protected']]=[255,0,255]
        scene['frame']=Image.fromarray(array)
        after,_=scene_atmosphere.estimate(scene)
        np.testing.assert_array_equal(before['transmission'],after['transmission'])
        np.testing.assert_array_equal(before['airlight'],after['airlight'])

    def test_different_depth_bright_patch_cannot_set_local_veil(self):
        scene=fixture(.2,size=(512,256))
        scene['depth_native'][:,300:]=.9
        before,_=scene_atmosphere.estimate(scene)
        array=np.asarray(scene['frame']).copy();array[:,300:]=[215,230,245]
        scene['frame']=Image.fromarray(array)
        after,_=scene_atmosphere.estimate(scene)
        np.testing.assert_array_equal(before['transmission'][:,180:230],after['transmission'][:,180:230])

    def test_cache_depends_on_background_not_object(self):
        scene=fixture(.3)
        with tempfile.TemporaryDirectory() as cache:
            first=scene_atmosphere.prepare(scene,cache)
            other=copy.deepcopy(scene);other['object_id']='red-cylinder';other['target_px']=500
            second=scene_atmosphere.prepare(other,cache)
            self.assertEqual(first['signature'],second['signature'])
            self.assertTrue(second['cache_hit'])
            altered=np.asarray(other['frame']).copy();altered[60:100,50:90]=0;other['frame']=Image.fromarray(altered)
            third=scene_atmosphere.prepare(other,cache)
            self.assertNotEqual(first['signature'],third['signature'])

    def test_same_pixel_different_objects_and_crops_receive_identical_haze(self):
        scene=fixture(.3)
        with tempfile.TemporaryDirectory() as cache:
            context=scene_atmosphere.prepare(scene,cache)
            small=[90,70,150,120];large=[40,30,200,160]
            first=scene_atmosphere.fields_for(context,small,np.ones((50,60)),{},
                {'site_id':'nonexistent','material':'red'},scene)
            second=scene_atmosphere.fields_for(context,large,np.ones((130,160))*10,{},
                {'site_id':'different','material':'chrome'},scene)
            np.testing.assert_array_equal(first[0],second[0][40:90,50:110])
            np.testing.assert_array_equal(first[2],second[2][40:90,50:110])
            self.assertIn('same_pixel',first[3]['depth_basis'])

    def test_off_leaves_both_background_and_object_transmission_one(self):
        scene=fixture(.3)
        with tempfile.TemporaryDirectory() as cache:
            context=scene_atmosphere.prepare(scene,cache)
            first=scene_atmosphere.fields_for(context,[0,0,30,30],np.zeros((30,30)),{},None,scene,enabled=False)
            np.testing.assert_array_equal(first[0],1.)
            np.testing.assert_array_equal(first[1],1.)

    def test_gain_does_not_mutate_cached_background_transmission(self):
        scene=fixture(.25)
        with tempfile.TemporaryDirectory() as cache:
            context=scene_atmosphere.prepare(scene,cache)
            a=scene_atmosphere.fields_for(context,[0,0,30,30],np.zeros((30,30)),{},None,scene,.7)
            b=scene_atmosphere.fields_for(context,[0,0,30,30],np.zeros((30,30)),{},None,scene,1.3)
            np.testing.assert_array_equal(a[1],b[1])
            self.assertGreater(float(a[0].mean()),float(b[0].mean()))

    def test_verified_depth_trend_attenuates_farther_object_without_refitting(self):
        scene=fixture(.25)
        with tempfile.TemporaryDirectory() as cache:
            context=scene_atmosphere.prepare(scene,cache)
            context=copy.deepcopy(context)
            context['summary'].update(depth_coefficient=.20,depth_near=1.,depth_far=0.)
            order={'confidence':'estimated','scale':.5,'offset':0.}
            near=scene_atmosphere.fields_for(context,[80,60,120,90],np.full((30,40),.5),order,None,scene)
            far=scene_atmosphere.fields_for(context,[80,60,120,90],np.full((30,40),2.),order,None,scene)
            self.assertGreater(float(near[0].mean()),float(far[0].mean()))
            np.testing.assert_array_equal(near[1],far[1])
            np.testing.assert_array_equal(near[2],far[2])
            order['confidence']='unknown'
            unknown_near=scene_atmosphere.fields_for(context,[80,60,120,90],np.full((30,40),.5),order,None,scene)
            unknown_far=scene_atmosphere.fields_for(context,[80,60,120,90],np.full((30,40),2.),order,None,scene)
            np.testing.assert_array_equal(unknown_near[0],unknown_far[0])

    def test_shadow_never_attenuates_airlight(self):
        air=np.ones((12,12,3),np.float32)*.5;t=np.ones((12,12),np.float32)*.7
        veil=air*(1-t[...,None]);background=veil+.2*t[...,None]
        black=shadowed_background(background,np.ones((12,12),np.float32),t,air)
        np.testing.assert_allclose(black,veil,atol=1e-7)

    def test_output_has_finite_bounded_fields_and_uncertainty(self):
        fields,summary=scene_atmosphere.estimate(fixture(.3,size=(1500,800)))
        self.assertEqual(max(summary['field_size']),512)
        for value in fields.values():self.assertTrue(np.isfinite(value).all())
        self.assertTrue(np.all(fields['amount_upper']>=fields['amount_lower']))
        self.assertIn('not_statistical',summary['uncertainty_basis'])


class RealismLightTests(unittest.TestCase):
    def fixture(self):
        scene=fixture(.10)
        candidate={'bounds':[145,90,235,170],'target_px':90,'site_id':'S1','support_ids':['S1'],
            'anchor_px':[190,150], 'camera':camera_for(np.array([[-.5,-.4,0],[.5,.4,.6]]),45)}
        environment,_=neighbourhood(scene,candidate)
        return scene,candidate,environment

    def test_light_preserves_asset_and_camera_parameters(self):
        scene,candidate,environment=self.fixture();prior=copy.deepcopy(environment)
        result,mask=realism_light.estimate_local_light(scene,candidate,environment)
        self.assertEqual(environment,prior)
        for key in ('dust_color','airborne_dust','haze','sensor','camera_psf'):
            self.assertEqual(result[key],prior[key])
        self.assertFalse(mask[scene['protected']].any())
        self.assertFalse(result['local_irradiance']['metric_illumination_known'])
        self.assertEqual(result['local_irradiance']['reflectance_range'],[.15,.45])

    def test_no_neutral_evidence_uses_finite_unknown_soft_prior(self):
        scene,candidate,environment=self.fixture();scene['allowed'][:]=False
        result,mask=realism_light.estimate_local_light(scene,candidate,environment)
        self.assertEqual(result['local_irradiance']['confidence'],'unknown')
        self.assertFalse(mask.any())
        self.assertTrue(np.isfinite(result['reflection_probe']).all())
        sky,lights=realism_light.light_rig(result,candidate['camera'])
        self.assertTrue(all(size>=4.8 for _,_,size in lights))
        self.assertTrue(np.isfinite(sky).all())

    def test_virtual_receiver_correction_is_bounded_and_object_independent(self):
        scene,candidate,environment=self.fixture()
        environment,_=realism_light.estimate_local_light(scene,candidate,environment)
        raw=np.full((20,30,4),.025,np.float32)
        with patch('cv2.imread',return_value=raw):
            first=realism_light.receiver_calibration('asset1',{'part_paths':{'without_shadow':'unused'}},environment)
            other=realism_light.receiver_calibration('asset2',{'part_paths':{'without_shadow':'unused'}},environment)
        self.assertEqual(first,other)
        self.assertLessEqual(abs(first['exposure_ev']),.5)
        self.assertIn('target_luma_interval',first)
        self.assertIn('requires_lighting_revision',first)

    def test_existing_fit_inside_interval_is_not_forced_to_exact_median(self):
        environment={'local_irradiance':{'target_clear_rgb_linear':[.18]*3,'reflectance_range':[.15,.45]}}
        # Search a diagnostic receiver value that lies inside the admissible camera range.
        raw=np.full((20,30,4),.18,np.float32)
        with patch('cv2.imread',return_value=raw):
            result=realism_light.receiver_calibration('',{'part_paths':{'without_shadow':'unused'}},environment)
        self.assertEqual(result['exposure_ev'],0.)
        self.assertFalse(result['limited'])


if __name__=='__main__':
    unittest.main()
