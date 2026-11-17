"""Offline local-veil regression; no models, network calls or rendering."""
import unittest
import tempfile

import numpy as np
from PIL import Image

from display_batch.local_atmosphere import constrain_dark_floor, local_veil_evidence, local_airlight, estimate, prepare
from display_batch.atmosphere import fogged_radiance


class LocalVeil(unittest.TestCase):
    def test_structured_raised_dark_floor_is_not_a_white_surface(self):
        yy,xx=np.mgrid[:96,:96]
        # Low-contrast steel texture under a raised camera black level.
        values=.20+.075*(.5+.5*np.sin(xx/3)*np.cos(yy/4))
        radiance=np.repeat(values[...,None],3,axis=2).astype(np.float32)
        valid=np.ones(values.shape,bool)
        original=radiance.copy();saved=valid.copy()
        fog,info=constrain_dark_floor(np.full(values.shape,.48),radiance,valid,
            np.array([.34]*3),{'kind':'veil','confidence':.7},valid)
        self.assertGreater(float(fog.mean()),.4)
        self.assertFalse(info['bright_surface_ambiguous'])
        self.assertGreaterEqual(info['local_veil_evidence']['structured_tiles'],3)
        self.assertTrue(np.array_equal(radiance,original))
        self.assertTrue(np.array_equal(valid,saved))
        clear=np.full((96,96,3),[.08,.012,.008],np.float32)
        old=fogged_radiance(clear,np.full((96,96),.9),np.full_like(clear,.34))
        new=fogged_radiance(clear,1-fog,np.full_like(clear,.34))
        self.assertGreater(float(new.mean()),float(old.mean()))
        self.assertLess(float(np.ptp(new,axis=2).mean()),float(np.ptp(old,axis=2).mean()))

    def test_a_flat_white_plate_remains_ambiguous_despite_a_haze_hint(self):
        values=np.full((96,96,3),.4,np.float32)
        fog,info=constrain_dark_floor(np.full((96,96),.48),values,
            np.ones((96,96),bool),np.array([.5]*3),{'kind':'veil','confidence':.9},np.ones((96,96),bool))
        self.assertLessEqual(float(fog.max()),.1)
        self.assertEqual(info['confidence'],'unknown')

    def test_one_textured_patch_cannot_unlock_the_whole_neighbourhood(self):
        values=np.full((96,96,3),.24,np.float32)
        yy,xx=np.mgrid[:24,:24]
        values[:24,:24]=(.20+.075*(.5+.5*np.sin(xx/3)*np.cos(yy/4)))[...,None]
        info=local_veil_evidence(values,np.ones((96,96),bool),{'kind':'veil','confidence':.9},np.ones((96,96),bool))
        self.assertFalse(info['supports_veil'])

    def test_clear_evidence_wins_over_a_raised_dark_floor(self):
        values=np.repeat(np.linspace(.17,.35,96,dtype=np.float32)[None,:,None],96,axis=0)
        values=np.repeat(values,3,axis=2)
        fog,_=constrain_dark_floor(np.full((96,96),.48),values,
            np.ones((96,96),bool),np.array([.5]*3),{'kind':'clear','confidence':.9})
        self.assertEqual(float(fog.max()),0.)

    def test_unconfirmed_hint_cannot_unlock_a_bright_ambiguous_surface(self):
        yy,xx=np.mgrid[:96,:96]
        values=np.repeat((.20+.075*(.5+.5*np.sin(xx/3)*np.cos(yy/4)))[...,None],3,axis=2)
        fog,info=constrain_dark_floor(np.full((96,96),.48),values,
            np.ones((96,96),bool),np.array([.34]*3),{'kind':'uncertain','confidence':.2})
        self.assertLessEqual(float(fog.max()),.1)
        self.assertEqual(info['confidence'],'unknown')

    def test_samples_outside_the_local_valid_mask_do_not_affect_the_result(self):
        yy,xx=np.mgrid[:96,:96]
        values=np.repeat((.20+.075*(.5+.5*np.sin(xx/3)*np.cos(yy/4)))[...,None],3,axis=2)
        valid=np.ones((96,96),bool);valid[:24]=False
        changed=values.copy();changed[~valid]=1.
        args=(np.full((96,96),.48),valid,np.array([.34]*3),{'kind':'veil','confidence':.7},valid)
        a,ia=constrain_dark_floor(args[0],values,*args[1:])
        b,ib=constrain_dark_floor(args[0],changed,*args[1:])
        self.assertTrue(np.array_equal(a,b));self.assertEqual(ia,ib)

    def test_airlight_is_local_and_does_not_use_one_bright_reflection(self):
        pixels=np.full((256,256,3),125,np.uint8)
        base={'frame':Image.fromarray(pixels),'allowed':np.ones((256,256),bool),
              'protected':np.zeros((256,256),bool),'depth_native':np.full((256,256),.5,np.float32)}
        args=([110,110,146,146],{'depth':.5},{},np.array([.1]*3))
        air,info=local_airlight(base,*args)
        # The 2x envelope sees neighbouring steel, not only the dark contact patch.
        self.assertGreater(float(air.mean()),.1)
        self.assertGreaterEqual(info['independent_tiles'],3)
        changed=pixels.copy();changed[:60]=[0,0,255];changed[92:109,92:109]=220
        after,_=local_airlight({**base,'frame':Image.fromarray(changed)},*args)
        self.assertTrue(np.allclose(air,after,atol=.004))

    @staticmethod
    def textured_scene(regions):
        yy,xx=np.mgrid[:256,:256]
        values=.20+.075*(.5+.5*np.sin(xx/3)*np.cos(yy/4))
        rgb=np.uint8(np.rint((1.055*values**(1/2.4)-.055)*255))
        pixels=np.repeat(rgb[...,None],3,axis=2)
        return {'frame':Image.fromarray(pixels),'allowed':np.ones((256,256),bool),
            'protected':np.zeros((256,256),bool),'depth_native':np.full((256,256),.5,np.float32),
            'supports':[{'id':'site','depth':.5}], 'atmosphere':{'summary':{'visual_evidence':
                {'kind':'veil','confidence':.9,'haze_regions':regions,'exclude_regions':[]}}}}

    def test_haze_elsewhere_does_not_unlock_textured_white_target(self):
        environment={'local_context':{'bounds':[16,16,112,112]},'integration_mode':'coherent'}
        fields,summary=estimate(self.textured_scene([[.7,.7,1,1]]),{'site_id':'site'},environment)
        self.assertLessEqual(float((1-fields['transmission']).max()),.100001)
        local=summary['dark_floor_constraint']['local_veil_evidence']
        self.assertEqual(local['local_haze_overlap'],0.)
        self.assertFalse(local['supports_veil'])
        fields,summary=estimate(self.textured_scene([[0,0,.5,.5]]),{'site_id':'site'},environment)
        self.assertGreater(float((1-fields['transmission']).mean()),.4)
        self.assertEqual(summary['dark_floor_constraint']['local_veil_evidence']['local_haze_overlap'],1.)

    def test_missing_regions_or_tiny_overlap_do_not_unlock_textured_white_target(self):
        environment={'local_context':{'bounds':[16,16,112,112]},'integration_mode':'coherent'}
        for regions in ([],[[.4,.4,1,1]]):
            fields,summary=estimate(self.textured_scene(regions),{'site_id':'site'},environment)
            self.assertLessEqual(float((1-fields['transmission']).max()),.100001)
            self.assertFalse(summary['dark_floor_constraint']['local_veil_evidence']['supports_veil'])

    def test_overlap_is_measured_on_valid_pixels_not_the_bounding_box(self):
        scene=self.textured_scene([[0,0,.25,.5]])
        # The apparent left-half overlap is outside the allowed load surface.
        scene['allowed'][:,:65]=False
        _,summary=estimate(scene,{'site_id':'site'},
            {'local_context':{'bounds':[16,16,112,112]},'integration_mode':'coherent'})
        self.assertEqual(summary['dark_floor_constraint']['local_veil_evidence']['local_haze_overlap'],0.)

    def test_updated_visual_regions_invalidate_local_fields_even_with_same_upstream_id(self):
        environment={'local_context':{'bounds':[16,16,112,112]},'integration_mode':'coherent'}
        records=[]
        with tempfile.TemporaryDirectory() as cache:
            for regions in ([[.7,.7,1,1]],[[0,0,.5,.5]]):
                scene=self.textured_scene(regions);scene['key']='same-background'
                scene['atmosphere']['signature']='same-provider-scene-id'
                records.append(prepare(scene,{'site_id':'site'},environment,cache))
        self.assertNotEqual(records[0]['signature'],records[1]['signature'])
        self.assertGreater(records[0]['summary']['transmission_range'][1],records[1]['summary']['transmission_range'][1])


if __name__=='__main__':unittest.main()
