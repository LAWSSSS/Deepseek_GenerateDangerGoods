"""Bounded asset controls and evidence profiles; no API or rendering required."""
import copy
import math
import unittest
import numpy as np

from display_batch.asset_profiles import PROFILES, load_profile, descriptor
from display_batch.physical_material import refine_finish
from display_batch.source_detail import clean_albedo


class AssetRefinementTests(unittest.TestCase):
    def test_source_colour_cleanup_preserves_protected_pixels_and_colour_family(self):
        y,x=np.mgrid[:128,:128]
        rgb=np.stack([120+20*np.sin(x/3),80+10*np.sin(x/3),40+5*np.sin(x/3)],axis=2).astype('uint8')
        mask=np.ones((128,128),bool);mask[30:70,40:80]=False
        out,stats=clean_albedo(rgb,mask)
        np.testing.assert_array_equal(out[~mask],rgb[~mask])
        self.assertTrue(stats['protected_texels_equal'])
        self.assertTrue(np.all(out[...,0]>out[...,1]))
        self.assertGreater(stats['changed_texels'],0)
    def setUp(self):
        self.finish={'signature':'source','recipe':{'relief_depth_m':.00008},'regions':{'materials':{}}}

    def test_absolute_controls_do_not_compound_on_resume(self):
        first=refine_finish(self.finish,{'oxide_gain':1.5})
        again=refine_finish(first,{'oxide_gain':1.5})
        self.assertEqual(first,again)
        changed=refine_finish(first,{'oxide_gain':.5})
        direct=refine_finish(self.finish,{'oxide_gain':.5})
        self.assertEqual(changed,direct)
        self.assertEqual(first['recipe'],self.finish['recipe'])
        self.assertNotIn('controls',self.finish)

    def test_invalid_controls_refused(self):
        for values in ({'oxide_gain':2},{'roughness_shift':-.09},{'relief_gain':math.nan},
                       {'scratch_gain':True},{'metalness':0}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                refine_finish(self.finish,values)

    def test_unknown_asset_preserved(self):
        self.assertIsNone(load_profile('0'*64))

    def test_profiles_are_hash_bound_and_not_scene_specific(self):
        self.assertEqual({p['asset_id'] for p in PROFILES.values()},{'3','13','22','31','33'})
        for digest in PROFILES:
            profile=load_profile(digest)
            self.assertEqual(profile['source_mesh_hash'],digest)
            self.assertNotIn('background',profile)
            self.assertGreater(len(profile['parts']),0)
            self.assertEqual(len({p['id'] for p in profile['parts']}),len(profile['parts']))
            before=copy.deepcopy(profile)
            descriptor(profile)['parts'].clear()
            self.assertEqual(profile,before)

    def test_machined_polymer_and_identity_regions_have_no_weathering_permission(self):
        for digest in PROFILES:
            for part in load_profile(digest)['parts']:
                if part['material'] in ('plastic','polymer','rubber','chrome','print','machined_steel','machined_metal','protected'):
                    self.assertEqual(part['allow'],[])
                    self.assertTrue(part['protect'])


if __name__=='__main__':unittest.main(verbosity=2)
