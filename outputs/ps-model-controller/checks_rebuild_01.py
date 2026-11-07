"""Integration checks against a generated 01 review candidate."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from mesh_rebuild_item import mask_fit
from scene_v2 import digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/01阀门-六视图重建-v6'


class RebuildChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('01 review candidate has not been generated')
        cls.record=json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.scene=trimesh.load(OUT/'mesh.glb',force='scene')

    def test_manifest_and_round_trip_preserve_identity_and_bounds(self):
        self.assertEqual(self.record['mesh_sha256'],digest(OUT/'mesh.glb'))
        self.assertEqual(self.record['shape_basis'],'user_six_view_reference')
        self.assertEqual(self.record['model_review'],'pending')
        self.assertEqual(self.record['scene_validation'],'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertTrue(np.allclose(self.scene.bounds,self.record['worker']['bounds'],atol=2e-6))
        self.assertEqual(self.record['mesh_audit']['open_parts'],['Continuous curved painted underline'])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertFalse(self.record['worker']['self_intersecting_components'])

    def test_actual_openings_and_component_identity(self):
        names=set(self.scene.graph.nodes_geometry)
        self.assertIn('Four-spoke wheel rim',names)
        self.assertEqual(sum(name.startswith('Handwheel spoke ') for name in names),4)
        self.assertTrue(self.record['isolated_opening_checks']['wheel']['passed'])
        self.assertTrue(self.record['isolated_opening_checks']['bottom_flange']['passed'])

    def test_pbr_textures_are_embedded(self):
        mesh=next(iter(self.scene.geometry.values()))
        material=mesh.visual.material
        for name in ('baseColorTexture','metallicRoughnessTexture','normalTexture'):
            self.assertEqual(getattr(material,name).size,(2048,2048))
        self.assertTrue((OUT/'model.blend').is_file())

    def test_six_views_turntable_and_changed_light(self):
        for view in ('front','back','left','right','top','bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
        self.assertEqual(len(list((OUT/'renders').glob('turn-*.png'))),24)
        first=np.asarray(Image.open(OUT/'renders/new-front.png').convert('RGBA'))
        second=np.asarray(Image.open(OUT/'renders/new-light-alternate.png').convert('RGBA'))
        a,b=first[:,:,3]>=128,second[:,:,3]>=128
        self.assertGreater(np.logical_and(a,b).sum()/np.logical_or(a,b).sum(),.999)
        self.assertGreater(np.abs(first[:,:,:3].astype(float)-second[:,:,:3].astype(float))[a].mean(),.5)
        self.assertLess(np.mean(first[:,:,:3].mean(axis=2)[a]<18),.8)

    def test_reference_alignment_does_not_distort_aspect_ratio(self):
        mask=np.zeros((80,100),np.uint8);mask[10:70,40:60]=1
        fitted=mask_fit(mask,size=400); y,x=np.nonzero(fitted)
        self.assertAlmostEqual((x.max()-x.min()+1)/(y.max()-y.min()+1),1/3,delta=.01)


if __name__=='__main__':
    unittest.main(verbosity=2)
