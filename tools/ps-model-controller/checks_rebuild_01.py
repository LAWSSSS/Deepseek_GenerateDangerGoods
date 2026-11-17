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
OUT=ROOT/'outputs/01阀门-六视图重建-v7'
PREVIOUS=ROOT/'outputs/01阀门-六视图重建-v6'


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

    def test_material_revision_preserves_previous_geometry(self):
        old=trimesh.load(PREVIOUS/'mesh.glb',force='scene')
        self.assertEqual(set(self.scene.graph.nodes_geometry),set(old.graph.nodes_geometry))
        def canonical_faces(faces):
            # glTF may reorder triangles between exports. Retain winding while
            # comparing the set of triangles rather than their sequence.
            faces=np.asarray(faces)
            shift=np.argmin(faces,axis=1)
            faces=np.take_along_axis(faces,(np.arange(3)[None,:]+shift[:,None])%3,axis=1)
            return faces[np.lexsort(faces.T[::-1])]
        for node in old.graph.nodes_geometry:
            before_transform,before_name=old.graph[node]
            after_transform,after_name=self.scene.graph[node]
            before,after=old.geometry[before_name],self.scene.geometry[after_name]
            self.assertTrue(np.allclose(before_transform,after_transform,atol=1e-7),node)
            self.assertTrue(np.array_equal(canonical_faces(before.faces),canonical_faces(after.faces)),node)
            self.assertTrue(np.allclose(before.vertices,after.vertices,atol=1e-7),node)

    def test_metal_and_paint_use_distinct_pbr_values(self):
        before=np.asarray(Image.open(PREVIOUS/'metallic_roughness.png').convert('RGB'),dtype=float)/255
        after=np.asarray(Image.open(OUT/'metallic_roughness.png').convert('RGB'),dtype=float)/255
        cast=np.s_[80:940,80:940];machined=np.s_[80:940,1100:1960];paint=np.s_[1100:1960,80:940]
        self.assertGreater(after[cast][...,2].mean(),before[cast][...,2].mean()+.2)
        self.assertLess(after[cast][...,1].mean(),before[cast][...,1].mean()-.2)
        self.assertLess(after[machined][...,1].mean(),before[machined][...,1].mean()-.1)
        self.assertLess(after[paint][...,2].mean(),.1)

    def test_same_light_material_comparison_keeps_silhouette(self):
        before=np.asarray(Image.open(OUT/'renders/previous-front.png').convert('RGBA'))
        after=np.asarray(Image.open(OUT/'renders/new-front.png').convert('RGBA'))
        a,b=before[:,:,3]>=128,after[:,:,3]>=128
        self.assertGreater(np.logical_and(a,b).sum()/np.logical_or(a,b).sum(),.995)
        self.assertGreater(np.abs(before[:,:,:3].astype(float)-after[:,:,:3].astype(float))[a&b].mean(),1)

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
