"""Integration checks against the generated 04 SN513 split plummer-block candidate."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/04轴承座-六视图重建-v2'
LIBRARY = ROOT/'outputs/三维模型库-圆润复验/results.jsonl'


class RebuildChecks04(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('04 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.scene = trimesh.load(OUT/'mesh.glb', force='scene')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 4)

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT/'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT/'model.blend'))
        self.assertEqual(self.record['shape_basis'], 'user_six_view_reference')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'bearing04-parametric-v1')
        self.assertEqual(self.record['number'], 4)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertIn('0.9568', self.record['original_structural_warning'])
        self.assertIn('skf', ' '.join(self.record['catalog_evidence']['sources']).lower())
        self.assertEqual(self.record['browser_check']['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(self.scene.bounds, self.record['worker']['bounds'], atol=2e-6))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 25)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_real_through_openings(self):
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Base plate', 'Housing lower half', 'Housing cap', 'Cap lug minus',
                     'Cap lug plus', 'Front bearing boss upper', 'Front bearing boss lower',
                     'Back bearing boss upper', 'Back bearing boss lower',
                     'Bearing outer ring', 'Bearing inner ring', 'Bearing cage ring',
                     'Cap bolt head minus', 'Cap bolt head plus', 'Bolt grade mark minus',
                     'Bolt grade mark plus', 'Zerk boss', 'Zerk red body', 'Zerk red ring',
                     'Zerk dark bore', 'Cast label front', 'Cast label back',
                     'Cast label top', 'Mounting pad minus', 'Mounting pad plus',
                     'Seat tab front', 'Seat tab back', 'Underside central rib'):
            self.assertIn(name, names)
        structural = self.record['structural_checks']
        self.assertTrue(structural['bearing_outer_ring'])
        self.assertTrue(structural['bearing_inner_ring'])
        self.assertTrue(structural['bearing_cage_ring'])
        self.assertEqual(structural['cap_lugs'], 2)
        self.assertEqual(structural['cap_bolt_heads'], 2)
        self.assertEqual(structural['bolt_grade_marks'], 2)
        self.assertTrue(structural['grease_nipple'])
        self.assertEqual(structural['cast_labels'], 3)
        self.assertEqual(structural['mounting_pads'], 2)
        self.assertEqual(structural['bearing_boss_halves'], 4)
        openings = self.record['isolated_opening_checks']
        self.assertEqual(openings['bearing']['detected_openings'], 1)
        self.assertEqual(openings['base']['detected_openings'], 2)
        self.assertEqual(openings['housing']['detected_openings'], 1)
        self.assertTrue(all(row['passed'] for row in openings.values()))
        self.assertTrue(self.record['checks']['mesh_integrity'])
        self.assertTrue(self.record['checks']['isolated_openings'])
        self.assertTrue(self.record['checks']['structural_counts'])

    def test_pbr_textures_are_embedded_and_distinct(self):
        materials = []
        for mesh in self.scene.geometry.values():
            material = mesh.visual.material
            if material is None:
                continue
            materials.append(material)
        self.assertGreaterEqual(len(materials), 4)
        textured = [m for m in materials if getattr(m, 'baseColorTexture', None) is not None]
        self.assertTrue(textured)
        self.assertTrue(any(getattr(m, 'metallicRoughnessTexture', None) is not None
                            and m.metallicRoughnessTexture.size == (2048, 2048)
                            for m in materials))
        self.assertTrue(any(getattr(m, 'normalTexture', None) is not None
                            and m.normalTexture.size == (2048, 2048)
                            for m in materials))
        checks = self.record['material_texture_checks']
        self.assertTrue(checks['passed'])
        self.assertLess(checks['green_paint']['metallic_mean'], .10)
        self.assertGreater(checks['dark_steel']['metallic_mean'], .75)
        self.assertLess(checks['bearing_steel']['roughness_mean'],
                        checks['green_paint']['roughness_mean'])
        self.assertGreater(checks['rusted_cast']['roughness_mean'], .70)
        self.assertTrue((OUT/'model.blend').is_file())

    def test_six_views_material_and_light_changes(self):
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
        self.assertEqual(len(list((OUT/'renders').glob('turn-*.png'))), 24)
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg', '金属反射-变光对照.jpg',
                     '转台-24方向.jpg', '浏览器核验.json'):
            self.assertTrue((OUT/name).is_file(), name)
        for row in self.record['reference_views']:
            self.assertIn('未标定', row['note'])
        first = np.asarray(Image.open(OUT/'renders/new-front.png').convert('RGBA'))
        second = np.asarray(Image.open(OUT/'renders/new-light-alternate.png').convert('RGBA'))
        a, b = first[:, :, 3] >= 128, second[:, :, 3] >= 128
        self.assertGreater(np.logical_and(a, b).sum()/np.logical_or(a, b).sum(), .998)
        self.assertGreater(np.abs(first[:, :, :3].astype(float) -
                                  second[:, :, :3].astype(float))[a].mean(), .5)
        old = np.asarray(Image.open(OUT/'renders/old-front.png').convert('RGBA'))
        new = np.asarray(Image.open(OUT/'renders/new-front.png').convert('RGBA'))
        self.assertGreater(np.abs(old[:, :, :3].astype(float) -
                                  new[:, :, :3].astype(float))[new[:, :, 3] >= 128].mean(), 1)

    def test_metal_reflection_moves_with_the_key_light(self):
        reflection = self.record['metal_reflection_check']
        self.assertTrue(reflection['changed'])
        self.assertGreater(reflection['mean_abs_luma_difference'], 2.0)
        self.assertTrue(self.record['checks']['metal_reflection'])
        first = np.asarray(Image.open(OUT/'renders/new-material-close.png').convert('RGBA'))
        second = np.asarray(Image.open(OUT/'renders/new-metal-alternate.png').convert('RGBA'))
        self.assertEqual(first.shape, second.shape)
        mask = first[:, :, 3] >= 128
        self.assertGreater(np.abs(first[:, :, :3].astype(float) -
                                  second[:, :, :3].astype(float))[mask].mean(), 2.0)

    def test_source_fidelity_is_explicitly_diagnostic_and_failed(self):
        fidelity = self.record['source_fidelity_diagnostic']
        self.assertFalse(fidelity['passed'])
        self.assertEqual(fidelity['filled_holes'], 1)
        self.assertIn('diagnostic', self.record['source_diagnostic_note'])
        self.assertIn('uncalibrated', self.record['ratio_note'])
        self.assertIn('inconsistent', self.record['ratio_note'])
        self.assertTrue(self.record['pending_review'])
        self.assertLess(fidelity['iou'], .90)

    def test_candidate_is_independent_and_hashed(self):
        self.assertNotIn('01阀门', str(OUT))
        self.assertNotIn('02球阀', str(OUT))
        self.assertNotIn('03轴承座', str(OUT))
        self.assertTrue((ROOT/'outputs/01阀门-六视图重建-v7/result.json').is_file())
        self.assertTrue((ROOT/'outputs/03轴承座-六视图重建-v3/result.json').is_file())
        self.assertTrue((OUT/'hashes.json').is_file())
        hashes = json.loads((OUT/'hashes.json').read_text(encoding='utf8'))
        for name in ('mesh.glb', 'model.blend', 'base_color.png',
                     'metallic_roughness.png', 'normal.png', '浏览器核验.json'):
            self.assertIn(name, hashes['files'])
            self.assertEqual(hashes['files'][name], digest(OUT/name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
