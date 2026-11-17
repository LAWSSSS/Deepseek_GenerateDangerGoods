"""Integration checks against the generated 05 vertical stainless-tank candidate."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/05储罐-六视图重建-v1'
LIBRARY = ROOT/'outputs/三维模型库-圆润复验/results.jsonl'


class RebuildChecks05(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('05 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.scene = trimesh.load(OUT/'mesh.glb', force='scene')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 5)

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT/'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT/'model.blend'))
        self.assertEqual(self.record['shape_basis'], 'user_six_view_reference')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'tank05-parametric-v1')
        self.assertEqual(self.record['number'], 5)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertIn('0.4615', self.record['original_structural_warning'])
        self.assertEqual(self.record['browser_check']['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(self.scene.bounds, self.record['worker']['bounds'], atol=2e-6))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 60)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_real_through_openings(self):
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Tank body', 'Shell seam upper', 'Shell seam lower',
                     'Manway neck', 'Manway flange', 'Manway gasket', 'Manway lid',
                     'Manway handle bar', 'Foot plate 1', 'Foot plate 2', 'Foot plate 3',
                     'Front port upper', 'Front port lower', 'Drain pipe',
                     'Drain valve body', 'Top nozzle 1', 'Top nozzle 2',
                     'Top nozzle 3', 'Top nozzle 4'):
            self.assertIn(name, names)
        structural = self.record['structural_checks']
        self.assertTrue(structural['tank_body'])
        self.assertEqual(structural['seam_bands'], 2)
        self.assertEqual(structural['legs'], 3)
        self.assertEqual(structural['leg_brackets'], 3)
        self.assertEqual(structural['foot_plates'], 3)
        self.assertTrue(structural['manway_neck'])
        self.assertTrue(structural['manway_lid'])
        self.assertEqual(structural['manway_clamp_lugs'], 6)
        self.assertEqual(structural['manway_clamp_bolts'], 6)
        self.assertEqual(structural['manway_clamp_eyes'], 6)
        self.assertEqual(structural['top_nozzles'], 4)
        self.assertEqual(structural['top_flanges'], 2)
        self.assertEqual(structural['top_ferrule_lips'], 2)
        self.assertEqual(structural['top_bolts'], 12)
        self.assertEqual(structural['side_ferrules'], 2)
        self.assertEqual(structural['front_ports'], 2)
        self.assertEqual(structural['open_bore_sleeves'], 7)
        self.assertTrue(structural['drain_valve_body'])
        openings = self.record['isolated_opening_checks']
        self.assertEqual(openings['top_nozzles']['detected_openings'], 4)
        self.assertEqual(openings['front_ports']['detected_openings'], 2)
        self.assertEqual(openings['drain']['detected_openings'], 1)
        self.assertEqual(openings['foot_slots']['detected_openings'], 3)
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
        self.assertGreater(checks['bright_stainless']['metallic_mean'], .90)
        self.assertLess(checks['bright_stainless']['roughness_mean'], .35)
        self.assertGreater(checks['stained_steel']['roughness_mean'], .70)
        self.assertLess(checks['stained_steel']['metallic_mean'], .60)
        self.assertTrue((OUT/'model.blend').is_file())
        for name in ('base_color.png', 'metallic_roughness.png', 'normal.png'):
            self.assertTrue((OUT/name).is_file(), name)

    def test_six_views_boards_and_diagnostic_state(self):
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
        self.assertEqual(len(list((OUT/'renders').glob('turn-*.png'))), 24)
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg', '金属反射-变光对照.jpg',
                     '转台-24方向.jpg', '浏览器核验.json'):
            self.assertTrue((OUT/name).is_file(), name)
        for name in ('detail-manway', 'detail-clamps', 'detail-top-nozzles',
                     'detail-front-port', 'detail-lower-port', 'detail-legs',
                     'detail-foot', 'detail-drain', 'detail-seam', 'detail-surface',
                     'top-nozzles-only', 'front-ports-only', 'drain-only', 'feet-only'):
            self.assertTrue((OUT/'renders'/f'{name}.png').is_file(), name)
        self.assertEqual(len(self.record['reference_views']), 6)
        for row in self.record['reference_views']:
            self.assertIsNotNone(row['uncalibrated_reference_mask_iou'])
            self.assertIn('未标定', row['note'])
        self.assertTrue(self.record['metal_reflection_check']['changed'])
        self.assertTrue(self.record['checks']['metal_reflection'])
        # The source-photo silhouette diagnostic may fail because the stored
        # camera is frontal while the supplied rust photo is 3/4.  It must not
        # be reported as an acceptance pass.
        diagnostic = self.record['source_fidelity_diagnostic']
        self.assertIn('source_fidelity_reason', self.record)
        if not diagnostic['passed']:
            self.assertFalse(self.record['checks'].get('source_fidelity', False))
            self.assertIn('not claimed as passed', self.record['source_fidelity_reason'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
