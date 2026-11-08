"""Integration checks against the generated two-can sealant candidate (item 13)."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/13密封胶罐-六视图重建-v2'
LIBRARY = ROOT/'outputs/三维模型库-圆润复验/results.jsonl'


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


class RebuildChecks13(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('13 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT/'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 13)
        cls.config = json.loads((ROOT/'outputs/ps-model-controller/sealant13_config.json')
                                .read_text(encoding='utf8'))

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT/'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT/'model.blend'))
        self.assertEqual(self.record['shape_basis'],
                         'source_photo_with_uncalibrated_category_reference')
        self.assertEqual(self.record['reference_role'], 'category_reference_unverified')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'sealant13-parametric-v1')
        self.assertEqual(self.record['number'], 13)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('fused', self.record['original_structural_warning'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        aligned = self.record['source_fidelity_aligned_3q_diagnostic']
        self.assertIsNotNone(aligned['iou'])
        self.assertFalse(aligned['acceptance_gate'])
        self.assertIn('not a calibrated camera', aligned['alignment'])
        self.assertIn('acceptance gate', self.record['source_fidelity_aligned_note'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertTrue(all(row['reference_single_can_mask_iou'] is not None
                            for row in self.record['reference_views']))
        self.assertTrue(self.record['original_source_fidelity_record']['passed'])
        self.assertEqual(self.record['reference_name'], '白色塑料圆罐六视图产品展示.png')

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(trimesh.load(OUT/'mesh.glb', force='scene').bounds,
                                    self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 26)
        self.assertEqual(self.record['mesh_audit']['parts'], 30)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        expected = (
            'Can front body', 'Can rear body', 'Can front lid', 'Can rear lid',
            'Can front lid seam', 'Can rear lid seam',
            'Can front body seam', 'Can rear body seam',
            'Can front recess rim', 'Can rear recess rim',
            'Can front latch upper', 'Can rear latch upper',
            'Can front latch lower', 'Can rear latch lower',
            'Can front bottom gate', 'Can rear bottom gate',
            'Can front bottom tab 1', 'Can front bottom tab 2',
            'Can rear bottom tab 1', 'Can rear bottom tab 2',
            'Can front bottom mark bar 1', 'Can rear bottom mark bar 1',
            'Can front bottom mark letter 1', 'Can rear bottom mark letter 1',
        )
        for name in expected:
            self.assertIn(name, self.parts)
        checks = self.record['isolated_opening_checks']
        self.assertEqual(set(checks),
                         {'single_can_silhouettes', 'two_can_top_separation',
                          'front_rear_body_vertex_gap'})
        self.assertTrue(checks['single_can_silhouettes']['passed'])
        self.assertEqual(checks['single_can_silhouettes']['detected_openings'],
                         {'front': 0, 'back': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        self.assertTrue(checks['two_can_top_separation']['passed'])
        self.assertEqual(checks['two_can_top_separation']['detected_components'], 2)
        self.assertTrue(checks['front_rear_body_vertex_gap']['passed'])
        self.assertGreater(checks['front_rear_body_vertex_gap']['metres'], .005)
        structural = self.record['structural_checks']
        self.assertEqual(structural['bodies'], 2)
        self.assertEqual(structural['lids'], 2)
        self.assertEqual(structural['recess_rings'], 2)
        self.assertEqual(structural['latch_uppers'], 2)
        self.assertEqual(structural['latch_lowers'], 2)
        self.assertEqual(structural['bottom_tabs'], 4)
        self.assertEqual(structural['bottom_gates'], 2)
        self.assertEqual(structural['bottom_mark_bars'], 6)
        self.assertEqual(structural['bottom_mark_letters'], 4)

    def test_materials_and_surface_response(self):
        for name in ('white_base', 'white_mr', 'white_normal',
                     'mark_base', 'mark_mr', 'mark_normal'):
            self.assertTrue((OUT/(name + '.png')).is_file(), name)
        self.assertTrue(self.record['white_texture_checks']['passed'])
        self.assertTrue(self.record['mark_texture_checks']['passed'])
        self.assertTrue(self.record['checks']['material_tiles'])
        for tile in self.record['white_texture_checks']['tiles']:
            self.assertLessEqual(tile['mean_metallic'], .06)
            self.assertGreaterEqual(tile['mean_roughness'], .30)
            self.assertLessEqual(tile['mean_roughness'], .78)
        self.assertEqual(self.record['checks']['metal_reflection'],
                         'not_applicable_plastic')
        self.assertEqual(self.record['metal_reflection_check']['status'],
                         'not_applicable_metal')
        response = self.record['metal_reflection_check']['dielectric_response']
        self.assertTrue(response['changed'])
        self.assertGreater(response['mean_abs_luma_difference'], 2.0)

    def test_browser_record(self):
        record = json.loads((OUT/'浏览器核验.json').read_text(encoding='utf8'))
        self.assertEqual(record['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])
        self.assertTrue(record['result']['old_new_differ'])
        self.assertTrue(record['result']['clay_material_differ'])
        self.assertTrue(record['result']['pointer_rotates'])
        self.assertEqual(record['result']['errors'], [])
        self.assertIn('/api/model/asset-71e8dec9675d-rebuild-v1',
                      record['result']['links'])
        self.assertIn('/api/review/13/board', record['result']['links'])
        self.assertIn('/api/review/13/blend', record['result']['links'])

    def test_boards_and_progress_exist(self):
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '表面反射-变光对照.jpg', '转台-24方向.jpg', 'hashes.json',
                     '浏览器核验.json', 'result.json', 'progress.json'):
            self.assertTrue((OUT/name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'single-{view}.png').is_file())
        for detail in ('front', 'top', 'lid-top', 'lid-rim', 'seam-latch',
                       'front-latch', 'bottom', 'bottom-mark', 'can-gap',
                       'body', 'surface', 'lid-recess'):
            self.assertTrue((OUT/'renders'/f'detail-{detail}.png').is_file(), detail)
        for group in ('body-only', 'lid-only', 'latch-only', 'bottom-parts-only',
                      'two-cans-top', 'new-light-alternate', 'new-surface-alternate',
                      'old-material-close'):
            self.assertTrue((OUT/'renders'/f'{group}.png').is_file(), group)
        for i in range(24):
            self.assertTrue((OUT/'renders'/f'turn-{i:02d}.png').is_file())
        with Image.open(OUT/'renders'/'new-front.png') as im:
            self.assertEqual(im.size, (768, 768))
        progress = json.loads((OUT/'progress.json').read_text(encoding='utf8'))
        self.assertEqual(progress['status'], 'candidate_generated_pending_review')
        self.assertTrue(progress['manual_only'])
        self.assertEqual(progress['next_number'], 14)
        self.assertTrue(progress['checks']['browser_page'])
        self.assertTrue(Path(progress['out']).is_dir())
        for key in ('result', 'mesh', 'blend'):
            self.assertTrue(Path(progress[key]).is_file(), key)
        external = ROOT/'work/direct-deepseek/13-progress.json'
        self.assertTrue(external.is_file())
        external_progress = json.loads(external.read_text(encoding='utf8'))
        self.assertEqual(external_progress['number'], 13)
        self.assertTrue(Path(external_progress['mesh']).is_file())
        self.assertTrue(Path(external_progress['blend']).is_file())


if __name__ == '__main__':
    unittest.main()
