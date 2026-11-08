"""Integration checks against the generated mountain-bike shock candidate (item 18)."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/18气压减震器-六视图重建-v1'
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


class RebuildChecks18(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'result.json').is_file():
            raise unittest.SkipTest('18 review candidate has not been generated')
        cls.record = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT / 'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 18)
        cls.config = json.loads((ROOT / 'outputs/ps-model-controller/shock18_config.json')
                                .read_text(encoding='utf8'))

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT / 'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT / 'model.blend'))
        self.assertEqual(self.record['shape_basis'],
                         'source_photo_with_uncalibrated_six_view_reference')
        self.assertEqual(self.record['reference_role'], 'visual_match_unverified')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'shock18-parametric-v1')
        self.assertEqual(self.record['number'], 18)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.record['original_mesh'])),
                         self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('not reused', self.record['original_structural_warning'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        best = self.record['source_fidelity_best_bbox_aligned']
        self.assertIsNotNone(best)
        self.assertFalse(best.get('acceptance_gate', True))
        self.assertIn('not a calibrated camera', best['alignment'])
        self.assertEqual(self.record['reference_name'], '黑色山地车后避震六视图图册.png')
        self.assertEqual(self.record['model_identity'],
                         self.config['proportion_source']['model_identity'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(trimesh.load(OUT / 'mesh.glb', force='scene').bounds,
                                    self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 27)
        self.assertEqual(self.record['mesh_audit']['parts'],
                         len(self.record['worker']['parts']))
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_real_bore_openings(self):
        expected = self.record['structural_expected']
        structural = self.record['structural_checks']
        for key, value in expected.items():
            self.assertEqual(structural[key], value, key)
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Air can', 'Top cap', 'Top yoke', 'Bottom yoke',
                     'Damper shaft', 'Seal head', 'Sag indicator ring',
                     'Red rebound knob', 'Blue lever head',
                     'Blue lever arm vertical', 'Top eyelet bushing',
                     'Bottom eyelet bushing'):
            self.assertIn(name, names)
        opening = self.record['isolated_opening_checks']
        self.assertTrue(opening['silhouettes']['passed'])
        self.assertEqual(opening['silhouettes']['detected']['top-eyelet-only'], 1)
        self.assertEqual(opening['silhouettes']['detected']['bottom-eyelet-only'], 1)
        self.assertEqual(opening['silhouettes']['detected']['eyelet-only'], 2)
        self.assertEqual(opening['silhouettes']['detected']['new-front'], 2)
        self.assertEqual(opening['silhouettes']['detected']['new-top'], 0)
        self.assertEqual(opening['silhouettes']['detected']['new-bottom'], 0)
        self.assertTrue(opening['rays']['passed'])
        self.assertTrue(opening['rays']['top_bore_clear'])
        self.assertTrue(opening['rays']['bottom_bore_clear'])
        self.assertTrue(opening['rays']['top_bushing_material_hit'])
        self.assertTrue(opening['rays']['bottom_bushing_material_hit'])

    def test_materials_and_metal_reflection(self):
        for name in ('can_base', 'can_mr', 'can_normal',
                     'cap_base', 'cap_mr', 'cap_normal',
                     'black_base', 'black_mr', 'black_normal',
                     'steel_base', 'steel_mr', 'steel_normal',
                     'lever_base', 'lever_mr', 'lever_normal'):
            self.assertTrue((OUT / (name + '.png')).is_file(), name)
        self.assertTrue(self.record['texture_checks']['passed'])
        self.assertTrue(self.record['checks']['material_tiles'])
        can = self.record['texture_checks']['materials']['can']
        self.assertGreaterEqual(can['black_band_metallic'], .70)
        self.assertGreaterEqual(can['decal_luma_std'], 20)
        reflection = self.record['metal_reflection_check']
        self.assertEqual(reflection['status'], 'applicable_metal')
        self.assertTrue(reflection['changed'])
        self.assertGreaterEqual(reflection['mean_abs_luma_difference'], 2.0)
        self.assertGreaterEqual(reflection['highlight_shift_px'], 0.0)

    def test_reference_diagnostics_and_boards(self):
        views = self.record['reference_views']
        self.assertEqual([row['view'] for row in views],
                         ['front', 'back', 'left', 'right', 'top', 'bottom'])
        for row in views:
            self.assertIsNotNone(row['reference_new_iou'])
            self.assertIsNotNone(row['reference_old_iou'])
            self.assertIsNotNone(row['old_new_iou'])
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '金属反射-变光对照.jpg', '原视角-对照.jpg', '转台-24方向.jpg',
                     'hashes.json', '浏览器核验.json', '检查页.html', 'result.json',
                     'progress.json'):
            self.assertTrue((OUT / name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT / 'renders' / f'new-{view}.png').is_file())
            self.assertTrue((OUT / 'renders' / f'old-{view}.png').is_file())
        for i in range(24):
            self.assertTrue((OUT / 'renders' / f'turn-{i:02d}.png').is_file())
        with Image.open(OUT / 'renders' / 'new-front.png') as im:
            self.assertEqual(im.size, (768, 768))

    def test_browser_and_progress_records(self):
        record = json.loads((OUT / '浏览器核验.json').read_text(encoding='utf8'))
        self.assertEqual(record['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])
        self.assertTrue(record['result']['old_new_differ'])
        self.assertTrue(record['result']['clay_material_differ'])
        self.assertTrue(record['result']['turntable_drag_changed'])
        self.assertEqual(record['result']['errors'], [])
        self.assertTrue(record['result']['has_glb'])
        self.assertTrue(record['result']['has_blend'])
        self.assertTrue(record['result']['has_board'])
        progress = json.loads((OUT / 'progress.json').read_text(encoding='utf8'))
        self.assertEqual(progress['status'], 'candidate_generated_pending_review')
        self.assertTrue(progress['manual_only'])
        self.assertEqual(progress['next_number'], 19)
        self.assertTrue(Path(progress['out']).is_dir())
        for key in ('result', 'mesh', 'blend'):
            self.assertTrue(Path(progress[key]).is_file(), key)
        external = ROOT / 'work/direct-deepseek/18-progress.json'
        self.assertTrue(external.is_file())
        external_progress = json.loads(external.read_text(encoding='utf8'))
        self.assertEqual(external_progress['number'], 18)
        self.assertEqual(external_progress['status'],
                         'candidate_generated_pending_review')
        for key in ('out', 'mesh', 'blend', 'checks', 'pending_review', 'next_number'):
            self.assertIn(key, external_progress)
        self.assertTrue(Path(external_progress['mesh']).is_file())
        self.assertTrue(Path(external_progress['blend']).is_file())
        self.assertEqual(external_progress['next_number'], 19)


if __name__ == '__main__':
    unittest.main()
