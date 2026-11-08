"""Integration checks against the generated gas-cylinder candidate (item 21)."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/21气瓶2-六视图重建-v4'
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


class RebuildChecks21(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'result.json').is_file():
            raise unittest.SkipTest('21 review candidate has not been generated')
        cls.record = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT / 'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 21)
        cls.config = json.loads((ROOT / 'outputs/ps-model-controller/gascylinder21_config.json')
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
        self.assertEqual(self.record['version'], 'gascylinder21-parametric-v1')
        self.assertEqual(self.record['number'], 21)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.record['original_mesh'])),
                         self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('not reused', self.record['original_structural_warning'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['acceptance_gate'])
        self.assertFalse(self.record['source_fidelity_bbox_aligned']['acceptance_gate'])
        self.assertEqual(self.record['reference_name'],
                         'ChatGPT 图像 2026年10月14日 10_01_32-2.png')
        self.assertEqual(self.record['model_identity'],
                         self.config['proportion_source']['model_identity'])
        self.assertEqual(self.record['next_number'], 22)

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(trimesh.load(OUT / 'mesh.glb', force='scene').bounds,
                                    self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertEqual(self.record['mesh_audit']['parts'], 15)
        self.assertEqual(self.record['mesh_audit']['parts'],
                         len(self.record['worker']['parts']))
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_real_recesses(self):
        expected = self.record['structural_expected']
        structural = self.record['structural_checks']
        for key, value in expected.items():
            self.assertEqual(structural[key], value, key)
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Gas cylinder body', 'Bottom brushed plate', 'Valve body',
                     'Outlet spout', 'Safety plug body', 'Safety plug knurl',
                     'Valve nameplate', 'Valve packing nut', 'Valve spindle',
                     'Black scalloped handwheel', 'Handwheel metal cap',
                     'Handwheel centre screw'):
            self.assertIn(name, names)
        opening = self.record['isolated_opening_checks']
        rays = opening['rays']
        self.assertTrue(rays['passed'])
        self.assertTrue(rays['spout']['bore_depth_proved'])
        self.assertTrue(rays['safety_plug']['socket_depth_proved'])
        self.assertTrue(opening['recess']['passed'])
        silhouette = self.record['rendered_silhouette_checks']
        self.assertTrue(silhouette['passed'])
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertEqual(silhouette['detected'][f'new-{view}'], 0)

    def test_materials_and_metal_reflection(self):
        for name in ('body_base', 'body_mr', 'body_normal',
                     'steel_base', 'steel_mr', 'steel_normal',
                     'black_base', 'black_mr', 'black_normal',
                     'cap_base', 'cap_mr', 'cap_normal',
                     'bottom_base', 'bottom_mr', 'bottom_normal'):
            self.assertTrue((OUT / (name + '.png')).is_file(), name)
        self.assertTrue(self.record['texture_checks']['passed'])
        self.assertTrue(self.record['checks']['material_tiles'])
        body = self.record['texture_checks']['materials']['body']
        self.assertGreaterEqual(body['mean_metallic'], .80)
        self.assertGreaterEqual(body['rust_fraction'], .02)
        self.assertLessEqual(body['rust_fraction'], .13)
        black = self.record['texture_checks']['materials']['black']
        self.assertLessEqual(black['mean_metallic'], .12)
        reflection = self.record['metal_reflection_check']
        self.assertEqual(reflection['status'], 'applicable_metal')
        self.assertTrue(reflection['changed'])
        self.assertTrue(reflection['mean_abs_luma_difference'] >= 2.0)

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
        self.assertEqual(progress['next_number'], 22)
        self.assertTrue(Path(progress['out']).is_dir())
        for key in ('result', 'mesh', 'blend'):
            self.assertTrue(Path(progress[key]).is_file(), key)
        external = ROOT / 'work/direct-deepseek/21-progress.json'
        self.assertTrue(external.is_file())
        external_progress = json.loads(external.read_text(encoding='utf8'))
        self.assertEqual(external_progress['number'], 21)
        self.assertEqual(external_progress['status'],
                         'candidate_generated_pending_review')
        for key in ('out', 'mesh', 'blend', 'checks', 'pending_review', 'next_number'):
            self.assertIn(key, external_progress)
        self.assertTrue(Path(external_progress['mesh']).is_file())
        self.assertTrue(Path(external_progress['blend']).is_file())
        self.assertEqual(external_progress['next_number'], 22)


if __name__ == '__main__':
    unittest.main()
