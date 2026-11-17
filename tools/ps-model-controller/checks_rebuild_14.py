"""Integration checks for the generated blue aerosol-can candidate (item 14).

These checks read the generated evidence and re-open the GLB. They never modify
the candidate. Ray-opening results are produced by the Blender worker (BVH)
because this Python environment does not include trimesh's optional rtree
dependency.
"""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest
from mesh_rebuild_item import inspect_mesh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/14杀虫剂罐-六视图重建-v2'
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
OLD_REL = 'outputs/三维模型库-圆润复验/items/14-asset-b0985bfda5f0/mesh.glb'
ALPHA_REL = 'outputs/三维模型库-圆润复验/items/14-asset-b0985bfda5f0/source-alpha.png'
CLEAN_REL = 'outputs/物品高清输入-39件/原貌HD/14-杀虫剂罐.png'
RUST_REL = 'outputs/物品高清输入-39件/生锈HD/14-杀虫剂罐.png'


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


class RebuildChecks14(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'result.json').is_file():
            raise unittest.SkipTest('14 review candidate has not been generated')
        cls.record = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT / 'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines()
                                   if line.strip()) if r['number'] == 14)
        cls.config = json.loads((ROOT / 'tools/ps-model-controller/aerosol14_config.json')
                                .read_text(encoding='utf8'))

    def test_manifest_hashes_and_review_only_state(self):
        r = self.record
        self.assertEqual(r['number'], 14)
        self.assertEqual(r['version'], 'aerosol14-parametric-v1')
        self.assertEqual(r['status'], 'review')
        self.assertEqual(r['model_review'], 'pending')
        self.assertEqual(r['scene_validation'], 'pending')
        self.assertTrue(r['manual_only'])
        self.assertEqual(r['shape_basis'], 'source_photo_with_uncalibrated_six_view_reference')
        self.assertEqual(r['reference_role'], 'visual_match_unverified')
        self.assertEqual(r['reference_name'], '蓝色榄菊杀虫喷雾六视图.png')
        self.assertEqual(r['reference_sha256'], digest(Path(r['reference'])))
        self.assertEqual(r['mesh_sha256'], digest(OUT / 'mesh.glb'))
        self.assertEqual(r['blend_sha256'], digest(OUT / 'model.blend'))
        self.assertEqual(Path(r['original_mesh']).resolve(), (ROOT / OLD_REL).resolve())
        # The library row stores an absolute path from the old main worktree.
        # This candidate uses the identical current-worktree file and verifies
        # its registered hash instead of reading the old tree.
        self.assertEqual(r['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(r['original_mesh_sha256'], digest(ROOT / OLD_REL))
        self.assertEqual(r['original_clean_sha256'], digest(ROOT / CLEAN_REL))
        self.assertEqual(r['original_rust_sha256'], digest(ROOT / RUST_REL))
        self.assertEqual(r['inputs']['source_alpha'], digest(ROOT / ALPHA_REL))
        self.assertNotIn('scrap-composite', json.dumps(r, ensure_ascii=False))
        hashes = json.loads((OUT / 'hashes.json').read_text(encoding='utf8'))
        for name, expected in hashes['files'].items():
            if name == 'hashes.json':
                self.assertEqual(expected, 'self')
                continue
            self.assertTrue((OUT / name).is_file(), name)
            self.assertEqual(expected, digest(OUT / name), name)

    def test_mesh_topology_and_parts(self):
        scene = trimesh.load(OUT / 'mesh.glb', force='scene')
        self.assertTrue(np.allclose(scene.bounds, self.record['worker']['bounds'], atol=2e-5))
        audit = self.record['mesh_audit']
        self.assertEqual(audit['open_parts'], [])
        self.assertFalse(audit['nonfinite_parts'])
        self.assertGreaterEqual(audit['parts'], 16)
        self.assertEqual(audit['parts'], audit['closed_after_welding_seams'])
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        expected = {
            'Can body', 'Can label', 'Can bottom cup', 'Can bottom rim',
            'Can bottom step', 'Can bottom ridge', 'Can cap', 'Can cap seam',
            'Can nozzle boss', 'Can nozzle collar', 'Can actuator',
            'Can actuator rib 1', 'Can actuator rib 2', 'Can actuator rib 3',
            'Can actuator rib 4', 'Can nozzle insert',
        }
        self.assertEqual(set(self.parts), expected)
        structural = self.record['structural_checks']
        for key in ('bodies', 'labels', 'caps', 'bottom_cups', 'bottom_rims',
                    'actuators', 'nozzle_inserts', 'nozzle_bosses'):
            self.assertEqual(structural[key], 1, key)
        self.assertEqual(structural['actuator_ribs'], 4)

    def test_openings_and_silhouettes(self):
        opening = self.record['isolated_opening_checks']
        self.assertTrue(opening['passed'])
        self.assertTrue(opening['well_recess_passed'])
        self.assertGreaterEqual(opening['well_depth_m'], .006)
        self.assertTrue(opening['cap_bottom_opening_passed'])
        self.assertGreaterEqual(opening['cap_hollow_difference_m'], .02)
        self.assertTrue(opening['nozzle_bore_passed'])
        self.assertLess(opening['bore_hit_x'], opening['actuator_front_x'] - .0005)
        self.assertAlmostEqual(opening['bore_wall_hit_x'], opening['actuator_front_x'],
                               delta=.002)
        self.assertEqual(opening, self.record['worker']['opening_checks'])
        silhouettes = self.record['rendered_silhouette_checks']
        self.assertTrue(silhouettes['passed'])
        self.assertEqual(set(silhouettes['detected_holes']),
                         {'front', 'back', 'left', 'right', 'top', 'bottom'})
        self.assertTrue(all(v == 0 for v in silhouettes['detected_holes'].values()))

    def test_materials_and_reflection(self):
        checks = self.record['texture_checks']
        self.assertTrue(checks['passed'])
        self.assertLessEqual(checks['label']['mean_metallic'], .03)
        self.assertLessEqual(checks['cap']['mean_metallic'], .03)
        self.assertGreaterEqual(checks['metal']['mean_metallic'], .95)
        self.assertGreaterEqual(checks['bottom']['mean_metallic'], .95)
        self.assertLess(checks['bottom']['code_region_min_luma'],
                        checks['bottom']['code_region_p90_luma'])
        reflection = self.record['reflection_checks']
        metal = reflection['metal_response']
        self.assertTrue(metal['changed'])
        self.assertIn('criterion', metal)
        self.assertTrue((metal['highlight_shift_px'] or 0) > 1.0 or
                        metal['mean_abs_luma_difference'] > 2.0)
        self.assertTrue(reflection['dielectric_changed'])
        self.assertTrue(self.record['checks']['metal_reflection'])
        self.assertTrue(self.record['checks']['surface_response'])

    def test_reference_and_source_diagnostics(self):
        views = self.record['reference_views']
        self.assertEqual({row['view'] for row in views}, set(REFRESH_VIEWS))
        for row in views:
            for key in ('reference_new_iou', 'reference_old_iou', 'old_new_iou'):
                self.assertIsNotNone(row[key], (row['view'], key))
                self.assertGreaterEqual(row[key], 0.0)
            self.assertIn('not a measured camera', row['note'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertIn('diagnostic', self.record['ratio_note'].lower())
        source = self.record['source_fidelity_diagnostic']
        self.assertIn('iou', source)
        aligned = self.record['source_fidelity_aligned_3q_diagnostic']
        self.assertIsNotNone(aligned)
        self.assertFalse(aligned['acceptance_gate'])
        self.assertIn('not a calibrated camera', aligned['alignment'])
        self.assertIn('acceptance gate', self.record['source_fidelity_aligned_note'])
        self.assertTrue(self.record['original_source_fidelity_record']['passed'])

    def test_browser_page_record(self):
        record = json.loads((OUT / '浏览器核验.json').read_text(encoding='utf8'))
        self.assertEqual(record['status'], 'passed')
        result = record['result']
        self.assertTrue(result['passed'])
        self.assertTrue(result['images_loaded'])
        self.assertTrue(result['old_new_differ'])
        self.assertTrue(result['clay_material_differ'])
        self.assertTrue(result['turntable_drag_changed'])
        self.assertTrue(result['view_switch_changed'])
        self.assertEqual(result['errors'], [])
        self.assertTrue(self.record['checks']['browser_page'])
        page = (OUT / '检查页.html').read_text(encoding='utf8')
        for need in ('manual_only', 'mesh.glb', 'model.blend', '六视图-新旧对照.jpg',
                     'browser_page</td><td><span class="ok">通过</span>',
                     '金属反射-变光对照.jpg'):
            self.assertIn(need, page)
        screenshot = OUT / '浏览器核验.png'
        self.assertTrue(screenshot.is_file())
        with Image.open(screenshot) as image:
            self.assertGreater(image.width, 1000)

    def test_boards_and_progress(self):
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '金属反射-变光对照.jpg', '转台-24方向.jpg', '检查页.html',
                     '浏览器核验.json', '浏览器核验.png', 'result.json',
                     'progress.json', 'hashes.json'):
            self.assertTrue((OUT / name).is_file(), name)
        for view in REFRESH_VIEWS:
            self.assertTrue((OUT / 'renders' / f'new-{view}.png').is_file())
            self.assertTrue((OUT / 'renders' / f'old-{view}.png').is_file())
        for detail in ('front', 'cap-front', 'cap-top', 'cap-well', 'nozzle',
                       'label-logo', 'label-lower', 'label-side', 'cap-seam',
                       'bottom', 'bottom-code', 'bottom-rim', 'body-side',
                       'surface', 'cap-scratch'):
            self.assertTrue((OUT / 'renders' / f'detail-{detail}.png').is_file(), detail)
        for group in ('cap-only', 'label-only', 'metal-only', 'actuator-only',
                      'cap-top-isolated', 'cap-bottom-isolated',
                      'new-light-alternate', 'new-bottom-alternate',
                      'new-clay-front', 'new-clay-close', 'old-material-close'):
            self.assertTrue((OUT / 'renders' / f'{group}.png').is_file(), group)
        for i in range(24):
            self.assertTrue((OUT / 'renders' / f'turn-{i:02d}.png').is_file())
        with Image.open(OUT / 'renders' / 'new-front.png') as image:
            self.assertEqual(image.size, (768, 768))
        progress = json.loads((OUT / 'progress.json').read_text(encoding='utf8'))
        for key in ('number', 'status', 'out', 'mesh', 'blend', 'checks',
                    'pending_review', 'next_number'):
            self.assertIn(key, progress)
        self.assertEqual(progress['number'], 14)
        self.assertEqual(progress['status'], 'candidate_generated_pending_review')
        self.assertTrue(progress['manual_only'])
        self.assertEqual(progress['next_number'], 15)
        self.assertTrue(progress['checks']['browser_page'])
        self.assertIn(str(ROOT).lower(), progress['out'].lower())
        self.assertTrue(Path(progress['out']).is_dir())
        for key in ('mesh', 'blend', 'result', 'detail_board',
                    'isolated_board', 'comparison_board'):
            self.assertTrue(Path(progress[key]).is_file(), key)
        external = ROOT / 'work/direct-deepseek/14-progress.json'
        self.assertTrue(external.is_file())
        external_progress = json.loads(external.read_text(encoding='utf8'))
        self.assertEqual(external_progress['number'], 14)
        self.assertEqual(Path(external_progress['out']).resolve(),
                         (OUT / 'progress.json').parent.resolve())
        self.assertTrue(Path(external_progress['out']).is_dir())
        for key in ('mesh', 'blend'):
            self.assertTrue(Path(external_progress[key]).is_file(), key)


REFRESH_VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')

if __name__ == '__main__':
    unittest.main()
