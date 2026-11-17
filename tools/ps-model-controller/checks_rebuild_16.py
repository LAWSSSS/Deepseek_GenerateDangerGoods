"""Integration checks for the generated red aerosol-can candidate (item 16)."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
CANDIDATES = sorted(
    (path for path in (ROOT / 'outputs').glob('16杀虫剂罐2-六视图重建-v*')
     if (path / 'result.json').is_file()),
    key=lambda path: int(path.name.rsplit('-v', 1)[-1]))
OUT = CANDIDATES[-1] if CANDIDATES else ROOT / 'outputs/16杀虫剂罐2-六视图重建-v4'


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


class RebuildChecks16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'result.json').is_file():
            raise unittest.SkipTest('16 review candidate has not been generated')
        cls.record = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT / 'mesh.glb')
        cls.row = next(
            row for row in
            (json.loads(line) for line in LIBRARY.read_text(encoding='utf8').splitlines()
             if line.strip())
            if row['number'] == 16)
        cls.config = json.loads(
            (ROOT / 'tools/ps-model-controller/spraycan16_config.json')
            .read_text(encoding='utf8'))

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['number'], 16)
        self.assertEqual(self.record['version'], 'spraycan16-parametric-v1')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['shape_basis'],
                         'source_photo_with_uncalibrated_visual_reference')
        self.assertEqual(self.record['reference_role'], 'visual_match_unverified')
        self.assertEqual(self.record['mesh_sha256'], digest(OUT / 'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT / 'model.blend'))
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertIn('elliptical top cross-section', self.record['original_structural_warning'])
        self.assertEqual(self.record['next_number'], 17)
        for key in ('original_mesh', 'source_photo', 'source_view', 'source_alpha'):
            path = Path(self.record[key]).resolve()
            self.assertTrue(path.is_file(), key)
            self.assertIn(ROOT, path.parents, key)
        self.assertIn('relocated', self.record['provenance_note'])
        raw = (OUT / 'result.json').read_text(encoding='utf8')
        self.assertNotIn('scrap-composite', raw)
        self.assertEqual(Path(self.record['mesh']).resolve().parent, OUT)
        self.assertEqual(Path(self.record['blend']).resolve().parent, OUT)

    def test_checks_and_diagnostics(self):
        checks = self.record['checks']
        for key in ('mesh_integrity', 'isolated_openings', 'structural_counts',
                    'cylindrical_body', 'material_textures', 'glb_materials',
                    'metal_reflection', 'dielectric_response',
                    'reference_diagnostic_only', 'source_fidelity',
                    'review_page_assets'):
            self.assertIs(checks[key], True, key)
        self.assertEqual(checks['browser_page'], 'not_run')
        self.assertFalse(self.record['page_assets_check']['missing_links'])
        self.assertTrue(self.record['page_assets_check']['passed'])
        self.assertEqual(self.record['glb_audit']['image_count'], 9)
        self.assertTrue(self.record['glb_audit']['passed'])
        for material in self.record['glb_audit']['materials']:
            if material['name'] in ('body', 'cap', 'base'):
                self.assertTrue(material['base_color_texture'])
                self.assertTrue(material['metallic_roughness_texture'])
                self.assertTrue(material['normal_texture'])
        fidelity = self.record['source_fidelity_diagnostic']
        self.assertTrue(fidelity['passed'])
        self.assertGreater(fidelity['iou'], 0.90)
        self.assertEqual(fidelity['filled_holes'], 0)
        self.assertIn('diagnostic', self.record['source_diagnostic_note'])
        for view, result in self.record['reference_views'].items():
            self.assertTrue(result['diagnostic_only'], view)

    def test_mesh_topology_and_cylinder(self):
        audit = self.record['mesh_audit']
        self.assertEqual(audit['parts'], 7)
        self.assertEqual(audit['closed_after_welding_seams'], 7)
        self.assertEqual(audit['open_parts'], [])
        self.assertEqual(audit['nonfinite_parts'], [])
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['worker']['cap_boundary_edges_after_boolean'], 0)
        cylinder = self.record['cylinder_check']
        self.assertLess(cylinder['max_abs_radius_deviation_m'], 1e-6)
        self.assertLess(cylinder['max_ring_radius_variation_m'], 1e-6)
        self.assertAlmostEqual(self.record['top_view_aspect']['new'], 1.0, places=4)
        self.assertGreater(abs(self.record['top_view_aspect']['old'] - 1.0), 0.30)
        self.assertEqual(set(self.parts), {
            'Can body', 'Can base', 'Cap shell', 'Cap recess rim',
            'Actuator button', 'Nozzle insert', 'Cap seam ring'})
        bounds = trimesh.load(OUT / 'mesh.glb', force='scene').bounds
        self.assertTrue(np.allclose(bounds, self.record['mesh_audit']['bounds'], atol=2e-5))

    def test_structural_openings(self):
        structural = self.record['structural_checks']
        self.assertEqual(structural['bodies'], 1)
        self.assertEqual(structural['bases'], 1)
        self.assertEqual(structural['caps'], 1)
        self.assertEqual(structural['recess_rims'], 1)
        self.assertEqual(structural['actuators'], 1)
        self.assertEqual(structural['nozzles'], 1)
        self.assertEqual(structural['seam_rings'], 1)
        ray = self.record['ray_checks']
        self.assertTrue(ray['cap_scoop']['passed'])
        self.assertGreater(ray['cap_scoop']['depth_m'], 0.0015)
        self.assertTrue(ray['nozzle_hole']['passed'])
        self.assertEqual(ray['nozzle_hole']['centre_hits_outside_insert'], 0)
        self.assertGreaterEqual(ray['nozzle_hole']['offset_hits_on_insert'], 1)
        self.assertTrue(ray['bottom_dish']['passed'])
        self.assertGreater(ray['bottom_dish']['centre_minus_rim_m'], 0.0010)

    def test_materials(self):
        stats = self.record['texture_stats']
        self.assertTrue(stats['body']['passed'])
        self.assertTrue(stats['cap']['passed'])
        self.assertTrue(stats['base']['passed'])
        self.assertLessEqual(stats['body']['mean_metallic'], 0.02)
        self.assertLessEqual(stats['cap']['mean_metallic'], 0.02)
        self.assertGreaterEqual(stats['base']['mean_metallic'], 0.95)
        self.assertGreaterEqual(stats['base']['mean_roughness'], 0.20)
        self.assertGreater(stats['body']['front_mean_luma'], 40)
        self.assertLess(stats['body']['dark_fraction_below_20'], 0.30)
        self.assertEqual(self.record['metal_reflection_check']['status'], 'passed')
        self.assertEqual(self.record['dielectric_response_check']['status'], 'passed')
        for name in ('body_base.png', 'body_mr.png', 'body_normal.png',
                     'cap_base.png', 'cap_mr.png', 'cap_normal.png',
                     'base_base.png', 'base_mr.png', 'base_normal.png'):
            self.assertTrue((OUT / name).is_file(), name)

    def test_boards_renders_and_page(self):
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg',
                     '结构开口-隔离检查.jpg', '材质-同光照对照.jpg',
                     '材质-变光检查.jpg', '表面反射-变光对照.jpg',
                     '转台-24方向.jpg', '源图-原视角诊断.jpg',
                     'review.html', 'hashes.json', 'progress.json', 'result.json'):
            self.assertTrue((OUT / name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            for prefix in ('new', 'old', 'clay'):
                self.assertTrue((OUT / 'renders' / f'{prefix}-{view}.png').is_file(),
                                f'{prefix}-{view}')
        for detail in ('cap-top', 'cap-recess', 'recess-rim', 'nozzle', 'cap-seam',
                       'front-label', 'back-label', 'body', 'body-surface',
                       'bottom', 'bottom-dish', 'metal-rim', 'surface'):
            self.assertTrue((OUT / 'renders' / f'detail-{detail}.png').is_file(), detail)
        for group in ('body-only', 'cap-only', 'cap-only-top', 'base-only',
                      'base-only-bottom', 'metal-base-key', 'metal-base-alt',
                      'new-light-alternate', 'new-surface-alternate',
                      'new-material-close', 'old-material-close'):
            self.assertTrue((OUT / 'renders' / f'{group}.png').is_file(), group)
        for i in range(24):
            self.assertTrue((OUT / 'renders' / f'turn-{i:02d}.png').is_file())
        with Image.open(OUT / 'renders' / 'new-front.png') as image:
            self.assertEqual(image.size, (768, 768))
        hashes = json.loads((OUT / 'hashes.json').read_text(encoding='utf8'))
        self.assertEqual(hashes['files']['mesh.glb'], self.record['mesh_sha256'])
        self.assertEqual(hashes['files']['model.blend'], self.record['blend_sha256'])
        self.assertIn('result.json', hashes['files'])
        self.assertEqual(hashes['inputs']['clean'], self.record['inputs']['clean'])
        self.assertEqual(hashes['inputs']['rust'], self.record['inputs']['rust'])
        self.assertEqual(hashes['inputs']['reference'], self.record['inputs']['reference'])
        self.assertEqual(hashes['inputs']['old_mesh'], self.record['original_mesh_sha256'])
        self.assertEqual(hashes['inputs']['source_view'], digest(Path(self.record['source_view'])))
        self.assertEqual(hashes['inputs']['clean'], digest(Path(self.record['source_photo'])))
        progress = json.loads((OUT / 'progress.json').read_text(encoding='utf8'))
        self.assertEqual(progress['status'], 'candidate_generated_pending_review')
        self.assertTrue(progress['manual_only'])
        self.assertEqual(progress['next_number'], 17)
        self.assertEqual(Path(progress['out']), OUT)
        self.assertEqual(Path(progress['mesh']), OUT / 'mesh.glb')
        self.assertEqual(Path(progress['blend']), OUT / 'model.blend')
        external = ROOT / 'work/direct-deepseek/16-progress.json'
        self.assertTrue(external.is_file())
        data = json.loads(external.read_text(encoding='utf8'))
        self.assertEqual(data['number'], 16)
        self.assertTrue(data['manual_only'])
        self.assertEqual(data['next_number'], 17)
        self.assertTrue(Path(data['out']).is_dir())
        self.assertTrue(Path(data['mesh']).is_file())
        self.assertTrue(Path(data['blend']).is_file())
        self.assertEqual(data['checks']['browser_page'], 'not_run')
        self.assertTrue(data['checks']['review_page_assets'])

    def test_config_matches_generated_dimensions(self):
        dimensions = self.record['dimensions']
        self.assertEqual(dimensions['total_height_m'], self.config['dimensions']['total_height_m'])
        self.assertEqual(dimensions['body_radius_m'], self.config['dimensions']['body_radius_m'])
        bounds = self.record['mesh_audit']['bounds']
        self.assertAlmostEqual(bounds[1][2] - bounds[0][2],
                               dimensions['total_height_m'] + 0.0005, places=4)
        self.assertAlmostEqual(bounds[1][0] - bounds[0][0],
                               2 * dimensions['body_radius_m'] + 0.0004, places=4)


if __name__ == '__main__':
    unittest.main()
