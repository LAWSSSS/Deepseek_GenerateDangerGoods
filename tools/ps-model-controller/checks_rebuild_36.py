"""Read-only validation of the generated item-36 review candidate.

Run after ``mesh_rebuild_item_36.py``.  It verifies the manifest, input/output
hashes, reviewed checks, topology, GLB material embedding, page assets and the
presence of the six-view/detail/material evidence.  It never edits the
candidate and never promotes it out of manual_only/pending review.
"""
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
    (path for path in (ROOT / 'outputs').glob('36自喷漆罐-六视图重建-v*')
     if (path / 'result.json').is_file()),
    key=lambda path: int(path.name.rsplit('-v', 1)[-1]))
OUT = CANDIDATES[-1] if CANDIDATES else ROOT / 'outputs/36自喷漆罐-六视图重建-v1'
REGISTERED = {
    'clean': '539f6d7bd495525fcf9c2f95e13f904a52e0cd74b2a8fed4c3609ecbb3d4b4c7',
    'rust': '330a386fc2cb67ab9f608383066198ae97c77f08272ebe34e63fd06a0235c3d9',
    'reference': 'e88d442b1240ab1d646d0406df181a020bcf1d2bfe82bc3696a24059c277122e',
    'old_mesh': '96a405f7659618550423b2d2838c84325ad6c7d8345dd0e540935a585555b024',
    'source_view': 'fdf14dbcb91cc3b79ad7b29763d417eee95ed3f3bcb0dbfd1ff3215dee99bd17',
    'source_alpha': '93bb7a1520abf7a5f4aad67d237a4a2d38519777c083a45d0bb344921fd53bfd',
}


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


class RebuildChecks36(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'result.json').is_file():
            raise unittest.SkipTest('36 review candidate has not been generated')
        cls.record = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
        cls.row = next(
            row for row in
            (json.loads(line) for line in LIBRARY.read_text(encoding='utf8').splitlines()
             if line.strip())
            if row['number'] == 36)
        cls.parts = load_parts(OUT / 'mesh.glb')

    def test_manifest_and_hashes(self):
        record = self.record
        self.assertEqual(record['number'], 36)
        self.assertEqual(record['version'], 'spraypaint36-parametric-v1')
        self.assertEqual(record['status'], 'review')
        self.assertEqual(record['model_review'], 'pending')
        self.assertEqual(record['scene_validation'], 'pending')
        self.assertTrue(record['manual_only'])
        self.assertEqual(record['shape_basis'],
                         'source_photo_with_uncalibrated_six_view_reference')
        self.assertEqual(record['reference_role'], 'visual_match_unverified')
        self.assertEqual(record['next_number'], 37)
        self.assertEqual(record['mesh_sha256'], digest(OUT / 'mesh.glb'))
        self.assertEqual(record['blend_sha256'], digest(OUT / 'model.blend'))
        self.assertEqual(record['original_mesh_sha256'], REGISTERED['old_mesh'])
        self.assertEqual(record['inputs']['clean'], REGISTERED['clean'])
        self.assertEqual(record['inputs']['rust'], REGISTERED['rust'])
        self.assertEqual(record['inputs']['reference'], REGISTERED['reference'])
        self.assertEqual(record['inputs']['source_view'], REGISTERED['source_view'])
        self.assertEqual(record['inputs']['source_alpha'], REGISTERED['source_alpha'])
        for key in ('mesh', 'blend', 'source_photo', 'source_view', 'source_alpha',
                    'reference', 'original_mesh'):
            path = Path(record[key]).resolve()
            self.assertTrue(path.is_file(), key)
            self.assertIn(ROOT, path.parents, key)
        self.assertEqual(Path(record['mesh']).parent, OUT)
        self.assertEqual(Path(record['blend']).parent, OUT)
        self.assertNotIn('scrap-composite', (OUT / 'result.json').read_text(encoding='utf8'))
        self.assertTrue(record['pending_review'])
        self.assertTrue(record['limitations'])
        self.assertEqual(record['old_model_analysis']['principal_axis'][2] < 0, True)

    def test_review_checks_and_evidence(self):
        checks = self.record['checks']
        for key in ('mesh_integrity', 'isolated_openings', 'hidden_valve',
                    'structural_counts', 'cylindrical_body', 'material_textures',
                    'glb_materials', 'metal_reflection', 'dielectric_response',
                    'reference_diagnostic_only', 'source_fidelity',
                    'review_page_assets'):
            self.assertIs(checks[key], True, key)
        self.assertEqual(checks['browser_page'], 'passed')
        self.assertTrue(self.record['page_assets_check']['passed'])
        self.assertFalse(self.record['page_assets_check']['missing_links'])
        glb = self.record['glb_audit']
        self.assertTrue(glb['passed'])
        self.assertGreaterEqual(glb['image_count'], 9)
        textured = [m for m in glb['materials'] if m['name'] in ('body', 'cap', 'base')]
        self.assertEqual(len(textured), 3)
        for material in textured:
            self.assertTrue(material['base_color_texture'])
            self.assertTrue(material['metallic_roughness_texture'])
            self.assertTrue(material['normal_texture'])
        fidelity = self.record['source_fidelity_diagnostic']
        self.assertTrue(fidelity['passed'])
        self.assertEqual(fidelity['filled_holes'], 0)
        self.assertIn('diagnostic', self.record['source_diagnostic_note'])
        for view, result in self.record['reference_views'].items():
            self.assertTrue(result['diagnostic_only'], view)
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '表面反射-变光对照.jpg', '转台-24方向.jpg',
                     '源图-原视角诊断.jpg', '圆柱截面-顶视对照.jpg',
                     'PBR图集-布局.jpg', '检查页.html', 'result.json',
                     'hashes.json', 'progress.json'):
            self.assertTrue((OUT / name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            for prefix in ('new', 'old', 'clay'):
                self.assertTrue((OUT / 'renders' / f'{prefix}-{view}.png').is_file(),
                                f'{prefix}-{view}')
        for index in range(24):
            self.assertTrue((OUT / 'renders' / f'turn-{index:02d}.png').is_file())
        for name in ('body_base', 'body_mr', 'body_normal', 'cap_base', 'cap_mr',
                     'cap_normal', 'base_base', 'base_mr', 'base_normal'):
            self.assertTrue((OUT / f'{name}.png').is_file(), name)
        page = (OUT / '检查页.html').read_text(encoding='utf8')
        self.assertIn('turn-00.png', page)
        self.assertIn('mesh.glb', page)
        self.assertIn('model.blend', page)

    def test_topology_and_geometry(self):
        audit = self.record['mesh_audit']
        self.assertEqual(audit['parts'], 4)
        self.assertEqual(audit['closed_after_welding_seams'], 4)
        self.assertEqual(audit['open_parts'], [])
        self.assertEqual(audit['nonfinite_parts'], [])
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['worker']['cap_boundary_edges_after_boolean'], 0)
        self.assertEqual(set(self.parts), {'Can body', 'Can base', 'Cap shell',
                                           'Valve stem'})
        cylinder = self.record['cylinder_check']
        self.assertLess(cylinder['max_abs_radius_deviation_m'], 1e-6)
        self.assertLess(cylinder['max_ring_radius_variation_m'], 1e-6)
        self.assertAlmostEqual(self.record['top_view_aspect']['new'], 1.0, places=2)
        self.assertGreater(self.record['top_view_aspect']['old'], 1.05)
        old = self.record['old_model_analysis']
        self.assertGreater(old['axis_tilt_from_z_deg'], 8.0)
        self.assertGreater(old['overall_cross_section_ellipticity'], 1.05)
        ray = self.record['ray_checks']
        self.assertTrue(ray['cap_cavity']['passed'])
        self.assertGreater(ray['cap_cavity']['cavity_depth_m'], 0.02)
        self.assertTrue(ray['bottom_dish']['passed'])
        self.assertGreater(ray['bottom_dish']['centre_minus_rim_m'], 0.001)
        self.assertTrue(ray['hidden_valve']['passed'])
        metal = self.record['metal_reflection_check']
        self.assertEqual(metal['status'], 'passed')
        self.assertGreaterEqual(metal['mean_metallic'], 0.98)
        dielectric = self.record['dielectric_response_check']
        self.assertEqual(dielectric['status'], 'passed')
        bounds = trimesh.load(OUT / 'mesh.glb', force='scene').bounds
        self.assertAlmostEqual(float(bounds[0][2]), 0.0, places=5)
        self.assertAlmostEqual(float(bounds[1][2]), 0.2, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
