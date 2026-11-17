"""Integration checks against the generated horizontal pressure-tank candidate (item 38)."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/38集油器1-六视图重建-v2'
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
CONFIG = ROOT / 'tools/ps-model-controller/collector38_config.json'


def localize(path_text):
    parts = Path(path_text).parts
    for anchor in ('outputs', 'work'):
        if anchor in parts:
            index = parts.index(anchor)
            return ROOT.joinpath(*parts[index:])
    return Path(path_text)


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


def ray_triangles(origin, direction, mesh):
    """Moller-Trumbore hits (t values) without the rtree dependency."""
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if not len(faces):
        return np.empty(0)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    pvec = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, pvec)
    ok = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tvec = origin - v0
    u = np.einsum('ij,ij->i', tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum('j,ij->i', direction, qvec) * inv
    t = np.einsum('ij,ij->i', e2, qvec) * inv
    hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return t[hit]


def first_hit(parts, names, origin, direction):
    values = []
    for name in names:
        if name not in parts:
            continue
        values.extend(ray_triangles(origin, direction, parts[name]).tolist())
    return min(values) if values else None


class RebuildChecks38(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'result.json').is_file():
            raise unittest.SkipTest('38 review candidate has not been generated')
        cls.record = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT / 'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 38)
        cls.config = json.loads(CONFIG.read_text(encoding='utf8'))

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT / 'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT / 'model.blend'))
        self.assertEqual(self.record['shape_basis'],
                         'source_photo_with_uncalibrated_visual_match_reference')
        self.assertEqual(self.record['reference_role'], 'visual_match_unverified')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'collector38-parametric-v1')
        self.assertEqual(self.record['number'], 38)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(localize(self.row['mesh'])),
                         self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertTrue(self.record['original_source_fidelity_record']['passed'])
        self.assertIn('old model-library source IoU', self.record['original_structural_warning'])
        self.assertIn('fused', self.record['original_structural_warning'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertIn('diagnostic', self.record['source_diagnostic_note'])
        self.assertTrue(all(row['reference_uncalibrated_mask_iou'] is not None
                            for row in self.record['reference_views']))
        self.assertIsNotNone(self.record['source_fidelity_diagnostic']['iou'])
        browser = self.record['browser_check']
        self.assertIn(browser['status'], ('passed', 'failed', 'not_run'))
        if browser['status'] == 'passed':
            self.assertTrue(self.record['checks']['browser_page'])
        else:
            self.assertFalse(self.record['checks']['browser_page'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(trimesh.load(OUT / 'mesh.glb', force='scene').bounds,
                                    self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 40)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        expected = (
            'Tank cylinder', 'Tank head left', 'Tank head right',
            'Tank weld seam left', 'Tank weld seam right',
            'Support left strap', 'Support right strap',
            'Support left front leg', 'Support left back leg',
            'Support right front leg', 'Support right back leg',
            'Support left front foot', 'Support left back foot',
            'Support right front foot', 'Support right back foot',
            'Regulator boss', 'Regulator riser', 'Regulator union',
            'Regulator body', 'Regulator knob', 'Regulator knob stem',
            'Regulator coupler', 'Regulator coupler liner',
            'Regulator coupler collar',
            'Safety boss', 'Safety body', 'Safety cap', 'Safety nipple',
            'Port upper boss', 'Port upper cap', 'Port lower boss', 'Port lower cap',
            'Drain boss', 'Drain nut', 'Drain body', 'Drain outlet',
            'Drain outlet liner', 'Drain handle',
            'Nameplate',
        )
        for name in expected:
            self.assertIn(name, self.parts)
        checks = self.record['isolated_opening_checks']
        self.assertEqual(set(checks), {'foot_holes', 'coupler_socket', 'drain_outlet'})
        self.assertEqual([checks[key]['detected_openings']
                          for key in ('foot_holes', 'coupler_socket', 'drain_outlet')],
                         [4, 1, 1])
        self.assertTrue(all(value['passed'] for value in checks.values()))
        structural = self.record['structural_checks']
        self.assertEqual(structural['tank_shell_parts'], 3)
        self.assertEqual(structural['weld_seam_rings'], 2)
        self.assertEqual(structural['support_straps'], 2)
        self.assertEqual(structural['support_legs'], 4)
        self.assertEqual(structural['support_feet'], 4)
        self.assertEqual(structural['regulator_parts'], 12)
        self.assertEqual(structural['safety_parts'], 4)
        self.assertEqual(structural['port_parts'], 4)
        self.assertEqual(structural['drain_parts'], 8)
        self.assertEqual(structural['weld_beads'], 5)
        self.assertEqual(structural['labels'], 1)
        self.assertTrue(self.record['checks']['structural_counts'])

    def test_ray_through_openings(self):
        sup = self.config['supports']
        for sy in sup['y_positions']:
            for sx in (-1, 1):
                self.assertIsNone(first_hit(
                    self.parts, [f'Support {"left" if sy < 0 else "right"} '
                                 f'{"front" if sx > 0 else "back"} foot'],
                    (sx * sup['hole_x'], sy, -0.02), (0, 0, 1)))
        # A ray offset from the hole must hit the foot plate.
        self.assertIsNotNone(first_hit(
            self.parts, ['Support left front foot'],
            (sup['hole_x'] + .020, sup['y_positions'][0], -0.02), (0, 0, 1)))
        reg = self.config['regulator']
        self.assertIsNone(first_hit(
            self.parts, ['Regulator coupler'],
            (0, reg['coupler_y0'] - .10, reg['body_center'][2]), (0, 1, 0)))
        self.assertIsNotNone(first_hit(
            self.parts, ['Regulator coupler'],
            ((reg['coupler_outer'] + reg['coupler_inner']) / 2,
             reg['coupler_y0'] - .10, reg['body_center'][2]), (0, 1, 0)))
        dr = self.config['drain']
        self.assertIsNone(first_hit(
            self.parts, ['Drain outlet'],
            (0, dr['y'], dr['outlet_z0'] - .05), (0, 0, 1)))
        self.assertIsNotNone(first_hit(
            self.parts, ['Drain outlet'],
            ((dr['outlet_outer'] + dr['outlet_inner']) / 2,
             dr['y'], dr['outlet_z0'] - .05), (0, 0, 1)))

    def test_materials_and_metal_reflection(self):
        for name in ('paint_base', 'paint_mr', 'paint_normal',
                     'brass_base', 'brass_mr', 'brass_normal',
                     'steel_base', 'steel_mr', 'steel_normal',
                     'black_base', 'black_mr', 'black_normal',
                     'label_base', 'label_mr', 'label_normal'):
            self.assertTrue((OUT / (name + '.png')).is_file(), name)
        self.assertTrue(self.record['paint_texture_checks']['passed'])
        self.assertTrue(self.record['metal_texture_checks']['passed'])
        self.assertTrue(self.record['label_texture_checks']['passed'])
        self.assertTrue(self.record['checks']['material_tiles'])
        reflection = self.record['metal_reflection_check']
        self.assertEqual(reflection['status'], 'passed')
        self.assertTrue(reflection['regulator']['changed'])
        self.assertTrue(reflection['drain']['changed'])
        self.assertGreater(reflection['regulator']['mean_abs_luma_difference'], 2.0)
        self.assertGreater(reflection['drain']['mean_abs_luma_difference'], 2.0)
        self.assertIn('dielectric', reflection['note'])

    def test_boards_review_page_and_progress(self):
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '金属反射-变光对照.jpg', '转台-24方向.jpg',
                     'review.html', 'old-mesh.glb', 'hashes.json',
                     '浏览器核验.json'):
            self.assertTrue((OUT / name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT / 'renders' / f'new-{view}.png').is_file())
            self.assertTrue((OUT / 'renders' / f'old-{view}.png').is_file())
        with Image.open(OUT / 'renders' / 'new-front.png') as image:
            self.assertEqual(image.size, (704, 704))
        self.assertTrue((OUT / 'viewer' / 'three.module.js').is_file())
        self.assertTrue((OUT / 'viewer' / 'addons' / 'loaders' / 'GLTFLoader.js').is_file())
        progress_path = ROOT / 'work/direct-deepseek/38-progress.json'
        self.assertTrue(progress_path.is_file())
        progress = json.loads(progress_path.read_text(encoding='utf8'))
        self.assertEqual(progress['number'], 38)
        self.assertEqual(progress['status'], 'candidate_generated_pending_review')
        self.assertEqual(progress['next_number'], 39)
        self.assertTrue(progress['manual_only'])
        self.assertTrue(Path(progress['out']).is_dir())
        self.assertTrue(Path(progress['mesh']).is_file())
        self.assertTrue(Path(progress['blend']).is_file())
        self.assertTrue(progress['pending_review'])
        self.assertEqual(progress['checks'], self.record['checks'])


if __name__ == '__main__':
    unittest.main()
