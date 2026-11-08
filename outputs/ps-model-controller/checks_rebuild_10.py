"""Integration checks against the generated 10 bottle-jack candidate."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/10千斤顶-六视图重建-v2'
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
        values.extend(ray_triangles(origin, direction, parts[name]).tolist())
    return min(values) if values else None


class RebuildChecks10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('10 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT/'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 10)

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT/'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT/'model.blend'))
        self.assertEqual(self.record['shape_basis'],
                         'source_photo_with_uncalibrated_visual_match_reference')
        self.assertEqual(self.record['reference_role'], 'visual_match_unverified')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'jack10-parametric-v1')
        self.assertEqual(self.record['number'], 10)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('left/right views collapsed', self.record['original_structural_warning'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        self.assertGreaterEqual(self.record['source_fidelity_diagnostic']['iou'], .55)
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertTrue(all(row['uncalibrated_reference_mask_iou'] is not None
                            for row in self.record['reference_views']))
        self.assertTrue(self.record['original_source_fidelity_record']['passed'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(trimesh.load(OUT/'mesh.glb', force='scene').bounds,
                                    self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 40)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        for name in ('Cast base', 'Base body collar', 'Base boss front 1',
                     'Base bolt front 1', 'Release valve hex', 'Jack body',
                     'Hex collar', 'Upper hex collar', 'Ram neck', 'Chrome ram',
                     'Saddle skirt', 'Saddle cap', 'Saddle top rim', 'Pump body',
                     'Pump lower collar', 'Pump yoke', 'Pump pivot pin',
                     'Pump handle socket', 'Socket bore liner', 'Socket rim open',
                     'Pump link plate', 'Pump link bolt 1', 'Pump lower pin',
                     'Carry handle loop', 'Handle loop mount 1',
                     'Detached handle bar', 'Handle bar bore liner',
                     'Handle bar top rim', 'Nameplate'):
            self.assertIn(name, self.parts)
        checks = self.record['isolated_opening_checks']
        self.assertEqual(set(checks), {'base_holes', 'socket_bore',
                                       'handle_loop', 'handle_bar'})
        self.assertTrue(all(v['passed'] for v in checks.values()))
        self.assertEqual([v['detected_openings'] for v in checks.values()], [4, 1, 1, 1])

    def test_ray_through_holes(self):
        # Four base corner openings.
        for x, y in ((.145, .16), (-.145, .16), (.145, -.16), (-.145, -.16)):
            self.assertIsNone(first_hit(self.parts, ['Cast base'],
                                        (x, y, -.05), (0, 0, 1)))
        config = json.loads((ROOT/'outputs/ps-model-controller/jack10_config.json')
                            .read_text(encoding='utf8'))
        pump = config['pump']
        center = np.asarray(pump['socket_center'], float)
        axis = np.asarray(pump['socket_axis'], float)
        axis /= np.linalg.norm(axis)
        self.assertIsNone(first_hit(self.parts,
                                    ['Pump handle socket', 'Socket bore liner'],
                                    center - axis * .30, axis))
        hb = config['handle_bar']
        bottom = np.asarray(hb['bottom'], float)
        top = np.asarray(hb['top'], float)
        baxis = top - bottom
        baxis /= np.linalg.norm(baxis)
        self.assertIsNone(first_hit(self.parts,
                                    ['Detached handle bar', 'Handle bar bore liner'],
                                    bottom - baxis * .20, baxis))
        # Carry-loop opening: the ray through the loop centre must miss the rod,
        # while a ray at the top of the loop must hit it.
        self.assertIsNone(first_hit(self.parts, ['Carry handle loop'],
                                    (-.5, -.190, .4675), (1, 0, 0)))
        self.assertIsNotNone(first_hit(self.parts, ['Carry handle loop'],
                                       (-.5, -.190, .600), (1, 0, 0)))

    def test_materials_and_reflection(self):
        for name in ('red_base', 'red_mr', 'red_normal', 'chrome_base',
                     'chrome_mr', 'chrome_normal', 'steel_base', 'steel_mr',
                     'steel_normal', 'saddle_base', 'saddle_mr', 'saddle_normal',
                     'label_base', 'label_mr', 'label_normal'):
            self.assertTrue((OUT/(name + '.png')).is_file(), name)
        self.assertTrue(self.record['red_texture_checks']['passed'])
        self.assertTrue(self.record['chrome_texture_checks']['passed'])
        self.assertTrue(self.record['steel_texture_checks']['passed'])
        self.assertTrue(self.record['saddle_texture_checks']['passed'])
        self.assertTrue(self.record['label_texture_checks']['passed'])
        self.assertTrue(self.record['checks']['material_tiles'])
        self.assertTrue(self.record['metal_reflection_check']['changed'])
        self.assertGreater(self.record['metal_reflection_check']['mean_abs_luma_difference'], 2.0)

    def test_boards_exist(self):
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '金属反射-变光对照.jpg', '转台-24方向.jpg', 'hashes.json'):
            self.assertTrue((OUT/name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
        with Image.open(OUT/'renders'/'new-front.png') as im:
            self.assertEqual(im.size, (768, 768))


if __name__ == '__main__':
    unittest.main()
