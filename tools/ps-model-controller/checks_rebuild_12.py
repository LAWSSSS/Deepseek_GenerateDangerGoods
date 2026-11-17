"""Integration checks against the generated 2 t trolley-jack candidate (item 12)."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/12千斤顶2-六视图重建-v3'
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


class RebuildChecks12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('12 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT/'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 12)
        cls.config = json.loads((ROOT/'tools/ps-model-controller/jack12_config.json')
                                .read_text(encoding='utf8'))

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
        self.assertEqual(self.record['version'], 'jack12-parametric-v1')
        self.assertEqual(self.record['number'], 12)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('smoothed, fused surface', self.record['original_structural_warning'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        self.assertFalse(self.record['source_fidelity_aligned_3q_diagnostic']['passed'])
        self.assertFalse(self.record['source_fidelity_aligned_3q_diagnostic']['acceptance_gate'])
        self.assertIsNotNone(self.record['source_fidelity_aligned_turn_index'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertTrue(all(row['uncalibrated_reference_mask_iou'] is not None
                            for row in self.record['reference_views']))
        self.assertTrue(self.record['original_source_fidelity_record']['passed'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(trimesh.load(OUT/'mesh.glb', force='scene').bounds,
                                    self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 55)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        for name in ('Frame left plate', 'Frame right plate', 'Frame floor',
                     'Frame pump crossmember', 'Frame left rib', 'Frame right rib',
                     'Front wheel left', 'Front wheel right', 'Front hub left',
                     'Front hub right', 'Front axle', 'Rear wheel left',
                     'Rear wheel right', 'Rear fork plate left 1',
                     'Caster stem left', 'Caster ear left', 'Rear cross rod',
                     'Lifting arm', 'Arm pivot boss', 'Arm pivot pin',
                     'Lifting head', 'Saddle post', 'Head pin', 'Head cotter pin',
                     'Link plate left', 'Link plate right', 'Saddle pad',
                     'Saddle rim', 'Pump cover', 'Pump top label',
                     'Pump side label left', 'Pump filler cap', 'Handle socket',
                     'Socket ear left', 'Socket bolt left', 'Socket bracket',
                     'Handle pipe', 'Handle grip', 'Grip end cap', 'Carry loop',
                     'Loop bush left', 'Loop bolt left', 'Front rating label',
                     'Warning label left', 'Warning label right'):
            self.assertIn(name, self.parts)
        checks = self.record['isolated_opening_checks']
        self.assertEqual(set(checks), {'carry_loop', 'front_hub_bore', 'socket_bore'})
        self.assertTrue(all(v['passed'] for v in checks.values()))
        self.assertEqual(checks['carry_loop']['detected_openings'], 0)
        self.assertTrue(checks['carry_loop']['inner_clear'])
        self.assertTrue(checks['carry_loop']['top_bar'])
        self.assertTrue(checks['carry_loop']['left_leg'])
        self.assertTrue(checks['carry_loop']['right_leg'])
        self.assertEqual(checks['front_hub_bore']['detected_openings'], 1)
        self.assertEqual(checks['socket_bore']['detected_openings'], 1)
        self.assertTrue(self.record['structural_checks']['frame_pump_crossmember'])
        self.assertEqual(self.record['structural_checks']['front_hubs'], 2)
        self.assertEqual(self.record['structural_checks']['front_hub_nuts'], 2)
        self.assertEqual(self.record['structural_checks']['rear_fork_plates'], 4)

    def test_ray_through_openings(self):
        w = self.config['wheels']
        # Carry-loop opening: a ray along X through the U centre must miss the loop.
        self.assertIsNone(first_hit(self.parts, ['Carry loop'], (-1.0, 0, 0.24), (1, 0, 0)))
        # The top bar of the loop must block the same ray height.
        self.assertIsNotNone(first_hit(self.parts, ['Carry loop'],
                                       (-1.0, 0, self.config['loop']['top_z']), (1, 0, 0)))
        # Front hub through-bore along Y must be open; the wall must block an offset ray.
        hub_origin = np.array([w['front_x'], -0.5, w['front_radius']])
        self.assertIsNone(first_hit(self.parts, ['Front hub left', 'Front hub right'],
                                    hub_origin, (0, 1, 0)))
        self.assertIsNotNone(first_hit(self.parts, ['Front hub left', 'Front hub right'],
                                       hub_origin + np.array([0, 0, .020]), (0, 1, 0)))
        # Handle-socket bore along X is open; its wall must block an offset ray.
        s = self.config['socket']
        self.assertIsNone(first_hit(self.parts, ['Handle socket'],
                                    (-0.5, 0, s['center'][2]), (1, 0, 0)))
        self.assertIsNotNone(first_hit(self.parts, ['Handle socket'],
                                       (-0.5, 0, s['center'][2] + .024), (1, 0, 0)))

    def test_materials_and_reflection(self):
        for name in ('red_base', 'red_mr', 'red_normal', 'chrome_base',
                     'chrome_mr', 'chrome_normal', 'steel_base', 'steel_mr',
                     'steel_normal', 'gold_base', 'gold_mr', 'gold_normal',
                     'black_base', 'black_mr', 'black_normal', 'yellow_base',
                     'yellow_mr', 'yellow_normal', 'rubber_base', 'rubber_mr',
                     'rubber_normal', 'saddle_base', 'saddle_mr', 'saddle_normal',
                     'ton_label_base', 'ton_label_mr', 'ton_label_normal',
                     'warning_label_base', 'warning_label_mr', 'warning_label_normal'):
            self.assertTrue((OUT/(name + '.png')).is_file(), name)
        for key in ('red_texture_checks', 'chrome_texture_checks',
                    'steel_texture_checks', 'gold_texture_checks',
                    'black_texture_checks', 'yellow_texture_checks',
                    'rubber_texture_checks', 'saddle_texture_checks',
                    'ton_label_checks', 'warning_label_checks'):
            self.assertTrue(self.record[key]['passed'], key)
        self.assertTrue(self.record['checks']['material_tiles'])
        self.assertTrue(self.record['metal_reflection_check']['changed'])
        self.assertGreater(self.record['metal_reflection_check']['mean_abs_luma_difference'], 2.0)

    def test_browser_record(self):
        record = json.loads((OUT/'浏览器核验.json').read_text(encoding='utf8'))
        self.assertEqual(record['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])
        self.assertTrue(record['result']['old_new_differ'])
        self.assertTrue(record['result']['clay_material_differ'])
        self.assertTrue(record['result']['pointer_rotates'])
        self.assertEqual(record['result']['errors'], [])
        self.assertIn('/api/model/asset-7f783ef94e50-rebuild-v1', record['result']['links'])

    def test_boards_exist(self):
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '金属反射-变光对照.jpg', '转台-24方向.jpg', 'hashes.json',
                     '浏览器核验.json'):
            self.assertTrue((OUT/name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
        for detail in ('saddle', 'saddle-top', 'arm-label', 'carry-loop',
                       'loop-mounts', 'pump-cover', 'pump-label',
                       'handle-socket', 'handle-grip', 'front-wheel',
                       'rear-caster', 'frame-side', 'arm-pivot', 'arm-link',
                       'base-bottom', 'rear-frame', 'surface', 'front'):
            self.assertTrue((OUT/'renders'/f'detail-{detail}.png').is_file(), detail)
        with Image.open(OUT/'renders'/'new-front.png') as im:
            self.assertEqual(im.size, (768, 768))


if __name__ == '__main__':
    unittest.main()
