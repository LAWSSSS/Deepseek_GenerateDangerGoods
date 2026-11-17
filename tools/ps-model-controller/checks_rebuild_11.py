"""Integration checks against the generated QL5 screw-jack candidate (item 11)."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/11千斤顶1-六视图重建-v3'
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


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class RebuildChecks11(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('11 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.parts = load_parts(OUT/'mesh.glb')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 11)
        cls.config = json.loads((ROOT/'tools/ps-model-controller/jack11_config.json')
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
        self.assertEqual(self.record['version'], 'jack11-parametric-v1')
        self.assertEqual(self.record['number'], 11)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('smoothed, fused surface', self.record['original_structural_warning'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        self.assertGreaterEqual(self.record['source_fidelity_diagnostic']['iou'], .20)
        self.assertFalse(self.record['source_fidelity_aligned_3q_diagnostic']['passed'])
        self.assertGreaterEqual(self.record['source_fidelity_aligned_3q_diagnostic']['iou'], .60)
        self.assertFalse(self.record['source_fidelity_aligned_3q_diagnostic']['acceptance_gate'])
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertTrue(all(row['uncalibrated_reference_mask_iou'] is not None
                            for row in self.record['reference_views']))
        self.assertTrue(self.record['original_source_fidelity_record']['passed'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(trimesh.load(OUT/'mesh.glb', force='scene').bounds,
                                    self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 45)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        for name in ('Cast base', 'Base bottom rib 1', 'Base center boss',
                     'Base center recess', 'Base body collar', 'Jack body',
                     'Top collar', 'Collar lug 1', 'Collar lug pin 1',
                     'Chrome screw', 'Screw cap lip', 'Screw cap top',
                     'Screw cap top rim', 'Nameplate', 'Emboss pad', 'Emboss text',
                     'Crank housing', 'Lever socket', 'Socket bore liner',
                     'Socket rim', 'Socket bolt 1', 'Lever square tube',
                     'Lever bore liner', 'Mechanism side plate',
                     'Mechanism side bolt', 'Lower bracket', 'Lower bracket bolt',
                     'Link plate', 'Link pin', 'Release knob stem',
                     'Release knob cap', 'Release knob bore liner',
                     'Release knob rim', 'Carry handle loop', 'Loop backplate',
                     'Body back rib', 'Loop mount 1', 'Loop pin 1'):
            self.assertIn(name, self.parts)
        checks = self.record['isolated_opening_checks']
        self.assertEqual(set(checks), {'lever_bore', 'handle_loop'})
        self.assertTrue(all(v['passed'] for v in checks.values()))
        self.assertEqual([v['detected_openings'] for v in checks.values()], [1, 1])

    def test_ray_through_openings(self):
        mech = self.config['mechanism']
        phi = math.radians(mech['azimuth_deg'])
        dr = np.array([math.cos(phi), math.sin(phi), 0.0])
        dz = np.array([0.0, 0.0, 1.0])
        center = dr * mech['socket_center_radius'] + dz * mech['socket_z']
        angle = math.radians(mech['lever_angle_deg'])
        dl = unit(dr * math.cos(angle) + dz * math.sin(angle))
        # A ray down the lever axis must pass through the hollow rectangular bar.
        self.assertIsNone(first_hit(self.parts,
                                    ['Lever square tube', 'Lever bore liner'],
                                    center + dl * (mech['lever_length'] + .20), -dl))
        # A ray inside the wall must hit the bar.
        up = np.array([0.0, 0.0, 1.0])
        u = unit(up - dl * np.dot(up, dl))
        wall_origin = center + dl * .20 + u * .017
        self.assertIsNotNone(first_hit(self.parts, ['Lever square tube'],
                                       wall_origin, dl))
        # Loop centre through the hole must miss the loop; the outer/top rod must hit.
        hl = self.config['handle_loop']
        lphi = math.radians(hl['azimuth_deg'])
        rvec = np.array([math.cos(lphi), math.sin(lphi), 0.0])
        tvec = np.array([-math.sin(lphi), math.cos(lphi), 0.0])
        loop_center = rvec * hl['center_radius'] + dz * hl['center_z']
        self.assertIsNone(first_hit(self.parts, ['Carry handle loop'],
                                    loop_center - tvec * .40, tvec))
        loop_top = rvec * hl['center_radius'] + dz * (hl['center_z'] + hl['half_z'])
        self.assertIsNotNone(first_hit(self.parts, ['Carry handle loop'],
                                       loop_top - tvec * .40, tvec))
        loop_outer = rvec * (hl['center_radius'] + hl['half_radius']) + dz * hl['center_z']
        self.assertIsNotNone(first_hit(self.parts, ['Carry handle loop'],
                                       loop_outer - tvec * .40, tvec))

    def test_materials_and_reflection(self):
        for name in ('green_base', 'green_mr', 'green_normal', 'chrome_base',
                     'chrome_mr', 'chrome_normal', 'steel_base', 'steel_mr',
                     'steel_normal', 'label_base', 'label_mr', 'label_normal'):
            self.assertTrue((OUT/(name + '.png')).is_file(), name)
        self.assertTrue(self.record['green_texture_checks']['passed'])
        self.assertTrue(self.record['chrome_texture_checks']['passed'])
        self.assertTrue(self.record['steel_texture_checks']['passed'])
        self.assertTrue(self.record['label_texture_checks']['passed'])
        self.assertTrue(self.record['plastic_checks']['passed'])
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
        self.assertIn('/api/model/asset-fecd0dbc8545-rebuild-v1', record['result']['links'])

    def test_boards_exist(self):
        for name in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                     '材质-同光照对照.jpg', '材质-变光检查.jpg',
                     '金属反射-变光对照.jpg', '转台-24方向.jpg', 'hashes.json',
                     '浏览器核验.json'):
            self.assertTrue((OUT/name).is_file(), name)
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
        for detail in ('lever-tip', 'lever-socket', 'knob', 'mechanism-side',
                       'loop', 'loop-mounts', 'collar', 'screw', 'cap', 'label',
                       'emboss', 'base-front', 'base-bottom', 'surface'):
            self.assertTrue((OUT/'renders'/f'detail-{detail}.png').is_file(), detail)
        with Image.open(OUT/'renders'/'new-front.png') as im:
            self.assertEqual(im.size, (768, 768))


if __name__ == '__main__':
    unittest.main()
