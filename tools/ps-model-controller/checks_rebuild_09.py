"""Integration checks against the generated 09 separator-filter candidate."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/09分离过滤器1-六视图重建-v4'
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


class RebuildChecks09(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('09 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.scene = trimesh.load(OUT/'mesh.glb', force='scene')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 9)

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
        self.assertEqual(self.record['version'], 'separator09-parametric-v1')
        self.assertEqual(self.record['number'], 9)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('one filled source hole', self.record['original_structural_warning'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        self.assertEqual(self.record['source_fidelity_diagnostic']['holes'], 1)
        self.assertEqual(self.record['source_fidelity_diagnostic']['filled_holes'], 1)
        self.assertEqual(self.record['source_fidelity_diagnostic']['narrow_recall'], 1.0)
        self.assertGreaterEqual(self.record['source_fidelity_diagnostic']['iou'], .80)
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertTrue(all(row['uncalibrated_reference_mask_iou'] is not None
                            for row in self.record['reference_views']))

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(self.scene.bounds, self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 80)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Separator body', 'Upper bulb seam A', 'Upper bulb seam B',
                     'Bulb to shell seam', 'Shell to lower bulb seam',
                     'Lower bulb to skirt seam', 'Skirt to bottom head seam',
                     'Top pipe', 'Top flange', 'Top flange hub',
                     'Side pipe', 'Side flange', 'Side port boss', 'Side red arrow',
                     'Top red arrow', 'Support skirt', 'Bottom head',
                     'Support leg 1', 'Leg gusset 1', 'Foot pad 1',
                     'Drain body', 'Drain sight glass', 'Drain brass center',
                     'Drain spigot nut', 'Drain spigot cone', 'Drain blue collar',
                     'Drain blue tube', 'Drain frame post 1', 'Bore liner top',
                     'Bore liner side'):
            self.assertIn(name, names)
        structural = self.record['structural_checks']
        self.assertTrue(structural['separator_body'])
        self.assertEqual(structural['seam_welds'], 6)
        self.assertEqual(structural['top_flange_parts'], 6)
        self.assertEqual(structural['vent_parts'], 3)
        self.assertEqual(structural['side_nozzle_parts'], 8)
        self.assertEqual(structural['nameplate_parts'], 6)
        self.assertEqual(structural['red_arrows'], 2)
        self.assertTrue(structural['skirt'])
        self.assertTrue(structural['bottom_head'])
        self.assertEqual(structural['legs'], 4)
        self.assertEqual(structural['gussets'], 4)
        self.assertEqual(structural['foot_pads'], 4)
        self.assertEqual(structural['drain_mount_parts'], 4)
        self.assertTrue(structural['drain_body'])
        self.assertEqual(structural['drain_end_caps'], 2)
        self.assertEqual(structural['drain_bands'], 2)
        self.assertEqual(structural['drain_sight_parts'], 4)
        self.assertEqual(structural['drain_spigot_parts'], 4)
        self.assertEqual(structural['drain_elbow_parts'], 3)
        self.assertTrue(structural['drain_steel_tube'])
        self.assertEqual(structural['drain_frame_parts'], 12)
        self.assertEqual(structural['bore_liners'], 2)
        openings = self.record['isolated_opening_checks']
        self.assertEqual(openings['top_flange']['detected_openings'], 9)
        self.assertEqual(openings['side_flange']['detected_openings'], 9)
        self.assertEqual(openings['foot_slot']['detected_openings'], 1)
        self.assertEqual(openings['drain_bore']['detected_openings'], 1)
        self.assertTrue(all(row['passed'] for row in openings.values()))
        self.assertTrue(self.record['checks']['mesh_integrity'])
        self.assertTrue(self.record['checks']['isolated_openings'])
        self.assertTrue(self.record['checks']['structural_counts'])

    def test_real_bores_and_holes_by_ray_casting(self):
        parts = load_parts(OUT/'mesh.glb')
        names = list(parts)
        # The exported model is centred: design z 0 maps to the record's min z.
        z0 = self.record['worker']['bounds'][0][2]
        z_side = .8107 + z0
        # Top flange bolt hole at phase 0 (+X): centre ray passes the flange
        # and reaches the bulb; the offset ray hits the flange top.
        centre = first_hit(parts, names, (.0635, 0, .55), (0, 0, -1))
        offset = first_hit(parts, names, (.0755, 0, .55), (0, 0, -1))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .05)
        # Top central bore: the ray reaches the closed body below the pipe,
        # while the offset ray hits the flange top.
        centre = first_hit(parts, names, (.010, .010, .55), (0, 0, -1))
        offset = first_hit(parts, names, (.045, 0, .55), (0, 0, -1))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .05)
        # Side flange bolt hole at phase 0 (+X): centre ray passes the flange
        # and reaches the bulb; the offset ray hits the flange face.
        centre = first_hit(parts, names, (.070, .32, z_side), (0, -1, 0))
        offset = first_hit(parts, names, (.082, .32, z_side), (0, -1, 0))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .05)
        # Side central bore: centre ray reaches the bulb; offset hits the face.
        centre = first_hit(parts, names, (0, .32, z_side), (0, -1, 0))
        offset = first_hit(parts, names, (.060, .32, z_side), (0, -1, 0))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .05)
        # Foot slot is a real through hole in the isolated pad.
        pad = ['Foot pad 1']
        slot_x = slot_y = .129 * np.cos(np.deg2rad(45))
        self.assertIsNone(first_hit(parts, pad, (slot_x, slot_y, .05), (0, 0, -1)))
        self.assertIsNotNone(first_hit(parts, pad, (slot_x + .010, slot_y, .05), (0, 0, -1)))
        # Drain brass/blue bore is open through the fitting group.
        drain = ['Drain spigot nut', 'Drain spigot cone', 'Drain blue collar', 'Drain blue tube']
        self.assertIsNone(first_hit(parts, drain, (.030, 0, -.55), (0, 0, 1)))
        self.assertIsNotNone(first_hit(parts, drain, (.038, 0, -.55), (0, 0, 1)))

    def test_browser_review_recorded(self):
        browser = self.record.get('browser_check')
        self.assertIsNotNone(browser)
        self.assertEqual(browser['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])
        self.assertEqual(browser['url'], 'http://127.0.0.1:8884/review/09')
        result = browser['result']
        self.assertTrue(result['old_new_differ'])
        self.assertTrue(result['clay_material_differ'])
        self.assertTrue(result['pointer_rotates'])
        self.assertEqual(result['errors'], [])
        for link in ('/api/review/09/board', '/api/review/09/blend',
                     '/api/review/09/topface', '/api/review/09/foot',
                     '/api/review/09/vent'):
            self.assertIn(link, result['links'])

    def test_pbr_textures_and_metal_response(self):
        materials = []
        for mesh in self.scene.geometry.values():
            material = mesh.visual.material
            if material is None:
                continue
            materials.append(material)
        self.assertGreaterEqual(len(materials), 5)
        textured = [m for m in materials if getattr(m, 'baseColorTexture', None) is not None]
        self.assertTrue(textured)
        self.assertTrue(any(getattr(m, 'metallicRoughnessTexture', None) is not None
                            and m.metallicRoughnessTexture.size == (2048, 2048)
                            for m in materials))
        self.assertTrue(any(getattr(m, 'normalTexture', None) is not None
                            and m.normalTexture.size == (2048, 2048)
                            for m in materials))
        self.assertTrue(any(getattr(m, 'baseColorTexture', None) is not None
                            and m.baseColorTexture.size == (1024, 1024)
                            for m in materials))
        steel = self.record['steel_texture_checks']
        self.assertTrue(steel['passed'])
        self.assertGreater(steel['steel_clean']['metallic_mean'], .85)
        self.assertLess(steel['steel_clean']['roughness_mean'], .45)
        self.assertLess(steel['steel_clean']['chroma'], 12)
        self.assertGreater(steel['steel_rusted']['roughness_mean'], .55)
        self.assertLess(steel['steel_rusted']['metallic_mean'], .80)
        self.assertGreater(steel['steel_rusted']['rust_fraction'], .10)
        brass = self.record['brass_texture_checks']
        self.assertTrue(brass['passed'])
        self.assertGreater(brass['metallic_mean'], .85)
        self.assertGreater(brass['red_minus_blue'], 25)
        self.assertTrue(self.record['checks']['material_tiles'])
        self.assertTrue(self.record['checks']['metal_reflection'])
        self.assertTrue(self.record['metal_reflection_check']['changed'])
        self.assertTrue((OUT/'model.blend').is_file())
        for name in ('steel_base.png', 'steel_mr.png', 'steel_normal.png',
                     'brass_base.png', 'brass_mr.png', 'brass_normal.png'):
            self.assertTrue((OUT/name).is_file(), name)
        for board in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                      '材质-同光照对照.jpg', '材质-变光检查.jpg',
                      '金属反射-变光对照.jpg', '转台-24方向.jpg'):
            self.assertTrue((OUT/board).is_file(), board)


if __name__ == '__main__':
    unittest.main()
