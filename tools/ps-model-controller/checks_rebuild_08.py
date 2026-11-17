"""Integration checks against the generated 08 separator-filter candidate."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/08分离过滤器-六视图重建-v3'
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


class RebuildChecks08(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('08 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.scene = trimesh.load(OUT/'mesh.glb', force='scene')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 8)

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
        self.assertEqual(self.record['version'], 'separator08-parametric-v1')
        self.assertEqual(self.record['number'], 8)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertEqual(self.record['inputs']['clean'], self.row['clean_sha256'])
        self.assertIn('floating fragments', self.record['original_structural_warning'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        self.assertEqual(self.record['source_fidelity_diagnostic']['holes'], 1)
        self.assertEqual(self.record['source_fidelity_diagnostic']['filled_holes'], 0)
        self.assertEqual(self.record['source_fidelity_diagnostic']['narrow_recall'], 1.0)
        self.assertGreaterEqual(self.record['source_fidelity_diagnostic']['iou'], .85)
        self.assertTrue(self.record['checks']['reference_diagnostic_only'])
        self.assertTrue(all(row['uncalibrated_reference_mask_iou'] is not None
                            for row in self.record['reference_views']))

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(self.scene.bounds, self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 65)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Separator body', 'Top seam weld', 'Bottom seam weld',
                     'Nozzle -Y pipe', 'Nozzle +Y pipe', 'Nozzle -Y flange',
                     'Nozzle +Y flange', 'Lift lug eye', 'Drain body',
                     'Drain knurled collar', 'Drain window lens +X',
                     'Drain window lens -X', 'Drain spigot', 'Drain bore end'):
            self.assertIn(name, names)
        structural = self.record['structural_checks']
        self.assertTrue(structural['separator_body'])
        self.assertEqual(structural['seam_welds'], 2)
        self.assertEqual(structural['nozzle_pipes'], 2)
        self.assertEqual(structural['nozzle_pipe_sleeves'], 2)
        self.assertEqual(structural['flanges'], 2)
        self.assertEqual(structural['raised_faces'], 4)
        self.assertEqual(structural['nozzle_welds'], 2)
        self.assertEqual(structural['lift_lug_parts'], 5)
        self.assertEqual(structural['drain_blue_parts'], 5)
        self.assertTrue(structural['drain_brass_nut'])
        self.assertTrue(structural['drain_washer'])
        self.assertTrue(structural['drain_knurled_collar'])
        self.assertEqual(structural['drain_collar_ribs'], 36)
        self.assertTrue(structural['drain_body'])
        self.assertEqual(structural['drain_window_bevels'], 2)
        self.assertEqual(structural['drain_window_lenses'], 2)
        self.assertEqual(structural['drain_window_backings'], 2)
        self.assertEqual(structural['drain_spigot_parts'], 3)
        self.assertTrue(structural['drain_bore_end'])
        openings = self.record['isolated_opening_checks']
        self.assertEqual(openings['flange_bores']['detected_openings'], 5)
        self.assertEqual(openings['lift_eye']['detected_openings'], 1)
        self.assertEqual(openings['drain_bore']['detected_openings'], 1)
        self.assertTrue(all(row['passed'] for row in openings.values()))
        self.assertTrue(self.record['checks']['mesh_integrity'])
        self.assertTrue(self.record['checks']['isolated_openings'])
        self.assertTrue(self.record['checks']['structural_counts'])

    def test_real_bores_and_holes_by_ray_casting(self):
        parts = load_parts(OUT/'mesh.glb')
        names = list(parts)
        # A bolt hole at 45 degrees on the +Y flange: the centre ray passes
        # through the hole and first hits the shell; an offset ray hits the
        # flange face much earlier.
        bolt_x = .0625 * np.cos(np.deg2rad(45))
        bolt_z = .220 + .0625 * np.sin(np.deg2rad(45))
        centre = first_hit(parts, names, (bolt_x, .32, bolt_z), (0, -1, 0))
        offset = first_hit(parts, names, (bolt_x + .0105, .32, bolt_z), (0, -1, 0))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .05)
        # The nozzle centre bore is open through the flange and nozzle until
        # the closed shell; the offset ray hits the flange face.
        centre = first_hit(parts, names, (0, .32, .220), (0, -1, 0))
        offset = first_hit(parts, names, (.035, .32, .220), (0, -1, 0))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .10)
        # A ray along the eye axis through the hole centre must hit nothing.
        lug = [name for name in names if name.startswith('Lift lug')]
        self.assertEqual(len(lug), 5)
        self.assertIsNone(first_hit(parts, lug, (-.12, 0, .459), (1, 0, 0)))
        # The spigot/nut passage is open; the centre ray reaches the bore end
        # while an offset ray hits the spigot bottom sooner.
        drain = [name for name in names if name.startswith('Drain')]
        centre = first_hit(parts, drain, (0, 0, -.55), (0, 0, 1))
        offset = first_hit(parts, drain, (.008, 0, -.55), (0, 0, 1))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .03)
        self.assertGreater(centre, -.47)

    def test_browser_review_recorded(self):
        browser = self.record.get('browser_check')
        self.assertIsNotNone(browser)
        self.assertEqual(browser['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])
        self.assertEqual(browser['url'], 'http://127.0.0.1:8882/review/08')
        result = browser['result']
        self.assertTrue(result['old_new_differ'])
        self.assertTrue(result['clay_material_differ'])
        self.assertTrue(result['pointer_rotates'])
        self.assertEqual(result['errors'], [])
        self.assertIn('/api/review/08/board', result['links'])
        self.assertIn('/api/review/08/blend', result['links'])

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
        paint = self.record['paint_texture_checks']
        self.assertTrue(paint['passed'])
        self.assertGreater(paint['blue_paint']['metallic_mean'], .40)
        self.assertLess(paint['blue_paint']['roughness_mean'], .60)
        self.assertGreater(paint['rusted_paint']['roughness_mean'], .60)
        self.assertLess(paint['rusted_paint']['metallic_mean'], .40)
        self.assertGreater(paint['rusted_paint']['rust_fraction'], .20)
        brass = self.record['brass_texture_checks']
        self.assertTrue(brass['passed'])
        self.assertGreater(brass['metallic_mean'], .85)
        self.assertGreater(brass['red_minus_blue'], 30)
        self.assertTrue(self.record['checks']['material_tiles'])
        self.assertTrue(self.record['checks']['metal_reflection'])
        self.assertTrue(self.record['metal_reflection_check']['changed'])
        self.assertTrue((OUT/'model.blend').is_file())
        for name in ('paint_base.png', 'paint_mr.png', 'paint_normal.png',
                     'brass_base.png', 'brass_mr.png', 'brass_normal.png'):
            self.assertTrue((OUT/name).is_file(), name)
        for board in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                      '材质-同光照对照.jpg', '材质-变光检查.jpg',
                      '金属反射-变光对照.jpg', '转台-24方向.jpg'):
            self.assertTrue((OUT/board).is_file(), board)


if __name__ == '__main__':
    unittest.main()
