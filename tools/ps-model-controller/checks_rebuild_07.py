"""Integration checks against the generated 07 vertical mixing-tank candidate."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/07储罐-六视图重建-v2'
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


class RebuildChecks07(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('07 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.scene = trimesh.load(OUT/'mesh.glb', force='scene')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 7)

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT/'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT/'model.blend'))
        self.assertEqual(self.record['shape_basis'],
                         'source_photo_with_uncalibrated_category_reference')
        self.assertEqual(self.record['reference_role'], 'category_reference')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'tank07-parametric-v1')
        self.assertEqual(self.record['number'], 7)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])
        self.assertEqual(self.record['reference_sha256'], self.record['inputs']['reference'])
        self.assertEqual(self.record['inputs']['rust'], self.row['rust_sha256'])
        self.assertIn('0.2716', self.record['original_structural_warning'])
        self.assertEqual(self.record['browser_check']['status'], 'passed')
        self.assertTrue(self.record['checks']['browser_page'])
        self.assertFalse(self.record['checks']['source_fidelity'])
        self.assertFalse(self.record['source_fidelity_diagnostic']['passed'])
        self.assertTrue(all(row['uncalibrated_reference_mask_iou'] is not None
                            for row in self.record['reference_views']))

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(self.scene.bounds, self.record['worker']['bounds'], atol=2e-5))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 110)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_real_through_openings(self):
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Tank body', 'Top head seam band', 'Mid shell seam band',
                     'Bottom seam band', 'Leg 1', 'Leg 2', 'Leg 3',
                     'Top port neck', 'Top port lid', 'Top pipe', 'Top pipe sleeve',
                     'Pipe valve body', 'Sight glass lens', 'Cone flange ring',
                     'Lower cone', 'Drain pipe', 'Skid base plate', 'Motor body',
                     'Gearbox', 'Fan cowl', 'Drive shaft'):
            self.assertIn(name, names)
        # The category reference's centre manway must not be copied onto this unit.
        self.assertFalse(any('manway' in name.lower() for name in names))
        structural = self.record['structural_checks']
        self.assertTrue(structural['tank_body'])
        self.assertEqual(structural['seam_bands'], 3)
        self.assertEqual(structural['legs'], 3)
        self.assertEqual(structural['foot_stems'], 3)
        self.assertEqual(structural['foot_collars'], 3)
        self.assertEqual(structural['foot_pads'], 3)
        self.assertEqual(structural['top_port_parts'], 15)
        self.assertEqual(structural['top_port_swing_bolts'], 4)
        self.assertEqual(structural['top_port_nuts'], 4)
        self.assertEqual(structural['lift_lug_parts'], 5)
        self.assertEqual(structural['pipe_top_bolts'], 4)
        self.assertEqual(structural['pipe_wall_bolts'], 4)
        self.assertEqual(structural['pipe_valve_flanges'], 2)
        self.assertEqual(structural['sight_glass_parts'], 5)
        self.assertEqual(structural['cone_flange_bolts'], 20)
        self.assertEqual(structural['cone_clamp_lugs'], 2)
        self.assertEqual(structural['cone_clamp_pins'], 2)
        self.assertEqual(structural['skid_rails'], 2)
        self.assertEqual(structural['skid_cross_rails'], 2)
        self.assertEqual(structural['skid_posts'], 2)
        self.assertEqual(structural['skid_columns'], 2)
        self.assertEqual(structural['skid_column_flanges'], 4)
        self.assertEqual(structural['drive_shaft_parts'], 3)
        self.assertEqual(structural['motor_fins'], 12)
        self.assertEqual(structural['gearbox_ribs'], 6)
        self.assertEqual(structural['fan_grille_bars'], 8)
        self.assertEqual(structural['motor_feet'], 2)
        openings = self.record['isolated_opening_checks']
        self.assertEqual(openings['top_port']['detected_openings'], 1)
        self.assertEqual(openings['lift_lug']['detected_openings'], 1)
        self.assertEqual(openings['drain']['detected_openings'], 1)
        self.assertTrue(all(row['passed'] for row in openings.values()))
        self.assertTrue(self.record['checks']['mesh_integrity'])
        self.assertTrue(self.record['checks']['isolated_openings'])
        self.assertTrue(self.record['checks']['structural_counts'])

    def test_real_bores_by_ray_casting(self):
        parts = load_parts(OUT/'mesh.glb')
        lug_names = [name for name in parts if name.startswith('Lift lug')]
        self.assertEqual(len(lug_names), 5)
        # A ray along the eye axis through the hole centre must hit nothing.
        lug_hits = first_hit(parts, lug_names, (-.12, .005, .411), (1, 0, 0))
        self.assertIsNone(lug_hits)
        # The drain bore is open: the centre ray passes through to the closed
        # lower-cone tip, while an offset ray hits the drain ring sooner.
        drain_names = [name for name in parts
                       if name.startswith('Drain') or name == 'Lower cone']
        centre = first_hit(parts, drain_names, (0, 0, -.46), (0, 0, 1))
        offset = first_hit(parts, drain_names, (.0105, 0, -.46), (0, 0, 1))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(offset)
        self.assertGreater(centre, offset + .005)
        self.assertGreater(centre, -.405)

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
        checks = self.record['material_texture_checks']
        self.assertTrue(checks['passed'])
        self.assertGreater(checks['bright_stainless']['metallic_mean'], .90)
        self.assertLess(checks['bright_stainless']['roughness_mean'], .35)
        self.assertGreater(checks['stained_steel']['roughness_mean'], .70)
        self.assertLess(checks['stained_steel']['metallic_mean'], .60)
        paint = self.record['paint_texture_checks']
        self.assertTrue(paint['passed'])
        self.assertGreater(paint['blue_minus_red'], 20)
        self.assertLess(paint['metallic_mean'], .30)
        self.assertTrue(self.record['checks']['material_tiles'])
        self.assertTrue(self.record['checks']['metal_reflection'])
        self.assertTrue(self.record['metal_reflection_check']['changed'])
        self.assertTrue((OUT/'model.blend').is_file())
        for name in ('base_color.png', 'metallic_roughness.png', 'normal.png',
                     'paint_base.png', 'paint_mr.png', 'paint_normal.png'):
            self.assertTrue((OUT/name).is_file(), name)
        for board in ('六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                      '材质-同光照对照.jpg', '材质-变光检查.jpg',
                      '金属反射-变光对照.jpg', '转台-24方向.jpg'):
            self.assertTrue((OUT/board).is_file(), board)


if __name__ == '__main__':
    unittest.main()
