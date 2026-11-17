"""Integration checks against the generated 02 DN200 ball-valve candidate."""
import json
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scene_v2 import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/02球阀-六视图重建-v1'
LIBRARY = ROOT/'outputs/三维模型库-圆润复验/results.jsonl'


class RebuildChecks02(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'result.json').is_file():
            raise unittest.SkipTest('02 review candidate has not been generated')
        cls.record = json.loads((OUT/'result.json').read_text(encoding='utf8'))
        cls.scene = trimesh.load(OUT/'mesh.glb', force='scene')
        cls.row = next(r for r in (json.loads(line) for line in
                                   LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
                       if r['number'] == 2)

    def test_manifest_hashes_and_review_only_state(self):
        self.assertEqual(self.record['mesh_sha256'], digest(OUT/'mesh.glb'))
        self.assertEqual(self.record['blend_sha256'], digest(OUT/'model.blend'))
        self.assertEqual(self.record['shape_basis'], 'user_six_view_reference')
        self.assertEqual(self.record['status'], 'review')
        self.assertEqual(self.record['model_review'], 'pending')
        self.assertEqual(self.record['scene_validation'], 'pending')
        self.assertTrue(self.record['manual_only'])
        self.assertEqual(self.record['version'], 'valve02-parametric-v1')
        self.assertEqual(self.record['number'], 2)
        self.assertIn('0.9287', self.record['original_structural_warning'])
        self.assertEqual(self.record['original_source_fidelity_record']['filled_holes'], 2)
        self.assertEqual(self.record['original_mesh_sha256'], self.row['mesh_sha256'])
        self.assertEqual(digest(Path(self.row['mesh'])), self.record['original_mesh_sha256'])

    def test_mesh_bounds_topology_and_no_self_intersections(self):
        self.assertTrue(np.allclose(self.scene.bounds, self.record['worker']['bounds'], atol=2e-6))
        self.assertEqual(self.record['mesh_audit']['open_parts'], [])
        self.assertFalse(self.record['mesh_audit']['nonfinite_parts'])
        self.assertGreaterEqual(self.record['mesh_audit']['parts'], 100)
        self.assertEqual(self.record['worker']['self_intersecting_components'], {})
        self.assertEqual(self.record['mesh_audit']['parts'],
                         self.record['mesh_audit']['closed_after_welding_seams'])

    def test_structural_parts_and_isolated_openings(self):
        names = {part['name'] for part in self.record['worker']['parts']}
        for name in ('Cast valve body', 'Body split flange', 'End flange plus', 'End flange minus',
                     'Ball', 'Handwheel rim', 'Foot base plus', 'Foot base minus',
                     'Cast label DN200', 'Cast label 25', 'Cast label WCB', 'Bottom boss'):
            self.assertIn(name, names)
        self.assertEqual(sum(n.startswith('Handwheel spoke ') for n in names), 3)
        self.assertEqual(sum(n.startswith('Split flange bolt head ') for n in names), 12)
        structural = self.record['structural_checks']
        self.assertEqual(structural['handwheel_spokes'], 3)
        self.assertEqual(structural['split_flange_bolts'], 12)
        self.assertEqual(structural['feet'], 2)
        self.assertTrue(structural['ball_present'])
        self.assertTrue(structural['cast_text_present'])
        openings = self.record['isolated_opening_checks']
        self.assertEqual(openings['wheel']['detected_openings'], 3)
        self.assertEqual(openings['end_flange']['detected_openings'], 9)
        self.assertEqual(openings['foot_plate']['detected_openings'], 2)
        self.assertTrue(all(row['passed'] for row in openings.values()))

    def test_pbr_textures_are_embedded_and_distinct(self):
        materials = []
        for mesh in self.scene.geometry.values():
            material = mesh.visual.material
            if material is None:
                continue
            materials.append(material)
        self.assertGreaterEqual(len(materials), 4)
        textured = [m for m in materials if getattr(m, 'baseColorTexture', None) is not None]
        self.assertTrue(textured)
        self.assertTrue(any(m is not None and getattr(m, 'metallicRoughnessTexture', None) is not None
                            and m.metallicRoughnessTexture.size == (2048, 2048) for m in materials))
        self.assertTrue(any(m is not None and getattr(m, 'normalTexture', None) is not None
                            and m.normalTexture.size == (2048, 2048) for m in materials))
        checks = self.record['material_texture_checks']
        self.assertTrue(checks['passed'])
        self.assertLess(checks['blue_paint']['metallic_mean'], .10)
        self.assertGreater(checks['cast']['metallic_mean'], .80)
        self.assertLess(checks['machined']['roughness_mean'], checks['cast']['roughness_mean'])
        self.assertGreater(checks['rust']['roughness_mean'], .70)
        self.assertTrue((OUT/'model.blend').is_file())

    def test_six_views_material_and_light_changes(self):
        for view in ('front', 'back', 'left', 'right', 'top', 'bottom'):
            self.assertTrue((OUT/'renders'/f'new-{view}.png').is_file())
            self.assertTrue((OUT/'renders'/f'old-{view}.png').is_file())
        self.assertEqual(len(list((OUT/'renders').glob('turn-*.png'))), 24)
        self.assertTrue((OUT/'六视图-新旧对照.jpg').is_file())
        self.assertTrue((OUT/'局部检查.jpg').is_file())
        self.assertTrue((OUT/'结构开口-隔离检查.jpg').is_file())
        self.assertTrue((OUT/'材质-同光照对照.jpg').is_file())
        self.assertTrue((OUT/'材质-变光检查.jpg').is_file())
        for row in self.record['reference_views']:
            self.assertIn('未标定', row['note'])
        first = np.asarray(Image.open(OUT/'renders/new-front.png').convert('RGBA'))
        second = np.asarray(Image.open(OUT/'renders/new-light-alternate.png').convert('RGBA'))
        a, b = first[:, :, 3] >= 128, second[:, :, 3] >= 128
        self.assertGreater(np.logical_and(a, b).sum()/np.logical_or(a, b).sum(), .999)
        self.assertGreater(np.abs(first[:, :, :3].astype(float) -
                                  second[:, :, :3].astype(float))[a].mean(), .5)
        old = np.asarray(Image.open(OUT/'renders/old-front.png').convert('RGBA'))
        new = np.asarray(Image.open(OUT/'renders/new-front.png').convert('RGBA'))
        self.assertGreater(np.abs(old[:, :, :3].astype(float) -
                                  new[:, :, :3].astype(float))[new[:, :, 3] >= 128].mean(), 1)

    def test_source_fidelity_is_explicitly_diagnostic(self):
        fidelity = self.record['source_fidelity_diagnostic']
        self.assertFalse(fidelity['passed'])
        self.assertIn('诊断', self.record['source_diagnostic_note'])
        self.assertIn('未标定', ' '.join(self.record['limitations']))
        self.assertTrue(self.record['pending_review'])

    def test_candidate_is_independent_of_the_01_output(self):
        self.assertNotIn('01阀门', str(OUT))
        self.assertTrue((ROOT/'outputs/01阀门-六视图重建-v7/result.json').is_file())
        self.assertTrue((OUT/'hashes.json').is_file())
        hashes = json.loads((OUT/'hashes.json').read_text(encoding='utf8'))
        for name in ('mesh.glb', 'model.blend', 'base_color.png',
                     'metallic_roughness.png', 'normal.png'):
            self.assertIn(name, hashes['files'])
            self.assertEqual(hashes['files'][name], digest(OUT/name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
