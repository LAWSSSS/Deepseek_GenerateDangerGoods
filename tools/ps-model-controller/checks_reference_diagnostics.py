"""Small offline guards for the reference-material diagnostic entry point."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from diagnose_reference_materials import binding_rows, inspection_pose, physical_length, pixel_comparison, selected_bindings
from build_reference_library import select_references


class ReferenceDiagnosticsTests(unittest.TestCase):
    def test_exact_identity_binding_and_no_filename_mapping(self):
        catalog = {'25': {'mesh_hash': 'blue'}, '31': {'mesh_hash': 'yellow'}}
        library = {'version': 'reference-material-library-v1', 'assets': {
            'blue': {'asset_id': '25', 'evidence': {'candidate_asset_ids': ['25'],
                                                  'style_only_asset_ids': ['31']}}}}
        self.assertEqual([r[0] for r in binding_rows(library, catalog)], ['25'])
        library['assets'] = {'yellow': {'asset_id': '31', 'evidence': {
            'candidate_asset_ids': ['25'], 'style_only_asset_ids': ['31']}}}
        with self.assertRaisesRegex(ValueError, 'only a style target'):
            binding_rows(library, catalog)

    def test_excluded_unknown_and_stale_bindings_rejected(self):
        for aid, sha, evidence in [('5', 'same', {}), ('25', 'wrong', {}),
                                   ('25', 'same', {'mapping_status': 'class_style_only'})]:
            with self.subTest(aid=aid, sha=sha, evidence=evidence):
                library = {'version': 'reference-material-library-v1', 'assets': {
                    sha: {'asset_id': aid, 'evidence': evidence}}}
                with self.assertRaises(ValueError):
                    binding_rows(library, {'5': {'mesh_hash': 'same'}, '25': {'mesh_hash': 'same'}})

    def test_pilot_selection_cannot_import_style_only_asset(self):
        catalog = {'25': {'mesh_hash': 'blue'}, '33': {'mesh_hash': 'red'}}
        library = {'version': 'reference-material-library-v1', 'assets': {
            'blue': {'asset_id': '25'}, 'red': {'asset_id': '33'}}}
        self.assertEqual([row[0] for row in selected_bindings(library, catalog, ['33'])], ['33'])
        with self.assertRaisesRegex(ValueError, 'lacks an identity binding'):
            selected_bindings(library, catalog, ['31'])

    def test_build_identity_uses_candidate_id_not_other_asset_references(self):
        catalog = {'25': {'mesh_hash': 'blue'}, '31': {'mesh_hash': 'yellow'}}
        row = {'mapping_status': 'identity_style_reference', 'candidate_asset_ids': ['25'],
               'style_only_asset_ids': ['31'], 'asset_references': [
                   {'asset_id': '31', 'mesh_sha256': 'yellow'}, {'asset_id': '25', 'mesh_sha256': 'blue'}]}
        self.assertEqual(list(select_references({'mappings': [row]}, catalog)), ['25'])
        row['candidate_asset_ids'] = ['31']
        row['asset_references'][0]['mesh_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'different source mesh'):
            select_references({'mappings': [row]}, catalog)

    def test_inspection_pose_preserves_shape_and_never_claims_physics(self):
        vertices = np.array([[x, y, z] for x in (-.1, .1) for y in (-2., 2.) for z in (-.15, .15)])
        pose = inspection_pose(vertices)
        rotation = np.asarray(pose['matrix'])[:3, :3]
        self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(3)))
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        self.assertEqual(pose['physics_status'], 'not_evaluated')
        self.assertTrue(pose['inspection_only'])
        a = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2)
        points = vertices @ rotation.T
        b = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        self.assertTrue(np.allclose(a, b))
        self.assertGreater(np.ptp(points[:, 0]), np.ptp(points[:, 1]))

    def test_size_estimate_explicit_and_no_historical_multiplier(self):
        asset = {'id': '31', 'semantic': {'major_mm': [500, 2000]}}
        size, basis = physical_length(asset, {'user_scale': .375})
        self.assertEqual(size, 1.)
        self.assertIn('estimate', basis)
        self.assertEqual(physical_length(asset, {'base_pose': {'length_m': .6}})[0], .6)

    def test_pixel_metrics_capture_tiny_changes_without_quality_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            a = np.zeros((16, 16, 4), np.uint8)
            a[4:12, 4:12] = [80, 90, 100, 255]
            b = a.copy()
            b[7, 7, 0] += 1
            first, second = Path(folder)/'a.png', Path(folder)/'b.png'
            Image.fromarray(a).save(first)
            Image.fromarray(b).save(second)
            result = pixel_comparison(first, second)
            self.assertTrue(result['alpha_identical'])
            self.assertEqual(result['alpha_iou'], 1.)
            self.assertEqual(result['changed_pixels'], 1)
            self.assertEqual(result['max_abs_rgba_255'], 1)
            self.assertAlmostEqual(result['core_rgb_mae_255'], 1/(64*3))
            self.assertIn('not_visual_quality_score', result['basis'])


if __name__ == '__main__':
    unittest.main()
