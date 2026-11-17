"""Offline tests for exact source material domains, with no Blender/API calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from display_batch.common import file_hash
from display_batch.reference_profiles import PROFILES, apply_reference_regions, atlas_mask


class ReferenceProfilesTests(unittest.TestCase):
    def test_grid_origin_and_protected_cells(self):
        top = atlas_mask(2, 'top', {0: 0, 3: 0}, 8)
        self.assertTrue(np.all(top[:4, :4, 0] == 255))
        self.assertTrue(np.all(top[4:, 4:, 0] == 255))
        self.assertFalse(top[:4, 4:].any())
        self.assertFalse(top[4:, :4].any())
        bottom = atlas_mask(4, 'bottom', {0: 1, 9: 1}, 16)
        self.assertTrue(np.all(bottom[12:16, :4, 1] == 255))
        self.assertTrue(np.all(bottom[4:8, 4:8, 1] == 255))

    def test_no_global_unknown_to_iron(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mesh.glb'; path.write_bytes(b'synthetic exact source')
            source = {'mesh': str(path), 'mesh_hash': file_hash(path)}
            original = {'unknown': {'eligible': False, 'metal_family': 'unknown', 'coverage': [0.]*4}}
            result, info = apply_reference_regions(source, original, folder)
            self.assertEqual(result, original)
            self.assertEqual(info['status'], 'no_exact_source_profile')
            self.assertIsNot(result, original)

    def test_stale_source_and_authoring_record_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mesh.glb'; path.write_bytes(b'source')
            asset = {'mesh': str(path), 'mesh_hash': 'wrong'}
            with self.assertRaisesRegex(ValueError, 'mesh hash'):
                apply_reference_regions(asset, {}, folder)
            digest = file_hash(path); asset['mesh_hash'] = digest
            with patch.dict(PROFILES, {digest: {'asset_id': 'test', 'request_sha256': 'missing'}}):
                with self.assertRaisesRegex(ValueError, 'authoring request'):
                    apply_reference_regions(asset, {}, folder)

    def test_label_protection_clears_old_mask_and_never_mutates_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mesh.glb'; path.write_bytes(b'label source')
            digest = file_hash(path)
            profile = {'asset_id': 'test', 'protect_all': True, 'reason': 'printed label', 'selectors': []}
            original = {'body': {'mask': 'old.png', 'mask_hash': 'old', 'eligible': True,
                                 'metal_family': 'ferrous', 'coverage': [1., 0., 0., 0.]}}
            with patch.dict(PROFILES, {digest: profile}):
                result, _ = apply_reference_regions({'mesh': str(path), 'mesh_hash': digest}, original, folder)
            self.assertEqual(original['body']['mask'], 'old.png')
            self.assertFalse(result['body']['eligible'])
            self.assertTrue(result['body']['protected'])
            self.assertNotIn('mask', result['body'])
            self.assertEqual(result['body']['coverage'], [0.]*4)

    def test_spring_coating_does_not_authorize_iron_or_oxide(self):
        profile = next(v for v in PROFILES.values() if v['asset_id'] == '27')
        self.assertFalse(profile['atlas']['authored_atlas'])
        self.assertEqual(profile['atlas']['metal_family'], 'unknown')
        self.assertEqual(profile['atlas']['channels'], {1: 1})
        for protected in (0, 2, 3, 4, 5, 6, 8, 9, 10):
            self.assertNotIn(protected, profile['atlas']['channels'])


if __name__ == '__main__':
    unittest.main()
