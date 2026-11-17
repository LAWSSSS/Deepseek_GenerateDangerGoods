import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from display_batch.ai import ProviderUnavailable
from display_batch.rendering import part_keys
from display_batch.surface_maps import partition
from validate_display_scanned import compare, passes_gate


class ScannedMaterialTests(unittest.TestCase):
    def test_print_and_polymer_are_preserved_while_metal_and_paint_are_separated(self):
        rgb = np.float32([[[.54, .56, .58], [.61, .55, .09], [.09, .09, .09], [.91, .91, .91]]])
        metal = np.float32([[.85, .12, .04, .01]])
        roughness = np.float32([[.25, .59, .80, .79]])
        atlas = partition("Yellow paint and rubber PBR atl", rgb, metal, roughness, ["paint", "steel", "rubber"])
        np.testing.assert_array_equal(atlas[0, :, :3], [[255, 0, 0], [0, 255, 0], [0, 0, 0], [0, 0, 0]])
        print_atlas = partition("Steel and green print PBR atlas", rgb, metal, roughness, ["paint", "steel"])
        np.testing.assert_array_equal(print_atlas[0, :, 1:3], np.zeros((4, 2)))
        polymer = partition("Black polypropylene base", rgb, metal, roughness, ["plastic"])
        self.assertFalse(polymer[..., :3].any())

    def test_scans_affect_object_and_indirect_shadow_and_hdri_affects_all(self):
        request = {"windows": {"object": {}, "shadow": {}}, "camera": {}, "environment": {},
                   "appearance": "local", "plane_normal": [0, 0, 1], "light_gain": 1.,
                   "mesh_hash": "source", "matrix": [], "yaw": 0, "roughness_delta": 0.,
                   "dust_gain": 1., "dust_control": 1., "material_classes": ["steel"]}
        old = part_keys(request, "OPTIX", "sources")
        scanned = {**request, "surface_scans": {"signature": "scan-and-mask"}}
        new = part_keys(scanned, "OPTIX", "sources")
        self.assertNotEqual(old["object"], new["object"])
        self.assertNotEqual(old["with_shadow"], new["with_shadow"])
        self.assertEqual(old["without_shadow"], new["without_shadow"])
        hdri = part_keys({**scanned, "scanned_hdri": {"sha256": "hdr"}}, "OPTIX", "sources")
        self.assertTrue(all(hdri[key] != new[key] for key in new))

    def test_stage_gate_requires_program_checks_and_all_families(self):
        cases = [{"asset_id": str(i // 2), "variants": {"B": {"audit": {"pass": True}}},
                  "opinions": {"B": {"pass": i != 0, "serious": False}}} for i in range(8)]
        self.assertTrue(passes_gate(cases, "B", 8, 7))
        cases[0]["opinions"]["B"]["serious"] = True
        self.assertFalse(passes_gate(cases, "B", 8, 7))
        cases[0]["opinions"]["B"]["serious"] = False
        cases[0]["variants"]["B"]["audit"]["pass"] = False
        self.assertFalse(passes_gate(cases, "B", 8, 7))
        self.assertFalse(passes_gate(cases[:-1], "B", 8, 7))


class ScannedReviewTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, error):
        return (SimpleNamespace(ask=AsyncMock(side_effect=error)),
                {"variants": {"A": {"image_hash": "a"}, "B": {"image_hash": "b"}}, "label_map": {"A": "A", "B": "B"}},
                {"name": "object", "semantic": {"material_classes": ["steel"]}})

    async def test_terminal_format_error_is_retained_after_resume(self):
        service, case, asset = self.fixture(RuntimeError("invalid after retry"))
        with tempfile.TemporaryDirectory() as folder:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "invalid after retry"):
                    await compare(service, case, asset, Path(folder))
            service.ask.assert_awaited_once()

    async def test_global_unavailability_remains_resumable(self):
        service, case, asset = self.fixture(ProviderUnavailable("offline"))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ProviderUnavailable):
                await compare(service, case, asset, Path(folder))
            self.assertFalse(list(Path(folder).iterdir()))
