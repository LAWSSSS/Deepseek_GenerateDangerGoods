"""Offline checks for support-only screening helpers and gates."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mesh_checks import placement_gate, support_screening_gate
from mesh_placement import open_fit
from mesh_screening import (DEFAULT_TARGETS, candidate_checks, load_revision_rows,
                            parse_targets, scene_summary, summary_payload)
from scene_v2 import NeedsReview


class ScreeningLoaderChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.outputs = self.root / "outputs"
        self.item = self.outputs / "10-revision"
        self.item.mkdir(parents=True)
        self.mesh = self.item / "mesh.glb"
        self.mesh.write_bytes(b"proxy-mesh")
        self.rust = self.item / "rust.png"
        self.rust.write_bytes(b"proxy-rust")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 1,
            "status": "revision_pending_review",
            "items": [{
                "id": "10", "number": 10, "name": "千斤顶（无独立操作杆）",
                "quantity": 1, "mesh": str(self.mesh),
                "mesh_sha256": hashlib.sha256(self.mesh.read_bytes()).hexdigest(),
                "rust": str(self.rust),
            }],
        }), encoding="utf8")
        self.library = self.root / "library.jsonl"
        self.library.write_text(json.dumps({
            "number": 10, "asset_id": "asset-base-10", "file": "千斤顶.jpg",
            "manual_only": False, "reconstruction": {"fidelity": {"passed": True}},
        }) + "\n", encoding="utf8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_selected_revision_keeps_unknown_source_evidence(self):
        rows = load_revision_rows(self.manifest, ["10"], self.library, root=self.root)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["asset_id"], "asset-base-10-feedback-10")
        self.assertIsNone(row["source_view"])
        self.assertFalse(row["source_fidelity"]["passed"])
        self.assertEqual(row["source_fidelity"]["status"], "unknown")
        self.assertEqual(row["mesh"], str(self.mesh.resolve()))
        self.assertTrue(row["manual_only"])

    def test_missing_revision_is_rejected(self):
        with self.assertRaises(NeedsReview):
            load_revision_rows(self.manifest, ["10", "13"], self.library, root=self.root)

    def test_quantity_other_than_one_is_rejected(self):
        data = json.loads(self.manifest.read_text(encoding="utf8"))
        data["items"][0]["quantity"] = 2
        self.manifest.write_text(json.dumps(data), encoding="utf8")
        with self.assertRaises(NeedsReview):
            load_revision_rows(self.manifest, ["10"], self.library, root=self.root)

    def test_mesh_hash_mismatch_is_rejected(self):
        data = json.loads(self.manifest.read_text(encoding="utf8"))
        data["items"][0]["mesh_sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(data), encoding="utf8")
        with self.assertRaises(NeedsReview):
            load_revision_rows(self.manifest, ["10"], self.library, root=self.root)

    def test_mesh_outside_outputs_is_rejected(self):
        outside = self.root / "outside.glb"
        outside.write_bytes(b"outside")
        data = json.loads(self.manifest.read_text(encoding="utf8"))
        data["items"][0]["mesh"] = str(outside)
        data["items"][0]["mesh_sha256"] = hashlib.sha256(b"outside").hexdigest()
        self.manifest.write_text(json.dumps(data), encoding="utf8")
        with self.assertRaises(NeedsReview):
            load_revision_rows(self.manifest, ["10"], self.library, root=self.root)


class ScreeningGateChecks(unittest.TestCase):
    def test_support_gate_requires_physics_and_contacts_only(self):
        good = {"physics_stable": True, "component_support": [1.0, .95]}
        self.assertEqual(support_screening_gate(good), [])
        self.assertEqual(support_screening_gate({**good, "physics_stable": False}), ["unstable"])
        self.assertEqual(support_screening_gate({**good, "component_support": [.8]}), ["unsupported_part"])
        self.assertEqual(support_screening_gate({**good, "component_support": []}), ["unsupported_part"])

    def test_formal_gate_rejects_missing_visibility(self):
        base = {"physics_stable": True, "component_support": [1.0], "collision": False,
                "occlusion_order": "clear", "scene_scale": "calibrated"}
        self.assertEqual(placement_gate({**base, "source_visible_fraction": .9}), [])
        self.assertIn("unseen_surface", placement_gate({**base, "source_visible_fraction": None}))
        self.assertIn("unseen_surface", placement_gate(base))

    def test_candidate_checks_keep_unknowns(self):
        checks = candidate_checks({"physics_stable": True, "component_support": [.97]})
        self.assertEqual(checks["physics"], "pass")
        self.assertEqual(checks["component_support"], "pass")
        for key in ("source_visibility", "occlusion", "scene_scale"):
            self.assertEqual(checks[key], "unknown")
        self.assertEqual(checks["photoshop"], "not_run")


class ScreeningFitChecks(unittest.TestCase):
    def test_screening_fit_uses_unknown_source_visibility(self):
        class Atlas:
            def clearance_crop(self, support_id, crop):
                return np.full((300, 400), 100, dtype=np.float32)

            def native_position_context(self, *args, **kwargs):
                return {"cost": .2, "relief": {"kind": "flat"}}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "A").mkdir()
            support = np.zeros((300, 400), np.uint8)
            support[5:-5, 5:-5] = 255
            Image.fromarray(support).save(root / "A" / "support.png")
            Image.new("L", (400, 300)).save(root / "A" / "foreground.png")
            pixels = np.zeros((60, 120, 4), np.uint8)
            pixels[4:52, 4:116] = (110, 100, 90, 255)
            shadow = np.zeros((60, 120), np.float32)
            shadow[52:57, 6:114] = .2
            left = np.zeros((60, 120), bool)
            left[48:55, 10:22] = True
            right = np.zeros_like(left)
            right[48:55, 96:108] = True
            view = (pixels, shadow, [left, right])
            frame = Image.new("RGB", (800, 600), (93, 89, 82))
            candidate = {"id": "A", "placement_mode": "open", "target_px": 120,
                         "crop_px": [200, 120, 600, 420], "center_px": [400, 270],
                         "support_id": "S1"}
            fit = open_fit(candidate, root, Atlas(), frame, view,
                           source_visible_fraction=None, physics_stable=True,
                           pixel_size_review=True, screening=True)
            self.assertTrue(fit["evidence"]["screening_only"])
            self.assertIsNone(fit["evidence"]["source_visible_fraction"])
            self.assertEqual(fit["evidence"]["occlusion_order"], "unknown")
            self.assertEqual(fit["evidence"]["collision"], "unknown")
            with self.assertRaises(NeedsReview):
                open_fit(candidate, root, Atlas(), frame, view,
                         source_visible_fraction=None, physics_stable=True,
                         pixel_size_review=True)
            with self.assertRaises(ValueError):
                open_fit(candidate, root, Atlas(), frame, view,
                         source_visible_fraction=.9, physics_stable=True, screening=True)


class ScreeningReportChecks(unittest.TestCase):
    def test_default_targets_are_provisional(self):
        targets = parse_targets([], ["10", "13", "19-1", "31"])
        self.assertEqual(targets, DEFAULT_TARGETS)
        override = parse_targets(["10:300"], ["10"])
        self.assertEqual(override["10"], 300)
        with self.assertRaises(ValueError):
            parse_targets(["10"], ["10"])

    def test_summary_keeps_explicit_unknowns(self):
        class Answer:
            def model_dump(self, mode="json"):
                return {"zones": ["Z1"], "refs": [["R1", "beam", 100, 200]],
                        "protect": [[0, 0, 10, 10]]}

        scene = {"answer": Answer(), "cache_hit": True, "camera_elevation_deg": 45,
                 "camera_elevation_interval": [35, 55]}
        sample = {"id": "10", "source_fidelity": "unknown", "source_visibility": "unknown",
                  "occlusion": "unknown", "scene_scale": "unknown", "sites": [],
                  "candidates": [], "blocked_reason": "no_support_sites"}
        payload = summary_payload(status="support_screening_only", manifest_path=Path("manifest.json"),
                                  manifest_sha256="a" * 64, background_sha256="b" * 64,
                                  scene=scene, samples=[sample], notes=["support only"])
        self.assertEqual(payload["status"], "support_screening_only")
        self.assertNotIn("passed", json.dumps(payload))
        self.assertTrue(payload["samples"][0]["source_visibility"] == "unknown")
        summary = scene_summary(scene)
        self.assertTrue(summary["cache_hit"])
        self.assertEqual(summary["zones"], ["Z1"])


if __name__ == "__main__":
    unittest.main()
