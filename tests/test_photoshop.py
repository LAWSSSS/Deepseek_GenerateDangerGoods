"""Photoshop adapter contract tests use mocks; one optional real-app smoke test."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scrap_composite.photoshop import (PhotoshopAdapter, PhotoshopExportRequest, availability,
                                      file_hash, load_request, validate_actions, read_adjustments,
                                      review_binding, validate_review_binding)


class FakePhotoshopSession:
    """Exercise export/guard plumbing, not Photoshop's actual colour processing."""
    def __init__(self, sabotage=None):
        self.calls = []
        self.sabotage = sabotage

    async def call_tool(self, name, arguments):
        assert name == "photoshop_execute_jsx"
        script = arguments["script"]
        prefix, _, template = script.partition(";\n")
        job = json.loads(prefix[len("var JOB="):])
        assert "function exportObject" in template
        self.calls.append(job)
        base = Image.open(job["background"]).convert("RGBA")
        obj = Image.open(job["object_rgba"]).convert("RGBA")
        shadow = Image.open(job["shadow_rgba"]).convert("RGBA") if "shadow_rgba" in job else None
        front = Image.open(job["foreground_rgba"]).convert("RGBA") if "foreground_rgba" in job else None

        def composite(layer):
            value = base.copy()
            if shadow is not None:
                value = Image.alpha_composite(value, shadow)
            value = Image.alpha_composite(value, layer)
            if front is not None:
                value = Image.alpha_composite(value, front)
            return value.convert("RGB")

        outputs = job["outputs"]
        baseline = composite(obj)
        baseline.save(outputs["baseline"])
        changed = np.asarray(obj).copy()
        values = changed[..., :3].astype(np.float32)
        for action, amount in job["actions"].items():
            before_action = values.copy()
            if action == "brightness":
                values += amount
            elif action == "warmth":
                values += [amount, 0, -amount]
            elif action == "saturation":
                neutral = values.mean(axis=2, keepdims=True)
                values = neutral + (values - neutral) * (1 + amount / 100)
            elif action in ('black_point','highlights'):
                values += amount*.1
            elif action == 'tint':
                values += [0,-amount,0]
            if action in job.get('action_masks',{}):
                mask = np.asarray(Image.open(job['action_masks'][action]))[...,3] > 0
                values[~mask] = before_action[~mask]
        changed[..., :3] = np.uint8(np.clip(values, 0, 255))
        actual = composite(Image.fromarray(changed))
        if self.sabotage == "outside":
            actual.putpixel((0, 0), (255, 0, 0))
        elif self.sabotage == "noop":
            actual = baseline
        actual.save(outputs["png"])
        actual.save(outputs["reopened"])
        if self.sabotage == "readback":
            actual.putpixel((4, 4), (255, 0, 0))
            actual.save(outputs["reopened"])
        if self.sabotage == "alpha":
            changed[18, 22, 3] = 0
        Image.fromarray(changed).save(outputs["object_alpha"])
        Path(outputs["psd"]).write_bytes(b"mock PSD; not a real Photoshop file")
        layers = []
        if front is not None:
            layers.append({"name": "Foreground and protected | original pixels", "clipped": False, "visible": True})
        layers += [{"name": "Object adjustment | " + name, "clipped": True, "visible": True}
                   for name in reversed(job["actions"])]
        layers.append({"name": "Object | " + job["object_id"], "clipped": False, "visible": True})
        if shadow is not None:
            layers.append({"name": "Shadow | verified RGBA", "clipped": False, "visible": True})
        layers.append({"name": "Background | source pixels", "clipped": False, "visible": True})
        if self.sabotage == "layer_order":
            layers.reverse()
        metadata = {"canvas": job["canvas"], "applied": len(job["actions"]), "layers": layers,
                    "mode": "RGB8", "reopened": True, "color_profile": "mock sRGB"}
        for index, layer in enumerate(layers):
            layer.update(id=index+1, opacity=100, blend_mode='BlendMode.NORMAL')
            if not layer['name'].startswith('Object adjustment | '):
                continue
            name = layer['name'].split(' | ')[1]; value = job['actions'][name]
            if name == 'saturation':
                layer['adjustment'] = [{'colorize':False,'adjustment':[{'hue':0,'saturation':value,'lightness':0}]}]
                continue
            channels = {'red':value,'blue':-value} if name == 'warmth' else {'green':-value} if name == 'tint' else {'composite':value}
            curves = []
            for channel,delta in channels.items():
                points = [(0,0),(48,48+delta*.48),(128,128+delta),(208,208+delta*.48),(255,255)]
                if name == 'black_point': points = [(0,0),(48,48+delta),(128,128),(208,208),(255,255)]
                elif name == 'highlights': points = [(0,0),(48,48),(128,128),(208,208+delta),(255,255)]
                curves.append({'channel':{'enumerated':channel},'curve':[{'horizontal':x,'vertical':y} for x,y in points]})
            layer['adjustment'] = [{'adjustment':curves}]
        metadata['state'] = {'version':'photoshop-state-v2','document':{'id':1},'layers':layers}
        if self.sabotage == 'readback_values':
            layers[0]['opacity'] = 13
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps({
            "success": True, "result": json.dumps(metadata)}))])


class PhotoshopAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scrap-ps-contract-")
        self.root = Path(self.temp.name)
        yy, xx = np.indices((48, 64))
        bg = np.stack((55 + xx, 50 + yy, 65 + xx // 2), axis=2).astype(np.uint8)
        obj = np.zeros((48, 64, 4), np.uint8)
        obj[12:38, 16:51] = [145, 85, 55, 255]
        obj[21:29, 28:36] = 0  # Transparent structural hole.
        obj[12, 16:51, 3] = 128  # Fractional outline coverage must stay intact.
        shadow = np.zeros_like(obj)
        shadow[35:41, 14:54] = [15, 15, 15, 35]
        fg = np.zeros((48, 64), np.uint8); fg[:, 43:46] = 255
        protected = np.zeros_like(fg); protected[:4] = 255
        for name, pixels in (("background", bg), ("object", obj), ("shadow", shadow),
                             ("foreground", fg), ("protected", protected)):
            Image.fromarray(pixels).save(self.root / (name + ".png"))
        self.request = PhotoshopExportRequest(background=self.root/"background.png",
            object_rgba=self.root/"object.png", object_id="valve", shadow_rgba=self.root/"shadow.png",
            foreground_mask=self.root/"foreground.png", protected_mask=self.root/"protected.png")

    def tearDown(self):
        self.temp.cleanup()

    async def test_absolute_adjustments_preserve_background_holes_alpha_and_psd_readback(self):
        session = FakePhotoshopSession()
        adapter = PhotoshopAdapter(session=session)
        request = replace(self.request, actions={"brightness": 10, "warmth": -3, "saturation": -15})
        a = await adapter.export(request, self.root/"first")
        b = await adapter.export(request, self.root/"second")
        self.assertEqual(a["status"], "exported_verified")
        self.assertTrue(all(a["validation"]["checks"].values()))
        self.assertGreater(a["validation"]["changed_object_pixels"], 0)
        self.assertEqual(a["outputs"]["png"]["sha256"], b["outputs"]["png"]["sha256"])
        self.assertEqual(a["new_model_requests"], 0)
        self.assertEqual(a["visual_acceptance"], "not_evaluated")
        self.assertEqual(len(session.calls), 2)

    async def test_zero_operations_match_independent_expected_png(self):
        adapter = PhotoshopAdapter(session=FakePhotoshopSession())
        baseline = await adapter.export(self.request, self.root/"baseline")
        request = replace(self.request, expected_png=Path(baseline["outputs"]["png"]["path"]))
        result = await adapter.export(request, self.root/"verified")
        self.assertEqual(result["validation"]["expected_max_rgb_error"], 0)
        self.assertEqual(result["validation"]["changed_object_pixels"], 0)

    async def test_nonzero_action_is_checked_against_actual_ps_baseline_not_prediction(self):
        bogus = self.root/"prediction.png"
        Image.new("RGB", (64, 48), "red").save(bogus)
        request = replace(self.request, actions={"brightness": 5}, expected_png=bogus)
        result = await PhotoshopAdapter(session=FakePhotoshopSession()).export(request, self.root/"actual")
        self.assertIsNone(result["validation"]["expected_max_rgb_error"])
        self.assertEqual(result["validation"]["basis"], "actual_PS_PNG_and_reopened_PSD_not_prediction")

    async def test_validation_failures_never_publish_png_or_psd(self):
        for mode in ("outside", "alpha", "readback", "layer_order", "noop"):
            with self.subTest(mode=mode):
                out = self.root/mode
                with self.assertRaisesRegex(ValueError, "photoshop_validation_failed"):
                    await PhotoshopAdapter(session=FakePhotoshopSession(mode)).export(
                        replace(self.request, actions={"brightness": 5}), out)
                self.assertFalse((out/"final.png").exists())
                self.assertFalse((out/"final.psd").exists())
                self.assertEqual(json.loads((out/"result.json").read_text())["status"], "failed")

    async def test_source_hash_and_rgba_canvas_are_enforced_before_execution(self):
        session = FakePhotoshopSession()
        with self.assertRaisesRegex(ValueError, "source_hash"):
            await PhotoshopAdapter(session=session).export(
                replace(self.request, source_hashes={"object_rgba": "0"*64}), self.root/"bad_hash")
        tiny = self.root/"tiny.png"; Image.new("RGBA", (8, 8)).save(tiny)
        with self.assertRaisesRegex(ValueError, "full_canvas"):
            await PhotoshopAdapter(session=session).export(replace(self.request, object_rgba=tiny), self.root/"bad_size")
        self.assertEqual(session.calls, [])

    async def test_output_directory_cannot_overwrite_previous_attempt(self):
        adapter = PhotoshopAdapter(session=FakePhotoshopSession())
        await adapter.export(self.request, self.root/"once")
        with self.assertRaisesRegex(ValueError, "new_or_empty"):
            await adapter.export(self.request, self.root/"once")

    async def test_readback_receipt_binds_image_state_and_prevents_stale_edits(self):
        adapter = PhotoshopAdapter(session=FakePhotoshopSession())
        first = await adapter.export(replace(self.request, actions={'brightness':5}), self.root/'bound')
        binding = review_binding(first)
        validate_review_binding(binding)
        self.assertEqual(first['receipt']['absolute_parameters']['brightness'],5)
        self.assertEqual(first['local_measurements']['physical_illumination'],'unknown')
        second = await adapter.export(replace(self.request, actions={'brightness':10},review_binding=binding),self.root/'next')
        self.assertEqual(second['receipt']['parameter_delta']['brightness'],5)
        self.assertGreater(second['receipt']['changed_pixels_vs_reviewed_image'],0)
        Path(binding['image_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'stale_review_image'):
            await adapter.export(replace(self.request,actions={'brightness':12},review_binding=binding),self.root/'stale')

    async def test_actual_adjustment_values_are_required_not_echoed_requests(self):
        session = FakePhotoshopSession()
        exported = await PhotoshopAdapter(session=session).export(replace(self.request,actions={'brightness':5}),self.root/'read')
        metadata = exported['photoshop']
        layer = next(l for l in metadata['state']['layers'] if l['name']=='Object adjustment | brightness')
        layer['adjustment'][0]['adjustment'][0]['curve'][2]['vertical'] = 180
        with self.assertRaisesRegex(ValueError,'curve_readback_mismatch'):
            read_adjustments(metadata,{'brightness':5})

    async def test_registered_material_mask_limits_saturation_to_named_part(self):
        part=np.zeros((48,64),np.uint8); part[12:38,16:26]=255
        Image.fromarray(part).save(self.root/'paint.png')
        request=replace(self.request,actions={'saturation':-20},action_masks={'saturation':self.root/'paint.png'},action_parts={'saturation':'paint_body'})
        result=await PhotoshopAdapter(session=FakePhotoshopSession()).export(request,self.root/'part')
        self.assertTrue(result['validation']['checks']['registered_part_masks'])
        self.assertEqual(result['receipt']['action_parts'],{'saturation':'paint_body'})
        self.assertLess(result['validation']['changed_object_pixels'],int((np.asarray(Image.open(self.request.object_rgba))[...,3]>0).sum()))
        with self.assertRaisesRegex(ValueError,'part_mask_registry'):
            await PhotoshopAdapter(session=FakePhotoshopSession()).export(replace(request,action_parts={}),self.root/'invalidpart')

    async def test_actual_photoshop_fixture_if_app_is_already_available(self):
        if os.environ.get("SCRAP_TEST_REAL_PHOTOSHOP") != "1":
            self.skipTest("Explicit real-Photoshop smoke test not enabled")
        state = availability()
        if not state["available"]:
            self.skipTest("Photoshop unavailable; mock checks do not claim real PSD validation")
        result = await PhotoshopAdapter().export(replace(self.request, actions={"brightness": 5}), self.root/"real")
        self.assertEqual(result["status"], "exported_verified")

    def test_only_compiled_numeric_whitelist_is_accepted(self):
        for actions in ({"script": "anything"}, {"blur_px": 2}, {"brightness": 49},
                        {"brightness": float("nan")}, {"warmth": True}, {"saturation": 1.5}):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                validate_actions(actions)
        self.assertEqual(validate_actions({"brightness": 0, "saturation": -40}), {"saturation": -40.})

    def test_registered_manifest_resolves_relative_files_and_rejects_unknown_fields(self):
        path = self.root/"manifest.json"
        data = {"background": "background.png", "object_rgba": "object.png", "actions": {"warmth": 2},
                "source_hashes": {"object_rgba": file_hash(self.request.object_rgba)}}
        path.write_text(json.dumps(data), encoding="utf8")
        request = load_request(path)
        self.assertEqual(request.background, self.request.background)
        data["jsx"] = "app.activeDocument.close()"
        path.write_text(json.dumps(data), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "invalid_photoshop_layer_manifest"):
            load_request(path)


if __name__ == "__main__":
    unittest.main()
