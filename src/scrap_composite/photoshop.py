"""Isolated Photoshop export of registered layers and bounded object adjustments.

No model text, filesystem paths supplied by a model, or arbitrary JSX is executed.
Each attempt starts from the original layers; the PSD is reopened before publication.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image


VERSION = "photoshop-object-export-v2"
LIMITS = {"brightness": (-48., 48.), "black_point": (-24., 24.), "highlights": (-24., 24.),
          "warmth": (-24., 24.), "tint": (-16., 16.), "saturation": (-40., 40.)}
UNITS = {name: "saturation_percent_point" if name == "saturation" else "curve_code_value_8bit"
         for name in LIMITS}
TEMPLATE = Path(__file__).with_name("photoshop_export.jsx")
STATE_TEMPLATE = Path(__file__).with_name("photoshop_readback.jsx")
INSPECT_TEMPLATE = Path(__file__).with_name("photoshop_inspect.jsx")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    pending.replace(path)


def validate_actions(actions: dict[str, float]) -> dict[str, float]:
    if not isinstance(actions, dict) or set(actions) - set(LIMITS):
        raise ValueError("unsupported_photoshop_object_action")
    result = {}
    for name in LIMITS:  # Deterministic curves, channel balance, then saturation.
        if name not in actions:
            continue
        value = actions[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("invalid_photoshop_action_value")
        low, high = LIMITS[name]
        if not low <= value <= high or (name == "saturation" and not float(value).is_integer()):
            raise ValueError("photoshop_action_outside_limits")
        if value:
            result[name] = float(value)
    return result


def read_adjustments(metadata: dict[str, Any], requested: dict[str, float]) -> dict[str, float]:
    """Decode actual AM values and compare the registered curve, including channel."""
    state = metadata.get("state", {})
    layers = state.get("layers", [])
    ids = [layer.get("id") for layer in layers]
    if (state.get("version") != "photoshop-state-v2" or not ids
            or any(type(value) is not int for value in ids) or len(ids) != len(set(ids))):
        raise ValueError("photoshop_missing_stable_layer_readback")
    actual = dict.fromkeys(LIMITS, 0.)
    for name, value in requested.items():
        rows = [layer for layer in layers if layer.get("name") == "Object adjustment | " + name]
        if len(rows) != 1:
            raise ValueError("photoshop_adjustment_readback_missing")
        layer = rows[0]
        if layer.get("opacity") != 100 or layer.get("blend_mode") != "BlendMode.NORMAL" or not layer.get("clipped"):
            raise ValueError("photoshop_adjustment_state_mismatch")
        try:
            body = layer["adjustment"][0]
            adjustments = body["adjustment"]
            if name == "saturation":
                if body.get("colorize") is not False or len(adjustments) != 1 or adjustments[0].get("hue") != 0 or adjustments[0].get("lightness") != 0:
                    raise ValueError("photoshop_saturation_readback_mismatch")
                actual[name] = float(adjustments[0]["saturation"])
                if actual[name] != value:
                    raise ValueError("photoshop_saturation_readback_mismatch")
                continue
            expected = {"composite": value}
            if name == "warmth":
                expected = {"red": value, "blue": -value}
            elif name == "tint":
                expected = {"green": -value}
            curves = {}
            for curve in adjustments:
                channel = curve["channel"]["enumerated"]
                if channel == "grain":  # Photoshop's charID Grn resolves to this string ID.
                    channel = "green"
                curves[channel] = {float(p["horizontal"]): float(p["vertical"]) for p in curve["curve"]}
            if set(curves) != set(expected):
                raise ValueError("photoshop_curve_channel_mismatch")
            for channel, delta in expected.items():
                points = curves[channel]
                expected_points = {0:0,48:48+delta*.48,128:128+delta,208:208+delta*.48,255:255}
                if name == "black_point":
                    expected_points = {0:0,48:48+delta,128:128,208:208,255:255}
                elif name == "highlights":
                    expected_points = {0:0,48:48,128:128,208:208+delta,255:255}
                # Photoshop stores 8-bit curve ordinates with integer quantization.
                if set(points) != set(expected_points) or any(abs(points[x]-y) >= 1.01 for x,y in expected_points.items()):
                    raise ValueError("photoshop_curve_readback_mismatch")
            point = 48 if name == "black_point" else 208 if name == "highlights" else 128
            channel = next(iter(expected))
            actual[name] = (curves[channel][point]-point) * (-1 if name == "tint" else 1)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("photoshop_adjustment_readback_unparseable") from error
    return actual


@dataclass(frozen=True)
class PhotoshopExportRequest:
    """All RGBA layers/masks have the full background canvas and fixed pixel origin.

    Foreground/protected masks are binary. Shadows have already been validated by
    the compositor, and are supplied as ordinary RGBA overlays, not multiply layers.
    Actions are absolute values relative to object_rgba, never the previous PSD.
    """
    background: Path
    object_rgba: Path
    object_id: str = "object"
    actions: dict[str, float] = field(default_factory=dict)
    shadow_rgba: Path | None = None
    foreground_mask: Path | None = None
    protected_mask: Path | None = None
    expected_png: Path | None = None
    geometry_signature: str | None = None
    source_hashes: dict[str, str] = field(default_factory=dict)
    review_binding: dict[str, str] = field(default_factory=dict)
    action_masks: dict[str, Path] = field(default_factory=dict)
    action_parts: dict[str, str] = field(default_factory=dict)


def validate_review_binding(binding: dict[str, str]) -> None:
    """Host-authored evidence paths; the model receives only hashes and layer IDs."""
    if not binding:
        return
    required = {"image_path", "image_sha256", "state_path", "state_sha256"}
    if set(binding) != required:
        raise ValueError("photoshop_invalid_review_binding")
    for key in ("image", "state"):
        if file_hash(Path(binding[key + "_path"])) != binding[key + "_sha256"]:
            raise ValueError("photoshop_stale_review_" + key)
    state = json.loads(Path(binding["state_path"]).read_text(encoding="utf8"))
    if state.get("image_sha256") != binding["image_sha256"]:
        raise ValueError("photoshop_review_image_state_mismatch")


def review_binding(exported: dict[str, Any]) -> dict[str, str]:
    image = exported["outputs"]["png"]
    state = exported["readback"]
    return {"image_path": image["path"], "image_sha256": image["sha256"],
            "state_path": state["path"], "state_sha256": state["sha256"]}


def load_request(path: str | Path) -> PhotoshopExportRequest:
    """Read a host-authored layer manifest; model output never enters this reader."""
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    names = set(PhotoshopExportRequest.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) - names:
        raise ValueError("invalid_photoshop_layer_manifest")
    for key in ("background", "object_rgba", "shadow_rgba", "foreground_mask", "protected_mask", "expected_png"):
        if data.get(key) is not None:
            data[key] = (path.parent / data[key]).resolve(strict=True)
    if 'action_masks' in data:
        data['action_masks'] = {key:(path.parent / value).resolve(strict=True) for key,value in data['action_masks'].items()}
    request = PhotoshopExportRequest(**data)
    validate_actions(request.actions)
    return request


def _image(path: Path, *, rgba: bool = False) -> np.ndarray:
    with Image.open(path) as image:
        if rgba and image.mode != "RGBA":
            raise ValueError("object_and_shadow_require_explicit_RGBA")
        if not rgba and image.mode not in ("RGB", "RGBA"):
            raise ValueError("background_requires_8bit_RGB")
        if not rgba and image.mode == "RGBA" and image.getextrema()[3] != (255, 255):
            raise ValueError("background_must_be_opaque")
        return np.asarray(image.convert("RGBA" if rgba else "RGB"), np.uint8).copy()


def _mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, bool)
    with Image.open(path) as image:
        if image.mode not in ("1", "L"):
            raise ValueError("protection_masks_require_binary_grayscale")
        value = np.asarray(image.convert("L"))
    if value.shape != shape or np.any((value != 0) & (value != 255)):
        raise ValueError("protection_masks_require_same_canvas_and_binary_values")
    return value == 255


def prepare_request(request: PhotoshopExportRequest, scratch: Path) -> dict[str, Any]:
    actions = validate_actions(request.actions)
    validate_review_binding(request.review_binding)
    if not isinstance(request.object_id, str) or not request.object_id or len(request.object_id) > 80:
        raise ValueError("invalid_object_identifier")
    if any(ord(char) < 32 for char in request.object_id):
        raise ValueError("invalid_object_identifier")
    inputs = {}
    for name in ("background", "object_rgba", "shadow_rgba", "foreground_mask", "protected_mask", "expected_png"):
        value = getattr(request, name)
        if value is not None:
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError("registered_input_is_not_a_file")
            inputs[name] = {"path": str(path), "sha256": file_hash(path)}
    if (set(request.action_masks) != set(request.action_parts) or set(request.action_masks) - set(LIMITS)
            or any(not isinstance(part,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',part) or part == 'whole'
                   for part in request.action_parts.values())):
        raise ValueError('photoshop_invalid_part_mask_registry')
    for name, path in request.action_masks.items():
        path = Path(path).resolve(strict=True)
        inputs['action_mask:'+name] = {'path':str(path),'sha256':file_hash(path)}
    if set(request.source_hashes) - set(inputs):
        raise ValueError("unknown_source_hash_identifier")
    for name, expected in request.source_hashes.items():
        if inputs[name]["sha256"] != expected:
            raise ValueError("registered_source_hash_changed")
    background = _image(Path(inputs["background"]["path"]))
    obj = _image(Path(inputs["object_rgba"]["path"]), rgba=True)
    shape = background.shape[:2]
    if obj.shape[:2] != shape:
        raise ValueError("photoshop_layers_must_share_full_canvas")
    foreground = _mask(request.foreground_mask, shape)
    protected = _mask(request.protected_mask, shape)
    blocked = foreground | protected
    obj[blocked] = 0
    if not np.any(obj[..., 3]):
        raise ValueError("no_visible_object_pixels")
    shadow = np.zeros_like(obj)
    if request.shadow_rgba is not None:
        shadow = _image(request.shadow_rgba, rgba=True)
        if shadow.shape != obj.shape:
            raise ValueError("photoshop_shadow_must_share_full_canvas")
        shadow[blocked] = 0
    if request.expected_png is not None and _image(request.expected_png).shape != background.shape:
        raise ValueError("expected_png_canvas_mismatch")
    layers = {"background": background, "object_rgba": obj}
    if request.shadow_rgba is not None:
        layers["shadow_rgba"] = shadow
    if blocked.any():
        restore = np.dstack((background, blocked.astype(np.uint8) * 255))
        layers["foreground_rgba"] = restore
    paths = {}
    for name, pixels in layers.items():
        path = scratch / (name + ".png")
        Image.fromarray(pixels).save(path)
        paths[name] = path.as_posix()
    action_masks = {}
    adjustment_region = np.zeros(shape,bool)
    for name in actions:
        if name in request.action_masks:
            mask = _mask(request.action_masks[name],shape) & (obj[...,3] > 0) & ~blocked
            if not mask.any():
                raise ValueError('photoshop_empty_part_mask')
            pixels = np.full((*shape,4),255,np.uint8)
            pixels[...,3] = mask.astype(np.uint8)*255
            path = scratch / ('part-mask-'+name+'.png')
            Image.fromarray(pixels).save(path)
            action_masks[name] = path.as_posix()
            adjustment_region |= mask
        else:
            adjustment_region |= obj[...,3] > 0
    manifest = {"version": VERSION, "object_id": request.object_id, "inputs": inputs,
                "actions": actions, "geometry_signature": request.geometry_signature,
                "template_sha256": file_hash(TEMPLATE), "readback_template_sha256": file_hash(STATE_TEMPLATE),
                "canvas": [shape[1], shape[0]], "action_units": {n: UNITS[n] for n in actions},
                "review_binding": request.review_binding,
                "action_parts": {n:request.action_parts.get(n,'whole') for n in actions},
                "action_semantics": "absolute_from_original_object_RGBA",
                "shadow_blend": "normal_prepared_RGBA"}
    return {"manifest": manifest, "paths": paths, "background": background,
            "object_alpha": obj[..., 3], "shadow_alpha": shadow[..., 3],
            "blocked": blocked, "actions": actions, "action_masks": action_masks,
            "adjustment_region": adjustment_region}


def verify_exports(prepared: dict[str, Any], request: PhotoshopExportRequest,
                   paths: dict[str, Path], metadata: dict[str, Any]) -> dict[str, Any]:
    """Verify the real Photoshop export, not a predicted adjustment image."""
    bg = prepared["background"]
    baseline = _image(paths["baseline"])
    final = _image(paths["png"])
    reopened = _image(paths["reopened"])
    alpha_image = _image(paths["object_alpha"], rgba=True)
    checks = {"canvas": all(a.shape == bg.shape for a in (baseline, final, reopened)),
              "alpha_canvas": alpha_image.shape[:2] == bg.shape[:2]}
    if not all(checks.values()):
        raise ValueError("photoshop_canvas_changed")
    alpha = prepared["object_alpha"]
    blocked = prepared["blocked"]
    source_guard = (alpha == 0) & (prepared["shadow_alpha"] == 0)
    checks.update(
        geometric_alpha=np.array_equal(alpha_image[..., 3], alpha),
        protected_and_foreground=np.array_equal(final[blocked], bg[blocked]),
        baseline_outside_layers=np.array_equal(baseline[source_guard], bg[source_guard]),
        only_object_adjusted=np.array_equal(final[alpha == 0], baseline[alpha == 0]),
        registered_part_masks=np.array_equal(final[~prepared['adjustment_region']],baseline[~prepared['adjustment_region']]),
        transparent_holes=np.array_equal(final[source_guard], bg[source_guard]),
        psd_readback=np.array_equal(final, reopened),
        applied_actions=metadata.get("applied") == len(prepared["actions"]),
        reported_canvas=metadata.get("canvas") == prepared["manifest"]["canvas"],
    )
    expected_names = ["Object | " + request.object_id, "Background | source pixels"]
    if "shadow_rgba" in prepared["paths"]:
        expected_names.append("Shadow | verified RGBA")
    if "foreground_rgba" in prepared["paths"]:
        expected_names.append("Foreground and protected | original pixels")
    expected_names += ["Object adjustment | " + name for name in prepared["actions"]]
    layers = metadata.get("layers", [])
    checks["layer_manifest"] = (len(layers) == len(expected_names)
                                and sorted(item.get("name", "") for item in layers) == sorted(expected_names))
    expected_order = (["Foreground and protected | original pixels"]
                      if "foreground_rgba" in prepared["paths"] else [])
    expected_order += ["Object adjustment | " + name for name in reversed(prepared["actions"])]
    expected_order += ["Object | " + request.object_id]
    if "shadow_rgba" in prepared["paths"]:
        expected_order.append("Shadow | verified RGBA")
    expected_order.append("Background | source pixels")
    checks["layer_order_and_visibility"] = ([item.get("name") for item in layers] == expected_order
                                            and all(item.get("visible") is True for item in layers))
    checks["adjustments_clipped"] = all(item.get("clipped") is True for item in layers
                                          if item.get("name", "").startswith("Object adjustment | "))
    checks["reopened_RGB8"] = metadata.get("mode") == "RGB8" and metadata.get("reopened") is True
    actual_actions = read_adjustments(metadata, prepared["actions"])
    checks["actual_adjustment_readback"] = True
    expected_error = None
    if not prepared["actions"]:
        checks["zero_action_matches_baseline"] = np.array_equal(final, baseline)
        if request.expected_png is not None:
            expected = _image(request.expected_png)
            expected_error = int(np.max(abs(final.astype(np.int16) - expected.astype(np.int16))))
            # Permit only one 8-bit quantization level for the independent compositor.
            checks["zero_action_matches_expected"] = expected_error <= 1
    changed = np.any(final != baseline, axis=2)
    if prepared["actions"]:
        checks["effective_object_change"] = bool(changed[alpha > 0].any())
    sources_unchanged = all(file_hash(Path(item["path"])) == item["sha256"]
                            for item in prepared["manifest"]["inputs"].values())
    checks["sources_unchanged"] = sources_unchanged
    failures = [name for name, okay in checks.items() if not okay]
    if failures:
        raise ValueError("photoshop_validation_failed:" + ",".join(failures))
    return {"status": "pass", "checks": checks, "changed_object_pixels": int(changed.sum()),
            "actual_parameters": actual_actions,
            "expected_max_rgb_error": expected_error,
            "basis": "actual_PS_PNG_and_reopened_PSD_not_prediction",
            "visual_acceptance": "not_evaluated", "physics_acceptance": "not_evaluated"}


def default_mcp() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    candidates = [root / "work/photoshop-mcp-env/Scripts/photoshop-mcp-server.exe",
                  Path(os.environ.get("LOCALAPPDATA", ".")) / "OpenAI/PhotoshopMCP/python/Scripts/photoshop-mcp-server.exe"]
    return next((p for p in candidates if p.is_file()), None)


def availability(mcp: str | Path | None = None) -> dict[str, Any]:
    """Read local availability only; never launch Photoshop or touch an old service."""
    command = Path(mcp) if mcp is not None else default_mcp()
    registered = False
    executable = None
    clsid = None
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "Photoshop.Application\\CLSID") as entry:
                clsid = winreg.QueryValueEx(entry, None)[0]
                registered = True
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "CLSID\\" + clsid + "\\LocalServer32") as entry:
                server = winreg.QueryValueEx(entry, None)[0]
            match = re.match(r'^\s*(?:"([^"\r\n]+\.exe)"|([^\r\n]+?\.exe))(?:\s|$)', server, re.I)
            if match:
                executable = Path(os.path.expandvars(match.group(1) or match.group(2)))
        except OSError:
            pass
    running = False
    if os.name == "nt":
        probe = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
            "[bool](Get-Process Photoshop -ErrorAction SilentlyContinue)"], capture_output=True, text=True,
            timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        running = probe.returncode == 0 and probe.stdout.strip().lower() == "true"
    installed = bool(executable and executable.is_file())
    # The installed MCP uses comtypes.CreateObject, which starts the registered
    # local Photoshop executable when needed. Availability itself remains read-only.
    ready = bool(command and command.is_file() and registered and (running or installed))
    return {"available": ready, "mcp_executable": str(command) if command else None,
            "photoshop_registered": registered, "photoshop_running": running,
            "photoshop_executable": str(executable) if executable else None,
            "photoshop_executable_exists": installed, "photoshop_clsid": clsid,
            "can_start_via_COM": bool(registered and installed),
            "started_processes": 0,
            "reason": None if ready else "Photoshop COM registration/executable or MCP executable is unavailable"}


@asynccontextmanager
async def _session(command: Path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    # The new bounded stdio client owns only this process. No existing service is stopped.
    with tempfile.TemporaryFile(mode="w+", encoding="utf8") as errors:
        async with stdio_client(StdioServerParameters(command=str(command)), errlog=errors) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await asyncio.wait_for(session.initialize(), timeout=30)
                yield session


class PhotoshopAdapter:
    def __init__(self, *, session=None, mcp: str | Path | None = None, timeout: float = 180.):
        self.session = session
        self.mcp = mcp
        self.timeout = timeout

    async def export(self, request: PhotoshopExportRequest, output_dir: str | Path) -> dict[str, Any]:
        started = time.monotonic()
        output = Path(output_dir).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("photoshop_output_directory_must_be_new_or_empty")
        if self.session is None:
            state = availability(self.mcp)
            if not state["available"]:
                raise RuntimeError("photoshop_unavailable")
        output.mkdir(parents=True, exist_ok=True)
        report = {"version": VERSION, "status": "failed", "new_model_requests": 0,
                  "visual_acceptance": "not_evaluated", "physics_acceptance": "not_evaluated"}
        try:
            with tempfile.TemporaryDirectory(prefix="ps-object-") as directory:
                scratch = Path(directory)
                prepared = prepare_request(request, scratch)
                _write_json(output / "manifest.json", prepared["manifest"])
                paths = {"png": scratch / "final.png", "psd": scratch / "final.psd",
                         "baseline": scratch / "baseline.png", "reopened": scratch / "reopened.png",
                         "object_alpha": scratch / "object-alpha.png"}
                payload = {**prepared["paths"], "canvas": prepared["manifest"]["canvas"],
                           "object_id": request.object_id, "actions": prepared["actions"],
                           "action_masks": prepared['action_masks'],
                           "outputs": {name: path.as_posix() for name, path in paths.items()}}
                script = "var JOB=" + json.dumps(payload, ensure_ascii=True, allow_nan=False) + ";\n" + STATE_TEMPLATE.read_text(encoding="utf8") + "\n" + TEMPLATE.read_text(encoding="utf8")
                async def execute(session):
                    response = await asyncio.wait_for(session.call_tool("photoshop_execute_jsx", {"script": script}), self.timeout)
                    if response.isError:
                        raise RuntimeError("photoshop_mcp_execution_failed")
                    message = json.loads(response.content[0].text)
                    if message.get("success") is not True:
                        raise RuntimeError("photoshop_fixed_template_failed")
                    return json.loads(message["result"])
                if self.session is not None:
                    metadata = await execute(self.session)
                else:
                    async with _session(Path(state["mcp_executable"])) as session:
                        metadata = await execute(session)
                if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
                    raise RuntimeError("photoshop_missing_export")
                validate_review_binding(request.review_binding)
                validation = verify_exports(prepared, request, paths, metadata)
                artifacts = {}
                for name, source in paths.items():
                    target = output / source.name
                    pending = target.with_suffix(target.suffix + ".tmp")
                    shutil.copyfile(source, pending)
                    pending.replace(target)
                    artifacts[name] = {"path": str(target), "sha256": file_hash(target)}
                report.update(status="exported_verified", validation=validation, outputs=artifacts,
                              photoshop=metadata, actions=prepared["actions"],
                              action_semantics="absolute_from_original_object_RGBA")
                state = {**metadata.get("state", {}), "image_sha256": artifacts["png"]["sha256"],
                         "psd_sha256": artifacts["psd"]["sha256"]}
                state_path = output / "photoshop-state.json"
                _write_json(state_path, state)
                report["readback"] = {"path": str(state_path), "sha256": file_hash(state_path)}
                from .photography_measurements import measure_export_regions, region_statistics
                report["local_measurements"] = measure_export_regions(
                    _image(paths["png"]), prepared["object_alpha"], blocked=prepared["blocked"],
                    shadow_alpha=prepared["shadow_alpha"])
                report['local_measurements']['parts'] = {
                    request.action_parts[name]: region_statistics(_image(paths['png']),
                        _mask(path,prepared['object_alpha'].shape) & (prepared['object_alpha'] >= 242) & ~prepared['blocked'])
                    for name,path in request.action_masks.items()}
                previous = {}
                if request.review_binding:
                    old_path = Path(request.review_binding["state_path"])
                    previous = json.loads(old_path.read_text(encoding="utf8")).get("applied_parameters", {})
                state["requested_parameters"] = prepared["actions"]
                state["applied_parameters"] = validation["actual_parameters"]
                state["action_parts"] = prepared['manifest']['action_parts']
                _write_json(state_path, state)
                report["readback"]["sha256"] = file_hash(state_path)
                report["receipt"] = {"version": "photoshop-receipt-v2", "previous_binding": request.review_binding,
                    "image_sha256": artifacts["png"]["sha256"], "state_sha256": report["readback"]["sha256"],
                    "requested_parameters": prepared["actions"], "absolute_parameters": validation["actual_parameters"], "units": UNITS,
                    "action_parts": prepared['manifest']['action_parts'],
                    "parameter_delta": {n: value - previous.get(n, 0.) for n, value in validation["actual_parameters"].items()},
                    "changed_pixels": validation["changed_object_pixels"], "basis": "actual_export_vs_original_layers",
                    "visual_status": "not_evaluated", "seconds": round(time.monotonic()-started, 3)}
                if request.review_binding:
                    prior_pixels = _image(Path(request.review_binding["image_path"]))
                    final_pixels = _image(paths["png"])
                    report["receipt"]["changed_pixels_vs_reviewed_image"] = int(np.any(prior_pixels != final_pixels,axis=2).sum())
        except Exception as exc:
            # No arbitrary exception or server body is copied into the ledger.
            report['status'] = 'failed'
            report["error_kind"] = type(exc).__name__
            report["error_code"] = str(exc).split(":", 1)[0] if str(exc).startswith("photoshop_") else "input_or_execution_failure"
            _write_json(output / "result.json", report)
            raise
        _write_json(output / "result.json", report)
        return report

    async def inspect(self, psd: str | Path, *, expected_sha256: str,
                      regions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Read a registered PSD and bounded pixels; never save changes to the source."""
        started = time.monotonic()
        path = Path(psd).resolve(strict=True)
        if path.suffix.lower() != ".psd" or file_hash(path) != expected_sha256:
            raise ValueError("photoshop_registered_PSD_changed")
        if regions is not None and (not isinstance(regions, list) or len(regions) > 12):
            raise ValueError("photoshop_too_many_measurement_regions")
        from .photography_measurements import measure_regions
        measure_regions(np.zeros((1,1,3),np.uint8),regions or [])  # Validate before launching Photoshop.
        with tempfile.TemporaryDirectory(prefix="ps-inspect-") as directory:
            preview = Path(directory) / "inspect.png"
            # Never open/close a user's already open PSD. Inspect an identical private copy.
            snapshot = Path(directory) / 'registered-copy.psd'
            shutil.copyfile(path,snapshot)
            if file_hash(snapshot) != expected_sha256:
                raise ValueError('photoshop_registered_PSD_changed')
            payload = {"psd": snapshot.as_posix(), "preview": preview.as_posix()}
            script = "var JOB=" + json.dumps(payload) + ";\n" + STATE_TEMPLATE.read_text(encoding="utf8") + "\n" + INSPECT_TEMPLATE.read_text(encoding="utf8")
            async def execute(session):
                response = await asyncio.wait_for(session.call_tool("photoshop_execute_jsx", {"script": script}), self.timeout)
                if response.isError:
                    raise RuntimeError("photoshop_inspection_failed")
                message = json.loads(response.content[0].text)
                if message.get("success") is not True:
                    raise RuntimeError("photoshop_inspection_failed")
                return json.loads(message["result"])
            if self.session is not None:
                state = await execute(self.session)
            else:
                available = availability(self.mcp)
                if not available["available"]:
                    raise RuntimeError("photoshop_unavailable")
                async with _session(Path(available["mcp_executable"])) as session:
                    state = await execute(session)
            pixels = _image(preview)
            measured = measure_regions(pixels, regions or [])
            if file_hash(path) != expected_sha256:
                raise ValueError("photoshop_inspection_modified_source")
            return {"state": state, "psd_sha256": expected_sha256,
                    "image_sha256": file_hash(preview), "measurements": measured,
                    "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "source_unchanged": True, "new_model_requests": 0,
                    "photoshop_calls": 1, "seconds": round(time.monotonic()-started,3)}
