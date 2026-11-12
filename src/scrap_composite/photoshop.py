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
from typing import Any

import numpy as np
from PIL import Image


VERSION = "photoshop-object-export-v1"
LIMITS = {"brightness": (-48., 48.), "warmth": (-24., 24.), "saturation": (-40., 40.)}
TEMPLATE = Path(__file__).with_name("photoshop_export.jsx")


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
    manifest = {"version": VERSION, "object_id": request.object_id, "inputs": inputs,
                "actions": actions, "geometry_signature": request.geometry_signature,
                "template_sha256": file_hash(TEMPLATE), "canvas": [shape[1], shape[0]],
                "action_semantics": "absolute_from_original_object_RGBA",
                "shadow_blend": "normal_prepared_RGBA"}
    return {"manifest": manifest, "paths": paths, "background": background,
            "object_alpha": obj[..., 3], "shadow_alpha": shadow[..., 3],
            "blocked": blocked, "actions": actions}


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
                           "outputs": {name: path.as_posix() for name, path in paths.items()}}
                script = "var JOB=" + json.dumps(payload, ensure_ascii=True, allow_nan=False) + ";\n" + TEMPLATE.read_text(encoding="utf8")
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
        except Exception as exc:
            # No arbitrary exception or server body is copied into the ledger.
            report["error_kind"] = type(exc).__name__
            report["error_code"] = str(exc).split(":", 1)[0] if str(exc).startswith("photoshop_") else "input_or_execution_failure"
            _write_json(output / "result.json", report)
            raise
        _write_json(output / "result.json", report)
        return report
