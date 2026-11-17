"""Input and reporting helpers for support-only screening of revised models.

The four revised samples have no verified source camera or source alpha.  This
module therefore builds library-shaped rows that keep source visibility,
occlusion and scale explicitly unknown.  It never upgrades a revision to the
approved model library.
"""
import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from scene_v2 import NeedsReview

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAMPLE_IDS = ("10", "13", "19-1", "31")
DEFAULT_TARGETS = {"10": 296.0, "13": 371.0, "19-1": 243.0, "31": 475.0}
TARGET_BASIS = "exploratory_preview_measured_major_px"
MANIFEST_STATUSES = {"revision_pending_review", "revision_complete", "revision_approved"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def _inside_outputs(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise NeedsReview(f"修订文件不存在：{path}") from error
    base = (Path(root) / "outputs").resolve()
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise NeedsReview(f"修订文件不在 outputs 内或不是文件：{path}")
    return resolved


def load_base_rows(library_path: Path) -> dict[int, dict]:
    """Load the approved 39-item library rows by number."""
    rows = {}
    for line in Path(library_path).read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("manual_only"):
            continue
        number = int(row["number"])
        if number in rows:
            raise NeedsReview(f"模型库编号重复：{number}")
        rows[number] = row
    if not rows:
        raise NeedsReview("模型库为空")
    return rows


def load_revision_rows(manifest_path: Path, ids: Iterable[str], library_path: Path,
                       *, root: Path = ROOT) -> list[dict]:
    """Validate only the requested revision IDs and return library-shaped rows."""
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    except (OSError, ValueError) as error:
        raise NeedsReview(f"修订清单无法读取：{manifest_path}") from error
    if manifest.get("schema_version") != 1 or manifest.get("status") not in MANIFEST_STATUSES:
        raise NeedsReview("修订清单版本或状态不可用")
    items = {}
    for item in manifest.get("items", []):
        ident = str(item.get("id", ""))
        if not ident or ident in items:
            raise NeedsReview("修订清单包含空 ID 或重复 ID")
        items[ident] = item
    requested = [str(value) for value in ids]
    if not requested or len(set(requested)) != len(requested):
        raise NeedsReview("筛查 ID 为空或重复")
    base_rows = load_base_rows(library_path)
    result = []
    for ident in requested:
        item = items.get(ident)
        if item is None:
            raise NeedsReview(f"修订清单缺少：{ident}")
        if item.get("quantity") != 1:
            raise NeedsReview(f"修订数量不是单件：{ident}")
        number = int(item.get("number"))
        if number not in base_rows:
            raise NeedsReview(f"模型库没有对应的原编号：{ident}")
        mesh = _inside_outputs(Path(item.get("mesh", "")), root)
        rust = _inside_outputs(Path(item.get("rust", "")), root)
        expected = str(item.get("mesh_sha256", ""))
        if digest(mesh) != expected:
            raise NeedsReview(f"修订网格哈希不符：{ident}")
        base = copy.deepcopy(base_rows[number])
        fidelity = {"passed": False, "status": "unknown",
                    "reason": "修订模型没有可核验的原视角相机与透明蒙版"}
        base.update({
            "asset_id": f"{base['asset_id']}-feedback-{ident}",
            "revision_id": ident,
            "number": number,
            "file": item.get("name") or base.get("file"),
            "mesh": str(mesh),
            "mesh_sha256": expected,
            "rust": str(rust),
            "rust_sha256": digest(rust),
            "source_view": None,
            "source_view_sha256": None,
            "source_alpha_sha256": None,
            "physical_size": None,
            "manual_only": True,
            "model_review": "pending",
            "scene_validation": "pending",
            "version": "user-feedback-screening-v1",
            "shape_basis": "user_feedback_revision",
            "source_fidelity": fidelity,
            "reconstruction": {
                **base.get("reconstruction", {}),
                "fidelity": fidelity,
            },
            "screening_input": {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": digest(manifest_path),
                "id": ident,
            },
        })
        result.append(base)
    return result


def parse_targets(values: Iterable[str], ids: Iterable[str]) -> dict[str, float]:
    """Parse ``10:296`` overrides, falling back to the exploratory preview sizes."""
    targets = {str(key): float(value) for key, value in DEFAULT_TARGETS.items()}
    for raw in values:
        text = str(raw).strip()
        if ":" not in text:
            raise ValueError(f"target 格式应为 ID:像素，例如 10:296；收到 {text}")
        key, value = text.split(":", 1)
        key = key.strip()
        if key not in targets:
            raise ValueError(f"未知筛查 ID：{key}")
        try:
            target = float(value)
        except ValueError as error:
            raise ValueError(f"无效目标像素：{text}") from error
        if not 8 <= target <= 3072:
            raise ValueError(f"目标像素不在可用范围：{text}")
        targets[key] = target
    return {str(ident): targets[str(ident)] for ident in ids}


def candidate_checks(evidence: Mapping) -> dict:
    """Report formal checks without turning unknown evidence into a pass."""
    parts = evidence.get("component_support") or []
    return {
        "physics": "pass" if evidence.get("physics_stable") else "fail",
        "component_support": "pass" if parts and min(parts) >= .90 else "fail",
        "source_visibility": "unknown",
        "occlusion": "unknown",
        "scene_scale": "unknown",
        "photoshop": "not_run",
    }


def scene_summary(scene: Mapping) -> dict:
    answer = scene["answer"].model_dump(mode="json")
    return {
        "cache_hit": bool(scene.get("cache_hit")),
        "elevation_deg": scene["camera_elevation_deg"],
        "elevation_interval": list(scene["camera_elevation_interval"]),
        "zones": answer.get("zones", []),
        "references": answer.get("refs", []),
        "protected": answer.get("protect", []),
    }


def summary_payload(*, status: str, manifest_path: Path, manifest_sha256: str,
                    background_sha256: str, scene: Mapping | None,
                    samples: list[dict], notes: list[str]) -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "background_sha256": background_sha256,
        "scene": scene_summary(scene) if scene is not None else None,
        "samples": samples,
        "notes": notes,
    }
