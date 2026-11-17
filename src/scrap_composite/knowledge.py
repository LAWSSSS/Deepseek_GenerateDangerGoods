"""Local, evidence-bound parameter knowledge. No network, renderer or legacy imports.

Historical camera 53 is the development partition and camera 36 is a rule holdout;
both have previously been viewed, so this is not an unseen-camera quality claim.
Rules start as trials. Only matched, independently reviewed holdout comparisons
can approve them. Recommendations never certify geometry or physical scale.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "parameter-knowledge-v1"
FEATURE_VERSION = "local-features-v1"
RULE_VERSION = "paired-background-groups-v2"
VERSION_KEYS = ("photography", "camera_pipeline", "fog_mode", "weathering", "integration_mode", "workflow", "feature_schema")
PHOTO_KEYS = ("lighting", "material", "texture")
ACTION_KEYS = frozenset(("fog_gain", "fog_extra", "pool_kernel", "pool_weight", "surface_dust",
                         "chroma", "contrast", "haze", "sharpness", "dust", "tone"))
GEOMETRY_DEFECTS = frozenset(("floating", "penetration", "scale", "camera", "occlusion", "upright", "structure"))
PHOTO_DEFECTS = frozenset(("color", "material", "lighting", "texture", "shadow", "edge", "patch_seam",
                         "haze_clear", "haze_heavy", "haze_color", "identity_loss", "reflection_flat",
                         "reflection_hard", "surface_clean", "surface_dust_heavy", "too_sharp", "too_blurry"))
# Fixed before reading outcomes; no per-image or camera-specific parameter bins.
FEATURE_BINS = {"luma_median": (50., 100., 150., 200.), "contrast_span": (35., 80., 140.),
                "camera_sigma_px": (.6, 1.2, 2.), "target_major_px": (128., 256., 512., 1024.),
                "noise_sigma": (.002, .006, .012), "veil_estimate": (.03, .1, .2, .4)}
MIN_DEVELOPMENT_BACKGROUNDS = 3
MIN_HOLDOUT_BACKGROUNDS = 2


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(*values):
    return hashlib.sha256(_json(values).encode("utf8")).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _material(value):
    if isinstance(value, str):
        return value or "unknown"
    if isinstance(value, (tuple, list)):
        values = sorted(set(str(v) for v in value))
        return values[0] if len(values) == 1 else "+".join(values) or "unknown"
    return "unknown"


def versions_from_params(params, photography_version):
    return {"photography": photography_version or "unknown",
            "camera_pipeline": params.get("camera_pipeline", "legacy_residual"),
            "fog_mode": params.get("fog_mode", "legacy_untyped"),
            "weathering": params.get("weathering", "legacy_untyped"),
            "integration_mode": params.get("integration_mode", "legacy"),
            "workflow": params.get("workflow", "legacy"),
            "feature_schema": FEATURE_VERSION}


def features_from_environment(environment, params, *, material_group, target_major_px):
    """Use pre-effect local measurements, never adjusted object RGB or final fog."""
    local = environment.get("local_context", {})
    spread = local.get("luma_range", [])
    camera = environment.get("camera_psf", {})
    sensor = environment.get("sensor", {})
    return {"material_group": _material(material_group),
            "luma_median": environment.get("luma_median", spread[1] if len(spread) == 3 else None),
            "contrast_span": spread[2] - spread[0] if len(spread) == 3 else None,
            "camera_sigma_px": camera.get("sigma_px", camera.get("sigma")),
            "noise_sigma": sensor.get("noise_sigma"), "veil_estimate": local.get("veil_estimate"),
            "target_major_px": target_major_px,
            "local_sample_confidence": local.get("confidence", "unknown"),
            "current_params": dict(params)}


def context_from_result(result, *, material_group=None):
    """Adapt existing draft/result records without interpreting missing data as zero."""
    params = result.get("params", {})
    photo = result.get("photo", {})
    local = photo.get("local_context", {})
    spread = local.get("luma_range", [])
    fixed = result.get("fixed_candidate", {})
    environment = {"local_context": local, "luma_median": spread[1] if len(spread) == 3 else None,
                   "camera_psf": {"sigma_px": photo.get("measured_camera_sigma")},
                   "sensor": photo.get("sensor", {})}
    features = features_from_environment(environment, params,
        material_group=material_group or result.get("material_group") or fixed.get("material_classes"),
        target_major_px=result.get("measured_major_px", fixed.get("expected_major_px")))
    return {"features": features, "versions": versions_from_params(params, photo.get("version"))}


def camera_group(background, explicit=None):
    """Used for evaluation partitioning only, never as a recommendation feature."""
    if explicit in ("53", "36"):
        return explicit
    name = str(background or "").replace("\\", "/").rsplit("/", 1)[-1]
    match = re.search(r"(?:^|_)(53|36)_\d+_\d+(?:\.[^.]+)?$", name)
    return match.group(1) if match else None


def _checks_summary(checks):
    summary = {}
    for name, value in (checks or {}).items():
        if not isinstance(value, dict):
            continue
        summary[name] = {k: value[k] for k in ("status", "basis", "reason", "measurements", "maximum_px",
            "scene_geometry_status", "threshold_px") if k in value}
    return summary


def _geometry_key(record):
    if record.get("geometry_key"):
        return record["geometry_key"]
    fixed = record.get("fixed_candidate", {})
    checks = record.get("checks", {})
    scale = checks.get("scale", {}).get("evidence", {})
    physics = checks.get("physics", {}).get("evidence", {})
    matrix = fixed.get("matrix", physics.get("matrix"))
    anchor = fixed.get("anchor_px", scale.get("anchor_px"))
    alpha = record.get("alpha_hash")
    visible = record.get("visible_alpha_hash")
    # A candidate name or a 'geometry unchanged' label alone is not a fingerprint.
    if not matrix or not anchor or not alpha or not visible:
        return None
    return _hash(matrix, anchor, alpha, visible, record.get("foreground_mask_hash"),
                 fixed.get("full_px", scale.get("world_pixels_per_unit")),
                 fixed.get("elevation"), scale.get("calibration_signature"), record.get("support_ids"))


def _context_bin(features):
    if features.get("material_group") in (None, "unknown", ""):
        return None
    if features.get("local_sample_confidence") not in ("estimated", "measured"):
        return None
    conditions = {"material_group": {"eq": features["material_group"]},
                  "local_sample_confidence": {"in": ["estimated", "measured"]}}
    for key, boundaries in FEATURE_BINS.items():
        value = features.get(key)
        if not _number(value) or value < 0:
            return None
        lower = 0.; upper = None
        for edge in boundaries:
            if value < edge:
                upper = edge; break
            lower = edge
        conditions[key] = {"min": lower, "max_exclusive": upper}
    return conditions


def _matches(features, conditions):
    for name, condition in conditions.items():
        value = features.get(name)
        if "eq" in condition and value != condition["eq"]:
            return False
        if "in" in condition and value not in condition["in"]:
            return False
        if "min" in condition:
            if not _number(value) or value < condition["min"]:
                return False
            if condition["max_exclusive"] is not None and value >= condition["max_exclusive"]:
                return False
    return True


def affected_stages(changes):
    stages = ["compose", "pixel_checks", "visual_review"]
    if "surface_dust" in changes:
        stages = ["material", "object_render", "with_object_shadow"] + stages
    return stages


def _review_quality(case):
    assessment = case.get("assessment", {})
    if assessment.get("source") not in ("vision", "user_structured") or not assessment.get("image_bound"):
        return None
    scores = assessment.get("scores", {})
    if not all(_number(scores.get(k)) and 0 <= scores[k] <= 4 for k in PHOTO_KEYS):
        return None
    return {k: scores[k] for k in PHOTO_KEYS}


def _eligible_case(case):
    reasons = []
    if not case.get("background_group"):
        reasons.append("independent_background_group_missing")
    if not case.get("geometry_key"):
        reasons.append("geometry_fingerprint_missing")
    if _context_bin(case.get("features", {})) is None:
        reasons.append("local_feature_evidence_missing")
    if any(case.get("versions", {}).get(k) in (None, "unknown", "") for k in VERSION_KEYS):
        reasons.append("algorithm_version_missing")
    if _review_quality(case) is None:
        reasons.append("image_bound_quality_assessment_missing")
    structure = case.get("assessment", {}).get("scores", {}).get("structure")
    if not _number(structure) or structure < 3:
        reasons.append("key_structure_not_verified")
    checks = case.get("checks", {})
    if any(c.get("status") == "fail" for c in checks.values() if isinstance(c, dict)):
        reasons.append("program_failure")
    if case.get("high_risk_projection"):
        reasons.append("untrusted_geometry")
    defects = case.get("assessment", {}).get("defects", [])
    if any(d.get("severity") in ("major", "fatal") and d.get("code") in GEOMETRY_DEFECTS
           and d.get("visibility", "visible") == "visible" for d in defects):
        reasons.append("explicit_geometry_or_structure_defect")
    if not all(checks.get(k, {}).get("status") == "pass" for k in ("protected", "outside_edit", "holes_alpha")):
        reasons.append("pixel_protection_not_verified")
    return reasons


class KnowledgeBase:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path)
        if readonly:
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        if not readonly:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY,format TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS source_locations(source_id TEXT NOT NULL,path TEXT NOT NULL,
              PRIMARY KEY(source_id,path));
            CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY,study_id TEXT NOT NULL,partition TEXT NOT NULL,
              asset_id TEXT NOT NULL,background_id TEXT NOT NULL,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS case_sources(case_id TEXT NOT NULL,source_id TEXT NOT NULL,ordinal TEXT NOT NULL,
              PRIMARY KEY(case_id,source_id,ordinal));
            CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY,kind TEXT NOT NULL,source_id TEXT NOT NULL,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cost_events(id TEXT PRIMARY KEY,ledger_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
              data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS trials(id TEXT PRIMARY KEY,partition TEXT NOT NULL,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rules(id TEXT PRIMARY KEY,state TEXT NOT NULL,data TEXT NOT NULL);
            """)
            old = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if old and old[0] != SCHEMA_VERSION:
                raise ValueError("unsupported knowledge schema; migrate explicitly")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('schema_version',?)", (SCHEMA_VERSION,))
            self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_args):
        if exc_type is None:
            self.db.commit()
        else:
            self.db.rollback()
        self.close()

    def record_case(self, case, *, source_id="runtime", ordinal="0"):
        case = dict(case)
        camera = case.get("camera_group")
        case["partition"] = "development" if camera == "53" else "holdout" if camera == "36" else "unassigned"
        # Repeated evaluations of one exact image and parameters are one observation,
        # not additional independent samples. Conflicting assessments remain facts.
        identity = case.get("image_hash") or case.get("image") or _hash(case)
        ident = _hash(identity, case.get("geometry_key"), case.get("params", {}), case.get("versions", {}))
        case["id"] = ident
        previous = self.db.execute("SELECT data FROM cases WHERE id=?", (ident,)).fetchone()
        if previous:
            old = json.loads(previous[0])
            if old.get("assessment") != case.get("assessment") or old.get("checks") != case.get("checks"):
                self._fact("case_assessment_conflict", source_id, {"case_id": ident,
                    "reason": "conflicting evaluation for one image; excluded from rule induction"})
        self.db.execute("INSERT OR IGNORE INTO cases VALUES(?,?,?,?,?,?)", (ident,
            case.get("study_id", "unknown"), case["partition"], str(case.get("asset_id", "unknown")),
            case.get("background_id", "unknown"), _json(case)))
        self.db.execute("INSERT OR IGNORE INTO case_sources VALUES(?,?,?)", (ident, source_id, str(ordinal)))
        self._fact("case_observation", source_id, {"case_id": ident, "assessment": case.get("assessment", {}),
            "checks": case.get("checks", {}), "eligibility_rejections": _eligible_case(case)})
        return ident

    def _fact(self, kind, source_id, data):
        self.db.execute("INSERT OR IGNORE INTO facts VALUES(?,?,?,?)", (_hash(kind, source_id, data), kind, source_id, _json(data)))

    def _normalize_case(self, row, *, study_id, parent=None):
        parent = parent or {}
        # A child image must carry its own scores, measurements and parameters.
        # Copying the chosen final image's review to a different iteration is invalid.
        inherited = {k: parent[k] for k in ("asset_id", "background", "group", "background_group", "camera_group", "fixed_candidate",
                     "geometry_key", "material_group", "background_hash") if k in parent}
        record = {**inherited, **row}
        context = context_from_result(record)
        features = record.get("features", context["features"])
        versions = record.get("algorithm_versions", context["versions"])
        review = record.get("review") or {}
        image = record.get("image", record.get("after_full"))
        background = record.get("background", record.get("group", ""))
        group = record.get("background_group") or record.get("group")
        legacy_frame_group = re.fullmatch(r"(36|53)_(\d+)_\d+", str(group))
        if not record.get("background_group") and legacy_frame_group:
            group = "-".join(legacy_frame_group.groups())
        checks = record.get("checks", record.get("new_program_checks", {}))
        scale = checks.get("scale", {}).get("evidence", {})
        gate = scale.get("projection_preflight", scale.get("calibration", {}).get("projection_preflight", {}))
        assessment = record.get("assessment") or {"source": "vision" if review else "unreviewed",
            "image_bound": bool(review and image), "scores": review.get("scores", {}),
            "defects": review.get("defects", []), "trace": review.get("trace"),
            "user_style_confirmation": record.get("user_confirmed_style", False)}
        return {"study_id": study_id, "sequence": record.get("iteration", record.get("number", 0)),
            "asset_id": str(record.get("asset_id", "unknown")),
            "background_id": record.get("background_hash") or _hash(str(background).replace("\\", "/")),
            "background_group": group,
            "background": str(background), "camera_group": camera_group(background, record.get("camera_group")),
            "image": image, "image_hash": record.get("image_hash"), "geometry_key": _geometry_key(record),
            "params": record.get("params", {}), "features": features, "versions": versions,
            "requested_configuration": record.get("configuration", {}).get("requested", {}),
            "resolved_configuration": record.get("configuration", {}).get("resolved", {}),
            "configuration_reasons": record.get("configuration", {}).get("reasons", {}),
            "applied_photo": record.get("photo", {}),
            "stop_reason": record.get("reason"),
            "render_evidence": {"cache": record.get("render_cache"), "resolution": record.get("render_resolution"),
                                "composition_manifest": record.get("composition_manifest")},
            "assessment": assessment, "checks": _checks_summary(checks),
            "high_risk_projection": (gate.get("state") == "high_risk_projection" or
                scale.get("calibration", {}).get("correspondence_status") == "inferred"),
            "scale_evidence": {"status": checks.get("scale", {}).get("status"),
                "basis": scale.get("basis"), "physical_size_basis": scale.get("physical_size", {}).get("basis")},
            "legacy_status": record.get("status", record.get("review_status", "unknown"))}

    def import_history(self, paths):
        """Explicit local result/usage paths only. Never scans credentials or follows links."""
        report = {"files": 0, "errors": [], "cases_before": self.counts()["cases"]}
        for value in paths:
            path = Path(value)
            try:
                if path.suffix.lower() not in (".json", ".jsonl") or path.name.startswith("."):
                    raise ValueError("only explicit result/usage JSON inputs are supported")
                data = ([json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
                        if path.suffix.lower() == ".jsonl" else json.loads(path.read_text(encoding="utf8")))
                source_id = _file_hash(path)
                with self.db:
                    known_source = self.db.execute("SELECT 1 FROM sources WHERE id=?", (source_id,)).fetchone()
                    self.db.execute("INSERT OR IGNORE INTO sources VALUES(?,?)", (source_id, path.suffix.lower()))
                    self.db.execute("INSERT OR IGNORE INTO source_locations VALUES(?,?)", (source_id, str(path.resolve())))
                    if isinstance(data, dict) and isinstance(data.get("calls"), list):
                        if not known_source:
                            self._import_usage(path, data, source_id)
                    else:
                        self._import_results(path, data, source_id)
                report["files"] += 1
            except (OSError, ValueError, TypeError, KeyError) as error:
                report["errors"].append({"path": str(path), "reason": str(error)})
        report["cases_after"] = self.counts()["cases"]
        return report

    def _import_usage(self, path, document, source_id):
        # A ledger is cumulative across checkpoints. Prefix calls retain identity
        # when the same file grows. Separate runs may send the same request again.
        ledger = document.get("ledger_name") or str(path.resolve())
        ledger_id = _hash("usage-ledger", ledger)
        for index, call in enumerate(document["calls"]):
            if call.get("state") == "not_sent":
                continue
            provider_id = call.get("provider_request_id") or call.get("request_id")
            ident = _hash("provider", call.get("model"), provider_id) if provider_id else _hash(ledger_id, index)
            usage = call.get("usage") or {}
            safe = {k: call.get(k) for k in ("role", "model", "request_signature", "attempt", "job_id", "seconds", "http_status")}
            safe.update(usage=usage, tariff=call.get("request_tariff", document.get("tariff")),
                        identity_basis="provider_request_id" if provider_id else "ledger_and_ordinal")
            old = self.db.execute("SELECT data FROM cost_events WHERE id=?", (ident,)).fetchone()
            if old and json.loads(old[0]) != safe:
                self._fact("cost_identity_conflict", source_id, {"event_id": ident, "ledger": ledger,
                    "reason": "same ledger ordinal differs; retained original event; costs incomplete"})
                continue
            self.db.execute("INSERT OR IGNORE INTO cost_events VALUES(?,?,?,?)", (ident, ledger_id, index, _json(safe)))
        self._fact("reported_cost_summary", source_id, {"ledger": ledger, "cost": document.get("cost"),
                    "aggregate_only_not_added_to_event_totals": True})

    def _import_results(self, path, data, source_id):
        if isinstance(data, dict) and data.get("kind") == "knowledge-cases-v1":
            for index, case in enumerate(data.get("cases", [])):
                self.record_case(case, source_id=source_id, ordinal=index)
            return
        rows = data if isinstance(data, list) else next((data[k] for k in ("records", "results", "cases", "rows")
                    if isinstance(data.get(k), list)), None) if isinstance(data, dict) else None
        if rows is None:
            self._fact("legacy_aggregate", source_id, {"keys": list(data) if isinstance(data, dict) else [],
                "summary": {k: data[k] for k in ("actual_api", "cost_total", "task_totals", "seconds", "render_counts",
                    "new_api_requests", "new_render_count", "quality_status") if isinstance(data, dict) and k in data},
                "eligible_for_rules": False, "added_to_cost_events": False})
            return
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            identity = row.get("job_id", row.get("id", row.get("number", index)))
            related = {}
            # Join material/geometry evidence already saved beside a batch result.
            # Restrict the read to this job's known candidate artifact, never an
            # arbitrary path from a result or a credential/configuration file.
            if re.fullmatch(r"[A-Za-z0-9_.-]+", str(row.get("job_id", ""))):
                candidate_path = path.parent/"jobs"/row["job_id"]/"candidates.json"
                if candidate_path.is_file():
                    candidate_document = json.loads(candidate_path.read_text(encoding="utf8"))
                    for candidate in candidate_document.get("candidates", []):
                        related[candidate["id"]] = {k: candidate[k] for k in ("id", "matrix", "anchor_px", "full_px",
                            "elevation", "expected_major_px", "material_classes") if k in candidate}
                    self._fact("joined_candidate_evidence", source_id, {"path":str(candidate_path.resolve()),
                        "sha256":_file_hash(candidate_path),"candidate_ids":sorted(related)})
            # Source folder only groups a comparison; it is not a model feature.
            study = _hash(str(path.parent.resolve()), identity)
            children = row.get("iterations") or (row.get("versions") if isinstance(row.get("versions"), list) else None)
            if children and all(isinstance(v, dict) for v in children):
                for number, child in enumerate(children):
                    candidate = related.get(child.get("params", {}).get("candidate"))
                    inherited = {**row, "fixed_candidate": candidate} if candidate else row
                    case = self._normalize_case(child, study_id=study, parent=inherited)
                    self.record_case(case, source_id=source_id, ordinal=f"{index}:{number}")
            else:
                case = self._normalize_case(row, study_id=study)
                self.record_case(case, source_id=source_id, ordinal=index)

    def counts(self):
        return {table: self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in ("sources", "cases", "facts", "cost_events", "trials", "rules")}

    def derive_trials(self):
        groups = defaultdict(list)
        for record in self.db.execute("SELECT data FROM cases"):
            case = json.loads(record[0]); groups[case["study_id"]].append(case)
        inserted = 0; rejected = Counter()
        conflicts = {json.loads(r[0])["case_id"] for r in self.db.execute("SELECT data FROM facts WHERE kind='case_assessment_conflict'")}
        with self.db:
            # Derived trials can be rebuilt; immutable source facts and cases remain.
            self.db.execute("DELETE FROM trials")
            for cases in groups.values():
                cases.sort(key=lambda c: (c.get("sequence", 0), c["id"]))
                for before, after in zip(cases, cases[1:]):
                    reasons = _eligible_case(before) + _eligible_case(after)
                    if before["id"] in conflicts or after["id"] in conflicts:
                        reasons.append("conflicting_assessments")
                    if before.get("geometry_key") != after.get("geometry_key"):
                        reasons.append("geometry_changed")
                    if before["versions"] != after["versions"]:
                        reasons.append("algorithm_changed")
                    if before["partition"] != after["partition"] or before["background_id"] != after["background_id"]:
                        reasons.append("comparison_context_changed")
                    if before.get("background_group") != after.get("background_group"):
                        reasons.append("background_group_changed")
                    conditions = _context_bin(before.get("features", {}))
                    if conditions != _context_bin(after.get("features", {})):
                        reasons.append("local_conditions_changed")
                    changes = {k: {"from": before["params"].get(k), "to": after["params"].get(k)}
                        for k in set(before["params"]) | set(after["params"]) if before["params"].get(k) != after["params"].get(k)}
                    if not changes or len(changes) > 3 or not set(changes) <= ACTION_KEYS:
                        reasons.append("uncontrolled_or_empty_action_bundle")
                    if reasons:
                        rejected.update(set(reasons)); continue
                    old = _review_quality(before); new = _review_quality(after)
                    defects = after["assessment"].get("defects", [])
                    severe = any(d.get("severity") in ("major", "fatal") and d.get("code") in PHOTO_DEFECTS for d in defects)
                    success = (min(new.values()) >= 3 and all(new[k] >= old[k] for k in PHOTO_KEYS)
                               and sum(new.values()) > sum(old.values()) and not severe)
                    # One asset/background/geometry/action comparison counts once,
                    # even when cached outputs were copied into several reports.
                    ident = _hash(before["asset_id"], before["background_id"], before["geometry_key"], before["versions"], changes)
                    trial = {"id": ident, "partition": before["partition"], "before": before["id"], "after": after["id"],
                        "asset_id": before["asset_id"], "background_id": before["background_id"],
                        "background_group": before["background_group"],
                        "conditions": conditions, "versions": before["versions"], "changes": changes,
                        "before_scores": old, "after_scores": new, "delta": sum(new.values())-sum(old.values()),
                        "success": success, "severe_photo_defect": severe, "scope": "photography_only",
                        "unknown_geometry_does_not_become_pass": True, "rule_method": RULE_VERSION}
                    inserted += self.db.execute("INSERT OR IGNORE INTO trials VALUES(?,?,?)", (ident, before["partition"], _json(trial))).rowcount
        return {"new_trials": inserted, "rejected_comparisons": dict(rejected)}

    @staticmethod
    def _aggregate(trials):
        by_background=defaultdict(list)
        quality_by_background=defaultdict(list)
        for trial in trials:
            by_background[trial["background_group"]].append(trial["delta"])
            quality_by_background[trial["background_group"]].append(min(trial["after_scores"].values()))
        group_deltas={group:statistics.median(values) for group,values in by_background.items()}
        return {"pairs": len(trials), "successes": sum(t["success"] for t in trials),
                "assets": len({t["asset_id"] for t in trials}), "backgrounds": len({t["background_group"] for t in trials}),
                "background_images": len({t["background_id"] for t in trials}),
                "severe_photo_defects": sum(t["severe_photo_defect"] for t in trials),
                "mean_score_delta": sum(t["delta"] for t in trials)/len(trials) if trials else None,
                "median_score_delta": statistics.median(group_deltas.values()) if group_deltas else None,
                "background_group_deltas": group_deltas,
                "backgrounds_below_photo_threshold": sorted(
                    group for group,values in quality_by_background.items() if statistics.median(values)<3),
                "trial_ids": sorted(t["id"] for t in trials)}

    def induce_rules(self):
        """Only development determines conditions/actions; holdout cannot create rules."""
        trials = [json.loads(r[0]) for r in self.db.execute("SELECT data FROM trials")]
        groups = defaultdict(list)
        for trial in trials:
            if trial["partition"] == "development":
                groups[_hash(trial["conditions"], trial["versions"], trial["changes"])].append(trial)
        result = Counter()
        with self.db:
            for existing in self.db.execute("SELECT id,data FROM rules").fetchall():
                retired = json.loads(existing["data"]); retired["state"] = "superseded"
                self.db.execute("UPDATE rules SET state='superseded',data=? WHERE id=?", (_json(retired), existing["id"]))
            for key, development in sorted(groups.items()):
                example = development[0]; stats = self._aggregate(development)
                if stats["backgrounds"] < MIN_DEVELOPMENT_BACKGROUNDS:
                    result["insufficient_development_groups"] += 1; continue
                ident = _hash(RULE_VERSION, key)
                holdout = [t for t in trials if t["partition"] == "holdout" and t["conditions"] == example["conditions"]
                           and t["versions"] == example["versions"] and t["changes"] == example["changes"]]
                validation = self._aggregate(holdout)
                dev_good = stats["severe_photo_defects"] == 0 and stats["median_score_delta"] > 0
                enough_holdout = validation["backgrounds"] >= MIN_HOLDOUT_BACKGROUNDS
                val_good = (enough_holdout and validation["severe_photo_defects"] == 0
                            and validation["median_score_delta"] > 0 and not validation["backgrounds_below_photo_threshold"])
                state = ("rejected" if not dev_good or validation["severe_photo_defects"] else
                         "trial" if not enough_holdout else "approved" if val_good else "rejected")
                rule = {"id": ident, "method": RULE_VERSION, "state": state, "conditions": example["conditions"],
                    "versions": example["versions"], "changes": example["changes"], "development": stats,
                    "holdout": validation, "scope": "photography_only", "holdout_is_previously_seen": True,
                    "affected_stages": affected_stages(example["changes"]),
                    "thresholds": {"development_backgrounds": MIN_DEVELOPMENT_BACKGROUNDS,
                                   "holdout_backgrounds": MIN_HOLDOUT_BACKGROUNDS, "paired_median_delta_strictly_positive": True},
                    "limitations": ["does_not_certify_scale_or_support", "observational_paired_trial_not_randomized_population_effect"]}
                self.db.execute("INSERT INTO rules VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,data=excluded.data",
                                (ident, state, _json(rule)))
                result[state] += 1
        return dict(result)

    def recommend(self, features, versions):
        empty = {"status": "no_recommendation", "reason": "no_approved_matching_evidence", "rule_ids": [],
                 "overrides": {}, "scope": "photography_only", "geometry_status_changed": False}
        if _context_bin(features) is None or any(versions.get(k) in (None, "unknown", "") for k in VERSION_KEYS):
            return {**empty, "reason": "missing_local_features_or_versions"}
        current = features.get("current_params", {})
        matches = []
        for row in self.db.execute("SELECT data FROM rules WHERE state='approved' ORDER BY id"):
            rule = json.loads(row[0])
            if rule.get("method") != RULE_VERSION or versions != rule["versions"] or not _matches(features, rule["conditions"]):
                continue
            if not all(current.get(key) == change["from"] for key, change in rule["changes"].items()):
                continue
            matches.append(rule)
        if not matches:
            return empty
        # Do not average incompatible recipes or cherry-pick by camera/image ID.
        values = {_json({k: c["to"] for k, c in r["changes"].items()}) for r in matches}
        if len(values) != 1:
            return {**empty, "reason": "conflicting_approved_rules"}
        rule = sorted(matches, key=lambda r: (-r["holdout"]["pairs"], -r["development"]["pairs"], r["id"]))[0]
        return {"status": "recommended", "reason": "matched_versioned_paired_evidence", "rule_ids": [rule["id"]],
                "overrides": {k: str(v["to"]) for k, v in rule["changes"].items()}, "scope": "photography_only",
                "geometry_status_changed": False, "affected_stages": affected_stages(rule["changes"]),
                "evidence": {"development": rule["development"], "holdout": rule["holdout"]}}

    def cost_summary(self):
        requests = tokens = missing = 0; low = high = 0.; currencies = set()
        for row in self.db.execute("SELECT data FROM cost_events"):
            item = json.loads(row[0]); usage = item.get("usage") or {}; requests += 1
            prompt = usage.get("prompt_tokens"); completion = usage.get("completion_tokens")
            tokens += usage.get("total_tokens") or ((prompt or 0)+(completion or 0))
            rate = item.get("tariff")
            if not rate or prompt is None or completion is None:
                missing += 1; continue
            currencies.add(rate.get("currency"))
            cached = usage.get("prompt_cache_hit_tokens")
            min_cached = prompt if cached is None else max(0, min(prompt, cached))
            max_cached = 0 if cached is None else min_cached
            a, b = rate["low"], rate["high"]
            low += (min_cached*a[0]+(prompt-min_cached)*a[1]+completion*a[2])/1e6
            high += (max_cached*b[0]+(prompt-max_cached)*b[1]+completion*b[2])/1e6
        conflicts = self.db.execute("SELECT count(*) FROM facts WHERE kind='cost_identity_conflict'").fetchone()[0]
        aggregates = sum(bool(set(json.loads(row[0]).get("summary", {})) & {"actual_api", "cost_total", "task_totals"})
                         for row in self.db.execute("SELECT data FROM facts WHERE kind='legacy_aggregate'"))
        return {"actual_requests": requests, "reported_tokens": tokens, "unpriced_or_unreported_calls": missing,
                "identity_conflicts": conflicts, "aggregate_records_not_added": aggregates,
                "currency": next(iter(currencies)) if len(currencies) == 1 else None,
                "estimated_min": round(low, 8) if len(currencies) <= 1 else None,
                "estimated_max": round(high, 8) if not missing and not conflicts and not aggregates and len(currencies) <= 1 else None,
                "complete": not missing and not conflicts and not aggregates and len(currencies) <= 1,
                "local_compute_cost": None, "basis": "deduplicated_actual_events_not_provider_invoice"}

    def export_report(self, output_dir):
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        cases = [json.loads(r[0]) for r in self.db.execute("SELECT data FROM cases ORDER BY partition,study_id,id")]
        rules = [json.loads(r[0]) for r in self.db.execute("SELECT data FROM rules ORDER BY state,id")]
        rejected = Counter(reason for case in cases for reason in _eligible_case(case))
        summary = {"schema_version": SCHEMA_VERSION, "counts": self.counts(), "cost": self.cost_summary(),
                   "partitions": dict(Counter(c["partition"] for c in cases)), "rule_states": dict(Counter(r["state"] for r in rules)),
                   "case_eligibility_rejections": dict(rejected), "rules": rules,
                   "coverage":"Only explicitly imported historical files; not the complete project cost ledger",
                   "feature_coverage":{key:sum(_number(c.get('features',{}).get(key)) for c in cases) for key in FEATURE_BINS},
                   "holdout_note": "53 historical development / 36 previously seen rule holdout; not independent unseen-camera acceptance"}
        (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
        body = ["<!doctype html><meta charset='utf-8'><title>参数证据库</title><style>body{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;text-align:left}pre{white-space:pre-wrap}img{max-width:100%;height:auto}summary{cursor:pointer}</style>",
                "<h1>参数证据库</h1><p>历史事实、视觉效果与物理证据分开保存。无有效规则时不修改生成参数。53 用于历史归纳，36 仅用于规则留出验证；均不是全新相机验收。</p>",
                "<h2>汇总</h2><pre>"+html.escape(json.dumps({k:v for k,v in summary.items() if k!='rules'},ensure_ascii=False,indent=2))+"</pre>",
                "<h2>规则</h2>"]
        if not rules:
            body.append("<p>暂无达到证据门槛的规则；推荐保持为空。</p>")
        for rule in rules:
            body.append("<details><summary>"+html.escape(rule["state"]+" · "+rule["id"][:12])+"</summary><pre>"+
                        html.escape(json.dumps(rule,ensure_ascii=False,indent=2))+"</pre></details>")
        body.append("<h2>历史记录（完整画面）</h2>")
        for case in cases:
            label=f"{case['partition']} · {case['asset_id']} · {case.get('legacy_status','unknown')} · {case['id'][:10]}"
            body.append("<details><summary>"+html.escape(label)+"</summary><pre>"+
                        html.escape(json.dumps(case,ensure_ascii=False,indent=2))+"</pre>")
            image=case.get("image")
            if image and Path(image).is_file():
                uri=html.escape(Path(image).resolve().as_uri(),quote=True)
                body.append(f'<a href="{uri}"><img loading="lazy" src="{uri}" alt="完整原图"></a>')
            body.append("</details>")
        (output/"index.html").write_text("\n".join(body),encoding="utf8")
        return summary


def recommend(features, versions, *, db_path):
    """Read-only integration hook. A missing library has no filesystem side effect."""
    if not Path(db_path).is_file():
        return {"status":"no_recommendation","reason":"knowledge_database_missing","rule_ids":[],
                "overrides":{},"scope":"photography_only","geometry_status_changed":False}
    with KnowledgeBase(db_path, readonly=True) as knowledge:
        return knowledge.recommend(features, versions)


def archive_run(folder, db_path):
    """Index the run's result records and actual request ledgers."""
    folder=Path(folder)
    paths=[path for path in (folder/'results.jsonl',folder/'workflow.json') if path.is_file()]
    paths.extend(sorted(folder.glob('usage-*.json')))
    with KnowledgeBase(db_path) as knowledge:
        return knowledge.import_history(paths)
