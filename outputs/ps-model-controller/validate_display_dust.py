"""Recompose ten fixed placements with one pooled object layer and local dust veil."""
import argparse
import asyncio
from collections import Counter
import html
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field

from display_batch.ai import ProviderUnavailable, Vision
from display_batch.common import HERE, atomic_json, file_hash, load_catalog, resolve, signature
from display_batch.compose import compose, cv2, sample_environment
from display_batch.costs import cumulative, estimate, label, tariff
from display_batch.framing import final_resolution, render_windows
from display_batch.fusion import topology
from display_batch.pipeline import batch_lock
from display_batch.rendering import PART_FILES, RENDER_VERSION, RenderCache
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig, Score, Strict
from feedback_rollback import BASELINE, ROLLBACK, SIZED, audit_png, placement_masks
from validate_display_costs import montage
from validate_display_local import read, records


ASSETS = {"3", "13", "22", "31", "33"}
OUTPUT = "outputs/薄雾池化-5x2-20261111"
ROLE = "local-dust-pooling-pair"
PROMPT = """Blind comparison of two composites in a scrap-hopper security camera.
Scene is the original same-position crop. Judge A and B against that crop.
Give each 0–4 scores for fog (airborne dust matches the immediate surroundings),
clarity (camera sharpness matches and is not smeared), fusion (no edge or rectangular
patch, coherent contact), and structure (visible holes, rods and edges remain readable).
3 means usable and 4 means convincing. Mark serious for a conspicuous new halo,
opaque wash, obscured key structure or pasted patch. Hidden details may be uncertain.
Choose the more naturally integrated image if it preserves the key structure, or equal
if the visible difference is immaterial. Use one short Chinese evidence sentence per image."""


class Judgment(Strict):
    fog: Score
    clarity: Score
    fusion: Score
    structure: Score
    serious: bool
    evidence: str = Field(max_length=100)


class Comparison(Strict):
    A: Judgment
    B: Judgment
    preferred: Literal["A", "B", "equal"]


def frozen_passes(row, history, sized, cache):
    old = history[row["number"]]
    saved = next(i for i in old["iterations"] if i["iteration"] == old["best_iteration"])
    if row["same_size_control"]:
        saved = read(sized / "size-controls" / old["job_id"] / "control.json")
    source = Path(row["archived_render"])
    meta = dict(row["archived_meta"])
    window = row["archived_window"]
    if window:
        windows = render_windows(row["pose"], row["audit"]["size"], final_resolution(row["pose"]))
        if windows["object"] != window:
            raise ValueError("Archived object projection grid changed")
        parts = {"object": str(source)}
        for part in ("with_shadow", "without_shadow"):
            key = saved["render_cache"][part]["signature"]
            info, reason = RenderCache.ready(cache / RENDER_VERSION / part / key, part)
            if info is None or info["camera"] != meta["camera"]:
                raise ValueError(f"Frozen {part} unavailable or camera differs: {reason}")
            parts[part] = info["path"]
        meta.update(object_window=window, shadow_window=windows["shadow"], part_paths=parts)
        passes = [Path(parts[k]) / f for k in parts for f in PART_FILES[k]]
    else:
        passes = [source / f for f in PART_FILES["object"] if (source / f).is_file()]
        linear_shadows = [source / PART_FILES[k][0] for k in ("with_shadow", "without_shadow")]
        passes.extend(linear_shadows if all(p.is_file() for p in linear_shadows) else
                      [source / "with-shadow.png", source / "without-shadow.png"])
        for required in ("object.png", "diffuse0001.exr", "specular0001.exr", "depth0001.exr"):
            if not (source / required).is_file():
                raise ValueError(f"Frozen object pass is missing: {required}")
    return source, meta, saved["shadow_hash"], {str(p): file_hash(p) for p in passes}


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def pair_audit(row, scene, source, meta, baseline, trial, archived, shadow_hash):
    pose = row["pose"]
    bounds = row["baseline"]["edit_bounds"]
    if any(value["edit_bounds"] != bounds for value in (baseline, trial)):
        raise ValueError("Edit bounds differ from the frozen projection")
    x, y, r, b = bounds
    alpha, foreground, visible, offset = placement_masks(scene, pose, source, row["archived_window"], meta, bounds)
    audit = audit_png(scene, trial, alpha, foreground, visible, pose, offset)
    a, z = read_rgb(baseline["image"]), read_rgb(trial["image"])
    same_archive = np.array_equal(a, archived)
    differences = np.any(a != z, axis=2)
    changed = differences[y:b, x:r]
    core = cv2.erode((visible > .9).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def edge_energy(rgb):
        gray = cv2.cvtColor(rgb[y:b, x:r], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=.125)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=.125)
        return float(np.percentile(cv2.magnitude(dx, dy)[core], 80)) if core.any() else None

    frozen_keys = ("alpha_hash", "visible_alpha_hash", "foreground_mask_hash")
    masks_equal = all(trial[k] == baseline[k] == row["baseline"][k] for k in frozen_keys)
    optics_equal = all(trial["photo"][k] == baseline["photo"][k] for k in
                       ("blur_sigma", "added_noise_sigma", "existing_noise_sigma", "optical_iou"))
    outside_visible = int(differences.sum() - (changed & (visible > 0)).sum())
    audit.update(archived_rgb_equal=same_archive, archive_pixel_difference=int(np.any(a != archived, axis=2).sum()),
                 frozen_masks_equal=masks_equal, camera_optics_equal=optics_equal,
                 shadow_equal=trial["shadow_hash"] == baseline["shadow_hash"] == shadow_hash,
                 pixels_changed=int(differences.sum()), added_pixels_outside_visible=outside_visible,
                 projected_holes=topology(alpha > .5)[1], edge_gradient_p80=[edge_energy(a), edge_energy(z)])
    audit["pass"] &= (masks_equal and optics_equal and audit["shadow_equal"] and not outside_visible)
    return audit


def make_comparison(folder, row, scene, original, trial, bounds):
    for name, path in (("baseline", original), ("trial", trial), ("scene", row["background"])):
        image = scene["frame"] if name == "scene" else Image.fromarray(read_rgb(path))
        crop = image.crop(tuple(bounds))
        crop.thumbnail((1050, 750))
        crop.save(folder / (name + ".jpg"), quality=94)
    order = ["baseline", "trial"]
    random.Random(signature(row["seed"], row["number"], ROLE)).shuffle(order)
    labels = dict(zip("AB", order))
    montage([(alias, folder / (name + ".jpg")) for alias, name in labels.items()] +
            [("Scene", folder / "scene.jpg")], folder / "blind.jpg")
    return labels


def recompose(folder, row, scene, source, meta, shadow_hash):
    start = time.monotonic()
    saved = folder / "composition.json"
    if saved.is_file():
        data = read(saved)
        for key in ("baseline", "trial"):
            if file_hash(data[key]) != data["image_hashes"][key]:
                raise ValueError(f"Saved {key} PNG changed")
        return data, True
    environment, samples = sample_environment(scene, row["pose"], local=True)
    args = (scene, row["pose"], source, meta, environment, samples, row["params"], row["seed"])
    trial_path = folder / "B.png"
    archived = read_rgb(row["image"])
    with tempfile.TemporaryDirectory(prefix="replay-", dir=folder) as tmp:
        a = compose(*args, Path(tmp) / "A.png")
        b = compose(*args, trial_path, dust_overlay=True)
        audit = pair_audit(row, scene, source, meta, a, b, archived, shadow_hash)
        if audit["archived_rgb_equal"]:
            original = row["image"]
        else:
            original = str(folder / "A.png")
            Path(a["image"]).replace(original)
    labels = make_comparison(folder, row, scene, original, trial_path, environment["local_context"]["bounds"])
    data = {"number": row["number"], "asset_id": row["asset_id"], "name": row["name"],
            "background": row["background"], "baseline": original, "trial": str(trial_path),
            "archived_baseline": row["image"], "label_map": labels,
            "image_hashes": {"baseline": file_hash(original), "trial": file_hash(trial_path)},
            "audit": audit, "effects": b["photo"]["dust_overlay"], "checks": b["checks"],
            "shadow_hash": b["shadow_hash"], "source_pose_signature": signature(row["pose"]),
            "seconds": round(time.monotonic() - start, 3), "status": "review" if audit["pass"] else "failed"}
    atomic_json(saved, data)
    return data, False


async def review(vision, case, folder):
    saved = folder / "review.json"
    if saved.is_file():
        record = read(saved)
        if record["image_hashes"] != case["image_hashes"]:
            raise ValueError("Blind review input PNG changed")
        return record
    record = {"image_hashes": case["image_hashes"]}
    try:
        answer = await vision.ask(ROLE, Comparison, PROMPT, {"object": case["name"]}, [folder / "blind.jpg"],
                                  image_limits=[(1280, 880)], max_output_tokens=1024)
    except ProviderUnavailable:
        raise
    except RuntimeError as error:
        record["error"] = str(error)
    else:
        labels = case["label_map"]
        record["scores"] = {name: getattr(answer, letter).model_dump() for letter, name in labels.items()}
        record["preference"] = {"trial": "improved", "baseline": "worse", "equal": "similar"}[
            labels.get(answer.preferred, "equal")]
    atomic_json(saved, record)
    return record


def gallery(out, cases, summary):
    def url(path):
        return html.escape(Path(os.path.relpath(path, out)).as_posix(), quote=True)

    result_names = {"improved": "改善", "similar": "相近", "worse": "变差", "pending": "待复核"}
    sections = []
    for background in sorted({c["background"] for c in cases}, key=lambda p: min(
            c["number"] for c in cases if c["background"] == p)):
        sections.append(f'<h2>{html.escape(Path(background).name)}</h2>')
        for c in (item for item in cases if item["background"] == background):
            review = c.get("review", {})
            preference = result_names[review.get("preference", "pending")]
            prefix = f'<section><h3>{c["number"]:02d} · {html.escape(c["name"])} · {preference}</h3>'
            if "trial" not in c:
                sections.append(prefix + f'<p>{html.escape(c["error"])}</p></section>')
                continue
            e = c["effects"]
            figures = []
            for key, title in (("baseline", "回退版"), ("trial", "池化＋局部薄雾")):
                crop = f'cases/{c["number"]:02d}/{key}.jpg'
                score = review.get("scores", {}).get(key)
                evidence = ""
                if score:
                    grades = "/".join(str(score[k]) for k in ("fog", "clarity", "fusion", "structure"))
                    evidence = f'<p>{grades} · {html.escape(score["evidence"])}</p>'
                figures.append(f'<figure><a href="{url(c[key])}"><img src="{crop}" loading="lazy" alt="{title}">'
                               f'{title}原尺寸 PNG</a>{evidence}</figure>')
            metrics = (f'池化 {e["pool_weight_max"]:.1%} · 基础雾化 {e["fog_base"]:.1%} · '
                       f'合计雾化中位数 {e["fog_total_median"]:.1%} · 长轴 {c["audit"]["measured_major_px"]} px')
            reason = html.escape(review.get("error", ""))
            sections.append(prefix + f'<p>{metrics}</p><div>{"".join(figures)}</div><p>{reason}</p></section>')
    counts = " · ".join(f'{name} {summary["preference"][key]}' for key, name in result_names.items())
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>薄雾与池化对照</title><style>'
            'body{font:16px system-ui,sans-serif;background:#192329;color:#edf3f5;margin:24px}a{color:#a9dfca}'
            'section{max-width:1100px;padding:12px 0;border-bottom:1px solid #41555d}section div{display:grid;'
            'grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%;display:block}'
            'h2{font-size:1.2rem;overflow-wrap:anywhere}</style><h1>5 件 × 2 张背景 · 局部薄雾与轻量池化</h1>'
            f'<p>{counts}。已生成 {summary["images"]}/10 张，程序检查 {summary["program_pass"]}/10 通过。</p>'
            f'<p>{html.escape(label(summary["cost_total"]))}</p>'
            '<p>图下评分顺序为雾感、清晰度、融合、结构。</p>'
            '<p><a href="summary.json">验证摘要</a> · <a href="cost-ledger.json">成本账本</a></p>')
    (out / "index.html").write_text(page + "".join(sections), encoding="utf8")


async def run(args):
    start = time.monotonic()
    out, rollback, old, sized = (resolve(p) for p in (args.out, args.rollback, args.baseline, args.size_controls))
    if out.exists() and any(out.iterdir()) and not (out / "inputs.json").is_file():
        raise ValueError("Use a fresh output directory for the fixed dust trial")
    out.mkdir(parents=True, exist_ok=True)
    with batch_lock(out):
        return await run_locked(args, start, out, rollback, old, sized)


async def run_locked(args, start, out, rollback, old, sized):
    config = BatchConfig(backgrounds=".", output=str(out), provider_config=args.provider_config,
                         offline=args.offline, photo_style="dusty_scrap", max_iterations=0)
    cache = resolve(config.cache)
    history = {int(r["job_id"].split("-")[0]): r for r in records(old)}
    rows = sorted((r for r in read(rollback / "cases.json") if r["asset_id"] in ASSETS), key=lambda r: r["number"])
    backgrounds = {r["background"] for r in rows}
    if len(rows) != 10 or len(backgrounds) != 2 or {(r["asset_id"], r["background"]) for r in rows} != {
            (asset, bg) for asset in ASSETS for bg in backgrounds}:
        raise ValueError("Expected five specified objects in each of the two archived backgrounds")
    catalog = load_catalog()
    hashes = {key: a["mesh_hash"] for key, a in catalog.items() if a["available"]}
    if hashes != read(rollback / "validation.json")["source_hashes"]:
        raise ValueError("Source models differ from the frozen cohort")
    for row in rows:
        asset = read(cache / "assets" / row["asset_signature"] / "result.json")
        if asset["mesh_hash"] != hashes[row["asset_id"]]:
            raise ValueError("Archived asset metadata does not match the source model")
        # The batch supplies the asset's classification at composition time, after saving placement.
        row["pose"]["material_classes"] = asset["semantic"]["material_classes"]
    passes = {row["number"]: frozen_passes(row, history, sized, cache) for row in rows}
    inputs = {"cohort": file_hash(rollback / "cases.json"),
              "sources": {name: file_hash(HERE / name) for name in
                          (Path(__file__).name, "display_batch/compose.py", "display_batch/local.py",
                           "display_batch/photography.py")},
              "provider": file_hash(args.provider_config or HERE / "batch-config.example.json"),
              "cases": [{"number": r["number"], "asset_id": r["asset_id"], "background_hash": r["background_hash"],
                         "image_hash": r["image_hash"], "material_classes": r["pose"]["material_classes"],
                         "cached_passes": passes[r["number"]][3]} for r in rows]}
    frozen = out / "inputs.json"
    if frozen.is_file() and read(frozen) != inputs:
        raise ValueError("Frozen trial inputs changed. Use a fresh output directory.")
    atomic_json(frozen, inputs)

    attempted = {call.get("job_id") for p in out.glob("usage-*.json") for call in read(p).get("calls", [])
                 if call.get("role") == ROLE and call.get("state") != "not_sent"}
    run_id = time.time_ns()
    vision = Vision(config, cache / "vision", out / f"usage-{run_id}.json")
    offline = Vision(config.model_copy(update={"offline": True}), cache / "vision", out / f"scenes-{run_id}.json")
    scenes = Scenes(cache, offline)
    cases = []
    composition_hits = 0
    paused = None
    review_start = None
    try:
        for row in rows:
            number = row["number"]
            folder = out / "cases" / f"{number:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if file_hash(row["background"]) != row["background_hash"] or file_hash(row["image"]) != row["image_hash"]:
                    raise ValueError("Archived background or baseline PNG changed")
                scene = await scenes.prepare(row["background"])
                source, meta, shadow_hash, _ = passes[number]
                case, hit = await asyncio.to_thread(recompose, folder, row, scene, source, meta, shadow_hash)
                composition_hits += int(hit)
            except (ValueError, OSError, cv2.error, ProviderUnavailable) as error:
                case = {k: row[k] for k in ("number", "asset_id", "name", "background")}
                case.update(status="failed", error=str(error))
            cases.append(case)
            print(json.dumps({"sample": number, "status": case["status"], "audit": case.get("audit", {}).get("pass"),
                              "error": case.get("error")}, ensure_ascii=False), flush=True)
        scene_hits = offline.cache_hits
        review_start = time.monotonic()
        for case in cases:
            if case["status"] != "review":
                continue
            number = case["number"]
            job = f"sample-{number:02d}"
            asyncio.current_task().set_name(job)
            folder = out / "cases" / f"{number:02d}"
            try:
                service = offline if job in attempted or paused else vision
                case["review"] = await review(service, case, folder)
            except ProviderUnavailable as error:
                if not service.offline:
                    paused = str(error)
                case["review"] = {"error": paused or str(error)}
            atomic_json(folder / "result.json", case)
            print(json.dumps({"sample": number, "preference": case["review"].get("preference", "pending"),
                              "error": case["review"].get("error")}, ensure_ascii=False), flush=True)
    finally:
        await offline.close()
        await vision.close()

    if hashes != {key: file_hash(a["mesh"]) for key, a in catalog.items() if a["available"]}:
        raise ValueError("Source model hashes changed during the trial")
    count = Counter(c.get("review", {}).get("preference", "pending") for c in cases)
    cost = cumulative(out)
    stats = {"seconds": round(time.monotonic() - start, 3), "review_seconds": round(time.monotonic() - review_start, 3),
             "new_renders": 0, "composition_cache_hits": composition_hits, "scene_cache_hits": scene_hits,
             "review_cache_hits": vision.cache_hits + offline.cache_hits - scene_hits,
             "retries": sum(c.get("attempt", 0) > 0 for c in vision.calls),
             "logical_reviews_sent": len({c["request_signature"] for c in vision.calls}), "provider_pause": paused}
    summary = {"cases": len(cases), "images": sum("trial" in c for c in cases),
               "program_pass": sum(c.get("audit", {}).get("pass", False) for c in cases),
               "archive_replays_equal": sum(c.get("audit", {}).get("archived_rgb_equal", False) for c in cases),
               "source_models_unchanged": len(hashes), "preference": {k: count[k] for k in
                   ("improved", "similar", "worse", "pending")}, "cost_total": cost, "this_run": stats}
    atomic_json(out / f"run-{run_id}.json", stats)
    atomic_json(out / "cost-ledger.json", {"batch_total": cost, "this_run": stats,
                "runs": [read(p) for p in sorted(out.glob("run-*.json"))],
                "by_case": [{"number": c["number"], "compose_and_audit_seconds": c.get("seconds"),
                             "api_this_run": estimate([a for a in vision.calls if a["job_id"] == f'sample-{c["number"]:02d}'],
                                                      tariff(config, vision.endpoint))} for c in cases]})
    atomic_json(out / "results.json", cases)
    atomic_json(out / "summary.json", summary)
    gallery(out, cases, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT)
    parser.add_argument("--rollback", default=ROLLBACK)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--size-controls", default=SIZED)
    parser.add_argument("--provider-config")
    parser.add_argument("--offline", action="store_true")
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False))
