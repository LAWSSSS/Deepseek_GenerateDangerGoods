"""Frozen A/B/C review of scanned PBR maps and locally tinted overcast lighting."""
import argparse
import asyncio
from collections import Counter
import html
import json
import os
from pathlib import Path
import random
import time
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field

from display_batch.ai import ProviderUnavailable, Vision
from display_batch.common import HERE, atomic_json, file_hash, load_catalog, resolve, signature
from display_batch.compose import compose, cv2, object_window, sample_environment, warp
from display_batch.costs import cumulative, estimate, label, tariff
from display_batch.framing import final_resolution
from display_batch.pipeline import batch_lock
from display_batch.rendering import RenderCache, render_dependencies
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig, Score, Strict
from display_batch.surface_maps import prepare_uv_masks
from display_batch.worker import Blender
from feedback_rollback import ROLLBACK, audit_png, placement_masks
from validate_display_costs import PhotoJudgment, montage
from validate_display_local import read


PILOT = {"13", "22", "31", "33"}
OUTPUT = "outputs/实拍材质环境光-试跑"
PROMPT = """Blind review of fixed-placement inserts in a scrap-hopper security camera. In the board,
'Scene' is the original local crop. Judge each letter on its own against the same local surroundings.
The extra model board is for material and structure reference. Lighting means local direction,
contrast and highlight shape. Material means convincing used metal, worn paint and dust, or
correct dielectric plastic. Texture means matching camera softness, noise and dust veil with natural
surface detail. Contact means weight and coherent contact shadow. Seam means no visible insertion
rectangle or composite boundary. Score 0–4: 3 usable with no conspicuous mismatch, 4 convincing.
List visible issues and set serious for a conspicuous color, highlight, shadow, patch boundary,
metallic plastic or damaged visible structure. Preserve uncertainty for hidden structures.
Apply the same standard to all letters. Keep one short concrete evidence sentence per letter."""


class Judgment(PhotoJudgment):
    contact: Score
    seam: Score
    issues: list[Literal["color", "reflection", "dust", "contact", "seam", "plastic", "structure", "other"]] = Field(max_length=4)

    def passes(self):
        return super().passes() and min(self.contact, self.seam) >= 3


class Pair(Strict):
    A: Judgment
    B: Judgment


class Triplet(Pair):
    C: Judgment


def checked_map(root, entry):
    path = root / entry["path"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Scanned material file changed: {path.name}")
    return str(path)


def scan_library(manifest):
    root = manifest.parent
    data = read(manifest)
    maps = {name: {k: checked_map(root, scan[k]) for k in ("detail", "roughness", "micro_normal")}
            for name, scan in data["materials"].items()}
    hdri = {**data["environment"], "path": checked_map(root, data["environment"])}
    key = signature({name: {k: data["materials"][name][k]["sha256"] for k in paths}
                     for name, paths in maps.items()})
    return {"maps": maps, "signature": key}, hdri


def visible_chroma(image, bounds, visible):
    with Image.open(image) as im:
        rgb = np.float32(im.convert("RGB").crop(tuple(bounds))) / 255
    interior = cv2.erode((visible > .90).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    mask = interior if interior.any() else visible > .5
    return float(np.percentile(np.linalg.norm(lab[..., 1:][mask], axis=1), 85))


def geometry_check(candidate, meta, archived, alpha, bounds, render_dir, offset):
    if meta["camera"] != archived["camera"]:
        raise ValueError("Trial render camera differs from the frozen camera")
    full = candidate["full_px"]
    delta = float(np.linalg.norm(np.subtract(meta["anchor_uv"], archived["anchor_uv"])) * full)
    if len(meta["contact_polygons"]) != len(archived["contact_polygons"]):
        raise ValueError("Trial render contact cluster count changed")
    contact_delta = max((float(np.linalg.norm(np.subtract(a, b), axis=1).max() * full)
                         for a, b in zip(meta["contact_polygons"], archived["contact_polygons"])), default=0.)
    with Image.open(render_dir / "object.png") as im:
        source = np.asarray(im.getchannel("A"), np.float32) / 255
    rendered = warp(source, full, offset, bounds, window=object_window(meta)) > .5
    original = alpha > .5
    iou = float((rendered & original).sum() / max(1, (rendered | original).sum()))
    return {"render_alpha_iou": iou, "render_anchor_delta_px": delta, "contact_delta_px": contact_delta,
            "pass": iou >= .975 and max(delta, contact_delta) <= 1.5}


def make_boards(folder, row, variants, scene, bounds):
    background = folder / "scene.jpg"
    image = scene["frame"].crop(tuple(bounds))
    image.thumbnail((1000, 700))
    image.save(background, quality=94)
    order = sorted(variants)
    random.Random(signature(row["seed"], row["number"], tuple(order))).shuffle(order)
    aliases = dict(zip("ABC", order))
    montage([("Scene", background)] + [(blind, variants[name]["context"]) for blind, name in aliases.items()],
            folder / "blind.jpg")
    montage([("现场局部", background)] + [(key + " " + {"A": "回退", "B": "实拍材质", "C": "实拍材质与环境光"}[key],
            value["context"]) for key, value in sorted(variants.items())], folder / "compare.jpg", large=True)
    return aliases


async def render_case(out, row, profiles, asset, scene, render_cache, scan, hdri, mask_root):
    folder = out / "cases" / f'{row["number"]:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    candidate = {**row["pose"], "fixed_alpha_dir": row["archived_render"],
                 "fixed_alpha_window": row["archived_window"]}
    bounds = row["baseline"]["edit_bounds"]
    alpha, foreground, visible, offset = placement_masks(
        scene, candidate, row["archived_render"], row["archived_window"], row["archived_meta"], bounds)
    variants = {"A": {"image": row["image"], "context": row["context"], "image_hash": row["image_hash"],
                      "audit": row["audit"], "chroma_p85": visible_chroma(row["image"], bounds, visible)}}
    photo_env, samples = await asyncio.to_thread(sample_environment, scene, candidate, local=True)
    partition, mask_hit = await asyncio.to_thread(prepare_uv_masks, asset, mask_root)
    scan_profile = {**scan, "partitions": partition, "signature": signature(scan["signature"], partition)}
    params = row["params"]
    baseline_light = photo_env
    if row["archived_window"] is None:
        baseline_light, _ = await asyncio.to_thread(sample_environment, scene, candidate)
    for profile in profiles:
        saved = folder / f"{profile}.json"
        if saved.is_file():
            value = read(saved)
            if file_hash(value["image"]) != value["image_hash"]:
                raise ValueError(f"Trial image changed: {value['image']}")
            variants[profile] = value
            continue
        start = time.monotonic()
        light = baseline_light if profile == "B" else photo_env
        directory, meta, hit = await render_cache.get(
            asset, candidate, params, {**light, "dust_color": photo_env["dust_color"], "haze": 0.},
            scene["size"], final_resolution(candidate), appearance="local", surface_scans=scan_profile,
            scanned_hdri=hdri if profile == "C" else None)
        geometry = geometry_check(candidate, meta, row["archived_meta"], alpha, bounds, directory, offset)
        anchored = {**meta, **{k: row["archived_meta"][k] for k in ("anchor_uv", "contact_polygons")}}
        comp = await asyncio.to_thread(compose, scene, candidate, directory, anchored, photo_env, samples,
                                       params, row["seed"], folder / (profile + ".png"))
        check = audit_png(scene, comp, alpha, foreground, visible, candidate, offset)
        comp["chroma_p85"] = visible_chroma(comp["image"], bounds, visible)
        chroma_limit = max(variants["A"]["chroma_p85"], comp["photo"]["local_chroma_cap"]) + 2
        check.update(raw_geometry=geometry, chroma_limit=chroma_limit, chroma_ok=comp["chroma_p85"] <= chroma_limit)
        check["pass"] &= geometry["pass"] and check["chroma_ok"]
        comp.update(audit=check, context=str(Path(comp["image"]).with_suffix(".context.jpg")),
                    image_hash=file_hash(comp["image"]), render_cache=meta["render_cache"],
                    render_timings=meta["render_timings"], scanned_materials=meta["scanned_materials"],
                    mask_cache_hit=mask_hit, shadow_fallback=meta.get("shadow_catcher_fallback"),
                    seconds=round(time.monotonic() - start, 3), render_cache_hit=hit)
        atomic_json(saved, comp)
        variants[profile] = comp
        print(json.dumps({"sample": row["number"], "profile": profile, "audit": check["pass"],
                          "iou": geometry["render_alpha_iou"], "shading_slots": len(comp["scanned_materials"]),
                          "shadow": comp["shadow_basis"], "cache": hit}, ensure_ascii=False), flush=True)
    aliases = make_boards(folder, row, variants, scene, photo_env["local_context"]["bounds"])
    result = {"number": row["number"], "asset_id": row["asset_id"], "name": row["name"],
              "background": row["background"], "variants": variants, "status": "review",
              "label_map": aliases, "board": str(folder / "compare.jpg"),
              "lighting": "local_luma_and_tint_overcast_azimuth_unresolved",
              "material_partition": partition["path"]}
    atomic_json(folder / "images.json", result)
    return result


async def compare(vision, case, asset, folder):
    evidence = folder / f'judgment-{"".join(sorted(case["variants"]))}.json'
    images = {k: v["image_hash"] for k, v in case["variants"].items()}
    if evidence.is_file():
        saved = read(evidence)
        if saved["images"] != images:
            raise ValueError("Blind review does not match the saved PNGs")
        if saved.get("error"):
            raise RuntimeError(saved["error"])
        return saved["opinions"]
    triplet = len(case["variants"]) == 3
    pictures = [folder / "blind.jpg"]
    limits = [(1280, 880)]
    if asset.get("board") and Path(asset["board"]).is_file():
        pictures.append(Path(asset["board"]))
        limits.append((400, 400))
    try:
        answer = await vision.ask("scanned-pbr-blind", Triplet if triplet else Pair, PROMPT,
                                  {"object": asset["name"], "materials": asset["semantic"]["material_classes"]},
                                  pictures, image_limits=limits, max_output_tokens=1200)
    except ProviderUnavailable:
        raise
    except RuntimeError as error:
        atomic_json(evidence, {"images": images, "error": str(error)})
        raise
    opinions = {name: {**getattr(answer, alias).model_dump(), "pass": getattr(answer, alias).passes()}
                for alias, name in case["label_map"].items()}
    atomic_json(evidence, {"images": images, "opinions": opinions})
    return opinions


def count_passes(cases, profile):
    return Counter(c["asset_id"] for c in cases if c["variants"][profile]["audit"]["pass"]
                   and c.get("opinions", {}).get(profile, {}).get("pass"))


def passes_gate(cases, profile, population, threshold):
    if len(cases) != population:
        return False
    counts = count_passes(cases, profile)
    return (sum(counts.values()) >= threshold and set(counts) == {c["asset_id"] for c in cases}
            and all(c["variants"][profile]["audit"]["pass"] and
                    not c.get("opinions", {}).get(profile, {}).get("serious", True) for c in cases))


def render_seconds(case, profile):
    timings = case["variants"][profile].get("render_timings", {})
    return sum(timings.get(k, 0.) for k in ("object_seconds", "with_shadow_seconds", "without_shadow_seconds"))


def gallery(out, cases, summary):
    def url(path):
        return html.escape(Path(os.path.relpath(path, out)).as_posix(), quote=True)

    sections = []
    for case in sorted(cases, key=lambda c: c["number"]):
        rows = []
        for key, item in sorted(case["variants"].items()):
            opinion = case.get("opinions", {}).get(key)
            ratings = "/".join(str(opinion[k]) for k in ("material", "lighting", "texture", "contact", "seam")) if opinion else "待复核"
            message = opinion["evidence"] if opinion else case.get("review_error", "")
            rows.append(f'<li><a href="{url(item["image"])}">{key} 原尺寸 PNG</a> · {ratings}'
                        f' · C85 {item["chroma_p85"]:.1f} · {html.escape(message)}</li>')
        sections.append(f'<section id="sample-{case["number"]:02d}"><h2>{case["number"]:02d} · '
                        f'{html.escape(case["name"])}</h2><img src="{url(case["board"])}" loading="lazy">'
                        f'<ul>{"".join(rows)}</ul></section>')
    counts = " · ".join(f"{k} {v}/8" for k, v in summary["pilot_photo_pass"].items())
    status = {"pilot_below_gate": "小样未达扩批门槛，批量使用回退版",
              "awaiting_style_review": "24 项已过门槛，等待整体风格确认",
              "cohort_below_gate": "24 项复验未过门槛，批量使用回退版",
              "paused_provider": "视觉服务暂停，保留生成记录"}.get(summary["state"], "生成中")
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>同摆位实拍材质对照</title>'
        '<style>body{margin:26px;background:#172128;color:#eee;font:16px sans-serif}'
        'a{color:#a8ddc3}img{max-width:100%;height:auto}section{max-width:1150px;padding:10px 0}'
        'li{padding:5px 0}</style><h1>实拍材质与环境光 · 同摆位盲评</h1>'
        f'<p>{status}。小样摄影达标 {counts}。</p><p>{html.escape(label(summary["cost_total"]))}</p>'
        '<p>评分依次为材质、受光、摄影质感、接地、插入边框。C85 为可见物体内部的 Lab 色度 85 分位。</p>'
        '<p><a href="summary.json">验证摘要</a> · <a href="cost-ledger.json">成本账本</a></p>'
        + "".join(sections), encoding="utf8")


async def run(args):
    started = time.monotonic()
    out, rollback, library = (resolve(p) for p in (args.out, args.rollback, args.library))
    if out.exists() and any(out.iterdir()) and not (out / "inputs.json").is_file():
        raise ValueError("Use a new directory for the scanned-material trial")
    out.mkdir(parents=True, exist_ok=True)
    with batch_lock(out):
        return await run_locked(args, started, out, rollback, library)


async def run_locked(args, started, out, rollback, library):
    rows = read(rollback / "cases.json")
    hashes = read(rollback / "validation.json")["source_hashes"]
    catalog = load_catalog()
    actual_hashes = {key: a["mesh_hash"] for key, a in catalog.items() if a["available"]}
    if hashes != actual_hashes or len(rows) != 24:
        raise ValueError("Rollback cohort or source model content changed")
    scan, hdri = scan_library(library)
    config = BatchConfig(backgrounds=".", output=str(out), provider_config=args.provider_config,
                         offline=args.offline, photo_style="dusty_scrap", max_iterations=0)
    cache = resolve(config.cache)
    signature_inputs = {
        "cohort": file_hash(rollback / "cases.json"), "scans": scan["signature"], "hdri": hdri["sha256"],
        "render": render_dependencies(), "sources": {name: file_hash(HERE / name) for name in
            (Path(__file__).name, "display_batch/compose.py", "display_batch/surface_maps.py", "display_batch/photography.py")},
        "pilot": [r["number"] for r in rows if r["asset_id"] in PILOT], "provider_config": args.provider_config,
    }
    inputs = out / "inputs.json"
    if inputs.is_file() and read(inputs) != signature_inputs:
        raise ValueError("Frozen trial inputs changed. Use a new output directory.")
    atomic_json(inputs, signature_inputs)
    run_id = time.time_ns()
    vision = Vision(config, cache / "vision", out / f"usage-{run_id}.json")
    offline = Vision(config.model_copy(update={"offline": True}), cache / "vision", out / f"scenes-{run_id}.json")
    scenes = Scenes(cache, offline)
    worker = Blender(out / f"worker-{run_id}", config.render_device)
    render_cache = RenderCache(cache, worker)
    results = {}
    review_failures = {}
    state = "pilot_below_gate"
    preferred = None

    async def check(row, profiles):
        number = row["number"]
        asyncio.current_task().set_name(f"sample-{number:02d}")
        folder = out / "cases" / f"{number:02d}"
        ident = row["asset_id"]
        asset = read(cache / "assets" / row["asset_signature"] / "result.json")
        if (asset["mesh_hash"] != hashes[ident] or file_hash(row["background"]) != row["background_hash"]
                or file_hash(row["image"]) != row["image_hash"]):
            raise ValueError(f"Frozen evidence changed for sample {number:02d}")
        scene = await scenes.prepare(row["background"])
        case = await render_case(out, row, profiles, asset, scene, render_cache, scan, hdri, cache / "scanned-uv")
        results[number] = case
        try:
            case["opinions"] = await compare(vision, case, asset, folder)
        except RuntimeError as error:
            case["review_error"] = str(error)
            review_failures[number] = str(error)
            if isinstance(error, ProviderUnavailable) and not vision.offline:
                raise
        atomic_json(folder / "result.json", case)
        print(json.dumps({"sample": number, "blind_photo_pass": {k: v["pass"] for k, v in case.get("opinions", {}).items()}},
                         ensure_ascii=False), flush=True)

    try:
        for row in rows:
            if row["asset_id"] in PILOT:
                await check(row, ("B", "C"))
        pilot = list(results.values())
        passing = [p for p in ("B", "C") if passes_gate(pilot, p, 8, 7)]
        if passing:
            preferred = max(passing, key=lambda p: (sum(count_passes(pilot, p).values()),
                            -sum(render_seconds(c, p) for c in pilot), p == "B"))
            for row in rows:
                if row["number"] not in results:
                    await check(row, (preferred,))
            state = "awaiting_style_review" if passes_gate(list(results.values()), preferred, 24, 20) else "cohort_below_gate"
    except ProviderUnavailable as error:
        state = "paused_provider"
        print(str(error), flush=True)
    finally:
        await worker.close()
        await offline.close()
        await vision.close()

    if hashes != {key: file_hash(a["mesh"]) for key, a in catalog.items() if a["available"]}:
        raise ValueError("Source model hashes changed during the trial")
    cases = list(results.values())
    pilot = [c for c in cases if c["asset_id"] in PILOT]
    cost = cumulative(out)
    summary = {"state": state, "profile": preferred, "cases": len(cases), "source_models_unchanged": len(hashes),
               "pilot_photo_pass": {key: sum(count_passes(pilot, key).values()) for key in ("A", "B", "C")},
               "cohort_photo_pass": sum(count_passes(cases, preferred).values()) if preferred else None,
               "program_pass": {key: sum(c["variants"][key]["audit"]["pass"] for c in cases if key in c["variants"])
                                for key in ("A", "B", "C")},
               "review_failures": review_failures, "cost_total": cost}
    run_stats = {"seconds": round(time.monotonic() - started, 3), "render_counts": dict(render_cache.totals),
                 "render_timings": dict(render_cache.timings), "worker_seconds": worker.seconds,
                 "vision_cache_hits": vision.cache_hits, "scene_cache_hits": offline.cache_hits,
                 "retries": sum(c.get("attempt", 0) > 0 for c in vision.calls),
                 "logical_reviews": len({c["request_signature"] for c in vision.calls}), "outcome": state}
    atomic_json(out / f"run-{run_id}.json", run_stats)
    by_case = [{"number": c["number"], "render_seconds": {p: render_seconds(c, p) for p in c["variants"]},
                "api_this_run": estimate([x for x in vision.calls if x["job_id"] == f'sample-{c["number"]:02d}'],
                                         tariff(config, vision.endpoint))} for c in cases]
    atomic_json(out / "cost-ledger.json", {"batch_total": cost, "this_run": run_stats, "by_case": by_case,
                "preparation": read(library.parent / "preparation.json"),
                "runs": [read(p) for p in sorted(out.glob("run-*.json"))]})
    atomic_json(out / "summary.json", {**summary, "this_run": run_stats})
    atomic_json(out / "results.json", cases)
    gallery(out, cases, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT)
    parser.add_argument("--rollback", default=ROLLBACK)
    parser.add_argument("--library", default="work/render-materials/polyhaven/library.json")
    parser.add_argument("--provider-config")
    parser.add_argument("--offline", action="store_true")
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False))
