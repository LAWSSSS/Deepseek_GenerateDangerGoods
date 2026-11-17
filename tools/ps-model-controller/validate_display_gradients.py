"""Compare single-variable pooling and haze levels on fixed full-frame composites."""
import argparse
import asyncio
import html
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from display_batch.ai import Vision
from display_batch.common import HERE, atomic_json, file_hash, load_catalog, resolve
from display_batch.compose import compose, cv2, object_window, sample_environment, srgb, warp
from display_batch.costs import cumulative
from display_batch.framing import final_resolution
from display_batch.local import camera_residual, foreground_dust
from display_batch.photography import LUMA, linear
from display_batch.pipeline import batch_lock
from display_batch.rendering import RenderCache, render_dependencies
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig
from display_batch.surface_maps import prepare_uv_masks
from display_batch.worker import Blender
from feedback_rollback import BASELINE, ROLLBACK, SIZED, audit_png, placement_masks
from validate_display_dust import ASSETS, frozen_passes, read_rgb
from validate_display_local import read, records
from validate_display_scanned import geometry_check


OUTPUT = "outputs/薄雾池化梯度与积灰哑光-20261111-v2"
POOL = (.15, .30, .45)
FOG = (1 / 3, 2 / 3, 1.)


async def matte_render(row, asset, scene, source, archived, render_cache, cache, environment, alpha, offset):
    pose = row["pose"]
    legacy = row["archived_window"] is None
    resolution = archived["resolution"] if legacy else final_resolution(pose)
    lighting = sample_environment(scene, pose)[0] if legacy else {**environment, "haze": 0.}
    lighting = {**lighting, "dust_color": environment["dust_color"]}
    partitions, _ = await asyncio.to_thread(prepare_uv_masks, asset, cache / "scanned-uv")
    directory, meta, hit = await render_cache.get(
        asset, pose, row["params"], lighting, scene["size"], resolution,
        appearance="local", full_frame=legacy, matte_partitions=partitions, object_only=True)
    geometry = geometry_check(pose, meta, archived, alpha, row["baseline"]["edit_bounds"], directory, offset)
    if not geometry["pass"] or not meta["matte_materials"]:
        raise ValueError(f"Matte object failed fixed geometry or material coverage: {geometry}")
    meta["anchor_uv"] = archived["anchor_uv"]
    meta["contact_polygons"] = archived["contact_polygons"]
    meta["shadow_window"] = archived.get("shadow_window")
    for part in ("with_shadow", "without_shadow"):
        meta["part_paths"][part] = archived.get("part_paths", {}).get(part, str(source))
    if legacy:
        meta.pop("object_window", None)
    return directory, meta, {"geometry": geometry, "materials": meta["matte_materials"], "render_cache_hit": hit,
                             "render_timings": meta["render_timings"]}


def specular_p99(directory, meta, pose, offset, bounds, visible):
    with Image.open(directory / "object.png") as im:
        coverage = np.asarray(im.getchannel("A"), np.float32) / 255
    data = cv2.imread(str(directory / "specular0001.exr"), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError("Missing specular EXR")
    full = pose["full_px"]
    window = object_window(meta)
    rgb = warp(data[..., :3][..., ::-1], full, offset, bounds, window=window)
    alpha = warp(coverage, full, offset, bounds, window=window)
    core = (visible > .9) & (alpha > .9)
    return float(np.percentile((rgb @ LUMA)[core] / alpha[core], 99))


def full_preview(rgb, path):
    image = Image.fromarray(rgb)
    image.thumbnail((1600, 900))
    image.save(path, quality=91)


def frame_stats(rgb, bounds, core):
    x, y, r, b = bounds
    luminance = (np.float32(rgb[y:b, x:r]) @ LUMA)
    dx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, scale=.125)
    dy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, scale=.125)
    return {"luma_p95": float(np.percentile(luminance[core], 95)),
            "edge_p80": float(np.percentile(cv2.magnitude(dx, dy)[core], 80))}


def apply_camera(layers, effect, environment, params, seed):
    value, _, optics = camera_residual(layers["shaded"], layers["rgb"], layers["alpha"], layers["background"],
                                        environment, params, seed, layers["writable"], object_delta=effect,
                                        fixed_sensor=layers.get("fixed_sensor"))
    return np.uint8(np.rint(np.clip(value, 0, 1) * 255)), optics


def limit_matte_brightness(layers, appearance, environment, params, seed, core, target):
    """Trim only object radiance if the matte surface would exceed the control's upper brightness."""
    clear = linear(appearance["clear_rgb"])
    veiled = linear(appearance["rgb"])
    reference = linear(layers["rgb"])
    visible_clear = clear * (1 - layers["air"]["amount"])

    def luma(gain):
        delta = veiled - reference + (gain - 1) * visible_clear
        rgb, _ = apply_camera(layers, delta, environment, params, seed)
        return float(np.percentile((np.float32(rgb) @ LUMA)[core], 95))

    if luma(1.) <= target:
        return appearance, 1.
    low, high = 0., 1.
    for _ in range(8):
        gain = (low + high) / 2
        if luma(gain) <= target:
            low = gain
        else:
            high = gain
    return {**appearance, "clear_rgb": srgb(clear * low),
            "rgb": srgb(veiled + (low - 1) * visible_clear)}, low


def export_effect(folder, name, title, original, baseline, layers, effect, bounds, environment, params, seed, core):
    x, y, r, b = bounds
    visible = layers["alpha"] > 0
    value, optics = apply_camera(layers, effect, environment, params, seed)
    for k in ("blur_sigma", "added_noise_sigma", "existing_noise_sigma", "optical_iou"):
        if optics[k] != baseline["photo"][k]:
            raise ValueError("Camera blur or noise changed in a single-variable comparison")
    rgb = original.copy()
    rgb[y:b, x:r][visible] = value[visible]
    image = folder / (name + ".png")
    Image.fromarray(rgb).save(image, compress_level=2)
    exported = read_rgb(image)
    changed = np.any(exported != original, axis=2)
    outside = int(changed.sum() - (changed[y:b, x:r] & visible).sum())
    if exported.shape != original.shape or not np.array_equal(rgb, exported) or outside:
        raise ValueError("Full-frame PNG or protected pixels failed verification")
    preview = image.with_suffix(".jpg")
    full_preview(exported, preview)
    return {"image": str(image), "preview": str(preview), "title": title,
            "image_hash": file_hash(image), "changed_pixels": int(changed.sum()), "outside_visible": outside,
            "size": [exported.shape[1], exported.shape[0]], **frame_stats(exported, bounds, core)}


def compose_case(folder, row, scene, source, archived, shadow_hash, matte, environment, samples, masks):
    start = time.monotonic()
    pose, bounds = row["pose"], row["baseline"]["edit_bounds"]
    alpha, foreground, visible, offset = masks
    core = cv2.erode((visible > .9).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    original = read_rgb(row["image"])
    original_stats = frame_stats(original, bounds, core)
    args = (scene, pose, source, archived, environment, samples, row["params"], row["seed"])
    matte_data = None
    with tempfile.TemporaryDirectory(prefix="composition-", dir=folder) as tmp:
        a = compose(*args, Path(tmp) / "A.png", capture_local=True)
        if not np.array_equal(original, read_rgb(a["image"])) or a["shadow_hash"] != shadow_hash:
            raise ValueError("Archived baseline pixels or shadow changed on replay")
        base_audit = audit_png(scene, a, alpha, foreground, visible, pose, offset)
        if not base_audit["pass"] or any(a[k] != row["baseline"][k] for k in
                                        ("alpha_hash", "visible_alpha_hash", "foreground_mask_hash")):
            raise ValueError("Archived baseline failed geometry or protection checks")
        layers = a.pop("local_layers")
        appearance = layers
        material_delta = np.zeros_like(layers["rgb"])
        if matte:
            directory, meta, matte_data = matte
            frozen = {**pose, "fixed_alpha_dir": str(source), "fixed_alpha_window": row["archived_window"]}
            b = compose(scene, frozen, directory, meta, environment, samples, row["params"], row["seed"],
                        Path(tmp) / "matte.png", capture_local=True, fixed_tone=a["photo"])
            if any(b[k] != a[k] for k in ("alpha_hash", "visible_alpha_hash", "foreground_mask_hash", "shadow_hash")):
                raise ValueError("Matte render moved coverage or shadows")
            appearance = b.pop("local_layers")
            appearance, gain = limit_matte_brightness(
                layers, appearance, environment, row["params"], row["seed"], core, original_stats["luma_p95"])
            material_delta = linear(appearance["rgb"]) - linear(layers["rgb"])
            before = specular_p99(source, archived, pose, offset, bounds, visible)
            after = specular_p99(directory, meta, pose, offset, bounds, visible)
            matte_data.update(specular_p99_before=before, specular_p99_after=after, matte_luminance_gain=gain,
                              specular_reduction=1 - after / before if before > 1e-6 else None)
    base_preview = folder / "archived.jpg"
    full_preview(original, base_preview)
    control = {"image": row["image"], "preview": str(base_preview), "image_hash": row["image_hash"],
               "title": "回退版", "outside_visible": 0, "size": list(scene["size"]),
               **original_stats}
    export_args = (original, a, layers)
    fixed_args = (bounds, environment, row["params"], row["seed"], core)
    zero = (export_effect(folder, "matte", "积灰哑光", *export_args, material_delta, *fixed_args)
            if matte else control)
    if matte_data is not None:
        matte_data["luma_p95_before"] = control["luma_p95"]
        matte_data["luma_p95_after"] = zero["luma_p95"]
        if zero["luma_p95"] > control["luma_p95"] + .05:
            raise ValueError("Matte surface remains brighter than its reference")
    zero.update(pool_weight=0., fog_fraction=0., fog_extra=0., fog_total=layers["air"]["amount"])
    groups = {"pool": [zero], "fog": [zero]}
    for kind, levels in (("pool", POOL), ("fog", FOG)):
        for i, value in enumerate(levels, 1):
            pool, fog = (value, 0.) if kind == "pool" else (0., value)
            delta, effects = foreground_dust(appearance["clear_rgb"], appearance["rgb"], visible, environment,
                                              layers["air"], pose["target_px"], pool_weight=pool, fog_fraction=fog)
            title = f'池化 {pool:.0%}' if kind == "pool" else f'追加薄雾 {i} 档'
            item = export_effect(folder, f"{kind}-{i}", title, *export_args, material_delta + delta, *fixed_args)
            item.update(pool_weight=effects["pool_weight_max"], fog_fraction=fog,
                        fog_extra=effects["fog_extra_max"], fog_total=effects["fog_total_median"])
            groups[kind].append(item)
        if len({item["image_hash"] for item in groups[kind]}) != 4:
            raise ValueError(f"Repeated levels in {kind} comparison")
    result = {"number": row["number"], "asset_id": row["asset_id"], "name": row["name"],
              "background": row["background"], "control": control, "matte": matte_data,
              "groups": groups, "audit": base_audit, "status": "review",
              "material_profile": "matte_dust" if matte else "archived",
              "seconds": round(time.monotonic() - start, 3)}
    atomic_json(folder / "result.json", result)
    return result


def output_frames(case):
    frames = [case["control"]]
    for group in case["groups"].values():
        frames.extend(group)
    return {item["image"]: item for item in frames}


def report(out, cases, summary):
    def link(path):
        try:
            value = Path(os.path.relpath(path, out)).as_posix()
        except ValueError:
            value = Path(path).as_uri()
        return html.escape(value, quote=True)

    def picture(item, title):
        return (f'<figure><a href="{link(item["image"])}"><img src="{link(item["preview"])}" '
                f'loading="lazy" width="1600" height="900" alt="{html.escape(title)}"></a>'
                f'<figcaption>{html.escape(title)} · <a href="{link(item["image"])}">原尺寸 PNG</a>'
                '</figcaption></figure>')

    blocks = []
    for background in sorted({c["background"] for c in cases},
                             key=lambda p: min(c["number"] for c in cases if c["background"] == p)):
        blocks.append(f'<h2>{html.escape(Path(background).name)}</h2>')
        for c in sorted((c for c in cases if c["background"] == background), key=lambda c: c["number"]):
            blocks.append(f'<section id="sample-{c["number"]:02d}"><h3>{c["number"]:02d} · {html.escape(c["name"])}</h3>')
            if "error" in c:
                blocks.append(f'<p>{html.escape(c["error"])}</p></section>')
                continue
            zero = c["groups"]["pool"][0]
            if c["matte"]:
                images = picture(c["control"], "回退材质") + picture(zero, "积灰哑光")
                blocks.append(f'<details open><summary>只改金属表面</summary><div class="frames">{images}</div></details>')
            for kind, title in (("pool", "池化梯度 · 附加薄雾为零"), ("fog", "薄雾梯度 · 池化为零")):
                images = []
                for item in c["groups"][kind]:
                    if kind == "pool":
                        caption = f'池化 {item["pool_weight"]:.0%}'
                    else:
                        caption = f'新增雾量峰值 {item["fog_extra"]:.1%} · 总雾量中位数 {item["fog_total"]:.1%}'
                    images.append(picture(item, caption))
                blocks.append(f'<details><summary>{title}</summary><div class="frames">{"".join(images)}</div></details>')
            blocks.append('</section>')
    heading = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
               '<title>池化、薄雾与积灰哑光对照</title><style>'
               'body{font:16px system-ui,sans-serif;background:#192329;color:#edf3f5;margin:24px}'
               'a{color:#a9dfca}h2{font-size:1.15rem;overflow-wrap:anywhere}section{max-width:1500px;padding:12px 0;'
               'border-bottom:1px solid #41555d}.frames{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}'
               'figure{margin:0}img{width:100%;height:auto;object-fit:contain;display:block}figcaption{padding:7px 0}'
               'summary{cursor:pointer;padding:8px 0}@media(max-width:780px){.frames{grid-template-columns:1fr}}</style>'
               '<h1>池化、薄雾与积灰哑光对照</h1><p>每张预览均为完整监控画面，点击查看 3840×2160 原图。</p>'
               '<p>金属两组梯度共用积灰哑光基线。塑料胶罐使用回退材质。额外薄雾按落点条件分档。</p>'
               f'<p>{summary["cases_pass"]}/10 组通过程序检查 · {summary["full_frames"]} 张全景 · API 费用 ¥0</p>'
               '<p><a href="summary.json">验证与成本</a> · <a href="results.json">逐组测量</a></p>')
    (out / "index.html").write_text(heading + "".join(blocks), encoding="utf8")


async def run(args):
    start = time.monotonic()
    out = resolve(args.out)
    if out.exists() and any(out.iterdir()) and not (out / "inputs.json").is_file():
        raise ValueError("Use a fresh directory for the single-variable comparison")
    out.mkdir(parents=True, exist_ok=True)
    with batch_lock(out):
        return await run_locked(args, start, out)


async def run_locked(args, start, out):
    rollback = resolve(ROLLBACK)
    config = BatchConfig(backgrounds=".", output=str(out), offline=True, render_device=args.device,
                         photo_style="dusty_scrap", max_iterations=0)
    cache = resolve(config.cache)
    catalog = load_catalog()
    hashes = {k: v["mesh_hash"] for k, v in catalog.items() if v["available"]}
    if hashes != read(rollback / "validation.json")["source_hashes"]:
        raise ValueError("Frozen source models changed")
    rows = [row for row in read(rollback / "cases.json") if row["asset_id"] in ASSETS]
    backgrounds = {row["background"] for row in rows}
    if len(rows) != 10 or len(backgrounds) != 2 or {(r["asset_id"], r["background"]) for r in rows} != {
            (asset, bg) for asset in ASSETS for bg in backgrounds}:
        raise ValueError("Expected exactly five assets across both frozen backgrounds")
    history = {int(r["job_id"].split("-")[0]): r for r in records(resolve(BASELINE))}
    passes = {row["number"]: frozen_passes(row, history, resolve(SIZED), cache) for row in rows}
    assets = {}
    for row in rows:
        asset = read(cache / "assets" / row["asset_signature"] / "result.json")
        if asset["mesh_hash"] != hashes[row["asset_id"]]:
            raise ValueError("Frozen asset metadata changed")
        assets[row["number"]] = asset
        row["pose"]["material_classes"] = asset["semantic"]["material_classes"]
    inputs = {"cohort": file_hash(rollback / "cases.json"), "pool": list(POOL), "fog": list(FOG),
              "device": args.device, "renderer": render_dependencies(),
              "sources": {name: file_hash(HERE / name) for name in
                          (Path(__file__).name, "display_batch/compose.py", "display_batch/local.py",
                           "display_batch/photography.py", "display_batch/surface_maps.py")},
              "passes": {str(i): v[3] for i, v in passes.items()}}
    path = out / "inputs.json"
    if path.is_file() and read(path) != inputs:
        raise ValueError("Frozen comparison inputs changed. Use a fresh output directory.")
    atomic_json(path, inputs)
    run_id = time.time_ns()
    vision = Vision(config, cache / "vision", out / f"usage-{run_id}.json")
    scenes = Scenes(cache, vision)
    worker = Blender(out / f"worker-{run_id}", args.device)
    renders = RenderCache(cache, worker)
    cases = []
    reused = 0
    try:
        for row in sorted(rows, key=lambda r: (int(r["asset_id"]), r["number"])):
            number = row["number"]
            folder = out / "cases" / f"{number:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if file_hash(row["image"]) != row["image_hash"] or file_hash(row["background"]) != row["background_hash"]:
                    raise ValueError("Frozen PNG or background changed")
                saved = folder / "result.json"
                if saved.is_file():
                    case = read(saved)
                    if any(file_hash(item["image"]) != item["image_hash"] for item in output_frames(case).values()):
                        raise ValueError("Saved full-frame comparison changed")
                    reused += 1
                else:
                    scene = await scenes.prepare(row["background"])
                    source, archived, shadow_hash, _ = passes[number]
                    bounds = row["baseline"]["edit_bounds"]
                    masks = placement_masks(scene, row["pose"], source, row["archived_window"], archived, bounds)
                    environment, samples = sample_environment(scene, row["pose"], local=True)
                    matte = None
                    if row["asset_id"] != "13":
                        matte = await matte_render(row, assets[number], scene, source, archived, renders,
                                                   cache, environment, masks[0], masks[3])
                    case = await asyncio.to_thread(compose_case, folder, row, scene, source, archived, shadow_hash,
                                                   matte, environment, samples, masks)
            except (ValueError, OSError, RuntimeError, cv2.error) as error:
                case = {k: row[k] for k in ("number", "asset_id", "name", "background")}
                case.update(status="failed", error=str(error))
            cases.append(case)
            print(json.dumps({"sample": number, "status": case["status"], "matte": case.get("matte"),
                              "error": case.get("error")}, ensure_ascii=False), flush=True)
    finally:
        await worker.close()
        await vision.close()
    if hashes != {k: file_hash(v["mesh"]) for k, v in catalog.items() if v["available"]}:
        raise ValueError("Source model content changed")
    completed = [c for c in cases if c["status"] == "review"]
    frames = {path: item for case in completed for path, item in output_frames(case).items()}
    cost = cumulative(out)
    summary = {"cases_pass": len(completed), "full_frames": len(frames),
               "new_pngs": sum(Path(path).is_relative_to(out) for path in frames),
               "matte_cases": sum(c["matte"] is not None for c in completed), "source_models": len(hashes),
               "new_render_counts": dict(renders.totals), "render_seconds": renders.timings.get("object_seconds", 0.),
               "seconds": round(time.monotonic() - start, 3), "composition_cache_hits": reused, "cost_total": cost}
    atomic_json(out / f"run-{run_id}.json", summary)
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "results.json", cases)
    report(out, cases, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT)
    parser.add_argument("--device", choices=("OPTIX", "CUDA", "CPU"), default="OPTIX")
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(int(result["cases_pass"] != 10))
