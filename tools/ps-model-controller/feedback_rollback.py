"""Republish the previous photography at the confirmed sizes using frozen placement evidence."""
import argparse
import asyncio
import hashlib
import html
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from display_batch.ai import Vision
from display_batch.common import atomic_json, file_hash, load_catalog, resolve, signature
from display_batch.compose import depth_on_image, warp
from display_batch.costs import cumulative, label
from display_batch.pile import depth_order
from display_batch.policy import resize_candidate
from display_batch.review import evidence_status
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig
from validate_display_costs import selected_geometry
from validate_display_local import read, records


BASELINE = "outputs/点评样片-低饱和扬尘-20261111"
SIZED = "outputs/点评样片-废钢材质重贴-20261111-v2"
ROLLBACK = "outputs/回退低饱和点评-20261111"


def placement_masks(scene, candidate, folder, window, meta, bounds):
    x, y, r, b = bounds
    size = candidate["full_px"]
    offset = np.asarray(candidate["anchor_px"]) - np.asarray(meta["anchor_uv"]) * size
    with Image.open(Path(folder) / "object.png") as png:
        source = np.asarray(png.getchannel("A"), np.float32) / 255
    alpha = warp(source, size, offset, bounds, window=window)
    if candidate.get("placement") and not candidate.get("legacy_occlusion"):
        z = depth_on_image(Path(folder) / "depth0001.exr", source, size, offset, bounds, meta["camera"], window)
        foreground, _ = depth_order(scene, candidate, bounds, z, alpha, refine=True)
    else:
        site = next(s for s in scene["supports"] if s["id"] == candidate["site_id"])
        support = np.logical_or.reduce([s["mask"][y:b, x:r] for s in scene["supports"]
                                       if s["id"] in candidate.get("support_ids", [site["id"]])])
        foreground = (scene["depth_native"][y:b, x:r] > site["depth"] + .10) & ~support
    protected = scene["protected"][y:b, x:r]
    foreground &= ~protected
    return alpha, foreground, alpha * (~foreground & ~protected), offset


def audit_png(scene, value, alpha, foreground, visible, candidate, offset):
    with Image.open(value["image"]) as png:
        rgb = np.asarray(png.convert("RGB"))
    frame = np.asarray(scene["frame"])
    if rgb.shape != frame.shape:
        raise ValueError("Saved PNG size differs from its background")
    x, y, r, b = value["edit_bounds"]
    changed = np.any(rgb != frame, axis=2)
    yy, xx = np.nonzero(alpha > .5)
    length = max(int(np.ptp(xx)), int(np.ptp(yy))) + 1
    anchor_error = float(np.linalg.norm(offset - candidate["offset"]))
    alpha_hash = hashlib.sha256(alpha.tobytes()).hexdigest()
    visible_hash = hashlib.sha256(visible.tobytes()).hexdigest()
    foreground_hash = hashlib.sha256(np.packbits(foreground)).hexdigest()
    result = {
        "size": list(scene["size"]), "measured_major_px": length,
        "expected_major_px": candidate["expected_major_px"], "anchor_error_px": anchor_error,
        "protected_pixels": int((changed & scene["protected"]).sum()),
        "foreground_pixels": int((changed[y:b, x:r] & foreground).sum()),
        "alpha_equal": alpha_hash == value["alpha_hash"],
        "visible_alpha_equal": visible_hash == value["visible_alpha_hash"],
        "foreground_equal": foreground_hash == value["foreground_mask_hash"],
        "occlusion_fraction": float(1 - visible.sum() / alpha.sum()),
        "evidence": evidence_status(value["checks"]),
    }
    changed[y:b, x:r] = False
    result["outside_rectangle_pixels"] = int(changed.sum())
    result["pass"] = (result["alpha_equal"] and result["visible_alpha_equal"] and result["foreground_equal"]
                      and anchor_error <= 1.5 and abs(length - candidate["expected_major_px"])
                      <= max(3, .05 * candidate["expected_major_px"]) and result["evidence"] != "fail"
                      and not any(result[k] for k in ("protected_pixels", "foreground_pixels", "outside_rectangle_pixels"))
                      and not value["checks"]["outside_edit"]["changed_pixels"])
    return result


def gallery(out, rows):
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    sheet = Image.new("RGB", (1440, 64 + 190 * ((len(rows) + 3) // 4)), "#172128")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 18), "回退低饱和 · 24 张固定摆位 · 最新尺寸", font=font, fill="white")
    cards = []
    for i, row in enumerate(rows):
        title = f'{row["number"]:02d} · {row["name"]} · {row["audit"]["measured_major_px"]} px'
        left, top = (i % 4) * 360 + 8, (i // 4) * 190 + 58
        draw.text((left + 4, top), title, font=font, fill="white")
        with Image.open(row["context"]) as im:
            im = im.convert("RGB")
            im.thumbnail((344, 151))
            sheet.paste(im, (left + (344 - im.width) // 2, top + 29))
        png = Path(row["image"]).relative_to(out).as_posix()
        crop = Path(row["context"]).relative_to(out).as_posix()
        cards.append(f'<figure><a href="{png}"><img src="{crop}" loading="lazy"></a>'
                     f'<figcaption>{html.escape(title)} · {row["status"]}</figcaption></figure>')
    sheet.save(out / "overview.jpg", quality=93)
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>回退低饱和点评</title>'
        '<style>body{background:#172128;color:#eee;font:16px sans-serif;margin:24px}a{color:#a6dbc3}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}'
        'figure{margin:0;background:#24353d;padding:10px}img{width:100%;height:210px;object-fit:contain}'
        'figcaption{padding:6px}</style><h1>回退低饱和 · 24 张固定摆位</h1>'
        '<p>31 液压缸按原始尺寸 0.375，33 灭火器按 0.25。点击局部查看原尺寸 PNG。</p>'
        f'<p>{html.escape(label(cumulative(out)))}</p><a href="overview.jpg">总览图</a>'
        '<main>' + "".join(cards) + '</main>', encoding="utf8")


async def run(out, old, sized):
    start = time.monotonic()
    if out.exists() and any(out.iterdir()) and not (out / "inputs.json").is_file():
        raise ValueError("Use a new output directory for rollback feedback")
    config = BatchConfig(backgrounds=".", output=str(out), photo_style="dusty_scrap", offline=True)
    out.mkdir(parents=True, exist_ok=True)
    cache = resolve(config.cache)
    previous = records(old)
    latest = {(r["asset_id"], r["background"]): r for r in records(sized)}
    manifest = {j["baseline_job_id"]: j for j in read(sized / "manifest.json")}
    old_jobs = {r["job_id"]: j for r, j in zip(previous, read(old / "inputs.json"), strict=True)}
    if len(previous) != 24 or len(latest) != 24 or len(manifest) != 24:
        raise ValueError("Both prior feedback cohorts must contain the same complete 24 placements")
    inputs = {"old": str(old), "sized": str(sized), "assets": config.asset_scale_factors,
              "baseline_hashes": {j: item["baseline_image_hash"] for j, item in manifest.items()}}
    if (out / "inputs.json").is_file() and read(out / "inputs.json") != inputs:
        raise ValueError("Frozen rollback inputs changed")
    atomic_json(out / "inputs.json", inputs)
    catalog = load_catalog()
    hashes = {key: item["mesh_hash"] for key, item in catalog.items() if item["available"]}
    if len(hashes) != 44 or hashes != read(sized / "source-hashes-after.json"):
        raise ValueError("Source models differ from the recorded 44 assets")
    vision = Vision(config, cache / "vision", out / "usage-rollback.json")
    scenes = Scenes(cache, vision)
    rows = []
    try:
        for row in sorted(previous, key=lambda r: r["job_id"]):
            ident = row["asset_id"]
            number = int(row["job_id"].split("-")[0])
            best, old_pose = selected_geometry(old, row)
            current, pose = selected_geometry(sized, latest[(ident, row["background"])])
            job = manifest[row["job_id"]]
            asset = read(cache / "assets" / row["asset_signature"] / "result.json")
            if (hashes[ident] != job["mesh_hash"] or asset["mesh_hash"] != hashes[ident]
                    or file_hash(row["background"]) != job["background_hash"]
                    or signature(old_pose) != job["baseline_pose"]
                    or file_hash(row["image"]) != job["baseline_image_hash"]):
                raise ValueError(f"Frozen source evidence changed for feedback {number:02d}")
            if (any(old_pose[k] != pose[k] for k in ("camera", "yaw", "matrix", "support_ids"))
                    or np.linalg.norm(np.subtract(old_pose["anchor_px"], pose["anchor_px"])) > 1.5):
                raise ValueError(f"Placement changed for feedback {number:02d}")
            scene = await scenes.prepare(row["background"])
            factor = config.asset_scale_factors.get(ident, 1.)
            if resize_candidate(asset, scene, pose, factor) is not pose:
                raise ValueError(f"Unexpected cumulative scale for feedback {number:02d}")
            if factor != 1:
                control = read(sized / "size-controls" / row["job_id"] / "control.json")
                original = pose["user_scale"]["original_expected_major_px"]
                if abs(pose["expected_major_px"] / original - factor) > .002:
                    raise ValueError(f"Unexpected display scale for feedback {number:02d}")
                source, window = Path(control["object_dir"]), control["object_window"]
                value, status = control, "review"
            else:
                if not np.isclose(old_pose["full_px"], pose["full_px"], rtol=0., atol=1e-7):
                    raise ValueError(f"Unexpected projection scale for feedback {number:02d}")
                source, window = Path(pose["fixed_alpha_dir"]), pose.get("fixed_alpha_window")
                value, status = best, row["status"]
            meta = read(source / "result.json" if window is None else source.parent / "result.json")
            if meta["camera"] != pose["camera"]:
                raise ValueError(f"Render camera changed for feedback {number:02d}")
            bounds = value["edit_bounds"]
            alpha, foreground, visible, offset = placement_masks(scene, pose, source, window, meta, bounds)
            audit = audit_png(scene, value, alpha, foreground, visible, pose, offset)
            if (not audit["pass"] or file_hash(value["image"]) != value["image_hash"]
                    or any(value[k] != current[k] for k in ("alpha_hash", "visible_alpha_hash", "foreground_mask_hash"))):
                raise ValueError(f"Frozen PNG audit failed for {number:02d}: {audit}")
            image = out / "images" / f"{number:02d}.png"
            image.parent.mkdir(exist_ok=True)
            if not image.is_file() or file_hash(image) != value["image_hash"]:
                shutil.copyfile(value["image"], image)
            context = image.with_suffix(".jpg")
            if not context.is_file():
                with Image.open(image) as im:
                    preview = im.crop(value["photo"]["local_context"]["bounds"])
                    preview.thumbnail((1000, 700))
                    preview.save(context, quality=94)
            rows.append({
                "number": number, "asset_id": ident, "name": catalog[ident]["name"],
                "asset_signature": row["asset_signature"], "background": row["background"],
                "background_hash": job["background_hash"], "seed": old_jobs[row["job_id"]]["seed"],
                "source_png": value["image"], "image": str(image), "context": str(context),
                "image_hash": value["image_hash"], "pose": pose, "params": best["params"],
                "baseline": {k: v for k, v in value.items() if k in ("checks", "photo", "edit_bounds",
                    "alpha_hash", "visible_alpha_hash", "foreground_mask_hash", "occlusion_fraction")},
                "archived_render": str(source), "archived_window": window, "archived_meta": meta,
                "status": status, "audit": audit, "cumulative_scale": factor,
                "same_size_control": factor != 1,
            })
    finally:
        await vision.close()
    after = {k: file_hash(v["mesh"]) for k, v in catalog.items() if v["available"]}
    if after != hashes:
        raise ValueError("Source models changed during rollback verification")
    atomic_json(out / "cases.json", rows)
    summary = {"images": len(rows), "same_size_controls": sum(r["same_size_control"] for r in rows),
               "checks_pass": all(r["audit"]["pass"] for r in rows), "source_models": len(after),
               "source_hashes": after, "new_renders": 0, "seconds": round(time.monotonic() - start, 3),
               "cost_total": cumulative(out)}
    atomic_json(out / "validation.json", summary)
    gallery(out, rows)
    return {k: v for k, v in summary.items() if k != "source_hashes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=ROLLBACK)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--size-controls", default=SIZED)
    args = parser.parse_args()
    print(asyncio.run(run(resolve(args.out), resolve(args.baseline), resolve(args.size_controls))))
