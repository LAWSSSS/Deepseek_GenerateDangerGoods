"""Support-only screening for the four revised samples.

This entry deliberately does not reuse the formal source-view gate.  The
revisions have no verified source camera or alpha, so the report keeps source
visibility, occlusion and scale unknown and evaluates only support geometry,
physical settling and actual contact patches.
"""
import argparse
import asyncio
import html
import json
import math
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from batch_v2 import Budget, BudgetReached, Runner
from mesh_placement import mesh_view, open_fit
from mesh_runtime import run_bounded, safe_tail
from mesh_scene import analyze_scene
from mesh_screening import (SAMPLE_IDS, TARGET_BASIS, candidate_checks,
                            digest, load_revision_rows, parse_targets, scene_summary,
                            summary_payload)
from mesh_site_selection import select_sites, site_candidate
from run import Config, load
from scene_pose import PoseAtlas
from scene_v2 import NeedsReview, Perception, atomic_json, read_image

HERE = Path(__file__).resolve().parent
MARKER = "SUPPORT_SCAN_OK"


def valid_zone(candidate, scene):
    x, y = candidate["center_px"]
    width, height = scene["size"]
    for ident in scene["answer"].zones:
        center = scene["zones"][ident]["anchors"][0]
        cx = center[0] * (width - 1) / 1000
        cy = center[1] * (height - 1) / 1000
        if math.hypot(x - cx, y - cy) <= max(2 * candidate["target_px"], .12 * width):
            return ident
    return None


def overlaps_protected(fit, protected):
    x, y = fit["origin"]
    h, w = fit["pixels"].shape[:2]
    return any(x < r and x + w > l and y < b and y + h > t for l, t, r, b in protected)


def parse_ints(text, *, name):
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(int(raw))
        except ValueError as error:
            raise ValueError(f"{name} 必须是整数列表：{text}") from error
    if not values:
        raise ValueError(f"{name} 不能为空")
    return values


def site_summary(site):
    return {
        "id": site["id"],
        "support_id": site["support_id"],
        "support_anchor_px": [round(float(v), 2) for v in site["support_anchor_px"]],
        "scene_score": site["scene_score"],
        "surface_error": site.get("surface_error"),
        "references": site.get("references"),
    }


def draw_site_overlay(frame, sites, target, path, sample_id):
    canvas = frame.copy()
    draw = ImageDraw.Draw(canvas)
    for site in sites:
        x, y = site["support_anchor_px"]
        radius = max(18, target / 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 215, 0), width=4)
        draw.line((x - 14, y, x + 14, y), fill=(255, 215, 0), width=3)
        draw.line((x, y - 14, x, y + 14), fill=(255, 215, 0), width=3)
        draw.text((x + 8, y - radius - 18), f"{sample_id} {site['id']} {site['scene_score']:.2f}",
                  fill=(255, 235, 80))
        for ref in (site.get("references") or {}).get("accepted", []):
            points = ref.get("points") or []
            if len(points) == 2:
                draw.line((tuple(points[0]), tuple(points[1])), fill=(70, 245, 150), width=3)
                draw.text((points[0][0] + 4, points[0][1] + 4),
                          f"{ref['id']} {'known' if ref.get('known') else 'local'}", fill=(70, 245, 150))
    draw.text((16, 16), f"{sample_id} support candidates · scale/source/occlusion unknown",
              fill=(255, 255, 255))
    canvas.save(path, quality=92)


def draw_candidate(frame, fit, path):
    base = frame.convert("RGBA")
    x, y = fit["origin"]
    overlay = Image.fromarray(fit["pixels"], mode="RGBA")
    base.alpha_composite(overlay, (int(x), int(y)))
    draw = ImageDraw.Draw(base)
    ax, ay = fit["support_anchor_px"]
    draw.ellipse((ax - 10, ay - 10, ax + 10, ay + 10), outline=(255, 215, 0), width=3)
    candidate = fit["candidate"]
    l, t, r, b = candidate["crop_px"]
    pad = max(30, round(.08 * candidate["target_px"]))
    crop = (max(0, l - pad), max(0, t - pad), min(frame.width, r + pad), min(frame.height, b + pad))
    base.crop(crop).convert("RGB").save(path, quality=94)


def render_views(row, args, scene, target, out_dir):
    """Settle the revision once per camera direction and keep the real render."""
    views = []
    attempts = []
    deadline = time.monotonic() + args.timeout
    for rotation_index, rotation in enumerate(args.rotation_list):
        for yaw in args.yaw_list:
            if time.monotonic() >= deadline:
                attempts.append({"rotation_deg": rotation, "yaw": yaw, "reason": "筛查达到单件时限"})
                break
            dest = out_dir / "views" / f"r{rotation_index}-y{yaw:03d}"
            dest.mkdir(parents=True, exist_ok=True)
            request = dest / "request.json"
            request.write_text(json.dumps({
                "mesh": row["mesh"],
                "out": str(dest),
                "rotation_deg": list(rotation),
                "camera_azimuth_deg": yaw,
                "camera_elevation_deg": scene["camera_elevation_deg"],
                "render_device": args.render_device,
                "max_components": 96,
            }), encoding="utf8")
            execution = run_bounded([
                str(args.blender), "--background", "--factory-startup", "--python-exit-code", "1",
                "--python", str(HERE / "screening_settle_worker.py"), "--", str(request),
            ], log=dest / "blender.log", seconds=min(300, max(1, deadline - time.monotonic())))
            attempt = {"rotation_deg": list(rotation), "yaw": yaw, "execution": execution}
            attempts.append(attempt)
            result_file = dest / "blender-result.json"
            if execution["state"] != "ok" or not result_file.is_file():
                attempt["reason"] = "Blender 落稳失败：" + safe_tail(dest / "blender.log", 180)
                continue
            result = json.loads(result_file.read_text(encoding="utf8"))
            attempt["physics"] = {
                "stable": result.get("physics_stable"),
                "components": result.get("component_count"),
                "failure_codes": result.get("failure_codes", []),
                "support_graph": result.get("support_graph"),
            }
            if not result.get("physics_stable"):
                attempt["reason"] = "刚体落稳失败：" + ",".join(result.get("failure_codes", []))
                continue
            for name in ("object.png", "with-shadow.png", "without-shadow.png"):
                if not (dest / name).is_file():
                    attempt["reason"] = f"缺少渲染输出：{name}"
                    break
            else:
                try:
                    view = mesh_view(dest, target, result["contacts_px"])
                except (NeedsReview, KeyError, ValueError, OSError) as error:
                    attempt["reason"] = str(error)[:220]
                    continue
                views.append({"folder": dest, "pose": result, "view": view,
                              "yaw": yaw, "rotation": list(rotation)})
                attempt["accepted_view"] = True
    return views, attempts


def fit_candidates(row, sites, views, atlas, scene, frame, args, out_dir):
    records = []
    accepted = []
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    target = float(row["approved_size_reference"]["major_px"])
    for view_index, view_record in enumerate(views):
        fit_root = out_dir / "fits" / f"v{view_index:02d}"
        for site in sites:
            if not valid_zone({"center_px": site["support_anchor_px"], "target_px": target}, scene):
                records.append({"sample": row["revision_id"], "site": site["id"], "view": view_index,
                                "status": "rejected", "reason": "承托点不在认可场景区域"})
                continue
            try:
                candidate = site_candidate(site, view_record["view"], atlas, fit_root, target=target)
                fit_mode = "site_anchor"
                shift = 0.0
                try:
                    fit = open_fit(candidate, fit_root, atlas, frame, view_record["view"],
                                   source_visible_fraction=None, physics_stable=True,
                                   pixel_size_review=True, fixed_anchor=site["support_anchor_px"],
                                   screening=True)
                except NeedsReview as first_error:
                    # Allow a bounded search around the ranked site; the report
                    # records how far the final contact moved from it.
                    fit = open_fit(candidate, fit_root, atlas, frame, view_record["view"],
                                   source_visible_fraction=None, physics_stable=True,
                                   pixel_size_review=True, screening=True)
                    fit_mode = "searched_around_site"
                    shift = math.dist(fit["support_anchor_px"], site["support_anchor_px"])
                    if shift > max(32.0, .15 * target):
                        raise NeedsReview(f"承托点偏移过大：{shift:.1f}px") from first_error
                if overlaps_protected(fit, scene["protected"]):
                    raise NeedsReview("物品范围与保护区相交")
            except (NeedsReview, ValueError, KeyError, OSError) as error:
                records.append({"sample": row["revision_id"], "site": site["id"], "view": view_index,
                                "status": "rejected", "reason": str(error)[:220]})
                continue
            candidate_id = f"{row['revision_id']}-{site['id']}-V{view_index:02d}"
            image_path = out_dir / "candidates" / f"{candidate_id}.png"
            draw_candidate(frame, fit, image_path)
            parts = fit["evidence"].get("component_support") or []
            record = {
                "id": candidate_id,
                "sample": row["revision_id"],
                "site": site["id"],
                "site_anchor_px": [round(float(v), 2) for v in site["support_anchor_px"]],
                "view": {"yaw": view_record["yaw"], "rotation_deg": view_record["rotation"]},
                "status": "candidate_pending_review",
                "target_px": target,
                "target_basis": TARGET_BASIS,
                "fit_mode": fit_mode,
                "anchor_shift_px": round(float(shift), 2),
                "support_anchor_px": [round(float(v), 2) for v in fit["support_anchor_px"]],
                "origin_px": [int(v) for v in fit["origin"]],
                "clearance_px": fit["clearance_px"],
                "contact_min": round(float(min(parts)), 4) if parts else None,
                "scene_score": site["scene_score"],
                "physics": {"stable": True, "components": view_record["pose"].get("component_count"),
                            "failure_codes": view_record["pose"].get("failure_codes", []),
                            "approximation": view_record["pose"].get("screening_approximation")},
                "checks": candidate_checks(fit["evidence"]),
                "references": site.get("references"),
                "image": str(image_path.relative_to(out_dir.parent)),
            }
            records.append(record)
            accepted.append((site, fit, record))
    accepted.sort(key=lambda item: (-float(item[0]["scene_score"]),
                                    -float(item[2]["contact_min"] or 0),
                                    -float(item[2]["clearance_px"]),
                                    item[2]["view"]["yaw"]))
    chosen = []
    used_sites = set()
    for site, _, record in accepted:
        if site["id"] in used_sites:
            continue
        used_sites.add(site["id"])
        chosen.append(record)
        if len(chosen) >= 3:
            break
    return records, chosen


def write_html(out_dir, summary):
    rows = []
    for sample in summary["samples"]:
        if sample.get("blocked_reason"):
            body = f'<p class="blocked">未生成候选：{html.escape(sample["blocked_reason"])}</p>'
        else:
            cards = []
            for candidate in sample.get("candidates", []):
                rel = Path(candidate["image"]).as_posix()
                cards.append(
                    '<figure><a href="{0}"><img src="{0}" alt="{1}"></a>'
                    '<figcaption>{1}<br>site {2} · yaw {3}° · contact {4} · clearance {5}px · {6}'
                    '</figcaption></figure>'.format(
                        html.escape(rel), html.escape(candidate["id"]), html.escape(candidate["site"]),
                        candidate["view"]["yaw"], candidate["contact_min"], candidate["clearance_px"],
                        html.escape(str(candidate.get("fit_mode", "")))))
            body = '<div class="cards">' + "".join(cards) + "</div>" if cards else \
                '<p class="blocked">没有通过支撑/落稳检查的候选。</p>'
        rows.append(
            f'<section><h2>{html.escape(sample["id"])} · {html.escape(str(sample.get("name", "")))}</h2>'
            f'<p>目标 {sample["target_px"]}px（{html.escape(sample["target_basis"])}） · '
            f'承托点 {len(sample.get("sites", []))} 个 · 候选 {len(sample.get("candidates", []))} 个</p>'
            f'<p class="unknown">源可见面、遮挡、现场尺度、Photoshop 均未验证。</p>{body}</section>')
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>四件修订样片 · 承托候选筛查</title>
<style>
body{{font:15px/1.5 sans-serif;margin:2em;background:#111;color:#eee}}h1{{font-size:24px}}
section{{border-top:1px solid #444;padding:1em 0}}.cards{{display:flex;gap:1em;flex-wrap:wrap}}
figure{{margin:0;background:#222;padding:.6em;max-width:420px}}img{{max-width:100%;display:block}}
figcaption{{font-size:12px;color:#bbb;margin-top:.4em}}.blocked{{color:#ffb4a8}}
.unknown{{color:#ffd479}}a{{color:#8bd5b7}}code{{color:#ffd479}}
</style></head><body>
<h1>四件修订样片 · 承托候选筛查</h1>
<p>状态：<code>{html.escape(summary["status"])}</code>。仅检查承托、落稳和实际接触；
源视角、遮挡、尺度与正式合成均未验证。落稳使用整件凸包单刚体近似，部件级支撑未验证。</p>
<p>背景 SHA-256：<code>{html.escape(summary["background_sha256"])}</code></p>
{''.join(rows)}
<p><a href="summary.json">summary.json</a> · <a href="sites.jsonl">sites.jsonl</a> ·
<a href="candidates.jsonl">candidates.jsonl</a></p>
</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf8")


def blocked_samples(rows, targets, reason):
    return [{"id": row["revision_id"], "asset_id": row["asset_id"], "number": row["number"],
             "name": row["file"], "target_px": targets[row["revision_id"]],
             "target_basis": TARGET_BASIS, "source_fidelity": "unknown",
             "source_visibility": "unknown", "occlusion": "unknown", "scene_scale": "unknown",
             "sites": [], "candidates": [], "blocked_reason": reason} for row in rows]


async def run(args):
    args.out = args.out.resolve()
    args.manifest = args.manifest.resolve()
    args.background = args.background.resolve()
    args.library = args.library.resolve()
    args.config = args.config.resolve()
    args.weights = args.weights.resolve()
    args.cache = args.cache.resolve()
    args.blender = args.blender.resolve()
    if (args.out / "summary.json").exists():
        raise FileExistsError("请选择新的输出目录")
    if args.budget != 0:
        raise ValueError("承托筛查只允许 --budget 0，不调用外部模型")
    ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    if not ids:
        raise ValueError("--ids 不能为空")
    unknown = [ident for ident in ids if ident not in SAMPLE_IDS]
    if unknown:
        raise ValueError(f"未知筛查 ID：{','.join(unknown)}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.background.is_file():
        raise FileNotFoundError(args.background)
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    args.out.mkdir(parents=True, exist_ok=False)
    args.rotation_list = [(0, 0, value) for value in parse_ints(args.rotations, name="rotations")]
    args.yaw_list = parse_ints(args.yaws, name="yaws")
    targets = parse_targets(args.target, ids)
    rows = load_revision_rows(args.manifest, ids, args.library)
    manifest_sha = digest(args.manifest)
    background_sha = digest(args.background)
    frame = read_image(args.background)
    atomic_json(args.out / "model-usage.json", [])
    notes = [
        "support screening only；source visibility、occlusion、scene scale 和 Photoshop 均未验证。",
        "target_px 来自最近探索预览的实测长边，仅用于候选比较，不是尺度批准。",
        "承托候选来自固定场景缓存分析；不调用 DeepSeek/Photoshop。",
        "修订模型为多部件装配体，落稳使用整件凸包单刚体近似；不验证部件级支撑和内部装配关系。",
    ]
    config = load(Config, args.config)
    engine = Perception(args.weights, args.cache)
    try:
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as http:
            runner = Runner(None, http, config, None, args.cache,
                            Budget(args.out / "model-usage.json", 0))
            scene = await analyze_scene(runner, {**rows[0], "job_id": "support-scan"},
                                        args.background, frame, engine, args.out)
    except (BudgetReached, RuntimeError, NeedsReview, ValueError, OSError) as error:
        reason = "blocked_scene_cache_missing" if isinstance(error, BudgetReached) else "blocked_scene_analysis"
        summary = summary_payload(status=reason, manifest_path=args.manifest,
                                  manifest_sha256=manifest_sha, background_sha256=background_sha,
                                  scene=None, samples=blocked_samples(rows, targets, reason), notes=notes)
        atomic_json(args.out / "summary.json", summary)
        write_html(args.out, summary)
        print(f"{reason}: {str(error)[:300]}", flush=True)
        return
    if not scene.get("cache_hit"):
        reason = "blocked_scene_cache_missing"
        summary = summary_payload(status=reason, manifest_path=args.manifest,
                                  manifest_sha256=manifest_sha, background_sha256=background_sha,
                                  scene=scene, samples=blocked_samples(rows, targets, reason), notes=notes)
        atomic_json(args.out / "summary.json", summary)
        write_html(args.out, summary)
        print(reason, flush=True)
        return
    atlas = PoseAtlas(*engine.scene(args.background))
    atlas.elevation = scene["camera_elevation_deg"]
    samples = []
    site_lines = []
    candidate_lines = []
    for row in rows:
        ident = row["revision_id"]
        target = targets[ident]
        row["job_id"] = f"support-scan-{ident}"
        row["approved_size_reference"] = {"major_px": target}
        sites_all, site_rejections = select_sites(atlas, scene, target, limit=max(8, args.max_sites * 4))
        sites = sites_all[:args.max_sites]
        sample_dir = args.out / ident
        sample_dir.mkdir(parents=True, exist_ok=True)
        for rank, site in enumerate(sites, 1):
            site_lines.append({"sample": ident, "rank": rank, **site})
        if sites:
            draw_site_overlay(frame, sites, target, sample_dir / "承托站点.jpg", ident)
        sample = {
            "id": ident, "asset_id": row["asset_id"], "number": row["number"], "name": row["file"],
            "target_px": target, "target_basis": TARGET_BASIS,
            "source_fidelity": "unknown", "source_visibility": "unknown",
            "occlusion": "unknown", "scene_scale": "unknown",
            "site_count": len(sites), "sites": [site_summary(site) for site in sites],
            "site_rejections": site_rejections,
            "candidates": [], "blocked_reason": None,
        }
        if not sites:
            sample["blocked_reason"] = "no_support_sites"
            samples.append(sample)
            continue
        if args.sites_only:
            sample["blocked_reason"] = None
            samples.append(sample)
            continue
        views, attempts = render_views(row, args, scene, target, sample_dir)
        atomic_json(sample_dir / "view-attempts.json", attempts)
        if not views:
            sample["blocked_reason"] = "no_stable_view"
            samples.append(sample)
            continue
        records, chosen = fit_candidates(row, sites, views, atlas, scene, frame, args, sample_dir)
        candidate_lines.extend(records)
        sample["candidates"] = chosen
        if not chosen:
            sample["blocked_reason"] = "no_candidate"
        samples.append(sample)
    status = "sites_only_pending_views" if args.sites_only else "support_screening_only"
    summary = summary_payload(status=status, manifest_path=args.manifest,
                              manifest_sha256=manifest_sha, background_sha256=background_sha,
                              scene=scene, samples=samples, notes=notes)
    atomic_json(args.out / "summary.json", summary)
    with (args.out / "sites.jsonl").open("w", encoding="utf8") as stream:
        for line in site_lines:
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
    with (args.out / "candidates.jsonl").open("w", encoding="utf8") as stream:
        for line in candidate_lines:
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
    write_html(args.out, summary)
    print(MARKER, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ids", default="10,13,19-1,31")
    parser.add_argument("--library", type=Path,
                        default=Path("outputs/三维模型库-圆润复验/results.jsonl"))
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--config", type=Path, default=HERE / "batch-config.example.json")
    parser.add_argument("--weights", type=Path, default=Path("work/v2-models"))
    parser.add_argument("--cache", type=Path, default=Path("work/v2-cache"))
    parser.add_argument("--blender", type=Path,
                        default=Path("work/blender45/blender-4.5.3-windows-x64/blender.exe"))
    parser.add_argument("--render-device", default="CPU")
    parser.add_argument("--max-sites", type=int, default=8)
    parser.add_argument("--yaws", default="0,90,180,270")
    parser.add_argument("--rotations", default="0")
    parser.add_argument("--sites-only", action="store_true")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.max_sites <= 32:
        parser.error("--max-sites 必须在 1..32")
    if args.timeout < 1:
        parser.error("--timeout 必须为正数")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
