"""Build the review-only part-based candidate for item 24 (air cylinder).

Item 24 is a SHAKO-badged square-profile double-acting cylinder, read from the
photograph as SUS63-100 (63 mm bore / 100 mm stroke).  The old TripoSR mesh is
a dark, wavy blob; the supplied six-view sheet is an uncalibrated AI-generated
SMC SU63-100 sheet that duplicates the end-face port on side faces.  The
candidate is therefore a part-based rebuild guided by the photograph for the
visible faces and by the sheet only for hidden-side topology.

The candidate never replaces the library mesh, stays manual-only and never
enters automatic placement or the formal composite.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import PIL
import trimesh
from PIL import Image, ImageDraw, ImageFont

from scene_v2 import digest
from mesh_rebuild_item import _tile_detail, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 24
VERSION = 'cylinder24-parametric-v1'
REFERENCE_NAME = '气动缸六视图产品图册.png'
REFERENCE_SHA256 = '33506beff8d3d7aa04c6138eb19e96e17a55700f786466b912534bc44eb10a17'
SOURCE_SIZE = (1536, 1024)
RUST_SIZE = (1536, 1024)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
TILE_NAMES = {
    0: '拉丝铝缸筒', 1: '黑色端盖', 2: '镀铬活塞杆', 3: '镀层钢螺母/螺钉',
    4: '黑色橡胶', 5: '螺纹钢', 6: '深色孔底/槽', 7: '备用', 8: '备用',
    9: '备用', 10: '备用', 11: '备用', 12: '备用', 13: '备用', 14: '备用',
    15: '备用',
}
METAL_TILES = ('aluminum', 'cap', 'chrome', 'steel', 'thread', 'dark')
DIELECTRIC_TILES = ('rubber',)
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'


def find_reference():
    path = ROOT / 'references/parallel-six-view' / REFERENCE_NAME
    if not path.is_file() or digest(path) != REFERENCE_SHA256:
        raise FileNotFoundError('local six-view reference is missing or has changed')
    return path


def local_path(value):
    path = Path(value)
    try:
        path.relative_to(ROOT)
        return path
    except ValueError:
        parts = path.parts
        if 'scrap-composite' in parts:
            index = parts.index('scrap-composite')
            return ROOT.joinpath(*parts[index + 1:])
        raise ValueError(f'library path is outside this worktree and cannot be mapped: {path}')


# ---------------------------------------------------------------------------
# PBR atlas and label
# ---------------------------------------------------------------------------
def tile_rect(idx, atlas_size, tile_size=512):
    col, row = idx % 4, idx // 4
    x0, y0 = col * tile_size, (3 - row) * tile_size
    return x0, y0, x0 + tile_size, y0 + tile_size


def normal_from_height(height, strength=1.0):
    height = np.asarray(height, np.float32)
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack((nx / length, ny / length, nz / length), axis=-1)
    return np.clip((out * .5 + .5) * 255, 0, 255).astype(np.uint8)


def brushed_field_u(tile, rng, sigma=0.7):
    """Horizontal streaks (constant along U, varying across V)."""
    line = rng.normal(size=(tile, 1)).astype(np.float32)
    field = np.repeat(line, tile, axis=1)
    field = cv2.blur(field, (1, 3))
    field = cv2.blur(field, (1, 7))
    field = (field - field.mean()) / max(1e-6, float(field.std()))
    return field


def safe_detail(image, rect, tile):
    try:
        return np.clip(_tile_detail(image, rect, size=tile), -25, 25)
    except Exception:
        return np.zeros((tile, tile), np.float32)


def rust_palette(rust):
    a = np.asarray(rust)
    mask = a[:, :, 3] >= 128
    rr, gg, bb = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    brown = mask & (rr >= bb + 15) & (rr >= gg - 5) & (rr >= 70) & (gg <= 160)
    strong = brown & (rr >= bb + 45)
    if strong.sum() >= 1000:
        rgb = np.median(a[:, :, :3][strong], axis=0).astype(np.float32)
    elif brown.sum() >= 500:
        rgb = np.median(a[:, :, :3][brown], axis=0).astype(np.float32)
    else:
        rgb = np.array([134, 101, 75], np.float32)
    return rgb, int(brown.sum()), int(strong.sum())


def make_label(out):
    """Redraw the observed label; it is not the factory print file."""
    W, H = 1024, 384
    img = Image.new('RGB', (W, H), (246, 246, 244))
    draw = ImageDraw.Draw(img)
    red = (205, 32, 30)
    black = (22, 22, 22)
    draw.rectangle((0, 0, W, 148), fill=red)
    draw.rectangle((770, 28, 1000, 118), fill=black)
    # Simple circled-S logo mark, drawn as an approximation of the printed mark.
    draw.ellipse((150, 34, 226, 110), outline=black, width=7)
    try:
        f_logo = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 52)
    except OSError:
        f_logo = ImageFont.load_default()
    draw.text((188, 72), 'S', fill=black, font=f_logo, anchor='mm')
    try:
        f_brand = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 58)
        f_model = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 100)
        f_small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 36)
    except OSError:
        f_brand = f_model = f_small = ImageFont.load_default()
    draw.text((250, 72), 'SHAKO', fill=black, font=f_brand, anchor='lm')
    draw.text((42, 255), 'SUS63·100', fill=black, font=f_model, anchor='lm')
    draw.text((46, 337), 'MAX. PRESS. 1.0 MPa', fill=black, font=f_small, anchor='lm')
    # Light aged-paper mottling and a few pale scratches; text stays legible.
    rng = np.random.default_rng(24024)
    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0.0, 2.2, arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB')
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for cx, cy, rx, ry, alpha in ((140, 300, 150, 42, 16), (720, 250, 190, 55, 12),
                                  (900, 335, 120, 34, 18), (330, 190, 90, 30, 10)):
        od.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(128, 98, 58, alpha))
    for x0, y0, x1, y1 in ((90, 300, 260, 306), (560, 190, 700, 196),
                           (820, 270, 960, 274)):
        od.line((x0, y0, x1, y1), fill=(170, 160, 150, 80), width=1)
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    path = out / 'label_base_color.png'
    img.save(path)
    return path, {
        'size': [W, H],
        'text': ['SHAKO', 'SUS63·100', 'MAX. PRESS. 1.0 MPa'],
        'note': ('按原照重绘的近似标签；商标、型号点和字距为近似，'
                 '不是原厂印刷文件。原照品牌可读为 SHAKO，型号读作 SUS63·100 '
                 '仍存在辨认不确定性。'),
    }


def make_atlases(rust_path, original_path, out, profile):
    rust = Image.open(rust_path).convert('RGBA')
    original = Image.open(original_path).convert('RGBA')
    if rust.size != RUST_SIZE:
        raise ValueError('24 做锈素材尺寸改变，采样坐标需要重新核对')
    if original.size != SOURCE_SIZE:
        raise ValueError('24 原图尺寸改变，像素坐标需要重新核对')
    size = profile['texture']['size']
    tile = profile['texture']['tile']
    tiles = profile['texture']['tiles']
    rng = np.random.default_rng(24024)
    base = np.zeros((size, size, 3), np.uint8)
    mr = np.zeros((size, size, 3), np.uint8)
    normal = np.zeros((size, size, 3), np.uint8)
    normal[:, :, 2] = 255
    fine = cv2.resize(rng.random((320, 320)).astype(np.float32), (tile, tile),
                      interpolation=cv2.INTER_CUBIC)
    coarse = cv2.resize(rng.random((24, 24)).astype(np.float32), (tile, tile),
                        interpolation=cv2.INTER_CUBIC)
    micro = cv2.resize(rng.random((512, 512)).astype(np.float32), (tile, tile),
                       interpolation=cv2.INTER_LINEAR)
    brush = brushed_field_u(tile, rng)
    rust_rgb, brown_px, strong_px = rust_palette(rust)
    dark_oxide = rust_rgb * .55
    # Small observed patches, guarded against transparent background.
    cap_detail = safe_detail(rust, (1150, 140, 1250, 240), tile)
    barrel_detail = safe_detail(rust, (650, 340, 760, 450), tile)
    rod_detail = safe_detail(rust, (120, 760, 220, 860), tile)

    def put(idx, color, roughness, metallic, height=None, *, clip_min=12):
        x0, y0, x1, y1 = tile_rect(idx, size, tile)
        sl = np.s_[y0:y1, x0:x1]
        color = np.clip(color, clip_min, 255).astype(np.uint8)
        base[sl] = color
        rough = np.broadcast_to(np.asarray(roughness, np.float32), color.shape[:2])
        metal = np.broadcast_to(np.asarray(metallic, np.float32), color.shape[:2])
        mr[sl] = np.stack((np.full(color.shape[:2], 255),
                           np.clip(rough * 255, 0, 255),
                           np.clip(metal * 255, 0, 255)), axis=-1).astype(np.uint8)
        if height is not None:
            normal[sl] = normal_from_height(height, .72)

    specs = {
        'aluminum': (barrel_detail, 'aluminum'),
        'cap': (cap_detail, 'cap'),
        'chrome': (rod_detail, 'chrome'),
        'steel': (rod_detail, 'steel'),
        'rubber': (micro - .5, 'rubber'),
        'thread': (rod_detail, 'thread'),
        'dark': (cap_detail * 0, 'dark'),
    }
    for name, (detail, key) in specs.items():
        idx = tiles[name]
        mat = profile['material_profile'][key]
        base_rgb = np.asarray(mat['base_rgb'], np.float32)[None, None, :]
        amount = float(mat.get('rust_amount', 0.0))
        bright = 1 + detail[:, :, None] * .0009 + (coarse[:, :, None] - .5) * .022 + \
            (fine[:, :, None] - .5) * .010
        if name in ('aluminum', 'chrome', 'steel', 'thread'):
            bright = bright * (1 + brush[:, :, None] * .008)
        if name == 'cap':
            bright = bright * (1 + micro[:, :, None] * .028)
            # Sparse worn-anodising scratches, not a uniform brightening.
            scratch = np.clip((brush - 1.45) / 1.2, 0, 1)
            bright = bright * (1 + scratch[:, :, None] * .28)
        color = base_rgb * bright
        if amount > 0:
            thr = .78 - .20 * amount
            rust_cov = np.clip((coarse - thr) / .14, 0, 1) * amount
            spots = np.clip((fine - .925) / .07, 0, 1) * amount * .35
            cov = np.clip(rust_cov + spots, 0, .32)
            color = color * (1 - cov[:, :, None] * .65) + \
                rust_rgb[None, None, :] * cov[:, :, None] * .65
            rough = mat['roughness'] * (1 - cov) + .66 * cov + (fine - .5) * .025
            metal = mat['metallic'] * (1 - cov * .70)
            height = detail * .0008 + (brush * .020 if name != 'cap' else 0) + \
                cov * (fine - .5) * .15
        else:
            rough = np.full((tile, tile), mat['roughness'], np.float32) + \
                (fine - .5) * .022
            metal = np.full((tile, tile), mat['metallic'], np.float32)
            height = (micro - .5) * .16
        if name == 'rubber':
            color = base_rgb * (1 + (fine[:, :, None] - .5) * .14 +
                                (micro[:, :, None] - .5) * .10 +
                                (coarse[:, :, None] - .5) * .06)
            height = (micro - .5) * .55 + (fine - .5) * .25
            metal = np.zeros((tile, tile), np.float32)
            rough = np.full((tile, tile), mat['roughness'], np.float32) + \
                (fine - .5) * .04
        if name == 'dark':
            color = base_rgb * (1 + (micro[:, :, None] - .5) * .16 +
                                (fine[:, :, None] - .5) * .08 +
                                (coarse[:, :, None] - .5) * .06)
            height = (micro - .5) * .55
        put(idx, color, rough, metal, height, clip_min=10 if name in ('rubber', 'cap', 'dark') else 24)

    # Spare tiles stay neutral mid-grey, never black.
    for idx in (7, 8, 9, 10, 11, 12, 13, 14, 15):
        put(idx, np.full((tile, tile, 3), 128, np.float32), .60, 0.0)

    paths = {}
    for name, arr in (('base_color', base), ('metallic_roughness', mr),
                      ('normal', normal)):
        p = out / f'{name}.png'
        Image.fromarray(arr.astype(np.uint8), 'RGB').save(p)
        paths[name] = str(p)
    label_path, label_meta = make_label(out)
    paths['label'] = str(label_path)

    sheet = Image.new('RGB', (size, size), (245, 245, 243))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(LABEL_FONT, 22)
    except OSError:
        font = ImageFont.load_default()
    for idx in range(16):
        col, row = idx % 4, idx // 4
        x, y = col * tile, row * tile
        sheet.paste(Image.fromarray(base[y:y + tile, x:x + tile]), (x, y))
        draw.rectangle((x, y, x + tile, y + tile), outline=(230, 230, 228), width=2)
        draw.text((x + 10, y + 8), f"{idx} {TILE_NAMES.get(idx, '备用')}",
                  fill=(20, 20, 20), font=font)
    sheet.save(out / '材质图集-布局.jpg', quality=93)
    evidence = {
        'rust_palette_rgb': [int(v) for v in rust_rgb],
        'rust_brown_pixels': brown_px,
        'rust_strong_pixels': strong_px,
        'label': label_meta,
        'note': ('锈色取自已对齐做锈图中较饱和的棕色像素中位数；纹理分布为程序化近似，'
                 '不是逐像素投影。金属/粗糙度/法线均为合成 PBR，不含烘死的灯光或阴影。'),
    }
    return paths, evidence


# ---------------------------------------------------------------------------
# Source measurements, old audit and diagnostics
# ---------------------------------------------------------------------------
def source_measurements(original_path, rust_path, source_alpha_path, profile):
    im = Image.open(original_path).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('原图尺寸改变，测量需要重新核对')
    alpha = np.asarray(Image.open(source_alpha_path).convert('L'))
    ys, xs = np.where(alpha > 128)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        'original_size': list(im.size),
        'rust_size': list(Image.open(rust_path).size),
        'source_alpha_size': list(alpha.shape[::-1]),
        'source_silhouette_bbox': bbox,
        'source_silhouette_extent_px': [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1],
        'source_silhouette_aspect_w_h': round((bbox[2] - bbox[0] + 1) /
                                             (bbox[3] - bbox[1] + 1), 4),
        'scale_note': profile['scale_note'],
        'note': ('源图是 3/4 商品照，不能作正交测量；这里的像素只能说明源视图轮廓范围。'
                 '候选尺寸来自配置中的比例推断，不是实测。'),
    }


def old_model_audit(row, library_row):
    path = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
    audit = None
    if path.is_file():
        for line in path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('number') == NUMBER:
                audit = item
                break
    return {
        'audit': audit,
        'old_mesh_sha256': library_row.get('mesh_sha256'),
        'old_source_fidelity': (library_row.get('reconstruction') or {}).get('fidelity'),
        'old_status': library_row.get('status'),
        'old_reason': library_row.get('reason'),
        'note': ('旧模型源视角 IoU 通过并不能证明结构；六方向审计图显示截面波浪、'
                 '端盖/进气口/标签糊化、杆端与端盖粘连。故不复用旧网格。'),
    }


def _mask_fit(mask, size=256):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.zeros((size, size), np.uint8)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = min((size - 8) / max(1, w), (size - 8) / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop.astype(np.uint8), (nw, nh),
                         interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def silhouette_diagnostics(out, source_alpha_path):
    source = (np.asarray(Image.open(source_alpha_path).convert('L')) > 128)
    target = _mask_fit(source.astype(np.uint8))
    target_flip = target[:, ::-1]
    rows = []
    for i in range(24):
        p = out / 'renders' / f'turn-{i:02d}.png'
        if not p.is_file():
            continue
        arr = np.asarray(Image.open(p).convert('RGBA'))
        mask = (arr[:, :, 3] > 128).astype(np.uint8)
        fitted = _mask_fit(mask)
        inter = int(np.logical_and(fitted, target).sum())
        union = int(np.logical_or(fitted, target).sum())
        inter_m = int(np.logical_and(fitted, target_flip).sum())
        union_m = int(np.logical_or(fitted, target_flip).sum())
        rows.append({
            'index': i,
            'azimuth_deg': i * 15,
            'iou': round(inter / union, 4) if union else 0.0,
            'iou_mirrored_reference': round(inter_m / union_m, 4) if union_m else 0.0,
            'render': str(p.relative_to(out)).replace('\\', '/'),
        })
    rows.sort(key=lambda r: max(r['iou'], r['iou_mirrored_reference']), reverse=True)
    best = rows[0] if rows else None
    return {
        'method': ('把源图透明轮廓与 24 个 15° 仰角 20° 的预渲染轮廓分别按外接框'
                   '缩放居中后计算 IoU；同时试镜像参考。未标定，只是诊断。'),
        'best': best,
        'top5': rows[:5],
        'diagnostic_only': True,
        'passed': None,
        'note': 'IoU 不是实测精度，不能证明尺寸、结构或装配关系；镜像结果同样不是验收。',
    }


def metal_reflection_check(out):
    a_path = out / 'renders' / 'metal-only-light-a.png'
    b_path = out / 'renders' / 'metal-only-light-b.png'
    if not (a_path.is_file() and b_path.is_file()):
        return {'changed': False, 'reason': 'renders missing'}
    a = np.asarray(Image.open(a_path).convert('RGBA')).astype(np.float32)
    b = np.asarray(Image.open(b_path).convert('RGBA')).astype(np.float32)
    mask = (a[:, :, 3] > 128) & (b[:, :, 3] > 128)
    if mask.sum() < 1000:
        return {'changed': False, 'reason': 'metal mask too small'}
    la = (0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * b[:, :, 2])
    lb = (0.2126 * b[:, :, 0] + 0.7152 * b[:, :, 1] + 0.0722 * b[:, :, 2])
    diff = np.abs(la[mask] - lb[mask])
    # Brightest-pixel centroid shift: a moving light must move the highlight.
    def centroid(luma):
        vals = np.where(mask, luma, -1)
        thresh = np.percentile(luma[mask], 97)
        yy, xx = np.where(vals >= thresh)
        return float(xx.mean()), float(yy.mean())
    ca, cb = centroid(la), centroid(lb)
    shift = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    changed = bool(diff.mean() > 1.0 and shift > 4.0)
    return {
        'changed': changed,
        'mean_abs_luma_difference': round(float(diff.mean()), 4),
        'p95_abs_luma_difference': round(float(np.percentile(diff, 95)), 4),
        'highlight_centroid_a': [round(v, 2) for v in ca],
        'highlight_centroid_b': [round(v, 2) for v in cb],
        'highlight_shift_px': round(shift, 2),
        'mask_pixels': int(mask.sum()),
        'note': ('同相机、仅移动主光的金属隔离渲染；只证明高光与亮度随光位变化，'
                 '不证明材质参数是实物的实测值。'),
    }


def texture_checks(paths, profile):
    base = np.asarray(Image.open(paths['base_color']).convert('RGB'))
    mr = np.asarray(Image.open(paths['metallic_roughness']).convert('RGB'))
    tiles = profile['texture']['tiles']
    size = profile['texture']['size']
    tile = profile['texture']['tile']
    rows = {}
    ok = True
    for name, idx in tiles.items():
        col, row = idx % 4, idx // 4
        x0, y0 = col * tile, (3 - row) * tile
        c = base[y0:y0 + tile, x0:x0 + tile].astype(np.float32)
        m = mr[y0:y0 + tile, x0:x0 + tile].astype(np.float32)
        luma = 0.2126 * c[:, :, 0] + 0.7152 * c[:, :, 1] + 0.0722 * c[:, :, 2]
        rough = m[:, :, 1] / 255.0
        metal = m[:, :, 2] / 255.0
        spec = profile['material_profile'][name]
        tile_ok = bool(luma.std() > 1.0 and luma.mean() > 20 and
                       rough.min() >= 0.03 and rough.max() <= 0.97 and
                       abs(float(metal.mean()) - spec['metallic']) < 0.08)
        rows[name] = {
            'tile': idx,
            'mean_luma': round(float(luma.mean()), 2),
            'luma_std': round(float(luma.std()), 3),
            'roughness_mean': round(float(rough.mean()), 3),
            'metallic_mean': round(float(metal.mean()), 3),
            'expected_metallic': spec['metallic'],
            'passed': tile_ok,
        }
        ok = ok and tile_ok
    label = np.asarray(Image.open(paths['label']).convert('RGB'))
    red = ((label[:, :, 0] > 150) & (label[:, :, 1] < 110) & (label[:, :, 2] < 110))
    white = (label.min(axis=2) > 220)
    dark = (label.max(axis=2) < 80)
    label_ok = bool(red.mean() > 0.08 and white.mean() > 0.25 and dark.mean() > 0.01)
    return {
        'tiles': rows,
        'label': {
            'size': list(label.shape[1::-1]),
            'red_fraction': round(float(red.mean()), 4),
            'near_white_fraction': round(float(white.mean()), 4),
            'dark_ink_fraction': round(float(dark.mean()), 4),
            'passed': label_ok,
        },
        'passed': bool(ok and label_ok),
        'note': ('图集检查只证明贴图非纯色、粗糙度/金属度范围和标签颜色/文字区域存在；'
                 '不证明它们是实物的逐像素测量。'),
    }


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def _load_rgba_on_gray(path, size=300, background=(226, 229, 226)):
    im = Image.open(path).convert('RGBA')
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), background)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    return canvas


def _font(size, bold=False):
    for path in ((LABEL_FONT_BOLD if bold else LABEL_FONT),
                 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(im, title, subtitle=None, size=300):
    canvas = Image.new('RGB', (size, size + (62 if subtitle else 42)), (248, 249, 247))
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, size + 4), title, fill=(25, 30, 27), font=_font(15, True))
    if subtitle:
        draw.text((8, size + 26), subtitle, fill=(85, 94, 89), font=_font(12))
    return canvas


def _board(items, cols, path, title, cell=300, note=None):
    rows = math.ceil(len(items) / cols)
    head = 54
    foot = 34 if note else 10
    canvas = Image.new('RGB', (cols * cell, head + rows * (cell + 62) + foot),
                       (238, 241, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill=(22, 28, 24), font=_font(20, True))
    for i, (img, label, sub) in enumerate(items):
        x, y = (i % cols) * cell, head + (i // cols) * (cell + 62)
        canvas.paste(img, (x, y))
        draw.rectangle((x, y, x + cell - 1, y + cell - 1),
                       outline=(210, 215, 211), width=1)
        draw.text((x + 8, y + cell + 4), label, fill=(25, 30, 27), font=_font(14, True))
        if sub:
            draw.text((x + 8, y + cell + 26), sub, fill=(85, 94, 89), font=_font(11))
    for c in range(1, cols):
        draw.line((c * cell, head, c * cell, head + rows * (cell + 62)),
                  fill=(236, 239, 236), width=3)
    for r in range(1, rows):
        draw.line((0, head + r * (cell + 62), cols * cell,
                   head + r * (cell + 62)), fill=(236, 239, 236), width=3)
    if note:
        draw.text((10, head + rows * (cell + 62) + 6), note,
                  fill=(95, 72, 38), font=_font(12))
    canvas.save(path, quality=92)
    return path


def build_boards(out):
    rd = out / 'renders'
    cn = {'front': '正视', 'back': '背视', 'left': '左视（杆端）',
          'right': '右视（后端）', 'top': '顶视', 'bottom': '底视'}
    items = []
    for v in VIEWS:
        items.append((_load_rgba_on_gray(rd / f'new-{v}.png'), f'新候选 · {cn[v]}',
                      '分部件重建'))
    for v in VIEWS:
        items.append((_load_rgba_on_gray(rd / f'old-{v}.png'), f'原模型 · {cn[v]}',
                      '旧网格，各视图按自身包围盒取景'))
    _board(items, 6, out / '六视图-新旧对照.jpg',
           '24 气缸 · 六方向新旧对照（新候选上排 / 原模型下排）', cell=300,
           note=('原模型为旧 TripoSR 网格；上下排取景尺度不同，不能据此比较尺寸。'
                 '六视图参考未标定。'))

    detail_items = [
        ('front-face', '前端盖端面：进气口/十字螺钉/加强筋/杆凸台'),
        ('rear-face', '后端盖端面：进气口/中央凸台/加强筋'),
        ('port-front', '前端进气口盲孔与内螺纹'),
        ('port-rear', '后端进气口盲孔与内螺纹'),
        ('rod-gland', '镀铬杆、垫圈、双六角螺母与防尘凸台'),
        ('rod-thread', '杆端公螺纹与螺母'),
        ('label', '标签重绘（SHAKO / SUS63·100 / 1.0 MPa）'),
        ('barrel-profile', '方形型材筒：四角圆角与纵向凹槽'),
        ('ea-eb', '顶面 EA/EB 铸字'),
        ('side-screw', '端盖侧面螺塞（功能未知，按可见位置推断）'),
        ('overall-3q', '整体 3/4 视角'),
    ]
    _board([(_load_rgba_on_gray(rd / f'detail-{n}.png'), t, '原照可见/推断')
            for n, t in detail_items], 4, out / '局部检查.jpg',
           '24 气缸 · 关键局部检查', cell=300)

    iso_items = [
        ('caps-only-top', '两端盖顶面（EA/EB、侧面螺塞）'),
        ('caps-only-bottom', '两端盖底面'),
        ('front-cap-only', '前端盖隔离'),
        ('rear-cap-only', '后端盖隔离'),
        ('rod-only', '活塞杆/螺母/防尘环隔离'),
        ('barrel-only', '缸筒/接缝/标签隔离'),
        ('barrel-label', '缸筒与标签侧面'),
    ]
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), t, '部件隔离')
            for n, t in iso_items], 4, out / '分部件隔离.jpg',
           '24 气缸 · 分部件隔离检查', cell=300)

    mat_items = [
        ('new-material-front', '新候选 · 材料正视'),
        ('new-clay-front', '新候选 · 灰模正视'),
        ('new-clay-3q', '新候选 · 灰模 3/4'),
        ('new-light-alternate', '同一正面、主光换位'),
        ('detail-overall-3q', '新候选 · 材料 3/4'),
    ]
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), t, '同相机')
            for n, t in mat_items], 3, out / '材质-同光照对照.jpg',
           '24 气缸 · 同光照材质 / 灰模对照', cell=300,
           note='灰模只检查形体；材质图不含烘焙灯光或阴影。')

    metal_items = [
        ('metal-only-light-a', '金属隔离 · 主光 A'),
        ('metal-only-light-b', '金属隔离 · 主光 B'),
    ]
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), t, '仅金属件，同相机')
            for n, t in metal_items], 2, out / '金属反射-变光对照.jpg',
           '24 气缸 · 金属反射变光检查', cell=420,
           note='高光移动是金属响应的证据，不等于实物金属参数的实测。')

    open_items = [
        ('caps-only-top', '两端盖顶面'),
        ('caps-only-bottom', '两端盖底面'),
        ('front-cap-only', '前端盖：真实杆通孔/端面进气口'),
        ('rear-cap-only', '后端盖：中央凸台/端面进气口'),
    ]
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), t, '隔离渲染')
            for n, t in open_items], 2, out / '开孔-隔离检查.jpg',
           '24 气缸 · 端盖开孔/结构隔离检查', cell=360,
           note=('进气口孔深与杆通孔另由 Blender BVH 射线检查记录；'
                 '隔离图本身不是装配验收。'))

    turns = [( _load_rgba_on_gray(rd / f'turn-{i:02d}.png', 220), f'{i * 15:03d}°', '')
             for i in range(24)]
    _board(turns, 6, out / '转台-24方向.jpg',
           '24 气缸 · 24 方向预渲染转台（仰角约 20°）', cell=220,
           note='仅外观；不证明隐藏结构、装配或落稳。')


def source_board(out, original_path, reference_path, silhouette):
    rd = out / 'renders'
    orig = Image.open(original_path).convert('RGB')
    orig.thumbnail((430, 430), Image.LANCZOS)
    ref = Image.open(reference_path).convert('RGB')
    ref.thumbnail((560, 400), Image.LANCZOS)
    best = silhouette.get('best') or {}
    best_img_path = out / best.get('render', 'renders/turn-21.png')
    best_img = _load_rgba_on_gray(best_img_path, 430)
    canvas = Image.new('RGB', (1500, 560), (240, 243, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), '24 气缸 · 原照 / 新候选 / 未标定六视图参考', fill=(22, 28, 24),
              font=_font(20, True))
    canvas.paste(orig, (10, 50))
    draw.text((10, 490), '原照（3/4 商品图，SHAKO 标签）', fill=(30, 36, 32), font=_font(14, True))
    canvas.paste(best_img, (470, 50))
    sub = (f"最佳方向 {best.get('azimuth_deg', '—')}°，IoU "
           f"{best.get('iou', '—')}（诊断）")
    draw.text((470, 490), '新候选 · 源视图最接近的预渲染方向', fill=(30, 36, 32), font=_font(14, True))
    draw.text((470, 512), sub, fill=(95, 72, 38), font=_font(12))
    canvas.paste(ref, (920, 50))
    draw.text((920, 490), '六视图参考（未标定；品牌/进气口面与原照冲突）',
              fill=(30, 36, 32), font=_font(14, True))
    canvas.save(out / '原照-新候选对照.jpg', quality=92)
    canvas2 = Image.new('RGB', (700, 480), (238, 241, 238))
    draw = ImageDraw.Draw(canvas2)
    draw.text((12, 10), '24 气缸 · 原视角轮廓诊断（未标定）', fill=(22, 28, 24),
              font=_font(18, True))
    diag = Image.open(out / 'renders' / 'source-alpha-diagnostic.png')
    canvas2.paste(diag, (10, 48))
    draw.text((12, 440), f"最佳 IoU {best.get('iou', '—')} / 镜像 "
                         f"{best.get('iou_mirrored_reference', '—')}（仅诊断）",
              fill=(95, 72, 38), font=_font(13))
    canvas2.save(out / '原视角诊断.jpg', quality=92)


def diagnostic_image(out, source_alpha_path, silhouette):
    source = (np.asarray(Image.open(source_alpha_path).convert('L')) > 128)
    target = _mask_fit(source.astype(np.uint8))
    best = silhouette.get('best') or {}
    arr = np.asarray(Image.open(out / best.get('render', 'renders/turn-21.png')).convert('RGBA'))
    fitted = _mask_fit((arr[:, :, 3] > 128).astype(np.uint8))
    canvas = np.zeros((256, 256 * 4 + 30, 3), np.uint8)
    canvas[:] = (238, 241, 238)
    panels = [
        ('source', target),
        ('new', fitted),
        ('overlap', ((target > 0) | (fitted > 0)).astype(np.uint8)),
        ('mirror', fitted[:, ::-1]),
    ]
    x = 0
    for name, mask in panels:
        if name == 'overlap':
            p = np.zeros((256, 256, 3), np.uint8)
            p[:, :] = (40, 44, 40)
            p[(target > 0) & (fitted > 0)] = (60, 150, 90)
            p[(target > 0) & (fitted == 0)] = (190, 70, 55)
            p[(target == 0) & (fitted > 0)] = (70, 110, 200)
        else:
            p = np.repeat((mask * 255)[:, :, None], 3, axis=2)
        canvas[:, x:x + 256] = p
        x += 266
    Image.fromarray(canvas).save(out / 'renders' / 'source-alpha-diagnostic.png')


# ---------------------------------------------------------------------------
# Report / result
# ---------------------------------------------------------------------------
def write_report(out, result):
    checks = result['checks']
    pending = '\n'.join(f'- {x}' for x in result['pending_review'])
    limitations = '\n'.join(f'- {x}' for x in result['limitations'])
    lines = [
        '# 24 气缸（SUS63·100 读法）分部件重建候选',
        '',
        '本候选为 **manual_only / 待审**，不替换原模型，不进入自动放置或正式合成。',
        '',
        '## 方法',
        '',
        '以原照可见结构为主，六视图只作未标定的外观与背侧拓扑参考。旧模型为单图',
        'TripoSR 网格，源视角 IoU 虽通过但六方向审计显示截面波浪、端盖/进气口/标签',
        '糊化和杆端粘连，故不复用旧网格。新候选按部件建立：方形挤压铝缸筒（四角圆角、',
        '每面三条纵向凹槽）、黑色端盖（端面进气口、真实内螺纹、前端杆通孔、四角十字',
        '槽螺钉、端面加强筋、EA/EB 铸字、侧面螺塞）、镀铬活塞杆、双六角螺母、橡胶',
        '防尘环、接缝垫和重绘标签。',
        '',
        '## 实际执行的检查',
        '',
        '| 检查 | 结果 |',
        '| --- | --- |',
    ]
    for key, value in checks.items():
        if isinstance(value, bool):
            value = '通过' if value else '未通过'
        lines.append(f'| {key} | {value} |')
    lines += [
        '',
        '## 待人工复核',
        pending,
        '',
        '## 限制',
        limitations,
        '',
        '## 文件',
        '',
        '- `mesh.glb`：分部件候选（坐标 Z 向上、轴线 Y、杆端 -Y）',
        '- `model.blend`：可编辑 Blender 文件（贴图已打包）',
        '- `result.json` / `hashes.json` / `worker-result.json` / `request.json`',
        '- `检查页.html`：独立静态检查页（预渲染证据）',
        '- `renders/`：六方向、局部、隔离、材质、变光与转台渲染',
        '',
        '所有 IoU 都是未标定诊断值，不是验收或实测精度。',
    ]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


def collect_hashes(out, inputs, scripts, blender_version):
    outputs = {}
    for p in sorted(out.rglob('*')):
        if p.is_file():
            outputs[str(p.relative_to(out)).replace('\\', '/')] = digest(p)
    return {
        'inputs': {k: {'path': str(v), 'sha256': digest(Path(v))} for k, v in inputs.items()},
        'scripts': {k: {'path': str(v), 'sha256': digest(Path(v))} for k, v in scripts.items()},
        'outputs': outputs,
        'blender': blender_version,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'opencv': cv2.__version__,
        'pillow': PIL.__version__,
        'trimesh': trimesh.__version__,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=NUMBER)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != NUMBER:
        p.error('this entry supports item 24 only')
    a.reference = (a.reference or find_reference()).resolve()
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    if a.out is None:
        a.out = (ROOT / 'work/direct-deepseek/24-draft' if a.draft
                 else ROOT / 'outputs/24气缸2-六视图重建-v1')
    a.out, a.library, a.blender = (x.resolve() for x in (a.out, a.library, a.blender))
    row = next(r for r in (json.loads(line) for line in
                           a.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == NUMBER)
    mesh_path = local_path(row['mesh'])
    rust_path = local_path(row['rust'])
    original_path = local_path(row['original'])
    source_view_path = local_path(row['source_view'])
    for name, path, expected in (('mesh', mesh_path, 'mesh_sha256'),
                                 ('rust', rust_path, 'rust_sha256'),
                                 ('original', original_path, 'clean_sha256')):
        if digest(path) != row[expected]:
            raise ValueError(f'{name} hash differs from the model library')
    alpha_path = source_view_path.parent / 'source-alpha.png'
    if digest(alpha_path) != row['source_alpha_sha256']:
        raise ValueError('source alpha hash differs from the model library')
    if digest(source_view_path) != row['source_view_sha256']:
        raise ValueError('source view hash differs from the model library')
    config_path = HERE / 'cylinder24_config.json'
    worker_path = HERE / 'mesh_rebuild_24_worker.py'
    checks_path = HERE / 'checks_rebuild_24.py'
    page_path = HERE / 'review24_page.py'
    profile = json.loads(config_path.read_text(encoding='utf8'))
    blender_version = subprocess.run([str(a.blender), '--version'], capture_output=True,
                                     text=True, check=True).stdout.splitlines()[0]
    signature_input = {
        'reference': digest(a.reference), 'mesh': row['mesh_sha256'],
        'rust': row['rust_sha256'], 'original': row['clean_sha256'],
        'source_alpha': digest(alpha_path), 'source_view': digest(source_view_path),
        'config': digest(config_path), 'driver': digest(Path(__file__)),
        'worker': digest(worker_path), 'checks': digest(checks_path),
        'page': digest(page_path) if page_path.is_file() else None,
        'blender': blender_version, 'python': sys.version.split()[0],
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'pillow': PIL.__version__, 'trimesh': trimesh.__version__,
        'draft': a.draft,
    }
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if a.out.exists() and any(a.out.iterdir()):
        record = a.out / 'result.json'
        if record.is_file():
            old = json.loads(record.read_text(encoding='utf8'))
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(
                    a.out / 'mesh.glb'):
                print(json.dumps({'status': 'review', 'cache_hit': True,
                                  'out': str(a.out)}, ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature')

    a.out.mkdir(parents=True, exist_ok=True)
    measurements = source_measurements(original_path, rust_path, alpha_path, profile)
    (a.out / 'source-measurements.json').write_text(
        json.dumps(measurements, ensure_ascii=False, indent=2), encoding='utf8')
    old_audit = old_model_audit(row, row)
    (a.out / 'old-model-audit.json').write_text(
        json.dumps(old_audit, ensure_ascii=False, indent=2), encoding='utf8')
    textures, texture_evidence = make_atlases(rust_path, original_path, a.out, profile)
    (a.out / 'texture-evidence.json').write_text(
        json.dumps(texture_evidence, ensure_ascii=False, indent=2), encoding='utf8')
    request = {'out': str(a.out), 'original_mesh': str(mesh_path),
               'textures': textures, 'config': profile,
               'source_camera': row['reconstruction']['source_camera'],
               'draft': a.draft}
    (a.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    with (a.out / 'blender.log').open('w', encoding='utf8') as log:
        process = subprocess.run([str(a.blender), '--background', '--factory-startup',
                                  '--python-exit-code', '1',
                                  '--python', str(worker_path), '--',
                                  str(a.out / 'request.json')],
                                 stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if process.returncode:
        raise RuntimeError(f'Blender failed; see {a.out / "blender.log"}')

    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    mesh_info = inspect_mesh(a.out / 'mesh.glb')
    silhouette = silhouette_diagnostics(a.out, alpha_path) if not a.draft else {
        'diagnostic_only': True, 'passed': None, 'best': None, 'top5': [],
        'note': 'draft 未渲染转台'}
    if not a.draft:
        diagnostic_image(a.out, alpha_path, silhouette)
        build_boards(a.out)
        source_board(a.out, original_path, a.reference, silhouette)
    metal = metal_reflection_check(a.out) if not a.draft else {
        'changed': None, 'note': 'draft 未渲染变光'}
    tex_checks = texture_checks(textures, profile)
    ports = worker['ray_checks']['ports']
    port_ok = all(v.get('blind') and v.get('depth_m') is not None and
                  0.010 <= v['depth_m'] <= 0.025 for v in ports.values())
    groove = worker['ray_checks']['barrel_grooves']
    groove_depth = (abs(groove.get('rib_crest', 0) - groove.get('groove_crest', 0))
                    if groove.get('rib_crest') is not None and
                    groove.get('groove_crest') is not None else None)
    required_renders = ([f'new-{v}.png' for v in VIEWS] +
                        [f'old-{v}.png' for v in VIEWS] +
                        [f'turn-{i:02d}.png' for i in range(24)] +
                        [f'detail-{n}.png' for n in
                         ('front-face', 'rear-face', 'port-front', 'port-rear',
                          'rod-gland', 'rod-thread', 'label', 'barrel-profile',
                          'ea-eb', 'side-screw', 'overall-3q')] +
                        ['caps-only-top.png', 'caps-only-bottom.png',
                         'front-cap-only.png', 'rear-cap-only.png',
                         'rod-only.png', 'barrel-only.png', 'barrel-label.png',
                         'new-material-front.png', 'new-clay-front.png',
                         'new-clay-3q.png', 'new-light-alternate.png',
                         'metal-only-light-a.png', 'metal-only-light-b.png'])
    renders_present = all((a.out / 'renders' / n).is_file() for n in required_renders) \
        if not a.draft else True

    checks = {
        'mesh_parts': mesh_info['parts'],
        'mesh_closed_after_welding_seams': mesh_info['closed_after_welding_seams'],
        'mesh_open_parts_glb': mesh_info['open_parts'],
        'closed_blender_parts': worker.get('closed_blender_parts'),
        'open_blender_parts': worker.get('open_blender_parts'),
        'mesh_all_closed': bool(mesh_info['closed_after_welding_seams'] ==
                                mesh_info['parts'] and
                                worker.get('closed_blender_parts') ==
                                mesh_info['parts']),
        'self_intersection_total': worker['self_intersection_total'],
        'self_intersection_free': worker['self_intersection_total'] == 0,
        'ports_blind_both': port_ok,
        'port_depths_m': {k: v.get('depth_m') for k, v in ports.items()},
        'rod_hole_through': worker['ray_checks']['rod_hole_through'],
        'barrel_groove_depth_m': groove_depth,
        'barrel_grooves_present': bool(groove_depth is not None and
                                       0.0010 <= groove_depth <= 0.0022),
        'texture_checks_passed': tex_checks['passed'],
        'metal_highlights_move': metal.get('changed'),
        'required_renders_present': renders_present,
        'source_silhouette_best_iou_diagnostic': (silhouette.get('best') or {}).get('iou'),
        'source_silhouette_mirrored_iou_diagnostic':
            (silhouette.get('best') or {}).get('iou_mirrored_reference'),
        'old_source_view_iou_diagnostic': (old_audit.get('old_source_fidelity') or {}).get('iou'),
        'manual_visual_review': 'not_performed',
        'webgl_interactive_viewer': 'not_available_in_this_worktree',
    }
    pending = [
        '人工视觉复核（本件只自动生成并检查，未由人眼验收外观）。',
        '六视图未标定，且与原照在品牌、型号和进气口所在面上冲突；尺寸、孔深、螺距和装配关系未实测。',
        '原照是单张 3/4 视角；背面、端盖内部、活塞/密封内部、缸筒壁厚和真实气路为推断。',
        '标签为按原照重绘的近似；SHAKO 商标、SUS63·100 字形和印刷位置不等于原厂文件。',
        'EA/EB 铸字、侧面螺塞功能、防尘环真实截面和四角固定方式为推断。',
        '落稳、接触点、现场尺度和场景光照融合未验证。',
        '浏览器检查只覆盖静态页面加载/图片切换，不覆盖 WebGL/GLB 实时渲染。',
    ]
    result = {
        'number': NUMBER,
        'version': VERSION,
        'status': 'review',
        'draft': bool(a.draft),
        'manual_only': True,
        'asset_id': row['asset_id'],
        'signature': signature,
        'original_mesh_sha256': row['mesh_sha256'],
        'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend_sha256': digest(a.out / 'model.blend'),
        'shape_basis': 'source_photo_visible_structure+uncalibrated_six_view_layout',
        'reference': {'path': str(a.reference.relative_to(ROOT)).replace('\\', '/'),
                      'sha256': digest(a.reference),
                      'role': 'visual_match_unverified',
                      'note': profile['reference']['note']},
        'inputs': {
            'original': {'path': str(original_path.relative_to(ROOT)).replace('\\', '/'),
                         'sha256': row['clean_sha256']},
            'rust': {'path': str(rust_path.relative_to(ROOT)).replace('\\', '/'),
                     'sha256': row['rust_sha256']},
            'old_mesh': {'path': str(mesh_path.relative_to(ROOT)).replace('\\', '/'),
                         'sha256': row['mesh_sha256']},
            'source_view': {'path': str(source_view_path.relative_to(ROOT)).replace('\\', '/'),
                            'sha256': row['source_view_sha256']},
            'source_alpha': {'path': str(alpha_path.relative_to(ROOT)).replace('\\', '/'),
                             'sha256': row['source_alpha_sha256']},
        },
        'source_measurements': measurements,
        'old_model_audit': old_audit,
        'texture_evidence': texture_evidence,
        'texture_checks': tex_checks,
        'metal_reflection_check': metal,
        'silhouette_diagnostics': silhouette,
        'worker': worker,
        'mesh_audit': mesh_info,
        'checks': checks,
        'isolated_opening_checks': {
            'ports': ports,
            'rod_hole_through': worker['ray_checks']['rod_hole_through'],
            'barrel_grooves': groove,
            'note': 'Blender BVH 射线检查；只覆盖所列端盖/缸筒几何。',
        },
        'pending_review': pending,
        'limitations': profile['limitations'],
        'files': {
            'mesh': 'mesh.glb', 'blend': 'model.blend',
            'base_color': 'base_color.png',
            'metallic_roughness': 'metallic_roughness.png',
            'normal': 'normal.png', 'label': 'label_base_color.png',
            'result': 'result.json', 'hashes': 'hashes.json',
            'worker_result': 'worker-result.json', 'request': 'request.json',
            'report': 'report.md', 'page': '检查页.html',
            'board_six_view': '六视图-新旧对照.jpg',
            'board_details': '局部检查.jpg',
            'board_parts': '分部件隔离.jpg',
            'board_material': '材质-同光照对照.jpg',
            'board_metal': '金属反射-变光对照.jpg',
            'board_openings': '开孔-隔离检查.jpg',
            'board_source': '原照-新候选对照.jpg',
            'board_source_audit': '原视角诊断.jpg',
            'board_turntable': '转台-24方向.jpg',
            'board_atlas': '材质图集-布局.jpg',
        },
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    (a.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    write_report(a.out, result)

    # Independent review page + optional headless browser self-test.
    browser = {'status': 'not_run', 'reason': 'draft run'}
    if not a.draft and page_path.is_file():
        sys.path.insert(0, str(HERE))
        import review24_page
        review24_page.build_page(a.out)
        browser = review24_page.browser_check(a.out)
    result['browser_check'] = {
        'status': browser.get('status'),
        'reason': browser.get('reason'),
        'result': browser.get('result'),
        'checked_at': browser.get('checked_at'),
    }
    (a.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding='utf8')

    hashes = collect_hashes(
        a.out,
        {'original': original_path, 'rust': rust_path, 'old_mesh': mesh_path,
         'source_view': source_view_path, 'source_alpha': alpha_path,
         'reference': a.reference},
        {'driver': Path(__file__), 'worker': worker_path, 'config': config_path,
         'checks': checks_path, 'page': page_path},
        blender_version)
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')

    if not a.draft:
        progress = {
            'number': NUMBER,
            'status': 'review',
            'out': str(a.out),
            'mesh': str(a.out / 'mesh.glb'),
            'blend': str(a.out / 'model.blend'),
            'checks': str(a.out / 'result.json'),
            'pending_review': pending,
            'next_number': 25,
        }
        progress_path = ROOT / 'work/direct-deepseek/24-progress.json'
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                 encoding='utf8')
    print(json.dumps({
        'status': 'review', 'out': str(a.out),
        'mesh': str(a.out / 'mesh.glb'),
        'parts': mesh_info['parts'],
        'closed': mesh_info['closed_after_welding_seams'],
        'self_intersections': worker['self_intersection_total'],
        'ports': ports,
        'best_iou': (silhouette.get('best') or {}).get('iou'),
        'metal_changed': metal.get('changed'),
        'browser': browser.get('status'),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
