"""Build the review-only part-based candidate for item 28 (coil-over shock).

Item 28 is a black motorcycle-style rear shock: a 12-turn black coil spring
around a black damper body, a chrome piston rod at the lower third, and two
eye mounts with rusted steel bushings.  The old TripoSR mesh is a wavy blob
with the spring fused to the body and both eye holes filled.  The supplied
six-view sheet is an uncalibrated AI-generated sheet that agrees on the
front/side layout but contradicts the photograph in its top/bottom panels.

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
NUMBER = 28
VERSION = 'shock28-parametric-v1'
REFERENCE_NAME = 'ChatGPT 图像 2026年10月14日 10_18_28-2.png'
REFERENCE_SHA256 = 'dcc140e00fd1878933b5fc066845f9ec64a23f3bea53e736436f5e97748922d2'
SOURCE_SIZE = (1136, 1384)
RUST_SIZE = (1136, 1384)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
DETAILS = ('top-eye', 'top-eye-bore', 'top-seat-notch', 'spring-coils',
           'body-rod', 'lower-seat', 'bottom-eye', 'bottom-eye-bore',
           'overall-3q', 'overall-side')
ISOLATIONS = ('seats-only-top', 'seats-only-bottom', 'top-eye-only',
              'bottom-eye-only', 'spring-only', 'body-rod-only',
              'new-material-front', 'new-clay-front', 'new-clay-3q',
              'new-light-alternate', 'metal-only-light-a', 'metal-only-light-b')
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
# PBR atlas
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
    brown = np.logical_and.reduce([mask, rr >= bb + 15, rr >= gg - 5,
                                   rr >= 70, gg <= 160])
    strong = np.logical_and(brown, rr >= bb + 45)
    if strong.sum() >= 1000:
        rgb = np.median(a[:, :, :3][strong], axis=0).astype(np.float32)
    elif brown.sum() >= 500:
        rgb = np.median(a[:, :, :3][brown], axis=0).astype(np.float32)
    else:
        rgb = np.array([142, 107, 79], np.float32)
    return rgb, int(brown.sum()), int(strong.sum())


def make_atlases(rust_path, original_path, out, profile):
    rust = Image.open(rust_path).convert('RGBA')
    original = Image.open(original_path).convert('RGBA')
    if rust.size != RUST_SIZE:
        raise ValueError('28 做锈素材尺寸改变，采样坐标需要重新核对')
    if original.size != SOURCE_SIZE:
        raise ValueError('28 原图尺寸改变，像素坐标需要重新核对')
    size = profile['texture']['size']
    tile = profile['texture']['tile']
    tiles = profile['texture']['tiles']
    rng = np.random.default_rng(28028)
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
    mid = cv2.resize(rng.random((96, 96)).astype(np.float32), (tile, tile),
                     interpolation=cv2.INTER_CUBIC)
    brush = brushed_field_u(tile, rng)
    rust_rgb, brown_px, strong_px = rust_palette(rust)
    paint_detail = safe_detail(rust, (470, 380, 570, 480), tile)
    eye_detail = safe_detail(rust, (590, 80, 680, 170), tile)
    rod_detail = safe_detail(rust, (500, 930, 560, 1010), tile)

    def put(idx, color, roughness, metallic, height=None, *, clip_min=10):
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
            normal[sl] = normal_from_height(height, .62)

    specs = {
        'paint': (paint_detail, 'paint', 0),
        'seat': (paint_detail, 'seat', 1),
        'steel': (eye_detail, 'steel', 2),
        'chrome': (rod_detail, 'chrome', 3),
        'rubber': (micro - .5, 'rubber', 4),
        'dark': (paint_detail * 0, 'dark', 5),
        'steel_ground': (rod_detail, 'steel_ground', 6),
    }
    for name, (detail, key, idx) in specs.items():
        mat = profile['material_profile'][key]
        base_rgb = np.asarray(mat['base_rgb'], np.float32)[None, None, :]
        amount = float(mat.get('rust_amount', 0.0))
        bright = 1 + detail[:, :, None] * .0009 + (coarse[:, :, None] - .5) * .022 + \
            (fine[:, :, None] - .5) * .010
        if name in ('steel', 'chrome', 'steel_ground'):
            bright = bright * (1 + brush[:, :, None] * .010)
        if name == 'paint':
            # Sparse worn-through paint scratches, not uniform brightening.
            scratch = np.clip((brush - 1.35) / 1.1, 0, 1)
            bright = bright * (1 + scratch[:, :, None] * .35)
        if name in ('paint', 'seat', 'steel'):
            # Fine orange-peel / cast surface variation.
            bright = bright * (1 + micro[:, :, None] * .030)
        color = base_rgb * bright
        if amount > 0:
            thr = .84 - .22 * amount
            rust_cov = np.clip((mid - thr) / .14, 0, 1) * amount
            spots = np.clip((fine - .930) / .07, 0, 1) * amount * .30
            cov = np.clip(rust_cov + spots, 0, .30)
            color = color * (1 - cov[:, :, None] * .70) + \
                rust_rgb[None, None, :] * cov[:, :, None] * .70
            rough = mat['roughness'] * (1 - cov) + .68 * cov + (fine - .5) * .025
            metal = mat['metallic'] * (1 - cov * .55)
            height = detail * .0008 + (brush * .020 if name != 'rubber' else 0) + \
                cov * (fine - .5) * .15
        else:
            rough = np.full((tile, tile), mat['roughness'], np.float32) + \
                (fine - .5) * .022
            metal = np.full((tile, tile), mat['metallic'], np.float32)
            height = (micro - .5) * .16
        if name == 'rubber':
            color = base_rgb * (1 + (fine[:, :, None] - .5) * .26 +
                                (micro[:, :, None] - .5) * .18 +
                                (coarse[:, :, None] - .5) * .10)
            height = (micro - .5) * .55 + (fine - .5) * .25
            metal = np.zeros((tile, tile), np.float32)
            rough = np.full((tile, tile), mat['roughness'], np.float32) + \
                (fine - .5) * .04
        if name == 'dark':
            color = base_rgb * (1 + (micro[:, :, None] - .5) * .30 +
                                (fine[:, :, None] - .5) * .22 +
                                (coarse[:, :, None] - .5) * .10)
            height = (micro - .5) * .55
        put(idx, color, rough, metal, height,
            clip_min=2 if name in ('rubber', 'dark') else
            (8 if name in ('paint', 'seat') else 24))

    for idx in range(7, 16):
        put(idx, np.full((tile, tile, 3), 128, np.float32), .60, 0.0)

    paths = {}
    for name, arr in (('base_color', base), ('metallic_roughness', mr),
                      ('normal', normal)):
        p = out / f'{name}.png'
        Image.fromarray(arr.astype(np.uint8), 'RGB').save(p)
        paths[name] = str(p)

    sheet = Image.new('RGB', (size, size), (245, 245, 243))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(LABEL_FONT, 22)
    except OSError:
        font = ImageFont.load_default()
    names = profile['texture'].get('tile_names', {})
    for idx in range(16):
        col, row = idx % 4, idx // 4
        x, y = col * tile, row * tile
        sheet.paste(Image.fromarray(base[y:y + tile, x:x + tile]), (x, y))
        draw.rectangle((x, y, x + tile, y + tile), outline=(230, 230, 228), width=2)
        label = names.get(str(idx), '备用中性灰')
        draw.text((x + 10, y + 8), f'{idx} {label}', fill=(20, 20, 20), font=font)
    sheet.save(out / '材质图集-布局.jpg', quality=93)
    evidence = {
        'rust_palette_rgb': [int(v) for v in rust_rgb],
        'rust_brown_pixels': brown_px,
        'rust_strong_pixels': strong_px,
        'note': ('锈色取自已对齐做锈图中较饱和的棕色像素中位数；纹理分布为程序化近似，'
                 '不是逐像素投影。做锈图环孔内的黄色斑块判断为做锈伪影，未复制。'
                 '金属/粗糙度/法线均为合成 PBR，不含烘死的灯光或阴影。'),
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
        'source_silhouette_extent_px': [bbox[2] - bbox[0] + 1,
                                        bbox[3] - bbox[1] + 1],
        'source_silhouette_aspect_w_h': round((bbox[2] - bbox[0] + 1) /
                                              (bbox[3] - bbox[1] + 1), 4),
        'estimated_photo_tilt_deg': -8.7,
        'landmark_estimates_px': {
            'top_eye_center': [657, 140],
            'top_seat_top': 235,
            'spring_top': 300,
            'spring_bottom': 1090,
            'body_bottom': 880,
            'lower_seat_top': 1040,
            'bottom_eye_center': [490, 1230],
            'note': '人工从原照与网格叠加图读取，只能说明未标定像素比例。',
        },
        'scale_note': profile['scale_note'],
        'note': ('源图是单张商品照，不能作正交测量；这里的像素只能说明源视图轮廓范围。'
                 '候选尺寸来自配置中的比例推断，不是实测。'),
    }


def old_model_audit(row):
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
        'old_mesh_sha256': row.get('mesh_sha256'),
        'old_source_fidelity': (row.get('reconstruction') or {}).get('fidelity'),
        'old_status': row.get('status'),
        'old_reason': row.get('reason'),
        'library_repair': row.get('library_repair'),
        'note': ('旧模型源视角 IoU 0.9044 看似接近旧门槛，但 5 个孔全部被填，'
                 '六方向审计图显示弹簧与缸体粘连、两个安装环孔封闭、表面波浪。'
                 '故不复用旧网格，改用分部件重建。'),
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


def _iou(mask_a, mask_b):
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return round(inter / union, 4) if union else 0.0


def silhouette_diagnostics(out, source_alpha_path):
    source = (np.asarray(Image.open(source_alpha_path).convert('L')) > 128)
    target = _mask_fit(source.astype(np.uint8))
    rows = []
    for i in range(24):
        p = out / 'renders' / f'turn-{i:02d}.png'
        if not p.is_file():
            continue
        arr = np.asarray(Image.open(p).convert('RGBA'))
        fitted = _mask_fit((arr[:, :, 3] > 128).astype(np.uint8))
        rows.append({'index': i, 'azimuth_deg': i * 15,
                     'iou': _iou(fitted, target),
                     'render': str(p.relative_to(out)).replace('\\', '/')})
    rows.sort(key=lambda r: r['iou'], reverse=True)
    def single(name):
        p = out / 'renders' / name
        if not p.is_file():
            return None
        arr = np.asarray(Image.open(p).convert('RGBA'))
        fitted = _mask_fit((arr[:, :, 3] > 128).astype(np.uint8))
        return _iou(fitted, target)
    return {
        'method': ('把源图透明轮廓与预渲染轮廓分别按外接框缩放居中后计算 IoU。'
                   'strict_source 为原库相机（直立），tilted_source 为仅诊断用的'
                   '绕 X 轴 -8.7° 倾斜补偿，turntable 为 24 个 15° 方位、仰角 20°。'
                   '三者都未标定，只是诊断。'),
        'strict_source_iou': single('new-source.png'),
        'tilted_source_iou': single('new-source-tilt.png'),
        'turntable_best': rows[0] if rows else None,
        'turntable_top5': rows[:5],
        'diagnostic_only': True,
        'passed': None,
        'note': 'IoU 不是实测精度，不能证明尺寸、结构或装配关系；任何数值都不作为验收通过。',
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
    la = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
    lb = 0.2126 * b[:, :, 0] + 0.7152 * b[:, :, 1] + 0.0722 * b[:, :, 2]
    diff = np.abs(la[mask] - lb[mask])

    def centroid(luma):
        thresh = np.percentile(luma[mask], 97)
        yy, xx = np.where(mask & (luma >= thresh))
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
        'note': ('同相机、仅移动主光的金属隔离渲染（钢衬套、亮内套、镀铬杆）。'
                 '只证明高光与亮度随光位变化，不证明材质参数是实物的实测值。'),
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
        tile_ok = bool(luma.std() > 1.0 and luma.mean() > 8 and
                       rough.min() >= 0.03 and rough.max() <= 0.97 and
                       abs(float(metal.mean()) - spec['metallic']) < 0.20)
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
    return {
        'tiles': rows,
        'passed': bool(ok),
        'note': ('图集检查只证明贴图非纯色、粗糙度/金属度范围和七类材质区域存在；'
                 '不证明它们是实物的逐像素测量。做锈图中环孔内的黄色斑块未使用。'),
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
    cn = {'front': '正视（环孔）', 'back': '背视（环孔）',
          'left': '左视（环体侧面）', 'right': '右视（环体侧面）',
          'top': '顶视', 'bottom': '底视'}
    items = []
    for v in VIEWS:
        items.append((_load_rgba_on_gray(rd / f'new-{v}.png'), f'新候选 · {cn[v]}',
                      '分部件重建'))
    for v in VIEWS:
        items.append((_load_rgba_on_gray(rd / f'old-{v}.png'), f'原模型 · {cn[v]}',
                      '旧网格，各视图按自身包围盒取景'))
    _board(items, 6, out / '六视图-新旧对照.jpg',
           '28 液压减震器 1 · 六方向新旧对照（新候选上排 / 原模型下排）', cell=300,
           note=('原模型为旧 TripoSR 网格；上下排取景尺度不同，不能据此比较尺寸。'
                 '六视图参考未标定。'))

    detail_text = {
        'top-eye': '上安装环：黑外环/钢衬套/亮内套',
        'top-eye-bore': '上环通孔（真实开孔）',
        'top-seat-notch': '上座盘与边缘缺口',
        'spring-coils': '弹簧圈距、线径与缸体间隙',
        'body-rod': '缸体下端与镀铬活塞杆',
        'lower-seat': '下座盘、橡胶环与杆孔',
        'bottom-eye': '下安装环：黑外环/钢衬套/亮内套',
        'bottom-eye-bore': '下环通孔（真实开孔）',
        'overall-3q': '整体 3/4 视角',
        'overall-side': '整体侧视（环体侧面）',
    }
    _board([(_load_rgba_on_gray(rd / f'detail-{n}.png'), detail_text[n],
             '原照可见/推断')
            for n in DETAILS], 4, out / '局部检查.jpg',
           '28 液压减震器 1 · 关键局部检查', cell=300)

    iso_text = {
        'seats-only-top': '座盘 3/4 俯视（弹簧座盘/缺口）',
        'seats-only-bottom': '座盘 3/4 仰视',
        'top-eye-only': '上环与衬套隔离',
        'bottom-eye-only': '下环与衬套隔离',
        'spring-only': '弹簧隔离（圈间空隙）',
        'body-rod-only': '缸体/腺体/镀铬杆隔离',
    }
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), iso_text[n], '部件隔离')
            for n in iso_text], 3, out / '分部件隔离.jpg',
           '28 液压减震器 1 · 分部件隔离检查', cell=320,
           note='隔离图只显示所列部件；不证明部件间装配关系。')

    open_text = {
        'top-eye-only': '上环：黑环/钢衬套/内套通孔',
        'bottom-eye-only': '下环：黑环/钢衬套/内套通孔',
        'spring-only': '弹簧：中心轴与圈间空隙',
        'body-rod-only': '缸体/杆：杆穿出而非粘连',
    }
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), open_text[n], '隔离渲染')
            for n in open_text], 2, out / '开孔-结构隔离检查.jpg',
           '28 液压减震器 1 · 开孔/结构隔离检查', cell=360,
           note=('两个环孔与弹簧中心轴另由 Blender BVH 射线检查记录；'
                 '隔离图本身不是装配验收。'))

    mat_items = [
        ('new-material-front', '新候选 · 材料正视'),
        ('new-clay-front', '新候选 · 灰模正视'),
        ('new-clay-3q', '新候选 · 灰模 3/4'),
        ('new-light-alternate', '同一正面、主光换位'),
        ('detail-overall-3q', '新候选 · 材料 3/4'),
    ]
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), t, '同相机')
            for n, t in mat_items], 3, out / '材质-同光照对照.jpg',
           '28 液压减震器 1 · 同光照材质 / 灰模对照', cell=300,
           note='灰模只检查形体；材质图不含烘焙灯光或阴影。')

    metal_items = [
        ('metal-only-light-a', '金属隔离 · 主光 A'),
        ('metal-only-light-b', '金属隔离 · 主光 B'),
    ]
    _board([(_load_rgba_on_gray(rd / f'{n}.png'), t, '仅金属件，同相机')
            for n, t in metal_items], 2, out / '金属反射-变光对照.jpg',
           '28 液压减震器 1 · 金属反射变光检查', cell=420,
           note='高光移动是金属响应的证据，不等于实物金属参数的实测。')

    turns = [(_load_rgba_on_gray(rd / f'turn-{i:02d}.png', 220), f'{i * 15:03d}°', '')
             for i in range(24)]
    _board(turns, 6, out / '转台-24方向.jpg',
           '28 液压减震器 1 · 24 方向预渲染转台（仰角约 20°）', cell=220,
           note='仅外观；不证明隐藏结构、装配或落稳。')


def source_board(out, original_path, reference_path, silhouette):
    rd = out / 'renders'
    # The source photo has a transparent background; composite it on light grey
    # for display instead of letting convert('RGB') produce a black panel.
    orig = _load_rgba_on_gray(original_path, 430)
    ref = Image.open(reference_path).convert('RGB')
    ref.thumbnail((560, 400), Image.LANCZOS)
    best = silhouette.get('turntable_best') or {}
    best_img_path = out / best.get('render', 'renders/turn-21.png')
    best_img = _load_rgba_on_gray(best_img_path, 430)
    tilt_img = _load_rgba_on_gray(rd / 'new-source-tilt.png', 430)
    canvas = Image.new('RGB', (1520, 620), (240, 243, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), '28 液压减震器 1 · 原照 / 新候选 / 未标定六视图参考',
              fill=(22, 28, 24), font=_font(20, True))
    canvas.paste(orig, (10, 55))
    draw.text((10, 585), '原照（单张商品图，约 8.7° 倾斜）',
              fill=(30, 36, 32), font=_font(14, True))
    canvas.paste(tilt_img, (470, 55))
    draw.text((470, 585), '新候选 · 倾斜补偿的源视角（仅诊断）',
              fill=(30, 36, 32), font=_font(14, True))
    canvas.paste(best_img, (940, 55))
    sub = (f"转台最佳 {best.get('azimuth_deg', '—')}°，IoU "
           f"{best.get('iou', '—')}（诊断）")
    draw.text((940, 585), '新候选 · 最接近的转台方向', fill=(30, 36, 32),
              font=_font(14, True))
    canvas.save(out / '原照-新候选对照.jpg', quality=92)

    canvas2 = Image.new('RGB', (1250, 520), (238, 241, 238))
    draw = ImageDraw.Draw(canvas2)
    draw.text((12, 10), '28 液压减震器 1 · 原视角轮廓诊断（未标定）',
              fill=(22, 28, 24), font=_font(18, True))
    diag = Image.open(out / 'renders' / 'source-alpha-diagnostic.png')
    canvas2.paste(diag, (10, 48))
    draw.text((12, 470), f"直立源视角 IoU {silhouette.get('strict_source_iou', '—')} / "
                         f"倾斜补偿 IoU {silhouette.get('tilted_source_iou', '—')} "
                         f"（仅诊断）", fill=(95, 72, 38), font=_font(13))
    canvas2.save(out / '原视角诊断.jpg', quality=92)


def diagnostic_image(out, source_alpha_path):
    source = (np.asarray(Image.open(source_alpha_path).convert('L')) > 128)
    target = _mask_fit(source.astype(np.uint8))

    def fitted(name):
        arr = np.asarray(Image.open(out / 'renders' / name).convert('RGBA'))
        return _mask_fit((arr[:, :, 3] > 128).astype(np.uint8))

    strict = fitted('new-source.png')
    tilted = fitted('new-source-tilt.png')
    canvas = np.zeros((256, 256 * 4 + 30, 3), np.uint8)
    canvas[:] = (238, 241, 238)
    panels = [('source', target), ('strict-fit', strict),
              ('tilt-overlap', None), ('tilt-fit', tilted)]
    x = 0
    for name, mask in panels:
        if name == 'tilt-overlap':
            p = np.zeros((256, 256, 3), np.uint8)
            p[:, :] = (40, 44, 40)
            p[np.logical_and(target > 0, tilted > 0)] = (60, 150, 90)
            p[np.logical_and(target > 0, tilted == 0)] = (190, 70, 55)
            p[np.logical_and(target == 0, tilted > 0)] = (70, 110, 200)
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
        '# 28 液压减震器 1 分部件重建候选',
        '',
        '本候选为 **manual_only / 待审**，不替换原模型，不进入自动放置或正式合成。',
        '',
        '## 方法',
        '',
        '以原照可见结构为主，六视图只作未标定的外观与背侧拓扑参考。旧模型为单图',
        'TripoSR 网格，源视角 IoU 0.9044，但 5 个孔全部被填，六方向审计显示弹簧与',
        '缸体粘连、两个安装环孔封闭、表面波浪，故不复用旧网格。新候选按部件建立：',
        '两个带黑外环/锈蚀钢衬套/亮内套的安装环（中心通孔保留）、上下座盘、黑色上',
        '缸体、下镀铬活塞杆、12 圈螺旋弹簧（圈间空隙和中心轴保留）、橡胶刮尘环与',
        '下座橡胶缓冲环。',
        '',
        '## 实际执行的检查',
        '',
        '| 检查 | 结果 |',
        '| --- | --- |',
    ]
    for key, value in checks.items():
        if isinstance(value, (dict, list)):
            continue
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
        '- `mesh.glb`：分部件候选（Z 向上、长轴 Z、环孔轴 X）',
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
        p.error('this entry supports item 28 only')
    a.reference = (a.reference or find_reference()).resolve()
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    if a.out is None:
        a.out = (ROOT / 'work/direct-deepseek/28-draft' if a.draft
                 else ROOT / 'outputs/28液压减震器1-六视图重建-v5')
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
    config_path = HERE / 'shock28_config.json'
    worker_path = HERE / 'mesh_rebuild_28_worker.py'
    checks_path = HERE / 'checks_rebuild_28.py'
    page_path = HERE / 'review28_page.py'
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
    old_audit = old_model_audit(row)
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
        'diagnostic_only': True, 'passed': None, 'strict_source_iou': None,
        'tilted_source_iou': None, 'turntable_best': None, 'turntable_top5': [],
        'note': 'draft 未渲染转台'}
    if not a.draft:
        diagnostic_image(a.out, alpha_path)
        build_boards(a.out)
        source_board(a.out, original_path, a.reference, silhouette)
    metal = metal_reflection_check(a.out) if not a.draft else {
        'changed': None, 'note': 'draft 未渲染变光'}
    tex_checks = texture_checks(textures, profile)
    bores = worker['ray_checks']['eye_bores']
    bores_ok = all(row['through'] for eye in bores.values() for row in eye.values())
    required_renders = ([f'new-{v}.png' for v in VIEWS] +
                        [f'old-{v}.png' for v in VIEWS] +
                        [f'turn-{i:02d}.png' for i in range(24)] +
                        [f'detail-{n}.png' for n in DETAILS] +
                        [f'{n}.png' for n in ISOLATIONS] +
                        ['source-alpha-diagnostic.png', 'new-source.png',
                         'new-source-tilt.png', 'old-overall-3q.png'])
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
        'eye_bores_through': bores_ok,
        'eye_bore_hits': bores,
        'spring_axis_open': worker['ray_checks']['spring_axis_open'],
        'spring_coil_turns': worker.get('spring_turns'),
        'texture_checks_passed': tex_checks['passed'],
        'metal_highlights_move': metal.get('changed'),
        'required_renders_present': renders_present,
        'source_silhouette_strict_iou_diagnostic': silhouette.get('strict_source_iou'),
        'source_silhouette_tilted_iou_diagnostic': silhouette.get('tilted_source_iou'),
        'source_silhouette_turntable_best_iou_diagnostic':
            (silhouette.get('turntable_best') or {}).get('iou'),
        'source_silhouette_turntable_best_azimuth':
            (silhouette.get('turntable_best') or {}).get('azimuth_deg'),
        'old_source_view_iou_diagnostic': (old_audit.get('old_source_fidelity') or {}).get('iou'),
        'manual_visual_review': 'not_performed',
        'webgl_interactive_viewer': 'not_available_in_this_worktree',
    }
    pending = [
        '人工视觉复核（本件只自动生成并检查，未由人眼验收外观）。',
        '六视图未标定，且顶视/底视的环孔方向与原照矛盾；尺寸、圈数、线径、座盘厚度、孔深和装配关系未实测。',
        '原照是单张商品图且物体倾斜约 8.7°；背面、上下环内部衬套配合、缸体内部和真实行程为推断。',
        '做锈图两个环孔内的亮黄色斑块判断为做锈伪影，未建模为贴纸；真实表面是否有小标签未确认。',
        '上座盘边缘缺口、下座橡胶缓冲环、橡胶刮尘环和缸体下端导向结构为推断。',
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
        'shape_basis': 'source_photo_visible_structure+uncalibrated_six_view_cross_check',
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
            'eye_bores': bores,
            'spring_axis_open': worker['ray_checks']['spring_axis_open'],
            'note': ('Blender BVH 射线检查；只覆盖两个安装环和内圈空心轴，'
                     '不证明部件间装配关系。'),
        },
        'pending_review': pending,
        'limitations': profile['limitations'],
        'files': {
            'mesh': 'mesh.glb', 'blend': 'model.blend',
            'base_color': 'base_color.png',
            'metallic_roughness': 'metallic_roughness.png',
            'normal': 'normal.png',
            'result': 'result.json', 'hashes': 'hashes.json',
            'worker_result': 'worker-result.json', 'request': 'request.json',
            'report': 'report.md', 'page': '检查页.html',
            'board_six_view': '六视图-新旧对照.jpg',
            'board_details': '局部检查.jpg',
            'board_parts': '分部件隔离.jpg',
            'board_material': '材质-同光照对照.jpg',
            'board_metal': '金属反射-变光对照.jpg',
            'board_openings': '开孔-结构隔离检查.jpg',
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

    browser = {'status': 'not_run', 'reason': 'draft run'}
    if not a.draft and page_path.is_file():
        sys.path.insert(0, str(HERE))
        import review28_page
        review28_page.build_page(a.out)
        browser = review28_page.browser_check(a.out)
    result['browser_check'] = {
        'status': browser.get('status'),
        'reason': browser.get('reason'),
        'result': browser.get('result'),
        'checked_at': browser.get('checked_at'),
    }
    (a.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    # Rebuild the page once so its footer shows the recorded browser status.
    # The self-test itself already ran on the identical UI above.
    if not a.draft and page_path.is_file():
        review28_page.build_page(a.out)

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
            'next_number': 29,
        }
        progress_path = ROOT / 'work/direct-deepseek/28-progress.json'
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                 encoding='utf8')
    print(json.dumps({
        'status': 'review', 'out': str(a.out),
        'mesh': str(a.out / 'mesh.glb'),
        'parts': mesh_info['parts'],
        'closed': mesh_info['closed_after_welding_seams'],
        'self_intersections': worker['self_intersection_total'],
        'eye_bores_through': bores_ok,
        'spring_axis_open': worker['ray_checks']['spring_axis_open'],
        'strict_source_iou': silhouette.get('strict_source_iou'),
        'tilted_source_iou': silhouette.get('tilted_source_iou'),
        'best_turntable_iou': (silhouette.get('turntable_best') or {}).get('iou'),
        'metal_changed': metal.get('changed'),
        'browser': browser.get('status'),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
