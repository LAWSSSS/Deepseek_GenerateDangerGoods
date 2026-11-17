"""Build the review-only red aerosol-can candidate (item 16).

Evidence:
- the clean/rust HD photographs of one used red slim 600 ml aerosol can;
- the user six-view sheet, matched by appearance only and explicitly
  uncalibrated (visual_match_unverified), used for the unseen back-label layout
  and for cap-scoop / bottom-dish topology, never for measurements.

The old TripoSR library mesh has a strongly elliptical top cross-section and a
smeared label; it is retained unchanged and is not used as the candidate's base
geometry.  The candidate is a parametric part rebuild with a true cylinder, a
closed cap solid with an oval scoop, a raised elliptical rim, a low actuator
with a real nozzle through-hole, a dark cap/body seam and a stamped aluminium
base with a concave dish.  All candidates remain manual-only and pending human
review; nothing replaces the library mesh or enters automatic placement.
"""
import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import PIL
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from scene_v2 import digest
from mesh_checks import source_fidelity
from mesh_rebuild_item import inspect_mesh, mask_fit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
REFERENCE_SHA256 = '3b51fb0bb0402247bb2ed868d7456afb57f15348939c043fe32c62d1b4bced56'
REFERENCE_NAME = 'ChatGPT 图像 2026年10月14日 10_08_37-1.png'
CLEAN_SHA256 = 'cc51d2528c792ecd13aa59c984f16e76a680dbb660715b2ab444cbca3dd1d497'
RUST_SHA256 = '1f3f60e5c9ee2d118b0f7e97211197defe84916d5636c86735b34b2bdae4e26d'
OLD_MESH_SHA256 = '118c17e1652aee982154edf8128ce3a3616a3b7712a00b7a983bf6115c895aec'
SOURCE_ALPHA_SHA256 = '5614f9bff5b81b2f4320bd885c3d861f3cd2e8f6c6f9c400050a68337099cfbc'
FONT = 'C:/Windows/Fonts/msyh.ttc'
FONTB = 'C:/Windows/Fonts/msyhbd.ttc'


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def find_reference():
    candidates = [ROOT / 'references/parallel-six-view', ROOT / 'references']
    for folder in candidates:
        if not folder.is_dir():
            continue
        direct = folder / REFERENCE_NAME
        if direct.is_file() and digest(direct) == REFERENCE_SHA256:
            return direct
        for path in sorted(folder.glob('*.png')):
            if digest(path) == REFERENCE_SHA256:
                return path
    raise FileNotFoundError('registered six-view reference not found in this worktree')


def current_worktree_path(value):
    """Relocate a library-record absolute path into this parallel worktree.

    The shared results.jsonl was written in the original main worktree.  This
    task must only read files inside the current worktree, so a path such as
    .../scrap-composite/outputs/... is remapped to this worktree's matching
    outputs/... file.  The recorded SHA-256 is verified after relocation.
    """
    path = Path(value)
    if path.is_file():
        resolved = path.resolve()
        if resolved == ROOT or ROOT in resolved.parents:
            return resolved
    parts = path.parts
    for base in ('outputs', 'references', 'work', 'external'):
        if base in parts:
            relative = Path(*parts[parts.index(base):])
            candidate = (ROOT / relative).resolve()
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f'input from the model library has no matching file in this worktree: {path.name}')


def relative_input(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def library_row(number=16):
    for line in LIBRARY.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('number') == number:
            return row
    raise KeyError(f'item {number} not found in the model library')


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONTB if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------
def height_to_normal(height, strength):
    h = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 0.8)
    dx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx = np.clip(-dx * strength, -0.30, 0.30)
    ny = np.clip(-dy * strength, -0.30, 0.30)
    nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny))
    return np.clip((np.stack((nx, ny, nz), axis=2) * 0.5 + 0.5) * 255,
                   0, 255).astype(np.uint8)


def orm_image(roughness, metallic):
    height, width = roughness.shape
    red = np.full((height, width), 255, np.uint8)
    return np.stack((red,
                     np.clip(roughness * 255, 0, 255).astype(np.uint8),
                     np.clip(metallic * 255, 0, 255).astype(np.uint8)),
                    axis=2)


def body_red_reference(clean_rgb, mask):
    rgb = clean_rgb[mask]
    red = rgb[np.logical_and.reduce((rgb[:, 0] > 90,
                                    rgb[:, 0] > rgb[:, 1] * 1.5,
                                    rgb[:, 0] > rgb[:, 2] * 1.4))]
    if len(red) < 100:
        return np.array([205, 8, 24], np.float32)
    return np.median(red, axis=0).astype(np.float32)


def draw_back_label(image, config):
    """Redraw an explicitly inferred back label onto the cylindrical UV image.

    The source photograph only shows the front.  The uncalibrated six-view
    reference shows a white instruction panel with a red header and a barcode;
    this drawing reproduces that layout with generic Chinese copy, not a claim
    of the actual printed wording.
    """
    width, height = image.size
    draw = ImageDraw.Draw(image)
    sampling = config['texture_sampling']
    u0, u1 = sampling['back_label_u']
    x0, x1 = int(u0 * width), int(u1 * width)
    pad = 22
    y0, y1 = 24, height - 20
    draw.rounded_rectangle((x0, y0, x1, y1), radius=12,
                           fill=(242, 239, 231), outline=(228, 223, 211), width=2)
    inner_l, inner_r = x0 + pad, x1 - pad
    cx = (x0 + x1) // 2

    def center(text, y, fnt, fill):
        box = draw.textbbox((0, 0), text, font=fnt)
        draw.text((cx - (box[2] - box[0]) / 2, y), text, font=fnt, fill=fill)

    center('AESTARI', y0 + 22, font(40, True), (196, 18, 28))
    center('中山凯达', y0 + 66, font(25, True), (196, 18, 28))
    draw.line((inner_l, y0 + 102, inner_r, y0 + 102), fill=(210, 205, 194), width=1)

    band_top = y0 + 116
    draw.rectangle((x0 + 8, band_top, x1 - 8, band_top + 54), fill=(198, 14, 24))
    center('克害威  杀虫气雾剂', band_top + 10, font(31, True), (255, 250, 236))
    draw.text((x1 - 128, band_top + 60), '600ml', font=font(20, True),
              fill=(170, 22, 30))

    y = band_top + 92
    sections = (
        ('产品特点', (
            '本品由拟除虫菊酯配制，对蚊、蝇、蟑螂等',
            '卫生害虫有效。低毒配方，气味清新。',
            '室内使用，作用迅速。',
        )),
        ('使用方法', (
            '使用前摇匀。距目标约 20-30 cm 喷射。',
            '避免向人体、食物及餐具喷射。',
            '喷洒时保持罐体直立。',
        )),
        ('注意事项', (
            '易燃，远离火源、热源。',
            '置于儿童不能触及处；勿投入火中或加热。',
            '禁止敲击、刺穿或焚烧罐体。',
            '空罐请按当地规定处理。',
        )),
        ('有效成分', (
            '拟除虫菊酯 0.35%（质量分数）',
            '剂型：气雾剂    净含量：600 ml',
            '外观：红色气雾罐；气味：清香型。',
        )),
        ('质量等级', (
            '合格品    执行标准：GB/T 18419',
            '生产日期/批号：见罐底    保质期：3 年',
            '生产许可证号：见罐身喷码。',
        )),
        ('产品用途', (
            '适用于家庭、宾馆、办公室等室内环境。',
            '喷洒后离开房间，约 30 分钟后通风。',
            '请勿用于食品加工区域。',
        )),
        ('中毒急救', (
            '如误服，立即携带本罐就医。',
            '皮肤接触后请用肥皂水清洗。',
            '如不慎入眼，用清水冲洗并就医。',
        )),
        ('储存运输', (
            '避免暴晒、雨淋和高温运输。',
            '贮存于阴凉、通风、干燥处。',
            '运输时防止剧烈碰撞。',
        )),
        ('废弃物处置', (
            '空罐不得刺穿或焚烧。',
            '请按当地规定分类回收。',
            '请勿将空罐交给儿童玩耍。',
        )),
    )
    body_font = font(21)
    header_font = font(24, True)
    line_step = 30
    for title, lines in sections:
        if y > y1 - 260:
            break
        draw.rectangle((inner_l, y, inner_l + 104, y + 31), fill=(198, 14, 24))
        draw.text((inner_l + 8, y + 3), title, font=header_font,
                  fill=(255, 250, 236))
        y += 40
        for line in lines:
            draw.text((inner_l + 6, y), line, font=body_font, fill=(40, 39, 36))
            y += line_step
        y += 16

    # A warning-icon row and a red service band fill the lower middle of the
    # inferred back label.
    if y < y1 - 260:
        icon_y = y + 8
        labels = ('喷洒', '禁火', '通风', '儿童')
        for i, text in enumerate(labels):
            cx_icon = inner_l + 34 + i * 86
            draw.ellipse((cx_icon - 28, icon_y, cx_icon + 28, icon_y + 56),
                         fill=(238, 206, 38), outline=(30, 30, 30), width=2)
            box = draw.textbbox((0, 0), text, font=font(19, True))
            draw.text((cx_icon - (box[2] - box[0]) / 2, icon_y + 16), text,
                      font=font(19, True), fill=(30, 30, 30))
        y = icon_y + 78
    if y < y1 - 210:
        draw.rectangle((inner_l, y, inner_r, y + 40), fill=(198, 14, 24))
        center('室内卫生杀虫剂 · 使用前请阅读说明', y + 7, font(22, True),
               (255, 250, 236))
        y += 52

    y = max(y + 8, y1 - 170)
    draw.line((inner_l, y, inner_r, y), fill=(210, 205, 194), width=1)
    draw.text((inner_l + 4, y + 8), '中山凯达日化有限公司', font=font(19, True),
              fill=(42, 41, 38))
    draw.text((inner_l + 4, y + 34), '地址：广东省中山市   净含量：600ml',
              font=font(17), fill=(55, 53, 49))
    rng = np.random.default_rng(16016)
    barcode_y = y1 - 78
    barcode_x = inner_l + 8
    barcode_w = min(300, (inner_r - inner_l) // 2)
    pos = barcode_x
    while pos < barcode_x + barcode_w:
        bar = int(rng.integers(1, 5))
        gap = int(rng.integers(1, 4))
        draw.rectangle((pos, barcode_y, pos + bar - 1, barcode_y + 48),
                       fill=(30, 30, 30))
        pos += bar + gap
    draw.text((barcode_x, barcode_y + 50), '6 923456 789012', font=font(16),
              fill=(45, 45, 45))
    draw.rectangle((inner_r - 118, barcode_y - 8, inner_r, barcode_y + 54),
                   outline=(170, 166, 155), width=1)
    draw.text((inner_r - 108, barcode_y + 2), '可回收', font=font(17, True),
              fill=(70, 120, 70))
    draw.text((inner_r - 108, barcode_y + 25), '600 ml', font=font(16),
              fill=(70, 120, 70))
    return image


def make_body_texture(clean_path, rust_path, out, config):
    clean = Image.open(clean_path).convert('RGBA')
    rust = Image.open(rust_path).convert('RGBA')
    sampling = config['texture_sampling']
    if list(clean.size) != sampling['clean_size']:
        raise ValueError('16 clean source dimensions changed; sampling rects need review')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    clean_rgb = np.asarray(clean)[:, :, :3].astype(np.float32)
    rust_rgb = np.asarray(rust)[:, :, :3].astype(np.float32)
    alpha = np.asarray(clean.getchannel('A'))
    width = height = 2048
    cx = sampling['body_cx_px']
    half = sampling['body_half_width_px']
    top, bottom = sampling['body_top_px'], sampling['body_bottom_px']
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    theta = np.clip(2 * np.pi * (u - sampling['front_u_center']),
                    -np.pi / 2, np.pi / 2)
    map_x = np.tile(cx + half * np.sin(theta) * 0.985, (height, 1)).astype(np.float32)
    map_y = np.tile((top + v * (bottom - top))[:, None], (1, width)).astype(np.float32)
    front = cv2.remap(clean_rgb, map_x, map_y, cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_REPLICATE)
    front_rust = cv2.remap(rust_rgb, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
    luma = front.mean(axis=2)
    rust_luma = front_rust.mean(axis=2)
    grime = np.clip((luma - rust_luma - 4.0) / 55.0, 0, 1)
    rust_sat = np.clip((front_rust.max(axis=2) - front_rust.min(axis=2)) / 90.0, 0, 1)
    grime = np.clip(grime * (0.35 + 0.65 * rust_sat), 0, 0.55)
    grime = gaussian_filter(grime, 1.2)
    grime_rgb = np.array([112, 86, 64], np.float32)
    rgb = front * (1 - grime[:, :, None] * 0.24) + \
        grime_rgb[None, None, :] * (grime[:, :, None] * 0.24)
    rng = np.random.default_rng(16001)
    noise = gaussian_filter(rng.normal(size=(height, width)).astype(np.float32), 1.0)
    rgb = np.clip(rgb * (1 + noise[:, :, None] * 0.012), 0, 255)

    body_mask = np.zeros(alpha.shape, bool)
    body_mask[int(top):int(bottom), int(cx - half):int(cx + half)] = \
        alpha[int(top):int(bottom), int(cx - half):int(cx + half)] > 200
    red = body_red_reference(clean_rgb, body_mask)
    back = np.tile(red, (height, width, 1))
    back *= (1 + noise[:, :, None] * 0.020)
    rgb[:, width // 2:] = back[:, width // 2:]
    rgb[:, :width // 2] = np.clip(rgb[:, :width // 2], 0, 255)

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert('RGB')
    image = draw_back_label(image, config)
    image = image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for k in range(26):
        x = int(rng.integers(40, width - 40))
        y = int(rng.integers(60, height - 60))
        length = int(rng.integers(20, 120))
        angle = rng.uniform(-0.35, 0.35)
        dx, dy = int(length * math.cos(angle)), int(length * math.sin(angle))
        light = bool(rng.random() > 0.35)
        color = (232, 224, 212, int(rng.integers(10, 30))) if light else \
            (100, 26, 26, int(rng.integers(8, 22)))
        odraw.line((x, y, x + dx, y + dy), fill=color, width=1)
    image = Image.alpha_composite(image, overlay).convert('RGB')
    rgb = np.asarray(image).astype(np.float32)
    rgb = np.clip(rgb * (1 + noise[:, :, None] * 0.006), 0, 255)
    image = Image.fromarray(rgb.astype(np.uint8))

    luma_final = rgb.mean(axis=2)
    high = luma_final - gaussian_filter(luma_final, 2.5)
    height_field = np.clip(0.55 * grime + 0.18 * high / 24.0 + 0.02 * noise, 0, 1.4)
    normal = height_to_normal(height_field, 0.42)
    roughness = np.clip(0.285 + 0.090 * grime + 0.020 * noise, 0.22, 0.52)
    metallic = np.zeros_like(roughness)
    mr = orm_image(roughness, metallic)
    paths = {}
    for name, array in (('body_base', np.asarray(image)),
                        ('body_mr', mr), ('body_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(luma_final.mean()), 2),
        'min_luma': round(float(luma_final.min()), 2),
        'dark_fraction_below_20': round(float((luma_final < 20).mean()), 4),
        'mean_chroma': round(float((rgb.max(axis=2) - rgb.min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'front_mean_luma': round(float(luma_final[:, :width // 2].mean()), 2),
        'back_mean_luma': round(float(luma_final[:, width // 2:].mean()), 2),
        'note': ('Front is projected from the clean photo; back is red plus an '
                 'explicitly inferred white instruction label. Grime is derived '
                 'from the rust photo difference and clamped.'),
    }
    return paths, stats


def make_cap_texture(clean_path, rust_path, out, config):
    clean = Image.open(clean_path).convert('RGBA')
    rust = Image.open(rust_path).convert('RGBA')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    sampling = config['texture_sampling']
    c = np.asarray(clean)
    r = np.asarray(rust)
    x0, x1 = 372, 657
    y0, y1 = int(sampling['cap_top_px']), int(sampling['cap_bottom_px'])
    cap = c[y0:y1, x0:x1, :3].astype(np.float32)
    cap_a = c[y0:y1, x0:x1, 3]
    cap_rust = r[y0:y1, x0:x1, :3].astype(np.float32)
    mask = cap_a > 200
    red = cap[mask]
    red = red[np.logical_and.reduce((red[:, 0] > 90,
                                     red[:, 0] > red[:, 1] * 1.4,
                                     red[:, 0] > red[:, 2] * 1.3))]
    base_red = np.median(red, axis=0) if len(red) > 100 else np.array([190, 9, 23])
    size = 1024
    rng = np.random.default_rng(16002)
    flow = cv2.resize(rng.normal(size=(20, 96)).astype(np.float32), (size, size),
                      interpolation=cv2.INTER_CUBIC)
    flow = cv2.GaussianBlur(flow, (0, 0), 2.0)
    peel = cv2.GaussianBlur(rng.normal(size=(size, size)).astype(np.float32), (0, 0), 1.1)
    luma = cap.mean(axis=2)
    rust_luma = cap_rust.mean(axis=2)
    grime_small = np.clip((luma - rust_luma - 3.0) / 45.0, 0, 1) * mask
    grime_small = cv2.GaussianBlur(grime_small.astype(np.float32), (0, 0), 2.0)
    grime = cv2.resize(grime_small, (size, size), interpolation=cv2.INTER_CUBIC)
    grime = gaussian_filter(grime, 1.0)
    rgb = np.tile(base_red, (size, size, 1))
    rgb *= (1 + flow[:, :, None] * 0.013 + peel[:, :, None] * 0.0055)
    rgb = rgb * (1 - grime[:, :, None] * 0.10) + \
        np.array([105, 72, 55], np.float32)[None, None, :] * (grime[:, :, None] * 0.10)
    rgb = np.clip(rgb, 0, 255)
    image = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    seam_x = int(round(config['features']['cap_mould_seam_texture_u'] * size))
    draw.line((seam_x - 2, 0, seam_x - 2, size), fill=(225, 96, 104), width=1)
    draw.line((seam_x - 1, 0, seam_x - 1, size), fill=(118, 3, 12), width=1)
    for k in range(14):
        x = int(rng.integers(20, size - 20))
        y = int(rng.integers(20, size - 20))
        length = int(rng.integers(12, 60))
        color = (232, 190, 185) if rng.random() > 0.4 else (120, 8, 16)
        draw.line((x, y, x, min(size - 1, y + length)), fill=color, width=1)
    rgb = np.asarray(image).astype(np.float32)
    texture_grime = np.clip(0.55 * grime + 0.10 * np.abs(peel), 0, 1)
    normal = height_to_normal(0.30 * texture_grime + 0.10 * peel, 0.30)
    roughness = np.clip(0.240 + 0.035 * peel + 0.075 * grime, 0.17, 0.40)
    metallic = np.zeros_like(roughness)
    mr = orm_image(roughness, metallic)
    paths = {}
    for name, array in (('cap_base', rgb.astype(np.uint8)),
                        ('cap_mr', mr), ('cap_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(rgb.mean()), 2),
        'mean_chroma': round(float((rgb.max(axis=2) - rgb.min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'min_roughness': round(float(roughness.min()), 4),
        'max_roughness': round(float(roughness.max()), 4),
        'base_red_sampled': [round(float(v), 1) for v in base_red],
        'note': ('Red moulded plastic, dielectric. Flow lines, orange peel, a '
                 'surface mould seam and faint grime are synthesised; the '
                 'roughness range is inferred.'),
    }
    return paths, stats


def make_base_texture(clean_path, rust_path, out, config):
    clean = Image.open(clean_path).convert('RGBA')
    rust = Image.open(rust_path).convert('RGBA')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    sampling = config['texture_sampling']
    c = np.asarray(clean)
    r = np.asarray(rust)
    y0 = int(sampling['metal_top_px'])
    y1 = int(sampling['metal_bottom_px'])
    metal = c[y0:y1, 380:650, :3].astype(np.float32)
    metal_a = c[y0:y1, 380:650, 3]
    metal_rust = r[y0:y1, 380:650, :3].astype(np.float32)
    sat = metal.max(axis=2) - metal.min(axis=2)
    silver = metal[np.logical_and(metal_a > 200, sat < 30)]
    sample = np.median(silver, axis=0) if len(silver) > 100 else np.array([150, 150, 158])
    size = 1024
    rng = np.random.default_rng(16003)
    base_color = np.array([176, 180, 188], np.float32) * 0.85 + sample * 0.15
    angular = cv2.resize(rng.normal(size=(1, 256)).astype(np.float32), (size, size),
                         interpolation=cv2.INTER_CUBIC)
    brushed = cv2.resize(rng.normal(size=(512, 1)).astype(np.float32), (size, size),
                         interpolation=cv2.INTER_CUBIC)
    stains = cv2.resize(rng.random((40, 40)).astype(np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    stains = cv2.GaussianBlur(stains, (0, 0), 10.0)
    stains = np.clip((stains - 0.55) * 2.4, 0, 1)
    luma = metal.mean(axis=2)
    rust_luma = metal_rust.mean(axis=2)
    rust_diff = np.clip((luma - rust_luma - 2.0) / 40.0, 0, 1) * (metal_a > 200)
    rust_diff = cv2.GaussianBlur(rust_diff.astype(np.float32), (0, 0), 2.0)
    rust_stain = cv2.resize(rust_diff, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = np.tile(base_color, (size, size, 1))
    rgb *= (1 + angular[:, :, None] * 0.018 + brushed[:, :, None] * 0.025)
    rgb = rgb * (1 - stains[:, :, None] * 0.06) + \
        np.array([120, 110, 95], np.float32)[None, None, :] * (stains[:, :, None] * 0.06)
    rgb = rgb * (1 - rust_stain[:, :, None] * 0.12) + \
        np.array([112, 82, 58], np.float32)[None, None, :] * (rust_stain[:, :, None] * 0.12)
    rgb = np.clip(rgb, 0, 255)
    normal = height_to_normal(0.28 * angular + 0.22 * brushed + 0.20 * stains, 0.24)
    roughness = np.clip(0.390 + 0.045 * brushed + 0.055 * stains + 0.035 * rust_stain,
                        0.26, 0.58)
    metallic = np.ones_like(roughness)
    mr = orm_image(roughness, metallic)
    paths = {}
    for name, array in (('base_base', rgb.astype(np.uint8)),
                        ('base_mr', mr), ('base_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(rgb.mean()), 2),
        'mean_chroma': round(float((rgb.max(axis=2) - rgb.min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'min_metallic': round(float(metallic.min()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'min_roughness': round(float(roughness.min()), 4),
        'max_roughness': round(float(roughness.max()), 4),
        'sampled_silver': [round(float(v), 1) for v in sample],
        'note': ('Stamped aluminium: metallic is 1 everywhere. Brush marks, '
                 'smudges and rim stains are synthesised/inferred; the source '
                 'photo only shows the base under one light.'),
    }
    return paths, stats


def texture_checks(body_stats, cap_stats, base_stats):
    body_ok = (40 <= body_stats['mean_luma'] <= 200 and
               body_stats['mean_metallic'] <= 0.02 and
               0.20 <= body_stats['mean_roughness'] <= 0.55 and
               body_stats['dark_fraction_below_20'] <= 0.30)
    cap_ok = (60 <= cap_stats['mean_luma'] <= 205 and
              cap_stats['mean_metallic'] <= 0.02 and
              0.15 <= cap_stats['mean_roughness'] <= 0.40)
    base_ok = (base_stats['mean_metallic'] >= 0.95 and
               0.15 <= base_stats['mean_roughness'] <= 0.55 and
               60 <= base_stats['mean_luma'] <= 235)
    return {
        'body': {'passed': bool(body_ok), **body_stats},
        'cap': {'passed': bool(cap_ok), **cap_stats},
        'base': {'passed': bool(base_ok), **base_stats},
        'note': ('Body print is dielectric (metallic 0) while the stamped base '
                 'is metallic 1. The checks reject black/low-roughness texture '
                 'sheets; they do not prove the hidden surface appearance.'),
    }


# ---------------------------------------------------------------------------
# Mesh / render checks
# ---------------------------------------------------------------------------
def load_named_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


def load_parts(path):
    return list(load_named_parts(path).values())


def combined_mesh(path):
    return trimesh.util.concatenate(load_parts(path))


def glb_audit(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack('<III', data[:12])
    if magic != 0x46546C67 or length != len(data):
        raise ValueError('not a valid GLB container')
    offset = 12
    document = None
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack('<II', data[offset:offset + 8])
        chunk = data[offset + 8:offset + 8 + chunk_length]
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode('utf-8'))
        offset += 8 + chunk_length
    if document is None:
        raise ValueError('GLB has no JSON chunk')
    materials = []
    for material in document.get('materials', []):
        pbr = material.get('pbrMetallicRoughness', {})
        materials.append({
            'name': material.get('name'),
            'base_color_texture': 'baseColorTexture' in pbr,
            'metallic_roughness_texture': 'metallicRoughnessTexture' in pbr,
            'normal_texture': 'normalTexture' in material,
        })
    textured = [item for item in materials if item['name'] in ('body', 'cap', 'base')]
    images = document.get('images', [])
    passed = (len(textured) == 3 and
              all(item['base_color_texture'] and item['metallic_roughness_texture'] and
                  item['normal_texture'] for item in textured) and
              len(images) >= 9 and
              all(image.get('mimeType') == 'image/png' for image in images))
    return {
        'passed': bool(passed),
        'version': version,
        'materials': materials,
        'image_count': len(images),
        'texture_count': len(document.get('textures', [])),
        'node_count': len(document.get('nodes', [])),
        'mesh_count': len(document.get('meshes', [])),
        'note': ('The GLB embeds 9 PNG images and separate base-colour, '
                 'metallic-roughness and normal textures for body, cap and base. '
                 'Nozzle/seam use scalar PBR factors. This audit does not prove '
                 'the visual appearance in every viewer.'),
    }


def ray_hits(mesh, origin, direction, multiple=False):
    """Brute-force Moller-Trumbore ray test (rtree is not installed)."""
    triangles = mesh.triangles.astype(np.float64)
    v0 = triangles[:, 0, :]
    e1 = triangles[:, 1, :] - v0
    e2 = triangles[:, 2, :] - v0
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    direction = direction / max(1e-12, np.linalg.norm(direction))
    p = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, p)
    valid = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[valid] = 1.0 / det[valid]
    tvec = origin - v0
    u = np.einsum('ij,ij->i', tvec, p) * inv
    q = np.cross(tvec, e1)
    v = np.einsum('ij,j->i', q, direction) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    ok = valid & (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & \
        (u + v <= 1 + 1e-9) & (t > 1e-9)
    if not ok.any():
        return np.empty((0, 3))
    points = origin + t[ok, None] * direction
    if multiple:
        return points[np.argsort(t[ok])]
    return points[np.argmin(t[ok])][None, :]


def ray_checks(path, config):
    mesh = combined_mesh(path)
    out = {}
    origins = np.array([[0.006, 0.012, 0.40], [0.021, 0.000, 0.40],
                        [0.011, 0.0025, 0.2945], [0.011, 0.0037, 0.2945],
                        [0.000, 0.000, -0.20], [0.024, 0.000, -0.20]])
    directions = np.array([[0, 0, -1], [0, 0, -1],
                           [-1, 0, 0], [-1, 0, 0],
                           [0, 0, 1], [0, 0, 1]], float)
    hits = {}
    for ray_index in range(len(origins)):
        hits[ray_index] = ray_hits(mesh, origins[ray_index], directions[ray_index])
    floor = hits[0][0][2] if len(hits[0]) else None
    top = hits[1][0][2] if len(hits[1]) else None
    scoop_depth = (top - floor) if floor is not None and top is not None else None
    out['cap_scoop'] = {
        'floor_z_m': round(float(floor), 6) if floor is not None else None,
        'outside_top_z_m': round(float(top), 6) if top is not None else None,
        'depth_m': round(float(scoop_depth), 6) if scoop_depth is not None else None,
        'passed': bool(scoop_depth is not None and scoop_depth > 0.0015),
        'note': ('Ray down inside the oval scoop (outside the actuator) hits '
                 'lower than a ray down on the intact cap top. It confirms a '
                 'real cavity, not a painted dark oval.'),
    }
    hits_multi = {}
    for ray_index, origin, direction in ((0, origins[2], directions[2]),
                                         (1, origins[3], directions[3])):
        hits_multi[ray_index] = ray_hits(mesh, origin, direction, multiple=True)
    centre = hits_multi[0]
    offset = hits_multi[1]
    centre_outside = [p for p in centre if p[0] > config['features']['nozzle']['x1'] + 0.0002]
    offset_inside = [p for p in offset
                     if config['features']['nozzle']['x0'] - 0.0002 <= p[0] <=
                     config['features']['nozzle']['x1'] + 0.0002]
    out['nozzle_hole'] = {
        'centre_hits_outside_insert': len(centre_outside),
        'offset_hits_on_insert': len(offset_inside),
        'centre_hits_x_m': [round(float(p[0]), 6) for p in centre],
        'offset_hits_x_m': [round(float(p[0]), 6) for p in offset],
        'passed': bool(len(centre_outside) == 0 and len(offset_inside) >= 1),
        'note': ('Centre ray along the nozzle axis is clear beyond the insert, '
                 'while an offset ray hits the insert wall. This proves a real '
                 'through-hole; its diameter is inferred.'),
    }
    centre_dish = hits[4][0][2] if len(hits[4]) else None
    rim = hits[5][0][2] if len(hits[5]) else None
    concavity = (centre_dish - rim) if centre_dish is not None and rim is not None else None
    out['bottom_dish'] = {
        'centre_z_m': round(float(centre_dish), 6) if centre_dish is not None else None,
        'rim_z_m': round(float(rim), 6) if rim is not None else None,
        'centre_minus_rim_m': round(float(concavity), 6) if concavity is not None else None,
        'passed': bool(concavity is not None and concavity > 0.0010),
        'note': ('Ray up at the base centre hits a higher surface than a ray up '
                 'near the rim, confirming the concave stamped dish.'),
    }
    return out


def alpha_bbox_ratio(path):
    with Image.open(path) as image:
        alpha = np.asarray(image.getchannel('A')) >= 128
    ys, xs = np.nonzero(alpha)
    if not len(xs):
        return None
    return round(float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)), 5)


def median_luma(path):
    with Image.open(path) as image:
        rgba = np.asarray(image.convert('RGBA')).astype(np.float32)
    mask = rgba[:, :, 3] >= 128
    if not mask.any():
        return None
    luma = rgba[:, :, :3].mean(axis=2)[mask]
    return round(float(np.median(luma)), 2)


def mean_difference(path_a, path_b):
    a = np.asarray(Image.open(path_a).convert('RGBA')).astype(np.float32)
    b = np.asarray(Image.open(path_b).convert('RGBA')).astype(np.float32)
    if a.shape != b.shape:
        raise ValueError('render sizes differ')
    mask = (a[:, :, 3] >= 128) & (b[:, :, 3] >= 128)
    if not mask.any():
        return None
    return round(float(np.abs(a[:, :, :3] - b[:, :, :3])[mask].mean()), 4)


def crop_alpha(image):
    image = image.convert('RGBA')
    box = image.getchannel('A').getbbox()
    return image.crop(box) if box else image


def fit_tile(image, size, background='#f4f6f6', crop=True):
    image = image.convert('RGBA')
    if crop:
        image = crop_alpha(image)
    factor = min((size[0] - 24) / max(1, image.width),
                 (size[1] - 24) / max(1, image.height))
    image = image.resize((max(1, round(image.width * factor)),
                          max(1, round(image.height * factor))),
                         Image.Resampling.LANCZOS)
    tile = Image.new('RGBA', size, background)
    tile.alpha_composite(image, ((size[0] - image.width) // 2,
                                 (size[1] - image.height) // 2))
    return tile.convert('RGB')


# ---------------------------------------------------------------------------
# Reference panels and boards
# ---------------------------------------------------------------------------
def reference_panels(reference):
    image = Image.open(reference).convert('RGB')
    width, height = image.size
    panels = {}
    for index, view in enumerate(VIEWS):
        x0 = (index % 3) * width // 3
        y0 = (index // 3) * height // 2
        panel = image.crop((x0 + 8, y0 + 8, x0 + width // 3 - 8,
                            y0 + height // 2 - 8))
        rgb = np.asarray(panel)
        mask = (rgb.min(axis=2) < 205).astype(np.uint8)
        mask[:4] = 0
        mask[-4:] = 0
        mask[:, :4] = 0
        mask[:, -4:] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count < 2:
            panels[view] = panel
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.flatnonzero(areas >= max(150, areas.max() * 0.04)) + 1
        clean = np.isin(labels, keep).astype(np.uint8)
        ys, xs = np.nonzero(clean)
        if not len(xs):
            panels[view] = panel
            continue
        pad = 8
        panels[view] = panel.crop((max(0, int(xs.min()) - pad),
                                   max(0, int(ys.min()) - pad),
                                   min(panel.width, int(xs.max()) + pad),
                                   min(panel.height, int(ys.max()) + pad)))
    return panels


def panel_mask(panel):
    rgb = np.asarray(panel.convert('RGB'))
    mask = (rgb.min(axis=2) < 205).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(150, areas.max() * 0.04)) + 1
    return np.isin(labels, keep).astype(np.uint8)


def mask_iou(a, b):
    if a is None or b is None:
        return None
    return round(float(np.logical_and(a, b).sum() /
                       max(1, np.logical_or(a, b).sum())), 4)


def make_six_board(out, panels):
    title = font(28, True)
    label = font(19, True)
    small = font(15)
    tile = (700, 430)
    board = Image.new('RGB', (3 * tile[0] + 40, 6 * (tile[1] + 48) + 90), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 14), '16 杀虫剂罐2 · 六方向：未标定视觉参考 / 原模型 / 新候选（同一中性光）',
              font=title, fill='#17232a')
    results = {}
    for i, view in enumerate(VIEWS):
        y = 64 + i * (tile[1] + 48)
        ref = panels[view]
        old = Image.open(out / 'renders' / f'old-{view}.png')
        new = Image.open(out / 'renders' / f'new-{view}.png')
        ref_fit = mask_fit(panel_mask(ref))
        new_fit = mask_fit(np.asarray(new.convert('RGBA').getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.convert('RGBA').getchannel('A')) >= 128)
        results[view] = {
            'reference_vs_new_mask_iou': mask_iou(ref_fit, new_fit),
            'reference_vs_old_mask_iou': mask_iou(ref_fit, old_fit),
            'old_vs_new_mask_iou': mask_iou(old_fit, new_fit),
            'diagnostic_only': True,
        }
        draw.text((14, y), f'{view.upper()}   参考（未标定）', font=label, fill='#35454c')
        draw.text((14 + tile[0], y), '原模型', font=label, fill='#35454c')
        draw.text((14 + 2 * tile[0], y), '新候选', font=label, fill='#35454c')
        board.paste(fit_tile(ref, tile), (14, y + 26))
        board.paste(fit_tile(old, tile), (14 + tile[0], y + 26))
        board.paste(fit_tile(new, tile), (14 + 2 * tile[0], y + 26))
        note = (f'诊断 IoU：参考/新 {results[view]["reference_vs_new_mask_iou"]} · '
                f'参考/旧 {results[view]["reference_vs_old_mask_iou"]} · '
                f'旧/新 {results[view]["old_vs_new_mask_iou"]}（未标定，不作验收）')
        draw.text((14, y + tile[1] + 30), note, font=small, fill='#5a6b72')
    board.save(out / '六视图-新旧对照.jpg', quality=93)
    return results


def make_detail_board(out):
    names = ('cap-top', 'cap-recess', 'recess-rim', 'nozzle', 'cap-seam',
             'front-label', 'back-label', 'body', 'body-surface', 'bottom',
             'bottom-dish', 'metal-rim', 'surface', 'new-source')
    titles = {
        'cap-top': '盖顶/椭圆形凹槽', 'cap-recess': '凹槽内部',
        'recess-rim': '凹槽凸边', 'nozzle': '喷嘴真实通孔',
        'cap-seam': '盖身接缝', 'front-label': '正面标签',
        'back-label': '背面标签（推断）', 'body': '筒身',
        'body-surface': '标签表面细节', 'bottom': '金属底',
        'bottom-dish': '底部凹碟', 'metal-rim': '卷边金属',
        'surface': '表面微纹理/划痕', 'new-source': '原视角诊断渲染',
    }
    tile = (620, 470)
    cols = 4
    rows = math.ceil(len(names) / cols)
    board = Image.new('RGB', (cols * tile[0] + 30, rows * (tile[1] + 44) + 70),
                      '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '16 杀虫剂罐2 · 关键局部检查（新版，中性光）',
              font=font(26, True), fill='#17232a')
    for i, name in enumerate(names):
        x = 10 + (i % cols) * tile[0]
        y = 56 + (i // cols) * (tile[1] + 44)
        filename = name if name == 'new-source' else 'detail-' + name
        image = Image.open(out / 'renders' / (filename + '.png'))
        board.paste(fit_tile(image, tile, '#ffffff'), (x, y))
        draw.text((x + 4, y + tile[1] + 8), f'{name} · {titles[name]}',
                  font=font(17, True), fill='#34444b')
    board.save(out / '局部检查.jpg', quality=93)


def make_isolated_board(out, ray):
    items = (('cap-only', '盖/凹槽/执行器/喷嘴（隔离）'),
             ('cap-only-top', '盖顶凹槽（隔离顶视）'),
             ('base-only', '金属底（隔离）'),
             ('base-only-bottom', '金属底凹碟（隔离底视）'),
             ('body-only', '筒身（隔离）'),
             ('new-top', '新版顶视轮廓（圆度检查）'))
    tile = (620, 500)
    board = Image.new('RGB', (3 * tile[0] + 30, 2 * (tile[1] + 50) + 190), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '16 杀虫剂罐2 · 隔离结构与开口检查（渲染 + 射线）',
              font=font(26, True), fill='#17232a')
    for i, (name, title) in enumerate(items):
        x = 10 + (i % 3) * tile[0]
        y = 56 + (i // 3) * (tile[1] + 50)
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile,
                             '#ffffff'), (x, y))
        draw.text((x + 4, y + tile[1] + 8), title, font=font(18, True),
                  fill='#34444b')
    y = 56 + 2 * (tile[1] + 50) + 6
    draw.text((18, y), '隔离开口射线检查', font=font(22, True), fill='#17232a')
    lines = [
        f"凹槽：内部地板 z={ray['cap_scoop']['floor_z_m']} m，外部盖顶 z="
        f"{ray['cap_scoop']['outside_top_z_m']} m，深度={ray['cap_scoop']['depth_m']} m",
        f"喷嘴：中心射线在插入件外侧命中 {ray['nozzle_hole']['centre_hits_outside_insert']} 次；"
        f"偏移射线命中插入件 {ray['nozzle_hole']['offset_hits_on_insert']} 次",
        f"底部：中心 z={ray['bottom_dish']['centre_z_m']} m，近边缘 z="
        f"{ray['bottom_dish']['rim_z_m']} m，中心高于边缘="
        f"{ray['bottom_dish']['centre_minus_rim_m']} m",
        '说明：射线只证明这些位置的局部几何关系，不能证明壁厚、内部装配或隐藏面。',
    ]
    for line in lines:
        y += 30
        draw.text((24, y), line, font=font(17), fill='#46585f')
    board.save(out / '结构开口-隔离检查.jpg', quality=93)


def make_material_boards(out):
    tile = (960, 720)
    board = Image.new('RGB', (2 * tile[0] + 30, tile[1] + 100), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '16 杀虫剂罐2 · 同光照材质对照（左：原模型，右：新候选）',
              font=font(26, True), fill='#17232a')
    for i, name in enumerate(('old-material-close', 'new-material-close')):
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile,
                             '#ffffff'), (10 + i * tile[0], 60))
        draw.text((16 + i * tile[0], tile[1] + 70),
                  '原模型（TripoSR，保留未改）' if i == 0 else '新候选（PBR 贴图）',
                  font=font(19, True), fill='#34444b')
    board.save(out / '材质-同光照对照.jpg', quality=94)

    board = Image.new('RGB', (2 * tile[0] + 30, 2 * tile[1] + 150), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '16 杀虫剂罐2 · 变光检查（高光必须随现场光移动）',
              font=font(26, True), fill='#17232a')
    pairs = (('new-material-close', '主光 A · 表面近景'),
             ('new-surface-alternate', '主光 B · 表面近景'),
             ('new-front', '主光 A · 正视'),
             ('new-light-alternate', '主光 B · 正视'))
    for i, (name, title) in enumerate(pairs):
        x = 10 + (i % 2) * tile[0]
        y = 60 + (i // 2) * (tile[1] + 40)
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile,
                             '#ffffff'), (x, y))
        draw.text((x + 4, y + tile[1] + 6), title, font=font(18, True),
                  fill='#34444b')
    board.save(out / '材质-变光检查.jpg', quality=94)

    tile2 = (760, 620)
    board = Image.new('RGB', (2 * tile2[0] + 30, 2 * tile2[1] + 150), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '16 杀虫剂罐2 · 表面反射变光对照（金属底 / 塑料高光）',
              font=font(25, True), fill='#17232a')
    pairs = (('metal-base-key', '金属底 · 主光 A'),
             ('metal-base-alt', '金属底 · 主光 B'),
             ('new-material-close', '塑料盖/标签 · 主光 A'),
             ('new-surface-alternate', '塑料盖/标签 · 主光 B'))
    for i, (name, title) in enumerate(pairs):
        x = 10 + (i % 2) * tile2[0]
        y = 60 + (i // 2) * (tile2[1] + 40)
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile2,
                             '#ffffff'), (x, y))
        draw.text((x + 4, y + tile2[1] + 6), title, font=font(18, True),
                  fill='#34444b')
    board.save(out / '表面反射-变光对照.jpg', quality=94)


def make_turntable(out):
    tile = (330, 330)
    board = Image.new('RGB', (6 * tile[0] + 24, 4 * (tile[1] + 30) + 60), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 10), '16 杀虫剂罐2 · 新候选 24 方向转台（中性光，仅供检查）',
              font=font(24, True), fill='#17232a')
    for i in range(24):
        x = 8 + (i % 6) * tile[0]
        y = 52 + (i // 6) * (tile[1] + 30)
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png')
        board.paste(fit_tile(image, tile, '#ffffff'), (x, y))
        draw.text((x + 4, y + tile[1] + 4), f'{i * 15:03d}°',
                  font=font(14, True), fill='#52646b')
    board.save(out / '转台-24方向.jpg', quality=93)


def make_source_board(out, source_photo, old_source_view=None):
    tile = (600, 760)
    board = Image.new('RGB', (3 * tile[0] + 30, tile[1] + 110), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '16 杀虫剂罐2 · 源图、旧模型原视角与新候选原视角诊断',
              font=font(25, True), fill='#17232a')
    old_image = None
    if old_source_view is not None and Path(old_source_view).is_file():
        old_image = Image.open(old_source_view)
    images = (Image.open(source_photo), old_image,
              Image.open(out / 'renders/new-source.png'))
    titles = ('原照（干净）', '旧模型原视角渲染', '新候选原视角诊断（未标定，不作验收）')
    for i, image in enumerate(images):
        if image is None:
            continue
        board.paste(fit_tile(image, tile, '#ffffff'), (10 + i * tile[0], 58))
        draw.text((16 + i * tile[0], tile[1] + 70), titles[i],
                  font=font(18, True), fill='#34444b')
    board.save(out / '源图-原视角诊断.jpg', quality=93)


# ---------------------------------------------------------------------------
# Independent review page
# ---------------------------------------------------------------------------
def make_review_page(out, report, texture_stats):
    checks = report['checks']

    def check_row(name, value, note=''):
        if value is True:
            status, color = 'PASS', '#1f7a3e'
        elif value is False:
            status, color = 'FAIL / 待审', '#a33a2b'
        else:
            status, color = str(value), '#6a5a1e'
        return (f'<tr><td>{name}</td><td style="color:{color};font-weight:700">'
                f'{status}</td><td>{note}</td></tr>')

    rows = ''.join([
        check_row('网格封闭/无自交', checks['mesh_integrity'],
                  '逐部件封闭；同部件 BVH 无自交；装配件允许有意搭接'),
        check_row('喷嘴通孔与凹槽/底碟', checks['isolated_openings'],
                  '射线检查；见结构开口隔离图'),
        check_row('圆柱截面', checks['cylindrical_body'],
                  f"筒身最大半径偏差 {report['cylinder_check']['max_abs_radius_deviation_m']:.9f} m；"
                  f"顶视宽高比 {report['top_view_aspect']['new']}"),
        check_row('材质贴图', checks['material_textures'],
                  '正面标签来自原照；背面标签为未标定参考下的推断重绘'),
        check_row('GLB 材质内嵌', checks['glb_materials'],
                  '9 张 PNG；筒身/盖/底各有 baseColor、metallicRoughness、normal'),
        check_row('金属反射', checks['metal_reflection'],
                  '金属底 metallic=1；两主光下亮度变化'),
        check_row('塑料高光变光', checks['dielectric_response'], '盖/标签随主光移动'),
        check_row('原视角轮廓诊断', report['source_fidelity_diagnostic'].get('passed'),
                  f"IoU {report['source_fidelity_diagnostic']['iou']}，仅作诊断"),
        check_row('检查页静态资源', report['page_assets_check'].get('passed'),
                  '本地 HTML/图片路径检查；未以无头浏览器声称交互通过'),
        check_row('人工验收', 'pending', '候选 manual_only，不进入自动合成'),
    ])
    summary = (f"<p>本件以原照为主，六视图参考为未标定视觉参考。"
               f"旧模型顶视截面椭圆、标签糊化，故未复用旧网格。"
               f"候选保留真实筒身圆柱截面、椭圆形盖顶凹槽、凸边、低执行器与真实喷嘴通孔、"
               f"盖身接缝和带凹碟的金属底。背面标签为推断重绘，"
               f"参考图不等同实测。</p>")
    html = REVIEW_TEMPLATE
    replacements = {
        '__SUMMARY__': summary,
        '__CHECK_ROWS__': rows,
        '__TITLE__': '16 杀虫剂罐2 · 三维模型独立检查',
        '__SUBTITLE__': '红色细身杀虫气雾罐 · 分部件重建候选 · 未标定六视图参考',
        '__STATUS__': 'manual_only · 待人工审阅',
        '__VERSION__': report['version'],
        '__SOURCE_IOU__': str(report['source_fidelity_diagnostic'].get('iou')),
        '__TOP_ASPECT__': str(report['top_view_aspect']),
        '__BODY_LUMA__': str(texture_stats['body']['mean_luma']),
        '__CAP_LUMA__': str(texture_stats['cap']['mean_luma']),
        '__BASE_METAL__': str(texture_stats['base']['mean_metallic']),
        '__REFERENCE_NOTE__': report['reference_note'],
        '__PENDING__': '；'.join(report['pending_review']),
    }
    for key, value in replacements.items():
        html = html.replace(key, str(value))
    (out / 'review.html').write_text(html, encoding='utf8')


def check_review_page(out):
    html_path = out / 'review.html'
    if not html_path.is_file():
        return {'passed': False, 'reason': 'review.html missing', 'links': []}
    text = html_path.read_text(encoding='utf8')
    links = re.findall(r'(?:src|href)="([^"]+)"', text)
    missing = []
    local = []
    for link in links:
        if link.startswith(('http:', 'https:', 'data:', '#')) or link.startswith('mailto:'):
            continue
        path = (out / unquote(link.split('?')[0])).resolve()
        local.append(link)
        if not path.is_file():
            missing.append(link)
    required = ['renders/new-front.png', 'renders/old-front.png',
                'renders/clay-front.png', 'renders/turn-00.png',
                '六视图-新旧对照.jpg', '局部检查.jpg', '结构开口-隔离检查.jpg',
                '材质-同光照对照.jpg', '材质-变光检查.jpg',
                '表面反射-变光对照.jpg', '转台-24方向.jpg',
                'mesh.glb', 'model.blend', 'result.json', 'hashes.json',
                'progress.json']
    missing_required = [name for name in required if not (out / name).is_file()]
    return {
        'passed': bool(not missing and not missing_required and
                       'window.__ready' in text and 'turn-00.png' in text),
        'local_links': len(local),
        'missing_links': missing,
        'missing_required': missing_required,
        'note': ('Static local asset/link check only. No headless-browser '
                 'interaction is claimed by this record.'),
    }


REVIEW_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#18242b;--muted:#5a6b72;--line:#d8e0e2;--bg:#f3f6f6;--card:#fff;--accent:#8e1b24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Microsoft YaHei",system-ui,sans-serif}
header{padding:26px 32px 20px;background:linear-gradient(135deg,#1d2b31,#3b1014);color:#fff}
header h1{margin:0 0 6px;font-size:26px}header p{margin:0;color:#d9e3e4}
.badge{display:inline-block;margin-left:10px;padding:3px 10px;border:1px solid #e7b9b9;border-radius:99px;font-size:13px;color:#ffd9d9}
main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(330px,.75fr);gap:20px;padding:20px 24px 48px;max-width:1660px;margin:auto}
section,aside{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 6px 18px rgba(20,35,40,.04)}
h2{font-size:17px;margin:4px 0 12px;border-left:4px solid var(--accent);padding-left:9px}
.viewer-stage{background:radial-gradient(circle at 50% 35%,#fff,#e8ecec);border:1px solid var(--line);border-radius:12px;min-height:520px;display:flex;align-items:center;justify-content:center;overflow:hidden}
.viewer-stage img{max-width:100%;max-height:560px;display:block}
.controls{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:10px}
button{font:inherit;padding:7px 12px;border-radius:9px;border:1px solid #c9d3d5;background:#f8fafa;color:var(--ink);cursor:pointer}
button.selected{background:var(--ink);border-color:var(--ink);color:#fff}
button:hover{border-color:#88999d}
.help{color:var(--muted);font-size:13px;margin:8px 0 0}
.turn-wrap{margin-top:18px;border-top:1px dashed var(--line);padding-top:14px}
#turnstage{background:radial-gradient(circle at 50% 40%,#fff,#e6ebeb);border:1px solid var(--line);border-radius:12px;min-height:430px;display:flex;align-items:center;justify-content:center;cursor:ew-resize;user-select:none;touch-action:none}
#turn{max-width:100%;max-height:460px}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{border-bottom:1px solid #e5ebec;padding:7px 6px;vertical-align:top;text-align:left}
th{color:var(--muted);font-weight:600}
.notice{background:#fff7f2;border:1px solid #efd4c6;border-radius:10px;padding:10px 12px;color:#5c3b2c;font-size:13px;margin:12px 0}
ul.links{list-style:none;margin:0;padding:0;display:grid;gap:7px}
ul.links a{color:#145a7a;text-decoration:none}ul.links a:hover{text-decoration:underline}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}
.grid a{display:block;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#fafcfc;color:#2c454e;font-size:12px;text-decoration:none}
.grid img{width:100%;height:130px;object-fit:contain;background:#fff;display:block}
.grid span{display:block;padding:5px 7px}
.small{font-size:12px;color:var(--muted)}
code{background:#eef3f3;border-radius:5px;padding:1px 4px}
footer{padding:0 24px 40px;color:var(--muted);font-size:12px;text-align:center}
@media(max-width:1000px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>__TITLE__<span class="badge">__STATUS__</span></h1><p>__SUBTITLE__ · 版本 <code>__VERSION__</code></p></header>
<main>
<section>
<h2>新旧模型 / 灰模 · 六方向</h2>
<div class="controls" id="modelButtons">
<button data-mode="new" class="selected">新候选</button>
<button data-mode="old">原模型</button>
<button data-mode="clay">灰模</button>
</div>
<div class="controls" id="viewButtons">
<button data-view="front" class="selected">正视</button>
<button data-view="back">背视</button>
<button data-view="left">左视</button>
<button data-view="right">右视</button>
<button data-view="top">顶视</button>
<button data-view="bottom">底视</button>
</div>
<div class="viewer-stage"><img id="view" src="renders/new-front.png" alt="new front"></div>
<p class="help">按钮切换新旧/灰模和六方向。所有图片来自本工作树的固定中性光 Blender 渲染，不是网页实时材质。</p>
<div class="turn-wrap">
<h2>转台检查 · 拖动旋转</h2>
<div id="turnstage"><img id="turn" src="renders/turn-00.png" alt="turntable"></div>
<div class="controls" style="margin-top:10px">
<input id="turnRange" type="range" min="0" max="23" value="0" style="flex:1">
<button id="turnPlay">播放 / 暂停</button>
<span id="turnLabel" class="small">000°</span>
</div>
<p class="help">拖动转台图或拖动滑块查看 24 个方向；这是预渲染帧，不能替代实际场景中的光照与遮挡检查。</p>
</div>
</section>
<aside>
<h2>判读摘要</h2>
__SUMMARY__
<div class="notice"><b>参考状态：</b>六视图映射为 visual_match_unverified（未标定）。__REFERENCE_NOTE__ 背面标签、盖内配合、壁厚、喷嘴孔径/内部、金属底冲压细节和隐藏面为推断。候选为 <b>manual_only</b>，不替换原模型，不进入自动放置或正式合成。</div>
<h2>检查记录</h2>
<table><thead><tr><th>项目</th><th>结果</th><th>说明</th></tr></thead><tbody>__CHECK_ROWS__</tbody></table>
<div class="notice"><b>关键数字：</b>原视角诊断 IoU __SOURCE_IOU__（单张照片、未标定，只作诊断）；顶视宽高比 __TOP_ASPECT__；标签/筒身平均亮度 __BODY_LUMA__；盖平均亮度 __CAP_LUMA__；金属底平均金属度 __BASE_METAL__。</div>
<h2>对照与文件</h2>
<ul class="links">
<li><a href="六视图-新旧对照.jpg">六视图新旧对照</a></li>
<li><a href="局部检查.jpg">关键局部检查</a> · <a href="结构开口-隔离检查.jpg">隔离结构与开口</a></li>
<li><a href="材质-同光照对照.jpg">同光照材质对照</a> · <a href="材质-变光检查.jpg">变光检查</a> · <a href="表面反射-变光对照.jpg">表面反射变光</a></li>
<li><a href="转台-24方向.jpg">24 方向转台拼图</a> · <a href="源图-原视角诊断.jpg">源图/原视角诊断</a></li>
<li><a href="mesh.glb">下载 GLB</a> · <a href="model.blend">下载 Blender 可编辑文件</a></li>
<li><a href="result.json">模型验证 JSON</a> · <a href="hashes.json">输入/输出哈希</a> · <a href="progress.json">进度记录</a></li>
</ul>
<div class="grid">
<a href="六视图-新旧对照.jpg"><img src="六视图-新旧对照.jpg"><span>六视图新旧对照</span></a>
<a href="局部检查.jpg"><img src="局部检查.jpg"><span>关键局部</span></a>
<a href="材质-同光照对照.jpg"><img src="材质-同光照对照.jpg"><span>材质同光照对照</span></a>
<a href="结构开口-隔离检查.jpg"><img src="结构开口-隔离检查.jpg"><span>隔离结构与开口</span></a>
</div>
<h2>待人工复核</h2>
<p class="small">__PENDING__</p>
</aside>
</main>
<footer>独立检查页 · 不依赖共享服务端或共享索引 · 模型/贴图/渲染均为本工作树生成，等待串行整合。</footer>
<script>
const views=['front','back','left','right','top','bottom'];
let mode='new', currentView='front';
const viewImg=document.getElementById('view');
const viewButtons=[...document.querySelectorAll('#viewButtons button')];
const modeButtons=[...document.querySelectorAll('#modelButtons button')];
function refreshView(){
  viewImg.src=`renders/${mode}-${currentView}.png`;
  viewButtons.forEach(b=>b.classList.toggle('selected',b.dataset.view===currentView));
  modeButtons.forEach(b=>b.classList.toggle('selected',b.dataset.mode===mode));
  window.__mode=mode; window.__view=currentView;
}
viewButtons.forEach(b=>b.addEventListener('click',()=>{currentView=b.dataset.view;refreshView();}));
modeButtons.forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;refreshView();}));
const turnImg=document.getElementById('turn');
const turnRange=document.getElementById('turnRange');
const turnLabel=document.getElementById('turnLabel');
const frames=[];
for(let i=0;i<24;i++){const n=String(i).padStart(2,'0');const im=new Image();im.src=`renders/turn-${n}.png`;frames.push(im);}
let frame=0,playing=false,timer=null;
function setFrame(value){
  frame=(value%24+24)%24;
  turnImg.src=frames[frame].src;
  turnRange.value=frame;
  turnLabel.textContent=String(frame*15).padStart(3,'0')+'°';
  window.__frame=frame;
}
turnRange.addEventListener('input',()=>setFrame(Number(turnRange.value)));
let dragging=false,startX=0,startFrame=0;
const stage=document.getElementById('turnstage');
stage.addEventListener('pointerdown',e=>{dragging=true;startX=e.clientX;startFrame=frame;stage.setPointerCapture(e.pointerId);});
stage.addEventListener('pointermove',e=>{if(!dragging)return;const delta=Math.round((e.clientX-startX)/18);setFrame(startFrame-delta);});
stage.addEventListener('pointerup',()=>{dragging=false;});
stage.addEventListener('pointercancel',()=>{dragging=false;});
document.getElementById('turnPlay').addEventListener('click',()=>{
  playing=!playing;
  if(playing){timer=setInterval(()=>setFrame(frame+1),180);}
  else{clearInterval(timer);}
});
refreshView();setFrame(0);
window.__ready=true;
</script></body></html>
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', type=int, default=16)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    args = parser.parse_args()
    if args.number != 16:
        parser.error('this entry supports item 16 only')
    args.reference = (args.reference or find_reference()).resolve()
    if args.out is None:
        args.out = ROOT / 'outputs/16杀虫剂罐2-六视图重建-v1'
    args.out, args.library, args.blender = (x.resolve() for x in
                                            (args.out, args.library, args.blender))
    if digest(args.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = library_row(16)
    # The shared library record stores absolute main-worktree paths.  Relocate
    # every input into this parallel worktree and verify the registered hash
    # before using it.
    old_mesh_path = current_worktree_path(row['mesh'])
    if digest(old_mesh_path) != OLD_MESH_SHA256:
        raise ValueError('old library mesh hash differs from the registered value')
    clean_path = current_worktree_path(row.get('clean', row['original']))
    if digest(clean_path) != CLEAN_SHA256:
        raise ValueError('clean source hash differs from the registered value')
    rust_path = current_worktree_path(row['rust'])
    if digest(rust_path) != RUST_SHA256:
        raise ValueError('rust source hash differs from the registered value')
    source_view_path = current_worktree_path(row['source_view'])
    alpha_path = source_view_path.parent / 'source-alpha.png'
    if not alpha_path.is_file() or ROOT not in alpha_path.resolve().parents:
        raise FileNotFoundError('source alpha missing from this worktree')
    if digest(alpha_path) != SOURCE_ALPHA_SHA256:
        raise ValueError('source alpha hash differs from the model library')
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    source_alpha_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    config_path = HERE / 'spraycan16_config.json'
    worker_path = HERE / 'mesh_rebuild_16_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    blender_version = subprocess.run([str(args.blender), '--version'],
                                     capture_output=True, text=True,
                                     check=True).stdout.splitlines()[0]
    common_inputs = {
        'reference': digest(args.reference), 'mesh': OLD_MESH_SHA256,
        'clean': CLEAN_SHA256, 'rust': RUST_SHA256,
        'source_alpha': digest(alpha_path),
        'source_view': digest(source_view_path),
        'input_paths': {
            'mesh': relative_input(old_mesh_path),
            'clean': relative_input(clean_path),
            'rust': relative_input(rust_path),
            'source_view': relative_input(source_view_path),
            'source_alpha': relative_input(alpha_path),
            'reference': relative_input(args.reference),
        },
        'source_camera': row['reconstruction']['source_camera'],
        'config': digest(config_path), 'worker': digest(worker_path),
        'helpers': {name: digest(HERE / name) for name in
                    ('blender_worker.py', 'mesh_checks.py', 'mesh_rebuild_item.py')},
        'blender': blender_version, 'python': sys.version.split()[0],
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'pillow': PIL.__version__, 'trimesh': trimesh.__version__,
    }
    run_signature = hashlib.sha256(
        json.dumps(common_inputs, sort_keys=True).encode()).hexdigest()
    signature_input = {**common_inputs, 'builder': digest(Path(__file__))}
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    resume = False
    if args.out.exists() and any(args.out.iterdir()):
        record = args.out / 'result.json'
        mesh = args.out / 'mesh.glb'
        if record.is_file() and mesh.is_file():
            old = json.loads(record.read_text(encoding='utf8'))
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh):
                print(json.dumps({'status': 'review', 'cache_hit': True,
                                  'out': str(args.out)}, ensure_ascii=False))
                return
        run_record = args.out / 'run-signature.json'
        required = (args.out / 'request.json', args.out / 'texture-stats.json',
                    args.out / 'mesh.glb', args.out / 'worker-result.json',
                    args.out / 'model.blend')
        if run_record.is_file():
            run_data = json.loads(run_record.read_text(encoding='utf8'))
            if run_data.get('run_signature') == run_signature and \
                    all(path.is_file() for path in required):
                resume = True
        if not resume:
            raise FileExistsError('output exists with a different signature; use a new directory')

    if resume:
        request = json.loads((args.out / 'request.json').read_text(encoding='utf8'))
        textures = request['textures']
        stats = json.loads((args.out / 'texture-stats.json').read_text(encoding='utf8'))
        print(json.dumps({'status': 'resuming_rendered_output', 'out': str(args.out)},
                         ensure_ascii=False), flush=True)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'run-signature.json').write_text(
            json.dumps({'run_signature': run_signature,
                        'inputs': common_inputs}, ensure_ascii=False, indent=2),
            encoding='utf8')
        body_textures, body_stats = make_body_texture(clean_path, rust_path, args.out, config)
        cap_textures, cap_stats = make_cap_texture(clean_path, rust_path, args.out, config)
        base_textures, base_stats = make_base_texture(clean_path, rust_path, args.out, config)
        textures = {**body_textures, **cap_textures, **base_textures}
        stats = texture_checks(body_stats, cap_stats, base_stats)
        (args.out / 'texture-stats.json').write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf8')
        request = {
            'out': str(args.out), 'original_mesh': str(old_mesh_path),
            'textures': textures,
            'config': config, 'source_camera': row['reconstruction']['source_camera'],
            'source_alpha_bbox': source_alpha_bbox,
        }
        (args.out / 'request.json').write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
        with (args.out / 'blender.log').open('w', encoding='utf8') as log:
            process = subprocess.run(
                [str(args.blender), '--background', '--factory-startup',
                 '--python-exit-code', '1', '--python', str(worker_path), '--',
                 str(args.out / 'request.json')],
                stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        if process.returncode:
            raise RuntimeError(f'Blender failed; see {args.out / "blender.log"}')

    worker = json.loads((args.out / 'worker-result.json').read_text(encoding='utf8'))
    mesh_audit = inspect_mesh(args.out / 'mesh.glb')
    glb = glb_audit(args.out / 'mesh.glb')
    with Image.open(alpha_path) as alpha, \
            Image.open(args.out / 'renders/new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                   np.asarray(rendered.getchannel('A')))
    ray = ray_checks(args.out / 'mesh.glb', config)
    body_mesh = load_named_parts(args.out / 'mesh.glb').get('Can body')
    if body_mesh is None:
        raise ValueError('exported GLB is missing the Can body node')
    body_vertices = np.asarray(body_mesh.vertices)
    z_values = body_vertices[:, 2]
    side = body_vertices[np.logical_and(z_values > 0.0085, z_values < 0.247)]
    radii = np.hypot(side[:, 0], side[:, 1])
    radius = config['dimensions']['body_radius_m']
    rings = {}
    for vertex in side:
        rings.setdefault(round(float(vertex[2]), 6), []).append(
            float(np.hypot(vertex[0], vertex[1])))
    ring_variation = max((max(values) - min(values) for values in rings.values()),
                         default=0.0)
    cylinder_check = {
        'body_radius_m': radius,
        'side_vertex_count': int(len(side)),
        'max_abs_radius_deviation_m': round(float(np.max(np.abs(radii - radius))), 9),
        'max_ring_radius_variation_m': round(float(ring_variation), 9),
        'top_view_aspect_new': None,
        'note': ('Only the straight side band 0.0085 < z < 0.247 m is measured; '
                 'the chamfered ends are excluded. The check confirms cylindrical '
                 'symmetry of the modelled side, not measured dimensions.'),
    }
    new_top_ratio = alpha_bbox_ratio(args.out / 'renders/new-top.png')
    old_top_ratio = alpha_bbox_ratio(args.out / 'renders/old-top.png')
    cylinder_check['top_view_aspect_new'] = new_top_ratio
    new_front_luma = median_luma(args.out / 'renders/new-front.png')
    old_front_luma = median_luma(args.out / 'renders/old-front.png')
    metal_difference = mean_difference(args.out / 'renders/metal-base-key.png',
                                       args.out / 'renders/metal-base-alt.png')
    dielectric_difference = mean_difference(args.out / 'renders/new-material-close.png',
                                            args.out / 'renders/new-surface-alternate.png')
    panels = reference_panels(args.reference)
    for view, panel in panels.items():
        panel.save(args.out / 'renders' / f'ref-{view}.png')
    reference_results = make_six_board(args.out, panels)
    make_detail_board(args.out)
    make_isolated_board(args.out, ray)
    make_material_boards(args.out)
    make_turntable(args.out)
    make_source_board(args.out, clean_path, source_view_path)

    names = [part['name'] for part in worker['parts']]
    structural = {
        'bodies': sum(name == 'Can body' for name in names),
        'bases': sum(name == 'Can base' for name in names),
        'caps': sum(name == 'Cap shell' for name in names),
        'recess_rims': sum(name == 'Cap recess rim' for name in names),
        'actuators': sum(name == 'Actuator button' for name in names),
        'nozzles': sum(name == 'Nozzle insert' for name in names),
        'seam_rings': sum(name == 'Cap seam ring' for name in names),
        'note': ('Part presence proves only that a feature was modelled; cap '
                 'wall thickness, the scoop floor thickness, the nozzle bore '
                 'diameter and all hidden faces are inferred.'),
    }
    structural_ok = bool(structural['bodies'] == 1 and structural['bases'] == 1 and
                         structural['caps'] == 1 and structural['recess_rims'] == 1 and
                         structural['actuators'] == 1 and structural['nozzles'] == 1 and
                         structural['seam_rings'] == 1)
    cylindrical_ok = bool(cylinder_check['max_abs_radius_deviation_m'] < 1e-6 and
                          cylinder_check['max_ring_radius_variation_m'] < 1e-6 and
                          new_top_ratio is not None and abs(new_top_ratio - 1.0) < 0.03)
    metal_reflection = {
        'status': 'passed' if (stats['base']['mean_metallic'] >= 0.95 and
                               metal_difference is not None and metal_difference > 2.0)
                  else 'failed_or_inconclusive',
        'mean_metallic': stats['base']['mean_metallic'],
        'mean_abs_luma_difference_moved_key': metal_difference,
        'note': ('The stamped base is metallic 1 and its render changes under a '
                 'moved key light. This is a response test, not a claim about '
                 'unseen surface finish.'),
    }
    dielectric = {
        'status': 'passed' if (stats['body']['mean_metallic'] <= 0.02 and
                               stats['cap']['mean_metallic'] <= 0.02 and
                               dielectric_difference is not None and
                               dielectric_difference > 2.0)
                  else 'failed_or_inconclusive',
        'mean_abs_luma_difference_moved_key': dielectric_difference,
        'note': 'Printed body and moulded cap are dielectric; their highlights move with the key light.',
    }
    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] == 7 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components'] and
                               worker['cap_boundary_edges_after_boolean'] == 0),
        'isolated_openings': bool(all(ray[key]['passed'] for key in
                                      ('cap_scoop', 'nozzle_hole', 'bottom_dish'))),
        'structural_counts': structural_ok,
        'cylindrical_body': cylindrical_ok,
        'material_textures': bool(all(stats[key]['passed'] for key in ('body', 'cap', 'base'))),
        'glb_materials': glb['passed'],
        'metal_reflection': metal_reflection['status'] == 'passed',
        'dielectric_response': dielectric['status'] == 'passed',
        'reference_diagnostic_only': True,
        'source_fidelity': bool(fidelity.get('passed', False)),
        'browser_page': 'not_run',
    }

    # Original six-view audit row for provenance.
    audit_fidelity = None
    audit_path = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
    if audit_path.is_file():
        for line in audit_path.read_text(encoding='utf8').splitlines():
            if line.strip() and json.loads(line).get('number') == 16:
                audit_fidelity = json.loads(line).get('source_fidelity')
                break

    version = config['version']
    report = {
        'version': version, 'number': 16, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': config['shape_basis'], 'reference_role': config['reference_role'],
        'method': 'parametric_part_rebuild_from_source_photo_and_uncalibrated_six_view_reference',
        'reference': str(args.reference), 'reference_sha256': digest(args.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'original_mesh': str(old_mesh_path), 'original_mesh_sha256': OLD_MESH_SHA256,
        'source_photo': str(clean_path), 'source_view': str(source_view_path),
        'source_alpha': str(alpha_path), 'original_clean_sha256': CLEAN_SHA256,
        'original_rust_sha256': RUST_SHA256,
        'provenance_note': (
            'The shared model-library record contains absolute paths from the '
            'original main worktree. Every input was relocated to the matching '
            'file inside this parallel worktree before use, and its registered '
            'SHA-256 was verified. No output is written outside this worktree.'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9581 only measures one silhouette. '
            'The six-view audit explicitly records an elliptical top cross-section '
            'and a smeared label; the old mesh is therefore not reused as the '
            'candidate base geometry.'),
        'original_source_fidelity_record': audit_fidelity,
        'catalog_evidence': config['proportion_source'],
        'dimensions': config['dimensions'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(args.out / 'mesh.glb'), 'mesh_sha256': digest(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'), 'blend_sha256': digest(args.out / 'model.blend'),
        'textures': textures, 'texture_stats': stats,
        'material_profile': config['material_profile'],
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The source camera is the calibrated single front view from the old '
            'model library. It is a diagnostic silhouette comparison, not a '
            'calibrated turntable and not an acceptance gate.'),
        'reference_views': reference_results,
        'top_view_aspect': {'new': new_top_ratio, 'old': old_top_ratio,
                            'note': ('Width/height of the isolated top-view alpha '
                                     'silhouette. The old ellipse is recorded as the '
                                     'defect that motivated the rebuild.')},
        'render_luma': {'new_front_median': new_front_luma,
                        'old_front_median': old_front_luma,
                        'note': 'Neutral-light luma only; not a lighting acceptance.'},
        'mesh_audit': mesh_audit, 'glb_audit': glb, 'worker': worker,
        'cylinder_check': cylinder_check,
        'ray_checks': ray, 'structural_checks': structural,
        'metal_reflection_check': metal_reflection,
        'dielectric_response_check': dielectric,
        'checks': checks,
        'page_assets_check': check_review_page(args.out),
        'limitations': [
            'The six-view sheet is an uncalibrated generated visual reference, not a measured turntable; it is used for topology and the unseen back-label layout only.',
            'Only one front photograph exists. The photo is used as the primary front proportion and label source; the back, cap interior, scoop floor, nozzle bore and bottom stamping are inferred.',
            'Working dimensions (298 mm tall, 59.2 mm diameter) follow the photo pixel ratio and a similar-catalogue plausibility check; they are not measured dimensions.',
            'The back label is a new generic drawing based on the reference layout; its exact Chinese wording, barcode and symbols are not claimed as the real print.',
            'The cap is a closed solid with a subtracted oval scoop; the actual cap wall thickness, snap fit and internal actuator mechanism are hidden and not reproduced.',
            'The nozzle through-hole is real, but its diameter, length and internal channel are inferred; the spray pattern is not modelled.',
            'The aluminium base is modelled with a concave dish and rolled rim, but the exact stamped profile, alloy, coating and hidden underside details are uncertain.',
            'The body print is projected from a single view; wrap-around registration, hidden print and exact ink gloss vary and were not measured.',
            'Assemblies intentionally overlap at the cap/body and actuator/scoop joints; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views, cap scoop, rim, seam, nozzle and bottom dish against the photo and the physical sample.',
            'Confirm the actual can height/diameter and the cap/scoop dimensions with calipers or a measured drawing.',
            'Confirm the unseen back label wording, barcode and print registration from the real can.',
            'Confirm cap wall thickness, snap fit, actuator mechanism and nozzle bore with the real part.',
            'Confirm the stamped aluminium base profile, coating and rim geometry.',
            'Verify coordinate/scale, stability, floor contact and lighting integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
        ],
        'next_number': 17,
    }
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf8')

    make_review_page(args.out, report, stats)
    # The static page check needs the files the page links to.  Write the
    # preliminary records, check every local link, then finalise the record and
    # hashes so the hash sheet describes the final files.
    (args.out / 'progress.json').write_text('{}', encoding='utf8')
    (args.out / 'hashes.json').write_text('{}', encoding='utf8')
    page_check = check_review_page(args.out)
    report['page_assets_check'] = page_check
    report['checks']['review_page_assets'] = page_check['passed']
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf8')

    progress = {
        'number': 16, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review', 'version': version,
        'out': str(args.out), 'result': str(args.out / 'result.json'),
        'mesh': str(args.out / 'mesh.glb'), 'blend': str(args.out / 'model.blend'),
        'textures': textures, 'review_page': str(args.out / 'review.html'),
        'comparison_board': str(args.out / '六视图-新旧对照.jpg'),
        'detail_board': str(args.out / '局部检查.jpg'),
        'isolated_board': str(args.out / '结构开口-隔离检查.jpg'),
        'checks': checks, 'texture_stats': stats, 'ray_checks': ray,
        'cylinder_check': cylinder_check, 'glb_audit': glb,
        'top_view_aspect': report['top_view_aspect'],
        'source_fidelity_diagnostic': fidelity,
        'reference_role': config['reference_role'],
        'manual_only': True,
        'pending_review': report['pending_review'], 'next_number': 17,
        'notes': ('The six-view sheet is an uncalibrated visual reference; the '
                  'single clean/rust photo is the primary front evidence. The old '
                  'model and other items were not modified. Candidate is '
                  'manual_only pending human review.'),
    }
    (args.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                           encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    external = {
        'number': 16, 'status': progress['status'],
        'out': str(args.out), 'mesh': str(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'), 'checks': checks,
        'pending_review': report['pending_review'], 'next_number': 17,
        'manual_only': True,
        'review_page': str(args.out / 'review.html'),
        'notes': progress['notes'],
    }
    (progress_dir / '16-progress.json').write_text(
        json.dumps(external, ensure_ascii=False, indent=2), encoding='utf8')
    files = [path for path in sorted(args.out.rglob('*')) if path.is_file()]
    hashes = {
        'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_16.py',
        'inputs': {
            'clean': digest(clean_path),
            'rust': digest(rust_path),
            'reference': digest(args.reference),
            'old_mesh': digest(old_mesh_path),
            'source_view': digest(source_view_path),
            'source_alpha': digest(alpha_path),
            'config': digest(config_path),
            'worker': digest(worker_path),
            'builder': digest(Path(__file__)),
        },
        'files': {str(path.relative_to(args.out)).replace('\\', '/'): digest(path)
                  for path in files},
    }
    hashes['files']['hashes.json'] = 'self'
    (args.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                          encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(args.out), 'checks': checks,
                      'texture_stats': stats, 'ray_checks': ray,
                      'source_fidelity': fidelity,
                      'top_view_aspect': report['top_view_aspect'],
                      'cylinder_check': cylinder_check,
                      'page_assets': page_check['passed']}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
