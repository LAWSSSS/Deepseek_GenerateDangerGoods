"""Build the review-only red gas-cylinder candidate (item 20).

Evidence:
- the clean/rust HD derivatives and the original 1254 px photograph of one
  tall red painted steel gas cylinder with a vertically stacked white CCS mark,
  a domed valve guard with an oval hand-hole, a collar/neck, a rounded convex
  bottom and localised rust;
- the user six-view sheet, matched by appearance only and explicitly
  uncalibrated (visual_match_unverified), used for topology and the unseen
  back, never for measurements.  Its narrower guard and open-looking top/side
  panels conflict with the closed dome in the photograph; the photograph wins.

The old TripoSR library mesh is a soft blob: the guard is fused, the oval
hand-hole is filled (the audit records one filled hole) and the surface is a
rough photo-projected shell.  It is kept unchanged and is not used as the
candidate's base geometry.  The candidate is a parametric part rebuild with a
true body of revolution, a hollow guard with a real oval front hand-hole and a
solid far wall, a separate brass valve and dark steel neck seal, a rounded
convex bottom with a small contact pad, and red enamel PBR textures that keep the
CCS print and local rust.  All candidates remain manual-only and pending human
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
REFERENCE_NAME = 'ChatGPT 图像 2026年10月14日 10_01_29-1.png'
REFERENCE_SHA256 = '94eaea1b92972cde9a6462896a6ab56594f9563ea8ee0f124107c9abd6d04cd7'
ORIGINAL_JPG = ROOT / 'external/异物生成/input/objects/气瓶1.jpg'
ORIGINAL_JPG_SHA256 = '35a5eee2eb09eb9bb7aa55fdb33211bf31d663ec6dc07d61cfa9d334894ecd89'
CLEAN_SHA256 = '5876dacad62c22c00b0ec6bd13df542d09f23b83699d375cec60b4c3bcf2a04e'
RUST_SHA256 = '351e3fa52ad6b533534cac3c1b628597bd535d9440a94dbba076e570d442fa41'
OLD_MESH_SHA256 = '98b04872072567c409e1b04e10e1f1e0eeb23132e5cff4e2c08b45d55e4df922'
SOURCE_ALPHA_SHA256 = '6590f5cb13e06fa41a72be509393b911565189e93464c334dc43c7f08aec49f5'
SOURCE_VIEW_SHA256 = '2559254614a161db58de4631b45c0ef953cee25e8eb4e9370b4273efcc622568'
FONT = 'C:/Windows/Fonts/msyh.ttc'
FONTB = 'C:/Windows/Fonts/msyhbd.ttc'
LETTER_FONT = 'C:/Windows/Fonts/arialbd.ttf'


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def find_reference():
    for folder in (ROOT / 'references/parallel-six-view', ROOT / 'references'):
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
    """Relocate a shared-library absolute path into this parallel worktree."""
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


def library_row(number=20):
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


def periodic_field(rng, height, width, period, sigma_y, sigma_x):
    """A horizontally periodic noise field (period divides width)."""
    if width % period:
        raise ValueError('period must divide width')
    tile = rng.normal(size=(height, period)).astype(np.float32)
    reps = int(math.ceil((width + 6 * period) / period))
    big = np.tile(tile, (1, reps))
    big = gaussian_filter(big, sigma=(sigma_y, sigma_x))
    start = 2 * period
    return big[:, start:start + width]


def norm01(value):
    low, high = float(value.min()), float(value.max())
    if high - low < 1e-9:
        return np.zeros_like(value)
    return (value - low) / (high - low)


def make_letter_mask(width, height, config, rng):
    """Draw the vertical CCS stack in cylindrical UV space."""
    letter_mask = np.zeros((height, width), np.float32)
    try:
        letter_font = ImageFont.truetype(LETTER_FONT, 360)
    except Exception:
        letter_font = ImageFont.load_default()
    for char, zc in zip('CCS', config['dimensions']['letter_centers_z_m']):
        design = Image.new('L', (420, 420), 0)
        d = ImageDraw.Draw(design)
        bbox = d.textbbox((0, 0), char, font=letter_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((420 - tw) / 2 - bbox[0], (420 - th) / 2 - bbox[1]), char,
               font=letter_font, fill=255)
        px_w = int(round(config['dimensions']['letter_width_m'] /
                         (math.tau * config['dimensions']['body_radius_m']) * width))
        px_h = int(round(config['dimensions']['letter_height_m'] /
                         config['dimensions']['neck_z1_m'] * height))
        glyph = design.resize((max(1, px_w), max(1, px_h)), Image.Resampling.LANCZOS)
        uc = config['dimensions']['letter_u_center']
        vc = zc / config['dimensions']['neck_z1_m']
        x0 = int(round(uc * width - glyph.width / 2))
        y0 = int(round((1 - vc) * height - glyph.height / 2))
        region = np.asarray(glyph).astype(np.float32) / 255.0
        ys0, ys1 = max(0, y0), min(height, y0 + glyph.height)
        xs0, xs1 = max(0, x0), min(width, x0 + glyph.width)
        if ys1 > ys0 and xs1 > xs0:
            sub = region[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
            letter_mask[ys0:ys1, xs0:xs1] = np.maximum(
                letter_mask[ys0:ys1, xs0:xs1], sub)
    edge_noise = norm01(periodic_field(rng, height, width, 128, 0.9, 0.9))
    letter_mask *= np.clip(0.86 + 0.14 * edge_noise, 0, 1)
    letter_mask = np.clip((letter_mask - 0.10) / 0.85, 0, 1)
    return gaussian_filter(letter_mask, sigma=0.6)


def draw_scratches(rgb, rng, count, light=True):
    height, width = rgb.shape[:2]
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(count):
        x = int(rng.integers(10, width - 10))
        y = int(rng.integers(10, height - 10))
        length = int(rng.integers(8, 64))
        angle = float(rng.normal(0.0, 0.10))
        dx = int(length * math.sin(angle))
        dy = int(length * math.cos(angle))
        if light and rng.random() > 0.4:
            color = (170, 168, 160, int(rng.integers(6, 20)))
        else:
            color = (30, 20, 18, int(rng.integers(8, 24)))
        draw.line((x, y, x + dx, y + dy), fill=color, width=1)
    layer = np.asarray(overlay).astype(np.float32)
    alpha = layer[:, :, 3:4] / 255.0
    out = rgb * (1 - alpha) + layer[:, :, :3] * alpha
    return np.clip(out, 0, 255)


def rust_palette(clean_rgb, rust_rgb, alpha, rect):
    """Sample rust colours only where the rust derivative actually differs."""
    x0, y0, x1, y1 = rect
    c = clean_rgb[y0:y1, x0:x1]
    r = rust_rgb[y0:y1, x0:x1]
    m = alpha[y0:y1, x0:x1] > 200
    diff = np.linalg.norm(c - r, axis=2)
    support = np.logical_and(m, diff > 28)
    if support.sum() > 400:
        sample = np.median(r[support], axis=0)
    else:
        sample = np.array([140, 84, 54], np.float32)
    warm = 0.25 * sample + 0.75 * np.array([140, 82, 46], np.float32)
    dark = 0.15 * sample + 0.85 * np.array([82, 48, 30], np.float32)
    return warm.astype(np.float32), dark.astype(np.float32)


def make_body_texture(clean_path, rust_path, out, config):
    clean = Image.open(clean_path).convert('RGBA')
    rust = Image.open(rust_path).convert('RGBA')
    sampling = config['texture_sampling']
    if list(clean.size) != sampling['clean_size']:
        raise ValueError('20 clean source dimensions changed; sampling rects need review')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    clean_rgb = np.asarray(clean)[:, :, :3].astype(np.float32)
    rust_rgb = np.asarray(rust)[:, :, :3].astype(np.float32)
    alpha = np.asarray(clean.getchannel('A'))
    width = height = 2048
    rng = np.random.default_rng(20001)
    low = norm01(periodic_field(rng, height, width, 256, 26, 34))
    fine = norm01(periodic_field(rng, height, width, 128, 1.7, 1.1))
    brush = norm01(periodic_field(rng, height, width, 128, 5.2, 0.7))
    streak = norm01(periodic_field(rng, height, width, 64, 36.0, 0.35))
    paint = (0.55 * (low - 0.5) + 0.25 * (fine - 0.5) +
             0.20 * (brush - 0.5) + 0.08 * (streak - 0.5))
    base_rgb = np.asarray(config['material_profile']['body']['base_rgb'], np.float32)
    rgb = np.tile(base_rgb, (height, width, 1))
    rgb *= (1 + paint[:, :, None] * 0.12)

    # Map the measured clean/rust difference from the straight-on photograph
    # onto the cylinder UV.  For every texture row we invert the orthographic
    # cylinder projection and sample the photograph at x = cx + R sin(theta).
    # Only the front half is observed; the back is an explicit, weaker mirror
    # plus noise and is marked as inferred.
    scale_m_per_px = config['dimensions']['total_height_m'] / 1178.0
    top_y = 42
    body_top_z = config['dimensions']['neck_z1_m']
    ev = np.zeros((height, width), np.float32)
    col = np.zeros((height, width, 3), np.float32)
    front_start = int(round(0.25 * width))
    front_end = int(round(0.75 * width))
    front_cols = np.arange(front_start, front_end)
    front_u = (front_cols + 0.5) / width
    theta = (0.5 - front_u) * math.tau
    for vi in range(height):
        v = 1.0 - (vi + 0.5) / height
        z = v * body_top_z
        if z <= 0.004:
            continue
        y = top_y + (config['dimensions']['total_height_m'] - z) / scale_m_per_px
        yi = int(round(y))
        if yi < 0 or yi >= clean.height:
            continue
        row = np.nonzero(alpha[yi] >= 128)[0]
        if len(row) < 20:
            continue
        x0, x1 = int(row.min()), int(row.max())
        half = max(1.0, (x1 - x0) / 2.0)
        cx = (x0 + x1) / 2.0
        xi = np.clip(np.round(cx + half * np.sin(theta)).astype(np.int64),
                     0, clean.width - 1)
        c = clean_rgb[yi, xi]
        r = rust_rgb[yi, xi]
        diff = np.linalg.norm(c - r, axis=1)
        drop = c.mean(axis=1) - r.mean(axis=1)
        e = np.clip((diff - 18.0) / 55.0, 0, 1)
        e = np.maximum(e, np.clip((drop - 2.0) / 28.0, 0, 1) * 0.85)
        ev[vi, front_cols] = e
        col[vi, front_cols] = r
    ev = gaussian_filter(ev, sigma=(2.0, 2.0))
    col = gaussian_filter(col, sigma=(2.0, 2.0, 0.0))
    # Fill the unobserved back half from the front evidence (mirror), with a
    # soft transition at both front/back boundaries instead of a hard seam.
    u_index = np.arange(width, dtype=np.float32) / width
    back_u = (0.5 - u_index) % 1.0
    back_idx = np.clip((back_u * width).astype(np.int64), 0, width - 1)
    edge = np.clip(np.minimum(u_index - 0.25, 0.75 - u_index) / 0.045, 0, 1)
    ev_back = ev[:, back_idx] * 0.48
    col_back = col[:, back_idx]
    ev = edge[None, :] * ev + (1 - edge[None, :]) * ev_back
    col = edge[None, :, None] * col + (1 - edge[None, :, None]) * col_back
    ev = gaussian_filter(ev, sigma=(3.0, 3.0))
    # Explicitly inferred height bands: the rust derivative is heaviest at the
    # guard/neck/shoulder and at the bottom, and the back is less weathered.
    vv = (1.0 - np.arange(height, dtype=np.float32) / height)[:, None] * body_top_z
    band = np.zeros((height, width), np.float32)
    band += 0.20 * np.exp(-((vv - 0.705) / 0.038) ** 2)
    band += 0.17 * np.exp(-((vv - 0.035) / 0.050) ** 2)
    band += 0.018 * np.exp(-((vv - 0.470) / 0.180) ** 2)
    ev = np.clip(ev + band * (0.45 + 0.55 * low), 0, 1)
    rust_alpha = np.clip((ev - 0.16) / 0.52, 0, 1) ** 0.90
    rust_alpha *= (0.58 + 0.42 * low)
    rust_alpha *= np.clip(0.72 + 0.55 * fine, 0, 1)
    speckle = np.clip((fine - 0.72) / 0.28, 0, 1) * np.clip((low - 0.35) / 0.65, 0, 1)
    rust_alpha = np.clip(rust_alpha + speckle * 0.25, 0, 0.80)
    warm, dark = rust_palette(clean_rgb, rust_rgb, alpha,
                              sampling['rust_color_region_px'])
    mix = np.clip(0.42 + 0.58 * fine, 0, 1)[:, :, None]
    rust_color = dark[None, None, :] * (1 - mix) + warm[None, None, :] * mix
    # Keep the rust from becoming a dark film: pull the blend toward brown-orange.
    rust_color = np.clip(0.84 * rust_color + 0.16 * np.array([132, 78, 46], np.float32),
                         0, 255)

    # White CCS print.  The mask is built before the rust blend so the
    # high-contrast print edge in the clean/rust pair does not become a broad
    # rust halo; a few small wear specks are added afterwards.
    letter_rgb = np.asarray([236, 233, 226], np.float32)
    letter_mask = make_letter_mask(width, height, config, rng)
    # The source print has a rusted/paint-worn edge; when projected it becomes a
    # glyph-shaped rust ring that reads as a ghost outline.  Keep a clean print
    # panel and add only a few small wear specks below.
    px_w = config['dimensions']['letter_width_m'] / \
        (math.tau * config['dimensions']['body_radius_m']) * width
    px_h = config['dimensions']['letter_height_m'] / \
        config['dimensions']['neck_z1_m'] * height
    letter_zone = np.zeros((height, width), np.float32)
    uc = config['dimensions']['letter_u_center']
    for zc in config['dimensions']['letter_centers_z_m']:
        vc = zc / config['dimensions']['neck_z1_m']
        x0 = int(round(uc * width - 0.75 * px_w))
        x1 = int(round(uc * width + 0.75 * px_w))
        y0 = int(round((1 - vc) * height - 0.75 * px_h))
        y1 = int(round((1 - vc) * height + 0.75 * px_h))
        letter_zone[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 1.0
    letter_zone = gaussian_filter(letter_zone, sigma=2.0)
    rust_alpha = rust_alpha * (1.0 - 0.97 * letter_zone)
    rgb = rgb * (1 - rust_alpha[:, :, None]) + rust_color * rust_alpha[:, :, None]
    rgb = draw_scratches(rgb, rng, 70)
    rgb = rgb * (1 - letter_mask[:, :, None]) + letter_rgb[None, None, :] * letter_mask[:, :, None]
    rust_speck = np.random.default_rng(20002).random((height, width)) < 0.012
    speck = np.logical_and(letter_mask > 0.5, rust_speck)
    rgb[speck] = 0.55 * rgb[speck] + 0.45 * warm
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    scratch_height = np.clip((255 - np.asarray(image).mean(axis=2)) / 255.0, 0, 1)
    height_field = (0.42 * brush + 0.24 * fine + 0.12 * low +
                    0.16 * rust_alpha + 0.06 * scratch_height)
    normal = height_to_normal(height_field, 0.32)
    roughness = np.clip(0.335 + 0.045 * (low - 0.5) + 0.035 * (fine - 0.5) +
                        0.015 * (streak - 0.5) + 0.50 * rust_alpha, 0.28, 0.92)
    roughness = np.where(letter_mask > 0.5, 0.38, roughness)
    metallic = np.clip(0.045 + 0.02 * (fine - 0.5), 0.02, 0.08)
    metallic = metallic * (1 - rust_alpha) + 0.0 * rust_alpha
    metallic = np.where(letter_mask > 0.5, 0.0, metallic)
    mr = orm_image(roughness, metallic)
    luma = np.asarray(image).mean(axis=2)
    white = np.logical_and.reduce((np.asarray(image).min(axis=2) > 175,
                                   luma > 180))
    paths = {}
    for name, array in (('body_base', np.asarray(image)),
                        ('body_mr', mr), ('body_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'min_luma': round(float(luma.min()), 2),
        'dark_fraction_below_20': round(float((luma < 20).mean()), 4),
        'mean_chroma': round(float((np.asarray(image).max(axis=2) -
                                    np.asarray(image).min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'min_metallic': round(float(metallic.min()), 4),
        'max_metallic': round(float(metallic.max()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'min_roughness': round(float(roughness.min()), 4),
        'max_roughness': round(float(roughness.max()), 4),
        'rust_fraction': round(float((rust_alpha > 0.25).mean()), 4),
        'white_fraction': round(float(white.mean()), 4),
        'letter_pixels': int(round(float((letter_mask > 0.5).sum()))),
        'sampled_warm_rust_rgb': [round(float(v), 1) for v in warm],
        'sampled_dark_rust_rgb': [round(float(v), 1) for v in dark],
        'note': ('Red enamel paint is dielectric (metallic ~0.045); its gloss '
                 'comes from the roughness/specular response and is tested '
                 'separately, with no explicit clear-coat layer or baked '
                 'highlight. The white CCS print is dielectric. Rust is derived '
                 'from the clean/rust difference on the observed front half; '
                 'the back is a weaker mirror plus noise and is inferred. The '
                 'print is redrawn with a thin worn edge; the paint is not made '
                 'metallic.'),
    }
    return paths, stats


def make_guard_texture(out, config):
    width = height = 1024
    rng = np.random.default_rng(20003)
    low = norm01(periodic_field(rng, height, width, 128, 14, 18))
    fine = norm01(periodic_field(rng, height, width, 64, 1.2, 0.8))
    brush = norm01(periodic_field(rng, height, width, 64, 3.6, 0.55))
    patch = norm01(periodic_field(rng, height, width, 128, 7.0, 9.0))
    patch2 = norm01(periodic_field(rng, height, width, 256, 5.0, 6.0))
    drip = norm01(periodic_field(rng, height, width, 64, 24.0, 0.5))
    base_rgb = np.asarray(config['material_profile']['guard']['base_rgb'], np.float32)
    rgb = np.tile(base_rgb, (height, width, 1))
    rgb *= (1 + (0.52 * (low - 0.5) + 0.26 * (fine - 0.5) +
                 0.22 * (brush - 0.5))[:, :, None] * 0.14)
    v = (1.0 - (np.arange(height) + 0.5) / height)[:, None]
    band = np.exp(-((v - 0.035) / 0.090) ** 2) + 0.60 * np.exp(-((v - 0.965) / 0.060) ** 2)
    clump = np.clip((0.58 * patch + 0.42 * patch2 - 0.45) * 2.2, 0, 1)
    rust_alpha = clump * np.clip(band * 1.8 + 0.14, 0, 1)
    rust_alpha *= np.clip(0.55 + 0.45 * drip, 0, 1)
    rust_alpha = np.clip(gaussian_filter(rust_alpha, 2.2) * 0.80, 0, 0.70)
    warm = np.array([140, 84, 48], np.float32)
    dark = np.array([88, 52, 32], np.float32)
    rust_color = (0.45 * dark + 0.55 * warm)[None, None, :] * (0.72 + 0.28 * fine[:, :, None])
    rgb = rgb * (1 - rust_alpha[:, :, None]) + rust_color * rust_alpha[:, :, None]
    rgb = draw_scratches(rgb, rng, 42)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    scratch_height = np.clip((255 - np.asarray(image).mean(axis=2)) / 255.0, 0, 1)
    height_field = 0.45 * brush + 0.25 * fine + 0.15 * low + 0.15 * rust_alpha
    normal = height_to_normal(height_field, 0.28)
    roughness = np.clip(0.335 + 0.04 * (low - 0.5) + 0.52 * rust_alpha, 0.28, 0.88)
    metallic = np.clip(0.04 + 0.02 * (fine - 0.5), 0.02, 0.07) * (1 - rust_alpha)
    mr = orm_image(roughness, metallic)
    luma = np.asarray(image).mean(axis=2)
    paths = {}
    for name, array in (('guard_base', np.asarray(image)),
                        ('guard_mr', mr), ('guard_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'min_luma': round(float(luma.min()), 2),
        'dark_fraction_below_20': round(float((luma < 20).mean()), 4),
        'mean_chroma': round(float((np.asarray(image).max(axis=2) -
                                    np.asarray(image).min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'rust_fraction': round(float((rust_alpha > 0.25).mean()), 4),
        'note': ('Red enamel guard with rust concentrated near its base and top, '
                 'as in the photograph. The exact guard interior and the back '
                 'are not visible and are inferred; the paint is dielectric with '
                 'no explicit clear-coat layer.'),
    }
    return paths, stats


def make_steel_texture(out, config):
    width = height = 512
    rng = np.random.default_rng(20004)
    low = norm01(periodic_field(rng, height, width, 128, 8, 12))
    fine = norm01(periodic_field(rng, height, width, 64, 1.0, 0.8))
    patch = norm01(periodic_field(rng, height, width, 128, 3.0, 4.0))
    base_rgb = np.asarray(config['material_profile']['steel']['base_rgb'], np.float32)
    rgb = np.tile(base_rgb, (height, width, 1))
    rgb *= (1 + (0.55 * (low - 0.5) + 0.25 * (fine - 0.5))[:, :, None] * 0.18)
    rust_alpha = np.clip((patch - 0.62) * 2.6, 0, 1) * 0.55
    rust_color = np.array([108, 68, 44], np.float32)
    rgb = rgb * (1 - rust_alpha[:, :, None]) + rust_color * rust_alpha[:, :, None]
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    height_field = 0.55 * fine + 0.30 * low + 0.15 * rust_alpha
    normal = height_to_normal(height_field, 0.20)
    roughness = np.clip(0.36 + 0.05 * (low - 0.5) + 0.30 * rust_alpha, 0.30, 0.72)
    metallic = np.clip(1.0 - 0.25 * rust_alpha, 0.70, 1.0)
    mr = orm_image(roughness, metallic)
    luma = np.asarray(image).mean(axis=2)
    paths = {}
    for name, array in (('steel_base', np.asarray(image)),
                        ('steel_mr', mr), ('steel_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'min_metallic': round(float(metallic.min()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'rust_fraction': round(float((rust_alpha > 0.25).mean()), 4),
        'note': ('Dark steel for the neck seal and valve body; metallic 1 with '
                 'small rust patches that lower metalness locally. This is the '
                 'part used for the metal-reflection response test.'),
    }
    return paths, stats


def texture_checks(body, guard, steel):
    body_ok = (38 <= body['mean_luma'] <= 115 and
               body['mean_chroma'] >= 55 and
               0.02 <= body['mean_metallic'] <= 0.14 and
               0.26 <= body['mean_roughness'] <= 0.58 and
               body['dark_fraction_below_20'] <= 0.30 and
               0.0015 <= body['white_fraction'] <= 0.06 and
               0.01 <= body['rust_fraction'] <= 0.35)
    guard_ok = (38 <= guard['mean_luma'] <= 115 and
                guard['mean_chroma'] >= 55 and
                0.02 <= guard['mean_metallic'] <= 0.14 and
                0.26 <= guard['mean_roughness'] <= 0.60 and
                guard['dark_fraction_below_20'] <= 0.30 and
                guard['rust_fraction'] <= 0.40)
    steel_ok = (30 <= steel['mean_luma'] <= 160 and
                steel['mean_metallic'] >= 0.90 and
                0.25 <= steel['mean_roughness'] <= 0.75)
    return {
        'body': {'passed': bool(body_ok), **body},
        'guard': {'passed': bool(guard_ok), **guard},
        'steel': {'passed': bool(steel_ok), **steel},
        'note': ('The cylinder and guard are painted steel: their enamel is '
                 'dielectric with no explicit clear-coat layer, so their '
                 'metalness is low by design. The neck seal and valve parts are '
                 'metal and carry the metal-reflection check. The checks reject '
                 'black, over-smooth or unpainted-looking sheets; they do not '
                 'prove the unseen surface appearance.'),
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


def combined_mesh(path):
    return trimesh.util.concatenate(list(load_named_parts(path).values()))


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
            'extensions': sorted(material.get('extensions', {}).keys()),
        })
    textured_names = ('body', 'guard', 'steel')
    textured = [item for item in materials if item['name'] in textured_names]
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
                 'metallic-roughness and normal textures for body, guard and '
                 'steel. Brass and clay use scalar PBR factors. Clear-coat is '
                 'exported through the glTF material extension when supported. '
                 'This audit does not prove the visual appearance in every viewer.'),
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


def ray_checks(path, config, worker):
    mesh = combined_mesh(path)
    D = config['dimensions']
    shift = worker['z_shift_m']
    zc = D['window_center_z_m'] + shift
    yc = D['window_center_y_m']
    record = {}

    hole = ray_hits(mesh, (0.20, yc + 0.0048, zc), (-1, 0, 0), multiple=True)
    first_x = float(hole[0][0]) if len(hole) else None
    record['guard_hand_hole'] = {
        'hits': int(len(hole)),
        'first_hit_x_m': round(first_x, 6) if first_x is not None else None,
        'passed': bool(first_x is not None and first_x < -0.015),
        'note': ('A horizontal ray inside the oval slot but away from the valve '
                 'axis passes through the front wall and stops on the solid far '
                 'wall. This proves a real front hand-hole while matching the '
                 'dark interior visible in the photograph; the reference panels '
                 'are internally contradictory about a far-side opening.'),
    }
    wall = ray_hits(mesh, (0.20, yc + 0.0125, zc), (-1, 0, 0), multiple=True)
    wall_x = float(wall[0][0]) if len(wall) else None
    record['guard_wall'] = {
        'hits': int(len(wall)),
        'first_hit_x_m': round(wall_x, 6) if wall_x is not None else None,
        'passed': bool(wall_x is not None and wall_x > 0.020),
        'note': ('A parallel ray outside the slot hits the near guard wall, so '
                 'the open first hit inside the slot is not simply missing the '
                 'whole guard.'),
    }
    valve = ray_hits(mesh, (0.20, yc, 0.7980 + shift), (-1, 0, 0), multiple=True)
    record['valve_present'] = {
        'hits': int(len(valve)),
        'passed': bool(len(valve) >= 1),
        'note': ('A ray at the window height lower edge hits the brass/steel '
                 'valve; the valve silhouette is a conservative service-valve '
                 'form, not a measured mechanism.'),
    }

    bottom_center = ray_hits(mesh, (0.0, 0.0, -0.10), (0, 0, 1))
    bottom_ring = ray_hits(mesh, (0.050, 0.0, -0.10), (0, 0, 1))
    z_center = float(bottom_center[0][2]) if len(bottom_center) else None
    z_ring = float(bottom_ring[0][2]) if len(bottom_ring) else None
    record['convex_bottom'] = {
        'centre_hit_z_m': round(z_center, 6) if z_center is not None else None,
        'ring_hit_z_m': round(z_ring, 6) if z_ring is not None else None,
        'rise_0_to_50mm_m': round(float(z_ring - z_center), 6)
        if z_center is not None and z_ring is not None else None,
        'passed': bool(z_center is not None and z_ring is not None and
                       z_ring > z_center + 0.006),
        'note': ('Rays from below show the centre contact is lower than the '
                 '50 mm radius, i.e. a rounded convex bottom with a small '
                 'contact patch, as in the photo silhouette. The underside '
                 'detail is not visible in the photo and remains inferred.'),
    }

    neck = ray_hits(mesh, (0.20, 0.0, 0.7744 + shift), (-1, 0, 0))
    record['neck_seal'] = {
        'hits': int(len(neck)),
        'passed': bool(len(neck) >= 1),
        'note': ('A horizontal ray at the guard/neck joint hits the steel seal '
                 'ring, confirming the visible dark joint. Wall thickness and '
                 'fit are inferred.'),
    }
    return record


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
    draw.text((20, 14),
              '20 气瓶1 · 六方向：未标定视觉参考 / 原模型 / 新候选（同一中性光）',
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
    names = ('guard-window', 'guard-window-close', 'guard-top', 'neck-collar',
             'shoulder', 'ccs-logo', 'body-surface', 'bottom-foot',
             'bottom-underside', 'valve-window', 'top-detail')
    titles = {
        'guard-window': '阀护罩椭圆开孔',
        'guard-window-close': '开孔边缘与内部阀体近景',
        'guard-top': '护罩顶部闭合曲面',
        'neck-collar': '护罩/颈部/钢质密封环',
        'shoulder': '瓶肩过渡与锈蚀',
        'ccs-logo': 'CCS 白色印刷（重绘，略有磨耗）',
        'body-surface': '红色漆面纹理与局部锈迹',
        'bottom-foot': '底部接触区',
        'bottom-underside': '底部仰视（推断）',
        'valve-window': '窗口内阀杆（黄铜/钢）',
        'top-detail': '顶视细节',
    }
    tile = (620, 470)
    cols = 4
    rows = math.ceil(len(names) / cols)
    board = Image.new('RGB', (cols * tile[0] + 30, rows * (tile[1] + 44) + 70),
                      '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '20 气瓶1 · 关键局部检查（新版，中性光）',
              font=font(26, True), fill='#17232a')
    for i, name in enumerate(names):
        x = 10 + (i % cols) * tile[0]
        y = 56 + (i // cols) * (tile[1] + 44)
        image = Image.open(out / 'renders' / ('detail-' + name + '.png'))
        board.paste(fit_tile(image, tile, '#e3e8e8'), (x, y))
        draw.text((x + 4, y + tile[1] + 8), f'{name} · {titles[name]}',
                  font=font(17, True), fill='#34444b')
    board.save(out / '局部检查.jpg', quality=93)


def make_isolated_board(out, ray):
    items = (('guard-only', '护罩隔离（前视角，可见双侧开孔）'),
             ('guard-top-only', '护罩隔离顶视（封闭顶部）'),
             ('valve-only', '阀体/六角/阀杆隔离'),
             ('body-only', '瓶体隔离（含瓶肩与底）'),
             ('body-top-only', '瓶体顶视轮廓（圆度检查）'),
             ('body-bottom-only', '瓶体底视轮廓（圆度检查）'))
    tile = (620, 500)
    board = Image.new('RGB', (3 * tile[0] + 30, 2 * (tile[1] + 50) + 260), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '20 气瓶1 · 隔离结构与开口检查（渲染 + 射线）',
              font=font(26, True), fill='#17232a')
    for i, (name, title) in enumerate(items):
        x = 10 + (i % 3) * tile[0]
        y = 56 + (i // 3) * (tile[1] + 50)
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile,
                             '#e3e8e8'), (x, y))
        draw.text((x + 4, y + tile[1] + 8), title, font=font(18, True),
                  fill='#34444b')
    y = 56 + 2 * (tile[1] + 50) + 6
    draw.text((18, y), '隔离开口/结构射线检查', font=font(22, True), fill='#17232a')
    lines = [
        f"护罩椭圆手孔：孔内射线首次命中 x={ray['guard_hand_hole']['first_hit_x_m']} m"
        f"（实体远壁）；孔外平行射线首次命中 x={ray['guard_wall']['first_hit_x_m']} m"
        f"（近壁），证明前壁真实开孔",
        f"阀体：射线命中 {ray['valve_present']['hits']} 次，确认手孔内存在阀体",
        f"底部：中心 z={ray['convex_bottom']['centre_hit_z_m']} m，"
        f"50 mm 半径 z={ray['convex_bottom']['ring_hit_z_m']} m，"
        f"外凸高度差={ray['convex_bottom']['rise_0_to_50mm_m']} m",
        "说明：射线只证明这些位置的局部几何关系，不能证明壁厚、内部装配或隐藏面。",
    ]
    for line in lines:
        y += 30
        draw.text((24, y), line, font=font(17), fill='#46585f')
    board.save(out / '结构开口-隔离检查.jpg', quality=93)


def make_material_boards(out):
    tile = (960, 720)
    board = Image.new('RGB', (2 * tile[0] + 30, tile[1] + 100), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '20 气瓶1 · 同光照材质对照（左：原模型，右：新候选）',
              font=font(26, True), fill='#17232a')
    for i, name in enumerate(('old-material-close', 'new-material-close')):
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile,
                             '#ffffff'), (10 + i * tile[0], 60))
        draw.text((16 + i * tile[0], tile[1] + 70),
                  '原模型（TripoSR，保留未改）' if i == 0 else '新候选（PBR 红漆/CCS/锈）',
                  font=font(19, True), fill='#34444b')
    board.save(out / '材质-同光照对照.jpg', quality=94)

    board = Image.new('RGB', (2 * tile[0] + 30, 2 * tile[1] + 150), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '20 气瓶1 · 变光检查（高光必须随现场光移动）',
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
    board = Image.new('RGB', (2 * tile2[0] + 30, 3 * tile2[1] + 190), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12),
              '20 气瓶1 · 反射变光对照（上两行：低环境光+移动主光，金属件；下两行：漆面高光）',
              font=font(23, True), fill='#17232a')
    pairs = (('metal-key', '钢质密封环/六角/黄铜阀杆 · 主光 A'),
             ('metal-alt', '钢质密封环/六角/黄铜阀杆 · 主光 B'),
             ('paint-key', '红色瓷漆表面 · 主光 A'),
             ('paint-alt', '红色瓷漆表面 · 主光 B'),
             ('new-material-close', '红漆/CCS · 主光 A'),
             ('new-surface-alternate', '红漆/CCS · 主光 B'))
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
    draw.text((18, 10), '20 气瓶1 · 新候选 24 方向转台（中性光，仅供检查）',
              font=font(24, True), fill='#17232a')
    for i in range(24):
        x = 8 + (i % 6) * tile[0]
        y = 52 + (i // 6) * (tile[1] + 30)
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png')
        board.paste(fit_tile(image, tile, '#ffffff'), (x, y))
        draw.text((x + 4, y + tile[1] + 4), f'{i * 15:03d}°',
                  font=font(14, True), fill='#52646b')
    board.save(out / '转台-24方向.jpg', quality=93)


def make_source_board(out, original_jpg, rust_path, old_source_view):
    tile = (620, 720)
    board = Image.new('RGB', (4 * tile[0] + 40, tile[1] + 120), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '20 气瓶1 · 原照 / 做锈素材 / 旧模型原视角 / 新候选原视角诊断',
              font=font(24, True), fill='#17232a')
    images = (Image.open(original_jpg), Image.open(rust_path),
              Image.open(old_source_view) if Path(old_source_view).is_file() else None,
              Image.open(out / 'renders/new-source.png'))
    titles = ('原照（1254 px，CCS 与锈迹）', '做锈 HD（局部锈迹来源）',
              '旧模型原视角渲染', '新候选原视角诊断（未标定，不作验收）')
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

    ray = report['ray_checks']
    rows = ''.join([
        check_row('网格封闭/无自交', checks['mesh_integrity'],
                  '逐部件封闭；同部件 BVH 无自交；装配件允许有意搭接'),
        check_row('椭圆手孔/护罩壁/阀体', checks['isolated_openings'],
                  f"孔内射线首次命中 x={ray['guard_hand_hole']['first_hit_x_m']} m（远壁）；"
                  f"孔外射线首次命中 x={ray['guard_wall']['first_hit_x_m']} m（近壁）；"
                  f"阀体命中 {ray['valve_present']['hits']}；"
                  f"颈部钢环命中 {ray['neck_seal']['hits']}"),
        check_row('结构部件计数', checks['structural_counts'],
                  f"共 {report['worker']['parts_count']} 个部件；瓶体、护罩、阀体、六角、阀杆、钢环齐全"),
        check_row('瓶体圆柱截面', checks['cylindrical_body'],
                  f"侧壁最大半径偏差 {report['cylinder_check']['max_abs_radius_deviation_m']:.9f} m；"
                  f"瓶体顶视宽高比 {report['cylinder_check']['top_view_aspect_new']}"),
        check_row('底部外凸/接触区', checks['convex_bottom'],
                  f"中心 z={ray['convex_bottom']['centre_hit_z_m']} m；"
                  f"50 mm 半径 z={ray['convex_bottom']['ring_hit_z_m']} m；"
                  f"高度差 {ray['convex_bottom']['rise_0_to_50mm_m']} m"),
        check_row('材质贴图', checks['material_textures'],
                  '红漆瓶体/红漆护罩/深色钢 三组 PBR；白 CCS 为局部电介质印刷'),
        check_row('GLB 材质内嵌', checks['glb_materials'],
                  f"{report['glb_audit']['image_count']} 张 PNG；三组各有 baseColor、metallicRoughness、normal"),
        check_row('金属反射（钢/黄铜）', checks['metal_reflection'],
                  f"金属件亮度差 {report['metal_reflection_check']['metal_luma_difference']}；"
                  f"钢平均金属度 {report['metal_reflection_check']['steel_mean_metallic']}；"
                  f"黄铜金属度 {report['metal_reflection_check']['brass_metallic']}"),
        check_row('漆面高光变光（非金属）', checks['paint_gloss'],
                  f"漆面主光 A/B 亮度差 {report['paint_gloss_check']['luma_difference']}；"
                  f"涂层按电介质建模，不计为金属反射"),
        check_row('六视图参考诊断', 'diagnostic_only',
                  '六视图未标定；其护罩比例与顶/侧视图和原照冲突，IoU 只用于发现差异'),
        check_row('原视角轮廓诊断', 'diagnostic_only',
                  f"IoU {report['source_fidelity_diagnostic']['iou']}；"
                  f"孔洞 {report['source_fidelity_diagnostic']['holes']}，"
                  f"被填 {report['source_fidelity_diagnostic']['filled_holes']}；"
                  f"细结构召回 {report['source_fidelity_diagnostic']['narrow_recall']}。"
                  f"源 alpha 的孔洞来自暗部误判；真实前侧手孔由射线检查确认，"
                  f"远壁实体导致该 alpha 诊断仍记为被填"),
        check_row('检查页静态资源', report['page_assets_check'].get('passed'),
                  '本地 HTML/图片路径检查；未以无头浏览器声称交互通过'),
        check_row('人工验收', 'pending', '候选 manual_only，不进入自动合成'),
    ])
    summary = (f"<p>本件以直立原照为主，六视图参考未标定且其护罩比例/顶侧视图与原照不完全一致。"
               f"旧模型是软团块，护罩与颈肩融合、椭圆手孔被填，故未复用旧网格。"
               f"候选保留瓶体回转外形、略带外凸的底部与中心接触区、瓶肩、颈圈、"
               f"闭合圆顶护罩及真实前侧椭圆手孔（远壁实体，符合原照暗色内腔）、"
               f"孔内保守阀体与钢质颈环，"
               f"并用 PBR 红漆、局部锈蚀和重绘的 CCS 印刷还原外观。"
               f"红色搪瓷按电介质漆面建模，金属反射检查适用于钢环/六角/黄铜阀杆；"
               f"漆面另做高光变光检查。</p>")
    html = REVIEW_TEMPLATE
    replacements = {
        '__SUMMARY__': summary,
        '__CHECK_ROWS__': rows,
        '__TITLE__': '20 气瓶1 · 三维模型独立检查',
        '__SUBTITLE__': '红色 CCS 气瓶 · 分部件重建候选 · 未标定六视图参考',
        '__STATUS__': 'manual_only · 待人工审阅',
        '__VERSION__': report['version'],
        '__SOURCE_IOU__': str(report['source_fidelity_diagnostic'].get('iou')),
        '__TOP_ASPECT__': str(report['cylinder_check']['top_view_aspect_new']),
        '__BODY_LUMA__': str(texture_stats['body']['mean_luma']),
        '__GUARD_LUMA__': str(texture_stats['guard']['mean_luma']),
        '__STEEL_METAL__': str(texture_stats['steel']['mean_metallic']),
        '__BODY_METAL__': str(texture_stats['body']['mean_metallic']),
        '__RUST_FRACTION__': str(texture_stats['body']['rust_fraction']),
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
                '源图-原视角诊断.jpg',
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
header code{color:#18242b}
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
<div class="notice"><b>参考状态：</b>六视图映射为 visual_match_unverified（未标定）。__REFERENCE_NOTE__ 背面、护罩内部/顶部、椭圆手孔深度与远壁实体判断、阀体内部、壁厚、底部仰视和隐藏面为推断。候选为 <b>manual_only</b>，不替换原模型，不进入自动放置或正式合成。</div>
<h2>检查记录</h2>
<table><thead><tr><th>项目</th><th>结果</th><th>说明</th></tr></thead><tbody>__CHECK_ROWS__</tbody></table>
<div class="notice"><b>关键数字：</b>原视角诊断 IoU __SOURCE_IOU__（直立原照与候选，未标定，只作诊断）；瓶体顶视宽高比 __TOP_ASPECT__；瓶体/护罩平均亮度 __BODY_LUMA__/__GUARD_LUMA__；瓶体与护罩金属度 __BODY_METAL__（漆面为非金属）；钢平均金属度 __STEEL_METAL__；瓶体锈蚀面积比例 __RUST_FRACTION__。</div>
<h2>对照与文件</h2>
<ul class="links">
<li><a href="六视图-新旧对照.jpg">六视图新旧对照</a></li>
<li><a href="局部检查.jpg">关键局部检查</a> · <a href="结构开口-隔离检查.jpg">隔离结构与开口</a></li>
<li><a href="材质-同光照对照.jpg">同光照材质对照</a> · <a href="材质-变光检查.jpg">变光检查</a> · <a href="表面反射-变光对照.jpg">表面反射变光</a></li>
<li><a href="转台-24方向.jpg">24 方向转台拼图</a> · <a href="源图-原视角诊断.jpg">原照/做锈素材/原视角诊断</a></li>
<li><a href="mesh.glb">下载 GLB</a> · <a href="model.blend">下载 Blender 可编辑文件</a></li>
<li><a href="result.json">模型验证 JSON</a> · <a href="hashes.json">输入/输出哈希</a> · <a href="progress.json">进度记录</a></li>
</ul>
<div class="grid">
<a href="六视图-新旧对照.jpg"><img src="六视图-新旧对照.jpg"><span>六视图新旧对照</span></a>
<a href="局部检查.jpg"><img src="局部检查.jpg"><span>关键局部</span></a>
<a href="材质-同光照对照.jpg"><img src="材质-同光照对照.jpg"><span>材质同光照对照</span></a>
<a href="表面反射-变光对照.jpg"><img src="表面反射-变光对照.jpg"><span>表面反射变光</span></a>
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
    parser.add_argument('--number', type=int, default=20)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--textures-only', action='store_true')
    args = parser.parse_args()
    if args.number != 20:
        parser.error('this entry supports item 20 only')
    args.reference = (args.reference or find_reference()).resolve()
    if args.out is None:
        args.out = ROOT / ('outputs/20气瓶1-六视图重建-draft' if args.draft
                           else 'outputs/20气瓶1-六视图重建-v1')
    args.out, args.library, args.blender = (x.resolve() for x in
                                            (args.out, args.library, args.blender))
    if digest(args.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = library_row(20)
    old_mesh_path = current_worktree_path(row['mesh'])
    if digest(old_mesh_path) != OLD_MESH_SHA256:
        raise ValueError('old library mesh hash differs from the registered value')
    clean_path = current_worktree_path(row.get('clean', row['original']))
    if digest(clean_path) != CLEAN_SHA256:
        raise ValueError('clean source hash differs from the registered value')
    rust_path = current_worktree_path(row['rust'])
    if digest(rust_path) != RUST_SHA256:
        raise ValueError('rust source hash differs from the registered value')
    if not ORIGINAL_JPG.is_file() or digest(ORIGINAL_JPG) != ORIGINAL_JPG_SHA256:
        raise ValueError('original 1254 px photograph hash differs from the HD catalogue')
    source_view_path = current_worktree_path(row['source_view'])
    if digest(source_view_path) != SOURCE_VIEW_SHA256:
        raise ValueError('source view hash differs from the model library')
    alpha_path = source_view_path.parent / 'source-alpha.png'
    if not alpha_path.is_file() or ROOT not in alpha_path.resolve().parents:
        raise FileNotFoundError('source alpha missing from this worktree')
    if digest(alpha_path) != SOURCE_ALPHA_SHA256:
        raise ValueError('source alpha hash differs from the model library')
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    source_alpha_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    config_path = HERE / 'cylinder20_config.json'
    worker_path = HERE / 'mesh_rebuild_20_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    if args.textures_only:
        args.out.mkdir(parents=True, exist_ok=True)
        body_textures, body_stats = make_body_texture(clean_path, rust_path, args.out, config)
        guard_textures, guard_stats = make_guard_texture(args.out, config)
        steel_textures, steel_stats = make_steel_texture(args.out, config)
        stats = texture_checks(body_stats, guard_stats, steel_stats)
        (args.out / 'texture-stats.json').write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'status': 'textures', 'out': str(args.out),
                          'texture_stats': stats}, ensure_ascii=False), flush=True)
        return
    blender_version = subprocess.run([str(args.blender), '--version'],
                                     capture_output=True, text=True,
                                     check=True).stdout.splitlines()[0]
    common_inputs = {
        'reference': digest(args.reference), 'mesh': OLD_MESH_SHA256,
        'clean': CLEAN_SHA256, 'rust': RUST_SHA256,
        'original_jpg': ORIGINAL_JPG_SHA256,
        'source_alpha': digest(alpha_path),
        'source_view': digest(source_view_path),
        'input_paths': {
            'mesh': relative_input(old_mesh_path),
            'clean': relative_input(clean_path),
            'rust': relative_input(rust_path),
            'original_jpg': relative_input(ORIGINAL_JPG),
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
        'draft': bool(args.draft),
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
        guard_textures, guard_stats = make_guard_texture(args.out, config)
        steel_textures, steel_stats = make_steel_texture(args.out, config)
        textures = {**body_textures, **guard_textures, **steel_textures}
        stats = texture_checks(body_stats, guard_stats, steel_stats)
        (args.out / 'texture-stats.json').write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf8')
        request = {
            'out': str(args.out), 'original_mesh': str(old_mesh_path),
            'textures': textures,
            'config': config, 'source_camera': row['reconstruction']['source_camera'],
            'source_alpha_bbox': source_alpha_bbox, 'draft': bool(args.draft),
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
    ray = ray_checks(args.out / 'mesh.glb', config, worker)
    parts = load_named_parts(args.out / 'mesh.glb')
    body_mesh = parts.get('Cylinder body')
    if body_mesh is None:
        raise ValueError('exported GLB is missing the Cylinder body node')
    body_vertices = np.asarray(body_mesh.vertices)
    all_radii = np.hypot(body_vertices[:, 0], body_vertices[:, 1])
    side = body_vertices[np.logical_and.reduce((body_vertices[:, 2] > 0.0850,
                                                body_vertices[:, 2] < 0.6805,
                                                all_radii > 0.02))]
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
        'note': ('Only the straight side rings (0.0850 < z < 0.6805 m, r > 20 mm) '
                 'are measured; the rounded bottom, shoulder and top cap are '
                 'excluded. The check confirms cylindrical symmetry of the '
                 'modelled side, not measured dimensions.'),
    }

    if args.draft:
        draft_report = {
            'number': 20, 'draft': True, 'out': str(args.out),
            'mesh': str(args.out / 'mesh.glb'),
            'mesh_sha256': digest(args.out / 'mesh.glb'),
            'blend': str(args.out / 'model.blend'),
            'texture_stats': stats, 'mesh_audit': mesh_audit,
            'glb_audit': glb, 'ray_checks': ray,
            'cylinder_check': cylinder_check, 'worker': worker,
        }
        (args.out / 'draft-result.json').write_text(
            json.dumps(draft_report, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'status': 'draft', 'out': str(args.out),
                          'mesh_audit': mesh_audit, 'ray_checks': ray,
                          'texture_stats': stats,
                          'worker_parts': worker['parts_count']},
                         ensure_ascii=False), flush=True)
        return

    fidelity = None
    with Image.open(alpha_path) as alpha, \
            Image.open(args.out / 'renders/new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                   np.asarray(rendered.getchannel('A')))
    new_top_ratio = alpha_bbox_ratio(args.out / 'renders/body-top-only.png')
    old_top_ratio = alpha_bbox_ratio(args.out / 'renders/old-top.png')
    cylinder_check['top_view_aspect_new'] = new_top_ratio
    new_front_luma = median_luma(args.out / 'renders/new-front.png')
    old_front_luma = median_luma(args.out / 'renders/old-front.png')
    metal_difference = mean_difference(args.out / 'renders/metal-key.png',
                                       args.out / 'renders/metal-alt.png')
    paint_difference = mean_difference(args.out / 'renders/paint-key.png',
                                       args.out / 'renders/paint-alt.png')
    panels = reference_panels(args.reference)
    for view, panel in panels.items():
        panel.save(args.out / 'renders' / f'ref-{view}.png')
    reference_results = make_six_board(args.out, panels)
    make_detail_board(args.out)
    make_isolated_board(args.out, ray)
    make_material_boards(args.out)
    make_turntable(args.out)
    make_source_board(args.out, ORIGINAL_JPG, rust_path, source_view_path)

    names = [part['name'] for part in worker['parts']]
    structural = {
        'bodies': sum(name == 'Cylinder body' for name in names),
        'guards': sum(name == 'Valve guard' for name in names),
        'seals': sum(name == 'Neck seal ring' for name in names),
        'valve_bodies': sum(name == 'Valve body' for name in names),
        'valve_hexes': sum(name == 'Valve hex nut' for name in names),
        'valve_stems': sum(name == 'Valve stem' for name in names),
        'note': ('Part presence proves only that a feature was modelled; wall '
                 'thickness, guard interior, valve bore, bottom plate and all '
                 'hidden faces are inferred.'),
    }
    structural_ok = bool(structural['bodies'] == 1 and structural['guards'] == 1 and
                         structural['seals'] == 1 and structural['valve_bodies'] == 1 and
                         structural['valve_hexes'] == 1 and structural['valve_stems'] == 1)
    cylindrical_ok = bool(cylinder_check['max_abs_radius_deviation_m'] < 1e-6 and
                          cylinder_check['max_ring_radius_variation_m'] < 1e-6 and
                          new_top_ratio is not None and abs(new_top_ratio - 1.0) < 0.03)
    metal_reflection = {
        'status': 'passed' if (stats['steel']['mean_metallic'] >= 0.90 and
                               profile_brass_ok(config) and
                               metal_difference is not None and metal_difference > 2.0)
                  else 'failed_or_inconclusive',
        'steel_mean_metallic': stats['steel']['mean_metallic'],
        'brass_metallic': config['material_profile']['brass']['metallic'],
        'metal_luma_difference': metal_difference,
        'note': ('The neck seal and valve parts are metallic 1 and show a moved '
                 'specular response in a dedicated low-ambient render with a '
                 'strong moved key. The red enamel cylinder/guard are dielectric '
                 'paint and are NOT counted as metal; their gloss is checked '
                 'separately. This is a response test, not a claim about unseen '
                 'surface finish.'),
    }
    paint_gloss = {
        'status': 'passed' if (paint_difference is not None and paint_difference > 1.0)
                  else 'failed_or_inconclusive',
        'luma_difference': paint_difference,
        'note': ('The dielectric red enamel must still show a moving specular '
                 'highlight under a moved key. This is a gloss response test, '
                 'not a metalness claim.'),
    }
    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] == 6 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components'] and
                               worker['guard_boundary_edges_after_boolean'] == 0),
        'isolated_openings': bool(all(ray[key]['passed'] for key in
                                      ('guard_hand_hole', 'guard_wall',
                                       'valve_present', 'neck_seal'))),
        'structural_counts': structural_ok,
        'cylindrical_body': cylindrical_ok,
        'convex_bottom': bool(ray['convex_bottom']['passed']),
        'material_textures': bool(all(stats[key]['passed'] for key in
                                      ('body', 'guard', 'steel'))),
        'glb_materials': glb['passed'],
        'metal_reflection': metal_reflection['status'] == 'passed',
        'paint_gloss': paint_gloss['status'] == 'passed',
        'reference_diagnostic_only': True,
        'source_fidelity_diagnostic': fidelity,
        'source_pose_diagnostic_only': True,
        'browser_page': 'not_run',
    }
    render_luma = {
        'new_front_median': new_front_luma,
        'old_front_median': old_front_luma,
        'not_black': bool(new_front_luma is not None and new_front_luma >= 25),
        'not_blown': bool(new_front_luma is not None and new_front_luma <= 190),
        'note': ('Neutral-light luma only; it is used to reject a black or '
                 'blown-out render, not as a lighting acceptance.'),
    }

    audit_fidelity = None
    audit_path = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
    if audit_path.is_file():
        for line in audit_path.read_text(encoding='utf8').splitlines():
            if line.strip() and json.loads(line).get('number') == 20:
                audit_fidelity = json.loads(line).get('source_fidelity')
                break

    version = config['version']
    report = {
        'version': version, 'number': 20, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': config['shape_basis'], 'reference_role': config['reference_role'],
        'method': 'parametric_part_rebuild_from_straight_source_photo_and_uncalibrated_six_view_reference',
        'reference': str(args.reference), 'reference_sha256': digest(args.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['reference_note'],
        'original_mesh': str(old_mesh_path), 'original_mesh_sha256': OLD_MESH_SHA256,
        'original_photo_1254': str(ORIGINAL_JPG),
        'original_photo_1254_sha256': ORIGINAL_JPG_SHA256,
        'source_photo': str(clean_path), 'source_view': str(source_view_path),
        'source_alpha': str(alpha_path), 'original_clean_sha256': CLEAN_SHA256,
        'original_rust_sha256': RUST_SHA256,
        'provenance_note': (
            'The shared model-library record contains absolute paths from the '
            'original main worktree. Every input was relocated to the matching '
            'file inside this parallel worktree before use, and its registered '
            'SHA-256 was verified. No output is written outside this worktree.'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9495 only measures one dark '
            'silhouette. The six-view audit renders show a fused soft guard and '
            'the source-fidelity record reports 1 filled hole; the old mesh is '
            'therefore not reused as the candidate base geometry.'),
        'original_source_fidelity_record': audit_fidelity,
        'catalog_evidence': config['proportion_source'],
        'identity_note': config['proportion_source']['identity_note'],
        'dimensions': config['dimensions'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(args.out / 'mesh.glb'), 'mesh_sha256': digest(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'), 'blend_sha256': digest(args.out / 'model.blend'),
        'textures': textures, 'texture_stats': stats,
        'material_profile': config['material_profile'],
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The source camera is the calibrated camera from the old model '
            'library. The photograph is essentially straight-on and the '
            'candidate is upright, so this IoU is a meaningful silhouette '
            'diagnostic; it is still not a calibrated turntable and not an '
            'acceptance gate.'),
        'source_alpha_hole_note': (
            'The 512 px source alpha contains a 5 x 14 px hole at the dark oval '
            'slot, which is a dark-region mask artifact. The candidate models a '
            'real front hand-hole with a solid far wall, following the dark '
            'interior in the photograph; the alpha-hole diagnostic therefore '
            'still records the hole as filled. The real opening is verified by '
            'the ray test, not by this silhouette diagnostic.'),
        'reference_views': reference_results,
        'reference_diagnostic_note': (
            'The six-view sheet is uncalibrated and its guard proportions and '
            'top/side panels visibly differ from the photograph; all IoU values '
            'are diagnostic only.'),
        'top_view_aspect': {'new': new_top_ratio, 'old': old_top_ratio,
                            'note': ('Width/height of the isolated body-only top '
                                     'silhouette (new) and the old model top render. '
                                     'Only the new isolated cylinder is expected to '
                                     'be circular.')},
        'render_luma': render_luma,
        'mesh_audit': mesh_audit, 'glb_audit': glb, 'worker': worker,
        'cylinder_check': cylinder_check,
        'ray_checks': ray, 'structural_checks': structural,
        'metal_reflection_check': metal_reflection,
        'paint_gloss_check': paint_gloss,
        'dielectric_note': {
            'status': 'partially_applicable',
            'note': ('The cylinder and guard are painted steel with a dielectric '
                     'enamel (no explicit clear-coat layer). They are '
                     'deliberately not modelled as metal; the metal-reflection '
                     'check applies to the exposed steel seal ring and the brass '
                     'valve stem. The paint gloss is checked separately. The '
                     'white CCS print is dielectric.'),
        },
        'checks': checks,
        'page_assets_check': check_review_page(args.out),
        'limitations': [
            'The six-view sheet is an uncalibrated generated visual reference, not a measured turntable. Its guard is narrower and its top/left/right panels show an open-looking guard that conflicts with the closed dome in the photograph.',
            'Only one straight-on photograph exists. The unseen back, guard interior/top, far-side window, valve internals, wall thickness and underside are inferred.',
            'Working dimensions (850 mm overall, 140 mm body diameter, 62.8 mm guard width) follow the photo pixel ratio and a similar-catalogue plausibility check; they are not caliper measurements and the exact model/SKU is unconfirmed.',
            'The oval hand-hole is cut through the front guard wall only, because the photograph shows a dark interior behind the slot. The uncalibrated reference repeats a slot in front, back, left and right panels, which is internally inconsistent and is treated as a conflict rather than evidence for a through-hole.',
            'The CCS print is redrawn from the photograph; exact typeface, ink gloss, registration and whether another print exists on the opposite side are inferred.',
            'The red enamel is dielectric paint (no explicit clear-coat layer), not bare metal. Metal reflection is therefore checked on the steel neck seal and brass valve stem; the paint is checked as gloss.',
            'The bottom is modelled as a rounded convex end with a small contact patch following the photo silhouette; the reference bottom panel suggests a circular recessed plate, so the underside remains uncertain.',
            'Assemblies intentionally overlap at the guard/neck and valve/neck joints; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'The source silhouette IoU is diagnostic only and cannot prove the hidden back or the exact guard interior.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the guard window position/size, the closed dome, the neck collar and the CCS print against the photo and the physical sample.',
            'Confirm the actual overall height, body diameter, guard width and wall thickness with calipers or a measured drawing.',
            'Confirm whether the far guard wall is solid, the hand-hole depth, the guard interior/top, valve identity/mechanism and neck seal on the real part.',
            'Confirm the CCS typeface, size, colour and whether a print exists on the opposite side.',
            'Confirm the bottom profile/underside; the photo supports a convex bottom while the uncalibrated reference bottom panel suggests a recessed plate.',
            'Verify coordinate/scale, stability, floor contact and lighting integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
        ],
        'next_number': 21,
    }
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf8')

    make_review_page(args.out, report, stats)
    (args.out / 'progress.json').write_text('{}', encoding='utf8')
    (args.out / 'hashes.json').write_text('{}', encoding='utf8')
    page_check = check_review_page(args.out)
    report['page_assets_check'] = page_check
    report['checks']['review_page_assets'] = page_check['passed']
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf8')

    progress = {
        'number': 20, 'name': row['file'], 'asset_id': row['asset_id'],
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
        'pending_review': report['pending_review'], 'next_number': 21,
        'notes': ('The six-view sheet is an uncalibrated visual reference; the '
                  'straight-on photograph is the primary evidence. The old model '
                  'and other items were not modified. Candidate is manual_only '
                  'pending human review.'),
    }
    (args.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                           encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    external = {
        'number': 20, 'status': progress['status'],
        'out': str(args.out), 'mesh': str(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'), 'checks': checks,
        'pending_review': report['pending_review'], 'next_number': 21,
        'manual_only': True,
        'review_page': str(args.out / 'review.html'),
        'notes': progress['notes'],
    }
    (progress_dir / '20-progress.json').write_text(
        json.dumps(external, ensure_ascii=False, indent=2), encoding='utf8')
    files = [path for path in sorted(args.out.rglob('*')) if path.is_file()]
    hashes = {
        'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_20.py',
        'inputs': {
            'clean': digest(clean_path),
            'rust': digest(rust_path),
            'original_jpg': digest(ORIGINAL_JPG),
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
                      'metal_reflection': metal_reflection,
                      'paint_gloss': paint_gloss,
                      'page_assets': page_check['passed']}, ensure_ascii=False),
          flush=True)


def profile_brass_ok(config):
    return config['material_profile']['brass']['metallic'] >= 0.95


if __name__ == '__main__':
    main()
