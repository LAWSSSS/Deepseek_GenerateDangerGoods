"""Build the review-only black AERO air-shock candidate (item 17).

Evidence:
- the clean/rust HD derivatives and the original 447 px photograph of one used
  black air shock with a red radial adjuster and a white triangle + AERO mark;
- the user six-view sheet, matched by appearance only and explicitly
  uncalibrated (visual_match_unverified), used for topology and the unseen
  back, never for measurements.

The old TripoSR library mesh is a soft blob: the top and bottom eyes are fused,
the adjuster is merged into the cap and the can has no true holes.  It is kept
unchanged and is not used as the candidate's base geometry.  The candidate is a
parametric part rebuild with a true cylindrical can, a real silver separator
band, a real bottom machined lip, a pedestal with a V relief, a clevis-style top
eye with a real through-hole, a radial red knob with a diamond-knurled grip and
smooth domed face, a tapered lower stem and a bottom eye with a real
through-hole and dark bushing.  All candidates remain manual-only and pending
human review; nothing replaces the library mesh or enters automatic placement.
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
REFERENCE_NAME = '黑色AERO减震器六视图图册.png'
REFERENCE_SHA256 = '05a49a00bfb9ba988c9e64b4197a062029c59b625d2010f36daab00c0104e5e8'
ORIGINAL_JPG = ROOT / 'external/异物生成/input/objects/气压减震器.jpg'
ORIGINAL_JPG_SHA256 = '81b00e175e785e7c93d334c39a64503f31d368012d1aad7cfbf5b4eaf2c79d72'
CLEAN_SHA256 = '779e65c43cfc691f98c2b202ca06d788f5de902df511db16282db24f0b84b9e8'
RUST_SHA256 = 'eac91dd7a2154fe3a2da8b871dc09f93e02a51e4a5631eec4e2c75f86f95066c'
OLD_MESH_SHA256 = 'aead1dba91d9584f1cb075955f36e7b100f1b45e65a81144da02ca23d6057d94'
SOURCE_ALPHA_SHA256 = 'd8a288f194e8b9231cee9e163ac2c96fd773bc52640f68a12f85fbb6de80314c'
FONT = 'C:/Windows/Fonts/msyh.ttc'
FONTB = 'C:/Windows/Fonts/msyhbd.ttc'
LOGO_FONT = 'C:/Windows/Fonts/ariblk.ttf'


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


def library_row(number=17):
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


def sample_region(rgb, alpha, rect):
    x0, y0, x1, y1 = rect
    region = rgb[y0:y1, x0:x1]
    mask = alpha[y0:y1, x0:x1] > 200
    return region, mask


def logo_mask(width, height, config, body_z0, body_z1, body_radius):
    """Draw the triangular AERO mark in cylinder-UV space.

    The design is drawn at its physical aspect ratio first and then stretched
    vertically into the UV rectangle, because the cylinder wraps circumference
    to u and height to v.  This keeps the mark correctly proportioned after
    wrapping.
    """
    logo = config['logo']
    u_px = int(round(logo['width_m'] / (2 * math.pi * body_radius) * width))
    v_px = int(round(logo['height_m'] / (body_z1 - body_z0) * height))
    design = Image.new('L', (560, 200), 0)
    draw = ImageDraw.Draw(design)
    draw.polygon(((280, 0), (80, 116), (480, 116)), fill=255)
    draw.polygon(((280, 40), (170, 116), (390, 116)), fill=0)
    try:
        logo_font = ImageFont.truetype(LOGO_FONT, 86)
    except Exception:
        logo_font = ImageFont.load_default()
    text = 'AERO'
    bbox = draw.textbbox((0, 0), text, font=logo_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_layer = Image.new('L', (tw + 10, th + 10), 0)
    ImageDraw.Draw(text_layer).text((5 - bbox[0], 5 - bbox[1]), text,
                                    font=logo_font, fill=255)
    text_layer = text_layer.resize((520, 64), Image.Resampling.LANCZOS)
    design.paste(text_layer, (20, 132))
    design = design.resize((max(1, u_px), max(1, v_px)), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (width, height), 0)
    x0 = int(round(logo['u_center'] * width - design.width / 2))
    y0 = int(round((1 - logo['v_center']) * height - design.height / 2))
    canvas.paste(design, (x0, y0))
    return np.asarray(canvas).astype(np.float32) / 255.0, u_px, v_px


def draw_scratches(rgb, rng, count, light=True):
    height, width = rgb.shape[:2]
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(count):
        x = int(rng.integers(10, width - 10))
        y = int(rng.integers(10, height - 10))
        length = int(rng.integers(6, 46))
        angle = float(rng.normal(0.0, 0.12))
        dx = int(length * math.sin(angle))
        dy = int(length * math.cos(angle))
        if light and rng.random() > 0.35:
            color = (150, 150, 150, int(rng.integers(8, 24)))
        else:
            color = (16, 16, 16, int(rng.integers(8, 26)))
        draw.line((x, y, x + dx, y + dy), fill=color, width=1)
    layer = np.asarray(overlay).astype(np.float32)
    alpha = layer[:, :, 3:4] / 255.0
    out = rgb * (1 - alpha) + layer[:, :, :3] * alpha
    return np.clip(out, 0, 255)


def make_body_texture(clean_path, rust_path, out, config):
    clean = Image.open(clean_path).convert('RGBA')
    rust = Image.open(rust_path).convert('RGBA')
    sampling = config['texture_sampling']
    if list(clean.size) != sampling['clean_size']:
        raise ValueError('17 clean source dimensions changed; sampling rects need review')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    clean_rgb = np.asarray(clean)[:, :, :3].astype(np.float32)
    rust_rgb = np.asarray(rust)[:, :, :3].astype(np.float32)
    alpha = np.asarray(clean.getchannel('A'))
    width = height = 2048
    rng = np.random.default_rng(17001)
    low = norm01(periodic_field(rng, height, width, 256, 28, 36))
    fine = norm01(periodic_field(rng, height, width, 128, 1.8, 1.2))
    brush = norm01(periodic_field(rng, height, width, 128, 6.0, 0.7))
    texture = (0.55 * (low - 0.5) + 0.25 * (fine - 0.5) +
               0.20 * (brush - 0.5))
    body_rgb = np.asarray(config['material_profile']['body']['base_rgb'], np.float32)
    rgb = np.tile(body_rgb, (height, width, 1))
    rgb *= (1 + texture[:, :, None] * 0.16)
    region_rgb, region_mask = sample_region(clean_rgb, alpha,
                                            sampling['body_region_px'])
    dark = region_rgb[np.logical_and(region_mask,
                                     region_rgb.mean(axis=2) < 70)]
    sampled_dark = np.median(dark, axis=0) if len(dark) > 100 else body_rgb
    rgb = rgb * 0.86 + sampled_dark[None, None, :] * 0.14
    # Faint warm dust/grime from the rust-photo difference, strongly clamped so
    # the anodised body does not turn brown or black.
    luma_clean = clean_rgb.mean(axis=2)
    luma_rust = rust_rgb.mean(axis=2)
    diff = np.clip((luma_clean - luma_rust - 3.0) / 50.0, 0, 1)
    diff_small = cv2.resize(diff, (width // 4, height // 4),
                            interpolation=cv2.INTER_AREA)
    diff_small = gaussian_filter(diff_small, 2.0)
    grime = np.clip(cv2.resize(diff_small, (width, height),
                               interpolation=cv2.INTER_CUBIC) - 0.35, 0, 0.65)
    grime = np.clip(grime * (0.35 + 0.65 * brush), 0, 0.32)
    rgb = rgb * (1 - grime[:, :, None] * 0.12) + \
        np.array([96, 78, 62], np.float32)[None, None, :] * (grime[:, :, None] * 0.12)
    rgb = draw_scratches(rgb, rng, 42)
    mask, u_px, v_px = logo_mask(width, height, config,
                                  config['dimensions']['body_z0_m'],
                                  config['dimensions']['body_z1_m'],
                                  config['dimensions']['body_radius_m'])
    logo_rgb = np.asarray(config['logo']['color'], np.float32)
    rgb = rgb * (1 - mask[:, :, None]) + logo_rgb[None, None, :] * mask[:, :, None]
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    scratch_height = np.clip((255 - np.asarray(image).mean(axis=2)) / 255.0, 0, 1)
    height_field = (0.45 * brush + 0.25 * fine + 0.12 * low +
                    0.06 * mask + 0.10 * scratch_height)
    normal = height_to_normal(height_field, 0.34)
    roughness = np.clip(0.33 + 0.035 * (low - 0.5) + 0.02 * (fine - 0.5) +
                        0.10 * mask + 0.02 * scratch_height, 0.24, 0.48)
    metallic = np.clip(0.64 + 0.05 * (low - 0.5) - 0.04 * mask, 0.52, 0.72)
    metallic = metallic * (1 - mask) + 0.0 * mask
    mr = orm_image(roughness, metallic)
    luma = np.asarray(image).mean(axis=2)
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
        'logo_mask_fraction': round(float(mask.mean()), 4),
        'logo_pixels': [u_px, v_px],
        'sampled_dark_rgb': [round(float(v), 1) for v in sampled_dark],
        'rust_grime_fraction_used': round(float(grime.mean()), 4),
        'note': ('Black anodised aluminium (metallic ~0.64). The white AERO mark '
                 'is redrawn from the original 447 px photo and the six-view '
                 'sheet and is dielectric paint (metallic 0). Ambient-light '
                 'gradients are not baked into the base colour.'),
    }
    return paths, stats


def make_frame_texture(out, config):
    width = height = 1024
    rng = np.random.default_rng(17002)
    low = norm01(periodic_field(rng, height, width, 128, 16, 22))
    fine = norm01(periodic_field(rng, height, width, 64, 1.2, 0.8))
    brush = norm01(periodic_field(rng, height, width, 64, 4.0, 0.6))
    body_rgb = np.asarray(config['material_profile']['frame']['base_rgb'], np.float32)
    rgb = np.tile(body_rgb, (height, width, 1))
    rgb *= (1 + (0.55 * (low - 0.5) + 0.25 * (fine - 0.5) +
                  0.20 * (brush - 0.5))[:, :, None] * 0.15)
    rgb = draw_scratches(rgb, rng, 30)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    height_field = 0.55 * brush + 0.30 * fine + 0.15 * low
    normal = height_to_normal(height_field, 0.30)
    roughness = np.clip(0.35 + 0.035 * (low - 0.5) + 0.02 * (fine - 0.5),
                        0.27, 0.47)
    metallic = np.clip(0.62 + 0.05 * (low - 0.5), 0.54, 0.70)
    mr = orm_image(roughness, metallic)
    luma = np.asarray(image).mean(axis=2)
    paths = {}
    for name, array in (('frame_base', np.asarray(image)),
                        ('frame_mr', mr), ('frame_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'dark_fraction_below_20': round(float((luma < 20).mean()), 4),
        'mean_chroma': round(float((np.asarray(image).max(axis=2) -
                                    np.asarray(image).min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'note': ('Plain black anodised finish for the cap, pedestal, lower stem '
                 'and eye lugs. Fine axial brushing and scratches are '
                 'inferred from the single photograph.'),
    }
    return paths, stats


def make_machined_texture(clean_path, rust_path, out, config):
    clean = Image.open(clean_path).convert('RGBA')
    rust = Image.open(rust_path).convert('RGBA')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    sampling = config['texture_sampling']
    c = np.asarray(clean)
    r = np.asarray(rust)
    x0, y0, x1, y1 = sampling['silver_region_px']
    region = c[y0:y1, x0:x1, :3].astype(np.float32)
    region_a = c[y0:y1, x0:x1, 3]
    rust_region = r[y0:y1, x0:x1, :3].astype(np.float32)
    sat = region.max(axis=2) - region.min(axis=2)
    region_luma = region.mean(axis=2)
    silver = region[np.logical_and.reduce((region_a > 200, sat < 24,
                                           region_luma > 95))]
    sample = np.median(silver, axis=0) if len(silver) > 100 else \
        np.array([150, 150, 150], np.float32)
    rust_luma = rust_region.mean(axis=2)
    clean_luma = region.mean(axis=2)
    stain_evidence = np.clip((clean_luma - rust_luma - 2.0) / 45.0, 0, 1)
    stain_coverage = float((stain_evidence > 0.12).mean())
    rust_sample = rust_region[np.logical_and.reduce((region_a > 200,
                                                     rust_luma > 55,
                                                     rust_region[:, :, 0] >
                                                     rust_region[:, :, 2] * 1.05))]
    rust_color = np.median(rust_sample, axis=0) if len(rust_sample) > 100 else \
        np.array([112, 82, 58], np.float32)
    if float(rust_color.max() - rust_color.min()) < 12.0:
        # The silver-band crop is dominated by neutral grey; keep the measured
        # luminance but restore an explicitly conservative brown tint so the
        # rust-photo evidence remains visible.
        rust_color = 0.45 * rust_color + 0.55 * np.array([112, 82, 58], np.float32)
    width = height = 1024
    rng = np.random.default_rng(17003)
    # Circumferential machining lines run along u: fine variation across v.
    rings = norm01(periodic_field(rng, height, width, 256, 0.6, 30.0))
    fine = norm01(periodic_field(rng, height, width, 64, 1.0, 0.9))
    stains = norm01(periodic_field(rng, height, width, 128, 18, 20))
    stains = np.clip((stains - 0.45) * 2.0, 0, 1)
    stains = gaussian_filter(stains, 1.2)
    base = np.array([178, 178, 176], np.float32) * 0.45 + sample.astype(np.float32) * 0.55
    rgb = np.tile(base, (height, width, 1))
    rgb *= (1 + 0.06 * (rings - 0.5) + 0.03 * (fine - 0.5))[:, :, None]
    stain_amount = np.clip(stains * min(0.62, 0.30 + stain_coverage * 0.6), 0, 0.55)
    rgb = rgb * (1 - stain_amount[:, :, None]) + \
        rust_color[None, None, :] * stain_amount[:, :, None]
    rgb = draw_scratches(rgb, rng, 26)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    height_field = 0.55 * rings + 0.30 * fine + 0.15 * stains
    normal = height_to_normal(height_field, 0.30)
    roughness = np.clip(0.34 + 0.03 * (rings - 0.5) + 0.03 * (fine - 0.5) +
                        0.08 * stain_amount, 0.25, 0.55)
    metallic = np.ones_like(roughness)
    mr = orm_image(roughness, metallic)
    luma = np.asarray(image).mean(axis=2)
    paths = {}
    for name, array in (('machined_base', np.asarray(image)),
                        ('machined_mr', mr), ('machined_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'dark_fraction_below_20': round(float((luma < 20).mean()), 4),
        'mean_chroma': round(float((np.asarray(image).max(axis=2) -
                                    np.asarray(image).min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'min_metallic': round(float(metallic.min()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'min_roughness': round(float(roughness.min()), 4),
        'max_roughness': round(float(roughness.max()), 4),
        'sampled_silver_rgb': [round(float(v), 1) for v in sample],
        'sampled_rust_rgb': [round(float(v), 1) for v in rust_color],
        'rust_evidence_coverage': round(stain_coverage, 4),
        'note': ('Bright machined steel/aluminium with metallic 1. The rust '
                 'stains are clamped and derived from the clean/rust photo '
                 'difference in the silver-band region; the exact coating and '
                 'stain distribution are inferred.'),
    }
    return paths, stats


def make_grip_texture(out, config):
    width = height = 1024
    rng = np.random.default_rng(17004)
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    U, V = np.meshgrid(u, v)
    ku, kv = 16, 4
    knurl = np.sin(2 * math.pi * (ku * U + kv * V)) + \
        np.sin(2 * math.pi * (ku * U - kv * V))
    knurl = (knurl - knurl.min()) / (knurl.max() - knurl.min())
    mottle = norm01(periodic_field(rng, height, width, 128, 20, 24))
    base = np.asarray(config['material_profile']['grip']['base_rgb'], np.float32)
    factor = 0.80 + 0.20 * knurl
    rgb = np.tile(base, (height, width, 1)) * factor[:, :, None]
    rgb *= (1 + 0.04 * (mottle - 0.5)[:, :, None])
    edge = np.clip(np.minimum(V, 1 - V) / 0.04, 0, 1)
    rgb *= (0.82 + 0.18 * edge)[:, :, None]
    rgb = np.clip(rgb, 0, 255)
    image = Image.fromarray(rgb.astype(np.uint8))
    normal = height_to_normal(0.9 * knurl + 0.1 * mottle, 1.1)
    roughness = np.clip(0.40 + 0.10 * (1 - knurl) + 0.03 * (mottle - 0.5),
                        0.32, 0.52)
    metallic = np.clip(0.55 + 0.03 * (mottle - 0.5), 0.48, 0.64)
    mr = orm_image(roughness, metallic)
    luma = np.asarray(image).mean(axis=2)
    chroma = np.asarray(image).max(axis=2) - np.asarray(image).min(axis=2)
    paths = {}
    for name, array in (('grip_base', np.asarray(image)),
                        ('grip_mr', mr), ('grip_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'mean_chroma': round(float(chroma.mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'knurl_diamonds_around': 2 * ku,
        'knurl_bands_along': 2 * kv,
        'note': ('Red anodised aluminium grip with a real diamond-knurl normal '
                 'map (2 x 16 diamonds around). The knurl pitch and the exact '
                 'red dye are inferred; the rendered silhouette is a cylinder.'),
    }
    return paths, stats


def texture_checks(body, frame, machined, grip):
    body_ok = (30 <= body['mean_luma'] <= 120 and
               0.45 <= body['mean_metallic'] <= 0.85 and
               0.22 <= body['mean_roughness'] <= 0.50 and
               body['dark_fraction_below_20'] <= 0.75)
    frame_ok = (25 <= frame['mean_luma'] <= 120 and
                0.45 <= frame['mean_metallic'] <= 0.85 and
                0.25 <= frame['mean_roughness'] <= 0.52)
    machined_ok = (90 <= machined['mean_luma'] <= 225 and
                   machined['mean_metallic'] >= 0.95 and
                   0.20 <= machined['mean_roughness'] <= 0.58)
    grip_ok = (35 <= grip['mean_luma'] <= 150 and
               grip['mean_chroma'] >= 40 and
               0.45 <= grip['mean_metallic'] <= 0.85 and
               0.20 <= grip['mean_roughness'] <= 0.50)
    return {
        'body': {'passed': bool(body_ok), **body},
        'frame': {'passed': bool(frame_ok), **frame},
        'machined': {'passed': bool(machined_ok), **machined},
        'grip': {'passed': bool(grip_ok), **grip},
        'note': ('Black anodised body/frame are metallic rather than pure '
                 'dielectric; the white AERO print and the dark bushing are the '
                 'only dielectric surfaces, and they are not structural metal. '
                 'The checks reject black/over-smooth and non-metallic machined '
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
    textured_names = ('body', 'frame', 'machined', 'grip')
    textured = [item for item in materials if item['name'] in textured_names]
    images = document.get('images', [])
    passed = (len(textured) == 4 and
              all(item['base_color_texture'] and item['metallic_roughness_texture'] and
                  item['normal_texture'] for item in textured) and
              len(images) >= 12 and
              all(image.get('mimeType') == 'image/png' for image in images))
    return {
        'passed': bool(passed),
        'version': version,
        'materials': materials,
        'image_count': len(images),
        'texture_count': len(document.get('textures', [])),
        'node_count': len(document.get('nodes', [])),
        'mesh_count': len(document.get('meshes', [])),
        'note': ('The GLB embeds 12 PNG images and separate base-colour, '
                 'metallic-roughness and normal textures for body, frame, '
                 'machined and grip. Knob cap and bushing use scalar PBR '
                 'factors. This audit does not prove the visual appearance in '
                 'every viewer.'),
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
    top_z = worker['final_hole_z_m']['top']
    bottom_z = worker['final_hole_z_m']['bottom']
    V = config['features']['top_pedestal']['v_notch']
    record = {}

    top_center = ray_hits(mesh, (0.05, 0.0, top_z), (-1, 0, 0), multiple=True)
    top_offset = ray_hits(mesh, (0.05, 0.010, top_z), (-1, 0, 0), multiple=True)
    record['top_eye_hole'] = {
        'centre_hits': len(top_center),
        'offset_hits': len(top_offset),
        'passed': bool(len(top_center) == 0 and len(top_offset) >= 1),
        'note': ('A centre ray along the eye axis is clear through the dark '
                 'bushing bore, while an offset ray hits the lug. This proves a '
                 'real through-hole; its diameter and the bushing fit are '
                 'inferred.'),
    }

    bottom_center = ray_hits(mesh, (0.05, 0.0, bottom_z), (-1, 0, 0), multiple=True)
    bottom_offset = ray_hits(mesh, (0.05, 0.009, bottom_z), (-1, 0, 0), multiple=True)
    record['bottom_eye_hole'] = {
        'centre_hits': len(bottom_center),
        'offset_hits': len(bottom_offset),
        'passed': bool(len(bottom_center) == 0 and len(bottom_offset) >= 1),
        'note': ('The centre ray passes through the bushing bore and the open '
                 'lug hole; an offset ray hits the lug/rim. The bushing wall '
                 'thickness and fit are inferred.'),
    }

    shift = worker['z_shift_m']
    ring_hit = ray_hits(mesh, (0.05, 0.0, 0.1046 + shift), (-1, 0, 0))
    body_hit = ray_hits(mesh, (0.05, 0.0, 0.1020 + shift), (-1, 0, 0))
    ring_x = float(ring_hit[0][0]) if len(ring_hit) else None
    body_x = float(body_hit[0][0]) if len(body_hit) else None
    inset = (body_x - ring_x) if ring_x is not None and body_x is not None else None
    record['silver_ring_groove'] = {
        'ring_hit_x_m': round(ring_x, 6) if ring_x is not None else None,
        'body_hit_x_m': round(body_x, 6) if body_x is not None else None,
        'inset_m': round(float(inset), 6) if inset is not None else None,
        'passed': bool(inset is not None and inset > 0.0003),
        'note': ('A radial ray at the silver band stops inside the adjacent body '
                 'radius, confirming a real recessed band rather than a painted '
                 'line. Band width and depth are inferred.'),
    }

    notch_center = ray_hits(mesh, (0.05, 0.0, 0.148 + shift), (-1, 0, 0))
    notch_side = ray_hits(mesh, (0.05, 0.010, 0.148 + shift), (-1, 0, 0))
    center_x = float(notch_center[0][0]) if len(notch_center) else None
    side_x = float(notch_side[0][0]) if len(notch_side) else None
    notch_depth = (side_x - center_x) if center_x is not None and side_x is not None else None
    record['pedestal_v_notch'] = {
        'centre_hit_x_m': round(center_x, 6) if center_x is not None else None,
        'side_hit_x_m': round(side_x, 6) if side_x is not None else None,
        'depth_m': round(float(notch_depth), 6) if notch_depth is not None else None,
        'passed': bool(notch_depth is not None and notch_depth > 0.004),
        'note': ('A ray through the V centre reaches the notch floor; a parallel '
                 'ray outside the V hits the front face. The V outline and depth '
                 'are inferred from the photograph and six-view sheet.'),
    }

    azimuth = math.radians(config['features']['knob']['azimuth_deg'])
    direction = np.array((math.cos(azimuth), math.sin(azimuth), 0.0))
    knob_center = np.array((0.0, 0.0,
                            config['features']['knob']['center_z_m'] + shift))
    named = load_named_parts(path)
    knob_names = ('Knob boss', 'Knob collar ring', 'Knob grip',
                  'Knob seam ring', 'Knob domed cap')
    knob_mesh = trimesh.util.concatenate([named[name] for name in knob_names
                                         if name in named])
    knob_hit = ray_hits(knob_mesh, tuple(knob_center + direction * 0.060),
                        tuple(-direction))
    hit_radius = None
    if len(knob_hit):
        offset = np.asarray(knob_hit[0]) - knob_center
        hit_radius = float(np.linalg.norm(offset[:2]))
    record['knob_solid'] = {
        'hits': int(len(knob_hit)),
        'hit_radius_m': round(hit_radius, 6) if hit_radius is not None else None,
        'passed': bool(knob_hit is not None and len(knob_hit) >= 1 and
                       hit_radius is not None and 0.040 <= hit_radius <= 0.048),
        'note': ('A ray along the knob axis from outside hits the solid domed cap '
                 'at the modelled 46.2 mm radius, confirming the adjuster is a '
                 'solid part rather than a hole. The red surface is modelled as '
                 'anodised metal; its internal clicks are not modelled.'),
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
              '17 气压减震器 · 六方向：未标定视觉参考 / 原模型 / 新候选（同一中性光）',
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
    names = ('top-eye', 'top-eye-hole', 'top-pedestal', 'knob', 'knob-knurl',
             'silver-ring', 'aero-logo', 'bottom-lip', 'lower-stem',
             'bottom-eye', 'bottom-eye-hole', 'surface', 'top-detail')
    titles = {
        'top-eye': '顶部吊耳', 'top-eye-hole': '通孔/深色衬套与银色孔沿',
        'top-pedestal': '顶部座与 V 形凹口', 'knob': '红色调节旋钮',
        'knob-knurl': '菱形滚花近景', 'silver-ring': '银色分隔环',
        'aero-logo': 'AERO 标志（重绘）', 'bottom-lip': '底部机加工亮环',
        'lower-stem': '下段锥形筒', 'bottom-eye': '底部吊耳',
        'bottom-eye-hole': '底部通孔/深色衬套', 'surface': '黑色阳极氧化表面',
        'top-detail': '顶视细节',
    }
    tile = (620, 470)
    cols = 4
    rows = math.ceil(len(names) / cols)
    board = Image.new('RGB', (cols * tile[0] + 30, rows * (tile[1] + 44) + 70),
                      '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '17 气压减震器 · 关键局部检查（新版，中性光）',
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
    items = (('top-mount-only', '顶部吊耳/座/旋钮（隔离）'),
             ('bottom-eye-only', '下段与底部吊耳（隔离）'),
             ('bottom-eye-bottom', '底部吊耳底视（隔离）'),
             ('body-only', '筒身/银环/顶盖（隔离）'),
             ('body-only-top', '筒身顶视轮廓（圆度检查）'))
    tile = (620, 500)
    board = Image.new('RGB', (3 * tile[0] + 30, 2 * (tile[1] + 50) + 230), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '17 气压减震器 · 隔离结构与开口检查（渲染 + 射线）',
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
        f"顶部吊耳：中心射线命中 {ray['top_eye_hole']['centre_hits']} 次，"
        f"偏移射线命中 {ray['top_eye_hole']['offset_hits']} 次",
        f"底部吊耳：中心射线命中 {ray['bottom_eye_hole']['centre_hits']} 次，"
        f"偏移射线命中 {ray['bottom_eye_hole']['offset_hits']} 次",
        f"银色环槽：环面 x={ray['silver_ring_groove']['ring_hit_x_m']} m，"
        f"相邻筒身 x={ray['silver_ring_groove']['body_hit_x_m']} m，内凹="
        f"{ray['silver_ring_groove']['inset_m']} m",
        f"顶部座 V 凹口：中心 x={ray['pedestal_v_notch']['centre_hit_x_m']} m，"
        f"侧向 x={ray['pedestal_v_notch']['side_hit_x_m']} m，深度="
        f"{ray['pedestal_v_notch']['depth_m']} m",
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
    draw.text((18, 12), '17 气压减震器 · 同光照材质对照（左：原模型，右：新候选）',
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
    draw.text((18, 12), '17 气压减震器 · 变光检查（高光必须随现场光移动）',
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
              '17 气压减震器 · 表面反射变光对照（上两行：低环境光+移动主光；下两行：中性主光）',
              font=font(24, True), fill='#17232a')
    pairs = (('metal-ring-key', '银色环/亮环 · 主光 A'),
             ('metal-ring-alt', '银色环/亮环 · 主光 B'),
             ('knob-key', '红色旋转钮 · 主光 A'),
             ('knob-alt', '红色旋转钮 · 主光 B'),
             ('new-material-close', '黑色阳极氧化筒身 · 主光 A'),
             ('new-surface-alternate', '黑色阳极氧化筒身 · 主光 B'))
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
    draw.text((18, 10), '17 气压减震器 · 新候选 24 方向转台（中性光，仅供检查）',
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
    draw.text((18, 12), '17 气压减震器 · 原照 / 做锈素材 / 旧模型原视角 / 新候选原视角诊断',
              font=font(24, True), fill='#17232a')
    images = (Image.open(original_jpg), Image.open(rust_path),
              Image.open(old_source_view) if Path(old_source_view).is_file() else None,
              Image.open(out / 'renders/new-source.png'))
    titles = ('原照（447 px，含 AERO 标志）', '做锈 HD（局部锈迹来源）',
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
        check_row('吊耳通孔 / V 凹口 / 银环槽', checks['isolated_openings'],
                  f"顶孔中心/偏移命中 {ray['top_eye_hole']['centre_hits']}/"
                  f"{ray['top_eye_hole']['offset_hits']}；底孔 "
                  f"{ray['bottom_eye_hole']['centre_hits']}/"
                  f"{ray['bottom_eye_hole']['offset_hits']}；"
                  f"银环内凹 {ray['silver_ring_groove']['inset_m']} m；"
                  f"V 凹口 {ray['pedestal_v_notch']['depth_m']} m"),
        check_row('结构部件计数', checks['structural_counts'],
                  f"共 {report['worker']['parts_count']} 个部件；吊耳、孔沿、滚花钮、衬套齐全"),
        check_row('筒身圆柱截面', checks['cylindrical_body'],
                  f"筒身最大半径偏差 {report['cylinder_check']['max_abs_radius_deviation_m']:.9f} m；"
                  f"筒身顶视宽高比 {report['cylinder_check']['top_view_aspect_new']}"),
        check_row('材质贴图', checks['material_textures'],
                  '黑阳极氧化/银色机加工/红色滚花/顶盖 四组 PBR；白漆为局部电介质'),
        check_row('GLB 材质内嵌', checks['glb_materials'],
                  f"{report['glb_audit']['image_count']} 张 PNG；四组各有 baseColor、metallicRoughness、normal"),
        check_row('金属反射', checks['metal_reflection'],
                  f"银环亮度差 {report['metal_reflection_check']['machined_luma_difference']}；"
                  f"红钮 {report['metal_reflection_check']['knob_luma_difference']}；"
                  f"黑筒身 {report['metal_reflection_check']['body_luma_difference']}"),
        check_row('非金属说明', 'not_applicable',
                  '衬套与 AERO 白漆为局部非金属；主承力表面均为金属，金属反射检查适用'),
        check_row('六视图参考诊断', 'diagnostic_only',
                  '六视图未标定；IoU 只用于发现差异，不作验收'),
        check_row('原视角轮廓诊断', 'diagnostic_only',
                  f"IoU {report['source_fidelity_diagnostic']['iou']}；原照为 3/4 倾斜姿态，"
                  f"候选直立，仅作诊断"),
        check_row('检查页静态资源', report['page_assets_check'].get('passed'),
                  '本地 HTML/图片路径检查；未以无头浏览器声称交互通过'),
        check_row('人工验收', 'pending', '候选 manual_only，不进入自动合成'),
    ])
    summary = (f"<p>本件以原照为主，六视图参考为未标定视觉参考。旧模型呈软团块，"
               f"顶部/底部吊耳与旋钮粘连、无真实通孔，故未复用旧网格。"
               f"候选保留真实筒身圆柱截面、银色分隔环、底部机加工亮环、"
               f"带 V 凹口的顶部座、真实贯通的上/下吊耳与深色衬套、"
               f"红色滚花旋钮和锥形下段。AERO 标志按原照与参考重绘，"
               f"参考图与尺寸均不等同实测。</p>")
    html = REVIEW_TEMPLATE
    replacements = {
        '__SUMMARY__': summary,
        '__CHECK_ROWS__': rows,
        '__TITLE__': '17 气压减震器 · 三维模型独立检查',
        '__SUBTITLE__': '黑色 AERO 气压减震器 · 分部件重建候选 · 未标定六视图参考',
        '__STATUS__': 'manual_only · 待人工审阅',
        '__VERSION__': report['version'],
        '__SOURCE_IOU__': str(report['source_fidelity_diagnostic'].get('iou')),
        '__TOP_ASPECT__': str(report['cylinder_check']['top_view_aspect_new']),
        '__BODY_LUMA__': str(texture_stats['body']['mean_luma']),
        '__FRAME_LUMA__': str(texture_stats['frame']['mean_luma']),
        '__MACHINED_METAL__': str(texture_stats['machined']['mean_metallic']),
        '__GRIP_METAL__': str(texture_stats['grip']['mean_metallic']),
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
<div class="notice"><b>参考状态：</b>六视图映射为 visual_match_unverified（未标定）。__REFERENCE_NOTE__ 背面、吊耳孔深/衬套配合、顶部座内部、旋钮内部、壁厚、标志印刷配准和隐藏面为推断。候选为 <b>manual_only</b>，不替换原模型，不进入自动放置或正式合成。</div>
<h2>检查记录</h2>
<table><thead><tr><th>项目</th><th>结果</th><th>说明</th></tr></thead><tbody>__CHECK_ROWS__</tbody></table>
<div class="notice"><b>关键数字：</b>原视角诊断 IoU __SOURCE_IOU__（3/4 原照与直立候选，未标定，只作诊断）；筒身顶视宽高比 __TOP_ASPECT__；筒身/框架平均亮度 __BODY_LUMA__/__FRAME_LUMA__；机加工金属度 __MACHINED_METAL__；旋钮金属度 __GRIP_METAL__。</div>
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
    parser.add_argument('--number', type=int, default=17)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    parser.add_argument('--draft', action='store_true')
    args = parser.parse_args()
    if args.number != 17:
        parser.error('this entry supports item 17 only')
    args.reference = (args.reference or find_reference()).resolve()
    if args.out is None:
        args.out = ROOT / ('outputs/17气压减震器-六视图重建-draft' if args.draft
                           else 'outputs/17气压减震器-六视图重建-v1')
    args.out, args.library, args.blender = (x.resolve() for x in
                                            (args.out, args.library, args.blender))
    if digest(args.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = library_row(17)
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
        raise ValueError('original 447 px photograph hash differs from the HD catalogue')
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

    config_path = HERE / 'shock17_config.json'
    worker_path = HERE / 'mesh_rebuild_17_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
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
        frame_textures, frame_stats = make_frame_texture(args.out, config)
        machined_textures, machined_stats = make_machined_texture(
            clean_path, rust_path, args.out, config)
        grip_textures, grip_stats = make_grip_texture(args.out, config)
        textures = {**body_textures, **frame_textures, **machined_textures,
                    **grip_textures}
        stats = texture_checks(body_stats, frame_stats, machined_stats, grip_stats)
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
    body_mesh = parts.get('Canister body')
    if body_mesh is None:
        raise ValueError('exported GLB is missing the Canister body node')
    body_vertices = np.asarray(body_mesh.vertices)
    all_radii = np.hypot(body_vertices[:, 0], body_vertices[:, 1])
    side = body_vertices[np.logical_and.reduce((body_vertices[:, 2] > 0.0675,
                                                body_vertices[:, 2] < 0.1031,
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
        'note': ('Only the straight side rings (0.0675 < z < 0.1031 m, r > 20 mm) '
                 'are measured; the chamfered ends and the flat centre vertices '
                 'are excluded. The check confirms cylindrical symmetry of the '
                 'modelled side, not measured dimensions.'),
    }

    if args.draft:
        draft_report = {
            'number': 17, 'draft': True, 'out': str(args.out),
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
    new_top_ratio = alpha_bbox_ratio(args.out / 'renders/body-only-top.png')
    old_top_ratio = alpha_bbox_ratio(args.out / 'renders/old-top.png')
    cylinder_check['top_view_aspect_new'] = new_top_ratio
    new_front_luma = median_luma(args.out / 'renders/new-front.png')
    old_front_luma = median_luma(args.out / 'renders/old-front.png')
    machined_difference = mean_difference(args.out / 'renders/metal-ring-key.png',
                                          args.out / 'renders/metal-ring-alt.png')
    knob_difference = mean_difference(args.out / 'renders/knob-key.png',
                                      args.out / 'renders/knob-alt.png')
    body_difference = mean_difference(args.out / 'renders/new-material-close.png',
                                      args.out / 'renders/new-surface-alternate.png')
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
        'bodies': sum(name == 'Canister body' for name in names),
        'silver_bands': sum(name == 'Silver separator band' for name in names),
        'top_caps': sum(name == 'Top cap' for name in names),
        'bottom_lips': sum(name == 'Bottom machined lip' for name in names),
        'lower_stems': sum(name == 'Lower stem' for name in names),
        'pedestal_bases': sum(name == 'Top pedestal base' for name in names),
        'pedestal_necks': sum(name == 'Top pedestal neck' for name in names),
        'top_eyes': sum(name == 'Top eye lug' for name in names),
        'top_eye_rims': sum(name in ('Top eye rim +X', 'Top eye rim -X') for name in names),
        'top_bushings': sum(name == 'Top eye bushing' for name in names),
        'knob_bosses': sum(name == 'Knob boss' for name in names),
        'knob_collars': sum(name == 'Knob collar ring' for name in names),
        'knob_grips': sum(name == 'Knob grip' for name in names),
        'knob_seams': sum(name == 'Knob seam ring' for name in names),
        'knob_caps': sum(name == 'Knob domed cap' for name in names),
        'bottom_eyes': sum(name == 'Bottom eye lug' for name in names),
        'bottom_eye_rims': sum(name in ('Bottom eye rim +X', 'Bottom eye rim -X')
                               for name in names),
        'bushings': sum(name == 'Bottom bushing' for name in names),
        'note': ('Part presence proves only that a feature was modelled; wall '
                 'thickness, bore depth, bushing fit, knob internals and all '
                 'hidden faces are inferred.'),
    }
    structural_ok = bool(structural['bodies'] == 1 and structural['silver_bands'] == 1 and
                         structural['top_caps'] == 1 and structural['bottom_lips'] == 1 and
                         structural['lower_stems'] == 1 and structural['pedestal_bases'] == 1 and
                         structural['pedestal_necks'] == 1 and structural['top_eyes'] == 1 and
                         structural['top_eye_rims'] == 2 and structural['top_bushings'] == 1 and
                         structural['knob_bosses'] == 1 and
                         structural['knob_collars'] == 1 and structural['knob_grips'] == 1 and
                         structural['knob_seams'] == 1 and structural['knob_caps'] == 1 and
                         structural['bottom_eyes'] == 1 and structural['bottom_eye_rims'] == 2 and
                         structural['bushings'] == 1)
    cylindrical_ok = bool(cylinder_check['max_abs_radius_deviation_m'] < 1e-6 and
                          cylinder_check['max_ring_radius_variation_m'] < 1e-6 and
                          new_top_ratio is not None and abs(new_top_ratio - 1.0) < 0.03)
    metal_reflection = {
        'status': 'passed' if (stats['machined']['mean_metallic'] >= 0.95 and
                               stats['grip']['mean_metallic'] >= 0.45 and
                               stats['body']['mean_metallic'] >= 0.45 and
                               machined_difference is not None and machined_difference > 2.0 and
                               knob_difference is not None and knob_difference > 2.0 and
                               body_difference is not None and body_difference > 1.0)
                  else 'failed_or_inconclusive',
        'machined_mean_metallic': stats['machined']['mean_metallic'],
        'grip_mean_metallic': stats['grip']['mean_metallic'],
        'body_mean_metallic': stats['body']['mean_metallic'],
        'machined_luma_difference': machined_difference,
        'knob_luma_difference': knob_difference,
        'body_luma_difference': body_difference,
        'note': ('The machined band/lip/rims are metallic 1, the red knob and '
                 'black anodised body are metallic. The ring and knob response '
                 'is measured in a dedicated low-ambient render with a strong '
                 'moved key so the specular lobe must move; the body response is '
                 'measured in the standard neutral close renders. This is a '
                 'response test, not a claim about unseen surface finish.'),
    }
    dielectric = {
        'status': 'not_applicable_as_primary_surface',
        'note': ('The only dielectric surfaces are the dark bottom bushing and '
                 'the white AERO paint; the load-bearing visible surfaces are '
                 'metal, so metal reflection checks apply. No dielectric '
                 'specular-movement acceptance is claimed for the paint.'),
    }
    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] == 20 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components'] and
                               worker['top_eye_boundary_edges_after_boolean'] == 0 and
                               worker['bottom_eye_boundary_edges_after_boolean'] == 0),
        'isolated_openings': bool(all(ray[key]['passed'] for key in
                                      ('top_eye_hole', 'bottom_eye_hole',
                                       'silver_ring_groove', 'pedestal_v_notch',
                                       'knob_solid'))),
        'structural_counts': structural_ok,
        'cylindrical_body': cylindrical_ok,
        'material_textures': bool(all(stats[key]['passed'] for key in
                                      ('body', 'frame', 'machined', 'grip'))),
        'glb_materials': glb['passed'],
        'metal_reflection': metal_reflection['status'] == 'passed',
        'reference_diagnostic_only': True,
        'source_fidelity_diagnostic': fidelity,
        'source_pose_diagnostic_only': True,
        'browser_page': 'not_run',
    }
    render_luma = {
        'new_front_median': new_front_luma,
        'old_front_median': old_front_luma,
        'not_black': bool(new_front_luma is not None and new_front_luma >= 25),
        'not_blown': bool(new_front_luma is not None and new_front_luma <= 170),
        'note': ('Neutral-light luma only; it is used to reject a black or '
                 'blown-out render, not as a lighting acceptance.'),
    }

    audit_fidelity = None
    audit_path = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
    if audit_path.is_file():
        for line in audit_path.read_text(encoding='utf8').splitlines():
            if line.strip() and json.loads(line).get('number') == 17:
                audit_fidelity = json.loads(line).get('source_fidelity')
                break

    version = config['version']
    report = {
        'version': version, 'number': 17, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': config['shape_basis'], 'reference_role': config['reference_role'],
        'method': 'parametric_part_rebuild_from_source_photo_and_uncalibrated_six_view_reference',
        'reference': str(args.reference), 'reference_sha256': digest(args.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'original_mesh': str(old_mesh_path), 'original_mesh_sha256': OLD_MESH_SHA256,
        'original_photo_447': str(ORIGINAL_JPG),
        'original_photo_447_sha256': ORIGINAL_JPG_SHA256,
        'source_photo': str(clean_path), 'source_view': str(source_view_path),
        'source_alpha': str(alpha_path), 'original_clean_sha256': CLEAN_SHA256,
        'original_rust_sha256': RUST_SHA256,
        'provenance_note': (
            'The shared model-library record contains absolute paths from the '
            'original main worktree. Every input was relocated to the matching '
            'file inside this parallel worktree before use, and its registered '
            'SHA-256 was verified. No output is written outside this worktree.'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9501 only measures one dark '
            'silhouette. The six-view audit renders show a fused top eye, lower '
            'eye and adjuster, a soft non-cylindrical can and no true holes; the '
            'old mesh is therefore not reused as the candidate base geometry.'),
        'original_source_fidelity_record': audit_fidelity,
        'catalog_evidence': config['proportion_source'],
        'identity_note': config['proportion_source']['identity_note'],
        'dimensions': config['dimensions'],
        'logo': config['logo'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(args.out / 'mesh.glb'), 'mesh_sha256': digest(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'), 'blend_sha256': digest(args.out / 'model.blend'),
        'textures': textures, 'texture_stats': stats,
        'material_profile': config['material_profile'],
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The source camera is the calibrated camera from the old model '
            'library. It is a single 3/4 photograph and the candidate is '
            'modelled upright, so the IoU is a diagnostic silhouette comparison, '
            'not a calibrated turntable and not an acceptance gate.'),
        'reference_views': reference_results,
        'reference_diagnostic_note': (
            'The six-view sheet is uncalibrated and its proportions visibly '
            'differ from the photograph; all IoU values are diagnostic only.'),
        'top_view_aspect': {'new': new_top_ratio, 'old': old_top_ratio,
                            'note': ('Width/height of the isolated body-only top '
                                     'silhouette (new) and the old model top render. '
                                     'Only the new isolated can is expected to be '
                                     'circular.')},
        'render_luma': render_luma,
        'mesh_audit': mesh_audit, 'glb_audit': glb, 'worker': worker,
        'cylinder_check': cylinder_check,
        'ray_checks': ray, 'structural_checks': structural,
        'metal_reflection_check': metal_reflection,
        'dielectric_check': dielectric,
        'checks': checks,
        'page_assets_check': check_review_page(args.out),
        'limitations': [
            'The six-view sheet is an uncalibrated generated visual reference, not a measured turntable; it is used for topology and appearance only.',
            'Only one 3/4 photograph exists. The unseen back, the bore depths, the top-pedestal interior, the knob mechanism and the bushing fit are inferred.',
            'Working dimensions (186 mm overall, 62 mm can diameter, about 160.5 mm eye-to-eye) follow the photo pixel ratio plus a similar-catalogue plausibility check; they are not measured dimensions and the exact model/SKU is unconfirmed.',
            'The AERO mark is redrawn from the low-resolution original photo and the six-view sheet; exact print size, ink gloss and registration are inferred. A faint white mark at the source photo left silhouette could not be identified and is not reproduced.',
            'The red knob silhouette is a cylinder with a diamond-knurl normal map; the exact knurl pitch, internal adjuster clicks and red dye are not measured.',
            'The clevis-style top eye, the V relief and the silver separator band follow the visible structure; hidden wall thickness and the bushing fit are inferred.',
            'The black anodised body is metallic in the PBR model; a real anodic oxide is a thin dielectric layer over metal, so the exact dielectric/metal mix is an approximation.',
            'Assemblies intentionally overlap at the pedestal/cap, lug/stem and knob/neck joints; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'The source silhouette IoU is diagnostic only: the photograph is a tilted 3/4 product shot while the candidate is modelled upright.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views, the top/bottom eyes, the V relief, the silver band, the red knob and the AERO mark against the photo and the physical sample.',
            'Confirm the actual overall length, can diameter, eye-to-eye distance and eye-hole diameters with calipers or a measured drawing.',
            'Confirm the unseen back, eye bore depths, bushing type/fit and knob internal mechanism on the real part.',
            'Confirm the AERO print size, colour, registration and whether another print exists on the opposite side.',
            'Confirm the bottom eye/bushing geometry and the lower stem taper with the real part.',
            'Verify coordinate/scale, stability, floor contact and lighting integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
        ],
        'next_number': 18,
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
        'number': 17, 'name': row['file'], 'asset_id': row['asset_id'],
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
        'pending_review': report['pending_review'], 'next_number': 18,
        'notes': ('The six-view sheet is an uncalibrated visual reference; the '
                  'single 3/4 photograph is the primary evidence. The old model '
                  'and other items were not modified. Candidate is manual_only '
                  'pending human review.'),
    }
    (args.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                           encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    external = {
        'number': 17, 'status': progress['status'],
        'out': str(args.out), 'mesh': str(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'), 'checks': checks,
        'pending_review': report['pending_review'], 'next_number': 18,
        'manual_only': True,
        'review_page': str(args.out / 'review.html'),
        'notes': progress['notes'],
    }
    (progress_dir / '17-progress.json').write_text(
        json.dumps(external, ensure_ascii=False, indent=2), encoding='utf8')
    files = [path for path in sorted(args.out.rglob('*')) if path.is_file()]
    hashes = {
        'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_17.py',
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
                      'page_assets': page_check['passed']}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
