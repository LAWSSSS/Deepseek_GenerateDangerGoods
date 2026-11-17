"""Build the review-only Botny spray-paint-can candidate (item 36).

Evidence
--------
* outputs/物品高清输入-39件/原貌HD/36-自喷漆罐.png and the matching rust
  derivative are the primary photographs of one used upright grey-cap aerosol
  spray-paint can.
* references/parallel-six-view/保赐利喷漆六视图产品目录.png is an
  uncalibrated visual reference of the same Botny design family. Its front
  panel is used only for silhouette/topology comparison; the back/side panels
  are used for the hidden half of the wrap label. The front text differs from
  the photographed can (the reference appears to state a different net weight),
  so the original photograph wins for every visible front detail.
* The old TripoSR library mesh has a top cross-section about 1.1x elliptical and
  a principal axis tilted about 11 degrees from Z, plus a fused cap/body
  silhouette and a smeared label. It is retained unchanged and is not reused.

The candidate is a parametric part rebuild. It remains manual_only / pending
human review; it never replaces the library mesh and never enters automatic
placement or the final composite.
"""
import argparse
import hashlib
import html
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
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
REFERENCE_SHA256 = 'e88d442b1240ab1d646d0406df181a020bcf1d2bfe82bc3696a24059c277122e'
REFERENCE_NAME = '保赐利喷漆六视图产品目录.png'
CLEAN_SHA256 = '539f6d7bd495525fcf9c2f95e13f904a52e0cd74b2a8fed4c3609ecbb3d4b4c7'
RUST_SHA256 = '330a386fc2cb67ab9f608383066198ae97c77f08272ebe34e63fd06a0235c3d9'
OLD_MESH_SHA256 = '96a405f7659618550423b2d2838c84325ad6c7d8345dd0e540935a585555b024'
SOURCE_VIEW_SHA256 = 'fdf14dbcb91cc3b79ad7b29763d417eee95ed3f3bcb0dbfd1ff3215dee99bd17'
SOURCE_ALPHA_SHA256 = '93bb7a1520abf7a5f4aad67d237a4a2d38519777c083a45d0bb344921fd53bfd'
FONT = 'C:/Windows/Fonts/msyh.ttc'
FONTB = 'C:/Windows/Fonts/msyhbd.ttc'
FONT_MONO = 'C:/Windows/Fonts/consola.ttf'
CHROME_CANDIDATES = (
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
)


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
    """Relocate a library-record absolute path into this parallel worktree."""
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


def library_row(number=36):
    for line in LIBRARY.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('number') == number:
            return row
    raise KeyError(f'item {number} not found in the model library')


def font(size, bold=False, mono=False):
    try:
        if mono:
            return ImageFont.truetype(FONT_MONO, size)
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


def flatten_lighting(rgb, target=235.0, mask_kind='white'):
    """Remove a smooth photographed illumination field from the albedo.

    The photograph contains studio highlights and shadows. Dividing by a
    smoothed 90th-percentile white/grey field keeps the print, scratches and
    dirt while flattening the broad shading, so the PBR renderer creates live
    highlights instead of replaying baked ones.
    """
    rgb = rgb.astype(np.float32)
    luma = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    if mask_kind == 'white':
        mask = (rgb.min(axis=2) > 135) & (chroma < 50)
    else:
        mask = (luma > 40) & (luma < 210) & (chroma < 28)
    if mask.sum() < 80:
        return np.clip(rgb, 0, 255)
    h, w = luma.shape
    block = 32
    gh, gw = max(1, h // block), max(1, w // block)
    field = np.full((gh, gw), np.nan, np.float32)
    for iy in range(gh):
        y0, y1 = iy * block, min(h, (iy + 1) * block)
        for ix in range(gw):
            x0, x1 = ix * block, min(w, (ix + 1) * block)
            vals = luma[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
            if vals.size >= 8:
                field[iy, ix] = np.percentile(vals, 90)
    global_field = float(np.percentile(luma[mask], 90))
    field = np.where(np.isnan(field), global_field, field)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), block * 1.2)
    field = np.maximum(field, 25.0)
    return np.clip(rgb * (target / field[:, :, None]), 0, 255)


def wrap_deg(value):
    return (value + 180.0) % 360.0 - 180.0


def source_weights(theta_deg):
    """Feathered angular weights for the front photo and six-view panels."""
    t = theta_deg % 360.0

    def ramp(x, a, b):
        if b <= a:
            return 0.0
        return float(np.clip((x - a) / (b - a), 0.0, 1.0))

    front_dist = min(abs(wrap_deg(theta_deg)), 90.0)
    front = 1.0 - ramp(front_dist, 80.0, 88.0)
    # The six-view left/right panels are not a consistent turntable; each is a
    # standalone 3/4 render that repeats the front logo/barcode.  Using them in
    # the wrap duplicated those marks, so they are not used for the texture.
    # The unobserved side gaps are filled by a plain, dirt-textured white label.
    right = 0.0
    back = ramp(t, 116.0, 120.0) * (1.0 - ramp(t, 240.0, 244.0))
    left = 0.0
    return {'front': front, 'right': right, 'back': back, 'left': left,
            'fill': 0.0}


def warp_cylindrical(src, theta_deg, v, *, center_x, half_top, half_bottom,
                     theta_center=0.0, image_w=None, image_h=None):
    """Sample a photograph/panel with x = cx + R sin(theta-theta_center)."""
    src = np.asarray(src)
    h, w = src.shape[:2]
    if image_w is None:
        image_w = 1
    if image_h is None:
        image_h = 1
    delta = np.deg2rad(theta_deg - theta_center)
    half = half_bottom + (half_top - half_bottom) * v
    map_x = (center_x + half * np.sin(delta)).astype(np.float32)
    map_y = ((1.0 - v) * (h - 1)).astype(np.float32)
    map_x = np.broadcast_to(map_x, (image_h, image_w)).astype(np.float32)
    map_y = np.broadcast_to(map_y, (image_h, image_w)).astype(np.float32)
    return cv2.remap(src, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT)


def make_noise(h, w, scale, seed):
    rng = np.random.default_rng(seed)
    low = rng.random((max(4, h // scale), max(4, w // scale))).astype(np.float32)
    noise = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    return gaussian_filter(noise, sigma=max(0.6, scale / 8.0))


def add_scratches(image, rng, count, length_range=(12, 90), alpha=(8, 28)):
    """Draw faint light/dark handling scratches into an RGB float image."""
    h, w = image.shape[:2]
    overlay = np.zeros_like(image)
    mask = np.zeros((h, w), np.float32)
    for _ in range(count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(*length_range))
        angle = rng.uniform(-0.25, 0.25) + (math.pi / 2 if rng.random() > 0.72 else 0.0)
        x2 = int(np.clip(x + length * math.cos(angle), 0, w - 1))
        y2 = int(np.clip(y + length * math.sin(angle), 0, h - 1))
        color = (232, 228, 220) if rng.random() > 0.42 else (86, 78, 66)
        cv2.line(overlay, (x, y), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.line(mask, (x, y), (x2, y2), 1.0, 1, cv2.LINE_AA)
    a = (mask * rng.uniform(*alpha) / 255.0)[:, :, None]
    return np.clip(image * (1.0 - a) + overlay * a, 0, 255)


def _crop_panel(reference, cfg):
    ox, oy = cfg['origin']
    x0, y0, x1, y1 = cfg['body_rect']
    return reference.crop((int(ox + x0), int(oy + y0),
                           int(ox + x1), int(oy + y1)))


def make_body_textures(clean_path, rust_path, reference_path, out, config):
    sampling = config['texture_sampling']
    photo = sampling['original_photo']
    panels_cfg = sampling['panels']
    clean = Image.open(clean_path).convert('RGB')
    rust = Image.open(rust_path).convert('RGB')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    reference = Image.open(reference_path).convert('RGB')

    x0, y0, x1, y1 = photo['body_rect']
    orig_clean = np.asarray(clean.crop((x0, y0, x1, y1))).astype(np.float32)
    orig_rust = np.asarray(rust.crop((x0, y0, x1, y1))).astype(np.float32)
    # The item's surface evidence is the rust derivative; keep a little of the
    # clean image so the print stays legible, then flatten the studio shading.
    orig = flatten_lighting(0.72 * orig_rust + 0.28 * orig_clean, 242.0, 'white')

    sources = {
        'front': {
            'image': orig,
            'center_x': photo['body_center_x'] - x0,
            'half_top': photo['body_half_width_top'],
            'half_bottom': photo['body_half_width_bottom'],
            'theta_center': 0.0,
        },
        'right': {
            'image': flatten_lighting(np.asarray(_crop_panel(reference, panels_cfg['right'])).astype(np.float32), 242.0, 'white'),
            'center_x': panels_cfg['right']['center_x'] - panels_cfg['right']['body_rect'][0],
            'half_top': panels_cfg['right']['half_width'],
            'half_bottom': panels_cfg['right']['half_width'],
            'theta_center': 90.0,
        },
        'back': {
            'image': flatten_lighting(np.asarray(_crop_panel(reference, panels_cfg['back'])).astype(np.float32), 242.0, 'white'),
            'center_x': panels_cfg['back']['center_x'] - panels_cfg['back']['body_rect'][0],
            'half_top': panels_cfg['back']['half_width'],
            'half_bottom': panels_cfg['back']['half_width'],
            'theta_center': 180.0,
        },
        'left': {
            'image': flatten_lighting(np.asarray(_crop_panel(reference, panels_cfg['left'])).astype(np.float32), 242.0, 'white'),
            'center_x': panels_cfg['left']['center_x'] - panels_cfg['left']['body_rect'][0],
            'half_top': panels_cfg['left']['half_width'],
            'half_bottom': panels_cfg['left']['half_width'],
            'theta_center': -90.0,
        },
    }
    fill = np.full((256, 256, 3), 242.0, np.float32)
    fill += (make_noise(256, 256, 8, 36046)[:, :, None] - 0.5) * 2.0
    sources['fill'] = {
        'image': fill, 'center_x': 128.0,
        'half_top': 128.0, 'half_bottom': 128.0, 'theta_center': 0.0,
    }

    size = int(sampling['body_texture_size'])
    u = (np.arange(size) + 0.5) / size
    v_row = 1.0 - (np.arange(size) + 0.5) / size
    theta_deg_1d = (u - sampling['front_u_center']) * 360.0
    theta_2d = np.broadcast_to(theta_deg_1d[None, :], (size, size))
    v_2d = np.broadcast_to(v_row[:, None], (size, size))

    weight_map = {}
    for key in ('front', 'right', 'back', 'left'):
        vals = np.array([source_weights(float(t))[key] for t in theta_deg_1d], np.float32)
        weight_map[key] = np.broadcast_to(vals[None, :], (size, size))
    fill_vals = np.clip(1.0 - sum(weight_map.values()), 0.0, 1.0)
    weight_map['fill'] = np.broadcast_to(fill_vals, (size, size)).astype(np.float32)
    weight_sum = sum(weight_map.values())
    weight_sum = np.maximum(weight_sum, 1e-6)

    rgb = np.zeros((size, size, 3), np.float32)
    for key, src in sources.items():
        sample = warp_cylindrical(
            src['image'], theta_2d, v_2d,
            center_x=src['center_x'], half_top=src['half_top'],
            half_bottom=src['half_bottom'], theta_center=src['theta_center'],
            image_w=size, image_h=size)
        rgb += sample * weight_map[key][:, :, None]
    rgb /= weight_sum[:, :, None]

    # Dirt -------------------------------------------------------------
    rng = np.random.default_rng(36036)
    coarse = make_noise(size, size, 64, 36001)
    fine = make_noise(size, size, 2, 36002)
    speck = make_noise(size, size, 1, 36003)
    hidden = 1.0 - np.clip(weight_map['front'] / weight_sum, 0.0, 1.0)
    stain = np.clip((coarse - 0.50) * 2.2, 0.0, 1.0) * \
        np.clip((fine - 0.32) * 1.7, 0.0, 1.0)
    stain *= 0.55 + 0.45 * hidden
    stain *= 0.70 + 0.45 * (1.0 - v_2d)
    stain = np.clip(gaussian_filter(stain, 1.2), 0.0, 1.0)
    opacity = np.clip(0.05 + 0.16 * stain + 0.06 * hidden, 0.0, 0.26)
    brown = np.array([112, 86, 62], np.float32)
    rgb = rgb * (1.0 - opacity[:, :, None]) + brown[None, None, :] * opacity[:, :, None]
    rust_mask = (speck > 0.992) & (fine > 0.45)
    rgb[rust_mask] = rgb[rust_mask] * 0.55 + np.array([124, 72, 42], np.float32) * 0.45
    dust_mask = (speck > 0.996) & (fine > 0.60)
    rgb[dust_mask] = rgb[dust_mask] * 0.80 + np.array([226, 220, 208], np.float32) * 0.20
    rgb = add_scratches(rgb, rng, 42, (14, 110), (5, 18))

    # Label wrap seam (inferred, hidden left/back transition) and edge rolloff.
    seam_col = int(((0.25 - 95.0 / 360.0) % 1.0) * size)
    for dx, color, alpha in ((-1, np.array([70, 66, 60], np.float32), 0.10),
                             (0, np.array([235, 233, 226], np.float32), 0.06),
                             (1, np.array([70, 66, 60], np.float32), 0.08)):
        x = (seam_col + dx) % size
        rgb[:, x, :] = rgb[:, x, :] * (1 - alpha) + color[None, :] * alpha
    bottom_edge = np.clip((v_2d - 0.012) / 0.012, 0.0, 1.0)
    top_edge = np.clip((0.988 - v_2d) / 0.012, 0.0, 1.0)
    rgb *= (0.96 + 0.04 * bottom_edge)[:, :, None] * (0.94 + 0.06 * top_edge)[:, :, None]
    rgb = np.clip(rgb, 0, 255)
    rgb_image = Image.fromarray(rgb.astype(np.uint8))

    luma = np.asarray(rgb_image).astype(np.float32).mean(axis=2)
    high = luma - gaussian_filter(luma, 2.5)
    height = np.clip(0.34 * stain + 0.14 * high / 30.0 + 0.06 * (fine - 0.5), 0.0, 1.2)
    normal = height_to_normal(height, 0.22)
    roughness = np.clip(0.408 + 0.130 * stain + 0.025 * (fine - 0.5)
                        + 0.015 * hidden, 0.30, 0.64)
    metallic = np.zeros_like(roughness)
    mr = orm_image(roughness, metallic)

    paths = {}
    for name, array in (('body_base', np.asarray(rgb_image)),
                        ('body_mr', mr), ('body_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    arr = np.asarray(rgb_image).astype(np.float32)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'min_luma': round(float(luma.min()), 2),
        'p95_luma': round(float(np.percentile(luma, 95)), 2),
        'dark_fraction_below_20': round(float((luma < 20).mean()), 4),
        'mean_chroma': round(float((arr.max(axis=2) - arr.min(axis=2)).mean()), 2),
        'mean_metallic': round(float(metallic.mean()), 4),
        'mean_roughness': round(float(roughness.mean()), 4),
        'min_roughness': round(float(roughness.min()), 4),
        'max_roughness': round(float(roughness.max()), 4),
        'front_mean_luma': round(float(luma[:, int(0.15 * size):int(0.35 * size)].mean()), 2),
        'back_mean_luma': round(float(luma[:, int(0.65 * size):int(0.85 * size)].mean()), 2),
        'note': ('Front is a flat-lit blend of the rust derivative (72%) and clean '
                 'photograph (28%), mapped over +/-85 deg; the back comes from the '
                 'uncalibrated six-view back panel. The left/right panels are not a '
                 'consistent turntable, so the unobserved side gaps use a plain '
                 'dirty-white label instead of duplicating their front logo/barcode. '
                 'Photographed broad shading was removed; print, dirt and scratches '
                 'remain. Hidden-half print and seam are inferred.'),
    }
    return paths, stats, rgb_image


def make_cap_textures(clean_path, rust_path, out, config):
    sampling = config['texture_sampling']
    photo = sampling['original_photo']
    clean = Image.open(clean_path).convert('RGB')
    rust = Image.open(rust_path).convert('RGB')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)
    x0, y0, x1, y1 = photo['cap_rect']
    cap_clean = np.asarray(clean.crop((x0, y0, x1, y1))).astype(np.float32)
    cap_rust = np.asarray(rust.crop((x0, y0, x1, y1))).astype(np.float32)
    cap = 0.72 * cap_rust + 0.28 * cap_clean
    cap_flat = flatten_lighting(cap, 145.0, 'grey')
    cap_luma = cap_flat.mean(axis=2)
    detail = cap_luma - gaussian_filter(cap_luma, 12.0)
    detail = detail / max(8.0, float(np.percentile(np.abs(detail), 98)))
    detail = np.clip(detail, -1.0, 1.0)

    tile = int(sampling['cap_texture_size'])
    rng = np.random.default_rng(36037)
    u = (np.arange(tile) + 0.5) / tile
    v_row = 1.0 - (np.arange(tile) + 0.5) / tile
    theta = (u - sampling['front_u_center']) * 360.0
    theta_2d = np.broadcast_to(theta[None, :], (tile, tile))
    v_2d = np.broadcast_to(v_row[:, None], (tile, tile))
    front_w = np.array([source_weights(float(t))['front'] for t in theta], np.float32)
    front_w = np.broadcast_to(front_w[None, :], (tile, tile))
    cap_detail = warp_cylindrical(
        detail.astype(np.float32), theta_2d, v_2d,
        center_x=photo['cap_center_x'] - x0,
        half_top=photo['cap_half_width_top'],
        half_bottom=photo['cap_half_width_bottom'],
        theta_center=0.0, image_w=tile, image_h=tile)

    flow = gaussian_filter(rng.normal(size=tile).astype(np.float32), 1.8)
    flow = flow / max(1e-6, float(flow.std()))
    flow_2d = np.broadcast_to(flow[None, :], (tile, tile))
    fine = make_noise(tile, tile, 2, 36038)
    coarse = make_noise(tile, tile, 48, 36039)
    grime = np.clip((coarse - 0.55) * 2.0, 0.0, 1.0) * \
        np.clip((fine - 0.35) * 1.6, 0.0, 1.0)
    grime = gaussian_filter(grime, 1.0) * (0.35 + 0.65 * (1.0 - front_w))
    grey = np.array([142.0, 145.0, 150.0], np.float32)
    side = np.empty((tile, tile, 3), np.float32)
    side[:] = grey
    side += (cap_detail * 7.0 + flow_2d * 2.6 + (fine - 0.5) * 2.0)[:, :, None]
    side = side * (1.0 - 0.10 * grime[:, :, None]) + \
        np.array([104, 92, 78], np.float32)[None, None, :] * (0.10 * grime[:, :, None])
    side = add_scratches(side, rng, 18, (10, 70), (6, 20))
    side = np.clip(side, 0, 255)

    # Cap top tile -----------------------------------------------------
    yy, xx = np.mgrid[0:tile, 0:tile].astype(np.float32)
    xt = (xx + 0.5) / tile * 2.0 - 1.0
    yt = (yy + 0.5) / tile * 2.0 - 1.0
    rr = np.sqrt(xt * xt + yt * yt)
    ang = np.arctan2(yt, xt)
    radial = gaussian_filter(rng.normal(size=tile).astype(np.float32), 1.6)
    radial = np.broadcast_to(radial[None, :], (tile, tile))
    top = np.empty((tile, tile, 3), np.float32)
    top[:] = grey
    top += (radial * 2.4 + (fine - 0.5) * 2.2)[:, :, None]
    edge_dark = np.exp(-((rr - 0.805) / 0.008) ** 2)
    edge_light = np.exp(-((rr - 0.866) / 0.010) ** 2)
    dimple_ring = np.exp(-((rr - 0.030) / 0.006) ** 2)
    dimple_spot = np.exp(-(rr / 0.014) ** 2)
    top -= edge_dark[:, :, None] * np.array([16, 15, 14], np.float32)
    top += edge_light[:, :, None] * np.array([8, 8, 8], np.float32)
    top -= dimple_ring[:, :, None] * np.array([22, 22, 23], np.float32)
    top -= dimple_spot[:, :, None] * np.array([10, 10, 11], np.float32)
    top += np.clip((0.11 - rr) / 0.11, 0, 1)[:, :, None] * 3.0
    top_grime = gaussian_filter(make_noise(tile, tile, 40, 36040), 2.0)
    top_grime = np.clip((top_grime - 0.52) * 2.2, 0, 1) * np.clip((fine - 0.30) * 1.5, 0, 1)
    top = top * (1.0 - 0.12 * top_grime[:, :, None]) + \
        np.array([104, 92, 78], np.float32)[None, None, :] * (0.12 * top_grime[:, :, None])
    top = add_scratches(top, rng, 20, (8, 55), (6, 18))
    top = np.clip(top, 0, 255)

    atlas_base = np.zeros((tile, tile * 2, 3), np.uint8)
    atlas_base[:, :tile] = side.astype(np.uint8)
    atlas_base[:, tile:] = top.astype(np.uint8)

    side_height = np.clip(0.55 * cap_detail + 0.20 * flow_2d + 0.10 * (fine - 0.5), 0, 1.4)
    side_normal = height_to_normal(side_height, 0.26)
    top_height = np.clip(0.6 * edge_dark + 0.5 * dimple_ring + 0.4 * dimple_spot
                         + 0.15 * radial + 0.10 * (fine - 0.5), 0, 1.4)
    top_normal = height_to_normal(top_height, 0.30)
    atlas_normal = np.zeros((tile, tile * 2, 3), np.uint8)
    atlas_normal[:, :tile] = side_normal
    atlas_normal[:, tile:] = top_normal

    side_rough = np.clip(0.382 + 0.030 * flow_2d + 0.070 * grime
                         + 0.018 * cap_detail, 0.30, 0.58)
    top_rough = np.clip(0.392 + 0.030 * radial + 0.060 * top_grime
                        + 0.030 * edge_dark, 0.30, 0.60)
    atlas_mr = np.zeros((tile, tile * 2, 3), np.uint8)
    atlas_mr[:, :tile] = orm_image(side_rough, np.zeros_like(side_rough))
    atlas_mr[:, tile:] = orm_image(top_rough, np.zeros_like(top_rough))

    paths = {}
    for name, array in (('cap_base', atlas_base), ('cap_mr', atlas_mr),
                        ('cap_normal', atlas_normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    luma = atlas_base.astype(np.float32).mean(axis=2)
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'side_mean_luma': round(float(luma[:, :tile].mean()), 2),
        'top_mean_luma': round(float(luma[:, tile:].mean()), 2),
        'mean_chroma': round(float((atlas_base.max(axis=2) - atlas_base.min(axis=2)).mean()), 2),
        'mean_metallic': 0.0,
        'mean_roughness': round(float((atlas_mr[:, :, 1].astype(np.float32) / 255).mean()), 4),
        'min_roughness': round(float((atlas_mr[:, :, 1].astype(np.float32) / 255).min()), 4),
        'max_roughness': round(float((atlas_mr[:, :, 1].astype(np.float32) / 255).max()), 4),
        'note': ('Grey moulded plastic, dielectric. Broad photographed shading was '
                 'flattened; flow lines, orange peel, a shallow top ring, a central '
                 'sprue dimple, scratches and faint grime are synthesised/inferred.'),
    }
    return paths, stats


def make_base_textures(clean_path, rust_path, out, config):
    sampling = config['texture_sampling']
    tile = int(sampling['base_texture_size'])
    rng = np.random.default_rng(36041)
    clean = Image.open(clean_path).convert('RGB')
    rust = Image.open(rust_path).convert('RGB')
    if rust.size != clean.size:
        rust = rust.resize(clean.size, Image.Resampling.LANCZOS)

    # Side/rim tile: brushed rolled aluminium.
    u = (np.arange(tile) + 0.5) / tile
    v = (np.arange(tile) + 0.5) / tile
    brush_v = gaussian_filter(rng.normal(size=tile).astype(np.float32), 1.2)
    brush_v = np.broadcast_to(brush_v[:, None], (tile, tile))
    fine = make_noise(tile, tile, 2, 36042)
    coarse = make_noise(tile, tile, 48, 36043)
    stain = np.clip((coarse - 0.56) * 2.4, 0, 1) * np.clip((fine - 0.45) * 1.6, 0, 1)
    metal = np.empty((tile, tile, 3), np.float32)
    metal[:] = np.array([176.0, 180.0, 186.0], np.float32)
    metal += (brush_v * 3.2 + (fine - 0.5) * 2.4)[:, :, None]
    metal = metal * (1.0 - 0.12 * stain[:, :, None]) + \
        np.array([118, 98, 78], np.float32)[None, None, :] * (0.12 * stain[:, :, None])
    metal = add_scratches(metal, rng, 24, (10, 80), (8, 24))
    metal = np.clip(metal, 0, 255)
    metal_rough = np.clip(0.318 + 0.032 * brush_v + 0.070 * stain
                          + 0.015 * (fine - 0.5), 0.24, 0.50)
    metal_normal = height_to_normal(0.55 * brush_v + 0.16 * (fine - 0.5), 0.22)

    # Bottom tile: stamped dish with ring geometry and an inferred ink/stamp code.
    yy, xx = np.mgrid[0:tile, 0:tile].astype(np.float32)
    xt = (xx + 0.5) / tile * 2.0 - 1.0
    yt = (yy + 0.5) / tile * 2.0 - 1.0
    rr = np.sqrt(xt * xt + yt * yt)
    ang = np.arctan2(yt, xt)
    radial = gaussian_filter(rng.normal(size=tile).astype(np.float32), 1.5)
    radial_2d = np.broadcast_to(radial[None, :], (tile, tile))
    fine2 = make_noise(tile, tile, 2, 36044)
    bottom = np.empty((tile, tile, 3), np.float32)
    bottom[:] = np.array([168.0, 172.0, 178.0], np.float32)
    band_outer = rr > 0.90
    groove = (rr > 0.72) & (rr <= 0.90)
    ring = (rr > 0.62) & (rr <= 0.72)
    bottom[band_outer] += 12.0
    bottom[groove] -= 20.0
    bottom[ring] += 22.0
    bottom += (radial_2d * 2.4 + (fine2 - 0.5) * 3.0)[:, :, None]
    dark_edge = np.exp(-((rr - 0.985) / 0.006) ** 2)
    bottom -= dark_edge[:, :, None] * 40.0
    bottom_stain = gaussian_filter(make_noise(tile, tile, 36, 36045), 2.0)
    bottom_stain = np.clip((bottom_stain - 0.55) * 2.2, 0, 1) * \
        np.clip((fine2 - 0.36) * 1.5, 0, 1)
    bottom = bottom * (1.0 - 0.13 * bottom_stain[:, :, None]) + \
        np.array([116, 94, 72], np.float32)[None, None, :] * (0.13 * bottom_stain[:, :, None])
    bottom = np.clip(bottom, 0, 255)

    # Draw the batch/expiry stamp as an inferred alphanumeric marking.
    stamp_image = Image.fromarray(bottom.astype(np.uint8))
    draw = ImageDraw.Draw(stamp_image)
    f1 = font(40, mono=True)
    f2 = font(36, mono=True)
    line1 = 'B250401'
    line2 = 'EXP 2028/04/01'
    b1 = draw.textbbox((0, 0), line1, font=f1)
    b2 = draw.textbbox((0, 0), line2, font=f2)
    cx = tile // 2
    y1 = int(tile * 0.455)
    y2 = int(tile * 0.535)
    tx1 = cx - (b1[2] - b1[0]) // 2
    tx2 = cx - (b2[2] - b2[0]) // 2
    draw.text((tx1 + 1, y1 + 1), line1, font=f1, fill=(196, 196, 190))
    draw.text((tx2 + 1, y2 + 1), line2, font=f2, fill=(196, 196, 190))
    draw.text((tx1, y1), line1, font=f1, fill=(55, 55, 58))
    draw.text((tx2, y2), line2, font=f2, fill=(55, 55, 58))
    bottom = np.asarray(stamp_image).astype(np.float32)

    # A stamp height mask for the normal map.
    stamp_mask = np.zeros((tile, tile), np.float32)
    mimg = Image.new('L', (tile, tile), 0)
    mdraw = ImageDraw.Draw(mimg)
    mdraw.text((tx1, y1), line1, font=f1, fill=255)
    mdraw.text((tx2, y2), line2, font=f2, fill=255)
    stamp_mask = np.asarray(mimg).astype(np.float32) / 255.0

    bottom_rough = np.clip(0.286 + 0.105 * groove + 0.028 * radial_2d
                           + 0.050 * bottom_stain + 0.050 * stamp_mask, 0.22, 0.54)
    bottom_height = np.clip(0.35 * dark_edge + 0.24 * (ring.astype(np.float32))
                            + 0.22 * (groove.astype(np.float32))
                            + 0.20 * radial_2d + 0.10 * (fine2 - 0.5)
                            + 0.16 * stamp_mask, 0, 1.5)
    bottom_normal = height_to_normal(bottom_height, 0.30)

    atlas_base = np.zeros((tile, tile * 2, 3), np.uint8)
    atlas_base[:, :tile] = metal.astype(np.uint8)
    atlas_base[:, tile:] = bottom.astype(np.uint8)
    atlas_normal = np.zeros((tile, tile * 2, 3), np.uint8)
    atlas_normal[:, :tile] = metal_normal
    atlas_normal[:, tile:] = bottom_normal
    atlas_mr = np.zeros((tile, tile * 2, 3), np.uint8)
    atlas_mr[:, :tile] = orm_image(metal_rough, np.ones_like(metal_rough))
    atlas_mr[:, tile:] = orm_image(bottom_rough, np.ones_like(bottom_rough))

    paths = {}
    for name, array in (('base_base', atlas_base), ('base_mr', atlas_mr),
                        ('base_normal', atlas_normal)):
        path = out / (name + '.png')
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    luma = atlas_base.astype(np.float32).mean(axis=2)
    stamp_luma = luma[:, tile:]
    stats = {
        'mean_luma': round(float(luma.mean()), 2),
        'side_mean_luma': round(float(luma[:, :tile].mean()), 2),
        'bottom_mean_luma': round(float(stamp_luma.mean()), 2),
        'mean_chroma': round(float((atlas_base.max(axis=2) - atlas_base.min(axis=2)).mean()), 2),
        'mean_metallic': round(float((atlas_mr[:, :, 2].astype(np.float32) / 255).mean()), 4),
        'min_metallic': round(float((atlas_mr[:, :, 2].astype(np.float32) / 255).min()), 4),
        'mean_roughness': round(float((atlas_mr[:, :, 1].astype(np.float32) / 255).mean()), 4),
        'min_roughness': round(float((atlas_mr[:, :, 1].astype(np.float32) / 255).min()), 4),
        'max_roughness': round(float((atlas_mr[:, :, 1].astype(np.float32) / 255).max()), 4),
        'stamp_dark_fraction': round(float((stamp_luma < 90).mean()), 4),
        'note': ('Stamped/rolled aluminium, metallic 1 everywhere. The batch/expiry '
                 'code and exact dish profile are copied from the uncalibrated reference '
                 'bottom panel and are explicitly inferred. No photographed specular '
                 'field is baked into the base colour.'),
    }
    return paths, stats


def texture_checks(body_stats, cap_stats, base_stats):
    body_ok = (100.0 <= body_stats['mean_luma'] <= 240.0 and
               body_stats['mean_metallic'] <= 0.02 and
               0.30 <= body_stats['mean_roughness'] <= 0.65 and
               body_stats['dark_fraction_below_20'] <= 0.20)
    cap_ok = (90.0 <= cap_stats['mean_luma'] <= 190.0 and
              cap_stats['mean_chroma'] <= 16.0 and
              cap_stats['mean_metallic'] <= 0.02 and
              0.28 <= cap_stats['mean_roughness'] <= 0.60)
    base_ok = (base_stats['mean_metallic'] >= 0.98 and
               0.20 <= base_stats['mean_roughness'] <= 0.56 and
               100.0 <= base_stats['mean_luma'] <= 230.0)
    return {
        'body': {'passed': bool(body_ok), **body_stats},
        'cap': {'passed': bool(cap_ok), **cap_stats},
        'base': {'passed': bool(base_ok), **base_stats},
        'note': ('The printed label and grey cap are dielectric (metallic 0); the '
                 'rolled/stamped aluminium is metallic 1. The checks reject black, '
                 'over-smooth or wrongly metallic atlases. They do not prove the '
                 'hidden surface appearance.'),
    }


# ---------------------------------------------------------------------------
# Mesh / GLB / ray checks
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
        'note': ('The GLB embeds nine PNG images and separate base-colour, '
                 'metallic-roughness and normal atlases for body, cap and base. '
                 'The valve uses scalar PBR factors. This audit does not prove '
                 'the visual appearance in every viewer.'),
    }


def ray_hits(mesh, origin, direction, multiple=False):
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


def ray_checks(path):
    cap = load_named_parts(path).get('Cap shell')
    base = load_named_parts(path).get('Can base')
    body = load_named_parts(path).get('Can body')
    valve = load_named_parts(path).get('Valve stem')
    out = {}
    centre = ray_hits(cap, (0.014, 0.0, 0.14), (0, 0, 1))
    outside = ray_hits(cap, (0.0345, 0.0, 0.14), (0, 0, 1))
    rim = ray_hits(cap, (0.0326, 0.0, 0.14), (0, 0, 1))
    top = ray_hits(cap, (0.014, 0.0, 0.21), (0, 0, -1))
    centre_z = float(centre[0][2]) if len(centre) else None
    rim_z = float(rim[0][2]) if len(rim) else None
    top_z = float(top[0][2]) if len(top) else None
    out['cap_cavity'] = {
        'centre_hit_z_m': round(centre_z, 6) if centre_z is not None else None,
        'bottom_rim_hit_z_m': round(rim_z, 6) if rim_z is not None else None,
        'top_hit_z_m': round(top_z, 6) if top_z is not None else None,
        'cavity_depth_m': round(centre_z - rim_z, 6) if None not in (centre_z, rim_z) else None,
        'top_thickness_m': round(top_z - centre_z, 6) if None not in (top_z, centre_z) else None,
        'outside_ray_hits': int(len(outside)),
        'passed': bool(centre_z is not None and rim_z is not None and top_z is not None
                       and centre_z - rim_z > 0.020
                       and top_z - centre_z > 0.0008
                       and len(outside) == 0),
        'note': ('A ray up inside the cap reaches the cavity ceiling about 47 mm '
                 'above the bottom rim, while a ray outside the outer radius misses. '
                 'This proves a real downward-open cavity; wall thickness and snap fit '
                 'remain inferred.'),
    }
    dish_centre = ray_hits(base, (0.0, 0.0, -0.02), (0, 0, 1))
    dish_rim = ray_hits(base, (0.0305, 0.0, -0.02), (0, 0, 1))
    dish_ring = ray_hits(base, (0.0225, 0.0, -0.02), (0, 0, 1))
    dc = float(dish_centre[0][2]) if len(dish_centre) else None
    dr = float(dish_rim[0][2]) if len(dish_rim) else None
    dg = float(dish_ring[0][2]) if len(dish_ring) else None
    out['bottom_dish'] = {
        'centre_z_m': round(dc, 6) if dc is not None else None,
        'outer_rim_z_m': round(dr, 6) if dr is not None else None,
        'raised_ring_z_m': round(dg, 6) if dg is not None else None,
        'centre_minus_rim_m': round(dc - dr, 6) if None not in (dc, dr) else None,
        'passed': bool(dc is not None and dr is not None and dc - dr > 0.001),
        'note': ('A ray up at the centre hits higher than a ray up on the ground rim, '
                 'confirming a concave stamped bottom. The batch stamp is texture-only '
                 'and its exact profile is inferred.'),
    }
    valve_hit = ray_hits(valve, (0.0, 0.0, 0.25), (0, 0, -1))
    body_hit = ray_hits(body, (0.0, 0.0, 0.25), (0, 0, -1))
    vtop = float(valve_hit[0][2]) if len(valve_hit) else None
    btop = float(body_hit[0][2]) if len(body_hit) else None
    out['hidden_valve'] = {
        'valve_top_z_m': round(vtop, 6) if vtop is not None else None,
        'body_cup_top_z_m': round(btop, 6) if btop is not None else None,
        'stem_above_cup_m': round(vtop - btop, 6) if None not in (vtop, btop) else None,
        'passed': bool(vtop is not None and btop is not None and vtop > btop),
        'note': ('The black valve stem exists above the hidden cup under the cap. Its '
                 'dimensions and internal mechanism are inferred; it is not visible in '
                 'the supplied photographs.'),
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


def _rgba(path):
    return np.asarray(Image.open(path).convert('RGBA')).astype(np.float32)


def mean_difference(path_a, path_b):
    a = _rgba(path_a)
    b = _rgba(path_b)
    if a.shape != b.shape:
        raise ValueError('render sizes differ')
    mask = (a[:, :, 3] >= 128) & (b[:, :, 3] >= 128)
    if not mask.any():
        return None
    return round(float(np.abs(a[:, :, :3] - b[:, :, :3])[mask].mean()), 4)


def highlight_centroid_shift(path_a, path_b):
    a = _rgba(path_a)
    b = _rgba(path_b)
    if a.shape != b.shape:
        raise ValueError('render sizes differ')
    mask = (a[:, :, 3] >= 128) & (b[:, :, 3] >= 128)
    if mask.sum() < 100:
        return None
    la = a[:, :, :3].mean(axis=2)
    lb = b[:, :, :3].mean(axis=2)
    ta = float(np.percentile(la[mask], 95))
    tb = float(np.percentile(lb[mask], 95))
    ma = mask & (la >= ta)
    mb = mask & (lb >= tb)
    if ma.sum() < 20 or mb.sum() < 20:
        return None
    ya, xa = np.nonzero(ma)
    yb, xb = np.nonzero(mb)
    shift = math.hypot(float(xa.mean() - xb.mean()), float(ya.mean() - yb.mean()))
    return round(shift, 3)


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


# ---------------------------------------------------------------------------
# Evidence boards
# ---------------------------------------------------------------------------
def make_six_board(out, panels):
    title = font(28, True)
    label = font(19, True)
    small = font(15)
    tile = (700, 430)
    board = Image.new('RGB', (3 * tile[0] + 40, 6 * (tile[1] + 48) + 90), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 14), '36 自喷漆罐 · 六方向：未标定视觉参考 / 原模型 / 新候选（同一中性光）',
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
        draw.text((14 + tile[0], y), '原模型（中性光，降曝光）', font=label, fill='#35454c')
        draw.text((14 + 2 * tile[0], y), '新候选', font=label, fill='#35454c')
        board.paste(fit_tile(ref, tile), (14, y + 26))
        board.paste(fit_tile(old, tile), (14 + tile[0], y + 26))
        board.paste(fit_tile(new, tile), (14 + 2 * tile[0], y + 26))
        note = (f'诊断 IoU：参考/新 {results[view]["reference_vs_new_mask_iou"]} · '
                f'参考/旧 {results[view]["reference_vs_old_mask_iou"]} · '
                f'旧/新 {results[view]["old_vs_new_mask_iou"]}（未标定，不作验收）')
        draw.text((14, y + tile[1] + 30), note, font=small, fill='#5a6b72')
    draw.text((14, board.height - 30),
              '注：参考面板是白标签/白背景，自动阈值掩码主要保留深色像素，正/背/侧 IoU 会明显偏低；'
              '原照 alpha 的正视诊断 IoU（见 result.json）更可靠。旧模型列仅降低曝光 1 档以显示细节，灯光方向不变。',
              font=small, fill='#5a6b72')
    board.save(out / '六视图-新旧对照.jpg', quality=93)
    return results


def make_detail_board(out):
    names = ('cap-top', 'cap-side', 'cap-seam', 'cap-dimples', 'front-label',
             'back-label', 'body', 'body-surface', 'bottom', 'bottom-dish',
             'bottom-stamp', 'metal-rim', 'valve', 'surface', 'new-source')
    titles = {
        'cap-top': '盖顶平面/外缘', 'cap-side': '灰塑料盖侧壁',
        'cap-seam': '盖/身接缝', 'cap-dimples': '盖顶中央浇口凹陷',
        'front-label': '正面标签（原照）', 'back-label': '背面标签（隐藏半推断）',
        'body': '筒身圆柱', 'body-surface': '标签表面细节',
        'bottom': '金属底', 'bottom-dish': '底部凹碟',
        'bottom-stamp': '底部批号/有效期（推断）', 'metal-rim': '卷边金属带',
        'valve': '隐藏阀杆（取下盖）', 'surface': '表面微纹理',
        'new-source': '原视角诊断渲染',
    }
    tile = (620, 470)
    cols = 4
    rows = math.ceil(len(names) / cols)
    board = Image.new('RGB', (cols * tile[0] + 30, rows * (tile[1] + 44) + 70),
                      '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '36 自喷漆罐 · 关键局部检查（新版，中性光）',
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
    items = (('cap-only', '盖子（隔离）'),
             ('cap-only-top', '盖顶（隔离顶视）'),
             ('cap-only-bottom', '盖子底部开口（正下）'),
             ('cap-only-bottom-3q', '盖子空腔（3/4 底面）'),
             ('body-only', '筒身（隔离）'),
             ('base-only', '金属底（隔离）'),
             ('base-only-bottom', '金属底凹碟（隔离底视）'),
             ('valve-only', '隐藏阀杆（隔离）'),
             ('new-top', '新版顶视轮廓（圆度）'))
    tile = (620, 500)
    board = Image.new('RGB', (3 * tile[0] + 30, 3 * (tile[1] + 50) + 210), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '36 自喷漆罐 · 隔离结构与开口检查（渲染 + 射线）',
              font=font(26, True), fill='#17232a')
    for i, (name, title) in enumerate(items):
        x = 10 + (i % 3) * tile[0]
        y = 56 + (i // 3) * (tile[1] + 50)
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile,
                             '#ffffff'), (x, y))
        draw.text((x + 4, y + tile[1] + 8), title, font=font(18, True),
                  fill='#34444b')
    y = 56 + 3 * (tile[1] + 50) + 4
    draw.text((18, y), '隔离开口射线检查', font=font(22, True), fill='#17232a')
    lines = [
        f"盖空腔：中心射线 z={ray['cap_cavity']['centre_hit_z_m']} m，底缘 z="
        f"{ray['cap_cavity']['bottom_rim_hit_z_m']} m，空腔深度="
        f"{ray['cap_cavity']['cavity_depth_m']} m；顶壁厚="
        f"{ray['cap_cavity']['top_thickness_m']} m；外侧射线命中 "
        f"{ray['cap_cavity']['outside_ray_hits']} 次",
        f"底部：中心 z={ray['bottom_dish']['centre_z_m']} m，外缘 z="
        f"{ray['bottom_dish']['outer_rim_z_m']} m，凸起环 z="
        f"{ray['bottom_dish']['raised_ring_z_m']} m",
        f"隐藏阀杆：顶 z={ray['hidden_valve']['valve_top_z_m']} m，杯顶 z="
        f"{ray['hidden_valve']['body_cup_top_z_m']} m",
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
    draw.text((18, 12), '36 自喷漆罐 · 同光照材质对照（左：原模型，右：新候选）',
              font=font(26, True), fill='#17232a')
    for i, name in enumerate(('old-material-close', 'new-material-close')):
        board.paste(fit_tile(Image.open(out / 'renders' / (name + '.png')), tile,
                             '#ffffff'), (10 + i * tile[0], 60))
        draw.text((16 + i * tile[0], tile[1] + 70),
                  '原模型（TripoSR，保留未改；降曝光显示细节）' if i == 0 else '新候选（PBR 贴图）',
                  font=font(19, True), fill='#34444b')
    board.save(out / '材质-同光照对照.jpg', quality=94)

    board = Image.new('RGB', (2 * tile[0] + 30, 2 * tile[1] + 150), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '36 自喷漆罐 · 变光检查（高光必须随现场光移动）',
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
    draw.text((18, 12), '36 自喷漆罐 · 表面反射变光对照（金属底/卷边 / 塑料高光）',
              font=font(25, True), fill='#17232a')
    pairs = (('metal-base-key', '金属底 · 主光 A'),
             ('metal-base-alt', '金属底 · 主光 B'),
             ('metal-rim-key', '金属卷边 · 主光 A'),
             ('metal-rim-alt', '金属卷边 · 主光 B'))
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
    draw.text((18, 10), '36 自喷漆罐 · 新候选 24 方向转台（中性光，仅供检查）',
              font=font(24, True), fill='#17232a')
    for i in range(24):
        x = 8 + (i % 6) * tile[0]
        y = 52 + (i // 6) * (tile[1] + 30)
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png')
        board.paste(fit_tile(image, tile, '#ffffff'), (x, y))
        draw.text((x + 4, y + tile[1] + 4), f'{i * 15:03d}°',
                  font=font(14, True), fill='#52646b')
    board.save(out / '转台-24方向.jpg', quality=93)


def make_source_board(out, source_photo, rust_photo, old_source_view=None):
    tile = (600, 760)
    board = Image.new('RGB', (4 * tile[0] + 30, tile[1] + 110), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '36 自喷漆罐 · 原照 / 做锈图 / 旧模型原视角 / 新候选原视角诊断',
              font=font(24, True), fill='#17232a')
    old_image = None
    if old_source_view is not None and Path(old_source_view).is_file():
        old_image = Image.open(old_source_view)
    images = (Image.open(source_photo), Image.open(rust_photo), old_image,
              Image.open(out / 'renders/new-source.png'))
    titles = ('原照（干净）', '做锈图（表面证据）', '旧模型原视角渲染',
              '新候选原视角诊断（未标定，不作验收）')
    for i, image in enumerate(images):
        if image is None:
            continue
        board.paste(fit_tile(image, tile, '#ffffff'), (10 + i * tile[0], 58))
        draw.text((16 + i * tile[0], tile[1] + 70), titles[i],
                  font=font(18, True), fill='#34444b')
    board.save(out / '源图-原视角诊断.jpg', quality=93)


def make_top_compare(out, new_ratio, old_ratio):
    tile = (720, 620)
    board = Image.new('RGB', (2 * tile[0] + 30, tile[1] + 150), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '36 自喷漆罐 · 顶视圆柱截面诊断（旧模型椭圆/倾斜 vs 新候选圆截面）',
              font=font(24, True), fill='#17232a')
    for i, (name, title) in enumerate((('new-top', '新候选顶视'), ('old-top', '原模型顶视'))):
        image = Image.open(out / 'renders' / (name + '.png'))
        board.paste(fit_tile(image, tile, '#ffffff'), (10 + i * tile[0], 58))
        draw.text((16 + i * tile[0], tile[1] + 68), title,
                  font=font(19, True), fill='#34444b')
    draw.text((18, tile[1] + 106), f'轮廓包围盒宽/高：新 {new_ratio} · 旧 {old_ratio}（旧模型存在椭圆截面与约 11° 轴线倾斜，详见 result.json）',
              font=font(17), fill='#5a6b72')
    board.save(out / '圆柱截面-顶视对照.jpg', quality=93)


def make_atlas_board(out):
    tile = (1000, 500)
    names = ('body_base', 'cap_base', 'base_base', 'body_normal', 'cap_normal', 'base_normal')
    board = Image.new('RGB', (2 * tile[0] + 30, 3 * (tile[1] + 40) + 70), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((18, 12), '36 自喷漆罐 · PBR 图集布局（base colour / normal；MR 记录在文件清单）',
              font=font(24, True), fill='#17232a')
    for i, name in enumerate(names):
        x = 10 + (i % 2) * tile[0]
        y = 56 + (i // 2) * (tile[1] + 40)
        image = Image.open(out / (name + '.png')).convert('RGB')
        board.paste(fit_tile(image, tile, '#ffffff', crop=False), (x, y))
        draw.text((x + 4, y + tile[1] + 6), name, font=font(17, True),
                  fill='#34444b')
    board.save(out / 'PBR图集-布局.jpg', quality=93)


# ---------------------------------------------------------------------------
# Independent review page and browser self-test
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#18242b;--muted:#53656c;--line:#cbd5d7;--paper:#f2f5f5;--card:#fff;--blue:#1757b4;--warn:#8a2b20;--ok:#1d5c3a}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 "Microsoft YaHei",system-ui,sans-serif}
header{background:linear-gradient(120deg,#151b25,#31465d);color:#fff;padding:22px 30px 18px}
header h1{margin:0 0 6px;font-size:26px}header p{margin:0;color:#dbe8f7;max-width:1180px}
.badge{display:inline-block;margin-left:10px;padding:3px 10px;border:1px solid #ffd88a;border-radius:999px;color:#ffe7b0;font-size:12px}
main{display:grid;grid-template-columns:minmax(0,1fr) 410px;gap:18px;padding:18px 22px 40px;max-width:1580px;margin:auto}
section,aside{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:0 2px 8px rgba(20,40,50,.05)}
h2{margin:4px 0 10px;font-size:18px}h3{margin:14px 0 6px;font-size:15px}
.notice{background:#fff6df;border:1px solid #eccf91;color:#5b4316;border-radius:10px;padding:10px 14px;margin:14px 22px 0;max-width:1530px}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin-bottom:12px}
.group{display:flex;gap:6px;flex-wrap:wrap}button{font:inherit;padding:6px 11px;border:1px solid #aebbc0;background:#f8fafb;color:#24353c;border-radius:7px;cursor:pointer}
button.active{background:var(--blue);border-color:var(--blue);color:#fff}button:hover{border-color:var(--blue)}
.viewer figure{margin:0;background:#e3e8e8;border-radius:10px;padding:10px;text-align:center}
#view-img{max-width:100%;height:560px;object-fit:contain;background:#dde3e3;border-radius:6px}
figcaption{color:var(--muted);font-size:13px;margin-top:6px}
#turn-wrap{background:#dde3e3;border-radius:10px;padding:10px;text-align:center;cursor:grab;touch-action:none;user-select:none}
#turn-wrap:active{cursor:grabbing}
#turn-img{max-width:100%;height:520px;object-fit:contain;pointer-events:none}
#turn-range{width:100%;margin-top:10px}
.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:6px}
.gallery figure{margin:0;background:#eef2f2;border:1px solid var(--line);border-radius:8px;padding:6px;text-align:center}
.gallery img{width:100%;height:170px;object-fit:contain}
.gallery figcaption{font-size:12px;min-height:32px}
.boards img{width:100%;border:1px solid var(--line);border-radius:8px;margin:6px 0 14px}
.links a{display:block;color:#155397;padding:2px 0}
.mono{font-family:Consolas,monospace;font-size:12px;word-break:break-all}
.ok{color:var(--ok);font-weight:700}.bad{color:var(--warn);font-weight:700}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #e3e8e9;padding:5px 4px;text-align:left;vertical-align:top;overflow-wrap:anywhere;word-break:break-word}
#selftest{white-space:pre-wrap;background:#111820;color:#cfe6d6;padding:12px 16px;margin:18px 22px;border-radius:8px;font:12px/1.5 Consolas,monospace;max-width:1530px}
@media(max-width:1100px){main{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

JS = r"""
const state={which:'new',mode:'material',view:'front',turn:0};
const $=id=>document.getElementById(id);
const viewNames={front:'正视',back:'背视',left:'左视',right:'右视',top:'顶视',bottom:'底视'};
function imgFor(){
  if(state.mode==='clay')return `renders/clay-${state.view}.png`;
  return `renders/${state.which}-${state.view}.png`;
}
function applyView(){
  const src=imgFor();$('view-img').src=src;$('view-img').alt=src;
  $('view-caption').textContent=`${state.which==='new'?'新候选':state.which==='old'?'原模型':'灰模'} · ${viewNames[state.view]||state.view}`;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===state.view));
  document.querySelectorAll('[data-which]').forEach(b=>b.classList.toggle('active',b.dataset.which===state.which));
  document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode));
}
function applyTurn(){
  const src=`renders/turn-${String(state.turn).padStart(2,'0')}.png`;
  $('turn-img').src=src;$('turn-range').value=state.turn;
  $('turn-caption').textContent=`第 ${state.turn+1}/24 方向（预渲染转台，不是实时 3D）`;
}
document.querySelectorAll('[data-which]').forEach(b=>b.onclick=()=>{state.which=b.dataset.which;state.mode='material';applyView();});
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;applyView();});
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{state.view=b.dataset.view;applyView();});
$('turn-range').oninput=e=>{state.turn=Number(e.target.value);applyTurn();};
let drag=null;const wrap=$('turn-wrap');
wrap.addEventListener('pointerdown',e=>{drag={x:e.clientX,turn:state.turn};wrap.setPointerCapture?.(e.pointerId);});
wrap.addEventListener('pointermove',e=>{if(!drag)return;const delta=Math.round((e.clientX-drag.x)/16);state.turn=(drag.turn+delta+24)%24;applyTurn();});
wrap.addEventListener('pointerup',()=>{drag=null;});
wrap.addEventListener('pointercancel',()=>{drag=null;});
applyView();applyTurn();
function waitImages(){
  return Promise.all([...document.images].map(img=>{
    if(img.complete)return img.naturalWidth?Promise.resolve():Promise.reject(new Error('broken '+img.src));
    return new Promise((res,rej)=>{img.onload=res;img.onerror=()=>rej(new Error('load '+img.src));});
  }));
}
window.addEventListener('load',async()=>{
  const errors=[];window.addEventListener('error',e=>errors.push(String(e.message||e.error||e)));
  try{
    await waitImages();
    const initialTurn=state.turn;
    const evt=(type,x)=>new PointerEvent(type,{bubbles:true,clientX:x,pointerId:1,pointerType:'mouse'});
    wrap.dispatchEvent(evt('pointerdown',120));wrap.dispatchEvent(evt('pointermove',300));wrap.dispatchEvent(evt('pointerup',300));
    const turnChanged=state.turn!==initialTurn;
    document.querySelector('[data-view="front"]').click();
    document.querySelector('[data-which="old"]').click();const oldSrc=$('view-img').getAttribute('src');
    document.querySelector('[data-which="new"]').click();const newSrc=$('view-img').getAttribute('src');
    document.querySelector('[data-mode="clay"]').click();const claySrc=$('view-img').getAttribute('src');
    document.querySelector('[data-mode="material"]').click();
    document.querySelector('[data-view="top"]').click();const topSrc=$('view-img').getAttribute('src');
    document.querySelector('[data-view="front"]').click();
    const result={
      images_loaded:true,error_count:errors.length,errors,
      old_new_differ:oldSrc!==newSrc,clay_material_differ:claySrc!==newSrc,
      turntable_drag_changed:turnChanged,view_switch_changed:topSrc!==newSrc,
      final_state:{...state},link_count:document.querySelectorAll('aside a').length,
      has_glb:!!document.querySelector('a[href="mesh.glb"]'),
      has_blend:!!document.querySelector('a[href="model.blend"]'),
      has_board:!!document.querySelector('a[href="六视图-新旧对照.jpg"]')
    };
    result.passed=!!(result.images_loaded&&!result.errors.length&&result.old_new_differ&&
      result.clay_material_differ&&result.turntable_drag_changed&&result.view_switch_changed&&
      result.has_glb&&result.has_blend&&result.has_board);
    document.getElementById('selftest').textContent=JSON.stringify(result);
    document.body.dataset.selftest=result.passed?'passed':'failed';
  }catch(e){
    const result={passed:false,images_loaded:false,errors:[...errors,String(e&&e.message||e)]};
    document.getElementById('selftest').textContent=JSON.stringify(result);
    document.body.dataset.selftest='failed';
  }
});
"""


def _yes(value):
    if value is True:
        return '<span class="ok">通过</span>'
    if value is False:
        return '<span class="bad">未通过</span>'
    return html.escape(str(value))


def make_review_page(out, report, texture_stats, browser_status='not_run'):
    checks = report['checks']
    metal = report.get('metal_reflection_check') or {}
    dielectric = report.get('dielectric_response_check') or {}
    cylinder = report.get('cylinder_check') or {}
    fidelity = report.get('source_fidelity_diagnostic') or {}
    top = report.get('top_view_aspect') or {}

    def row(name, value, note=''):
        if value is True:
            status, color = 'PASS', '#1d5c3a'
        elif value is False:
            status, color = 'FAIL / 待审', '#8a2b20'
        else:
            status, color = str(value), '#6a5a1e'
        return (f'<tr><td>{html.escape(name)}</td><td style="color:{color};font-weight:700">'
                f'{status}</td><td>{html.escape(note)}</td></tr>')

    rows = ''.join([
        row('网格封闭 / 同部件无自交', checks['mesh_integrity'],
            '四个部件逐件封闭；同部件 BVH 无自交；装配件允许有意搭接'),
        row('盖空腔与底部凹碟', checks['isolated_openings'],
            f"盖空腔深度 {report['ray_checks']['cap_cavity']['cavity_depth_m']} m；"
            f"底部中心高出外缘 {report['ray_checks']['bottom_dish']['centre_minus_rim_m']} m"),
        row('圆柱截面', checks['cylindrical_body'],
            f"最大半径偏差 {cylinder.get('max_abs_radius_deviation_m')} m；"
            f"环半径变化 {cylinder.get('max_ring_radius_variation_m')} m；"
            f"新顶视宽高比 {top.get('new')}"),
        row('材质贴图', checks['material_textures'],
            '原照前幅 + 参考隐藏半；拍摄明暗已压平，污渍/印刷保留'),
        row('GLB 材质内嵌', checks['glb_materials'],
            '9 张 PNG；body/cap/base 各有 baseColor、metallicRoughness、normal'),
        row('金属反射', checks['metal_reflection'],
            f"金属底/卷边两主光平均亮度差：底 {metal.get('base_mean_abs_luma_difference')}，"
            f"边 {metal.get('rim_mean_abs_luma_difference')}"),
        row('塑料/标签高光变光', checks['dielectric_response'],
            f"平均亮度差 {dielectric.get('mean_abs_luma_difference_moved_key')}"),
        row('原视角轮廓诊断', checks['source_fidelity'],
            f"IoU {fidelity.get('iou')}，填孔 {fidelity.get('filled_holes')}；只作诊断"),
        row('检查页静态资源', checks.get('review_page_assets'),
            '本地 HTML/图片路径检查；浏览器自检状态见下方'),
        row('人工验收', 'pending', '候选 manual_only，不进入自动合成'),
    ])
    summary = (
        '<p>本件以原照与做锈图为可见面的首要证据，六视图为未标定视觉参考。'
        '旧模型顶视截面约 1.1:1 椭圆并约有 11° 轴线倾斜，标签糊化，因此未复用旧网格。'
        '候选保留真实圆柱筒身、灰色中空盖与顶缘圆角、盖顶中央浇口凹陷、'
        '卷边金属底、凹碟与凸起内环、隐藏阀杆，以及可移动的金属高光。'
        '可见正面使用做锈图为主、原照辅助的表面；背面来自未标定参考，'
        '参考前幅的净含量字样与原照不一致，未用于可见正面。'
        '六视图左右面板不是一致转台且重复正标/条码，因此不参与包裹纹理；未观测的侧面用带污渍的空白标签补齐。'
        '顶部/底部参考 IoU 可用，正/背/侧参考 IoU 受白标签/白背景自动掩码影响而偏低，只作诊断。'
        '所有候选均为 manual_only、待审。</p>')
    pending = ''.join(f'<li>{html.escape(x)}</li>' for x in report['pending_review'])
    limitations = ''.join(f'<li>{html.escape(x)}</li>' for x in report['limitations'])
    detail_names = (('detail-cap-top', '盖顶'), ('detail-cap-side', '盖侧'),
                    ('detail-cap-seam', '盖缝'), ('detail-cap-dimples', '浇口凹陷'),
                    ('detail-front-label', '正面标签'), ('detail-back-label', '背面标签'),
                    ('detail-bottom-stamp', '底部钢印'), ('detail-metal-rim', '金属卷边'),
                    ('new-material-close', '材质近景'),
                    ('new-surface-alternate', '变光近景'), ('new-source', '原视角诊断'),
                    ('clay-front', '灰模'))
    gallery = ''.join(
        f'<figure><img src="renders/{name}.png" alt="{html.escape(label)}">'
        f'<figcaption>{html.escape(label)}</figcaption></figure>'
        for name, label in detail_names)
    view_buttons = ''.join(
        f'<button data-view="{view}">{label}</button>' for view, label in
        (('front', '正视'), ('back', '背视'), ('left', '左视'),
         ('right', '右视'), ('top', '顶视'), ('bottom', '底视')))
    boards = ''.join(
        f'<h3>{label}</h3><a href="{name}"><img src="{name}" alt="{label}"></a>'
        for name, label in (
            ('六视图-新旧对照.jpg', '六方向新旧对照'),
            ('局部检查.jpg', '关键局部检查'),
            ('结构开口-隔离检查.jpg', '隔离结构与开口'),
            ('材质-同光照对照.jpg', '同光照材质对照'),
            ('材质-变光检查.jpg', '材质变光检查'),
            ('表面反射-变光对照.jpg', '金属反射变光'),
            ('圆柱截面-顶视对照.jpg', '圆柱截面顶视诊断'),
            ('源图-原视角诊断.jpg', '原照 / 原视角诊断'),
            ('转台-24方向.jpg', '24 方向转台'),
            ('PBR图集-布局.jpg', 'PBR 图集布局')))
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>36 自喷漆罐 · 独立检查页（待审）</title><style>{CSS}</style></head>
<body data-number="36">
<header><h1>36 自喷漆罐 · 分部件重建候选<span class="badge">manual_only · 待人工复核</span></h1>
<p>保赐利 BOTNY 灰盖自喷漆罐：圆柱白标签筒身、灰色塑料中空盖、卷边金属底、凹碟底与隐藏阀杆。旧模型存在椭圆截面和约 11° 轴线倾斜，本页展示新候选与旧模型、六视图参考、关键局部和 PBR 贴图的对照。</p></header>
<div class="notice"><b>未验收声明：</b>候选不替换原模型，不进入自动放置或正式合成。六视图未标定；原视角 IoU 和参考 IoU 都是诊断值，不是测量精度。尺寸、背面/侧面标签、盖内配合、壁厚、阀杆、底部钢印、落稳和现场尺度均待人工复核。浏览器自检只覆盖静态页面，不覆盖 WebGL/GLB 实时渲染。</div>
<main>
<section>
<h2>六方向 · 新旧模型 / 灰模</h2>
<div class="controls">
<div class="group"><button data-which="new" class="active">新候选</button><button data-which="old">原模型</button></div>
<div class="group"><button data-mode="material" class="active">材质</button><button data-mode="clay">灰模</button></div>
<div class="group">{view_buttons}</div>
</div>
<figure><img id="view-img" src="renders/new-front.png" alt="new front"><figcaption id="view-caption">新候选 · 正视</figcaption></figure>
<div style="margin-top:18px"><h2>预渲染 24 方向转台</h2>
<div id="turn-wrap"><img id="turn-img" src="renders/turn-00.png" alt="turntable"></div>
<input id="turn-range" type="range" min="0" max="23" value="0">
<p id="turn-caption" style="color:#53656c;font-size:13px">第 1/24 方向</p>
<p style="color:#53656c;font-size:13px">本页没有 WebGL/GLB 实时查看器；GLB 和 Blender 文件供后续串行整合与人工检查。</p>
</div>
<div class="gallery" style="margin-top:16px"><h2 style="grid-column:1/-1">关键局部</h2>{gallery}</div>
<div class="boards" style="margin-top:16px"><h2>对照板</h2>{boards}</div>
</section>
<aside>
<h2>判读摘要</h2>
{summary}
<div class="notice"><b>参考状态：</b>六视图映射为 visual_match_unverified（未标定）。参考前幅净含量字样与原照不一致；背面/侧面标签、盖内配合、壁厚、阀杆、底部钢印和隐藏面为推断。候选为 <b>manual_only</b>，不替换原模型，不进入自动放置或正式合成。</div>
<h2>检查记录</h2>
<table><thead><tr><th>项目</th><th>结果</th><th>说明</th></tr></thead><tbody>{rows}</tbody></table>
<div class="notice"><b>关键数字：</b>原视角诊断 IoU {html.escape(str(fidelity.get('iou')))}（未标定，只作诊断）；顶视宽高比 新 {html.escape(str(top.get('new')))} / 旧 {html.escape(str(top.get('old')))}；标签平均亮度 {html.escape(str(texture_stats['body']['mean_luma']))}；盖平均亮度 {html.escape(str(texture_stats['cap']['mean_luma']))}；金属底平均金属度 {html.escape(str(texture_stats['base']['mean_metallic']))}。</div>
<h2>对照与文件</h2>
<ul class="links">
<li><a href="六视图-新旧对照.jpg">六视图新旧对照</a></li>
<li><a href="局部检查.jpg">关键局部检查</a> · <a href="结构开口-隔离检查.jpg">隔离结构与开口</a></li>
<li><a href="材质-同光照对照.jpg">同光照材质对照</a> · <a href="材质-变光检查.jpg">材质变光</a> · <a href="表面反射-变光对照.jpg">表面反射变光</a></li>
<li><a href="圆柱截面-顶视对照.jpg">圆柱截面顶视诊断</a> · <a href="源图-原视角诊断.jpg">原照/原视角诊断</a></li>
<li><a href="转台-24方向.jpg">24 方向转台拼图</a> · <a href="PBR图集-布局.jpg">PBR 图集布局</a></li>
<li><a href="mesh.glb">下载 GLB（分部件候选）</a> · <a href="model.blend">下载 Blender 可编辑文件</a></li>
<li><a href="result.json">模型验证 JSON</a> · <a href="hashes.json">输入/输出哈希</a> · <a href="progress.json">本目录进度</a></li>
<li><a href="浏览器核验.json">浏览器自检记录</a>（状态：{html.escape(browser_status)}）</li>
</ul>
<h3>待人工复核</h3><ul>{pending}</ul>
<h3>主要限制</h3><ul>{limitations}</ul>
<p class="mono">version: {html.escape(report['version'])}<br>
reference: {html.escape(report['reference'])}<br>
browser: {html.escape(browser_status)}</p>
</aside>
</main>
<pre id="selftest">pending</pre>
<script>{JS}</script>
</body></html>"""
    (out / '检查页.html').write_text(page, encoding='utf8')
    return out / '检查页.html'


def check_review_page(out):
    html_path = out / '检查页.html'
    if not html_path.is_file():
        return {'passed': False, 'reason': '检查页.html missing', 'links': []}
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
                       'window.__ready' not in text and 'turn-00.png' in text),
        'local_links': len(local),
        'missing_links': missing,
        'missing_required': missing_required,
        'note': ('Static local asset/link check only. Browser interaction is '
                 'recorded separately; WebGL/GLB rendering is not claimed.'),
    }


def _chrome():
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return shutil.which('chrome') or shutil.which('msedge')


def browser_check(out):
    out = Path(out)
    page = out / '检查页.html'
    chrome = _chrome()
    record = {
        'status': 'not_run',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'url': page.as_uri(),
        'method': ('Standalone HTML self-test in headless Chrome via file://; '
                   'checks image loading, old/new switch, clay/material switch, view '
                   'switch and synthetic turntable drag. It does not verify WebGL '
                   'rendering, GLB parsing, scene placement or model acceptance.'),
        'result': {},
    }
    if chrome is None:
        record['reason'] = 'Chrome/Edge headless executable not found'
        (out / '浏览器核验.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
        return record
    common = [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
              '--allow-file-access-from-files', '--virtual-time-budget=15000',
              '--window-size=1400,1000']
    try:
        dumped = subprocess.run(common + ['--dump-dom', page.as_uri()],
                                capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=180)
        screenshot = out / '浏览器核验.png'
        shot = subprocess.run(common + ['--screenshot=' + str(screenshot),
                                        page.as_uri()],
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=180)
        match = re.search(r'<pre id="selftest"[^>]*>(.*?)</pre>', dumped.stdout,
                          re.S | re.I)
        payload = html.unescape(match.group(1)) if match else ''
        try:
            result = json.loads(payload)
        except Exception:
            result = {'passed': False, 'parse_error': True, 'payload': payload[:2000]}
        result['chrome_returncode_dump_dom'] = dumped.returncode
        result['chrome_returncode_screenshot'] = shot.returncode
        result['screenshot'] = str(screenshot) if screenshot.is_file() else None
        record['result'] = result
        record['status'] = 'passed' if result.get('passed') else 'failed'
        if record['status'] != 'passed':
            record['reason'] = 'self-test did not pass'
    except Exception as exc:
        record['status'] = 'failed'
        record['reason'] = str(exc)
    (out / '浏览器核验.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
    return record


# ---------------------------------------------------------------------------
# Old-model geometry analysis
# ---------------------------------------------------------------------------
def old_model_analysis(path):
    mesh = trimesh.load(path, force='mesh')
    verts = np.asarray(mesh.vertices, np.float64)
    centre = verts.mean(axis=0)
    cov = np.cov((verts - centre).T)
    values, vectors = np.linalg.eigh(cov)
    principal = vectors[:, int(np.argmax(values))]
    tilt_deg = math.degrees(math.acos(min(1.0, abs(float(principal[2])))))
    basis = [vectors[:, i] for i in range(3) if i != int(np.argmax(values))]
    projected = np.column_stack([(verts - centre) @ b for b in basis])
    pvals = np.linalg.eigvalsh(np.cov(projected.T))
    ellipticity = math.sqrt(max(pvals) / max(1e-12, min(pvals)))
    section_ratios = []
    for z in np.linspace(verts[:, 2].min() + 0.08, verts[:, 2].max() - 0.08, 7):
        try:
            section = mesh.section(plane_origin=[0.0, 0.0, float(z)],
                                   plane_normal=[0.0, 0.0, 1.0])
            if section is None or len(section.vertices) < 8:
                continue
            pts = np.asarray(section.vertices, np.float64)[:, :2]
            pts = pts - pts.mean(axis=0)
            w = np.linalg.eigvalsh(np.cov(pts.T))
            if min(w) > 2e-7:
                section_ratios.append(math.sqrt(float(max(w) / min(w))))
        except Exception:
            continue
    return {
        'principal_axis': [round(float(x), 6) for x in principal],
        'axis_tilt_from_z_deg': round(float(tilt_deg), 3),
        'overall_cross_section_ellipticity': round(float(ellipticity), 4),
        'z_section_ellipticity_min': round(float(min(section_ratios)), 4) if section_ratios else None,
        'z_section_ellipticity_max': round(float(max(section_ratios)), 4) if section_ratios else None,
        'z_section_count': len(section_ratios),
        'note': ('The old library mesh was measured in place, without modification. '
                 'Ellipticity is the principal-axis cross-section major/minor ratio; '
                 'z-section ratios include the tilted-axis effect. This is a geometry '
                 'diagnostic, not a calibrated measurement of the real can.'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', type=int, default=36)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    args = parser.parse_args()
    if args.number != 36:
        parser.error('this entry supports item 36 only')
    args.reference = (args.reference or find_reference()).resolve()
    if args.out is None:
        args.out = ROOT / 'outputs/36自喷漆罐-六视图重建-v1'
    args.out, args.library, args.blender = (x.resolve() for x in
                                            (args.out, args.library, args.blender))
    if digest(args.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = library_row(36)
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
    if digest(source_view_path) != SOURCE_VIEW_SHA256:
        raise ValueError('source-view hash differs from the registered value')
    alpha_path = source_view_path.parent / 'source-alpha.png'
    if not alpha_path.is_file() or ROOT not in alpha_path.resolve().parents:
        raise FileNotFoundError('source alpha missing from this worktree')
    if digest(alpha_path) != SOURCE_ALPHA_SHA256:
        raise ValueError('source alpha hash differs from the model library')
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    source_alpha_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    config_path = HERE / 'spraypaint36_config.json'
    worker_path = HERE / 'mesh_rebuild_36_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    blender_version = subprocess.run([str(args.blender), '--version'],
                                     capture_output=True, text=True,
                                     check=True).stdout.splitlines()[0]
    common_inputs = {
        'reference': digest(args.reference), 'mesh': OLD_MESH_SHA256,
        'clean': CLEAN_SHA256, 'rust': RUST_SHA256,
        'source_alpha': digest(alpha_path), 'source_view': digest(source_view_path),
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
            json.dumps({'run_signature': run_signature, 'inputs': common_inputs},
                       ensure_ascii=False, indent=2), encoding='utf8')
        body_textures, body_stats, _ = make_body_textures(
            clean_path, rust_path, args.reference, args.out, config)
        cap_textures, cap_stats = make_cap_textures(clean_path, rust_path, args.out, config)
        base_textures, base_stats = make_base_textures(
            clean_path, rust_path, args.out, config)
        textures = {**body_textures, **cap_textures, **base_textures}
        stats = texture_checks(body_stats, cap_stats, base_stats)
        (args.out / 'texture-stats.json').write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf8')
        request = {
            'out': str(args.out), 'original_mesh': str(old_mesh_path),
            'textures': textures, 'config': config,
            'source_camera': row['reconstruction']['source_camera'],
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
    ray = ray_checks(args.out / 'mesh.glb')
    parts = load_named_parts(args.out / 'mesh.glb')
    body_mesh = parts.get('Can body')
    if body_mesh is None:
        raise ValueError('exported GLB is missing the Can body node')
    body_vertices = np.asarray(body_mesh.vertices)
    z_values = body_vertices[:, 2]
    side = body_vertices[np.logical_and(z_values > 0.0063, z_values < 0.1500)]
    radii = np.hypot(side[:, 0], side[:, 1])
    radius = config['dimensions']['body_radius_m']
    rings = {}
    for vertex in side:
        rings.setdefault(round(float(vertex[2]), 6), []).append(
            float(np.hypot(vertex[0], vertex[1])))
    ring_variation = max((max(values) - min(values) for values in rings.values()),
                         default=0.0)
    new_top_ratio = alpha_bbox_ratio(args.out / 'renders/new-top.png')
    old_top_ratio = alpha_bbox_ratio(args.out / 'renders/old-top.png')
    cylinder_check = {
        'body_radius_m': radius,
        'side_vertex_count': int(len(side)),
        'max_abs_radius_deviation_m': round(float(np.max(np.abs(radii - radius))), 9),
        'max_ring_radius_variation_m': round(float(ring_variation), 9),
        'top_view_aspect_new': new_top_ratio,
        'top_view_aspect_old': old_top_ratio,
        'note': ('Only the straight side band 0.0063 < z < 0.1500 m is measured. '
                 'The check confirms cylindrical symmetry of the modelled side, '
                 'not measured dimensions.'),
    }
    old_analysis = old_model_analysis(old_mesh_path)
    new_front_luma = median_luma(args.out / 'renders/new-front.png')
    old_front_luma = median_luma(args.out / 'renders/old-front.png')
    base_difference = mean_difference(args.out / 'renders/metal-base-key.png',
                                      args.out / 'renders/metal-base-alt.png')
    rim_difference = mean_difference(args.out / 'renders/metal-rim-key.png',
                                     args.out / 'renders/metal-rim-alt.png')
    base_shift = highlight_centroid_shift(args.out / 'renders/metal-base-key.png',
                                          args.out / 'renders/metal-base-alt.png')
    rim_shift = highlight_centroid_shift(args.out / 'renders/metal-rim-key.png',
                                         args.out / 'renders/metal-rim-alt.png')
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
    make_source_board(args.out, clean_path, rust_path, source_view_path)
    make_top_compare(args.out, new_top_ratio, old_top_ratio)
    make_atlas_board(args.out)

    names = [part['name'] for part in worker['parts']]
    structural = {
        'bodies': sum(name == 'Can body' for name in names),
        'bases': sum(name == 'Can base' for name in names),
        'caps': sum(name == 'Cap shell' for name in names),
        'valves': sum(name == 'Valve stem' for name in names),
        'note': ('Part presence proves only that a feature was modelled. Cap wall '
                 'thickness, cavity fit, valve mechanism, hidden print and all '
                 'hidden faces are inferred.'),
    }
    structural_ok = bool(structural['bodies'] == 1 and structural['bases'] == 1 and
                         structural['caps'] == 1 and structural['valves'] == 1)
    cylindrical_ok = bool(
        cylinder_check['max_abs_radius_deviation_m'] < 1e-6 and
        cylinder_check['max_ring_radius_variation_m'] < 1e-6 and
        new_top_ratio is not None and abs(new_top_ratio - 1.0) < 0.03)
    metal_reflection = {
        'status': ('passed' if (stats['base']['mean_metallic'] >= 0.98 and
                                ((base_difference is not None and base_difference > 1.0) or
                                 (rim_difference is not None and rim_difference > 0.8)))
                   else 'failed_or_inconclusive'),
        'mean_metallic': stats['base']['mean_metallic'],
        'base_mean_abs_luma_difference': base_difference,
        'rim_mean_abs_luma_difference': rim_difference,
        'base_highlight_centroid_shift_px': base_shift,
        'rim_highlight_centroid_shift_px': rim_shift,
        'note': ('The aluminium base and rolled rim are metallic 1 and their '
                 'renders change under a moved key light. This is a response '
                 'test, not a claim about the unseen surface finish.'),
    }
    dielectric = {
        'status': ('passed' if (stats['body']['mean_metallic'] <= 0.02 and
                                stats['cap']['mean_metallic'] <= 0.02 and
                                dielectric_difference is not None and
                                dielectric_difference > 1.0)
                   else 'failed_or_inconclusive'),
        'mean_abs_luma_difference_moved_key': dielectric_difference,
        'note': ('The printed label and grey cap are dielectric; their highlights '
                 'move with the key light. This does not validate ink gloss.'),
    }
    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] == 4 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components'] and
                               worker['cap_boundary_edges_after_boolean'] == 0),
        'isolated_openings': bool(ray['cap_cavity']['passed'] and
                                  ray['bottom_dish']['passed']),
        'hidden_valve': bool(ray['hidden_valve']['passed']),
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

    audit_fidelity = None
    audit_warning = None
    audit_path = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
    if audit_path.is_file():
        for line in audit_path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('number') == 36:
                audit_fidelity = item.get('source_fidelity')
                audit_warning = item.get('warnings')
                break

    version = config['version']
    report = {
        'version': version, 'number': 36, 'asset_id': row['asset_id'],
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
            'The old model-library source IoU 0.9759 only measures one silhouette. '
            'The six-view audit records an elliptical top cross-section. This '
            'candidate measures the old mesh in place: principal-axis tilt '
            f"{old_analysis['axis_tilt_from_z_deg']} deg and cross-section "
            f"ellipticity {old_analysis['overall_cross_section_ellipticity']}:1, "
            'so the old mesh is not reused as the candidate base geometry.'),
        'original_source_fidelity_record': audit_fidelity,
        'original_audit_warnings': audit_warning,
        'old_model_analysis': old_analysis,
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
        'reference_mask_note': (
            'The six-view panels have a white label on a white background; the '
            'automatic threshold mask keeps mostly dark label/cap pixels, so the '
            'reference front/back/side IoUs are depressed and are not comparable '
            'to the original-alpha source IoU. They are diagnostic only.'),
        'render_exposure_note': (
            'New-candidate renders use -0.6 EV. The old model uses the same lights '
            'and background at -1.6 EV (one stop lower) because its vertex-colour '
            'material clips under the candidate exposure; this affects display '
            'brightness, not light direction.'),
        'top_view_aspect': {'new': new_top_ratio, 'old': old_top_ratio,
                            'note': ('Width/height of the isolated top-view alpha '
                                     'silhouette. The old ellipse and axis tilt are '
                                     'recorded as the defect that motivated the rebuild.')},
        'render_luma': {'new_front_median': new_front_luma,
                        'old_front_median': old_front_luma,
                        'note': 'Neutral-light luma only; not a lighting acceptance.'},
        'mesh_audit': mesh_audit, 'glb_audit': glb, 'worker': worker,
        'cylinder_check': cylinder_check,
        'ray_checks': ray, 'structural_checks': structural,
        'metal_reflection_check': metal_reflection,
        'dielectric_response_check': dielectric,
        'checks': checks,
        'limitations': [
            'The six-view sheet is an uncalibrated generated visual reference, not a measured turntable; its front text differs from the photographed can.',
            'Only one front photograph exists. The back, hidden side transitions, cap interior, valve and the actual bottom stamping/code are inferred.',
            'Working dimensions (200 mm tall, 64.4 mm body diameter, 66 mm cap diameter) follow the photo/six-view pixel ratio and a nominal 400 ml envelope; they are not measured dimensions.',
            'The label front is projected from the 3/4 photograph over +/-85 degrees. Wrap registration, side-transition seams and hidden print are inferred; the six-view left/right panels are not a consistent turntable.',
            'The cap is a closed solid with a cylindrical cavity cut from below. Wall thickness, snap fit, cavity depth, top dimple and mould seam are inferred.',
            'The valve cup/stem under the cap is hidden and simplified; its material and internal mechanism are not observable.',
            'The bottom dish profile, raised ring and batch/expiry stamp are inferred from the uncalibrated reference; the reference code is not a claim about this exact can.',
            'Broad photographed lighting was flattened out of the albedo; the exact ink gloss, plastic roughness and aluminium finish were not measured.',
            'Cap/body and base/body assemblies intentionally overlap; same-part BVH checks do not prove part-to-part clearance or physical stability.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views, cap edge, seam, label wrap and bottom dish against the physical can.',
            'Confirm the actual body/cap diameter, total height and label extent with calipers or a measured drawing.',
            'Confirm the hidden back/side label wording, barcode and net weight; the reference front appears to state a different weight from the source photograph.',
            'Confirm cap wall thickness, snap fit, cavity depth, top dimple and mould seam with the real part.',
            'Confirm the valve stem/cup and the stamped bottom profile and batch/expiry code.',
            'Verify coordinate/scale, stability, floor contact and lighting integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
        ],
        'next_number': 37,
    }
    (args.out / 'result.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (args.out / 'progress.json').write_text('{}', encoding='utf8')
    (args.out / 'hashes.json').write_text('{}', encoding='utf8')
    make_review_page(args.out, report, stats, browser_status='pending')
    page_check = check_review_page(args.out)
    browser = browser_check(args.out)
    report['browser_check'] = browser
    report['checks']['browser_page'] = browser['status']
    report['checks']['review_page_assets'] = page_check['passed']
    report['page_assets_check'] = page_check
    (args.out / 'result.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    make_review_page(args.out, report, stats, browser_status=browser['status'])
    page_check = check_review_page(args.out)
    report['page_assets_check'] = page_check
    report['checks']['review_page_assets'] = page_check['passed']
    (args.out / 'result.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

    progress = {
        'number': 36, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review', 'version': version,
        'out': str(args.out), 'result': str(args.out / 'result.json'),
        'mesh': str(args.out / 'mesh.glb'), 'blend': str(args.out / 'model.blend'),
        'textures': textures, 'review_page': str(args.out / '检查页.html'),
        'comparison_board': str(args.out / '六视图-新旧对照.jpg'),
        'detail_board': str(args.out / '局部检查.jpg'),
        'isolated_board': str(args.out / '结构开口-隔离检查.jpg'),
        'checks': checks, 'texture_stats': stats, 'ray_checks': ray,
        'cylinder_check': cylinder_check, 'glb_audit': glb,
        'top_view_aspect': report['top_view_aspect'],
        'source_fidelity_diagnostic': fidelity,
        'old_model_analysis': old_analysis,
        'reference_role': config['reference_role'],
        'manual_only': True,
        'pending_review': report['pending_review'], 'next_number': 37,
        'notes': ('The six-view sheet is an uncalibrated visual reference; the '
                  'single clean/rust photo is the primary visible-front evidence. '
                  'The old model and other items were not modified. Candidate is '
                  'manual_only pending human review.'),
    }
    (args.out / 'progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    external = {
        'number': 36, 'status': progress['status'],
        'out': str(args.out), 'mesh': str(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'), 'checks': checks,
        'pending_review': report['pending_review'], 'next_number': 37,
        'manual_only': True,
        'review_page': str(args.out / '检查页.html'),
        'notes': progress['notes'],
    }
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '36-progress.json').write_text(
        json.dumps(external, ensure_ascii=False, indent=2), encoding='utf8')
    files = [path for path in sorted(args.out.rglob('*')) if path.is_file()]
    hashes = {
        'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_36.py',
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
    (args.out / 'hashes.json').write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(args.out), 'checks': checks,
                      'texture_stats': stats, 'ray_checks': ray,
                      'source_fidelity': fidelity,
                      'top_view_aspect': report['top_view_aspect'],
                      'cylinder_check': cylinder_check,
                      'old_model_analysis': old_analysis,
                      'browser': browser['status'],
                      'page_assets': page_check['passed']}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()

