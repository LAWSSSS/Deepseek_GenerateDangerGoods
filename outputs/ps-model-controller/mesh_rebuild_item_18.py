"""Build the review-only mountain-bike rear air-shock candidate (item 18).

The clean/rust photographs show one used black DNM-style rear air shock at a
three-quarter angle.  The user six-view sheet is an uncalibrated visual-match
atlas of a closely related DNM shock; it is used for part topology and
appearance only, never as measured geometry.  The candidate is a parametric,
part-based rebuild with real upper/lower clevis bores and steel bushings, a
black anodised air can with a wrap decal, a top cap with red rebound knob and
blue three-position lever, a seal head, blue sag O-ring and damper shaft.  It
never replaces the library mesh, stays manual-only and never enters automatic
placement.  A new output directory is required.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes

from scene_v2 import digest
from mesh_checks import source_fidelity
from mesh_carve import interior_holes
from mesh_rebuild_item import mask_fit, inspect_mesh, _tile_detail

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE_SHA256 = '1f26cd1a91d11a90e91594bfb33fb01e277765a1844d74752806441af4fdb236'
REFERENCE_NAME = '黑色山地车后避震六视图图册.png'
PANEL_BOX = {
    'front': (0, 0, 512, 512),
    'back': (512, 0, 1024, 512),
    'left': (1024, 0, 1536, 512),
    'right': (0, 512, 512, 1024),
    'top': (512, 512, 1024, 1024),
    'bottom': (1024, 512, 1536, 1024),
}
OLD_REL = 'outputs/三维模型库-圆润复验/items/18-asset-52fb28249a6a/mesh.glb'
ALPHA_REL = 'outputs/三维模型库-圆润复验/items/18-asset-52fb28249a6a/source-alpha.png'
CLEAN_REL = 'outputs/物品高清输入-39件/原貌HD/18-气压减震器1.png'
RUST_REL = 'outputs/物品高清输入-39件/生锈HD/18-气压减震器1.png'
SOURCE_VIEW_REL = 'outputs/三维模型库-圆润复验/items/18-asset-52fb28249a6a/source-view.png'
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
LOGO_FONT = 'C:/Windows/Fonts/ariblk.ttf'
BOLD_FONT = 'C:/Windows/Fonts/arialbd.ttf'
MONO_FONT = 'C:/Windows/Fonts/consola.ttf'


def find_reference():
    local = ROOT / 'references/parallel-six-view' / REFERENCE_NAME
    if local.is_file() and digest(local) == REFERENCE_SHA256:
        return local
    folder = Path.home() / 'Desktop' / '六视图'
    if folder.is_dir():
        path = folder / REFERENCE_NAME
        if path.is_file() and digest(path) == REFERENCE_SHA256:
            return path
    raise FileNotFoundError('registered six-view reference not found')


def panel_image(reference_image, name):
    return reference_image.crop(PANEL_BOX[name])


def reference_mask(panel):
    """Approximate foreground mask of one reference panel (diagnostic only)."""
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    mask = (rgb.min(axis=2) < 225).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .90):] = 0
    mask[:5] = 0
    mask[-5:] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(30, areas.max() * .004)) + 1
    return np.isin(labels, keep).astype(np.uint8)


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------
def _noise(size, rng, cells, blur=0.0):
    if isinstance(size, tuple):
        height, width = size
    else:
        height = width = size
    field = cv2.resize(rng.random((cells, cells), dtype=np.float32), (width, height),
                       interpolation=cv2.INTER_CUBIC)
    if blur:
        field = cv2.GaussianBlur(field, (0, 0), blur)
    return field


def _normal_from_height(height, strength=0.6, blur=1.0):
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), blur)
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx = np.clip(-dx * strength, -.18, .18)
    ny = np.clip(dy * strength, -.18, .18)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    return np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                   0, 255).astype(np.uint8)


def _mr_map(rough, metallic):
    return np.stack((np.full(rough.shape, 255, np.uint8),
                     np.clip(rough * 255, 0, 255).astype(np.uint8),
                     np.clip(metallic * 255, 0, 255).astype(np.uint8)),
                    axis=2).astype(np.uint8)


def _draw_polyline(draw, points, fill, width):
    draw.line(points, fill=fill, width=width, joint='curve')
    for x, y in (points[0], points[-1]):
        r = width // 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _label_graphics(draw, width, height):
    """Approximate the observed DNM/DRIFT wrap; not an exact artwork copy."""
    cx = width // 2
    cy = int(height * .51)
    grey = (150, 152, 156, 255)
    red = (188, 28, 36, 255)
    white = (232, 234, 236, 255)
    dark = (20, 21, 23, 255)
    # Angular silver wings behind the front X.
    for sign in (-1, 1):
        pts = [(cx + sign * 610, cy - 170), (cx + sign * 300, cy - 230),
               (cx + sign * 220, cy - 30), (cx + sign * 470, cy + 110),
               (cx + sign * 610, cy + 60)]
        draw.polygon(pts, fill=grey, outline=red)
        draw.line([pts[0], pts[1]], fill=red, width=7)
    # Layered X: red edge, silver body, dark core.
    for fill, stroke in ((red, 132), (grey, 104), (dark, 48)):
        _draw_polyline(draw, [(cx - 210, cy - 230), (cx + 210, cy + 230)],
                       fill, stroke)
        _draw_polyline(draw, [(cx + 210, cy - 230), (cx - 210, cy + 230)],
                       fill, stroke)
    # Front lettering.
    try:
        font_logo = ImageFont.truetype(LOGO_FONT, 112)
        font_sub = ImageFont.truetype(BOLD_FONT, 27)
        font_drift = ImageFont.truetype(LOGO_FONT, 90)
    except Exception:
        font_logo = font_sub = font_drift = ImageFont.load_default()
    draw.text((cx - 575, cy - 30), 'DNM', font=font_logo, fill=white,
              stroke_width=4, stroke_fill=dark, anchor='lm')
    draw.text((cx - 572, cy + 55), 'SUSPENSION', font=font_sub, fill=white,
              anchor='lm')
    # Red chevron above DRIFT.
    draw.polygon([(cx + 70, cy + 55), (cx + 150, cy - 2), (cx + 230, cy + 55),
                  (cx + 196, cy + 55), (cx + 150, cy + 26), (cx + 104, cy + 55)],
                 fill=red)
    draw.text((cx + 78, cy + 205), 'DRIFT', font=font_drift, fill=white,
              stroke_width=9, stroke_fill=red, anchor='lm')
    # Back-side wrap (drawn across the seam at x=0/width).
    for tx in (-190, width - 190):
        draw.text((tx, cy - 55), 'DNM', font=font_logo, fill=white,
                  stroke_width=4, stroke_fill=dark, anchor='lm')
        draw.text((tx + 18, cy + 55), 'RC-8RC', font=font_sub, fill=white,
                  anchor='lm')
    for sign in (-1, 1):
        base = 0 if sign < 0 else width
        pts = [(base + sign * 240, cy - 250), (base + sign * 680, cy - 190),
               (base + sign * 720, cy + 90), (base + sign * 380, cy + 170)]
        draw.polygon(pts, fill=grey, outline=red)
        draw.line([pts[0], pts[1]], fill=red, width=6)
    # Upper lock/pedal/open icon (front).
    icon_y = int(height * .17)
    draw.ellipse((cx - 24, icon_y, cx + 24, icon_y + 48), outline=white, width=5)
    draw.line([(cx, icon_y + 48), (cx, icon_y + 112)], fill=white, width=6)
    draw.line([(cx - 20, icon_y + 112), (cx + 20, icon_y + 112)],
              fill=white, width=6)
    try:
        font_small = ImageFont.truetype(MONO_FONT, 26)
    except Exception:
        font_small = ImageFont.load_default()
    draw.text((cx + 46, icon_y - 2), 'OPEN', font=font_small, fill=white)
    draw.text((cx - 170, icon_y + 58), 'LOCK', font=font_small, fill=white)
    draw.text((cx + 8, icon_y + 132), 'PEDAL', font=font_small, fill=white)


def build_can_textures(rust, out, cfg):
    width, height = cfg['texture']['can_size']
    rng = np.random.default_rng(18181)
    mat = cfg['material_profile']['can']
    base_rgb = np.asarray(mat['base_rgb'], np.float32)
    low = _noise((height, width), rng, 42, 2.0)
    fine = rng.normal(0, 1, (height, width)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    circ = (.55 * np.sin(TAU_V * yy / height * 185 + 0.9) +
            .30 * np.sin(TAU_V * yy / height * 61 - 1.1))
    rgb = base_rgb[None, None, :] * (1 + .035 * (low[:, :, None] - .5) +
                                     .006 * fine[:, :, None])
    rgb += circ[:, :, None] * .9
    # Small observed patches for microdetail and grime (used as fields, not as
    # a direct photo transfer).
    rust_im = Image.open(rust).convert('RGBA')
    rects = cfg['texture']['rust_patch_rects']
    detail_a = cv2.resize(_tile_detail(rust_im, tuple(rects['black_upper'])),
                          (width, height), interpolation=cv2.INTER_CUBIC)
    detail_b = cv2.resize(_tile_detail(rust_im, tuple(rects['black_lower'])),
                          (width, height), interpolation=cv2.INTER_CUBIC)
    detail = .55 * detail_a + .45 * detail_b
    grime_field = _noise((height, width), rng, 34, 4.0)
    grime = np.clip((grime_field - .76) * 5.5, 0, 1)
    grime *= np.clip((fine * .5 + .5) * 1.25, 0, 1)
    grime *= .35 + .65 * np.clip((yy / height - .55) * 2.2, 0, 1)
    # Decal layer.
    decal = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(decal)
    _label_graphics(draw, width, height)
    decal_array = np.asarray(decal).astype(np.float32)
    ink = decal_array[:, :, 3:4] / 255.0
    rgb = rgb * (1 - ink) + decal_array[:, :, :3] * ink
    # Grime, scuffs and a few light scratches; kept subtle so the anodised
    # black does not turn into a flat black silhouette.
    grime_rgb = np.asarray(mat['grime_rgb'], np.float32)
    keep = 1 - .65 * ink
    rgb = rgb * (1 - .08 * grime[:, :, None] * keep) + \
        grime_rgb[None, None, :] * (.08 * grime[:, :, None] * keep)
    scratch = np.clip((detail - 5.5) * .009, 0, 1) * np.clip(1 - ink[:, :, 0] * .9, 0, 1)
    rgb += np.array([120, 118, 114], np.float32)[None, None, :] * scratch[:, :, None]
    rgb = np.clip(rgb, 0, 255)
    rough = (mat['base_roughness'] * (1 - ink[:, :, 0]) +
             mat['ink_roughness'] * ink[:, :, 0] +
             .08 * grime + .012 * np.clip(detail / 12, -.5, .5))
    metal = (mat['metallic'] * (1 - ink[:, :, 0]) +
             mat['ink_metallic'] * ink[:, :, 0]) * (1 - .35 * grime)
    mr = _mr_map(np.clip(rough, .16, .72), np.clip(metal, 0, 1))
    height_field = (.55 * detail / 120.0 + .25 * circ +
                    .20 * cv2.GaussianBlur(ink[:, :, 0], (0, 0), 1.0))
    normal = _normal_from_height(height_field, .10, .9)
    paths = {}
    for name, arr in (('can_base', rgb.astype(np.uint8)), ('can_mr', mr),
                      ('can_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_cap_textures(rust, out, cfg):
    width, height = cfg['texture']['cap_size']
    rng = np.random.default_rng(18282)
    mat = cfg['material_profile']['cap']
    base_rgb = np.asarray(mat['base_rgb'], np.float32)
    low = _noise((height, width), rng, 22, 2.0)
    fine = rng.normal(0, 1, (height, width)).astype(np.float32)
    rgb = base_rgb[None, None, :] * (1 + .04 * (low[:, :, None] - .5) +
                                     .008 * fine[:, :, None])
    decal = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(decal)
    try:
        font_text = ImageFont.truetype(BOLD_FONT, 48)
        font_small = ImageFont.truetype(MONO_FONT, 25)
    except Exception:
        font_text = font_small = ImageFont.load_default()
    draw.text((width // 2, 328), 'Rebound', font=font_text,
              fill=(214, 216, 218, 255), anchor='mm')
    draw.line([(width // 2 - 92, 390), (width // 2 + 92, 390)],
              fill=(214, 216, 218, 255), width=4)
    for sign in (-1, 1):
        x = width // 2 + sign * 92
        draw.polygon([(x, 390), (x - sign * 14, 380), (x - sign * 14, 400)],
                     fill=(214, 216, 218, 255))
    draw.text((width // 2, 440), 'DNM SUSPENSION', font=font_small,
              fill=(170, 172, 174, 255), anchor='mm')
    decal_array = np.asarray(decal).astype(np.float32)
    ink = decal_array[:, :, 3:4] / 255.0
    rgb = rgb * (1 - ink) + decal_array[:, :, :3] * ink
    rgb = np.clip(rgb, 0, 255)
    rough = mat['base_roughness'] * (1 - ink[:, :, 0]) + .47 * ink[:, :, 0]
    metal = mat['metallic'] * (1 - ink[:, :, 0]) + .05 * ink[:, :, 0]
    mr = _mr_map(np.clip(rough, .18, .55), np.clip(metal, 0, 1))
    height_field = .6 * fine / 8.0 + .4 * cv2.GaussianBlur(ink[:, :, 0], (0, 0), 1.0)
    normal = _normal_from_height(height_field, .25, .8)
    paths = {}
    for name, arr in (('cap_base', rgb.astype(np.uint8)), ('cap_mr', mr),
                      ('cap_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_black_textures(rust, out, cfg):
    size = cfg['texture']['generic_size'][0]
    rng = np.random.default_rng(18383)
    mat = cfg['material_profile']['black']
    base_rgb = np.asarray(mat['base_rgb'], np.float32)
    low = _noise(size, rng, 28, 2.0)
    fine = rng.normal(0, 1, (size, size)).astype(np.float32)
    streak = cv2.GaussianBlur(rng.normal(0, 1, (size // 8, size // 8)).astype(np.float32),
                              (0, 0), 1.0)
    streak = cv2.resize(streak, (size, size), interpolation=cv2.INTER_CUBIC)
    stripe = .25 * np.sin(np.linspace(0, math.tau * 170, size))[None, :]
    rgb = base_rgb[None, None, :] * (1 + .04 * (low[:, :, None] - .5) +
                                     .008 * fine[:, :, None])
    rgb += (stripe[:, :, None] + .6 * streak[:, :, None]) * .7
    rgb = np.clip(rgb, 0, 255)
    rough = np.clip(mat['base_roughness'] + .05 * (low - .5) +
                    .012 * fine + .02 * streak, .20, .58)
    metal = np.full((size, size), mat['metallic'], np.float32)
    mr = _mr_map(rough, metal)
    height_field = .55 * fine / 8.0 + .30 * stripe + .15 * streak
    normal = _normal_from_height(height_field, .30, .8)
    paths = {}
    for name, arr in (('black_base', rgb.astype(np.uint8)), ('black_mr', mr),
                      ('black_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_steel_textures(out, cfg):
    size = cfg['texture']['steel_size'][0]
    rng = np.random.default_rng(18484)
    mat = cfg['material_profile']['steel']
    base_rgb = np.asarray(mat['base_rgb'], np.float32)
    low = _noise(size, rng, 20, 1.5)
    fine = rng.normal(0, 1, (size, size)).astype(np.float32)
    brushed = cv2.GaussianBlur(rng.normal(0, 1, (size // 10, size // 10)).astype(np.float32),
                               (0, 0), .8)
    brushed = cv2.resize(brushed, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = base_rgb[None, None, :] * (1 + .05 * (low[:, :, None] - .5) +
                                     .010 * fine[:, :, None] +
                                     .025 * brushed[:, :, None])
    rgb = np.clip(rgb, 0, 255)
    rough = np.clip(mat['base_roughness'] + .05 * (low - .5) +
                    .015 * fine + .03 * brushed, .15, .52)
    metal = np.full((size, size), mat['metallic'], np.float32)
    mr = _mr_map(rough, metal)
    height_field = .5 * brushed + .35 * fine / 8.0 + .15 * low
    normal = _normal_from_height(height_field, .25, .7)
    paths = {}
    for name, arr in (('steel_base', rgb.astype(np.uint8)), ('steel_mr', mr),
                      ('steel_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_lever_textures(out, cfg):
    size = cfg['texture']['lever_size'][0]
    rng = np.random.default_rng(18585)
    mat = cfg['material_profile']['lever_mark']
    base_rgb = np.asarray(mat['base_rgb'], np.float32)
    fine = rng.normal(0, 1, (size, size)).astype(np.float32)
    low = _noise(size, rng, 18, 1.2)
    rgb = base_rgb[None, None, :] * (1 + .04 * (low[:, :, None] - .5) +
                                     .006 * fine[:, :, None])
    decal = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(decal)
    try:
        font_mark = ImageFont.truetype(LOGO_FONT, 108)
    except Exception:
        font_mark = ImageFont.load_default()
    ink = (236, 239, 242, 255)
    draw.text((size // 2, 215), '+ L -', font=font_mark, fill=ink, anchor='mm')
    draw.arc((size // 2 - 118, 250, size // 2 + 118, 380), 20, 160,
             fill=ink, width=13)
    draw.line([(size // 2, 250), (size // 2, 300)], fill=ink, width=13)
    draw.ellipse((18, 18, size - 18, size - 18), outline=(24, 112, 220, 255),
                 width=5)
    decal_array = np.asarray(decal).astype(np.float32)
    alpha = decal_array[:, :, 3:4] / 255.0
    rgb = rgb * (1 - alpha) + decal_array[:, :, :3] * alpha
    rgb = np.clip(rgb, 0, 255)
    rough = mat['base_roughness'] * (1 - alpha[:, :, 0]) + \
        mat['ink_roughness'] * alpha[:, :, 0]
    metal = mat['metallic'] * (1 - alpha[:, :, 0]) + \
        mat['ink_metallic'] * alpha[:, :, 0]
    mr = _mr_map(np.clip(rough, .18, .58), np.clip(metal, 0, 1))
    height_field = .5 * fine / 8.0 + .5 * cv2.GaussianBlur(alpha[:, :, 0], (0, 0), .8)
    normal = _normal_from_height(height_field, .20, .6)
    paths = {}
    for name, arr in (('lever_base', rgb.astype(np.uint8)), ('lever_mr', mr),
                      ('lever_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


TAU_V = math.tau


def texture_checks(textures, cfg):
    def load(key):
        return np.asarray(Image.open(textures[key]).convert('RGB')).astype(np.float32)

    can_base = load('can_base')
    can_mr = load('can_mr')
    can_normal = load('can_normal')
    h, w = can_base.shape[:2]
    top_black = can_base[0:110, 0:300]
    bottom_black = can_base[int(h * .92):, :]
    decal = can_base[int(h * .16):int(h * .88), 250:1800]
    decal_mr = can_mr[int(h * .16):int(h * .88), 250:1800]
    black_mr = can_mr[0:110, 0:300]
    bottom_mr = can_mr[int(h * .92):, :]
    luma = lambda a: float(a.mean())
    chroma = lambda a: float((a.max(axis=2) - a.min(axis=2)).mean())
    black_luma = float(np.concatenate([top_black.reshape(-1, 3),
                                       bottom_black.reshape(-1, 3)]).mean())
    black_metal = float(np.concatenate([black_mr[..., 2].reshape(-1),
                                        bottom_mr[..., 2].reshape(-1)]).mean() / 255)
    black_rough = float(np.concatenate([black_mr[..., 1].reshape(-1),
                                        bottom_mr[..., 1].reshape(-1)]).mean() / 255)
    decal_luma = float(decal.mean())
    decal_std = float(decal.mean(axis=2).std())
    decal_flat = decal.reshape(-1, 3)
    decal_mr_flat = decal_mr.reshape(-1, 3)
    ink_mask = ((decal_flat.mean(axis=1) > 60) |
                ((decal_flat.max(axis=1) - decal_flat.min(axis=1)) > 20))
    if int(ink_mask.sum()) >= 100:
        decal_ink_metal = float(decal_mr_flat[ink_mask, 2].mean() / 255)
        decal_ink_rough = float(decal_mr_flat[ink_mask, 1].mean() / 255)
    else:
        decal_ink_metal = decal_ink_rough = None
    decal_metal = float(decal_mr[..., 2].mean() / 255)
    decal_rough = float(decal_mr[..., 1].mean() / 255)
    can_ok = bool(10 <= black_luma <= 85 and black_metal >= .70 and
                  .16 <= black_rough <= .66 and decal_std >= 20 and
                  decal_ink_metal is not None and decal_ink_metal <= .45 and
                  decal_ink_rough is not None and .20 <= decal_ink_rough <= .70 and
                  float(can_normal.std()) >= 1.0)
    checks = {'can': {
        'black_band_luma': round(black_luma, 2),
        'black_band_metallic': round(black_metal, 4),
        'black_band_roughness': round(black_rough, 4),
        'decal_luma': round(decal_luma, 2),
        'decal_luma_std': round(decal_std, 2),
        'decal_ink_metallic': None if decal_ink_metal is None else round(decal_ink_metal, 4),
        'decal_ink_roughness': None if decal_ink_rough is None else round(decal_ink_rough, 4),
        'decal_region_metallic': round(decal_metal, 4),
        'decal_region_roughness': round(decal_rough, 4),
        'normal_std': round(float(can_normal.std()), 3),
        'passed': can_ok,
        'note': ('Black anodised areas must stay metallic; printed decal ink is '
                 'dielectric. Metallic/roughness are also reported for the ink '
                 'pixels only, because the label region also contains black metal.'),
    }}
    cap_base = load('cap_base')
    cap_mr = load('cap_mr')
    cap_normal = load('cap_normal')
    cap_ok = bool(15 <= float(cap_base.mean()) <= 90 and
                  float(cap_mr[..., 2].mean() / 255) >= .60 and
                  .18 <= float(cap_mr[..., 1].mean() / 255) <= .66 and
                  float(cap_normal.std()) >= 1.0 and
                  float(cap_base.max()) >= 180)
    checks['cap'] = {
        'mean_luma': round(float(cap_base.mean()), 2),
        'mean_metallic': round(float(cap_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(cap_mr[..., 1].mean() / 255), 4),
        'normal_std': round(float(cap_normal.std()), 3),
        'has_white_print': bool(float(cap_base.max()) >= 180),
        'passed': cap_ok,
    }
    black_base = load('black_base')
    black_mr = load('black_mr')
    black_normal = load('black_normal')
    black_ok = bool(15 <= float(black_base.mean()) <= 90 and
                    float(black_mr[..., 2].mean() / 255) >= .80 and
                    .20 <= float(black_mr[..., 1].mean() / 255) <= .60 and
                    float(black_normal.std()) >= 1.0)
    checks['black'] = {
        'mean_luma': round(float(black_base.mean()), 2),
        'mean_metallic': round(float(black_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(black_mr[..., 1].mean() / 255), 4),
        'normal_std': round(float(black_normal.std()), 3),
        'passed': black_ok,
    }
    steel_base = load('steel_base')
    steel_mr = load('steel_mr')
    steel_normal = load('steel_normal')
    steel_ok = bool(80 <= float(steel_base.mean()) <= 220 and
                    float(steel_mr[..., 2].mean() / 255) >= .80 and
                    .15 <= float(steel_mr[..., 1].mean() / 255) <= .60 and
                    float(steel_normal.std()) >= 1.0)
    checks['steel'] = {
        'mean_luma': round(float(steel_base.mean()), 2),
        'mean_metallic': round(float(steel_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(steel_mr[..., 1].mean() / 255), 4),
        'normal_std': round(float(steel_normal.std()), 3),
        'passed': steel_ok,
    }
    lever_base = load('lever_base')
    lever_mr = load('lever_mr')
    lever_normal = load('lever_normal')
    lever_mean = lever_base.reshape(-1, 3).mean(axis=0)
    lever_ok = bool(lever_mean[2] > lever_mean[0] + 30 and
                    .40 <= float(lever_mr[..., 2].mean() / 255) <= 1.0 and
                    .18 <= float(lever_mr[..., 1].mean() / 255) <= .60 and
                    float(lever_base.max()) >= 200 and
                    float(lever_normal.std()) >= 1.0)
    checks['lever'] = {
        'mean_rgb': [round(float(v), 2) for v in lever_mean],
        'mean_metallic': round(float(lever_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(lever_mr[..., 1].mean() / 255), 4),
        'white_mark_present': bool(float(lever_base.max()) >= 200),
        'normal_std': round(float(lever_normal.std()), 3),
        'passed': lever_ok,
    }
    passed = all(v['passed'] for v in checks.values())
    return {'passed': passed, 'materials': checks,
            'note': ('Tile/region statistics only; they cannot prove hidden-surface '
                     'colour, exact ink, or scene-light response.')}


# ---------------------------------------------------------------------------
# Geometry / opening / reflection checks
# ---------------------------------------------------------------------------
def silhouette_checks(out):
    expected = {
        'top-eyelet-only': 1,
        'bottom-eyelet-only': 1,
        'eyelet-only': 2,
        'new-front': 2,
        'new-top': 0,
        'new-bottom': 0,
    }
    counts = {}
    passed = True
    for name, want in expected.items():
        with Image.open(out / 'renders' / f'{name}.png') as image:
            _, count = interior_holes(np.asarray(image.getchannel('A')))
        counts[name] = int(count)
        passed = passed and int(count) == want
    return {'expected': expected, 'detected': counts, 'passed': bool(passed),
            'method': ('Enclosed silhouette holes in isolated eyelet renders and '
                       'the full front/top/bottom views. The clevis slot is open, '
                       'so the top/bottom views should not report an enclosed hole.')}


def ray_checks(out, worker):
    scene = trimesh.load(out / 'mesh.glb', force='scene')
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    triangles = np.asarray(combined.triangles, dtype=np.float64)
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0

    def ray_hits(origin, direction):
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        h = np.cross(direction, edge2)
        a = np.einsum('ij,ij->i', edge1, h)
        valid = np.abs(a) > 1e-12
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]
        s = origin - v0
        u = f * np.einsum('ij,ij->i', s, h)
        q = np.cross(s, edge1)
        v = f * (q @ direction)
        t = f * np.einsum('ij,ij->i', edge2, q)
        hit = (valid & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) &
               (t > 1e-6))
        return bool(hit.any())

    top_z = float(worker['top_eyelet_center_z'])
    bottom_z = float(worker['bottom_eyelet_center_z'])
    origins = [[-0.05, 0.0, top_z],
               [-0.05, 0.0, top_z + 0.0080],
               [-0.05, 0.0, bottom_z],
               [-0.05, 0.0, bottom_z + 0.0070]]
    hits = [ray_hits(o, (1.0, 0.0, 0.0)) for o in origins]
    expected = [False, True, False, True]
    return {
        'top_bore_clear': not bool(hits[0]),
        'top_bushing_material_hit': bool(hits[1]),
        'bottom_bore_clear': not bool(hits[2]),
        'bottom_bushing_material_hit': bool(hits[3]),
        'raw_hits': [bool(v) for v in hits],
        'expected': expected,
        'passed': bool([bool(v) for v in hits] == expected),
        'method': ('Ray along the nominal X bore axis at the eyelet centre must miss '
                   'all geometry; a parallel ray at 8.0/7.0 mm offset must hit the '
                   'bushing or plate. This proves a real through-bore, not a painted hole.'),
    }


def _highlight_centroid(image):
    a = np.asarray(image.convert('RGBA')).astype(np.float32)
    mask = a[:, :, 3] >= 128
    if mask.sum() < 100:
        return None
    luma = .2126 * a[:, :, 0] + .7152 * a[:, :, 1] + .0722 * a[:, :, 2]
    threshold = max(120.0, float(np.percentile(luma[mask], 99.5)))
    selected = mask & (luma >= threshold)
    if selected.sum() < 8:
        return None
    ys, xs = np.nonzero(selected)
    return float(xs.mean()), float(ys.mean()), float(luma[selected].mean())


def reflection_checks(out):
    first = Image.open(out / 'renders' / 'new-material-close.png')
    second = Image.open(out / 'renders' / 'new-surface-alternate.png')
    if first.size != second.size:
        raise ValueError('metal light comparison images differ in size')
    a = np.asarray(first.convert('RGBA')).astype(np.float32)
    b = np.asarray(second.convert('RGBA')).astype(np.float32)
    mask = (a[:, :, 3] >= 128) & (b[:, :, 3] >= 128)
    delta = np.abs(a[:, :, :3] - b[:, :, :3]).mean(axis=2)[mask]
    ca, cb = _highlight_centroid(first), _highlight_centroid(second)
    shift = None
    if ca is not None and cb is not None:
        shift = round(float(math.hypot(ca[0] - cb[0], ca[1] - cb[1])), 3)
    mean_diff = float(delta.mean())
    changed = bool((shift is not None and shift > 1.0) or mean_diff > 2.0)
    return {
        'status': 'applicable_metal',
        'metal_response': {
            'mean_abs_luma_difference': round(mean_diff, 4),
            'highlight_centroid_a': None if ca is None else [round(v, 2) for v in ca],
            'highlight_centroid_b': None if cb is None else [round(v, 2) for v in cb],
            'highlight_shift_px': shift,
            'changed': changed,
            'criterion': ('highlight centroid shift > 1 px OR mean visible-metal '
                          'luma difference > 2.0; both values are recorded'),
            'note': ('Same close-up camera, different key-light direction. The air '
                     'can, cap and eyelets are metallic; highlights must move with '
                     'the live light and are not baked into the base colour.'),
        },
        'dielectric_note': ('Only the blue sag O-ring is dielectric rubber; the main '
                            'close-up is dominated by metal and is used for the metal '
                            'response check.'),
    }


# ---------------------------------------------------------------------------
# Reference / source diagnostics
# ---------------------------------------------------------------------------
def _mask_iou(reference_mask, rendered_alpha):
    ref_fit = mask_fit(reference_mask)
    new_fit = mask_fit(np.asarray(rendered_alpha) >= 128)
    if ref_fit is None or new_fit is None:
        return {'iou': None, 'alignment': 'bounding-box fit to 400 px'}
    score = source_fidelity(ref_fit * 255, new_fit * 255)
    score['alignment'] = 'mask bounding-box fit to 400 px; not a calibrated camera'
    return score


def reference_views(out, reference):
    image = Image.open(reference).convert('RGB')
    rows = []
    for name in VIEWS:
        panel = panel_image(image, name)
        ref_mask = reference_mask(panel)
        with Image.open(out / 'renders' / f'new-{name}.png') as new_image:
            new_alpha = new_image.getchannel('A')
        with Image.open(out / 'renders' / f'old-{name}.png') as old_image:
            old_alpha = old_image.getchannel('A')
        new_score = _mask_iou(ref_mask, new_alpha)
        old_score = _mask_iou(ref_mask, old_alpha)
        old_new = _mask_iou(ref_mask, old_alpha)  # filled below with model-to-model
        old_new = source_fidelity(mask_fit(np.asarray(old_alpha) >= 128) * 255,
                                  mask_fit(np.asarray(new_alpha) >= 128) * 255)
        flipped = source_fidelity(mask_fit(np.fliplr(ref_mask)) * 255,
                                  mask_fit(np.asarray(new_alpha) >= 128) * 255)
        rows.append({
            'view': name,
            'reference_new_iou': new_score.get('iou'),
            'reference_new_filled_holes': new_score.get('filled_holes'),
            'reference_old_iou': old_score.get('iou'),
            'old_new_iou': round(float(old_new['iou']), 4),
            'reference_new_iou_mirrored_reference': flipped.get('iou'),
            'reference_new_holes': new_score.get('holes'),
            'note': ('未标定图册；按外接框等比对齐的诊断值，不是测量精度，也不是验收门槛。'
                     '左侧/右侧面板可能是镜像或非一致转台，镜像 IoU 仅作方向不确定性的诊断。'),
        })
    return rows


def reference_ratio_diagnostics(reference):
    image = Image.open(reference).convert('RGB')
    rows = []
    for name in VIEWS:
        panel = panel_image(image, name)
        mask = reference_mask(panel)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        rows.append({'view': name, 'panel': list(PANEL_BOX[name]),
                     'foreground_bbox': [int(xs.min()), int(ys.min()),
                                         int(xs.max()), int(ys.max())],
                     'foreground_fraction': round(float(mask.mean()), 4)})
    return rows


def source_diagnostics(out, alpha_path, worker):
    with Image.open(alpha_path) as alpha_image:
        source_alpha = alpha_image.convert('L')
    source_mask = np.asarray(source_alpha) >= 128
    with Image.open(out / 'renders' / 'new-source.png') as rendered:
        strict = source_fidelity(np.asarray(source_alpha),
                                 np.asarray(rendered.getchannel('A')))
    strict['acceptance_gate'] = False
    rolled = {}
    for roll in (-38.6, 38.6):
        path = out / 'renders' / f'new-source-roll{roll:+.1f}.png'
        if not path.is_file():
            continue
        with Image.open(path) as rendered:
            raw = source_fidelity(np.asarray(source_alpha),
                                  np.asarray(rendered.getchannel('A')))
            aligned = _mask_iou(source_mask, rendered.getchannel('A'))
        raw['acceptance_gate'] = False
        aligned['acceptance_gate'] = False
        rolled[f'{roll:+.1f}'] = {'raw': raw, 'bbox_aligned': aligned}
    best = None
    if rolled:
        best = max(rolled.items(),
                   key=lambda item: item[1]['bbox_aligned'].get('iou') or -1)
    return {
        'strict': strict,
        'rolled': rolled,
        'best_roll': None if best is None else best[0],
        'best_bbox_aligned': None if best is None else best[1]['bbox_aligned'],
        'source_major_axis_below_horizontal_deg': 51.43,
        'note': ('The source photo is a diagonal three-quarter view. The strict '
                 'source camera (roll 0) is not a fair pose comparison. Two rolled '
                 'cameras (±38.6°) are uncalibrated diagnostics only and are never '
                 'acceptance gates.'),
    }


def audit_row(number):
    if not AUDIT.is_file():
        return None
    for line in AUDIT.read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('number') == number:
                return row
    return None


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def _tile_image(path, size):
    image = Image.open(path).convert('RGBA')
    rect = image.getchannel('A').getbbox()
    if rect:
        image = image.crop(rect)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def _panel_tile(panel, mask, size, background='white'):
    ys, xs = np.nonzero(mask)
    if len(xs):
        x0, y0 = max(0, int(xs.min()) - 6), max(0, int(ys.min()) - 6)
        x1, y1 = min(panel.width, int(xs.max()) + 7), min(panel.height, int(ys.max()) + 7)
        panel = panel.crop((x0, y0, x1, y1))
    panel = panel.convert('RGBA')
    panel.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new('RGBA', (size[0] + 10, size[1] + 10), background)
    tile.alpha_composite(panel, ((tile.width - panel.width) // 2,
                                 (tile.height - panel.height) // 2))
    return tile


def make_report_18(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 22)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2240, 2620), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12),
              '18 气压减震器 · 黑色山地车后避震 · 六视图（未标定） / 原模型 / 新候选',
              font=font, fill='#18242b')
    rows = reference_views(out, reference)
    for i, name in enumerate(VIEWS):
        y = 56 + i * 415
        x = 14
        panel = panel_image(reference_image, name)
        mask = reference_mask(panel)
        ref_tile = _panel_tile(panel, mask, (500, 348), 'white')
        row = rows[i]
        board.paste(ref_tile.convert('RGB'), (x, y + 26))
        draw.text((x + 4, y), '参考图册（未标定）', font=small, fill='#27363c')
        for j, (which, label) in enumerate((('old', '原模型（转正/缩放仅作对照）'),
                                            ('new', '新候选'))):
            image = _tile_image(out / 'renders' / f'{which}-{name}.png', (500, 348))
            tile = Image.new('RGBA', (520, 358), '#d9dedd')
            tile.alpha_composite(image, ((520 - image.width) // 2,
                                         (358 - image.height) // 2))
            board.paste(tile.convert('RGB'), (x + 530 + j * 550, y + 26))
            draw.text((x + 534 + j * 550, y), label, font=small, fill='#27363c')
        draw.text((x + 4, y + 386),
                  (f'参考/新诊断 IoU {row["reference_new_iou"]} · 参考/旧 '
                   f'{row["reference_old_iou"]} · 旧/新 {row["old_new_iou"]} · '
                   f'参考镜像/新 {row["reference_new_iou_mirrored_reference"]}'),
                  font=small, fill='#52646b')
    draw.text((20, 2540),
              ('重要：六视图图册未标定且并非同一台实物的严格正交转台；底视面板对下叉耳孔'
               '朝向的描绘与正视/侧视互相矛盾，型号字样在 DRIFT 与 RC-8RC 之间不一致。'
               '以上 IoU 仅是外接框对齐诊断，不是尺寸测量或验收通过。原模型为对照仅做了'
               '转正、缩放与平移，库文件未改动。'),
              font=small, fill='#8a2b20')
    board.save(out / '六视图-新旧对照.jpg', quality=95)
    return rows


def make_detail_board(out):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    items = (
        ('detail-front', '整体正视（材质）'),
        ('detail-back', '整体背视（材质）'),
        ('detail-side', '整体侧视（材质）'),
        ('detail-top', '顶视：耳叉、调节旋钮与拨杆'),
        ('detail-bottom', '底视：下叉耳与蓝环'),
        ('detail-top-eyelet', '上叉耳 3/4 局部'),
        ('detail-top-eyelet-front', '上叉耳通孔（正视）'),
        ('detail-top-eyelet-top', '上叉耳 C 形开口（顶视）'),
        ('detail-bottom-eyelet', '下叉耳 3/4 局部'),
        ('detail-bottom-eyelet-front', '下叉耳通孔（正视）'),
        ('detail-adjusters', '红色回弹旋钮 / 蓝色三档拨杆'),
        ('detail-blue-lever', '蓝色拨杆弯折段与红色滚花座'),
        ('detail-red-knob', '红色滚花旋钮'),
        ('detail-cap', '顶盖与 Rebound 印字'),
        ('detail-label-front', '正面 DRIFT / DNM 贴标'),
        ('detail-label-back', '背面 DNM / RC-8RC 贴标'),
        ('detail-sag-ring', '蓝色 O 形下沉指示环与密封头'),
        ('detail-shaft-eyelet', '下轴与下叉耳过渡'),
        ('detail-can-grooves', '气罐下缘环槽'),
        ('detail-surface', '黑色阳极氧化表面与磨损'),
        ('new-clay-front', '灰模正视'),
        ('new-clay-close', '灰模 3/4 局部'),
    )
    tile_w, tile_h = 300, 760
    cols = 8
    rows = (len(items) + cols - 1) // cols
    board = Image.new('RGB', (tile_w * cols + 20, 60 + rows * (tile_h + 40)),
                      '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '18 气压减震器 · 关键局部检查（新候选；灰模为几何对照）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = 10 + (i % cols) * tile_w
        y = 55 + (i // cols) * (tile_h + 40)
        path = out / 'renders' / f'{name}.png'
        if not path.is_file():
            continue
        image = _tile_image(path, (tile_w - 18, tile_h - 30))
        tile = Image.new('RGBA', (tile_w - 8, tile_h), '#d9dedd')
        tile.alpha_composite(image, ((tile.width - image.width) // 2,
                                     (tile.height - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, board.height - 30),
              ('这些局部图用于人工检查结构；不能证明未观察面、内部气室/阻尼结构、'
               '耳孔轴向的真实装配、螺纹/卡扣配合、落稳或现场尺度。'),
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, silhouette, rays):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    items = (
        ('top-eyelet-only', '上叉耳隔离（含钢衬套）'),
        ('bottom-eyelet-only', '下叉耳隔离（含钢衬套）'),
        ('eyelet-only', '上下叉耳同时隔离'),
        ('top-eyelet-only-3q', '上叉耳 3/4 隔离'),
        ('bottom-eyelet-only-3q', '下叉耳 3/4 隔离'),
        ('adjusters-only', '调节旋钮/拨杆隔离'),
        ('can-label-only', '气罐与贴标隔离'),
        ('seal-shaft-only', '密封头/蓝环/下轴隔离'),
        ('top-view-all', '整体顶视（耳叉开口）'),
        ('new-clay-front', '灰模正视'),
    )
    board = Image.new('RGB', (5 * 360 + 20, 2 * 930 + 70), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '18 气压减震器 · 隔离结构与真实开口检查',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = 10 + (i % 5) * 360
        y = 55 + (i // 5) * 930
        image = _tile_image(out / 'renders' / f'{name}.png', (330, 800))
        tile = Image.new('RGBA', (348, 815), '#d9dedd')
        tile.alpha_composite(image, ((348 - image.width) // 2,
                                     (815 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
        detail = ''
        if name == 'top-eyelet-only':
            detail = f'封闭孔 {silhouette["detected"].get(name)}/1'
        elif name == 'bottom-eyelet-only':
            detail = f'封闭孔 {silhouette["detected"].get(name)}/1'
        elif name == 'eyelet-only':
            detail = f'封闭孔 {silhouette["detected"].get(name)}/2'
        elif name == 'top-view-all':
            detail = f'封闭孔 {silhouette["detected"].get("new-top")}/0'
        if detail:
            draw.text((x + 6, y + 815), detail,
                      font=small, fill='#1d5c3a' if silhouette['passed'] else '#8a2b20')
    ray_text = ('射线：上孔通透 {}，上衬套材料命中 {}；下孔通透 {}，下衬套材料命中 {}。'
                '隔离渲染只证明这些孔在对应视角是真实贯通开口，不证明孔深、衬套配合或'
                '装配方向。').format(
        '通过' if rays['top_bore_clear'] else '未通过',
        '通过' if rays['top_bushing_material_hit'] else '未通过',
        '通过' if rays['bottom_bore_clear'] else '未通过',
        '通过' if rays['bottom_bushing_material_hit'] else '未通过')
    draw.text((20, board.height - 42), ray_text, font=small,
              fill='#1d5c3a' if rays['passed'] else '#8a2b20')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out, reflection):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '18 气压减震器 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选：黑色阳极氧化 / 贴标 / 钢衬套'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 830))
        tile = Image.new('RGBA', (680, 850), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (850 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), ('同一中性光与相机；高光来自实时照明，不烘焙在底色中。'
                          '原模型为单图重建的顶点色，不代表旧模型整体通过。'),
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 960), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '18 气压减震器 · 变光检查（同视角，主光位置 A/B）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光 A · 正视'),
                                       ('new-light-alternate', '主光 B · 正视'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 780))
        tile = Image.new('RGBA', (680, 800), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (800 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 900), ('主光移动后高光和明暗应随视角变化；本图只检查反射响应，'
                          '不证明现场光照方向。'), font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    metal = reflection['metal_response']
    board = Image.new('RGB', (1440, 960), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '18 气压减震器 · 金属反射变光对照（同相机，主光 A/B）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光 A'),
                                       ('new-surface-alternate', '主光 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 780))
        tile = Image.new('RGBA', (680, 800), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (800 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 900),
              (f'可见金属平均亮度差 {metal["mean_abs_luma_difference"]}；'
               f'高光质心偏移 {metal["highlight_shift_px"]} px；'
               f'{"通过" if metal["changed"] else "未通过"}。'
               '高光随实时主光移动，不是烘焙底色；这不是现场光照验收。'),
              font=small, fill='#1d5c3a' if metal['changed'] else '#8a2b20')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_source_board(out, source_diag, clean, rust):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1500, 1030), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '18 气压减震器 · 原视角审计（照片 3/4 视角，非标定）',
              font=font, fill='#18242b')
    best_roll = source_diag.get('best_roll') or '-38.6'
    rolled_path = out / 'renders' / f'new-source-roll{best_roll}.png'
    items = (
        (clean, '原貌照片（3/4 视角）', 'photo'),
        (rust, '做锈素材（同视角）', 'photo'),
        (rolled_path, f'新候选 · 倾斜相机诊断 roll {best_roll}°', 'render'),
    )
    for i, (path, label, kind) in enumerate(items):
        x = 20 + i * 490
        if kind == 'photo':
            image = Image.open(path).convert('RGBA')
            white = Image.new('RGBA', image.size, (255, 255, 255, 255))
            white.alpha_composite(image)
            image = white.convert('RGB')
            rect = image.getbbox()
            if rect:
                image = image.crop(rect)
            image.thumbnail((450, 820), Image.Resampling.LANCZOS)
            tile = Image.new('RGB', (460, 830), 'white')
            tile.paste(image, ((460 - image.width) // 2,
                               (830 - image.height) // 2))
        else:
            image = _tile_image(path, (450, 820))
            tile = Image.new('RGBA', (460, 830), '#d9dedd')
            tile.alpha_composite(image, ((460 - image.width) // 2,
                                         (830 - image.height) // 2))
            tile = tile.convert('RGB')
        board.paste(tile, (x, 60))
        draw.text((x + 4, 34), label, font=small, fill='#27363c')
    strict = source_diag['strict']
    rolled = source_diag['rolled'].get(best_roll, {})
    draw.text((20, 920),
              (f'严格原相机（roll 0）IoU {strict.get("iou")}（照片为斜放 3/4 视角，'
               f'该姿态不匹配，仅诊断）；倾斜诊断 roll {best_roll}° 原始 IoU '
               f'{rolled.get("raw", {}).get("iou")}、外接框对齐 IoU '
               f'{rolled.get("bbox_aligned", {}).get("iou")}。相机 roll 和缩放均为'
               '未标定最佳努力，不是验收门槛。'),
              font=small, fill='#8a2b20')
    board.save(out / '原视角-对照.jpg', quality=95)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    tile = 250
    board = Image.new('RGB', (tile * 6, tile * 4 + 34), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '18 气压减震器 · 24 方向转台（中性光；透明背景）',
              font=font, fill='#27363c')
    for i in range(24):
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png').convert('RGBA')
        image.thumbnail((tile - 16, tile - 16), Image.Resampling.LANCZOS)
        cell = Image.new('RGBA', (tile, tile), '#d9dedd')
        cell.alpha_composite(image, ((tile - image.width) // 2,
                                     (tile - image.height) // 2))
        board.paste(cell.convert('RGB'), ((i % 6) * tile, 34 + (i // 6) * tile))
    board.save(out / '转台-24方向.jpg', quality=94)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def required_files(out, draft):
    files = [out / 'model.blend', out / 'mesh.glb',
             out / 'can_base.png', out / 'can_mr.png', out / 'can_normal.png',
             out / 'cap_base.png', out / 'cap_mr.png', out / 'cap_normal.png',
             out / 'black_base.png', out / 'black_mr.png', out / 'black_normal.png',
             out / 'steel_base.png', out / 'steel_mr.png', out / 'steel_normal.png',
             out / 'lever_base.png', out / 'lever_mr.png', out / 'lever_normal.png',
             out / 'result.json', out / 'hashes.json', out / 'worker-result.json',
             out / 'request.json']
    if not draft:
        files += [out / '六视图-新旧对照.jpg', out / '局部检查.jpg',
                  out / '结构开口-隔离检查.jpg', out / '材质-同光照对照.jpg',
                  out / '材质-变光检查.jpg', out / '金属反射-变光对照.jpg',
                  out / '原视角-对照.jpg', out / '转台-24方向.jpg',
                  out / '浏览器核验.json', out / '检查页.html']
        files += [out / 'renders' / f'{which}-{view}.png'
                  for which in ('old', 'new') for view in VIEWS]
        files += [out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
    files += [out / 'renders' / f'new-{view}.png' for view in VIEWS]
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', type=int, default=18)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--draft', action='store_true', help='quick render')
    parser.add_argument('--library', type=Path,
                        default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    args = parser.parse_args()
    if args.number != 18:
        parser.error('this entry supports item 18 only')
    args.reference = (args.reference or find_reference()).resolve()
    if args.out is None:
        args.out = ROOT / ('outputs/18-shock-draft1' if args.draft
                           else 'outputs/18气压减震器-六视图重建-v1')
    args.out, args.library, args.blender = (x.resolve() for x in
                                            (args.out, args.library, args.blender))
    if digest(args.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = next(r for r in (json.loads(line) for line in
                           args.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == 18)
    old_mesh = ROOT / OLD_REL
    alpha_path = ROOT / ALPHA_REL
    clean = ROOT / CLEAN_REL
    rust = ROOT / RUST_REL
    source_view = ROOT / SOURCE_VIEW_REL
    for path, expected, label in ((old_mesh, row['mesh_sha256'], 'old mesh'),
                                  (rust, row['rust_sha256'], 'rust image'),
                                  (clean, row['clean_sha256'], 'clean image'),
                                  (alpha_path, row['source_alpha_sha256'], 'source alpha')):
        if not path.is_file():
            raise FileNotFoundError(f'missing current-worktree input: {path}')
        if digest(path) != expected:
            raise ValueError(f'{label} hash differs from the model library')
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    if not len(xs):
        raise ValueError('source alpha is empty')
    source_alpha_bbox = [int(xs.min()), int(ys.min()),
                         int(xs.max()), int(ys.max())]
    config_path = HERE / 'shock18_config.json'
    worker_path = HERE / 'mesh_rebuild_18_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(args.blender), '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    signature_input = {
        'reference': digest(args.reference), 'mesh': row['mesh_sha256'],
        'rust': row['rust_sha256'], 'clean': row['clean_sha256'],
        'source_alpha': digest(alpha_path),
        'source_camera': row['reconstruction']['source_camera'],
        'config': digest(config_path), 'builder': digest(Path(__file__)),
        'worker': digest(worker_path),
        'helpers': {name: digest(HERE / name) for name in
                    ('blender_worker.py', 'mesh_checks.py', 'mesh_carve.py',
                     'mesh_rebuild_item.py')},
        'blender': version, 'python': sys.version.split()[0],
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'pillow': PIL.__version__, 'trimesh': trimesh.__version__,
        'draft': args.draft,
    }
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if args.out.exists() and any(args.out.iterdir()):
        record = args.out / 'result.json'
        mesh = args.out / 'mesh.glb'
        if record.is_file() and mesh.is_file():
            old_record = json.loads(record.read_text(encoding='utf8'))
            if old_record.get('signature') == signature and \
                    old_record.get('mesh_sha256') == digest(mesh) and \
                    all(path.is_file() for path in required_files(args.out, args.draft)):
                print(json.dumps({'status': 'review', 'cache_hit': True,
                                  'out': str(args.out)}, ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    args.out.mkdir(parents=True, exist_ok=True)
    textures = {}
    textures.update(build_can_textures(rust, args.out, config))
    textures.update(build_cap_textures(rust, args.out, config))
    textures.update(build_black_textures(rust, args.out, config))
    textures.update(build_steel_textures(args.out, config))
    textures.update(build_lever_textures(args.out, config))
    request = {
        'out': str(args.out), 'original_mesh': str(old_mesh),
        'textures': textures, 'config': config,
        'source_camera': row['reconstruction']['source_camera'],
        'source_alpha_bbox': source_alpha_bbox, 'draft': args.draft,
    }
    (args.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2),
                                           encoding='utf8')
    with (args.out / 'blender.log').open('w', encoding='utf8') as log:
        process = subprocess.run(
            [str(args.blender), '--background', '--factory-startup',
             '--python-exit-code', '1', '--python', str(worker_path), '--',
             str(args.out / 'request.json')],
            stdout=log, stderr=subprocess.STDOUT, timeout=10800)
    if process.returncode:
        raise RuntimeError(f'Blender failed; see {args.out / "blender.log"}')

    worker = json.loads((args.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(18)
    mesh_audit = inspect_mesh(args.out / 'mesh.glb')
    parts = load_parts(args.out / 'mesh.glb')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'air_cans': sum(name == 'Air can' for name in names),
        'top_caps': sum(name == 'Top cap' for name in names),
        'top_cap_seams': sum(name == 'Top cap seam' for name in names),
        'top_yokes': sum(name == 'Top yoke' for name in names),
        'top_eyelet_plates': sum(name.startswith('Top eyelet plate') for name in names),
        'top_bushings': sum(name == 'Top eyelet bushing' for name in names),
        'bottom_yokes': sum(name == 'Bottom yoke' for name in names),
        'bottom_eyelet_plates': sum(name.startswith('Bottom eyelet plate') for name in names),
        'bottom_bushings': sum(name == 'Bottom eyelet bushing' for name in names),
        'shafts': sum(name == 'Damper shaft' for name in names),
        'shaft_collars': sum(name == 'Shaft collar' for name in names),
        'seal_heads': sum(name == 'Seal head' for name in names),
        'seal_ribs': sum(name.startswith('Seal rib ') for name in names),
        'sag_rings': sum(name == 'Sag indicator ring' for name in names),
        'sag_screws': sum(name == 'Sag ring screw' for name in names),
        'red_knobs': sum(name == 'Red rebound knob' for name in names),
        'red_knob_bosses': sum(name == 'Red knob boss' for name in names),
        'red_knob_tops': sum(name == 'Red knob top' for name in names),
        'red_knob_grooves': sum(name == 'Red knob top groove' for name in names),
        'blue_collars': sum(name == 'Blue lever red collar' for name in names),
        'blue_heads': sum(name == 'Blue lever head' for name in names),
        'blue_arms': sum(name.startswith('Blue lever arm') for name in names),
        'blue_marks': sum(name == 'Blue lever top mark' for name in names),
        'note': ('Part presence proves only that a feature was modelled, not its '
                 'size, fit, hidden faces, adjuster function or internal damper.'),
    }
    expected = {
        'air_cans': 1, 'top_caps': 1, 'top_cap_seams': 1, 'top_yokes': 1,
        'top_eyelet_plates': 2, 'top_bushings': 1,
        'bottom_yokes': 1, 'bottom_eyelet_plates': 2, 'bottom_bushings': 1,
        'shafts': 1, 'shaft_collars': 1, 'seal_heads': 1, 'seal_ribs': 2,
        'sag_rings': 1, 'sag_screws': 1, 'red_knobs': 1, 'red_knob_bosses': 1,
        'red_knob_tops': 1, 'red_knob_grooves': 1, 'blue_collars': 1,
        'blue_heads': 1, 'blue_arms': 2, 'blue_marks': 1,
    }
    structural_ok = all(structural[key] == value for key, value in expected.items())
    texture_ok = texture_checks(textures, config)
    silhouette = silhouette_checks(args.out)
    rays = ray_checks(args.out, worker)
    reflection = reflection_checks(args.out)
    views = make_report_18(args.out, args.reference)
    reference_ratios = reference_ratio_diagnostics(args.reference)
    source_diag = source_diagnostics(args.out, alpha_path, worker)
    browser = {'status': 'not_run', 'reason': 'draft mode' if args.draft else 'pending'}
    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 27 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'structural_counts': bool(structural_ok),
        'eyelet_bore_silhouettes': bool(silhouette['passed']),
        'eyelet_bore_rays': bool(rays['passed']),
        'material_tiles': bool(texture_ok['passed']),
        'metal_reflection': bool(reflection['metal_response']['changed']),
        'source_fidelity': bool(source_diag['strict'].get('passed', False)),
        'reference_diagnostic_only': True,
        'browser_page': browser.get('status') == 'passed',
        'manual_only': True,
    }
    report = {
        'version': config['version'],
        'number': 18,
        'asset_id': row['asset_id'],
        'name': row['file'],
        'status': 'review',
        'model_review': 'pending',
        'scene_validation': 'pending',
        'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_six_view_reference',
        'reference_role': 'visual_match_unverified',
        'method': ('parametric_part_rebuild_from_source_photo_with_uncalibrated_'
                   'six_view_reference'),
        'reference': str(args.reference),
        'reference_sha256': digest(args.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'model_identity': config['proportion_source']['model_identity'],
        'original_mesh': str(old_mesh),
        'original_mesh_sha256': row['mesh_sha256'],
        'original_clean': str(clean),
        'original_clean_sha256': row['clean_sha256'],
        'original_rust': str(rust),
        'original_rust_sha256': row['rust_sha256'],
        'source_view': str(source_view),
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9489 passed only the numeric outer '
            'silhouette and filled one hole. The six-view audit shows a blurred, '
            'low-frequency surface, fused eyelet/knob/lever details and no legible '
            'bores; the old mesh is therefore not reused for the candidate base '
            'geometry.'),
        'catalog_evidence': {
            'category': '黑色山地车后避震 / 气压减震器',
            'brand_visible': 'DNM (lettering partly curved away from the camera)',
            'labels_visible': 'DRIFT on the front, DNM / RC-8RC on the back',
            'model': '具体型号未确认：同一图册的正/背面板字样不一致',
            'note': ('No external product page was used to claim measured dimensions; '
                     'the photo and the uncalibrated atlas are the only evidence.'),
        },
        'eye_to_eye_m': config['proportion_source']['eye_to_eye_m'],
        'eye_to_eye_note': ('190 mm is a plausible mountain-bike rear-shock size used '
                            'to scale the photo ratios; it is not measured.'),
        'inputs': signature_input,
        'signature': signature,
        'mesh': str(args.out / 'mesh.glb'),
        'mesh_sha256': digest(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'),
        'blend_sha256': digest(args.out / 'model.blend'),
        'textures': textures,
        'texture_checks': texture_ok,
        'material_profile': config['material_profile'],
        'source_fidelity_diagnostic': source_diag['strict'],
        'source_fidelity_rolled_diagnostics': source_diag['rolled'],
        'source_fidelity_best_roll': source_diag['best_roll'],
        'source_fidelity_best_bbox_aligned': source_diag['best_bbox_aligned'],
        'source_diagnostic_note': source_diag['note'],
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratios,
        'ratio_note': ('The six-view sheet is not calibrated and is not a consistent '
                       'turntable; reference/model IoU is a bounding-box-aligned '
                       'silhouette diagnostic, not a measurement or acceptance gate.'),
        'mesh_audit': mesh_audit,
        'structural_checks': structural,
        'structural_expected': expected,
        'isolated_opening_checks': {'silhouettes': silhouette, 'rays': rays},
        'rendered_silhouette_checks': silhouette,
        'reflection_checks': reflection,
        'metal_reflection_check': {'status': reflection['status'],
                                   **reflection['metal_response']},
        'checks': checks,
        'browser_check': browser,
        'worker': worker,
        'draft': args.draft,
        'limitations': [
            'The six-view atlas is uncalibrated and internally inconsistent: the bottom panel draws the lower eyelet bore as if it faced the camera while the front/side panels show a horizontal bore axis; the front/back panels show DRIFT and RC-8RC lettering for what may be one wrapped label or two variants.',
            'All dimensions follow photo/atlas pixel ratios and a plausible 190 mm eye-to-eye size; they are not measured. Air-can diameter, shaft diameter, eyelet bore/bushing sizes and adjuster geometry are inferred.',
            'The DNM / DRIFT / RC-8RC / SUSPENSION decal and the Rebound / lever prints are redrawn approximations, not exact artwork; typography, panel seam positions and ink colours may differ.',
            'The upper and lower clevis plates are modelled as two plates with a steel bushing; the real plate spacing, bushing material, bolt/spacer fit and bore depth are not measured.',
            'The blue three-position lever is modelled as a bent arm on a knurled red collar; its detent positions, internal valve, exact angle and whether the lever in the photo is in the LOCK, PEDAL or OPEN position are inferred.',
            'The blue sag O-ring and its small screw are modelled from the visible ring; whether the screw is a marker, bleed port or fastener is not confirmed.',
            'The air can and seal head are closed solids; the internal air spring, IFP, damper shim stack, oil and shaft seals are not represented.',
            'Knurl tooth counts, groove counts and surface grime are approximations synthesised from small observed patches; they are not a pixel-accurate transfer.',
            'Assemblies intentionally overlap at the yoke/cap, knob/plate, arm/cap and seal-head/can interfaces; BVH checks only same-part self-intersections, not part-to-part fit.',
            'The old model was rotated/scaled/translated for display comparison only; that transformation is not a repair and the library file is untouched.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the source photo and atlas, especially eyelet bore sizes, clevis slot, top-cap adjusters, blue lever path, sag ring and label wrap seams.',
            'Confirm the exact model identity and decal artwork (DRIFT vs RC-8RC) from a physical sample or a manufacturer drawing; the current lettering is an approximation.',
            'Measure eye-to-eye length, air-can and shaft diameters, eyelet bore/bushing dimensions, plate spacing, lever angle and adjuster travel.',
            'Confirm hidden geometry: internal air spring/damper, seal head construction, bushing material and whether the lower eyelet is a clevis or a single strap.',
            'Verify the metal reflection under the final scene lighting; the current check only proves highlight movement between two studio key-light directions.',
            'Verify coordinate system, scale, stability, floor contact and placement in the manual review page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated orthographic front/side/top views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 19,
    }
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf8')
    if not args.draft:
        make_detail_board(args.out)
        make_isolated_board(args.out, silhouette, rays)
        make_material_boards(args.out, reflection)
        make_source_board(args.out, source_diag, clean, rust)
        make_turntable(args.out)
        from review18_page import build_page, browser_check
        build_page(args.out)
        browser = browser_check(args.out)
        checks['browser_page'] = browser.get('status') == 'passed'
        report['checks'] = checks
        report['browser_check'] = browser
        (args.out / 'result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        # Rebuild the page with the final browser result reflected in the summary,
        # then run the browser check a second time so its record and screenshot
        # show the final page (the first screenshot necessarily predates the
        # browser result it is producing).
        build_page(args.out)
        browser = browser_check(args.out)
        checks['browser_page'] = browser.get('status') == 'passed'
        report['checks'] = checks
        report['browser_check'] = browser
        (args.out / 'result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        build_page(args.out)
    files = [path for path in sorted(args.out.rglob('*')) if path.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_18.py',
              'files': {str(path.relative_to(args.out)).replace('\\', '/'): digest(path)
                        for path in files}}
    hashes['files']['hashes.json'] = 'self'
    (args.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                          encoding='utf8')
    progress = {
        'number': 18,
        'name': row['file'],
        'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'],
        'out': str(args.out),
        'result': str(args.out / 'result.json'),
        'mesh': str(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(args.out / '六视图-新旧对照.jpg'),
        'detail_board': str(args.out / '局部检查.jpg'),
        'isolated_board': str(args.out / '结构开口-隔离检查.jpg'),
        'checks': checks,
        'texture_checks': texture_ok,
        'isolated_opening_checks': report['isolated_opening_checks'],
        'reference_views': views,
        'source_fidelity_diagnostic': source_diag['strict'],
        'source_fidelity_best_bbox_aligned': source_diag['best_bbox_aligned'],
        'reference_role': 'visual_match_unverified',
        'manual_only': True,
        'pending_review': report['pending_review'],
        'next_number': 19,
        'notes': ('The six-view atlas is an uncalibrated visual-match reference; the '
                  'source photo is the primary shape evidence. The old model and other '
                  'items were not modified. Candidate is manual_only; scene placement, '
                  'stability and formal compositing are not validated.'),
    }
    (args.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                            encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '18-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(args.out), 'checks': checks,
                      'source_fidelity': source_diag['strict'],
                      'best_bbox_aligned': source_diag['best_bbox_aligned']},
                     ensure_ascii=False), flush=True)


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


if __name__ == '__main__':
    main()
