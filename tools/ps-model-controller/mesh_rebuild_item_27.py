"""Build the review-only part-based candidate for item 27 (motorcycle rear shock).

The source photograph is a three-quarter view of a black/blue YU-AN-style rear
shock absorber: a black anodized damper body, a blue coil spring, a clevis top
mount, a U-shaped bottom clevis with a through-bolt, a braided stainless hose
and a piggyback reservoir with a knurled blue cap and silver band.  The old
single-image TripoSR mesh is a soft shell whose spring, body and reservoir
fuse; it is retained only as a comparison.

The candidate never replaces the library mesh, stays manual-only and never
enters automatic placement or the formal composite.  A new output directory is
required for the final run; drafts go under work/.
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
from scipy.spatial import cKDTree

from scene_v2 import digest
from mesh_checks import source_fidelity
from mesh_rebuild_item import _tile_detail, mask_fit, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 27
VERSION = 'shock27-parametric-v1'
REFERENCE_NAME = '黑蓝摩托车后减震器六视图展示.png'
REFERENCE_SHA256 = '059cc68bde6adbd04c2cf95d5dea603b97a16db02c58552c8f12ee0244e3f2c7'
SOURCE_SIZE = (1254, 1254)
RUST_SIZE = (1254, 1254)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
LABEL_FONT_ITALIC = 'C:/Windows/Fonts/arialbi.ttf'
TILE_NAMES = {
    0: '黑色阳极氧化', 1: '蓝色弹簧漆', 2: '蓝色阳极氧化', 3: '镀铬杆',
    4: '钢/编织软管', 5: '深色钢/内六角', 6: '黑色橡胶', 7: '锈蚀参考',
    8: '青铜色导向环', 9: '银色中段带', 10: '编织软管', 11: '备用',
    12: '备用', 13: '备用', 14: '备用', 15: '备用',
}
METAL_TILES = ('black_ano', 'blue_ano', 'chrome', 'steel', 'dark_steel',
               'bronze', 'silver_band')
DIELECTRIC_TILES = ('blue_spring', 'rubber', 'rust', 'spare11',
                    'spare12', 'spare13', 'spare14', 'spare15')
DETAIL_NAMES = ('top-eye', 'top-bolt', 'label', 'spring-upper', 'spring-lower',
                'bottom-clevis', 'hose-top', 'hose-run', 'reservoir-banjo',
                'reservoir-cap', 'overall-3q')
ISOLATION_NAMES = ('top-eye-only', 'bottom-clevis-only', 'spring-only',
                   'reservoir-only', 'hose-only', 'new-material-front',
                   'new-clay-front', 'new-light-alternate', 'metal-only-light-a',
                   'metal-only-light-b')


def find_reference():
    """Use the local copy inside this worktree, never the old desktop path."""
    path = ROOT / 'references/parallel-six-view' / REFERENCE_NAME
    if not path.is_file() or digest(path) != REFERENCE_SHA256:
        raise FileNotFoundError('local six-view reference is missing or has changed')
    return path


def local_path(value):
    """Map a stored library path that still names the old main tree onto this worktree."""
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


def brushed_field(tile, rng, sigma=0.8):
    line = rng.normal(size=(1, tile)).astype(np.float32)
    field = np.repeat(line, tile, axis=0)
    field = cv2.blur(field, (3, 1))
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=0)
    field = (field - field.mean()) / max(1e-6, float(field.std()))
    return field


def safe_detail(image, rect, tile):
    try:
        return cv2.resize(_tile_detail(image, rect, size=tile), (tile, tile),
                          interpolation=cv2.INTER_CUBIC)
    except ValueError:
        return np.zeros((tile, tile), np.float32)


def make_label_textures(out):
    """Draw the visible YU-AN cap label and a small silver-band etching.

    The text is copied from the photograph but the typeface and wear are
    approximate; the original artwork and exact font are not available.
    """
    rng = np.random.default_rng(27027)
    # Cap label: black field, cyan border/checker strip, white italic wordmark.
    w, h = 1024, 256
    label = Image.new('RGB', (w, h), (10, 11, 13))
    d = ImageDraw.Draw(label)
    cyan = (72, 205, 226)
    white = (235, 238, 233)
    d.rectangle((8, 8, w - 9, h - 9), outline=cyan, width=7)
    d.rectangle((18, 18, w - 19, h - 19), outline=(18, 22, 26), width=3)
    for y in (22, h - 44):
        for i in range(0, w - 40, 18):
            if (i // 18) % 2 == 0:
                d.rectangle((20 + i, y, 20 + i + 16, y + 20), fill=cyan)
    font = ImageFont.truetype(LABEL_FONT_ITALIC, 150)
    text = 'YU-AN'
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1] - 2
    d.text((x + 5, y + 5), text, font=font, fill=(8, 18, 24))
    d.text((x + 2, y + 2), text, font=font, fill=cyan)
    d.text((x, y), text, font=font, fill=white)
    # Wear: a few grey scratches and small chipped spots.
    for _ in range(26):
        x0 = int(rng.integers(20, w - 40))
        y0 = int(rng.integers(30, h - 30))
        x1 = x0 + int(rng.integers(4, 42))
        y1 = y0 + int(rng.integers(-4, 5))
        value = int(rng.integers(32, 72))
        d.line((x0, y0, x1, y1), fill=(value, value, value), width=1)
    for _ in range(18):
        x0 = int(rng.integers(24, w - 24))
        y0 = int(rng.integers(30, h - 30))
        r = int(rng.integers(1, 4))
        d.ellipse((x0, y0, x0 + r, y0 + r), fill=(52, 49, 43))
    label.save(out / 'label_cap.png')

    # Silver band: fine vertical brushing with a small dark laser mark.
    w2, h2 = 1024, 128
    band = np.zeros((h2, w2, 3), np.uint8)
    base = np.linspace(168, 193, h2)[:, None, None]
    base = np.repeat(base, w2, axis=1)
    brush = rng.normal(size=(1, w2)).astype(np.float32)
    brush = cv2.GaussianBlur(brush, (1, 0), sigmaX=0)
    base = base * (1 + brush[:, :, None] * .012)
    band[:, :, :] = np.clip(base, 0, 255).astype(np.uint8)
    im = Image.fromarray(band)
    d2 = ImageDraw.Draw(im)
    d2.rectangle((0, 4, w2 - 1, 6), fill=(105, 107, 108))
    d2.rectangle((0, h2 - 7, w2 - 1, h2 - 5), fill=(105, 107, 108))
    font2 = ImageFont.truetype(LABEL_FONT_ITALIC, 68)
    text2 = 'YU-AN'
    bbox2 = d2.textbbox((0, 0), text2, font=font2)
    tw2, th2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    d2.text(((w2 - tw2) // 2 - bbox2[0], (h2 - th2) // 2 - bbox2[1]),
            text2, font=font2, fill=(46, 48, 50))
    im.save(out / 'label_band.png')
    return {'label_cap': str(out / 'label_cap.png'),
            'label_band': str(out / 'label_band.png'),
            'note': ('YU-AN 文字按原照可见内容复制；字体、磨损和银色带刻字为近似绘制，'
                     '不是原厂矢量稿。')}


def make_atlases(rust_path, original_path, out, profile):
    rust = Image.open(rust_path).convert('RGBA')
    if rust.size != RUST_SIZE:
        raise ValueError('27 做锈素材尺寸改变，采样坐标需要重新核对')
    original = Image.open(original_path).convert('RGBA')
    if original.size != SOURCE_SIZE:
        raise ValueError('27 原图尺寸改变，测量坐标需要重新核对')
    size = profile['texture']['size']
    tile = profile['texture']['tile']
    tiles = profile['texture']['tiles']
    rng = np.random.default_rng(27027)
    base = np.zeros((size, size, 3), np.uint8)
    mr = np.zeros((size, size, 3), np.uint8)
    normal = np.zeros((size, size, 3), np.uint8)
    normal[:, :, 2] = 255
    fine = cv2.resize(rng.random((360, 360)).astype(np.float32), (tile, tile),
                      interpolation=cv2.INTER_CUBIC)
    coarse = cv2.resize(rng.random((28, 28)).astype(np.float32), (tile, tile),
                        interpolation=cv2.INTER_CUBIC)
    blotch = cv2.resize(rng.random((96, 96)).astype(np.float32), (tile, tile),
                        interpolation=cv2.INTER_CUBIC)
    micro = cv2.resize(rng.random((512, 512)).astype(np.float32), (tile, tile),
                       interpolation=cv2.INTER_LINEAR)
    brush = brushed_field(tile, rng)
    brush2 = brushed_field(tile, rng, sigma=1.5)
    # rust palette from the corroborating rust photograph, not a fixed swatch
    a = np.asarray(rust)
    mask = a[:, :, 3] >= 128
    rr, gg, bb = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    brown = np.logical_and.reduce([mask, rr >= bb + 15, rr >= gg - 5,
                                   rr >= 70, gg <= 160])
    strong = np.logical_and.reduce([brown, rr >= bb + 45])
    if strong.sum() >= 1000:
        rust_rgb = np.median(a[:, :, :3][strong], axis=0).astype(np.float32)
    elif brown.sum() >= 500:
        rust_rgb = np.median(a[:, :, :3][brown], axis=0).astype(np.float32)
    else:
        rust_rgb = np.array([134, 101, 75], np.float32)
    dark_oxide = rust_rgb * .52
    black_detail = safe_detail(rust, (390, 300, 440, 350), tile)
    blue_detail = safe_detail(rust, (420, 450, 480, 510), tile)
    ano_detail = safe_detail(rust, (720, 1000, 760, 1040), tile)
    steel_detail = safe_detail(rust, (445, 346, 500, 390), tile)
    band_detail = safe_detail(rust, (780, 1030, 830, 1080), tile)
    bronze_detail = safe_detail(rust, (395, 1080, 425, 1110), tile)
    rubber_detail = safe_detail(rust, (500, 600, 540, 640), tile)
    # Cross-hatched height field for the braided hose; this is a normal/roughness
    # cue only, not a per-strand geometry.
    yy, xx = np.mgrid[0:tile, 0:tile].astype(np.float32)
    braid_detail = (.30 * np.sin((xx + yy) * math.tau / 9.0) +
                    .30 * np.sin((xx - yy) * math.tau / 9.0))

    def put(idx, color, roughness, metallic, height=None, *, clip_min=20):
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
            normal[sl] = normal_from_height(height, .75)

    specs = {
        'black_ano': (black_detail, 'black_ano'),
        'blue_spring': (blue_detail, 'blue_spring'),
        'blue_ano': (ano_detail, 'blue_ano'),
        'chrome': (steel_detail, 'chrome'),
        'steel': (steel_detail, 'steel'),
        'dark_steel': (black_detail, 'dark_steel'),
        'rubber': (rubber_detail, 'rubber'),
        'rust': (np.zeros((tile, tile), np.float32), 'rust'),
        'bronze': (bronze_detail, 'bronze'),
        'silver_band': (band_detail, 'silver_band'),
        'braid': (braid_detail, 'braid'),
    }
    for name, (detail, key) in specs.items():
        idx = tiles[name]
        mat = profile['material_profile'][key]
        base_rgb = np.asarray(mat['base_rgb'], np.float32)[None, None, :]
        amount = float(mat.get('rust_amount', 0.0))
        bright = (1 + detail[:, :, None] * .0024 +
                  (coarse[:, :, None] - .5) * .050 +
                  (fine[:, :, None] - .5) * .020)
        if name in ('chrome', 'steel', 'bronze', 'silver_band'):
            bright = bright * (1 + brush[:, :, None] * .020)
        if name in ('blue_spring', 'blue_ano', 'black_ano'):
            bright = bright * (1 + brush2[:, :, None] * .012 +
                               (micro[:, :, None] - .5) * .026)
        color = base_rgb * bright
        if amount > 0:
            blotch_mask = np.clip((blotch - (.93 - .10 * amount)) / .05, 0, 1)
            spot_mask = np.clip((fine - .93) / .05, 0, 1) * \
                np.clip((coarse - .45) / .45, 0, 1)
            edge_mask = np.clip((coarse - (.86 - .05 * amount)) / .08, 0, 1)
            cov = np.clip(edge_mask * amount * .50 +
                          blotch_mask * min(.85, .15 + amount * 10.0) +
                          spot_mask * min(.55, .10 + amount * 4.0), 0, .90)
            color = color * (1 - cov[:, :, None] * .90) + \
                rust_rgb[None, None, :] * cov[:, :, None] * .90
            rough = mat['roughness'] * (1 - cov) + .76 * cov + (fine - .5) * .035
            metal = mat['metallic'] * (1 - cov * .90)
            height = detail * .0022 + cov * (fine - .5) * .42
            if name in ('chrome', 'steel', 'bronze', 'silver_band'):
                height = height + brush * .06
        else:
            rough = np.full((tile, tile), mat['roughness'], np.float32) + \
                (fine - .5) * .035
            metal = np.full((tile, tile), mat['metallic'], np.float32)
            height = (micro - .5) * .45 + brush * .06
        if name == 'rust':
            color = rust_rgb[None, None, :] * (0.72 + .55 * (coarse[:, :, None] * .5 +
                                                                 fine[:, :, None] * .5))
            color = color * (1 - .25 * (micro[:, :, None] - .5)) + \
                dark_oxide[None, None, :] * (0.25 * (coarse[:, :, None] - .3))[:, :, :]
            rough = np.full((tile, tile), mat['roughness'], np.float32) + (fine - .5) * .05
            metal = np.zeros((tile, tile), np.float32)
            height = (fine - .5) * .55 + (micro - .5) * .25
        if name == 'rubber':
            color = base_rgb * (1 + (fine[:, :, None] - .5) * .05 +
                                (micro[:, :, None] - .5) * .035)
            height = (micro - .5) * .35
            metal = np.zeros((tile, tile), np.float32)
            rough = np.full((tile, tile), mat['roughness'], np.float32)
        if name == 'braid':
            color = base_rgb * (1 + (fine[:, :, None] - .5) * .03 +
                                (micro[:, :, None] - .5) * .02)
            height = braid_detail
            metal = np.full((tile, tile), mat['metallic'], np.float32)
            rough = np.full((tile, tile), mat['roughness'], np.float32) + \
                (fine - .5) * .03
        put(idx, color, rough, metal, height,
            clip_min=6 if name == 'rubber' else 22)
    # spare tiles: neutral mid-grey with subtle variation, never black
    for idx in (11, 12, 13, 14, 15):
        grey = 118 * (1 + (fine[:, :, None] - .5) * .06 +
                      (micro[:, :, None] - .5) * .035)
        put(idx, grey, .55, 0.0,
            (micro - .5) * .20 + (fine - .5) * .15)
    paths = {}
    for name, arr in (('base_color', base), ('metallic_roughness', mr),
                      ('normal', normal)):
        p = out / f'{name}.png'
        Image.fromarray(arr.astype(np.uint8), 'RGB').save(p)
        paths[name] = str(p)
    sheet = Image.new('RGB', (size, size), (245, 245, 243))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(LABEL_FONT, 22)
    for idx in range(16):
        col, row = idx % 4, idx // 4
        x, y = col * tile, row * tile
        sy = (3 - row) * tile
        sheet.paste(Image.fromarray(base[sy:sy + tile, x:x + tile]), (x, y))
        draw.rectangle((x, y, x + tile, y + tile), outline=(230, 230, 228), width=2)
        draw.text((x + 10, y + 8), f"{idx} {TILE_NAMES.get(idx, '备用')}",
                  fill=(20, 20, 20), font=font)
    sheet.save(out / '材质图集-布局.jpg', quality=93)
    paths.update(make_label_textures(out))
    return paths, {'rust_palette_rgb': [int(v) for v in rust_rgb],
                   'rust_brown_pixels': int(brown.sum()),
                   'rust_strong_pixels': int(strong.sum()),
                   'note': ('锈色取自已对齐做锈图中较饱和的棕色像素中位数（避免把灰色磨痕误作锈色）；'
                            '分布为程序化近似，不是逐像素投影。')}


# ---------------------------------------------------------------------------
# Source measurements and old-model audit
# ---------------------------------------------------------------------------
def source_measurements(original_path, rust_path, profile):
    im = Image.open(original_path).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('原图尺寸改变，测量坐标需要重新核对')
    a = np.asarray(im)
    mask = a[:, :, 3] >= 128
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    hsv = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2HSV)
    blue = np.logical_and.reduce([
        mask, np.greater_equal(hsv[:, :, 0], 95), np.less_equal(hsv[:, :, 0], 135),
        np.greater_equal(hsv[:, :, 1], 60), np.greater_equal(hsv[:, :, 2], 35)])
    n, _, stats, cent = cv2.connectedComponentsWithStats(blue.astype(np.uint8), 8)
    blue_components = [{'bbox': [int(s[0]), int(s[1]), int(s[2]), int(s[3])],
                        'area': int(s[4]),
                        'centre': [round(float(c[0]), 2), round(float(c[1]), 2)]}
                       for s, c in zip(stats[1:], cent[1:]) if s[4] >= 800]
    steel = np.logical_and.reduce([
        mask, np.less_equal(hsv[:, :, 1], 55), np.greater_equal(hsv[:, :, 2], 120)])
    n, _, stats, cent = cv2.connectedComponentsWithStats(steel.astype(np.uint8), 8)
    steel_components = [{'bbox': [int(s[0]), int(s[1]), int(s[2]), int(s[3])],
                         'area': int(s[4]),
                         'centre': [round(float(c[0]), 2), round(float(c[1]), 2)]}
                        for s, c in zip(stats[1:], cent[1:]) if s[4] >= 150]
    brown = np.logical_and.reduce([
        mask, np.greater_equal(hsv[:, :, 0], 5), np.less_equal(hsv[:, :, 0], 35),
        np.greater_equal(hsv[:, :, 1], 45), np.greater_equal(hsv[:, :, 2], 55)])
    n, _, stats, cent = cv2.connectedComponentsWithStats(brown.astype(np.uint8), 8)
    brown_components = [{'bbox': [int(s[0]), int(s[1]), int(s[2]), int(s[3])],
                         'area': int(s[4]),
                         'centre': [round(float(c[0]), 2), round(float(c[1]), 2)]}
                        for s, c in zip(stats[1:], cent[1:]) if s[4] >= 120]
    rust = np.asarray(Image.open(rust_path).convert('RGB')).astype(float)
    diff = np.linalg.norm(rust - a[:, :, :3].astype(float), axis=2)
    rr, gg, bb = rust[:, :, 0], rust[:, :, 1], rust[:, :, 2]
    rust_brown = np.logical_and.reduce([mask, np.greater_equal(rr, bb + 12),
                                        np.greater_equal(rr, gg - 8),
                                        np.greater_equal(rr, 70),
                                        np.less_equal(gg, 150)])
    d = profile['dimensions']
    top_top = d['top_clevis']['eye_z'] + d['top_clevis']['eye_outer_radius']
    bottom_bottom = d['bottom_clevis']['eye_z'] - d['bottom_clevis']['eye_outer_radius']
    overall = top_top - bottom_bottom
    r = d['reservoir']
    axis = np.asarray(r['axis'], float)
    axis /= np.linalg.norm(axis)
    far_y = r['origin'][1] + axis[1] * r['cap_t1'] + r['cap_radius']
    max_width = far_y + d['spring']['mean_radius'] + d['spring']['wire_radius']
    spring_od = 2 * (d['spring']['mean_radius'] + d['spring']['wire_radius'])
    return {
        'image': str(original_path), 'size': SOURCE_SIZE,
        'alpha_area_px': int(mask.sum()), 'bbox': bbox,
        'min_area_rect': {'centre': [round(float(cx), 2), round(float(cy), 2)],
                          'size': [round(float(w), 2), round(float(h), 2)],
                          'angle_deg': round(float(angle), 2),
                          'length_over_width': round(float(max(w, h) / min(w, h)), 4)},
        'blue_components': sorted(blue_components, key=lambda z: -z['area'])[:8],
        'steel_components': sorted(steel_components, key=lambda z: -z['area'])[:8],
        'brown_components': sorted(brown_components, key=lambda z: -z['area'])[:8],
        'rust_color_shift': {'mean_rgb_distance': round(float(diff[mask].mean()), 3),
                             'p50': round(float(np.percentile(diff[mask], 50)), 3),
                             'p90': round(float(np.percentile(diff[mask], 90)), 3),
                             'brown_fraction': round(float(rust_brown.sum() / mask.sum()), 4)},
        'configured': {'overall_length_m': round(overall, 4),
                       'max_width_m': round(float(max_width), 4),
                       'overall_length_over_width': round(overall / float(max_width), 4),
                       'spring_outer_diameter_m': round(spring_od, 4),
                       'spring_wire_diameter_m': round(2 * d['spring']['wire_radius'], 4),
                       'reservoir_diameter_m': round(2 * d['reservoir']['cap_radius'], 4),
                       'top_eye_bore_diameter_m': round(2 * d['top_clevis']['bore_radius'], 4),
                       'bottom_eye_bore_diameter_m': round(2 * d['bottom_clevis']['bore_radius'], 4)},
        'note': ('原照是 3/4 视角，minAreaRect 只描述照片中的可见轮廓；'
                 '工作尺度按合理比例设置，不是实测。蓝色/钢色/棕色组件用于确认弹簧漆面、'
                 '软管与接头的可见性，不证明真实尺寸或材料。'),
    }


def old_model_audit(row):
    mesh = trimesh.load(row['mesh'], force='scene')
    geoms = list(mesh.geometry.values())
    v = np.vstack([g.vertices for g in geoms])
    f = np.vstack([g.faces + sum(len(x.vertices) for x in geoms[:i])
                   for i, g in enumerate(geoms)])
    merged = trimesh.Trimesh(v, f, process=True)
    comps = sorted(merged.split(only_watertight=False), key=lambda x: len(x.vertices),
                   reverse=True)
    main = comps[0]
    tree = cKDTree(main.vertices)
    fragments = []
    for c in comps[1:]:
        dist, _ = tree.query(c.vertices)
        fragments.append({'vertices': len(c.vertices),
                          'bounds': np.round(c.bounds, 4).tolist(),
                          'min_gap_m': round(float(dist.min()), 4)})
    return {
        'mesh': row['mesh'], 'mesh_sha256': row['mesh_sha256'],
        'vertices': int(len(merged.vertices)), 'faces': int(len(merged.faces)),
        'bounds': np.round(merged.bounds, 4).tolist(),
        'components': len(comps),
        'main_vertices': len(main.vertices),
        'fragments': fragments,
        'old_bounds_ratio_width_over_height': round(
            float((main.bounds[1][1] - main.bounds[0][1]) /
                  (main.bounds[1][2] - main.bounds[0][2])), 4),
        'old_depth_m': round(float(main.bounds[1][0] - main.bounds[0][0]), 4),
        'note': ('旧网格是单张做锈图的 TripoSR 结果再经圆润/孔洞修复的单部件壳；'
                 '源图轮廓门槛虽通过，但六方向审计提示侧面弹簧与连接部的厚度和完整性仍需核对，'
                 '目视仍见弹簧圈与本体粘连、储油罐表面起伏、软管接头与滚花细节缺失。'
                 '因此本件不复用旧网格，改为按可见结构分部件重建。'),
    }


def audit_row(number):
    path = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
    for line in path.read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('number') == number:
                return row
    raise ValueError('audit row not found')


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def _font(size, bold=False):
    try:
        return ImageFont.truetype(LABEL_FONT_BOLD if bold else LABEL_FONT, size)
    except OSError:
        return ImageFont.load_default()


def _load_rgba(path):
    return Image.open(path).convert('RGBA')


def _on_white(im, size, background=(232, 232, 230)):
    im = im.convert('RGBA')
    canvas = Image.new('RGB', (size, size), background)
    im.thumbnail((size - 12, size - 12), Image.LANCZOS)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    return canvas


def _panel(im, size, title, subtitle=None):
    panel = Image.new('RGB', (size, size + 46), (250, 250, 248))
    panel.paste(_on_white(im, size), (0, 0))
    d = ImageDraw.Draw(panel)
    d.text((8, size + 5), title, fill=(24, 24, 24), font=_font(18, True))
    if subtitle:
        d.text((8, size + 26), subtitle, fill=(80, 80, 78), font=_font(13))
    return panel


def _wrap(draw, text, font, width):
    lines, current = [], ''
    for char in text:
        trial = current + char
        if draw.textlength(trial, font=font) > width and current:
            lines.append(current)
            current = char
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def make_board(out, items, columns, title, note, cell=300, path_name='board.jpg'):
    rows = math.ceil(len(items) / columns)
    header = 62
    width = columns * cell
    note_font = _font(13)
    probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    note_lines = _wrap(probe, note, note_font, width - 32)
    footer = 26 + 20 * len(note_lines)
    height = header + rows * (cell + 46) + footer
    board = Image.new('RGB', (width, height), (244, 244, 242))
    d = ImageDraw.Draw(board)
    d.text((16, 12), title, fill=(20, 20, 20), font=_font(24, True))
    for i, (label, sub, image) in enumerate(items):
        p = _panel(image, cell, label, sub)
        board.paste(p, ((i % columns) * cell, header + (i // columns) * (cell + 46)))
    for i, line in enumerate(note_lines):
        d.text((16, height - footer + 8 + i * 20), line, fill=(70, 70, 68),
               font=note_font)
    path = out / path_name
    board.save(path, quality=93)
    return path


def six_view_board(out):
    items = []
    for view in VIEWS:
        items.append((f'新版 · {view}', '分部件重建', _load_rgba(out / 'renders' / f'new-{view}.png')))
    for view in VIEWS:
        items.append((f'旧模型 · {view}', '原单图重建（对照）', _load_rgba(out / 'renders' / f'old-{view}.png')))
    return make_board(out, items, 6, '27 液压减震器 · 六方向新旧对照',
                      '同一中性光、正交相机；旧模型保留原样，仅作对照。新版为分部件重建，不用平滑掩盖缺失结构。',
                      path_name='六视图-新旧对照.jpg')


def detail_board(out):
    specs = [
        ('top-eye', '上耳环 / 贯穿孔 / 衬套'),
        ('top-bolt', '颈部内六角锁紧螺钉'),
        ('label', 'YU-AN 标签（近似字形）'),
        ('spring-upper', '上弹簧座 / 弹簧圈'),
        ('spring-lower', '下座 / 镀铬杆 / 导向环'),
        ('bottom-clevis', '下 U 形夹 / 贯穿孔 / 螺栓'),
        ('hose-top', '软管上端 90 度接头'),
        ('hose-run', '编织软管走向'),
        ('reservoir-banjo', '储油罐近端接头'),
        ('reservoir-cap', '滚花端盖 / 内六角'),
        ('overall-3q', '整体 3/4 视角'),
    ]
    items = [(f'局部 · {label}', '同光照近景', _load_rgba(out / 'renders' / f'detail-{name}.png'))
             for name, label in specs]
    return make_board(out, items, 3, '27 液压减震器 · 关键局部检查',
                      '局部近景用于检查耳环孔、衬套、弹簧圈、镀铬杆、软管接头、滚花端盖和标签；'
                      '不能据此证明不可见背面、内部油路或真实尺寸。',
                      cell=340, path_name='局部检查.jpg')


def parts_board(out):
    items = [
        ('上耳环组件', '两片夹板/贯穿孔/钢衬套/锁紧螺钉',
         _load_rgba(out / 'renders' / 'top-eye-only.png')),
        ('下 U 形夹组件', '两片夹板/贯穿孔/钢衬套/两侧螺栓头',
         _load_rgba(out / 'renders' / 'bottom-clevis-only.png')),
        ('弹簧与弹簧座', '螺旋弹簧/上下座/端部',
         _load_rgba(out / 'renders' / 'spring-only.png')),
        ('储油罐组件', '蓝色筒/银色带/滚花盖/近端接头',
         _load_rgba(out / 'renders' / 'reservoir-only.png')),
        ('软管组件', '编织软管/两端接头/卡套',
         _load_rgba(out / 'renders' / 'hose-only.png')),
    ]
    return make_board(out, items, 2, '27 液压减震器 · 分部件隔离检查',
                      '隔离渲染用于确认零件独立存在与相对位置；只覆盖当前视角，装配关系仍待审。',
                      path_name='分部件隔离.jpg')


def material_board(out):
    items = [
        ('新版材质主视', 'PBR 图集 / 中性光', _load_rgba(out / 'renders' / 'new-material-front.png')),
        ('新版灰度主视', '同光照灰模', _load_rgba(out / 'renders' / 'new-clay-front.png')),
        ('旧模型主视', '旧顶点色网格', _load_rgba(out / 'renders' / 'old-front.png')),
    ]
    return make_board(out, items, 3, '27 液压减震器 · 同光照材质/灰模对照',
                      '蓝色弹簧漆面、标签和黑色橡胶为非金属，金属反射检查不适用于它们；'
                      '黑色阳极氧化件、蓝色储油罐、镀铬杆、钢接头和软管为金属，保留随光位变化的反射。',
                      path_name='材质-同光照对照.jpg')


def metal_board(out, metrics):
    items = [
        ('主光 A', '金属件隔离（杆/储油罐/软管/接头）',
         _load_rgba(out / 'renders' / 'metal-only-light-a.png')),
        ('主光 B', '同一相机，仅移动主光',
         _load_rgba(out / 'renders' / 'metal-only-light-b.png')),
    ]
    note = (f"非背景像素差分 {metrics['mean_abs_difference']:.2f}/255；"
            f"最亮 1% 高光质心位移 {metrics['highlight_shift_px']:.1f} px；"
            f"高阈值 {metrics.get('high_threshold_luma', 0):.0f} 位移 "
            f"{metrics.get('high_threshold_shift_px', 0):.1f} px；"
            f"金属像素 {metrics['metal_pixels']}。仅证明金属高光随光位移动，不证明材质测量正确。")
    return make_board(out, items, 2, '27 液压减震器 · 金属反射变光检查', note,
                      cell=360, path_name='金属反射-变光对照.jpg')


def opening_board(out, counts):
    items = [
        ('上耳环', f"贯穿孔 {'通过' if counts['checks']['top_eye_through'] else '未通过'}",
         _load_rgba(out / 'renders' / 'top-eye-only.png')),
        ('下 U 形夹', f"贯穿孔 {'通过' if counts['checks']['bottom_eye_through'] else '未通过'}",
         _load_rgba(out / 'renders' / 'bottom-clevis-only.png')),
        ('螺旋弹簧', f"非实心/开放螺旋 {'通过' if counts['checks']['spring_open_helix'] else '未通过'}",
         _load_rgba(out / 'renders' / 'spring-only.png')),
        ('储油罐端面', f"内六角 {'通过' if counts['checks']['reservoir_socket'] else '未通过'} · "
                       f"滚花 {counts.get('knurl_count')}/12",
         _load_rgba(out / 'renders' / 'reservoir-only.png')),
    ]
    note = (f"射线检查：上耳环贯穿孔 {counts.get('top_eye_through')}/1；"
            f"下 U 形夹贯穿孔 {counts.get('bottom_eye_through')}/1；"
            f"钢衬套 {counts.get('bushings_present')}/2；"
            f"下侧螺栓头 {counts.get('bottom_bolts')}/2；"
            f"弹簧径向射线命中 {counts.get('spring_hits')}/{counts.get('spring_rays')}（开放螺旋）；"
            f"储油罐端面内六角 {'通过' if counts['checks']['reservoir_socket'] else '未通过'}。"
            '只证明这些方向/位置的几何开孔与装配间隙，不代表内部油路或隐藏结构。')
    return make_board(out, items, 2, '27 液压减震器 · 开孔与隔离检查', note,
                      cell=340, path_name='开孔-隔离检查.jpg')


def source_board(out, metrics):
    items = [
        ('旧库源视图（诊断）', '输入源视图 / 非本件原照',
         Image.open(metrics['source_view']).convert('RGBA')),
        ('新版严格原相机', f"IoU {metrics['strict']['iou']:.4f}（尺度不可比，仅方位诊断）",
         _load_rgba(out / 'renders' / 'new-source.png')),
        (f"包围盒对齐最优 {metrics['best_view']}°", f"IoU {metrics['best']['iou']:.4f}（仅诊断）",
         _load_rgba(out / 'renders' / f"turn-{int(metrics['best_view']) // 15:02d}.png")),
    ]
    note = ('旧模型库相机源自单图 TripoSR 的任意朝向，旧库网格归一化到约 1 m；'
            '本件按库约定重建为主轴沿 Z、正面相机 +X，工作尺度约 0.31 m，'
            '所以严格相机 IoU 的尺度不可比。包围盒对齐诊断也不是标定，不构成验收通过。')
    return make_board(out, items, 3, '27 液压减震器 · 原视角轮廓诊断', note,
                      path_name='原视角诊断.jpg')


def original_board(out):
    orig = Image.open(ROOT / 'outputs/物品高清输入-39件/原貌HD/27-液压减震器-校正版.png').convert('RGBA')
    new = _load_rgba(out / 'renders' / 'new-material-front.png')
    items = [
        ('原貌照片（校正版）', '3/4 视角，带局部锈迹', orig),
        ('新版主视渲染', 'PBR 材质 + 中性光', new),
        ('六视图参考', '未标定的黑蓝后减震器外观匹配', Image.open(
            ROOT / 'references/parallel-six-view' / REFERENCE_NAME).convert('RGBA')),
    ]
    return make_board(out, items, 3, '27 液压减震器 · 原照与新候选（附六视图参考）',
                      '参考图只用于外观和背侧拓扑；原照的锈迹、标签、软管方位和 3/4 视角以照片为准，不把参考当实测。',
                      path_name='原照-新候选对照.jpg', cell=360)


def turntable_board(out):
    items = []
    for i in range(24):
        items.append((f'{i * 15:03d}°', '新版转台', _load_rgba(out / 'renders' / f'turn-{i:02d}.png')))
    return make_board(out, items, 6, '27 液压减震器 · 24 方向转台',
                      '转台只用于人工旋转检查；六视图参考未标定，不能把轮廓相似称为实测精度。',
                      cell=150, path_name='转台-24方向.jpg')


def atlas_board(out):
    items = [
        ('底色图集', 'sRGB / 2048 px', Image.open(out / 'base_color.png').convert('RGBA')),
        ('金属-粗糙度图集', 'G=粗糙度 / B=金属度', Image.open(out / 'metallic_roughness.png').convert('RGBA')),
        ('法线图集', '非颜色 / 低强度', Image.open(out / 'normal.png').convert('RGBA')),
        ('标签贴图', '顶盖 YU-AN', Image.open(out / 'label_cap.png').convert('RGBA')),
        ('银色带刻字', '近似绘制', Image.open(out / 'label_band.png').convert('RGBA')),
    ]
    return make_board(out, items, 3, '27 液压减震器 · PBR 贴图集',
                      '金属件金属度按材质设置；锈斑处金属度下降（锈为电介质）。蓝色弹簧漆、橡胶与备用格金属度为 0；标签为电介质。',
                      cell=420, path_name='材质图集-对照.jpg')


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def texture_checks(out, profile):
    base = np.asarray(Image.open(out / 'base_color.png').convert('RGB'))
    mr = np.asarray(Image.open(out / 'metallic_roughness.png').convert('RGB'))
    normal = np.asarray(Image.open(out / 'normal.png').convert('RGB'))
    size = profile['texture']['size']
    tile = profile['texture']['tile']
    tiles = profile['texture']['tiles']
    w = np.array([.2126, .7152, .0722])
    result = {'tiles': {}, 'labels': {}, 'notes': []}
    for name, idx in tiles.items():
        x0, y0, x1, y1 = tile_rect(idx, size, tile)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        b = base[cy - 24:cy + 24, cx - 24:cx + 24].reshape(-1, 3).astype(float)
        m = mr[cy - 24:cy + 24, cx - 24:cx + 24].reshape(-1, 3).astype(float)
        n = normal[cy - 24:cy + 24, cx - 24:cx + 24].reshape(-1, 3).astype(float)
        luma = float(np.median(b @ w))
        region = base[y0 + int(.15 * tile):y1 - int(.15 * tile),
                      x0 + int(.15 * tile):x1 - int(.15 * tile)].astype(float)
        lum = region @ w
        hh, ww = lum.shape
        lr = abs(float(lum[:, :ww // 3].mean() - lum[:, -ww // 3:].mean()))
        tb = abs(float(lum[:hh // 3, :].mean() - lum[-hh // 3:, :].mean()))
        r, g, bl = region[:, :, 0], region[:, :, 1], region[:, :, 2]
        brown = np.logical_and.reduce([r >= bl + 15, r >= g - 5, r >= 60, g <= 165])
        full = base[y0:y1, x0:x1].astype(float)
        fr, fg, fb = full[:, :, 0], full[:, :, 1], full[:, :, 2]
        brown_full = np.logical_and.reduce([fr >= fb + 15, fr >= fg - 5,
                                            fr >= 60, fg <= 165])
        result['tiles'][name] = {
            'index': idx,
            'median_luma': round(luma, 2),
            'base_std': round(float(b.std()), 2),
            'tile_std': round(float(region.std()), 2),
            'roughness': round(float(np.median(m[:, 1]) / 255), 3),
            'metallic_median': round(float(np.median(m[:, 2]) / 255), 3),
            'metallic_p90': round(float(np.percentile(m[:, 2], 90) / 255), 3),
            'metallic_p95': round(float(np.percentile(m[:, 2], 95) / 255), 3),
            'normal_std': round(float(n.std()), 2),
            'luma_gradient_lr': round(lr, 2),
            'luma_gradient_tb': round(tb, 2),
            'rust_pixel_fraction': round(float(brown_full.sum() / brown_full.size), 4),
            'mean_rgb': [int(v) for v in np.median(region.reshape(-1, 3), axis=0)],
        }
    for name in ('label_cap', 'label_band'):
        arr = np.asarray(Image.open(out / f'{name}.png').convert('RGB')).astype(float)
        lum = arr @ w
        result['labels'][name] = {
            'size': [int(arr.shape[1]), int(arr.shape[0])],
            'median_luma': round(float(np.median(lum)), 2),
            'std': round(float(lum.std()), 2),
            'p05': round(float(np.percentile(lum, 5)), 2),
            'p95': round(float(np.percentile(lum, 95)), 2),
        }
    T = result['tiles']
    checks = {}
    metal_thresholds = {'black_ano': .55, 'blue_ano': .70, 'chrome': .95,
                        'steel': .90, 'dark_steel': .80, 'bronze': .90,
                        'silver_band': .90, 'braid': .90}
    for name, thr in metal_thresholds.items():
        checks[f'{name}_metal'] = T[name]['metallic_p90'] >= thr
    for name in ('blue_spring', 'rubber', 'rust'):
        checks[f'{name}_dielectric'] = T[name]['metallic_p95'] <= .05
    rough_limits = {
        'black_ano': (.28, .48), 'blue_spring': (.15, .40), 'blue_ano': (.18, .38),
        'chrome': (.04, .18), 'steel': (.16, .38), 'dark_steel': (.28, .48),
        'rubber': (.65, .92), 'rust': (.68, .95), 'bronze': (.18, .40),
        'silver_band': (.12, .32), 'braid': (.16, .40), 'spare11': (.40, .70),
        'spare12': (.40, .70), 'spare13': (.40, .70), 'spare14': (.40, .70),
        'spare15': (.40, .70),
    }
    for name, (lo, hi) in rough_limits.items():
        checks[f'{name}_roughness'] = lo <= T[name]['roughness'] <= hi
    checks['bright_not_black'] = all(
        T[n]['median_luma'] >= (80 if n in ('chrome', 'steel', 'silver_band',
                                            'bronze', 'braid')
                                else (10 if n in ('black_ano', 'rubber') else 20))
        for n in ('black_ano', 'blue_spring', 'blue_ano', 'chrome', 'steel',
                  'dark_steel', 'bronze', 'silver_band', 'braid', 'rust'))
    checks['all_tiles_have_variation'] = all(T[n]['tile_std'] >= .6 for n in T)
    checks['normal_not_flat'] = all(T[n]['normal_std'] >= .6 for n in T)
    checks['no_baked_gradient'] = all(
        max(T[n]['luma_gradient_lr'], T[n]['luma_gradient_tb']) <= 8.0 for n in T)
    checks['rust_on_paint_or_metal'] = any(
        T[n]['rust_pixel_fraction'] >= .0005
        for n in ('blue_spring', 'black_ano', 'steel', 'braid'))
    checks['blue_spring_hue'] = bool(
        T['blue_spring']['mean_rgb'][2] >= T['blue_spring']['mean_rgb'][0] + 40
        and T['blue_spring']['mean_rgb'][2] >= T['blue_spring']['mean_rgb'][1] + 25)
    checks['label_cap_contrast'] = (result['labels']['label_cap']['std'] >= 20
                                    and result['labels']['label_cap']['median_luma'] >= 8)
    checks['label_band_contrast'] = result['labels']['label_band']['std'] >= 5
    result['checks'] = checks
    result['passed'] = all(checks.values())
    result['notes'].append('图集统计只证明贴图数值与变化量，不证明纹理在实物上的真实分布。')
    result['notes'].append('锈斑、漆面磨损、标签和编织软管纹理为基于做锈图的有限采样与程序化近似，不是逐像素投影。')
    return result


def source_diagnostics(out, profile, alpha_path, source_view_path):
    source = np.asarray(Image.open(alpha_path).convert('L'))
    rendered = np.asarray(_load_rgba(out / 'renders' / 'new-source.png').getchannel('A'))
    strict = source_fidelity(source, rendered)
    strict['acceptance_gate'] = False
    strict['not_comparable_scale'] = True
    strict['camera'] = '旧模型库标定相机；旧库网格被归一化到约 1 m，本候选为约 0.31 m 工作尺度，严格相机下尺度不可比'
    strict['scale_note'] = ('严格 IoU 只用于确认同一相机下的大致方位，不能与旧模型数值直接比较；'
                            '更有意义的是按包围盒对齐的正视/转台诊断。')
    sf = mask_fit(source >= 128)
    nf = mask_fit(rendered >= 128)
    front_fit = source_fidelity(sf * 255, nf * 255) if sf is not None and nf is not None else None
    if front_fit:
        front_fit['alignment'] = '仅按包围盒缩放/平移，不是标定'
        front_fit['acceptance_gate'] = False
    best = None
    for i in range(24):
        turn_path = out / 'renders' / f'turn-{i:02d}.png'
        if not turn_path.is_file():
            continue
        a = np.asarray(_load_rgba(turn_path).getchannel('A'))
        fit = mask_fit(a >= 128)
        if sf is None or fit is None:
            continue
        score = source_fidelity(sf * 255, fit * 255)
        if best is None or score['iou'] > best[1]['iou']:
            best = (i * 15, score)
    best_metrics = best[1] if best else None
    if best_metrics:
        best_metrics['view'] = f'{best[0]}°'
        best_metrics['alignment'] = '仅按包围盒缩放/平移，不是标定'
        best_metrics['acceptance_gate'] = False
    return {'source_alpha': str(alpha_path), 'source_view': str(source_view_path),
            'strict': strict, 'front_box_fit': front_fit,
            'best_view': best[0] if best else 0, 'best': best_metrics or {}}


def opening_checks(out, worker):
    r = worker.get('ray_checks', {})
    eyes = r.get('eyes', {})
    bushings = r.get('bushings', {})
    bolts = r.get('bolts', {})
    spring = r.get('spring', {})
    reservoir = r.get('reservoir', {})
    objects = r.get('objects', {})
    top_through = bool(eyes.get('top', {}).get('through'))
    bottom_through = bool(eyes.get('bottom', {}).get('through'))
    bushing_count = sum(1 for v in bushings.values() if v.get('present'))
    return {
        'top_eye_through': int(top_through),
        'bottom_eye_through': int(bottom_through),
        'bushings_present': bushing_count,
        'bottom_bolts': bolts.get('bottom_head_count'),
        'top_clamp_bolt': bool(bolts.get('top_clamp')),
        'spring_hits': spring.get('radial_rays_hit'),
        'spring_rays': spring.get('radial_rays'),
        'spring_open': bool(spring.get('open_helix')),
        'knurl_count': bolts.get('knurl_count'),
        'reservoir': reservoir,
        'objects': objects,
        'checks': {
            'top_eye_through': top_through,
            'bottom_eye_through': bottom_through,
            'two_bushings_present': bushing_count == 2,
            'two_bottom_bolts': bolts.get('bottom_head_count') == 2,
            'top_clamp_bolt': bool(bolts.get('top_clamp')),
            'spring_open_helix': bool(spring.get('open_helix')),
            'reservoir_socket': bool(reservoir.get('end_socket_hit')),
            'twelve_knurls': bolts.get('knurl_count') == 12,
        },
        'note': ('射线检查直接打在上/下夹板、衬套和弹簧网格上：两个安装孔中心为真实贯穿孔，'
                 '钢衬套在孔内可命中，弹簧从轴线向外的射线只有部分命中（开放螺旋而非实心圆柱），'
                 '储油罐端面内六角可命中。只证明这些几何特征，不证明内部油路、密封或装配关系。'),
    }


def metal_reflection_check(out):
    a = np.asarray(_load_rgba(out / 'renders' / 'metal-only-light-a.png'))
    b = np.asarray(_load_rgba(out / 'renders' / 'metal-only-light-b.png'))
    mask = a[:, :, 3] >= 128
    if mask.sum() < 1000:
        return {'metal_pixels': int(mask.sum()), 'passed': False,
                'reason': '金属隔离渲染非背景像素不足'}
    la = a[:, :, :3].astype(float) @ np.array([.2126, .7152, .0722])
    lb = b[:, :, :3].astype(float) @ np.array([.2126, .7152, .0722])
    diff = np.abs(la - lb)[mask]
    qa, qb = np.quantile(la[mask], .99), np.quantile(lb[mask], .99)
    yqa, xqa = np.nonzero(mask & (la >= qa))
    yqb, xqb = np.nonzero(mask & (lb >= qb))
    ca = np.array([xqa.mean(), yqa.mean()]) if len(xqa) else np.array([0, 0])
    cb = np.array([xqb.mean(), yqb.mean()]) if len(xqb) else np.array([0, 0])
    top1_shift = float(np.linalg.norm(ca - cb))
    max_luma = max(float(la[mask].max()), float(lb[mask].max()))
    threshold = max(180.0, .75 * max_luma)
    ya, xa = np.nonzero(mask & (la >= threshold))
    yb, xb = np.nonzero(mask & (lb >= threshold))
    high_shift = (float(np.linalg.norm(np.array([xa.mean(), ya.mean()]) -
                                       np.array([xb.mean(), yb.mean()])))
                  if len(xa) and len(xb) else 0.0)
    result = {
        'metal_pixels': int(mask.sum()),
        'mean_abs_difference': round(float(diff.mean()), 3),
        'highlight_shift_px': round(top1_shift, 2),
        'highlight_method': '最亮 1% 非背景金属像素质心位移',
        'highlight_pixels_a': int(len(xqa)),
        'highlight_pixels_b': int(len(xqb)),
        'high_threshold_luma': round(threshold, 2),
        'high_threshold_shift_px': round(high_shift, 2),
        'high_threshold_pixels_a': int(len(xa)),
        'high_threshold_pixels_b': int(len(xb)),
        'light_a_highlight_centre': [round(float(ca[0]), 1), round(float(ca[1]), 1)],
        'light_b_highlight_centre': [round(float(cb[0]), 1), round(float(cb[1]), 1)],
    }
    result['passed'] = (result['mean_abs_difference'] >= 1.0 and top1_shift >= 2.0
                        and len(xqa) >= 50 and len(xqb) >= 50)
    result['note'] = ('只说明金属最亮镜面高光随光位移动；不证明反射率、材质测量或环境光匹配。'
                      '蓝色弹簧漆、标签、橡胶与锈为电介质，本检查不适用于它们。')
    return result


# ---------------------------------------------------------------------------
# Report and hashes
# ---------------------------------------------------------------------------
def write_report(out, result):
    variant = out.name.rsplit('-', 1)[-1]
    lines = [
        f'# 27 液压减震器 · 六视图重建 {variant}（manual_only，待审）',
        '',
        '## 结论',
        '',
        '本候选把 27 号黑蓝摩托车后减震器按可见结构分成独立部件重建：上耳环夹板与钢衬套、'
        '黑色阳极氧化顶盖与 YU-AN 标签、上/下弹簧座、蓝色螺旋弹簧、黑色阻尼筒、镀铬活塞杆、'
        '青铜导向环、下筒体、下 U 形夹与贯穿螺栓、编织软管，以及带滚花端盖和银色中段带的副储油罐。'
        '它不替换原模型，不进入自动放置或正式合成；所有诊断均不是验收通过。',
        '',
        '## 为什么重建而不是保守修复',
        '',
        f"- 旧网格有 {result['old_audit']['vertices']} 个顶点、{result['old_audit']['faces']} 个面，"
        '但仍是单部件壳；目视弹簧圈与本体粘连、储油罐表面起伏、软管接头和滚花细节缺失。',
        f"- 原视角审计记录：IoU {result['audit']['source_fidelity']['iou']}，"
        f"被填孔 {result['audit']['source_fidelity']['filled_holes']}，"
        f"通过={result['audit']['source_fidelity']['passed']}；"
        '轮廓数值虽然通过旧门槛，但六方向审计仍提示弹簧与连接部厚度需核对，'
        '所以本件改用保留真实开孔和分离弹簧的分部件重建。',
        '',
        '## 输入与参考',
        '',
        '- 原图/做锈图/旧网格/源视图透明度哈希见 `hashes.json`。',
        f"- 六视图：`{REFERENCE_NAME}`，角色 **visual_match_unverified**（未标定）；"
        '各面板对上耳环朝向和软管方位不一致，候选以原照可见结构和物理合理性为准，'
        '参考只用于外观和背侧拓扑，不当作测量或同一实物。',
        '',
        '## 结构',
        '',
        f"- 工作尺度：总高约 {result['measurements']['configured']['overall_length_m']*1000:.0f} mm、"
        f"最大宽约 {result['measurements']['configured']['max_width_m']*1000:.0f} mm、"
        f"弹簧外径 {result['measurements']['configured']['spring_outer_diameter_m']*1000:.1f} mm、"
        f"钢丝直径 {result['measurements']['configured']['spring_wire_diameter_m']*1000:.1f} mm。"
        '这些是按合理比例设置的工作尺度，不是实测。',
        '- 上耳环为两片平行夹板，中心为沿 Y 的真实贯穿孔和钢衬套；颈部正面有内六角锁紧螺钉。',
        '- 下 U 形夹为两片平行夹板，中心为沿 X 的真实贯穿孔和钢衬套，两侧钢色六角螺栓头。',
        '- 蓝色弹簧为真正的螺旋管，不与阻尼筒合并；弹簧内部保留上下座、镀铬杆和青铜导向环。',
        '- 副储油罐保留蓝色筒身、银色中段带、12 道滚花、端面内六角、近端 90 度接头和六角螺母；'
        '不锈钢编织软管按原照走向连接顶盖右侧与储油罐近端。',
        f"- 网格：{result['mesh']['parts']} 个部件，{result['mesh']['polygons']} 面；"
        f"闭合部件 {result['mesh']['closed_after_welding_seams']}/{result['mesh']['parts']}；"
        f"同部件自交 {len(result['worker']['self_intersecting_components'])} 个。",
        '',
        '## 材质',
        '',
        '- 黑色阳极氧化件、蓝色阳极氧化储油罐、镀铬杆、钢接头/软管和青铜导向环为金属，'
        '金属度按材质设置，保留随光位变化的反射。',
        '- 蓝色弹簧漆、YU-AN 标签、黑色橡胶和锈为电介质，金属反射检查不适用于它们；'
        '这些表面仍保留非金属高光、粗糙度和局部磨损。',
        '- 锈色取自已对齐做锈图的棕色像素中位数；分布为程序化近似，不是逐像素投影。'
        '避免把固定灯光/阴影烘进底色；图集有梯度检查。',
        f"- 图集数值检查：{'通过' if result['textures']['passed'] else '未通过'}；"
        '详见 `result.json` 的 tile/label 数值。',
        '',
        '## 检查结果（实际执行）',
        '',
        f"- {result['versions']['blender']}，Cycles CPU，六视图 {result['render']['six_view_samples']} 采样；"
        'GLB/Blend/贴图/对照图均由本工作树内 Blender/Python 生成。',
        f"- 严格原相机轮廓：IoU {result['diagnostics']['strict']['iou']}，"
        f"填孔 {result['diagnostics']['strict']['filled_holes']}，细结构召回 "
        f"{result['diagnostics']['strict']['narrow_recall']}。"
        '旧库网格归一化到约 1 m，本候选约 0.31 m，严格相机下尺度不可比，此项只作方位诊断；'
        '按包围盒对齐的数值更有参考意义，但仍不是验收。',
        f"- 包围盒对齐最优转台 {result['diagnostics']['best'].get('view', 'n/a')}："
        f"IoU {result['diagnostics']['best'].get('iou', 'n/a')}（未标定，仅诊断）。",
        f"- 开孔：上耳环贯穿孔 {result['openings'].get('top_eye_through')}/1；"
        f"下 U 形夹贯穿孔 {result['openings'].get('bottom_eye_through')}/1；"
        f"钢衬套 {result['openings'].get('bushings_present')}/2；"
        f"下侧螺栓头 {result['openings'].get('bottom_bolts')}/2；"
        f"弹簧径向射线命中 {result['openings'].get('spring_hits')}/"
        f"{result['openings'].get('spring_rays')}（开放螺旋）；"
        f"储油罐端面内六角 {'通过' if result['openings']['checks']['reservoir_socket'] else '未通过'}；"
        f"滚花 {result['openings'].get('knurl_count')}/12。",
        (f"- 金属变光：差分 {result['metal'].get('mean_abs_difference')}/255，"
         f"最亮 1% 高光质心位移 {result['metal'].get('highlight_shift_px')} px"
         f"（高阈值位移 {result['metal'].get('high_threshold_shift_px')} px）。"
         '仅说明高光随光位移动。'
         if result['metal'] else '- 金属变光：草稿模式未执行。'),
        '- 浏览器交互：未执行（本工作树缺少本地 three.js/node_modules；未安装依赖，也未声称通过）。',
        '',
        '## 待人工复核',
        '',
        '- 六视图参考未标定且各面板对夹板朝向/软管方位不一致；工作尺度、弹簧线径、耳环孔径、'
        '螺纹规格、孔深和装配关系未实测。',
        '- 原照是 3/4 视角；严格原相机与包围盒对齐 IoU 只作诊断。',
        '- 背面、阻尼筒内部、活塞/阀片/油路、储油罐内部、软管内部通道、衬套配合和壁厚为推断。',
        '- 上耳环夹板朝向、下 U 形夹开口方向和软管路径中段为解释性重建。',
        '- YU-AN 标签文字按可见内容复制，字形为近似绘制；储油罐银色带刻字未准确复现。',
        '- 锈迹、漆面磨损和编织软管纹理为近似；不能把参考当实测。',
        '- 副储油罐仅由软管悬吊；落稳、接触、现场尺度、遮挡顺序与环境光融合未验证。',
        '- 浏览器交互检查未执行；需在串行整合时接入共享查看器并由人工复核。',
        '- 本候选不替换原模型，不进入自动放置或正式合成。',
        '',
        '## 文件',
        '',
        '- `mesh.glb`：Z 向上、库坐标约定，纹理内嵌。',
        '- `model.blend`：可编辑几何、PBR 节点、图集与标签已打包。',
        '- `base_color.png` / `metallic_roughness.png` / `normal.png`：2048 PBR 图集。',
        '- `label_cap.png` / `label_band.png`：YU-AN 标签与银色带贴图。',
        '- `六视图-新旧对照.jpg`、`局部检查.jpg`、`分部件隔离.jpg`、`材质-同光照对照.jpg`、'
        '`金属反射-变光对照.jpg`、`开孔-隔离检查.jpg`、`原视角诊断.jpg`、'
        '`原照-新候选对照.jpg`、`转台-24方向.jpg`、`材质图集-对照.jpg`。',
        '- `result.json`、`hashes.json`、`source-measurements.json`、`old-model-audit.json`、`worker-result.json`。',
        '',
        '## 复现',
        '',
        '```powershell',
        '& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/mesh_rebuild_item_27.py '
        f'--out "outputs/{out.name}-repro"',
        '& work/triposr-env/Scripts/python.exe -X utf8 -B tools/ps-model-controller/checks_rebuild_27.py '
        f'--out "{out}"',
        '```',
        '',
        '输出目录必须为新目录；相同签名的完整结果可复用。',
    ]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


def write_review_page(out, result):
    """Standalone offline review page using the rendered evidence only."""
    rel = out.name
    base = f'../{rel}'
    d = result['diagnostics']
    tiles = [
        ('六方向新旧对照', f'{base}/六视图-新旧对照.jpg'),
        ('关键局部', f'{base}/局部检查.jpg'),
        ('分部件隔离', f'{base}/分部件隔离.jpg'),
        ('同光照材质/灰模', f'{base}/材质-同光照对照.jpg'),
        ('金属反射变光', f'{base}/金属反射-变光对照.jpg'),
        ('开孔/衬套/弹簧隔离', f'{base}/开孔-隔离检查.jpg'),
        ('原视角轮廓诊断', f'{base}/原视角诊断.jpg'),
        ('原照/新候选/六视图参考', f'{base}/原照-新候选对照.jpg'),
        ('24 方向转台', f'{base}/转台-24方向.jpg'),
        ('PBR 图集', f'{base}/材质图集-对照.jpg'),
    ]
    pending = ''.join(f'<li>{p}</li>' for p in result['pending_review'])
    gallery = ''.join(
        f'<a class="card" href="{src}" target="_blank"><img src="{src}" alt="{title}">'
        f'<span>{title}</span></a>' for title, src in tiles)
    metrics = [
        ('严格原相机轮廓 IoU（尺度不可比，仅方位诊断）', d['strict']['iou']),
        ('正视图包围盒对齐 IoU（仅诊断）',
         (d.get('front_box_fit') or {}).get('iou', '未执行')),
        ('24 方向包围盒对齐最优 IoU（仅诊断）',
         (d.get('best') or {}).get('iou', '未执行')),
        ('上耳环贯穿孔', f"{result['openings'].get('top_eye_through')}/1"),
        ('下 U 形夹贯穿孔', f"{result['openings'].get('bottom_eye_through')}/1"),
        ('孔内钢衬套', f"{result['openings'].get('bushings_present')}/2"),
        ('下侧螺栓头', f"{result['openings'].get('bottom_bolts')}/2"),
        ('弹簧开放螺旋',
         f"{result['openings'].get('spring_hits')}/{result['openings'].get('spring_rays')} 射线命中"),
        ('储油罐端面内六角',
         '通过' if result['openings']['checks']['reservoir_socket'] else '未通过'),
        ('储油罐滚花', f"{result['openings'].get('knurl_count')}/12"),
        ('网格部件/闭合',
         f"{result['mesh']['parts']} / {result['mesh']['closed_after_welding_seams']}"),
        ('同部件自交', len(result['worker']['self_intersecting_components'])),
        ('贴图数值检查', result['textures'].get('passed')),
        ('金属高光随光位移动', result['metal'].get('passed')),
        ('金属高光质心位移 (px)', result['metal'].get('highlight_shift_px')),
        ('浏览器交互检查', '未执行'),
        ('原模型源图轮廓审计（旧）',
         f"IoU {result['audit']['source_fidelity']['iou']}，填孔 "
         f"{result['audit']['source_fidelity']['filled_holes']}，通过 "
         f"{result['audit']['source_fidelity']['passed']}"),
    ]
    metric_rows = ''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in metrics)
    turn_frames = ',\n'.join(f"'{base}/renders/turn-{i:02d}.png'" for i in range(24))
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>27 液压减震器 · 三维模型检查（待审）</title>
<style>
:root{{--ink:#1d2320;--muted:#5f6a64;--line:#d5d9d5;--panel:#fbfcfa;--accent:#2f5f8a}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.65 "Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:#eef1ee}}
header{{background:#1f2c3a;color:#f4f7f2;padding:26px 30px}}header h1{{margin:0 0 6px;font-size:25px}}
header p{{margin:0;color:#c9d6ca}}main{{max-width:1180px;margin:24px auto 60px;padding:0 18px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}}
h2{{margin:0 0 12px;font-size:19px;color:var(--accent)}}h3{{margin:16px 0 8px;font-size:16px}}
.notice{{border-left:5px solid #b5762a;background:#fff7e8;padding:12px 16px;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}}
.card{{display:block;background:#fff;border:1px solid var(--line);border-radius:9px;overflow:hidden;color:inherit;text-decoration:none}}
.card img{{width:100%;height:210px;object-fit:contain;background:#e7eae6;display:block}}
.card span{{display:block;padding:8px 10px;font-size:13px}}
.viewer{{display:grid;grid-template-columns:1fr 260px;gap:18px}}@media(max-width:820px){{.viewer{{grid-template-columns:1fr}}}}
.turnbox{{background:#e3e7e2;border-radius:10px;min-height:380px;display:flex;align-items:center;justify-content:center;user-select:none;touch-action:none;cursor:grab}}
.turnbox img{{max-height:420px;max-width:100%}}
aside{{font-size:13px;color:var(--muted)}}button{{font:inherit;padding:7px 13px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer}}
button:hover{{border-color:var(--accent)}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}}
th{{width:460px;color:var(--muted);font-weight:500}}code{{background:#eef1ee;padding:1px 5px;border-radius:4px}}
ul{{margin:8px 0 0 20px;padding:0}}a{{color:var(--accent)}}.tag{{display:inline-block;background:#f0e5cf;color:#77531a;border-radius:20px;padding:2px 10px;font-size:12px}}
</style></head><body>
<header><h1>27 液压减震器 · 六视图重建（{VERSION}） <span class="tag">manual_only · 待审</span></h1>
<p>黑蓝摩托车后减震器 · 上耳环/弹簧/阻尼筒/下 U 形夹/编织软管/副储油罐分部件重建 · 六视图为未标定的外观匹配参考</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b> 本候选不替换原模型，不进入自动放置或正式合成。
所有 IoU 均为未标定诊断，不是验收通过。背面、内部油路、真实尺寸、落稳和现场光照仍待人工复核。
浏览器交互检查未执行；本页用 Blender 渲染图为离线证据。</div></section>

<section><h2>旋转检查（24 方向实测渲染）</h2><div class="viewer">
<div class="turnbox" id="turnbox"><img id="turn" src="{base}/renders/turn-00.png" alt="转台"></div>
<aside><p>拖动图片旋转；也可自动播放。</p><p><button id="play">自动旋转</button> <button id="reset">回到 000°</button></p>
<p>当前角度：<b id="angle">000°</b></p><p>转台只覆盖外观，不证明隐藏结构、同部件装配或落稳。</p>
<p><a href="{base}/mesh.glb">下载 GLB</a><br><a href="{base}/model.blend">下载 Blender 可编辑文件</a></p></aside></div></section>

<section><h2>对照图</h2><div class="grid">{gallery}</div></section>

<section><h2>实际执行的检查</h2><table>{metric_rows}</table>
<p class="notice" style="margin-top:14px">图集检查只证明贴图数值与变化；金属变光只证明高光移动；开孔/弹簧射线只覆盖上耳环、下 U 形夹、弹簧和储油罐端面几何。
未执行项已明确写为“未执行”，不能解读为通过。</p></section>

<section><h2>待人工复核</h2><ul>{pending}</ul></section>

<section><h2>文件与记录</h2>
<p><a href="{base}/result.json">result.json</a> · <a href="{base}/hashes.json">hashes.json</a> ·
<a href="{base}/source-measurements.json">原照测量</a> ·
<a href="{base}/old-model-audit.json">旧模型审计</a> ·
<a href="{base}/worker-result.json">Blender worker 记录</a> ·
<a href="{base}/report.md">中文报告</a></p>
<p>贴图：<a href="{base}/base_color.png">底色</a> ·
<a href="{base}/metallic_roughness.png">金属/粗糙度</a> ·
<a href="{base}/normal.png">法线</a> · <a href="{base}/label_cap.png">YU-AN 标签</a> ·
<a href="{base}/label_band.png">银色带</a> · <a href="{base}/材质图集-布局.jpg">图集布局</a></p>
<p>候选为 <code>{VERSION}</code>，参考角色 <code>visual_match_unverified</code>；
原始模型和共享页面未被修改。串行整合后再接入共享查看器与人工验收。</p></section>
</main>
<script>
const frames=[{turn_frames}];
let index=0,timer=null,dragging=false,lastX=0;
const img=document.getElementById('turn'),angle=document.getElementById('angle'),play=document.getElementById('play');
function draw(){{img.src=frames[index];angle.textContent=String(index*15).padStart(3,'0')+'°';}}
function step(delta){{index=(index+delta+24)%24;draw();}}
document.getElementById('turnbox').addEventListener('pointerdown',e=>{{dragging=true;lastX=e.clientX;e.target.setPointerCapture(e.pointerId);}});
document.getElementById('turnbox').addEventListener('pointermove',e=>{{if(!dragging)return;const dx=e.clientX-lastX;if(Math.abs(dx)>=12){{step(dx>0?1:-1);lastX=e.clientX;}}}});
document.getElementById('turnbox').addEventListener('pointerup',()=>{{dragging=false;}});
document.getElementById('turnbox').addEventListener('pointercancel',()=>{{dragging=false;}});
document.getElementById('reset').addEventListener('click',()=>{{index=0;draw();}});
play.addEventListener('click',()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='自动旋转';}}else{{timer=setInterval(()=>step(1),180);play.textContent='停止旋转';}}}});
draw();
</script></body></html>'''
    (HERE / 'model-review-27.html').write_text(html, encoding='utf8')


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_hashes(out, inputs, scripts, blender_version):
    outputs = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'hashes.json':
            outputs[str(path.relative_to(ROOT)).replace('\\', '/')] = {
                'sha256': hash_file(path), 'bytes': path.stat().st_size}
    return {
        'inputs': {name: {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                          'sha256': digest(path)} for name, path in inputs.items()},
        'scripts': {name: {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                           'sha256': digest(path)} for name, path in scripts.items()},
        'blender': blender_version,
        'outputs': outputs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=27)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != NUMBER:
        p.error('this entry supports item 27 only')
    a.reference = (a.reference or find_reference()).resolve()
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    if a.out is None:
        a.out = (ROOT / 'work/direct-deepseek/27-draft' if a.draft
                 else ROOT / 'outputs/27液压减震器-六视图重建-v1')
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
    config_path = HERE / 'shock27_config.json'
    worker_path = HERE / 'mesh_rebuild_27_worker.py'
    profile = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    scripts = {'driver': Path(__file__), 'worker': worker_path,
               'config': config_path}
    checks_path = HERE / 'checks_rebuild_27.py'
    if checks_path.is_file():
        scripts['checks'] = checks_path
    signature_input = {
        'reference': digest(a.reference), 'mesh': row['mesh_sha256'],
        'rust': row['rust_sha256'], 'original': row['clean_sha256'],
        'source_alpha': digest(alpha_path), 'source_view': digest(source_view_path),
        'source_camera': row['reconstruction']['source_camera'],
        'config': digest(config_path), 'driver': digest(Path(__file__)),
        'worker': digest(worker_path),
        'helpers': {name: digest(HERE / name) for name in
                    ('blender_worker.py', 'mesh_checks.py', 'mesh_carve.py',
                     'mesh_rebuild_item.py', 'scene_v2.py')},
        'blender': version, 'python': sys.version.split()[0],
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'pillow': PIL.__version__, 'trimesh': trimesh.__version__,
        'draft': a.draft,
    }
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if a.out.exists() and any(a.out.iterdir()):
        record = a.out / 'result.json'
        mesh = a.out / 'mesh.glb'
        if record.is_file() and mesh.is_file():
            old = json.loads(record.read_text(encoding='utf8'))
            required = [a.out / 'model.blend', a.out / 'base_color.png',
                        a.out / 'metallic_roughness.png', a.out / 'normal.png',
                        a.out / 'label_cap.png', a.out / 'label_band.png',
                        a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '六视图-新旧对照.jpg', a.out / '局部检查.jpg',
                             a.out / '分部件隔离.jpg', a.out / '材质-同光照对照.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '开孔-隔离检查.jpg',
                             a.out / '原视角诊断.jpg', a.out / '原照-新候选对照.jpg',
                             a.out / '转台-24方向.jpg', a.out / '材质图集-对照.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
                required += [a.out / 'renders' / f'detail-{name}.png'
                             for name in DETAIL_NAMES]
                required += [a.out / 'renders' / f'{name}.png'
                             for name in ISOLATION_NAMES]
            if (old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh)
                    and all(path.is_file() for path in required)):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    measurements = source_measurements(original_path, rust_path, profile)
    (a.out / 'source-measurements.json').write_text(
        json.dumps(measurements, ensure_ascii=False, indent=2), encoding='utf8')
    old_audit = old_model_audit({**row, 'mesh': str(mesh_path)})
    audit = audit_row(NUMBER)
    (a.out / 'old-model-audit.json').write_text(
        json.dumps({'old': old_audit, 'audit': audit}, ensure_ascii=False, indent=2),
        encoding='utf8')
    textures, texture_evidence = make_atlases(rust_path, original_path, a.out, profile)
    request = {'out': str(a.out), 'original_mesh': str(mesh_path), 'textures': textures,
               'config': profile, 'source_camera': row['reconstruction']['source_camera'],
               'draft': a.draft}
    (a.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2),
                                        encoding='utf8')
    with (a.out / 'blender.log').open('w', encoding='utf8') as log:
        process = subprocess.run([str(a.blender), '--background', '--factory-startup',
                                  '--python-exit-code', '1',
                                  '--python', str(worker_path), '--',
                                  str(a.out / 'request.json')],
                                 stdout=log, stderr=subprocess.STDOUT, timeout=5400)
    if process.returncode:
        raise RuntimeError(f'Blender failed; see {a.out / "blender.log"}')

    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    mesh_info = inspect_mesh(a.out / 'mesh.glb')
    diagnostics = source_diagnostics(a.out, profile, alpha_path, source_view_path)
    openings = opening_checks(a.out, worker)
    metal = metal_reflection_check(a.out) if not a.draft else {}
    textures_checks = texture_checks(a.out, profile)

    mesh_sha = digest(a.out / 'mesh.glb')
    result = {
        'number': NUMBER,
        'version': VERSION,
        'status': 'review',
        'draft': bool(a.draft),
        'manual_only': True,
        'asset_id': row['asset_id'],
        'original_mesh_sha256': row['mesh_sha256'],
        'mesh_sha256': mesh_sha,
        'blend_sha256': digest(a.out / 'model.blend'),
        'shape_basis': 'source_photo_visible_structure+uncalibrated_six_view_layout',
        'reference': {'path': str(a.reference.relative_to(ROOT)).replace('\\', '/'),
                      'sha256': digest(a.reference), 'role': 'visual_match_unverified',
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
            'reference': {'path': str(a.reference.relative_to(ROOT)).replace('\\', '/'),
                          'sha256': digest(a.reference)},
        },
        'measurements': measurements,
        'texture_evidence': texture_evidence,
        'old_audit': old_audit,
        'audit': audit,
        'worker': worker,
        'mesh': {**mesh_info, 'parts': mesh_info['parts'],
                 'closed_after_welding_seams': mesh_info['closed_after_welding_seams'],
                 'open_parts': mesh_info['open_parts'],
                 'nonfinite_parts': mesh_info['nonfinite_parts'],
                 'polygons': worker['polygons'], 'triangles': worker['triangles']},
        'textures': textures_checks,
        'diagnostics': diagnostics,
        'openings': openings,
        'metal': metal,
        'versions': {'blender': version, 'python': sys.version.split()[0],
                     'numpy': np.__version__, 'opencv': cv2.__version__,
                     'pillow': PIL.__version__, 'trimesh': trimesh.__version__},
        'render': profile['render'],
        'checks': {
            'mesh_closed': not mesh_info['open_parts'],
            'no_nonfinite': not mesh_info['nonfinite_parts'],
            'no_self_intersections': not worker['self_intersecting_components'],
            'texture_values': textures_checks['passed'],
            'top_eye_through': bool(openings['checks']['top_eye_through']),
            'bottom_eye_through': bool(openings['checks']['bottom_eye_through']),
            'two_bushings_present': bool(openings['checks']['two_bushings_present']),
            'two_bottom_bolts': bool(openings['checks']['two_bottom_bolts']),
            'spring_open_helix': bool(openings['checks']['spring_open_helix']),
            'reservoir_socket': bool(openings['checks']['reservoir_socket']),
            'twelve_knurls': bool(openings['checks']['twelve_knurls']),
            'metal_highlights_move': bool(metal.get('passed')),
            'nonmetal_metal_reflection_applicable': False,
            'browser_review_executed': False,
            'source_fidelity_acceptance': False,
        },
        'pending_review': [
            '六视图参考未标定且各面板对夹板朝向/软管方位不一致；工作尺度、弹簧线径、耳环孔径、螺纹规格、孔深和装配关系未实测。',
            '原照是 3/4 视角；严格原相机与包围盒对齐 IoU 只作诊断。',
            '背面、阻尼筒内部、活塞/阀片/油路、储油罐内部、软管内部通道、衬套配合和壁厚为推断。',
            '上耳环夹板朝向、下 U 形夹开口方向和软管路径中段为解释性重建。',
            'YU-AN 标签文字按可见内容复制，字形为近似绘制；储油罐银色带刻字未准确复现。',
            '锈迹、漆面磨损和编织软管纹理为近似；不能把参考当实测。',
            '副储油罐仅由软管悬吊；落稳、接触、现场尺度、遮挡顺序与环境光融合未验证。',
            '浏览器交互检查未执行（本工作树缺少本地 three.js/node_modules）；需串行整合后人工复核。',
            '本候选不替换原模型，不进入自动放置或正式合成。',
        ],
        'limitations': profile['limitations'] + [
            '轮廓 IoU 与包围盒对齐只作诊断，不是标定测量。',
            '材质图集分布与金属变光检查只证明数值/高光变化，不证明真实表面。',
        ],
        'signature': signature,
    }
    result['texture_paths'] = {
        k: str(Path(v).relative_to(ROOT)).replace('\\', '/')
        for k, v in textures.items()
        if k in ('base_color', 'metallic_roughness', 'normal',
                 'label_cap', 'label_band')}
    (a.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    six_view_board(a.out)
    atlas_board(a.out)
    if not a.draft:
        detail_board(a.out)
        parts_board(a.out)
        material_board(a.out)
        metal_board(a.out, metal)
        opening_board(a.out, openings)
        source_board(a.out, diagnostics)
        original_board(a.out)
        turntable_board(a.out)
        write_review_page(a.out, result)
    write_report(a.out, result)
    inputs = {'original': original_path, 'rust': rust_path,
              'old_mesh': mesh_path, 'source_view': source_view_path,
              'source_alpha': alpha_path, 'reference': a.reference,
              'config': config_path}
    hashes = collect_hashes(a.out, inputs, scripts, version)
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    summary = {'status': 'review', 'manual_only': True, 'out': str(a.out),
               'mesh': str(a.out / 'mesh.glb'), 'blend': str(a.out / 'model.blend'),
               'checks': result['checks'],
               'strict_iou': diagnostics['strict']['iou'],
               'best_fit_iou': diagnostics['best'].get('iou'),
               'top_eye_through': openings.get('top_eye_through'),
               'bottom_eye_through': openings.get('bottom_eye_through'),
               'bushings_present': openings.get('bushings_present'),
               'spring_open': openings.get('spring_open')}
    if not a.draft:
        progress = {
            'number': NUMBER,
            'status': 'review',
            'out': str(a.out),
            'mesh': str(a.out / 'mesh.glb'),
            'blend': str(a.out / 'model.blend'),
            'checks': {
                'mesh_closed': result['checks']['mesh_closed'],
                'no_nonfinite': result['checks']['no_nonfinite'],
                'no_self_intersections': result['checks']['no_self_intersections'],
                'texture_values': result['checks']['texture_values'],
                'top_eye_through': openings.get('top_eye_through'),
                'bottom_eye_through': openings.get('bottom_eye_through'),
                'bushings_present': openings.get('bushings_present'),
                'bottom_bolts': openings.get('bottom_bolts'),
                'spring_open_helix': openings.get('spring_open'),
                'spring_rays_hit': openings.get('spring_hits'),
                'spring_rays': openings.get('spring_rays'),
                'reservoir_socket': result['checks']['reservoir_socket'],
                'knurl_count': openings.get('knurl_count'),
                'metal_highlights_move': result['checks']['metal_highlights_move'],
                'browser_review_executed': False,
                'source_fidelity_acceptance': False,
                'source_strict_iou': diagnostics['strict']['iou'],
                'source_front_box_fit_iou': (diagnostics.get('front_box_fit') or {}).get('iou'),
                'turntable_best_box_fit_iou': (diagnostics.get('best') or {}).get('iou'),
                'note': '轮廓指标均为未标定诊断，不作验收通过。',
            },
            'pending_review': result['pending_review'],
            'next_number': 28,
        }
        (ROOT / 'work/direct-deepseek/27-progress.json').write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
