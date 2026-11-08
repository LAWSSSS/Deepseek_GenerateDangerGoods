"""Build the review-only part-based candidate for item 26 (red oil cylinder 1).

The source photograph is a three-quarter view of a red welded hydraulic
cylinder: a single eye clevis with a steel bushing, a double-groove collar, a
barrel with a large barcode label, two welded side ports at different axial
stations on the same visible side, and a bottom cap with a short painted rod
stub containing a small central blind hole.  The old single-image TripoSR mesh
is a soft low-poly blob that fuses the ports and makes the label unreadable, so
it is retained only as a comparison.

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
NUMBER = 26
VERSION = 'cylinder26-parametric-v1'
REFERENCE_NAME = 'ChatGPT 图像 2026年10月14日 10_19_49.png'
REFERENCE_SHA256 = '0ae609e44cfa7d25c0a71c53dcb4518da269aeef62442b4270658423039d19c2'
SOURCE_SIZE = (1254, 1254)
RUST_SIZE = (1254, 1254)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
TILE_NAMES = {
    0: '红漆', 1: '暗红漆', 2: '钢衬套/孔口环', 3: '深色钢孔内',
    4: '焊缝红漆', 5: '锈蚀色', 6: '橡胶/暗槽', 7: '印刷标签',
    8: '备用8', 9: '备用9', 10: '备用10', 11: '备用11',
    12: '备用12', 13: '备用13', 14: '备用14', 15: '备用15',
}
METAL_TILES = ('steel', 'dark_steel')
DIELECTRIC_TILES = ('red', 'red_dark', 'weld', 'rust', 'rubber', 'label')
DETAIL_NAMES = ('eye-bushing', 'eye-bore', 'collar', 'label', 'port-upper',
                'port-lower', 'bottom-cap', 'bottom-stub', 'barrel-seam',
                'overall-3q')
ISOLATION_NAMES = ('eye-only', 'collar-only', 'barrel-only',
                   'port-upper-only', 'port-lower-only', 'bottom-end-only',
                   'new-material-front', 'new-clay-front',
                   'new-light-alternate', 'metal-only-light-a',
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
    """Top-down image rect; the worker mirrors the v coordinate for Blender."""
    col, row = idx % 4, idx // 4
    x0, y0 = col * tile_size, row * tile_size
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


def draw_label_art(size):
    """Approximate the visible printed layout; text and brand are not copied."""
    w, h = 1024, 512
    art = Image.new('RGB', (w, h), (242, 242, 239))
    d = ImageDraw.Draw(art)
    # faint paper noise
    rng = np.random.default_rng(2626)
    noise = rng.normal(0, 3.2, (h, w, 1)).repeat(3, axis=2)
    arr = np.clip(np.asarray(art, np.float32) + noise, 0, 255).astype(np.uint8)
    art = Image.fromarray(arr, 'RGB')
    d = ImageDraw.Draw(art)
    # barcode
    x = 40
    rng2 = np.random.default_rng(26261)
    while x < 252:
        width = int(rng2.integers(3, 13))
        gap = int(rng2.integers(2, 9))
        if x + width > 258:
            break
        d.rectangle((x, 120, x + width, 420), fill=(15, 15, 15))
        x += width + gap
    for i in range(4):
        d.rectangle((42, 436 + i * 9, 248, 440 + i * 9), fill=(58, 58, 58))
    # thin red alignment bar
    d.rectangle((512, 120, 520, 426), fill=(190, 26, 30))
    # large character-like mark (the photograph shows a large Chinese glyph;
    # the exact word is not legible and is not reproduced as a brand)
    try:
        font_big = ImageFont.truetype(LABEL_FONT, 250)
        d.text((300, 78), '中', font=font_big, fill=(16, 16, 16))
    except OSError:
        d.rectangle((330, 120, 470, 410), outline=(16, 16, 16), width=18)
        d.rectangle((385, 165, 415, 365), fill=(16, 16, 16))
    # small black text-like lines under the big mark
    for i, width in enumerate((118, 96, 132, 84)):
        d.rectangle((310, 350 + i * 22, 310 + width, 356 + i * 22),
                    fill=(48, 48, 48))
    # black logo / code patch
    d.rectangle((566, 54, 760, 158), fill=(14, 14, 14))
    for i, (gx, gy, gw, gh) in enumerate((
            (584, 72, 12, 44), (604, 72, 30, 12), (604, 104, 30, 12),
            (646, 72, 12, 44), (668, 72, 30, 12), (668, 104, 30, 12),
            (710, 72, 12, 44), (730, 72, 18, 30))):
        d.rectangle((gx, gy, gx + gw, gy + gh), fill=(235, 235, 232))
    for i in range(3):
        d.rectangle((580, 170 + i * 10, 748, 174 + i * 10), fill=(70, 70, 70))
    # red warning triangle with white exclamation mark
    d.polygon(((596, 350), (716, 350), (656, 236)), fill=(190, 24, 28))
    d.polygon(((612, 340), (700, 340), (656, 252)), fill=(242, 242, 239))
    d.polygon(((618, 336), (694, 336), (656, 260)), fill=(190, 24, 28))
    d.rectangle((649, 286, 663, 318), fill=(242, 242, 239))
    d.rectangle((649, 324, 663, 333), fill=(242, 242, 239))
    # small instruction-like lines at the right
    for i, width in enumerate((156, 132, 148, 104)):
        d.rectangle((580, 384 + i * 18, 580 + width, 389 + i * 18),
                    fill=(52, 52, 52))
    return np.asarray(art, np.uint8)


def make_atlases(rust_path, original_path, out, profile):
    rust = Image.open(rust_path).convert('RGBA')
    if rust.size != RUST_SIZE:
        raise ValueError('26 做锈素材尺寸改变，采样坐标需要重新核对')
    original = Image.open(original_path).convert('RGBA')
    if original.size != SOURCE_SIZE:
        raise ValueError('26 原图尺寸改变，测量坐标需要重新核对')
    size = profile['texture']['size']
    tile = profile['texture']['tile']
    tiles = profile['texture']['tiles']
    rng = np.random.default_rng(26026)
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
    # rust palette from the corroborating rust photograph, not a fixed swatch.
    # The object itself is red, so a plain "red > blue" test would select the
    # paint instead of the rust.  Require a brown hue and a real difference from
    # the clean photograph.
    a = np.asarray(rust)
    clean = np.asarray(original.convert('RGB')).astype(float)
    mask = a[:, :, 3] >= 128
    rr, gg, bb = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    diff = np.linalg.norm(a[:, :, :3].astype(float) - clean, axis=2)
    hsv_rust = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2HSV)
    hr = hsv_rust[:, :, 0].astype(int)
    sr = hsv_rust[:, :, 1].astype(int)
    vr = hsv_rust[:, :, 2].astype(int)
    brown = np.logical_and.reduce([
        mask, diff > 12, hr >= 8, hr <= 32, sr >= 35, vr >= 40,
        rr > gg, gg > bb, rr >= 70, gg <= 150])
    strong = np.logical_and(brown, diff > 25)
    if strong.sum() >= 500:
        rust_rgb = np.median(a[:, :, :3][strong], axis=0).astype(np.float32)
    elif brown.sum() >= 500:
        rust_rgb = np.median(a[:, :, :3][brown], axis=0).astype(np.float32)
    else:
        rust_rgb = np.array([125, 84, 58], np.float32)
    dark_oxide = rust_rgb * .52
    red_detail = safe_detail(rust, (560, 560, 620, 620), tile)
    red2_detail = safe_detail(rust, (360, 760, 420, 820), tile)
    weld_detail = safe_detail(rust, (720, 760, 780, 820), tile)
    steel_detail = safe_detail(rust, (1080, 220, 1140, 280), tile)
    dark_detail = safe_detail(rust, (1050, 180, 1090, 220), tile)

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
            normal[sl] = normal_from_height(height, .35)

    specs = {
        'red': (red_detail, 'red'),
        'red_dark': (red2_detail, 'red_dark'),
        'steel': (steel_detail, 'steel'),
        'dark_steel': (dark_detail, 'dark_steel'),
        'weld': (weld_detail, 'weld'),
        'rust': (np.zeros((tile, tile), np.float32), 'rust'),
        'rubber': (micro - .5, 'rubber'),
    }
    for name, (detail, key) in specs.items():
        idx = tiles[name]
        mat = profile['material_profile'][key]
        base_rgb = np.asarray(mat['base_rgb'], np.float32)[None, None, :]
        amount = float(mat.get('rust_amount', 0.0))
        bright = (1 + detail[:, :, None] * .0022 +
                  (coarse[:, :, None] - .5) * .030 +
                  (fine[:, :, None] - .5) * .012 +
                  (micro[:, :, None] - .5) * .014)
        if name in ('steel', 'dark_steel'):
            bright = bright * (1 + brush[:, :, None] * .020)
        if name in ('red', 'red_dark', 'weld'):
            bright = bright * (1 + brush2[:, :, None] * .010)
        color = base_rgb * bright
        if amount > 0:
            blotch_mask = np.clip((blotch - (.94 - .08 * amount)) / .05, 0, 1)
            spot_mask = np.clip((fine - .95) / .05, 0, 1) * \
                np.clip((coarse - .50) / .45, 0, 1)
            edge_mask = np.clip((coarse - (.88 - .04 * amount)) / .08, 0, 1)
            cap = .80 if name == 'weld' else .55
            cov = np.clip(edge_mask * amount * .45 +
                          blotch_mask * min(.70, .10 + amount * 9.0) +
                          spot_mask * min(.45, .08 + amount * 3.0), 0, cap)
            color = color * (1 - cov[:, :, None] * .88) + \
                rust_rgb[None, None, :] * cov[:, :, None] * .88
            rough = mat['roughness'] * (1 - cov) + .78 * cov + (fine - .5) * .030
            metal = np.full((tile, tile), mat['metallic'], np.float32) * (1 - cov * .88)
            height = detail * .0018 + cov * (fine - .5) * .34
            if name in ('steel', 'dark_steel'):
                height = height + brush * .05
        else:
            rough = np.full((tile, tile), mat['roughness'], np.float32) + \
                (fine - .5) * .030
            metal = np.full((tile, tile), mat['metallic'], np.float32)
            height = (micro - .5) * .36 + brush * .05
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
        put(idx, color, rough, metal, height, clip_min=6 if name == 'rubber' else 22)
    # label tile: printed dielectric artwork in the central 2:1 band
    idx = tiles['label']
    mat = profile['material_profile']['label']
    base_rgb = np.asarray(mat['base_rgb'], np.float32)[None, None, :]
    color = base_rgb * (1 + (micro[:, :, None] - .5) * .018)
    put(idx, color, mat['roughness'], 0.0, (micro - .5) * .18)
    art = draw_label_art(tile)
    art = cv2.resize(art, (tile, tile // 2), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = tile_rect(idx, size, tile)
    band_top = y0 + (tile - art.shape[0]) // 2
    base[band_top:band_top + art.shape[0], x0:x1] = art
    # spare tiles: neutral mid-grey with subtle variation, never black
    for idx2 in range(8, 16):
        grey = 118 * (1 + (fine[:, :, None] - .5) * .06 +
                      (micro[:, :, None] - .5) * .035)
        put(idx2, grey, .55, 0.0,
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
    for idx2 in range(16):
        col, row = idx2 % 4, idx2 // 4
        x, y = col * tile, row * tile
        sheet.paste(Image.fromarray(base[y:y + tile, x:x + tile]), (x, y))
        draw.rectangle((x, y, x + tile, y + tile), outline=(230, 230, 228), width=2)
        draw.text((x + 10, y + 8), f"{idx2} {TILE_NAMES.get(idx2, '备用')}",
                  fill=(20, 20, 20), font=font)
    sheet.save(out / '材质图集-布局.jpg', quality=93)
    return paths, {'rust_palette_rgb': [int(v) for v in rust_rgb],
                   'rust_brown_pixels': int(brown.sum()),
                   'rust_strong_pixels': int(strong.sum()),
                   'label_note': ('标签条形码、标志块、警告三角和文字行按原照可见版式近似重绘；'
                                  '原始文字/品牌不可辨，未复制，不是对印刷内容的还原。'),
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
    red = np.logical_and.reduce([
        mask,
        np.logical_or(np.less_equal(hsv[:, :, 0], 14), np.greater_equal(hsv[:, :, 0], 166)),
        np.greater_equal(hsv[:, :, 1], 90), np.greater_equal(hsv[:, :, 2], 45)])
    n, _, stats, cent = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    red_components = [{'bbox': [int(s[0]), int(s[1]), int(s[2]), int(s[3])],
                       'area': int(s[4]),
                       'centre': [round(float(c[0]), 2), round(float(c[1]), 2)]}
                      for s, c in zip(stats[1:], cent[1:]) if s[4] >= 800]
    steel = np.logical_and.reduce([
        mask, np.less_equal(hsv[:, :, 1], 48), np.greater_equal(hsv[:, :, 2], 110)])
    n, _, stats, cent = cv2.connectedComponentsWithStats(steel.astype(np.uint8), 8)
    steel_components = [{'bbox': [int(s[0]), int(s[1]), int(s[2]), int(s[3])],
                        'area': int(s[4]),
                        'centre': [round(float(c[0]), 2), round(float(c[1]), 2)]}
                       for s, c in zip(stats[1:], cent[1:]) if s[4] >= 40]
    black = np.logical_and.reduce([
        mask, np.less_equal(hsv[:, :, 2], 75), np.less_equal(hsv[:, :, 1], 90)])
    n, _, stats, cent = cv2.connectedComponentsWithStats(black.astype(np.uint8), 8)
    black_components = [{'bbox': [int(s[0]), int(s[1]), int(s[2]), int(s[3])],
                         'area': int(s[4]),
                         'centre': [round(float(c[0]), 2), round(float(c[1]), 2)]}
                        for s, c in zip(stats[1:], cent[1:]) if s[4] >= 30]
    rust = np.asarray(Image.open(rust_path).convert('RGB')).astype(float)
    diff = np.linalg.norm(rust - a[:, :, :3].astype(float), axis=2)
    rr, gg, bb = rust[:, :, 0], rust[:, :, 1], rust[:, :, 2]
    brown = np.logical_and.reduce([mask, np.greater_equal(rr, bb + 12),
                                   np.greater_equal(rr, gg - 8), np.greater_equal(rr, 70),
                                   np.less_equal(gg, 150)])
    d = profile['dimensions']
    overall = (d['eye']['center_y'] + d['eye']['outer_radius']) - \
        d['bottom_end']['stub_y0']
    max_width = d['ports']['base_z'] + d['ports']['boss_length'] + \
        d['ports']['spigot_length'] + d['ports']['mouth_ring_r_out'] + d['barrel']['radius']
    return {
        'image': str(original_path), 'size': SOURCE_SIZE,
        'alpha_area_px': int(mask.sum()), 'bbox': bbox,
        'min_area_rect': {'centre': [round(float(cx), 2), round(float(cy), 2)],
                          'size': [round(float(w), 2), round(float(h), 2)],
                          'angle_deg': round(float(angle), 2),
                          'length_over_width': round(float(max(w, h) / min(w, h)), 4)},
        'red_components': sorted(red_components, key=lambda z: -z['area'])[:6],
        'steel_components': sorted(steel_components, key=lambda z: -z['area'])[:8],
        'black_label_components': sorted(black_components, key=lambda z: -z['area'])[:8],
        'rust_color_shift': {'mean_rgb_distance': round(float(diff[mask].mean()), 3),
                             'p50': round(float(np.percentile(diff[mask], 50)), 3),
                             'p90': round(float(np.percentile(diff[mask], 90)), 3),
                             'brown_fraction': round(float(brown.sum() / mask.sum()), 4)},
        'configured': {'overall_length_m': round(overall, 4),
                       'max_width_m': round(max_width, 4),
                       'overall_length_over_width': round(overall / max_width, 4),
                       'barrel_radius_m': d['barrel']['radius'],
                       'eye_bore_radius_m': d['eye']['bore_radius'],
                       'eye_outer_radius_m': d['eye']['outer_radius'],
                       'stub_radius_m': d['bottom_end']['stub_radius']},
        'note': ('原照是 3/4 视角，minAreaRect 只描述照片中的可见轮廓；'
                 '工作尺度按合理比例设置，不是实测。红色/钢色/黑色组件用于确认漆面、'
                 '耳环衬套和标签的可见性，不证明真实尺寸或材料。'),
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
        d, _ = tree.query(c.vertices)
        fragments.append({'vertices': len(c.vertices),
                          'bounds': np.round(c.bounds, 4).tolist(),
                          'min_gap_m': round(float(d.min()), 4)})
    centered = main.vertices - main.vertices.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    worker = row['reconstruction']['worker']
    return {
        'mesh': row['mesh'], 'mesh_sha256': row['mesh_sha256'],
        'vertices': int(worker.get('vertices', 0)),
        'faces': int(worker.get('faces', 0)),
        'export_vertices': int(len(merged.vertices)),
        'export_faces': int(len(merged.faces)),
        'bounds': np.round(merged.bounds, 4).tolist(),
        'components': len(comps),
        'main_export_vertices': len(main.vertices),
        'fragments': fragments,
        'principal_axis': np.round(axis, 4).tolist(),
        'principal_axis_note': ('旧网格主轴不与库坐标的 Y 轴对齐；六方向渲染显示表面模糊、'
                                '油口与缸筒粘连、标签不可辨。'),
        'old_main_length_over_width': round(float(max(main.bounds[1] - main.bounds[0]) /
                                                   max(1e-9, min(main.bounds[1] - main.bounds[0]))), 4),
        'old_main_depth_m': round(float(main.bounds[1][0] - main.bounds[0][0]), 4),
        'note': ('旧网格是单张做锈图的 TripoSR 结果，库记录仅约 4.8k 顶点、9.7k 面，'
                 '表面严重模糊；GLB 导出按三角面拆分顶点后统计约 5.1 万顶点（不是新几何）。'
                 '两个油口与缸筒粘连、耳环孔形状不规则、标签不可辨。原视角轮廓诊断通过旧门槛'
                 '（IoU 约 0.927、未填孔），但单一轮廓不能证明结构，故本件不复用旧网格，'
                 '改为按可见结构分部件重建。'),
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
        items.append((f'新版 · {view}', '分部件重建',
                      _load_rgba(out / 'renders' / f'new-{view}.png')))
    for view in VIEWS:
        items.append((f'旧模型 · {view}', '原单图重建（对照）',
                      _load_rgba(out / 'renders' / f'old-{view}.png')))
    return make_board(out, items, 6, '26 油缸1 · 六方向新旧对照',
                      '同一中性光、正交相机；旧模型保留原样，仅作对照。新版为分部件重建，'
                      '不用平滑掩盖缺失结构。库坐标 left/right 为沿缸轴的两端视图。',
                      path_name='六视图-新旧对照.jpg')


def detail_board(out):
    specs = [
        ('eye-bushing', '单耳环 / 钢衬套'),
        ('eye-bore', '耳环贯穿孔'),
        ('collar', '上端套环 / 两道凸环与暗槽'),
        ('label', '缸筒正面标签（近似重绘）'),
        ('port-upper', '上油口 / 焊缝 / 螺纹短接头 / 盲孔'),
        ('port-lower', '下油口 / 焊缝 / 螺纹短接头 / 盲孔'),
        ('bottom-cap', '底端加宽环带 / 端盖'),
        ('bottom-stub', '底端杆头 / 中心盲孔'),
        ('barrel-seam', '缸筒上接缝'),
        ('overall-3q', '整体 3/4 视角'),
    ]
    items = [(f'局部 · {label}', '同光照近景',
              _load_rgba(out / 'renders' / f'detail-{name}.png'))
             for name, label in specs]
    return make_board(out, items, 5, '26 油缸1 · 关键局部检查',
                      '局部近景用于检查耳环孔、衬套、套环、标签、两个油口的焊缝/螺纹/盲孔、'
                      '底端环带和杆头盲孔；不能据此证明不可见背面、内部通道或真实尺寸。',
                      path_name='局部检查.jpg')


def parts_board(out):
    items = [
        ('耳环组件', '厚板/贯穿孔/钢衬套/凸起环', _load_rgba(out / 'renders' / 'eye-only.png')),
        ('上端套环组件', '套环/两道凸环/暗槽/焊缝', _load_rgba(out / 'renders' / 'collar-only.png')),
        ('缸筒组件', '缸筒/上下接缝', _load_rgba(out / 'renders' / 'barrel-only.png')),
        ('上油口组件', '凸台/焊缝/螺纹短接头/盲孔', _load_rgba(out / 'renders' / 'port-upper-only.png')),
        ('下油口组件', '凸台/焊缝/螺纹短接头/盲孔', _load_rgba(out / 'renders' / 'port-lower-only.png')),
        ('底端组件', '环带/端盖/锥形过渡/杆头/盲孔', _load_rgba(out / 'renders' / 'bottom-end-only.png')),
    ]
    return make_board(out, items, 2, '26 油缸1 · 分部件隔离检查',
                      '隔离渲染用于确认零件独立存在与相对位置；只覆盖当前视角，装配关系仍待审。',
                      path_name='分部件隔离.jpg')


def material_board(out):
    items = [
        ('新版材质正视', 'PBR 图集 / 中性光', _load_rgba(out / 'renders' / 'new-material-front.png')),
        ('新版灰度正视', '同光照灰模', _load_rgba(out / 'renders' / 'new-clay-front.png')),
        ('旧模型正视', '旧顶点色网格', _load_rgba(out / 'renders' / 'old-front.png')),
    ]
    return make_board(out, items, 3, '26 油缸1 · 同光照材质/灰模对照',
                      '红漆、橡胶暗槽、锈和印刷标签为电介质；钢衬套、孔口环和深色孔内衬为金属，'
                      '保留随光位变化的反射。金属反射检查不适用于非金属。',
                      path_name='材质-同光照对照.jpg')


def metal_board(out, metrics):
    items = [
        ('主光 A', '金属件隔离（当前取景为耳环钢衬套）',
         _load_rgba(out / 'renders' / 'metal-only-light-a.png')),
        ('主光 B', '同一相机，仅移动主光',
         _load_rgba(out / 'renders' / 'metal-only-light-b.png')),
    ]
    note = (f"当前取景为耳环钢衬套；油口孔口环/孔内衬见局部图。"
            f"非背景像素差分 {metrics['mean_abs_difference']:.2f}/255；"
            f"最亮 1% 高光质心位移 {metrics['highlight_shift_px']:.1f} px；"
            f"高阈值 {metrics.get('high_threshold_luma', 0):.0f} 位移 "
            f"{metrics.get('high_threshold_shift_px', 0):.1f} px；"
            f"金属像素 {metrics['metal_pixels']}。仅证明金属高光随光位移动，不证明材质测量正确。")
    return make_board(out, items, 2, '26 油缸1 · 金属反射变光检查', note,
                      cell=360, path_name='金属反射-变光对照.jpg')


def opening_board(out, counts):
    items = [
        ('耳环贯穿孔', f"{'通过' if counts['checks']['eye_hole_through'] else '未通过'}",
         _load_rgba(out / 'renders' / 'eye-only.png')),
        ('上油口盲孔', f"{'通过' if counts['checks']['two_port_mouths_open'] else '未通过'}",
         _load_rgba(out / 'renders' / 'port-upper-only.png')),
        ('下油口盲孔', f"{'通过' if counts['checks']['two_port_mouths_open'] else '未通过'}",
         _load_rgba(out / 'renders' / 'port-lower-only.png')),
        ('底端杆头盲孔', f"{'通过' if counts['checks']['stub_hole_open'] else '未通过'}",
         _load_rgba(out / 'renders' / 'detail-bottom-stub.png')),
        ('标签', f"{'存在' if counts['checks']['label_present'] else '未找到'}",
         _load_rgba(out / 'renders' / 'detail-label.png')),
    ]
    note = (f"射线检查：耳环贯穿孔 {'通过' if counts['checks']['eye_hole_through'] else '未通过'}；"
            f"钢衬套 {'存在' if counts['checks']['bushing_present'] else '未找到'}；"
            f"油口 {counts.get('ports')}/2，孔口开放 {counts.get('port_mouths_open')}/2；"
            f"杆头盲孔 {'通过' if counts['checks']['stub_hole_open'] else '未通过'}。"
            '只证明这些方向/位置的几何开孔与可见盲孔，不代表内部油路或隐藏结构。')
    return make_board(out, items, 3, '26 油缸1 · 开孔与隔离检查', note,
                      cell=340, path_name='开孔-隔离检查.jpg')


def source_board(out, metrics):
    items = [
        ('旧库源视图（诊断）', '输入源视图 / 非本件原照',
         Image.open(metrics['source_view']).convert('RGBA')),
        ('新版严格原相机', f"IoU {metrics['strict']['iou']:.4f}（仅诊断）",
         _load_rgba(out / 'renders' / 'new-source.png')),
        (f"包围盒对齐最优 {metrics['best_view']}°", f"IoU {metrics['best']['iou']:.4f}（仅诊断）",
         _load_rgba(out / 'renders' / f"turn-{int(metrics['best_view']) // 15:02d}.png")),
    ]
    note = ('旧模型库相机源自单图 TripoSR 的任意朝向，本件已按库约定重建为轴线沿 Y；'
            '严格原相机与包围盒对齐都不是标定，IoU 只作诊断，不构成验收通过。')
    return make_board(out, items, 3, '26 油缸1 · 原视角轮廓诊断', note,
                      path_name='原视角诊断.jpg')


def original_board(out):
    orig = Image.open(ROOT / 'outputs/物品高清输入-39件/原貌HD/26-油缸1.png').convert('RGBA')
    new = _load_rgba(out / 'renders' / 'new-material-front.png')
    items = [
        ('原貌照片', '3/4 视角，红色漆面与局部锈迹', orig),
        ('新版正视渲染', 'PBR 材质 + 中性光', new),
        ('六视图参考', '未标定的红色液压缸外观匹配', Image.open(
            ROOT / 'references/parallel-six-view' / REFERENCE_NAME).convert('RGBA')),
    ]
    return make_board(out, items, 3, '26 油缸1 · 原照与新候选（附六视图参考）',
                      '参考图未标定且面板互相矛盾（顶视耳环孔方向、油口对置关系与原照不一致）；'
                      '本件以原照为准，参考只用于外观和背侧拓扑，不把参考当实测。',
                      path_name='原照-新候选对照.jpg', cell=360)


def turntable_board(out):
    items = []
    for i in range(24):
        items.append((f'{i * 15:03d}°', '新版转台',
                      _load_rgba(out / 'renders' / f'turn-{i:02d}.png')))
    return make_board(out, items, 6, '26 油缸1 · 24 方向转台',
                      '转台只用于人工旋转检查；六视图参考未标定，不能把轮廓相似称为实测精度。',
                      cell=150, path_name='转台-24方向.jpg')


def atlas_board(out):
    items = [
        ('底色图集', 'sRGB / 2048 px', Image.open(out / 'base_color.png').convert('RGBA')),
        ('金属-粗糙度图集', 'G=粗糙度 / B=金属度',
         Image.open(out / 'metallic_roughness.png').convert('RGBA')),
        ('法线图集', '非颜色 / 低强度', Image.open(out / 'normal.png').convert('RGBA')),
    ]
    return make_board(out, items, 3, '26 油缸1 · PBR 贴图集',
                      '钢/深色钢件金属度按材质设置；红漆、橡胶、锈和标签金属度为 0。'
                      '标签格为近似重绘的印刷版式。',
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
    result = {'tiles': {}, 'notes': []}
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
    T = result['tiles']
    checks = {}
    metal_thresholds = {'steel': .90, 'dark_steel': .70}
    for name, thr in metal_thresholds.items():
        checks[f'{name}_metal'] = T[name]['metallic_p90'] >= thr
    for name in DIELECTRIC_TILES:
        checks[f'{name}_dielectric'] = T[name]['metallic_p95'] <= .05
    rough_limits = {
        'red': (.20, .50), 'red_dark': (.28, .55), 'steel': (.12, .38),
        'dark_steel': (.25, .55), 'weld': (.30, .60), 'rust': (.68, .95),
        'rubber': (.60, .92), 'label': (.28, .62),
        'spare8': (.40, .70), 'spare9': (.40, .70), 'spare10': (.40, .70),
        'spare11': (.40, .70), 'spare12': (.40, .70), 'spare13': (.40, .70),
        'spare14': (.40, .70), 'spare15': (.40, .70),
    }
    for name, (lo, hi) in rough_limits.items():
        checks[f'{name}_roughness'] = lo <= T[name]['roughness'] <= hi
    checks['bright_not_black'] = (
        all(T[n]['median_luma'] >= 20
            for n in ('red', 'red_dark', 'weld', 'rust', 'label', 'dark_steel'))
        and T['steel']['median_luma'] >= 70)
    checks['rubber_is_dark_but_not_black'] = T['rubber']['median_luma'] >= 6
    checks['label_has_printed_contrast'] = T['label']['base_std'] >= 8.0
    checks['all_tiles_have_variation'] = all(T[n]['tile_std'] >= .6 for n in T)
    checks['normal_not_flat'] = all(T[n]['normal_std'] >= .6 for n in T)
    checks['no_baked_gradient'] = all(
        max(T[n]['luma_gradient_lr'], T[n]['luma_gradient_tb']) <= 10.0
        for n in T if n != 'label')
    checks['rust_on_paint_or_metal'] = any(
        T[n]['rust_pixel_fraction'] >= .0005 for n in ('red', 'red_dark', 'weld'))
    checks['red_hue'] = bool(
        T['red']['mean_rgb'][0] >= T['red']['mean_rgb'][1] + 60
        and T['red']['mean_rgb'][0] >= T['red']['mean_rgb'][2] + 60)
    result['checks'] = checks
    result['passed'] = all(checks.values())
    result['notes'].append('图集统计只证明贴图数值与变化量，不证明纹理在实物上的真实分布。')
    result['notes'].append('锈斑与漆面细节为基于做锈图的有限采样与程序化近似，不是逐像素投影。')
    result['notes'].append('标签格为近似重绘的印刷版式；文字/品牌不可辨且未复制，'
                           '印刷对比不是烘焙光照，故不参与“无烘焙梯度”检查。')
    return result


def source_diagnostics(out, profile, alpha_path, source_view_path):
    source = np.asarray(Image.open(alpha_path).convert('L'))
    rendered = np.asarray(_load_rgba(out / 'renders' / 'new-source.png').getchannel('A'))
    strict = source_fidelity(source, rendered)
    strict['acceptance_gate'] = False
    strict['camera'] = '旧模型库标定相机；本件轴线已按库约定重排，因此只作诊断'
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
    eye = r.get('eye', {})
    bushing = r.get('bushing', {})
    ports = r.get('ports', {})
    stub = r.get('stub', {})
    label = r.get('label', {})
    mouth_open = sum(1 for v in ports.values()
                     if v.get('mouth_open') and v.get('blind_bottom_hit'))
    return {
        'eye_holes_through': 1 if eye.get('through') else 0,
        'bushings_present': 1 if bushing.get('present') else 0,
        'ports': int(r.get('port_count') or 0),
        'port_mouths_open': mouth_open,
        'stub_hole_open': bool(stub.get('hole_open') and stub.get('blind_bottom_hit')),
        'label_present': bool(label.get('present')),
        'details': {'eye': eye, 'bushing': bushing, 'ports': ports,
                    'stub': stub, 'label': label},
        'checks': {
            'eye_hole_through': bool(eye.get('through')),
            'bushing_present': bool(bushing.get('present')),
            'two_ports': int(r.get('port_count') or 0) == 2,
            'two_port_mouths_open': mouth_open == 2,
            'stub_hole_open': bool(stub.get('hole_open') and stub.get('blind_bottom_hit')),
            'label_present': bool(label.get('present')),
        },
        'note': ('射线检查直接打在耳环板、衬套、两个油口凸台/短接头和杆头网格上：'
                 '耳环中心为真实贯穿孔，衬套在孔内可命中；两个油口与杆头孔为真实可见盲孔，'
                 '中心射线穿过孔口并命中孔底。只证明这些几何特征，不证明内部油路、密封或装配关系。'),
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
                      '红漆、橡胶、锈和印刷标签为电介质，本检查不适用于它们。')
    return result


# ---------------------------------------------------------------------------
# Report and hashes
# ---------------------------------------------------------------------------
def write_report(out, result):
    variant = out.name.rsplit('-', 1)[-1]
    lines = [
        f'# 26 油缸1 · 六视图重建 {variant}（manual_only，待审）',
        '',
        '## 结论',
        '',
        '本候选把 26 号红色焊接式单耳环液压缸按可见结构分成独立部件重建：单耳环与钢衬套、'
        '上端双凸环套环、缸筒与上下接缝、两个同侧不同轴向位置的焊接油口（凸台/螺纹短接头/'
        '可见盲孔）、底端加宽环带/端盖/锥形过渡/短杆头与中心盲孔，以及缸筒正面的印刷标签近似。'
        '它不替换原模型，不进入自动放置或正式合成；所有诊断均不是验收通过。',
        '',
        '## 为什么重建而不是保守修复',
        '',
        f"- 旧网格库记录只有 {result['old_audit']['vertices']} 个顶点、"
        f"{result['old_audit']['faces']} 个面（GLB 按三角面拆分后统计约 "
        f"{result['old_audit']['export_vertices']} 顶点），表面严重模糊；"
        '两个油口与缸筒粘连、耳环孔形状不规则、标签不可辨。',
        f"- 原视角审计记录：IoU {result['audit']['source_fidelity']['iou']}，"
        f"被填孔 {result['audit']['source_fidelity']['filled_holes']}，"
        f"通过={result['audit']['source_fidelity']['passed']}；单一轮廓通过不能证明结构，"
        '六方向审计和目视都显示旧网格不适合直接保守修复，因此分部件重建。',
        '',
        '## 输入与参考',
        '',
        '- 原图/做锈图/旧网格/源视图透明度哈希见 `hashes.json`。',
        f"- 六视图：`{REFERENCE_NAME}`，角色 **visual_match_unverified**（未标定）；"
        '只用于外观和背侧拓扑，不当作测量或同一实物。',
        '- 参考面板本身矛盾：顶视把耳环孔画成沿缸轴可见；前视/底视把两个油口画成同轴对置，'
        '而原照显示两个油口在同一侧、轴向位置不同。本件以原照为准。',
        '',
        '## 结构',
        '',
        f"- 工作尺度：总长约 {result['measurements']['configured']['overall_length_m']*1000:.0f} mm、"
        f"缸筒半径 {result['measurements']['configured']['barrel_radius_m']*1000:.1f} mm、"
        f"耳环孔径 {result['measurements']['configured']['eye_bore_radius_m']*2000:.1f} mm、"
        f"杆头半径 {result['measurements']['configured']['stub_radius_m']*1000:.1f} mm。"
        '这些是按合理比例设置的工作尺度，不是实测。',
        '- 单耳环为厚板，真实贯穿孔和孔内钢衬套；颈部落到双凸环上端套环。',
        '- 两个油口在同侧（+Z），上油口约在缸筒中下部，下油口靠近底端；原照可见，'
        '与参考的“同轴对置”矛盾。每个油口由焊接凸台、带螺纹短接头、深色衬管、孔口钢环和'
        '可见盲孔组成。',
        '- 底端由加宽环带、端盖、锥形过渡和短杆头组成；杆头端面中心有真实盲孔和深色孔内衬。',
        f"- 网格：{result['mesh']['parts']} 个部件，{result['mesh']['polygons']} 面；"
        f"闭合部件 {result['mesh']['closed_after_welding_seams']}/{result['mesh']['parts']}；"
        f"同部件自交 {len(result['worker']['self_intersecting_components'])} 个。",
        '',
        '## 材质',
        '',
        '- 红漆、橡胶暗槽、锈和印刷标签为电介质；钢衬套、油口孔口环和深色孔内衬为金属，'
        '金属度按材质设置，保留随光位变化的反射。',
        '- 锈斑处金属度下降（锈为电介质）；红漆、橡胶和标签为非金属，金属反射检查不适用于它们。',
        '- 锈色取自已对齐做锈图的棕色像素中位数；分布为程序化近似，不是逐像素投影。'
        '避免把固定灯光/阴影烘进底色；图集有梯度检查（标签的印刷对比除外）。',
        '- 标签条形码、标志块、红色警告三角和文字行按原照可见版式近似重绘；'
        '原始文字/品牌不可辨，未复制，不是对印刷内容的还原。',
        f"- 图集数值检查：{'通过' if result['textures']['passed'] else '未通过'}；"
        '详见 `result.json` 的 tile 数值。',
        '',
        '## 检查结果（实际执行）',
        '',
        f"- {result['versions']['blender']}，Cycles CPU，六视图 {result['render']['six_view_samples']} 采样；"
        'GLB/Blend/贴图/对照图均由本工作树内 Blender/Python 生成。',
        f"- 严格原相机轮廓：IoU {result['diagnostics']['strict']['iou']}，"
        f"填孔 {result['diagnostics']['strict']['filled_holes']}，细结构召回 "
        f"{result['diagnostics']['strict']['narrow_recall']}（仅诊断，非验收）。",
        f"- 包围盒对齐最优转台 {result['diagnostics']['best'].get('view', 'n/a')}："
        f"IoU {result['diagnostics']['best'].get('iou', 'n/a')}（未标定，仅诊断）。",
        f"- 开孔：耳环贯穿孔 {result['openings'].get('eye_holes_through')}/1；"
        f"钢衬套 {result['openings'].get('bushings_present')}/1；"
        f"油口 {result['openings'].get('ports')}/2；"
        f"油口盲孔开放 {result['openings'].get('port_mouths_open')}/2；"
        f"杆头盲孔 {'通过' if result['openings']['checks']['stub_hole_open'] else '未通过'}；"
        f"标签 {'存在' if result['openings']['checks']['label_present'] else '未找到'}。",
        (f"- 金属变光：差分 {result['metal'].get('mean_abs_difference')}/255，"
         f"最亮 1% 高光质心位移 {result['metal'].get('highlight_shift_px')} px"
         f"（高阈值位移 {result['metal'].get('high_threshold_shift_px')} px）。"
         '仅说明高光随光位移动。'
         if result['metal'] else '- 金属变光：草稿模式未执行。'),
        '- 浏览器交互：未执行（本工作树缺少本地 three.js/node_modules；未安装依赖，也未声称通过）。',
        '',
        '## 待人工复核',
        '',
        '- 六视图参考未标定且面板互相矛盾；工作尺度、缸径、耳环孔径、油口螺纹规格、孔深、'
        '壁厚和装配关系未实测。',
        '- 原照是 3/4 视角；严格原相机与包围盒对齐 IoU 只作诊断。',
        '- 背面、缸筒内部、活塞/密封内部、油口内部通道（建模为可见盲孔）、杆头连接方式和'
        '耳环衬套配合为推断。',
        '- 两个油口按原照放在同一侧且轴向位置不同；参考画成同轴对置，不能同时成立。',
        '- 标签文字/品牌不可辨且未复制；条形码、标志块、警告三角和文字行是近似重绘。',
        '- 锈迹分布、漆面磨损和焊缝形状为近似；钢衬套/孔口环真实材料未检测。',
        '- 落稳、接触、现场尺度、遮挡顺序与环境光融合未验证。',
        '- 浏览器交互检查未执行；需在串行整合时接入共享查看器并由人工复核。',
        '- 本候选不替换原模型，不进入自动放置或正式合成。',
        '',
        '## 文件',
        '',
        '- `mesh.glb`：Z 向上、库坐标约定，纹理内嵌。',
        '- `model.blend`：可编辑几何、PBR 节点、图集已打包。',
        '- `base_color.png` / `metallic_roughness.png` / `normal.png`：2048 PBR 图集。',
        '- `六视图-新旧对照.jpg`、`局部检查.jpg`、`分部件隔离.jpg`、`材质-同光照对照.jpg`、'
        '`金属反射-变光对照.jpg`、`开孔-隔离检查.jpg`、`原视角诊断.jpg`、'
        '`原照-新候选对照.jpg`、`转台-24方向.jpg`、`材质图集-布局.jpg`。',
        '- `result.json`、`hashes.json`、`source-measurements.json`、`old-model-audit.json`、'
        '`worker-result.json`。',
        '',
        '## 复现',
        '',
        '```powershell',
        '& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item_26.py '
        f'--out "outputs/{out.name}-repro"',
        '& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_26.py '
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
        ('开孔/衬套/盲孔隔离', f'{base}/开孔-隔离检查.jpg'),
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
        ('严格原相机轮廓 IoU（未标定，仅诊断）', d['strict']['iou']),
        ('正视图包围盒对齐 IoU（仅诊断）',
         (d.get('front_box_fit') or {}).get('iou', '未执行')),
        ('24 方向包围盒对齐最优 IoU（仅诊断）',
         (d.get('best') or {}).get('iou', '未执行')),
        ('耳环贯穿孔', f"{result['openings'].get('eye_holes_through')}/1"),
        ('孔内钢衬套', f"{result['openings'].get('bushings_present')}/1"),
        ('油口数量 / 盲孔开放', f"{result['openings'].get('ports')}/2 · "
                                 f"{result['openings'].get('port_mouths_open')}/2"),
        ('杆头盲孔', '通过' if result['openings']['checks']['stub_hole_open'] else '未通过'),
        ('标签', '存在' if result['openings']['checks']['label_present'] else '未找到'),
        ('网格部件/闭合', f"{result['mesh']['parts']} / {result['mesh']['closed_after_welding_seams']}"),
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
<title>26 油缸1 · 三维模型检查（待审）</title>
<style>
:root{{--ink:#1d2320;--muted:#5f6a64;--line:#d5d9d5;--panel:#fbfcfa;--accent:#8a2f2f}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.65 "Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:#eef1ee}}
header{{background:#3a2422;color:#f7f1ee;padding:26px 30px}}header h1{{margin:0 0 6px;font-size:25px}}
header p{{margin:0;color:#e0cfc8}}main{{max-width:1180px;margin:24px auto 60px;padding:0 18px}}
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
<header><h1>26 油缸1 · 六视图重建（{VERSION}） <span class="tag">manual_only · 待审</span></h1>
<p>红色焊接式单耳环液压缸 · 耳环/衬套/套环/两个同侧油口/底端杆头/标签分部件重建 · 六视图为未标定的外观匹配参考</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b> 本候选不替换原模型，不进入自动放置或正式合成。
所有 IoU 均为未标定诊断，不是验收通过。参考六视图存在面板矛盾；背面、内部油路、真实尺寸、落稳和现场光照仍待人工复核。
浏览器交互检查未执行；本页用 Blender 渲染图为离线证据。</div></section>

<section><h2>旋转检查（24 方向实测渲染）</h2><div class="viewer">
<div class="turnbox" id="turnbox"><img id="turn" src="{base}/renders/turn-00.png" alt="转台"></div>
<aside><p>拖动图片旋转；也可自动播放。</p><p><button id="play">自动旋转</button> <button id="reset">回到 000°</button></p>
<p>当前角度：<b id="angle">000°</b></p><p>转台只覆盖外观，不证明隐藏结构、同部件装配或落稳。</p>
<p><a href="{base}/mesh.glb">下载 GLB</a><br><a href="{base}/model.blend">下载 Blender 可编辑文件</a></p></aside></div></section>

<section><h2>对照图</h2><div class="grid">{gallery}</div></section>

<section><h2>实际执行的检查</h2><table>{metric_rows}</table>
<p class="notice" style="margin-top:14px">图集检查只证明贴图数值与变化；金属变光只证明高光移动；开孔射线只覆盖耳环、两个油口和杆头几何。
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
<a href="{base}/normal.png">法线</a> · <a href="{base}/材质图集-布局.jpg">图集布局</a></p>
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
    (HERE / 'model-review-26.html').write_text(html, encoding='utf8')


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
    p.add_argument('--number', type=int, default=26)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != NUMBER:
        p.error('this entry supports item 26 only')
    a.reference = (a.reference or find_reference()).resolve()
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    if a.out is None:
        a.out = (ROOT / 'work/direct-deepseek/26-draft' if a.draft
                 else ROOT / 'outputs/26油缸1-六视图重建-v1')
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
    config_path = HERE / 'cylinder26_config.json'
    worker_path = HERE / 'mesh_rebuild_26_worker.py'
    profile = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    scripts = {'driver': Path(__file__), 'worker': worker_path,
               'config': config_path}
    checks_path = HERE / 'checks_rebuild_26.py'
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
                    ('blender_worker.py', 'mesh_checks.py',
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
                        a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '六视图-新旧对照.jpg', a.out / '局部检查.jpg',
                             a.out / '分部件隔离.jpg', a.out / '材质-同光照对照.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '开孔-隔离检查.jpg',
                             a.out / '原视角诊断.jpg', a.out / '原照-新候选对照.jpg',
                             a.out / '转台-24方向.jpg', a.out / '材质图集-布局.jpg']
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
            'eye_hole_through': bool(openings['checks']['eye_hole_through']),
            'bushing_present': bool(openings['checks']['bushing_present']),
            'two_ports': bool(openings['checks']['two_ports']),
            'two_port_mouths_open': bool(openings['checks']['two_port_mouths_open']),
            'stub_hole_open': bool(openings['checks']['stub_hole_open']),
            'label_present': bool(openings['checks']['label_present']),
            'metal_highlights_move': bool(metal.get('passed')),
            'nonmetal_metal_reflection_applicable': False,
            'browser_review_executed': False,
            'source_fidelity_acceptance': False,
        },
        'pending_review': [
            '六视图参考未标定且面板互相矛盾（顶视耳环孔方向、油口对置关系与原照不一致）；'
            '工作尺度、缸径、耳环孔径、油口螺纹规格、孔深、壁厚和装配关系未实测。',
            '原照是 3/4 视角；严格原相机与包围盒对齐 IoU 只作诊断。',
            '背面、缸筒内部、活塞/密封内部、油口内部通道（建模为可见盲孔）、杆头连接方式和'
            '耳环衬套配合为推断。',
            '两个油口按原照放在同一侧且轴向位置不同；参考画成同轴对置，不能同时成立。',
            '标签文字/品牌不可辨且未复制；条形码、标志块、警告三角和文字行是近似重绘。',
            '锈迹分布、漆面磨损和焊缝形状为近似；钢衬套/孔口环真实材料未检测。',
            '落稳、接触、现场尺度、遮挡顺序与环境光融合未验证。',
            '浏览器交互检查未执行（本工作树缺少本地 three.js/node_modules）；需串行整合后人工复核。',
            '本候选不替换原模型，不进入自动放置或正式合成。',
        ],
        'limitations': profile['limitations'] + [
            '轮廓 IoU 与包围盒对齐只作诊断，不是标定测量。',
            '材质图集分布、金属变光与射线开孔检查只证明数值/几何特征，不证明真实表面或内部结构。',
        ],
        'signature': signature,
    }
    result['texture_paths'] = {k: str(Path(v).relative_to(ROOT)).replace('\\', '/')
                               for k, v in textures.items()}
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
               'eye_holes_through': openings.get('eye_holes_through'),
               'bushings_present': openings.get('bushings_present'),
               'ports': openings.get('ports'),
               'port_mouths_open': openings.get('port_mouths_open')}
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
                'eye_hole_through': result['checks']['eye_hole_through'],
                'bushing_present': result['checks']['bushing_present'],
                'ports': openings.get('ports'),
                'port_mouths_open': openings.get('port_mouths_open'),
                'stub_hole_open': result['checks']['stub_hole_open'],
                'label_present': result['checks']['label_present'],
                'metal_highlights_move': result['checks']['metal_highlights_move'],
                'browser_review_executed': False,
                'source_fidelity_acceptance': False,
                'source_strict_iou': diagnostics['strict']['iou'],
                'source_front_box_fit_iou': (diagnostics.get('front_box_fit') or {}).get('iou'),
                'turntable_best_box_fit_iou': (diagnostics.get('best') or {}).get('iou'),
                'note': '轮廓指标均为未标定诊断，不作验收通过。',
            },
            'pending_review': result['pending_review'],
            'next_number': 27,
        }
        (ROOT / 'work/direct-deepseek/26-progress.json').write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
