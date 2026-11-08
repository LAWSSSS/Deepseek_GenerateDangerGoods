"""Build the review-only six-cylinder gas-bottle group candidate (item 19).

The original photograph is a group of six separate cylinders of different
heights and paint colours.  The old single-image reconstruction fused a large
back wall and left two floating fragments (see the six-view audit), so this
entry builds each cylinder as a separate part from the measured photo
silhouette.  The user six-view sheet is an uncalibrated category reference of
one blue cylinder and is used only for the valve/regulator topology; it is not
treated as the same unit or as a measurement.

The candidate never replaces the library mesh, stays manual-only and never
enters automatic placement.  A new output directory is required for the final
run; a separate short-lived directory is used for drafts.
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
from scipy.spatial import cKDTree

from scene_v2 import digest
from mesh_checks import source_fidelity
from mesh_carve import interior_holes
from mesh_rebuild_item import _tile_detail, mask_fit, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 19
VERSION = 'cylinders19-parametric-v1'
REFERENCE_NAME = '蓝色气瓶六视图技术展示.png'
REFERENCE_SHA256 = 'db523981aaa81e97f9dc260920408806e6cf0512320cd95d6330078a8bcec8f7'
SOURCE_SIZE = (1275, 1233)
RUST_SIZE = (1276, 1233)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
TILE_NAMES = {
    0: '红色漆', 1: '绿色漆一', 2: '绿色漆二', 3: '蓝色漆一',
    4: '蓝色漆二', 5: '蓝色漆三', 6: '黄铜', 7: '镀铬钢',
    8: '压力表盘', 9: '蓝色塑料', 10: '白色标牌', 11: '深色钢/密封',
}


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
    """Full tile rectangle in image pixels (UVs leave their own 1% inset)."""
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


def draw_gauge_tile(size=512):
    canvas = Image.new('RGB', (size, size), (34, 34, 34))
    draw = ImageDraw.Draw(canvas)
    c = size / 2
    r = size * .47
    draw.ellipse((c - r, c - r, c + r, c + r), fill=(228, 226, 216),
                 outline=(52, 52, 52), width=max(2, size // 120))
    draw.ellipse((c - r * .80, c - r * .80, c + r * .80, c + r * .80),
                 outline=(120, 118, 112), width=max(1, size // 260))
    # ticks: generic pressure scale, deliberately no invented numeric readings
    for i in range(41):
        angle = math.radians(-225 + 270 * i / 40)
        outer = r * .88
        inner = r * .78 if i % 4 else r * .71
        x0, y0 = c + outer * math.cos(angle), c + outer * math.sin(angle)
        x1, y1 = c + inner * math.cos(angle), c + inner * math.sin(angle)
        draw.line((x0, y0, x1, y1), fill=(35, 35, 35), width=4 if i % 4 == 0 else 2)
    # red high-pressure arc
    box = (c - r * .91, c - r * .91, c + r * .91, c + r * .91)
    draw.arc(box, start=26, end=52, fill=(190, 40, 36), width=max(2, size // 90))
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size // 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((c - size * .09, c + r * .27), 'MPa', fill=(30, 30, 30), font=font)
    draw.ellipse((c - size * .035, c - size * .035, c + size * .035, c + size * .035),
                 fill=(198, 196, 188), outline=(70, 70, 70), width=2)
    return np.asarray(canvas)


def make_atlases(rust_path, out, profile):
    rust = Image.open(rust_path).convert('RGBA')
    if rust.size != RUST_SIZE:
        raise ValueError('19 做锈素材尺寸改变，采样坐标需要重新核对')
    size = profile['texture']['size']
    tile = profile['texture']['tile']
    tiles = profile['texture']['tiles']
    samples = profile['texture']['samples']
    rng = np.random.default_rng(19019)
    base = np.zeros((size, size, 3), np.uint8)
    mr = np.zeros((size, size, 3), np.uint8)
    normal = np.zeros((size, size, 3), np.uint8)
    normal[:, :, 2] = 255
    yy, xx = np.mgrid[0:tile, 0:tile]
    v = 1 - (yy + .5) / tile
    fine = cv2.resize(rng.random((320, 320)).astype(np.float32), (tile, tile),
                      interpolation=cv2.INTER_CUBIC)
    coarse = cv2.resize(rng.random((24, 24)).astype(np.float32), (tile, tile),
                        interpolation=cv2.INTER_CUBIC)
    peel = cv2.GaussianBlur(rng.normal(size=(tile // 8, tile // 8)).astype(np.float32),
                            (0, 0), 1.4)
    peel = cv2.resize(peel, (tile, tile), interpolation=cv2.INTER_CUBIC)
    peel /= max(1e-6, float(peel.std()))

    def put(idx, color, roughness, metallic, height=None):
        x0, y0, x1, y1 = tile_rect(idx, size, tile)
        sl = np.s_[y0:y1, x0:x1]
        base[sl] = np.clip(color, 0, 255).astype(np.uint8)
        rough_array = np.broadcast_to(np.asarray(roughness, np.float32), color.shape[:2])
        metallic_array = np.full(color.shape[:2], float(metallic) * 255)
        mr[sl] = np.stack((np.full(color.shape[:2], 255),
                           np.clip(rough_array * 255, 0, 255),
                           np.clip(metallic_array, 0, 255)), axis=-1).astype(np.uint8)
        if height is not None:
            normal[sl] = normal_from_height(height, .9)

    for cid in ('red', 'green1', 'green2', 'blue1', 'blue2', 'blue3'):
        idx = tiles[cid]
        mat = profile['material_profile']['paint'][cid]
        detail = cv2.resize(_tile_detail(rust, tuple(samples[cid]), size=tile),
                            (tile, tile), interpolation=cv2.INTER_CUBIC)
        rustp = profile['material_profile']['rust']
        rust_col = np.asarray(rustp['base_rgb'], np.float32)
        dark_col = np.asarray(rustp['dark_rgb'], np.float32)
        paint = np.asarray(mat['base_rgb'], np.float32)
        color = paint[None, None, :] * (1 + detail[:, :, None] * .0018 +
                                        (coarse[:, :, None] - .5) * .035 +
                                        (fine[:, :, None] - .5) * .018)
        wave = cv2.GaussianBlur(cv2.resize(
            rng.normal(size=(16, 16)).astype(np.float32), (tile, tile),
            interpolation=cv2.INTER_CUBIC), (0, 0), 3.0)
        wave /= max(1e-6, float(np.abs(wave).max()))
        band = np.clip((rustp['band_v'] + .025 * wave - v) / rustp['fade_v'], 0, 1)
        spot_field = cv2.GaussianBlur(cv2.resize(
            rng.random((32, 32)).astype(np.float32), (tile, tile),
            interpolation=cv2.INTER_CUBIC), (0, 0), 2.6)
        q = float(np.quantile(spot_field, .994))
        spots = np.clip((spot_field - q) / .025, 0, 1) * \
            np.clip((.42 - v) / .42, 0, 1)
        rust_cov = np.clip(band + .55 * spots, 0, 1)
        rust_rgb = rust_col[None, None, :] * (0.78 + .42 * fine[:, :, None]) + \
            (dark_col - rust_col)[None, None, :] * np.clip((.5 - fine[:, :, None]) * .5, 0, 1)
        color = color * (1 - rust_cov[:, :, None] * .88) + \
            rust_rgb * (rust_cov[:, :, None] * .88)
        rim = np.clip((.012 - v) / .012, 0, 1) * .24
        color = color * (1 - rim[:, :, None]) + dark_col[None, None, :] * rim[:, :, None]
        rough = mat['roughness'] * (1 - rust_cov) + rustp['roughness'] * rust_cov + \
            (fine - .5) * .035
        height = detail * .0018 + rust_cov * (fine - .5) * .35 + peel * .02
        put(idx, color, rough, 0.0, height)

    # brass
    brass = profile['material_profile']['brass']
    patina = cv2.GaussianBlur(cv2.resize(rng.random((28, 28)).astype(np.float32),
                                         (tile, tile), interpolation=cv2.INTER_CUBIC), (0, 0), 2.0)
    bcol = np.asarray(brass['base_rgb'], np.float32)[None, None, :] * \
        (0.86 + .30 * fine[:, :, None] + .18 * (patina[:, :, None] - .5))
    put(tiles['brass'], bcol, brass['roughness'] + patina * .12, 1.0,
        (fine - .5) * .5 + (patina - .5) * .8)
    # chrome
    chrome = profile['material_profile']['chrome']
    brush = cv2.GaussianBlur(rng.normal(size=(tile, tile)).astype(np.float32), (0, 0), .8)
    brush /= max(1e-6, float(brush.std()))
    ccol = np.asarray(chrome['base_rgb'], np.float32)[None, None, :] * \
        (0.97 + .025 * brush[:, :, None])
    put(tiles['chrome'], ccol, chrome['roughness'] + np.abs(brush) * .035, 1.0,
        brush * .05)
    # gauge
    gauge = draw_gauge_tile(tile)
    put(tiles['gauge'], gauge, profile['material_profile']['gauge']['roughness'], 0.0)
    # blue plastic
    plastic = profile['material_profile']['plastic_blue']
    pcol = np.asarray(plastic['base_rgb'], np.float32)[None, None, :] * \
        (0.95 + .06 * fine[:, :, None] + .02 * peel[:, :, None])
    put(tiles['plastic_blue'], pcol, plastic['roughness'] + (fine - .5) * .03, 0.0,
        peel * .20)
    # white inspection label with two non-legible bars
    label = Image.new('RGB', (tile, tile), (226, 224, 214))
    d = ImageDraw.Draw(label)
    d.rectangle((tile * .18, tile * .28, tile * .82, tile * .72),
                outline=(162, 160, 152), width=3)
    d.rectangle((tile * .24, tile * .40, tile * .76, tile * .46), fill=(174, 172, 164))
    d.rectangle((tile * .24, tile * .54, tile * .62, tile * .59), fill=(184, 182, 174))
    put(tiles['label_white'], np.asarray(label, np.float32),
        profile['material_profile']['label_white']['roughness'], 0.0)
    # dark steel / seal
    dark = profile['material_profile']['dark']
    dcol = np.asarray(dark['base_rgb'], np.float32)[None, None, :] * \
        (0.92 + .10 * fine[:, :, None])
    put(tiles['dark'], dcol, dark['roughness'], dark['metallic'], peel * .25)
    # unused spare tiles: neutral mid-grey, never black
    for idx in (12, 13, 14, 15):
        put(idx, np.full((tile, tile, 3), 128, np.float32), .6, 0.0)

    paths = {}
    for name, arr, mode in (('base_color', base, 'RGB'),
                            ('metallic_roughness', mr, 'RGB'),
                            ('normal', normal, 'RGB')):
        p = out / f'{name}.png'
        Image.fromarray(arr.astype(np.uint8), mode).save(p)
        paths[name] = str(p)
    # layout evidence sheet
    sheet = Image.new('RGB', (size, size), (245, 245, 243))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(LABEL_FONT, 22)
    for idx in range(16):
        col, row = idx % 4, idx // 4
        x, y = col * tile, row * tile
        sheet.paste(Image.fromarray(base[y:y + tile, x:x + tile]), (x, y))
        draw.rectangle((x, y, x + tile, y + tile), outline=(230, 230, 228), width=2)
        draw.text((x + 10, y + 8), f"{idx} {TILE_NAMES.get(idx, '备用')}",
                  fill=(20, 20, 20), font=font)
    sheet.save(out / '材质图集-布局.jpg', quality=93)
    return paths


# ---------------------------------------------------------------------------
# Source measurements and old-model audit
# ---------------------------------------------------------------------------
def source_measurements(original_path, profile):
    im = Image.open(original_path).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('原图尺寸改变，测量坐标需要重新核对')
    a = np.asarray(im)
    mask = a[:, :, 3] >= 128
    hsv = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2HSV)
    rows = {}
    for c in profile['cylinders']:
        px = c['source_px']
        x0, x1 = int(px['center_x'] - px['width'] / 2), int(px['center_x'] + px['width'] / 2)
        sub = mask[:, max(0, x0):x1 + 1]
        ys, xs = np.nonzero(sub)
        top, bot = int(ys.min()), int(ys.max())
        measured = {
            'centre_x': float((xs.min() + xs.max()) / 2 + x0),
            'width_px': int(xs.max() - xs.min() + 1),
            'top_px': top,
            'bottom_px': bot,
            'height_px': int(bot - top + 1),
        }
        rows[c['id']] = {
            'measured': measured,
            'configured': {'centre_x': px['center_x'], 'width_px': px['width'],
                           'top_px': px['top'], 'bottom_px': px['bottom']},
            'delta_px': {
                'centre_x': round(measured['centre_x'] - px['center_x'], 3),
                'width_px': round(measured['width_px'] - px['width'], 3),
                'top_px': round(measured['top_px'] - px['top'], 3),
                'bottom_px': round(measured['bottom_px'] - px['bottom'], 3),
            },
        }
    # dark oval lifting holes in the two green guards
    holes = {}
    for cid, box in (('green1', (374, 80, 474, 250)), ('green2', (530, 235, 650, 400))):
        x0, y0, x1, y1 = box
        sub = hsv[y0:y1, x0:x1]
        sm = mask[y0:y1, x0:x1]
        dark = np.logical_and(sm, np.logical_and(sub[:, :, 2] < 115,
                                                 sub[:, :, 1] < 175))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
        candidates = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 100 <= area <= 900 and h > w * 1.4 and w >= 8:
                candidates.append({'bbox': [int(x + x0), int(y + y0), int(w), int(h)],
                                   'area': int(area),
                                   'centre': [round(float(cent[i][0] + x0), 2),
                                              round(float(cent[i][1] + y0), 2)]})
        holes[cid] = sorted(candidates, key=lambda z: abs(z['bbox'][0] - (x0 + 50)))[:2]
    # rust fraction near the base and in the middle, as texture evidence only
    rust = {}
    for c in profile['cylinders']:
        px = c['source_px']
        x0, x1 = int(px['center_x'] - px['width'] * .35), int(px['center_x'] + px['width'] * .35)
        b0, b1 = px['bottom'] - 40, px['bottom'] - 5
        m0, m1 = px['top'] + (px['bottom'] - px['top']) // 2, px['top'] + (px['bottom'] - px['top']) // 2 + 35
        def frac(y0, y1):
            s = a[y0:y1, x0:x1]
            sm = s[:, :, 3] >= 128
            r = s[:, :, 0].astype(int); g = s[:, :, 1].astype(int); b = s[:, :, 2].astype(int)
            brown = (r > b + 10) & (r >= g - 10) & (sm)
            return round(float(brown.sum() / max(1, sm.sum())), 4)
        rust[c['id']] = {'base_fraction': frac(b0, b1), 'middle_fraction': frac(m0, m1)}
    return {'image': str(original_path), 'size': SOURCE_SIZE,
            'scale_m_per_px': profile['scale_m_per_px'],
            'cylinders': rows, 'guard_hole_candidates': holes, 'rust_fraction': rust,
            'note': '轮廓测量只来自原照正投影；数值未标定，不表示实物尺寸。'}


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
    points = main.vertices
    tree = cKDTree(points)
    fragments = []
    for c in comps[1:]:
        d, _ = tree.query(c.vertices)
        fragments.append({'vertices': len(c.vertices),
                          'bounds': np.round(c.bounds, 4).tolist(),
                          'min_gap_m': round(float(d.min()), 4)})
    return {
        'mesh': row['mesh'], 'mesh_sha256': row['mesh_sha256'],
        'bounds': np.round(merged.bounds, 4).tolist(),
        'components': len(comps),
        'main_vertices': len(main.vertices),
        'fragments': fragments,
        'photo_row_width_over_height': round(984 / 1196, 4),
        'old_main_width_over_height': round(float((main.bounds[1][1] - main.bounds[0][1]) /
                                                  (main.bounds[1][2] - main.bounds[0][2])), 4),
        'old_main_depth_m': round(float(main.bounds[1][0] - main.bounds[0][0]), 4),
        'note': ('旧网格把六只瓶融合成一排并带出背墙/浮块；主连通块深度远大于单瓶直径，'
                 '侧视出现原图没有的漂浮块，因此不复用旧网格。'),
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
    return make_board(out, items, 6, '19 气瓶组 · 六方向新旧对照',
                      '同一中性光、正交相机；旧模型保留原样，仅作对照。新版各瓶独立，不用平滑掩盖缺失结构。',
                      path_name='六视图-新旧对照.jpg')


def detail_board(out):
    specs = [
        ('red-cap', '红瓶阀帽/肩部'), ('green1-hole', '绿瓶一护圈椭圆孔'),
        ('green2-hole', '绿瓶二护圈椭圆孔'), ('blue1-valve', '蓝瓶一黄铜阀/手轮'),
        ('blue2-gauge', '蓝瓶二减压表/旋钮'), ('blue3-gauge', '蓝瓶三减压表'),
        ('bottom-rust', '底部锈蚀带/卷边'), ('row-gaps', '六瓶间隔'),
        ('labels', '白色检验标牌（字迹不可辨）'),
    ]
    items = [(f'局部 · {label}', '同光照近景', _load_rgba(out / 'renders' / f'detail-{name}.png'))
             for name, label in specs]
    return make_board(out, items, 3, '19 气瓶组 · 关键局部检查',
                      '局部近景用于检查护圈孔、阀/表、底部锈蚀与瓶间空隙；不能据此证明不可见背面或真实尺寸。',
                      path_name='局部检查.jpg')


def material_board(out):
    items = [
        ('新版材质正视图', 'PBR 图集 / 中性光', _load_rgba(out / 'renders' / 'new-material-front.png')),
        ('新版灰度正视图', '同光照灰模', _load_rgba(out / 'renders' / 'new-clay-front.png')),
        ('旧模型正视图', '旧顶点色网格', _load_rgba(out / 'renders' / 'old-front.png')),
    ]
    return make_board(out, items, 3, '19 气瓶组 · 同光照材质/灰模对照',
                      '漆面按真实构造作非金属电介质（金属度 0）；黄铜与镀铬阀件为金属（金属度 1），其反射由变光检查单独验证。',
                      path_name='材质-同光照对照.jpg')


def metal_board(out, metrics):
    items = [
        ('主光 A', '蓝瓶一黄铜/镀铬阀件隔离', _load_rgba(out / 'renders' / 'metal-only-light-a.png')),
        ('主光 B', '蓝瓶一黄铜/镀铬阀件隔离', _load_rgba(out / 'renders' / 'metal-only-light-b.png')),
    ]
    note = (f"非背景像素差分 {metrics['mean_abs_difference']:.2f}/255；"
            f"最亮 1% 高光质心位移 {metrics['highlight_shift_px']:.1f} px；"
            f"高阈值 {metrics.get('high_threshold_luma', 0):.0f} 位移 "
            f"{metrics.get('high_threshold_shift_px', 0):.1f} px；"
            f"金属像素 {metrics['metal_pixels']}。仅证明最亮镜面高光随光位移动，不证明材质测量正确。")
    return make_board(out, items, 2, '19 气瓶组 · 金属反射变光检查', note,
                      cell=360, path_name='金属反射-变光对照.jpg')


def opening_board(out, counts):
    items = [
        ('护圈隔离 · 正视', '椭圆孔应通透', _load_rgba(out / 'renders' / 'guards-only-front.png')),
        ('护圈隔离 · 背视', '同一通孔背面', _load_rgba(out / 'renders' / 'guards-only-back.png')),
        ('六瓶筒身 · 顶视', '应保持 6 个独立截面', _load_rgba(out / 'renders' / 'bodies-only-top.png')),
    ]
    note = (f"正视孤立孔 {counts['front_holes']}，背视孤立孔 {counts['back_holes']}；"
            f"顶视筒身独立分量 {counts['top_components']}。只说明这些方向可见，背侧隐藏结构仍待审。")
    return make_board(out, items, 3, '19 气瓶组 · 开孔与独立瓶身检查', note,
                      path_name='开口-隔离检查.jpg')


def source_board(out, metrics):
    items = [
        ('原照透明轮廓', '输入源视图', Image.open(metrics['source_view']).convert('RGBA')),
        ('新版严格原相机', f"IoU {metrics['strict']['iou']:.4f}（仅诊断）",
         _load_rgba(out / 'renders' / 'new-source.png')),
        (f"包围盒对齐最优 {metrics['best_view']}°", f"IoU {metrics['best']['iou']:.4f}（仅诊断）",
         _load_rgba(out / 'renders' / f"turn-{int(metrics['best_view']) // 15:02d}.png")),
    ]
    note = ('原相机来自旧模型库，未为本件重新标定；包围盒对齐也不是标定。IoU 与细结构召回只作诊断，'
            '不构成验收通过。')
    return make_board(out, items, 3, '19 气瓶组 · 原视角轮廓诊断', note,
                      path_name='原视角诊断.jpg')


def original_board(out):
    orig = Image.open(ROOT / 'outputs/物品高清输入-39件/原貌HD/19-气瓶.png').convert('RGBA')
    new = _load_rgba(out / 'renders' / 'new-material-front.png')
    items = [
        ('原貌照片', '六只不同颜色/高度', orig),
        ('新版正视渲染', 'PBR 材质 + 中性光', new),
        ('六视图参考（类别）', '单只蓝色气瓶，未标定', Image.open(
            ROOT / 'references/parallel-six-view' / REFERENCE_NAME).convert('RGBA')),
    ]
    return make_board(out, items, 3, '19 气瓶组 · 原照与新候选（附类别参考）',
                      '参考图只用于蓝色瓶阀/表拓扑；原照的六只瓶数量、颜色与高度差以照片为准，不把参考当实测。',
                      path_name='原照-新候选对照.jpg', cell=360)


def turntable_board(out):
    items = []
    for i in range(24):
        items.append((f'{i * 15:03d}°', '新版转台', _load_rgba(out / 'renders' / f'turn-{i:02d}.png')))
    return make_board(out, items, 6, '19 气瓶组 · 24 方向转台',
                      '转台只用于人工旋转检查；六视图参考未标定，不能把轮廓相似称为实测精度。',
                      cell=150, path_name='转台-24方向.jpg')


def atlas_board(out):
    items = [
        ('底色图集', 'sRGB / 2048 px', Image.open(out / 'base_color.png').convert('RGBA')),
        ('金属-粗糙度图集', 'G=粗糙度 / B=金属度', Image.open(out / 'metallic_roughness.png').convert('RGBA')),
        ('法线图集', '非颜色 / 低强度', Image.open(out / 'normal.png').convert('RGBA')),
    ]
    return make_board(out, items, 3, '19 气瓶组 · PBR 贴图集',
                      '漆面金属度为 0（非金属漆层）；黄铜/镀铬金属度为 1。底部锈蚀带来自做锈素材的有限采样与程序化分布。',
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
    result = {'tiles': {}, 'notes': []}
    for name, idx in tiles.items():
        x0, y0, x1, y1 = tile_rect(idx, size, tile)
        center = (slice((y0 + y1) // 2 - 16, (y0 + y1) // 2 + 16),
                  slice((x0 + x1) // 2 - 16, (x0 + x1) // 2 + 16))
        b = base[center].reshape(-1, 3).astype(float)
        m = mr[center].reshape(-1, 3).astype(float)
        n = normal[center].reshape(-1, 3).astype(float)
        luma = float(np.median(b @ np.array([.2126, .7152, .0722])))
        result['tiles'][name] = {
            'index': idx,
            'median_luma': round(luma, 2),
            'base_std': round(float(b.std()), 2),
            'roughness': round(float(np.median(m[:, 1]) / 255), 3),
            'metallic': round(float(np.median(m[:, 2]) / 255), 3),
            'normal_std': round(float(n.std()), 2),
        }
    for name in ('red', 'green1', 'green2', 'blue1', 'blue2', 'blue3'):
        t = result['tiles'][name]
        x0, y0, x1, y1 = tile_rect(tiles[name], size, tile)
        band_rows = max(4, int((y1 - y0) * .035))
        band = base[y1 - band_rows:y1, x0:x1].reshape(-1, 3)
        upper = base[y0 + int((y1 - y0) * .35):y0 + int((y1 - y0) * .65),
                     x0:x1].reshape(-1, 3)
        band_med = np.median(band, axis=0)
        upper_med = np.median(upper, axis=0)
        t['rust_band_rgb'] = [int(v) for v in band_med]
        t['upper_rgb'] = [int(v) for v in upper_med]
        t['rust_band_luma'] = round(float(band_med @ np.array([.2126, .7152, .0722])), 2)
        t['rust_band_distance'] = round(float(np.linalg.norm(band_med - upper_med)), 2)
        band_hsv = cv2.cvtColor(band_med.astype(np.uint8).reshape(1, 1, 3),
                                cv2.COLOR_RGB2HSV)[0, 0]
        t['rust_band_hue'] = int(band_hsv[0])
        t['rust_band_saturation'] = int(band_hsv[1])
    paint = [result['tiles'][n] for n in ('red', 'green1', 'green2', 'blue1', 'blue2', 'blue3')]
    metal = [result['tiles'][n] for n in ('brass', 'chrome')]
    result['checks'] = {
        'paint_metallic_zero': all(t['metallic'] <= .02 for t in paint),
        'paint_roughness_range': all(.20 <= t['roughness'] <= .55 for t in paint),
        'paint_not_black': all(t['median_luma'] >= 50 for t in paint),
        'paint_has_variation': all(t['base_std'] >= 1.2 for t in paint),
        'rust_band_present': all(t['rust_band_distance'] >= 6 for t in paint),
        'metal_metallic_one': all(t['metallic'] >= .98 for t in metal),
        'metal_roughness_range': all(.10 <= t['roughness'] <= .55 for t in metal),
        'normal_not_flat': all(t['normal_std'] >= 1.0 for t in paint + metal),
        'gauge_has_contrast': result['tiles']['gauge']['base_std'] >= 10,
    }
    result['passed'] = all(result['checks'].values())
    result['notes'].append('图集统计只证明贴图数值与变化量，不证明纹理在实物上的真实分布。')
    return result


def source_diagnostics(out, profile, alpha_path, source_view_path):
    source = np.asarray(Image.open(alpha_path).convert('L'))
    rendered = np.asarray(_load_rgba(out / 'renders' / 'new-source.png').getchannel('A'))
    strict = source_fidelity(source, rendered)
    strict['acceptance_gate'] = False
    strict['camera'] = '旧模型库标定相机，未为本件重新标定'
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


def opening_checks(out):
    front = np.asarray(_load_rgba(out / 'renders' / 'guards-only-front.png').getchannel('A'))
    back = np.asarray(_load_rgba(out / 'renders' / 'guards-only-back.png').getchannel('A'))
    holes_f, count_f = interior_holes(front)
    holes_b, count_b = interior_holes(back)
    top = np.asarray(_load_rgba(out / 'renders' / 'bodies-only-top.png').getchannel('A'))
    n, labels, stats, cent = cv2.connectedComponentsWithStats((top >= 128).astype(np.uint8), 8)
    comps = [{'area': int(s[4]), 'centre': [round(float(c[0]), 1), round(float(c[1]), 1)]}
             for s, c in zip(stats[1:], cent[1:]) if s[4] >= 100]
    return {
        'front_holes': int(count_f), 'back_holes': int(count_b),
        'top_components': len(comps), 'top_component_details': comps,
        'checks': {
            'two_guard_holes_front': count_f == 2,
            'two_guard_holes_back': count_b == 2,
            'six_separate_bodies_top': len(comps) == 6,
        },
        'note': ('护圈为实体上真实钻通孔；正视/背视各 2 个孤立孔。顶视只隐藏阀/表看筒身，'
                 '不能证明所有隐藏零件关系。'),
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
    # With the focused blue-1 valve the top 1% is one coherent set of specular
    # lobes, so its centroid is a stable measure.  A high absolute threshold is
    # also recorded as a secondary diagnostic.
    qa = np.quantile(la[mask], .99)
    qb = np.quantile(lb[mask], .99)
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
    result['note'] = ('只说明金属最亮镜面高光随光位移动；不证明反射率、材质测量或环境光匹配。')
    return result


def body_gap(worker, out=None):
    gaps = worker.get('body_gaps') or []
    return {'pairs': gaps,
            'minimum_m': min((g['min_gap_m'] for g in gaps), default=None),
            'passed': bool(gaps and min(g['min_gap_m'] for g in gaps) > .003),
            'note': '仅测相邻筒身网格最近顶点距离，不证明落稳、接触或装配关系。'}


# ---------------------------------------------------------------------------
# Report and hashes
# ---------------------------------------------------------------------------
def write_report(out, result):
    lines = [
        '# 19 气瓶组 · 六视图重建 v1（manual_only，待审）',
        '',
        '## 结论',
        '',
        '本候选把六只气瓶分别建成独立部件，保留原照中的高度/直径差、红/绿/蓝漆色、'
        '两个绿色护圈椭圆通孔、蓝瓶一黄铜阀与蓝色手轮、蓝瓶二/三减压表与蓝色旋钮。'
        '它不替换原模型，不进入自动放置或正式合成；所有诊断均不是验收通过。',
        '',
        '## 为什么重建而不是保守修复',
        '',
        f"- 旧网格有 {result['old_audit']['components']} 个连通分量；主块深度 "
        f"{result['old_audit']['old_main_depth_m']:.3f} m，远大于单瓶直径，背视/侧视出现原图没有的浮块。",
        f"- 旧网格桶排宽/高比 {result['old_audit']['old_main_width_over_height']}，"
        f"原照测量 {result['old_audit']['photo_row_width_over_height']}；旧体量被压窄并带出背墙。",
        f"- 原视角审计记录：{result['audit']['warnings'][0] if result['audit'].get('warnings') else '无'}；"
        '旧源图门槛通过不代表结构正确。',
        '',
        '## 输入与参考',
        '',
        '- 原图/做锈图/旧网格/源视图透明度哈希见 `hashes.json`。',
        '- 六视图：`蓝色气瓶六视图技术展示.png`，角色 **category_reference**（单只带减压表蓝瓶，未标定）；'
        '只用于蓝色瓶阀/表拓扑，不当作同一实物、数量或尺寸依据。',
        '',
        '## 结构',
        '',
        '- 六只独立筒身 + 底部卷边；两只绿色护圈为实体上钻通孔（正视/背视各检出 2 个孔）。',
        '- 红瓶为封闭阀帽；绿瓶为圆顶护圈 + 白色检验标牌（字迹不可辨，未复制）；'
        '蓝瓶一为黄铜阀体、六角锁母、侧出气口、蓝色花瓣手轮；蓝瓶二/三为镀铬减压器、'
        '白色压力表盘、3D 指针、顶部与侧向蓝色花瓣旋钮、出气嘴。',
        f"- 网格：{result['mesh']['parts']} 个部件，{result['mesh']['polygons']} 面；"
        f"闭合部件 {result['mesh']['closed_after_welding_seams']}/{result['mesh']['parts']}；"
        f"同部件自交 {len(result['worker']['self_intersecting_components'])} 个。",
        '',
        '## 材质',
        '',
        '- 漆面/锈蚀：非金属电介质（金属度 0），粗糙度约 0.30–0.43；底部锈蚀带来自做锈素材有限采样，'
        '分布为程序化近似，不是逐点实测。',
        '- 黄铜/镀铬/表圈：金属度 1，粗糙度约 0.18–0.34；`金属反射-变光对照.jpg` 显示高光随光位移动。',
        '- 漆面不上金属反射不是缺陷：真实构造是金属筒体外的非金属漆层。',
        f"- 图集检查通过：{result['textures']['passed']}；详见 `result.json` 的 tile 数值。",
        '',
        '## 检查结果（实际执行）',
        '',
        f"- Blender {result['versions']['blender']}，Cycles CPU，六视图 {result['render']['six_view_samples']} 采样；"
        f"GLB/Blend/贴图/对照图均由本工作树内 Blender/Python 生成。",
        f"- 严格原相机轮廓：IoU {result['diagnostics']['strict']['iou']}，"
        f"填孔 {result['diagnostics']['strict']['filled_holes']}，细结构召回 "
        f"{result['diagnostics']['strict']['narrow_recall']}（仅诊断，非验收）。",
        f"- 包围盒对齐最优转台 {result['diagnostics']['best'].get('view', 'n/a')}："
        f"IoU {result['diagnostics']['best'].get('iou', 'n/a')}（未标定，仅诊断）。",
        (f"- 开孔/独立瓶身：正 {result['openings'].get('front_holes')} 孔，"
         f"背 {result['openings'].get('back_holes')} 孔，"
         f"顶视筒身 {result['openings'].get('top_components')} 个分量。"
         if result['openings'] else '- 开孔/独立瓶身：草稿模式未执行。'),
        (f"- 瓶间最小网格间隙：{result['body_gap'].get('minimum_m')} m（只测顶点距离）。"
         if result['body_gap'] else '- 瓶间最小网格间隙：草稿模式未执行。'),
        (f"- 金属变光：差分 {result['metal'].get('mean_abs_difference')}/255，"
         f"高光质心位移 {result['metal'].get('highlight_shift_px')} px。"
         if result['metal'] else '- 金属变光：草稿模式未执行。'),
        f"- 浏览器交互：未执行（本工作树缺少本地 three.js/node_modules；未安装依赖，也未声称通过）。",
        '',
        '## 待人工复核',
        '',
        '- 六视图参考未标定且是单只蓝瓶；整体尺度、各瓶高度/直径、红色与绿色瓶的真实阀型、'
        '护圈开孔数量/深度、减压表读数与型号均未实测。',
        '- 背面、底面中心、护圈内部、阀/表内部、瓶间距与排序、螺纹/接口、表面锈蚀真实分布为推断。',
        '- 落稳、接触、现场尺度、遮挡顺序、场景光照融合均未验证；不能进入正式合成。',
        '- 网页交互检查未执行；需在串行整合时接入共享查看器并由人工复核。',
        '',
        '## 文件',
        '',
        '- `mesh.glb`：Z 向上、库坐标约定，纹理内嵌。',
        '- `model.blend`：可编辑几何、PBR 节点、图集已打包。',
        '- `base_color.png` / `metallic_roughness.png` / `normal.png`：2048 PBR 图集。',
        '- `六视图-新旧对照.jpg`、`局部检查.jpg`、`材质-同光照对照.jpg`、'
        '`金属反射-变光对照.jpg`、`开口-隔离检查.jpg`、`原视角诊断.jpg`、'
        '`原照-新候选对照.jpg`、`转台-24方向.jpg`、`材质图集-布局.jpg`。',
        '- `result.json`、`hashes.json`、`source-measurements.json`、`old-model-audit.json`、`worker-result.json`。',
        '',
        '## 复现',
        '',
        '```powershell',
        '& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/mesh_rebuild_item_19.py '
        '--out "outputs/19气瓶组-六视图重建-v1"',
        '& work/triposr-env/Scripts/python.exe -X utf8 -B outputs/ps-model-controller/checks_rebuild_19.py',
        '```',
        '',
        '输出目录必须为新目录；相同签名的完整结果可复用。',
    ]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


def write_review_page(out, result):
    """Standalone offline review page using the rendered evidence only."""
    rel = out.name
    base = f'../{rel}'
    checks = result['checks']
    d = result['diagnostics']
    tiles = [
        ('六方向新旧对照', f'{base}/六视图-新旧对照.jpg'),
        ('关键局部', f'{base}/局部检查.jpg'),
        ('同光照材质/灰模', f'{base}/材质-同光照对照.jpg'),
        ('金属反射变光', f'{base}/金属反射-变光对照.jpg'),
        ('开孔/独立瓶身', f'{base}/开口-隔离检查.jpg'),
        ('原视角轮廓诊断', f'{base}/原视角诊断.jpg'),
        ('原照/新候选/类别参考', f'{base}/原照-新候选对照.jpg'),
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
        ('护圈通孔（正视/背视）',
         f"{result['openings'].get('front_holes')} / {result['openings'].get('back_holes')}"),
        ('顶视独立筒身', result['openings'].get('top_components')),
        ('网格部件/闭合', f"{result['mesh']['parts']} / {result['mesh']['closed_after_welding_seams']}"),
        ('同部件自交', len(result['worker']['self_intersecting_components'])),
        ('相邻筒身最小顶点间隙 (m)', result['body_gap'].get('minimum_m')),
        ('贴图数值检查', result['textures'].get('passed')),
        ('金属高光随光位移动', result['metal'].get('passed')),
        ('金属高光质心位移 (px)', result['metal'].get('highlight_shift_px')),
        ('浏览器交互检查', '未执行'),
    ]
    metric_rows = ''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in metrics)
    turn_frames = ',\n'.join(f"'{base}/renders/turn-{i:02d}.png'" for i in range(24))
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>19 气瓶组 · 三维模型检查（待审）</title>
<style>
:root{{--ink:#1d2320;--muted:#5f6a64;--line:#d5d9d5;--panel:#fbfcfa;--accent:#356b55}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.65 "Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:#eef1ee}}
header{{background:#203029;color:#f4f7f2;padding:26px 30px}}header h1{{margin:0 0 6px;font-size:25px}}
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
<header><h1>19 气瓶组 · 六视图重建（{VERSION}） <span class="tag">manual_only · 待审</span></h1>
<p>六只独立气瓶（红 1、绿 2、蓝 3）· 原照轮廓测量为主 · 六视图为未标定的单只蓝瓶类别参考</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b> 本候选不替换原模型，不进入自动放置或正式合成。
所有 IoU 均为未标定诊断，不是验收通过。背面、内部、真实尺寸、落稳和现场光照仍待人工复核。
浏览器交互检查未执行；本页用 Blender 渲染图为离线证据。</div></section>

<section><h2>旋转检查（24 方向实测渲染）</h2><div class="viewer">
<div class="turnbox" id="turnbox"><img id="turn" src="{base}/renders/turn-00.png" alt="转台"></div>
<aside><p>拖动图片旋转；也可自动播放。</p><p><button id="play">自动旋转</button> <button id="reset">回到 000°</button></p>
<p>当前角度：<b id="angle">000°</b></p><p>转台只覆盖外观，不证明隐藏结构、同部件装配或落稳。</p>
<p><a href="{base}/mesh.glb">下载 GLB</a><br><a href="{base}/model.blend">下载 Blender 可编辑文件</a></p></aside></div></section>

<section><h2>对照图</h2><div class="grid">{gallery}</div></section>

<section><h2>实际执行的检查</h2><table>{metric_rows}</table>
<p class="notice" style="margin-top:14px">图集检查只证明贴图数值与变化；金属变光只证明高光移动；开孔检查只覆盖当前可见方向。
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
<p>候选为 <code>cylinders19-parametric-v1</code>，参考角色 <code>category_reference</code>；
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
    (HERE / 'model-review-19.html').write_text(html, encoding='utf8')


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
    p.add_argument('--number', type=int, default=19)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != NUMBER:
        p.error('this entry supports item 19 only')
    a.reference = (a.reference or find_reference()).resolve()
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    if a.out is None:
        a.out = (ROOT / 'work/direct-deepseek/19-draft' if a.draft
                 else ROOT / 'outputs/19气瓶组-六视图重建-v1')
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
    config_path = HERE / 'cylinders19_config.json'
    worker_path = HERE / 'mesh_rebuild_19_worker.py'
    profile = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    scripts = {'driver': Path(__file__), 'worker': worker_path,
               'config': config_path}
    checks_path = HERE / 'checks_rebuild_19.py'
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
                        a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '六视图-新旧对照.jpg', a.out / '局部检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '金属反射-变光对照.jpg',
                             a.out / '开口-隔离检查.jpg', a.out / '原视角诊断.jpg',
                             a.out / '原照-新候选对照.jpg', a.out / '转台-24方向.jpg',
                             a.out / '材质图集-布局.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
            if (old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh)
                    and all(path.is_file() for path in required)):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    measurements = source_measurements(original_path, profile)
    (a.out / 'source-measurements.json').write_text(
        json.dumps(measurements, ensure_ascii=False, indent=2), encoding='utf8')
    old_audit = old_model_audit({**row, 'mesh': str(mesh_path)})
    audit = audit_row(NUMBER)
    (a.out / 'old-model-audit.json').write_text(
        json.dumps({'old': old_audit, 'audit': audit}, ensure_ascii=False, indent=2),
        encoding='utf8')
    textures = make_atlases(rust_path, a.out, profile)
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
                                 stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if process.returncode:
        raise RuntimeError(f'Blender failed; see {a.out / "blender.log"}')

    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    mesh_info = inspect_mesh(a.out / 'mesh.glb')
    diagnostics = source_diagnostics(a.out, profile, alpha_path, source_view_path)
    openings = opening_checks(a.out)
    metal = metal_reflection_check(a.out) if not a.draft else {}
    textures_checks = texture_checks(a.out, profile)
    gaps = body_gap(worker)

    md5_new = digest(a.out / 'mesh.glb')
    result = {
        'number': NUMBER,
        'version': VERSION,
        'status': 'review',
        'draft': bool(a.draft),
        'manual_only': True,
        'asset_id': row['asset_id'],
        'original_mesh_sha256': row['mesh_sha256'],
        'mesh_sha256': md5_new,
        'blend_sha256': digest(a.out / 'model.blend'),
        'shape_basis': 'source_photo_silhouette_measurements+uncalibrated_category_reference',
        'reference': {'path': str(a.reference.relative_to(ROOT)).replace('\\', '/'),
                      'sha256': digest(a.reference), 'role': 'category_reference',
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
        'body_gap': gaps,
        'versions': {'blender': version, 'python': sys.version.split()[0],
                     'numpy': np.__version__, 'opencv': cv2.__version__,
                     'pillow': PIL.__version__, 'trimesh': trimesh.__version__},
        'render': profile['render'],
        'checks': {
            'mesh_closed': not mesh_info['open_parts'],
            'no_nonfinite': not mesh_info['nonfinite_parts'],
            'no_self_intersections': not worker['self_intersecting_components'],
            'texture_values': textures_checks['passed'],
            'guard_holes_detected': bool(openings.get('checks', {}).get('two_guard_holes_front')
                                         and openings.get('checks', {}).get('two_guard_holes_back')),
            'six_separate_bodies_top': bool(openings.get('checks', {}).get('six_separate_bodies_top')),
            'body_vertex_gap': bool(gaps.get('passed')),
            'metal_highlights_move': bool(metal.get('passed')),
            'browser_review_executed': False,
            'source_fidelity_acceptance': False,
        },
        'pending_review': [
            '六视图参考未标定且只有一只蓝瓶；整体尺度与各瓶高度/直径未实测。',
            '红瓶阀帽、两只绿瓶真实阀型、护圈开孔数量/深度、蓝瓶减压表型号与读数均未实测。',
            '背面、底面中心、护圈/阀/表内部、瓶间排序与真实间距、螺纹接口、锈蚀分布为推断。',
            '落稳、接触、现场尺度、遮挡顺序与环境光融合未验证。',
            '浏览器交互检查未执行（本工作树缺少本地 three.js/node_modules）；需串行整合后人工复核。',
            '旧源图门槛通过不等于结构正确；本候选不替换原模型，不进入自动放置或正式合成。',
        ],
        'limitations': [
            '轮廓 IoU 与包围盒对齐只作诊断，不是标定测量。',
            '材质图集分布与金属变光检查只证明数值/高光变化，不证明真实表面。',
            '护圈通孔只按可见椭圆近似，孔深/内壁与背面开口数量为推断。',
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
               'guard_holes': [openings.get('front_holes'), openings.get('back_holes')],
               'top_bodies': openings.get('top_components')}
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
                'guard_holes_detected': result['checks']['guard_holes_detected'],
                'six_separate_bodies_top': result['checks']['six_separate_bodies_top'],
                'body_vertex_gap': result['checks']['body_vertex_gap'],
                'metal_highlights_move': result['checks']['metal_highlights_move'],
                'browser_review_executed': False,
                'source_fidelity_acceptance': False,
                'source_strict_iou': diagnostics['strict']['iou'],
                'source_front_box_fit_iou': (diagnostics.get('front_box_fit') or {}).get('iou'),
                'turntable_best_box_fit_iou': (diagnostics.get('best') or {}).get('iou'),
                'note': '轮廓指标均为未标定诊断，不作验收通过。',
            },
            'pending_review': result['pending_review'],
            'next_number': 20,
        }
        (ROOT / 'work/direct-deepseek/19-progress.json').write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
