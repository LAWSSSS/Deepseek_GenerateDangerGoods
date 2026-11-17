"""Build the review-only QL5 screw-jack candidate (item 11).

Shape follows the original clean/rust photograph for proportions and visible
features.  The user-supplied six-view sheet is an uncalibrated visual-match
reference of the same green QL5-style mechanical screw jack.  The sheet is
used for hidden topology and as a diagnostic comparison, never as measured
geometry, and its panels are not a consistent turntable (the operating lever
is drawn as a round pipe in the side panels but is a hollow rectangular bar in
the photo).  The candidate never replaces the library mesh and never enters
automatic placement.  A new output directory is required.
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
from mesh_rebuild_item import _tile_detail, mask_fit, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE_SHA256 = '41b108928de2c486c51c21fc3ac8def1d99137a46b96312fa5a5d512ef655702'
SOURCE_SIZE = (1254, 1254)
TAU = math.tau
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
LABEL_FONT_ARIAL = 'C:/Windows/Fonts/arialbd.ttf'


def find_reference():
    folder = Path.home() / 'Desktop' / '六视图'
    if folder.is_dir():
        for path in sorted(folder.glob('*.png')):
            if digest(path) == REFERENCE_SHA256:
                return path
    raise FileNotFoundError('registered six-view reference not found')


# ---------------------------------------------------------------------------
# PBR textures
# ---------------------------------------------------------------------------
def make_green_atlas(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('11 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((840, 600, 950, 740), (840, 720, 950, 860),
                (840, 900, 950, 1040), (840, 540, 950, 680))]
    rng = np.random.default_rng(21111)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [(profile['green']['clean'], 3.2),
             (profile['green']['weathered'], 3.8),
             (profile['green']['worn'], 2.6),
             (profile['green']['rusted'], 2.2)]
    rust_rgb = np.array((108, 74, 50), np.float32)
    dust_rgb = np.array((77, 74, 68), np.float32)
    scratch_rgb = np.array((196, 201, 197), np.float32)
    for i, (mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((110, 110), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((820, 820), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Cast-enamel orange-peel plus a light brush-like paint texture.
        peel = cv2.GaussianBlur(rng.normal(size=(size // 4, size // 4)).astype(np.float32),
                                (0, 0), 1.2)
        peel = cv2.resize(peel, (size, size), interpolation=cv2.INTER_CUBIC)
        peel *= brush_strength / max(1e-6, float(peel.std()))
        field = .50 * (coarse - .5) + .30 * (fine - .5) + .20 * (speck - .5)
        amount = mat['rust_amount']
        # Sparse rust blooms, edge wear and a little dust.
        patch = np.clip((coarse - (1 - amount * 1.8)) * 8, 0, 1)
        patch = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), 1.2)
        rust = patch * np.clip((fine - .06) * 1.25, 0, 1)
        rust = np.clip(rust + np.clip((speck - .997) * 90, 0, 1) * amount * 1.4, 0, 1)
        dust = np.clip((speck - .9982) * 45, 0, 1) * .040
        grime = np.clip((coarse - .92) * 10, 0, 1) * min(1.0, amount * 3.0)
        color = np.asarray(mat['base_rgb'], np.float32)
        rgb = color[None, None, :] * (1 + field[:, :, None] * .120)
        rgb += detail[:, :, None] * .075 + peel[:, :, None] * .50
        rgb += ((speck - .5) * .22)[:, :, None]
        rgb = rgb * (1 - grime[:, :, None] * .30) + \
            np.array((61, 64, 49), np.float32)[None, None, :] * (grime[:, :, None] * .30)
        scratches = np.clip((peel - 2.7) * .040, 0, 1) * (1 - rust) * (i == 2)
        rgb = rgb * (1 - scratches[:, :, None] * .26) + \
            scratch_rgb[None, None, :] * (scratches[:, :, None] * .26)
        rust_rgb_local = rust_rgb[None, None, :] * (.80 + .36 * fine[:, :, None])
        rgb = rgb * (1 - rust[:, :, None]) + rust_rgb_local * rust[:, :, None]
        rgb = rgb * (1 - dust[:, :, None] * .45) + \
            dust_rgb[None, None, :] * (dust[:, :, None] * .45)
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        roughness = (mat['roughness'] + rust * .24 + dust * .10
                     + field * .030 + peel * .005)
        metallic = mat['metallic'] * (1 - rust * .80)
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip(roughness * 255, 0, 255),
                           np.clip(metallic * 255, 0, 255)),
                          axis=2).astype(np.uint8)
        bump_source = peel + detail * .28
        bump = bump_source - cv2.GaussianBlur(bump_source, (0, 0), 7.0)
        bump = cv2.GaussianBlur(bump, (0, 0), .9) / 255
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 10
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 10
        nx = np.clip(-dx * .9, -.12, .12)
        ny = np.clip(dy * .9, -.12, .12)
        nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                             0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('green_base', base), ('green_mr', mr), ('green_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_chrome(out, profile):
    size = 1024
    rng = np.random.default_rng(21112)
    coarse = cv2.resize(rng.random((80, 80), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((900, 900), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    brush = cv2.resize(rng.normal(size=(180, size)).astype(np.float32),
                       (size, size), interpolation=cv2.INTER_CUBIC)
    brush = cv2.GaussianBlur(brush, (0, 0), 1.0)
    brush *= 4.0 / max(1e-6, float(brush.std()))
    base_rgb = np.asarray(profile['chrome']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + brush[:, :, None] * .016
                                          + (coarse[:, :, None] - .5) * .016)
    # Sparse dull/rust spots observed on the used screw, kept subtle.
    spots = np.clip((coarse - .975) * 35, 0, 1)
    spots = cv2.GaussianBlur(spots, (0, 0), 2.5)
    rgb = rgb * (1 - spots[:, :, None] * .38) + \
        np.array((118, 94, 74), np.float32)[None, None, :] * (spots[:, :, None] * .38)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['chrome']['roughness'] + brush * .006
                    + (coarse - .5) * .018 + spots * .16, .08, .30)
    metallic = np.full((size, size), profile['chrome']['metallic'], np.float32)
    metallic *= (1 - spots * .40)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur(brush * .45, (0, 0), .7)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 11
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 11
    nx = np.clip(-dx * .45, -.05, .05)
    ny = np.clip(dy * .12, -.025, .025)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('chrome_base', rgb), ('chrome_mr', mr),
                      ('chrome_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_steel(out, profile):
    size = 1024
    rng = np.random.default_rng(21113)
    coarse = cv2.resize(rng.random((64, 64), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((size, size), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_NEAREST)
    base_rgb = np.asarray(profile['steel']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (coarse[:, :, None] - .5) * .10
                                          + (fine[:, :, None] - .5) * .04)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['steel']['roughness'] + (coarse - .5) * .12
                    + (fine - .5) * .04, .18, .55)
    metallic = np.full((size, size), profile['steel']['metallic'], np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur((fine - .5) * 1.2 + (coarse - .5) * .8, (0, 0), .6)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 12
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 12
    nx = np.clip(-dx * .45, -.04, .04)
    ny = np.clip(dy * .45, -.04, .04)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('steel_base', rgb), ('steel_mr', mr),
                      ('steel_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_label(out):
    width, height = 768, 1280
    rng = np.random.default_rng(21114)
    image = Image.new('RGB', (width, height), (228, 224, 211))
    draw = ImageDraw.Draw(image)
    # Aged paper mottling and edge grime.
    for _ in range(900):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        shade = int(rng.integers(195, 235))
        r = int(rng.integers(1, 5))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(shade, shade - 3, shade - 12))
    draw.rectangle((8, 8, width - 9, height - 9), outline=(54, 48, 38), width=5)
    draw.rectangle((22, 22, width - 23, height - 23), outline=(120, 112, 96), width=2)
    red = (198, 36, 32)
    dark = (26, 24, 21)
    # Stylised red diamond mark with a generic white tool silhouette.
    cx, cy = 150, 150
    draw.polygon(((cx, cy - 86), (cx + 150, cy), (cx, cy + 86), (cx - 150, cy)),
                 outline=red, width=7)
    draw.polygon(((cx, cy - 58), (cx + 104, cy), (cx, cy + 58), (cx - 104, cy)),
                 outline=red, width=4)
    draw.rectangle((cx - 58, cy - 7, cx + 58, cy + 7), fill=red)
    draw.rectangle((cx - 9, cy - 45, cx + 9, cy + 45), fill=red)
    draw.text((350, 96), '液压工具', font=ImageFont.truetype(LABEL_FONT_BOLD, 44),
              fill=dark)
    draw.line((32, 270, width - 32, 270), fill=(70, 64, 54), width=4)
    draw.text((40, 300), 'QL 5 螺旋千斤顶', font=ImageFont.truetype(LABEL_FONT_BOLD, 64),
              fill=dark)
    font = ImageFont.truetype(LABEL_FONT, 40)
    font_bold = ImageFont.truetype(LABEL_FONT_BOLD, 44)
    draw.text((42, 420), '额定载荷：5 t', font=font_bold, fill=dark)
    draw.text((42, 500), '最低高度：220 mm', font=font, fill=dark)
    draw.text((42, 570), '起升高度：150 mm', font=font, fill=dark)
    draw.text((42, 690), '出厂日期：20    年    月', font=font, fill=dark)
    draw.text((42, 820), '使用前请检查螺旋与承载面', font=font, fill=dark)
    draw.line((32, 930, width - 32, 930), fill=(98, 90, 76), width=3)
    draw.text((42, 960), '三龙起重机械有限公司', font=ImageFont.truetype(LABEL_FONT, 34),
              fill=(48, 44, 38))
    draw.text((42, 1020), '（标签为按可辨文字重绘，非原印刷件）',
              font=ImageFont.truetype(LABEL_FONT, 25), fill=(85, 78, 68))
    # A small worn area and a crease.
    draw.line((560, 300, 590, 880), fill=(190, 185, 172), width=4)
    base = np.asarray(image)
    mr = np.zeros((height, width, 3), np.uint8)
    mr[..., 0] = 255
    mr[..., 1] = int(.62 * 255)
    mr[..., 2] = 0
    normal = np.zeros((height, width, 3), np.uint8)
    normal[..., 0] = 128
    normal[..., 1] = 128
    normal[..., 2] = 255
    paths = {}
    for name, arr in (('label_base', base), ('label_mr', mr), ('label_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


# ---------------------------------------------------------------------------
# Texture checks
# ---------------------------------------------------------------------------
def green_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    tiles = {'green_clean': np.s_[0:1024, 0:1024],
             'green_weathered': np.s_[0:1024, 1024:2048],
             'green_worn': np.s_[1024:2048, 0:1024],
             'green_rusted': np.s_[1024:2048, 1024:2048]}
    values = {}
    for name, sl in tiles.items():
        rgb = base[sl]
        means = rgb.mean(axis=(0, 1))
        rust = ((rgb[..., 0] > rgb[..., 1] * .82) & (rgb[..., 0] > 58)).mean()
        values[name] = {
            'base_rgb_mean': [round(float(v), 1) for v in means],
            'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
            'metallic_mean': round(float(mr[sl][..., 2].mean()), 4),
            'chroma': round(float(means.max() - means.min()), 2),
            'green_dominance': round(float(means[1] - (means[0] + means[2]) / 2), 2),
            'rust_fraction': round(float(rust), 4)}
    c, w, r = values['green_clean'], values['green_weathered'], values['green_rusted']
    values['passed'] = bool(
        c['metallic_mean'] < .15 and .28 < c['roughness_mean'] < .52 and
        c['chroma'] > 55 and c['green_dominance'] > 35 and c['rust_fraction'] < .02 and
        w['metallic_mean'] < .15 and w['chroma'] > 45 and w['green_dominance'] > 25 and
        r['roughness_mean'] > .55 and r['metallic_mean'] < .15 and
        r['rust_fraction'] > .10)
    values['note'] = ('Tile means; rust is detected by brown colour. The jack is '
                      'painted cast iron/enamel, so metallic stays low; highlights '
                      'come from live lighting, not from baked colour.')
    return values


def chrome_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    means = base.mean(axis=(0, 1))
    values = {'base_rgb_mean': [round(float(v), 1) for v in means],
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'chroma': round(float(means.max() - means.min()), 2)}
    values['passed'] = bool(means.min() > 148 and values['chroma'] < 20 and
                            values['metallic_mean'] > .88 and
                            values['roughness_mean'] < .32)
    values['note'] = ('Worn chrome lifting screw with fine vertical scratches and '
                      'sparse dull spots; the texture does not encode a light direction.')
    return values


def steel_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    means = base.mean(axis=(0, 1))
    values = {'base_rgb_mean': [round(float(v), 1) for v in means],
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'chroma': round(float(means.max() - means.min()), 2)}
    values['passed'] = bool(values['metallic_mean'] > .80 and
                            values['roughness_mean'] < .50 and
                            values['chroma'] < 18)
    values['note'] = 'Steel bolt heads, pins and cap top; no measured fastener data.'
    return values


def label_texture_checks(base_path):
    rgb = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    bright = ((rgb[..., 0] > 170) & (rgb[..., 1] > 165) & (rgb[..., 2] > 150)).mean()
    red = ((rgb[..., 0] > 150) & (rgb[..., 1] < 100) & (rgb[..., 2] < 100)).mean()
    dark = (rgb.max(axis=2) < 90).mean()
    values = {'bright_fraction': round(float(bright), 4),
              'red_fraction': round(float(red), 4),
              'dark_fraction': round(float(dark), 4),
              'size': [int(rgb.shape[1]), int(rgb.shape[0])]}
    values['passed'] = bool(bright > .35 and red > .0015 and dark > .01)
    values['note'] = ('Label artwork is recreated from legible photo text; the '
                      'original graphic, small print and certification marks are '
                      'not reproduced.')
    return values


def plastic_check(profile):
    rgb = np.asarray(profile['bright_green']['base_rgb'], np.float32)
    values = {'base_rgb': [int(v) for v in rgb],
              'roughness': profile['bright_green']['roughness'],
              'metallic': profile['bright_green']['metallic'],
              'green_dominance': round(float(rgb[1] - (rgb[0] + rgb[2]) / 2), 1)}
    values['passed'] = bool(values['green_dominance'] > 70 and
                            values['metallic'] < .05 and
                            .20 < values['roughness'] < .70)
    values['note'] = 'Bright green plastic release/lock knob; colour and finish are inferred.'
    return values


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 10, y0 + 8, x0 + w // 3 - 10, y0 + h // 2 - 8))


def reference_mask_11(panel):
    """Approximate foreground mask for diagnostics only."""
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = rgb.mean(axis=2)
    mask = (gray < 214).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .92):] = 0
    mask[:6] = 0
    mask[-6:] = 0
    mask[:, :6] = 0
    mask[:, -6:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(300, areas.max() * .02)) + 1
    return np.isin(labels, keep).astype(np.uint8)


VIEW_NOTES = {
    'front': ('参考正视把提环画在筒体右侧、杠杆在左；原照则明确显示提环与杠杆同在'
              '可见的一侧（拉杆左侧），参考在该处呈镜像，候选以原照为准。参考还把杠杆'
              '画成圆管，照片清楚显示为绿色空心矩形杆。'),
    'back': ('背面在原照不可见；参考背视用于推断齿轮座背面和底座轮廓。参考的提环侧向'
             '与照片矛盾，按镜像处理，不作为依据；杠杆角度与左右视也不一致。'),
    'left': ('左视参考给出杠杆水平伸出、齿轮座侧面和提环侧面；照片中杠杆约向上 35°，'
             '候选取一致的向上姿态，左右视为推断。'),
    'right': ('右视参考给出齿轮座圆端面/螺栓、亮绿拉钮和提环边缘；'
              '齿轮内部、连杆背面和拉钮轴为推断。'),
    'top': ('顶视参考给出圆形底座、螺杆顶面、12 面绿色上环和杠杆方位；提环在画面'
            '下方，与原照的 -Y 侧一致。底座下方的五条筋和中心凸台只据未标定底视参考。'),
    'bottom': ('底视参考给出五条放射筋与中心圆凸台；原照只看到底座正面，'
               '底面细节和筋的数量为推断，中央凹处不是通孔。'),
}


def make_report_11(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '11 千斤顶1 · QL5 螺旋千斤顶 · 六视图参考（未标定）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_11(panel))
        actual = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.getchannel('A')) >= 128)
        iou = (float(np.logical_and(expected, actual).sum() /
                     max(1, np.logical_or(expected, actual).sum()))
               if expected is not None and actual is not None else None)
        old_new = (float(np.logical_and(old_fit, actual).sum() /
                         max(1, np.logical_or(old_fit, actual).sum()))
                   if old_fit is not None and actual is not None else None)
        rows.append({'view': name,
                     'uncalibrated_reference_mask_iou': round(iou, 4) if iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4) if old_new is not None else None,
                     'note': '按最长边等比对齐；参考图未标定，参考分割为近似诊断，仅供发现差异；' + VIEW_NOTES[name]})
        draw.text((x + 12, y), f'{name.upper()}  参考图 / 原模型 / 新候选',
                  font=small, fill='#27363c')
        ref_mask = reference_mask_11(panel)
        rx, ry, rw, rh = cv2.boundingRect(ref_mask)
        panel = panel.crop((max(0, rx - 5), max(0, ry - 5),
                            min(panel.width, rx + rw + 5),
                            min(panel.height, ry + rh + 5)))
        for j, image in enumerate((panel.convert('RGBA'), old, new)):
            if j:
                rect = image.getchannel('A').getbbox()
                if rect:
                    image = image.crop(rect)
            factor = min(182 / image.width, 485 / image.height)
            image = image.resize((max(1, round(image.width * factor)),
                                  max(1, round(image.height * factor))),
                                 Image.Resampling.LANCZOS)
            tile = Image.new('RGBA', (190, 495), 'white')
            tile.alpha_composite(image, ((190 - image.width) // 2,
                                         (495 - image.height) // 2))
            board.paste(tile.convert('RGB'), (x + 5 + j * 195, y + 32))
        draw.text((x + 12, y + 548),
                  (f'参考粗略轮廓 IoU：{iou:.3f} · 原/新 IoU：{old_new:.3f}')
                  if iou is not None and old_new is not None else '轮廓不可比较',
                  font=small, fill='#52646b')
    draw.text((20, 1230),
              '重要：参考图未标定且各面板不是同一转台；图中分数不是实测精度。参考把杠杆画成圆管，'
              '并把提环画在筒体右侧（正视），而照片显示提环与杠杆同在可见侧且杠杆为空心矩形杆；'
              '候选以照片为准。原模型虽通过单张源图轮廓门槛，但六方向审计显示表面糊化、标签不可辨、'
              '提环/齿轮座与筒体粘连，故未复用旧网格。',
              font=small, fill='#8a2b20')
    board.save(out / '六视图-新旧对照.jpg', quality=95)
    return rows


def _tile_image(path, size):
    image = Image.open(path).convert('RGBA')
    rect = image.getchannel('A').getbbox()
    if rect:
        image = image.crop(rect)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def make_detail_board(out):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    items = (
        ('detail-lever-tip', '杠杆空心矩形截面与开口端'),
        ('detail-lever-socket', '杠杆插座、齿轮座与两颗螺栓'),
        ('detail-knob', '亮绿拉钮与中央凹孔'),
        ('detail-mechanism-side', '齿轮座圆端面与中心六角螺栓'),
        ('detail-loop', '提环正面与两处连接座'),
        ('detail-loop-mounts', '提环上连接座与销'),
        ('detail-collar', '12 面上环与螺杆'),
        ('detail-screw', '镀铬螺杆表面'),
        ('detail-cap', '螺杆顶帽与凸缘'),
        ('detail-label', '重绘标签（文字近似，非原图印刷）'),
        ('detail-emboss', '筒体铸字 5T（字体与浮雕为推断）'),
        ('detail-base-front', '底座裙边、筒体肩部与 5T'),
        ('detail-base-bottom', '底面五条放射筋与中心凸台'),
        ('detail-surface', '绿色漆面、磨损与锈斑'),
        ('new-front', '整体正视（透明背景）'),
    )
    board = Image.new('RGB', (2000, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '11 千斤顶1 · 关键局部检查（新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = (i % 8) * 250
        y = 55 + (i // 8) * 780
        path = out / 'renders' / f'{name}.png'
        image = _tile_image(path, (232, 660))
        tile = Image.new('RGBA', (242, 675), 'white')
        tile.alpha_composite(image, ((242 - image.width) // 2,
                                     (675 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, 1600),
              '这些局部图用于人工检查结构；不能证明未观察面、尺寸、装配关系、孔深、螺纹或内部通道。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 16)
    board = Image.new('RGB', (1800, 900), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '11 千斤顶1 · 隔离开口与结构检查',
              font=font, fill='#18242b')
    items = (('lever-bore-only', '杠杆空心矩形通孔', 'lever_bore'),
             ('handle-loop-only', '提环内开口', 'handle_loop'),
             ('base-bottom-only', '底面筋与中心凸台（非通孔）', None),
             ('body-only', '封闭筒体（外观）', None))
    for i, (name, label, key) in enumerate(items):
        x = i * 440
        image = _tile_image(out / 'renders' / f'{name}.png', (410, 620))
        tile = Image.new('RGBA', (425, 640), 'white')
        tile.alpha_composite(image, ((425 - image.width) // 2,
                                     (640 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 8, 92))
        if key:
            check = checks[key]
            status = '通过' if check['passed'] else '未通过'
            text = f'{label}：{check["detected_openings"]}/{check["expected"]} {status}'
            color = '#1d5c3a' if check['passed'] else '#8a2b20'
        else:
            text = f'{label}（不参与孔数判定）'
            color = '#52646b'
        draw.text((x + 10, 48), text, font=small, fill=color)
    draw.text((20, 805),
              '透明背景隔离渲染；杠杆矩形管和提环内圈为真实透明开口。孔数不能证明孔深、'
              '壁厚、螺纹、内部齿轮或装配关系；中央凹处为盲孔，不参与通孔判定。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '11 千斤顶1 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选绿漆、镀铬与钢件 PBR'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (840 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), '同一中性光与相机；高光来自实时照明，不烘焙在底色中。'
                         '原模型为单图重建的顶点色，不代表旧模型整体通过。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '11 千斤顶1 · 变光检查（同视角）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光位置 A'),
                                       ('new-light-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '灯光移动后高光和明暗应随视角变化；本图只检查反射响应，不证明现场光照方向。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '11 千斤顶1 · 金属反射变光对照',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光 A'),
                                       ('new-metal-alternate', '主光 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '镀铬螺杆、钢制螺栓/销与亮绿塑料拉钮的高光随灯光位置移动；不能据此推断现场光源方向或强度。',
              font=small, fill='#52646b')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 17)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    for i in range(24):
        image = _tile_image(out / 'renders' / f'turn-{i:02d}.png', (220, 215))
        tile = Image.new('RGBA', (230, 222), 'white')
        tile.alpha_composite(image, ((230 - image.width) // 2,
                                     (222 - image.height) // 2))
        x = (i % 6) * 240 + 5
        y = (i // 6) * 250 + 4
        board.paste(tile.convert('RGB'), (x, y))
        draw.text((x + 8, y + 224), f'{i * 15:03d}°', font=font, fill='#344044')
    board.save(out / '转台-24方向.jpg', quality=94)


def audit_row(number):
    if not AUDIT.is_file():
        return None
    for line in AUDIT.read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('number') == number:
                return row
    return None


def reference_ratio_diagnostics(reference):
    image = Image.open(reference).convert('RGB')
    out = {}
    for i, name in enumerate(VIEWS):
        panel = panel_image(image, i)
        mask = reference_mask_11(panel)
        points = cv2.findNonZero(mask)
        if points is None:
            out[name] = None
            continue
        x, y, ww, hh = cv2.boundingRect(points)
        out[name] = {'bbox_px': [int(x), int(y), int(ww), int(hh)],
                     'width_over_height': round(ww / max(1, hh), 4)}
    out['proportion_note'] = (
        'Clean-photo alpha: 998 px wide x 1118 px tall including the raised '
        'lever. Candidate keeps about 0.68 m overall height. These are pixel '
        'ratios and a model-library visual scale, not measurements. The printed '
        '220/150 mm label data was not used to set dimensions.')
    out['inconsistency_note'] = (
        'The reference panels are not a consistent turntable: the lever is a '
        'round pipe in the front/back/left/right panels but a hollow '
        'rectangular bar in the photo; its angle is near horizontal in the '
        'left/right panels but strongly raised in the photo/front/back panels; '
        'and the front/back panels place the carry loop on the opposite side '
        'from the photo, so they are treated as mirrored there.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=11)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 11:
        p.error('this entry supports item 11 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/11-jack-draft1' if a.draft
                        else 'outputs/11千斤顶1-六视图重建-v1')
    a.out, a.library, a.blender = (x.resolve() for x in (a.out, a.library, a.blender))
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = next(r for r in (json.loads(line) for line in
                           a.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == a.number)
    for name, expected in (('mesh', 'mesh_sha256'), ('rust', 'rust_sha256'),
                           ('original', 'clean_sha256')):
        if digest(Path(row[name])) != row[expected]:
            raise ValueError(f'{name} hash differs from the model library')
    alpha_path = Path(row['source_view']).parent / 'source-alpha.png'
    if digest(alpha_path) != row['source_alpha_sha256']:
        raise ValueError('source alpha hash differs from the model library')
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    if not len(xs):
        raise ValueError('source alpha is empty')
    source_alpha_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    config_path = HERE / 'jack11_config.json'
    worker_path = HERE / 'mesh_rebuild_11_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True, text=True,
                             check=True).stdout.splitlines()[0]
    signature_input = {'reference': digest(a.reference), 'mesh': row['mesh_sha256'],
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
                       'draft': a.draft}
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if a.out.exists() and any(a.out.iterdir()):
        record = a.out / 'result.json'
        mesh = a.out / 'mesh.glb'
        if record.is_file() and mesh.is_file():
            old = json.loads(record.read_text(encoding='utf8'))
            required = [a.out / 'model.blend', a.out / 'green_base.png',
                        a.out / 'green_mr.png', a.out / 'green_normal.png',
                        a.out / 'chrome_base.png', a.out / 'chrome_mr.png',
                        a.out / 'chrome_normal.png', a.out / 'steel_base.png',
                        a.out / 'steel_mr.png', a.out / 'steel_normal.png',
                        a.out / 'label_base.png', a.out / '六视图-新旧对照.jpg',
                        a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '转台-24方向.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('detail-lever-tip', 'detail-lever-socket', 'detail-knob',
                              'detail-mechanism-side', 'detail-loop', 'detail-loop-mounts',
                              'detail-collar', 'detail-screw', 'detail-cap',
                              'detail-label', 'detail-emboss', 'detail-base-front',
                              'detail-base-bottom', 'detail-surface',
                              'lever-bore-only', 'handle-loop-only',
                              'base-bottom-only', 'body-only',
                              'new-light-alternate', 'new-material-close',
                              'new-metal-alternate', 'old-material-close')]
                required += [a.out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_green_atlas(Path(row['rust']), a.out, config['material_profile'])
    textures.update(make_chrome(a.out, config['material_profile']))
    textures.update(make_steel(a.out, config['material_profile']))
    textures.update(make_label(a.out))
    request = {'out': str(a.out), 'original_mesh': row['mesh'], 'textures': textures,
               'config': config, 'source_camera': row['reconstruction']['source_camera'],
               'source_alpha_bbox': source_alpha_bbox, 'draft': a.draft}
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

    with Image.open(alpha_path) as alpha, \
            Image.open(a.out / 'renders' / 'new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                   np.asarray(rendered.getchannel('A')))
    views = make_report_11(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(11)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    aligned_fidelity = None
    if not a.draft:
        make_turntable(a.out)
        # Best-effort silhouette diagnostic from the 315-degree turntable view,
        # aligned by bounding box. The photo is a three-quarter view, so the
        # fixed front-camera score above is not a fair pose comparison. This is
        # still an uncalibrated diagnostic, never an acceptance gate.
        with Image.open(alpha_path) as source_alpha_image, \
                Image.open(a.out / 'renders/turn-21.png') as turn:
            source_fit_mask = mask_fit(np.asarray(source_alpha_image.convert('L')) >= 128)
            turn_fit_mask = mask_fit(np.asarray(turn.getchannel('A')) >= 128)
        if source_fit_mask is not None and turn_fit_mask is not None:
            aligned_fidelity = source_fidelity(source_fit_mask * 255, turn_fit_mask * 255)
            aligned_fidelity['view'] = '315-degree turntable (photo-like three-quarter side)'
            aligned_fidelity['alignment'] = 'mask bounding box fit to 400 px; not a calibrated camera'
            aligned_fidelity['acceptance_gate'] = False
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('lever_bore', 'lever-bore-only.png', 1),
                ('handle_loop', 'handle-loop-only.png', 1)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    green_checks = green_texture_checks(a.out / 'green_base.png', a.out / 'green_mr.png')
    chrome_checks = chrome_texture_checks(a.out / 'chrome_base.png',
                                          a.out / 'chrome_mr.png')
    steel_checks = steel_texture_checks(a.out / 'steel_base.png',
                                        a.out / 'steel_mr.png')
    label_checks = label_texture_checks(a.out / 'label_base.png')
    plastic_checks = plastic_check(config['material_profile'])
    names = [part['name'] for part in worker['parts']]
    structural = {
        'cast_base': 'Cast base' in names,
        'base_bottom_ribs': sum(name.startswith('Base bottom rib') for name in names),
        'base_bottom_ribs_expected': config['base']['bottom']['rib_count'],
        'base_center_boss': 'Base center boss' in names,
        'base_center_recess': 'Base center recess' in names,
        'base_body_collar': 'Base body collar' in names,
        'jack_body': 'Jack body' in names,
        'top_collar': 'Top collar' in names,
        'collar_lugs': sum(name.startswith('Collar lug ') and
                           not name.startswith('Collar lug pin') for name in names),
        'collar_lugs_expected': 2,
        'collar_lug_pins': sum(name.startswith('Collar lug pin') for name in names),
        'collar_lug_pins_expected': 2,
        'chrome_screw': 'Chrome screw' in names,
        'cap_lip': 'Screw cap lip' in names,
        'cap_top': 'Screw cap top' in names,
        'cap_top_rim': 'Screw cap top rim' in names,
        'nameplate': 'Nameplate' in names,
        'emboss_pad': 'Emboss pad' in names,
        'emboss_text': 'Emboss text' in names,
        'crank_housing': 'Crank housing' in names,
        'lever_socket': 'Lever socket' in names,
        'socket_bore_liner': 'Socket bore liner' in names,
        'socket_rim': 'Socket rim' in names,
        'socket_bolts': sum(name.startswith('Socket bolt') for name in names),
        'socket_bolts_expected': 2,
        'lever_square_tube': 'Lever square tube' in names,
        'lever_bore_liner': 'Lever bore liner' in names,
        'mechanism_side_plate': 'Mechanism side plate' in names,
        'mechanism_side_bolt': 'Mechanism side bolt' in names,
        'lower_bracket': 'Lower bracket' in names,
        'lower_bracket_bolt': 'Lower bracket bolt' in names,
        'link_plate': 'Link plate' in names,
        'link_pin': 'Link pin' in names,
        'knob_stem': 'Release knob stem' in names,
        'knob_cap': 'Release knob cap' in names,
        'knob_bore_liner': 'Release knob bore liner' in names,
        'knob_rim': 'Release knob rim' in names,
        'carry_loop': 'Carry handle loop' in names,
        'loop_backplate': 'Loop backplate' in names,
        'body_back_rib': 'Body back rib' in names,
        'loop_mounts': sum(name.startswith('Loop mount') for name in names),
        'loop_mounts_expected': 2,
        'loop_pins': sum(name.startswith('Loop pin') for name in names),
        'loop_pins_expected': 2,
        'note': ('Part presence proves only that a feature was modelled, not its '
                 'size, fit or hidden faces. The gear internals, pivot axes, the '
                 'knob function and the hidden back side are partly inferred.'),
    }
    reflection = {'status': 'not_run'}
    if not a.draft:
        first = np.asarray(Image.open(a.out / 'renders/new-material-close.png').convert('RGBA'))
        second = np.asarray(Image.open(a.out / 'renders/new-metal-alternate.png').convert('RGBA'))
        mask = first[:, :, 3] >= 128
        reflection = {
            'mean_abs_luma_difference': round(float(np.abs(first[:, :, :3].astype(float) -
                                                          second[:, :, :3].astype(float))[mask].mean()), 4),
            'changed': bool(np.abs(first[:, :, :3].astype(float) -
                                   second[:, :, :3].astype(float))[mask].mean() > 2.0),
            'note': 'Same camera, different key-light direction; highlights must move.'
        }
    structural_ok = bool(
        structural['cast_base'] and
        structural['base_bottom_ribs'] == structural['base_bottom_ribs_expected'] and
        structural['base_center_boss'] and structural['base_center_recess'] and
        structural['base_body_collar'] and structural['jack_body'] and
        structural['top_collar'] and
        structural['collar_lugs'] == structural['collar_lugs_expected'] and
        structural['collar_lug_pins'] == structural['collar_lug_pins_expected'] and
        structural['chrome_screw'] and structural['cap_lip'] and
        structural['cap_top'] and structural['cap_top_rim'] and
        structural['nameplate'] and structural['emboss_pad'] and
        structural['emboss_text'] and structural['crank_housing'] and
        structural['lever_socket'] and structural['socket_bore_liner'] and
        structural['socket_rim'] and
        structural['socket_bolts'] == structural['socket_bolts_expected'] and
        structural['lever_square_tube'] and structural['lever_bore_liner'] and
        structural['mechanism_side_plate'] and
        structural['mechanism_side_bolt'] and structural['lower_bracket'] and
        structural['lower_bracket_bolt'] and structural['link_plate'] and
        structural['link_pin'] and structural['knob_stem'] and
        structural['knob_cap'] and structural['knob_bore_liner'] and
        structural['knob_rim'] and
        structural['carry_loop'] and structural['loop_backplate'] and
        structural['body_back_rib'] and
        structural['loop_mounts'] == structural['loop_mounts_expected'] and
        structural['loop_pins'] == structural['loop_pins_expected'])
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 25 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': structural_ok,
        'material_tiles': bool(green_checks['passed'] and chrome_checks['passed'] and
                               steel_checks['passed'] and label_checks['passed'] and
                               plastic_checks['passed']),
        'metal_reflection': bool(reflection.get('changed', False)),
        'reference_diagnostic_only': True,
        'browser_page': False,
        'source_fidelity': bool(fidelity.get('passed', False)),
    }
    if a.draft:
        passed['isolated_openings'] = 'not_run'
        passed['metal_reflection'] = 'not_run'
        passed['material_tiles'] = 'not_run'
    report = {
        'version': config['version'], 'number': 11, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_visual_match_reference',
        'reference_role': 'visual_match_unverified',
        'method': 'parametric_part_rebuild_from_source_photo_proportions_with_visual_match_reference',
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_clean_sha256': row['clean_sha256'],
        'original_rust_sha256': row['rust_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.955 only measures one silhouette; '
            'the six-view audit shows a smoothed, fused surface: the label is '
            'unreadable, the carry loop and crank/gear assembly merge into the '
            'body, and the lever tip is blobby. The old mesh is therefore not '
            'reused for the candidate base geometry.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'green_texture_checks': green_checks,
        'chrome_texture_checks': chrome_checks,
        'steel_texture_checks': steel_checks,
        'label_texture_checks': label_checks,
        'plastic_checks': plastic_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'source_diagnostic_note': (
            'The supplied clean photo is a three-quarter product view with the '
            'lever raised, while the strict source-camera render looks along +X '
            '(label-forward). The strict score is therefore not a fair pose '
            'comparison. A second best-effort diagnostic aligns the 315-degree '
            'turntable silhouette to the source alpha by bounding box; it is also '
            'uncalibrated and is not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is an uncalibrated visual-match '
                       'reference, not a calibrated image set; its lever shape '
                       'and angle are inconsistent between panels. It is used '
                       'only for hidden topology and gross differences.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view sheet is uncalibrated and not a measured image set; the reference segmentation is a diagnostic.',
            'The reference panels are not a consistent turntable: the lever is drawn as a round pipe in several panels while the photo shows a hollow rectangular bar, and its angle differs between panels.',
            'The working overall height (about 0.68 m), base size and all other sizes follow photo pixel ratios and the model-library visual scale; they are not measured dimensions.',
            'The printed 220/150 mm label data is a product marking, not a measurement of this used jack, and was not used to set dimensions.',
            'The rectangular lever wall thickness, length, socket angle and open far end are inferred; the photo shows the section and opening but not their exact dimensions.',
            'The crank/gear housing internals, side plate, bolt pattern, link geometry and the exact pivot axes are inferred.',
            'The bright green knob axis, bore depth and function are inferred.',
            'The carry-loop hidden body-side mounts, rod diameter and upper pin are inferred.',
            'The 5T cast relief font and height are inferred from the photo.',
            'The five bottom ribs and centre boss follow the uncalibrated bottom reference panel; the central recess is a blind cast detail, not a through-hole.',
            'The two small collar ears and their pins are inferred from small tabs in the reference panels.',
            'The label artwork is recreated from legible text; the original graphic, small print and certification marks are not reproduced.',
            'The chrome screw surface, paint wear and rust are synthesised; they are not a photograph of hidden faces.',
            'Rust patches are synthesised from small observed patches, not from the actual hidden surfaces.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the lever section/opening, crank housing, carry loop and label.',
            'Confirm the lever cross-section, wall thickness, length, socket angle and whether the far end is open on the actual jack.',
            'Confirm the crank/gear housing form, side-plate bolt pattern, link and pivot axes from a side/back photograph or measured sample.',
            'Confirm the bright green knob axis, bore depth and function.',
            'Confirm the carry-loop attachment points, rod diameter and hidden back-side form.',
            'Confirm the 5T relief font, height and whether it is cast or stamped.',
            'Decide whether the five bottom ribs and centre boss match the actual casting; the bottom is only seen in the uncalibrated reference.',
            'Check the label dimensions, exact text and whether the original label is still present; the candidate label is a recreation.',
            'Check the chrome screw finish and the upper collar facet count against a higher-resolution sample.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 12,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_11.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 11, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed, 'material_checks': green_checks,
        'chrome_checks': chrome_checks, 'steel_checks': steel_checks,
        'label_checks': label_checks, 'plastic_checks': plastic_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'reference_role': 'visual_match_unverified',
        'pending_review': report['pending_review'],
        'next_number': 12,
        'manual_only': True,
        'notes': ('The user six-view sheet is an uncalibrated visual-match reference '
                  'whose panels disagree on the lever form and angle; the photo is '
                  'the primary shape evidence. The old model and items 01-10 were '
                  'not modified. Candidate is manual_only. Browser review page and '
                  'visual acceptance are pending unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '11-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
