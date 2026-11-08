"""Build the review-only 10 t bottle-jack candidate (item 10).

Shape follows the original clean/rust photograph for proportions and visible
features.  The user-supplied six-view sheet is an uncalibrated visual-match
reference of the same TY10010-style 10 t bottle jack: a red cast base, painted
reservoir, chrome ram, checkered saddle, side pump with an angled handle
socket, carry loop and a detached handle tube.  The sheet is used for hidden
topology and as a diagnostic comparison, never as measured geometry.  The
candidate never replaces the library mesh and never enters automatic
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
from mesh_rebuild_item import _tile_detail, mask_fit, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE_SHA256 = 'd21cebc1ef8ddfe91959c39760bdbaa4da0e453abac0804c0d87aaf88edb414c'
SOURCE_SIZE = (1254, 1254)
TAU = math.tau
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'


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
def make_red_atlas(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('10 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((330, 330, 500, 430), (250, 980, 360, 1080),
                (430, 900, 540, 990), (500, 300, 590, 390))]
    rng = np.random.default_rng(21010)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [(profile['red']['clean'], 3.0),
             (profile['red']['weathered'], 3.6),
             (profile['red']['worn'], 2.4),
             (profile['red']['rusted'], 2.0)]
    rust_rgb = np.array((112, 79, 54), np.float32)
    dust_rgb = np.array((82, 77, 70), np.float32)
    scratch_rgb = np.array((206, 198, 190), np.float32)
    for i, (mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((110, 110), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((820, 820), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Cast-enamel orange-peel: fine isotropic cells, not a directional brush.
        peel = cv2.GaussianBlur(rng.normal(size=(size // 4, size // 4)).astype(np.float32),
                                (0, 0), 1.2)
        peel = cv2.resize(peel, (size, size), interpolation=cv2.INTER_CUBIC)
        peel *= brush_strength / max(1e-6, float(peel.std()))
        field = .50 * (coarse - .5) + .30 * (fine - .5) + .20 * (speck - .5)
        amount = mat['rust_amount']
        # Sparse rust blooms, edge wear and a little dust.
        patch = np.clip((coarse - (1 - amount * 1.6)) * 18, 0, 1)
        rust = patch * np.clip((fine - .18) * 1.25, 0, 1)
        rust = np.clip(rust + np.clip((speck - .996) * 55, 0, 1) * amount, 0, 1)
        dust = np.clip((speck - .9982) * 45, 0, 1) * .035
        color = np.asarray(mat['base_rgb'], np.float32)
        rgb = color[None, None, :] * (1 + field[:, :, None] * .045)
        rgb += detail[:, :, None] * .075 + peel[:, :, None] * .55
        rgb += ((speck - .5) * .24)[:, :, None]
        scratches = np.clip((peel - 2.6) * .045, 0, 1) * (1 - rust) * (i == 2)
        rgb = rgb * (1 - scratches[:, :, None] * .30) + \
            scratch_rgb[None, None, :] * (scratches[:, :, None] * .30)
        rust_rgb_local = rust_rgb[None, None, :] * (.78 + .40 * fine[:, :, None])
        rgb = rgb * (1 - rust[:, :, None]) + rust_rgb_local * rust[:, :, None]
        rgb = rgb * (1 - dust[:, :, None] * .45) + \
            dust_rgb[None, None, :] * (dust[:, :, None] * .45)
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        roughness = (mat['roughness'] + rust * .24 + dust * .10
                     + field * .035 + peel * .006)
        metallic = mat['metallic'] * (1 - rust * .80)
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip(roughness * 255, 0, 255),
                           np.clip(metallic * 255, 0, 255)),
                          axis=2).astype(np.uint8)
        bump_source = peel + detail * .30
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
    for name, arr in (('red_base', base), ('red_mr', mr), ('red_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_chrome(out, profile):
    size = 1024
    rng = np.random.default_rng(21011)
    coarse = cv2.resize(rng.random((80, 80), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((900, 900), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    brush = cv2.resize(rng.normal(size=(180, size)).astype(np.float32),
                       (size, size), interpolation=cv2.INTER_CUBIC)
    brush = cv2.GaussianBlur(brush, (0, 0), 1.0)
    brush *= 4.0 / max(1e-6, float(brush.std()))
    base_rgb = np.asarray(profile['chrome']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + brush[:, :, None] * .020
                                          + (coarse[:, :, None] - .5) * .035
                                          + (fine[:, :, None] - .5) * .018)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['chrome']['roughness'] + brush * .012
                    + (coarse - .5) * .025 + (fine - .5) * .010, .08, .30)
    metallic = np.full((size, size), profile['chrome']['metallic'], np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur(brush * .55 + fine * .25, (0, 0), .7)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 11
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 11
    nx = np.clip(-dx * .55, -.05, .05)
    ny = np.clip(dy * .25, -.025, .025)
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
    rng = np.random.default_rng(21012)
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


def make_saddle(out, profile):
    size = 1024
    rng = np.random.default_rng(21013)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    period = 128.0
    # Diamond/checker knurl: crossed sinusoids, flattened into raised diamonds.
    lattice = (np.cos(TAU * (xx + yy) / period) +
               np.cos(TAU * (xx - yy) / period))
    height = np.clip(lattice + .35, 0, 1.7) / 1.7
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 1.2)
    base_rgb = np.asarray(profile['saddle']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (height[:, :, None] - .5) * .16)
    rgb += (rng.normal(size=(size, size, 1)) * 2.2).astype(np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['saddle']['roughness'] + (height - .5) * .10
                    + rng.normal(size=(size, size)).astype(np.float32) * .015,
                    .30, .60)
    metallic = np.full((size, size), profile['saddle']['metallic'], np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) * 7.0
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) * 7.0
    nx = np.clip(-dx, -.65, .65)
    ny = np.clip(dy, -.65, .65)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('saddle_base', rgb), ('saddle_mr', mr),
                      ('saddle_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_label(out):
    width, height = 768, 1280
    image = Image.new('RGB', (width, height), (8, 17, 38))
    draw = ImageDraw.Draw(image)
    yellow = (247, 205, 26)
    dark = (8, 16, 36)
    draw.rectangle((0, 0, width, 226), fill=yellow)
    draw.rectangle((0, 1080, width, height), fill=yellow)
    for y in range(226, 1080):
        t = (y - 226) / max(1, 1080 - 226)
        col = (int(10 + 8 * (1 - t)), int(26 + 16 * (1 - t)), int(64 + 28 * (1 - t)))
        draw.line((0, y, width, y), fill=col)
    # Diagonal blue light streaks in the dark field.
    for i in range(-8, 12):
        x = i * 90
        draw.line((x, 1080, x + 420, 226), fill=(14, 43, 94), width=6)
    # Stylised five-petal mark (recreated, not the original artwork).
    cx, cy = 78, 92
    for k in range(5):
        a = math.radians(90 + k * 72)
        px, py = cx + 24 * math.cos(a), cy + 24 * math.sin(a)
        draw.ellipse((px - 20, py - 20, px + 20, py + 20),
                     fill=(250, 246, 236), outline=(196, 40, 44), width=2)
    draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=(224, 168, 20))
    font_small = ImageFont.truetype(LABEL_FONT, 26)
    font_head = ImageFont.truetype(LABEL_FONT_BOLD, 74)
    font_title = ImageFont.truetype(LABEL_FONT_BOLD, 84)
    font_en = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 22)
    font_ten = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 210)
    font_safety = ImageFont.truetype(LABEL_FONT_BOLD, 38)
    font_body = ImageFont.truetype(LABEL_FONT, 28)
    font_spec = ImageFont.truetype(LABEL_FONT, 27)
    font_code = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 66)
    draw.text((138, 18), '立式', font=font_head, fill=(28, 26, 22))
    draw.text((138, 92), '油压千斤顶', font=font_title, fill=(28, 26, 22))
    draw.text((140, 186), 'HYDRAULIC BOTTLE JACK', font=font_en, fill=(46, 42, 32))
    draw.text((32, 286), '10t', font=font_ten, fill=yellow)
    draw.text((34, 520), '10 т', font=ImageFont.truetype(LABEL_FONT, 34), fill=(220, 220, 224))
    draw.text((252, 286), '⚠ 安全注意事项', font=font_safety, fill=yellow)
    lines = ('· 使用前请详细阅读说明书', '· 严禁超载使用',
             '· 千斤顶顶下不得站人', '· 请坚硬平整的地面上使用',
             '· 严禁随意拆卸或改装')
    for i, text in enumerate(lines):
        draw.text((252, 348 + i * 47), text, font=font_body, fill=(236, 238, 240))
    # Small diagram panel.
    draw.rounded_rectangle((248, 600, 520, 930), radius=10, fill=(9, 38, 92),
                           outline=(70, 120, 200), width=3)
    draw.rectangle((368, 646, 400, 900), fill=(210, 60, 55))
    draw.rectangle((354, 880, 414, 912), fill=(210, 60, 55))
    draw.rectangle((374, 618, 394, 648), fill=(220, 60, 55))
    draw.ellipse((466, 666, 492, 692), fill=(210, 60, 55))
    draw.line((448, 678, 490, 678), fill=(230, 230, 235), width=3)
    draw.line((392, 618, 392, 930), fill=(210, 220, 235), width=2)
    draw.line((380, 618, 404, 618), fill=(210, 220, 235), width=2)
    draw.line((380, 930, 404, 930), fill=(210, 220, 235), width=2)
    draw.text((292, 744), 'TY10010', font=ImageFont.truetype(LABEL_FONT_BOLD, 30),
              fill=(238, 240, 245))
    draw.text((276, 792), '10 t', font=ImageFont.truetype(LABEL_FONT_BOLD, 42),
              fill=yellow)
    draw.text((34, 946), '最低高度: 200mm', font=font_spec, fill=(236, 238, 240))
    draw.text((34, 990), '起升高度: 125mm', font=font_spec, fill=(236, 238, 240))
    draw.text((34, 1034), '调整高度: 60mm', font=font_spec, fill=(236, 238, 240))
    # Generic circular quality marks (not reproductions of certification logos).
    for cx2, cy2, label in ((600, 880, 'QC'), (700, 880, '10T')):
        draw.ellipse((cx2 - 48, cy2 - 48, cx2 + 48, cy2 + 48),
                     outline=(200, 204, 210), width=4, fill=(32, 34, 40))
        draw.text((cx2 - 22, cy2 - 20), label, font=ImageFont.truetype(LABEL_FONT_BOLD, 26),
                  fill=(230, 230, 234))
    draw.text((34, 1108), 'TY10010', font=font_code, fill=dark)
    draw.text((600, 1120), '10t', font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 50),
              fill=dark)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(30, 30, 30), width=3)
    base = np.asarray(image)
    mr = np.zeros((height, width, 3), np.uint8)
    mr[..., 0] = 255
    mr[..., 1] = int(.55 * 255)
    mr[..., 2] = 0
    normal = np.zeros((height, width, 3), np.uint8)
    normal[..., 0] = 128
    normal[..., 1] = 128
    normal[..., 2] = 255
    paths = {}
    for name, arr in (('label_base', base), ('label_mr', mr),
                      ('label_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


# ---------------------------------------------------------------------------
# Texture checks
# ---------------------------------------------------------------------------
def red_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    tiles = {'red_clean': np.s_[0:1024, 0:1024],
             'red_weathered': np.s_[0:1024, 1024:2048],
             'red_worn': np.s_[1024:2048, 0:1024],
             'red_rusted': np.s_[1024:2048, 1024:2048]}
    values = {}
    for name, sl in tiles.items():
        rgb = base[sl]
        means = rgb.mean(axis=(0, 1))
        rust = ((rgb[..., 1] > rgb[..., 0] * .45) &
                (rgb[..., 2] > rgb[..., 0] * .34)).mean()
        values[name] = {
            'base_rgb_mean': [round(float(v), 1) for v in means],
            'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
            'metallic_mean': round(float(mr[sl][..., 2].mean()), 4),
            'chroma': round(float(means.max() - means.min()), 2),
            'rust_fraction': round(float(rust), 4)}
    c, w, r = values['red_clean'], values['red_weathered'], values['red_rusted']
    values['passed'] = bool(
        c['metallic_mean'] < .15 and .28 < c['roughness_mean'] < .52 and
        c['chroma'] > 90 and c['rust_fraction'] < .02 and
        w['metallic_mean'] < .15 and w['chroma'] > 80 and
        r['roughness_mean'] > .55 and r['metallic_mean'] < .15 and
        r['rust_fraction'] > .10)
    values['note'] = ('Tile means; rust is detected by brown colour. The red is '
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
    values['passed'] = bool(means.min() > 150 and values['chroma'] < 18 and
                            values['metallic_mean'] > .90 and
                            values['roughness_mean'] < .30)
    values['note'] = 'Polished chrome ram; directional bright brushing is weak and does not encode a light direction.'
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
    values['note'] = 'Zinc-plated steel pins and release valve.'
    return values


def saddle_texture_checks(base_path, mr_path, normal_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    normal = np.asarray(Image.open(normal_path).convert('RGB'), dtype=np.float32)
    means = base.mean(axis=(0, 1))
    normal_dev = float(np.abs(normal[..., :2] - 128).mean())
    values = {'red_mean': round(float(means[0]), 1),
              'green_mean': round(float(means[1]), 1),
              'blue_mean': round(float(means[2]), 1),
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'normal_xy_deviation': round(normal_dev, 3)}
    values['passed'] = bool(means[0] > 130 and means[1] < 70 and
                            values['metallic_mean'] < .15 and
                            values['normal_xy_deviation'] > 8.0)
    values['note'] = 'Checker knurl normal map only; pitch is not measured.'
    return values


def label_texture_checks(base_path):
    rgb = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    yellow = ((rgb[..., 0] > 180) & (rgb[..., 1] > 150) & (rgb[..., 2] < 90)).mean()
    dark = (rgb.max(axis=2) < 90).mean()
    values = {'yellow_fraction': round(float(yellow), 4),
              'dark_fraction': round(float(dark), 4),
              'size': [int(rgb.shape[1]), int(rgb.shape[0])]}
    values['passed'] = bool(yellow > .12 and dark > .12)
    values['note'] = 'Label artwork is recreated from legible photo text; small print and the original graphic are not reproduced.'
    return values


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 10, y0 + 8, x0 + w // 3 - 10, y0 + h // 2 - 8))


def reference_mask_10(panel):
    """Approximate combined mask (jack + detached bar) for diagnostics only."""
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = rgb.mean(axis=2)
    mask = (gray < 205).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .91):] = 0
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
    'front': ('参考正视与照片同为标签朝前、左侧提环、右侧泵体与斜握把插座；'
              '参考未标定，只作外观诊断；手柄筒在参考图各面板中被单独摆放，'
              '本候选按独立附件站立放置，姿态为推断'),
    'back': ('背面在原照不可见；参考背视为提环与泵体镜像、底座拱形凸台和螺栓的推断依据；'
             '参考图把独立手柄画在画面右侧，与固定世界坐标不一致，说明各面板不是同一转台'),
    'left': ('左视参考给出提环侧面、标签包裹和底座侧面轮廓；'
             '左背面的提环连接、底座凸台与释放阀位置为推断'),
    'right': ('右视参考展示泵体、连杆双销、倾斜握把插座和底座销；'
              '插座内部、销轴直径与泵内弹簧/单向阀为推断'),
    'top': ('顶视参考给出近似方形底座、四角孔、中心圆台与十字筋、提环和泵头方位；'
            '四角孔是安装孔还是铸造孔未确认，候选按通孔建模'),
    'bottom': ('底视参考给出四角孔与十字加强筋；原照只看到底座前面，'
               '底面细节主要依据未标定参考图，背侧和底座内部为推断'),
}


def make_report_10(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '10 千斤顶 · 六视图参考（未标定）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_10(panel))
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
        ref_mask = reference_mask_10(panel)
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
              '重要：参考图未标定，图中分数不是实测精度；原模型虽通过单张源图轮廓门槛，'
              '但六方向审计显示左/右视压扁、手柄与主体粘连。候选按照片比例与可见结构分部件重建，'
              '参考图只用于背侧/底面拓扑与外观对照；孔洞/材质/反射检查另行记录。',
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
        ('detail-saddle', '顶部红色菱形滚花托盘'),
        ('detail-ram-hex', '镀铬柱塞、六角锁母与上环'),
        ('detail-shoulder', '筒体肩部与铸造过渡'),
        ('detail-label', '重绘标签（文字近似，非原图印刷）'),
        ('detail-loop', '左侧提环与两处连接座'),
        ('detail-pump-head', '泵体、叉头与倾斜握把插座'),
        ('detail-pump-socket', '握把插座通孔内部'),
        ('detail-pump-link', '泵体连杆、双销与下销'),
        ('detail-base-front', '底座前面拱形凸台与螺栓'),
        ('detail-base-bottom', '底座底面十字筋与四角孔'),
        ('detail-release-valve', '释放阀六角螺塞'),
        ('detail-handle-bar', '独立手柄筒（推断站立姿态）'),
        ('detail-surface', '红色漆面、磨损与锈斑'),
        ('new-front', '整体正视（透明背景）'),
    )
    board = Image.new('RGB', (2000, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '10 千斤顶 · 关键局部检查（新候选）',
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
    draw.text((20, 12), '10 千斤顶 · 隔离开口与结构检查',
              font=font, fill='#18242b')
    items = (('base-holes-only', '底座四角通孔', 'base_holes'),
             ('socket-bore-only', '握把插座通孔', 'socket_bore'),
             ('handle-loop-only', '提环内开口', 'handle_loop'),
             ('handle-bar-only', '手柄筒通孔', 'handle_bar'),
             ('body-only', '封闭筒体（外观）', None))
    for i, (name, label, key) in enumerate(items):
        x = i * 360
        image = _tile_image(out / 'renders' / f'{name}.png', (330, 620))
        tile = Image.new('RGBA', (345, 640), 'white')
        tile.alpha_composite(image, ((345 - image.width) // 2,
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
              '透明背景隔离渲染；孔数由连通域检测。底座四角孔、握把插座通孔、提环内开口和手柄筒通孔为真实透明开口；'
              '孔数不能证明孔深、螺纹、内部通道或装配关系。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '10 千斤顶 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选红漆、镀铬与镀锌钢 PBR'))):
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
    draw.text((20, 12), '10 千斤顶 · 变光检查（同视角）',
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
    draw.text((20, 12), '10 千斤顶 · 金属反射变光对照',
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
    draw.text((20, 810), '镀铬柱塞和镀锌钢件的高光随灯光位置移动；不能据此推断现场光源方向或强度。',
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
        mask = reference_mask_10(panel)
        points = cv2.findNonZero(mask)
        if points is None:
            out[name] = None
            continue
        x, y, ww, hh = cv2.boundingRect(points)
        out[name] = {'bbox_px': [int(x), int(y), int(ww), int(hh)],
                     'width_over_height': round(ww / max(1, hh), 4)}
    out['proportion_note'] = (
        'Clean-photo alpha: main jack 659 px wide x 1196 px tall; detached '
        'handle bar 405 x 864 px. Candidate keeps the main jack at about '
        '0.975 m working height. These are pixel ratios and a model-library '
        'visual scale, not measurements. The printed 200/125/60 mm label data '
        'was not used to set dimensions.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=10)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 10:
        p.error('this entry supports item 10 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/10-jack-draft1' if a.draft
                        else 'outputs/10千斤顶-六视图重建-v1')
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
    config_path = HERE / 'jack10_config.json'
    worker_path = HERE / 'mesh_rebuild_10_worker.py'
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
            required = [a.out / 'model.blend', a.out / 'red_base.png',
                        a.out / 'red_mr.png', a.out / 'red_normal.png',
                        a.out / 'chrome_base.png', a.out / 'chrome_mr.png',
                        a.out / 'chrome_normal.png', a.out / 'steel_base.png',
                        a.out / 'steel_mr.png', a.out / 'steel_normal.png',
                        a.out / 'saddle_base.png', a.out / 'saddle_mr.png',
                        a.out / 'saddle_normal.png', a.out / 'label_base.png',
                        a.out / '六视图-新旧对照.jpg', a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '转台-24方向.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('detail-saddle', 'detail-ram-hex', 'detail-shoulder',
                              'detail-label', 'detail-loop', 'detail-pump-head',
                              'detail-pump-socket', 'detail-pump-link',
                              'detail-base-front', 'detail-base-bottom',
                              'detail-release-valve', 'detail-handle-bar',
                              'detail-surface', 'base-holes-only',
                              'socket-bore-only', 'handle-loop-only',
                              'handle-bar-only', 'body-only', 'new-light-alternate',
                              'new-material-close', 'new-metal-alternate',
                              'old-material-close')]
                required += [a.out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_red_atlas(Path(row['rust']), a.out, config['material_profile'])
    textures.update(make_chrome(a.out, config['material_profile']))
    textures.update(make_steel(a.out, config['material_profile']))
    textures.update(make_saddle(a.out, config['material_profile']))
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
    views = make_report_10(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(10)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    if not a.draft:
        make_turntable(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('base_holes', 'base-holes-only.png', 4),
                ('socket_bore', 'socket-bore-only.png', 1),
                ('handle_loop', 'handle-loop-only.png', 1),
                ('handle_bar', 'handle-bar-only.png', 1)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    material_checks = red_texture_checks(a.out / 'red_base.png', a.out / 'red_mr.png')
    chrome_checks = chrome_texture_checks(a.out / 'chrome_base.png',
                                          a.out / 'chrome_mr.png')
    steel_checks = steel_texture_checks(a.out / 'steel_base.png',
                                        a.out / 'steel_mr.png')
    saddle_checks = saddle_texture_checks(a.out / 'saddle_base.png',
                                          a.out / 'saddle_mr.png',
                                          a.out / 'saddle_normal.png')
    label_checks = label_texture_checks(a.out / 'label_base.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'cast_base': 'Cast base' in names,
        'base_corner_holes': len(config['base']['corner_holes']['centers']),
        'base_bottom_parts': sum(name.startswith('Base bottom') for name in names),
        'base_bottom_parts_expected': 4,
        'base_body_collar': 'Base body collar' in names,
        'base_bosses': sum(name.startswith('Base boss') for name in names),
        'base_bosses_expected': 6,
        'base_bolts': sum(name.startswith('Base bolt') for name in names),
        'base_bolts_expected': 6,
        'release_valve_parts': sum(name.startswith('Release valve') for name in names),
        'release_valve_parts_expected': 2,
        'jack_body': 'Jack body' in names,
        'hex_collar': 'Hex collar' in names,
        'upper_hex_collar': 'Upper hex collar' in names,
        'ram_neck': 'Ram neck' in names,
        'chrome_ram': 'Chrome ram' in names,
        'saddle_skirt': 'Saddle skirt' in names,
        'saddle_cap': 'Saddle cap' in names,
        'saddle_top_rim': 'Saddle top rim' in names,
        'pump_body': 'Pump body' in names,
        'pump_lower_collar': 'Pump lower collar' in names,
        'pump_yoke': 'Pump yoke' in names,
        'pump_pivot_pin': 'Pump pivot pin' in names,
        'pump_socket': 'Pump handle socket' in names,
        'socket_bore_liner': 'Socket bore liner' in names,
        'socket_rims': sum(name.startswith('Socket rim') for name in names),
        'socket_rims_expected': 2,
        'pump_link_plate': 'Pump link plate' in names,
        'pump_link_bolts': sum(name.startswith('Pump link bolt') for name in names),
        'pump_link_bolts_expected': 2,
        'pump_lower_pin': 'Pump lower pin' in names,
        'carry_loop': 'Carry handle loop' in names,
        'loop_mounts': sum(name.startswith('Handle loop mount') for name in names),
        'loop_mounts_expected': 2,
        'detached_handle_bar': 'Detached handle bar' in names,
        'handle_bar_bore_liner': 'Handle bar bore liner' in names,
        'handle_bar_rim': 'Handle bar top rim' in names,
        'nameplate': 'Nameplate' in names,
        'note': ('Part presence proves only that a feature was modelled, not its size, '
                 'fit or hidden faces. The far socket opening, pump internals, base '
                 'boss count, bottom openings and detached-handle pose are partly inferred.'),
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
        structural['cast_base'] and structural['base_bottom_parts'] == 4 and
        structural['base_body_collar'] and
        structural['base_bosses'] == 6 and structural['base_bolts'] == 6 and
        structural['release_valve_parts'] == 2 and structural['jack_body'] and
        structural['hex_collar'] and structural['upper_hex_collar'] and
        structural['ram_neck'] and structural['chrome_ram'] and
        structural['saddle_skirt'] and structural['saddle_cap'] and
        structural['saddle_top_rim'] and structural['pump_body'] and
        structural['pump_lower_collar'] and structural['pump_yoke'] and
        structural['pump_pivot_pin'] and structural['pump_socket'] and
        structural['socket_bore_liner'] and
        structural['socket_rims'] == 2 and structural['pump_link_plate'] and
        structural['pump_link_bolts'] == 2 and structural['pump_lower_pin'] and
        structural['carry_loop'] and structural['loop_mounts'] == 2 and
        structural['detached_handle_bar'] and structural['handle_bar_bore_liner'] and
        structural['handle_bar_rim'] and structural['nameplate'])
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 30 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': structural_ok,
        'material_tiles': bool(material_checks['passed'] and chrome_checks['passed'] and
                               steel_checks['passed'] and saddle_checks['passed'] and
                               label_checks['passed']),
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
        'version': config['version'], 'number': 10, 'asset_id': row['asset_id'],
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
            'The old model-library source IoU 0.945 only measures one silhouette; '
            'the six-view audit shows the left/right views collapsed into a thin '
            'shape and the detached handle fused/floating near the body. The old '
            'mesh is therefore not reused for the candidate base geometry.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'red_texture_checks': material_checks,
        'chrome_texture_checks': chrome_checks,
        'steel_texture_checks': steel_checks,
        'saddle_texture_checks': saddle_checks,
        'label_texture_checks': label_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The supplied clean photo shows the jack at a 3/4 angle with the detached '
            'handle floating to one side. The candidate keeps the photo proportions '
            'but displays the detached handle standing on the floor; that pose is '
            'inferred, so the silhouette score is diagnostic only and is not used as '
            'an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is an uncalibrated visual-match '
                       'reference of the same visible bottle-jack design, not a '
                       'calibrated image set. The detached handle is composited '
                       'separately in the reference panels; the sheet is used only '
                       'for hidden topology and gross differences.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view sheet is uncalibrated and not a measured image set; the reference segmentation is a diagnostic.',
            'The working main-jack height (about 0.975 m), base size and all other sizes follow photo pixel ratios and the model-library visual scale; they are not measured dimensions.',
            'The printed 200/125/60 mm label data is a product marking, not a measurement of this used jack, and was not used to set dimensions.',
            'The detached handle bar is shown in an inferred standing pose; the source photo shows it floating beside the jack and the reference panels composite it independently, so its pose, contact and stability are not established.',
            'The four base corner openings are visible in the bottom reference panel, but whether they are mounting holes, casting openings or something else is not established.',
            'The number and exact form of the base arch bosses are partly inferred; only the two front bosses are visible in the photo, and the back pair is modelled by symmetry.',
            'The pump socket angle, its far opening, the pivot/link pins and the internal pump spring/check-valve are inferred.',
            'The carry-loop back-side attachments, rod diameter and exact profile are inferred.',
            'The label artwork is recreated from legible text; the original graphic, small print and certification marks are not reproduced.',
            'The chrome ram surface and paint wear are synthesised; they are not a photograph of hidden faces.',
            'Rust patches are synthesised from small observed patches, not from the actual hidden surfaces.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the base bosses, pump socket angle, carry loop and detached handle.',
            'Confirm from a measured sample whether the four base corner openings are through-holes, and measure their diameter and depth.',
            'Confirm the number and positions of the base arch bosses and bolt caps on the back side.',
            'Confirm the pump socket bore diameter, depth, far opening and the pump linkage geometry.',
            'Confirm the carry-loop attachment points, rod diameter and hidden back-side form.',
            'Confirm the label dimensions, exact text and whether the original label is still present; the candidate label is a recreation.',
            'Decide how the detached handle should be represented in the scene (separate accessory, stored, or inserted); its current standing pose is inferred.',
            'Check the checker saddle pitch and the chrome ram finish against a higher-resolution sample.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 11,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_10.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 10, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed, 'material_checks': material_checks,
        'chrome_checks': chrome_checks, 'steel_checks': steel_checks,
        'saddle_checks': saddle_checks, 'label_checks': label_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'reference_role': 'visual_match_unverified',
        'pending_review': report['pending_review'],
        'next_number': 11,
        'manual_only': True,
        'notes': ('The user six-view sheet is an uncalibrated visual-match reference; '
                  'the old model and items 01-09 were not modified. Candidate is '
                  'manual_only. Browser review page and visual acceptance are pending '
                  'unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '10-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
