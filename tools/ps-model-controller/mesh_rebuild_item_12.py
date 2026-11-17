"""Build the review-only 2 t trolley-jack candidate (item 12).

Shape follows the original clean/rust photograph for proportions and visible
features.  The user-supplied six-view sheet is an uncalibrated visual-match
reference of the same red 2 t hydraulic trolley jack.  The sheet is used for
hidden topology and as a diagnostic comparison, never as measured geometry,
and its panels disagree on the handle pose (upright in the front/back panels,
horizontal in the side panels); the candidate follows the photo.  The
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
REFERENCE_SHA256 = '5328071a4877ae53132194b1b5e92706b89af0f73182f0678f060c0dd75113ba'
SOURCE_SIZE = (1536, 1024)
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
def make_red_atlas(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('12 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((600, 650, 700, 760), (300, 560, 430, 690),
                (950, 600, 1050, 700), (1200, 650, 1300, 750))]
    rng = np.random.default_rng(21212)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [(profile['red']['clean'], 2.8),
             (profile['red']['weathered'], 3.4),
             (profile['red']['worn'], 2.2),
             (profile['red']['rusted'], 2.0)]
    rust_rgb = np.array((108, 74, 50), np.float32)
    dust_rgb = np.array((80, 75, 67), np.float32)
    scratch_rgb = np.array((203, 197, 189), np.float32)
    for i, (mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((110, 110), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((820, 820), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Stoved enamel orange-peel plus light stamped-steel texture.
        peel = cv2.GaussianBlur(rng.normal(size=(size // 4, size // 4)).astype(np.float32),
                                (0, 0), 1.2)
        peel = cv2.resize(peel, (size, size), interpolation=cv2.INTER_CUBIC)
        peel *= brush_strength / max(1e-6, float(peel.std()))
        field = .50 * (coarse - .5) + .30 * (fine - .5) + .20 * (speck - .5)
        amount = mat['rust_amount']
        patch = np.clip((coarse - (1 - amount * 1.8)) * 9, 0, 1)
        patch = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), 1.2)
        rust = patch * np.clip((fine - .08) * 1.25, 0, 1)
        rust = np.clip(rust + np.clip((speck - .997) * 90, 0, 1) * amount * 1.3, 0, 1)
        dust = np.clip((speck - .9982) * 45, 0, 1) * .035
        grime = np.clip((coarse - .92) * 10, 0, 1) * min(1.0, amount * 2.6)
        color = np.asarray(mat['base_rgb'], np.float32)
        rgb = color[None, None, :] * (1 + field[:, :, None] * .055)
        rgb += detail[:, :, None] * .070 + peel[:, :, None] * .52
        rgb += ((speck - .5) * .22)[:, :, None]
        rgb = rgb * (1 - grime[:, :, None] * .28) + \
            np.array((61, 60, 52), np.float32)[None, None, :] * (grime[:, :, None] * .28)
        scratches = np.clip((peel - 2.6) * .045, 0, 1) * (1 - rust) * (i == 2)
        rgb = rgb * (1 - scratches[:, :, None] * .28) + \
            scratch_rgb[None, None, :] * (scratches[:, :, None] * .28)
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
    rng = np.random.default_rng(21213)
    coarse = cv2.resize(rng.random((80, 80), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((900, 900), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    brush = cv2.resize(rng.normal(size=(180, size)).astype(np.float32),
                       (size, size), interpolation=cv2.INTER_CUBIC)
    brush = cv2.GaussianBlur(brush, (0, 0), 1.0)
    brush *= 4.0 / max(1e-6, float(brush.std()))
    base_rgb = np.asarray(profile['chrome']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + brush[:, :, None] * .014
                                          + (coarse[:, :, None] - .5) * .014)
    spots = np.clip((coarse - .978) * 30, 0, 1)
    spots = cv2.GaussianBlur(spots, (0, 0), 2.2)
    rgb = rgb * (1 - spots[:, :, None] * .32) + \
        np.array((122, 98, 76), np.float32)[None, None, :] * (spots[:, :, None] * .32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['chrome']['roughness'] + brush * .006
                    + (coarse - .5) * .016 + spots * .14, .08, .30)
    metallic = np.full((size, size), profile['chrome']['metallic'], np.float32)
    metallic *= (1 - spots * .35)
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


def make_plain_metal(prefix, out, profile_key, profile, seed):
    size = 1024
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.random((64, 64), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((size, size), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_NEAREST)
    base_rgb = np.asarray(profile[profile_key]['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (coarse[:, :, None] - .5) * .10
                                          + (fine[:, :, None] - .5) * .04)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile[profile_key]['roughness'] + (coarse - .5) * .12
                    + (fine - .5) * .04, .15, .60)
    metallic = np.full((size, size), profile[profile_key]['metallic'], np.float32)
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
    for name, arr in ((prefix + '_base', rgb), (prefix + '_mr', mr),
                      (prefix + '_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_black(out, profile):
    size = 1024
    rng = np.random.default_rng(21214)
    coarse = cv2.resize(rng.random((90, 90), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((900, 900), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    base_rgb = np.asarray(profile['black']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (coarse[:, :, None] - .5) * .35
                                          + (fine[:, :, None] - .5) * .12)
    # Paint chips and sparse rust on edges.
    chips = np.clip((coarse - .94) * 24, 0, 1)
    chips = cv2.GaussianBlur(chips, (0, 0), 1.0)
    rgb = rgb * (1 - chips[:, :, None] * .35) + \
        np.array((105, 74, 55), np.float32)[None, None, :] * (chips[:, :, None] * .35)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['black']['roughness'] + (coarse - .5) * .10
                    + chips * .20, .30, .72)
    metallic = np.full((size, size), profile['black']['metallic'], np.float32)
    metallic *= (1 - chips * .4)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur((fine - .5) * 1.1 + (coarse - .5) * .7, (0, 0), .55)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 14
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 14
    nx = np.clip(-dx * .4, -.04, .04)
    ny = np.clip(dy * .4, -.04, .04)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('black_base', rgb), ('black_mr', mr),
                      ('black_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_yellow(out, profile):
    size = 1024
    rng = np.random.default_rng(21215)
    coarse = cv2.resize(rng.random((80, 80), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((900, 900), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    base_rgb = np.asarray(profile['yellow']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (coarse[:, :, None] - .5) * .06
                                          + (fine[:, :, None] - .5) * .025)
    grime = np.clip((coarse - .90) * 8, 0, 1) * .24
    rgb = rgb * (1 - grime[:, :, None]) + \
        np.array((70, 62, 42), np.float32)[None, None, :] * grime[:, :, None]
    chips = np.clip((fine - .995) * 200, 0, 1) * .5
    rgb = rgb * (1 - chips[:, :, None]) + \
        np.array((64, 60, 54), np.float32)[None, None, :] * chips[:, :, None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['yellow']['roughness'] + (coarse - .5) * .08 + grime * .2,
                    .25, .62)
    metallic = np.full((size, size), profile['yellow']['metallic'], np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur((fine - .5) * .8 + (coarse - .5) * .5, (0, 0), .5)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 18
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 18
    nx = np.clip(-dx * .35, -.03, .03)
    ny = np.clip(dy * .35, -.03, .03)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('yellow_base', rgb), ('yellow_mr', mr),
                      ('yellow_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_rubber(out, profile):
    size = 1024
    rng = np.random.default_rng(21216)
    fine = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_NEAREST)
    coarse = cv2.resize(rng.random((90, 90), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    base_rgb = np.asarray(profile['rubber']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (fine[:, :, None] - .5) * .55
                                          + (coarse[:, :, None] - .5) * .30)
    dust = np.clip((coarse - .82) * 3.5, 0, 1) * .18
    rgb = rgb * (1 - dust[:, :, None]) + \
        np.array((72, 68, 61), np.float32)[None, None, :] * dust[:, :, None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['rubber']['roughness'] + (fine - .5) * .06, .65, .95)
    metallic = np.full((size, size), profile['rubber']['metallic'], np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur((fine - .5) * 1.4 + (coarse - .5) * .5, (0, 0), .5)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 9
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 9
    nx = np.clip(-dx * .6, -.10, .10)
    ny = np.clip(dy * .6, -.10, .10)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('rubber_base', rgb), ('rubber_mr', mr),
                      ('rubber_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_saddle(out, profile):
    size = 1024
    rng = np.random.default_rng(21217)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    period = 128.0
    lattice = (np.cos(TAU * (xx + yy) / period) +
               np.cos(TAU * (xx - yy) / period))
    height = np.clip(lattice + .35, 0, 1.7) / 1.7
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 1.2)
    base_rgb = np.asarray(profile['saddle']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (height[:, :, None] - .5) * .20)
    rgb += (rng.normal(size=(size, size, 1)) * 2.0).astype(np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['saddle']['roughness'] + (height - .5) * .10
                    + rng.normal(size=(size, size)).astype(np.float32) * .015,
                    .30, .62)
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


def make_ton_label(out):
    width = height = 1024
    image = Image.new('RGB', (width, height), (243, 199, 27))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(20, 20, 20), width=10)
    # Red diagonal hazard corners.
    for x0 in (-120, width - 300):
        for k in range(4):
            draw.polygon(((x0 + k * 70, 0), (x0 + k * 70 + 34, 0),
                          (x0 + k * 70 + 34 + 150, 150),
                          (x0 + k * 70 + 150, 150)), fill=(196, 32, 30))
    for x0 in (-60, width - 240):
        for k in range(3):
            draw.polygon(((x0 + k * 70, height), (x0 + k * 70 + 34, height),
                          (x0 + k * 70 + 34 - 150, height - 150),
                          (x0 + k * 70 - 150, height - 150)), fill=(196, 32, 30))
    draw.text((150, 60), '2 TON', font=ImageFont.truetype(LABEL_FONT_ARIAL, 150),
              fill=(18, 18, 18))
    draw.text((154, 230), 'HYDRAULIC TROLLEY JACK',
              font=ImageFont.truetype(LABEL_FONT_ARIAL, 44), fill=(26, 26, 26))
    draw.text((150, 300), '额定载荷 2 吨', font=ImageFont.truetype(LABEL_FONT_BOLD, 56),
              fill=(22, 22, 22))
    lines = ('使用前请阅读说明书', '严禁超载', '不要在松软地面使用',
             '顶升后请使用支撑架', '下降时缓慢松开回油阀')
    for i, text in enumerate(lines):
        draw.text((150, 390 + i * 54), '· ' + text,
                  font=ImageFont.truetype(LABEL_FONT, 36), fill=(30, 30, 30))
    draw.line((150, 680, width - 150, 680), fill=(40, 40, 40), width=4)
    draw.text((150, 702), 'MODEL: 2T-01', font=ImageFont.truetype(LABEL_FONT_ARIAL, 38),
              fill=(30, 30, 30))
    draw.text((150, 756), 'LIFT: 135-320 mm', font=ImageFont.truetype(LABEL_FONT_ARIAL, 38),
              fill=(30, 30, 30))
    draw.text((150, 810), '（标签按可辨文字重绘，非原印刷件）',
              font=ImageFont.truetype(LABEL_FONT, 29), fill=(78, 66, 34))
    base = np.asarray(image)
    mr = np.zeros((height, width, 3), np.uint8)
    mr[..., 0] = 255
    mr[..., 1] = int(.60 * 255)
    mr[..., 2] = 0
    normal = np.zeros((height, width, 3), np.uint8)
    normal[..., 0] = 128
    normal[..., 1] = 128
    normal[..., 2] = 255
    paths = {}
    for name, arr in (('ton_label_base', base), ('ton_label_mr', mr),
                      ('ton_label_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_warning_label(out):
    width, height = 2048, 512
    image = Image.new('RGB', (width, height), (240, 194, 25))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(24, 24, 24), width=8)
    for k in range(6):
        draw.polygon(((width - 260 + k * 46, 0), (width - 230 + k * 46, 0),
                      (width - 130 + k * 46, height), (width - 160 + k * 46, height)),
                     fill=(196, 34, 30))
    draw.text((70, 44), '⚠ WARNING', font=ImageFont.truetype(LABEL_FONT_ARIAL, 92),
              fill=(20, 20, 20))
    draw.text((70, 160), '千斤顶安全操作说明',
              font=ImageFont.truetype(LABEL_FONT_BOLD, 72), fill=(22, 22, 22))
    lines = ('额定载荷 2 TON · 严禁超载', '使用前检查焊缝、轮轴与液压系统',
             '请在坚硬平整地面使用', '顶升后必须使用安全支撑架',
             '人体任何部位不得进入顶升物下方')
    for i, text in enumerate(lines):
        draw.text((70, 260 + i * 46), '· ' + text,
                  font=ImageFont.truetype(LABEL_FONT, 34), fill=(32, 32, 32))
    base = np.asarray(image)
    mr = np.zeros((height, width, 3), np.uint8)
    mr[..., 0] = 255
    mr[..., 1] = int(.58 * 255)
    mr[..., 2] = 0
    normal = np.zeros((height, width, 3), np.uint8)
    normal[..., 0] = 128
    normal[..., 1] = 128
    normal[..., 2] = 255
    paths = {}
    for name, arr in (('warning_label_base', base), ('warning_label_mr', mr),
                      ('warning_label_normal', normal)):
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
    values['note'] = ('Tile means; rust is detected by brown colour. The jack is '
                      'painted stamped steel, so metallic stays low; highlights '
                      'come from live lighting, not from baked colour.')
    return values


def metallic_texture_checks(base_path, mr_path, *, min_metallic, max_roughness,
                            max_chroma, min_luma):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    means = base.mean(axis=(0, 1))
    values = {'base_rgb_mean': [round(float(v), 1) for v in means],
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'chroma': round(float(means.max() - means.min()), 2)}
    values['passed'] = bool(means.min() > min_luma and values['chroma'] < max_chroma and
                            values['metallic_mean'] > min_metallic and
                            values['roughness_mean'] < max_roughness)
    return values


def black_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    means = base.mean(axis=(0, 1))
    values = {'base_rgb_mean': [round(float(v), 1) for v in means],
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'chroma': round(float(means.max() - means.min()), 2)}
    values['passed'] = bool(means.max() < 70 and values['chroma'] < 28 and
                            .20 < values['metallic_mean'] < .55 and
                            .35 < values['roughness_mean'] < .68)
    values['note'] = 'Black painted/cast jack parts; paint chips and rust specks are synthesised.'
    return values


def yellow_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    means = base.mean(axis=(0, 1))
    values = {'base_rgb_mean': [round(float(v), 1) for v in means],
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'yellow_dominance': round(float(means[0] + means[1] - 2 * means[2]), 1)}
    values['passed'] = bool(means[0] > 170 and means[1] > 110 and means[2] < 90 and
                            values['yellow_dominance'] > 150 and
                            values['metallic_mean'] < .08 and
                            .25 < values['roughness_mean'] < .65)
    values['note'] = 'Yellow enamel carry loop; colour and wear are inferred.'
    return values


def rubber_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    means = base.mean(axis=(0, 1))
    values = {'base_rgb_mean': [round(float(v), 1) for v in means],
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4)}
    values['passed'] = bool(means.max() < 50 and values['metallic_mean'] < .05 and
                            values['roughness_mean'] > .70)
    values['note'] = 'Rubber tyres and handle grip; tread pattern is not visible and not modelled.'
    return values


def saddle_texture_checks(base_path, mr_path, normal_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    normal = np.asarray(Image.open(normal_path).convert('RGB'), dtype=np.float32)
    means = base.mean(axis=(0, 1))
    normal_dev = float(np.abs(normal[..., :2] - 128).mean())
    values = {'base_rgb_mean': [round(float(v), 1) for v in means],
              'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'normal_xy_deviation': round(normal_dev, 3)}
    values['passed'] = bool(means.max() < 90 and values['metallic_mean'] > .35 and
                            values['normal_xy_deviation'] > 8.0)
    values['note'] = 'Checker knurl normal map only; pitch is not measured.'
    return values


def label_checks(base_path, *, min_yellow, min_dark):
    rgb = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    yellow = ((rgb[..., 0] > 170) & (rgb[..., 1] > 120) & (rgb[..., 2] < 100)).mean()
    red = ((rgb[..., 0] > 150) & (rgb[..., 1] < 100) & (rgb[..., 2] < 100)).mean()
    dark = (rgb.max(axis=2) < 90).mean()
    values = {'yellow_fraction': round(float(yellow), 4),
              'red_fraction': round(float(red), 4),
              'dark_fraction': round(float(dark), 4),
              'size': [int(rgb.shape[1]), int(rgb.shape[0])]}
    values['passed'] = bool(yellow > min_yellow and dark > min_dark)
    values['note'] = ('Label artwork is recreated from legible photo text; the '
                      'original graphic, small print and certification marks are '
                      'not reproduced.')
    return values


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 10, y0 + 8, x0 + w // 3 - 10, y0 + h // 2 - 8))


def reference_mask_12(panel):
    """Approximate foreground mask for diagnostics only."""
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = rgb.mean(axis=2)
    mask = (gray < 230).astype(np.uint8)
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
    'front': ('参考正视从顶升头/鞍座端看；参考把手柄画成竖立，原照与左右视面板显示'
              '水平操作位置，候选按照片建模，该姿态为推断。'),
    'back': ('背面在原照不可见；参考背视给出泵盖、手柄插座与后脚轮轮廓；'
             '插座内部锁定机构、泵内油路与隐藏面为推断。'),
    'left': ('左视参考与原照同侧，给出大片侧板、前后轮、黄色提环、泵盖和水平手柄；'
             '本候选取该姿态为主参考。'),
    'right': ('右视为镜像推断：侧板、泵盖、手柄插座、后脚轮和黄色提环的背面。'),
    'top': ('顶视参考给出四轮位置、鞍座圆盘、黄色提环、2 TON 泵盖标签与手柄方位；'
            '轮距、轮径和手柄伸出长度为按比例推断。'),
    'bottom': ('底视参考给出红色底板、贯通前轴、后横杆与后脚轮；'
               '底面焊缝、加强筋和脚轮内部为推断。'),
}


def make_report_12(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '12 千斤顶2 · 2 吨卧式液压千斤顶 · 六视图参考（未标定）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_12(panel))
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
        ref_mask = reference_mask_12(panel)
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
              '重要：参考图未标定且各面板不是同一转台；图中分数不是实测精度。正视/背视把'
              '手柄画成竖立，左右视和原照显示为水平操作位置，候选以照片为准。原模型虽通过'
              '单张源图轮廓门槛，但六方向审计显示表面糊化、车轮/手柄/泵体粘连、标签不可辨，'
              '故未复用旧网格。',
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
        ('detail-saddle', '鞍座圆盘与顶升头（滚花为贴图，齿数为推断）'),
        ('detail-saddle-top', '鞍座顶面凹盘'),
        ('detail-arm-label', '顶升臂侧面警示标签（重绘）'),
        ('detail-carry-loop', '黄色 U 形提环'),
        ('detail-loop-mounts', '提环侧板安装座与螺栓'),
        ('detail-pump-cover', '黑色泵盖与两侧 2 TON 标签'),
        ('detail-pump-label', '泵盖标签（文字重绘，非原印刷件）'),
        ('detail-handle-socket', '手柄插座、侧耳与螺栓'),
        ('detail-handle-grip', '红色手柄管与黑色橡胶握把'),
        ('detail-front-wheel', '前大轮、轮毂通孔与贯穿轴'),
        ('detail-rear-caster', '后脚轮、叉架、转向轴与红色安装耳'),
        ('detail-frame-side', '红色侧板、压筋与前部螺栓'),
        ('detail-arm-pivot', '顶升臂后端枢轴套与钢销'),
        ('detail-arm-link', '黑色连杆板与上下销'),
        ('detail-base-bottom', '底面：底板、前轴、后横杆与脚轮'),
        ('detail-rear-frame', '后部车架、泵盖与插座连接'),
        ('detail-surface', '红色漆面、磨损与锈斑'),
        ('detail-front', '整体前视（透明背景）'),
        ('new-front', '六方向正视（透明背景）'),
    )
    board = Image.new('RGB', (2000, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '12 千斤顶2 · 关键局部检查（新候选）',
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
              '这些局部图用于人工检查结构；不能证明未观察面、尺寸、装配关系、孔深、'
              '焊缝内部或液压油路。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 16)
    board = Image.new('RGB', (1800, 900), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '12 千斤顶2 · 隔离开口与结构检查',
              font=font, fill='#18242b')
    items = (('loop-only', '黄色 U 形提环开口（非封闭孔）', 'carry_loop'),
             ('hub-bores-only', '前轮毂通孔（同轴重叠投影）', 'front_hub_bore'),
             ('socket-bore-only', '手柄插座内孔', 'socket_bore'),
             ('frame-only', '红色车架与顶升臂（外观）', None))
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
            if key == 'carry_loop':
                text = f'{label}：{status}（拱下通透 {check["inner_clear"]}）'
            else:
                text = f'{label}：{check["detected_openings"]}/{check["expected"]} {status}'
            color = '#1d5c3a' if check['passed'] else '#8a2b20'
        else:
            text = f'{label}（不参与孔数判定）'
            color = '#52646b'
        draw.text((x + 10, 48), text, font=small, fill=color)
    draw.text((20, 805),
              '透明背景隔离渲染；提环为下端开口的 U 形杆（非封闭孔），前轮毂孔和'
              '手柄插座孔是真实透明通孔。孔数不能证明孔深、壁厚、内部锁定机构、'
              '液压油路或装配关系。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '12 千斤顶2 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选红漆、黑漆、橡胶与镀铬 PBR'))):
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
    draw.text((20, 12), '12 千斤顶2 · 变光检查（同视角）',
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
    draw.text((20, 12), '12 千斤顶2 · 金属反射变光对照（红色漆面 / 镀铬轴）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光位置 A'),
                                       ('new-metal-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 760))
        tile = Image.new('RGBA', (680, 780), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (780 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 55))
        draw.text((x + 24, 28), label, font=small, fill='#27363c')
    draw.text((20, 820), '同一相机、不同主光方向；滚花瓣、镀铬轴和螺栓高光必须移动。'
                         '漆面金属度低，主要反射来自清漆层与实时灯光。',
              font=small, fill='#52646b')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    tile = 250
    board = Image.new('RGB', (tile * 6, tile * 4 + 34), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '12 千斤顶2 · 24 方向转台（灰色为透明背景上的中性光渲染）',
              font=font, fill='#27363c')
    for i in range(24):
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png').convert('RGBA')
        image.thumbnail((tile - 16, tile - 16), Image.Resampling.LANCZOS)
        cell = Image.new('RGBA', (tile, tile), '#ffffff')
        cell.alpha_composite(image, ((tile - image.width) // 2,
                                     (tile - image.height) // 2))
        board.paste(cell.convert('RGB'), ((i % 6) * tile, 34 + (i // 6) * tile))
    board.save(out / '转台-24方向.jpg', quality=94)


def audit_row(number):
    path = AUDIT
    if not path.is_file():
        return None
    for line in path.read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('number') == number:
                return row
    return None


def reference_ratio_diagnostics(reference):
    image = Image.open(reference).convert('RGB')
    results = []
    for i, name in enumerate(VIEWS):
        panel = panel_image(image, i)
        mask = reference_mask_12(panel)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        results.append({'view': name, 'panel_width': panel.width,
                        'panel_height': panel.height,
                        'foreground_bbox': [int(xs.min()), int(ys.min()),
                                            int(xs.max()), int(ys.max())],
                        'foreground_fraction': round(float(mask.mean()), 4)})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=12)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 12:
        p.error('this entry supports item 12 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/12-jack-draft1' if a.draft
                        else 'outputs/12千斤顶2-六视图重建-v1')
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
    config_path = HERE / 'jack12_config.json'
    worker_path = HERE / 'mesh_rebuild_12_worker.py'
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
                        a.out / 'black_base.png', a.out / 'black_mr.png',
                        a.out / 'black_normal.png', a.out / 'yellow_base.png',
                        a.out / 'yellow_mr.png', a.out / 'yellow_normal.png',
                        a.out / 'rubber_base.png', a.out / 'rubber_mr.png',
                        a.out / 'rubber_normal.png', a.out / 'saddle_base.png',
                        a.out / 'saddle_mr.png', a.out / 'saddle_normal.png',
                        a.out / 'ton_label_base.png', a.out / 'ton_label_mr.png',
                        a.out / 'ton_label_normal.png', a.out / 'warning_label_base.png',
                        a.out / 'warning_label_mr.png', a.out / 'warning_label_normal.png',
                        a.out / '六视图-新旧对照.jpg', a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '转台-24方向.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'detail-{name}.png' for name in
                             ('saddle', 'saddle-top', 'arm-label', 'carry-loop',
                              'loop-mounts', 'pump-cover', 'pump-label',
                              'handle-socket', 'handle-grip', 'front-wheel',
                              'rear-caster', 'frame-side', 'arm-pivot', 'arm-link',
                              'base-bottom', 'rear-frame', 'surface', 'front')]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('loop-only', 'hub-bores-only', 'socket-bore-only',
                              'frame-only', 'new-light-alternate',
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
    textures.update(make_plain_metal('steel', a.out, 'steel',
                                     config['material_profile'], 21218))
    textures.update(make_plain_metal('gold', a.out, 'gold',
                                     config['material_profile'], 21219))
    textures.update(make_black(a.out, config['material_profile']))
    textures.update(make_yellow(a.out, config['material_profile']))
    textures.update(make_rubber(a.out, config['material_profile']))
    textures.update(make_saddle(a.out, config['material_profile']))
    textures.update(make_ton_label(a.out))
    textures.update(make_warning_label(a.out))
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
    views = make_report_12(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(12)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    aligned_fidelity = None
    aligned_best = None
    if not a.draft:
        make_turntable(a.out)
        # Best-effort silhouette diagnostic over all 24 turntable views,
        # aligned by bounding box. The photo is a three-quarter view, so the
        # strict source-camera score is not a fair pose comparison. This is
        # still an uncalibrated diagnostic, never an acceptance gate.
        with Image.open(alpha_path) as source_alpha_image:
            source_fit_mask = mask_fit(np.asarray(source_alpha_image.convert('L')) >= 128)
        best = None
        for i in range(24):
            with Image.open(a.out / 'renders' / f'turn-{i:02d}.png') as turn:
                turn_fit_mask = mask_fit(np.asarray(turn.getchannel('A')) >= 128)
            if source_fit_mask is None or turn_fit_mask is None:
                continue
            score = source_fidelity(source_fit_mask * 255, turn_fit_mask * 255)
            if best is None or score['iou'] > best[1]['iou']:
                best = (i, score)
        if best is not None:
            aligned_best = best[0]
            aligned_fidelity = best[1]
            aligned_fidelity['view'] = f'{best[0] * 15}-degree turntable view'
            aligned_fidelity['alignment'] = 'mask bounding box fit to 400 px; not a calibrated camera'
            aligned_fidelity['acceptance_gate'] = False
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('front_hub_bore', 'hub-bores-only.png', 1),
                ('socket_bore', 'socket-bore-only.png', 1)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        # The carry handle is a U-shaped bar attached at both ends; it is open
        # at the bottom, so an enclosed hole is not expected. Check the alpha
        # clearance under the arch and the presence of the arch and both legs.
        with Image.open(a.out / 'renders' / 'loop-only.png') as im:
            alpha = np.asarray(im.getchannel('A'))
        solid = alpha >= 128
        ys, xs = np.nonzero(solid)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cx = (x0 + x1) // 2
        w, h = x1 - x0, y1 - y0
        inner_clear = not bool(solid[int(y0 + .72 * h), cx])
        top_bar = bool(solid[int(y0 + .05 * h), cx])
        left_leg = bool(solid[int(y0 + .70 * h), int(x0 + .08 * w)])
        right_leg = bool(solid[int(y0 + .70 * h), int(x1 - .08 * w)])
        _, count = interior_holes(alpha)
        hole_checks['carry_loop'] = {
            'detected_openings': int(count), 'expected': 0,
            'passed': bool(inner_clear and top_bar and left_leg and right_leg),
            'method': 'alpha clearance through the open U-shaped handle',
            'inner_clear': inner_clear, 'top_bar': top_bar,
            'left_leg': left_leg, 'right_leg': right_leg,
            'note': ('The yellow carry handle is a U-shaped bar attached at both '
                     'ends and open at the bottom, so an enclosed hole is not '
                     'expected; the arch clearance and both legs are checked.'),
            'file': str(a.out / 'renders' / 'loop-only.png')}
        make_isolated_board(a.out, hole_checks)
    red_checks = red_texture_checks(a.out / 'red_base.png', a.out / 'red_mr.png')
    chrome_checks = metallic_texture_checks(
        a.out / 'chrome_base.png', a.out / 'chrome_mr.png',
        min_metallic=.90, max_roughness=.30, max_chroma=18, min_luma=150)
    chrome_checks['note'] = 'Polished chrome wheel axle; highlights come from live lighting.'
    steel_checks = metallic_texture_checks(
        a.out / 'steel_base.png', a.out / 'steel_mr.png',
        min_metallic=.80, max_roughness=.50, max_chroma=18, min_luma=90)
    steel_checks['note'] = 'Plated pins, bolts and caster stems; no measured fastener data.'
    gold_checks = metallic_texture_checks(
        a.out / 'gold_base.png', a.out / 'gold_mr.png',
        min_metallic=.80, max_roughness=.45, max_chroma=120, min_luma=60)
    gold_checks['note'] = 'Zinc/gold-plated nuts and fittings; colour is inferred.'
    black_checks = black_texture_checks(a.out / 'black_base.png', a.out / 'black_mr.png')
    yellow_checks = yellow_texture_checks(a.out / 'yellow_base.png', a.out / 'yellow_mr.png')
    rubber_checks = rubber_texture_checks(a.out / 'rubber_base.png', a.out / 'rubber_mr.png')
    saddle_checks = saddle_texture_checks(a.out / 'saddle_base.png',
                                          a.out / 'saddle_mr.png',
                                          a.out / 'saddle_normal.png')
    ton_checks = label_checks(a.out / 'ton_label_base.png', min_yellow=.20, min_dark=.05)
    warning_checks = label_checks(a.out / 'warning_label_base.png',
                                  min_yellow=.30, min_dark=.02)
    names = [part['name'] for part in worker['parts']]
    head = config['head']
    structural = {
        'frame_side_plates': sum(name.startswith('Frame left plate') or
                                 name.startswith('Frame right plate') for name in names),
        'frame_floor': 'Frame floor' in names,
        'frame_pump_crossmember': 'Frame pump crossmember' in names,
        'frame_ribs': sum(name.startswith('Frame left rib') or
                          name.startswith('Frame right rib') for name in names),
        'front_wheels': sum(name.startswith('Front wheel') for name in names),
        'front_hubs': int('Front hub left' in names) + int('Front hub right' in names),
        'front_hub_nuts': sum(name.startswith('Front hub nut') for name in names),
        'front_axle': 'Front axle' in names,
        'rear_wheels': sum(name.startswith('Rear wheel') for name in names),
        'rear_fork_plates': sum(name.startswith('Rear fork plate') for name in names),
        'rear_castors': sum(name.startswith('Caster stem') for name in names),
        'caster_ears': sum(name.startswith('Caster ear') for name in names),
        'caster_top_bolts': sum(name.startswith('Caster top bolt') for name in names),
        'rear_cross_rod': 'Rear cross rod' in names,
        'lifting_arm': 'Lifting arm' in names,
        'arm_pivot_boss': 'Arm pivot boss' in names,
        'arm_pivot_pin': 'Arm pivot pin' in names,
        'lifting_head': 'Lifting head' in names,
        'saddle_post': 'Saddle post' in names,
        'head_pin': 'Head pin' in names,
        'head_cotter_pin': 'Head cotter pin' in names,
        'link_plates': sum(name.startswith('Link plate') for name in names),
        'link_pins': sum(name.startswith('Link pin') for name in names),
        'saddle_pad': 'Saddle pad' in names,
        'saddle_rim': 'Saddle rim' in names,
        'pump_cover': 'Pump cover' in names,
        'pump_side_labels': sum(name.startswith('Pump side label') for name in names),
        'pump_top_label': 'Pump top label' in names,
        'pump_bolts': sum(name.startswith('Pump bolt') for name in names),
        'pump_filler_cap': 'Pump filler cap' in names,
        'handle_socket': 'Handle socket' in names,
        'socket_ears': sum(name.startswith('Socket ear') for name in names),
        'socket_bolts': sum(name.startswith('Socket bolt') for name in names),
        'socket_bracket': 'Socket bracket' in names,
        'handle_pipe': 'Handle pipe' in names,
        'handle_grip': 'Handle grip' in names,
        'grip_end_cap': 'Grip end cap' in names,
        'carry_loop': 'Carry loop' in names,
        'loop_bushes': sum(name.startswith('Loop bush') for name in names),
        'loop_bolts': sum(name.startswith('Loop bolt') for name in names),
        'front_rating_label': 'Front rating label' in names,
        'warning_labels': sum(name.startswith('Warning label') for name in names),
        'note': ('Part presence proves only that a feature was modelled, not its '
                 'size, fit or hidden faces. Hidden pump internals, the socket '
                 'latch, weld seams, tyre tread and caster swivel details are '
                 'inferred.'),
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
        structural['frame_side_plates'] == 2 and structural['frame_floor'] and
        structural['frame_pump_crossmember'] and
        structural['frame_ribs'] == 2 and
        structural['front_wheels'] == 2 and structural['front_hubs'] == 2 and
        structural['front_hub_nuts'] == 2 and structural['front_axle'] and
        structural['rear_wheels'] == 2 and structural['rear_fork_plates'] == 4 and
        structural['rear_castors'] == 2 and structural['caster_ears'] == 2 and
        structural['caster_top_bolts'] == 2 and structural['rear_cross_rod'] and
        structural['lifting_arm'] and structural['arm_pivot_boss'] and
        structural['arm_pivot_pin'] and structural['lifting_head'] and
        structural['saddle_post'] and structural['head_pin'] and
        structural['head_cotter_pin'] and structural['link_plates'] == 2 and
        structural['link_pins'] == 4 and structural['saddle_pad'] and
        structural['saddle_rim'] and structural['pump_cover'] and
        structural['pump_side_labels'] == 2 and structural['pump_top_label'] and
        structural['pump_bolts'] == 3 and structural['pump_filler_cap'] and
        structural['handle_socket'] and structural['socket_ears'] == 2 and
        structural['socket_bolts'] == 2 and structural['socket_bracket'] and
        structural['handle_pipe'] and structural['handle_grip'] and
        structural['grip_end_cap'] and structural['carry_loop'] and
        structural['loop_bushes'] == 2 and structural['loop_bolts'] == 2 and
        structural['front_rating_label'] and structural['warning_labels'] == 2)
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 55 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': structural_ok,
        'material_tiles': bool(red_checks['passed'] and chrome_checks['passed'] and
                               steel_checks['passed'] and gold_checks['passed'] and
                               black_checks['passed'] and yellow_checks['passed'] and
                               rubber_checks['passed'] and saddle_checks['passed'] and
                               ton_checks['passed'] and warning_checks['passed']),
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
        'version': config['version'], 'number': 12, 'asset_id': row['asset_id'],
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
            'The old model-library source IoU 0.949 only measures one silhouette. '
            'The six-view audit shows a smoothed, fused surface: wheels, handle '
            'and pump merge into the body, the 2 TON label is unreadable and the '
            'saddle knurl is lost. The old mesh is therefore not reused for the '
            'candidate base geometry.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'red_texture_checks': red_checks,
        'chrome_texture_checks': chrome_checks,
        'steel_texture_checks': steel_checks,
        'gold_texture_checks': gold_checks,
        'black_texture_checks': black_checks,
        'yellow_texture_checks': yellow_checks,
        'rubber_texture_checks': rubber_checks,
        'saddle_texture_checks': saddle_checks,
        'ton_label_checks': ton_checks,
        'warning_label_checks': warning_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'source_fidelity_aligned_turn_index': aligned_best,
        'source_diagnostic_note': (
            'The supplied clean photo is a three-quarter product view with the '
            'saddle at the left, while the strict source-camera render looks along '
            '+X (from the handle end). The strict score is therefore not a fair pose '
            'comparison. A second best-effort diagnostic scans all 24 turntable '
            'views and aligns each silhouette to the source alpha by bounding box; '
            'it is uncalibrated and is not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is an uncalibrated visual-match '
                       'reference, not a calibrated image set; its handle pose '
                       'differs between panels. It is used only for hidden '
                       'topology and gross differences.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view sheet is uncalibrated and not a measured image set; the reference segmentation is a diagnostic.',
            'The reference panels are not a consistent turntable: the handle is drawn upright in the front/back panels and horizontal in the left/right panels; the candidate follows the photo.',
            'All sizes follow photo/reference pixel ratios and a plausible 2-ton trolley-jack size; they are not measured dimensions.',
            'The 2 TON rating is legible but the small print, model number, lift range and certification marks are recreated or omitted.',
            'The handle length beyond the photo crop, the black grip position and the handle wall thickness are inferred.',
            'The pump internals, release valve, socket latch and hydraulic passages are hidden and are not modelled as functional flow passages.',
            'Weld seams, stamping details, side-plate inner faces and frame cross-members are partly inferred.',
            'Tyre tread, wheel bearing details and caster swivel stops are not visible and not modelled.',
            'The saddle knurl is a texture/normal map; the tooth count and dish depth are inferred.',
            'The yellow carry-loop tube diameter, forward lean and mounting bolts are inferred.',
            'Paint wear, rust patches and black paint chips are synthesised from small observed patches, not photographed hidden surfaces.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the handle pose, carry loop, pump cover and socket.',
            'Confirm the handle length, grip position, wall thickness and whether the real handle is a single or two-piece tube.',
            'Confirm the pump/valve cover shape, label artwork and bolt layout from a clearer photo or sample.',
            'Confirm the handle-socket casting, latch and side bolts.',
            'Confirm wheel diameters, wheelbase, track width, caster offset and tyre construction with measurements.',
            'Confirm the lifting arm section, pivot positions and link-plate geometry.',
            'Confirm the saddle knurl/tooth count, dish depth and saddle-post attachment.',
            'Confirm the carry-loop tube diameter, lean angle and mounting hardware.',
            'Check the red/black/yellow paint shades and the label text against a higher-resolution sample.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 13,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_12.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 12, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed,
        'red_checks': red_checks,
        'chrome_checks': chrome_checks, 'steel_checks': steel_checks,
        'gold_checks': gold_checks, 'black_checks': black_checks,
        'yellow_checks': yellow_checks, 'rubber_checks': rubber_checks,
        'saddle_checks': saddle_checks, 'ton_label_checks': ton_checks,
        'warning_label_checks': warning_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'reference_role': 'visual_match_unverified',
        'pending_review': report['pending_review'],
        'next_number': 13,
        'manual_only': True,
        'notes': ('The user six-view sheet is an uncalibrated visual-match reference '
                  'whose panels disagree on the handle pose; the photo is the primary '
                  'shape evidence. The old model and items 01-11 were not modified. '
                  'Candidate is manual_only. Browser review page and visual '
                  'acceptance are pending unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '12-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
