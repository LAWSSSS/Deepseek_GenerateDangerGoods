"""Build the review-only part-based candidate for item 30 (hydraulic cylinder).

The photograph shows a black-painted HCIC hydraulic cylinder with a chrome
piston rod, a forged rod-end eye containing a steel bore bushing, an auxiliary
single-shear lug under the eye, a welded base clevis with a steel bushing, two
hex port fittings at the base end and a welded elbow port near the rod gland.
The supplied six-view sheet is a category reference for a *different* black
cylinder with a yellow label; it is uncalibrated and is not used as a
measurement.  The photograph wins wherever the reference disagrees.

The old TripoSR shell fuses the eye, lugs, ports and rod and has no true through
bores.  It is retained unchanged for comparison; the candidate is a new
part-based mesh with real openings, separate metal and paint materials and PBR
textures.  Everything stays manual_only/pending review and never replaces the
library mesh or enters automatic placement or the formal composite.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from mesh_rebuild_item import mask_fit
from mesh_checks import source_fidelity
from scene_v2 import digest

NUMBER = 30
VERSION = 'cylinder30-parametric-v1'
REFERENCE_NAME = 'ChatGPT 图像 2026年10月14日 10_10_23-1.png'
REFERENCE_SHA256 = 'e6b0ea89ed28ad22ebaac5058f7796b9af2717139789893a10258ea57467b9f4'
SOURCE_SIZE = (1254, 1254)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
FONT = 'C:/Windows/Fonts/msyh.ttc'
FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'

DIMENSIONS = {
    'barrel': {'radius': 0.060, 'z0': -0.275, 'z1': 0.275},
    'dome': {'z0': -0.320, 'z1': -0.275, 'radius': 0.060},
    'base_lug': {'x_half': 0.013, 'half_width': 0.042, 'z_top': -0.235,
                 'z_center': -0.388, 'z_bottom': -0.430,
                 'bore_radius': 0.0185, 'bush_outer': 0.0260,
                 'bush_x_half': 0.0145},
    'head': {'collar_radius': 0.065, 'collar_z0': 0.266, 'collar_z1': 0.303,
             'gland_radius': 0.053, 'gland_z0': 0.303, 'gland_z1': 0.322,
             'wiper_radius': 0.050, 'wiper_z0': 0.322, 'wiper_z1': 0.326},
    'rod': {'radius': 0.019, 'z0': 0.305, 'z1': 0.438,
            'step_radius': 0.024, 'step_z0': 0.438, 'step_z1': 0.459},
    'eye': {'y': 0.004, 'z': 0.565, 'outer_radius': 0.070,
            'bore_radius': 0.0335, 'x_half': 0.024,
            'bush_outer': 0.047, 'bush_x_half': 0.026},
    'eye_socket': {'z0': 0.452, 'z1': 0.488, 'r0': 0.030, 'r1': 0.036},
    'eye_neck': {'z0': 0.485, 'z1': 0.523, 'r0': 0.034, 'r1': 0.043},
    'small_clevis': {'y': -0.054, 'z': 0.492, 'outer_radius': 0.029,
                     'bore_radius': 0.0135, 'x_half': 0.012,
                     'bush_outer': 0.0205, 'bush_x_half': 0.014,
                     'boss_r': 0.022, 'boss_y0': -0.018},
    'base_ports': {'azimuth_deg': 70, 'z_a': -0.205, 'z_b': -0.245,
                   'hex_r': 0.0125, 'hex_len': 0.018,
                   'nipple_r': 0.0080, 'nipple_len': 0.014,
                   'pad_len': 0.044, 'pad_width': 0.038},
    'elbow_port': {'azimuth_deg': 72, 'z': 0.205, 'hex_r': 0.0135,
                   'hex_len': 0.018, 'tube_r': 0.0068,
                   'nipple_r': 0.0085, 'nipple_len': 0.016},
    'logo': {'z_center': -0.03, 'length': 0.27, 'arc_width': 0.058,
             'radius': 0.0604},
    'overall': {'base_bottom': -0.430, 'eye_top': 0.635},
}
RENDER = {
    'six_resolution': 512, 'six_samples': 48,
    'detail_resolution': 512, 'detail_samples': 40,
    'turn_resolution': 320, 'turn_samples': 11, 'turn_count': 24,
    'reference_role': 'category_reference_uncalibrated',
}


# ---------------------------------------------------------------------------
# Inputs and paths
# ---------------------------------------------------------------------------
def local_paths():
    return {
        'original': ROOT / 'outputs/物品高清输入-39件/原貌HD/30-液压缸.png',
        'rust': ROOT / 'outputs/物品高清输入-39件/生锈HD/30-液压缸.png',
        'reference': ROOT / 'references/parallel-six-view' / REFERENCE_NAME,
        'old_mesh': ROOT / ('outputs/三维模型库-圆润复验/items/'
                            '30-asset-cd0a8343411b/mesh.glb'),
        'source_view': ROOT / ('outputs/三维模型库-圆润复验/items/'
                               '30-asset-cd0a8343411b/source-view.png'),
        'source_alpha': ROOT / ('outputs/三维模型库-圆润复验/items/'
                                '30-asset-cd0a8343411b/source-alpha.png'),
        'audit': ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl',
    }


def verify_inputs(paths):
    expected = {
        'original': 'e7958c627361733e49a8308fe954e2a6aef10ef32b91c8e8d73676ef3665d70c',
        'rust': '99b0730ccf9cd8a47bf5ffda28603fc6f59803ee9b1491c3f9a316dedc3bce51',
        'reference': REFERENCE_SHA256,
        'old_mesh': 'c84d6f0bda80dc91f5720082e6cd49edd2f5e066db98082a9ee3d34c5b4adf10',
        'source_view': 'f3d133659075758a6becc1ecd34d910ee1aafc0425e413c984f86db5f9ae9f6b',
        'source_alpha': '85a6cabe8476aa5e503d03ff3e89417175e8ebb6b3e2766a1274d346a60577af',
    }
    for key, value in expected.items():
        if digest(paths[key]) != value:
            raise ValueError(f'{key} hash differs from the registered value')
    if Image.open(paths['original']).size != SOURCE_SIZE:
        raise ValueError('original image size changed')


# ---------------------------------------------------------------------------
# Procedural PBR textures
# ---------------------------------------------------------------------------
def periodic_noise(h, w, scale, seed):
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(h, w)).astype(np.float32)
    spectrum = np.fft.fft2(white)
    fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
    fx = np.fft.fftfreq(w).astype(np.float32)[None, :]
    sigma = 1.0 / max(2.0, scale)
    kernel = np.exp(-(fx * fx + fy * fy) / (2 * sigma * sigma))
    result = np.real(np.fft.ifft2(spectrum * kernel)).astype(np.float32)
    result -= result.mean()
    result /= max(1e-6, result.std())
    return result


def fbm(h, w, seed, scales, weights):
    acc = np.zeros((h, w), np.float32)
    for i, (scale, weight) in enumerate(zip(scales, weights)):
        acc += weight * periodic_noise(h, w, scale, seed + 97 * i)
    acc -= acc.mean()
    acc /= max(1e-6, acc.std())
    return acc


def smoothstep(x, a, b):
    t = np.clip((np.asarray(x, np.float32) - a) /
                np.maximum(1e-8, np.asarray(b, np.float32) - a), 0, 1)
    return t * t * (3 - 2 * t)


def normal_from_height(height, strength=1.0):
    height = cv2.GaussianBlur(np.asarray(height, np.float32), (0, 0), .8)
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack((nx / length, ny / length, nz / length), axis=-1)
    return np.clip((out * .5 + .5) * 255, 0, 255).astype(np.uint8)


def pack_mr(roughness, metallic):
    out = np.zeros(roughness.shape + (3,), np.uint8)
    out[..., 0] = 255
    out[..., 1] = np.clip(roughness * 255, 0, 255)
    out[..., 2] = np.clip(metallic * 255, 0, 255)
    return out


def save_rgb(path, array):
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), 'RGB').save(path)


def vertical_scratches(h, w, seed, count=260):
    rng = np.random.default_rng(seed)
    canvas = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for _ in range(count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(8, 190))
        drift = int(rng.integers(-3, 4))
        value = int(rng.integers(55, 165))
        draw.line((x, y, x + drift, y + length), fill=value, width=1)
    return np.asarray(canvas, np.float32) / 255.0


def make_paint_worn(out):
    h = w = 1024
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    # Image row 0 is texture v=1 in Blender, so build the rust gradient in
    # texture coordinates and place it at the image bottom (barrel base).
    v = 1 - np.linspace(0, 1, h, endpoint=False)[:, None]
    n_broad = fbm(h, w, 3101, (240, 78), (.7, .3))
    n_mid = periodic_noise(h, w, 105, 3102)
    n_fine = periodic_noise(h, w, 28, 3103)
    n_micro = periodic_noise(h, w, 6, 3104)
    base_weight = .30 + .70 * np.exp(-((v - .015) / .22) ** 2)
    port_zone = .62 * np.exp(-((u - .25) ** 2) / (2 * .065 ** 2)) * (
        np.exp(-((v - .10) / .070) ** 2) +
        .9 * np.exp(-((v - .875) / .055) ** 2))
    field = .66 * n_mid + .36 * n_fine + .22 * n_broad
    threshold = 1.18 - .60 * base_weight - .44 * port_zone
    rust = smoothstep(field, threshold, threshold + .85)
    rust *= .22 + .78 * smoothstep(n_broad, -1.2, .8)
    rust *= smoothstep(n_fine, -1.5, .3)
    rust = np.clip(rust, 0, 1)
    rust_bin = (rust > .58).astype(np.uint8)
    edge = cv2.dilate(rust_bin, np.ones((5, 5), np.uint8)) - rust_bin
    chip = (edge > 0) & (n_fine > 1.0) & (n_micro > .0)
    scratch = vertical_scratches(h, w, 3105, 40)
    paint = np.dstack([26 + 4 * n_broad + 1.4 * n_fine,
                       28 + 4 * n_broad + 1.4 * n_fine,
                       31 + 4 * n_broad + 1.4 * n_fine])
    rust_col = np.dstack([84 + 27 * n_mid + 10 * n_fine,
                          49 + 16 * n_mid + 7 * n_fine,
                          29 + 10 * n_mid + 5 * n_fine])
    mix = (rust * .95)[..., None]
    img = paint * (1 - mix) + rust_col * mix
    chip_col = np.dstack([112 + 16 * n_fine, 109 + 16 * n_fine,
                          104 + 16 * n_fine])
    img[chip] = chip_col[chip]
    img = img * (1 - .12 * scratch[..., None]) + \
        np.dstack([135., 134., 131.]) * (.12 * scratch[..., None])
    rough = .225 + .040 * n_broad + .020 * n_fine
    rough = rough * (1 - rust) + .83 * rust
    rough = rough * (1 - chip) + .62 * chip
    rough = np.clip(rough + .045 * scratch, 0, 1)
    metal = .018 + .012 * n_fine
    metal = metal * (1 - rust) + .13 * rust
    metal = metal * (1 - chip) + .62 * chip
    metal = np.clip(metal + .20 * scratch, 0, 1)
    height = .10 * n_fine + .45 * rust * (n_micro * .35 + .14) - .12 * chip + \
        .02 * scratch
    normal = normal_from_height(height, .75)
    base_path = out / 'paint_worn_base.png'
    mr_path = out / 'paint_worn_mr.png'
    normal_path = out / 'paint_worn_normal.png'
    save_rgb(base_path, img)
    save_rgb(mr_path, pack_mr(rough, metal))
    save_rgb(normal_path, normal)
    return {'base': str(base_path), 'mr': str(mr_path),
            'normal': str(normal_path)}, {
        'paint_mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
        'rust_fraction': round(float(((img[..., 0] > img[..., 1] + 12) &
                                      (img[..., 0] > 62)).mean()), 4),
        'roughness_mean': round(float(rough.mean()), 3),
        'metallic_mean': round(float(metal.mean()), 3),
    }


def make_paint_clean(out):
    h = w = 512
    n_broad = fbm(h, w, 3201, (140, 34), (.7, .3))
    n_fine = periodic_noise(h, w, 9, 3202)
    img = np.dstack([29 + 4 * n_broad + 1.5 * n_fine,
                     31 + 4 * n_broad + 1.5 * n_fine,
                     34 + 4 * n_broad + 1.5 * n_fine])
    rough = .26 + .035 * n_broad + .015 * n_fine
    metal = .025 + .015 * n_fine
    normal = normal_from_height(.22 * n_fine + .10 * n_broad, .9)
    paths = (out / 'paint_clean_base.png', out / 'paint_clean_mr.png',
             out / 'paint_clean_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}


def make_forged(out):
    h = w = 1024
    n1 = fbm(h, w, 3301, (190, 48, 13), (.55, .30, .15))
    n2 = periodic_noise(h, w, 6, 3302)
    n3 = periodic_noise(h, w, 2.2, 3303)
    rust = smoothstep(n1 + .30 * n2, 1.95, 2.9)
    rust *= .15 + .85 * smoothstep(n3, -1.0, 1.0)
    rust *= .55
    img = np.dstack([38 + 8 * n1 + 3 * n2, 40 + 8 * n1 + 3 * n2,
                     42 + 8 * n1 + 3 * n2])
    rust_col = np.dstack([82 + 20 * n1 + 8 * n3, 49 + 12 * n1 + 5 * n3,
                          29 + 8 * n2 + 3 * n3])
    mix = (rust * .70)[..., None]
    img = img * (1 - mix) + rust_col * mix
    rough = .44 + .06 * n1 + .02 * n2
    rough = rough * (1 - rust) + .82 * rust
    metal = .06 + .02 * n2
    metal = metal * (1 - rust) + .12 * rust
    height = .13 * n1 + .05 * n2 - .08 * rust * (n3 * .5 + .5)
    normal = normal_from_height(height, .38)
    paths = (out / 'forged_base.png', out / 'forged_mr.png',
             out / 'forged_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, {
        'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
        'rust_fraction': round(float(((img[..., 0] > img[..., 1] + 12) &
                                      (img[..., 0] > 55)).mean()), 4),
    }


def make_chrome(out):
    h = w = 1024
    rng = np.random.default_rng(3401)
    line = rng.normal(size=(1, w)).astype(np.float32)
    streak = np.repeat(line, h, axis=0)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=7.0, sigmaY=0)
    streak -= streak.mean()
    streak /= max(1e-6, streak.std())
    n_fine = periodic_noise(h, w, 3, 3402)
    v = np.linspace(0, 1, h)[:, None]
    dull = smoothstep(v, .94, 1.0) * .35 + smoothstep(-v, -.96, -.91) * .30
    base = 203 + 4.5 * streak + 1.0 * n_fine
    img = np.dstack([base + 2 * dull, base + 1.5 * dull, base - 1 * dull])
    rough = .052 + .012 * streak + .004 * n_fine + .10 * dull
    rough = np.clip(rough, 0, 1)
    metal = np.full((h, w), .998, np.float32)
    scratches = vertical_scratches(h, w, 3403, 28)
    rough = np.clip(rough + .035 * scratches, 0, 1)
    height = .04 * n_fine + .015 * scratches
    normal = normal_from_height(height, .3)
    paths = (out / 'chrome_base.png', out / 'chrome_mr.png',
             out / 'chrome_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, {
        'base_mean': round(float(img.mean()), 2),
        'roughness_mean': round(float(rough.mean()), 4),
        'metallic_mean': round(float(metal.mean()), 4),
    }


def make_machined(out):
    h = w = 1024
    n1 = fbm(h, w, 3501, (110, 30, 8), (.55, .30, .15))
    n2 = periodic_noise(h, w, 3, 3502)
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    turning = .45 * np.sin(v * math.tau * 170 + .8 * n1) + \
        .22 * np.sin(v * math.tau * 47 + 1.7 * n2)
    base = 158 + 6 * n1 + 2 * n2 + 1.2 * turning
    img = np.dstack([base, base + 1, base + 1.5])
    rough = .30 + .035 * n1 + .015 * n2 + .012 * turning
    rough = np.clip(rough, 0, 1)
    metal = np.full((h, w), .995, np.float32)
    height = .06 * n1 + .035 * turning + .015 * n2
    normal = normal_from_height(height, .35)
    paths = (out / 'machined_base.png', out / 'machined_mr.png',
             out / 'machined_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, {
        'roughness_mean': round(float(rough.mean()), 4),
        'metallic_mean': round(float(metal.mean()), 4),
    }


def make_dark_steel(out):
    h = w = 512
    n1 = fbm(h, w, 3601, (90, 22, 6), (.6, .3, .1))
    n2 = periodic_noise(h, w, 2, 3602)
    base = 57 + 8 * n1 + 3 * n2
    img = np.dstack([base, base + 1.5, base + 2])
    rough = np.clip(.50 + .07 * n1 + .03 * n2, 0, 1)
    metal = np.clip(.88 + .04 * n2, 0, 1)
    normal = normal_from_height(.16 * n1 + .06 * n2, .4)
    paths = (out / 'dark_steel_base.png', out / 'dark_steel_mr.png',
             out / 'dark_steel_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}


def make_rust(out):
    h = w = 512
    n1 = fbm(h, w, 3701, (100, 28, 8), (.55, .3, .15))
    n2 = periodic_noise(h, w, 3, 3702)
    img = np.dstack([112 + 28 * n1 + 10 * n2,
                     62 + 18 * n1 + 7 * n2,
                     34 + 10 * n1 + 5 * n2])
    rough = np.clip(.86 + .06 * n1 + .03 * n2, 0, 1)
    metal = np.clip(.12 + .04 * n2, 0, 1)
    normal = normal_from_height(.30 * n1 + .15 * n2, .65)
    paths = (out / 'rust_base.png', out / 'rust_mr.png',
             out / 'rust_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, {
        'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
        'roughness_mean': round(float(rough.mean()), 4),
    }


def make_logo(out):
    w, h = 1024, 256
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    silver = (222, 224, 220, 255)
    font = ImageFont.truetype('C:/Windows/Fonts/ariblk.ttf', 200)
    # Approximate the photographed HCIC mark: a stylised S followed by HCIC.
    mark = 'S'
    mark_bbox = draw.textbbox((0, 0), mark, font=font)
    mw = mark_bbox[2] - mark_bbox[0]
    d = 20
    draw.text((d - mark_bbox[0], 22 - mark_bbox[1]), mark, font=font, fill=silver)
    draw.rectangle((d - 4, 24, d + 12, 226), fill=silver)
    text = 'HCIC'
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = int(w * .57 - tw / 2 - bbox[0])
    y = 26 - bbox[1]
    draw.text((x, y), text, font=font, fill=silver)
    # Wear: make small paint losses in the letters transparent.
    arr = np.asarray(img).copy()
    rng = np.random.default_rng(3801)
    for _ in range(90):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        length = int(rng.integers(2, 28))
        cv2.line(arr, (cx, cy), (cx + length, cy + int(rng.integers(-2, 3))),
                 (0, 0, 0, 0), 1)
    img = Image.fromarray(arr, 'RGBA')
    path = out / 'logo_hcic.png'
    img.save(path)
    alpha = np.asarray(img)[..., 3]
    return {'base': str(path)}, {
        'size': [w, h], 'alpha_coverage': round(float((alpha > 128).mean()), 4)}


def build_textures(texture_dir):
    texture_dir.mkdir(parents=True, exist_ok=True)
    specs = {}
    stats = {}
    specs['paint_worn'], stats['paint_worn'] = make_paint_worn(texture_dir)
    specs['paint_clean'] = make_paint_clean(texture_dir)
    specs['forged'], stats['forged'] = make_forged(texture_dir)
    specs['chrome'], stats['chrome'] = make_chrome(texture_dir)
    specs['machined'], stats['machined'] = make_machined(texture_dir)
    specs['dark_steel'] = make_dark_steel(texture_dir)
    specs['rust'], stats['rust'] = make_rust(texture_dir)
    specs['logo'], stats['logo'] = make_logo(texture_dir)
    labels = {
        'paint_worn': ('黑色亮漆筒体/底盖（磨损、掉漆与锈斑）', 0.02, 0.24, 0.75, .07),
        'paint_clean': ('黑色亮漆头盖', 0.02, 0.25, 0.65, .08),
        'forged': ('黑色锻件/底座夹板（亚光）', 0.06, 0.44, 0.12, .20),
        'chrome': ('镀铬活塞杆', 1.0, 0.06, 0.0, .12),
        'machined': ('机加工钢衬套/接头', 1.0, 0.30, 0.0, .12),
        'dark_steel': ('深色钢/油口内孔', 0.88, 0.50, 0.0, .12),
        'rust': ('焊接飞溅/锈蚀', 0.12, 0.86, 0.0, .12),
        'logo': ('HCIC 银色标识（字形近似）', 0.85, 0.30, 0.0, .12),
    }
    normal_strength = {'paint_worn': .40, 'paint_clean': .30, 'forged': .22,
                       'chrome': .20, 'machined': .15, 'dark_steel': .25,
                       'rust': .40, 'logo': .20}
    materials = {}
    for key, spec in specs.items():
        label, metallic, roughness, coat, coat_rough = labels[key]
        entry = dict(spec)
        entry.update({'label': label, 'metallic': metallic,
                      'roughness': roughness, 'coat': coat,
                      'coat_roughness': coat_rough,
                      'alpha': key == 'logo',
                      'normal_strength': normal_strength[key]})
        materials[key] = entry
    passed = (
        0 < stats['paint_worn']['paint_mean_rgb'][0] < 80 and
        .01 <= stats['paint_worn']['rust_fraction'] <= .35 and
        stats['chrome']['metallic_mean'] > .9 and
        stats['chrome']['roughness_mean'] < .25 and
        .03 <= stats['logo']['alpha_coverage'] <= .60 and
        stats['rust']['mean_rgb'][0] > stats['rust']['mean_rgb'][1] >
        stats['rust']['mean_rgb'][2])
    stats['passed'] = bool(passed)
    stats['note'] = ('贴图由本件代码程序化生成；筒体锈斑分布参照做锈图，'
                     'HCIC 字形为按照片重绘的近似，不是原始商标文件。')
    return materials, stats


# ---------------------------------------------------------------------------
# Image boards
# ---------------------------------------------------------------------------
def load_rgba(path):
    return Image.open(path).convert('RGBA')


def on_bg(image, size, background=(232, 234, 232)):
    image = image.convert('RGBA')
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), background)
    canvas.paste(image, ((size - image.width) // 2,
                         (size - image.height) // 2), image)
    return canvas


def font(size, bold=False):
    path = FONT_BOLD if bold and Path(FONT_BOLD).is_file() else FONT
    return ImageFont.truetype(path, size)


def panel(image, size, title, subtitle=None):
    pad = 10
    head = 34 if subtitle else 24
    canvas = Image.new('RGB', (size + 2 * pad, size + head + 2 * pad),
                       (250, 251, 249))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(on_bg(image, size), (pad, pad))
    draw.text((pad, pad + size + 4), title, font=font(15, True), fill=(29, 35, 32))
    if subtitle:
        draw.text((pad, pad + size + 20), subtitle, font=font(12), fill=(92, 104, 98))
    return canvas


def compose(path, panels, columns, title, note, width=None):
    first = panels[0]
    cell_w = (first[0] if isinstance(first, tuple) else first).width
    cell_h = (first[0] if isinstance(first, tuple) else first).height
    rows = math.ceil(len(panels) / columns)
    head = 74
    if width is None:
        width = 16 + columns * (cell_w + 10) + 16
    canvas = Image.new('RGB', (width, head + rows * cell_h + 46), (238, 241, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 18), title, font=font(24, True), fill=(30, 44, 58))
    draw.text((22, 48), note, font=font(13), fill=(87, 98, 92))
    for index, item in enumerate(panels):
        image, caption = item if isinstance(item, tuple) else (item, '')
        row, col = divmod(index, columns)
        x = 16 + col * (cell_w + 10)
        y = head + row * (cell_h + 8)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1),
                       outline=(206, 213, 207))
        if caption:
            draw.text((x + 8, y + cell_h - 19), caption, font=font(12, True),
                      fill=(40, 50, 45))
    canvas.save(path, quality=92)


def view_cell(render_dir, prefix, view, size, subtitle=None):
    return panel(load_rgba(render_dir / f'{prefix}-{view}.png'), size,
                 f'{prefix.replace("old", "旧模型").replace("new", "新版")} {view}',
                 subtitle)


def make_six_view_board(out):
    render_dir = out / 'renders'
    ref = Image.open(local_paths()['reference']).convert('RGB').resize((1536, 1024))
    ref_panel = panel(ref, 700, '六视图参考',
                      'category_reference，未标定，非同一型号；黄色标签未复制')
    new_panels = [panel(load_rgba(render_dir / f'new-{view}.png'), 420,
                        f'新版 {view}') for view in VIEWS]
    old_panels = [panel(load_rgba(render_dir / f'old-{view}.png'), 420,
                        f'旧模型 {view}') for view in VIEWS]
    cell_w = new_panels[0].width
    cell_h = new_panels[0].height
    width = 16 + 3 * (cell_w + 10) + 16
    head = 74
    headings = 28
    height = (head + ref_panel.height + 18 + headings +
              2 * (cell_h + 8) + 18 + headings + 2 * (cell_h + 8) + 46)
    canvas = Image.new('RGB', (width, height), (238, 241, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 18), '30 液压缸 · 六方向新旧对照', font=font(24, True),
              fill=(30, 44, 58))
    draw.text((22, 48), '参考仅作类别对照；新版按原照可见结构重建，旧模型为 TripoSR 单体壳。',
              font=font(13), fill=(87, 98, 92))
    y = head
    canvas.paste(ref_panel, ((width - ref_panel.width) // 2, y))
    draw.rectangle(((width - ref_panel.width) // 2, y,
                    (width + ref_panel.width) // 2, y + ref_panel.height - 1),
                   outline=(206, 213, 207))
    y += ref_panel.height + 18
    draw.text((22, y), '新版 · 六方向（每格独立正交相机）', font=font(18, True),
              fill=(47, 93, 131))
    y += headings
    for i, image in enumerate(new_panels):
        row, col = divmod(i, 3)
        x = 16 + col * (cell_w + 10)
        yy = y + row * (cell_h + 8)
        canvas.paste(image, (x, yy))
        draw.rectangle((x, yy, x + cell_w - 1, yy + cell_h - 1),
                       outline=(206, 213, 207))
    y += 2 * (cell_h + 8) + 10
    draw.text((22, y), '旧模型 · 六方向（同一组方向，旧壳姿态/结构未修改）',
              font=font(18, True), fill=(138, 90, 30))
    y += headings
    for i, image in enumerate(old_panels):
        row, col = divmod(i, 3)
        x = 16 + col * (cell_w + 10)
        yy = y + row * (cell_h + 8)
        canvas.paste(image, (x, yy))
        draw.rectangle((x, yy, x + cell_w - 1, yy + cell_h - 1),
                       outline=(206, 213, 207))
    canvas.save(out / '六视图-新旧对照.jpg', quality=92)


def make_boards(out, textures_stats):
    render_dir = out / 'renders'
    make_six_view_board(out)

    details = ('rod-eye', 'small-clevis', 'base-lug', 'head-gland', 'logo',
               'base-ports', 'elbow-port', 'overall')
    labels = {'rod-eye': '杆端大耳环/衬套', 'small-clevis': '小耳板/衬套',
              'base-lug': '底座夹板/衬套', 'head-gland': '头盖/杆封',
              'logo': 'HCIC 标识', 'base-ports': '底部双接头',
              'elbow-port': '弯头油口', 'overall': '整体 3/4'}
    panels = []
    for name in details:
        panels.append(panel(load_rgba(render_dir / f'detail-{name}.png'), 360,
                            f'新版 · {labels[name]}'))
    for name in details:
        panels.append(panel(load_rgba(render_dir / f'old-detail-{name}.png'), 360,
                            f'旧模型 · {labels[name]}'))
    compose(out / '关键局部-新旧对照.jpg', panels, 4,
            '30 液压缸 · 关键局部新旧对照',
            '旧模型局部按旧网格主轴近似定位，仅示融合/糊化，不代表同部位精确对齐。')

    isolation = [
        ('new-front.png', '新版整体'), ('isolate-rod-eye.png', '大耳环/衬套隔离'),
        ('isolate-small-clevis.png', '小耳板/衬套隔离'),
        ('isolate-base-lug.png', '底座夹板/衬套隔离'),
        ('isolate-ports.png', '油口附件隔离'), ('isolate-metal.png', '金属件隔离'),
    ]
    panels = [panel(load_rgba(render_dir / name), 420, label)
              for name, label in isolation]
    compose(out / '结构开口-隔离检查.jpg', panels, 3,
            '30 液压缸 · 结构与贯穿孔隔离检查',
            '隔离渲染用于检查耳环、耳板、底座孔与油口；是否真正贯穿另由射线检查确认。')

    panels = [
        panel(load_rgba(render_dir / 'new-material-3q.png'), 480, '新版 · 同光照 3/4'),
        panel(load_rgba(render_dir / 'old-material-3q.png'), 480, '旧模型 · 同光照 3/4'),
        panel(load_rgba(render_dir / 'new-clay-front.png'), 480, '新版 · 灰模正视图'),
        panel(load_rgba(render_dir / 'new-front.png'), 480, '新版 · 材质正视图'),
    ]
    compose(out / '材质-同光照对照.jpg', panels, 2,
            '30 液压缸 · 材质与灰模对照',
            '黑漆为电介质+清漆；镀铬杆、钢衬套和接头为金属。灰模不参与外观验收。')

    panels = [
        panel(load_rgba(render_dir / 'new-front.png'), 480, '新版 · 主光 A'),
        panel(load_rgba(render_dir / 'new-light-alternate.png'), 480, '新版 · 主光 B'),
        panel(load_rgba(render_dir / 'metal-only-light-a.png'), 480, '金属隔离 · 主光 A'),
        panel(load_rgba(render_dir / 'metal-only-light-b.png'), 480, '金属隔离 · 主光 B'),
    ]
    compose(out / '表面反射-变光对照.jpg', panels, 2,
            '30 液压缸 · 变光与金属反射对照',
            '高光随主光位置移动；只证明金属镜面反射变化，不证明反射率实测或场景光照匹配。')

    panels = []
    for i in range(24):
        angle = i * 15
        panels.append(panel(load_rgba(render_dir / f'turn-new-{i:02d}.png'), 250,
                            f'{angle:03d}°'))
    compose(out / '转台-24方向.jpg', panels, 6,
            '30 液压缸 · 新版 24 方向转台',
            '每 15° 一张 Blender 实测渲染；覆盖外观，不证明隐藏结构或装配。')

    src = load_rgba(local_paths()['original'])
    rust = load_rgba(local_paths()['rust'])
    ref_small = Image.open(local_paths()['reference']).convert('RGB')
    panels = [
        panel(src, 480, '原照（原貌）'), panel(rust, 480, '做锈图'),
        panel(load_rgba(render_dir / 'new-source.png'), 480, '新版 · 近似原照视角'),
        panel(load_rgba(render_dir / 'old-source.png'), 480, '旧模型 · 库标定相机'),
        panel(ref_small, 480, '六视图参考（类别）'),
        panel(load_rgba(render_dir / 'new-front.png'), 480, '新版 · 正视图'),
    ]
    compose(out / '原照-新候选-参考.jpg', panels, 3,
            '30 液压缸 · 原照、旧模型、新候选与类别参考',
            '原照为 3/4 视角且无标定；新版视角为近似对照，IoU 只作诊断。')

    tex_items = [
        ('paint_worn_base.png', '筒体底色/锈斑'), ('paint_worn_mr.png', '筒体金属/粗糙度'),
        ('paint_worn_normal.png', '筒体法线'), ('chrome_base.png', '镀铬底色'),
        ('chrome_mr.png', '镀铬金属/粗糙度'), ('machined_mr.png', '机加工钢金属/粗糙度'),
    ]
    panels = [panel(load_rgba(out / 'textures' / name), 360, label)
              for name, label in tex_items]
    compose(out / 'PBR贴图-对照.jpg', panels, 3,
            '30 液压缸 · PBR 贴图检查',
            '程序化贴图并打包进 GLB；金属/粗糙度贴图通道为 R=255、G=粗糙度、B=金属度。'
            '锈斑为外观仿真，不是实测材料图。')


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def rotated_source_mask(paths):
    alpha = np.asarray(Image.open(paths['source_alpha']).convert('L'))
    ys, xs = np.nonzero(alpha >= 128)
    points = np.column_stack([xs, ys]).astype(np.float32)
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    angle = math.degrees(math.atan2(major[1], major[0]))
    # Pick the rotation that makes the long axis most vertical; this avoids
    # sign/coordinate-convention mistakes in the image-space eigenvector.
    best = None
    for candidate in (angle + 90, 90 - angle, angle - 90, -angle - 90):
        rotation = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])),
                                           candidate, 1.0)
        rotated = cv2.warpAffine(alpha, rotation,
                                 (alpha.shape[1], alpha.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        rect = cv2.boundingRect(cv2.findNonZero((rotated >= 128).astype(np.uint8)))
        aspect = rect[3] / max(1, rect[2])
        if best is None or aspect > best[0]:
            best = (aspect, candidate, rotated)
    return alpha, best[2], round(best[1], 2)


def silhouette_iou(a, b):
    fa = mask_fit(a >= 128)
    fb = mask_fit(b >= 128)
    if fa is None or fb is None:
        return None
    return source_fidelity(fa * 255, fb * 255)


def reference_front_mask(paths):
    image = np.asarray(Image.open(paths['reference']).convert('RGB'))
    panel = image[0:460, 0:512]
    gray = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)
    mask = (gray < 238).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def source_diagnostics(out, paths):
    strict_source = np.asarray(Image.open(paths['source_alpha']).convert('L'))
    raw_source, rotated, rotation_deg = rotated_source_mask(paths)
    render_dir = out / 'renders'
    new_source = np.asarray(load_rgba(render_dir / 'new-source-alpha.png').getchannel('A'))
    old_source = np.asarray(load_rgba(render_dir / 'old-source-alpha.png').getchannel('A'))
    strict = source_fidelity(strict_source, new_source)
    strict['acceptance_gate'] = False
    strict['scale_note'] = '手工近似原照相机；未标定，严格 IoU 只作方位诊断'
    old_strict = source_fidelity(strict_source, old_source)
    old_strict['acceptance_gate'] = False
    source_box = silhouette_iou(raw_source, new_source)
    if source_box:
        source_box['alignment'] = '仅按包围盒缩放/平移，不是标定'
        source_box['acceptance_gate'] = False
    old_source_box = silhouette_iou(raw_source, old_source)
    if old_source_box:
        old_source_box['alignment'] = '仅按包围盒缩放/平移，不是标定'
        old_source_box['acceptance_gate'] = False
    new_front = np.asarray(load_rgba(render_dir / 'sil-new-00.png').getchannel('A'))
    rotated_front = silhouette_iou(rotated, new_front)
    if rotated_front:
        rotated_front['alignment'] = ('原照轮廓按主轴旋转后与新模型正视做包围盒对齐；'
                                      '仅诊断，不是标定')
        rotated_front['acceptance_gate'] = False
    reference = silhouette_iou(reference_front_mask(paths), new_front)
    if reference:
        reference['alignment'] = '类别参考正视图，未标定；仅诊断'
        reference['acceptance_gate'] = False
    best = None
    for i in range(24):
        path = render_dir / f'sil-new-{i:02d}.png'
        if not path.is_file():
            continue
        alpha = np.asarray(load_rgba(path).getchannel('A'))
        score = silhouette_iou(rotated, alpha)
        if score is None:
            continue
        if best is None or score['iou'] > best[1]['iou']:
            best = (i * 15, score)
    best_metrics = best[1] if best else {}
    if best_metrics:
        best_metrics.update({'view': f'{best[0]}°',
                             'alignment': '原照主轴旋转后按包围盒对齐；仅诊断',
                             'acceptance_gate': False})
    return {
        'source_alpha': str(paths['source_alpha'].relative_to(ROOT)),
        'source_view': str(paths['source_view'].relative_to(ROOT)),
        'strict': strict,
        'old_strict': old_strict,
        'source_box_fit': source_box,
        'old_source_box_fit': old_source_box,
        'source_rotated_front_fit': rotated_front,
        'category_reference_front_fit': reference,
        'source_vertical_rotation_deg': rotation_deg,
        'best_view': best[0] if best else 0,
        'best': best_metrics,
        'note': ('所有 IoU 均为未标定诊断。新旧模型使用不同相机/尺度；'
                 '原照为 3/4 产品图，不能当作摄影测量。'),
    }


def metal_reflection_check(out):
    a = np.asarray(load_rgba(out / 'renders' / 'metal-only-light-a.png'))
    b = np.asarray(load_rgba(out / 'renders' / 'metal-only-light-b.png'))
    mask = np.logical_and(a[:, :, 3] >= 128, b[:, :, 3] >= 128)
    if mask.sum() < 800:
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
    threshold = max(180.0, .75 * max(float(la[mask].max()), float(lb[mask].max())))
    ya, xa = np.nonzero(mask & (la >= threshold))
    yb, xb = np.nonzero(mask & (lb >= threshold))
    shift = float(np.linalg.norm(ca - cb))
    high_shift = (float(np.linalg.norm(np.array([xa.mean(), ya.mean()]) -
                                       np.array([xb.mean(), yb.mean()])))
                  if len(xa) and len(xb) else 0.0)
    result = {
        'metal_pixels': int(mask.sum()),
        'mean_abs_difference': round(float(diff.mean()), 3),
        'highlight_shift_px': round(shift, 2),
        'high_threshold_luma': round(threshold, 2),
        'high_threshold_shift_px': round(high_shift, 2),
        'highlight_centre_a': [round(float(ca[0]), 1), round(float(ca[1]), 1)],
        'highlight_centre_b': [round(float(cb[0]), 1), round(float(cb[1]), 1)],
    }
    result['passed'] = (result['mean_abs_difference'] >= 1.0 and shift >= 2.0 and
                        len(xqa) >= 40 and len(xqb) >= 40)
    result['note'] = ('只证明金属镜面高光随灯位移动；黑漆、锈与锻件涂层为电介质，'
                      '本检查不适用于它们，也不证明环境光匹配。')
    return result


def image_luma_stats(out):
    stats = {}
    for name in ('new-front.png', 'old-front.png', 'new-material-3q.png'):
        rgba = np.asarray(load_rgba(out / 'renders' / name))
        arr = rgba[:, :, :3]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        mask = rgba[:, :, 3] >= 128
        filled = gray.copy()
        filled[~mask] = float(np.median(gray[mask]))
        laplacian = cv2.Laplacian(filled, cv2.CV_64F)
        stats[name] = {
            'dark_fraction': round(float((gray[mask] < 18).mean()), 4),
            'median_luma': round(float(np.median(gray[mask])), 2),
            'laplacian_variance': round(float(laplacian[mask].var()), 2),
        }
    return stats


def source_measurements(paths):
    image = Image.open(paths['original']).convert('RGBA')
    arr = np.asarray(image)
    mask = arr[:, :, 3] >= 128
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    points = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
    steel = np.logical_and.reduce([mask, hsv[:, :, 1] <= 48,
                                   hsv[:, :, 2] >= 130])
    n, _, stats, cent = cv2.connectedComponentsWithStats(steel.astype(np.uint8), 8)
    steel_components = [{'bbox': [int(s[0]), int(s[1]), int(s[2]), int(s[3])],
                         'area': int(s[4])}
                        for s in stats[1:] if s[4] >= 160]
    rust = np.asarray(Image.open(paths['rust']).convert('RGB')).astype(float)
    clean = arr[:, :, :3].astype(float)
    diff = np.linalg.norm(rust - clean, axis=2)
    brown = np.logical_and.reduce([mask, rust[:, :, 0] >= rust[:, :, 2] + 14,
                                   rust[:, :, 0] >= rust[:, :, 1] - 6,
                                   rust[:, :, 0] >= 72, rust[:, :, 1] <= 155])
    return {
        'image': str(paths['original'].relative_to(ROOT)), 'size': SOURCE_SIZE,
        'alpha_area_px': int(mask.sum()), 'bbox': bbox,
        'min_area_rect': {
            'centre': [round(float(cx), 2), round(float(cy), 2)],
            'size': [round(float(w), 2), round(float(h), 2)],
            'angle_deg': round(float(angle), 2),
            'length_over_width': round(float(max(w, h) / min(w, h)), 4)},
        'steel_components': sorted(steel_components, key=lambda z: -z['area'])[:8],
        'rust_color_shift': {
            'mean_rgb_distance': round(float(diff[mask].mean()), 3),
            'p50': round(float(np.percentile(diff[mask], 50)), 3),
            'p90': round(float(np.percentile(diff[mask], 90)), 3),
            'brown_fraction': round(float(brown.sum() / mask.sum()), 4)},
        'configured': {
            'overall_length_m': round(DIMENSIONS['overall']['eye_top'] -
                                      DIMENSIONS['overall']['base_bottom'], 4),
            'barrel_length_m': round(DIMENSIONS['barrel']['z1'] -
                                     DIMENSIONS['barrel']['z0'], 4),
            'barrel_od_m': round(2 * DIMENSIONS['barrel']['radius'], 4),
            'overall_length_over_barrel_od': round(
                (DIMENSIONS['overall']['eye_top'] - DIMENSIONS['overall']['base_bottom']) /
                (2 * DIMENSIONS['barrel']['radius']), 3),
            'rod_diameter_m': round(2 * DIMENSIONS['rod']['radius'], 4),
            'eye_bore_m': round(2 * DIMENSIONS['eye']['bore_radius'], 4),
            'base_lug_bore_m': round(2 * DIMENSIONS['base_lug']['bore_radius'], 4),
        },
        'note': ('工作尺度按原照比例设置，不是实测。minAreaRect 描述的是照片中的'
                 '可见轮廓，包含透视与 3/4 姿态。'),
    }


def old_model_info(paths):
    scene = trimesh.load(paths['old_mesh'], force='scene')
    geoms = list(scene.geometry.values())
    verts = np.vstack([g.vertices for g in geoms])
    faces = np.vstack([g.faces + sum(len(x.vertices) for x in geoms[:i])
                       for i, g in enumerate(geoms)])
    merged = trimesh.Trimesh(verts, faces, process=True)
    components = merged.split(only_watertight=False)
    audit = None
    for line in paths['audit'].read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get('number') == NUMBER:
            audit = entry
            break
    return {
        'mesh': str(paths['old_mesh'].relative_to(ROOT)),
        'mesh_sha256': digest(paths['old_mesh']),
        'vertices': int(len(merged.vertices)), 'faces': int(len(merged.faces)),
        'bounds': np.round(merged.bounds, 4).tolist(),
        'components': len(components),
        'watertight': bool(merged.is_watertight),
        'audit': audit,
        'note': ('旧模型是单张原貌图经 TripoSR 得到的单体壳；审计记录源图轮廓 IoU '
                 '0.9067、填孔 0，但六方向图仍显示耳环、夹板、杆身和油口融合，'
                 '表面为灰白噪声，不能作为本件结构精度或材质依据。'),
    }


def opening_checks(worker):
    rays = worker.get('ray_checks', {})
    eyes = rays.get('eyes', {})
    bushings = rays.get('bushings', {})
    ports = rays.get('ports', {})
    through = sum(1 for key in ('rod_eye', 'small_clevis', 'base_lug')
                  if eyes.get(key, {}).get('through'))
    rings = sum(1 for key in ('rod_eye', 'small_clevis', 'base_lug')
                if eyes.get(key, {}).get('ring_solid') or
                eyes.get(key, {}).get('lug_solid') or
                eyes.get(key, {}).get('plate_solid'))
    bushing_count = sum(1 for v in bushings.values() if v.get('present'))
    liner_count = sum(1 for v in bushings.values() if v.get('liner_clear'))
    nipple = ports.get('base_nipple_a', {})
    return {
        'through_bores': through,
        'bore_rings_solid': rings,
        'bushings_present': bushing_count,
        'bore_liners_clear': liner_count,
        'base_port_blind_plug': bool(nipple.get('blind_plug_hit')),
        'base_port_outer_depth_m': nipple.get('outer_face_depth_m'),
        'logo_present': bool(rays.get('objects', {}).get('logo')),
        'checks': {
            'three_through_bores': through == 3,
            'three_solid_rings': rings == 3,
            'three_bushings': bushing_count == 3,
            'three_bore_liners_clear': liner_count == 3,
            'base_port_blind': bool(nipple.get('blind_plug_hit')),
            'logo_present': bool(rays.get('objects', {}).get('logo')),
        },
        'note': ('射线直接打在大耳环、小耳板、底座夹板、衬套和油口网格上；'
                 '三个孔中心贯穿、环体/板体为实体、孔内有钢衬套。'
                 '油口为盲孔端塞，不证明内部油路。'),
    }


# ---------------------------------------------------------------------------
# Text report, page, hashes
# ---------------------------------------------------------------------------
def write_report(out, result):
    checks = result['checks']
    diagnostics = result['diagnostics']
    lines = [
        '# 30 液压缸 · 分部件重建候选（manual_only，待审）',
        '',
        f"- 版本：`{VERSION}`",
        '- 原照：黑色 HCIC 液压缸，镀铬活塞杆、杆端大耳环、小耳板、焊接底座夹板、'
        '底部双油口与弯头油口。',
        '- 六视图角色：`category_reference`，未标定、非同一型号，只作类别对照；'
        '参考中的黄色标签未复制。',
        '- 旧模型：TripoSR 单体壳，结构融合、三处孔不真实，未复用；仅渲染对照。',
        '',
        '## 已确认与推断',
        '',
        '已确认（原照可见）：黑色亮漆筒体与银色 HCIC 标识、镀铬杆与台阶、'
        '杆端大耳环及亮色轴套、小耳板及轴套、焊接底座夹板及轴套、'
        '底部两个六角油口与一个弯头油口、头盖/杆封、若干焊缝和锈斑。',
        '',
        '推断/未确认：背面细节、油口内部油路、小耳板是单板还是双耳、'
        '焊缝具体走向、头盖内部密封、耳环轴套是否为球铰、真实尺寸与材质牌号、'
        'HCIC 标识的准确字形/字体、锈斑分布。',
        '',
        '## 实际执行的检查',
        '',
        f"- 贯穿孔：{checks['openings']['through_bores']}/3；"
        f"孔内钢衬套：{checks['openings']['bushings_present']}/3。",
        f"- 部件网格闭合：{checks['mesh']['closed_parts']}/"
        f"{checks['mesh']['parts']}；同部件自交："
        f"{len(checks['mesh']['self_intersecting'])} 项。",
        f"- 贴图数值检查：{'通过' if checks['textures']['passed'] else '未通过'}。",
        f"- 金属高光变光：{'通过' if checks['metal']['passed'] else '未通过'}；"
        f"质心位移 {checks['metal'].get('highlight_shift_px')} px。",
        f"- 原照轮廓诊断（未标定，非验收）：严格 IoU "
        f"{diagnostics['strict']['iou']}，包围盒对齐 IoU "
        f"{(diagnostics.get('source_box_fit') or {}).get('iou', '未执行')}，"
        f"主轴旋转后正视 IoU "
        f"{(diagnostics.get('source_rotated_front_fit') or {}).get('iou', '未执行')}，"
        f"24 方向最优 IoU {diagnostics['best'].get('iou', '未执行')}。",
        '- 浏览器运行时交互：未执行（本件生成独立离线页，使用 Blender 渲染帧）。',
        '',
        '## 待审',
        '',
    ]
    lines += [f'- {item}' for item in result['pending_review']]
    lines += ['', '## 文件', '',
              f"- GLB：`{out / 'mesh.glb'}`",
              f"- Blender：`{out / 'model.blend'}`",
              f"- 检查页：`{out / 'review-30.html'}`",
              '', '候选不替换原模型，不进入自动放置或正式合成。']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf8')


def write_review_page(out, result):
    """Standalone offline page: rendered turntable frames, no external modules."""
    d = result['diagnostics']
    pending = ''.join(f'<li>{item}</li>' for item in result['pending_review'])
    metrics = [
        ('贯穿孔 / 衬套', f"{result['checks']['openings']['through_bores']}/3 · "
         f"{result['checks']['openings']['bushings_present']}/3"),
        ('网格闭合部件', f"{result['checks']['mesh']['closed_parts']}/"
         f"{result['checks']['mesh']['parts']}"),
        ('同部件自交', f"{len(result['checks']['mesh']['self_intersecting'])} 项"),
        ('PBR 贴图数值', '通过' if result['checks']['textures']['passed'] else '未通过'),
        ('金属变光高光位移',
         f"{result['checks']['metal'].get('highlight_shift_px')} px · "
         f"{'通过' if result['checks']['metal'].get('passed') else '未通过'}"),
        ('原照轮廓诊断（未标定）', f"严格 {d['strict']['iou']} · 包围盒 "
         f"{(d.get('source_box_fit') or {}).get('iou', '未执行')} · 主轴正视 "
         f"{(d.get('source_rotated_front_fit') or {}).get('iou', '未执行')}"),
        ('24 方向最优轮廓诊断', f"{d['best'].get('iou', '未执行')} "
         f"({d.get('best', {}).get('view', '未执行')})"),
        ('旧模型库审计', 'IoU 0.9067，填孔 0，通过旧轮廓门槛；但结构融合'),
        ('浏览器运行时检查', '未执行（离线页仅用渲染帧）'),
    ]
    metric_rows = ''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in metrics)
    gallery = [
        ('六视图新旧对照', '六视图-新旧对照.jpg'),
        ('关键局部新旧对照', '关键局部-新旧对照.jpg'),
        ('结构与开口隔离', '结构开口-隔离检查.jpg'),
        ('材质/灰模', '材质-同光照对照.jpg'),
        ('变光与金属反射', '表面反射-变光对照.jpg'),
        ('原照/旧模型/新候选', '原照-新候选-参考.jpg'),
        ('PBR 贴图', 'PBR贴图-对照.jpg'),
        ('24 方向总览', '转台-24方向.jpg'),
    ]
    cards = ''.join(
        f'<a class="card" href="{src}" target="_blank"><img src="{src}" alt="{title}">'
        f'<span>{title}</span></a>' for title, src in gallery)
    new_frames = ',\n'.join(f"'renders/turn-new-{i:02d}.png'" for i in range(24))
    old_frames = ',\n'.join(f"'renders/turn-old-{i:02d}.png'" for i in range(24))
    clay_frames = ',\n'.join(f"'renders/turn-clay-{i:02d}.png'" for i in range(24))
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>30 液压缸 · 分部件重建检查（manual_only）</title>
<style>
:root{{--ink:#1c2320;--muted:#5f6b64;--line:#d4dad4;--panel:#fbfcfa;--accent:#2c5d83;--warn:#8a5a16}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.65 "Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:#edf1ed}}
header{{background:#1f2d3b;color:#f5f8f3;padding:26px 30px}}header h1{{margin:0 0 6px;font-size:25px}}header p{{margin:0;color:#c9d6d0}}
main{{max-width:1180px;margin:24px auto 60px;padding:0 18px}}section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}}
h2{{margin:0 0 12px;font-size:19px;color:var(--accent)}}.notice{{border-left:5px solid var(--warn);background:#fff7e7;padding:12px 16px;border-radius:6px}}
.viewer{{display:grid;grid-template-columns:1fr 280px;gap:18px}}@media(max-width:820px){{.viewer{{grid-template-columns:1fr}}}}
.turnbox{{background:#e2e7e1;border-radius:10px;min-height:420px;display:flex;align-items:center;justify-content:center;user-select:none;touch-action:none;cursor:grab;overflow:hidden}}
.turnbox img{{max-height:500px;max-width:100%}}aside{{font-size:13px;color:var(--muted)}}button{{font:inherit;padding:7px 13px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;margin:2px}}button.sel{{background:#2c5d83;color:#fff;border-color:#2c5d83}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}}th{{width:260px;color:var(--muted);font-weight:500}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}}.card{{display:block;background:#fff;border:1px solid var(--line);border-radius:9px;overflow:hidden;color:inherit;text-decoration:none}}
.card img{{width:100%;height:210px;object-fit:contain;background:#e7ebe6;display:block}}.card span{{display:block;padding:8px 10px;font-size:13px}}
code{{background:#edf1ed;padding:1px 5px;border-radius:4px}}a{{color:var(--accent)}}.tag{{display:inline-block;background:#f1e4cb;color:#77521a;border-radius:20px;padding:2px 10px;font-size:12px}}
</style></head><body>
<header><h1>30 液压缸 · 分部件重建 {VERSION} <span class="tag">manual_only · 待审</span></h1>
<p>黑色 HCIC 液压缸 · 筒体/镀铬杆/大耳环/小耳板/底座夹板/油口分部件重建 · 离线检查页</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b>本候选不替换原模型，不进入自动放置或正式合成。
六视图是同类别、未标定的参考（非同一型号）；所有 IoU 只作诊断，不是验收通过。
背面、油路、焊缝走向、真实尺寸、落稳与现场光照均待人工复核。浏览器运行时交互未执行；
本页的旋转视图是 Blender 实测渲染帧。</div></section>

<section><h2>旋转检查</h2><div class="viewer">
<div class="turnbox" id="turnbox"><img id="turn" src="renders/turn-new-00.png" alt="转台"></div>
<aside><p>拖动图片旋转，或使用下方按钮。每 15° 一帧，共 24 帧。</p>
<p><button data-mode="new" class="sel">新版</button><button data-mode="old">旧模型</button><button data-mode="clay">灰模</button></p>
<p><button id="prev">上一帧</button><button id="next">下一帧</button><button id="play">自动旋转</button><button id="reset">重置</button></p>
<p>当前角度：<b id="angle">000°</b> · 模式：<b id="modeLabel">新版</b></p>
<p><a href="mesh.glb">下载 GLB</a><br><a href="model.blend">下载 Blender 文件</a></p></aside></div></section>

<section><h2>对照图</h2><div class="grid">{cards}</div></section>

<section><h2>检查记录</h2><table>{metric_rows}</table>
<p class="notice" style="margin-top:12px">未执行项明确写为“未执行”，不能解读为通过。金属变光只证明高光移动；
开孔射线只覆盖三个可见贯穿孔、衬套和油口端塞。</p></section>

<section><h2>待人工复核</h2><ul>{pending}</ul></section>

<section><h2>文件与记录</h2><p><a href="result.json">result.json</a> ·
<a href="hashes.json">hashes.json</a> · <a href="worker-result.json">Blender worker 记录</a> ·
<a href="report.md">中文报告</a> · <a href="source-measurements.json">原照测量</a> ·
<a href="old-model-audit.json">旧模型审计</a></p>
<p>贴图：<a href="textures/paint_worn_base.png">筒体底色</a> ·
<a href="textures/paint_worn_mr.png">筒体金属/粗糙度</a> ·
<a href="textures/chrome_mr.png">镀铬金属/粗糙度</a> ·
<a href="textures/logo_hcic.png">HCIC 标识</a></p>
<p>候选 <code>{VERSION}</code>，参考角色 <code>category_reference</code>；原模型和共享页面未被修改。</p></section>
</main>
<script>
const sets={{new:[{new_frames}],old:[{old_frames}],clay:[{clay_frames}]}};
let mode='new',index=0,timer=null,dragging=false,lastX=0;
const labels={{new:'新版',old:'旧模型',clay:'灰模'}};
const img=document.getElementById('turn'),angle=document.getElementById('angle'),modeLabel=document.getElementById('modeLabel');
function draw(){{img.src=sets[mode][index];angle.textContent=String(index*15).padStart(3,'0')+'°';modeLabel.textContent=labels[mode];}}
function step(delta){{index=(index+delta+24)%24;draw();}}
function setMode(value){{mode=value;index=0;for(const b of document.querySelectorAll('button[data-mode]'))b.classList.toggle('sel',b.dataset.mode===value);draw();}}
for(const b of document.querySelectorAll('button[data-mode]'))b.addEventListener('click',()=>setMode(b.dataset.mode));
document.getElementById('prev').addEventListener('click',()=>step(-1));
document.getElementById('next').addEventListener('click',()=>step(1));
document.getElementById('reset').addEventListener('click',()=>{{index=0;draw();}});
document.getElementById('play').addEventListener('click',()=>{{if(timer){{clearInterval(timer);timer=null;document.getElementById('play').textContent='自动旋转';}}else{{timer=setInterval(()=>step(1),190);document.getElementById('play').textContent='停止旋转';}}}});
const box=document.getElementById('turnbox');
box.addEventListener('pointerdown',e=>{{dragging=true;lastX=e.clientX;box.setPointerCapture(e.pointerId);}});
box.addEventListener('pointermove',e=>{{if(!dragging)return;const dx=e.clientX-lastX;if(Math.abs(dx)>=12){{step(dx>0?1:-1);lastX=e.clientX;}}}});
box.addEventListener('pointerup',()=>{{dragging=false;}});box.addEventListener('pointercancel',()=>{{dragging=false;}});
draw();
</script></body></html>'''
    (out / 'review-30.html').write_text(html, encoding='utf8')


def collect_hashes(out, paths, scripts):
    outputs = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'hashes.json':
            outputs[str(path.relative_to(ROOT)).replace('\\', '/')] = {
                'sha256': digest(path), 'bytes': path.stat().st_size}
    return {
        'inputs': {key: {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                         'sha256': digest(path)} for key, path in paths.items()},
        'scripts': {name: {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                           'sha256': digest(path)} for name, path in scripts.items()},
        'outputs': outputs,
    }


def write_progress(result, out, status):
    progress = {
        'number': NUMBER,
        'status': status,
        'out': str(out),
        'mesh': str(out / 'mesh.glb'),
        'blend': str(out / 'model.blend'),
        'checks': result['checks'],
        'pending_review': result['pending_review'],
        'next_number': 31,
    }
    target = ROOT / 'work/direct-deepseek/30-progress.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                      encoding='utf8')
    return target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', type=int, default=NUMBER)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    parser.add_argument('--skip-checks', action='store_true')
    args = parser.parse_args()
    if args.number != NUMBER:
        parser.error('this entry supports item 30 only')
    if args.draft:
        RENDER.update({'six_samples': 12, 'detail_samples': 12,
                       'turn_samples': 5})
    if args.out is None:
        args.out = (ROOT / 'work/direct-deepseek/30-draft' if args.draft
                    else ROOT / 'outputs/30液压缸-六视图重建-v1')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'output directory is not empty: {out}')
    paths = local_paths()
    verify_inputs(paths)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'renders').mkdir(exist_ok=True)
    print('building PBR textures ...', flush=True)
    materials, texture_stats = build_textures(out / 'textures')
    (out / 'texture-checks.json').write_text(
        json.dumps(texture_stats, ensure_ascii=False, indent=2), encoding='utf8')
    request = {
        'number': NUMBER,
        'version': VERSION,
        'out': str(out),
        'source_camera': {'eye': [1.9, 0, 0], 'target': [0, 0, 0],
                          'fov_deg': 37.461451, 'shift_x': 0.001525,
                          'shift_y': 0.000522},
        'original_mesh': str(paths['old_mesh']),
        'config': {'dimensions': DIMENSIONS, 'render': RENDER},
        'materials': materials,
        'textures': {key: value for key, value in materials.items()},
        'draft': bool(args.draft),
    }
    (out / 'request.json').write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
    worker = HERE / 'mesh_rebuild_30_worker.py'
    cmd = [str(args.blender), '-b', '--factory-startup', '--python', str(worker),
           '--', str(out / 'request.json')]
    print('rendering with Blender ...', flush=True)
    with open(out / 'blender.log', 'w', encoding='utf8') as log:
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), timeout=7200)
    if process.returncode != 0:
        raise RuntimeError(f'Blender worker failed with code {process.returncode}; '
                           f'see {out / "blender.log"}')
    if not (out / 'mesh.glb').is_file() or not (out / 'model.blend').is_file():
        raise RuntimeError('Blender worker did not produce mesh.glb/model.blend')
    worker_result = json.loads((out / 'worker-result.json').read_text(encoding='utf8'))
    print('composing comparison boards ...', flush=True)
    make_boards(out, texture_stats)
    diagnostics = source_diagnostics(out, paths)
    metal = metal_reflection_check(out)
    luma = image_luma_stats(out)
    openings = opening_checks(worker_result)
    measurements = source_measurements(paths)
    (out / 'source-measurements.json').write_text(
        json.dumps(measurements, ensure_ascii=False, indent=2), encoding='utf8')
    old_info = old_model_info(paths)
    (out / 'old-model-audit.json').write_text(
        json.dumps(old_info, ensure_ascii=False, indent=2), encoding='utf8')
    closed_parts = worker_result.get('closed_parts', 0)
    parts = worker_result.get('parts', [])
    checks = {
        'openings': openings,
        'textures': texture_stats,
        'metal': metal,
        'luma': luma,
        'mesh': {
            'parts': len(parts),
            'closed_parts': closed_parts,
            'closures': worker_result.get('closures', {}),
            'self_intersecting': worker_result.get('self_intersecting_components', {}),
            'polygons': worker_result.get('polygons'),
            'triangles': worker_result.get('triangles'),
            'bounds': worker_result.get('bounds'),
        },
        'diagnostics': diagnostics,
    }
    pending = [
        '六视图为未标定的同类别参考（不同型号），背面与隐藏结构为推断；'
        '黄色参考标签未复制。',
        '小耳板按单板建模；原照是否双耳、耳环内部是否为球铰待实物复核。',
        '油口内部油路、端塞螺纹、头盖内部密封与杆封结构未验证。',
        '焊缝走向、打磨状态、HCIC 标识的准确字形/字体和锈斑分布为近似。',
        '真实尺寸、材质牌号、落稳姿态、现场尺度和合成环境光未验证。',
        '原照轮廓 IoU 为未标定诊断，不能当作精度或验收结论。',
        '浏览器运行时交互未执行；离线页只使用 Blender 渲染帧。',
    ]
    result = {
        'number': NUMBER,
        'name': '液压缸',
        'version': VERSION,
        'status': 'manual_only_pending_review',
        'manual_only': True,
        'replaces_library_mesh': False,
        'enters_automatic_placement': False,
        'reference': {
            'role': 'category_reference',
            'file': str(paths['reference'].relative_to(ROOT)),
            'sha256': digest(paths['reference']),
            'note': ('同一类别但型号/接口未确认；未标定，参考中的黄色标签与原照不同，'
                     '未复制。'),
        },
        'source': {'original': str(paths['original'].relative_to(ROOT)),
                   'rust': str(paths['rust'].relative_to(ROOT)),
                   'original_sha256': digest(paths['original']),
                   'rust_sha256': digest(paths['rust'])},
        'method': ('按原照可见结构分部件重建：真实三处贯穿孔、镀铬杆、黑色亮漆筒体、'
                   '锻件/焊缝、六角油口与弯头；程序化 PBR 贴图与离线检查页。'),
        'confirmed': [
            '黑色亮漆筒体与银色 HCIC 标识', '镀铬活塞杆与台阶/沟槽',
            '杆端大耳环与亮色钢衬套', '小耳板与钢衬套',
            '焊接底座夹板与钢衬套', '底部两个六角油口', '杆封附近的弯头油口',
            '头盖/杆封/防尘圈', '筒体-底盖与夹板-底盖焊缝', '掉漆与锈斑'],
        'inferred': pending,
        'mesh': {
            'parts': len(parts), 'closed_parts': closed_parts,
            'polygons': worker_result.get('polygons'),
            'triangles': worker_result.get('triangles'),
            'bounds': worker_result.get('bounds'),
            'coordinate_system': worker_result.get('coordinate_system'),
            'metal_objects': worker_result.get('metal_objects'),
            'nonmetal_objects': worker_result.get('nonmetal_objects'),
        },
        'checks': checks,
        'openings': openings,
        'diagnostics': diagnostics,
        'old_model': old_info,
        'pending_review': pending,
        'files': {
            'mesh': str(out / 'mesh.glb'),
            'blend': str(out / 'model.blend'),
            'page': str(out / 'review-30.html'),
            'textures': str(out / 'textures'),
            'boards': {
                'six_view': str(out / '六视图-新旧对照.jpg'),
                'details': str(out / '关键局部-新旧对照.jpg'),
                'openings': str(out / '结构开口-隔离检查.jpg'),
                'material': str(out / '材质-同光照对照.jpg'),
                'light': str(out / '表面反射-变光对照.jpg'),
                'source': str(out / '原照-新候选-参考.jpg'),
                'turntable': str(out / '转台-24方向.jpg'),
                'textures': str(out / 'PBR贴图-对照.jpg'),
            },
        },
    }
    (out / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    write_report(out, result)
    write_review_page(out, result)
    scripts = {'driver': Path(__file__), 'worker': worker}
    checks_path = HERE / 'checks_rebuild_30.py'
    if checks_path.is_file():
        scripts['checks'] = checks_path
    hashes = collect_hashes(out, paths, scripts)
    (out / 'hashes.json').write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf8')
    if not args.draft:
        write_progress(result, out, 'candidate_built_pending_checks')
    print(f'candidate written to {out}')
    if not args.skip_checks and checks_path.is_file():
        print('running independent checks ...', flush=True)
        run = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(checks_path),
                              '--out', str(out)], cwd=str(ROOT))
        if run.returncode != 0:
            raise RuntimeError('independent checks reported failures; '
                               'see checks-30.json')
    print('done')


if __name__ == '__main__':
    main()
