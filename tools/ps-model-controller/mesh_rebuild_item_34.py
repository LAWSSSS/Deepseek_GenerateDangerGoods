"""Build the review-only part-based candidate for item 34.

Item 34 is a green Oulunte (欧伦泰) MPZ/3 3-litre water-based fire extinguisher.
The clean photograph and the matching rust derivative show a green painted
steel cylinder with a rolled bottom rim, a rounded shoulder and short neck, a
green valve with a chrome hose coupling, a round pressure gauge, a lime safety
ring, a yellow tamper strap / hanging tag, an upper squeeze lever and a lower
fixed handle with a real gap, and a black rubber hose ending in a black nozzle
at the lower left.

The supplied six-view sheet is an uncalibrated visual reference for the same
kind of extinguisher (role ``visual_match_unverified``).  It agrees on the
overall layout and the back label, but the handle angle and some small details
differ from the photograph; the photograph wins and the sheet is not used as a
measurement.

The old TripoSR shell fuses the handles, gauge and hose, has no open squeeze
gap, smears the labels and does not model the rolled bottom rim.  It is kept
unchanged for comparison.  The candidate below is a new part-based mesh with
true openings, separate painted metal / rubber / chrome / brass / paper
materials and procedural PBR maps.  It remains manual_only/pending review and
never replaces the library mesh or enters automatic placement or the formal
composite.
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

NUMBER = 34
VERSION = 'fireext34-parametric-v1'
REFERENCE_NAME = '绿色灭火器六视图产品展示.png'
REFERENCE_SHA256 = '2ab6641f322fb308e5f3e0f9c5cb6cc9e3f7acdcbb44d0fefe4ab31c171b9cee'
SOURCE_SIZE = (1029, 1528)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
FONT = 'C:/Windows/Fonts/msyh.ttc'
FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
FONT_LATIN = 'C:/Windows/Fonts/arial.ttf'
FONT_LATIN_BOLD = 'C:/Windows/Fonts/arialbd.ttf'

DIMENSIONS = {
    'body': {'radius': 0.150, 'z0': -0.320, 'z1': 0.285, 'bottom': -0.380,
             'top': 0.442, 'neck_r': 0.046},
    'bottom': {'rim_r': 0.1406, 'rim_z': -0.367, 'inner_r': 0.0725,
               'inner_z': -0.357, 'center_z': -0.351},
    'valve': {'r': 0.036, 'z0': 0.445, 'z1': 0.522, 'top_r': 0.033,
              'top_z1': 0.538, 'pivot': [0.0, 0.018, 0.545],
              'pivot_r': 0.020, 'pivot_len': 0.072},
    'gauge': {'center': [0.0, -0.060, 0.514], 'housing_r': 0.040,
              'dial_r': 0.0360, 'bezel_r': 0.045, 'bezel_in': 0.0370,
              'front_x': 0.072},
    'handles': {
        'lower': [[0, 0.010, 0.510], [0, 0.055, 0.505], [0, 0.110, 0.500],
                  [0, 0.165, 0.497], [0, 0.215, 0.499], [0, 0.250, 0.506],
                  [0, 0.278, 0.512], [0, 0.292, 0.514]],
        'upper': [[0, 0.010, 0.545], [0, 0.055, 0.565], [0, 0.105, 0.590],
                  [0, 0.160, 0.615], [0, 0.215, 0.640], [0, 0.265, 0.658],
                  [0, 0.298, 0.668], [0, 0.315, 0.665]],
        'lower_rx': 0.019, 'lower_rz': 0.021,
        'upper_rx': 0.017, 'upper_rz': 0.022},
    'safety': {'ring_center': [0.030, 0.045, 0.560], 'ring_major': 0.016,
               'ring_minor': 0.0035, 'strap_center': [0.0, 0.018, 0.545],
               'strap_major': 0.026, 'strap_minor': 0.004,
               'tag_center': [0.006, 0.078, 0.492],
               'tag_size': [0.004, 0.086, 0.118], 'tag_rot': 0.12},
    'hose': {'radius': 0.0115,
             'path': [[0.020, -0.100, 0.518], [0.020, -0.145, 0.510],
                      [0.028, -0.180, 0.485], [0.040, -0.198, 0.435],
                      [0.050, -0.206, 0.335], [0.056, -0.206, 0.225],
                      [0.060, -0.198, 0.095], [0.060, -0.190, -0.035],
                      [0.055, -0.184, -0.155], [0.042, -0.180, -0.235]],
             'coupling_from': [0.020, -0.058, 0.516],
             'coupling_to': [0.020, -0.112, 0.516], 'coupling_r': 0.019,
             'nozzle_center': [0.040, -0.180, -0.276],
             'outlet_z': -0.316, 'outlet_r_out': 0.017, 'outlet_r_in': 0.010,
             'plug_z': -0.312},
    'label': {
        'front': {'center_angle': 0.0, 'half_angle': 1.00, 'z0': -0.205,
                  'z1': 0.150, 'radius': 0.1506, 'thickness': 0.0008},
        'back': {'center_angle': math.pi, 'half_angle': 0.63, 'z0': -0.185,
                 'z1': 0.125, 'radius': 0.1506, 'thickness': 0.0008},
        'emboss': {'center_angle': 0.0, 'half_angle': 0.40, 'z0': -0.258,
                   'z1': -0.222, 'radius': 0.1507, 'thickness': 0.0006}},
    'overall': {'bottom': -0.380, 'top': 0.668},
}
RENDER = {
    'six_resolution': 512, 'six_samples': 40,
    'detail_resolution': 512, 'detail_samples': 36,
    'turn_resolution': 320, 'turn_samples': 9, 'turn_count': 24,
    'reference_role': 'visual_match_unverified',
}


# ---------------------------------------------------------------------------
# Inputs and paths
# ---------------------------------------------------------------------------
def local_paths():
    return {
        'original': ROOT / 'outputs/物品高清输入-39件/原貌HD/34-灭火器1.png',
        'rust': ROOT / 'outputs/物品高清输入-39件/生锈HD/34-灭火器1.png',
        'reference': ROOT / 'references/parallel-six-view' / REFERENCE_NAME,
        'old_mesh': ROOT / ('outputs/三维模型库-圆润复验/items/'
                            '34-asset-8861c44bb417/mesh.glb'),
        'source_view': ROOT / ('outputs/三维模型库-圆润复验/items/'
                               '34-asset-8861c44bb417/source-view.png'),
        'source_alpha': ROOT / ('outputs/三维模型库-圆润复验/items/'
                                '34-asset-8861c44bb417/source-alpha.png'),
        'audit': ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl',
    }


def verify_inputs(paths):
    expected = {
        'original': '5f1e90d92690d12a10a12ca2b2f01474c3316013417efa3579211b15c7a75420',
        'rust': '3b8d44a83bc798db7c0596050a411f1b7ca2d6a895c8af2227456bb55efd0ec9',
        'reference': REFERENCE_SHA256,
        'old_mesh': '192c1f7c2604b2a02d52aad811e92c26d318c3a56bb0d0c5b2fe616b0d50938c',
        'source_view': 'ff5daf8ea914817df63f3e2d02973227a79f0b0c5c54a7a383f56764a69bfff1',
        'source_alpha': '2627fc138aed6299a7ce78d1bdbf8cfb8e876d9c8137530c489fb5b7cdf9c09d',
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


def vertical_scratches(h, w, seed, count=220):
    rng = np.random.default_rng(seed)
    canvas = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for _ in range(count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(6, 150))
        drift = int(rng.integers(-3, 4))
        value = int(rng.integers(45, 150))
        draw.line((x, y, x + drift, y + length), fill=value, width=1)
    return np.asarray(canvas, np.float32) / 255.0


def make_green_paint(out, key, seed, *, clean=False):
    h = w = 1024
    v = 1 - np.linspace(0, 1, h, endpoint=False)[:, None]
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    n_broad = fbm(h, w, seed, (260, 82), (.7, .3))
    n_mid = periodic_noise(h, w, 104, seed + 1)
    n_fine = periodic_noise(h, w, 26, seed + 2)
    n_micro = periodic_noise(h, w, 6, seed + 3)
    base_weight = .22 + .78 * np.exp(-((v - .015) / .18) ** 2)
    shoulder = .55 * np.exp(-((v - .92) / .06) ** 2)
    field = .66 * n_mid + .36 * n_fine + .22 * n_broad
    wear = 0.34 if clean else 1.0
    threshold = 1.42 - .78 * (base_weight + shoulder) * wear
    rust = smoothstep(field, threshold, threshold + .75)
    rust *= .20 + .80 * smoothstep(n_broad, -1.0, .9)
    rust *= smoothstep(n_fine, -1.4, .3)
    rust = np.clip(rust * (.58 if clean else 1.0), 0, 1)
    if clean:
        rust[:] = 0.0
    rust_bin = (rust > .58).astype(np.uint8)
    edge = cv2.dilate(rust_bin, np.ones((5, 5), np.uint8)) - rust_bin
    chip = (edge > 0) & (n_fine > 1.05) & (n_micro > -.1)
    if clean:
        chip[:] = False
    scratch = vertical_scratches(h, w, seed + 4, 72 if not clean else 0)
    base = 134 if clean else 140
    paint = np.dstack([28 + 3.2 * n_broad + 1.3 * n_fine,
                       base + 8 * n_broad + 3.2 * n_fine,
                       46 + 3.2 * n_broad + 1.4 * n_fine])
    rust_col = np.dstack([88 + 26 * n_mid + 9 * n_fine,
                          50 + 15 * n_mid + 6 * n_fine,
                          29 + 9 * n_mid + 4 * n_fine])
    mix = (rust * .92)[..., None]
    img = paint * (1 - mix) + rust_col * mix
    chip_col = np.dstack([120 + 14 * n_fine, 121 + 14 * n_fine,
                          112 + 14 * n_fine])
    img[chip] = chip_col[chip]
    scratch_mix = 0.0 if clean else .09
    img = img * (1 - scratch_mix * scratch[..., None]) + \
        np.dstack([146., 148., 138.]) * (scratch_mix * scratch[..., None])
    rough = .245 + .038 * n_broad + .018 * n_fine
    rough = rough * (1 - rust) + .83 * rust
    rough = rough * (1 - chip) + .60 * chip
    rough = np.clip(rough + .05 * scratch, 0, 1)
    metal = .085 + .025 * n_fine
    metal = metal * (1 - rust) + .10 * rust
    metal = metal * (1 - chip) + .46 * chip
    metal = np.clip(metal + .18 * scratch, 0, 1)
    height = .09 * n_fine + .38 * rust * (n_micro * .30 + .12) - .10 * chip + \
        .018 * scratch
    if clean:
        height = .035 * n_fine + .02 * n_micro
    normal = normal_from_height(height, .34 if clean else .62)
    paths = (out / f'{key}_base.png', out / f'{key}_mr.png',
             out / f'{key}_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {
        'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
        'green_dominant': bool(img[..., 1].mean() > img[..., 0].mean() + 25 and
                               img[..., 1].mean() > img[..., 2].mean() + 25),
        'rust_fraction': round(float(((img[..., 0] > img[..., 1] + 10) &
                                      (img[..., 0] > 60)).mean()), 4),
        'chip_fraction': round(float(chip.mean()), 4),
        'roughness_mean': round(float(rough.mean()), 4),
        'metallic_mean': round(float(metal.mean()), 4),
    }
    min_rust = 0.0 if clean else 0.001
    stats['passed'] = bool(
        stats['green_dominant'] and
        min_rust <= stats['rust_fraction'] <= (0.14 if not clean else 0.08) and
        0.12 <= stats['roughness_mean'] <= 0.55 and
        stats['metallic_mean'] <= 0.30)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_rubber(out):
    h = w = 512
    n1 = fbm(h, w, 4301, (120, 34, 9), (.55, .30, .15))
    n2 = periodic_noise(h, w, 5, 4302)
    n3 = periodic_noise(h, w, 2, 4303)
    base = 27 + 4 * n1 + 1.6 * n2
    img = np.dstack([base, base + 1.0, base + 2.0])
    rough = np.clip(.52 + .055 * n1 + .018 * n2, 0, 1)
    metal = np.zeros((h, w), np.float32)
    height = .14 * n1 + .06 * n2 + .02 * n3
    normal = normal_from_height(height, .55)
    paths = (out / 'rubber_base.png', out / 'rubber_mr.png',
             out / 'rubber_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
             'roughness_mean': round(float(rough.mean()), 4),
             'metallic_mean': 0.0}
    stats['passed'] = bool(stats['mean_rgb'][0] < 60 and
                           .38 <= stats['roughness_mean'] <= .72)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_chrome(out):
    h = w = 512
    rng = np.random.default_rng(4401)
    line = rng.normal(size=(1, w)).astype(np.float32)
    streak = np.repeat(line, h, axis=0)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=7.0, sigmaY=0)
    streak -= streak.mean()
    streak /= max(1e-6, streak.std())
    n_fine = periodic_noise(h, w, 3, 4402)
    base = 203 + 4.5 * streak + 1.0 * n_fine
    img = np.dstack([base + 1.5, base + .8, base - .8])
    rough = np.clip(.065 + .012 * streak + .004 * n_fine, 0, 1)
    metal = np.full((h, w), .998, np.float32)
    scratches = vertical_scratches(h, w, 4403, 24)
    rough = np.clip(rough + .03 * scratches, 0, 1)
    normal = normal_from_height(.04 * n_fine + .012 * scratches, .3)
    paths = (out / 'chrome_base.png', out / 'chrome_mr.png',
             out / 'chrome_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {'base_mean': round(float(img.mean()), 2),
             'roughness_mean': round(float(rough.mean()), 4),
             'metallic_mean': round(float(metal.mean()), 4)}
    stats['passed'] = bool(stats['metallic_mean'] > .9 and
                           stats['roughness_mean'] < .25)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_brass(out):
    h = w = 512
    n1 = fbm(h, w, 4501, (100, 30, 8), (.55, .30, .15))
    n2 = periodic_noise(h, w, 4, 4502)
    tarnish = smoothstep(n1 + .30 * n2, 1.15, 2.4) * .45
    img = np.dstack([148 + 12 * n1 + 4 * n2,
                     110 + 10 * n1 + 3 * n2,
                     47 + 7 * n1 + 2 * n2])
    img = img * (1 - tarnish[..., None]) + \
        np.dstack([105., 82., 45.]) * tarnish[..., None]
    rough = np.clip(.24 + .06 * n1 + .02 * n2 + .10 * tarnish, 0, 1)
    metal = np.full((h, w), .995, np.float32)
    normal = normal_from_height(.11 * n1 + .04 * n2, .42)
    paths = (out / 'brass_base.png', out / 'brass_mr.png',
             out / 'brass_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
             'roughness_mean': round(float(rough.mean()), 4),
             'metallic_mean': round(float(metal.mean()), 4)}
    stats['passed'] = bool(stats['metallic_mean'] > .9 and
                           stats['mean_rgb'][0] > stats['mean_rgb'][1] >
                           stats['mean_rgb'][2])
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_dark_steel(out):
    h = w = 512
    n1 = fbm(h, w, 4601, (80, 22, 6), (.6, .3, .1))
    n2 = periodic_noise(h, w, 2, 4602)
    base = 57 + 8 * n1 + 3 * n2
    img = np.dstack([base, base + 1.2, base + 2.0])
    rough = np.clip(.42 + .07 * n1 + .03 * n2, 0, 1)
    metal = np.clip(.88 + .04 * n2, 0, 1)
    normal = normal_from_height(.14 * n1 + .05 * n2, .4)
    paths = (out / 'dark_steel_base.png', out / 'dark_steel_mr.png',
             out / 'dark_steel_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}


def make_yellow_plastic(out):
    h = w = 512
    n1 = fbm(h, w, 4701, (90, 26), (.7, .3))
    n2 = periodic_noise(h, w, 5, 4702)
    img = np.dstack([238 + 5 * n1 + 1.5 * n2,
                     198 + 6 * n1 + 2 * n2,
                     35 + 4 * n1 + 1.5 * n2])
    rough = np.clip(.34 + .04 * n1 + .015 * n2, 0, 1)
    metal = np.zeros((h, w), np.float32)
    normal = normal_from_height(.10 * n1 + .04 * n2, .45)
    paths = (out / 'yellow_plastic_base.png', out / 'yellow_plastic_mr.png',
             out / 'yellow_plastic_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
             'roughness_mean': round(float(rough.mean()), 4)}
    stats['passed'] = bool(stats['mean_rgb'][0] > 180 and
                           stats['mean_rgb'][1] > 150 and
                           stats['mean_rgb'][2] < 100)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_lime_plastic(out):
    h = w = 256
    n1 = fbm(h, w, 4801, (60, 18), (.7, .3))
    n2 = periodic_noise(h, w, 4, 4802)
    img = np.dstack([172 + 7 * n1 + 2 * n2,
                     222 + 6 * n1 + 2 * n2,
                     44 + 5 * n1 + 1.5 * n2])
    rough = np.clip(.30 + .04 * n1 + .015 * n2, 0, 1)
    metal = np.zeros((h, w), np.float32)
    normal = normal_from_height(.08 * n1 + .03 * n2, .4)
    paths = (out / 'lime_plastic_base.png', out / 'lime_plastic_mr.png',
             out / 'lime_plastic_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}


def latin(size, bold=False):
    path = FONT_LATIN_BOLD if bold and Path(FONT_LATIN_BOLD).is_file() else FONT_LATIN
    return ImageFont.truetype(path, size)


def lab_font(size, bold=False):
    path = FONT_BOLD if bold and Path(FONT_BOLD).is_file() else FONT
    return ImageFont.truetype(path, size)


def draw_icon(draw, letter, center):
    cx, cy = center
    green = (26, 116, 52)
    if letter == 'A':
        draw.rectangle((cx - 24, cy - 4, cx + 24, cy + 30), outline=green,
                       width=3)
        draw.line((cx - 18, cy - 4, cx + 18, cy - 4), fill=green, width=3)
        draw.line((cx - 14, cy - 4, cx - 10, cy - 22), fill=green, width=4)
        draw.line((cx + 14, cy - 4, cx + 10, cy - 22), fill=green, width=4)
    elif letter == 'B':
        draw.polygon([(cx, cy - 26), (cx + 20, cy + 18), (cx - 20, cy + 18)],
                     outline=green, width=3)
        draw.polygon([(cx, cy - 8), (cx + 10, cy + 14), (cx - 10, cy + 14)],
                     fill=green)
    elif letter == 'C':
        draw.rounded_rectangle((cx - 14, cy - 26, cx + 14, cy + 28),
                               radius=8, outline=green, width=3)
        draw.line((cx - 6, cy - 32, cx + 6, cy - 32), fill=green, width=4)
        draw.line((cx - 8, cy - 26, cx - 8, cy + 10), fill=green, width=2)
    else:
        draw.line((cx + 2, cy - 28, cx - 16, cy + 2), fill=green, width=4)
        draw.line((cx - 16, cy + 2, cx + 14, cy + 2), fill=green, width=4)
        draw.line((cx + 14, cy + 2, cx - 4, cy + 30), fill=green, width=4)


def draw_house(draw, center):
    cx, cy = center
    green = (26, 116, 52)
    draw.line((cx - 52, cy + 8, cx, cy - 42, cx + 52, cy + 8), fill=green,
              width=5, joint='curve')
    draw.rectangle((cx - 42, cy + 8, cx + 42, cy + 56), outline=green, width=4)
    draw.rectangle((cx - 12, cy + 24, cx + 12, cy + 56), outline=green, width=3)
    draw.line((cx - 30, cy + 38, cx + 30, cy + 38), fill=green, width=3)
    draw.line((cx - 30, cy + 50, cx + 30, cy + 50), fill=green, width=3)


def paper_finish(img, seed, *, edge=14):
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    n = fbm(h, w, seed + 30, (170, 38, 7), (.55, .3, .15))
    arr += (2.2 * n)[..., None]
    yy, xx = np.mgrid[0:h, 0:w]
    border = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    grime = np.clip(1 - border / edge, 0, 1) * .05
    arr *= (1 - grime[..., None])
    rng = np.random.default_rng(seed + 31)
    for _ in range(60):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        ln = int(rng.integers(3, 34))
        cv2.line(arr, (x, y), (x + ln, y + int(rng.integers(-2, 3))),
                 (196, 198, 188), 1)
    return np.clip(arr, 0, 255)


def make_label_front(out):
    w, h = 1024, 1204
    img = Image.new('RGB', (w, h), (244, 244, 238))
    d = ImageDraw.Draw(img)
    green = (26, 116, 52)
    d.rectangle((14, 14, w - 15, h - 15), outline=green, width=3)
    d.rectangle((28, 28, w - 28, 254), outline=green, width=4)
    d.text((52, 40), '欧伦泰', font=lab_font(86, True), fill=green)
    d.text((430, 116), 'Since 1988', font=latin(30, True), fill=green)
    cx, cy, r = 888, 98, 58
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=green, width=4)
    d.text((cx, cy), 'CCC', font=latin(27, True), fill=green, anchor='mm')
    d.text((52, 138), '3L手提式水基型灭火器', font=lab_font(42, True), fill=green)
    d.text((52, 202), 'MPZ / 3', font=latin(25, True), fill=green)
    d.text((730, 188), 'H001827', font=latin(18), fill=green)
    d.line((28, 254, w - 28, 254), fill=green, width=4)
    boxes = [('A', 'A类火  普通固体材料火'), ('B', 'B类火  可燃液体火'),
             ('C', 'C类火  气体和蒸气火'), ('E', 'E类火  带电物质火')]
    for i, (letter, caption) in enumerate(boxes):
        x0, y0 = 32, 272 + i * 160
        x1, y1 = 224, y0 + 146
        d.rectangle((x0, y0, x1, y1), outline=green, width=3)
        d.text((x0 + 8, y0 + 5), letter, font=lab_font(32, True), fill=green)
        draw_icon(d, letter, (x0 + 104, y0 + 74))
        d.text((x0 + 26, y1 - 30), caption, font=lab_font(13), fill=green)
    d.rectangle((244, 272, 992, 930), outline=green, width=4)
    draw_house(d, (334, 354))
    d.text((468, 300), '浙江名牌', font=lab_font(27, True), fill=green)
    d.text((468, 338), '2014(工)-026', font=lab_font(20), fill=green)
    rows = [('1. 拉出保险销', 'Pull out safety pin'),
            ('2. 站在上风处', '喷嘴对准火源根部 / Grip hose nozzle'),
            ('3. 按下压把', '左右摆动灭火 / Press lever to put out fire')]
    for i, (cn, en) in enumerate(rows):
        y = 462 + i * 96
        d.line((262, y - 14, 974, y - 14), fill=green, width=2)
        d.text((288, y), cn, font=lab_font(28, True), fill=green)
        d.text((288, y + 38), en, font=lab_font(18), fill=green)
    d.rectangle((262, 764, 974, 900), outline=green, width=2)
    d.text((274, 770), '注意事项', font=lab_font(22, True), fill=green)
    caution = ['• 使用前检查压力表指针是否在绿色区域。',
               '• 每月检查一次，压力不足时由专业单位充装。',
               '• 存放于干燥通风处，避免暴晒、腐蚀和剧烈碰撞。',
               '• 使用后请及时补充并保持铅封完好。']
    for i, line in enumerate(caution):
        d.text((276, 804 + i * 21), line, font=lab_font(13), fill=green)
    d.text((300, 936), '浙江欧伦泰防火设备有限公司', font=lab_font(27, True),
           fill=green)
    d.text((248, 974), 'ZHEJIANG OULUNTAI FIRE SAFETY EQUIPMENT CO., LTD.',
           font=latin(12), fill=green)
    d.text((248, 996), '地址：浙江省江山市  电话：0574-88126666  www.ouluntai.com',
           font=lab_font(11), fill=green)
    d.text((248, 1024), '产品质量由中国人民财产保险股份有限公司承保',
           font=lab_font(19, True), fill=green)
    d.text((248, 1060), '使用前请仔细阅读本说明', font=lab_font(15), fill=green)
    d.rectangle((850, 1040, 970, 1100), outline=green, width=2)
    for i in range(18):
        x = 856 + i * 6
        d.line((x, 1046, x, 1094), fill=green, width=1 + (i % 2))
    arr = paper_finish(img, 5001)
    base = Image.fromarray(arr.astype(np.uint8), 'RGB')
    base_path = out / 'label_front_base.png'
    base.save(base_path)
    harr, warr = arr.shape[:2]
    n = fbm(harr, warr, 5002, (120, 26), (.65, .35))
    ink = (arr[..., 1] > arr[..., 0] + 18) & (arr[..., 1] > arr[..., 2] + 10)
    rough = .56 + .05 * n
    rough = rough * (1 - ink) + .66 * ink
    metal = np.full_like(rough, .01)
    normal = normal_from_height(.10 * n + .12 * ink.astype(np.float32), .34)
    mr_path = out / 'label_front_mr.png'
    normal_path = out / 'label_front_normal.png'
    save_rgb(mr_path, pack_mr(rough, metal))
    save_rgb(normal_path, normal)
    stats = {'ink_fraction': round(float(ink.mean()), 4),
             'white_fraction': round(float((arr.min(axis=2) > 210).mean()), 4),
             'roughness_mean': round(float(rough.mean()), 4)}
    stats['passed'] = bool(.03 <= stats['ink_fraction'] <= .55 and
                           stats['white_fraction'] > .35)
    return {'base': str(base_path), 'mr': str(mr_path),
            'normal': str(normal_path)}, stats


def make_label_back(out):
    w, h = 768, 1244
    img = Image.new('RGB', (w, h), (244, 244, 238))
    d = ImageDraw.Draw(img)
    green = (26, 116, 52)
    d.rectangle((14, 14, w - 15, h - 15), outline=green, width=3)
    d.rectangle((34, 34, w - 35, h - 35), outline=green, width=3)
    d.text((w // 2, 58), '使用说明', font=lab_font(46, True), fill=green,
           anchor='ma')
    lines = [
        '1. 将灭火器放置在干燥、通风、便于取用处。',
        '2. 使用前检查压力表指针是否在绿色区域。',
        '3. 拉出保险销，握住喷管，站在上风处。',
        '4. 将喷嘴对准火源根部，按下压把。',
        '5. 左右摆动喷嘴，由近及远扑灭火焰。',
        '6. 灭火后注意防止复燃，并保持现场通风。',
        '7. 每月检查一次压力表与铅封是否完好。',
        '8. 压力不足或使用后须由专业单位充装。',
        '9. 本产品为 3L 水基型灭火器，适用于 A/B/E 类火。',
        '10. 禁止将灭火器放置在高温、潮湿或腐蚀环境。',
    ]
    for i, line in enumerate(lines):
        d.text((70, 138 + i * 36), line, font=lab_font(18), fill=green)
    d.rectangle((50, 530, w - 52, 826), outline=green, width=3)
    d.text((66, 542), '注意事项', font=lab_font(27, True), fill=green)
    caution = [
        '• 每月检查压力表指示是否在绿区。',
        '• 保险销必须完好，不得随意拆除。',
        '• 存放环境温度宜在 -10℃ 至 +45℃。',
        '• 避免日晒、雨淋和腐蚀性气体。',
        '• 使用后及时补充，不得私自拆装阀体。',
    ]
    for i, line in enumerate(caution):
        d.text((72, 592 + i * 42), line, font=lab_font(17), fill=green)
    d.line((50, 870, w - 52, 870), fill=green, width=2)
    d.text((70, 892), '生产批号：2024.05.18', font=lab_font(19), fill=green)
    d.text((70, 926), '生产日期：2024.05.18', font=lab_font(19), fill=green)
    d.text((70, 992), '浙江欧伦泰防火设备有限公司', font=lab_font(24, True),
           fill=green)
    d.text((70, 1028), 'ZHEJIANG OULUNTAI FIRE SAFETY EQUIPMENT CO., LTD.',
           font=latin(11), fill=green)
    d.text((70, 1050), '地址：浙江省江山市  电话：0574-88126666', font=lab_font(12),
           fill=green)
    d.text((70, 1100), '产品质量由中国人民财产保险股份有限公司承保',
           font=lab_font(15), fill=green)
    d.text((70, 1150), '请妥善保管本说明', font=lab_font(12), fill=green)
    arr = paper_finish(img, 5201)
    base_path = out / 'label_back_base.png'
    Image.fromarray(arr.astype(np.uint8), 'RGB').save(base_path)
    harr, warr = arr.shape[:2]
    n = fbm(harr, warr, 5202, (110, 24), (.65, .35))
    ink = (arr[..., 1] > arr[..., 0] + 18) & (arr[..., 1] > arr[..., 2] + 10)
    rough = .58 + .05 * n
    rough = rough * (1 - ink) + .68 * ink
    metal = np.full_like(rough, .01)
    normal = normal_from_height(.09 * n + .12 * ink.astype(np.float32), .32)
    mr_path = out / 'label_back_mr.png'
    normal_path = out / 'label_back_normal.png'
    save_rgb(mr_path, pack_mr(rough, metal))
    save_rgb(normal_path, normal)
    stats = {'ink_fraction': round(float(ink.mean()), 4),
             'white_fraction': round(float((arr.min(axis=2) > 210).mean()), 4),
             'roughness_mean': round(float(rough.mean()), 4)}
    stats['passed'] = bool(.02 <= stats['ink_fraction'] <= .45 and
                           stats['white_fraction'] > .35)
    return {'base': str(base_path), 'mr': str(mr_path),
            'normal': str(normal_path)}, stats


def make_gauge_face(out):
    w = h = 512
    img = Image.new('RGB', (w, h), (243, 244, 239))
    d = ImageDraw.Draw(img)
    cx, cy = 256, 256
    red = (204, 44, 38)
    green = (40, 150, 62)
    yellow = (231, 193, 42)
    dark = (28, 34, 30)

    def polar(a_deg, radius):
        a = math.radians(a_deg)
        return (cx + radius * math.sin(a), cy - radius * math.cos(a))

    def arc(a0, a1, color, r0=188, r1=214):
        pts = [polar(a, r1) for a in np.linspace(a0, a1, 40)]
        pts += [polar(a, r0) for a in np.linspace(a1, a0, 40)]
        d.polygon(pts, fill=color)

    arc(-150, -82, red)
    arc(-82, 34, green)
    arc(34, 96, yellow)
    for a in np.arange(-150, 97, 10):
        r0 = 168 if int(a) % 20 == 0 else 174
        d.line((polar(a, r0), polar(a, 184)), fill=dark, width=2)
    d.text(polar(-116, 142), '1.2', font=latin(21, True), fill=dark, anchor='mm')
    d.text(polar(-76, 142), '1.4', font=latin(21, True), fill=dark, anchor='mm')
    d.text(polar(70, 142), '2.5', font=latin(21, True), fill=dark, anchor='mm')
    d.text((372, 328), 'MPa', font=latin(19, True), fill=dark)
    d.text((128, 330), 'MC', font=latin(16, True), fill=red)
    d.text((256, 372), '灭火器用', font=lab_font(17), fill=dark, anchor='mm')
    d.text((256, 402), 'PRESSURE GAUGE', font=latin(10), fill=dark, anchor='mm')
    d.ellipse((cx - 44, cy - 44, cx + 44, cy + 44), outline=(120, 126, 118),
              width=2)
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(5301)
    arr += rng.normal(0, .7, arr.shape)
    base_path = out / 'gauge_face_base.png'
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB').save(base_path)
    rr = np.clip(.22 + .06 * (arr[..., 0] / 255), .15, .40)
    metal = np.full(rr.shape, .01, np.float32)
    mr_path = out / 'gauge_face_mr.png'
    save_rgb(mr_path, pack_mr(rr, metal))
    normal = normal_from_height(.06 * (arr[..., 0] / 255 - .5), .3)
    normal_path = out / 'gauge_face_normal.png'
    save_rgb(normal_path, normal)
    white = float((arr.min(axis=2) > 200).mean())
    colored = float(((arr[..., 0] > arr[..., 1] + 40) |
                     (arr[..., 1] > arr[..., 0] + 30)).mean())
    stats = {'white_fraction': round(white, 4),
             'colored_fraction': round(colored, 4),
             'roughness_mean': round(float(rr.mean()), 4)}
    stats['passed'] = bool(white > .35 and colored > .03 and
                           stats['roughness_mean'] < .45)
    return {'base': str(base_path), 'mr': str(mr_path),
            'normal': str(normal_path)}, stats


def make_yellow_tag(out):
    w = 256
    img = Image.new('RGB', (w, w), (241, 199, 31))
    d = ImageDraw.Draw(img)
    d.rectangle((8, 8, w - 9, w - 9), outline=(24, 24, 22), width=9)
    d.text((w // 2, 42), '灭火器', font=lab_font(28, True),
           fill=(24, 24, 22), anchor='ma')
    d.text((w // 2, 86), '使用方法', font=lab_font(19, True),
           fill=(24, 24, 22), anchor='ma')
    for i in range(5):
        y = 126 + i * 20
        d.line((34, y, w - 34, y), fill=(24, 24, 22), width=2)
    d.text((w // 2, 226), '使用前请阅读', font=lab_font(12),
           fill=(24, 24, 22), anchor='ma')
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(5401)
    arr += rng.normal(0, 1.1, arr.shape)
    base_path = out / 'yellow_tag_base.png'
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB').save(base_path)
    yellow = float(((arr[..., 0] > 180) & (arr[..., 1] > 140) &
                    (arr[..., 2] < 110)).mean())
    black = float((arr.max(axis=2) < 80).mean())
    stats = {'yellow_fraction': round(yellow, 4),
             'border_fraction': round(black, 4)}
    stats['passed'] = bool(yellow > .45 and black > .04)
    return {'base': str(base_path)}, stats


def make_emboss(out):
    w, h = 1024, 320
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = lab_font(86, True)
    text = 'PT  PZ3.0MPa  3L'
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((w - tw) / 2 - bbox[0], (h - th) / 2 - bbox[1]), text, font=font,
           fill=(58, 132, 74, 255))
    arr = np.asarray(img)
    alpha = arr[..., 3]
    path = out / 'emboss_base.png'
    img.save(path)
    stats = {'alpha_coverage': round(float((alpha > 128).mean()), 4)}
    stats['passed'] = bool(.04 <= stats['alpha_coverage'] <= .45)
    return {'base': str(path)}, stats


def build_textures(texture_dir):
    texture_dir.mkdir(parents=True, exist_ok=True)
    specs = {}
    stats = {}
    specs['paint_green_worn'], stats['paint_green_worn'] = make_green_paint(
        texture_dir, 'paint_green_worn', 3101)
    specs['paint_green_clean'], stats['paint_green_clean'] = make_green_paint(
        texture_dir, 'paint_green_clean', 3201, clean=True)
    specs['rubber'], stats['rubber'] = make_rubber(texture_dir)
    specs['chrome'], stats['chrome'] = make_chrome(texture_dir)
    specs['brass'], stats['brass'] = make_brass(texture_dir)
    specs['dark_steel'] = make_dark_steel(texture_dir)
    specs['yellow_plastic'], stats['yellow_plastic'] = make_yellow_plastic(
        texture_dir)
    specs['lime_plastic'] = make_lime_plastic(texture_dir)
    specs['yellow_tag'], stats['yellow_tag'] = make_yellow_tag(texture_dir)
    specs['label_front'], stats['label_front'] = make_label_front(texture_dir)
    specs['label_back'], stats['label_back'] = make_label_back(texture_dir)
    specs['gauge_face'], stats['gauge_face'] = make_gauge_face(texture_dir)
    specs['emboss'], stats['emboss'] = make_emboss(texture_dir)
    labels = {
        'paint_green_worn': ('绿色亮漆钢筒（局部锈蚀与划痕）', .10, .28, .55, .10),
        'paint_green_clean': ('绿色亮漆阀体/把手（轻度磨损）', .10, .26, .52, .10),
        'rubber': ('黑色橡胶软管/喷头', .0, .55, 0, .12),
        'chrome': ('镀铬软管接头/表圈/夹箍', 1.0, .08, 0, .10),
        'brass': ('黄铜阀座/接头', 1.0, .25, 0, .12),
        'dark_steel': ('深色钢针/螺钉/喷口内衬', .88, .44, 0, .12),
        'yellow_plastic': ('黄色保险封条', .0, .36, .08, .12),
        'lime_plastic': ('荧光绿保险环', .0, .32, .05, .12),
        'yellow_tag': ('黄色悬挂说明牌', .0, .38, .05, .12),
        'label_front': ('正面白色/绿色印刷标贴（重绘近似）', .0, .58, 0, .12),
        'label_back': ('背面使用说明标贴（重绘近似）', .0, .58, 0, .12),
        'gauge_face': ('压力表表盘', .0, .28, .30, .10),
        'emboss': ('筒身下部压印文字（近似）', .05, .36, 0, .12),
        'glass': ('压力表玻璃（折射近似）', .0, .035, 0, .05),
    }
    normal_strength = {
        'paint_green_worn': .50, 'paint_green_clean': .24, 'rubber': .55,
        'chrome': .22, 'brass': .38, 'dark_steel': .30,
        'yellow_plastic': .38, 'lime_plastic': .35, 'yellow_tag': .30,
        'label_front': .30, 'label_back': .30, 'gauge_face': .25,
    }
    materials = {}
    for key, spec in specs.items():
        label, metallic, roughness, coat, coat_rough = labels[key]
        entry = dict(spec)
        entry.update({'label': label, 'metallic': metallic,
                      'roughness': roughness, 'coat': coat,
                      'coat_roughness': coat_rough,
                      'alpha': key == 'emboss',
                      'normal_strength': normal_strength.get(key, .25)})
        materials[key] = entry
    materials['glass'] = {'label': labels['glass'][0], 'glass': True,
                          'metallic': 0.0, 'roughness': .035,
                          'alpha': False}
    passed = all(
        stats[k].get('passed', True) for k in
        ('paint_green_worn', 'paint_green_clean', 'rubber', 'chrome', 'brass',
         'yellow_plastic', 'yellow_tag', 'label_front', 'label_back',
         'gauge_face', 'emboss'))
    stats['passed'] = bool(passed)
    stats['note'] = ('贴图由本件代码程序化生成；绿色漆面、锈斑、标签文字和表盘'
                     '均为按原照重绘的近似，不是实测材料图或原始印刷文件。')
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


def make_six_view_board(out):
    render_dir = out / 'renders'
    ref = Image.open(local_paths()['reference']).convert('RGB').resize((1536, 1024))
    ref_panel = panel(ref, 700, '六视图参考',
                      'visual_match_unverified，未标定；把手角度与照片略有差异，以照片为准')
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
    draw.text((22, 18), '34 灭火器1 · 六方向新旧对照', font=font(24, True),
              fill=(30, 44, 58))
    draw.text((22, 48), '参考未标定，仅作外观/背标对照；新版按原照可见结构重建，'
                        '旧模型为 TripoSR 单体壳。',
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


def make_boards(out, texture_stats):
    render_dir = out / 'renders'
    make_six_view_board(out)

    details = ('gauge', 'handles', 'hose-top', 'nozzle', 'label-front',
               'label-back', 'neck', 'bottom', 'overall')
    labels = {'gauge': '压力表/表圈', 'handles': '上压把/下把手与开口',
              'hose-top': '软管上接头', 'nozzle': '喷头/出口',
              'label-front': '正面标贴', 'label-back': '背面说明',
              'neck': '瓶颈/阀座', 'bottom': '底圈/底面', 'overall': '整体 3/4'}
    panels = []
    for name in details:
        panels.append(panel(load_rgba(render_dir / f'detail-{name}.png'), 360,
                            f'新版 · {labels[name]}'))
    for name in details:
        panels.append(panel(load_rgba(render_dir / f'old-detail-{name}.png'), 360,
                            f'旧模型 · {labels[name]}'))
    compose(out / '关键局部-新旧对照.jpg', panels, 4,
            '34 灭火器1 · 关键局部新旧对照',
            '旧模型局部按旧网格主轴近似定位，仅示融合/糊化，不代表同部位精确对齐。')

    isolation = [
        ('new-front.png', '新版整体'), ('isolate-gauge.png', '压力表隔离'),
        ('isolate-handles.png', '把手/保险隔离'), ('isolate-hose.png', '软管/喷头隔离'),
        ('isolate-labels.png', '标贴/压印隔离'), ('isolate-metal.png', '金属件隔离'),
    ]
    panels = [panel(load_rgba(render_dir / name), 420, label)
              for name, label in isolation]
    compose(out / '结构开口-隔离检查.jpg', panels, 3,
            '34 灭火器1 · 结构、开口与材质隔离检查',
            '隔离渲染用于检查把手开口、保险环、软管/喷头与标贴；贯穿/盲孔由射线检查确认。')

    panels = [
        panel(load_rgba(render_dir / 'new-material-3q.png'), 480, '新版 · 同光照 3/4'),
        panel(load_rgba(render_dir / 'old-material-3q.png'), 480, '旧模型 · 同光照 3/4'),
        panel(load_rgba(render_dir / 'new-clay-front.png'), 480, '新版 · 灰模正视图'),
        panel(load_rgba(render_dir / 'new-front.png'), 480, '新版 · 材质正视图'),
    ]
    compose(out / '材质-同光照对照.jpg', panels, 2,
            '34 灭火器1 · 材质与灰模对照',
            '绿色漆面为电介质+清漆；镀铬、黄铜和深色钢为金属。灰模不参与外观验收。')

    panels = [
        panel(load_rgba(render_dir / 'new-front.png'), 480, '新版 · 主光 A'),
        panel(load_rgba(render_dir / 'new-light-alternate.png'), 480, '新版 · 主光 B'),
        panel(load_rgba(render_dir / 'metal-only-light-a.png'), 480, '金属隔离 · 主光 A'),
        panel(load_rgba(render_dir / 'metal-only-light-b.png'), 480, '金属隔离 · 主光 B'),
    ]
    compose(out / '表面反射-变光对照.jpg', panels, 2,
            '34 灭火器1 · 变光与金属反射对照',
            '高光随主光位置移动；只证明金属镜面反射变化，不证明反射率实测或场景光照匹配。')

    panels = []
    for i in range(24):
        angle = i * 15
        panels.append(panel(load_rgba(render_dir / f'turn-new-{i:02d}.png'), 250,
                            f'{angle:03d}°'))
    compose(out / '转台-24方向.jpg', panels, 6,
            '34 灭火器1 · 新版 24 方向转台',
            '每 15° 一张 Blender 实测渲染；覆盖外观，不证明隐藏结构或装配。')

    src = load_rgba(local_paths()['original'])
    rust = load_rgba(local_paths()['rust'])
    ref_small = Image.open(local_paths()['reference']).convert('RGB')
    panels = [
        panel(src, 480, '原照（原貌）'), panel(rust, 480, '做锈图'),
        panel(load_rgba(render_dir / 'new-source-3q.png'), 480, '新版 · 近似原照视角'),
        panel(load_rgba(render_dir / 'old-source.png'), 480, '旧模型 · 库标定相机'),
        panel(ref_small, 480, '六视图参考（未标定）'),
        panel(load_rgba(render_dir / 'new-front.png'), 480, '新版 · 正视图'),
    ]
    compose(out / '原照-新候选-参考.jpg', panels, 3,
            '34 灭火器1 · 原照、旧模型、新候选与参考',
            '原照为近似正面产品图且无标定；新版视角为近似对照，IoU 只作诊断。')

    tex_items = [
        ('paint_green_worn_base.png', '筒体绿色底色'),
        ('paint_green_worn_mr.png', '筒体金属/粗糙度'),
        ('paint_green_worn_normal.png', '筒体法线'),
        ('rubber_base.png', '橡胶底色'),
        ('chrome_mr.png', '镀铬金属/粗糙度'),
        ('brass_mr.png', '黄铜金属/粗糙度'),
        ('label_front_base.png', '正面标贴底色'),
        ('gauge_face_base.png', '压力表盘'),
        ('yellow_tag_base.png', '黄色说明牌'),
    ]
    panels = [panel(load_rgba(out / 'textures' / name), 360, label)
              for name, label in tex_items]
    compose(out / 'PBR贴图-对照.jpg', panels, 3,
            '34 灭火器1 · PBR 贴图检查',
            '程序化贴图并打包进 GLB；金属/粗糙度通道为 R=255、G=粗糙度、B=金属度。'
            '锈斑、标签与表盘为外观仿真，不是实测材料图或原始印刷文件。')


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def original_alpha(paths):
    return np.asarray(Image.open(paths['original']).convert('RGBA'))[..., 3]


def rotated_source_mask(paths):
    alpha = original_alpha(paths)
    ys, xs = np.nonzero(alpha >= 128)
    points = np.column_stack([xs, ys]).astype(np.float32)
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    angle = math.degrees(math.atan2(major[1], major[0]))
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
    # Front panel of the 3x2 sheet; discard the caption strip at the bottom.
    panel = image[0:470, 0:512]
    gray = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)
    mask = (gray < 232).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def source_diagnostics(out, paths):
    raw_source, rotated, rotation_deg = rotated_source_mask(paths)
    render_dir = out / 'renders'
    new_source = np.asarray(load_rgba(render_dir / 'new-source-alpha.png').getchannel('A'))
    old_source = np.asarray(load_rgba(render_dir / 'old-source-alpha.png').getchannel('A'))
    strict_source = np.asarray(Image.open(paths['source_alpha']).convert('L'))
    if float(strict_source.std()) < 1e-3:
        strict = {
            'iou': None, 'available': False, 'acceptance_gate': False,
            'reason': ('库内 source-alpha 为全不透明通道，不能作为标定轮廓；'
                       '严格 IoU 未计算。'),
        }
        old_strict = dict(strict)
    else:
        strict = source_fidelity(strict_source, new_source)
        strict['available'] = True
        strict['acceptance_gate'] = False
        strict['scale_note'] = '手工近似原照相机；未标定，严格 IoU 只作方位诊断'
        old_strict = source_fidelity(strict_source, old_source)
        old_strict['available'] = True
        old_strict['acceptance_gate'] = False
    # A camera-consistent, but still uncalibrated, shape comparison: use the
    # library's old source-view foreground (white background) against both
    # source renders.  This tests the new silhouette against the old model's
    # own calibrated camera, not against the original photograph.
    source_view = np.asarray(Image.open(paths['source_view']).convert('RGB'))
    view_mask = (source_view.min(axis=2) < 235).astype(np.uint8) * 255
    view_mask = cv2.morphologyEx(view_mask, cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(view_mask, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        view_mask = np.isin(labels, [keep]).astype(np.uint8) * 255
    old_view = source_fidelity(view_mask, old_source)
    old_view.update({'alignment': '旧模型源视图前景与旧模型同相机渲染；仅诊断',
                     'acceptance_gate': False})
    new_view = source_fidelity(view_mask, new_source)
    new_view.update({'alignment': '旧模型源视图前景与新模型同相机渲染；仅诊断',
                     'acceptance_gate': False})
    source_box = silhouette_iou(raw_source, new_source)
    if source_box:
        source_box['alignment'] = '原照透明轮廓与新源渲染按包围盒缩放/平移；不是标定'
        source_box['acceptance_gate'] = False
    old_source_box = silhouette_iou(raw_source, old_source)
    if old_source_box:
        old_source_box['alignment'] = '原照透明轮廓与旧源渲染按包围盒缩放/平移；不是标定'
        old_source_box['acceptance_gate'] = False
    new_front = np.asarray(load_rgba(render_dir / 'sil-new-00.png').getchannel('A'))
    rotated_front = silhouette_iou(rotated, new_front)
    if rotated_front:
        rotated_front['alignment'] = ('原照轮廓按主轴旋转后与新模型正视做包围盒对齐；'
                                      '仅诊断，不是标定')
        rotated_front['acceptance_gate'] = False
    reference = silhouette_iou(reference_front_mask(paths), new_front)
    if reference:
        reference['alignment'] = '未标定六视图参考正视图；仅诊断'
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
        'source_alpha_all_opaque': bool(float(strict_source.std()) < 1e-3),
        'source_view': str(paths['source_view'].relative_to(ROOT)),
        'strict': strict,
        'old_strict': old_strict,
        'old_source_view_fit': old_view,
        'new_source_view_fit': new_view,
        'source_box_fit': source_box,
        'old_source_box_fit': old_source_box,
        'source_rotated_front_fit': rotated_front,
        'reference_front_fit': reference,
        'source_vertical_rotation_deg': rotation_deg,
        'best_view': best[0] if best else 0,
        'best': best_metrics,
        'note': ('所有 IoU 均为未标定诊断。严格同相机 IoU 使用库内 '
                 'source-alpha 的旧模型源视图轮廓；其余为包围盒/主轴对齐诊断。'
                 '原照为近似正面产品图且软管/把手改变轮廓，不能当作摄影测量。'),
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
    result['note'] = ('只证明未涂装的镀铬/黄铜/深色钢高光随灯位移动；'
                      '绿色漆面、橡胶、塑料和纸张为电介质，本检查不适用于它们，'
                      '也不证明环境光匹配。')
    return result


def image_luma_stats(out):
    stats = {}
    for name in ('new-front.png', 'old-front.png', 'new-material-3q.png',
                 'new-clay-front.png'):
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
    green = np.logical_and.reduce([mask, hsv[:, :, 0] >= 35, hsv[:, :, 0] <= 90,
                                   hsv[:, :, 1] >= 70, hsv[:, :, 2] >= 40])
    rust_image = Image.open(paths['rust']).convert('RGB').resize(
        (arr.shape[1], arr.shape[0]), Image.LANCZOS)
    rust = np.asarray(rust_image).astype(float)
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
        'green_fraction': round(float(green.sum() / mask.sum()), 4),
        'rust_color_shift': {
            'mean_rgb_distance': round(float(diff[mask].mean()), 3),
            'p50': round(float(np.percentile(diff[mask], 50)), 3),
            'p90': round(float(np.percentile(diff[mask], 90)), 3),
            'brown_fraction': round(float(brown.sum() / mask.sum()), 4)},
        'configured': {
            'overall_height_m': round(DIMENSIONS['overall']['top'] -
                                      DIMENSIONS['overall']['bottom'], 4),
            'body_diameter_m': round(2 * DIMENSIONS['body']['radius'], 4),
            'body_height_m': round(DIMENSIONS['body']['z1'] -
                                   DIMENSIONS['body']['z0'], 4),
            'overall_over_diameter': round(
                (DIMENSIONS['overall']['top'] - DIMENSIONS['overall']['bottom']) /
                (2 * DIMENSIONS['body']['radius']), 3),
            'gauge_diameter_m': round(2 * DIMENSIONS['gauge']['bezel_r'], 4),
            'hose_outer_diameter_m': round(2 * DIMENSIONS['hose']['radius'], 4),
            'handle_gap_m': round(0.633 - 0.498, 4),
        },
        'note': ('工作尺度按原照比例设置，不是实测。minAreaRect 描述的是照片中的'
                 '可见轮廓，包含透视与软管伸出。'),
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
                 '0.9363、填孔 0，但六方向图仍显示把手、表、软管融合，表面拉伸，'
                 '不能作为本件结构精度或材质依据。'),
    }


def opening_checks(worker):
    rays = worker.get('ray_checks', {})
    handle = rays.get('handle', {})
    safety = rays.get('safety', {})
    hose = rays.get('hose', {})
    gauge = rays.get('gauge', {})
    objects = rays.get('objects', {})
    checks = {
        'handle_gap': bool(handle.get('gap_clear')),
        'handle_lower_solid': bool(handle.get('lower_solid')),
        'handle_upper_solid': bool(handle.get('upper_solid')),
        'pivot_solid': bool(handle.get('pivot_solid')),
        'safety_ring_hole': bool(safety.get('hole_clear')),
        'safety_ring_solid': bool(safety.get('ring_solid')),
        'tamper_strap': bool(safety.get('tamper_present')),
        'yellow_tag': bool(safety.get('tag_present')),
        'hose_solid': bool(hose.get('hose_solid')),
        'hose_coupling_solid': bool(hose.get('coupling_solid')),
        'nozzle_body_solid': bool(hose.get('nozzle_body_solid')),
        'nozzle_outlet_ring': bool(hose.get('outlet_ring_solid')),
        'nozzle_blind_plug': bool(hose.get('dark_plug_blind')),
        'nozzle_hole_clear': bool(hose.get('plug_ray_clear_outlet')),
        'gauge_glass': bool(gauge.get('glass_hit')),
        'gauge_bezel': bool(gauge.get('bezel_hit')),
        'gauge_dial': bool(gauge.get('dial_present')),
        'gauge_needle': bool(gauge.get('needle_present')),
        'front_label': bool(objects.get('front_label')),
        'back_label': bool(objects.get('back_label')),
        'emboss_label': bool(objects.get('emboss_label')),
    }
    checks['passed'] = all(checks.values())
    checks['gap_blocker'] = handle.get('gap_blocker')
    checks['note'] = ('射线检查覆盖把手开口、保险环孔、软管/接头、喷头出口盲孔、'
                      '压力表与标贴；不代表阀门内部结构或软管耐压。')
    return checks


# ---------------------------------------------------------------------------
# Text report, page, hashes
# ---------------------------------------------------------------------------
def write_report(out, result):
    checks = result['checks']
    diagnostics = result['diagnostics']
    openings = checks['openings']
    strict_text = (diagnostics['strict'].get('iou')
                   if diagnostics['strict'].get('iou') is not None
                   else '不可用（库 source-alpha 全不透明）')
    lines = [
        '# 34 灭火器1 · 分部件重建候选（manual_only，待审）',
        '',
        f"- 版本：`{VERSION}`",
        '- 原照：绿色欧伦泰 MPZ/3 3L 水基型灭火器，绿色钢筒、绿色阀体/把手、'
        '镀铬软管接头、圆形压力表、荧光绿保险环、黄色封条/说明牌、黑色软管与喷头。',
        '- 六视图角色：`visual_match_unverified`，未标定；参考与照片的把手角度和'
        '局部细节不一致，以照片为准，参考只作外观/背标对照。',
        '- 旧模型：TripoSR 单体壳，把手/表/软管融合、标贴拉伸、底圈不清，未复用；'
        '仅渲染对照。',
        '',
        '## 已确认与推断',
        '',
        '已确认（原照可见）：绿色亮漆筒体与卷边底圈、圆肩/短颈、绿色阀体与上下把手、'
        '把手之间的真实开口、圆形压力表（红/绿/黄区）、荧光绿保险环、黄色封条与'
        '悬挂说明牌、镀铬软管接头和表圈、黑色橡胶软管及下部黑色喷头、正面品牌标贴'
        '和背面使用说明、筒身下部压印文字。',
        '',
        '推断/未确认：把手精确截面与角度、阀门内部结构、保险销机构、软管与阀体的'
        '内部连接、喷头内部通道、软管卡箍形式、表盘刻度/文字的准确内容、标签原始'
        '字体与版式、背面和底面隐藏细节、真实尺寸与材料牌号、落稳姿态与现场尺度。',
        '',
        '## 实际执行的检查',
        '',
        f"- 把手开口：{'通过' if openings['handle_gap'] else '未通过'}；"
        f"上下把手/销轴实体："
        f"{sum(openings[k] for k in ('handle_lower_solid', 'handle_upper_solid', 'pivot_solid'))}/3。",
        f"- 保险环孔：{'通过' if openings['safety_ring_hole'] else '未通过'}；"
        f"环体/封条/黄牌："
        f"{sum(openings[k] for k in ('safety_ring_solid', 'tamper_strap', 'yellow_tag'))}/3。",
        f"- 软管/接头/喷头：软管 "
        f"{'通过' if openings['hose_solid'] else '未通过'}，接头 "
        f"{'通过' if openings['hose_coupling_solid'] else '未通过'}，喷头 "
        f"{'通过' if openings['nozzle_body_solid'] else '未通过'}，出口环 "
        f"{'通过' if openings['nozzle_outlet_ring'] else '未通过'}，盲孔 "
        f"{'通过' if openings['nozzle_blind_plug'] else '未通过'}。",
        f"- 压力表：玻璃 {'通过' if openings['gauge_glass'] else '未通过'}，"
        f"表圈 {'通过' if openings['gauge_bezel'] else '未通过'}，"
        f"表盘 {'通过' if openings['gauge_dial'] else '未通过'}，"
        f"指针 {'通过' if openings['gauge_needle'] else '未通过'}。",
        f"- 标贴：正面 {'通过' if openings['front_label'] else '未通过'}，"
        f"背面 {'通过' if openings['back_label'] else '未通过'}，"
        f"压印 {'通过' if openings['emboss_label'] else '未通过'}。",
        f"- 部件网格闭合：{checks['mesh']['closed_parts']}/"
        f"{checks['mesh']['parts']}；同部件自交："
        f"{len(checks['mesh']['self_intersecting'])} 项。",
        f"- 贴图数值检查：{'通过' if checks['textures']['passed'] else '未通过'}。",
        f"- 金属高光变光：{'通过' if checks['metal']['passed'] else '未通过'}；"
        f"质心位移 {checks['metal'].get('highlight_shift_px')} px。",
        f"- 原照轮廓诊断（未标定，非验收）：严格同相机 IoU "
        f"{strict_text}，包围盒对齐 IoU "
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
              f"- 检查页：`{out / 'review-34.html'}`",
              '', '候选不替换原模型，不进入自动放置或正式合成。']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf8')


def write_review_page(out, result):
    d = result['diagnostics']
    pending = ''.join(f'<li>{item}</li>' for item in result['pending_review'])
    o = result['checks']['openings']
    strict_text = (d['strict'].get('iou')
                   if d['strict'].get('iou') is not None
                   else '不可用（库 source-alpha 全不透明）')
    metrics = [
        ('把手开口 / 实体', f"{'通过' if o['handle_gap'] else '未通过'} · "
         f"{sum(o[k] for k in ('handle_lower_solid', 'handle_upper_solid', 'pivot_solid'))}/3"),
        ('保险环孔 / 环体', f"{'通过' if o['safety_ring_hole'] else '未通过'} · "
         f"{sum(o[k] for k in ('safety_ring_solid', 'tamper_strap', 'yellow_tag'))}/3"),
        ('软管/接头/喷头', f"{sum(o[k] for k in ('hose_solid', 'hose_coupling_solid', 'nozzle_body_solid', 'nozzle_outlet_ring', 'nozzle_blind_plug'))}/5"),
        ('压力表', f"{sum(o[k] for k in ('gauge_glass', 'gauge_bezel', 'gauge_dial', 'gauge_needle'))}/4"),
        ('标贴', f"{sum(o[k] for k in ('front_label', 'back_label', 'emboss_label'))}/3"),
        ('网格闭合部件', f"{result['checks']['mesh']['closed_parts']}/"
         f"{result['checks']['mesh']['parts']}"),
        ('同部件自交', f"{len(result['checks']['mesh']['self_intersecting'])} 项"),
        ('PBR 贴图数值', '通过' if result['checks']['textures']['passed'] else '未通过'),
        ('金属变光高光位移',
         f"{result['checks']['metal'].get('highlight_shift_px')} px · "
         f"{'通过' if result['checks']['metal'].get('passed') else '未通过'}"),
        ('原照轮廓诊断（未标定）', f"严格同相机 {strict_text} · 包围盒 "
         f"{(d.get('source_box_fit') or {}).get('iou', '未执行')} · 主轴正视 "
         f"{(d.get('source_rotated_front_fit') or {}).get('iou', '未执行')}"),
        ('24 方向最优轮廓诊断', f"{d['best'].get('iou', '未执行')} "
         f"({d.get('best', {}).get('view', '未执行')})"),
        ('旧模型库审计', 'IoU 0.9363，填孔 0；但把手/表/软管融合、表面拉伸'),
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
<title>34 灭火器1 · 分部件重建检查（manual_only）</title>
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
<header><h1>34 灭火器1 · 分部件重建 {VERSION} <span class="tag">manual_only · 待审</span></h1>
<p>绿色 3L 水基型灭火器 · 筒体/阀体/上下把手/压力表/软管喷头/标贴分部件重建 · 离线检查页</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b>本候选不替换原模型，不进入自动放置或正式合成。
六视图参考未标定且与原照把手角度/局部细节不完全一致；所有 IoU 只作诊断，不是验收通过。
背面、阀内结构、软管内部、标签原始印刷、真实尺寸、落稳与现场光照均待人工复核。
浏览器运行时交互未执行；本页的旋转视图是 Blender 实测渲染帧。</div></section>

<section><h2>旋转检查</h2><div class="viewer">
<div class="turnbox" id="turnbox"><img id="turn" src="renders/turn-new-00.png" alt="转台"></div>
<aside><p>拖动图片旋转，或使用下方按钮。每 15° 一帧，共 24 帧。</p>
<p><button data-mode="new" class="sel">新版</button><button data-mode="old">旧模型</button><button data-mode="clay">灰模</button></p>
<p><button id="prev">上一帧</button><button id="next">下一帧</button><button id="play">自动旋转</button><button id="reset">重置</button></p>
<p>当前角度：<b id="angle">000°</b> · 模式：<b id="modeLabel">新版</b></p>
<p><a href="mesh.glb">下载 GLB</a><br><a href="model.blend">下载 Blender 文件</a></p></aside></div></section>

<section><h2>对照图</h2><div class="grid">{cards}</div></section>

<section><h2>检查记录</h2><table>{metric_rows}</table>
<p class="notice" style="margin-top:12px">未执行项明确写为“未执行”，不能解读为通过。金属变光只证明未涂装金属的高光移动；
绿色漆面/橡胶/塑料/纸张不适用金属反射检查。开口射线只覆盖实际建出的开口和零件。</p></section>

<section><h2>待人工复核</h2><ul>{pending}</ul></section>

<section><h2>文件与记录</h2><p><a href="result.json">result.json</a> ·
<a href="hashes.json">hashes.json</a> · <a href="worker-result.json">Blender worker 记录</a> ·
<a href="report.md">中文报告</a> · <a href="source-measurements.json">原照测量</a> ·
<a href="old-model-audit.json">旧模型审计</a></p>
<p>贴图：<a href="textures/paint_green_worn_base.png">筒体绿色底色</a> ·
<a href="textures/paint_green_worn_mr.png">筒体金属/粗糙度</a> ·
<a href="textures/chrome_mr.png">镀铬金属/粗糙度</a> ·
<a href="textures/label_front_base.png">正面标贴</a></p>
<p>候选 <code>{VERSION}</code>，参考角色 <code>visual_match_unverified</code>；原模型和共享页面未被修改。</p></section>
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
    (out / 'review-34.html').write_text(html, encoding='utf8')


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
        'next_number': 35,
    }
    target = ROOT / 'work/direct-deepseek/34-progress.json'
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
        parser.error('this entry supports item 34 only')
    if args.draft:
        RENDER.update({'six_samples': 10, 'detail_samples': 24,
                       'turn_samples': 4, 'turn_count': 24})
    if args.out is None:
        args.out = (ROOT / 'work/direct-deepseek/34-draft' if args.draft
                    else ROOT / 'outputs/34灭火器1-六视图重建-v1')
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
                          'fov_deg': 38.102696, 'shift_x': -0.002019,
                          'shift_y': -0.002529},
        'original_mesh': str(paths['old_mesh']),
        'config': {'dimensions': DIMENSIONS, 'render': RENDER},
        'materials': materials,
        'textures': materials,
        'draft': bool(args.draft),
    }
    (out / 'request.json').write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
    worker = HERE / 'mesh_rebuild_34_worker.py'
    cmd = [str(args.blender), '-b', '--factory-startup', '--python', str(worker),
           '--', str(out / 'request.json')]
    print('rendering with Blender ...', flush=True)
    with open(out / 'blender.log', 'w', encoding='utf8') as log:
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), timeout=10800)
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
        '六视图为未标定的外观参考，且把手角度/局部细节与原照不完全一致；'
        '背面与隐藏结构为推断，以原照为准。',
        '把手精确截面、阀门内部结构、保险销机构与铅封形式未验证。',
        '软管与阀体内部连接、软管卡箍形式、喷头内部通道与耐压性能未验证。',
        '压力表刻度/文字与绿色/黄色区间的准确值、标签原始字体/版式、'
        '筒身压印文字的完整内容为近似重绘。',
        '真实尺寸、材料牌号、落稳姿态、现场尺度和合成环境光未验证。',
        '原照轮廓 IoU 为未标定诊断，不能当作精度或验收结论。',
        '浏览器运行时交互未执行；离线页只使用 Blender 渲染帧。',
    ]
    result = {
        'number': NUMBER,
        'name': '灭火器1（绿色 3L 水基型灭火器）',
        'version': VERSION,
        'status': 'manual_only_pending_review',
        'manual_only': True,
        'replaces_library_mesh': False,
        'enters_automatic_placement': False,
        'reference': {
            'role': 'visual_match_unverified',
            'file': str(paths['reference'].relative_to(ROOT)),
            'sha256': digest(paths['reference']),
            'note': ('绿色 3L 水基型灭火器六视图；未标定，把手角度与部分细节'
                     '和原照不同，以原照为准，参考只作外观与背标对照。'),
        },
        'source': {'original': str(paths['original'].relative_to(ROOT)),
                   'rust': str(paths['rust'].relative_to(ROOT)),
                   'original_sha256': digest(paths['original']),
                   'rust_sha256': digest(paths['rust'])},
        'method': ('按原照可见结构分部件重建：卷边筒体、圆肩短颈、阀体、上下把手与'
                   '真实开口、压力表、保险环/封条/黄牌、软管与喷头、正/背标贴；'
                   '程序化 PBR 贴图与离线检查页。'),
        'confirmed': [
            '绿色亮漆筒体与卷边底圈', '圆肩/短颈/阀体', '上下把手与挤压开口',
            '圆形压力表（红/绿/黄区）', '荧光绿保险环与黄色封条/说明牌',
            '镀铬软管上接头与表圈', '黑色橡胶软管与下部喷头',
            '正面品牌标贴与背面使用说明', '筒身下部压印文字', '绿色漆面局部磨损'],
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
            'page': str(out / 'review-34.html'),
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
    checks_path = HERE / 'checks_rebuild_34.py'
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
                               'see checks-34.json')
    print('done')


if __name__ == '__main__':
    main()
