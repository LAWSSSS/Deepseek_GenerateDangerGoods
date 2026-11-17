"""Build the review-only item-39 vertical oil-receiver candidate.

The clean/rust photograph shows a used teal HPEOK PHO-10 vertical oil receiver
(10 L / 3.7 gal label).  The user six-view sheet is an uncalibrated visual match
of the same model family and is used for topology only.  The candidate is a
part-based rebuild, never replaces the library mesh, stays manual-only and never
enters automatic placement.  All diagnostic IoU values are uncalibrated.
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
import PIL
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes

from mesh_carve import interior_holes
from mesh_checks import source_fidelity
from mesh_rebuild_item import mask_fit, inspect_mesh
from review39_page import write_page

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NUMBER = 39
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
REFERENCE_NAME = '六视角青绿色压力油罐产品图лириға.png'
REFERENCE_PATH = ROOT / 'references/parallel-six-view' / REFERENCE_NAME
REFERENCE_SHA256 = '33be479f04a8362dd478c99d06b7887b4cf39821b4327ff861a6284cc489fbfd'
SOURCE_SIZE = (1173, 1341)
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
CHROME = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
OUT_DIR = ROOT / 'outputs/39集油器2-六视图重建-v1'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def resize_noise(rng, shape, out_shape, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(rng.random(shape).astype(np.float32), out_shape,
                      interpolation=interpolation)


def save_maps(out, prefix, rgb, rough, metal, bump, normal_strength=1.0, flip=True):
    h, w = rgb.shape[:2]
    bump = np.asarray(bump, np.float32)
    smooth = cv2.GaussianBlur(bump, (0, 0), 1.0)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx = np.clip(-dx * normal_strength, -.35, .35)
    ny = np.clip(dy * normal_strength, -.35, .35)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    base = np.clip(rgb, 0, 255).astype(np.uint8)
    mr = np.stack((np.full((h, w), 255, np.uint8),
                   (np.clip(rough, 0, 1) * 255).astype(np.uint8),
                   (np.clip(metal, 0, 1) * 255).astype(np.uint8)), axis=2)
    if flip:
        base, mr, normal = base[::-1], mr[::-1], normal[::-1]
    paths = {}
    for key, arr in (('base', base), ('mr', mr), ('normal', normal)):
        path = out / f'{prefix}_{key}.png'
        Image.fromarray(arr).save(path)
        paths[f'{prefix}_{key}'] = str(path)
    return paths


def source_colours(rust_path):
    image = Image.open(rust_path)
    rgb = np.asarray(image.convert('RGB')).astype(np.float32)
    alpha = np.asarray(image.getchannel('A')) >= 128
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    teal = alpha & (g > 55) & (b > 55) & (g > r * 1.25) & (r < 110)
    rust = alpha & (r > 62) & (r > g + 15) & (r > b + 22) & (g < 145)
    teal_med = np.median(rgb[teal], axis=0) if teal.any() else np.array([18, 133, 133])
    rust_med = np.median(rgb[rust], axis=0) if rust.any() else np.array([118, 72, 42])
    return {'teal': teal_med, 'rust': rust_med, 'teal_pixels': int(teal.sum()),
            'rust_pixels': int(rust.sum())}


def make_body_textures(out, rust_path, profile):
    h, w = 1024, 2048
    rng = np.random.default_rng(39039)
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx + .5) / w
    v = (yy + .5) / h
    coarse = resize_noise(rng, (40, 80), (w, h))
    mid = resize_noise(rng, (160, 320), (w, h))
    fine = resize_noise(rng, (h, w), (w, h), cv2.INTER_LINEAR)
    detail = resize_noise(rng, (420, 840), (w, h), cv2.INTER_LINEAR)
    colours = source_colours(rust_path)
    teal = colours['teal'] * np.array([1.00, 1.075, 1.075])
    rust = np.clip(colours['rust'] * np.array([1.28, .87, .70]), 0, 255)
    # Cylindrical UV collapses at the poles; fade fine detail and drip-like
    # patterns over the domes so the top/bottom views do not show radial bands.
    dome_fade = np.clip(np.minimum(v / .19, (1 - v) / .19), 0, 1)
    dome_fade = dome_fade * dome_fade * (3 - 2 * dome_fade)
    paint = teal[None, None, :] * (1.0 + .060 * (coarse[..., None] - .5)
                                   + .028 * (mid[..., None] - .5)
                                   + .014 * dome_fade[..., None] * (detail[..., None] - .5))
    # Soft vertical shading.
    paint *= (0.95 + 0.07 * v[..., None])
    seam = np.zeros((h, w), np.float32)
    for seam_v in (0.119, 0.908):
        seam += np.exp(-((v - seam_v) / .020) ** 2)
    seam = np.clip(seam, 0, 1)
    front = np.exp(-((u - .5) / .16) ** 2)
    patch = np.clip((coarse - .54) * 3.2, 0, 1) * np.clip((mid - .38) * 2.0, 0, 1)
    streak = np.zeros((h, w), np.float32)
    for u0, v0, length, strength in (
            (.500, .790, .18, .85), (.500, .280, .22, .75),
            (.452, .790, .11, .34), (.548, .280, .12, .30),
            (.500, .615, .13, .45), (.422, .520, .09, .24), (.578, .450, .10, .24)):
        streak = np.maximum(streak, np.exp(-((u - u0) / .012) ** 2)
                            * np.clip((v0 - v) / length, 0, 1) * (v <= v0) * strength)
    random_streak = resize_noise(rng, (24, 96), (w, h))
    streak = np.clip(streak + np.clip(random_streak - .65, 0, 1) * .55, 0, 1)
    streak *= dome_fade
    label_edge = (np.exp(-((np.abs(u - .5) - .108) / .016) ** 2)
                  * np.exp(-((np.abs(v - .540) - .073) / .020) ** 2)) * dome_fade
    rust_mask = np.clip(.95 * seam + .42 * patch * (0.40 + .60 * front)
                        + .62 * streak + .38 * label_edge, 0, 1) ** 1.24
    rust_mask = np.clip(rust_mask * .82, 0, .82)
    rust_colour = rust[None, None, :] * (1.0 + .16 * (mid[..., None] - .5)
                                        + .08 * (fine[..., None] - .5))
    dark_rust = np.array(profile['rust_dark_rgb'], np.float32)
    rust_colour = rust_colour * (1 - .30 * patch[..., None]) + \
        dark_rust[None, None, :] * (.30 * patch[..., None])
    grime = np.clip(.50 * seam + .30 * streak + .14 * (1 - v) * front, 0, .6) * .45
    grime_colour = np.array([54, 48, 38], np.float32)
    # The cylindrical projection collapses all azimuths at the poles.  Average
    # azimuth-dependent fields over the domes so the top/bottom views show a
    # uniform painted surface instead of radial wedges; the seam rings remain
    # since they depend only on v.
    dome_weight = (1 - dome_fade)[..., None]

    def azimuth_mean(field):
        if field.ndim == 3:
            return np.repeat(field.mean(axis=1, keepdims=True), w, axis=1)
        return np.repeat(field.mean(axis=1, keepdims=True), w, axis=1)

    paint_dome = teal[None, None, :] * (0.95 + 0.07 * v[..., None])
    paint = paint * dome_fade[..., None] + paint_dome * dome_weight
    # On the domes keep only the circumferential seam rust, not the averaged
    # patch/streak field (which would become a brown cap at the pole).
    seam_only = np.clip(seam, 0, 1)
    rust_dome = np.clip(0.85 * seam_only - .05, 0, .75)
    rust_mask = rust_mask * dome_fade + rust_dome * (1 - dome_fade)
    rust_colour = rust_colour * dome_fade[..., None] + rust[None, None, :] * dome_weight
    grime_dome = np.clip(.40 * seam_only, 0, .40) * .35
    grime = grime * dome_fade + grime_dome * (1 - dome_fade)
    rgb = paint * (1 - rust_mask[..., None]) + rust_colour * rust_mask[..., None]
    rgb = rgb * (1 - grime[..., None]) + grime_colour[None, None, :] * grime[..., None]
    # Chipped paint and a few hairline scratches expose steel.
    scratch = np.zeros((h, w), np.uint8)
    for _ in range(150):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        length = int(rng.integers(8, 70))
        angle = rng.normal(0, .5)
        x1 = int(np.clip(x0 + length * math.cos(angle), 0, w - 1))
        y1 = int(np.clip(y0 + length * math.sin(angle), 0, h - 1))
        cv2.line(scratch, (x0, y0), (x1, y1), 1, 1)
    scratch = cv2.GaussianBlur(scratch.astype(np.float32), (0, 0), .55) * dome_fade
    chips = (scratch > .35) | (fine > .9965)
    chips &= (rust_mask < .45) & (dome_fade > .25)
    steel = np.array(profile['exposed_steel_rgb'], np.float32)
    rgb = rgb * (1 - chips[..., None] * .85) + steel[None, None, :] * (chips[..., None] * .85)
    rough = (.335 + .035 * (mid - .5) + .30 * rust_mask + .16 * grime
             - .08 * chips.astype(np.float32))
    metal = .90 * chips.astype(np.float32)
    bump = (.65 * dome_fade * (fine - .5) + .80 * dome_fade * (detail - .5)
            + .70 * (rust_mask - .5) + .9 * scratch)
    bump = bump * dome_fade + azimuth_mean(bump) * (1 - dome_fade)
    paths = save_maps(out, 'body', rgb, rough, metal, bump, normal_strength=1.15)
    stats = texture_stats(rgb, rough, metal, bump)
    stats.update({'rust_fraction': round(float(rust_mask.mean()), 4),
                  'teal_fraction': round(float((teal_fraction(rgb)).mean()), 4),
                  'label_edge_rust_pixels': int((label_edge > .2).sum())})
    stats['passed'] = bool(60 <= stats['mean_luma'] <= 185
                           and .03 <= stats['rust_fraction'] <= .32
                           and stats['dark_fraction'] < .20
                           and stats['metallic_mean'] < .16
                           and .28 <= stats['roughness_mean'] <= .68
                           and stats['normal_std'] >= .08)
    return paths, stats


def teal_fraction(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (g > 55) & (b > 55) & (g > r * 1.2)


def texture_stats(rgb, rough, metal, bump):
    rgb = np.asarray(rgb, np.float32)
    luma = rgb @ np.array([.2126, .7152, .0722], np.float32)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return {
        'mean_rgb': [round(float(x), 2) for x in rgb.reshape(-1, 3).mean(axis=0)],
        'mean_luma': round(float(luma.mean()), 2),
        'mean_chroma': round(float(chroma.mean()), 2),
        'dark_fraction': round(float((luma < 25).mean()), 4),
        'roughness_mean': round(float(np.asarray(rough).mean()), 4),
        'metallic_mean': round(float(np.asarray(metal).mean()), 4),
        'normal_std': round(float(np.asarray(bump).std()), 3),
    }


def make_paint_atlas(out, profile, rust_path):
    rng = np.random.default_rng(39040)
    size = 1024
    tiles = []
    colours = source_colours(rust_path)
    teal = colours['teal'] * np.array([1.00, 1.075, 1.075])
    rust = np.clip(colours['rust'] * np.array([1.28, .87, .70]), 0, 255)
    for index, (rust_amount, chip_amount) in enumerate(((0.0, 0.0), (.07, .006),
                                                        (.55, .02), (.10, .22))):
        yy, xx = np.mgrid[0:size, 0:size]
        coarse = resize_noise(rng, (24, 24), (size, size))
        fine = resize_noise(rng, (size, size), (size, size), cv2.INTER_LINEAR)
        rgb = teal[None, None, :] * (1 + .05 * (coarse[..., None] - .5)
                                     + .02 * (fine[..., None] - .5))
        if rust_amount and index == 2:
            threshold = float(np.quantile(coarse, 1 - rust_amount))
            mask = np.clip((coarse - threshold) * 4.0, 0, 1)
        elif index == 1:
            mask = np.clip((fine - .93) * 14, 0, 1) * .95
        else:
            mask = np.zeros((size, size), np.float32)
        rust_colour = rust[None, None, :] * (1 + .15 * (fine[..., None] - .5))
        rgb = rgb * (1 - mask[..., None] * .85) + rust_colour * (mask[..., None] * .85)
        chips = np.zeros((size, size), np.float32)
        if chip_amount:
            scr = np.zeros((size, size), np.uint8)
            for _ in range(70):
                x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
                ang = rng.normal(0, .6)
                ln = int(rng.integers(5, 55))
                cv2.line(scr, (x0, y0),
                         (int(np.clip(x0 + ln * math.cos(ang), 0, size - 1)),
                          int(np.clip(y0 + ln * math.sin(ang), 0, size - 1))), 1, 1)
            chips = cv2.GaussianBlur(scr.astype(np.float32), (0, 0), .5)
            patch_hi = float(np.quantile(fine, 1 - chip_amount))
            chips = np.maximum(chips, np.clip((fine - patch_hi) * 12, 0, 1))
        steel = np.array(profile['exposed_steel_rgb'], np.float32)
        rgb = rgb * (1 - chips[..., None] * .85) + steel * (chips[..., None] * .85)
        rough = .34 + .30 * mask + .03 * (fine - .5) - .05 * chips
        metal = .85 * chips
        bump = .7 * (fine - .5) + .9 * mask
        tiles.append((rgb, rough, metal, bump))
    atlas = np.zeros((size * 2, size * 2, 3), np.float32)
    rough_atlas = np.zeros((size * 2, size * 2), np.float32)
    metal_atlas = np.zeros_like(rough_atlas)
    bump_atlas = np.zeros_like(rough_atlas)
    for index, (rgb, rough, metal, bump) in enumerate(tiles):
        yy, xx = divmod(index, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        atlas[sl] = rgb
        rough_atlas[sl] = rough
        metal_atlas[sl] = metal
        bump_atlas[sl] = bump
    paths = save_maps(out, 'paint', atlas, rough_atlas, metal_atlas, bump_atlas,
                      normal_strength=1.0, flip=False)
    checks = {'tiles': []}
    for index, (rgb, rough, metal, bump) in enumerate(tiles):
        stats = texture_stats(rgb, rough, metal, bump)
        rust_frac = float(np.clip(((rgb[..., 0] > rgb[..., 1] + 12)
                                   & (rgb[..., 0] > rgb[..., 2] + 20)).mean(), 0, 1))
        stats.update({'tile': index, 'rust_fraction': round(rust_frac, 4)})
        if index == 0:
            stats['passed'] = bool(70 <= stats['mean_luma'] <= 180
                                   and stats['metallic_mean'] < .05
                                   and .28 <= stats['roughness_mean'] <= .50)
        elif index == 1:
            stats['passed'] = bool(60 <= stats['mean_luma'] <= 180
                                   and .01 <= stats['rust_fraction'] <= .25
                                   and stats['metallic_mean'] < .05)
        elif index == 2:
            stats['passed'] = bool(stats['rust_fraction'] >= .25
                                   and stats['roughness_mean'] >= .42)
        else:
            stats['passed'] = bool(stats['metallic_mean'] >= .15
                                   and stats['rust_fraction'] <= .05)
        checks['tiles'].append(stats)
    checks['passed'] = all(tile['passed'] for tile in checks['tiles'])
    return paths, checks


def rectified_label(clean_path, out):
    rgb = np.asarray(Image.open(clean_path).convert('RGB'))
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    mask = ((r > 130) & (g > 110) & (b < 110) & (r > b + 60)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise ValueError('HPEOK label was not found in the clean source')
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == best)
    points = np.stack((xs, ys), axis=1).astype(np.float32)
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    quad = np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                     points[np.argmax(sums)], points[np.argmax(diffs)]], np.float32)
    tl = quad[np.argmin(quad.sum(axis=1))]
    br = quad[np.argmax(quad.sum(axis=1))]
    tr = quad[np.argmin(np.diff(quad, axis=1).ravel())]
    bl = quad[np.argmax(np.diff(quad, axis=1).ravel())]
    src = np.array([tl, tr, br, bl], np.float32)
    width = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    height = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    width, height = max(width, 32), max(height, 32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], np.float32)
    rectified = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(src, dst),
                                    (width, height), flags=cv2.INTER_CUBIC)
    Image.fromarray(rectified).save(out / 'label_rectified_source.png')
    big = cv2.resize(rectified, (720, 540), interpolation=cv2.INTER_CUBIC)
    return big, rectified


def make_label_textures(out, clean_path):
    big, rectified = rectified_label(clean_path, out)
    rng = np.random.default_rng(39041)
    grain = resize_noise(rng, (540, 720), (720, 540), cv2.INTER_LINEAR)
    rgb = big.astype(np.float32)
    rgb += (grain[..., None] - .5) * 4.0
    rgb = np.clip(rgb, 0, 255)
    fine = resize_noise(rng, (540, 720), (720, 540), cv2.INTER_LINEAR)
    rough = np.clip(.535 + .04 * (fine - .5), .42, .65)
    metal = np.zeros_like(rough)
    bump = .8 * (fine - .5)
    # The label is stored upright: the texture top row maps to UV v=1.
    paths = save_maps(out, 'label', rgb, rough, metal, bump, normal_strength=.45,
                      flip=False)
    stats = texture_stats(rgb, rough, metal, bump)
    # Contrast is dominated by the printed blue/black lettering on yellow paper.
    stats['text_contrast'] = round(float(rgb.std()), 3)
    stats['passed'] = bool(105 <= stats['mean_luma'] <= 225
                           and stats['mean_chroma'] >= 35
                           and stats['metallic_mean'] < .05
                           and .40 <= stats['roughness_mean'] <= .68
                           and stats['text_contrast'] >= 18)
    stats['rectified_source_size'] = [int(rectified.shape[1]), int(rectified.shape[0])]
    return paths, stats


def make_metal_textures(out, kind, profile):
    size = 512
    rng = np.random.default_rng(39042 if kind == 'metal' else 39043)
    yy, xx = np.mgrid[0:size, 0:size]
    brushed = resize_noise(rng, (size, 40), (size, size))
    fine = resize_noise(rng, (size, size), (size, size), cv2.INTER_LINEAR)
    if kind == 'metal':
        base_rgb = np.array([190, 192, 194], np.float32)
        rough = .30 + .035 * (brushed - .5) + .015 * (fine - .5)
        warm = False
    else:
        base_rgb = np.array([196, 148, 62], np.float32)
        rough = .32 + .04 * (brushed - .5) + .015 * (fine - .5)
        warm = True
    rgb = base_rgb[None, None, :] * (1 + .10 * (brushed[..., None] - .5)
                                     + .035 * (fine[..., None] - .5))
    metal = np.ones((size, size), np.float32)
    bump = .9 * (brushed - .5) + .5 * (fine - .5)
    paths = save_maps(out, kind, rgb, rough, metal, bump, normal_strength=.9)
    stats = texture_stats(rgb, rough, metal, bump)
    stats['passed'] = bool(stats['metallic_mean'] >= .90
                           and .22 <= stats['roughness_mean'] <= .48
                           and stats['normal_std'] >= .05)
    if warm:
        stats['passed'] = bool(stats['passed']
                               and stats['mean_rgb'][0] > stats['mean_rgb'][1]
                               > stats['mean_rgb'][2])
    return paths, stats


def make_dielectric_textures(out, kind, colour, roughness, seed):
    size = 256
    rng = np.random.default_rng(seed)
    fine = resize_noise(rng, (size, size), (size, size), cv2.INTER_LINEAR)
    coarse = resize_noise(rng, (16, 16), (size, size))
    rgb = np.array(colour, np.float32)[None, None, :] * (
        1 + .05 * (fine[..., None] - .5) + .03 * (coarse[..., None] - .5))
    rough = np.clip(roughness + .035 * (coarse - .5) + .01 * (fine - .5), .05, .95)
    metal = np.zeros((size, size), np.float32)
    bump = .8 * (fine - .5) + .4 * (coarse - .5)
    paths = save_maps(out, kind, rgb, rough, metal, bump, normal_strength=.5)
    stats = texture_stats(rgb, rough, metal, bump)
    if kind == 'black':
        stats['passed'] = bool(stats['mean_luma'] < 70 and stats['metallic_mean'] < .05
                               and .30 <= stats['roughness_mean'] <= .62)
    elif kind == 'red':
        stats['passed'] = bool(stats['mean_rgb'][0] > stats['mean_rgb'][1] + 25
                               and stats['metallic_mean'] < .05
                               and .25 <= stats['roughness_mean'] <= .55)
    else:
        stats['passed'] = bool(stats['metallic_mean'] >= .90)
    return paths, stats


def make_textures(out, rust_path, clean_path, profile):
    textures = {}
    body_paths, body_checks = make_body_textures(out, rust_path, profile)
    textures.update(body_paths)
    paint_paths, paint_checks = make_paint_atlas(out, profile, rust_path)
    textures.update(paint_paths)
    label_paths, label_checks = make_label_textures(out, clean_path)
    textures.update(label_paths)
    metal_paths, metal_checks = make_metal_textures(out, 'metal', profile)
    textures.update(metal_paths)
    brass_paths, brass_checks = make_metal_textures(out, 'brass', profile)
    textures.update(brass_paths)
    black_paths, black_checks = make_dielectric_textures(
        out, 'black', (18, 18, 20), profile['black_plastic_roughness'], 39044)
    textures.update(black_paths)
    red_paths, red_checks = make_dielectric_textures(
        out, 'red', (145, 22, 24), profile['red_plastic_roughness'], 39045)
    textures.update(red_paths)
    checks = {
        'body': body_checks,
        'paint': paint_checks,
        'label': label_checks,
        'metal': metal_checks,
        'brass': brass_checks,
        'black': black_checks,
        'red': red_checks,
    }
    checks['passed'] = all(value['passed'] for value in checks.values()
                           if isinstance(value, dict))
    checks['summary'] = {
        '筒体': f"rust {body_checks['rust_fraction']:.3f} · "
                f"R {body_checks['roughness_mean']:.3f} · "
                f"M {body_checks['metallic_mean']:.3f}",
        '小件漆面图集': 'pass' if paint_checks['passed'] else 'fail',
        '标签': f"luma {label_checks['mean_luma']:.1f} · "
                f"chroma {label_checks['mean_chroma']:.1f} · "
                f"R {label_checks['roughness_mean']:.3f}",
        '不锈钢': f"M {metal_checks['metallic_mean']:.2f} · "
                  f"R {metal_checks['roughness_mean']:.3f}",
        '黄铜': f"M {brass_checks['metallic_mean']:.2f} · "
                f"R {brass_checks['roughness_mean']:.3f}",
        '黑色塑料': f"luma {black_checks['mean_luma']:.1f} · M {black_checks['metallic_mean']:.3f}",
        '红色塑料': f"R {red_checks['mean_rgb'][0]:.0f} · M {red_checks['metallic_mean']:.3f}",
    }
    return textures, checks


# ---------------------------------------------------------------------------
# Reference and boards
# ---------------------------------------------------------------------------
def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 10, y0 + 8, x0 + w // 3 - 10, y0 + h // 2 - 8))


def reference_mask(panel):
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = rgb.mean(axis=2)
    mask = (gray < 238).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .95):] = 0
    mask[:4] = 0
    mask[-4:] = 0
    mask[:, :4] = 0
    mask[:, -4:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(300, areas.max() * .02)) + 1
    return np.isin(labels, keep).astype(np.uint8)


VIEW_NOTES = {
    'front': '参考正视与原照同一外观家族：双视镜口、黄色标签、上下阀组；参考未标定。',
    'back': '原照背面不可见；参考背视无端口与标签，候选按旋转对称与参考推断。',
    'left': '参考左视显示端口凸出和上下阀组方向；侧向服务接头方位为推断。',
    'right': '参考右视与左视镜像；原照右侧被轮廓遮挡，细节为推断。',
    'top': '参考顶视确认阀组横轴、左快接/右黑旋钮及肩部侧接头；参考顶视缩放大于正视。',
    'bottom': '参考底视确认底部排放阀组方向；底面接触、落稳和支撑方式未验证。',
}


def make_report_39(out, reference, reference_rows):
    font = ImageFont.truetype(LABEL_FONT, 22)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2240, 2660), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '39 集油器2 · 立式青绿压力油罐 · 六视图视觉匹配参考（未标定）、原模型与新候选对照',
              font=font, fill='#15262d')
    reference_image = Image.open(reference).convert('RGB')
    for i, name in enumerate(VIEWS):
        y = 56 + i * 420
        panel = panel_image(reference_image, i)
        mask = reference_mask(panel)
        rx, ry, rw, rh = cv2.boundingRect(mask)
        panel = panel.crop((max(0, rx - 8), max(0, ry - 8),
                            min(panel.width, rx + rw + 8),
                            min(panel.height, ry + rh + 8)))
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        draw.text((12, y), f'{name.upper()}   六视图参考 / 原模型 / 新候选',
                  font=small, fill='#27363c')
        for j, image in enumerate((panel.convert('RGBA'), old, new)):
            if j:
                box = image.getchannel('A').getbbox()
                if box:
                    image = image.crop(box)
            factor = min(500 / image.width, 340 / image.height)
            image = image.resize((max(1, round(image.width * factor)),
                                  max(1, round(image.height * factor))),
                                 Image.Resampling.LANCZOS)
            tile = Image.new('RGBA', (540, 355), 'white' if j == 0 else '#dde3e2')
            tile.alpha_composite(image, ((540 - image.width) // 2,
                                         (355 - image.height) // 2))
            board.paste(tile.convert('RGB'), (10 + j * 552, y + 26))
        row = reference_rows[i]
        text = (f"参考-新诊断 IoU {row['reference_new_mask_iou']} · "
                f"旧-新轮廓 IoU {row['old_new_silhouette_iou']} · "
                f"{VIEW_NOTES[name]}")
        draw.text((12, y + 388), text, font=small, fill='#52646b')
    draw.text((20, 2600),
              '重要：参考图为未标定视觉匹配图，各面板缩放不一致；IoU 只用于发现形体差异，'
              '不是测量精度或验收通过。原模型六方向审计显示阀组呈团块、标签与视镜口细节丢失，'
              '故未复用旧网格；旧图使用中性光，未完整复现其顶点色，只比较几何。'
              '背面、内部通道、螺纹、壁厚和落稳仍待人工复核。',
              font=small, fill='#8a2b20')
    board.save(out / '六视图-新旧对照.jpg', quality=95)


def _tile_image(path, size):
    image = Image.open(path).convert('RGBA')
    rect = image.getchannel('A').getbbox()
    if rect:
        image = image.crop(rect)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def make_source_board(out, source_clean, reference, fidelity):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1900, 830), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 10), '39 集油器2 · 原照 / 原模型源视角 / 新候选源视角 / 六视图正视参考',
              font=font, fill='#15262d')
    source = Image.open(source_clean).convert('RGBA')
    box = source.getchannel('A').getbbox()
    if box:
        source = source.crop(box)
    panel = panel_image(Image.open(reference).convert('RGB'), 0)
    mask = reference_mask(panel)
    rx, ry, rw, rh = cv2.boundingRect(mask)
    panel = panel.crop((max(0, rx - 8), max(0, ry - 8),
                        min(panel.width, rx + rw + 8),
                        min(panel.height, ry + rh + 8)))
    items = ((source, '原照（clean）'), (Image.open(out / 'renders' / 'old-source.png').convert('RGBA'), '原模型源视角'),
             (Image.open(out / 'renders' / 'new-source.png').convert('RGBA'), '新候选源视角'),
             (panel.convert('RGBA'), '六视图正视参考'))
    for i, (image, label) in enumerate(items):
        if i:
            bb = image.getchannel('A').getbbox()
            if bb:
                image = image.crop(bb)
        factor = min(420 / image.width, 650 / image.height)
        image = image.resize((max(1, round(image.width * factor)),
                              max(1, round(image.height * factor))),
                             Image.Resampling.LANCZOS)
        tile = Image.new('RGBA', (460, 690), 'white' if i == 0 else '#dde3e2')
        tile.alpha_composite(image, ((460 - image.width) // 2, (690 - image.height) // 2))
        board.paste(tile.convert('RGB'), (20 + i * 470, 66))
        draw.text((24 + i * 470, 41), label, font=small, fill='#27363c')
    draw.text((20, 772),
              f"严格源相机轮廓诊断 IoU {fidelity['iou']:.4f}，填孔 {fidelity['filled_holes']}，"
              f"窄结构召回 {fidelity['narrow_recall']:.3f}；相机与参考均未标定，仅作诊断，不是验收通过。",
              font=small, fill='#52646b')
    draw.text((20, 797),
              '原模型为单图重建；当前 Cycles 对照使用中性光且未完整复现其顶点色，旧图只比较几何，不评价旧材质。',
              font=small, fill='#8a2b20')
    board.save(out / '原照-新候选-参考.jpg', quality=95)


def make_detail_board(out, label_checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 14)
    items = (
        ('overall-front', '整体正视（透明背景）'),
        ('top-manifold', '顶部阀组：六角螺母/阀体/黄铜塞'),
        ('top-block', '顶部阀体、黄铜塞与红帽'),
        ('top-knob', '右旋钮：黑色滚花与端帽'),
        ('top-coupler', '左快接：不锈钢台阶与卡槽'),
        ('upper-port', '上视镜口：青绿凸台、金属环、红点'),
        ('lower-port', '下视镜口与三个黑色指示点'),
        ('label', 'HPEOK PHO-10 标签（实际矫正贴图）'),
        ('weld-upper', '上环焊缝与锈蚀'),
        ('weld-lower', '下环焊缝与锈蚀'),
        ('side-fitting', '左肩侧向服务接头（推断方位）'),
        ('bottom-manifold', '底部排放阀组'),
        ('bottom-block', '底部阀体、黄铜塞与旋钮'),
        ('body-surface', '筒体漆面、锈斑与划痕'),
    )
    board = Image.new('RGB', (2100, 1620), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '39 集油器2 · 关键局部检查（新候选）', font=font, fill='#15262d')
    for i, (name, label) in enumerate(items):
        x = (i % 7) * 296
        y = 55 + (i // 7) * 760
        image = _tile_image(out / 'renders' / f'detail-{name}.png', (272, 680))
        tile = Image.new('RGBA', (282, 700), '#dde3e2')
        tile.alpha_composite(image, ((282 - image.width) // 2, (700 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 5, y + 30))
        draw.text((x + 7, y), label, font=small, fill='#27363c')
    draw.text((20, 1570),
              '这些局部图用于人工检查结构；不能证明未观察面、壁厚、螺纹、内部通道、'
              f'落稳或现场尺度。标签贴图锐度约为原图 4 倍放大（原始矫正尺寸 '
              f'{label_checks["rectified_source_size"]}）。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, openings, structural):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1900, 1470), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '39 集油器2 · 隔离结构与开口检查', font=font, fill='#15262d')
    items = (('body-only', '隔离筒体与两道焊缝'),
             ('ports-only', '隔离双视镜口与指示点'),
             ('label-only', '隔离标签曲面'),
             ('top-valve-only', '隔离顶部阀组'),
             ('bottom-valve-only', '隔离底部阀组'),
             ('side-fitting-only', '隔离侧向服务接头'),
             ('silhouette-front', '正视轮廓：内孔检测'),
             ('silhouette-top', '顶视轮廓：内孔检测'))
    for i, (name, label) in enumerate(items):
        x = (i % 4) * 474
        y = 55 + (i // 4) * 700
        image = _tile_image(out / 'renders' / f'{name}.png', (450, 620))
        tile = Image.new('RGBA', (462, 640), '#dde3e2')
        tile.alpha_composite(image, ((462 - image.width) // 2, (640 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 6, y + 30))
        if name == 'silhouette-front':
            text = f'{label}：{openings["silhouette_front_holes"]}/0'
            color = '#1d5c3a' if openings['silhouette_front_holes'] == 0 else '#8a2b20'
        elif name == 'silhouette-top':
            text = f'{label}：{openings["silhouette_top_holes"]}/0'
            color = '#1d5c3a' if openings['silhouette_top_holes'] == 0 else '#8a2b20'
        else:
            text = label
            color = '#27363c'
        draw.text((x + 8, y), text, font=small, fill=color)
    draw.text((20, 1420),
              f'透明背景隔离渲染。物体为封闭压力容器，可见端口均为封闭视镜/堵头；'
              f'隔离轮廓内孔 {openings["silhouette_front_holes"]}/0（正视）与 '
              f'{openings["silhouette_top_holes"]}/0（顶视）。'
              '内孔检测不能证明壁厚、内部通道、螺纹或阀门可动性。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def _image_diff(first_path, second_path):
    first = np.asarray(Image.open(first_path).convert('RGBA')).astype(np.float32)
    second = np.asarray(Image.open(second_path).convert('RGBA')).astype(np.float32)
    if first.shape != second.shape:
        raise ValueError('light comparison images differ in size')
    mask = (first[..., 3] >= 128) & (second[..., 3] >= 128)
    if not mask.any():
        return {'mean_abs_difference': 0.0, 'pixels': 0, 'changed': False}
    difference = np.abs(first[..., :3] - second[..., :3])[mask].mean()
    luma_a = first[..., :3].mean(axis=2)
    luma_b = second[..., :3].mean(axis=2)
    threshold = np.quantile(luma_a[mask], .90)
    hot = mask & (luma_a >= threshold)
    ys, xs = np.nonzero(hot)
    ys2, xs2 = np.nonzero(hot)
    centroid_a = (xs.mean(), ys.mean()) if len(xs) else (0.0, 0.0)
    bright_b = mask & (luma_b >= np.quantile(luma_b[mask], .90))
    ys3, xs3 = np.nonzero(bright_b)
    centroid_b = (xs3.mean(), ys3.mean()) if len(xs3) else (0.0, 0.0)
    shift = float(math.dist(centroid_a, centroid_b))
    return {'mean_abs_difference': round(float(difference), 3),
            'pixels': int(mask.sum()), 'highlight_centroid_shift_px': round(shift, 2),
            'changed': bool(difference >= 1.0)}


def make_material_boards(out, metal, paint):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1500, 900), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '39 集油器2 · 同光照材质对照（原模型 / 新候选）', font=font, fill='#15262d')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（几何/中性光；顶点色未完整复现）'),
                                       ('new-material-close', '新候选 PBR（漆面/锈/金属）'))):
        x = i * 740
        image = _tile_image(out / 'renders' / f'{name}.png', (690, 780))
        tile = Image.new('RGBA', (710, 800), '#dde3e2')
        tile.alpha_composite(image, ((710 - image.width) // 2, (800 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 865), '同一中性光与相机；高光来自实时照明，不烘焙在底色中。原模型为单图重建的顶点色，不代表旧模型整体通过。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1500, 900), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), f'39 集油器2 · 漆面变光检查（平均差 {paint["mean_abs_difference"]}）',
              font=font, fill='#15262d')
    for i, (name, label) in enumerate((('new-paint-light-a', '主光位置 A'),
                                       ('new-paint-light-b', '主光位置 B'))):
        x = i * 740
        image = _tile_image(out / 'renders' / f'{name}.png', (690, 780))
        tile = Image.new('RGBA', (710, 800), '#dde3e2')
        tile.alpha_composite(image, ((710 - image.width) // 2, (800 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 865), '只显示漆面部件；两组主光下漆面高光与明暗应移动。该检查不覆盖金属件，也不证明现场光照。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1500, 900), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), f'39 集油器2 · 金属件高光变光（平均差 {metal["mean_abs_difference"]}）',
              font=font, fill='#15262d')
    for i, (name, label) in enumerate((('new-metal-light-a', '主光位置 A'),
                                       ('new-metal-light-b', '主光位置 B'))):
        x = i * 740
        image = _tile_image(out / 'renders' / f'{name}.png', (690, 780))
        tile = Image.new('RGBA', (710, 800), '#dde3e2')
        tile.alpha_composite(image, ((710 - image.width) // 2, (800 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 865), f'只显示不锈钢/黄铜/青铜金属件；高光质心移动 '
                         f'{metal.get("highlight_centroid_shift_px", 0)} px。'
                         '该检查不覆盖黑色塑料、视镜与漆面，也不证明环境匹配。',
              font=small, fill='#52646b')
    board.save(out / '表面反射-变光对照.jpg', quality=96)


def make_texture_board(out, textures, checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 14)
    items = (
        ('body_base', '筒体底色：青绿漆/锈/划痕'),
        ('body_mr', '筒体 MR：G 粗糙度 / B 金属度'),
        ('body_normal', '筒体法线'),
        ('paint_base', '小件漆面图集（2×2）'),
        ('label_base', 'HPEOK 标签（矫正后 4× 放大）'),
        ('metal_base', f'不锈钢：M {checks["metal"]["metallic_mean"]:.2f}'),
        ('brass_base', f'黄铜：M {checks["brass"]["metallic_mean"]:.2f}'),
        ('black_base', '黑色塑料：非金属'),
    )
    board = Image.new('RGB', (2080, 1260), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '39 集油器2 · PBR 贴图与数值检查', font=font, fill='#15262d')
    for i, (name, label) in enumerate(items):
        x = (i % 4) * 516
        y = 55 + (i // 4) * 590
        image = Image.open(out / f'{name}.png').convert('RGB')
        image.thumbnail((490, 500), Image.Resampling.LANCZOS)
        tile = Image.new('RGBA', (500, 510), '#dde3e2')
        tile.paste(image.convert('RGBA'), ((500 - image.width) // 2, (510 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 8, y + 28))
        draw.text((x + 10, y), label, font=small, fill='#27363c')
    draw.text((20, 1215),
              f"贴图数值检查：{'通过' if checks['passed'] else '未通过'}。"
              f"筒体锈覆盖 {checks['body']['rust_fraction']:.3f}，暗块 {checks['body']['dark_fraction']:.3f}；"
              f"标签 luma {checks['label']['mean_luma']:.1f}、chroma {checks['label']['mean_chroma']:.1f}；"
              f"金属/黄铜金属度均值 {checks['metal']['metallic_mean']:.2f}/{checks['brass']['metallic_mean']:.2f}。",
              font=small, fill='#52646b')
    board.save(out / 'PBR贴图-对照.jpg', quality=95)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    tile = 250
    board = Image.new('RGB', (tile * 6, tile * 4 + 34), '#eef2f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '39 集油器2 · 24 方向转台（中性光透明背景）', font=font, fill='#27363c')
    for i in range(24):
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png').convert('RGBA')
        image.thumbnail((tile - 16, tile - 16), Image.Resampling.LANCZOS)
        cell = Image.new('RGBA', (tile, tile), '#dde3e2')
        cell.alpha_composite(image, ((tile - image.width) // 2,
                                     (tile - image.height) // 2))
        board.paste(cell.convert('RGB'), ((i % 6) * tile, 34 + (i // 6) * tile))
    board.save(out / '转台-24方向.jpg', quality=94)


# ---------------------------------------------------------------------------
# Measurements and checks
# ---------------------------------------------------------------------------
def luma_stats(path):
    image = Image.open(path).convert('RGBA')
    alpha = np.asarray(image.getchannel('A'))
    rgb = np.asarray(image.convert('RGB')).astype(np.float32)
    mask = alpha >= 128
    if not mask.any():
        return {'median_luma': 0, 'dark_fraction': 1, 'laplacian_variance': 0}
    gray = rgb.mean(axis=2)
    values = gray[mask]
    contours = cv2.Laplacian((gray * (mask > 0)).astype(np.uint8), cv2.CV_32F)
    return {
        'median_luma': round(float(np.median(values)), 1),
        'dark_fraction': round(float((values < 25).mean()), 4),
        'dark_fraction_below_20': round(float((values < 20).mean()), 4),
        'laplacian_variance': round(float(contours[mask].var()), 2),
        'pixels': int(mask.sum()),
    }


def reference_view_checks(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    rows = []
    for i, name in enumerate(VIEWS):
        panel = panel_image(reference_image, i)
        expected = mask_fit(reference_mask(panel))
        with Image.open(out / 'renders' / f'new-{name}.png') as image:
            new = mask_fit(np.asarray(image.getchannel('A')) >= 128)
        with Image.open(out / 'renders' / f'old-{name}.png') as image:
            old = mask_fit(np.asarray(image.getchannel('A')) >= 128)
        ref_iou = None
        old_iou = None
        if expected is not None and new is not None:
            ref_iou = round(float(np.logical_and(expected, new).sum() /
                                  max(1, np.logical_or(expected, new).sum())), 4)
        if old is not None and new is not None:
            old_iou = round(float(np.logical_and(old, new).sum() /
                                  max(1, np.logical_or(old, new).sum())), 4)
        rows.append({'view': name, 'reference_new_mask_iou': ref_iou,
                     'old_new_silhouette_iou': old_iou,
                     'note': VIEW_NOTES[name] +
                             ' IoU 按最长边等比对齐，仅诊断。'})
    return rows


def structural_checks(worker):
    names = [part['name'] for part in worker['parts']]
    counts = {
        'bodies': sum(name == 'Body' for name in names),
        'weld_seams': sum(name.startswith('Weld seam') for name in names),
        'port_bosses': sum(name.startswith('Port ') and name.endswith('boss') for name in names),
        'port_glass': sum(name.endswith('glass') for name in names),
        'port_red_dots': sum(name.endswith('red dot') for name in names),
        'port_screws': sum('screw' in name for name in names),
        'labels': sum('label' in name.lower() for name in names),
        'top_parts': sum(name.startswith('Top ') for name in names),
        'bottom_parts': sum(name.startswith('Bottom ') for name in names),
        'side_fitting_parts': sum(name.startswith('Side fitting') for name in names),
        'stainless_parts': len(worker['material_parts'].get('steel', [])),
        'brass_parts': len(worker['material_parts'].get('brass', [])),
        'black_plastic_parts': len(worker['material_parts'].get('black', [])),
        'paint_parts': len(worker['material_parts'].get('paint', [])),
        'body_parts': len(worker['material_parts'].get('body', [])),
    }
    ok = bool(counts['bodies'] == 1 and counts['weld_seams'] == 2
              and counts['port_bosses'] == 2 and counts['port_glass'] == 2
              and counts['port_red_dots'] == 2 and counts['port_screws'] == 6
              and counts['labels'] == 1 and counts['top_parts'] >= 10
              and counts['bottom_parts'] >= 10 and counts['side_fitting_parts'] == 3)
    return {'counts': counts, 'passed': ok,
            'materials': sorted(set(worker['material_key'].values())) if 'material_key' in worker else worker.get('materials', []),
            'note': ('Part presence proves only that a feature was modelled, not its size, '
                     'fit, thread, hidden faces or assembly function.')}


def audit_row(number):
    path = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
    if not path.is_file():
        return None
    for line in path.read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('number') == number:
                return row
    return None


def source_measurements(row):
    return {
        'photo_foreground_px': [315, 1233],
        'body_max_diameter_px': 311,
        'body_height_excluding_valves_px': 1045,
        'overall_height_to_diameter': round(1233 / 311, 4),
        'body_height_to_diameter': round(1045 / 311, 4),
        'candidate_body_diameter_m': .165,
        'candidate_overall_height_m': .648,
        'scale_basis': 'visible 10 L / 3.7 gal label + clean-photo pixel ratio',
        'scale_is_measured': False,
        'note': ('No calibrated camera or measured drawing is available. The 10 L label '
                 'implies a smaller vessel than the general Bitzer-catalogue interval in '
                 'asset-catalog.json; that interval is a similar-category placeholder, not '
                 'this unit’s size.'),
        'label_text_readable': [
            'HPEOK', '油分离器', 'Oil Receiver', 'Type PHO-10', 'Inlet 3/8" ODF',
            'Outlet 3/8" ODF', '3450 kPa 500 psi', 'CE', 'PED', '10 L / 3.7 gal',
            '7 kg', '2018.12.13', 'Made in China'],
        'label_text_unreadable': ['exact OEM permits', 'serial/QR payload',
                                  'lower-line microtext'],
    }


def run_browser(out, page):
    if not CHROME.is_file():
        return {'status': 'not_executed', 'reason': 'headless Chrome not found'}
    screenshot = out / 'renders' / 'page-screenshot.png'
    command = [str(CHROME), '--headless=new', '--disable-gpu', '--no-sandbox',
               '--hide-scrollbars', '--window-size=1440,1200',
               f'--screenshot={screenshot}', page.resolve().as_uri()]
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if process.returncode or not screenshot.is_file():
        return {'status': 'failed', 'returncode': process.returncode,
                'stderr': process.stderr[-800:]}
    image = Image.open(screenshot).convert('RGB')
    arr = np.asarray(image)
    return {'status': 'headless_chrome_screenshot_ok',
            'screenshot': str(screenshot),
            'size': list(image.size),
            'mean_luma': round(float(arr.mean()), 2),
            'nonblank': bool(arr.std() > 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    parser.add_argument('--library', type=Path,
                        default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    parser.add_argument('--skip-blender', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    library = args.library.resolve()
    blender = args.blender.resolve()
    config_path = HERE / 'oilreceiver39_config.json'
    worker_path = HERE / 'mesh_rebuild_39_worker.py'

    row = next(r for r in (json.loads(line) for line in
                           library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == NUMBER)
    reference = REFERENCE_PATH.resolve()
    if not reference.is_file():
        raise FileNotFoundError(f'six-view reference missing: {reference}')
    if digest(reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
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

    config = json.loads(config_path.read_text(encoding='utf8'))
    if config.get('number') != NUMBER:
        raise ValueError('config number mismatch')
    signature_input = {
        'reference': digest(reference), 'mesh': row['mesh_sha256'],
        'rust': row['rust_sha256'], 'clean': row['clean_sha256'],
        'source_alpha': digest(alpha_path),
        'source_camera': row['reconstruction']['source_camera'],
        'config': digest(config_path), 'builder': digest(Path(__file__)),
        'worker': digest(worker_path),
        'review_page': digest(HERE / 'review39_page.py'),
        'helpers': {name: digest(HERE / name) for name in
                    ('blender_worker.py', 'mesh_checks.py', 'mesh_carve.py',
                     'mesh_rebuild_item.py')},
        'blender': subprocess.run([str(blender), '--version'], capture_output=True,
                                  text=True, check=True).stdout.splitlines()[0],
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'opencv': cv2.__version__, 'pillow': PIL.__version__,
        'trimesh': trimesh.__version__,
    }
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'source-measurements.json').write_text(
        json.dumps(source_measurements(row), ensure_ascii=False, indent=2), encoding='utf8')
    old_audit = audit_row(NUMBER)
    (out / 'old-model-audit.json').write_text(
        json.dumps(old_audit or {}, ensure_ascii=False, indent=2), encoding='utf8')

    textures = None
    if not (out / 'worker-result.json').is_file() or args.skip_blender is False:
        textures, texture_checks = make_textures(
            out, Path(row['rust']), Path(row['original']), config['material_profile'])
        (out / 'texture-checks.json').write_text(
            json.dumps(texture_checks, ensure_ascii=False, indent=2), encoding='utf8')
    else:
        texture_checks = json.loads((out / 'texture-checks.json').read_text(encoding='utf8'))
        textures = {path.stem: str(path) for path in out.glob('*.png')}

    request = {'out': str(out), 'original_mesh': row['mesh'], 'textures': textures,
               'config': config, 'source_camera': row['reconstruction']['source_camera'],
               'source_alpha_bbox': source_alpha_bbox, 'draft': False,
               'signature': signature}
    (out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2),
                                      encoding='utf8')

    if not args.skip_blender:
        with (out / 'blender.log').open('w', encoding='utf8') as log:
            process = subprocess.run(
                [str(blender), '--background', '--factory-startup',
                 '--python-exit-code', '1', '--python', str(worker_path),
                 '--', str(out / 'request.json')],
                stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        if process.returncode:
            raise RuntimeError(f'Blender failed; see {out / "blender.log"}')

    worker = json.loads((out / 'worker-result.json').read_text(encoding='utf8'))
    mesh_audit = inspect_mesh(out / 'mesh.glb')
    with Image.open(alpha_path) as alpha, \
            Image.open(out / 'renders' / 'new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                   np.asarray(rendered.getchannel('A')))
    reference_views = reference_view_checks(out, reference)
    openings = {}
    for name, key in (('silhouette-front', 'silhouette_front_holes'),
                      ('silhouette-top', 'silhouette_top_holes')):
        with Image.open(out / 'renders' / f'{name}.png') as image:
            _, count = interior_holes(np.asarray(image.getchannel('A')))
        openings[key] = int(count)
    openings['silhouette_front_passed'] = openings['silhouette_front_holes'] == 0
    openings['silhouette_top_passed'] = openings['silhouette_top_holes'] == 0
    openings['note'] = ('The vessel is modelled as a closed pressure container with capped '
                        'sight/plug ports; the silhouette check only proves that the isolated '
                        'renders have no enclosed holes. It does not prove wall thickness or '
                        'internal passages.')
    structural = structural_checks(worker)
    metal = _image_diff(out / 'renders' / 'new-metal-light-a.png',
                        out / 'renders' / 'new-metal-light-b.png')
    paint = _image_diff(out / 'renders' / 'new-paint-light-a.png',
                        out / 'renders' / 'new-paint-light-b.png')
    metal['passed'] = bool(metal['changed'] and metal['pixels'] >= 800)
    paint['passed'] = bool(paint['changed'])
    luma = {name: luma_stats(out / 'renders' / name) for name in
            ('new-front.png', 'new-material-close.png', 'detail-label.png',
             'new-paint-light-a.png')}
    luma_passed = bool(luma['new-front.png']['median_luma'] > 20
                       and luma['new-front.png']['dark_fraction'] < .35
                       and luma['new-material-close.png']['median_luma'] > 20
                       and luma['new-material-close.png']['laplacian_variance'] > 20)

    make_report_39(out, reference, reference_views)
    make_source_board(out, Path(row['original']), reference, fidelity)
    make_detail_board(out, texture_checks['label'])
    make_isolated_board(out, openings, structural)
    make_material_boards(out, metal, paint)
    make_texture_board(out, textures, texture_checks)
    make_turntable(out)

    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 45
                               and not mesh_audit['nonfinite_parts']
                               and not mesh_audit['open_parts']
                               and worker['closed_parts'] == len(worker['parts'])
                               and not worker['self_intersecting_components']),
        'isolated_openings': bool(openings['silhouette_front_passed']
                                  and openings['silhouette_top_passed']),
        'structural_counts': structural['passed'],
        'material_tiles': bool(texture_checks['passed']),
        'metal_reflection': bool(metal['passed']),
        'paint_response': bool(paint['passed']),
        'luma_not_crushed': luma_passed,
        'reference_diagnostic_only': True,
        'source_fidelity_diagnostic': bool(fidelity.get('passed', False)),
        'browser_page': False,
    }
    report = {
        'version': config['version'], 'number': NUMBER, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'replaces_library_mesh': False, 'enters_automatic_placement': False,
        'shape_basis': 'source_photo_with_uncalibrated_six_view_visual_reference',
        'reference_role': 'visual_match_unverified',
        'method': 'parametric_part_rebuild_from_photo_and_uncalibrated_visual_match_sheet',
        'reference': str(reference), 'reference_sha256': digest(reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'scale_note': config['scale_note'],
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_clean_sha256': row['clean_sha256'],
        'original_rust_sha256': row['rust_sha256'],
        'original_source_fidelity_record': (old_audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.965 only measures one silhouette. '
            'The six-view audit shows a blob-like top/bottom manifold, indistinct '
            'sight ports and label, lost knob knurling and a dark, muddy surface; '
            'the old mesh is therefore not reused for the candidate base geometry.'),
        'catalog_evidence': {
            'object_size_interval_mm': [1000, 1500],
            'minor_interval_mm': [245, 310],
            'basis': 'similar_catalog',
            'source': 'https://www.bitzer.de/shared_media/documentation/ba-603-2-au.pdf',
            'note': ('The catalogue interval is a similar-category placeholder, not a '
                     'measurement of this 10 L HPEOK unit. No exact PHO-10 dimension '
                     'table was found in public search; the label volume and photo '
                     'ratio are used for the working scale.'),
        },
        'inputs': signature_input, 'signature': signature,
        'mesh': str(out / 'mesh.glb'), 'mesh_sha256': digest(out / 'mesh.glb'),
        'blend': str(out / 'model.blend'), 'blend_sha256': digest(out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'texture_checks': texture_checks,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The source photo is a head-on front view with unknown camera calibration. '
            'The worker fits the new render to the library source-alpha height; the IoU '
            'is a diagnostic only and is never used as an acceptance gate.'),
        'reference_views': reference_views,
        'reference_ratio_note': (
            'The six-view panels have different scales (the top/bottom panels are '
            'zoomed); IoU is computed after bounding-box scaling and only checks '
            'silhouette family, not measured geometry.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': openings,
        'structural_checks': structural, 'checks': checks,
        'metal_response': metal, 'paint_response': paint, 'luma': luma,
        'worker': worker, 'draft': False,
        'limitations': [
            'The six-view sheet is an uncalibrated visual match; panel scales differ and it is not a measured drawing.',
            'Sizes follow the visible 10 L label and clean-photo pixel ratios; they are not measured dimensions.',
            'The back, internal cavity, wall thickness, head thickness, port threads and valve bores are hidden and inferred.',
            'The angled upper-left service fitting is visible but its exact function, thread and azimuth are not confirmed.',
            'The valve blocks, quick-connect barbs, knob detents and brass plug slots are simplified appearance approximations.',
            'The HPEOK label texture is a perspective-rectified crop of the visible photo; blurred or hidden microtext is preserved, not re-rendered.',
            'Paint/rust distribution is synthesised from observed patches; the exact weathering pattern on hidden faces is unknown.',
            'The vessel stands on its bottom drain valve in the candidate because no feet or bracket are visible; stability and support are not validated.',
            'Assemblies intentionally overlap at necks, ports and seams; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'The old-model comparison renders use neutral lighting and the current Cycles import does not fully reproduce its vertex colours; they are geometry comparisons, not an old-material acceptance.',
            'Site scale, floor contact, stability, anchor/bracket, and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the top/bottom valve orientation, knob sizes and side-fitting azimuth.',
            'Confirm the body diameter, overall height, 10 L volume and 7 kg label values from a sample or measured drawing.',
            'Confirm port construction (sight glass vs plug), thread size, red/black indicator meaning and hidden internal passages.',
            'Confirm the small angled service fitting function and whether the red cap at the top exists on the actual unit.',
            'Confirm wall thickness, head thickness, material grade and working-pressure certification; the label values are not treated as verified here.',
            'Confirm paint colour, rust/stain distribution and label placement; hidden-face weathering is inferred.',
            'Use the old-model renders for geometry comparison only; its vertex colours are not fully reproduced in the current Cycles import.',
            'Verify coordinate, scale, stability, floor contact, support/bracket and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or a measured drawing if exact geometry is required.',
        ],
        'next_number': 40,
        'checks_record': str(out / 'checks-39.json'),
    }
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding='utf8')

    page = write_page(out, report, {
        'checks': checks, 'fidelity': fidelity, 'reference_views': reference_views,
        'textures': texture_checks, 'metal': metal, 'paint': paint,
        'openings': openings, 'structural': structural,
        'browser': {'status': 'pending'}, 'pending': report['pending_review'],
        'luma': luma})
    browser = run_browser(out, page)
    report['browser_check'] = browser
    checks['browser_page'] = browser.get('status') == 'headless_chrome_screenshot_ok'
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding='utf8')
    write_page(out, report, {
        'checks': checks, 'fidelity': fidelity, 'reference_views': reference_views,
        'textures': texture_checks, 'metal': metal, 'paint': paint,
        'openings': openings, 'structural': structural,
        'browser': browser, 'pending': report['pending_review'], 'luma': luma})

    progress = {
        'number': NUMBER,
        'status': 'candidate_generated_pending_review',
        'out': str(out),
        'mesh': str(out / 'mesh.glb'),
        'blend': str(out / 'model.blend'),
        'checks': checks,
        'pending_review': report['pending_review'],
        'next_number': 40,
        'version': config['version'],
        'manual_only': True,
        'replaces_library_mesh': False,
        'enters_automatic_placement': False,
        'record': str(out / 'result.json'),
        'checks_record': str(out / 'checks-39.json'),
        'hashes': str(out / 'hashes.json'),
        'notes': ('Item 39 is the last numbered item; next_number 40 is recorded only '
                  'as the continuation convention. Candidate is manual_only and was not '
                  'fed to automatic placement or final compositing.'),
    }
    (out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    (ROOT / 'work/direct-deepseek/39-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')

    hashes = {
        'algorithm': 'sha256',
        'generated_by': 'mesh_rebuild_item_39.py',
        'number': NUMBER,
        'inputs': {
            'original': sha_record(Path(row['original'])),
            'rust': sha_record(Path(row['rust'])),
            'six_view_reference': sha_record(reference),
            'old_mesh': sha_record(Path(row['mesh'])),
            'source_view': sha_record(Path(row['source_view'])),
            'source_alpha': sha_record(alpha_path),
            'config': sha_record(config_path),
            'builder': sha_record(Path(__file__)),
            'worker': sha_record(worker_path),
            'review_page_generator': sha_record(HERE / 'review39_page.py'),
        },
        'outputs': {},
    }
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'hashes.json':
            hashes['outputs'][str(path.relative_to(out)).replace('\\', '/')] = {
                'sha256': digest(path), 'bytes': path.stat().st_size}
    (out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                     encoding='utf8')

    print(json.dumps({'status': 'review', 'out': str(out), 'checks': checks,
                      'fidelity': fidelity, 'reference_views': reference_views,
                      'metal': metal, 'paint': paint, 'openings': openings,
                      'structural': structural['counts'], 'browser': browser},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
