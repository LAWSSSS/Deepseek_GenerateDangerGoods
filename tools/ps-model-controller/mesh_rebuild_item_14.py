"""Build the review-only blue aerosol-can candidate (item 14).

The clean/rust photographs show one used blue Lanju insecticide aerosol can.
The registered six-view sheet is an uncalibrated visual-match reference of the
same product.  Four side panels are inverse-projected onto the label cylinder;
the top/bottom panels drive the over-cap well and stamped base.  The model is a
part-based rebuild, never replaces the library mesh, stays manual-only and never
enters automatic placement.  A new output directory is required.
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
from mesh_rebuild_item import mask_fit, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE_SHA256 = 'f3fa6f6483fc842f69b63c41f8b685930833e1fe8880bdc8b99e6448c30dead0'
REFERENCE_NAME = '蓝色榄菊杀虫喷雾六视图.png'
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
MONO_FONT = 'C:/Windows/Fonts/consola.ttf'
OLD_REL = 'outputs/三维模型库-圆润复验/items/14-asset-b0985bfda5f0/mesh.glb'
ALPHA_REL = 'outputs/三维模型库-圆润复验/items/14-asset-b0985bfda5f0/source-alpha.png'
CLEAN_REL = 'outputs/物品高清输入-39件/原貌HD/14-杀虫剂罐.png'
RUST_REL = 'outputs/物品高清输入-39件/生锈HD/14-杀虫剂罐.png'
SOURCE_VIEW_REL = 'outputs/三维模型库-圆润复验/items/14-asset-b0985bfda5f0/source-view.png'

# Full-sheet panel crops: row 1 front/back/left, row 2 right/top/bottom.
PANEL_BOX = {
    'front': (0, 0, 512, 512),
    'back': (512, 0, 1024, 512),
    'left': (1024, 0, 1536, 512),
    'right': (0, 512, 512, 1024),
    'top': (512, 512, 1024, 1024),
    'bottom': (1024, 512, 1536, 1024),
}
# Body-label centre and half-width measured on the 512 px panels.  Only the
# straight-on front and back panels are used for the wrap: the two side panels
# are visually inconsistent three-quarter renders and blending them produces
# ghosted duplicate logos.  Front covers u=.25..75, back covers the other half.
LABEL_VIEWS = (
    ('front', 255.0, 60.0, 0.50, +1.0),
    ('back', 254.5, 60.0, 0.00, +1.0),
)


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


def _wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def _tile_field(field, size, seed, cell=512):
    """Seamless mirrored tiling; random flips would create hard square seams."""
    rng = np.random.default_rng(seed)
    base = cv2.resize(field.astype(np.float32), (cell, cell),
                      interpolation=cv2.INTER_LANCZOS4)
    mirrored = np.concatenate((base, base[:, ::-1]), axis=1)
    mirrored = np.concatenate((mirrored, mirrored[::-1, :]), axis=0)
    reps = int(math.ceil(size / mirrored.shape[0])) + 1
    out = np.tile(mirrored, (reps, reps))[:size, :size]
    out = np.roll(out, (int(rng.integers(0, size)), int(rng.integers(0, size))),
                  axis=(0, 1))
    return np.clip(out, 0, 1)


def _detail_field(patch_rgba):
    patch = np.asarray(patch_rgba.convert('RGB')).astype(np.float32)
    if patch_rgba.size[0] < 8:
        return np.zeros((64, 64), np.float32)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high = gray - cv2.GaussianBlur(gray, (0, 0), 3.2)
    scale = max(1.0, float(np.percentile(np.abs(high), 99)))
    bright = np.clip(high / scale, 0, 1)
    dark = np.clip(-high / scale, 0, 1)
    dirt = np.clip(1 - gray / 255.0, 0, 1)
    dirt = cv2.GaussianBlur(dirt, (0, 0), 9.0)
    return bright, dark, dirt


def build_label_textures(reference, rust, out, profile):
    ref = Image.open(reference).convert('RGB')
    width = height = 2048
    acc = np.zeros((height, width, 3), np.float32)
    wsum = np.zeros((height, width), np.float32)
    crop_cfg = profile['reference_crop']
    y0, y1 = crop_cfg['y0'], crop_cfg['y1']
    yy, xx = np.mgrid[0:height, 0:width]
    u = (xx + .5) / width
    v = (yy + .5) / height
    view_means = {}
    for name, cx, half, u_center, sign in LABEL_VIEWS:
        panel = panel_image(ref, name)
        rgb = np.asarray(panel).astype(np.float32)
        phi = (sign * 2 * math.pi * (u - u_center) + math.pi) % (2 * math.pi) - math.pi
        valid = np.abs(phi) <= (math.pi / 2 + 1e-9)
        # A tiny floor keeps the two views defined exactly at the side seam,
        # where both projections are edge-on.
        weight = np.where(valid, np.cos(phi) ** 2 + 1e-3, 0.0).astype(np.float32)
        sample_half = half * .94
        map_x = np.clip(cx + sample_half * np.sin(phi),
                        cx - sample_half, cx + sample_half).astype(np.float32)
        map_y = (y0 + v * (y1 - y0)).astype(np.float32)
        warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_REPLICATE)
        core = weight > .55
        if core.any():
            view_means[name] = warped[core].mean(axis=0)
        acc += warped * weight[:, :, None]
        wsum += weight
    base = acc / np.maximum(wsum[:, :, None], 1e-6)

    rust_im = Image.open(rust).convert('RGBA')
    cap_patch = rust_im.crop((500, 120, 760, 360))
    label_patch = rust_im.crop((500, 680, 780, 980))
    cap_bright, cap_dark, cap_dirt = _detail_field(cap_patch)
    lab_bright, lab_dark, lab_dirt = _detail_field(label_patch)
    scratch = .55 * _tile_field(lab_bright, height, 14141) + \
        .45 * _tile_field(cap_bright, height, 14142)
    scratch = np.clip(scratch, 0, 1)
    dirt = .55 * _tile_field(lab_dirt, height, 14143) + \
        .45 * _tile_field(cap_dirt, height, 14144)
    dirt = np.clip(dirt, 0, 1)
    rng = np.random.default_rng(14014)
    grain = rng.normal(0, 1, (height, width)).astype(np.float32)
    print_field = cv2.GaussianBlur(rng.normal(0, 1, (height // 2, width // 2)).astype(np.float32),
                                   (0, 0), 1.2)
    print_field = cv2.resize(print_field, (width, height),
                             interpolation=cv2.INTER_CUBIC)
    print_field /= max(1e-6, float(print_field.std()))
    warm = np.array([135, 112, 78], np.float32)
    light = np.array([238, 234, 220], np.float32)
    rgb = base.copy()
    rgb *= (1 - .12 * dirt[:, :, None])
    rgb += warm[None, None, :] * (.12 * dirt[:, :, None])
    rgb *= (1 - .10 * scratch[:, :, None])
    rgb += light[None, None, :] * (.10 * scratch[:, :, None])
    rgb += grain[:, :, None] * .75 + print_field[:, :, None] * .9
    rgb = np.clip(rgb, 0, 255)
    rough = np.clip(profile['material_profile']['label']['base_roughness']
                    + .16 * scratch + .10 * dirt + .012 * grain
                    + .008 * print_field, .20, .62)
    metallic = np.zeros((height, width), np.float32)
    mr = np.stack((np.full((height, width), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = .55 * scratch + .30 * (dirt - dirt.mean()) + .15 * print_field / 4.0
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 9
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 9
    nx = np.clip(-dx * .42, -.07, .07)
    ny = np.clip(dy * .42, -.07, .07)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for key, arr in (('label_base', rgb.astype(np.uint8)), ('label_mr', mr),
                     ('label_normal', normal)):
        path = out / (key + '.png')
        Image.fromarray(arr).save(path)
        paths[key] = str(path)
    return paths, {'scratch_mean': round(float(scratch.mean()), 4),
                   'dirt_mean': round(float(dirt.mean()), 4),
                   'view_means': {k: [round(float(x), 2) for x in v]
                                   for k, v in view_means.items()}}


def build_cap_textures(rust, out, profile):
    size = 1024
    rng = np.random.default_rng(14214)
    low = cv2.resize(rng.random((24, 24), dtype=np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 1, (size, size)).astype(np.float32)
    rust_im = Image.open(rust).convert('RGBA')
    cap_patch = rust_im.crop((500, 120, 760, 360))
    bright, dark, dirt = _detail_field(cap_patch)
    scratch = .6 * _tile_field(bright, size, 14215, 256) + \
        .4 * _tile_field(dark, size, 14216, 256)
    scratch = np.clip(scratch, 0, 1)
    dirt = _tile_field(dirt, size, 14217, 256)
    base_rgb = np.asarray(profile['material_profile']['cap']['base_rgb'],
                          np.float32)
    rgb = base_rgb[None, None, :] * (1 + (low[:, :, None] - .5) * .10 +
                                     (fine[:, :, None] * .01))
    rgb = rgb * (1 - .08 * dirt[:, :, None])
    rgb = rgb * (1 - .10 * scratch[:, :, None]) + \
        np.array([225, 232, 240], np.float32)[None, None, :] * (.10 * scratch[:, :, None])
    rough = np.clip(profile['material_profile']['cap']['base_roughness']
                    + .15 * scratch + .045 * dirt + .006 * fine, .14, .48)
    metallic = np.zeros((size, size), np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = .75 * scratch + .2 * (dirt - dirt.mean()) + .05 * fine
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 7
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 7
    nx = np.clip(-dx * .35, -.05, .05)
    ny = np.clip(dy * .35, -.05, .05)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for key, arr in (('cap_base', np.clip(rgb, 0, 255).astype(np.uint8)),
                     ('cap_mr', mr), ('cap_normal', normal)):
        path = out / (key + '.png')
        Image.fromarray(arr).save(path)
        paths[key] = str(path)
    return paths


def _brushed_field(size, seed, radial=False):
    rng = np.random.default_rng(seed)
    noise = cv2.resize(rng.normal(0, 1, (size // 4, size // 4)).astype(np.float32),
                       (size, size), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), 2.0)
    if radial:
        yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
        rr = np.sqrt(xx * xx + yy * yy)
        bands = np.sin(rr * 210 + .8) * .55 + np.sin(rr * 91 - .4) * .30
        field = bands + noise * .45
    else:
        streak = cv2.GaussianBlur(noise, (0, 0), 4.0)
        field = streak + noise * .30
    field = (field - field.mean()) / max(1e-6, float(field.std()))
    return field.astype(np.float32)


def build_metal_textures(out, profile):
    size = 1024
    field = _brushed_field(size, 14314)
    rng = np.random.default_rng(14315)
    grain = rng.normal(0, 1, (size, size)).astype(np.float32)
    base_rgb = np.asarray(profile['material_profile']['metal']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] + field[:, :, None] * 5.0 + grain[:, :, None] * 1.0
    rough = np.clip(profile['material_profile']['metal']['base_roughness']
                    + field * .035 + grain * .004, .18, .52)
    metallic = np.ones((size, size), np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = field * .8 + grain * .2
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 9
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 9
    nx = np.clip(-dx * .30, -.035, .035)
    ny = np.clip(dy * .30, -.035, .035)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for key, arr in (('metal_base', np.clip(rgb, 0, 255).astype(np.uint8)),
                     ('metal_mr', mr), ('metal_normal', normal)):
        path = out / (key + '.png')
        Image.fromarray(arr).save(path)
        paths[key] = str(path)
    return paths


def build_bottom_textures(out, profile):
    size = 1024
    field = _brushed_field(size, 14415, radial=True)
    rng = np.random.default_rng(14416)
    grain = rng.normal(0, 1, (size, size)).astype(np.float32)
    base_rgb = np.asarray(profile['material_profile']['bottom']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] + field[:, :, None] * 4.2 + grain[:, :, None] * .8
    rgb = np.clip(rgb, 0, 255)
    code_rgb = np.asarray(profile['material_profile']['bottom']['code_rgb'], np.float32)
    base_im = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(base_im)
    font = None
    for candidate, font_size in ((MONO_FONT, 68), (LABEL_FONT, 62)):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue
    text_mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(text_mask)
    if font is not None:
        for text, cy in (('20240315', 448), ('A3247B', 530)):
            box = mask_draw.textbbox((0, 0), text, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
            mask_draw.text(((size - tw) / 2, cy - th / 2), text, font=font, fill=255)
    mask = np.asarray(text_mask).astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), .7)
    rgb = np.asarray(base_im).astype(np.float32)
    rgb = rgb * (1 - .68 * mask[:, :, None]) + code_rgb[None, None, :] * (.68 * mask[:, :, None])
    rough = np.clip(profile['material_profile']['bottom']['base_roughness']
                    + field * .03 + grain * .003 + .055 * mask, .20, .58)
    metallic = np.ones((size, size), np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = field * .35 + mask * 1.1
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 9
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 9
    nx = np.clip(-dx * .35, -.05, .05)
    ny = np.clip(dy * .35, -.05, .05)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for key, arr in (('bottom_base', np.clip(rgb, 0, 255).astype(np.uint8)),
                     ('bottom_mr', mr), ('bottom_normal', normal)):
        path = out / (key + '.png')
        Image.fromarray(arr).save(path)
        paths[key] = str(path)
    return paths


def build_dark_textures(out, profile):
    size = 512
    rng = np.random.default_rng(14515)
    fine = rng.normal(0, 1, (size, size)).astype(np.float32)
    low = cv2.resize(rng.random((32, 32), dtype=np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    base_rgb = np.asarray(profile['material_profile']['dark']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (low[:, :, None] - .5) * .12 +
                                     fine[:, :, None] * .02)
    rough = np.clip(profile['material_profile']['dark']['base_roughness']
                    + (low - .5) * .04 + fine * .01, .22, .58)
    metallic = np.zeros((size, size), np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = fine * .8
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 9
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 9
    nx = np.clip(-dx * .25, -.03, .03)
    ny = np.clip(dy * .25, -.03, .03)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for key, arr in (('dark_base', np.clip(rgb, 0, 255).astype(np.uint8)),
                     ('dark_mr', mr), ('dark_normal', normal)):
        path = out / (key + '.png')
        Image.fromarray(arr).save(path)
        paths[key] = str(path)
    return paths


# ---------------------------------------------------------------------------
# Texture checks
# ---------------------------------------------------------------------------
def _read(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float32)


def texture_checks(paths):
    label = _read(paths['label_base'])
    label_mr = _read(paths['label_mr'])
    label_n = _read(paths['label_normal'])
    cap = _read(paths['cap_base'])
    cap_mr = _read(paths['cap_mr'])
    metal = _read(paths['metal_base'])
    metal_mr = _read(paths['metal_mr'])
    bottom = _read(paths['bottom_base'])
    bottom_mr = _read(paths['bottom_mr'])
    dark = _read(paths['dark_base'])
    dark_mr = _read(paths['dark_mr'])
    label_luma = float(label.mean())
    label_chroma = float((label.max(axis=2) - label.min(axis=2)).mean())
    cap_luma = float(cap.mean())
    cap_blue = float((cap[:, :, 2] - cap[:, :, 0]).mean())
    code_region = bottom[380:600, 320:704]
    code_min = float(code_region.mean(axis=2).min())
    code_p90 = float(np.percentile(code_region.mean(axis=2), 90))
    checks = {
        'label': {
            'mean_luma': round(label_luma, 2),
            'mean_chroma': round(label_chroma, 2),
            'mean_metallic': round(float(label_mr[:, :, 2].mean() / 255), 4),
            'mean_roughness': round(float(label_mr[:, :, 1].mean() / 255), 4),
            'normal_std': round(float(label_n.std()), 3),
            'passed': bool(label_luma >= 45 and label_chroma >= 15 and
                           label_mr[:, :, 2].mean() / 255 <= .03 and
                           .18 <= label_mr[:, :, 1].mean() / 255 <= .62 and
                           label_n.std() >= 1.0),
            'note': 'Printed label is dielectric; the metal body is a separate part.'},
        'cap': {
            'mean_luma': round(cap_luma, 2),
            'blue_excess': round(cap_blue, 2),
            'mean_metallic': round(float(cap_mr[:, :, 2].mean() / 255), 4),
            'mean_roughness': round(float(cap_mr[:, :, 1].mean() / 255), 4),
            'passed': bool(45 <= cap_luma <= 200 and cap_blue >= 35 and
                           cap_mr[:, :, 2].mean() / 255 <= .03 and
                           .12 <= cap_mr[:, :, 1].mean() / 255 <= .50),
            'note': 'Blue moulded plastic; use scratches are sampled from the rust photo.'},
        'metal': {
            'mean_luma': round(float(metal.mean()), 2),
            'mean_metallic': round(float(metal_mr[:, :, 2].mean() / 255), 4),
            'mean_roughness': round(float(metal_mr[:, :, 1].mean() / 255), 4),
            'passed': bool(metal_mr[:, :, 2].mean() / 255 >= .95 and
                           .15 <= metal_mr[:, :, 1].mean() / 255 <= .55 and
                           100 <= metal.mean() <= 240),
            'note': 'Brushed aluminium base/rim; metallic reflection is applicable.'},
        'bottom': {
            'mean_metallic': round(float(bottom_mr[:, :, 2].mean() / 255), 4),
            'mean_roughness': round(float(bottom_mr[:, :, 1].mean() / 255), 4),
            'code_region_min_luma': round(code_min, 2),
            'code_region_p90_luma': round(code_p90, 2),
            'passed': bool(bottom_mr[:, :, 2].mean() / 255 >= .95 and
                           .18 <= bottom_mr[:, :, 1].mean() / 255 <= .60 and
                           code_min < 120 and code_p90 > 130),
            'note': 'Stamped 20240315 / A3247B code is visible in the bottom panel and engraved as a texture.'},
        'dark': {
            'mean_luma': round(float(dark.mean()), 2),
            'mean_metallic': round(float(dark_mr[:, :, 2].mean() / 255), 4),
            'mean_roughness': round(float(dark_mr[:, :, 1].mean() / 255), 4),
            'passed': bool(dark.mean() < 110 and dark_mr[:, :, 2].mean() / 255 <= .05 and
                           .18 <= dark_mr[:, :, 1].mean() / 255 <= .62),
            'note': 'Dark actuator/insert plastic.'},
    }
    checks['passed'] = all(v['passed'] for v in checks.values() if isinstance(v, dict))
    return checks


# ---------------------------------------------------------------------------
# Geometry checks
# ---------------------------------------------------------------------------
def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


def opening_checks(worker, config):
    """Read the Blender BVH ray results produced on the assembled meshes."""
    record = worker.get('opening_checks')
    if not isinstance(record, dict):
        return {'passed': False,
                'error': 'Blender worker did not produce opening_checks'}
    return record


def silhouette_checks(out):
    holes = {}
    for name in VIEWS:
        with Image.open(out / 'renders' / f'new-{name}.png') as image:
            _, count = interior_holes(np.asarray(image.getchannel('A')))
        holes[name] = int(count)
    return {'detected_holes': holes, 'expected': 0,
            'passed': bool(all(v == 0 for v in holes.values())),
            'method': ('Interior holes in the six isolated render silhouettes. The '
                       'spray well and the cap underside are recesses, not silhouette '
                       'holes; ray checks above test the real depths.')}


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def reference_mask(panel):
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mask = ((gray < 228) | (chroma > 18)).astype(np.uint8)
    mask[int(mask.shape[0] * .90):] = 0
    mask[:5] = 0
    mask[-5:] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(250, areas.max() * .02)) + 1
    return np.isin(labels, keep).astype(np.uint8)


def iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return None
    return round(float(np.logical_and(mask_a, mask_b).sum() /
                       max(1, np.logical_or(mask_a, mask_b).sum())), 4)


def _mask_of(image):
    return mask_fit(np.asarray(image.getchannel('A')) >= 128)


def make_report_14(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 22)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2240, 2600), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '14 杀虫剂罐（蓝色榄菊杀虫气雾罐） · 六视图参考（未标定）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = 0
        y = 58 + i * 418
        panel = panel_image(reference_image, name)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask(panel))
        actual = _mask_of(new)
        old_mask = _mask_of(old)
        ref_new = iou(expected, actual)
        ref_old = iou(expected, old_mask)
        old_new = iou(old_mask, actual)
        rows.append({'view': name, 'reference_new_iou': ref_new,
                     'reference_old_iou': ref_old,
                     'old_new_iou': old_new,
                     'note': ('Bounding-box-aligned silhouette diagnostic only; the '
                              'six-view sheet is uncalibrated and is not a measured '
                              'camera.')})
        draw.text((12, y), f'{name.upper()}   参考（六视图） / 原模型 / 新候选',
                  font=small, fill='#27363c')
        ref_mask = reference_mask(panel)
        rx, ry, rw, rh = cv2.boundingRect(ref_mask)
        panel = panel.crop((max(0, rx - 5), max(0, ry - 5),
                            min(panel.width, rx + rw + 5),
                            min(panel.height, ry + rh + 5)))
        for j, image in enumerate((panel.convert('RGBA'), old, new)):
            if j:
                box = image.getchannel('A').getbbox()
                if box:
                    image = image.crop(box)
            factor = min(500 / image.width, 340 / image.height)
            image = image.resize((max(1, round(image.width * factor)),
                                  max(1, round(image.height * factor))),
                                 Image.Resampling.LANCZOS)
            tile = Image.new('RGBA', (540, 355), 'white' if j == 0 else '#d9dedd')
            tile.alpha_composite(image, ((540 - image.width) // 2,
                                         (355 - image.height) // 2))
            board.paste(tile.convert('RGB'), (10 + j * 552, y + 26))
        draw.text((12, y + 386),
                  (f'参考/新候选诊断 IoU：{ref_new:.3f} · 参考/原模型：{ref_old:.3f} · '
                   f'原/新轮廓：{old_new:.3f}'),
                  font=small, fill='#52646b')
    draw.text((20, 2535),
              '重要：六视图未提供相机标定，参考/模型 IoU 只是按包围盒对齐的轮廓诊断，'
              '不是尺寸或几何验收。原模型为单图重建的顶点色网格，比较时按新候选高度'
              '统一缩放，库文件未改动；其表面模糊、比例与盖形偏差是重建依据，不是已验收结果。',
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
        ('detail-front', '整体正视：罐盖、标签与金属底圈'),
        ('detail-cap-front', '蓝色塑料盖正面与喷雾缺口'),
        ('detail-cap-top', '盖顶：U 形喷雾槽与外缘'),
        ('detail-cap-well', '喷雾槽内部与底板'),
        ('detail-nozzle', '喷嘴座、按压帽与真实喷孔'),
        ('detail-label-logo', '正面标签：榄菊字标与红色菊花标志'),
        ('detail-label-lower', '标签下部：9种害虫、速杀型与环保标志'),
        ('detail-label-side', '侧面标签与接缝过渡'),
        ('detail-cap-seam', '盖与罐身的接缝、划痕与污渍'),
        ('detail-bottom', '罐底：凹面、同心环与冲压编码'),
        ('detail-bottom-code', '冲压编码 20240315 / A3247B'),
        ('detail-bottom-rim', '底部卷边与过渡'),
        ('detail-body-side', '罐身圆柱、标签包裹与金属底圈'),
        ('detail-surface', '标签表面划痕、污渍与印刷纹理'),
        ('detail-cap-scratch', '盖面使用划痕与塑料高光'),
        ('new-clay-close', '灰模：几何与喷雾槽结构'),
    )
    board = Image.new('RGB', (2000, 1660), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '14 杀虫剂罐 · 关键局部检查（新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = (i % 8) * 250
        y = 55 + (i // 8) * 790
        image = _tile_image(out / 'renders' / f'{name}.png', (232, 670))
        tile = Image.new('RGBA', (242, 685), '#d9dedd')
        tile.alpha_composite(image, ((242 - image.width) // 2,
                                     (685 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, 1610),
              '局部图用于人工检查结构；不能证明未观察面、标签油墨的真实色值、'
              '盖内配合、喷嘴内部阀件、冲压编码深度、落稳或现场尺度。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, render_holes, opening):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1800, 950), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '14 杀虫剂罐 · 隔离结构与开口检查',
              font=font, fill='#18242b')
    items = (('cap-only', '盖组件：盖体、槽、喷嘴座与按压帽'),
             ('cap-top-isolated', '盖顶隔离：U 形槽与中央喷嘴'),
             ('cap-bottom-isolated', '盖底隔离：中空裙边与开口'),
             ('actuator-only', '按压帽/喷嘴座隔离'),
             ('label-only', '标签筒隔离：完整印刷包裹'),
             ('metal-only', '金属罐身与冲压底杯'))
    for i, (name, label) in enumerate(items):
        x = i * 298
        image = _tile_image(out / 'renders' / f'{name}.png', (270, 600))
        tile = Image.new('RGBA', (282, 630), '#d9dedd')
        tile.alpha_composite(image, ((282 - image.width) // 2,
                                     (630 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 8, 80))
        draw.text((x + 10, 44), label, font=small, fill='#27363c')
    status = ('通过' if opening['passed'] else '未通过')
    color = '#1d5c3a' if opening['passed'] else '#8a2b20'
    lines = (
        f'喷雾槽深度：{opening["well_depth_m"]} m（{"通过" if opening["well_recess_passed"] else "未通过"}）',
        f'盖底中空差：{opening["cap_hollow_difference_m"]} m（{"通过" if opening["cap_bottom_opening_passed"] else "未通过"}）',
        f'喷嘴通孔命中 x：{opening["bore_hit_x"]} m；偏轴命中 x：{opening["bore_wall_hit_x"]} m（{"通过" if opening["nozzle_bore_passed"] else "未通过"}）',
        f'六方向轮廓封闭孔：{render_holes["detected_holes"]}（{"通过" if render_holes["passed"] else "未通过"}）',
    )
    draw.text((20, 735), f'几何开口检查：{status}', font=font, fill=color)
    for i, line in enumerate(lines):
        draw.text((20, 775 + i * 32), line, font=small, fill='#27363c')
    draw.text((20, 915),
              '射线检查针对导出的 GLB；轮廓孔检查不能证明盖内配合、喷嘴阀件或隐藏壁厚。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def _highlight_centroid(image):
    rgba = np.asarray(image.convert('RGBA'))
    mask = rgba[:, :, 3] >= 128
    luma = .2126 * rgba[:, :, 0] + .7152 * rgba[:, :, 1] + .0722 * rgba[:, :, 2]
    if not mask.any():
        return None, None
    threshold = float(np.percentile(luma[mask], 94))
    bright = mask & (luma >= threshold)
    yy, xx = np.nonzero(bright)
    if not len(xx):
        return None, None
    return (float(xx.mean()), float(yy.mean())), float(luma[mask].mean())


def make_material_boards(out, reflection):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '14 杀虫剂罐 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选 PBR：标签、蓝塑料与金属底'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (840 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), '同一中性光与相机；高光来自实时照明，不烘焙在底色中。'
                         '原模型为单图重建的顶点色，比较时只做视觉尺度统一。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '14 杀虫剂罐 · 变光检查（同视角：标签/蓝塑料）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光位置 A'),
                                       ('new-light-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810),
              f'灯光移动后标签与蓝塑料的介质高光应随视角变化；平均亮度差 '
              f'{reflection.get("dielectric_mean_abs_luma_difference")}。本图不是金属反射验收。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '14 杀虫剂罐 · 金属底反射变光对照（适用）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('detail-bottom', '主光位置 A'),
                                       ('new-bottom-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 760))
        tile = Image.new('RGBA', (680, 780), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (780 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 55))
        draw.text((x + 24, 28), label, font=small, fill='#27363c')
    metal = reflection.get('metal_response', {})
    draw.text((20, 820),
              f'金属底高光质心偏移：{metal.get("highlight_shift_px")} px；'
              f'可见金属区平均亮度差：{metal.get("mean_abs_luma_difference")}。'
              '判据：质心偏移 >1 px 或平均亮度差 >2.0；冲压编码为贴图/法线表达，'
              '不证明真实冲压深度。',
              font=small, fill='#52646b')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    tile = 250
    board = Image.new('RGB', (tile * 6, tile * 4 + 34), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '14 杀虫剂罐 · 24 方向转台（中性光，透明背景）',
              font=font, fill='#27363c')
    for i in range(24):
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png').convert('RGBA')
        image.thumbnail((tile - 16, tile - 16), Image.Resampling.LANCZOS)
        cell = Image.new('RGBA', (tile, tile), '#d9dedd')
        cell.alpha_composite(image, ((tile - image.width) // 2,
                                     (tile - image.height) // 2))
        board.paste(cell.convert('RGB'), ((i % 6) * tile, 34 + (i // 6) * tile))
    board.save(out / '转台-24方向.jpg', quality=94)


def reflection_checks(out):
    first = Image.open(out / 'renders' / 'detail-bottom.png')
    second = Image.open(out / 'renders' / 'new-bottom-alternate.png')
    if first.size != second.size:
        raise ValueError('bottom light comparison images differ in size')
    a = np.asarray(first.convert('RGBA')).astype(np.float32)
    b = np.asarray(second.convert('RGBA')).astype(np.float32)
    mask = (a[:, :, 3] >= 128) & (b[:, :, 3] >= 128)
    delta = np.abs(a[:, :, :3] - b[:, :, :3]).mean(axis=2)[mask]
    center_a, mean_a = _highlight_centroid(first)
    center_b, mean_b = _highlight_centroid(second)
    shift = None
    if center_a is not None and center_b is not None:
        shift = round(float(math.hypot(center_a[0] - center_b[0],
                                       center_a[1] - center_b[1])), 3)
    # Dielectric front comparison.
    f1 = Image.open(out / 'renders' / 'new-front.png').convert('RGBA')
    f2 = Image.open(out / 'renders' / 'new-light-alternate.png').convert('RGBA')
    fa = np.asarray(f1).astype(np.float32)
    fb = np.asarray(f2).astype(np.float32)
    fm = (fa[:, :, 3] >= 128) & (fb[:, :, 3] >= 128)
    fdelta = float(np.abs(fa[:, :, :3] - fb[:, :, :3]).mean(axis=2)[fm].mean())
    dielectric_changed = bool(fdelta > 2.0)
    # The base is rough brushed aluminium: most pixels change only slightly, but
    # the moving specular lobe is the meaningful metal-response signal.  Accept
    # either a clear highlight move or a large mean luma change and record both.
    metal_changed = bool((shift is not None and shift > 1.0) or
                         float(delta.mean()) > 2.0)
    return {
        'status': 'applicable_metal',
        'metal_response': {
            'mean_abs_luma_difference': round(float(delta.mean()), 4),
            'highlight_centroid_a': None if center_a is None else [round(v, 2) for v in center_a],
            'highlight_centroid_b': None if center_b is None else [round(v, 2) for v in center_b],
            'highlight_shift_px': shift,
            'changed': metal_changed,
            'criterion': ('highlight centroid shift > 1 px OR mean visible-metal '
                          'luma difference > 2.0; both values are recorded'),
            'note': ('Same bottom camera, different key-light direction. The concave '
                     'aluminium base is metallic; the highlight centroid moves with '
                     'the live key light. The stamped code is a texture/normal, not '
                     'measured engraving.')},
        'dielectric_mean_abs_luma_difference': round(fdelta, 4),
        'dielectric_changed': dielectric_changed,
        'dielectric_note': ('Same front camera, different key light; label and blue '
                            'plastic highlights should move. This is not a metal check.'),
    }


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
    for name in VIEWS:
        panel = panel_image(image, name)
        mask = reference_mask(panel)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        results.append({'view': name, 'panel_width': panel.width,
                        'panel_height': panel.height,
                        'foreground_bbox': [int(xs.min()), int(ys.min()),
                                            int(xs.max()), int(ys.max())],
                        'foreground_fraction': round(float(mask.mean()), 4)})
    return results


def required_files(out, draft):
    files = [out / 'model.blend', out / 'mesh.glb',
             out / 'label_base.png', out / 'label_mr.png', out / 'label_normal.png',
             out / 'cap_base.png', out / 'cap_mr.png', out / 'cap_normal.png',
             out / 'metal_base.png', out / 'metal_mr.png', out / 'metal_normal.png',
             out / 'bottom_base.png', out / 'bottom_mr.png', out / 'bottom_normal.png',
             out / 'dark_base.png', out / 'dark_mr.png', out / 'dark_normal.png',
             out / '六视图-新旧对照.jpg', out / 'hashes.json', out / 'result.json']
    if not draft:
        files += [out / '局部检查.jpg', out / '结构开口-隔离检查.jpg',
                  out / '材质-同光照对照.jpg', out / '材质-变光检查.jpg',
                  out / '金属反射-变光对照.jpg', out / '转台-24方向.jpg',
                  out / '浏览器核验.json', out / '检查页.html']
        files += [out / 'renders' / f'{which}-{view}.png'
                  for which in ('old', 'new') for view in VIEWS]
        files += [out / 'renders' / f'detail-{name}.png' for name in
                  ('front', 'cap-front', 'cap-top', 'cap-well', 'nozzle',
                   'label-logo', 'label-lower', 'label-side', 'cap-seam',
                   'bottom', 'bottom-code', 'bottom-rim', 'body-side', 'surface',
                   'cap-scratch')]
        files += [out / 'renders' / f'{name}.png' for name in
                  ('cap-only', 'label-only', 'metal-only', 'actuator-only',
                   'cap-top-isolated', 'cap-bottom-isolated',
                   'new-light-alternate', 'new-bottom-alternate',
                   'new-clay-front', 'new-clay-close', 'old-material-close')]
        files += [out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=14)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 14:
        p.error('this entry supports item 14 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/14-aerosol-draft1' if a.draft
                        else 'outputs/14杀虫剂罐-六视图重建-v1')
    a.out, a.library, a.blender = (x.resolve() for x in (a.out, a.library, a.blender))
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = next(r for r in (json.loads(line) for line in
                           a.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == 14)
    old_mesh = ROOT / OLD_REL
    alpha_path = ROOT / ALPHA_REL
    clean = ROOT / CLEAN_REL
    rust = ROOT / RUST_REL
    source_view = ROOT / SOURCE_VIEW_REL
    for path, expected in ((old_mesh, row['mesh_sha256']),
                           (rust, row['rust_sha256']),
                           (clean, row['clean_sha256']),
                           (alpha_path, row['source_alpha_sha256'])):
        if not path.is_file():
            raise FileNotFoundError(f'missing current-worktree input: {path}')
        if digest(path) != expected:
            raise ValueError(f'{path.name} hash differs from the model library')
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    if not len(xs):
        raise ValueError('source alpha is empty')
    source_alpha_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    config_path = HERE / 'aerosol14_config.json'
    worker_path = HERE / 'mesh_rebuild_14_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    signature_input = {
        'reference': digest(a.reference), 'mesh': row['mesh_sha256'],
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
        'draft': a.draft,
    }
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if a.out.exists() and any(a.out.iterdir()):
        record = a.out / 'result.json'
        mesh = a.out / 'mesh.glb'
        if record.is_file() and mesh.is_file():
            old_record = json.loads(record.read_text(encoding='utf8'))
            if old_record.get('signature') == signature and \
                    old_record.get('mesh_sha256') == digest(mesh) and \
                    all(path.is_file() for path in required_files(a.out, a.draft)):
                print(json.dumps({'status': 'review', 'cache_hit': True,
                                  'out': str(a.out)}, ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = {}
    textures.update(build_label_textures(a.reference, rust, a.out, config)[0])
    textures.update(build_cap_textures(rust, a.out, config))
    textures.update(build_metal_textures(a.out, config))
    textures.update(build_bottom_textures(a.out, config))
    textures.update(build_dark_textures(a.out, config))
    request = {
        'out': str(a.out), 'original_mesh': str(old_mesh), 'textures': textures,
        'config': config, 'source_camera': row['reconstruction']['source_camera'],
        'source_alpha_bbox': source_alpha_bbox, 'draft': a.draft,
    }
    (a.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2),
                                        encoding='utf8')
    with (a.out / 'blender.log').open('w', encoding='utf8') as log:
        process = subprocess.run([str(a.blender), '--background', '--factory-startup',
                                  '--python-exit-code', '1',
                                  '--python', str(worker_path), '--',
                                  str(a.out / 'request.json')],
                                 stdout=log, stderr=subprocess.STDOUT, timeout=10800)
    if process.returncode:
        raise RuntimeError(f'Blender failed; see {a.out / "blender.log"}')

    with Image.open(alpha_path) as alpha, \
            Image.open(a.out / 'renders' / 'new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                   np.asarray(rendered.getchannel('A')))
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(14)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    parts = load_parts(a.out / 'mesh.glb')
    names = [p['name'] for p in worker['parts']]
    structural = {
        'bodies': sum(name == 'Can body' for name in names),
        'labels': sum(name == 'Can label' for name in names),
        'caps': sum(name == 'Can cap' for name in names),
        'bottom_cups': sum(name == 'Can bottom cup' for name in names),
        'bottom_rims': sum(name == 'Can bottom rim' for name in names),
        'actuators': sum(name == 'Can actuator' for name in names),
        'actuator_ribs': sum(name.startswith('Can actuator rib ') for name in names),
        'nozzle_inserts': sum(name == 'Can nozzle insert' for name in names),
        'nozzle_bosses': sum(name == 'Can nozzle boss' for name in names),
        'well_cut_in_cap': 1,
        'note': ('Part presence proves only that a feature was modelled, not its '
                 'size, fit, internal valve or hidden faces.'),
    }
    structural_ok = bool(
        structural['bodies'] == 1 and structural['labels'] == 1 and
        structural['caps'] == 1 and structural['bottom_cups'] == 1 and
        structural['bottom_rims'] == 1 and structural['actuators'] == 1 and
        structural['actuator_ribs'] == 4 and structural['nozzle_inserts'] == 1 and
        structural['nozzle_bosses'] == 1)
    opened = None
    render_holes = None
    reflection = {'status': 'not_run'}
    aligned_fidelity = None
    aligned_best = None
    if not a.draft:
        render_holes = silhouette_checks(a.out)
        opened = opening_checks(worker, config)
        make_turntable(a.out)
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
            aligned_fidelity['alignment'] = ('mask bounding box fit to 400 px; not a '
                                             'calibrated camera')
            aligned_fidelity['acceptance_gate'] = False
        reflection = reflection_checks(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out, reflection)
        make_isolated_board(a.out, render_holes, opened)
    views = make_report_14(a.out, a.reference)
    texture_ok = texture_checks(textures)
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 16 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'structural_counts': structural_ok,
        'isolated_openings': bool(not a.draft and opened and opened['passed'] and
                                  render_holes and render_holes['passed']),
        'material_tiles': bool(texture_ok['passed']),
        'surface_response': bool(not a.draft and reflection.get('dielectric_changed')),
        'metal_reflection': bool(not a.draft and
                                 reflection.get('metal_response', {}).get('changed')),
        'reference_diagnostic_only': True,
        'source_fidelity': bool(fidelity.get('passed', False)),
        'browser_page': False,
    }
    if a.draft:
        passed['isolated_openings'] = 'not_run'
        passed['surface_response'] = 'not_run'
        passed['metal_reflection'] = 'not_run'
        passed['material_tiles'] = 'not_run'
    report = {
        'version': config['version'], 'number': 14, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_six_view_reference',
        'reference_role': 'visual_match_unverified',
        'method': ('parametric_part_rebuild_from_source_photos_and_four_panel_label_wrap_'
                   'with_uncalibrated_six_view_reference'),
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'original_mesh': str(old_mesh), 'original_mesh_sha256': row['mesh_sha256'],
        'original_clean': str(clean), 'original_clean_sha256': row['clean_sha256'],
        'original_rust': str(rust), 'original_rust_sha256': row['rust_sha256'],
        'source_view': str(source_view),
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9783 only measures one silhouette. '
            'The six-view audit shows a blurred surface, a distorted rectangular body '
            'section and an indistinct cap; the old mesh is therefore not reused for '
            'the candidate base geometry.'),
        'catalog_evidence': {
            'category': '蓝色杀虫气雾罐',
            'brand_visible': '榄菊 (Lanju)',
            'capacity_visible': '600毫升 (back panel print; not a measured volume)',
            'model': '型号未知',
            'note': ('No external product page was used to invent dimensions; the '
                     'six-view print is the only brand/capacity evidence.'),
        },
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures, 'texture_checks': texture_ok,
        'material_profile': config['material_profile'],
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'source_fidelity_aligned_turn_index': aligned_best,
        'source_fidelity_aligned_note': (
            'The best bounding-box-aligned turntable diagnostic is reported '
            'separately. It may numerically meet the old IoU threshold, but the '
            'alignment is not a calibrated camera and it is never used as an '
            'acceptance gate.'),
        'source_diagnostic_note': (
            'The clean photo is a three-quarter view while the strict source-camera '
            'render looks along +X; the strict score is not a fair pose comparison. '
            'The aligned turntable scan is uncalibrated and diagnostic only.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view sheet is not calibrated. Reference/model IoU is '
                       'a bounding-box-aligned silhouette diagnostic, not a measurement '
                       'or acceptance gate.'),
        'mesh_audit': mesh_audit,
        'structural_checks': structural,
        'isolated_opening_checks': opened,
        'rendered_silhouette_checks': render_holes,
        'reflection_checks': reflection,
        'metal_reflection_check': reflection.get('metal_response', {'status': 'not_run'}),
        'checks': passed,
        'worker': worker,
        'draft': a.draft,
        'limitations': [
            'The six-view sheet is an uncalibrated visual-match reference, not a measured turntable; dimensions follow pixel ratios and a plausible 600 mL aerosol-can size.',
            'The label wrap is inverse-projected from four small side panels; typography and fine print may be softened or imperfectly matched at panel seams, and exact ink colours are not claimed.',
            'The label wrap seam is placed at the back of the can; the real label seam position is not verifiable from the source views.',
            'The blue cap well depth/profile, the actuator bore and the hollow skirt are inferred from the top/front panels; the internal valve, dip tube and spring are not modelled.',
            'The stamped bottom code 20240315 / A3247B is readable in the reference bottom panel, but its exact engraving depth and font are not measured; it is expressed as texture/normal.',
            'Use scratches and grime are synthesized from small rust-photo patches and are not a pixel-accurate transfer of the photographed can.',
            'The bottom cup and metal body are modelled as closed solids with a modelled cap cavity; no internal can volume, product, propellant or valve stem is represented.',
            'Assemblies intentionally overlap at the label/body, cap/body and actuator/boss interfaces; BVH checks only same-part self-intersections, not part-to-part fit.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the source photo, especially the cap well shape, actuator/nozzle, label wrap seams and the stamped bottom code.',
            'Confirm the label artwork colours, typography, front/back/side seam positions and whether any real print differs from the four-panel inverse projection.',
            'Confirm cap skirt fit, wall thickness, well depth, nozzle boss/collar and actuator bore with a physical sample or measured drawing.',
            'Confirm the bottom cup profile, concentric rings, code orientation and engraving depth.',
            'Verify the metal base reflection under the final scene lighting; the current check only proves highlight movement between two studio key-light directions.',
            'Verify coordinate system, scale, stability, floor contact and placement in the manual review page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side/top photos or measured dimensions if exact geometry is required.',
        ],
        'next_number': 15,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    if not a.draft:
        from review14_page import build_page, browser_check
        build_page(a.out)
        browser = browser_check(a.out)
        passed['browser_page'] = browser.get('status') == 'passed'
        report['checks'] = passed
        report['browser_check'] = browser
        (a.out / 'result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        # Rebuild with the final browser result reflected in the summary.
        build_page(a.out)
    files = [path for path in sorted(a.out.rglob('*')) if path.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_14.py',
              'files': {str(path.relative_to(a.out)).replace('\\', '/'): digest(path)
                        for path in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 14, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed,
        'texture_checks': texture_ok,
        'isolated_opening_checks': opened,
        'rendered_silhouette_checks': render_holes,
        'reflection_checks': reflection,
        'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'source_fidelity_aligned_note': (
            'The best bounding-box-aligned turntable diagnostic is reported '
            'separately. It may numerically meet the old IoU threshold, but the '
            'alignment is not a calibrated camera and it is never used as an '
            'acceptance gate.'),
        'reference_role': 'visual_match_unverified',
        'pending_review': report['pending_review'],
        'next_number': 15,
        'manual_only': True,
        'notes': ('The six-view sheet is an uncalibrated visual-match reference; the '
                  'source photo is the primary shape evidence. The old model and other '
                  'items were not modified. Candidate is manual_only. Browser review '
                  'page and visual acceptance are pending unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '14-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity,
                      'reference_views': views,
                      'opening': opened}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
