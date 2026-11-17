"""Build the review-only gas-cylinder candidate (item 21, 气瓶2).

The clean/rust photographs show one used tall brushed-metal cylinder with a
rounded shoulder, a narrow neck, a silver valve body, a threaded side outlet, a
small knurled safety plug, a front nameplate and a black scalloped handwheel.
The user six-view sheet is an uncalibrated visual-match atlas: its panels use
different scales and the left/right panels both draw the outlet on the same
side, so it is used for part topology and appearance only, never as measured
geometry.  The candidate is a parametric, part-based rebuild.  It never
replaces the library mesh, stays manual-only and never enters automatic
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
from mesh_rebuild_item import mask_fit, inspect_mesh, _tile_detail

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE_SHA256 = 'b93d0a63c3c5e9aa6f8d3cadd61f6922a142077be5724fb6f998be1f60e9c48e'
REFERENCE_NAME = 'ChatGPT 图像 2026年10月14日 10_01_32-2.png'
PANEL_BOX = {
    'front': (0, 0, 512, 512),
    'back': (512, 0, 1024, 512),
    'left': (1024, 0, 1536, 512),
    'right': (0, 512, 512, 1024),
    'top': (512, 512, 1024, 1024),
    'bottom': (1024, 512, 1536, 1024),
}
OLD_REL = 'outputs/三维模型库-圆润复验/items/21-asset-43e7201a6314/mesh.glb'
ALPHA_REL = 'outputs/三维模型库-圆润复验/items/21-asset-43e7201a6314/source-alpha.png'
CLEAN_REL = 'outputs/物品高清输入-39件/原貌HD/21-气瓶2.png'
RUST_REL = 'outputs/物品高清输入-39件/生锈HD/21-气瓶2.png'
SOURCE_VIEW_REL = 'outputs/三维模型库-圆润复验/items/21-asset-43e7201a6314/source-view.png'
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
MONO_FONT = 'C:/Windows/Fonts/consola.ttf'


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


def reference_mask(panel):
    """Approximate foreground mask of one reference panel (diagnostic only)."""
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    mask = (rgb.min(axis=2) < 250).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .88):] = 0
    mask[:3] = 0
    mask[-3:] = 0
    mask[:, :3] = 0
    mask[:, -3:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(30, areas.max() * .004)) + 1
    mask = np.isin(labels, keep).astype(np.uint8)
    return binary_fill_holes(mask).astype(np.uint8)


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------
def _noise(size, rng, cells, blur=0.0):
    if isinstance(size, tuple):
        height, width = size
    else:
        height = width = size
    field = cv2.resize(rng.random((cells, cells), dtype=np.float32), (width, height),
                       interpolation=cv2.INTER_CUBIC)
    if blur:
        field = cv2.GaussianBlur(field, (0, 0), blur)
    return field


def _normal_from_height(height, strength=0.6, blur=1.0):
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), blur)
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx = np.clip(-dx * strength, -.18, .18)
    ny = np.clip(dy * strength, -.18, .18)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    return np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                   0, 255).astype(np.uint8)


def _mr_map(rough, metallic):
    return np.stack((np.full(rough.shape, 255, np.uint8),
                     np.clip(rough * 255, 0, 255).astype(np.uint8),
                     np.clip(metallic * 255, 0, 255).astype(np.uint8)),
                    axis=2).astype(np.uint8)


def _detail_field(rust_image, rect, size):
    detail = cv2.resize(_tile_detail(rust_image, tuple(rect)), size,
                        interpolation=cv2.INTER_CUBIC)
    low = float(detail.min())
    high = float(detail.max())
    return (detail - low) / max(1e-9, high - low)


def build_body_textures(rust, clean, out, cfg):
    width, height = cfg['texture']['body_size']
    rng = np.random.default_rng(212101)
    mat = cfg['material_profile']['body']
    base = np.asarray(mat['base_rgb'], np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    v = yy / max(1, height - 1)
    # Anisotropic brushed-metal field: variation across the circumference,
    # long correlation along the cylinder height.
    coarse = rng.normal(0, 1, (24, 256)).astype(np.float32)
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    brush = cv2.GaussianBlur(coarse, (0, 0), sigmaX=1.2, sigmaY=22.0)
    fine_brush = rng.normal(0, 1, (64, 2048)).astype(np.float32)
    fine_brush = cv2.resize(fine_brush, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_brush = cv2.GaussianBlur(fine_brush, (0, 0), sigmaX=.7, sigmaY=1.5)
    brush_total = brush + .55 * fine_brush
    fine = rng.normal(0, 1, (height, width)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=.5, sigmaY=1.2)
    low = _noise((height, width), rng, 16, 6.0)
    rust_noise = _noise((height, width), rng, 80, 2.0)
    rust_noise = (rust_noise - rust_noise.min()) / max(1e-9, float(np.ptp(rust_noise)))
    shoulder = np.exp(-((v - .85) / .080) ** 2)
    bottom = np.exp(-((v - .03) / .040) ** 2)
    prob = np.clip((rust_noise - .51) * 3.2, 0, 1)
    rust_alpha = prob * (.18 + 1.80 * shoulder + 1.60 * bottom)
    streak_source = rng.normal(0, 1, (8, 512)).astype(np.float32)
    streak = cv2.resize(streak_source, (width, height), interpolation=cv2.INTER_CUBIC)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=2.0, sigmaY=45.0)
    streak = np.clip((streak - .55) * 2.2, 0, 1)
    rust_alpha = np.clip(rust_alpha + .35 * streak * (.10 + shoulder), 0, .88)
    rust_image = Image.open(rust).convert('RGBA')
    rects = cfg['texture']['rust_patch_rects']
    detail_a = _detail_field(rust_image, rects['shoulder'], (width, height))
    detail_b = _detail_field(rust_image, rects['bottom'], (width, height))
    detail = .55 * detail_a + .45 * detail_b
    breakup = _noise((height, width), rng, 220, .6)
    rust_alpha = np.clip(rust_alpha * (.45 + .75 * breakup) *
                         (.60 + .55 * detail) * 1.15, 0, .95)
    rgb = base[None, None, :] * (1 + .022 * brush[:, :, None] +
                                 .008 * fine_brush[:, :, None] +
                                 .010 * fine[:, :, None] +
                                 .020 * (low[:, :, None] - .5))
    grime = np.clip((low - .58) * 2.4, 0, 1) * (.15 + .90 * bottom + .35 * shoulder)
    grime = np.clip(grime + .18 * np.clip((streak - .70) * 2, 0, 1) * bottom, 0, .70)
    grime_rgb = np.asarray(mat['grime_rgb'], np.float32)
    rgb = rgb * (1 - .16 * grime[:, :, None]) + \
        grime_rgb[None, None, :] * (.16 * grime[:, :, None])
    rust_rgb = np.asarray(mat['rust_rgb'], np.float32)
    rust_var = .75 + .45 * detail[:, :, None]
    rgb = rgb * (1 - rust_alpha[:, :, None]) + \
        np.clip(rust_rgb[None, None, :] * rust_var, 0, 255) * rust_alpha[:, :, None]
    pit_noise = rng.random((height, width), dtype=np.float32)
    pit = cv2.GaussianBlur(pit_noise, (0, 0), .6)
    pit = np.clip((pit - .975) * 30, 0, 1)
    rgb = rgb * (1 - .30 * pit[:, :, None]) + \
        np.asarray([48, 44, 40], np.float32)[None, None, :] * (.30 * pit[:, :, None])
    rgb = np.clip(rgb, 0, 255)
    metal = np.ones((height, width), np.float32)
    metal -= rust_alpha * (1 - mat['rust_metallic'])
    metal -= .22 * grime
    metal -= .45 * pit
    metal = np.clip(metal, .03, 1)
    rough = mat['base_roughness'] + .44 * rust_alpha + .12 * grime + .05 * pit + \
        .018 * brush_total + .006 * fine
    rough = np.clip(rough, .16, .86)
    height_field = .45 * brush + .25 * fine_brush + .20 * fine / 3 + \
        .35 * rust_alpha + .5 * pit + .25 * detail
    normal = _normal_from_height(height_field, .18, 1.2)
    paths = {}
    for name, arr in (('body_base', rgb.astype(np.uint8)),
                      ('body_mr', _mr_map(rough, metal)),
                      ('body_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_steel_textures(out, cfg):
    size = cfg['texture']['steel_size'][0]
    rng = np.random.default_rng(212202)
    mat = cfg['material_profile']['steel']
    base = np.asarray(mat['base_rgb'], np.float32)
    low = _noise(size, rng, 20, 1.5)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32),
                            (0, 0), .6)
    brushing = cv2.GaussianBlur(rng.normal(0, 1, (size // 8, size // 32)).astype(np.float32),
                                (0, 0), 1.0)
    brushing = cv2.resize(brushing, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = base[None, None, :] * (1 + .020 * (low[:, :, None] - .5) +
                                 .006 * fine[:, :, None] +
                                 .012 * brushing[:, :, None])
    rgb = np.clip(rgb, 0, 255)
    rough = np.clip(mat['base_roughness'] + .030 * (low - .5) +
                    .010 * fine + .012 * brushing, .15, .42)
    metal = np.full((size, size), mat['metallic'], np.float32)
    height = .65 * brushing + .45 * fine / 2 + .30 * low
    normal = _normal_from_height(height, .40, .7)
    paths = {}
    for name, arr in (('steel_base', rgb.astype(np.uint8)),
                      ('steel_mr', _mr_map(rough, metal)),
                      ('steel_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_black_textures(out, cfg):
    size = cfg['texture']['black_size'][0]
    rng = np.random.default_rng(212303)
    mat = cfg['material_profile']['black']
    base = np.asarray(mat['base_rgb'], np.float32)
    low = _noise(size, rng, 24, 2.0)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32),
                            (0, 0), .5)
    rgb = base[None, None, :] * (1 + .070 * (low[:, :, None] - .5) +
                                 .020 * fine[:, :, None])
    dust = rng.random((size, size), dtype=np.float32)
    dust = cv2.GaussianBlur(dust, (0, 0), 1.0)
    dust = np.clip((dust - .92) * 6.0, 0, 1)
    rgb += dust[:, :, None] * 22.0
    specks = rng.random((size, size), dtype=np.float32)
    specks = np.clip((specks - .9975) * 300, 0, 1)
    rgb += specks[:, :, None] * 45.0
    rgb = np.clip(rgb, 0, 255)
    rough = np.clip(mat['base_roughness'] + .080 * (low - .5) +
                    .030 * fine + .060 * dust, .24, .68)
    metal = np.full((size, size), mat['metallic'], np.float32)
    height = .55 * fine / 3 + .35 * low + .30 * dust
    normal = _normal_from_height(height, .45, .7)
    paths = {}
    for name, arr in (('black_base', rgb.astype(np.uint8)),
                      ('black_mr', _mr_map(rough, metal)),
                      ('black_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_cap_textures(out, cfg):
    size = cfg['texture']['cap_size'][0]
    rng = np.random.default_rng(212404)
    mat = cfg['material_profile']['cap']
    base = np.asarray(mat['base_rgb'], np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    radius = np.hypot((xx - (size - 1) / 2) / (size / 2),
                      (yy - (size - 1) / 2) / (size / 2))
    rings = .5 + .5 * np.sin(radius * 240 + rng.random() * math.tau)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32),
                            (0, 0), .5)
    low = _noise(size, rng, 12, 3.0)
    rgb = base[None, None, :] * (1 + .030 * (rings[:, :, None] - .5) +
                                 .006 * fine[:, :, None] +
                                 .012 * (low[:, :, None] - .5))
    rgb = np.clip(rgb, 0, 255)
    rough = np.clip(mat['base_roughness'] + .020 * (rings - .5) +
                    .010 * fine, .15, .78)
    metal = np.full((size, size), mat['metallic'], np.float32)
    height = .65 * rings + .25 * fine / 3 + .10 * low
    normal = _normal_from_height(height, .14, .7)
    paths = {}
    for name, arr in (('cap_base', rgb.astype(np.uint8)),
                      ('cap_mr', _mr_map(rough, metal)),
                      ('cap_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def build_bottom_textures(out, cfg):
    size = cfg['texture']['cap_size'][0]
    rng = np.random.default_rng(212505)
    mat = cfg['material_profile']['bottom']
    base = np.asarray(mat['base_rgb'], np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    radius = np.hypot((xx - (size - 1) / 2) / (size / 2),
                      (yy - (size - 1) / 2) / (size / 2))
    rings = .5 + .5 * np.sin(radius * 230 + rng.random() * math.tau)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32),
                            (0, 0), .5)
    low = _noise(size, rng, 12, 2.0)
    grime = np.clip((radius - .60) * 2.2, 0, 1) * np.clip((low - .35) * 1.7, 0, 1)
    rust_noise = _noise(size, rng, 56, 1.5)
    rust = np.clip((rust_noise - .78) * 3.0, 0, 1) * np.clip((radius - .55) * 2.0, 0, 1)
    grime_rgb = np.asarray(mat['grime_rgb'], np.float32)
    rust_rgb = np.asarray(mat['rust_rgb'], np.float32)
    rgb = base[None, None, :] * (1 + .025 * (rings[:, :, None] - .5) +
                                 .006 * fine[:, :, None] +
                                 .020 * (low[:, :, None] - .5))
    rgb = rgb * (1 - .15 * grime[:, :, None]) + \
        grime_rgb[None, None, :] * (.15 * grime[:, :, None])
    rust_alpha = .55 * rust
    rgb = rgb * (1 - rust_alpha[:, :, None]) + \
        rust_rgb[None, None, :] * rust_alpha[:, :, None]
    rgb = np.clip(rgb, 0, 255)
    metal = np.clip(1 - .18 * grime - .90 * rust, .03, 1)
    rough = np.clip(mat['base_roughness'] + .10 * grime + .42 * rust +
                    .020 * (rings - .5) + .008 * fine, .18, .82)
    height = .65 * rings + .20 * fine / 3 + .15 * rust + .10 * low
    normal = _normal_from_height(height, .15, .7)
    paths = {}
    for name, arr in (('bottom_base', rgb.astype(np.uint8)),
                      ('bottom_mr', _mr_map(rough, metal)),
                      ('bottom_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def texture_checks(textures, cfg):
    def load(key):
        return np.asarray(Image.open(textures[key]).convert('RGB')).astype(np.float32)

    body_base = load('body_base')
    body_mr = load('body_mr')
    body_normal = load('body_normal')
    body_luma = body_base.mean(axis=2)
    metal = body_mr[..., 2] / 255.0
    rough = body_mr[..., 1] / 255.0
    rust_mask = (metal < .35) & (rough > .55) & (body_luma < 150)
    rust_fraction = float(rust_mask.mean())
    body_normal_xy = float(body_normal[..., :2].std())
    body_ok = bool(120 <= float(body_luma.mean()) <= 220 and
                   float(metal.mean()) >= .80 and
                   .20 <= float(rough.mean()) <= .60 and
                   .02 <= rust_fraction <= .13 and
                   body_normal_xy >= 1.0)
    checks = {'body': {
        'mean_luma': round(float(body_luma.mean()), 2),
        'mean_metallic': round(float(metal.mean()), 4),
        'mean_roughness': round(float(rough.mean()), 4),
        'rust_fraction': round(rust_fraction, 4),
        'rust_mean_metallic': round(float(metal[rust_mask].mean()), 4) if rust_mask.any() else None,
        'rust_mean_roughness': round(float(rough[rust_mask].mean()), 4) if rust_mask.any() else None,
        'normal_xy_std': round(body_normal_xy, 3),
        'passed': body_ok,
        'note': ('Rust/dielectric mask = metallic<0.35, roughness>0.55 and '
                 'luma<150. The atlas is generated; the observed photo patches '
                 'supply only high-frequency detail and are not transferred as '
                 'a baked lit image.'),
    }}
    steel_base = load('steel_base')
    steel_mr = load('steel_mr')
    steel_normal = load('steel_normal')
    steel_normal_xy = float(steel_normal[..., :2].std())
    steel_ok = bool(120 <= float(steel_base.mean()) <= 220 and
                    float(steel_mr[..., 2].mean() / 255) >= .88 and
                    .15 <= float(steel_mr[..., 1].mean() / 255) <= .45 and
                    steel_normal_xy >= 1.0)
    checks['steel'] = {
        'mean_luma': round(float(steel_base.mean()), 2),
        'mean_metallic': round(float(steel_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(steel_mr[..., 1].mean() / 255), 4),
        'normal_xy_std': round(steel_normal_xy, 3),
        'passed': steel_ok,
    }
    black_base = load('black_base')
    black_mr = load('black_mr')
    black_normal = load('black_normal')
    black_normal_xy = float(black_normal[..., :2].std())
    black_ok = bool(15 <= float(black_base.mean()) <= 75 and
                    float(black_mr[..., 2].mean() / 255) <= .12 and
                    .24 <= float(black_mr[..., 1].mean() / 255) <= .70 and
                    black_normal_xy >= 1.0)
    checks['black'] = {
        'mean_luma': round(float(black_base.mean()), 2),
        'mean_metallic': round(float(black_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(black_mr[..., 1].mean() / 255), 4),
        'normal_xy_std': round(black_normal_xy, 3),
        'passed': black_ok,
        'note': ('The handwheel is modelled as black plastic/composite, so its '
                 'metallic is 0; the metal-reflection check does not apply to it.'),
    }
    cap_base = load('cap_base')
    cap_mr = load('cap_mr')
    cap_normal = load('cap_normal')
    cap_normal_xy = float(cap_normal[..., :2].std())
    cap_ok = bool(120 <= float(cap_base.mean()) <= 220 and
                  float(cap_mr[..., 2].mean() / 255) >= .85 and
                  .14 <= float(cap_mr[..., 1].mean() / 255) <= .75 and
                  cap_normal_xy >= 1.0)
    checks['cap'] = {
        'mean_luma': round(float(cap_base.mean()), 2),
        'mean_metallic': round(float(cap_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(cap_mr[..., 1].mean() / 255), 4),
        'normal_xy_std': round(cap_normal_xy, 3),
        'passed': cap_ok,
    }
    bottom_base = load('bottom_base')
    bottom_mr = load('bottom_mr')
    bottom_normal = load('bottom_normal')
    bottom_normal_xy = float(bottom_normal[..., :2].std())
    bottom_ok = bool(120 <= float(bottom_base.mean()) <= 220 and
                     float(bottom_mr[..., 2].mean() / 255) >= .80 and
                     .18 <= float(bottom_mr[..., 1].mean() / 255) <= .60 and
                     bottom_normal_xy >= 1.0)
    checks['bottom'] = {
        'mean_luma': round(float(bottom_base.mean()), 2),
        'mean_metallic': round(float(bottom_mr[..., 2].mean() / 255), 4),
        'mean_roughness': round(float(bottom_mr[..., 1].mean() / 255), 4),
        'normal_xy_std': round(bottom_normal_xy, 3),
        'passed': bottom_ok,
        'note': ('The bottom plate is a thin closed disc used to carry the '
                 'concentric brushed finish observed in the source; it is not '
                 'claimed to be a separate real component.'),
    }
    passed = all(v['passed'] for v in checks.values())
    return {'passed': passed, 'materials': checks,
            'note': ('Tile statistics only; they cannot prove hidden-surface '
                     'colour, exact rust distribution or scene-light response.')}


# ---------------------------------------------------------------------------
# Geometry / opening / reflection checks
# ---------------------------------------------------------------------------
def silhouette_checks(out):
    expected = {f'new-{view}': 0 for view in VIEWS}
    counts = {}
    passed = True
    for name, want in expected.items():
        with Image.open(out / 'renders' / f'{name}.png') as image:
            _, count = interior_holes(np.asarray(image.getchannel('A')))
        counts[name] = int(count)
        passed = passed and int(count) == want
    return {'expected': expected, 'detected': counts, 'passed': bool(passed),
            'method': ('Enclosed silhouette holes in the six isolated views. The '
                       'outlet bore is blind and must remain opaque end-on; the '
                       'real recess is proved by the ray and end-on luminance checks, '
                       'not by a silhouette hole.')}


def _first_hit(mesh, origin, direction):
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    h = np.cross(direction, edge2)
    a = np.einsum('ij,ij->i', edge1, h)
    valid = np.abs(a) > 1e-12
    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]
    s = origin - v0
    u = f * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, edge1)
    v = f * (q @ direction)
    t = f * np.einsum('ij,ij->i', edge2, q)
    hit = (valid & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-7))
    if not hit.any():
        return None
    return float(t[hit].min())


def opening_checks(out, worker):
    parts = load_parts(out / 'mesh.glb')
    spout = parts['Outlet spout']
    plug = parts['Safety plug body']
    sc = worker['spout']
    pl = worker['safety_plug']
    origin_y = 0.050
    center_t = _first_hit(spout, (0.0, origin_y, sc['z']), (0.0, -1.0, 0.0))
    offset_t = _first_hit(spout, (sc['bore_r'] * 1.30, origin_y, sc['z']),
                          (0.0, -1.0, 0.0))
    tip_t = origin_y - sc['y_tip']
    blind_t = origin_y - sc['blind_end']
    spout_ok = bool(center_t is not None and offset_t is not None and
                    center_t >= blind_t - 0.0015 and
                    offset_t <= tip_t + 0.0025 and
                    center_t - offset_t >= 0.015)
    plug_origin_y = -0.050
    plug_center_t = _first_hit(plug, (0.0, plug_origin_y, pl['center_z']),
                               (0.0, 1.0, 0.0))
    plug_offset_t = _first_hit(plug, (pl['socket_r'] * 1.80, plug_origin_y,
                                      pl['center_z']), (0.0, 1.0, 0.0))
    socket_bottom_y = pl['mount_y'] - (pl['length'] - pl['socket_depth'])
    outer_face_y = pl['mount_y'] - pl['length']
    plug_center_expected = -plug_origin_y + socket_bottom_y
    plug_offset_expected = -plug_origin_y + outer_face_y
    plug_ok = bool(plug_center_t is not None and plug_offset_t is not None and
                   plug_center_t >= plug_center_expected - 0.0015 and
                   plug_offset_t <= plug_offset_expected + 0.0015 and
                   plug_center_t - plug_offset_t >= 0.0012)
    return {
        'spout': {
            'center_first_hit_t': None if center_t is None else round(center_t, 5),
            'offset_first_hit_t': None if offset_t is None else round(offset_t, 5),
            'expected_tip_t': round(tip_t, 5),
            'expected_blind_t': round(blind_t, 5),
            'bore_depth_proved': spout_ok,
        },
        'safety_plug': {
            'center_first_hit_t': None if plug_center_t is None else round(plug_center_t, 5),
            'offset_first_hit_t': None if plug_offset_t is None else round(plug_offset_t, 5),
            'expected_socket_bottom_t': round(plug_center_expected, 5),
            'expected_outer_face_t': round(plug_offset_expected, 5),
            'socket_depth_proved': plug_ok,
        },
        'passed': bool(spout_ok and plug_ok),
        'method': ('Rays along the +Y outlet axis and the -Y safety-plug axis. '
                   'The centre ray must pass the open tip and first hit the blind '
                   'end much deeper than the parallel offset ray hits the outer '
                   'rim/face. This proves a real recess, not a painted circle. '
                   'It does not prove the internal valve seat or gas passage.'),
    }


def recess_check(out, worker):
    path = out / 'renders' / 'spout-only-end.png'
    if not path.is_file():
        return {'passed': False,
                'reason': 'isolated spout end-on render is produced only in full mode'}
    with Image.open(path) as image:
        a = np.asarray(image.convert('RGBA')).astype(np.float32)
    alpha = a[:, :, 3] >= 128
    ys, xs = np.nonzero(alpha)
    if not len(xs):
        return {'passed': False, 'reason': 'empty spout end-on render'}
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    radius = min(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    rr = np.hypot(xx - cx, yy - cy) / max(1e-9, radius)
    lum = .2126 * a[:, :, 0] + .7152 * a[:, :, 1] + .0722 * a[:, :, 2]
    center = alpha & (rr < .42)
    rim = alpha & (rr > .72) & (rr < .95)
    if center.sum() < 20 or rim.sum() < 20:
        return {'passed': False, 'reason': 'too few pixels for recess luma check'}
    center_luma = float(lum[center].mean())
    rim_luma = float(lum[rim].mean())
    return {
        'center_luma': round(center_luma, 3),
        'rim_luma': round(rim_luma, 3),
        'difference': round(rim_luma - center_luma, 3),
        'passed': bool(rim_luma - center_luma >= 8.0),
        'method': ('End-on orthographic render of the isolated outlet. The centre '
                   'disc (inside the bore) must be darker than the metal rim. This '
                   'is a supporting visual check; the ray distances are the primary '
                   'geometric proof.'),
    }


def _highlight_centroid(image):
    a = np.asarray(image.convert('RGBA')).astype(np.float32)
    mask = a[:, :, 3] >= 128
    if mask.sum() < 100:
        return None
    luma = .2126 * a[:, :, 0] + .7152 * a[:, :, 1] + .0722 * a[:, :, 2]
    threshold = max(120.0, float(np.percentile(luma[mask], 99.5)))
    selected = mask & (luma >= threshold)
    if selected.sum() < 8:
        return None
    ys, xs = np.nonzero(selected)
    return float(xs.mean()), float(ys.mean()), float(luma[selected].mean())


def reflection_checks(out):
    first = Image.open(out / 'renders' / 'new-material-close.png')
    second = Image.open(out / 'renders' / 'new-surface-alternate.png')
    if first.size != second.size:
        raise ValueError('metal light comparison images differ in size')
    a = np.asarray(first.convert('RGBA')).astype(np.float32)
    b = np.asarray(second.convert('RGBA')).astype(np.float32)
    mask = (a[:, :, 3] >= 128) & (b[:, :, 3] >= 128)
    delta = np.abs(a[:, :, :3] - b[:, :, :3]).mean(axis=2)[mask]
    ca, cb = _highlight_centroid(first), _highlight_centroid(second)
    shift = None
    if ca is not None and cb is not None:
        shift = round(float(math.hypot(ca[0] - cb[0], ca[1] - cb[1])), 3)
    mean_diff = float(delta.mean())
    changed = bool((shift is not None and shift > 1.0) or mean_diff > 2.0)
    return {
        'status': 'applicable_metal',
        'metal_response': {
            'mean_abs_luma_difference': round(mean_diff, 4),
            'highlight_centroid_a': None if ca is None else [round(v, 2) for v in ca],
            'highlight_centroid_b': None if cb is None else [round(v, 2) for v in cb],
            'highlight_shift_px': shift,
            'changed': changed,
            'criterion': ('highlight centroid shift > 1 px OR mean visible-metal '
                          'luma difference > 2.0; both values are recorded'),
            'note': ('Same close-up camera, different key-light direction. The '
                     'body, valve and metal cap must respond to the live light; '
                     'highlights are not baked into the base colour.'),
        },
        'dielectric_note': ('The black handwheel is modelled as plastic/composite '
                            'and is not used to claim a metal response; the close-up '
                            'is dominated by the aluminium body and steel valve.'),
    }


# ---------------------------------------------------------------------------
# Reference / source diagnostics
# ---------------------------------------------------------------------------
def _mask_iou(reference_mask, rendered_alpha):
    ref_fit = mask_fit(reference_mask)
    new_fit = mask_fit(np.asarray(rendered_alpha) >= 128)
    if ref_fit is None or new_fit is None:
        return {'iou': None, 'alignment': 'bounding-box fit to 400 px'}
    score = source_fidelity(ref_fit * 255, new_fit * 255)
    score['alignment'] = 'mask bounding-box fit to 400 px; not a calibrated camera'
    return score


def _model_iou(alpha_a, alpha_b):
    fit_a = mask_fit(np.asarray(alpha_a) >= 128)
    fit_b = mask_fit(np.asarray(alpha_b) >= 128)
    if fit_a is None or fit_b is None:
        return None
    return round(float(source_fidelity(fit_a * 255, fit_b * 255)['iou']), 4)


def reference_views(out, reference):
    image = Image.open(reference).convert('RGB')
    rows = []
    for name in VIEWS:
        panel = panel_image(image, name)
        ref_mask = reference_mask(panel)
        with Image.open(out / 'renders' / f'new-{name}.png') as new_image:
            new_alpha = new_image.getchannel('A')
        with Image.open(out / 'renders' / f'old-{name}.png') as old_image:
            old_alpha = old_image.getchannel('A')
        new_score = _mask_iou(ref_mask, new_alpha)
        old_score = _mask_iou(ref_mask, old_alpha)
        old_new = _model_iou(old_alpha, new_alpha)
        flipped_fit = mask_fit(np.fliplr(ref_mask))
        flipped = None
        if flipped_fit is not None:
            new_fit = mask_fit(np.asarray(new_alpha) >= 128)
            if new_fit is not None:
                flipped = source_fidelity(flipped_fit * 255, new_fit * 255)
        rows.append({
            'view': name,
            'reference_new_iou': new_score.get('iou'),
            'reference_new_filled_holes': new_score.get('filled_holes'),
            'reference_old_iou': old_score.get('iou'),
            'old_new_iou': old_new,
            'reference_new_iou_mirrored_reference': None if flipped is None else flipped.get('iou'),
            'reference_new_holes': new_score.get('holes'),
            'note': ('未标定图册；按外接框等比对齐的诊断值，不是测量精度或验收门槛。'
                     '图册左右视都画出气嘴且顶视比例与正视图不同，镜像 IoU 只用于'
                     '记录方向不确定性。'),
        })
    return rows


def reference_ratio_diagnostics(reference):
    image = Image.open(reference).convert('RGB')
    rows = []
    for name in VIEWS:
        panel = panel_image(image, name)
        mask = reference_mask(panel)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        rows.append({'view': name, 'panel': list(PANEL_BOX[name]),
                     'foreground_bbox': [int(xs.min()), int(ys.min()),
                                         int(xs.max()), int(ys.max())],
                     'foreground_fraction': round(float(mask.mean()), 4)})
    return rows


def source_diagnostics(out, alpha_path, worker):
    with Image.open(alpha_path) as alpha_image:
        source_alpha = alpha_image.convert('L')
    with Image.open(out / 'renders' / 'new-source.png') as rendered:
        strict = source_fidelity(np.asarray(source_alpha),
                                 np.asarray(rendered.getchannel('A')))
    strict['acceptance_gate'] = False
    source_mask = np.asarray(source_alpha) >= 128
    with Image.open(out / 'renders' / 'new-source.png') as rendered:
        aligned = _mask_iou(source_mask, rendered.getchannel('A'))
    aligned['acceptance_gate'] = False
    rolled = {}
    for roll in ('-12.0', '+12.0'):
        path = out / 'renders' / f'new-source-roll{roll}.png'
        if not path.is_file():
            continue
        with Image.open(path) as rendered:
            raw = source_fidelity(np.asarray(source_alpha),
                                  np.asarray(rendered.getchannel('A')))
            raw['acceptance_gate'] = False
            rolled[roll] = raw
    return {
        'strict': strict,
        'bbox_aligned': aligned,
        'rolled': rolled,
        'note': ('The source photo is a near-front view; the strict library camera '
                 'is fitted to the source-alpha height. The bbox-aligned IoU ignores '
                 'scale/position. Both are uncalibrated diagnostics, never acceptance '
                 'gates. The ±12° roll renders only show orientation sensitivity.'),
    }


def audit_row(number):
    if not AUDIT.is_file():
        return None
    for line in AUDIT.read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('number') == number:
                return row
    return None


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def _tile_image(path, size):
    image = Image.open(path).convert('RGBA')
    rect = image.getchannel('A').getbbox()
    if rect:
        image = image.crop(rect)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def _panel_tile(panel, mask, size, background='white'):
    ys, xs = np.nonzero(mask)
    if len(xs):
        x0, y0 = max(0, int(xs.min()) - 6), max(0, int(ys.min()) - 6)
        x1, y1 = min(panel.width, int(xs.max()) + 7), min(panel.height, int(ys.max()) + 7)
        panel = panel.crop((x0, y0, x1, y1))
    panel = panel.convert('RGBA')
    panel.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new('RGBA', (size[0] + 10, size[1] + 10), background)
    tile.alpha_composite(panel, ((tile.width - panel.width) // 2,
                                 (tile.height - panel.height) // 2))
    return tile


def make_report_21(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 22)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2240, 2620), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12),
              '21 气瓶2 ·  brushed-metal cylinder / valve / handwheel · 六视图（未标定） / 原模型 / 新候选',
              font=font, fill='#18242b')
    rows = reference_views(out, reference)
    for i, name in enumerate(VIEWS):
        y = 56 + i * 415
        x = 14
        panel = panel_image(reference_image, name)
        mask = reference_mask(panel)
        ref_tile = _panel_tile(panel, mask, (500, 348), 'white')
        row = rows[i]
        board.paste(ref_tile.convert('RGB'), (x, y + 26))
        draw.text((x + 4, y), '参考图册（未标定）', font=small, fill='#27363c')
        for j, (which, label) in enumerate((('old', '原模型（转正/缩放仅作对照）'),
                                            ('new', '新候选'))):
            image = _tile_image(out / 'renders' / f'{which}-{name}.png', (500, 348))
            tile = Image.new('RGBA', (520, 358), '#d9dedd')
            tile.alpha_composite(image, ((520 - image.width) // 2,
                                         (358 - image.height) // 2))
            board.paste(tile.convert('RGB'), (x + 530 + j * 550, y + 26))
            draw.text((x + 534 + j * 550, y), label, font=small, fill='#27363c')
        draw.text((x + 4, y + 386),
                  (f'参考/新诊断 IoU {row["reference_new_iou"]} · 参考/旧 '
                   f'{row["reference_old_iou"]} · 旧/新 {row["old_new_iou"]} · '
                   f'参考镜像/新 {row["reference_new_iou_mirrored_reference"]}'),
                  font=small, fill='#52646b')
    draw.text((20, 2540),
              ('重要：六视图图册未标定且不是一致转台：正视整体高宽比约 4.19，照片约 4.81；'
               '左视和右视都把出气嘴画在同一侧，顶视的物体像素尺寸也与正视图不一致。'
               '以上 IoU 仅是外接框对齐诊断，不是尺寸测量或验收通过。原模型为对照仅做了'
               '转正、缩放与平移，库文件未改动。'),
              font=small, fill='#8a2b20')
    board.save(out / '六视图-新旧对照.jpg', quality=95)
    return rows


def make_detail_board(out):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    items = (
        ('detail-front', '整体正视（材质）'),
        ('detail-back', '整体背视（材质）'),
        ('detail-side', '整体侧视（材质）'),
        ('detail-top', '顶视：手轮、出气嘴与阀体'),
        ('detail-bottom', '底视：闭口平底与圆边'),
        ('detail-valve', '阀体 3/4 局部'),
        ('detail-handwheel', '黑色手轮 3/4'),
        ('detail-handwheel-top', '手轮顶视：金属中心盖与螺钉'),
        ('detail-flutes', '手轮凹槽与边缘'),
        ('detail-spout', '出气嘴 3/4'),
        ('detail-spout-end', '出气嘴端面：真实盲孔'),
        ('detail-spout-side', '出气嘴侧视与螺纹'),
        ('detail-threads', '螺纹段近景'),
        ('detail-safety-plug', '左侧滚花安全塞与盲孔'),
        ('detail-plaque', '阀体正面铭牌（字迹不可辨）'),
        ('detail-neck-rings', '瓶颈薄环与阀座'),
        ('detail-shoulder', '肩部过渡与焊/旋压带'),
        ('detail-bottom-edge', '底缘圆角'),
        ('detail-surface', '筒身拉丝、划痕与麻点'),
        ('detail-rust-shoulder', '肩部锈斑与污渍'),
        ('detail-rust-bottom', '底缘锈斑与污渍'),
        ('new-clay-front', '灰模正视'),
        ('new-clay-close', '灰模阀体/手轮 3/4'),
    )
    tile_w, tile_h = 300, 760
    cols = 8
    rows = (len(items) + cols - 1) // cols
    board = Image.new('RGB', (tile_w * cols + 20, 60 + rows * (tile_h + 40)),
                      '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '21 气瓶2 · 关键局部检查（新候选；灰模为几何对照）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = 10 + (i % cols) * tile_w
        y = 55 + (i // cols) * (tile_h + 40)
        path = out / 'renders' / f'{name}.png'
        if not path.is_file():
            continue
        image = _tile_image(path, (tile_w - 18, tile_h - 30))
        tile = Image.new('RGBA', (tile_w - 8, tile_h), '#d9dedd')
        tile.alpha_composite(image, ((tile.width - image.width) // 2,
                                     (tile.height - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, board.height - 30),
              ('这些局部图用于人工检查结构；不能证明未观察面、阀内气路/阀座、'
               '螺纹配合、瓶壁厚度、落稳或现场尺度。'),
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, silhouette, rays, recess):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    items = (
        ('spout-only', '出气嘴隔离（3/4）'),
        ('spout-only-end', '出气嘴端面隔离：盲孔'),
        ('valve-spout-only', '阀体+出气嘴+安全塞隔离'),
        ('handwheel-only', '手轮+金属盖+螺钉隔离'),
        ('safety-plug-only', '滚花安全塞隔离'),
        ('neck-only', '瓶颈+颈环+阀体隔离'),
        ('top-view-all', '整体顶视'),
        ('new-clay-close', '灰模阀体/手轮 3/4'),
    )
    board = Image.new('RGB', (4 * 470 + 20, 2 * 930 + 70), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '21 气瓶2 · 隔离结构与真实开口检查',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = 10 + (i % 4) * 470
        y = 55 + (i // 4) * 930
        image = _tile_image(out / 'renders' / f'{name}.png', (440, 800))
        tile = Image.new('RGBA', (458, 815), '#d9dedd')
        tile.alpha_composite(image, ((458 - image.width) // 2,
                                     (815 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
        detail = ''
        if name == 'spout-only-end':
            detail = f'端面中心/边缘亮度差 {recess.get("difference")}'
        if detail:
            draw.text((x + 6, y + 815), detail, font=small,
                      fill='#1d5c3a' if recess.get('passed') else '#8a2b20')
    spout = rays['spout']
    plug = rays['safety_plug']
    ray_text = (
        '射线：出气嘴中心首次命中深度 {} m（预期盲端 {} m），偏置射线 {} m（预期端面 {} m），'
        '盲孔深度证明 {}；安全塞中心 {} m / 偏置 {} m，盲孔证明 {}。'
        '隔离渲染和射线只证明这些开口/凹槽是真实几何，不证明阀内气路、螺纹旋合或孔深测量。'
    ).format(spout.get('center_first_hit_t'), spout.get('expected_blind_t'),
             spout.get('offset_first_hit_t'), spout.get('expected_tip_t'),
             '通过' if spout.get('bore_depth_proved') else '未通过',
             plug.get('center_first_hit_t'), plug.get('offset_first_hit_t'),
             '通过' if plug.get('socket_depth_proved') else '未通过')
    draw.text((20, board.height - 44), ray_text, font=small,
              fill='#1d5c3a' if rays.get('passed') else '#8a2b20')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out, reflection):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '21 气瓶2 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（旧重建）'),
                                       ('new-material-close', '新候选：拉丝铝 / 钢阀体 / 黑手轮'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 830))
        tile = Image.new('RGBA', (680, 850), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (850 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), ('同一中性光与相机；高光来自实时照明，不烘焙在底色中。'
                          '原模型为单图重建，不代表旧模型整体通过。'),
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 960), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '21 气瓶2 · 变光检查（同视角，主光位置 A/B）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光 A · 正视'),
                                       ('new-light-alternate', '主光 B · 正视'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 780))
        tile = Image.new('RGBA', (680, 800), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (800 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 900), ('主光移动后高光和明暗应随视角变化；本图只检查反射响应，'
                          '不证明现场光照方向。'), font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    metal = reflection['metal_response']
    board = Image.new('RGB', (1440, 960), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '21 气瓶2 · 金属反射变光对照（同相机，主光 A/B）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光 A'),
                                       ('new-surface-alternate', '主光 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 780))
        tile = Image.new('RGBA', (680, 800), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (800 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 900),
              (f'可见金属平均亮度差 {metal["mean_abs_luma_difference"]}；'
               f'高光质心偏移 {metal["highlight_shift_px"]} px；'
               f'{"通过" if metal["changed"] else "未通过"}。'
               '高光随实时主光移动，不是烘焙底色；这不是现场光照验收。'),
              font=small, fill='#1d5c3a' if metal['changed'] else '#8a2b20')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_source_board(out, source_diag, clean, rust):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1500, 1030), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '21 气瓶2 · 原视角审计（照片近正视，非标定）',
              font=font, fill='#18242b')
    items = (
        (clean, '原貌照片', 'photo'),
        (rust, '做锈素材（同视角）', 'photo'),
        (out / 'renders' / 'new-source.png', '新候选 · 原相机拟合（诊断）', 'render'),
    )
    for i, (path, label, kind) in enumerate(items):
        x = 20 + i * 490
        if kind == 'photo':
            image = Image.open(path).convert('RGBA')
            white = Image.new('RGBA', image.size, (255, 255, 255, 255))
            white.alpha_composite(image)
            image = white.convert('RGB')
            rect = image.getbbox()
            if rect:
                image = image.crop(rect)
            image.thumbnail((450, 820), Image.Resampling.LANCZOS)
            tile = Image.new('RGB', (460, 830), 'white')
            tile.paste(image, ((460 - image.width) // 2,
                               (830 - image.height) // 2))
        else:
            image = _tile_image(path, (450, 820))
            tile = Image.new('RGBA', (460, 830), '#d9dedd')
            tile.alpha_composite(image, ((460 - image.width) // 2,
                                         (830 - image.height) // 2))
            tile = tile.convert('RGB')
        board.paste(tile, (x, 60))
        draw.text((x + 4, 34), label, font=small, fill='#27363c')
    strict = source_diag['strict']
    aligned = source_diag['bbox_aligned']
    rolled = source_diag.get('rolled', {})
    roll_text = '；'.join(
        f'{key}° IoU {value.get("iou")}' for key, value in rolled.items()) or '未渲染'
    draw.text((20, 920),
              (f'严格原相机 IoU {strict.get("iou")}（填孔 {strict.get("filled_holes")}，'
               f'细结构召回 {strict.get("narrow_recall")}）；外接框对齐 IoU '
               f'{aligned.get("iou")}；倾斜诊断 {roll_text}。相机缩放/对位为未标定'
               '最佳努力，均为诊断而非验收门槛。'),
              font=small, fill='#8a2b20')
    board.save(out / '原视角-对照.jpg', quality=95)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    tile = 250
    board = Image.new('RGB', (tile * 6, tile * 4 + 34), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '21 气瓶2 · 24 方向转台（中性光；透明背景）',
              font=font, fill='#27363c')
    for i in range(24):
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png').convert('RGBA')
        image.thumbnail((tile - 16, tile - 16), Image.Resampling.LANCZOS)
        cell = Image.new('RGBA', (tile, tile), '#d9dedd')
        cell.alpha_composite(image, ((tile - image.width) // 2,
                                     (tile - image.height) // 2))
        board.paste(cell.convert('RGB'), ((i % 6) * tile, 34 + (i // 6) * tile))
    board.save(out / '转台-24方向.jpg', quality=94)


def load_parts(path):
    scene = trimesh.load(path, force='scene')
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    return parts


def required_files(out, draft):
    files = [out / 'model.blend', out / 'mesh.glb',
             out / 'body_base.png', out / 'body_mr.png', out / 'body_normal.png',
             out / 'steel_base.png', out / 'steel_mr.png', out / 'steel_normal.png',
             out / 'black_base.png', out / 'black_mr.png', out / 'black_normal.png',
             out / 'cap_base.png', out / 'cap_mr.png', out / 'cap_normal.png',
             out / 'bottom_base.png', out / 'bottom_mr.png', out / 'bottom_normal.png',
             out / 'result.json', out / 'hashes.json', out / 'worker-result.json',
             out / 'request.json']
    if not draft:
        files += [out / '六视图-新旧对照.jpg', out / '局部检查.jpg',
                  out / '结构开口-隔离检查.jpg', out / '材质-同光照对照.jpg',
                  out / '材质-变光检查.jpg', out / '金属反射-变光对照.jpg',
                  out / '原视角-对照.jpg', out / '转台-24方向.jpg',
                  out / '浏览器核验.json', out / '检查页.html']
        files += [out / 'renders' / f'{which}-{view}.png'
                  for which in ('old', 'new') for view in VIEWS]
        files += [out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
    files += [out / 'renders' / f'new-{view}.png' for view in VIEWS]
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', type=int, default=21)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--draft', action='store_true', help='quick render')
    parser.add_argument('--library', type=Path,
                        default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    args = parser.parse_args()
    if args.number != 21:
        parser.error('this entry supports item 21 only')
    args.reference = (args.reference or find_reference()).resolve()
    if args.out is None:
        args.out = ROOT / ('outputs/21-gas-cylinder-draft1' if args.draft
                           else 'outputs/21气瓶2-六视图重建-v1')
    args.out, args.library, args.blender = (x.resolve() for x in
                                            (args.out, args.library, args.blender))
    if digest(args.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = next(r for r in (json.loads(line) for line in
                           args.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == 21)
    old_mesh = ROOT / OLD_REL
    alpha_path = ROOT / ALPHA_REL
    clean = ROOT / CLEAN_REL
    rust = ROOT / RUST_REL
    source_view = ROOT / SOURCE_VIEW_REL
    for path, expected, label in ((old_mesh, row['mesh_sha256'], 'old mesh'),
                                  (rust, row['rust_sha256'], 'rust image'),
                                  (clean, row['clean_sha256'], 'clean image'),
                                  (alpha_path, row['source_alpha_sha256'], 'source alpha')):
        if not path.is_file():
            raise FileNotFoundError(f'missing current-worktree input: {path}')
        if digest(path) != expected:
            raise ValueError(f'{label} hash differs from the model library')
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    if not len(xs):
        raise ValueError('source alpha is empty')
    source_alpha_bbox = [int(xs.min()), int(ys.min()),
                         int(xs.max()), int(ys.max())]
    old_scene = trimesh.load(old_mesh, force='scene')
    old_extent = float(np.max(old_scene.bounds[1] - old_scene.bounds[0]))
    config_path = HERE / 'gascylinder21_config.json'
    worker_path = HERE / 'mesh_rebuild_21_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(args.blender), '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    signature_input = {
        'reference': digest(args.reference), 'mesh': row['mesh_sha256'],
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
        'draft': args.draft,
    }
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if args.out.exists() and any(args.out.iterdir()):
        record = args.out / 'result.json'
        mesh = args.out / 'mesh.glb'
        if record.is_file() and mesh.is_file():
            old_record = json.loads(record.read_text(encoding='utf8'))
            if old_record.get('signature') == signature and \
                    old_record.get('mesh_sha256') == digest(mesh) and \
                    all(path.is_file() for path in required_files(args.out, args.draft)):
                print(json.dumps({'status': 'review', 'cache_hit': True,
                                  'out': str(args.out)}, ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    args.out.mkdir(parents=True, exist_ok=True)
    textures = {}
    textures.update(build_body_textures(rust, clean, args.out, config))
    textures.update(build_steel_textures(args.out, config))
    textures.update(build_black_textures(args.out, config))
    textures.update(build_cap_textures(args.out, config))
    textures.update(build_bottom_textures(args.out, config))
    request = {
        'out': str(args.out), 'original_mesh': str(old_mesh),
        'textures': textures, 'config': config,
        'source_camera': row['reconstruction']['source_camera'],
        'source_alpha_bbox': source_alpha_bbox,
        'old_span_raw': old_extent,
        'draft': args.draft,
    }
    (args.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2),
                                           encoding='utf8')
    with (args.out / 'blender.log').open('w', encoding='utf8') as log:
        process = subprocess.run(
            [str(args.blender), '--background', '--factory-startup',
             '--python-exit-code', '1', '--python', str(worker_path), '--',
             str(args.out / 'request.json')],
            stdout=log, stderr=subprocess.STDOUT, timeout=10800)
    if process.returncode:
        raise RuntimeError(f'Blender failed; see {args.out / "blender.log"}')

    worker = json.loads((args.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(21)
    mesh_audit = inspect_mesh(args.out / 'mesh.glb')
    parts = load_parts(args.out / 'mesh.glb')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'bodies': sum(name == 'Gas cylinder body' for name in names),
        'bottom_plates': sum(name == 'Bottom brushed plate' for name in names),
        'neck_rings': sum(name.startswith('Neck ring ') for name in names),
        'valve_bodies': sum(name == 'Valve body' for name in names),
        'spouts': sum(name == 'Outlet spout' for name in names),
        'safety_plug_bodies': sum(name == 'Safety plug body' for name in names),
        'safety_plug_knurls': sum(name == 'Safety plug knurl' for name in names),
        'plaques': sum(name == 'Valve nameplate' for name in names),
        'packing_nuts': sum(name == 'Valve packing nut' for name in names),
        'spindles': sum(name == 'Valve spindle' for name in names),
        'handwheels': sum(name == 'Black scalloped handwheel' for name in names),
        'caps': sum(name == 'Handwheel metal cap' for name in names),
        'screws': sum(name == 'Handwheel centre screw' for name in names),
        'note': ('Part presence proves only that a feature was modelled, not its '
                 'size, fit, hidden faces, valve function or internal gas passage.'),
    }
    expected = {
        'bodies': 1, 'bottom_plates': 1, 'neck_rings': 3, 'valve_bodies': 1,
        'spouts': 1, 'safety_plug_bodies': 1, 'safety_plug_knurls': 1,
        'plaques': 1, 'packing_nuts': 1, 'spindles': 1, 'handwheels': 1,
        'caps': 1, 'screws': 1,
    }
    structural_ok = all(structural[key] == value for key, value in expected.items())
    texture_ok = texture_checks(textures, config)
    silhouette = silhouette_checks(args.out)
    rays = opening_checks(args.out, worker)
    recess = recess_check(args.out, worker)
    reflection = reflection_checks(args.out)
    views = make_report_21(args.out, args.reference)
    reference_ratios = reference_ratio_diagnostics(args.reference)
    source_diag = source_diagnostics(args.out, alpha_path, worker)
    browser = {'status': 'not_run', 'reason': 'draft mode' if args.draft else 'pending'}
    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] == 15 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'structural_counts': bool(structural_ok),
        'spout_bore_rays': bool(rays['passed']),
        'spout_recess_visible': bool(recess['passed']),
        'silhouette_no_filled_holes': bool(silhouette['passed']),
        'material_tiles': bool(texture_ok['passed']),
        'metal_reflection': bool(reflection['metal_response']['changed']),
        'source_fidelity': bool(source_diag['strict'].get('passed', False)),
        'reference_diagnostic_only': True,
        'browser_page': browser.get('status') == 'passed',
        'manual_only': True,
    }
    report = {
        'version': config['version'],
        'number': 21,
        'asset_id': row['asset_id'],
        'name': row['file'],
        'status': 'review',
        'model_review': 'pending',
        'scene_validation': 'pending',
        'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_six_view_reference',
        'reference_role': 'visual_match_unverified',
        'method': ('parametric_part_rebuild_from_source_photo_with_uncalibrated_'
                   'six_view_reference'),
        'reference': str(args.reference),
        'reference_sha256': digest(args.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'model_identity': config['proportion_source']['model_identity'],
        'original_mesh': str(old_mesh),
        'original_mesh_sha256': row['mesh_sha256'],
        'original_clean': str(clean),
        'original_clean_sha256': row['clean_sha256'],
        'original_rust': str(rust),
        'original_rust_sha256': row['rust_sha256'],
        'source_view': str(source_view),
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9388 passed the numeric outer '
            'silhouette with zero filled holes and narrow recall 1.0. The six-view '
            'audit nonetheless shows a blurred, low-frequency surface, a fused '
            'valve/handwheel, no crisp outlet bore or thread ridges, and a speckled '
            'body. The old mesh is therefore not reused for the candidate base '
            'geometry.'),
        'catalog_evidence': {
            'category': '小型高压气瓶 / CO2-风格气瓶',
            'brand_visible': '未辨出品牌；照片与图册均无清晰铭文',
            'labels_visible': '无清晰可读标签；阀体正面有一块无字铭牌/凸台',
            'model': '具体型号未确认，不冒充实测型号',
            'note': ('No external product page was used to claim measured dimensions; '
                     'the photo and the uncalibrated atlas are the only evidence.'),
        },
        'nominal_height_m': config['proportion_source']['nominal_height_m'],
        'height_note': ('0.420 m is a plausible small CO2/air-cylinder height used to '
                        'scale the photo ratios; it is not measured.'),
        'inputs': signature_input,
        'signature': signature,
        'mesh': str(args.out / 'mesh.glb'),
        'mesh_sha256': digest(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'),
        'blend_sha256': digest(args.out / 'model.blend'),
        'textures': textures,
        'texture_checks': texture_ok,
        'material_profile': config['material_profile'],
        'source_fidelity_diagnostic': source_diag['strict'],
        'source_fidelity_bbox_aligned': source_diag['bbox_aligned'],
        'source_fidelity_rolled_diagnostics': source_diag['rolled'],
        'source_diagnostic_note': source_diag['note'],
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratios,
        'ratio_note': ('The six-view sheet is not calibrated: the front overall '
                       'height/width is about 4.19 versus 4.81 in the photo, and the '
                       'left/right panels both draw the outlet on the same side. '
                       'Reference/model IoU is a bounding-box-aligned silhouette '
                       'diagnostic, not a measurement or acceptance gate.'),
        'mesh_audit': mesh_audit,
        'structural_checks': structural,
        'structural_expected': expected,
        'isolated_opening_checks': {'rays': rays, 'recess': recess},
        'rendered_silhouette_checks': silhouette,
        'reflection_checks': reflection,
        'metal_reflection_check': {'status': reflection['status'],
                                   **reflection['metal_response']},
        'checks': checks,
        'browser_check': browser,
        'worker': worker,
        'draft': args.draft,
        'limitations': [
            'The six-view atlas is uncalibrated and internally inconsistent: its front height/width is about 4.19 versus 4.81 in the photo, the left and right panels both draw the outlet on the same side, and the top/bottom disc scale differs from the side panels. It is used for topology and appearance only.',
            'All dimensions follow photo pixel ratios and a plausible 0.420 m nominal height; they are not measured. Body diameter, shoulder radius, neck length, valve size, outlet bore and thread pitch are inferred.',
            'The outlet thread is modelled as a revolved annular-groove simplification, not a true helical thread; the thread form, pitch, thread standard and engagement are not confirmed.',
            'The outlet bore is blind and ends at a modelled internal wall. The internal valve seat, gas passage, safety mechanism and whether the plug is a burst disc, bleed screw or blanking plug are not represented or confirmed.',
            'The knurled safety plug socket is modelled as a round recess; the exact drive type (hex, Torx, slot) is not legible in the photo.',
            'The valve nameplate is modelled as a plain raised plate; any stamped text is not legible and was not invented.',
            'The handwheel is modelled as black plastic/composite with 14 scallops, based on about 7-8 visible grooves on the front half of the photo; the exact count, material and internal attachment are inferred. It is dielectric.',
            'The body rust and grime pattern is synthesized from the photo/rust-material statistics and small observed patches; it is not a pixel-accurate transfer and the hidden side is not observed.',
            'The concentric bottom finish is carried by a thin closed disc (Bottom brushed plate) inside the rounded body rim; it is a material/geometry convenience, not a claim that the real cylinder has a separate bottom plate.',
            'The neck rings and the shoulder are closed revolved surfaces; no internal cylinder wall, valve thread or pressure boundary is represented.',
            'Assemblies intentionally overlap at the body/neck/valve, spout/valve, plug/valve, plaque/valve, cap/wheel and screw/cap interfaces; BVH checks only same-part self-intersections, not part-to-part fit.',
            'The old model was rotated/scaled/translated for display comparison only; that transformation is not a repair and the library file is untouched.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the source photo and atlas, especially the outlet bore/threads, safety plug, valve silhouette, handwheel scallops and neck rings.',
            'Confirm the exact model identity and any manufacturer stamping from a physical sample; no brand or model is claimed.',
            'Measure the cylinder diameter/height, neck/valve dimensions, outlet bore/thread standard, handwheel diameter and safety-plug drive if exact geometry is required.',
            'Confirm hidden geometry: internal valve, gas passage, safety device, thread engagement, wall thickness and the handwheel attachment.',
            'Verify the rust/grime distribution on the hidden side and under final scene lighting; the current check only proves highlight movement between two studio key-light directions.',
            'Verify coordinate system, scale, stability, floor contact and placement in the manual review page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated orthographic front/side/top views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 22,
    }
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf8')
    if not args.draft:
        make_detail_board(args.out)
        make_isolated_board(args.out, silhouette, rays, recess)
        make_material_boards(args.out, reflection)
        make_source_board(args.out, source_diag, clean, rust)
        make_turntable(args.out)
        from review21_page import build_page, browser_check
        build_page(args.out)
        browser = browser_check(args.out)
        checks['browser_page'] = browser.get('status') == 'passed'
        report['checks'] = checks
        report['browser_check'] = browser
        (args.out / 'result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        build_page(args.out)
        browser = browser_check(args.out)
        checks['browser_page'] = browser.get('status') == 'passed'
        report['checks'] = checks
        report['browser_check'] = browser
        (args.out / 'result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        build_page(args.out)
    files = [path for path in sorted(args.out.rglob('*')) if path.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_21.py',
              'files': {str(path.relative_to(args.out)).replace('\\', '/'): digest(path)
                        for path in files}}
    hashes['files']['hashes.json'] = 'self'
    (args.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                          encoding='utf8')
    progress = {
        'number': 21,
        'name': row['file'],
        'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'],
        'out': str(args.out),
        'result': str(args.out / 'result.json'),
        'mesh': str(args.out / 'mesh.glb'),
        'blend': str(args.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(args.out / '六视图-新旧对照.jpg'),
        'detail_board': str(args.out / '局部检查.jpg'),
        'isolated_board': str(args.out / '结构开口-隔离检查.jpg'),
        'checks': checks,
        'texture_checks': texture_ok,
        'isolated_opening_checks': report['isolated_opening_checks'],
        'reference_views': views,
        'source_fidelity_diagnostic': source_diag['strict'],
        'source_fidelity_bbox_aligned': source_diag['bbox_aligned'],
        'reference_role': 'visual_match_unverified',
        'manual_only': True,
        'pending_review': report['pending_review'],
        'next_number': 22,
        'notes': ('The six-view atlas is an uncalibrated visual-match reference; the '
                  'source photo is the primary shape evidence. The old model and other '
                  'items were not modified. Candidate is manual_only; scene placement, '
                  'stability and formal compositing are not validated.'),
    }
    (args.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                            encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '21-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(args.out), 'checks': checks,
                      'source_fidelity': source_diag['strict'],
                      'bbox_aligned': source_diag['bbox_aligned']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
