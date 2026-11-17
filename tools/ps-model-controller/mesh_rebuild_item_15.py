"""Build the review-only Raid-type aerosol-can candidate (item 15).

The original clean/rust photograph decides proportions and the visible front
structure.  The user six-view sheet is an uncalibrated visual match; its panels
are used only for the hidden-half label wrap and the top/bottom topology, never
as measured or identical-unit geometry.  The candidate is a part-based rebuild:
a closed printed tinplate body, brushed concave metal base and rusty rolled rim,
a glossy black plastic overcap with two raised side horns and a real two-lobed
window, an internal valve stem/actuator/trigger, and a two-bore nozzle face.
It never replaces the library mesh, is manual-only and never enters automatic
placement.  A new output directory is required.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import PIL
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import binary_fill_holes

from mesh_carve import interior_holes
from mesh_checks import source_fidelity
from mesh_rebuild_item import inspect_mesh, mask_fit
from scene_v2 import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
REFERENCE_SHA256 = '662069f8e42e0cb0b50fc0ba6c6109b9cdf6cbad3290af51b510216c919355e4'
REFERENCE_NAME = 'ChatGPT 图像 2026年10月14日 10_08_40-2.png'
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'


def font(size):
    try:
        return ImageFont.truetype(LABEL_FONT, size)
    except OSError:
        return ImageFont.load_default()


def find_reference():
    folder = ROOT / 'references/parallel-six-view'
    if not folder.is_dir():
        raise FileNotFoundError('references/parallel-six-view is missing')
    for path in sorted(folder.glob('*.png')):
        if digest(path) == REFERENCE_SHA256:
            return path
    raise FileNotFoundError('registered six-view reference not found')


def audit_row(number):
    if not AUDIT.is_file():
        return None
    for line in AUDIT.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('number') == number:
            return row
    return None


def library_row(number):
    for line in LIBRARY.read_text(encoding='utf8').splitlines():
        if line.strip() and json.loads(line).get('number') == number:
            return json.loads(line)
    raise KeyError(number)


def localize(path):
    """Rebase a library path from the old main worktree onto this worktree.

    The library records absolute paths from the original main checkout.  This
    worktree keeps the same relative layout, so the recorded relative tail is
    looked up here; reading the old main worktree is not allowed.
    """
    p = Path(path)
    if not p.is_absolute():
        return (ROOT / p).resolve()
    try:
        p.resolve().relative_to(ROOT.resolve())
        return p.resolve()
    except ValueError:
        pass
    for marker in ('outputs', 'work', 'external', 'references', 'vendor'):
        if marker in p.parts:
            index = len(p.parts) - 1 - p.parts[::-1].index(marker)
            return ROOT.joinpath(*p.parts[index:]).resolve()
    raise ValueError(f'cannot localize path: {path}')


def wrap_pi(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Reference panels
# ---------------------------------------------------------------------------
def panel_origin(name):
    return {'front': (0, 0), 'back': (512, 0), 'left': (1024, 0),
            'right': (0, 512), 'top': (512, 512), 'bottom': (1024, 512)}[name]


def panel_rgb(reference, name):
    image = Image.open(reference).convert('RGB')
    ox, oy = panel_origin(name)
    return np.asarray(image.crop((ox, oy, ox + 512, oy + 512))).astype(np.float32)


def panel_crop(reference, config, name):
    p = config['reference_panels'][name]
    ox, oy = p['origin']
    x0, y0, x1, y1 = p['rect']
    image = Image.open(reference).convert('RGB')
    return image.crop((ox + x0, oy + y0, ox + x1, oy + y1))


def reference_can_crop(reference, name):
    rgb = panel_rgb(reference, name)
    luma = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mask = ((luma < 245) | (chroma > 12)).astype(np.uint8)
    mask[:6] = 0
    mask[-40:] = 0
    mask[:, :6] = 0
    mask[:, -6:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(200, areas.max() * .05)) + 1
    mask = np.isin(labels, keep).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    pad = 8
    return Image.fromarray(rgb.astype(np.uint8)).crop(
        (max(0, xs.min() - pad), max(0, ys.min() - pad),
         min(512, xs.max() + pad + 1), min(512, ys.max() + pad + 1)))


def reference_mask_panel(reference, name):
    rgb = panel_rgb(reference, name)
    luma = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mask = ((luma < 246) | (chroma > 14)).astype(np.uint8)
    mask[:8] = 0
    mask[-45:] = 0
    mask[:, :8] = 0
    mask[:, -8:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(300, areas.max() * .03)) + 1
    return np.isin(labels, keep).astype(np.uint8)


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------
def resize_lanczos(array, size):
    return cv2.resize(array.astype(np.float32), size,
                      interpolation=cv2.INTER_LANCZOS4)


def bilinear(image, sx, sy):
    h, w = image.shape[:2]
    x0 = np.clip(np.floor(sx).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(np.int32), 0, h - 2)
    fx = np.clip(sx - x0, 0, 1)[..., None]
    fy = np.clip(sy - y0, 0, 1)[..., None]
    c00 = image[y0, x0]
    c10 = image[y0, x0 + 1]
    c01 = image[y0 + 1, x0]
    c11 = image[y0 + 1, x0 + 1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + \
        (c01 * (1 - fx) + c11 * fx) * fy


def make_label_textures(row, reference, out, config):
    clean = Image.open(row['original']).convert('RGB')
    rust = Image.open(row['rust']).convert('RGB')
    x0, y0, x1, y1 = config['hd_label_rect']
    hd = np.asarray(clean.crop((x0, y0, x1, y1))).astype(np.float32)
    hd_rust = np.asarray(rust.crop((x0, y0, x1, y1))).astype(np.float32)
    if hd.shape != hd_rust.shape:
        raise ValueError('clean/rust label crops differ in size')

    size = 2048
    front_w, front_h = 1024, 2048
    hd_rs = resize_lanczos(hd, (front_w, front_h))
    rust_rs = resize_lanczos(hd_rust, (front_w, front_h))

    # Flat-field the studio photo against the evenly lit uncalibrated front
    # reference.  The reference is not a calibrated albedo, so only a heavily
    # smoothed illumination ratio is used; the photo keeps the high-frequency
    # print detail and colour.
    ref_front = panel_crop(reference, config, 'front')
    ref_front = np.asarray(ref_front).astype(np.float32)
    ref_rs = resize_lanczos(ref_front, (front_w, front_h))
    l_hd = cv2.cvtColor(hd_rs.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    l_ref = cv2.cvtColor(ref_rs.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ratio = (cv2.GaussianBlur(l_hd, (0, 0), 55) + 8) / \
        (cv2.GaussianBlur(l_ref, (0, 0), 55) + 8)
    ratio = cv2.GaussianBlur(np.clip(ratio, 0.60, 1.60), (0, 0), 24)
    ratio /= max(1e-6, float(np.median(ratio)))
    hd_flat = np.clip(hd_rs / ratio[..., None], 0, 255)

    # Grime from the rust/clean alpha-aligned pair.  Rust is browner and darker
    # than the clean print; the smoothed field is used as a PBR roughness mask.
    l_clean = cv2.cvtColor(hd_rs.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    l_rust = cv2.cvtColor(rust_rs.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = np.clip((l_clean - l_rust) / np.maximum(l_clean, 25.0) * 1.7, 0, 1)
    b_ratio = (rust_rs[..., 2] + 4) / (hd_rs[..., 2] + 4)
    brown = np.clip((1.0 - b_ratio) * 2.0, 0, 1)
    grime = np.clip(dark * 0.60 + brown * 0.40, 0, 1)
    grime = (0.62 * cv2.GaussianBlur(grime, (0, 0), 2.2) +
             0.38 * cv2.GaussianBlur(grime, (0, 0), 15.0))
    grime = np.clip(grime * 1.15, 0, 1)
    grime_rgb = np.asarray(config['material_profile']['label']['grime_rgb'],
                           np.float32)
    hd_grimed = (hd_flat * (1.0 - 0.34 * grime[..., None]) +
                 grime_rgb[None, None, :] * (0.34 * grime[..., None]))

    # Upscale the other reference panels.  These are uncalibrated but they are
    # the only evidence for the hidden half of the printed wrap.
    panel_data = {}
    for name in ('front', 'left', 'back', 'right'):
        if name == 'front':
            continue
        crop = np.asarray(panel_crop(reference, config, name)).astype(np.float32)
        panel_data[name] = resize_lanczos(crop, (front_w, front_h))
    panel_data['hd'] = np.clip(hd_grimed, 0, 255)
    # The reference sheet labels its side panels from the object's sides, while
    # the project cameras use -Y for "left" and +Y for "right".  The sheet's
    # left panel is therefore the +Y (project right) view and vice versa.
    angles = {'hd': 0.0, 'left': math.radians(90), 'back': math.radians(180),
              'right': math.radians(270)}
    weights = {'hd': 2.6, 'left': 0.75, 'back': 0.75, 'right': 0.75}

    # Colour-normalise the reference panels to the photo's saturated magenta
    # areas so that side/back transitions do not show a strong cast.
    def magenta_median(image):
        flat = image.reshape(-1, 3)
        sat = flat.max(axis=1) - flat.min(axis=1)
        keep = (sat > 55) & (flat.mean(axis=1) > 35)
        if keep.sum() < 50:
            return None
        return np.median(flat[keep], axis=0)

    target = magenta_median(hd_flat)
    if target is not None:
        for name in ('left', 'back', 'right'):
            med = magenta_median(panel_data[name])
            if med is not None:
                gain = np.clip(target / np.maximum(med, 1.0), 0.72, 1.35)
                panel_data[name] = np.clip(panel_data[name] * gain, 0, 255)

    u_phi = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 * math.pi
    v = 1.0 - (np.arange(size, dtype=np.float32) + 0.5) / size
    acc = np.zeros((size, size, 3), np.float32)
    wacc = np.zeros((size, size), np.float32)
    grime_acc = np.zeros((size, size), np.float32)
    hd_weight = np.zeros((size, size), np.float32)
    for name, image in panel_data.items():
        h, w = image.shape[:2]
        d = wrap_pi(u_phi[None, :] - angles[name])
        valid = np.abs(d) < (math.pi / 2 - 1e-4)
        sx = (w - 1) / 2 + (w / 2 - 1) * np.sin(d)
        sy = (1.0 - v[:, None]) * (h - 1)
        sample = bilinear(image, np.broadcast_to(sx, (size, size)),
                          np.broadcast_to(sy, (size, size)))
        cosd = np.clip(np.cos(d), 0, 1)
        # Compensate the broad cylindrical shading baked into the source
        # panels; the 3D render will add its own live illumination.
        sample = sample / (0.86 + 0.14 * cosd)[..., None]
        weight = np.where(valid, (cosd ** 1.2) * weights[name], 0.0)
        acc += sample * weight[..., None]
        wacc += weight
        if name == 'hd':
            grime_sample = bilinear(grime[..., None], sx, sy)[..., 0]
            grime_acc += grime_sample * weight
            hd_weight += weight
    atlas = acc / np.maximum(wacc[..., None], 1e-6)
    grime_atlas = grime_acc / np.maximum(hd_weight, 1e-6)

    # Procedural grime for the hidden half, where only the uncalibrated side
    # panels exist.  It is weaker than the measured front grime.
    rng = np.random.default_rng(1515)
    low = cv2.resize(rng.random((48, 48)).astype(np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((768, 768)).astype(np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    streak = cv2.GaussianBlur((low - 0.5) * 2.4, (0, 0), 2.0)
    field = np.clip(0.60 * (low - 0.42) + 0.28 * streak + 0.12 * (fine - 0.5),
                    0, 1)
    front_norm = hd_weight / np.maximum(wacc, 1e-6)
    edge = np.clip((np.abs(v - 0.5) - 0.30) / 0.20, 0, 1)
    proc = field * (0.20 + 0.65 * (1.0 - front_norm)) * (0.32 + 0.68 * edge)
    grime_atlas = np.clip(grime_atlas * 1.0 + proc * 0.55, 0, 1)
    atlas = (atlas * (1.0 - 0.18 * proc[..., None]) +
             grime_rgb[None, None, :] * (0.18 * proc[..., None]))
    atlas = np.clip(atlas, 0, 255)
    atlas_img = Image.fromarray(atlas.astype(np.uint8))
    atlas_img = atlas_img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=55,
                                                         threshold=2))
    atlas = np.asarray(atlas_img).astype(np.float32)
    # A very small print grain avoids a suspicious plastic-perfect atlas while
    # staying far below the visible label linework.
    grain = rng.normal(0, 1.0, (size, size, 1)).repeat(3, axis=2)
    atlas = np.clip(atlas + grain, 0, 255)
    base_path = out / 'label_base.png'
    Image.fromarray(atlas.astype(np.uint8)).save(base_path)

    profile = config['material_profile']['label']
    rough = (profile['base_roughness'] +
             0.035 * (fine - 0.5) +
             0.09 * np.clip(grime_atlas * 1.4, 0, 1) +
             0.04 * (low - 0.5))
    rough = np.clip(rough, 0.22, 0.78)
    mr = np.stack((np.full(rough.shape, 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   np.zeros(rough.shape, np.uint8)), axis=2)
    mr_path = out / 'label_mr.png'
    Image.fromarray(mr).save(mr_path)

    peel = cv2.GaussianBlur((low - 0.5) * 1.4 + (fine - 0.5) * 0.8, (0, 0), 1.0)
    height = peel * 0.55 + 0.45 * cv2.GaussianBlur(fine - 0.5, (0, 0), 2.2)
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 18
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 18
    nx = np.clip(-dx * 0.55, -.035, .035)
    ny = np.clip(dy * 0.55, -.035, .035)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * 0.5 + 0.5) * 255,
                     0, 255).astype(np.uint8)
    normal_path = out / 'label_normal.png'
    Image.fromarray(normal).save(normal_path)
    return {'label_base': str(base_path), 'label_mr': str(mr_path),
            'label_normal': str(normal_path),
            'grime_ratio_front': round(float(np.mean(grime[grime > 0.15])), 4)
            if (grime > 0.15).any() else 0.0}


def make_cap_textures(out, config):
    size = 1024
    rng = np.random.default_rng(1516)
    low = cv2.resize(rng.random((42, 42)).astype(np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((size, size)).astype(np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    peel = cv2.GaussianBlur(rng.normal(size=(size // 4, size // 4)).astype(np.float32),
                            (0, 0), 1.1)
    peel = cv2.resize(peel, (size, size), interpolation=cv2.INTER_CUBIC)
    peel /= max(1e-6, float(peel.std()))
    base_rgb = np.asarray(config['material_profile']['cap']['base_rgb'], np.float32)
    dust = np.clip((low - 0.72) * 2.2, 0, 1) * 0.28
    base = base_rgb[None, None, :] * (1.0 + (low - 0.5)[..., None] * 0.10 +
                                      (fine - 0.5)[..., None] * 0.18 +
                                      peel[..., None] * 0.03)
    base = base + np.asarray((55, 38, 24), np.float32)[None, None, :] * dust[..., None]
    base = np.clip(base, 0, 255)
    base_path = out / 'cap_base.png'
    Image.fromarray(base.astype(np.uint8)).save(base_path)
    rough = (config['material_profile']['cap']['base_roughness'] +
             (low - 0.5) * 0.035 + (fine - 0.5) * 0.025 +
             dust * 0.22 + peel * 0.008)
    rough = np.clip(rough, 0.115, 0.42)
    mr = np.stack((np.full(rough.shape, 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   np.zeros(rough.shape, np.uint8)), axis=2)
    mr_path = out / 'cap_mr.png'
    Image.fromarray(mr).save(mr_path)
    height = 0.55 * peel + 0.45 * (fine - 0.5)
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 16
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 16
    nx = np.clip(-dx * 0.55, -.06, .06)
    ny = np.clip(dy * 0.55, -.06, .06)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * 0.5 + 0.5) * 255,
                     0, 255).astype(np.uint8)
    normal_path = out / 'cap_normal.png'
    Image.fromarray(normal).save(normal_path)
    return {'cap_base': str(base_path), 'cap_mr': str(mr_path),
            'cap_normal': str(normal_path)}


def make_metal_rust_textures(out, config):
    size = 1024
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2
    radius = np.hypot(xx - cx, yy - cy)
    rng = np.random.default_rng(1517)
    fine = cv2.resize(rng.random((size, size)).astype(np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    low = cv2.resize(rng.random((52, 52)).astype(np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    ring = np.sin(radius * (2 * math.pi / 7.0)) * np.exp(-radius / 720.0)
    scratch = np.zeros((size, size), np.float32)
    for _ in range(38):
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(50, 260)
        rr = rng.uniform(30, 480)
        px = cx + rr * math.cos(angle)
        py = cy + rr * math.sin(angle)
        x0 = int(np.clip(px - length * 0.5, 0, size - 1))
        x1 = int(np.clip(px + length * 0.5, 0, size - 1))
        y0 = int(np.clip(py, 0, size - 1))
        cv2.line(scratch, (x0, y0), (x1, y0), 1.0, 1, cv2.LINE_AA)
    metal_base = (np.asarray(config['material_profile']['metal']['base_rgb'],
                             np.float32)[None, None, :] *
                  (1.0 + ring[..., None] * 0.045 +
                   (low - 0.5)[..., None] * 0.055 +
                   (fine - 0.5)[..., None] * 0.025 +
                   scratch[..., None] * 0.18))
    metal_base = np.clip(metal_base, 0, 255)
    Image.fromarray(metal_base.astype(np.uint8)).save(out / 'metal_base.png')
    rough = np.clip(0.265 + ring * 0.030 + (fine - 0.5) * 0.035 +
                    scratch * 0.20 + (low - 0.5) * 0.035, 0.16, 0.52)
    mr = np.stack((np.full(rough.shape, 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   np.full(rough.shape, 255, np.uint8)), axis=2)
    Image.fromarray(mr).save(out / 'metal_mr.png')
    height = 0.60 * ring + 0.30 * (fine - 0.5) + 0.10 * scratch
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 15
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 15
    nx = np.clip(-dx * 0.55, -.08, .08)
    ny = np.clip(dy * 0.55, -.08, .08)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * 0.5 + 0.5) * 255,
                     0, 255).astype(np.uint8)
    Image.fromarray(normal).save(out / 'metal_normal.png')

    # Rust: mottled brown oxide with dark pits and a few remaining metal flecks.
    coarse = cv2.resize(rng.random((36, 36)).astype(np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    coarse = cv2.GaussianBlur(coarse, (0, 0), 3.0)
    pits = np.clip((coarse - 0.58) * 4.0, 0, 1)
    flakes = np.clip((fine - 0.72) * 5.0, 0, 1)
    rust_rgb = np.asarray((92, 52, 27), np.float32)[None, None, :]
    rust_dark = np.asarray((43, 24, 14), np.float32)[None, None, :]
    rust_light = np.asarray((146, 82, 39), np.float32)[None, None, :]
    rust_base = (rust_rgb * (1.0 + (coarse - 0.5)[..., None] * 0.55) +
                 rust_light * (flakes[..., None] * 0.55) -
                 rust_dark * (pits[..., None] * 0.40))
    rust_base = np.clip(rust_base, 0, 255)
    Image.fromarray(rust_base.astype(np.uint8)).save(out / 'rust_base.png')
    rust_rough = np.clip(0.76 + (coarse - 0.5) * 0.20 + (fine - 0.5) * 0.10 -
                         flakes * 0.18, 0.55, 0.94)
    rust_metal = np.clip(flakes * 0.55 + 0.03, 0, 1)
    mr = np.stack((np.full(rust_rough.shape, 255, np.uint8),
                   (rust_rough * 255).astype(np.uint8),
                   (rust_metal * 255).astype(np.uint8)), axis=2)
    Image.fromarray(mr).save(out / 'rust_mr.png')
    height = 0.65 * coarse + 0.35 * fine - 0.45 * pits + 0.25 * flakes
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 11
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 11
    nx = np.clip(-dx * 0.55, -.16, .16)
    ny = np.clip(dy * 0.55, -.16, .16)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * 0.5 + 0.5) * 255,
                     0, 255).astype(np.uint8)
    Image.fromarray(normal).save(out / 'rust_normal.png')

    # Dark dust variant for the cap seam.  It has its own dark base colour,
    # roughness and normal tiles so the seam reads as grime, not orange rust.
    dust_base = np.clip(
        np.asarray((42, 30, 23), np.float32)[None, None, :] *
        (1.0 + (coarse - 0.5)[..., None] * 0.55 +
         (fine - 0.5)[..., None] * 0.22), 0, 255)
    Image.fromarray(dust_base.astype(np.uint8)).save(out / 'dust_base.png')
    dust_rough = np.clip(0.64 + (coarse - 0.5) * 0.18 + (fine - 0.5) * 0.08,
                         0.42, 0.86)
    mr = np.stack((np.full(dust_rough.shape, 255, np.uint8),
                   (dust_rough * 255).astype(np.uint8),
                   np.zeros(dust_rough.shape, np.uint8)), axis=2)
    Image.fromarray(mr).save(out / 'dust_mr.png')
    Image.fromarray(normal).save(out / 'dust_normal.png')
    return {
        'metal_base': str(out / 'metal_base.png'),
        'metal_mr': str(out / 'metal_mr.png'),
        'metal_normal': str(out / 'metal_normal.png'),
        'rust_base': str(out / 'rust_base.png'),
        'rust_mr': str(out / 'rust_mr.png'),
        'rust_normal': str(out / 'rust_normal.png'),
        'dust_base': str(out / 'dust_base.png'),
        'dust_mr': str(out / 'dust_mr.png'),
        'dust_normal': str(out / 'dust_normal.png'),
    }


def texture_stats(base, mr, name):
    base_rgb = np.asarray(Image.open(base).convert('RGB')).astype(np.float32)
    mr_rgb = np.asarray(Image.open(mr).convert('RGB')).astype(np.float32)
    luma = base_rgb.mean(axis=2)
    chroma = base_rgb.max(axis=2) - base_rgb.min(axis=2)
    return {
        'texture': name,
        'mean_luma': round(float(luma.mean()), 3),
        'std_luma': round(float(luma.std()), 3),
        'mean_chroma': round(float(chroma.mean()), 3),
        'mean_roughness': round(float(mr_rgb[..., 1].mean() / 255), 4),
        'mean_metallic': round(float(mr_rgb[..., 2].mean() / 255), 4),
        'roughness_range': [round(float(mr_rgb[..., 1].min() / 255), 4),
                            round(float(mr_rgb[..., 1].max() / 255), 4)],
    }


def label_texture_checks(base, mr, normal):
    stats = texture_stats(base, mr, 'label')
    base_rgb = np.asarray(Image.open(base).convert('RGB')).astype(np.float32)
    h, w = base_rgb.shape[:2]
    # Front seam is u=0/1; use the two edge strips, excluding the extreme seam
    # texel.  Back information panel is around u=0.5.
    front = np.concatenate((base_rgb[:, :int(w * .05)],
                            base_rgb[:, int(w * .95):]), axis=1)
    back = base_rgb[:, int(w * .44):int(w * .56)]
    front_luma = float(front.mean())
    back_luma = float(back.mean())
    front_chroma = float((front.max(axis=2) - front.min(axis=2)).mean())
    normal_rgb = np.asarray(Image.open(normal).convert('RGB')).astype(np.float32)
    stats.update({
        'front_luma': round(front_luma, 3),
        'back_luma': round(back_luma, 3),
        'front_chroma': round(front_chroma, 3),
        'normal_std': round(float(normal_rgb.std()), 3),
    })
    stats['passed'] = bool(front_luma > 65 and back_luma > 65 and
                           front_chroma > 16 and stats['std_luma'] > 12 and
                           stats['mean_metallic'] < 0.05 and
                           0.20 <= stats['mean_roughness'] <= 0.62 and
                           stats['normal_std'] > 1.0)
    return stats


# ---------------------------------------------------------------------------
# Boards and page
# ---------------------------------------------------------------------------
def alpha_crop(image):
    image = image.convert('RGBA')
    box = image.getchannel('A').getbbox()
    if box:
        image = image.crop(box)
    return image


def tile(image, size, background='#d9dedd'):
    image = alpha_crop(image)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new('RGBA', size, background)
    result.alpha_composite(image, ((size[0] - image.width) // 2,
                                   (size[1] - image.height) // 2))
    return result.convert('RGB')


def fit_iou(a, b):
    fa = mask_fit(a)
    fb = mask_fit(b)
    if fa is None or fb is None:
        return None
    union = np.logical_or(fa, fb).sum()
    return float(np.logical_and(fa, fb).sum() / max(1, union))


VIEW_NOTES = {
    'front': ('原照前视是唯一可直接核对完整标签的视图；参考正视的标签排布与之基本一致，'
              '但参考为未标定重绘，喷头和盖形仅作拓扑参考。'),
    'back': ('原照看不到背面；背视参考给出白色说明面板、警示图标和条码的排布，'
             '文字细小且不可辨，候选只保留其大致分区。'),
    'left': ('项目相机 left=-Y，对应参考图标的“右视”物理侧面；'
             '对照行使用互换后的参考面板。该面板同时显示说明面板边缘、火焰警示和部分正面标签，'
             '说明参考各面板之间并不一致；候选按环绕标签近似处理。'),
    'right': ('项目相机 right=+Y，对应参考图标的“左视”物理侧面；'
              '对照行使用互换后的参考面板。该面板给出四种害虫图标和“快速有效 全家安心”侧标；'
              '参考并非严格转台，接缝内容为近似拼接。'),
    'top': ('顶视参考的盖形是简化且偏椭圆的重绘，正圆截面和两个喷孔为一致性推断；'
            '原照的黑色盖体双耳和中央开口是主要形体依据。'),
    'bottom': ('底视参考给出金属凹底、同心拉丝和外卷边；原照底部仅见部分银色卷边，'
               '凹底深度与内部结构为推断。'),
}


def make_report(out, reference, config):
    draw_font = font(22)
    small = font(15)
    board = Image.new('RGB', (2240, 2460), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12),
              '15 杀虫剂罐（Raid 型 600 ml 气雾罐） · 六视图参考（未标定）、原模型与新候选对照',
              font=draw_font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        y = 56 + i * 398
        # The sheet labels side panels from the object's sides; the project
        # cameras use -Y for left and +Y for right, so the displayed reference
        # for the two side rows is swapped.  This is a camera-side convention,
        # not a claim that the uncalibrated sheet is a consistent turntable.
        ref_name = {'left': 'right', 'right': 'left'}.get(name, name)
        ref_panel = reference_can_crop(reference, ref_name)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        ref_mask = reference_mask_panel(reference, ref_name)
        new_mask = np.asarray(new.getchannel('A')) >= 128
        old_mask = np.asarray(old.getchannel('A')) >= 128
        ref_iou = fit_iou(ref_mask, new_mask)
        old_new = fit_iou(old_mask, new_mask)
        rows.append({'view': name,
                     'reference_new_mask_iou': round(ref_iou, 4)
                     if ref_iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4)
                     if old_new is not None else None,
                     'note': VIEW_NOTES[name]})
        draw.text((12, y),
                  f'{name.upper()}   六视图参考（未标定） / 原模型 / 新候选',
                  font=small, fill='#27363c')
        if ref_panel is None:
            ref_panel = Image.new('RGBA', (240, 480), 'white')
        for j, image in enumerate((ref_panel.convert('RGBA'), old, new)):
            width = 420 if j == 0 else 330
            height = 340
            cell = tile(image, (width, height), 'white' if j == 0 else '#d9dedd')
            board.paste(cell, (10 + j * 560, y + 24))
        draw.text((12, y + 370),
                  (f'参考-新轮廓诊断 IoU：{ref_iou:.3f} · 原/新轮廓 IoU：{old_new:.3f}'
                   if ref_iou is not None and old_new is not None
                   else '轮廓不可比较（诊断）'),
                  font=small, fill='#52646b')
    draw.text((20, 2410),
              '诊断说明：六视图未标定且部分面板互相矛盾；IoU 只用于发现明显轮廓/盖形差异，'
              '不是实测精度或验收门槛。原模型虽通过单张源图轮廓门槛，但六方向审计可见表面糊化、'
              '盖形和标签细节丢失，故未复用旧网格。',
              font=small, fill='#8a2b20')
    board.save(out / '六视图-新旧对照.jpg', quality=95)
    return rows


def make_detail_board(out):
    draw_font = font(20)
    small = font(15)
    items = (
        ('detail-cap-front', '盖体前视：双耳、鞍形顶与中央开口'),
        ('detail-cap-top', '盖顶：双耳高差与开口位置'),
        ('detail-window', '双瓣开口内缘（真实切口）'),
        ('detail-nozzle', '双孔喷头与内部连杆'),
        ('detail-cap-seam', '盖沿接缝与褐色污垢'),
        ('detail-bottom', '金属凹底与同心拉丝'),
        ('detail-bottom-rim', '外卷边与锈蚀环'),
        ('detail-label-front', '正面标签：Raid/雷达标识'),
        ('detail-label-back', '背面说明面板与条码'),
        ('detail-label-left', '侧面害虫图标与侧标'),
        ('detail-label-bottom', '标签下沿与卷边交界'),
        ('detail-surface', '标签与盖体的表面反射'),
        ('cap-only', '隔离盖体（含开口）'),
        ('nozzle-only', '隔离阀杆/按钮/扳机'),
        ('nozzle-face', '喷头正面：两个真实通孔'),
        ('bottom-only', '隔离底盖与卷边'),
    )
    board = Image.new('RGB', (2000, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '15 杀虫剂罐 · 关键局部检查（新候选）',
              font=draw_font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = (i % 8) * 250
        y = 55 + (i // 8) * 780
        cell = tile(Image.open(out / 'renders' / f'{name}.png'), (232, 660))
        board.paste(cell, (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, 1600),
              '局部图用于人工核对外观与结构；不能证明壁厚、阀杯内部、盖体卡扣、'
              '标签原始印刷、落稳或现场尺度。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    draw_font = font(20)
    small = font(16)
    board = Image.new('RGB', (1900, 960), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '15 杀虫剂罐 · 隔离结构与开口检查',
              font=draw_font, fill='#18242b')
    items = (('cap-only', '隔离盖体：双耳与双瓣开口'),
             ('cap-top', '盖顶：开口与顶面'),
             ('nozzle-only', '阀杆/按钮/扳机组件'),
             ('nozzle-face', '喷头正面：应见 2 个通孔'),
             ('bottom-only', '金属凹底与外卷边'),
             ('window-look', '从开口看内部组件'))
    for i, (name, label) in enumerate(items):
        x = i * 315
        cell = tile(Image.open(out / 'renders' / f'{name}.png'), (280, 660))
        board.paste(cell, (x + 10, 80))
        draw.text((x + 12, 44), label, font=small, fill='#27363c')
    window_ok = checks.get('window_opening', {}).get('passed')
    nozzle_ok = checks.get('nozzle_opening', {}).get('passed')
    text = (f'开口射线检查：盖体窗口 {"通过" if window_ok else "未通过"} · '
            f'双喷孔 {"通过" if nozzle_ok else "未通过"}')
    draw.text((20, 790), text, font=small,
              fill='#1d5c3a' if (window_ok and nozzle_ok) else '#8a2b20')
    draw.text((20, 835),
              '射线/通孔检查只证明本模型存在真实几何开口，不能证明真实物品的'
              '阀芯结构、孔深、喷雾性能或盖内装配关系。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out, reflection):
    draw_font = font(21)
    small = font(15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '15 杀虫剂罐 · 同光照材质对照（原模型 / 新候选）',
              font=draw_font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选 PBR：标签、黑塑料与金属'))):
        board.paste(tile(Image.open(out / 'renders' / f'{name}.png'), (680, 840)),
                    (20 + i * 720, 70))
        draw.text((24 + i * 720, 38), label, font=small, fill='#27363c')
    draw.text((20, 950),
              '同一中性光与相机；高光由实时照明产生，不把固定灯光烘进底色。'
              '原模型为单图重建的顶点色，不代表旧模型整体通过。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '15 杀虫剂罐 · 变光检查（同视角）',
              font=draw_font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光位置 A'),
                                       ('new-light-alternate', '主光位置 B'))):
        board.paste(tile(Image.open(out / 'renders' / f'{name}.png'), (680, 720)),
                    (20 + i * 720, 65))
        draw.text((24 + i * 720, 36), label, font=small, fill='#27363c')
    draw.text((20, 810),
              '灯光移动后标签清漆和黑塑料高光应随视角改变；本图只检查反射响应，'
              '不证明现场光照方向。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '15 杀虫剂罐 · 金属反射变光对照（底部铝质凹底）',
              font=draw_font, fill='#18242b')
    for i, (name, label) in enumerate((('detail-bottom', '主光位置 A'),
                                       ('bottom-light-alternate', '主光位置 B'))):
        board.paste(tile(Image.open(out / 'renders' / f'{name}.png'), (680, 760)),
                    (20 + i * 720, 55))
        draw.text((24 + i * 720, 28), label, font=small, fill='#27363c')
    draw.text((20, 820),
              (f'金属反射响应：平均绝对亮度差 {reflection.get("mean_abs_luma_difference", 0):.2f}；'
               f'状态 {reflection.get("status", "not_run")}。标签与黑盖为介质材料，'
               '不适用金属反射验收；本图只检查底部金属。'),
              font=small, fill='#52646b')
    board.save(out / '表面反射-变光对照.jpg', quality=96)


def make_turntable(out):
    draw_font = font(18)
    size = 250
    board = Image.new('RGB', (size * 6, size * 4 + 34), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '15 杀虫剂罐 · 24 方向转台（中性光，透明背景合成灰底）',
              font=draw_font, fill='#27363c')
    for i in range(24):
        cell = tile(Image.open(out / 'renders' / f'turn-{i:02d}.png'),
                    (size - 16, size - 16))
        board.paste(cell, ((i % 6) * size + 8, 34 + (i // 6) * size + 8))
    board.save(out / '转台-24方向.jpg', quality=94)


def make_reference_assets(out, reference, config):
    assets = out / 'review-assets'
    assets.mkdir(exist_ok=True)
    paths = {}
    for name in VIEWS:
        image = reference_can_crop(reference, name)
        if image is None:
            continue
        path = assets / f'reference-{name}.png'
        image.save(path)
        paths[name] = path
    return paths


def make_review_page(out, report, page_result):
    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>15 杀虫剂罐 · 独立检查页</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}
body{margin:0;font:15px/1.55 "Microsoft YaHei",system-ui,sans-serif;color:#1c2a31;background:#eef1f2}
header{padding:22px 28px;background:#16242b;color:#f3f6f7}
header h1{margin:0 0 6px;font-size:23px}header p{margin:2px 0;color:#c2d0d5}
.badge{display:inline-block;margin-left:8px;padding:2px 9px;border-radius:10px;background:#8a5a18;color:#fff;font-size:12px}
main{max-width:1500px;margin:0 auto;padding:20px}
section{background:#fff;border-radius:8px;box-shadow:0 1px 4px #0002;margin-bottom:18px;padding:18px}
h2{font-size:18px;margin:0 0 12px;border-left:4px solid #b0472c;padding-left:9px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.grid img{width:100%;height:auto;background:#d9dedd;border-radius:5px}
figure{margin:0}figcaption{font-size:13px;color:#496068;margin-top:4px}
.viewer{display:grid;grid-template-columns:1fr 300px;gap:18px;align-items:start}
#turn{width:100%;max-height:680px;object-fit:contain;background:#d9dedd;border-radius:6px;cursor:grab;touch-action:none}
.side p{margin:4px 0 10px}.status{font-size:13px;color:#0f5b3a}
.warn{border-left:4px solid #b0472c;background:#fbf2ef;padding:10px 12px;border-radius:4px;font-size:13px}
ul{margin:6px 0 0 20px;padding:0}li{margin:3px 0}
a{color:#155e86}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dde4e6;padding:5px 7px;text-align:left}
</style></head><body data-page-ready="false">
<header><div><h1>15 杀虫剂罐（Raid 型 600 ml 气雾罐）<span class="badge">manual_only · 待人工审阅</span></h1>
<p>独立检查页，未接入共享放置服务；所有候选不得自动进入正式合成。</p></div></header>
<main>
<section><h2>旋转检查</h2><div class="viewer">
<div><img id="turn" src="renders/turn-00.png" alt="24 方向转台"></div>
<div class="side"><p><b>操作：</b>拖动图片旋转，或使用按钮。</p>
<p><button id="prev">上一帧</button> <button id="next">下一帧</button> <button id="auto">自动旋转</button></p>
<p id="status" class="status">正在加载转台帧…</p>
<p><a href="mesh.glb">下载 GLB</a> · <a href="model.blend">下载 Blender 文件</a></p>
<p><a href="result.json">模型验证记录</a> · <a href="hashes.json">输入输出哈希</a></p>
</div></div></section>
<section><h2>六视图新旧对照</h2><img src="六视图-新旧对照.jpg" alt="六视图新旧对照">
<p class="warn">参考图未标定，部分面板互相矛盾；对照用于人工发现形体差异，不是实测精度或验收通过。</p></section>
<section><h2>关键局部</h2><img src="局部检查.jpg" alt="关键局部检查"></section>
<section><h2>隔离结构与开口</h2><img src="结构开口-隔离检查.jpg" alt="隔离结构检查"></section>
<section><h2>材质与反射</h2><div class="grid">
<figure><img src="材质-同光照对照.jpg" alt="同光照材质"><figcaption>同光照新旧材质</figcaption></figure>
<figure><img src="材质-变光检查.jpg" alt="变光"><figcaption>两组主光下的高光响应</figcaption></figure>
<figure><img src="表面反射-变光对照.jpg" alt="金属反射"><figcaption>底部金属反射对照</figcaption></figure>
<figure><img src="转台-24方向.jpg" alt="转台"><figcaption>24 方向全览</figcaption></figure>
</div></section>
<section><h2>检查摘要</h2>
<table><tbody>
<tr><th>版本</th><td>spray15-parametric-v1</td></tr>
<tr><th>方法</th><td>按照片比例与原视角可见结构分部件重建；六视图仅作未标定外观参考</td></tr>
<tr><th>网格</th><td>__MESH_SUMMARY__</td></tr>
<tr><th>开口</th><td>__OPENING_SUMMARY__</td></tr>
<tr><th>材质</th><td>__MATERIAL_SUMMARY__</td></tr>
<tr><th>金属反射</th><td>__METAL_SUMMARY__</td></tr>
<tr><th>源图轮廓诊断</th><td>__SOURCE_SUMMARY__</td></tr>
<tr><th>浏览器检查</th><td><a href="browser-check.json">见 browser-check.json</a></td></tr>
</tbody></table></section>
<section><h2>待人工审阅</h2><ul>__PENDING__</ul>
<p class="warn">候选为 manual_only；不替换原模型、不进入自动放置或正式合成。
落稳、现场尺度、环境光融合和真实实物几何均未验收。</p></section>
</main>
<script>
const frames=Array.from({length:24},(_,i)=>`renders/turn-${String(i).padStart(2,'0')}.png`);
const img=document.getElementById('turn'),status=document.getElementById('status');
let index=0,loaded=0,auto=false,timer=null,drag=null;
const pre=[];
frames.forEach((src,i)=>{const im=new Image();im.onload=()=>{loaded++;
 if(loaded===frames.length){status.textContent='转台帧加载完成（24/24）';
  document.body.dataset.pageReady='true';}
};im.onerror=()=>{status.textContent='转台帧加载失败：'+src;};im.src=src;pre.push(im);});
function show(i){index=(i+frames.length)%frames.length;img.src=frames[index];}
document.getElementById('prev').onclick=()=>show(index-1);
document.getElementById('next').onclick=()=>show(index+1);
document.getElementById('auto').onclick=e=>{auto=!auto;e.target.textContent=auto?'停止旋转':'自动旋转';
 clearInterval(timer);if(auto)timer=setInterval(()=>show(index+1),160);};
img.addEventListener('pointerdown',e=>{drag={x:e.clientX,i:index};img.setPointerCapture(e.pointerId);img.style.cursor='grabbing';});
img.addEventListener('pointermove',e=>{if(drag){const step=Math.round((e.clientX-drag.x)/18);show(drag.i+step);}});
img.addEventListener('pointerup',()=>{drag=null;img.style.cursor='grab';});
img.addEventListener('pointercancel',()=>{drag=null;img.style.cursor='grab';});
</script></body></html>"""
    mesh = report['worker']
    html = html.replace('__MESH_SUMMARY__', (
        f"{mesh['parts_count']} 个部件，{mesh['triangles']} 三角面；"
        f"封闭检查 {report['mesh_audit']['closed_after_welding_seams']}/"
        f"{report['mesh_audit']['parts']}，开放部件 "
        f"{len(report['mesh_audit']['open_parts'])} 个。"))
    html = html.replace('__OPENING_SUMMARY__', (
        f"盖体双瓣窗口射线 {'通过' if report['checks']['window_opening'] else '未通过'}；"
        f"喷头双孔射线 {'通过' if report['checks']['nozzle_opening'] else '未通过'}；"
        f"喷头正面通孔计数 {report['opening_checks']['nozzle_face_holes']}。"))
    html = html.replace('__MATERIAL_SUMMARY__', (
        f"标签贴图 {'通过' if report['label_texture_checks']['passed'] else '未通过'}；"
        f"黑塑料金属度 {report['cap_texture_checks']['mean_metallic']}；"
        f"金属贴图金属度 {report['metal_texture_checks']['mean_metallic']}。"))
    html = html.replace('__METAL_SUMMARY__', (
        f"{report['metal_reflection_check']['status']}；"
        f"平均绝对亮度差 {report['metal_reflection_check']['mean_abs_luma_difference']}。"))
    html = html.replace('__SOURCE_SUMMARY__', (
        f"IoU {report['source_fidelity_diagnostic']['iou']}（包围盒对齐诊断，未标定，"
        f"不是验收门槛）。"))
    pending = ''.join(f'<li>{item}</li>' for item in report['pending_review'])
    html = html.replace('__PENDING__', pending)
    path = out / 'review-15.html'
    path.write_text(html, encoding='utf8')
    return path


# ---------------------------------------------------------------------------
# Browser check of the independent page
# ---------------------------------------------------------------------------
def browser_check(page):
    candidates = [
        Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),
        Path('C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'),
        Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'),
        Path('C:/Program Files/Microsoft/Edge/Application/msedge.exe'),
    ]
    executable = next((p for p in candidates if p.is_file()), None)
    if executable is None:
        return {'status': 'not_run', 'reason': 'no local Chrome/Edge executable'}
    profile = Path(tempfile.mkdtemp(prefix='spray15-browser-'))
    screenshot = page.parent / 'review-15-page.png'
    url = page.as_uri()
    command = [str(executable), '--headless=new', '--disable-gpu',
               '--allow-file-access-from-files', '--hide-scrollbars',
               f'--user-data-dir={profile}',
               '--window-size=1440,1400', '--virtual-time-budget=12000',
               f'--screenshot={screenshot}', '--dump-dom', url]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=120, encoding='utf8', errors='replace')
    except Exception as exc:
        return {'status': 'failed', 'reason': str(exc)[:300]}
    dom = result.stdout or ''
    ready = 'data-page-ready="true"' in dom
    return {
        'status': 'passed' if ready and screenshot.is_file() else 'failed',
        'executable': str(executable),
        'exit_code': result.returncode,
        'page_ready_marker': ready,
        'screenshot': str(screenshot),
        'screenshot_exists': screenshot.is_file(),
        'stderr_tail': (result.stderr or '')[-600:],
        'note': ('Headless browser load/screenshot of the independent static page. '
                 'It does not prove interactive 3D or visual acceptance.'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=15)
    p.add_argument('--out', type=Path,
                   default=ROOT / 'outputs/15杀虫剂罐-六视图重建-v1')
    p.add_argument('--library', type=Path, default=LIBRARY)
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    p.add_argument('--draft', action='store_true')
    a = p.parse_args()
    if a.number != 15:
        p.error('this entry only builds item 15')
    a.out = a.out.resolve()
    a.library = a.library.resolve()
    a.blender = a.blender.resolve()
    if not a.blender.is_file():
        raise FileNotFoundError(a.blender)
    row = library_row(15)
    row = dict(row)
    for key in ('mesh', 'rust', 'original', 'source_view'):
        row[key] = str(localize(row[key]))
    for name, expected in (('mesh', 'mesh_sha256'), ('rust', 'rust_sha256'),
                           ('original', 'clean_sha256')):
        if digest(Path(row[name])) != row[expected]:
            raise ValueError(f'{name} hash differs from the model library')
    alpha_path = Path(row['source_view']).parent / 'source-alpha.png'
    if digest(alpha_path) != row['source_alpha_sha256']:
        raise ValueError('source alpha hash differs from the model library')
    reference = find_reference()
    if digest(reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    config_path = HERE / 'spray15_config.json'
    worker_path = HERE / 'mesh_rebuild_15_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))

    version = subprocess.run([str(a.blender), '--version'], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    signature_input = {
        'reference': digest(reference), 'mesh': row['mesh_sha256'],
        'rust': row['rust_sha256'], 'clean': row['clean_sha256'],
        'source_alpha': digest(alpha_path),
        'source_camera': row['reconstruction']['source_camera'],
        'config': digest(config_path), 'builder': digest(Path(__file__)),
        'worker': digest(worker_path),
        'helpers': {name: digest(HERE / name) for name in
                    ('blender_worker.py', 'mesh_checks.py', 'mesh_carve.py',
                     'mesh_rebuild_item.py', 'scene_v2.py')},
        'blender': version, 'python': sys.version.split()[0],
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'pillow': PIL.__version__, 'trimesh': trimesh.__version__,
        'draft': a.draft,
    }
    signature = hashlib.sha256(
        json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if a.out.exists() and any(a.out.iterdir()):
        record = a.out / 'result.json'
        mesh = a.out / 'mesh.glb'
        if record.is_file() and mesh.is_file():
            old = json.loads(record.read_text(encoding='utf8'))
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh):
                print(json.dumps({'status': 'review', 'cache_hit': True,
                                  'out': str(a.out)}, ensure_ascii=False))
                return
        raise FileExistsError('output directory already exists with a different '
                              'signature; use a new version directory')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = {}
    textures.update(make_label_textures(row, reference, a.out, config))
    textures.update(make_cap_textures(a.out, config))
    textures.update(make_metal_rust_textures(a.out, config))
    request = {
        'out': str(a.out), 'original_mesh': row['mesh'], 'textures': textures,
        'config': config, 'source_camera': row['reconstruction']['source_camera'],
        'draft': a.draft,
    }
    (a.out / 'request.json').write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
    with (a.out / 'blender.log').open('w', encoding='utf8') as log:
        process = subprocess.run(
            [str(a.blender), '--background', '--factory-startup',
             '--python-exit-code', '1', '--python', str(worker_path), '--',
             str(a.out / 'request.json')],
            stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if process.returncode:
        raise RuntimeError(f'Blender failed; see {a.out / "blender.log"}')

    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    label_checks = label_texture_checks(textures['label_base'],
                                        textures['label_mr'],
                                        textures['label_normal'])
    cap_stats = texture_stats(textures['cap_base'], textures['cap_mr'], 'cap')
    cap_stats['passed'] = bool(cap_stats['mean_luma'] < 65 and
                               cap_stats['std_luma'] > 1.0 and
                               cap_stats['mean_metallic'] < 0.05 and
                               0.10 <= cap_stats['mean_roughness'] <= 0.45)
    metal_stats = texture_stats(textures['metal_base'], textures['metal_mr'], 'metal')
    metal_stats['passed'] = bool(metal_stats['mean_luma'] > 120 and
                                 metal_stats['mean_metallic'] > 0.85 and
                                 0.15 <= metal_stats['mean_roughness'] <= 0.55)
    rust_stats = texture_stats(textures['rust_base'], textures['rust_mr'], 'rust')
    rust_stats['passed'] = bool(rust_stats['mean_luma'] > 35 and
                                rust_stats['mean_metallic'] < 0.35 and
                                0.55 <= rust_stats['mean_roughness'] <= 0.95)

    with Image.open(alpha_path) as alpha_image:
        source_alpha = np.asarray(alpha_image.convert('L'))
    with Image.open(a.out / 'renders' / 'new-front.png') as new_front:
        new_alpha = np.asarray(new_front.getchannel('A'))
    source_fit = mask_fit(source_alpha >= 128)
    new_fit = mask_fit(new_alpha >= 128)
    if source_fit is not None and new_fit is not None:
        fidelity = source_fidelity(source_fit * 255, new_fit * 255)
        fidelity['alignment'] = ('bounding-box fit of the source alpha and new-front '
                                 'alpha; the source camera is not calibrated for the '
                                 'new geometry')
        fidelity['acceptance_gate'] = False
    else:
        fidelity = {'iou': None, 'holes': None, 'filled_holes': None,
                    'narrow_recall': None, 'passed': False,
                    'alignment': 'mask fit failed'}

    report_rows = make_report(a.out, reference, config)
    source_holes, source_hole_count = interior_holes(source_alpha)
    new_front_holes, new_front_hole_count = interior_holes(new_alpha)
    rays = worker.get('structural_rays', {})
    window_ok = bool(rays.get('window') and
                     all(r['clearance_ok'] for r in rays['window']) and
                     rays.get('window_control_ok'))
    nozzle_face_path = a.out / 'renders' / 'nozzle-face.png'
    nozzle_hole_count = None
    if nozzle_face_path.is_file():
        with Image.open(nozzle_face_path) as im:
            _, nozzle_hole_count = interior_holes(np.asarray(im.getchannel('A')))
    nozzle_ok = bool(rays.get('nozzle_bores') and
                     all(r['through_ok'] for r in rays['nozzle_bores']) and
                     rays.get('nozzle_control_ok') and nozzle_hole_count == 2)
    opening_checks = {
        'window_opening': {
            'passed': window_ok,
            'rays': rays.get('window'),
            'control_solid_rim_hit_m': rays.get('window_control_solid_rim_hit_m'),
            'control_passed': rays.get('window_control_ok'),
            'method': rays.get('method'),
        },
        'nozzle_opening': {
            'passed': nozzle_ok,
            'rays': rays.get('nozzle_bores'),
            'control_solid_hit_m': rays.get('nozzle_control_solid_hit_m'),
            'control_passed': rays.get('nozzle_control_ok'),
            'nozzle_face_holes': nozzle_hole_count,
            'expected_nozzle_face_holes': 2,
        },
        'nozzle_face_holes': nozzle_hole_count,
    }

    reflection = {'status': 'not_run'}
    detail_bottom = a.out / 'renders' / 'detail-bottom.png'
    alternate_bottom = a.out / 'renders' / 'bottom-light-alternate.png'
    if detail_bottom.is_file() and alternate_bottom.is_file():
        first = np.asarray(Image.open(detail_bottom).convert('RGBA')).astype(float)
        second = np.asarray(Image.open(alternate_bottom).convert('RGBA')).astype(float)
        mask = (first[..., 3] >= 128) & (second[..., 3] >= 128)
        difference = float(np.abs(first[..., :3] - second[..., :3])[mask].mean()) \
            if mask.any() else 0.0
        reflection = {
            'status': 'checked_bottom_metal',
            'mean_abs_luma_difference': round(difference, 4),
            'changed': bool(difference > 2.0),
            'note': ('Same bottom camera, two key-light positions, comparing live '
                     'metal highlights. The printed label and black plastic cap are '
                     'dielectric and are not part of this metal check.'),
            'applicability': 'bottom aluminium dish and rolled rim only',
        }
    else:
        reflection = {'status': 'not_run', 'reason': 'bottom comparison renders missing'}

    old_new = {}
    for view in VIEWS:
        old = Image.open(a.out / 'renders' / f'old-{view}.png').convert('RGBA')
        new = Image.open(a.out / 'renders' / f'new-{view}.png').convert('RGBA')
        old_new[view] = fit_iou(np.asarray(old.getchannel('A')) >= 128,
                                np.asarray(new.getchannel('A')) >= 128)

    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 15 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker.get('self_intersecting_components')),
        'window_opening': window_ok,
        'nozzle_opening': nozzle_ok,
        'material_tiles': bool(label_checks['passed'] and cap_stats['passed'] and
                               metal_stats['passed'] and rust_stats['passed']),
        'metal_reflection': bool(reflection.get('changed', False)),
        'source_fidelity': bool(fidelity.get('passed', False)),
        'source_fidelity_is_diagnostic': True,
        'reference_diagnostic_only': True,
        'browser_page': 'not_run',
        'manual_only': True,
        'model_review': 'pending',
        'scene_validation': 'pending',
    }
    audit = audit_row(15)
    report = {
        'version': config['version'], 'number': 15, 'asset_id': row['asset_id'],
        'name': row['file'], 'out': str(a.out), 'status': 'review',
        'model_review': 'pending', 'scene_validation': 'pending',
        'manual_only': True,
        'input_paths': {
            'clean': row['original'], 'rust': row['rust'],
            'original_mesh': row['mesh'],
            'source_alpha': str(alpha_path), 'reference': str(reference),
        },
        'shape_basis': config['shape_basis'],
        'reference_role': config['reference_role'],
        'method': ('parametric_part_rebuild_from_source_photo_with_uncalibrated_'
                   'six_view_label_wrap'),
        'reference': str(reference), 'reference_sha256': digest(reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_clean_sha256': row['clean_sha256'],
        'original_rust_sha256': row['rust_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.972 only measures one silhouette. '
            'The six-view audit shows a distinctly oval top section, soft/blurred '
            'surface texture and lost cap/nozzle detail; the old mesh is therefore '
            'not reused as the candidate base geometry.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'dimensions': config['dimensions'],
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'worker': worker, 'draft': a.draft,
        'mesh_audit': mesh_audit,
        'label_texture_checks': label_checks,
        'cap_texture_checks': cap_stats,
        'metal_texture_checks': metal_stats,
        'rust_texture_checks': rust_stats,
        'opening_checks': opening_checks,
        'source_fidelity_diagnostic': fidelity,
        'source_view_hole_count': source_hole_count,
        'new_front_hole_count': new_front_hole_count,
        'source_diagnostic_note': (
            'The strict source silhouette is compared after bounding-box fit of '
            'the two alpha masks. The source camera was calibrated for the old '
            'mesh, not this rebuild, so the score is a diagnostic only and is never '
            'used as an acceptance gate.'),
        'reference_views': report_rows,
        'old_new_silhouette_iou': old_new,
        'metal_reflection_check': reflection,
        'checks': checks,
        'limitations': [
            'The six-view sheet is an uncalibrated visual match, and its panels are not a consistent turntable; top view is oval, cap shape is simplified and side-panel label ends do not join seamlessly.',
            'All dimensions follow source-photo pixel ratios and a plausible 600 ml aerosol-can size; they are not measured dimensions.',
            'The hidden half of the printed wrap is reconstructed from uncalibrated front/back/side panels; the exact label artwork, print colours and seam position are approximate.',
            'The black overcap horn shape, two-lobed window outline, wall thickness, cap fit and hidden internal valve/actuator/trigger linkage are inferred from the visible front and the simplified top reference.',
            'The two nozzle bores are real model openings but their exact diameter, depth, spray geometry and internal valve are not measured.',
            'The metal base texture, concave depth and rust-ring extent are inferred from the partially visible bottom and the uncalibrated bottom reference.',
            'The printed back information panel and barcode are reproduced only as a low-resolution wrap from the uncalibrated reference; the text is not legible and must not be treated as the real label.',
            'The body is modelled as a closed solid, so the can interior, propellant, valve cup details and liquid volume are not represented.',
            'Parts intentionally overlap at the cap/body, shoulder/body and rim/body; BVH checks only same-part self-intersection, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the original photo, especially the overcap horn shape, two-lobed window outline, nozzle position and cap seam.',
            'Confirm the hidden-half label layout, the back information panel text/barcode and the side-panel seams against a real can or a calibrated photoset.',
            'Confirm the two nozzle bore diameter/depth, the internal valve/actuator/trigger linkage and whether the top reference is the exact same product.',
            'Confirm the bottom concave depth, rolled-rim geometry and rust distribution from additional bottom views or a measured sample.',
            'Confirm the can diameter/height and the 600 ml nominal size relationship; the current scale is photo-derived, not measured.',
            'Verify coordinate, scale, stability, floor contact and light integration in a manual placement session; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side/top/bottom views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 16,
    }
    if not a.draft:
        make_detail_board(a.out)
        make_isolated_board(a.out, opening_checks)
        make_material_boards(a.out, reflection)
        make_turntable(a.out)
        make_reference_assets(a.out, reference, config)
        page = make_review_page(a.out, report, {'status': 'not_run'})
        page_result = browser_check(page)
        report['page_check'] = page_result
        report['checks']['browser_page'] = page_result['status']
    else:
        page = a.out / 'review-15.html'
        page_result = {'status': 'not_run_draft'}
        report['page_check'] = page_result
        report['checks']['browser_page'] = 'not_run_draft'
        report['checks']['window_opening'] = 'not_run_draft'
        report['checks']['nozzle_opening'] = 'not_run_draft'
        report['checks']['metal_reflection'] = 'not_run_draft'
    (a.out / 'result.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (a.out / 'browser-check.json').write_text(
        json.dumps(page_result, ensure_ascii=False, indent=2), encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_15.py',
              'inputs': {k: signature_input[k] for k in
                         ('reference', 'mesh', 'rust', 'clean', 'source_alpha')},
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p)
                        for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf8')
    progress = {
        'number': 15, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'turntable_board': str(a.out / '转台-24方向.jpg'),
        'review_page': str(page),
        'checks': checks, 'opening_checks': opening_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'reference_views': report_rows,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'pending_review': report['pending_review'],
        'next_number': 16, 'manual_only': True,
        'notes': ('Independent rebuild candidate for item 15 only; the old model and '
                  'all other items were not modified. The shared server, index and '
                  'pages were not changed. Candidate is manual_only; visual and '
                  'physical acceptance are pending.'),
    }
    (a.out / 'progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '15-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out),
                      'checks': checks, 'opening': opening_checks,
                      'source_fidelity': fidelity,
                      'page': page_result.get('status')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
