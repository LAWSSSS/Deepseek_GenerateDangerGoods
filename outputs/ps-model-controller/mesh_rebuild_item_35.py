"""Build the review-only fire-extinguisher candidate (item 35).

The original clean/rust photograph shows a red MFZ/ABC3 dry-powder extinguisher
with a printed front label, a gauge, a black squeeze handle, a white flared
nozzle horn and no external hose.  The user six-view sheet is an uncalibrated
visual-match reference of the same family; it is used for back/side topology
only and is never treated as measured geometry.

The candidate is a part-based rebuild, never replaces the library mesh, stays
manual-only and never enters automatic placement.  A new output directory is
required.  All quality numbers are diagnostics, not acceptance gates.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
NUMBER = 35
VERSION = 'extinguisher35-parametric-v1'
REFERENCE_SHA256 = '3a706420374349abea3a8cff8b8960f94d4287d0aab4ccf94346408b49348292'
REFERENCE_NAME = '红色干粉灭火器六视图展示.png'
CLEAN_SHA256 = '5060300385c64c19f169c35b982ef001435a315bf4b5431f24851415b5766f94'
RUST_SHA256 = 'f158cab97099bce0241e23d1a24a4447205c867cb77f3fd94267ced15151b146'
OLD_MESH_SHA256 = '021c8b7a7b9b0961426cb098704550b56b3d85c9752df95f336711bbebee7a28'
OLD_ALPHA_SHA256 = 'c3dee82121aa8d2639137a725ac61faa3bc5d82c9591790928d756740f78b328'
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'

CLEAN = ROOT / 'outputs/物品高清输入-39件/原貌HD/35-灭火器2.png'
RUST = ROOT / 'outputs/物品高清输入-39件/生锈HD/35-灭火器2.png'
OLD_DIR = ROOT / 'outputs/三维模型库-圆润复验/items/35-asset-ee2ef0c2ee73'
OLD_MESH = OLD_DIR / 'mesh.glb'
OLD_ALPHA = OLD_DIR / 'source-alpha.png'
LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE = ROOT / 'references/parallel-six-view' / REFERENCE_NAME

DIMENSIONS = {
    'body_radius': 0.0750,
    'base_radius': 0.0765,
    'bottom_center_z': 0.0045,
    'body_top_z': 0.3456,
    'neck_radius': 0.0245,
    'neck_top_z': 0.3920,
    'collar_bottom_z': 0.3786,
    'collar_top_z': 0.3921,
    'collar_radius': 0.0247,
    'valve_bottom_z': 0.3920,
    'valve_center_z': 0.4140,
    'valve_top_z': 0.4360,
    'valve_height': 0.0440,
    'gauge_center_z': 0.4039,
    'gauge_face_x': 0.0395,
    'pivot_z': 0.4140,
    'nozzle_axis_z': 0.4075,
    'horn_base_outer': 0.0160,
    'horn_tip_outer': 0.0240,
    'horn_base_inner': 0.0100,
    'horn_tip_inner': 0.0170,
    'label_bottom_z': 0.0734,
    'label_top_z': 0.2874,
}

PIXEL_MEASUREMENTS = {
    'source_size': [1024, 1536],
    'body_silhouette_x': [269, 732],
    'body_bottom_y': 1519,
    'body_top_seam_y': 450,
    'label_bounds_px': [315, 630, 710, 1292],
    'gauge_face_center_px': [502.5, 269.5],
    'gauge_face_radius_px': 55.4,
    'collar_y': [306, 348],
    'nozzle_bbox_px': [326, 183, 499, 332],
    'assumed_body_diameter_m': 0.150,
    'pixel_scale_m_per_px': 0.150 / 464.0,
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find_reference():
    if REFERENCE.is_file() and digest(REFERENCE) == REFERENCE_SHA256:
        return REFERENCE
    raise FileNotFoundError('registered six-view reference not found in references/parallel-six-view')


# ---------------------------------------------------------------------------
# Texture synthesis
# ---------------------------------------------------------------------------
def _flat_luminance(rgb, sigma=90.0):
    """Remove the photograph's large-scale illumination gradient, keep detail."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    light = lab[:, :, 0]
    low = cv2.GaussianBlur(light, (0, 0), sigma)
    target = float(np.median(light))
    lab[:, :, 0] = np.clip(light - low + target, 0.0, 100.0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _fbm(size, rng, scales=((96, .55), (24, .30), (5, .15))):
    field = np.zeros((size, size), np.float32)
    for cells, weight in scales:
        small = cv2.resize(rng.normal(size=(cells, cells)).astype(np.float32),
                           (size, size), interpolation=cv2.INTER_CUBIC)
        field += weight * small
    field -= field.mean()
    field /= max(1e-6, field.std())
    return np.clip(field, -3, 3)


def _detail_patch(image, rect):
    patch = np.asarray(image.crop(rect).convert('RGBA'))
    if float((patch[:, :, 3] > 240).mean()) < .95:
        raise ValueError(f'material sample rect contains transparency: {rect}')
    gray = cv2.cvtColor(patch[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 10)
    detail -= detail.mean()
    return detail


def _resize_detail(detail, size):
    return cv2.resize(detail, (size, size), interpolation=cv2.INTER_CUBIC)


def _tile_specs(size, rng):
    return {
        'red': dict(base=(191, 18, 24), rough=.25, patch=(400, 500, 600, 620),
                    fleck=.0035, fleck_rgb=(96, 48, 25)),
        'black': dict(base=(23, 24, 27), rough=.20, patch=(720, 260, 740, 280),
                      fleck=.0015, fleck_rgb=(122, 86, 58)),
        'white': dict(base=(232, 230, 224), rough=.36, patch=(360, 250, 410, 300),
                      fleck=.0010, fleck_rgb=(120, 116, 108)),
        'collar': dict(base=(152, 174, 38), rough=.37, patch=(440, 320, 560, 342),
                       fleck=.0015, fleck_rgb=(92, 92, 48)),
    }


def make_atlas(clean, rust, out):
    clean_im = Image.open(clean).convert('RGBA')
    rust_im = Image.open(rust).convert('RGBA')
    rng = np.random.default_rng(350035)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = _tile_specs(size, rng)
    order = ('red', 'black', 'white', 'collar')
    tile_metrics = []
    for i, key in enumerate(order):
        spec = specs[key]
        color = np.array(spec['base'], np.float32)
        detail = _resize_detail(_detail_patch(clean_im, spec['patch']), size)
        coarse = _fbm(size, rng, ((80, .6), (20, .3), (4, .1)))
        fine = cv2.resize(rng.normal(size=(size, size)).astype(np.float32),
                          (size, size), interpolation=cv2.INTER_LINEAR)
        field = .55 * coarse + .25 * fine
        rgb = color[None, None, :] * (1 + field[:, :, None] * .025)
        rgb += detail[:, :, None] * (.55 if key == 'red' else .42)
        # Sparse wear flecks: softened blobs, never a uniform dark wash.
        speck = cv2.resize(rng.random((size, size), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_CUBIC)
        speck = cv2.GaussianBlur(speck, (0, 0), 1.1)
        threshold = float(np.quantile(speck, 1 - spec['fleck']))
        flecks = np.clip((speck - threshold) / max(1e-6, 1 - threshold), 0, 1)
        flecks = cv2.GaussianBlur(flecks.astype(np.float32), (0, 0), 1.8)
        fleck_rgb = np.array(spec['fleck_rgb'], np.float32)
        rgb = rgb * (1 - flecks[:, :, None] * .72) + fleck_rgb[None, None, :] * (flecks[:, :, None] * .72)
        wear_mask = np.zeros((size, size), np.float32)
        if key == 'red':
            # Localised rust/grime from the supplied weathered photograph; the
            # strongest brown pixels only, so the enamel never becomes a dark wash.
            rust_rect = (560, 430, 700, 530)
            rust_detail = _resize_detail(_detail_patch(rust_im, rust_rect), size)
            rgb += rust_detail[:, :, None] * .30
            rc = np.asarray(rust_im.crop(rust_rect).convert('RGB')).astype(np.float32)
            score = (rc[:, :, 0] - rc[:, :, 1]) + (rc[:, :, 1] - rc[:, :, 2]) * .5
            thr = float(np.quantile(score, .955))
            mask = np.clip((score - thr) / max(1e-6, float(score.max()) - thr), 0, 1)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)
            mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.2)
            rust_rgb = np.array((108, 57, 27), np.float32)
            rgb = rgb * (1 - mask[:, :, None] * .55) + \
                rust_rgb[None, None, :] * (mask[:, :, None] * .55)
            wear_mask = mask
            # A few fine scuffs, kept light so the enamel does not turn black.
            scratch = np.zeros((size, size), np.float32)
            for _ in range(26):
                x0, y0 = rng.integers(0, size, 2)
                x1 = int(x0 + rng.normal(0, 70))
                y1 = int(y0 + rng.normal(0, 18))
                cv2.line(scratch, (int(x0), int(y0)), (int(x1), int(y1)), 1.0,
                         int(rng.integers(1, 3)), cv2.LINE_AA)
            scratch = cv2.GaussianBlur(scratch, (0, 0), .8)
            rgb = rgb * (1 - scratch[:, :, None] * .18) + \
                np.array((238, 148, 132), np.float32)[None, None, :] * (scratch[:, :, None] * .18)
        rgb = np.clip(rgb, 0, 255)
        roughness = np.clip(spec['rough'] + flecks * .24 + field * .025
                            + detail / 25.0 * .02 + wear_mask * .30, .10, .70)
        metallic = np.zeros((size, size), np.float32)
        bump = detail * .35 + flecks * 18.0 + field * .6
        bump = cv2.GaussianBlur(bump, (0, 0), .8)
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 14.0
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 14.0
        nx = np.clip(-dx * .35, -.05, .05)
        ny = np.clip(dy * .35, -.05, .05)
        nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = rgb.astype(np.uint8)
        mr[sl] = np.stack((np.full((size, size), 255, np.uint8),
                           (roughness * 255).astype(np.uint8),
                           (metallic * 255).astype(np.uint8)), axis=2)
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                             0, 255).astype(np.uint8)
        tile_metrics.append({
            'tile': i, 'key': key,
            'mean_luma': round(float(rgb.mean()), 2),
            'mean_chroma': round(float((rgb.max(axis=2) - rgb.min(axis=2)).mean()), 2),
            'mean_metallic': round(float(metallic.mean()), 4),
            'mean_roughness': round(float(roughness.mean()), 4),
            'normal_std': round(float(normal[sl].std()), 3),
        })
    paths = {}
    for name, arr in (('base_color', base), ('metallic_roughness', mr), ('normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    thresholds = {
        'red': dict(luma=(50, 120), chroma=(55, 255), rough=(.15, .45)),
        'black': dict(luma=(0, 45), chroma=(0, 40), rough=(.10, .40)),
        'white': dict(luma=(175, 245), chroma=(0, 30), rough=(.25, .55)),
        'collar': dict(luma=(90, 170), chroma=(40, 255), rough=(.25, .55)),
    }
    for item in tile_metrics:
        spec = thresholds[item['key']]
        item['passed'] = bool(spec['luma'][0] <= item['mean_luma'] <= spec['luma'][1]
                              and spec['chroma'][0] <= item['mean_chroma'] <= spec['chroma'][1]
                              and spec['rough'][0] <= item['mean_roughness'] <= spec['rough'][1]
                              and item['mean_metallic'] <= .02
                              and item['normal_std'] >= 1.0)
    return paths, {'passed': all(t['passed'] for t in tile_metrics),
                   'tiles': tile_metrics,
                   'note': ('Red/black enamel and white/olive plastics stay dielectric; '
                            'wear changes colour and roughness, not metalness.')}


def make_label(clean, out):
    im = np.asarray(Image.open(clean).convert('RGB'))
    x0, x1, y0, y1 = 314, 711, 629, 1294
    cx, radius = 500.5, 232.0
    th0 = math.asin((x0 - cx) / radius)
    th1 = math.asin((x1 - cx) / radius)
    width, height = 1536, 2048
    angles = np.linspace(th0, th1, width)
    xs = cx + radius * np.sin(angles)
    ys = np.linspace(y0, y1, height)
    mx, my = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    label = cv2.remap(im, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    label = _flat_luminance(label, 120.0)
    # Faint paper grain, no baked lighting.
    rng = np.random.default_rng(350036)
    grain = rng.normal(0, 2.2, (height, width, 1)).astype(np.float32)
    label = np.clip(label.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    Image.fromarray(label).save(out / 'label_base.png')
    rough = np.full((height, width), .38, np.float32)
    mr = np.stack((np.full_like(rough, 255), (rough * 255).astype(np.uint8),
                   np.zeros_like(rough, np.uint8)), axis=2).astype(np.uint8)
    Image.fromarray(mr).save(out / 'label_mr.png')
    fine = cv2.resize(rng.normal(size=(height // 2, width // 2)).astype(np.float32),
                      (width, height), interpolation=cv2.INTER_LINEAR)
    bump = cv2.GaussianBlur(fine, (0, 0), .8)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 18
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 18
    nx = np.clip(-dx * .25, -.025, .025)
    ny = np.clip(dy * .25, -.025, .025)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(normal).save(out / 'label_normal.png')
    gray = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = (label.astype(np.float32).max(axis=2) - label.astype(np.float32).min(axis=2))
    red = float(((label[:, :, 0] > 110)
                 & (label[:, :, 1] < label[:, :, 0] * .62)
                 & (label[:, :, 2] < label[:, :, 0] * .62)).mean())
    white = float((gray > 190).mean())
    check = {'passed': bool(55 <= float(gray.mean()) <= 205 and float(gray.std()) >= 18
                            and red >= .25 and white >= .03),
             'mean_luma': round(float(gray.mean()), 2),
             'luma_std': round(float(gray.std()), 2),
             'red_fraction': round(red, 4),
             'white_fraction': round(white, 4),
             'source_note': ('Rectified from the front-label rectangle with inverse '
                             'cylindrical sampling and large-scale lighting removed; '
                             'the degree of wrap and exact label arc are inferred.'),
             'rectified_from_px': [x0, y0, x1, y1],
             'angle_range_deg': [round(math.degrees(th0), 2), round(math.degrees(th1), 2)]}
    return {'label_base': str(out / 'label_base.png'),
            'label_mr': str(out / 'label_mr.png'),
            'label_normal': str(out / 'label_normal.png')}, check


def make_dial(clean, out):
    im = np.asarray(Image.open(clean).convert('RGB'))
    cx, cy, radius = 502.5, 269.5, 55.4
    size = 1024
    u = np.linspace(-1, 1, size, dtype=np.float32)
    v = np.linspace(1, -1, size, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    mx = (cx + uu * radius).astype(np.float32)
    my = (cy - vv * radius).astype(np.float32)
    dial = cv2.remap(im, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Inpaint the photographed needle and central screw; a 3D needle and screw
    # are modelled instead.  The dial markings are kept photographically.
    mask = np.zeros((size, size), np.uint8)
    rr = np.sqrt(uu * uu + vv * vv)
    angle = math.radians(-53)
    du = np.array([math.cos(angle), math.sin(angle)], np.float32)
    dv = np.array([-math.sin(angle), math.cos(angle)], np.float32)
    along = uu * du[0] + vv * du[1]
    across = uu * dv[0] + vv * dv[1]
    needle = ((along > .015) & (along < 1.05) & (np.abs(across) < .115)
              & (rr < 1.0))
    mask[needle] = 255
    mask[rr < .30] = 255
    dial = cv2.inpaint(dial, mask, 9, cv2.INPAINT_TELEA)
    dial = _flat_luminance(dial, 120.0)
    blur = cv2.GaussianBlur(dial, (0, 0), 1.6)
    dial = np.clip(dial.astype(np.float32) * 1.18 - blur.astype(np.float32) * .18,
                   0, 255).astype(np.uint8)
    outside = rr > .97
    dial[outside] = (238, 238, 236)
    Image.fromarray(dial).save(out / 'dial_base.png')
    rough = np.full((size, size), .30, np.float32)
    mr = np.stack((np.full_like(rough, 255), (rough * 255).astype(np.uint8),
                   np.zeros_like(rough, np.uint8)), axis=2).astype(np.uint8)
    Image.fromarray(mr).save(out / 'dial_mr.png')
    bump = cv2.GaussianBlur(cv2.resize(
        np.random.default_rng(350037).normal(size=(128, 128)).astype(np.float32),
        (size, size), interpolation=cv2.INTER_CUBIC), (0, 0), .7)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 22
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 22
    nx = np.clip(-dx * .2, -.02, .02)
    ny = np.clip(dy * .2, -.02, .02)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(normal).save(out / 'dial_normal.png')
    hsv = cv2.cvtColor(dial, cv2.COLOR_RGB2HSV)
    h, s, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    disc = rr <= .95
    yellow = float(((h >= 15) & (h <= 40) & (s > 90) & (val > 90) & disc).sum() / disc.sum())
    green = float(((h >= 40) & (h <= 85) & (s > 90) & (val > 60) & disc).sum() / disc.sum())
    red = float(((((h <= 8) | (h >= 170)) & (s > 100) & (val > 90) & disc)).sum() / disc.sum())
    white = float(((s < 45) & (val > 150) & disc).sum() / disc.sum())
    dark = float(((val < 90) & disc).sum() / disc.sum())
    check = {'passed': bool(yellow > .003 and green > .003 and red > .003
                            and white > .10 and dark > .005),
             'yellow_fraction': round(yellow, 4), 'green_fraction': round(green, 4),
             'red_fraction': round(red, 4), 'white_fraction': round(white, 4),
             'dark_fraction': round(dark, 4),
             'source_note': ('Dial markings are rectified from the photograph; the '
                             'photographed needle/screw were inpainted and replaced by '
                             '3D parts. The exact scale values are not calibrated.')}
    return {'dial_base': str(out / 'dial_base.png'),
            'dial_mr': str(out / 'dial_mr.png'),
            'dial_normal': str(out / 'dial_normal.png')}, check


# ---------------------------------------------------------------------------
# Mesh utilities
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


def mesh_topology(path):
    scene = trimesh.load(path, force='scene')
    open_parts, nonfinite = [], []
    for node in scene.graph.nodes_geometry:
        _, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not np.isfinite(mesh.vertices).all():
            nonfinite.append(node)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        if not mesh.is_watertight:
            open_parts.append(node)
    return {'parts': len(scene.graph.nodes_geometry),
            'closed_after_welding_seams': len(scene.graph.nodes_geometry) - len(open_parts),
            'open_parts': open_parts, 'nonfinite_parts': nonfinite,
            'bounds': scene.bounds.tolist(),
            'note': 'Assembly parts intentionally interpenetrate; topology cannot prove fit or settling.'}


def _ray_hits(mesh, origin, direction):
    """Vectorised Moller-Trumbore ray/triangle intersections (no rtree needed)."""
    tris = np.asarray(mesh.triangles, dtype=float)
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    pvec = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, pvec)
    valid = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[valid] = 1.0 / det[valid]
    tvec = origin - tris[:, 0]
    u = np.einsum('ij,ij->i', tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum('j,ij->i', direction, qvec) * inv
    t = np.einsum('ij,ij->i', e2, qvec) * inv
    mask = valid & (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & \
        (u + v <= 1 + 1e-9) & (t > 1e-7)
    return t[mask]


def _ray_clear(mesh, origin, direction, length):
    hits = _ray_hits(mesh, origin, direction)
    within = int(((hits >= 0) & (hits <= length)).sum())
    return within == 0, within


def opening_checks(path, dimensions):
    parts = load_parts(path)
    checks = {}
    # Nozzle horn through-bore: cast along the horn axis from outside the tip.
    horn = parts['35 nozzle horn']
    base = np.array([0.0, -0.0180, dimensions['nozzle_axis_z'] - 0.0020])
    tip = np.array([0.0, -0.0600, dimensions['nozzle_axis_z'] + 0.0050])
    axis = tip - base
    axis /= np.linalg.norm(axis)
    origin = tip + axis * 0.02
    clear, hits = _ray_clear(horn, origin, -axis, float(np.linalg.norm(tip - base)) * 1.02 + 0.02)
    checks['nozzle_bore_ray'] = {
        'passed': bool(clear), 'hits_within_bore_length': hits,
        'method': ('Ray along the horn axis from outside the tip to just past the '
                   'base opening; the bore is open and must not hit the horn shell.')}
    # Safety pin ring: centre ray must pass, offset ray must hit metal.
    pin = parts['35 safety pin']
    c = np.array([0.005, 0.050, 0.4245])
    clear_center, _ = _ray_clear(pin, c - np.array([0.02, 0, 0]),
                                 np.array([1.0, 0, 0]), 0.04)
    hit_offset, n_offset = _ray_clear(pin, c - np.array([0.02, 0, 0]) + np.array([0, 0, 0.0046]),
                                      np.array([1.0, 0, 0]), 0.04)
    checks['safety_pin_ring'] = {
        'passed': bool(clear_center and not hit_offset),
        'centre_ray_clear': bool(clear_center),
        'offset_ray_hits': int(n_offset),
        'method': 'Ray through the ring centre must miss; a ray through the torus body must hit.'}
    # Concave bottom: centre is higher than the outer rim.
    body = parts['35 body']
    tc = _ray_hits(body, (0.0, 0.0, -0.05), (0.0, 0.0, 1.0))
    tr = _ray_hits(body, (0.070, 0.0, -0.05), (0.0, 0.0, 1.0))
    zc = float(np.min(tc) - 0.05) if len(tc) else None
    zr = float(np.min(tr) - 0.05) if len(tr) else None
    checks['concave_bottom'] = {
        'passed': bool(zc is not None and zr is not None and zc > zr + 0.0015),
        'centre_hit_z': None if zc is None else round(zc, 5),
        'rim_hit_z': None if zr is None else round(zr, 5),
        'method': 'Vertical up-rays at the bottom centre and near the outer rim.'}
    checks['passed'] = bool(all(v['passed'] for k, v in checks.items() if isinstance(v, dict)))
    return checks


def silhouette_holes(path):
    with Image.open(path) as image:
        alpha = np.asarray(image.getchannel('A'))
    solid = (alpha >= 128).astype(np.uint8)
    contours, hierarchy = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] < 0 or cv2.contourArea(contour) < 16:
                continue
            region = np.zeros(solid.shape, np.uint8)
            cv2.drawContours(region, contours, i, 1, -1)
            interior = (region > 0) & (solid == 0)
            if interior.sum() >= 16:
                holes += 1
    return holes


def metal_highlight_shift(path_a, path_b):
    def centroid(path):
        with Image.open(path) as image:
            arr = np.asarray(image.convert('RGBA')).astype(np.float32)
        alpha = arr[:, :, 3] > 32
        luma = arr[:, :, :3].mean(axis=2)
        if not alpha.any():
            return None, 0.0
        values = luma[alpha]
        threshold = float(np.quantile(values, .90))
        mask = alpha & (luma >= threshold)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None, 0.0
        return (float(xs.mean()), float(ys.mean())), float(values.max())
    ca, ma = centroid(path_a)
    cb, mb = centroid(path_b)
    if ca is None or cb is None:
        return {'passed': False, 'centroid_a': None, 'centroid_b': None,
                'shift_px': None, 'note': 'A chrome-only render had no visible highlight.'}
    shift = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    return {'passed': bool(shift >= 2.0), 'centroid_a': [round(ca[0], 2), round(ca[1], 2)],
            'centroid_b': [round(cb[0], 2), round(cb[1], 2)],
            'shift_px': round(shift, 2),
            'peak_luma_a': round(ma, 1), 'peak_luma_b': round(mb, 1),
            'note': ('Chrome-only render under two key-light directions; this proves '
                     'the highlights move, not that the environment matches the site.')}


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def _load_rgba(path):
    return Image.open(path).convert('RGBA')


def _contain(image, size, background=(236, 239, 237)):
    image = image.copy()
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new('RGB', size, background)
    mask = image if image.mode in ('RGBA', 'LA') else None
    canvas.paste(image, ((size[0] - image.width) // 2,
                         (size[1] - image.height) // 2), mask)
    return canvas


def _reference_panel(reference, index):
    image = Image.open(reference).convert('RGB')
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 8, y0 + 6, x0 + w // 3 - 8, y0 + h // 2 - 8))


def _mask_fit(mask, size=400):
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    factor = (size - 20) / max(w, h)
    resized = cv2.resize(mask[y:y + h, x:x + w].astype(np.uint8),
                         (max(1, round(w * factor)), max(1, round(h * factor))),
                         interpolation=cv2.INTER_NEAREST)
    result = np.zeros((size, size), np.uint8)
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    result[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return result


def _iou(a, b):
    if a is None or b is None:
        return None
    union = np.logical_or(a, b).sum()
    return None if union == 0 else float(np.logical_and(a, b).sum() / union)


def _reference_mask(panel):
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    mask = (rgb.min(axis=2) < 226).astype(np.uint8)
    mask[int(mask.shape[0] * .92):] = 0
    mask[:5] = 0; mask[-5:] = 0; mask[:, :5] = 0; mask[:, -5:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(200, areas.max() * .02)) + 1
    return np.isin(labels, keep).astype(np.uint8)


VIEW_NOTES = {
    'front': '原照正视显示标签、压力表、手柄和白色喇叭口；候选按此朝向 +X。',
    'back': '原照背面不可见；参考背视只用于确认无外置软管和背面轮廓。',
    'left': '参考左视与喇叭口同侧；候选把喇叭口沿 -Y 放置，使正视左侧出现。',
    'right': '参考右视显示手柄侧；候选手柄沿 +Y 伸展。',
    'top': '顶视参考显示表盘、手柄、保险销和喇叭口俯视关系；阀体内部为推断。',
    'bottom': '底视参考显示平底与底圈；候选保留凹底和底部卷边，凹深为推断。',
}


def make_six_view_board(out, reference):
    font = ImageFont.truetype(LABEL_FONT, 22)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2400, 2760), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2（MFZ/ABC3 干粉）· 六视图参考（未标定）/ 原模型 / 新候选',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        y = 56 + i * 445
        panel = _reference_panel(reference, i)
        old = _load_rgba(out / 'renders' / f'old-{name}.png')
        new = _load_rgba(out / 'renders' / f'new-{name}.png')
        ref_fit = _mask_fit(_reference_mask(panel))
        old_fit = _mask_fit(np.asarray(old.getchannel('A')) >= 128)
        new_fit = _mask_fit(np.asarray(new.getchannel('A')) >= 128)
        ref_iou = _iou(ref_fit, new_fit)
        old_new = _iou(old_fit, new_fit)
        rows.append({'view': name,
                     'reference_new_mask_iou': None if ref_iou is None else round(ref_iou, 4),
                     'old_new_silhouette_iou': None if old_new is None else round(old_new, 4),
                     'note': '按最长边等比对齐，仅用于发现形体差异，不是标定精度。' + VIEW_NOTES[name]})
        draw.text((12, y), f'{name.upper()}   参考 / 原模型 / 新候选',
                  font=small, fill='#27363c')
        board.paste(_contain(panel, (580, 380)), (10, y + 24))
        board.paste(_contain(old, (580, 380)), (610, y + 24))
        board.paste(_contain(new, (580, 380)), (1210, y + 24))
        draw.multiline_text((1810, y + 40),
                            f'参考-新 IoU\n{ref_iou if ref_iou is None else round(ref_iou, 3)}',
                            font=small, fill='#27363c', spacing=6)
        draw.multiline_text((1810, y + 150),
                            f'旧-新轮廓 IoU\n{old_new if old_new is None else round(old_new, 3)}',
                            font=small, fill='#27363c', spacing=6)
        draw.multiline_text((1810, y + 260), VIEW_NOTES[name], font=small,
                            fill='#4c5a60', spacing=4)
        if i < len(VIEWS) - 1:
            draw.line((0, y + 430, 2400, y + 430), fill='#d3d9d6', width=2)
    board.save(out / '六视图-新旧对照.jpg', quality=90)
    return rows


def make_detail_board(out):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2400, 1560), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · 关键局部（新候选渲染）', font=font, fill='#18242b')
    items = (
        ('detail-gauge.png', '压力表：铬圈、表盘、玻璃、3D 指针'),
        ('detail-handle.png', '手柄：上下压杆、支架、销轴、保险销'),
        ('detail-nozzle.png', '白色喇叭口：外扩锥面与真实通孔'),
        ('detail-label.png', '标签：照片矫正纹理与曲面贴合'),
        ('detail-top.png', '顶部总成俯视：表盘/手柄/喇叭口关系'),
        ('detail-seam.png', '上部卷边与筒体过渡'),
        ('detail-bottom.png', '凹底、底圈与下部卷边'),
        ('detail-overall-3q.png', '整体 3/4 视角'),
    )
    for i, (name, caption) in enumerate(items):
        x = (i % 4) * 600
        y = 60 + (i // 4) * 740
        image = _load_rgba(out / 'renders' / name)
        board.paste(_contain(image, (580, 640)), (x + 10, y))
        draw.text((x + 14, y + 648), caption, font=small, fill='#27363c')
    board.save(out / '局部检查.jpg', quality=90)


def make_isolated_board(out, checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2400, 1320), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · 分部件隔离与开孔检查', font=font, fill='#18242b')
    items = (
        ('body-only.png', '筒体/卷边/底圈'),
        ('label-only.png', '标签壳'),
        ('valve-only.png', '阀体/颈部'),
        ('gauge-only.png', '表盘总成'),
        ('handle-only.png', '手柄/支架/保险销'),
        ('nozzle-only.png', '喇叭口侧视'),
        ('nozzle-through.png', '喇叭口轴向（应见 1 个孔）'),
        ('safety-ring.png', '保险销环（应见 1 个孔）'),
    )
    for i, (name, caption) in enumerate(items):
        x = (i % 4) * 600
        y = 60 + (i // 4) * 620
        image = _load_rgba(out / 'renders' / name)
        board.paste(_contain(image, (580, 530)), (x + 10, y))
        draw.text((x + 14, y + 538), caption, font=small, fill='#27363c')
    draw.multiline_text(
        (20, 1270),
        (f"几何射线：喇叭口通孔 {checks['nozzle_bore_ray']['passed']}；"
         f"保险销环 {checks['safety_pin_ring']['passed']}；"
         f"凹底 {checks['concave_bottom']['passed']}。"
         ' 隔离图只证明可见开口，不证明孔深、壁厚或装配关系。'),
        font=small, fill='#4c5a60')
    board.save(out / '结构开口-隔离检查.jpg', quality=90)


def make_material_boards(out):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2200, 1220), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · 同光照材质 / 灰模对照', font=font, fill='#18242b')
    board.paste(_contain(_load_rgba(out / 'renders' / 'new-material-front.png'), (1060, 1080)),
                (20, 60))
    board.paste(_contain(_load_rgba(out / 'renders' / 'new-clay-front.png'), (1060, 1080)),
                (1110, 60))
    draw.text((30, 1150), 'PBR 材质（红漆/黑漆/白塑/橄榄绿/铬/玻璃）', font=small, fill='#27363c')
    draw.text((1120, 1150), '灰模（同一相机与灯光）', font=small, fill='#27363c')
    board.save(out / '材质-同光照对照.jpg', quality=90)

    board = Image.new('RGB', (2200, 1220), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · 主光方向改变后的漆面与金属响应', font=font, fill='#18242b')
    board.paste(_contain(_load_rgba(out / 'renders' / 'new-material-front.png'), (1060, 1080)),
                (20, 60))
    board.paste(_contain(_load_rgba(out / 'renders' / 'new-material-alternate.png'), (1060, 1080)),
                (1110, 60))
    draw.text((30, 1150), '主光 A（右前上）', font=small, fill='#27363c')
    draw.text((1120, 1150), '主光 B（左后上）；高光位置随光位变化', font=small, fill='#27363c')
    board.save(out / '材质-变光检查.jpg', quality=90)

    board = Image.new('RGB', (1500, 800), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · 仅铬件在两种光位下的高光（金属反射证据）',
              font=font, fill='#18242b')
    board.paste(_contain(_load_rgba(out / 'renders' / 'metal-light-a.png'), (700, 700)),
                (20, 60))
    board.paste(_contain(_load_rgba(out / 'renders' / 'metal-light-b.png'), (700, 700)),
                (760, 60))
    draw.text((30, 760), '光位 A：铬圈/销轴/螺钉', font=small, fill='#27363c')
    draw.text((770, 760), '光位 B：同一铬件的高光移动', font=small, fill='#27363c')
    board.save(out / '表面反射-变光对照.jpg', quality=90)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    cols, rows = 6, 4
    cell = 360
    board = Image.new('RGB', (cols * cell, rows * cell + 40), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 10), '35 灭火器2 · 24 方向转台（外观检查，不证明隐藏面）',
              font=font, fill='#18242b')
    for i in range(24):
        x = (i % cols) * cell
        y = 40 + (i // cols) * cell
        image = _load_rgba(out / 'renders' / f'turn-{i:02d}.png')
        board.paste(_contain(image, (cell - 10, cell - 36)), (x + 5, y + 2))
        draw.text((x + 10, y + cell - 30), f'{i * 15:03d}°', font=font, fill='#3a484e')
    board.save(out / '转台-24方向.jpg', quality=90)


def make_texture_boards(out):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2200, 1220), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · PBR 图集布局（2×2：红漆 / 黑漆 / 白塑 / 橄榄绿）',
              font=font, fill='#18242b')
    for name, caption, xy in (('base_color.png', '底色', (20, 60)),
                              ('metallic_roughness.png', '金属度(R=恒1)/粗糙度(G)/金属度(B)',
                               (760, 60)),
                              ('normal.png', '法线', (1500, 60))):
        image = Image.open(out / name).convert('RGB').resize((680, 680), Image.LANCZOS)
        board.paste(image, xy)
        draw.text((xy[0], xy[1] + 692), caption, font=small, fill='#27363c')
    board.save(out / '材质图集-布局.jpg', quality=90)

    board = Image.new('RGB', (2200, 1250), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · 矫正后的标签与表盘纹理', font=font, fill='#18242b')
    label = Image.open(out / 'label_base.png').convert('RGB')
    label.thumbnail((760, 1120), Image.LANCZOS)
    board.paste(label, (40, 70))
    dial = Image.open(out / 'dial_base.png').convert('RGB').resize((900, 900), Image.LANCZOS)
    board.paste(dial, (880, 70))
    draw.text((40, 1180), '标签：逆圆柱采样 + 去除大尺度光照', font=small, fill='#27363c')
    draw.text((880, 1000), '表盘：照片矫正；指针/中心螺钉已修补并由 3D 件替代',
              font=small, fill='#27363c')
    board.save(out / '展签-表盘-贴图.jpg', quality=90)


def make_source_board(out, result):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2200, 900), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '35 灭火器2 · 原照 / 原模型原视角 / 新候选原视角 / 六视图参考',
              font=font, fill='#18242b')
    clean = Image.open(CLEAN).convert('RGBA')
    old = Image.open(out / 'renders' / 'old-front.png').convert('RGBA')
    new = Image.open(out / 'renders' / 'new-source.png').convert('RGBA')
    ref = Image.open(REFERENCE).convert('RGB')
    board.paste(_contain(clean, (520, 760)), (10, 70))
    board.paste(_contain(old, (520, 760)), (540, 70))
    board.paste(_contain(new, (520, 760)), (1070, 70))
    board.paste(_contain(ref, (520, 760)), (1600, 70))
    draw.text((14, 835), '原照（清晰）', font=small, fill='#27363c')
    draw.text((544, 835), '原模型（TripoSR）', font=small, fill='#27363c')
    draw.text((1074, 835),
              f"新候选原相机 IoU {result['source_fidelity_diagnostic']['iou']:.3f}（仅诊断）",
              font=small, fill='#27363c')
    draw.text((1604, 835), '六视图参考（未标定）', font=small, fill='#27363c')
    board.save(out / '原照-新候选对照.jpg', quality=90)


# ---------------------------------------------------------------------------
# Review page
# ---------------------------------------------------------------------------
def make_review_page(out, result, tables):
    from review35_page import write_page
    write_page(out, result, tables)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=35)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views only')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != NUMBER:
        p.error('this entry supports item 35 only')
    if a.out is None:
        a.out = ROOT / ('outputs/35-extinguisher-draft1' if a.draft
                        else 'outputs/35灭火器2-六视图重建-v1')
    a.out, a.blender = a.out.resolve(), a.blender.resolve()
    reference = find_reference()
    for path, expected, label in ((CLEAN, CLEAN_SHA256, 'clean photo'),
                                  (RUST, RUST_SHA256, 'rust photo'),
                                  (OLD_MESH, OLD_MESH_SHA256, 'library mesh'),
                                  (OLD_ALPHA, OLD_ALPHA_SHA256, 'source alpha')):
        if digest(path) != expected:
            raise ValueError(f'{label} hash differs from the registered value')
    if digest(reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = next(r for r in (json.loads(line) for line in
                           LIBRARY.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == NUMBER)
    audit = next(r for r in (json.loads(line) for line in
                             AUDIT.read_text(encoding='utf8').splitlines() if line.strip())
                 if r['number'] == NUMBER)
    with Image.open(OLD_ALPHA) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    source_alpha_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    worker_path = HERE / 'mesh_rebuild_35_worker.py'
    review_path = HERE / 'review35_page.py'
    version = subprocess.run([str(a.blender), '--version'], capture_output=True, text=True,
                             check=True).stdout.splitlines()[0]
    signature_input = {'reference': digest(reference), 'clean': CLEAN_SHA256,
                       'rust': RUST_SHA256, 'mesh': OLD_MESH_SHA256,
                       'source_alpha': OLD_ALPHA_SHA256,
                       'source_camera': row['reconstruction']['source_camera'],
                       'dimensions': DIMENSIONS, 'builder': digest(Path(__file__)),
                       'worker': digest(worker_path), 'review': digest(review_path),
                       'helpers': {name: digest(HERE / name) for name in
                                   ('blender_worker.py', 'mesh_checks.py',
                                    'mesh_carve.py', 'mesh_rebuild_item.py')},
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
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
            raise FileExistsError('output already exists with a different signature or files')
        # An interrupted first run (no result.json) is resumed in place; this
        # directory belongs to the current item-35 task, not to an older version.
    a.out.mkdir(parents=True, exist_ok=True)

    atlas_paths, texture_checks = make_atlas(CLEAN, RUST, a.out)
    label_paths, label_check = make_label(CLEAN, a.out)
    dial_paths, dial_check = make_dial(CLEAN, a.out)
    textures = {**atlas_paths, **label_paths, **dial_paths}
    request = {'out': str(a.out), 'original_mesh': str(OLD_MESH),
               'source_camera': row['reconstruction']['source_camera'],
               'source_alpha_bbox': source_alpha_bbox,
               'textures': textures, 'dimensions': DIMENSIONS, 'draft': a.draft}
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

    from mesh_checks import source_fidelity
    with Image.open(OLD_ALPHA) as alpha, \
            Image.open(a.out / 'renders' / 'new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                   np.asarray(rendered.getchannel('A')))
    source_fit = _mask_fit(np.asarray(Image.open(OLD_ALPHA).convert('L')) >= 128)
    new_fit = _mask_fit(np.asarray(Image.open(a.out / 'renders' / 'new-source.png')
                                   .getchannel('A')) >= 128)
    aligned_fidelity = {'iou': _iou(source_fit, new_fit),
                        'alignment': ('longest-edge bounding-box alignment; the source photo '
                                      'is not a calibrated camera'),
                        'acceptance_gate': False}
    best = None
    for i in range(24):
        with Image.open(a.out / 'renders' / f'turn-{i:02d}.png') as turn:
            fit = _mask_fit(np.asarray(turn.getchannel('A')) >= 128)
        score = _iou(source_fit, fit)
        if score is not None and (best is None or score > best[1]):
            best = (i, score)
    best_aligned = {'best_turntable_view_deg': None if best is None else best[0] * 15,
                    'iou': None if best is None else round(best[1], 4),
                    'acceptance_gate': False,
                    'note': '24-direction bounding-box-aligned diagnostic only.'}

    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    mesh_audit = mesh_topology(a.out / 'mesh.glb')
    openings = opening_checks(a.out / 'mesh.glb', DIMENSIONS)
    render_openings = {
        'nozzle_through_interior_holes': silhouette_holes(a.out / 'renders' / 'nozzle-through.png'),
        'safety_ring_interior_holes': silhouette_holes(a.out / 'renders' / 'safety-ring.png'),
    }
    render_openings['passed'] = bool(render_openings['nozzle_through_interior_holes'] == 1
                                     and render_openings['safety_ring_interior_holes'] == 1)
    metal = metal_highlight_shift(a.out / 'renders' / 'metal-light-a.png',
                                  a.out / 'renders' / 'metal-light-b.png')
    report_rows = make_six_view_board(a.out, reference) if not a.draft else []
    if not a.draft:
        make_detail_board(a.out)
        make_isolated_board(a.out, openings)
        make_material_boards(a.out)
        make_turntable(a.out)
        make_texture_boards(a.out)
    make_source_board(a.out, {'source_fidelity_diagnostic': {**fidelity, 'iou': fidelity['iou']}})

    mesh_sha = digest(a.out / 'mesh.glb')
    blend_sha = digest(a.out / 'model.blend')
    name_counts = {}
    for part in worker['parts']:
        name = part['name']
        if name == '35 body':
            key = 'body'
        elif 'rolled seam' in name:
            key = 'rolled_seams'
        elif name == '35 bottom lip':
            key = 'bottom_lip'
        elif name == '35 valve collar':
            key = 'valve_collar'
        elif name in ('35 lower handle', '35 upper handle lever'):
            key = 'handle_bars'
        elif name == '35 handle bracket':
            key = 'handle_bracket'
        elif name.startswith('35 valve'):
            key = 'valve_body'
        elif 'gauge' in name:
            key = 'gauge_parts'
        elif 'pivot' in name:
            key = 'pivot_parts'
        elif name == '35 safety pin':
            key = 'safety_pin'
        elif 'seal' in name:
            key = 'tamper_seal'
        elif name.startswith('35 nozzle'):
            key = 'nozzle_parts'
        elif name == '35 label':
            key = 'label'
        else:
            key = 'other'
        name_counts[key] = name_counts.get(key, 0) + 1
    structural = {
        'counts': name_counts,
        'gauge_chrome_parts': sum(1 for part in worker['parts']
                                  if 'chrome' in part['name'] or 'screw' in part['name']),
        'note': ('Part presence proves only that a feature was modelled, not its size, '
                 'fit, hidden faces or assembly. Gauge internals, valve internals, '
                 'bracket fastening and horn wall thickness are inferred.'),
    }
    checks = {
        'source_fidelity': bool(fidelity['passed']),
        'source_fidelity_acceptance': False,
        'reference_diagnostic_only': True,
        'mesh_closed_after_welding': bool(not mesh_audit['open_parts']),
        'same_part_self_intersections': bool(not worker['self_intersecting_components']),
        'structural_parts': bool(len(worker['parts']) >= 20),
        'isolated_openings_geometry': openings['passed'],
        'isolated_openings_rendered': render_openings['passed'],
        'material_tiles': texture_checks['passed'],
        'label_texture': label_check['passed'],
        'dial_texture': dial_check['passed'],
        'metal_reflection': metal['passed'],
        'browser_page': False,
        'browser_review_executed': False,
        'scene_validation': False,
        'settle_validation': False,
        'measurement_calibration': False,
    }
    pending = [
        '六视图参考未标定，且与实物不是同一台；只作外观与背侧拓扑参考，不是实测。',
        '照片无标定：筒体直径按 MFZ/ABC3 常见 150 mm 假设，整件高度、表盘直径、喇叭口尺寸和倾角均为像素比例推断。',
        '按钮/手柄内部杠杆、阀体内部通道、保险销机构、玻璃厚度、密封和壁厚不可见，未验证。',
        '标签印刷仅由照片矫正；标签包角、拼缝位置、背面标签状态未知；表盘刻度真实数值未标定。',
        '锈蚀只做局部磨损提示；真实锈蚀分布、掉漆和凹痕未逐点复刻。',
        '原照背面、底面中心凹深、手柄背面与隐藏面未观察到。',
        '落稳、接触面、现场尺度、遮挡顺序和现场光照未验证，不能进入自动放置或正式合成。',
        '浏览器交互检查未执行；本页使用 Blender 渲染图和离线文件。',
    ]
    inputs = {
        'clean': CLEAN_SHA256, 'rust': RUST_SHA256,
        'reference': REFERENCE_SHA256, 'original_mesh': OLD_MESH_SHA256,
        'source_alpha': OLD_ALPHA_SHA256,
    }
    result = {
        'schema_version': 1,
        'number': NUMBER,
        'version': VERSION,
        'status': 'review',
        'model_review': 'pending',
        'scene_validation': 'pending',
        'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_visual_match_reference',
        'reference_role': 'visual_match_unverified',
        'reference': {'name': REFERENCE_NAME, 'sha256': REFERENCE_SHA256,
                      'role': 'visual_match_unverified',
                      'note': '按外观匹配；图片未提供相机标定，结构不能视为实测。'},
        'inputs': inputs,
        'original_mesh_sha256': OLD_MESH_SHA256,
        'original_source_fidelity_record': audit.get('source_fidelity'),
        'original_structural_warning': ('旧模型六视角审计见 outputs/三维模型-39件六视角审计/35；'
                                        '原模型背面出现与照片不符的竖向附着物，表面糊化、'
                                        '手柄/表盘/标签粘连。'),
        'original_audit': audit,
        'mesh_sha256': mesh_sha,
        'blend_sha256': blend_sha,
        'signature': signature,
        'dimensions': DIMENSIONS,
        'pixel_measurements': PIXEL_MEASUREMENTS,
        'source_measurements_note': ('像素比例来自正照片；以同类 MFZ/ABC3 常见 150 mm 筒径'
                                     '为包络假设，不是对实物的测量。'),
        'reference_views': report_rows,
        'source_fidelity_diagnostic': {**fidelity, 'acceptance_gate': False,
                                       'note': '标定相机诊断；未标定，不能作为验收。'},
        'source_fidelity_aligned_diagnostic': aligned_fidelity,
        'source_fidelity_aligned_best_turntable': best_aligned,
        'checks': checks,
        'isolated_opening_checks': openings,
        'isolated_opening_render_checks': render_openings,
        'texture_checks': texture_checks,
        'label_texture_check': label_check,
        'dial_texture_check': dial_check,
        'metal_reflection_check': metal,
        'mesh_audit': mesh_audit,
        'structural_checks': structural,
        'worker': worker,
        'pending_review': pending,
        'review_page': 'index.html',
    }
    (a.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    (a.out / 'old-model-audit.json').write_text(json.dumps(
        {'audit': audit, 'library_record': {k: row.get(k) for k in
                                            ('mesh', 'mesh_sha256', 'rust', 'rust_sha256',
                                             'original', 'clean_sha256', 'source_view',
                                             'source_view_sha256', 'source_alpha_sha256')}},
        ensure_ascii=False, indent=2), encoding='utf8')
    (a.out / 'source-measurements.json').write_text(json.dumps(
        {'number': NUMBER, 'pixel_measurements': PIXEL_MEASUREMENTS,
         'derived_dimensions_m': DIMENSIONS,
         'assumptions': ['150 mm body diameter from a similar MFZ/ABC3 catalogue envelope',
                         'uncalibrated single front photograph; no depth or true scale',
                         'label arc and gauge tilt inferred',
                         'six-view sheet is a visual-match reference, not a measurement'],
         'observed_text': 'MFZ/ABC3 / 手提式干粉灭火器 / 3±0.09kg / 1.2MPa'},
        ensure_ascii=False, indent=2), encoding='utf8')
    tables = {'checks': checks, 'label': label_check, 'dial': dial_check,
              'texture': texture_checks, 'metal': metal, 'openings': openings,
              'render_openings': render_openings, 'reference_views': report_rows,
              'fidelity': fidelity, 'aligned': aligned_fidelity,
              'best_aligned': best_aligned, 'structural': structural, 'pending': pending}
    if not a.draft:
        make_review_page(a.out, result, tables)
    progress = {'number': NUMBER, 'status': 'review', 'manual_only': True,
                'version': VERSION, 'out': str(a.out), 'mesh': str(a.out / 'mesh.glb'),
                'blend': str(a.out / 'model.blend'),
                'checks': checks, 'pending_review': pending, 'next_number': 36}
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    report = [
        '# 35 灭火器2 · 六视图分部件重建候选', '',
        f'- 版本：`{VERSION}`；状态：`review`；`manual_only=True`，待人工复核，不进入正式合成。',
        f'- 输入：原照、做锈图、六视图（视觉匹配、未标定）、旧模型哈希均已核对。',
        f'- 输出：`mesh.glb`（Z-up，与库约定一致）、`model.blend`、PBR 贴图、六视图新旧对照、局部/隔离/材质/转台图。', '',
        '## 已建模结构', '',
        '筒体、凹底、底圈、上下卷边、绿色阀颈、阀体、铬制表圈、照片矫正表盘、3D 指针、玻璃、中心螺钉、上下手柄、支架、销轴、保险销（真实环孔）、红色铅封、白色外扩喇叭口（真实通孔）、曲面标签。', '',
        '## 实际执行的检查', '',
        f"- 网格部件/闭合：{mesh_audit['parts']} / {mesh_audit['closed_after_welding_seams']}",
        f"- 同部件自交：{worker['self_intersecting_components'] or '无'}",
        f"- 几何开孔：喇叭口通孔 {openings['nozzle_bore_ray']['passed']}，保险销环 {openings['safety_pin_ring']['passed']}，凹底 {openings['concave_bottom']['passed']}",
        f"- 隔离渲染孔数：喇叭口轴向 {render_openings['nozzle_through_interior_holes']}，保险销环 {render_openings['safety_ring_interior_holes']}",
        f"- 贴图数值：{texture_checks['passed']}；标签：{label_check['passed']}；表盘：{dial_check['passed']}",
        f"- 铬件高光随光位移动：{metal['passed']}（位移 {metal.get('shift_px')} px）",
        f"- 原相机轮廓 IoU（仅诊断，未标定）：{fidelity['iou']:.4f}；填孔 {fidelity['filled_holes']}",
        f"- 旧模型原图审计：{audit['source_fidelity']['iou']:.4f}，通过 {audit['source_fidelity']['passed']}",
        '', '## 待人工复核', ''] + [f'- {item}' for item in pending] + [
        '', '## 文件', '',
        '`mesh.glb`, `model.blend`, `base_color.png`, `metallic_roughness.png`, `normal.png`, '
        '`label_base.png`, `dial_base.png`, `六视图-新旧对照.jpg`, `局部检查.jpg`, '
        '`结构开口-隔离检查.jpg`, `材质-同光照对照.jpg`, `材质-变光检查.jpg`, '
        '`表面反射-变光对照.jpg`, `转台-24方向.jpg`, `result.json`, `hashes.json`, `index.html`.',
    ]
    (a.out / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf8')
    hashes = {'inputs': inputs,
              'outputs': {p.name: digest(p) for p in sorted(a.out.iterdir())
                          if p.is_file() and p.name != 'hashes.json'},
              'renders': {p.name: digest(p) for p in sorted((a.out / 'renders').iterdir())
                          if p.is_file()}}
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out),
                      'mesh_sha256': mesh_sha, 'blend_sha256': blend_sha,
                      'checks': checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
