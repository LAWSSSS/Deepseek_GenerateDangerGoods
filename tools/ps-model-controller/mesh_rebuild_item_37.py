"""Build the review-only part-based candidate for item 37.

Item 37 is the black ``集油器`` photograph: an inline pneumatic oil collector /
filter with a rectangular slotted mounting plate, a ridged neck, an
overhanging cap, a black plastic cage with real windows and a white woven
filter element, a wide collar, a smoke-grey translucent bowl with internal
ribs, and a knurled drain knob ending in a brass pin.

The six-view map has no reference for item 37 (``missing``), so only the
photograph and its visible proportions are used; all hidden surfaces are
inferred.  The old TripoSR shell is a fused, smoothed, opaque blob and is kept
unchanged for comparison.  The candidate is manual-only / pending review and
never replaces the library mesh, enters automatic placement or the formal
composite.
"""
import argparse
import hashlib
import json
import math
import os
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

NUMBER = 37
VERSION = 'oilcollector37-parametric-v1'
SOURCE_SIZE = (1071, 1468)
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
FONT = 'C:/Windows/Fonts/msyh.ttc'

# Working scale: the photograph's 1389 px overall height is mapped to 1.0 m.
# This is not a measurement; it keeps the candidate comparable with the
# library's calibrated camera, whose old mesh is 1.0135 m tall.
DIMENSIONS = {
    'plate': {
        'width': 0.3168, 'height': 0.1937, 'thickness': 0.0120,
        'z0': 0.3063, 'z1': 0.5000,
        'holes': [
            {'y': -0.0947, 'z': 0.4287, 'w': 0.0640, 'h': 0.0475},
            {'y': 0.0947, 'z': 0.4287, 'w': 0.0640, 'h': 0.0475},
        ],
    },
    'neck': {
        'width': 0.2203, 'depth': 0.1700, 'z0': 0.2322, 'z1': 0.3063,
        'rib_radius': 0.0110,
        'rib_y': [-0.0880, -0.0440, 0.0, 0.0440, 0.0880],
    },
    'cap': {'radius': 0.2477, 'z0': 0.2012, 'z1': 0.2322},
    'cage': {
        'r_top': 0.2102, 'r_bottom': 0.2022,
        'z_top': 0.1998, 'z_bottom': -0.0950, 'wall': 0.0120,
        'bands': {
            'top': {'z0': 0.1170, 'z1': 0.1998},
            'mid1': {'z0': 0.0356, 'z1': 0.0724},
            'mid2': {'z0': -0.0436, 'z1': -0.0097},
            'bottom': {'z0': -0.0950, 'z1': -0.0860},
        },
        'posts': [
            {'name': 'front', 'a0': -3.0, 'a1': 3.0, 'segs': 4},
            {'name': 'side_r', 'a0': 67.0, 'a1': 113.0, 'segs': 18},
            {'name': 'back', 'a0': 177.0, 'a1': 183.0, 'segs': 4},
            {'name': 'side_l', 'a0': 247.0, 'a1': 293.0, 'segs': 18},
        ],
    },
    'filter': {'r': 0.1900, 'z0': -0.0980, 'z1': 0.1960},
    'slots': {
        'angles': [42.0, 138.0, 222.0, 318.0], 'half_angle': 9.0,
        'z0': 0.1360, 'z1': 0.1670,
        'r_inset': 0.2090, 'r_frame': 0.2125,
    },
    'collar': {'r_in': 0.2060, 'r_out': 0.2455,
               'z0': -0.1587, 'z1': -0.0738},
    'bowl': {
        'r_top': 0.2420, 'z0': -0.3745, 'z1': -0.1587,
        'rib_angles': [45.0, 135.0, 225.0, 315.0],
        'rib_half_angle': 2.6, 'rib_z0': -0.3000, 'rib_z1': -0.1660,
    },
    'drain': {
        'neck_radius': 0.0753, 'neck_z0': -0.3960, 'neck_z1': -0.3740,
        'knob_radius': 0.0684, 'knob_z0': -0.4600, 'knob_z1': -0.3960,
        'ridges': 48, 'ridge_depth': 0.0025,
        'brass': {'top_z': -0.4626, 'recess_z': -0.4880},
    },
    'overall': {'top': 0.5000, 'bottom': -0.5000, 'width': 0.4954,
                'source_pixels': [688, 1389]},
}
RENDER = {
    'six_resolution': 512, 'six_samples': 36,
    'detail_resolution': 512, 'detail_samples': 32,
    'turn_resolution': 320, 'turn_samples': 8, 'turn_count': 24,
    'device': 'OPTIX', 'reference_role': 'missing',
}


# ---------------------------------------------------------------------------
# Inputs and paths
# ---------------------------------------------------------------------------
def local_paths():
    return {
        'original': ROOT / 'outputs/物品高清输入-39件/原貌HD/37-集油器.png',
        'rust': ROOT / 'outputs/物品高清输入-39件/生锈HD/37-集油器.png',
        'old_mesh': ROOT / ('outputs/三维模型库-圆润复验/items/'
                            '37-asset-d656d54d7faa/mesh.glb'),
        'source_view': ROOT / ('outputs/三维模型库-圆润复验/items/'
                               '37-asset-d656d54d7faa/source-view.png'),
        'source_alpha': ROOT / ('outputs/三维模型库-圆润复验/items/'
                                '37-asset-d656d54d7faa/source-alpha.png'),
        'old_result': ROOT / ('outputs/三维模型库-圆润复验/items/'
                              '37-asset-d656d54d7faa/result.json'),
        'audit': ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl',
    }


def verify_inputs(paths):
    expected = {
        'original': '92236b201b3a3550c6b4d0823e198358e0279d7007744eb83ee8f65c3069c974',
        'rust': 'b323c7442d7eda527de223e572da13402e74d489e478850bd3d77ebdbc71af3d',
        'old_mesh': '79ddbb74ab99799944cd2907d462a631950ba07ae3c6ec05c5a9d5b8817822ed',
        'source_view': 'f81f5287da0f7f7a938f3d0f3ba474da1c488c05946b2354d0b9a4db0de89297',
        'source_alpha': 'd5c778e16a5404b2d68e23a4e1e312b8924f5dff8e780c169b255313a6f27e22',
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


def make_black_plastic(out, key, seed):
    h = w = 1024
    n_broad = fbm(h, w, seed, (220, 70), (.62, .38))
    n_mid = periodic_noise(h, w, 9, seed + 1)
    n_fine = periodic_noise(h, w, 3, seed + 2)
    base = 33.0 + 2.6 * n_broad + 1.2 * n_mid + .8 * n_fine
    img = np.dstack([base, base + .7, base + 1.4])
    rough = np.clip(.435 + .030 * n_broad + .015 * n_mid, 0, 1)
    metal = np.zeros((h, w), np.float32)
    height = .15 * n_mid + .08 * n_fine + .06 * n_broad
    normal = normal_from_height(height, .45)
    paths = (out / f'{key}_base.png', out / f'{key}_mr.png',
             out / f'{key}_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {
        'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
        'grain_std': round(float(img[..., 0].std()), 3),
        'roughness_mean': round(float(rough.mean()), 4),
        'metallic_mean': round(float(metal.mean()), 4),
        'dark_fraction': round(float((img[..., 0] < 70).mean()), 4),
    }
    stats['passed'] = bool(stats['mean_rgb'][0] < 70 and
                           stats['grain_std'] > 2.0 and
                           .30 <= stats['roughness_mean'] <= .62 and
                           stats['metallic_mean'] <= .02)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_black_matte(out, key, seed):
    h = w = 512
    n1 = fbm(h, w, seed, (120, 40), (.6, .4))
    n2 = periodic_noise(h, w, 4, seed + 1)
    base = 24.0 + 3.0 * n1 + 1.5 * n2
    img = np.dstack([base, base + .5, base + 1.0])
    rough = np.clip(.715 + .035 * n1 + .015 * n2, 0, 1)
    metal = np.zeros((h, w), np.float32)
    normal = normal_from_height(.13 * n1 + .05 * n2, .35)
    paths = (out / f'{key}_base.png', out / f'{key}_mr.png',
             out / f'{key}_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
             'roughness_mean': round(float(rough.mean()), 4),
             'metallic_mean': round(float(metal.mean()), 4)}
    stats['passed'] = bool(stats['mean_rgb'][0] < 55 and
                           .60 <= stats['roughness_mean'] <= .85 and
                           stats['metallic_mean'] <= .02)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_filter_mesh(out, key, seed):
    h = w = 1024
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    period = 72.0
    warp = .5 + .5 * np.sin(math.tau * u * period)
    weft = .5 + .5 * np.sin(math.tau * v * period)
    weave = warp * weft
    n_broad = fbm(h, w, seed, (110, 30), (.6, .4))
    n_fine = periodic_noise(h, w, 3, seed + 1)
    base = 150.0 + 78.0 * weave + 6.0 * n_broad + 3.0 * n_fine
    img = np.dstack([base, base - 3.0, base - 9.0])
    rough = np.clip(.55 - .18 * weave + .03 * n_broad, 0, 1)
    metal = np.full((h, w), .10, np.float32)
    height = .60 * weave + .10 * n_broad + .04 * n_fine
    normal = normal_from_height(height, 1.05)
    paths = (out / f'{key}_base.png', out / f'{key}_mr.png',
             out / f'{key}_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    low, high = np.percentile(img[..., 0], [5, 95])
    stats = {
        'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
        'weave_contrast': round(float(high - low), 2),
        'roughness_mean': round(float(rough.mean()), 4),
        'metallic_mean': round(float(metal.mean()), 4),
    }
    stats['passed'] = bool(stats['mean_rgb'][0] > 120 and
                           stats['weave_contrast'] > 45 and
                           .35 <= stats['roughness_mean'] <= .70 and
                           stats['metallic_mean'] <= .25)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_smoke_bowl(out, key, seed):
    h = w = 512
    n1 = fbm(h, w, seed, (160, 50), (.65, .35))
    n2 = periodic_noise(h, w, 5, seed + 1)
    base = 74.0 + 5.0 * n1 + 1.6 * n2
    img = np.dstack([base, base + .4, base + .8])
    rough = np.clip(.050 + .015 * n1 + .008 * n2, 0, 1)
    metal = np.zeros((h, w), np.float32)
    normal = normal_from_height(.05 * n1 + .02 * n2, .22)
    paths = (out / f'{key}_base.png', out / f'{key}_mr.png',
             out / f'{key}_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
             'roughness_mean': round(float(rough.mean()), 4),
             'metallic_mean': round(float(metal.mean()), 4)}
    stats['passed'] = bool(stats['mean_rgb'][0] < 110 and
                           stats['roughness_mean'] < .12 and
                           stats['metallic_mean'] <= .02)
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def make_brass(out, key, seed):
    h = w = 512
    n1 = fbm(h, w, seed, (100, 30, 8), (.55, .30, .15))
    n2 = periodic_noise(h, w, 4, seed + 1)
    tarnish = smoothstep(n1 + .30 * n2, 1.15, 2.4) * .40
    img = np.dstack([150 + 12 * n1 + 4 * n2,
                     113 + 10 * n1 + 3 * n2,
                     50 + 7 * n1 + 2 * n2])
    img = img * (1 - tarnish[..., None]) + \
        np.dstack([108., 85., 48.]) * tarnish[..., None]
    rough = np.clip(.235 + .055 * n1 + .020 * n2 + .09 * tarnish, 0, 1)
    metal = np.full((h, w), .995, np.float32)
    normal = normal_from_height(.11 * n1 + .04 * n2, .42)
    paths = (out / f'{key}_base.png', out / f'{key}_mr.png',
             out / f'{key}_normal.png')
    save_rgb(paths[0], img)
    save_rgb(paths[1], pack_mr(rough, metal))
    save_rgb(paths[2], normal)
    stats = {'mean_rgb': [round(float(img[..., i].mean()), 2) for i in range(3)],
             'roughness_mean': round(float(rough.mean()), 4),
             'metallic_mean': round(float(metal.mean()), 4)}
    stats['passed'] = bool(stats['metallic_mean'] > .90 and
                           stats['roughness_mean'] < .36 and
                           stats['mean_rgb'][0] > stats['mean_rgb'][1] >
                           stats['mean_rgb'][2])
    return {'base': str(paths[0]), 'mr': str(paths[1]),
            'normal': str(paths[2])}, stats


def build_textures(texture_dir):
    texture_dir.mkdir(parents=True, exist_ok=True)
    specs = {}
    stats = {}
    specs['black_plastic'], stats['black_plastic'] = make_black_plastic(
        texture_dir, 'black_plastic', 3701)
    specs['black_matte'], stats['black_matte'] = make_black_matte(
        texture_dir, 'black_matte', 3801)
    specs['filter_mesh'], stats['filter_mesh'] = make_filter_mesh(
        texture_dir, 'filter_mesh', 3901)
    specs['smoke_bowl'], stats['smoke_bowl'] = make_smoke_bowl(
        texture_dir, 'smoke_bowl', 4001)
    specs['brass'], stats['brass'] = make_brass(texture_dir, 'brass', 4101)
    labels = {
        'black_plastic': ('黑色工程塑料（支架/颈部/端盖/护罩/环座/排污钮）',
                          0.0, .435, .05, .14),
        'black_matte': ('黑色哑光凹槽内衬', 0.0, .715, 0.0, .12),
        'filter_mesh': ('白色编织滤网（外观近似）', .12, .55, 0.0, .12),
        'smoke_bowl': ('烟灰色半透明塑料集污杯', 0.0, .06, 0.0, .08),
        'brass': ('黄铜排污阀芯', 1.0, .28, 0.0, .12),
    }
    normal_strength = {
        'black_plastic': .45, 'black_matte': .30, 'filter_mesh': 1.05,
        'smoke_bowl': .22, 'brass': .38,
    }
    materials = {}
    for key, spec in specs.items():
        label, metallic, roughness, coat, coat_rough = labels[key]
        entry = dict(spec)
        entry.update({
            'label': label, 'metallic': metallic, 'roughness': roughness,
            'coat': coat, 'coat_roughness': coat_rough, 'alpha': False,
            'normal_strength': normal_strength[key],
        })
        if key == 'smoke_bowl':
            entry['transmission'] = .94
            entry['ior'] = 1.49
        materials[key] = entry
    stats['passed'] = bool(all(stats[k]['passed'] for k in stats))
    stats['note'] = ('贴图由本件代码程序化生成；黑色塑料颗粒、编织滤网网纹'
                     '和烟灰半透明杯均为按原照重绘的外观近似，不是实测材料图。'
                     '只有黄铜阀芯按金属处理。')
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
    path = FONT
    if bold and Path('C:/Windows/Fonts/msyhbd.ttc').is_file():
        path = 'C:/Windows/Fonts/msyhbd.ttc'
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
        draw.text((pad, pad + size + 20), subtitle, font=font(12),
                  fill=(92, 104, 98))
    return canvas


def compose(path, panels, columns, title, note, width=None):
    first = panels[0]
    cell_w = (first[0] if isinstance(first, tuple) else first).width
    cell_h = (first[0] if isinstance(first, tuple) else first).height
    rows = math.ceil(len(panels) / columns)
    head = 74
    if width is None:
        width = 16 + columns * (cell_w + 10) + 16
    canvas = Image.new('RGB', (width, head + rows * cell_h + 46),
                       (238, 241, 238))
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
    new_panels = [panel(load_rgba(render_dir / f'new-{view}.png'), 400,
                        f'新版 {view}') for view in VIEWS]
    old_panels = [panel(load_rgba(render_dir / f'old-{view}.png'), 400,
                        f'旧模型 {view}') for view in VIEWS]
    src = panel(load_rgba(local_paths()['original']), 400, '原照（原貌，未标定）')
    rust = panel(load_rgba(local_paths()['rust']), 400, '做锈图（外观素材）')
    cell_w = new_panels[0].width
    cell_h = new_panels[0].height
    width = 16 + 3 * (cell_w + 10) + 16
    head = 74
    headings = 28
    height = (head + cell_h + 18 + headings + 2 * (cell_h + 8) + 18 +
              headings + 2 * (cell_h + 8) + 46)
    canvas = Image.new('RGB', (width, height), (238, 241, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 18), '37 集油器 · 六方向新旧对照', font=font(24, True),
              fill=(30, 44, 58))
    draw.text((22, 48), '六视图映射为 missing（无可靠对应参考）；只按原照可见结构重建，'
                        '旧模型为 TripoSR 单体壳。', font=font(13), fill=(87, 98, 92))
    y = head
    for index, image in enumerate((src, rust)):
        x = 16 + index * (cell_w + 10) + (width - 2 * cell_w - 10 - 32) // 2
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1),
                       outline=(206, 213, 207))
    y += cell_h + 18
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
    draw.text((22, y), '旧模型 · 六方向（旧壳结构与姿态未修改）', font=font(18, True),
              fill=(138, 90, 30))
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
    details = ('bracket', 'neck', 'cap', 'cage', 'windows', 'slots',
               'collar', 'bowl', 'drain', 'overall')
    labels = {
        'bracket': '安装板/双通孔', 'neck': '带筋颈部',
        'cap': '端盖/顶缘', 'cage': '护罩/三排窗口',
        'windows': '窗口与白色滤网', 'slots': '上部浅凹槽',
        'collar': '宽环座', 'bowl': '半透明集污杯/内筋',
        'drain': '滚花排污钮/黄铜芯', 'overall': '整体 3/4',
    }
    panels = []
    for name in details:
        panels.append(panel(load_rgba(render_dir / f'detail-{name}.png'), 330,
                            f'新版 · {labels[name]}'))
    for name in details:
        panels.append(panel(load_rgba(render_dir / f'old-detail-{name}.png'), 330,
                            f'旧模型 · {labels[name]}'))
    compose(out / '关键局部-新旧对照.jpg', panels, 5,
            '37 集油器 · 关键局部新旧对照',
            '旧模型局部按旧网格主轴近似定位，仅示融合/糊化/不透明，不代表同部位精确对齐。')

    isolation = [
        ('isolate-bracket.png', '安装板/双通孔隔离'),
        ('isolate-cage.png', '护罩/滤网隔离（窗口通透）'),
        ('isolate-cap-collar.png', '端盖/颈部/环座隔离'),
        ('isolate-bowl.png', '半透明杯/内筋隔离'),
        ('isolate-drain.png', '排污钮/黄铜芯隔离'),
        ('isolate-brass.png', '黄铜芯单独隔离'),
    ]
    panels = [panel(load_rgba(render_dir / name), 400, label)
              for name, label in isolation]
    compose(out / '结构开口-隔离检查.jpg', panels, 3,
            '37 集油器 · 结构、开口与材质隔离检查',
            '窗口、双通孔与黄铜凹槽由射线检查确认；隔离渲染用于人工观察装配与材质。')

    panels = [
        panel(load_rgba(render_dir / 'new-material-3q.png'), 460, '新版 · 同光照 3/4'),
        panel(load_rgba(render_dir / 'old-material-3q.png'), 460, '旧模型 · 同光照 3/4'),
        panel(load_rgba(render_dir / 'new-clay-front.png'), 460, '新版 · 灰模正视图'),
        panel(load_rgba(render_dir / 'new-front.png'), 460, '新版 · 材质正视图'),
    ]
    compose(out / '材质-同光照对照.jpg', panels, 2,
            '37 集油器 · 材质与灰模对照',
            '黑色工程塑料为电介质；滤网为电介质主导；只有黄铜按金属处理。'
            '灰模不参与外观验收。')

    panels = [
        panel(load_rgba(render_dir / 'new-front.png'), 460, '新版 · 主光 A'),
        panel(load_rgba(render_dir / 'new-light-alternate.png'), 460, '新版 · 主光 B'),
        panel(load_rgba(render_dir / 'metal-only-light-a.png'), 460, '黄铜隔离 · 主光 A'),
        panel(load_rgba(render_dir / 'metal-only-light-b.png'), 460, '黄铜隔离 · 主光 B'),
    ]
    compose(out / '表面反射-变光对照.jpg', panels, 2,
            '37 集油器 · 变光与黄铜反射对照',
            '只对黄铜阀芯做金属高光移动检查；黑塑料、滤网与半透明杯不适用。')

    panels = [
        panel(load_rgba(local_paths()['original']), 460, '原照（原貌）'),
        panel(load_rgba(local_paths()['rust']), 460, '做锈图'),
        panel(load_rgba(render_dir / 'new-source-3q.png'), 460, '新版 · 近似原照视角'),
        panel(load_rgba(render_dir / 'old-source.png'), 460, '旧模型 · 库标定相机'),
        panel(load_rgba(render_dir / 'new-front.png'), 460, '新版 · 正视图'),
        panel(load_rgba(render_dir / 'new-left.png'), 460, '新版 · 左视图'),
    ]
    compose(out / '原照-新候选-参考.jpg', panels, 3,
            '37 集油器 · 原照、旧模型、新候选',
            '六视图映射缺失；原照为近似正面产品图且未标定，IoU 只作诊断。')

    tex_items = [
        ('black_plastic_base.png', '黑塑料底色'),
        ('black_plastic_mr.png', '黑塑料金属/粗糙度'),
        ('black_plastic_normal.png', '黑塑料法线'),
        ('filter_mesh_base.png', '滤网底色'),
        ('filter_mesh_normal.png', '滤网法线'),
        ('smoke_bowl_base.png', '半透明杯底色'),
        ('smoke_bowl_mr.png', '半透明杯金属/粗糙度'),
        ('brass_base.png', '黄铜底色'),
        ('brass_mr.png', '黄铜金属/粗糙度'),
    ]
    panels = [panel(load_rgba(out / 'textures' / name), 340, label)
              for name, label in tex_items]
    compose(out / 'PBR贴图-对照.jpg', panels, 3,
            '37 集油器 · PBR 贴图检查',
            '程序化贴图并打包进 GLB；金属/粗糙度通道为 R=255、G=粗糙度、B=金属度。'
            '网纹、颗粒和烟灰色为外观近似，不是实测材料图。')

    panels = []
    for i in range(24):
        angle = i * 15
        panels.append(panel(load_rgba(render_dir / f'turn-new-{i:02d}.png'), 240,
                            f'{angle:03d}°'))
    compose(out / '转台-24方向.jpg', panels, 6,
            '37 集油器 · 新版 24 方向转台',
            '每 15° 一张 Blender 实测渲染；覆盖外观，不证明隐藏结构或装配。')


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


def source_diagnostics(out, paths):
    raw_source, rotated, rotation_deg = rotated_source_mask(paths)
    render_dir = out / 'renders'
    new_source = np.asarray(load_rgba(render_dir / 'new-source-alpha.png').getchannel('A'))
    old_source = np.asarray(load_rgba(render_dir / 'old-source-alpha.png').getchannel('A'))
    strict_source = np.asarray(Image.open(paths['source_alpha']).convert('L'))
    if float(strict_source.std()) < 1e-3:
        strict = {'iou': None, 'available': False, 'acceptance_gate': False,
                  'reason': '库内 source-alpha 为全不透明通道，不能作为标定轮廓。'}
        old_strict = dict(strict)
    else:
        strict = source_fidelity(strict_source, new_source)
        strict['available'] = True
        strict['acceptance_gate'] = False
        strict['scale_note'] = ('库标定相机；未实测标定，严格 IoU 只作方位诊断')
        old_strict = source_fidelity(strict_source, old_source)
        old_strict['available'] = True
        old_strict['acceptance_gate'] = False
    source_view = np.asarray(Image.open(paths['source_view']).convert('RGBA'))
    view_mask = (source_view[:, :, 3] >= 128).astype(np.uint8) * 255
    view_mask = cv2.morphologyEx(view_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
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
        'reference_role': 'missing',
        'reference_note': '六视图映射中本件为 missing，无可靠对应参考；未做参考图 IoU。',
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
        'source_vertical_rotation_deg': rotation_deg,
        'best_view': best[0] if best else 0,
        'best': best_metrics,
        'note': ('所有 IoU 均为未标定诊断。严格同相机 IoU 使用库内 source-alpha '
                 '的旧模型轮廓；其余为包围盒/主轴对齐诊断。本件无六视图参考，'
                 '不能当作摄影测量或验收。'),
    }


def metal_reflection_check(out):
    a = np.asarray(load_rgba(out / 'renders' / 'metal-only-light-a.png'))
    b = np.asarray(load_rgba(out / 'renders' / 'metal-only-light-b.png'))
    mask = np.logical_and(a[:, :, 3] >= 128, b[:, :, 3] >= 128)
    if mask.sum() < 800:
        return {'metal_pixels': int(mask.sum()), 'passed': False,
                'reason': '黄铜隔离渲染非背景像素不足，金属反射检查不成立'}
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
    result['note'] = ('只证明黄铜阀芯的高光随灯位移动；黑色塑料、编织滤网和'
                      '烟灰色半透明杯为电介质主导，本检查不适用于它们，'
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
    rgb = arr[:, :, :3].astype(float)
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY).astype(float)
    dark = np.logical_and(mask, gray < 60)
    light = np.logical_and(mask, rgb.min(axis=2) > 130)
    brass = np.logical_and.reduce([mask, rgb[:, :, 0] > 85,
                                   rgb[:, :, 0] > rgb[:, :, 1] + 12,
                                   rgb[:, :, 1] > rgb[:, :, 2] + 12])
    rust_image = Image.open(paths['rust']).convert('RGB').resize(
        (arr.shape[1], arr.shape[0]), Image.LANCZOS)
    rust = np.asarray(rust_image).astype(float)
    clean = rgb
    diff = np.linalg.norm(rust - clean, axis=2)
    brown = np.logical_and.reduce([mask, rust[:, :, 0] >= rust[:, :, 2] + 14,
                                   rust[:, :, 0] >= rust[:, :, 1] - 6,
                                   rust[:, :, 0] >= 72, rust[:, :, 1] <= 155])
    ow, oh = DIMENSIONS['overall']['width'], (DIMENSIONS['overall']['top'] -
                                              DIMENSIONS['overall']['bottom'])
    return {
        'image': str(paths['original'].relative_to(ROOT)), 'size': SOURCE_SIZE,
        'alpha_area_px': int(mask.sum()), 'bbox': bbox,
        'min_area_rect': {
            'centre': [round(float(cx), 2), round(float(cy), 2)],
            'size': [round(float(w), 2), round(float(h), 2)],
            'angle_deg': round(float(angle), 2),
            'height_over_width': round(float(max(w, h) / min(w, h)), 4)},
        'colour_fractions': {
            'dark_plastic': round(float(dark.sum() / mask.sum()), 4),
            'light_filter': round(float(light.sum() / mask.sum()), 4),
            'brass_like': round(float(brass.sum() / mask.sum()), 4)},
        'rust_color_shift': {
            'mean_rgb_distance': round(float(diff[mask].mean()), 3),
            'p50': round(float(np.percentile(diff[mask], 50)), 3),
            'p90': round(float(np.percentile(diff[mask], 90)), 3),
            'brown_fraction': round(float(brown.sum() / mask.sum()), 4)},
        'configured': {
            'overall_height_m': round(oh, 4),
            'overall_width_m': round(ow, 4),
            'overall_height_over_width': round(oh / ow, 4),
            'mounting_plate_width_over_overall': round(
                DIMENSIONS['plate']['width'] / ow, 4),
            'cage_diameter_over_overall': round(
                2 * DIMENSIONS['cage']['r_top'] / ow, 4),
            'source_height_over_width_px': round(
                (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]), 4),
        },
        'note': ('工作尺度按原照比例设置，不是实测；minAreaRect 描述照片可见轮廓，'
                 '包含少量透视与轮廓抗锯齿。'),
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
    old_result = None
    if paths['old_result'].is_file():
        data = json.loads(paths['old_result'].read_text(encoding='utf8'))
        old_result = {
            'mesh_sha256': data.get('mesh_sha256'),
            'status': data.get('status'),
            'code': data.get('code'),
            'fidelity': data.get('reconstruction', {}).get('fidelity'),
            'source_camera': data.get('reconstruction', {}).get('source_camera'),
            'library_repair': data.get('library_repair'),
        }
    return {
        'mesh': str(paths['old_mesh'].relative_to(ROOT)),
        'mesh_sha256': digest(paths['old_mesh']),
        'vertices': int(len(merged.vertices)), 'faces': int(len(merged.faces)),
        'bounds': np.round(merged.bounds, 4).tolist(),
        'components': len(components),
        'watertight': bool(merged.is_watertight),
        'audit': audit,
        'library_result': old_result,
        'note': ('旧模型是单张原貌图经 TripoSR 得到的单体壳；审计记录源图轮廓 IoU '
                 '0.9678、填孔 1，未通过。六方向图显示窗口为浅槽、碗体不透明、'
                 '支架与排污钮糊化，不能作为本件结构精度或材质依据。'),
    }


def opening_checks(worker):
    rays = worker.get('ray_checks', {})
    bracket = rays.get('bracket', {})
    cap = rays.get('cap', {})
    neck = rays.get('neck', {})
    cage = rays.get('cage', {})
    filt = rays.get('filter', {})
    collar = rays.get('collar', {})
    bowl = rays.get('bowl', {})
    drain = rays.get('drain', {})
    slots = rays.get('slots', {})
    objects = rays.get('objects', {})
    windows = cage.get('windows', {})
    checks = {
        'bracket_hole_0': bool(bracket.get('hole_0_clear')),
        'bracket_hole_1': bool(bracket.get('hole_1_clear')),
        'bracket_plate_solid': bool(bracket.get('plate_solid')),
        'bracket_fillets': bool(bracket.get('fillet_l_present') and
                                bracket.get('fillet_r_present')),
        'cap_solid': bool(cap.get('solid')),
        'cap_bead': bool(cap.get('bead_present')),
        'neck_core_solid': bool(neck.get('core_solid')),
        'neck_ribs': bool(neck.get('front_rib_solid') and
                          neck.get('back_rib_present')),
        'cage_front_windows_open': bool(cage.get('all_front_windows_open')),
        'cage_front_post_solid': bool(cage.get('front_post_solid')),
        'cage_side_post_solid': bool(cage.get('side_post_solid')),
        'cage_top_band_solid': bool(cage.get('top_band_solid')),
        'cage_mid_band_solid': bool(cage.get('mid_band_solid')),
        'filter_visible_in_windows': bool(
            windows and all(v.get('filter_hit') for v in windows.values())),
        'filter_core_solid': bool(filt.get('core_solid')),
        'collar_solid': bool(collar.get('solid')),
        'collar_seam': bool(collar.get('seam_present')),
        'bowl_wall': bool(bowl.get('wall_hit')),
        'bowl_ribs': bool(bowl.get('ribs_present')),
        'drain_neck_solid': bool(drain.get('neck_solid')),
        'drain_knob_solid': bool(drain.get('knob_solid')),
        'brass_solid': bool(drain.get('brass_solid')),
        'brass_blind_recess': bool(drain.get('brass_blind_recess')),
        'slot_insets': bool(slots.get('present')),
        'objects_partial': bool(objects.get('bracket') and objects.get('cap') and
                                objects.get('filter') and objects.get('collar') and
                                objects.get('bowl') and objects.get('brass')),
    }
    checks['passed'] = all(checks.values())
    checks['windows'] = windows
    checks['brass_recess_depth_m'] = drain.get('brass_recess_depth_m')
    checks['note'] = ('射线检查覆盖两个安装通孔、六扇前侧窗口、滤网可见性、'
                      '护罩立柱/横带、环座、杯壁、排污钮与黄铜浅凹槽；'
                      '不证明背面、孔深、内部螺纹或装配关系。')
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
        '# 37 集油器 · 分部件重建候选（manual_only，待审）',
        '',
        f"- 版本：`{VERSION}`",
        '- 原照：黑色罐式部件，顶部矩形安装板（双长圆通孔）、带筋颈部、圆形端盖、'
        '黑色护罩（4 立柱 × 3 排窗口，白色编织滤网）、宽环座、烟灰色半透明集污杯'
        '（内部竖筋）、滚花排污钮和黄铜排污芯。',
        '- 六视图角色：`missing`（映射表无可靠对应参考）；只按原照可见结构重建，'
        '隐藏面为推断。',
        '- 旧模型：TripoSR 单体壳，窗口为浅槽、碗体不透明、支架/排污钮糊化，未复用；'
        '仅渲染对照。',
        '',
        '## 已确认与推断',
        '',
        '已确认（原照可见）：安装板轮廓与两个长圆通孔、带筋颈部、圆形端盖与顶缘、'
        '护罩的三排窗口、前侧中央窄柱与两侧宽柱、窗口后的白色编织滤网、'
        '上部两处浅凹槽、宽环座、烟灰半透明杯与两条可见内筋、滚花排污钮与黄铜芯。',
        '',
        '推断/未确认：背面窗口与立柱布局、四根内筋的完整分布、安装板厚度与孔深、'
        '颈部背面构造、护罩与环座的内部卡扣、滤网材料与过滤精度、'
        '杯体壁厚与密封方式、排污钮内部阀结构、黄铜芯凹槽深度、'
        '真实尺寸/型号/材料牌号、落稳姿态与现场尺度。',
        '',
        '## 实际执行的检查',
        '',
        f"- 安装板双通孔：{'通过' if openings['bracket_hole_0'] and openings['bracket_hole_1'] else '未通过'}；"
        f"板体与圆角过渡：{'通过' if openings['bracket_plate_solid'] and openings['bracket_fillets'] else '未通过'}。",
        f"- 端盖/颈部：端盖 {'通过' if openings['cap_solid'] else '未通过'}，"
        f"顶缘压边 {'通过' if openings['cap_bead'] else '未通过'}，"
        f"颈部与筋 {'通过' if openings['neck_core_solid'] and openings['neck_ribs'] else '未通过'}。",
        f"- 护罩窗口：六扇前侧窗口通透 "
        f"{'通过' if openings['cage_front_windows_open'] else '未通过'}；"
        f"滤网在窗口后可见 {'通过' if openings['filter_visible_in_windows'] else '未通过'}；"
        f"立柱/横带实体 {'通过' if openings['cage_front_post_solid'] and openings['cage_side_post_solid'] and openings['cage_top_band_solid'] and openings['cage_mid_band_solid'] else '未通过'}。",
        f"- 环座/杯体：环座 {'通过' if openings['collar_solid'] else '未通过'}，"
        f"杯壁 {'通过' if openings['bowl_wall'] else '未通过'}，"
        f"内筋 {'通过' if openings['bowl_ribs'] else '未通过'}。",
        f"- 排污组件：颈 {'通过' if openings['drain_neck_solid'] else '未通过'}，"
        f"滚花钮 {'通过' if openings['drain_knob_solid'] else '未通过'}，"
        f"黄铜芯 {'通过' if openings['brass_solid'] else '未通过'}，"
        f"浅凹槽 {'通过' if openings['brass_blind_recess'] else '未通过'} "
        f"（深度约 {openings.get('brass_recess_depth_m')} m）。",
        f"- 部件网格闭合：{checks['mesh']['closed_parts']}/"
        f"{checks['mesh']['parts']}；同部件自交："
        f"{len(checks['mesh']['self_intersecting'])} 项。",
        f"- 贴图数值检查：{'通过' if checks['textures']['passed'] else '未通过'}。",
        f"- 黄铜高光变光：{'通过' if checks['metal']['passed'] else '未通过'}；"
        f"质心位移 {checks['metal'].get('highlight_shift_px')} px。",
        f"- 原照轮廓诊断（未标定，非验收）：严格同相机 IoU {strict_text}，"
        f"包围盒对齐 IoU {(diagnostics.get('source_box_fit') or {}).get('iou', '未执行')}，"
        f"主轴旋转后正视 IoU "
        f"{(diagnostics.get('source_rotated_front_fit') or {}).get('iou', '未执行')}，"
        f"24 方向最优 IoU {diagnostics['best'].get('iou', '未执行')}。",
        '- 六视图参考 IoU：未执行（映射为 missing，无可靠参考）。',
        '- 浏览器运行时交互：未执行（本件生成独立离线页，使用 Blender 渲染帧）。',
        '',
        '## 待审',
        '',
    ]
    lines += [f'- {item}' for item in result['pending_review']]
    lines += ['', '## 文件', '',
              f"- GLB：`{out / 'mesh.glb'}`",
              f"- Blender：`{out / 'model.blend'}`",
              f"- 检查页：`{out / 'review-37.html'}`",
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
        ('安装板双通孔', ('通过' if o['bracket_hole_0'] and o['bracket_hole_1']
                       else '未通过') + ' · 2/2'),
        ('护罩六扇前侧窗口', ('通过' if o['cage_front_windows_open'] else '未通过')
         + ' · 6/6'),
        ('窗口后滤网可见', '通过' if o['filter_visible_in_windows'] else '未通过'),
        ('滤网实体', '通过' if o['filter_core_solid'] else '未通过'),
        ('环座 / 杯壁 / 内筋',
         f"{'通过' if o['collar_solid'] else '未通过'} · "
         f"{'通过' if o['bowl_wall'] else '未通过'} · "
         f"{'通过' if o['bowl_ribs'] else '未通过'}"),
        ('排污钮 / 黄铜芯',
         f"{'通过' if o['drain_knob_solid'] else '未通过'} · "
         f"{'通过' if o['brass_solid'] else '未通过'}"),
        ('黄铜浅凹槽', f"{'通过' if o['brass_blind_recess'] else '未通过'} · "
         f"深约 {o.get('brass_recess_depth_m')} m"),
        ('网格闭合部件', f"{result['checks']['mesh']['closed_parts']}/"
         f"{result['checks']['mesh']['parts']}"),
        ('同部件自交', f"{len(result['checks']['mesh']['self_intersecting'])} 项"),
        ('PBR 贴图数值', '通过' if result['checks']['textures']['passed'] else '未通过'),
        ('黄铜变光高光位移',
         f"{result['checks']['metal'].get('highlight_shift_px')} px · "
         f"{'通过' if result['checks']['metal'].get('passed') else '未通过'}"),
        ('原照轮廓诊断（未标定）',
         f"严格同相机 {strict_text} · 包围盒 "
         f"{(d.get('source_box_fit') or {}).get('iou', '未执行')} · 主轴正视 "
         f"{(d.get('source_rotated_front_fit') or {}).get('iou', '未执行')}"),
        ('24 方向最优轮廓诊断', f"{d['best'].get('iou', '未执行')} "
         f"({d.get('best', {}).get('view', '未执行')})"),
        ('六视图参考', 'missing（无可靠对应参考，未做参考 IoU）'),
        ('旧模型库审计', 'IoU 0.9678，填孔 1；窗口为浅槽、碗体不透明、局部糊化'),
        ('浏览器运行时检查', '未执行（离线页仅用渲染帧）'),
    ]
    metric_rows = ''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in metrics)
    gallery = [
        ('六视图新旧对照', '六视图-新旧对照.jpg'),
        ('关键局部新旧对照', '关键局部-新旧对照.jpg'),
        ('结构与开口隔离', '结构开口-隔离检查.jpg'),
        ('材质/灰模', '材质-同光照对照.jpg'),
        ('变光与黄铜反射', '表面反射-变光对照.jpg'),
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
<title>37 集油器 · 分部件重建检查（manual_only）</title>
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
<header><h1>37 集油器 · 分部件重建 {VERSION} <span class="tag">manual_only · 待审</span></h1>
<p>黑色罐式油气过滤器 · 安装板/带筋颈/护罩滤网/半透明杯/黄铜排污芯分部件重建 · 离线检查页</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b>本候选不替换原模型，不进入自动放置或正式合成。
六视图映射中本件为 <code>missing</code>，没有可靠对应参考；只按原照可见结构重建，
背面、内部构造、孔深、真实尺寸、落稳与现场光照均待人工复核。所有 IoU 只作诊断，不是验收通过。
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
<p class="notice" style="margin-top:12px">未执行项明确写为“未执行”，不能解读为通过。
黄铜变光只证明黄铜阀芯的高光移动；黑色塑料、编织滤网和烟灰色半透明杯不适用金属反射检查。
射线检查只覆盖实际建出的开口和零件，不证明隐藏面或装配关系。</p></section>

<section><h2>待人工复核</h2><ul>{pending}</ul></section>

<section><h2>文件与记录</h2><p><a href="result.json">result.json</a> ·
<a href="hashes.json">hashes.json</a> · <a href="worker-result.json">Blender worker 记录</a> ·
<a href="report.md">中文报告</a> · <a href="source-measurements.json">原照测量</a> ·
<a href="old-model-audit.json">旧模型审计</a> · <a href="texture-checks.json">贴图检查</a></p>
<p>贴图：<a href="textures/black_plastic_base.png">黑塑料底色</a> ·
<a href="textures/filter_mesh_base.png">滤网底色</a> ·
<a href="textures/smoke_bowl_base.png">半透明杯底色</a> ·
<a href="textures/brass_mr.png">黄铜金属/粗糙度</a></p>
<p>候选 <code>{VERSION}</code>，参考角色 <code>missing</code>；原模型和共享页面未被修改。</p></section>
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
    (out / 'review-37.html').write_text(html, encoding='utf8')


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
        'next_number': 38,
    }
    target = ROOT / 'work/direct-deepseek/37-progress.json'
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
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--blender', type=Path,
                        default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    parser.add_argument('--skip-checks', action='store_true')
    args = parser.parse_args()
    if args.number != NUMBER:
        parser.error('this entry supports item 37 only')
    if args.draft:
        RENDER.update({'six_samples': 10, 'detail_samples': 16,
                       'turn_samples': 4, 'turn_count': 24})
    if args.out is None:
        args.out = (ROOT / 'work/direct-deepseek/37-draft' if args.draft
                    else ROOT / 'outputs/37集油器-分部件重建-v1')
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
                          'fov_deg': 38.05401, 'shift_x': -0.000224,
                          'shift_y': 0.002531},
        'original_mesh': str(paths['old_mesh']),
        'config': {'dimensions': DIMENSIONS, 'render': RENDER},
        'materials': materials,
        'textures': materials,
        'draft': bool(args.draft),
        'geometry_only': bool(args.geometry_only),
    }
    (out / 'request.json').write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
    worker = HERE / 'mesh_rebuild_37_worker.py'
    cmd = [str(args.blender), '-b', '--factory-startup', '--python', str(worker),
           '--', str(out / 'request.json')]
    print('rendering with Blender ...', flush=True)
    with open(out / 'blender.log', 'w', encoding='utf8') as log:
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), timeout=14400)
    if process.returncode != 0:
        raise RuntimeError(f'Blender worker failed with code {process.returncode}; '
                           f'see {out / "blender.log"}')
    if args.geometry_only:
        debug = json.loads((out / 'geometry-debug.json').read_text(encoding='utf8'))
        print(json.dumps(debug, ensure_ascii=False, indent=2)[:12000])
        return
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
    parts = worker_result.get('parts', [])
    checks = {
        'openings': openings,
        'textures': texture_stats,
        'metal': metal,
        'luma': luma,
        'mesh': {
            'parts': len(parts),
            'closed_parts': worker_result.get('closed_parts', 0),
            'closures': worker_result.get('closures', {}),
            'self_intersecting': worker_result.get('self_intersecting_components', {}),
            'polygons': worker_result.get('polygons'),
            'triangles': worker_result.get('triangles'),
            'bounds': worker_result.get('bounds'),
        },
        'diagnostics': diagnostics,
    }
    pending = [
        '六视图映射为 missing，无可靠对应参考；背面窗口/立柱、环座内部和隐藏面为推断。',
        '安装板厚度、长圆孔的真实边缘倒角与孔深、颈部背面构造未验证。',
        '护罩与环座的内部卡扣、滤网材料/过滤精度、杯体壁厚与密封方式未验证。',
        '排污钮内部阀结构、黄铜芯凹槽深度与螺纹规格未验证。',
        '真实尺寸、型号、材料牌号、落稳姿态、现场尺度和合成环境光未验证。',
        '原照轮廓 IoU 为未标定诊断，不能当作精度或验收结论。',
        '浏览器运行时交互未执行；离线页只使用 Blender 渲染帧。',
    ]
    result = {
        'number': NUMBER,
        'name': '集油器（黑色罐式油气过滤/集油部件）',
        'version': VERSION,
        'status': 'manual_only_pending_review',
        'manual_only': True,
        'replaces_library_mesh': False,
        'enters_automatic_placement': False,
        'reference': {
            'role': 'missing',
            'file': None,
            'sha256': None,
            'note': ('六视图映射中本件为 missing：型号及部件类别未确认，'
                     '没有可靠对应参考。未使用相似型号冒充。'),
        },
        'source': {'original': str(paths['original'].relative_to(ROOT)),
                   'rust': str(paths['rust'].relative_to(ROOT)),
                   'original_sha256': digest(paths['original']),
                   'rust_sha256': digest(paths['rust'])},
        'method': ('按原照可见结构分部件重建：安装板双通孔、带筋颈部、端盖、'
                   '4 立柱 × 3 排窗口护罩与白色编织滤网、宽环座、'
                   '烟灰半透明集污杯与内筋、滚花排污钮和黄铜排污芯；'
                   '程序化 PBR 贴图与离线检查页。'),
        'confirmed': [
            '矩形安装板与两个长圆通孔', '带筋颈部', '圆形端盖与顶缘',
            '护罩三排窗口与中央窄柱/两侧宽柱', '窗口后白色编织滤网',
            '上部浅凹槽', '宽环座', '烟灰色半透明集污杯与两条可见内筋',
            '滚花排污钮与黄铜排污芯',
        ],
        'inferred': pending,
        'mesh': {
            'parts': len(parts),
            'closed_parts': worker_result.get('closed_parts', 0),
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
            'page': str(out / 'review-37.html'),
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
    checks_path = HERE / 'checks_rebuild_37.py'
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
                               'see checks-37.json')
    print('done')


if __name__ == '__main__':
    main()
