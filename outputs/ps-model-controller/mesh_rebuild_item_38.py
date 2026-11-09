"""Build the review-only horizontal red pressure-tank candidate (item 38).

The original clean/rust photograph is the primary appearance and proportion
source.  The user-supplied six-view collage is an uncalibrated visual-match
product render of the same horizontal red pressure-tank family; it is used for
hidden/back-side topology and as a diagnostic comparison, never as measured
geometry.  The candidate is a part-based rebuild (shell, dished heads, weld
seams, folded supports with real foot holes, top regulator, safety valve, right
head ports, bottom drain, recreated label, brass/steel/black PBR materials).  It
never replaces the library mesh, stays manual-only and never enters automatic
placement or final compositing.  A new output directory is required.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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
REFERENCE_REL = 'references/parallel-six-view/红色储气罐六视图产品图.png'
REFERENCE_SHA256 = 'ef190d7fd88127a05d482d27edba11f426f1cd758e2d8552b149ec0a4bd9114e'
REFERENCE_NAME = '红色储气罐六视图产品图.png'
SOURCE_SIZE = (1254, 1254)
TAU = math.tau
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
ARIAL_BOLD = 'C:/Windows/Fonts/arialbd.ttf'
ARIAL = 'C:/Windows/Fonts/arial.ttf'


def localize(path_text):
    """Map an old main-work-tree absolute path to this work tree."""
    parts = Path(path_text).parts
    for anchor in ('outputs', 'work'):
        if anchor in parts:
            index = parts.index(anchor)
            return ROOT.joinpath(*parts[index:])
    return Path(path_text)


def find_reference():
    path = ROOT / REFERENCE_REL
    if path.is_file() and digest(path) == REFERENCE_SHA256:
        return path
    raise FileNotFoundError('registered six-view reference not found in this work tree')


# ---------------------------------------------------------------------------
# PBR textures
# ---------------------------------------------------------------------------
def make_paint_atlas(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('38 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((160, 470, 320, 630), (800, 600, 940, 740),
                (600, 600, 740, 740), (1000, 600, 1140, 740))]
    rng = np.random.default_rng(38038)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [(profile['red']['clean'], 3.0),
             (profile['red']['weathered'], 3.6),
             (profile['red']['stained'], 2.8),
             (profile['red']['rusted'], 2.2)]
    rust_rgb = np.array((116, 76, 52), np.float32)
    dust_rgb = np.array((78, 72, 66), np.float32)
    scratch_rgb = np.array((198, 190, 182), np.float32)
    for i, (mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((110, 110), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((820, 820), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Enamel orange-peel field; the source highlight is removed by
        # _tile_detail, so no fixed light direction is baked into the colour.
        peel = cv2.GaussianBlur(rng.normal(size=(size // 4, size // 4)).astype(np.float32),
                                (0, 0), 1.2)
        peel = cv2.resize(peel, (size, size), interpolation=cv2.INTER_CUBIC)
        peel *= brush_strength / max(1e-6, float(peel.std()))
        field = .50 * (coarse - .5) + .30 * (fine - .5) + .20 * (speck - .5)
        amount = mat['rust_amount']
        patch = np.clip((coarse - (1 - amount * 2.2)) * 24, 0, 1) if amount > 0 else 0
        rust = patch * np.clip((fine - .15) * 1.35, 0, 1) if amount > 0 else 0
        rust = np.clip(rust + (np.clip((speck - .992) * 70, 0, 1) * amount), 0, 1)
        dust = np.clip((speck - .9980) * 45, 0, 1) * .035
        color = np.asarray(mat['base_rgb'], np.float32)
        rgb = color[None, None, :] * (1 + field[:, :, None] * .050)
        rgb += detail[:, :, None] * .070 + peel[:, :, None] * .45
        rgb += ((speck - .5) * .20)[:, :, None]
        scratches = np.clip((peel - 2.7) * .040, 0, 1) * (1 - rust) * (i >= 1)
        rgb = rgb * (1 - scratches[:, :, None] * .28) + \
            scratch_rgb[None, None, :] * (scratches[:, :, None] * .28)
        rust_local = rust_rgb[None, None, :] * (.78 + .42 * fine[:, :, None])
        rgb = rgb * (1 - rust[:, :, None]) + rust_local * rust[:, :, None]
        rgb = rgb * (1 - dust[:, :, None] * .45) + \
            dust_rgb[None, None, :] * (dust[:, :, None] * .45)
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        roughness = (mat['roughness'] + rust * .22 + dust * .09
                     + field * .030 + peel * .005)
        metallic = np.zeros_like(roughness)
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip(roughness * 255, 0, 255),
                           np.clip(metallic * 255, 0, 255)),
                          axis=2).astype(np.uint8)
        bump_source = peel + detail * .30
        bump = bump_source - cv2.GaussianBlur(bump_source, (0, 0), 7.0)
        bump = cv2.GaussianBlur(bump, (0, 0), .9) / 255
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 10
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 10
        nx = np.clip(-dx * .85, -.11, .11)
        ny = np.clip(dy * .85, -.11, .11)
        nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                             0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('paint_base', base), ('paint_mr', mr),
                      ('paint_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_metal_texture(out, name, base_rgb, roughness, metallic, seed,
                       patina=False, pebble=False):
    size = 512
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.random((48, 48), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((size, size), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    brush = cv2.resize(rng.normal(size=(96, size)).astype(np.float32),
                       (size, size), interpolation=cv2.INTER_CUBIC)
    brush = cv2.GaussianBlur(brush, (0, 0), .8)
    brush *= 3.0 / max(1e-6, float(brush.std()))
    color = np.asarray(base_rgb, np.float32)
    rgb = color[None, None, :] * (1 + (coarse[:, :, None] - .5) * .06
                                        + (fine[:, :, None] - .5) * .025
                                        + brush[:, :, None] * .012)
    if patina:
        spot = np.clip((coarse - .78) * 4.0, 0, 1) * np.clip((fine - .25) * 1.4, 0, 1)
        dark = np.array((78, 66, 40), np.float32)
        rgb = rgb * (1 - spot[:, :, None] * .35) + dark[None, None, :] * (spot[:, :, None] * .35)
    if pebble:
        pebble_field = cv2.GaussianBlur(rng.normal(size=(size // 3, size // 3)).astype(np.float32),
                                        (0, 0), 1.0)
        pebble_field = cv2.resize(pebble_field, (size, size), interpolation=cv2.INTER_CUBIC)
        pebble_field *= 1.0 / max(1e-6, float(pebble_field.std()))
        rgb += pebble_field[:, :, None] * 2.2
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(roughness + (coarse - .5) * .08
                    + (fine - .5) * .025 + brush * .010, .16, .72)
    metal = np.full((size, size), metallic, np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metal * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur(brush * .55 + (fine - .5) * .7, (0, 0), .8)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 12
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 12
    nx = np.clip(-dx * .50, -.05, .05)
    ny = np.clip(dy * .50, -.05, .05)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for suffix, arr in (('_base', rgb), ('_mr', mr), ('_normal', normal)):
        path = out / (name + suffix + '.png')
        Image.fromarray(arr).save(path)
        paths[name + suffix] = str(path)
    return paths


def make_label(out):
    width, height = 1024, 512
    image = Image.new('RGB', (width, height), (246, 205, 24))
    draw = ImageDraw.Draw(image)
    dark = (24, 24, 24)
    yellow = (246, 205, 24)
    draw.rectangle((0, 0, width - 1, height - 1), outline=dark, width=8)
    draw.rectangle((18, 18, width - 19, height - 19), outline=dark, width=3)
    font_ce = ImageFont.truetype(ARIAL_BOLD, 112)
    font_head = ImageFont.truetype(ARIAL_BOLD, 54)
    font_body = ImageFont.truetype(ARIAL, 30)
    font_small = ImageFont.truetype(ARIAL, 24)
    draw.text((44, 34), 'CE', font=font_ce, fill=dark)
    draw.text((46, 166), 'PRESSURE', font=font_small, fill=dark)
    draw.text((46, 196), 'EQUIPMENT', font=font_small, fill=dark)
    draw.text((286, 38), 'AIR RECEIVER', font=font_head, fill=dark)
    draw.text((288, 103), 'HORIZONTAL PRESSURE TANK', font=font_small, fill=dark)
    rows = (('MAX WORKING PRESSURE', '8 bar'),
            ('TEST PRESSURE', '12 bar'),
            ('VOLUME', '50 L'),
            ('TEMPERATURE', '-10 / +100 C'),
            ('SERIAL', 'SN 0000'))
    y = 160
    for label, value in rows:
        draw.text((286, y), label, font=font_body, fill=dark)
        draw.text((710, y), value, font=font_body, fill=dark)
        y += 48
    # QR-like block (generic; not a real certificate or serial code).
    rng = np.random.default_rng(38039)
    qx, qy, cell, n = 60, 300, 8, 15
    draw.rectangle((qx - 5, qy - 5, qx + cell * n + 5, qy + cell * n + 5),
                   fill=(255, 255, 255), outline=dark, width=2)
    for iy in range(n):
        for ix in range(n):
            if rng.random() < .46:
                draw.rectangle((qx + ix * cell, qy + iy * cell,
                                qx + (ix + 1) * cell - 1,
                                qy + (iy + 1) * cell - 1), fill=dark)
    draw.text((286, 430), 'Recreated label - original print, serial and certification are not reproduced',
              font=font_small, fill=(70, 60, 30))
    base = np.asarray(image)
    mr = np.zeros((height, width, 3), np.uint8)
    mr[..., 0] = 255
    mr[..., 1] = int(.48 * 255)
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
def paint_texture_checks(base_path, mr_path, normal_path):
    base = np.asarray(Image.open(base_path).convert('RGB')).astype(np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB')).astype(np.float32) / 255
    normal = np.asarray(Image.open(normal_path).convert('RGB')).astype(np.float32)
    tiles = {'clean': np.s_[0:1024, 0:1024],
             'weathered': np.s_[0:1024, 1024:2048],
             'stained': np.s_[1024:2048, 0:1024],
             'rusted': np.s_[1024:2048, 1024:2048]}
    values = {}
    for name, sl in tiles.items():
        rgb = base[sl]
        means = rgb.mean(axis=(0, 1))
        rust = ((rgb[..., 0] > 90) & (rgb[..., 1] < rgb[..., 0] * .72) &
                (rgb[..., 2] < rgb[..., 1] * .95) &
                (rgb[..., 0] - rgb[..., 1] > 25)).mean()
        values[name] = {
            'base_rgb_mean': [round(float(v), 1) for v in means],
            'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
            'metallic_mean': round(float(mr[sl][..., 2].mean()), 4),
            'chroma': round(float(means.max() - means.min()), 2),
            'rust_fraction': round(float(rust), 4),
            'normal_std': round(float(normal[sl].std()), 3)}
    c, w, s, r = (values['clean'], values['weathered'],
                  values['stained'], values['rusted'])
    values['passed'] = bool(
        c['metallic_mean'] < .10 and .22 < c['roughness_mean'] < .45 and
        c['chroma'] > 90 and c['rust_fraction'] < .05 and c['normal_std'] > 1.0 and
        w['metallic_mean'] < .10 and w['chroma'] > 80 and w['rust_fraction'] < .12 and
        s['metallic_mean'] < .10 and .33 < s['roughness_mean'] < .62 and
        r['metallic_mean'] < .10 and r['roughness_mean'] > .50 and
        r['rust_fraction'] > .10)
    values['note'] = ('Painted steel/enamel must stay dielectric (metallic near zero); '
                      'rust changes roughness and colour, not metalness. Highlights are '
                      'live lighting, not baked into the base colour.')
    return values


def metal_texture_checks(base_paths):
    values = {}
    for name, path in base_paths.items():
        base = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
        mr = np.asarray(Image.open(path.replace('_base', '_mr')).convert('RGB')).astype(np.float32) / 255
        means = base.mean(axis=(0, 1))
        values[name] = {
            'base_rgb_mean': [round(float(v), 1) for v in means],
            'roughness_mean': round(float(mr[..., 1].mean()), 4),
            'metallic_mean': round(float(mr[..., 2].mean()), 4),
            'chroma': round(float(means.max() - means.min()), 2),
            'luma': round(float(means.mean()), 2)}
    b, s, k = values['brass'], values['steel'], values['black']
    values['passed'] = bool(
        b['metallic_mean'] > .85 and .18 < b['roughness_mean'] < .48 and
        b['chroma'] > 40 and b['luma'] > 90 and
        s['metallic_mean'] > .85 and .18 < s['roughness_mean'] < .48 and
        s['chroma'] < 20 and s['luma'] > 110 and
        k['metallic_mean'] < .10 and .40 < k['roughness_mean'] < .70 and
        k['luma'] < 80)
    values['note'] = ('Brass and steel are metallic; black knob/handle rubber is '
                      'dielectric. Metal highlights must move with live lighting.')
    return values


def label_texture_checks(base_path):
    rgb = np.asarray(Image.open(base_path).convert('RGB')).astype(np.float32)
    yellow = ((rgb[..., 0] > 180) & (rgb[..., 1] > 150) & (rgb[..., 2] < 110)).mean()
    mr = np.asarray(Image.open(str(base_path).replace('_base', '_mr')).convert('RGB')).astype(np.float32) / 255
    return {'passed': bool(yellow > .25 and mr[..., 2].mean() < .10 and
                           rgb.mean() > 100),
            'yellow_fraction': round(float(yellow), 4),
            'metallic_mean': round(float(mr[..., 2].mean()), 4),
            'note': ('Recreated yellow equipment label; the exact original text, '
                     'serial, certification and QR payload are not reproduced.')}


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 6, y0 + 6, x0 + w // 3 - 6, y0 + h // 2 - 34))


def reference_mask(panel):
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    mask = (np.min(rgb, axis=2) < 238).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .93):] = 0
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
    keep = np.flatnonzero(areas >= max(300, areas.max() * .02)) + 1
    return np.isin(labels, keep).astype(np.uint8)


VIEW_NOTES = {
    'front': ('原始照片与六视图正视均显示黄色标签、左侧调压器、中央安全阀、'
              '右端两只红色六角螺塞、底部排水阀和两组支撑。参考为同类产品渲染，'
              '未标定；比例以原照为准。'),
    'back': ('背面原照不可见；六视图背视给出镜像的调压器、安全阀、右端端口和'
             '两组支撑。背面管口朝向与焊缝细节为推断。'),
    'left': ('左端视可见左封头与调压器立管；六视图左视更像端视产品图。'
             '左封头曲面、支撑背面和隐藏焊缝为对称推断。'),
    'right': ('右端视显示右封头上两只上下排列的红色六角螺塞、底部排水阀和'
              '顶部安全阀投影；端口孔深与内部螺纹为推断。'),
    'top': ('顶视给出调压器（黑色旋钮朝左）、安全阀和四个支撑脚外伸凸耳；'
            '俯视产品图不是标定正投影，法兰/支脚的角度和宽度为估计。'),
    'bottom': ('底视给出两条鞍带、四个支撑脚外伸凸耳和底部排水阀；'
               '排水出口与安装孔的真实孔径/位置为按照片比例推断。'),
}


def make_report_38(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 22)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2240, 2600), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '38 集油器1 · 卧式红色压力罐 · 六视图新旧对照（参考未标定）',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        y = 56 + i * 420
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        ref_mask = reference_mask(panel)
        old_mask = np.asarray(old.getchannel('A')) >= 128
        new_mask = np.asarray(new.getchannel('A')) >= 128
        ref_fit = mask_fit(ref_mask)
        old_fit = mask_fit(old_mask)
        new_fit = mask_fit(new_mask)
        ref_iou = (float(np.logical_and(ref_fit, new_fit).sum() /
                         max(1, np.logical_or(ref_fit, new_fit).sum()))
                   if ref_fit is not None and new_fit is not None else None)
        old_new = (float(np.logical_and(old_fit, new_fit).sum() /
                         max(1, np.logical_or(old_fit, new_fit).sum()))
                   if old_fit is not None and new_fit is not None else None)
        rows.append({'view': name,
                     'reference_uncalibrated_mask_iou': round(ref_iou, 4)
                     if ref_iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4)
                     if old_new is not None else None,
                     'note': ('两张轮廓按最长边等比对齐；参考为同类产品渲染且未标定，'
                              'IoU 只用于发现形体差异。' + VIEW_NOTES[name])})
        draw.text((12, y), f'{name.upper()}   六视图参考（未标定） / 原模型 / 新候选',
                  font=small, fill='#27363c')
        rx, ry, rw, rh = cv2.boundingRect(ref_mask)
        panel = panel.crop((max(0, rx - 6), max(0, ry - 6),
                            min(panel.width, rx + rw + 6),
                            min(panel.height, ry + rh + 6)))
        for j, image in enumerate((panel.convert('RGBA'), old, new)):
            if j:
                box = image.getchannel('A').getbbox()
                if box:
                    image = image.crop(box)
            factor = min(520 / image.width, 350 / image.height)
            image = image.resize((max(1, round(image.width * factor)),
                                  max(1, round(image.height * factor))),
                                 Image.Resampling.LANCZOS)
            tile = Image.new('RGBA', (560, 365), 'white' if j == 0 else '#d9dedd')
            tile.alpha_composite(image, ((560 - image.width) // 2,
                                         (365 - image.height) // 2))
            board.paste(tile.convert('RGB'), (10 + j * 572, y + 26))
        draw.text((12, y + 392),
                  (f'参考未标定诊断 IoU：{ref_iou:.3f} · 原/新轮廓 IoU：{old_new:.3f}'
                   if ref_iou is not None and old_new is not None else '轮廓不可比较'),
                  font=small, fill='#52646b')
    draw.text((20, 2540),
              '重要：六视图是同类红色卧式压力罐的产品渲染，未提供相机标定，不能作为实测几何；'
              '原照为该件的原貌/做锈图，是比例和可见结构的主要依据。原模型通过单张源图轮廓门槛，'
              '但六方向审计显示表面糊化、支脚与封头粘连、调压器和标签细节不清，故未复用旧网格。',
              font=small, fill='#8a2b20')
    board.save(out / '六视图-新旧对照.jpg', quality=95)
    return rows


def make_detail_board(out):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    items = (
        ('detail-regulator', '顶部调压器/过滤器：黑色旋钮、银灰本体、黄铜塞'),
        ('detail-safety', '顶部安全阀：红色焊座与黄铜阀体/滚花帽'),
        ('detail-ports', '右封头两只红色六角螺塞与焊环'),
        ('detail-drain', '底部排水阀：红座、银螺母、黄铜体和黑色旋钮'),
        ('detail-foot-left', '左前支撑脚：三角折板与外伸法兰孔'),
        ('detail-foot-right', '右后支撑脚：法兰孔与鞍带'),
        ('detail-label', '重绘黄色标签（原文与序列号未复制）'),
        ('detail-seam', '环焊缝与筒体/封头过渡'),
        ('detail-top', '顶视：调压器、安全阀与四个支撑凸耳'),
        ('detail-bottom', '底视：鞍带、排水阀与支撑脚孔'),
        ('detail-surface', '红色漆面：光泽、细小橘皮与局部锈斑'),
        ('detail-front', '整体前视（透明背景）'),
        ('detail-left', '左端视（调压器立管投影）'),
        ('detail-right', '右端视（端口与排水阀）'),
    )
    board = Image.new('RGB', (2000, 1700), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '38 集油器1 · 关键局部检查（新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = (i % 7) * 284
        y = 55 + (i // 7) * 800
        path = out / 'renders' / f'{name}.png'
        if not path.is_file():
            continue
        image = Image.open(path).convert('RGBA')
        box = image.getchannel('A').getbbox()
        if box:
            image = image.crop(box)
        image.thumbnail((266, 680), Image.Resampling.LANCZOS)
        tile = Image.new('RGBA', (276, 700), '#d9dedd')
        tile.alpha_composite(image, ((276 - image.width) // 2,
                                     (700 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, 1640),
              '局部图用于人工核对结构；不能证明未观察面、壁厚、内部腔体、螺纹孔深、'
              '装配关系、落稳或现场尺度。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2200, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '38 集油器1 · 隔离结构与开口检查',
              font=font, fill='#18242b')
    items = (
        ('feet-holes-only', '四只支撑脚：真实法兰孔'),
        ('support-only', '两条鞍带 + 四腿 + 四只法兰脚'),
        ('regulator-only', '调压器/过滤器组件'),
        ('safety-only', '安全阀组件'),
        ('ports-only', '右封头两只螺塞/焊环'),
        ('drain-only', '排水阀组件'),
        ('tank-only', '筒体、两封头与环焊缝'),
        ('coupler-bore-only', '调压器快接插座孔（隔离）'),
        ('drain-outlet-only', '排水出口孔（隔离）'),
        ('label-only', '重绘标签曲面'),
    )
    for i, (name, label) in enumerate(items):
        x = (i % 5) * 436
        y = 60 + (i // 5) * 465
        image = Image.open(out / 'renders' / f'{name}.png').convert('RGBA')
        box = image.getchannel('A').getbbox()
        if box:
            image = image.crop(box)
        image.thumbnail((400, 350), Image.Resampling.LANCZOS)
        tile = Image.new('RGBA', (416, 380), '#d9dedd')
        tile.alpha_composite(image, ((416 - image.width) // 2,
                                     (380 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 10, y + 30))
        if name == 'feet-holes-only':
            count = checks['foot_holes']['detected_openings']
            text = f'{label}：{count}/4 ' + ('通过' if count == 4 else '未通过')
            color = '#1d5c3a' if count == 4 else '#8a2b20'
        elif name == 'coupler-bore-only':
            count = checks['coupler_socket']['detected_openings']
            text = f'{label}：{count}/1 ' + ('通过' if count == 1 else '未通过')
            color = '#1d5c3a' if count == 1 else '#8a2b20'
        elif name == 'drain-outlet-only':
            count = checks['drain_outlet']['detected_openings']
            text = f'{label}：{count}/1 ' + ('通过' if count == 1 else '未通过')
            color = '#1d5c3a' if count == 1 else '#8a2b20'
        else:
            text = label
            color = '#52646b'
        draw.text((x + 14, y), text, font=small, fill=color)
    draw.text((20, 960),
              '隔离渲染用于确认支脚孔、快接插座孔和排水出口孔的可见开口；不能证明孔深、'
              '螺纹、内部腔体、零件间真实装配或承压安全。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out, metal_checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1500, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '38 集油器1 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型（单图重建顶点色）'),
                                       ('new-material-close', '新候选 PBR 漆面/金属/橡胶'))):
        x = i * 750
        image = Image.open(out / 'renders' / f'{name}.png').convert('RGBA')
        box = image.getchannel('A').getbbox()
        if box:
            image = image.crop(box)
        image.thumbnail((690, 820), Image.Resampling.LANCZOS)
        tile = Image.new('RGBA', (710, 840), '#d9dedd')
        tile.alpha_composite(image, ((710 - image.width) // 2,
                                     (840 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), '同一中性光与相机；高光来自实时照明，不烘焙在底色中。'
                         '原模型为单图重建的顶点色，不代表旧模型整体通过。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1500, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '38 集油器1 · 变光检查（同视角）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光位置 A'),
                                       ('new-light-alternate', '主光位置 B'))):
        x = i * 750
        image = Image.open(out / 'renders' / f'{name}.png').convert('RGBA')
        box = image.getchannel('A').getbbox()
        if box:
            image = image.crop(box)
        image.thumbnail((690, 700), Image.Resampling.LANCZOS)
        tile = Image.new('RGBA', (710, 720), '#d9dedd')
        tile.alpha_composite(image, ((710 - image.width) // 2,
                                     (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '灯光移动后漆面与金属高光应随视角变化；本图只检查反射响应，'
                         '不证明现场光照方向。', font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1900, 1120), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '38 集油器1 · 金属反射变光对照（隔离黄铜/钢件）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('metal-reg-a', '调压器金属 A：主光 A'),
                                       ('metal-reg-b', '调压器金属 B：主光 B'),
                                       ('metal-drain-a', '排水阀金属 A：主光 A'),
                                       ('metal-drain-b', '排水阀金属 B：主光 B'))):
        x = (i % 2) * 950
        y = 60 + (i // 2) * 520
        image = Image.open(out / 'renders' / f'{name}.png').convert('RGBA')
        box = image.getchannel('A').getbbox()
        if box:
            image = image.crop(box)
        image.thumbnail((840, 440), Image.Resampling.LANCZOS)
        tile = Image.new('RGBA', (900, 480), '#d9dedd')
        tile.alpha_composite(image, ((900 - image.width) // 2,
                                     (480 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, y + 30))
        draw.text((x + 24, y), label, font=small, fill='#27363c')
    draw.text((20, 1075),
              (f'隔离金属件同视角变光：调压器亮度差 {metal_checks["regulator"]["mean_abs_luma_difference"]:.2f}，'
               f'排水阀 {metal_checks["drain"]["mean_abs_luma_difference"]:.2f}。'
               '红色漆面为介电涂层，不参与金属反射验收；金属检查只针对黄铜/钢件。'),
              font=small, fill='#52646b')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    tile = 250
    board = Image.new('RGB', (tile * 6, tile * 4 + 34), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '38 集油器1 · 24 方向转台（透明背景中性光）',
              font=font, fill='#27363c')
    for i in range(24):
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png').convert('RGBA')
        image.thumbnail((tile - 16, tile - 16), Image.Resampling.LANCZOS)
        cell = Image.new('RGBA', (tile, tile), '#d9dedd')
        cell.alpha_composite(image, ((tile - image.width) // 2,
                                     (tile - image.height) // 2))
        board.paste(cell.convert('RGB'), ((i % 6) * tile, 34 + (i // 6) * tile))
    board.save(out / '转台-24方向.jpg', quality=94)


# ---------------------------------------------------------------------------
# Review page and viewer assets
# ---------------------------------------------------------------------------
def make_viewer(out):
    source = ROOT / 'work/direct-deepseek/38-viewer/package'
    viewer = out / 'viewer'
    viewer.mkdir(exist_ok=True)
    copies = (
        ('build/three.module.js', 'three.module.js'),
        ('build/three.core.js', 'three.core.js'),
        ('examples/jsm/loaders/GLTFLoader.js', 'addons/loaders/GLTFLoader.js'),
        ('examples/jsm/controls/OrbitControls.js', 'addons/controls/OrbitControls.js'),
        ('examples/jsm/utils/BufferGeometryUtils.js', 'addons/utils/BufferGeometryUtils.js'),
    )
    available = source.is_dir()
    if available:
        for src, dst in copies:
            target = viewer / dst
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / src, target)
        (viewer / 'README.txt').write_text(
            'three.js 0.180.0 (MIT) local copy for the independent item-38 review page.\n'
            'Files copied from the npm package archive used to build this review viewer.\n',
            encoding='utf8')
    return available


def make_review_page(out, record):
    orig_iou = (record.get('original_source_fidelity_record') or {}).get('iou', 'n/a')
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>38 集油器1 · 三维模型独立检查</title>
<style>
:root{{--bg:#eef1f2;--panel:#fff;--ink:#18242b;--muted:#52646b;--line:#cfd7d8;--accent:#8a2b20;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI","Microsoft YaHei",sans-serif}}
header{{padding:18px 24px;background:#fff;border-bottom:1px solid var(--line)}}
header h1{{margin:0 0 4px;font-size:22px}}
header p{{margin:0;color:var(--muted)}}
main{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(340px,.65fr);gap:18px;padding:18px}}
section.viewer,aside{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
.bar{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}}
button{{border:1px solid var(--line);background:#f7f9f9;color:var(--ink);border-radius:7px;padding:7px 12px;cursor:pointer}}
button.selected{{background:#22424b;color:#fff;border-color:#22424b}}
.canvas-wrap{{position:relative;width:100%;aspect-ratio:1.25;background:#d9dedd;border-radius:8px;overflow:hidden}}
canvas{{width:100%;height:100%;display:block;touch-action:none}}
#loading{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#43545a;background:rgba(217,222,221,.8)}}
.notice{{border-left:4px solid var(--accent);background:#fbf4f1;padding:10px 12px;margin:12px 0;color:#5d2a22;font-size:14px}}
aside h2{{font-size:16px;margin:16px 0 6px}}
aside ul{{padding-left:18px;margin:6px 0}}
aside a{{color:#1f5f78}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;padding:0 18px 24px}}
.gallery figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px}}
.gallery img{{width:100%;height:auto;display:block;border-radius:6px}}
.gallery figcaption{{font-size:13px;color:var(--muted);margin-top:6px}}
code{{background:#eef2f2;padding:1px 4px;border-radius:4px}}
</style>
<script type="importmap">{{"imports":{{"three":"./viewer/three.module.js","three/addons/":"./viewer/addons/"}}}}</script>
</head><body data-number="38" data-version="{record['version']}" data-status="manual_only_pending_review">
<header><div><h1>38 集油器1 · 卧式红色压力罐 · 独立模型检查</h1>
<p>分部件重建候选 · manual_only · 待人工复核 · 不进入自动放置或正式合成</p></div></header>
<main>
<section class="viewer">
<div class="bar"><button id="new" class="selected">新候选</button><button id="old">原始模型</button>
<button id="material" class="selected">材质</button><button id="clay">灰模</button><button id="reset">重置视角</button></div>
<div class="canvas-wrap"><canvas id="canvas"></canvas><div id="loading">正在加载模型…</div></div>
<p class="help">拖动旋转 · 滚轮缩放 · 右键平移。检查页使用实时中性光与程序化反射环境；场景光照需另行验证。</p>
<div class="notice"><b>状态：待审。</b>六视图为未标定的同类红色卧式压力罐产品渲染，原照为原貌/做锈图并作为主要比例依据。
原模型虽通过单张源图轮廓门槛（IoU {orig_iou}），
但六方向审计显示表面糊化、支脚/封头粘连、调压器和标签细节不清，故未复用旧网格。
新候选保留筒体、两封头、两道环焊缝、两组鞍带/四腿/四只带孔法兰脚、顶部调压器与安全阀、右封头两只螺塞、
底部排水阀和重绘标签；黄铜/钢件使用金属 PBR。支撑构造、端口孔深、排水出口、背面/隐藏面和标签原文为推断。
本页仅验证加载、新旧切换、灰模/材质和拖动旋转，不代表模型或场景验收。</div>
</section>
<aside>
<h2>对照与文件</h2>
<ul>
<li><a href="./六视图-新旧对照.jpg">六视图新旧对照</a></li>
<li><a href="./局部检查.jpg">关键局部检查</a></li>
<li><a href="./结构开口-隔离检查.jpg">隔离结构与开口检查</a></li>
<li><a href="./材质-同光照对照.jpg">同光照材质对照</a> · <a href="./材质-变光检查.jpg">变光检查</a></li>
<li><a href="./金属反射-变光对照.jpg">金属反射变光对照</a></li>
<li><a href="./转台-24方向.jpg">24 方向转台</a></li>
<li><a href="./renders/detail-regulator.png">调压器局部</a> · <a href="./renders/detail-drain.png">排水阀局部</a> · <a href="./renders/detail-label.png">标签局部</a></li>
<li><a href="./mesh.glb">下载新候选 GLB</a> · <a href="./model.blend">下载可编辑 Blender</a></li>
<li><a href="./result.json">检查记录 JSON</a> · <a href="./hashes.json">文件哈希</a> · <a href="./浏览器核验.json">浏览器核验</a></li>
</ul>
<h2>检查重点</h2>
<ul><li>筒体圆度、两封头曲面、环焊缝位置与连续性。</li>
<li>四只支撑脚的鞍带/腿/法兰孔是否清晰且不融合。</li>
<li>顶部调压器、安全阀、右封头两螺塞、底部排水阀是否为独立零件。</li>
<li>红色漆面是否保持光泽与局部锈斑，黄铜/钢件高光是否随灯光移动。</li>
<li>标签是否贴合筒体，文字仅为重绘示意，不是原印刷。</li></ul>
<h2>已知待审</h2>
<ul><li>六视图未标定；支撑构造、法兰孔位置/孔径、端口孔深、排水出口为推断。</li>
<li>筒体按封闭实体建模，未表示内部腔体、壁厚、螺纹与承压能力。</li>
<li>背面、隐藏焊缝、装配关系和真实材质厚度未验证。</li>
<li>落稳、现场尺度、场景接触点和环境光融合未验证。</li></ul>
<p id="status" role="status">正在读取模型…</p>
</aside></main>
<section class="gallery">
<figure><img src="./六视图-新旧对照.jpg" alt="六视图对照"><figcaption>六方向参考/原模型/新候选对照。参考未标定，IoU 仅作诊断。</figcaption></figure>
<figure><img src="./局部检查.jpg" alt="局部检查"><figcaption>关键局部：调压器、安全阀、端口、排水阀、支脚、标签、焊缝。</figcaption></figure>
<figure><img src="./结构开口-隔离检查.jpg" alt="隔离开口"><figcaption>隔离结构：四只法兰孔、快接插座孔、排水出口孔；不能证明孔深/螺纹。</figcaption></figure>
<figure><img src="./材质-同光照对照.jpg" alt="材质对照"><figcaption>同光照材质对照；旧模型为单图重建顶点色。</figcaption></figure>
<figure><img src="./金属反射-变光对照.jpg" alt="金属反射"><figcaption>隔离黄铜/钢件在两组主光下的反射变化。</figcaption></figure>
<figure><img src="./转台-24方向.jpg" alt="转台"><figcaption>24 方向转台，用于检查曲面和支脚是否扭曲。</figcaption></figure>
</section>
<p id="selftest-result" hidden></p>
<script type="module">
import * as THREE from 'three';
import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';

const $=id=>document.getElementById(id);
const canvas=$('canvas');
const renderer=new THREE.WebGLRenderer({{canvas,antialias:true,alpha:false,preserveDrawingBuffer:true}});
renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene();scene.background=new THREE.Color('#d9dedd');
const camera=new THREE.PerspectiveCamera(35,1,.01,100);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,.72));
for(const [position,intensity] of [[[4,6,5],1.35],[[-4,3,-4],.65]]){{
  const light=new THREE.DirectionalLight(0xffffff,intensity);light.position.set(...position);scene.add(light);
}}
// Neutral softbox environment, same approach as the shared review page.
const width=512,height=256,pixels=new Uint8Array(width*height*4);
const panels=[[.13,.46,.045,.28,.82],[.39,.43,.055,.34,.92],[.72,.47,.035,.25,.74],[.91,.30,.09,.11,.52]];
for(let y=0;y<height;y++)for(let x=0;x<width;x++){{
  const u=x/width,v=y/height;let value=.30+.10*Math.cos((v-.35)*Math.PI);
  for(const [cx,cy,sx,sy,power] of panels){{const d=Math.abs(u-cx),dx=Math.min(d,1-d)/sx,dy=Math.abs(v-cy)/sy;value+=power*Math.exp(-Math.pow(dx,6)-Math.pow(dy,6))}}
  const channel=Math.round(Math.min(1,value)*255),i=(y*width+x)*4;pixels[i]=pixels[i+1]=pixels[i+2]=channel;pixels[i+3]=255;
}}
const texture=new THREE.DataTexture(pixels,width,height,THREE.RGBAFormat);
texture.colorSpace=THREE.SRGBColorSpace;texture.mapping=THREE.EquirectangularReflectionMapping;texture.needsUpdate=true;
const generator=new THREE.PMREMGenerator(renderer);scene.environment=generator.fromEquirectangular(texture).texture;
texture.dispose();generator.dispose();
const clay=new THREE.MeshStandardMaterial({{color:0xb4b5b0,roughness:.85,metalness:0}});
let current=null,mode='material',loadSerial=0;
function resize(){{const rect=canvas.getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix()}}
function reset(){{camera.position.set(2.5,1.6,3.8);camera.position.normalize().multiplyScalar(3.1);controls.target.set(0,0,0);controls.update()}}
function styleModel(){{if(!current)return;current.traverse(obj=>{{if(obj.isMesh)obj.material=mode==='clay'?clay:obj.userData.originalMaterial}})}}
function mark(ids,id){{for(const name of ids)$(name).classList.toggle('selected',name===id)}}
const frames=()=>new Promise(resolve=>setTimeout(resolve,80));
function capture(){{renderer.render(scene,camera);return canvas.toDataURL()}}
function sampleLuma(){{
  const copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;
  const ctx=copy.getContext('2d');ctx.drawImage(canvas,0,0);
  const data=ctx.getImageData(0,0,copy.width,copy.height).data,background=[data[0],data[1],data[2]],luma=[];
  for(let i=0;i<data.length;i+=4){{
    const difference=Math.abs(data[i]-background[0])+Math.abs(data[i+1]-background[1])+Math.abs(data[i+2]-background[2]);
    if(difference>20)luma.push(.2126*data[i]+.7152*data[i+1]+.0722*data[i+2]);
  }}
  luma.sort((a,b)=>a-b);
  return {{pixels:luma.length,median:luma.length?luma[Math.floor(luma.length/2)]:0,
           dark_fraction:luma.length?luma.filter(v=>v<18).length/luma.length:1}};
}}
async function load(which){{
  const serial=++loadSerial;$('loading').textContent='正在加载模型…';$('loading').style.display='flex';
  try{{
    const path=which==='new'?'./mesh.glb':'./old-mesh.glb';
    const gltf=await new GLTFLoader().loadAsync(path);if(serial!==loadSerial)return false;
    if(current){{scene.remove(current);current.traverse(o=>{{if(o.isMesh)o.geometry.dispose()}})}}
    const model=new THREE.Group();gltf.scene.rotation.x=-Math.PI/2;model.add(gltf.scene);
    model.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(model),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    model.position.sub(center);model.scale.setScalar(1.35/Math.max(size.x,size.y,size.z));model.position.multiplyScalar(model.scale.x);
    model.traverse(obj=>{{if(obj.isMesh)obj.userData.originalMaterial=obj.material}});
    current=model;scene.add(model);styleModel();mark(['old','new'],which);$('loading').style.display='none';
    $('status').textContent=which==='new'?'新候选：分部件重建，manual_only，待人工复核。':'原始模型：单图重建，结构待审。';
    window.reviewState={{which,mode,id:which==='new'?'38-rebuild-v1':'38-old'}};
    return true;
  }}catch(error){{$('loading').textContent='加载失败：'+error.message;$('status').textContent='模型加载失败';throw error}}
}}
for(const which of ['new','old'])$(which).onclick=()=>load(which).catch(()=>{{}});
for(const m of ['clay','material'])$(m).onclick=()=>{{mode=m;styleModel();mark(['clay','material'],m);if(window.reviewState)window.reviewState.mode=m}};
$('reset').onclick=reset;window.addEventListener('resize',resize);resize();reset();
function animate(){{requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}}animate();
async function selftest(){{
  document.body.dataset.selftest='running';document.body.dataset.step='start';const result={{}};
  try{{
    await load('new');await frames();document.body.dataset.step='new1';const newImage=capture();result.new_loaded=!!current;
    await load('old');await frames();document.body.dataset.step='old';const oldImage=capture();result.old_loaded=!!current;
    result.old_new_differ=newImage!==oldImage;
    await load('new');document.body.dataset.step='new2';mode='clay';styleModel();mark(['clay','material'],'clay');await frames();const clayImage=capture();
    mode='material';styleModel();mark(['clay','material'],'material');await frames();const materialImage=capture();
    result.clay_material_differ=clayImage!==materialImage;document.body.dataset.step='modes';
    const before=capture();camera.position.set(2.0,1.8,-3.2);controls.target.set(0,0,0);controls.update();renderer.render(scene,camera);await frames();const after=capture();
    result.pointer_rotates=before!==after;
    result.preview_luma=sampleLuma();document.body.dataset.step='sample';
    result.errors=[];
    const passed=result.new_loaded&&result.old_loaded&&result.old_new_differ&&result.clay_material_differ&&
      result.pointer_rotates&&result.preview_luma.pixels>100&&result.preview_luma.median>18&&result.preview_luma.dark_fraction<.8;
    result.passed=passed;
    document.body.dataset.selftest=passed?'passed':'failed';
    $('selftest-result').hidden=false;$('selftest-result').textContent=JSON.stringify(result);
  }}catch(error){{
    result.errors=[String(error&&error.message||error)];result.passed=false;
    document.body.dataset.selftest='failed';
    $('selftest-result').hidden=false;$('selftest-result').textContent=JSON.stringify(result);
  }}
}}
window.addEventListener('error',e=>{{document.body.dataset.selftest='failed';document.body.dataset.step='error';const p=document.getElementById('selftest-result');p.hidden=false;p.textContent=JSON.stringify({{error:String(e.message),stack:e.error&&e.error.stack}});}});
window.addEventListener('unhandledrejection',e=>{{document.body.dataset.selftest='failed';document.body.dataset.step='rejection';const p=document.getElementById('selftest-result');p.hidden=false;p.textContent=JSON.stringify({{error:String(e.reason&&e.reason.message||e.reason)}});}});
if(new URLSearchParams(location.search).get('selftest')==='1'){{selftest()}}else{{load('new').catch(()=>{{}})}}
</script>
</body></html>
"""
    (out / 'review.html').write_text(html, encoding='utf8')


# ---------------------------------------------------------------------------
# Helpers for diagnostics
# ---------------------------------------------------------------------------
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


def image_luma_difference(first_path, second_path):
    with Image.open(first_path) as first_image, Image.open(second_path) as second_image:
        first = np.asarray(first_image.convert('RGBA'))
        second = np.asarray(second_image.convert('RGBA'))
    mask = (first[:, :, 3] >= 128) | (second[:, :, 3] >= 128)
    if not mask.any():
        return 0.0
    first_luma = (first[:, :, :3].astype(float) * np.array([.2126, .7152, .0722])).sum(axis=2)
    second_luma = (second[:, :, :3].astype(float) * np.array([.2126, .7152, .0722])).sum(axis=2)
    return float(np.abs(first_luma - second_luma)[mask].mean())


def isolated_opening_checks(out):
    checks = {}
    for key, filename, expected in (
            ('foot_holes', 'feet-holes-only.png', 4),
            ('coupler_socket', 'coupler-bore-only.png', 1),
            ('drain_outlet', 'drain-outlet-only.png', 1)):
        with Image.open(out / 'renders' / filename) as image:
            _, count = interior_holes(np.asarray(image.getchannel('A')))
        checks[key] = {'detected_openings': int(count), 'expected': expected,
                       'passed': bool(count == expected),
                       'file': str(out / 'renders' / filename),
                       'method': ('Enclosed background regions in an isolated alpha '
                                  'render; it does not prove bore depth, thread, '
                                  'wall thickness or internal cavities.')}
    return checks


def structural_checks(worker):
    names = [part['name'] for part in worker['parts']]
    return {
        'tank_shell_parts': sum(n in ('Tank cylinder', 'Tank head left',
                                      'Tank head right') for n in names),
        'weld_seam_rings': sum(n.startswith('Tank weld seam') for n in names),
        'support_straps': sum(n.endswith(' strap') for n in names),
        'support_legs': sum(n.endswith(' leg') for n in names),
        'support_feet': sum(n.endswith(' foot') for n in names),
        'support_foot_holes': sum(n.endswith(' foot') for n in names),
        'regulator_parts': sum(n.startswith('Regulator') for n in names),
        'safety_parts': sum(n.startswith('Safety') for n in names),
        'port_parts': sum(n.startswith('Port ') for n in names),
        'drain_parts': sum(n.startswith('Drain') for n in names),
        'weld_beads': sum(n.startswith('Weld bead') for n in names),
        'labels': sum(n == 'Nameplate' for n in names),
        'note': ('Part presence proves only that a feature was modelled, not its '
                 'size, fit or hidden faces. Support construction, port depth, '
                 'drain outlet and internal cavity are inferred.'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=38)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    p.add_argument('--browser', action='store_true', default=True,
                   help='attempt the local headless-browser check')
    p.add_argument('--no-browser', dest='browser', action='store_false')
    a = p.parse_args()
    if a.number != 38:
        p.error('this entry supports item 38 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/38-collector-draft1' if a.draft
                        else 'outputs/38集油器1-六视图重建-v1')
    a.out, a.library, a.blender = (x.resolve() for x in (a.out, a.library, a.blender))
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = next(r for r in (json.loads(line) for line in
                           a.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == a.number)
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
    with Image.open(alpha_path) as alpha_image:
        alpha_array = np.asarray(alpha_image.convert('L'))
    ys, xs = np.nonzero(alpha_array >= 128)
    if not len(xs):
        raise ValueError('source alpha is empty')
    source_alpha_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    config_path = HERE / 'collector38_config.json'
    worker_path = HERE / 'mesh_rebuild_38_worker.py'
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
            required = [a.out / 'model.blend', a.out / 'paint_base.png',
                        a.out / 'paint_mr.png', a.out / 'paint_normal.png',
                        a.out / 'brass_base.png', a.out / 'steel_base.png',
                        a.out / 'black_base.png', a.out / 'label_base.png',
                        a.out / '六视图-新旧对照.jpg', a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '转台-24方向.jpg',
                             a.out / 'review.html', a.out / 'old-mesh.glb']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'detail-{name}.png' for name in
                             ('regulator', 'safety', 'ports', 'drain', 'foot-left',
                              'foot-right', 'label', 'seam', 'top', 'bottom', 'surface')]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('feet-holes-only', 'coupler-bore-only',
                              'drain-outlet-only', 'support-only', 'regulator-only',
                              'safety-only', 'ports-only', 'drain-only', 'tank-only',
                              'label-only', 'new-light-alternate',
                              'new-metal-alternate', 'metal-reg-a', 'metal-reg-b',
                              'metal-drain-a', 'metal-drain-b')]
                required += [a.out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_paint_atlas(Path(row['rust']), a.out, config['material_profile'])
    textures.update(make_metal_texture(a.out, 'brass', [181, 134, 58], .30, 1.0, 38040,
                                       patina=True))
    textures.update(make_metal_texture(a.out, 'steel', [186, 191, 194], .30, 1.0, 38041))
    textures.update(make_metal_texture(a.out, 'black', [28, 28, 30], .55, 0.0, 38042,
                                       pebble=True))
    textures.update(make_label(a.out))
    shutil.copyfile(Path(row['mesh']), a.out / 'old-mesh.glb')
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
    views = make_report_38(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(38)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    opened = None if a.draft else isolated_opening_checks(a.out)
    paint_checks = paint_texture_checks(a.out / 'paint_base.png',
                                        a.out / 'paint_mr.png',
                                        a.out / 'paint_normal.png')
    metal_checks = metal_texture_checks({
        'brass': str(a.out / 'brass_base.png'),
        'steel': str(a.out / 'steel_base.png'),
        'black': str(a.out / 'black_base.png')})
    label_checks = label_texture_checks(a.out / 'label_base.png')
    reflection = {'status': 'not_run'}
    if not a.draft:
        make_detail_board(a.out)
        make_isolated_board(a.out, opened)
        reg_diff = image_luma_difference(a.out / 'renders' / 'metal-reg-a.png',
                                         a.out / 'renders' / 'metal-reg-b.png')
        drain_diff = image_luma_difference(a.out / 'renders' / 'metal-drain-a.png',
                                           a.out / 'renders' / 'metal-drain-b.png')
        reflection = {
            'status': 'passed' if min(reg_diff, drain_diff) > 2.0 else 'failed',
            'regulator': {'mean_abs_luma_difference': round(reg_diff, 4),
                          'changed': bool(reg_diff > 2.0)},
            'drain': {'mean_abs_luma_difference': round(drain_diff, 4),
                      'changed': bool(drain_diff > 2.0)},
            'note': ('Same camera, two key-light directions, isolated brass/steel '
                     'parts only. The red paint is a dielectric coating and is not '
                     'part of this metal-reflection check.')}
        make_material_boards(a.out, reflection)
        make_turntable(a.out)
        make_review_page(a.out, {**{'version': config['version']},
                                 'original_source_fidelity_record':
                                     (audit or {}).get('source_fidelity', {})})
        make_viewer(a.out)
    structural = structural_checks(worker)
    checks = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 40 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(opened and all(v['passed'] for v in opened.values())),
        'structural_counts': bool(structural['tank_shell_parts'] == 3 and
                                  structural['weld_seam_rings'] == 2 and
                                  structural['support_straps'] == 2 and
                                  structural['support_legs'] == 4 and
                                  structural['support_feet'] == 4 and
                                  structural['regulator_parts'] == 12 and
                                  structural['safety_parts'] == 4 and
                                  structural['port_parts'] == 4 and
                                  structural['drain_parts'] == 8 and
                                  structural['weld_beads'] == 5 and
                                  structural['labels'] == 1),
        'material_tiles': bool(paint_checks['passed'] and metal_checks['passed'] and
                               label_checks['passed']),
        'metal_reflection': (reflection.get('status') if not a.draft else 'not_run'),
        'reference_diagnostic_only': True,
        'browser_page': False,
        'source_fidelity': bool(fidelity.get('passed', False)),
    }
    if a.draft:
        checks['isolated_openings'] = 'not_run'
        checks['metal_reflection'] = 'not_run'
        checks['material_tiles'] = 'not_run'
    browser = {'status': 'not_run', 'reason': 'not attempted'}
    if a.browser and not a.draft:
        try:
            from checks_review_browser_38 import run as run_browser
            browser = run_browser(a.out)
        except Exception as error:
            browser = {'status': 'not_run',
                       'reason': f'browser harness unavailable: {error}'}
        checks['browser_page'] = bool(browser.get('status') == 'passed')
    report = {
        'version': config['version'], 'number': 38, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_visual_match_reference',
        'reference_role': 'visual_match_unverified',
        'method': ('part_rebuild_horizontal_red_pressure_tank_with_uncalibrated_'
                   'six_view_visual_match_reference'),
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_clean_sha256': row['clean_sha256'],
        'original_rust_sha256': row['rust_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9589 only measures one silhouette. '
            'The six-view audit shows a smoothed/fused shell, indistinct supports and '
            'feet, merged regulator/safety fittings and an unreadable label; the old '
            'mesh is therefore not reused for the candidate base geometry.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'layout': {'tank_axis': 'Y', 'front_camera': '+X', 'left_camera': '-Y',
                   'right_camera': '+Y', 'up': '+Z'},
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'old_mesh_copy': str(a.out / 'old-mesh.glb'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'paint_texture_checks': paint_checks,
        'metal_texture_checks': metal_checks,
        'label_texture_checks': label_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The strict source-camera render is a diagnostic only: the six-view '
            'collage is uncalibrated and the original photo is a product view. '
            'It is never used as an acceptance gate for this item.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is an uncalibrated visual-match product '
                       'render of the same horizontal red pressure-tank family; it '
                       'is not a calibrated image set and not the same physical unit. '
                       'Mask IoU is a diagnostic of outline only.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': opened,
        'structural_checks': structural, 'checks': checks,
        'browser_check': browser,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view collage is an uncalibrated visual-match product render; the original photo is the primary shape evidence and the panels are not overlaid as measured geometry.',
            'All sizes follow photo pixel ratios and plausible industrial pressure-tank proportions; they are not measured dimensions.',
            'The tank shell is modelled as closed solids; internal cavity, wall thickness, threaded ports and pressure capacity are not represented.',
            'The folded support construction (saddle strap, two legs per side, flanged feet) is inferred from the visible front/top/bottom/side panels; the exact sheet thickness, bend radii and hole positions are not measured.',
            'The right-head port plugs are modelled as closed painted hex caps; bore depth, thread and whether they are plugs or valves are inferred.',
            'The drain side fittings are modelled as closed brass hex nipples; the bottom outlet bore is inferred and partly hidden by the black handle.',
            'The regulator/filter internals, safety-valve internals, coupler locking mechanism and drain-valve internals are not represented.',
            'The yellow label is a recreated generic label; the original text, serial, certification marks and QR payload are not reproduced.',
            'The red paint is dielectric; metal-reflection checks apply only to the isolated brass/steel parts, not to the painted shell.',
            'Assemblies intentionally overlap at the weld junctions, bosses and brackets; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked; browser review only covers loading, old/new switching, clay/material and rotation.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the support brackets, foot holes, ports, drain and top fittings.',
            'Confirm the folded-support construction, flange hole positions/diameters and weld layout from a sample or measured drawing.',
            'Confirm the right-head port type/depth, the bottom drain outlet and the regulator/coupler port orientation.',
            'Confirm the original label text, serial, certification marks and QR payload from a readable sample.',
            'Confirm the paint colour/gloss, rust distribution and the brass/steel/black material assignments from the physical object.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 39,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_38.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 38, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'material_board': str(a.out / '材质-同光照对照.jpg'),
        'metal_board': str(a.out / '金属反射-变光对照.jpg'),
        'turntable_board': str(a.out / '转台-24方向.jpg'),
        'review_page': str(a.out / 'review.html'),
        'checks': checks,
        'paint_checks': paint_checks, 'metal_checks': metal_checks,
        'label_checks': label_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': opened,
        'structural_checks': structural,
        'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'browser_check': browser,
        'reference_role': 'visual_match_unverified',
        'pending_review': report['pending_review'],
        'next_number': 39,
        'manual_only': True,
        'notes': ('Original photo is primary; user six-view collage is an uncalibrated '
                  'visual-match product render. The old model and other items were not '
                  'modified. Candidate is manual_only and pending review. Browser check '
                  'status is recorded honestly; visual acceptance is pending.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '38-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': checks,
                      'source_fidelity': fidelity, 'views': views,
                      'browser': browser.get('status')}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        progress_dir = ROOT / 'work/direct-deepseek'
        progress_dir.mkdir(parents=True, exist_ok=True)
        failed = {
            'number': 38,
            'status': 'incomplete',
            'out': str((ROOT / 'outputs/38集油器1-六视图重建-v1').resolve()),
            'mesh': 'incomplete',
            'blend': 'incomplete',
            'checks': {'error': f'{type(error).__name__}: {error}'},
            'pending_review': ['candidate generation failed; see checks.error and blender.log'],
            'next_number': 39,
        }
        (progress_dir / '38-progress.json').write_text(
            json.dumps(failed, ensure_ascii=False, indent=2), encoding='utf8')
        raise
