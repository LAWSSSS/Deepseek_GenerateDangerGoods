"""Build the review-only sealant-can candidate (item 13).

The original clean/rust photograph shows two upright used white plastic cans.
The user-supplied six-view sheet is an uncalibrated category reference of one
clean can of the same white round-can family: it is used for lid, body and
bottom topology only and is never treated as measured or identical-unit
geometry.  The candidate is a part-based rebuild of two separate cans (no
fused body), never replaces the library mesh, stays manual-only and never enters
automatic placement.  A new output directory is required.
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
from scipy.spatial import cKDTree

from scene_v2 import digest
from mesh_checks import source_fidelity
from mesh_carve import interior_holes
from mesh_rebuild_item import _tile_detail, mask_fit, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE_SHA256 = '75dbfb93b46dd9479e7510f506d31b3566d92bcffc01986ba1c2fb30065a99cf'
REFERENCE_NAME = '白色塑料圆罐六视图产品展示.png'
SOURCE_SIZE = (1283, 1226)
LABEL_FONT = 'C:/Windows/Fonts/msyh.ttc'
LABEL_FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'


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
def make_white_atlas(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('13 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((250, 600, 390, 740), (850, 500, 990, 640),
                (150, 850, 290, 990), (980, 700, 1120, 840))]
    rng = np.random.default_rng(13131)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [(profile['white']['clean'], 2.2),
             (profile['white']['weathered'], 2.8),
             (profile['white']['stained'], 1.9),
             (profile['white']['worn'], 2.1)]
    grime_rgb = np.array((128, 104, 78), np.float32)
    scratch_rgb = np.array((238, 235, 228), np.float32)
    for i, (mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((112, 112), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((860, 860), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Moulded plastic has a faint orange-peel field; scratches are added on
        # the worn tile only.
        peel = cv2.GaussianBlur(rng.normal(size=(size // 4, size // 4)).astype(np.float32),
                                (0, 0), 1.1)
        peel = cv2.resize(peel, (size, size), interpolation=cv2.INTER_CUBIC)
        peel *= brush_strength / max(1e-6, float(peel.std()))
        field = .50 * (coarse - .5) + .30 * (fine - .5) + .20 * (speck - .5)
        stain = mat['stain']
        coverage = min(.14, stain * .40)
        stain_field = cv2.resize(rng.random((36, 36), dtype=np.float32), (size, size),
                                 interpolation=cv2.INTER_CUBIC)
        stain_field = cv2.GaussianBlur(stain_field, (0, 0), 2.0)
        if coverage > 1e-6:
            threshold = float(np.quantile(stain_field, 1 - coverage))
            grime = np.clip((stain_field - threshold) * 30, 0, 1)
            grime *= np.clip((fine - .20) * 1.20, 0, 1)
            grime = cv2.GaussianBlur(grime.astype(np.float32), (0, 0), 2.5)
        else:
            grime = np.zeros_like(coarse)
        dust = np.zeros_like(coarse)
        color = np.asarray(mat['base_rgb'], np.float32)
        rgb = color[None, None, :] * (1 + field[:, :, None] * .016)
        rgb += detail[:, :, None] * .020 + peel[:, :, None] * .12
        rgb += ((speck - .5) * .04)[:, :, None]
        scratches = np.clip((peel - 2.5) * .035, 0, 1) * (i == 3)
        rgb = rgb * (1 - scratches[:, :, None] * .30) + \
            scratch_rgb[None, None, :] * (scratches[:, :, None] * .30)
        rgb = rgb * (1 - grime[:, :, None] * .52) + \
            grime_rgb[None, None, :] * (grime[:, :, None] * .52)
        rgb = rgb * (1 - dust[:, :, None] * .40) + \
            grime_rgb[None, None, :] * (dust[:, :, None] * .40)
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        roughness = (mat['roughness'] + grime * .11 + dust * .06
                     + field * .025 + peel * .003)
        metallic = np.zeros_like(roughness)
        mr[sl] = np.stack((np.full_like(roughness, 255),
                           np.clip(roughness * 255, 0, 255),
                           np.clip(metallic * 255, 0, 255)),
                          axis=2).astype(np.uint8)
        bump_source = peel + detail * .25
        bump = bump_source - cv2.GaussianBlur(bump_source, (0, 0), 7.0)
        bump = cv2.GaussianBlur(bump, (0, 0), .9) / 255
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 11
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 11
        nx = np.clip(-dx * .55, -.06, .06)
        ny = np.clip(dy * .55, -.06, .06)
        nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                             0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('white_base', base), ('white_mr', mr),
                      ('white_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_mark_texture(out, profile):
    size = 512
    rng = np.random.default_rng(13132)
    coarse = cv2.resize(rng.random((48, 48), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((size, size), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_NEAREST)
    base_rgb = np.asarray(profile['mark']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + (coarse[:, :, None] - .5) * .10
                                          + (fine[:, :, None] - .5) * .035)
    rough = np.clip(profile['mark']['roughness'] + (coarse - .5) * .10
                    + (fine - .5) * .03, .35, .80)
    metallic = np.zeros((size, size), np.float32)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur((fine - .5) * .9 + (coarse - .5) * .6, (0, 0), .7)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 13
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 13
    nx = np.clip(-dx * .35, -.03, .03)
    ny = np.clip(dy * .35, -.03, .03)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('mark_base', np.clip(rgb, 0, 255).astype(np.uint8)),
                      ('mark_mr', mr), ('mark_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


# ---------------------------------------------------------------------------
# Texture checks
# ---------------------------------------------------------------------------
def _tile_region(arr, i, size=1024):
    yy, xx = divmod(i, 2)
    return arr[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]


def white_texture_checks(base_path, mr_path, normal_path):
    base = np.asarray(Image.open(base_path).convert('RGB')).astype(np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB')).astype(np.float32)
    normal = np.asarray(Image.open(normal_path).convert('RGB')).astype(np.float32)
    tiles = []
    expected = (190.0, 160.0, 135.0, 175.0)
    for i in range(4):
        tile = _tile_region(base, i)
        luma = float(tile.mean())
        chroma = float((tile.max(axis=2) - tile.min(axis=2)).mean())
        metal = float(_tile_region(mr, i)[:, :, 2].mean() / 255)
        rough = float(_tile_region(mr, i)[:, :, 1].mean() / 255)
        normal_std = float(_tile_region(normal, i).std())
        tiles.append({'tile': i, 'mean_luma': round(luma, 2),
                      'mean_chroma': round(chroma, 2),
                      'mean_metallic': round(metal, 4),
                      'mean_roughness': round(rough, 4),
                      'normal_std': round(normal_std, 3),
                      'luma_min_expected': expected[i],
                      'passed': bool(luma >= expected[i] and chroma <= 35 and
                                     metal <= .06 and .30 <= rough <= .78 and
                                     normal_std >= 1.0)})
    passed = all(t['passed'] for t in tiles)
    return {'passed': passed, 'tiles': tiles,
            'note': ('White plastic must stay dielectric (metallic near zero) with a '
                     'mid roughness range; brown grime changes colour, not metalness.')}


def mark_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB')).astype(np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB')).astype(np.float32)
    luma = float(base.mean())
    chroma = float((base.max(axis=2) - base.min(axis=2)).mean())
    metal = float(mr[:, :, 2].mean() / 255)
    rough = float(mr[:, :, 1].mean() / 255)
    return {'passed': bool(60 <= luma <= 180 and chroma <= 22 and metal <= .06
                           and .40 <= rough <= .80),
            'mean_luma': round(luma, 2), 'mean_chroma': round(chroma, 2),
            'mean_metallic': round(metal, 4), 'mean_roughness': round(rough, 4),
            'note': 'Dark grey mould-mark plastic; colour is inferred from the bottom panel.'}


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


def minimum_vertex_gap(parts, left_prefix, right_prefix):
    left = np.concatenate([parts[name].vertices for name in parts
                           if name.startswith(left_prefix) and
                           name.endswith(' body')])
    right = np.concatenate([parts[name].vertices for name in parts
                            if name.startswith(right_prefix) and
                            name.endswith(' body')])
    distance, _ = cKDTree(right).query(left)
    return float(distance.min())


def two_can_components(alpha):
    solid = (np.asarray(alpha) >= 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    if count < 2:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= max(300, areas.max() * .02)
    return int(keep.sum())


def source_framed_silhouettes(out):
    checks = {}
    for name in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        with Image.open(out / 'renders' / f'single-{name}.png') as image:
            holes, count = interior_holes(np.asarray(image.getchannel('A')))
        checks[name] = int(count)
    with Image.open(out / 'renders' / 'two-cans-top.png') as image:
        components = two_can_components(image.getchannel('A'))
    return {'single_can_silhouette_holes': checks,
            'single_can_expected_holes': 0,
            'single_can_passed': bool(all(v == 0 for v in checks.values())),
            'two_can_top_components': components,
            'two_can_top_expected_components': 2,
            'two_can_top_passed': bool(components == 2),
            'method': ('Enclosed holes in the isolated single-can silhouettes and '
                       'separate silhouettes in the top view; this does not prove '
                       'wall thickness, lid fit or hidden internal cavities.')}


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------
def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 10, y0 + 8, x0 + w // 3 - 10, y0 + h // 2 - 8))


def reference_mask(panel):
    """Approximate single-can foreground mask for diagnostics only."""
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = rgb.mean(axis=2)
    mask = (gray < 248).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .90):] = 0
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
    'front': ('参考正视是单只干净罐；原照为两只使用过的罐，前罐偏左且更近。'
              '本件以原照决定两只罐的相对位置，参考只用于罐盖/罐身拓扑。'),
    'back': ('背面在原照不可见；参考背视给出罐盖外缘与罐身轮廓。'
             '盖内配合、罐壁厚度与隐藏面为旋转对称推断。'),
    'left': ('左视参考显示侧面的提手式卡扣；原照前罐左侧也能看到同类凸起，'
             '候选把前罐卡扣转向 -Y 使其以侧面轮廓出现。'),
    'right': ('右视为参考的镜像视图；原照右侧主要由后罐遮挡，'
              '后罐卡扣朝接近 +X 的方向。'),
    'top': ('顶视参考给出外缘凸边、浅凹盖面和中央盖板；'
            '两只罐的间距按原照投影估计。'),
    'bottom': ('底视参考给出外圈、中央浇口和回收标志；'
               '原照底部只能看到局部小凸脚，标志细节与深度为推断。'),
}


def make_report_13(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype(LABEL_FONT, 22)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (2240, 2580), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '13 密封胶罐 · 两只白色塑料罐 · 六视图类别参考（未标定）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = 0
        y = 56 + i * 415
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        single = Image.open(out / 'renders' / f'single-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask(panel))
        actual = mask_fit(np.asarray(single.getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.getchannel('A')) >= 128)
        new_fit = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        ref_iou = (float(np.logical_and(expected, actual).sum() /
                         max(1, np.logical_or(expected, actual).sum()))
                   if expected is not None and actual is not None else None)
        old_new = (float(np.logical_and(old_fit, new_fit).sum() /
                         max(1, np.logical_or(old_fit, new_fit).sum()))
                   if old_fit is not None and new_fit is not None else None)
        rows.append({'view': name,
                     'reference_single_can_mask_iou': round(ref_iou, 4)
                     if ref_iou is not None else None,
                     'old_new_two_can_silhouette_iou': round(old_new, 4)
                     if old_new is not None else None,
                     'note': ('按最长边等比对齐；参考为单只罐类别图且未标定，'
                              '单罐 IoU 只用于发现罐形/盖形差异。' + VIEW_NOTES[name])})
        draw.text((12, y), f'{name.upper()}   类别参考（单罐） / 原模型（两罐） / 新候选（两罐） / 新候选单罐',
                  font=small, fill='#27363c')
        ref_mask = reference_mask(panel)
        rx, ry, rw, rh = cv2.boundingRect(ref_mask)
        panel = panel.crop((max(0, rx - 5), max(0, ry - 5),
                            min(panel.width, rx + rw + 5),
                            min(panel.height, ry + rh + 5)))
        for j, image in enumerate((panel.convert('RGBA'), old, new, single)):
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
                  (f'参考单罐诊断 IoU：{ref_iou:.3f} · 原/新两罐轮廓 IoU：{old_new:.3f}'
                   if ref_iou is not None and old_new is not None else '轮廓不可比较'),
                  font=small, fill='#52646b')
    draw.text((20, 2520),
              '重要：参考图是单只干净罐的类别图且未标定，原照是两只使用过的罐；'
              '两者数量不同，参考轮廓不能直接与两罐轮廓比较。原模型虽通过单张源图轮廓门槛，'
              '但六方向审计可见表面纹理粗糙、两罐粘连、盖缝与卡扣不清，故未复用旧网格。',
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
        ('detail-front', '两只罐整体前视（透明背景）'),
        ('detail-top', '两只罐顶视：独立盖面与外缘'),
        ('detail-lid-top', '后罐盖顶：外缘凸边与浅凹面'),
        ('detail-lid-recess', '后罐盖面浅凹与中央盖板'),
        ('detail-lid-rim', '盖沿圆角与外缘'),
        ('detail-seam-latch', '后罐盖缝与双段式卡扣'),
        ('detail-front-latch', '前罐卡扣（侧面轮廓，-Y 方向）'),
        ('detail-can-gap', '两罐之间的间隙（推断间距）'),
        ('detail-bottom', '后罐底面：外圈与两只小凸脚'),
        ('detail-bottom-mark', '底面中央浇口与回收式标志'),
        ('detail-body', '罐身锥度与下部圆角'),
        ('detail-surface', '白色塑料表面、污渍与划痕'),
        ('single-front', '单罐诊断前视（类别参考）'),
        ('single-left', '单罐左侧视（卡扣轮廓）'),
        ('body-only', '隔离罐身（两只）'),
        ('lid-only', '隔离罐盖、盖缝与卡扣'),
    )
    board = Image.new('RGB', (2000, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '13 密封胶罐 · 关键局部检查（新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = (i % 8) * 250
        y = 55 + (i // 8) * 780
        path = out / 'renders' / f'{name}.png'
        image = _tile_image(path, (232, 660))
        tile = Image.new('RGBA', (242, 675), '#d9dedd')
        tile.alpha_composite(image, ((242 - image.width) // 2,
                                     (675 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, 1600),
              '这些局部图用于人工检查结构；不能证明未观察面、壁厚、盖内配合、'
              '卡扣机构、标志文字、落稳或现场尺度。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype(LABEL_FONT, 20)
    small = ImageFont.truetype(LABEL_FONT, 16)
    board = Image.new('RGB', (1800, 900), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '13 密封胶罐 · 隔离结构与开口检查',
              font=font, fill='#18242b')
    items = (('body-only', '隔离罐身：两只独立实体'),
             ('lid-only', '隔离罐盖/盖缝/卡扣'),
             ('latch-only', '卡扣附件：每罐双段块'),
             ('bottom-parts-only', '底脚/浇口/标志'),
             ('two-cans-top', '顶视两罐：应分离'),
             ('single-front', '单罐轮廓：封闭孔检测'))
    for i, (name, label) in enumerate(items):
        x = i * 298
        image = _tile_image(out / 'renders' / f'{name}.png', (270, 600))
        tile = Image.new('RGBA', (282, 630), '#d9dedd')
        tile.alpha_composite(image, ((282 - image.width) // 2,
                                     (630 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 8, 80))
        if name == 'two-cans-top':
            text = (f'{label}：{checks["two_can_top_components"]}/'
                    f'{checks["two_can_top_expected_components"]} '
                    f'{"通过" if checks["two_can_top_passed"] else "未通过"}')
            color = '#1d5c3a' if checks['two_can_top_passed'] else '#8a2b20'
        elif name == 'single-front':
            count = checks['single_can_silhouette_holes']['front']
            text = (f'{label}：{count}/0 '
                    f'{"通过" if count == 0 else "未通过"}')
            color = '#1d5c3a' if count == 0 else '#8a2b20'
        else:
            text = f'{label}'
            color = '#52646b'
        draw.text((x + 10, 44), text, font=small, fill=color)
    draw.text((20, 805),
              '透明背景隔离渲染。物体本身没有功能通孔；两罐顶视分离用于确认未把两只罐'
              '融合成一个网格。孔数/轮廓不能证明壁厚、盖内配合、卡扣机构或隐藏细节。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype(LABEL_FONT, 21)
    small = ImageFont.truetype(LABEL_FONT, 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '13 密封胶罐 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选白色塑料 PBR（污渍与划痕）'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), '#d9dedd')
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
    draw.text((20, 12), '13 密封胶罐 · 变光检查（同视角）',
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
    draw.text((20, 810), '灯光移动后塑料高光和明暗应随视角变化；本图只检查反射响应，'
                         '不证明现场光照方向。', font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '13 密封胶罐 · 表面反射变光对照（全塑料介质，金属反射不适用）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光位置 A'),
                                       ('new-surface-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 760))
        tile = Image.new('RGBA', (680, 780), '#d9dedd')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (780 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 55))
        draw.text((x + 24, 28), label, font=small, fill='#27363c')
    draw.text((20, 820), '本件没有可确认的金属零件；图中检查的是介质塑料在两组主光下的'
                         '高光移动，不能作为金属反射验收。', font=small, fill='#52646b')
    board.save(out / '表面反射-变光对照.jpg', quality=96)


def make_turntable(out):
    font = ImageFont.truetype(LABEL_FONT, 18)
    tile = 250
    board = Image.new('RGB', (tile * 6, tile * 4 + 34), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), '13 密封胶罐 · 24 方向转台（灰色为透明背景上的中性光渲染）',
              font=font, fill='#27363c')
    for i in range(24):
        image = Image.open(out / 'renders' / f'turn-{i:02d}.png').convert('RGBA')
        image.thumbnail((tile - 16, tile - 16), Image.Resampling.LANCZOS)
        cell = Image.new('RGBA', (tile, tile), '#d9dedd')
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=13)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 13:
        p.error('this entry supports item 13 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/13-sealant-draft1' if a.draft
                        else 'outputs/13密封胶罐-六视图重建-v1')
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
    config_path = HERE / 'sealant13_config.json'
    worker_path = HERE / 'mesh_rebuild_13_worker.py'
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
            required = [a.out / 'model.blend', a.out / 'white_base.png',
                        a.out / 'white_mr.png', a.out / 'white_normal.png',
                        a.out / 'mark_base.png', a.out / 'mark_mr.png',
                        a.out / 'mark_normal.png', a.out / '六视图-新旧对照.jpg',
                        a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '表面反射-变光对照.jpg', a.out / '转台-24方向.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'single-{view}.png' for view in VIEWS]
                required += [a.out / 'renders' / f'detail-{name}.png' for name in
                             ('front', 'top', 'lid-top', 'lid-rim', 'seam-latch',
                              'front-latch', 'bottom', 'bottom-mark', 'can-gap',
                              'body', 'surface', 'lid-recess')]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('body-only', 'lid-only', 'latch-only',
                              'bottom-parts-only', 'two-cans-top',
                              'new-light-alternate', 'new-surface-alternate',
                              'old-material-close')]
                required += [a.out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_white_atlas(Path(row['rust']), a.out, config['material_profile'])
    textures.update(make_mark_texture(a.out, config['material_profile']))
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
    views = make_report_13(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(13)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    opened = None if a.draft else source_framed_silhouettes(a.out)
    parts = load_parts(a.out / 'mesh.glb')
    separation = minimum_vertex_gap(parts, 'Can front ', 'Can rear ')
    hole_checks = {}
    aligned_fidelity = None
    aligned_best = None
    if not a.draft:
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
            aligned_fidelity['alignment'] = 'mask bounding box fit to 400 px; not a calibrated camera'
            aligned_fidelity['acceptance_gate'] = False
        make_detail_board(a.out)
        make_material_boards(a.out)
        make_isolated_board(a.out, opened)
        hole_checks = {
            'single_can_silhouettes': {
                'detected_openings': opened['single_can_silhouette_holes'],
                'expected': 0,
                'passed': opened['single_can_passed'],
                'method': opened['method']},
            'two_can_top_separation': {
                'detected_components': opened['two_can_top_components'],
                'expected': 2,
                'passed': opened['two_can_top_passed'],
                'file': str(a.out / 'renders' / 'two-cans-top.png')},
            'front_rear_body_vertex_gap': {
                'metres': round(separation, 6),
                'passed': bool(separation > .005),
                'note': ('Minimum nearest-vertex gap between the two body meshes; '
                         'it shows the two cans are separate, not that every '
                         'attachment is collision-free.')},
        }
    white_checks = white_texture_checks(a.out / 'white_base.png',
                                        a.out / 'white_mr.png',
                                        a.out / 'white_normal.png')
    mark_checks = mark_texture_checks(a.out / 'mark_base.png', a.out / 'mark_mr.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'bodies': sum(name.endswith(' body') for name in names),
        'lids': sum(name.endswith(' lid') for name in names),
        'lid_seam_rings': sum(name.endswith(' lid seam') for name in names),
        'body_seam_rings': sum(name.endswith(' body seam') for name in names),
        'recess_rings': sum(name.endswith(' recess rim') for name in names),
        'latch_uppers': sum(name.endswith(' latch upper') for name in names),
        'latch_lowers': sum(name.endswith(' latch lower') for name in names),
        'bottom_tabs': sum(' bottom tab ' in name for name in names),
        'bottom_gates': sum(name.endswith(' bottom gate') for name in names),
        'bottom_mark_bars': sum(' bottom mark bar ' in name for name in names),
        'bottom_mark_letters': sum(' bottom mark letter ' in name for name in names),
        'latch_attachments_per_can': 2,
        'note': ('Part presence proves only that a feature was modelled, not its '
                 'size, fit or hidden faces. Lid interior fit, latch mechanism, wall '
                 'thickness and bottom-tab function are inferred.'),
    }
    reflection = {'status': 'not_run'}
    if not a.draft:
        first = np.asarray(Image.open(a.out / 'renders/new-material-close.png').convert('RGBA'))
        second = np.asarray(Image.open(a.out / 'renders/new-surface-alternate.png').convert('RGBA'))
        mask = first[:, :, 3] >= 128
        difference = np.abs(first[:, :, :3].astype(float) -
                            second[:, :, :3].astype(float))[mask].mean()
        reflection = {
            'status': 'not_applicable_metal',
            'reason': ('The visible object is white plastic with no confirmed metal '
                       'parts; metallic reflection cannot be evaluated on this item.'),
            'dielectric_response': {
                'mean_abs_luma_difference': round(float(difference), 4),
                'changed': bool(difference > 2.0),
                'note': ('Same camera, different key-light direction; plastic '
                         'highlights must move with live lighting. This is not a '
                         'metal-reflection acceptance.')},
        }
    structural_ok = bool(
        structural['bodies'] == 2 and structural['lids'] == 2 and
        structural['lid_seam_rings'] == 2 and structural['body_seam_rings'] == 2 and
        structural['recess_rings'] == 2 and
        structural['latch_uppers'] == 2 and structural['latch_lowers'] == 2 and
        structural['bottom_tabs'] == 4 and structural['bottom_gates'] == 2 and
        structural['bottom_mark_bars'] == 6 and structural['bottom_mark_letters'] == 4)
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 26 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': structural_ok,
        'material_tiles': bool(white_checks['passed'] and mark_checks['passed']),
        'surface_response': bool(reflection.get('dielectric_response', {}).get('changed', False)),
        'metal_reflection': 'not_applicable_plastic',
        'reference_diagnostic_only': True,
        'browser_page': False,
        'source_fidelity': bool(fidelity.get('passed', False)),
    }
    if a.draft:
        passed['isolated_openings'] = 'not_run'
        passed['surface_response'] = 'not_run'
        passed['material_tiles'] = 'not_run'
    report = {
        'version': config['version'], 'number': 13, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_category_reference',
        'reference_role': 'category_reference_unverified',
        'method': 'parametric_part_rebuild_two_cans_from_source_photo_with_single_can_category_reference',
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'reference_name': REFERENCE_NAME,
        'reference_note': config['proportion_source']['reference_note'],
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_clean_sha256': row['clean_sha256'],
        'original_rust_sha256': row['rust_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old model-library source IoU 0.9775 only measures one silhouette. '
            'The six-view audit shows rough, low-frequency surface texture, fused '
            'cans, indistinct lid seams and lost latches; the old mesh is therefore '
            'not reused for the candidate base geometry.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'layout': config['layout'],
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'white_texture_checks': white_checks,
        'mark_texture_checks': mark_checks,
        'surface_response_check': reflection,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'source_fidelity_aligned_turn_index': aligned_best,
        'source_fidelity_aligned_note': (
            'The best bounding-box-aligned turntable diagnostic is reported '
            'separately. It may numerically meet the old IoU threshold, but the '
            'alignment is not a calibrated camera and it is never used as an '
            'acceptance gate.'),
        'source_diagnostic_note': (
            'The clean photo is a three-quarter view of two cans while the strict '
            'source-camera render looks along +X. The strict score is not a fair pose '
            'comparison. A second best-effort diagnostic scans all 24 turntable views '
            'and aligns each silhouette to the source alpha by bounding box; it is '
            'uncalibrated and is not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is one clean can of the same category, '
                       'not a calibrated image set and not the same physical unit; '
                       'the source photo has two used cans. Single-can mask IoU is a '
                       'diagnostic of can/lid shape only.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'can_separation': {'front_rear_body_vertex_gap_m': round(separation, 6)},
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view sheet is an uncalibrated single-can category reference; the source photo shows two used cans, so panel contours cannot be overlaid on the two-can silhouette.',
            'All sizes follow photo pixel ratios and a plausible industrial sealant-can size; they are not measured dimensions.',
            'The two-can spacing and relative rotations are estimated from the three-quarter photo; the rear can is partly occluded.',
            'The reference panels are clean renders while the source is a used, stained object; the white plastic colour and grime distribution are synthesised from small observed patches.',
            'Lid interior fit, wall thickness, latch mechanism, lid gasket and the function of the bottom tabs are hidden and inferred.',
            'The bottom recycling-style mark is a simplified triangular outline with two letter-like bars; the exact symbol, resin code and engraving depth are not reproduced.',
            'The bodies are modelled as closed solids, so no internal cavity, liquid volume or lid thread is represented.',
            'The item has no confirmed metal parts; metallic reflection is not applicable, and the dielectric highlight check is not a metal acceptance.',
            'Assemblies intentionally overlap at the lid/body and seam rings; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the two-can spacing, lid seam, latch orientation and bottom tabs.',
            'Confirm the front can latch orientation and whether the small side protrusion is the same latch seen in profile.',
            'Confirm lid overhang, rim height, recess depth, wall thickness and lid/body fit from a sample or measured drawing.',
            'Confirm the bottom-tab shape/function and the exact recycling mark and resin code.',
            'Confirm the white plastic surface colour, stain distribution and whether any label or print is missing from the visible faces.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 14,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_13.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 13, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed,
        'white_checks': white_checks, 'mark_checks': mark_checks,
        'surface_response_check': reflection,
        'isolated_openings': hole_checks,
        'can_separation': report['can_separation'],
        'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'source_fidelity_aligned_3q_diagnostic': aligned_fidelity,
        'source_fidelity_aligned_note': (
            'The best bounding-box-aligned turntable diagnostic is reported '
            'separately. It may numerically meet the old IoU threshold, but the '
            'alignment is not a calibrated camera and it is never used as an '
            'acceptance gate.'),
        'reference_role': 'category_reference_unverified',
        'pending_review': report['pending_review'],
        'next_number': 14,
        'manual_only': True,
        'notes': ('The user six-view sheet is an uncalibrated single-can category '
                  'reference; the source photo has two used cans and is the primary '
                  'shape evidence. The old model and items 01-12 were not modified. '
                  'Candidate is manual_only. Browser review page and visual acceptance '
                  'are pending unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '13-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views,
                      'can_separation': report['can_separation']}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
