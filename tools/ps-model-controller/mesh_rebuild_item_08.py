"""Build the review-only separator-filter candidate (item 08).

The shape follows the original rust photograph for proportions and visible
features.  The user-supplied six-view sheet is an uncalibrated visual-match
reference of the same visible design: two opposed four-bolt side flanges, a
top lifting eye and a bottom brass automatic drain.  The sheet is used for the
hidden/back-side topology and as a diagnostic comparison, never as measured
geometry.  The working size follows the model-library visual scale.  The
candidate never replaces the library mesh and is never used by the automatic
placement pipeline.  A new output directory is required.
"""
import argparse
import hashlib
import json
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
from mesh_rebuild_item import _tile_detail, mask_fit, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'
REFERENCE_SHA256 = '52ed638397205f79746f1649f355791e781ffa39062fba968bccf920e1598a18'


def find_reference():
    folder = Path.home() / 'Desktop' / '六视图'
    if folder.is_dir():
        for path in sorted(folder.glob('*.png')):
            if digest(path) == REFERENCE_SHA256:
                return path
    raise FileNotFoundError('registered six-view reference not found')


def make_paint_atlas_08(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != (1229, 1280):
        raise ValueError('08 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((500, 420, 620, 540), (480, 650, 600, 770),
                (560, 250, 680, 370), (600, 760, 720, 880))]
    rng = np.random.default_rng(20808)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    # Four blue-paint tiles: vivid, weathered, rust-dominant and chipped.
    # Rust and scratches come from small observed patches of the supplied
    # rust image; no highlight direction is baked into the base colour.
    specs = [(profile['blue_paint'], 2.4),
             (profile['weathered_blue'], 4.2),
             (profile['rusted_paint'], 0.6),
             (profile['chipped_blue'], 3.0)]
    rust_rgb = np.array((124, 92, 64), np.float32)
    chip_rgb = np.array((150, 152, 148), np.float32)
    for i, (mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((56, 56), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((560, 560), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Vertical brushing: a narrow column field is stretched down the v axis.
        brush = cv2.resize(rng.normal(size=(22, size)).astype(np.float32),
                           (size, size), interpolation=cv2.INTER_CUBIC)
        brush = cv2.GaussianBlur(brush, (0, 0), 2.4)
        brush *= brush_strength / max(1e-6, float(brush.std()))
        field = .45 * (coarse - .5) + .35 * (fine - .5) + .20 * (speck - .5)
        amount = mat['rust_amount']
        # Broad, soft-edged rust blotches with strong colour mixing, plus a
        # weaker vertical run; the observed photo has clustered rust, not a
        # uniform stripe pattern.
        patch = np.clip((coarse - (.90 - amount * .55)) * 18, 0, 1)
        rust = patch * (.55 + .45 * fine)
        streak = np.clip((-brush - 2.2) * .14, 0, 1) * amount * .5
        rust = np.clip(rust + streak, 0, 1)
        chip_threshold = .990 if i < 3 else .984
        chips = np.clip((speck - chip_threshold) * (45 if i < 3 else 28), 0, 1)
        color = np.asarray(mat['base_rgb'], np.float32)
        rgb = color[None, None, :] * (1 + field[:, :, None] * .09)
        rgb += detail[:, :, None] * .15 + brush[:, :, None] * .12
        rgb += ((speck - .5) * .6)[:, :, None]
        # Scratches: thin light lines from the vertical brush high tail.
        scratches = np.clip((brush - 4.0) * .10, 0, 1) * (1 - rust)
        rgb = rgb * (1 - scratches[:, :, None] * .22) + \
            np.array((186, 200, 208), np.float32) * (scratches[:, :, None] * .22)
        rgb = rgb * (1 - rust[:, :, None]) + rust_rgb[None, None, :] * rust[:, :, None]
        rgb = rgb * (1 - chips[:, :, None] * .85) + \
            chip_rgb[None, None, :] * (chips[:, :, None] * .85)
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        roughness = (mat['roughness'] + rust * .24 + chips * .06
                     + field * .05 + brush * .006)
        metallic = mat['metallic'] * (1 - rust * .80) * (1 - chips * .15) + chips * .82
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip(roughness * 255, 0, 255),
                           np.clip(metallic * 255, 0, 255)),
                          axis=2).astype(np.uint8)
        bump = cv2.GaussianBlur(detail + brush * 2.0, (0, 0), .6) / 255
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 8
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 8
        nx = np.clip(-dx * .75, -.11, .11)
        ny = np.clip(dy * .75, -.11, .11)
        nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                             0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('paint_base', base), ('paint_mr', mr), ('paint_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_brass_08(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != (1229, 1280):
        raise ValueError('08 rust source dimensions changed; sampling rects need review')
    detail = _tile_detail(im, (600, 1090, 650, 1180))
    size = 1024
    rng = np.random.default_rng(20809)
    coarse = cv2.resize(rng.random((48, 48), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((480, 480), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    speck = cv2.resize(rng.random((size, size), dtype=np.float32), (size, size),
                       interpolation=cv2.INTER_NEAREST)
    field = .45 * (coarse - .5) + .35 * (fine - .5) + .20 * (speck - .5)
    grime = np.clip((coarse - .72) * 6, 0, 1) * .55
    patina = np.clip((speck - .994) * 60, 0, 1) * .45
    base_rgb = np.asarray(profile['brass']['base_rgb'], np.float32)
    rgb = base_rgb[None, None, :] * (1 + field[:, :, None] * .10)
    rgb += detail[:, :, None] * .22
    rgb = rgb * (1 - grime[:, :, None] * .55) + \
        np.array((82, 70, 48), np.float32) * (grime[:, :, None] * .55)
    rgb = rgb * (1 - patina[:, :, None] * .45) + \
        np.array((72, 116, 92), np.float32) * (patina[:, :, None] * .45)
    base = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(profile['brass']['roughness'] + field * .10
                    + grime * .22 + patina * .16, .15, .80)
    metallic = np.clip(profile['brass']['metallic'] - grime * .18
                       - patina * .10, .55, 1.0)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur(detail + speck * 1.2, (0, 0), .7)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 9
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 9
    nx = np.clip(-dx * .55, -.09, .09)
    ny = np.clip(dy * .55, -.09, .09)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('brass_base', base), ('brass_mr', mr), ('brass_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def paint_texture_checks(path):
    base = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(str(path).replace('paint_base', 'paint_mr'))
                    .convert('RGB'), dtype=np.float32) / 255
    tiles = {'blue_paint': np.s_[0:1024, 0:1024],
             'weathered_blue': np.s_[0:1024, 1024:2048],
             'rusted_paint': np.s_[1024:2048, 0:1024],
             'chipped_blue': np.s_[1024:2048, 1024:2048]}
    values = {}
    for name, sl in tiles.items():
        rgb = base[sl]
        values[name] = {
            'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
            'metallic_mean': round(float(mr[sl][..., 2].mean()), 4),
            'blue_minus_red': round(float(rgb[..., 2].mean() - rgb[..., 0].mean()), 2),
            'red_minus_blue': round(float(rgb[..., 0].mean() - rgb[..., 2].mean()), 2),
            'rust_fraction': round(float(((rgb[..., 0] > rgb[..., 1] + 8) &
                                          (rgb[..., 1] > rgb[..., 2] + 3)).mean()), 4)}
    values['passed'] = bool(
        values['blue_paint']['metallic_mean'] > .40 and
        values['blue_paint']['roughness_mean'] < .60 and
        values['blue_paint']['blue_minus_red'] > 30 and
        values['weathered_blue']['metallic_mean'] > .35 and
        values['weathered_blue']['roughness_mean'] < .65 and
        values['weathered_blue']['blue_minus_red'] > 25 and
        values['rusted_paint']['roughness_mean'] > .60 and
        values['rusted_paint']['metallic_mean'] < .40 and
        values['rusted_paint']['red_minus_blue'] > 10 and
        values['chipped_blue']['metallic_mean'] > .45)
    values['note'] = ('Tile means; reflection direction is not baked into base '
                      'colour. Rust/chips are diagnostically detected by colour.')
    return values


def brass_texture_checks(base_path, mr_path):
    rgb = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    values = {'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'red_minus_blue': round(float(rgb[..., 0].mean() - rgb[..., 2].mean()), 2),
              'green_minus_blue': round(float(rgb[..., 1].mean() - rgb[..., 2].mean()), 2),
              'passed': bool(.20 < mr[..., 1].mean() < .70 and
                             mr[..., 2].mean() > .85 and
                             rgb[..., 0].mean() - rgb[..., 2].mean() > 30)}
    values['note'] = 'Brass drain tiles only; the painted steel is checked separately.'
    return values


def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 10, y0 + 8, x0 + w // 3 - 10, y0 + h // 2 - 8))


def reference_mask_08(panel):
    """Approximate object mask for an uncalibrated reference panel.

    The panels have a white background and a grey rendered shadow.  Blue paint
    and brass are chromatic; the dark bore is low-value.  This segmentation is
    a diagnostic, not a calibrated silhouette measurement.
    """
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = mx - mn
    mask = (((sat > 16) & (mx < 250)) | (mn < 105)).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .92):] = 0
    mask[:6] = 0
    mask[-6:] = 0
    mask[:, :6] = 0
    mask[:, -6:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = binary_fill_holes(mask).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        best, best_distance = None, None
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] < 900:
                continue
            distance = ((centroids[j, 0] - w / 2) ** 2 +
                        (centroids[j, 1] - h / 2) ** 2) ** .5
            if best is None or distance < best_distance:
                best, best_distance = j, distance
        if best is not None:
            mask = (labels == best).astype(np.uint8)
    return mask


VIEW_NOTES = {
    'front': '参考图与照片同为双侧法兰、顶部提环、底部自动排水器的立式分离器；参考未标定，只作外观诊断',
    'back': '背面在原照不可见；按参考图推断为对称的第二侧法兰与背面视窗，未见其他接口',
    'left': '左侧法兰正对相机；参考图与照片均显示 4 个螺栓孔和中心通孔，孔深与内部通道未证实',
    'right': '右侧法兰为镜像推断；法兰面、螺栓孔与焊缝细节仍待实测',
    'top': '顶视为圆顶、锻造提环和两侧法兰；提环顶部加强带按参考图推断',
    'bottom': '底视以自动排水器为中心；参考底图的灰色椭圆是渲染背景/阴影，未复制',
}


def make_report_08(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '08 分离过滤器 · 六视图参考（未标定）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_08(panel))
        actual = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.getchannel('A')) >= 128)
        iou = (float(np.logical_and(expected, actual).sum() /
                     max(1, np.logical_or(expected, actual).sum()))
               if expected is not None and actual is not None else None)
        old_new = (float(np.logical_and(old_fit, actual).sum() /
                         max(1, np.logical_or(old_fit, actual).sum()))
                   if old_fit is not None and actual is not None else None)
        note = ('按最长边等比对齐；参考图未标定，参考分割为近似诊断，仅供发现差异；'
                + VIEW_NOTES[name])
        rows.append({'view': name,
                     'uncalibrated_reference_mask_iou': round(iou, 4) if iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4) if old_new is not None else None,
                     'note': note})
        draw.text((x + 12, y), f'{name.upper()}  参考图 / 原模型 / 新候选',
                  font=small, fill='#27363c')
        ref_mask = reference_mask_08(panel)
        rx, ry, rw, rh = cv2.boundingRect(ref_mask)
        panel = panel.crop((max(0, rx - 5), max(0, ry - 5),
                            min(panel.width, rx + rw + 5),
                            min(panel.height, ry + rh + 5)))
        for j, image in enumerate((panel.convert('RGBA'), old, new)):
            if j:
                rect = image.getchannel('A').getbbox()
                if rect:
                    image = image.crop(rect)
            factor = min(182 / image.width, 485 / image.height)
            image = image.resize((max(1, round(image.width * factor)),
                                  max(1, round(image.height * factor))),
                                 Image.Resampling.LANCZOS)
            tile = Image.new('RGBA', (190, 495), 'white')
            tile.alpha_composite(image, ((190 - image.width) // 2,
                                         (495 - image.height) // 2))
            board.paste(tile.convert('RGB'), (x + 5 + j * 195, y + 32))
        draw.text((x + 12, y + 548),
                  (f'参考粗略轮廓 IoU：{iou:.3f} · 原/新 IoU：{old_new:.3f}')
                  if iou is not None and old_new is not None else '轮廓不可比较',
                  font=small, fill='#52646b')
    draw.text((20, 1230),
              '重要：参考图未标定，图中分数不是实测精度；原模型为单张重建，六视角审计记录到漂浮碎片和融合的细结构。'
              '候选以原照比例与可见结构为准，参考图只用于背面拓扑与外观对照；孔洞/材质/反射检查另行记录。',
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
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    items = (
        ('detail-lift-eye', '顶部锻造提环与加强带'),
        ('detail-lift-eye-side', '提环侧视与鞍座'),
        ('detail-flange-left', '左侧四螺栓法兰与中心通孔'),
        ('detail-flange-right', '右侧法兰（背面推断）'),
        ('detail-nozzle-weld', '喷嘴与筒体角焊缝'),
        ('detail-top-seam', '上部环焊缝'),
        ('detail-bottom-seam', '下部环焊缝'),
        ('detail-drain', '黄铜自动排水器总成'),
        ('detail-drain-window', '椭圆视窗与绿色镜片'),
        ('detail-drain-knurl', '滚花压紧环'),
        ('detail-drain-bottom', '底部排污螺纹口'),
        ('detail-surface', '蓝色金属漆、锈斑与划痕'),
    )
    board = Image.new('RGB', (1800, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '08 分离过滤器 · 关键局部检查（新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = (i % 6) * 300
        y = 55 + (i // 6) * 780
        image = _tile_image(out / 'renders' / f'{name}.png', (270, 660))
        tile = Image.new('RGBA', (285, 675), 'white')
        tile.alpha_composite(image, ((285 - image.width) // 2,
                                     (675 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 7, y + 30))
        draw.text((x + 10, y), label, font=small, fill='#27363c')
    draw.text((20, 1600), '这些局部图用于人工检查结构；不能证明未观察面、尺寸、'
                          '装配关系、孔深、螺纹或内部通道。', font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    board = Image.new('RGB', (1800, 820), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '08 分离过滤器 · 隔离开口与结构检查',
              font=font, fill='#18242b')
    items = (('flange-bores-only', '法兰中心孔与每侧 4 个螺栓孔', 'flange_bores'),
             ('lift-eye-only', '提环真实通孔', 'lift_eye'),
             ('drain-bore-only', '排污喷管/螺母通孔', 'drain_bore'),
             ('body-only', '封闭筒体与环形焊缝（外观）', None))
    for i, (name, label, key) in enumerate(items):
        x = i * 450
        image = _tile_image(out / 'renders' / f'{name}.png', (420, 560))
        tile = Image.new('RGBA', (435, 570), 'white')
        tile.alpha_composite(image, ((435 - image.width) // 2,
                                     (570 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 8, 92))
        if key:
            check = checks[key]
            status = '通过' if check['passed'] else '未通过'
            text = f'{label}：检测 {check["detected_openings"]} / 预期 {check["expected"]}  {status}'
            color = '#1d5c3a' if check['passed'] else '#8a2b20'
        else:
            text = f'{label}（不参与孔数判定）'
            color = '#52646b'
        draw.text((x + 10, 48), text, font=small, fill=color)
    draw.text((20, 740), '透明背景隔离渲染；孔数由连通域检测。法兰孔、提环孔和排污喷管/螺母孔为真实透明开口；'
                         '排污体本体是封闭实体，可见排污孔在主体处封闭；简体仅用于外观检查。'
                         '孔数不能证明孔深、螺纹、内部通道或装配关系。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '08 分离过滤器 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选 PBR 蓝色金属漆与黄铜'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), 'white')
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
    draw.text((20, 12), '08 分离过滤器 · 变光检查（同视角）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光位置 A'),
                                       ('new-light-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '灯光移动后高光和明暗应随视角变化；'
                         '本图只检查反射响应，不证明现场光照方向。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '08 分离过滤器 · 金属反射变光对照',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光 A'),
                                       ('new-metal-alternate', '主光 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '金属高光随灯光位置移动；不能据此推断现场光源方向或强度。',
              font=small, fill='#52646b')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_turntable(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    for i in range(24):
        image = _tile_image(out / 'renders' / f'turn-{i:02}.png', (220, 215))
        tile = Image.new('RGBA', (230, 222), 'white')
        tile.alpha_composite(image, ((230 - image.width) // 2,
                                     (222 - image.height) // 2))
        x = (i % 6) * 240 + 5
        y = (i // 6) * 250 + 4
        board.paste(tile.convert('RGB'), (x, y))
        draw.text((x + 8, y + 224), f'{i * 15:03d}°', font=font, fill='#344044')
    board.save(out / '转台-24方向.jpg', quality=94)


def audit_row(number):
    if not AUDIT.is_file():
        return None
    for line in AUDIT.read_text(encoding='utf8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('number') == number:
                return row
    return None


def reference_ratio_diagnostics(reference):
    image = Image.open(reference).convert('RGB')
    out = {}
    for i, name in enumerate(VIEWS):
        panel = panel_image(image, i)
        mask = reference_mask_08(panel)
        points = cv2.findNonZero(mask)
        if points is None:
            out[name] = None
            continue
        x, y, ww, hh = cv2.boundingRect(points)
        out[name] = {'bbox_px': [int(x), int(y), int(ww), int(hh)],
                     'width_over_height': round(ww / max(1, hh), 4)}
    out['proportion_note'] = (
        'Rust-photo alpha: total height about 1232 px, cylindrical shell '
        'diameter about 285 px, flange outer diameter about 222 px, nozzle '
        'pipe outer diameter about 88 px. The candidate uses a shell diameter '
        'of 0.232 m and total height about 1.0 m. These are pixel ratios and a '
        'model-library visual scale, not measurements. The six-view sheet is '
        'an uncalibrated visual-match reference, not the same photograph.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=8)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views and surfaces only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 8:
        p.error('this entry supports item 08 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/08-separator-draft1' if a.draft
                        else 'outputs/08分离过滤器-六视图重建-v1')
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
    config_path = HERE / 'separator08_config.json'
    worker_path = HERE / 'mesh_rebuild_08_worker.py'
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
                        a.out / 'brass_base.png', a.out / 'brass_mr.png',
                        a.out / 'brass_normal.png', a.out / '六视图-新旧对照.jpg',
                        a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '转台-24方向.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('detail-lift-eye', 'detail-lift-eye-side',
                              'detail-flange-left', 'detail-flange-right',
                              'detail-nozzle-weld', 'detail-top-seam',
                              'detail-bottom-seam', 'detail-drain',
                              'detail-drain-window', 'detail-drain-knurl',
                              'detail-drain-bottom', 'detail-surface',
                              'flange-bores-only', 'lift-eye-only',
                              'drain-bore-only', 'body-only',
                              'new-light-alternate', 'new-material-close',
                              'new-metal-alternate', 'old-material-close')]
                required += [a.out / 'renders' / f'turn-{i:02}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_paint_atlas_08(Path(row['rust']), a.out, config['material_profile'])
    textures.update(make_brass_08(Path(row['rust']), a.out, config['material_profile']))
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
    views = make_report_08(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(8)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    if not a.draft:
        make_turntable(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('flange_bores', 'flange-bores-only.png', 5),
                ('lift_eye', 'lift-eye-only.png', 1),
                ('drain_bore', 'drain-bore-only.png', 1)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    material_checks = paint_texture_checks(a.out / 'paint_base.png')
    brass_checks = brass_texture_checks(a.out / 'brass_base.png', a.out / 'brass_mr.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'separator_body': 'Separator body' in names,
        'seam_welds': sum(name in {'Top seam weld', 'Bottom seam weld'} for name in names),
        'seam_welds_expected': 2,
        'nozzle_pipes': sum(name.endswith(' pipe') and name.startswith('Nozzle ') for name in names),
        'nozzle_pipes_expected': 2,
        'nozzle_pipe_sleeves': sum(name.endswith(' pipe sleeve') for name in names),
        'nozzle_pipe_sleeves_expected': 2,
        'flanges': sum(name.endswith(' flange') and name.startswith('Nozzle ') for name in names),
        'flanges_expected': 2,
        'raised_faces': sum('raised face' in name for name in names),
        'raised_faces_expected': 4,
        'nozzle_welds': sum(name.startswith('Nozzle ') and name.endswith(' weld') for name in names),
        'nozzle_welds_expected': 2,
        'flange_bolt_holes_expected': 8,
        'lift_lug_parts': sum(name.startswith('Lift lug') for name in names),
        'lift_lug_parts_expected': 5,
        'drain_blue_parts': sum(name.startswith('Drain blue') or
                                name.startswith('Drain thread') or
                                name == 'Drain head weld' for name in names),
        'drain_blue_parts_expected': 5,
        'drain_brass_nut': 'Drain brass nut' in names,
        'drain_washer': 'Drain washer' in names,
        'drain_knurled_collar': 'Drain knurled collar' in names,
        'drain_collar_ribs': sum(name.startswith('Drain collar rib') for name in names),
        'drain_collar_ribs_expected': 36,
        'drain_body': 'Drain body' in names,
        'drain_window_bevels': sum(name.startswith('Drain window bezel') for name in names),
        'drain_window_bevels_expected': 2,
        'drain_window_lenses': sum(name.startswith('Drain window lens') for name in names),
        'drain_window_lenses_expected': 2,
        'drain_window_backings': sum(name.startswith('Drain window backing') for name in names),
        'drain_window_backings_expected': 2,
        'drain_spigot_parts': sum(name in {'Drain spigot nut', 'Drain spigot',
                                           'Drain spigot lip'} for name in names),
        'drain_spigot_parts_expected': 3,
        'drain_bore_end': 'Drain bore end' in names,
        'isolated_openings': hole_checks,
        'note': ('Part presence proves only that a feature was modelled, not its size, '
                 'fit or hidden faces. The rear nozzle, back window, flange faces, '
                 'eye band, drain internals and all hidden faces are partly inferred.'),
    }
    reflection = {'status': 'not_run'}
    if not a.draft:
        first = np.asarray(Image.open(a.out / 'renders/new-material-close.png').convert('RGBA'))
        second = np.asarray(Image.open(a.out / 'renders/new-metal-alternate.png').convert('RGBA'))
        mask = first[:, :, 3] >= 128
        reflection = {
            'mean_abs_luma_difference': round(float(np.abs(first[:, :, :3].astype(float) -
                                                          second[:, :, :3].astype(float))[mask].mean()), 4),
            'changed': bool(np.abs(first[:, :, :3].astype(float) -
                                   second[:, :, :3].astype(float))[mask].mean() > 2.0),
            'note': 'Same camera, different key-light direction; highlights must move.'
        }
    structural_ok = bool(
        structural['separator_body'] and structural['seam_welds'] == 2 and
        structural['nozzle_pipes'] == 2 and structural['nozzle_pipe_sleeves'] == 2 and
        structural['flanges'] == 2 and structural['raised_faces'] == 4 and
        structural['nozzle_welds'] == 2 and structural['lift_lug_parts'] == 5 and
        structural['drain_blue_parts'] == 5 and structural['drain_brass_nut'] and
        structural['drain_washer'] and structural['drain_knurled_collar'] and
        structural['drain_collar_ribs'] == 36 and structural['drain_body'] and
        structural['drain_window_bevels'] == 2 and
        structural['drain_window_lenses'] == 2 and
        structural['drain_window_backings'] == 2 and
        structural['drain_spigot_parts'] == 3 and
        structural['drain_bore_end'])
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 65 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': structural_ok,
        'material_tiles': bool(material_checks['passed'] and brass_checks['passed']),
        'metal_reflection': bool(reflection.get('changed', False)),
        'reference_diagnostic_only': True,
        'browser_page': False,
        'source_fidelity': bool(fidelity.get('passed', False)),
    }
    if a.draft:
        passed['isolated_openings'] = 'not_run'
        passed['metal_reflection'] = 'not_run'
        passed['material_tiles'] = 'not_run'
    report = {
        'version': config['version'], 'number': 8, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_visual_match_reference',
        'reference_role': 'visual_match_unverified',
        'method': 'parametric_part_rebuild_from_source_photo_proportions_with_visual_match_reference',
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_clean_sha256': row['clean_sha256'],
        'original_rust_sha256': row['rust_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old mesh library source IoU 0.9375 only measures one silhouette; '
            'the six-view audit records floating fragments in side/top views and '
            'the single-image mesh fuses the flanges, lifting-eye hole and brass '
            'drain details. This rebuild therefore does not reuse the old mesh.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'paint_texture_checks': material_checks,
        'brass_texture_checks': brass_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The supplied rust photo and the stored source camera come from the '
            'single-image library reconstruction; the new-source framing distance '
            'was scaled by the new/old span and then fitted to the source-alpha '
            'height and centred laterally on the source bbox so framing does not '
            'dominate the silhouette score. The score is diagnostic only and is '
            'not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is an uncalibrated visual-match '
                       'reference of the same visible separator design, not a '
                       'calibrated image set. It is used for the rear nozzle and '
                       'rear window topology and to reveal gross differences; the '
                       'candidate follows the photo for observed structure.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view sheet is uncalibrated and not a measured image set; the reference segmentation is a diagnostic.',
            'The working shell diameter (0.232 m) and total height (about 1.0 m) follow photo pixel ratios and the model-library visual scale; they are not measured dimensions.',
            'The rear nozzle, rear drain window, flange faces, bore depths, bolt-hole diameters, bolt circle and raised-face profile are inferred.',
            'The eye-ring top band, saddle-pad profile and forged-eye cross-section are inferred from the photo and the uncalibrated sheet.',
            'The automatic-drain internal mechanism, thread forms, both-window glazing and the stamped markings are not reproduced; markings are omitted.',
            'The nozzle bores are open through each flange/nozzle but stop at the closed shell; internal separator connections are not modelled or verified.',
            'The bottom drain bore is open through the spigot and nut and is capped at the body; the internal drain passage is not modelled or verified.',
            'No support, bracket or feet are visible; floor contact, stability and mounting are not established.',
            'Rust, paint chips and scratches are synthesized from small observed patches; they are not a photograph of hidden faces.',
            'Isolated opening counts prove only transparent openings in the isolated renders; they do not prove hole depth, threads, fits or assembly.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the two flange faces, the lifting eye and the brass drain.',
            'Confirm whether the rear nozzle and rear drain window match the photographed unit; they are inferred from the uncalibrated reference.',
            'Confirm flange thickness, bolt-hole diameter, bolt circle, raised-face profile and bore depth against a measured or higher-resolution sample.',
            'Confirm the thread forms and internal mechanism of the brass automatic drain and whether both oval windows are glazed.',
            'Confirm the eye-ring top band, eye cross-section and saddle-pad shape.',
            'Confirm the attachment/mounting method: no support is visible, so stability and floor contact are unverified.',
            'Confirm the internal separator element and whether the nozzle bores connect through the vessel.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 9,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_08.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 8, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed, 'material_checks': material_checks,
        'brass_checks': brass_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'reference_role': 'visual_match_unverified',
        'pending_review': report['pending_review'],
        'next_number': 9,
        'manual_only': True,
        'notes': ('The user six-view sheet is an uncalibrated visual-match reference; '
                  'the old model and items 01-07 were not modified. Candidate is '
                  'manual_only. Browser review page and visual acceptance are pending '
                  'unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '08-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
