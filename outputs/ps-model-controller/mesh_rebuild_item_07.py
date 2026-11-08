"""Build the review-only vertical stainless mixing-tank candidate (item 07).

The shape follows the original rust photograph for proportions and visible
features.  The user-supplied six-view sheet is the registered category
reference for item 07: it shows a similar vertical tank but its centre manway
and ring of top nozzles do not exist on the photographed unit.  The sheet is
used only for the general three-leg layout and as a diagnostic comparison,
never as a measured model.  The working size follows the model-library visual
scale.  The candidate never replaces the library mesh and is never used by the
automatic placement pipeline.  A new output directory is required.
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
REFERENCE_SHA256 = '40de210e08b406a107fe3e34a88458a6e7cb0e7c02428fce19d8653fb80b7ef6'


def find_reference():
    folder = Path.home() / 'Desktop' / '六视图'
    if folder.is_dir():
        for path in sorted(folder.glob('*.png')):
            if digest(path) == REFERENCE_SHA256:
                return path
    raise FileNotFoundError('registered six-view reference not found')


def make_atlas_07(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != (1086, 1448):
        raise ValueError('07 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((760, 250, 880, 370), (300, 690, 420, 810),
                (640, 850, 760, 970), (260, 450, 380, 570))]
    rng = np.random.default_rng(20707)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    # Base colours are neutral stainless steel; highlight direction is left to
    # the live lighting.  Rust and brushing come from small observed patches
    # of the supplied rust image, not from hidden faces.  The streaks are
    # elongated vertically because the photographed tank has vertical runs.
    specs = [((178, 178, 176), profile['bright_stainless'], 2.0),
             ((151, 152, 151), profile['brushed_stainless'], 4.8),
             ((126, 120, 110), profile['weathered_stainless'], 4.0),
             ((106, 98, 88), profile['stained_steel'], 2.6)]
    for i, (color, mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((56, 56), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((560, 560), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Vertical brushing: noise varies across the image and is stretched in v.
        brush = cv2.resize(rng.normal(size=(size, 22)).astype(np.float32),
                           (size, size), interpolation=cv2.INTER_CUBIC)
        brush = cv2.GaussianBlur(brush, (0, 0), 2.4)
        brush *= brush_strength / max(1e-6, float(brush.std()))
        field = .45 * (coarse - .5) + .35 * (fine - .5) + .20 * (speck - .5)
        rust_amount = mat['rust_amount']
        rust = np.clip((coarse - (.82 - rust_amount * .85)) * 8, 0, 1) * rust_amount
        streak = np.clip((-brush - 1.5) * .30, 0, 1) * rust_amount * 2.6
        rust = np.clip(rust + streak, 0, 1)
        rgb = np.asarray(color, np.float32)[None, None, :] * (1 + field[:, :, None] * .055)
        rgb += detail[:, :, None] * .24 + brush[:, :, None] * .50
        rgb += ((speck - .5) * 0.5)[:, :, None]
        rgb = rgb * (1 - rust[:, :, None]) + np.array((118, 71, 41), np.float32) * rust[:, :, None]
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip((mat['roughness'] + rust * .14
                                    + field * .05 + brush * .006) * 255, 0, 255),
                           np.clip((mat['metallic'] * (1 - rust * .58)) * 255, 0, 255)),
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
    for name, arr in (('base_color', base), ('metallic_roughness', mr), ('normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_paint_07(out):
    """Blue motor paint with wear; base colour and metal/roughness are separate."""
    size = 1024
    rng = np.random.default_rng(20708)
    coarse = cv2.resize(rng.random((48, 48), dtype=np.float32), (size, size),
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((400, 400), dtype=np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
    speck = cv2.resize(rng.random((size, size), dtype=np.float32), (size, size),
                       interpolation=cv2.INTER_NEAREST)
    field = .45 * (coarse - .5) + .35 * (fine - .5) + .20 * (speck - .5)
    # Small, sparse wear and chips only; the photographed motor is a deep,
    # mostly solid blue with grime, not a blotchy camouflage pattern.
    wear = np.clip((fine - .86) * 9, 0, 1)
    chips = np.clip((speck - .988) * 30, 0, 1)
    rust = np.clip((coarse - .82) * 8, 0, 1) * .25
    blue = np.array((22, 54, 96), np.float32)
    rgb = blue[None, None, :] * (1 + field[:, :, None] * .10)
    rubbed = np.array((84, 96, 112), np.float32)
    rgb = rgb * (1 - wear[:, :, None] * .28) + rubbed * (wear[:, :, None] * .28)
    rgb = rgb * (1 - chips[:, :, None] * .8) + np.array((158, 162, 160), np.float32) * chips[:, :, None] * .8
    rgb = rgb * (1 - rust[:, :, None] * .45) + np.array((112, 66, 38), np.float32) * rust[:, :, None] * .45
    base = np.clip(rgb, 0, 255).astype(np.uint8)
    rough = np.clip(.40 + fine * .12 + wear * .18 + chips * .08, .15, .95)
    metallic = np.clip(.14 - wear * .05 - rust * .05, .02, .25)
    mr = np.stack((np.full((size, size), 255, np.uint8),
                   (rough * 255).astype(np.uint8),
                   (metallic * 255).astype(np.uint8)), axis=2)
    bump = cv2.GaussianBlur(fine + speck * .4, (0, 0), .7)
    dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 10
    dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 10
    nx = np.clip(-dx * .45, -.08, .08)
    ny = np.clip(dy * .45, -.08, .08)
    nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
    normal = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                     0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('paint_base', base), ('paint_mr', mr), ('paint_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def material_texture_checks(path):
    mr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    tiles = {'bright_stainless': np.s_[0:1024, 0:1024],
             'brushed_stainless': np.s_[0:1024, 1024:2048],
             'weathered_stainless': np.s_[1024:2048, 0:1024],
             'stained_steel': np.s_[1024:2048, 1024:2048]}
    values = {name: {'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
                     'metallic_mean': round(float(mr[sl][..., 2].mean()), 4)}
              for name, sl in tiles.items()}
    values['passed'] = (
        values['bright_stainless']['metallic_mean'] > .90 and
        values['bright_stainless']['roughness_mean'] < .35 and
        values['brushed_stainless']['metallic_mean'] > .85 and
        values['weathered_stainless']['metallic_mean'] > .60 and
        values['stained_steel']['roughness_mean'] > .70 and
        values['stained_steel']['metallic_mean'] < .60)
    values['note'] = 'Tile means; reflection direction is not baked into base color.'
    return values


def paint_texture_checks(base_path, mr_path):
    rgb = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    blueness = float(rgb[..., 2].mean() - rgb[..., 0].mean())
    values = {'roughness_mean': round(float(mr[..., 1].mean()), 4),
              'metallic_mean': round(float(mr[..., 2].mean()), 4),
              'blue_minus_red': round(blueness, 2),
              'passed': bool(.30 < mr[..., 1].mean() < .70 and
                             mr[..., 2].mean() < .30 and blueness > 20)}
    values['note'] = 'Blue paint only; the steel tiles are checked separately.'
    return values


def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 12, y0 + 8, x0 + w // 3 - 12, y0 + h // 2 - 8))


def reference_mask_07(panel):
    """Approximate metal-object mask for an uncalibrated reference panel.

    GrabCut plus a brightness floor; it is a diagnostic segmentation, not a
    calibrated silhouette measurement.  The same method is used for all six
    panels so the numbers can only reveal gross proportion differences.
    """
    rgb = np.asarray(panel.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, (5, 5, w - 10, h - 10), bgd, fgd, 4,
                cv2.GC_INIT_WITH_RECT)
    coarse = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    out = (coarse & (gray > 115)).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    out = binary_fill_holes(out).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(out, 8)
    if n > 1:
        best, best_distance = None, None
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] < 1000:
                continue
            distance = ((centroids[j, 0] - w / 2) ** 2 +
                        (centroids[j, 1] - h / 2) ** 2) ** .5
            if best is None or distance < best_distance:
                best, best_distance = j, distance
        if best is not None:
            out = (labels == best).astype(np.uint8)
    return out


VIEW_NOTES = {
    'front': '参考图是相近立式罐：顶部为中央人孔与管口环，照片为偏置盲盖大端口、电机与侧管；候选按照片，不套用参考顶部',
    'back': '背面在原照不可见；腿部按 120° 三腿推断，背面管口/焊缝细节未证实',
    'left': '参考左视有上下侧管口，照片只显示前左侧循环管；未加参考侧的额外管口',
    'right': '参考右视的侧管口与照片不一致；仅按照片建模电机和支架可见部分',
    'top': '参考顶视为中央人孔和管口环；照片为偏置大端口、电机支架与循环管；顶部结构按照片并标注推断',
    'bottom': '参考底视为中央排污和三条支腿；候选按照片的螺栓锥底、下锥排污与 120° 三腿，参考仅作类别印证',
}


def make_report_07(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '07 立式不锈钢搅拌罐 · 参考图（类别参考）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_07(panel))
        actual = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.getchannel('A')) >= 128)
        iou = (float(np.logical_and(expected, actual).sum() /
                     max(1, np.logical_or(expected, actual).sum()))
               if expected is not None and actual is not None else None)
        old_new = (float(np.logical_and(old_fit, actual).sum() /
                         max(1, np.logical_or(old_fit, actual).sum()))
                   if old_fit is not None and actual is not None else None)
        note = ('按最长边等比对齐；参考图未标定且仅为相近类别，参考分割为近似诊断，'
                '仅供发现差异；' + VIEW_NOTES[name])
        rows.append({'view': name,
                     'uncalibrated_reference_mask_iou': round(iou, 4) if iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4) if old_new is not None else None,
                     'note': note})
        draw.text((x + 12, y), f'{name.upper()}  参考图（类别）/ 原模型 / 新候选',
                  font=small, fill='#27363c')
        ref_mask = reference_mask_07(panel)
        rx, ry, rw, rh = cv2.boundingRect(ref_mask)
        panel = panel.crop((max(0, rx - 5), max(0, ry - 5),
                            min(panel.width, rx + rw + 5), min(panel.height, ry + rh + 5)))
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
              '重要：本件六视图映射为 category_reference。参考罐为中央人孔、顶部管口环和不同支腿，'
              '原照为偏置盲盖大端口、蓝色电机支架和侧循环管；候选以原照为准，不把参考顶部当作本件结构。'
              '参考图未标定，图中分数不是实测精度；原模型和候选的金属/孔洞检查另行记录。',
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
        ('detail-top-port', '偏置大端口、盲盖与四个摆动螺栓'),
        ('detail-lift-lug', '提环、真实通孔与底座'),
        ('detail-motor', '蓝色电机、齿轮箱与散热筋'),
        ('detail-skid', '不锈钢支架、立柱与驱动轴'),
        ('detail-side-pipe', '前左侧循环长管与弯头'),
        ('detail-pipe-valve', '低侧法兰、阀门与 T 形手柄'),
        ('detail-sight-glass', '胶囊视镜、蓝镜片与六角螺塞'),
        ('detail-cone-flange', '锥底法兰、20 螺栓与两个夹紧耳'),
        ('detail-legs', '120° 三腿、板折痕与焊缝'),
        ('detail-foot', '支脚螺杆与调平垫'),
        ('detail-seam', '上部/中部/下部三条环焊缝带'),
        ('detail-surface', '拉丝不锈钢、垂直锈痕与反射'),
    )
    board = Image.new('RGB', (1800, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '07 立式不锈钢搅拌罐 · 关键局部检查（新候选）',
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
                          '装配关系、孔深或内部通道。', font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    board = Image.new('RGB', (1800, 820), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '07 立式不锈钢搅拌罐 · 隔离开口与结构检查',
              font=font, fill='#18242b')
    items = (('top-port-only', '顶部端口通孔（不含盖）', 'top_port'),
             ('lift-lug-only', '提环真实通孔', 'lift_lug'),
             ('drain-only', '底部排污通孔', 'drain'),
             ('legs-only', '三条折板支腿（外观）', None),
             ('motor-only', '电机与不锈钢支架（外观）', None))
    for i, (name, label, key) in enumerate(items):
        x = i * 360
        image = _tile_image(out / 'renders' / f'{name}.png', (330, 560))
        tile = Image.new('RGBA', (345, 570), 'white')
        tile.alpha_composite(image, ((345 - image.width) // 2,
                                     (570 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 8, 92))
        if key:
            check = checks[key]
            status = ('通过' if check['passed'] else '未通过')
            text = f'{label}：检测 {check["detected_openings"]} / 预期 {check["expected"]}  {status}'
            color = '#1d5c3a' if check['passed'] else '#8a2b20'
        else:
            text = f'{label}（不参与孔数判定）'
            color = '#52646b'
        draw.text((x + 10, 48), text, font=small, fill=color)
    draw.text((20, 740), '透明背景隔离渲染；孔数由连通域检测。顶部端口、提环和排污口为真实透明开口；'
                         '支腿与电机仅用于结构外观检查。孔数不能证明孔深、螺纹、内部通道或装配关系。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '07 立式不锈钢搅拌罐 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型金属表面（顶点色）'),
                                       ('new-material-close', '新候选 PBR 拉丝不锈钢'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (840 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), '同一中性光与相机；高光来自实时照明，不烘焙在底色中。'
                         '原模型为单图重建的像素/顶点色，不代表旧模型整体通过。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '07 立式不锈钢搅拌罐 · 变光检查（同视角）',
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
    draw.text((20, 12), '07 立式不锈钢搅拌罐 · 金属反射变光对照',
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
        mask = reference_mask_07(panel)
        points = cv2.findNonZero(mask)
        if points is None:
            out[name] = None
            continue
        x, y, ww, hh = cv2.boundingRect(points)
        out[name] = {'bbox_px': [int(x), int(y), int(ww), int(hh)],
                     'width_over_height': round(ww / max(1, hh), 4)}
    out['proportion_note'] = (
        'Original photograph: body diameter about 710 px, total height about '
        '1425 px (diameter/height about 0.498), shell height about 785 px and '
        'leg length about 580 px. The candidate uses diameter 0.498 m and '
        'total height about 1.0 m. These are pixel ratios and a model-library '
        'visual scale, not measurements. The six-view sheet is a category '
        'reference; its panels are not the same tank and are not calibrated.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=7)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views and surfaces only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 7:
        p.error('this entry supports item 07 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/07-tank-draft1' if a.draft
                        else 'outputs/07储罐-六视图重建-v1')
    a.out, a.library, a.blender = (x.resolve() for x in (a.out, a.library, a.blender))
    if digest(a.reference) != REFERENCE_SHA256:
        raise ValueError('six-view reference hash differs from the registered value')
    row = next(r for r in (json.loads(line) for line in
                           a.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == a.number)
    for name, expected in (('mesh', 'mesh_sha256'), ('rust', 'rust_sha256')):
        if digest(Path(row[name])) != row[expected]:
            raise ValueError(f'{name} hash differs from the model library')
    alpha_path = Path(row['source_view']).parent / 'source-alpha.png'
    if digest(alpha_path) != row['source_alpha_sha256']:
        raise ValueError('source alpha hash differs from the model library')
    config_path = HERE / 'tank07_config.json'
    worker_path = HERE / 'mesh_rebuild_07_worker.py'
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
            required = [a.out / 'model.blend', a.out / 'base_color.png',
                        a.out / 'metallic_roughness.png', a.out / 'normal.png',
                        a.out / 'paint_base.png', a.out / 'paint_mr.png',
                        a.out / 'paint_normal.png', a.out / '六视图-新旧对照.jpg',
                        a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '转台-24方向.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('detail-top-port', 'detail-lift-lug', 'detail-motor',
                              'detail-skid', 'detail-side-pipe', 'detail-pipe-valve',
                              'detail-sight-glass', 'detail-cone-flange', 'detail-legs',
                              'detail-foot', 'detail-seam', 'detail-surface',
                              'top-port-only', 'lift-lug-only', 'drain-only',
                              'legs-only', 'motor-only', 'new-light-alternate',
                              'new-material-close', 'new-metal-alternate',
                              'old-material-close')]
                required += [a.out / 'renders' / f'turn-{i:02}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_atlas_07(Path(row['rust']), a.out, config['material_profile'])
    textures.update(make_paint_07(a.out))
    request = {'out': str(a.out), 'original_mesh': row['mesh'], 'textures': textures,
               'config': config, 'source_camera': row['reconstruction']['source_camera'],
               'draft': a.draft}
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
    views = make_report_07(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(7)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    if not a.draft:
        make_turntable(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('top_port', 'top-port-only.png', 1),
                ('lift_lug', 'lift-lug-only.png', 1),
                ('drain', 'drain-only.png', 1)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    material_checks = material_texture_checks(a.out / 'metallic_roughness.png')
    paint_checks = paint_texture_checks(a.out / 'paint_base.png', a.out / 'paint_mr.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'tank_body': 'Tank body' in names,
        'seam_bands': sum(name.endswith('seam band') or name.endswith('seam band')
                          for name in names),
        'seam_bands_expected': 3,
        'legs': sum(name in {'Leg 1', 'Leg 2', 'Leg 3'} for name in names),
        'legs_expected': 3,
        'foot_stems': sum(name.startswith('Leg ') and name.endswith('foot stem')
                          for name in names),
        'foot_stems_expected': 3,
        'foot_collars': sum(name.startswith('Leg ') and name.endswith('foot collar')
                            for name in names),
        'foot_collars_expected': 3,
        'foot_pads': sum(name.startswith('Leg ') and name.endswith('foot pad')
                         for name in names),
        'foot_pads_expected': 3,
        'top_port_parts': sum(name.startswith('Top port') for name in names),
        'top_port_parts_expected': 15,
        'top_port_swing_bolts': sum(name.startswith('Top port swing bolt')
                                    for name in names),
        'top_port_swing_bolts_expected': 4,
        'top_port_nuts': sum(name.startswith('Top port nut') for name in names),
        'top_port_nuts_expected': 4,
        'lift_lug_parts': sum(name.startswith('Lift lug') for name in names),
        'lift_lug_parts_expected': 5,
        'top_pipe': 'Top pipe' in names,
        'top_pipe_sleeve': 'Top pipe sleeve' in names,
        'pipe_top_stub': 'Pipe top stub' in names,
        'pipe_top_stub_sleeve': 'Pipe top stub sleeve' in names,
        'pipe_top_bolts': sum(name.startswith('Pipe top bolt') for name in names),
        'pipe_top_bolts_expected': 4,
        'pipe_wall_bolts': sum(name.startswith('Pipe wall bolt') for name in names),
        'pipe_wall_bolts_expected': 4,
        'pipe_valve': 'Pipe valve body' in names,
        'pipe_valve_stem': 'Pipe valve stem' in names,
        'pipe_valve_handle': 'Pipe valve handle' in names,
        'pipe_valve_flanges': sum(name.startswith('Pipe valve flange') for name in names),
        'pipe_valve_flanges_expected': 2,
        'sight_glass_parts': sum(name.startswith('Sight glass') for name in names),
        'sight_glass_parts_expected': 5,
        'cone_flange_bolts': sum(name.startswith('Cone flange bolt') for name in names),
        'cone_flange_bolts_expected': 20,
        'cone_clamp_lugs': sum(name.startswith('Cone clamp lug') for name in names),
        'cone_clamp_lugs_expected': 2,
        'cone_clamp_pins': sum(name.startswith('Cone clamp pin') for name in names),
        'cone_clamp_pins_expected': 2,
        'lower_cone': 'Lower cone' in names,
        'drain_pipe': 'Drain pipe' in names,
        'drain_sleeve': 'Drain pipe sleeve' in names,
        'drain_lip': 'Drain lip' in names,
        'skid_rails': sum(name.startswith('Skid rail') for name in names),
        'skid_rails_expected': 2,
        'skid_cross_rails': sum(name.startswith('Skid cross rail') for name in names),
        'skid_cross_rails_expected': 2,
        'skid_posts': sum(name in {'Skid post minus', 'Skid post plus'} for name in names),
        'skid_posts_expected': 2,
        'skid_post_pads': sum(name.startswith('Skid post pad') for name in names),
        'skid_post_pads_expected': 2,
        'skid_columns': sum(name.startswith('Skid column ') and 'flange' not in name
                            for name in names),
        'skid_columns_expected': 2,
        'skid_column_flanges': sum(name.startswith('Skid column') and 'flange' in name
                                   for name in names),
        'skid_column_flanges_expected': 4,
        'drive_shaft_parts': sum(name.startswith('Drive shaft') for name in names),
        'drive_shaft_parts_expected': 3,
        'motor_body': 'Motor body' in names,
        'motor_fins': sum(name.startswith('Motor fin ') for name in names),
        'motor_fins_expected': 12,
        'gearbox_ribs': sum(name.startswith('Gearbox rib') for name in names),
        'gearbox_ribs_expected': 6,
        'fan_grille_bars': sum(name.startswith('Fan grille bar') for name in names),
        'fan_grille_bars_expected': 8,
        'motor_feet': sum(name.startswith('Motor foot') for name in names),
        'motor_feet_expected': 2,
        'isolated_openings': hole_checks,
        'note': ('Part presence proves only that a feature was modelled, not its size, fit '
                 'or hidden faces. The leg cross-section, skid rear topology, sight-glass '
                 'pad, valve direction and all hidden faces are partly inferred.'),
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
        structural['tank_body'] and structural['seam_bands'] == 3 and
        structural['legs'] == 3 and structural['foot_stems'] == 3 and
        structural['foot_collars'] == 3 and structural['foot_pads'] == 3 and
        structural['top_port_parts'] == 15 and
        structural['top_port_swing_bolts'] == 4 and structural['top_port_nuts'] == 4 and
        structural['lift_lug_parts'] == 5 and
        structural['top_pipe'] and structural['top_pipe_sleeve'] and
        structural['pipe_top_stub'] and structural['pipe_top_stub_sleeve'] and
        structural['pipe_top_bolts'] == 4 and structural['pipe_wall_bolts'] == 4 and
        structural['pipe_valve'] and structural['pipe_valve_stem'] and
        structural['pipe_valve_handle'] and structural['pipe_valve_flanges'] == 2 and
        structural['sight_glass_parts'] == 5 and
        structural['cone_flange_bolts'] == 20 and
        structural['cone_clamp_lugs'] == 2 and structural['cone_clamp_pins'] == 2 and
        structural['lower_cone'] and structural['drain_pipe'] and
        structural['drain_sleeve'] and structural['drain_lip'] and
        structural['skid_rails'] == 2 and structural['skid_cross_rails'] == 2 and
        structural['skid_posts'] == 2 and structural['skid_post_pads'] == 2 and
        structural['skid_columns'] == 2 and structural['skid_column_flanges'] == 4 and
        structural['drive_shaft_parts'] == 3 and structural['motor_body'] and
        structural['motor_fins'] == 12 and structural['gearbox_ribs'] == 6 and
        structural['fan_grille_bars'] == 8 and structural['motor_feet'] == 2)
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 110 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': structural_ok,
        'material_tiles': bool(material_checks['passed'] and paint_checks['passed']),
        'metal_reflection': bool(reflection.get('changed', False)),
        'reference_diagnostic_only': True,
        'browser_page': False,
        'source_fidelity': False,
    }
    if a.draft:
        passed['isolated_openings'] = 'not_run'
        passed['metal_reflection'] = 'not_run'
        passed['material_tiles'] = 'not_run'
    report = {
        'version': config['version'], 'number': 7, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'source_photo_with_uncalibrated_category_reference',
        'reference_role': 'category_reference',
        'method': 'parametric_part_rebuild_from_source_photo_proportions_with_category_reference',
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old mesh source-view IoU 0.9589 only measures one silhouette; '
            'it fills five holes, narrow recall is 0.2716 and the record fails. '
            'The single-image mesh merges the three legs, the top port, the '
            'motor skid, the sight glass, the pipe and the valve, and has no '
            'real bores, so this rebuild does not reuse it.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'material_texture_checks': material_checks,
        'paint_texture_checks': paint_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The supplied rust photo is a single near-frontal studio image with a '
            'different scale and pose from the stored source camera; the new-source '
            'framing distance was scaled by the new/old span. The score is diagnostic '
            'only and is not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is an uncalibrated category reference of a '
                       'similar vertical tank and is not the photographed unit. Its top '
                       'centre manway and nozzle ring are not modelled. It is used only '
                       'for the general three-leg layout and to reveal gross differences; '
                       'the candidate follows the photo for observed structure.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view sheet is a category reference of a similar tank, uncalibrated and not the photographed unit; its centre manway and nozzle ring are not modelled.',
            'The working diameter (0.498 m) and total height (about 1.0 m) follow the photo pixel ratios and the model-library visual scale; they are not measured dimensions.',
            'The leg cross-section, fold angle, top attachment and leveling-foot construction are inferred from the photo and the category sheet.',
            'The rear skid/frame members, post count and hidden motor supports are inferred; only the visible +Y side can be compared with the photo.',
            'The sight-glass pad profile, lens colour, plug thread and its exact wall position are inferred.',
            'The recirculation pipe wall-flange azimuth, valve stem direction, bore connections and bolt counts are inferred.',
            'The top port swing-bolt arrangement, lid dome and neck bore depth are inferred.',
            'Bore depths, threads, wall piercing and internal connections are not measured.',
            'Rust and brush distribution is synthesized from small patches of the supplied rust image; it is not a photograph of hidden faces.',
            'Isolated opening counts prove only transparent openings in the isolated render; they do not prove hole depth, threads, fits or assembly.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the three-leg layout, the top port, the blue motor/skid and the front-left pipe.',
            'Decide whether the rear leg and the hidden skid members match the photographed unit; they are inferred at 120 degrees/rectangular frame.',
            'Confirm the leg fold, top weld, foot stem and pad against a measured sample.',
            'Confirm the top port lid, swing-bolt count, neck bore and offset against a measured or higher-resolution sample.',
            'Confirm the sight-glass frame, lens colour, plug and its position on the lower shell.',
            'Confirm the recirculation pipe routing, wall-flange position, valve stem/handle and bore connections.',
            'Confirm the cone flange bolt count, clamp lugs and bottom drain bore.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 8,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_07.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 7, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed, 'material_checks': material_checks,
        'paint_checks': paint_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'source_fidelity_diagnostic': fidelity,
        'reference_role': 'category_reference',
        'pending_review': report['pending_review'],
        'next_number': 8,
        'manual_only': True,
        'notes': ('The reference is a category reference, not the photographed unit; '
                  'the old model and items 01-06 were not modified. Candidate is '
                  'manual_only. Browser review page and visual acceptance are pending '
                  'unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '07-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
