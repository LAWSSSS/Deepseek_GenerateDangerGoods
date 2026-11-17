"""Build the review-only separator-filter candidate (item 09).

The shape follows the original rust photograph for proportions and visible
features.  The user-supplied six-view sheet is an uncalibrated visual-match
reference of the same visible design: a vertical separator with a top flange,
one side nozzle, a lower skirt on four legs and a dark automatic drain.  The
sheet is used for the hidden/back-side topology and as a diagnostic comparison,
never as measured geometry.  The working size follows the model-library visual
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
REFERENCE_SHA256 = '03fc1ff6751852fd693ee75ca20ad741b0fbcbe6d7319baf030a6406d5273cd1'
SOURCE_SIZE = (1254, 1254)


def find_reference():
    folder = Path.home() / 'Desktop' / '六视图'
    if folder.is_dir():
        for path in sorted(folder.glob('*.png')):
            if digest(path) == REFERENCE_SHA256:
                return path
    raise FileNotFoundError('registered six-view reference not found')


def make_steel_atlas_09(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('09 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((540, 430, 660, 550), (520, 880, 640, 1000),
                (470, 180, 590, 300), (590, 1095, 640, 1145))]
    rng = np.random.default_rng(20909)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [(profile['steel_clean'], 3.5),
             (profile['steel_weathered'], 4.0),
             (profile['steel_weld'], 3.0),
             (profile['steel_rusted'], 2.5)]
    rust_rgb = np.array((120, 92, 70), np.float32)
    dust_rgb = np.array((78, 76, 73), np.float32)
    scratch_rgb = np.array((196, 201, 206), np.float32)
    for i, (mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((120, 120), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((900, 900), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Vertical brushing: fine columns stretched along v.
        brush = cv2.resize(rng.normal(size=(160, size)).astype(np.float32),
                           (size, size), interpolation=cv2.INTER_CUBIC)
        brush = cv2.GaussianBlur(brush, (0, 0), 1.0)
        brush *= brush_strength / max(1e-6, float(brush.std()))
        field = .50 * (coarse - .5) + .30 * (fine - .5) + .20 * (speck - .5)
        amount = mat['rust_amount']
        # Sparse, fine rust clusters rather than broad painted blotches.
        patch = np.clip((coarse - (1 - amount)) * 20, 0, 1)
        rust = patch * np.clip((fine - .15) * 1.2, 0, 1)
        streak = np.clip((-brush - 2.5) * .15, 0, 1) * amount * .8
        rust = np.clip(rust + streak, 0, 1)
        dust = np.clip((speck - .9985) * 40, 0, 1) * .03
        color = np.asarray(mat['base_rgb'], np.float32)
        rgb = color[None, None, :] * (1 + field[:, :, None] * .030)
        rgb += detail[:, :, None] * .09 + brush[:, :, None] * .22
        rgb += ((speck - .5) * .20)[:, :, None]
        scratches = np.clip((brush - 3.8) * .06, 0, 1) * (1 - rust)
        rgb = rgb * (1 - scratches[:, :, None] * .12) + \
            scratch_rgb[None, None, :] * (scratches[:, :, None] * .12)
        rust_rgb_local = rust_rgb[None, None, :] * (.82 + .36 * fine[:, :, None])
        rgb = rgb * (1 - rust[:, :, None]) + rust_rgb_local * rust[:, :, None]
        rgb = rgb * (1 - dust[:, :, None] * .45) + \
            dust_rgb[None, None, :] * (dust[:, :, None] * .45)
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        roughness = (mat['roughness'] + rust * .22 + dust * .10
                     + field * .030 + brush * .010)
        metallic = mat['metallic'] * (1 - rust * .70)
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip(roughness * 255, 0, 255),
                           np.clip(metallic * 255, 0, 255)),
                          axis=2).astype(np.uint8)
        # Brushed normal from the vertical streak field; the observed patch is
        # only a weak secondary detail so no broad dents appear.
        bump_source = brush + detail * .35
        bump = bump_source - cv2.GaussianBlur(bump_source, (0, 0), 9.0)
        bump = cv2.GaussianBlur(bump, (0, 0), .8) / 255
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 8
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 8
        nx = np.clip(-dx * .55, -.08, .08)
        ny = np.clip(dy * .25, -.04, .04)
        nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255,
                             0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('steel_base', base), ('steel_mr', mr), ('steel_normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def make_brass_09(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != SOURCE_SIZE:
        raise ValueError('09 rust source dimensions changed; sampling rects need review')
    detail = _tile_detail(im, (610, 1110, 640, 1135))
    size = 1024
    rng = np.random.default_rng(20910)
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


def steel_texture_checks(base_path, mr_path):
    base = np.asarray(Image.open(base_path).convert('RGB'), dtype=np.float32)
    mr = np.asarray(Image.open(mr_path).convert('RGB'), dtype=np.float32) / 255
    tiles = {'steel_clean': np.s_[0:1024, 0:1024],
             'steel_weathered': np.s_[0:1024, 1024:2048],
             'steel_weld': np.s_[1024:2048, 0:1024],
             'steel_rusted': np.s_[1024:2048, 1024:2048]}
    values = {}
    for name, sl in tiles.items():
        rgb = base[sl]
        means = rgb.mean(axis=(0, 1))
        values[name] = {
            'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
            'metallic_mean': round(float(mr[sl][..., 2].mean()), 4),
            'chroma': round(float(means.max() - means.min()), 2),
            'rust_fraction': round(float(((rgb[..., 0] > rgb[..., 1] + 8) &
                                          (rgb[..., 1] > rgb[..., 2] + 3)).mean()), 4)}
    values['passed'] = bool(
        values['steel_clean']['metallic_mean'] > .85 and
        values['steel_clean']['roughness_mean'] < .45 and
        values['steel_clean']['chroma'] < 12 and
        values['steel_clean']['rust_fraction'] < .02 and
        values['steel_weathered']['metallic_mean'] > .80 and
        values['steel_weathered']['roughness_mean'] < .60 and
        values['steel_weld']['metallic_mean'] > .75 and
        values['steel_weld']['roughness_mean'] >= .42 and
        values['steel_rusted']['roughness_mean'] > .55 and
        values['steel_rusted']['metallic_mean'] < .80 and
        values['steel_rusted']['rust_fraction'] > .10)
    values['note'] = ('Tile means; reflection direction is not baked into base '
                      'colour. Rust/dust are diagnostically detected by colour.')
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
                             rgb[..., 0].mean() - rgb[..., 2].mean() > 25)}
    values['note'] = 'Brass drain tiles only; the steel is checked separately.'
    return values


def panel_image(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return image.crop((x0 + 10, y0 + 8, x0 + w // 3 - 10, y0 + h // 2 - 8))


def reference_mask_09(panel):
    """Approximate object mask for an uncalibrated reference panel.

    The panels have a white background, a grey rendered shadow and a label.
    The mask is a diagnostic, not a calibrated silhouette measurement.
    """
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = rgb.mean(axis=2)
    mask = (gray < 232).astype(np.uint8)
    h, w = mask.shape
    mask[int(h * .91):] = 0
    mask[:6] = 0
    mask[-6:] = 0
    mask[:, :6] = 0
    mask[:, -6:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
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
    'front': '参考与照片同为上法兰、右侧单管口、下部裙座四腿和底部排水器；参考未标定，只作外观诊断；照片侧法兰呈 3/4 视角，与对称四腿投影不一致，候选按对称正视解释',
    'back': '背面在原照不可见；按参考图推断为四腿对称、无第二个侧面法兰',
    'left': '参考左视把侧法兰画成正对相机；原照只看到一侧，左侧背面为推断',
    'right': '右视侧法兰面；法兰厚度、螺栓孔直径、孔深与背面细节仍待实测',
    'top': '顶视为上法兰 8 孔与中心通孔、侧管口和小排气塞；排气塞方位按照片，尺寸为推断',
    'bottom': '底视以四腿、排水器和方形吊架为中心；原照只看到两腿和部分吊架，背侧与内部为推断',
}


def make_report_09(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '09 分离过滤器 1 · 六视图参考（未标定）、原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = panel_image(reference_image, i)
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_09(panel))
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
        ref_mask = reference_mask_09(panel)
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
              '重要：参考图未标定，图中分数不是实测精度；原模型为单张重建，六视角审计记录到源视图细结构未通过。'
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
        ('detail-top-flange', '顶法兰下部、中心孔与 8 个螺栓孔'),
        ('detail-top-face', '顶法兰上表面 8 个螺栓孔'),
        ('detail-side-flange', '侧法兰、中心孔与 8 个螺栓孔'),
        ('detail-side-port', '侧管口小圆螺塞'),
        ('detail-vent', '顶部排气/丝堵'),
        ('detail-nameplate', '不锈钢铭牌（文字不可辨，未复制）'),
        ('detail-upper-seams', '上部两道绳状环焊缝'),
        ('detail-lower-seams', '下封头与裙座环焊缝'),
        ('detail-leg-gusset', '支腿、三角加强筋'),
        ('detail-foot', '支脚板与长圆孔'),
        ('detail-drain', '底部自动排水器正面'),
        ('detail-drain-mount', '排水器法兰、双螺母与底部封头'),
        ('detail-drain-cage', '排水器方管吊架'),
        ('detail-drain-bottom', '蓝管、黄铜锥体与排污口'),
        ('detail-surface', '拉丝不锈钢、焊缝与锈斑'),
        ('new-front', '整体正视（透明背景）'),
    )
    board = Image.new('RGB', (2000, 1650), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '09 分离过滤器 1 · 关键局部检查（新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate(items):
        x = (i % 8) * 250
        y = 55 + (i // 8) * 780
        path = out / 'renders' / f'{name}.png'
        image = _tile_image(path, (232, 660))
        tile = Image.new('RGBA', (242, 675), 'white')
        tile.alpha_composite(image, ((242 - image.width) // 2,
                                     (675 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 4, y + 30))
        draw.text((x + 6, y), label, font=small, fill='#27363c')
    draw.text((20, 1600), '这些局部图用于人工检查结构；不能证明未观察面、尺寸、装配关系、孔深、螺纹或内部通道。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    board = Image.new('RGB', (1800, 900), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '09 分离过滤器 1 · 隔离开口与结构检查',
              font=font, fill='#18242b')
    items = (('top-flange-only', '顶法兰：中心+8 孔', 'top_flange'),
             ('side-flange-only', '侧法兰：中心+8 孔', 'side_flange'),
             ('foot-slot-only', '支脚长圆孔', 'foot_slot'),
             ('drain-bore-only', '排水器通孔', 'drain_bore'),
             ('body-only', '封闭筒体（外观）', None))
    for i, (name, label, key) in enumerate(items):
        x = i * 360
        image = _tile_image(out / 'renders' / f'{name}.png', (330, 620))
        tile = Image.new('RGBA', (345, 640), 'white')
        tile.alpha_composite(image, ((345 - image.width) // 2,
                                     (640 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 8, 92))
        if key:
            check = checks[key]
            status = '通过' if check['passed'] else '未通过'
            text = f'{label}：{check["detected_openings"]}/{check["expected"]} {status}'
            color = '#1d5c3a' if check['passed'] else '#8a2b20'
        else:
            text = f'{label}（不参与孔数判定）'
            color = '#52646b'
        draw.text((x + 10, 48), text, font=small, fill=color)
    draw.text((20, 805), '透明背景隔离渲染；孔数由连通域检测。顶/侧法兰的中心孔与螺栓孔、支脚长圆孔和蓝管通孔为真实透明开口；'
                         '排水器本体为封闭外观件。孔数不能证明孔深、螺纹、内部通道或装配关系。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '09 分离过滤器 1 · 同光照材质对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型表面（顶点色）'),
                                       ('new-material-close', '新候选拉丝不锈钢 PBR 与黄铜'))):
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
    draw.text((20, 12), '09 分离过滤器 1 · 变光检查（同视角）',
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
    draw.text((20, 810), '灯光移动后高光和明暗应随视角变化；本图只检查反射响应，不证明现场光照方向。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '09 分离过滤器 1 · 金属反射变光对照',
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
        image = _tile_image(out / 'renders' / f'turn-{i:02d}.png', (220, 215))
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
        mask = reference_mask_09(panel)
        points = cv2.findNonZero(mask)
        if points is None:
            out[name] = None
            continue
        x, y, ww, hh = cv2.boundingRect(points)
        out[name] = {'bbox_px': [int(x), int(y), int(ww), int(hh)],
                     'width_over_height': round(ww / max(1, hh), 4)}
    out['proportion_note'] = (
        'Clean-photo alpha: total height about 1194 px, upper bulb max '
        'diameter about 300 px, shell diameter about 192 px, lower bulb about '
        '291 px, side flange about 220 px, top flange about 200 px. The '
        'candidate uses a total height of about 1.0 m and a shell radius of '
        '0.0804 m. These are pixel ratios and a model-library visual scale, '
        'not measurements. The six-view sheet is an uncalibrated visual-match '
        'reference, not the same photograph.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=9)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views and surfaces only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 9:
        p.error('this entry supports item 09 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/09-separator-draft1' if a.draft
                        else 'outputs/09分离过滤器1-六视图重建-v1')
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
    config_path = HERE / 'separator09_config.json'
    worker_path = HERE / 'mesh_rebuild_09_worker.py'
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
            required = [a.out / 'model.blend', a.out / 'steel_base.png',
                        a.out / 'steel_mr.png', a.out / 'steel_normal.png',
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
                             ('detail-top-flange', 'detail-top-face', 'detail-vent', 'detail-side-flange',
                              'detail-side-port', 'detail-nameplate',
                              'detail-upper-seams', 'detail-lower-seams',
                              'detail-leg-gusset', 'detail-foot', 'detail-drain',
                              'detail-drain-mount', 'detail-drain-cage',
                              'detail-drain-bottom', 'detail-surface',
                              'top-flange-only', 'side-flange-only',
                              'foot-slot-only', 'drain-bore-only', 'body-only',
                              'new-light-alternate', 'new-material-close',
                              'new-metal-alternate', 'old-material-close')]
                required += [a.out / 'renders' / f'turn-{i:02d}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_steel_atlas_09(Path(row['rust']), a.out, config['material_profile'])
    textures.update(make_brass_09(Path(row['rust']), a.out, config['material_profile']))
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
    views = make_report_09(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(9)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    if not a.draft:
        make_turntable(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('top_flange', 'top-flange-only.png', 9),
                ('side_flange', 'side-flange-only.png', 9),
                ('foot_slot', 'foot-slot-only.png', 1),
                ('drain_bore', 'drain-bore-only.png', 1)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    material_checks = steel_texture_checks(a.out / 'steel_base.png', a.out / 'steel_mr.png')
    brass_checks = brass_texture_checks(a.out / 'brass_base.png', a.out / 'brass_mr.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'separator_body': 'Separator body' in names,
        'seam_welds': sum(name in {'Upper bulb seam A', 'Upper bulb seam B',
                                   'Bulb to shell seam', 'Shell to lower bulb seam',
                                   'Lower bulb to skirt seam', 'Skirt to bottom head seam'}
                          for name in names),
        'seam_welds_expected': 6,
        'top_flange_parts': sum(name.startswith('Top flange') or
                                name.startswith('Top raised face') or
                                name.startswith('Top pipe') for name in names),
        'top_flange_parts_expected': 6,
        'top_flange_bolt_holes': config['top_flange']['bolt_holes'],
        'vent_parts': sum(name.startswith('Vent ') for name in names),
        'vent_parts_expected': 3,
        'side_nozzle_parts': sum(name.startswith('Side ') and
                                 not name.endswith('red arrow') for name in names),
        'side_nozzle_parts_expected': 8,
        'side_flange_bolt_holes': config['side_nozzle']['bolt_holes'],
        'nameplate_parts': sum(name.startswith('Nameplate') for name in names),
        'nameplate_parts_expected': 6,
        'red_arrows': sum(name.endswith('red arrow') for name in names),
        'red_arrows_expected': 2,
        'skirt': 'Support skirt' in names,
        'bottom_head': 'Bottom head' in names,
        'legs': sum(name.startswith('Support leg') for name in names),
        'legs_expected': 4,
        'gussets': sum(name.startswith('Leg gusset') for name in names),
        'gussets_expected': 4,
        'foot_pads': sum(name.startswith('Foot pad') for name in names),
        'foot_pads_expected': 4,
        'drain_mount_parts': sum(name.startswith('Drain neck') or
                                 name.startswith('Drain mount') for name in names),
        'drain_mount_parts_expected': 4,
        'drain_body': 'Drain body' in names,
        'drain_end_caps': sum(name.startswith('Drain end cap') for name in names),
        'drain_end_caps_expected': 2,
        'drain_bands': sum(name.startswith('Drain band') for name in names),
        'drain_bands_expected': 2,
        'drain_sight_parts': sum(name in {'Drain sight ring', 'Drain sight glass',
                                          'Drain brass center', 'Drain brass lever'}
                                 for name in names),
        'drain_sight_parts_expected': 4,
        'drain_spigot_parts': sum(name in {'Drain spigot nut', 'Drain spigot cone',
                                           'Drain blue collar', 'Drain blue tube'}
                                  for name in names),
        'drain_spigot_parts_expected': 4,
        'drain_elbow_parts': sum(name.startswith('Drain elbow') for name in names),
        'drain_elbow_parts_expected': 3,
        'drain_steel_tube': 'Drain steel tube' in names,
        'drain_frame_parts': sum(name.startswith('Drain frame') for name in names),
        'drain_frame_parts_expected': 12,
        'bore_liners': sum(name.startswith('Bore liner') for name in names),
        'bore_liners_expected': 2,
        'isolated_openings': hole_checks,
        'note': ('Part presence proves only that a feature was modelled, not its size, '
                 'fit or hidden faces. The rear/back topology, flange faces, bore depths, '
                 'drain internals, nameplate text, vent size and all hidden faces are '
                 'partly inferred.'),
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
        structural['separator_body'] and structural['seam_welds'] == 6 and
        structural['top_flange_parts'] == 6 and structural['vent_parts'] == 3 and
        structural['side_nozzle_parts'] == 8 and
        structural['nameplate_parts'] == 6 and structural['red_arrows'] == 2 and
        structural['skirt'] and structural['bottom_head'] and
        structural['legs'] == 4 and structural['gussets'] == 4 and
        structural['foot_pads'] == 4 and structural['drain_mount_parts'] == 4 and
        structural['drain_body'] and structural['drain_end_caps'] == 2 and
        structural['drain_bands'] == 2 and structural['drain_sight_parts'] == 4 and
        structural['drain_spigot_parts'] == 4 and structural['drain_elbow_parts'] == 3 and
        structural['drain_steel_tube'] and structural['drain_frame_parts'] == 12 and
        structural['bore_liners'] == 2)
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 75 and
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
        'version': config['version'], 'number': 9, 'asset_id': row['asset_id'],
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
            'The old model-library source IoU 0.9177 only measures one silhouette; '
            'the audit records one filled source hole and the single-image mesh '
            'fuses the top flange, side nozzle, leg and drain details. This rebuild '
            'therefore does not reuse the old mesh.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'steel_texture_checks': material_checks,
        'brass_texture_checks': brass_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The supplied clean photo is a product 3/4 view whose side-flange ellipse '
            'is not consistent with the symmetric four-leg projection; the candidate '
            'uses the symmetric canonical interpretation (side nozzle along +Y). The '
            'stored single-image source camera is at +X. The framing distance was '
            'fitted to the source-alpha height and centred laterally, so the score is '
            'diagnostic only and is not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is an uncalibrated visual-match '
                       'reference of the same visible separator design, not a '
                       'calibrated image set. It is used for the hidden/back-side '
                       'topology and to reveal gross differences; the candidate '
                       'follows the photo for observed structure.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view sheet is uncalibrated and not a measured image set; the reference segmentation is a diagnostic.',
            'The working total height (about 1.0 m), shell radius (0.0804 m) and all other sizes follow photo pixel ratios and the model-library visual scale; they are not measured dimensions.',
            'The photo is a 3/4 product view: the side-flange ellipse and the symmetric four-leg projection cannot both be exactly true. The candidate follows the symmetric canonical interpretation.',
            'The rear/back topology, leg azimuths, gusset form, foot-pad size, flange thickness, bolt-hole diameters, bolt circle, raised-face profile and bore depth are inferred.',
            'The automatic-drain model, internal mechanism, thread forms, sight-glass construction, brass lever and pneumatic hose routing are inferred; the blue hose is an observed local fitting, not a catalogue-verified part.',
            'The drain-cage rod diameter, attachment bolts and mount-bracket shape are inferred.',
            'The nameplate text is illegible in the source and deliberately omitted; the plate size, inset and rivets are approximate.',
            'The vent/plug azimuth, hex size and top-flange bolt phase are inferred.',
            'The nozzle bores are open through each flange and pipe but stop at the closed shell; internal separator connections are not modelled or verified.',
            'The bottom drain passage is open through the brass/blue fitting and is capped at the closed drain body; the internal drain passage is not modelled or verified.',
            'The four feet are modelled as flat supports, but floor contact, stability, site scale and the settled pose are not established.',
            'Rust, dust and scratches are synthesized from small observed patches; they are not a photograph of hidden faces.',
            'Isolated opening counts prove only transparent openings in the isolated renders; they do not prove hole depth, threads, fits or assembly.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the photo, especially the top and side flange faces, the four legs and the drain assembly.',
            'Confirm the back-side topology and whether a second side nozzle or other port exists; it is not modelled.',
            'Confirm flange thickness, bolt-hole diameter, bolt circle, raised-face profile and bore depth against a measured or higher-resolution sample.',
            'Confirm the leg azimuths, gusset form, foot-pad size and slot dimensions; the photo shows only the two near legs.',
            'Confirm the exact automatic-drain model, its internal mechanism, thread forms, sight-glass construction and hose routing.',
            'Confirm the drain-cage rods, attachment bolts and the mount bracket; they are inferred from the photo and the uncalibrated sheet.',
            'Confirm the nameplate text and the red-arrow marking sizes; the text is omitted.',
            'Confirm the vent/plug size and azimuth and the top-flange bolt phase.',
            'Confirm the internal separator element and whether the nozzle bores connect through the vessel.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 10,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_09.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 9, 'name': row['file'], 'asset_id': row['asset_id'],
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
        'next_number': 10,
        'manual_only': True,
        'notes': ('The user six-view sheet is an uncalibrated visual-match reference; '
                  'the old model and items 01-08 were not modified. Candidate is '
                  'manual_only. Browser review page and visual acceptance are pending '
                  'unless recorded separately.'),
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '09-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
