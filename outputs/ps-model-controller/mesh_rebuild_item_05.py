"""Build the review-only vertical stainless storage-tank candidate (item 05).

The shape follows the original 3/4 rust photograph for proportions and the
user-supplied, uncalibrated six-view sheet for construction details.  The
working diameter follows the model-library visual scale and is not a measured
dimension.  The candidate never replaces the library mesh and is never used by
the automatic placement pipeline.  A new output directory is required.
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


def make_atlas_05(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != (941, 1672):
        raise ValueError('05 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((420, 120, 540, 240), (250, 400, 370, 520),
                (600, 900, 720, 1020), (800, 1450, 860, 1530))]
    rng = np.random.default_rng(20505)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    # Base colours are neutral stainless steel; all highlight direction is left
    # to the live lighting.  Rust and brushing are synthesized from small
    # observed patches of the supplied rust image, not from hidden faces.
    specs = [((172, 173, 171), profile['bright_stainless'], 2.6),
             ((146, 147, 146), profile['brushed_stainless'], 6.0),
             ((124, 119, 111), profile['weathered_stainless'], 4.5),
             ((112, 103, 92), profile['stained_steel'], 3.0)]
    for i, (color, mat, brush_strength) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((56, 56), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((560, 560), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_LINEAR)
        speck = cv2.resize(rng.random((1000, 1000), dtype=np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        # Vertical brushing: a narrow random field stretched across the tile.
        brush = cv2.resize(rng.normal(size=(size, 24)).astype(np.float32),
                           (size, size), interpolation=cv2.INTER_CUBIC)
        brush = cv2.GaussianBlur(brush, (0, 0), 2.6)
        brush *= brush_strength / max(1e-6, float(brush.std()))
        field = .45 * (coarse - .5) + .35 * (fine - .5) + .20 * (speck - .5)
        rust_amount = mat['rust_amount']
        rust = np.clip((coarse - (.82 - rust_amount * .85)) * 8, 0, 1) * rust_amount
        streak = np.clip((-brush - 1.6) * .28, 0, 1) * rust_amount * 2.2
        rust = np.clip(rust + streak, 0, 1)
        rgb = np.asarray(color, np.float32)[None, None, :] * (1 + field[:, :, None] * .055)
        rgb += detail[:, :, None] * .22 + brush[:, :, None] * .55
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
        bump = cv2.GaussianBlur(detail + brush * 2.2, (0, 0), .6) / 255
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


def reference_mask_05(panel):
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    border = np.concatenate((rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3),
                             rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)))
    bg = np.median(border, axis=0)
    diff = np.max(np.abs(rgb - bg[None, None, :]), axis=2)
    edges = cv2.Canny(gray, 18, 60)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(gray.shape, np.uint8)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [contour], -1, 1, -1)
    add = ((diff > 40) & (cv2.dilate(out, np.ones((9, 9), np.uint8)) > 0)).astype(np.uint8)
    out = np.maximum(out, add)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] >= max(500, areas.max() * .05)]
        out = np.isin(labels, keep).astype(np.uint8)
    return out


def make_report_05(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    w, h = reference_image.size
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '05 立式不锈钢储罐 · 六视图、'
                        '原模型与新候选对照', font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = reference_image.crop(((i % 3) * w // 3, (i // 3) * h // 2,
                                      (i % 3 + 1) * w // 3, (i // 3 + 1) * h // 2))
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_05(panel))
        actual = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.getchannel('A')) >= 128)
        iou = (float(np.logical_and(expected, actual).sum() /
                     max(1, np.logical_or(expected, actual).sum()))
               if expected is not None and actual is not None else None)
        old_new = (float(np.logical_and(old_fit, actual).sum() /
                         max(1, np.logical_or(old_fit, actual).sum()))
                   if old_fit is not None and actual is not None else None)
        note = '按最长边等比对齐；参考图未标定，仅供发现差异'
        if name in ('left', 'right'):
            note += ('；侧面板的上下侧管口与正视图/原照不一致，'
                     '按推断处理')
        if name in ('top', 'bottom'):
            note += '；该参考面板未标定，不能作为尺寸依据'
        rows.append({'view': name,
                     'uncalibrated_reference_mask_iou': round(iou, 4) if iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4) if old_new is not None else None,
                     'note': note})
        draw.text((x + 12, y), f'{name.upper()}  参考图 / 原模型 / '
                               '新候选', font=small, fill='#27363c')
        ref_mask = reference_mask_05(panel)
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
                  (f'参考粗略轮廓 IoU：{iou:.3f} · '
                   f'原/新 IoU：{old_new:.3f}')
                  if iou is not None and old_new is not None else '轮廓不可比较',
                  font=small, fill='#52646b')
    draw.text((20, 1255),
              '参考图未标定，侧管口、顶部小管口与正视/原照存在镜像和数量不一致；'
              '本次以原照可核对的正面双管口、顶部人孔和三支腿为主，其余标为推断。'
              '图中分数不是实测精度，不能据此判定结构合格。',
              font=small, fill='#52646b')
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
    board = Image.new('RGB', (1800, 1290), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '05 立式不锈钢储罐 · 关键局部'
                        '检查（新候选）', font=font, fill='#18242b')
    items = (
        ('detail-manway', '人孔盖、卡箍与提手'),
        ('detail-clamps', '六个摆动卡箍与吊环'),
        ('detail-top-nozzles', '四个顶部管口与法兰'),
        ('detail-front-port', '上部前管口与暗孔'),
        ('detail-lower-port', '下部前管口与暗孔'),
        ('detail-legs', '三支板式支腿与上托座'),
        ('detail-foot', '支脚板与安装长槽'),
        ('detail-drain', '底部排污口与阀体'),
        ('detail-seam', '上下封头焊缝带'),
        ('detail-surface', '拉丝不锈钢、锈痕与反射'),
    )
    for i, (name, label) in enumerate(items):
        x = (i % 5) * 360
        y = 55 + (i // 5) * 600
        image = _tile_image(out / 'renders' / f'{name}.png', (325, 500))
        tile = Image.new('RGBA', (340, 510), 'white')
        tile.alpha_composite(image, ((340 - image.width) // 2,
                                     (510 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 10, y + 30))
        draw.text((x + 14, y), label, font=small, fill='#27363c')
    draw.text((20, 1255), '这些局部图用于人工检查结构；'
                          '不能证明未观察面、尺寸、'
                          '装配关系或内部通道。', font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    board = Image.new('RGB', (1800, 780), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '05 立式不锈钢储罐 · 隔离开口'
                        '检查', font=font, fill='#18242b')
    items = (('top-nozzles-only', '四个顶部管口通孔', 'top_nozzles'),
             ('front-ports-only', '上下前管口通孔', 'front_ports'),
             ('drain-only', '底部排污通孔', 'drain'),
             ('feet-only', '三个支脚板长槽', 'foot_slots'))
    for i, (name, label, key) in enumerate(items):
        x = i * 450
        image = _tile_image(out / 'renders' / f'{name}.png', (390, 560))
        tile = Image.new('RGBA', (410, 570), 'white')
        tile.alpha_composite(image, ((410 - image.width) // 2,
                                     (570 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 92))
        check = checks[key]
        draw.text((x + 24, 48), f'{label}：检测 '
                                f'{check["detected_openings"]} / 预期 {check["expected"]}'
                  + ('  通过' if check['passed'] else '  未通过'), font=small,
                  fill='#1d5c3a' if check['passed'] else '#8a2b20')
    draw.text((20, 720), '透明背景隔离渲染；孔数由连通域检测。'
                         '顶部管口、前管口和排污口为开放通孔；'
                         '支脚板长槽为推断的安装孔。'
                         '孔数不能证明孔深、螺纹、内部通道或装配关系。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '05 立式不锈钢储罐 · 同光照材质'
                        '对照（原模型 / 新候选）', font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型'
                                        '金属表面（顶点色）'),
                                       ('new-material-close', '新候选 '
                                        'PBR 拉丝不锈钢'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (840 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), '同一中性光与相机；高光来自实时照明，'
                         '不烘焙在底色中。原模型为单图重建的'
                         '像素/顶点色，不代表旧模型整体通过。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '05 立式不锈钢储罐 · 变光检查'
                        '（同视角）', font=font, fill='#18242b')
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
    draw.text((20, 12), '05 立式不锈钢储罐 · 金属反射'
                        '变光对照', font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光 A'),
                                       ('new-metal-alternate', '主光 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2,
                                     (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '金属高光随灯光位置移动；'
                         '不能据此推断现场光源方向或强度。',
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
    w, h = image.size
    out = {}
    for i, name in enumerate(VIEWS):
        panel = image.crop(((i % 3) * w // 3, (i // 3) * h // 2,
                            (i % 3 + 1) * w // 3, (i // 3 + 1) * h // 2))
        mask = reference_mask_05(panel)
        points = cv2.findNonZero(mask)
        if points is None:
            out[name] = None
            continue
        x, y, ww, hh = cv2.boundingRect(points)
        out[name] = {'bbox_px': [int(x), int(y), int(ww), int(hh)],
                     'width_over_height': round(ww / max(1, hh), 4)}
    out['proportion_note'] = (
        'Original 3/4 photo: body diameter about 830 px, total height about '
        '1658 px (width/height about 0.50); the candidate uses diameter 0.56 m '
        'and total height 1.118 m (0.501). The six-view front panel is about '
        '0.558 wide/height, i.e. wider than the source photo. The candidate '
        'follows the source-photo proportion. All values are pixel ratios, '
        'not calibrated dimensions.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=5)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views and surfaces only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 5:
        p.error('this entry supports item 05 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/05-tank-draft1' if a.draft
                        else 'outputs/05储罐-六视图重建-v1')
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
    config_path = HERE / 'tank05_config.json'
    worker_path = HERE / 'mesh_rebuild_05_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True, text=True,
                             check=True).stdout.splitlines()[0]
    signature_input = {'reference': digest(a.reference), 'mesh': row['mesh_sha256'],
                       'rust': row['rust_sha256'], 'source_alpha': digest(alpha_path),
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
                        a.out / '六视图-新旧对照.jpg', a.out / 'hashes.json']
            if not a.draft:
                required += [a.out / '局部检查.jpg', a.out / '结构开口-隔离检查.jpg',
                             a.out / '材质-同光照对照.jpg', a.out / '材质-变光检查.jpg',
                             a.out / '金属反射-变光对照.jpg', a.out / '转台-24方向.jpg']
                required += [a.out / 'renders' / f'{which}-{view}.png'
                             for which in ('old', 'new') for view in VIEWS]
                required += [a.out / 'renders' / f'{name}.png' for name in
                             ('detail-manway', 'detail-clamps', 'detail-top-nozzles',
                              'detail-front-port', 'detail-lower-port', 'detail-legs',
                              'detail-foot', 'detail-drain', 'detail-seam',
                              'detail-surface', 'top-nozzles-only', 'front-ports-only',
                              'drain-only', 'feet-only', 'new-light-alternate',
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
    textures = make_atlas_05(Path(row['rust']), a.out, config['material_profile'])
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
    views = make_report_05(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(5)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    if not a.draft:
        make_turntable(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('top_nozzles', 'top-nozzles-only.png', 4),
                ('front_ports', 'front-ports-only.png', 2),
                ('drain', 'drain-only.png', 1),
                ('foot_slots', 'feet-only.png', 3)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    material_checks = material_texture_checks(a.out / 'metallic_roughness.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'tank_body': 'Tank body' in names,
        'seam_bands': sum(name.startswith('Shell seam') for name in names),
        'seam_bands_expected': 2,
        'legs': sum(name.startswith('Leg ') and name.endswith(' plate') for name in names),
        'legs_expected': 3,
        'leg_brackets': sum(name.startswith('Leg ') and name.endswith(' bracket')
                            for name in names),
        'leg_brackets_expected': 3,
        'foot_plates': sum(name.startswith('Foot plate ') for name in names),
        'foot_plates_expected': 3,
        'manway_neck': 'Manway neck' in names,
        'manway_flange': 'Manway flange' in names,
        'manway_gasket': 'Manway gasket' in names,
        'manway_lid': 'Manway lid' in names,
        'manway_hinges': sum(name.startswith('Manway hinge ') for name in names),
        'manway_hinges_expected': 2,
        'manway_handle_parts': sum(name.startswith('Manway handle ') for name in names),
        'manway_handle_parts_expected': 3,
        'manway_clamp_lugs': sum(name.startswith('Manway clamp ') and name.endswith(' lug')
                                 for name in names),
        'manway_clamp_lugs_expected': 6,
        'manway_clamp_bolts': sum(name.startswith('Manway clamp ') and name.endswith(' bolt')
                                  for name in names),
        'manway_clamp_bolts_expected': 6,
        'manway_clamp_eyes': sum(name.startswith('Manway clamp ') and name.endswith(' eye')
                                 for name in names),
        'manway_clamp_eyes_expected': 6,
        'top_nozzles': sum(name in {f'Top nozzle {i}' for i in range(1, 5)} for name in names),
        'top_nozzles_expected': 4,
        'top_flanges': sum(name.startswith('Top nozzle') and name.endswith(' flange')
                           for name in names),
        'top_flanges_expected': 2,
        'top_ferrule_lips': sum(name.startswith('Top nozzle') and name.endswith(' lip')
                                for name in names),
        'top_ferrule_lips_expected': 2,
        'top_bolts': sum(name.startswith('Top nozzle') and ' bolt ' in name for name in names),
        'top_bolts_expected': 12,
        'side_ferrules': sum(name.startswith('Side ferrule ') and not name.endswith(' cap')
                             for name in names),
        'side_ferrules_expected': 2,
        'side_ferrule_caps': sum(name.startswith('Side ferrule ') and name.endswith(' cap')
                                 for name in names),
        'side_ferrule_caps_expected': 2,
        'front_ports': sum(name in {'Front port upper', 'Front port lower'} for name in names),
        'front_ports_expected': 2,
        'front_port_hex_bosses': sum(name.startswith('Front port ') and name.endswith(' hex boss')
                                     for name in names),
        'front_port_hex_bosses_expected': 2,
        'front_port_flanges': sum(name.startswith('Front port ') and name.endswith(' flange')
                                  for name in names),
        'front_port_flanges_expected': 2,
        'open_bore_sleeves': sum(name.endswith(' sleeve') for name in names),
        'open_bore_sleeves_expected': 7,
        'drain_pipe': 'Drain pipe' in names,
        'drain_flange': 'Drain flange' in names,
        'drain_valve_body': 'Drain valve body' in names,
        'drain_collar': 'Drain collar' in names,
        'drain_valve_stem': 'Drain valve stem' in names,
        'drain_valve_bar': 'Drain valve bar' in names,
        'isolated_openings': hole_checks,
        'note': 'Part presence proves only that a feature was modelled, not its size, fit or hidden faces. The lower front port and the side ferrules are inferred.',
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
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] >= 60 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': bool(
            structural['tank_body'] and structural['seam_bands'] == 2 and
            structural['legs'] == 3 and structural['leg_brackets'] == 3 and
            structural['foot_plates'] == 3 and structural['manway_neck'] and
            structural['manway_flange'] and structural['manway_gasket'] and
            structural['manway_lid'] and structural['manway_hinges'] == 2 and
            structural['manway_handle_parts'] == 3 and
            structural['manway_clamp_lugs'] == 6 and
            structural['manway_clamp_bolts'] == 6 and
            structural['manway_clamp_eyes'] == 6 and
            structural['top_nozzles'] == 4 and structural['top_flanges'] == 2 and
            structural['top_ferrule_lips'] == 2 and structural['top_bolts'] == 12 and
            structural['side_ferrules'] == 2 and structural['side_ferrule_caps'] == 2 and
            structural['front_ports'] == 2 and structural['front_port_hex_bosses'] == 2 and
            structural['front_port_flanges'] == 2 and
            structural['open_bore_sleeves'] == 7 and structural['drain_pipe'] and
            structural['drain_flange'] and structural['drain_valve_body'] and
            structural['drain_collar'] and structural['drain_valve_stem'] and
            structural['drain_valve_bar']),
        'material_tiles': bool(material_checks['passed']),
        'metal_reflection': bool(reflection.get('changed', False)),
        'reference_diagnostic_only': True,
    }
    if a.draft:
        passed['isolated_openings'] = 'not_run'
        passed['metal_reflection'] = 'not_run'
    report = {
        'version': config['version'], 'number': 5, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'user_six_view_reference',
        'method': 'parametric_part_rebuild_from_source_photo_proportions_and_uncalibrated_six_view',
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old mesh source-view IoU 0.969 only measures one silhouette; narrow '
            'structure recall is 0.4615 and the record fails. The single-image mesh '
            'merges the three legs, loses the small nozzles and clamps, and has no real '
            'openings, so this rebuild does not reuse it.'),
        'catalog_evidence': config['proportion_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'material_texture_checks': material_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The source photo is a single 3/4 rust image with a different scale and '
            'pose; the new-source framing distance was scaled by the new/old span. '
            'The score is diagnostic only and is not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is uncalibrated. Its front panel is wider '
                       'than the original photo, and its left/right and top panels '
                       'repeat or mirror nozzles inconsistently. The candidate follows '
                       'the original photo proportions and the structures that can be '
                       'checked in the front/top/bottom panels.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view reference is uncalibrated and internally inconsistent; absolute size, camera and proportions are estimates.',
            'The working diameter (0.56 m) follows the model-library visual scale and is not a measured dimension of the photographed tank.',
            'The reference front panel is proportionally wider than the original 3/4 photo; the rebuild follows the photo, not the collage width.',
            'The lower front port, the two upper side ferrules and the foot-plate mounting slots are inferred from partial panels; they are not measured.',
            'The reference repeats or mirrors upper/lower side nozzles; only the two front-facing ports are supported by the original photo, and the side ferrules are kept small and capped.',
            'The manway clamp count (6), lid detail, hinge positions and nozzle/flange layouts are read from the top panel and are partly inferred.',
            'Nozzle bores are modelled as open thick-walled tubes with dark inner walls; bore depth, threads, wall piercing and internal connections are inferred.',
            'Paint/rust distribution is synthesized from small patches of the supplied rust image; it is not a photograph of hidden faces.',
            'Isolated opening counts prove only transparent openings in the render; they do not prove hole depth, threads, fits or assembly.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, floor contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the reference by eye, especially the manway clamps, the top nozzle layout and the leg brackets.',
            'Decide whether the lower front port and the two capped side ferrules really exist on the photographed unit; the six-view sheet is mirror-inconsistent there.',
            'Confirm the 3-leg layout, leg angles, foot-plate slots and leg-top bracket form against a measured sample.',
            'Confirm the front-port heights and bores, the bottom drain/valve form and whether the tank wall is pierced as modelled.',
            'Check the manway clamp count and swing-bolt orientation, the lid handle and the gasket gap.',
            'Verify coordinate, scale, stability, floor contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 6,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_05.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 5, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'textures': textures,
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed, 'material_checks': material_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'pending_review': report['pending_review'],
        'next_number': 6,
        'notes': 'The old model and items 01-04 were not modified; candidate is manual_only. Browser review page was not part of this run unless recorded separately.'
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '05-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
