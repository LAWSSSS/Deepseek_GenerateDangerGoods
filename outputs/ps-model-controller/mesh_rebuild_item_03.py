"""Build the review-only ASAHI P210 pillow-block candidate (item 03).

The candidate is derived from the user-supplied, uncalibrated six-view image.
Public UCP210/P210/UC210 catalog dimensions are used only as a proportion
sanity check.  The candidate never replaces the library mesh and is never used
by the automatic placement pipeline.  A new output directory is required.
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
REFERENCE_SHA256 = '5fa3a3c669a60e8f4c2deb0e54d6f484e088e9e78c683bde22f9cf76233731a6'


def find_reference():
    folder = Path.home() / 'Desktop' / '六视图'
    if folder.is_dir():
        for path in sorted(folder.glob('*.png')):
            if digest(path) == REFERENCE_SHA256:
                return path
    raise FileNotFoundError('registered six-view reference not found')


def make_atlas_03(source, out, profile):
    im = Image.open(source).convert('RGBA')
    if im.size != (1478, 1064):
        raise ValueError('03 rust source dimensions changed; sampling rects need review')
    patches = [_tile_detail(im, rect) for rect in
               ((850, 220, 940, 300), (420, 420, 500, 500),
                (555, 305, 635, 385), (1240, 890, 1310, 960))]
    rng = np.random.default_rng(20403)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [((42, 83, 126), profile['blue_paint']['roughness'],
              profile['blue_paint']['metallic'], profile['blue_paint']['rust_amount']),
             ((182, 184, 186), profile['bearing_steel']['roughness'],
              profile['bearing_steel']['metallic'], profile['bearing_steel']['rust_amount']),
             ((140, 142, 145), profile['dark_steel']['roughness'],
              profile['dark_steel']['metallic'], profile['dark_steel']['rust_amount']),
             ((110, 85, 65), profile['rusted_cast']['roughness'],
              profile['rusted_cast']['metallic'], profile['rusted_cast']['rust_amount'])]
    detail_scale = (.32, .18, .20, .35)
    for i, (color, roughness, metal, rust_amount) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((120, 120), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        specks = cv2.resize(rng.random((400, 400), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_LINEAR)
        rust = np.clip((coarse - (.79 - rust_amount)) * 5, 0, 1) * rust_amount * 3
        rgb = np.asarray(color, np.float32)[None, None, :] + detail[:, :, None] * detail_scale[i]
        rgb = rgb * (1 - rust[:, :, None]) + np.array((118, 72, 41), np.float32) * rust[:, :, None]
        rgb += ((specks - .5) * 3)[:, :, None]
        yy, xx = divmod(i, 2)
        sl = np.s_[yy * size:(yy + 1) * size, xx * size:(xx + 1) * size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip((roughness + rust * .12 + detail / 1000) * 255, 0, 255),
                           np.clip((metal * (1 - rust * .65)) * 255, 0, 255)), axis=2).astype(np.uint8)
        bump = cv2.GaussianBlur(detail, (0, 0), .6) / 255
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3) / 8
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3) / 8
        nx = np.clip(-dx * .8, -.12, .12)
        ny = np.clip(dy * .8, -.12, .12)
        nz = np.sqrt(np.maximum(0, 1 - nx * nx - ny * ny))
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2) * .5 + .5) * 255, 0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('base_color', base), ('metallic_roughness', mr), ('normal', normal)):
        path = out / (name + '.png')
        Image.fromarray(arr).save(path)
        paths[name] = str(path)
    return paths


def material_texture_checks(path):
    mr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    tiles = {'blue_paint': np.s_[0:1024, 0:1024], 'bearing_steel': np.s_[0:1024, 1024:2048],
             'dark_steel': np.s_[1024:2048, 0:1024], 'rusted_cast': np.s_[1024:2048, 1024:2048]}
    values = {name: {'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
                     'metallic_mean': round(float(mr[sl][..., 2].mean()), 4)}
              for name, sl in tiles.items()}
    values['passed'] = (values['bearing_steel']['roughness_mean'] <
                        values['blue_paint']['roughness_mean'] and
                        values['blue_paint']['metallic_mean'] < .10 and
                        values['dark_steel']['metallic_mean'] > .75 and
                        values['rusted_cast']['roughness_mean'] > .70)
    values['note'] = 'Tile means; reflection direction is not baked into base color.'
    return values


def reference_mask_03(panel):
    rgb = np.asarray(panel.convert('RGB')).astype(np.int16)
    h, w, _ = rgb.shape
    border = np.concatenate((rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3),
                             rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)))
    bg = np.median(border, axis=0)
    diff = np.max(np.abs(rgb - bg[None, None, :]), axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    bright = (rgb.max(axis=2) > 200) & (diff > 18)
    mask = ((sat > 20) | bright | (diff > 45)).astype(np.uint8)
    mask[int(h * .91):] = 0
    mask[:5] = 0
    mask[-5:] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(30, stats[1:, cv2.CC_STAT_AREA].max() * .002))
    keep = keep[keep != 0]
    return np.isin(labels, keep).astype(np.uint8)


def make_report_03(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    w, h = reference_image.size
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '03 P210 轴承座 · 六视图、'
                        '原模型与新候选对照',
              font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = reference_image.crop(((i % 3) * w // 3, (i // 3) * h // 2,
                                      (i % 3 + 1) * w // 3, (i // 3 + 1) * h // 2))
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask_03(panel))
        actual = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.getchannel('A')) >= 128)
        iou = (float(np.logical_and(expected, actual).sum() /
                     max(1, np.logical_or(expected, actual).sum()))
               if expected is not None and actual is not None else None)
        old_new = (float(np.logical_and(old_fit, actual).sum() /
                         max(1, np.logical_or(old_fit, actual).sum()))
                   if old_fit is not None and actual is not None else None)
        note = ('按最长边等比对齐；'
                '参考图未标定，仅供发现差异')
        if name in ('top', 'bottom'):
            note += ('；该参考面板是斜视展示，'
                     '不能与正交渲染精确匹配')
        rows.append({'view': name,
                     'uncalibrated_reference_mask_iou': round(iou, 4) if iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4) if old_new is not None else None,
                     'note': note})
        draw.text((x + 12, y), f'{name.upper()}  参考图 / 原模型 / '
                               '新候选', font=small, fill='#27363c')
        ref_mask = reference_mask_03(panel)
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
            tile.alpha_composite(image, ((190 - image.width) // 2, (495 - image.height) // 2))
            board.paste(tile.convert('RGB'), (x + 5 + j * 195, y + 32))
        draw.text((x + 12, y + 548),
                  (f'参考粗略轮廓 IoU：{iou:.3f} · '
                   f'原/新 IoU：{old_new:.3f}')
                  if iou is not None and old_new is not None else
                  '轮廓不可比较',
                  font=small, fill='#52646b')
    draw.text((20, 1255),
              '参考图未标定。顶/底参考面板'
              '为斜视展示；图中分数不是'
              '实测精度，不能据此判定'
              '结构合格。',
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
    board = Image.new('RGB', (1800, 1260), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '03 轴承座 · 关键局部'
                        '检查（新候选）', font=font, fill='#18242b')
    items = (
        ('detail-bearing', '轴承内圈、外圈与通孔'),
        ('detail-set-screw', '紧定螺钉与径向孔'),
        ('detail-zerk', '顶部油嘴'),
        ('detail-label-asahi', 'ASAHI 铸字'),
        ('detail-label-p210', 'P210 铸字'),
        ('detail-slot', '安装槽与凸台'),
        ('detail-bottom', '底部拱与加工垫'),
        ('detail-surface', '漆面、锈斑与金属反射'),
    )
    for i, (name, label) in enumerate(items):
        x = (i % 4) * 450
        y = 55 + (i // 4) * 590
        image = _tile_image(out / 'renders' / f'{name}.png', (410, 500))
        tile = Image.new('RGBA', (420, 510), 'white')
        tile.alpha_composite(image, ((420 - image.width) // 2, (510 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 10, y + 30))
        draw.text((x + 14, y), label, font=small, fill='#27363c')
    draw.text((20, 1225), '这些局部图用于人工'
                          '检查结构；不能证明'
                          '未观察面、尺寸或'
                          '装配关系。', font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    board = Image.new('RGB', (1800, 760), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '03 轴承座 · 隔离开口'
                        '检查', font=font, fill='#18242b')
    items = (('bearing-only', '轴承内圈通孔', 'bearing'),
             ('base-only', '底板安装槽', 'base'),
             ('housing-only', '壳体中心孔', 'housing'))
    for i, (name, label, key) in enumerate(items):
        x = i * 600
        image = _tile_image(out / 'renders' / f'{name}.png', (500, 560))
        tile = Image.new('RGBA', (530, 570), 'white')
        tile.alpha_composite(image, ((530 - image.width) // 2, (570 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 92))
        check = checks[key]
        draw.text((x + 24, 48), f'{label}：检测 '
                                f'{check["detected_openings"]} / 预期 {check["expected"]}'
                  + ('  通过' if check['passed'] else '  未通过'), font=small,
                  fill='#1d5c3a' if check['passed'] else '#8a2b20')
    draw.text((20, 715), '透明背景隔离渲染；'
                         '孔数由连通域检测。'
                         '紧定螺钉为径向孔，'
                         '不计入正视通孔数。',
              font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '03 轴承座 · 同光照材质'
                        '对照（原模型 / 新候选）',
              font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型'
                                        '金属表面（顶点色）'),
                                       ('new-material-close', '新候选 '
                                        'PBR 金属表面'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2, (840 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), '同一中性光与相机；'
                         '高光来自实时照明，'
                         '不烘焙在底色中。'
                         '原模型为单图重建的'
                         '像素/顶点色，不代表'
                         '旧模型整体通过。',
              font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '03 轴承座 · 变光检查'
                        '（同视角）', font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光位置 A'),
                                       ('new-light-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2, (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '灯光移动后高光和明暗'
                         '应随视角变化；本图'
                         '只检查反射响应，'
                         '不证明现场光照方向。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '03 轴承座 · 金属反射'
                        '变光对照', font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-material-close', '主光 A'),
                                       ('new-metal-alternate', '主光 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2, (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '金属高光随灯光位置'
                         '移动；不能据此推断'
                         '现场光源方向或强度。',
              font=small, fill='#52646b')
    board.save(out / '金属反射-变光对照.jpg', quality=96)


def make_turntable(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    for i in range(24):
        image = _tile_image(out / 'renders' / f'turn-{i:02}.png', (220, 215))
        tile = Image.new('RGBA', (230, 222), 'white')
        tile.alpha_composite(image, ((230 - image.width) // 2, (222 - image.height) // 2))
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
        mask = reference_mask_03(panel)
        points = cv2.findNonZero(mask)
        if points is None:
            out[name] = None
            continue
        x, y, ww, hh = cv2.boundingRect(points)
        out[name] = {'bbox_px': [int(x), int(y), int(ww), int(hh)],
                     'width_over_height': round(ww / max(1, hh), 4)}
    out['catalog_note'] = ('UCP210/P210 catalog L/H2 = 1.807 and L/A = 3.433. '
                          'The reference panels give smaller ratios; the source collage '
                          'appears horizontally compressed or vertically stretched and '
                          'is not calibrated. Catalog ratios are used only as a sanity check.')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=3)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--draft', action='store_true', help='quick render: six views and surfaces only')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 3:
        p.error('this entry supports item 03 only')
    a.reference = (a.reference or find_reference()).resolve()
    if a.out is None:
        a.out = ROOT / ('outputs/03-bearing-draft' if a.draft
                        else 'outputs/03轴承座-六视图重建-v3')
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
    config_path = HERE / 'bearing03_config.json'
    worker_path = HERE / 'mesh_rebuild_03_worker.py'
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
                required += [a.out / 'renders' / name for name in
                             ('detail-bearing.png', 'detail-set-screw.png', 'detail-zerk.png',
                              'detail-label-asahi.png', 'detail-label-p210.png',
                              'detail-slot.png', 'detail-bottom.png', 'detail-surface.png',
                              'bearing-only.png', 'base-only.png', 'housing-only.png',
                              'new-light-alternate.png', 'new-material-close.png',
                              'new-metal-alternate.png', 'old-material-close.png')]
                required += [a.out / 'renders' / f'turn-{i:02}.png' for i in range(24)]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('output already exists with a different signature or files')

    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_atlas_03(Path(row['rust']), a.out, config['material_profile'])
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
    views = make_report_03(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(3)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    if not a.draft:
        make_turntable(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('bearing', 'bearing-only.png', 1),
                ('base', 'base-only.png', 2),
                ('housing', 'housing-only.png', 1)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    material_checks = material_texture_checks(a.out / 'metallic_roughness.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'bearing_outer_ring': 'Bearing outer ring' in names,
        'bearing_inner_ring': 'Bearing inner ring' in names,
        'bearing_seals': sum(name.startswith('Bearing seal ') for name in names),
        'bearing_seals_expected': 2,
        'set_screws': sum(name.startswith('Set screw ') for name in names),
        'set_screws_expected': 2,
        'grease_nipple': all(('Zerk hex body' in names, 'Zerk domed tip' in names)),
        'cast_labels': sum(name.startswith('Cast label ') for name in names),
        'cast_labels_expected': 2,
        'foot_collars': sum(name.startswith('Foot collar ') for name in names),
        'bottom_pads': sum(name.startswith('Bottom machined pad ') for name in names),
        'isolated_openings': hole_checks,
        'note': 'Part presence proves only that a feature was modelled, not its size, fit or hidden faces.'
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
        'mesh_integrity': bool(mesh_audit['parts'] >= 20 and
                               not mesh_audit['nonfinite_parts'] and
                               not mesh_audit['open_parts'] and
                               not worker['self_intersecting_components']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': bool(structural['bearing_outer_ring'] and
                                  structural['bearing_inner_ring'] and
                                  structural['bearing_seals'] == 2 and
                                  structural['set_screws'] == 2 and
                                  structural['grease_nipple'] and
                                  structural['cast_labels'] == 2 and
                                  structural['foot_collars'] == 2 and
                                  structural['bottom_pads'] == 2),
        'material_tiles': bool(material_checks['passed']),
        'metal_reflection': bool(reflection.get('changed', False)),
        'reference_diagnostic_only': True,
    }
    report = {
        'version': config['version'], 'number': 3, 'asset_id': row['asset_id'],
        'name': row['file'], 'status': 'review', 'model_review': 'pending',
        'scene_validation': 'pending', 'manual_only': True,
        'shape_basis': 'user_six_view_reference',
        'method': 'parametric_part_rebuild_from_uncalibrated_six_view_reference',
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': (
            'The old mesh source-view IoU 0.9608 only measures one silhouette. It has no '
            'separate bearing inner/outer rings or set screws, the grease nipple is a lump, '
            'and the back is weakly observed, so this rebuild does not reuse it.'),
        'catalog_evidence': config['ratio_source'],
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'material_texture_checks': material_checks,
        'metal_reflection_check': reflection,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': (
            'The source photo is a single 3/4 rust image with a different scale and pose; '
            'the new-source framing distance was scaled by the new/old span. The score is '
            'diagnostic only and is not used as an acceptance gate.'),
        'reference_views': views,
        'reference_ratio_diagnostics': reference_ratio_diagnostics(a.reference),
        'ratio_note': ('The six-view collage is uncalibrated and appears anisotropic; '
                       'catalog ratios are a sanity check, not a measurement of the photographed object.'),
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            'The six-view reference is uncalibrated; absolute size, camera and proportions are estimates.',
            'Public UCP210/P210/UC210 catalog ratios are used for plausibility; this is the same model family but not a measurement of the photographed unit.',
            'The reference collage appears horizontally compressed/vertically stretched; it is not used as a calibrated silhouette target.',
            'The bearing insert is drawn as a symmetric UC210-type assembly; the real axial offset of the extended inner ring is simplified and inferred.',
            'Set-screw angular spacing (120 degrees) follows common UC practice; the reference perspective is too weak to measure it exactly.',
            'The back face, internal grease passage, seal construction, casting draft and lettering stroke details are inferred.',
            'Paint wear and rust distribution are synthesized from small patches of the supplied rust image; they are not photographs of hidden faces.',
            'Isolated opening counts prove only transparent openings in the render; they do not prove hole depth, threads, fits or assembly.',
            'Assemblies intentionally overlap; BVH checks only same-part self-intersections, not part-to-part collisions.',
            'Site scale, stability, contact and environment-light integration have not been checked.',
        ],
        'pending_review': [
            'Compare the six views and detail boards with the reference by eye, especially the base length/height ratio and the reference collage anisotropy.',
            'Confirm the bearing insert: asymmetric axial offset, set-screw count/angles, seal rings and bore diameter.',
            'Confirm the cast labels ASAHI/P210, the raised foot collars, the bottom machined pads and the underside arch.',
            'Check the grease-nipple shape and the top boss transition.',
            'Verify coordinate, scale, stability, contact and light integration in the manual placement page; do not feed this candidate to automatic placement or final compositing.',
            'Provide calibrated front/side views or measured dimensions if exact geometry is required.',
        ],
        'next_number': 4,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_03.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 3, 'name': row['file'], 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'comparison_board': str(a.out / '六视图-新旧对照.jpg'),
        'detail_board': str(a.out / '局部检查.jpg'),
        'isolated_board': str(a.out / '结构开口-隔离检查.jpg'),
        'checks': passed, 'material_checks': material_checks,
        'metal_reflection_check': reflection,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'pending_review': report['pending_review'],
        'next_number': 4,
        'notes': 'The old model, item 01 and item 02 were not modified; candidate is manual_only.'
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '03-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
