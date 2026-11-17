"""Build the review-only DN200 ball-valve candidate (item 02).

The candidate is generated from a user-supplied, uncalibrated six-view image.
It never replaces the library mesh and is never used by the automatic
placement pipeline.  Existing files are not overwritten: the output directory
must be new, or contain a complete result with the same signature.
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
from mesh_rebuild_item import _tile_detail, mask_fit, reference_mask, inspect_mesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')
AUDIT = ROOT / 'outputs/三维模型-39件六视角审计/audit.jsonl'


def make_atlas_02(source, out, profile):
    """Create one 2048 px atlas with cast/machined/paint/rust tiles."""
    im = Image.open(source).convert('RGBA')
    if im.size != (1254, 1254):
        raise ValueError('02 做锈素材尺寸改变，采样坐标需要重新核对')
    patches = [_tile_detail(im, rect) for rect in
               ((740, 690, 790, 740), (840, 860, 880, 900),
                (500, 240, 560, 290), (450, 900, 480, 940))]
    rng = np.random.default_rng(20402)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    specs = [((152, 150, 144), profile['cast']['roughness'], profile['cast']['metallic'],
              profile['cast']['rust_amount']),
             ((168, 168, 164), profile['machined']['roughness'], profile['machined']['metallic'],
              profile['machined']['rust_amount']),
             ((30, 66, 108), profile['blue_paint']['roughness'], profile['blue_paint']['metallic'],
              profile['blue_paint']['rust_amount']),
             ((122, 98, 74), profile['rust']['roughness'], profile['rust']['metallic'],
              profile['rust']['rust_amount'])]
    for i, (color, roughness, metal, rust_amount) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((120, 120), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_CUBIC)
        specks = cv2.resize(rng.random((400, 400), dtype=np.float32), (size, size),
                            interpolation=cv2.INTER_LINEAR)
        rust = np.clip((coarse - (.79 - rust_amount)) * 5, 0, 1) * rust_amount * 3
        rgb = np.asarray(color, np.float32)[None, None, :] + detail[:, :, None] * .42
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
    tiles = {'cast': np.s_[0:1024, 0:1024], 'machined': np.s_[0:1024, 1024:2048],
             'blue_paint': np.s_[1024:2048, 0:1024], 'rust': np.s_[1024:2048, 1024:2048]}
    values = {name: {'roughness_mean': round(float(mr[sl][..., 1].mean()), 4),
                     'metallic_mean': round(float(mr[sl][..., 2].mean()), 4)}
              for name, sl in tiles.items()}
    values['passed'] = (values['machined']['roughness_mean'] < values['cast']['roughness_mean'] and
                        values['blue_paint']['metallic_mean'] < .10 and
                        values['cast']['metallic_mean'] > .80 and
                        values['rust']['roughness_mean'] > .70)
    values['note'] = '纹理中的粗糙度/金属度分区均值；反光方向未烘焙，随灯光变化'
    return values


def make_report_02(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    w, h = reference_image.size
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1800, 1300), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '02 DN200 球阀 · 六视图、原模型与新候选对照', font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3) * 600
        y = 50 + (i // 3) * 600
        panel = reference_image.crop(((i % 3) * w // 3, (i // 3) * h // 2,
                                      (i % 3 + 1) * w // 3, (i // 3 + 1) * h // 2))
        old = Image.open(out / 'renders' / f'old-{name}.png').convert('RGBA')
        new = Image.open(out / 'renders' / f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask(panel))
        actual = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        old_fit = mask_fit(np.asarray(old.getchannel('A')) >= 128)
        iou = (float(np.logical_and(expected, actual).sum() /
                     max(1, np.logical_or(expected, actual).sum()))
               if expected is not None and actual is not None else None)
        old_new = (float(np.logical_and(old_fit, actual).sum() /
                         max(1, np.logical_or(old_fit, actual).sum()))
                   if old_fit is not None and actual is not None else None)
        note = '按最长边等比对齐；参考图未标定，仅供发现差异'
        if name in ('top', 'bottom'):
            note += '；该参考面板是斜视展示，不能与正交渲染精确匹配'
        rows.append({'view': name,
                     'uncalibrated_reference_mask_iou': round(iou, 4) if iou is not None else None,
                     'old_new_silhouette_iou': round(old_new, 4) if old_new is not None else None,
                     'note': note})
        draw.text((x + 12, y), f'{name.upper()}  参考图 / 原模型 / 新候选', font=small, fill='#27363c')
        ref_mask = reference_mask(panel)
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
                  (f'参考粗略轮廓 IoU：{iou:.3f} · 原/新 IoU：{old_new:.3f}'
                   if iou is not None and old_new is not None else '轮廓不可比较'),
                  font=small, fill='#52646b')
    draw.text((20, 1255),
              '参考图未标定。顶/底参考面板为斜视展示；图中分数不是实测精度，不能据此判定结构合格。',
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
    board = Image.new('RGB', (1800, 1250), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '02 球阀 · 关键局部检查（新候选）', font=font, fill='#18242b')
    items = (('detail-wheel', '手轮三辐与三个开口'), ('detail-flange', '端法兰八孔、密封面与球芯'),
             ('detail-body', 'DN200/25/WCB 铸字'), ('detail-foot', '支脚、底板与安装孔'),
             ('detail-bottom', '底部圆凸台与六角盲孔'), ('detail-surface', '铸钢/漆面/锈蚀材质'))
    for i, (name, label) in enumerate(items):
        x = (i % 3) * 600
        y = 55 + (i // 3) * 590
        image = _tile_image(out / 'renders' / f'{name}.png', (560, 520))
        tile = Image.new('RGBA', (570, 530), 'white')
        tile.alpha_composite(image, ((570 - image.width) // 2, (530 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 10, y + 30))
        draw.text((x + 14, y), label, font=small, fill='#27363c')
    draw.text((20, 1210), '这些局部图用于人工检查结构；不能证明未观察面、尺寸或装配关系。',
              font=small, fill='#52646b')
    board.save(out / '局部检查.jpg', quality=95)


def make_isolated_board(out, checks):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    board = Image.new('RGB', (1800, 760), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '02 球阀 · 隔离开口检查', font=font, fill='#18242b')
    items = (('wheel-only', '手轮', 'wheel'),
             ('flange-only', '端法兰（含中心孔）', 'end_flange'),
             ('foot-plate-only', '支脚底板', 'foot_plate'))
    for i, (name, label, key) in enumerate(items):
        x = i * 600
        image = _tile_image(out / 'renders' / f'{name}.png', (500, 560))
        tile = Image.new('RGBA', (530, 570), 'white')
        tile.alpha_composite(image, ((530 - image.width) // 2, (570 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        check = checks[key]
        draw.text((x + 24, 26), f'{label}：检测 {check["detected_openings"]} / 预期 {check["expected"]}'
                  + (' ✓' if check['passed'] else ' ✗'), font=small,
                  fill='#1d5c3a' if check['passed'] else '#8a2b20')
    draw.text((20, 700), '透明背景隔离渲染。孔数由连通域检测；中心孔在球阀关闭时由球芯遮挡，'
                         '本检查移除球芯以核对通孔。', font=small, fill='#52646b')
    board.save(out / '结构开口-隔离检查.jpg', quality=95)


def make_material_boards(out):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1440, 1000), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '02 球阀 · 同光照材质对照（原模型 / 新候选）', font=font, fill='#18242b')
    for i, (name, label) in enumerate((('old-material-close', '原模型金属表面（顶点色）'),
                                       ('new-material-close', '新候选 PBR 金属表面'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 820))
        tile = Image.new('RGBA', (680, 840), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2, (840 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 70))
        draw.text((x + 24, 38), label, font=small, fill='#27363c')
    draw.text((20, 950), '同一中性光与相机；高光来自实时照明，不烘焙在底色中。'
                         '原模型为单图重建的顶点色，不代表旧模型整体通过。', font=small, fill='#52646b')
    board.save(out / '材质-同光照对照.jpg', quality=96)

    board = Image.new('RGB', (1440, 860), '#eef1f2')
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), '02 球阀 · 变光检查（同视角）', font=font, fill='#18242b')
    for i, (name, label) in enumerate((('new-front', '主光位置 A'), ('new-light-alternate', '主光位置 B'))):
        x = i * 720
        image = _tile_image(out / 'renders' / f'{name}.png', (660, 700))
        tile = Image.new('RGBA', (680, 720), 'white')
        tile.alpha_composite(image, ((680 - image.width) // 2, (720 - image.height) // 2))
        board.paste(tile.convert('RGB'), (x + 20, 65))
        draw.text((x + 24, 36), label, font=small, fill='#27363c')
    draw.text((20, 810), '灯光移动后高光和明暗应随视角变化；本图只检查反射响应，不证明现场光照方向。',
              font=small, fill='#52646b')
    board.save(out / '材质-变光检查.jpg', quality=96)


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


def ratio_diagnostics(config):
    total = 2 * config['flange_outer_y']
    reference = {'wheel_diameter': 210 / 315, 'body_diameter': 185 / 315,
                 'flange_diameter': 175 / 315,
                 'wheel_center_height': 365 / 315, 'body_center_height': 120 / 315}
    candidate = {'wheel_diameter': 2 * (config['wheel_radius'] + config['wheel_tube']) / total,
                 'body_diameter': 2 * config['body_radius'] / total,
                 'flange_diameter': 2 * config['flange_radius'] / total,
                 'wheel_center_height': config['gear_center_z'] / total,
                 'body_center_height': config['body_center_z'] / total}
    return {name: {'reference_approx': round(value, 4), 'model': round(candidate[name], 4),
                   'relative_difference': round(abs(candidate[name] - value) / value, 4)}
            for name, value in reference.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, default=2)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--draft', action='store_true', help='快速检查，只渲染六方向与表面')
    p.add_argument('--library', type=Path,
                   default=ROOT / 'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path,
                   default=ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 2:
        p.error('该入口仅支持 02 球阀；其他物品需单独建立参考与建模方案')
    a.reference, a.out, a.library, a.blender = (x.resolve() for x in
                                                (a.reference, a.out, a.library, a.blender))
    row = next(r for r in (json.loads(line) for line in
                           a.library.read_text(encoding='utf8').splitlines() if line.strip())
               if r['number'] == a.number)
    for name, expected in (('mesh', 'mesh_sha256'), ('rust', 'rust_sha256')):
        if digest(Path(row[name])) != row[expected]:
            raise ValueError(f'{name} 哈希与模型库不一致')
    alpha_path = Path(row['source_view']).parent / 'source-alpha.png'
    if digest(alpha_path) != row['source_alpha_sha256']:
        raise ValueError('源图透明度哈希与模型库不一致')
    if digest(a.reference) != 'fafbf84cf06b0e862af2900ff146f4c9d6c8c1c1991f85e287484b8b9dd98d1a':
        raise ValueError('六视图参考哈希与登记值不一致')
    config_path = HERE / 'valve02_config.json'
    worker_path = HERE / 'mesh_rebuild_02_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True, text=True,
                             check=True).stdout.splitlines()[0]
    signature_input = {'reference': digest(a.reference), 'mesh': row['mesh_sha256'],
                       'rust': row['rust_sha256'],
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
                        a.out / '六视图-新旧对照.jpg', a.out / '局部检查.jpg',
                        a.out / '结构开口-隔离检查.jpg', a.out / '材质-同光照对照.jpg',
                        a.out / '材质-变光检查.jpg', a.out / 'hashes.json']
            required += [a.out / 'renders' / f'{which}-{view}.png'
                         for which in ('old', 'new') for view in VIEWS]
            if not a.draft:
                required += [a.out / '转台-24方向.jpg'] + [
                    a.out / 'renders' / f'turn-{i:02}.png' for i in range(24)]
                required += [a.out / 'renders' / name for name in
                             ('detail-wheel.png', 'detail-flange.png', 'detail-body.png',
                              'detail-foot.png', 'detail-bottom.png', 'detail-surface.png',
                              'new-light-alternate.png', 'new-material-close.png',
                              'old-material-close.png', 'wheel-only.png', 'flange-only.png',
                              'foot-plate-only.png')]
            if old.get('signature') == signature and old.get('mesh_sha256') == digest(mesh) \
                    and all(path.is_file() for path in required):
                print(json.dumps({'status': 'review', 'cache_hit': True, 'out': str(a.out)},
                                 ensure_ascii=False))
                return
        raise FileExistsError('输出已存在但签名或文件不匹配，请选择新的空目录')
    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_atlas_02(Path(row['rust']), a.out, config['material_profile'])
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
                                 stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    if process.returncode:
        raise RuntimeError(f'Blender 失败，查看 {a.out / "blender.log"}')
    with Image.open(alpha_path) as alpha, \
            Image.open(a.out / 'renders' / 'new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')),
                                   np.asarray(rendered.getchannel('A')))
    views = make_report_02(a.out, a.reference)
    worker = json.loads((a.out / 'worker-result.json').read_text(encoding='utf8'))
    audit = audit_row(2)
    mesh_audit = inspect_mesh(a.out / 'mesh.glb')
    hole_checks = {}
    if not a.draft:
        make_turntable(a.out)
        make_detail_board(a.out)
        make_material_boards(a.out)
        for key, filename, expected in (
                ('wheel', 'wheel-only.png', 3),
                ('end_flange', 'flange-only.png', 9),
                ('foot_plate', 'foot-plate-only.png', 2)):
            with Image.open(a.out / 'renders' / filename) as im:
                _, count = interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[key] = {'detected_openings': int(count), 'expected': expected,
                                'passed': int(count) == expected,
                                'file': str(a.out / 'renders' / filename)}
        make_isolated_board(a.out, hole_checks)
    material_checks = material_texture_checks(a.out / 'metallic_roughness.png')
    names = [part['name'] for part in worker['parts']]
    structural = {
        'handwheel_spokes': sum(name.startswith('Handwheel spoke ') for name in names),
        'handwheel_spokes_expected': 3,
        'split_flange_bolts': sum(name.startswith('Split flange bolt head ') for name in names),
        'split_flange_bolts_expected': 12,
        'feet': sum(name.startswith('Foot base ') for name in names),
        'feet_expected': 2,
        'foot_plate_holes_expected_per_plate': 2,
        'ball_present': 'Ball' in names,
        'cast_text_present': all(('Cast label DN200' in names, 'Cast label 25' in names,
                                  'Cast label WCB' in names)),
        'isolated_openings': hole_checks,
        'note': '部件数量与隔离开口只证明已建模特征，不证明尺寸、装配或隐藏面。'
    }
    ratios = ratio_diagnostics(config)
    passed = {
        'mesh_integrity': bool(mesh_audit['parts'] > 40 and not mesh_audit['nonfinite_parts']),
        'isolated_openings': bool(hole_checks and all(v['passed'] for v in hole_checks.values())),
        'structural_counts': bool(structural['handwheel_spokes'] == 3 and
                                  structural['split_flange_bolts'] == 12 and
                                  structural['feet'] == 2 and structural['ball_present'] and
                                  structural['cast_text_present']),
        'material_tiles': bool(material_checks['passed']),
        'reference_diagnostic_only': True,
    }
    report = {
        'version': config['version'], 'number': 2, 'asset_id': row['asset_id'],
        'status': 'review', 'model_review': 'pending', 'scene_validation': 'pending',
        'manual_only': True, 'shape_basis': 'user_six_view_reference',
        'method': 'parametric_part_rebuild_from_uncalibrated_six_view_reference',
        'reference': str(a.reference), 'reference_sha256': digest(a.reference),
        'original_mesh': row['mesh'], 'original_mesh_sha256': row['mesh_sha256'],
        'original_source_fidelity_record': (audit or {}).get('source_fidelity'),
        'original_structural_warning': '原模型源视图 IoU 0.9287，但有 2 个孔被填；'
                                       '不能当作结构通过，本次没有沿用该网格',
        'inputs': signature_input, 'signature': signature,
        'mesh': str(a.out / 'mesh.glb'), 'mesh_sha256': digest(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'), 'blend_sha256': digest(a.out / 'model.blend'),
        'textures': textures,
        'material_profile': config['material_profile'],
        'material_texture_checks': material_checks,
        'source_fidelity_diagnostic': fidelity,
        'source_diagnostic_note': '源图是单张做锈图，姿态/透视与六视图候选不完全一致；'
                                  '该分数只作诊断，不作为本次结构验收门槛。',
        'reference_views': views, 'reference_ratio_diagnostics': ratios,
        'ratio_note': '前视面板人工约取像素，仅用于相对比例检查；参考图未标定，不是实测尺寸。',
        'mesh_audit': mesh_audit, 'isolated_opening_checks': hole_checks,
        'structural_checks': structural, 'checks': passed,
        'worker': worker, 'draft': a.draft,
        'limitations': [
            '六视图未标定，模型尺寸、相机和比例仅为相对估算',
            '参考图前视与背视的剖分法兰左右位置不一致，候选按前视物理一致性保留在 -Y 侧，背面会呈镜像差异',
            '参考顶视/底视是斜视展示图，不能与正交顶/底渲染精确对照',
            '支脚在侧视、底视中的连接方式不一致；候选采用双立板鞍座，属于推断',
            '手轮上的 OPEN/SHUT 凸字、铸件表面局部缺陷和不可见内部阀球通道未重建',
            '隔离开口检查只证明透明渲染中的孔数；不证明孔深、螺纹、装配过盈或真实密封',
            '部件之间有预期装配交叠，BVH 仅检查同部件自交，未判定零件间碰撞',
            '现场尺度、落稳、支撑接触和环境光融合尚未检查',
            '参考图与候选均为外观/结构参考，不是摄影测量或精确实测',
        ],
        'pending_review': [
            '人工对照六视图与局部图，确认手轮、剖分法兰、端法兰、支脚和底部凸台',
            '确认背面剖分法兰的左右方向：参考图正/背视互相矛盾，当前按前视推断',
            '确认支脚形式：侧视/底视显示双立板鞍座，正/背视更接近单立板；当前采用鞍座',
            '确认参考顶/底斜视面板对应关系，必要时补充真正的正交参考',
            '补充手轮 OPEN/SHUT 凸字与不可见内部通道的独立证据',
            '在人工放置页面检查坐标、比例、落稳、接触和环境融合，不能直接进入正式合成',
        ],
        'next_number': 3,
    }
    (a.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    files = [p for p in sorted(a.out.rglob('*')) if p.is_file()]
    hashes = {'algorithm': 'sha256', 'generated_by': 'mesh_rebuild_item_02.py',
              'files': {str(p.relative_to(a.out)).replace('\\', '/'): digest(p) for p in files}}
    hashes['files']['hashes.json'] = 'self'
    (a.out / 'hashes.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                       encoding='utf8')
    progress = {
        'number': 2, 'name': 'aDN350 球阀1', 'asset_id': row['asset_id'],
        'status': 'candidate_generated_pending_review',
        'version': config['version'], 'out': str(a.out),
        'result': str(a.out / 'result.json'), 'mesh': str(a.out / 'mesh.glb'),
        'blend': str(a.out / 'model.blend'),
        'checks': passed, 'material_checks': material_checks,
        'isolated_openings': hole_checks, 'reference_views': views,
        'original_source_fidelity': (audit or {}).get('source_fidelity'),
        'pending_review': report['pending_review'],
        'next_number': 3,
        'notes': '旧模型和 01 未修改；候选为 manual_only，未接入正式合成。'
    }
    (a.out / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                         encoding='utf8')
    progress_dir = ROOT / 'work/direct-deepseek'
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / '02-progress.json').write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': 'review', 'out': str(a.out), 'checks': passed,
                      'source_fidelity': fidelity, 'views': views}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
