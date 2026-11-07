"""Build a single review-only model using the supplied six-view reference.

The first implementation intentionally supports item 01 only.  It never
replaces the original library model or claims a physical-size calibration.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = ('front', 'back', 'left', 'right', 'top', 'bottom')


def _tile_detail(image, rect, size=1024):
    """Use only a small observed material patch, removing slow light gradients."""
    patch = np.asarray(image.crop(rect).convert('RGBA'))
    if np.mean(patch[:, :, 3] > 240) < .95:
        raise ValueError('材质采样区域包含过多透明背景')
    gray = cv2.cvtColor(patch[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 12)
    # Remove the source's directional highlight before synthesizing a
    # non-repeating field with the observed patch's spatial frequencies.
    detail -= np.mean(detail, axis=0, keepdims=True)
    detail -= np.mean(detail, axis=1, keepdims=True)
    # Mirroring makes the texture continuous at repeat boundaries.
    mirrored = np.concatenate((detail, detail[:, ::-1]), axis=1)
    mirrored = np.concatenate((mirrored, mirrored[::-1]), axis=0)
    repeated = np.tile(mirrored, (size // mirrored.shape[0] + 1, size // mirrored.shape[1] + 1))[:size, :size]
    rng = np.random.default_rng(sum(rect))
    spectrum = np.abs(np.fft.rfft2(repeated))
    random_spectrum = np.fft.rfft2(rng.normal(size=(size, size)))
    phase = random_spectrum / np.maximum(np.abs(random_spectrum), 1e-12)
    synthesized = np.fft.irfft2(spectrum*phase, s=(size,size)).real.astype(np.float32)
    return np.clip(synthesized, -25, 25)


def make_atlas(source, out):
    """Create one base-color, metallic-roughness and normal atlas from source."""
    im = Image.open(source).convert('RGBA')
    if im.size != (1086, 1448):
        raise ValueError('01 做锈素材尺寸改变，采样坐标需要重新核对')
    patches = [_tile_detail(im, rect) for rect in
               ((305, 980, 370, 1190), (470, 545, 535, 700),
                (500, 225, 555, 345), (305, 980, 370, 1190))]
    rng = np.random.default_rng(10401)
    size = 1024
    base = np.empty((2048, 2048, 3), np.uint8)
    mr = np.empty_like(base)
    normal = np.empty_like(base)
    # Image coordinates are top-left; Blender UV mapping is bottom-left.
    specs = [((159, 158, 152), .75, .62, .045), ((164, 164, 158), .42, .78, .012),
             ((28, 73, 115), .49, .05, .025), ((132, 123, 109), .79, .38, .12)]
    for i, (color, roughness, metal, rust_amount) in enumerate(specs):
        detail = patches[i]
        coarse = cv2.resize(rng.random((46, 46), dtype=np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        specks = cv2.resize(rng.random((180, 180), dtype=np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        rust = np.clip((coarse - (.79 - rust_amount)) * 5, 0, 1) * rust_amount * 3
        rgb = np.asarray(color, np.float32)[None, None, :] + detail[:, :, None] * .42
        rgb = rgb * (1-rust[:, :, None]) + np.array((115, 69, 38), np.float32) * rust[:, :, None]
        rgb += ((specks-.5)*3)[:, :, None]
        yy, xx = divmod(i, 2); sl = np.s_[yy*size:(yy+1)*size, xx*size:(xx+1)*size]
        base[sl] = np.clip(rgb, 0, 255).astype(np.uint8)
        mr[sl] = np.stack((np.full_like(rust, 255),
                           np.clip((roughness + rust*.12 + detail/1000)*255, 0, 255),
                           np.clip((metal*(1-rust*.65))*255, 0, 255)), axis=2).astype(np.uint8)
        bump = cv2.GaussianBlur(detail, (0, 0), .6)/255
        dx = cv2.Sobel(bump, cv2.CV_32F, 1, 0, ksize=3)/8
        dy = cv2.Sobel(bump, cv2.CV_32F, 0, 1, ksize=3)/8
        nx = np.clip(-dx*.8, -.12, .12); ny = np.clip(dy*.8, -.12, .12)
        nz = np.sqrt(np.maximum(0, 1-nx*nx-ny*ny))
        normal[sl] = np.clip((np.stack((nx, ny, nz), axis=2)*.5+.5)*255, 0, 255).astype(np.uint8)
    paths = {}
    for name, arr in (('base_color', base), ('metallic_roughness', mr), ('normal', normal)):
        path = out / (name + '.png'); Image.fromarray(arr).save(path); paths[name] = str(path)
    return paths


def mask_fit(mask, size=400):
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    factor = (size-20)/max(w, h)
    resized = cv2.resize(mask[y:y+h, x:x+w].astype(np.uint8),
                         (max(1, round(w*factor)), max(1, round(h*factor))), interpolation=cv2.INTER_NEAREST)
    result = np.zeros((size, size), np.uint8)
    top, left = (size-resized.shape[0])//2, (size-resized.shape[1])//2
    result[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
    return result


def reference_mask(panel):
    rgb = np.asarray(panel.convert('RGB'))
    # Discard labels and shadows and retain foreground near the center only.
    mask = (np.min(rgb, axis=2) < 222).astype(np.uint8)
    mask[int(mask.shape[0]*.91):] = 0
    mask[:5] = 0; mask[-5:] = 0; mask[:, :5] = 0; mask[:, -5:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(18, stats[1:, cv2.CC_STAT_AREA].max()*.002))
    keep = keep[keep != 0]
    return np.isin(labels, keep).astype(np.uint8)


def make_report(out, reference):
    reference_image = Image.open(reference).convert('RGB')
    w, h = reference_image.size
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    board = Image.new('RGB', (1800, 1270), '#eef1f2'); draw = ImageDraw.Draw(board)
    draw.text((20, 12), '01 长阀杆工业阀门 · 六视图、新旧模型对照', font=font, fill='#18242b')
    rows = []
    for i, name in enumerate(VIEWS):
        x = (i % 3)*600; y = 50 + (i // 3)*600
        panel = reference_image.crop(((i%3)*w//3, (i//3)*h//2, (i%3+1)*w//3, (i//3+1)*h//2))
        old = Image.open(out/'renders'/f'old-{name}.png').convert('RGBA')
        new = Image.open(out/'renders'/f'new-{name}.png').convert('RGBA')
        expected = mask_fit(reference_mask(panel))
        actual = mask_fit(np.asarray(new.getchannel('A')) >= 128)
        iou = float(np.logical_and(expected, actual).sum()/max(1, np.logical_or(expected, actual).sum())) if expected is not None and actual is not None else None
        rows.append({'view': name, 'uncalibrated_reference_mask_iou': round(iou, 4) if iou is not None else None,
                     'note': '按最长边等比对齐，参考图未标定，仅供发现差异'})
        draw.text((x+12, y), f'{name.upper()}  参考图 / 原模型 / 新模型', font=small, fill='#27363c')
        ref_mask = reference_mask(panel)
        ref_bounds = cv2.boundingRect(ref_mask)
        rx, ry, rw, rh = ref_bounds
        panel = panel.crop((max(0,rx-5),max(0,ry-5),min(panel.width,rx+rw+5),min(panel.height,ry+rh+5)))
        for j, image in enumerate((panel.convert('RGBA'), old, new)):
            if j:
                rect=image.getchannel('A').getbbox()
                if rect: image=image.crop(rect)
            factor=min(182/image.width, 485/image.height)
            image=image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))),Image.Resampling.LANCZOS)
            tile = Image.new('RGBA', (190, 495), 'white')
            tile.alpha_composite(image, ((190-image.width)//2, (495-image.height)//2))
            board.paste(tile.convert('RGB'), (x+5+j*195, y+32))
        draw.text((x+12, y+548), f'参考粗略轮廓 IoU：{iou:.3f}' if iou is not None else '轮廓不可比较', font=small, fill='#52646b')
    board.save(out/'六视图-新旧对照.jpg', quality=95)
    return rows


def make_turntable(out):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    board=Image.new('RGB',(1440,1000),'#eef1f2');draw=ImageDraw.Draw(board)
    for i in range(24):
        image=Image.open(out/'renders'/f'turn-{i:02}.png').convert('RGBA')
        rect=image.getchannel('A').getbbox()
        if rect:image=image.crop(rect)
        image.thumbnail((220,215),Image.Resampling.LANCZOS)
        tile=Image.new('RGBA',(230,222),'white')
        tile.alpha_composite(image,((230-image.width)//2,(222-image.height)//2))
        x=(i%6)*240+5;y=(i//6)*250+4
        board.paste(tile.convert('RGB'),(x,y));draw.text((x+8,y+224),f'{i*15:03d}°',font=font,fill='#344044')
    board.save(out/'转台-24方向.jpg',quality=94)


def inspect_mesh(path):
    scene=trimesh.load(path,force='scene')
    open_parts=[];nonfinite=[];counts=[]
    for node in scene.graph.nodes_geometry:
        _,name=scene.graph[node]
        mesh=scene.geometry[name].copy()
        if not np.isfinite(mesh.vertices).all():nonfinite.append(node)
        # glTF duplicates vertices at UV and hard-normal seams. Merge only in
        # this audit copy; never alter the exported textured file.
        mesh.merge_vertices(merge_tex=True,merge_norm=True)
        if not mesh.is_watertight:open_parts.append(node)
        counts.append(len(mesh.faces))
    return {'parts':len(counts),'closed_after_welding_seams':len(counts)-len(open_parts),
            'open_parts':open_parts,'nonfinite_parts':nonfinite,
            'bounds':scene.bounds.tolist(),
            'note':'装配件故意相交；拓扑检查不能证明零件间无穿插或物理落稳'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--number', type=int, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--draft', action='store_true', help='快速检查，只渲染六方向')
    p.add_argument('--library', type=Path, default=ROOT/'outputs/三维模型库-圆润复验/results.jsonl')
    p.add_argument('--blender', type=Path, default=ROOT/'work/blender45/blender-4.5.3-windows-x64/blender.exe')
    a = p.parse_args()
    if a.number != 1:
        p.error('当前仅支持 01；后续物品需单独建立参考与建模方案')
    a.reference, a.out, a.library, a.blender = (x.resolve() for x in (a.reference, a.out, a.library, a.blender))
    row = next(r for r in (json.loads(line) for line in a.library.read_text(encoding='utf8').splitlines() if line.strip()) if r['number']==a.number)
    for name, expected in (('mesh', 'mesh_sha256'), ('rust', 'rust_sha256')):
        if digest(Path(row[name])) != row[expected]:
            raise ValueError(f'{name} 哈希与模型库不一致')
    alpha_path=Path(row['source_view']).parent/'source-alpha.png'
    if digest(alpha_path)!=row['source_alpha_sha256']:
        raise ValueError('源图透明度哈希与模型库不一致')
    config_path = HERE/'valve01_config.json'
    worker_path = HERE/'mesh_rebuild_item_worker.py'
    config = json.loads(config_path.read_text(encoding='utf8'))
    version = subprocess.run([str(a.blender), '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    signature_input = {'reference':digest(a.reference), 'mesh':row['mesh_sha256'], 'rust':row['rust_sha256'],
                       'source_alpha':digest(alpha_path), 'source_camera':row['reconstruction']['source_camera'],
                       'config':digest(config_path), 'builder':digest(Path(__file__)), 'worker':digest(worker_path),
                       'helpers':{name:digest(HERE/name) for name in ('blender_worker.py','mesh_checks.py','mesh_carve.py')},
                       'blender':version, 'python':sys.version.split()[0], 'numpy':np.__version__,
                       'opencv':cv2.__version__, 'pillow':PIL.__version__, 'trimesh':trimesh.__version__, 'draft':a.draft}
    signature = hashlib.sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    if a.out.exists() and any(a.out.iterdir()):
        record=a.out/'result.json';mesh=a.out/'mesh.glb'
        if record.is_file() and mesh.is_file():
            old=json.loads(record.read_text(encoding='utf8'))
            required=[a.out/'model.blend', a.out/'base_color.png', a.out/'metallic_roughness.png',
                      a.out/'normal.png', a.out/'六视图-新旧对照.jpg']
            required += [a.out/'renders'/f'{which}-{view}.png' for which in ('old','new') for view in VIEWS]
            if not a.draft:
                required += [a.out/'转台-24方向.jpg']+[a.out/'renders'/f'turn-{i:02}.png' for i in range(24)]
                required += [a.out/'renders'/name for name in ('detail-wheel.png','detail-stem.png','detail-side.png',
                            'detail-surface.png','new-light-alternate.png','wheel-only-top.png','flange-only-bottom.png')]
            if old.get('signature')==signature and old.get('mesh_sha256')==digest(mesh) and all(p.is_file() for p in required):
                print(json.dumps({'status':'review','cache_hit':True,'out':str(a.out)},ensure_ascii=False));return
        raise FileExistsError('输出已存在但签名或文件不匹配，请选择新的空目录')
    a.out.mkdir(parents=True, exist_ok=True)
    textures = make_atlas(Path(row['rust']), a.out)
    request = {'out':str(a.out), 'original_mesh':row['mesh'], 'textures':textures, 'config':config,
               'source_camera':row['reconstruction']['source_camera'], 'draft':a.draft}
    (a.out/'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
    with (a.out/'blender.log').open('w', encoding='utf8') as log:
        process = subprocess.run([str(a.blender), '--background', '--factory-startup', '--python-exit-code', '1',
                                  '--python', str(worker_path), '--', str(a.out/'request.json')],
                                 stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    if process.returncode:
        raise RuntimeError(f'Blender 失败，查看 {a.out / "blender.log"}')
    with Image.open(Path(row['source_view']).parent/'source-alpha.png') as alpha, Image.open(a.out/'renders'/'new-source.png') as rendered:
        fidelity = source_fidelity(np.asarray(alpha.convert('L')), np.asarray(rendered.getchannel('A')))
    views = make_report(a.out, a.reference)
    worker = json.loads((a.out/'worker-result.json').read_text(encoding='utf8'))
    mesh_audit=inspect_mesh(a.out/'mesh.glb')
    hole_checks={}
    if not a.draft:
        make_turntable(a.out)
        for name,filename,expected in (('wheel','wheel-only-top.png',4),('bottom_flange','flange-only-bottom.png',8)):
            with Image.open(a.out/'renders'/filename) as im:
                _,count=interior_holes(np.asarray(im.getchannel('A')))
            hole_checks[name]={'detected_openings':int(count),'expected':expected,'passed':int(count)==expected}
    # Approximate pixel landmarks from the front panel, manually recorded as
    # projections rather than real-world dimensions. Camera is uncalibrated.
    reference_ratios={'body_width':165/451, 'wheel_width':162/451, 'exposed_stem_length':80/451}
    candidate_ratios={'body_width':2*config['body_radius']/worker['span'],
                      'wheel_width':2*config['wheel_radius']/worker['span'],
                      'exposed_stem_length':(config['stem_top']-config['stem_bottom'])/worker['span']}
    ratios={name:{'reference_approx':round(value,4), 'model':round(candidate_ratios[name],4),
                  'relative_difference':round(abs(candidate_ratios[name]-value)/value,4)}
            for name,value in reference_ratios.items()}
    report = {'version':config['version'], 'number':1, 'asset_id':row['asset_id'],
              'status':'review', 'model_review':'pending', 'scene_validation':'pending', 'manual_only':True,
              'shape_basis':'user_six_view_reference', 'reference':str(a.reference), 'inputs':signature_input,
              'signature':signature, 'mesh':str(a.out/'mesh.glb'), 'mesh_sha256':digest(a.out/'mesh.glb'),
              'original_mesh_sha256':row['mesh_sha256'], 'source_fidelity_diagnostic':fidelity,
              'source_diagnostic_camera':'原重建相机的焦距与偏移，增加约 13° 俯角；近似对照，并非重新标定',
              'reference_views':views, 'front_ratio_diagnostics':ratios, 'mesh_audit':mesh_audit,
              'isolated_opening_checks':hole_checks,
              'ratio_note':'参考面板人工约取像素，投影与三维参数直接对比，仅作粗略诊断', 'worker':worker,
              'limitations':['六视图未标定，尺寸和相机仅为相对估算', '侧背面材质和不可见结构为推断',
                             '参考右视图的侧盖可见性与其余视角不完全一致，保留单侧侧盖',
                             'BVH 面交叠只检查同部件；装配件之间有预期交叠，未判定碰撞',
                             '现场尺度、物理姿态和环境融合尚未检查', '源图轮廓只作为诊断，未用于否决六视图形体'],
              'draft':a.draft}
    (a.out/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status':'review', 'out':str(a.out), 'source_fidelity':fidelity, 'views':views}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
