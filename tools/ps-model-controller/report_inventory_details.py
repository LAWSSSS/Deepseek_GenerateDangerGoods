"""Freeze current inventory inputs and verify editable candidates and full-view comparisons."""
import argparse
from html import escape
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash, load_catalog
from display_batch.inventory import STORAGE_TANK_IDS


PRIOR_CANDIDATES = {'1', '25', '33'}


def freeze(destination):
    rows = load_catalog()
    records = list(rows.values())
    atomic_json(destination, records)
    return records


def render_pair(folder, size):
    frames = []
    for filename in ('before.png', 'after.png'):
        with Image.open(folder/filename) as image:
            if image.size != (size, size) or image.mode != 'RGBA':
                raise ValueError('Unexpected full-view image format')
            frames.append(np.asarray(image).copy())
    a, b = (values[..., 3]>=128 for values in frames)
    union = (a|b).sum()
    if not union:
        raise ValueError('Candidate renders are empty')
    overlap = float((a&b).sum()/union)
    if overlap < .985:
        raise ValueError(f'Outline IoU {overlap:.4f} is below .985')
    common = a&b
    delta = np.abs(frames[0][common, :3].astype(int)-frames[1][common, :3].astype(int))
    return overlap, int((delta.max(axis=1)>3).sum())


def build_report(output, inventory, archive, size):
    rows = {row['id']:row for row in json.loads(inventory.read_text(encoding='utf8'))}
    current = load_catalog()
    hashes = {aid:r['mesh_hash'] for aid,r in current.items()}
    if hashes != {aid:r['mesh_hash'] for aid,r in rows.items()}:
        raise ValueError('Source inventory changed during refinement')
    selection = set(rows) - STORAGE_TANK_IDS - PRIOR_CANDIDATES
    record = json.loads((output/'asset-candidates.json').read_text(encoding='utf8'))
    if set(record['assets']) != selection:
        raise ValueError('Run does not cover the remaining active inventory')
    cards = []
    results = []
    files = {}
    for aid, item in record['assets'].items():
        label = escape(aid.zfill(2)+' '+rows[aid]['name'])
        if item['status'] == 'review':
            folder = output/aid
            if item['sha256'] != file_hash(folder/'candidate.blend') or not item['reopened']:
                raise ValueError('Candidate Blender readback hash mismatch')
            iou, changed = render_pair(folder, size)
            if changed == 0:
                raise ValueError('Bevel added topology with no measurable render change')
            name = aid.zfill(2)+'-'+rows[aid]['name']+'.png'
            files[name] = folder/'after.png'
            results.append({**item, 'outline_iou':iou,'visible_pixels_changed_3levels':changed,
                            'image_sha256':file_hash(files[name])})
            cards.append(f'<section><h2>{label}</h2><p>{escape(item["part_evidence"])}。'
                         f'面数 {item["faces_before"]:,} → {item["faces"]:,}，轮廓 IoU {iou:.4f}。</p>'
                         f'<a href="{aid}/candidate.blend">可编辑 Blender 候选</a>'
                         f'<div class="pair"><figure><figcaption>细化前</figcaption>'
                         f'<a href="{aid}/before.png"><img loading="lazy" src="{aid}/before.png"></a></figure>'
                         f'<figure><figcaption>细化后</figcaption><a href="{aid}/after.png">'
                         f'<img loading="lazy" src="{aid}/after.png"></a></figure></div></section>')
        else:
            results.append(item)
            cards.append(f'<section><h2>{label}</h2><p>待处理 · {escape(item["reason"])}</p></section>')
    archive_info = None
    if archive:
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as package:
            for name, path in files.items():
                package.write(path, arcname=name)
        import hashlib
        with zipfile.ZipFile(archive) as package:
            if package.testzip() or set(package.namelist()) != set(files) or any(
                    hashlib.sha256(package.read(name)).hexdigest()!=file_hash(path) for name,path in files.items()):
                raise ValueError('Flat PNG archive verification failed')
        archive_info = {'path':str(archive), 'pngs':len(files), 'sha256':file_hash(archive)}
    summary = {'candidate_count':len(files),'planned':len(selection),'source_models_verified':len(rows),
               'storage_tanks_removed':sorted(STORAGE_TANK_IDS,key=int),
               'model_api_calls':0,'api_cost':0,'blender_seconds':record['seconds'],
               'size_px':size,'archive':archive_info,'assets':results}
    atomic_json(output/'validation.json', summary)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport"'
            ' content="width=device-width"><title>物品部件细化</title>'
            '<style>body{max-width:1500px;margin:28px auto;padding:0 16px;font:16px system-ui;background:#eef1f3;color:#161d24}'
            'section{background:#fff;margin:18px 0;padding:16px;border-radius:7px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}'
            'figure{margin:8px 0}img{width:100%;display:block;background:#eff2f2}figcaption{padding:5px 0}</style>'
            f'<h1>其余物品的部件细化</h1><p>{len(files)}/{len(selection)} 件输出候选。完整画面采用同一机位和照明。'
            '储罐 05、06、07、38、39 已从可用清单移除。现场接触待复验。</p>'
            '<a href="validation.json">逐件核验</a>'+''.join(cards))
    (output/'index.html').write_text(page,encoding='utf8')
    return {k:v for k,v in summary.items() if k!='assets'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--resolution',type=int,default=768)
    args = parser.parse_args()
    if args.output is None:
        print(f'Frozen {len(freeze(args.inventory))} source models')
    else:
        print(json.dumps(build_report(args.output.resolve(),args.inventory,args.archive,args.resolution),
                         ensure_ascii=False))


if __name__ == '__main__':
    main()
