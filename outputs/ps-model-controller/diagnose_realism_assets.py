"""Offline, fixed-pose clay/source/physical-layer diagnostics for five assets.

No scene photograph is composited and no model API is called. Freeze first, then
render 35 small object plates using the same worker as production.
"""
import argparse
import asyncio
import copy
import html
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image

from display_batch.common import ROOT, atomic_json, file_hash, resolve, signature
from display_batch.generated_material import load_pack
from display_batch.physical_material import prepare_finish
from display_batch.rendering import RenderCache
from display_batch.worker import Blender


VERSION = 'realism-asset-diagnostics-v1'
ASSETS = ('3', '13', '22', '31', '33')
SOURCE = 'outputs/全部启用物品-三场景-20261116'
PACK = 'work/render-materials/nearby-worn-steel-v15/pack.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def freeze(source, cache, pack):
    records = sorted((json.loads(line) for line in (source/'results.jsonl').read_text(encoding='utf8').splitlines()),
                     key=lambda row: row['job_id'])
    cases = []
    for asset_id in ASSETS:
        rows = [r for r in records if r['asset_id'] == asset_id and r.get('image')]
        if not rows:
            raise ValueError(f'no previously deliverable frozen pose for asset {asset_id}')
        row = rows[0]  # Stable job order; never rank by aesthetic score.
        entry = next(e for e in row['iterations'] if e['iteration'] == row['best_iteration'])
        candidates = read(source/'jobs'/row['job_id']/'candidates.json')['candidates']
        candidate = next(c for c in candidates if c['id'] == entry['params']['candidate'])
        asset_file = cache/'assets'/row['asset_signature']/'result-strict.json'
        if not asset_file.is_file():
            asset_file = asset_file.with_name('result.json')
        asset = read(asset_file)
        if file_hash(asset['mesh']) != asset['mesh_hash']:
            raise ValueError('source model changed since frozen batch')
        physical_size = candidate.get('scale',{}).get('physical_size',{})
        size = candidate.get('base_pose',{}).get('length_m')
        if size is None and physical_size.get('nominal_mm'):
            size = physical_size['nominal_mm']/1000.
        finish = prepare_finish(asset, cache, pack, size)
        cases.append({'asset_id':asset_id, 'name':asset.get('name',asset_id), 'source_job':row['job_id'],
                      'asset_record':str(asset_file), 'asset_record_hash':file_hash(asset_file),
                      'mesh_hash':asset['mesh_hash'], 'candidate':candidate,
                      'finish':finish, 'physical_size':physical_size,
                      'scene_contact_status':'inherited_unknown_not_asset_acceptance',
                      'selection':'first_deliverable_job_by_id_no_visual_ranking'})
    return {'version':VERSION, 'source':str(source), 'pack':pack['signature'], 'cases':cases,
            'planned_object_renders':len(cases)*7, 'api_calls':0,
            'scope':'material_diagnostic_not_scene_acceptance'}


def neutral_light(right=False):
    return {'world_color':[.36,.36,.36], 'light_color':[1.,1.,1.], 'dust_color':[.20,.185,.165],
            'light_level':1., 'light_spread':.6, 'integration_mode':'coherent',
            'local_irradiance':{'direction':{'direction':[.8 if right else -.8,-.25], 'confidence':.8},
                                'level':1., 'fill_ratio':.28},
            'basis':'neutral_controlled_broad_light_no_background_fit'}


def exr(path):
    image = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or not np.isfinite(image).all():
        raise ValueError(f'invalid diagnostic EXR: {path}')
    return image[...,:3][...,::-1]


def coverage_stats(folder, alpha):
    path=folder/'physical-regions0001.exr'
    if not path.is_file():
        return {'status':'not_applicable'}
    values=exr(path);core=alpha>.98
    if not core.any():
        return {'status':'unknown', 'reason':'no_opaque_pixels'}
    return {'status':'measured','core_pixels':int(core.sum()),
            'metal':float((values[...,0][core]>.95).mean()),
            'paint':float((values[...,1][core]>.95).mean()),
            'protected_or_unknown':float((values[...,:2].max(axis=2)[core]<.05).mean()),
            'basis':'antialiased_source_region_AOV_not_final_metalness'}


def write_report(out, rows, summary):
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>资产写实诊断</title>',
          '<style>body{font:16px sans-serif;max-width:1600px;margin:auto;padding:24px;background:#ddd;color:#222}.row{display:flex;flex-wrap:wrap}figure{margin:8px;width:30%;min-width:260px}img{width:100%;height:auto;background:#777}a{color:#075d66}pre{white-space:pre-wrap}</style>',
          '<h1>灰模 → 原 PBR → 三层材质：受控诊断</h1><p>不叠加薄雾、调色、噪声或相机模糊。仅为资产诊断，不代表现场接触、尺度或最终融合通过。</p>',
          '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>']
    for case in rows:
        body.append('<h2>'+html.escape(case['asset_id']+' '+case['name'])+'</h2><div class="row">')
        for item in case['renders']:
            uri=html.escape(item['image'],quote=True)
            body.append('<figure><figcaption>'+html.escape(item['label'])+'</figcaption><a href="'+uri+'"><img src="'+uri+'"></a></figure>')
        body.append('</div><pre>'+html.escape(json.dumps(case.get('checks',{}),ensure_ascii=False,indent=2))+'</pre>')
    body.append('</html>')
    (out/'index.html').write_text(''.join(body),encoding='utf8')


async def run(args):
    source,cache,out=resolve(args.source),resolve(args.cache),resolve(args.output)
    out.mkdir(parents=True,exist_ok=True)
    pack=load_pack(resolve(args.pack))
    frozen=freeze(source,cache,pack)
    path=out/'frozen.json'
    if path.is_file() and signature(read(path))!=signature(frozen):
        raise ValueError('diagnostic inputs changed; use a fresh output directory')
    atomic_json(path,frozen)
    if args.freeze_only:
        print(json.dumps({'frozen':str(path),'cases':len(frozen['cases']),'renders':35},ensure_ascii=False))
        return
    worker=Blender(out/'worker',args.device)
    renderer=RenderCache(cache,worker)
    started=time.monotonic();results=[]
    try:
        for case in frozen['cases']:
            asset=read(case['asset_record']);candidate=case['candidate'];renders=[];alpha={};paths={}
            variants=[(f'{mode}-{light}',mode,light,0) for mode in ('clay','source','physical') for light in ('left','right')]
            variants.append(('physical-turned','physical','left',55))
            for tag,mode,light,turn in variants:
                params={'material':'original','light':'scene','dust':'auto','surface_dust':.5,
                        'integration_mode':'coherent','render_detail':'standard','asset_diagnostic':mode}
                posed=copy.deepcopy(candidate);posed['yaw']=float(posed['yaw'])+turn
                finish=case['finish'] if mode=='physical' else None
                folder,meta,hit=await renderer.get(asset,posed,params,neutral_light(light=='right'),(3840,2160),512,
                            appearance='local',full_frame=True,object_only=True,generated_finish=finish)
                image=out/case['asset_id']/(tag+'.png');image.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(folder/'object.png',image)
                with Image.open(image) as im:
                    if im.size!=(512,512):raise ValueError('unexpected diagnostic plate dimensions')
                    alpha[tag]=np.asarray(im.getchannel('A'),np.float32)/255.
                paths[tag]=folder
                if not np.any(alpha[tag]>.5):raise ValueError('empty diagnostic object')
                renders.append({'tag':tag,'label':{'clay':'灰模','source':'原 PBR','physical':'三层材质'}[mode]+' · '+
                                ('转向 55°' if turn else ('左侧柔光' if light=='left' else '右侧柔光')),
                                'image':image.relative_to(out).as_posix(),'sha256':file_hash(image),
                                'cache_hit':hit,'render_cache':meta['render_cache'],
                                'source_regions':coverage_stats(folder,alpha[tag]),
                                'material_metadata':meta.get('generated_finish')})
                print(json.dumps({'asset':case['asset_id'],'diagnostic':tag,'cache_hit':hit},ensure_ascii=False),flush=True)
            a,b=alpha['source-left']>.5,alpha['physical-left']>.5
            iou=float((a&b).sum()/max(1,(a|b).sum()))
            core=(alpha['physical-left']>.98)&(alpha['physical-right']>.98)
            finish_maps=[exr(paths[tag]/'generated-finish0001.exr') for tag in ('physical-left','physical-right')]
            map_mae=float(np.abs(finish_maps[0][core]-finish_maps[1][core]).mean()) if core.any() else None
            specs=[exr(paths[tag]/'specular0001.exr') for tag in ('physical-left','physical-right')]
            reflection_change=float(np.abs(specs[0][core]-specs[1][core]).mean()) if core.any() else None
            checks={'source_mesh_unchanged':file_hash(asset['mesh'])==case['mesh_hash'],
                    'source_to_physical_alpha_iou':iou,'material_map_mae_changed_light':map_mae,
                    'specular_mae_changed_light':reflection_change,
                    'pixel_geometry_pass':iou>=.995,'material_map_light_invariance_pass':map_mae is not None and map_mae<.01,
                    'reflection_responds':reflection_change is not None and reflection_change>1e-5,
                    'visual_realism':'unreviewed','geometry_against_source_photos':'requires_visual_comparison_not_proven_by_alpha'}
            row={'asset_id':case['asset_id'],'name':case['name'],'renders':renders,'checks':checks,
                 'material_audit':case['finish']['material_audit'],'physical_size':case['physical_size']}
            results.append(row);atomic_json(out/'results.json',results)
            write_report(out,results,{'state':'running','completed_assets':len(results),'api_calls':0})
    finally:
        await worker.close()
    summary={'version':VERSION,'state':'complete','assets':len(results),'plates':sum(len(r['renders']) for r in results),
             'wall_seconds':round(time.monotonic()-started,3),'worker_seconds':round(worker.seconds,3),
             'render_counts':dict(renderer.totals),'api_calls':0,'tokens':0,'api_cost':0,
             'local_compute_cost':'unknown','visual_status':'unreviewed'}
    atomic_json(out/'summary.json',summary);write_report(out,results,summary)
    print(json.dumps(summary,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=SOURCE)
    parser.add_argument('--cache',default='work/display-batch-cache')
    parser.add_argument('--pack',default=PACK)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('OPTIX','CPU'),default='OPTIX')
    parser.add_argument('--freeze-only',action='store_true')
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__':main()
