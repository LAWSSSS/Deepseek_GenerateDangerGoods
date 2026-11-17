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
import math
import subprocess

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image, ImageDraw

from display_batch.common import ROOT, HERE, BLENDER, atomic_json, file_hash, resolve, signature
from display_batch.generated_material import load_pack
from display_batch.physical_material import prepare_finish, refine_finish, CONTROL_LIMITS
from display_batch.rendering import RenderCache
from display_batch.worker import Blender
from display_batch.asset_profiles import load_profile, descriptor
from display_batch.geometry import pose_points, camera_for
from display_batch.rendering import render_request
from display_batch.framing import render_windows


VERSION = 'realism-asset-diagnostics-v1'
ASSETS = ('3', '13', '22', '31', '33')
SOURCE = 'outputs/全部启用物品-三场景-20261116'
PACK = 'work/render-materials/nearby-worn-steel-v15/pack.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def freeze(source, cache, pack, refinement=False, resolution=512):
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
        profile=load_profile(asset['mesh_hash']) if refinement else None
        finish = prepare_finish(asset, cache, pack, size, profile=profile)
        cases.append({'asset_id':asset_id, 'name':asset.get('name',asset_id), 'source_job':row['job_id'],
                      'asset_record':str(asset_file), 'asset_record_hash':file_hash(asset_file),
                      'mesh_hash':asset['mesh_hash'], 'candidate':candidate,
                      'finish':finish, 'physical_size':physical_size,
                      'scene_contact_status':'inherited_unknown_not_asset_acceptance',
                      'selection':'first_deliverable_job_by_id_no_visual_ranking'})
    return {'version':VERSION+('-evidence-v2' if refinement else ''), 'source':str(source), 'pack':pack['signature'], 'cases':cases,
            'asset_refinement':refinement,'resolution':resolution,
            'planned_object_renders':len(cases)*7, 'api_calls':0,
            'scope':'material_diagnostic_not_scene_acceptance'}


def inspection_camera(asset, candidate):
    vertices=np.load(asset['geometry'])['vertices']
    points,_=pose_points(vertices,candidate['matrix'],candidate['yaw'])
    camera=camera_for(points,candidate['elevation'])
    camera['fov']=math.degrees(2*math.atan(math.tan(math.radians(camera['fov']/2))*.62))
    return camera


def comparison_board(out, case, renders):
    """Every tile is a complete render, never a cropped or selected fragment."""
    side=1024; canvas=Image.new('RGB',(side*2,4*(side+40)),(112,112,112));draw=ImageDraw.Draw(canvas)
    for i,item in enumerate(renders):
        x,y=(i%2)*side,(i//2)*(side+40)
        with Image.open(out/item['image']) as im:
            im=im.resize((side,side),Image.Resampling.LANCZOS)
            canvas.paste(im,(x,y+40),im.getchannel('A'))
        draw.text((x+15,y+12),case['asset_id']+' / '+item['tag'],fill='white')
    path=out/case['asset_id']/'complete-comparison.png';canvas.save(path)
    return path.relative_to(out).as_posix()


def export_candidate(out, asset, candidate, params, finish, resolution):
    finish=refine_finish(finish,{k:params[k] for k in CONTROL_LIMITS if k in params})
    folder=out/asset['id'];request=render_request(asset,candidate,params,neutral_light())
    request.update(mesh_hash=asset['mesh_hash'],appearance='local',
        material_classes=asset['semantic']['material_classes'],dust_control=1.,generated_finish=finish,
        windows=render_windows(candidate,(3840,2160),resolution,full_frame=True),parts={},
        device='CPU',persistent_data=False,candidate_path=str(folder/'candidate.blend'))
    path=folder/'candidate-request.json';atomic_json(path,request)
    script=HERE/'export_asset_candidate.py'
    result=subprocess.run([str(BLENDER),'--background','--factory-startup','--python-exit-code','1',
                           '--python',str(script),'--',str(path)],capture_output=True,text=True,
                          encoding='utf8',errors='replace',timeout=120)
    (folder/'candidate-export.log').write_text(result.stdout+result.stderr,encoding='utf8')
    if result.returncode:raise RuntimeError('candidate export failed: '+str(folder/'candidate-export.log'))
    return read(folder/'candidate-check.json')


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
        if case.get('comparison'):
            body.append('<p><a href="'+case['comparison']+'">完整模型对照原图</a></p>')
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
    frozen=freeze(source,cache,pack,args.refinement,args.resolution)
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
                if args.refinement:
                    posed['camera']=inspection_camera(asset,posed)
                    params.update(asset_refinement='component_v1',oxide_gain=1.,scratch_gain=1.,relief_gain=1.,roughness_shift=0.)
                finish=case['finish'] if mode=='physical' else None
                folder,meta,hit=await renderer.get(asset,posed,params,neutral_light(light=='right'),(3840,2160),args.resolution,
                            appearance='local',full_frame=True,object_only=True,generated_finish=finish)
                image=out/case['asset_id']/(tag+'.png');image.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(folder/'object.png',image)
                with Image.open(image) as im:
                    if im.size!=(args.resolution,args.resolution):raise ValueError('unexpected diagnostic plate dimensions')
                    alpha[tag]=np.asarray(im.getchannel('A'),np.float32)/255.
                paths[tag]=folder
                if not np.any(alpha[tag]>.5):raise ValueError('empty diagnostic object')
                renders.append({'tag':tag,'label':{'clay':'灰模','source':'原 PBR','physical':'三层材质'}[mode]+' · '+
                                ('转向 55°' if turn else ('左侧柔光' if light=='left' else '右侧柔光')),
                                'image':image.relative_to(out).as_posix(),'sha256':file_hash(image),
                                'cache_hit':hit,'render_cache':meta['render_cache'],
                                'source_regions':coverage_stats(folder,alpha[tag]),
                                'material_metadata':meta.get('generated_finish')})
                if args.refinement and tag=='physical-left':
                    export_candidate(out,asset,posed,params,finish,args.resolution)
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
            if args.refinement:
                row['asset_profile']=descriptor(case['finish']['asset_profile'])
                row['candidate_check']=read(out/asset['id']/'candidate-check.json')
                row['comparison']=comparison_board(out,case,renders)
                maps=finish_maps[0][alpha['physical-left']>.98]
                row['layer_coverage']={'mean':maps.mean(axis=0).tolist(),'p95':np.quantile(maps,.95,axis=0).tolist(),
                                       'channels':['oxidation','abrasion','dust']}
            results.append(row);atomic_json(out/'results.json',results)
            write_report(out,results,{'state':'running','completed_assets':len(results),'api_calls':0})
    finally:
        await worker.close()
    summary={'version':frozen['version'],'state':'complete','assets':len(results),'plates':sum(len(r['renders']) for r in results),
             'wall_seconds':round(time.monotonic()-started,3),'worker_seconds':round(worker.seconds,3),
             'render_counts':dict(renderer.totals),'api_calls':0,'tokens':0,'api_cost':0,
             'local_compute_cost':'unknown','visual_status':'unreviewed'}
    atomic_json(out/'summary.json',summary);write_report(out,results,summary)
    print(json.dumps(summary,ensure_ascii=False))


async def add_turn_reference(args):
    """Add the missing source-PBR control for an already frozen +55° inspection."""
    out=resolve(args.output);frozen=read(out/'frozen.json');rows=read(out/'results.json')
    if not frozen.get('asset_refinement'):raise ValueError('turned control requires the frozen asset inspection camera')
    case=next(c for c in frozen['cases'] if c['asset_id']==args.add_turn_reference)
    row=next(r for r in rows if r['asset_id']==args.add_turn_reference)
    if any(r['tag']=='source-turned' for r in row['renders']):return
    asset=read(case['asset_record'])
    if file_hash(asset['mesh'])!=case['mesh_hash']:raise ValueError('source model changed')
    posed=copy.deepcopy(case['candidate']);posed['yaw']=float(posed['yaw'])+55
    posed['camera']=inspection_camera(asset,posed)
    params={'material':'original','light':'scene','dust':'auto','surface_dust':.5,
            'integration_mode':'coherent','render_detail':'standard','asset_diagnostic':'source',
            'asset_refinement':'component_v1','oxide_gain':1.,'scratch_gain':1.,'relief_gain':1.,'roughness_shift':0.}
    worker=Blender(out/'turn-reference-worker',args.device);renderer=RenderCache(resolve(args.cache),worker)
    start=time.monotonic()
    try:
        folder,meta,hit=await renderer.get(asset,posed,params,neutral_light(),(3840,2160),frozen['resolution'],
                                         appearance='local',full_frame=True,object_only=True)
    finally:await worker.close()
    image=out/asset['id']/'source-turned.png';shutil.copy2(folder/'object.png',image)
    row['renders'].append({'tag':'source-turned','label':'原 PBR · 同一转向 55° 对照','image':image.relative_to(out).as_posix(),
                           'sha256':file_hash(image),'cache_hit':hit,'render_cache':meta['render_cache'],
                           'same_view_as':'physical-turned'})
    row['comparison']=comparison_board(out,case,row['renders']);atomic_json(out/'results.json',rows)
    extra={'asset_id':asset['id'],'reason':'same_view_source_control_for_readable_barrel',
           'wall_seconds':time.monotonic()-start,'render_counts':dict(renderer.totals),'api_calls':0}
    atomic_json(out/'turn-reference-cost.json',extra)
    summary=read(out/'summary.json');summary['plates']=sum(len(r['renders']) for r in rows)
    summary['extra_control']=extra;atomic_json(out/'summary.json',summary);write_report(out,rows,summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=SOURCE)
    parser.add_argument('--cache',default='work/display-batch-cache')
    parser.add_argument('--pack',default=PACK)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('OPTIX','CPU'),default='OPTIX')
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--refinement',action='store_true',help='opt-in evidence profiles and asset inspection framing')
    parser.add_argument('--resolution',type=int,choices=(512,1024),default=512)
    parser.add_argument('--add-turn-reference',choices=ASSETS)
    args=parser.parse_args()
    asyncio.run(add_turn_reference(args) if args.add_turn_reference else run(args))


if __name__=='__main__':main()
