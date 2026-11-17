"""Offline verification of delivered coherent PNGs; never renders or calls an API."""
import argparse
import asyncio
import copy
import html
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np
from PIL import Image
from scene_v2 import Perception,digest,key
from display_batch.common import resolve,atomic_json,file_hash,implementation_manifest,load_catalog
from display_batch.compose import compose,sample_environment
from display_batch.effects import lighting_for
from display_batch.local_atmosphere import prepare as prepare_air
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig,EffectsReview,Review
from display_batch.strict_contact import visual_veto
from validate_coherent_expanded import source_hashes
from validate_reference_projection import read,records
from verify_geometry_rebuild import source_job,saved_render


def require(condition,reason):
    if not condition:raise ValueError(reason)


def cached_perception(original):
    def scene(engine,source):
        cache=engine.cache/f'scene-{key(digest(source),engine.versions)}.npz'
        require(cache.is_file(),'offline_perception_cache_missing')
        return original(engine,source)
    return scene


def delivery_checks(row,entry,candidate):
    """Stored visual/physical evidence must remain a gate after final-file copying."""
    require(row['status'] in ('review','accepted_display'),'image_delivered_with_invalid_status')
    require(entry.get('deliverable') is True,'best_iteration_not_deliverable')
    require(candidate.get('strict_contact') is True,'missing_strict_contact_mode')
    physics=candidate.get('terrain_physics',{})
    require(physics.get('stable') is True,'unstable_pose_delivered')
    require(not physics.get('failure_codes'),'physical_failure_delivered')
    require(candidate.get('contact_gap',{}).get('status')=='pass','missing_verified_contact_gap')
    require(candidate['contact_gap']['maximum_px']<=2.,'visible_gap_over_two_pixels')
    require(physics.get('axis_angle_deg',180)<=15.,'upright_pose_delivered')
    require(not any(c.get('status')=='fail' for c in entry['checks'].values()),'program_failure_delivered')
    review=(EffectsReview if 'trace' in entry['review'] else Review).model_validate(entry['review'])
    veto=visual_veto(review)
    require(not veto and not entry.get('physical_visual_veto'),'visual_physical_veto_delivered')
    if row['status']=='accepted_display':
        require(review.passes(),'accepted_without_visual_pass')
        require(all(c.get('status')=='pass' for c in entry['checks'].values()),'accepted_with_unknown_program_evidence')
    return {'stable':True,'contact_gap_px':candidate['contact_gap']['maximum_px'],
            'axis_angle_deg':physics['axis_angle_deg'],'visual_veto':[],
            'scene_physics_status':physics.get('status','unknown')}


def page(out,result):
    sections=[]
    for row in result['jobs']:
        images=''
        for label,path in (('原始完整背景',row.get('background')),('实际交付完整图',row.get('image'))):
            if path:
                uri=Path(path).resolve().as_uri()
                images+=f'<figure><figcaption>{label}</figcaption><a href="{html.escape(uri)}"><img loading="lazy" src="{html.escape(uri)}"></a></figure>'
        sections.append(f'<section><h2>{html.escape(row["job_id"])} · {html.escape(row["status"])}</h2>'
            f'<p>{html.escape(row.get("error","离线检查通过；物理未知仍为待审"))}</p>{images}</section>')
    summary={k:v for k,v in result.items() if k not in ('jobs','implementation','source_model_hashes')}
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>完整画面离线核验</title>'
        '<style>body{background:#15191d;color:#e8ecef;font:16px system-ui;margin:24px}img{width:100%;height:auto}figure{margin:16px 0}section{margin:32px 0}pre{white-space:pre-wrap}</style>'
        '<h1>完整画面离线核验</h1><p>仅复用既有渲染分层，不调用模型，不重新渲染。</p><pre>'
        +html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(sections),encoding='utf8')


async def verify(batch,out,manifest=None,limit=None,allow_partial=False):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    frozen=read(manifest or batch.parent/'manifest.json')
    expected=frozen['source_models'];before=source_hashes(load_catalog())
    require(len(expected)==44,'expected_44_source_models')
    require(before==expected,'source_models_changed_before_verification')
    inputs=read(batch/'inputs.json');snapshot=records(batch)
    require(len({r['job_id'] for r in snapshot})==len(snapshot),'duplicate_result_job_id')
    if limit is not None:snapshot=snapshot[:limit]
    pipe=Pipeline(BatchConfig.model_validate({**read(batch/'config.json'),'output':str(out),'offline':True}))
    async def forbidden(*args,**kwargs):raise RuntimeError('offline_verifier_forbids_blender')
    pipe.blender.ask=forbidden
    rows=[]
    try:
        with patch.object(Perception,'scene',cached_perception(Perception.scene)):
            for row in snapshot:
                checked={'job_id':row['job_id'],'asset_id':row['asset_id'],'background':row['background'],
                         'image':row.get('image'),'generated':bool(row.get('image')),'status':'pass'}
                try:
                    if not row.get('image'):
                        leaked=[str(batch/folder/(row['job_id']+'.png')) for folder in ('accepted','review','estimated')
                                if (batch/folder/(row['job_id']+'.png')).exists()]
                        require(not leaked,'rejected_result_has_delivery_file')
                        checked.update(result_status=row['status'],failure_reason=row.get('reason'),failed_result_not_delivered=True)
                    else:
                        job=source_job(batch,row)
                        entry=next(e for e in row['iterations'] if e['iteration']==row['best_iteration'])
                        require(file_hash(row['image'])==entry['image_hash'],'delivered_file_differs_from_reviewed_png')
                        candidate=copy.deepcopy(next(c for c in read(job/'candidates.json')['candidates'] if c['id']==entry['params']['candidate']))
                        checked['delivery']=delivery_checks(row,entry,candidate)
                        scene=await pipe.scenes.prepare(row['background'])
                        scene['atmosphere']=await pipe.atmospheres.prepare(scene)
                        params=entry['params'];folder,meta=saved_render(pipe.cache,entry)
                        require(params.get('integration_mode')=='coherent','wrong_integration_mode')
                        env,samples=sample_environment(scene,candidate,local=True)
                        env['integration_mode']=params['integration_mode']
                        env['atmospheric_context']=prepare_air(scene,candidate,env,pipe.cache)
                        env=lighting_for(scene,candidate,env,params['effects_lighting'],preserve=True)
                        env['calibration']=entry['photo']['calibration']
                        seed=next(j['seed'] for j in inputs if j['asset_id']==row['asset_id'] and j['background']==row['background'])
                        target=out/'replays'/(row['job_id']+'.png');target.parent.mkdir(exist_ok=True)
                        reproduced=compose(scene,candidate,folder,meta,env,samples,params,seed,target,capture_local=True)
                        require(file_hash(target)==entry['image_hash'],'PNG_replay_mismatch')
                        hashes={name:reproduced[name]==entry[name] for name in
                                ('alpha_hash','visible_alpha_hash','foreground_mask_hash','shadow_hash')}
                        require(all(hashes.values()),'geometry_or_foreground_or_shadow_hash_mismatch')
                        with Image.open(row['image']) as im:
                            size=list(im.size);final=np.asarray(im.convert('RGB'))
                        original=np.asarray(scene['frame'])
                        with Image.open(row['background']) as source:source_size=list(source.size)
                        require(size==list(scene['size'])==source_size,'output_dimensions_changed')
                        changed=np.any(final!=original,axis=2);bounds=entry['edit_bounds'];x,y,r,b=bounds
                        outside=changed.copy();outside[y:b,x:r]=False
                        protected=int(np.count_nonzero(changed&scene['protected']))
                        outside_pixels=int(np.count_nonzero(outside))
                        masked_changes=int(np.count_nonzero(changed[y:b,x:r]&~reproduced['local_layers']['writable']))
                        require(protected==outside_pixels==masked_changes==0,'protected_outside_or_foreground_pixels_changed')
                        require(not any(c.get('status')=='fail' for c in reproduced['checks'].values()),'replayed_program_check_failed')
                        require(reproduced['checks']['occlusion']['foreground_changed_pixels']==0,'foreground_changed')
                        checked.update(exact_png_replay=True,reviewed_png_hash=entry['image_hash'],size=size,
                            mask_hashes=hashes,protected_changed_pixels=protected,outside_edit_rectangle_changed_pixels=outside_pixels,
                            nonwritable_changed_pixels=masked_changes,foreground_changed_pixels=0,
                            checks=reproduced['checks'],replayed_image=str(target))
                        del reproduced
                except Exception as error:
                    checked.update(status='fail',error=f'{type(error).__name__}: {error}')
                rows.append(checked)
                print(json.dumps({'verified':row['job_id'],'status':checked['status'],'error':checked.get('error')},ensure_ascii=False),flush=True)
        after=source_hashes(load_catalog())
        require(not pipe.vision.calls,'offline_validation_sent_an_API_request')
        require(not pipe.renders.totals and pipe.blender.seconds==0,'offline_validation_rendered')
        complete=len(snapshot)==len(inputs)==frozen['denominator'] and limit is None
        result={'passed':all(r['status']=='pass' for r in rows) and before==after and (complete or allow_partial),
            'complete_batch':complete,'partial_snapshot_allowed':allow_partial,'batch':str(batch),
            'implementation':implementation_manifest(),'jobs':rows,'checked_jobs':len(rows),'expected_jobs':len(inputs),
            'exact_replays':sum(r.get('exact_png_replay',False) for r in rows),
            'failed_verifications':sum(r['status']=='fail' for r in rows),'source_models_count':len(after),
            'source_models_unchanged':before==after==expected,'source_model_hashes':after,
            'new_api_requests':0,'new_render_count':0,'worker_seconds':0.,'seconds':time.monotonic()-started,
            'scope':'actual PNG byte replay and pixel protection; existing estimated geometry remains review'}
        atomic_json(out/'verification.json',result);page(out,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','implementation','source_model_hashes')},ensure_ascii=False),flush=True)
        return result
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--out',required=True)
    p.add_argument('--manifest');p.add_argument('--limit',type=int);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args()
    result=asyncio.run(verify(resolve(a.batch),resolve(a.out),resolve(a.manifest) if a.manifest else None,a.limit,a.allow_partial))
    raise SystemExit(0 if result['passed'] else 1)
