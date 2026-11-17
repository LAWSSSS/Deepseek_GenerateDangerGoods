"""Cache-only diagnostics: no model requests and no composite image generation."""
import asyncio
import time

from .common import atomic_json
from .pipeline import Pipeline,make_jobs,batch_lock
from .scene import choose_candidates


async def diagnose(config):
    config=config.model_copy(update={'offline':True})
    pipe=Pipeline(config);records=[];started=time.monotonic()
    try:
        with batch_lock(pipe.out):
            for job in make_jobs(config,pipe.catalog):
                record={'asset_id':job['asset_id'],'background':job['background'],'status':'unknown'}
                try:
                    asset=await pipe.prepare_asset(job['asset_id'])
                    record.update(physics_signature=asset.get('physics_signature'),
                                  stable_poses=len(asset['poses']),asset_reason=asset.get('reason'))
                    if asset['poses']:
                        scene=await pipe.scenes.prepare(job['background'])
                        candidates,rejected=await asyncio.to_thread(choose_candidates,asset,scene,job['seed'],config)
                        record.update(scene_signature=scene['key'],candidates=candidates,rejected=rejected,
                                      status='pass' if candidates else 'review')
                except Exception as error:record['reason']=str(error)[:300]
                records.append(record)
                atomic_json(pipe.out/'diagnostics.json',{'mode':'offline_no_composite','formal_status':'unchanged',
                    'records':records,'seconds':round(time.monotonic()-started,3),'api_calls':len(pipe.vision.calls)})
    finally:
        await pipe.blender.close();await pipe.vision.close();pipe.store.close()
    return {'diagnosed':len(records),'api_calls':len(pipe.vision.calls),'composite_images':0}
