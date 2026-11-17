"""Reproducible feedback batch: retained archive geometry plus automatically placed new families."""
import argparse
import asyncio
import json
from pathlib import Path

from display_batch.common import atomic_json, resolve, file_hash, load_catalog
from display_batch.pipeline import Pipeline
from display_batch.policy import resize_candidate
from display_batch.schema import BatchConfig
from display_batch.compose import sample_environment
from display_batch.costs import tariff, estimate
from validate_display_local import cached_case, records


class FeedbackPipeline(Pipeline):
    def __init__(self,config,baseline):
        super().__init__(config)
        self.baseline=baseline
        self.archive={(r['asset_id'],str(Path(r['background']).resolve())):r for r in records(baseline) if r.get('image')}
        self.frozen={}

    async def choose_placements(self,asset,scene,seed):
        key=(asset['id'],str(Path(scene['path']).resolve()))
        row=self.archive.get(key)
        if not row:return await super().choose_placements(asset,scene,seed)
        old,candidate,_,_,render,meta=await cached_case(self.baseline,row,self.cache,self.scenes)
        candidate['feedback_baseline']={'job_id':row['job_id'],'image':row['image'],
            'measured_major_px':row['measured_major_px'],'expected_major_px':candidate['expected_major_px']}
        factor=self.config.asset_scale_factors.get(asset['id'],1.)
        candidate=await asyncio.to_thread(resize_candidate,asset,scene,candidate,factor)
        candidate['feedback_frozen']=factor==1
        self.frozen[(asset['id'],candidate['id'])]=(render,meta)
        return [candidate],[]

    async def validate_local_pose(self,asset,scene,candidate):
        if candidate.get('feedback_frozen'):return candidate
        return await super().validate_local_pose(asset,scene,candidate)

    async def render(self,asset,scene,candidate,params,resolution):
        if candidate.get('feedback_frozen'):
            render,meta=self.frozen[(asset['id'],candidate['id'])]
            env,samples=await asyncio.to_thread(sample_environment,scene,candidate,local=True)
            meta={**meta,'render_cache':{part:{'hit':True,'reason':'frozen_archive_reuse'}
                    for part in ('object','with_shadow','without_shadow')},'render_timings':{},'asset_imports':0}
            return render,meta,env,samples,True
        return await super().render(asset,scene,candidate,params,resolution)


async def main(args):
    out=resolve(args.out);selection=json.loads((out/'selection.json').read_text(encoding='utf8'))
    workflow={name:file_hash(Path(__file__).parent/name) for name in
        ('feedback_display.py','validate_display_local.py','validate_display_costs.py')}
    manifest=[{'background':bg,'asset_id':ident,'group':'user_feedback','workflow_sources':workflow,
               'seed':selection['seed']+i*len(selection['asset_ids'])+j}
              for i,bg in enumerate(selection['backgrounds']) for j,ident in enumerate(selection['asset_ids'])]
    atomic_json(out/'manifest.json',manifest)
    config=BatchConfig(backgrounds=str(Path(selection['backgrounds'][0]).parent),output=str(out),
        assets=selection['asset_ids'],count=len(manifest),manifest=str(out/'manifest.json'),
        appearance='local',photo_style='dusty_scrap',max_iterations=0,seed=selection['seed'])
    pipeline=FeedbackPipeline(config,resolve(args.baseline))
    # Attribute already incurred preparation calls using the same configured service.
    prep=out/'usage-preparation.json'
    if prep.is_file():
        doc=json.loads(prep.read_text(encoding='utf8'))
        if 'tariff' not in doc and all(c.get('model')==pipeline.vision.endpoint.model for c in doc['calls']):
            rates=tariff(config,pipeline.vision.endpoint)
            doc.update(tariff=rates,cost=estimate(doc['calls'],rates),ledger_name=prep.name)
            atomic_json(prep,doc)
    hashes={k:file_hash(v['mesh']) for k,v in load_catalog().items() if v['available']}
    atomic_json(out/'source-hashes-before.json',hashes)
    result=await pipeline.run()
    atomic_json(out/'source-hashes-after.json',{k:file_hash(v['mesh']) for k,v in load_catalog().items() if v['available']})
    print(json.dumps({k:v for k,v in result.items() if k!='usage_current_run'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--baseline',default='outputs/摄影嵌入-验证-v2-20261110/regression')
    asyncio.run(main(parser.parse_args()))
