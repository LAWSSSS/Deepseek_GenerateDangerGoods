"""Recheck the 24 existing placements with new scrap materials and confirmed sizes."""
import argparse
import asyncio
import json
from pathlib import Path

from display_batch.common import HERE, atomic_json, file_hash, resolve, signature
from display_batch.compose import compose, sample_environment
from display_batch.framing import final_resolution, render_windows
from display_batch.pipeline import Pipeline
from display_batch.policy import resize_candidate
from display_batch.review import evidence_status
from display_batch.schema import BatchConfig
from validate_display_costs import selected_geometry
from validate_display_local import cached_case, read, records


BASELINE = 'outputs/点评样片-低饱和扬尘-20261111'
ARCHIVE = 'outputs/摄影嵌入-验证-v2-20261110/regression'


class ScrapFeedback(Pipeline):
    def __init__(self,config,baseline):
        super().__init__(config)
        self.baseline = baseline
        self.previous = {(r['asset_id'],str(Path(r['background']).resolve())):r for r in records(baseline)}
        self.archive = {}

    async def choose_placements(self,asset,scene,seed):
        previous = self.previous[(asset['id'],str(Path(scene['path']).resolve()))]
        best,candidate = selected_geometry(self.baseline,previous)
        factor = self.config.asset_scale_factors.get(asset['id'],1.)
        updated = await asyncio.to_thread(resize_candidate,asset,scene,candidate,factor)
        resized = updated is not candidate
        updated = dict(updated)
        updated['fixed_pose'] = True
        updated['feedback_resized'] = resized
        updated['feedback_baseline'] = {
            'job_id':previous['job_id'],'image':previous['image'],
            'measured_major_px':best['checks']['scale']['measurement_px'],
            'expected_major_px':candidate['expected_major_px'],
            'pose_signature':signature(candidate['matrix'],candidate['camera'],candidate['anchor_px'],candidate['full_px'])}
        if not resized:
            directory,window = await self.original_coverage(asset,scene,previous,best,candidate)
            updated.update(fixed_alpha_dir=str(directory),fixed_alpha_window=window)
        return [updated],[]

    async def original_coverage(self,asset,scene,previous,best,candidate):
        entry = best['render_cache']['object']
        if entry.get('signature'):
            record = self.renders.root/'object'/entry['signature']/'result.json'
            window = render_windows(candidate,scene['size'],final_resolution(candidate))['object']
            return Path(read(record)['path']),window
        if not self.archive:
            self.archive = {r['job_id']:r for r in records(resolve(ARCHIVE))}
        archived = self.archive[candidate['feedback_baseline']['job_id']]
        _,_,_,_,directory,_ = await cached_case(resolve(ARCHIVE),archived,self.cache,self.scenes)
        return directory,None

    async def validate_local_pose(self,asset,scene,candidate):
        if candidate['feedback_resized']:
            return await super().validate_local_pose(asset,scene,candidate)
        return candidate

    async def render(self,asset,scene,candidate,params,resolution):
        comparison = None
        if candidate['feedback_resized'] and not candidate.get('fixed_alpha_dir'):
            comparison = await self.render_size_control(asset,scene,candidate,resolution)
            candidate.update(fixed_alpha_dir=comparison['object_dir'],fixed_alpha_window=comparison['object_window'])
        folder,meta,environment,samples,hit = await super().render(asset,scene,candidate,params,resolution)
        if comparison is not None and comparison.get('newly_rendered'):
            # Attribute the control's actual work to the same job and keep the part names separate.
            meta['render_cache'] = {**meta['render_cache'],
                **{'control_'+k:v for k,v in comparison['render_cache'].items()}}
            meta['render_timings'] = {**meta['render_timings'],
                **{'control_'+k:v for k,v in comparison['render_timings'].items()}}
            meta['asset_imports'] += comparison['asset_imports']
        return folder,meta,environment,samples,hit

    async def render_size_control(self,asset,scene,candidate,resolution):
        baseline_id = candidate['feedback_baseline']['job_id']
        output = self.out/'size-controls'/baseline_id
        marker = output/'control.json'
        key = signature(asset['mesh_hash'],candidate,resolution,file_hash(__file__))
        if marker.is_file():
            previous = read(marker)
            if previous['signature']==key and Path(previous['image']).is_file() and Path(previous['object_dir']).is_dir():
                return previous
        output.mkdir(parents=True,exist_ok=True)
        # Reuse the exact current geometry and the previous batch's chosen photographic settings.
        previous = self.previous[(asset['id'],str(Path(scene['path']).resolve()))]
        best,_ = selected_geometry(self.baseline,previous)
        params = best['params']
        environment,samples = await asyncio.to_thread(sample_environment,scene,candidate,local=True)
        folder,meta,hit = await self.renders.get(asset,candidate,params,{**environment,'haze':0.},
                                                scene['size'],resolution,appearance='local')
        old_inputs = read(self.baseline/'inputs.json')
        ordinal = int(baseline_id.split('-')[0])-1
        control = await asyncio.to_thread(compose,scene,candidate,folder,meta,environment,samples,
                                          params,old_inputs[ordinal]['seed'],output/'source-material.png')
        if evidence_status(control['checks'])=='fail':
            raise ValueError('same_size_control_failed_program_checks')
        control.update(signature=key,object_dir=str(folder),object_window=meta['object_window'],
                       render_cache=meta['render_cache'],render_timings=meta['render_timings'],
                       asset_imports=meta['asset_imports'],render_cache_hit=hit,image_hash=file_hash(control['image']))
        atomic_json(marker,control)
        return {**control,'newly_rendered':True}


async def run(args):
    baseline = resolve(args.baseline)
    output = resolve(args.out)
    rows = records(baseline)
    old_inputs = read(baseline/'inputs.json')
    if len(rows)!=24 or len(old_inputs)!=24 or not all(r.get('image') for r in rows):
        raise ValueError('需要上一轮完整的 24 张点评图及冻结摆位')
    if output==baseline:
        raise ValueError('新材质对照需要独立输出目录')
    hashes = {name:file_hash(HERE/name) for name in
              ('feedback_scrap_material.py','validate_display_costs.py','validate_display_local.py')}
    manifest = []
    for row in rows:
        ordinal = int(row['job_id'].split('-')[0])-1
        previous = old_inputs[ordinal]
        best,candidate = selected_geometry(baseline,row)
        if file_hash(row['image'])!=best['image_hash']:
            raise ValueError('点评基线图片已经改变')
        manifest.append({**previous,'group':'scrap_material_feedback','baseline_job_id':row['job_id'],
                         'baseline_image_hash':best['image_hash'],'baseline_pose':signature(candidate),
                         'workflow_sources':hashes})
    manifest_path = output/'manifest.json'
    if manifest_path.is_file() and read(manifest_path)!=manifest:
        raise ValueError('冻结输入改变，请使用新输出目录')
    atomic_json(manifest_path,manifest)
    config = BatchConfig(backgrounds=str(Path(manifest[0]['background']).parent),output=str(output),
        count=len(manifest),manifest=str(manifest_path),photo_style='dusty_scrap',
        material_adaptation='local_scrap',max_iterations=args.iterations,offline=args.offline,
        provider_config=args.provider_config)
    pipeline = ScrapFeedback(config,baseline)
    before = {k:file_hash(v['mesh']) for k,v in pipeline.catalog.items() if v['available']}
    for job in manifest:
        if before[job['asset_id']]!=job['mesh_hash'] or file_hash(job['background'])!=job['background_hash']:
            raise ValueError('基线背景或模型内容发生变化')
    source_record = output/'source-hashes-before.json'
    if not source_record.is_file():
        atomic_json(source_record,before)
    atomic_json(output/'baseline.json',{'directory':str(baseline),'jobs':len(rows),'scale_factors':config.asset_scale_factors})
    summary = await pipeline.run()
    after = {k:file_hash(v['mesh']) for k,v in pipeline.catalog.items() if v['available']}
    atomic_json(output/'source-hashes-after.json',after)
    if after!=read(source_record):
        raise ValueError('源模型散列发生变化')
    print(json.dumps({k:v for k,v in summary.items() if k!='usage_current_run'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--baseline',default=BASELINE)
    parser.add_argument('--iterations',type=int,default=2)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--provider-config')
    asyncio.run(run(parser.parse_args()))
