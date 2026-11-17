"""Test verified six-view Blender candidates in two real monitoring frames."""
import argparse
import asyncio
import json
import time
from pathlib import Path

from display_batch.common import atomic_json, file_hash, load_catalog, resolve
from display_batch.compose import compose
from display_batch.pipeline import Pipeline, final_resolution
from display_batch.schema import BatchConfig
from validate_estimated_preview import freeze, job_key, preview_records, report, read, source_hashes


ASSETS = ('1', '25', '33')
SCENES = ('36_6_6', '53_7_35')


class SixViewTrial(Pipeline):
    def default_params(self, candidate):
        # Reuse the local light and camera stages without generating another
        # material, or matching the object's tonal range to the ground.
        return {**super().default_params(candidate), 'realism_version':'realism-v1'}


def verified_catalog(manifest, frozen):
    library = read(manifest)
    if library['version'] != 'packed-reference-candidates-v1' or set(library['assets']) != set(ASSETS):
        raise ValueError('Expected the three verified packed six-view candidates')
    catalog = load_catalog()
    if source_hashes(catalog) != frozen['source_models']:
        raise ValueError('A source model differs from the frozen inventory')
    for aid, record in library['assets'].items():
        row = catalog[aid]
        path = Path(record['path']).resolve()
        if (not path.is_relative_to(manifest.parent) or path.suffix.lower() != '.blend'
                or row['mesh_hash'] != record['source_mesh_hash'] or not record['reopened']
                or file_hash(path) != record['sha256']):
            raise ValueError(f'Candidate hash, source identity or validation differs for {aid}')
        catalog[aid] = {**row, 'source_mesh':row['mesh'], 'source_mesh_hash':row['mesh_hash'],
                        'mesh':str(path), 'mesh_hash':record['sha256'], 'packed_subject':record['subject']}
    return catalog


async def run(out, backgrounds, manifest, scenes, provider_config):
    frozen = freeze(out, backgrounds, ASSETS, scenes)
    catalog = verified_catalog(manifest, frozen)
    raw = {**read(out/'config.json'), 'review_profile':'photographer', 'max_iterations':1,
           'effects':{'fog_extra':0., 'pool_kernel':1, 'pool_weight':0., 'surface_dust':0.,
                      'lighting':'local', 'ai_adjust':False, 'fog_mode':'local_auto',
                      'weathering':'preserve_source', 'shadow_mode':'matched_receiver'}}
    if provider_config is not None:
        raw['provider_config'] = str(provider_config)
    inputs = {'manifest_sha256':file_hash(manifest), 'manifest':str(manifest), 'config':raw,
              'selection':{'objects':list(ASSETS), 'scenes':list(scenes), 'basis':'one camera 36 and one camera 53'}}
    binding = out/'trial-inputs.json'
    if binding.is_file() and read(binding) != inputs:
        raise ValueError('Trial inputs changed. Use a fresh output directory.')
    atomic_json(binding, inputs)
    await SixViewTrial(BatchConfig.model_validate(raw), catalog=catalog).run()
    return validation(out)


def preserve_program_checked_draft(out, row, expected):
    if row.get('image') or row.get('reason') != 'review_invalid_after_one_repair':
        return row
    draft_path = Path(row['draft_record']).resolve()
    if not draft_path.is_relative_to(out) or draft_path.parent.name != row['job_id']:
        raise ValueError('Draft does not belong to the current test')
    draft = read(draft_path)
    if (draft.get('packed_subject') != expected or not draft.get('checks')
            or any(check['status'] == 'fail' for check in draft['checks'].values())
            or file_hash(draft['image']) != draft['image_hash']):
        raise ValueError('An unreviewed draft failed its candidate or pixel guards')
    entry = {**draft, 'review':None, 'review_status':'unknown', 'unreviewed_draft':True,
             'review_error':'invalid_after_one_repair', 'deliverable':True}
    checked = {**row, 'iterations':[*row['iterations'], entry]}
    Pipeline.finish_image(out/'待审复核', row['job_id'], checked, entry)
    return checked


def validation(out):
    binding = read(out/'trial-inputs.json')
    manifest = Path(binding['manifest'])
    if file_hash(manifest) != binding['manifest_sha256']:
        raise ValueError('Candidate manifest changed during this trial')
    frozen = read(out/'frozen.json')
    catalog = verified_catalog(manifest, frozen)
    if any(file_hash(src['path']) != src['sha256'] for src in frozen['backgrounds'].values()):
        raise ValueError('Background changed during this trial')
    records = preview_records(out)
    for key, row in records.items():
        asset = catalog[row['asset_id']]
        expected = {'object':asset['packed_subject'], 'sha256':asset['mesh_hash']}
        if row.get('image'):
            best = next(e for e in row['iterations'] if e['iteration'] == row['best_iteration'])
            if best.get('packed_subject') != expected:
                raise ValueError('Image is not bound to the selected packed candidate')
        elif row.get('draft_record'):
            records[key] = preserve_program_checked_draft(out, row, expected)
    result = report(out, trial_records=records)
    checks = read(out/'validation.json')
    checks['candidate_manifest'] = str(manifest)
    checks['candidate_manifest_sha256'] = binding['manifest_sha256']
    atomic_json(out/'validation.json', checks)
    return result


async def retry_surface(out):
    # A wide single plane can avoid unstable slopes in the first shortlist.
    # Reuse the same model, scene scale, check thresholds and two photographs.
    binding = read(out/'trial-inputs.json')
    manifest = Path(binding['manifest'])
    if file_hash(manifest) != binding['manifest_sha256']:
        raise ValueError('Candidate manifest changed before the surface retry')
    frozen = read(out/'frozen.json')
    catalog = verified_catalog(manifest, frozen)
    if any(file_hash(src['path']) != src['sha256'] for src in frozen['backgrounds'].values()):
        raise ValueError('Background changed before the surface retry')
    batch = out/'attempts'/'single-surface'
    jobs_path = batch/'retry-jobs.json'
    if jobs_path.is_file():
        jobs = read(jobs_path)
    else:
        rows = preview_records(out)
        recoverable = {'no_geometric_candidate', 'no_candidate_survived_program_checks',
                       'no_deliverable_candidate_after_physical_veto'}
        jobs = [job for job in frozen['jobs'] if rows.get(job_key(job), {}).get('reason') in recoverable
                and not rows[job_key(job)].get('image')]
        if not jobs:
            return validation(out)
        atomic_json(jobs_path, jobs)
    if not jobs or any(job not in frozen['jobs'] for job in jobs):
        raise ValueError('Surface retry jobs are outside the frozen trial')
    raw = {**binding['config'], 'output':str(batch), 'manifest':str(jobs_path),
           'count':len(jobs), 'placement_mode':'surface'}
    await SixViewTrial(BatchConfig.model_validate(raw), catalog=catalog).run()
    return validation(out)


def frozen_render(pipe, entry, packed):
    parts = {}
    for part, cache in entry['render_cache'].items():
        key = cache['signature']
        if len(key) != 64 or any(ch not in '0123456789abcdef' for ch in key):
            raise ValueError('Invalid saved render signature')
        parts[part], _ = pipe.renders.ready(pipe.renders.root/part/key, part)
        if not parts[part]:
            raise ValueError('Saved render channel is missing')
    obj, with_shadow, without_shadow = (parts[k] for k in ('object', 'with_shadow', 'without_shadow'))
    if (obj['camera'] != with_shadow['camera'] or obj['camera'] != without_shadow['camera']
            or with_shadow['window'] != without_shadow['window']
            or with_shadow['receiver_signature'] != without_shadow['receiver_signature']):
        raise ValueError('Saved render part grids or receivers differ')
    meta = {k:obj[k] for k in ('camera', 'device', 'anchor_uv', 'contact_polygons',
                              'material_readback', 'lighting_readback') if k in obj}
    meta.update(object_window=obj['window'], shadow_window=with_shadow['window'],
                part_paths={k:v['path'] for k,v in parts.items()},
                resolution=entry['render_resolution'], packed_subject=packed,
                shadow_pair_validation=entry['checks'].get('shadow_pair'))
    return Path(obj['path']), meta, True


async def recompose_atmosphere(out):
    # Reapply only the current air field to already eligible geometry. Any
    # old visual scores remain attached to the preceding exported image.
    from display_batch.scene_atmosphere import VERSION

    binding = read(out/'trial-inputs.json')
    manifest = Path(binding['manifest'])
    if file_hash(manifest) != binding['manifest_sha256']:
        raise ValueError('Candidate manifest changed before recomposition')
    frozen = read(out/'frozen.json')
    catalog = verified_catalog(manifest, frozen)
    batch = out/'attempts'/'z-air-field'
    raw = {**binding['config'], 'output':str(batch), 'offline':True}
    pipe = SixViewTrial(BatchConfig.model_validate(raw), catalog=catalog)
    rows = preview_records(out)
    previous = [row for row in rows.values() if not row.get('recomposed_from')]
    saved = {key:row for key,row in rows.items() if row.get('recomposed_from')}
    results = []
    started = time.monotonic()
    try:
        for row in previous:
            asset = catalog[row['asset_id']]
            packed = {'object':asset['packed_subject'], 'sha256':asset['mesh_hash']}
            row = preserve_program_checked_draft(out, row, packed)
            if not row.get('image'):
                continue
            best = next(e for e in row['iterations'] if e['iteration'] == row['best_iteration'])
            atmosphere = best['photo'].get('atmosphere', {})
            if atmosphere.get('version') == VERSION:
                continue
            if best.get('packed_subject') != packed or file_hash(best['image']) != best['image_hash']:
                raise ValueError('Previous image or packed model differs')
            payload = next(j for j in frozen['jobs'] if job_key(j) == job_key(row))
            if file_hash(payload['background']) != row['background_hash']:
                raise ValueError('Background changed before recomposition')
            parent = Path(best['image']).parent
            choices = read(parent/'candidates.json')['candidates']
            candidate = next(c for c in choices if c['id'] == best['params']['candidate'])
            scene = await pipe.evidence_scene(row)
            asset = await pipe.prepare_asset(row['asset_id'])
            params = best['params']
            frozen_channels = frozen_render(pipe, best, packed)

            async def use_frozen_channels(*args, **kwargs):
                return frozen_channels

            normal_render = pipe.renders.get
            pipe.renders.get = use_frozen_channels
            try:
                render, meta, environment, samples, hit = await pipe.render(
                    asset, scene, candidate, params, final_resolution(candidate))
            finally:
                pipe.renders.get = normal_render
            target = batch/'frames'/row['job_id']
            target.mkdir(parents=True, exist_ok=True)
            comp = await asyncio.to_thread(compose, scene, candidate, render, meta, environment, samples,
                                            params, payload['seed'], target/'image.png')
            for field in ('alpha_hash', 'visible_alpha_hash', 'foreground_mask_hash', 'shadow_hash'):
                if comp[field] != best[field]:
                    raise ValueError('Recomposition changed frozen '+field)
            if any(check['status'] == 'fail' for check in comp['checks'].values()):
                raise ValueError('Recomposed frame failed pixel or geometry checks')
            entry = {**comp, 'params':params, 'iteration':best['iteration'], 'packed_subject':packed,
                     'image_hash':file_hash(comp['image']), 'review':None, 'review_status':'unknown',
                     'deliverable':True, 'render_cache_hit':hit}
            revised = {k:v for k,v in row.items()
                       if k not in ('iterations', 'insertion_trace', 'draft_record', 'actions')}
            revised.update(iterations=[entry], reason='air_field_updated_visual_review_pending',
                           recomposed_from={'image':best['image'], 'sha256':best['image_hash'],
                                           'atmosphere_version':atmosphere.get('version')})
            Pipeline.finish_image(batch, row['job_id'], revised, entry)
            results.append(revised)
            saved[job_key(revised)] = revised
    finally:
        await pipe.blender.close()
        await pipe.vision.close()
        pipe.store.close()
    if saved:
        (batch/'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in saved.values()),
                                           encoding='utf8')
    atomic_json(batch/f'run-{pipe.run_id}.json', {
        'seconds':round(time.monotonic()-started,3), 'gpu_seconds':round(pipe.blender.seconds,3),
        'api_seconds':0., 'render_counts':{'archived_pass_sets_reused':len(results)},
        'air_field_version':VERSION,
        'recomposed':len(results)})
    return validation(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report', 'retry-surface', 'recompose-atmosphere'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--backgrounds')
    parser.add_argument('--candidates')
    parser.add_argument('--scenes', nargs=2, default=SCENES)
    parser.add_argument('--provider-config')
    args = parser.parse_args()
    out = resolve(args.out)
    if args.command == 'run':
        if not args.backgrounds or not args.candidates:
            parser.error('run requires --backgrounds and --candidates')
        value = asyncio.run(run(out, resolve(args.backgrounds), resolve(args.candidates), args.scenes,
                                resolve(args.provider_config) if args.provider_config else None))
    elif args.command == 'retry-surface':
        value = asyncio.run(retry_surface(out))
    elif args.command == 'recompose-atmosphere':
        value = asyncio.run(recompose_atmosphere(out))
    else:
        value = validation(out)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
