"""Fixed-geometry comparison inputs from archived display jobs."""

from display_batch.common import ROOT, file_hash, signature
from validate_local_fusion_six import read
from verify_geometry_rebuild import saved_render


ARCHIVED_EXTINGUISHER = (
    ('36_6_6', 'test-all-two-scenes-20261113', '0035-65d4bd7c6d', True),
    ('53_7_35', 'five-items-three-new-scenes-20261113', '0005-dc0e6fdd85', False),
)

ARCHIVED_JACK = (
    ('36_6_6', 'test-all-two-scenes-20261113', '0008-c6599e2f08', True),
    ('53_4_21', 'test-all-two-scenes-20261113', '0049-0de1f8c71e', True),
)


def extinguisher_cases(cache):
    return archived_cases(cache, '33', ARCHIVED_EXTINGUISHER)


def jack_cases(cache):
    return archived_cases(cache, '11', ARCHIVED_JACK)


def archived_cases(cache, asset_id, archives):
    # This diagnostic loader is only called by the fixed-frame validator.
    from validate_generated_pbr import Case, RENDER_SOURCE_FILES

    cases = []
    for group, archive, job_id, old_shadow_key in archives:
        run = ROOT / 'work' / archive / 'run'
        job = run / 'jobs' / job_id
        report = read(job / 'result.json')
        if report['asset_id'] != asset_id or report['group'].replace('-', '_') != group:
            raise ValueError(f'wrong archived metal object: {job_id}')
        entry = next(row for row in report['iterations'] if row['iteration'] == report['best_iteration'])
        inputs = next(row for row in read(run / 'inputs.json')
                      if row['background'] == report['background'] and row['asset_id'] == asset_id)
        image_hash = file_hash(report['image'])
        if image_hash != entry['image_hash'] or file_hash(report['background']) != inputs['background_hash']:
            raise ValueError(f'archived metal frame or background changed: {job_id}')
        asset_record = cache / 'assets' / report['asset_signature'] / 'result-strict.json'
        if read(asset_record)['mesh_hash'] != inputs['mesh_hash']:
            raise ValueError(f'archived metal asset changed: {job_id}')
        params = entry['params']
        candidate = next(row for row in read(job / 'candidates.json')['candidates']
                         if row['id'] == params['candidate'])
        folder, meta = saved_render(cache, entry)
        sources = read(run / 'implementation.json')['sources']
        sources = {key.replace('\\', '/'): value for key, value in sources.items()}
        render_deps = signature('blender-4.5.3', {name: sources[name] for name in RENDER_SOURCE_FILES})
        reference = {'signature': entry['render_cache']['without_shadow']['signature'],
                     'dependencies': render_deps, 'standard_profile_without_detail_key': old_shadow_key}
        cases.append(Case(f'{group}-{asset_id}', group, asset_id, asset_record, report['scene_signature'],
                          report['background'], report['image'], image_hash, candidate, params, entry,
                          folder, meta, reference, entry['render_resolution'], inputs['seed']))
    return cases
