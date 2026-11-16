"""Fixed-geometry comparison inputs for the 33 extinguisher."""

from display_batch.common import ROOT, file_hash, signature
from validate_local_fusion_six import read
from verify_geometry_rebuild import saved_render


ARCHIVED_EXTINGUISHER = (
    ('36_6_6', 'test-all-two-scenes-20261113', '0035-65d4bd7c6d', True),
    ('53_7_35', 'five-items-three-new-scenes-20261113', '0005-dc0e6fdd85', False),
)


def extinguisher_cases(cache):
    # This diagnostic loader is only called by the fixed-frame validator.
    from validate_generated_pbr import Case, RENDER_SOURCE_FILES

    cases = []
    for group, archive, job_id, old_shadow_key in ARCHIVED_EXTINGUISHER:
        run = ROOT / 'work' / archive / 'run'
        job = run / 'jobs' / job_id
        report = read(job / 'result.json')
        if report['asset_id'] != '33' or report['group'].replace('-', '_') != group:
            raise ValueError(f'wrong archived extinguisher: {job_id}')
        entry = next(row for row in report['iterations'] if row['iteration'] == report['best_iteration'])
        inputs = next(row for row in read(run / 'inputs.json')
                      if row['background'] == report['background'] and row['asset_id'] == '33')
        image_hash = file_hash(report['image'])
        if image_hash != entry['image_hash'] or file_hash(report['background']) != inputs['background_hash']:
            raise ValueError(f'archived extinguisher frame or background changed: {job_id}')
        asset_record = cache / 'assets' / report['asset_signature'] / 'result-strict.json'
        if read(asset_record)['mesh_hash'] != inputs['mesh_hash']:
            raise ValueError(f'archived extinguisher asset changed: {job_id}')
        params = entry['params']
        candidate = next(row for row in read(job / 'candidates.json')['candidates']
                         if row['id'] == params['candidate'])
        folder, meta = saved_render(cache, entry)
        sources = read(run / 'implementation.json')['sources']
        sources = {key.replace('\\', '/'): value for key, value in sources.items()}
        render_deps = signature('blender-4.5.3', {name: sources[name] for name in RENDER_SOURCE_FILES})
        reference = {'signature': entry['render_cache']['without_shadow']['signature'],
                     'dependencies': render_deps, 'standard_profile_without_detail_key': old_shadow_key}
        cases.append(Case(f'{group}-33', group, '33', asset_record, report['scene_signature'],
                          report['background'], report['image'], image_hash, candidate, params, entry,
                          folder, meta, reference, entry['render_resolution'], inputs['seed']))
    return cases
