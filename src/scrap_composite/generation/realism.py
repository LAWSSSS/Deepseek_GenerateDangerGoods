"""Opt-in, shared configuration for the isolated realism experiment."""
VERSION = 'realism-v1'
DEFAULT_PACK = 'work/render-materials/nearby-worn-steel-v15/pack.json'


def preset_values(values):
    result = dict(values)
    result.update(workflow='evidence_v1', review_profile='photographer',
                  integration_mode='coherent', camera_pipeline='premultiplied',
                  support_geometry='structured', appearance='local', photo_style='dusty_scrap',
                  chroma_mode='local_budget')
    effects = dict(result.get('effects') or {})
    effects.update(surface_finish='physical_layers', weathering='preserve_source',
                   fog_mode='local_auto', fog_extra=0., lighting='local',
                   shadow_mode='matched_receiver', rust_spots=0., rust_relief=False,
                   rust_relief_strength=1.)
    if not effects.get('surface_finish_manifest'):
        effects['surface_finish_manifest']=DEFAULT_PACK
    effects.setdefault('surface_dust', .5)
    effects.setdefault('pool_kernel', 1)
    effects.setdefault('pool_weight', 0.)
    result['effects'] = effects
    return result


def enabled(params):
    return params.get('realism_version') == VERSION
