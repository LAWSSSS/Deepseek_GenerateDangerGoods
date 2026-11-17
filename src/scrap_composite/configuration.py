"""Explicit settings for the evidence-based workflow."""


def evidence_settings(values, explicit_fields):
    settings = dict(values)
    explanations = {}
    required = {
        'integration_mode': 'coherent',
        'camera_pipeline': 'premultiplied',
        'support_geometry': 'structured',
    }
    for field, value in required.items():
        if settings.get(field) != value:
            explanations[field] = 'evidence_v1'
            settings[field] = value
    effects = settings.get('effects')
    if effects is None:
        settings['effects'] = {
            'fog_extra': 0., 'pool_kernel': 1, 'pool_weight': 0.,
            'surface_dust': .5, 'lighting': 'local', 'fog_mode': 'local_auto',
            'weathering': 'preserve_source', 'shadow_mode': 'matched_receiver',
        }
        explanations['effects'] = 'local atmosphere and a single camera blur'
    elif 'effects' in explicit_fields:
        explanations['effects'] = 'explicit settings, subject to the shared camera blur budget'
    return settings, explanations
