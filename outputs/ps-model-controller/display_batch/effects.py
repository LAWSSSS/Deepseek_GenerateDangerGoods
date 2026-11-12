"""Shared, opt-in controls for online review and deterministic image experiments."""
from .schema import EffectSettings

VERSION = 'online-effects-v1'


def parameters(settings):
    if settings is None:
        return {}
    settings = EffectSettings.model_validate(settings)
    return {'effects_enabled': True, 'effects_version': VERSION,
            'fog_extra': format(settings.fog_extra, 'g'),
            'pool_kernel': str(settings.pool_kernel),
            'surface_dust': format(settings.surface_dust, 'g'),
            'pool_weight': settings.pool_weight, 'effects_lighting': settings.lighting,
            'effects_ai_adjust': settings.ai_adjust}


def dust_profile(scale):
    settings = EffectSettings(surface_dust=scale)
    # Identical shader profile to the user's 0.5 trial. No second weathering layer.
    return {'version': 'surface-dust-strength-v2', 'dust_scale': settings.surface_dust,
            'paint_dust_scale': .5, 'single_deposit': True}


def lighting_for(scene, pose, environment, mode):
    from .local_light import estimate_local_light
    from .global_light import estimate_global_mean
    try:
        local, _ = estimate_local_light(scene, pose, environment, statistic='mean', margin_scale=1.75)
    except ValueError as error:
        if mode != 'global_mean' or str(error) != 'insufficient_local_neutral_light_evidence':
            raise
        local = {**environment, 'local_irradiance': {
            'bounds': environment['local_context']['bounds'], 'fill_ratio': .65,
            'direction': {'direction': None, 'confidence': 0., 'basis': str(error)}}}
    return estimate_global_mean(scene, local)[0] if mode == 'global_mean' else local


def optical_delta(clear, veiled, alpha, environment, air, target_px, params):
    from .local import foreground_dust
    settings = EffectSettings(fog_extra=float(params['fog_extra']),
                              pool_kernel=int(params['pool_kernel']),
                              pool_weight=params['pool_weight'],
                              surface_dust=float(params['surface_dust']))
    # Disabling atmosphere disables both the estimated and additional veil.
    fog = settings.fog_extra if params.get('atmosphere', 'auto') != 'off' else 0.
    weight = settings.pool_weight if settings.pool_kernel > 1 else 0.
    delta, measured = foreground_dust(clear, veiled, alpha, environment, air, target_px,
        pool_kernel=settings.pool_kernel, pool_weight=weight, fog_weight=fog,
        uniform=True, fog_profile='wide')
    measured.update(version=VERSION, requested=settings.model_dump(),
                    surface_dust_basis='separate_nonmetal_material_layer',
                    atmosphere_off=params.get('atmosphere') == 'off')
    return delta, measured


def render_identity(params, style):
    fields = ['candidate', 'material', 'light']
    if style in ('scene', 'native'):
        fields.append('dust')
    if params.get('effects_enabled'):
        fields.extend(('effects_version', 'surface_dust', 'effects_lighting'))
    return tuple(params[k] for k in fields)
