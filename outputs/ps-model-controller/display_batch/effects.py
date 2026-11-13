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
            'effects_ai_adjust': settings.ai_adjust, 'fog_mode':settings.fog_mode,
            'fog_gain':format(settings.fog_gain,'g'), 'weathering':settings.weathering,
            'shadow_mode':settings.shadow_mode, 'render_detail':settings.render_detail,
            'rust_spots':format(settings.rust_spots,'g'), 'rust_pattern':settings.rust_pattern,
            'contrast_matching':settings.contrast_matching}


def dust_profile(scale, mode='legacy_matte'):
    settings = EffectSettings(surface_dust=scale)
    # Identical shader profile to the user's 0.5 trial. No second weathering layer.
    if mode=='preserve_source' and scale>.5:raise ValueError('surface_dust_limit')
    result={'version': 'surface-dust-strength-v2', 'dust_scale': settings.surface_dust,
            'paint_dust_scale': .5, 'single_deposit': True}
    if mode=='preserve_source':result.update(version='source-weathering-v1',mode=mode)
    return result


def lighting_for(scene, pose, environment, mode, *, preserve=False):
    from .local_light import estimate_local_light
    from .global_light import estimate_global_mean
    try:
        local, _ = estimate_local_light(scene, pose, environment, statistic='mean', margin_scale=1.75,
                                       reject_extremes=preserve)
    except ValueError as error:
        if (mode != 'global_mean' and not preserve) or str(error) != 'insufficient_local_neutral_light_evidence':
            raise
        local = {**environment, 'local_irradiance': {
            'bounds': environment['local_context']['bounds'], 'fill_ratio': .65,
            'level':float(environment.get('light_level',1.)),
            'target_clear_rgb_linear': environment['dust_color'],
            'statistic':'arithmetic_mean','confidence':'unknown',
            'direction': {'direction': None, 'confidence': 0., 'basis': str(error)}}}
    return estimate_global_mean(scene, local)[0] if mode == 'global_mean' else local


def optical_delta(clear, veiled, alpha, environment, air, target_px, params):
    from .local import foreground_dust
    if params.get('fog_mode')=='local_auto':
        import numpy as np
        from .local import pool_object_rgb
        from .photography import linear
        weight=float(params['pool_weight']) if int(params['pool_kernel'])>1 else 0.
        if params.get('workflow') == 'evidence_v1':
            # The camera PSF owns all automatic spatial filtering.
            weight = 0.
        optics=environment['camera_psf'];context=environment['local_context']
        if optics['edge_samples']<12 or context['sample_pixels']<20:weight=0.
        clear_linear=linear(clear)
        transmission=air['transmission_field']
        delta=linear(veiled)-clear_linear
        delta+=(pool_object_rgb(clear_linear,alpha,int(params['pool_kernel']))-clear_linear)*weight*transmission[...,None]
        delta*=np.float32(alpha>0)[...,None]
        fog=1-transmission[alpha>.5]
        return delta,{'version':'online-local-effects-v2','fog_mode':'local_auto',
            'fog_gain':float(params.get('fog_gain',1)), 'fog_base':float(np.median(fog)) if fog.size else 0.,
            'fog_extra_max':0.,'fog_total_median':float(np.median(fog)) if fog.size else 0.,
            'fog_total_max':float(fog.max()) if fog.size else 0.,'fog_color_linear':air['color_linear'],
            'pool_kernel':[int(params['pool_kernel'])]*2,'pool_weight':weight,'pool_uniform':True,
            **({'filtering_basis':'single_premultiplied_camera_psf'} if params.get('workflow')=='evidence_v1' else {}),
            'edge_samples':optics['edge_samples'],'sample_pixels':context['sample_pixels'],
            'requested':{k:params.get(k) for k in ('fog_mode','fog_extra','fog_gain','pool_kernel','surface_dust','weathering')},
            'additional_fog_applied':False,'atmosphere_off':params.get('atmosphere')=='off'}
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
        fields.extend(('weathering','shadow_mode'))
    fields.extend(('render_detail','rust_spots','rust_pattern'))
    return tuple(params.get(k) for k in fields)
