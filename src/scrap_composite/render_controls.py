"""Registered absolute render controls; no image-specific or provider rules."""
import math

VERSION = 'measured-render-controls-v1'
MATERIAL_CONTROLS = {
    'oxide_gain': ('0.5', '1', '1.5'),
    'scratch_gain': ('0.5', '1', '1.5'),
    'roughness_shift': ('-0.08', '0', '0.08'),
    'relief_gain': ('0.5', '1', '1.5'),
}
LIGHT_CONTROLS = {
    'light_width_scale': ('0.75', '1', '1.25'),
    'fill_gain': ('0.75', '1', '1.25'),
}
RENDER_CONTROLS = {**MATERIAL_CONTROLS, **LIGHT_CONTROLS}
DEFAULTS = {key: ('0' if key == 'roughness_shift' else '1') for key in RENDER_CONTROLS}


def validated(values, *, require_all=False):
    """Return canonical values; bools/NaNs and undeclared controls are rejected."""
    if set(values) - set(RENDER_CONTROLS):
        raise ValueError('unregistered render control')
    result = dict(DEFAULTS) if require_all else {}
    for key, value in values.items():
        if isinstance(value, bool):
            raise ValueError('render control must be an absolute number')
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('non-finite render control')
        choices = [v for v in RENDER_CONTROLS[key] if math.isclose(number, float(v), abs_tol=1e-9)]
        if not choices:
            raise ValueError('unregistered render control value: ' + key)
        result[key] = choices[0]
    return result


def material_controls(params):
    return {key: float(value) for key, value in validated(
        {k: params[k] for k in MATERIAL_CONTROLS if k in params}).items()}


def light_controls(params):
    return {key: float(value) for key, value in validated(
        {k: params[k] for k in LIGHT_CONTROLS if k in params}).items()}
