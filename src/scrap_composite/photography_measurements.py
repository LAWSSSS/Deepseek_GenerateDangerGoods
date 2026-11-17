"""Photographic measurements in the exported sRGB image and its local samples."""
import json
import math

import cv2
import numpy as np

from .photography_protocol import PhotoMeasurements

_REFERENCE = {
    'nearby_same_support_and_depth': 'same_support',
    'nearby_same_depth_surface_estimate': 'same_depth',
    'original_surface_under_object_local_only': 'under_object',
}


def _number(value, scale=1.):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return round(value * scale, 4)
    return None


def _pixels(pixels):
    """Display luma and Lab chroma describe pixels, with albedo and illumination mixed."""
    if pixels is None or len(pixels) < 32:
        return {}
    rgb = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
    y = rgb @ np.array([.2126, .7152, .0722], np.float32)
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3) / 255, cv2.COLOR_RGB2LAB)[:, 0, :]
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    neutral = (chroma <= 8) & (y > 12) & (y < 243)
    values = {'y_255': tuple(float(v) for v in np.percentile(y, [10, 50, 90]).round(2)),
              'chroma_c50_c90': tuple(float(v) for v in np.percentile(chroma, [50, 90]).round(2)),
              'near_black_white_fraction': (round(float((y <= 2).mean()), 5),
                                            round(float((rgb.max(axis=1) >= 253).mean()), 5))}
    if neutral.sum() >= 32:
        values['neutral_ab'] = tuple(float(v) for v in np.median(lab[neutral, 1:], axis=0).round(2))
    return values


def measurements(photo, *, object_pixels=None, reference_pixels=None, visible_shadow_pixels=None,
                 lighting=None, specular_share=None):
    """Attach known quantities only. Historical records may lack exported pixel statistics."""
    recorded = photo.get('review_measurements')
    if recorded is not None and object_pixels is None and reference_pixels is None:
        return PhotoMeasurements.model_validate_json(json.dumps(recorded))
    local = photo.get('local_context', {})
    air = photo.get('atmosphere', {})
    transmission = air.get('transmission_range')
    direction = (lighting or {}).get('direction')
    share = tuple(float(v) for v in np.percentile(np.clip(specular_share, 0, 1), [50, 90]).round(4)) if (
        specular_share is not None and len(specular_share) >= 32) else None
    material_chroma = {kind: (_number(row['c50']), _number(row['c90']))
                       for kind, row in photo.get('final_chroma', {}).get('groups', {}).items()
                       if kind in ('paint', 'metal') and 'c50' in row and 'c90' in row}
    statistics = {}
    for name, pixels in (('local', reference_pixels), ('object', object_pixels)):
        statistics.update({f'{name}_{key}': value for key, value in _pixels(pixels).items()
                           if name == 'object' or key != 'near_black_white_fraction'})
    return PhotoMeasurements(
        samples=len(reference_pixels) if reference_pixels is not None else int(local.get('sample_pixels', 0)),
        reference=_REFERENCE.get(local.get('basis'), 'unknown') if local.get('confidence') == 'estimated' else 'unknown',
        local_noise_255=_number(photo.get('sensor', {}).get('noise_sigma'), 255),
        added_noise_255=_number(photo.get('added_noise_sigma'), 255),
        camera_sigma_px=_number(photo.get('measured_camera_sigma', photo.get('camera_psf', {}).get('sigma_px'))),
        applied_sigma_px=_number(photo.get('blur_sigma')),
        fog=_number(photo.get('haze')),
        transmission_p05_p95=(_number(transmission[0]), _number(transmission[-1])) if transmission else None,
        atmosphere_confidence='estimated' if air.get('confidence') == 'estimated' else 'unknown',
        visible_shadow_pixels=visible_shadow_pixels if visible_shadow_pixels is not None else photo.get('visible_shadow_pixels'),
        material_chroma_c50_c90=material_chroma,
        lighting_direction_xy=tuple(direction) if direction else None,
        lighting_confidence=_number((lighting or {}).get('confidence')),
        specular_share_p50_p90=share,
        **statistics,
    )
