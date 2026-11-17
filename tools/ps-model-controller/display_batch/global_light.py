"""Explicit full-frame arithmetic-mean lighting diagnostic, never a local fallback."""
import hashlib

import numpy as np

from .photography import linear, LUMA


VERSION = 'full-frame-linear-mean-v1'


def estimate_global_mean(scene, local_environment):
    """Change mean energy/tint only; retain the control's direction and fill shape.

    All source-frame pixels participate equally, including outside-bin pixels and
    text. No depth, neutrality, percentile or local atmospheric correction is used.
    A scalar image mean is not measured incident radiance or a direction estimate.
    """
    rgb = np.asarray(scene['frame'], np.float32) / 255.
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError('invalid_full_frame_rgb')
    height, width = rgb.shape[:2]
    if (width, height) != tuple(scene['size']):
        raise ValueError('full_frame_size_mismatch')
    color = np.mean(linear(rgb), axis=(0, 1), dtype=np.float64)
    luminance = float(color @ LUMA)
    tint = np.clip(color / max(luminance, .01), .85, 1.15)
    level = float(np.clip(luminance / .18, .3, 1.8))
    mask = np.ones((height, width), bool)
    previous = local_environment['local_irradiance']
    evidence = {
        'version': VERSION, 'basis': 'all_source_frame_pixels_equal_weight',
        'statistic': 'arithmetic_mean', 'sample_count': width * height,
        'sample_hash': hashlib.sha256(np.packbits(mask)).hexdigest(),
        'bounds': [0, 0, width, height], 'original_bounds': previous['bounds'],
        'target_clear_rgb_linear': color.tolist(),
        'observed_mean_rgb_linear': color.tolist(),
        'observed_mean_rgb_srgb': np.mean(rgb, axis=(0, 1), dtype=np.float64).tolist(),
        'dehaze_compensation': 'none', 'filtering': 'none',
        'direction': previous['direction'], 'fill_ratio': previous['fill_ratio'],
        'direction_basis': 'frozen_local_control_direction_not_inferred_from_global_mean',
        'level': level, 'global_fallback': False, 'explicit_global_trial': True,
        'aggregation': 'equal_weight_per_frame_pixel_in_camera_linear_space',
        'assumptions': ['image_mean_is_not_incident_HDR', 'receiver_reflectance_0.3',
                        'whole_frame_includes_non_support_surfaces_and_text']}
    result = {**local_environment, 'world_color': (.20 * level * tint).tolist(),
              'light_color': (tint / tint.max()).tolist(), 'light_level': level,
              'local_irradiance': evidence,
              'lighting_basis': 'explicit_full_frame_arithmetic_mean_control'}
    # No spatial environment reconstruction is part of this scalar-mean trial.
    result.pop('reflection_probe', None)
    return result, mask
