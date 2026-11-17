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


def region_statistics(rgb, mask):
    """Pixel statistics with declared units, not physical material/lighting values."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask, bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError('measurement_canvas_mismatch')
    count = int(mask.sum())
    if count < 32:
        return {'status': 'unknown', 'samples': count, 'reason': 'fewer_than_32_valid_pixels'}
    sample = rgb[mask].astype(np.float32)
    result = _pixels(sample)
    decoded = sample / 255.
    linear = np.where(decoded <= .04045, decoded / 12.92, ((decoded + .055) / 1.055) ** 2.4)
    luma = linear @ np.array([.2126, .7152, .0722], np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    magnitude = np.hypot(gx, gy)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gradient = magnitude[interior]
    threshold = min(4., float(np.percentile(gradient, 35))) if len(gradient) else 0.
    flat = interior & (magnitude <= threshold)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), .8)
    flat_values = residual[flat]
    noise = float(np.median(np.abs(flat_values - np.median(flat_values))) / .67449) if len(flat_values) >= 32 else None
    result.update(status='measured', samples=count,
        mean_rgb_255=[round(float(v), 3) for v in sample.mean(axis=0)],
        linear_luminance_p10_p50_p90=[round(float(v), 5) for v in np.percentile(luma, [10, 50, 90])],
        local_contrast_255=round(float(np.percentile(sample @ [.2126,.7152,.0722], 90) - np.percentile(sample @ [.2126,.7152,.0722], 10)), 3),
        edge_gradient_p50_p90_255=[round(float(v), 3) for v in np.percentile(gradient, [50, 90])] if len(gradient) else None,
        flat_region_samples=int(flat.sum()), flat_residual_noise_estimate_255=round(noise, 4) if noise is not None else None,
        basis='actual_exported_sRGB_pixels; linear values use inverse sRGB transfer',
        limitations=['albedo_and_illumination_mixed', 'edge_gradient_is_not_camera_PSF', 'noise_estimate_may_include_fine_texture'])
    return result


def measure_regions(rgb, regions):
    """Normalized host/model ROI IDs; no paths, scripts or unbounded pixel output."""
    result = []
    height, width = rgb.shape[:2]
    for region in regions:
        if set(region) != {'id', 'bounds'} or not isinstance(region['id'], str) or not 1 <= len(region['id']) <= 80:
            raise ValueError('invalid_measurement_region')
        box = region['bounds']
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1 for x in box)):
            raise ValueError('invalid_measurement_bounds')
        l, t, r, b = box
        if l >= r or t >= b:
            raise ValueError('empty_measurement_bounds')
        left, top, right, bottom = int(l*width), int(t*height), math.ceil(r*width), math.ceil(b*height)
        crop = rgb[top:bottom, left:right]
        result.append({'id': region['id'], 'bounds': list(box), 'pixel_bounds': [left,top,right,bottom],
                       **region_statistics(crop, np.ones(crop.shape[:2], bool))})
    return result


def measure_export_regions(rgb, alpha, *, blocked=None, shadow_alpha=None):
    """Bounded image neighborhood is measured, never called a verified same-depth reference."""
    alpha = np.asarray(alpha)
    if alpha.shape != rgb.shape[:2]:
        raise ValueError('measurement_alpha_canvas_mismatch')
    forbidden = np.zeros(alpha.shape, bool) if blocked is None else np.asarray(blocked, bool)
    object_mask = (alpha >= 242) & ~forbidden
    ys, xs = np.nonzero(alpha > 0)
    if not len(xs):
        return {'status': 'unknown', 'reason': 'no_visible_object'}
    h, w = alpha.shape
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
    pad = max(4, int(max(right-left, bottom-top) * .65))
    bounds = [max(0,left-pad), max(0,top-pad), min(w,right+pad), min(h,bottom+pad)]
    nearby = np.zeros(alpha.shape, bool)
    nearby[bounds[1]:bounds[3], bounds[0]:bounds[2]] = True
    nearby[top:bottom,left:right] = False  # Hole pixels do not establish the same support/depth.
    nearby &= (cv2.dilate((alpha > 0).astype(np.uint8), np.ones((7,7), np.uint8)) == 0) & ~forbidden
    if shadow_alpha is not None:
        nearby &= np.asarray(shadow_alpha) == 0
    y = rgb.astype(np.float32) @ [.2126,.7152,.0722]
    nearby &= (y > 12) & (y < 235) & (rgb.max(axis=2) < 250)
    return {'version': 'photoshop-region-measurements-v1', 'object': region_statistics(rgb, object_mask),
            'neighborhood': region_statistics(rgb, nearby), 'bounds': bounds,
            'reference': 'bounded_image_neighborhood', 'reference_confidence': 'unknown_same_material_and_depth',
            'excluded': ['object_and_boundary', 'foreground', 'protection', 'known_shadow', 'deep_dark', 'clipped_highlight'],
            'physical_illumination': 'unknown', 'material_roughness': 'unknown'}


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
