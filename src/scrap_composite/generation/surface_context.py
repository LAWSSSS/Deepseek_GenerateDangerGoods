"""Restrained camera-space correction against steel near the landing point."""
import cv2
import numpy as np


def nearby_weathered_steel(scene, candidate, reference_pixels):
    """Detect a bright support sample beside darker weathered scrap at the same depth."""
    x, y, right, bottom = map(int, candidate['bounds'])
    width, height = scene['size']
    margin = int(np.clip(candidate['target_px'] * .38, 24, 195))
    left, top = max(0, x - margin), max(0, y - margin)
    region = np.s_[top:min(height, bottom + margin), left:min(width, right + margin)]
    site = next(item for item in scene['supports'] if item['id'] == candidate['site_id'])
    valid = (scene['allowed'][region] & ~scene['protected'][region]
             & (np.abs(scene['depth_native'][region] - site['depth']) <= .085))
    valid[max(0, y-top):max(0, bottom-top), max(0, x-left):max(0, right-left)] = False
    if np.count_nonzero(valid) < 500 or len(reference_pixels) < 100:
        return {'applied': False, 'reason': 'insufficient_nearby_steel'}
    image = np.asarray(scene['frame'])[region]
    nearby = cv2.cvtColor(np.float32(image[valid])[:, None] / 255., cv2.COLOR_RGB2LAB)[:, 0]
    nearby = nearby[(nearby[:, 0] >= 17.) & (nearby[:, 0] <= 72.)]
    if len(nearby) < 500:
        return {'applied': False, 'reason': 'insufficient_nearby_midtone_steel'}
    previous = cv2.cvtColor(np.float32(reference_pixels)[:, None] / 255., cv2.COLOR_RGB2LAB)[:, 0]
    low, middle, high = np.percentile(nearby[:, 0], (15, 50, 90))
    support_median = float(np.median(previous[:, 0]))
    warm = nearby[(nearby[:, 1] >= 1.3) & (nearby[:, 2] >= 4.)]
    evidence = {'sample_pixels': len(nearby), 'weathered_pixels': len(warm), 'ring_luma_lab': float(middle),
                'support_luma_lab': support_median, 'depth_tolerance': .085}
    if support_median - middle < 9. or len(warm) < max(250, .025 * len(nearby)):
        return {**evidence, 'applied': False, 'reason': 'no_supported_bright_reference_mismatch'}
    rust_luma, a, b = np.median(warm, axis=0)
    target = float(np.clip(rust_luma + 3., middle, high + 3.))
    rgb = cv2.cvtColor(np.float32([[[38., a * 1.25, b * 1.45]]]), cv2.COLOR_LAB2RGB)[0, 0]
    pigment = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return {**evidence, 'applied': True, 'target_luma_lab': target, 'low_luma_lab': float(low),
            'highlight_luma_lab': max(target + 11., float(high) + 7.),
            'chroma_ab': [float(a), float(b)], 'pigment_linear': pigment.tolist()}


def match_surface_context(rgb, alpha, reference_pixels, context, *, deep_grime=False, caked_dirt=False,
                          weathered_reference=None):
    """Match low-frequency object colour and leave background pixels exact."""
    reference = np.asarray(reference_pixels, np.float32) / 255.
    core = alpha > .96
    summary = {'status': 'unknown', 'basis': context.get('basis'), 'sample_pixels': len(reference)}
    if (len(reference) < 100 or core.sum() < 100
            or context.get('basis') not in ('nearby_same_support_and_depth', 'nearby_same_depth_surface_estimate')):
        return rgb, summary
    local = cv2.cvtColor(reference[:, None, :], cv2.COLOR_RGB2LAB)[:, 0]
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    low, center, high = np.percentile(local[:, 0], (15, 50, 85))
    weathered = weathered_reference and weathered_reference.get('applied')
    if weathered_reference is not None:
        summary['nearby_weathered_steel'] = weathered_reference
    if weathered:
        low, center, high = (weathered_reference[field] for field in
                             ('low_luma_lab', 'target_luma_lab', 'highlight_luma_lab'))
    shade = lab[..., 0]
    obj_low, obj_center, obj_high = np.percentile(shade[core], (15, 50, 85))
    difference = center - obj_center
    slack, gain, max_shift = (2.8, .86, 10.) if deep_grime or caked_dirt else (4., .72, 8.)
    if caked_dirt:
        max_shift = 14.
    if weathered:
        slack, gain, max_shift = 1.6, .85, 12.
    shift = float(np.clip(np.sign(difference) * max(0., abs(difference) - slack) * gain, -max_shift, max_shift))
    allowed_span = max(14., (high - low) * 1.15)
    contrast = float(np.clip(allowed_span / max(obj_high - obj_low, 1.), .77, 1.))
    neutral = np.float32(weathered_reference['chroma_ab']) if weathered else np.median(local[:, 1:], axis=0)
    summary.update(reference_luma_lab=float(center), reference_chroma_ab=neutral.tolist())
    offset = lab[..., 1:] - neutral
    radius = np.linalg.norm(offset, axis=2)
    limit = max(4., float(np.percentile(np.linalg.norm(local[:, 1:] - neutral, axis=1), 85)) + 2.)
    trim = np.clip((radius - limit) / np.maximum(radius, 1.), 0, .72) * .56

    # A low-frequency field adjusts illumination while leaving the high spatial
    # frequencies from engraving, rust pits, and scratches in place.
    field = cv2.GaussianBlur(shade * alpha, (0, 0), 7.) / np.maximum(cv2.GaussianBlur(alpha, (0, 0), 7.), 1e-4)
    authority = np.clip((alpha - .20) / .72, 0, 1)
    authority *= 1. - .55 * np.clip((np.abs(shade - field) - 4.) / 14., 0, 1)
    authority *= 1. - .48 * np.clip((shade - max(76., obj_high)) / 16., 0, 1)
    shoulder = np.clip((field + shift - high) * .60, 0, 7.) if weathered else 0.
    mask = (authority > 0) & ((trim > 0) | (shift != 0) | (contrast < 1) | (shoulder > 0) | bool(weathered))
    if not mask.any():
        return rgb, {**summary, 'status': 'within_context'}
    lab[..., 0] = np.clip(shade + authority * (shift + (contrast - 1.) * (field - obj_center) - shoulder), 0, 100)
    lab[..., 1:] -= (authority * trim)[..., None] * offset
    if weathered:
        color_shift = np.clip((neutral - np.median(lab[..., 1:][core], axis=0)) * .65, -5., 5.)
        lab[..., 1:] += authority[..., None] * color_shift
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    result = rgb.copy()
    result[mask] = corrected[mask]
    return result, {**summary, 'status': 'applied', 'luma_shift_lab': shift, 'low_frequency_contrast_gain': contrast,
                    'luma_gap_before_lab': float(abs(difference)),
                    'luma_gap_after_lab': float(abs(center - np.median(lab[..., 0][core]))),
                    'chroma_adjusted_pixels': int(np.count_nonzero(mask & (trim > .03)))}
