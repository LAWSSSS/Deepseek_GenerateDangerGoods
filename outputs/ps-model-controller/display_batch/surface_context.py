"""Restrained camera-space correction against steel near the landing point."""
import cv2
import numpy as np


def match_surface_context(rgb, alpha, reference_pixels, context, *, deep_grime=False):
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
    shade = lab[..., 0]
    obj_low, obj_center, obj_high = np.percentile(shade[core], (15, 50, 85))
    difference = center - obj_center
    slack, gain, max_shift = (2.8, .86, 10.) if deep_grime else (4., .72, 8.)
    shift = float(np.clip(np.sign(difference) * max(0., abs(difference) - slack) * gain, -max_shift, max_shift))
    allowed_span = max(14., (high - low) * 1.15)
    contrast = float(np.clip(allowed_span / max(obj_high - obj_low, 1.), .77, 1.))
    neutral = np.median(local[:, 1:], axis=0)
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
    mask = (authority > 0) & ((trim > 0) | (shift != 0) | (contrast < 1))
    if not mask.any():
        return rgb, {**summary, 'status': 'within_context'}
    lab[..., 0] = np.clip(shade + authority * (shift + (contrast - 1.) * (field - obj_center)), 0, 100)
    lab[..., 1:] -= (authority * trim)[..., None] * offset
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    result = rgb.copy()
    result[mask] = corrected[mask]
    return result, {**summary, 'status': 'applied', 'luma_shift_lab': shift, 'low_frequency_contrast_gain': contrast,
                    'luma_gap_before_lab': float(abs(difference)),
                    'luma_gap_after_lab': float(abs(center - np.median(lab[..., 0][core]))),
                    'chroma_adjusted_pixels': int(np.count_nonzero(mask & (trim > .03)))}
