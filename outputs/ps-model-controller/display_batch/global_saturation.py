"""Diagnostic full-background mean saturation cap, with Lab lightness preserved.

The original, complete photograph supplies HSV S. Existing local colour limits
run first. This extra cap can only reduce colour; it never normalizes brightness,
changes a material shader, or modifies geometric coverage.
"""
import cv2
import numpy as np

VERSION = 'global-mean-saturation-v2'
TOLERANCE = .006  # final 8-bit RGB / HSV quantization


def hsv(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)


def background_statistics(rgb):
    image = np.asarray(rgb, np.float32) / 255
    values = hsv(image)[..., 1]
    return {'version': VERSION, 'mean_s': float(values.mean(dtype=np.float64)),
            'p90_s': float(np.percentile(values, 90)), 'pixels': int(values.size),
            'basis': 'all_pixels_of_original_background_including_outside_hopper',
            'space': 'HSV_S_0_to_1', 'quantization_tolerance': TOLERANCE}


def colour_mask(rgb, alpha):
    lab = cv2.cvtColor(np.asarray(rgb, np.float32), cv2.COLOR_RGB2LAB)
    c = np.linalg.norm(lab[..., 1:], axis=2)
    # Saturated dark paint can have low Lab chroma: use saturation itself,
    # not a fixed C threshold, to decide whether it belongs in this experiment.
    neutral_label = (lab[..., 0] > 65) & (c < 4)
    return (alpha > .95) & (hsv(rgb)[..., 1] > .04) & (lab[..., 0] > 2) & (lab[..., 0] < 98) & ~neutral_label


def measure(rgb, mask, stats):
    values = hsv(rgb)[..., 1][mask]
    if len(values) < 24:
        return {'status': 'unknown', 'reason': 'too_few_coloured_opaque_pixels',
                'pixels': int(len(values)), 'target_mean_s': stats['mean_s']}
    mean = float(values.mean(dtype=np.float64))
    return {'status': 'pass' if mean <= stats['mean_s'] + TOLERANCE else 'fail',
            'pixels': int(len(values)), 'mean_s': mean,
            'p90_s': float(np.percentile(values, 90)),
            'target_mean_s': stats['mean_s'], 'tolerance': TOLERANCE}


def apply(rgb, alpha, stats):
    original = np.asarray(rgb, np.float32)
    mask = colour_mask(original, alpha)
    before = measure(original, mask, stats)
    report = {'version': VERSION, 'background': stats, 'before': before,
              'gain': 1., 'lightness_preserved': True, 'only_reduces': True}
    if before['status'] != 'fail':
        report.update(after=before, applied=False)
        return original.copy(), report, mask
    lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    samples = lab[mask].reshape(-1, 1, 3)
    target = max(0., stats['mean_s'] - .002)
    low, high = 0., 1.
    # Lab chroma -> HSV saturation is nonlinear. Solve the mean, not an RGB
    # multiplier that would also darken the object. No repeated rendering/API.
    for _ in range(15):
        gain = (low + high) / 2
        trial = samples.copy(); trial[..., 1:] *= gain
        converted = cv2.cvtColor(trial, cv2.COLOR_LAB2RGB)
        if float(hsv(converted)[..., 1].mean(dtype=np.float64)) <= target:
            low = gain
        else:
            high = gain
    chroma = np.linalg.norm(lab[..., 1:], axis=2)
    transition = np.clip((hsv(original)[..., 1] - .02) / .02, 0, 1)
    transition[(lab[..., 0] > 65) & (chroma < 4)] = 0
    transition = transition * transition * (3 - 2 * transition)
    gain_map = 1 - (1 - low) * transition
    lab[..., 1:] *= gain_map[..., None]
    converted = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    changed = (alpha > 0) & (transition > 0)
    result = original.copy(); result[changed] = converted[changed]
    quantized = np.rint(result * 255) / 255
    after = measure(quantized, mask, stats)
    report.update(gain=low, after=after, applied=True,
                  mean_reduction=before['mean_s'] - after['mean_s'])
    return result, report, mask
