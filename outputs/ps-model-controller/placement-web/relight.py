"""Fit soft scene lighting and the camera response of a placement preview."""
import cv2
import numpy as np
from PIL import Image

from photometry import statistics


def aligned_alpha(alpha, size, x, y):
    frame = Image.new('L', size)
    frame.paste(alpha, (x, y))
    return np.asarray(frame)


def local_samples(frame, support, excluded, alpha):
    """Sample outside the projected silhouette, at original scene resolution."""
    silhouette = alpha >= 128
    if not silhouette.any():
        return np.zeros_like(support, dtype=bool)
    _, _, w, h = cv2.boundingRect(silhouette.astype(np.uint8))
    major = max(w, h)
    distance = cv2.distanceTransform((~silhouette).astype(np.uint8), cv2.DIST_L2, 5)
    sample = (support & ~excluded & (distance >= max(2, .035 * major))
              & (distance <= max(18, .55 * major)))
    gray = np.asarray(frame.convert('L'))
    if np.count_nonzero(sample) >= 100:
        lower, upper = np.percentile(gray[sample], [10, 96])
        sample &= (gray >= max(5, lower)) & (gray <= min(249, upper))
    return sample


def estimate_render_environment(frame, samples, lighting=None):
    """Derive ambient colour from the frame, using independent rods for direction."""
    measured = statistics(np.asarray(frame), samples)
    scene = measured[0] if measured else None
    neutral_ab = (scene or {}).get('neutral_ab') or [0.0, 0.0]
    # A small cast on broad light sources preserves albedo and metal response.
    lab = np.array([[[55.0, *np.clip(neutral_ab, -7, 7)]]], dtype=np.float32)
    tint = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]
    tint = np.clip(1 + .45 * (tint / tint.mean() - 1), .91, 1.09)
    luminance = scene['luma_p20_p50_p80'][1] if scene else 110
    ambient = float(np.clip(.36 + (luminance - 110) * .001, .27, .43))
    direction = None
    rods = lighting or {}
    if rods.get('status') == 'estimated':
        candidate = np.asarray(rods.get('direction_screen'), dtype=float)
        if (candidate.shape == (2,) and np.all(np.isfinite(candidate))
                and np.linalg.norm(candidate) > 1e-4):
            candidate /= np.linalg.norm(candidate)
            if candidate[1] <= .5:
                direction = [round(float(value), 4) for value in candidate]
    return {
        'status': 'estimated' if direction is not None else 'diffuse_estimate' if scene else 'insufficient_samples',
        'confidence': 'low' if direction is not None else 'unknown',
        'world_color': [round(float(value), 5) for value in ambient * tint],
        'light_color': [round(float(value), 5) for value in tint / tint.max()],
        'world_strength': .76,
        'key_direction': direction,
        'key_strength': .48 if direction is not None else .18,
        'light_direction': 'low_confidence_screen_direction' if direction is not None else 'unknown',
        'scene_luma_p20_p50_p80': scene['luma_p20_p50_p80'] if scene else None,
        'scene_neutral_ab': neutral_ab if scene else None,
        'sample_pixels': scene['samples'] if scene else 0,
    }


def ambient_match(frame, support, foreground, obj, x, y, *, anchored=False, seed=0, samples=None):
    """Apply bounded lightness, white balance and sensor response, retaining alpha."""
    alpha = np.asarray(obj.getchannel('A'))
    if samples is None:
        samples = local_samples(frame, support, foreground, aligned_alpha(obj.getchannel('A'), frame.size, x, y))
    else:
        # Reuse exactly the scene evidence that lit the 3D render. Always
        # exclude the actual rendered object as well as the client silhouette.
        samples = samples & support & ~foreground & (aligned_alpha(obj.getchannel('A'), frame.size, x, y) == 0)
    rgb = np.asarray(obj.convert('RGB'))
    core = alpha >= 190
    measured = statistics(np.asarray(frame), samples)
    original = statistics(rgb, core)
    if measured is None or original is None:
        return obj.copy(), {'status': 'insufficient_samples', 'light_direction': 'unknown',
                            'reason': '局部环境或物体像素不足'}
    scene, _ = measured
    before, _ = original

    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    scene_gray = np.float32(scene['luma_p20_p50_p80']).reshape(1, 3, 1) / 255
    scene_lightness = cv2.cvtColor(np.repeat(scene_gray, 3, axis=2), cv2.COLOR_RGB2LAB)[0, :, 0]
    low, middle, high = np.percentile(lab[:, :, 0][core], [20, 50, 80])
    scene_range = max(6, scene_lightness[2] - scene_lightness[0])
    # The surface's median includes bright bare steel. Keep coloured and
    # oxidized objects slightly below it while retaining their form contrast.
    target = .8 * scene_lightness[1] + .2 * scene_lightness[0]
    exposure = float(np.clip((target - middle) * (.58 if anchored else .50), -16, 13))
    contrast = float(np.clip(1 + .14 * np.log(scene_range / max(6, high - low)), .86, 1.12))
    lightness = middle + exposure + (lab[:, :, 0] - middle) * contrast
    knee = float(np.clip(scene_lightness[2] + 3.5, 47, 91))
    shoulder = float(np.clip(.26 * scene_range + 7, 8, 16))
    highlight = knee + shoulder * (-np.expm1(-np.maximum(lightness - knee, 0) / shoulder))
    lab[:, :, 0] = np.clip(np.minimum(lightness, highlight), 0, 100)

    source_cast = np.asarray(before['neutral_ab'] or [0.0, 0.0])
    scene_cast = np.asarray(scene['neutral_ab'] or [0.0, 0.0])
    cast = np.clip((scene_cast - source_cast) * .55, -3.0, 3.0)
    # Bound the change to paint and label chroma independently of exposure.
    chroma = float(np.clip(1 - .009 * max(0, before['chroma_c'] - scene['chroma_c']), .68, 1))
    lab[:, :, 1:] = source_cast + (lab[:, :, 1:] - source_cast) * chroma + cast
    matched = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255

    # Match detail in the final pixel grid with normalized, alpha-weighted
    # filtering. Colour outside the silhouette cannot bleed into openings.
    detail_ratio = ((before['detail'] or 0) + 1) / ((scene['detail'] or 0) + 1)
    scene_noise = scene['noise'] or 0
    # Do not blur already smoother objects because the scene is noisy. This
    # used to suppress model surface detail even when it was below the scene.
    blur = float(np.clip(.42 * np.sqrt(max(0, np.log(detail_ratio))), 0, .85)) if detail_ratio > 1.25 else 0.0
    sharpen = float(np.clip(.20 * np.log(1 / detail_ratio), 0, .28)) if detail_ratio < .8 else 0.0
    if blur > 0 or sharpen > 0:
        coverage = alpha.astype(np.float32) / 255
        sigma = blur if blur > 0 else .65
        weights = cv2.GaussianBlur(coverage, (0, 0), sigma)
        filtered = cv2.GaussianBlur(matched * coverage[:, :, None], (0, 0), sigma)
        filtered = np.where(weights[:, :, None] > 1e-5,
                            filtered / np.maximum(weights[:, :, None], 1e-5), matched)
        # Restrained sharpening uses existing RGB details; alpha and openings
        # remain unchanged and the correction cannot exceed three RGB levels.
        matched = filtered if blur > 0 else matched + np.clip((matched-filtered)*sharpen, -3, 3)

    # Reproduce the measured noise level without copying a background pattern.
    response_stats = statistics(np.uint8(np.rint(np.clip(matched, 0, 255))), core)
    object_noise = (response_stats[0]['noise'] if response_stats else before['noise']) or 0
    grain = float(np.clip(np.sqrt(max(0, scene_noise ** 2 - object_noise ** 2)) * .82, 0, 2.6))
    if grain >= .35:
        noise = np.random.default_rng(seed).normal(0, grain, alpha.shape).astype(np.float32)
        matched += np.clip(noise, -6, 6)[:, :, None]
    else:
        grain = 0.0
    matched = np.uint8(np.rint(np.clip(matched, 0, 255)))
    result = Image.fromarray(matched).convert('RGBA')
    result.putalpha(obj.getchannel('A'))
    after = statistics(matched, core)
    return result, {
        'status': 'measured', 'method': 'bounded_camera_response', 'color_space': 'sRGB',
        'scene': scene, 'object_before': before, 'object_after': after[0] if after else None,
        'support_samples': scene['samples'], 'light_direction': 'unknown',
        'lightness_offset': round(exposure, 3), 'form_contrast_gain': round(contrast, 3),
        'highlight_knee_lightness': round(knee, 3),
        'chroma_gain': round(chroma, 3), 'neutral_cast_ab': cast.round(3).tolist(),
        'detail_blur_sigma_px': round(blur, 3), 'sensor_grain_sigma': round(grain, 3),
        'detail_sharpen_amount': round(sharpen, 3),
    }


def lighting_samples(frame, support, excluded, alpha):
    """Choose one auditable scene sample for both rendering and final tone."""
    if support is not None:
        samples = local_samples(frame, support, excluded, alpha)
        if statistics(np.asarray(frame), samples) is not None:
            return samples, 'projected_support_only'
    samples = local_samples(frame, np.ones_like(excluded), excluded, alpha)
    return samples, 'surrounding_scene_unverified'
