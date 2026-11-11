"""Local, robust illumination evidence and a shared diffuse-receiver calibration.

A photograph is not an HDR measurement. All radiance, depth and direction outputs
remain estimates; remote scene pixels never enter this fit.
"""
import hashlib
import math

import numpy as np


VERSION = 'local-tile-irradiance-v1'
LUMA = np.float32([.2126, .7152, .0722])


def fit_direction(centers, luminance):
    """Fit tile log-brightness, requiring spatial diversity and split-fit agreement."""
    xy = np.asarray(centers, np.float64)
    values = np.log(np.maximum(luminance, .005))
    unknown = {'direction': None, 'confidence': 0., 'basis': 'insufficient_consistent_local_cues'}
    if len(xy) < 8 or np.linalg.cond(np.cov(xy.T)) > 8:
        return unknown
    design = np.column_stack((np.ones(len(xy)), xy))
    weights = np.ones(len(xy))
    for _ in range(5):
        coefficients = np.linalg.lstsq(design * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None)[0]
        residual = values - design @ coefficients
        scale = max(.035, 1.4826 * float(np.median(abs(residual - np.median(residual)))))
        weights = np.minimum(1., 1.5 * scale / np.maximum(abs(residual), .001))
    vector = coefficients[1:]
    strength = float(np.linalg.norm(vector))
    error = float(np.median(abs(residual)))
    split = []
    for phase in (0, 1):
        matrix = design[phase::2]
        if np.linalg.cond(matrix) > 12:
            return unknown
        split.append(np.linalg.lstsq(matrix, values[phase::2], rcond=None)[0][1:])
    agreement = float(split[0] @ split[1] / max(1e-6, np.linalg.norm(split[0]) * np.linalg.norm(split[1])))
    if strength < .10 or error > .18 or error > strength * .6 or agreement < .85:
        return {**unknown, 'residual_log': error, 'split_agreement': agreement}
    return {'direction': (vector / strength).tolist(), 'confidence': float(min(.65, .65 * agreement * (1-error/.3))),
            'basis': 'consistent_local_tile_gradient_not_measured_light', 'residual_log': error,
            'split_agreement': agreement, 'gradient_log': strength}


def estimate_local_light(scene, candidate, environment):
    import cv2
    from .photography import linear
    x, y, r, b = environment['local_context']['bounds']
    rgb = np.float32(np.asarray(scene['frame'])[y:b, x:r]) / 255
    gray = rgb @ LUMA
    site = next(s for s in scene['supports'] if s['id'] == candidate['site_id'])
    depth = scene['depth_native'][y:b, x:r]
    support = np.logical_or.reduce([s['mask'][y:b, x:r] for s in scene['supports']
                                   if s['id'] in candidate.get('support_ids', [site['id']])])
    valid = scene['allowed'][y:b, x:r] & ~scene['protected'][y:b, x:r]
    valid &= abs(depth - site['depth']) <= .085
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    valid &= (rgb.max(axis=2) < .93) & (gray > .07) & (spread < .17)
    left, top, right, bottom = candidate['bounds']
    ring = np.ones_like(valid)
    ring[max(0, top-y):min(b-y, bottom-y), max(0, left-x):min(r-x, right-x)] = False
    selected = valid & ring & support
    basis = 'near_same_depth_support_neutral_tiles'
    if selected.sum() < 100:
        selected = valid & ring
        basis = 'near_same_depth_neutral_tiles'
    if selected.sum() < 60:
        selected = valid & support
        basis = 'local_under_object_neutral_support_estimate'
    if selected.sum() < 40:
        raise ValueError('insufficient_local_neutral_light_evidence')
    lo, hi = np.percentile(gray[selected], [12, 88])
    selected &= (gray >= lo) & (gray <= hi)
    # Erode metal boundaries so adjacent gaps and bright edges cannot bias a tile.
    eroded = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if eroded.sum() >= 40:
        selected = eroded
    observed = linear(rgb)
    air = environment.get('airborne_dust', {})
    amount = min(.28, max(0., air.get('amount', 0.)))
    air_color = np.asarray(air.get('color_linear', environment['dust_color']), np.float32)
    clear = np.clip((observed - amount * air_color) / max(.72, 1-amount), observed * .65, observed * 1.15)
    h, w = selected.shape
    tiles, centers, colors, counts = [], [], [], []
    for iy in range(6):
        for ix in range(6):
            l, t, rr, bb = ix*w//6, iy*h//6, (ix+1)*w//6, (iy+1)*h//6
            mask = selected[t:bb, l:rr]
            if mask.sum() < 12:
                continue
            color = np.median(clear[t:bb, l:rr][mask], axis=0)
            center = [(l+rr)/w-1, (t+bb)/h-1]
            colors.append(color); centers.append(center); counts.append(int(mask.sum()))
            tiles.append({'bounds': [x+l,y+t,x+rr,y+bb], 'pixels': int(mask.sum()), 'rgb_linear': color.tolist()})
    if len(tiles) < 3:
        colors = [np.median(clear[selected], axis=0)]
        centers = [[0., 0.]]
    # Equal tile voting prevents one large sheet from dominating many smaller pieces.
    color = np.median(colors, axis=0)
    luminance = float(color @ LUMA)
    tint = np.clip(color / max(luminance, .01), .85, 1.15)
    direction = fit_direction(centers, np.asarray(colors) @ LUMA)
    q25, q75 = np.percentile(clear[selected] @ LUMA, [25, 75])
    fill = float(np.clip(q25 / max(q75, .015), .38, .85))
    level = float(np.clip(luminance / .18, .3, 1.8))
    mask = np.zeros(scene['size'][::-1], bool)
    mask[y:b, x:r] = selected
    evidence = {'version': VERSION, 'bounds': [x,y,r,b], 'basis': basis, 'sample_count': int(selected.sum()),
                'sample_hash': hashlib.sha256(np.packbits(mask)).hexdigest(), 'tiles': tiles,
                'target_clear_rgb_linear': color.tolist(), 'direction': direction,
                'level': level, 'fill_ratio': fill, 'depth_tolerance': .085, 'global_fallback': False,
                'assumptions': ['same_depth_neutral_surface', 'receiver_reflectance_0.3', 'unknown_HDR_and_surface_normals']}
    result = {**environment, 'world_color': (.20 * level * tint).tolist(), 'light_color': (tint/tint.max()).tolist(),
              'key_direction': direction['direction'], 'light_level': level, 'local_irradiance': evidence,
              'lighting_basis': 'bounded_local_tiles_and_diffuse_receiver_calibration'}
    # Keep air, surface dust and camera characteristics independent of this estimate.
    return result, mask


def light_rig(environment, camera, gain=1., shape=(128, 256)):
    from .geometry import illumination_axis
    evidence = environment['local_irradiance']
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    sky = .55 + .95 * (yy / height) ** 2
    direction = evidence['direction']['direction']
    confidence = evidence['direction']['confidence']
    level, fill = evidence['level'], evidence['fill_ratio']
    if direction:
        axis = np.asarray(illumination_axis(camera, direction), float)
        offset = axis * 4.6
        azimuth = math.atan2(axis[1], axis[0])
        # Broad first-order lobe; never a sharp synthetic reflection strip.
        sky *= 1 + .30 * confidence * np.cos(xx/width*2*math.pi-azimuth)
    else:
        offset = (0., 0., 4.6)
    key = 155 * level * gain
    # Reflected fill must follow the camera-relative observed hemisphere rather
    # than a fixed world +Y direction, which can darken the visible side arbitrarily.
    toward = np.asarray(camera.get('eye',[0,-6,6]),float)-np.asarray(camera.get('target',[0,0,0]),float)
    toward[2]=max(.5,abs(toward[2]))
    toward=tuple(toward/max(np.linalg.norm(toward),1e-6)*4.)
    lights = [(offset, key, 7.5 - 2 * confidence), (toward, key * fill * 1.2, 8.)]
    return np.float32(sky), lights


def receiver_calibration(render_dir, meta, environment):
    """Calibrate one rendered neutral receiver, not each object's brightness histogram."""
    import cv2
    from pathlib import Path
    from .photography import display
    path = Path(meta['part_paths']['without_shadow']) / 'without-shadow-linear0001.exr'
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError('missing_local_light_receiver_reference')
    rgb = raw[..., :3][..., ::-1]
    usable = np.isfinite(rgb).all(axis=2) & ((rgb @ LUMA) > .015)
    if usable.sum() < 40:
        raise ValueError('insufficient_local_light_receiver_reference')
    sample = np.median(rgb[usable], axis=0)[None, None, :]
    target = np.asarray(environment['local_irradiance']['target_clear_rgb_linear'], np.float32)
    from .compose import srgb
    target_luma = float(srgb(target) @ LUMA)
    low, high = -2., 2.
    for _ in range(16):
        midpoint = (low+high)/2
        if float(display(sample * 2**midpoint)[0,0] @ LUMA) < target_luma:
            low = midpoint
        else:
            high = midpoint
    ev = (low+high)/2
    fitted = float(display(sample*2**ev)[0,0] @ LUMA)
    return {'exposure_ev': ev, 'white_balance': [1.,1.,1.], 'limited': abs(fitted-target_luma)>.025,
            'basis': 'same_light_0.3_diffuse_receiver_not_object_median', 'reference_pixels': int(usable.sum()),
            'target_luma': target_luma, 'fitted_luma': fitted, 'receiver_rgb_linear': sample[0,0].tolist(),
            'uncertainty': 'neutral_reflectance_and_surface_normal_assumptions'}
