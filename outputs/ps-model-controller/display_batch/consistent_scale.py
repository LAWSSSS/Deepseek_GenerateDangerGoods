"""One length per asset; shared image calibration never changes that length to fit."""
import copy
import math
import numpy as np

VERSION = 'physical-consistent-v2'


def calibration_to_native(evidence, factor, size):
    """Return independent native-pixel calibration; references are shared during search."""
    result = copy.deepcopy(evidence)
    result['coordinate_image_size'] = list(size)
    result['pixels_per_mm'] /= factor
    result['calibration_interval_px_per_mm'] = [v / factor for v in result['calibration_interval_px_per_mm']]
    for reference in result['references']:
        reference['anchor_px'] = [v / factor for v in reference['anchor_px']]
        reference['pixels_per_mm'] /= factor
        reference['interval'] = [v / factor for v in reference['interval']]
    return result


def asset_profile(asset, factor=1.):
    previous = asset.get('physical_size', {})
    lo, hi = previous.get('source_interval_mm', asset['semantic']['major_mm'])
    measured = asset.get('measured_major_mm')
    nominal = float(measured or math.sqrt(lo * hi))
    return {'version': VERSION, 'source_interval_mm': [lo, hi],
            'source_nominal_mm': nominal, 'nominal_mm': nominal * factor,
            'interval_mm': [lo * factor, hi * factor], 'user_factor': factor,
            'basis': 'measured_asset_length' if measured else 'cached_semantic_geometric_midpoint',
            'user_basis': 'explicit_user_size_preference_not_a_measurement',
            'status': 'pass' if measured and factor == 1 else 'unknown'}


def scaled_asset(asset, factor):
    result = copy.copy(asset)
    result['physical_size'] = asset_profile(asset, factor)
    return result


def scene_calibration(scene):
    elevation = math.radians(sum(scene['answer']['elevation']) / 2)
    rows = []; rejected = []; seen = set()
    for s in scene['supports']:
        if not s.get('reference') or s.get('reference_status') == 'unknown':
            continue
        # Segmentation inherits a neighbour's reference for legacy display estimates.
        # It does not observe a second ruler at this segment's depth.
        if s.get('plane_basis') == 'segmentation_only_hidden_height_unknown':
            rejected.append({'id': s['id'], 'reason': 'inherited_reference_has_no_local_measurement'})
            continue
        reference_key = tuple(sorted(tuple(round(float(v), 6) for v in p) for p in s['reference']))
        if reference_key in seen:
            rejected.append({'id': s['id'], 'reason': 'duplicate_physical_reference'})
            continue
        seen.add(reference_key)
        ends = np.asarray(s['reference']) * np.subtract(scene['size'], 1)
        delta = ends[1] - ends[0]
        unprojected = float(np.linalg.norm([delta[0], delta[1] / max(.2, math.sin(elevation))]))
        lo, hi = s['reference_mm']
        if unprojected < 8 or not 0 < lo <= hi:
            continue
        rows.append({'id': s['id'], 'anchor_px': ends.mean(axis=0).tolist(),
                     'depth': s['depth'], 'reference': s['reference'], 'reference_mm': [lo, hi],
                     'pixels_per_mm': unprojected / math.sqrt(lo * hi),
                     'interval': [unprojected / hi, unprojected / lo],
                     'confidence': float(s.get('confidence', .5)),
                     'measured': s.get('reference_status') == 'measured'})
    # This prior belongs to the background, never to the desired object's screen size.
    hopper = np.asarray(scene['answer']['hopper'])
    span = float(np.ptp(hopper[:, 0]) * (scene['size'][0] - 1))
    return {'version': VERSION, 'references': rows, 'rejected_references': rejected,
            'elevation_deg': math.degrees(elevation),
            'fallback_pixels_per_mm': span / math.sqrt(2000 * 5000),
            'fallback_basis': 'single_scene_hopper_width_prior_2000_to_5000_mm',
            'depth_is_metric': False}


def local_scale(scene, site, anchor, profile):
    calibration = scene.setdefault('scale_calibration', scene_calibration(scene))
    rows = calibration['references']
    nearby = [r for r in rows if abs(r['depth'] - site['depth']) <= .085]
    rejected = list(calibration.get('rejected_references', []))
    rejected += [{'id': r['id'], 'reason': 'cross_depth_reference'} for r in rows if r not in nearby]
    selected = nearby or rows
    basis = 'shared_same_layer_references' if nearby else 'shared_uncalibrated_scene_estimate'
    if selected:
        values = np.log([r['pixels_per_mm'] for r in selected])
        center = np.median(values)
        valid = np.abs(values - center) <= max(.25, 2.5 * float(np.median(abs(values - center))))
        for r, good in zip(selected, valid):
            if not good: rejected.append({'id': r['id'], 'reason': 'inconsistent_reference_scale'})
        selected = [r for r, good in zip(selected, valid) if good]
        diagonal = float(np.linalg.norm(scene['size']))
        weights = np.array([max(.05, r['confidence']) /
            (1 + np.linalg.norm(np.subtract(anchor, r['anchor_px'])) / (diagonal * .12)) /
            max(1., r['reference_mm'][1] / r['reference_mm'][0]) for r in selected])
        ppm = float(np.exp(np.average(np.log([r['pixels_per_mm'] for r in selected]), weights=weights)))
        uncertainty = [min(r['interval'][0] for r in selected), max(r['interval'][1] for r in selected)]
    else:
        ppm = calibration['fallback_pixels_per_mm']
        uncertainty = [ppm / math.sqrt(2.5), ppm * math.sqrt(2.5)]
        basis = calibration['fallback_basis']
    target = ppm * profile['nominal_mm']
    measured = bool(nearby) and len(selected) >= 2 and all(r['measured'] for r in selected)
    status = 'pass' if measured and profile['status'] == 'pass' and uncertainty[1]/uncertainty[0] <= 1.25 else 'unknown'
    return [target, target], {'version': VERSION, 'status': status, 'basis': basis,
        'fixed_physical_size': True, 'physical_size': profile, 'pixels_per_mm': ppm,
        'calibration_interval_px_per_mm': uncertainty, 'references': selected,
        'rejected_references': rejected, 'anchor_px': list(anchor), 'coordinate_image_size': list(scene['size']),
        'reference_projection_correction': 1., 'size_selected_to_fit': False}
