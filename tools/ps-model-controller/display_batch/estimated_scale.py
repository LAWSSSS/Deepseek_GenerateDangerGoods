"""Provisional scene scale from model observations and the known vehicle width."""
import math

import numpy as np

from .common import signature

VERSION = 'visible-layer-preview-scale-v1'
DEPTH_TOLERANCE = .085


def allows_estimate(gate):
    """Missing observations may use a prior; conflicting observations still stop."""
    conflicts = {'vehicle_class_conflicts_with_supplied_reference', 'inconsistent_width_correspondences'}
    return not conflicts.intersection(gate.get('risk_codes', ()))


def visible_reference(scene, site):
    """Check the model's reference line against its visible support and depth."""
    if site.get('normal_inherited') or site.get('reference_status') == 'unknown':
        return None, 'inherited_or_unidentified_reference'
    points = np.asarray(site.get('reference'), dtype=float)
    lengths = np.asarray(site.get('reference_mm'), dtype=float)
    if (points.shape != (2, 2) or lengths.shape != (2,) or not np.isfinite(points).all()
            or not np.isfinite(lengths).all() or not (0 < lengths[0] <= lengths[1])
            or np.any((points < 0) | (points > 1))):
        return None, 'invalid_reference_line_or_length'
    endpoints = points * np.subtract(scene['size'], 1)
    dx, dy = endpoints[1] - endpoints[0]
    if math.hypot(dx, dy) < 8:
        return None, 'reference_too_short'
    samples = np.rint(np.linspace(*endpoints, 25)).astype(int)
    x, y = samples.T
    depth = scene['depth_native'][y, x]
    visible = (site['mask'][y, x] & scene['allowed'][y, x] & ~scene['protected'][y, x]
               & np.isfinite(depth) & (abs(depth - site['depth']) <= DEPTH_TOLERANCE))
    coverage = float(visible.mean())
    if coverage < .60:
        return None, 'reference_not_on_visible_support_layer'
    elevation = math.radians(sum(scene['answer']['elevation']) / 2)
    span = math.hypot(dx, dy / max(.2, math.sin(elevation)))
    return {
        'id': site['id'], 'anchor_px': endpoints.mean(axis=0).tolist(), 'endpoints_px': endpoints.tolist(),
        'depth': float(site['depth']), 'reference_mm': lengths.tolist(),
        'pixels_per_mm': float(span / math.sqrt(lengths.prod())), 'interval': (span / lengths[::-1]).tolist(),
        'confidence': float(site['confidence']), 'line_coverage': coverage,
        'basis': 'model_category_length_on_visible_surface', 'status': 'unknown',
    }, None


def prepare_preview_scale(scene, reference, gate):
    """Use a broad scene prior when a measured width correspondence is missing."""
    width_mm = float(reference['internal_width_m']) * 1000
    polygon = np.asarray(scene['answer']['hopper'], dtype=float)
    span = float(np.ptp(polygon[:, 0]) * (scene['size'][0] - 1))
    if not math.isfinite(span) or span < 32:
        raise ValueError('insufficient_visible_cargo_span_for_estimate')
    nominal = span / width_mm
    interval = [nominal * .60, nominal * 1.65]
    rows, rejected, seen = [], [], set()
    for site in scene['supports']:
        row, reason = visible_reference(scene, site)
        if row is None:
            rejected.append({'id':site['id'], 'reason':reason})
            continue
        line = tuple(sorted(tuple(round(v, 1) for v in p) for p in row['endpoints_px']))
        if line in seen:
            rejected.append({'id':site['id'], 'reason':'duplicate_reference_line'})
            continue
        seen.add(line)
        if row['interval'][0] > interval[1] or row['interval'][1] < interval[0]:
            rejected.append({'id':site['id'], 'reason':'reference_conflicts_with_broad_scene_prior',
                             'estimated_interval_px_per_mm':row['interval']})
            continue
        rows.append(row)
    result = {'version':VERSION, 'status':'unknown', 'references':rows, 'rejected_references':rejected,
              'prior':{'basis':'visible_cargo_span_and_user_vehicle_width_approximation',
                       'visible_span_px':span, 'vehicle_width_mm':width_mm, 'pixels_per_mm':nominal,
                       'interval_px_per_mm':interval, 'image_span_is_measured_cross_section':False},
              'elevation_interval':scene['answer']['elevation'], 'vehicle_reference':reference,
              'projection_preflight':gate, 'depth_is_metric':False}
    result['signature'] = signature(scene['key'], result)
    return result


def local_scale(scene, site, anchor, profile):
    """Fix the physical size across scenes and never choose a size to make a pose fit."""
    data = scene['estimated_preview_scale']
    factor = scene.get('_image_scale', 1.)
    prior = data['prior']
    nominal = prior['pixels_per_mm']
    lo, hi = prior['interval_px_per_mm']
    rejected = list(data['rejected_references'])
    nearby = [row for row in data['references'] if abs(row['depth'] - site['depth']) <= DEPTH_TOLERANCE]
    rejected.extend({'id':row['id'], 'reason':'cross_depth_reference'}
                    for row in data['references'] if row not in nearby)
    same = [row for row in nearby if row['id'] == site['id']]
    selected = same or nearby
    if selected:
        center = np.asarray(anchor) / factor
        diagonal = np.linalg.norm(scene['size']) / factor
        values = np.asarray([np.clip(row['pixels_per_mm'], lo, hi) for row in selected])
        weights = np.asarray([row['line_coverage'] * max(.1, row['confidence']) /
                              (1 + np.linalg.norm(center-row['anchor_px']) / (.12 * diagonal))
                              for row in selected])
        local = float(np.exp(np.average(np.log(values), weights=weights)))
        # Give an unmeasured category dimension limited weight against the scene prior.
        ppm = math.exp(.65 * math.log(nominal) + .35 * math.log(local))
        basis = 'same_visible_layer_category_estimate_with_vehicle_width_prior'
    else:
        ppm = nominal
        basis = prior['basis']
    references = [{**row, 'anchor_px':(np.asarray(row['anchor_px'])*factor).tolist(),
                   'endpoints_px':(np.asarray(row['endpoints_px'])*factor).tolist(),
                   'pixels_per_mm':row['pixels_per_mm']*factor, 'interval':[v*factor for v in row['interval']]}
                  for row in selected]
    target = profile['nominal_mm'] * ppm * factor
    evidence = {'version':'reference-projection-v1', 'estimation_version':VERSION,
                'status':'unknown', 'basis':basis, 'fixed_physical_size':True, 'physical_size':profile,
                'pixels_per_mm':ppm*factor, 'calibration_interval_px_per_mm':[lo*factor,hi*factor],
                'calibration':{}, 'calibration_signature':data['signature'], 'references':references,
                'rejected_references':rejected, 'scale_prior':prior,
                'projection_preflight':data['projection_preflight'], 'anchor_px':list(anchor),
                'coordinate_image_size':list(scene['size']), 'reference_projection_correction':1.,
                'size_selected_to_fit':False, 'depth_is_metric':False}
    return [target, target], evidence
