"""Connect the shared model contracts to the existing display queue."""
from scrap_composite.workflow import observe, plan_execution

from .review import ACTION_CAUSES, repair
from .schema import Action, EffectsReview
from .strict_contact import reconcile_pose_evidence


async def inspect_frame(client, asset, comp, image_hash):
    report = await observe(
        client, image_hash, [comp['image'], comp['crop'], asset['board']],
        {'name': asset['name'], 'edit_bounds_px': comp['edit_bounds'],
         'program_checks': {key: value['status'] for key, value in comp['checks'].items()}},
        prompt='Preserve original material and identifying structures. Judge the actual image, including '
               'lying pose, floating, penetration, occlusion, local light, contrast, haze and camera edges. '
               'Use major or fatal for a concrete defect that compromises display. Do not suggest operations.',
    )
    defects = [{'code': f.code, 'severity': f.severity, 'region': f.region or (0., 0., 1., 1.),
                'scope': f.scope, 'visibility': f.visibility, 'evidence': f.evidence[:160]}
               for f in report.findings]
    review = EffectsReview(scores=report.scores.model_dump(), trace=report.trace,
                           defects=defects, actions=[], evidence=report.summary)
    return report, reconcile_pose_evidence(review, comp['checks'])


def initial_recipe(db_path, asset, candidate, environment, params):
    from scrap_composite.knowledge import features_from_environment, recommend
    from .photography import VERSION
    from .schema import EFFECT_ACTIONS, PHOTO_CONTROLS, TONE_PROFILES

    features = features_from_environment(environment, params,
        material_group=asset['semantic'].get('material_classes'),
        target_major_px=candidate.get('expected_major_px',candidate['target_px']))
    versions = algorithm_versions(params, environment.get('local_context',{}).get('version') or VERSION)
    result = recommend(features, versions, db_path=db_path)
    approved_controls = {name:PHOTO_CONTROLS[name] for name in ('contrast', 'sharpness', 'chroma', 'haze')}
    approved_controls.update({'tone':TONE_PROFILES,'fog_gain':EFFECT_ACTIONS['fog_gain'],
                              'surface_dust':EFFECT_ACTIONS['surface_dust']})
    updates = result['overrides']
    if any(key not in params or value not in approved_controls.get(key, ()) for key,value in updates.items()):
        return params, {**result, 'status':'no_recommendation','reason':'rule_control_unavailable','overrides':{}}
    if 'fog_gain' in updates and (params.get('fog_mode')!='local_auto' or params.get('atmosphere')=='off'):
        return params, {**result, 'status':'no_recommendation','reason':'local_atmosphere_disabled','overrides':{}}
    if params.get('weathering')=='preserve_source' and float(updates.get('surface_dust',0))>.5:
        return params, {**result, 'status':'no_recommendation','reason':'surface_dust_limit','overrides':{}}
    return {**params, **updates}, {**result,'features':features,'versions':versions}


def algorithm_versions(params, photo_version):
    from scrap_composite.knowledge import versions_from_params
    from .common import implementation_manifest, signature

    manifest=implementation_manifest()
    return {**versions_from_params(params, photo_version),
            'implementation':signature(manifest['sources'], manifest['core_sources'], manifest['dependencies'])}


def planner_capabilities(params, allowed):
    return {kind: {'values': values, 'causes': sorted(ACTION_CAUSES[kind])}
            for kind, values in allowed.items() if kind in params and kind in ACTION_CAUSES}


async def plan_frame(client, report, review, comp, params, allowed, ids, seen, *, dust_in_shader, recommendation):
    capabilities = planner_capabilities(params, allowed)
    if not capabilities or not any(
        finding.visibility == 'visible' and any(finding.code in c['causes'] for c in capabilities.values())
        for finding in report.findings
    ):
        return None, [], None
    try:
        plan, updates = await plan_execution(
            client, report, {kind: params[kind] for kind in capabilities}, capabilities,
            {key: comp['photo'].get(key) for key in ('before_luma', 'after_luma', 'contrast_gain',
                                                   'blur_sigma', 'haze', 'exposure_ev')},
            recommendation,
        )
    except ValueError as error:
        return None, [{'disposition': 'invalid_execution_plan', 'reason': str(error)}], None
    selected = [Action(kind=kind, value=value) for kind, value in updates.items()]
    updated, decisions = repair(params, review, allowed, ids, seen, dust_in_shader=dust_in_shader, suggestions=selected)
    return updated, decisions, plan.model_dump(mode='json')
