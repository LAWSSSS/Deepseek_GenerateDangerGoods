"""Connect the shared model contracts to the existing display queue."""
from scrap_composite.photography_agent import run_photography
from scrap_composite.photography_controls import composite_changes
from scrap_composite.photography_measurements import measurements
from scrap_composite.photography_protocol import PhotoContext, PhotoReport
from scrap_composite.workflow import observe, plan_execution

from .ai import InvalidStructuredReply
from .review import ACTION_CAUSES, effect_direction, repair
from .schema import Action, EFFECT_ACTIONS, EffectsReview, LIGHT_PROFILES, PHOTO_CONTROLS, TONE_PROFILES, apply_actions
from .strict_contact import pose_review_context, reconcile_pose_evidence


def frame_inputs(asset, comp):
    from .asset_profiles import load_profile, descriptor
    width, height = comp['checks']['size']['pixels']
    left, top, right, bottom = comp['edit_bounds']
    semantic = asset.get('semantic', {})
    pose = pose_review_context(comp['checks'])
    paths = [(kind, comp[key]) for kind, key in (('frame', 'image'), ('local', 'crop')) if comp.get(key)]
    if asset.get('board'):
        paths.append(('asset', asset['board']))
    profile=load_profile(asset.get('mesh_hash'))
    context = PhotoContext(
        name=asset['name'][:80], target_region=(left/width, top/height, right/width, bottom/height),
        images=[kind for kind, _ in paths], material=[s[:24] for s in semantic.get('material_classes', [])[:6]],
        features=[s[:160] for s in semantic.get('key_features', [])[:12] if s.strip()],
        intended_pose='lying_on_side', axis_deg=pose['measured_axis_angle_deg'],
        checks={key: value['status'] for key, value in comp['checks'].items()},
        measurements=measurements(comp.get('photo', {})),
        asset_profile=descriptor(profile) if profile else {},
        execution_state=comp.get('execution_state', {}),
        measurement_evidence={key:comp['photo'][key] for key in ('effects','calibration','material_readback','lighting_readback') if key in comp.get('photo',{})},
    )
    return context, [path for _, path in paths]


async def inspect_frame(client, asset, comp, image_hash, *, profile='legacy', parameters=None):
    if profile == 'photographer':
        if parameters is not None:
            prior = dict(comp.get('execution_state', {}))
            if prior.get('backend') == 'photoshop':
                prior['photoshop_parameters'] = prior.pop('parameters')
            comp={**comp,'execution_state':{**prior,'parameters':dict(parameters), 'image_sha256':image_hash}}
        context, images = frame_inputs(asset, comp)
        decision = await run_photography(client, context, image_hash, images=images, parameters=parameters)
        return decision.report, _review(decision.report.legacy(), comp['checks'])
    width, height = comp['checks']['size']['pixels']
    left, top, right, bottom = comp['edit_bounds']
    target_region = [left/width, top/height, right/width, bottom/height]
    report = await observe(
        client, image_hash, [comp['image'], comp['crop'], asset['board']],
        {'name': asset['name'], 'target_region_in_first_image': target_region,
         'pose_measurement': pose_review_context(comp['checks']),
         'program_checks': {key: value['status'] for key, value in comp['checks'].items()},
         'execution_state': comp.get('execution_state', {})},
        prompt='Preserve original material and identifying structures. Judge the actual image, including '
               'lying pose, floating, penetration, occlusion, local light, contrast, haze and camera edges. '
               'The screen angle alone does not establish an upright pose. Call a contact or depth-order '
               'defect visible only when you can point to an actual gap, intersection or wrong overlap. '
               'Use occlusion for a wrong front/back overlap, shadow for missing contact shadows, and edge '
               'for abrupt transitions. A missing or weak shadow by itself is a shadow finding. '
               'Describe a visible gap separately if present. Locate findings in target_region_in_first_image. '
               'Mark hidden or ambiguous contact as unassessable. '
               'Use major or fatal for a concrete defect that compromises display. Do not suggest operations.',
    )
    return report, _review(report, comp['checks'])


def _review(report, checks):
    defects = [{'code': f.code, 'severity': f.severity, 'region': f.region or (0., 0., 1., 1.),
                'scope': f.scope if f.code=='structure' else 'image',
                'visibility': f.visibility, 'evidence': f.evidence[:160],
                'physical_cue': f.physical_cue, 'observed_state': f.observed_state,
                'reference_role': f.reference_role, 'reference_region': f.reference_region}
               for f in report.findings]
    review = EffectsReview(scores=report.scores.model_dump(), trace=report.trace,
                           defects=defects, actions=[], evidence=report.summary)
    return reconcile_pose_evidence(review, checks)


def initial_recipe(db_path, asset, candidate, environment, params):
    from scrap_composite.knowledge import features_from_environment, recommend
    from .photography import VERSION
    from .schema import EFFECT_ACTIONS, PHOTO_CONTROLS, TONE_PROFILES, bounded_surface_dust

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
    if bounded_surface_dust(params) and float(updates.get('surface_dust',0))>.5:
        return params, {**result, 'status':'no_recommendation','reason':'surface_dust_limit','overrides':{}}
    return {**params, **updates}, {**result,'features':features,'versions':versions}


def algorithm_versions(params, photo_version):
    from scrap_composite.knowledge import versions_from_params
    from .common import implementation_manifest, signature

    manifest=implementation_manifest()
    return {**versions_from_params(params, photo_version),
            'implementation':signature(manifest['sources'], manifest['core_sources'], manifest['dependencies'])}


def planner_capabilities(params, allowed, report):
    """Offer only controls with a visible cause and a permitted direction."""
    capabilities = {}
    for kind, values in allowed.items():
        if kind not in params or kind not in ACTION_CAUSES:
            continue
        findings = [f for f in report.findings if f.visibility == 'visible'
                    and f.code in ACTION_CAUSES[kind] and (kind != 'candidate' or f.scope == 'image')]
        causes = {f.code for f in findings}
        current = params[kind]
        if kind in EFFECT_ACTIONS:
            values = [v for v in values if effect_direction(kind, v, current, causes)]
        elif kind == 'sharpness':
            controls = PHOTO_CONTROLS[kind]
            values = [v for v in values if
                      (causes == {'too_sharp'} and controls[v] < controls[current]) or
                      (causes == {'too_blurry'} and controls[v] > controls[current])]
        if not (causes and values):
            continue
        capability = {'values': values, 'causes': sorted(causes), 'evidence_ids':[f.id for f in findings]}
        if kind == 'tone':
            capability['meaning'] = 'darker/brighter change exposure; cooler/warmer change white balance'
        elif kind == 'light':
            capability['meaning'] = {v:LIGHT_PROFILES[v] for v in values}
        capabilities[kind] = capability
    return capabilities


async def plan_frame(client, report, review, comp, params, allowed, ids, seen, *, dust_in_shader, recommendation):
    if isinstance(report, PhotoReport):
        from .common import file_hash
        try:
            observed = report.context.execution_state.get('parameters')
            if observed is not None and observed != params:
                raise ValueError('parameters changed since image diagnosis')
            changes = composite_changes(params, allowed, comp['photo'], tone_profiles=TONE_PROFILES,
                                        light_profiles=LIGHT_PROFILES)
            changes = [change for change in changes if not seen({**params, change.control: change.value})]
            decision = await run_photography(client, report.context, file_hash(comp['image']),
                target=report.target, changes=changes, plan=True, report=report, parameters=params)
            if decision.updates and file_hash(comp['image']) != report.image_sha256:
                raise ValueError('photography plan image changed before execution')
        except (OSError, ValueError, InvalidStructuredReply) as error:
            return None, [{'disposition': 'invalid_execution_plan', 'reason': str(error)}], None
        plan = decision.plan.model_dump(mode='json')
        if not decision.updates:
            return None, [{'disposition': decision.plan.reason}], plan
        operation = decision.plan.operation
        selected = Action(kind=operation.control, value=operation.value)
        updated = apply_actions(params, [selected], ids)
        return updated, [{'action': selected.model_dump(), 'from': operation.current,
                          'stage': operation.stage, 'evidence_id': operation.evidence_id,
                          'disposition': 'scheduled'}], plan
    capabilities = planner_capabilities(params, allowed, report)
    if not capabilities:
        return None, [], None
    try:
        plan, updates = await plan_execution(
            client, report, {kind: params[kind] for kind in capabilities}, capabilities,
            {key: comp['photo'].get(key) for key in ('before_luma', 'after_luma', 'contrast_gain',
                                                   'blur_sigma', 'haze', 'exposure_ev')},
            recommendation,
        )
    except (ValueError, InvalidStructuredReply) as error:
        return None, [{'disposition': 'invalid_execution_plan', 'reason': str(error)}], None
    selected = [Action(kind=kind, value=value) for kind, value in updates.items()]
    updated, decisions = repair(params, review, allowed, ids, seen, dust_in_shader=dust_in_shader, suggestions=selected)
    return updated, decisions, plan.model_dump(mode='json')
