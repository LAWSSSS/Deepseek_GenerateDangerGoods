"""Review the exported PNG and spend corrections on defects they can address."""
import math

from .schema import (Action, LIGHT_PROFILES, MATERIAL_PROFILES, PHOTO_CONTROLS, TONE_PROFILES,
                     EFFECT_ACTIONS, Review, EffectsReview, FusionObservation, apply_actions, effects_review_schema)
from .strict_contact import pose_review_context, reconcile_pose_evidence


POSE_GUIDANCE = """ The required pose is lying on its side in the scrap pile, including cans and cylinders.
Reference images show identity and visible structure only; their upright or studio pose is not a target.
Never label a sideways object upright because the reference stands upright. Use upright only for
a visibly upright final object. An axis-angle measurement can corroborate lying but cannot disprove
visible floating, penetration or wrong depth order; report those concrete contact defects separately."""
POSE_GUIDANCE += """ Lighting, haze, edge, colour and camera texture are always IMAGE scope.
Asset scope is reserved for intrinsic, independently confirmed structural damage, never atmosphere."""


COMPACT_PROMPT = """Inspect this exported scrap-hopper composite: panorama, crop, current model reference.
Give seven 0–4 scores. 3 is usable; 4 is convincing. Check contact, depth order, lighting/color, edges,
sensor texture, material, and visible structure. Assess lying pose, scale, floating, penetration,
shadows, metal reflections, paint, rust and plastic. The requested finish is faded grey scrap.
Normal partial occlusion and foreshortening can hide features. Mark hidden or tiny features
unassessable; reserve asset scope for visible intrinsic damage. Respect intentional model revisions.
Use major/fatal only for concrete defects that compromise display. Judge the actual image.
Offer up to three effective actions from allowed values for a single cause, or none.
Candidate IDs identify geometric alternatives. More chroma means more desaturation.""" + POSE_GUIDANCE

FUSION_PROMPT = """Judge whether the inserted part belongs to the photographed scrap pile.
The first image is the delivered panorama and the second is its crop. Subsequent images are
references for identity and structure. Evaluate seams on the delivered photograph only.
Look for a halo or cutout edge, clean studio reflections, conflicting color or sensor texture,
an inconsistent contact shadow, floating, implausible scale or depth order. A recognizable object,
clean plastic, and real metal glints can occur naturally. Require concrete visible evidence for a defect.
Set trace to none when edge, light, contact and camera response give no visible insertion clue,
uncertain when evidence is insufficient, and visible when a clue remains. Identify the most
noticeable clue in evidence. Check material, structure and the original five scores on the same
0–4 scale, with 3 usable and 4 convincing. Respect occlusion and intentional model revisions.
Give only visible image defects. Each defect must describe the cue and where it appears.
Distinguish haze_clear (object too clear for nearby haze), haze_heavy (excessive object veil),
light_direction, reflection_flat (lost metal response), reflection_hard (studio-like reflection),
and surface_clean (evidenced missing surface dirt). Airborne haze does not imply deposited dust.
Use patch_seam for an artificial rectangular inserted region or shadow receiver cutoff,
and edge for the object's own outline. These are different defects.
Judge airlight, raised shadow levels and contrast against nearby surfaces of similar depth.
Do not demand a hazy object to have showroom crispness or all materials to match the ground colour.
Return JSON with trace, scores, defects and evidence. Each defect has only code, severity and evidence.""" + POSE_GUIDANCE

PLACEMENT_DEFECTS={'upright','floating','penetration','occlusion','scale','camera','shadow','structure'}
ACTION_CAUSES={
    'candidate':PLACEMENT_DEFECTS,
    'material':{'material','texture','reflection_flat','reflection_hard'},
    'light':{'lighting','shadow','color','material','light_direction','reflection_flat','reflection_hard'},
    'tone':{'color','lighting'},
    'chroma':{'color','material'},
    'dust':{'material','color','texture','surface_clean'},
    'contrast':{'lighting','texture','material'},
    'haze':{'color','lighting','texture','haze_clear','haze_heavy'},
    'sharpness':{'edge','texture','too_sharp','too_blurry'},
    'fog_extra':{'haze_clear','haze_heavy'},
    'fog_gain':{'haze_clear','haze_heavy'},
    'pool_kernel':{'too_sharp','too_blurry'},
    'surface_dust':{'surface_clean','surface_dust_heavy','identity_loss'},
}
CAUSE_SCORES={'upright':'support','floating':'support','penetration':'support','scale':'support',
              'camera':'support','shadow':'lighting','color':'lighting','haze_clear':'lighting',
              'haze_heavy':'lighting','light_direction':'lighting','reflection_flat':'material',
              'reflection_hard':'material','surface_clean':'material','surface_dust_heavy':'material',
              'too_sharp':'texture','too_blurry':'texture','patch_seam':'edge',
              'identity_loss':'material','haze_color':'lighting'}


def evidence_status(checks):
    statuses={check['status'] for check in checks.values()}
    return 'fail' if 'fail' in statuses else 'unknown' if 'unknown' in statuses or not statuses else 'pass'


def allowed_actions(params, comp, candidate_ids, *, render_style, adjustable=True):
    photo=comp.get('photo',{})
    options={'candidate':candidate_ids,'material':list(MATERIAL_PROFILES),'light':list(LIGHT_PROFILES),
             'tone':list(TONE_PROFILES),**{k:list(v) for k,v in PHOTO_CONTROLS.items()}}
    if render_style in ('scene','native') and not adjustable:
        options.pop('material');options.pop('dust')
    elif params['photo_model']=='camera' and render_style=='reference':
        options.pop('dust')
    if params.get('atmosphere')=='off':options.pop('haze',None)
    if params.get('chroma_mode') in ('local_budget','global_mean'):options.pop('chroma',None)
    if params.get('effects_enabled'):
        # Never offer two controls for the same physical effect.
        # The dedicated matte shader preserves source roughness, so the old
        # material profiles are not executable in this mode either.
        for name in ('dust','haze','sharpness','material'):options.pop(name,None)
        if (params.get('workflow')=='evidence_v1' and params.get('fog_mode')=='local_auto'
                and photo.get('effects',{}).get('edge_samples',0)>=12):
            options['sharpness']=list(PHOTO_CONTROLS['sharpness'])
        if params.get('effects_ai_adjust',True):
            options.update({k:list(values) for k,values in EFFECT_ACTIONS.items()})
            if params.get('fog_mode')=='local_auto':options.pop('fog_extra',None)
            else:options.pop('fog_gain',None)
            if params.get('weathering')=='preserve_source':
                options['surface_dust']=[v for v in options['surface_dust'] if float(v)<=.5]
            measured=photo.get('effects',{})
            if params.get('atmosphere')=='off' or measured.get('fog_base',0)<=0 or measured.get('sample_pixels',0)<20:
                options.pop('fog_extra',None)
                options.pop('fog_gain',None)
            if (params.get('workflow')=='evidence_v1' and params.get('fog_mode')=='local_auto'
                    or params.get('pool_weight',0)<=0 or measured.get('edge_samples',0)<12 or measured.get('sample_pixels',0)<20):
                options.pop('pool_kernel',None)
            if not measured.get('surface_dust_applied'):
                options.pop('surface_dust',None)
            if photo.get('local_context',{}).get('confidence','estimated')!='estimated':
                options.pop('fog_extra',None);options.pop('pool_kernel',None)
                options.pop('fog_gain',None)
    controls=photo.get('effective_controls',{})
    for kind,values in list(options.items()):
        current=params[kind]
        values=[v for v in values if (not math.isclose(float(v),float(current)) if kind in EFFECT_ACTIONS else v!=current)]
        if current in controls and kind in controls[current]:
            current_value=controls[current][kind]
            values=[v for v in values if not math.isclose(controls[v][kind],current_value,abs_tol=1e-4)]
        if kind=='tone' and photo.get('exposure_limited'):
            forbidden='brighter' if photo['exposure_error']<0 else 'darker'
            values=[v for v in values if v!=forbidden]
        if values:options[kind]=values
        else:options.pop(kind)
    return options


def fusion_actions(review, comp, allowed):
    """Select the repair in the program, using the recognizer's visible evidence."""
    defects={d.code for d in review.defects if d.visibility=='visible' and (d.scope=='image' or d.code!='structure')}
    severe={d.code for d in review.defects if d.visibility=='visible' and (d.scope=='image' or d.code!='structure')
            and d.severity in ('major','fatal')}
    issues=severe or defects
    photo=comp['photo']
    options=[]
    if 'haze_clear' in issues:options.append(('haze','more'))
    if 'haze_heavy' in issues:options.append(('haze','less'))
    if issues&{'floating','penetration','occlusion','scale','camera','upright'} and allowed.get('candidate'):
        options.append(('candidate',allowed['candidate'][0]))
    if issues&{'lighting','material'} and photo.get('local_tones',{}).get('bright_band_fraction',0)>.12:
        options.append(('light','softer'))
    if 'light_direction' in issues:options.append(('light','softer'))
    if 'reflection_flat' in issues:options.append(('material','clearer'))
    if 'reflection_hard' in issues:options.append(('material','softer'))
    if 'surface_clean' in issues:options.append(('dust','more'))
    if 'color' in issues and photo.get('after_chroma',0)>photo.get('scene_chroma',0)+3:
        options.append(('chroma','more'))
    if 'lighting' in issues and not photo.get('calibration'):
        difference=photo.get('after_luma',0)-photo.get('target_luma',0)
        if abs(difference)>10:options.append(('tone','darker' if difference>0 else 'brighter'))
    if 'lighting' in issues and photo.get('exposure_limited'):
        options.append(('light','defined'))
    if 'shadow' in issues and comp['checks'].get('physics',{}).get('status')=='pass':
        options.append(('light','defined'))
    if 'edge' in issues:
        options.append(('sharpness','less'))
    for kind,value in options:
        if value in allowed.get(kind,()):
            return [Action(kind=kind,value=value)]
    return []


def repair(params, review, allowed, candidate_ids, seen, *, dust_in_shader=False, suggestions=None):
    """Select one cause and one dependency stage. Rejected actions remain auditable."""
    severe={d.code for d in review.defects if d.severity in ('major','fatal') and d.visibility=='visible'}
    visible={d.code for d in review.defects if d.visibility=='visible'}
    limiting={code for causes in ACTION_CAUSES.values() for code in causes
              if getattr(review.scores,CAUSE_SCORES.get(code,code))<3}
    if not (severe or limiting) and not review.passes():
        # The five original scores may still sum to 15 even when each is usable.
        limiting={'lighting','texture','edge'} if not review.scores.passes() else set()
    selected=[];log=[];selected_causes=None;stage=None
    for action in review.actions if suggestions is None else suggestions:
        kind,value=action.kind,action.value
        dependency='placement' if kind=='candidate' else 'lighting' if kind=='light' else (
            'material' if kind in ('material','surface_dust') or (kind=='dust' and dust_in_shader) else 'photography')
        causes=ACTION_CAUSES[kind]&(severe or limiting or visible)
        if kind in EFFECT_ACTIONS:
            # Haze/camera/dust defects on a rendered image remain actionable even
            # when a provider incorrectly labels their scope as an asset defect.
            # This never authorizes structural repair or changes any score/veto.
            causes=ACTION_CAUSES[kind]&{d.code for d in review.defects if d.visibility=='visible'}
        if kind=='candidate':
            causes&={d.code for d in review.defects if d.visibility=='visible' and d.scope=='image'}
        row={'action':action.model_dump(),'from':params.get(kind),'stage':dependency,'causes':sorted(causes)}
        if kind not in params:reason='unavailable_or_saturated'
        elif value==params[kind]:reason='repeated_parameters'
        elif value not in allowed.get(kind,[]):reason='unavailable_or_saturated'
        elif not causes:reason='no_related_defect'
        elif kind in EFFECT_ACTIONS and not effect_direction(kind,value,params[kind],causes):reason='wrong_effect_direction'
        elif kind=='sharpness' and params.get('workflow')=='evidence_v1' and not (
                ('too_sharp' in causes and 'too_blurry' not in causes and PHOTO_CONTROLS[kind][value]<PHOTO_CONTROLS[kind][params[kind]])
                or ('too_blurry' in causes and 'too_sharp' not in causes and PHOTO_CONTROLS[kind][value]>PHOTO_CONTROLS[kind][params[kind]])):
            reason='wrong_effect_direction'
        elif any(a.kind==kind for a in selected):reason='duplicate_action'
        elif stage and (stage!=dependency or not (selected_causes&causes)):reason='different_cause_or_stage'
        else:
            reason='scheduled'
            selected.append(action);stage=dependency
            selected_causes=causes if selected_causes is None else selected_causes&causes
        log.append({**row,'disposition':reason})
    updated=apply_actions(params,selected,candidate_ids) if selected else None
    if updated is not None and seen(updated):
        for row in log:
            if row['disposition']=='scheduled':row['disposition']='repeated_parameters'
        updated=None
    return updated,log


def effect_direction(kind,value,current,causes):
    if kind=='surface_dust' and 'identity_loss' in causes:
        return 'surface_clean' not in causes and float(value)<float(current)
    more,less={'fog_extra':('haze_clear','haze_heavy'),
               'fog_gain':('haze_clear','haze_heavy'),
               'pool_kernel':('too_sharp','too_blurry'),
               'surface_dust':('surface_clean','surface_dust_heavy')}[kind]
    # Conflicting observations cannot justify an automatic numeric change.
    if more in causes and less in causes:return False
    return (more in causes and float(value)>float(current)) or (less in causes and float(value)<float(current))


async def inspect(vision,prompt,asset,comp,params,allowed,render_dir,*,compact=False,
                  candidate_evidence=(),geometry_board=None):
    checks=comp['checks']
    payload={'name':asset['name'],'features':asset['semantic']['key_features'],'params':params,
             'allowed':allowed,'placement_options':list(candidate_evidence),
             'pose_policy':pose_review_context(checks)}
    images=[comp['preview'],comp['crop'],asset['board']]
    if params.get('effects_enabled'):
        prompt=FUSION_PROMPT.replace(
            'Return JSON with trace, scores, defects and evidence. Each defect has only code, severity and evidence.',
            'Return JSON matching the supplied schema, including actions and defect region/scope/visibility.')
        prompt+=''' Only suggest up to three actions from allowed, preferably one cause per round.
fog_extra is an ADDITIONAL airlight opacity, not total haze. It attenuates reflection and body colour
together without changing geometric alpha. surface_dust is separate deposited nonmetal dust on
metal/paint; painted metal already uses half coverage and plastic is untouched.
pool_kernel is an alpha-weighted average pooling square in final image pixels; 1 disables pooling.
Use haze_clear/haze_heavy for air, surface_clean/surface_dust_heavy for material coverage,
too_sharp/too_blurry for camera detail. Increase/decrease the corresponding numeric control.
Use exact allowed strings as values. Never compensate wrong lighting with dust or blur.
If evidence is uncertain or a control is unavailable, give no action for it.
Assess photographic agreement against the local neighbourhood, not distant bright areas.
The supplied numerical measurements are not proof of visual acceptance.'''
        if params.get('weathering')=='preserve_source':
            prompt+=''' Preserve original identity: moderate weathering, recognizable paint/hue,
markings and source texture; matte does not mean uniformly grey. Use identity_loss when an
opaque dust layer erases the original appearance. Do not increase dust to hide lighting defects.
The user limits surface_dust to 0.5; paint already receives half. Haze_color means wrong airlight
colour, which an opacity change alone cannot fix. For local_auto, fog_extra is inactive and
fog_gain multiplies the locally estimated optical depth; 1 is the measured estimate, not 100% fog.
Never repair a rectangular patch_seam by more haze. Do not invent a different real object size.'''
        payload.update(effects=comp.get('photo',{}).get('effects'),
                       evidence_status=evidence_status(checks),
                       checks={k:v['status'] for k,v in checks.items()})
        if params.get('chroma_mode') in ('local_budget','global_mean'):
            payload['final_local_chroma']=comp.get('photo',{}).get('final_chroma')
            payload['global_saturation']=comp.get('photo',{}).get('global_saturation')
            prompt+=' The final image already has a measured local chroma ceiling. Do not add dust or haze to hide a colour discrepancy; identify wrong illumination or airlight when visible.'
        from pathlib import Path
        context=Path(comp['image']).with_suffix('.context.jpg')
        if context.is_file():images[1]=context
        answer=await vision.ask('effects-review',effects_review_schema(allowed),prompt,payload,images,
            image_limits=[(1280,720),(1200,900),(512,512)],max_output_tokens=2048,plain_json=True)
        return reconcile_pose_evidence(answer,checks)
    native=params.get('photo_model') in ('native','local')
    if native:
        # The recognizer sees material references, without physics histories.
        images.insert(2,render_dir/'object.png')
        local=params.get('photo_model')=='local'
        if local:
            from pathlib import Path
            images[1]=Path(comp['image']).with_suffix('.context.jpg')
        prompt=FUSION_PROMPT
        if local:
            # Do not let distant luminance dominate the photographic decision.
            images=[images[1],images[2],images[3]]
            prompt=FUSION_PROMPT.replace(
                'The first image is the delivered panorama and the second is its crop. Subsequent images are\nreferences for identity and structure. Evaluate seams on the delivered photograph only.',
                'The first image is the delivered local neighbourhood. Images 2 and 3 are isolated material/identity references. Evaluate seams and photography only on image 1.')
            prompt+=' Faded paint and deposited dust are intentional: do not demand the fresh saturated reference colour. Keep material differences; do not demand all materials to look like rusty steel.'
        answer=await vision.ask('fusion-recognition',FusionObservation,prompt,
            {'name':asset['name'],'features':asset['semantic']['key_features'],
             'material_classes':asset['semantic'].get('material_classes',[]),
             'pose_policy':pose_review_context(checks)},images,plain_json=True)
        return reconcile_pose_evidence(answer.review(),checks)
    if compact:
        fields=('status','measurement_px','expected_px','measurement_deg','measurements','projection_error_px')
        payload['checks']={k:{f:c[f] for f in fields if f in c} for k,c in checks.items()}
        if comp.get('photo',{}).get('exposure_limited'):
            payload['exposure_limit']='Prefer lighting correction if exposure cannot be matched.'
        limits=[(768,512),(1024,900),(512,512)]
        if geometry_board:
            images.append(geometry_board);limits.append((768,512))
        answer=await vision.ask('review-compact',Review,COMPACT_PROMPT,payload,images,
                                image_limits=limits,max_output_tokens=1536)
        return reconcile_pose_evidence(answer,checks)
    payload.update(checks=checks,estimated_size_mm=asset['semantic']['major_mm'],
        source_note='Use the current model and its intentional revisions. Assess visible structure at the actual pose.',
        profile_meanings={'clearer':'reduce roughness','softer':'soften reflection',
                          'weathered':'increase surface dust on exposed paint',
                          'defined':'stronger broad light','darker':'reduce exposure'})
    images.insert(2,render_dir/'object.png')
    if geometry_board:images.append(geometry_board)
    answer=await vision.ask('review',Review,prompt+POSE_GUIDANCE,payload,images)
    return reconcile_pose_evidence(answer,checks)
