"""Review the exported PNG and spend corrections on defects they can address."""
import math

from .schema import Action, LIGHT_PROFILES, MATERIAL_PROFILES, PHOTO_CONTROLS, TONE_PROFILES, Review, FusionObservation, apply_actions


COMPACT_PROMPT = """Inspect this exported scrap-hopper composite: panorama, crop, current model reference.
Give seven 0–4 scores. 3 is usable; 4 is convincing. Check contact, depth order, lighting/color, edges,
sensor texture, material, and visible structure. Assess lying pose, scale, floating, penetration,
shadows, metal reflections, paint, rust and plastic. The requested finish is faded grey scrap.
Normal partial occlusion and foreshortening can hide features. Mark hidden or tiny features
unassessable; reserve asset scope for visible intrinsic damage. Respect intentional model revisions.
Use major/fatal only for concrete defects that compromise display. Judge the actual image.
Offer up to three effective actions from allowed values for a single cause, or none.
Candidate IDs identify geometric alternatives. More chroma means more desaturation."""

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
Judge airlight, raised shadow levels and contrast against nearby surfaces of similar depth.
Do not demand a hazy object to have showroom crispness or all materials to match the ground colour.
Return JSON with trace, scores, defects and evidence. Each defect has only code, severity and evidence."""

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
    'sharpness':{'edge','texture'},
}
CAUSE_SCORES={'upright':'support','floating':'support','penetration':'support','scale':'support',
              'camera':'support','shadow':'lighting','color':'lighting','haze_clear':'lighting',
              'haze_heavy':'lighting','light_direction':'lighting','reflection_flat':'material',
              'reflection_hard':'material','surface_clean':'material'}


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
    controls=photo.get('effective_controls',{})
    for kind,values in list(options.items()):
        current=params[kind]
        values=[v for v in values if v!=current]
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
    defects={d.code for d in review.defects if d.visibility=='visible' and d.scope=='image'}
    severe={d.code for d in review.defects if d.visibility=='visible' and d.scope=='image'
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
            'material' if kind=='material' or (kind=='dust' and dust_in_shader) else 'photography')
        causes=ACTION_CAUSES[kind]&(severe or limiting or visible)
        if kind=='candidate':
            causes&={d.code for d in review.defects if d.visibility=='visible' and d.scope=='image'}
        row={'action':action.model_dump(),'from':params[kind],'stage':dependency,'causes':sorted(causes)}
        if value==params[kind]:reason='repeated_parameters'
        elif value not in allowed.get(kind,[]):reason='unavailable_or_saturated'
        elif not causes:reason='no_related_defect'
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


async def inspect(vision,prompt,asset,comp,params,allowed,render_dir,*,compact=False,
                  candidate_evidence=(),geometry_board=None):
    checks=comp['checks']
    payload={'name':asset['name'],'features':asset['semantic']['key_features'],'params':params,
             'allowed':allowed,'placement_options':list(candidate_evidence)}
    images=[comp['preview'],comp['crop'],asset['board']]
    native=params.get('photo_model')=='native'
    if native:
        # Keep the full set of visual references. The model only recognizes, so
        # physics histories and the allowed repair graph are unnecessary.
        images.insert(2,render_dir/'object.png')
        answer=await vision.ask('fusion-recognition',FusionObservation,FUSION_PROMPT,
            {'name':asset['name'],'features':asset['semantic']['key_features'],
             'material_classes':asset['semantic'].get('material_classes',[])},images,plain_json=True)
        return answer.review()
    if compact:
        fields=('status','measurement_px','expected_px','measurement_deg','measurements','projection_error_px')
        payload['checks']={k:{f:c[f] for f in fields if f in c} for k,c in checks.items()}
        if comp.get('photo',{}).get('exposure_limited'):
            payload['exposure_limit']='Prefer lighting correction if exposure cannot be matched.'
        limits=[(768,512),(1024,900),(512,512)]
        if geometry_board:
            images.append(geometry_board);limits.append((768,512))
        return await vision.ask('review-compact',Review,COMPACT_PROMPT,payload,images,
                                image_limits=limits,max_output_tokens=1536)
    payload.update(checks=checks,estimated_size_mm=asset['semantic']['major_mm'],
        source_note='Use the current model and its intentional revisions. Assess visible structure at the actual pose.',
        profile_meanings={'clearer':'reduce roughness','softer':'soften reflection',
                          'weathered':'increase surface dust on exposed paint',
                          'defined':'stronger broad light','darker':'reduce exposure'})
    images.insert(2,render_dir/'object.png')
    if geometry_board:images.append(geometry_board)
    return await vision.ask('review',Review,prompt,payload,images)
