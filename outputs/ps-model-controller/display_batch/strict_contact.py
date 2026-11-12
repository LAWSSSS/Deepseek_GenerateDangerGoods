"""Strict eligibility is independent of display-review uncertainty policy."""
import json
import math
import re
import time
from pathlib import Path
import numpy as np
from .common import signature, atomic_json, file_hash
from .metric_physics import VERSION, tolerances
from .geometry import project


def retryable(pose):
    return set(pose.get('failure_codes',[]))=={'not_converged'}


def visual_veto(review):
    return [d.model_dump() for d in review.defects if d.code in ('floating','penetration','occlusion','upright')
            and d.visibility=='visible']


def pose_review_context(checks):
    """A geometric measurement informs interpretation, not visual acceptance."""
    lying=checks.get('lying',{})
    angle=lying.get('measurement_deg')
    measured=isinstance(angle,(int,float)) and not isinstance(angle,bool) and math.isfinite(angle)
    return {'required_pose':'lying_on_side', 'reference_pose_is_not_target':True,
            'measured_axis_angle_deg':angle if measured else None,
            'maximum_axis_angle_deg':15, 'measurement_status':lying.get('status','unknown'),
            'limitation':'Estimated geometry does not override visible floating, penetration or depth-order defects.'}


def reconcile_pose_evidence(review,checks):
    """Keep a contradictory reference-pose complaint unknown, never accepted.

    This narrowly handles an observed invalid review: the model says the object
    lies on its side, but calls that an upright defect because the source photo
    is upright. A plausible upright cue, or any contact defect, still vetoes.
    """
    context=pose_review_context(checks)
    angle=context['measured_axis_angle_deg']
    if context['measurement_status']!='pass' or angle is None or not 0<=angle<=15:
        return review
    reconciled=[];defects=[]
    for defect in review.defects:
        evidence=defect.evidence.lower()
        # Both an explicit current lying pose and an upright *reference* are
        # required; a measured low axis angle alone can never clear an AI veto.
        lying=bool(re.search(r'横向侧躺|横躺|侧卧|平躺|lies? (?:flat|on (?:its|the) side)|lying (?:flat|on (?:its|the) side)|horizontally (?:placed|lying)',evidence))
        denied_lying=bool(re.search(r'(?:并非|不是|没有|未|不|应该|应当|必须|需要)[^，。；,;.!?]{0,8}(?:横向侧躺|横躺|侧卧|平躺)|(?:not|should be|must be)[^.;!?]{0,12}(?:lying|horizontally|flat)',evidence))
        reference=bool(re.search(r'(?:参考图|源图|原图)[^。；;.!?]{0,32}(?:直立|竖立)|(?:reference|source)(?: image| photo)?[^.;!?]{0,48}(?:upright|standing|vertical)',evidence))
        contact_cue=bool(re.search(r'悬空|穿透|浮起|floating|penetrat|hovering|unsupported',evidence))
        if defect.code=='upright' and defect.visibility=='visible' and lying and reference and not denied_lying and not contact_cue:
            reconciled.append(defect.model_dump())
            defects.append(defect.model_copy(update={'visibility':'unassessable',
                'evidence':('参考姿态误用，与已测横躺姿态矛盾；原评语：'+defect.evidence)[:160]}))
        else:defects.append(defect)
    if not reconciled:return review
    checks['review_pose_consistency']={'status':'unknown','reason':'reference_pose_conflicts_with_required_lying_pose',
        'measurement_deg':angle,'threshold_deg':15,'original_defects':reconciled,
        'policy':'retain_review_only; preserve severity and all independent physical vetoes'}
    # Do not raise scores, erase severity or set trace=none. A contradictory
    # assessment is an unresolved review, not evidence of physical success.
    return review.model_copy(update={'defects':defects})


async def simulate(asset,blender,matrices,receiver=None,normal=(0,0,1)):
    profile=asset['physical_size'];length=profile['nominal_mm']/1000
    key=signature(VERSION,asset['physics_signature'],profile['signature'],matrices,receiver,normal,
                  file_hash(Path(__file__).with_name('metric_physics.py')),
                  file_hash(Path(__file__).with_name('contact_audit.py')),
                  file_hash(Path(__file__).parents[1]/'blender_display_worker.py'))
    path=Path(asset['folder'])/'metric-physics'/(key+'.json')
    if path.is_file():return {**json.loads(path.read_text(encoding='utf8')),'cache_hit':True}
    request={'geometry':asset['geometry'],'proxy':asset.get('proxy',str(Path(asset['folder'])/'collision.json')),
             'axis':asset['axis'],'assembly':asset['semantic']['assembly'],'initial_matrices':matrices,
             'plane_normal':list(normal),'length_m':length}
    if receiver:request['receiver']=receiver
    started=time.monotonic();result=await blender.ask('settle',request,timeout=600)
    for i,pose in enumerate(result['poses']):
        if retryable(pose):
            retry=await blender.ask('settle',{**request,'initial_matrices':[matrices[i]],'high_precision':True},
                                    timeout=max(1,600-(time.monotonic()-started)))
            result['poses'][i]={**retry['poses'][0],'id':pose['id'],'first_attempt':pose,'precision_retries':1}
    result.update(signature=key,cache_hit=False,seconds=time.monotonic()-started)
    atomic_json(path,result);return result


def validate_receiver(candidate):
    receiver=candidate['receiver']
    if not receiver.get('patches'):raise ValueError('receiver_geometry_insufficient')
    if len(candidate['support_ids'])>1 and receiver.get('height_confidence')!='measured':
        raise ValueError('separate_support_heights_unresolved')


def projected_gap(pose,candidate):
    p=np.asarray(pose.get('contact_points',[]));q=np.asarray(pose.get('receiver_points',[]))
    if not len(p):return {'status':'fail','reason':'no_true_mesh_contact'}
    pixels=np.linalg.norm(project(p,candidate['camera'])-project(q,candidate['camera']),axis=1)*candidate['full_px']
    return {'status':'pass' if float(pixels.max())<=2 else 'fail', 'maximum_px':float(pixels.max()),
            'threshold_px':2.,'basis':'true_mesh_to_receiver_projected_separation',
            'scene_geometry_status':'estimated'}
