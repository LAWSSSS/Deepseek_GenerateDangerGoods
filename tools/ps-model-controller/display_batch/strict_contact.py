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
from scrap_composite.photo_evidence import contradiction, physical_evidence_issue


def retryable(pose):
    return set(pose.get('failure_codes',[]))=={'not_converged'}


def review_evidence_issue(defect):
    """Audit legacy assertions without converting an invalid finding to a pass."""
    pieces=re.split(r'(?:[。，；,;]\s*|\bbut the )(?=(?:参考图|识别参考|源图|原图|图[234]|reference|source))',
                    defect.evidence,maxsplit=1,flags=re.I)
    cue=pieces[0];reference=pieces[1] if len(pieces)>1 else ''
    factor,direction={'too_sharp':('sharpness','decrease'),'too_blurry':('sharpness','increase'),
        'haze_clear':('haze_amount','increase'),'haze_heavy':('haze_amount','decrease'),
        'identity_loss':('identity','unknown')}.get(defect.code,(defect.code,'unknown'))
    return (contradiction(factor,direction,cue,reference) or
        physical_evidence_issue(defect.code,cue,reference,getattr(defect,'physical_cue',None),
                                getattr(defect,'reference_role',None)))


def visual_veto(review):
    return [d.model_dump() for d in review.defects if d.code in ('floating','penetration','occlusion','upright')
            and d.visibility=='visible' and review_evidence_issue(d) is None]


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
    unresolved=[];safe=[]
    for defect in review.defects:
        issue=review_evidence_issue(defect) if defect.visibility=='visible' else None
        if issue:
            unresolved.append({'reason':issue,'original_defect':defect.model_dump()})
            safe.append(defect.model_copy(update={'visibility':'unassessable','physical_cue':None}))
        else:safe.append(defect)
    if unresolved:
        checks['review_evidence_consistency']={'status':'unknown','reason':'unsupported_or_contradictory_review',
            'observations':unresolved,'policy':'preserve_scores_and_severity; no_automatic_acceptance'}
        review=review.model_copy(update={'defects':safe})
        pose_errors=[item['original_defect'] for item in unresolved
                     if item['reason']=='reference_pose_does_not_establish_standing']
        if pose_errors:
            checks['review_pose_consistency']={'status':'unknown',
                'reason':'reference_pose_conflicts_with_required_lying_pose',
                'original_defects':pose_errors,'policy':'preserve_scores_and_severity; no_automatic_acceptance'}
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
    if candidate.get('integration_mode')=='coherent':
        if receiver.get('contact_evidence_version')!='visible-support-v1':
            raise ValueError('visible_receiver_evidence_missing_regenerate_candidate')
        if any(e.get('basis')=='nonplanar_label' for e in receiver.get('visible_support',[])):
            raise ValueError('nonplanar_receiver_requires_independent_geometry')


def visible_contact_coverage(points,candidate):
    """Test contact clusters against finite receivers tied to actual visible pixels."""
    import cv2
    x,y,r,b=candidate['bounds'];height,width=b-y,r-x
    if height<=0 or width<=0:return {'status':'fail','reason':'empty_contact_bounds'}
    mask=np.zeros((height,width),np.uint8);camera=candidate['camera'];full=candidate['full_px']
    offset=np.asarray(candidate['offset'])-[x,y]
    for patch in candidate['receiver']['patches']:
        projected=project(np.asarray(patch['vertices']),camera)*full+offset
        for face in patch['faces']:
            cv2.fillPoly(mask,[np.rint(projected[face]*16).astype(np.int32)],1,shift=4)
    uv=project(points,camera)*full+offset
    coordinates=np.rint(uv).astype(int)
    inside=(coordinates[:,0]>=0)&(coordinates[:,0]<width)&(coordinates[:,1]>=0)&(coordinates[:,1]<height)
    valid=np.zeros(len(points),bool)
    valid[inside]=mask[coordinates[inside,1],coordinates[inside,0]]>0
    cloud=np.zeros_like(mask)
    cloud[coordinates[inside,1],coordinates[inside,0]]=1
    _,labels=cv2.connectedComponents(cv2.dilate(cloud,np.ones((5,5),np.uint8)))
    groups=np.zeros(len(points),int);groups[inside]=labels[coordinates[inside,1],coordinates[inside,0]]
    coverages=[float(valid[groups==g].mean()) for g in sorted(set(groups))]
    passed=bool(coverages and min(coverages)>=.9)
    return {'status':'pass' if passed else 'fail','cluster_coverages':coverages,'threshold':.9,
        'contact_points':len(points),'outside_visible_receiver_points':int((~valid).sum()),
        'basis':'projected_true_mesh_contacts_on_depth_consistent_visible_receiver_pixels',
        'proves_actual_height':False}


def projected_gap(pose,candidate):
    p=np.asarray(pose.get('contact_points',[]));q=np.asarray(pose.get('receiver_points',[]))
    if not len(p):return {'status':'fail','reason':'no_true_mesh_contact'}
    pixels=np.linalg.norm(project(p,candidate['camera'])-project(q,candidate['camera']),axis=1)*candidate['full_px']
    result={'status':'pass' if float(pixels.max())<=2 else 'fail', 'maximum_px':float(pixels.max()),
            'threshold_px':2.,'basis':'true_mesh_to_receiver_projected_separation',
            'scene_geometry_status':'estimated'}
    if candidate.get('integration_mode')=='coherent':
        coverage=visible_contact_coverage(p,candidate)
        simulation_status=result['status']
        measured=candidate['receiver'].get('scene_contact_status')=='pass'
        status='fail' if simulation_status=='fail' or coverage['status']=='fail' else 'pass' if measured else 'unknown'
        result.update(status=status,simulation_status=simulation_status,visible_support=coverage,
            scene_geometry_status='measured' if measured else 'estimated',
            reason='measured_surface_and_contact' if measured else 'contact_to_virtual_receiver_only; actual_height_unverified',
            contact_evidence_version='visible-support-v1')
    return result
