"""Strict eligibility is independent of display-review uncertainty policy."""
import json
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


async def simulate(asset,blender,matrices,receiver=None,normal=(0,0,1)):
    profile=asset['physical_size'];length=profile['nominal_mm']/1000
    key=signature(VERSION,asset['physics_signature'],profile['signature'],matrices,receiver,normal,
                  file_hash(Path(__file__).with_name('metric_physics.py')))
    path=Path(asset['folder'])/'metric-physics'/(key+'.json')
    if path.is_file():return {**json.loads(path.read_text(encoding='utf8')),'cache_hit':True}
    request={'geometry':asset['geometry'],'proxy':str(Path(asset['folder'])/'collision.json'),
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
