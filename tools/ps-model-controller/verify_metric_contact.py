"""Actual Blender metre/compound smoke checks; synthetic fixtures, not image acceptance."""
import asyncio
from pathlib import Path
import numpy as np
import trimesh
import cv2
from PIL import Image
from display_batch.common import ROOT, atomic_json
from display_batch.worker import Blender
from display_batch.strict_contact import retryable


def check_images(out):
    small=np.asarray(Image.open(out/'512/object.png').getchannel('A'))
    large=np.asarray(Image.open(out/'1024/object.png').getchannel('A'))
    assert small.shape==(512,512) and large.shape==(1024,1024)
    a=small>127;b=cv2.resize(large,(512,512),interpolation=cv2.INTER_AREA)>127
    iou=float((a&b).sum()/(a|b).sum());assert iou>=.975
    return {'alpha_iou_512_1024':iou,'threshold':.975,'png_dimensions_checked':True}


async def main():
    out=ROOT/'work/reference-projection/metric-fixtures';out.mkdir(parents=True,exist_ok=True)
    box=trimesh.creation.box(extents=[1,.5,.25]);geometry=out/'box.npz'
    np.savez_compressed(geometry,vertices=box.vertices,faces=box.faces)
    proxy=out/'collision.json'
    atomic_json(proxy,{'centroid':[0,0,0],'volume':float(box.volume),'parts':[
        {'vertices':box.vertices.tolist(),'faces':box.faces.tolist(),'volume':float(box.volume)}]})
    worker=Blender(out/'worker','OPTIX');rows=[];renders=[]
    try:
        for length in (.1,1.):
            result=await worker.ask('settle',{'geometry':str(geometry),'proxy':str(proxy),'axis':[1,0,0],
                'assembly':'single_piece','initial_matrices':[np.eye(4).tolist()],'length_m':length},timeout=120)
            pose=result['poses'][0];rows.append(result)
            assert pose['stable'],pose
            assert not pose['result_translation_adjusted']
            assert pose['max_contact_gap_m']<=pose['tolerances']['contact_m']
        # A narrow, displaced steel piece cannot support the box's centre of mass.
        # Image-space winding is clockwise: receiver construction must turn the
        # passive triangles upward, otherwise objects ABOVE it look penetrating.
        reversed_patch={'site_id':'reversed','vertices':[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],
                        'faces':[[0,2,1],[0,3,2]]}
        request={'geometry':str(geometry),'proxy':str(proxy),'axis':[1,0,0],
            'assembly':'single_piece','initial_matrices':[np.eye(4).tolist()],'length_m':.2,
            'receiver':{'patches':[reversed_patch]}}
        result=await worker.ask('settle',request,timeout=120)
        if retryable(result['poses'][0]):
            first=result
            result=await worker.ask('settle',{**request,'high_precision':True},timeout=120)
            result['first_attempt']=first;result['precision_retries']=1
        # This regression is about winding, not relaxing convergence. Bullet may
        # jitter on a zero-thickness triangle receiver; that stays rejected.
        pose=result['poses'][0]
        assert set(pose['failure_codes']) <= {'not_converged'},pose['failure_codes']
        assert pose['max_penetration_m']==0 and pose['contact_count']>0 and pose['center_supported']
        rows.append(result)
        forward={**reversed_patch,'faces':[[0,1,2],[0,2,3]]}
        comparison=await worker.ask('settle',{**request,'receiver':{'patches':[forward]},
            'high_precision':bool(result.get('precision_retries'))},timeout=120)
        other=comparison['poses'][0]
        assert pose['failure_codes']==other['failure_codes']
        np.testing.assert_allclose(pose['matrix'],other['matrix'],atol=1e-6)
        rows.append(comparison)
        patch={'site_id':'test','vertices':[[.35,-.4,0],[.6,-.4,0],[.6,.4,0],[.35,.4,0]],'faces':[[0,1,2],[0,2,3]]}
        result=await worker.ask('settle',{'geometry':str(geometry),'proxy':str(proxy),'axis':[1,0,0],
            'assembly':'single_piece','initial_matrices':[np.eye(4).tolist()],'length_m':.2,
            'receiver':{'patches':[patch]}},timeout=120)
        rows.append(result);assert not result['poses'][0]['stable']
        model=out/'fixture.glb';box.export(model)
        receiver={'patches':[{'site_id':'floor','vertices':[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],'faces':[[0,1,2],[0,2,3]]}]}
        for resolution in (512,1024):
            response=await worker.ask('render',{'mesh':str(model),'matrix':rows[0]['poses'][0]['matrix'],
                'yaw':0,'elevation':45,'contact_band':.002,'receiver':receiver,
                'environment':{},'resolution':resolution,'output':str(out/str(resolution))},timeout=180)
            renders.append(response)
        np.testing.assert_allclose(renders[0]['anchor_uv'],renders[1]['anchor_uv'],atol=1e-7)
        atomic_json(out/'validation.json',{'fixtures':rows,'renders':renders,
            'image_checks':check_images(out),
            'object_renders':2,'shadow_related_renders':4,'api_calls':0,'seconds':worker.seconds,
            'finite_receiver_not_converged':not pose['stable'],
            'finite_receiver_winding_equivalent':True,
            'synthetic_not_visual_acceptance':True,'passed':True})
        print('actual_blender_metric_fixtures_passed',flush=True)
    finally:await worker.close()


if __name__=='__main__':asyncio.run(main())
