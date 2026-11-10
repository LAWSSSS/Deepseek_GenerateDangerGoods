"""Real Blender/Bullet regressions, distinct from the 1000-job queue simulation."""
import asyncio
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import trimesh

from display_batch.common import ROOT,atomic_json
from display_batch.worker import Blender
from display_batch.geometry import align_up,transformed


def fixture(folder,name,pieces):
    mesh=trimesh.util.concatenate(pieces)
    geometry=folder/(name+'.npz');proxy=folder/(name+'.json')
    np.savez_compressed(geometry,vertices=mesh.vertices,faces=mesh.faces)
    volume=sum(p.volume for p in pieces)
    centroid=sum(p.center_mass*p.volume for p in pieces)/volume
    atomic_json(proxy,{'parts':[{'vertices':p.vertices.tolist(),'faces':p.faces.tolist(),'volume':p.volume} for p in pieces],
        'centroid':centroid.tolist(),'volume':volume,'mass_basis':'uniform_density_known_closed_test_components'})
    return {'geometry':str(geometry),'proxy':str(proxy),'axis':[1,0,0],
            'assembly':'rigid_assembly','initial_matrices':[np.eye(4).tolist()]}


async def main():
    out=ROOT/'outputs/多背景批量-物理回归-20261110';out.mkdir(parents=True,exist_ok=True)
    worker=Blender(out/'worker','CPU');records={}
    try:
        with tempfile.TemporaryDirectory(prefix='display-physics-') as directory:
            folder=Path(directory)
            left=trimesh.creation.box(extents=[.4,.3,.2]);left.apply_translation([2.7,0,.1])
            right=trimesh.creation.box(extents=[.25,.2,.2]);right.apply_translation([3.3,0,.1])
            request=fixture(folder,'offset-assembly',[left,right])
            result=await worker.ask('settle',request)
            pose=result['poses'][0]
            assert pose['stable'],pose
            assert pose['collision_parts']==2
            with np.load(request['geometry']) as data:vertices=data['vertices'].copy()
            world=transformed(vertices,pose['matrix'])
            assert world[:,2].min()>=-.004
            # Normalizing the COM must preserve component offsets and all inter-vertex distances.
            np.testing.assert_allclose(np.ptp(world,axis=0),np.ptp(vertices,axis=0),atol=.01)
            records['offset_center_and_rigid_assembly']={'passed':True,**result}
            cylinder=trimesh.creation.cylinder(radius=.13,height=.8,sections=48)
            cylinder.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2,[0,1,0]))
            request=fixture(folder,'cylinder',[cylinder])
            flat=await worker.ask('settle',request)
            assert flat['poses'][0]['stable'],flat
            normal=[0,-math.sin(math.radians(12)),math.cos(math.radians(12))]
            request.update(plane_normal=normal,initial_matrices=[align_up(normal).tolist()])
            inclined=await worker.ask('settle',request)
            assert not inclined['poses'][0]['stable'],inclined
            records['cylinder_roll_rejected']={'passed':True,'flat':flat,'inclined':inclined}
    finally:await worker.close()
    atomic_json(out/'checks.json',{'kind':'actual_blender_bullet','device':'CPU','checks':records,
                                  'seconds':worker.seconds})
    print(json.dumps({'passed':len(records),'seconds':round(worker.seconds,3),'report':str(out/'checks.json')}))


if __name__=='__main__':asyncio.run(main())
