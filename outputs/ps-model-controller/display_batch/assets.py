import asyncio
import importlib.metadata
import json
import math
import time
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import trimesh

from .common import atomic_json, signature, file_hash
from .schema import AssetAnswer

ASSET_VERSION='asset-v1'
PHYSICS_VERSION='compound-lying-v1'
ASSET_PROMPT='''Inspect the actual current single-object model/reference board and source photo.
Return a plausible real-world longest dimension interval in millimeters, never infer meters from GLB coordinates.
Describe key visible structures, material classes and whether components form one rigid assembly, a single piece,
or genuinely loose objects. Decorative labels/bolts do not make an assembly loose. Do not reintroduce removed parts.
upright_axis refers to supplied normalized mesh extents only; use none when not inferable.
These are display estimates, not measured dimensions. Keep evidence concise.'''
ASSET_PROMPT += ''' The current model name and CURRENT model board take precedence over older source photos
for intentional revisions. A closed opaque housing must not retain an old cutaway/observation window.
Use source photos for material and size evidence, not to undo a documented current revision.'''


def make_proxy(geometry, path):
    import coacd
    coacd.set_log_level('error')
    data=np.load(geometry)
    mesh=trimesh.Trimesh(data['vertices'],data['faces'],process=True)
    mesh.remove_unreferenced_vertices()
    pieces=mesh.split(only_watertight=False)
    closed=[p for p in pieces if p.is_watertight and abs(p.volume)>1e-10]
    total_area=sum(p.area for p in pieces)
    closed_fraction=sum(p.area for p in closed)/max(total_area,1e-9)
    if not closed or closed_fraction<.80:
        raise ValueError('insufficient_closed_volume_for_mass')
    volume=sum(abs(p.volume) for p in closed)
    centroid=sum(p.center_mass*abs(p.volume) for p in closed)/volume
    # Work on collision-only geometry. UVs, render topology and openings are untouched.
    source=trimesh.util.concatenate(closed)
    hulls=coacd.run_coacd(coacd.Mesh(source.vertices,source.faces),threshold=.02,
                          max_convex_hull=64,preprocess_mode='auto',resolution=1500,
                          mcts_nodes=10,mcts_iterations=60,mcts_max_depth=3,
                          merge=True,decimate=True,max_ch_vertex=64,seed=17)
    parts=[]
    for v,f in hulls:
        hull=trimesh.Trimesh(v,f,process=False)
        parts.append({'vertices':np.round(v,8).tolist(),'faces':f.tolist(),
                      'volume':float(abs(hull.volume))})
    if not parts:raise ValueError('no_collision_parts')
    result={'parts':parts,'centroid':centroid.tolist(),'volume':float(volume),
            'closed_surface_fraction':float(closed_fraction),
            'mass_basis':'uniform_density_closed_components; overlaps are estimated',
            'coacd_version':importlib.metadata.version('coacd'),
            'geometry_hash':file_hash(geometry)}
    atomic_json(path,result)
    return result


async def bounded_proxy(geometry,path):
    from .common import HERE
    log=Path(path).with_suffix('.log')
    with log.open('w',encoding='utf8') as output:
        proc=await asyncio.create_subprocess_exec(sys.executable,'-X','utf8',str(HERE/'display_collision_worker.py'),
                str(geometry),str(path),stdout=output,stderr=asyncio.subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            code=await asyncio.wait_for(proc.wait(),timeout=240)
        except asyncio.TimeoutError:
            proc.kill();await proc.wait()
            raise TimeoutError('collision_preparation_exceeded_240_seconds') from None
    if code or not Path(path).is_file():raise ValueError('collision_preparation_failed; inspect collision.log')


def initial_poses(vertices):
    points=vertices[::max(1,len(vertices)//30000)]
    _,vectors=np.linalg.eigh(np.cov(points.T))
    long=vectors[:,-1]
    # Deterministic PCA signs and right-handed frame; roll explores different resting sides.
    if long[np.argmax(abs(long))]<0:long=-long
    other=vectors[:,1];normal=np.cross(long,other);other=np.cross(normal,long)
    basis=np.stack([long,other,normal])
    matrices=[]
    for roll in (0,60,120,180,240,300):
        a=math.radians(roll)
        r=np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
        m=np.eye(4);m[:3,:3]=r@basis;matrices.append(m.tolist())
    return long.tolist(),matrices


async def prepare(row, root, blender, vision):
    start=time.monotonic()
    key=signature(ASSET_VERSION,row['mesh_hash'],file_hash(row['board']),
                  file_hash(row['original']) if row.get('original') else None)
    folder=Path(root)/'assets'/key;folder.mkdir(parents=True,exist_ok=True)
    export_path=folder/'export.json';geometry=folder/'geometry.npz'
    if export_path.is_file() and geometry.is_file():
        exported=json.loads(export_path.read_text(encoding='utf8'))
    else:
        exported=await blender.ask('export',{'mesh':row['mesh'],'geometry':str(geometry)})
        atomic_json(export_path,exported)
    previews=[]
    for i,path in enumerate([row['board'],row.get('original')]):
        if not path:continue
        out=folder/f'reference-{i}.jpg'
        if not out.is_file():
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1280,1000));im.save(out,quality=85)
        previews.append(out)
    semantic=await vision.ask('asset',AssetAnswer,ASSET_PROMPT,
                              {'name':row['name'],'extents':exported['extents'],
                               'material_names':[m['name'] for m in exported['materials']]},previews)
    atomic_json(folder/'semantic.json',semantic.model_dump())
    result={**row,'folder':str(folder),'geometry':str(geometry),'inspection':exported,
            'semantic':semantic.model_dump(),'signature':key,'poses':[]}
    if semantic.assembly in ('loose_parts','unknown'):
        result['reason']='assembly_relationship_unknown_or_loose'
        atomic_json(folder/'result.json',result);return result
    proxy=folder/'collision.json'
    if not proxy.is_file():
        await bounded_proxy(geometry,proxy)
    data=np.load(geometry)
    axis,initial=initial_poses(data['vertices'])
    pose_key=signature(PHYSICS_VERSION,file_hash(proxy),file_hash(geometry),axis,initial,
                       'blender-4.5.3',semantic.assembly)
    pose_file=folder/f'poses-{pose_key[:16]}.json'
    hit=pose_file.is_file()
    if hit:physics=json.loads(pose_file.read_text(encoding='utf8'))
    else:
        physics=await blender.ask('settle',{'geometry':str(geometry),'proxy':str(proxy),
                    'axis':axis,'initial_matrices':initial,'assembly':semantic.assembly},
                    timeout=max(1,600-(time.monotonic()-start)))
        atomic_json(pose_file,physics)
    poses=[p for p in physics['poses'] if p['stable']]
    # Compact bodies must use their low, broad side. A tall stable end is not a fallback.
    ext=sorted(exported['extents'])
    if poses and ext[-1]/max(ext[-2],1e-6)<1.5:
        low=min(p['height'] for p in poses);poses=[p for p in poses if p['height']<=low*1.15]
    unique=[]
    for p in sorted(poses,key=lambda p:p['height']):
        if not any(np.linalg.norm(np.array(p['matrix'])[:3,:3]-np.array(q['matrix'])[:3,:3])<.05 for q in unique):
            unique.append(p)
    result.update(poses=unique,axis=axis,physics=physics,physics_cache_hit=hit,
                  physics_signature=pose_key,preparation_seconds=round(time.monotonic()-start,3),
                  reason=None if unique else 'no_stable_lying_pose')
    atomic_json(folder/'result.json',result)
    return result
