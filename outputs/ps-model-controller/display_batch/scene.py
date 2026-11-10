import asyncio
import json
import math
import time
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from scene_v2 import Perception
from .common import WEIGHTS, atomic_json, file_hash, signature
from .schema import SceneAnswer
from .geometry import (camera_for, contact_geometry, contact_world, pose_points, project, contact_support,
                       scale_from_reference, terrain_matrix, horizontal_angle, transformed)

SCENE_VERSION='independent-scene-v3'
SCENE_PROMPT='''Analyze this particular scrap hopper photograph, independently of any other image.
Identify only the interior load region as hopper. Exclude walls, outside ground, walkways and empty air.
Protect timestamp/logo/camera text polygons. Propose up to eight spatially distributed LOW supporting surfaces:
a contiguous floor patch or one broad steel face, never a polygon bridging unrelated bars or gaps.
For each surface give an interior contact anchor, polygon, estimated slope, and a visible reference edge
ON THAT SAME SURFACE at the same depth, with two endpoints and a plausible millimeter length interval.
Prefer wide low near-horizontal supports. These are estimates, not measured dimensions.
Infer camera elevation above horizontal as a bounded interval. light_direction is screen x right/y down
towards the illuminating region, or null if insufficient evidence. haze is 0..0.35.
Do not put supports on walls. All polygons and anchors use normalized x,y 0..1. A surface can be absent.
Return concise physical evidence, not assurances of success.'''


def polygon_mask(polygons, size):
    w,h=size;mask=np.zeros((h,w),np.uint8)
    for polygon in polygons:
        p=np.rint(np.asarray(polygon)*[w-1,h-1]).astype(np.int32)
        cv2.fillPoly(mask,[p],1)
    return mask.astype(bool)


class Scenes:
    def __init__(self, root, vision):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.vision=vision;self.engine=None;self.engine_lock=asyncio.Lock()
        self.loaded={};self.locks={}

    async def prepare(self, path):
        key=file_hash(path)
        lock=self.locks.setdefault(key,asyncio.Lock())
        async with lock:return await self._prepare(path)

    async def _prepare(self, path):
        key=signature(SCENE_VERSION,file_hash(path))
        if key in self.loaded:return self.loaded[key]
        start=time.monotonic();folder=self.root/'scenes'/key;folder.mkdir(parents=True,exist_ok=True)
        with Image.open(path) as im:frame=ImageOps.exif_transpose(im).convert('RGB')
        preview=folder/'background.jpg'
        if not preview.is_file():
            thumb=frame.copy();thumb.thumbnail((1536,1024));thumb.save(preview,quality=88)
        answer=await self.vision.ask('scene',SceneAnswer,SCENE_PROMPT,{},[preview])
        atomic_json(folder/'analysis.json',answer.model_dump())
        async with self.engine_lock:
            if self.engine is None:self.engine=Perception(WEIGHTS,self.root)
            rgb,depth,masks,size=await asyncio.to_thread(self.engine.scene,Path(path))
        hopper=polygon_mask([answer.hopper],size)
        excluded=polygon_mask(answer.excluded,size);protected=polygon_mask(answer.protected,size)
        allowed=hopper & ~excluded & ~protected
        supports=[]
        for raw in answer.surfaces:
            surface=raw.model_dump();mask=polygon_mask([raw.polygon],size)&allowed
            x,y=np.rint(np.array(raw.anchor)*[size[0]-1,size[1]-1]).astype(int)
            # Confidence ranks proposals; image/physics checks decide acceptance.
            if raw.confidence<.3 or raw.slope_deg>30 or not mask[y,x]:continue
            # SAM can refine a proposal, but never grow it into another surface.
            small=cv2.resize(mask.astype(np.uint8),(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)>0
            sx,sy=round(x*(depth.shape[1]-1)/(size[0]-1)),round(y*(depth.shape[0]-1)/(size[1]-1))
            candidates=[m for m in masks if m[sy,sx] and (m&small).sum()/max(1,small.sum())>.55]
            if candidates:
                seg=min(candidates,key=lambda m:abs(int(m.sum())-int(small.sum())))
                mask &= cv2.resize(seg.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
            if mask.sum()<300:continue
            yy,xx=np.nonzero(cv2.resize(mask.astype(np.uint8),(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST))
            step=max(1,len(xx)//1200);xx=xx[::step];yy=yy[::step]
            design=np.column_stack([xx/depth.shape[1],yy/depth.shape[0],np.ones(len(xx))])
            coeff=np.linalg.lstsq(design,depth[yy,xx],rcond=None)[0]
            residual=float(np.median(abs(design@coeff-depth[yy,xx])))
            if residual>.065:continue
            direction=np.array([coeff[0],-coeff[1]/max(.2,math.sin(math.radians(sum(answer.elevation)/2)))])
            if np.linalg.norm(direction)<1e-5:direction=np.array([0.,-1.])
            direction/=np.linalg.norm(direction);angle=math.radians(raw.slope_deg)
            normal=np.r_[-direction*math.sin(angle),math.cos(angle)]
            surface.update(mask=mask,anchor_px=[int(x),int(y)],depth=float(depth[sy,sx]))
            surface.update(plane_normal=normal.tolist(),plane_residual=residual,
                           plane_basis='uncalibrated_depth_gradient_and_visual_slope_estimate')
            supports.append(surface)
        scene={'key':key,'path':str(path),'folder':str(folder),'frame':frame,'size':size,
               'depth':depth,'allowed':allowed,'protected':protected,'supports':supports,
               'answer':answer.model_dump(),'basis':'estimated','seconds':time.monotonic()-start}
        scene['depth_native']=cv2.resize(depth,size,interpolation=cv2.INTER_LINEAR)
        self.loaded[key]=scene
        while len(self.loaded)>2:self.loaded.pop(next(iter(self.loaded)))
        return scene


def projection_geometry(vertices,faces,pose,yaw,elevation,normal=(0,0,1),patches=None):
    p,_=pose_points(vertices,pose['matrix'],yaw)
    camera=camera_for(p,elevation)
    uv=cv2.convexHull(np.float32(project(p,camera))).reshape(-1,2).astype(float)
    polys,anchor=contact_geometry(p,faces,camera,normal=normal,patches=patches)
    return {'camera':camera,'uv':uv,'contacts':polys,'anchor':anchor}


def cached_projection(asset,vertices,faces,pose,yaw,elevation,normal=(0,0,1),patches=None):
    key=signature('projection-v3',asset['signature'],pose['matrix'],yaw,elevation,normal)
    folder=Path(asset['folder'])/'projections';folder.mkdir(exist_ok=True)
    path=folder/(key+'.npz')
    if path.is_file():
        with np.load(path) as data:
            points=data['points'];offsets=data['offsets'];anchor=data['anchor']
            return {'camera':json.loads(str(data['camera'])),'uv':data['uv'],
                    'contacts':[points[a:b] for a,b in zip(offsets[:-1],offsets[1:])],
                    'anchor':anchor if len(anchor) else None}
    geom=projection_geometry(vertices,faces,pose,yaw,elevation,normal,patches)
    offsets=np.cumsum([0]+[len(p) for p in geom['contacts']])
    fd,name=tempfile.mkstemp(dir=folder,suffix='.npz');os.close(fd)
    try:
        np.savez_compressed(name,uv=geom['uv'],points=np.concatenate(geom['contacts']) if geom['contacts'] else np.empty((0,2)),
                            offsets=offsets,anchor=geom['anchor'] if geom['anchor'] is not None else [],
                            camera=json.dumps(geom['camera']))
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
    return geom


def fitted_masks(geom, anchor_px, full_px, scene):
    if geom['anchor'] is None:return None
    offset=np.asarray(anchor_px)-np.asarray(geom['anchor'])*full_px
    uv=geom['uv'];xy=uv*full_px+offset
    low=np.floor(xy.min(axis=0)-4).astype(int);high=np.ceil(xy.max(axis=0)+4).astype(int)
    w,h=scene['size']
    if min(low)<0 or high[0]>=w or high[1]>=h:return None
    x,y=low;cw,ch=high-low
    if min(cw,ch)<12 or max(cw,ch)>w*.6:return None
    contact=np.zeros((ch,cw),np.uint8)
    for poly in geom['contacts']:
        points=np.rint((poly*full_px+offset-low)*16).astype(np.int32)
        cv2.fillConvexPoly(contact,points,255,shift=4)
    hull=cv2.convexHull(np.float32(xy-low))
    silhouette=np.zeros((ch,cw),np.uint8);cv2.fillConvexPoly(silhouette,hull.astype(np.int32),1)
    return {'origin':[int(x),int(y)],'bounds':[int(x),int(y),int(high[0]),int(high[1])],
            'offset':offset.tolist(),'contact':contact,'hull':silhouette>0}


def fit_checks(fit, surface, scene):
    x,y,r,b=fit['bounds'];support=surface['mask'][y:b,x:r]
    cover=contact_support(fit['contact'],support)
    protected=scene['protected'][y:b,x:r]
    allowed=scene['allowed'][y:b,x:r]
    hull=fit['hull']
    depth=scene['depth_native'][y:b,x:r]
    front=(depth>surface['depth']+.10)&~support
    return cover, not np.any(protected&hull),float((allowed&hull).sum()/max(1,hull.sum())),\
        float((front&hull).sum()/max(1,hull.sum()))


def support_anchors(site):
    """A few interior maxima, derived from the surface mask rather than image coordinates."""
    mask=site['mask'];h,w=mask.shape
    factor=min(1.,640/max(w,h))
    small=cv2.resize(mask.astype(np.uint8),None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
    distance=cv2.distanceTransform(small,cv2.DIST_L2,5)
    anchors=[site['anchor_px']]
    radius=max(8,round(math.sqrt(float(small.sum()))*.22))
    for _ in range(2):
        _,peak,_,point=cv2.minMaxLoc(distance)
        if peak<2:break
        anchor=[min(w-1,round(point[0]/factor)),min(h-1,round(point[1]/factor))]
        if mask[anchor[1],anchor[0]] and all(np.linalg.norm(np.subtract(anchor,a))*factor>radius*.5 for a in anchors):
            anchors.append(anchor)
        cv2.circle(distance,point,radius,0,-1)
    return anchors


def choose_candidates(asset,scene,seed=0):
    data=np.load(asset['geometry']);vertices=data['vertices'];faces=data['faces']
    elevations=sorted(set([*scene['answer']['elevation'],sum(scene['answer']['elevation'])/2]))
    midpoint=sum(scene['answer']['elevation'])/2
    records=[];rejections=[];geometry_cache={}
    flat_contacts={p['id']:contact_world(transformed(vertices,p['matrix']),faces) for p in asset['poses']}
    rng=np.random.default_rng(seed)
    # First choose distributed sites; only then search the cached horizontal poses and yaws.
    surfaces=sorted(scene['supports'],key=lambda s:(s['slope_deg'],-s['confidence']))[:8]
    for site in surfaces:
        anchors=[];interval=None
        for anchor in support_anchors(site):
            try:
                interval=scale_from_reference(site,asset['semantic']['major_mm'],*scene['size'],scene['depth'],anchor)
                anchors.append(anchor)
            except ValueError as error:
                rejections.append({'site':site['id'],'anchor':anchor,'reason':str(error)})
        if not anchors:continue
        a,b=np.asarray(site['reference'])*[scene['size'][0]-1,scene['size'][1]-1]
        delta=b-a;plane_correction=np.linalg.norm([delta[0],delta[1]/math.sin(math.radians(midpoint))])/max(1,np.linalg.norm(delta))
        scales=[float(np.sqrt(interval[0]*interval[1])),float(interval[0]*1.08),float(interval[1]/1.08)]
        for pose in asset['poses']:
            for yaw in range(0,360,30):
                normal=site.get('plane_normal',[0,0,1])
                matrix=terrain_matrix(pose['matrix'],yaw,normal)
                if horizontal_angle(asset['axis'],matrix)>15:continue
                local_pose={**pose,'matrix':matrix.tolist()}
                rotation=terrain_matrix(np.eye(4),yaw,normal)
                patches=[transformed(q,rotation) for q in flat_contacts[pose['id']]]
                geometries=[]
                for elevation in elevations:
                    key=(site['id'],pose['id'],yaw,elevation)
                    if key not in geometry_cache:
                        geometry_cache[key]=cached_projection(asset,vertices,faces,local_pose,0,elevation,normal,patches)
                    geometries.append((elevation,geometry_cache[key]))
                for scale_index,scale in enumerate(scales):
                    if not interval[0]<=scale<=interval[1]:continue
                    world_ppm=scale*plane_correction
                    selected=None
                    for anchor_index,anchor in enumerate(anchors):
                        fits=[];valid=True;minimum=1.;coverage=1.;front=0.
                        for elevation,geom in geometries:
                            distance=np.linalg.norm(np.array(geom['camera']['eye'])-geom['camera']['target'])
                            full_px=world_ppm*2*distance*math.tan(math.radians(geom['camera']['fov']/2))
                            fit=fitted_masks(geom,anchor,full_px,scene)
                            if fit is None:valid=False;break
                            covers,protected_ok,inside,occluded=fit_checks(fit,site,scene)
                            if not covers or min(covers)<.9 or not protected_ok or inside<.98 or occluded>.30:
                                valid=False;break
                            minimum=min(minimum,min(covers));coverage=min(coverage,inside);front=max(front,occluded)
                            fits.append((elevation,fit,full_px,geom))
                        if valid:
                            selected=anchor_index;break
                    if selected is None:continue
                    elevation,fit,full_px,geom=next(item for item in fits if item[0]==midpoint)
                    x,y,r,b=fit['bounds']
                    if max(r-x,b-y)<64:continue
                    ident=f"{site['id']}-{pose['id']}-Y{yaw}-Z{scale_index}-A{selected}"
                    records.append({'id':ident,'site_id':site['id'],'pose_id':pose['id'],'matrix':matrix.tolist(),
                    'yaw':0,'horizontal_yaw_deg':yaw,'plane_normal':normal,'plane_basis':site.get('plane_basis'),
                    'elevation':midpoint,'elevation_interval':scene['answer']['elevation'],
                    'anchor_px':anchor,'full_px':full_px,'target_px':max(r-x,b-y),
                    'expected_major_px':float(np.ptp(geom['uv'],axis=0).max()*full_px),
                    'axis_angle_deg':horizontal_angle(asset['axis'],matrix),
                    'offset':fit['offset'],'bounds':fit['bounds'],'anchor_uv':geom['anchor'].tolist(),
                    'contact_min':minimum,'inside':coverage,'foreground_fraction':front,
                    'scale':{'basis':'estimated_local_same_surface','major_mm':asset['semantic']['major_mm'],
                             'reference_mm':site['reference_mm'],'reference':site['reference'],
                             'projected_reference_ratio_range':interval,'world_pixels_per_unit':world_ppm},
                    'score':site['confidence']+minimum-front-pose['height']*.15-(.12 if scale_index else 0)+float(rng.uniform(0,.04))})
        if not any(c['site_id']==site['id'] for c in records):
            rejections.append({'site':site['id'],'reason':'support_camera_clearance_or_scale_rejected'})
    # Spread early choices over distinct supports when available.
    chosen=[]
    for candidate in sorted(records,key=lambda c:-c['score']):
        if not any(c['site_id']==candidate['site_id'] for c in chosen):chosen.append(candidate)
        if len(chosen)==3:break
    for candidate in sorted(records,key=lambda c:-c['score']):
        if candidate not in chosen:chosen.append(candidate)
        if len(chosen)==3:break
    return chosen,rejections
