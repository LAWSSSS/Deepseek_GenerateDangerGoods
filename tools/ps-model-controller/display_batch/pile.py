"""Local piecewise support and depth ordering. All uncalibrated evidence stays explicit."""
import math
import json
import re
import threading
from pathlib import Path
import cv2
import numpy as np
from numba import njit
from .common import signature
from .geometry import project,contact_support,scale_from_reference

VERSION='piecewise-pile-v4-vehicle-projection'
_grab_lock=threading.Lock()


def support_plane_evidence(site):
    """A single segmentation mask is not proof of one physical plane."""
    measured=site.get('plane_status')=='measured' and site.get('plane_origin_m') is not None
    label=str(site.get('label','')).lower()
    kind=site.get('geometry_kind','unknown')
    nonplanar=kind in ('curved','compound','cylindrical') or bool(re.search(r'曲面|弯曲|卷曲|卷板|弯板|圆管|圆柱面|堆叠|叠放|叠层|多层|钢筋堆|curved|curled|cylindrical|round pipe|stacked|pile of|overlapping',label))
    inherited=bool(str(site.get('id','')).startswith('D') or site.get('normal_inherited'))
    return {'eligible':bool(measured or not nonplanar),'geometry_kind':kind,
            'basis':'measured_planes' if measured else 'nonplanar_label' if nonplanar else
                    'inherited_neighbour_normal' if inherited else 'independent_visible_surface_estimate',
            'penalty':0. if measured else .45 if inherited else .08,
            'status':'pass' if measured else 'unknown'}


def shape_priority(vertices,axis):
    """Geometry only: estimate round transverse section and elongation once per asset."""
    v=np.asarray(vertices,float);axis=np.asarray(axis,float);axis/=max(np.linalg.norm(axis),1e-12)
    centered=v-v.mean(0);axial=centered@axis
    radial=centered-axial[:,None]*axis
    _,singular,basis=np.linalg.svd(radial,full_matrices=False)
    spread=np.ptp(radial@basis[:2].T,axis=0)
    ratio=float(max(spread)/max(min(spread),1e-8))
    radii=np.linalg.norm(radial,axis=1)
    outer=np.percentile(radii,[50,90])
    roundness=float(np.clip((1.55-ratio)/.40,0,1)*np.clip((1.6-outer[1]/max(outer[0],1e-8))/.40,0,1))
    return {'elongation':float(np.ptp(axial)/max(spread.max(),1e-8)),'roundness':roundness}


def stability_priority(vertices,matrix,normal,axis,centroid,shape,band=.002):
    """Static gravity hints order costly simulations; they never certify stability."""
    from .geometry import transformed
    from .contact_audit import hull2
    points=transformed(vertices,matrix);center=transformed(np.asarray(centroid)[None],matrix)[0]
    n=np.asarray(normal,float);n/=max(np.linalg.norm(n),1e-12)
    heights=points@n;contact=points[heights<=heights.min()+band]
    hull=hull2(contact[:,:2]);outside=0.
    if len(hull)>=3:
        edge=np.roll(hull,-1,axis=0)-hull
        distance=(edge[:,0]*(center[1]-hull[:,1])-edge[:,1]*(center[0]-hull[:,0]))/np.maximum(np.linalg.norm(edge,axis=1),1e-12)
        outside=max(0.,-float(distance.min()))
    elif len(hull)==2:
        edge=hull[1]-hull[0];u=np.clip(np.dot(center[:2]-hull[0],edge)/max(np.dot(edge,edge),1e-12),0,1)
        outside=float(np.linalg.norm(center[:2]-hull[0]-u*edge))
    else:outside=float(np.linalg.norm(center[:2]-hull[0])) if len(hull) else 1.
    world_axis=np.asarray(matrix)[:3,:3]@np.asarray(axis)
    world_axis-=np.dot(world_axis,n)*n;world_axis/=max(np.linalg.norm(world_axis),1e-12)
    gravity=np.array([0.,0.,-1.]);down=gravity-n*np.dot(gravity,n)
    rolling=float(np.linalg.norm(np.cross(down,world_axis))*shape['roundness'])
    slope=math.degrees(math.acos(np.clip(n[2],-1,1)))
    sensitivity=1. if shape['elongation']>1.5 or shape['roundness']>.4 else .5
    penalty=2.2*rolling+.015*slope*sensitivity+min(.8,outside*5)
    return {'penalty':float(penalty),'gravity_outside_contact_envelope':outside,
            'slope_deg':slope,'cross_axis_rolling_risk':rolling,'shape':shape,
            'status':'unknown','basis':'static_ordering_only; finite_support_simulation_still_required'}


def diverse_candidates(rows,limit):
    """Visit distinct supporting surfaces before alternate yaw on the same surface."""
    ranked=sorted(rows,key=lambda r:-r['score']);seen=set();first=[];remaining=[]
    for row in ranked:
        key=tuple(sorted(row.get('support_ids',[row['site_id']])))
        if key in seen:remaining.append(row)
        else:seen.add(key);first.append(row)
    return (first+remaining)[:limit]


@njit(cache=True)
def zbuffer(xy,z,faces,size):
    output=np.full((size,size),np.inf,np.float32)
    for face in faces:
        a,b,c=xy[face[0]],xy[face[1]],xy[face[2]]
        za,zb,zc=z[face[0]],z[face[1]],z[face[2]]
        if min(za,zb,zc)<=.001:continue
        denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denom)<1e-10:continue
        xmin=max(0,int(math.floor(min(a[0],b[0],c[0]))));xmax=min(size-1,int(math.ceil(max(a[0],b[0],c[0]))))
        ymin=max(0,int(math.floor(min(a[1],b[1],c[1]))));ymax=min(size-1,int(math.ceil(max(a[1],b[1],c[1]))))
        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/denom
                v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/denom
                w=1-u-v
                if min(u,v,w)<-1e-7:continue
                d=1/(u/za+v/zb+w/zc)
                if d<output[y,x]:output[y,x]=d
    return output


def mesh_raster(points,faces,camera,size=384,origin=(0,0),span=1.):
    xy=(project(points,camera)-origin)/span*size
    z=(points-np.asarray(camera['eye']))@np.asarray(camera['basis'])[2]
    depth=zbuffer(xy,z,np.asarray(faces,np.int32),size)
    return np.isfinite(depth),np.where(np.isfinite(depth),depth,0)


def world_on_plane(pixels,camera,offset,full_px,normal=(0,0,1),origin=(0,0,0)):
    uv=(np.asarray(pixels)-offset)/full_px
    scale=2*math.tan(math.radians(camera['fov'])/2)
    coords=np.column_stack([(uv[:,0]-.5)*scale,(.5-uv[:,1])*scale,np.ones(len(uv))])
    direction=coords@np.asarray(camera['basis'])
    eye=np.asarray(camera['eye']);normal=np.asarray(normal);origin=np.asarray(origin)
    denominator=direction@normal
    denominator=np.where(abs(denominator)<1e-7,-1e-7,denominator)
    distance=((origin-eye)@normal)/denominator
    return eye+direction*distance[:,None],distance


def support_union(scene,ids,bounds=None):
    masks=[s['mask'] for s in scene['supports'] if s['id'] in ids]
    if bounds:
        x,y,r,b=bounds;masks=[m[y:b,x:r] for m in masks]
    return np.logical_or.reduce(masks)


def depth_order(scene,candidate,bounds,object_z,alpha,*,refine=False):
    """Fit relative inverse-depth to the local support; never equate neural depth with meters."""
    x,y,r,b=bounds;h,w=alpha.shape
    support=support_union(scene,candidate.get('support_ids',[candidate['site_id']]),bounds)
    site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
    depth=scene['depth_native'][y:b,x:r]
    yy,xx=np.mgrid[y:b,x:r];pixels=np.column_stack([xx.ravel()+.5,yy.ravel()+.5])
    camera=candidate['camera'];offset=candidate['offset'];full_px=candidate['full_px']
    _,ground=world_on_plane(pixels,camera,offset,full_px,candidate.get('plane_normal',[0,0,1]))
    ground=ground.reshape(h,w)
    valid=support&(ground>0)&np.isfinite(ground)
    sample=np.flatnonzero(valid)[::max(1,int(valid.sum()/500))]
    confidence='unknown';residual=.035
    distance=float(np.linalg.norm(np.subtract(camera['eye'],camera['target'])))
    a=distance*.45;b0=site['depth']-a/max(distance,1e-5)
    if len(sample)>30:
        inv=1/ground.ravel()[sample];values=depth.ravel()[sample]
        fit=np.polyfit(inv,values,1) if np.ptp(inv)>1e-5 else np.array([a,b0])
        error=float(np.median(abs(values-np.polyval(fit,inv))))
        if distance*.05<fit[0]<distance*6 and error<.04:
            a,b0=fit;confidence='estimated';residual=max(.012,error*2)
        else:
            b0=float(np.median(values-a*inv))
    actual=np.where(object_z>0,object_z,ground)
    object_depth=a/np.maximum(actual,1e-6)+b0
    # A supporting piece can rise in front of another part of the object. Membership
    # in the receiver set is not a blanket exemption from pixel-level occlusion.
    front=depth>object_depth+residual
    # A thin front bar is allowed; the background behind the object is never restored.
    front &= ~scene['protected'][y:y+h,x:x+w]
    if refine and front.any() and not front.all() and h*w<2_000_000:
        foreground=cv2.erode(front.astype(np.uint8),np.ones((3,3),np.uint8))>0
        surround=cv2.dilate(front.astype(np.uint8),np.ones((5,5),np.uint8))>0
        if foreground.sum()>15 and (~surround).sum()>15:
            labels=np.where(front,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
            labels[foreground]=cv2.GC_FGD;labels[~surround]=cv2.GC_BGD
            with _grab_lock:
                cv2.setRNGSeed(17)
                cv2.grabCut(np.asarray(scene['frame'])[y:y+h,x:x+w].copy(),labels,None,
                    np.zeros((1,65)),np.zeros((1,65)),1,cv2.GC_INIT_WITH_MASK)
            front=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD])
    fraction=float((alpha*front).sum()/max(alpha.sum(),1e-6))
    uncertain=((abs(depth-object_depth)<=residual)&(alpha>.5))
    return front,{'basis':'local_inverse_depth_fit','confidence':confidence,'scale':float(a),
        'offset':float(b0),'uncertainty':float(residual),'occluded_fraction':fraction,
        'ambiguous_fraction':float(uncertain.sum()/max(1,(alpha>.5).sum()))}


def local_scale(scene,site,anchor,major_mm):
    try:
        if site.get('reference_status')=='unknown':raise ValueError('no_measured_reference')
        interval=scale_from_reference(site,major_mm,*scene['size'],scene['depth'],anchor)
        return interval,{'basis':'same_layer_reference','status':'pass','reference':site['reference'],
                         'reference_mm':site['reference_mm'],'anchor_px':anchor}
    except ValueError as error:
        rejection=str(error)
    # No reference from another depth is silently treated as a measured local scale.
    estimates=[]
    for other in scene['supports']:
        if not other.get('reference') or other.get('reference_status')=='unknown':continue
        ends=np.asarray(other['reference'])*np.subtract(scene['size'],1)
        pixels=float(np.linalg.norm(ends[0]-ends[1]));lo,hi=other['reference_mm']
        estimates.append([pixels*major_mm[0]/hi,pixels*major_mm[1]/lo])
    if estimates:
        interval=np.median(estimates,axis=0).tolist();basis='scene_reference_interval_without_depth_calibration'
    else:
        w=scene['size'][0];interval=[w*.5*major_mm[0]/5000,w*.8*major_mm[1]/2000]
        basis='assumed_hopper_width_2000_to_5000_mm'
    low=max(80.*scene.get('_image_scale',1.),min(interval[0],scene['size'][0]*.18))
    high=max(low*1.15,min(interval[1],scene['size'][0]*.28))
    return [low,high],{'basis':basis,'status':'unknown','rejected_reference_reason':rejection,
                       'anchor_px':anchor,'assumption':'display estimate, not measured scene dimensions'}


def visible_support_mask(scene,site):
    """Constrain an estimated face to visible, locally depth-consistent pixels.

    Neural depth is used to reject discontinuities only. It supplies neither
    metric height nor a measured surface normal, so evidence stays unknown.
    """
    cache=scene.setdefault('_visible_support_masks',{})
    if site['id'] in cache:return cache[site['id']]
    mask=np.asarray(site['mask'],bool)&scene['allowed']&~scene['protected']
    depth=np.asarray(scene['depth_native'],np.float32)
    valid=mask&np.isfinite(depth);yy,xx=np.nonzero(valid)
    evidence={'version':'visible-support-v1','status':'unknown','site_id':site['id'],
        'basis':'observed_mask_and_relative_depth_continuity; not_metric_surface_height',
        'normal_inherited':bool(site.get('normal_inherited') or str(site['id']).startswith('D')),
        'input_pixels':int(mask.sum()),'finite_depth_pixels':int(valid.sum())}
    if len(xx)<32:
        evidence.update(reason='insufficient_depth_samples',retained_pixels=int(valid.sum()),
                        retained_fraction=float(valid.sum()/max(1,mask.sum())))
        cache[site['id']]=(valid,evidence);return valid,evidence
    step=max(1,len(xx)//2048);sx=xx[::step];sy=yy[::step]
    width,height=scene['size'];design=np.column_stack([sx/max(width,1),sy/max(height,1),np.ones(len(sx))])
    values=depth[sy,sx];kept=np.ones(len(values),bool)
    for _ in range(3):
        coeff=np.linalg.lstsq(design[kept],values[kept],rcond=None)[0]
        error=values-design@coeff
        median=float(np.median(error[kept]));mad=float(np.median(abs(error[kept]-median)))*1.4826
        tolerance=float(np.clip(3*mad,.012,.030))
        subset=abs(error-median)<=tolerance
        if subset.sum()<32:break
        kept=subset
    residual=np.abs(depth[yy,xx]-(xx/max(width,1)*coeff[0]+yy/max(height,1)*coeff[1]+coeff[2]+median))
    consistent=np.zeros_like(mask);consistent[yy,xx]=residual<=tolerance
    # Inspect a depth-map pixel neighbourhood at native resolution. Erode only
    # a confirmed discontinuity, never invent support inside a small mask hole.
    scale=max(1,int(round(width/max(scene.get('depth',depth).shape[1],1))))
    kernel=np.ones((2*scale+1,2*scale+1),np.uint8)
    local_min=cv2.erode(np.where(valid,depth,1.).astype(np.float32),kernel)
    local_max=cv2.dilate(np.where(valid,depth,0.).astype(np.float32),kernel)
    discontinuity=(local_max-local_min)>max(.035,2*tolerance)
    result=valid&consistent&~discontinuity
    evidence.update(plane_fit_coefficients=coeff.tolist(),relative_depth_tolerance=tolerance,
        fit_mad=mad,retained_pixels=int(result.sum()),retained_fraction=float(result.sum()/max(1,mask.sum())),
        rejected_depth_residual_pixels=int((valid&~consistent).sum()),
        rejected_discontinuity_pixels=int((valid&discontinuity).sum()),
        metric_height_known=site.get('plane_status')=='measured',
        reason='visible_surface_subset_only; height_and_occluded_contact_unverified')
    cache[site['id']]=(result,evidence);return result,evidence


def receiver_mesh(scene,candidate,*,bounds=None):
    """Separate finite patches; holes and gaps are absent from the collision receiver."""
    x,y,r,b=bounds or candidate['bounds']
    camera=candidate['camera'];offset=candidate['offset'];full=candidate['full_px']
    patches=[]
    strict=candidate.get('strict_contact',False)
    coherent=candidate.get('integration_mode')=='coherent'
    visible_evidence=[]
    support_ids=candidate.get('support_ids',[candidate['site_id']])
    independent=all(s.get('plane_origin_m') is not None and s.get('plane_status')=='measured'
                    for s in scene['supports'] if s['id'] in support_ids)
    if strict and len(support_ids)>1 and not independent:
        return {'patches':[],'height_confidence':'unknown','basis':'separate_support_heights_unresolved',
                'hidden_heights':'unknown','signature':signature('strict-receiver-v1',support_ids,'unresolved')}
    for site in scene['supports']:
        if site['id'] not in candidate.get('support_ids',[candidate['site_id']]):continue
        if coherent:
            if not support_plane_evidence(site)['eligible']:
                visible_evidence.append({**support_plane_evidence(site),'site_id':site['id'],
                    'reason':'nonplanar_surface_requires_independent_planes','retained_pixels':0})
                continue
            visible,proof=visible_support_mask(scene,site);visible_evidence.append(proof)
            mask=visible[y:b,x:r]
        else:mask=site['mask'][y:b,x:r]
        step=max(3,int(max(mask.shape)/50));vertices=[];faces=[]
        rows=list(range(0,max(0,mask.shape[0]-step),step));cols=list(range(0,max(0,mask.shape[1]-step),step))
        cells=np.array([[mask[cy:cy+step+1,cx:cx+step+1].all() if coherent else
                         mask[cy:cy+step+1,cx:cx+step+1].mean()>=.95 for cx in cols] for cy in rows])
        # Merge only identical occupied runs. Long bare gaps never become convex support.
        active={};rectangles=[]
        for row,values in enumerate(cells):
            padded=np.r_[False,values,False].astype(np.int8);starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
            runs=set(zip(starts.tolist(),ends.tolist()))
            for run,top in list(active.items()):
                if run not in runs:rectangles.append((run[0],top,run[1],row));del active[run]
            for run in runs:active.setdefault(run,row)
        rectangles.extend((run[0],top,run[1],len(rows)) for run,top in active.items())
        for left,top,right,bottom in rectangles:
            pixels=np.array([[x+left*step,y+top*step],[x+right*step,y+top*step],
                             [x+right*step,y+bottom*step],[x+left*step,y+bottom*step]])
            normal=candidate.get('plane_normal',[0,0,1]);origin=[0,0,0]
            if strict and independent:
                normal=site['plane_normal']
                unit=candidate['scale']['physical_size']['nominal_mm']/1000
                origin=(np.asarray(site['plane_origin_m'])/unit).tolist()
            world,_=world_on_plane(pixels,camera,offset,full,normal,origin)
            start=len(vertices);vertices.extend(world.tolist());faces.extend([[start,start+1,start+2],[start,start+2,start+3]])
        if faces:patches.append({'site_id':site['id'],'vertices':vertices,'faces':faces})
    sites=[s for s in scene['supports'] if s['id'] in candidate.get('support_ids',[candidate['site_id']])]
    coplanar=(all(s.get('slope_deg',0)<=7.5 and s.get('plane_residual',0)<=.04 for s in sites)
              and (max(s['depth'] for s in sites)-min(s['depth'] for s in sites))<=.04)
    if strict:
        return {'patches':patches,'basis':'independent_metric_supports' if independent else 'single_surface_estimated_plane',
                'height_confidence':'measured' if independent else 'estimated_single',
                'hidden_heights':'measured' if independent else 'unknown',
                **({'visible_support':visible_evidence,'contact_evidence_version':'visible-support-v1',
                    'scene_contact_status':'pass' if independent else 'unknown'} if coherent else {}),
                'signature':signature('strict-receiver-v2-visible' if coherent else 'strict-receiver-v1',patches,independent)}
    return {'patches':patches,'basis':'finite_coplanar_support_patches_estimated_from_image',
            'height_confidence':'estimated_coplanar' if coplanar else 'unknown',
            'hidden_heights':'unknown','signature':signature(VERSION,patches,coplanar)}


def anchors_near_foreground(scene,site,support,anchors):
    """Sample existing support next to a separately observed foreground surface."""
    closer=[s['mask'] for s in scene['supports'] if s['id']!=site['id'] and s['depth']>site['depth']+.045]
    if not closer:
        return []
    front=np.logical_or.reduce(closer)&~support
    if not front.any():
        return []
    interior=cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,5)
    distance=cv2.distanceTransform((~front).astype(np.uint8),cv2.DIST_L2,5)
    margin=max(2.,float(interior.max())*.22)
    radius=max(8.,float(interior.max()))
    score=np.where((interior>=margin)&(distance<2*radius),
                   np.minimum(interior/radius,1)*np.exp(-distance/radius),0).astype(np.float32)
    for x,y in anchors:
        cv2.circle(score,(round(x),round(y)),max(6,round(radius*.55)),0,-1)
    found=[]
    for _ in range(2):
        _,peak,_,point=cv2.minMaxLoc(score)
        if peak<.06:break
        found.append(list(point))
        cv2.circle(score,point,max(6,round(radius*.80)),0,-1)
    return found


def diagnostic(scene,candidate,out):
    from PIL import Image,ImageDraw,ImageFont
    image=scene['frame'].copy();draw=ImageDraw.Draw(image,'RGBA')
    for site in scene['supports']:
        if site['id'] in candidate.get('support_ids',[candidate['site_id']]):
            mask=site['mask'].astype(np.uint8)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour)>2:draw.line([tuple(p) for p in contour[:,0]],fill=(40,230,150,255),width=5)
    for poly in candidate.get('contact_polygons',[]):
        xy=np.asarray(poly)*candidate['full_px']+candidate['offset']
        if len(xy)>2:draw.polygon([tuple(p) for p in xy],fill=(255,160,40,130))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    draw.text((25,90),f"{candidate.get('placement','surface')} · 支撑/深度为估计 · {candidate['scale']['basis']}",font=font,fill='yellow')
    image.thumbnail((1600,1000));image.save(out,quality=90)


def pocket_proposals(asset,scene,site,mask,config):
    """Scale fixes the search window; proposals never change physical object size."""
    from scrap_composite.pockets import propose
    fixed=asset.get('physical_size')
    if fixed:
        if config.scale_mode=='reference_projection':
            from .reference_projection import local_scale as scale
        else:
            from .consistent_scale import local_scale as scale
        interval,basis=scale(scene,site,site['anchor_px'],fixed)
    else:interval,basis=local_scale(scene,site,site['anchor_px'],asset['semantic']['major_mm'])
    target=math.sqrt(max(interval[0],1)*max(interval[1],1))
    proposals,summary=propose(scene['depth_native'],mask,scene['allowed'],scene['protected'],target)
    summary['scale_basis']={'status':basis.get('status','unknown'),'basis':basis.get('basis'),
                            'physical_size_fixed':bool(fixed),'range_px':interval}
    return proposals,summary


def scale_pocket(pocket,factor):
    """Move diagnostics from the coarse search grid to the original image grid."""
    p=json.loads(json.dumps(pocket));p['anchor_px']=(np.asarray(p['anchor_px'])/factor).tolist()
    p['clearance_px']/=factor
    for scale in p['rim_evidence']:
        scale['radius_px']/=factor
        scale['rim_points_px']=(np.asarray(scale['rim_points_px']).reshape(-1,2)/factor).tolist()
    return p


def choose_pile_candidates(asset,scene,seed=0,config=None):
    from .scene import cached_projection,fitted_masks
    factor=min(1.,960/max(scene['size']))
    if factor==1:return _choose_pile_candidates(asset,scene,seed,config)
    size=tuple(round(v*factor) for v in scene['size'])
    small={**scene,'size':size,'frame':scene['frame'].resize(size),'_image_scale':factor,
        '_visible_support_masks':{},
        'depth_native':cv2.resize(scene['depth_native'],size,interpolation=cv2.INTER_LINEAR),
        'allowed':cv2.resize(scene['allowed'].astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0,
        'protected':cv2.resize(scene['protected'].astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0,
        'supports':[{**s,'mask':cv2.resize(s['mask'].astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0,
                     'anchor_px':np.rint(np.asarray(s['anchor_px'])*factor).astype(int).tolist()} for s in scene['supports']]}
    rows,reject=_choose_pile_candidates(asset,small,seed,config,limit=12)
    for rejection in reject:
        if 'anchor' in rejection:rejection['anchor']=(np.asarray(rejection['anchor'])/factor).tolist()
    data=np.load(asset['geometry']);selected=[]
    for row in rows:
        if row.get('pocket'):row['pocket']=scale_pocket(row['pocket'],factor)
        if row.get('pocket_search'):
            summary=json.loads(json.dumps(row['pocket_search']));row['pocket_search']=summary
            summary['search_grid_scale_to_original']=factor
            if summary.get('bounds'):summary['bounds']=(np.asarray(summary['bounds'])/factor).tolist()
            if summary.get('target_px'):summary['target_px']/=factor
            if summary.get('scale_basis',{}).get('range_px'):
                summary['scale_basis']['range_px']=(np.asarray(summary['scale_basis']['range_px'])/factor).tolist()
        row['anchor_px']=(np.asarray(row['anchor_px'])/factor).tolist();row['full_px']/=factor
        for name in ('target_px','expected_major_px'):row[name]/=factor
        row['scale']['world_pixels_per_unit']/=factor
        row['scale']['range_px']=[v/factor for v in row['scale']['range_px']]
        row['scale']['anchor_px']=row['anchor_px']
        if row['scale'].get('fixed_physical_size'):
            from .consistent_scale import calibration_to_native
            row['scale']=calibration_to_native(row['scale'],factor,scene['size'])
        if row.get('strict_contact') or row['scale'].get('calibration',{}).get('status')=='pass':
            from .scene import projection_geometry
            geom=projection_geometry(data['vertices'],data['faces'],{'matrix':row['matrix']},0,row['elevation'],
                                     row['plane_normal'],camera=row['camera'])
        else:
            geom=cached_projection(asset,data['vertices'],data['faces'],{'matrix':row['matrix']},0,row['elevation'])
        fit=fitted_masks(geom,row['anchor_px'],row['full_px'],scene)
        if fit is None:reject.append({'candidate':row['id'],'reason':'full_resolution_outside_image'});continue
        x,y,r,b=fit['bounds'];mask=fit['hull']
        if np.any(scene['protected'][y:b,x:r]&mask) or (scene['allowed'][y:b,x:r]&mask).sum()/max(1,mask.sum())<.98:
            reject.append({'candidate':row['id'],'reason':'full_resolution_clearance_or_protection_failed'});continue
        row.update(bounds=fit['bounds'],offset=fit['offset'])
        support=support_union(scene,row['support_ids'],fit['bounds'])
        if row.get('integration_mode')=='coherent':
            support=np.logical_or.reduce([visible_support_mask(scene,s)[0][y:b,x:r]
                for s in scene['supports'] if s['id'] in row['support_ids']])
        cover=contact_support(fit['contact'],support)
        row['contact_min']=min(cover) if cover else 0;row['contact_coverages']=cover
        if row['contact_min']<.9:
            if row.get('strict_contact'):
                reject.append({'candidate':row['id'],'reason':'full_resolution_contact_below_90_percent',
                    'contact_coverages':cover,'anchor_px':row['anchor_px'],'support_ids':row['support_ids'],
                    'visible_support':row.get('visible_support_evidence')});continue
            row['uncertainties'].append('full_resolution_support_unknown')
        if row.get('pocket'):
            from scrap_composite.pockets import check_projection
            evidence=check_projection(row['pocket'],fit['hull'],fit['contact'],fit['bounds'],support,
                                      scene['allowed'][y:b,x:r],scene['protected'][y:b,x:r])
            row['pocket_projection']=evidence
            if evidence['program_fit']=='fail':
                reject.append({'candidate':row['id'],'reason':'full_resolution_pocket_projection_failed','evidence':evidence});continue
        row['receiver']=receiver_mesh(scene,row)
        selected.append(row)
        if len(selected)==3:break
    return selected,reject


def _choose_pile_candidates(asset,scene,seed=0,config=None,limit=3):
    from .scene import cached_projection,fitted_masks,support_anchors
    from .geometry import terrain_matrix,horizontal_angle
    from .schema import BatchConfig
    config=config or BatchConfig(backgrounds='unused',output='unused')
    strict=config.contact_mode=='strict'
    native=config.appearance=='native'
    coherent=getattr(config,'integration_mode','legacy')=='coherent'
    pockets_enabled=coherent and getattr(config,'workflow','legacy')=='evidence_v1'
    data=np.load(asset['geometry']);vertices=data['vertices'];faces=data['faces']
    midpoint=sum(scene['answer']['elevation'])/2;records=[];reject=[];seen=set();geometries={};world_patches={}
    priorities={};shape=shape_priority(vertices,asset['axis']) if coherent else None
    centroid=vertices.mean(0)
    if coherent and asset.get('proxy'):
        centroid=np.asarray(json.loads(Path(asset['proxy']).read_text(encoding='utf8'))['centroid'])
    rng=np.random.default_rng(seed);diagonal=np.linalg.norm(scene['size'])
    # Diversify sites before any expensive model render, using only image-derived locations.
    primary=[]
    def site_rank(s):
        hint=support_plane_evidence(s)
        return -s['confidence']+(hint['penalty']+.02*s.get('slope_deg',0) if coherent else 0)
    for s in sorted(scene['supports'],key=site_rank):
        if coherent and strict and not support_plane_evidence(s)['eligible']:
            reject.append({'site':s['id'],'reason':'nonplanar_surface_requires_independent_planes',
                           'evidence':support_plane_evidence(s),'label':s.get('label')})
            continue
        if any(np.linalg.norm(np.subtract(s['anchor_px'],q['anchor_px']))<diagonal*.045 for q in primary):continue
        primary.append(s)
        if len(primary)==8:break
    for site in primary:
        neighbors=sorted([s for s in scene['supports'] if s['id']!=site['id'] and
            abs(s['depth']-site['depth'])<.16 and np.linalg.norm(np.subtract(s['anchor_px'],site['anchor_px']))<diagonal*.25],
            key=lambda s:np.linalg.norm(np.subtract(s['anchor_px'],site['anchor_px'])))[:3]
        groups=[('surface',[site])]
        if strict:
            measured=[s for s in [site,*neighbors] if s.get('plane_status')=='measured' and s.get('plane_origin_m') is not None]
            if len(measured)>1 and any(s['id']==site['id'] for s in measured):groups.extend([('combined',measured),('embedded',measured)])
            groups.append(('embedded',[site]))
        elif neighbors:groups.extend([('combined',[site,*neighbors]),('embedded',[site,*neighbors])])
        else:groups.append(('embedded',[site]))
        groups=[g for g in groups if config.placement_mode in ('auto',g[0])]
        site_count=0;reason_counts={}
        for mode,group in groups:
            ids=[s['id'] for s in group];union=support_union(scene,ids)
            visible_evidence=[]
            if coherent:
                pieces=[visible_support_mask(scene,s) for s in group]
                union=np.logical_or.reduce([piece[0] for piece in pieces]);visible_evidence=[piece[1] for piece in pieces]
                if not union.any():
                    reason_counts['no_depth_consistent_visible_support']=reason_counts.get('no_depth_consistent_visible_support',0)+1
                    reject.append({'site':site['id'],'reason':'no_depth_consistent_visible_support',
                                   'visible_support':visible_evidence});continue
            anchor_site={**site,'mask':union}
            anchors=support_anchors(anchor_site)
            pockets=[];pocket_summary=None
            if pockets_enabled:
                try:pockets,pocket_summary=pocket_proposals(asset,scene,site,union,config)
                except ValueError as error:
                    pocket_summary={'status':'unknown','reason':'pocket_scale_unavailable','detail':str(error)}
                if pockets:
                    anchors=[p['anchor_px'] for p in pockets]+[a for a in anchors if all(np.linalg.norm(np.subtract(a,p['anchor_px']))>3 for p in pockets)]
                else:reject.append({'site':site['id'],'reason':'no_confirmed_pocket_planar_candidates_retained','pocket_analysis':pocket_summary})
            if native and config.placement_mode in ('auto','embedded'):
                anchors+=anchors_near_foreground(scene,site,union,anchors)
            for anchor_index,anchor in enumerate(anchors):
                pocket=next((p for p in pockets if np.linalg.norm(np.subtract(p['anchor_px'],anchor))<1e-4),None)
                fixed_size = asset.get('physical_size')
                if fixed_size:
                    if config.scale_mode=='reference_projection':
                        from .reference_projection import local_scale as physical_scale
                    else:
                        from .consistent_scale import local_scale as physical_scale
                    interval,evidence=physical_scale(scene,site,anchor,fixed_size)
                else:
                    interval,evidence=local_scale(scene,site,anchor,asset['semantic']['major_mm'])
                if evidence['status']=='unknown' and config.uncertain_policy=='skip':
                    reject.append({'site':site['id'],'anchor':anchor,'reason':'scale_evidence_insufficient'});continue
                low,high=interval
                image_scale=scene.get('_image_scale',1.)
                if not fixed_size and (low>scene['size'][0]*.28 or high<64*image_scale):
                    evidence={**evidence,'status':'unknown','basis':'bounded_display_scale_from_conflicting_reference',
                              'original_interval':interval}
                    low=max(80*image_scale,min(low,scene['size'][0]*.14));high=max(low*1.2,min(high,scene['size'][0]*.25))
                scales=[low] if fixed_size else sorted(set([math.sqrt(low*high),low*1.05,high*.95]))
                delta=np.diff(np.asarray(site['reference'])*np.subtract(scene['size'],1),axis=0)[0]
                correction=np.linalg.norm([delta[0],delta[1]/max(.2,math.sin(math.radians(midpoint)))])/max(1,np.linalg.norm(delta))
                if evidence['status']=='unknown' or fixed_size:correction=1.
                for pose in asset['poses']:
                    for yaw in range(0,360,30):
                        normal=site.get('plane_normal',[0,0,1]) if strict else [0,0,1]
                        calibrated=(evidence.get('calibration',{}).get('status')=='pass' or evidence.get('calibration',{}).get('projectable'))
                        matrix=terrain_matrix(pose['matrix'],yaw,normal)
                        tilt=horizontal_angle(asset['axis'],matrix)
                        if tilt>15:continue
                        key=(pose['id'],yaw,tuple(normal))
                        if coherent and key not in priorities:
                            priorities[key]=stability_priority(vertices,matrix,normal,asset['axis'],centroid,shape)
                        priority=priorities.get(key)
                        if key not in geometries:
                            geometries[key]=cached_projection(asset,vertices,faces,{**pose,'matrix':matrix.tolist()},0,midpoint,normal)
                        geom=geometries[key]
                        distance=np.linalg.norm(np.subtract(geom['camera']['eye'],geom['camera']['target']))
                        projected_span=float(np.ptp(geom['uv'],axis=0).max())
                        for scale_index,target in enumerate(scales):
                            if not low<=target<=high:continue
                            # Physical scale stays fixed while yaw changes projected length.
                            world_ppm=target*correction
                            full_px=world_ppm*2*distance*math.tan(math.radians(geom['camera']['fov']/2))
                            projected_major=full_px*projected_span
                            if calibrated:
                                from .reference_projection import metric_projection
                                from .geometry import contact_world, transformed
                                from .scene import projection_geometry
                                patch_key=(pose['id'],yaw,tuple(normal))
                                if patch_key not in world_patches:
                                    world_patches[patch_key]=contact_world(transformed(vertices,matrix),faces,normal=normal)
                                patches=world_patches[patch_key]
                                if not patches:continue
                                points=np.concatenate(patches);center=(points.min(0)+points.max(0))/2
                                factor=scene.get('_image_scale',1.)
                                cam,frame=metric_projection(evidence['calibration'],np.asarray(anchor)/factor,
                                    fixed_size['nominal_mm'],np.asarray(scene['size'])/factor,center)
                                geom=projection_geometry(vertices,faces,{'matrix':matrix.tolist()},0,midpoint,normal,camera=cam,
                                    patches=patches,raster_size=128 if image_scale<1 else 384)
                                full_px=frame*factor;projected_major=full_px*float(np.ptp(geom['uv'],axis=0).max())
                                key=(pose['id'],yaw,tuple(normal),site['id'],anchor_index)
                                geometries[key]=geom
                            if projected_major<64*image_scale:continue
                            fit=fitted_masks(geom,anchor,full_px,scene)
                            if fit is None:reason_counts['outside_image_or_size'] = reason_counts.get('outside_image_or_size',0)+1;continue
                            x,y,r,b=fit['bounds'];mask=fit['hull'];area=max(1,mask.sum())
                            protected=bool(np.any(scene['protected'][y:b,x:r]&mask))
                            inside=float((scene['allowed'][y:b,x:r]&mask).sum()/area)
                            if protected or inside<.98:
                                reason='protected_region_conflict' if protected else 'outside_hopper'
                                reason_counts[reason]=reason_counts.get(reason,0)+1;continue
                            support=union[y:b,x:r];cover=contact_support(fit['contact'],support)
                            minimum=min(cover) if cover else 0
                            if strict and minimum<.9:
                                reason_counts['contact_below_90_percent']=reason_counts.get('contact_below_90_percent',0)+1;continue
                            pocket_fit=None
                            if pocket:
                                from scrap_composite.pockets import check_projection
                                pocket_fit=check_projection(pocket,mask,fit['contact'],fit['bounds'],support,
                                                            scene['allowed'][y:b,x:r],scene['protected'][y:b,x:r])
                                if pocket_fit['program_fit']=='fail':
                                    reason_counts['pocket_real_mesh_fit_failed']=reason_counts.get('pocket_real_mesh_fit_failed',0)+1;continue
                            front=scene['depth_native'][y:b,x:r]>site['depth']+.035
                            fraction=float((front&mask).sum()/area)
                            if minimum<.20 and fraction<.10:
                                reason_counts['visible_contact_without_support']=reason_counts.get('visible_contact_without_support',0)+1;continue
                            uncertain=[]
                            if minimum<.9:uncertain.append('contact_support_incomplete_or_hidden')
                            if evidence['status']=='unknown':uncertain.append('scale_evidence_insufficient')
                            if any(s.get('evidence_status')=='unknown' for s in group):uncertain.append('support_geometry_uncertain')
                            if coherent and any(support_plane_evidence(s)['basis']=='inherited_neighbour_normal' for s in group):
                                uncertain.append('inherited_support_normal_unverified')
                            if uncertain and config.uncertain_policy=='skip':continue
                            if mode=='embedded' and fraction<.05:continue
                            bonus=.30 if mode=='embedded' and .10<=fraction<=.40 else .08 if mode=='combined' else 0
                            score=site['confidence']*.5+minimum*.7+bonus-.18*len(uncertain)-max(0,fraction-.4)*2
                            score-=abs(target-math.sqrt(low*high))/max(high,1)*.1
                            if strict:score+=(.35 if len(group)==1 else 0)-pose['height']*.2-site.get('slope_deg',0)*.01
                            if coherent:score-=priority['penalty']+support_plane_evidence(site)['penalty']
                            if pocket:score+=.55*pocket['score']
                            ident=f"{site['id']}-{mode}-{pose['id']}-Y{yaw}-Z{scale_index}-A{anchor_index}"
                            if ident in seen:continue
                            seen.add(ident);site_count+=1
                            calibrated=bool(evidence.get('calibration',{}).get('metric_camera'))
                            actual_elevation=math.degrees(math.asin(-geom['camera']['basis'][2][2])) if calibrated else midpoint
                            records.append({'id':ident,'site_id':site['id'],'support_ids':ids,'placement':mode,
                                'material_classes':asset['semantic']['material_classes'],
                                'pose_id':pose['id'],'matrix':matrix.tolist(),'yaw':0,'horizontal_yaw_deg':yaw,
                                'plane_normal':normal,'plane_basis':'local_visible_supports_estimated','strict_contact':strict,
                                'camera':geom['camera'],'elevation':actual_elevation,
                                'elevation_interval':[actual_elevation,actual_elevation] if calibrated else scene['answer']['elevation'],
                                'anchor_px':anchor,'full_px':full_px,'target_px':projected_major,'expected_major_px':projected_major,
                                'axis_angle_deg':tilt,'offset':fit['offset'],'bounds':fit['bounds'],
                                'anchor_uv':geom['anchor'].tolist(),
                                'contact_min':minimum,'contact_coverages':cover,'inside':inside,'foreground_fraction':fraction,
                                'uncertainties':uncertain,'base_pose':pose,
                                **({'stability_priority':priority,'support_plane_evidence':support_plane_evidence(site),
                                    'integration_mode':'coherent','visible_support_evidence':visible_evidence} if coherent else {}),
                                **({'workflow':'evidence_v1','pocket':pocket,'pocket_projection':pocket_fit,
                                    'pocket_search':pocket_summary,'placement_basis':'visible_recess_candidate' if pocket else 'planar_fallback'} if pockets_enabled else {}),
                                'scale':{**evidence,'major_mm':asset['semantic']['major_mm'],'range_px':[low,high],
                                    'reference_projection_correction':float(correction),'world_pixels_per_unit':world_ppm},
                                'score':score+float(rng.uniform(0,.005)),'projection_key':key})
        if not site_count:reject.append({'site':site['id'],'reason':'all_local_candidates_rejected','details':reason_counts})
    # Keep a small diverse shortlist, then compute per-pixel depth ordering on actual meshes.
    shortlist=[]
    for row in sorted(records,key=lambda r:-r['score']):
        if sum(q['site_id']==row['site_id'] and q['placement']==row['placement'] for q in shortlist)>=2:continue
        shortlist.append(row)
        if not coherent and len(shortlist)>=12:break
    if coherent:shortlist=diverse_candidates(shortlist,12)
    chosen=[]
    for row in shortlist:
        geom=geometries[row.pop('projection_key')]
        row['contact_polygons']=[p.tolist() for p in geom['contacts']]
        fit=fitted_masks(geom,row['anchor_px'],row['full_px'],scene)
        # Only pieces touched by actual contact clusters receive the object. Nearby
        # steel with no contact remains an independent potential foreground occluder.
        x,y,r,b=fit['bounds']
        touched=[s['id'] for s in scene['supports'] if s['id'] in row['support_ids'] and
                 np.any((fit['contact']>0)&s['mask'][y:b,x:r])]
        if touched:row['support_ids']=touched
        front,depth_info=depth_order(scene,row,fit['bounds'],fit['depth'],fit['hull'].astype(float))
        fraction=depth_info['occluded_fraction'];row['depth_order']=depth_info;row['foreground_fraction']=fraction
        if fraction>.45:
            reject.append({'candidate':row['id'],'reason':'excessive_foreground_occlusion','fraction':fraction});continue
        if row['placement']=='embedded' and fraction<.10:row['placement']='combined' if len(row['support_ids'])>1 else 'surface'
        elif config.placement_mode=='auto' and .10<=fraction<=.40:row['placement']='embedded'
        elif row['placement']=='combined' and len(row['support_ids'])<2:row['placement']='surface'
        if depth_info['confidence']=='unknown' or depth_info['ambiguous_fraction']>.25:
            row['uncertainties'].append('occlusion_order_uncertain')
        if config.uncertain_policy=='skip' and row['uncertainties']:continue
        if native and depth_info['confidence']=='estimated' and depth_info['ambiguous_fraction']<=.25:
            # Prefer actual partial cover only among comparably supported placements.
            row['integration_bonus']=.24*max(0,1-abs(fraction-.21)/.17)
            row['score']+=row['integration_bonus']
        if not native and any(q['site_id']==row['site_id'] and q['placement']==row['placement'] for q in chosen):continue
        row['receiver']=receiver_mesh(scene,row)
        row['evidence_confidence']='unknown' if row['uncertainties'] else 'estimated'
        chosen.append(row)
        if not native and not coherent and len(chosen)==limit:break
    if native:
        selected={}
        for row in sorted(chosen,key=lambda r:(len(set(r['uncertainties'])),-r['score'])):
            selected.setdefault((row['site_id'],row['placement']),row)
        chosen=list(selected.values())[:limit]
    if coherent:chosen=diverse_candidates(chosen,limit)
    return chosen,reject
