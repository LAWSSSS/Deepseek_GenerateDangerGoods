"""Local piecewise support and depth ordering. All uncalibrated evidence stays explicit."""
import math
import threading
from pathlib import Path
import cv2
import numpy as np
from numba import njit
from .common import signature
from .geometry import project,contact_support,scale_from_reference

VERSION='piecewise-pile-v3-depth-order'
_grab_lock=threading.Lock()


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


def mesh_raster(points,faces,camera,size=384):
    xy=project(points,camera)*size
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


def receiver_mesh(scene,candidate,*,bounds=None):
    """Separate finite patches; holes and gaps are absent from the collision receiver."""
    x,y,r,b=bounds or candidate['bounds']
    camera=candidate['camera'];offset=candidate['offset'];full=candidate['full_px']
    patches=[]
    for site in scene['supports']:
        if site['id'] not in candidate.get('support_ids',[candidate['site_id']]):continue
        mask=site['mask'][y:b,x:r]
        step=max(3,int(max(mask.shape)/50));vertices=[];faces=[]
        rows=list(range(0,max(0,mask.shape[0]-step),step));cols=list(range(0,max(0,mask.shape[1]-step),step))
        cells=np.array([[mask[cy:cy+step+1,cx:cx+step+1].mean()>=.95 for cx in cols] for cy in rows])
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
            world,_=world_on_plane(pixels,camera,offset,full,candidate.get('plane_normal',[0,0,1]))
            start=len(vertices);vertices.extend(world.tolist());faces.extend([[start,start+1,start+2],[start,start+2,start+3]])
        if faces:patches.append({'site_id':site['id'],'vertices':vertices,'faces':faces})
    sites=[s for s in scene['supports'] if s['id'] in candidate.get('support_ids',[candidate['site_id']])]
    coplanar=(all(s.get('slope_deg',0)<=7.5 and s.get('plane_residual',0)<=.04 for s in sites)
              and (max(s['depth'] for s in sites)-min(s['depth'] for s in sites))<=.04)
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


def choose_pile_candidates(asset,scene,seed=0,config=None):
    from .scene import cached_projection,fitted_masks
    factor=min(1.,960/max(scene['size']))
    if factor==1:return _choose_pile_candidates(asset,scene,seed,config)
    size=tuple(round(v*factor) for v in scene['size'])
    small={**scene,'size':size,'frame':scene['frame'].resize(size),'_image_scale':factor,
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
        row['anchor_px']=(np.asarray(row['anchor_px'])/factor).tolist();row['full_px']/=factor
        for name in ('target_px','expected_major_px'):row[name]/=factor
        row['scale']['world_pixels_per_unit']/=factor
        row['scale']['range_px']=[v/factor for v in row['scale']['range_px']]
        row['scale']['anchor_px']=row['anchor_px']
        if row['scale'].get('fixed_physical_size'):
            from .consistent_scale import calibration_to_native
            row['scale']=calibration_to_native(row['scale'],factor,scene['size'])
        geom=cached_projection(asset,data['vertices'],data['faces'],{'matrix':row['matrix']},0,row['elevation'])
        fit=fitted_masks(geom,row['anchor_px'],row['full_px'],scene)
        if fit is None:reject.append({'candidate':row['id'],'reason':'full_resolution_outside_image'});continue
        x,y,r,b=fit['bounds'];mask=fit['hull']
        if np.any(scene['protected'][y:b,x:r]&mask) or (scene['allowed'][y:b,x:r]&mask).sum()/max(1,mask.sum())<.98:
            reject.append({'candidate':row['id'],'reason':'full_resolution_clearance_or_protection_failed'});continue
        row.update(bounds=fit['bounds'],offset=fit['offset'])
        cover=contact_support(fit['contact'],support_union(scene,row['support_ids'],fit['bounds']))
        row['contact_min']=min(cover) if cover else 0;row['contact_coverages']=cover
        if row['contact_min']<.9:row['uncertainties'].append('full_resolution_support_unknown')
        row['receiver']=receiver_mesh(scene,row)
        selected.append(row)
        if len(selected)==3:break
    return selected,reject


def _choose_pile_candidates(asset,scene,seed=0,config=None,limit=3):
    from .scene import cached_projection,fitted_masks,support_anchors
    from .geometry import terrain_matrix,horizontal_angle
    from .schema import BatchConfig
    config=config or BatchConfig(backgrounds='unused',output='unused')
    native=config.appearance=='native'
    data=np.load(asset['geometry']);vertices=data['vertices'];faces=data['faces']
    midpoint=sum(scene['answer']['elevation'])/2;records=[];reject=[];seen=set();geometries={}
    rng=np.random.default_rng(seed);diagonal=np.linalg.norm(scene['size'])
    # Diversify sites before any expensive model render, using only image-derived locations.
    primary=[]
    for s in sorted(scene['supports'],key=lambda s:-s['confidence']):
        if any(np.linalg.norm(np.subtract(s['anchor_px'],q['anchor_px']))<diagonal*.045 for q in primary):continue
        primary.append(s)
        if len(primary)==8:break
    for site in primary:
        neighbors=sorted([s for s in scene['supports'] if s['id']!=site['id'] and
            abs(s['depth']-site['depth'])<.16 and np.linalg.norm(np.subtract(s['anchor_px'],site['anchor_px']))<diagonal*.25],
            key=lambda s:np.linalg.norm(np.subtract(s['anchor_px'],site['anchor_px'])))[:3]
        groups=[('surface',[site])]
        if neighbors:groups.extend([('combined',[site,*neighbors]),('embedded',[site,*neighbors])])
        else:groups.append(('embedded',[site]))
        groups=[g for g in groups if config.placement_mode in ('auto',g[0])]
        site_count=0;reason_counts={}
        for mode,group in groups:
            ids=[s['id'] for s in group];union=support_union(scene,ids)
            anchor_site={**site,'mask':union}
            anchors=support_anchors(anchor_site)
            if native and config.placement_mode in ('auto','embedded'):
                anchors+=anchors_near_foreground(scene,site,union,anchors)
            for anchor_index,anchor in enumerate(anchors):
                fixed_size = asset.get('physical_size')
                if fixed_size:
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
                        matrix=terrain_matrix(pose['matrix'],yaw,[0,0,1])
                        tilt=horizontal_angle(asset['axis'],matrix)
                        if tilt>15:continue
                        key=(pose['id'],yaw)
                        if key not in geometries:
                            geometries[key]=cached_projection(asset,vertices,faces,{**pose,'matrix':matrix.tolist()},0,midpoint)
                        geom=geometries[key]
                        distance=np.linalg.norm(np.subtract(geom['camera']['eye'],geom['camera']['target']))
                        projected_span=float(np.ptp(geom['uv'],axis=0).max())
                        for scale_index,target in enumerate(scales):
                            if not low<=target<=high:continue
                            # Physical scale stays fixed while yaw changes projected length.
                            world_ppm=target*correction
                            full_px=world_ppm*2*distance*math.tan(math.radians(geom['camera']['fov']/2))
                            projected_major=full_px*projected_span
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
                            front=scene['depth_native'][y:b,x:r]>site['depth']+.035
                            fraction=float((front&mask).sum()/area)
                            if minimum<.20 and fraction<.10:
                                reason_counts['visible_contact_without_support']=reason_counts.get('visible_contact_without_support',0)+1;continue
                            uncertain=[]
                            if minimum<.9:uncertain.append('contact_support_incomplete_or_hidden')
                            if evidence['status']=='unknown':uncertain.append('scale_evidence_insufficient')
                            if any(s.get('evidence_status')=='unknown' for s in group):uncertain.append('support_geometry_uncertain')
                            if uncertain and config.uncertain_policy=='skip':continue
                            if mode=='embedded' and fraction<.05:continue
                            bonus=.30 if mode=='embedded' and .10<=fraction<=.40 else .08 if mode=='combined' else 0
                            score=site['confidence']*.5+minimum*.7+bonus-.18*len(uncertain)-max(0,fraction-.4)*2
                            score-=abs(target-math.sqrt(low*high))/max(high,1)*.1
                            ident=f"{site['id']}-{mode}-{pose['id']}-Y{yaw}-Z{scale_index}-A{anchor_index}"
                            if ident in seen:continue
                            seen.add(ident);site_count+=1
                            records.append({'id':ident,'site_id':site['id'],'support_ids':ids,'placement':mode,
                                'material_classes':asset['semantic']['material_classes'],
                                'pose_id':pose['id'],'matrix':matrix.tolist(),'yaw':0,'horizontal_yaw_deg':yaw,
                                'plane_normal':[0,0,1],'plane_basis':'local_visible_supports_estimated',
                                'camera':geom['camera'],'elevation':midpoint,'elevation_interval':scene['answer']['elevation'],
                                'anchor_px':anchor,'full_px':full_px,'target_px':projected_major,'expected_major_px':projected_major,
                                'axis_angle_deg':tilt,'offset':fit['offset'],'bounds':fit['bounds'],
                                'anchor_uv':geom['anchor'].tolist(),
                                'contact_min':minimum,'contact_coverages':cover,'inside':inside,'foreground_fraction':fraction,
                                'uncertainties':uncertain,'base_pose':pose,
                                'scale':{**evidence,'major_mm':asset['semantic']['major_mm'],'range_px':[low,high],
                                    'reference_projection_correction':float(correction),'world_pixels_per_unit':world_ppm},
                                'score':score+float(rng.uniform(0,.005)),'projection_key':key})
        if not site_count:reject.append({'site':site['id'],'reason':'all_local_candidates_rejected','details':reason_counts})
    # Keep a small diverse shortlist, then compute per-pixel depth ordering on actual meshes.
    shortlist=[]
    for row in sorted(records,key=lambda r:-r['score']):
        if sum(q['site_id']==row['site_id'] and q['placement']==row['placement'] for q in shortlist)>=2:continue
        shortlist.append(row)
        if len(shortlist)>=12:break
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
        if not native and len(chosen)==limit:break
    if native:
        selected={}
        for row in sorted(chosen,key=lambda r:(len(set(r['uncertainties'])),-r['score'])):
            selected.setdefault((row['site_id'],row['placement']),row)
        chosen=list(selected.values())[:limit]
    return chosen,reject
