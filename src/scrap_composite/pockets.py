"""Image-evidence pocket proposals, independent of render and model services.

The input depth is relative inverse depth (larger is nearer). A proposal is a
visible support patch surrounded by nearer material at several object-related
scales. It is not a recovered metric cavity or proof of hidden support.
"""
import math
import cv2
import numpy as np

VERSION = 'visible-bottom-pockets-v1'
SCALES = (.35, .55, .80)


def _smooth(values, valid, sigma):
    weights=cv2.GaussianBlur(valid.astype(np.float32),(0,0),sigma)
    smooth=cv2.GaussianBlur(np.where(valid,values,0).astype(np.float32),(0,0),sigma)
    return smooth/np.maximum(weights,1e-6),weights


def _trend(depth, valid):
    yy,xx=np.nonzero(valid);h,w=depth.shape
    stride=max(1,len(xx)//3000);xx=xx[::stride];yy=yy[::stride]
    if len(xx)<24:return np.zeros_like(depth),np.zeros(3),.03
    design=np.column_stack([xx/max(w,1),yy/max(h,1),np.ones(len(xx))])
    values=depth[yy,xx];keep=np.ones(len(values),bool)
    for _ in range(3):
        coeff=np.linalg.lstsq(design[keep],values[keep],rcond=None)[0]
        residual=values-design@coeff
        center=np.median(residual[keep]);mad=float(np.median(abs(residual[keep]-center)))*1.4826
        proposed=abs(residual-center)<=max(.012,3*mad)
        if proposed.sum()<24:break
        keep=proposed
    grid_y,grid_x=np.indices(depth.shape)
    fitted=grid_x/max(w,1)*coeff[0]+grid_y/max(h,1)*coeff[1]+coeff[2]
    # Fine depth residual estimates numerical/texture variation, not cavity height.
    fine=depth-cv2.GaussianBlur(depth,(0,0),1.)
    noise=float(np.median(abs(fine[valid]-np.median(fine[valid]))))*1.4826
    return (depth-fitted).astype(np.float32),coeff,max(.002,noise)


def propose(depth, support_mask, allowed, protected, target_px, *, max_candidates=3):
    """Return deterministic pocket anchors and a reproducible evidence summary.

    RGB brightness is deliberately absent: a dark painted region or shadow
    cannot become a depression without consistent independent depth evidence.
    A zero-valued support mask always returns no pockets, even in a depth valley.
    """
    depth=np.asarray(depth,np.float32);support=np.asarray(support_mask,bool)
    if depth.shape!=support.shape or depth.shape!=np.shape(allowed) or depth.shape!=np.shape(protected):
        raise ValueError('pocket_inputs_must_share_pixel_grid')
    if not math.isfinite(float(target_px)) or target_px<=0:raise ValueError('invalid_projected_object_size')
    valid=np.asarray(allowed,bool)&~np.asarray(protected,bool)&np.isfinite(depth)
    bottom=support&valid
    summary={'version':VERSION,'status':'unknown','target_px':float(target_px),'scale_factors':list(SCALES),
        'basis':'visible_bottom_and_multiscale_relative_depth_rims','depth_units':'relative_inverse_depth',
        'metric_cavity_recovered':False,'physics_validated':False,'bottom_pixels':int(bottom.sum()),
        'parameters':{'minimum_supported_directions':2,'minimum_scale_votes':2,'minimum_direction_separation_deg':135}}
    yy,xx=np.nonzero(bottom)
    if len(xx)<24:return [],{**summary,'reason':'no_visible_bottom'}
    # Bound the analysis around this support. Unrelated scene colour/depth does not vote.
    pad=max(12,int(math.ceil(target_px*.95)));h,w=depth.shape
    l=max(0,int(xx.min())-pad);r=min(w,int(xx.max())+pad+1)
    t=max(0,int(yy.min())-pad);b=min(h,int(yy.max())+pad+1)
    factor=min(1.,512/max(r-l,b-t));size=(max(8,round((r-l)*factor)),max(8,round((b-t)*factor)))
    d=cv2.resize(np.where(np.isfinite(depth[t:b,l:r]),depth[t:b,l:r],0),size,interpolation=cv2.INTER_LINEAR)
    v=cv2.resize(valid[t:b,l:r].astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    floor=cv2.resize(bottom[t:b,l:r].astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    # Fit the floor, not the elevated rims: a lone nearby wall must not tilt the
    # reference plane and manufacture a second wall on its opposite side.
    residual,coeff,noise=_trend(d,v&floor)
    extent=max(6.,target_px*factor);sigma=max(.8,extent*.035)
    center,weights=_smooth(residual,v,sigma)
    bottom_weight=cv2.GaussianBlur(floor.astype(np.float32),(0,0),sigma)
    safe=floor&(bottom_weight>=.80)&(weights>=.85)
    tolerance=max(.012,noise*4)
    gy,gx=np.indices(d.shape,dtype=np.float32)
    score=np.zeros_like(d);votes=np.zeros(d.shape,np.uint8);details=[]
    for scale in SCALES:
        radius=extent*scale;directional=[];evidence=[]
        for angle in np.arange(8)*math.pi/4:
            mx=gx+math.cos(angle)*radius;my=gy+math.sin(angle)*radius
            rim=cv2.remap(center,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            rim_weight=cv2.remap(weights,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            margin=rim-center
            directional.append((rim_weight>.80)&(margin>tolerance))
            evidence.append(margin)
        directions=np.stack(directional)
        # Adjacent samples along one bar are one side, not an enclosing pocket.
        separated=np.zeros(d.shape,bool)
        for i in range(8):
            for j in range(i+1,8):
                if min(j-i,8-(j-i))>=3:separated|=directions[i]&directions[j]
        vote=(directions.sum(0)>=2)&separated&safe
        strength=np.sort(np.stack(evidence),axis=0)[-2]
        score+=np.where(vote,np.clip(strength/max(tolerance,1e-6),0,4)/4,0)
        votes+=vote.astype(np.uint8);details.append((directions,np.stack(evidence),radius))
    score=np.where((votes>=2)&safe,score/len(SCALES),0).astype(np.float32)
    # Keep width as a soft preference; true mesh/contact geometry is checked later.
    clearance=cv2.distanceTransform(floor.astype(np.uint8),cv2.DIST_L2,5)
    score*=np.clip(clearance/max(2.,extent*.30),0,1)
    candidates=[];working=score.copy()
    for _ in range(max_candidates):
        _,peak,_,point=cv2.minMaxLoc(working)
        if peak<=.08:break
        cx,cy=point;anchor=[float(l+cx/factor),float(t+cy/factor)]
        evidence=[]
        for directions,margins,radius in details:
            indices=np.flatnonzero(directions[:,cy,cx]).tolist()
            evidence.append({'radius_px':float(radius/factor),'supported_directions':indices,
                'relative_depth_margins':[float(margins[i,cy,cx]) for i in range(8)],
                'rim_points_px':[[anchor[0]+math.cos(i*math.pi/4)*radius/factor,
                                  anchor[1]+math.sin(i*math.pi/4)*radius/factor] for i in indices]})
        candidates.append({'anchor_px':anchor,'score':float(peak),'scale_votes':int(votes[cy,cx]),
            'clearance_px':float(clearance[cy,cx]/factor),'visible_bottom_fraction':float(bottom_weight[cy,cx]),
            'rim_evidence':evidence,'status':'unknown','version':VERSION,
            'reason':'recess_candidate_only; actual_height_and_contact_need_verification'})
        cv2.circle(working,point,max(5,round(extent*.32)),0,-1)
    summary.update(bounds=[l,t,r,b],analysis_size=list(size),trend_coefficients=coeff.tolist(),
        depth_noise=noise,rim_threshold=tolerance,proposed_count=len(candidates),
        reason='multiscale_visible_pockets' if candidates else 'no_consistent_visible_pocket; keep_planar_fallback')
    return candidates,summary


def check_projection(pocket, silhouette, contacts, bounds, support, allowed, protected):
    """Use actual projected silhouette/contact pixels, including holes and thin parts."""
    silhouette=np.asarray(silhouette,bool);contacts=np.asarray(contacts)>0
    if any(np.shape(value)!=silhouette.shape for value in (contacts,support,allowed,protected)):
        raise ValueError('pocket_projection_grids_mismatch')
    support=np.asarray(support,bool);inside=float((silhouette&allowed).sum()/max(1,silhouette.sum()))
    count,labels=cv2.connectedComponents(contacts.astype(np.uint8))
    coverage=[float(support[labels==i].mean()) for i in range(1,count)]
    reasons=[]
    if not coverage or min(coverage)<.9:reasons.append('actual_contact_cluster_below_90_percent')
    if inside<.98:reasons.append('actual_silhouette_outside_allowed')
    if np.any(silhouette&protected):reasons.append('protected_conflict')
    l,t,r,b=bounds;rim_overlaps=0;rim_points=0
    for scale in pocket.get('rim_evidence',[]):
        for x,y in scale['rim_points_px']:
            xx,yy=round(x-l),round(y-t)
            if 0<=xx<silhouette.shape[1] and 0<=yy<silhouette.shape[0]:
                rim_points+=1;rim_overlaps+=int(silhouette[yy,xx])
    return {'status':'fail' if reasons else 'unknown','program_fit':'fail' if reasons else 'pass',
        'contact_coverages':coverage,'inside_fraction':inside,'reasons':reasons,
        'rim_overlap_samples':rim_overlaps,'rim_samples_in_crop':rim_points,
        'occlusion_check_required':bool(rim_overlaps),'hidden_contact_proven':False,
        'basis':'real_mesh_projection_and_visible_bottom; final_metric_contact_still_required'}
