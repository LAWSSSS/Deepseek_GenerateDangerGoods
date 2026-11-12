"""A supplied interior width fixes scale; camera and pile heights remain estimated."""
import json
import math
import importlib.metadata
from pathlib import Path
from typing import Literal
import cv2
import numpy as np
from pydantic import Field
from scipy.optimize import least_squares
from .schema import Strict, Point, Unit
from .common import atomic_json, signature, file_hash

VERSION='vehicle-width-projection-v4-risk-preflight'
PREFLIGHT_VERSION='vehicle-projection-preflight-v1'
OBSERVATION_VERSION='vehicle-observed-width-v1'
WIDTH_CONFLICT_LIMIT=.18


class Height(Strict):
    surface_id: str
    below_rim_m: tuple[float,float]
    confidence: Unit
    evidence: str = Field(max_length=140)


class VehicleGeometry(Strict):
    vehicle_class: Literal['semitrailer','uncertain','other']
    confidence: Unit
    rim_visibility: Literal['clear','partial','inferred']
    # Each pair is the two inner rim endpoints of ONE transverse cross-section.
    width_spans: list[tuple[Point,Point]] = Field(min_length=2,max_length=6)
    left_rim: tuple[Point,Point]
    right_rim: tuple[Point,Point]
    heights: list[Height] = Field(max_length=24)
    evidence: str = Field(max_length=400)


class WidthObservation(Strict):
    points: tuple[Point,Point]
    endpoint_visibility: tuple[Literal['visible','occluded','outside'],Literal['visible','occluded','outside']]
    correspondence_basis: Literal['opposing_posts','transverse_member','matched_seams','unknown']
    evidence: str = Field(max_length=180)


class ObservedVehicleGeometry(Strict):
    """An empty observation is valid; a hidden truck rim is not a ruler."""
    vehicle_class: Literal['semitrailer','uncertain','other']
    confidence: Unit
    rim_visibility: Literal['clear','partial','inferred']
    width_observations: list[WidthObservation] = Field(max_length=6)
    left_rim: tuple[Point,Point] | None
    right_rim: tuple[Point,Point] | None
    heights: list[Height] = Field(max_length=24)
    evidence: str = Field(max_length=400)


OBSERVED_PROMPT='''Identify OBSERVED inside-top-rim geometry of this semi-trailer.
The supplied interior width is a real-world length, not evidence that the image
shows both width endpoints. Return width_observations=[] if either endpoint is
hidden/outside or opposite longitudinal stations cannot be matched. Empty output
is correct and does not need repair. Do not use image edges, guessed walls, steel
pile outlines, equal image-y coordinates, or inferred vanishing points as rulers.
Each usable observation must connect two visible INSIDE TOP RIM points at ONE
longitudinal station. Name the opposing posts, matched seams or transverse member
that establishes correspondence, and state endpoint visibility independently.
Give left_rim/right_rim as visible straight longitudinal inner-rim segments only;
use null when unavailable. Never extend a segment into hidden image regions.
The user supplied box length/height do not prove unseen corners are image corners.
For each supplied support ID give a conservative below-rim height interval in
metres, explaining which wall and rim evidence is visible; it remains estimated.
Do not select object position, size or camera focal length.'''


PROMPT='''Locate the INSIDE TOP RIM of this semi-trailer, using the full image.
The user supplied the interior width, not its outside width. Return 3-5 distributed
transverse width_spans, from LEFT inner top rim to RIGHT inner top rim at the SAME
longitudinal station. Use opposing posts/seams to establish correspondence. Do not
pair points merely because their image y coordinate is equal. No steel pile edges,
outside edges, truck chassis or camera text are width endpoints. left_rim/right_rim
are two visible endpoints along the respective straight longitudinal inside rim.
For each supplied support ID estimate a conservative interval of height BELOW that
rim in metres, based on visible walls and the nominal interior height. Zero is rim
height; positive is lower. These heights are estimates, never measured neural depth.
Do not select an object location or object size. Report ambiguity concisely.
The width is USER-SUPPLIED for these backgrounds, not inferred from the picture.
Set rim_visibility=inferred if either rim is not visible; partial if partly hidden.
In that case any spans are explicitly a tentative vehicle boundary estimate, NOT
observed calibration points. Never call an inferred endpoint a reliable ruler.'''


def validate_reference(ref):
    if ref.get('vehicle_class')!='semitrailer':raise ValueError('unsupported vehicle class for this width calibration')
    w=ref.get('internal_width_m')
    if not isinstance(w,(int,float)) or not 1.<w<4.:raise ValueError('invalid interior width in metres')
    source=ref.get('source',{})
    if source.get('kind') not in ('user_supplied','vehicle_class_prior','manufacturer') or not source.get('reference'):
        raise ValueError('vehicle width source required')
    for key,bounds in (('nominal_internal_height_m',(.5,5.)),('nominal_internal_length_m',(2.,25.))):
        value=ref.get(key)
        if value is not None and (not isinstance(value,(int,float)) or not bounds[0]<=value<=bounds[1]):
            raise ValueError('invalid '+key)


def basis(elevation,roll):
    e=math.radians(elevation);r=math.radians(roll)
    right=np.array([1.,0,0]);up=np.array([0,math.sin(e),math.cos(e)])
    forward=np.array([0,math.cos(e),-math.sin(e)])
    return np.stack([right*math.cos(r)+up*math.sin(r),
                     -right*math.sin(r)+up*math.cos(r),forward])


def bound_hits(parameters,lower,upper):
    """Record pose bounds as well as lens/height bounds; narrow probes are fixed."""
    hits=[]
    for index,label in enumerate(('elevation','roll','yaw','focal_length','camera_height')):
        if index==0 and upper[index]-lower[index]<=1:continue
        tolerance=.05 if index<3 else .005
        if min(parameters[index]-lower[index],upper[index]-parameters[index])<tolerance:
            hits.append(label)
    return hits


def projection_preflight(calibration,width_m=None):
    """Classify an estimate without changing its camera or upgrading its evidence.

    Also reads older calibration records. Callers can stop before local physics
    or rendering, while keeping the complete rejected estimate for diagnostics.
    A historical image is not rerated or regenerated by this function.
    """
    health=calibration.get('fit_health',{})
    final=set(health.get('fitted_bound_hits',[]))
    initial=set(health.get('unconstrained_bound_hits',[]))
    risk=[];limitations=[]
    observation=calibration.get('observation_audit')
    if observation is not None and not observation.get('fit_ready',False):
        risk.append('insufficient_observed_width_correspondences')
    if observation is not None and health.get('mode') in (
            'explicit_camera_prior_fallback','single_inferred_width_with_explicit_camera_prior'):
        risk.append('observed_fit_requires_unmeasured_camera_prior')
    try:
        camera=calibration['metric_camera'];K=np.asarray(camera['K'],float);R=np.asarray(camera['R'],float)
        H=np.asarray(calibration['homography_mm_to_px'],float)
        if (K.shape!=(3,3) or R.shape!=(3,3) or H.shape!=(3,3) or
                not all(np.isfinite(v).all() for v in (K,R,H)) or K[0,0]<=0 or K[1,1]<=0 or
                not np.allclose(R@R.T,np.eye(3),atol=1e-5) or np.linalg.det(R)<.999 or abs(np.linalg.det(H))<1e-12):
            risk.append('invalid_metric_camera')
    except (KeyError,TypeError,ValueError):risk.append('missing_metric_camera')
    # Older files did not record roll/yaw bounds. Recover the optimizer's actual
    # limits from their final parameters, without re-fitting or inventing poses.
    for name,limit,label in (('roll_deg',45.,'roll'),('vehicle_yaw_deg',65.,'yaw')):
        value=calibration.get(name)
        if value is not None and (not math.isfinite(float(value)) or abs(float(value))>=limit-.05):final.add(label)
    interval=calibration.get('elevation_prior_interval')
    elevation=calibration.get('elevation_deg')
    if interval and elevation is not None and interval[1]-interval[0]>1:
        if min(float(elevation)-interval[0],interval[1]-float(elevation))<.05:final.add('elevation')
    line_evidence=bool(health.get('rim_line_constraints'))
    if 'rim_edge_evidence' in calibration:line_evidence=bool(calibration['rim_edge_evidence'].get('both_supported'))
    if calibration.get('rim_visibility')=='inferred':line_evidence=False
    if not line_evidence:limitations.append('camera_direction_without_supported_image_rims')
    if final&{'roll','yaw'}:risk.append('final_orientation_at_optimizer_bound')
    if len(final&{'roll','yaw','elevation'})>=2:
        risk.append('multiple_orientation_bounds')
    if not line_evidence and 'elevation' in final and initial&{'roll','yaw','focal_length','camera_height'}:
        risk.append('unobserved_direction_at_compound_bounds')
    relative=health.get('maximum_relative_width_residual')
    if relative is None and width_m and calibration.get('width_residual_m'):
        relative=max(calibration['width_residual_m'])/width_m
    if relative is not None and (not math.isfinite(float(relative)) or float(relative)>WIDTH_CONFLICT_LIMIT):
        risk.append('inconsistent_width_correspondences')
    if not calibration.get('projectable',False):risk.append('camera_not_projectable')
    state='high_risk_projection' if risk else 'renderable_estimate'
    return {'version':PREFLIGHT_VERSION,'state':state,'allow_expensive_steps':not risk,
            'status':'unknown','can_certify_real_scale':False,'risk_codes':list(dict.fromkeys(risk)),
            'limitations':limitations+['camera_and_support_height_remain_estimated'],
            'measurements':{'initial_bound_hits':sorted(initial),'final_bound_hits':sorted(final),
                'maximum_relative_width_residual':relative,'width_conflict_limit':WIDTH_CONFLICT_LIMIT,
                'rim_direction_image_supported':line_evidence,'residual_px':calibration.get('residual_px')},
            'policy':'stop_before_local_physics_and_render' if risk else 'estimated_image_allowed_with_review_only',
            'historical_results_reclassified':False}


def rim_edge_evidence(image_path,left_rim,right_rim,visibility):
    """Audit proposed rims against continuous image edges, without inventing posts.

    A narrow deterministic line search can refine an already visible rim. It
    cannot turn hidden rims or arbitrary pile lines into observed calibration.
    """
    rgb=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if rgb is None:raise ValueError('vehicle_image_unreadable')
    h,w=rgb.shape[:2];factor=min(1.,1280/max(h,w))
    grey=cv2.cvtColor(cv2.resize(rgb,None,fx=factor,fy=factor),cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(grey,60,150)
    distance=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    size=np.array([grey.shape[1]-1,grey.shape[0]-1]);diag=float(np.linalg.norm(size))
    rows=[]
    for proposed in (left_rim,right_rim):
        p=np.asarray(proposed,float)*size;delta=p[1]-p[0];length=float(np.linalg.norm(delta))
        if length<diag*.12:
            rows.append({'supported':False,'reason':'rim_segment_too_short','proposed':proposed});continue
        tangent=delta/length;normal=np.array([-tangent[1],tangent[0]])
        midpoint=p.mean(axis=0);best=None
        for angle in (-2.,-1.,0.,1.,2.):
            a=math.radians(angle);direction=tangent*math.cos(a)+normal*math.sin(a)
            for shift in np.linspace(-diag*.008,diag*.008,17):
                ends=midpoint+normal*shift+np.array([-.5,.5])[:,None]*length*direction
                samples=ends[0]+np.linspace(.04,.96,160)[:,None]*(ends[1]-ends[0])
                xy=np.rint(samples).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<=size[0])&(xy[:,1]>=0)&(xy[:,1]<=size[1])
                hits=np.zeros(len(xy),bool);hits[inside]=distance[xy[inside,1],xy[inside,0]]<=1.5
                blocks=np.array([v.mean() for v in np.array_split(hits,8)])
                score=float(hits.mean()-.15*abs(shift)/(diag*.008)-.015*abs(angle))
                if best is None or score>best[0]:best=(score,ends,float(hits.mean()),int((blocks>=.35).sum()))
        _,ends,coverage,blocks=best
        supported=visibility!='inferred' and coverage>=.55 and blocks>=6
        rows.append({'supported':supported,'proposed':proposed,'refined':np.clip(ends/size,0,1).tolist(),
                     'edge_coverage':coverage,'supported_blocks':blocks,'total_blocks':8,
                     'reason':'continuous_image_edge' if supported else 'insufficient_visible_rim_evidence'})
    return {'method':'bounded_continuous_edge_search','rims':rows,
            'both_supported':all(row['supported'] for row in rows),
            'limitation':'Image edges constrain line location; they do not establish transverse correspondence or vehicle identity.'}


def audit_width_observations(observed,size,edges):
    """Keep visible, independent, ordered transverse observations only.

    Coordinates are never snapped, extended or replaced. Edge agreement checks
    a proposed fact; it does not establish vehicle identity or correspondence.
    """
    size=np.asarray(size,float)-1;diagonal=float(np.linalg.norm(size))
    tolerance=max(3.,diagonal*.004)
    rims=edges.get('rims',[]);rows=[];accepted=[]
    rims_supported=bool(edges.get('both_supported')) and len(rims)==2
    lines=[np.asarray(r.get('refined',r.get('proposed')),float)*size for r in rims] if rims_supported else []
    for index,item in enumerate(observed.get('width_observations',[])):
        points=np.asarray(item['points'],float)*size;reasons=[];stations=[];distances=[]
        if item.get('endpoint_visibility') not in (['visible','visible'],('visible','visible')):
            reasons.append('endpoints_not_both_observed')
        if item.get('correspondence_basis') not in ('opposing_posts','transverse_member','matched_seams') or not item.get('evidence','').strip():
            reasons.append('transverse_station_evidence_missing')
        if observed.get('rim_visibility')=='inferred' or not rims_supported:
            reasons.append('inside_rims_not_image_supported')
        if np.linalg.norm(points[1]-points[0])<size[0]*.15:
            reasons.append('width_span_too_short')
        for point,line in zip(points,lines):
            delta=line[1]-line[0];length=float(np.linalg.norm(delta))
            if length<1:
                reasons.append('degenerate_rim_segment');continue
            station=float(np.dot(point-line[0],delta)/(length*length))
            distance=float(np.linalg.norm(point-(line[0]+station*delta)))
            stations.append(station);distances.append(distance)
            if distance>tolerance:reasons.append('width_endpoint_off_visible_inside_rim')
            if station < -tolerance/length or station > 1+tolerance/length:
                reasons.append('width_endpoint_extrapolated_past_visible_rim')
        if not reasons and any(np.max(np.linalg.norm(points-old['points_px'],axis=1))<=tolerance for old in accepted):
            reasons.append('duplicate_transverse_section')
        row={'index':index,'usable':not reasons,'reasons':list(dict.fromkeys(reasons)),
             'points_px':points.tolist(),'rim_distances_px':distances,'rim_stations':stations,
             'correspondence_basis':item.get('correspondence_basis'),'evidence':item.get('evidence')}
        rows.append(row)
        if not reasons:accepted.append(row)
    # Rim endpoint order is arbitrary. Normalize the right-line direction to the
    # left one before checking crossing correspondences.
    reversed_right=False
    if rims_supported:
        reversed_right=bool(np.dot(lines[0][1]-lines[0][0],lines[1][1]-lines[1][0])<0)
    for i,left in enumerate(accepted):
        for right in accepted[i+1:]:
            a=np.subtract(left['rim_stations'],right['rim_stations'])
            if reversed_right:a[1]*=-1
            if a[0]*a[1]<0:
                for row in (left,right):
                    row['usable']=False
                    if 'crossed_transverse_correspondences' not in row['reasons']:
                        row['reasons'].append('crossed_transverse_correspondences')
    valid=[row for row in rows if row['usable']]
    distributed=False
    if len(valid)>=2:
        centers=np.array([row['points_px'] for row in valid]).mean(axis=1)
        distributed=bool(np.max(np.linalg.norm(centers[:,None]-centers[None,:],axis=2))>=diagonal*.05)
    ready=len(valid)>=2 and distributed
    return {'version':OBSERVATION_VERSION,'status':'unknown','fit_ready':ready,
            'reason':'observed_correspondences_available' if ready else 'no_independent_observed_width_calibration',
            'accepted_indices':[row['index'] for row in valid],'observations':rows,
            'distributed_sections':distributed,'endpoint_tolerance_px':tolerance,
            'coordinates_modified':False,'image_rims_supported':rims_supported,
            'limitation':'Visible-edge and pairing evidence is still unmeasured; camera and support heights remain estimated.'}


def fit_observed_geometry(observed,size,width,elevation,elevation_interval,edges):
    """No inferred spans enter this opt-in fit; missing facts remain explicit."""
    audit=audit_width_observations(observed,size,edges)
    result={'status':'unknown','projectable':False,'reason':audit['reason']}
    if audit['fit_ready']:
        spans=[observed['width_observations'][index]['points'] for index in audit['accepted_indices']]
        try:
            result=camera_fit(spans,size,width,elevation,elevation_interval,
                              rim_lines=[row['refined'] for row in edges['rims']],allow_inferred_prior=False)
        except ValueError as error:result['reason']=str(error)
    result.update(observation_audit=audit,rim_edge_evidence=edges,rim_visibility=observed['rim_visibility'],
                  correspondence_status='observed_unmeasured' if audit['fit_ready'] else 'unavailable')
    result['projection_preflight']=projection_preflight(result,width)
    return result


def camera_fit(spans,size,width,elevation,elevation_interval=None,*,rim_lines=None,allow_inferred_prior=False):
    """Fit equal-width parallel sections; focal/elevation priors resolve weak views.

    No given length or depth value is treated as measured. Report prior dependence.
    """
    size=np.asarray(size,float);pixels=np.asarray(spans,float)*np.subtract(size,1)
    if len(pixels)<2 or np.any(np.linalg.norm(pixels[:,1]-pixels[:,0],axis=1)<size[0]*.15):
        raise ValueError('insufficient_inside_width_spans')
    # Coordinate gauge: the section direction defines world X. Camera azimuth
    # relative to the vehicle is estimated by rotating world points about Z.
    center=size/2;f0=max(size)/(2*math.tan(math.radians(50)/2))
    def decode(p):
        elev,roll,yaw,logf,logh=p;f=math.exp(logf);h=math.exp(logh)
        b=basis(elev,roll);a=math.radians(yaw)
        rot=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1.]])
        b=b@rot
        cv=np.stack([b[0],-b[1],b[2]])
        rays=np.column_stack([(pixels.reshape(-1,2)-center)/f,np.ones(len(pixels)*2)])@cv
        dz=np.minimum(rays[:,2],-.01)
        q=np.array([0,0,h])+rays*(-h/dz)[:,None]
        return f,h,cv,q.reshape(-1,2,3),rays
    def residual(p):
        f,h,cv,q,rays=decode(p);delta=q[:,1]-q[:,0]
        # Longitudinal rims are straight and parallel: equal transverse widths
        # alone previously admitted two drifting, unrelated boundary curves.
        left_x=float(q[:,0,0].mean());right_x=float(q[:,1,0].mean())
        straight=np.r_[(q[:,0,0]-left_x),(q[:,1,0]-right_x)]/width
        rim=[]
        if rim_lines is not None:
            points=np.asarray(rim_lines,float).reshape(-1,2)*np.subtract(size,1)
            rr=np.column_stack([(points-center)/f,np.ones(len(points))])@cv
            qq=np.array([0,0,h])+rr*(-h/np.minimum(rr[:,2],-.01))[:,None]
            rim=np.r_[(qq[:2,0]-left_x),(qq[2:,0]-right_x)]/width
        values=np.r_[((delta[:,:2]-[width,0])/width).ravel(),straight*.6,rim,
            # Priors are explicit, not measurement constraints.
            (p[0]-elevation)/25*.06,(p[3]-math.log(f0))*.05,p[1]/45*.02,
            np.maximum(rays[:,2]+.05,0)*10]
        return values
    span=np.median(np.linalg.norm(pixels[:,1]-pixels[:,0],axis=1))
    h0=f0*width/span*math.sin(math.radians(elevation))
    initial=[elevation,0,0,math.log(f0),math.log(max(1.,h0))]
    low,high=elevation_interval or [max(12,elevation-10),min(85,elevation+10)]
    low=max(10,min(84.9,float(low)));high=min(85,max(low+.1,float(high)))
    initial[0]=float(np.clip(initial[0],low,high))
    lower=[low,-45,-65,math.log(max(size)*.45),math.log(.8)]
    upper=[high,45,65,math.log(max(size)*3),math.log(35.)]
    fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.03,max_nfev=250)
    fitted=fit.x.copy()
    boundary=bound_hits(fitted,lower,upper)
    reasons=[]
    if 'focal_length' in boundary or 'camera_height' in boundary:reasons.append('camera_fit_at_bound')
    if not fit.success:reasons.append('camera_fit_not_converged')
    mode='width_and_parallel_rims_fit'
    if reasons:
        # Unknown correspondences must not drive the lens toward orthographic.
        # Freeze the declared lens/elevation prior and estimate only orientation
        # and distance. Keep residuals and prior dependence visible downstream.
        fixed=initial.copy()
        def restricted(p):
            fixed[1],fixed[2],fixed[4]=p
            return residual(fixed)
        fallback=least_squares(restricted,[fitted[1],fitted[2],initial[4]],
            bounds=([lower[1],lower[2],lower[4]],[upper[1],upper[2],upper[4]]),
            loss='soft_l1',f_scale=.03,max_nfev=250)
        fitted=np.asarray(initial,float);fitted[[1,2,4]]=fallback.x
        mode='explicit_camera_prior_fallback'
    f,h,R,q,rays=decode(fitted)
    delta=q[:,1]-q[:,0];errors=np.linalg.norm(delta[:,:2]-[width,0],axis=1)
    anchor_index=None
    if reasons and allow_inferred_prior:
        # Inferred spans are competing boundary guesses, not independent rulers.
        # Use a declared central anchor instead of averaging conflicting guesses
        # into a deceptively precise distance. Other spans remain diagnostics.
        anchor_index=int(np.argmin(np.linalg.norm(pixels.mean(axis=1)-center,axis=1)))
        dx=float(delta[anchor_index,0])
        if dx<=0:raise ValueError('inside_width_endpoint_order_conflict')
        fitted[4]+=math.log(width/dx)
        f,h,R,q,rays=decode(fitted)
        delta=q[:,1]-q[:,0];errors=np.linalg.norm(delta[:,:2]-[width,0],axis=1)
        mode='single_inferred_width_with_explicit_camera_prior'
        if np.max(errors)>width*WIDTH_CONFLICT_LIMIT:reasons.append('inferred_width_spans_conflict')
    if (np.max(errors)>width*WIDTH_CONFLICT_LIMIT and anchor_index is None) or np.any(rays[:,2]>=-.01) or not .8<=h<=35:
        raise ValueError('inside_width_correspondences_conflict')
    K=np.array([[f,0,center[0]],[0,f,center[1]],[0,0,1.]])
    eye=np.array([0,0,h]);t=-R@eye
    H=K@np.column_stack([R[:,0]/1000,R[:,1]/1000,t])
    # Enforce the known width while retaining each section's fitted longitudinal station.
    mid=q.mean(axis=1);world=np.stack([mid-[width/2,0,0],mid+[width/2,0,0]],axis=1)
    projection=(world.reshape(-1,3)@R.T+t)@K.T
    projection=projection[:,:2]/projection[:,2,None]
    residual_px=float(np.sqrt(np.mean(np.sum((projection-pixels.reshape(-1,2))**2,axis=1))))
    result={'status':'unknown','projectable':True,
        'reason':'single_inferred_width_and_camera_prior_only' if anchor_index is not None else 'camera_and_pile_heights_estimated',
        'homography_mm_to_px':H.tolist(),'metric_camera':{'K':K.tolist(),'R':R.tolist(),
            'eye_m':eye.tolist(),'t_m':t.tolist()},'elevation_deg':float(fitted[0]),
        'roll_deg':float(fitted[1]),'vehicle_yaw_deg':float(fitted[2]),
        'width_residual_m':errors.tolist(),'residual_px':residual_px,
        'image_points_px':pixels.tolist(),'focal_prior_fov_deg':50.,
        'elevation_prior_interval':[low,high],
        'fit_health':{'mode':mode,'fallback_reasons':reasons,'unconstrained_bound_hits':boundary,
                      'fitted_bound_hits':bound_hits(fitted,lower,upper),
                      'unconstrained_focal_px':math.exp(float(fit.x[3])),
                      'unconstrained_elevation_deg':float(fit.x[0]),
                      'unconstrained_roll_deg':float(fit.x[1]),
                      'unconstrained_vehicle_yaw_deg':float(fit.x[2]),
                      'rim_line_constraints':rim_lines is not None,
                      'scale_anchor_index':anchor_index,
                      'scale_anchor_rule':'nearest_image_center_before_generation' if anchor_index is not None else None,
                      'maximum_relative_width_residual':float(errors.max()/width),
                      'longitudinal_straightness_m':float(np.std(q[:,:,0],axis=0).max())},
        'uses_camera_prior':True,'absolute_depth_from_neural_map':False}
    result['projection_preflight']=projection_preflight(result,width)
    return result


def plane_camera(calibration,below):
    result={**calibration};cam={**calibration['metric_camera']}
    eye=np.asarray(cam['eye_m'])+np.array([0,0,below]);R=np.asarray(cam['R']);K=np.asarray(cam['K'])
    t=-R@eye;cam.update(eye_m=eye.tolist(),t_m=t.tolist())
    result.update(metric_camera=cam,homography_mm_to_px=(K@np.column_stack([R[:,0]/1000,R[:,1]/1000,t])).tolist(),
                  height_below_rim_m=below)
    return result


async def prepare(scene,reference,vision,cache,*,observation_mode='legacy'):
    validate_reference(reference)
    if observation_mode not in ('legacy','audited'):raise ValueError('invalid_vehicle_observation_mode')
    audited=observation_mode=='audited'
    prompt=OBSERVED_PROMPT if audited else PROMPT
    schema=ObservedVehicleGeometry if audited else VehicleGeometry
    version=OBSERVATION_VERSION if audited else VERSION
    key=signature(version,file_hash(__file__),np.__version__,importlib.metadata.version('scipy'),
                  file_hash(scene['path']),reference,scene['key'],prompt,vision.endpoint.model_dump())
    path=Path(cache)/'vehicle-scale'/(key+'.json')
    if path.is_file():return json.loads(path.read_text(encoding='utf8'))
    supports=[{'id':s['id'],'anchor':(np.asarray(s['anchor_px'])/np.subtract(scene['size'],1)).tolist(),
               'label':s['label']} for s in scene['supports']]
    answer=await vision.ask('vehicle-width-observation' if audited else 'vehicle-width-geometry',schema,prompt,
        {'vehicle_reference':reference,'supports':supports},[scene['path']],
        image_limits=[(1920,1080)],plain_json=True)
    observed=answer.model_dump()
    if answer.vehicle_class=='other':raise ValueError('vehicle_class_conflicts_with_supplied_reference')
    edges=(rim_edge_evidence(scene['path'],answer.left_rim,answer.right_rim,answer.rim_visibility)
           if answer.left_rim is not None and answer.right_rim is not None else
           {'method':'no_observed_rim_segments','rims':[],'both_supported':False})
    if audited:
        fit=fit_observed_geometry(observed,scene['size'],reference['internal_width_m'],
                                  sum(scene['answer']['elevation'])/2,scene['answer']['elevation'],edges)
    else:
        lines=[r['refined'] for r in edges['rims']] if edges['both_supported'] else None
        fit=camera_fit(answer.width_spans,scene['size'],reference['internal_width_m'],sum(scene['answer']['elevation'])/2,
                       scene['answer']['elevation'],rim_lines=lines,
                       allow_inferred_prior=answer.rim_visibility!='clear' or not edges['both_supported'])
        fit['rim_edge_evidence']=edges
        fit['rim_visibility']=answer.rim_visibility
        fit['correspondence_status']='observed_unmeasured' if answer.rim_visibility=='clear' and edges['both_supported'] and fit['fit_health']['scale_anchor_index'] is None else 'inferred'
        fit['projection_preflight']=projection_preflight(fit,reference['internal_width_m'])
        if answer.rim_visibility!='clear' and fit['fit_health']['scale_anchor_index'] is None:
            fit['reason']='rim_correspondence_camera_and_pile_heights_estimated'
    heights={}
    nominal=float(reference.get('nominal_internal_height_m',2.2))
    for h in answer.heights:
        lo,hi=h.below_rim_m
        if not 0<=lo<=hi<=nominal*1.25:continue
        heights[h.surface_id]={'interval_m':[lo,hi],'nominal_m':(lo+hi)/2,
                              'confidence':h.confidence,'evidence':h.evidence,'status':'estimated'}
    for s in scene['supports']:
        heights.setdefault(s['id'],{'interval_m':[0,nominal],'nominal_m':nominal/2,
            'confidence':0.,'status':'unknown','evidence':'wall-to-support height unavailable; nominal interval retained'})
    result={'version':version,'signature':key,'reference':reference,'observation':observed,
        'calibration':fit,'support_heights':heights,'status':'unknown',
        'observation_mode':observation_mode,
        'projection_preflight':fit['projection_preflight'],
        'width_status':'user_supplied' if reference['source']['kind']=='user_supplied' else 'class_prior',
        'camera_status':'estimated','depth_is_metric':False}
    atomic_json(path,result);return result


def local_scale(scene,site,anchor,profile):
    data=scene['vehicle_scale'];factor=scene.get('_image_scale',1.)
    preflight=projection_preflight(data['calibration'],data.get('reference',{}).get('internal_width_m'))
    # Legacy replay keeps its historical estimated placement semantics. The new
    # coherent executor opts into enforcement before candidate generation.
    if data.get('enforce_projection_preflight') and not preflight['allow_expensive_steps']:
        raise ValueError('high_risk_vehicle_projection:'+','.join(preflight['risk_codes']))
    height=data['support_heights'][site['id']]
    calibration=plane_camera(data['calibration'],height['nominal_m'])
    native=np.asarray(anchor)/factor
    def ppm(cal):
        H=np.asarray(cal['homography_mm_to_px']);p=cv2.perspectiveTransform(native.reshape(1,1,2),np.linalg.inv(H))[0,0]
        q=cv2.perspectiveTransform(np.array([[p,p+[1,0],p+[0,1]]]),H)[0]
        return float(np.linalg.svd((q[1:]-q[0]).T,compute_uv=False).max())*factor
    scale=ppm(calibration)
    interval=sorted(ppm(plane_camera(data['calibration'],v)) for v in height['interval_m'])
    evidence={'version':VERSION,'status':'unknown','basis':'supplied_vehicle_inside_width_and_estimated_camera_support_height',
        'physical_size':profile,'fixed_physical_size':True,'pixels_per_mm':scale,
        'calibration_interval_px_per_mm':interval,'calibration':calibration,'calibration_signature':data['signature'],
        'vehicle_reference':data['reference'],'support_height':height,'references':[],
        'projection_preflight':preflight,
        'rejected_references':[],'anchor_px':list(anchor),'coordinate_image_size':list(scene['size']),
        'size_selected_to_fit':False,'reference_projection_correction':1.}
    target=profile['nominal_mm']*scale
    return [target,target],evidence


def check_projection_interval(scene,candidate,pose,vertices,faces):
    """Risk checks at angle/height endpoints, with the same physical contact anchor.

    Refit the supplied width at each angle. These are uncertainty probes, never
    evidence of an independently measured camera or support surface.
    """
    from .reference_projection import metric_projection
    from .scene import projection_geometry,fitted_masks
    from .pile import support_union
    from .geometry import contact_support
    data=scene['vehicle_scale'];base=data['calibration'];size=scene['size']
    enforce=bool(data.get('enforce_projection_preflight'))
    key=signature(data['signature'],'camera-risk-endpoints-v2-preflight',enforce)
    cached=scene.setdefault('_vehicle_risk_views',{})
    if key not in cached:
        spans=np.asarray(base['image_points_px'])/np.subtract(size,1)
        views=[]
        for elevation in sorted(set(base['elevation_prior_interval'])):
            try:
                view=camera_fit(spans,size,data['reference']['internal_width_m'],elevation,
                                [elevation-.01,elevation+.01])
                gate=view['projection_preflight']
                if not enforce or gate['allow_expensive_steps']:views.append({'elevation':elevation,'calibration':view})
                else:views.append({'elevation':elevation,'reason':'high_risk_camera_interval_probe',
                                   'projection_preflight':gate})
            except ValueError as error:views.append({'elevation':elevation,'reason':str(error)})
        cached[key]=views
    heights=candidate['scale']['support_height']['interval_m']
    points=np.asarray(pose['contact_points'])
    center=(points.min(0)+points.max(0))/2
    rows=[]
    for view in cached[key]:
        for height in sorted(set(heights)):
            row={'elevation':view['elevation'],'height_below_rim_m':height,'status':'unknown'}
            if 'calibration' not in view:
                rows.append({**row,'reason':view['reason'],
                             'projection_preflight':view.get('projection_preflight')});continue
            cal=plane_camera(view['calibration'],height)
            camera,full=metric_projection(cal,candidate['anchor_px'],pose['length_m']*1000,size,center)
            geom=projection_geometry(vertices,faces,pose,0,view['elevation'],candidate['plane_normal'],
                camera=camera,contact_band=candidate['contact_band'],receiver=candidate['receiver'])
            fit=fitted_masks(geom,candidate['anchor_px'],full,scene)
            if fit is None:
                rows.append({**row,'reason':'outside_image'});continue
            x,y,r,b=fit['bounds'];mask=fit['hull']
            cover=contact_support(fit['contact'],support_union(scene,candidate['support_ids'],fit['bounds']))
            protected=bool(np.any(mask&scene['protected'][y:b,x:r]))
            inside=float((mask&scene['allowed'][y:b,x:r]).sum()/max(1,mask.sum()))
            passed=bool(cover and min(cover)>=.9 and not protected and inside>=.98)
            rows.append({**row,'status':'pass' if passed else 'unknown',
                'reason':'estimated_projection_clear' if passed else 'support_or_clearance_uncertain',
                'coverages':cover,'inside_hopper_fraction':inside,'protected_overlap':protected,
                'projected_major_px':float(np.ptp(geom['uv'],axis=0).max()*full)})
    return {'status':'unknown','basis':'supplied_width_angle_and_height_endpoint_probes',
            'reason':'camera_rim_and_support_height_not_measured','checks':rows,
            'elevation_interval':base['elevation_prior_interval'],'height_interval_m':heights}
