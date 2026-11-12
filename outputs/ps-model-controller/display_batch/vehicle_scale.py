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

VERSION='vehicle-width-projection-v2'


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


def camera_fit(spans,size,width,elevation,elevation_interval=None):
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
        values=np.r_[((delta[:,:2]-[width,0])/width).ravel(),
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
    f,h,R,q,rays=decode(fit.x)
    delta=q[:,1]-q[:,0];errors=np.linalg.norm(delta[:,:2]-[width,0],axis=1)
    if np.max(errors)>width*.18 or np.any(rays[:,2]>=-.01):
        raise ValueError('inside_width_correspondences_conflict')
    K=np.array([[f,0,center[0]],[0,f,center[1]],[0,0,1.]])
    eye=np.array([0,0,h]);t=-R@eye
    H=K@np.column_stack([R[:,0]/1000,R[:,1]/1000,t])
    # Enforce the known width while retaining each section's fitted longitudinal station.
    mid=q.mean(axis=1);world=np.stack([mid-[width/2,0,0],mid+[width/2,0,0]],axis=1)
    projection=(world.reshape(-1,3)@R.T+t)@K.T
    projection=projection[:,:2]/projection[:,2,None]
    residual_px=float(np.sqrt(np.mean(np.sum((projection-pixels.reshape(-1,2))**2,axis=1))))
    return {'status':'unknown','projectable':True,'reason':'camera_and_pile_heights_estimated',
        'homography_mm_to_px':H.tolist(),'metric_camera':{'K':K.tolist(),'R':R.tolist(),
            'eye_m':eye.tolist(),'t_m':t.tolist()},'elevation_deg':float(fit.x[0]),
        'roll_deg':float(fit.x[1]),'vehicle_yaw_deg':float(fit.x[2]),
        'width_residual_m':errors.tolist(),'residual_px':residual_px,
        'image_points_px':pixels.tolist(),'focal_prior_fov_deg':50.,
        'elevation_prior_interval':[low,high],
        'uses_camera_prior':True,'absolute_depth_from_neural_map':False}


def plane_camera(calibration,below):
    result={**calibration};cam={**calibration['metric_camera']}
    eye=np.asarray(cam['eye_m'])+np.array([0,0,below]);R=np.asarray(cam['R']);K=np.asarray(cam['K'])
    t=-R@eye;cam.update(eye_m=eye.tolist(),t_m=t.tolist())
    result.update(metric_camera=cam,homography_mm_to_px=(K@np.column_stack([R[:,0]/1000,R[:,1]/1000,t])).tolist(),
                  height_below_rim_m=below)
    return result


async def prepare(scene,reference,vision,cache):
    validate_reference(reference)
    key=signature(VERSION,file_hash(__file__),np.__version__,importlib.metadata.version('scipy'),
                  file_hash(scene['path']),reference,scene['key'],PROMPT,vision.endpoint.model_dump())
    path=Path(cache)/'vehicle-scale'/(key+'.json')
    if path.is_file():return json.loads(path.read_text(encoding='utf8'))
    supports=[{'id':s['id'],'anchor':(np.asarray(s['anchor_px'])/np.subtract(scene['size'],1)).tolist(),
               'label':s['label']} for s in scene['supports']]
    answer=await vision.ask('vehicle-width-geometry',VehicleGeometry,PROMPT,
        {'vehicle_reference':reference,'supports':supports},[scene['path']],
        image_limits=[(1920,1080)],plain_json=True)
    observed=answer.model_dump()
    if answer.vehicle_class=='other':raise ValueError('vehicle_class_conflicts_with_supplied_reference')
    fit=camera_fit(answer.width_spans,scene['size'],reference['internal_width_m'],sum(scene['answer']['elevation'])/2,scene['answer']['elevation'])
    fit['rim_visibility']=answer.rim_visibility
    fit['correspondence_status']='observed_unmeasured' if answer.rim_visibility=='clear' else 'inferred'
    if answer.rim_visibility!='clear':fit['reason']='rim_correspondence_camera_and_pile_heights_estimated'
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
    result={'version':VERSION,'signature':key,'reference':reference,'observation':observed,
        'calibration':fit,'support_heights':heights,'status':'unknown',
        'width_status':'user_supplied' if reference['source']['kind']=='user_supplied' else 'class_prior',
        'camera_status':'estimated','depth_is_metric':False}
    atomic_json(path,result);return result


def local_scale(scene,site,anchor,profile):
    data=scene['vehicle_scale'];factor=scene.get('_image_scale',1.)
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
    key=signature(data['signature'],'camera-risk-endpoints-v1')
    cached=scene.setdefault('_vehicle_risk_views',{})
    if key not in cached:
        spans=np.asarray(base['image_points_px'])/np.subtract(size,1)
        views=[]
        for elevation in sorted(set(base['elevation_prior_interval'])):
            try:
                view=camera_fit(spans,size,data['reference']['internal_width_m'],elevation,
                                [elevation-.01,elevation+.01])
                views.append({'elevation':elevation,'calibration':view})
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
                rows.append({**row,'reason':view['reason']});continue
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
