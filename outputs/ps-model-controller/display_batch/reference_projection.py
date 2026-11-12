"""Metric plane/camera calibration and explicitly uncalibrated local estimates."""
import copy
import json
import math
import cv2
import numpy as np
from .common import signature, resolve, file_hash

VERSION = 'reference-projection-v1'


def verified_reference(row):
    source = row.get('source', {})
    return bool(row.get('entity_id') and row.get('feature') and row.get('surface_id') and
                row.get('identity_status') == 'confirmed' and
                source.get('kind') in ('manufacturer', 'measured') and
                source.get('reference') and source.get('sha256') and row.get('unit') == 'mm')


def fit_plane(rows, camera=None):
    """Need noncollinear metric correspondences; a known line alone is insufficient."""
    points=[];pixels=[]
    for r in rows:
        if verified_reference(r):
            world=np.asarray(r.get('plane_points_mm', []), float)
            image=np.asarray(r.get('image_points_px', []), float)
            if world.ndim==2 and world.shape[1]==2 and world.shape==image.shape:
                points.extend(world.tolist());pixels.extend(image.tolist())
    p=np.asarray(points,float);q=np.asarray(pixels,float)
    if len(p)<4 or np.linalg.matrix_rank(p-p.mean(axis=0))<2:
        return {'status':'unknown','reason':'insufficient_noncollinear_metric_correspondences'}
    if len(set(map(tuple,p)))!=len(p) or len(set(map(tuple,q)))!=len(q):
        return {'status':'unknown','reason':'duplicate_calibration_points'}
    H,_=cv2.findHomography(p,q,0)
    if H is None or abs(np.linalg.det(H))<1e-12:
        return {'status':'unknown','reason':'degenerate_plane_calibration'}
    predicted=cv2.perspectiveTransform(p[None],H)[0]
    residual=float(np.sqrt(np.mean(np.sum((predicted-q)**2,axis=1))))
    result={'status':'unknown','homography_mm_to_px':H.tolist(),'residual_px':residual,
            'point_count':len(p),'reason':'intrinsics_unavailable_3d_projection_unverified'}
    if residual>2:
        result['reason']='conflicting_references';return result
    if not camera or camera.get('status')!='measured':return result
    K=np.asarray(camera.get('matrix'),float);dist=np.asarray(camera.get('distortion',[0]*5),float)
    # The current square crop renderer cannot reproduce skew, anisotropic pixels or distortion.
    if K.shape!=(3,3) or abs(K[0,0]/K[1,1]-1)>.005 or abs(K[0,1])>.01 or np.max(abs(dist))>1e-8:
        result['reason']='camera_requires_rectification';return result
    p3=np.column_stack([p/1000,np.zeros(len(p))])
    ok,rvec,tvec=cv2.solvePnP(p3,q,K,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:result['reason']='pose_fit_failed';return result
    reprojection=cv2.projectPoints(p3,rvec,tvec,K,dist)[0][:,0]
    error=float(np.sqrt(np.mean(np.sum((reprojection-q)**2,axis=1))))
    R=cv2.Rodrigues(rvec)[0];eye=(-R.T@tvec).ravel()
    if error>2 or eye[2]<=0 or ((p3@R.T+tvec.ravel())[:,2]<=0).any():
        result['reason']='metric_camera_conflicts_with_plane';return result
    result.update(status='pass',reason=None,residual_px=error,
                  metric_camera={'K':K.tolist(),'R':R.tolist(),'t_m':tvec.ravel().tolist(),
                                 'eye_m':eye.tolist()})
    return result


def audit(scene, registry_path=None):
    external={}
    if registry_path:
        external=json.loads(resolve(registry_path).read_text(encoding='utf8')).get(file_hash(scene['path']), {})
    rows=[];rejected=[];seen=set()
    for site in scene['supports']:
        ref=site.get('reference')
        if not ref:continue
        ends=np.asarray(ref)*np.subtract(scene['size'],1)
        identity=tuple(sorted(tuple(np.round(p,2)) for p in ends))
        if site.get('plane_basis')=='segmentation_only_hidden_height_unknown' or identity in seen:
            rejected.append({'id':site['id'],'reason':'inherited_or_duplicate_entity'});continue
        seen.add(identity)
        rows.append({'id':site['id'],'entity_id':site['id'],'surface_id':site['id'],
                     'feature':site.get('label','unidentified_segment'),'image_points_px':ends.tolist(),
                     'anchor_px':ends.mean(axis=0).tolist(),'depth':site['depth'],
                     'interval_mm':site['reference_mm'],'status':'unknown',
                     'reason':'no_identified_product_or_measured_dimension_source','source':None})
    registered=[];entities=set()
    for row in external.get('references', []):
        if row.get('entity_id') in entities:
            rejected.append({'id':row.get('entity_id'),'reason':'duplicate_physical_entity'});continue
        entities.add(row.get('entity_id'))
        if not verified_reference(row):
            rejected.append({'id':row.get('entity_id'),'reason':'unverified_identity_or_dimension_source'});continue
        registered.append(row)
    planes={}
    for sid in {r['surface_id'] for r in registered}:
        planes[sid]=fit_plane([r for r in registered if r['surface_id']==sid],external.get('camera'))
    result={'version':VERSION,'status':'pass' if any(p['status']=='pass' for p in planes.values()) else 'unknown',
            'references':rows,'verified_references':registered,'rejected_references':rejected,
            'planes':planes,'depth_is_metric':False,'image_size':list(scene['size'])}
    result['signature']=signature(VERSION,scene['key'],result)
    return result


def local_scale(scene,site,anchor,profile):
    # Native evidence is kept separate from the search pyramid. Scale once at its boundary.
    audit_data=scene['reference_audit'];factor=scene.get('_image_scale',1.)
    native_anchor=np.asarray(anchor)/factor
    rows=audit_data['references'];same=[r for r in rows if r['surface_id']==site['id']]
    nearby=same or [r for r in rows if abs(r['depth']-site['depth'])<=.085]
    rejected=list(audit_data['rejected_references'])+[
        {'id':r['id'],'reason':'cross_depth_reference'} for r in rows if r not in nearby]
    calibration=audit_data['planes'].get(site['id'],{})
    plane_verified=calibration.get('status')=='pass'
    basis='same_surface_reference_category_estimate' if same else 'same_layer_category_estimate'
    chosen=[]
    if nearby:
        # Conflicting guesses are never averaged into a precise number. Select the closest
        # entity deterministically, retaining the entire disagreement in the evidence.
        chosen=[min(nearby,key=lambda r:np.linalg.norm(native_anchor-r['anchor_px']))]
        row=chosen[0];ends=np.asarray(row['image_points_px']);d=ends[1]-ends[0]
        elev=math.radians(sum(scene['answer']['elevation'])/2)
        span=float(np.linalg.norm([d[0],d[1]/max(.2,math.sin(elev))]))
        lo,hi=row['interval_mm'];ppm=span/math.sqrt(lo*hi);interval=[span/hi,span/lo]
        selected={**row,'pixels_per_mm':ppm*factor,'interval':[v*factor for v in interval]}
        selected['anchor_px']=(np.asarray(row['anchor_px'])*factor).tolist();chosen=[selected]
    else:
        # Honest estimate only, with explicit wide uncertainty. Not a reference entity.
        span=float(np.ptp(np.asarray(scene['answer']['hopper'])[:,0])*scene['size'][0])/factor
        ppm=span/math.sqrt(2000*5000);interval=[span/5000,span/2000]
        basis='uncalibrated_hopper_width_prior_not_a_measured_ruler'
    if plane_verified:
        H=np.asarray(calibration['homography_mm_to_px']);inv=np.linalg.inv(H)
        world=cv2.perspectiveTransform(native_anchor.reshape(1,1,2),inv)[0,0]
        q=cv2.perspectiveTransform(np.array([[world,world+[1,0],world+[0,1]]]),H)[0]
        ppm=float(np.linalg.svd((q[1:]-q[0]).T,compute_uv=False).max())
        interval=[ppm,ppm];basis='known_plane_and_metric_camera'
    status='pass' if profile['status']=='pass' and plane_verified else 'unknown'
    evidence={'version':VERSION,'status':status,'basis':basis,'fixed_physical_size':True,
        'physical_size':profile,'pixels_per_mm':ppm*factor,
        'calibration_interval_px_per_mm':[v*factor for v in interval],
        'references':chosen,'verified_references':audit_data['verified_references'],
        'rejected_references':rejected,'calibration':calibration,'calibration_signature':audit_data['signature'],
        'anchor_px':list(anchor),'coordinate_image_size':list(scene['size']),
        'size_selected_to_fit':False,'reference_projection_correction':1.}
    target=profile['nominal_mm']*ppm*factor
    return [target,target],evidence


def metric_projection(camera_evidence, anchor, physical_mm, image_size, contact_center):
    """Construct a crop camera from OpenCV's measured plane pose, in normalized model units."""
    H=np.asarray(camera_evidence['homography_mm_to_px'])
    ground=cv2.perspectiveTransform(np.asarray(anchor,float).reshape(1,1,2),np.linalg.inv(H))[0,0]/1000
    cam=camera_evidence['metric_camera'];R=np.asarray(cam['R']);K=np.asarray(cam['K'])
    unit=physical_mm/1000
    translation=np.r_[ground,0]-np.asarray(contact_center)*unit
    eye=(np.asarray(cam['eye_m'])-translation)/unit
    basis=np.stack([R[0],-R[1],R[2]]);full=float(max(image_size))
    return {'eye':eye.tolist(),'target':(eye+R[2]).tolist(),'basis':basis.tolist(),
            'fov':math.degrees(2*math.atan(full/(2*K[0,0])))},full
