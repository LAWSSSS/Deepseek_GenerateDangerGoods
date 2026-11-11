"""Explicit user asset choices; never mutate source geometry or asset caches."""
import copy
import numpy as np
from .scene import projection_geometry, fitted_masks
from .pile import support_union, receiver_mesh
from .geometry import contact_support


def asset_id(value):
    head,*tail=str(value).split('-')
    return str(int(head))+('-'+tail[0] if tail else '')


def resize_candidate(asset,scene,candidate,factor):
    previous=candidate.get('user_scale',{})
    old_factor=previous.get('factor',1.)
    if factor==old_factor:return candidate
    c=copy.deepcopy(candidate)
    step=factor/old_factor
    data=np.load(asset['geometry'])
    geom=projection_geometry(data['vertices'],data['faces'],{'matrix':c['matrix']},
        c.get('yaw',0),c['elevation'],camera=c.get('camera'))
    full=c['full_px']*step;fit=fitted_masks(geom,c['anchor_px'],full,scene)
    if fit is None:raise ValueError('resized_pose_outside_image')
    c['user_scale']={'factor':factor,'basis':'explicit_user_display_size_not_physical_measurement',
        'previous_factor':old_factor,'relative_factor':step,
        'previous_expected_major_px':c['expected_major_px'],
        'original_expected_major_px':previous.get('original_expected_major_px',c['expected_major_px']),
        'original_full_px':previous.get('original_full_px',c['full_px']),
        'original_anchor_px':previous.get('original_anchor_px',list(c['anchor_px'])),
        'previous_anchor_px':list(c['anchor_px'])}
    c.update(full_px=full,offset=fit['offset'],bounds=fit['bounds'],
        target_px=max(fit['bounds'][2]-fit['bounds'][0],fit['bounds'][3]-fit['bounds'][1]),
        expected_major_px=float(np.ptp(geom['uv'],axis=0).max()*full),anchor_uv=geom['anchor'].tolist(),
        contact_polygons=[p.tolist() for p in geom['contacts']])
    c['scale']={**c.get('scale',{}),'status':'unknown','user_factor':factor,
        'basis':'user_display_scale_requires_local_reference_revalidation'}
    if 'world_pixels_per_unit' in c['scale']:c['scale']['world_pixels_per_unit']*=step
    c['uncertainties']=list(dict.fromkeys(c.get('uncertainties',[])+['user_scale_reference_unknown']))
    coverage=contact_support(fit['contact'],support_union(scene,c['support_ids'],fit['bounds']))
    c['contact_min']=min(coverage) if coverage else 0.
    c.pop('terrain_physics',None)
    c['receiver']=receiver_mesh(scene,c)
    return c
