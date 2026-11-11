"""Reduce the material dust layer while freezing the approved optical treatment."""
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from .common import atomic_json, file_hash
from .compose import compose
from .global_light import estimate_global_mean
from .local import foreground_dust
from .local_light import estimate_local_light
from .photography import linear
from validate_display_dust import read_rgb
from validate_display_gradients import apply_camera
from validate_display_local_light import audit_geometry, frame, PROFILE


def dust_profile(scale):
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('invalid_surface_dust_scale')
    return {**PROFILE,'version':'surface-dust-strength-v2','dust_scale':float(scale)}


def frozen_environment(scene,row,environment,previous):
    local,_=estimate_local_light(scene,row['pose'],environment,statistic='mean',margin_scale=1.75)
    lighting,_=estimate_global_mean(scene,local)
    if lighting['local_irradiance']!=previous['light_evidence']:
        raise ValueError('approved_global_lighting_changed')
    return lighting


def compose_surface_dust(folder,row,scene,source,archived,shadow_hash,environment,samples,masks,
                         previous,lighting,rendered,scale):
    start=time.monotonic()
    approved=previous['groups']['combined'][-1]
    if file_hash(approved['image'])!=approved['image_hash']:
        raise ValueError('approved_image_changed')
    bounds=row['baseline']['edit_bounds'];x,y,r,b=bounds
    visible=masks[2];core=cv2.erode(np.uint8(visible>.9),np.ones((5,5),np.uint8))>0
    pose={**row['pose'],'fixed_alpha_dir':str(source),'fixed_alpha_window':row['archived_window']}
    variant=previous['lighting_variants']['global'];calibration=variant['calibration']
    old_info=variant['render']
    output=folder/'surface-dust.png'
    with tempfile.TemporaryDirectory(prefix='replay-',dir=folder) as temporary:
        tmp=Path(temporary)
        baseline=compose(scene,row['pose'],source,archived,environment,samples,row['params'],row['seed'],
                         tmp/'archive.png',capture_local=True)
        if not np.array_equal(read_rgb(baseline['image']),read_rgb(row['image'])) or baseline['shadow_hash']!=shadow_hash:
            raise ValueError('archive_did_not_replay')
        audit_geometry(baseline,row,scene,masks)
        layers=baseline.pop('local_layers')

        def finish(directory,meta,path):
            value=compose(scene,pose,directory,meta,{**lighting,'calibration':calibration},samples,
                row['params'],row['seed'],path,capture_local=True,
                fixed_tone={**baseline['photo'],'exposure_ev':calibration['exposure_ev'],'contrast_gain':1.})
            appearance=value.pop('local_layers')
            sensor={**layers,'shaded':appearance['shaded'],'fixed_sensor':baseline['photo']}
            delta=linear(appearance['rgb'])-linear(layers['rgb'])
            effect,measured=foreground_dust(appearance['clear_rgb'],appearance['rgb'],visible,environment,
                layers['air'],row['pose']['target_px'],pool_weight=1.,pool_kernel=3,fog_weight=.35,
                uniform=True,fog_profile='wide')
            pixels,optics=apply_camera(sensor,delta+effect,environment,row['params'],row['seed'])
            for key in ('blur_sigma','added_noise_sigma','existing_noise_sigma','optical_iou'):
                if optics[key]!=baseline['photo'][key]:raise ValueError('camera_response_changed')
            full=read_rgb(value['image']).copy();full[y:b,x:r][visible>0]=pixels[visible>0]
            Image.fromarray(full).save(path,compress_level=2)
            audit=audit_geometry(value,row,scene,masks)
            return value,measured,audit

        old,old_effects,_=finish(Path(old_info['render_dir']),old_info['render_meta'],tmp/'approved.png')
        if not np.array_equal(read_rgb(old['image']),read_rgb(approved['image'])):
            raise ValueError('approved_pool3_fog35_did_not_replay')
        if rendered is not None:
            directory,meta,info=rendered
            value,effects,audit=finish(directory,meta,output)
        else:
            info=old_info
            Image.fromarray(read_rgb(approved['image'])).save(output,compress_level=2)
            value={**old,'image':str(output)};effects=old_effects
            audit=audit_geometry(value,row,scene,masks)
        if any(effects[k]!=approved['effects'][k] for k in
               ('pool_kernel','pool_weight_max','fog_extra_max','fog_total_median','fog_color_linear')):
            raise ValueError('approved_optics_changed')
        item=frame(output,f'表面浮灰 ×{scale:g} · 3×3 池化＋35% 薄雾' if rendered else '塑料对照 · 无额外粉尘层',bounds,core)
        item.update(pool_kernel=3,pool_weight=effects['pool_weight_max'],fog_weight=.35,
                    fog_extra=effects['fog_extra_max'],fog_total=effects['fog_total_median'],effects=effects)
        composition={key:value[key] for key in ('image','edit_bounds','alpha_hash','visible_alpha_hash',
                                                'foreground_mask_hash','shadow_hash','checks')}
        changed=int(np.any(read_rgb(output)!=read_rgb(approved['image']),axis=2).sum())
        if (rendered is None and changed) or (rendered is not None and not changed):
            raise ValueError('surface_change_or_plastic_control_invalid')
        details={'render':info,'calibration':calibration,'photo':value['photo'],'checks':value['checks'],
                 'audit':audit,'light_evidence':lighting['local_irradiance']}
    case={key:previous[key] for key in ('number','asset_id','name','background','light_evidence')}
    case.update(experiment='dust',previous=approved,material=item,groups={'surface':[approved,item]},
        surface_dust={'relative_scale':scale,'applied':rendered is not None,
            'profile':dust_profile(scale) if rendered else None,'changed_pixels':changed,
            'basis':'additional_nonmetal_surface_layer_only','paint_relative_scale':.5},
        lighting_variants={'global':details},photo=details['photo'],audit=audit,composition=composition,
        status='review',replayed_controls=1,
        user_confirmed_style={'pool_kernel':3,'fog_extra':.35,'new_material_pending':True},
        seconds=round(time.monotonic()-start,3))
    atomic_json(folder/'result.json',case)
    return case
