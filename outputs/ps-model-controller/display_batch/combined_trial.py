"""Replay the frozen full-frame lighting and add pooling before a fog layer."""
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from .common import atomic_json, file_hash
from .compose import compose
from .global_light import estimate_global_mean
from .local import foreground_dust
from .local_light import estimate_local_light
from .photography import linear
from validate_display_dust import read_rgb
from validate_display_gradients import apply_camera, export_effect
from validate_display_local_light import audit_geometry


def compose_combined(folder,row,scene,source,archived,shadow_hash,environment,samples,masks,previous):
    start=time.monotonic()
    bounds=row['baseline']['edit_bounds'];x,y,r,b=bounds
    visible=masks[2]
    core=cv2.erode(np.uint8(visible>.9),np.ones((5,5),np.uint8))>0
    controls=[previous['material'],next(v for v in previous['groups']['fog'] if v['fog_weight']==.35)]
    if any(file_hash(v['image'])!=v['image_hash'] for v in controls):
        raise ValueError('previous_controls_changed')
    mean_env,_=estimate_local_light(scene,row['pose'],environment,statistic='mean',margin_scale=1.75)
    env,_=estimate_global_mean(scene,mean_env)
    if env['local_irradiance']!=previous['light_evidence']:
        raise ValueError('global_lighting_evidence_changed')
    variant=previous['lighting_variants']['global']
    info=variant['render'];directory=Path(info['render_dir']);meta=info['render_meta']
    pose={**row['pose'],'fixed_alpha_dir':str(source),'fixed_alpha_window':row['archived_window']}
    original=read_rgb(row['image'])
    with tempfile.TemporaryDirectory(prefix='replay-',dir=folder) as temporary:
        tmp=Path(temporary)
        a=compose(scene,row['pose'],source,archived,environment,samples,row['params'],row['seed'],
                  tmp/'archive.png',capture_local=True)
        if not np.array_equal(read_rgb(a['image']),original) or a['shadow_hash']!=shadow_hash:
            raise ValueError('archived_baseline_did_not_replay')
        audit_geometry(a,row,scene,masks)
        layers=a.pop('local_layers')
        calibration=variant['calibration']
        result=compose(scene,pose,directory,meta,{**env,'calibration':calibration},samples,row['params'],
                       row['seed'],tmp/'global.png',capture_local=True,
                       fixed_tone={**a['photo'],'exposure_ev':calibration['exposure_ev'],'contrast_gain':1.})
        audit=audit_geometry(result,row,scene,masks)
        appearance=result.pop('local_layers')
        sensor={**layers,'shaded':appearance['shaded'],'fixed_sensor':a['photo']}
        delta=linear(appearance['rgb'])-linear(layers['rgb'])
        pixels,_=apply_camera(sensor,delta,environment,row['params'],row['seed'])
        complete=read_rgb(result['image']).copy()
        complete[y:b,x:r][visible>0]=pixels[visible>0]
        if not np.array_equal(complete,read_rgb(controls[0]['image'])):
            raise ValueError('full_frame_lighting_zero_did_not_replay')
        # Reproduce the existing fog-only control to validate the frozen optical path.
        only_fog,_=foreground_dust(appearance['clear_rgb'],appearance['rgb'],visible,environment,layers['air'],
            row['pose']['target_px'],pool_weight=0.,fog_weight=.35,uniform=True,fog_profile='wide')
        replay=export_effect(tmp,'fog-control','复核',complete,a,sensor,delta+only_fog,bounds,
                             environment,row['params'],row['seed'],core)
        if not np.array_equal(read_rgb(replay['image']),read_rgb(controls[1]['image'])):
            raise ValueError('fog35_control_did_not_replay')
        effect,measurements=foreground_dust(appearance['clear_rgb'],appearance['rgb'],visible,environment,
            layers['air'],row['pose']['target_px'],pool_weight=1.,pool_kernel=6,fog_weight=.35,
            uniform=True,fog_profile='wide')
        item=export_effect(folder,'pool6-fog35','6×6 均匀池化 ＋ 35% 追加薄雾',complete,a,sensor,delta+effect,
                            bounds,environment,row['params'],row['seed'],core)
        item.update(pool_kernel=6,pool_weight=measurements['pool_weight_max'],fog_weight=.35,
            fog_extra=measurements['fog_extra_max'],fog_total=measurements['fog_total_median'],effects=measurements)
        if item['image_hash'] in {v['image_hash'] for v in controls}:
            raise ValueError('combined_effect_not_distinct')
    case={k:previous[k] for k in ('number','asset_id','name','background','light_evidence','lighting_variants')}
    case.update(experiment='combined',previous=controls[0],material=controls[1],
        groups={'combined':[controls[0],controls[1],item]},audit=audit,status='review',
        reused_render=True,current_render_count=0,replayed_controls=2,
        photo=result['photo'],checks=result['checks'],seconds=round(time.monotonic()-start,3))
    atomic_json(folder/'result.json',case)
    return case
