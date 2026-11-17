"""Independent wide optical ladders on mean-lit and reconstructed fixed geometry."""
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import atomic_json, file_hash
from .compose import compose
from .local import foreground_dust
from .local_light import receiver_calibration
from .photography import linear
from validate_display_dust import read_rgb
from validate_display_gradients import apply_camera, export_effect, full_preview
from validate_display_local_light import audit_geometry, frame, PROFILE


def compose_wide(folder,row,scene,source,archived,shadow_hash,variants,environment,samples,
                 light_samples,masks,previous,pool_levels,fog_levels,experiment='wide'):
    start=time.monotonic()
    fine=experiment=='fine'
    bounds=row['baseline']['edit_bounds'];x,y,r,b=bounds
    visible=masks[2]
    core=cv2.erode(np.uint8(visible>.9),np.ones((5,5),np.uint8))>0
    pose={**row['pose'],'fixed_alpha_dir':str(source),'fixed_alpha_window':row['archived_window']}
    original=read_rgb(row['image'])
    with tempfile.TemporaryDirectory(prefix='replay-',dir=folder) as temporary:
        a=compose(scene,row['pose'],source,archived,environment,samples,row['params'],row['seed'],
                  Path(temporary)/'baseline.png',capture_local=True)
        if not np.array_equal(read_rgb(a['image']),original) or a['shadow_hash']!=shadow_hash:
            raise ValueError('archived_baseline_did_not_replay')
        audit_geometry(a,row,scene,masks)
        layers=a.pop('local_layers')
        outputs={};details={};rendered_layers={}
        for label,(env,rendered) in variants.items():
            directory,meta,info=rendered
            calibration=receiver_calibration(directory,meta,env)
            calibrated={**env,'calibration':calibration}
            tone={**a['photo'],'exposure_ev':calibration['exposure_ev'],'contrast_gain':1.}
            result=compose(scene,pose,directory,meta,calibrated,samples,row['params'],row['seed'],
                           folder/(label+'.png'),capture_local=True,fixed_tone=tone)
            appearance=result.pop('local_layers')
            sensor={**layers,'shaded':appearance['shaded'],'fixed_sensor':a['photo']}
            delta=linear(appearance['rgb'])-linear(layers['rgb'])
            pixels,optics=apply_camera(sensor,delta,environment,row['params'],row['seed'])
            output=read_rgb(result['image']).copy()
            output[y:b,x:r][visible>0]=pixels[visible>0]
            Image.fromarray(output).save(result['image'],compress_level=2)
            audit=audit_geometry(result,row,scene,masks)
            result['photo']['frozen_sensor']=optics
            titles={'mean':'扩大范围均值照明','probe':'局部近似光场重建','global':'全图平均照明 · 池化关闭 · 追加薄雾关闭'}
            outputs[label]=frame(result['image'],titles[label],bounds,core)
            details[label]={'render':info,'calibration':calibration,'photo':result['photo'],
                            'checks':result['checks'],'audit':audit,'light_evidence':env['local_irradiance']}
            rendered_layers[label]=(appearance,sensor,delta,output)
        active='global' if fine else 'probe'
        appearance,sensor,delta,complete=rendered_layers[active]
        zero=outputs[active]
        zero.update(pool_kernel=1,pool_weight=0.,fog_weight=0.,fog_extra=0.,fog_total=layers['air']['amount'])
        groups={'pool':[zero],'fog':[zero]}
        for kind,levels in (('pool',pool_levels),('fog',fog_levels)):
            for i,value in enumerate(levels,1):
                kernel,weight,fog=(value,1.,0.) if kind=='pool' else (1,0.,value)
                effect,measurements=foreground_dust(appearance['clear_rgb'],appearance['rgb'],visible,
                    environment,layers['air'],row['pose']['target_px'],pool_weight=weight,pool_kernel=kernel,
                    fog_weight=fog,uniform=True,fog_profile='wide')
                title=f'{kernel}×{kernel} 均匀池化' if kind=='pool' else f'追加薄雾 {fog:.0%}'
                item=export_effect(folder,f'{kind}-{i}',title,complete,a,sensor,delta+effect,bounds,
                                   environment,row['params'],row['seed'],core)
                item.update(pool_kernel=kernel,pool_weight=measurements['pool_weight_max'],fog_weight=fog,
                    fog_extra=measurements['fog_extra_max'],fog_total=measurements['fog_total_median'],effects=measurements)
                groups[kind].append(item)
            if len({v['image_hash'] for v in groups[kind]})!=len(levels)+1:
                raise ValueError('identical_optical_levels')
        initial=previous['material'] if fine else previous['groups']['pool'][0]
        if file_hash(initial['image'])!=initial['image_hash']:
            raise ValueError('previous_full_frame_changed')
        diagnostic=np.asarray(scene['frame']).copy()
        diagnostic[light_samples]=np.uint8(diagnostic[light_samples]*.6+np.array([30,210,140])*.4)
        mean_evidence=variants['global' if fine else 'mean'][0]['local_irradiance']
        for box,color in ((mean_evidence['original_bounds'],(240,190,30)),(mean_evidence['bounds'],(20,240,110))):
            lx,ly,rr,bb=box;cv2.rectangle(diagnostic,(lx,ly),(rr-1,bb-1),color,4)
        full_preview(diagnostic,folder/'light-samples.jpg')
        # Preserve the floating probe as data. The user report uses full scene images only.
        if not fine:
            probe=np.asarray(variants['probe'][0]['reflection_probe'],np.float32)
            np.save(folder/'estimated-light-probe.npy',probe)
        result={'number':row['number'],'asset_id':row['asset_id'],'name':row['name'],'background':row['background'],
            'experiment':experiment,'previous':initial,'material':outputs['global' if fine else 'mean'],'groups':groups,
            'lighting_variants':details,'light_evidence':details[active]['light_evidence'],
            'light_samples':str(folder/'light-samples.jpg'),
            'material_info':{'profile':PROFILE if row['asset_id']!='13' else None,
                             'new_light':variants[active][1][2],'calibration':details[active]['calibration']},
            'photo':details[active]['photo'],'checks':details[active]['checks'],'audit':details[active]['audit'],
            'status':'review','seconds':round(time.monotonic()-start,3)}
        if not fine:result['estimated_probe']=str(folder/'estimated-light-probe.npy')
    atomic_json(folder/'result.json',result)
    return result
