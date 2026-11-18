"""Restrained broad illumination with explicit reflectance/normal uncertainty.

Only scene surface observations condition illumination. They never rewrite the
asset's albedo, wear, dust colour, roughness, or material classification.
"""
from pathlib import Path
import numpy as np
from .local_light import estimate_local_light as legacy_estimate, coherent_probe, coherent_rig
LUMA=np.float32([.2126,.7152,.0722])

VERSION='bounded-broad-light-v1'
REFLECTANCE_RANGE=(.15,.45)
NORMAL_RESPONSE_RANGE=(.55,1.)


def estimate_local_light(scene,candidate,environment):
    from .atmosphere import regions_mask
    hints=scene.get('atmosphere',{}).get('summary',{}).get('visual_evidence',{})
    protected=np.asarray(scene['protected'],bool)|regions_mask(hints.get('exclude_regions',[]),np.asarray(scene['protected']).shape)
    working={**scene,'protected':protected}
    local_input={**environment,'integration_mode':'coherent'}
    try:
        result,mask=legacy_estimate(working,candidate,local_input,statistic='mean',margin_scale=1.75,reject_extremes=True)
    except ValueError as error:
        if str(error)!='insufficient_local_neutral_light_evidence':
            raise
        mask=np.zeros(np.asarray(scene['protected']).shape,bool)
        evidence={'bounds':list(environment['local_context']['bounds']),'sample_count':0,
            'tiles':[],'target_clear_rgb_linear':[.18]*3,'direction':{'direction':None,'confidence':0.},
            'level':1.,'fill_ratio':.6,'basis':'unobserved_local_light_soft_prior','global_fallback':False}
        result={**environment,'local_irradiance':evidence,'light_level':1.,'world_color':[.2]*3,'light_color':[1.]*3}
    evidence=dict(result['local_irradiance'])
    tiles=evidence.get('tiles',[])
    # Local pixel means are retained within each tile. Trimming across spatially
    # separate tiles avoids one broad white sheet establishing the entire light.
    colors=np.float32([t['rgb_linear'] for t in tiles])
    if len(colors)>=5:
        lum=colors@LUMA
        lo,hi=np.quantile(lum,[.15,.85])
        selected=colors[(lum>=lo)&(lum<=hi)]
        color=selected.mean(axis=0)
    else:
        color=np.asarray(evidence['target_clear_rgb_linear'],np.float32)
    luminance=max(.005,float(color@LUMA))
    rho=float(np.sqrt(np.prod(REFLECTANCE_RANGE)))
    normal=float(np.sqrt(np.prod(NORMAL_RESPONSE_RANGE)))
    level=float(np.clip(luminance/(rho*normal),.3,1.8))
    # These are admissible illumination scales, not a claim of a calibrated lux value.
    interval=[luminance/(REFLECTANCE_RANGE[1]*NORMAL_RESPONSE_RANGE[1]),
              luminance/(REFLECTANCE_RANGE[0]*NORMAL_RESPONSE_RANGE[0])]
    direction=dict(evidence['direction'])
    direction['confidence']=min(.45,float(direction.get('confidence',0)))
    if len(tiles)<8 or direction['confidence']<.20:
        direction.update(direction=None,confidence=0.,basis='soft_prior_insufficient_independent_surface_cues')
    tint=np.clip(color/luminance,.92,1.08)
    evidence.update(version=VERSION,confidence='estimated' if len(tiles)>=5 else 'unknown',
        target_clear_rgb_linear=color.tolist(),level=level,direction=direction,
        reflectance_range=list(REFLECTANCE_RANGE),normal_response_range=list(NORMAL_RESPONSE_RANGE),
        irradiance_scale_range=interval,material_independent=True,metric_illumination_known=False,
        aggregation='trimmed_equal_spatial_tile_votes_after_screened_pixel_means',
        assumptions=['local_surface_reflectance_interval','unknown_normals','bounded_veil_compensation','unobserved_sky_prior'])
    # A photograph still cannot distinguish albedo changes from illumination.
    result.update(world_color=(.20*level*tint).tolist(),light_color=(tint/tint.max()).tolist(),
        light_level=level,key_direction=direction['direction'],local_irradiance=evidence,
        lighting_basis=VERSION,integration_mode='coherent',realism_light=True)
    result=coherent_probe(result,candidate)
    result['lighting_basis']=VERSION
    result['local_irradiance']['reflection_field']['absolute_radiance']='unknown'
    result['local_irradiance']['reflection_field']['basis']='low_frequency_local_reflection_with_material_uncertainty'
    return result,mask


def light_rig(environment,camera,gain=1.,shape=(128,256)):
    """No narrow reflection strips; reuse a bounded sky and broad sources."""
    sky,lights=coherent_rig(environment,camera,gain,shape)
    # Keep both emitters broad even when a scene gradient happens to be strong.
    minimum = 4.8 * float(environment.get('photographic_light',{}).get('light_width_scale',1.))
    lights=[(position,power,max(minimum,size)) for position,power,size in lights]
    return sky,lights


def receiver_calibration(render_dir,meta,environment):
    """Fit a virtual neutral receiver to an interval, never an object's histogram.

The rendered receiver is 0.3 diffuse by construction; the real observed steel is
not assumed to have that reflectance. When zero EV already lies in its plausible
interval, no exact matching is forced. Additional correction is at most 0.5 EV.
"""
    import cv2
    from .photography import display
    path=Path(meta['part_paths']['without_shadow'])/'without-shadow-linear0001.exr'
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError('missing_realism_receiver_reference')
    rgb=raw[...,:3][...,::-1]
    valid=np.isfinite(rgb).all(axis=2)&((rgb@LUMA)>.015)
    if valid.sum()<40:
        raise ValueError('insufficient_realism_receiver_reference')
    sample=np.mean(rgb[valid],axis=0)[None,None,:]
    evidence=environment['local_irradiance']
    target=np.asarray(evidence['target_clear_rgb_linear'],np.float32)
    rho_low,rho_high=evidence.get('reflectance_range',REFLECTANCE_RANGE)
    # Camera-decoded photograph and scene-linear render are not interchangeable:
    # compare after exactly the same scene display conversion as object rendering.
    def srgb(x):
        return np.clip(np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-.055),0,1)
    target_low=float(srgb(target*(.30/rho_high))@LUMA)
    target_high=float(srgb(target*(.30/rho_low))@LUMA)
    original=float(display(sample)[0,0]@LUMA)
    target_luma=float(np.clip(original,target_low,target_high))
    low,high=-.5,.5
    ev=0.
    if not target_low<=original<=target_high:
        for _ in range(16):
            midpoint=(low+high)/2
            if float(display(sample*2**midpoint)[0,0]@LUMA)<target_luma:
                low=midpoint
            else:
                high=midpoint
        ev=(low+high)/2
    fitted=float(display(sample*2**ev)[0,0]@LUMA)
    limited=not target_low-.025<=fitted<=target_high+.025
    return {'version':VERSION,'exposure_ev':ev,'white_balance':[1.,1.,1.],
        'limited':limited,'requires_lighting_revision':limited,
        'basis':'virtual_receiver_with_scene_reflectance_interval_no_object_histogram',
        'reference_pixels':int(valid.sum()),'receiver_rgb_linear':sample[0,0].tolist(),
        'target_luma':target_luma,'target_luma_interval':[target_low,target_high],
        'fitted_luma':fitted,'reflectance_range':[rho_low,rho_high],
        'uncertainty':'scene_reflectance_normals_and_camera_response_unmeasured',
        'material_independent':True,'extra_exposure_bounds':[-.5,.5]}
