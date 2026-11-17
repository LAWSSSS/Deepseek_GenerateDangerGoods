"""Local, robust illumination evidence and a shared diffuse-receiver calibration.

A photograph is not an HDR measurement. All radiance, depth and direction outputs
remain estimates; remote scene pixels never enter this fit.
"""
import hashlib
import math

import numpy as np


VERSION = 'local-tile-irradiance-v1'
COHERENT_VERSION = 'local-hemisphere-reflection-v1'
LUMA = np.float32([.2126, .7152, .0722])


def clear_reflection_samples(observed, environment, bounds, selected):
    """Remove the same bounded local camera veil used by object compositing.

    The expanded lighting neighbourhood can exceed the fitted veil field. Only
    a short, low-strength edge continuation is allowed there; distant pixels are
    left observed rather than borrowing a scene-wide atmosphere estimate.
    """
    from pathlib import Path
    from .atmosphere import field_crop
    context = environment.get('atmospheric_context')
    if environment.get('integration_mode') != 'coherent' or not context:
        air = environment.get('airborne_dust', {})
        amount = min(.28, max(0., air.get('amount', 0.)))
        air_color = np.asarray(air.get('color_linear', environment['dust_color']), np.float32)
        if environment.get('integration_mode') == 'coherent':
            # Missing local evidence never silently reinstates a global haze fit.
            return observed.copy(), {'source':'missing_local_field_observed_radiance',
                'confidence':'unknown','global_fallback':False,'compensated_fraction':0.}
        clear = np.clip((observed-amount*air_color)/max(.72,1-amount), observed*.65, observed*1.15)
        return clear, {'source':'legacy_local_scalar','amount':amount}
    summary=context['summary'];l,t,r,b=summary['bounds'];x,y,rr,bb=bounds
    with np.load(Path(context['folder'])/'fields.npz') as fields:
        transmission=np.asarray(fields['transmission'],np.float32)
        airlight=np.asarray(fields['airlight'],np.float32)
    if (transmission.ndim!=2 or airlight.shape!=(*transmission.shape,3) or
            not np.isfinite(transmission).all() or not np.isfinite(airlight).all()):
        raise ValueError('invalid_local_atmosphere_reflection_field')
    translated=[x-l,y-t,rr-l,bb-t]
    transmission=field_crop(transmission,translated,(r-l,b-t))
    airlight=field_crop(airlight,translated,(r-l,b-t))
    yy,xx=np.mgrid[y:bb,x:rr].astype(np.float32)
    inside=(xx>=l)&(xx<r)&(yy>=t)&(yy<b)
    distance=np.maximum.reduce([l-xx,xx-(r-1),t-yy,yy-(b-1),np.zeros_like(xx)])
    halo=min(32.,max(8.,min(r-l,b-t)*.08))
    edge_amount=np.minimum(1-np.clip(transmission,.45,1.),.08)*np.clip(1-distance/halo,0,1)
    effective_t=np.where(inside,np.clip(transmission,.45,1.),1-edge_amount)
    corrected=(observed-(1-effective_t[...,None])*airlight)/effective_t[...,None]
    clear=np.clip(corrected,observed*.65,observed*1.15)
    # Unknown exterior evidence may only make a very small radiometric change.
    clear=np.where(inside[...,None],clear,np.clip(clear,observed*.95,observed*1.05))
    samples=selected if selected.any() else np.ones(inside.shape,bool)
    return clear, {'source':'local_atmosphere_transmission_and_airlight_fields',
        'signature':context.get('signature'),'version':summary.get('version'),
        'confidence':summary.get('confidence','unknown'),'field_bounds':[l,t,r,b],
        'sampling_bounds':list(bounds),'covered_sample_fraction':float(inside[samples].mean()),
        'compensated_fraction':float((effective_t[samples]<1).mean()),
        'transmission_range':np.percentile(effective_t[samples],[5,50,95]).tolist(),
        'outside_policy':'edge_halo_max_8pct_veil_5pct_radiance_then_unmodified',
        'outside_halo_px':halo,'radiance_ratio_bounds':[.65,1.15],'global_fallback':False}


def fit_direction(centers, luminance):
    """Fit tile log-brightness, requiring spatial diversity and split-fit agreement."""
    xy = np.asarray(centers, np.float64)
    values = np.log(np.maximum(luminance, .005))
    unknown = {'direction': None, 'confidence': 0., 'basis': 'insufficient_consistent_local_cues'}
    if len(xy) < 8 or np.linalg.cond(np.cov(xy.T)) > 8:
        return unknown
    design = np.column_stack((np.ones(len(xy)), xy))
    weights = np.ones(len(xy))
    for _ in range(5):
        coefficients = np.linalg.lstsq(design * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None)[0]
        residual = values - design @ coefficients
        scale = max(.035, 1.4826 * float(np.median(abs(residual - np.median(residual)))))
        weights = np.minimum(1., 1.5 * scale / np.maximum(abs(residual), .001))
    vector = coefficients[1:]
    strength = float(np.linalg.norm(vector))
    error = float(np.median(abs(residual)))
    split = []
    for phase in (0, 1):
        matrix = design[phase::2]
        if np.linalg.cond(matrix) > 12:
            return unknown
        split.append(np.linalg.lstsq(matrix, values[phase::2], rcond=None)[0][1:])
    agreement = float(split[0] @ split[1] / max(1e-6, np.linalg.norm(split[0]) * np.linalg.norm(split[1])))
    if strength < .10 or error > .18 or error > strength * .6 or agreement < .85:
        return {**unknown, 'residual_log': error, 'split_agreement': agreement}
    return {'direction': (vector / strength).tolist(), 'confidence': float(min(.65, .65 * agreement * (1-error/.3))),
            'basis': 'consistent_local_tile_gradient_not_measured_light', 'residual_log': error,
            'split_agreement': agreement, 'gradient_log': strength}


def estimate_local_light(scene, candidate, environment, *, statistic='median', margin_scale=1., reject_extremes=False):
    import cv2
    from .photography import linear
    if statistic not in ('median','mean') or not 1 <= margin_scale <= 2.5:
        raise ValueError('invalid_local_light_sampling')
    x, y, r, b = environment['local_context']['bounds']
    original_bounds = [x,y,r,b]
    if margin_scale != 1.:
        w,h = scene['size'];left,top,right,bottom=candidate['bounds']
        x=max(0,round(left-(left-x)*margin_scale));y=max(0,round(top-(top-y)*margin_scale))
        r=min(w,round(right+(r-right)*margin_scale));b=min(h,round(bottom+(b-bottom)*margin_scale))
    rgb = np.float32(np.asarray(scene['frame'])[y:b, x:r]) / 255
    gray = rgb @ LUMA
    site = next(s for s in scene['supports'] if s['id'] == candidate['site_id'])
    depth = scene['depth_native'][y:b, x:r]
    support = np.logical_or.reduce([s['mask'][y:b, x:r] for s in scene['supports']
                                   if s['id'] in candidate.get('support_ids', [site['id']])])
    valid = scene['allowed'][y:b, x:r] & ~scene['protected'][y:b, x:r]
    valid &= abs(depth - site['depth']) <= .085
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    valid &= (rgb.max(axis=2) < .93) & (gray > .07) & (spread < .17)
    left, top, right, bottom = candidate['bounds']
    ring = np.ones_like(valid)
    ring[max(0, top-y):min(b-y, bottom-y), max(0, left-x):min(r-x, right-x)] = False
    selected = valid & ring & support
    basis = 'near_same_depth_support_neutral_tiles'
    if selected.sum() < 100:
        selected = valid & ring
        basis = 'near_same_depth_neutral_tiles'
    if selected.sum() < 60:
        selected = valid & support
        basis = 'local_under_object_neutral_support_estimate'
    if selected.sum() < 40:
        raise ValueError('insufficient_local_neutral_light_evidence')
    if statistic == 'median' or reject_extremes:
        lo, hi = np.percentile(gray[selected], [12, 88])
        selected &= (gray >= lo) & (gray <= hi)
    # Erode metal boundaries so adjacent gaps and bright edges cannot bias a tile.
    eroded = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if eroded.sum() >= 40:
        selected = eroded
    observed = linear(rgb)
    clear, veil_compensation = clear_reflection_samples(observed,environment,[x,y,r,b],selected)
    h, w = selected.shape
    tiles, centers, colors, counts = [], [], [], []
    for iy in range(6):
        for ix in range(6):
            l, t, rr, bb = ix*w//6, iy*h//6, (ix+1)*w//6, (iy+1)*h//6
            mask = selected[t:bb, l:rr]
            if mask.sum() < 12:
                continue
            reducer = np.mean if statistic == 'mean' else np.median
            color = reducer(clear[t:bb, l:rr][mask], axis=0)
            center = [(l+rr)/w-1, (t+bb)/h-1]
            colors.append(color); centers.append(center); counts.append(int(mask.sum()))
            tiles.append({'bounds': [x+l,y+t,x+rr,y+bb], 'pixels': int(mask.sum()), 'rgb_linear': color.tolist()})
    if len(tiles) < 3:
        colors = [(np.mean if statistic == 'mean' else np.median)(clear[selected], axis=0)]
        centers = [[0., 0.]]
    # Equal tile voting prevents one large sheet from dominating many smaller pieces.
    color = np.mean(clear[selected],axis=0) if statistic == 'mean' else np.median(colors, axis=0)
    luminance = float(color @ LUMA)
    tint = np.clip(color / max(luminance, .01), .85, 1.15)
    direction = fit_direction(centers, np.asarray(colors) @ LUMA)
    q25, q75 = np.percentile(clear[selected] @ LUMA, [25, 75])
    fill = float(np.clip(q25 / max(q75, .015), .38, .85))
    level = float(np.clip(luminance / .18, .3, 1.8))
    mask = np.zeros(scene['size'][::-1], bool)
    mask[y:b, x:r] = selected
    evidence = {'version': VERSION, 'bounds': [x,y,r,b], 'basis': basis, 'sample_count': int(selected.sum()),
                'sample_hash': hashlib.sha256(np.packbits(mask)).hexdigest(), 'tiles': tiles,
                'target_clear_rgb_linear': color.tolist(), 'direction': direction,
                'level': level, 'fill_ratio': fill, 'depth_tolerance': .085, 'global_fallback': False,
                'assumptions': ['same_depth_neutral_surface', 'receiver_reflectance_0.3', 'unknown_HDR_and_surface_normals']}
    if environment.get('integration_mode') == 'coherent':
        evidence['veil_compensation'] = veil_compensation
    if statistic != 'median' or margin_scale != 1.:
        evidence.update(statistic='arithmetic_mean',margin_scale=margin_scale,original_bounds=original_bounds,
                        aggregation='equal_weight_per_valid_pixel_in_camera_linear_space',
                        version='local-mean-expanded-v2')
    if reject_extremes:evidence.update(version='local-mean-screened-v3',reject_extremes=True)
    result = {**environment, 'world_color': (.20 * level * tint).tolist(), 'light_color': (tint/tint.max()).tolist(),
              'key_direction': direction['direction'], 'light_level': level, 'local_irradiance': evidence,
              'lighting_basis': 'bounded_local_tiles_and_diffuse_receiver_calibration'}
    # Keep air, surface dust and camera characteristics independent of this estimate.
    if environment.get('integration_mode') == 'coherent':
        result = coherent_probe(result, candidate)
    return result, mask


def light_rig(environment, camera, gain=1., shape=(128, 256)):
    from .geometry import illumination_axis
    evidence = environment['local_irradiance']
    if environment.get('integration_mode') == 'coherent':
        return coherent_rig(environment, camera, gain, shape)
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    sky = .55 + .95 * (yy / height) ** 2
    direction = evidence['direction']['direction']
    confidence = evidence['direction']['confidence']
    level, fill = evidence['level'], evidence['fill_ratio']
    if direction:
        axis = np.asarray(illumination_axis(camera, direction), float)
        offset = axis * 4.6
        azimuth = math.atan2(axis[1], axis[0])
        # Broad first-order lobe; never a sharp synthetic reflection strip.
        sky *= 1 + .30 * confidence * np.cos(xx/width*2*math.pi-azimuth)
    else:
        offset = (0., 0., 4.6)
    key = 155 * level * gain
    # Reflected fill must follow the camera-relative observed hemisphere rather
    # than a fixed world +Y direction, which can darken the visible side arbitrarily.
    toward = np.asarray(camera.get('eye',[0,-6,6]),float)-np.asarray(camera.get('target',[0,0,0]),float)
    toward[2]=max(.5,abs(toward[2]))
    toward=tuple(toward/max(np.linalg.norm(toward),1e-6)*4.)
    lights = [(offset, key, 7.5 - 2 * confidence), (toward, key * fill * 1.2, 8.)]
    if evidence.get('reconstruction'):
        # World radiance now supplies directional fill; avoid counting it twice.
        lights = [(offset,power*.40,size*1.15) for offset,power,size in lights]
    return np.float32(sky), lights


def coherent_rig(environment, camera, gain=1., shape=(128, 256)):
    """A broad upper source and restrained fill, retaining material curvature.

    A local brightness gradient is not a measured light. Unknown directions keep
    an azimuthally symmetric sky rather than inventing a studio reflection strip.
    """
    from .geometry import illumination_axis
    evidence = environment['local_irradiance']
    direction = evidence['direction']['direction']
    confidence = float(evidence['direction']['confidence'])
    height, width = shape
    latitude = (np.arange(height)+.5)/height*math.pi-math.pi/2
    z = np.sin(latitude)[:, None]
    # The truck/scrap hemisphere is substantially dimmer than the open sky. The
    # legacy nearly uniform hemisphere and equal frontal fill erased curved form.
    sky = np.broadcast_to(np.where(z >= 0, .38+1.05*z, .12+.26*(z+1)), shape).copy()
    offset = np.asarray(illumination_axis(camera, direction))*4.6 if direction else np.array([0.,0.,4.6])
    toward = np.asarray(camera.get('eye',[0,-6,6]),float)-np.asarray(camera.get('target',[0,0,0]),float)
    toward[2] = max(.5, abs(toward[2]))
    toward *= 4./max(np.linalg.norm(toward),1e-6)
    key = 115*float(evidence['level'])*gain
    fill_ratio = .14+.16*float(np.clip(evidence['fill_ratio'],0,1))
    lights = [(tuple(offset), key, 4.8-1.4*confidence), (tuple(toward), key*fill_ratio, 5.8)]
    controls = environment.get('photographic_light')
    if controls:
        # Absolute, bounded changes to source size and fill; source power stays
        # unchanged when widening a light. These are not surface roughness edits.
        width = float(controls.get('light_width_scale', 1.))
        fill = float(controls.get('fill_gain', 1.))
        if not (.75 <= width <= 1.25 and .75 <= fill <= 1.25):
            raise ValueError('invalid photographic light controls')
        lights = [(position, power*(fill if i else 1.), size*width)
                  for i, (position, power, size) in enumerate(lights)]
    return np.float32(sky), lights


def coherent_probe(environment, candidate):
    """Use bounded local tile contrast for reflection, not photo pixels as albedo.

    The unobserved sky and lower hemisphere remain explicit priors. Local tiles
    only modulate broad lobes; their unknown normals/albedo limit confidence.
    """
    evidence = environment['local_irradiance']
    camera = candidate['camera']
    right, up, _ = np.asarray(camera['basis'], np.float64)
    anchor = np.asarray(candidate['anchor_px'], np.float64)
    x,y,r,b = evidence['bounds']
    height,width = 64,128
    yy,xx = np.mgrid[:height,:width]
    latitude = (yy+.5)/height*math.pi-math.pi/2
    longitude = math.pi-(xx+.5)/width*2*math.pi
    rays = np.stack((np.cos(latitude)*np.cos(longitude),np.cos(latitude)*np.sin(longitude),np.sin(latitude)),axis=-1)
    mean = np.maximum(evidence['target_clear_rgb_linear'],.005)
    numerator = np.zeros((height,width,3),np.float64)
    denominator = np.zeros((height,width),np.float64)
    sectors = set()
    tiles = evidence.get('tiles',[])
    for tile in tiles:
        l,t,rr,bb = tile['bounds']
        delta = (np.array([(l+rr)/2,(t+bb)/2])-anchor)/[max(1,r-x),max(1,b-y)]
        axis = right*delta[0]-up*delta[1]
        axis[2] = 0.
        length = np.linalg.norm(axis)
        if length < .025:continue
        axis /= length
        axis[2] = .12
        axis /= np.linalg.norm(axis)
        weight = np.exp(4.*(rays@axis-1.))
        # Reflectance and irradiance are inseparable from one photograph. Do not
        # let bright steel or a dark gap become an inferred high-energy emitter.
        ratio = np.clip(np.asarray(tile['rgb_linear'])/mean,.45,1.7)-1.
        numerator += weight[...,None]*ratio
        denominator += weight
        sectors.add(int((math.atan2(delta[1],delta[0])+math.pi)/math.pi*4)%8)
    confidence = min(.4,len(sectors)/8*.4)*min(1.,len(tiles)/12)
    z = rays[...,2]
    field = np.where(z>=0,.38+1.05*z,.12+.26*(z+1))[...,None]
    field = field*(1+confidence*numerator/np.maximum(denominator[...,None],1e-6))
    probe = np.float32(np.maximum(field,.02)*np.asarray(environment['world_color']))
    info = {'version':COHERENT_VERSION,'status':'estimated','confidence':confidence,
            'basis':'bounded_nearby_reflected_radiance_with_unobserved_sky_prior',
            'observed_tiles':len(tiles),'observed_sectors':len(sectors),
            'unobserved_sky':'broad_upper_hemisphere_not_measured_HDR',
            'fill_ratio_policy':'restrained_camera_fill_preserves_curved_form',
            'sha256':hashlib.sha256(probe.tobytes()).hexdigest()}
    return {**environment,'reflection_probe':probe.tolist(),
            'local_irradiance':{**evidence,'reflection_field':info},
            'lighting_basis':COHERENT_VERSION}


def receiver_calibration(render_dir, meta, environment):
    """Calibrate one rendered neutral receiver, not each object's brightness histogram."""
    import cv2
    from pathlib import Path
    from .photography import display
    path = Path(meta['part_paths']['without_shadow']) / 'without-shadow-linear0001.exr'
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError('missing_local_light_receiver_reference')
    rgb = raw[..., :3][..., ::-1]
    usable = np.isfinite(rgb).all(axis=2) & ((rgb @ LUMA) > .015)
    if usable.sum() < 40:
        raise ValueError('insufficient_local_light_receiver_reference')
    reducer = np.mean if environment['local_irradiance'].get('statistic')=='arithmetic_mean' else np.median
    sample = reducer(rgb[usable], axis=0)[None, None, :]
    target = np.asarray(environment['local_irradiance']['target_clear_rgb_linear'], np.float32)
    from .compose import srgb
    target_luma = float(srgb(target) @ LUMA)
    low, high = -2., 2.
    for _ in range(16):
        midpoint = (low+high)/2
        if float(display(sample * 2**midpoint)[0,0] @ LUMA) < target_luma:
            low = midpoint
        else:
            high = midpoint
    ev = (low+high)/2
    fitted = float(display(sample*2**ev)[0,0] @ LUMA)
    return {'exposure_ev': ev, 'white_balance': [1.,1.,1.], 'limited': abs(fitted-target_luma)>.025,
            'basis': 'same_light_0.3_diffuse_receiver_not_object_median', 'reference_pixels': int(usable.sum()),
            'target_luma': target_luma, 'fitted_luma': fitted, 'receiver_rgb_linear': sample[0,0].tolist(),
            'uncertainty': 'neutral_reflectance_and_surface_normal_assumptions'}


def reconstruct_probe(environment, candidate):
    """Fit a broad local reflection field, with an explicit unobserved-sky prior.

    This is an appearance-conditioned light estimate, not recovery of measured HDR
    or inversion of unknown scene albedos and normals. No scene pixels become texture.
    """
    evidence = environment['local_irradiance']
    camera=candidate['camera']
    right,up,_=np.asarray(camera['basis'],np.float64)
    target=np.asarray(candidate['anchor_px'],np.float64)
    x,y,r,b=evidence['bounds']
    directions=[];ratios=[];sectors=set()
    mean=np.maximum(evidence['target_clear_rgb_linear'],.005)
    for tile in evidence['tiles']:
        l,t,rr,bb=tile['bounds']
        delta=(np.array([(l+rr)/2,(t+bb)/2])-target)/[max(1,r-x),max(1,b-y)]
        axis=right*delta[0]-up*delta[1]
        axis[2]=0.
        length=float(np.linalg.norm(axis))
        if length<.025:continue
        axis/=length
        axis[2]=.28
        axis/=np.linalg.norm(axis)
        directions.append(axis)
        ratios.append(np.clip(np.asarray(tile['rgb_linear'])/mean,.4,1.8)-1.)
        sectors.add(int((math.atan2(delta[1],delta[0])+math.pi)/math.pi*4)%8)
    height,width=64,128
    yy,xx=np.mgrid[:height,:width]
    # Blender image buffers use a bottom-left origin for equirectangular textures.
    latitude=(yy+.5)/height*math.pi-math.pi/2
    longitude=math.pi-(xx+.5)/width*2*math.pi
    rays=np.stack((np.cos(latitude)*np.cos(longitude),np.cos(latitude)*np.sin(longitude),np.sin(latitude)),axis=-1)
    total=np.zeros((height,width,3),np.float64);weight=np.zeros((height,width),np.float64)
    for axis,ratio in zip(directions,ratios):
        influence=np.exp(3.*(rays@axis-1.))
        total+=influence[...,None]*ratio
        weight+=influence
    confidence=min(.65,len(sectors)/8*.65)*min(1.,len(directions)/12)
    field=1+confidence*total/np.maximum(weight[...,None],1e-6)
    dome=.65+.70*np.clip(rays[...,2],0,1)
    field*=dome[...,None]
    solid_angle=np.cos(latitude)
    field/=np.sum(field*solid_angle[...,None],axis=(0,1))/solid_angle.sum()
    probe=np.float32(field*np.asarray(environment['world_color']))
    info={'method':'local_low_frequency_spherical_lobes', 'status':'estimated',
          'confidence':confidence,'observed_sectors':len(sectors),'observed_tiles':len(directions),
          'shape':[width,height],'lobe_concentration':3.,
          'unobserved_sky':'diffuse_dome_prior_not_measured',
          'absolute_radiance':'unknown','scene_albedo_and_normals':'unknown',
          'sha256':hashlib.sha256(probe.tobytes()).hexdigest(),
          'basis':'mean_local_reflected_radiance_not_measured_incident_HDR'}
    return {**environment,'reflection_probe':probe.tolist(),
            'local_irradiance':{**evidence,'reconstruction':info},
            'lighting_basis':'local_mean_conditioned_low_frequency_environment_reconstruction'}
