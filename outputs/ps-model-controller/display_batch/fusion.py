"""Estimate local reflection, contact, and camera response from the source frame."""
import numpy as np
import cv2

from .geometry import project, illumination_axis
from .photography import LUMA, linear


def reflection_probe(frame, valid, candidate, environment):
    """Project broad ground colors into a world-space reflection probe."""
    u, v = np.meshgrid((np.arange(128)+.5)/128, (np.arange(64)+.5)/64)
    azimuth, polar = np.pi*(1-2*u), np.pi*(1-v)
    directions = np.stack((np.sin(polar)*np.cos(azimuth), np.sin(polar)*np.sin(azimuth),
                           np.cos(polar)), axis=-1)
    camera = candidate['camera']
    center = np.asarray(camera['target']).copy()
    center[2] = np.clip(center[2], .12, .48)
    elevation = directions[...,2]
    length = center[2]/np.maximum(-elevation, .09)
    points = center + length[...,None]*directions
    coordinates = project(points.reshape(-1,3),camera).reshape(64,128,2)
    pixels = coordinates*candidate['full_px']+candidate['offset']-.5

    # Masked low-pass field. Individual bars, markings and edges cannot become textures.
    x,y,r,b = candidate['bounds']
    pad = round(candidate['target_px']*1.2)
    l,t = max(0,x-pad),max(0,y-pad)
    frame = frame[t:b+pad,l:r+pad]
    valid = valid[t:b+pad,l:r+pad]
    h,w = frame.shape[:2]
    size = (max(1,w//8),max(1,h//8))
    weight = cv2.resize(valid.astype(np.float32),size,interpolation=cv2.INTER_AREA)
    source = cv2.resize(linear(np.float32(frame)/255)*valid[...,None],size,interpolation=cv2.INTER_AREA)
    sigma = max(1.,min(4.,candidate['target_px']*.025/8))
    weight = cv2.GaussianBlur(weight,(0,0),sigma)
    field = cv2.GaussianBlur(source,(0,0),sigma)/np.maximum(weight[...,None],1e-5)
    mx = np.float32((pixels[...,0]-l+.5)*size[0]/w-.5)
    my = np.float32((pixels[...,1]-t+.5)*size[1]/h-.5)
    sampled = cv2.remap(field,mx,my,cv2.INTER_LINEAR)
    coverage = cv2.remap(weight,mx,my,cv2.INTER_LINEAR)
    ground = (elevation<-.1)&(length<2.)&(coverage>.5)
    baseline = np.maximum(environment['dust_color'],.025)
    calibrated=bool(environment.get('calibration'))
    reflection = np.clip(sampled/baseline,.12,2.4) if calibrated else np.clip(sampled/baseline,.40,1.75)**.65
    confidence = np.clip((coverage-.45)*2,0,1)*np.clip(-elevation*4,0,1)*ground
    neutral = np.asarray(environment['world_color'])
    sky = .35+1.25*np.maximum(elevation,0) if calibrated else .72+.46*np.maximum(elevation,0)
    if environment.get('key_direction') is not None:
        axis=illumination_axis(camera,environment['key_direction'])
        facing=np.sum(directions*axis,axis=-1)
        sky+=.62*environment['light_confidence']*np.exp(-(1-facing)/.31)*np.maximum(elevation,0)
    radiance = neutral*sky[...,None]
    radiance *= 1+(.72*reflection-1)*confidence[...,None]
    return np.round(radiance,5).tolist(),float(ground.sum()/max(1,(elevation<-.1).sum()))


def camera_psf(frame, valid):
    """Estimate edge spread in native pixels from well-defined scene edges."""
    gray = np.float32(frame) @ LUMA / 255
    smooth = cv2.GaussianBlur(gray,(0,0),.55)
    dx = cv2.Sobel(smooth,cv2.CV_32F,1,0,scale=.125)
    dy = cv2.Sobel(smooth,cv2.CV_32F,0,1,scale=.125)
    strength = cv2.magnitude(dx,dy)
    edges = cv2.Canny(np.uint8(np.rint(np.clip(smooth,0,1)*255)),35,85)>0
    safe = cv2.erode(valid.astype(np.uint8),np.ones((9,9),np.uint8))>0
    yy,xx = np.nonzero(edges & safe & (strength>.022))
    stride = max(1,len(xx)//640)
    yy,xx = yy[::stride],xx[::stride]
    if not len(xx):
        return {'sigma_px':.56,'edge_samples':0,'basis':'bounded_camera_fallback'}
    span = np.arange(-4,4.01,.5,dtype=np.float32)
    normals = strength[yy,xx]
    mapx = np.float32(xx[:,None])+dx[yy,xx,None]/normals[:,None]*span
    mapy = np.float32(yy[:,None])+dy[yy,xx,None]/normals[:,None]*span
    profiles = cv2.remap(gray,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    widths = []
    for profile in profiles:
        low,high = np.median(profile[:3]),np.median(profile[-3:])
        if high-low<.075:
            continue
        normalized = (profile-low)/(high-low)
        if not .18<=normalized[len(span)//2]<=.82 or (np.diff(normalized)<-.13).sum()>2:
            continue
        width = np.interp(.9,np.maximum.accumulate(normalized),span)-np.interp(
            .1,np.maximum.accumulate(normalized),span)
        if .65<=width<=4.3:
            widths.append(width)
    if len(widths)<12:
        return {'sigma_px':.56,'edge_samples':len(widths),'basis':'bounded_camera_fallback'}
    return {'sigma_px':float(np.clip(np.percentile(widths,35)/2.563,.33,1.15)),
            'edge_samples':len(widths),'basis':'natural_edge_spread'}


def camera_tones(rgb, alpha, reference, metallic, polymer):
    """Match broad tones while retaining source texture and small specular glints."""
    lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    light = lab[...,0]
    core = alpha>.9
    if core.sum()<20:
        core=alpha>.75
    ref = cv2.cvtColor(np.float32(reference[:,None]/255),cv2.COLOR_RGB2LAB)[:,0,0]
    p50,p90,p99 = np.percentile(ref,[50,90,99])
    sigma=max(2.,min(alpha.shape)*.018)
    weight=cv2.GaussianBlur(alpha,(0,0),sigma)
    low=cv2.GaussianBlur(light*alpha,(0,0),sigma)/np.maximum(weight,1e-6)
    detail = light-low
    metal = np.clip(metallic,0,1) if metallic is not None else np.zeros(alpha.shape,np.float32)
    bias = 5. if polymer else 1.
    offset = float(np.clip((p50+bias-np.median(low[core]))*.73,-16,13))
    base = low+offset
    knee = p90+3.5+3*metal
    excess = np.maximum(base-knee,0)
    base -= excess*(.30+.30*metal)
    # Small highlights can extend above the nearby median without a long clipped band.
    headroom = np.maximum(p99+9-base,2)
    bright_detail=np.maximum(detail,0)
    detail=np.minimum(detail,0)+bright_detail/(1+bright_detail/headroom)
    lab[...,0] = np.clip(base+detail,0,100)
    return np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),0,1),{
        'lightness_offset':offset,'scene_lightness_p90':float(p90),
        'bright_band_fraction':float((excess[core]>0).mean())}


def contact_darkening(foot, support, foreground, target_px, supported):
    contact = (foot>0)&support&~foreground
    if not supported or not contact.any():
        return np.zeros(foot.shape,np.float32)
    distance = cv2.distanceTransform((~contact).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    radius = np.clip(target_px*.009,1.5,8.)
    return np.float32(.30*np.exp(-.5*(distance/radius)**2))*support*(~foreground)


def topology(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    components = int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>=3))
    _, _, empty, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8),connectivity=4)
    h,w = labels.shape
    holes = sum(area>=3 and x>0 and y>0 and x+width<w and y+height<h
                for x,y,width,height,area in empty[1:])
    return components,holes


def optical_layer(rgb, alpha, psf, sharpness):
    """Blur coverage and premultiplied light together, preserving geometric holes."""
    sigma = float(np.sqrt(max(psf**2-.38**2,0))/sharpness)
    premult = linear(rgb)*alpha[...,None]
    original = alpha>.5
    # If a tiny hole limits the measured PSF, retain the strongest safe fraction.
    for trial in (sigma,sigma*.75,sigma*.5,sigma*.25):
        if trial<=.18:continue
        softened = cv2.GaussianBlur(alpha,(0,0),trial,borderType=cv2.BORDER_CONSTANT)
        mask = softened>.5
        overlap = (mask&original).sum()/max(1,(mask|original).sum())
        if overlap>=.995 and topology(mask)==topology(original):
            return (cv2.GaussianBlur(premult,(0,0),trial,borderType=cv2.BORDER_CONSTANT),
                    softened,trial,float(overlap))
    return premult,alpha,0.,1.
