"""Bounded screen-space ambient matching from a user-selected support surface."""
import cv2
import numpy as np
from PIL import Image

from photometry import fit_photo, statistics


def _srgb_to_linear(value):
    value = float(np.clip(value, 0, 1))
    return value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4


def estimate_render_environment(frame_crop, support, center, size, lighting=None):
    """Estimate a bounded world colour and soft key for the Blender preview.

    The result is a screen-space appearance estimate. It is not a measured
    3-D light direction; callers must keep ``light_direction`` unknown unless
    the rod estimator explicitly returns an estimate.
    """
    support = np.asarray(support, dtype=bool)
    measured = statistics(np.asarray(frame_crop.convert('RGB')), support)
    fallback = {
        'status': 'insufficient_samples',
        'confidence': 'unknown',
        'world_color': [0.32, 0.32, 0.32],
        'world_strength': 0.55,
        'key_direction': [-0.45, -0.60],
        'key_strength': 0.65,
        'fill_strength': 0.25,
        'light_direction': 'unknown',
        'reason': '承托面样本不足，使用中性环境',
    }
    if measured is None:
        return fallback
    scene, _ = measured
    luma = float(scene['luma_p20_p50_p80'][1]) / 255.0
    contrast = float(scene['contrast_p80_p20']) / 255.0
    # Keep the world nearly neutral; a small neutral-AB tint is enough to
    # avoid a pure-white studio look without inventing a coloured light.
    ab = scene.get('neutral_ab') or [0.0, 0.0]
    lab = np.array([[[50.0, float(ab[0]), float(ab[1])]]], dtype=np.float32)
    tint = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0].astype(np.float32)
    tint /= max(float(tint.mean()), 1e-4)
    grey = luma * (0.88 + 0.12 * np.clip(tint, .7, 1.3))
    world = np.rint(np.clip(grey, .18, .42) * 255).astype(np.uint8)
    world_color = [_srgb_to_linear(channel / 255) for channel in world]
    world_strength = float(np.clip(.35 + contrast * .55, .30, .78))

    rods = lighting or {}
    direction = None
    confidence = 'assumed'
    status = 'assumed'
    light_direction = 'unknown'
    if rods.get('status') == 'estimated':
        raw = rods.get('direction_screen') or []
        if len(raw) == 2 and np.all(np.isfinite(raw)) and np.linalg.norm(raw) > 1e-4:
            candidate = np.asarray(raw, dtype=float)
            candidate /= np.linalg.norm(candidate)
            # Reject a nearly vertical/bottom-up cue as a scene light; those
            # cues are more likely bright support pixels than illumination.
            if candidate[1] <= .5:
                direction = candidate
                status = 'estimated'
                confidence = 'low'
                light_direction = 'low_confidence_screen_direction'
    if direction is None:
        rows, cols = np.nonzero(support)
        if len(rows) >= 120:
            cx, cy = center
            side = max(1.0, float(size))
            nx = (cols - cx) / side
            ny = (rows - cy) / side
            if np.percentile(nx, 90) - np.percentile(nx, 10) > .7 and \
                    np.percentile(ny, 90) - np.percentile(ny, 10) > .7:
                design = np.column_stack((np.ones(len(rows)), nx, ny))
                coeff, *_ = np.linalg.lstsq(design, np.asarray(frame_crop.convert('L'))[rows, cols], rcond=None)
                residual = np.asarray(frame_crop.convert('L'))[rows, cols] - design @ coeff
                mad = float(np.median(np.abs(residual - np.median(residual))))
                gradient = coeff[1:]
                if np.linalg.norm(gradient) > max(2.0, mad * .25):
                    candidate = gradient / np.linalg.norm(gradient)
                    # A bright bottom edge usually comes from the support
                    # surface or a shadow, not from a light source below the
                    # scrap pile; reject that low-confidence direction.
                    if candidate[1] <= .25:
                        direction = candidate
                        status = 'gradient_estimate'
                        confidence = 'low'
                        light_direction = 'screen_gradient_assumption'
    if direction is None:
        direction = np.array([-0.45, -0.60], dtype=float)
        direction /= np.linalg.norm(direction)
        status = 'assumed'
        confidence = 'assumed'
    return {
        'status': status,
        'confidence': confidence,
        'world_color': [round(float(v), 5) for v in world_color],
        'world_strength': round(world_strength, 3),
        'key_direction': [round(float(v), 4) for v in direction],
        'key_strength': round(float(np.clip(.55 + contrast * .7, .45, 1.15)), 3),
        'fill_strength': round(float(np.clip(.18 + contrast * .25, .16, .32)), 3),
        'light_direction': light_direction,
        'scene_luma_p20_p50_p80': scene['luma_p20_p50_p80'],
        'scene_neutral_ab': scene.get('neutral_ab'),
    }


def ambient_match(frame, support, foreground, obj, x, y):
    """Return RGB-adjusted RGBA and evidence; never alter the object's alpha.

    Coordinates are in frame pixels. A screen-space brightness gradient is an
    appearance estimate, not a measured 3-D light direction or shadow.
    """
    width, height = frame.size
    side = max(obj.size)
    cx, cy = x + obj.width / 2, y + obj.height / 2
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    mask = np.asarray(support, dtype=bool).copy()
    mask &= ~np.asarray(foreground, dtype=bool)
    mask &= (distance > side*.55) & (distance < side*.9)
    gray = np.asarray(frame.convert('L'))
    if mask.sum() < 100:
        return obj.copy(), {'status':'insufficient_samples', 'reason':'局部承托面像素不足'}
    # Darkest pixels can include cast shadows; saturated highlights are also
    # excluded. Selection of the actual support remains the user's decision.
    lower = max(8, float(np.percentile(gray[mask],20)))
    mask &= (gray >= lower) & (gray < 245)
    bbox = obj.getchannel('A').getbbox()
    if not bbox:
        return obj.copy(), {'status':'insufficient_samples', 'reason':'物体透明度为空'}
    tone, photo, grain = fit_photo(frame, mask, obj.crop(bbox), 0, (bbox[3]-bbox[1],bbox[2]-bbox[0]))
    if photo.get('status') != 'measured':
        return obj.copy(), photo
    arr = np.array(obj, dtype=np.uint8)
    original_alpha = arr[:,:,3].copy()
    rgb = arr[:,:,:3].astype(np.float32)
    curve = tone.get('luma_curve')
    if curve:
        lut = np.interp(np.arange(256), [v[0] for v in curve], [v[1] for v in curve])
        # Avoid flattening fine surface detail when the support is very dark.
        lut = .25*np.arange(256) + .75*lut
        rgb = np.interp(rgb, np.arange(256), lut)
    if tone.get('rgb_cast'):
        rgb += np.asarray(tone['rgb_cast'], dtype=np.float32)
    saturation = float(np.clip(tone.get('saturation',0), -12, 8))
    if saturation:
        hsv = cv2.cvtColor(np.clip(rgb/255,0,1).astype(np.float32),cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*(1+saturation/100),0,1)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)*255

    # A neutral-lit reconstruction can be much brighter than a CCTV scene.
    # The initial bounded curve alone cannot close a large exposure gap, so
    # apply a second bounded global gain while retaining relative material form.
    core=original_alpha>=190
    exposure_gain=1.0
    if core.sum()>=60:
        luminance=cv2.cvtColor(np.clip(rgb,0,255).astype('uint8'),cv2.COLOR_RGB2GRAY)
        current=float(np.median(luminance[core]))
        target=float(photo['scene']['luma_p20_p50_p80'][1])
        exposure_gain=float(np.clip(target/max(current,1),.65,1.2))
        rgb*=exposure_gain

    # Fit only when support samples span both horizontal and vertical sides.
    # The residual-based cutoff rejects weak gradients; their physical cause
    # could equally be paint, shading, or shadows, so the estimate is bounded.
    rows, cols = np.nonzero(mask)
    if len(rows) > 20000:
        stride = int(np.ceil(len(rows)/20000))
        rows, cols = rows[::stride], cols[::stride]
    nx = (cols-cx)/side; ny = (rows-cy)/side
    gradient = None
    if (len(rows)>=300 and np.percentile(nx,90)-np.percentile(nx,10)>.7
            and np.percentile(ny,90)-np.percentile(ny,10)>.7):
        design = np.column_stack((np.ones(len(rows)),nx,ny))
        coeff, *_ = np.linalg.lstsq(design,gray[rows,cols].astype(float),rcond=None)
        residual = gray[rows,cols]-design@coeff
        mad = float(np.median(np.abs(residual-np.median(residual))))
        if np.linalg.norm(coeff[1:]) > max(2, mad*.20):
            gx, gy = np.clip(coeff[1:],-12,12)*.45
            oy, ox = np.mgrid[:obj.height,:obj.width]
            field = np.clip(gx*(x+ox-cx)/side + gy*(y+oy-cy)/side,-6,6)
            rgb += field[:,:,None]
            gradient = [round(float(gx),3), round(float(gy),3)]
    # Match CCTV detail at the final pixel size. Filter premultiplied RGB and
    # divide by the filtered coverage so transparent holes and edges never
    # introduce dark fringes; preserve the original alpha exactly.
    source_detail=photo['object_before'].get('detail')
    scene_detail=photo['scene'].get('detail')
    blur=0.0
    if source_detail is not None and scene_detail is not None:
        ratio=source_detail/(scene_detail+1)
        blur=float(np.clip((ratio-1)*.5,0,.8))
    if blur>=.15:
        coverage=original_alpha.astype(np.float32)/255
        weight=cv2.GaussianBlur(coverage,(0,0),blur)
        premultiplied=cv2.GaussianBlur(rgb.astype(np.float32)*coverage[:,:,None],(0,0),blur)
        softened=premultiplied/np.maximum(weight[:,:,None],1e-5)
        rgb=np.where((weight>1e-5)[:,:,None],softened,rgb)
    grain_applied=False
    if grain is not None and x>=0 and y>=0 and x+obj.width<=width and y+obj.height<=height:
        residual=(grain[y:y+obj.height,x:x+obj.width].astype(np.float32)-128)/2.5
        rgb+=np.clip(residual,-2,2)[:,:,None]
        grain_applied=True
    arr[:,:,:3] = np.rint(np.clip(rgb,0,255)).astype(np.uint8)
    arr[:,:,3] = original_alpha
    output = Image.fromarray(arr)
    measured = statistics(np.ascontiguousarray(arr[:,:,:3]),original_alpha>=190)
    photo.update({'method':'bounded_local_ambient_and_color', 'support_samples':int(mask.sum()),
                  'screen_gradient_rgb_per_object_width':gradient,
                  'exposure_gain':round(exposure_gain,4),
                  'detail_blur_sigma_px':round(blur,3) if blur>=.15 else 0,
                  'grain_from_scene':grain_applied,
                  'light_direction':'unknown', 'support_verification':'user_selected_pending',
                  'object_after': measured[0] if measured else None})
    return output, photo
