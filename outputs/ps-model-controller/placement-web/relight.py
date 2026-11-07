"""Bounded screen-space ambient matching from a user-selected support surface."""
import cv2
import numpy as np
from PIL import Image

from photometry import fit_photo, statistics


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
    tone, photo, _ = fit_photo(frame, mask, obj.crop(bbox), 0, (bbox[3]-bbox[1],bbox[2]-bbox[0]))
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
    arr[:,:,:3] = np.rint(np.clip(rgb,0,255)).astype(np.uint8)
    arr[:,:,3] = original_alpha
    output = Image.fromarray(arr)
    measured = statistics(np.ascontiguousarray(arr[:,:,:3]),original_alpha>=190)
    photo.update({'method':'bounded_local_ambient_and_color', 'support_samples':int(mask.sum()),
                  'screen_gradient_rgb_per_object_width':gradient,
                  'exposure_gain':round(exposure_gain,4),
                  'detail_blur_sigma_px':round(blur,3) if blur>=.15 else 0,
                  'light_direction':'unknown', 'support_verification':'user_selected_pending',
                  'object_after': measured[0] if measured else None})
    return output, photo
