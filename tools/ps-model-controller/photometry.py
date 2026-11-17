"""Measure and match an object's appearance at its actual output resolution."""
import io

import cv2
import numpy as np
from PIL import Image, ImageCms


def srgb(image):
    profile = image.info.get("icc_profile")
    rgb = image.convert("RGB")
    if profile:
        try:
            return ImageCms.profileToProfile(rgb,ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                             ImageCms.createProfile("sRGB"),outputMode="RGB")
        except (OSError,ValueError,ImageCms.PyCMSError):
            pass
    return rgb


def statistics(rgb, mask):
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    selected = mask & (gray>4) & (gray<250)
    if selected.sum()<60:
        return None
    sample = lab[selected]
    chroma = np.linalg.norm(sample[:,1:],axis=1)
    neutral = sample[chroma<18,1:]
    luma = np.percentile(gray[selected],[20,50,80])
    core = cv2.erode(selected.astype("uint8"),np.ones((3,3),np.uint8))>0
    laplacian = cv2.Laplacian(gray,cv2.CV_32F)
    detail = float(np.median(np.abs(laplacian[core]))) if core.any() else None
    residual = gray-cv2.GaussianBlur(gray,(0,0),.85)
    texture = cv2.GaussianBlur(np.abs(laplacian),(5,5),0)
    flat = core & (texture<=np.quantile(texture[core],.30)) if core.any() else core
    noise = float(np.median(np.abs(residual[flat]))/.6745) if flat.sum()>=30 else None
    data = {"samples":int(selected.sum()),"luma_p20_p50_p80":[round(float(x),2) for x in luma],
            "contrast_p80_p20":round(float(luma[-1]-luma[0]),2),
            "lab_ab":[round(float(x),2) for x in np.median(sample[:,1:],axis=0)],
            "neutral_ab":[round(float(x),2) for x in np.median(neutral,axis=0)] if len(neutral)>=40 else None,
            "chroma_c":round(float(np.median(chroma)),2),
            "detail":round(detail,3) if detail is not None else None,
            "noise":round(noise,3) if noise is not None else None}
    return data,residual[flat]


def fit_photo(frame_crop, support, asset, angle, shape):
    rgba = srgb(asset).convert("RGBA")
    rgba.putalpha(asset.getchannel("A"))
    rotated = rgba.rotate(-angle,resample=Image.Resampling.BICUBIC,expand=True)
    obj = np.asarray(rotated.crop(rotated.getchannel("A").getbbox()).resize(
        (shape[1],shape[0]),Image.Resampling.LANCZOS))
    measured = statistics(np.asarray(srgb(frame_crop)),support)
    original = statistics(np.ascontiguousarray(obj[:,:,:3]),obj[:,:,3]>=190)
    tone = {"brightness":0,"warmth":0,"saturation":0,"blur_px":0}
    if measured is None or original is None:
        return tone,{"status":"insufficient_samples","light_direction":None},None
    scene,residual = measured
    before,_ = original
    curve = [[0,0]]
    for source,target in zip(before["luma_p20_p50_p80"],scene["luma_p20_p50_p80"],strict=True):
        x = round(source)
        if x>=curve[-1][0]+6 and x<=249:
            y = int(np.clip(round(target),max(curve[-1][1]+1,x-40),min(250,x+40)))
            if y>curve[-1][1]:
                curve.append([x,y])
    if len(curve)>1:
        tone["luma_curve"] = curve+[[255,255]]
    if scene["neutral_ab"] is not None and before["neutral_ab"] is not None:
        delta = np.clip(np.subtract(scene["neutral_ab"],before["neutral_ab"]),-8,8)
        tint = cv2.cvtColor(np.array([[[53,*delta]]],dtype=np.float32),cv2.COLOR_LAB2RGB)[0,0]*255
        tone["rgb_cast"] = [round(float(v),1) for v in np.clip(tint-tint.mean(),-12,12)]
    tone["saturation"] = round(float(np.clip((scene["chroma_c"]-before["chroma_c"])*.45,-16,8)))
    if scene["detail"] is not None and before["detail"] is not None:
        excess = before["detail"]/(scene["detail"]+1)
        if excess>1.45:
            tone["blur_px"] = round(min(1.25,.24*(excess-1)),2)
    grain = None
    if scene["noise"] is not None and before["noise"] is not None and scene["noise"]-before["noise"]>.9:
        # Tile only the measured flat-area residual. The PSD clips it to the object.
        values = np.clip((residual-np.median(residual))*2.5,-9,9)
        height,width = support.shape
        grain = np.rint(128+np.resize(values,(height,width))).astype("uint8")
    profile = {"status":"measured","color_space":"sRGB","scene":scene,"object_before":before,
               "light_direction":None,"grain_from_scene":grain is not None}
    return tone,profile,grain


def quiet_photo(frame_crop, support, nearby, asset, x, y, target):
    """Match the surrounding steel's light and color while retaining object form."""
    base=cv2.cvtColor(np.asarray(srgb(frame_crop),dtype=np.float32)/255,cv2.COLOR_RGB2LAB)
    rgba=np.array(asset)
    lab=cv2.cvtColor(rgba[:,:,:3].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    alpha=rgba[:,:,3].astype(np.float32)/255
    core=alpha>.75
    sample_mask=nearby if nearby.sum()>=150 else support
    samples=base[sample_mask]
    if len(samples)<60 or core.sum()<60:
        raise ValueError("可用的物品或承托面像素不足")
    center=np.median(samples,axis=0)
    scene_low,scene_high=np.percentile(samples[:,0],[20,80])
    light_range=float(scene_high-scene_low)

    # Use only the low-frequency light field from the plate, so its texture
    # and edges do not show through the opaque object.
    weight=support.astype(np.float32)
    ambient_sigma=max(14.,.093*target)
    weight_blur=cv2.GaussianBlur(weight,(0,0),ambient_sigma)
    ambient=cv2.GaussianBlur(base*weight[:,:,None],(0,0),ambient_sigma)
    ambient/=np.maximum(weight_blur[:,:,None],.0001)
    ambient[weight_blur<.08]=center
    height,width=alpha.shape
    ambient=ambient[y:y+height,x:x+width]

    form_sigma=max(2.,.016*target)
    light=lab[:,:,0]
    form=cv2.GaussianBlur(light*alpha,(0,0),form_sigma)/np.maximum(
        cv2.GaussianBlur(alpha,(0,0),form_sigma),.0001)
    low,middle,high=np.percentile(form[core],[20,50,80])
    form_gain=min(.65,.81*light_range/max(8.,high-low))
    form_limit=float(np.clip(.73*light_range,6.,21.))
    shape=form_limit*np.tanh((form-middle)*form_gain/form_limit)
    shape=np.where(shape<0,1.18*shape,.90*shape)
    fine=base[:,:,0]-cv2.GaussianBlur(base[:,:,0],(0,0),1.6)
    detail_limit=float(np.clip(np.percentile(np.abs(fine[sample_mask]),87),3.,10.))
    detail=np.clip(.54*(light-form),-1.55*detail_limit,detail_limit)
    # Keep fine pitting from the original object so the muted finish stays opaque.
    micro=cv2.GaussianBlur(light*alpha,(0,0),.95)/np.maximum(cv2.GaussianBlur(alpha,(0,0),.95),.0001)
    pitting=np.clip(.24*(light-micro),-3.2,2.4)
    illumination=center[0]+np.clip(ambient[:,:,0]-center[0],-9.,9.)*.65
    lab[:,:,0]=np.clip(illumination+shape+detail+pitting-2.,0,100)

    chroma=lab[:,:,1:]
    neutral=core&(np.linalg.norm(chroma,axis=2)<18)
    source_cast=np.median(chroma[neutral],axis=0) if neutral.sum()>=50 else np.zeros(2,np.float32)
    local_chroma=np.linalg.norm(samples[:,1:]-center[1:],axis=1)
    cap=float(np.clip(.9*np.percentile(local_chroma,85),4.3,10.))
    tint=.30*(chroma-source_cast)
    tint*=np.minimum(1.,cap/np.maximum(np.linalg.norm(tint,axis=2,keepdims=True),.001))
    lab[:,:,1:]=center[1:]+.58*np.clip(ambient[:,:,1:]-center[1:],-4.,4.)+tint
    rgb=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    rgba[:,:,:3]=np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
    return Image.fromarray(rgba),{
        "background_lab":[round(float(v),2) for v in center],
        "object_form_gain":round(float(form_gain),3),
        "accent_chroma_cap":round(cap,2),
    }
