"""Place a settled 3-D render on a verified open surface without 2-D pose fakery."""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from batch_v2 import save_mask
from mesh_checks import placement_gate, support_screening_gate
from metric_scale import silhouette_axes
from photometry import fit_photo
from relief import RELIEF_PRIORITY
from scene_v2 import NeedsReview, read_image


def mesh_view(folder, target, contacts):
    """Scale one real Blender render and every component's projected contact patch together."""
    rgba=np.asarray(read_image(folder/"object.png","RGBA"))
    with Image.open(folder/"with-shadow.png") as im:
        with_shadow=np.asarray(im.convert("RGB"),dtype=np.float32)
    with Image.open(folder/"without-shadow.png") as im:
        without_shadow=np.asarray(im.convert("RGB"),dtype=np.float32)
    def linear(rgb):
        rgb=rgb/255
        return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    brightness_with=cv2.cvtColor(linear(with_shadow),cv2.COLOR_RGB2GRAY)
    brightness_without=cv2.cvtColor(linear(without_shadow),cv2.COLOR_RGB2GRAY)
    shadow=np.clip(1-brightness_with/np.maximum(brightness_without,.01),0,1)
    shadow[rgba[:,:,3]>1]=0
    # Noise outside the rendered shadow is not an edit to the scene.
    shadow=np.where(shadow>.035,shadow,0)
    major=silhouette_axes(rgba[:,:,3])[0]
    if major<=0:
        raise NeedsReview("三维渲染没有物品像素")
    scale=target/major
    size=(max(1,round(rgba.shape[1]*scale)),max(1,round(rgba.shape[0]*scale)))
    # Resize premultiplied RGB so transparent edges cannot pick up dark halos.
    pixels=rgba.astype(np.float32)/255
    pixels[:,:,:3]*=pixels[:,:,3:4]
    pixels=cv2.resize(pixels,size,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    np.divide(pixels[:,:,:3],np.maximum(pixels[:,:,3:4],1/255),out=pixels[:,:,:3])
    pixels=np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)
    shade=cv2.resize(shadow,size,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    bands=[]
    for group in contacts:
        points=np.rint(np.asarray(group,dtype=np.float32)*scale).astype(np.int32)
        if len(points)<1:
            raise NeedsReview("三维部件没有可投影的接触点")
        mask=np.zeros(size[::-1],np.uint8)
        points[:,0]=np.clip(points[:,0],0,size[0]-1)
        points[:,1]=np.clip(points[:,1],0,size[1]-1)
        radius=max(1,round(.008*target))
        for point in points:
            cv2.circle(mask,tuple(point),radius,1,-1)
        count,components=cv2.connectedComponents(mask)
        bands.extend(components==i for i in range(1,count))
    occupied=(pixels[:,:,3]>0)|(shade>0)|np.logical_or.reduce(bands)
    x,y,w,h=cv2.boundingRect(occupied.astype("uint8"))
    if not w or not h:
        raise NeedsReview("三维渲染为空")
    crop=np.s_[y:y+h,x:x+w]
    return pixels[crop],shade[crop],[band[crop] for band in bands]


def mesh_open_candidates(view, target, atlas, folder, eligible, *, limit=18):
    """Propose spaced positions using the actual rendered outline, then save their masks."""
    pixels, _, contacts = view
    h, w = pixels.shape[:2]
    contact_y,contact_x=np.nonzero(np.logical_or.reduce(contacts))
    if not len(contact_x):
        raise NeedsReview("三维物体缺少接触锚点")
    anchor=[float(contact_x.mean()),float(contact_y.mean())]
    sw, sh = np.ceil(np.array([w, h]) / atlas.factor).astype(int)
    silhouette = cv2.resize((pixels[:,:,3]>=128).astype(np.float32), (sw,sh),
                            interpolation=cv2.INTER_AREA) > 0
    stride = max(2, round(.06*target/min(atlas.factor)))
    padding = max(32, round(.14*target))
    proposed = []
    for surface in atlas.surfaces:
        clear = (surface.distance >= max(4,.012*target)/min(atlas.factor)) & (
            atlas.open_clearance(surface) >= max(6,.025*target)/min(atlas.factor))
        if sw > clear.shape[1] or sh > clear.shape[0]:
            continue
        collisions = cv2.matchTemplate((~clear).astype(np.float32), silhouette.astype(np.float32), cv2.TM_CCORR)
        ys, xs = np.nonzero(collisions[::stride,::stride] < .5)
        for x, y in zip(xs*stride, ys*stride):
            left, top = map(int,np.rint([x*atlas.factor[0], y*atlas.factor[1]]))
            crop = (max(0,left-padding),max(0,top-padding),min(atlas.size[0],left+w+padding),
                    min(atlas.size[1],top+h+padding))
            candidate = {"placement_mode":"open", "target_px":target, "crop_px":list(crop),
                         "center_px":[left+w/2,top+h/2],
                         "support_anchor_px":[left+anchor[0],top+anchor[1]],
                         "support_id":f"S{surface.piece}"}
            if not eligible(candidate):
                continue
            context = atlas.position_context(surface,np.s_[y:y+sh,x:x+sw],silhouette,
                                             target/min(atlas.factor),measure_relief=True)
            if context['relief']['kind'] != 'raised':
                rank = (RELIEF_PRIORITY.get(context['relief']['kind'],0),-context['cost'])
                proposed.append((rank,candidate,surface))
    selected = []
    for _, candidate, surface in sorted(proposed,key=lambda item:item[0],reverse=True):
        if any(math.dist(candidate['center_px'], prior['center_px']) < .14*target for prior in selected):
            continue
        ident = f"M{len(selected)+1:02}"
        candidate['id'] = ident
        destination = folder / ident
        destination.mkdir(parents=True,exist_ok=True)
        l,t,r,b = candidate['crop_px']
        support = cv2.resize(surface.mask.astype(np.uint8),atlas.size,
                             interpolation=cv2.INTER_NEAREST)[t:b,l:r]
        Image.fromarray(support*255).save(destination/'support.png')
        Image.new('L',(r-l,b-t)).save(destination/'foreground.png')
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def open_fit(candidate,folder,atlas,frame,view,*,source_visible_fraction,physics_stable,
             pixel_size_review=False,scale_at=None,fixed_anchor=None,screening=False):
    """Search only clear openings. Unknown intersecting scrap is never accepted.

    ``screening=True`` is a support-only diagnostic path.  It requires the
    source visibility to remain unknown, and checks only physics and contacts.
    Occlusion, scale and source visibility are recorded as unknown.
    """
    if screening and source_visible_fraction is not None:
        raise ValueError("screening mode requires source_visible_fraction=None")
    if candidate["placement_mode"]!="open":
        raise NeedsReview("堆内钢件体积与物品的深度关系暂不能可靠核验")
    pixels,shade,bands=view
    target=candidate["target_px"]
    l,t,r,b=candidate["crop_px"]
    support=np.asarray(read_image(folder/candidate["id"]/"support.png","L"))>0
    foreground=np.asarray(read_image(folder/candidate["id"]/"foreground.png","L"))>=128
    clearance=atlas.clearance_crop(candidate["support_id"],candidate["crop_px"])
    plane_distance=cv2.distanceTransform(support.astype("uint8"),cv2.DIST_L2,3)
    solid=pixels[:,:,3]>=128
    h,w=solid.shape
    center=np.asarray(candidate["center_px"],dtype=float)-[l,t]
    ideal=center-[w/2,h/2]
    contact=np.logical_or.reduce(bands)
    cy,cx=np.nonzero(contact)
    if not len(cx):
        raise NeedsReview("三维物体缺少接触锚点")
    contact_offset=np.array([cx.mean(),cy.mean()])
    if fixed_anchor is not None:
        ideal=np.asarray(fixed_anchor,dtype=float)-[l,t]-contact_offset
    step=max(4,round(.05*target))
    best=None
    rejected={}
    def reject(reason):
        rejected[reason]=rejected.get(reason,0)+1
    offsets=(0,) if fixed_anchor is not None else (0,-step,step,-2*step,2*step)
    for dy in offsets:
        for dx in offsets:
            x,y=np.rint(ideal+[dx,dy]).astype(int)
            if x<0 or y<0 or x+w>support.shape[1] or y+h>support.shape[0]:
                reject("outside_crop")
                continue
            if t+y<math.ceil(.09*frame.height) or t+y+h>math.floor(.82*frame.height):
                reject("protected_frame")
                continue
            area=np.s_[y:y+h,x:x+w]
            bed=support[area]
            parts=[float((part&bed).sum()/part.sum()) for part in bands]
            if foreground[area][solid].any():
                reject("foreground_overlap")
                continue
            clearance_pixels = contact if screening else solid
            gap=float(clearance[area][clearance_pixels].min()) if clearance_pixels.any() else 0
            edge_pixels = contact if screening else solid
            edge=float(plane_distance[area][edge_pixels].min()) if edge_pixels.any() else 0
            if gap<max(6,.025*target):
                reject("insufficient_clearance")
                continue
            if edge<max(4,.012*target):
                reject("near_support_edge")
                continue
            scale = None
            scale_status = candidate.get("scene_scale_status","unknown")
            if scale_at is not None:
                location = {**candidate,"support_anchor_px":[l+x+float(cx.mean()),t+y+float(cy.mean())]}
                try:
                    scale = scale_at(location)
                except NeedsReview:
                    reject("scale_rejected")
                    continue
                scale_status = {"estimated":"consistent_estimate","calibrated":"calibrated"}.get(scale["status"],"unknown")
            evidence={"source_visible_fraction":source_visible_fraction,"physics_stable":physics_stable,
                      "component_support":parts,"collision":False,"occlusion_order":"clear",
                      "scene_scale":scale_status}
            if screening:
                # This path has no verified source view or foreground mask.
                # Keep every such check explicitly unknown; never call it clear.
                evidence.update({"source_visible_fraction":None,"collision":"unknown",
                                 "occlusion_order":"unknown","screening_only":True})
                issues=support_screening_gate(evidence)
            else:
                issues=placement_gate(evidence)
                if pixel_size_review:
                    # A human inspection image can use the approved pixel size
                    # with scale still explicitly unknown in its evidence.
                    issues=[issue for issue in issues if issue!="scene_scale_uncertain"]
            if issues:
                reject("gate:"+",".join(issues))
                continue
            context=atlas.native_position_context(candidate["support_id"],l+x,t+y,solid,target,None,
                                                  measure_relief=True)
            if context["relief"]["kind"]=="raised":
                reject("raised_surface")
                continue
            rank=(scale_status!="unknown",RELIEF_PRIORITY.get(context["relief"]["kind"],0),
                  -context["cost"], min(parts),min(1.,gap/(.16*target)))
            if best is None or rank>best["rank"]:
                best={"rank":rank,"origin":[int(l+x),int(y+t)],"pixels":pixels,
                      "shadow":np.where(bed,shade,0),"scene_support":support,"scene_foreground":foreground,
                      "contacts":bands,"evidence":evidence,"context":context,"candidate":candidate,
                      "clearance_px":round(gap,2),"scale":scale,
                      "support_anchor_px":[l+x+float(cx.mean()),t+y+float(cy.mean())]}
    if best is None:
        detail=", ".join(f"{key}:{value}" for key,value in sorted(rejected.items())) or "no_offsets"
        raise NeedsReview("实际三维轮廓、部件接触点或场景尺度未通过空位检查（"+detail+"）")
    return best


def compose_spec(fit,frame,scratch,label,*,protected=()):
    candidate=fit["candidate"]
    x,y=fit["origin"]
    pixels=fit["pixels"]
    h,w=pixels.shape[:2]
    alpha=pixels[:,:,3]
    shade=np.rint(np.clip(fit["shadow"]/.55,0,1)*255).astype(np.uint8)
    # Restrict receiver edits to the actual support, never the rest of the scene.
    l,t,r,b=candidate["crop_px"]
    scratch.mkdir(parents=True,exist_ok=True)
    object_path=scratch/"object.png"
    asset=Image.fromarray(pixels)
    box=asset.getchannel("A").getbbox()
    if box is None:
        raise NeedsReview("三维物品透明度为空")
    asset.crop(box).save(object_path)
    shadow_entry=save_mask(shade,scratch/"shadow.png",[x,y,x+w,y+h],(12,11,10))
    if shadow_entry is None:
        raise NeedsReview("三维物体没有可用的接触光影")
    patch=frame.crop((l,t,r,b))
    support=fit["scene_support"]
    lx,ly=x-l,y-t
    if lx<0 or ly<0 or lx+w>r-l or ly+h>b-t:
        raise NeedsReview("物品或阴影超出允许编辑范围")
    occupied=np.zeros(support.shape,np.uint8)
    occupied[ly:ly+h,lx:lx+w]=alpha
    distance=cv2.distanceTransform((occupied<128).astype("uint8"),cv2.DIST_L2,3)
    nearby=(distance>.04*candidate["target_px"])&(distance<.28*candidate["target_px"]) & support
    nearby &= ~fit.get("scene_foreground",np.zeros_like(support))
    shadow_mask=np.zeros(support.shape,bool)
    shadow_mask[ly:ly+h,lx:lx+w]=fit["shadow"]>.035
    nearby &= ~shadow_mask
    for pl,pt,pr,pb in protected:
        x1,y1=max(0,pl-l),max(0,pt-t)
        x2,y2=min(r-l,pr-l),min(b-t,pb-t)
        if x1<x2 and y1<y2:
            nearby[y1:y2,x1:x2]=False
    tone,photography,grain=fit_photo(patch,nearby,asset,0,(box[3]-box[1],box[2]-box[0]))
    # Retain 3-D shading and the approved color: avoid global blur and aggressive curves.
    tone["blur_px"]=0
    if tone.get("luma_curve"):
        tone["luma_curve"]=[[a,round(.25*a+.75*z,2)] for a,z in tone["luma_curve"]]
    spec={"id":label,"asset":object_path.as_posix(),
          "center":[x+(box[0]+box[2])/2,y+(box[1]+box[3])/2],"width":box[2]-box[0],"angle":0,
          "canvas":list(frame.size),"crop":candidate["crop_px"],"tone":tone,"shade":shadow_entry,
          "occlusion":None,"object_alpha":(scratch/"alpha.png").as_posix(),
          "png":(scratch/"render.png").as_posix(),"psd":(scratch/"render.psd").as_posix(),
          "placement_mode":"open"}
    if grain is not None:
        grain_path=scratch/"grain.png"
        Image.fromarray(grain).convert("RGB").save(grain_path)
        spec["grain"]=grain_path.as_posix()
    return spec,photography
