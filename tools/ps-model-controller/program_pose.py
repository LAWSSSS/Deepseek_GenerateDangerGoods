"""Fit an isolated object photo to a verified support, using only local pixels."""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from batch_v2 import save_mask
from metric_scale import silhouette_axes
from photometry import fit_photo, quiet_photo
from relief import RELIEF_PRIORITY, relief_value
from scene_v2 import (NeedsReview, boundary_agreement, contact_band, crossing,
                      original_edge_neighborhood, read_image)


# Load-bearing regions in the normalized, opaque bounds of these four input photos.
# The optional axis is the centerline of the shock body or cylinder, independent of accessories.
PARTS = {
    "asset-928d815bf447": (None, (
        ((0,0,1,1), (0,.63,.26,.94)), ((0,0,1,1), (.70,.76,1,1)))),
    "asset-4f18e23e2b38": (None, (
        ((0,0,.625,1), (0,.82,.625,1)), ((.625,0,1,1), (.625,0,1,1)))),
    "asset-9ade180e0f43": (((.20,.11),(.20,.90)), (
        ((0,0,.50,1), (0,.31,.50,.75)), ((.55,.78,1,1), (.55,.78,1,1)))),
    "asset-cd0a8343411b": (((.07,.94),(.90,.105)), (
        ((0,0,1,1), (.14,.40,.63,.88)),)),
}


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))


def isolated_asset(path):
    rgba=read_image(path,"RGBA")
    alpha=np.asarray(rgba.getchannel("A"))
    opaque=alpha>=128
    if not .025<=opaque.mean()<=.85:
        raise NeedsReview("需要带透明背景的独立物品图")
    distance=cv2.distanceTransform((~opaque).astype(np.uint8),cv2.DIST_L2,3)
    keep=(alpha>=8)&(distance<=max(3,round(max(rgba.size)*.003)))
    rgba.putalpha(Image.fromarray(np.where(keep,alpha,0).astype(np.uint8)))
    x,y,w,h=cv2.boundingRect(opaque.astype(np.uint8))
    return rgba.crop((x,y,x+w,y+h))


def section(shape,rect):
    h,w=shape
    l,t,r,b=np.rint(np.asarray(rect)*[w,h,w,h]).astype(int)
    result=np.zeros(shape,bool)
    result[t:b,l:r]=True
    return result


def projected_asset(asset, parts, angle, target):
    factor=target/silhouette_axes(np.asarray(asset.getchannel("A")))[0]
    # Downsample before affine rotation to keep the edges, holes and narrow hose antialiased.
    asset=asset.resize(tuple(max(1,round(2*factor*d)) for d in asset.size),Image.Resampling.LANCZOS)
    w,h=asset.size
    pixels=np.asarray(asset,dtype=np.float32)/255
    alpha=pixels[:,:,3]
    def warp(a,flags=cv2.INTER_LINEAR):
        return cv2.warpAffine(a,matrix,size,flags=flags)
    premultiplied=pixels.copy()
    premultiplied[:,:,:3]*=alpha[:,:,None]
    scale=.5
    best=None
    for _ in range(4):
        matrix=cv2.getRotationMatrix2D((w/2,h/2),-angle,scale)
        corners=cv2.transform(np.array([[[0,0],[w,0],[w,h],[0,h]]],np.float32),matrix)[0]
        lo=np.floor(corners.min(axis=0)).astype(int)-3
        hi=np.ceil(corners.max(axis=0)).astype(int)+3
        matrix[:,2]-=lo
        size=tuple(map(int,hi-lo))
        output=warp(premultiplied)
        measured=silhouette_axes(np.rint(255*output[:,:,3]).astype(np.uint8))[0]
        error=abs(measured-target)
        if best is None or error<best[0]:
            best=(error,matrix,size,output)
        if error<=1:
            break
        scale*=target/measured
    _,matrix,size,output=best
    np.divide(output[:,:,:3],np.maximum(1/255,output[:,:,3:]),out=output[:,:,:3])
    rendered=np.rint(np.clip(output,0,1)*255).astype(np.uint8)
    solid=rendered[:,:,3]>=128
    bands=[]
    feet=[]
    for part,load in parts:
        part_mask=warp(section(alpha.shape,part).astype(np.uint8),cv2.INTER_NEAREST)>0
        load_mask=warp(section(alpha.shape,load).astype(np.uint8),cv2.INTER_NEAREST)>0
        band=contact_band(solid&part_mask,max(1,round(target*.48)))&load_mask
        if np.count_nonzero(band)<max(5,target*.08):
            raise NeedsReview("现有素材视角无法定位完整承托部件")
        bands.append(band)
        feet.append(cv2.dilate(band.astype(np.uint8),disk(max(2,round(target*.012))))>0)
    x,y,w,h=cv2.boundingRect((rendered[:,:,3]>0).astype(np.uint8))
    pad=max(3,round(target*.015))
    crop=np.s_[max(0,y-pad):y+h+pad,max(0,x-pad):x+w+pad]
    return rendered[crop],solid[crop],[part[crop] for part in bands],[part[crop] for part in feet]


def source_axis_angle(asset,axis):
    (x1,y1),(x2,y2)=axis
    return math.degrees(math.atan2((y2-y1)*asset.height,(x2-x1)*asset.width))


def placement_direction(asset,axis,rotation):
    if axis is None:
        return None
    angle=math.radians(source_axis_angle(asset,axis)+rotation)
    return math.cos(angle),math.sin(angle)


def angles_for(asset,axis,candidate):
    if axis is None:
        # Keep the vertical direction of the two seated/upright source photos.
        return (0,-4,4)
    (x1,y1),(x2,y2)=candidate["axis_px"]
    direction=math.degrees(math.atan2(y2-y1,x2-x1))-source_axis_angle(asset,axis)
    return tuple(round((direction+turn+delta+180)%360-180,2)
                 for delta in (0,-5,5,-12,12,-19,19) for turn in (0,180))


def previous_site_context(row,atlas,asset):
    before=row["previous_placement"]
    axis,parts=PARTS[row["asset_id"]]
    rotation=before["rotation"]
    pixels,solid,_,_=projected_asset(asset,parts,rotation,row["targets_px"][0])
    left,top,_,_=cv2.boundingRect((pixels[:,:,3]>0).astype(np.uint8))
    x,y=before["bounds"][:2]
    return atlas.native_position_context(before["support_id"],x-left,y-top,solid,row["targets_px"][0],
                                         placement_direction(asset,axis,rotation),
                                         measure_relief=row.get("placement_profile")=="recess_context")


def compatible_context(context,previous):
    terrain=context.get("relief")
    if terrain and terrain["kind"]=="raised":
        return False
    if not previous:
        return True
    if context["glare"]>max(.16,previous["glare"]+.06):
        return False
    if not terrain:
        return context["cost"]<=previous["cost"]-.015
    earlier=previous["relief"]
    improvement=RELIEF_PRIORITY[terrain["kind"]]-RELIEF_PRIORITY[earlier["kind"]]
    if improvement!=0:
        return improvement>0 and context["cost"]<=previous["cost"]+.12
    return (context["cost"]<=previous["cost"]-.015
            or (relief_value(terrain)-relief_value(earlier)>=.13
                and context["cost"]<=previous["cost"]+.08))


def context_report(context):
    return {name:round(value,4) if isinstance(value,float) else value for name,value in context.items()}


def outline_open_slots(folder,row,atlas,frame,asset):
    """Recover upright openings the coarse volume proxy may have missed."""
    axis,parts=PARTS[row["asset_id"]]
    if axis is not None:
        return []
    target=row["targets_px"][0]
    major=target/atlas.factor[0]
    previous=row.get("previous_placement")
    prefer_recess=row.get("placement_profile")=="recess_context"
    if previous:
        center=np.asarray(previous["bounds"]).reshape(2,2).mean(axis=0)/atlas.factor
    ranked=[]
    for angle in (0,-4,4):
        _,solid,bands,feet=projected_asset(asset,parts,angle,target)
        height,width=solid.shape
        size=tuple(np.rint(np.array([width,height])/atlas.factor).astype(int))
        mask=cv2.resize(solid.astype(np.uint8)*255,size,interpolation=cv2.INTER_AREA)>64
        union=solid|np.logical_or.reduce(feet)
        stencil=(cv2.resize(union.astype(np.uint8)*255,size,interpolation=cv2.INTER_AREA)>64).astype(np.float32)
        for plate in atlas.surfaces:
            left,top,w,h=plate.bounds
            if w<size[0] or h<size[1]:
                continue
            region=np.s_[top:top+h,left:left+w]
            clear=(plate.distance[region]>=1.)&(atlas.open_clearance(plate)[region]>=max(1.5,.025*major))
            match=cv2.matchTemplate(clear.astype(np.float32),stencil,cv2.TM_CCORR)
            valid=np.argwhere(match>stencil.sum()-.75)
            for y,x in valid[::max(1,len(valid)//90)]:
                px,py=int(left+x),int(top+y)
                if previous and math.dist((px+size[0]/2,py+size[1]/2),center)<.25*major:
                    continue
                where=np.s_[py:py+size[1],px:px+size[0]]
                context=atlas.position_context(plate,where,mask,major,measure_relief=prefer_recess)
                if context["glare"]>.16 or not compatible_context(context,previous["context"] if previous else None):
                    continue
                cost=context["cost"]
                if prefer_recess:
                    terrain=context["relief"]
                    cost-=.58*(terrain["kind"]=="recess")+.32*relief_value(terrain)
                ranked.append((cost,px,py,angle,plate,solid,bands))
    candidates=[]
    picked=[]
    for cost,px,py,angle,plate,solid,bands in sorted(ranked,key=lambda record:record[0]):
        if any(math.dist((px,py),point)<max(10,.19*major) for point in picked):
            continue
        ident="MNO"[len(candidates)]
        picked.append((px,py))
        x,y=atlas.screen([[px,py]])[0]
        height,width=solid.shape
        center_x,center_y=x+width//2,y+height//2
        side=round(1.65*max(height,width)+90)
        l,t=np.clip([center_x-side//2,center_y-side//2],0,np.asarray(frame.size)-side).astype(int).tolist()
        crop=[l,t,l+side,t+side]
        hull=cv2.convexHull(cv2.findNonZero(solid.astype(np.uint8))).reshape(-1,2)+[x,y]
        contacts=[]
        for band in bands:
            yy,xx=np.nonzero(band)
            contacts.append([x+round(float(np.median(xx))),y+round(float(np.median(yy)))])
        support=cv2.resize(plate.mask.astype(np.uint8),frame.size,interpolation=cv2.INTER_NEAREST)[t:t+side,l:l+side]
        root=folder/ident
        root.mkdir(exist_ok=True)
        Image.fromarray(support*255).save(root/"support.png")
        Image.new("L",(side,side)).save(root/"foreground.png")
        guide=frame.crop(tuple(crop))
        draw=ImageDraw.Draw(guide)
        points=[tuple(point) for point in (hull-[l,t]).tolist()]
        draw.line(points+[points[0]],fill="#40cde9",width=3)
        for cx,cy in contacts:
            draw.ellipse((cx-l-4,cy-t-4,cx-l+4,cy-t+4),fill="#8dec74")
        guide.save(root/"guide.jpg",quality=92)
        candidates.append(dict(id=ident,placement_mode="open",stance="upright" if row["kind"]=="jack" else "seated",
                               target_px=target,crop_px=crop,center_px=[center_x,center_y],hull_px=hull.tolist(),
                               pose_deg=None,source_rotation_deg=angle,asset_origin_px=[x,y],contacts_px=contacts,
                               support_id=f"S{plate.piece}",front_ids=[],projected_cover=0,footprint_fit=1.,
                               score=round(1-.90*cost,4)))
        if len(candidates)==3:
            break
    return candidates


def fit_candidate(folder,row,candidate,frame,atlas,asset):
    target=candidate["target_px"]
    l,t,r,b=candidate["crop_px"]
    root=folder/candidate["id"]
    patch=frame.crop((l,t,r,b))
    support=np.asarray(read_image(root/"support.png","L"))>0
    front=np.asarray(read_image(root/"foreground.png","L"))
    empty=candidate["placement_mode"]=="open"
    clearance=atlas.clearance_crop(candidate["support_id"],candidate["crop_px"]) if empty else None
    plane_distance=cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,3) if empty else None
    original_edges=None if empty else original_edge_neighborhood(np.asarray(patch))
    hull=np.asarray(candidate["hull_px"])
    ideal=(hull.min(axis=0)+hull.max(axis=0))/2-[l,t]
    axis,parts=PARTS[row["asset_id"]]
    # Lying the upright bottle-jack photo on its side would leave its loose handle
    # and the end-face geometry inconsistent with this bed.
    if row["kind"]=="jack" and candidate["stance"]!="upright":
        raise NeedsReview("现有独立图只支持千斤顶直立姿态")
    best=None
    locked=row.get("quiet_pose")
    prefer_recess=row.get("placement_profile")=="recess_context"
    discreet=prefer_recess or row.get("placement_profile")=="edge_context"
    previous=row.get("previous_placement")
    if previous:
        prior_bounds=np.asarray(previous["bounds"]).reshape(2,2)
        prior_center=prior_bounds.mean(axis=0)
    if locked:
        angles=(locked["rotation"],)
    elif "asset_origin_px" in candidate:
        angles=(candidate["source_rotation_deg"],)
    else:
        angles=angles_for(asset,axis,candidate)
    step=max(4,round(.055*target))
    offsets=[(x,y) for x in (0,-step,step,-2*step,2*step)
                   for y in (0,-step,step,-2*step,2*step)]
    for angle in angles:
        try:
            rgba,solid,bands,feet=projected_asset(asset,parts,angle,target)
        except NeedsReview:
            continue
        height,width=solid.shape
        silhouette_size=int(solid.sum())
        footprint=np.logical_or.reduce(feet)
        reserved=solid|footprint
        direction=placement_direction(asset,axis,angle) if discreet else None
        if locked:
            left,top,_,_=cv2.boundingRect((rgba[:,:,3]>0).astype(np.uint8))
            positions=[(locked["bounds"][0]-l-left,locked["bounds"][1]-t-top,0,0)]
        elif "asset_origin_px" in candidate:
            origin_x,origin_y=candidate["asset_origin_px"]
            nudge=max(3,round(max(atlas.factor)))
            positions=[(origin_x-l+dx,origin_y-t+dy,dx,dy)
                       for dx in (0,-nudge,nudge) for dy in (0,-nudge,nudge)]
        else:
            positions=[(*np.rint(ideal-[width/2,height/2]+[dx,dy]).astype(int),dx,dy)
                       for dx,dy in offsets]
        for x,y,dx,dy in positions:
            if x<0 or y<0 or x+width>patch.width or y+height>patch.height:
                continue
            if t+y<math.ceil(.09*frame.height) or t+y+height>math.floor(.82*frame.height):
                continue
            area=np.s_[y:y+height,x:x+width]
            bed=support[area]
            fit=float((footprint&bed).sum()/footprint.sum())
            support_per_part=min(float((part&bed).sum()/part.sum()) for part in feet)
            covered=float((solid&(front[area]>=128)).sum()/silhouette_size)
            if empty:
                depth_gap=float(clearance[area][reserved].min())
                # Thin foreground steel may be absent from the coarse depth map.
                # Require the projected outline to clear the actual plate boundary too.
                plane_gap=float(plane_distance[area][reserved].min())
                min_gap=max(4,.012*target) if discreet else (max(2,.012*target) if axis is not None else 1.)
                if min(fit,support_per_part)<.90 or depth_gap<max(6,.025*target) or plane_gap<min_gap or covered:
                    continue
                gap=min(depth_gap,plane_gap)
                score=.32*fit+.12*min(1,depth_gap/(.16*target))+.14*min(1,plane_gap/(.07*target))
            else:
                gap=None
                if min(fit,support_per_part)<.65 or not .026<=covered<=.66:
                    continue
                alpha=rgba[:,:,3]
                if (boundary_agreement(alpha,front[area],original_edges[area])<.48
                        or not crossing(alpha,front[area])):
                    continue
                score=.32*fit+.12-.55*abs(covered-candidate["projected_cover"])
            rotation_cost=abs((angle+90)%180-90)
            score+=.38*candidate["score"]-.24*math.hypot(dx,dy)/target-.001*rotation_cost
            context=None
            if discreet:
                center=np.array([l+x+width/2,t+y+height/2])
                if previous and np.linalg.norm(center-prior_center)<max(42,.25*target):
                    continue
                context=atlas.native_position_context(candidate["support_id"],l+x,t+y,solid,target,direction,
                                                      measure_relief=prefer_recess)
                if not compatible_context(context,previous["context"] if previous else None):
                    continue
                score=.22*score-context["cost"]
                if prefer_recess:
                    terrain=context["relief"]
                    score+=.58*(terrain["kind"]=="recess")+.32*relief_value(terrain)
            if best is None or score>best[0]:
                best=(score,angle,int(x),int(y),rgba,bands,fit,support_per_part,gap,covered,context)
    if best is None:
        raise NeedsReview("独立物品轮廓无法同时满足承托、净空和遮挡条件")
    return best


def program_spec(folder,row,candidate,fit,frame,scratch):
    _,angle,x,y,pixels,bands,footprint,per_part,gap,covered,context=fit
    l,t,r,b=candidate["crop_px"]
    target=candidate["target_px"]
    height,width=pixels.shape[:2]
    asset=Image.fromarray(pixels)
    box=asset.getchannel("A").getbbox()
    asset_path=scratch/"object.png"
    root=folder/candidate["id"]
    support=np.asarray(read_image(root/"support.png","L"))>0
    foreground=np.asarray(read_image(root/"foreground.png","L"))
    placed=np.zeros(support.shape,np.uint8)
    placed[y:y+height,x:x+width]=pixels[:,:,3]

    # Short ambient contact shadows follow the actual loaded contours. The
    # separate handle or reservoir contributes independently on the same steel.
    bearing=np.zeros(support.shape,bool)
    bearing[y:y+height,x:x+width]=np.logical_or.reduce(bands)
    distance=cv2.distanceTransform((~bearing).astype(np.uint8),cv2.DIST_L2,3)
    reach=max(5,.047*target)
    short=np.exp(-distance/max(1.8,.017*target))
    shadow=np.rint(245*short*np.maximum(0,1-distance/reach)).astype(np.uint8)
    shadow[~support | (placed>=250) | (foreground>=128)]=0
    shade=save_mask(shadow,scratch/"contact.png",candidate["crop_px"],(12,11,10))
    if shade is None:
        raise NeedsReview("支点附近没有可用的接触阴影像素")

    patch=frame.crop((l,t,r,b))
    around=cv2.distanceTransform((placed<128).astype(np.uint8),cv2.DIST_L2,3)
    nearby=support&(around>.04*target)&(around<.28*target)&(foreground<128)
    tone,photography,grain=fit_photo(patch,nearby,asset,0,(box[3]-box[1],box[2]-box[0]))
    if tone.get("luma_curve"):
        tone["luma_curve"]=[[source,round(.50*matched+.50*source,2)] for source,matched in tone["luma_curve"]]
    tone["blur_px"]=min(.88,max(.25,tone["blur_px"]))
    unmatched=None
    if row.get("blend_profile")=="quiet":
        unmatched=scratch/"object-original.png"
        asset.crop(box).save(unmatched)
        asset,profile=quiet_photo(patch,support&(foreground<128),nearby,asset,x,y,target)
        photography["quiet"]=profile
        tone={name:0 for name in ("brightness","warmth","saturation")} | {"blur_px":tone["blur_px"]}
    asset.crop(box).save(asset_path)
    spec=dict(id=candidate["id"],asset=asset_path.as_posix(),center=[l+x+(box[0]+box[2])/2,t+y+(box[1]+box[3])/2],
              width=box[2]-box[0],angle=0,canvas=list(frame.size),crop=candidate["crop_px"],
              tone=tone,shade=shade,occlusion=None,object_alpha=(scratch/"alpha.png").as_posix(),
              png=(scratch/"staged.png").as_posix(),psd=(scratch/"staged.psd").as_posix(),
              placement_mode=candidate["placement_mode"])
    if unmatched:
        spec["unmatched_asset"]=unmatched.as_posix()
    if candidate["placement_mode"]=="scrap":
        edit=(placed>0)|(shadow>0)
        local=foreground*(cv2.dilate(edit.astype(np.uint8),disk(4))>0)
        spec["occlusion"]=save_mask(local,scratch/"front.png",candidate["crop_px"],(255,255,255))
        spec["foreground_alpha"]=(scratch/"front-alpha.png").as_posix()
    if grain is not None:
        grain_path=scratch/"grain.png"
        Image.fromarray(grain).convert("RGB").save(grain_path)
        spec["grain"]=grain_path.as_posix()
    contacts=[]
    for band in bands:
        yy,xx=np.nonzero(band)
        contacts.append([round(l+x+float(np.median(xx))),round(t+y+float(np.median(yy)))])
    return spec,dict(placement_mode=candidate["placement_mode"],footprint_fit=round(footprint,3),
                     proxy_footprint_fit=candidate["footprint_fit"],min_part_support=round(per_part,3),
                     estimated_clearance_px=round(gap,1) if gap is not None else None,
                     contact_points_px=contacts,photography=photography,source_rotation_deg=angle,
                     estimated_occlusion=round(covered,4),
                     placement_context=context_report(context) if context else None)
