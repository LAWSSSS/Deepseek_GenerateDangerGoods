"""Extract a placed object and its contact light from a registered CCTV crop."""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from metric_scale import silhouette_axes
from photometry import fit_photo
from scene_pose import asset_shape, make_projection, open_asset_shapes
from scene_v2 import NeedsReview, angle_delta, contact_band, digest, read_image


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))


def gray(rgb):
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


def components(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(int(stats[i,cv2.CC_STAT_AREA]),labels==i) for i in range(1,n)]


def register_crop(base,edit,guard):
    h,w=base.shape[:2]
    if abs(edit.width/edit.height-w/h)>.04:
        raise NeedsReview("生成裁区长宽比与固定画面不符")
    edited=np.asarray(edit.resize((w,h),Image.Resampling.LANCZOS))
    detector=cv2.SIFT_create(nfeatures=2300,contrastThreshold=.019)
    source,desc=detector.detectAndCompute(gray(edited),None)
    target,desc_base=detector.detectAndCompute(gray(base),guard.astype(np.uint8)*255)
    if desc is None or desc_base is None or len(desc)<2:
        raise NeedsReview("背景边缘不足以配准生成裁区")
    pairs=cv2.BFMatcher().knnMatch(desc_base,desc,k=2)
    correspond=[]
    for first,second in pairs:
        a=source[first.trainIdx].pt
        b=target[first.queryIdx].pt
        if first.distance<.78*second.distance and math.dist(a,b)<.13*w:
            correspond.append((a,b))
    if len(correspond)<9:
        raise NeedsReview(f"生成裁区仅有 {len(correspond)} 组可信背景边缘")
    a,b=(np.ascontiguousarray(points) for points in np.asarray(correspond,np.float32).transpose(1,0,2))
    matrix,flags=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=max(2.,w*.005))
    if matrix is None or flags is None or int(flags.sum())<9:
        raise NeedsReview("生成裁区的钢件边缘无法统一配准")
    good=flags.ravel()>0
    spread=cv2.contourArea(cv2.convexHull(b[good]))/(w*h)
    scale=math.hypot(matrix[0,0],matrix[1,0])
    angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
    if spread<.075 or not .89<=scale<=1.12 or abs(angle)>3.:
        raise NeedsReview("生成裁区改变了机位、裁切或局部钢件几何")
    aligned=cv2.warpAffine(edited,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    valid=cv2.warpAffine(np.ones((h,w),np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST)>0
    stable=guard&valid
    residual=np.abs(gray(aligned).astype(np.float32)-gray(base))
    if np.median(residual[stable])>22:
        raise NeedsReview("配准后周围钢件的外观偏差过大")
    return aligned,stable,{"matches":int(flags.sum()),"spread":round(spread,3),
                            "scale":round(scale,4),"rotation_deg":round(angle,2),
                            "median_background_error":round(float(np.median(residual[stable])),2)}


def extract_object(rgb,base,proxy,guard,target,engine,kind):
    from mobile_sam import SamPredictor
    import torch

    area=proxy.sum()
    hull=cv2.convexHull(cv2.findNonZero(proxy.astype(np.uint8)))
    expected=np.zeros(proxy.shape,np.uint8)
    cv2.fillConvexPoly(expected,hull,1)
    envelope=cv2.dilate(expected,disk(max(8,round(target*.12))))>0
    change=cv2.GaussianBlur(rgb.astype(np.float32)-base.astype(np.float32),(0,0),1.25)
    change-=np.median(change[guard],axis=0)
    change=np.max(np.abs(change),axis=2)
    signal=change>max(15,float(np.quantile(change[guard],.88))+4)
    x,y,w,h=cv2.boundingRect(envelope.astype(np.uint8))
    local=signal&envelope
    if (local&proxy).sum()<.13*area:
        raise NeedsReview("选定落点没有足够的物品像素")
    predictor=SamPredictor(engine.sam())
    with torch.inference_mode():
        predictor.set_image(np.ascontiguousarray(rgb))
        broad=predictor.predict(box=np.array([x,y,x+w-1,y+h-1]),multimask_output=True)[0]
        distance=cv2.distanceTransform(local.astype(np.uint8),cv2.DIST_L2,3)
        masks=list(broad)
        for _ in range(5):
            yy,xx=np.unravel_index(np.argmax(distance),distance.shape)
            if distance[yy,xx]<1.7:
                break
            masks.extend(predictor.predict(point_coords=np.array([[xx,yy]]),
                                          point_labels=np.array([1]),multimask_output=True)[0])
            cv2.circle(distance,(int(xx),int(yy)),max(10,round(.14*target)),0,-1)
    proposals=[]
    for mask in masks:
        # SAM may predict isolated pixels on unrelated steel far from the object.
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        sizes=stats[:,cv2.CC_STAT_AREA]
        keep=(sizes>=8)&(np.bincount(labels[envelope],minlength=count)>=.86*sizes)
        keep[0]=False
        mask=keep[labels]
        n=mask.sum()
        if n<max(24,area*.025) or n>area*1.65 or (mask&~envelope).sum()>.075*n:
            continue
        evidence=float((mask&signal).sum()/n)
        overlap=float((mask&proxy).sum()/(mask|proxy).sum())
        if evidence<.36:
            continue
        proposals.append((overlap+.24*evidence,mask))
    mains=[p for p in proposals if p[1].sum()>.32*area]
    if not mains:
        raise NeedsReview("无法从生成裁区确认完整物品轮廓")
    primary=max(mains,key=lambda p:p[0])[1]
    solid=primary.copy()
    if kind=="shock":
        for _,part in sorted(proposals,key=lambda p:p[0],reverse=True):
            size=part.sum()
            if size<.35*solid.sum() and (part&solid).sum()<.14*size and (part&proxy).sum()>.43*size:
                solid|=part
    if kind=="jack":
        # Segment the loose handle independently of the largest connected body.
        primary=principal(primary)
        red=rgb[:,:,0].astype(np.int16)-np.maximum(rgb[:,:,1],rgb[:,:,2])
        old=base[:,:,0].astype(np.int16)-np.maximum(base[:,:,1],base[:,:,2])
        px,py,pw,ph=cv2.boundingRect(primary.astype(np.uint8))
        head=np.zeros(primary.shape,bool)
        head[max(0,py-round(.12*target)):py+round(.10*target),px:px+pw]=True
        # SAM can split the knurled red cap from the chrome plunger.
        cap=(red>22)&(red-old>9)&head&~primary
        joined=cv2.morphologyEx((primary|cap).astype(np.uint8),cv2.MORPH_CLOSE,disk(4))>0
        primary=principal(joined)
        loose=(red>18)&((red-old)>7)&envelope&~(cv2.dilate(primary.astype(np.uint8),disk(3))>0)
        loose=cv2.morphologyEx(loose.astype(np.uint8),cv2.MORPH_CLOSE,disk(5))>0
        handles=[]
        for size,part in components(loose):
            if not area*.012<=size<=area*.29:
                continue
            _,(a,b),_=cv2.minAreaRect(cv2.findNonZero(part.astype(np.uint8)))
            if min(a,b)>1 and max(a,b)/min(a,b)>3 and .38*target<=max(a,b)<=.97*target:
                handles.append((max(a,b),part))
        if not handles:
            raise NeedsReview("独立千斤顶手柄的完整轮廓未确认")
        _,handle=max(handles,key=lambda item:item[0])
        px,py,pw,ph=cv2.boundingRect(handle.astype(np.uint8))
        box=np.array([max(0,px-5),max(0,py-5),min(rgb.shape[1]-1,px+pw+5),min(rgb.shape[0]-1,py+ph+5)])
        dist=cv2.distanceTransform(handle.astype(np.uint8),cv2.DIST_L2,3)
        sy,sx=np.unravel_index(np.argmax(dist),dist.shape)
        with torch.inference_mode():
            variations=predictor.predict(point_coords=np.array([[sx,sy]]),point_labels=np.array([1]),
                                          box=box,multimask_output=True)[0]
        close=[m for m in variations if (m&handle).sum()>=.70*handle.sum() and m.sum()<=1.9*handle.sum()]
        if close:
            handle=max(close,key=lambda m:(m&handle).sum()/max(1,(m|handle).sum()))
        solid=primary|handle
    if kind=="ram":
        # Keep small gaps caused by SAM out of the steel connection at the rod end.
        # Recover a bridge only where the generated crop differs from original steel.
        joined=cv2.morphologyEx(solid.astype(np.uint8),cv2.MORPH_CLOSE,disk(max(3,round(.014*target))))>0
        evidence=(change>max(10,float(np.median(change[guard]))+3))
        solid|=joined&envelope&(cv2.dilate(evidence.astype(np.uint8),disk(2))>0)
        if principal(solid).sum()<.98*solid.sum():
            raise NeedsReview("活塞杆与吊耳之间的原有连接未完整提取")
    if (solid&~envelope).sum()>.065*solid.sum():
        raise NeedsReview("物品轮廓超出已验证的空位")
    return solid,primary


def principal(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return labels==int(1+np.argmax(stats[1:,cv2.CC_STAT_AREA])) if n>1 else mask


def match_placement(row,candidate,solid,primary,proxy,support,clearance):
    l,t,_,_=candidate["crop_px"]
    target=candidate["target_px"]
    source=np.asarray(read_image(Path(row["rust"]),"RGBA").getchannel("A"))
    kind,parts,feet=asset_shape(row["file"],source)
    _,parts,feet=next(v for stance,*v in open_asset_shapes(kind,parts,feet) if stance==candidate["stance"])
    origin=np.asarray(candidate["center_px"])-[l,t]
    guide_hull=np.asarray(candidate["hull_px"],np.float32)-np.array([l,t],np.float32)
    alpha=solid.astype(np.uint8)*255
    major,minor,_=silhouette_axes(alpha)
    scale=target/major
    if not .80<=scale<=1.22:
        raise NeedsReview(f"生成物品长轴 {major:.1f} px，超出允许的等比对齐范围")
    pmajor,pminor,pangle=silhouette_axes(principal(proxy).astype(np.uint8)*255)
    mmajor,mminor,mangle=silhouette_axes(principal(primary).astype(np.uint8)*255)
    center=np.array(cv2.minAreaRect(cv2.findNonZero(alpha))[0])
    guide_center=np.asarray(cv2.minAreaRect(guide_hull)[0])
    if np.linalg.norm(center-guide_center)>.20*target:
        raise NeedsReview("生成物品偏离选定承托面")
    height,width=support.shape
    near=cv2.dilate(support.astype(np.uint8),disk(3))>0
    fits=[]
    drift=round(.043*target)
    yaw,pitch,roll=candidate["pose_deg"]
    # A seated bearing can turn on the same plate. Verify the actual yaw against
    # the same depth surface rather than rotating the photograph to an old sketch.
    angles=[yaw] if kind!="bearing" else [yaw+d for d in (0,-8,8,-16,16,-24,24,-32,32)]
    for azimuth in angles:
        projected=make_projection(parts,feet,azimuth,pitch,roll,target)
        hull=projected.hull+origin
        (cx,cy),(w,h),angle=cv2.minAreaRect(hull.astype(np.float32))
        ratio=max(w,h)/max(1,min(w,h))
        if not .60<=(major/minor)/ratio<=1.65:
            continue
        predicted_angle=(angle if w>=h else angle+90) if kind=="bearing" else pangle
        correction=angle_delta(predicted_angle,mangle) if pmajor/pminor>1.3 and mmajor/mminor>1.3 else 0.
        if abs(correction)>19.:
            continue
        ground=np.rint(projected.footprint+origin).astype(np.int32)
        contacts=np.rint(projected.contacts+origin).astype(np.int32)
        desired=np.asarray([cx,cy])
        transform=cv2.getRotationMatrix2D(tuple(center),-float(np.clip(correction,-4,4)),scale)
        transform[:,2]+=desired-center
        for dx,dy in ((0,0),(-drift,0),(drift,0),(0,-drift),(0,drift)):
            matrix=transform.copy()
            matrix[:,2]+=[dx,dy]
            moved=cv2.warpAffine(alpha,matrix,(width,height),flags=cv2.INTER_LINEAR)>=128
            envelope=np.zeros(moved.shape,np.uint8)
            cv2.fillConvexPoly(envelope,np.rint(hull+[dx,dy]).astype(np.int32),1)
            envelope=cv2.dilate(envelope,disk(max(6,round(.072*target))))>0
            if not moved.any() or (moved&envelope).sum()<.94*moved.sum():
                continue
            footprint=np.zeros(moved.shape,np.uint8)
            cv2.fillConvexPoly(footprint,ground+[dx,dy],1)
            fit=float((footprint&support).sum()/footprint.sum())
            gap=float(clearance[moved|(footprint>0)].min())
            if fit<.90 or gap<max(6.,.025*target):
                continue
            contact_distance=cv2.distanceTransform((~(moved&near)).astype(np.uint8),cv2.DIST_L2,3)
            at=contacts+[dx,dy]
            if any(not (0<=u<width and 0<=v<height) or not support[v,u]
                   or contact_distance[v,u]>max(5,target*.061) for u,v in at):
                continue
            band=contact_band(moved,cv2.boundingRect(moved.astype(np.uint8))[2])
            resting=float((band&near).sum()/max(1,band.sum()))
            if resting<.60:
                continue
            movement=np.linalg.norm(desired-center+[dx,dy])
            cost=.002*abs(correction)+.0015*movement
            fits.append((resting+.28*fit-cost,matrix,moved,at,fit,resting,gap,[azimuth,pitch,roll]))
    if not fits:
        raise NeedsReview("对齐后的实际轮廓、支点或落地足迹不满足空位要求")
    _,matrix,moved,contacts,fit,resting,gap,pose=max(fits,key=lambda item:item[0])
    return matrix,moved,contacts,{"footprint_fit":round(fit,3),"planned_support":round(resting,3),
                                  "contact_points_px":(contacts+[l,t]).tolist(),"context_scale":round(scale,4),
                                  "estimated_clearance_px":round(gap,1),"fitted_pose_deg":pose}


def bearing_pixels(moved, contacts, target, kind):
    """Track separate contacts even when parts overlap in image columns."""
    main=np.zeros(moved.shape,np.uint8)
    radius=max(5,round((.19 if kind=="bearing" else .16)*target))
    for x,y in contacts[:2]:
        cv2.circle(main,(int(x),int(y)),radius,1,-1)
    if kind!="bearing":
        cv2.line(main,tuple(contacts[0]),tuple(contacts[1]),1,2*radius)
    body=moved
    parts=[]
    for point in contacts[2:]:
        a,b=contacts[:2]
        ab=b-a
        nearest=a+ab*np.clip((point-a)@ab/max(1,ab@ab),0,1)
        direction=point-nearest
        yy,xx=np.ogrid[:moved.shape[0],:moved.shape[1]]
        # Local side of the plane containing the handle or reservoir. The hose
        # stays with the object; it does not carry the spring body's load.
        side=(xx-point[0])*direction[0]+(yy-point[1])*direction[1]>-.37*(direction@direction)
        region=((xx-point[0])**2+(yy-point[1])**2)<(.24*target)**2
        part=moved&side
        parts.append(contact_band(part,max(1,round(.30*target)))&region)
        body=body&~side
    parts.insert(0,contact_band(body,max(1,round(.50*target)))&(main>0))
    return parts


def context_spec(folder,row,candidate,scratch,engine,frame,atlas):
    receipt=row["generations"][-1]
    asset=folder/receipt["file"]
    if receipt["mode"]!="context" or receipt["candidate"]!=candidate["id"] or digest(asset)!=receipt["sha256"]:
        raise ValueError("局部生成回执与落点或内容哈希不一致")
    root=folder/candidate["id"]
    base=np.asarray(read_image(root/"context.png"))
    proxy=np.asarray(read_image(root/"proxy.png","L"))>0
    support=np.asarray(read_image(root/"support.png","L"))>0
    clearance=atlas.clearance_crop(candidate["support_id"],candidate["crop_px"])
    guard=cv2.dilate(proxy.astype(np.uint8),disk(max(12,round(.19*candidate["target_px"]))))==0
    aligned,stable,registration=register_crop(base,read_image(asset),guard)
    solid,primary=extract_object(aligned,base,proxy,stable,candidate["target_px"],engine,row["kind"])
    matrix,moved,contacts,metrics=match_placement(row,candidate,solid,primary,proxy,support,clearance)
    h,w=base.shape[:2]
    l,t,r,b=candidate["crop_px"]
    def warp(value, interpolation=cv2.INTER_LINEAR):
        return cv2.warpAffine(value,matrix,(w,h),flags=interpolation)

    alpha=warp(solid.astype(np.uint8)*255)
    alpha=cv2.GaussianBlur(alpha,(5,5),.72)
    alpha[alpha<3]=0
    pixels=warp(aligned,cv2.INTER_CUBIC)
    rgba=Image.fromarray(np.dstack((pixels,alpha)))
    box=rgba.getchannel("A").getbbox()
    path=scratch/"object.png"
    rgba.crop(box).save(path)

    delta=cv2.GaussianBlur(gray(aligned).astype(np.float32)-gray(base),(0,0),.9)
    native_distance=cv2.distanceTransform((~solid).astype(np.uint8),cv2.DIST_L2,3)
    target=candidate["target_px"]
    nearby=support & (native_distance>.075*target) & (native_distance<.18*target)
    delta-=float(np.median(delta[nearby if nearby.sum()>150 else stable]))
    delta=warp(np.clip(delta*1.32,-53,5))
    reach=max(5,round(.050*target))
    parts=bearing_pixels(moved,contacts,target,row["kind"])
    distance=np.full(moved.shape,reach+1.,np.float32)
    floor=support&~moved
    sample_counts=[]
    for i,bearing in enumerate(parts):
        d=cv2.distanceTransform((~bearing).astype(np.uint8),cv2.DIST_L2,3)
        samples=int(np.count_nonzero(floor & (d<reach) & (delta<-4)))
        sample_counts.append(samples)
        if samples<max(8,round((.04 if i==0 else .025)*target)):
            raise NeedsReview("生成裁区缺少可与受力点对齐的接触阴影")
        distance=np.minimum(distance,d)
    # Transfer the generated low-frequency light over the real steel. Under the
    # supported edge the local contact darkening also reduces plate reflections.
    allowed=floor&(distance<reach)
    opacity=np.rint(255*np.clip(1-distance/reach,0,1)**.96*allowed).astype(np.uint8)
    tight=.24*np.exp(-distance/max(2.,.015*target))
    delta=np.minimum(delta,0)-tight*gray(base)
    if not opacity.any():
        raise NeedsReview("生成裁区缺少可与受力点对齐的接触阴影")
    contact=Image.fromarray(np.dstack((np.clip(base.astype(np.float32)+delta[:,:,None],0,255).astype(np.uint8),opacity)))
    bounds=contact.getchannel("A").getbbox()
    contact_path=scratch/"contact-delta.png"
    contact.crop(bounds).save(contact_path)
    around=cv2.distanceTransform((~moved).astype(np.uint8),cv2.DIST_L2,3)
    lighting=support&~moved&(around>.035*target)&(around<.35*target)
    tone,photo,grain=fit_photo(Image.fromarray(base),lighting,rgba,0,(box[3]-box[1],box[2]-box[0]))
    if tone.get("luma_curve"):
        tone["luma_curve"]=[[x,round(.27*y+.73*x,2)] for x,y in tone["luma_curve"]]
    if tone.get("rgb_cast"):
        tone["rgb_cast"]=[round(v*.34,2) for v in tone["rgb_cast"]]
    tone["saturation"]=round(tone["saturation"]*(.93 if row["kind"]=="jack" else .55))
    tone["blur_px"]=min(.76,max(tone["blur_px"],.30))
    spec={"id":candidate["id"],"asset":path.as_posix(),"canvas":list(frame.size),
          "center":[l+(box[0]+box[2])/2,t+(box[1]+box[3])/2],"width":box[2]-box[0],
          "angle":0,"crop":[l,t,r,b],"tone":tone,
          "shade":None,"occlusion":None,"contact_delta":{"path":contact_path.as_posix(),"x":l+bounds[0],"y":t+bounds[1]},
          "object_alpha":(scratch/"alpha.png").as_posix(),"png":(scratch/"staged.png").as_posix(),
          "psd":(scratch/"staged.psd").as_posix(),"placement_mode":"open"}
    if grain is not None:
        grain_path=scratch/"grain.png"
        Image.fromarray(grain).convert("RGB").save(grain_path)
        spec["grain"]=grain_path.as_posix()
    return spec,{**metrics,"registration":registration,"placement_mode":"open","photography":photo,
                 "contact_evidence_px":sample_counts}
