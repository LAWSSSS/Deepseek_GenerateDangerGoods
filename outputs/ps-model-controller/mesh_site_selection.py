"""Select scene support anchors before inspecting any candidate object pose."""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from metric_scale import SceneReference, ScaleReview, depth_at, project_span, reference_geometry, reference_radius
from scene_v2 import NeedsReview, angle_delta


def reference_evidence(scene,anchor):
    """Check local reference availability without assuming an object projection."""
    size=scene["size"]
    if "depth_native" not in scene:
        scene["depth_native"]=cv2.resize(scene["depth"],size,interpolation=cv2.INTER_LINEAR)
    depth=scene["depth_native"]
    accepted=[]; rejected=[]
    for ref in scene["references"].values():
        known=isinstance(ref,SceneReference)
        try:
            if not known and ref.span_mm[1]/ref.span_mm[0]>2.5:
                raise ScaleReview("reference_size_weak","参照尺寸范围过宽")
            p,q,length,ref_depth,tolerance,radius=reference_geometry(
                ref,size,depth,minimum=24 if known else max(24,size[0]*.025))
            midpoint=((p[0]+q[0])/2,(p[1]+q[1])/2)
            delta=abs(ref_depth-depth_at(depth,tuple(map(round,anchor)),radius))
            if math.dist(midpoint,anchor)>reference_radius(size[0],length) or delta>tolerance:
                raise ScaleReview("reference_off_plane","参照不在同一局部深度")
            angle=math.degrees(math.atan2(q[1]-p[1],q[0]-p[0]))
            sine=math.sin(math.radians(scene["camera_elevation_deg"]))
            corrected=math.hypot(q[0]-p[0],(q[1]-p[1])/sine)
            _,scale_range,_=project_span(ref,corrected,(1.,1.))
            accepted.append({"id":ref.id,"known":known,"points":[list(p),list(q)],
                             "depth_delta":round(delta,4),"angle_deg":round(angle,4),
                             "scale_range":[float(x) for x in scale_range]})
        except ScaleReview as issue:
            rejected.append({"id":ref.id,"reason":issue.code})
    known=[ref for ref in accepted if ref["known"]]
    usable=known or accepted
    independent={tuple(sorted(map(tuple,ref["points"]))) for ref in usable}
    available=bool(known) or len(independent)>=2
    reason=None
    if available and not known and any(abs(angle_delta(ref["angle_deg"],usable[0]["angle_deg"]))>24 for ref in usable):
        available=False;reason="reference_direction_conflict"
    if available and max(ref["scale_range"][0] for ref in usable)>min(ref["scale_range"][1] for ref in usable):
        available=False;reason="reference_conflict"
    return {"available":available,"reason":reason,"accepted":accepted,"rejected":rejected}


def select_sites(atlas,scene,target,*,limit=64):
    """Rank object-independent support positions and preserve spatial diversity."""
    width,height=atlas.size
    factor=atlas.factor
    step=max(2,round(.055*target/min(factor)))
    target_small=target/min(factor)
    texture,light,glare=atlas.context_maps(target_small)
    proposals=[]; reasons={}
    for surface in atlas.surfaces:
        minimum=max(4,.012*target)/min(factor)
        clearance=max(6,.025*target)/min(factor)
        clear=(surface.distance>=minimum)&(atlas.open_clearance(surface)>=clearance)
        ys,xs=np.nonzero(clear[::step,::step])
        support_fraction=cv2.boxFilter(surface.mask.astype(np.float32),-1,
                                        (max(3,round(target_small*.22))|1,)*2,
                                        normalize=True,borderType=cv2.BORDER_CONSTANT)
        for x,y in zip(xs*step,ys*step):
            anchor=[float(x*factor[0]),float(y*factor[1])]
            if not (0<=anchor[0]<width and 0<=anchor[1]<height): continue
            if any(l<=anchor[0]<r and t<=anchor[1]<b for l,t,r,b in scene["protected"]): continue
            valid=False
            for ident in scene["answer"].zones:
                center=scene["zones"][ident]["anchors"][0]
                point=[center[0]*(width-1)/1000,center[1]*(height-1)/1000]
                if math.dist(anchor,point)<=max(2*target,.12*width): valid=True;break
            if not valid: continue
            evidence=reference_evidence(scene,anchor)
            if not evidence["available"]:
                reason=evidence.get("reason") or "insufficient_local_references"
                reasons[reason]=reasons.get(reason,0)+1
                continue
            edge=float(surface.distance[y,x]); max_edge=max(5,.26*target_small)
            edge_proximity=float(np.clip(1-edge/max_edge,0,1))
            brightness=float(light[y,x]); reflected=float(glare[y,x])
            bright_penalty=.46*np.clip((brightness-.29)/.38,0,1)+.54*min(1,reflected/.46)
            reach=float(np.clip(min(edge,float(atlas.open_clearance(surface)[y,x]))/
                                max(1,.35*target_small),0,1))
            score=(.35*float(support_fraction[y,x])+.38*reach+.08*edge_proximity+
                   .12*float(np.clip(texture[y,x],0,1))-.12*float(bright_penalty))
            proposals.append({"support_id":f"S{surface.piece}","support_anchor_px":anchor,
                              "scene_score":round(float(score),6),"surface_error":float(surface.error),
                              "references":evidence,"target_px":target})
    proposals.sort(key=lambda site:site["scene_score"],reverse=True)
    selected=[]
    for site in proposals:
        if any(site["support_id"]==other["support_id"] and
               math.dist(site["support_anchor_px"],other["support_anchor_px"])<.09*target for other in selected):
            continue
        selected.append({**site,"id":f"S{len(selected)+1:03}"})
        if len(selected)>=limit: break
    return selected,reasons


def site_candidate(site,view,atlas,folder,*,target=None):
    """Build masks around a fixed scene anchor after a 3-D view is known."""
    pixels,_,bands=view
    h,w=pixels.shape[:2]
    cy,cx=np.nonzero(np.logical_or.reduce(bands))
    if not len(cx):raise NeedsReview("视角缺少接触点")
    x,y=map(int,np.rint(np.asarray(site["support_anchor_px"])-[cx.mean(),cy.mean()]))
    pad=max(32,round(.14*(target or site["target_px"])))
    l,t=max(0,x-pad),max(0,y-pad)
    r,b=min(atlas.size[0],x+w+pad),min(atlas.size[1],y+h+pad)
    if x<0 or y<0 or x+w>atlas.size[0] or y+h>atlas.size[1] or r<=l or b<=t:
        raise NeedsReview("固定接触点导致物体超出画面")
    surface=next((s for s in atlas.surfaces if f"S{s.piece}"==site["support_id"]),None)
    if surface is None:raise NeedsReview("承托面不存在")
    dest=Path(folder)/site["id"];dest.mkdir(parents=True,exist_ok=True)
    support=cv2.resize(surface.mask.astype(np.uint8),atlas.size,interpolation=cv2.INTER_NEAREST)[t:b,l:r]
    Image.fromarray(support*255).save(dest/"support.png")
    Image.new("L",(r-l,b-t)).save(dest/"foreground.png")
    return {"id":site["id"],"placement_mode":"open","target_px":target or site["target_px"],
            "crop_px":[l,t,r,b],"center_px":[x+w/2,y+h/2],"support_anchor_px":site["support_anchor_px"],
            "support_id":site["support_id"]}
