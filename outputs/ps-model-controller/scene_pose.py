"""Coarse support and volume checks for the fixed-camera pose review."""
from collections import Counter
from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from metric_scale import silhouette_axes
from scene_v2 import NeedsReview, boundary_agreement, crossing, original_edge_neighborhood
from relief import ReliefField, plane_fit, relief_value


ELEVATION = 57
STEP_MARGIN = .015


def tube(profile, side=12):
    """Return a faceted shell around the local X axis."""
    theta = np.linspace(0,2*np.pi,side,endpoint=False)
    vertices = [(x,ry*np.cos(a)+y,rz*np.sin(a)+z)
                for x,ry,rz,y,z in profile for a in theta]
    faces = [tuple(range(side-1,-1,-1)),tuple(range(len(vertices)-side,len(vertices)))]
    for i in range(len(profile)-1):
        for j in range(side):
            a,b = i*side+j,i*side+(j+1)%side
            faces.append((a,b,b+side,a+side))
    return np.asarray(vertices),faces


def cylinder(profile, y=0., z=0.):
    return tube([(x,r,r,y,z) for x,r in profile])


def extrusion(poly, y0, y1):
    n=len(poly)
    return (np.asarray([(x,y,z) for y in (y0,y1) for x,z in poly]),
            [tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]+
            [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)])


def box(x0,x1,y0,y1,z0,z1):
    return extrusion([(x0,z0),(x1,z0),(x1,z1),(x0,z1)],y0,y1)


def asset_shape(file, alpha):
    major,minor,_=silhouette_axes(alpha)
    ratio=min(.78,max(.12,minor/major))
    if "轴承" in file:
        arch=[(.34*math.cos(a),.31+.33*math.sin(a)) for a in np.linspace(0,np.pi,13)]
        return "bearing",[box(-.5,.5,-.16,.16,0,.13),
                          extrusion([(.34,.10),*arch,(-.34,.10)],-.11,.11)],(-.32,.32)
    if "储罐" in file:
        radius=min(.32,max(.20,ratio*.46))
        shell=cylinder([(-.5,.39*radius),(-.38,radius),(.37,radius),
                        (.46,.77*radius),(.50,.36*radius)])
        legs=box(-.52,-.37,-radius,radius,-radius,.04)
        lid=box(.32,.45,-.12,.12,.08,radius+.06)
        return "tank",[shell,legs,lid],(-.28,.22)
    if "液压减震器" in file:
        return "shock",[cylinder([(-.5,.06),(-.32,.095),(-.21,.115),(.22,.115),(.5,.055)]),
                        cylinder([(-.37,.09),(-.13,.09)],.30,-.014)],(-.30,.15)
    if "千斤顶" in file:
        if "千斤顶2" in file:
            return "jack",[box(-.5,.5,-.20,.20,0,.25)],(-.28,.27)
        body=cylinder([(-.48,.24),(-.36,.24),(-.32,.18),(.29,.18),(.34,.09),(.48,.09)])
        handle=cylinder([(-.48,.028),(.44,.028)],.38,-.21)
        return "jack",[body,handle],(-.29,.17)
    if any(word in file for word in ("缸","减震","过滤","集油器")):
        radius=min(.17,max(.083,ratio*.47))
        return "ram",[cylinder([(-.5,radius*.51),(-.40,radius),(.15,radius),
                                (.21,radius*.46),(.39,radius*.46),(.43,radius*.76),(.5,radius*.62)])],(-.32,.13)
    radius=ratio*.46
    return "body",[cylinder([(-.50,radius*.74),(-.34,radius),(.32,radius),
                             (.46,radius*.64),(.50,radius*.39)])],(-.27,.24)


def open_asset_shapes(kind, parts, feet):
    if kind=="jack" and len(parts)==2:
        body,faces=parts[0]
        upright=body[:,[1,2,0]].copy()
        base=box(-.235,.235,-.20,.20,-.48,-.38)
        handle=cylinder([(-.43,.026),(.43,.026)],.38,-.454)
        yield "upright","jack_upright",[(upright,faces),base,handle],((-.18,0),(.18,0),(0,.38))
        feet=((-0.29,0),(.17,0),(0,.38))
    if kind=="shock":
        feet=((-0.30,0),(.15,0),(-.25,.30))
    yield "lying" if kind in ("jack","ram","shock") else "seated",kind,parts,feet


def rotation(yaw,pitch,roll):
    a,b,c=np.radians((yaw,pitch,roll))
    ca,sa,cb,sb,cc,sc=math.cos(a),math.sin(a),math.cos(b),math.sin(b),math.cos(c),math.sin(c)
    return (np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]]) @
            np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]]) @
            np.array([[1,0,0],[0,cc,-sc],[0,sc,cc]]))


def project(xyz):
    elev=math.radians(ELEVATION)
    x,y,z=xyz.T
    return np.column_stack((x,-math.sin(elev)*y-math.cos(elev)*z))


def silhouette(points):
    hull=cv2.convexHull(points.astype(np.float32))
    return float(max(cv2.minAreaRect(hull)[1]))


@dataclass
class Projection:
    hull: np.ndarray
    mesh: list
    contacts: np.ndarray
    footprint: np.ndarray
    extent: tuple
    low: np.ndarray
    high: np.ndarray
    solid: np.ndarray
    ground: np.ndarray
    axis: np.ndarray


def make_projection(parts, feet, yaw, pitch, roll, major):
    matrix=rotation(yaw,pitch,roll)
    shapes=[(v@matrix.T,faces) for v,faces in parts]
    zmin=min(v[:,2].min() for v,_ in shapes)
    shapes=[(v-[0,0,zmin],faces) for v,faces in shapes]
    points=np.vstack([v for v,_ in shapes])
    projected=project(points)
    scale=major/silhouette(projected)
    hull=cv2.convexHull((projected*scale).astype(np.float32)).reshape(-1,2)
    floors=project(points*np.array([1,1,0]))*scale
    footprint=cv2.convexHull(floors.astype(np.float32)).reshape(-1,2)
    contact_xy=np.asarray(feet)
    if contact_xy.ndim==1:
        contact_xy=np.c_[contact_xy,np.zeros(len(feet))]
    contacts=project((np.c_[contact_xy,np.zeros(len(feet))]@matrix.T)*[1,1,0])*scale
    axis=project((np.array([[-.44,0,.16],[.44,0,.16]])@matrix.T)-[0,0,zmin])*scale
    both=np.vstack((hull,footprint))
    left,top=np.floor(both.min(axis=0)).astype(int)-2
    right,bottom=np.ceil(both.max(axis=0)).astype(int)+3
    shape=(bottom-top,right-left)
    low=np.full(shape,np.inf,np.float32)
    high=np.full(shape,-np.inf,np.float32)
    mesh=[]
    for vertices,faces in shapes:
        xy=project(vertices)*scale
        for face in faces:
            ids=list(face)
            mesh.append((xy[ids],float(np.mean(-vertices[ids,1]+vertices[ids,2]))))
            for j in range(1,len(ids)-1):
                indices=[ids[0],ids[j],ids[j+1]]
                corners=xy[indices]-[left,top]
                matrix2=np.vstack([corners.T,[1,1,1]])
                if abs(np.linalg.det(matrix2))<.02:
                    continue
                v=np.linalg.inv(matrix2)
                tx,ty=np.maximum(np.floor(corners.min(axis=0)).astype(int),0)
                tr,tb=np.minimum(np.ceil(corners.max(axis=0)).astype(int)+1,shape[::-1])
                yy,xx=np.ogrid[ty:tb,tx:tr]
                a,b,c=(v[i,0]*xx+v[i,1]*yy+v[i,2] for i in range(3))
                inside=(a>=-.001)&(b>=-.001)&(c>=-.001)
                z=vertices[indices,2]*scale
                height=a*z[0]+b*z[1]+c*z[2]
                near,far=low[ty:tb,tx:tr],high[ty:tb,tx:tr]
                near[inside]=np.minimum(near[inside],height[inside])
                far[inside]=np.maximum(far[inside],height[inside])
    solid=np.isfinite(low)
    ground=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(ground,np.rint(footprint-[left,top]).astype(np.int32),1)
    return Projection(hull,mesh,contacts,footprint,(int(left),int(top),int(right),int(bottom)),
                      np.where(solid,low,0),np.where(solid,high,0),solid,ground>0,axis)


@dataclass
class Surface:
    piece: int
    mask: np.ndarray
    near: np.ndarray
    distance: np.ndarray
    plane: np.ndarray
    error: float
    bounds: tuple


class PoseAtlas:
    def __init__(self, rgb, depth, masks, size):
        self.rgb,self.depth,self.masks,self.size=rgb,depth,masks,size
        h,w=depth.shape
        self.factor=np.array([size[0]/w,size[1]/h])
        self.safe=np.ones((h,w),bool)
        self.safe[:math.ceil(h*.09)+2]=False
        self.safe[math.floor(h*.82)-2:]=False
        self.safe[:,:3]=False
        self.safe[:,-3:]=False
        self._clearance={}
        self._context_maps={}
        self._boundary_normals={}
        self._relief={}
        # Detect the bright, long vertical separator of the scrap bay.
        neutral=np.min(rgb,axis=2)>215
        vertical=neutral[round(h*.13):round(h*.79)].mean(axis=0)
        rail=np.flatnonzero((vertical>.42)&(np.arange(w)<w*.36)&(np.arange(w)>w*.13))
        if len(rail)>=8:
            self.safe[:,:rail[-1]+7]=False
        self.surfaces=[]
        yy,xx=np.indices(depth.shape,dtype=np.float32)
        for i,mask in enumerate(masks):
            area=mask&self.safe
            if not 240<=area.sum()<=.12*area.size:
                continue
            dist=cv2.distanceTransform(area.astype(np.uint8),cv2.DIST_L2,3)
            if dist.max()<5.2:
                continue
            fit,error=plane_fit(depth,area)
            if error>.034:
                continue
            plane=(fit[0]+fit[1]*xx+fit[2]*yy).astype(np.float32)
            near=cv2.dilate(area.astype(np.uint8),np.ones((5,5),np.uint8))>0
            surface=Surface(i,area,near,dist,plane,error,cv2.boundingRect(area.astype(np.uint8)))
            self.surfaces.append(surface)

    def screen(self, points):
        return np.rint(np.asarray(points)*self.factor).astype(int).tolist()

    def open_clearance(self, surface):
        if surface.piece not in self._clearance:
            raised=(self.depth>surface.plane+max(.018,2.2*surface.error)).astype(np.uint8)
            raised=cv2.morphologyEx(raised,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
            self._clearance[surface.piece]=cv2.distanceTransform(
                (~(raised | ~self.safe)).astype(np.uint8),cv2.DIST_L2,3)
        return self._clearance[surface.piece]

    def clearance_crop(self, support_id, crop):
        surface=next((s for s in self.surfaces if f"S{s.piece}"==support_id),None)
        if surface is None:
            raise NeedsReview("原有空位的承托面无法重现")
        l,t,r,b=crop
        return cv2.resize(self.open_clearance(surface),self.size,interpolation=cv2.INTER_LINEAR)[t:b,l:r]*min(self.factor)

    def context_maps(self, major):
        radius=max(2,round(.10*major))
        if radius not in self._context_maps:
            gray=cv2.cvtColor(self.rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            detail=np.abs(gray-cv2.GaussianBlur(gray,(0,0),1.55))
            texture=cv2.GaussianBlur((detail>.039).astype(np.float32),(0,0),radius)
            texture/=max(.08,float(np.percentile(texture[self.safe],93)))
            light=cv2.GaussianBlur(gray,(0,0),radius)
            glare=cv2.GaussianBlur((gray>.68).astype(np.float32),(0,0),radius)
            self._context_maps[radius]=(texture,light,glare)
        return self._context_maps[radius]

    def position_context(self, surface, region, silhouette, major, direction=None, *, measure_relief=False):
        texture,light,glare=self.context_maps(major)
        distance=surface.distance[region][silhouette]
        near_edge=np.clip(1-distance/max(5,.26*major),0,1)
        edge=float(near_edge.mean())
        activity=float(np.clip(texture[region][silhouette].mean(),0,1))
        brightness=float(light[region][silhouette].mean())
        reflected=float(glare[region][silhouette].mean())
        bright_penalty=.46*np.clip((brightness-.29)/.38,0,1)+.54*min(1,reflected/.46)
        ys,xs=np.nonzero(silhouette)
        width,height=self.depth.shape[::-1]
        x=(region[1].start+xs.mean())/width
        y=(region[0].start+ys.mean())/height
        focus=math.exp(-5*((x-.5)**2+(y-.5)**2))
        alignment=0.
        if direction is not None and near_edge.sum()>1:
            if surface.piece not in self._boundary_normals:
                self._boundary_normals[surface.piece]=(
                    cv2.Sobel(surface.distance,cv2.CV_32F,1,0),
                    cv2.Sobel(surface.distance,cv2.CV_32F,0,1))
            nx,ny=(value[region][silhouette] for value in self._boundary_normals[surface.piece])
            length=np.maximum(np.hypot(nx,ny),.01)
            parallel=np.abs(-ny*direction[0]+nx*direction[1])/length
            alignment=float(np.average(parallel,weights=near_edge))
        cost=.43*(1-min(1,edge/.65))+.27*(1-activity)+.23*bright_penalty+.07*focus
        cost-=.08*edge*alignment
        context={"cost":float(cost),"near_steel_edge":edge,"texture":activity,"brightness":brightness,
                 "glare":reflected,"edge_alignment":alignment}
        if measure_relief:
            if surface.piece not in self._relief:
                self._relief[surface.piece]=ReliefField(self.depth,surface.plane,self.safe,surface.error)
            bounds=(region[1].start+int(xs.min()),region[0].start+int(ys.min()),
                    region[1].start+int(xs.max())+1,region[0].start+int(ys.max())+1)
            context["relief"]=self._relief[surface.piece].measure(bounds,major)
        return context

    def native_position_context(self,support_id,x,y,solid,target,direction,*,measure_relief=False):
        surface=next((s for s in self.surfaces if f"S{s.piece}"==support_id),None)
        if surface is None:
            raise NeedsReview("没有可用的承托面用于比较选位")
        h,w=solid.shape
        l,t,r,b=np.rint(np.asarray([x,y,x+w,y+h])/np.tile(self.factor,2)).astype(int)
        mask=cv2.resize(solid.astype(np.uint8)*255,(r-l,b-t),interpolation=cv2.INTER_AREA)>64
        return self.position_context(surface,np.s_[t:b,l:r],mask,target/self.factor[0],direction,
                                     measure_relief=measure_relief)

    def candidates(self, kind, parts, feet, targets, *, placement_mode="scrap", discreet=False,
                   previous_center=None, prefer_recess=False):
        all_candidates=[]
        reasons=Counter()
        conflict=None
        w=self.depth.shape[1]
        # Multiple genuinely different spatial attitudes, with roll around the part's long axis.
        poses=[(-68,0,15),(-44,0,-15),(-21,0,25),(10,0,-20),(31,0,15),(55,0,-20),(75,0,15)]
        if kind=="bearing":
            poses=[(yaw,0,0) for yaw in (-51,-31,-12,16,35,52)]
        open_bed=placement_mode=="open"
        previous_center=np.asarray(previous_center)/self.factor if previous_center is not None else None
        if open_bed or kind=="jack_upright":
            poses=[(yaw,0,0) for yaw in (-72,-51,-31,-12,0,16,35,52,70)]
        for rank,target in enumerate(targets):
            major=target/self.factor[0]
            stride=max(4,min(26,round(major*.13)))
            for yaw,pitch,roll in poses:
                p=make_projection(parts,feet,yaw,pitch,roll,major)
                direction=None
                if discreet and kind in ("ram","shock"):
                    axis=p.axis[1]-p.axis[0]
                    direction=axis/np.linalg.norm(axis)
                l,t,r,b=p.extent
                count=p.solid.sum()
                for surface in self.surfaces:
                    sx,sy,sw,sh=surface.bounds
                    # Search each recognized surface at a density tied to projected size.
                    iy,ix=np.unravel_index(np.argmax(surface.distance),surface.distance.shape)
                    centers={(int(ix),int(iy))}
                    centers.update((x,y) for y in range(sy+stride//2,sy+sh,stride)
                                   for x in range(sx+stride//2,sx+sw,stride) if surface.distance[y,x]>=2.5)
                    for x,y in sorted(centers):
                        if discreet and previous_center is not None and math.dist((x,y),previous_center)<.21*major:
                            continue
                        left,top,right,bottom=x+l,y+t,x+r,y+b
                        if min(left,top)<0 or right>w or bottom>self.depth.shape[0]:
                            reasons["outside_frame"]+=1
                            continue
                        region=np.s_[top:bottom,left:right]
                        bed=surface.mask if open_bed else surface.near
                        ground_fit=np.count_nonzero(p.ground&bed[region])/p.ground.sum()
                        protected=1-float(self.safe[region][p.solid|p.ground].mean())
                        if protected:
                            reasons["protected_or_wall"]+=1
                            score=ground_fit*(1-protected)-protected
                            if conflict is None or score>conflict[0]:
                                conflict=(score,p,x,y,surface.piece,target,ground_fit,protected)
                            continue
                        foot=np.rint(p.contacts+[x,y]).astype(int)
                        contact=surface.distance[foot[:,1],foot[:,0]]
                        minimum_fit=.90 if open_bed else (.60 if kind in ("bearing","tank") else .50)
                        if min(contact)<(2. if open_bed else 1.) or ground_fit<minimum_fit:
                            reasons["insufficient_bed"]+=1
                            if conflict is None or ground_fit>conflict[0]:
                                conflict=(ground_fit,p,x,y,surface.piece,target,ground_fit,0.)
                            continue
                        if open_bed:
                            clearance=float(self.open_clearance(surface)[region][p.solid|p.ground].min())
                            if clearance<max(1.5,major*.025):
                                reasons["open_clearance"]+=1
                                continue
                            score=ground_fit*.55+min(float(min(contact))/max(6,major*.10),1)*.20
                            score+=min(clearance/max(6,major*.13),1)*.21
                            score-=surface.error*3+rank*.06
                            if discreet:
                                on_plate=float((p.solid&surface.mask[region]).sum()/count)
                                if on_plate<.81:
                                    reasons["projected_outline_off_plate"]+=1
                                    continue
                                context=self.position_context(surface,region,p.solid,major,direction,
                                                              measure_relief=prefer_recess)
                                score-=.90*context["cost"]
                                score-=.46*(1-on_plate)
                                if prefer_recess:
                                    terrain=context["relief"]
                                    if terrain["kind"]=="raised":
                                        reasons["raised_surface"]+=1
                                        continue
                                    score+=.58*(terrain["kind"]=="recess")+.32*relief_value(terrain)
                            all_candidates.append((score,p,x,y,surface,[],0.,ground_fit,0.,0.,target,(yaw,pitch,roll)))
                            continue
                        predicted=surface.plane[region]
                        observed=self.depth[region]
                        noise=max(STEP_MARGIN,surface.error*1.7)
                        # A relative depth gradient along the bed is the local depth conversion.
                        step=float(np.clip((surface.plane[min(y+1,self.depth.shape[0]-1),x]-
                                            surface.plane[y,x])/math.cos(math.radians(ELEVATION)),.001,.0042))
                        lo,hi=predicted+p.low*step,predicted+p.high*step
                        above=observed>hi+noise
                        conflict_px=p.solid & ~surface.near[region] & (observed>lo+noise) & ~above
                        collision=conflict_px.sum()/count
                        if collision>.026:
                            reasons["depth_collision"]+=1
                            continue
                        fronts=[]
                        front_mask=np.zeros_like(p.solid)
                        for n,m in enumerate(self.masks):
                            if n==surface.piece:
                                continue
                            local=m[region]
                            overlap=local&p.solid
                            if overlap.sum()<max(4,count*.020):
                                continue
                            close=float(np.mean(above[overlap]))
                            ambiguous=float(np.mean(conflict_px[overlap]))
                            if close<.74 or ambiguous>.16 or not crossing(p.solid.astype(np.uint8)*255,local.astype(np.uint8)*255):
                                continue
                            if (overlap&surface.mask[region]).sum()>.20*overlap.sum():
                                continue
                            fronts.append(n)
                            front_mask|=local
                        cover=np.count_nonzero(front_mask&p.solid)/count
                        unknown=np.count_nonzero(above&p.solid&~front_mask&~surface.near[region])/count
                        if unknown>.035:
                            reasons["unresolved_depth"]+=1
                            continue
                        if not .025<=cover<=.58:
                            reasons["no_continuous_foreground"]+=1
                            continue
                        score=ground_fit*.53+min(float(min(contact))/max(8,major*.10),1)*.20
                        score-=abs(cover-.19)*.90+rank*.06+collision*2.2+unknown*1.7
                        if discreet:
                            context=self.position_context(surface,region,p.solid,major,direction,
                                                          measure_relief=prefer_recess)
                            score-=.77*context["cost"]
                            if prefer_recess:
                                terrain=context["relief"]
                                if terrain["kind"]=="raised":
                                    reasons["raised_surface"]+=1
                                    continue
                                score+=.58*(terrain["kind"]=="recess")+.32*relief_value(terrain)
                        all_candidates.append((score,p,x,y,surface,fronts,cover,ground_fit,collision,unknown,target,(yaw,pitch,roll)))
        ranked=sorted(all_candidates,key=lambda item:item[0],reverse=True)
        selected=[]
        for item in ranked:
            _,_,x,y,surface,_,_,_,_,_,target,pose=item
            spacing=max(10,target/self.factor[0]*.25) if discreet else max(25,target/self.factor[0]*.45)
            if any(math.hypot(x-q[2],y-q[3])<spacing
                   and ((not discreet and surface.piece==q[4].piece) or abs(pose[0]-q[-1][0])<28) for q in selected):
                continue
            selected.append(item)
            if len(selected)==9:
                break
        return selected,dict(reasons),conflict


def refine_foreground(atlas, frame, engine, selected):
    _,p,x,y,surface,fronts,*_=selected
    xy=atlas.screen(p.hull+[x,y])
    q=np.asarray(xy)
    pad=max(60,round(np.ptp(q,axis=0).max()*.18))
    l,t=np.maximum(0,q.min(axis=0)-pad)
    r,b=np.minimum(frame.size,q.max(axis=0)+pad)
    crop=(int(l),int(t),int(r),int(b))
    original=frame.crop(crop)
    rgb=np.asarray(original)
    full=np.logical_or.reduce([atlas.masks[n] for n in fronts])
    prior=cv2.resize(full.astype(np.uint8),frame.size,interpolation=cv2.INTER_NEAREST)[t:b,l:r]>0
    expected=np.zeros(prior.shape,np.uint8)
    cv2.fillConvexPoly(expected,np.asarray(xy,np.int32)-[l,t],255)
    edge=original_edge_neighborhood(rgb)
    if boundary_agreement(expected,prior.astype(np.uint8)*255,edge)<.65:
        try:
            changed=engine.refine_crop(original,prior.astype(np.uint8)*255)
            if boundary_agreement(expected,changed.astype(np.uint8)*255,edge)>boundary_agreement(expected,prior.astype(np.uint8)*255,edge):
                prior=changed
        except (ValueError,RuntimeError,NeedsReview):
            pass
    agreement=boundary_agreement(expected,prior.astype(np.uint8)*255,edge)
    if agreement<.46 or not crossing(expected,prior.astype(np.uint8)*255):
        return None
    mask=cv2.GaussianBlur(prior.astype(np.uint8)*255,(3,3),.58)
    return crop,mask,agreement


def draw_projection(atlas, frame, p, x, y, crop, *, front=None, fail=False):
    """Show the spatial proxy separately from the actual render."""
    l,t,r,b=crop
    overlay=frame.crop(crop).convert("RGBA")
    ink=Image.new("RGBA",overlay.size)
    draw=ImageDraw.Draw(ink)
    def points(xy):
        return [(a-l,b-t) for a,b in atlas.screen(xy+[x,y])]
    for coords,z in sorted(p.mesh,key=lambda item:item[1]):
        color=(230,76,60,120) if fail else (31,119+int(np.clip(z*75,-20,60)),208,170)
        draw.polygon(points(coords),fill=color)
    draw.line(points(p.hull)+points(p.hull[:1]),fill=(255,120,83,255) if fail else (182,227,255,255),width=4)
    draw.line(points(p.axis),fill="#ffcc60",width=5)
    for i,(u,v) in enumerate(points(p.contacts),1):
        draw.ellipse((u-7,v-7,u+7,v+7),fill="#9be96a",outline="#19351c",width=2)
        draw.text((u+9,v),f"C{i}",fill="white",stroke_fill="black",stroke_width=2)
    result=Image.alpha_composite(overlay,ink)
    if front is not None:
        front_color=Image.new("RGBA",result.size,(32,255,192,0))
        front_color.putalpha(Image.fromarray((front.astype(np.float32)*.53).astype(np.uint8)))
        result=Image.alpha_composite(result,front_color)
    return result.convert("RGB")
