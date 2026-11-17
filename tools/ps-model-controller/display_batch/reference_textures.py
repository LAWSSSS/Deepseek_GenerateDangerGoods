"""Conservative offline extraction of estimated surface maps from six-view sheets.

These are appearance estimates, not measured PBR or evidence of real displacement.
All maps share quilting coordinates. Source pictures and source assets are untouched.
"""
from collections import Counter
from pathlib import Path
import json
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .common import atomic_json, file_hash, signature

VERSION = 'reference-pbr-v1'
ALGORITHM = 'surface-quilting-v5'
MAPS = ('albedo', 'roughness', 'oxide', 'abrasion', 'height', 'dust')


def srgb_decode(value):
    value=np.asarray(value,np.float32)
    return np.where(value <= .04045,value/12.92,((value+.055)/1.055)**2.4)


def srgb_encode(value):
    value=np.maximum(value,0)
    return np.where(value <= .0031308,12.92*value,1.055*value**(1/2.4)-.055)


def panel_boxes(rgb):
    """Find continuous gray divider lines near the expected 3 by 2 grid."""
    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    def line(axis, target, radius):
        length=w if axis==0 else h
        start=max(1,int(target-radius)); end=min(length-1,int(target+radius))
        strip=gray[:,start:end] if axis==0 else gray[start:end,:]
        mean=strip.mean(axis=axis);std=strip.std(axis=axis)
        candidates=np.where((std<5)&(mean>170)&(mean<250))[0]
        if not len(candidates):return int(target)
        score=std[candidates]+np.abs(candidates+start-target)/length*.5
        return int(candidates[np.argmin(score)]+start)
    xs=[0,line(0,w/3,w*.06),line(0,2*w/3,w*.06),w]
    ys=[0,line(1,h/2,h*.15),h]
    return [(xs[x],ys[y],xs[x+1],ys[y+1]) for y in range(2) for x in range(3)]


def foreground(rgb):
    """Edge-connected near-white background; keep large connected objects only."""
    f=rgb.astype(np.float32);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    white=((f.min(2)>205)&(f.max(2)-f.min(2)<25)).astype(np.uint8)
    count,labels=cv2.connectedComponents(white,connectivity=8)
    border=np.unique(np.r_[labels[0,:],labels[-1,:],labels[:,0],labels[:,-1]])
    border=border[border!=0]
    bg=np.isin(labels,border)
    mask=(~bg).astype(np.uint8)
    margin=max(4,round(min(rgb.shape[:2])*.012))
    mask[:margin]=0;mask[-max(margin,int(rgb.shape[0]*.10)):]=0
    mask[:,:margin]=0;mask[:,-margin:]=0
    count,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
    keep=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=max(96,rgb.shape[0]*rgb.shape[1]*.006)]
    mask=np.isin(components,keep).astype(np.uint8)
    return mask,gray


def text_regions(gray):
    """Find repeated glyph-sized components on text baselines; deliberately conservative."""
    residual=np.abs(gray.astype(np.float32)-cv2.GaussianBlur(gray.astype(np.float32),(0,0),3))
    ink=(residual>35).astype(np.uint8)
    count,_,stats,centers=cv2.connectedComponentsWithStats(ink,8)
    boxes=[]
    for x,y,w,h,area in stats[1:]:
        if 3<=w<=65 and 5<=h<=50 and .10<=area/(w*h)<=.8 and area>=8:
            boxes.append((int(x),int(y),int(w),int(h)))
    result=np.zeros(gray.shape,np.uint8)
    for x,y,w,h in boxes:
        row=[b for b in boxes if abs((b[1]+b[3]*.5)-(y+h*.5))<max(4,h*.45)
             and .5*h<=b[3]<=1.8*h and abs(b[0]-x)<h*5]
        if len(row)>=4 and max(b[0]+b[2] for b in row)-min(b[0] for b in row)>h*2:
            for xx,yy,ww,hh in row:
                result[max(0,yy-3):yy+hh+3,max(0,xx-3):xx+ww+3]=1
    return cv2.dilate(result,np.ones((5,5),np.uint8))


def structural_edge(rgb):
    """Reject broad step edges while keeping thin scratches with similar sides."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    side=gray.shape[0]
    smooth=cv2.GaussianBlur(gray,(0,0),1.3)
    small_edges=cv2.Canny(cv2.GaussianBlur(gray,(0,0),.6),25,55)
    contours,_=cv2.findContours(small_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter=cv2.arcLength(contour,True)
        if perimeter<22:continue
        polygon=cv2.approxPolyDP(contour,.035*perimeter,True)
        x,y,w,h=cv2.boundingRect(contour);area=abs(cv2.contourArea(contour))
        if (4<=len(polygon)<=6 and min(w,h)>=4 and max(w,h)>10 and
                max(w,h)/min(w,h)>1.4 and area>30 and area/(w*h)>.48):
            return True
    coherent=cv2.Canny(smooth,25,60)
    straight=cv2.HoughLinesP(coherent,1,np.pi/180,threshold=max(14,side//5),
                            minLineLength=side*.52,maxLineGap=3)
    if straight is not None:return True
    circles=cv2.HoughCircles(smooth,cv2.HOUGH_GRADIENT,1,minDist=side/3,
                            param1=70,param2=20,minRadius=8,maxRadius=int(side*.4))
    if circles is not None:return True
    edges=cv2.Canny(gray,65,140)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(22,side//3),
                         minLineLength=side*.65,maxLineGap=6)
    if lines is None:return False
    for x1,y1,x2,y2 in lines[:,0]:
        dx,dy=x2-x1,y2-y1;length=max(np.hypot(dx,dy),1.)
        xx=np.linspace(x1,x2,24);yy=np.linspace(y1,y2,24)
        nx,ny=-dy/length*4,dx/length*4
        def sample(sign):
            return gray[np.clip(np.rint(yy+sign*ny).astype(int),0,side-1),
                        np.clip(np.rint(xx+sign*nx).astype(int),0,side-1)].astype(float)
        delta=np.abs(sample(1)-sample(-1))
        if np.mean(delta>28)>.65:return True
    return False


def dark_structure(gray):
    smooth=cv2.GaussianBlur(gray.astype(np.float32),(0,0),1.5)
    ink=(smooth<max(40,float(np.median(smooth))*.72)).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    side=len(gray)
    for x,y,w,h,area in stats[1:]:
        if area>side*side*.012 and max(w,h)>side*.25 and max(w,h)/max(1,min(w,h))>1.8:
            return True
    return False


def recessed_slot(gray):
    local=cv2.GaussianBlur(gray.astype(np.float32),(0,0),3)
    dark=(local-gray.astype(np.float32)>17).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(dark,8)
    for x,y,w,h,area in stats[1:]:
        if min(w,h)>=3 and max(w,h)>=9 and max(w,h)/min(w,h)>2 and area>15 and area/(w*h)>.3:
            return True
    return False


def select_patches(rgb, seed=173, patch_size=None):
    boxes=panel_boxes(rgb); panels=[]; excluded=Counter(); candidates=[]
    min_width=min(b[2]-b[0] for b in boxes)
    size=patch_size or 64
    if size<64:raise ValueError('source patches must be at least 64 pixels; no synthetic resolution upscaling')
    colored=[];foreground_pixels=0;paint_pixels=0
    for number,(x,y,r,b) in enumerate(boxes,1):
        panel=rgb[y:b,x:r];fg,gray=foreground(panel);text=text_regions(gray)
        hsv=cv2.cvtColor(panel,cv2.COLOR_RGB2HSV)
        paint=((hsv[...,0]<10)|(hsv[...,0]>80))&(hsv[...,1]>85)&(hsv[...,2]>65)&(fg>0)
        foreground_pixels+=int(fg.sum());paint_pixels+=int(paint.sum())
        colored.extend(hsv[...,0][paint][::max(1,int(paint.sum()/3000))].tolist())
        panels.append((number,(x,y,r,b),panel,fg,gray,text,hsv))
    # Saturated red/blue/green paint is separated from brown oxidation and gray metal.
    if colored:
        hist=np.bincount(np.array(colored)//10,minlength=18)
        dominant=int(hist.argmax())*10+5
        paint_mode=hist.max()>700 and paint_pixels/max(foreground_pixels,1)>.25
    else:dominant=0;paint_mode=False
    for number,bounds,panel,fg,gray,text,hsv in panels:
        eroded=cv2.erode(fg,np.ones((9,9),np.uint8))
        for y in range(8,panel.shape[0]-size-8,max(16,size//4)):
            for x in range(8,panel.shape[1]-size-8,max(16,size//4)):
                cut=panel[y:y+size,x:x+size];g=gray[y:y+size,x:x+size]
                reason=None
                if eroded[y:y+size,x:x+size].mean()<.999:reason='background_edge_caption_or_hole'
                elif text[y:y+size,x:x+size].mean()>.025:reason='suspected_text_or_label'
                elif np.mean(g>239)>.005:reason='strong_highlight_or_white_label'
                elif cv2.connectedComponentsWithStats((g<24).astype(np.uint8),8)[2][1:,cv2.CC_STAT_AREA].max(initial=0)>size*size*.015:reason='black_seam_or_deep_shadow'
                elif paint_mode and (dark_structure(g) or recessed_slot(g)):reason='long_dark_structure_or_mixed_material'
                elif structural_edge(cut):reason='long_structural_edge'
                if reason:
                    excluded[reason]+=1;continue
                hcut=hsv[y:y+size,x:x+size]
                distance=np.abs(hcut[...,0].astype(float)-dominant);distance=np.minimum(distance,180-distance)
                paint_share=float(((distance<16)&(hcut[...,1]>65)).mean())
                if paint_mode and paint_share<.96:
                    excluded['different_material_family']+=1;continue
                if not paint_mode and np.mean(((hcut[...,0]>50)|(hcut[...,0]<7))&(hcut[...,1]>50))>.025:
                    excluded['colored_label_or_different_material']+=1;continue
                # Broad illumination slopes are penalized, not mistaken for interesting texture.
                smooth=cv2.GaussianBlur(g.astype(float),(0,0),size/7)
                slope=float(np.std(smooth));detail=float(np.std(g-smooth))
                cells=[g[yy:yy+8,xx:xx+8].std() for yy in range(0,size,8) for xx in range(0,size,8)]
                if np.mean(np.array(cells)<2.5)>.08:
                    excluded['smooth_background_or_baked_reflection']+=1;continue
                if slope>14:
                    excluded['broad_structural_shading']+=1;continue
                score=detail/(1+slope*.25)
                candidates.append({'panel':number,'roi':[bounds[0]+x,bounds[1]+y,bounds[0]+x+size,bounds[1]+y+size],
                                   'panel_bounds':list(bounds),'score':score,'detail_255':detail,
                                   'low_frequency_std_255':slope,'paint_share':paint_share,'pixels':cut})
    selected=[];by_panel={i:sorted([c for c in candidates if c['panel']==i],key=lambda c:-c['score']) for i in range(1,7)}
    for round_id in range(3):
        for number in range(1,7):
            for candidate in by_panel[number]:
                x,y,r,b=candidate['roi']
                if any(s['panel']==number and max(0,min(r,s['roi'][2])-max(x,s['roi'][0]))*
                       max(0,min(b,s['roi'][3])-max(y,s['roi'][1]))>size*size*.3 for s in selected):continue
                selected.append(candidate);break
    return selected,{'patch_size_native_px':size,'candidates':len(candidates),'selected':len(selected),
        'panels_used':sorted({c['panel'] for c in selected}),'rejected':dict(excluded),
        'material_selection':'colored_paint' if paint_mode else 'neutral_metal_or_oxide',
        'dominant_paint_hue':dominant if paint_mode else None,
        'colored_paint_fraction_of_foreground':paint_pixels/max(foreground_pixels,1),
        'panel_bounds':[list(b) for b in boxes], 'seed':seed}


def remove_planar_illumination(rgb):
    linear=srgb_decode(rgb.astype(np.float32)/255)
    y=linear@np.array([.2126,.7152,.0722],np.float32)
    yy,xx=np.mgrid[-1:1:complex(y.shape[0]),-1:1:complex(y.shape[1])]
    basis=np.stack([np.ones_like(xx),xx,yy],-1)
    smooth=cv2.GaussianBlur(y,(0,0),max(3,y.shape[0]/8))
    coef=np.linalg.lstsq(basis.reshape(-1,3),smooth.ravel(),rcond=None)[0]
    plane=basis@coef
    ratio=np.clip(np.median(plane)/np.maximum(plane,.025),.8,1.25)
    return np.clip(linear*ratio[...,None],0,1).astype(np.float32)


def estimated_maps(rgb, paint=False, protect_paint=False):
    albedo=remove_planar_illumination(rgb)
    srgb=np.uint8(np.clip(srgb_encode(albedo)*255,0,255))
    hsv=cv2.cvtColor(srgb,cv2.COLOR_RGB2HSV).astype(np.float32)
    h,s,v=hsv[...,0],hsv[...,1]/255,hsv[...,2]/255
    oxide=np.clip((s-.18)/.5,0,1)*np.clip((h-6)/7,0,1)*np.clip((42-h)/13,0,1)*np.clip((.92-v)/.25,0,1)
    if paint and protect_paint:
        # The dominant chromatic coating is not rust. Only a different brown
        # family can suggest oxide; hue alone cannot prove physical chemistry.
        valid=(s>.3)&(v>.2)
        if valid.any():
            dominant=float(np.bincount(h[valid].astype(int),minlength=180).argmax())
            distance=np.abs(h-dominant);distance=np.minimum(distance,180-distance)
            # Brown rust can lie close to a red coating. Use departure from the
            # coating family, not a fixed orange cutoff or darkness as rust.
            brown=np.clip((h-3)/4,0,1)*np.clip((42-h)/13,0,1)
            oxide=(np.clip((s-.18)/.5,0,1)*brown*np.clip((.92-v)/.25,0,1)*
                   np.clip((distance-1.5)/4,0,1))
    oxide=cv2.GaussianBlur(oxide,(0,0),.6)
    lum=albedo@np.array([.2126,.7152,.0722],np.float32)
    yy,xx=np.mgrid[-1:1:complex(lum.shape[0]),-1:1:complex(lum.shape[1])]
    basis=np.stack([np.ones_like(xx),xx,yy,xx*xx,yy*yy,xx*yy],-1)
    trend=basis@np.linalg.lstsq(basis.reshape(-1,6),lum.ravel(),rcond=None)[0]
    residual=(lum-trend).astype(np.float32)
    fine=cv2.GaussianBlur(residual,(0,0),.7)-cv2.GaussianBlur(residual,(0,0),2.2)
    margin=max(5,int(lum.shape[0]*.08));fine[:margin]=0;fine[-margin:]=0;fine[:,:margin]=0;fine[:,-margin:]=0
    # Fixed amplitude, never normalize a flat gradient into invented bump.
    height=.5+np.clip(fine,-.05,.05)*2.
    abrasion=np.clip((fine-.008)/.06,0,1)*(1-oxide*.6)
    if paint and protect_paint and np.any((s>.3)&(v>.2)):
        # A neutral island in chromatic paint is a candidate exposed chip,
        # including its interior. Bright edges alone are not a wear mask.
        coat_saturation=float(np.median(s[(s>.3)&(v>.2)]))
        neutral_chip=(np.clip((coat_saturation-s-.20)/.35,0,1)*
                      np.clip((v-.2)/.25,0,1)*np.clip((.9-v)/.15,0,1))
        abrasion=np.maximum(abrasion,neutral_chip*(1-oxide))
    dust=np.clip((.36-s)/.36,0,1)*np.clip((v-.3)/.45,0,1)*.45
    roughness=np.clip((.52 if paint else .38)+oxide*.43+dust*.16+np.clip(np.abs(fine)*.6,0,.04)-abrasion*.12,.3,.93)
    return np.dstack((albedo,roughness,oxide,abrasion,height,dust)).astype(np.float32)


def _seam(error):
    cost=error.astype(float).copy();parents=np.zeros(error.shape,np.int32)
    for y in range(1,cost.shape[0]):
        previous=np.pad(cost[y-1],(1,1),constant_values=np.inf)
        candidates=np.stack((previous[:-2],previous[1:-1],previous[2:]))
        move=np.argmin(candidates,axis=0)-1
        parents[y]=np.arange(cost.shape[1])+move
        cost[y]+=candidates[move+1,np.arange(cost.shape[1])]
    path=np.zeros(cost.shape[0],int);path[-1]=np.argmin(cost[-1])
    for y in range(len(path)-1,0,-1):path[y-1]=parents[y,path[y]]
    return path


def quilt(patches, size=512, seed=173, paint=False):
    if size!=512:raise ValueError('reference texture resolution is fixed at 512; native samples are not upscaled')
    rng=np.random.default_rng(seed)
    tiles=[estimated_maps(p['pixels'],paint) for p in patches]
    block=tiles[0].shape[0];overlap=block//4;stride=block-overlap
    canvas=np.zeros((size+block,size+block,8),np.float32)
    placements=[]
    for y in range(0,size,stride):
        for x in range(0,size,stride):
            variants=[]
            for index,tile in enumerate(tiles):
                k=int(rng.integers(4));flip=bool(rng.integers(2));t=np.rot90(tile,k)
                if flip:t=t[:,::-1]
                error=0.
                if x:error+=float(np.mean((canvas[y:y+block,x:x+overlap,:3]-t[:,:overlap,:3])**2))
                if y:error+=float(np.mean((canvas[y:y+overlap,x:x+block,:3]-t[:overlap,:,:3])**2))
                variants.append((error,index,k,flip,t))
            variants.sort(key=lambda item:item[0])
            _,index,k,flip,t=variants[int(rng.integers(min(3,len(variants))))]
            weight=np.ones((block,block),np.float32)
            if x:
                error=np.mean((canvas[y:y+block,x:x+overlap,:3]-t[:,:overlap,:3])**2,axis=2)
                seam=_seam(error)
                weight[:,:overlap]=np.clip((np.arange(overlap)[None,:]-seam[:,None]+1)/3,0,1)
            if y:
                error=np.mean((canvas[y:y+overlap,x:x+block,:3]-t[:overlap,:,:3])**2,axis=2)
                seam=_seam(error.T)
                weight[:overlap]=np.minimum(weight[:overlap],np.clip((np.arange(overlap)[:,None]-seam[None,:]+1)/3,0,1))
            window=canvas[y:y+block,x:x+block]
            window[:]=window*(1-weight[...,None])+t*weight[...,None]
            placements.append({'xy':[x,y],'source_index':index,'quarter_turns':k,'flip_x':flip})
    result=canvas[:size,:size].copy()
    # Periodic correction shares the exact same weights for every map.
    width=12
    for i in range(width):
        alpha=.5*(1-i/width)**2
        a=result[:,i].copy();b=result[:,-1-i].copy()
        result[:,i]=a*(1-alpha)+b*alpha;result[:,-1-i]=b*(1-alpha)+a*alpha
        a=result[i].copy();b=result[-1-i].copy()
        result[i]=a*(1-alpha)+b*alpha;result[-1-i]=b*(1-alpha)+a*alpha
    return result,placements


def seam_diagnostics(albedo, stride):
    gx=np.mean(np.abs(np.diff(albedo,axis=1)),axis=2);gy=np.mean(np.abs(np.diff(albedo,axis=0)),axis=2)
    general=float((gx.mean()+gy.mean())*.5)
    rows=list(range(stride-1,len(albedo)-1,stride))
    grid=float((gx[:,rows].mean()+gy[rows].mean())*.5) if rows else general
    return {'periodic_edge_max_linear':float(max(np.max(np.abs(albedo[0]-albedo[-1])),
                                                np.max(np.abs(albedo[:,0]-albedo[:,-1])))),
            'interior_grid_gradient_ratio':grid/max(general,1e-6),
            'interior_mean_gradient_linear':general,'seam_blend_width_px':12}


def prepare_reference(path, output, *, seed=173):
    path=Path(path).resolve(strict=True);output=Path(output).resolve()
    input_sha=file_hash(path)
    key=signature(ALGORITHM,input_sha,seed,512,cv2.__version__,np.__version__)
    output.mkdir(parents=True,exist_ok=True)
    record_path=output/'reference.json'
    if record_path.exists():
        old=json.loads(record_path.read_text(encoding='utf8'))
        if old.get('signature')!=key:raise ValueError('changed inputs or extraction algorithm require a new output directory')
        if all(Path(m['path']).is_file() and file_hash(m['path'])==m['sha256'] for m in old.get('maps',{}).values()):
            return {**old,'cache_hit':True}
        raise ValueError('cached reference maps missing or modified; use a new output directory')
    rgb=np.array(Image.open(path).convert('RGB'))
    selected,stats=select_patches(rgb,seed)
    record={'version':VERSION,'algorithm':ALGORITHM,'signature':key,'input_sha256':input_sha,
            'input_path':str(path),'scale_m':.12,'scale_basis':'estimated_pattern_period_not_measured_object_size',
            'status':'review','cache_hit':False,'maps':{},'sources':[], 'stats':stats,
            'height_encoding':'signed_midpoint_0.5','height_amplitude_max_m':.00008,
            'physical_status':'estimated_not_measured_PBR',
            'risks':['Generated/reference image shading is not measured reflectance.',
                     'Text/highlight rejection is heuristic; ambiguous surfaces require visual review.',
                     'Texture scale and microrelief are estimates; preserve source material and labels.']}
    if len(selected)<3 or len(stats['panels_used'])<2:
        record['reason']='insufficient_clean_surface_patches_across_views'
        atomic_json(record_path,record);return record
    for patch in selected:
        record['sources'].append({k:v for k,v in patch.items() if k!='pixels'}|
            {'source_sha256':input_sha,'reason':'eroded continuous interior; no detected text, bright label, black seam or broad step edge'})
    texture,placements=quilt(selected,seed=seed,paint=stats['material_selection']=='colored_paint')
    arrays={'albedo':np.uint8(np.clip(srgb_encode(texture[...,:3])*255,0,255))}
    arrays.update({name:np.uint8(np.rint(np.clip(texture[...,i+3],0,1)*255)) for i,name in enumerate(MAPS[1:])})
    for name,pixels in arrays.items():
        dest=output/f'{name}.png';Image.fromarray(pixels).save(dest)
        record['maps'][name]={'path':str(dest),'sha256':file_hash(dest),
                             'color_space':'sRGB' if name=='albedo' else 'Non-Color'}
    record['stats']['seams']=seam_diagnostics(texture[...,:3],stats['patch_size_native_px']*3//4)
    record['stats']['height_rms_about_midpoint']=float(np.sqrt(np.mean((texture[...,6]-.5)**2)))
    record['stats']['roughness_p05_p50_p95']=[float(v) for v in np.percentile(texture[...,3],[5,50,95])]
    base=texture[...,:3][texture[...,4]<.15]
    if len(base)<128:base=texture[...,:3].reshape(-1,3)
    lo,hi=np.percentile(base,[10,90],axis=0)
    core=base[np.all((base>=lo)&(base<=hi),axis=1)]
    record['albedo_mean_linear']=[float(v) for v in (core.mean(0) if len(core) else base.mean(0))]
    neutral=texture[...,:3][texture[...,7]>.08]
    record['dust_color_linear']=[float(v) for v in (np.median(neutral,axis=0) if len(neutral)>128 else np.array([.18,.17,.15]))]
    record['dust_color_basis']='neutral_surface_samples' if len(neutral)>128 else 'conservative_estimate_no_neutral_surface'
    record['quilting']={'placements':placements,'native_pixel_scale':True,'output_resolution':[512,512]}
    record['reason']='estimated_surface_maps_require_asset_preview_review'
    thumb=160;board=Image.new('RGB',(thumb*4,thumb*((len(selected)+3)//4)),(242,242,242));draw=ImageDraw.Draw(board)
    for index,p in enumerate(selected):
        image=Image.fromarray(p['pixels']);image.thumbnail((thumb-8,thumb-25))
        x=(index%4)*thumb;y=(index//4)*thumb
        board.paste(image,(x+4,y+4));draw.text((x+4,y+thumb-18),f"view {p['panel']} ROI {p['roi'][0]},{p['roi'][1]}",fill=(30,30,30))
    board.save(output/'sample-diagnostics.png')
    record['sample_diagnostics']=str(output/'sample-diagnostics.png')
    atomic_json(record_path,record)
    return record


def _flat_implementation_hash():
    """Cache actual extraction/serialization code, not just a manually bumped label."""
    return signature(file_hash(Path(__file__)),file_hash(Path(__file__).with_name('common.py')))


def prepare_flat_master(path, output, provenance, *, size=512, material_kind='auto'):
    """Derive maps from a complete generated surface master; no panels or quilting.

    A master remains an estimated colour pattern. It cannot supply measured
    normal/depth or guarantee seamlessness, even when its generator requested it.
    """
    path=Path(path).resolve(strict=True);output=Path(output).resolve()
    if size not in (512,1024):raise ValueError('flat master size must be 512 or 1024')
    if material_kind not in ('auto','metal','paint'):raise ValueError('invalid material kind')
    if not isinstance(provenance,dict) or not str(provenance.get('prompt','')).strip():
        raise ValueError('flat master needs the original generation prompt')
    prompt=provenance['prompt']
    if not isinstance(prompt,str) or len(prompt)>20000:
        raise ValueError('generation prompt must be complete text within 20000 characters')
    sources=[]
    for item in provenance.get('reference_images',[]):
        original=Path(item['path']).resolve(strict=True)
        checksum=file_hash(original)
        if item.get('sha256') and checksum!=item['sha256']:
            raise ValueError('original six-view reference checksum mismatch')
        source={'path':str(original),'sha256':checksum}
        if item.get('role'):
            if not isinstance(item['role'],str) or len(item['role'])>256:
                raise ValueError('reference role must be short text')
            source['role']=item['role']
        sources.append(source)
    if not sources:raise ValueError('flat master must name its original six-view references')
    metadata={key:provenance[key] for key in ('generator','model','created_at','generated_at') if key in provenance}
    metadata.update(prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode('utf8')).hexdigest(),reference_images=sources)
    input_sha=file_hash(path)
    implementation_sha=_flat_implementation_hash()
    key=signature('flat-master-maps-v1',implementation_sha,input_sha,metadata,size,material_kind,cv2.__version__,np.__version__)
    output.mkdir(parents=True,exist_ok=True);record_path=output/'reference.json'
    if record_path.exists():
        old=json.loads(record_path.read_text(encoding='utf8'))
        if old.get('signature')!=key:raise ValueError('changed flat master or provenance requires a new directory')
        if all(Path(m['path']).is_file() and file_hash(m['path'])==m['sha256'] for m in old.get('maps',{}).values()):
            return {**old,'cache_hit':True}
        raise ValueError('cached flat-master maps missing or modified')
    with Image.open(path) as opened:
        source=opened.convert('RGB');width,height=source.size
        rgb=np.array(source)
    record={'version':VERSION,'algorithm':'flat-master-maps-v1','signature':key,'input_sha256':input_sha,
        'implementation_sha256':implementation_sha,
        'input_path':str(path),'status':'review','cache_hit':False,'maps':{},'scale_m':.12,
        'scale_basis':'estimated_pattern_period_not_measured_object_size',
        'physical_status':'estimated_not_measured_PBR','height_encoding':'signed_midpoint_0.5',
        'height_amplitude_max_m':.00008,'generation':metadata,
        'sources':[{'type':'generated_flat_master','path':str(path),'sha256':input_sha,
                    'roi':[0,0,width,height],'original_references':sources}],
        'stats':{'input_resolution':[width,height],'no_panel_segmentation':True,'no_quilting':True},
        'risks':['Generated colour is not measured reflectance, roughness or displacement.',
                 'Tiny baked highlights, repeated motifs and generator artefacts still require preview review.',
                 'Microheight derives only from bounded band-pass detail; no physical depth is asserted.']}
    if max(width,height)<size or min(width,height)<128:
        record['reason']='flat_master_resolution_insufficient_no_upscaling'
        atomic_json(record_path,record);return record
    # Preserve the entire surface and aspect ratio; never crop or stretch a master.
    factor=size/max(width,height)
    target=(round(width*factor),round(height*factor))
    small=np.array(Image.fromarray(rgb).resize(target,Image.Resampling.LANCZOS))
    hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
    paint=(material_kind=='paint' or (material_kind=='auto' and
        bool(np.mean(((hsv[...,0]<10)|(hsv[...,0]>80))&(hsv[...,1]>85))>.25)))
    maps=estimated_maps(small,paint=paint,protect_paint=True)
    # Same bounded periodic edge transition for every channel. This is not quilting.
    edge_width=12
    before=float(max(np.abs(maps[0,:,:3]-maps[-1,:,:3]).max(),
                     np.abs(maps[:,0,:3]-maps[:,-1,:3]).max()))
    for i in range(edge_width):
        alpha=.5*(1-i/edge_width)**2
        a=maps[:,i].copy();b=maps[:,-1-i].copy()
        maps[:,i]=a*(1-alpha)+b*alpha;maps[:,-1-i]=b*(1-alpha)+a*alpha
        a=maps[i].copy();b=maps[-1-i].copy()
        maps[i]=a*(1-alpha)+b*alpha;maps[-1-i]=b*(1-alpha)+a*alpha
    arrays={'albedo':np.uint8(np.clip(srgb_encode(maps[...,:3])*255,0,255))}
    arrays.update({name:np.uint8(np.rint(np.clip(maps[...,i+3],0,1)*255)) for i,name in enumerate(MAPS[1:])})
    for name,pixels in arrays.items():
        dest=output/f'{name}.png';Image.fromarray(pixels).save(dest)
        record['maps'][name]={'path':str(dest),'sha256':file_hash(dest),
                             'color_space':'sRGB' if name=='albedo' else 'Non-Color'}
    base=maps[...,:3][maps[...,4]<.15]
    if len(base)<128:base=maps[...,:3].reshape(-1,3)
    lo,hi=np.percentile(base,[10,90],axis=0);core=base[np.all((base>=lo)&(base<=hi),axis=1)]
    record['albedo_mean_linear']=[float(v) for v in (core.mean(0) if len(core) else base.mean(0))]
    neutral=maps[...,:3][maps[...,7]>.08]
    record['dust_color_linear']=[float(v) for v in (np.median(neutral,axis=0) if len(neutral)>128 else [.18,.17,.15])]
    record['dust_color_basis']='neutral_surface_samples' if len(neutral)>128 else 'conservative_estimate'
    record['stats'].update(output_resolution=list(target),source_pixels_per_output_pixel=1/factor,
        material_kind_requested=material_kind,material_kind_applied='paint' if paint else 'metal',
        oxide_coverage_above_half=float(np.mean(maps[...,4]>.5)),
        seam_before_max_linear=before,seam_after_max_linear=float(max(np.abs(maps[0,:,:3]-maps[-1,:,:3]).max(),
            np.abs(maps[:,0,:3]-maps[:,-1,:3]).max())),edge_transition_px=edge_width,
        height_rms_about_midpoint=float(np.sqrt(np.mean((maps[...,6]-.5)**2))),
        roughness_p05_p50_p95=[float(v) for v in np.percentile(maps[...,3],[5,50,95])])
    record['reason']='estimated_flat_master_requires_material_preview_review'
    atomic_json(record_path,record);return record
