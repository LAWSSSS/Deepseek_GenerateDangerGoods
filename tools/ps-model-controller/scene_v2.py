"""CPU perception and geometry for the Photoshop batch runner."""
import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from relief import RELIEF_PRIORITY, ReliefField, plane_fit

VERSION = "v2.1"
RESAMPLE = Image.Resampling.LANCZOS
MIN_COVER, MAX_COVER = .40, .85
WEAR_SAMPLE = Path(__file__).resolve().parents[2] / "external/textures/wear-sample.jpg"


class NeedsReview(Exception):
    """The local evidence is insufficient to produce a safe candidate."""


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def wear_sample_path():
    if not WEAR_SAMPLE.is_file():
        raise FileNotFoundError(f"Provide a local wear texture JPEG at {WEAR_SAMPLE}")
    return WEAR_SAMPLE


def key(*parts):
    return hashlib.sha256(json.dumps([VERSION, *parts], sort_keys=True).encode()).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_image(path, mode="RGB"):
    with Image.open(path) as original:
        return ImageOps.exif_transpose(original).convert(mode)


def contact_sheet(paths, labels, out, audit=False):
    tile_w, tile_h = (448, 390) if audit else (320, 280)
    columns = min(2,len(paths)) if audit else len(paths)
    board = Image.new("RGB", (tile_w*columns, tile_h*math.ceil(len(paths)/columns)), (32, 32, 32))
    draw = ImageDraw.Draw(board)
    for index, (path, label) in enumerate(zip(paths, labels)):
        view = read_image(path)
        view.thumbnail((tile_w-12, tile_h-32), RESAMPLE)
        x,y = tile_w*(index%columns),tile_h*(index//columns)
        board.paste(view, (x+(tile_w-view.width)//2, y+24+(tile_h-24-view.height)//2))
        draw.text((x+12, y+8), label, fill="white")
    board.save(out, quality=82, optimize=True)


class Perception:
    def __init__(self, weights, cache):
        self.weights, self.cache = weights, cache
        self._matte = self._depth = self._sam = None
        for path in (weights/"mobile_sam.pt", weights/"depth/model.safetensors",
                     weights/"matte/birefnet-general-lite.onnx"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing local weights: {path}")
        self.versions = [digest(weights/"mobile_sam.pt"),
                         digest(weights/"depth/model.safetensors"),
                         digest(weights/"matte/birefnet-general-lite.onnx")]
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        cv2.setNumThreads(2)

    def matte(self, source, *, allow_partial=False):
        from PIL import ImageFilter

        rgba = read_image(source, "RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        transparent = bool((alpha < 10).mean() > .02)
        if not transparent:
            if self._matte is None:
                os.environ["U2NET_HOME"] = str(self.weights/"matte")
                from rembg import new_session
                self._matte = new_session("birefnet-general-lite", providers=["CPUExecutionProvider"])
            from rembg import remove
            mask = remove(rgba.convert("RGB"), session=self._matte, only_mask=True)
            alpha = np.asarray(mask)
            # Remove isolated dust, keeping holes and thin connected parts.
            count, components, stats, _ = cv2.connectedComponentsWithStats((alpha > 45).astype("uint8"))
            if count > 1:
                largest = int(stats[1:, cv2.CC_STAT_AREA].max())
                valid = np.zeros(count, dtype=bool)
                valid[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(8, largest * .012)
                keep = cv2.dilate(valid[components].astype("uint8"), np.ones((5,5), np.uint8))
                alpha = np.where(keep, alpha, 0).astype("uint8")
            # Pull the nearest opaque colors into the semitransparent edge.
            opaque = alpha >= 235
            if opaque.any():
                _, labels = cv2.distanceTransformWithLabels((~opaque).astype("uint8"), cv2.DIST_L2, 5,
                                                            labelType=cv2.DIST_LABEL_PIXEL)
                rgb = np.array(rgba.convert("RGB"))
                palette = np.zeros((int(labels.max())+1, 3), np.uint8)
                palette[labels[opaque]] = rgb[opaque]
                edge = (alpha >= 25) & (alpha < 235)
                rgb[edge] = palette[labels[edge]]
                rgba = Image.fromarray(rgb).convert("RGBA")
        alpha = np.where(alpha < 25, 0, alpha).astype("uint8")
        ys, xs = np.nonzero(alpha >= 96)
        coverage = len(xs) / alpha.size
        if not .035 <= coverage <= .92 or not len(xs):
            raise NeedsReview("素材分割面积异常")
        rect = (max(0, int(xs.min())-2), max(0, int(ys.min())-2),
                min(rgba.width, int(xs.max())+3), min(rgba.height, int(ys.max())+3))
        if min(rect[2]-rect[0], rect[3]-rect[1]) < 55:
            raise NeedsReview("素材细节分辨率不足")
        if not allow_partial and not transparent and (rect[0]==0 or rect[1]==0 or rect[2]==rgba.width or rect[3]==rgba.height):
            raise NeedsReview("抠图触及原图边界，轮廓可能截断")
        rgba.putalpha(Image.fromarray(alpha))
        matte = rgba.crop(rect)
        matte.thumbnail((1600, 1600), RESAMPLE)
        # Use the photo's texture and surface position to define an irregular wear mask.
        lum = matte.convert("L")
        smooth = np.asarray(lum.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32)
        grain = np.abs(np.asarray(lum, dtype=np.float32) - smooth)
        texture = read_image(wear_sample_path())
        texture = ImageOps.fit(texture, matte.size, RESAMPLE)
        alpha = np.asarray(matte.getchannel("A"))
        interior = cv2.distanceTransform((alpha>=160).astype("uint8"),cv2.DIST_L2,5)
        rng=np.random.default_rng(int(digest(source)[:8],16))
        broad=cv2.resize(rng.uniform(0,1,(10,8)).astype("float32"),matte.size,interpolation=cv2.INTER_CUBIC)
        wear = np.clip(.24+broad*.81+np.minimum(grain*.025,.18),0,.94)
        wear *= np.minimum(interior/5,1)*(alpha/255)
        texture.putalpha(Image.fromarray(np.round(wear*255).astype("uint8")))
        return matte, texture, transparent

    def sam(self):
        if self._sam is None:
            import torch
            from mobile_sam import sam_model_registry
            torch.set_num_threads(4)
            self._sam = sam_model_registry["vit_t"](checkpoint=str(self.weights/"mobile_sam.pt"))
            self._sam.eval()
        return self._sam

    def scene(self, source):
        cache_file = self.cache / f"scene-{key(digest(source), self.versions)}.npz"
        frame = read_image(source)
        original_size = frame.size
        frame.thumbnail((960, 960), RESAMPLE)
        rgb = np.asarray(frame)
        if cache_file.is_file():
            with np.load(cache_file) as data:
                depth = data["depth"]
                masks = np.unpackbits(data["masks"], axis=1, count=rgb.shape[0]*rgb.shape[1])
            return rgb, depth, masks.reshape((-1,*depth.shape)).astype(bool), original_size
        import torch
        from mobile_sam import SamAutomaticMaskGenerator
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if self._depth is None:
            root = self.weights/"depth"
            self._depth = (AutoImageProcessor.from_pretrained(root, local_files_only=True),
                           AutoModelForDepthEstimation.from_pretrained(root, local_files_only=True).eval())
        processor, model = self._depth
        with torch.inference_mode():
            d = model(**processor(images=frame, return_tensors="pt")).predicted_depth
            d = torch.nn.functional.interpolate(d.unsqueeze(1), size=rgb.shape[:2],
                                                 mode="bicubic", align_corners=False)[0,0].numpy()
            low, high = np.percentile(d, [5,95])
            depth = np.clip((d-low)/max(float(high-low),1e-6),0,1).astype("float32")
            generator = SamAutomaticMaskGenerator(self.sam(), points_per_side=12, points_per_batch=16,
                                                  pred_iou_thresh=.78, stability_score_thresh=.86)
            records = generator.generate(np.ascontiguousarray(rgb))
        masks = [r["segmentation"] for r in records
                 if .0006 * depth.size <= r["area"] <= .48 * depth.size]
        masks = np.asarray(masks, dtype=bool).reshape((-1, *depth.shape))
        temp = cache_file.with_suffix(".tmp.npz")
        np.savez_compressed(temp, depth=depth, masks=np.packbits(masks.reshape(len(masks),depth.size),axis=1))
        temp.replace(cache_file)
        return rgb, depth, masks, original_size

    def seeded_masks(self, crop, points):
        """Use original-pixel crops so narrow bars are not lost in the scene thumbnail."""
        from mobile_sam import SamPredictor
        import torch

        frame = Image.fromarray(crop)
        frame.thumbnail((960, 960), RESAMPLE)
        original_size = crop.shape[1], crop.shape[0]
        coords = np.asarray(points, dtype=np.float32) * np.asarray(frame.size) / original_size
        predictor = SamPredictor(self.sam())
        results = []
        with torch.inference_mode():
            predictor.set_image(np.asarray(frame))
            for index, (x, y) in enumerate(points):
                selected = list(range(len(points))) if index==0 else [0,index]
                labels = np.array([int(j==index) for j in selected])
                masks, scores, _ = predictor.predict(
                    point_coords=coords[selected], point_labels=labels, multimask_output=True)
                valid = []
                for choice in np.argsort(-scores):
                    if scores[choice] < .60:
                        continue
                    mask = cv2.resize(masks[choice].astype("uint8"), original_size,
                                      interpolation=cv2.INTER_NEAREST)
                    _, components = cv2.connectedComponents(mask)
                    component = components[y, x]
                    if not component:
                        continue
                    mask = components == component
                    if mask.sum() < 12 or any(mask[points[j][1],points[j][0]]
                                              for j in selected if j!=index):
                        continue
                    if any(np.count_nonzero(mask & other) > .90*np.count_nonzero(mask | other)
                           for other in valid):
                        continue
                    if index and frame.size!=original_size:
                        distance = cv2.distanceTransform(mask.astype("uint8"),cv2.DIST_L2,5)
                        if float(np.median(distance[mask])) < 5:
                            try:
                                mask = self.refine_bar(crop,mask,(x,y))
                            except NeedsReview:
                                continue
                            if mask[points[0][1],points[0][0]]:
                                continue
                    valid.append(mask)
                    if len(valid) == 2:
                        break
                results.append(valid)
        return results

    def refine_bar(self,crop,mask,point):
        ys,xs = np.nonzero(mask)
        left,top = max(0,int(xs.min())-12),max(0,int(ys.min())-12)
        right,bottom = min(mask.shape[1],int(xs.max())+13),min(mask.shape[0],int(ys.max())+13)
        rect = np.s_[top:bottom,left:right]
        refined = self.refine_crop(Image.fromarray(crop[rect]),mask[rect])
        x,y = point
        if not refined[y-top,x-left]:
            raise NeedsReview("细钢筋细化偏离前景取样点")
        result = np.zeros_like(mask)
        result[rect] = refined
        return result

    def refine(self, source, roi, seed_mask):
        """Refine only a foreground crossing in the native crop, with positive/negative seeds."""
        return self.refine_crop(read_image(source).crop(roi),seed_mask)

    def refine_crop(self, full, seed_mask):
        from mobile_sam import SamPredictor
        import torch
        size = full.size
        full.thumbnail((960,960), RESAMPLE)
        seed = cv2.resize(seed_mask.astype("uint8"),full.size,interpolation=cv2.INTER_NEAREST)
        dist = cv2.distanceTransform(seed,cv2.DIST_L2,5)
        if dist.max() < 1.5:
            raise NeedsReview("细钢筋遮挡边缘无法确认")
        # Broad plate boundaries use seeded segmentation. Thin bars need the crop SAM pass.
        if float(np.median(dist[seed>0])) >= 7:
            kernel=np.ones((7,7),np.uint8)
            inside=cv2.erode(seed,kernel)
            outside=cv2.dilate(seed,kernel)
            trimap=np.where(outside>0,cv2.GC_PR_BGD,cv2.GC_BGD).astype("uint8")
            trimap[seed>0]=cv2.GC_PR_FGD
            trimap[inside>0]=cv2.GC_FGD
            if not inside.any() or not (trimap==cv2.GC_BGD).any():
                raise NeedsReview("遮挡细化缺少前景或背景种子")
            model=np.zeros((1,65),np.float64)
            cv2.grabCut(np.asarray(full),trimap,None,model,model.copy(),2,cv2.GC_INIT_WITH_MASK)
            result=np.isin(trimap,[cv2.GC_FGD,cv2.GC_PR_FGD])
            if np.count_nonzero(result&(seed>0))/max(1,np.count_nonzero(result|(seed>0))) < .65:
                raise NeedsReview("原分辨率钢板边缘不稳定")
            return cv2.resize(result.astype("uint8"),size,interpolation=cv2.INTER_NEAREST)>0
        y,x = np.unravel_index(np.argmax(dist),dist.shape)
        outside = cv2.dilate(seed,np.ones((15,15),np.uint8))-seed
        neg = np.argwhere(outside)
        if not len(neg):
            raise NeedsReview("遮挡边缘缺少背景参照")
        yy,xx = neg[np.argmin(((neg-[y,x])**2).sum(axis=1))]
        predictor = SamPredictor(self.sam())
        with torch.inference_mode():
            predictor.set_image(np.asarray(full))
            options, _, _ = predictor.predict(point_coords=np.array([[x,y],[xx,yy]]),
                                                point_labels=np.array([1,0]),multimask_output=True)
        overlap = [np.count_nonzero(o & (seed>0))/max(1,np.count_nonzero(o | (seed>0))) for o in options]
        best = int(np.argmax(overlap))
        if overlap[best] < .48:
            raise NeedsReview("原分辨率遮挡分割与候选不一致")
        return cv2.resize(options[best].astype("uint8"),size,interpolation=cv2.INTER_NEAREST)>0


@dataclass
class Candidate:
    asset: Path
    center: tuple
    width: int
    angle: float
    support: float
    rank: float
    occlusion: float
    occluder: np.ndarray
    shade: np.ndarray
    tone: dict


def axis(points):
    _, vectors = np.linalg.eigh(np.cov(points.T))
    x,y = vectors[:,-1]
    return math.degrees(math.atan2(y,x))


def angle_delta(a,b):
    return (a-b+90)%180-90


def render_mask(alpha, width, angle):
    raw = Image.fromarray(alpha).rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
    box = raw.point(lambda p: 255 if p>=32 else 0).getbbox()
    if not box:
        raise NeedsReview("旋转后物品蒙版为空")
    raw = raw.crop(box)
    return np.asarray(raw.resize((width,max(2,round(raw.height*width/raw.width))),RESAMPLE)) >= 100


def local_tone(rgb, support, object_rgb, object_alpha):
    lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    values = lab[support]
    obj = cv2.cvtColor(object_rgb,cv2.COLOR_RGB2LAB)[object_alpha>=180]
    if len(values)<30 or not len(obj):
        return dict(brightness=0,warmth=0,saturation=0,blur_px=0)
    diff = np.median(values,axis=0)-np.median(obj,axis=0)
    scene_sharp = float(np.median(np.abs(cv2.Laplacian(lab[:,:,0],cv2.CV_32F)[support])))
    obj_luma = cv2.cvtColor(object_rgb,cv2.COLOR_RGB2GRAY)
    obj_interior = cv2.erode((object_alpha>=180).astype("uint8"),np.ones((5,5),np.uint8))>0
    sharp = float(np.median(np.abs(cv2.Laplacian(obj_luma,cv2.CV_32F)[obj_interior]))) if obj_interior.any() else 0
    blur = .6 if sharp>scene_sharp*1.45+2 else 0
    return dict(brightness=float(np.clip(diff[0]*.5,-24,17)),
                warmth=float(np.clip((diff[1]+diff[2])*.25,-9,9)),
                saturation=int(np.clip((np.linalg.norm(np.median(values[:,1:],axis=0)-128)
                                        -np.linalg.norm(np.median(obj[:,1:],axis=0)-128))*.55,-22,0)),
                blur_px=blur)


@dataclass
class GapCandidate:
    asset: Path
    center: tuple[int, int]
    width: int
    angle: int
    crop: tuple[int, int, int, int]
    support: np.ndarray
    foreground: np.ndarray
    shadow: np.ndarray
    foreground_shadow: np.ndarray
    keep: tuple[float, float]
    tone: dict
    photography: dict | None = None
    grain: np.ndarray | None = None
    edge_neighborhood: np.ndarray | None = None
    relief: dict | None = None


def frame_point(point, size):
    return tuple(round(unit*(length-1)/1000) for unit,length in zip(point,size,strict=True))


def end_points(asset):
    alpha = np.asarray(read_image(asset,"RGBA").getchannel("A"))
    solid = alpha>=128
    ys,xs = np.nonzero(solid)
    if len(xs)<3:
        raise NeedsReview("物品素材没有足够的不透明识别区域")
    points = np.c_[xs,ys]
    theta = math.radians(axis(points[::max(1,len(points)//3000)]))
    direction = np.array([math.cos(theta),math.sin(theta)])
    if direction[1]<0:
        direction *= -1
    length = points @ direction
    low,high = np.quantile(length,[.10,.90])
    distance = cv2.distanceTransform(solid.astype("uint8"),cv2.DIST_L2,5)[ys,xs]
    return [[round(int(x)*1000/(alpha.shape[1]-1)),round(int(y)*1000/(alpha.shape[0]-1))]
            for segment in (length<=low,length>=high)
            for x,y in [points[int(np.argmax(np.where(segment,distance,-1)))]]]


def seam_hints(depth,masks,width_range,eligible=None,limit=3,*,local=False):
    """Mark depth-separated adjacent edges to help the planner read a crowded thumbnail."""
    h,w = depth.shape
    radius = max(4,min(28,round(w*sum(width_range)*.23)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)
    pieces = []
    for mask in sorted(masks,key=np.count_nonzero,reverse=True):
        area = int(mask.sum())
        if .002*depth.size <= area <= .23*depth.size:
            solid = mask.astype("uint8")
            distance = cv2.distanceTransform(solid,cv2.DIST_L2,5)
            pieces.append((mask,area,float(np.median(depth[mask])),distance))
        if len(pieces)==22:
            break
    candidates = []
    for foreground,area,near,front_distance in pieces:
        around = cv2.dilate(foreground.astype("uint8"),kernel)>0
        for support,rear_area,far,rear_distance in pieces:
            if near-far < .04 or np.count_nonzero(foreground & support)>.10*min(area,rear_area):
                continue
            seam = around & support & ~foreground
            if eligible is not None:
                seam &= eligible
            seam[:round(h*.14)] = False
            seam[round(h*.77):] = False
            seam[:,:round(w*.09)] = False
            seam[:,round(w*.91):] = False
            if np.count_nonzero(seam)<radius*2:
                continue
            _,clusters,stats,_ = cv2.connectedComponentsWithStats(seam.astype("uint8"))
            k = int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))+1
            ys,xs = np.nonzero(clusters==k)
            x0,y0 = round(float(np.median(xs))),round(float(np.median(ys)))
            close = (xs-x0)**2+(ys-y0)**2
            seat = int(np.argmax(np.minimum(rear_distance[ys,xs],radius*.35)*radius-close*.16))
            sx,sy = int(xs[seat]),int(ys[seat])
            ry,rx = np.s_[max(0,sy-2*radius):sy+2*radius+1,max(0,sx-2*radius):sx+2*radius+1]
            yy,xx = np.nonzero(foreground[ry,rx])
            if not len(xx):
                continue
            xx,yy = xx+rx.start, yy+ry.start
            close = (xx-sx)**2+(yy-sy)**2
            front = int(np.argmax(np.minimum(front_distance[yy,xx],radius*.30)*radius-close*.29))
            fx,fy = int(xx[front]),int(yy[front])
            delta=near-far
            if local:
                reach=max(3,radius//2)
                delta=local_depth(depth,foreground,(fx,fy),reach)-local_depth(depth,support,(sx,sy),reach)
                if delta<.03:
                    continue
            score = math.log1p(len(xs)) + min(.28,delta)*4 + np.clip((sy-fy)/radius,-1,1)*.65
            candidates.append((score,(round((sx+fx)/2),round((sy+fy)/2)),(sx,sy),(fx,fy)))
    hints = []
    anchors = []
    for _,center,seat,front in sorted(candidates,reverse=True):
        if any(math.dist(center,p)<w*.13 for p in anchors):
            continue
        anchors.append(center)
        hints.append([[round(x*1000/(w-1)),round(y*1000/(h-1))] for x,y in (center,seat,front)])
        if len(hints)==limit:
            break
    return hints


def posed_alpha(alpha, width, angle, feature, pixel_scale=None):
    image = Image.fromarray(alpha)
    rotated = image.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
    box = rotated.getbbox()
    left, top, right, bottom = box
    theta = math.radians(angle)
    x, y = feature[0]+.5-image.width/2, feature[1]+.5-image.height/2
    px = x*math.cos(theta)-y*math.sin(theta)+rotated.width/2-left
    py = x*math.sin(theta)+y*math.cos(theta)+rotated.height/2-top
    keep = px/(right-left), py/(bottom-top)
    if pixel_scale is not None:
        width = max(2,round((right-left)*pixel_scale))
    height = max(2, round(width*(bottom-top)/(right-left)))
    return np.asarray(rotated.crop(box).resize((width,height),RESAMPLE)), keep


def contact_band(solid, width):
    bottom = solid.shape[0]-1-np.argmax(solid[::-1],axis=0)
    return solid & (np.arange(solid.shape[0])[:,None] >= bottom-max(1,round(width*.025)))


def feature_visibility(alpha, foreground, point, width):
    x, y = point
    radius = max(3,round(width*.055))
    left, top = max(0,math.floor(x-radius)), max(0,math.floor(y-radius))
    right = min(alpha.shape[1],math.ceil(x+radius)+1)
    bottom = min(alpha.shape[0],math.ceil(y+radius)+1)
    if right <= left or bottom <= top:
        return 0.
    yy, xx = np.ogrid[top:bottom,left:right]
    disk = (xx-x)**2+(yy-y)**2 <= radius**2
    opacity = alpha[top:bottom,left:right][disk].astype(np.float32)/255
    visible = 1-foreground[top:bottom,left:right][disk].astype(np.float32)/255
    mass = opacity.sum()
    return min(1.,float(opacity @ visible / mass)) if mass >= 8 else 0.


def crossing(alpha, foreground):
    solid = (alpha>=128).astype("uint8")
    contours, _ = cv2.findContours(solid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    shell = cv2.drawContours(np.zeros_like(solid),contours,-1,1,cv2.FILLED)
    ring = (cv2.dilate(shell,np.ones((3,3),np.uint8))>0) & (shell==0)
    _, components = cv2.connectedComponents((foreground>=128).astype("uint8"))
    linked = np.unique(components[ring])
    linked = linked[linked!=0]
    overlap = (solid>0) & (components!=0)
    return bool(overlap.any() and np.count_nonzero(overlap & np.isin(components,linked)) >= .90*overlap.sum())


def original_edge_neighborhood(rgb):
    filtered=cv2.GaussianBlur(rgb,(3,3),.65)
    edges=cv2.Canny(filtered,26,58,L2gradient=True)
    return cv2.dilate(edges,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0


def boundary_agreement(alpha,foreground,neighborhood):
    boundary=cv2.morphologyEx((foreground>=128).astype("uint8"),cv2.MORPH_GRADIENT,
                              np.ones((3,3),np.uint8))>0
    crossing_pixels=boundary & (alpha>=128)
    count=np.count_nonzero(crossing_pixels)
    return np.count_nonzero(crossing_pixels & neighborhood)/count if count>=6 else 0.


MIN_EDGE_AGREEMENT = .62


def measure_geometry(alpha, foreground, support, point, width):
    opacity = alpha.astype(np.float32)/255
    cover = float(np.sum(opacity*(foreground.astype(np.float32)/255))/max(1,opacity.sum()))
    if not MIN_COVER <= cover <= MAX_COVER:
        raise NeedsReview(f"物品实际遮挡率 {cover:.1%}，要求 40%–85%")
    if not crossing(alpha,foreground):
        raise NeedsReview("前景蒙版没有从物品外侧穿过真实轮廓")
    keep = feature_visibility(alpha,foreground,point,width)
    if keep < .60:
        raise NeedsReview("指定识别特征被挡住或露出不足")
    band = contact_band(alpha>=128,width)
    tolerance = max(1,round(width*.012))
    near = cv2.dilate(support.astype("uint8"),np.ones((2*tolerance+1,)*2,np.uint8))>0
    contact = float(np.count_nonzero(band & near)/max(1,band.sum()))
    if contact < .18:
        raise NeedsReview("物品下缘缺少可验证的承托")
    return {"occlusion_ratio":cover,"support":contact,"feature_visible":keep}


def local_depth(depth, mask, point, radius):
    x, y = point
    ys = slice(max(0,y-radius),y+radius+1)
    xs = slice(max(0,x-radius),x+radius+1)
    return float(np.median(depth[ys,xs][mask[ys,xs]]))


def gap_planes(depth, options, points, width, scale=None):
    radius = max(4,round(width*.20))
    for support in options[0]:
        rear = local_depth(depth,support,points[0],radius)
        if scale is not None and abs(rear-scale.depth)>scale.depth_tolerance:
            continue
        layers = []
        for point, choices in zip(points[1:],options[1:],strict=True):
            layers.append([mask for mask in choices
                           if local_depth(depth,mask,point,radius)>=rear+.025
                           and np.count_nonzero(mask & support)<.22*min(mask.sum(),support.sum())])
        for choices in product(*(layers_i+[None] for layers_i in layers)):
            pieces = [mask for mask in choices if mask is not None]
            if pieces:
                front = np.logical_or.reduce(pieces)
                yield support & ~front, front


def gap_candidates(engine, frame, depth, site, asset, width_range, scale=None, *, photo=False, protected=()):
    size = frame.size
    asset_image = read_image(asset,"RGBA")
    rgba = np.asarray(asset_image)
    feature = frame_point(site.keep,(rgba.shape[1],rgba.shape[0]))
    fx, fy = feature
    alpha = rgba[:,:,3]
    if alpha[fy,fx] < 128:
        radius = max(2,round(min(alpha.shape)*.10))
        l,t = max(0,fx-radius),max(0,fy-radius)
        yy,xx = np.nonzero(alpha[t:fy+radius+1,l:fx+radius+1]>=128)
        if not len(xx):
            raise NeedsReview("识别特征坐标未落在物品材质上")
        i = int(np.argmin((xx+l-fx)**2+(yy+t-fy)**2))
        feature = int(xx[i]+l),int(yy[i]+t)

    cx,cy = frame_point(site.pose[:2],size)
    anchor_points = [frame_point(p,size) for p in (site.seat,*site.front)]
    if len(set(anchor_points)) != len(anchor_points):
        raise NeedsReview("承托和前景取样点重合")
    rotation = site.pose[-1]
    upper,lower = math.ceil(.09*size[1]),math.floor(.82*size[1])
    if scale is None:
        width = round(size[0]*site.pose[2]/1000)
        low,high = (round(size[0]*bound) for bound in width_range)
        if not low-2 <= width <= high+2 or width < 12:
            raise NeedsReview("模型建议尺寸超出任务范围")
        width = min(high,max(low,width))
        widths = list(dict.fromkeys(min(high,max(low,round(width*s))) for s in (1,.78,1.20)))
        sx,sy = anchor_points[0]
        fx,fy = anchor_points[1]
        ys,xs = np.nonzero(alpha>=128)
        stride = max(1,len(xs)//3000)
        source_axis = axis(np.c_[xs[::stride],ys[::stride]])
        seam_axis = math.degrees(math.atan2(sy-fy,sx-fx))+90
        aligned = round(angle_delta(seam_axis,source_axis+rotation))
        angles = list(dict.fromkeys((rotation+delta+180)%360-180
                                    for delta in (0,-12,12,-24,24,aligned)))
        poses = [(w,a,*posed_alpha(alpha,w,a,feature)) for w,a in product(widths,angles)]
    else:
        major = scale.record["projected_major_px"]
        if not 12 <= major <= math.hypot(size[0]-8,lower-upper-8):
            raise NeedsReview("按物理尺度投影后，物品无法在安全画面范围内辨识或容纳")
        adjustment = angle_delta(scale.reference_angle,scale.asset_angle+rotation)
        if abs(adjustment)>55:
            raise NeedsReview("建议姿态与尺度参照的投影方向不相容")
        aligned = rotation+round(adjustment)
        angles = list(dict.fromkeys((a+180)%360-180 for a in (rotation,aligned,aligned-12,aligned+12)
                                    if abs(angle_delta(scale.reference_angle,scale.asset_angle+a))<=16))
        poses = []
        for angle in angles:
            silhouette,keep = posed_alpha(alpha,None,angle,feature,pixel_scale=scale.factor)
            h,w = silhouette.shape
            if w<=size[0]-8 and h<=lower-upper-8:
                poses.append((w,angle,silhouette,keep))
        if not poses:
            raise NeedsReview("按物理尺度投影后，该姿态在画面内放不下")
        widths = [p[0] for p in poses]
        width = round(max(max(widths),max(p[2].shape[0] for p in poses)*.72))
    if any(math.dist((cx,cy),p)>max(width*1.55,size[0]*.035 if scale else 0) for p in anchor_points):
        raise NeedsReview("承托或前景取样点离落点过远")
    pad_x = round(max(widths)*.86+width*.58)
    pad_y = round(max(p[2].shape[0] for p in poses)*.5+width*.80)
    pad = max(12,round(width*.22))
    left = max(0,min(cx-pad_x,min(p[0] for p in anchor_points)-pad))
    top = max(0,min(cy-pad_y,min(p[1] for p in anchor_points)-pad))
    right = min(size[0],max(cx+pad_x+1,max(p[0] for p in anchor_points)+pad+1))
    bottom = min(size[1],max(cy+pad_y+1,max(p[1] for p in anchor_points)+pad+1))
    rect = left,top,right,bottom
    frame_crop = frame.crop(rect)
    crop = np.asarray(frame_crop)
    edge_neighborhood=original_edge_neighborhood(crop) if photo else None
    points = [(x-left,y-top) for x,y in anchor_points]
    options = engine.seeded_masks(crop,points)
    crop_depth = depth[top:bottom,left:right]
    planes = list(gap_planes(crop_depth,options,points,width,scale))
    if not planes:
        raise NeedsReview("局部分割未验证前景钢件在承托件之前")

    offsets = [round(width*v) for v in (0,-.20,.20,-.40,.40,-.58,.58)]
    coarse_centers = {(cx+dx,cy+dy) for dx,dy in product(offsets,repeat=2)}
    covered = grounded = featured = edge_rejected = raised_rejected = False
    matches = []
    step=max(1,math.ceil(max(crop_depth.shape)/960))
    depth_sample=crop_depth[::step,::step]
    ys,xs=np.ogrid[top:bottom:step,left:right:step]
    safe=np.broadcast_to((ys>=upper)&(ys<lower),depth_sample.shape).copy()
    for pl,pt,pr,pb in protected:
        safe&=~((xs>=pl)&(xs<pr)&(ys>=pt)&(ys<pb))
    for support,front in planes:
        terrain_field=None
        if photo:
            floor,error=plane_fit(depth_sample,support[::step,::step]&safe,limit=6000)
            if floor is not None and error<=.034:
                local_y,local_x=np.ogrid[:depth_sample.shape[0],:depth_sample.shape[1]]
                plane=floor[0]+floor[1]*local_x+floor[2]*local_y
                terrain_field=ReliefField(depth_sample,plane,safe,error)
        mask=front.astype("uint8")*255
        if photo:
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        foreground = cv2.GaussianBlur(mask,(3,3),.60)
        for w,angle,silhouette,keep in poses:
            h = silhouette.shape[0]
            mass = silhouette.astype(np.float32)/255
            total = mass.sum()
            band = contact_band(silhouette>=128,w)
            if not band.any():
                continue
            yy,xx = np.nonzero(band)
            sx,sy = anchor_points[0]
            centers = coarse_centers.copy()
            for percentile in (.20,.5,.80):
                x_contact = round(float(np.quantile(xx,percentile)))
                distance = np.abs(xx-x_contact)
                y_contact = int(np.max(yy[distance<=distance.min()+max(2,w*.08)]))
                center = sx+w//2-x_contact, sy+h//2-y_contact
                if math.dist(center,(cx,cy))<=width*.85:
                    centers.add(center)
            tolerance = max(1,round(w*.012))
            near = cv2.dilate(support.astype("uint8"),np.ones((2*tolerance+1,)*2,np.uint8))>0
            for x,y in sorted(centers):
                l,t = x-w//2,y-h//2
                r,b = l+w,t+h
                if l<left+4 or r>right-4 or t<max(top+4,upper+4) or b>min(bottom-4,lower-4):
                    continue
                if any(l<pr and r>pl and t<pb and b>pt for pl,pt,pr,pb in protected):
                    continue
                region = np.s_[t-top:b-top,l-left:r-left]
                overlap = float(np.sum(mass*foreground[region])/(255*total))
                if not MIN_COVER <= overlap <= MAX_COVER:
                    continue
                covered = True
                contact = float(np.count_nonzero(band & near[region])/band.sum())
                if contact < .18:
                    continue
                grounded = True
                if feature_visibility(silhouette,foreground[region],(keep[0]*w,keep[1]*h),w) < .60:
                    continue
                featured = True
                edge_score=0.
                if edge_neighborhood is not None:
                    edge_score=boundary_agreement(silhouette,foreground[region],edge_neighborhood[region])
                    if edge_score<MIN_EDGE_AGREEMENT:
                        edge_rejected=True
                        continue
                visible_contact = np.count_nonzero(band & near[region] & (foreground[region]<128))/band.sum()
                score = (contact*.24+visible_contact*.13-abs(overlap-.62)*.58
                         -math.dist((cx,cy),(x,y))/width*.13-abs(angle_delta(angle,rotation))*.003
                         +edge_score*.18)
                if scale is None:
                    score -= .07*w/width
                terrain=None
                if terrain_field is not None:
                    bounds=((l-left)//step,(t-top)//step,math.ceil((r-left)/step),math.ceil((b-top)/step))
                    terrain=terrain_field.measure(bounds,max(w,h)/step)
                    if terrain["kind"]=="raised":
                        raised_rejected=True
                        continue
                priority=RELIEF_PRIORITY[terrain["kind"]] if terrain else 0
                matches.append(((priority,score),x,y,w,angle,region,keep,silhouette,band,support,foreground,terrain))

    results = []
    for _,x,y,w,angle,region,keep,silhouette,band,support,foreground,terrain in sorted(
            matches,key=lambda match:match[0],reverse=True):
        if any(math.dist((x,y),c.center)<min(w,c.width)*.32 for c in results):
            continue
        placed = np.zeros(crop.shape[:2],dtype=np.uint8)
        placed[region] = silhouette
        feature_position = ((region[1].start+keep[0]*w),(region[0].start+keep[1]*silhouette.shape[0]))
        try:
            measure_geometry(placed,foreground,support,feature_position,w)
        except NeedsReview:
            continue
        contact = np.zeros_like(placed)
        contact[region] = band*255
        if scale is not None:
            contact_depth = crop_depth[(contact>0) & support]
            if not len(contact_depth) or abs(float(np.median(contact_depth))-scale.depth)>scale.depth_tolerance:
                continue
        shadow = cv2.dilate(contact,np.ones((max(3,round(w*.04)),max(3,round(w*.055))),np.uint8))
        shadow = cv2.GaussianBlur(shadow,(0,0),max(1,w*.013))*support
        shadow[:max(0,upper-top)] = 0
        shadow[max(0,lower-top):] = 0
        for pl,pt,pr,pb in protected:
            il,it,ir,ib = max(pl,left),max(pt,top),min(pr,right),min(pb,bottom)
            if il<ir and it<ib:
                shadow[it-top:ib-top,il-left:ir-left] = 0
        if not shadow.any():
            continue
        near_object = cv2.dilate((placed>=128).astype("uint8"),
                                 np.ones((max(3,round(w*.17)),)*2,np.uint8))>0
        visible_support = support & (foreground<128) & (placed<128) & near_object
        photography,grain = None,None
        if photo:
            from photometry import fit_photo

            sample_area = cv2.dilate((placed>=128).astype("uint8"),
                                     np.ones((max(3,round(w*.35)),)*2,np.uint8))>0
            sample_mask = support & (foreground<128) & (placed<128) & sample_area
            if scale is not None:
                sample_mask &= np.abs(crop_depth-scale.depth)<=scale.depth_tolerance
            tone,photography,grain = fit_photo(frame_crop,sample_mask,asset_image,angle,silhouette.shape)
        else:
            tone = local_tone(crop,visible_support,np.ascontiguousarray(rgba[:,:,:3]),alpha)
        # The near steel darkens only a narrow lip on the object. Photoshop clips
        # this shadow to the actual transformed object layer.
        clear = (foreground<128).astype("uint8")
        distance = cv2.distanceTransform(clear,cv2.DIST_L2,5)
        falloff = max(1.5,w*.025)
        lip = np.uint8(np.clip(1-distance/falloff,0,1)**2*255)
        lip *= clear & near_object
        results.append(GapCandidate(asset,(x,y),w,angle,rect,support,foreground,shadow,lip,keep,tone,
                                    photography,grain,edge_neighborhood,terrain))
        if len(results)==2:
            break
    if not results:
        reason = ("没有达到 40%–85% 的前景交叠" if not covered else
                  "遮挡达标时缺少承托" if not grounded else
                  "遮挡和承托达标时特征被遮住" if not featured else
                  "合格轮廓周边显示承托面处于凸位" if raised_rejected else
                  "前景蒙版穿过物品的切边没有贴合原图钢件边缘" if edge_rejected else
                  "无法验证轮廓交叠或接触阴影")
        raise NeedsReview(reason)
    return results


def propose(rgb, depth, masks, size, assets, width_range, limit=9):
    h,w = depth.shape
    sx,sy = size[0]/w,size[1]/h
    supports = sorted(masks, key=np.count_nonzero, reverse=True)
    prepared = []
    for path in assets:
        rgba = np.asarray(read_image(path,"RGBA"))
        ys,xs = np.nonzero(rgba[:,:,3]>=100)
        if len(xs)<100:
            continue
        step=max(1,len(xs)//2500)
        prepared.append((path,rgba,axis(np.c_[xs[::step],ys[::step]]),
                         rgba.shape[0]/rgba.shape[1]>1.30))
    scored = []
    mask_depths = np.array([np.median(depth[m]) for m in masks])
    for support in supports[:26]:
        ys,xs = np.nonzero(support)
        if not .011*h*w <= len(xs) <= .30*h*w:
            continue
        sample = np.c_[xs,ys]
        direction = axis(sample[::max(1,len(xs)//2500)])
        center = np.median(sample,axis=0)
        disp = float(np.median(depth[support]))
        # A sloped support supplies an axis and several anchors, not a claimed metric plane.
        radians=math.radians(direction)
        ray=np.array([math.cos(radians),math.sin(radians)])
        distances=(sample-center)@ray
        offsets=[0,np.quantile(distances,.30),np.quantile(distances,.70)]
        for path,rgba,object_axis,elongated in prepared:
            for width_frac in (width_range[0]*.62+width_range[1]*.38, width_range[1]):
                width = max(12,round(w*width_frac))
                angle=float(np.clip(angle_delta(direction,object_axis),-45,45))
                vertical=(object_axis+angle)%180
                if elongated and 70<vertical<110:
                    angle=float(np.clip(angle+(70-vertical if vertical<90 else 110-vertical),-45,45))
                silhouette=render_mask(rgba[:,:,3],width,angle)
                oh,ow=silhouette.shape
                if oh>h*.48:
                    continue
                relx=np.nonzero(silhouette)[1]
                contact=np.zeros_like(silhouette)
                for col in np.unique(relx):
                    yy=np.nonzero(silhouette[:,col])[0]
                    contact[max(0,yy[-1]-3):yy[-1]+1,col]=silhouette[max(0,yy[-1]-3):yy[-1]+1,col]
                for offset in offsets:
                    cx,cy=center+ray*offset
                    left,top=round(cx-ow/2),round(cy-oh/2)
                    right,bottom=left+ow,top+oh
                    if left<.05*w or right>.96*w or top<.10*h or bottom>.805*h:
                        continue
                    sl=np.s_[top:bottom,left:right]
                    area=np.count_nonzero(silhouette)
                    support_local=support[sl]
                    contact_rate=np.count_nonzero(contact&support_local)/max(1,np.count_nonzero(contact))
                    support_rate=np.count_nonzero(silhouette&support_local)/area
                    if contact_rate<.32 or support_rate<.24:
                        continue
                    forward=np.zeros(silhouette.shape,bool)
                    possible=[j for j in np.flatnonzero(mask_depths>disp+.075)
                              if np.count_nonzero(masks[j][sl]&silhouette)>=.025*area]
                    for j in possible:
                        patch=masks[j][sl]
                        if np.count_nonzero(patch&silhouette)<=.26*area:
                            forward |= patch
                    overlap=np.count_nonzero(forward&silhouette)/area
                    if overlap>.27:
                        continue
                    local=np.zeros((h,w),bool)
                    local[sl]=forward
                    contact_row = np.zeros((h,w),np.uint8)
                    contact_row[sl]=((contact & support_local)*255).astype("uint8")
                    shadow = cv2.dilate(contact_row,np.ones((5,7),np.uint8))
                    shadow = cv2.GaussianBlur(shadow,(0,0),1.5)
                    cast=cv2.warpAffine(shadow,np.float32([[1,0,1],[0,1,4]]),(w,h))
                    shadow=np.maximum(shadow,cv2.GaussianBlur(cast,(0,0),2.1)//2)*support
                    score=contact_rate*.44+support_rate*.25+min(.16,overlap)*1.6
                    region=np.s_[max(0,top-oh//2):min(h,bottom+oh//2),
                                 max(0,left-ow//2):min(w,right+ow//2)]
                    scored.append(Candidate(path,(round((left+ow/2)*sx),round((top+oh/2)*sy)),
                                            round(ow*sx),angle,contact_rate,score,overlap,
                                            local,shadow,(region,support,rgba)))
                    if len(scored)>48:
                        scored=sorted(scored,key=lambda c:c.rank,reverse=True)[:32]
    result=[]
    for candidate in sorted(scored,key=lambda c:c.rank,reverse=True):
        if any(math.dist(candidate.center,other.center)<min(candidate.width,other.width)*.48
               for other in result):
            continue
        region,support,rgba=candidate.tone
        nearby=support[region]&~candidate.occluder[region]
        candidate.tone=local_tone(np.ascontiguousarray(rgb[region]),nearby,
                                  np.ascontiguousarray(rgba[:,:,:3]),rgba[:,:,3])
        result.append(candidate)
        if len(result)>=limit:
            break
    return result


if __name__ == "__main__":
    import argparse
    cli=argparse.ArgumentParser(description="Isolated CPU foreground-matte worker")
    cli.add_argument("weights",type=Path)
    cli.add_argument("cache",type=Path)
    cli.add_argument("source",type=Path)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    try:
        matte,wear,preworn=Perception(args.weights,args.cache).matte(args.source)
        matte.save(args.out/"matte.png")
        wear.save(args.out/"wear.png")
        atomic_json(args.out/"matte.json",{"preworn":preworn})
    except NeedsReview as error:
        atomic_json(args.out/"matte.json",{"reason":str(error)})
