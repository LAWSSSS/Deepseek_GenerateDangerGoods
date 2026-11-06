"""Convert a local scene measurement and a dimensioned cutout into a pixel scale."""
from dataclasses import dataclass
import math
from typing import Annotated, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator

from run import Strict
from scene_v2 import NeedsReview, frame_point


Coordinate = Annotated[int, Field(strict=True, ge=0, le=1000)]
Point = tuple[Coordinate, Coordinate]
Millimeters = Annotated[float, Field(ge=1, le=30000, allow_inf_nan=False)]
Span = tuple[Millimeters, Millimeters]
ReferenceID = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9]{0,7}$")]


class ObjectSize(Strict):
    major_mm: Span
    minor_mm: Span
    basis: Literal["measured", "exact_catalog", "similar_catalog"]
    source: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def ordered(self):
        if not self.major_mm[0] <= self.major_mm[1] or not self.minor_mm[0] <= self.minor_mm[1]:
            raise ValueError("dimension intervals must be [minimum, maximum] in millimeters")
        if self.major_mm[1] < self.minor_mm[0]:
            raise ValueError("major axis must be at least as long as the minor axis")
        return self


class SceneSpan(Strict):
    label: str = Field(min_length=1, max_length=80)
    span_mm: Span
    points: tuple[Point, Point]

    @model_validator(mode="after")
    def ordered(self):
        if self.span_mm[0] > self.span_mm[1] or self.points[0] == self.points[1]:
            raise ValueError("reference needs an ordered length interval and distinct endpoints")
        return self


class SceneAnchor(SceneSpan):
    id: ReferenceID


class SceneReference(SceneAnchor):
    basis: Literal["measured", "exact_catalog"]
    source: str = Field(min_length=1, max_length=600)


class SceneEstimate(Strict):
    references: list[SceneAnchor] = Field(max_length=3)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [reference.id for reference in self.references]
        if len(ids)!=len(set(ids)):
            raise ValueError("scene reference IDs must be unique")
        return self


class ScaleReview(NeedsReview):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ScaleFit:
    asset_angle: float
    reference_angle: float
    depth: float
    depth_tolerance: float
    factor: float
    record: dict


def silhouette_axes(alpha):
    points = cv2.findNonZero((alpha>=128).astype("uint8"))
    if points is None or len(points)<3:
        raise ScaleReview("asset_shape", "物品轮廓不足以校验尺寸")
    _,(width,height),angle = cv2.minAreaRect(points)
    if min(width,height)<2:
        raise ScaleReview("asset_shape", "物品轮廓过窄，无法校验尺寸")
    if width < height:
        return float(height),float(width),float(angle+90)
    return float(width),float(height),float(angle)


def depth_at(depth, point, radius):
    x,y = point
    return float(np.median(depth[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]))


def reference_geometry(reference, frame_size, depth, minimum=None):
    p,q = (frame_point(point,frame_size) for point in reference.points)
    segment_px = math.dist(p,q)
    known = isinstance(reference,SceneReference)
    required = minimum if minimum is not None else (24 if known else max(24,frame_size[0]*.045))
    if segment_px < required:
        raise ScaleReview("reference_too_short", "参照线段过短，像素测量不可靠")
    tolerance = .048 if known else .085
    radius = max(2,min(10,round(segment_px*.02)))
    samples = [depth_at(depth,tuple(round(a+(b-a)*t) for a,b in zip(p,q,strict=True)),radius)
               for t in (.16,.33,.50,.67,.84)]
    if max(samples)-min(samples)>tolerance*1.3:
        raise ScaleReview("reference_not_local", "参照线段跨越不同深度")
    return p,q,segment_px,float(np.median(samples)),tolerance,radius


def reference_radius(frame_width, segment_px):
    return min(frame_width*.23,max(frame_width*.075,segment_px*2.25))


def reference_region(coarse_depth, frame_size, geometry):
    p,q,segment_px,ref_depth,tolerance,_ = geometry
    width,height = frame_size
    h,w = coarse_depth.shape
    ys,xs = np.ogrid[:h,:w]
    dx = xs*(width-1)/max(1,w-1)-(p[0]+q[0])/2
    dy = ys*(height-1)/max(1,h-1)-(p[1]+q[1])/2
    return ((dx**2+dy**2<=reference_radius(width,segment_px)**2)
            & (np.abs(coarse_depth-ref_depth)<=tolerance))


def project_span(reference, segment_px, major_mm):
    low,high = reference.span_mm
    endpoint_error = 2 if isinstance(reference,SceneReference) else max(3,segment_px*.06)
    scale_low,scale_high = (segment_px-endpoint_error)/high,(segment_px+endpoint_error)/low
    px_per_mm = segment_px/math.sqrt(low*high)
    return px_per_mm,(scale_low,scale_high),(
        major_mm[0]*scale_low,major_mm[1]*scale_high)


def fit_scale(alpha, frame_size, depth, seat, object_size, reference):
    axis_px,cross_px,asset_angle = silhouette_axes(alpha)
    ratio = axis_px/cross_px
    # Allow small silhouette/view differences, including fasteners at the outline.
    lo = max(object_size.major_mm[0],ratio*object_size.minor_mm[0]*.85)
    hi = min(object_size.major_mm[1],ratio*object_size.minor_mm[1]*1.18)
    if lo > hi:
        raise ScaleReview("view_size_mismatch", "素材长宽比例和所给实物尺寸不符")
    p,q,segment_px,ref_depth,depth_tolerance,radius = reference_geometry(reference,frame_size,depth)
    known = isinstance(reference,SceneReference)
    midpoint = tuple(round((a+b)/2) for a,b in zip(p,q,strict=True))
    support = frame_point(seat,frame_size)
    w = frame_size[0]
    if math.dist(midpoint,support)>reference_radius(w,segment_px):
        raise ScaleReview("reference_too_far", "参照件离承托点过远，无法沿用局部尺度")
    seat_depth = depth_at(depth,support,radius)
    if abs(ref_depth-seat_depth)>depth_tolerance:
        raise ScaleReview("reference_depth_mismatch", "参照线段与承托处不在可比的局部深度")

    px_per_mm,(scale_low,scale_high),(span_min,span_max) = project_span(reference,segment_px,(lo,hi))
    if span_max/span_min>3.5:
        raise ScaleReview("scale_too_uncertain", "物品和参照尺寸的不确定范围过大")
    nominal_mm = math.sqrt(lo*hi)
    nominal_px = nominal_mm*px_per_mm
    verified = (known and object_size.basis!="similar_catalog" and span_max/span_min<=1.25)
    reference_angle = math.degrees(math.atan2(q[1]-p[1],q[0]-p[0]))
    record = {
        "status":"calibrated" if verified else "estimated",
        "method":"local_reference_segment",
        "reference":reference.model_dump(mode="json"),
        "reference_basis":"provided" if known else "vision_estimate",
        "reference_points_px":[list(p),list(q)],
        "reference_span_px":round(segment_px,2),
        "support_point_px":list(support),
        "relative_depth_delta":round(abs(ref_depth-seat_depth),4),
        "px_per_mm":round(px_per_mm,5),
        "px_per_mm_range":[round(scale_low,5),round(scale_high,5)],
        "object_major_mm_used":round(nominal_mm,1),
        "projected_major_px":round(nominal_px,2),
        "projected_minor_px":round(nominal_px/ratio,2),
        "projected_major_px_range":[round(span_min,2),round(span_max,2)],
    }
    return ScaleFit(asset_angle,reference_angle,ref_depth,depth_tolerance,nominal_px/axis_px,record)


def verify_scale(alpha, record):
    major,minor,_ = silhouette_axes(alpha)
    for value,key in ((major,"projected_major_px"),(minor,"projected_minor_px")):
        expected = record[key]
        if abs(value-expected)>max(3.,expected*.05):
            raise ScaleReview("render_scale_mismatch", "Photoshop 变换后的实际尺寸与投影尺寸不符")
    return {**record,"rendered_major_px":round(major,2),"rendered_minor_px":round(minor,2)}


def fit_consensus(alpha, frame_size, depth, seat, object_size, references):
    """Intersect local scale intervals without promoting guesses to measurements."""
    from scene_v2 import angle_delta

    major,cross,asset_angle = silhouette_axes(alpha)
    aspect = major/cross
    low = max(object_size.major_mm[0],aspect*object_size.minor_mm[0]*.85)
    high = min(object_size.major_mm[1],aspect*object_size.minor_mm[1]*1.18)
    if low>high:
        raise ScaleReview("view_size_mismatch","素材外廓与所给实物尺寸不符")
    support = frame_point(seat,frame_size)
    usable = []
    rejected = []
    for reference in references:
        known = isinstance(reference,SceneReference)
        try:
            if not known and reference.span_mm[1]/reference.span_mm[0]>2.5:
                raise ScaleReview("reference_size_weak","参照件毫米范围过宽，无法独立佐证局部尺度")
            p,q,pixels,ref_depth,tolerance,radius = reference_geometry(
                reference,frame_size,depth,minimum=24 if known else max(24,frame_size[0]*.025))
            midpoint = ((p[0]+q[0])/2,(p[1]+q[1])/2)
            delta = abs(ref_depth-depth_at(depth,support,radius))
            if math.dist(midpoint,support)>reference_radius(frame_size[0],pixels) or delta>tolerance:
                raise ScaleReview("reference_off_plane","参照不在承托附近的可比深度")
        except ScaleReview as error:
            rejected.append({"ref":reference.id,"code":error.code})
            continue
        _,interval,_ = project_span(reference,pixels,(low,high))
        angle = math.degrees(math.atan2(q[1]-p[1],q[0]-p[0]))
        usable.append((known,reference,p,q,pixels,ref_depth,tolerance,delta,interval,angle))
    known = [m for m in usable if m[0]]
    if known:
        usable = known
    else:
        independent = {(m[1].id,frozenset((m[2],m[3]))) for m in usable}
        if len({ident for ident,_ in independent})<2 or len({points for _,points in independent})<2:
            raise ScaleReview("insufficient_local_references","该承托区缺少两件有效的独立尺度参照")
        if max(abs(angle_delta(m[-1],usable[0][-1])) for m in usable)>24:
            raise ScaleReview("reference_direction_conflict","参照投影方向不一致")
    scale_low = max(m[-2][0] for m in usable)
    scale_high = min(m[-2][1] for m in usable)
    if scale_low>scale_high:
        raise ScaleReview("reference_conflict","同层参照的比例区间不重叠")
    span_low,span_high = low*scale_low,high*scale_high
    if span_high/span_low>1.8:
        raise ScaleReview("scale_too_uncertain","物品投影尺寸区间超过 1.8 倍")
    _,ref,p,q,pixels,ref_depth,tolerance,delta,_,angle = max(usable,key=lambda m:m[4])
    nominal_scale = math.sqrt(scale_low*scale_high)
    nominal_mm = math.sqrt(low*high)
    nominal_px = nominal_mm*nominal_scale
    status = ("calibrated" if known and object_size.basis!="similar_catalog"
              and span_high/span_low<=1.25 else "estimated")
    record = {"status":status,"method":"local_reference_consensus",
              "reference":ref.model_dump(mode="json"),
              "references":[m[1].model_dump(mode="json") for m in usable],
              "reference_basis":"provided" if known else "vision_estimate",
              "reference_points_px":[list(p),list(q)],"reference_span_px":round(pixels,2),
              "support_point_px":list(support),"relative_depth_delta":round(delta,4),
              "px_per_mm":round(nominal_scale,5),
              "px_per_mm_range":[round(scale_low,5),round(scale_high,5)],
              "object_major_mm_used":round(nominal_mm,1),
              "projected_major_px":round(nominal_px,2),
              "projected_minor_px":round(nominal_px/aspect,2),
              "projected_major_px_range":[round(span_low,2),round(span_high,2)],
              "rejected_references":rejected}
    return ScaleFit(asset_angle,angle,ref_depth,tolerance,nominal_px/major,record)
