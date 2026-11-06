"""Cached scene analysis and scale-consensus checks for the 3-D pilot."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import cv2
from PIL import Image
from pydantic import Field, model_validator

from auto_v3 import (SCENE_PROMPT, SceneAnswer, atlas_geometry, checked_scene, draw_atlas,
                     protected_rects)
from metric_scale import ObjectSize, ScaleReview, depth_at, fit_consensus
from run import load
from scene_v2 import atomic_json, digest, key


Angle=Annotated[int,Field(ge=10,le=85)]


class MeshSceneAnswer(SceneAnswer):
    elevation_deg: tuple[Angle,Angle]

    @model_validator(mode="after")
    def elevation_is_ordered(self):
        lo,hi=self.elevation_deg
        if lo>=hi:
            raise ValueError("Camera elevation must be an ordered interval")
        return self


PROMPT=(SCENE_PROMPT+" Also add elevation_deg:[minimum,maximum] in degrees above the "
        "horizontal, inferred from the camera view. Do not claim precision if uncertain.")


async def analyze_scene(runner,row,background,frame,engine,folder):
    endpoint=runner.config.vision
    if endpoint is None or runner.http is None:
        raise RuntimeError("必须配置视觉模型进行场景核验")
    rgb,depth,masks,size=engine.scene(background)
    zones,refs=atlas_geometry(depth,masks)
    if not zones or len(refs)<2:
        raise ScaleReview("insufficient_scene_evidence","场景缺少可核验的独立参照")
    preview=folder/"scene-atlas.jpg"
    draw_atlas(frame,zones,refs,preview)
    signature=key("mesh-scene-v1",digest(background),engine.versions,endpoint.model_dump(),zones,refs)
    cached=runner.cache/f"mesh-scene-{signature}.json"
    if cached.is_file():
        answer=load(MeshSceneAnswer,cached)
        hit=True
    else:
        answer=await runner.ask(SimpleNamespace(job_id=row["job_id"]),row,
                                endpoint.model_copy(update={"max_output_tokens":256}),
                                "mesh_scene",MeshSceneAnswer,PROMPT,
                                {"zones":list(zones),"refs":list(refs)},preview,
                                tool="inspect_scene",detail="high")
        checked_scene(answer,zones,refs)
        atomic_json(cached,answer.model_dump())
        hit=False
    measurements=checked_scene(answer,zones,refs)
    lo,hi=answer.elevation_deg
    row["scene_analysis"]={"answer":answer.model_dump(),"cache_hit":hit}
    if hi-lo>25:
        raise ScaleReview("camera_uncertain","场景俯视角不足以验证三维相机")
    return {"answer":answer,"zones":zones,"references":measurements,
            "protected":protected_rects(answer,size),"depth":depth,"size":size,"cache_hit":hit,
            "camera_elevation_deg":(lo+hi)/2}


def candidate_scale(row,candidate,alpha,scene):
    width,height=scene["size"]
    x,y=candidate["center_px"]
    seat=(round(1000*x/(width-1)),round(1000*y/(height-1)))
    if "depth_native" not in scene:
        scene["depth_native"]=cv2.resize(scene["depth"],scene["size"],interpolation=cv2.INTER_LINEAR)
    depth=scene["depth_native"]
    if not row.get("physical_size"):
        approved=row.get("approved_size_reference")
        if not approved:
            raise ScaleReview("missing_object_size","物品真实尺寸和已确认的同层像素参照均缺失")
        l,t,r,b=approved["bounds_px"]
        ref=(round((l+r)/2),round((t+b)/2))
        delta=abs(depth_at(depth,(int(x),int(y)),5)-depth_at(depth,ref,5))
        distance=((x-ref[0])**2+(y-ref[1])**2)**.5
        target=candidate["target_px"]
        if delta>.045 or distance>.22*width or abs(target-approved["major_px"])>max(3,.05*target):
            raise ScaleReview("approved_reference_not_local","已确认尺寸的参照位置不在同一局部深度")
        return {"status":"estimated","method":"approved_pixels_at_comparable_depth",
                "projected_major_px_range":[round(target*.95,2),round(target*1.05,2)],
                "reference_bounds_px":approved["bounds_px"],"relative_depth_delta":round(delta,4),
                "distance_px":round(distance,1)}
    # The pilot reads persisted JSON, where dimensional tuples are arrays.
    size=ObjectSize.model_validate_json(json.dumps(row["physical_size"]))
    fit=fit_consensus(alpha,scene["size"],depth,seat,size,list(scene["references"].values()))
    low,high=fit.record["projected_major_px_range"]
    if not low<=candidate["target_px"]<=high:
        raise ScaleReview("approved_size_conflicts_with_scene","已确认的画面尺寸超出局部参照范围")
    return fit.record
