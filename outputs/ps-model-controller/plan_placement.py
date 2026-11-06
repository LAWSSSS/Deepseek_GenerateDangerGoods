"""Ask a multimodal planner where to place an existing cutout in a selected scene crop."""
import argparse
import asyncio
import json
from pathlib import Path


import httpx
from pydantic import Field, model_validator

from run import Config, ModelClient, Strict, Unit, load


class Placement(Strict):
    center: tuple[Unit, Unit]
    width: float = Field(ge=.12, le=.50)
    angle: float = Field(ge=-65, le=65)
    exposure: float = Field(ge=-20, le=20)
    shadow_opacity: int = Field(ge=12, le=50)
    occluders: list[list[tuple[Unit, Unit]]] = Field(max_length=2)
    note: str = Field(max_length=150)

    @model_validator(mode="after")
    def validate_geometry(self):
        x, y = self.center
        if not (.15 <= x <= .85 and .16 <= y <= .84 and self.width/2 < x < 1-self.width/2):
            raise ValueError("object center must stay inside the crop")
        for poly in self.occluders:
            if not 3 <= len(poly) <= 16:
                raise ValueError("occluder must have 3–16 vertices")
            if all(abs(px-x)>self.width*.75 or abs(py-y)>.42 for px,py in poly):
                raise ValueError("occlusion polygon is too far from the object")
        return self


PROMPT = """Plan Photoshop placement of one existing cutout. Image 1 is the CCTV background crop; image 2 is the object on a neutral matte. Return only JSON:
{"center":[0.5,0.55],"width":0.16,"angle":0,"exposure":0,"shadow_opacity":25,"occluders":[],"note":""}
Coordinates 0..1 from the top left of image 1. Width is relative to the crop, across the rotated cutout before occlusion. Choose the smallest credible scale within 0.12..0.50. Recess the object into a real scrap gap with solid support; avoid display-like placement on top. Prefer actual foreground scrap hiding most of the outline while leaving a characteristic part visible. Trace only verified foreground intersections; never cut an empty gap or invent a foreground piece. Use [] when unsure. Preserve the asset's shape and aged surface, match local light and contact shadow, and protect background, camera angle, timestamp and camera text.
Use center x 0.25..0.70, y 0.35..0.75; clockwise angle -40..40; signed midtone exposure -12..12; integer shadow_opacity 12..50; up to two polygons of 3–12 [x,y] points. Exactly seven keys, note under 100 characters."""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--instruction", default="Place this cutout naturally among the scrap.")
    args = parser.parse_args()
    config = load(Config, args.config)
    if not config.planner:
        parser.error("planner endpoint required")
    report = {"calls":[]}
    async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as http:
        client = ModelClient(http, report)
        try:
            payload = {"request":args.instruction}
            previews = (args.scene,args.asset)
            prompt = PROMPT
            if args.trial:
                previews += (args.trial,)
                prompt += " Image 3 is the Photoshop trial. Correct unsupported pose, oversized scale or false occlusion. Keep a small recognizable feature exposed where real foreground scrap crosses it; remove polygons that cut across empty space."
                if args.prior:
                    payload["prior"] = Placement.model_validate_json(json.dumps(json.loads(args.prior.read_text(encoding="utf8"))["placement"])).model_dump()
            placement = await client.ask(config.planner,"placement",Placement,prompt,payload,previews)
        except (ValueError, RuntimeError) as error:
            print(json.dumps({**report,"error":str(error)},ensure_ascii=False))
            raise SystemExit(2) from None
    report["placement"] = placement.model_dump()
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
