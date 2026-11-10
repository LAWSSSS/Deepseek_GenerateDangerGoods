from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Unit = Annotated[float, Field(ge=0, le=1)]
Point = tuple[Unit, Unit]
Polygon = Annotated[list[Point], Field(min_length=3, max_length=24)]
Score = Annotated[int, Field(strict=True, ge=0, le=4)]


class BatchConfig(Strict):
    backgrounds: str
    output: str
    assets: list[str] = Field(default_factory=list)
    count: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=20261110, ge=0)
    max_iterations: int = Field(default=2, ge=0, le=10)
    provider_config: str | None = None
    render_device: Literal["OPTIX", "CUDA", "CPU"] = "OPTIX"
    offline: bool = False
    cache: str = "work/display-batch-cache"
    # A manifest is a list of {background, asset_id, group}; it fixes evaluation jobs.
    manifest: str | None = None
    ai_concurrency: int = Field(default=2, ge=1, le=2)
    cpu_concurrency: int = Field(default=2, ge=1, le=2)
    price_input_per_million: float | None = Field(default=None, ge=0)
    price_output_per_million: float | None = Field(default=None, ge=0)


class Surface(Strict):
    id: str = Field(pattern=r"^S[0-9]{1,2}$")
    polygon: Polygon
    label: str = Field(max_length=80)
    anchor: Point
    reference: tuple[Point, Point]
    reference_mm: tuple[float, float]
    confidence: Unit
    slope_deg: float = Field(ge=0, le=45)
    @model_validator(mode="after")
    def reference_bounds(self):
        lo, hi = self.reference_mm
        if not 5 <= lo <= hi <= 30000 or self.reference[0] == self.reference[1]:
            raise ValueError("reference must have positive ordered millimeter bounds")
        return self


class SceneAnswer(Strict):
    hopper: Polygon
    excluded: list[Polygon] = Field(max_length=12)
    protected: list[Polygon] = Field(max_length=8)
    surfaces: list[Surface] = Field(max_length=8)
    elevation: tuple[float, float]
    light_direction: tuple[float, float] | None
    light_confidence: Unit
    haze: float = Field(ge=0, le=.35)
    evidence: str = Field(max_length=320)
    @model_validator(mode="after")
    def validate_scene(self):
        lo, hi = self.elevation
        if not 10 <= lo <= hi <= 85 or hi-lo > 35:
            raise ValueError("unusable camera interval")
        ids = [s.id for s in self.surfaces]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate surface IDs")
        if self.light_direction and any(abs(v)>1 for v in self.light_direction):
            raise ValueError("invalid screen light direction")
        return self


class AssetAnswer(Strict):
    major_mm: tuple[float, float]
    assembly: Literal["rigid_assembly", "single_piece", "loose_parts", "unknown"]
    upright_axis: Literal["x", "y", "z", "none"]
    material_classes: list[Literal["steel", "chrome", "paint", "rust", "rubber", "plastic"]]
    key_features: list[str] = Field(max_length=6)
    evidence: str = Field(max_length=300)
    @model_validator(mode="after")
    def dimensions(self):
        if not 5 <= self.major_mm[0] <= self.major_mm[1] <= 20000:
            raise ValueError("invalid physical size interval")
        return self


class Scores(Strict):
    support: Score
    occlusion: Score
    lighting: Score
    edge: Score
    texture: Score
    material: Score
    structure: Score
    def passes(self):
        old = [self.support, self.occlusion, self.lighting, self.edge, self.texture]
        return min(old) >= 3 and sum(old) >= 16 and min(self.material, self.structure) >= 3


class Defect(Strict):
    code: Literal["upright", "floating", "penetration", "occlusion", "scale", "camera",
                  "material", "lighting", "color", "shadow", "edge", "texture", "structure"]
    severity: Literal["minor", "major", "fatal"]
    region: tuple[Unit, Unit, Unit, Unit]
    evidence: str = Field(max_length=160)
    scope: Literal['image','asset'] = 'image'


class Action(Strict):
    kind: Literal["candidate", "material", "light", "tone"]
    value: str = Field(max_length=32)


class Review(Strict):
    scores: Scores
    defects: list[Defect] = Field(max_length=8)
    actions: list[Action] = Field(max_length=3)
    evidence: str = Field(max_length=240)
    def passes(self):
        return self.scores.passes() and not any(d.severity in ("major", "fatal") for d in self.defects)


class Probe(Strict):
    image_visible: bool
    description: str = Field(min_length=2, max_length=100)


MATERIAL_PROFILES = {"original": 0., "clearer": -.08, "softer": .08, "weathered": .03}
LIGHT_PROFILES = {"scene": 1., "softer": .65, "defined": 1.25}
TONE_PROFILES = {"scene": (0., 0.), "darker": (-.25, 0.), "brighter": (.25, 0.),
                 "cooler": (0., -1.5), "warmer": (0., 1.5)}


def apply_actions(params, actions, candidate_ids):
    result = dict(params)
    allowed = {"candidate": set(candidate_ids), "material": set(MATERIAL_PROFILES),
               "light": set(LIGHT_PROFILES), "tone": set(TONE_PROFILES)}
    seen = set()
    for action in actions:
        if action.kind in seen or action.value not in allowed[action.kind]:
            raise ValueError("duplicate or unavailable modification")
        seen.add(action.kind)
        result[action.kind] = action.value
    return result


def version_rank(checks, review):
    hard = sum(c.get("status") == "pass" for c in checks.values())
    failures = sum(c.get("status") != "pass" for c in checks.values())
    if review is None:
        return (-failures, hard, -99, -99, 0)
    return (-failures, hard, -sum(d.severity == "fatal" for d in review.defects),
            -sum(d.severity == "major" for d in review.defects), sum(review.scores.model_dump().values()))
