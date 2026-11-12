from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Unit = Annotated[float, Field(ge=0, le=1)]
Point = tuple[Unit, Unit]
Polygon = Annotated[list[Point], Field(min_length=3, max_length=24)]
Score = Annotated[int, Field(strict=True, ge=0, le=4)]


class EffectSettings(Strict):
    """Opt-in online trial. Fog is extra airlight, dust is a material layer."""
    fog_extra: float = Field(default=.35, ge=0, le=.85, strict=True)
    pool_kernel: int = Field(default=3, ge=1, le=64, strict=True)
    pool_weight: float = Field(default=1., ge=0, le=1, strict=True)
    surface_dust: float = Field(default=.5, ge=0, le=1, strict=True)
    lighting: Literal['local', 'global_mean'] = 'global_mean'
    ai_adjust: bool = True
    fog_mode: Literal['manual_extra', 'local_auto'] = 'manual_extra'
    weathering: Literal['legacy_matte', 'preserve_source'] = 'legacy_matte'
    shadow_mode: Literal['legacy', 'matched_receiver'] = 'legacy'
    fog_gain: float = Field(default=1., ge=0, le=1.5, strict=True)

    @model_validator(mode='after')
    def preserve_limits(self):
        if self.weathering == 'preserve_source' and self.surface_dust > .5:
            raise ValueError('preserve_source surface_dust must not exceed 0.5')
        return self


class BatchConfig(Strict):
    backgrounds: str
    output: str
    assets: list[str] = Field(default_factory=list)
    excluded_assets: list[str] = Field(default_factory=lambda:['5','6','7'])
    asset_scale_factors: dict[str, Annotated[float, Field(ge=.1, le=2.)]] = Field(default_factory=lambda:{'31':.375,'33':.25})
    count: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=20261110, ge=0)
    max_iterations: int = Field(default=2, ge=0, le=10)
    provider_config: str | None = None
    render_device: Literal["OPTIX", "CUDA", "CPU"] = "OPTIX"
    offline: bool = False
    photo_style: Literal['dusty_scrap', 'scrap_gray', 'retain_color'] = 'scrap_gray'
    placement_mode: Literal['auto', 'surface', 'combined', 'embedded'] = 'auto'
    uncertain_policy: Literal['render_review', 'skip'] = 'render_review'
    cache: str = "work/display-batch-cache"
    # A manifest is a list of {background, asset_id, group}; it fixes evaluation jobs.
    manifest: str | None = None
    ai_concurrency: int = Field(default=2, ge=1, le=2)
    cpu_concurrency: int = Field(default=2, ge=1, le=2)
    price_input_per_million: float | None = Field(default=None, ge=0)
    price_output_per_million: float | None = Field(default=None, ge=0)
    price_cached_input_per_million: float | None = Field(default=None, ge=0)
    price_currency: str = Field(default='CNY', pattern=r'^[A-Z]{3}$')
    # New photographic profiles are promoted after staged evaluation.
    appearance: Literal['reference','camera','materials','scene','native','local'] = 'reference'
    atmosphere: Literal['auto','off'] = 'auto'
    review_format: Literal['full','compact'] = 'full'
    effects: EffectSettings | None = None
    scale_mode: Literal['legacy', 'physical_consistent', 'reference_projection'] = 'legacy'
    contact_mode: Literal['legacy', 'strict'] = 'legacy'
    chroma_mode: Literal['legacy', 'local_budget', 'global_mean'] = 'legacy'
    dimension_registry: str | None = None
    scene_reference_registry: str | None = None
    geometry_preset: Literal['legacy', 'rebuild'] = 'legacy'
    vehicle_reference: dict | None = None
    integration_mode: Literal['legacy', 'coherent'] = 'legacy'
    camera_pipeline: Literal['legacy_residual', 'premultiplied'] = 'legacy_residual'
    support_geometry: Literal['legacy', 'structured'] = 'legacy'

    @model_validator(mode='after')
    def dust_uses_neighbourhood(self):
        if self.geometry_preset=='rebuild':
            if not self.vehicle_reference:
                raise ValueError('geometry rebuild requires a vehicle_reference with internal_width_m and source')
            from .vehicle_scale import validate_reference
            validate_reference(self.vehicle_reference)
            self.scale_mode='reference_projection';self.contact_mode='strict';self.chroma_mode='local_budget'
            self.effects=EffectSettings(fog_extra=0.,pool_kernel=3,surface_dust=.5,lighting='local',
                fog_mode='local_auto',weathering='preserve_source',shadow_mode='matched_receiver')
            self.asset_scale_factors={}
        if self.integration_mode=='coherent' and self.effects is None:
            self.effects=EffectSettings(fog_extra=0.,pool_kernel=3,surface_dust=.5,lighting='local',
                fog_mode='local_auto',weathering='preserve_source',shadow_mode='matched_receiver')
        # Selecting this style in either CLI or web must actually use local evidence.
        if self.photo_style=='dusty_scrap':self.appearance='local'
        if self.effects is not None:
            self.appearance='local'
            self.photo_style='dusty_scrap'
        return self


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


class StructuredSurface(Surface):
    geometry_kind: Literal['planar','curved','compound','unknown']


class StructuredSceneAnswer(SceneAnswer):
    surfaces: list[StructuredSurface] = Field(max_length=8)


class AtmosphereAnswer(Strict):
    kind: Literal['clear','haze','veil','mixed','uncertain']
    confidence: Unit
    haze_regions: list[tuple[Unit, Unit, Unit, Unit]] = Field(max_length=6)
    light_regions: list[tuple[Unit, Unit, Unit, Unit]] = Field(max_length=6)
    exclude_regions: list[tuple[Unit, Unit, Unit, Unit]] = Field(max_length=6)
    light_direction: tuple[float, float] | None
    evidence: str = Field(max_length=240)

    @model_validator(mode='after')
    def ordered_regions(self):
        for left,top,right,bottom in self.haze_regions+self.light_regions+self.exclude_regions:
            if left>=right or top>=bottom:
                raise ValueError('region must be ordered [left, top, right, bottom]')
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


DefectCode = Literal["upright", "floating", "penetration", "occlusion", "scale", "camera",
                     "material", "lighting", "color", "shadow", "edge", "texture", "structure",
                     "haze_clear", "haze_heavy", "light_direction", "reflection_flat", "reflection_hard", "surface_clean", "patch_seam",
                     "surface_dust_heavy", "too_sharp", "too_blurry", "identity_loss", "haze_color"]


class Defect(Strict):
    code: DefectCode
    severity: Literal["minor", "major", "fatal"]
    region: tuple[Unit, Unit, Unit, Unit]
    evidence: str = Field(max_length=160)
    scope: Literal['image','asset'] = 'image'
    visibility: Literal['visible','occluded','unassessable'] = 'visible'


class Action(Strict):
    kind: Literal["candidate", "material", "light", "tone", "chroma", "dust", "contrast", "haze", "sharpness",
                  "fog_extra", "pool_kernel", "surface_dust", "fog_gain"]
    value: str = Field(max_length=32)


class Review(Strict):
    scores: Scores
    defects: list[Defect] = Field(max_length=8)
    actions: list[Action] = Field(max_length=3)
    evidence: str = Field(max_length=240)
    def passes(self):
        return self.scores.passes() and not any(d.severity in ("major", "fatal") for d in self.defects)

    def photo_passes(self):
        return min(self.scores.lighting,self.scores.material,self.scores.texture)>=3 and not any(
            d.severity in ('major','fatal') and d.code in ('color','lighting','material','texture',
                'haze_clear','haze_heavy','light_direction','reflection_flat','reflection_hard','surface_clean','patch_seam',
                'surface_dust_heavy','too_sharp','too_blurry','identity_loss','haze_color')
            for d in self.defects)


class FusionReview(Review):
    trace: Literal['none','uncertain','visible']
    actions: list[Action] = Field(default_factory=list,max_length=0)

    def passes(self):
        return self.trace=='none' and super().passes()


class EffectsReview(Review):
    trace: Literal['none','uncertain','visible']

    def passes(self):
        return self.trace=='none' and super().passes()


def effects_review_schema(allowed):
    """Constrain each kind/value pair in the provider schema, as well as the executor."""
    types=[]
    for kind,values in allowed.items():
        if values:
            types.append(create_model('Allowed_'+kind, __base__=Action,
                kind=(Literal[kind], ...),value=(Literal[tuple(values)], ...)))
    if not types:
        return create_model('AvailableEffectsReview',__base__=EffectsReview,
                            actions=(list[Action],Field(max_length=0)))
    return create_model('AvailableEffectsReview',__base__=EffectsReview,
                        actions=(list[Union[tuple(types)]],Field(max_length=3)))


class FusionCue(Strict):
    code: DefectCode
    severity: Literal['minor','major','fatal']
    evidence: str = Field(max_length=160)


class FusionObservation(Strict):
    trace: Literal['none','uncertain','visible']
    scores: Scores
    defects: list[FusionCue] = Field(max_length=6)
    evidence: str = Field(max_length=240)

    def review(self):
        # Recognition describes visible image cues. Asset quarantine remains a separate check.
        return FusionReview(trace=self.trace,scores=self.scores,evidence=self.evidence,
                            defects=[Defect(**d.model_dump(),region=(0,0,1,1)) for d in self.defects])


class Probe(Strict):
    image_visible: bool
    description: str = Field(min_length=2, max_length=100)


class StructureConfirmation(Strict):
    confirmed_damage: bool
    visibility: Literal['visible','occluded','unassessable']
    evidence: str = Field(min_length=2, max_length=240)


MATERIAL_PROFILES = {"original": 0., "clearer": -.08, "softer": .08, "weathered": .03}
LIGHT_PROFILES = {"scene": 1., "softer": .65, "defined": 1.25}
TONE_PROFILES = {"scene": (0., 0.), "darker": (-.25, 0.), "brighter": (.25, 0.),
                 "cooler": (0., -1.5), "warmer": (0., 1.5)}
PHOTO_CONTROLS = {name: {'auto':1., 'less':.7, 'more':1.3}
                  for name in ('chroma','dust','contrast','haze','sharpness')}
EFFECT_ACTIONS = {
    'fog_gain': ('0', '0.5', '0.75', '1', '1.25', '1.5'),
    'fog_extra': ('0', '0.15', '0.25', '0.35', '0.45', '0.55', '0.7', '0.85'),
    'pool_kernel': ('1', '2', '3', '6', '12', '16', '32', '64'),
    'surface_dust': ('0', '0.25', '0.5', '0.75', '1'),
}


def apply_actions(params, actions, candidate_ids):
    result = dict(params)
    allowed = {"candidate": set(candidate_ids), "material": set(MATERIAL_PROFILES),
               "light": set(LIGHT_PROFILES), "tone": set(TONE_PROFILES)}
    allowed.update({k:set(v) for k,v in PHOTO_CONTROLS.items()})
    allowed.update({k:set(v) for k,v in EFFECT_ACTIONS.items() if params.get('effects_enabled')})
    seen = set()
    for action in actions:
        if action.kind in seen or action.value not in allowed.get(action.kind, ()):
            raise ValueError("duplicate or unavailable modification")
        seen.add(action.kind)
        result[action.kind] = action.value
    return result


def version_rank(checks, review):
    hard = sum(c.get("status") == "pass" for c in checks.values())
    failures = sum(c.get("status") == "fail" for c in checks.values())
    unknown = sum(c.get("status") == "unknown" for c in checks.values())
    if review is None:
        return (-failures, -99, -99, -unknown, hard, 0)
    return (-failures, -sum(d.severity == "fatal" for d in review.defects),
            -sum(d.severity == "major" for d in review.defects), -unknown, hard,
            sum(review.scores.model_dump().values()))
