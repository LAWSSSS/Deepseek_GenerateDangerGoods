"""Provider-neutral, versioned observations and bounded action plans."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VERSION = 'visual-evidence-v1'
Unit = Annotated[float, Field(ge=0, le=1)]
Score = Annotated[int, Field(ge=0, le=4)]
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')]
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Value = str | float | int

DefectCode = Literal[
    'upright', 'floating', 'penetration', 'occlusion', 'scale', 'camera',
    'material', 'lighting', 'color', 'shadow', 'edge', 'texture', 'structure',
    'haze_clear', 'haze_heavy', 'light_direction', 'reflection_flat',
    'reflection_hard', 'surface_clean', 'patch_seam', 'surface_dust_heavy',
    'too_sharp', 'too_blurry', 'identity_loss', 'haze_color',
]


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Scores(Strict):
    support: Score
    occlusion: Score
    lighting: Score
    edge: Score
    texture: Score
    material: Score
    structure: Score


class FindingDetails(Strict):
    code: DefectCode
    region: tuple[Unit, Unit, Unit, Unit] | None
    severity: Literal['minor', 'major', 'fatal']
    visibility: Literal['visible', 'occluded', 'unassessable']
    scope: Literal['image', 'asset']
    confidence: Unit
    direction: Literal['increase', 'decrease', 'unknown']
    evidence: str = Field(min_length=1, max_length=240)

    @model_validator(mode='after')
    def validate_region(self):
        if self.region is not None:
            left, top, right, bottom = self.region
            if left >= right or top >= bottom:
                raise ValueError('region must be a nonempty rectangle in full image coordinates')
        elif self.visibility == 'visible':
            raise ValueError('visible findings need a region')
        return self


PHYSICAL_CUES = {'floating':'visible_air_gap', 'penetration':'surface_intersection',
                 'occlusion':'reversed_overlap', 'upright':'standing_on_end'}
PhysicalCue = Literal['visible_air_gap', 'surface_intersection', 'reversed_overlap', 'standing_on_end']


class Finding(FindingDetails):
    id: Identifier
    target: Identifier
    physical_cue: PhysicalCue | None = None


class FrameFinding(FindingDetails):
    physical_cue: PhysicalCue | None = None

    @model_validator(mode='after')
    def check_physical_cue(self):
        expected = PHYSICAL_CUES.get(self.code) if self.visibility == 'visible' else None
        if self.physical_cue != expected:
            raise ValueError('visible geometric defects require their matching direct physical cue')
        return self


class VisualObservation(Strict):
    """The single-object provider response, before the host assigns identity."""
    trace: Literal['none', 'uncertain', 'visible']
    scores: Scores
    findings: list[FrameFinding] = Field(max_length=8)
    summary: str = Field(default='', max_length=240)

    def for_frame(self, image_sha256, target):
        return VisualReport(version=VERSION, image_sha256=image_sha256, trace=self.trace,
                            scores=self.scores, summary=self.summary,
                            findings=[Finding(id=f'f{i}', target=target, **finding.model_dump())
                                      for i, finding in enumerate(self.findings, 1)])


class VisualReport(Strict):
    version: Literal['visual-evidence-v1']
    image_sha256: Digest
    trace: Literal['none', 'uncertain', 'visible']
    scores: Scores
    findings: list[Finding] = Field(max_length=8)
    summary: str = Field(max_length=240)

    @model_validator(mode='after')
    def unique_findings(self):
        if len({finding.id for finding in self.findings}) != len(self.findings):
            raise ValueError('duplicate finding ID')
        return self

    def bind(self, image_sha256, target):
        if self.image_sha256 != image_sha256 or any(f.target != target for f in self.findings):
            raise ValueError('observation does not match current image and object')
        return self


class ProposedOperation(Strict):
    capability: Identifier
    value: Value
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=4)
    expected_effect: str = Field(min_length=1, max_length=120)


class Operation(ProposedOperation):
    target: Identifier


class PlanProposal(Strict):
    status: Literal['ready', 'noop', 'unsupported']
    operations: list[ProposedOperation] = Field(max_length=3)
    note: str = Field(max_length=200)

    @model_validator(mode='after')
    def validate_status(self):
        if bool(self.operations) != (self.status == 'ready'):
            raise ValueError('ready requires operations and other statuses require an empty plan')
        return self

    def for_frame(self, image_sha256, target):
        return ExecutionPlan(version='execution-plan-v1', image_sha256=image_sha256,
                             status=self.status, note=self.note,
                             operations=[Operation(target=target, **operation.model_dump())
                                         for operation in self.operations])


class ExecutionPlan(PlanProposal):
    version: Literal['execution-plan-v1']
    image_sha256: Digest
    operations: list[Operation] = Field(max_length=3)


def validate_plan(plan, report, current, capabilities, *, target='object'):
    """Compile absolute values against the current image, registered capabilities and evidence."""
    if plan.image_sha256 != report.image_sha256:
        raise ValueError('stale plan image')
    evidence = {finding.id: finding for finding in report.findings}
    updates = {}
    for operation in plan.operations:
        name, value = operation.capability, operation.value
        if operation.target != target or name not in capabilities or name in updates:
            raise ValueError('unregistered, duplicate or wrong-target operation')
        capability = capabilities[name]
        causes = [evidence.get(key) for key in operation.evidence_ids]
        if any(f is None or f.target != target or f.visibility != 'visible'
               or f.code not in capability['causes'] or f.id not in capability.get('evidence_ids', [f.id])
               for f in causes):
            raise ValueError('operation lacks current visible evidence')
        if value == current.get(name):
            raise ValueError('unchanged absolute parameter')
        if 'values' in capability:
            if value not in capability['values']:
                raise ValueError('value is unavailable for this operation')
        else:
            low, high = capability['range']
            if not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError('numeric operation exceeds its range')
            direction = 'increase' if value > current.get(name, 0) else 'decrease'
            if capability.get('integer') and int(value) != value:
                raise ValueError('operation requires an integer')
            if not any(f.direction == direction for f in causes):
                raise ValueError('numeric action direction is unsupported by evidence')
        updates[name] = value
    return updates
