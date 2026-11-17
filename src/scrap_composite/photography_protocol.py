"""Short, typed observations for documentary compositing and retouching."""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contracts import (DefectCode, Digest, Finding, PHYSICAL_CUES, PhysicalCue, ObservedState, ReferenceRole,
                        Scores, Strict, Unit, VisualReport)
from .photo_evidence import contradiction, identity_reference, physical_evidence_issue, studio_reference

BRIEF_VERSION = 'photo-brief-v2'
PLAN_VERSION = 'photo-plan-v1'
SCOPE_VERSION = 'photo-options-v1'
MIN_ACTION_CONFIDENCE = .7
VISION_TOKENS = 1536
PLAN_TOKENS = 256
PHOTO_JOB_ID = ContextVar('photo_job_id', default=None)

# The order expresses photographic dependency. A lower priority correction must
# not cover a more serious problem in the lighting, material, or receiver.
FACTORS = {
    'contact_shadow': ('shadow', 'lighting', 'receiver', '接触暗部的位置、强度或软硬与接触斑不符'),
    'key_direction': ('light_direction', 'lighting', 'lighting', '同层投影与亮面的主光方向不一致'),
    'light_ratio': ('lighting', 'lighting', 'lighting', '同朝向表面的主补光相对环境光的能量比例不一致'),
    'specular_width': ('material', 'material', 'lighting_material', '反光带的宽度或边界与同类金属不符'),
    'specular_shape': ('material', 'material', 'lighting_material', '高光未随曲面、视角或周围遮蔽变化'),
    'dust_coverage': ('surface_dust_heavy', 'material', 'material', '物体表面的附着浮灰分布偏差'),
    'wear_detail': ('texture', 'texture', 'material', '原有锈斑、磨损与划痕的层次或尺度不符'),
    'material_response': ('material', 'material', 'material', '同类表面的漫反射与镜面反射关系不符'),
    'haze_amount': ('haze_clear', 'lighting', 'air', '同层暗部与对比共同显示空气遮蔽偏差'),
    'haze_tint': ('haze_color', 'lighting', 'air', '同层中性暗部的雾幕颜色偏差'),
    'exposure': ('lighting', 'lighting', 'tone', '同类受光表面整体中间调偏亮或偏暗'),
    'black_point': ('lighting', 'lighting', 'tone', '非高光暗部的黑位与同层材质不一致'),
    'highlight_clip': ('lighting', 'lighting', 'tone', '非镜面亮部层次被截断或压平'),
    'tonal_contrast': ('lighting', 'lighting', 'tone', '排除高光和孔洞后中尺度明暗跨度偏差'),
    'warmth': ('color', 'lighting', 'tone', '同类中性漫反射区域的冷暖偏差'),
    'tint': ('color', 'lighting', 'tone', '同类中性漫反射区域的绿洋红偏差'),
    'chroma': ('color', 'lighting', 'tone', '同类材质区域的色度偏差'),
    'edge_transition': ('edge', 'edge', 'composite', '轮廓有色边、白边或混合过渡不自然'),
    'patch_seam': ('patch_seam', 'edge', 'composite', '阴影接收面或编辑区边界出现接缝'),
    'sharpness': ('too_sharp', 'texture', 'camera', '成图像素上同层同类边缘的清晰度偏差'),
    'sensor_noise': ('texture', 'texture', 'camera', '平坦区域的相机噪声幅度不一致'),
    'compression': ('texture', 'texture', 'camera', '压缩方块或振铃只出现在一侧'),
    'identity': ('identity_loss', 'material', 'asset', '原色、标识或表面识别细节被覆盖'),
}
GEOMETRY = {'upright': 'support', 'floating': 'support', 'penetration': 'support',
            'occlusion': 'occlusion', 'scale': 'support', 'camera': 'support', 'structure': 'structure'}
DIRECTED = {'light_ratio', 'dust_coverage', 'haze_amount', 'exposure', 'tonal_contrast', 'warmth', 'chroma', 'sharpness'}
MAX_FINDINGS = 5
Factor = Literal[tuple((*FACTORS, *GEOMETRY))]
Direction = Literal['increase', 'decrease', 'unknown']
Region = tuple[Unit, Unit, Unit, Unit]
ShortText = Annotated[str, Field(min_length=1, max_length=42)]
OptionID = Annotated[str, Field(pattern=r'^a[1-3]$')]
EvidenceID = Annotated[str, Field(pattern=r'^f[1-5]$')]
_FRAME_REGION = (0., 0., 1., 1.)

_GAP_WORDS = re.compile(r'缝隙|间隙|空隙|空档|透空|透出背景|visible gap|air gap|clear air below|daylight', re.IGNORECASE)
_SHADOW_WORDS = re.compile(r'阴影|投影|压痕|暗部|shadow', re.IGNORECASE)
_NEGATED_GAP = re.compile(r'(?:无|没有|未见|看不到)(?:明显|可见|任何)?(?:的)?(?:空气)?(?:缝隙|间隙|空隙|空档)'
                          r'|no (?:visible )?(?:air )?gap', re.IGNORECASE)


def in_target_neighbourhood(region, target_region):
    if target_region is None:
        return True
    left, top, right, bottom = target_region
    pad_x, pad_y = (right-left)*.25, (bottom-top)*.25
    x, y, r, b = region
    return min(r, right+pad_x) > max(x, left-pad_x) and min(b, bottom+pad_y) > max(y, top-pad_y)


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf8')).hexdigest()


class PhotoFinding(Strict):
    factor: Factor
    region: Region | None
    severity: Literal['minor', 'major', 'fatal']
    visibility: Literal['visible', 'occluded', 'unassessable']
    confidence: Unit
    direction: Direction
    cue: ShortText
    reference: ShortText | None
    physical_cue: PhysicalCue | None = None
    observed_state: ObservedState | None = None
    reference_role: ReferenceRole | None = None
    reference_region: Region | None = None

    @model_validator(mode='after')
    def valid_cue(self):
        if self.region is not None:
            left, top, right, bottom = self.region
            if left >= right or top >= bottom:
                raise ValueError('empty finding region')
        if not self.cue.strip():
            raise ValueError('missing observed cue')
        if self.reference_region is not None:
            l, t, r, b = self.reference_region
            if l >= r or t >= b:
                raise ValueError('empty reference region')
        return self

    @property
    def code(self) -> DefectCode:
        if self.factor in GEOMETRY:
            return self.factor
        if self.direction == 'decrease':
            return {'haze_amount': 'haze_heavy', 'sharpness': 'too_sharp'}.get(
                self.factor, FACTORS[self.factor][0])
        if self.direction == 'increase':
            return {'haze_amount': 'haze_clear', 'sharpness': 'too_blurry',
                    'dust_coverage': 'surface_clean'}.get(self.factor, FACTORS[self.factor][0])
        return {'haze_amount': 'lighting', 'sharpness': 'texture', 'dust_coverage': 'material'}.get(
            self.factor, FACTORS[self.factor][0])

    @property
    def score_key(self):
        return GEOMETRY[self.factor] if self.factor in GEOMETRY else FACTORS[self.factor][1]


class PhotoBrief(Strict):
    """Provider reply. Identity and finding IDs are assigned by the application."""
    trace: Literal['none', 'uncertain', 'visible']
    scores: Scores
    findings: list[PhotoFinding] = Field(max_length=MAX_FINDINGS)


class PhotoFindingV2(PhotoFinding):
    """New provider output; PhotoFinding still reads historical replies."""
    observed_state: ObservedState
    reference_role: ReferenceRole
    reference_region: Region | None

    @model_validator(mode='after')
    def valid_semantics(self):
        desired = {'too_high':'decrease', 'too_low':'increase', 'mismatch':'unknown', 'unknown':'unknown'}
        if self.direction != desired[self.observed_state]:
            raise ValueError('direction must oppose observed_state; mismatch/unknown require unknown')
        if self.visibility != 'visible':
            if self.observed_state != 'unknown' or self.physical_cue is not None:
                raise ValueError('unassessable findings cannot claim a state or physical cue')
            return self
        if self.observed_state == 'unknown' or self.reference_role == 'unknown':
            raise ValueError('unknown evidence must be unassessable, not visible')
        if self.region in (None, _FRAME_REGION) or not self.reference:
            raise ValueError('visible findings need local current-frame evidence and a reference')
        if self.factor in DIRECTED and self.observed_state not in ('too_high', 'too_low'):
            raise ValueError('numeric photographic factors require too_high or too_low')
        if self.reference_role == 'identity':
            if self.factor not in ('identity', 'structure'):
                raise ValueError('identity reference cannot judge scene photography, pose or occlusion')
        elif self.reference_region in (None, _FRAME_REGION):
            raise ValueError('scene comparison needs localized reference_region in first image')
        elif identity_reference(self.reference) or studio_reference(self.reference):
            raise ValueError('scene comparison cannot use source/studio reference')
        issue = contradiction(self.factor, self.direction, self.cue, self.reference)
        if issue:
            raise ValueError(issue)
        expected = PHYSICAL_CUES.get(self.factor)
        if self.physical_cue != expected:
            raise ValueError('physical_cue must match visible geometry; photography requires null')
        issue = physical_evidence_issue(self.factor, self.cue, self.reference,
                                        self.physical_cue, self.reference_role)
        if issue:
            raise ValueError(issue)
        return self


class PhotoBriefV2(PhotoBrief):
    findings: list[PhotoFindingV2] = Field(max_length=MAX_FINDINGS)


def validate_brief_context(brief, context):
    for finding in brief.findings:
        if finding.visibility != 'visible':
            continue
        if finding.reference_role == 'identity' and 'asset' not in context.images:
            raise ValueError('identity reference was not supplied')
        if not in_target_neighbourhood(finding.region, context.target_region):
            raise ValueError('finding does not belong to current target')


class PhotoReport(PhotoBrief):
    version: Literal['photo-brief-v1', 'photo-brief-v2'] = 'photo-brief-v1'
    image_sha256: Digest
    target: Annotated[str, Field(pattern=r'^[^\x00-\x1f]{1,80}$')]
    context: PhotoContext
    reconciled: list[Literal['geometry_cue', 'region', 'score', 'trace', 'unlocalized']] = Field(default_factory=list, max_length=5)

    @classmethod
    def bind(cls, brief, image_sha256, target, context):
        """Resolve contradictory visual scores conservatively, without another API call."""
        scores = brief.scores.model_dump()
        findings = []
        corrections = set()
        for finding in brief.findings:
            finding, repairs = reconcile_finding(finding, context.target_region)
            corrections.update(repairs)
            findings.append(finding)
            if finding.visibility == 'visible':
                limit = 3 if finding.severity == 'minor' else 2
                if scores[finding.score_key] > limit:
                    scores[finding.score_key] = limit
                    corrections.add('score')
        trace = brief.trace
        if any(f.visibility == 'visible' for f in findings):
            trace = 'visible'
        elif trace == 'visible' or findings or min(scores.values()) < 3:
            trace = 'uncertain'
        if trace != brief.trace:
            corrections.add('trace')
        return cls(version=BRIEF_VERSION if isinstance(brief, PhotoBriefV2) else 'photo-brief-v1',
                   image_sha256=image_sha256, target=target, context=context, trace=trace,
                   scores=Scores(**scores), findings=findings, reconciled=sorted(corrections))

    @model_validator(mode='after')
    def coherent_report(self):
        if self.version == BRIEF_VERSION:
            typed = PhotoBriefV2.model_validate(self.model_dump(include={'trace','scores','findings'}))
            validate_brief_context(typed, self.context)
        visible = [finding for finding in self.findings if finding.visibility == 'visible']
        if bool(visible) != (self.trace == 'visible') or self.trace == 'none' and (
                self.findings or any(score < 3 for _, score in self.scores)):
            raise ValueError('diagnosis trace disagrees with its evidence')
        for finding in self.findings:
            canonical, _ = reconcile_finding(finding, self.context.target_region)
            if finding != canonical:
                raise ValueError('diagnosis finding lacks valid current-frame evidence')
            if finding.visibility == 'visible' and getattr(self.scores, finding.score_key) > (3 if finding.severity == 'minor' else 2):
                raise ValueError('diagnosis scores disagree with severity')
        return self

    def legacy(self):
        """Keep existing ranking, geometry and Photoshop validators on the same evidence."""
        return VisualReport(version='visual-evidence-v1', image_sha256=self.image_sha256,
                            trace=self.trace, scores=self.scores, summary='', findings=[
            Finding(id=f'f{index}', target='object', code=f.code, region=f.region,
                    severity=f.severity, visibility=f.visibility, scope='image', confidence=f.confidence,
                    direction=f.direction, physical_cue=f.physical_cue,
                    observed_state=f.observed_state, reference_role=f.reference_role,
                    reference_region=f.reference_region,
                    evidence='。'.join(filter(None, (f.cue, f.reference))))
            for index, f in enumerate(self.findings, 1)])


def reconcile_finding(finding, target_region):
    """Keep questionable regions and geometry claims as unassessable observations."""
    if finding.visibility != 'visible':
        return finding.model_copy(update={'direction': 'unknown', 'physical_cue': None}), set()
    geometry = finding.factor in GEOMETRY
    region = finding.region
    repairs = set()
    if region == _FRAME_REGION and not geometry and target_region not in (None, _FRAME_REGION):
        region = target_region
        repairs.add('region')
    if (region is None or region == _FRAME_REGION or not finding.reference or not finding.reference.strip()
            or not in_target_neighbourhood(region, target_region)):
        repairs.add('unlocalized')
        if geometry:
            repairs.add('geometry_cue')
        return finding.model_copy(update={'visibility': 'unassessable', 'direction': 'unknown',
                                         'physical_cue': None}), repairs
    direct_cue = PHYSICAL_CUES.get(finding.factor)
    weak_shadow_as_gap = finding.factor == 'floating' and (
        _NEGATED_GAP.search(finding.cue) or _SHADOW_WORDS.search(finding.cue) and not _GAP_WORDS.search(finding.cue))
    if direct_cue and (finding.physical_cue != direct_cue or weak_shadow_as_gap):
        return finding.model_copy(update={'visibility': 'unassessable', 'direction': 'unknown',
                                         'physical_cue': None}), {'geometry_cue'}
    return finding.model_copy(update={'region': region, 'physical_cue': direct_cue}), repairs


class PhotoOption(Strict):
    id: OptionID
    control: Annotated[str, Field(pattern=r'^[a-z_]{1,32}$')]
    current: str | int | float
    value: str | int | float
    effect: ShortText
    stage: Literal['composite', 'render', 'photoshop']
    evidence_id: EvidenceID


class ControlChange(Strict):
    factor: Factor
    direction: Literal['increase', 'decrease']
    control: Annotated[str, Field(pattern=r'^[a-z_]{1,32}$')]
    current: str | int | float
    value: str | int | float
    effect: ShortText
    stage: Literal['composite', 'render', 'photoshop']


class PhotoMeasurements(Strict):
    samples: int = Field(default=0, ge=0)
    reference: Literal['same_support', 'same_depth', 'under_object', 'unknown'] = 'unknown'
    local_y_255: tuple[float, float, float] | None = None
    object_y_255: tuple[float, float, float] | None = None
    local_chroma_c50_c90: tuple[float, float] | None = None
    object_chroma_c50_c90: tuple[float, float] | None = None
    material_chroma_c50_c90: dict[Literal['paint', 'metal'], tuple[float, float]] = Field(default_factory=dict)
    local_neutral_ab: tuple[float, float] | None = None
    object_neutral_ab: tuple[float, float] | None = None
    object_near_black_white_fraction: tuple[Unit, Unit] | None = None
    local_noise_255: float | None = None
    added_noise_255: float | None = None
    camera_sigma_px: float | None = None
    applied_sigma_px: float | None = None
    fog: Unit | None = None
    transmission_p05_p95: tuple[Unit, Unit] | None = None
    atmosphere_confidence: Literal['estimated', 'unknown'] = 'unknown'
    visible_shadow_pixels: int | None = Field(default=None, ge=0)
    lighting_direction_xy: tuple[float, float] | None = None
    lighting_confidence: Unit | None = None
    specular_share_p50_p90: tuple[Unit, Unit] | None = None


class PhotoContext(Strict):
    name: Annotated[str, Field(max_length=80)] = ''
    target_region: Region | None = None
    images: list[Literal['frame', 'local', 'asset', 'context']] = Field(min_length=1, max_length=4)
    material: list[Annotated[str, Field(max_length=24)]] = Field(default_factory=list, max_length=6)
    features: list[ShortText] = Field(default_factory=list, max_length=4)
    intended_pose: Literal['lying_on_side'] | None = None
    axis_deg: float | None = None
    checks: dict[str, Literal['pass', 'fail', 'unknown']] = Field(default_factory=dict)
    measurements: PhotoMeasurements = Field(default_factory=PhotoMeasurements)
    question: Annotated[str, Field(max_length=240)] | None = None

    @model_validator(mode='after')
    def valid_frame(self):
        if self.images[0] != 'frame' or 'frame' in self.images[1:]:
            raise ValueError('the first image must be the current full frame')
        if self.target_region is not None:
            left, top, right, bottom = self.target_region
            if left >= right or top >= bottom:
                raise ValueError('empty target region')
        return self


class PhotoChoice(Strict):
    """The execution model selects one complete operation from a small menu."""
    option: OptionID | None
    reason: Literal['match', 'uncertain', 'tradeoff']

    @model_validator(mode='after')
    def consistent_choice(self):
        if bool(self.option) != (self.reason == 'match'):
            raise ValueError('choice and reason disagree')
        return self


class PhotoPlan(Strict):
    version: Literal['photo-plan-v1'] = PLAN_VERSION
    image_sha256: Digest
    report_sha256: Digest
    state_sha256: Digest
    status: Literal['ready', 'noop', 'unsupported']
    reason: Literal['match', 'uncertain', 'tradeoff', 'diagnose_only', 'geometry', 'program_check',
                    'no_visible_defect', 'confidence', 'conflict', 'no_supported_control']
    operation: PhotoOption | None = None

    @model_validator(mode='after')
    def consistent_plan(self):
        if bool(self.operation) != (self.status == 'ready') or (self.status == 'ready') != (self.reason == 'match'):
            raise ValueError('plan status, reason and operation disagree')
        return self


def compact_schema(model):
    """Omit annotation-only JSON Schema keys from the prompt, retaining validation locally."""
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {key: ({name: visit(schema) for name, schema in item.items()}
                      if key in ('properties', '$defs') else visit(item))
                for key, item in value.items() if key not in ('title', 'description', 'default')}

    return visit(model.model_json_schema())
