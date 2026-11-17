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

BRIEF_VERSION = 'photo-brief-v3'
PLAN_VERSION = 'photo-plan-v2'
SCOPE_VERSION = 'photo-options-v2'
MIN_ACTION_CONFIDENCE = .7
VISION_TOKENS = 3072
PLAN_TOKENS = 1024
PHOTO_JOB_ID = ContextVar('photo_job_id', default=None)

# The order expresses photographic dependency. A lower priority correction must
# not cover a more serious problem in the lighting, material, or receiver.
FACTORS = {
    'contact_shadow': ('shadow', 'lighting', 'receiver', '接触暗部的位置、强度或软硬与接触斑不符'),
    'key_direction': ('light_direction', 'lighting', 'lighting', '同层投影与亮面的主光方向不一致'),
    'light_ratio': ('lighting', 'lighting', 'lighting', '同朝向表面的主补光相对环境光的能量比例不一致'),
    'key_fill_ratio': ('lighting', 'lighting', 'lighting', '主光与补光的能量比例偏差；增大比例意味着相对减少补光'),
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
DIRECTED = {'light_ratio', 'key_fill_ratio', 'dust_coverage', 'haze_amount', 'exposure', 'tonal_contrast', 'warmth', 'chroma', 'sharpness'}
MAX_FINDINGS = 8
Factor = Literal[tuple((*FACTORS, *GEOMETRY))]
Direction = Literal['increase', 'decrease', 'unknown']
Region = tuple[Unit, Unit, Unit, Unit]
ShortText = Annotated[str, Field(min_length=1, max_length=160)]
DetailText = Annotated[str, Field(max_length=320)]
PhotoReferenceRole = Literal['scene', 'identity', 'diagnostic', 'unknown']
Backend = Literal['photoshop', 'composite', 'blender', 'geometry', 'unknown']
Cause = Literal['geometry', 'surface_roughness', 'surface_normals', 'oxide_distribution',
                'scratch_distribution', 'dust_distribution', 'light_size', 'light_direction',
                'light_energy', 'atmosphere', 'tone', 'camera', 'composite', 'unknown']
OptionID = Annotated[str, Field(pattern=r'^a[1-3]$')]
EvidenceID = Annotated[str, Field(pattern=r'^f[1-8]$')]
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
    reference_role: PhotoReferenceRole | None = None
    reference_region: Region | None = None
    part_id: Annotated[str, Field(min_length=1, max_length=80)] = 'whole'
    hypothesis: DetailText = ''
    counterevidence: DetailText = ''
    verification: DetailText = ''
    backend: Backend = 'unknown'
    root_cause: Cause = 'unknown'
    measurement_refs: list[str] = Field(default_factory=list, max_length=6)
    needed_measurements: list[str] = Field(default_factory=list, max_length=4)
    reference_image_id: int | None = Field(default=None, ge=1, le=4)
    change_vs_reference: Literal['unknown', 'same', 'improved', 'regressed', 'different'] = 'unknown'

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


class FindingIssue(Strict):
    evidence_id: EvidenceID
    factor: Factor
    severity: Literal['minor', 'major', 'fatal']
    reason: str = Field(max_length=500)


def finding_issue(finding):
    """Validate one v3 observation without discarding its valid siblings."""
    if finding.reference_role == 'diagnostic':
        if finding.factor in GEOMETRY and finding.factor != 'structure':
            return 'asset previews cannot prove scene contact, scale or occlusion'
        if finding.visibility == 'visible':
            if finding.region in (None, _FRAME_REGION) or not finding.reference or not finding.hypothesis:
                return 'diagnostic finding needs a local fact, reference and causal hypothesis'
            desired = {'too_high':'decrease', 'too_low':'increase', 'mismatch':'unknown', 'unknown':'unknown'}
            if finding.observed_state not in desired or desired[finding.observed_state] != finding.direction:
                return 'direction disagrees with observed_state'
            if finding.observed_state == 'unknown' or finding.physical_cue is not None:
                return 'diagnostic observation cannot assert scene physical contact'
            return contradiction(finding.factor, finding.direction, finding.cue, finding.reference)
        return None
    try:
        PhotoFindingV2.model_validate(finding.model_dump())
    except ValueError as error:
        return str(error).split('For further information')[0][-480:]
    if finding.visibility == 'visible' and not finding.hypothesis:
        return 'visible finding needs a causal hypothesis separate from the observed fact'
    return None


class PhotoBriefV3(PhotoBrief):
    """Wire types remain strict; semantic faults are isolated per finding by the host."""


class PixelComparison(Strict):
    """Host measurements, never values inferred from the model's descriptions."""
    version: Literal['pixel-comparison-v1'] = 'pixel-comparison-v1'
    candidate_image_id: int = Field(ge=1, le=4)
    reference_image_id: int = Field(ge=1, le=4)
    candidate_sha256: Digest
    reference_sha256: Digest
    same_geometry: bool
    same_lighting: bool
    alpha_identical: bool
    compared_entire_object: bool
    max_abs_rgba_255: float = Field(ge=0, le=255)
    changed_pixel_fraction: Unit
    core_mae_255: float = Field(ge=0, le=255)


def near_identical_comparison(context, reference_image_id):
    """A narrow comparative guard, not a quality or defect detector."""
    hashes = context.execution_state.get('input_image_hashes', [])
    if len(hashes) != len(context.images) or reference_image_id is None:
        return None
    records = context.measurement_evidence.get('comparisons', [])
    if not isinstance(records, list):
        return None
    for record in records:
        try:
            measured = PixelComparison.model_validate(record)
        except (ValueError, TypeError):
            continue
        if (measured.candidate_image_id != 1 or measured.reference_image_id != reference_image_id or
                reference_image_id == 1 or reference_image_id > len(hashes)):
            continue
        if (measured.candidate_sha256 != hashes[0] or
                measured.reference_sha256 != hashes[reference_image_id-1]):
            continue
        if (measured.same_geometry and measured.same_lighting and measured.alpha_identical and
                measured.compared_entire_object and measured.max_abs_rgba_255 <= 1. and
                measured.changed_pixel_fraction <= .001 and measured.core_mae_255 <= .01):
            return measured
    return None


def comparison_issue(finding, context):
    if finding.change_vs_reference == 'unknown':
        return None  # Absolute defects remain assessable even if both images are identical.
    if finding.reference_image_id is None or finding.reference_image_id > len(context.images):
        return 'relative finding needs a supplied reference_image_id'
    if finding.change_vs_reference in ('improved', 'regressed', 'different') and near_identical_comparison(
            context, finding.reference_image_id) is not None:
        return 'relative_change_not_supported_by_near_identical_full_rgba; absolute quality remains unknown'
    return None


def audit_brief(brief, context):
    findings, issues = [], []
    for index, finding in enumerate(brief.findings, 1):
        issue = finding_issue(finding)
        issue = comparison_issue(finding, context) or issue
        if finding.visibility == 'visible':
            if finding.region is None or not in_target_neighbourhood(finding.region, context.target_region):
                issue = 'finding does not belong to the current target'
            elif finding.reference_role == 'identity' and 'asset' not in context.images:
                issue = 'identity reference was not supplied'
            elif finding.reference_role == 'diagnostic' and context.diagnosis_mode != 'asset':
                issue = 'diagnostic reference requires asset diagnostic mode'
            elif finding.measurement_refs and not set(finding.measurement_refs).issubset(
                    set(context.measurement_evidence) | set(context.measurements.model_dump(exclude_none=True))):
                issue = 'measurement reference was not supplied'
            parts = context.asset_profile.get('parts', [])
            known_parts = {p.get('id', p.get('part_id')) for p in parts if isinstance(p, dict)}
            if finding.part_id != 'whole' and finding.part_id not in known_parts:
                issue = 'part ID has no supplied asset evidence or registered mask'
        if issue:
            issues.append(FindingIssue(evidence_id=f'f{index}', factor=finding.factor,
                                       severity=finding.severity, reason=issue))
            finding = finding.model_copy(update={'visibility':'unassessable', 'observed_state':'unknown',
                'direction':'unknown', 'physical_cue':None})
        findings.append(finding)
    return brief.model_copy(update={'findings':findings}), issues


def merge_semantic_repair(original, repaired, context):
    """Only listed rows may change; missing or reordered repairs stay unknown."""
    _, issues = audit_brief(original, context)
    invalid = {int(issue.evidence_id[1:])-1 for issue in issues}
    findings = list(original.findings)
    for index in invalid:
        if index >= len(repaired.findings):
            continue
        old, new = original.findings[index], repaired.findings[index]
        if (old.factor, old.part_id) == (new.factor, new.part_id):
            _, remaining = audit_brief(PhotoBriefV3(trace=repaired.trace, scores=repaired.scores,
                                                   findings=[new]), context)
            if new.visibility != 'visible' or remaining:
                rank = {'minor':0, 'major':1, 'fatal':2}
                severity = max((old.severity, new.severity), key=rank.get)
                new = new.model_copy(update={'severity':severity})
            findings[index] = new
    return original.model_copy(update={'findings':findings})


def validate_brief_context(brief, context):
    for finding in brief.findings:
        if finding.visibility != 'visible':
            continue
        if finding.reference_role == 'identity' and 'asset' not in context.images:
            raise ValueError('identity reference was not supplied')
        if not in_target_neighbourhood(finding.region, context.target_region):
            raise ValueError('finding does not belong to current target')


class PhotoReport(PhotoBrief):
    version: Literal['photo-brief-v1', 'photo-brief-v2', 'photo-brief-v3'] = 'photo-brief-v1'
    image_sha256: Digest
    target: Annotated[str, Field(pattern=r'^[^\x00-\x1f]{1,80}$')]
    context: PhotoContext
    reconciled: list[Literal['geometry_cue', 'region', 'score', 'trace', 'unlocalized']] = Field(default_factory=list, max_length=5)
    issues: list[FindingIssue] = Field(default_factory=list, max_length=8)
    raw_findings: list[PhotoFinding] = Field(default_factory=list, max_length=8)
    state_sha256: Digest | None = None

    @classmethod
    def bind(cls, brief, image_sha256, target, context):
        """Resolve contradictory visual scores conservatively, without another API call."""
        issues = []
        raw_findings = list(brief.findings) if isinstance(brief, PhotoBriefV3) else []
        if isinstance(brief, PhotoBriefV3):
            brief, issues = audit_brief(brief, context)
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
            elif isinstance(brief, PhotoBriefV3) and finding.severity in ('major', 'fatal'):
                scores[finding.score_key] = min(scores[finding.score_key], 2)
        trace = brief.trace
        if any(f.visibility == 'visible' for f in findings):
            trace = 'visible'
        elif trace == 'visible' or findings or min(scores.values()) < 3:
            trace = 'uncertain'
        if trace != brief.trace:
            corrections.add('trace')
        return cls(version=BRIEF_VERSION if isinstance(brief, PhotoBriefV3) else
                   'photo-brief-v2' if isinstance(brief, PhotoBriefV2) else 'photo-brief-v1',
                   image_sha256=image_sha256, target=target, context=context, trace=trace,
                   scores=Scores(**scores), findings=findings, reconciled=sorted(corrections), issues=issues,
                   raw_findings=raw_findings,
                   state_sha256=digest(context.model_dump(mode='json')))

    @model_validator(mode='after')
    def coherent_report(self):
        recorded_image = self.context.execution_state.get('image_sha256')
        if recorded_image is not None and recorded_image != self.image_sha256:
            raise ValueError('Photoshop execution state belongs to another image')
        image_hashes = self.context.execution_state.get('input_image_hashes')
        if image_hashes and image_hashes[0] != self.image_sha256:
            raise ValueError('ordered input hashes belong to another image')
        if self.version == 'photo-brief-v2':
            typed = PhotoBriefV2.model_validate(self.model_dump(include={'trace','scores','findings'}))
            validate_brief_context(typed, self.context)
        elif self.version == BRIEF_VERSION:
            if self.state_sha256 != digest(self.context.model_dump(mode='json')):
                raise ValueError('diagnosis context/state hash changed')
            _, problems = audit_brief(PhotoBriefV3(trace=self.trace, scores=self.scores, findings=self.findings), self.context)
            if any(f.visibility == 'visible' and f'f{i}' in {p.evidence_id for p in problems}
                   for i, f in enumerate(self.findings, 1)):
                raise ValueError('visible v3 finding has invalid evidence')
            ranks = {'minor':0,'major':1,'fatal':2}
            for raw, finding in zip(self.raw_findings, self.findings):
                if ((raw.factor, raw.part_id) == (finding.factor, finding.part_id) and
                        finding.visibility != 'visible' and ranks[finding.severity] < ranks[raw.severity]):
                    raise ValueError('unresolved finding severity was downgraded after diagnosis')
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
                    observed_state=f.observed_state, reference_role='unknown' if f.reference_role == 'diagnostic' else f.reference_role,
                    reference_region=f.reference_region,
                    evidence=legacy_evidence(f))
            for index, f in enumerate(self.findings, 1)])


def legacy_evidence(finding):
    text = '。'.join(filter(None, (finding.cue, finding.reference)))
    return text if len(text) <= 240 else text[:205] + '…完整事实与参照见同图片绑定的摄影报告。'


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
    unit: str = Field(default='preset', max_length=48)
    backend: Backend = 'unknown'
    cause: Cause = 'unknown'
    part_id: str = Field(default='whole', max_length=80)


class ControlChange(Strict):
    factor: Factor
    direction: Literal['increase', 'decrease']
    control: Annotated[str, Field(pattern=r'^[a-z_]{1,32}$')]
    current: str | int | float
    value: str | int | float
    effect: ShortText
    stage: Literal['composite', 'render', 'photoshop']
    unit: str = Field(default='preset', max_length=48)
    backend: Backend = 'unknown'
    cause: Cause = 'unknown'
    part_id: str = Field(default='whole', max_length=80)


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
    images: list[Literal['frame', 'local', 'asset', 'context', 'gray', 'source_pbr', 'material_preview', 'relight', 'turn']] = Field(min_length=1, max_length=4)
    material: list[Annotated[str, Field(max_length=24)]] = Field(default_factory=list, max_length=6)
    features: list[ShortText] = Field(default_factory=list, max_length=12)
    intended_pose: Literal['lying_on_side'] | None = None
    axis_deg: float | None = None
    checks: dict[str, Literal['pass', 'fail', 'unknown']] = Field(default_factory=dict)
    measurements: PhotoMeasurements = Field(default_factory=PhotoMeasurements)
    question: Annotated[str, Field(max_length=6000)] | None = None
    diagnosis_mode: Literal['scene', 'asset'] = 'scene'
    detail: Literal['auto', 'standard', 'deep'] = 'auto'
    first_asset_diagnosis: bool = False
    unsuccessful_rounds: int = Field(default=0, ge=0, le=10)
    asset_profile: dict = Field(default_factory=dict)
    execution_state: dict = Field(default_factory=dict)
    measurement_evidence: dict = Field(default_factory=dict)

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
    version: Literal['photo-plan-v1', 'photo-plan-v2'] = PLAN_VERSION
    image_sha256: Digest
    report_sha256: Digest
    state_sha256: Digest
    status: Literal['ready', 'noop', 'needs_measurement', 'unsupported']
    reason: Literal['match', 'uncertain', 'tradeoff', 'diagnose_only', 'geometry', 'program_check',
                    'no_visible_defect', 'confidence', 'conflict', 'no_supported_control', 'measurement', 'unresolved_evidence']
    operation: PhotoOption | None = None
    needed_measurements: list[str] = Field(default_factory=list, max_length=4)

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
