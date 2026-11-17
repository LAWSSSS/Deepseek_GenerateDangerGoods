"""Photographer-to-retoucher handoff on the existing model transport and cost ledger."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from .photo_evidence import contradiction, studio_reference

from .photography_protocol import (BRIEF_VERSION, DIRECTED, FACTORS, GEOMETRY, MIN_ACTION_CONFIDENCE, PHOTO_JOB_ID,
                                  PLAN_TOKENS, PLAN_VERSION, SCOPE_VERSION, VISION_TOKENS, ControlChange,
                                  PhotoBriefV2, PhotoBriefV3, PhotoChoice, PhotoContext, PhotoOption, PhotoPlan, PhotoReport,
                                  compact_schema, digest, audit_brief, merge_semantic_repair)

PROMPT_VERSION = 'professional-photography-v3'

_SEVERITY = {'minor': 0, 'major': 1, 'fatal': 2}
_STAGE_ORDER = {stage: i for i, stage in enumerate(
    ('receiver', 'lighting', 'lighting_material', 'asset', 'material', 'air', 'tone', 'composite', 'camera'))}
_PLAN_METRICS = {
    'exposure': {'local_y_255', 'object_y_255', 'object_near_black_white_fraction'},
    'tonal_contrast': {'local_y_255', 'object_y_255'},
    'warmth': {'local_neutral_ab', 'object_neutral_ab'},
    'chroma': {'local_chroma_c50_c90', 'object_chroma_c50_c90', 'material_chroma_c50_c90'},
    'haze_amount': {'fog', 'transmission_p05_p95', 'atmosphere_confidence'},
    'sharpness': {'camera_sigma_px', 'applied_sigma_px'},
    'specular_width': {'specular_share_p50_p90', 'lighting_confidence'},
    'specular_shape': {'specular_share_p50_p90', 'lighting_direction_xy', 'lighting_confidence'},
    'material_response': {'specular_share_p50_p90'},
}


class ModelClient(Protocol):
    async def ask(self, role, schema, prompt, payload, images=(), **options): ...


def _score_groups():
    by_score = {}
    for factor, score in {**GEOMETRY, **{name: rule[1] for name, rule in FACTORS.items()}}.items():
        by_score.setdefault(score, []).append(factor)
    return by_score


@lru_cache(maxsize=2)
def role_prompt(role):
    prompt = Path(__file__).with_name(f'photo_{role}.txt').read_text(encoding='utf8').strip()
    if role == 'observer':
        mapping = '\n'.join(f'{score} = {", ".join(factors)}' for score, factors in _score_groups().items())
        return f'{prompt}\n评分项对应因素\n{mapping}'
    return prompt


def detail_policy(context):
    reasons = []
    if context.first_asset_diagnosis or context.diagnosis_mode == 'asset': reasons.append('asset_diagnosis')
    if len(context.material) > 1 or len(context.asset_profile.get('parts', [])) > 1: reasons.append('mixed_material')
    if context.unsuccessful_rounds: reasons.append('previous_no_improvement')
    if context.target_region:
        l,t,r,b = context.target_region
        if (r-l)*(b-t) < .02: reasons.append('small_target')
    deep = context.detail == 'deep' or context.detail == 'auto' and bool(reasons)
    return {'level':'deep' if deep else 'standard', 'reasons':reasons,
            'visual_tokens':6144 if deep else VISION_TOKENS,
            'execution_tokens':2048 if deep else PLAN_TOKENS}


def composed_prompt(role, context):
    """Versioned static modules: scene instructions never masquerade as asset evidence."""
    modules = ['material'] if context.diagnosis_mode == 'asset' else ['material', 'scene']
    text = '\n\n'.join(Path(__file__).with_name(f'photo_knowledge_{name}.txt').read_text(encoding='utf8').strip()
                       for name in modules)
    return role_prompt(role) + '\n\n' + text


def observation_payload(context, image_sha256, target, parameters=None):
    def role(kind):
        return 'identity' if kind == 'asset' else 'diagnostic' if context.diagnosis_mode == 'asset' else 'scene'
    return {'protocol':BRIEF_VERSION, 'prompt_version':PROMPT_VERSION, 'image_sha256':image_sha256,
            'target_id':target, 'state_sha256':digest([context.model_dump(mode='json'), parameters]),
            **context.model_dump(mode='json', exclude_none=True), 'detail_policy':detail_policy(context),
            'reference_roles':[{'image':i, 'role':role(kind), 'coordinates':'first_frame_normalized', 'kind':kind}
                               for i,kind in enumerate(context.images,1)]}


@dataclass(frozen=True)
class PhotoScope:
    report: PhotoReport
    options: tuple[PhotoOption, ...]
    state_sha256: str
    reason: str | None = None
    required_measurements: tuple[str, ...] = ()

    def bind(self, choice=None):
        if choice is not None and self.reason is not None:
            raise ValueError('a stopped photography review cannot select an operation')
        operation = None
        if choice is not None and choice.option is not None:
            operation = next((o for o in self.options if o.id == choice.option), None)
            if operation is None:
                raise ValueError('option is unavailable for the current evidence')
        reason = self.reason or (choice.reason if choice else 'diagnose_only')
        return PhotoPlan(image_sha256=self.report.image_sha256,
                         report_sha256=digest(self.report.model_dump(mode='json')),
                         state_sha256=self.state_sha256,
                         status='ready' if operation else 'needs_measurement' if reason == 'measurement' else 'unsupported' if reason in
                         ('geometry', 'program_check', 'no_supported_control') else 'noop',
                         reason=reason, operation=operation,
                         needed_measurements=(list(dict.fromkeys(m for f in self.report.findings
                                                  for m in f.needed_measurements))[:4] or
                                              list(self.required_measurements) or
                                              ['material_and_light_isolation']) if reason == 'measurement' else [])


def scope_plan(report, context, changes, *, plan=True, parameters=None):
    """Keep uncertainty, clashes and upstream defects out of the execution request."""
    if report.context != context:
        raise ValueError('photography report context changed')
    # Pydantic models contain mutable dicts. Object equality cannot detect an
    # in-place readback edit shared by the report and caller; validate the whole
    # serialized report, including the stored context hash, again at execution.
    report = PhotoReport.model_validate_json(report.model_dump_json())
    state_sha256 = digest([SCOPE_VERSION, report.image_sha256, context.model_dump(mode='json'), parameters,
                           [change.model_dump(mode='json') for change in changes]])

    def stop(reason, measurement=None):
        return PhotoScope(report, (), state_sha256, reason, (measurement,) if measurement else ())

    if not plan:
        return stop('diagnose_only')
    if 'fail' in context.checks.values():
        return stop('program_check')
    if parameters and any(c.control in parameters and parameters[c.control] != c.current for c in changes):
        raise ValueError('registered control current value disagrees with current state')
    if report.issues:
        # An unresolved major finding cannot disappear behind a valid cosmetic action.
        if any(issue.severity in ('major', 'fatal') for issue in report.issues):
            return stop('unresolved_evidence')
    if report.version == BRIEF_VERSION and any(f.visibility != 'visible' and f.severity in ('major', 'fatal')
                                              for f in report.findings):
        return stop('measurement' if any(f.needed_measurements for f in report.findings) else 'unresolved_evidence')
    visible = [(i, f) for i, f in enumerate(report.findings, 1) if f.visibility == 'visible']
    if any(contradiction(f.factor, f.direction, f.cue, f.reference or '') or
           (f.factor not in ('identity', 'structure') and
            (f.reference_role == 'identity' or studio_reference(f.reference))) for _, f in visible):
        return stop('uncertain')
    if 'geometry_cue' in report.reconciled or any(f.factor in GEOMETRY for _, f in visible):
        return stop('geometry')
    if not visible:
        return stop('no_visible_defect')
    directions = {}
    for _, f in visible:
        if f.direction != 'unknown':
            directions.setdefault(f.factor, set()).add(f.direction)
    if any(len(items) > 1 for items in directions.values()):
        return stop('conflict')
    ranked = sorted(visible, key=lambda entry: (-_SEVERITY[entry[1].severity],
                    _STAGE_ORDER[FACTORS[entry[1].factor][2]], -entry[1].confidence))
    index, primary = ranked[0]
    if primary.needed_measurements:
        return stop('measurement')
    causal_control = primary.factor in ('specular_width', 'specular_shape', 'material_response', 'wear_detail')
    if report.version == BRIEF_VERSION and causal_control and primary.root_cause == 'unknown':
        return stop('measurement')
    if report.version == BRIEF_VERSION and causal_control:
        if primary.direction == 'unknown':
            return stop('measurement')
        key = 'lighting_readback' if primary.root_cause.startswith('light_') else 'material_readback'
        readback = context.measurement_evidence.get(key)
        if not readback or key not in primary.measurement_refs:
            return stop('measurement', key)
    if any(f.severity == 'fatal' and f.factor != primary.factor for _, f in visible):
        return stop('conflict')
    if 'unlocalized' in report.reconciled:
        return stop('uncertain')
    if primary.confidence < MIN_ACTION_CONFIDENCE:
        return stop('confidence')
    if primary.factor in DIRECTED and primary.direction == 'unknown':
        return stop('uncertain')
    matched = [change for change in changes if change.factor == primary.factor
               and change.direction == primary.direction and change.value != change.current
               and change.part_id == primary.part_id
               and (primary.backend == 'unknown' or change.backend in ('unknown', primary.backend))
               and (not causal_control or change.cause == primary.root_cause)]
    # One root cause per round. An unsupported key light or contact shadow stays
    # in review; exposure, desaturation and haze cannot stand in for it.
    if not matched:
        return stop('no_supported_control')
    options = tuple(PhotoOption(id=f'a{i}', evidence_id=f'f{index}', **c.model_dump(exclude={'factor', 'direction'}))
                    for i, c in enumerate(matched[:3], 1))
    return PhotoScope(report, options, state_sha256)


def compile_plan(plan, scope):
    """Compare the bound plan to the current scope before returning an absolute update."""
    if plan.operation is None:
        return {}
    expected = scope.bind(PhotoChoice(option=plan.operation.id, reason='match'))
    if plan != expected:
        raise ValueError('stale or modified photography plan')
    return {plan.operation.control: plan.operation.value}


@dataclass
class PhotoTask:
    context: PhotoContext
    image_sha256: str
    target: str
    images: tuple[str | Path, ...]
    changes: tuple[ControlChange, ...]
    plan: bool
    report: PhotoReport | None
    parameters: dict | None


@dataclass(frozen=True)
class PhotoDecision:
    report: PhotoReport
    plan: PhotoPlan
    updates: dict


@lru_cache(maxsize=1)
def _graph():
    from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext, StepContext

    @dataclass
    class Retouch(BaseNode[PhotoTask, ModelClient, PhotoDecision]):
        scope: PhotoScope

        async def run(self, ctx: GraphRunContext[PhotoTask, ModelClient]) -> End[PhotoDecision]:
            scope = self.scope
            index = int(scope.options[0].evidence_id[1:]) - 1
            finding = scope.report.findings[index]
            payload = {'protocol': PLAN_VERSION,
                       'prompt_version':PROMPT_VERSION, 'image_sha256':scope.report.image_sha256,
                       'state_sha256':scope.state_sha256,
                       'finding_id':scope.options[0].evidence_id,
                       'finding': finding.model_dump(mode='json', exclude_none=True),
                       'execution_state':scope.report.context.execution_state,
                       'measurement_evidence':scope.report.context.measurement_evidence,
                       'asset_profile':scope.report.context.asset_profile,
                       'options': [o.model_dump(mode='json') for o in scope.options]}
            measured = scope.report.context.measurements.model_dump(
                mode='json', include=_PLAN_METRICS.get(finding.factor, set()), exclude_none=True)
            if measured:
                payload['measurements'] = measured
            choice = await ctx.deps.ask('photo-execution', PhotoChoice, composed_prompt('editor', ctx.state.context), payload,
                                        plain_json=True, planning=True,
                                        max_output_tokens=detail_policy(ctx.state.context)['execution_tokens'],
                                        compact=True, validate=scope.bind)
            plan = scope.bind(choice)
            return End(PhotoDecision(scope.report, plan, compile_plan(plan, scope)))

    @dataclass
    class Inspect(BaseNode[PhotoTask, ModelClient, PhotoDecision]):
        async def run(self, ctx: GraphRunContext[PhotoTask, ModelClient]) -> Retouch | End[PhotoDecision]:
            task = ctx.state

            report = task.report
            if report is None:
                policy = detail_policy(task.context)
                payload = observation_payload(task.context, task.image_sha256, task.target, task.parameters)
                original = None
                for attempt in range(2):
                    try:
                        brief = await ctx.deps.ask('photo-vision', PhotoBriefV3, composed_prompt('observer', task.context),
                            payload, task.images, plain_json=True, compact=True, max_attempts=1,
                            max_output_tokens=policy['visual_tokens'],
                            image_limits=tuple((1600,1200) if policy['level']=='deep' or kind in
                                ('local','gray','source_pbr','material_preview','relight','turn') else (1280,960)
                                for kind in task.context.images))
                    except (ValueError, RuntimeError) as error:
                        # Service/auth failures belong to the batch pause path, not to semantic repair.
                        if error.__class__.__name__ == 'ProviderUnavailable':
                            raise
                        if attempt:
                            if original is not None:
                                brief = original
                                break
                            raise
                        policy = {**policy, 'level':'deep', 'visual_tokens':6144,
                                  'reasons':[*policy['reasons'], 'invalid_or_truncated_response']}
                        payload = {**payload, 'format_correction':'Previous response failed the typed schema; return only valid JSON, retaining all evidence.'}
                        continue
                    if not isinstance(brief, PhotoBriefV3):
                        break  # Read legacy/fake-client responses without inventing v3 evidence.
                    _, issues = audit_brief(brief, task.context)
                    if attempt and original is not None:
                        brief = merge_semantic_repair(original, brief, task.context)
                    if not issues or attempt:
                        break
                    original = brief
                    payload = {**payload, 'semantic_repair':{
                        'instruction':'Repair only the listed findings; preserve other observations. If unsupported, mark unassessable/unknown. Do not drop severe unknowns.',
                        'issues':[issue.model_dump() for issue in issues],
                        'previous_response':brief.model_dump(mode='json')}}
                report = PhotoReport.bind(brief, task.image_sha256, task.target, task.context)
            elif report.image_sha256 != task.image_sha256 or report.target != task.target:
                raise ValueError('photography report does not match the current image and object')
            scope = scope_plan(report, task.context, task.changes, plan=task.plan, parameters=task.parameters)
            return End(PhotoDecision(report, scope.bind(), {})) if scope.reason else Retouch(scope)

    builder = GraphBuilder(name='documentary_photography', state_type=PhotoTask, deps_type=ModelClient,
                           output_type=PhotoDecision, auto_instrument=False)

    @builder.step
    async def start(ctx: StepContext[PhotoTask, ModelClient, None]) -> Inspect:
        return Inspect()

    builder.add(builder.edge_from(builder.start_node).to(start), builder.node(Inspect), builder.node(Retouch))
    return builder.build()


async def run_photography(client, context, image_sha256, *, images=(), target='object', changes=(),
                          plan=False, report=None, parameters=None):
    """Run one visual diagnosis and, when requested and applicable, one retouch decision."""
    if report is None and len(images) != len(context.images):
        raise ValueError('photography input image roles do not match the supplied images')
    task = PhotoTask(context, image_sha256, target, tuple(images), tuple(changes), plan, report, parameters)
    job = PHOTO_JOB_ID.set(asyncio.current_task().get_name())
    try:
        return await _graph().run(state=task, deps=client)
    finally:
        PHOTO_JOB_ID.reset(job)


def interface():
    return {'profile': 'photographer', 'framework': 'pydantic-graph',
            'versions': {'diagnosis': BRIEF_VERSION, 'plan': PLAN_VERSION, 'scope': SCOPE_VERSION},
            'output_token_limits': {'visual': VISION_TOKENS, 'execution': PLAN_TOKENS},
            'adaptive_limits': {'deep_visual':6144, 'deep_execution':2048}, 'prompt_version':PROMPT_VERSION,
            'context_schema': PhotoContext.model_json_schema(), 'diagnosis_schema': PhotoReport.model_json_schema(),
            'plan_schema': PhotoPlan.model_json_schema(),
            'provider_responses': {'visual': compact_schema(PhotoBriefV3), 'execution': compact_schema(PhotoChoice)},
            'factors': {name: {'score': score, 'stage': stage, 'cue': meaning}
                        for name, (_, score, stage, meaning) in FACTORS.items()}}
