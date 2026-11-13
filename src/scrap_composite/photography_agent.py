"""Photographer-to-retoucher handoff on the existing model transport and cost ledger."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .photography_protocol import (BRIEF_VERSION, DIRECTED, FACTORS, GEOMETRY, MIN_ACTION_CONFIDENCE, PHOTO_JOB_ID,
                                  PLAN_TOKENS, PLAN_VERSION, SCOPE_VERSION, VISION_TOKENS, ControlChange,
                                  PhotoBrief, PhotoChoice, PhotoContext, PhotoOption, PhotoPlan, PhotoReport,
                                  compact_schema, digest)

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


@dataclass(frozen=True)
class PhotoScope:
    report: PhotoReport
    options: tuple[PhotoOption, ...]
    state_sha256: str
    reason: str | None = None

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
                         status='ready' if operation else 'unsupported' if reason in
                         ('geometry', 'program_check', 'no_supported_control') else 'noop',
                         reason=reason, operation=operation)


def scope_plan(report, context, changes, *, plan=True, parameters=None):
    """Keep uncertainty, clashes and upstream defects out of the execution request."""
    if report.context != context:
        raise ValueError('photography report context changed')
    state_sha256 = digest([SCOPE_VERSION, parameters, [change.model_dump(mode='json') for change in changes]])

    def stop(reason):
        return PhotoScope(report, (), state_sha256, reason)

    if not plan:
        return stop('diagnose_only')
    if 'fail' in context.checks.values():
        return stop('program_check')
    visible = [(i, f) for i, f in enumerate(report.findings, 1) if f.visibility == 'visible']
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
    if any(f.severity == 'fatal' and f.factor != primary.factor for _, f in visible):
        return stop('conflict')
    if 'unlocalized' in report.reconciled:
        return stop('uncertain')
    if primary.confidence < MIN_ACTION_CONFIDENCE:
        return stop('confidence')
    if primary.factor in DIRECTED and primary.direction == 'unknown':
        return stop('uncertain')
    matched = [change for change in changes if change.factor == primary.factor
               and change.direction == primary.direction and change.value != change.current]
    # One root cause per round. An unsupported key light or contact shadow stays
    # in review; exposure, desaturation and haze cannot stand in for it.
    if not matched or primary.factor not in DIRECTED:
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
                       'finding': finding.model_dump(mode='json', include={'factor', 'direction', 'cue', 'reference'}),
                       'options': [o.model_dump(mode='json', exclude={'evidence_id', 'stage'}) for o in scope.options]}
            measured = scope.report.context.measurements.model_dump(
                mode='json', include=_PLAN_METRICS.get(finding.factor, set()), exclude_none=True)
            if measured:
                payload['measurements'] = measured
            choice = await ctx.deps.ask('photo-execution', PhotoChoice, role_prompt('editor'), payload,
                                        plain_json=True, planning=True, max_output_tokens=PLAN_TOKENS,
                                        compact=True, validate=scope.bind)
            plan = scope.bind(choice)
            return End(PhotoDecision(scope.report, plan, compile_plan(plan, scope)))

    @dataclass
    class Inspect(BaseNode[PhotoTask, ModelClient, PhotoDecision]):
        async def run(self, ctx: GraphRunContext[PhotoTask, ModelClient]) -> Retouch | End[PhotoDecision]:
            task = ctx.state

            report = task.report
            if report is None:
                brief = await ctx.deps.ask('photo-vision', PhotoBrief, role_prompt('observer'),
                    {'protocol': BRIEF_VERSION, **task.context.model_dump(mode='json', exclude_none=True)},
                    task.images, plain_json=True, compact=True, max_output_tokens=VISION_TOKENS,
                    image_limits=tuple((768, 512) if kind == 'asset' else (1024, 768) for kind in task.context.images))
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
            'context_schema': PhotoContext.model_json_schema(), 'diagnosis_schema': PhotoReport.model_json_schema(),
            'plan_schema': PhotoPlan.model_json_schema(),
            'provider_responses': {'visual': compact_schema(PhotoBrief), 'execution': compact_schema(PhotoChoice)},
            'factors': {name: {'score': score, 'stage': stage, 'cue': meaning}
                        for name, (_, score, stage, meaning) in FACTORS.items()}}
