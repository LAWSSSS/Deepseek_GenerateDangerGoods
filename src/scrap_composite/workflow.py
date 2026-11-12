"""Shared model steps. The caller owns image production and hard checks."""
from .contracts import ExecutionPlan, VisualReport, validate_plan


async def observe(client, image_sha256, images, context, *, target='object', prompt=''):
    report = await client.ask(
        'visual-evidence', VisualReport,
        'Inspect the actual first, full-frame composite. Further images provide context only. '
        'Return visual-evidence-v1 JSON with the supplied image hash and target. Score each item 0–4. '
        'Use full-frame normalized rectangles and cite visible evidence. Do not invent exact scale, '
        'hidden contact or program checks. Direction means increase/decrease in the measured '
        'quantity needed to correct the finding, or unknown. ' + prompt,
        {**context, 'image_sha256': image_sha256, 'target': target}, images,
        plain_json=True, max_output_tokens=3072,
        validate=lambda candidate: candidate.bind(image_sha256, target),
    )
    return report.bind(image_sha256, target)


async def plan_execution(client, report, current, capabilities, measurements, recommendation=None, *, target='object'):
    plan = await client.ask(
        'execution-plan', ExecutionPlan,
        'Return execution-plan-v1 JSON. Use at most three registered operations with absolute values '
        'and IDs from the current visible findings. Use the supplied image hash and target. '
        'Use noop for no useful operation or unsupported for defects outside the capabilities. '
        'Never generate scripts, paths, geometry coordinates, or permission to skip checks.',
        {'observation': report.model_dump(mode='json'), 'target': target,
         'current_parameters': current, 'capabilities': capabilities,
         'measurements': measurements, 'recommendation': recommendation},
        plain_json=True, max_output_tokens=1024, planning=True,
        validate=lambda candidate: validate_plan(candidate, report, current, capabilities, target=target),
    )
    return plan, validate_plan(plan, report, current, capabilities, target=target)
