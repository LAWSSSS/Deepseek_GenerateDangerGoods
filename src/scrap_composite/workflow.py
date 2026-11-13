"""Shared model steps. The caller owns image production and hard checks."""
from .contracts import PlanProposal, VisualObservation, validate_plan


async def observe(client, image_sha256, images, context, *, target='object', prompt=''):
    observation = await client.ask(
        'visual-evidence', VisualObservation,
        'Inspect the actual first, full-frame composite. Further images provide context only. '
        'Return JSON containing only trace, scores, findings and summary. Score each item 0–4. '
        'Every finding concerns the sole inserted object. Use [left, top, right, bottom] in the '
        'first full image and cite visible evidence. Do not invent exact scale, '
        'hidden contact or program checks. Use physical_cue for direct geometry evidence only: '
        'floating=visible_air_gap, penetration=surface_intersection, occlusion=reversed_overlap, '
        'upright=standing_on_end. If no direct cue is visible, report shadow or edge if applicable, '
        'or use unassessable for hidden geometry. Direction means increase/decrease in the measured '
        'quantity needed to correct the finding, or unknown. ' + prompt,
        {**context, 'image_sha256': image_sha256, 'target': target}, images,
        plain_json=True, max_output_tokens=3072,
    )
    return observation.for_frame(image_sha256, target)


async def plan_execution(client, report, current, capabilities, measurements, recommendation=None, *, target='object'):
    report.bind(report.image_sha256, target)
    supported = {}
    for name, capability in capabilities.items():
        evidence_ids = [finding.id for finding in report.findings if finding.visibility == 'visible'
                        and finding.code in capability['causes']
                        and finding.id in capability.get('evidence_ids', [finding.id])
                        and ('values' in capability or finding.direction != 'unknown')]
        if evidence_ids:
            supported[name] = {**capability, 'evidence_ids': evidence_ids}
    if not supported:
        return PlanProposal(status='noop', operations=[], note='当前证据没有可执行操作').for_frame(
            report.image_sha256, target), {}

    def compile_proposal(proposal):
        plan = proposal.for_frame(report.image_sha256, target)
        return plan, validate_plan(plan, report, current, supported, target=target)

    proposal = await client.ask(
        'execution-plan', PlanProposal,
        'Return JSON containing only status, operations and note. Use at most three registered operations '
        'with absolute values different from the current ones. For each operation copy an exact value '
        'from its values list when present, including its JSON type. Use evidence_ids from the selected '
        'capability. A single supported operation is sufficient. '
        'Use noop for no useful operation or unsupported for defects outside the capabilities. '
        'Never generate scripts, paths, geometry coordinates, or permission to skip checks.',
        {'observation': report.model_dump(mode='json'), 'target': target,
         'current_parameters': {name: current[name] for name in supported}, 'capabilities': supported,
         'measurements': measurements, 'recommendation': recommendation},
        plain_json=True, max_output_tokens=1024, planning=True,
        validate=compile_proposal,
    )
    return compile_proposal(proposal)
