"""Apply model-selected object adjustments to independently verified Photoshop exports."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from .photoshop import LIMITS, file_hash, load_request
from .photography_agent import run_photography
from .photography_controls import photoshop_changes
from .photography_measurements import measurements
from .photography_protocol import PhotoContext
from .workflow import observe, plan_execution

CAPABILITIES = {
    'brightness': {'range': LIMITS['brightness'], 'causes': ['lighting', 'color']},
    'warmth': {'range': LIMITS['warmth'], 'causes': ['color', 'lighting', 'haze_color']},
    'saturation': {'range': LIMITS['saturation'], 'causes': ['color', 'material'], 'integer': True},
}
GEOMETRY = {'floating', 'penetration', 'upright', 'occlusion', 'scale', 'camera'}
REQUIRED_CHECKS = {'physics', 'scale', 'support', 'occlusion', 'anchor', 'protected', 'outside_edit', 'holes_alpha', 'size'}


def load_composition(path):
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding='utf8'))
    if record.get('version') != 'verified-composition-v1' or record.get('color_encoding') != 'sRGB':
        raise ValueError('unsupported_composition_manifest')
    checks = record.get('program_checks', {})
    if not REQUIRED_CHECKS <= checks.keys() or any(
        item.get('status') not in ('pass', 'unknown') for item in checks.values()
    ):
        raise ValueError('composition_has_unverified_or_failed_program_checks')
    layers = (path.parent / record['layers']).resolve()
    if not layers.is_relative_to(path.parent) or file_hash(layers) != record['layer_manifest_sha256']:
        raise ValueError('composition_layer_manifest_changed')
    request = load_request(layers)
    if (not request.expected_png or not request.geometry_signature
            or request.geometry_signature != record['alpha_evidence_sha256']
            or file_hash(path.parent / 'alpha-evidence.npz') != request.geometry_signature
            or not {'background','expected_png','object_rgba','foreground_mask','protected_mask'} <= request.source_hashes.keys()):
        raise ValueError('composition_lacks_bound_layer_evidence')
    if request.actions:
        raise ValueError('composition_source_must_contain_unadjusted_layers')
    return request, checks


def severe(report):
    return [item for item in report.findings if item.visibility == 'visible' and item.severity in ('major', 'fatal')]


def geometry_defects(report):
    return [item for item in report.findings if item.visibility == 'visible' and item.code in GEOMETRY]


def photo_rank(report):
    scores = report.scores
    return (-len(severe(report)), scores.lighting + scores.texture + scores.material, scores.edge)


def visual_pass(report):
    scores = report.scores
    original = [scores.support, scores.occlusion, scores.lighting, scores.edge, scores.texture]
    return (report.trace == 'none' and min(original) >= 3 and sum(original) >= 16
            and min(scores.material, scores.structure) >= 3 and not severe(report))


def photo_context(source, image, checks):
    if file_hash(source.object_rgba) != source.source_hashes['object_rgba']:
        raise ValueError('composition object layer changed')
    if file_hash(image['path']) != image['sha256']:
        raise ValueError('review image changed')
    evidence = source.object_rgba.with_name('alpha-evidence.npz')
    if file_hash(evidence) != source.geometry_signature:
        raise ValueError('composition alpha evidence changed')
    with np.load(evidence, allow_pickle=False) as alpha_evidence:
        optical = alpha_evidence['optical']
        left, top, right, bottom = map(int, alpha_evidence['bounds'])
    with Image.open(image['path']) as exported:
        width, height = exported.size
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError('review alpha bounds are outside the canvas')
        frame = np.asarray(exported.convert('RGB').crop((left, top, right, bottom)))
    y, x = np.nonzero(optical > .01)
    if frame.shape[:2] != optical.shape or not len(x):
        raise ValueError('review image and object layer do not share a valid canvas')
    region = (float((x.min()+left)/width), float((y.min()+top)/height),
              float((x.max()+left+1)/width), float((y.max()+top+1)/height))
    return PhotoContext(target_region=region, images=['frame'],
                        checks={key: item['status'] for key, item in checks.items()},
                        measurements=measurements({}, object_pixels=frame[optical > .95]))


async def refine_photoshop(composition, output, adapter, client, *, max_iterations=2, recommendation=None,
                           review_profile='legacy'):
    """Review actual outputs and retain the best geometrically eligible PSD/PNG pair."""
    if type(max_iterations) is not int or not 0 <= max_iterations <= 10:
        raise ValueError('max_iterations_outside_0_to_10')
    if review_profile not in ('legacy', 'photographer'):
        raise ValueError('unknown photography review profile')
    source, checks = load_composition(composition)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / 'workflow.json'
    if ledger.exists():
        raise ValueError('photoshop_workflow_output_already_used')
    result = {'version': 'photoshop-review-v1', 'status': 'running', 'trials': [],
              'composition': str(composition), 'program_checks': checks, 'review_profile': review_profile}

    def persist():
        staging = ledger.with_suffix('.tmp')
        staging.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        staging.replace(ledger)

    persist()
    current = dict.fromkeys(CAPABILITIES, 0.)
    visited_params, visited_images = set(), set()
    best = None
    started = time.monotonic()
    try:
        for iteration in range(max_iterations + 1):
            state = tuple(current.items())
            if state in visited_params:
                result['reason'] = 'repeated_parameters'
                break
            visited_params.add(state)
            before = time.monotonic()
            record = {'iteration': iteration, 'parameters': dict(current)}
            result['trials'].append(record)
            try:
                exported = await adapter.export(replace(source, actions=dict(current)), output / f'trial-{iteration}')
                record.update(export=exported, photoshop_seconds=round(time.monotonic()-before, 3))
                image = exported['outputs']['png']
                if image['sha256'] in visited_images:
                    result['reason'] = 'repeated_image'
                    break
                visited_images.add(image['sha256'])
                if review_profile == 'photographer':
                    context = photo_context(source, image, checks)
                    decision = await run_photography(client, context, image['sha256'], images=[image['path']],
                                                     target=source.object_id)
                    record['photography_diagnosis'] = decision.report.model_dump(mode='json')
                    report = decision.report.legacy()
                else:
                    report = await observe(client, image['sha256'], [image['path']], {
                        'program_checks': {key: item['status'] for key, item in checks.items()},
                        'photoshop_parameters': current,
                    }, target=source.object_id, prompt='Assess lighting, color, contrast, support and recognizable structure.')
                record['visual_evidence'] = report.model_dump(mode='json')
                if geometry_defects(report):
                    result['reason'] = 'geometry_recheck_required'
                    result['geometry_findings'] = [f.model_dump(mode='json') for f in geometry_defects(report)]
                    best = None
                    break
                if best is not None:
                    previous = best[0]
                    if (photo_rank(report) <= photo_rank(previous) or
                        any(getattr(report.scores, name) < getattr(previous.scores, name)
                            for name in ('structure', 'support', 'occlusion'))):
                        result['reason'] = 'no_improvement'
                        break
                best = report, exported, iteration
                if visual_pass(report):
                    result['reason'] = 'visual_checks_passed'
                    break
                if iteration == max_iterations:
                    result['reason'] = 'iterations_exhausted'
                    break
                if review_profile == 'photographer':
                    changes = [change for change in photoshop_changes(current, LIMITS)
                               if tuple({**current, change.control: change.value}.items()) not in visited_params]
                    decision = await run_photography(client, context, file_hash(image['path']), target=source.object_id,
                        changes=changes, plan=True, report=decision.report, parameters=current)
                    plan, updates = decision.plan, decision.updates
                    if updates and file_hash(image['path']) != plan.image_sha256:
                        raise ValueError('photography plan image changed before execution')
                else:
                    if not any(f.visibility == 'visible' and f.direction != 'unknown'
                               and any(f.code in cap['causes'] for cap in CAPABILITIES.values())
                               for f in report.findings):
                        result['reason'] = 'no_supported_action_evidence'
                        break
                    plan, updates = await plan_execution(client, report, current, CAPABILITIES,
                        {'photoshop_checks': exported['validation'], 'current_scores': report.scores.model_dump()},
                        recommendation, target=source.object_id)
                record['execution_plan'] = plan.model_dump(mode='json')
                if not updates:
                    result['reason'] = plan.reason if review_profile == 'photographer' else plan.status
                    break
                current = {**current, **updates}
            except (ValueError, OSError, RuntimeError) as error:
                record['error_kind'] = type(error).__name__
                record['reason'] = str(error)[:180]
                result['reason'] = 'execution_or_review_stopped'
                break
            finally:
                persist()
        if best is None:
            result['status'] = 'failed' if result.get('reason') == 'geometry_recheck_required' else 'review'
        else:
            report, exported, iteration = best
            result.update(best_iteration=iteration, outputs={},
                          status='accepted_display' if visual_pass(report) and
                              all(item['status']=='pass' for item in checks.values()) else 'review')
            for kind in ('png', 'psd'):
                artifact = exported['outputs'][kind]
                dest = output / ('final.' + kind)
                shutil.copy2(artifact['path'], dest)
                result['outputs'][kind] = {'path':str(dest), 'sha256':artifact['sha256']}
        return result
    finally:
        result['seconds'] = round(time.monotonic()-started,3)
        if hasattr(client, 'usage'):
            result['usage'] = client.usage()
        persist()
