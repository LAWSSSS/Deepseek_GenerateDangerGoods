"""Diagnose an existing full frame and preview a bounded retouching plan."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

from display_batch.ai import InvalidStructuredReply, ProviderUnavailable, Vision
from display_batch.common import FileLease, atomic_json, file_hash, resolve
from display_batch.review import allowed_actions
from display_batch.schema import Action, BatchConfig, TONE_PROFILES, apply_actions
from display_batch.structured import frame_inputs
from scrap_composite.photography_agent import interface, run_photography
from scrap_composite.photography_controls import composite_changes, photoshop_changes
from scrap_composite.photography_protocol import PhotoContext, PhotoReport
from scrap_composite.photoshop import LIMITS
from scrap_composite.photoshop_workflow import load_composition, photo_context


async def review(args):
    output = args.out.resolve()
    if args.draft:
        saved = json.loads(args.draft.read_text(encoding='utf8'))
        comp = saved.get('composition', saved)
        current = saved['params']
        image = Path(comp['image']).resolve(strict=True)
        checksum = saved['image_hash']
        target = 'object'
        asset = {'name': args.name, 'semantic': {'material_classes': args.material}, 'board': args.reference}
        if file_hash(image) != checksum:
            raise ValueError('待复核图片与成图记录散列不一致')
        context, images = frame_inputs(asset, comp)
        changes = []
        if args.plan:
            style = {'local':'local', 'native':'native', 'reference':'reference', 'camera':'scene'}[current['photo_model']]
            allowed = allowed_actions(current, comp, [current['candidate']], render_style=style)
            changes = composite_changes(current, allowed, comp['photo'], tone_profiles=TONE_PROFILES)
    elif args.manifest:
        source, checks = load_composition(args.manifest)
        image = Path(source.expected_png)
        checksum = source.source_hashes['expected_png']
        target = source.object_id
        context = photo_context(source, {'path': image, 'sha256': checksum}, checks)
        images = [image]
        current = dict.fromkeys(LIMITS, 0)
        changes = photoshop_changes(current, LIMITS) if args.plan else []
    else:
        image = args.image.resolve(strict=True)
        checksum = file_hash(image)
        target = 'object'
        images, kinds = [image], ['frame']
        for path, kind in ((args.crop, 'local'), (args.reference, 'asset')):
            if path:
                images.append(path.resolve(strict=True))
                kinds.append(kind)
        context = PhotoContext(name=args.name, target_region=tuple(args.region) if args.region else None,
                               images=kinds, material=args.material, question=args.question)
        current, changes = {}, []
    report = None
    if args.diagnosis:
        saved = json.loads(args.diagnosis.read_text(encoding='utf8'))
        report = PhotoReport.model_validate_json(json.dumps(saved.get('diagnosis', saved)))
        if report.context != context or report.image_sha256 != checksum or report.target != target:
            raise ValueError('已有诊断与当前图层清单或图片不一致')
    config = BatchConfig(backgrounds='.', output=str(output), provider_config=args.provider,
                         review_profile='photographer', offline=args.offline)
    lease = FileLease(output/'.photo-review.lock')
    if not lease.acquire():
        raise ValueError('已有复核在使用该输出目录')
    try:
        if (output/'review.json').exists():
            raise ValueError('输出目录已有复核结果，请选择新目录')
        if file_hash(image) != checksum:
            raise ValueError('待复核图片与记录的散列不一致')
        client = Vision(config, resolve(config.cache)/'vision', output/f'usage-photo-review-{time.time_ns()}.json')
        asyncio.current_task().set_name('photo-review')
        result = {'status': 'review', 'image': str(image), 'current_parameters': current}
        if args.draft:
            result['draft'] = str(args.draft.resolve())
        if args.manifest:
            result['manifest'] = str(args.manifest.resolve())
        try:
            diagnosis = await run_photography(client, context, checksum, images=images, target=target, report=report)
            result['diagnosis'] = diagnosis.report.model_dump(mode='json')
            if args.plan:
                decision = await run_photography(client, context, file_hash(image), target=target,
                                                 changes=changes, plan=True, report=diagnosis.report, parameters=current)
            else:
                decision = diagnosis
            result.update(status='completed', plan=decision.plan.model_dump(mode='json'))
            if args.draft and decision.updates:
                actions = [Action(kind=name, value=value) for name, value in decision.updates.items()]
                result['proposed_parameters'] = apply_actions(current, actions, [current['candidate']])
        except (InvalidStructuredReply, ProviderUnavailable, OSError, ValueError) as error:
            result['reason'] = str(error)
        finally:
            await client.close()
        result['usage'] = client.usage()
        atomic_json(output/'review.json', result)
        return result
    finally:
        lease.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('schema', help='查看摄影复核协议、因素和输出预算')
    run = commands.add_parser('review', help='对现有全景做摄影复核')
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--draft', type=Path, help='读取已有 draft-N.json 或带 composition 的样片结果')
    source.add_argument('--manifest', type=Path, help='使用经过验证的 composition.json')
    run.add_argument('--crop', type=Path)
    run.add_argument('--reference', type=Path)
    run.add_argument('--region', type=float, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    run.add_argument('--name', default='')
    run.add_argument('--material', action='append', default=[])
    run.add_argument('--question')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--provider')
    run.add_argument('--offline', action='store_true')
    run.add_argument('--plan', action='store_true', help='为现有成图或 Photoshop 图层清单生成预览计划')
    run.add_argument('--diagnosis', type=Path, help='复用同一成图及上下文保存的结构化诊断')
    args = parser.parse_args()
    if args.command == 'schema':
        print(json.dumps(interface(), ensure_ascii=False, indent=2))
        return 0
    if (args.plan and args.image) or (args.diagnosis and not args.plan):
        parser.error('--plan 需要 --draft 或 --manifest，--diagnosis 需要 --plan')
    if args.manifest and (args.crop or args.reference or args.region or args.name or args.question or args.material):
        parser.error('--manifest 已提供目标区域与图层证据')
    if args.draft and (args.crop or args.region or args.question):
        parser.error('--draft 使用已有成图记录的局部图和目标区域')
    try:
        result = asyncio.run(review(args))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({'report': str(args.out.resolve()/'review.json'), 'status': result['status'],
                      'plan': result.get('plan', {}).get('status'), 'cost': result['usage']['cost'],
                      **({'reason': result['reason']} if 'reason' in result else {})}, ensure_ascii=False))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
