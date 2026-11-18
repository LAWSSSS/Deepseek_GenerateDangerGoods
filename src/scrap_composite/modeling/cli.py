"""Add, rebuild, inspect and release one local model."""
import argparse
import json

from .contracts import BuildOptions
from .service import ModelingService


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', help='独立建模目录，用于本地验证')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('build')
    ident = run.add_mutually_exclusive_group(required=True)
    ident.add_argument('--name', help='新物品名称')
    ident.add_argument('--asset', help='已有物品编号')
    run.add_argument('--source')
    run.add_argument('--view', action='append', default=[], metavar='DIRECTION=PATH')
    run.add_argument('--subject', help='object:名称 或 collection:名称')
    run.add_argument('--operation', choices=('rebuild','retexture','verify'), default='rebuild')
    run.add_argument('--major-mm', type=float)
    run.add_argument('--dimension-source')
    run.add_argument('--material-library')
    for name in ('resume','status'):
        commands.add_parser(name).add_argument('job')
    pub = commands.add_parser('publish')
    pub.add_argument('asset')
    pub.add_argument('version')
    commands.add_parser('list')
    args = parser.parse_args(argv)
    service = ModelingService(args.workspace)
    try:
        if args.command == 'build':
            views = dict(entry.split('=',1) for entry in args.view)
            raw = {key:getattr(args,key) for key in BuildOptions.model_fields if getattr(args,key) is not None}
            record = service.submit(name=args.name, asset_id=args.asset, source=args.source, views=views,
                                    options=BuildOptions.model_validate(raw))
            print(json.dumps({'job':record['id'], 'asset':record['asset']['id']}, ensure_ascii=False), flush=True)
            result = service.run(record['id'])
        elif args.command == 'resume':
            result = service.run(args.job)
        elif args.command == 'status':
            result = service.registry.get(args.job)
        elif args.command == 'publish':
            result = service.publish(args.asset, args.version)
        else:
            result = service.items()
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if isinstance(result, dict) and result.get('state') == 'failed' else 0
