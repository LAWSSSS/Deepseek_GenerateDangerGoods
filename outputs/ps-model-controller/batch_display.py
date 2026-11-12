"""Headless multi-background display batches, independent of Photoshop acceptance."""
import argparse
import asyncio
import json
from pathlib import Path
import re

from display_batch.common import atomic_json, resolve
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig


def evaluation_manifests(backgrounds,out):
    # Camera labels are confined to evaluation splitting, never used in production inference.
    groups={'development':[],'heldout':[]}
    for path in sorted(Path(backgrounds).glob('*.jpg')):
        camera=re.search(r'_(36|53)_\d+_\d+\.jpg$',path.name)
        if camera is None:raise ValueError('背景缺少可核对的评估相机分组')
        group='heldout' if camera[1]=='36' else 'development'
        groups[group].extend({'background':str(path.resolve()),'asset_id':k,'group':group}
                             for k in ('3','6','10','13','19-1','31'))
    for group,jobs in groups.items():atomic_json(Path(out)/(group+'.json'),jobs)
    return {k:len(v) for k,v in groups.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--config',type=Path,required=True)
    diagnostic=sub.add_parser('diagnose');diagnostic.add_argument('--config',type=Path,required=True)
    for command in (run,diagnostic):
        command.add_argument('--integration-mode',choices=('legacy','coherent'),default=None,
            help='覆盖配置中的融合路径：legacy 复现原流程；coherent 使用局部受光与反射协同小样策略')
    bench=sub.add_parser('evaluation-manifests');bench.add_argument('--backgrounds',required=True);bench.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='evaluation-manifests':
        print(evaluation_manifests(args.backgrounds,args.out));return
    raw=json.loads(args.config.read_text(encoding='utf8'))
    if args.integration_mode is not None:raw['integration_mode']=args.integration_mode
    config=BatchConfig.model_validate(raw)
    if args.command=='diagnose':
        from display_batch.diagnostics import diagnose
        print(asyncio.run(diagnose(config)));return
    summary=asyncio.run(Pipeline(config).run())
    brief={**summary,'usage_current_run':{k:v for k,v in summary['usage_current_run'].items() if k!='calls'}}
    print(json.dumps(brief,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
