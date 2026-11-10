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
    bench=sub.add_parser('evaluation-manifests');bench.add_argument('--backgrounds',required=True);bench.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='evaluation-manifests':
        print(evaluation_manifests(args.backgrounds,args.out));return
    config=BatchConfig.model_validate_json(args.config.read_text(encoding='utf8'))
    if args.command=='diagnose':
        from display_batch.diagnostics import diagnose
        print(asyncio.run(diagnose(config)));return
    summary=asyncio.run(Pipeline(config).run())
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
