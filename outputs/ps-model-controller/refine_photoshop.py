"""Review and refine one verified composition through a local Photoshop session."""
import argparse
import asyncio
import json
from pathlib import Path

from display_batch.ai import Vision
from display_batch.common import atomic_json, resolve, signature
from display_batch.schema import BatchConfig
from scrap_composite.photoshop import PhotoshopAdapter, availability
from scrap_composite.photoshop_workflow import refine_photoshop


async def run(manifest, output, *, iterations=2, provider=None, mcp=None):
    output = Path(output)
    state = await asyncio.to_thread(availability, mcp)
    if not state['available']:
        result={'status':'review','reason':'photoshop_unavailable','availability':state,
                'new_api_requests':0,'outputs':{}}
        atomic_json(output/'unavailable.json',result)
        return result
    settings=BatchConfig(backgrounds='.',output=str(output),provider_config=provider,workflow='evidence_v1')
    client=Vision(settings,resolve(settings.cache)/'vision',
                  output/f'usage-photoshop-{signature(str(output.resolve()))[:12]}.json')
    try:
        return await refine_photoshop(manifest,output,PhotoshopAdapter(mcp=mcp),client,max_iterations=iterations)
    finally:
        await client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,choices=range(11),default=2)
    parser.add_argument('--provider')
    parser.add_argument('--mcp',type=Path)
    args=parser.parse_args()
    result=asyncio.run(run(args.manifest,args.out,iterations=args.iterations,provider=args.provider,mcp=args.mcp))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
