"""Apply a validated placement plan to an existing transparent image with Photoshop MCP."""
import argparse
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from plan_placement import Placement

HERE=Path(__file__).resolve().parent


async def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--background", type=Path, required=True)
    cli.add_argument("--asset", type=Path, required=True)
    cli.add_argument("--crop", type=int, nargs=4, required=True, metavar=("X","Y","W","H"))
    cli.add_argument("--plan", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--mcp",default=str(Path(os.environ.get("LOCALAPPDATA","."))/
                     "OpenAI/PhotoshopMCP/python/Scripts/photoshop-mcp-server.exe"))
    args=cli.parse_args()
    report=json.loads(args.plan.read_text(encoding="utf-8-sig"))
    placement=Placement.model_validate_json(json.dumps(report["placement"]))
    if args.crop[0]<0 or args.crop[1]<0 or args.crop[2]<1 or args.crop[3]<1:
        cli.error("invalid crop")
    target=args.out.resolve()
    target.mkdir(parents=True,exist_ok=False)
    psd,png=target/"final.psd",target/"final.png"
    with tempfile.TemporaryDirectory(prefix="ps-place-") as name, (target/"mcp.log").open("w",encoding="utf-8") as log:
        temp=Path(name)
        background=temp/("background"+args.background.suffix)
        asset=temp/("cutout"+args.asset.suffix)
        shutil.copyfile(args.background,background)
        shutil.copyfile(args.asset,asset)
        job={"background":background.as_posix(),"asset":asset.as_posix(),"crop":args.crop,
             "placement":placement.model_dump(),"psd":psd.as_posix(),"png":png.as_posix()}
        code="var JOB="+json.dumps(job,ensure_ascii=True,allow_nan=False)+";\n"+(HERE/"place.jsx").read_text(encoding="utf8")
        server=StdioServerParameters(command=str(Path(args.mcp).resolve(strict=True)))
        async with stdio_client(server,errlog=log) as (read,write):
            async with ClientSession(read,write) as session:
                await asyncio.wait_for(session.initialize(),timeout=30)
                result=await asyncio.wait_for(session.call_tool("photoshop_execute_jsx",{"script":code}),timeout=120)
                payload=json.loads(result.content[0].text)
                if result.isError or not payload.get("success") or not psd.is_file() or not png.is_file():
                    raise RuntimeError(f"Photoshop placement failed: {payload.get('error','missing output')}")
                values=[float(x) for x in payload["result"].split(",")]
    report.update(crop=args.crop,canvas=values[:2],object_bounds=values[2:6],
                  occluder_count=int(values[6]),outputs={"psd":str(psd),"png":str(png)})
    (target/"result.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    asyncio.run(main())