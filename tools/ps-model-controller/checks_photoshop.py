"""Run the Photoshop metric, foreground stack and pixel guard smoke check."""
import argparse
import asyncio
import os
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from batch_v2 import Config, MetricGap, Runner, lock_directory
from metric_scale import ObjectSize, SceneAnchor, SceneReference, fit_consensus
from scene_v2 import gap_candidates

width,height=800,600
background=np.full((height,width,3),43,dtype=np.uint8)
background[240:420,371:625]=(90,92,99)
background[190:360,190:371]=(95,59,51)
noise=np.random.default_rng(11).integers(-5,6,size=(height,width,1),dtype=np.int16)
background=np.clip(background.astype(np.int16)+noise,0,255).astype(np.uint8)
depth=np.full((height,width),.39,dtype=np.float32)
depth[190:360,190:371]=.63
rgba=np.zeros((100,244,4),dtype=np.uint8)
rgba[2:-2,2:-2]=(123,125,132,255)

def norm(x,y):
    return round(x*1000/(width-1)),round(y*1000/(height-1))

class FixtureMasks:
    def seeded_masks(self,crop,points):
        return [[np.all(np.abs(crop.astype(np.int16)-color)<=5,axis=2)] for color in [(90,92,99),(95,59,51)]]

async def check(root,log,mcp):
    frame=Image.fromarray(background)
    src,asset=root/"frame.png",root/"asset.png"
    frame.save(src)
    Image.fromarray(rgba).save(asset)
    size=ObjectSize(major_mm=(1000,1000),minor_mm=(400,400),basis="measured",source="synthetic metric fixture")
    ref=SceneReference(id="R1",label="120-pixel gauge",span_mm=(600,600),points=(norm(400,345),norm(520,345)),
                       basis="measured",source="synthetic metric fixture")
    site=MetricGap(view=1,pose=(*norm(370,290),0),seat=norm(430,325),front=[norm(343,307)],
                   keep=(850,420),ref="R1")
    fit=fit_consensus(rgba[:,:,3],frame.size,depth,site.seat,size,[ref])
    candidates=gap_candidates(FixtureMasks(),frame,depth,site,asset,(.035,.10),fit,photo=True,protected=[(17,110,98,131)])
    cache=root/"cache"
    export=root/"export"
    cache.mkdir()
    export.mkdir()
    async with stdio_client(StdioServerParameters(command=str(mcp)),errlog=log) as (reader,writer):
        async with ClientSession(reader,writer) as session:
            await session.initialize()
            runner=Runner(session,None,Config(),None,cache,None)
            spec=runner.materialize_gap(candidates[0],frame.size,root,"A",1)
            spec["scale"]=fit.record
            spec["protected"]=[(17,110,98,131)]
            report={}
            assert spec["photography"]["status"]=="measured" and spec["photography"]["grain_from_scene"]
            preview=(await runner.ps(dict(mode="compose",source=src.as_posix(),candidates=[spec])))[0]
            assert "error" not in preview,preview
            await runner.publish(src.as_posix(),src,spec,export,root,report)
            actual=report["final"]
            assert actual["scale"]["status"]=="calibrated"
            assert abs(actual["scale"]["rendered_major_px"]-200)<=3,actual
            assert (export/"final.psd").is_file()
            psd_path=json.dumps((export/"final.psd").as_posix())
            jsx='var doc=app.open(new File('+psd_path+')); var top=[], middle=[]; for(var i=0;i<doc.layers.length;i++)top.push(doc.layers[i].name); var group=doc.layerSets[0]; for(var j=0;j<group.artLayers.length;j++)middle.push(group.artLayers[j].name); doc.close(SaveOptions.DONOTSAVECHANGES); JSON.stringify({top:top,middle:middle});'
            call=await session.call_tool("photoshop_execute_jsx",{"script":jsx})
            layers=json.loads(json.loads(call.content[0].text)["result"])
            assert len(layers["top"])==3 and "original pixels only" in layers["top"][0],layers
            assert any("color cast" in n for n in layers["middle"]) and any("CCTV texture" in n for n in layers["middle"]),layers
            a=np.asarray(Image.open(spec["preview"])).astype(int)
            b=np.asarray(Image.open(root/"final.jpg")).astype(int)
            assert a.shape==b.shape and np.abs(a-b).mean()<1,(a.shape,b.shape,np.abs(a-b).mean())
            top=np.asarray(Image.open(root/"final-foreground-alpha.png"))
            l,t,r,b=spec["crop"]
            opaque=top[:,:,3]==255
            assert opaque.any() and np.array_equal(top[:,:,:3][opaque],background[t:b,l:r][opaque])
            l,t,r,b=actual["foreground_bounds"]
            assert (r-l)*(b-t)<.1*width*height,actual["foreground_bounds"]
            anchors=[
                SceneAnchor(id="R2",label="estimated span 2",span_mm=(555,650),points=(norm(400,345),norm(520,345))),
                SceneAnchor(id="R3",label="estimated span 3",span_mm=(705,780),points=(norm(420,368),norm(570,368))),
            ]
            fit_guess=fit_consensus(rgba[:,:,3],frame.size,depth,site.seat,size,anchors)
            spec_guess=runner.materialize_gap(
                gap_candidates(FixtureMasks(),frame,depth,site,asset,(.035,.10),fit_guess,photo=True)[0],
                frame.size,root,"B",1)
            spec_guess["scale"]=fit_guess.record
            guess={}
            await runner.publish(src.as_posix(),src,spec_guess,export,root,guess,estimated=True)
            assert guess["final"]["scale"]["status"]=="estimated" and "outputs" not in guess
            assert all((export/p).is_file() for p in guess["preview_outputs"].values())
            return {"expected_px":200,"projected_px":actual["scale"]["projected_major_px"],
                    "rendered_px":actual["scale"]["rendered_major_px"],
                    "occlusion_ratio":round(actual["occlusion_ratio"],3),
                    "occlusion_edge_agreement":actual["occlusion_edge_agreement"],
                    "top_layer_bounds":actual["foreground_bounds"],
                    "layers":layers,"estimated_deliverables":guess["preview_outputs"]}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache",type=Path,required=True,help="batch cache directory to lock during the check")
    cli.add_argument("--mcp",type=Path,default=Path(os.environ.get("LOCALAPPDATA","."))/
                     "OpenAI/PhotoshopMCP/python/Scripts/photoshop-mcp-server.exe")
    args=cli.parse_args()
    with lock_directory(args.cache.resolve()), tempfile.TemporaryDirectory(prefix="ps-metric-") as tmp:
        with (Path(tmp)/"mcp.log").open("w",encoding="utf-8") as log:
            report=asyncio.run(check(Path(tmp),log,args.mcp.resolve(strict=True)))
    print(json.dumps(report,ensure_ascii=True))


if __name__=="__main__":
    main()
