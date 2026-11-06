"""Synthetic 3-D-render layer placement and pixel-protection smoke test."""
import argparse
import asyncio
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from batch_v2 import Budget, Runner, lock_directory
from mesh_placement import compose_spec, open_fit
from metric_scale import silhouette_axes
from pose_retest import validate
from run import Config
from scene_v2 import read_image


class SyntheticAtlas:
    def clearance_crop(self,support_id,crop):
        return np.full((300,400),100,dtype=np.float32)

    def native_position_context(self,*args,**kwargs):
        return {"cost":.2,"relief":{"kind":"flat"}}


async def check(root,mcp):
    rng=np.random.default_rng(11)
    pixels=np.clip(np.full((600,800,3),92,dtype=np.int16)+rng.integers(-4,5,(600,800,1)),0,255).astype("uint8")
    frame=Image.fromarray(pixels)
    source=root/"frame.png"
    frame.save(source)
    candidate_dir=root/"A"
    candidate_dir.mkdir()
    support=np.zeros((300,400),np.uint8)
    support[5:-5,5:-5]=255
    Image.fromarray(support).save(candidate_dir/"support.png")
    Image.new("L",(400,300)).save(candidate_dir/"foreground.png")
    asset=np.zeros((80,180,4),np.uint8)
    asset[4:68,7:173]=(122,105,91,255)
    shadow=np.zeros((80,180),np.float32)
    shadow[68:75,8:172]=.18
    left=np.zeros((80,180),bool);left[64:72,12:30]=True
    right=np.zeros_like(left);right[64:72,149:167]=True
    target=float(silhouette_axes(asset[:,:,3])[0])
    candidate={"id":"A","placement_mode":"open","target_px":target,"crop_px":[200,120,600,420],
               "center_px":[400,270],"support_id":"S1","scene_scale_status":"calibrated"}
    fit=open_fit(candidate,root,SyntheticAtlas(),frame,(asset,shadow,[left,right]),
                 source_visible_fraction=.9,physics_stable=True)
    spec,_=compose_spec(fit,frame,root/"render","A")
    with (root/"mcp.log").open("w",encoding="utf8") as log:
        async with stdio_client(StdioServerParameters(command=str(mcp)),errlog=log) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await session.initialize()
                runner=Runner(session,None,Config(),None,root,Budget(root/"usage.json",0))
                reference=read_image(await runner.reference(source.as_posix(),source,root))
                outcome=(await runner.ps({"mode":"compose","source":source.as_posix(),"candidates":[spec]}))[0]
                _,measured=validate(reference,spec,outcome,target)
                assert Path(spec["png"]).is_file() and Path(spec["psd"]).is_file()
                assert measured["protected_pixels_verified"]
                assert abs(measured["rendered_major_px"]-target)<=max(3,.05*target)
                return measured


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache",type=Path,required=True)
    cli.add_argument("--mcp",type=Path,default=Path(os.environ.get("LOCALAPPDATA","."))/
                     "OpenAI/PhotoshopMCP/python/Scripts/photoshop-mcp-server.exe")
    args=cli.parse_args()
    with lock_directory(args.cache.resolve()),tempfile.TemporaryDirectory(prefix="mesh-ps-check-") as temp:
        result=asyncio.run(check(Path(temp),args.mcp.resolve(strict=True)))
    print("Photoshop 3-D layer smoke passed:",result)


if __name__=="__main__":
    main()
