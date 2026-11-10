"""Independent fixed-placement replay audit: geometry, alpha, occlusion and saved pixels."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from display_batch.ai import Vision
from display_batch.common import ROOT,atomic_json
from display_batch.compose import warp
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig


async def audit(folder,out):
    folder=Path(folder);source=json.loads((folder/'results.json').read_text(encoding='utf8'))
    cache=ROOT/'work/display-batch-cache'
    vision=Vision(BatchConfig(backgrounds='unused',output=str(folder),offline=True),cache/'vision',folder/'audit-usage.json')
    scenes=Scenes(cache,vision,legacy=True);checks=[]
    try:
        for row in source['records']:
            candidate=row['fixed_candidate'];old=Path(row['old_image']);batch=old.parents[1]
            original_record=next(json.loads(line) for line in (batch/'results.jsonl').read_text(encoding='utf8').splitlines()
                                 if json.loads(line)['job_id']==old.stem)
            choice=next(v for v in original_record['iterations'] if v['iteration']==original_record['best_iteration'])
            candidates=json.loads((batch/'jobs'/old.stem/'candidates.json').read_text(encoding='utf8'))['candidates']
            before=next(c for c in candidates if c['id']==choice['params']['candidate'])
            geometry_unchanged=all(candidate.get(k)==before.get(k) for k in
                ('matrix','anchor_px','full_px','yaw','plane_normal','elevation','bounds','expected_major_px'))
            scene=await scenes.prepare(row['background']);bounds=row['edit_bounds'];x,y,r,b=bounds
            render=Path(candidate['fixed_alpha_dir']);meta=json.loads((render/'result.json').read_text(encoding='utf8'))
            with Image.open(render/'object.png') as im:alpha=np.asarray(im.getchannel('A'),np.float32)/255
            offset=np.asarray(candidate['anchor_px'])-np.asarray(meta['anchor_uv'])*candidate['full_px']
            alpha=warp(alpha,candidate['full_px'],offset,bounds)
            site=next(s for s in scene['supports'] if s['id']==candidate['site_id'])
            front=(scene['depth_native'][y:b,x:r]>site['depth']+.10)&~site['mask'][y:b,x:r]
            visible=alpha*(~front&~scene['protected'][y:b,x:r])
            with Image.open(row['image']) as im:actual=np.asarray(im.convert('RGB'))
            original=np.asarray(scene['frame']);same_size=original.shape==actual.shape
            changed=np.any(actual!=original,axis=2)
            outside=changed.copy();outside[y:b,x:r]=False
            c={'id':row['id'],'geometry_unchanged':geometry_unchanged,'size_pass':same_size,
                'alpha_unchanged':hashlib.sha256(alpha.tobytes()).hexdigest()==row['alpha_hash'],
                'occlusion_unchanged':hashlib.sha256(visible.tobytes()).hexdigest()==row['visible_alpha_hash'],
                'outside_changed_pixels':int(outside.sum()),
                'protected_changed_pixels':int((changed&scene['protected']).sum()),
                'foreground_changed_pixels':int(changed[y:b,x:r][front].sum()),
                'shadow_source_unchanged':Path(candidate['fixed_shadow_dir'])==render}
            c['passed']=all(c[k] for k in ('geometry_unchanged','size_pass','alpha_unchanged','occlusion_unchanged','shadow_source_unchanged')) and not any(
                c[k] for k in ('outside_changed_pixels','protected_changed_pixels','foreground_changed_pixels'))
            checks.append(c)
    finally:await vision.close()
    result={'kind':'independent_fixed_geometry_readback','images':len(checks),'passed':sum(c['passed'] for c in checks),
            'new_api_calls':len(vision.calls),'new_render_calls':0,'records':checks}
    atomic_json(out,result);print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--photo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();asyncio.run(audit(a.photo,a.out))
