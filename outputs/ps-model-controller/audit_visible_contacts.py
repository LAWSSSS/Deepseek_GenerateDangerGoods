"""Audit existing placements against visible support evidence, without generating composites."""
import argparse
import asyncio
import copy
import html
import json
from pathlib import Path
import time
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from display_batch.common import resolve,atomic_json
from display_batch.geometry import project
from display_batch.pile import receiver_mesh,visible_support_mask,support_plane_evidence
from display_batch.scene import Scenes
from display_batch.strict_contact import projected_gap
from validate_reference_projection import read,records
from verify_geometry_rebuild import source_job
from verify_coherent_expanded import cached_perception,Perception


class FrozenSceneAnswers:
    """Replay saved segmentation proposals, never request fresh analysis."""
    async def ask(self,role,schema,prompt,payload,images,**kwargs):
        if role!='scene':raise RuntimeError('offline_scene_audit_forbids_API')
        return schema.model_validate(read(Path(images[0]).parent/'analysis.json'))


async def audit(batch,out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    cache=resolve(read(batch/'config.json')['cache']);scenes=Scenes(cache,FrozenSceneAnswers());rows=[]
    with patch.object(Perception,'scene',cached_perception(Perception.scene)):
        for row in records(batch):
            if not row.get('image'):continue
            scene=await scenes.prepare(row['background'])
            entry=next(e for e in row['iterations'] if e['iteration']==row['best_iteration'])
            job=source_job(batch,row)
            candidate=copy.deepcopy(next(c for c in read(job/'candidates.json')['candidates'] if c['id']==entry['params']['candidate']))
            candidate['integration_mode']='coherent'
            candidate['receiver']=receiver_mesh(scene,candidate)
            proof=candidate['receiver'].get('visible_support',[])
            pose=candidate['terrain_physics'];updated=projected_gap(pose,candidate)
            rgb=np.asarray(scene['frame']).copy();mask=np.zeros(scene['protected'].shape,bool)
            sites=[s for s in scene['supports'] if s['id'] in candidate['support_ids']]
            for site in sites:
                if support_plane_evidence(site)['eligible']:mask|=visible_support_mask(scene,site)[0]
            x,y,r,b=candidate['bounds'];bounded=np.zeros_like(mask);bounded[y:b,x:r]=mask[y:b,x:r]
            rgb[bounded]=np.uint8(rgb[bounded]*.7+np.array([20,210,150])*.3)
            diagnostic=Image.fromarray(rgb);draw=ImageDraw.Draw(diagnostic)
            points=np.asarray(pose.get('contact_points',[]))
            xy=project(points,candidate['camera'])*candidate['full_px']+candidate['offset']
            for px,py in xy:
                ix,iy=int(round(px)),int(round(py));valid=0<=iy<mask.shape[0] and 0<=ix<mask.shape[1] and mask[iy,ix]
                draw.ellipse((px-2,py-2,px+2,py+2),fill='#ffd75c' if valid else '#ff4268')
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',32)
            draw.text((32,150),f"{row['asset_id']}：可见支持子集（绿）；接触投影（黄/红）；实际高度仍待审。",font=font,fill='white',stroke_width=2,stroke_fill='black')
            path=out/(row['job_id']+'-visible-support.png');diagnostic.save(path)
            rows.append({'job_id':row['job_id'],'asset_id':row['asset_id'],'background':row['background'],
                'old_image':row['image'],'previous_contact_gap':candidate.get('contact_gap'),
                'visible_contact_audit':updated,'receiver_visible_evidence':proof,
                'support_labels':[{'id':s['id'],'label':s.get('label'),'kind':s.get('geometry_kind','unknown'),
                                   'plane_evidence':support_plane_evidence(s)} for s in sites],
                'diagnostic':str(path),'new_simulation':False})
            print(json.dumps({'audited':row['job_id'],'visible_contact_status':updated['status']},ensure_ascii=False),flush=True)
    summary={'batch':str(batch),'generation_state':'user_paused','images_audited':len(rows),
        'contact_statuses':{status:sum(r['visible_contact_audit']['status']==status for r in rows) for status in ('pass','unknown','fail')},
        'new_API_requests':0,'new_render_count':0,'new_physics_simulations':0,'seconds':time.monotonic()-started,
        'scope':'existing poses only; actual support heights remain unverified; no updated composite delivered','jobs':rows}
    atomic_json(out/'contact-audit.json',summary)
    body=['<!doctype html><meta charset="utf-8"><title>实际可见支持证据审计</title><style>body{font:16px system-ui;background:#171d22;color:#eee;margin:24px}img{width:100%;height:auto}section{margin:32px 0}pre{white-space:pre-wrap}</style>',
          '<h1>暂停期间的可见支持证据审计</h1><p>仅重新核对旧图落点，不生成新合成图。绿色表示局部深度连续的可见支持子集；接触投影黄色在子集内、红色在子集外。可见覆盖不证明真实接触高度。</p>',
          '<pre>'+html.escape(json.dumps({k:v for k,v in summary.items() if k!='jobs'},ensure_ascii=False,indent=2))+'</pre>']
    for row in rows:
        body.append('<section><h2>'+html.escape(row['job_id'])+'</h2>')
        for label,path in [('原交付完整图（仅作审计）',row['old_image']),('完整画面接触证据诊断',row['diagnostic'])]:
            body.append('<p>'+label+'</p><img loading="lazy" src="'+Path(path).resolve().as_uri()+'">')
        body.append('</section>')
    (out/'index.html').write_text(''.join(body),encoding='utf8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();result=asyncio.run(audit(resolve(args.batch),resolve(args.out)))
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'},ensure_ascii=False))
