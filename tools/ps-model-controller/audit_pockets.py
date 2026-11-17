"""Full-frame pocket/contact diagnostics for historical PNGs; no API, render or simulation."""
import argparse
import asyncio
import html
from pathlib import Path
import time
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from display_batch.common import resolve,atomic_json
from display_batch.geometry import project,transformed,contact_world
from display_batch.pile import visible_support_mask,support_plane_evidence
from display_batch.scene import Scenes,projection_geometry,fitted_masks
from display_batch.reference_projection import metric_projection
from scrap_composite.pockets import propose,check_projection
from audit_visible_contacts import FrozenSceneAnswers
from validate_reference_projection import read,records
from verify_geometry_rebuild import source_job
from verify_coherent_expanded import cached_perception,Perception


async def audit(batch,out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    cache=resolve(read(batch/'config.json')['cache']);scenes=Scenes(cache,FrozenSceneAnswers())
    rows=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    with patch.object(Perception,'scene',cached_perception(Perception.scene)):
        for row in records(batch):
            if not row.get('image'):continue
            scene=await scenes.prepare(row['background']);entry=next(e for e in row['iterations'] if e['iteration']==row['best_iteration'])
            old=next(c for c in read(source_job(batch,row)/'candidates.json')['candidates'] if c['id']==entry['params']['candidate'])
            asset=read(cache/'assets'/row['asset_signature']/'result-strict.json');data=np.load(asset['geometry'])
            vertices,faces=data['vertices'],data['faces'];normal=old['plane_normal'];matrix=old['matrix']
            points=transformed(vertices,matrix);patches=contact_world(points,faces,normal=normal)
            contact=np.concatenate(patches);center=(contact.min(0)+contact.max(0))/2
            calibration=old['scale']['calibration'];millimeters=old['scale']['physical_size']['nominal_mm']
            diagnostic=scene['frame'].copy();draw=ImageDraw.Draw(diagnostic,'RGBA')
            site_records=[];proposals=[]
            sites=sorted(scene['supports'],key=lambda s:(bool(s.get('normal_inherited')), -s['confidence']))[:8]
            for site in sites:
                evidence=support_plane_evidence(site)
                if not evidence['eligible']:
                    site_records.append({'site_id':site['id'],'label':site.get('label'),'reason':'nonplanar_no_independent_bottom','evidence':evidence});continue
                mask,visible=visible_support_mask(scene,site)
                camera,full=metric_projection(calibration,site['anchor_px'],millimeters,scene['size'],center)
                span=float(np.ptp(project(points,camera),axis=0).max()*full)
                pockets,summary=propose(scene['depth_native'],mask,scene['allowed'],scene['protected'],span,max_candidates=2)
                site_records.append({'site_id':site['id'],'label':site.get('label'),'visible_support':visible,'pocket_analysis':summary})
                for pocket in pockets:
                    anchor=pocket['anchor_px'];camera,full=metric_projection(calibration,anchor,millimeters,scene['size'],center)
                    geom=projection_geometry(vertices,faces,{'matrix':matrix},0,old['elevation'],normal,
                                             patches=patches,camera=camera,raster_size=256)
                    fit=fitted_masks(geom,anchor,full,scene)
                    if fit is None:
                        proposals.append({'site_id':site['id'],'pocket':pocket,'projection':{'program_fit':'fail','reasons':['outside_image']}});continue
                    x,y,r,b=fit['bounds']
                    result=check_projection(pocket,fit['hull'],fit['contact'],fit['bounds'],mask[y:b,x:r],
                                            scene['allowed'][y:b,x:r],scene['protected'][y:b,x:r])
                    record={'site_id':site['id'],'label':site.get('label'),'pocket':pocket,'projection':result,
                        'bounds':fit['bounds'],'silhouette_full_pixel_extent':float(np.ptp(geom['uv'],axis=0).max()*full),
                        'pose_basis':'unchanged_historical_pose_for_projection_only','scale_basis':'historical_estimated_camera_with_fixed_physical_size',
                        'physical_size_mm':millimeters,'new_simulation':False}
                    # Store coordinates, not replacement masks. All visuals are diagnostic overlays.
                    contours,_=cv2.findContours(fit['hull'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    record['outline_pixels']=[(c[:,0]+[x,y]).tolist() for c in contours if len(c)>2]
                    record['contact_polygons_px']=[(p*full+fit['offset']).tolist() for p in geom['contacts']]
                    proposals.append(record)
            ranked=sorted(proposals,key=lambda p:(p['projection']['program_fit']!='pass',-p['pocket']['score']))
            selected=[];seen=set()
            for proposal in ranked:
                if proposal['site_id'] in seen:continue
                selected.append(proposal);seen.add(proposal['site_id'])
                if len(selected)==3:break
            for index,p in enumerate(selected,1):
                color=(80,230,170,255) if p['projection']['program_fit']=='pass' else (255,90,100,255)
                for outline in p.get('outline_pixels',[]):draw.line([tuple(v) for v in outline],fill=color,width=4)
                for polygon in p.get('contact_polygons_px',[]):
                    if len(polygon)>2:draw.polygon([tuple(v) for v in polygon],fill=(255,190,40,120))
                anchor=p['pocket']['anchor_px'];x,y=anchor
                draw.ellipse((x-12,y-12,x+12,y+12),outline=color,width=4)
                draw.text((x+16,y+8),f"{index} · {p['site_id']} · 投影{p['projection']['program_fit']}",font=font,fill=color,stroke_width=2,stroke_fill='black')
                for scale in p['pocket']['rim_evidence']:
                    for rx,ry in scale['rim_points_px']:draw.ellipse((rx-4,ry-4,rx+4,ry+4),fill=(70,170,255,255))
            ax,ay=old['anchor_px'];draw.line((ax-14,ay-14,ax+14,ay+14),fill=(255,255,255,255),width=4)
            draw.line((ax-14,ay+14,ax+14,ay-14),fill=(255,255,255,255),width=4)
            draw.text((32,150),f"{row['asset_id']} 凹处候选诊断；白叉=旧锚点，蓝点=挡边证据。只是投影检查，未仿真/未出新合成。",font=font,fill='white',stroke_width=2,stroke_fill='black')
            path=out/(row['job_id']+'-pockets.png');diagnostic.save(path)
            item={'job_id':row['job_id'],'asset_id':row['asset_id'],'old_image':row['image'],'background':row['background'],
                'old_anchor_px':old['anchor_px'],'physical_size_mm':millimeters,'diagnostic':str(path),
                'sites':site_records,'proposals':proposals,'displayed_candidates':selected,
                'pocket_proposals':len(proposals),'projection_fit_count':sum(p['projection']['program_fit']=='pass' for p in proposals)}
            rows.append(item);print(f"{row['job_id']}: {len(proposals)} pockets, {item['projection_fit_count']} projection fits",flush=True)
    result={'generation_state':'user_paused','images':len(rows),'jobs':rows,'new_api_requests':0,'new_render_count':0,
        'new_physics_simulations':0,'seconds':time.monotonic()-started,
        'images_with_pockets':sum(r['pocket_proposals']>0 for r in rows),
        'images_with_projection_fit':sum(r['projection_fit_count']>0 for r in rows),
        'note':'Historical cameras/scales remain estimated. Projection fit is not physical contact or visual acceptance.'}
    atomic_json(out/'pockets.json',result)
    header='<style>body{font:16px system-ui;background:#171c22;color:#eee;margin:24px}img{width:100%;height:auto}section{margin:40px 0}pre{white-space:pre-wrap}</style>'
    body=['<!doctype html><meta charset="utf-8"><title>16张凹处/接触诊断</title>'+header,
          '<h1>16 张历史完整画面凹处/接触诊断</h1><p>批量保持暂停。以下不含新合成图；绿线仅表示真实网格投影适配，未证明真实承托。</p>']
    for row in rows:
        body.append('<section><h2>'+html.escape(row['job_id'])+f" · 凹处候选 {row['pocket_proposals']} / 投影适配 {row['projection_fit_count']}</h2>")
        for label,path in [('历史完整图',row['old_image']),('完整画面凹处与真实轮廓/接触投影诊断',row['diagnostic'])]:
            body.append('<p>'+label+'</p><img loading="lazy" src="'+Path(path).resolve().as_uri()+'">')
        body.append('</section>')
    (out/'index.html').write_text(''.join(body),encoding='utf8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();asyncio.run(audit(resolve(a.batch),resolve(a.out)))
