"""Numbered feedback gallery and independent saved-PNG audit; no image corrections."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from display_batch.common import atomic_json,resolve,load_catalog,file_hash
from display_batch.costs import cumulative,label


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def report(out):
    rows=[json.loads(line) for line in (out/'results.jsonl').read_text(encoding='utf8').splitlines()]
    rows.sort(key=lambda r:r['job_id']);catalog=load_catalog();summary=read(out/'summary.json')
    cost=cumulative(out);summary['cost_total']=cost
    atomic_json(out/'summary.json',summary)
    audit=[];cards=[];tiles=[]
    reason_names={'assembly_relationship_unknown_or_loose':'部件连接关系待确认',
        'no_candidate_survived_program_checks':'候选未通过程序检查','iterations_exhausted':'本轮已完成一次复核，等待点评',
        'visual_ok_evidence_unresolved':'视觉复核通过，物理依据仍待确认'}
    for row in rows:
        number=int(row['job_id'].split('-')[0]);name=catalog[row['asset_id']]['name']
        title=f'{number:02d} · {row["asset_id"]} {name}'
        entry={'number':number,'asset_id':row['asset_id'],'status':row['status'],'png':row.get('image')}
        candidate=None
        if row.get('image'):
            best=next(i for i in row['iterations'] if i['iteration']==row['best_iteration'])
            with Image.open(row['background']) as im:bg=np.asarray(im.convert('RGB'))
            with Image.open(row['image']) as im:actual=np.asarray(im.convert('RGB'))
            l,t,r,b=best['edit_bounds'];changed=np.any(bg!=actual,axis=2)
            changed[t:b,l:r]=False
            entry.update(size_ok=actual.shape==bg.shape,outside_rectangle_changes=int(changed.sum()),
                png_hash_ok=file_hash(row['image'])==best['image_hash'],
                protected_changes=best['checks']['protected']['changed_pixels'],
                foreground_changes=best['checks']['occlusion']['foreground_changed_pixels'],
                outside_edit_changes=best['checks']['outside_edit']['changed_pixels'],
                anchor_error=best['checks']['anchor']['error_px'],
                optical_iou=best['photo'].get('optical_iou'),
                local_air=best['photo'].get('local_air'),chroma_before=best['photo'].get('source_chroma'),
                chroma_after=best['photo'].get('after_chroma'),chroma_environment=best['photo'].get('scene_chroma'))
            candidates=read(out/'jobs'/row['job_id']/'candidates.json')['candidates']
            candidate=next(c for c in candidates if c['id']==best['params']['candidate'])
            if candidate.get('user_scale'):
                user_scale=candidate['user_scale'];baseline=candidate.get('feedback_baseline',{})
                entry['scale_change']={**user_scale,'current_full_px':candidate['full_px'],
                    'current_expected_px':candidate['expected_major_px'],
                    'measured_unoccluded_px':best['checks']['scale']['measurement_px'],
                    'ratio':candidate['expected_major_px']/user_scale['original_expected_major_px'],
                    'old_measured_visible_px':baseline.get('measured_major_px')}
            image=Path(row['image']).relative_to(out).as_posix()
            context=Path(best['image']).with_suffix('.context.jpg')
            if not context.is_file():context=Path(row['crop'])
            crop=context.relative_to(out).as_posix()
            review=best['review'];details='；'.join(d['evidence'] for d in review['defects']) or review['evidence']
            photos=f"局部扬尘混色估计 {100*best['photo']['local_air']['amount']:.1f}% · 调色阶段色度 {best['photo']['source_chroma']:.1f} → {best['photo']['after_chroma']:.1f}"
            cards.append(f'<section id="n{number:02d}"><h2>{html.escape(title)}</h2>'
                f'<p>{html.escape(row["status"]+" · "+photos)}</p><a href="{image}">打开原尺寸 PNG</a>'
                f'<div><a href="{image}"><img src="{image}" loading="lazy"></a><img src="{crop}" loading="lazy"></div>'
                f'<p>自动复核：{html.escape(details)}</p><p>{html.escape(label(row.get("cost_total_at_output")))}</p></section>')
            tiles.append((number,title,str(context),row['status']))
            atomic_json(Path(row['image']).with_suffix('.cost.json'),{
                'batch_total':cost,'total_at_output':row.get('cost_total_at_output'),'job':row['cost'],
                'compute_seconds':row.get('timings'),'note':'API 公开价估算，含重试；本机电费和设备折旧未定价'})
        else:
            message=reason_names.get(row.get('reason'),row.get('reason','未出图'))
            cards.append(f'<section id="n{number:02d}"><h2>{html.escape(title)}</h2><p>未出图：{html.escape(message)}</p></section>')
            tiles.append((number,title,None,message))
        audit.append(entry)
    before=read(out/'source-hashes-before.json');after=read(out/'source-hashes-after.json')
    result={'images':len([a for a in audit if a['png']]),'planned':len(rows),'source_models_unchanged':before==after,
        'source_model_count':len(before),'items':audit,'cost_total':cost,
        'checks_pass':all(a.get('size_ok',True) and a.get('png_hash_ok',True) and
            not any(a.get(k,0) for k in ('outside_rectangle_changes','protected_changes','foreground_changes','outside_edit_changes')) for a in audit)}
    atomic_json(out/'audit.json',result)
    generation=read(out/'generation-summary.json') if (out/'generation-summary.json').is_file() else summary
    atomic_json(out/'cost-ledger.json',{'batch_total':cost,'preparation':read(out/'preparation.json'),
        'generation':generation.get('run'),'latest_run':summary.get('run'),
        'by_job':[{'number':a['number'],'job':r['cost'],'seconds':r.get('seconds')} for a,r in zip(audit,rows)]})
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    tilemap={t[0]:t for t in tiles};count=len(read(out/'selection.json')['asset_ids']);overview=[]
    for start in range(1,count+1,6):
        canvas=Image.new('RGB',(1440,120+min(6,count-start+1)*330),'#172128');draw=ImageDraw.Draw(canvas)
        draw.text((22,16),'低饱和与局部扬尘 · 点评样片（按编号反馈）',font=font,fill='white')
        draw.text((22,57),label(cost),font=small,fill='#bfd0d8')
        for line,assetno in enumerate(range(start,min(start+6,count+1))):
            for col,num in enumerate((assetno,assetno+count)):
                if num not in tilemap:continue
                _,title,picture,status=tilemap[num];x=col*720+18;y=105+line*330
                draw.text((x,y),title,font=font,fill='white')
                if picture:
                    with Image.open(picture) as im:
                        im=im.convert('RGB');im.thumbnail((684,280));canvas.paste(im,(x,y+38))
                else:draw.text((x,y+90),status[:34],font=small,fill='#edbb7d')
        path=out/f'overview-{1+(start-1)//6:02d}.jpg';canvas.save(path,quality=92);overview.append(path.name)
    page='<!doctype html><meta charset="utf-8"><title>物品点评样片</title><style>body{background:#172128;color:#eee;font:16px sans-serif;margin:28px}a{color:#8bd5b7}section{border-top:1px solid #456;padding:20px 0}img{max-width:49%;max-height:650px}p{line-height:1.7}nav{position:sticky;top:0;background:#172128;padding:12px}nav a{margin-right:15px}</style>'
    page+='<h1>12 类物品 × 2 张背景 · 点评样片</h1><p>31 号液压缸缩小一半；05–07 储罐不再生成。低饱和与扬尘只依据落点附近样本。每张成图已自动复核，待审不等于通过。</p>'
    page+='<p>'+html.escape(label(cost))+'</p><p>本机计算按时间记录，未虚构电费或订阅成本。费用包含本次准备、重试和生成，不包含历史缓存费用。</p>'
    page+='<nav>'+''.join(f'<a href="#n{n:02d}">{n:02d}</a>' for n in range(1,len(rows)+1))+'</nav>'
    page+=''.join(f'<a href="{p}">总览 {i+1}</a> ' for i,p in enumerate(overview))+''.join(cards)
    (out/'feedback.html').write_text(page,encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    report(resolve(parser.parse_args().out))
