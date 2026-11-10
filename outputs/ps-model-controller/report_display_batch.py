"""Build a complete, non-cherry-picked cross-background display evaluation gallery."""
import argparse
from collections import Counter,defaultdict
import html
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont,ImageOps

from display_batch.common import atomic_json,file_hash

REASONS={
    'display_checks_passed':'展示检查通过',
    'no_supported_lying_candidate':'未找到满足支撑与尺度条件的位置',
    'no_geometric_candidate':'候选触及边界、保护区或缺少可用接触依据',
    'no_candidate_survived_local_physics':'局部落稳或接触复验未通过',
    'asset_structure_requires_review':'模型结构需要复核',
    'iterations_exhausted':'已达到修正轮数上限',
    'repeated_parameters':'建议参数重复，已停止',
    'no_improvement':'连续修正没有改善',
    'no_effective_image_change':'参数变化未改变实际图片，停止重复复核',
    'final_program_checks_failed':'最终尺寸图片未通过程序检查',
    'no_valid_action':'没有可执行的修正建议',
    'invalid_ai_action':'模型建议超出允许范围',
}
GROUP_LABELS={'development':'开发集历史验证','heldout':'留出集历史验证',
              'regression_53':'相机53回归','regression_36':'相机36回归','recovery':'原拒绝任务复验'}


def scale_diagnostic(record,job,selected):
    candidates=json.loads((job/'candidates.json').read_text(encoding='utf8'))['candidates']
    candidate=next(c for c in candidates if c['id']==selected['params']['candidate'])
    geometry=job/(candidate['id']+'-geometry.jpg')
    if not geometry.is_file():return None
    with Image.open(record['background']) as original:width,height=original.size
    image=Image.open(geometry).convert('RGB');image.load()
    canvas=Image.new('RGB',(image.width,image.height+110),(23,31,38));canvas.paste(image,(0,0))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    x,y=candidate['anchor_px'];x*=image.width/width;y*=image.height/height
    draw.ellipse((x-6,y-6,x+6,y+6),fill=(255,80,200))
    scale=candidate['scale'];reference=scale.get('reference')
    if reference:
        points=[(u*image.width,v*image.height) for u,v in reference]
        draw.line(points,fill=(255,80,200),width=4)
    text=f"{record['asset_id']} · {record['status']} · 未遮挡投影 {selected['checks']['scale']['measurement_px']:.1f} px · 遮挡 {selected.get('occlusion_fraction',0):.1%}"
    draw.text((16,image.height+8),text,font=font,fill='white')
    mm=scale.get('reference_mm');detail=f"参照尺寸区间 {mm} mm（估计）" if mm else '缺少同层尺寸参照：仅作展示估计'
    draw.text((16,image.height+40),detail,font=font,fill=(237,190,99))
    draw.text((16,image.height+72),'绿色：承托片；橙色：接触斑；粉色：接触锚点/尺度参照。隐藏高度并非实测。',font=font,fill=(195,210,215))
    target=job/'support-scale-inspection.jpg';canvas.save(target,quality=90);return target


def report(batches,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    records=[];runs=[]
    for folder in map(Path,batches):
        summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
        runs.append({'path':str(folder.resolve()),'counts':summary['counts'],'groups':summary['groups'],
                     'run':summary['run'],'api_calls':summary['usage_current_run']['api_calls'],
                     'tokens':summary['usage_current_run'].get('total_tokens'),
                     'implementation_hash':file_hash(folder/'implementation.json')})
        records.extend({**json.loads(line),'_batch_folder':str(folder.resolve())}
                       for line in (folder/'results.jsonl').read_text(encoding='utf8').splitlines())
    groups={};failures=Counter();by_background=defaultdict(list)
    for record in records:
        g=groups.setdefault(record['group'],Counter());g['total']+=1;g[record['status']]+=1
        g['images']+=bool(record.get('image'))
        if record['status']!='accepted_display':failures[record['reason']]+=1
        by_background[(record['group'],record['background'])].append(record)
    groups={k:{**v,'acceptance_rate':v['accepted_display']/v['total'],'image_rate':v['images']/v['total']} for k,v in groups.items()}
    groups={k:{**v,'target_rate':.8,'target_met':v['acceptance_rate']>=.8} for k,v in groups.items()}
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),19);small=ImageFont.truetype(str(font_path),16)
    colors={'accepted_display':(96,209,153),'review':(237,190,99),'failed':(242,120,120)}
    labels={'accepted_display':'展示合格','review':'待审','failed':'执行失败'}
    sections=[];overview=[];pixel_checks=Counter();angles=[];anchor_errors=[]
    for index,((group,background),items) in enumerate(by_background.items(),1):
        items.sort(key=lambda r:('3','6','10','13','19-1','31').index(r['asset_id']))
        sheet=Image.new('RGB',(1800,490),(23,31,38));draw=ImageDraw.Draw(sheet)
        draw.text((18,14),f'{index:02} · {GROUP_LABELS.get(group,group)} · {Path(background).name}',font=small,fill='white')
        draw.text((18,42),'背景原图 · 网页中点击编号可查看完整合成',font=small,fill=(180,194,202))
        with Image.open(background) as original:
            original.thumbnail((560,390));sheet.paste(original,(18,65))
        for i,r in enumerate(items):
            x=600+(i%3)*400;y=55+(i//3)*215
            draw.text((x,y),f'{r["asset_id"]} · {labels[r["status"]]}',font=font,fill=colors[r['status']])
            if r.get('crop'):
                with Image.open(r['crop']) as crop:
                    crop=ImageOps.contain(crop.convert('RGB'),(385,175));sheet.paste(crop,(x,y+32))
            else:
                reason=REASONS.get(r['reason'],r['reason'])
                draw.multiline_text((x,y+50),'未生成合成图\n'+reason[:22],font=small,fill=(190,190,190))
            for name,check in r.get('checks',{}).items():
                pixel_checks[name+'_'+check['status']]+=1
            angle=r.get('checks',{}).get('lying',{}).get('measurement_deg')
            if angle is not None:angles.append(angle)
            error=r.get('checks',{}).get('anchor',{}).get('projection_error_px')
            if error is not None:anchor_errors.append(error)
        image_name=f'background-{index:02}.jpg';sheet.save(out/image_name,quality=92)
        thumbnail=sheet.resize((1200,327),Image.Resampling.LANCZOS);overview.append(thumbnail)
        links=[]
        for r in items:
            title=html.escape(r['asset_id']+' '+labels[r['status']]+' · '+REASONS.get(r['reason'],r['reason']))
            if r.get('image'):
                # Relative filesystem paths also work when the report directory is served locally.
                import os
                target=Path(os.path.relpath(r['image'],out)).as_posix()
                links.append(f'<a href="{html.escape(target,quote=True)}">{title}</a>')
            else:links.append('<span>'+title+'</span>')
            import os
            job=Path(r['_batch_folder'])/'jobs'/r['job_id']
            artifacts=[('候选与拒绝依据',job/'candidates.json'),('完整记录',job/'result.json')]
            if r.get('best_iteration') is not None:
                selected=next(v for v in r['iterations'] if v['iteration']==r['best_iteration'])
                artifacts.extend([('承托与接触斑',job/(selected['params']['candidate']+'-geometry.jpg')),
                    ('前后遮挡',Path(selected['image']).with_suffix('.occlusion.png'))])
                marked=scale_diagnostic(r,job,selected)
                if marked:artifacts.append(('尺度与支撑检查图',marked))
            for label,path in artifacts:
                if path.is_file():
                    relative=Path(os.path.relpath(path,out)).as_posix()
                    links.append(f'<a href="{html.escape(relative,quote=True)}">{r["asset_id"]} {label}</a>')
        sections.append(f'<section><img loading="lazy" src="{image_name}"><div>'+''.join(links)+'</div></section>')
    pages=[]
    for start in range(0,len(overview),6):
        page=Image.new('RGB',(1200,327*len(overview[start:start+6])),(23,31,38))
        for j,im in enumerate(overview[start:start+6]):page.paste(im,(0,j*327))
        name=f'overview-{start//6+1}.jpg';page.save(out/name,quality=90);pages.append(name)
    stats={'kind':'actual_generation','groups':groups,'runs':runs,'failure_reasons':dict(failures),
           'tasks':len(records),'generated_pngs':sum(bool(r.get('image')) for r in records),
           'final_checks':dict(pixel_checks),'max_axis_angle_deg':max(angles,default=None),
           'max_anchor_projection_error_px':max(anchor_errors,default=None),
           'formal_acceptance':'unchanged','overview_pages':pages}
    atomic_json(out/'validation.json',stats)
    table='<table><tr><th>分组</th><th>任务</th><th>展示合格</th><th>待审</th><th>失败</th><th>有图率</th><th>合格率 / 目标</th></tr>'
    for key,g in groups.items():
        label=GROUP_LABELS.get(key,key)
        table+=f'<tr><td>{label}</td><td>{g["total"]}</td><td>{g.get("accepted_display",0)}</td><td>{g.get("review",0)}</td><td>{g.get("failed",0)}</td><td>{g["image_rate"]:.1%}</td><td>{g["acceptance_rate"]:.1%} / 80%</td></tr>'
    page='''<!doctype html><meta charset="utf-8"><title>跨背景批量验证</title><style>
body{background:#171f26;color:#e8eef2;font:16px system-ui;margin:30px auto;max-width:1800px}h1{font-size:28px}p{line-height:1.8}img{width:100%}section{margin-top:28px}a,span{display:inline-block;margin:8px;color:#a5cddb}table{border-collapse:collapse}td,th{padding:12px 18px;border:1px solid #52616c}</style>
<h1>跨背景批量验证 · 全部任务</h1><p>每张原尺寸 PNG 放置一个物品。以下展示每个任务的程序最佳版本；没有成图的任务保留原因。场景几何、尺度和光照为估计；展示合格不改变旧流程正式验收。</p>'''+table+'</table><p><a href="validation.json">完整统计</a></p>'+''.join(sections)
    (out/'index.html').write_text(page,encoding='utf8')
    print(json.dumps({k:v for k,v in stats.items() if k not in ('runs','final_checks')},ensure_ascii=False,indent=2))
    return stats


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batches',nargs='+',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();report(args.batches,args.out)
