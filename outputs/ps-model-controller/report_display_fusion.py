"""Publish all frozen regression evidence and separate image rate, quality and actual usage."""
import argparse
from collections import Counter,defaultdict
import html
import json
import os
from pathlib import Path
import statistics
from display_batch.common import ROOT,atomic_json,file_hash
from display_batch.schema import Review
from report_display_batch import report as batch_report


def usage(folder):
    files=list(Path(folder).rglob('usage-*.json'));rows=[]
    for path in files:
        data=json.loads(path.read_text(encoding='utf8'))
        rows.append({'file':str(path),'api_calls':data['api_calls'],'reported_tokens':data.get('reported_tokens',0),
                     'unreported_calls':data.get('unreported_calls',0)})
    return {'api_calls':sum(r['api_calls'] for r in rows),'reported_tokens':sum(r['reported_tokens'] for r in rows),
            'unreported_calls':sum(r['unreported_calls'] for r in rows),'files':rows}


def build(root,photo,history):
    root=Path(root).resolve();photo=Path(photo).resolve()
    for stage in ('recovery','regression'):
        state=json.loads((root/stage/'summary.json').read_text(encoding='utf8'))['run']['state']
        if state!='complete':raise ValueError(f'{stage} is {state}; final report requires completed batches')
    regression=batch_report([root/'regression'],root/'regression-gallery')
    recovery=batch_report([root/'recovery'],root/'recovery-gallery')
    p=json.loads((photo/'results.json').read_text(encoding='utf8'))
    reductions=[r['comparison']['excess_reduction'] for r in p['records'] if r['comparison']['excess_reduction'] is not None]
    versions=[]
    for folder in map(Path,history):
        if (folder/'results.json').is_file():
            data=json.loads((folder/'results.json').read_text(encoding='utf8'))
            versions.append({'path':str(folder.resolve()),'images':data['images'],'photo_pass':data['photo_pass'],
                             'rate':data['photo_pass_rate'],'last_run_seconds':data['seconds'],
                             'resume_count':len(list(folder.glob('usage-*.json')))-1,'usage':usage(folder)})
    selected_usage={str(Path(h).resolve()):usage(h) for h in history}
    # The validation root also includes recovery and regression usage, with no cache calls counted.
    selected_usage[str(root)]=usage(root)
    counts=Counter();placements=Counter();unknown=Counter();failures=Counter()
    quality=defaultdict(lambda:{'tasks':0,'images':0,'statuses':Counter(),'scores':defaultdict(list),'defects':Counter(),'visual_pass':0})
    gate_counts=Counter()
    lines=(root/'regression/results.jsonl').read_text(encoding='utf8').splitlines()
    regression_rows={}
    for line in lines:
        row=json.loads(line);counts[row['status']]+=1
        regression_rows[(row['asset_id'],str(Path(row['background']).resolve()).casefold())]=row
        q=quality[row['asset_id']];q['tasks']+=1;q['images']+=bool(row.get('image'));q['statuses'][row['status']]+=1
        selected=next((r for r in row.get('iterations',[]) if r['iteration']==row.get('best_iteration')),None)
        if selected:
            for name,value in selected.get('review',{}).get('scores',{}).items():q['scores'][name].append(value)
            for defect in selected.get('review',{}).get('defects',[]):q['defects'][defect['code']]+=1
        if row.get('image'):
            checks=row.get('checks',{});states=[c['status'] for c in checks.values()]
            program='fail' if 'fail' in states else 'pass' if states and all(s=='pass' for s in states) else 'unknown'
            reviewed=selected and selected.get('review')
            visual=('pass' if Review.model_validate(reviewed).passes() else 'fail') if reviewed else 'unknown'
            gate_counts[f'visual_{visual}_program_{program}']+=1
            q['visual_pass']+=visual=='pass'
        else:gate_counts['without_image']+=1
        if row.get('image'):placements[row.get('placement','surface')]+=1
        for name,c in row.get('checks',{}).items():
            if c['status']=='unknown':unknown[name]+=1
            if c['status']=='fail':failures[name]+=1
    total=len(lines);images=regression['generated_pngs']
    paired=[]
    for line in (root/'recovery/results.jsonl').read_text(encoding='utf8').splitlines():
        previous=json.loads(line)
        current=regression_rows.get((previous['asset_id'],str(Path(previous['background']).resolve()).casefold()))
        if previous.get('image') and current and current.get('image'):
            paired.append({'recovery_job':previous['job_id'],'regression_job':current['job_id'],
                           'asset_id':current['asset_id'],'identical_png':file_hash(previous['image'])==file_hash(current['image'])})
    quality={asset:{'tasks':q['tasks'],'images':q['images'],'statuses':dict(q['statuses']),'visual_pass':q['visual_pass'],
                   'mean_scores':{k:statistics.mean(v) for k,v in q['scores'].items()},
                   'defects':dict(q['defects'])} for asset,q in quality.items()}
    audits={}
    for name,path in [('png_readback',root/'png-audit.json'),('render_resolution',root/'render-audit/checks.json'),
                      ('asset_attributes',root/'asset-audit.json'),('source_model_hashes',root/'asset-hash-check.json')]:
        if path.is_file():
            data=json.loads(path.read_text(encoding='utf8'))
            audits[name]={'file':str(path),**{k:v for k,v in data.items() if k not in ('records','checks')}}
            if name=='render_resolution':
                audits[name].update(anchor_passed=sum(r['anchor_pass'] for r in data['checks']),
                                   alpha_passed=sum(r['alpha_pass'] for r in data['checks']))
    result={'validation':'cross_background_regression_not_unseen_holdout','tasks':total,'counts':dict(counts),
        'generated_pngs':images,'image_rate':images/max(1,total),'accepted_display_rate':counts['accepted_display']/max(1,total),
        'image_target_met':images/max(1,total)>=.9,'display_target_met':counts['accepted_display']/max(1,total)>=.8,
        'photo':{'images':p['images'],'photo_pass':p['photo_pass'],'rate':p['photo_pass_rate'],'target_met':p['photo_pass_rate']>=.8,
                 'colored_cases':len(reductions),'excess_chroma_reduction_median':statistics.median(reductions) if reductions else None,
                 'cases_reduced_at_least_half':sum(v>=.5 for v in reductions),'versions':versions},
        'recovery':recovery,'regression':regression,'placements':dict(placements),'unknown_checks':dict(unknown),
        'quality_by_asset':quality,'audits':audits,
        'visual_and_program_gates':dict(gate_counts),
        'paired_reuse':{'comparable_images':len(paired),'identical_pngs':sum(r['identical_png'] for r in paired),'records':paired},
        'failed_checks_in_delivered_results':dict(failures),'usage':selected_usage,
        'all_experiments_actual_api_calls':sum(u['api_calls'] for u in selected_usage.values()),
        'all_experiments_reported_tokens':sum(u['reported_tokens'] for u in selected_usage.values()),
        'unreported_calls':sum(u['unreported_calls'] for u in selected_usage.values()),'money_cost':None,
        'limitations':['finite coplanar support proxies, not measured irregular pile',
                      'relative depth and scale are estimates; unknown is review',
                      'all 23 backgrounds were seen during development',
                      'queue simulation is not real 1000-image generation']}
    atomic_json(root/'delivery-summary.json',result)
    def link(path):return html.escape(Path(os.path.relpath(path,root)).as_posix(),quote=True)
    cards=[('固定摆位摄影对照',photo/'index.html'),('48项恢复与承托/遮挡诊断',root/'recovery-gallery/index.html'),
           ('138项跨背景完整总览',root/'regression-gallery/index.html'),('独立PNG回读',root/'png-audit.json'),
           ('跨分辨率轮廓与锚点检查',root/'render-audit/checks.json'),('实测记录',root/'delivery-summary.json')]
    links=''.join(f'<a href="{link(path)}">{label}</a>' for label,path in cards)
    resolution=audits.get('render_resolution',{})
    check_text=(f"六类缓存渲染核查：锚点 {resolution.get('anchor_passed',0)}/{resolution.get('checked',0)} 通过；"
                f"轮廓 {resolution.get('alpha_passed',0)}/{resolution.get('checked',0)} 通过，未通过项保留原门槛与差异图。") if resolution else '跨分辨率渲染核查尚无记录。'
    text=f'''<!doctype html><meta charset="utf-8"><title>摄影融合与废钢承托验证</title>
<style>body{{max-width:1100px;margin:36px auto;padding:20px;background:#172128;color:#e7eef0;font:17px sans-serif;line-height:1.7}}a{{color:#9fdcc6;display:block;padding:10px 0}}img{{max-width:100%}}strong{{color:#edbe63}}</style>
<h1>摄影融合与废钢承托验证</h1><p>完整回归：{total} 项。已出图 <strong>{images} 张（{result['image_rate']:.1%}）</strong>；
展示合格 <strong>{counts['accepted_display']} 张（{result['accepted_display_rate']:.1%}）</strong>；待审 {counts['review']} 项，执行失败 {counts['failed']} 项。</p>
<p>固定摆位摄影评分：{p['photo_pass']}/{p['images']}（{p['photo_pass_rate']:.1%}）。未知证据不等于物理通过，出图率不等于展示合格率。</p>
{links}<p>阴影/承托采用图像估计，隐藏高度仍有不确定性。现有23张背景均已参与开发，本轮是跨背景回归。</p>
<p>{check_text}</p>
<p>本次全部列入统计的实验：{result['all_experiments_actual_api_calls']} 次实际 API，{result['all_experiments_reported_tokens']} 已报告 token；未配置价格，不估算费用。</p>
<img src="regression-gallery/overview-1.jpg" alt="按清单顺序展示，包含待审与失败项">'''
    (root/'index.html').write_text(text,encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k in ('tasks','counts','generated_pngs','image_rate','accepted_display_rate','all_experiments_actual_api_calls','all_experiments_reported_tokens')},ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--photo',required=True)
    p.add_argument('--history',nargs='+',required=True)
    a=p.parse_args();build(a.root,a.photo,a.history)
