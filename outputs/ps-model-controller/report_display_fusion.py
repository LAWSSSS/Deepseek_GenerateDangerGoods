"""Publish all frozen regression evidence and separate image rate, quality and actual usage."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import statistics
from display_batch.common import ROOT,atomic_json
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
    lines=(root/'regression/results.jsonl').read_text(encoding='utf8').splitlines()
    for line in lines:
        row=json.loads(line);counts[row['status']]+=1
        if row.get('image'):placements[row.get('placement','surface')]+=1
        for name,c in row.get('checks',{}).items():
            if c['status']=='unknown':unknown[name]+=1
            if c['status']=='fail':failures[name]+=1
    total=len(lines);images=regression['generated_pngs']
    result={'validation':'cross_background_regression_not_unseen_holdout','tasks':total,'counts':dict(counts),
        'generated_pngs':images,'image_rate':images/max(1,total),'accepted_display_rate':counts['accepted_display']/max(1,total),
        'image_target_met':images/max(1,total)>=.9,'display_target_met':counts['accepted_display']/max(1,total)>=.8,
        'photo':{'images':p['images'],'photo_pass':p['photo_pass'],'rate':p['photo_pass_rate'],'target_met':p['photo_pass_rate']>=.8,
                 'colored_cases':len(reductions),'excess_chroma_reduction_median':statistics.median(reductions) if reductions else None,
                 'cases_reduced_at_least_half':sum(v>=.5 for v in reductions),'versions':versions},
        'recovery':recovery,'regression':regression,'placements':dict(placements),'unknown_checks':dict(unknown),
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
           ('138项跨背景完整总览',root/'regression-gallery/index.html'),('实测记录',root/'delivery-summary.json')]
    links=''.join(f'<a href="{link(path)}">{label}</a>' for label,path in cards)
    text=f'''<!doctype html><meta charset="utf-8"><title>摄影融合与废钢承托验证</title>
<style>body{{max-width:1100px;margin:36px auto;padding:20px;background:#172128;color:#e7eef0;font:17px sans-serif;line-height:1.7}}a{{color:#9fdcc6;display:block;padding:10px 0}}img{{max-width:100%}}strong{{color:#edbe63}}</style>
<h1>摄影融合与废钢承托验证</h1><p>完整回归：{total} 项。已出图 <strong>{images} 张（{result['image_rate']:.1%}）</strong>；
展示合格 <strong>{counts['accepted_display']} 张（{result['accepted_display_rate']:.1%}）</strong>；待审 {counts['review']} 项，执行失败 {counts['failed']} 项。</p>
<p>固定摆位摄影评分：{p['photo_pass']}/{p['images']}（{p['photo_pass_rate']:.1%}）。未知证据不等于物理通过，出图率不等于展示合格率。</p>
{links}<p>阴影/承托采用图像估计，隐藏高度仍有不确定性。现有23张背景均已参与开发，本轮是跨背景回归。</p>
<p>本次全部列入统计的实验：{result['all_experiments_actual_api_calls']} 次实际 API，{result['all_experiments_reported_tokens']} 已报告 token；未配置价格，不估算费用。</p>
<img src="regression-gallery/overview-1.jpg" alt="按清单顺序展示，包含待审与失败项">'''
    (root/'index.html').write_text(text,encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k in ('tasks','counts','generated_pngs','image_rate','accepted_display_rate','all_experiments_actual_api_calls','all_experiments_reported_tokens')},ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--photo',required=True)
    p.add_argument('--history',nargs='+',required=True)
    a=p.parse_args();build(a.root,a.photo,a.history)
