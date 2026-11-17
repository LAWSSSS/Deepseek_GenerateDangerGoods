"""Build a full-frame, offline asset checkpoint report from immutable evidence.

No render, Photoshop operation, model request, or input-record update is performed.
Repeated review directories are deduplicated by logical request and attempt.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import zipfile

import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pixel_comparison(source,current):
    with Image.open(source) as im:a=np.array(im.convert('RGBA'))
    with Image.open(current) as im:b=np.array(im.convert('RGBA'))
    if a.shape!=b.shape:raise ValueError('source/current image dimensions differ')
    delta=np.abs(a[...,:3].astype(np.int16)-b[...,:3].astype(np.int16))
    aa,bb=a[...,3]>127,b[...,3]>127
    core=(a[...,3]>250)&(b[...,3]>250)
    rgba_delta=np.abs(a.astype(np.int16)-b.astype(np.int16))
    return {'canvas':[a.shape[1],a.shape[0]],'alpha_equal':bool(np.array_equal(a[...,3],b[...,3])),
        'alpha_iou':float((aa&bb).sum()/max(1,(aa|bb).sum())),
        'changed_pixels_all':int(np.any(delta>0,axis=2).sum()),
        'changed_pixels_visible':int((np.any(delta>0,axis=2)&(aa|bb)).sum()),
        'max_rgb_delta_255':int(delta.max()),'rgb_mae_full_255':float(delta.mean()),
        'rgb_mae_object_255':float(delta[core].mean()) if core.any() else None,
        'indistinguishable_within_one_code':bool(delta.max()<=1),
        'comparison_evidence':{'version':'pixel-comparison-v1','candidate_image_id':1,'reference_image_id':2,
            'candidate_sha256':file_digest(current),'reference_sha256':file_digest(source),
            'same_geometry':True,'same_lighting':True,'alpha_identical':bool(np.array_equal(a[...,3],b[...,3])),
            'compared_entire_object':True,'max_abs_rgba_255':int(rgba_delta.max()),
            'changed_pixel_fraction':float(np.any(rgba_delta>0,axis=2).mean()),
            'core_mae_255':float(delta[core].mean()) if core.any() else None},
        'interpretation':'difference_check_only_not_absolute_realism_quality'}


def collect_reviews(roots):
    logical={};events={};duplicates=0
    for root in roots:
        for path in sorted(root.rglob('usage.json')):
            manifest_path=path.with_name('manifest.json');diagnosis_path=path.with_name('diagnosis.json')
            if not manifest_path.is_file() or not diagnosis_path.is_file():continue
            manifest=read(manifest_path);diagnosis=read(diagnosis_path);usage=read(path)
            opinion=diagnosis.get('diagnosis') or {}
            input_signature=diagnosis.get('signature') or digest(manifest)
            calls=usage.get('calls',[])
            if not calls and usage.get('api_calls',0):
                raise ValueError('usage has totals but no request events: '+str(path))
            provider_scope=[diagnosis.get('provider'),manifest.get('endpoint',diagnosis.get('endpoint')),
                            diagnosis.get('model',manifest.get('model'))]
            request_ids=[call.get('provider_request_id') or call.get('request_id') for call in calls]
            run_id=usage.get('run_id') or diagnosis.get('run_id')
            if isinstance(run_id,str) and run_id:
                identity=digest(['declared_run',provider_scope,run_id]);identity_basis='declared_run_id'
            elif request_ids and all(isinstance(value,str) and value for value in request_ids):
                identity=digest(['provider_run',provider_scope,input_signature,request_ids]);identity_basis='provider_request_sequence'
            else:
                # Input signatures identify identical questions, not executions.
                # Without a request/run ID, two directories may be separate paid
                # executions. Count them separately and expose that ambiguity.
                local_path=os.path.normcase(str(diagnosis_path.resolve()))
                identity=digest(['local_run',input_signature,local_path]);identity_basis='resolved_local_diagnosis_path'
            item={'logical_id':identity,'input_signature':input_signature,'run_identity_basis':identity_basis,
                  'asset_id':str(manifest.get('context',{}).get('asset_profile',{}).get('asset_id',path.parent.name)),
                  'root':str(root),'manifest':str(manifest_path),'diagnosis_path':str(diagnosis_path),
                  'usage_path':str(path),'status':diagnosis.get('status','unknown'),
                  'model':diagnosis.get('model',manifest.get('model')),'seconds':diagnosis.get('seconds'),
                  'scores':opinion.get('scores'),
                  'findings':opinion.get('findings',[]),
                  'image_bindings':manifest.get('images',[]),
                  'claim_status':'raw_model_opinion_not_verified_quality'}
            if identity in logical and logical[identity]['input_signature']!=input_signature:
                raise ValueError('conflicting input signatures for declared request run')
            logical.setdefault(identity,item)
            for index,call in enumerate(calls):
                attempt=call.get('attempt',index)
                provider_id=request_ids[index]
                event_id=(digest(['provider_request',provider_scope,provider_id])
                          if isinstance(provider_id,str) and provider_id else digest([identity,attempt]))
                value={'logical_id':identity,'attempt':attempt,'model':call.get('model',item['model']),
                       'provider_request_id':provider_id,'usage':call.get('usage',{}),
                       'http_status':call.get('http_status'),'source':str(path)}
                if event_id in events:
                    ignored={'source','logical_id','attempt'} if provider_id else {'source'}
                    previous={k:v for k,v in events[event_id].items() if k not in ignored}
                    current={k:v for k,v in value.items() if k not in ignored}
                    if previous!=current:raise ValueError('conflicting usage for same logical request attempt')
                    duplicates+=1
                else:events[event_id]=value
    tokens=sum(event['usage'].get('total_tokens',0) or 0 for event in events.values())
    missing=sum(event['usage'].get('total_tokens') is None for event in events.values())
    by_input={}
    for item in logical.values():by_input.setdefault(item['input_signature'],[]).append(item)
    ambiguities=[{'input_signature':key,'diagnosis_paths':[item['diagnosis_path'] for item in items],
                  'status':'counted_separately','reason':'same_input_in_distinct_directories_without_complete_run_provenance; copy_or_actual_rerun_unknown'}
                 for key,items in by_input.items() if len(items)>1 and any(
                     item['run_identity_basis']=='resolved_local_diagnosis_path' for item in items)]
    return list(logical.values()),{'actual_requests':len(events),'logical_requests':len(logical),
        'retries':sum(max(0,sum(v['logical_id']==key for v in events.values())-1) for key in logical),
        'reported_tokens':tokens,'unreported_requests':missing,'duplicate_events_excluded':duplicates,
        'provenance_ambiguities':ambiguities,
        'events':list(events.values()),'credits':None,'currency_cost':None,
        'cost_basis':'provider Credits/cash not supplied; no guessed per-token tariff'}


def collect_ps(root):
    exports=[]
    for path in sorted(root.rglob('result.json')):
        record=read(path)
        if not record.get('receipt') or not record.get('photoshop'):continue
        exports.append({'path':str(path),'status':record.get('status'),
            'seconds':record['receipt'].get('seconds'),'validation':record.get('validation'),
            'applied_parameters':record['receipt'].get('absolute_parameters'),
            'pixel_changes':record['receipt'].get('changed_pixels'),
            'visual_acceptance':record.get('visual_acceptance'),
            'evidence_scope':'Photoshop_fixture_not_asset_scene_output'})
    requested=root/'requested-measurement.json'
    measurement=read(requested) if requested.is_file() else {}
    measured_seconds=measurement.get('actual',{}).get('seconds')
    known=sum(row['seconds'] or 0 for row in exports)+(measured_seconds or 0)
    return {'exports':exports,'actual_export_count':len(exports),'requested_measurement_path':str(requested) if requested.exists() else None,
            'requested_measurement_seconds':measured_seconds,'known_operation_seconds':known,
            'inspection_path':str(root/'inspection.json') if (root/'inspection.json').exists() else None,
            'initial_inspection_seconds':None,'total_wall_seconds':None,
            'time_note':'only recorded operation durations summed; initial inspection and full wall time unknown'}


def build_report(source,review_roots,ps_root):
    rows=read(source/'results.json');frozen=read(source/'frozen.json')
    reviews,usage=collect_reviews(review_roots)
    frozen_cases={c['asset_id']:c for c in frozen['cases']}
    assets=[];images=[]
    for row in rows:
        ident=row['asset_id'];folder=source/ident
        comparison=pixel_comparison(folder/'source-left.png',folder/'physical-left.png')
        flags=[]
        if comparison['indistinguishable_within_one_code']:
            flags.append('源与新图最多相差1/255；模型声称本轮造成明显变糊、厚灰或退化缺少像素依据。此项不能证明原始质量良好。')
            flags.append('没有可靠的可见改进/退化证据，绝对质量仍待审。')
        rendered=[]
        for item in row['renders']:
            path=source/item['image'];sha=file_digest(path)
            if sha!=item['sha256']:raise ValueError('delivered preview changed: '+str(path))
            rendered.append({**item,'path':str(path),'file_sha256_verified':True})
            images.append((ident+'-'+item['tag']+'.png',path))
        board=source/row['comparison']
        images.append((ident+'-complete-comparison.png',board))
        material=read(folder/'candidate-check.json')
        relevant=[r for r in reviews if r['asset_id']==ident]
        # Every raw review remains accessible; the last input directory is the
        # designated final review, not whichever has the most favourable score.
        final=relevant[-1] if relevant else None
        case=frozen_cases[ident]
        assets.append({'asset_id':ident,'name':row['name'],'quality_status':'review_required_not_passed',
            'source_mesh_hash':case['mesh_hash'],'geometry_modified':False,
            'material_signature':material.get('material_signature'),
            'render_signature_evidence':[{ 'tag':item['tag'],'render_cache':item.get('render_cache'),
                                          'png_sha256':item['sha256'],'file_sha256_verified':True} for item in rendered],
            'frozen_finish_signature':case['finish']['signature'],
            'pixel_comparison':comparison,'program_checks':row['checks'],
            'measurement_evidence':{'comparisons':[comparison['comparison_evidence']]},
            'candidate_check':material,'candidate_path':str(folder/'candidate.blend'),
            'material_readback_path':str(folder/'candidate-material-readback.json'),
            'complete_comparison':str(board),'renders':rendered,'all_reviews':relevant,'final_review':final,
            'interpretation_flags':flags,'known_limits':row.get('known_limits',[]),
            'layer_coverage':row.get('layer_coverage'),
            'source_hash_verified':file_digest(read(case['asset_record'])['mesh'])==case['mesh_hash']})
    hashes=read(source/'source-assets-check.json')
    render_cost=read(source/'render-cost-ledger.json')
    report={'version':'asset-checkpoint-report-v1','source':str(source),'review_roots':[str(p) for p in review_roots],
        'state':'model_review_checkpoint','production_default_changed':False,'scene_trials_run':0,
        'next_step':'用户确认模型效果后，再评估12项场景小样；当前四件写实质量未通过。',
        'assets':assets,'source_models':hashes,'usage':usage,'render_cost':render_cost,'photoshop':collect_ps(ps_root),
        'scores_policy':'raw_scores_are_model_opinions; contradictions retained; physical/scale unknown not passed',
        'generated_images':len([p for p in images if '-complete-comparison' not in p[0]]),
        'complete_boards':len(assets),'no_input_records_modified':True}
    return report,images


def link(path,out):
    try:return quote(os.path.relpath(Path(path),out).replace('\\','/'),safe='/.:')
    except ValueError:return Path(path).resolve().as_uri()


def render_html(report,out):
    esc=html.escape
    def anchor(path,label):return '<a href="'+esc(link(path,out),quote=True)+'">'+esc(label)+'</a>'
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>模型写实修复检查点</title>',
          '<style>body{max-width:1600px;margin:24px auto;padding:0 20px;font:16px/1.6 sans-serif;background:#eee;color:#222}table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:8px;text-align:left}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}figure{margin:0;background:#777}figcaption{padding:8px;background:#eee}img{display:block;width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:2px solid #aaa;margin-top:28px;padding-top:16px}.notice{background:#fff1ce;padding:16px}a{color:#075574}</style>',
          '<h1>四件模型写实修复：模型确认检查点</h1><p class="notice">当前四件模型的写实质量仍未通过。本页保留全部完整画面与模型原始意见；程序检查通过不等于视觉质量通过。未运行12项场景测试，生产默认未切换。</p>',
          '<p>灰模／原PBR／新分层材质在两种宽面照明下比较，另有转向检查。没有现场薄雾、调色、噪声或相机模糊。13胶罐作为塑料保护对照。</p>']
    u=report['usage'];r=report['render_cost'];ps=report['photoshop']
    body.append('<h2>实际用量</h2><table><tr><th>项目</th><th>实际记录</th></tr>')
    for label,value in [('视觉请求',f"{u['actual_requests']}次实际请求 / {u['logical_requests']}次逻辑请求 / {u['retries']}次重试"),
                        ('供应商 token',str(u['reported_tokens'])),('物体渲染',f"{r['actual_object_renders']}次 / {r['actual_render_seconds']:.3f}秒，含失败尝试"),
                        ('Photoshop',f"{ps['actual_export_count']}次实际夹具导出；已记录操作耗时合计 {ps['known_operation_seconds']:.3f}秒"),
                        ('费用','Credits、现金费用与本地计算费用未知；不按未配置价表估算'),
                        ('源模型',f"{report['source_models']['count']}个；散列全部不变：{report['source_models']['all_unchanged']}")]:
        body.append('<tr><td>'+esc(label)+'</td><td>'+esc(value)+'</td></tr>')
    body.append('</table><p>早期失败评审、格式修复和失败导出后的渲染均计入；同一输出目录重复引用不重复计费。相同输入在不同目录运行，缺少供应商请求ID或运行ID时分别计入，并标明复制来源未知。初次失败运行、早期PS检查缺少完整墙钟计时，未凑造总耗时。</p>')
    if u.get('provenance_ambiguities'):
        body.append('<details><summary>请求来源存在歧义，已分别计入</summary><pre>'+esc(json.dumps(u['provenance_ambiguities'],ensure_ascii=False,indent=2))+'</pre></details>')
    if report.get('zip_path'):body.append('<p>'+anchor(report['zip_path'],'桌面平铺图片 ZIP：仅图片、无文件夹')+'</p>')
    body.append('<p>'+anchor(out/'report.json','完整证据与成本 JSON')+'</p>')
    for asset in report['assets']:
        body.append('<section><h2>'+esc(asset['asset_id']+' '+asset['name'])+' · 待人工确认</h2>')
        body.append('<p>'+anchor(asset['candidate_path'],'独立候选 Blender 文件')+' · '+anchor(asset['complete_comparison'],'完整模型对照板')+' · '+anchor(asset['material_readback_path'],'实际材质读回')+'</p>')
        final=asset['final_review']
        if final:
            body.append('<p>最终千问原始评分（仅模型意见）：'+esc(json.dumps(final['scores'],ensure_ascii=False))+'；状态 '+esc(final['status'])+'。'+anchor(final['diagnosis_path'],'查看完整原始诊断')+'</p>')
        else:body.append('<p>最终评审缺失，保留未知。</p>')
        for warning in asset['interpretation_flags']+asset['known_limits']:body.append('<p class="notice">'+esc(warning)+'</p>')
        body.append('<details><summary>像素、材质与原始资产证据</summary><pre>'+esc(json.dumps({k:asset[k] for k in ('pixel_comparison','program_checks','source_mesh_hash','source_hash_verified','material_signature','render_signature_evidence','layer_coverage')},ensure_ascii=False,indent=2))+'</pre></details>')
        body.append('<div class="grid">')
        for image in asset['renders']:
            uri=esc(link(image['path'],out),quote=True)
            body.append('<figure><a href="'+uri+'"><img loading="lazy" src="'+uri+'" alt="'+esc(image['label'],quote=True)+'"></a><figcaption>'+esc(image['label'])+' · 完整画面</figcaption></figure>')
        body.append('</div><details><summary>全部评审记录（含失败与早期结果）</summary>')
        for review in asset['all_reviews']:
            body.append('<p>'+anchor(review['diagnosis_path'],review['status']+' / '+review['logical_id'][:12])+' · '+anchor(review['usage_path'],'实际用量')+'</p>')
        body.append('</details></section>')
    body.append('<section><h2>Photoshop通道验证</h2><p>下列是工具夹具验证，不能冒充场景融合成图或模型质量验证。</p>')
    for export in ps['exports']:body.append('<p>'+anchor(export['path'],'实际导出与保护像素检查')+' · '+esc(str(export['seconds']))+'秒</p>')
    if ps['requested_measurement_path']:body.append('<p>'+anchor(ps['requested_measurement_path'],'模型按需请求的区域测量回执')+'</p>')
    body.append('</section></html>')
    return ''.join(body)


def flat_zip(images,directory,name):
    directory.mkdir(parents=True,exist_ok=True)
    if Path(name).name!=name or not name.endswith('.zip'):raise ValueError('zip name must be a filename ending in .zip')
    stem=Path(name).stem;candidate=directory/name;number=2
    while candidate.exists():candidate=directory/(stem+'-'+str(number)+'.zip');number+=1
    names=[name for name,_ in images]
    if len(set(names))!=len(names):raise ValueError('duplicate flattened image name')
    with zipfile.ZipFile(candidate,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in images:archive.write(path,arcname=name)
    with zipfile.ZipFile(candidate) as archive:
        if archive.testzip() is not None or any('/' in name or '\\' in name for name in archive.namelist()):
            raise ValueError('invalid flat image archive')
    return candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--review-dir',type=Path,action='append',required=True,help='ordered early to final; repeat directories are safe')
    parser.add_argument('--ps-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--zip-dir',type=Path)
    parser.add_argument('--zip-name',default='asset-checkpoint-images.zip')
    args=parser.parse_args();source=args.source_dir.resolve();out=args.out.resolve()
    roots=[p.resolve() for p in args.review_dir];ps=args.ps_dir.resolve()
    if out in {source,ps,*roots}:raise ValueError('report output must differ from evidence directories')
    report,images=build_report(source,roots,ps);out.mkdir(parents=True,exist_ok=True)
    if args.zip_dir:report['zip_path']=str(flat_zip(images,args.zip_dir.resolve(),args.zip_name))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'index.html').write_text(render_html(report,out),encoding='utf8')
    print(json.dumps({'report':str(out/'index.html'),'images':report['generated_images'],'boards':report['complete_boards'],
                     'requests':report['usage']['actual_requests'],'tokens':report['usage']['reported_tokens'],
                     'zip':report.get('zip_path')},ensure_ascii=False))


if __name__=='__main__':main()
