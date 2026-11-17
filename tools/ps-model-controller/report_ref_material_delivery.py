"""Hash-bound, offline delivery report and flat PNG archive; no rendering/API.

Do not count a protected control or a changed pixel as a successful retexture.
The report retains separate execution, material-change and visual statuses.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import re
import struct
from urllib.parse import quote
import zipfile

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[2]
LABELS = {'source-left': '原材质 · 左侧光', 'new-left': '候选材质 · 同视角同光',
          'new-right': '候选材质 · 换光', 'new-turned': '候选材质 · 转向55°'}
STATUSES = {'modified_candidate': '有实际材质改动，视觉待审',
            'protected_unchanged': '完整保护，保留原貌／无有效修改',
            'protected_numeric_difference': '完整保护／无有效修改；有至多1级像素数值差异',
            'protection_difference_unresolved': '保护控制出现较大或未量化差异，待核对',
            'no_effective_change': '没有测得有效修改',
            'insufficient_evidence': '修改证据不足，待核对'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path, out):
    path = Path(path).resolve()
    try:
        return Path(os.path.relpath(path, out)).as_posix()
    except ValueError:  # Windows cannot express a relative path across drives.
        return path.as_uri()


def evidence(path, out, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected and actual != expected:
        raise ValueError('evidence hash mismatch: ' + str(path))
    return {'path': relative(path, out), 'sha256': actual,
            'recorded_hash_verified': bool(expected), 'bytes': path.stat().st_size}


def portable(value, out):
    if isinstance(value, dict):
        return {key: portable(item, out) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, out) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return relative(value, out)
    return value


def classify(row):
    audit = row.get('material_audit', [])
    eligible = sum(bool(item.get('eligible')) for item in audit)
    new = next((item for item in row['renders'] if item['tag'] == 'new-left'), {})
    materials = (new.get('material_readback') or {}).get('materials', [])
    applied = [item for item in materials if item.get('physical_layers_applied')]
    connected = [item for item in applied if item.get('reference_response', {}).get('status') == 'measured']
    comparison = row.get('checks', {}).get('source_to_new', {})
    delta = comparison.get('changed_pixels')
    if audit and not eligible and materials and not applied:
        if delta == 0:
            status = 'protected_unchanged'
        elif delta is not None and comparison.get('max_abs_rgba_255', 256) <= 1:
            status = 'protected_numeric_difference'
        else:
            status = 'protection_difference_unresolved'
    elif eligible and applied and delta is not None:
        status = 'modified_candidate' if delta > 0 else 'no_effective_change'
    else:
        status = 'insufficient_evidence'
    return {'status': status, 'label': STATUSES[status], 'eligible_material_slots': eligible,
            'ineligible_material_slots': len(audit)-eligible,
            'applied_material_slots': len(applied), 'connected_reference_slots': len(connected),
            'pixel_difference_is_quality_score': False,
            'coverage_basis': 'material_slot_counts_and_UV_texels_are_not_surface_area',
            'visible_source_regions': new.get('source_regions'), 'visual_acceptance': 'unreviewed'}


def generation_ledger(paths, out, used_masters):
    records, masters = [], []
    request_count = 0
    for path in paths:
        path = Path(path).resolve()
        data = read(path)
        records.append(evidence(path, out))
        request_count += data.get('requests', 1 if data.get('prompt') else 0)
        items = data.get('masters', [data] if data.get('prompt') else [])
        for item in items:
            prompt = item.get('prompt', '')
            prompt_sha = hashlib.sha256(prompt.encode('utf8')).hexdigest()
            if item.get('prompt_sha256') and item['prompt_sha256'] != prompt_sha:
                raise ValueError('generation prompt hash mismatch: ' + str(path))
            refs = [{**evidence(ref['path'], out, ref.get('sha256')),
                     'role': ref.get('role', 'style_reference')} for ref in item.get('reference_images', [])]
            master_path = path.parent / item['path'] if item.get('path') else path.parent / (path.stem.replace('-provenance', '') + '.png')
            record = {**evidence(master_path, out, item.get('sha256')), 'prompt': prompt,
                      'prompt_sha256': prompt_sha, 'reference_images': refs,
                      'used_in_this_run': item.get('sha256') in used_masters,
                      'generation_record': relative(path, out)}
            masters.append(record)
    return {'generator': 'built-in image_gen', 'recorded_requests': request_count,
            'token_usage': None, 'currency_cost': None, 'credits': None,
            'usage_status': 'supplier_usage_not_returned_unknown_not_zero',
            'records': records, 'masters': masters}


def run_cost(path, out):
    path = Path(path).resolve()
    data = read(path)
    return {'record': evidence(path, out), **{key: data.get(key) for key in
        ('wall_seconds', 'worker_seconds', 'candidate_export_seconds', 'render_counts',
         'render_timings', 'api_calls', 'tokens', 'api_cost', 'local_compute_cost')}}


def build(out, generations=(), prior_runs=(), local_stages=(), identity_audit=None):
    frozen, results, summary = [read(out / name) for name in ('frozen.json', 'results.json', 'summary.json')]
    if summary.get('state') not in ('complete', 'completed_with_failures'):
        raise ValueError('wait for a completed run before creating delivery')
    if summary['completed_assets'] != len(results):
        raise ValueError('result count differs from summary')
    cases = {row['asset_id']: row for row in frozen['cases']}
    rows, master_hashes = [], set()
    for row in results:
        aid = row['asset_id']
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', aid):
            raise ValueError('unsafe asset id')
        case = cases[aid]
        plates = []
        seen = set()
        for plate in row['renders']:
            tag = plate['tag']
            if tag not in LABELS or tag in seen:
                raise ValueError('unexpected/duplicate plate tag')
            seen.add(tag)
            record = evidence(out / plate['image'], out, plate['sha256'])
            with (out / plate['image']).open('rb') as stream:
                header = stream.read(24)
            if header[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('not a PNG plate')
            dimensions = struct.unpack('>II', header[16:24])
            if dimensions != (frozen['resolution'], frozen['resolution']):
                raise ValueError('unexpected full-image dimensions')
            plates.append({**record, 'tag': tag, 'label': LABELS[tag],
                           'dimensions': list(dimensions), 'archive_name': aid.zfill(2)+'-'+tag+'.png'})
        if seen != set(frozen['variants']):
            raise ValueError('completed asset has missing variants')
        plate_paths = {plate['tag']: out / plate['path'] for plate in plates}
        with Image.open(plate_paths['source-left']) as original, Image.open(plate_paths['new-left']) as candidate_image:
            alpha_max = ImageChops.difference(original.getchannel('A'), candidate_image.getchannel('A')).getextrema()[1]
        binding = row['reference_binding']
        source = binding['evidence']['source_asset']
        reference = read(binding['reference_path'])
        master_hashes.add(reference['input_sha256'])
        maps = {key: evidence(item['path'], out, item['sha256']) for key, item in reference['maps'].items()}
        provenance = {
            'mesh': evidence(source['mesh_path'], out, case['mesh_hash']),
            'original_image': evidence(source['original_image_path'], out, source['original_image_sha256']),
            'six_view': evidence(binding['evidence']['six_view_path'], out, binding['evidence']['six_view_sha256']),
            'surface_package': evidence(binding['reference_path'], out, binding['reference_sha256']),
            'master': evidence(reference['input_path'], out, reference['input_sha256']),
            'maps': maps, 'identity_evidence': binding['evidence']['identity_evidence'],
            'surface_role': 'reusable_generated_motif_not_individual_measured_material',
            'surface_package_id': binding['surface_package'],
            'cautions': binding['evidence'].get('cautions', []),
            'source_role': binding['evidence'].get('six_view_role')}
        candidate = out / aid / 'candidate.blend'
        check = read(out / aid / 'candidate-check.json')
        connected = check.get('reference_response_connected')
        candidate_status = ('measured_connected' if connected and all(connected.values()) else
                            'not_applicable_protected' if connected == {} and not any(x.get('eligible') for x in row['material_audit']) else
                            'not_measured_or_failed')
        rows.append({'asset_id': aid, 'name': row['name'], **classify(row),
                     'plates': plates, 'checks': {**row['checks'], 'delivery_alpha_max_abs_255': alpha_max}, 'provenance': provenance,
                     'material_audit': row['material_audit'],
                     'candidate': {**evidence(candidate, out), 'check': portable(check, out),
                                   'reference_connection_status': candidate_status},
                     'comparison': evidence(out / row['comparison'], out)})
    counts = dict(Counter(row['status'] for row in rows))
    catalog = summary.get('catalog_hash_checks', {})
    record = {'version': 'reference-material-delivery-v1',
              'scope': 'complete_model_material_review_not_scene_acceptance',
              'input_records': [evidence(out / name, out) for name in ('frozen.json', 'results.json', 'summary.json')],
              'selected_assets': len(frozen['selected_asset_ids']), 'completed_assets': len(rows),
              'full_images': sum(len(row['plates']) for row in rows), 'variants': frozen['variants'],
              'material_status_counts': counts, 'visually_accepted_assets': None,
              'full_batch_visual_acceptance_evaluated': False,
              'source_candidate_alpha_max_abs_255': max((row['checks']['delivery_alpha_max_abs_255'] for row in rows), default=None),
              'source_candidate_alpha_iou_min': min((row['checks']['source_to_new']['alpha_iou'] for row in rows), default=None),
              'visual_status': 'unreviewed', 'scene_contact_and_scale': 'not_evaluated',
              'production_default_changed': summary.get('production_default_changed'),
              'original_catalog_hashes_unchanged': sum(v is True for v in catalog.values()),
              'original_catalog_hashes_total': len(catalog),
              'failures': summary.get('failures', []), 'rows': rows,
              'costs': {'this_run': run_cost(out / 'summary.json', out),
                        'prior_runs_separate_not_final_throughput': [run_cost(path, out) for path in prior_runs],
                        'image_generation': generation_ledger(generations, out, master_hashes),
                        'local_preparation_stages': [
                            {'record': evidence(path, out),
                             **{key: value for key, value in read(path).items()
                                if not isinstance(value, (dict, list))}}
                            for path in local_stages],
                        'local_compute_currency_cost': None},
              'known_limits': summary.get('known_limits', []),
              'path_basis': 'relative_to_delivery_json_directory; other_drive_sources_use_file_URI_and_remain_external'}
    if identity_audit:
        audit = read(identity_audit)
        all_references = [evidence(item['image_path'], out, item['sha256'])
                          for item in audit['mappings']]
        if len(all_references) != audit['reference_count']:
            raise ValueError('identity audit reference count mismatch')
        record['identity_audit'] = {'record': evidence(identity_audit, out),
            'all_reference_images': all_references, 'reference_images_verified': len(all_references),
            'summary': audit['summary'], 'excluded_asset_ids': audit['excluded_asset_ids'],
            'catalog_hashes_match_freeze': audit['catalog_mesh_hashes_at_read'] == frozen['catalog_mesh_hashes']}
    return record


def link(path, label):
    return '<a href="'+html.escape(quote(path, safe='/.:_-#%'), quote=True)+'">'+html.escape(label)+'</a>'


def pre(value):
    return '<pre>'+html.escape(json.dumps(value, ensure_ascii=False, indent=2))+'</pre>'


def attach_interactive_review(out, report, folder):
    folder = Path(folder).resolve()
    diagnosis, usage, manifest = [read(folder / name) for name in ('diagnosis.json', 'usage.json', 'manifest.json')]
    if diagnosis.get('status') != 'completed':
        raise ValueError('interactive review is not completed')
    images = [evidence(item['path'], out, item['sha256']) for item in manifest['images']]
    image_hashes = {image['sha256'] for image in images}
    target_ids = [row['asset_id'] for row in report['rows'] if row['comparison']['sha256'] in image_hashes]
    if not target_ids:
        raise ValueError('interactive review does not bind a current complete comparison board')
    report['interactive_review'] = {
        'records': {name: evidence(folder / name, out) for name in ('diagnosis.json', 'usage.json', 'manifest.json')},
        'target_asset_ids': target_ids, 'images': images, 'model': diagnosis['model'],
        'assessment': diagnosis['diagnosis'], 'seconds': diagnosis['seconds'],
        'scope': 'targeted_visual_opinion_and_causal_hypotheses_not_formal_acceptance_or_batch_score',
        'resolution_limit': 'review_uses_resized_JPEG_inputs; tiny_scratches_and_surface_relief_may_be_unresolved'}
    encoded_inputs = []
    for path in sorted(folder.glob('input-*.jpg')):
        with Image.open(path) as image:
            dimensions = list(image.size)
        encoded_inputs.append({**evidence(path, out), 'dimensions': dimensions})
    report['interactive_review']['prepared_review_images'] = encoded_inputs
    report['visual_status'] = 'partial_interactive_review_no_formal_batch_acceptance'
    report['costs']['interactive_review'] = {**usage, 'seconds': diagnosis['seconds']}


def write_html(out, report):
    pieces = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
              '<title>六视图参考材质完整模型交付</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px/1.55 sans-serif;color:#202020;background:#eee}a{color:#075d66}img{display:block;width:100%;height:auto;background:#777}figure{margin:0}.plates{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:12px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;text-align:left;padding:8px}.notice{padding:16px;background:#fff4d0}@media(max-width:700px){.plates{grid-template-columns:1fr}}</style>',
              '<h1>六视图参考材质：完整模型对照</h1>',
              '<p class="notice">本页是候选材质检查，不是写实质量或场景验收。完整保护的资产没有有效重贴，不计为成功修改。源模型与生产默认保持不变；现场承托、真实尺度及最终合成未验证。</p>',
              '<p>已输出 '+str(report['completed_assets'])+'/'+str(report['selected_assets'])+' 件、'+str(report['full_images'])+' 张完整模型图。全批正式视觉合格率未评估，未宣布任何候选通过。源模型散列一致：'+str(report['original_catalog_hashes_unchanged'])+'/'+str(report['original_catalog_hashes_total'])+'。</p>',
              '<p>原／候选轮廓 IoU 最小值：'+str(report['source_candidate_alpha_iou_min'])+'；重新读取PNG测得 alpha 最大误差：'+str(report['source_candidate_alpha_max_abs_255'])+'/255。</p>',
              '<ul>'+''.join('<li>'+STATUSES[key]+'：'+str(value)+' 件</li>' for key,value in report['material_status_counts'].items())+'</ul>',
              '<p>'+link('delivery.json','完整证据与散列')+' · '+link('summary.json','渲染运行摘要')+' · '+link('frozen.json','冻结输入')+'</p>',
              '<p>'+link(report['archive']['path'],'桌面完整 PNG 压缩包')+'</p>' if report.get('archive') else '',
              '<p>原／候选左光对照采用相同几何、视角和灯光。换光与转向用于观察响应；转向图没有同方向原材质控制，不能单独归因为材质改善。以下图片均保持完整画布，可点击原尺寸查看。</p>',
              '<table><tr><th>物品</th><th>材质执行</th><th>候选重新打开后连接检查</th><th>视觉结论</th></tr>']
    for row in report['rows']:
        pieces.append('<tr><td>'+link('#asset-'+row['asset_id'],row['asset_id']+' '+row['name'])+'</td><td>'+html.escape(row['label'])+'</td><td>'+html.escape(row['candidate']['reference_connection_status'])+'</td><td>待审</td></tr>')
    pieces.append('</table>')
    if report.get('interactive_review'):
        review = report['interactive_review']
        pieces += ['<h2>定向交互诊断：'+html.escape('、'.join(review['target_asset_ids']))+'</h2>',
                   '<p>以下是模型对指定完整对照板的视觉意见及原因假设，只覆盖上述物品，不是全部14件的合格率，也未自动执行建议。评审使用缩小的JPEG输入，小划痕与微起伏可能看不清；生成六视图不提供真实厚度、合金或BRDF测量。</p>',
                   '<p>'+html.escape(review['assessment'].get('summary',''))+'</p>',
                   '<p>'+link(review['records']['diagnosis.json']['path'],'完整诊断及限制')+' · '+link(review['records']['manifest.json']['path'],'提问与输入图散列')+'</p>']
    for row in report['rows']:
        aid=row['asset_id']; provenance=row['provenance']
        pieces += ['<h2 id="asset-'+aid+'">'+html.escape(aid+' '+row['name'])+'</h2><p><strong>'+html.escape(row['label'])+'</strong></p>',
                   '<p>'+link(row['comparison']['path'],'完整对照板')+' · '+link(row['candidate']['path'],'独立候选 Blender')+' · '+link(aid+'/candidate-check.json','候选回读')+'</p>',
                   '<p>'+link(provenance['original_image']['path'],'原物品图')+' → '+link(provenance['six_view']['path'],'对应六视图')+' → '+link(provenance['master']['path'],'使用的平面母图')+' → '+link(provenance['surface_package']['path'],'估计通道与生成记录')+'</p>',
                   '<p>'+html.escape(provenance['identity_evidence'])+' 材质包：'+html.escape(provenance['surface_package_id'])+'。共用母图只提供纹样风格，不是每件物品的实测材质。</p>', '<div class="plates">']
        for plate in row['plates']:
            url=html.escape(quote(plate['path'], safe='/._-'),quote=True)
            pieces.append('<figure><figcaption>'+html.escape(plate['label'])+'</figcaption><a href="'+url+'"><img loading="lazy" src="'+url+'" alt="'+html.escape(aid+' '+plate['label'],quote=True)+'"></a><small>SHA-256 '+plate['sha256']+'</small></figure>')
        pieces += ['</div><details><summary>材料覆盖、保护与实际检查</summary>', pre({
            'status': row['status'], 'material_audit': row['material_audit'],
            'visible_source_regions': row['visible_source_regions'], 'checks': row['checks'],
            'candidate_reference_connection': row['candidate']['reference_connection_status'],
            'cautions': provenance['cautions']}), '</details>']
    pieces += ['<h2>生成来源、用量与限制</h2><p>本地渲染记录的零 API 不包含此前内置生图。供应商未返回的 token、Credits 和金额均为未知，不能记为零。本地预检与最终运行分别计时；导出时间可能已含于运行耗时，不重复相加。</p>', pre(report['costs']),
               '<details><summary>全部参考图身份审计（含未用于绑定的图）</summary>',pre(report.get('identity_audit',{})),'</details>',
               '<h2>失败和未知项</h2>',pre({'failures':report['failures'],'limits':report['known_limits']}),'</html>']
    (out/'delivery.html').write_text(''.join(pieces),encoding='utf8')


def flat_zip(out, report, target, include_boards=False):
    target=Path(target).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    plates=[plate for row in report['rows'] for plate in row['plates']]
    if include_boards:
        plates += [{**row['comparison'], 'archive_name': row['asset_id'].zfill(2)+'-complete-comparison.png'}
                   for row in report['rows']]
    names=[plate['archive_name'] for plate in plates]
    if len(set(names))!=len(names):
        raise ValueError('duplicate flat archive filename')
    temporary=target.with_suffix(target.suffix+'.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for plate in plates:
            archive.write(out/plate['path'],plate['archive_name'])
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist()!=names or archive.testzip() is not None:
            raise ValueError('archive entry or CRC check failed')
        for plate in plates:
            if hashlib.sha256(archive.read(plate['archive_name'])).hexdigest()!=plate['sha256']:
                raise ValueError('archive image hash mismatch')
    temporary.replace(target)
    return {**evidence(target,out),'entries':len(names),'structure':'flat_PNG_only',
            'full_model_images':report['full_images'],
            'complete_comparison_boards':len(report['rows']) if include_boards else 0,
            'image_hashes_verified_against_render_records':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--generation',action='append',default=[],type=Path)
    parser.add_argument('--prior-run',action='append',default=[],type=Path,
                        help='prior summary.json; kept separate from final-run throughput')
    parser.add_argument('--flat-zip',type=Path)
    parser.add_argument('--include-boards',action='store_true',help='also include full comparison boards in the flat archive')
    parser.add_argument('--local-stage',action='append',default=[],type=Path,
                        help='optional extraction/preparation cost record; no invented totals')
    parser.add_argument('--identity-audit',type=Path)
    parser.add_argument('--interactive-review',type=Path)
    args=parser.parse_args()
    out=args.run.resolve()
    report=build(out,args.generation,args.prior_run,args.local_stage,args.identity_audit)
    if args.interactive_review:
        attach_interactive_review(out,report,args.interactive_review)
    if args.flat_zip:
        report['archive']=flat_zip(out,report,args.flat_zip,args.include_boards)
    (out/'delivery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    write_html(out,report)
    print(json.dumps({key:report[key] for key in ('completed_assets','full_images','material_status_counts','visual_status')},ensure_ascii=False))


if __name__=='__main__':
    main()
