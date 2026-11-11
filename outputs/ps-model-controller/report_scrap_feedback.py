"""Publish equal-size material comparisons, exact image checks, and batch costs."""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from display_batch.common import atomic_json, file_hash, load_catalog, resolve
from display_batch.costs import cumulative, estimate, label
from validate_display_costs import selected_geometry
from validate_display_local import read, records


STATUS = {'accepted_display':'展示合格','review':'待审','failed':'失败'}
SURFACE_DEFECTS = {'reflection_flat','reflection_hard','surface_clean','dust_heavy','patch_seam'}


def paired_crops(before,after,bounds,folder):
    with Image.open(before) as a, Image.open(after) as b:
        first,second = a.convert('RGB').crop(bounds),b.convert('RGB').crop(bounds)
    first.thumbnail((850,630),Image.Resampling.LANCZOS)
    second.thumbnail((850,630),Image.Resampling.LANCZOS)
    for name,image in [('before',first),('after',second)]:
        image.save(folder/(name+'.jpg'),quality=95)
    return second


def audit_png(row,best,comparison):
    with Image.open(row['background']) as original, Image.open(row['image']) as result:
        bg = np.asarray(ImageOps.exif_transpose(original).convert('RGB'))
        actual = np.asarray(result.convert('RGB'))
    size_ok = actual.shape==bg.shape and actual.shape[2]==3
    changed = int(bg.shape[0]*bg.shape[1])
    if size_ok:
        difference = np.any(bg!=actual,axis=2)
        l,t,r,b = best['edit_bounds']
        difference[t:b,l:r] = False
        changed = int(difference.sum())
    checks = best['checks']
    return {'size_ok':size_ok,'size':list(actual.shape[1::-1]),'outside_rectangle_changes':changed,
        'verified_png_hash':file_hash(row['image'])==best['image_hash'],
        'protected_changes':checks['protected']['changed_pixels'],
        'foreground_changes':checks['occlusion']['foreground_changed_pixels'],
        'outside_edit_changes':checks['outside_edit']['changed_pixels'],
        'render_coverage_iou':checks['render_coverage']['iou'],
        'optical_iou':best['photo']['optical_iou'],
        'same_alpha':best['alpha_hash']==comparison['alpha_hash'],
        'same_visible_alpha':best['visible_alpha_hash']==comparison['visible_alpha_hash'],
        'same_occlusion':best['foreground_mask_hash']==comparison['foreground_mask_hash'],
        'anchor_error_px':checks['anchor']['error_px']}


def usage_by_job(ledgers):
    totals = {}
    roles = Counter()
    for ledger in ledgers:
        groups = defaultdict(list)
        for call in ledger.get('calls',[]):
            if call.get('state')!='not_sent':
                groups[call.get('job_id')].append(call)
                roles[call['role']]+=1
        for ident,calls in groups.items():
            value = estimate(calls,ledger.get('tariff'))
            row = totals.setdefault(ident,{'requests':0,'retries':0,'tokens':0,'currency':value['currency'],
                'estimated_api_min':0.,'estimated_api_max':0.})
            row['requests'] += value['actual_requests']
            row['retries'] += sum(c.get('attempt',0)>0 for c in calls)
            row['tokens'] += value['reported_tokens']
            for key in ('estimated_api_min','estimated_api_max'):
                previous,added = row[key],value[key]
                row[key] = (round(previous+added,8) if previous is not None and added is not None
                            and row['currency']==value['currency'] else None)
    return totals,roles


def write_report(out):
    baseline = Path(read(out/'baseline.json')['directory'])
    original_rows = {r['job_id']:r for r in records(baseline)}
    summary = read(out/'summary.json')
    rows = records(out)
    inputs = read(out/'inputs.json')
    catalog = load_catalog()
    cost = cumulative(out)
    runs = [read(path) for path in sorted(out.glob('run-*.json'))]
    usage,roles = usage_by_job([read(path) for path in sorted(out.glob('usage-*.json'))])
    batch = {'api':cost,'runs':runs,'request_roles':dict(roles),
             'wall_seconds':sum(r.get('seconds',0) for r in runs),
             'jobs':[{'id':r['job_id'],'cost':r['cost'],'timings':r.get('timings',{}),
                      'seconds':r.get('seconds'),'api_all_attempts':usage.get(r['job_id'])} for r in rows]}
    atomic_json(out/'cost-ledger.json',batch)
    audits = []
    previews = []
    cards = []
    defect_counts = Counter()
    class_passes = Counter()
    severe_surface = 0
    for row in rows:
        number = int(row['job_id'].split('-')[0])
        title = f'{number:02d} · {catalog[row["asset_id"]]["name"]}'
        status = STATUS.get(row['status'],row['status'])
        escaped = html.escape(title)
        if not row.get('image'):
            cards.append(f'<section id="n{number:02d}"><h2>{escaped}</h2><p>{html.escape(status+" · "+row["reason"])}</p></section>')
            audits.append({'number':number,'status':row['status'],'reason':row['reason']})
            continue
        previous = original_rows[inputs[number-1]['baseline_job_id']]
        old,_ = selected_geometry(baseline,previous)
        control = out/'size-controls'/previous['job_id']/'control.json'
        comparison = read(control) if control.is_file() else old
        comparison_image = comparison['image'] if control.is_file() else previous['image']
        best,candidate = selected_geometry(out,row)
        details = best['photo']['surface_material']
        folder = out/'comparison'/f'{number:02d}'
        folder.mkdir(parents=True,exist_ok=True)
        bounds = best['photo']['local_context']['bounds']
        thumbnail = paired_crops(comparison_image,row['image'],bounds,folder)
        previews.append((title,thumbnail))
        for key,name in [('board','nearby.jpg'),('material_regions','regions.png'),('surface_dust','dust.png')]:
            shutil.copy2(details[key],folder/name)
        audit = audit_png(row,best,comparison)
        audit.update(number=number,asset_id=row['asset_id'],status=row['status'],
                     photo_status=row['photo_status'],visual_status=row['visual_status'],
                     reference_status=details['status'],source_reference=details['manifest'],
                     uv_masks=details['uv_manifest'],measured_major_px=best['checks']['scale']['measurement_px'],
                     same_size_control=control.is_file())
        if control.is_file():
            audit['scale_change'] = {
                **candidate['user_scale'],
                'from_px':old['checks']['scale']['measurement_px'],
                'to_px':audit['measured_major_px'],
                'anchor_shift_px':float(np.linalg.norm(np.subtract(candidate['anchor_px'],candidate['user_scale']['previous_anchor_px'])))}
        audits.append(audit)
        scores = row['scores']
        review = best['review']
        defects = review['defects']
        defect_counts.update(d['code'] for d in defects)
        severe_surface += any(d['severity'] in ('major','fatal') and d['code'] in SURFACE_DEFECTS for d in defects)
        if row['photo_status']=='pass':
            class_passes[row['asset_id']]+=1
        image = Path(row['image']).relative_to(out).as_posix()
        loc = folder.relative_to(out).as_posix()
        ratio = f" · {audit['scale_change']['from_px']:g} → {audit['measured_major_px']:g} px" if control.is_file() else ''
        confidence = '局部材质已识别' if details['status']=='estimated' else '局部材质证据不足'
        scores_text = f'材质 {scores["material"]} · 光照 {scores["lighting"]} · 摄影质感 {scores["texture"]}'
        notes = '；'.join(d['evidence'] for d in defects[:4]) or review['evidence']
        cost_path = Path(row['image']).with_suffix('.cost.json')
        atomic_json(cost_path,{'batch_total':cost,'total_at_output':row.get('cost_total_at_output'),
                               'job':row['cost'],'api_all_attempts':usage.get(row['job_id']),
                               'timings':row.get('timings')})
        cards.append(f'<section id="n{number:02d}"><h2>{escaped}</h2>'
            f'<p>{html.escape(status+ratio+" · "+confidence+" · "+scores_text)}</p>'
            f'<div class="pair"><figure><img src="{loc}/before.jpg" loading="lazy" alt="原材质局部">'
            f'<figcaption>原材质（同尺寸）</figcaption></figure>'
            f'<figure><a href="{image}"><img src="{loc}/after.jpg" loading="lazy" alt="新材质局部"></a>'
            '<figcaption>适配附近废钢</figcaption></figure></div>'
            f'<p>{html.escape(notes)}</p><p><a href="{image}">原尺寸 PNG</a> · '
            f'<a href="{loc}/nearby.jpg">附近废钢参照</a> · <a href="{loc}/regions.png">材质分区</a> · '
            f'<a href="{loc}/dust.png">表面积灰</a> · '
            f'<a href="{cost_path.relative_to(out).as_posix()}">费用</a></p></section>')
    source_before = read(out/'source-hashes-before.json')
    source_after = read(out/'source-hashes-after.json')
    validated = [a for a in audits if 'size_ok' in a]
    safety = all(a['size_ok'] and a['verified_png_hash'] and a['same_alpha'] and
        a['same_visible_alpha'] and a['same_occlusion'] and not any(a[k] for k in
        ('outside_rectangle_changes','protected_changes','foreground_changes','outside_edit_changes')) for a in validated)
    photo_passes = sum(a.get('photo_status')=='pass' for a in audits)
    quality_gate = len(validated)==24 and photo_passes>=20 and len(class_passes)==12 and severe_surface==0
    validation = {'planned':24,'images':len(validated),'photo_pass':photo_passes,
        'photo_pass_by_asset':dict(class_passes),'severe_surface_images':severe_surface,
        'quality_gate':quality_gate,'defects':dict(defect_counts),'counts':summary['counts'],
        'source_model_count':len(source_before),'source_models_unchanged':source_before==source_after,
        'image_checks_pass':bool(validated) and safety,'size_control_count':sum(a['same_size_control'] for a in validated),
        'items':audits}
    atomic_json(out/'validation.json',validation)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    board = Image.new('RGB',(1440,64+220*((len(previews)+3)//4)),'#232b30')
    draw = ImageDraw.Draw(board)
    draw.text((18,16),'废钢材质、表面积灰与局部反光 · 点评图',font=font,fill='white')
    for i,(title,im) in enumerate(previews):
        x,y = i%4*360+12,64+i//4*220
        draw.text((x,y),title,font=font,fill='white')
        im.thumbnail((332,179),Image.Resampling.LANCZOS)
        board.paste(im,(x,y+30))
    board.save(out/'overview.jpg',quality=93)
    counts = summary['counts']
    lead = f'{len(validated)}/24 张出图 · {counts.get("accepted_display",0)} 张展示合格 · {counts.get("review",0)} 项待审'
    intro = ('液压缸见 10、22，灭火器见 11、23。两种材质使用相同的尺寸和摆位。'
             ' 分区图以蓝色标记金属，橙色标记漆面，棕红标记锈蚀，绿色标记塑料或橡胶。')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>废钢材质点评</title><style>body{background:#232b30;color:#eef2f4;font:16px system-ui,sans-serif;'
            'max-width:1440px;margin:32px auto;padding:0 22px}a{color:#9ddfc7}h1{font-size:26px}h2{font-size:19px}'
            'section{border-top:1px solid #536068;padding:18px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}'
            'figure{margin:0}img{max-width:100%;max-height:540px}figcaption{color:#becbd1;padding-top:6px}'
            'nav{padding:12px 0;line-height:2.3}nav a{padding-right:15px}p{line-height:1.55}</style>'
            '<h1>环境废钢材质点评</h1>')
    page += f'<p>{lead} · 摄影达标 {photo_passes}/24</p><p>{intro}</p><p>{html.escape(label(cost))}</p>'
    page += '<p><a href="overview.jpg">样片总览</a> · <a href="validation.json">验证记录</a> · <a href="cost-ledger.json">成本账本</a></p>'
    page += '<nav>'+''.join(f'<a href="#n{i:02d}">{i:02d}</a>' for i in range(1,25))+'</nav>'+''.join(cards)
    (out/'feedback.html').write_text(page,encoding='utf8')
    print(json.dumps({k:v for k,v in validation.items() if k!='items'},ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    write_report(resolve(parser.parse_args().out))
