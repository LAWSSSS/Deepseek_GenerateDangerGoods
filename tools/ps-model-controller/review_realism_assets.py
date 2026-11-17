"""Five bounded asset reviews using the independent batch Vision service, no render."""
import argparse
import asyncio
import html
import json
from pathlib import Path
import time
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field, model_validator

from display_batch.ai import Vision, InvalidStructuredReply
from display_batch.common import atomic_json, file_hash, resolve, signature
from display_batch.schema import Strict, BatchConfig, Unit


VERSION='realism-isolated-asset-review-v1'
TAGS=('clay-left','clay-right','source-left','source-right',
      'physical-left','physical-right','physical-turned')
ASSETS=('3','13','22','31','33')


class Finding(Strict):
    view: Literal['clay-left','clay-right','source-left','source-right',
                  'physical-left','physical-right','physical-turned']
    region: tuple[Unit,Unit,Unit,Unit]
    kind: Literal['silhouette','surface_geometry','part_missing','material','texture','identity']
    severity: Literal['minor','major']
    reference: Literal['original','model_board','controlled_plates','unassessable']
    evidence: str=Field(max_length=100)

    @model_validator(mode='after')
    def ordered(self):
        x,y,r,b=self.region
        if x>=r or y>=b:raise ValueError('region must be ordered within the named individual 512 plate')
        return self


class Observation(Strict):
    status: Literal['consistent','inconsistent','unknown','not_applicable']
    evidence: str=Field(max_length=100)


class AssetReview(Strict):
    asset_id: str
    geometry: Literal['no_visible_defect','confirmed_defect','unknown']
    findings: list[Finding]=Field(max_length=5)
    source_material: int=Field(ge=0,le=4)
    physical_material: int=Field(ge=0,le=4)
    physical_texture: int=Field(ge=0,le=4)
    plastic_protection: Observation
    texture_turning: Observation
    light_response: Observation
    main_cause: Literal['geometry','material','both','unresolved']
    recipe_decision: Literal['ready_for_scene_trial','adjust_material_first','geometry_candidate_needed','unknown']
    actions: list[Literal['inspect_geometry_reference','reduce_uniform_deposit','retain_metal_reflection',
                         'vary_wear_scale','protect_label_polymer','inspect_more_views','no_change']]=Field(max_length=3)
    summary: str=Field(max_length=180)

    @model_validator(mode='after')
    def supported_geometry(self):
        if self.geometry=='confirmed_defect' and not any(
            f.kind in ('silhouette','surface_geometry','part_missing') and
            f.reference in ('original','model_board') for f in self.findings):
            raise ValueError('confirmed geometry requires a localized geometry finding and identity reference')
        if self.recipe_decision=='geometry_candidate_needed' and self.geometry!='confirmed_defect':
            raise ValueError('unknown geometry does not justify reconstruction')
        return self


PROMPT='''Act as a professional product photographer and PBR asset diagnostician. Inspect the supplied
controlled preview atlas, original identity photograph, and existing model-view board. Atlas cells are
labelled clay-left/right, source-left/right, physical-left/right, physical-turned (+55 degrees).
Each plate is complete, with two broad neutral lights, no scene fog, grading or camera blur.
Original photograph and model board are IDENTITY/GEOMETRY references only, never standards for
exposure, reflection brightness, haze or camera sharpness. A board may contain old/new synthesized
views: it is secondary evidence, not ground-truth shape. Do not infer hidden or foreshortened parts lost.
First isolate shape defects visible in clay and supported by the original. Then compare source PBR
with physical layers: patchy oxidation/deposition, sparse scratches, curved metal reflections,
paint/markings retained, and plastic/rubber not metalized. More dirt is not inherently realistic.
regions are normalized coordinates INSIDE the named 512x512 plate, not the atlas. Do not call a
smooth cylinder a geometry defect without direct photo evidence. One turned view may not establish
texture attachment; use unknown if correspondence is invisible. Assess changed-light response.
Score 0..4, 3 usable for the next scene trial; do not claim scene scale/contact/fusion passes.
Return concise Chinese evidence, explicit uncertainties, at most 3 general actions. This decides
whether a UNIFORM material recipe is suitable for a 12-case trial, not per-object hand-tuning.'''


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def audit_evidence(folder,row):
    """Keep raw scores; numerical contradictions can only lower actionability."""
    review=row.get('review',{});ident=row['asset_id'];comparisons={};flags=[]
    for light in ('left','right'):
        with Image.open(folder/ident/('source-'+light+'.png')) as im:source=np.asarray(im).copy()
        with Image.open(folder/ident/('physical-'+light+'.png')) as im:physical=np.asarray(im).copy()
        core=(source[...,3]>250)&(physical[...,3]>250)
        delta=np.abs(source[...,:3].astype(np.int16)-physical[...,:3].astype(np.int16))
        changed=np.any(delta>1,axis=2)&core
        comparisons[light]={'core_pixels':int(core.sum()),'alpha_equal':bool(np.array_equal(source[...,3],physical[...,3])),
            'rgb_mae_255':float(delta[core].mean()) if core.any() else None,
            'rgb_max_255':int(delta[core].max()) if core.any() else None,
            'pixels_changed_over_one_code':int(changed.sum()),
            'changed_fraction_over_one_code':float(changed.sum()/max(1,core.sum()))}
    visually_same=all(x['rgb_max_255'] is not None and x['rgb_max_255']<=1 for x in comparisons.values())
    if visually_same and any(f['kind'] in ('material','texture','identity') and f['view'].startswith('physical')
                            for f in review.get('findings',[])):
        flags.append({'code':'source_physical_equal_within_one_code',
                      'evidence':'同光源原PBR与分层材质至多相差1/255；不能把原有斑点归因为新增沉积。'})
    finding_checks=[]
    for index,finding in enumerate(review.get('findings',[])):
        with Image.open(folder/ident/(finding['view']+'.png')) as im:alpha=np.asarray(im.getchannel('A'))
        h,w=alpha.shape;x,y,r,b=finding['region'];left,top,right,bottom=round(x*w),round(y*h),round(r*w),round(b*h)
        pixels=int((alpha[top:bottom,left:right]>127).sum())
        box_area=max(1,(right-left)*(bottom-top))
        check={'index':index,'named_plate_opaque_pixels':pixels,'opaque_fraction':pixels/box_area,
               'status':'localized' if pixels>=9 else 'unknown'}
        if pixels<9:
            flags.append({'code':'region_not_localized_on_named_plate','finding':index,
                          'evidence':'给定区域几乎没有目标像素，疑似误用图板坐标；不能作为可执行部位。'})
        if finding['kind'] in ('material','texture') and (
                finding['reference'] in ('original','model_board') or any(s in finding['evidence'] for s in ('原图','原作','棚拍'))):
            flags.append({'code':'identity_reference_used_as_appearance_standard','finding':index,
                          'evidence':'身份参照不能决定受控宽光下的反射形态或沉积优劣。'})
        if finding['kind']=='identity':
            flags.append({'code':'new_identity_loss_unproven','finding':index,
                          'evidence':'缺少相同可见部位的源PBR→分层材质标识损失证据；背面及低分辨率不可辨不算新增损坏。'})
        finding_checks.append(check)
    light=review.get('light_response',{}).get('evidence','')
    if any(s in light for s in ('原图','原作','棚拍')):
        flags.append({'code':'identity_reference_used_as_light_standard',
                      'evidence':'窄条棚拍高光不能作为两宽面光下必须出现的反射。'})
    if review.get('geometry')=='confirmed_defect':
        flags.append({'code':'two_lights_not_two_geometric_views',
                      'evidence':'左右光使用同一相机，重复不可见不能证明部件缺失；另须核对已确认模型修订与参考示意性质。'})
    return {'version':'asset-evidence-audit-v1','source_physical':comparisons,
            'source_physical_within_one_code':visually_same,'finding_localization':finding_checks,
            'flags':flags,'decision':'unknown' if flags else review.get('recipe_decision','unknown'),
            'geometry_reconstruction_authorized':False,
            'program_gate':'diagnostic_only_no_asset_quarantine_or_automatic_rebuild',
            'raw_model_scores_preserved':True}


def make_atlas(folder,case):
    """Whole diagnostic plates, composited onto neutral grey without cropping."""
    canvas=Image.new('RGB',(1024,4*544),(112,112,112));draw=ImageDraw.Draw(canvas)
    for i,tag in enumerate(TAGS):
        x,y=(i%2)*512,(i//2)*544
        with Image.open(folder/case['asset_id']/(tag+'.png')) as image:
            if image.size!=(512,512):raise ValueError('diagnostic plate must remain full 512x512')
            canvas.paste(image.convert('RGB'),(x,y+32),image.getchannel('A'))
        draw.text((x+12,y+10),tag,fill=(245,245,245))
    draw.text((526,1660),'WHOLE PLATES / NO FOG / NO CAMERA BLUR',fill=(245,245,245))
    path=folder/'review'/('asset-'+case['asset_id']+'-plates.png')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return path


def report(folder,rows,usage):
    body=['<!doctype html><meta charset="utf-8"><title>隔离资产评审</title>',
          '<style>body{max-width:1500px;margin:24px auto;background:#eee;color:#222;font:16px sans-serif}img{max-width:100%;height:auto}pre{white-space:pre-wrap}section{border-top:1px solid #aaa;padding:20px}</style>',
          '<h1>五资产隔离评审</h1><p>完整受控预览；原图仅核对身份与形状。本页不代表现场尺度、接触或最终融合通过。</p>',
          '<pre>'+html.escape(json.dumps({k:usage.get(k) for k in ('actual_requests','reported_tokens','cost','render_count','wall_seconds')},ensure_ascii=False,indent=2))+'</pre>']
    for row in rows:
        uri=Path(row['atlas']).relative_to(folder).as_posix()
        body.extend(['<section><h2>'+html.escape(row['asset_id'])+'</h2>',
                     '<a href="'+uri+'"><img src="'+uri+'"></a>',
                     '<pre>'+html.escape(json.dumps({'raw_model_review':row.get('review'),
                         'evidence_consistency':row.get('evidence_consistency')},ensure_ascii=False,indent=2))+'</pre></section>'])
    (folder/'asset-review.html').write_text(''.join(body),encoding='utf8')


async def run(args):
    folder=resolve(args.folder); configuration=resolve(args.config)
    frozen=read(folder/'frozen.json'); cases=frozen['cases']
    if tuple(c['asset_id'] for c in cases)!=ASSETS:raise ValueError('expected the frozen five assets, no expansion')
    result_rows={r['asset_id']:r for r in read(folder/'results.json')}
    if sum(len(r['renders']) for r in result_rows.values())!=35:raise ValueError('all 35 diagnostic plates required')
    if args.audit_only:
        rows=read(folder/'asset-reviews.json')
        for row in rows:
            row['evidence_consistency']=audit_evidence(folder,row)
            atomic_json(folder/'review'/('asset-'+row['asset_id']+'.json'),row)
        atomic_json(folder/'asset-reviews.json',rows)
        report(folder,rows,read(folder/'usage-asset-review.json'))
        print(json.dumps({'audited':len(rows),'new_requests':0,'new_renders':0}))
        return
    config=BatchConfig.model_validate(read(configuration));config.ai_concurrency=1
    ledger=folder/'usage-asset-review.json'
    # Preserve prior paid attempts on resume. Vision deduplicates image reviews in cache.
    previous=read(ledger) if ledger.is_file() else {}
    vision=Vision(config,resolve(config.cache)/'vision',ledger)
    vision.calls=list(previous.get('calls',[]));vision.cache_hits=int(previous.get('cache_hits',0))
    started=time.monotonic();rows=[];budget=set(c['request_signature'] for c in vision.calls if 'request_signature' in c)
    try:
        for case in cases:
            ident=case['asset_id']; asset=read(case['asset_record'])
            if file_hash(case['asset_record'])!=case['asset_record_hash'] or file_hash(asset['mesh'])!=case['mesh_hash']:
                raise ValueError('frozen asset changed before review')
            atlas=make_atlas(folder,case)
            originals=[Path(asset['original']),Path(asset['board'])]
            if not all(p.is_file() for p in originals):raise FileNotFoundError('identity reference unavailable')
            record_path=folder/'review'/('asset-'+ident+'.json')
            semantic={'version':VERSION,'asset_id':ident,'asset_record_hash':case['asset_record_hash'],
                'images':[file_hash(p) for p in [atlas,*originals]],'model':vision.endpoint.model,
                'schema':AssetReview.model_json_schema(),'prompt':PROMPT}
            review_signature=signature(semantic)
            if record_path.is_file() and read(record_path).get('signature')==review_signature:
                cached=read(record_path);cached['evidence_consistency']=audit_evidence(folder,cached)
                atomic_json(record_path,cached);rows.append(cached);continue
            if len(budget)>=5:
                raise ValueError('five logical asset-review budget exhausted')
            row={'asset_id':ident,'atlas':str(atlas),'signature':review_signature,
                 'identity_references':[str(p) for p in originals],'scope':'isolated_asset_diagnostic_not_scene_acceptance'}
            payload={'asset_id':ident,'name':case['name'],'roles':['controlled_preview_atlas','identity_original','identity_model_board'],
                     'plate_labels':list(TAGS),'material_classes':asset.get('semantic',{}).get('material_classes',[]),
                     'reported_program_checks':result_rows[ident]['checks']}
            def validate(answer):
                if answer.asset_id!=ident:raise ValueError('asset_id must match current asset')
            try:
                answer=await vision.ask('realism-isolated-asset-review',AssetReview,PROMPT,payload,[atlas,*originals],
                    image_limits=[(1024,2176),(1280,1280),(1536,1024)],plain_json=True,
                    max_output_tokens=2048,validate=validate)
                row.update(status='reviewed',review=answer.model_dump())
            except InvalidStructuredReply as error:
                row.update(status='unknown',reason=str(error))
            budget=set(c['request_signature'] for c in vision.calls if 'request_signature' in c)
            if len(vision.calls)>10:raise AssertionError('request budget exceeded')
            row['evidence_consistency']=audit_evidence(folder,row)
            atomic_json(record_path,row);rows.append(row)
            atomic_json(folder/'asset-reviews.json',rows)
            print(json.dumps({'asset_id':ident,'status':row['status'],
                              'recipe':row.get('review',{}).get('recipe_decision')},ensure_ascii=False),flush=True)
    finally:
        await vision.close()
        usage=vision.usage();usage.update(version=VERSION,render_count=0,render_seconds=0.,
            actual_requests=len(vision.calls),
            logical_reviews=len(set(c.get('request_signature') for c in vision.calls)),
            logical_review_limit=5,actual_request_limit=10,repair_limit_per_request=1,
            wall_seconds=round(previous.get('wall_seconds',0)+time.monotonic()-started,3),
            provider_credits=None,actual_billed_amount=None,
            reference_scope='identity_photos_are_not_lighting_standards')
        atomic_json(ledger,usage);atomic_json(folder/'asset-reviews.json',rows);report(folder,rows,usage)
    print(json.dumps({'reviewed':len(rows),'requests':usage.get('actual_requests'),
                      'tokens':usage.get('reported_tokens'),'cost':usage['cost']},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',default='outputs/资产分层写实诊断-20261117')
    parser.add_argument('--config',default='outputs/分层写实-12组-20261117/config.json')
    parser.add_argument('--audit-only',action='store_true')
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__':main()
