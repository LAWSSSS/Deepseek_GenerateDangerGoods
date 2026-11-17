"""Frozen 4 x 3 realism experiment: numeric contact checks, A/B/C full frames."""
import argparse
import asyncio
import copy
import html
import json
from pathlib import Path
import time
import zipfile
from uuid import uuid4

from PIL import Image
from pydantic import Field
from display_batch.common import ROOT, HERE, atomic_json, file_hash, signature, load_catalog, resolve, implementation_manifest
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig, Strict, EffectsReview
from display_batch.compose import compose
from display_batch.costs import estimate
from display_batch.framing import final_resolution
from display_batch.structured import inspect_frame
from display_batch.strict_contact import visual_veto
from scrap_composite.photography_protocol import PhotoReport

ASSETS = ('3','22','31','33')
GROUPS = ('36_6_6','36_6_24','53_7_35')
SOURCE = ROOT/'outputs/全部启用物品-三场景-20261116'
DEFAULT_OUT = ROOT/'outputs/分层写实-12组-20261117'
DEFAULT_DIAGNOSTICS = ROOT/'outputs/资产分层写实诊断-20261117'
EXPERIMENT_VERSION = 'realism-abc-v2'
COMPARISON_IMAGE_VERSION = 'realism-controlled-image-v1'
BLIND_REVIEW_VERSION = 'realism-target-frame-local-v2'
# Upgrade only the known prior harness, after verifying every original input.
# Reporting/accounting edits must not invalidate an already rendered A/B image.
LEGACY_HARNESS_HASH = '1c8bf625cb2e75d9b376b56c21921100ae1bcc3991174b8d8c39154c585d44f9'
BLIND_PROMPT = ('Act as a documentary photographer. Compare ONLY the named inserted target inside target_region, '
    'not the quality of the original scrap background. Each X/Y/Z has one complete frame then its target crop. '
    'target_region is normalized in each FULL frame; crops show exactly that region at larger scale. '
    'Judge target material, local lighting and camera integration against adjacent same-depth surfaces. '
    'Preserve authentic material differences; more dirt is not inherently more realistic. '
    'Return best, proposal relative to baseline, and scores for the proposal target only. '
    'Give brief Chinese evidence locating a target feature, not generic background rust or dust.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def geometry_key(candidate):
    # Include masks, local receiver, camera and projection, not only pose/anchor.
    return signature(candidate)


def experiment_cost(out, vision, asset_diagnostics=None):
    """Merge C and comparison requests; the live ledger replaces its on-disk copy."""
    documents={}
    folders=[Path(out),Path(out)/'C']
    if asset_diagnostics is not None:folders.append(Path(asset_diagnostics))
    for folder in folders:
        for pattern in ('usage-*.json','comparison-usage-*.json'):
            for path in sorted(folder.glob(pattern)):
                documents[str(path.resolve())]=read(path)
    documents[str(Path(vision.usage_path).resolve())]=vision.usage()
    requests=tokens=unknown=hits=0;low=high=0.;complete=True;currencies=set();sources=set();mixed=False
    for doc in documents.values():
        cost=doc.get('cost') or estimate(doc.get('calls',[]),doc.get('tariff'))
        requests+=cost['actual_requests'];tokens+=cost.get('reported_tokens',doc.get('reported_tokens',0)) or 0
        unknown+=cost['unpriced_or_unreported_calls'];hits+=doc.get('cache_hits',0)
        if cost.get('currency'):currencies.add(cost['currency'])
        rates=cost.get('tariff') or doc.get('tariff') or {}
        if rates.get('source'):sources.add(rates['source'])
        mixed |= cost.get('mixed_currency',False)
        low+=cost.get('estimated_api_min') or 0.;high+=cost.get('known_portion_max') or 0.
        complete &= cost['complete']
    mixed |= len(currencies)>1
    return {'actual_requests':requests,'reported_tokens':tokens,'cache_hits':hits,
        'currency':next(iter(currencies)) if len(currencies)==1 else None,
        'estimated_api_min':None if mixed else round(low,8),
        'estimated_api_max':round(high,8) if complete and not mixed else None,
        'known_portion_max':None if mixed else round(high,8),'complete':complete and not mixed,
        'unpriced_or_unreported_calls':unknown,'ledger_files':sorted(documents),
        'local_compute_cost':None,'basis':'C_comparison_and_optional_asset_diagnostics_actual_requests_not_invoice',
        'asset_diagnostics':str(asset_diagnostics) if asset_diagnostics is not None else None,
        'sources':sorted(sources)}


def endpoint_identity(vision):
    # Endpoint configuration stores an environment-variable name, never the key.
    return vision.endpoint.model_dump(exclude={'key_env'})


def comparison_signature(pipe, asset, scene, candidate, params, seed, resolution, *, legacy=False):
    manifest=resolve(params['surface_finish_manifest'])
    implementation=(read(pipe.out/'implementation.json') if legacy else implementation_manifest())
    return signature(EXPERIMENT_VERSION,implementation,LEGACY_HARNESS_HASH if legacy else COMPARISON_IMAGE_VERSION,
        geometry_key(candidate),asset['mesh_hash'],asset['physical_size'],
        file_hash(asset['board']) if asset.get('board') else None,
        file_hash(scene['path']),scene.get('signature'),scene.get('key'),
        file_hash(manifest),params,seed,resolution,
        pipe.config.model_dump(exclude={'output','count','offline','manifest'}),endpoint_identity(pipe.vision))


def evidence_states(item):
    """Photography, seven-score visual quality and physical evidence remain separate."""
    checks=item.get('checks') or {}
    states=[c.get('status','unknown') for c in checks.values()]
    program='fail' if 'fail' in states else 'unknown' if not states or 'unknown' in states else 'pass'
    result={'photography_status':'unknown','visual_status':'unknown','program_evidence':program,
            'review_protocol':'unknown','deliverable':program!='fail' and item.get('deliverable') is not False}
    if item.get('physical_visual_veto') or item.get('physical_veto'):
        result.update(deliverable=False,program_evidence='fail')
    if not item.get('review'):
        return {**result,'review_reason':item.get('review_error') or 'review_missing_after_bounded_attempts'}
    try:
        # The strict protocol accepts JSON arrays for tuple fields in JSON mode.
        # Python-mode validation of json.loads output would reject every real region.
        report=PhotoReport.model_validate_json(json.dumps(item.get('photography_diagnosis') or item.get('review_evidence')))
        review=EffectsReview.model_validate_json(json.dumps(item.get('review')))
        if report.image_sha256!=item.get('image_hash'):raise ValueError('review_image_hash_mismatch')
        if report.scores.model_dump()!=review.scores.model_dump():raise ValueError('review_scores_changed')
        if report.reconciled or checks.get('review_evidence_consistency',{}).get('status')=='unknown':
            raise ValueError('review_reconciled_or_conflicting')
    except (ValueError,TypeError) as error:
        return {**result,'review_reason':str(error)[:120]}
    veto=visual_veto(review) or item.get('physical_visual_veto') or item.get('physical_veto')
    if veto:result.update(deliverable=False,program_evidence='fail')
    visible_severe=any(d.severity in ('major','fatal') and d.visibility=='visible' for d in review.defects)
    result['review_protocol']='pass'
    if result['deliverable']:
        result['photography_status']='pass' if review.photo_passes() and not visible_severe else 'fail'
        result['visual_status']='pass' if review.passes() else 'fail'
    else:result.update(photography_status='fail',visual_status='fail')
    return result


def validate_blind(answer,order):
    winner=order['XYZ'.index(answer.best)]
    if (winner=='C' and answer.relative_to_baseline=='worse') or (
            winner=='A' and answer.relative_to_baseline=='better'):
        raise ValueError('best_and_relative_comparison_disagree')


def blind_signature(vision, versions, order, payload):
    return signature(BLIND_REVIEW_VERSION,BLIND_PROMPT,BlindComparison.model_json_schema(),payload,
        endpoint_identity(vision),[file_hash(versions[v][kind]) for v in order for kind in ('image','crop')],
        [(1536,864),(1024,1024)]*3,True)


def records(source):
    return [json.loads(line) for line in (source/'results.jsonl').read_text(encoding='utf8').splitlines() if line]


def proposal_entry(row, output):
    """Expose a checked draft as unscored review evidence; never rescue physical failure."""
    if row.get('status')=='failed':return None
    if row.get('image'):
        chosen=next((e for e in row.get('iterations',[]) if e['iteration']==row.get('best_iteration')),None)
        if chosen is None:return None
        chosen=copy.deepcopy(chosen);chosen['image']=row['image']
    elif row.get('unreviewed_draft') and row.get('draft_record'):
        path=Path(row['draft_record']).resolve()
        if not path.is_relative_to(Path(output).resolve()) or not path.is_file():return None
        chosen=read(path)
        chosen.update(review=None,scores=None,review_status='unknown',
            review_error=row.get('reason','review_missing_after_bounded_attempts'),
            draft_record=str(path),unreviewed_draft=True)
    else:return None
    checks=chosen.get('checks') or {}
    if not checks or any(c.get('status')=='fail' for c in checks.values()):return None
    if chosen.get('deliverable') is False or chosen.get('physical_visual_veto') or chosen.get('physical_veto'):return None
    image=Path(chosen.get('image','')).resolve()
    if not image.is_relative_to(Path(output).resolve()) or not image.is_file():return None
    digest=file_hash(image)
    if chosen.get('image_hash')!=digest:return None
    chosen['scores']=(chosen.get('review') or {}).get('scores')
    return chosen


def freeze(source, out):
    inputs=read(source/'inputs.json')
    inputs=[x for x in inputs if x['asset_id'] in ASSETS and x['group'] in GROUPS]
    if len(inputs)!=12 or len({(x['asset_id'],x['group']) for x in inputs})!=12:
        raise ValueError('expected frozen four assets by three scenes')
    inputs=[{k:v for k,v in x.items() if k in ('asset_id','background','group','seed')} for x in inputs]
    catalog=load_catalog()
    frozen={'source':str(source),'inputs':inputs,'source_models':{k:v['mesh_hash'] for k,v in catalog.items()},
            'source_summary_sha256':file_hash(source/'summary.json')}
    path=out/'frozen.json'
    if path.is_file() and read(path)!=frozen:raise ValueError('frozen experiment inputs changed')
    atomic_json(path,frozen);atomic_json(out/'manifest.json',inputs)
    return frozen


class PilotPipeline(Pipeline):
    def __init__(self,config,source):
        super().__init__(config)
        self.frozen_geometry={}
        for row in records(source):
            if row['asset_id'] not in ASSETS:continue
            entries=[e for e in row.get('iterations',[]) if not any(
                v['status']=='fail' for v in e['checks'].values())]
            if not entries:continue
            entry=entries[0]
            data=read(source/'jobs'/row['job_id']/'candidates.json')
            candidate=next((c for c in data['candidates'] if c['id']==entry['params']['candidate']),None)
            if candidate:
                self.frozen_geometry[(row['asset_id'],str(Path(row['background']).resolve()))]=candidate

    async def choose_placements(self,asset,scene,seed):
        candidates,rejected=await super().choose_placements(asset,scene,seed)
        old=self.frozen_geometry.get((str(asset['id']),str(Path(scene['path']).resolve())))
        if old is not None:
            # Numeric validation in Pipeline.job runs again before rendering.
            fixed=copy.deepcopy(old)
            candidates=[fixed]+[c for c in candidates if c['id']!=fixed['id']]
        return candidates,rejected


class BlindComparison(Strict):
    best: str = Field(pattern=r'^[XYZ]$')
    relative_to_baseline: str = Field(pattern=r'^(better|similar|worse)$')
    material: int = Field(ge=0,le=4)
    lighting: int = Field(ge=0,le=4)
    photography: int = Field(ge=0,le=4)
    evidence: str = Field(max_length=200)


async def comparisons(pipe, source, out, rows):
    old={(r['asset_id'],r['background']):r for r in records(source)}
    results=[]
    for row in rows:
        key=row['group']+'-'+row['asset_id']
        folder=out/'comparisons'/key;folder.mkdir(parents=True,exist_ok=True)
        record={'key':key,'asset_id':row['asset_id'],'group':row['group'],'status':row['status'],
                'reason':row['reason'],'versions':{}}
        chosen=proposal_entry(row,pipe.out)
        if chosen is None:
            results.append(record);atomic_json(folder/'result.json',record);continue
        candidate=next(c for c in read(pipe.out/'jobs'/row['job_id']/'candidates.json')['candidates']
                       if c['id']==chosen['params']['candidate'])
        asset=await pipe.prepare_asset(row['asset_id'])
        size_path=pipe.out/'scale-models'/(str(row['asset_id'])+'.json')
        if not size_path.is_file():raise ValueError('missing_C_physical_size:'+str(row['asset_id']))
        asset['physical_size']=read(size_path)
        if not asset['physical_size'].get('nominal_mm'):raise ValueError('invalid_C_physical_size')
        scene=await pipe.scenes.prepare(row['background'])
        history=old[(row['asset_id'],row['background'])]
        old_entries=history.get('iterations',[])
        old_entry=next((e for e in old_entries if e['params']['candidate']==candidate['id']),None)
        base_params=copy.deepcopy(old_entry['params'] if old_entry else pipe.default_params(candidate['id']))
        base_params={**base_params,'surface_finish':'generated_pbr',
                     'surface_finish_manifest':'work/render-materials/nearby-worn-steel-v15/pack.json',
                     'candidate':candidate['id'],'review_profile':'photographer',
                     'render_detail':chosen['params'].get('render_detail','standard')}
        base_params.pop('realism_version',None)
        base_params.pop('diagnostic_surface_context',None)
        record['geometry_signature']=geometry_key(candidate)
        record['physical_size']=asset['physical_size']
        record['versions']['C']={**chosen,'geometry_signature':record['geometry_signature']}
        record['versions']['C'].update(evidence_states(record['versions']['C']))
        seed=next(j['seed'] for j in read(out/'manifest.json') if j['asset_id']==row['asset_id'] and j['group']==row['group'])
        resolution=final_resolution(candidate)
        for version in ('A','B'):
            target=folder/(version+'.png')
            params=base_params if version=='A' else {**base_params,'surface_finish':'physical_layers',
                'diagnostic_surface_context':'v15'}
            cache_key=comparison_signature(pipe,asset,scene,candidate,params,seed,resolution)
            previous=folder/(version+'.json')
            if previous.is_file():
                cached=read(previous)
                legacy_key=comparison_signature(pipe,asset,scene,candidate,params,seed,resolution,legacy=True)
                if (cached.get('comparison_signature') in (cache_key,legacy_key) and target.is_file()
                        and file_hash(target)==cached.get('image_hash')):
                    cached.update(evidence_states(cached))
                    if cached['comparison_signature']!=cache_key:
                        cached.update(comparison_signature=cache_key,cache_migrated_from=legacy_key,
                            cache_migration_basis='frozen_reviewed_control_validated_against_C_archived_implementation',
                            image_implementation_signature=signature(read(pipe.out/'implementation.json')),
                            reused_prior_reviewed_control=True)
                        atomic_json(previous,cached)
                    record['versions'][version]=cached;continue
            # Old images with only matching geometry do not establish material,
            # camera-processing or implementation identity. RenderCache decides reuse.
            started=time.monotonic()
            render,meta,environment,samples,hit=await pipe.render(asset,scene,candidate,params,resolution)
            render_seconds=time.monotonic()-started
            # B isolates material from C's removal of the old surface matching.
            # A and B use the same sampled weathered-steel evidence for this diagnostic.
            if version=='B' and 'nearby_weathered_steel' not in environment:
                import numpy as np
                from display_batch.generated_material import finish_regions,load_pack
                from display_batch.surface_context import nearby_weathered_steel
                old_regions=finish_regions(asset,load_pack(resolve(params['surface_finish_manifest'])),pipe.cache)
                if any(m.get('authored_atlas') and m['coverage'][2]>0 for m in old_regions['materials'].values()):
                    environment={**environment,'nearby_weathered_steel':nearby_weathered_steel(
                        scene,candidate,np.asarray(scene['frame'])[samples])}
            started=time.monotonic()
            item=await asyncio.to_thread(compose,scene,candidate,render,meta,environment,samples,params,seed,target)
            item.update(image_hash=file_hash(target),geometry_signature=record['geometry_signature'],params=params,
                comparison_signature=cache_key,physical_size_sha256=file_hash(size_path),render_resolution=meta.get('resolution'),
                render_cache_hit=hit,render_cache=meta.get('render_cache',{}),render_timings=meta.get('render_timings',{}),
                timings={'render_stage_seconds':render_seconds,'compose_seconds':time.monotonic()-started})
            try:
                # Review the actual PNG. Never transfer old picture scores.
                observation,review=await inspect_frame(pipe.vision,asset,item,item['image_hash'],profile='photographer')
                item.update(review=review.model_dump(),scores=review.scores.model_dump(),
                            review_evidence=observation.model_dump(mode='json'),physical_veto=visual_veto(review))
            except Exception as error:
                if type(error).__name__ not in ('InvalidStructuredReply',):raise
                item.update(review_status='unknown',review_error=str(error)[:200])
            item.update(evidence_states(item))
            atomic_json(previous,item);record['versions'][version]=item
        # A/C use separate shader algorithms and hence coverage is checked, never assumed.
        record['control_checks']={field:{v:record['versions'][v].get(field) for v in ('A','B','C')}
            for field in ('alpha_hash','visible_alpha_hash','foreground_mask_hash','measured_major_px','render_resolution')}
        record['control_changes']={v:{k:{'A':base_params.get(k),'value':record['versions'][v]['params'].get(k)}
            for k in sorted(set(base_params)|set(record['versions'][v]['params']))
            if base_params.get(k)!=record['versions'][v]['params'].get(k)} for v in ('B','C')}
        record['same_geometry']=all(record['versions'][v]['geometry_signature']==record['geometry_signature'] for v in ('A','B','C'))
        # Deterministic order is independent of outcome.
        order=('C','A','B') if int(row['asset_id'])%2 else ('B','C','A')
        width,height=chosen['checks']['size']['pixels']
        left,top,right,bottom=chosen['edit_bounds']
        payload={'labels':['X full','X target crop','Y full','Y target crop','Z full','Z target crop'],
                 'baseline':('XYZ')[order.index('A')],'proposal':('XYZ')[order.index('C')],
                 'target_name':asset['name'],'target_region':[left/width,top/height,right/width,bottom/height],
                 'material_classes':asset.get('semantic',{}).get('material_classes',[]),
                 'review_version':BLIND_REVIEW_VERSION}
        blind_key=blind_signature(pipe.vision,record['versions'],order,payload)
        blind_path=folder/'blind-comparison.json'
        cached_blind=read(blind_path) if blind_path.is_file() else {}
        try:
            if cached_blind.get('signature')==blind_key and cached_blind.get('status')=='unknown':
                record['comparison_cache_hit']=True
                record['comparison']={'status':'unknown','error':cached_blind.get('error','review_missing')}
                record['comparison_signature']=blind_key
                results.append(record);atomic_json(folder/'result.json',record)
                report(out,results,pipe)
                continue
            answer=None
            if cached_blind.get('signature')==blind_key and 'answer' in cached_blind:
                try:
                    answer=BlindComparison.model_validate(cached_blind['answer']);validate_blind(answer,order)
                except ValueError:answer=None
            if answer is not None:record['comparison_cache_hit']=True
            else:
                answer=await pipe.vision.ask('realism-blind-compare',BlindComparison,BLIND_PROMPT,payload,
                    [record['versions'][v][kind] for v in order for kind in ('image','crop')],
                    image_limits=[(1536,864),(1024,1024)]*3,
                    plain_json=True,validate=lambda value:validate_blind(value,order))
                atomic_json(blind_path,{'signature':blind_key,'answer':answer.model_dump(),
                    'image_hashes':{v:{kind:file_hash(record['versions'][v][kind]) for kind in ('image','crop')} for v in order},
                    'order':order,'review_version':BLIND_REVIEW_VERSION,'target':payload})
                record['comparison_cache_hit']=False
            record['comparison']={**answer.model_dump(),'order':dict(zip('XYZ',order))}
        except Exception as error:
            if type(error).__name__!='InvalidStructuredReply':raise
            record['comparison']={'status':'unknown','error':str(error)[:200]}
            atomic_json(blind_path,{'signature':blind_key,**record['comparison'],
                'image_hashes':{v:{kind:file_hash(record['versions'][v][kind]) for kind in ('image','crop')} for v in order},
                'order':order,'review_version':BLIND_REVIEW_VERSION,'target':payload})
        record['comparison_signature']=blind_key
        record['comparison_version']=BLIND_REVIEW_VERSION
        results.append(record);atomic_json(folder/'result.json',record)
        report(out,results,pipe)
        print(json.dumps({'comparison':key,'status':record.get('comparison')},ensure_ascii=False),flush=True)
    return results


def report(out,results,pipe):
    summary=read(pipe.out/'summary.json')
    images=sum(bool(r['versions'].get('C')) for r in results)
    passed=[]
    visual_passed=[];evidence_counts={'pass':0,'unknown':0,'fail':0};statuses={}
    for r in results:
        c=r['versions'].get('C',{})
        state=evidence_states(c) if c else {'photography_status':'fail','visual_status':'fail',
            'program_evidence':'fail','review_protocol':'unknown','deliverable':False}
        if r['status']=='failed':state.update(photography_status='fail',visual_status='fail',deliverable=False)
        statuses[r['key']]=state;evidence_counts[state['program_evidence']]+=1
        if state['photography_status']=='pass':passed.append(r['key'])
        if state['visual_status']=='pass':visual_passed.append(r['key'])
    groups={g:sum(k.startswith(g+'-') for k in passed) for g in GROUPS}
    evaluation={'total':12,'reported':len(results),'images':images,'photography_without_severe_defects':len(passed),
                'passed_keys':passed,'groups':groups,'target_met':len(results)==12 and len(passed)>=10 and min(groups.values())>=3,
                'seven_score_visual_passes':len(visual_passed),'seven_score_passed_keys':visual_passed,
                'program_evidence_counts':evidence_counts,'item_statuses':statuses,
                'comparison_completed':sum('best' in r.get('comparison',{}) for r in results),
                'comparison_policy':'blind_three_scores_are_diagnostic_only; seven_score_and_program_gates_are_independent',
                'unscored_review_images':sum(bool(r['versions'].get('C',{}).get('unreviewed_draft')) for r in results),
                'production_default_changed':False,'cost_total':experiment_cost(out,pipe.vision,getattr(pipe,'asset_diagnostics',None)),
                'pipeline_summary':summary}
    atomic_json(out/'evaluation.json',evaluation)
    parts=['<!doctype html><meta charset="utf-8"><title>分层写实12组对照</title><style>body{max-width:1600px;margin:24px auto;background:#172128;color:#eee;font:16px sans-serif;padding:20px}img{width:100%;height:auto}a{color:#9ee7cf}section{border-top:1px solid #666;padding:20px 0}</style>',
           '<h1>分层写实：12组完整画面对照</h1><p>A 最新v15基线；B 分层材质＋原v15后置处理（仅对照启用）；C 分层材质＋场景成像。'
           '三组使用C的同一候选、实物尺寸与采样分辨率；旧v15含旧焊缝修饰，遮罩与尺寸差异另列。现场尺度与隐藏接触仍可能未知。'
           '三图偏好比较不能替代七项评分、程序接触与像素检查。已通过程序门槛但模型回复无效的草稿仅作待审图，不计摄影或视觉达标；不追加C评审请求。</p>',
           '<pre>'+html.escape(json.dumps({k:v for k,v in evaluation.items() if k!='pipeline_summary'},ensure_ascii=False,indent=2))+'</pre>']
    for r in results:
        parts.append('<section><h2>'+html.escape(r['key'])+'</h2><p>'+html.escape(r['reason'])+'</p>')
        for v in ('A','B','C'):
            item=r['versions'].get(v)
            if not item:continue
            uri=Path(item['image']).resolve().relative_to(out.resolve()).as_posix()
            annotation=' · 评审缺失，待审，无评分' if item.get('unreviewed_draft') else ''
            parts.append('<h3>'+v+annotation+'</h3><a href="'+html.escape(uri)+'"><img loading="lazy" src="'+html.escape(uri)+'"></a>')
        parts.append('<pre>'+html.escape(json.dumps(r.get('comparison',{}),ensure_ascii=False,indent=2))+'</pre></section>')
    (out/'index.html').write_text(''.join(parts),encoding='utf8')
    return evaluation


async def run(args):
    started=time.monotonic();source=args.source.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=freeze(source,out)
    raw=read(source/'config.json');raw.update(output=str(out/'C'),manifest=str(out/'manifest.json'),
        preset='realism_v1',count=12,assets=list(ASSETS),max_iterations=2,cpu_concurrency=1,ai_concurrency=1)
    raw['effects']['ai_adjust']=True
    config=BatchConfig.model_validate(raw);atomic_json(out/'config.json',config.model_dump())
    if args.freeze_only:return
    c_out=resolve(config.output)
    if not (c_out/'summary.json').is_file() or read(c_out/'summary.json')['run']['state']!='complete':
        pipe=PilotPipeline(config,source)
        await pipe.run()
    # run() closes clients; comparisons are a separate accounted execution.
    pipe=PilotPipeline(config,source)
    pipe.asset_diagnostics=args.asset_diagnostics.resolve() if args.asset_diagnostics is not None else None
    pipe.vision.usage_path=out/('usage-comparison-'+pipe.run_id+'-'+uuid4().hex[:8]+'.json')
    try:
        rows=records(pipe.out)
        expected={(x['asset_id'],x['group']) for x in frozen['inputs']}
        if len(rows)!=12 or {(r['asset_id'],r['group']) for r in rows}!=expected:
            raise ValueError('C results do not contain exactly the 12 frozen combinations')
        rows=sorted(rows,key=lambda r:(GROUPS.index(r['group']),ASSETS.index(r['asset_id'])))
        results=await comparisons(pipe,source,out,rows)
        evaluation=report(out,results,pipe)
        now=load_catalog();changed=[k for k,v in frozen['source_models'].items() if now[k]['mesh_hash']!=v]
        sizes=[]
        for r in results:
            for v,item in r['versions'].items():
                background=next(j['background'] for j in frozen['inputs'] if j['asset_id']==r['asset_id'] and j['group']==r['group'])
                with Image.open(item['image']) as im,Image.open(background) as bg:
                    if im.size!=bg.size:sizes.append(r['key']+'-'+v)
        atomic_json(out/'output-audit.json',{'source_models_checked':len(now),'changed':changed,'size_failures':sizes})
        if changed or sizes:raise ValueError('output invariants failed')
        target=Path.home()/'Desktop'/'废钢合成-分层写实12组-20261117.zip'
        if target.exists():target=target.with_name(target.stem+'-'+time.strftime('%H%M%S')+'-'+uuid4().hex[:6]+'.zip')
        finals=[r for r in results if 'C' in r['versions'] and evidence_states(r['versions']['C'])['deliverable']
                and r['status']!='failed']
        if finals:
            with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for r in finals:z.write(r['versions']['C']['image'],arcname=r['key']+'.png')
            with zipfile.ZipFile(target) as z:
                if z.testzip() is not None:raise ValueError('zip checksum error')
        evaluation.update(wall_seconds=time.monotonic()-started,zip=str(target) if finals else None)
        atomic_json(out/'evaluation.json',evaluation)
    finally:
        await pipe.blender.close();await pipe.vision.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--asset-diagnostics',type=Path,default=DEFAULT_DIAGNOSTICS,
                        help='Include this asset-diagnostic directory\'s actual request ledgers in the round cost')
    parser.add_argument('--no-asset-diagnostics',action='store_const',dest='asset_diagnostics',const=None)
    asyncio.run(run(parser.parse_args()))
