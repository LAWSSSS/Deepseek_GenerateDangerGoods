"""Actual-request accounting; published tariff estimates are not provider invoices."""
import json
from pathlib import Path
from urllib.parse import urlsplit

SOURCE='https://api-docs.deepseek.com/zh-cn/quick_start/pricing/'
FLASH={'deepseek-flash','deepseek-v4-flash','deepseek-vision-exp'}


def tariff(config, endpoint):
    inp=getattr(config,'price_input_per_million',None)
    out=getattr(config,'price_output_per_million',None)
    if inp is not None and out is not None:
        cached=getattr(config,'price_cached_input_per_million',None)
        rate=[inp if cached is None else cached,inp,out]
        return {'currency':getattr(config,'price_currency','CNY'),'low':rate,'high':rate,
                'basis':'user_configured_rates_per_million','source':None}
    if urlsplit(endpoint.url).hostname=='api.deepseek.com' and endpoint.model in FLASH:
        return {'currency':'CNY','low':[.02,1.,4.],'high':[.04,2.,8.],
                'basis':'published_off_peak_to_peak_range_per_million_not_invoice',
                'source':SOURCE,'verified_on':'2026-11-11'}
    return None


def estimate(calls, rates):
    sent=[c for c in calls if c.get('state')!='not_sent']
    low=high=0.;unknown=0;reported=0
    currencies=set()
    for call in sent:
        u=call.get('usage') or {};p=u.get('prompt_tokens');o=u.get('completion_tokens')
        reported+=u.get('total_tokens') or ((p or 0)+(o or 0))
        request_rates=call.get('request_tariff',rates)
        if request_rates is not None:
            currencies.add(request_rates['currency'])
        if request_rates is None or p is None or o is None:
            unknown+=1;continue
        hit=u.get('prompt_cache_hit_tokens')
        # Missing cache split is bounded, never silently counted as free.
        h_lo=p if hit is None else max(0,min(p,hit))
        h_hi=0 if hit is None else h_lo
        lo,hi=request_rates['low'],request_rates['high']
        low+=(h_lo*lo[0]+(p-h_lo)*lo[1]+o*lo[2])/1e6
        high+=(h_hi*hi[0]+(p-h_hi)*hi[1]+o*hi[2])/1e6
    mixed=len(currencies)>1
    return {'currency':next(iter(currencies)) if len(currencies)==1 else None, 'mixed_currency':mixed,
        'estimated_api_min':None if mixed else round(low,8),'estimated_api_max':None if mixed or unknown else round(high,8),
        'known_portion_max':None if mixed else round(high,8),'complete':unknown==0 and not mixed,'unpriced_or_unreported_calls':unknown,
        'actual_requests':len(sent),'reported_tokens':reported,'tariff':rates,
        'local_compute_cost':None,'local_compute_basis':'price_not_configured_time_reported_separately'}


def cumulative(folder, current=None):
    """One ledger per run, including interrupted attempts and preparation; no cache bills."""
    documents={}
    for path in sorted(Path(folder).glob('usage-*.json')):
        documents[path.name]=json.loads(path.read_text(encoding='utf8'))
    if current is not None:
        documents[current.get('ledger_name','usage-current.json')]=current
    low=high=0.;complete=True;requests=tokens=unknown=0;currencies=set();sources=set();mixed=False
    for doc in documents.values():
        # Older ledgers may be costed only when their provider metadata establishes a tariff.
        cost=doc.get('cost') or estimate(doc.get('calls',[]),doc.get('tariff'))
        requests+=cost['actual_requests'];tokens+=doc.get('reported_tokens',0) or 0
        unknown+=cost['unpriced_or_unreported_calls']
        if cost['currency']:currencies.add(cost['currency'])
        if cost.get('tariff',{}):
            source=cost['tariff'].get('source')
            if source:sources.add(source)
        mixed |= cost.get('mixed_currency',False)
        low+=cost['estimated_api_min'] or 0.;high+=cost['known_portion_max'] or 0.;complete &= cost['complete']
    mixed |= len(currencies)>1
    return {'currency':next(iter(currencies)) if len(currencies)==1 else None,
        'estimated_api_min':None if mixed else round(low,8),
        'estimated_api_max':round(high,8) if complete and not mixed else None,
        'complete':complete and not mixed,'actual_requests':requests,'reported_tokens':tokens,
        'unpriced_or_unreported_calls':unknown,'ledger_files':list(documents),
        'local_compute_cost':None,'basis':'cumulative_actual_requests_including_retries_not_invoice',
        'sources':sorted(sources)}


def label(cost):
    if not cost:return '费用待统计'
    lo,hi=cost.get('estimated_api_min'),cost.get('estimated_api_max')
    if lo is None:return 'API 费用存在不同币种，详见账本'
    amount=(f'{lo:.4f}–{hi:.4f}' if hi is not None else f'至少 {lo:.4f}（有未计价调用）')
    return f"累计 API 费用估算 {cost.get('currency') or ''} {amount}；{cost['actual_requests']} 次请求，{cost['reported_tokens']} token；本机计算另记耗时"
