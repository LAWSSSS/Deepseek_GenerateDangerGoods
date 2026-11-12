import asyncio
import json
import time
import io
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, model_validator
from PIL import Image, ImageOps
from run import Endpoint, ModelClient, token_summary
from .common import HERE, atomic_json, configure_credential, file_hash, signature
from .costs import tariff, estimate


class VisionEndpoint(Endpoint):
    max_output_tokens: int = Field(default=2048, ge=128, le=4096)
    response_protocol: Literal['tools','qwen-json'] = 'tools'

    @model_validator(mode='after')
    def batch_endpoint(self):
        host=urlsplit(self.url).hostname or ''
        if host.startswith('token-plan.') or host=='coding.dashscope.aliyuncs.com':
            raise ValueError('套餐端点仅供交互式工具使用。批处理请使用独立的视觉 API。')
        return self


class ProviderUnavailable(RuntimeError):
    pass


def semantic_payload(value):
    """Operational cache metadata must not turn an identical review into a paid request."""
    if isinstance(value,dict):
        return {k:semantic_payload(v) for k,v in value.items() if k not in ('cache_hit','render_cache_hit')}
    if isinstance(value,(list,tuple)):return [semantic_payload(v) for v in value]
    return value


class Vision:
    def __init__(self, config, cache, usage_path):
        raw = json.loads(Path(config.provider_config or HERE/"batch-config.example.json").read_text(encoding="utf8"))
        endpoint = dict(raw.get("vision") or raw)
        endpoint.update(max_output_tokens=3072, timeout_seconds=120)
        self.endpoint = VisionEndpoint.model_validate(endpoint)
        self.planner = self.endpoint
        if getattr(config, 'workflow', 'legacy') == 'evidence_v1' and raw.get('planner'):
            self.planner = VisionEndpoint.model_validate({
                **raw['planner'], 'max_output_tokens': 1024, 'timeout_seconds': 120})
        self.offline = config.offline
        self.config, self.cache, self.usage_path = config, Path(cache), Path(usage_path)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.semaphore = asyncio.Semaphore(config.ai_concurrency)
        self.configured = configure_credential(self.endpoint.key_env)
        for service in (self.endpoint, self.planner):
            configure_credential(service.key_env)
            if os.environ.get(service.key_env or '', '').startswith('sk-sp-'):
                raise ValueError('百炼批处理需要独立 API 凭据，不能使用套餐密钥')
        self.http = httpx.AsyncClient()
        self.cache_hits = 0

    async def close(self):
        atomic_json(self.usage_path,self.usage())
        await self.http.aclose()

    def usage(self):
        result = token_summary(self.calls)
        result.update(cache_hits=self.cache_hits, calls=self.calls)
        rates=tariff(self.config,self.endpoint)
        result.update(ledger_name=self.usage_path.name,tariff=rates,cost=estimate(self.calls,rates))
        cfg = self.config
        if cfg.price_input_per_million is not None and cfg.price_output_per_million is not None:
            cost=result['cost']
            exact=cost['complete'] and cost['estimated_api_min']==cost['estimated_api_max']
            result['reported_cost']=cost['estimated_api_min'] if exact else None
            result['cost_complete']=exact
        return result

    async def ask(self, role, schema, prompt, payload, images=(), *, cache=True,
                  image_limits=None,max_output_tokens=None,plain_json=False,planning=False,validate=None):
        previews=[]
        service=self.planner if planning else self.endpoint
        endpoint=service if max_output_tokens is None else service.model_copy(
            update={'max_output_tokens':max_output_tokens})
        for i,image in enumerate(images):
            image=Path(image)
            limit=image_limits[i] if image_limits is not None else (1280,1024)
            encoded_key=signature('jpeg-v1',file_hash(image)) if limit==(1280,1024) else signature(
                'jpeg-v2',limit,file_hash(image))
            dest=self.cache/('preview-'+encoded_key+'.jpg')
            if not dest.is_file():
                with Image.open(image) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail(limit)
                    for quality in (85,75,60,45):
                        data=io.BytesIO();im.save(data,format='JPEG',quality=quality)
                        if len(data.getvalue())<=384*1024:break
                    else:
                        im.thumbnail((800,800));data=io.BytesIO();im.save(data,format='JPEG',quality=65)
                    dest.write_bytes(data.getvalue())
            previews.append(dest)
        images=tuple(previews)
        qwen=endpoint.response_protocol=='qwen-json'
        use_json=qwen or plain_json
        endpoint_key=endpoint.model_dump(exclude={'response_protocol'} if not qwen else None)
        key = signature("vision-json-v1" if use_json else "vision-v1", endpoint_key, schema.model_json_schema(),
                        role, prompt, semantic_payload(payload), [file_hash(p) for p in images])
        path = self.cache/f"{key}.json"
        if cache and path.is_file():
            try:
                answer=schema.model_validate_json(path.read_text(encoding="utf8"))
                if validate is not None:validate(answer)
            except ValueError:
                pass
            else:
                self.cache_hits += 1
                return answer
        if self.offline:
            raise ProviderUnavailable("离线模式缺少模型缓存："+role)
        if (not configure_credential(endpoint.key_env) if planning else not self.configured):
            raise ProviderUnavailable("未配置模型服务凭据："+str(endpoint.key_env))
        async with self.semaphore:
            last_error = None
            retry_payload = payload
            for attempt in range(2):
                report = {"calls": []}
                start = time.monotonic()
                try:
                    mode = " Return one JSON object matching the supplied schema." if use_json else " Return only the requested JSON tool arguments."
                    answer = await ModelClient(self.http, report).ask(endpoint, role, schema,
                        prompt + mode + " Coordinates are normalized [0,1]."
                        " Write evidence in concise Chinese, at most 60 Chinese characters. Respect all schema string limits.",
                        retry_payload, images, tool=None if use_json else "inspect_display",
                        detail=None if qwen else "high",qwen_json=qwen,plain_json=use_json)
                    if validate is not None:validate(answer)
                    if cache:
                        atomic_json(path, answer.model_dump())
                    return answer
                except (ValueError, RuntimeError, httpx.HTTPError) as error:
                    last_error = error
                    code = (report["calls"][-1].get("http_status") if report["calls"] else None)
                    if code in (400, 401, 403, 404):
                        raise ProviderUnavailable(f"视觉服务不可用：HTTP {code}") from None
                    if attempt == 0:
                        retry_payload={**payload,'format_correction':(report['calls'][-1].get('schema_errors')
                            if report['calls'] else []) or 'Previous reply was invalid. Use exactly the JSON schema.'}
                        await asyncio.sleep(1)
                finally:
                    for call in report["calls"]:
                        call.update(request_signature=key, attempt=attempt,
                                    job_id=asyncio.current_task().get_name(),
                                    seconds=round(time.monotonic()-start, 3))
                        if planning:
                            call['request_tariff'] = tariff(self.config, endpoint)
                        self.calls.append(call)
                    atomic_json(self.usage_path, self.usage())
            transport_failed=bool(report['calls'] and report['calls'][-1].get('transport_error'))
            if transport_failed or isinstance(last_error,httpx.TransportError) or code in (429,500,502,503,504):
                raise ProviderUnavailable(f'视觉服务暂不可用：{code or type(last_error).__name__}，已用完一次重试') from None
            raise RuntimeError(f"{role} 两次请求未得到有效结果：{type(last_error).__name__}") from None
