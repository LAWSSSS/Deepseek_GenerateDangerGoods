import asyncio
import json
import time
import io
from pathlib import Path

import httpx
from pydantic import Field
from PIL import Image, ImageOps
from run import Endpoint, ModelClient, token_summary
from .common import HERE, atomic_json, configure_credential, file_hash, signature


class VisionEndpoint(Endpoint):
    max_output_tokens: int = Field(default=2048, ge=128, le=4096)


class ProviderUnavailable(RuntimeError):
    pass


class Vision:
    def __init__(self, config, cache, usage_path):
        raw = json.loads(Path(config.provider_config or HERE/"batch-config.example.json").read_text(encoding="utf8"))
        endpoint = dict(raw.get("vision") or raw)
        endpoint.update(max_output_tokens=3072, timeout_seconds=120)
        self.endpoint = VisionEndpoint.model_validate(endpoint)
        self.offline = config.offline
        self.config, self.cache, self.usage_path = config, Path(cache), Path(usage_path)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.semaphore = asyncio.Semaphore(config.ai_concurrency)
        self.http = httpx.AsyncClient()
        self.configured = configure_credential(self.endpoint.key_env)
        self.cache_hits = 0

    async def close(self):
        await self.http.aclose()

    def usage(self):
        result = token_summary(self.calls)
        result.update(cache_hits=self.cache_hits, calls=self.calls)
        cfg = self.config
        if cfg.price_input_per_million is not None and cfg.price_output_per_million is not None:
            result["reported_cost"] = sum((r.get("usage", {}).get("prompt_tokens", 0)*cfg.price_input_per_million
                + r.get("usage", {}).get("completion_tokens", 0)*cfg.price_output_per_million)/1e6 for r in self.calls)
            result["cost_complete"] = not result["unreported_calls"]
        return result

    async def ask(self, role, schema, prompt, payload, images=(), *, cache=True):
        previews=[]
        for image in images:
            image=Path(image)
            dest=self.cache/('preview-'+signature('jpeg-v1',file_hash(image))+'.jpg')
            if not dest.is_file():
                with Image.open(image) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1280,1024))
                    for quality in (85,75,60,45):
                        data=io.BytesIO();im.save(data,format='JPEG',quality=quality)
                        if len(data.getvalue())<=384*1024:break
                    else:
                        im.thumbnail((800,800));data=io.BytesIO();im.save(data,format='JPEG',quality=65)
                    dest.write_bytes(data.getvalue())
            previews.append(dest)
        images=tuple(previews)
        key = signature("vision-v1", self.endpoint.model_dump(), schema.model_json_schema(),
                        role, prompt, payload, [file_hash(p) for p in images])
        path = self.cache/f"{key}.json"
        if cache and path.is_file():
            self.cache_hits += 1
            return schema.model_validate_json(path.read_text(encoding="utf8"))
        if self.offline:
            raise ProviderUnavailable("离线模式缺少模型缓存："+role)
        if not self.configured:
            raise ProviderUnavailable("未配置视觉服务凭据："+str(self.endpoint.key_env))
        async with self.semaphore:
            last_error = None
            retry_payload = payload
            for attempt in range(2):
                report = {"calls": []}
                start = time.monotonic()
                try:
                    answer = await ModelClient(self.http, report).ask(self.endpoint, role, schema,
                        prompt + " Return only the requested JSON tool arguments. Coordinates are normalized [0,1]."
                        " Write evidence in concise Chinese, at most 60 Chinese characters. Respect all schema string limits.",
                        retry_payload, images, tool="inspect_display", detail="high")
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
                        self.calls.append(call)
                    atomic_json(self.usage_path, self.usage())
            raise RuntimeError(f"{role} 两次请求未得到有效结果：{type(last_error).__name__}") from None
