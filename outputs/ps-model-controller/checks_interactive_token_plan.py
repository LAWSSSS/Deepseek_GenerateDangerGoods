"""Mock-only interactive Token Plan safety, ledger and cache regressions."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

import interactive_token_plan as tool


DUMMY_KEY = 'sk-sp-unit-test-no-real-credential-00000000'
ANSWER = {'image_visible': True, 'summary': '表面过于平滑，接触关系无法确认。',
          'findings': [{'image': 1, 'category': 'appearance', 'severity': 'major',
                        'evidence': '物品表面的纹理强度低于邻近钢料。'}],
          'next_steps': ['先比较原渲染与最终像素的清晰度，不增加薄雾。']}


def response(content=None):
    return {'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps(ANSWER if content is None else content, ensure_ascii=False)}}],
        'usage': {'prompt_tokens': 120, 'completion_tokens': 40, 'total_tokens': 160}}


class InteractiveTokenPlanTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='scrap-token-plan-test-')
        self.folder = Path(self.temporary.name)
        self.image = self.folder/'fixture.png'
        Image.new('RGB', (32, 24), (82, 91, 100)).save(self.image)
        self.output = self.folder/'diagnosis'
        self.credential = patch.dict(os.environ, {tool.KEY_ENV: DUMMY_KEY})
        self.credential.start()
        self.configure = patch.object(tool, 'configure', return_value=True)
        self.configure.start()
        self.sleep = patch.object(tool.asyncio, 'sleep', new_callable=AsyncMock)
        self.sleep.start()

    def tearDown(self):
        self.sleep.stop()
        self.configure.stop()
        self.credential.stop()
        self.temporary.cleanup()

    async def invoke(self, handler, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await tool.diagnose([self.image], '只诊断此图的材质和受光。',
                                       self.output, http=client, **kwargs)

    def ledger(self):
        return json.loads((self.output/'usage.json').read_text(encoding='utf8'))

    async def test_one_explicit_image_json_success_and_unknown_credit_cost(self):
        sent = []
        def handler(request):
            sent.append(request)
            body = json.loads(request.content)
            self.assertEqual(str(request.url), tool.URL)
            self.assertEqual(body['model'], tool.MODEL)
            self.assertEqual(body['response_format'], {'type': 'json_object'})
            self.assertEqual(len(body['messages'][1]['content']), 2)
            self.assertTrue(body['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/'))
            return httpx.Response(200, json=response())
        result = await self.invoke(handler)
        self.assertEqual(len(sent), 1)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['new_api_requests'], 1)
        self.assertEqual(result['new_reported_tokens'], 160)
        self.assertFalse(result['batch_resumed'])
        self.assertEqual(result['new_render_count'], 0)
        self.assertEqual(result['acceptance_status'], 'interactive_diagnostic_only')
        self.assertIsNone(result['credits'])
        self.assertIsNone(result['currency_cost'])
        self.assertIsNone(self.ledger()['credits'])
        self.assertIsNone(self.ledger()['currency_cost'])
        for path in self.output.glob('*.json'):
            self.assertNotIn(DUMMY_KEY, path.read_text(encoding='utf8'))

    async def test_401_stops_without_retry_and_does_not_log_server_body(self):
        sent = []
        def handler(request):
            sent.append(request)
            return httpx.Response(401, json={'error': {'message': DUMMY_KEY+' private server text'}})
        result = await self.invoke(handler)
        self.assertEqual(len(sent), 1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['http_status'], 401)
        self.assertEqual(self.ledger()['api_calls'], 1)
        self.assertEqual(self.ledger()['unreported_calls'], 1)
        self.assertIsNone(self.ledger()['total_tokens'])
        for path in self.output.glob('*.json'):
            self.assertNotIn(DUMMY_KEY, path.read_text(encoding='utf8'))
            self.assertNotIn('private server text', path.read_text(encoding='utf8'))

    async def test_photographer_uses_shared_brief_and_binds_the_actual_frame(self):
        def handler(request):
            data = json.loads(request.content)
            self.assertEqual(data['max_completion_tokens'], 1536)
            self.assertEqual(data['enable_thinking'], False)
            return httpx.Response(200, json=response({'trace':'none', 'findings':[], 'scores':dict.fromkeys(
                ('support','occlusion','lighting','edge','texture','material','structure'), 4)}))

        result = await self.invoke(handler, report_format='photographer')
        self.assertEqual(result['new_api_requests'], 1)
        self.assertEqual(result['diagnosis']['image_sha256'], tool.file_hash(self.image))
        self.assertEqual(result['diagnosis']['version'], 'photo-brief-v2')
        self.assertEqual(result['diagnosis']['context']['images'], ['frame'])

    async def test_503_gets_exactly_one_retry_and_both_attempts_are_accounted(self):
        sent = []
        def handler(request):
            sent.append(request)
            return httpx.Response(503, json={'error': 'unavailable'}) if len(sent)==1 else httpx.Response(200, json=response())
        result = await self.invoke(handler)
        self.assertEqual(len(sent), 2)
        self.assertEqual(result['status'], 'completed')
        usage = self.ledger()
        self.assertEqual(usage['api_calls'], 2)
        self.assertEqual(usage['reported_tokens'], 160)
        self.assertEqual(usage['unreported_calls'], 1)
        self.assertEqual([c['attempt'] for c in usage['calls']], [0, 1])
        self.assertIsNone(usage['total_tokens'])

    async def test_repeated_503_stops_at_two_requests_and_failure_is_not_retried_on_resume(self):
        sent = []
        def handler(request):
            sent.append(request)
            return httpx.Response(503, json={'error': 'unavailable'})
        result = await self.invoke(handler)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(sent), 2)
        with self.assertRaisesRegex(ValueError, 'Previous request failed'):
            await self.invoke(handler)
        self.assertEqual(len(sent), 2)

    async def test_exact_cache_does_not_read_credential_or_send_request(self):
        await self.invoke(lambda request: httpx.Response(200, json=response()))
        before = (self.output/'usage.json').read_bytes()
        with patch.object(tool, 'configure', side_effect=AssertionError('cache must not read credentials')):
            result = await self.invoke(lambda request: self.fail('cache must not send requests'))
        self.assertTrue(result['cache_hit'])
        self.assertEqual(result['new_api_requests'], 0)
        self.assertEqual(result['new_reported_tokens'], 0)
        self.assertEqual((self.output/'usage.json').read_bytes(), before)

    async def test_changed_image_cannot_reuse_existing_diagnosis(self):
        await self.invoke(lambda request: httpx.Response(200, json=response()))
        Image.new('RGB', (32, 24), (101, 91, 82)).save(self.image)
        with self.assertRaisesRegex(ValueError, 'changed inputs'):
            await self.invoke(lambda request: self.fail('changed output directory must not send requests'))

    async def test_invalid_image_reference_retries_once_without_accepting_bad_answer(self):
        bad = {**ANSWER, 'findings': [{**ANSWER['findings'][0], 'image': 2}]}
        sent = []
        def handler(request):
            sent.append(request)
            return httpx.Response(200, json=response(bad))
        result = await self.invoke(handler)
        self.assertEqual(len(sent), 2)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['diagnosis'])
        self.assertEqual(self.ledger()['reported_tokens'], 320)

    async def test_interrupted_ledger_requires_explicit_new_attempt(self):
        self.output.mkdir()
        (self.output/'usage.json').write_text(json.dumps({'calls': [{'http_status': 200}], 'api_calls': 1}), encoding='utf8')
        before = (self.output/'usage.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'incomplete|Interrupted|interrupted|ledger'):
            await self.invoke(lambda request: self.fail('interrupted charged request must not be repeated'))
        self.assertEqual((self.output/'usage.json').read_bytes(), before)

    async def test_explicit_image_bound_and_official_endpoint(self):
        with self.assertRaises(ValueError):
            await tool.diagnose([self.image]*5, '诊断', self.output)
        with self.assertRaises(ValueError):
            tool.InteractiveEndpoint(url='https://example.com/collect', model=tool.MODEL, key_env=tool.KEY_ENV)

    @unittest.skipUnless(os.name=='nt', 'Windows DPAPI only')
    async def test_dpapi_round_trip_uses_only_temporary_dummy_credential(self):
        target = self.folder/'fixture.dpapi'
        with patch.object(tool, 'credential_path', return_value=target):
            self.assertEqual(tool.store_credential(DUMMY_KEY), target)
            encrypted = target.read_bytes()
            self.assertNotIn(DUMMY_KEY.encode(), encrypted)
            self.assertEqual(tool.protect(encrypted, decrypt=True).decode(), DUMMY_KEY)
            self.assertEqual(tool.store_credential(DUMMY_KEY), target)
            self.assertEqual(target.read_bytes(), encrypted)
            with self.assertRaisesRegex(ValueError, 'different'):
                tool.store_credential(DUMMY_KEY+'-other')


if __name__ == '__main__':
    unittest.main()
