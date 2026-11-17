"""Photographic cause-to-control regression and HTTP contract checks without a live service."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from PIL import Image
from starlette.applications import Starlette
from starlette.testclient import TestClient

import batch_display
from display_batch import web
from display_batch.ai import Vision
from display_batch.common import file_hash
from display_batch.schema import BatchConfig, TONE_PROFILES
from display_batch.structured import inspect_frame, plan_frame
from scrap_composite.photography_agent import compile_plan, run_photography, scope_plan
from scrap_composite.photography_controls import composite_changes, photoshop_changes
from scrap_composite.photography_measurements import measurements
from scrap_composite.photography_protocol import PhotoBrief, PhotoChoice, PhotoContext, PhotoMeasurements, PhotoReport
from scrap_composite.photoshop import LIMITS


def finding(factor='exposure', direction='increase', **fields):
    data = {'factor': factor, 'direction': direction, 'region': [.28, .35, .63, .67],
            'severity': 'major', 'visibility': 'visible', 'confidence': .9,
            'cue': '物品漫反射中间调偏暗', 'reference': '附近同层相近材质的亮面'}
    return {**data, **fields}


def brief(*findings, trace='visible'):
    scores = dict.fromkeys(('support', 'occlusion', 'lighting', 'edge', 'texture', 'material', 'structure'), 2)
    return PhotoBrief.model_validate_json(json.dumps({'trace': trace, 'scores': scores, 'findings': findings}))


class PhotographyTests(unittest.IsolatedAsyncioTestCase):
    async def test_incompatible_causes_and_wrong_direction_do_not_reach_planner(self):
        context = PhotoContext(images=['frame'])
        controls = photoshop_changes(dict.fromkeys(LIMITS, 0), LIMITS)
        for items, expected in (
            ([finding('warmth', 'decrease'), finding('warmth', 'increase')], 'conflict'),
            ([finding('contact_shadow', 'unknown'), finding()], 'no_supported_control'),
            ([finding('floating', 'unknown', physical_cue='visible_air_gap'), finding()], 'geometry'),
            ([finding('floating', 'unknown', physical_cue='visible_air_gap', cue='仅见接触投影偏弱'), finding()], 'geometry'),
            ([finding('exposure', 'decrease', confidence=.5)], 'confidence'),
            ([finding('haze_amount', 'increase')], 'no_supported_control'),
        ):
            with self.subTest(expected=expected):
                client = type('Client', (), {'ask': AsyncMock(return_value=brief(*items))})()
                result = await run_photography(client, context, 'a'*64, images=['frame.png'],
                                               plan=True, changes=controls)
                self.assertEqual(result.plan.reason, expected)
                self.assertEqual(result.updates, {})
                self.assertEqual(client.ask.await_count, 1)

    def test_control_uses_actual_effect_and_does_not_reset_another_axis(self):
        params = {'contrast': 'auto', 'chroma': 'auto', 'sharpness': 'auto', 'tone': 'brighter'}
        allowed = {name: ['less','more'] for name in ('contrast','chroma','sharpness')}
        allowed['tone'] = ['scene','cooler','warmer']
        values = {'auto': {'contrast': .9,'chroma': .5,'sharpness': 1.},
                  'less': {'contrast': .93,'chroma': .6,'sharpness': 1.4},
                  'more': {'contrast': .87,'chroma': .4,'sharpness': .8}}
        changes = composite_changes(params, allowed, {'effective_controls': values}, tone_profiles=TONE_PROFILES)
        offered = {(c.factor, c.direction): c.value for c in changes}
        self.assertEqual(offered['tonal_contrast','increase'], 'less')
        self.assertEqual(offered['chroma','decrease'], 'more')
        self.assertEqual(offered['sharpness','decrease'], 'less')
        self.assertNotIn(('warmth','decrease'), offered)
        self.assertNotIn(('warmth','increase'), offered)

    async def test_main_geometry_cue_is_preserved_and_score_is_reconciled_without_retry(self):
        source = brief(finding('floating', 'unknown', cue='底部和承托之间可见间隙', physical_cue='visible_air_gap'))
        source = source.model_copy(update={'trace':'none', 'scores':source.scores.model_copy(update={'support':4})})
        client = type('Client', (), {'ask': AsyncMock(return_value=source)})()
        result = await run_photography(client, PhotoContext(images=['frame']), 'a'*64, images=['frame.png'], plan=True)
        self.assertEqual(result.plan.reason, 'geometry')
        self.assertEqual(result.report.findings[0].visibility, 'visible')
        self.assertEqual(result.report.scores.support, 2)
        self.assertEqual(result.report.reconciled, ['score','trace'])
        self.assertEqual(client.ask.await_count, 1)

    async def test_editor_can_decline_exposure_when_highlights_are_at_risk(self):
        context = PhotoContext(images=['frame'], measurements=PhotoMeasurements(object_near_black_white_fraction=(0., .04)))
        client = type('Client', (), {'ask': AsyncMock(side_effect=[
            brief(finding()), PhotoChoice(option=None, reason='tradeoff')])})()
        result = await run_photography(client, context, 'a'*64, images=['frame.png'],
                                       plan=True, changes=photoshop_changes(dict.fromkeys(LIMITS, 0), LIMITS))
        self.assertEqual(result.plan.reason, 'tradeoff')
        self.assertEqual(result.updates, {})
        self.assertEqual(client.ask.await_args.args[3]['measurements']['object_near_black_white_fraction'], [0., .04])

    def test_stale_state_report_or_option_is_rejected(self):
        context = PhotoContext(images=['frame'])
        report = PhotoReport(**brief(finding()).model_dump(), image_sha256='a'*64, target='object', context=context)
        current = dict.fromkeys(LIMITS, 0)
        changes = photoshop_changes(current, LIMITS)
        scope = scope_plan(report, context, changes, parameters=current)
        plan = scope.bind(PhotoChoice(option='a1', reason='match'))
        self.assertEqual(compile_plan(plan, scope), {'brightness': 6})
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            scope.bind(PhotoChoice(option='a2', reason='match'))
        new_current = {**current, 'warmth': 4}
        with self.assertRaisesRegex(ValueError, 'current value'):
            scope_plan(report, context, changes, parameters=new_current)
        new_scope = scope_plan(report, context, photoshop_changes(new_current, LIMITS), parameters=new_current)
        with self.assertRaisesRegex(ValueError, 'stale or modified'):
            compile_plan(plan, new_scope)
        with self.assertRaisesRegex(ValueError, 'stale or modified'):
            compile_plan(plan.model_copy(update={'report_sha256': 'b'*64}), scope)
        with self.assertRaisesRegex(ValueError, 'context changed'):
            scope_plan(report, context.model_copy(update={'checks': {'support': 'fail'}}), changes)

    async def test_batch_graph_json_transport_and_cache(self):
        requests = []

        async def mock(request):
            data = json.loads(request.content)
            requests.append(data)
            content = data['messages'][1]['content']
            payload = json.loads(content[0]['text'] if isinstance(content, list) else content)
            schema = payload['response_JSON_schema']
            self.assertNotIn('title', schema)
            if payload['protocol'] == 'photo-brief-v3':
                self.assertEqual(len(content), 3)
                reply = brief(finding(observed_state='too_low', reference_role='scene',
                    reference_region=[.1,.35,.27,.67], hypothesis='局部照明偏弱，需排除材质反射率差异')).model_dump(mode='json')
            else:
                self.assertIsInstance(content, str)
                self.assertEqual(len(payload['options']), 1)
                self.assertEqual(payload['options'][0]['value'], 'brighter')
                self.assertEqual(payload['image_sha256'], file_hash(image))
                reply = {'option': 'a1', 'reason': 'match'}
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(reply)}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}})

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'frame.png'
            Image.new('RGB', (80,60), '#777777').save(image)
            provider = root/'provider.json'
            provider.write_text(json.dumps({role: {'url': 'http://127.0.0.1/chat/completions', 'model': model}
                                           for role, model in [('vision','fixture-vision'), ('planner','fixture-planner')]}))
            config = BatchConfig(backgrounds=temp, output=temp, provider_config=str(provider), review_profile='photographer')
            client = Vision(config, root/'cache', root/'usage.json')
            await client.http.aclose()
            client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock))
            comp = {'image': str(image), 'crop': str(image), 'edit_bounds': (16,18,58,44), 'checks': {
                'size': {'pixels': [80,60],'status': 'pass'}, 'physics': {'status':'unknown'}}, 'photo': {}}
            params = {'tone':'scene'}
            asyncio.current_task().set_name('photo-transport-fixture')
            try:
                for _ in range(2):
                    report, visual = await inspect_frame(client, {'name':'气缸'}, comp, file_hash(image), profile='photographer')
                    updates, steps, plan = await plan_frame(client, report, visual, comp, params,
                        {'tone':['darker','brighter','cooler','warmer']}, [], lambda _: False,
                        dust_in_shader=True, recommendation=None)
                    self.assertEqual(updates, {'tone':'brighter'})
                    self.assertEqual(steps[0]['evidence_id'], 'f1')
                    self.assertEqual(plan['image_sha256'], file_hash(image))
                self.assertEqual([r['model'] for r in requests], ['fixture-vision','fixture-planner'])
                self.assertEqual([r['max_tokens'] for r in requests], [3072,1024])
                self.assertEqual(client.usage()['api_calls'], 2)
                self.assertEqual(client.cache_hits, 2)
                self.assertEqual(client.usage()['reported_tokens'], 300)
                self.assertEqual([c['job_id'] for c in client.calls], ['photo-transport-fixture']*2)
                image.write_bytes(b'different image')
                updated, _, _ = await plan_frame(client, report, visual, comp, params,
                    {'tone':['brighter']}, [], lambda _: False, dust_in_shader=True, recommendation=None)
                self.assertIsNone(updated)
                self.assertEqual(len(requests), 2)
            finally:
                await client.close()

    def test_visible_evidence_has_local_bounds_and_comparable_reference(self):
        context = PhotoContext(images=['frame'], target_region=(.75,.75,.9,.9))
        for observation in (brief(finding()), brief(finding(reference=None, region=[.76,.76,.8,.8]))):
            report = PhotoReport.bind(observation, 'a'*64, 'object', context)
            self.assertEqual(report.findings[0].visibility, 'unassessable')
            self.assertEqual(report.trace, 'uncertain')
            self.assertIn('unlocalized', report.reconciled)
            self.assertEqual(scope_plan(report, context, []).reason, 'no_visible_defect')
        broad = PhotoReport.bind(brief(finding(region=[0,0,1,1])), 'a'*64, 'object', context)
        self.assertEqual(broad.findings[0].region, context.target_region)

    def test_neutral_luma_and_unknown_fog_are_independent(self):
        m = measurements({'local_context': {'basis':'nearby_same_support_and_depth','confidence':'estimated'}},
                         object_pixels=np.full((96,3), 64, np.uint8),
                         reference_pixels=np.full((96,3), 128, np.uint8))
        self.assertEqual(m.local_y_255, (128.,128.,128.))
        self.assertEqual(m.object_y_255, (64.,64.,64.))
        self.assertAlmostEqual(m.object_chroma_c50_c90[0], 0., places=2)
        self.assertEqual(m.reference, 'same_support')
        self.assertIsNone(m.fog)


class PhotographyEntryTests(unittest.TestCase):
    def test_profile_entry_runs_the_shared_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp)/'config.json'
            config.write_text(json.dumps({'backgrounds':temp,'output':temp}))
            with patch('sys.argv',['batch_display','run','--config',str(config),'--review-profile','photographer']), \
                 patch.object(batch_display,'Pipeline') as pipeline, contextlib.redirect_stdout(io.StringIO()):
                pipeline.return_value.run = AsyncMock(return_value={'usage_current_run':{'calls':[]}})
                batch_display.main()
                parsed = pipeline.call_args.args[0]
                self.assertEqual((parsed.workflow,parsed.review_profile), ('evidence_v1','photographer'))
                self.assertEqual(parsed.effects.fog_mode, 'local_auto')

    def test_web_reports_effective_profile_without_starting_a_real_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bg = root/'backgrounds'; bg.mkdir()
            Image.new('RGB',(32,32)).save(bg/'one.jpg')
            with patch.object(web,'ROOT',root), patch.object(web,'DATA',root/'configs'), \
                 patch.object(web,'load_catalog',return_value={'3': {'available':True,'mesh_hash':'fixture'}}), \
                 patch.object(web,'launch'), TestClient(Starlette(routes=web.routes)) as client:
                schema = client.get('/api/photography/schema').json()
                self.assertEqual(schema['framework'], 'pydantic-graph')
                res = client.post('/api/batches/test', json={'backgrounds':str(bg), 'assets':['3'],
                                                           'review_profile':'photographer'})
                self.assertEqual(res.status_code, 202, res.text)
                info = res.json()
                self.assertEqual(info['effects']['fog_mode'], 'local_auto')
                self.assertEqual(client.get(info['status_url']).json()['review_profile'], 'photographer')


if __name__ == '__main__':
    unittest.main()
