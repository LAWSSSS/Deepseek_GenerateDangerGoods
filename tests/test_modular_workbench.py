"""Regression coverage for the new publication and Photoshop routing boundaries."""
import asyncio
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image

from scrap_composite.contracts import PlanProposal, VisualObservation
from scrap_composite.data.files import file_hash
from scrap_composite.data.registry import Registry, published_catalog
from scrap_composite.data.snapshot import mismatched_inputs, snapshot
from scrap_composite.generation.execution import PhotoshopUnavailable, needs_photoshop, photoshop_plan
from scrap_composite.generation.pipeline import Pipeline
from scrap_composite.generation.schema import BatchConfig, EffectsReview
from scrap_composite.layers import save_bundle
from scrap_composite.modeling.service import ModelingService, file_record
from scrap_composite.photoshop_workflow import REQUIRED_CHECKS

SCORES = dict(support=4, occlusion=4, lighting=4, edge=4, texture=4, material=4, structure=4)


def observation(path, code=None):
    findings = []
    if code:
        findings = [dict(code=code, severity='major', visibility='visible', scope='image',
                         region=(.25,.25,.7,.7), confidence=.9, direction='decrease', evidence='局部观察',
                         physical_cue='visible_air_gap' if code=='floating' else None)]
    return VisualObservation(scores={**SCORES, 'lighting':2 if code else 4},
                             trace='visible' if code else 'none', findings=findings).for_frame(file_hash(path), 'object')


def composition(folder):
    background = np.full((24, 32, 3), 48, dtype=np.uint8)
    alpha = np.zeros(background.shape[:2], dtype=np.float32)
    alpha[5:20, 8:24] = 1
    alpha[10:14, 14:18] = 0
    image = background.copy()
    image[alpha > 0] = (170, 120, 90)
    path = folder / 'composite.png'
    Image.fromarray(image).save(path)
    checks = {name:{'status':'pass'} for name in REQUIRED_CHECKS}
    mask = np.zeros(alpha.shape, dtype=bool)
    proof = save_bundle(folder/'layers', {'frame': Image.fromarray(background), 'protected': mask},
                        path, (0, 0, 32, 24), background/255, alpha, alpha, mask, checks)
    return {'image':str(path), 'composition_manifest':str(proof), 'checks':checks, 'photo':{},
            'crop':str(path), 'preview':str(path), 'edit_bounds':(0,0,32,24), 'measured_major_px':16}


class PublicationChecks(unittest.TestCase):
    def ready(self, registry, record, stamp):
        folder = registry.directory(record['id'])
        folder.mkdir(parents=True)
        artifacts = {}
        for kind in ('render','preview','board','verification'):
            path = folder / kind
            path.write_bytes(f'{stamp}-{kind}'.encode())
            artifacts[kind] = file_record(path)
        record.update(state='ready', artifacts=artifacts, packed_subject='PublishedSubject')
        registry.save(record)
        return record

    def test_publication_and_batch_snapshot_keep_the_resolved_version(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = Registry(root)
            first = registry.create('测试件', {})
            with self.assertRaises(ValueError):
                registry.publish('40', first['id'])
            self.ready(registry, first, 'one')
            registry.publish('40', first['id'])
            background = root/'background.png'
            Image.new('RGB',(10,10)).save(background)
            jobs = [{'asset_id':'40','background':str(background),'background_hash':file_hash(background)}]
            config = BatchConfig(backgrounds=str(root), output=str(root/'results'))
            pinned = snapshot(config, jobs, published_catalog(root))
            replacement = self.ready(registry, registry.create(None, {}, asset=first['asset']), 'two')
            self.assertEqual(published_catalog(root)['40']['version'], first['id'])
            registry.publish('40', replacement['id'])
            self.assertEqual(pinned['catalog']['40']['version'], first['id'])
            self.assertEqual(published_catalog(root)['40']['version'], replacement['id'])
            self.assertIsNone(mismatched_inputs(pinned))
            Path(first['artifacts']['render']['path']).write_bytes(b'changed')
            self.assertIn('40', mismatched_inputs(pinned))

    def test_shared_service_keeps_uploaded_photo_and_directional_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'photo.png'
            Image.new('RGBA',(40,40)).save(source)
            service = ModelingService(root/'library')
            record = service.submit(name='新样品', source=source, views={'front':source})
            self.assertEqual(record['request']['source'], record['request']['views']['front'])
            self.assertEqual(record['request']['source']['sha256'], file_hash(source))
            self.assertFalse(published_catalog(service.registry.root))


class RoutingChecks(unittest.IsolatedAsyncioTestCase):
    async def test_registered_ps_plan_requires_visible_supported_issue(self):
        with TemporaryDirectory() as temporary:
            comp = composition(Path(temporary))
            finding = observation(comp['image'], 'color')
            values = {name:0 for name in ('brightness','black_point','highlights','warmth','tint','saturation')}
            self.assertFalse(needs_photoshop('off', observation=finding, remaining=2))
            self.assertFalse(needs_photoshop('on_issue', observation=finding, remaining=0))
            self.assertFalse(needs_photoshop('on_issue', observation=observation(comp['image'],'floating'), remaining=2))
            self.assertTrue(needs_photoshop('always'))
            plan = PlanProposal(status='ready', note='微调饱和度', operations=[
                dict(capability='saturation', value=-6, evidence_ids=['f1'], expected_effect='局部颜色接近背景')])
            client = SimpleNamespace(ask=AsyncMock(return_value=plan))
            updated, _ = await photoshop_plan(client, finding, comp, values, set())
            self.assertEqual(updated['saturation'], -6)
            plan.operations[0].value = 6
            updated, result = await photoshop_plan(client, finding, comp, values, set())
            self.assertIsNone(updated)
            self.assertEqual(result['status'],'invalid')

    async def run_job(self, root, mode, *, defect=None, rounds=2, unavailable=False):
        pipe = Pipeline.__new__(Pipeline)
        pipe.config = BatchConfig(backgrounds=str(root),output=str(root),photoshop_mode=mode,max_iterations=rounds)
        pipe.out = pipe.cache = root
        pipe.store = SimpleNamespace(checkpoint=lambda *args: None)
        pipe.cpu = asyncio.Semaphore(1)
        pipe.vision = SimpleNamespace(calls=[], usage=lambda:{})
        asset = {'signature':'asset','poses':[{}],'physics_signature':'physical','semantic':{}, 'name':'测试'}
        pipe.prepare_asset = AsyncMock(return_value=asset)
        pipe.scenes = SimpleNamespace(prepare=AsyncMock(return_value={'key':'scene'}))
        pipe.evidence_scene = pipe.scenes.prepare
        pipe.choose_placements = AsyncMock(return_value=([{'id':'A','target_px':16,'score':1}], []))
        pipe.render = AsyncMock(return_value=(root,{},None,None,False))
        comp = composition(root)
        timeline = []

        def compose(*args, **kwargs):
            path = Path(args[-1])
            shutil.copyfile(comp['image'], path)
            return comp

        async def observe(_, __, frame, image_hash, *args, **kwargs):
            seen = defect if not timeline or timeline == ['ps'] else None
            report = observation(frame['image'], seen)
            timeline.append('review:' + ('ps' if 'photoshop' in frame else 'png'))
            review = EffectsReview(scores=report.scores.model_dump(), trace=report.trace, actions=[],
                defects=[dict(code=seen, severity='major', region=[.3,.3,.6,.6],
                              physical_cue='visible_air_gap' if seen=='floating' else None,
                              evidence='底部与钢板之间可见空气间隙' if seen=='floating' else '物体过于鲜艳')] if seen else [], evidence='fixture')
            return report, review

        async def execute(_, folder, source, result, iteration, *, actions=None, **kwargs):
            timeline.append('ps')
            if unavailable:
                raise PhotoshopUnavailable('fixture')
            path = root/f'ps-{iteration}.png'
            with Image.open(source['image']) as image:
                result_image = image.copy()
                if actions:
                    result_image.putpixel((9,6),(140,100,80))
                result_image.save(path)
            psd = root/f'ps-{iteration}.psd'
            psd.write_bytes(b'fixture')
            state = {'parameters': actions or {}}
            return {**source,'image':str(path),'photoshop':{'outputs':{
                    'psd':{'path':str(psd),'sha256':file_hash(psd)}},'readback':{}},'execution_state':state}

        job = {'id':'fixture','checkpoint':{},'payload':{'asset_id':'40','background':'fixture','seed':1}}
        with patch('scrap_composite.generation.pipeline.compose', side_effect=compose), \
                patch('scrap_composite.generation.pipeline.review_frame', side_effect=observe), \
                patch.object(pipe, '_ps_frame', side_effect=execute), \
                patch('scrap_composite.generation.pipeline.sample_environment',return_value=({},[])), \
                patch('scrap_composite.generation.structured.initial_recipe',side_effect=lambda *args:(args[-1],{})), \
                patch('scrap_composite.generation.pipeline.photoshop_plan', new=AsyncMock(
                    return_value=({'brightness':-8},{'status':'ready'}))), \
                patch('scrap_composite.generation.pipeline.next_frame', new=AsyncMock(return_value=(None,[],None))):
            result = await pipe.job(job)
        return result, timeline

    async def test_one_loop_reviews_only_the_actual_output_for_all_three_modes(self):
        cases = [('off',None,2,['review:png']), ('always',None,0,['ps','review:ps']),
                 ('on_issue',None,2,['review:png']), ('on_issue','color',1,['review:png','ps','review:ps']),
                 ('on_issue','color',0,['review:png']), ('on_issue','floating',2,['review:png'])]
        for mode, defect, rounds, expected in cases:
            with self.subTest(mode=mode,defect=defect,rounds=rounds), TemporaryDirectory() as temporary:
                result, actual = await self.run_job(Path(temporary),mode,defect=defect,rounds=rounds)
                self.assertEqual(actual,expected,result)
                self.assertEqual(bool(result.get('psd')),'review:ps' in actual)
                if defect == 'floating':
                    self.assertEqual(result['status'],'failed',result)
                    self.assertNotIn('image',result)

    async def test_unavailable_ps_is_retriable(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaises(PhotoshopUnavailable):
                await self.run_job(Path(temporary),'always',unavailable=True)
