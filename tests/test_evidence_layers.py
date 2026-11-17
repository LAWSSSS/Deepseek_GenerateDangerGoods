"""Exercise layer reconstruction and the isolated Photoshop refinement loop."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, PngImagePlugin

from scrap_composite.contracts import PlanProposal, VisualObservation
from scrap_composite.layers import save_bundle
from scrap_composite.photography_protocol import PhotoBrief, PhotoChoice
from scrap_composite.photoshop import file_hash
from scrap_composite.photoshop_workflow import REQUIRED_CHECKS, refine_photoshop


class LayerWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, root):
        original=np.full((8,8,3),20,np.uint8)
        image=original.copy()
        image[3:5,3:5]=120
        alpha=np.zeros((8,8),np.float32)
        alpha[3:5,3:5]=.15
        expected=root/'composite.png'
        Image.fromarray(image).save(expected)
        mask=np.zeros(alpha.shape,bool)
        checks={name:{'status':'pass'} for name in REQUIRED_CHECKS}
        checks['physics']['status']='unknown'
        return save_bundle(root/'bundle',{'frame':Image.fromarray(original),'protected':mask},
                           expected,(0,0,8,8),original/255,alpha,alpha,mask,checks)

    def test_bright_coverage_in_display_encoding_preserves_geometry_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=self.fixture(root)
            bundle=manifest.parent
            with np.load(bundle/'alpha-evidence.npz') as proof:
                self.assertAlmostEqual(float(proof['geometric'][3,3]),.15)
            base=Image.open(bundle/'background.png').convert('RGBA')
            actual=Image.alpha_composite(base,Image.open(bundle/'object_rgba.png')).convert('RGB')
            expected=Image.open(root/'composite.png').convert('RGB')
            self.assertLessEqual(np.abs(np.asarray(actual).astype(int)-np.asarray(expected)).max(),1)

    async def test_worse_structure_keeps_verified_baseline_and_unknown_physics(self):
        class Exporter:
            def __init__(self):
                self.actions=[]

            async def export(self, request, folder):
                self.actions.append(request.actions)
                folder.mkdir()
                image=Image.open(request.expected_png).convert('RGB')
                image.putpixel((3,3),(120+int(request.actions['brightness']),120,120))
                image.save(folder/'final.png')
                # The native format and readback are covered by test_photoshop and the real fixture.
                (folder/'final.psd').write_bytes(b'fake-adapter-fixture')
                return {'outputs':{ext:{'path':str(folder/f'final.{ext}'),'sha256':file_hash(folder/f'final.{ext}')}
                                   for ext in ('png','psd')},'validation':{'status':'pass'}}

        class Client:
            async def ask(self, role, schema, prompt, payload, images=(), **kwargs):
                if role=='execution-plan':
                    return PlanProposal(status='ready',note='test',
                        operations=[{'capability':'brightness','value':8,
                                     'evidence_ids':['f1'],'expected_effect':'改善亮度'}])
                adjusted=bool(payload['photoshop_parameters']['brightness'])
                scores=dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),3)
                scores.update(lighting=4 if adjusted else 2, structure=2 if adjusted else 3)
                return VisualObservation(trace='visible',scores=scores,summary='test',
                    findings=[{'code':'lighting','severity':'minor',
                               'visibility':'visible','scope':'image','region':(.3,.3,.7,.7),
                               'confidence':.9,'direction':'increase','evidence':'物品过暗'}])

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); adapter=Exporter()
            result=await refine_photoshop(self.fixture(root),root/'result',adapter,Client())
            self.assertEqual([a['brightness'] for a in adapter.actions],[0,8])
            self.assertEqual(result['best_iteration'],0)
            self.assertEqual(result['status'],'review')
            self.assertEqual(result['reason'],'no_improvement')
            self.assertTrue(json.loads((root/'result/workflow.json').read_text(encoding='utf8'))['outputs']['png'])

    async def test_photographer_executes_own_layer_and_reviews_the_new_png(self):
        class Exporter:
            def __init__(self):
                self.values = []

            async def export(self, request, folder):
                folder.mkdir()
                value = request.actions['brightness']
                self.values.append(value)
                with Image.open(request.expected_png) as source:
                    image = source.copy()
                image.putpixel((3,3), (120 + int(value),120,120))
                image.save(folder/'final.png')
                (folder/'final.psd').write_bytes(b'fake-adapter-fixture')
                return {'outputs':{ext:{'path':str(folder/f'final.{ext}'),'sha256':file_hash(folder/f'final.{ext}')}
                                   for ext in ('png','psd')},'validation':{'status':'pass'}}

        class Client:
            def __init__(self):
                self.roles = []

            async def ask(self, role, schema, prompt, payload, images=(), **kwargs):
                self.roles.append(role)
                if schema is PhotoChoice:
                    self.assert_choice_payload(payload)
                    return PhotoChoice(option='a1', reason='match')
                improved = len(self.roles) == 3
                scores = dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),4)
                scores['lighting'] = 4 if improved else 2
                return PhotoBrief(trace='none' if improved else 'visible', scores=scores, findings=[] if improved else [
                    {'factor':'exposure','region':(.375,.375,.625,.625),'severity':'major','visibility':'visible',
                     'confidence':.9,'direction':'increase','cue':'漫反射面偏暗','reference':'附近同类材质'}])

            @staticmethod
            def assert_choice_payload(payload):
                assert len(payload['options']) == 1 and payload['options'][0]['control'] == 'brightness'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, client = Exporter(), Client()
            result = await refine_photoshop(self.fixture(root), root/'result', adapter, client, review_profile='photographer')
            self.assertEqual(adapter.values, [0.,6.])
            self.assertEqual(client.roles, ['photo-vision','photo-execution','photo-vision'])
            self.assertEqual(result['best_iteration'], 1)
            self.assertEqual(result['status'], 'review')
            first, second = [trial['photography_diagnosis']['image_sha256'] for trial in result['trials']]
            self.assertNotEqual(first, second)

    async def test_png_metadata_only_change_stops_before_review_with_or_without_receipt(self):
        for with_receipt in (False, True):
            with self.subTest(with_receipt=with_receipt), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                class Exporter:
                    calls=0
                    async def export(self,request,folder):
                        self.calls+=1;folder.mkdir()
                        with Image.open(request.expected_png) as source:image=source.convert('RGB')
                        metadata=PngImagePlugin.PngInfo()
                        metadata.add_text('export_attempt',str(self.calls))
                        image.save(folder/'final.png',pnginfo=metadata)
                        (folder/'final.psd').write_bytes(b'fixture')
                        result={'outputs':{ext:{'path':str(folder/f'final.{ext}'),
                                    'sha256':file_hash(folder/f'final.{ext}')} for ext in ('png','psd')},
                                'validation':{'status':'pass'}}
                        if with_receipt and self.calls>1:
                            result['receipt']={'changed_pixels_vs_reviewed_image':0}
                        return result
                class Client:
                    visual_calls=0
                    plan_calls=0
                    async def ask(self,role,schema,prompt,payload,images=(),**kwargs):
                        if role=='execution-plan':
                            self.plan_calls+=1
                            return PlanProposal(status='ready',note='fixture',operations=[
                                {'capability':'brightness','value':8,'evidence_ids':['f1'],'expected_effect':'提亮中间调'}])
                        self.visual_calls+=1
                        scores=dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),3)
                        scores['lighting']=2
                        return VisualObservation(trace='visible',scores=scores,summary='fixture',findings=[
                            {'code':'lighting','severity':'minor','visibility':'visible','scope':'image',
                             'region':(.3,.3,.7,.7),'confidence':.9,'direction':'increase','evidence':'物品偏暗'}])
                adapter,client=Exporter(),Client()
                result=await refine_photoshop(self.fixture(root),root/'result',adapter,client)
                self.assertEqual(adapter.calls,2)
                self.assertEqual(client.visual_calls,1)
                self.assertEqual(client.plan_calls,1)
                self.assertEqual(result['reason'],'no_effective_pixel_change')
                self.assertEqual(result['best_iteration'],0)
                self.assertEqual(result['status'],'review')
                a,b=result['trials']
                self.assertNotEqual(a['export']['outputs']['png']['sha256'],b['export']['outputs']['png']['sha256'])
                self.assertEqual(a['pixel_sha256'],b['pixel_sha256'])
                self.assertEqual(result['outputs']['png']['sha256'],a['export']['outputs']['png']['sha256'])
