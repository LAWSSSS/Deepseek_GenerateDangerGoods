"""Exercise layer reconstruction and the isolated Photoshop refinement loop."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scrap_composite.contracts import ExecutionPlan, VisualReport
from scrap_composite.layers import save_bundle
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
                    return ExecutionPlan(version='execution-plan-v1',status='ready',note='test',
                        image_sha256=payload['observation']['image_sha256'],
                        operations=[{'capability':'brightness','target':'object','value':8,
                                     'evidence_ids':['a'],'expected_effect':'改善亮度'}])
                adjusted=bool(payload['photoshop_parameters']['brightness'])
                scores=dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),3)
                scores.update(lighting=4 if adjusted else 2, structure=2 if adjusted else 3)
                return VisualReport(version='visual-evidence-v1',image_sha256=payload['image_sha256'],
                    trace='visible',scores=scores,summary='test',
                    findings=[{'id':'a','target':'object','code':'lighting','severity':'minor',
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
