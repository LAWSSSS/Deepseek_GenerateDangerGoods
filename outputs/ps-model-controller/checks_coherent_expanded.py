"""No-network checks for frozen new-scene selection and the 36-case denominator."""
from pathlib import Path
import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash
from validate_coherent_expanded import FEATURES, EXCLUDED_GROUPS, freeze, report, select_backgrounds, total_cost, run


class ExpandedTests(unittest.TestCase):
    def test_pause_blocks_generation_before_starting_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);atomic_json(out/'paused.json',{'requested_by':'user'})
            with patch('display_batch.pipeline.Pipeline') as pipeline:
                with self.assertRaisesRegex(ValueError,'generation_paused_by_user'):
                    asyncio.run(run(out/'source',out))
                pipeline.assert_not_called()

    def test_fresh_runner_process_initializes_exr_before_any_codec(self):
        # Parent tests may already have enabled OpenEXR. The production runner
        # must work in a fresh shell with that environment variable absent.
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'linear.exr'
            script=f'''
import sys
assert 'cv2' not in sys.modules
import validate_coherent_expanded
import cv2,numpy as np
# Reproduce the freeze stage's earlier JPEG/PNG codec initialization, then EXR.
ok,buffer=cv2.imencode('.png',np.zeros((8,8,3),np.uint8));assert ok
assert cv2.imdecode(buffer,cv2.IMREAD_UNCHANGED).shape==(8,8,3)
image=np.broadcast_to(np.array([.125,.5,2.],np.float32),(12,15,3)).copy()
assert cv2.imwrite({str(path)!r},image)
actual=cv2.imread({str(path)!r},cv2.IMREAD_UNCHANGED)
assert actual is not None and actual.dtype==np.float32
np.testing.assert_allclose(actual,image,rtol=0,atol=1e-7)
print('FRESH_PROCESS_EXR_ROUNDTRIP_OK')
'''
            environment=os.environ.copy();environment.pop('OPENCV_IO_ENABLE_OPENEXR',None)
            result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parent,
                                  capture_output=True,text=True,encoding='utf8',timeout=60,env=environment,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('FRESH_PROCESS_EXR_ROUNDTRIP_OK',result.stdout)

    def test_selection_is_order_independent_and_seeded_on_content(self):
        rng=np.random.default_rng(19)
        rows=[{'sha256':str(i),'features':dict(zip(FEATURES,rng.normal(size=len(FEATURES))))} for i in range(20)]
        before,method=select_backgrounds(rows,seed=7)
        after,again=select_backgrounds(list(reversed(rows)),seed=7)
        self.assertEqual(before,after);self.assertEqual(method,again)
        self.assertEqual(len({r['sha256'] for r in before}),3)
        self.assertFalse(method['selection_uses_generated_results'])

    def test_cost_includes_interrupted_batch_ledgers_without_run_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,tokens,extra in [('batch-interrupted',100,True),('batch-current',200,False)]:
                folder=root/name;folder.mkdir()
                calls=[{'state':'success','usage':{'prompt_tokens':tokens-20,'completion_tokens':20,'total_tokens':tokens}}]
                if extra:calls.append({'state':'network_error','usage':None})
                atomic_json(folder/'usage-same-name.json',{'calls':calls,'reported_tokens':tokens})
            atomic_json(root/'batch-config.json',{'not':'a batch directory'})
            cost=total_cost(root)
            self.assertEqual(cost['actual_requests'],3)
            self.assertEqual(cost['reported_tokens'],300)
            self.assertEqual(len(cost['batches']),2)
            self.assertFalse(cost['complete'])
            self.assertIsNone(cost['estimated_api_max'])

    def test_freeze_replay_counts_failures_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir();backgrounds=root/'backgrounds';backgrounds.mkdir()
            catalog={}
            for number in range(1,45):
                mesh=root/f'model-{number}.bin';mesh.write_bytes(str(number).encode())
                catalog[str(number)]={'mesh':str(mesh),'mesh_hash':file_hash(mesh),'available':True,'name':str(number)}
            assets=('1','3','8','10','13','14','17','19-1','22','31','33','37')
            catalog['19-1']=catalog.pop('19')
            baseline=root/'baseline.json';atomic_json(baseline,[{'asset_id':a} for _ in range(2) for a in assets])
            groups=[*EXCLUDED_GROUPS,'53_1_1','53_1_2','53_1_3','53_1_4']
            yy,xx=np.mgrid[:48,:64]
            for i,g in enumerate(groups):
                pixels=np.uint8((xx[...,None]*np.array([i+1,2,3])+yy[...,None]+20*i)%240)
                Image.fromarray(pixels).save(backgrounds/(g+'.png'))
            atomic_json(source/'config.json',{'backgrounds':str(backgrounds),'output':str(source)})
            out=root/'out'
            with patch('validate_coherent_expanded.load_catalog',return_value=catalog):
                frozen=freeze(source,out,str(baseline));again=freeze(source,out,str(baseline))
                self.assertEqual(frozen,again)
                self.assertEqual(len(frozen['jobs']),36)
                self.assertEqual(len(frozen['source_models']),44)
                self.assertFalse(set(EXCLUDED_GROUPS)&{r['group'] for r in frozen['selected_backgrounds']})
                result=report(out)
                self.assertEqual(result['denominator'],36)
                self.assertEqual(result['states'],{'not_run':36})
                self.assertEqual(result['cost']['actual_requests'],0)
                self.assertEqual(result['generated_images'],0)
                self.assertIsNone(result['seconds'])
                atomic_json(out/'paused.json',{'requested_by':'user'})
                stopped=report(out)
                self.assertEqual(stopped['states'],{'not_run_paused':36})
                self.assertEqual(stopped['completed_jobs'],0)
                self.assertIn('用户已暂停', (out/'index.html').read_text(encoding='utf8'))
                page=(out/'index.html').read_text(encoding='utf8')
                self.assertNotIn('object-fit:cover',page)
                self.assertTrue(result['source_models_unchanged'])
                Path(catalog['1']['mesh']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'frozen_source_models_changed'):
                    freeze(source,out,str(baseline))


if __name__=='__main__':unittest.main(verbosity=2)
