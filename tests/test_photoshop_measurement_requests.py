"""Bounded model-requested measurement calls cannot mutate Photoshop or infer BRDF."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scrap_composite.photoshop import file_hash
from scrap_composite.photoshop_workflow import fulfill_measurements
from scrap_composite.photography_protocol import PhotoContext


class MeasurementRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_runs_one_registered_read_and_binds_identical_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pixels=np.full((30,40,3),90,np.uint8)
            Image.fromarray(pixels).save(root/'image.png')
            (root/'source.psd').write_bytes(b'fixture PSD')
            image={'path':str(root/'image.png'),'sha256':file_hash(root/'image.png')}
            (root/'state.json').write_text(json.dumps({'image_sha256':image['sha256']}),encoding='utf8')
            exported={'outputs':{'png':image,'psd':{'path':str(root/'source.psd'),'sha256':file_hash(root/'source.psd')}},
                      'readback':{'path':str(root/'state.json'),'sha256':file_hash(root/'state.json')}}
            class Adapter:
                calls=0
                async def inspect(self,path,*,expected_sha256,regions):
                    self.calls+=1
                    self.regions=regions
                    return {'state':{'document':{'id':7}},'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                            'source_unchanged':True,'measurements':[]}
            adapter=Adapter()
            report=SimpleNamespace(findings=[SimpleNamespace(needed_measurements=['local_pixel_statistics'],
                                   region=(.1,.2,.3,.4),reference_region=(.4,.5,.6,.7))])
            context=PhotoContext(images=['frame'],execution_state={'measurement_capabilities':['local_pixel_statistics']})
            updated,evidence=await fulfill_measurements(adapter,exported,context,report)
            self.assertEqual(adapter.calls,1)
            self.assertEqual(len(adapter.regions),2)
            self.assertEqual(evidence['status'],'measured')
            self.assertEqual(updated.measurement_evidence['requested_measurements']['surface_roughness'],'unknown')
            report.findings[0].needed_measurements=['true_BRDF']
            _,evidence=await fulfill_measurements(adapter,exported,context,report)
            self.assertEqual(evidence['status'],'unsupported')
            self.assertEqual(adapter.calls,1)


if __name__=='__main__': unittest.main()
