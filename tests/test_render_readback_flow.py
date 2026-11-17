"""Worker readbacks survive render-part caches and reach structured scene input."""
import sys
from pathlib import Path
import tempfile
import unittest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ps-model-controller'))
from display_batch.rendering import RenderCache, PART_FILES
from display_batch.structured import frame_inputs


class RenderReadbackFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_worker_readback_is_retained_on_cache_hit(self):
        material={'basis':'actual_node_fixture','control_domains':{'oxide_gain':False}}
        lighting={'basis':'actual_lamp_fixture','lights':[{'energy_W':17}]}
        class Worker:
            device='CPU'
            calls=0
            async def ask(self,operation,request,timeout):
                self.calls+=1
                folder=Path(request['parts']['object'])
                for name in PART_FILES['object']:
                    if name.endswith('.png'):Image.new('RGBA',(4,4)).save(folder/name)
                    else:(folder/name).write_bytes(b'fake EXR fixture; never parsed')
                return {'camera':{},'device':'CPU','anchor_uv':[.5,.5],'contact_polygons':[],
                        'imports':1,'timings':{},'material_readback':material,'lighting_readback':lighting}
        with tempfile.TemporaryDirectory() as directory:
            worker=Worker();cache=RenderCache(directory,worker)
            window={'rect':[0,0,4,4]}
            request={'windows':{'object':window,'shadow':window},'environment':{},'render_detail':'standard'}
            _,first,hit=await cache._get(request,{'object':'fixture'}, {},False)
            self.assertFalse(hit)
            _,second,hit=await cache._get(request,{'object':'fixture'}, {},False)
            self.assertTrue(hit)
            self.assertEqual(worker.calls,1)
            for meta in (first,second):
                self.assertEqual(meta['material_readback'],material)
                self.assertEqual(meta['lighting_readback'],lighting)

    def test_structured_context_keeps_applied_readback_evidence(self):
        material={'basis':'actual_node_fixture','state_sha256':'a'*64}
        lighting={'basis':'actual_lamp_fixture','state_sha256':'b'*64}
        comp={'checks':{'size':{'status':'pass','pixels':[100,100]}},'edit_bounds':[10,10,40,40],
              'image':'fixture.png','photo':{'material_readback':material,'lighting_readback':lighting}}
        context,_=frame_inputs({'name':'fixture','semantic':{}},comp)
        self.assertEqual(context.measurement_evidence['material_readback'],material)
        self.assertEqual(context.measurement_evidence['lighting_readback'],lighting)


if __name__=='__main__':unittest.main()
