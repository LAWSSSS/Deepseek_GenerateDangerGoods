"""Run the render-worker light path in Blender's own Python, without rendering."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from display_batch.common import BLENDER, HERE


class RealismWorkerTests(unittest.TestCase):
    @unittest.skipUnless(BLENDER.is_file(), 'Blender is not installed')
    def test_actual_blender_light_rig_has_no_host_imaging_dependencies(self):
        with TemporaryDirectory(prefix='scrap-realism-worker-') as temporary:
            folder=Path(temporary);result=folder/'result.json';script=folder/'smoke.py'
            script.write_text('''import sys, json, importlib.abc
from pathlib import Path
import bpy
import numpy as np

# Do not accidentally pass after someone adds host image packages to Blender.
# The worker path must use only Blender's existing Python/numpy environment.
class HostImagingGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cv2', 'PIL', 'pydantic'}:
            raise ModuleNotFoundError('Host-only dependency reached Blender light rig: '+fullname)
        return None

sys.meta_path.insert(0, HostImagingGuard())
sys.path.insert(0, CONTROLLER_PATH)
from display_batch.realism_light import light_rig
from display_batch.geometry import camera_for

camera=camera_for(np.array([[-1.,-1.,0.],[1.,1.,1.]]),45.)
outputs=[]
for direction,confidence in [(None,0.),([.6,-.8],.4)]:
    environment={'local_irradiance':{'direction':{'direction':direction,'confidence':confidence},
                                    'level':.7,'fill_ratio':.6}}
    sky,lights=light_rig(environment,camera,1.,(32,64))
    brighter_sky,brighter=light_rig(environment,camera,1.25,(32,64))
    assert sky.shape==(32,64) and np.isfinite(sky).all()
    assert len(lights)==2 and all(len(position)==3 and power>0 and size>=4.8 for position,power,size in lights)
    assert np.isfinite(np.array([(*p,power,size) for p,power,size in lights])).all()
    np.testing.assert_array_equal(sky,brighter_sky)
    np.testing.assert_allclose([light[1] for light in brighter],np.array([light[1] for light in lights])*1.25)
    outputs.append({'direction':direction,'lights':len(lights),'sky_shape':list(sky.shape)})
assert not any(name in sys.modules for name in ('cv2','PIL','pydantic'))
Path(RESULT_PATH).write_text(json.dumps({'passed':True,'blender':bpy.app.version_string,
    'render_calls':0,'gpu_used':False,'cases':outputs}),encoding='utf8')
'''.replace('CONTROLLER_PATH',repr(str(HERE))).replace('RESULT_PATH',repr(str(result))),encoding='utf8')
            process=subprocess.run([str(BLENDER),'--background','--factory-startup','--python-exit-code','1',
                                    '--python',str(script)],capture_output=True,text=True,
                                    encoding='utf8',errors='replace',timeout=60)
            output=process.stdout[-6000:]+process.stderr[-6000:]
            self.assertEqual(process.returncode,0,output)
            self.assertTrue(result.is_file(),output)
            data=json.loads(result.read_text(encoding='utf8'))
            self.assertTrue(data['passed'])
            self.assertEqual(data['render_calls'],0)
            self.assertFalse(data['gpu_used'])


if __name__=='__main__':unittest.main(verbosity=2)
