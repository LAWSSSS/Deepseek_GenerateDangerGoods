"""In-process CLI/API checks; launch is mocked, no server or renderer starts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,patch
from PIL import Image
from starlette.applications import Starlette
from starlette.testclient import TestClient

import batch_display
from display_batch import web
from display_batch.schema import BatchConfig


class CoherentInterfaceTests(unittest.TestCase):
    def test_cli_preserves_config_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            for stored,override,expected in [(None,None,'legacy'),('coherent',None,'coherent'),
                                              ('legacy','coherent','coherent'),('coherent','legacy','legacy')]:
                raw={'backgrounds':temp,'output':str(Path(temp)/'out')}
                if stored is not None:raw['integration_mode']=stored
                path.write_text(json.dumps(raw),encoding='utf8')
                args=['batch_display.py','run','--config',str(path)]
                if override:args+=['--integration-mode',override]
                with patch('sys.argv',args),patch.object(batch_display,'Pipeline') as pipeline,contextlib.redirect_stdout(io.StringIO()):
                    pipeline.return_value.run=AsyncMock(return_value={'usage_current_run':{'calls':[]}})
                    batch_display.main()
                    self.assertEqual(pipeline.call_args.args[0].integration_mode,expected)
                    pipeline.return_value.run.assert_awaited_once()

    def test_api_persists_reports_and_validates_identical_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bg=root/'backgrounds';bg.mkdir();Image.new('RGB',(32,32)).save(bg/'one.jpg')
            catalog={'3':{'available':True,'mesh_hash':'fixture'}}
            with patch.object(web,'ROOT',root),patch.object(web,'DATA',root/'configs'),\
                 patch.object(web,'load_catalog',return_value=catalog),patch.object(web,'launch') as launch,\
                 TestClient(Starlette(routes=web.routes)) as client:
                for endpoint in ('/api/batches','/api/batches/test'):
                    for requested in (None,'legacy','coherent'):
                        data={'backgrounds':str(bg),'assets':['3'],'count':1}
                        if requested is not None:data['integration_mode']=requested
                        response=client.post(endpoint,json=data)
                        self.assertEqual(response.status_code,202,response.text)
                        info=response.json();expected=requested or 'legacy'
                        self.assertEqual(info['integration_mode'],expected)
                        self.assertEqual(info['effective_modes']['integration'],expected)
                        stored=json.loads((root/'configs'/(info['id']+'.json')).read_text(encoding='utf8'))
                        self.assertEqual(stored['integration_mode'],expected)
                        self.assertEqual(client.get(info['status_url']).json()['integration_mode'],expected)
                        if expected=='coherent':
                            self.assertEqual(stored['effects']['fog_mode'],'local_auto')
                            self.assertEqual(stored['effects']['weathering'],'preserve_source')
                            self.assertEqual(stored['effects']['lighting'],'local')
                            self.assertEqual(stored['effects']['fog_extra'],0.)
                self.assertEqual(launch.call_count,6)
                manual=client.post('/api/batches/test',json={'backgrounds':str(bg),'assets':['3'],'count':1,
                    'integration_mode':'coherent','effects':{'fog_mode':'manual_extra','fog_extra':.2,'pool_kernel':6}})
                self.assertEqual(manual.status_code,202,manual.text)
                self.assertEqual(manual.json()['effects']['fog_mode'],'manual_extra')
                self.assertEqual(manual.json()['effects']['fog_extra'],.2)
                self.assertEqual(manual.json()['effects']['pool_kernel'],6)
                self.assertEqual(launch.call_count,7)
                bad=client.post('/api/batches',json={'backgrounds':str(bg),'count':1,'integration_mode':'unknown'})
                self.assertEqual(bad.status_code,400)
                self.assertEqual(launch.call_count,7)
                modes=client.get('/api/batches/effects').json()['integration_modes']
                self.assertEqual(modes['default'],'legacy')
                self.assertFalse(modes['promoted_default'])
                self.assertEqual({v['value'] for v in modes['options']},{'legacy','coherent'})

    def test_example_uses_opt_in_without_changing_existing_default(self):
        folder=Path(__file__).parent
        example=BatchConfig.model_validate_json((folder/'coherent-integration-config.example.json').read_text(encoding='utf8'))
        self.assertEqual(example.integration_mode,'coherent')
        self.assertEqual(example.effects.weathering,'preserve_source')
        self.assertEqual(example.effects.surface_dust,.5)
        old=BatchConfig(backgrounds='unused',output='unused')
        self.assertEqual(old.integration_mode,'legacy')
        self.assertEqual(BatchConfig.model_validate_json(old.model_dump_json()).integration_mode,'legacy')


if __name__=='__main__':unittest.main(verbosity=2)
