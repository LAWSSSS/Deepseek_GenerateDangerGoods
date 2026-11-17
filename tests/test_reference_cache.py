"""Frozen material inputs and in-job render reuse must follow effective state."""
import asyncio
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ps-model-controller'))
from display_batch.common import file_hash
from display_batch.effects import parameters, render_identity
from display_batch.pipeline import Pipeline
from display_batch.reference_material import (MAPS, ReferenceInputChanged, compatible_snapshot,
    freeze_library, frozen_reference, verify_snapshot)
from display_batch.schema import BatchConfig, EffectSettings
from display_batch.store import Store
from scrap_composite.render_controls import DEFAULTS, RENDER_CONTROLS


class ReferenceCacheTests(unittest.TestCase):
    def fixture(self, folder):
        maps={}
        for kind in MAPS:
            path=folder/(kind+'.png')
            Image.new('RGB' if kind=='albedo' else 'L',(32,32),128).save(path)
            maps[kind]={'path':str(path),'sha256':file_hash(path)}
        entry=folder/'entry.json'
        entry.write_text(json.dumps({'version':'reference-pbr-v1','status':'estimated','scale_m':.12,
                                     'maps':maps}),encoding='utf8')
        library=folder/'library.json'
        library.write_text(json.dumps({'version':'reference-material-library-v1','assets':{
            'a'*64:{'asset_id':'25','reference_path':'entry.json','reference_sha256':file_hash(entry)},
            # Unselected invalid entries must not block this batch.
            'b'*64:{'asset_id':'99','reference_path':'missing.json','reference_sha256':'missing'}}}),encoding='utf8')
        return library

    def test_effective_material_and_light_actions_change_memory_render_identity(self):
        base={'candidate':'S1','material':'original','light':'scene',**parameters(EffectSettings(
            surface_finish='physical_layers',surface_finish_manifest='fixture',asset_refinement='component_v1'))}
        initial=render_identity(base,'local')
        for key,choices in RENDER_CONTROLS.items():
            value=next(choice for choice in choices if choice!=DEFAULTS[key])
            with self.subTest(control=key):
                self.assertNotEqual(initial,render_identity({**base,key:value},'local'))
        for key,value in (('fog_gain','1.5'),('fog_extra','.55'),('pool_kernel','6'),('pool_weight',.5),('tone','darker')):
            self.assertEqual(initial,render_identity({**base,key:value},'local'))
        self.assertEqual(initial,render_identity({**base,**{key:float(value) for key,value in DEFAULTS.items()}},'local'))
        with self.assertRaises(ValueError):render_identity({**base,'relief_gain':True},'local')

    def test_frozen_selection_is_independent_and_tampering_is_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);library=self.fixture(root)
            snapshot=freeze_library(library,{'a'*64:'25'})
            self.assertEqual(set(snapshot['entries']),{'a'*64})
            self.assertTrue(compatible_snapshot(snapshot,snapshot))
            one=frozen_reference(snapshot,'a'*64,library)
            one['maps']['height']['sha256']='edited-in-caller'
            self.assertNotEqual(one['maps'],frozen_reference(snapshot,'a'*64,library)['maps'])
            with self.assertRaisesRegex(ValueError,'asset ID'):freeze_library(library,{'a'*64:'wrong'})
            with self.assertRaisesRegex(ReferenceInputChanged,'preflight'):frozen_reference(None,'a'*64,library)
            Image.new('L',(32,32),80).save(root/'roughness.png')
            with self.assertRaisesRegex(ReferenceInputChanged,'input changed'):verify_snapshot(snapshot)

    def test_same_path_updated_pack_invalidates_resume_and_pilot_reuse(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);library=self.fixture(root)
            first=freeze_library(library,{'a'*64:'25'})
            entry=root/'entry.json';record=json.loads(entry.read_text())
            Image.new('L',(32,32),90).save(root/'roughness.png')
            record['maps']['roughness']['sha256']=file_hash(root/'roughness.png')
            entry.write_text(json.dumps(record),encoding='utf8')
            index=json.loads(library.read_text());index['assets']['a'*64]['reference_sha256']=file_hash(entry)
            library.write_text(json.dumps(index),encoding='utf8')
            second=freeze_library(library,{'a'*64:'25'})
            self.assertNotEqual(first['signature'],second['signature'])
            self.assertFalse(compatible_snapshot(second,first))
            with self.assertRaises(ReferenceInputChanged):verify_snapshot(first)
            queue=Store(root/'jobs.sqlite')
            try:
                queue.initialize(first['signature'],[])
                with self.assertRaises(ValueError):queue.initialize(second['signature'],[])
            finally:queue.close()

    def test_pipeline_preflights_before_queue_and_old_mode_has_no_new_signature_field(self):
        class CapturedSignature(Exception):pass
        with TemporaryDirectory() as temp:
            root=Path(temp);library=self.fixture(root)
            jobs=[{'asset_id':'25','mesh_hash':'a'*64}]
            def capture(config):
                captured=[];pipe=Pipeline.__new__(Pipeline)
                pipe.config=config;pipe.catalog={};pipe.out=root;pipe.surface_pack=None
                pipe.vision=SimpleNamespace(endpoint=SimpleNamespace(model_dump=lambda:{}))
                def initialize(value,_):
                    captured.append(value);raise CapturedSignature()
                pipe.store=SimpleNamespace(initialize=initialize)
                with patch('display_batch.pipeline.make_jobs',return_value=jobs),patch('display_batch.pipeline.implementation_manifest',return_value={}):
                    with self.assertRaises(CapturedSignature):asyncio.run(pipe.run())
                return captured[0],getattr(pipe,'reference_snapshot',None)
            config=BatchConfig(backgrounds=str(root),output=str(root),effects={
                'surface_finish':'physical_layers','surface_finish_manifest':'fixture',
                'reference_material_manifest':str(library)})
            one,snapshot=capture(config)
            self.assertIsNotNone(snapshot)
            entry=root/'entry.json';data=json.loads(entry.read_text());data['scale_m']=.10
            entry.write_text(json.dumps(data),encoding='utf8')
            index=json.loads(library.read_text());index['assets']['a'*64]['reference_sha256']=file_hash(entry)
            library.write_text(json.dumps(index),encoding='utf8')
            two,_=capture(config)
            self.assertNotEqual(one,two)
            old=BatchConfig(backgrounds=str(root),output=str(root))
            first,none=capture(old);self.assertIsNone(none)
            library.write_text('invalid unselected library',encoding='utf8')
            second,none=capture(old)
            self.assertEqual(first,second);self.assertIsNone(none)

    def test_changed_inputs_pause_queue_without_invoking_model_or_publishing(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);library=self.fixture(root)
            config=BatchConfig(backgrounds=str(root),output=str(root),offline=True,cpu_concurrency=1,
                effects={'surface_finish':'physical_layers','surface_finish_manifest':'fixture',
                         'reference_material_manifest':str(library)})
            pipe=Pipeline.__new__(Pipeline)
            pipe.config=config;pipe.catalog={};pipe.out=root;pipe.surface_pack=None;pipe.run_id='fixture'
            pipe.store=Store(root/'jobs.sqlite')
            pipe.vision=SimpleNamespace(endpoint=SimpleNamespace(model_dump=lambda:{}),usage=lambda:[],
                                       calls=[],ask=AsyncMock(),close=AsyncMock())
            pipe.blender=SimpleNamespace(close=AsyncMock(),seconds=0,operation_seconds={},start_seconds=0,
                                         wait_seconds=0,memory_retries=0)
            pipe.renders=SimpleNamespace(totals={},timings={})
            jobs=[{'asset_id':'25','mesh_hash':'a'*64,'background':'fixture.png'}]
            async def changed(_):
                Image.new('L',(32,32),70).save(root/'roughness.png')
                pipe.assert_reference_inputs()
                raise AssertionError('changed input was accepted')
            pipe.job=changed
            def summary(*args):return {'run':dict(args[3])}
            with patch('display_batch.pipeline.make_jobs',return_value=jobs),patch('display_batch.pipeline.implementation_manifest',return_value={}),\
                 patch('display_batch.pipeline.catalog_preflight',return_value=[]),patch('display_batch.pipeline.write_summary',side_effect=summary):
                result=asyncio.run(pipe.run())
            self.assertEqual(result['run']['state'],'paused_inputs')
            self.assertIn('reference input changed',result['run']['error'])
            pipe.vision.ask.assert_not_called()
            reopened=Store(root/'jobs.sqlite')
            try:
                row=reopened.rows()[0]
                self.assertEqual(row['state'],'queued')
                self.assertIsNone(row['result'])
            finally:reopened.close()


if __name__=='__main__':unittest.main()
