"""CPU-only public-entry/cache contracts for the opt-in realism experiment."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageOps
from pydantic import ValidationError

from display_batch.common import ROOT, WEIGHTS
from display_batch.effects import parameters, render_identity, optical_delta
from display_batch.pipeline import Pipeline
from display_batch.rendering import part_keys
from display_batch.schema import BatchConfig, EffectSettings, Action, apply_actions
from display_batch import scene_atmosphere
from display_batch.scene import polygon_mask
from display_batch.web import effect_options
from scene_v2 import digest, key


def config(**kw):
    return BatchConfig(backgrounds='fixtures/backgrounds', output='fixtures/output', **kw)


def params(configuration):
    # Exercise the production resolver without constructor side effects or services.
    pipeline=Pipeline.__new__(Pipeline)
    pipeline.config=configuration
    return pipeline.default_params('S1-0')


def request():
    return {'windows':{'object':{'resolution':[512,512]},'shadow':{'resolution':[512,512]}},
        'camera':{'elevation':45}, 'environment':{'realism_light':True,'light_level':.8,
        'dust_color':[.2,.2,.2],'haze':.15,'local_context':{'surface_texture_rms':.03}},
        'appearance':'local','plane_normal':[0,0,1],'light_gain':1.,'mesh_hash':'fixture-mesh',
        'matrix':np.eye(4).tolist(),'yaw':0.,'roughness_delta':0.,'dust_gain':1.,'dust_control':1.,
        'surface_dust':.5,'material_classes':['steel'],
        'generated_finish':{'signature':'fixture-physical-material','version':'physical-layers-v1',
                            'regions':{'signature':'fixture-material-regions'}}}


class EntryContracts(unittest.TestCase):
    def test_veil_at_missing_tile_transitions_without_a_hard_seam(self):
        valid = np.array([[True, False], [True, False]])
        grid = np.float32([[.4, 0.], [.4, 0.]])
        field = scene_atmosphere._field(grid, valid, (240, 60), 0.)
        self.assertAlmostEqual(float(field[30, 0]), .4)
        self.assertEqual(float(field[30, -1]), 0.)
        self.assertLess(float(np.abs(np.diff(field[30])).max()), .01)
        self.assertTrue(np.all(np.diff(field[30]) <= 1e-6))
        air = scene_atmosphere._field(np.repeat(grid[..., None], 3, axis=2), valid, (240, 60), .5)
        self.assertTrue(np.allclose(air[:, -1, :], .5))

    def test_old_configuration_and_manual_fog_semantics_stay_unchanged(self):
        legacy=config()
        self.assertEqual(legacy.preset,'existing')
        self.assertEqual(legacy.workflow,'legacy')
        self.assertIsNone(legacy.effects)
        self.assertNotIn('realism_version',params(legacy))
        manual=config(effects={'fog_extra':.35,'pool_kernel':3,'surface_dust':.5})
        self.assertEqual(manual.effects.fog_mode,'manual_extra')
        self.assertEqual(parameters(manual.effects)['fog_extra'],'0.35')
        self.assertNotIn('realism_version',params(manual))

    def test_realism_preset_resolves_end_to_end_and_roundtrips(self):
        current=config(preset='realism_v1')
        self.assertEqual(current.workflow,'evidence_v1')
        self.assertEqual(current.camera_pipeline,'premultiplied')
        self.assertEqual(current.review_profile,'photographer')
        self.assertEqual(current.chroma_mode,'local_budget')
        self.assertEqual(current.effects.surface_finish,'physical_layers')
        self.assertEqual(current.effects.fog_mode,'local_auto')
        self.assertEqual(current.effects.fog_extra,0.)
        self.assertEqual(current.effects.surface_dust,.5)
        self.assertEqual(params(current)['realism_version'],'realism-v1')
        restored=BatchConfig.model_validate_json(current.model_dump_json())
        self.assertEqual(params(current),params(restored))

    def test_old_effects_with_null_manifest_can_switch_to_realism(self):
        legacy=EffectSettings().model_dump()
        current=config(preset='realism_v1',effects=legacy)
        self.assertTrue(current.effects.surface_finish_manifest)
        self.assertEqual(current.chroma_mode,'local_budget')

    def test_physical_dust_zero_allowed_but_excess_or_duplicate_rust_rejected(self):
        current=config(preset='realism_v1',effects={'surface_dust':0.})
        self.assertEqual(current.effects.surface_dust,0.)
        with self.assertRaises(ValidationError):
            config(preset='realism_v1',effects={'surface_dust':.75})
        with self.assertRaises(ValidationError):
            EffectSettings(surface_finish='physical_layers',surface_finish_manifest='pack.json',rust_spots=.5)
        with self.assertRaises(ValueError):
            apply_actions(params(current),[Action(kind='surface_dust',value='0.75')],['S1-0'])

    def test_switching_finish_changes_render_identity_but_fog_does_not(self):
        current=params(config(preset='realism_v1'))
        old={**current,'surface_finish':'generated_pbr'}
        self.assertNotEqual(render_identity(old,'local'),render_identity(current,'local'))
        photo={**current,'fog_gain':'1.25','pool_kernel':'6','tone':'darker'}
        self.assertEqual(render_identity(current,'local'),render_identity(photo,'local'))
        self.assertNotEqual(render_identity(current,'local'),
                            render_identity({**current,'surface_dust':'0.25'},'local'))

    def test_web_schema_exposes_trial_without_changing_production_default(self):
        response=asyncio.run(effect_options(None))
        body=json.loads(response.body)
        self.assertEqual(response.status_code,200)
        self.assertIn('realism_v1',body['presets'])
        self.assertEqual(body['defaults']['surface_finish'],'existing')
        self.assertIn('physical_layers',body['schema']['properties']['surface_finish']['enum'])
        self.assertFalse(body['integration_modes']['promoted_default'])

    def test_scene_air_and_preset_camera_preserve_geometry_holes_and_foreground(self):
        from checks_realism_environment import fixture, srgb
        from display_batch.local import camera_residual
        from display_batch.atmosphere import fogged_radiance
        from display_batch.photography import linear
        scene=fixture(.3,size=(160,120)); settings=params(config(preset='realism_v1'))
        alpha=np.zeros((120,160),np.float32);alpha[18:102,25:135]=1.;alpha[45:75,65:95]=0.
        original_alpha=alpha.copy(); writable=np.ones_like(alpha,bool);writable[:,105:112]=False
        background=linear(np.asarray(scene['frame'],np.float32)/255)
        rgb=np.full_like(background,.32)
        environment={'camera_psf':{'sigma_px':.95,'edge_samples':50},
                     'sensor':{'noise_sigma':.004},'local_context':{'sample_pixels':900}}
        with tempfile.TemporaryDirectory() as folder:
            context=scene_atmosphere.prepare(scene,folder)
            t,_,air,_=scene_atmosphere.fields_for(context,[0,0,160,120],np.ones_like(alpha),{},None,scene)
            veiled=srgb(fogged_radiance(linear(rgb),t,air))
            delta,record=optical_delta(rgb,veiled,alpha,environment,
                {'transmission_field':t,'color_linear':air.mean(axis=(0,1)).tolist()},100,settings)
            result,edited,details=camera_residual(background,rgb,alpha,background,environment,
                                                   settings,42,writable,object_delta=delta)
        np.testing.assert_array_equal(alpha,original_alpha)
        np.testing.assert_allclose(linear(result[~writable]),background[~writable],atol=1e-7)
        np.testing.assert_allclose(linear(result[54:66,74:86]),background[54:66,74:86],atol=1e-7)
        np.testing.assert_allclose(linear(result[:8]),background[:8],atol=1e-7)
        self.assertFalse(edited[~writable].any())
        self.assertTrue(details['geometric_alpha_unchanged'])
        self.assertTrue(details['effects_filtered_once'])
        self.assertFalse(record['additional_fog_applied'])


class CacheContracts(unittest.TestCase):
    def test_physical_material_is_not_keyed_by_background_pigment_or_haze(self):
        original=request(); changed=copy.deepcopy(original)
        changed['environment'].update(dust_color=[.5,.1,.3],haze=.4,
                                      local_context={'surface_texture_rms':.2})
        self.assertEqual(part_keys(original,'CPU','code-v1'),part_keys(changed,'CPU','code-v1'))
        # Historical finishes retain their old environment-dependent surface behavior.
        original['generated_finish']['version']='generated-pbr-v1'
        changed['generated_finish']['version']='generated-pbr-v1'
        self.assertNotEqual(part_keys(original,'CPU','code-v1')['object'],
                            part_keys(changed,'CPU','code-v1')['object'])

    def test_material_change_keeps_empty_receiver_but_light_and_code_invalidate_all(self):
        base=request(); first=part_keys(base,'CPU','code-v1')
        material=copy.deepcopy(base);material['generated_finish']['signature']='new-material'
        second=part_keys(material,'CPU','code-v1')
        self.assertEqual(first['without_shadow'],second['without_shadow'])
        self.assertNotEqual(first['object'],second['object'])
        self.assertNotEqual(first['with_shadow'],second['with_shadow'])
        light=copy.deepcopy(base);light['environment']['light_level']=1.1
        for name in first:
            self.assertNotEqual(first[name],part_keys(light,'CPU','code-v1')[name])
            self.assertNotEqual(first[name],part_keys(base,'CPU','code-v2')[name])

    def test_shadow_extent_does_not_move_or_invalidate_object_grid(self):
        base=request();wide=copy.deepcopy(base)
        wide['windows']['shadow']['resolution']=[768,768]
        first=part_keys(base,'CPU','code-v1'); second=part_keys(wide,'CPU','code-v1')
        self.assertEqual(first['object'],second['object'])
        self.assertNotEqual(first['with_shadow'],second['with_shadow'])
        self.assertNotEqual(first['without_shadow'],second['without_shadow'])


class ActualBackgroundContract(unittest.TestCase):
    def test_scene_fields_are_consistent_across_two_actual_candidate_crops(self):
        source=ROOT/'outputs/全部启用物品-三场景-20261116'
        if not (source/'results.jsonl').is_file():self.skipTest('optional local historical batch absent')
        row=next(json.loads(line) for line in (source/'results.jsonl').read_text(encoding='utf8').splitlines()
                 if line and json.loads(line).get('scene_signature'))
        background=Path(row['background']); cache=ROOT/'work/display-batch-cache'
        analysis=cache/'scenes'/row['scene_signature']/'analysis.json'
        weights=[WEIGHTS/'mobile_sam.pt',WEIGHTS/'depth/model.safetensors',WEIGHTS/'matte/birefnet-general-lite.onnx']
        if not all(p.is_file() for p in [background,analysis,*weights]):
            self.skipTest('optional cached perception inputs absent')
        perception=cache/('scene-'+key(digest(background),[digest(p) for p in weights])+'.npz')
        if not perception.is_file():self.skipTest('no existing perception cache; inference deliberately disabled')
        with np.load(perception) as data:depth=data['depth'].copy()
        answer=json.loads(analysis.read_text(encoding='utf8'))
        with Image.open(background) as im:frame=ImageOps.exif_transpose(im).convert('RGB')
        size=frame.size; protected=polygon_mask(answer.get('protected',[]),size)
        allowed=polygon_mask([answer['hopper']],size)&~polygon_mask(answer.get('excluded',[]),size)&~protected
        scene={'key':row['scene_signature'],'frame':frame,'size':size,'allowed':allowed,
               'protected':protected,'depth_native':cv2.resize(depth,size,interpolation=cv2.INTER_LINEAR)}
        # The background is real; no image observer or new perception runs in this test.
        candidate_data=json.loads((source/'jobs'/row['job_id']/'candidates.json').read_text(encoding='utf8'))
        candidates=candidate_data['candidates'][:2]
        self.assertEqual(len(candidates),2)
        w,h=size; common=[w//3,h//3,w//3+64,h//3+64]
        larger=[common[0]-16,common[1]-16,common[2]+16,common[3]+16]
        with tempfile.TemporaryDirectory() as folder:
            first=scene_atmosphere.prepare(scene,folder)
            second=scene_atmosphere.prepare({**scene,'asset_id':'other','candidate':candidates[1]},folder)
            self.assertEqual(first['signature'],second['signature'])
            t1,b1,a1,_=scene_atmosphere.fields_for(first,common,np.zeros((64,64)),{},candidates[0],scene)
            t2,b2,a2,_=scene_atmosphere.fields_for(second,larger,np.ones((96,96)),{},candidates[1],scene)
            np.testing.assert_array_equal(t1,t2[16:-16,16:-16])
            np.testing.assert_array_equal(b1,b2[16:-16,16:-16])
            np.testing.assert_array_equal(a1,a2[16:-16,16:-16])
            self.assertTrue(np.isfinite(t1).all())
            self.assertEqual(first['summary']['bounds'],[0,0,w,h])


if __name__=='__main__':unittest.main()
