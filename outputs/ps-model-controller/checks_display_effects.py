"""Offline contracts for online effects. Fake HTTP/model calls are not visual acceptance."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image
from pydantic import ValidationError
from starlette.testclient import TestClient

from display_batch import web
from display_batch.effects import parameters, optical_delta, render_identity, dust_profile
from display_batch.review import allowed_actions, repair, inspect
from display_batch.pipeline import write_summary
from display_batch.schema import BatchConfig, EffectSettings, EffectsReview, Action, Defect, Scores, effects_review_schema
sys.path.insert(0, str(Path(__file__).parent/'placement-web'))
from server import app


def params(**changes):
    return {'candidate':'C1','material':'original','light':'scene','tone':'scene',
            'photo_model':'local','photo_style':'dusty_scrap','atmosphere':'auto',
            **dict.fromkeys(('haze','dust','chroma','contrast','sharpness'),'auto'),
            **parameters(EffectSettings()),**changes}


def comp():
    return {'photo':{'effects':{'fog_base':.2,'edge_samples':40,'sample_pixels':100,
                               'surface_dust_applied':True}}, 'checks':{'support':{'status':'unknown'}}}


def observed(code, action):
    return EffectsReview(trace='visible',scores=Scores(**dict.fromkeys(
        ('support','occlusion','lighting','edge','texture','material','structure'),3)),
        defects=[Defect(code=code,severity='major',region=(.2,.2,.8,.8),evidence='测试明确缺陷')],
        actions=[action],evidence='离线接口测试')


class EffectTests(unittest.TestCase):
    def test_legacy_and_opt_in_defaults(self):
        old=BatchConfig(backgrounds='.',output='x')
        self.assertIsNone(old.effects)
        self.assertEqual(old.appearance,'reference')
        new=BatchConfig(backgrounds='.',output='x',effects={})
        self.assertEqual((new.effects.pool_kernel,new.effects.fog_extra,new.effects.surface_dust),(3,.35,.5))
        self.assertEqual((new.appearance,new.photo_style),('local','dusty_scrap'))

    def test_invalid_values_are_rejected_not_clamped(self):
        for data in ({'pool_kernel':0},{'pool_kernel':3.5},{'pool_kernel':True},
                     {'fog_extra':35},{'fog_extra':float('nan')},{'surface_dust':-1},
                     {'surface_dust':float('inf')},{'pool_weight':1.2},{'unknown':1}):
            with self.subTest(data=data),self.assertRaises(ValidationError):EffectSettings(**data)

    def test_shader_profile_has_independent_paint_scale(self):
        self.assertEqual(dust_profile(.5),{'version':'surface-dust-strength-v2',
            'paint_dust_scale':.5,'single_deposit':True,'dust_scale':.5})

    def test_photographic_actions_share_render_material_changes_invalidate(self):
        original=params()
        key=render_identity(original,'local')
        for name,value in [('fog_extra','0.55'),('pool_kernel','6'),('pool_weight',.5)]:
            self.assertEqual(key,render_identity({**original,name:value},'local'))
        for name,value in [('surface_dust','0.25'),('effects_lighting','local'),('light','softer')]:
            self.assertNotEqual(key,render_identity({**original,name:value},'local'))

    def test_optics_are_independent_preserve_alpha_holes_and_exterior(self):
        rng=np.random.default_rng(20)
        rgb=rng.uniform(.2,.8,(60,70,3)).astype(np.float32)
        alpha=np.zeros((60,70),np.float32);alpha[8:52,8:62]=1;alpha[25:35,29:39]=0
        before=alpha.copy()
        env={'local_context':{'confidence':'estimated','sample_pixels':100,'basis':'test'},
             'camera_psf':{'sigma_px':.8,'edge_samples':40}}
        air={'amount':.2,'color_linear':[.3,.3,.3]}
        a,ma=optical_delta(rgb,rgb,alpha,env,air,200,params())
        b,mb=optical_delta(rgb,rgb,alpha,env,air,200,params(surface_dust='0.25'))
        np.testing.assert_array_equal(a,b)
        c,mc=optical_delta(rgb,rgb,alpha,env,air,200,params(fog_extra='0.55'))
        self.assertGreater(np.max(abs(a-c)),.01)
        self.assertEqual(ma['pool_weight_max'],mc['pool_weight_max'])
        d,md=optical_delta(rgb,rgb,alpha,env,air,200,params(pool_kernel='6'))
        self.assertGreater(np.max(abs(a-d)),.01)
        self.assertEqual(ma['fog_extra_max'],md['fog_extra_max'])
        for effect in (a,b,c,d):self.assertFalse(np.any(effect[alpha==0]))
        np.testing.assert_array_equal(alpha,before)

    def test_disabled_and_insufficient_evidence_effects_have_zero_delta(self):
        rgb=np.full((40,40,3),.4,np.float32);alpha=np.ones((40,40),np.float32)
        env={'local_context':{'confidence':'unknown','sample_pixels':0,'basis':'test'},
             'camera_psf':{'sigma_px':.8,'edge_samples':0}}
        air={'amount':.2,'color_linear':[.3]*3}
        delta,_=optical_delta(rgb,rgb,alpha,env,air,100,params())
        self.assertFalse(np.any(delta))
        env['local_context'].update(confidence='estimated',sample_pixels=100)
        env['camera_psf']['edge_samples']=40
        delta,info=optical_delta(rgb,rgb,alpha,env,air,100,params(atmosphere='off',pool_kernel='1'))
        self.assertFalse(np.any(delta))
        self.assertEqual(info['fog_extra_max'],0)

    def test_online_actions_exclude_plastic_off_zero_and_locked_effects(self):
        value=comp();value['photo']['effects']['surface_dust_applied']=False
        choices=allowed_actions(params(atmosphere='off',pool_weight=0),value,['C1'],render_style='local')
        for key in ('surface_dust','fog_extra','pool_kernel','dust','haze','sharpness'):
            self.assertNotIn(key,choices)
        choices=allowed_actions(params(effects_ai_adjust=False),comp(),['C1'],render_style='local')
        for key in ('surface_dust','fog_extra','pool_kernel'):self.assertNotIn(key,choices)
        uncertain=comp();uncertain['photo']['local_context']={'confidence':'unknown'}
        choices=allowed_actions(params(),uncertain,['C1'],render_style='local')
        for key in ('fog_extra','pool_kernel'):self.assertNotIn(key,choices)

    def test_review_action_direction_scope_and_stage(self):
        current=params();allowed=allowed_actions(current,comp(),['C1'],render_style='local')
        for kind,code,value,stage in [('fog_extra','haze_clear','0.45','photography'),
                                    ('pool_kernel','too_blurry','2','photography'),
                                    ('surface_dust','surface_dust_heavy','0.25','material')]:
            review=observed(code,Action(kind=kind,value=value))
            updated,log=repair(current,review,allowed,['C1'],lambda _:False)
            self.assertEqual(updated[kind],value)
            self.assertEqual(log[0]['stage'],stage)
        review=observed('haze_heavy',Action(kind='fog_extra',value='0.55'))
        updated,log=repair(current,review,allowed,['C1'],lambda _:False)
        self.assertIsNone(updated);self.assertEqual(log[0]['disposition'],'wrong_effect_direction')
        review=observed('haze_heavy',Action(kind='surface_dust',value='0.25'))
        self.assertIsNone(repair(current,review,allowed,['C1'],lambda _:False)[0])

    def test_duplicate_unsupported_and_repeated_actions_do_not_schedule(self):
        current=params();allowed=allowed_actions(current,comp(),['C1'],render_style='local')
        for action in (Action(kind='fog_extra',value='0.35'),Action(kind='fog_extra',value='1')):
            self.assertIsNone(repair(current,observed('haze_clear',action),allowed,['C1'],lambda _:False)[0])
        self.assertIsNone(repair(current,observed('haze_clear',Action(kind='fog_extra',value='0.45')),
                                 allowed,['C1'],lambda _:True)[0])

    def test_model_schema_rejects_unoffered_kind_value_pairs(self):
        schema=effects_review_schema({'fog_extra':['0.25','0.45'],'surface_dust':['0.75']})
        valid=observed('haze_clear',Action(kind='fog_extra',value='0.45')).model_dump()
        self.assertEqual(schema.model_validate(valid).actions[0].value,'0.45')
        for action in ({'kind':'surface_dust','value':'0.8'},
                       {'kind':'fog_extra','value':'0.75'},{'kind':'pool_kernel','value':'3'}):
            with self.assertRaises(ValidationError):schema.model_validate({**valid,'actions':[action]})
        empty=effects_review_schema({})
        with self.assertRaises(ValidationError):empty.model_validate(valid)
        empty.model_validate({**valid,'actions':[]})

    def test_http_options_and_test_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bg=root/'backgrounds';bg.mkdir();Image.new('RGB',(32,32)).save(bg/'a.jpg')
            catalog={'3':{'available':True,'mesh_hash':'test'}}
            with patch.object(web,'ROOT',root),patch.object(web,'DATA',root/'configs'),\
                 patch.object(web,'load_catalog',return_value=catalog),patch.object(web,'launch') as launch,TestClient(app) as client:
                self.assertEqual(client.get('/api/batches/effects').json()['defaults']['fog_extra'],.35)
                for extra in ({'effects':{'fog_extra':35}},{'effects':None},{'offline':True},{'count':13}):
                    response=client.post('/api/batches/test',json={'backgrounds':str(bg),'assets':['3'],**extra})
                    self.assertEqual(response.status_code,400)
                launch.assert_not_called()
                response=client.post('/api/batches/test',json={'backgrounds':str(bg),'assets':['3']})
                self.assertEqual(response.status_code,202)
                info=response.json();stored=json.loads((root/'configs'/(info['id']+'.json')).read_text(encoding='utf8'))
                self.assertEqual(stored['count'],1)
                self.assertEqual(stored['effects']['surface_dust'],.5)
                self.assertEqual(client.get(info['status_url']).json()['effects'],info['effects'])
                launch.assert_called_once()

    def test_reused_batch_history_does_not_escape_artifact_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'new';out.mkdir()
            result={'job_id':'j','asset_id':'3','status':'review','image':str(out/'final.png'),
                    'iterations':[{'iteration':0,'params':params(),'image':str(root/'old'/'iteration.png')}]}
            store=SimpleNamespace(rows=lambda:[{'payload':{},'state':'review','result':result}],
                                  counts=lambda:{'review':1})
            write_summary(out,store)
            page=(out/'index.html').read_text(encoding='utf8')
            self.assertIn('final.png',page)
            self.assertNotIn('iteration.png',page)


class ReviewInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_review_contract_contains_controls_and_final_full_image(self):
        review=observed('surface_dust_heavy',Action(kind='surface_dust',value='0.25'))
        vision=SimpleNamespace(ask=AsyncMock(return_value=review))
        p=params();value=comp();value.update(image='frame.png',preview='frame.full.jpg',crop='frame.crop.jpg')
        asset={'name':'test','semantic':{'key_features':['hole']},'board':'model.jpg'}
        allowed=allowed_actions(p,value,['C1'],render_style='local')
        result=await inspect(vision,'',asset,value,p,allowed,Path('render'))
        args=vision.ask.call_args.args
        self.assertEqual(args[0],'effects-review')
        self.assertTrue(issubclass(args[1],EffectsReview))
        self.assertIn('surface_dust',args[3]['allowed'])
        self.assertEqual(args[4][0],'frame.full.jpg')
        self.assertEqual(result.actions[0].kind,'surface_dust')


if __name__=='__main__':unittest.main(verbosity=2)
