"""Regressions at the visual report, planner and camera boundaries."""
import unittest
from unittest.mock import AsyncMock

import numpy as np

from display_batch.common import ROOT
from display_batch.effects import optical_delta
from display_batch.schema import BatchConfig
from display_batch.structured import planner_capabilities
from scrap_composite.contracts import ExecutionPlan, PlanProposal, VisualObservation, validate_plan
from scrap_composite.workflow import observe, plan_execution


IMAGE = 'f' * 64


def observation():
    return provider_observation().for_frame(IMAGE, 'object')


def provider_observation():
    return VisualObservation(trace='visible',
        scores=dict.fromkeys(('support', 'occlusion', 'lighting', 'edge', 'texture', 'material', 'structure'), 3),
        findings=[{'code':'lighting','region':(.1,.2,.3,.4),
                   'severity':'major','visibility':'visible','scope':'image','confidence':.8,
                   'direction':'increase','evidence':'物品局部过暗'}], summary='照明待修正')


def plan(value=8, image=IMAGE):
    return ExecutionPlan(version='execution-plan-v1', image_sha256=image, status='ready', note='局部提亮',
        operations=[{'capability':'brightness','target':'object','value':value,
                     'evidence_ids':['f1'],'expected_effect':'提亮物品暗部'}])


CAPABILITIES = {'brightness': {'range': [-20,20], 'causes':['lighting']}}


class WorkflowChecks(unittest.IsolatedAsyncioTestCase):
    async def test_observe_then_compile_absolute_action(self):
        proposal=PlanProposal.model_validate(plan().model_dump(exclude={
            'version':True, 'image_sha256':True, 'operations':{'__all__':{'target'}}}))
        client=type('Model', (), {'ask':AsyncMock(side_effect=[provider_observation(),proposal])})()
        report=await observe(client, IMAGE, ['frame.png'], {'name':'part'})
        action, changes=await plan_execution(client, report, {'brightness':0}, CAPABILITIES, {})
        self.assertEqual(changes, {'brightness':8})
        self.assertTrue(client.ask.await_args.kwargs['planning'])
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            validate_plan(action, report, changes, CAPABILITIES)

    def test_stale_wrong_direction_out_of_range(self):
        for action, message in [(plan(image='a'*64), 'stale'),(plan(-8),'direction'),(plan(80),'range')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError,message):
                validate_plan(action, observation(), {'brightness':0}, CAPABILITIES)

    async def test_host_binds_frame_and_target(self):
        client=type('Model', (), {'ask':AsyncMock(return_value=provider_observation())})()
        report=await observe(client, 'b'*64, ['frame.png'], {}, target='part-1')
        self.assertEqual(report.image_sha256,'b'*64)
        self.assertEqual(report.findings[0].id,'f1')
        self.assertEqual(report.findings[0].target,'part-1')
        with self.assertRaisesRegex(ValueError,'does not match'):
            report.bind(IMAGE,'part-1')
        forged=provider_observation().model_dump()
        forged['findings'][0]['target']='another-object'
        with self.assertRaisesRegex(ValueError,'extra_forbidden'):
            VisualObservation.model_validate(forged)

    async def test_planner_receives_only_supported_capabilities(self):
        reply=PlanProposal(status='noop',operations=[],note='fixture')
        client=type('Model', (), {'ask':AsyncMock(return_value=reply)})()
        options={**CAPABILITIES, 'noise':{'values':['less'],'causes':['texture']}}
        action, changes=await plan_execution(client, observation(), {'brightness':0,'noise':'auto'}, options, {})
        self.assertEqual(changes,{})
        self.assertEqual(action.image_sha256,IMAGE)
        offered=client.ask.await_args.args[3]['capabilities']
        self.assertEqual(set(offered),{'brightness'})
        self.assertEqual(offered['brightness']['evidence_ids'],['f1'])

    def test_preview_planner_does_not_offer_wrong_dust_or_geometry_operations(self):
        source=provider_observation().model_dump()
        source['findings']=[{**source['findings'][0],'code':code,'scope':scope}
                            for code, scope in (('surface_dust_heavy','image'),('structure','asset'))]
        report=VisualObservation.model_validate(source).for_frame(IMAGE,'object')
        allowed={'surface_dust':['0.25','0.75'],'candidate':['S2']}
        params={'surface_dust':'0.5','candidate':'S1'}
        self.assertEqual(planner_capabilities(params, allowed, report),
                         {'surface_dust':{'values':['0.25'],'causes':['surface_dust_heavy'],'evidence_ids':['f1']}})

    def test_geometry_finding_requires_direct_matching_cue(self):
        payload=provider_observation().model_dump()
        finding=payload['findings'][0]
        finding.update(code='floating',evidence='物品轮廓与钢料之间有可见间隙')
        with self.assertRaisesRegex(ValueError,'direct physical cue'):
            VisualObservation.model_validate(payload)
        finding['physical_cue']='visible_air_gap'
        bound=VisualObservation.model_validate(payload).for_frame(IMAGE,'object')
        self.assertEqual(bound.findings[0].physical_cue,'visible_air_gap')
        finding.update(code='shadow',evidence='接触阴影偏弱')
        with self.assertRaisesRegex(ValueError,'direct physical cue'):
            VisualObservation.model_validate(payload)
        finding['physical_cue']=None
        self.assertEqual(VisualObservation.model_validate(payload).findings[0].code,'shadow')

    def test_config_preserves_explicit_effects_on_new_geometry_path(self):
        config=BatchConfig(backgrounds='x',output='x',workflow='evidence_v1',geometry_preset='rebuild',
            vehicle_reference={'vehicle_class':'semitrailer','internal_width_m':2.45,
                'source':{'kind':'user_supplied','reference':'fixture'}},
            effects={'fog_mode':'local_auto','pool_kernel':3,'pool_weight':1.})
        self.assertEqual(config.effects.pool_kernel,3)
        self.assertEqual(config.contact_mode,'strict')
        self.assertEqual(config.camera_pipeline,'premultiplied')
        self.assertIn('effects',config._resolution_notes)

    def test_auto_optics_applies_haze_once_without_pooling(self):
        clear=np.random.default_rng(0).random((20,20,3),dtype=np.float32)
        alpha=np.ones((20,20),np.float32)
        transmission=np.ones_like(alpha)
        air={'transmission_field':transmission,'color_linear':[0.,0.,0.]}
        env={'camera_psf':{'edge_samples':24},'local_context':{'sample_pixels':500}}
        params={'workflow':'evidence_v1','fog_mode':'local_auto','pool_kernel':'3','pool_weight':1.}
        delta, applied=optical_delta(clear,clear,alpha,env,air,20,params)
        self.assertEqual(applied['pool_weight'],0.)
        self.assertEqual(applied['requested']['pool_kernel'],'3')
        np.testing.assert_array_equal(delta,0.)


if __name__=='__main__':
    unittest.main()
