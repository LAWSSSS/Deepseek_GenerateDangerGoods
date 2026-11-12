"""Regressions at the visual report, planner and camera boundaries."""
import unittest
from unittest.mock import AsyncMock

import numpy as np

from display_batch.common import ROOT
from display_batch.effects import optical_delta
from display_batch.schema import BatchConfig
from scrap_composite.contracts import ExecutionPlan, VisualReport, validate_plan
from scrap_composite.workflow import observe, plan_execution


IMAGE = 'f' * 64


def observation():
    return VisualReport(version='visual-evidence-v1', image_sha256=IMAGE, trace='visible',
        scores=dict.fromkeys(('support', 'occlusion', 'lighting', 'edge', 'texture', 'material', 'structure'), 3),
        findings=[{'id':'dim','target':'object','code':'lighting','region':(.1,.2,.3,.4),
                   'severity':'major','visibility':'visible','scope':'image','confidence':.8,
                   'direction':'increase','evidence':'物品局部过暗'}], summary='照明待修正')


def plan(value=8, image=IMAGE):
    return ExecutionPlan(version='execution-plan-v1', image_sha256=image, status='ready', note='局部提亮',
        operations=[{'capability':'brightness','target':'object','value':value,
                     'evidence_ids':['dim'],'expected_effect':'提亮物品暗部'}])


CAPABILITIES = {'brightness': {'range': [-20,20], 'causes':['lighting']}}


class WorkflowChecks(unittest.IsolatedAsyncioTestCase):
    async def test_observe_then_compile_absolute_action(self):
        client=type('Model', (), {'ask':AsyncMock(side_effect=[observation(),plan()])})()
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

    async def test_bind_model_result_to_frame_and_target(self):
        client=type('Model', (), {'ask':AsyncMock(return_value=observation())})()
        with self.assertRaisesRegex(ValueError,'does not match'):
            await observe(client, 'b'*64, ['frame.png'], {})

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
