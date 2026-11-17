"""Causal diagnosis and measured-state handoff regressions, with no live API."""
import json
import unittest
from unittest.mock import AsyncMock

from scrap_composite.photography_protocol import (
    PhotoBriefV3, PhotoChoice, PhotoContext, PhotoReport, ControlChange, audit_brief,
    merge_semantic_repair, near_identical_comparison,
)
from scrap_composite.photography_agent import (
    run_photography, scope_plan, compile_plan, detail_policy, composed_prompt, observation_payload,
)


def finding(**overrides):
    return {'factor':'exposure', 'region':[.3,.3,.6,.6], 'severity':'major', 'visibility':'visible',
            'confidence':.9, 'direction':'increase', 'cue':'筒身非高光部分中间调偏暗',
            'reference':'首图左侧同层中性表面', 'observed_state':'too_low',
            'reference_role':'scene', 'reference_region':[.1,.3,.25,.6], 'physical_cue':None,
            'hypothesis':'局部照明不足，仍需排除材质反射率差异',
            'verification':'读取同层区域和目标中间调', **overrides}


def brief(*items):
    return PhotoBriefV3.model_validate_json(json.dumps({'trace':'visible' if items else 'none',
        'scores':dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),4),
        'findings':items}))


def change(**overrides):
    return ControlChange(factor='specular_width', direction='decrease', control='roughness_shift',
        current=0., value=-.04, effect='减小粗糙度偏移并验证反光宽度', stage='render',
        backend='blender', unit='roughness', cause='surface_roughness', **overrides)


class PhotographyV3Tests(unittest.IsolatedAsyncioTestCase):
    def comparison_context(self, **overrides):
        measured={'version':'pixel-comparison-v1','candidate_image_id':1,'reference_image_id':2,
            'candidate_sha256':'a'*64,'reference_sha256':'b'*64,'same_geometry':True,'same_lighting':True,
            'alpha_identical':True,'compared_entire_object':True,'max_abs_rgba_255':1.,
            'changed_pixel_fraction':45/1048576,'core_mae_255':.0000376423,**overrides}
        return PhotoContext(images=['frame','source_pbr'],diagnosis_mode='asset',
            execution_state={'input_image_hashes':['a'*64,'b'*64]},
            measurement_evidence={'comparisons':[measured]})

    def test_near_identical_comparison_only_invalidates_claimed_change(self):
        context=self.comparison_context()
        absolute=finding(reference_role='diagnostic',cue='筒体整体仍有塑料感')
        relative=finding(reference_role='diagnostic',cue='新版比图2的源材质更像塑料',
            reference_image_id=2,change_vs_reference='regressed')
        report=PhotoReport.bind(brief(absolute,relative),'a'*64,'object',context)
        self.assertEqual(report.findings[0].visibility,'visible')
        self.assertEqual(report.findings[1].visibility,'unassessable')
        self.assertEqual(report.raw_findings[1].cue,relative['cue'])
        self.assertEqual(report.raw_findings[1].change_vs_reference,'regressed')
        self.assertIn('near_identical_full_rgba',report.issues[0].reason)
        self.assertEqual(report.scores.lighting,2)
        self.assertEqual(PhotoReport.model_validate_json(report.model_dump_json()),report)

    def test_comparative_guard_requires_full_valid_hash_bound_measurement(self):
        for overrides in ({'reference_sha256':'c'*64},{'candidate_sha256':'c'*64},
                {'same_lighting':False},{'same_geometry':False},{'alpha_identical':False},
                {'compared_entire_object':False},{'max_abs_rgba_255':2.},
                {'changed_pixel_fraction':.1},{'core_mae_255':1.}):
            with self.subTest(overrides=overrides):
                context=self.comparison_context(**overrides)
                self.assertIsNone(near_identical_comparison(context,2))
                report=PhotoReport.bind(brief(finding(reference_role='diagnostic',reference_image_id=2,
                    change_vs_reference='different')),'a'*64,'object',context)
                self.assertEqual(report.findings[0].visibility,'visible')
        missing=self.comparison_context().model_copy(update={'execution_state':{}})
        self.assertIsNone(near_identical_comparison(missing,2))

    def test_same_pixels_never_promote_absolute_quality(self):
        context=self.comparison_context()
        report=PhotoReport.bind(brief(finding(reference_role='diagnostic',reference_image_id=2,
            change_vs_reference='same',cue='两版都缺少加工反射')),'a'*64,'object',context)
        self.assertEqual(report.findings[0].visibility,'visible')
        self.assertEqual(report.scores.lighting,2)
        self.assertFalse(report.issues)

    def test_invalid_finding_isolated_without_erasing_valid_fact(self):
        context=PhotoContext(images=['frame'])
        original=brief(finding(), finding(factor='sharpness', direction='decrease',
            observed_state='too_high', cue='边缘明显模糊'))
        report=PhotoReport.bind(original,'a'*64,'object',context)
        self.assertEqual(report.findings[0].visibility,'visible')
        self.assertEqual(report.findings[1].visibility,'unassessable')
        self.assertEqual(report.issues[0].evidence_id,'f2')
        self.assertEqual(report.scores.texture,2)
        self.assertEqual(scope_plan(report,context,()).reason,'unresolved_evidence')
        self.assertEqual(PhotoReport.model_validate_json(report.model_dump_json()),report)

    async def test_targeted_retry_once_preserves_valid_and_dropped_severe_findings(self):
        context=PhotoContext(images=['frame'])
        original=brief(finding(),finding(factor='sharpness', direction='decrease',
                                        observed_state='too_high',cue='边缘明显模糊'))
        replacement=brief(finding(cue='企图修改无问题条目'))
        client=type('Client',(),{'ask':AsyncMock(side_effect=[original,replacement])})()
        result=await run_photography(client,context,'a'*64,images=['fixture.png'],plan=True)
        self.assertEqual(client.ask.await_count,2)
        self.assertEqual(result.report.findings[0].cue,original.findings[0].cue)
        self.assertEqual(len(result.report.findings),2)
        self.assertEqual(result.report.findings[1].visibility,'unassessable')
        self.assertEqual(result.plan.reason,'unresolved_evidence')
        args=client.ask.await_args
        self.assertIn('semantic_repair',args.args[3])
        self.assertEqual(args.kwargs['max_attempts'],1)

    def test_repair_cannot_downgrade_unresolved_major_finding(self):
        context=PhotoContext(images=['frame'])
        original=brief(finding(factor='sharpness',direction='decrease',observed_state='too_high',
                               cue='边缘明显模糊'),finding())
        for replacement in (
                finding(factor='sharpness',direction='unknown',observed_state='unknown',
                        visibility='unassessable',severity='minor'),
                finding(factor='sharpness',direction='decrease',observed_state='too_high',
                        cue='边缘明显模糊',severity='minor')):
            with self.subTest(replacement=replacement):
                repaired=merge_semantic_repair(original,brief(replacement,finding()),context)
                self.assertEqual(repaired.findings[0].severity,'major')
                report=PhotoReport.bind(repaired,'a'*64,'object',context)
                exposure=ControlChange(factor='exposure',direction='increase',control='brightness',
                    current=0,value=6,effect='抬高中间调',stage='photoshop')
                scope=scope_plan(report,context,[exposure])
                self.assertEqual(scope.reason,'unresolved_evidence')
                self.assertFalse(scope.options)

    def test_scope_revalidates_in_place_state_and_other_report_mutations(self):
        context=PhotoContext(images=['frame'],execution_state={'image_sha256':'a'*64,'document_id':7})
        report=PhotoReport.bind(brief(finding()),'a'*64,'object',context)
        # Both arguments deliberately share the same mutable context object.
        report.context.execution_state['document_id']=8
        self.assertIs(report.context,context)
        with self.assertRaisesRegex(ValueError,'hash changed'):
            scope_plan(report,context,[])
        clean=PhotoContext(images=['frame'],execution_state={'image_sha256':'a'*64})
        report=PhotoReport.bind(brief(finding()),'a'*64,'object',clean)
        report.image_sha256='b'*64
        with self.assertRaisesRegex(ValueError,'another image'):
            scope_plan(report,clean,[])
        report=PhotoReport.bind(brief(finding()),'a'*64,'object',clean)
        report.findings[0].direction='decrease'
        with self.assertRaisesRegex(ValueError,'invalid evidence'):
            scope_plan(report,clean,[])
        context=PhotoContext(images=['frame'],execution_state={'input_image_hashes':['a'*64]})
        report=PhotoReport.bind(brief(finding()),'a'*64,'object',context)
        report.image_sha256='b'*64
        with self.assertRaisesRegex(ValueError,'another image'):
            scope_plan(report,context,[])
        report=PhotoReport.bind(brief(finding(direction='decrease')),'a'*64,'object',context)
        report.findings[0].severity='minor'
        with self.assertRaisesRegex(ValueError,'severity was downgraded'):
            scope_plan(report,context,[])

    async def test_material_cause_selects_roughness_not_light_width(self):
        context=PhotoContext(images=['frame','relight'],diagnosis_mode='asset',material=['steel','paint'],
            measurement_evidence={'material_readback':{'whole':{'roughness':.5}}})
        evidence=brief(finding(factor='specular_width',direction='decrease',observed_state='too_high',
            reference_role='diagnostic',backend='blender',root_cause='surface_roughness',
            measurement_refs=['material_readback'],
            reference='图2同一宽面灯下目标反射，排除了灯面积变化'))
        rough=change()
        light=rough.model_copy(update={'control':'light_width_scale','cause':'light_size','unit':'ratio'})
        report=PhotoReport.bind(evidence,'a'*64,'object',context)
        client=type('Client',(),{'ask':AsyncMock(return_value=PhotoChoice(option='a1',reason='match'))})()
        decision=await run_photography(client,context,'a'*64,report=report,plan=True,
            changes=[light,rough],parameters={'roughness_shift':0.,'light_width_scale':0.})
        self.assertEqual(decision.updates,{'roughness_shift':-.04})
        self.assertEqual(len(client.ask.await_args.args[3]['options']),1)
        self.assertEqual(client.ask.await_args.args[3]['options'][0]['cause'],'surface_roughness')
        self.assertEqual(client.ask.await_args.kwargs['max_output_tokens'],2048)

    def test_ambiguous_reflection_requests_measurement_and_never_uses_dust(self):
        context=PhotoContext(images=['frame','relight'],diagnosis_mode='asset')
        report=PhotoReport.bind(brief(finding(factor='specular_width',reference_role='diagnostic',
            root_cause='unknown')),'a'*64,'object',context)
        plan=scope_plan(report,context,[change()]).bind()
        self.assertEqual(plan.status,'needs_measurement')
        self.assertEqual(plan.needed_measurements,['material_and_light_isolation'])
        self.assertIsNone(plan.operation)

    def test_claimed_changed_protected_material_needs_actual_readback(self):
        context=PhotoContext(images=['frame','source_pbr'],diagnosis_mode='asset',
            asset_profile={'parts':[{'id':'bearing','protect':['source material'],'allow':[]}]})
        f=finding(factor='specular_width',direction='decrease',observed_state='too_high',part_id='bearing',
            reference_role='diagnostic',backend='blender',root_cause='surface_roughness',
            cue='轴承圈比源材质失去反射',hypothesis='金属度可能被降低，粗糙度可能被提高')
        report=PhotoReport.bind(brief(f),'a'*64,'object',context)
        plan=scope_plan(report,context,[change(part_id='bearing')]).bind()
        self.assertEqual(plan.status,'needs_measurement')
        self.assertEqual(plan.needed_measurements,['material_readback'])
        self.assertIsNone(plan.operation)

    def test_no_part_to_whole_fallback(self):
        context=PhotoContext(images=['frame'],asset_profile={'parts':[{'id':'rod'}]})
        report=PhotoReport.bind(brief(finding(part_id='rod')),'a'*64,'object',context)
        operation=ControlChange(factor='exposure',direction='increase',control='brightness',current=0,
            value=6,effect='曲线中间调提高6码值',stage='photoshop',unit='curve_code_value')
        self.assertEqual(scope_plan(report,context,[operation]).reason,'no_supported_control')
        exact=operation.model_copy(update={'part_id':'rod'})
        self.assertEqual(scope_plan(report,context,[exact]).options[0].part_id,'rod')

    def test_fabricated_part_and_measurement_are_unknown(self):
        context=PhotoContext(images=['frame'])
        for fields in ({'part_id':'invisible_rod'},{'measurement_refs':['imaginary_roughness']}):
            report=PhotoReport.bind(brief(finding(**fields)),'a'*64,'object',context)
            self.assertTrue(report.issues)
            self.assertEqual(report.findings[0].visibility,'unassessable')

    def test_known_measurement_and_bound_photoshop_state(self):
        context=PhotoContext(images=['frame'],measurement_evidence={'local_sample':{'pixels':100}},
            execution_state={'image_sha256':'a'*64,'document_id':7,'layer_id':12,'curve_points':[[0,0],[128,136],[255,255]]})
        report=PhotoReport.bind(brief(finding(measurement_refs=['local_sample'])),'a'*64,'object',context)
        self.assertFalse(report.issues)
        with self.assertRaisesRegex(ValueError,'another image'):
            PhotoReport.bind(brief(finding()),'b'*64,'object',context)
        stale=report.model_dump(mode='json')
        stale['context']['execution_state']['layer_id']=13
        with self.assertRaisesRegex(ValueError,'hash changed'):
            PhotoReport.model_validate_json(json.dumps(stale))

    def test_asset_reference_does_not_establish_scene_floating(self):
        context=PhotoContext(images=['frame','gray'],diagnosis_mode='asset')
        report=PhotoReport.bind(brief(finding(factor='floating',observed_state='mismatch',direction='unknown',
            cue='底部存在空气间隙',physical_cue='visible_air_gap',reference_role='diagnostic')),'a'*64,'object',context)
        self.assertEqual(report.findings[0].visibility,'unassessable')

    def test_adaptive_detail_keeps_long_evidence_and_relevant_knowledge(self):
        simple=PhotoContext(images=['frame'])
        asset=PhotoContext(images=['frame','source_pbr'],diagnosis_mode='asset',first_asset_diagnosis=True,
            question='诊断：'+'保留细节。'*80)
        self.assertEqual(detail_policy(simple)['visual_tokens'],3072)
        self.assertEqual(detail_policy(asset)['visual_tokens'],6144)
        prompt=composed_prompt('observer',asset)
        self.assertIn('光源角尺寸',prompt)
        self.assertNotIn('【现场摄影与合成专业检查】',prompt)
        payload=observation_payload(asset,'a'*64,'cylinder')
        self.assertEqual(payload['question'],asset.question)
        self.assertEqual(payload['reference_roles'][1]['role'],'diagnostic')

    def test_long_evidence_legacy_adapter_points_to_complete_record(self):
        context=PhotoContext(images=['frame'])
        report=PhotoReport.bind(brief(finding(cue='筒身偏暗，'+'局部仍有层次。'*20,
            reference='左侧同层表面，'+'应排除高光。'*20)),'a'*64,'object',context)
        legacy=report.legacy()
        self.assertLessEqual(len(legacy.findings[0].evidence),240)
        self.assertIn('完整事实与参照',legacy.findings[0].evidence)
        self.assertGreater(len(report.findings[0].cue)+len(report.findings[0].reference),240)


if __name__=='__main__':
    unittest.main()
