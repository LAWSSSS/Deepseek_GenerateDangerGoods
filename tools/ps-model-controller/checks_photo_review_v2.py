"""Real historical review contradictions, roles and one-repair transport safety."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import batch_display  # establishes the package path for compatibility entrypoints
import httpx
from PIL import Image
from display_batch.ai import InvalidStructuredReply, Vision
from display_batch.schema import BatchConfig, EffectsReview
from display_batch.strict_contact import reconcile_pose_evidence, visual_veto
from display_batch.structured import _review
from scrap_composite.photography_agent import run_photography, scope_plan
from scrap_composite.photography_protocol import PhotoBrief, PhotoBriefV2, PhotoContext, PhotoReport

SCORES = dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'), 2)


def finding(**overrides):
    return {'factor':'sharpness','region':[.3,.3,.6,.6],'severity':'major','visibility':'visible',
            'confidence':.9,'direction':'increase','cue':'物品轮廓和螺栓边界模糊',
            'reference':'左侧同层废钢边缘','physical_cue':None,'observed_state':'too_low',
            'reference_role':'scene','reference_region':[.1,.3,.25,.6],**overrides}


def brief(f):
    return {'trace':'visible','scores':SCORES,'findings':[f]}


def review(code, evidence, **extra):
    return EffectsReview.model_validate_json(json.dumps({'scores':SCORES,'trace':'visible',
        'defects':[{'code':code,'severity':'major','region':[.3,.3,.6,.6], 'evidence':evidence,**extra}],
        'actions':[],'evidence':''}))


class ReviewV2Tests(unittest.IsolatedAsyncioTestCase):
    def test_historical_opposite_sharpness_and_gray_haze_are_rejected(self):
        for value in (finding(direction='decrease',observed_state='too_high'),
                      finding(factor='haze_amount',direction='increase',observed_state='too_low',
                              cue='全画面灰雾导致暗部抬升、反差低'),
                      finding(factor='upright',direction='unknown',observed_state='mismatch',
                              cue='筒身横向嵌入废钢夹缝，未见端面立起', physical_cue='standing_on_end')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PhotoBriefV2.model_validate_json(json.dumps(brief(value)))

    def test_state_and_direction_conflict_is_not_silently_flipped(self):
        with self.assertRaisesRegex(ValueError,'direction must oppose'):
            PhotoBriefV2.model_validate_json(json.dumps(brief(finding(direction='decrease'))))

    def test_source_role_cannot_supply_photographic_or_overlap_baseline(self):
        for factor, reference in [('haze_amount','参考棚拍无雾'),('material_response','参考件反光明亮'),
                                  ('occlusion','参考图完整六视图，结构无遮挡')]:
            f=finding(factor=factor,direction='unknown',observed_state='mismatch',
                      reference_role='identity',reference_region=None,reference=reference)
            with self.subTest(factor=factor),self.assertRaises(ValueError):
                PhotoBriefV2.model_validate_json(json.dumps(brief(f)))
        with self.assertRaisesRegex(ValueError,'source/studio'):
            PhotoBriefV2.model_validate_json(json.dumps(brief(finding(reference='参考棚拍无雾'))))

    def test_generic_scene_reference_word_is_not_mistaken_for_identity_photo(self):
        for text in ('首图中左侧现场参考图区域','left steel in the first scene reference'):
            PhotoBriefV2.model_validate_json(json.dumps(brief(finding(reference=text))))
        with self.assertRaisesRegex(ValueError,'source/studio'):
            PhotoBriefV2.model_validate_json(json.dumps(brief(finding(reference='source photo left steel'))))

    def test_old_reply_readable_new_reply_binds_metadata_through_legacy(self):
        old=finding()
        for key in ('observed_state','reference_role','reference_region'):old.pop(key)
        PhotoBrief.model_validate_json(json.dumps(brief(old)))
        new=finding(factor='floating',direction='unknown',observed_state='mismatch',
                    cue='底部与承托面之间可见空气缝隙',physical_cue='visible_air_gap')
        report=PhotoReport.bind(PhotoBriefV2.model_validate_json(json.dumps(brief(new))),
                                'a'*64,'object',PhotoContext(images=['frame']))
        decoded=PhotoReport.model_validate_json(report.model_dump_json())
        adapted=_review(decoded.legacy(),{})
        defect=adapted.defects[0]
        self.assertEqual(defect.physical_cue,'visible_air_gap')
        self.assertEqual(defect.reference_role,'scene')
        self.assertEqual(defect.reference_region,(.1,.3,.25,.6))
        self.assertEqual(len(visual_veto(adapted)),1)

    def test_normal_occlusion_stays_unresolved_without_score_promotion(self):
        old=review('occlusion','气缸大部分被链条压盖，仅端盖局部露出。参考图完整六视图，确认拉杆与端盖结构')
        checks={};new=reconcile_pose_evidence(old,checks)
        self.assertEqual(visual_veto(new),[])
        self.assertEqual(new.defects[0].visibility,'unassessable')
        self.assertEqual(new.scores,old.scores)
        self.assertFalse(new.passes())
        self.assertEqual(checks['review_evidence_consistency']['status'],'unknown')

    def test_real_gaps_penetration_and_wrong_scene_overlap_remain_vetoes(self):
        for code,cue in [('floating','筒体与废钢间亮边透空，无接触影'),
                         ('penetration','钢板实体穿透缸筒内壁'),
                         ('occlusion','同一前景钢板在相连边界前后次序反转')]:
            with self.subTest(code=code):self.assertEqual(len(visual_veto(review(code,cue))),1)
        for cue in ['仅接触阴影不足','没有可见空气间隙，仅接触影偏弱']:
            self.assertEqual(visual_veto(review('floating',cue)),[])

    def test_unknown_response_cannot_drive_adjustment(self):
        f=finding(visibility='unassessable',observed_state='unknown',direction='unknown',
                  reference_role='unknown',reference_region=None,reference=None)
        PhotoBriefV2.model_validate_json(json.dumps(brief(f)))
        with self.assertRaises(ValueError):
            PhotoBriefV2.model_validate_json(json.dumps(brief({**f,'visibility':'visible'})))

    def test_cached_v2_cannot_reintroduce_invalid_reference_roles(self):
        value=PhotoReport.bind(PhotoBriefV2.model_validate_json(json.dumps(brief(finding()))),
                              'a'*64,'object',PhotoContext(images=['frame']))
        stored=value.model_dump(mode='json')
        stored['findings'][0]['reference_role']='identity'
        with self.assertRaises(ValueError):
            PhotoReport.model_validate_json(json.dumps(stored))

    def test_legacy_direction_error_remains_readable_but_cannot_drive_plan(self):
        context=PhotoContext(images=['frame'])
        source=PhotoBrief.model_validate_json(json.dumps(brief(finding(
            direction='decrease',observed_state=None,reference_role=None,reference_region=None))))
        report=PhotoReport.bind(source,'a'*64,'object',context)
        self.assertEqual(scope_plan(report,context,()).reason,'uncertain')
        self.assertEqual(report.findings[0].direction,'decrease')

    async def test_two_invalid_replies_stop_after_one_format_repair(self):
        calls=[]
        async def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(brief(
                finding(direction='decrease',observed_state='too_high')))}}],
                'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);png=root/'frame.png';Image.new('RGB',(16,16)).save(png)
            provider=root/'provider.json'
            provider.write_text(json.dumps({'url':'https://unit.test/v1/chat/completions','model':'test'}))
            config=BatchConfig(backgrounds=temp,output=temp,count=1,provider_config=str(provider))
            with patch('display_batch.ai.configure_credential',return_value=True), \
                 patch('display_batch.ai.asyncio.sleep'):
                client=Vision(config,root/'cache',root/'usage.json')
                await client.http.aclose()
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                    client.http=http
                    result = await run_photography(client,PhotoContext(images=['frame']),'a'*64,images=[png])
                    self.assertEqual(result.report.findings[0].visibility, 'unassessable')
                    self.assertEqual(len(result.report.issues), 1)
            self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main(verbosity=2)
