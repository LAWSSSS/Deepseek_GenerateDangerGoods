"""Offline regressions for contradictory reference-pose reviews and hard vetoes."""
import asyncio
import copy
from pathlib import Path
import unittest

from display_batch.review import inspect
from display_batch.schema import EffectsReview, FusionObservation
from display_batch.strict_contact import reconcile_pose_evidence, visual_veto


BAD_EVIDENCE='罐体横向侧躺，参考图为直立圆柱，姿态与单只罐不符'


def review(evidence=BAD_EVIDENCE,code='upright'):
    return EffectsReview.model_validate({'trace':'visible',
        'scores':dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),4),
        'defects':[{'code':code,'severity':'major','region':[.43,.33,.71,.75],
                    'evidence':evidence,'scope':'image','visibility':'visible'}],
        'actions':[],'evidence':'Review fixture'})


def checks(angle=7.447061446):
    return {'lying':{'status':'pass','measurement_deg':angle,'threshold_deg':15},
            'physics':{'status':'unknown'}}


class PoseReviewTests(unittest.TestCase):
    def test_observed_reference_contradiction_remains_unknown_not_accepted(self):
        original=review();audit=checks()
        resolved=reconcile_pose_evidence(original,audit)
        self.assertEqual(visual_veto(resolved),[])
        self.assertEqual(resolved.defects[0].visibility,'unassessable')
        self.assertEqual(resolved.defects[0].severity,'major')
        self.assertFalse(resolved.passes())
        self.assertEqual(resolved.scores,original.scores)
        self.assertEqual(original.defects[0].visibility,'visible')
        self.assertEqual(audit['review_pose_consistency']['status'],'unknown')
        self.assertEqual(audit['review_pose_consistency']['original_defects'][0]['evidence'],BAD_EVIDENCE)

    def test_current_standing_evidence_still_vetoes(self):
        for text in ('物体竖立，参考图为直立圆柱',
                     'The object is upright; source image is upright.',
                     '物体以端部站立，参考图为直立圆柱'):
            value=reconcile_pose_evidence(review(text),checks())
            self.assertEqual(len(visual_veto(value)),1)
        for text in ('物体并非横躺，参考图为直立圆柱',
                     '物体应该平躺，参考图为直立圆柱',
                     'The object is not lying flat; reference image is upright.'):
            # A recommendation/negative lying statement is not a positive
            # standing-on-end observation. Keep the low-quality review unknown.
            value=reconcile_pose_evidence(review(text),checks())
            self.assertEqual(visual_veto(value),[])
            self.assertFalse(value.passes())

    def test_reference_only_conflict_remains_unknown_independent_of_geometry(self):
        for audit in ({},checks(None),checks(30),checks(float('nan')),
                      {'lying':{'status':'unknown','measurement_deg':7}}):
            value=reconcile_pose_evidence(review(),audit)
            self.assertEqual(visual_veto(value),[])
            self.assertFalse(value.passes())
            self.assertEqual(len(visual_veto(reconcile_pose_evidence(review('物体以端部站立'),audit))),1)

    def test_real_contact_defects_survive_measured_lying(self):
        for code,evidence in [('floating','底部与钢板之间可见空气间隙'),
                              ('penetration','活塞杆穿透钢板实体'),
                              ('occlusion','同一钢板连续边缘前后次序错误')]:
            self.assertEqual(len(visual_veto(reconcile_pose_evidence(review(evidence,code),checks()))),1)
        mixed=review()
        mixed.defects.append(review('底部存在明显悬空间隙','floating').defects[0])
        answer=reconcile_pose_evidence(mixed,checks())
        self.assertEqual([d['code'] for d in visual_veto(answer)],['floating'])
        self.assertFalse(answer.passes())
        # An unsupported upright label does not replace explicit gap evidence;
        # the independently reported floating observation above remains a veto.
        answer=reconcile_pose_evidence(review(BAD_EVIDENCE+'，底部悬空'),checks())
        self.assertFalse(answer.passes())

    def test_english_reference_contradiction(self):
        answer=reconcile_pose_evidence(review('The can is lying on its side but the reference image is upright.'),checks())
        self.assertEqual(answer.defects[0].visibility,'unassessable')

    def test_every_inspection_path_carries_pose_policy_and_reconciles(self):
        class Vision:
            async def ask(self,kind,schema,prompt,payload,images,**kwargs):
                self.call=(kind,prompt,copy.deepcopy(payload))
                value=review().model_dump()
                if schema is FusionObservation:
                    value.pop('actions')
                    value['defects']=[{k:v for k,v in d.items() if k in ('code','severity','evidence')}
                                      for d in value['defects']]
                elif 'trace' not in schema.model_fields:value.pop('trace')
                return schema.model_validate(value)
        for mode,compact,effects in [('reference',False,True),('local',False,False),
                                     ('native',False,False),('reference',True,False),('reference',False,False)]:
            vision=Vision();audit=checks()
            comp={'checks':audit,'image':'unused.png','preview':'unused.jpg','crop':'unused.jpg','photo':{}}
            params={'photo_model':mode,'effects_enabled':effects}
            asset={'name':'can','board':'source.jpg','semantic':{'key_features':[],
                   'material_classes':['plastic'],'major_mm':[100,200]}}
            answer=asyncio.run(inspect(vision,'Full review.',asset,comp,params,{},Path('.'),compact=compact))
            self.assertIn('Reference images show identity',vision.call[1])
            self.assertTrue(vision.call[2]['pose_policy']['reference_pose_is_not_target'])
            self.assertEqual(answer.defects[0].visibility,'unassessable')
            self.assertEqual(audit['review_pose_consistency']['status'],'unknown')


if __name__=='__main__':unittest.main(verbosity=2)
