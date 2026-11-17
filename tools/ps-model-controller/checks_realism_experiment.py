"""Offline publication, persisted-review and cost guards for the A/B/C harness."""
import copy
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import batch_display  # establish source-package path, no workers or API clients
from PIL import Image
from display_batch.common import atomic_json, file_hash, signature
from display_batch.pipeline import Pipeline
from display_batch.schema import BatchConfig
from display_batch.structured import _review
from scrap_composite.photography_protocol import PhotoBriefV2, PhotoContext, PhotoReport
import validate_realism as experiment
from scrap_composite.knowledge import KnowledgeBase, _eligible_case, archive_run


def persisted_review(*, minor=False):
    scores=dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),4)
    findings=[]
    if minor:
        scores['texture']=3
        findings=[{'factor':'sharpness','region':[.3,.3,.6,.6],'severity':'minor','visibility':'visible',
                   'confidence':.9,'direction':'decrease','cue':'目标边缘稍微过锐','reference':'左侧同层废钢边缘',
                   'observed_state':'too_high','reference_role':'scene','reference_region':[.1,.3,.25,.6],
                   'physical_cue':None}]
    brief=PhotoBriefV2.model_validate_json(json.dumps({'trace':'visible' if minor else 'none',
                                                      'scores':scores,'findings':findings}))
    report=PhotoReport.bind(brief,'a'*64,'object',PhotoContext(images=['frame'],target_region=(.3,.3,.6,.6)))
    review=_review(report.legacy(),{})
    # Real result files serialize tuple regions as JSON arrays.
    return json.loads(json.dumps({'photography_diagnosis':report.model_dump(mode='json'),
        'review':review.model_dump(mode='json'),'image_hash':'a'*64,
        'checks':{'size':{'status':'pass'},'physical_contact':{'status':'unknown'}},'deliverable':True}))


def empty_vision(path):
    return SimpleNamespace(usage_path=path,usage=lambda:{'calls':[],'cache_hits':0})


def unreviewed_image(folder):
    folder.mkdir(parents=True,exist_ok=True)
    image=folder/'actual.png';crop=folder/'actual.crop.jpg'
    Image.new('RGB',(16,12),(80,90,95)).save(image)
    Image.new('RGB',(8,6),(80,90,95)).save(crop)
    return {'image':str(image),'crop':str(crop),'image_hash':file_hash(image),
        'iteration':0,'params':{},'measured_major_px':12.,'photo':{},'deliverable':True,
        'checks':{'size':{'status':'pass'},'support':{'status':'unknown'}},
        'review':None,'review_status':'unknown','review_error':'invalid_after_one_repair'}


class RealismExperimentTests(unittest.TestCase):
    def test_finish_image_preserves_actual_unreviewed_png_without_scores_or_acceptance(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);entry=unreviewed_image(root/'draft')
            result={'status':'accepted_display','iterations':[entry]}
            Pipeline.finish_image(root/'output','job-1',result,entry)
            self.assertEqual(result['status'],'review')
            self.assertEqual(result['review_status'],'unknown')
            self.assertIsNone(result['scores'])
            self.assertEqual(result['photo_status'],'unknown')
            self.assertEqual(result['visual_status'],'unknown')
            self.assertTrue(result['unreviewed_draft'])
            self.assertEqual(Path(result['image']).parent.name,'review')
            self.assertEqual(file_hash(result['image']),entry['image_hash'])
            self.assertEqual(file_hash(result['crop']),file_hash(entry['crop']))
            self.assertIsNone(result['iterations'][0]['review'])

    def test_finish_image_rejects_program_failure_before_copy(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);entry=unreviewed_image(root/'draft')
            entry['checks']['visible_gap']={'status':'fail'}
            result={'status':'review'}
            with self.assertRaisesRegex(ValueError,'program_failure_is_not_deliverable'):
                Pipeline.finish_image(root/'output','job-1',result,entry)
            self.assertNotIn('image',result)
            self.assertFalse((root/'output').exists())

    def test_reuse_of_unknown_review_copies_png_without_none_crash_or_api_cost(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';destination=root/'destination'
            entry=unreviewed_image(source/'draft')
            original={'status':'review','seconds':3.,'iterations':[entry]}
            Pipeline.finish_image(source,'old-job',original,entry)
            config=BatchConfig(backgrounds='fixture',output=str(destination),preset='realism_v1')
            endpoint={'model':'fixture'};implementation={'version':'fixture'}
            atomic_json(source/'implementation.json',implementation)
            atomic_json(source/'config.json',config.model_dump())
            atomic_json(source/'vision-signature.json',{'signature':signature(endpoint)})
            payload={'asset_id':'3','background':'fixture'}
            previous=Mock();previous.rows.return_value=[{'id':'old-job','state':'review','payload':payload,'result':original}]
            current=Mock();current.rows.return_value=[{'id':'new-job','state':'queued','payload':payload}]
            pipe=Pipeline.__new__(Pipeline);pipe.config=config;pipe.out=destination;pipe.store=current
            pipe.vision=SimpleNamespace(endpoint=SimpleNamespace(model_dump=lambda:endpoint))
            pipe.dimension_reference_signature=lambda:None
            with patch('display_batch.pipeline.Store',return_value=previous):
                pipe.reuse_completed(source,implementation)
            self.assertEqual(current.finish.call_count,1)
            ident,state,result=current.finish.call_args.args
            self.assertEqual((ident,state),('new-job','review'))
            self.assertIsNone(result['scores']);self.assertIsNone(result['iterations'][0]['review'])
            self.assertEqual(result['new_api_calls'],0)
            self.assertEqual(result['cost']['reported_tokens'],0)
            self.assertEqual(file_hash(result['image']),entry['image_hash'])
            previous.close.assert_called_once()

    def test_knowledge_archive_keeps_null_review_unscored(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);entry=unreviewed_image(root/'draft')
            row={'asset_id':'3','group':'36_6_6','background':'fixture','status':'review',
                 'image':entry['image'],'scores':None,'iterations':[entry]}
            (root/'results.jsonl').write_text(json.dumps(row)+'\n',encoding='utf8')
            archive_run(root,root/'knowledge.sqlite')
            with KnowledgeBase(root/'knowledge.sqlite') as knowledge:
                values=[json.loads(value) for (value,) in knowledge.db.execute('SELECT data FROM cases')]
                self.assertEqual(knowledge.derive_trials()['new_trials'],0)
                self.assertEqual(knowledge.counts()['rules'],0)
            self.assertTrue(values)
            for value in values:
                self.assertFalse(value['assessment']['image_bound'])
                self.assertEqual(value['assessment']['scores'],{})
                self.assertEqual(value['assessment']['source'],'unreviewed')
                self.assertIn('image_bound_quality_assessment_missing',_eligible_case(value))

    def test_json_tuple_roundtrip_keeps_valid_review_and_separate_physical_unknown(self):
        for minor in (False,True):
            with self.subTest(minor=minor):
                item=persisted_review(minor=minor)
                self.assertIsInstance(item['photography_diagnosis']['context']['target_region'],list)
                if minor:self.assertIsInstance(item['photography_diagnosis']['findings'][0]['region'],list)
                states=experiment.evidence_states(item)
                self.assertEqual(states['review_protocol'],'pass')
                self.assertEqual(states['photography_status'],'pass')
                self.assertEqual(states['visual_status'],'fail' if minor else 'pass')
                self.assertEqual(states['program_evidence'],'unknown')
                self.assertTrue(states['deliverable'])

    def test_wrong_image_or_conflicting_review_stays_unknown_without_score_transfer(self):
        for mutate in (lambda v:v.update(image_hash='b'*64),
                       lambda v:v['review']['scores'].update(texture=1),
                       lambda v:v['checks'].update(review_evidence_consistency={'status':'unknown'})):
            item=persisted_review();mutate(item)
            state=experiment.evidence_states(item)
            self.assertEqual(state['photography_status'],'unknown')
            self.assertEqual(state['visual_status'],'unknown')
            self.assertEqual(state['review_protocol'],'unknown')

    def test_unreviewed_checked_draft_is_viewable_but_never_counted(self):
        with TemporaryDirectory() as temporary:
            out=Path(temporary);c=out/'C';c.mkdir()
            image=c/'draft.png';Image.new('RGB',(16,12),(80,90,95)).save(image)
            draft={'image':str(image),'image_hash':file_hash(image),'iteration':0,
                'checks':{'size':{'status':'pass'},'support':{'status':'unknown'}},'deliverable':True}
            atomic_json(c/'draft.json',draft);atomic_json(c/'summary.json',{'status':'review'})
            row={'status':'review','unreviewed_draft':True,'draft_record':str(c/'draft.json'),
                 'reason':'review_invalid_after_one_repair'}
            selected=experiment.proposal_entry(row,c)
            self.assertIsNotNone(selected)
            self.assertIsNone(selected['review']);self.assertIsNone(selected['scores'])
            state=experiment.evidence_states(selected)
            self.assertTrue(state['deliverable']);self.assertEqual(state['photography_status'],'unknown')
            pipe=SimpleNamespace(out=c,vision=empty_vision(out/'usage-live.json'))
            results=[{'key':'36_6_6-3','status':'review','reason':row['reason'],'versions':{'C':selected}}]
            evaluation=experiment.report(out,results,pipe)
            self.assertEqual(evaluation['total'],12)
            self.assertEqual(evaluation['images'],1)
            self.assertEqual(evaluation['unscored_review_images'],1)
            self.assertEqual(evaluation['photography_without_severe_defects'],0)
            self.assertEqual(evaluation['seven_score_visual_passes'],0)
            self.assertFalse(evaluation['target_met'])
            self.assertIn('C/draft.png',(out/'index.html').read_text(encoding='utf8'))

    def test_program_failure_or_physical_veto_cannot_be_rescued_as_draft(self):
        with TemporaryDirectory() as temporary:
            out=Path(temporary);image=out/'draft.png';Image.new('RGB',(8,8)).save(image)
            base={'image':str(image),'image_hash':file_hash(image),'iteration':0,
                  'checks':{'size':{'status':'pass'}},'deliverable':True}
            for key,value in [('checks',{'size':{'status':'pass'},'gap':{'status':'fail'}}),
                              ('physical_visual_veto',[{'code':'floating'}]),('deliverable',False)]:
                with self.subTest(key=key):
                    draft={**base,key:value};atomic_json(out/'draft.json',draft)
                    row={'status':'review','unreviewed_draft':True,'draft_record':str(out/'draft.json')}
                    self.assertIsNone(experiment.proposal_entry(row,out))
                    self.assertFalse(experiment.evidence_states(draft)['deliverable'])
            atomic_json(out/'draft.json',base)
            self.assertIsNone(experiment.proposal_entry({'status':'failed','unreviewed_draft':True,
                'draft_record':str(out/'draft.json')},out))

    def test_blind_winner_relative_contradictions_are_rejected_for_every_order(self):
        for order in itertools.permutations(('A','B','C')):
            for winner,relative in [('A','better'),('C','worse')]:
                answer=experiment.BlindComparison(best='XYZ'[order.index(winner)],relative_to_baseline=relative,
                    material=3,lighting=3,photography=3,evidence='目标局部对照')
                with self.subTest(order=order,winner=winner),self.assertRaisesRegex(ValueError,'disagree'):
                    experiment.validate_blind(answer,order)
            answer=experiment.BlindComparison(best='XYZ'[order.index('B')],relative_to_baseline='worse',
                material=2,lighting=2,photography=2,evidence='B最佳，C低于A')
            experiment.validate_blind(answer,order)

    def test_cost_merges_all_stages_once_and_live_usage_replaces_disk(self):
        rates={'currency':'CNY','low':[1.,2.,3.],'high':[1.,2.,3.],'source':'fixture'}
        def ledger(count,hits):
            return {'calls':[{'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,
                'prompt_cache_hit_tokens':0}} for _ in range(count)],'tariff':rates,'cache_hits':hits}
        with TemporaryDirectory() as temporary:
            out=Path(temporary)/'experiment';c=out/'C';diagnostics=Path(temporary)/'diagnostics'
            current=out/'comparison-usage-live.json'
            atomic_json(c/'usage-C.json',ledger(2,2))
            atomic_json(diagnostics/'usage-assets.json',ledger(1,1))
            atomic_json(current,ledger(1,99))  # stale disk copy must be replaced
            live=ledger(3,7);vision=SimpleNamespace(usage_path=current,usage=lambda:copy.deepcopy(live))
            result=experiment.experiment_cost(out,vision,diagnostics)
            self.assertEqual(result['actual_requests'],6)
            self.assertEqual(result['reported_tokens'],720)
            self.assertEqual(result['cache_hits'],10)
            self.assertEqual(len(result['ledger_files']),3)
            self.assertAlmostEqual(result['estimated_api_min'],.00156)
            self.assertEqual(result['estimated_api_min'],result['estimated_api_max'])
            self.assertTrue(result['complete'])
            # The same directory may be supplied twice; absolute ledger paths
            # remain unique and the live copy still wins.
            repeated=experiment.experiment_cost(out,vision,c)
            self.assertEqual(repeated['actual_requests'],5)
            self.assertEqual(len(repeated['ledger_files']),2)


if __name__=='__main__':unittest.main(verbosity=2)
