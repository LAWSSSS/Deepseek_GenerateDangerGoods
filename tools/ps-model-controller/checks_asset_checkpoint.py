"""Offline report safeguards: cost deduplication, pixel evidence, flat archives."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from PIL import Image
from report_asset_checkpoint import collect_reviews,flat_zip,pixel_comparison


class AssetCheckpointTests(unittest.TestCase):
    def test_failed_review_and_repeated_directory_preserve_cost_once(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp)/'22';folder.mkdir()
            for name,data in {
                'manifest.json':{'context':{'asset_profile':{'asset_id':'22'}}},
                'diagnosis.json':{'signature':'logical','status':'failed','diagnosis':None},
                'usage.json':{'api_calls':2,'calls':[{'attempt':0,'usage':{'total_tokens':10}},
                                                  {'attempt':1,'usage':{'total_tokens':15}}]},
            }.items():(folder/name).write_text(json.dumps(data),encoding='utf8')
            reviews,usage=collect_reviews([Path(temp),Path(temp)])
            self.assertEqual((usage['actual_requests'],usage['logical_requests'],usage['retries'],usage['reported_tokens']),(2,1,1,25))
            self.assertEqual(usage['duplicate_events_excluded'],2)
            self.assertIsNone(reviews[0]['scores'])

    def test_conflicting_usage_for_same_attempt_rejected(self):
        with TemporaryDirectory() as temp:
            for i,tokens in enumerate((10,11)):
                folder=Path(temp)/str(i);folder.mkdir()
                (folder/'manifest.json').write_text('{}')
                (folder/'diagnosis.json').write_text('{"signature":"same"}')
                (folder/'usage.json').write_text(json.dumps({'run_id':'same-real-run','calls':[{'attempt':0,'usage':{'total_tokens':tokens}}]}))
            with self.assertRaisesRegex(ValueError,'conflicting'):collect_reviews([Path(temp)])

    def test_same_input_in_different_directories_is_not_an_execution_id(self):
        with TemporaryDirectory() as temp:
            for i in range(2):
                folder=Path(temp)/str(i);folder.mkdir()
                (folder/'manifest.json').write_text('{}')
                (folder/'diagnosis.json').write_text('{"signature":"same-input"}')
                (folder/'usage.json').write_text(json.dumps({'calls':[{'attempt':0,'usage':{'total_tokens':10}}]}))
            rows,usage=collect_reviews([Path(temp),Path(temp)])
            self.assertEqual(usage['actual_requests'],2)
            self.assertEqual(usage['logical_requests'],2)
            self.assertEqual(usage['reported_tokens'],20)
            self.assertEqual(len(usage['provenance_ambiguities']),1)
            self.assertTrue(all(row['input_signature']=='same-input' for row in rows))

    def test_provider_request_ids_can_identify_copied_events(self):
        with TemporaryDirectory() as temp:
            for i in range(2):
                folder=Path(temp)/str(i);folder.mkdir()
                (folder/'manifest.json').write_text('{"endpoint":"https://provider.example","model":"model"}')
                (folder/'diagnosis.json').write_text('{"signature":"same-input"}')
                (folder/'usage.json').write_text(json.dumps({'calls':[{'attempt':0,'provider_request_id':'req-123','usage':{'total_tokens':10}}]}))
            _,usage=collect_reviews([Path(temp)])
            self.assertEqual(usage['actual_requests'],1)
            self.assertEqual(usage['logical_requests'],1)
            self.assertEqual(usage['reported_tokens'],10)
            self.assertEqual(usage['duplicate_events_excluded'],1)

    def test_pixel_record_is_bound_and_does_not_claim_quality(self):
        with TemporaryDirectory() as temp:
            a=Path(temp)/'source.png';b=Path(temp)/'current.png'
            image=Image.new('RGBA',(20,20),(100,90,80,255));image.save(a)
            image.putpixel((3,3),(101,90,80,255));image.save(b)
            result=pixel_comparison(a,b)
            self.assertEqual(result['changed_pixels_all'],1)
            self.assertEqual(result['comparison_evidence']['max_abs_rgba_255'],1)
            self.assertEqual(result['alpha_iou'],1)
            self.assertIn('not_absolute_realism_quality',result['interpretation'])
            self.assertNotEqual(result['comparison_evidence']['candidate_sha256'],result['comparison_evidence']['reference_sha256'])

    def test_archive_is_flat_and_never_overwrites(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp);image=folder/'input.png';Image.new('RGB',(16,16)).save(image)
            first=flat_zip([('3-source.png',image)],folder,'images.zip')
            original=first.read_bytes()
            second=flat_zip([('3-source.png',image)],folder,'images.zip')
            self.assertNotEqual(first,second)
            self.assertEqual(first.read_bytes(),original)
            with zipfile.ZipFile(second) as archive:self.assertEqual(archive.namelist(),['3-source.png'])


if __name__=='__main__':unittest.main(verbosity=2)
