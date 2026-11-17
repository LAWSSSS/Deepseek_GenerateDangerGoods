"""Offline checks for user exclusions, half-size contact anchors, dust, and billing."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
import trimesh
from display_batch.costs import tariff,estimate,cumulative
from display_batch.ai import Vision
from display_batch.schema import BatchConfig
from display_batch.pipeline import make_jobs
from display_batch.policy import resize_candidate
from display_batch.scene import projection_geometry,fitted_masks
from display_batch.local import neighbourhood,local_veil
from display_batch.photography import match_linear
from checks_display_local import LocalTests


class FeedbackTests(unittest.TestCase):
    def test_defaults_exclude_all_tanks_even_in_explicit_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'image.png';image.write_bytes(b'test')
            catalog={str(k):{'mesh_hash':'x'} for k in (3,5,6,7,31)}
            cfg=BatchConfig(backgrounds=tmp,output=tmp,count=30)
            self.assertEqual(cfg.appearance,'reference')
            self.assertEqual(cfg.photo_style,'scrap_gray')
            self.assertEqual(BatchConfig(backgrounds=tmp,output=tmp,photo_style='dusty_scrap').appearance,'local')
            jobs=make_jobs(cfg,catalog)
            self.assertEqual({j['asset_id'] for j in jobs},{'3','31'})
            self.assertEqual(jobs,make_jobs(cfg,catalog))
            manifest=root/'manifest.json';manifest.write_text(json.dumps([
                {'asset_id':k,'background':str(image)} for k in ('05','6','7','31')]))
            cfg.manifest=str(manifest)
            self.assertEqual([j['asset_id'] for j in make_jobs(cfg,catalog)],['31'])
            cfg.manifest=None;cfg.assets=['05','6','7']
            with self.assertRaisesRegex(ValueError,'移出'):make_jobs(cfg,catalog)

    def test_halving_linear_extent_keeps_contact_anchor_and_rebuilds_receiver(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh=trimesh.creation.box(extents=[1,.3,.2]);mesh.vertices[:,2]+=.1
            path=Path(tmp)/'geometry.npz';np.savez(path,vertices=mesh.vertices,faces=mesh.faces)
            asset={'geometry':str(path)};pose={'matrix':np.eye(4).tolist()}
            geom=projection_geometry(mesh.vertices,mesh.faces,pose,0,45)
            scene={'size':[600,400],'supports':[{'id':'S1','mask':np.ones((400,600),bool)}]}
            full=400;anchor=[300,230];fit=fitted_masks(geom,anchor,full,scene)
            c={**pose,'camera':geom['camera'],'elevation':45,'yaw':0,'full_px':full,'anchor_px':anchor,
                'expected_major_px':float(np.ptp(geom['uv'],axis=0).max()*full),'support_ids':['S1'],
                'bounds':fit['bounds'],'offset':fit['offset'],'terrain_physics':{'status':'pass'}}
            with patch('display_batch.policy.receiver_mesh',return_value={'signature':'new'}) as rebuilt:
                new=resize_candidate(asset,scene,c,.5)
            self.assertEqual(new['anchor_px'],anchor)
            self.assertAlmostEqual(new['expected_major_px']/c['expected_major_px'],.5)
            self.assertNotIn('terrain_physics',new);rebuilt.assert_called_once()
            self.assertEqual(new['scale']['status'],'unknown')
            self.assertEqual(resize_candidate(asset,scene,new,.5),new)
            self.assertEqual(c['full_px'],400)
            with patch('display_batch.policy.receiver_mesh',return_value={'signature':'quarter'}):
                smaller=resize_candidate(asset,scene,new,.375)
            self.assertEqual(smaller['user_scale']['previous_factor'],.5)
            self.assertEqual(smaller['user_scale']['relative_factor'],.75)
            self.assertAlmostEqual(smaller['expected_major_px']/c['expected_major_px'],.375)
            self.assertEqual(smaller['anchor_px'],anchor)
            self.assertEqual(resize_candidate(asset,scene,smaller,.375),smaller)

    def test_raised_local_dust_not_global_and_clear_region_stays_clear(self):
        scene,c=LocalTests().fixture();env,_=neighbourhood(scene,c)
        rgb=np.full((20,20,3),.1,np.float32)
        _,old=local_veil(rgb,env,{'photo_style':'scrap_gray'})
        _,new=local_veil(rgb,env,{'photo_style':'dusty_scrap'})
        self.assertGreater(new['amount'],old['amount'])
        scene['frame'][:]=np.tile(np.linspace(2,200,440,dtype=np.uint8)[None,:,None],(300,1,3))
        c['bounds']=[20,120,100,210]
        clear,_=neighbourhood(scene,c)
        self.assertLess(clear['airborne_dust']['amount'],.02)
        unchanged,off=local_veil(rgb,env,{'photo_style':'dusty_scrap','atmosphere':'off'})
        np.testing.assert_allclose(unchanged,rgb,atol=1e-6);self.assertEqual(off['amount'],0.)

    def test_stronger_desaturation_retains_highlight_and_alpha(self):
        alpha=np.ones((60,60),np.float32);alpha[20:30,20:30]=0;before=alpha.copy()
        raw=np.full((60,60,3),[.7,.025,.012],np.float32);raw[40:45,5:50]=2.
        pixels=np.full((200,3),[130,126,123],np.float32)
        env={'local_context':{'version':'test'},'sensor':{'surface_detail_rms':.008},'material_classes':['paint','steel']}
        old,oi=match_linear(raw,alpha,pixels,np.zeros_like(alpha),{'photo_style':'scrap_gray'},4,env,reference=True,camera_capture=False)
        new,ni=match_linear(raw,alpha,pixels,np.zeros_like(alpha),{'photo_style':'dusty_scrap'},4,env,reference=True,camera_capture=False)
        self.assertLess(ni['after_chroma'],oi['after_chroma']*.8)
        self.assertGreater(float(new[42,15].mean()-new[10,15].mean()),.1)
        np.testing.assert_array_equal(alpha,before)

    def test_cost_counts_retries_and_cached_tokens_resume_without_double_counting(self):
        cfg=BatchConfig(backgrounds='x',output='y')
        rates=tariff(cfg,SimpleNamespace(url='https://api.deepseek.com/chat/completions',model='deepseek-flash'))
        call={'usage':{'prompt_tokens':1000,'completion_tokens':200,'total_tokens':1200,'prompt_cache_hit_tokens':600}}
        cost=estimate([call,call,{'state':'not_sent'}],rates)
        self.assertEqual(cost['actual_requests'],2);self.assertAlmostEqual(cost['estimated_api_min'],.002424)
        with tempfile.TemporaryDirectory() as tmp:
            doc={'ledger_name':'usage-one.json','calls':[call,call],'reported_tokens':2400,'cost':cost}
            (Path(tmp)/doc['ledger_name']).write_text(json.dumps(doc))
            self.assertEqual(cumulative(tmp,doc)['actual_requests'],2)
            empty={'ledger_name':'usage-two.json','calls':[],'reported_tokens':0,'cost':estimate([],rates)}
            self.assertEqual(cumulative(tmp,empty)['estimated_api_max'],cost['estimated_api_max'])
        self.assertIsNone(estimate([{'usage':{}}],rates)['estimated_api_max'])
        self.assertEqual(estimate([call],None)['reported_tokens'],1200)
        self.assertIsNone(estimate([call],None)['estimated_api_max'])
        self.assertIsNone(tariff(cfg,SimpleNamespace(url='https://another.invalid',model='deepseek-flash')))

    def test_custom_cached_price_has_one_consistent_total(self):
        cfg=BatchConfig(backgrounds='x',output='y',price_input_per_million=2,
            price_cached_input_per_million=.2,price_output_per_million=4)
        vision=Vision.__new__(Vision);vision.config=cfg
        vision.endpoint=SimpleNamespace(url='https://another.invalid',model='test')
        vision.usage_path=Path('usage-test.json');vision.cache_hits=3
        vision.calls=[{'usage':{'prompt_tokens':1000,'completion_tokens':100,'total_tokens':1100,'prompt_cache_hit_tokens':500}}]
        result=vision.usage()
        self.assertAlmostEqual(result['reported_cost'],.0015)
        self.assertEqual(result['reported_cost'],result['cost']['estimated_api_max'])
        self.assertEqual(result['cost']['actual_requests'],1)


if __name__=='__main__':unittest.main(verbosity=2)
