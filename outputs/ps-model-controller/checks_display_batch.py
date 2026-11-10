"""Offline regression suite. Model/queue tests use explicit fakes, not real acceptance."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import httpx
import sys

from display_batch.schema import (BatchConfig, Review, Scores, Action, Defect, Probe,
                                  apply_actions, version_rank)
from display_batch.store import Store
from display_batch.geometry import (camera_for, project, contact_geometry, contact_support,
                                     horizontal_angle, scale_from_reference,contact_world,align_up,material_detail_allowed)
from display_batch.compose import match_rgb, linear, srgb, warp
from display_batch.ai import Vision, ProviderUnavailable
from display_batch.pipeline import Pipeline, batch_lock, final_resolution
from display_batch.scene import support_anchors
from display_batch.common import atomic_json,file_hash
from display_batch.assets import prepare as prepare_asset
from display_batch.schema import AssetAnswer
from display_batch import web as batch_web
sys.path.insert(0,str(Path(__file__).parent/'placement-web'))
from server import app
from starlette.testclient import TestClient


def review(value=3,actions=(),defects=()):
    scores=dict.fromkeys(('support','occlusion','lighting','edge','texture','material','structure'),value)
    if value==3:scores['lighting']=4
    return Review(scores=Scores(**scores),actions=list(actions),defects=list(defects),evidence='测试')


class GeometryTests(unittest.TestCase):
    def test_inclined_contact_preserves_disconnected_patches(self):
        points=np.array([[0,0,0],[.2,0,0],[0,.2,0],[1,0,0],[1.2,0,0],[1,.2,0]],float)
        faces=np.array([[0,1,2],[3,4,5]])
        normal=np.array([.2,0,np.sqrt(.96)])
        rotation=align_up(normal)
        tilted=points@rotation[:3,:3].T
        patches=contact_world(tilted,faces,normal=normal)
        self.assertEqual(len(patches),2)
        for patch in patches:np.testing.assert_allclose(patch@normal,0,atol=1e-10)
        self.assertGreater(np.linalg.norm(patches[0].mean(0)-patches[1].mean(0)),.9)

    def test_mixed_atlas_not_mistaken_for_pure_rubber(self):
        self.assertFalse(material_detail_allowed('black rubber seal'))
        self.assertFalse(material_detail_allowed('whiteplastic',True))
        self.assertTrue(material_detail_allowed('yellow paint, chrome, steel, bronze and rubber PBR atlas',True))

    def test_anchor_search_stays_inside_support(self):
        mask=np.zeros((200,400),bool);mask[20:180,70:300]=True
        anchors=support_anchors({'mask':mask,'anchor_px':[80,40]})
        self.assertGreaterEqual(len(anchors),2)
        self.assertTrue(all(mask[y,x] for x,y in anchors))

    def test_final_sampling_accounts_for_render_frustum(self):
        self.assertEqual(final_resolution({'target_px':200,'full_px':470}),1024)
        self.assertEqual(final_resolution({'target_px':400,'full_px':880}),2048)
        self.assertEqual(final_resolution({'target_px':2000,'full_px':4400}),2048)

    def test_separated_feet_do_not_create_support_in_gap(self):
        foot=np.zeros((80,100),np.uint8);foot[30:40,10:20]=255;foot[30:40,70:80]=255
        support=np.zeros_like(foot,bool);support[:,30:60]=True
        self.assertEqual(contact_support(foot,support),[0.,0.])
        support[30:40,10:20]=True
        self.assertEqual(contact_support(foot,support),[1.,0.])

    def test_clipped_triangle_keeps_only_ground_patch(self):
        p=np.array([[0.,0,0],[1,0,.2],[0,1,.2]])
        camera=camera_for(p,45);polys,anchor=contact_geometry(p,np.array([[0,1,2]]),camera)
        self.assertEqual(len(polys),1)
        full=project(p,camera)
        self.assertLess(np.ptp(polys[0],axis=0).max(),np.ptp(full,axis=0).max()*.15)

    def test_contact_anchor_independent_of_render_and_shadow_size(self):
        anchor=np.array([.4123,.6831]);target=np.array([1345.2,866.8]);full=812.4
        for resolution in (512,1024,2048):
            offset=target-anchor*full
            actual=(anchor*resolution)*(full/resolution)+offset
            np.testing.assert_allclose(actual,target,atol=1e-10)
        # Increasing a cropped shadow margin changes crop origin, never the anchor.
        for crop in ([1000,500],[930,410]):
            local=anchor*full+offset-crop
            np.testing.assert_allclose(local+crop,target,atol=1e-10)

    def test_cross_depth_reference_rejected(self):
        depth=np.zeros((100,100),np.float32);depth[:,70:]=.25
        surface={'reference':[[.1,.5],[.8,.5]],'reference_mm':[200,400]}
        with self.assertRaisesRegex(ValueError,'cross_depth'):
            scale_from_reference(surface,[180,220],100,100,depth,[20,50])

    def test_horizontal_rule_rejects_standing_cylinder(self):
        self.assertAlmostEqual(horizontal_angle([0,0,1],np.eye(4)),90)
        m=np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        self.assertAlmostEqual(horizontal_angle([0,0,1],m),0)

    def test_linear_shadow_and_no_alpha_bleed(self):
        values=np.linspace(0,1,100,dtype=np.float32)
        np.testing.assert_allclose(srgb(linear(values)),values,atol=2e-7)
        src=np.zeros((32,32,4),np.float32);src[5:27,5:27,:]=1;src[13:19,13:19,:]=0
        out=warp(src,64,[10,10],[0,0,100,100])
        self.assertEqual(float(out[40,40,3]),0.)
        self.assertEqual(float(out[0,0,3]),0.)

    def test_tone_preserves_highlight_separation(self):
        rgb=np.full((40,40,3),.48,np.float32);rgb[10:20,10:20]=.95
        spec=np.zeros((40,40),np.float32);spec[10:20,10:20]=1
        alpha=np.ones((40,40),np.float32)
        matched,record=match_rgb(rgb,alpha,np.full((500,3),75),spec,'darker',3)
        self.assertGreater(float(matched[12:18,12:18].mean()-matched[25:35,25:35].mean()),.35)
        self.assertLess(record['exposure_ev'],0)


class QueueTests(unittest.TestCase):
    def test_1000_jobs_restart_exactly_once(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'jobs.sqlite';store=Store(path)
            jobs=[{'n':i} for i in range(1000)];store.initialize('frozen',jobs)
            completed=[]
            for _ in range(137):
                job=store.claim();completed.append(job['id']);store.finish(job['id'],'review',{'id':job['id']})
            interrupted=store.claim();store.checkpoint(interrupted['id'],'review',{'iteration':1})
            store.close();store=Store(path);store.initialize('frozen',jobs);store.recover()
            restarted=store.claim();self.assertEqual(restarted['id'],interrupted['id'])
            self.assertEqual(restarted['checkpoint']['iteration'],1)
            store.finish(restarted['id'],'accepted_display',{'id':restarted['id']});completed.append(restarted['id'])
            while job:=store.claim():
                self.assertNotIn(job['id'],completed);completed.append(job['id'])
                store.finish(job['id'],'review',{'id':job['id']})
            self.assertEqual(len(set(completed)),1000)
            self.assertEqual(sum(store.counts().values()),1000)
            with self.assertRaises(ValueError):store.initialize('changed',jobs)
            store.close()

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with batch_lock(d):
                with self.assertRaises(RuntimeError):
                    with batch_lock(d):pass

    def test_actions_are_bounded_and_id_checked(self):
        params={'candidate':'A','material':'original','light':'scene','tone':'scene'}
        with self.assertRaises(ValueError):apply_actions(params,[Action(kind='candidate',value='unknown')],['A'])
        with self.assertRaises(ValueError):apply_actions(params,[Action(kind='tone',value='execute(code)')],['A'])
        with self.assertRaises(ValueError):apply_actions(params,[Action(kind='tone',value='darker')]*2,['A'])
        self.assertEqual(apply_actions(params,[Action(kind='material',value='clearer')],['A'])['candidate'],'A')

    def test_ai_score_cannot_override_physical_failure(self):
        passed={'support':{'status':'pass'}};failed={'support':{'status':'fail'}}
        self.assertGreater(version_rank(passed,review(3)),version_rank(failed,review(4)))


class WebTests(unittest.TestCase):
    def test_batch_validation_and_no_arbitrary_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);background=root/'background';background.mkdir();Image.new('RGB',(30,20)).save(background/'frame.jpg')
            with patch.object(batch_web,'ROOT',root),patch.object(batch_web,'DATA',root/'configs'),\
                 patch.object(batch_web,'load_catalog',return_value={'3':{'mesh_hash':'hash'}}),\
                 patch.object(batch_web,'launch') as launch,TestClient(app) as client:
                payload={'backgrounds':str(background),'assets':['missing'],'count':1,'output':'outside'}
                self.assertEqual(client.post('/api/batches',json=payload).status_code,400)
                launch.assert_not_called()
                payload.update(assets=['3'],photo_style='scrap_gray',placement_mode='embedded',uncertain_policy='render_review')
                response=client.post('/api/batches',json=payload)
                self.assertEqual(response.status_code,202);info=response.json()
                self.assertTrue(Path(info['output']).is_relative_to(root/'outputs'))
                saved=json.loads((root/'configs'/(info['id']+'.json')).read_text(encoding='utf8'))
                self.assertEqual(saved['placement_mode'],'embedded')
                self.assertEqual(saved['photo_style'],'scrap_gray')
                launch.assert_called_once_with(info['id'])
                status=client.get('/api/batches/'+info['id']).json()
                self.assertFalse(status['running']);self.assertFalse(status['has_results'])
                self.assertEqual(client.get('/api/batches/../../secret').status_code,404)

    def test_artifacts_cannot_escape_batch_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();ident='0123456789abcdef'
            config=BatchConfig(backgrounds=d,output=str(out));atomic_json(root/(ident+'.json'),config.model_dump())
            atomic_json(out/'summary.json',{'ok':True});atomic_json(root/'secret.json',{'private':'test-only'})
            with patch.object(batch_web,'DATA',root),TestClient(app) as client:
                prefix='/api/batches/'+ident+'/files/'
                self.assertEqual(client.get(prefix+'summary.json').status_code,200)
                self.assertEqual(client.get(prefix+'%2e%2e%2fsecret.json').status_code,404)


class APITests(unittest.IsolatedAsyncioTestCase):
    async def test_wrapped_transport_error_pauses_after_one_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);provider=root/'provider.json'
            atomic_json(provider,{'vision':{'url':'http://127.0.0.1:9999/v1/chat/completions','model':'fake','key_env':None}})
            vision=Vision(BatchConfig(backgrounds=d,output=d,provider_config=str(provider)),root/'cache',root/'usage.json')
            def unavailable(request):raise httpx.ConnectError('offline fixture',request=request)
            await vision.http.aclose();vision.http=httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
            with self.assertRaises(ProviderUnavailable):await vision.ask('test',Probe,'Test',{})
            self.assertEqual(len(vision.calls),2)
            self.assertTrue(all(c['transport_error']=='ConnectError' for c in vision.calls))
            await vision.close()

    async def test_global_outage_stops_after_one_retry(self):
        for status in (429,503):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as d:
                root=Path(d);provider=root/'provider.json'
                atomic_json(provider,{'vision':{'url':'http://127.0.0.1:9999/v1/chat/completions','model':'fake','key_env':None}})
                vision=Vision(BatchConfig(backgrounds=d,output=d,provider_config=str(provider)),root/'cache',root/'usage.json')
                await vision.http.aclose();vision.http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(status)))
                with self.assertRaises(ProviderUnavailable):await vision.ask('test',Probe,'Test',{})
                self.assertEqual(len(vision.calls),2);await vision.close()

    async def test_cache_hit_metadata_does_not_cause_new_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);provider=root/'provider.json'
            atomic_json(provider,{'vision':{'url':'http://127.0.0.1:9999/v1/chat/completions','model':'fake','key_env':None}})
            vision=Vision(BatchConfig(backgrounds=d,output=d,provider_config=str(provider)),root/'cache',root/'usage.json')
            answer=json.dumps({'image_visible':True,'description':'图像可见'})
            await vision.http.aclose();vision.http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={
                'choices':[{'finish_reason':'stop','message':{'tool_calls':[{'type':'function','function':{'name':'inspect_display','arguments':answer}}]}}],
                'usage':{'prompt_tokens':5,'completion_tokens':5,'total_tokens':10}})))
            await vision.ask('test',Probe,'Test',{'physics':{'cache_hit':False,'value':.8}})
            await vision.ask('test',Probe,'Test',{'physics':{'cache_hit':True,'value':.8}})
            self.assertEqual(len(vision.calls),1)
            await vision.ask('test',Probe,'Test',{'physics':{'cache_hit':True,'value':.9}})
            self.assertEqual(len(vision.calls),2);await vision.close()

    async def test_failed_asset_preparation_is_reused_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pipe=Pipeline.__new__(Pipeline);pipe.prepare_lock=asyncio.Lock()
            pipe.assets={};pipe.cache=root;pipe.blender=None
            pipe.catalog={'A':{'mesh_hash':'mesh','name':'test'}}
            endpoint=type('Endpoint',(),{'model_dump':lambda self:{'model':'fake'}})()
            pipe.vision=type('Vision',(),{'endpoint':endpoint})()
            with patch('display_batch.pipeline.prepare_asset',side_effect=ValueError('invalid volume')) as prepare:
                first=await pipe.prepare_asset('A');pipe.assets={}
                second=await pipe.prepare_asset('A')
                self.assertEqual(prepare.call_count,1);self.assertEqual(first,second)
                self.assertEqual(second['poses'],[])

    async def test_loose_or_unknown_assembly_never_simulated(self):
        for relationship in ('loose_parts','unknown'):
            with self.subTest(relationship=relationship),tempfile.TemporaryDirectory() as d:
                root=Path(d);board=root/'board.jpg';Image.new('RGB',(20,20)).save(board)
                class FakeWorker:
                    async def ask(self,operation,payload):
                        if operation!='export':raise AssertionError('uncertain assembly simulated')
                        np.savez(payload['geometry'],vertices=np.eye(3),faces=[[0,1,2]])
                        return {'extents':[1,1,1],'materials':[]}
                class FakeVision:
                    async def ask(self,*args):
                        return AssetAnswer(major_mm=(100,200),assembly=relationship,upright_axis='none',
                                           material_classes=['steel'],key_features=[],evidence='关系未知')
                row={'mesh_hash':'mesh','board':str(board),'mesh':'model','name':'test'}
                result=await prepare_asset(row,root/'cache',FakeWorker(),FakeVision())
                self.assertEqual(result['poses'],[])
                self.assertEqual(result['reason'],'assembly_relationship_unknown_or_loose')

    async def test_schema_retry_bounded_and_cache_has_zero_new_calls(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image=root/'image.png';Image.new('RGB',(1600,1200),'gray').save(image)
            endpoint=root/'provider.json';atomic_json(endpoint,{'vision':{'url':'http://127.0.0.1:9999/v1/chat/completions','model':'fake','key_env':None}})
            config=BatchConfig(backgrounds=d,output=d,provider_config=str(endpoint))
            vision=Vision(config,root/'cache',root/'usage.json');requests=[]
            async def respond(request):
                requests.append(json.loads(request.content))
                answer='{}' if len(requests)==1 else json.dumps({'image_visible':True,'description':'灰色图像'})
                return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'tool_calls':[{'type':'function','function':{'name':'inspect_display','arguments':answer}}]}}],
                                               'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})
            await vision.http.aclose();vision.http=httpx.AsyncClient(transport=httpx.MockTransport(respond))
            result=await vision.ask('test',Probe,'Test',{},[image]);self.assertTrue(result.image_visible)
            await vision.ask('test',Probe,'Test',{},[image])
            self.assertEqual(len(requests),2);self.assertEqual(len(vision.calls),2);self.assertEqual(vision.cache_hits,1)
            self.assertIn('format_correction',requests[1]['messages'][1]['content'][0]['text'])
            await vision.close()

    async def test_auth_error_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'provider.json';atomic_json(path,{'vision':{'url':'http://127.0.0.1:9999/v1/chat/completions','model':'fake','key_env':None}})
            vision=Vision(BatchConfig(backgrounds=d,output=d,provider_config=str(path)),Path(d)/'cache',Path(d)/'usage.json')
            await vision.http.aclose();vision.http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(401)))
            with self.assertRaises(ProviderUnavailable):await vision.ask('test',Probe,'Test',{})
            self.assertEqual(len(vision.calls),1);await vision.close()


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_quarantine_stops_later_images(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pipe=Pipeline.__new__(Pipeline);pipe.out=root;pipe.store=Store(root/'jobs.sqlite')
            pipe.model_review={'A':{'mesh_hash':'h','job_id':'prior','defects':[],'independent_confirmation':True}}
            pipe.vision=type('CachedVision',(),{'calls':[]})()
            payload={'asset_id':'A','mesh_hash':'h','background':'unused','seed':0}
            pipe.store.initialize('test',[payload])
            result=await pipe.job(pipe.store.claim())
            self.assertEqual(result['reason'],'asset_structure_requires_review')
            self.assertEqual(result['iterations'],[]);self.assertEqual(result['new_api_calls'],0)
            pipe.store.close()

    async def run_loop(self,max_iterations,reviews,hard_pass=True):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pipe=Pipeline.__new__(Pipeline);pipe.config=BatchConfig(backgrounds=d,output=d,max_iterations=max_iterations)
            pipe.out=root;pipe.cpu=asyncio.Semaphore(2);pipe.store=Store(root/'jobs.sqlite')
            payload={'asset_id':'A','background':'background','seed':3}
            pipe.store.initialize('test',[payload]);job=pipe.store.claim()
            asset={'signature':'asset','poses':[{}],'physics_signature':'physics','name':'测试',
                   'semantic':{'key_features':[],'major_mm':[100,200]},'rust':'rust','board':'board'}
            async def prep(k):return asset
            pipe.prepare_asset=prep
            class FakeScene:
                async def prepare(self,p):return {'key':'scene'}
            pipe.scenes=FakeScene()
            class FakeVision:
                def __init__(self):self.calls=[]
                async def ask(self,*args,**kwargs):
                    i=len(self.calls);self.calls.append({'fake':True});value=reviews[min(i,len(reviews)-1)]
                    if isinstance(value,Exception):raise value
                    return value
            pipe.vision=FakeVision()
            async def render(*args):return (root,{},None,None,False)
            pipe.render=render
            candidates=[{'id':'A','target_px':200,'score':1}]
            def compose(*args):
                path=args[-1];Image.new('RGB',(12,12),({'scene':40,'darker':80,'cooler':100}.get(args[-3]['tone'],120),0,0)).save(path)
                crop=path.with_suffix('.crop.jpg');Image.new('RGB',(12,12)).save(crop)
                return {'checks':{'support':{'status':'pass' if hard_pass else 'fail'}},
                        'image':str(path),'crop':str(crop),'preview':str(crop),'measured_major_px':200}
            with patch('display_batch.pipeline.choose_candidates',return_value=(candidates,[])),patch('display_batch.pipeline.compose',side_effect=compose):
                result=await pipe.job(job)
            calls=len(pipe.vision.calls);pipe.store.close()
            if result.get('image'):result['_copied_hash']=file_hash(result['image'])
            return result,calls

    async def test_zero_iterations_still_reviews_initial(self):
        result,calls=await self.run_loop(0,[review(2,[Action(kind='tone',value='darker')])])
        self.assertEqual(calls,1);self.assertEqual(result['status'],'review');self.assertIn('/review/',result['image'].replace('\\','/'))

    async def test_one_repair_then_pass(self):
        result,calls=await self.run_loop(2,[review(2,[Action(kind='tone',value='darker')]),review(3)])
        self.assertEqual(calls,2);self.assertEqual(result['status'],'accepted_display')

    async def test_repeat_stops_and_keeps_best(self):
        result,calls=await self.run_loop(10,[review(2,[Action(kind='tone',value='darker')]),review(1,[Action(kind='tone',value='darker')])])
        self.assertEqual(calls,2);self.assertEqual(result['best_iteration'],0);self.assertEqual(result['reason'],'repeated_parameters')

    async def test_high_score_never_accepts_failed_contact(self):
        result,calls=await self.run_loop(2,[review(4)],hard_pass=False)
        self.assertEqual(result['status'],'review')

    async def test_default_two_repairs_are_bounded(self):
        result,calls=await self.run_loop(2,[review(1,[Action(kind='tone',value='darker')]),
                                          review(2,[Action(kind='tone',value='cooler')]),review(2)])
        self.assertEqual(calls,3);self.assertEqual(len(result['iterations']),3)

    async def test_failed_later_review_preserves_best_reviewed_image(self):
        result,calls=await self.run_loop(2,[review(2,[Action(kind='tone',value='darker')]),RuntimeError('invalid response')])
        self.assertEqual(result['status'],'failed');self.assertEqual(result['best_iteration'],0)
        self.assertEqual(result['_copied_hash'],result['iterations'][0]['image_hash']);self.assertEqual(calls,2)

    async def test_failed_first_review_keeps_program_checks(self):
        result,calls=await self.run_loop(0,[RuntimeError('invalid response')])
        self.assertEqual(result['status'],'failed');self.assertTrue(result['unreviewed_draft'])
        self.assertEqual(result['checks']['support']['status'],'pass');self.assertEqual(calls,1)

    async def test_unchanged_pixels_stop_before_another_paid_review(self):
        result,calls=await self.run_loop(2,[review(2,[Action(kind='chroma',value='more')])])
        self.assertEqual(calls,1);self.assertEqual(result['reason'],'no_effective_image_change')

    async def test_occluded_structure_does_not_quarantine_model(self):
        defect=Defect(code='structure',scope='asset',severity='major',visibility='occluded',
                      region=[.2,.2,.5,.5],evidence='钢料遮挡，无法判断')
        result,calls=await self.run_loop(0,[review(2,defects=[defect])])
        self.assertEqual(calls,1);self.assertNotEqual(result['reason'],'asset_structure_requires_review')


if __name__=='__main__':unittest.main(verbosity=2)
