"""Offline checks for state, masking, and loopback boundaries."""
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image
from starlette.testclient import TestClient
import server
from relight import ambient_match
from occlusion import refine_thin_mask, thin_line_seeds
from lighting_estimate import estimate_light_from_rods


class PlacementChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        server.DATA = Path(self.tmp.name)
        server.SESSIONS.clear()
        self.client = TestClient(server.app)
        im = Image.new('RGB',(128,96),(70,80,90))
        buf = io.BytesIO(); im.save(buf,format='PNG')
        response = self.client.post('/api/sessions',files={'background':('test.png',buf.getvalue(),'image/png')})
        self.assertEqual(response.status_code,200)
        self.s = response.json(); self.url = '/api/sessions/'+self.s['id']
        self.asset = next(iter(server.ROWS))
        self.state = {'asset':self.asset,'x':64,'y':60,'size':40,'azimuth':80,'elevation':45}
        self.assertEqual(self.client.post(self.url+'/state',json=self.state).status_code,200)
        self.live = server.SESSIONS[self.s['id']]

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def masks(self):
        self.live['candidates'] = [{'id':1},{'id':2}]
        folder = Path(self.live['directory']) / f"analysis-{self.live['revision']}"
        folder.mkdir()
        a = np.zeros((96,128),np.uint8); a[20:80,20:24] = 255
        b = np.zeros((96,128),np.uint8); b[30:35,10:100] = 255
        Image.fromarray(a).save(folder/'1.png'); Image.fromarray(b).save(folder/'2.png')
        return a,b

    def test_library_and_rejected_foreign_origin(self):
        response = self.client.get('/api/assets')
        original=[r for r in response.json() if not r['manual_only']]
        self.assertEqual(len(original),39)
        self.assertEqual(sum(not r['passed'] for r in original),15)
        self.assertEqual(self.client.get('/api/assets',headers={'origin':'https://evil.example'}).status_code,403)
        self.assertEqual(self.client.get('/api/assets',headers={'host':'evil.example'}).status_code,403)

    def test_masks_union_undo_and_revision(self):
        a,b = self.masks(); rev = self.live['revision']
        response = self.client.post(self.url+'/selection',json={'revision':rev,'selected':[1,2]})
        self.assertEqual(response.status_code,200)
        self.assertTrue(np.array_equal(np.asarray(server.foreground(self.live)),np.maximum(a,b)))
        self.client.post(self.url+'/undo')
        self.assertFalse(np.asarray(server.foreground(self.live)).any())
        self.client.post(self.url+'/state',json={**self.state,'x':65})
        self.assertEqual(self.client.post(self.url+'/selection',json={'revision':rev,'selected':[1]}).status_code,400)
        self.assertFalse(self.live['candidates'])

    def test_foreground_restores_original_pixels(self):
        a,_ = self.masks(); self.live['selected']=[1]
        original = Image.open(self.live['background']).convert('RGBA')
        composite = Image.new('RGBA',original.size,(255,0,0,255))
        composite.paste(original,(0,0),server.foreground(self.live))
        self.assertTrue(np.array_equal(np.asarray(composite)[a>0],np.asarray(original)[a>0]))
        self.assertTrue(np.all(np.asarray(composite)[a==0,0]==255))

    def test_support_and_old_export_invalidation(self):
        self.masks(); self.live['export']={'directory':'old'}
        rev=self.live['revision']
        result=self.client.post(self.url+'/support',json={'revision':rev,'support':1})
        self.assertEqual(result.status_code,200)
        self.assertEqual(self.live['support'],1)
        self.assertNotIn('export',self.live)
        self.assertEqual(self.client.get(self.url+'/export/manual.png').status_code,404)

    def test_old_preview_is_rejected_before_rendering(self):
        old = self.live['revision']
        self.client.post(self.url+'/state',json={**self.state,'x':65})
        result = self.client.post(self.url+'/preview',json={'revision':old,'image':''})
        self.assertEqual(result.status_code,409)

    def test_support_cannot_also_be_foreground(self):
        self.masks(); rev = self.live['revision']
        self.assertEqual(self.client.post(self.url+'/support',json={'revision':rev,'support':1}).status_code,200)
        self.assertEqual(self.client.post(self.url+'/selection',json={'revision':rev,'selected':[1]}).status_code,400)

    def test_fallback_color_is_explicitly_unverified(self):
        obj=Image.new('RGBA',(40,40),(160,140,120,255))
        matched,info=server.match_local_photo(self.live,obj,44,40)
        self.assertEqual(info['status'],'measured')
        self.assertEqual(info['support_verification'],'surrounding_scene_unverified')
        self.assertTrue(np.array_equal(np.asarray(matched.getchannel('A')),np.asarray(obj.getchannel('A'))))

    def test_candidate_cannot_enter_auto_pipeline(self):
        candidate=next((key for key,row in server.ROWS.items() if row.get('manual_only')),None)
        if not candidate:self.skipTest('candidate not installed')
        self.client.post(self.url+'/state',json={**self.state,'asset':candidate})
        response=self.client.post(self.url+'/automatic')
        self.assertEqual(response.status_code,400)

    def test_reference_rebuild_stays_manual_with_failed_source_diagnostic(self):
        original=server.ROWS[self.asset]
        folder=Path(self.tmp.name)/'candidate';folder.mkdir()
        mesh=folder/'mesh.glb';mesh.write_bytes(b'review-only fixture')
        reference=folder/'reference.png';reference.write_bytes(b'reference fixture')
        info={'draft':False,'status':'review','shape_basis':'user_six_view_reference','number':1,
              'asset_id':original['asset_id'],'original_mesh_sha256':original['mesh_sha256'],
              'inputs':{'rust':original['rust_sha256'],'reference':server.digest(reference)},
              'reference':str(reference),'mesh_sha256':server.digest(mesh),
              'version':'valve01-parametric-v6','source_fidelity_diagnostic':{'passed':False,'iou':.67}}
        (folder/'result.json').write_text(json.dumps(info),encoding='utf8')
        with patch.object(server,'ROWS',{original['asset_id']:original}):
            server.register_rebuilt_candidate(folder,variant='v7')
            self.assertEqual(len(server.ROWS),1)
        with patch.object(server,'ROWS',{original['asset_id']:original}):
            server.register_rebuilt_candidate(folder)
            rebuilt=next(row for row in server.ROWS.values() if row.get('manual_only'))
            self.assertFalse(rebuilt['reconstruction']['fidelity']['passed'])
            self.assertEqual(rebuilt['model_review'],'pending')
            self.assertEqual(rebuilt['scene_validation'],'pending')
            self.assertEqual(self.client.post(self.url+'/state',json={**self.state,'asset':rebuilt['asset_id']}).status_code,200)
            self.assertEqual(self.client.post(self.url+'/automatic').status_code,400)
        mesh.write_bytes(b'tampered')
        with patch.object(server,'ROWS',{original['asset_id']:original}):
            server.register_rebuilt_candidate(folder)
            self.assertEqual(len(server.ROWS),1)

    def test_model_review_page_and_api_contract(self):
        for number in (1,2,3,4):
            page=self.client.get(f'/review/{number:02d}')
            self.assertEqual(page.status_code,200)
            self.assertIn('模型待人工检查',page.text)
            self.assertEqual(self.client.get(f'/api/review/{number:02d}/unknown').status_code,404)
        rows=self.client.get('/api/assets').json()
        self.assertTrue(all('version' in r and 'shape_basis' in r and 'model_review' in r and 'scene_validation' in r for r in rows))
        for version in ('valve02-parametric-v1','bearing03-parametric-v1','bearing04-parametric-v1'):
            candidate=next((r for r in rows if r.get('version')==version),None)
            if candidate:
                self.assertTrue(candidate['manual_only'])
                self.assertEqual(candidate['model_review'],'pending')
                self.assertEqual(candidate['scene_validation'],'pending')

    def test_review_images_open_inline_and_blend_downloads(self):
        root=Path(self.tmp.name);folder=root/'outputs/01阀门-六视图重建-v7';folder.mkdir(parents=True)
        (folder/'六视图-新旧对照.jpg').write_bytes(b'image fixture')
        (folder/'model.blend').write_bytes(b'blend fixture')
        with patch.object(server,'ROOT',root):
            self.assertTrue(self.client.get('/api/review/01/board').headers['content-disposition'].startswith('inline'))
            self.assertTrue(self.client.get('/api/review/01/blend').headers['content-disposition'].startswith('attachment'))

    def test_thin_line_cannot_be_support(self):
        self.masks();self.live['candidates'][0]['kind']='thin_line_proposal'
        result=self.client.post(self.url+'/support',json={'revision':self.live['revision'],'support':1})
        self.assertEqual(result.status_code,400)

    def test_validate_bad_pose_and_model(self):
        for bad in ({'size':float('nan')},{'asset':'../../bad'},{'elevation':90}):
            response=self.client.post(self.url+'/state',json={**self.state,**bad}) if 'size' not in bad else None
            if response is not None:self.assertEqual(response.status_code,400)
        with self.assertRaises(ValueError):server.validate_state({**self.state,'size':float('nan')},self.live)


class RelightChecks(unittest.TestCase):
    def test_ambient_matching_preserves_alpha_and_excludes_foreground(self):
        frame = Image.new('RGB',(240,240),(75,80,85))
        obj = Image.new('RGBA',(80,80),(150,130,110,0))
        alpha = np.zeros((80,80),np.uint8); alpha[10:70,10:70]=255; alpha[30:50,30:50]=0
        obj.putalpha(Image.fromarray(alpha))
        support = np.ones((240,240),bool)
        foreground = np.zeros((240,240),bool); foreground[:,120:]=True
        result, info = ambient_match(frame,support,foreground,obj,80,80)
        self.assertEqual(info['status'],'measured')
        self.assertEqual(info['light_direction'],'unknown')
        self.assertTrue(np.array_equal(np.asarray(result.getchannel('A')),alpha))
        self.assertLess(info['support_samples'],np.count_nonzero(support))
        target=info['scene']['luma_p20_p50_p80'][1]
        self.assertLess(abs(info['object_after']['luma_p20_p50_p80'][1]-target),
                        abs(info['object_before']['luma_p20_p50_p80'][1]-target))
        self.assertFalse(np.array_equal(np.asarray(result)[:,:,:3],np.asarray(obj)[:,:,:3]))
        self.assertTrue(np.all(np.asarray(obj)[:,:,0]==150))

    def test_insufficient_support_keeps_original(self):
        frame=Image.new('RGB',(100,100),(80,80,80))
        obj=Image.new('RGBA',(30,30),(120,110,100,255))
        mask=np.zeros((100,100),bool)
        result,info=ambient_match(frame,mask,mask,obj,35,35)
        self.assertEqual(info['status'],'insufficient_samples')
        self.assertTrue(np.array_equal(np.asarray(obj),np.asarray(result)))

    def test_gradient_is_bounded_and_not_claimed_as_light_direction(self):
        x = np.linspace(65,150,240,dtype=np.uint8)
        pixels = np.repeat(np.broadcast_to(x,(240,240))[:,:,None],3,axis=2).copy()
        frame = Image.fromarray(pixels)
        obj = Image.new('RGBA',(80,80),(145,135,125,255))
        support = np.ones((240,240),bool)
        result,info = ambient_match(frame,support,np.zeros_like(support),obj,80,80)
        self.assertEqual(info['light_direction'],'unknown')
        self.assertIsNotNone(info['screen_gradient_rgb_per_object_width'])
        self.assertLessEqual(max(abs(v) for v in info['screen_gradient_rgb_per_object_width']),5.4)
        self.assertTrue(np.array_equal(np.asarray(result.getchannel('A')),np.asarray(obj.getchannel('A'))))

    def test_blur_never_changes_openings_or_transparent_edges(self):
        frame=Image.new('RGB',(240,240),(95,95,95))
        pixels=np.zeros((80,80,4),np.uint8)
        yy,xx=np.indices((80,80))
        pixels[:,:,:3]=np.where(((xx+yy)%2)[:,:,None]==0,90,180)
        pixels[10:70,10:70,3]=255; pixels[30:50,30:50,3]=0
        obj=Image.fromarray(pixels)
        result,info=ambient_match(frame,np.ones((240,240),bool),np.zeros((240,240),bool),obj,80,80)
        self.assertGreater(info['detail_blur_sigma_px'],0)
        self.assertTrue(np.array_equal(np.asarray(result.getchannel('A')),pixels[:,:,3]))


class ThinOcclusionChecks(unittest.TestCase):
    def test_native_line_pair_proposal(self):
        pixels=np.full((240,240,3),180,np.uint8)
        pixels[30:210,115:121]=40
        seeds=thin_line_seeds(Image.fromarray(pixels),(120,120),160)
        self.assertTrue(seeds)
        self.assertGreater(max(int(seed[30:210,115:121].sum()) for seed in seeds),200)

    def test_refinement_is_bounded_and_does_not_change_depth_order(self):
        frame=Image.new('RGB',(100,100),(100,100,100))
        seed=np.zeros((100,100),bool);seed[15:85,48:52]=True
        class Fake:
            def refine_crop(self,image,mask):
                output=mask.copy();output[:,1:] |= mask[:,:-1];return output
        refined,info=refine_thin_mask(Fake(),frame,seed,(0,0,100,100),radius=3)
        self.assertEqual(info['status'],'refined')
        self.assertEqual(info['order'],'unknown')
        self.assertTrue(refined[20:80,52].all())
        self.assertFalse(refined[:,60:].any())

    def test_untrusted_refinement_is_not_applied(self):
        frame=Image.new('RGB',(100,100),(100,100,100))
        seed=np.zeros((100,100),bool);seed[15:85,48:52]=True
        class Fake:
            def refine_crop(self,image,mask):return np.ones(mask.shape,bool)
        refined,info=refine_thin_mask(Fake(),frame,seed,(0,0,100,100),radius=3)
        self.assertEqual(info['status'],'unknown')
        self.assertTrue(np.array_equal(refined,seed))


class SceneLightChecks(unittest.TestCase):
    def test_no_cues_remains_unknown(self):
        frame=Image.new('RGB',(200,200),(100,100,100))
        result=estimate_light_from_rods(frame,(100,100),150)
        self.assertEqual(result['status'],'unknown')

    def test_diverse_consistent_cues_produce_bounded_direction(self):
        import cv2
        shape=(300,300);masks=[]
        for start,end in [((30,50),(120,50)),((170,40),(170,130)),((90,180),(170,260)),((200,200),(260,260))]:
            mask=np.zeros(shape,np.uint8);cv2.line(mask,start,end,1,12);masks.append(mask>0)
        yy,xx=np.indices(shape);pixels=np.full(shape,120.0)
        for mask in masks:
            y,x=np.nonzero(mask);cx=x.mean();cy=y.mean()
            pixels[y,x]=120+2*(x-cx)+1*(y-cy)
        pixels=np.uint8(np.clip(pixels,0,255))
        frame=Image.fromarray(np.repeat(pixels[:,:,None],3,axis=2))
        with patch('lighting_estimate.thin_line_seeds',return_value=masks):
            result=estimate_light_from_rods(frame,(150,150),250)
        self.assertEqual(result['status'],'estimated')
        self.assertEqual(result['confidence'],'low')
        self.assertGreater(result['direction_screen'][0],.5)


if __name__=='__main__':
    unittest.main()
