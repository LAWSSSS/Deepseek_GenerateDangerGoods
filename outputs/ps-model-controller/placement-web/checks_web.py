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
from relight import ambient_match, estimate_render_environment
from contact_shadow import make_contact_shadow, projected_footprint
from photometry import quiet_photo
from occlusion import foreground_matte, refine_thin_mask, thin_line_seeds
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

    def test_placement_catalog_matches_3d_viewer(self):
        response = self.client.get('/api/assets?view=placement')
        self.assertEqual(response.status_code, 200)
        rows = response.json()
        self.assertEqual(len(rows), 44)
        self.assertEqual([row['id'] for row in rows if row['number'] == 19],
                         [f'19-{index}' for index in range(1, 7)])
        self.assertTrue(all(row['available'] for row in rows))
        self.assertEqual({row['id'] for row in rows}, set(server.PLACEMENT_ROWS))
        self.assertEqual({row['id'] for row in rows if row['number'] == 10}, {'10'})
        self.assertEqual({row['id'] for row in rows if row['number'] == 13}, {'13'})
        self.assertEqual({row['id'] for row in rows if row['number'] == 31}, {'31'})

    def test_library_and_rejected_foreign_origin(self):
        response = self.client.get('/api/assets')
        original=[r for r in response.json() if not r['manual_only']]
        baseline=[r for r in server.ROWS.values() if not r.get('manual_only')]
        self.assertEqual(len(baseline),39)
        self.assertEqual(sum(not r.get('reconstruction',{}).get('fidelity',{}).get('passed') for r in baseline),15)
        hidden={r['number'] for r in baseline if r.get('superseded_by_feedback')}
        self.assertEqual({r['number'] for r in original},set(range(1,40))-hidden)
        self.assertEqual(self.client.get('/api/assets',headers={'origin':'https://evil.example'}).status_code,403)
        self.assertEqual(self.client.get('/api/assets',headers={'host':'evil.example'}).status_code,403)

    def test_user_feedback_models_are_single_and_supersede_groups(self):
        rows=self.client.get('/api/assets').json()
        for number,count in ((10,1),(13,1),(19,6),(31,1)):
            candidates=[row for row in rows if row['number']==number]
            self.assertEqual(len(candidates),count)
            self.assertTrue(all(row['quantity']==1 and row['manual_only'] for row in candidates))
            self.assertTrue(all(row['model_review']=='pending' for row in candidates))
            for row in candidates:
                self.assertEqual(self.client.get('/api/model/'+row['id']).status_code,200)
                result=self.client.post(self.url+'/state',json={**self.state,'asset':row['id'],'quantity':2})
                self.assertEqual(result.status_code,400)
                result=self.client.post(self.url+'/state',json={**self.state,'asset':row['id'],'quantity':1})
                self.assertEqual(result.status_code,200)
        old=next(key for key,row in server.ROWS.items() if row.get('number')==19 and not row.get('manual_only'))
        self.assertEqual(self.client.post(self.url+'/state',json={**self.state,'asset':old}).status_code,400)
        page=self.client.get('/review/13')
        self.assertEqual(page.status_code,200)
        self.assertIn('单只模型检查',page.text)
        self.assertEqual(self.client.get('/api/review/13/board').status_code,200)
        self.assertEqual(self.client.get('/api/review/13/blend').status_code,200)
        page=self.client.get('/review/10')
        self.assertEqual(page.status_code,200)
        self.assertIn('去操作杆模型检查',page.text)
        self.assertEqual(self.client.get('/api/review/10/board').status_code,200)
        self.assertEqual(self.client.get('/api/review/10/blend').status_code,200)

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
        with Image.open(self.live['background']) as frame:
            matched,info=server.match_local_photo(self.live,frame,obj,44,40,None,np.zeros((96,128),bool))
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
        for number in (1,2,3,4,5,6,7,8,9):
            page=self.client.get(f'/review/{number:02d}')
            self.assertEqual(page.status_code,200)
            self.assertIn('模型待人工检查',page.text)
            self.assertEqual(self.client.get(f'/api/review/{number:02d}/unknown').status_code,404)
        rows=self.client.get('/api/assets').json()
        self.assertTrue(all('version' in r and 'shape_basis' in r and 'model_review' in r and 'scene_validation' in r for r in rows))
        for version in ('valve02-parametric-v1','bearing03-parametric-v1','bearing04-parametric-v1','tank05-parametric-v1','tank06-parametric-v1','separator08-parametric-v1','separator09-parametric-v1'):
            candidate=next((r for r in rows if r.get('version')==version),None)
            if candidate:
                self.assertTrue(candidate['manual_only'])
                self.assertEqual(candidate['model_review'],
                                 'approved' if candidate['number'] in server.load_feedback()['approved'] else 'pending')
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
        for bad in ({'size':float('nan')},{'asset':'../../bad'},{'elevation':90},{'elevation':-90}):
            response=self.client.post(self.url+'/state',json={**self.state,**bad}) if 'size' not in bad else None
            if response is not None:self.assertEqual(response.status_code,400)
        for elevation in (-89,0,89):
            response=self.client.post(self.url+'/state',json={**self.state,'elevation':elevation})
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['state']['elevation'],elevation)
        response=self.client.post(self.url+'/state',json={**self.state,'azimuth':360})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['state']['azimuth'],360)
        response=self.client.post(self.url+'/state',json={**self.state,'rot_x':30,'rot_y':-20,'rot_z':45})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['state']['rot_x'],30)
        self.assertEqual(response.json()['state']['rot_y'],-20)
        self.assertEqual(response.json()['state']['rot_z'],45)
        response=self.client.post(self.url+'/state',json={**self.state,'rot_x':181})
        self.assertEqual(response.status_code,400)
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

    def test_material_color_survives_exposure_match(self):
        x = np.linspace(65,150,240,dtype=np.uint8)
        pixels = np.repeat(np.broadcast_to(x,(240,240))[:,:,None],3,axis=2).copy()
        frame = Image.fromarray(pixels)
        obj = Image.new('RGBA',(80,80),(173,53,39,255))
        support = np.ones((240,240),bool)
        result,info = ambient_match(frame,support,np.zeros_like(support),obj,80,80)
        self.assertEqual(info['light_direction'],'unknown')
        self.assertGreater(info['object_after']['chroma_c'], .64 * info['object_before']['chroma_c'])
        self.assertTrue(-16 <= info['lightness_offset'] <= 13)
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

    def test_render_environment_does_not_turn_background_gradient_into_key_light(self):
        x = np.linspace(60,170,240,dtype=np.uint8)
        pixels = np.repeat(np.broadcast_to(x,(240,240))[:,:,None],3,axis=2).copy()
        frame = Image.fromarray(pixels)
        samples = np.ones((240,240),bool)
        env = estimate_render_environment(frame, samples)
        self.assertEqual(env['status'], 'diffuse_estimate')
        self.assertEqual(len(env['world_color']), 3)
        self.assertTrue(all(0 <= v <= 1 for v in env['world_color']))
        self.assertGreaterEqual(env['world_strength'], .30)
        self.assertLessEqual(env['world_strength'], .78)
        self.assertIsNone(env['key_direction'])
        self.assertEqual(env['light_direction'], 'unknown')
        estimate = estimate_render_environment(frame, samples, {'status':'estimated','direction_screen':[.6,-.8]})
        self.assertEqual(estimate['key_direction'], [.6,-.8])
        self.assertEqual(estimate['confidence'], 'low')
        empty = estimate_render_environment(Image.new('RGB',(300,300),(80,80,80)),
                                            np.zeros((300,300),bool))
        self.assertEqual(empty['status'], 'insufficient_samples')
        self.assertEqual(empty['light_direction'], 'unknown')

    def test_quiet_photo_preserves_alpha_and_is_bounded(self):
        x = np.linspace(60,120,240,dtype=np.uint8)
        pixels = np.repeat(np.broadcast_to(x,(240,240))[:,:,None],3,axis=2).copy()
        frame = Image.fromarray(pixels)
        obj = Image.new('RGBA',(80,80),(150,120,100,0))
        alpha = np.zeros((80,80),np.uint8); alpha[10:70,10:70]=255
        obj.putalpha(Image.fromarray(alpha))
        support = np.ones((240,240),bool)
        nearby = np.ones((240,240),bool)
        result, evidence = quiet_photo(frame, support, nearby, obj, 80, 80, 80)
        self.assertTrue(np.array_equal(np.asarray(result.getchannel('A')), alpha))
        self.assertIn('background_lab', evidence)
        self.assertIn('accent_chroma_cap', evidence)

class ContactShadowChecks(unittest.TestCase):
    def test_shadow_is_bounded_to_support_and_keeps_alpha_input(self):
        support = np.zeros((200,200),bool)
        support[100:132,60:140] = True
        foreground = np.zeros_like(support)
        foreground[:,105:108] = True
        alpha = np.zeros((200,200),np.uint8)
        alpha[55:126,65:135] = 255
        footprint = np.asarray(projected_footprint([[[.37,.60],[.63,.60],[.63,.64],[.37,.64]]], 200)) > 0
        before = alpha.copy()
        layer, info = make_contact_shadow(footprint, support, foreground, alpha, 70)
        self.assertTrue(np.array_equal(alpha, before))
        self.assertIsNotNone(layer)
        self.assertEqual(info['status'], 'estimated')
        shadow_alpha = np.asarray(layer.getchannel('A'))
        self.assertGreater(int(shadow_alpha.sum()), 0)
        l,t,r,b = info['crop']
        allowed = (support & ~foreground & (alpha < 128))[t:b,l:r]
        self.assertFalse((shadow_alpha[~allowed] > 0).any())

    def test_shadow_refuses_missing_support_or_foreground(self):
        support = np.zeros((200,200),bool)
        footprint = support.copy()
        footprint[110:116, 80:130] = True
        alpha = np.zeros((200,200),np.uint8)
        layer, info = make_contact_shadow(footprint, support, np.zeros_like(support), alpha, 70)
        self.assertIsNone(layer)
        self.assertNotEqual(info['status'], 'estimated')
        support[50:150,40:160] = True
        foreground = np.ones((200,200),bool)
        layer, info = make_contact_shadow(footprint, support, foreground, alpha, 70)
        self.assertIsNone(layer)

    def test_each_disconnected_contact_patch_needs_support(self):
        footprint = np.zeros((120, 120), bool)
        footprint[85:89, 20:30] = True
        footprint[85:89, 90:100] = True
        support = np.zeros_like(footprint)
        support[80:100, :65] = True
        front = np.zeros_like(footprint)
        session = {'candidates': [{'id': 1}], 'selected': []}
        _, _, mask, scores, contact = server.review_contact(session, footprint, {1: support}, front)
        self.assertIsNone(mask)
        self.assertEqual(scores[1], 0)
        self.assertEqual(contact['patches'], 2)
        support[80:100, 80:110] = True
        ident, _, mask, _, contact = server.review_contact(session, footprint, {1: support}, front)
        self.assertEqual(ident, 1)
        self.assertIsNotNone(mask)
        self.assertEqual(contact['min_coverage'], 1)
        self.assertEqual(contact['physical_support'], 'unknown')


class ThinOcclusionChecks(unittest.TestCase):
    def test_foreground_matte_keeps_opaque_core_and_stays_inside_region(self):
        selected = np.zeros((80, 80), bool)
        selected[10:70, 30:36] = True
        matte = foreground_matte(selected)
        self.assertTrue((matte[15:65, 31:35] == 255).all())
        self.assertTrue((matte[15:65, 30] < 255).all())
        self.assertFalse(matte[~selected].any())

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
