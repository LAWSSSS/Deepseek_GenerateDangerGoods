"""Offline checks for state, masking, and loopback boundaries."""
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from starlette.testclient import TestClient
import server
from relight import ambient_match


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
        self.assertEqual(len(response.json()),39)
        self.assertEqual(sum(not r['passed'] for r in response.json()),15)
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


if __name__=='__main__':
    unittest.main()
