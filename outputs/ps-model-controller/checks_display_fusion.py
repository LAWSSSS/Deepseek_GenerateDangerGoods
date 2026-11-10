"""Regressions for photography, honest uncertainty, real silhouettes and two-way occlusion."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import unittest
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from display_batch.photography import match_linear,display,sensor_statistics
from display_batch.geometry import camera_for,project
from display_batch.pile import zbuffer,mesh_raster,receiver_mesh,depth_order,local_scale
from display_batch.schema import BatchConfig,Action,apply_actions,version_rank


class PhotoTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(3)
        frame=np.uint8(np.clip(120+rng.normal(0,2,(64,64,3)),0,255))
        mask=np.ones((64,64),bool);alpha=mask.astype(np.float32);alpha[20:30,20:30]=0
        rgb=np.full((64,64,3),[.4,.03,.015],np.float32)
        env={'sensor':sensor_statistics(frame,mask),'haze':.05}
        return frame,alpha,rgb,env

    def test_strong_color_reduction_keeps_alpha_hole(self):
        frame,alpha,rgb,env=self.fixture();original=alpha.copy()
        matched,info=match_linear(rgb,alpha,frame.reshape(-1,3),np.zeros_like(alpha),{},17,env)
        np.testing.assert_array_equal(alpha,original)
        self.assertLessEqual(info['chroma_gain'],.55)
        self.assertGreater(info['excess_chroma_reduction'],.5)
        self.assertTrue(np.isfinite(matched).all())

    def test_metal_highlight_and_paint_stay_different(self):
        frame,alpha,rgb,env=self.fixture();rgb[:,40:48]=2
        spec=np.zeros_like(alpha);spec[:,40:48]=1
        metal=spec.copy()
        matched,info=match_linear(rgb,alpha,frame.reshape(-1,3),spec,{},17,env,metallic=metal)
        self.assertGreater(matched[:,41:47].mean(),matched[:,3:12].mean()+.1)
        self.assertTrue(info['specular_protection'])

    def test_sensor_noise_is_not_scene_contrast(self):
        ramp=np.tile(np.linspace(20,220,128,dtype=np.float32),(128,1))
        frame=np.repeat(ramp[:,:,None],3,axis=2).astype(np.uint8)
        stats=sensor_statistics(frame,np.ones((128,128),bool))
        self.assertLess(stats['noise_sigma'],.003)

    def test_plastic_retains_surface_instead_of_becoming_metal_or_dust(self):
        frame,alpha,rgb,env=self.fixture()
        _,paint=match_linear(rgb,alpha,frame.reshape(-1,3),np.zeros_like(alpha),{},17,env)
        env['material_classes']=['plastic']
        _,plastic=match_linear(rgb,alpha,frame.reshape(-1,3),np.zeros_like(alpha),{},17,env)
        self.assertTrue(plastic['polymer_preservation'])
        self.assertLess(plastic['dust_mean'],paint['dust_mean']*.5)
        self.assertLess(plastic['surface_wear_rms'],paint['surface_wear_rms']*.5)

    def test_photo_controls_do_not_change_candidate(self):
        p=apply_actions({'candidate':'A'},[Action(kind='chroma',value='more'),Action(kind='dust',value='less')],['A'])
        self.assertEqual(p['candidate'],'A')
        with self.assertRaises(ValueError):apply_actions(p,[Action(kind='dust',value='unbounded')],['A'])


class PileTests(unittest.TestCase):
    def fixture(self):
        w=h=100;mask=np.zeros((h,w),bool);mask[55:90,10:30]=True;mask[55:90,70:90]=True
        sites=[]
        for i,(a,b) in enumerate(((10,30),(70,90))):
            m=np.zeros_like(mask);m[55:90,a:b]=True
            sites.append({'id':f'S{i}','mask':m,'anchor_px':[(a+b)//2,75],'depth':.5,
                'reference':[[a/99,70/99],[b/99,70/99]],'reference_mm':[100,200]})
        camera=camera_for(np.array([[-.5,-.2,0],[.5,.2,.3]]),45)
        scene={'size':[w,h],'supports':sites,'depth_native':np.full((h,w),.5,np.float32),
               'depth':np.full((h,w),.5,np.float32),'protected':np.zeros_like(mask),'frame':Image.new('RGB',(w,h),'gray')}
        candidate={'site_id':'S0','support_ids':['S0','S1'],'camera':camera,'offset':[0,0],
                   'full_px':100,'bounds':[0,0,100,100],'plane_normal':[0,0,1]}
        return scene,candidate

    def test_actual_silhouette_retains_hole(self):
        vertices=[];faces=[]
        for x1,y1,x2,y2 in [(-1,-1,-.4,1),(.4,-1,1,1),(-.4,-1,.4,-.4),(-.4,.4,.4,1)]:
            start=len(vertices);vertices.extend([[x1,y1,0],[x2,y1,0],[x2,y2,0],[x1,y2,0]])
            faces.extend([[start,start+1,start+2],[start,start+2,start+3]])
        camera={'eye':[0,0,5],'target':[0,0,0],'basis':[[1,0,0],[0,1,0],[0,0,-1]],'fov':40}
        mask,depth=mesh_raster(np.array(vertices,float),np.array(faces),camera,128)
        self.assertFalse(mask[64,64]);self.assertGreater(mask.sum(),1000)

    def test_crossing_depth_uses_per_pixel_z(self):
        xy=np.array([[2.,2.],[30.,2.],[2.,30.],[2.,2.],[30.,2.],[2.,30.]])
        z=np.array([1.,4.,4.,3.,3.,3.]);faces=np.array([[0,1,2],[3,4,5]])
        depths=zbuffer(xy,z,faces,32)
        self.assertLess(depths[4,4],3);self.assertAlmostEqual(float(depths[3,27]),3,places=5)

    def test_finite_receivers_do_not_bridge_gap(self):
        scene,candidate=self.fixture();receiver=receiver_mesh(scene,candidate)
        self.assertEqual(len(receiver['patches']),2)
        for patch in receiver['patches']:
            points=np.array(patch['vertices'])[np.array(patch['faces'])].mean(axis=1)
            pixels=project(points,candidate['camera'])*100
            self.assertFalse(((pixels[:,0]>35)&(pixels[:,0]<65)).any())

    def test_front_and_back_steel_can_coexist(self):
        scene,candidate=self.fixture();scene['depth_native'][:50,:50]=.85;scene['depth_native'][:50,50:]=.15
        alpha=np.zeros((100,100),np.float32);alpha[10:45,10:90]=1
        z=np.full((100,100),12,np.float32)
        front,info=depth_order(scene,candidate,candidate['bounds'],z,alpha)
        self.assertTrue(front[20,20]);self.assertFalse(front[20,70])
        self.assertGreater(info['occluded_fraction'],.3)

    def test_scale_is_computed_at_each_anchor_and_never_promoted(self):
        scene,candidate=self.fixture();scene['depth'][5,5]=.9
        _,near=local_scale(scene,scene['supports'][0],[20,75],[150,200])
        _,far=local_scale(scene,scene['supports'][0],[5,5],[150,200])
        self.assertEqual(near['status'],'pass');self.assertEqual(far['status'],'unknown')
        self.assertEqual(far['rejected_reference_reason'],'cross_depth_reference')

    def test_support_membership_does_not_erase_foreground(self):
        scene,candidate=self.fixture()
        scene['supports'][0]['mask'][10:35,10:35]=True
        scene['depth_native'][10:35,10:35]=.95
        alpha=np.zeros((100,100),np.float32);alpha[10:45,10:90]=1
        front,info=depth_order(scene,candidate,candidate['bounds'],np.full((100,100),12,np.float32),alpha)
        self.assertTrue(front[20,20]);self.assertFalse(front[20,70])

    def test_unknown_is_better_than_fail_but_never_pass(self):
        unknown={'support':{'status':'unknown'}};failed={'support':{'status':'fail'}}
        self.assertGreater(version_rank(unknown,None),version_rank(failed,None))
        config=BatchConfig(backgrounds='x',output='y')
        self.assertEqual(config.uncertain_policy,'render_review');self.assertEqual(config.photo_style,'scrap_gray')


if __name__=='__main__':unittest.main(verbosity=2)
