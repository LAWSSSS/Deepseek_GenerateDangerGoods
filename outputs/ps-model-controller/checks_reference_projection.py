"""Offline regression: evidence provenance, metric projection, strict eligibility and RGB budgets."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import cv2
import numpy as np
from display_batch.schema import BatchConfig, Scores, Defect, Review
from display_batch.specifications import profile
from display_batch.reference_projection import fit_plane, audit, local_scale, metric_projection
from display_batch.strict_contact import retryable, visual_veto, simulate, validate_receiver, projected_gap
from display_batch.metric_physics import tolerances
from display_batch.chroma_budget import masks, budget, measure, compress, lab_of
from display_batch.geometry import project
from display_batch.pile import receiver_mesh
from display_batch.pipeline import Pipeline


def reference():
    K=np.array([[1000.,0,800],[0,1000,500],[0,0,1]])
    R=cv2.Rodrigues(np.array([2.4,0.,0.]))[0];eye=np.array([0,-1.,1.2]);t=-R@eye
    p=np.array([[-200,-100],[200,-100],[200,100],[-200,100],[0,0]],float)
    xyz=np.column_stack([p/1000,np.zeros(len(p))])
    q=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],t,K,np.zeros(5))[0][:,0]
    row={'entity_id':'measured_plate','surface_id':'S1','feature':'known_rectangle_corners',
         'identity_status':'confirmed','unit':'mm','source':{'kind':'measured','reference':'test_fixture','sha256':'fixture'},
         'plane_points_mm':p.tolist(),'image_points_px':q.tolist()}
    return row,{'status':'measured','matrix':K.tolist()}


class ScaleTests(unittest.TestCase):
    def test_exact_model_feature_and_source_required(self):
        asset={'mesh_hash':'x','semantic':{'major_mm':[450,900]}}
        ident={'confidence':'clear','brand':'A','model':'B','feature':'body_length','extension':'extended'}
        source={'url':'https://manufacturer.invalid/table','sha256':'hash','page':1}
        cat={'specifications':[{'brand':'A','model':'B','feature':'body_length','length_mm':600,'source':source}]}
        self.assertEqual(profile(asset,ident,cat)['nominal_mm'],600)
        for change in ({'model':'B1'},{'feature':'extended_total_length'},{'confidence':'partial'}):
            p=profile(asset,{**ident,**change},cat);self.assertEqual(p['status'],'unknown')
            self.assertAlmostEqual(p['nominal_mm'],np.sqrt(450*900))
        self.assertEqual(profile(asset,ident,cat)['user_factor'],1)

    def test_single_line_never_two_dimensional_calibration(self):
        row,cam=reference();row['plane_points_mm']=row['plane_points_mm'][:2];row['image_points_px']=row['image_points_px'][:2]
        self.assertEqual(fit_plane([row],cam)['status'],'unknown')

    def test_wrong_source_duplicate_conflict_and_unknown_camera(self):
        row,cam=reference()
        self.assertEqual(fit_plane([{**row,'identity_status':'similar_product'}],cam)['status'],'unknown')
        self.assertEqual(fit_plane([row,row],cam)['reason'],'duplicate_calibration_points')
        bad=copy.deepcopy(row);bad['image_points_px'][-1][0]+=80
        self.assertEqual(fit_plane([bad],cam)['status'],'unknown')
        self.assertEqual(fit_plane([row])['status'],'unknown')

    def test_metric_camera_and_near_far_projection(self):
        row,cam=reference();fit=fit_plane([row],cam);self.assertEqual(fit['status'],'pass')
        self.assertLess(fit['residual_px'],.01)
        original=fit['metric_camera'];K=np.asarray(original['K']);R=np.asarray(original['R']);t=np.asarray(original['t_m'])
        spans=[]
        for y in (-.1,.4):
            anchor=cv2.projectPoints(np.array([[0.,y,0.]]),cv2.Rodrigues(R)[0],t,K,np.zeros(5))[0][0,0]
            camera,full=metric_projection(fit,anchor,200,[1600,1000],[0,0,0])
            points=np.array([[-.5,0,0],[.5,0,0],[0,0,.2]])
            expected=cv2.projectPoints(points*.2+[0,y,0],cv2.Rodrigues(R)[0],t,K,np.zeros(5))[0][:,0]
            projected=project(points,camera)*full
            projected+=anchor-project(np.zeros((1,3)),camera)[0]*full
            np.testing.assert_allclose(projected,expected,atol=.001)
            spans.append(np.linalg.norm(projected[0]-projected[1]))
        self.assertGreater(spans[0],spans[1])

    def test_cross_depth_and_inherited_references_do_not_become_measured(self):
        sites=[{'id':'S1','depth':.1,'reference':[[.1,.4],[.3,.4]],'reference_mm':[100,300]},
               {'id':'S2','depth':.8,'reference':[[.5,.5],[.8,.5]],'reference_mm':[100,300]},
               {'id':'D1','depth':.7,'reference':[[.1,.4],[.3,.4]],'reference_mm':[100,300],
                'plane_basis':'segmentation_only_hidden_height_unknown'}]
        scene={'key':'k','size':[1000,500],'supports':sites,'answer':{'hopper':[[0,0],[1,0],[1,1]],'elevation':[35,55]}}
        scene['reference_audit']=audit(scene)
        _,evidence=local_scale(scene,sites[0],[200,200],{'nominal_mm':200,'status':'pass'})
        self.assertEqual(evidence['status'],'unknown');self.assertEqual(len(evidence['references']),1)
        self.assertIn('cross_depth_reference',[r['reason'] for r in evidence['rejected_references']])
        self.assertEqual(len(scene['reference_audit']['references']),2)


class PhysicalTests(unittest.IsolatedAsyncioTestCase):
    async def test_contact_clusters_keep_independent_heights(self):
        from display_batch.contact_geometry import patches, description
        from display_batch.geometry import camera_for
        points=np.array([[-.4,-.1,0],[-.2,-.1,0],[-.3,.1,0],[.2,-.1,.2],[.4,-.1,.2],[.3,.1,.2]])
        faces=np.array([[0,1,2],[3,4,5]])
        receiver={'patches':[{'vertices':[[-1,-1,z],[1,-1,z],[1,1,z],[-1,1,z]],'faces':[[0,1,2],[0,2,3]]} for z in (0,.2)]}
        world=patches(points,faces,receiver,.001)
        self.assertEqual(len(world),2)
        self.assertAlmostEqual(world[0][:,2].mean(),0);self.assertAlmostEqual(world[1][:,2].mean(),.2)
        camera=camera_for(points,45);polys,anchor=description(points,faces,camera,receiver,.001)
        for resolution in (512,1024,2048):
            np.testing.assert_allclose(anchor, (anchor*resolution)/resolution)
        self.assertEqual(len(polys),2)

    async def test_only_unconverged_retries_once_and_cache_reuses(self):
        with tempfile.TemporaryDirectory() as folder:
            asset={'folder':folder,'geometry':'g','axis':[1,0,0],'physical_size':{'nominal_mm':200,'signature':'dimension'},
                   'physics_signature':'p','semantic':{'assembly':'single_piece'}}
            def pose(codes):return {'stable':not codes,'failure_codes':codes,'id':'P0'}
            worker=type('Worker',(),{'ask':AsyncMock(side_effect=[{'poses':[pose(['not_converged'])]},
                                                                {'poses':[pose(['not_converged'])]}])})()
            result=await simulate(asset,worker,[np.eye(4).tolist()])
            self.assertFalse(result['poses'][0]['stable']);self.assertEqual(worker.ask.await_count,2)
            cached=await simulate(asset,worker,[np.eye(4).tolist()]);self.assertTrue(cached['cache_hit'])
            self.assertEqual(worker.ask.await_count,2)
        for codes in [['floating'],['ground_penetration'],['not_converged','floating']]:
            self.assertFalse(retryable({'failure_codes':codes}))

    async def test_different_heights_never_flattened(self):
        candidate={'bounds':[0,0,100,100],'camera':{'eye':[0,-2,2],'basis':[[1,0,0],[0,.707,.707],[0,.707,-.707]],'fov':45},
            'offset':[0,0],'full_px':100,'site_id':'S1','support_ids':['S1','S2'],'strict_contact':True,
            'scale':{'physical_size':{'nominal_mm':1000}}}
        scene={'supports':[{'id':'S1','mask':np.ones((100,100),bool),'depth':.5},
                           {'id':'S2','mask':np.ones((100,100),bool),'depth':.5}]}
        self.assertFalse(receiver_mesh(scene,candidate)['patches'])
        for site,z in zip(scene['supports'],[0,.2]):
            site.update(plane_status='measured',plane_origin_m=[0,0,z],plane_normal=[0,0,1])
        result=receiver_mesh(scene,candidate)
        self.assertEqual(result['height_confidence'],'measured')
        self.assertAlmostEqual(np.array(result['patches'][0]['vertices'])[:,2].mean(),0)
        self.assertAlmostEqual(np.array(result['patches'][1]['vertices'])[:,2].mean(),.2)

    async def test_visible_physical_veto_and_pixel_gap(self):
        review=Review(scores=Scores(**dict.fromkeys(['support','occlusion','lighting','edge','texture','material','structure'],4)),
            actions=[],evidence='test',defects=[Defect(code='floating',severity='major',region=[0,0,1,1],evidence='visible gap')])
        self.assertTrue(visual_veto(review))
        review.defects[0].severity='minor';self.assertTrue(visual_veto(review))
        review.defects[0].visibility='unassessable';self.assertFalse(visual_veto(review))
        camera={'eye':[0,0,2],'basis':[[1,0,0],[0,1,0],[0,0,-1]],'fov':45}
        pose={'contact_points':[[0,.1,0]],'receiver_points':[[0,0,0]]}
        self.assertEqual(projected_gap(pose,{'camera':camera,'full_px':1000})['status'],'fail')
        self.assertLess(tolerances(.1)['contact_m'],tolerances(1)['contact_m'])

    async def test_failed_version_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as f:
            with self.assertRaises(ValueError):Pipeline.finish_image(Path(f),'id',{'status':'review'},{'deliverable':False})


class ChromaTests(unittest.TestCase):
    def test_ai_cannot_repeatedly_lower_computed_color_budget(self):
        from display_batch.review import allowed_actions
        from display_batch.schema import PHOTO_CONTROLS
        params={'candidate':'A','material':'original','light':'scene','tone':'scene','photo_model':'local',
                'chroma_mode':'local_budget',**{k:'auto' for k in PHOTO_CONTROLS}}
        options=allowed_actions(params,{'photo':{}},['A','B'],render_style='local')
        self.assertNotIn('chroma',options)
    def test_paint_highlight_texture_alpha_labels_and_final_quantiles(self):
        rgb=np.zeros((60,120,3),np.float32);rgb[:,:40]=[.8,.1,.09];rgb[:,40:80]=[.85,.75,.05];rgb[:,80:]=[.05,.2,.8]
        rgb[0:10]=.9;rgb[40:50]*=1.1
        alpha=np.ones((60,120),np.float32);alpha[25:30,50:60]=0
        source=rgb.copy();groups=masks(rgb,alpha,np.zeros_like(alpha),['steel','paint'])
        limits=budget(np.full((200,3),100,np.uint8));out,evidence=compress(rgb,groups,limits,alpha)
        self.assertEqual(measure(np.round(out*255)/255,groups,limits)['status'],'pass')
        np.testing.assert_array_equal(out[alpha==0],source[alpha==0]);np.testing.assert_array_equal(out[:10],source[:10])
        self.assertLess(np.max(abs(lab_of(out)[...,0]-lab_of(source)[...,0])),.5)
        a=lab_of(out)[10,10,1:];b=lab_of(source)[10,10,1:]
        self.assertGreater(float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),.99)
        self.assertGreater(out[44,10].mean(),out[20,10].mean())
        self.assertEqual(evidence['groups']['paint']['pixels'],5950)

    def test_nonmetal_no_forced_metal_or_color_change(self):
        rgb=np.full((40,40,3),[.7,.1,.1],np.float32);alpha=np.ones((40,40),np.float32)
        groups=masks(rgb,alpha,None,['plastic','rubber']);self.assertEqual(groups,{})
        out,_=compress(rgb,groups,budget(np.full((60,3),90)),alpha)
        np.testing.assert_array_equal(rgb,out)

    def test_new_controls_opt_in_legacy_scale_kept(self):
        old=BatchConfig(backgrounds='x',output='y');self.assertEqual(old.scale_mode,'legacy')
        new=BatchConfig(backgrounds='x',output='y',scale_mode='reference_projection',contact_mode='strict',chroma_mode='local_budget')
        self.assertEqual(new.asset_scale_factors['31'],.375) # replay metadata retained, new profile never reads it


class StrictLoopTests(unittest.IsolatedAsyncioTestCase):
    async def loop(self,reviews,alternatives=3):
        from checks_display_batch import LoopTests
        original=BatchConfig
        candidates=[{'id':chr(65+i),'target_px':200,'score':3-i} for i in range(alternatives)]
        with patch('checks_display_batch.BatchConfig',side_effect=lambda **kw:original(**kw,contact_mode='strict')),\
             patch.object(Pipeline,'choose_placements',new=AsyncMock(return_value=(candidates,[]))),\
             patch('display_batch.pipeline.file_hash',side_effect=lambda p:Path(p).stem),\
             patch('display_batch.pipeline.diagnostic'):
            return await LoopTests().run_loop(2,reviews,alternatives=alternatives)

    async def test_three_floating_images_never_delivered(self):
        from checks_display_batch import review
        bad=review(4,defects=[Defect(code='floating',severity='major',region=[0,0,1,1],evidence='clear air below foot')])
        result,calls=await self.loop([bad])
        self.assertEqual(result['status'],'failed');self.assertNotIn('image',result)
        self.assertEqual(calls,3);self.assertEqual(result['candidate_visits'],['A','B','C'])
        self.assertTrue(all(e['deliverable'] is False for e in result['iterations']))

    async def test_only_valid_backup_can_be_selected(self):
        from checks_display_batch import review
        bad=review(4,defects=[Defect(code='penetration',severity='major',region=[0,0,1,1],evidence='crosses steel')])
        result,calls=await self.loop([bad,review(3)])
        self.assertEqual(calls,2);self.assertEqual(result['best_iteration'],1)
        self.assertEqual(result['status'],'accepted_display')


if __name__=='__main__':unittest.main(verbosity=2)
