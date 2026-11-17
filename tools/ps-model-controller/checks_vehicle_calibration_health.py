"""Offline calibration health checks; no scene or review API calls."""
import math
import asyncio
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock,patch
import cv2
import numpy as np
from display_batch.vehicle_scale import (basis,camera_fit,rim_edge_evidence,projection_preflight,
                                        bound_hits,prepare,VehicleGeometry,local_scale)


def consistent_calibration():
    size=np.array([1600,1000]);f=1600/(2*math.tan(math.radians(50)/2))
    b=basis(55,8);R=np.stack([b[0],-b[1],b[2]])
    K=np.array([[f,0,800],[0,f,500],[0,0,1.]])
    eye=np.array([0.,0.,4.]);spans=[]
    for y in (1.,2.,3.):
        points=np.array([[-1.225,y,0],[1.225,y,0]])
        q=(points-eye)@R.T@K.T
        spans.append((q[:,:2]/q[:,2,None]/(size-1)).tolist())
    rims=[[spans[0][0],spans[-1][0]],[spans[0][1],spans[-1][1]]]
    return camera_fit(spans,size,2.45,55,[50,60],rim_lines=rims)


class VehicleCalibrationHealthTests(unittest.TestCase):
    def test_metric_parallel_rims_recover_perspective(self):
        fit=consistent_calibration()
        self.assertLess(fit['residual_px'],1.)
        self.assertEqual(fit['fit_health']['fallback_reasons'],[])
        self.assertEqual(fit['status'],'unknown')
        self.assertTrue(fit['fit_health']['rim_line_constraints'])
        self.assertEqual(fit['projection_preflight']['state'],'renderable_estimate')
        self.assertFalse(fit['projection_preflight']['can_certify_real_scale'])

    def test_constant_inferred_width_does_not_force_orthographic_lens(self):
        spans=[[[.19,y],[.90,y]] for y in (.17,.41,.65,.9)]
        fit=camera_fit(spans,[3840,2160],2.45,35,[25,45],allow_inferred_prior=True)
        health=fit['fit_health']
        self.assertIn('focal_length',health['unconstrained_bound_hits'])
        self.assertEqual(health['mode'],'single_inferred_width_with_explicit_camera_prior')
        self.assertEqual(health['scale_anchor_index'],1)
        self.assertAlmostEqual(fit['metric_camera']['K'][0][0],3840/(2*math.tan(math.radians(25))),places=4)
        self.assertEqual(fit['elevation_deg'],35)
        self.assertEqual(fit['status'],'unknown')
        self.assertIn('inferred_width_spans_conflict',health['fallback_reasons'])
        self.assertGreater(health['maximum_relative_width_residual'],.18)
        self.assertEqual(fit['projection_preflight']['state'],'high_risk_projection')
        self.assertIn('inconsistent_width_correspondences',fit['projection_preflight']['risk_codes'])

    def test_conflicting_observed_spans_do_not_use_inferred_escape(self):
        spans=[[[.19,y],[.90,y]] for y in (.17,.41,.65,.9)]
        with self.assertRaisesRegex(ValueError,'correspondences_conflict'):
            camera_fit(spans,[3840,2160],2.45,35,[25,45])

    def test_edge_support_requires_long_continuous_visible_rims(self):
        size=np.array([640,480]);rims=[[[.15,.1],[.25,.9]],[[.85,.1],[.75,.9]]]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'frame.png'
            frame=np.zeros((size[1],size[0],3),np.uint8)
            for points in rims:
                p=np.rint(np.asarray(points)*(size-1)).astype(int)
                cv2.line(frame,tuple(p[0]),tuple(p[1]),(230,230,230),2)
            cv2.imencode('.png',frame)[1].tofile(path)
            clear=rim_edge_evidence(path,*rims,'partial')
            self.assertTrue(clear['both_supported'])
            self.assertTrue(all(r['edge_coverage']>=.55 for r in clear['rims']))
            inferred=rim_edge_evidence(path,*rims,'inferred')
            self.assertFalse(inferred['both_supported'])
            frame[:]=120;cv2.imencode('.png',frame)[1].tofile(path)
            self.assertFalse(rim_edge_evidence(path,*rims,'clear')['both_supported'])

    def test_pose_bounds_cover_roll_yaw_and_fixed_endpoint_probe(self):
        lower=[30,-45,-65,1.,0.];upper=[55,45,65,3.,4.]
        self.assertEqual(bound_hits([30,45,65,2.,2.],lower,upper),['elevation','roll','yaw'])
        # Intentionally fixed endpoint probes are not an inferred elevation fit.
        self.assertEqual(bound_hits([30,0,0,2.,2.],[30,-45,-65,1.,0.],[30.1,45,65,3.,4.]),[])

    def test_diagonal_inferred_sections_detect_pose_bounds_without_new_camera(self):
        spans=[[[.04,.06],[.98,.62]],[[.06,.18],[.98,.64]],
               [[.07,.30],[.98,.66]],[[.09,.45],[.98,.68]]]
        fit=camera_fit(spans,[3840,2160],2.45,42.5,[30,55],allow_inferred_prior=True)
        self.assertAlmostEqual(fit['roll_deg'],45.,places=5)
        self.assertAlmostEqual(fit['elevation_deg'],30.,places=5)
        self.assertEqual(fit['fit_health']['mode'],'width_and_parallel_rims_fit')
        self.assertEqual(fit['fit_health']['fallback_reasons'],[])
        gate=fit['projection_preflight']
        self.assertFalse(gate['allow_expensive_steps'])
        self.assertIn('multiple_orientation_bounds',gate['risk_codes'])
        self.assertIn('unobserved_direction_at_compound_bounds',gate['risk_codes'])
        self.assertEqual(gate['status'],'unknown')

    def test_old_records_read_only_and_missing_directions_not_upgraded(self):
        calibration=consistent_calibration();calibration.pop('projection_preflight')
        calibration.pop('fit_health');calibration['rim_visibility']='inferred'
        original=copy.deepcopy(calibration)
        gate=projection_preflight(calibration,2.45)
        self.assertEqual(gate['state'],'renderable_estimate')
        self.assertIn('camera_direction_without_supported_image_rims',gate['limitations'])
        self.assertEqual(calibration,original)
        self.assertFalse(gate['historical_results_reclassified'])
        calibration['roll_deg']=45.;calibration['elevation_deg']=50.
        gate=projection_preflight(calibration,2.45)
        self.assertEqual(gate['state'],'high_risk_projection')
        self.assertIn('multiple_orientation_bounds',gate['risk_codes'])

    def test_gate_runs_before_local_scale_or_expensive_candidate_work(self):
        calibration=consistent_calibration();calibration['vehicle_yaw_deg']=65.
        scene={'vehicle_scale':{'calibration':calibration,'reference':{'internal_width_m':2.45},
                                'enforce_projection_preflight':True}}
        with self.assertRaisesRegex(ValueError,'high_risk_vehicle_projection:final_orientation'):
            local_scale(scene,{'id':'unused'},[0,0],{})
        for broken in ({},dict(calibration,metric_camera={'K':[[0.]],'R':[[0.]]})):
            self.assertFalse(projection_preflight(broken,2.45)['allow_expensive_steps'])

    def test_legacy_scale_replay_records_risk_without_enforcing_new_gate(self):
        calibration=consistent_calibration();calibration['vehicle_yaw_deg']=65.
        scene={'size':[1600,1000],'vehicle_scale':{'signature':'fixture','calibration':calibration,
            'reference':{'internal_width_m':2.45},'support_heights':{
                'S1':{'nominal_m':.5,'interval_m':[.4,.6],'status':'estimated'}}}}
        interval,evidence=local_scale(scene,{'id':'S1'},[800,500],{'nominal_mm':206})
        self.assertGreater(interval[0],0.)
        self.assertEqual(evidence['projection_preflight']['state'],'high_risk_projection')
        self.assertEqual(evidence['status'],'unknown')

    def test_pipeline_preflight_stops_before_asset_identification_or_physics(self):
        from display_batch.pipeline import Pipeline
        calibration=consistent_calibration();calibration['roll_deg']=45.
        gate=projection_preflight(calibration,2.45)
        scene={'key':'fixture','vehicle_scale':{'calibration':calibration,'projection_preflight':gate,
            'observation_mode':'audited'}}
        with tempfile.TemporaryDirectory() as temp:
            pipeline=object.__new__(Pipeline)
            pipeline.config=SimpleNamespace(geometry_preset='rebuild',integration_mode='coherent')
            pipeline.out=Path(temp)
            pipeline.vision=SimpleNamespace(ask=AsyncMock(side_effect=AssertionError('API must not run')))
            pipeline.blender=SimpleNamespace(ask=AsyncMock(side_effect=AssertionError('GPU must not run')))
            with patch('display_batch.specifications.identify',new=AsyncMock(side_effect=AssertionError('asset identification must not run'))) as identify,\
                 patch('display_batch.strict_contact.simulate',new=AsyncMock(side_effect=AssertionError('physics must not run'))) as simulate,\
                 patch('display_batch.pipeline.choose_candidates',side_effect=AssertionError('candidate search must not run')) as choose:
                candidates,rejections=asyncio.run(pipeline.choose_placements({},scene,123))
            self.assertEqual(candidates,[])
            self.assertEqual(rejections[0]['reason'],'high_risk_vehicle_projection')
            self.assertEqual(rejections[0]['evidence']['state'],'high_risk_projection')
            identify.assert_not_awaited();simulate.assert_not_awaited();choose.assert_not_called()
            pipeline.vision.ask.assert_not_awaited();pipeline.blender.ask.assert_not_awaited()
            self.assertTrue((Path(temp)/'scale-scenes/fixture.json').is_file())

    def test_preparation_carries_gate_without_promoting_unseen_rims(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'scene.png'
            cv2.imencode('.png',np.full((360,640,3),120,np.uint8))[1].tofile(path)
            spans=[[[.19,y],[.90,y]] for y in (.17,.41,.65,.9)]
            answer=VehicleGeometry(vehicle_class='semitrailer',confidence=.5,rim_visibility='clear',
                width_spans=spans,left_rim=((.19,.17),(.19,.9)),right_rim=((.9,.17),(.9,.9)),
                heights=[],evidence='synthetic proposed rims; image is uniform')
            vision=SimpleNamespace(ask=AsyncMock(return_value=answer),endpoint=SimpleNamespace(model_dump=lambda:{}))
            scene={'path':str(path),'key':'fixture','size':[640,360],
                   'answer':{'elevation':[25.,45.]},'supports':[{'id':'S1','anchor_px':[320,180],'label':'plate'}]}
            reference={'vehicle_class':'semitrailer','internal_width_m':2.45,
                       'source':{'kind':'user_supplied','reference':'fixture'}}
            value=asyncio.run(prepare(scene,reference,vision,root/'cache'))
            self.assertEqual(value['calibration']['correspondence_status'],'inferred')
            self.assertEqual(value['projection_preflight']['state'],'high_risk_projection')
            self.assertEqual(value['status'],'unknown')
            self.assertEqual(value['calibration']['projection_preflight'],value['projection_preflight'])
            vision.ask.assert_awaited_once()


if __name__=='__main__':unittest.main(verbosity=2)
