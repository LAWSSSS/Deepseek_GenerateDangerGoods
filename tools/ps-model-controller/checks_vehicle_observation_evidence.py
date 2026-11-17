"""Observed geometry fixtures only. Never calls a service, physics or a renderer."""
import asyncio
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch

import cv2
import numpy as np

from display_batch.vehicle_scale import (ObservedVehicleGeometry,VehicleGeometry,basis,
    audit_width_observations,fit_observed_geometry,prepare,projection_preflight)


def fixture():
    size=np.array([1600,1000]);f=1600/(2*math.tan(math.radians(25)))
    b=basis(55,8);R=np.stack([b[0],-b[1],b[2]])
    K=np.array([[f,0,800],[0,f,500],[0,0,1.]])
    points=[]
    for y in (2.5,3.5,4.5):
        q=(np.array([[-1.225,y,0],[1.225,y,0]])-[0,0,4.])@R.T@K.T
        points.append((q[:,:2]/q[:,2,None]/(size-1)).tolist())
    rims=[[points[0][0],points[-1][0]],[points[0][1],points[-1][1]]]
    observed={'vehicle_class':'semitrailer','confidence':.8,'rim_visibility':'clear',
        'left_rim':rims[0],'right_rim':rims[1],
        'width_observations':[{'points':p,'endpoint_visibility':['visible','visible'],
            'correspondence_basis':'opposing_posts','evidence':f'Observed opposing posts at station {i}'}
            for i,p in enumerate(points)],'heights':[],'evidence':'synthetic visible rim fixture'}
    edges={'rims':[{'proposed':line,'refined':line,'supported':True} for line in rims],
           'both_supported':True}
    return observed,size,edges


class VehicleObservationEvidenceTests(unittest.TestCase):
    def test_pipeline_replaces_legacy_geometry_before_expensive_steps(self):
        from display_batch.pipeline import Pipeline
        observed,size,edges=fixture()
        observed['width_observations']=[]
        fit=fit_observed_geometry(observed,size,2.45,55,[50,60],edges)
        replacement={'observation_mode':'audited','calibration':fit,
                     'projection_preflight':fit['projection_preflight']}
        scene={'key':'fixture','vehicle_scale':{'observation_mode':'legacy','calibration':{'projectable':True}}}
        with tempfile.TemporaryDirectory() as temp:
            pipeline=object.__new__(Pipeline)
            pipeline.config=SimpleNamespace(geometry_preset='rebuild',integration_mode='coherent',vehicle_reference={'fixture':True})
            pipeline.out=Path(temp);pipeline.cache=Path(temp)/'cache';pipeline.vision=object()
            with patch('display_batch.vehicle_scale.prepare',new=AsyncMock(return_value=replacement)) as prepare_new:
                candidates,rejections=asyncio.run(pipeline.choose_placements({},scene,123))
            self.assertEqual(candidates,[])
            self.assertEqual(prepare_new.await_args.kwargs,{'observation_mode':'audited'})
            self.assertEqual(scene['vehicle_scale']['observation_mode'],'audited')
            self.assertEqual(rejections[0]['reason'],'high_risk_vehicle_projection')

    def test_absent_rims_is_valid_output_and_does_not_fit_an_invented_camera(self):
        observed=ObservedVehicleGeometry(vehicle_class='semitrailer',confidence=.3,
            rim_visibility='inferred',width_observations=[],left_rim=None,right_rim=None,
            heights=[],evidence='Both rim endpoints hidden by scrap.').model_dump()
        with patch('display_batch.vehicle_scale.camera_fit',side_effect=AssertionError('must not fit')) as fit:
            result=fit_observed_geometry(observed,[3840,2160],2.45,35,[25,45],{'rims':[],'both_supported':False})
        fit.assert_not_called()
        self.assertFalse(result['projectable'])
        self.assertEqual(result['correspondence_status'],'unavailable')
        self.assertNotIn('metric_camera',result)
        self.assertFalse(result['projection_preflight']['allow_expensive_steps'])

    def test_valid_observed_spans_fit_without_changing_coordinates(self):
        observed,size,edges=fixture();before=copy.deepcopy(observed)
        fit=fit_observed_geometry(observed,size,2.45,55,[50,60],edges)
        self.assertTrue(fit['projectable'])
        self.assertEqual(fit['correspondence_status'],'observed_unmeasured')
        self.assertEqual(fit['projection_preflight']['state'],'renderable_estimate')
        self.assertLess(fit['residual_px'],1.)
        self.assertFalse(fit['observation_audit']['coordinates_modified'])
        self.assertFalse(fit['projection_preflight']['can_certify_real_scale'])
        self.assertEqual(observed,before)

    def test_continuous_rims_do_not_promote_wrong_width_endpoints(self):
        observed,size,edges=fixture()
        invalid=copy.deepcopy(observed['width_observations'][1])
        invalid['points'][0][0]+=.09
        observed['width_observations'].append(invalid)
        result=fit_observed_geometry(observed,size,2.45,55,[50,60],edges)
        audit=result['observation_audit']
        self.assertEqual(audit['accepted_indices'],[0,1,2])
        self.assertIn('width_endpoint_off_visible_inside_rim',audit['observations'][3]['reasons'])
        self.assertLess(result['residual_px'],1.)
        self.assertEqual(len(result['image_points_px']),3)

    def test_hidden_endpoints_unknown_correspondence_and_duplicates_not_rulers(self):
        observed,size,edges=fixture()
        observed['width_observations'][0]['endpoint_visibility'][0]='occluded'
        observed['width_observations'][1]['correspondence_basis']='unknown'
        observed['width_observations'].append(copy.deepcopy(observed['width_observations'][2]))
        audit=audit_width_observations(observed,size,edges)
        self.assertEqual(audit['accepted_indices'],[2])
        self.assertFalse(audit['fit_ready'])
        self.assertIn('duplicate_transverse_section',audit['observations'][3]['reasons'])

    def test_crossed_pairings_rejected_and_reversed_line_endpoint_order_supported(self):
        observed,size,edges=fixture()
        reverse=copy.deepcopy(edges);reverse['rims'][1]['refined'].reverse()
        self.assertTrue(audit_width_observations(observed,size,reverse)['fit_ready'])
        a=observed['width_observations']
        a[0]['points'][1],a[2]['points'][1]=a[2]['points'][1],a[0]['points'][1]
        result=audit_width_observations(observed,size,edges)
        self.assertFalse(result['fit_ready'])
        self.assertTrue(all('crossed_transverse_correspondences' in row['reasons'] for row in result['observations']))

    def test_unknown_correspondence_is_not_upgraded_by_zero_fit_residual(self):
        observed,size,edges=fixture()
        for row in observed['width_observations']:row['correspondence_basis']='unknown'
        result=fit_observed_geometry(observed,size,2.45,55,[50,60],edges)
        self.assertNotIn('homography_mm_to_px',result)
        self.assertFalse(result['projection_preflight']['allow_expensive_steps'])

    def test_audited_prior_fallback_is_not_a_measured_camera(self):
        observed,size,edges=fixture()
        fit=fit_observed_geometry(observed,size,2.45,55,[50,60],edges)
        fit['fit_health']['mode']='explicit_camera_prior_fallback'
        gate=projection_preflight(fit,2.45)
        self.assertIn('observed_fit_requires_unmeasured_camera_prior',gate['risk_codes'])

    def test_audited_missing_evidence_cached_separately_from_legacy_schema(self):
        observed,size,edges=fixture()
        answer=ObservedVehicleGeometry(vehicle_class='semitrailer',confidence=.3,
            rim_visibility='inferred',width_observations=[],left_rim=None,right_rim=None,
            heights=[],evidence='No observed inside-rim correspondence.')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'scene.png'
            cv2.imencode('.png',np.full((1000,1600,3),120,np.uint8))[1].tofile(path)
            vision=SimpleNamespace(ask=AsyncMock(return_value=answer),endpoint=SimpleNamespace(model_dump=lambda:{}))
            scene={'path':str(path),'key':'fixture','size':size.tolist(),
                'answer':{'elevation':[50,60]},'supports':[]}
            reference={'vehicle_class':'semitrailer','internal_width_m':2.45,
                       'source':{'kind':'user_supplied','reference':'fixture'}}
            first=asyncio.run(prepare(scene,reference,vision,root/'cache',observation_mode='audited'))
            second=asyncio.run(prepare(scene,reference,vision,root/'cache',observation_mode='audited'))
            self.assertEqual(first,second)
            vision.ask.assert_awaited_once()
            self.assertIs(vision.ask.call_args.args[1],ObservedVehicleGeometry)
            self.assertEqual(first['observation_mode'],'audited')
            legacy=VehicleGeometry(vehicle_class='semitrailer',confidence=.8,rim_visibility='clear',
                width_spans=[r['points'] for r in observed['width_observations']],
                left_rim=observed['left_rim'],right_rim=observed['right_rim'],heights=[],evidence='fixture')
            vision.ask=AsyncMock(return_value=legacy)
            with patch('display_batch.vehicle_scale.rim_edge_evidence',return_value=edges):
                old=asyncio.run(prepare(scene,reference,vision,root/'cache'))
            self.assertIs(vision.ask.call_args.args[1],VehicleGeometry)
            self.assertNotEqual(first['signature'],old['signature'])
            self.assertEqual(old['observation_mode'],'legacy')
            self.assertTrue(old['calibration']['projectable'])


if __name__=='__main__':unittest.main(verbosity=2)
