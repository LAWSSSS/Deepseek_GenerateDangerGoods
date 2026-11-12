"""Regression for supplied-width projection, contact geometry and rebuild presets."""
import copy
import tempfile
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
import trimesh
from display_batch.common import file_hash,signature
from display_batch.schema import BatchConfig
from display_batch.collision import build
from display_batch.contact_audit import contact_witnesses,support
from display_batch.vehicle_scale import camera_fit,plane_camera,local_scale,basis,validate_reference,check_projection_interval
from display_batch.reference_projection import metric_projection
from display_batch.geometry import project
from display_batch.scene import projection_geometry,fitted_masks
from display_batch.framing import render_windows
from validate_geometry_rebuild import REFERENCE


def calibration():
    size=[1600,1000];f=1600/(2*np.tan(np.radians(25)))
    b=basis(55,0);R=np.stack([b[0],-b[1],b[2]])
    eye=np.array([0.,0.,4.]);K=np.array([[f,0,800],[0,f,500],[0,0,1.]])
    spans=[]
    for y in [1.,2.,3.]:
        p=np.array([[-1.225,y,0],[1.225,y,0]])
        q=(p-eye)@R.T@K.T;spans.append((q[:,:2]/q[:,2,None]/np.subtract(size,1)).tolist())
    return camera_fit(spans,size,2.45,55,[50,60])


class GeometryRebuildTests(unittest.TestCase):
    def test_preset_effective_and_old_mode_unchanged(self):
        cfg=BatchConfig(backgrounds='x',output='y',geometry_preset='rebuild',vehicle_reference=REFERENCE)
        self.assertEqual((cfg.scale_mode,cfg.contact_mode,cfg.chroma_mode),('reference_projection','strict','local_budget'))
        self.assertEqual(cfg.asset_scale_factors,{})
        self.assertEqual(cfg.effects.surface_dust,.5)
        self.assertEqual(cfg.effects.pool_kernel,3)
        self.assertEqual(cfg.effects.fog_mode,'local_auto')
        self.assertEqual(BatchConfig(backgrounds='x',output='y').asset_scale_factors['31'],.375)
        with self.assertRaises(ValueError):BatchConfig(backgrounds='x',output='y',geometry_preset='rebuild')

    def test_wrong_class_outer_width_missing_source_rejected(self):
        for ref in ({**REFERENCE,'vehicle_class':'railway'},
                    {k:v for k,v in REFERENCE.items() if k!='internal_width_m'},
                    {**REFERENCE,'source':{}},{**REFERENCE,'internal_width_m':float('nan')},
                    {**REFERENCE,'nominal_internal_height_m':float('inf')},
                    {**REFERENCE,'nominal_internal_length_m':-1}):
            with self.assertRaises(ValueError):validate_reference(ref)

    def test_http_trial_persists_same_rebuild_profile(self):
        from starlette.applications import Starlette
        from starlette.testclient import TestClient
        from display_batch import web
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bg=root/'backgrounds';bg.mkdir();Image.new('RGB',(32,32)).save(bg/'a.jpg')
            catalog={'3':{'available':True,'mesh_hash':'fixture'}}
            with patch.object(web,'ROOT',root),patch.object(web,'DATA',root/'configs'),\
                 patch.object(web,'load_catalog',return_value=catalog),patch.object(web,'launch') as launch,\
                 TestClient(Starlette(routes=web.routes)) as client:
                response=client.post('/api/batches/test',json={'backgrounds':str(bg),'assets':['3'],
                    'geometry_preset':'rebuild','vehicle_reference':REFERENCE,'asset_scale_factors':{'3':.25}})
                self.assertEqual(response.status_code,202,response.text)
                info=response.json();stored=json.loads((root/'configs'/(info['id']+'.json')).read_text(encoding='utf8'))
                self.assertEqual(stored['asset_scale_factors'],{})
                self.assertEqual(info['effective_modes']['contact'],'strict')
                self.assertEqual(info['effects']['pool_kernel'],3)
                self.assertEqual(client.get(info['status_url']).json()['vehicle_reference']['internal_width_m'],2.45)
                launch.assert_called_once()

    def test_known_width_camera_remains_estimated(self):
        cal=calibration()
        self.assertEqual(cal['status'],'unknown')
        self.assertTrue(cal['projectable']);self.assertLess(cal['residual_px'],.1)
        self.assertFalse(cal['absolute_depth_from_neural_map'])

    def test_fixed_size_and_near_far_projection(self):
        cal=calibration();cam=cal['metric_camera'];R=np.array(cam['R']);K=np.array(cam['K']);eye=np.array(cam['eye_m'])
        lengths=[]
        for y in [1.,3.]:
            p=np.array([[0,y,0]]);q=(p-eye)@R.T@K.T;anchor=q[0,:2]/q[0,2]
            crop,full=metric_projection(cal,anchor,206,[1600,1000],[0,0,0])
            lengths.append(float(np.linalg.norm(np.diff(project([[-.5,0,0],[.5,0,0]],crop),axis=0))*full))
        self.assertGreater(lengths[0],lengths[1])
        self.assertNotEqual(signature(REFERENCE),signature({**REFERENCE,'internal_width_m':2.55}))

    def test_height_uncertainty_separate_from_dimensions(self):
        cal=calibration();profile={'nominal_mm':206,'status':'pass'}
        scene={'size':[1600,1000],'vehicle_scale':{'calibration':cal,'signature':'x','reference':REFERENCE,
            'support_heights':{'S1':{'nominal_m':.6,'interval_m':[.3,.9]}}}}
        interval,evidence=local_scale(scene,{'id':'S1'},[800,500],profile)
        self.assertEqual(interval[0],interval[1]);self.assertEqual(evidence['physical_size']['nominal_mm'],206)
        self.assertEqual(evidence['status'],'unknown')
        self.assertGreater(evidence['calibration_interval_px_per_mm'][1],evidence['calibration_interval_px_per_mm'][0])
        self.assertGreater(plane_camera(cal,.9)['metric_camera']['eye_m'][2],cal['metric_camera']['eye_m'][2])

    def test_component_proxy_keeps_gap_and_source_hash(self):
        left=trimesh.creation.box(extents=[.2,.3,.1]);left.apply_translation([-.3,0,0])
        right=left.copy();right.apply_translation([.6,0,0]);mesh=trimesh.util.concatenate([left,right])
        with tempfile.TemporaryDirectory() as d:
            geometry=Path(d)/'g.npz';np.savez(geometry,vertices=mesh.vertices,faces=mesh.faces)
            original=file_hash(geometry);proxy=build(geometry,Path(d)/'proxy.json')
            self.assertEqual(len(proxy['parts']),2);self.assertEqual(file_hash(geometry),original)
            self.assertLess(proxy['max_outward_error'],1e-8)
            for part in proxy['parts']:
                x=np.asarray(part['vertices'])[:,0];self.assertTrue(x.max()<0 or x.min()>0)

    def test_camera_endpoint_probes_keep_unknown_evidence(self):
        cal=calibration();size=[1600,1000];mask=np.ones((1000,1600),bool)
        scene={'size':size,'supports':[{'id':'S1','mask':mask}],
            'allowed':mask,'protected':~mask,'vehicle_scale':{
                'calibration':cal,'signature':'test','reference':REFERENCE}}
        mesh=trimesh.creation.box(extents=[1,.3,.2]);mesh.apply_translation([0,0,.1])
        contacts=mesh.vertices[np.isclose(mesh.vertices[:,2],0)]
        pose={'matrix':np.eye(4).tolist(),'contact_points':contacts.tolist(),'length_m':.206}
        candidate={'scale':{'support_height':{'interval_m':[.3,.9]}},'anchor_px':[800,500],
            'plane_normal':[0,0,1],'contact_band':.002,'support_ids':['S1'],
            'receiver':{'patches':[{'vertices':[[-4,-4,0],[4,-4,0],[4,4,0],[-4,4,0]],
                'faces':[[0,1,2],[0,2,3]]}]}}
        checks=check_projection_interval(scene,candidate,pose,mesh.vertices,mesh.faces)
        self.assertEqual(checks['status'],'unknown')
        self.assertEqual(len(checks['checks']),4)
        self.assertTrue(all(c['status']=='pass' for c in checks['checks']))
        self.assertGreater(np.ptp([c['projected_major_px'] for c in checks['checks']]),1)

    def test_real_triangle_interior_contact_witness(self):
        p=np.array([[-1.,-1.,0],[1.,-1.,0],[0,1.,0]])
        floor=np.array([[-2.,-2.,0],[2.,-2.,0],[0,2.,0]])
        samples=contact_witnesses(p,np.array([[0,1,2]]),floor,np.array([[0,1,2]]),.001)
        self.assertTrue(np.any(np.linalg.norm(samples-p.mean(0),axis=1)<1e-8))
        floating=samples+[0,0,.005]
        self.assertTrue(np.all(floating[:,2]>.001))

    def test_line_contact_requires_com_and_roll_stability(self):
        a=np.linspace(0,2*np.pi,721)[:-1]
        vertices=np.array([[x,np.sin(t),1+np.cos(t)] for x in [-1,1] for t in a])
        contacts=np.array([[-1.,0,0],[0,0,0],[1.,0,0]])
        result=support(contacts,np.array([0.,0,1.]),vertices,.001,.0001,True)
        self.assertTrue(result['supported']);self.assertEqual(result['kind'],'line')
        self.assertFalse(support(contacts,np.array([0.,.1,1.]),vertices,.001,.0001,True)['supported'])
        self.assertFalse(support(contacts,np.array([0.,0,1.]),vertices,.001,.0001,False)['supported'])

    def test_adaptive_raster_and_shadow_do_not_move_anchor(self):
        cal=calibration();camera,full=metric_projection(cal,[800,500],100,[1600,1000],[0,0,0])
        mesh=trimesh.creation.box(extents=[1,.3,.2]);mesh.apply_translation([0,0,.1])
        g=projection_geometry(mesh.vertices,mesh.faces,{'matrix':np.eye(4).tolist()},0,55,camera=camera)
        self.assertLess(g['raster_span_uv'],.3)
        scene={'size':[1600,1000]};fit=fitted_masks(g,[800,500],full,scene)
        self.assertIsNotNone(fit);self.assertGreater(fit['hull'].sum(),100)
        c={'bounds':fit['bounds'],'full_px':full,'offset':fit['offset'],'target_px':100,
           'scale':{'calibration':cal}}
        first=render_windows(c,scene['size'],512);second=render_windows(c,scene['size'],1024)
        self.assertEqual(c['offset'],fit['offset'])
        self.assertGreater(first['shadow']['frame'],512)
        self.assertNotEqual(first['object']['frame'],second['object']['frame'])
        np.testing.assert_allclose(g['anchor']*full+fit['offset'],[800,500])


if __name__=='__main__':unittest.main(verbosity=2)
