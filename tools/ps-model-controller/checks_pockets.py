"""Synthetic geometric regressions for visible-bottom recess proposals."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scrap_composite.pockets import propose,check_projection


def basin(size=240):
    yy,xx=np.indices((size,size));rr=((xx-size/2)**2+(yy-size/2)**2)**.5
    # The floor is farther than enclosing rims; every visible pixel is real material.
    depth=(.60-.15*np.exp(-(rr/42)**4)).astype(np.float32)
    support=rr<52;allowed=np.ones_like(support);protected=np.zeros_like(support)
    return depth,support,allowed,protected


class PocketTests(unittest.TestCase):
    def test_multiscale_visible_bottom_is_proposed(self):
        args=basin();rows,summary=propose(*args,140)
        self.assertTrue(rows);self.assertGreaterEqual(rows[0]['scale_votes'],2)
        self.assertLess(np.linalg.norm(np.subtract(rows[0]['anchor_px'],[120,120])),24)
        self.assertFalse(summary['metric_cavity_recovered']);self.assertEqual(rows[0]['status'],'unknown')

    def test_linear_perspective_slope_is_not_a_pocket(self):
        yy,xx=np.indices((240,240));d=(.2+xx*.001+yy*.0005).astype(np.float32)
        rows,summary=propose(d,np.ones_like(d,bool),np.ones_like(d,bool),np.zeros_like(d,bool),100)
        self.assertEqual(rows,[]);self.assertIn('fallback',summary['reason'])

    def test_convex_pipe_top_is_not_a_bottom(self):
        depth,support,allowed,protected=basin();rows,_=propose(1.-depth,support,allowed,protected,140)
        self.assertEqual(rows,[])

    def test_gap_and_protected_hole_cannot_supply_bottom(self):
        depth,support,allowed,protected=basin()
        self.assertEqual(propose(depth,np.zeros_like(support),allowed,protected,140)[0],[])
        protected|=support
        self.assertEqual(propose(depth,support,allowed,protected,140)[0],[])

    def test_depth_without_two_distinct_rim_directions_does_not_pass(self):
        d=np.full((240,240),.5,np.float32);d[:,180:]=.65
        support=np.zeros_like(d,bool);support[100:140,100:140]=True
        rows,_=propose(d,support,np.ones_like(support),np.zeros_like(support),140)
        self.assertEqual(rows,[])

    def test_object_window_changes_search_and_does_not_scale_object(self):
        args=basin();a,sa=propose(*args,80);b,sb=propose(*args,180)
        self.assertEqual(sa['target_px'],80);self.assertEqual(sb['target_px'],180)
        self.assertNotEqual(a[0]['rim_evidence'][0]['radius_px'],b[0]['rim_evidence'][0]['radius_px'])
        self.assertFalse(sa['physics_validated']);self.assertFalse(sb['physics_validated'])

    def test_actual_hole_in_silhouette_not_treated_as_collision(self):
        silhouette=np.ones((40,40),bool);silhouette[10:30,10:30]=False
        protected=~silhouette;contacts=np.zeros_like(silhouette);contacts[32:36,5:35]=True
        result=check_projection({},silhouette,contacts,[0,0,40,40],silhouette,np.ones_like(silhouette),protected)
        self.assertEqual(result['program_fit'],'pass');self.assertEqual(result['status'],'unknown')
        solid=check_projection({},np.ones_like(silhouette),contacts,[0,0,40,40],silhouette,np.ones_like(silhouette),protected)
        self.assertIn('protected_conflict',solid['reasons'])

    def test_separate_contact_cluster_cannot_bridge_empty_gap(self):
        mask=np.ones((40,40),bool);contact=np.zeros_like(mask);contact[5:10,5:10]=True;contact[30:35,30:35]=True
        support=mask.copy();support[30:35,30:35]=False
        result=check_projection({},mask,contact,[0,0,40,40],support,mask,np.zeros_like(mask))
        self.assertEqual(result['program_fit'],'fail');self.assertEqual(result['contact_coverages'],[1.,0.])

    def test_adapter_uses_fixed_physical_projection_without_rewriting_size(self):
        from display_batch.pile import pocket_proposals
        depth,support,allowed,protected=basin()
        scene={'depth_native':depth,'allowed':allowed,'protected':protected}
        profile={'nominal_mm':206.,'interval_mm':[206.,206.]}
        asset={'physical_size':profile};site={'anchor_px':[120,120]}
        with patch('display_batch.reference_projection.local_scale',return_value=([140.,140.],{'status':'unknown','basis':'estimated_camera'})) as scale:
            proposals,summary=pocket_proposals(asset,scene,site,support,SimpleNamespace(scale_mode='reference_projection'))
        self.assertTrue(proposals);self.assertEqual(summary['target_px'],140.)
        self.assertTrue(summary['scale_basis']['physical_size_fixed'])
        self.assertEqual(profile,{'nominal_mm':206.,'interval_mm':[206.,206.]})
        self.assertEqual(scale.call_args.args[-1],profile)

    def test_coarse_diagnostic_coordinates_return_to_native_grid(self):
        from display_batch.pile import scale_pocket
        pockets,_=propose(*basin(),140);original=pockets[0]
        native=scale_pocket(original,.25)
        self.assertTrue(np.allclose(native['anchor_px'],np.asarray(original['anchor_px'])*4))
        self.assertTrue(np.allclose(native['rim_evidence'][0]['rim_points_px'],np.asarray(original['rim_evidence'][0]['rim_points_px'])*4))
        self.assertEqual(native['score'],original['score'])


if __name__=='__main__':unittest.main()
