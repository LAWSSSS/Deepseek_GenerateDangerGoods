"""Visible receiver evidence must not promote an estimated plane to real contact."""
import unittest
import numpy as np
from display_batch.pile import receiver_mesh,visible_support_mask,support_plane_evidence
from display_batch.strict_contact import projected_gap,validate_receiver,visible_contact_coverage
from display_batch.geometry import project


def fixture(size=240):
    mask=np.ones((size,size),bool)
    site={'id':'S1','label':'单片钢板','mask':mask,'slope_deg':0.,'plane_residual':0.,'depth':.5}
    scene={'size':[size,size],'allowed':mask.copy(),'protected':np.zeros_like(mask),
           'depth_native':np.full(mask.shape,.5,np.float32),'depth':np.full((60,60),.5,np.float32),'supports':[site]}
    camera={'eye':[0.,0.,5.],'target':[0.,0.,0.],'basis':[[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]],'fov':45.}
    candidate={'id':'fixture','site_id':'S1','support_ids':['S1'],'bounds':[0,0,size,size],
        'camera':camera,'offset':[0.,0.],'full_px':float(size),'strict_contact':True,
        'plane_normal':[0,0,1],'integration_mode':'coherent'}
    return scene,site,candidate


class VisibleSupportTests(unittest.TestCase):
    def test_coherent_receiver_never_fills_one_pixel_hole(self):
        scene,site,candidate=fixture();site['mask'][52,52]=False
        candidate['receiver']=receiver_mesh(scene,candidate)
        # Invert the camera ray for a contact exactly at the known missing pixel.
        from display_batch.pile import world_on_plane
        points,_=world_on_plane(np.array([[52.,52.]]),candidate['camera'],[0.,0.],candidate['full_px'])
        result=visible_contact_coverage(points,candidate)
        self.assertEqual(result['status'],'fail')
        self.assertEqual(result['outside_visible_receiver_points'],1)

    def test_depth_jump_is_not_new_support(self):
        scene,site,_=fixture();scene['depth_native'][80:110,80:110]=.85
        mask,proof=visible_support_mask(scene,site)
        self.assertFalse(mask[90,90]);self.assertTrue(mask[30,30])
        self.assertGreater(proof['retained_fraction'],.7)
        self.assertEqual(proof['status'],'unknown')
        self.assertFalse(proof['metric_height_known'])

    def test_smooth_perspective_trend_keeps_supported_pixels(self):
        scene,site,_=fixture();yy,xx=np.indices(site['mask'].shape)
        scene['depth_native']=(.3+xx*.0003+yy*.0004).astype(np.float32)
        mask,proof=visible_support_mask(scene,site)
        self.assertGreater(mask.mean(),.98)
        self.assertEqual(proof['status'],'unknown')

    def test_virtual_contact_gap_is_not_measured_contact(self):
        scene,_,candidate=fixture();candidate['receiver']=receiver_mesh(scene,candidate)
        pose={'contact_points':[[-.1,0.,0.],[.1,0.,0.]],'receiver_points':[[-.1,0.,0.],[.1,0.,0.]]}
        result=projected_gap(pose,candidate)
        self.assertEqual(result['simulation_status'],'pass')
        self.assertEqual(result['visible_support']['status'],'pass')
        self.assertEqual(result['status'],'unknown')
        self.assertEqual(result['maximum_px'],0.)

    def test_missing_visible_evidence_requires_new_candidate(self):
        scene,_,candidate=fixture();candidate['receiver']=receiver_mesh(scene,candidate)
        validate_receiver(candidate)
        candidate['receiver'].pop('contact_evidence_version')
        with self.assertRaisesRegex(ValueError,'visible_receiver_evidence_missing'):validate_receiver(candidate)

    def test_round_pipe_not_confused_with_square_tube_face(self):
        self.assertFalse(support_plane_evidence({'id':'S1','label':'圆管顶面'})['eligible'])
        self.assertTrue(support_plane_evidence({'id':'S1','label':'方管平整顶面'})['eligible'])
        self.assertTrue(support_plane_evidence({'id':'S1','label':'横向钢管顶面'})['eligible'])

    def test_legacy_receiver_semantics_unchanged(self):
        scene,_,candidate=fixture();candidate.pop('integration_mode')
        candidate['receiver']=receiver_mesh(scene,candidate)
        pose={'contact_points':[[0.,0.,0.]],'receiver_points':[[0.,0.,0.]]}
        self.assertEqual(projected_gap(pose,candidate)['status'],'pass')
        self.assertNotIn('visible_support',candidate['receiver'])


if __name__=='__main__':unittest.main()
