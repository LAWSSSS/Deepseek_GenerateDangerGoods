"""Offline regression of support evidence and pre-simulation ordering."""
import unittest
import numpy as np
import trimesh
from display_batch.pile import support_plane_evidence,shape_priority,stability_priority,diverse_candidates


class SupportPriorityTests(unittest.TestCase):
    def test_curved_and_stacked_mask_is_not_a_plane(self):
        for label in ('中左曲面钢片','堆叠角钢','右中卷曲板顶面','大面积弯板顶面','薄钢片叠层','curved steel','stacked plates'):
            hint=support_plane_evidence({'id':'S1','label':label})
            self.assertFalse(hint['eligible'])
            self.assertEqual(hint['status'],'unknown')
        self.assertTrue(support_plane_evidence({'id':'S1','label':'单片平整钢板'})['eligible'])

    def test_structured_surface_kind_prevents_flattening_even_with_plain_label(self):
        from display_batch.schema import SceneAnswer,StructuredSurface
        for kind in ('curved','compound'):
            self.assertFalse(support_plane_evidence({'id':'S1','label':'钢料','geometry_kind':kind})['eligible'])
        self.assertNotIn('geometry_kind',SceneAnswer.model_json_schema()['$defs']['Surface']['properties'])
        self.assertIn('geometry_kind',StructuredSurface.model_json_schema()['required'])

    def test_unknown_and_inherited_evidence_not_converted_to_failure(self):
        direct=support_plane_evidence({'id':'S1','label':'钢件'})
        derived=support_plane_evidence({'id':'D1','label':'segmented interior steel; estimated support'})
        self.assertTrue(derived['eligible'])
        self.assertEqual(derived['status'],'unknown')
        self.assertGreater(derived['penalty'],direct['penalty'])

    def test_independent_measured_geometry_preserved(self):
        hint=support_plane_evidence({'id':'S1','label':'curved steel','plane_status':'measured','plane_origin_m':[0,0,1]})
        self.assertTrue(hint['eligible'])
        self.assertEqual(hint['penalty'],0)

    def test_candidate_budget_diversifies_actual_support(self):
        rows=[{'id':str(i),'site_id':s,'support_ids':[s],'score':10-i} for i,s in enumerate(['S1','S1','S1','S2','S3','S2'])]
        selected=diverse_candidates(rows,3)
        self.assertEqual([r['site_id'] for r in selected],['S1','S2','S3'])
        self.assertEqual(len(diverse_candidates(rows[:2],3)),2)
        self.assertEqual([r['id'] for r in rows],list(map(str,range(6))))

    def test_cylinder_cross_slope_ranked_below_along_slope(self):
        cylinder=trimesh.creation.cylinder(radius=.14,height=1.,sections=64)
        vertices=np.asarray(cylinder.vertices);axis=np.array([0.,0.,1.]);shape=shape_priority(vertices,axis)
        self.assertGreater(shape['roundness'],.8)
        normal=np.array([0.,np.sin(np.radians(12)),np.cos(np.radians(12))])
        def orientation(direction):
            direction=np.asarray(direction);direction-=np.dot(direction,normal)*normal;direction/=np.linalg.norm(direction)
            x=np.cross(direction,normal);y=np.cross(direction,x)
            m=np.eye(4);m[:3,:3]=np.column_stack([x,y,direction]);return m
        cross=stability_priority(vertices,orientation([1.,0,0]),normal,axis,[0,0,0],shape)
        along=stability_priority(vertices,orientation([0.,1.,0]),normal,axis,[0,0,0],shape)
        self.assertGreater(cross['cross_axis_rolling_risk'],along['cross_axis_rolling_risk']+.1)
        self.assertGreater(cross['penalty'],along['penalty'])
        self.assertEqual(cross['status'],'unknown')

    def test_flat_supported_box_does_not_receive_roll_penalty(self):
        box=trimesh.creation.box(extents=[1.,.5,.3]);axis=np.array([1.,0,0]);shape=shape_priority(box.vertices,axis)
        hint=stability_priority(box.vertices,np.eye(4),[0,0,1],axis,[0,0,0],shape)
        self.assertEqual(hint['cross_axis_rolling_risk'],0.)
        self.assertEqual(hint['gravity_outside_contact_envelope'],0.)
        self.assertEqual(hint['penalty'],0.)


if __name__=='__main__':unittest.main()
