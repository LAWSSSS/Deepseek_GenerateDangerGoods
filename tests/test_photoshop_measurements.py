import unittest
import numpy as np
from scrap_composite.photography_measurements import measure_regions, measure_export_regions


class PhotoshopRegionTests(unittest.TestCase):
    def test_bounded_reference_excludes_object_and_protection(self):
        rgb=np.full((180,240,3),80,np.uint8)
        alpha=np.zeros((180,240),np.uint8); alpha[70:100,90:120]=255
        rgb[alpha>0]=[180,45,40]
        blocked=np.zeros(alpha.shape,bool); blocked[60:110,125:130]=True
        rgb[blocked]=[245,0,255]
        result=measure_export_regions(rgb,alpha,blocked=blocked)
        self.assertEqual(result['neighborhood']['mean_rgb_255'],[80,80,80])
        self.assertEqual(result['reference_confidence'],'unknown_same_material_and_depth')
        self.assertEqual(result['object']['samples'],900)
        self.assertGreater(result['bounds'][0],0)

    def test_roi_bounds_and_units_are_explicit(self):
        rgb=np.full((100,100,3),128,np.uint8)
        row=measure_regions(rgb,[{'id':'local','bounds':[.1,.2,.4,.6]}])[0]
        self.assertEqual(row['samples'],1200)
        self.assertEqual(row['pixel_bounds'],[10,20,40,60])
        self.assertAlmostEqual(row['linear_luminance_p10_p50_p90'][1],.21586,places=4)
        for bad in ([0,0,2,1],[.5,0,.5,1],[float('nan'),0,1,1]):
            with self.assertRaises(ValueError):
                measure_regions(rgb,[{'id':'bad','bounds':bad}])

    def test_small_region_is_unknown(self):
        row=measure_regions(np.zeros((20,20,3),np.uint8),[{'id':'tiny','bounds':[0,0,.1,.1]}])[0]
        self.assertEqual(row['status'],'unknown')


if __name__=='__main__': unittest.main()
