"""Offline regressions for calibrated light, bounded veil and contact appearance."""
import copy
import unittest
import numpy as np
from display_batch.local_atmosphere import constrain_dark_floor
from display_batch.compose import contact_shading_supported
from display_batch.photography import match_linear
from display_batch.schema import BatchConfig


class CoherentIntegration(unittest.TestCase):
    def test_mislabelled_haze_scope_still_allows_bounded_camera_repair(self):
        from display_batch.review import repair
        from display_batch.schema import EffectsReview, Scores, Defect, Action
        review=EffectsReview(trace='visible',scores=Scores(**dict.fromkeys(
            ('support','occlusion','lighting','edge','texture','material','structure'),3)),
            defects=[Defect(code='haze_clear',scope='asset',severity='minor',region=(.1,.1,.8,.8),evidence='nearby haze stronger')],
            actions=[Action(kind='fog_gain',value='1.25')],evidence='camera veil')
        before=review.model_dump()
        changed,log=repair({'candidate':'C1','fog_gain':'1','effects_enabled':True},review,{'fog_gain':['1.25']},['C1'],lambda _:False)
        self.assertEqual(changed['fog_gain'],'1.25');self.assertEqual(log[0]['disposition'],'scheduled')
        self.assertEqual(review.model_dump(),before)

    def test_bright_neutral_surface_does_not_prove_fog(self):
        rgb=np.full((64,64,3),.4,np.float32);valid=np.ones((64,64),bool)
        fog,info=constrain_dark_floor(np.full((64,64),.48),rgb,valid,np.array([.5]*3),{'kind':'haze'})
        self.assertLessEqual(fog.max(),.1);self.assertEqual(info['confidence'],'unknown')

    def test_haze_dark_floor_orders_clear_and_dusty(self):
        base=np.broadcast_to(np.linspace(.004,.4,64,dtype=np.float32)[None,:,None],(64,64,3)).copy()
        amounts=[]
        for source in (base,base*.7+.3*.6):
            fog,_=constrain_dark_floor(np.full((64,64),.48),source,np.ones((64,64),bool),np.array([.6]*3),{'kind':'mixed'})
            amounts.append(float(fog.mean()))
        self.assertLess(amounts[0],.06);self.assertGreater(amounts[1],.20)

    def test_clear_visual_evidence_and_missing_samples(self):
        rgb=np.full((20,20,3),.3,np.float32)
        f,_=constrain_dark_floor(np.ones((20,20)),rgb,np.ones((20,20),bool),np.array([.5]*3),{'kind':'clear','confidence':.9})
        self.assertEqual(f.max(),0.)
        f,info=constrain_dark_floor(np.ones((20,20)),rgb,np.eye(20,dtype=bool),np.array([.5]*3),{})
        self.assertLessEqual(f.max(),.08);self.assertEqual(info['confidence'],'unknown')

    def test_unknown_receiver_renders_conditional_contact_without_acceptance(self):
        c={'terrain_physics':{'stable':True,'status':'unknown','failure_codes':[],
            'center_supported':True,'contact_count':12},'contact_gap':{'status':'pass'}}
        original=copy.deepcopy(c)
        self.assertTrue(contact_shading_supported(c,[.95],True))
        self.assertFalse(contact_shading_supported(c,[.95],False))
        self.assertEqual(c,original)
        for key,value in [('stable',False),('failure_codes',['floating']),('center_supported',False),('contact_count',0)]:
            changed=copy.deepcopy(c);changed['terrain_physics'][key]=value
            self.assertFalse(contact_shading_supported(changed,[.95],True))
        self.assertFalse(contact_shading_supported(c,[.8],True))
        c['contact_gap']['status']='fail';self.assertFalse(contact_shading_supported(c,[1.],True))

    def test_calibration_is_shared_not_clipped_to_extra_tone_budget(self):
        alpha=np.ones((32,32),np.float32);saved=alpha.copy();raw=np.full((32,32,3),.10,np.float32)
        env={'material_classes':['steel'],'calibration':{'exposure_ev':1.1,'white_balance':[1.,1.,1.],'limited':False}}
        params={'weathering':'preserve_source','tone':'scene','photo_style':'dusty_scrap'}
        args=(raw,alpha,np.full((200,3),110,np.float32),np.zeros((32,32),np.float32))
        _,old=match_linear(*args,params,1,env,reference=True,camera_capture=False,surface_weathering=False)
        rgb,new=match_linear(*args,{**params,'integration_mode':'coherent'},1,env,reference=True,camera_capture=False,surface_weathering=False)
        self.assertAlmostEqual(old['exposure_ev'],.5);self.assertAlmostEqual(new['exposure_ev'],1.1)
        self.assertAlmostEqual(new['photographic_adjustment_ev'],0.);self.assertTrue(np.array_equal(alpha,saved))
        self.assertTrue(np.isfinite(rgb).all())

    def test_new_contact_evidence_never_uses_shadow_to_clear_unknown_or_failed_support(self):
        c={'terrain_physics':{'stable':True,'status':'unknown','failure_codes':[],
            'center_supported':True,'contact_count':12},
            'receiver':{'contact_evidence_version':'visible-support-v1'},
            'contact_gap':{'status':'unknown','simulation_status':'pass','visible_support':{'status':'pass'}}}
        original=copy.deepcopy(c)
        self.assertTrue(contact_shading_supported(c,[.99],True))
        self.assertEqual(c,original)
        c['contact_gap']['visible_support']['status']='fail'
        self.assertFalse(contact_shading_supported(c,[1.],True))
        c['terrain_physics']['status']='pass'
        self.assertFalse(contact_shading_supported(c,[1.],True))

    def test_old_config_stays_legacy_and_new_mode_serializes(self):
        cfg=BatchConfig(backgrounds='example',output='example')
        self.assertEqual(cfg.integration_mode,'legacy')
        cfg.integration_mode='coherent'
        self.assertEqual(BatchConfig.model_validate_json(cfg.model_dump_json()).integration_mode,'coherent')


if __name__=='__main__':unittest.main()
