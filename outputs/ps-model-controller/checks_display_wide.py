"""Large equal-weight pooling and arithmetic-mean local environment reconstruction."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

import checks_display_local
from display_batch.common import BLENDER,HERE
from display_batch.geometry import camera_for
from display_batch.local import neighbourhood,foreground_dust,pool_object_rgb,local_veil
from display_batch.local_light import estimate_local_light,reconstruct_probe
from display_batch.photography import linear


class WideEffectsTests(unittest.TestCase):
    def test_uniform_large_pool_has_unit_weight_at_edges_and_on_thin_rods(self):
        rgb,alpha,env=checks_display_local.ForegroundDustTests().fixture()
        before=alpha.copy()
        for size in (16,32,64):
            delta,info=foreground_dust(rgb,rgb,alpha,env,{'amount':0.,'color_linear':[.4]*3},300,
                pool_weight=1.,pool_kernel=size,fog_weight=0.,uniform=True,fog_profile='wide')
            expected=(pool_object_rgb(linear(rgb),alpha,size)-linear(rgb))*(alpha>0)[...,None]
            np.testing.assert_allclose(delta,expected,atol=1e-6)
            self.assertEqual(info['pool_weight_max'],1.)
            self.assertTrue(info['pool_uniform'])
            self.assertFalse(delta[alpha==0].any())
        np.testing.assert_array_equal(alpha,before)

    def test_pool64_normalizes_partial_cells_without_importing_transparent_colour(self):
        rgb=np.full((73,95,3),.99,np.float32);alpha=np.zeros((73,95),np.float32)
        alpha[2:70,3:92]=.7;alpha[22:34,32:45]=0
        rgb[alpha>0]=[.2,.4,.6]
        out=pool_object_rgb(rgb,alpha,64)
        np.testing.assert_allclose(out[alpha>0],rgb[alpha>0],atol=1e-6)

    def test_wide_fog_is_actual_opacity_and_uses_alpha_without_an_inner_dark_rim(self):
        rgb,alpha,env=checks_display_local.ForegroundDustTests().fixture()
        veiled,air=local_veil(rgb,{**env,'haze':.24},{'photo_style':'dusty_scrap'})
        for strength in (.30,.60,.85):
            delta,info=foreground_dust(rgb,veiled,alpha,env,air,300,pool_weight=0.,fog_weight=strength,
                uniform=True,fog_profile='wide')
            expected=(np.float32(air['color_linear'])-linear(veiled))*strength*(alpha>0)[...,None]
            np.testing.assert_allclose(delta,expected,atol=1e-6)
            self.assertAlmostEqual(info['fog_extra_max'],strength,places=6)
            self.assertEqual(info['fog_feather_px'],0)
            self.assertLessEqual(info['fog_total_max'],.900001)
        off,info=foreground_dust(rgb,rgb,alpha,env,{'amount':0.,'color_linear':[.4]*3},300,
            pool_weight=0.,fog_weight=.85,fog_profile='wide')
        self.assertFalse(off.any())


class MeanLightTests(unittest.TestCase):
    def fixture(self):
        scene,candidate=checks_display_local.LocalTests().fixture()
        gray=np.full((300,440),.25,np.float32)
        gray[::3]=.65
        scene['frame']=np.uint8(np.repeat(gray[...,None],3,axis=2)*255)
        candidate.update(camera=camera_for(np.array([[-.5,-.3,0],[.5,.3,.4]]),45),anchor_px=[215,200])
        return scene,candidate

    def test_light_uses_arithmetic_pixel_mean_not_median_or_equal_tile_voting(self):
        scene,candidate=self.fixture();env,_=neighbourhood(scene,candidate)
        env['airborne_dust']['amount']=0.
        new,mask=estimate_local_light(scene,candidate,env,statistic='mean',margin_scale=1.75)
        expected=np.mean(linear(np.float32(scene['frame'])/255)[mask],axis=0)
        np.testing.assert_allclose(new['local_irradiance']['target_clear_rgb_linear'],expected,atol=1e-6)
        self.assertGreater(float(np.linalg.norm(expected-np.median(linear(np.float32(scene['frame'])/255)[mask],axis=0))),.01)
        self.assertEqual(new['local_irradiance']['statistic'],'arithmetic_mean')

    def test_expansion_is_local_and_remote_pixels_do_not_enter_probe(self):
        scene,candidate=self.fixture();env,_=neighbourhood(scene,candidate)
        new,mask=estimate_local_light(scene,candidate,env,statistic='mean',margin_scale=1.75)
        x,y,r,b=new['local_irradiance']['bounds']
        old=env['local_context']['bounds']
        self.assertGreater((r-x)*(b-y),(old[2]-old[0])*(old[3]-old[1]))
        remote=np.ones(mask.shape,bool);remote[y:b,x:r]=False
        scene['frame'][remote]=[255,0,255]
        other,othermask=estimate_local_light(scene,candidate,env,statistic='mean',margin_scale=1.75)
        self.assertEqual(new,other)
        np.testing.assert_array_equal(mask,othermask)
        self.assertEqual(reconstruct_probe(new,candidate),reconstruct_probe(other,candidate))

    def test_probe_keeps_mean_energy_and_separates_air_and_material(self):
        scene,candidate=self.fixture();env,_=neighbourhood(scene,candidate)
        new,_=estimate_local_light(scene,candidate,env,statistic='mean',margin_scale=1.75)
        before=copy.deepcopy(new)
        probe=reconstruct_probe(new,candidate)
        self.assertEqual(before,new)
        for k in ('airborne_dust','dust_color','sensor','camera_psf'):
            self.assertEqual(probe[k],new[k])
        p=np.asarray(probe['reflection_probe'])
        latitude=(np.arange(64)+.5)/64*np.pi-np.pi/2
        weights=np.broadcast_to(np.cos(latitude)[:,None],(64,128))
        np.testing.assert_allclose(np.sum(p*weights[...,None],axis=(0,1))/weights.sum(),new['world_color'],atol=1e-6)
        self.assertTrue(np.isfinite(p).all() and (p>=0).all())
        self.assertEqual(probe['local_irradiance']['reconstruction']['status'],'estimated')

    def test_blender_environment_direction_is_not_mirrored(self):
        code=f'''
import sys,math
import numpy as np
sys.path.insert(0,{str(HERE)!r})
import bpy
from mathutils import Vector
import blender_display_worker as worker
scene=worker.reset('CPU',512)
scene.render.resolution_x=scene.render.resolution_y=64
for obj in list(scene.objects):bpy.data.objects.remove(obj,do_unlink=True)
bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,radius=1)
sphere=bpy.context.object
mat=bpy.data.materials.new('neutral');mat.use_nodes=True
shader=mat.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.4,.4,.4,1)
shader.inputs['Roughness'].default_value=1;shader.inputs['Specular IOR Level'].default_value=0
sphere.data.materials.append(mat)
camera=worker.legacy.make_camera(scene,(4,0,0),(0,0,0))
camera.data.type='ORTHO';camera.data.ortho_scale=2.6
yy,xx=np.mgrid[:64,:128]
lat=(yy+.5)/64*math.pi-math.pi/2;az=math.pi-(xx+.5)/128*2*math.pi
world_y=np.cos(lat)*np.sin(az)
probe=np.repeat((.1+1.9*np.maximum(world_y,0))[...,None],3,axis=2)
env={{'world_color':[.2]*3,'light_color':[1.]*3,'reflection_probe':probe.tolist(),
 'local_irradiance':{{'direction':{{'direction':None,'confidence':0.}},'level':1.,'fill_ratio':.5}}}}
state={{}};worker.scene_lighting(state,{{'eye':[4,0,0],'target':[0,0,0]}},env,1.,True)
for light in state['lights']:light.data.energy=0
scene.cycles.samples=16
scene.render.filepath=__file__+'.png';bpy.ops.render.render(write_still=True)
image=bpy.data.images.load(scene.render.filepath,check_existing=False)
pixels=np.array(image.pixels[:]).reshape(64,64,4)
left=pixels[22:42,18:28,:3].mean();right=pixels[22:42,36:46,:3].mean()
assert right>left*1.15,(left,right)
print('PROBE_DIRECTION_OK')
'''
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/'probe.py';script.write_text(code,encoding='utf8')
            completed=subprocess.run([str(BLENDER),'--background','--factory-startup','--python',str(script)],
                capture_output=True,text=True,encoding='utf8',timeout=90)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertIn('PROBE_DIRECTION_OK',completed.stdout,completed.stdout[-2000:]+completed.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
