"""Offline appearance extraction: sampling, native resolution and conservative maps."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ps-model-controller'))
from display_batch.reference_textures import (
    foreground, panel_boxes, estimated_maps, select_patches, prepare_reference, prepare_flat_master, quilt,
)


def sheet():
    rng=np.random.default_rng(14)
    rgb=np.full((400,600,3),255,np.uint8)
    for row in range(2):
        for col in range(3):
            x,y=col*200,row*200
            noise=rng.normal(0,9,(140,130,1))
            surface=np.clip(np.array([145,60,45])[None,None,:]+noise,0,255).astype(np.uint8)
            rgb[y+25:y+165,x+35:x+165]=surface
            cv2.putText(rgb,'Front',(x+55,y+193),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    return rgb


class ReferenceTexturesTests(unittest.TestCase):
    def test_nonuniform_panel_dividers_and_caption_background_are_excluded(self):
        image=np.full((400,600,3),255,np.uint8)
        image[:,199:201]=220;image[:,399:401]=220;image[239:241]=220
        boxes=panel_boxes(image)
        self.assertLessEqual(abs(boxes[0][3]-240),2)
        self.assertLessEqual(abs(boxes[1][0]-200),2)
        sample=sheet()[:200,:200]
        mask,_=foreground(sample)
        self.assertEqual(int(mask[-20:].sum()),0)
        self.assertEqual(int(mask[:,:10].sum()),0)
        self.assertGreater(mask[40:150,50:150].mean(),.99)

    def test_selected_patches_stay_inside_objects_and_never_use_panel_captions(self):
        patches,stats=select_patches(sheet())
        self.assertGreaterEqual(len(patches),3)
        self.assertGreaterEqual(len(stats['panels_used']),2)
        for p in patches:
            x,y,r,b=p['roi'];panel=(p['panel']-1)
            origin_x=(panel%3)*200;origin_y=(panel//3)*200
            self.assertGreaterEqual(x,origin_x+35)
            self.assertLessEqual(r,origin_x+165)
            self.assertGreaterEqual(y,origin_y+25)
            self.assertLessEqual(b,origin_y+165)
            self.assertEqual(p['pixels'].shape,(64,64,3))

    def test_white_label_and_broad_black_structure_are_not_sampled(self):
        image=sheet()
        for row in range(2):
            for col in range(3):
                x,y=col*200,row*200
                image[y+25:y+85,x+35:x+165]=245
                cv2.putText(image,'ABC123',(x+40,y+65),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2)
        patches,stats=select_patches(image)
        for p in patches:
            self.assertGreaterEqual(p['roi'][1]%200,85)
        self.assertGreater(stats['rejected'].get('strong_highlight_or_white_label',0)+
                           stats['rejected'].get('background_edge_caption_or_hole',0),0)

    def test_flat_gradient_cannot_be_normalized_into_large_height_variation(self):
        ramp=np.tile(np.linspace(70,160,64,dtype=np.uint8),(64,1))
        maps=estimated_maps(np.repeat(ramp[...,None],3,axis=2))
        self.assertLess(float(np.abs(maps[...,6]-.5).max()),.005)
        constant=estimated_maps(np.full((64,64,3),110,np.uint8))
        np.testing.assert_allclose(constant[...,6],.5,atol=1e-6)
        self.assertLess(float(np.std(np.diff(maps[...,3],n=2,axis=1))),.002)

    def test_all_maps_share_tiling_edges_and_keep_native_patch_sampling(self):
        patches,_=select_patches(sheet())
        result,placements=quilt(patches,seed=17,paint=True)
        self.assertEqual(result.shape,(512,512,8))
        np.testing.assert_allclose(result[0],result[-1],atol=1e-7)
        np.testing.assert_allclose(result[:,0],result[:,-1],atol=1e-7)
        again,_=quilt(patches,seed=17,paint=True)
        np.testing.assert_array_equal(result,again)
        self.assertTrue(placements)
        with self.assertRaises(ValueError):quilt(patches,size=2048)
        with self.assertRaises(ValueError):select_patches(sheet(),patch_size=32)

    def test_pack_hash_cache_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png';Image.fromarray(sheet()).save(source)
            before=source.read_bytes()
            first=prepare_reference(source,root/'a',seed=19)
            second=prepare_reference(source,root/'b',seed=19)
            self.assertEqual(first['signature'],second['signature'])
            self.assertEqual({k:v['sha256'] for k,v in first['maps'].items()},
                             {k:v['sha256'] for k,v in second['maps'].items()})
            self.assertEqual(len(first['maps']),6)
            self.assertEqual(first['status'],'review')
            self.assertEqual(first['physical_status'],'estimated_not_measured_PBR')
            self.assertLessEqual(first['height_amplitude_max_m'],.00008)
            self.assertTrue(prepare_reference(source,root/'a',seed=19)['cache_hit'])
            self.assertEqual(source.read_bytes(),before)
            Image.new('RGB',(600,400)).save(source)
            with self.assertRaisesRegex(ValueError,'changed inputs'):prepare_reference(source,root/'a',seed=19)

    def test_insufficient_evidence_is_review_without_fabricated_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'blank.png';Image.new('RGB',(600,400),'white').save(source)
            record=prepare_reference(source,root/'out')
            self.assertEqual(record['maps'],{})
            self.assertEqual(record['status'],'review')
            self.assertIn('insufficient',record['reason'])

    def flat_fixture(self, root, size=(640,640)):
        master=root/'master.png';reference=root/'reference.png'
        rng=np.random.default_rng(91)
        noise=rng.normal(0,7,(size[1],size[0],1))
        pixels=np.clip(np.array([145,47,36])+noise,0,255).astype(np.uint8)
        Image.fromarray(pixels).save(master);Image.fromarray(sheet()).save(reference)
        provenance={'prompt':'Evenly lit red worn paint; no labels or geometry.',
                    'generator':'fixture','reference_images':[{'path':str(reference)}],
                    'unrelated_private_config':'must not be serialized'}
        return master,reference,provenance

    def test_flat_master_uses_complete_surface_not_panel_extraction_or_quilting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);master,reference,provenance=self.flat_fixture(root,size=(768,512))
            before=master.read_bytes();ref_before=reference.read_bytes()
            with (patch('display_batch.reference_textures.select_patches',side_effect=AssertionError('no panels')),
                  patch('display_batch.reference_textures.quilt',side_effect=AssertionError('no quilt'))):
                record=prepare_flat_master(master,root/'pack',provenance,material_kind='paint')
            self.assertEqual(record['sources'][0]['roi'],[0,0,768,512])
            self.assertEqual(record['stats']['output_resolution'],[512,341])
            with Image.open(record['maps']['albedo']['path']) as opened:
                self.assertEqual(opened.size,(512,341))
            self.assertEqual(len(record['maps']),6)
            self.assertEqual(record['maps']['albedo']['color_space'],'sRGB')
            self.assertEqual(record['maps']['height']['color_space'],'Non-Color')
            self.assertNotIn('quilting',record)
            self.assertTrue(record['stats']['no_quilting'])
            self.assertEqual(master.read_bytes(),before);self.assertEqual(reference.read_bytes(),ref_before)

    def test_flat_master_provenance_cache_and_complete_prompt_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);master,reference,provenance=self.flat_fixture(root)
            record=prepare_flat_master(master,root/'a',provenance)
            other=prepare_flat_master(master,root/'b',provenance)
            self.assertEqual(record['generation']['prompt'],provenance['prompt'])
            self.assertEqual(record['generation']['prompt_sha256'],hashlib.sha256(provenance['prompt'].encode()).hexdigest())
            self.assertEqual(record['generation']['reference_images'][0]['sha256'],hashlib.sha256(reference.read_bytes()).hexdigest())
            self.assertNotIn('unrelated_private_config',record['generation'])
            self.assertEqual(len(record['implementation_sha256']),64)
            self.assertEqual(record['signature'],other['signature'])
            self.assertEqual({k:v['sha256'] for k,v in record['maps'].items()},
                             {k:v['sha256'] for k,v in other['maps'].items()})
            self.assertTrue(prepare_flat_master(master,root/'a',provenance)['cache_hit'])
            with patch('display_batch.reference_textures._flat_implementation_hash',return_value='a'*64):
                with self.assertRaisesRegex(ValueError,'changed flat master'):
                    prepare_flat_master(master,root/'a',provenance)
            with self.assertRaisesRegex(ValueError,'changed flat master'):
                prepare_flat_master(master,root/'a',provenance|{'prompt':'Changed generation prompt'})
            wrong=provenance|{'reference_images':[{'path':str(reference),'sha256':'0'*64}]}
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                prepare_flat_master(master,root/'c',wrong)
            Path(record['maps']['height']['path']).write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'missing or modified'):
                prepare_flat_master(master,root/'a',provenance)

    def test_edit_input_and_style_ancestor_roles_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);master,reference,provenance=self.flat_fixture(root)
            provenance['reference_images']=[{'path':str(reference),'role':'style_ancestor_not_direct_input'},
                                           {'path':str(master),'role':'actual_edit_input'}]
            record=prepare_flat_master(master,root/'pack',provenance)
            refs=record['generation']['reference_images']
            self.assertEqual(refs[0]['role'],'style_ancestor_not_direct_input')
            self.assertEqual(refs[1]['role'],'actual_edit_input')

    def test_flat_master_resolution_preserves_detail_without_upscaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);master,_,provenance=self.flat_fixture(root,size=(1050,1050))
            record=prepare_flat_master(master,root/'a',provenance,size=1024)
            self.assertEqual(record['stats']['output_resolution'],[1024,1024])
            self.assertGreater(record['stats']['source_pixels_per_output_pixel'],1)
            small,_,prov=self.flat_fixture(root,size=(256,256))
            rejected=prepare_flat_master(small,root/'small',prov,size=512)
            self.assertEqual(rejected['maps'],{})
            self.assertIn('no_upscaling',rejected['reason'])

    def test_red_coating_is_not_mistaken_for_complete_oxide_layer(self):
        # Even a red-orange coating near the broad rust hue estimate stays coating.
        hsv=np.full((128,128,3),(10,200,135),np.uint8)
        hsv[40:70,40:70]=(27,190,90)
        rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        maps=estimated_maps(rgb,paint=True,protect_paint=True)
        self.assertLess(float(maps[:30,:,4].max()),.001)
        self.assertGreater(float(maps[45:65,45:65,4].mean()),.5)
        self.assertLess(float(np.mean(maps[...,4]>.5)),.1)

    def test_nearby_brown_rust_and_neutral_chip_interiors_match_paint_masks(self):
        hsv=np.full((128,128,3),(3,230,140),np.uint8)
        hsv[25:50,25:50]=(7,220,120)
        hsv[70:100,70:100]=(20,20,130)
        maps=estimated_maps(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),paint=True,protect_paint=True)
        self.assertLess(float(maps[:15,:,4].max()),.001)
        self.assertGreater(float(maps[30:45,30:45,4].mean()),.5)
        self.assertGreater(float(maps[78:92,78:92,5].mean()),.8)
        self.assertLess(float(maps[78:92,78:92,4].mean()),.01)

    def test_flat_master_gradient_does_not_become_bump(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);master,_,provenance=self.flat_fixture(root,size=(512,512))
            ramp=np.tile(np.linspace(85,155,512,dtype=np.uint8),(512,1))
            Image.fromarray(np.repeat(ramp[...,None],3,axis=2)).save(master)
            record=prepare_flat_master(master,root/'a',provenance)
            with Image.open(record['maps']['height']['path']) as opened:
                height=np.asarray(opened,dtype=float)/255
            self.assertLess(float(np.max(np.abs(height-.5))),.003)
            self.assertLessEqual(record['height_amplitude_max_m'],.00008)
            self.assertEqual(record['physical_status'],'estimated_not_measured_PBR')


if __name__=='__main__':unittest.main()
