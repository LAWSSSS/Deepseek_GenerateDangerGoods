"""Remove unsupported medium-scale albedo streaks inside proven material regions.

This is not a relight or a scene colour transfer. Low-frequency source colour and
fine source detail remain; normal, roughness, metalness and protected texels stay.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .common import file_hash, signature
from .surface_maps import glb_images, save_image

VERSION='source-albedo-band-cleanup-v1'


def clean_albedo(rgb, mask, strength=.75):
    rgb=np.float32(rgb)/255.
    weight=np.float32(mask)
    if not weight.any():return np.uint8(np.rint(rgb*255)),{'changed_texels':0}
    sigma=max(2.,min(rgb.shape[:2])*.014)
    den=cv2.GaussianBlur(weight,(0,0),sigma)
    smooth=cv2.GaussianBlur(rgb*weight[...,None],(0,0),sigma)/np.maximum(den[...,None],1e-6)
    fine=rgb-cv2.GaussianBlur(rgb,(0,0),1.2)
    target=np.clip(smooth+fine*.4,0,1)
    # Keep protected boundaries exact; normalized blur cannot draw in label ink.
    interior=cv2.erode(weight,np.ones((3,3),np.uint8))
    amount=np.clip(interior,0,1)*strength
    out=rgb*(1-amount[...,None])+target*amount[...,None]
    result=np.uint8(np.rint(np.clip(out,0,1)*255))
    protected=weight==0
    result[protected]=np.uint8(np.rint(rgb[protected]*255))
    delta=np.abs(result.astype(float)-rgb*255)
    return result,{'changed_texels':int(np.any(delta>.5,axis=2).sum()),'sigma_texels':sigma,
                   'strength':strength,'rgb_mae_255':float(delta[weight>0].mean()),
                   'protected_texels_equal':bool(np.array_equal(result[protected],np.uint8(np.rint(rgb[protected]*255))))}


def prepare_source_detail(asset, materials, root, profile):
    settings=profile.get('source_detail_cleanup',{})
    if not settings:return materials
    source,read=glb_images(asset['mesh'])
    output={name:dict(entry) for name,entry in materials.items()}
    for material in source:
        name=material.get('name');entry=output.get(name)
        if not entry or not entry.get('eligible') or entry.get('protected'):continue
        pbr=material.get('pbrMetallicRoughness',{});image=read(pbr.get('baseColorTexture'))
        if image is None:continue
        # Preserve source binding/factors; only the exact original image node is substituted.
        if entry.get('mask'):
            with Image.open(entry['mask']) as im:regions=np.asarray(im.convert('RGB'))
            regions=cv2.resize(regions,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
            eligible=regions.max(axis=2)>127
        else:eligible=np.full(image.shape[:2],any(entry.get('constant',[])[:3]))
        if settings.get('neutral_only'):
            eligible&=(image.max(axis=2).astype(int)-image.min(axis=2).astype(int))<32
        cleaned,stats=clean_albedo(image,eligible,float(settings.get('strength',.75)))
        key=signature(VERSION,asset['mesh_hash'],name,entry,settings,file_hash(__file__))
        path=Path(root)/key/'albedo.png';path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():save_image(Image.fromarray(cleaned),path)
        entry['albedo_variant']={'path':str(path),'sha256':file_hash(path),'source_texture_index':pbr['baseColorTexture']['index'],
                                 'stats':stats,'basis':settings['evidence'],'normal_preserved':True,
                                 'roughness_preserved':True,'version':VERSION}
    return output
