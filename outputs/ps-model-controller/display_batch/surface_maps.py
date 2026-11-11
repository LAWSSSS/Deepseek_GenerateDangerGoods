"""Build UV masks from the original glTF material channels and component names."""
import io
import json
import os
import re
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import atomic_json, file_hash, signature
from .photography import linear


VERSION = 'scanned-pbr-uv-partition-v1'
CHANNELS = ('metal','paint','rust','polymer')


def save_image(image,path,format='PNG',**options):
    """Publish a complete image before its cache record becomes visible."""
    fd, temporary = tempfile.mkstemp(dir=path.parent,prefix=path.name,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            image.save(stream,format=format,**options)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def partition(name, rgb, metal, roughness, classes):
    """Unclassified texels have zero weight in all four channels."""
    name = name.lower()
    shape = metal.shape
    empty = np.zeros((*shape,4),np.uint8)
    if re.search(r'glass|transparent|半透明|玻璃|filter.mesh|编织|(^|\s)(label|mark)(\s|$)',name):
        # Some body materials mention an embedded label. Only standalone labels
        # are excluded here, with neutral lettering preserved in the UV map below.
        if not (('body' in name or 'atlas' in name) and 'glass' not in name):
            return empty
    mixed = bool(re.search(r'\batl(as)?\b|mixed',name))
    polymer = not mixed and bool(re.search(
        r'plastic|rubber|polypropylene|dielectric|gasket|seal|bushing|hose|塑料|橡胶|密封',name))
    if polymer or (classes and set(classes) <= {'plastic','rubber'}):
        empty[...,3] = 255
        return empty
    metallic = metal >= .66
    dielectric = metal <= .22
    explicit_paint = bool(re.search(r'paint|enamel|weathered.blue|chipped.blue|漆',name))
    explicit_rust = bool(re.search(r'rust|oxide|锈',name))
    spread = rgb.max(axis=2)-rgb.min(axis=2)
    # Dark seals, pale labels and neutral unknown regions remain untouched.
    colored = (spread>.105)&(rgb.max(axis=2)>.17)&(rgb.max(axis=2)<.90)
    oxide_color = (rgb[...,0]>rgb[...,1]*1.12)&(rgb[...,1]>rgb[...,2]*1.04)
    rust = dielectric & colored & oxide_color & (roughness>=.73) & (explicit_rust or 'rust' in classes)
    paint = dielectric & colored & ~rust & (explicit_paint or (mixed and 'paint' in classes and 'print' not in name and 'decal' not in name))
    empty[...,0] = metallic*255
    empty[...,1] = paint*255
    empty[...,2] = rust*255
    return empty


def glb_images(path):
    """Read embedded images lazily. Materials with unsupported bindings stay as authored."""
    with Path(path).open('rb') as f:
        magic, version, _ = struct.unpack('<4sII',f.read(12))
        if magic!=b'glTF' or version!=2:
            raise ValueError('unsupported_gltf_material_source')
        length, _ = struct.unpack('<I4s',f.read(8))
        doc = json.loads(f.read(length))
        bin_start = f.tell()+8
    decoded = {}

    def image(info):
        if not info:
            return None
        if info.get('texCoord',0)!=0 or info.get('extensions'):
            raise ValueError('different_or_transformed_uv')
        idx = doc['textures'][info['index']]['source']
        if idx not in decoded:
            desc = doc['images'][idx]
            view = doc['bufferViews'][desc['bufferView']]
            if view.get('buffer',0)!=0:
                raise ValueError('external_texture_buffer')
            with Path(path).open('rb') as f:
                f.seek(bin_start+view.get('byteOffset',0))
                data = f.read(view['byteLength'])
            with Image.open(io.BytesIO(data)) as im:
                im = im.convert('RGB')
                im.thumbnail((2048,2048),Image.Resampling.BOX)
                decoded[idx] = np.asarray(im)
        return decoded[idx]

    return doc.get('materials',[]), image


def prepare_uv_masks(asset, root):
    classes = asset['semantic']['material_classes']
    key = signature(VERSION,file_hash(__file__),asset['mesh_hash'],classes)
    folder = Path(root)/key
    record = folder/'masks.json'
    if record.is_file():
        cached = json.loads(record.read_text(encoding='utf8'))
        if all(Path(m['mask']).is_file() and file_hash(m['mask']) == m['mask_hash']
               for m in cached['materials'].values() if m.get('mask')):
            return cached,True
    materials, read_image = glb_images(asset['mesh'])
    folder.mkdir(parents=True,exist_ok=True)
    rows = {}
    for index, material in enumerate(materials):
        name = material.get('name') or f'Material_{index}'
        pbr = material.get('pbrMetallicRoughness',{})
        transparency = (material.get('alphaMode','OPAQUE')!='OPAQUE' or
            bool(material.get('extensions',{}).get('KHR_materials_transmission')))
        if transparency:
            rows[name] = {'coverage':[0.]*4,'basis':'preserved_transparent_material'}
            continue
        try:
            base = read_image(pbr.get('baseColorTexture'))
            mr = read_image(pbr.get('metallicRoughnessTexture'))
        except (ValueError, KeyError, IndexError, OSError):
            rows[name] = {'coverage':[0.]*4,'basis':'unsupported_uv_binding_preserved'}
            continue
        h,w = base.shape[:2] if base is not None else mr.shape[:2] if mr is not None else (1,1)
        factor = np.asarray(pbr.get('baseColorFactor',[1,1,1,1])[:3],np.float32)
        # Use display RGB for hue tests, with the source factor applied in linear light.
        rgb = np.broadcast_to(factor**(1/2.2),(h,w,3)) if base is None else (
            linear(np.float32(base)/255)*factor)**(1/2.2)
        metal = np.full((h,w),pbr.get('metallicFactor',1.),np.float32)
        rough = np.full((h,w),pbr.get('roughnessFactor',1.),np.float32)
        if mr is not None:
            if mr.shape[:2]!=(h,w):
                mr = cv2.resize(mr,(w,h),interpolation=cv2.INTER_LINEAR)
            metal *= np.float32(mr[...,2])/255
            rough *= np.float32(mr[...,1])/255
        masks = partition(name,rgb,metal,rough,classes)
        coverage = (masks>0).mean(axis=(0,1)).tolist()
        entry = {'coverage':coverage,'basis':'source_uv_metallic_roughness_and_component',
                 'size':[w,h]}
        if any(coverage):
            if np.all(masks==masks[0,0]):
                entry['constant'] = (masks[0,0]/255).tolist()
            else:
                path = folder/f'material-{index:02d}.png'
                save_image(Image.fromarray(masks),path)
                entry.update(mask=str(path),mask_hash=file_hash(path))
        rows[name] = entry
    result = {'signature':key,'channels':CHANNELS,'materials':rows,'path':str(record)}
    atomic_json(record,result)
    return result,False
