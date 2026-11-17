"""Read-only PBR/normal audit of all currently selected GLB assets."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from display_batch.common import atomic_json,load_catalog


def read_glb(path):
    with Path(path).open('rb') as stream:
        header=stream.read(12)
        if header[:4]!=b'glTF':raise ValueError('not_glb')
        doc=None;binary=None
        while chunk:=stream.read(8):
            size,kind=struct.unpack('<II',chunk);data=stream.read(size)
            if kind==0x4e4f534a:doc=json.loads(data)
            elif kind==0x004e4942:binary=data
    return doc,binary


def accessor(doc,binary,index):
    spec=doc['accessors'][index];view=doc['bufferViews'][spec['bufferView']]
    types={5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}
    components={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[spec['type']]
    dtype=np.dtype(types[spec['componentType']]);stride=view.get('byteStride',components*dtype.itemsize)
    return np.ndarray((spec['count'],components),dtype=dtype,buffer=binary,
                      offset=view.get('byteOffset',0)+spec.get('byteOffset',0),strides=(stride,dtype.itemsize))


def inspect(row):
    doc,binary=read_glb(row['mesh']);errors=[];primitives=0;normals=0;triangles=0
    for mesh in doc.get('meshes',[]):
        for part in mesh['primitives']:
            primitives+=1;attributes=part['attributes']
            points=accessor(doc,binary,attributes['POSITION'])
            if not np.isfinite(points).all():errors.append('nonfinite_position')
            if 'NORMAL' in attributes:
                normal=accessor(doc,binary,attributes['NORMAL']);normals+=len(normal)
                if not np.isfinite(normal).all() or (np.linalg.norm(normal,axis=1)<.5).any():errors.append('invalid_normal')
            else:errors.append('normal_requires_importer_recompute')
            if 'indices' in part:triangles+=doc['accessors'][part['indices']]['count']//3
            if 'material' in part:
                material=doc['materials'][part['material']];pbr=material.get('pbrMetallicRoughness',{})
                if any('Texture' in key for key in pbr) and 'TEXCOORD_0' not in attributes:errors.append('missing_uv')
    materials=[]
    for material in doc.get('materials',[]):
        pbr=material.get('pbrMetallicRoughness',{})
        materials.append({'name':material.get('name'),'base_color_texture':'baseColorTexture' in pbr,
             'metallic_roughness_texture':'metallicRoughnessTexture' in pbr,
             'metallic_factor':pbr.get('metallicFactor',1),'roughness_factor':pbr.get('roughnessFactor',1),
             'normal_texture':'normalTexture' in material,'alpha_mode':material.get('alphaMode','OPAQUE'),
             'gltf_channel_contract':{'base_color':'sRGB','roughness':'G_linear','metallic':'B_linear','normal':'RGB_linear'}})
    return {'asset_id':row['id'],'mesh_hash':row['mesh_hash'],'status':'pass' if not errors else 'review',
            'errors':sorted(set(errors)),'primitives':primitives,'triangles':triangles,'normal_vectors':normals,
            'materials':materials,'source_modified':False,'scope':'attribute_and_material_contract_not_visual_acceptance'}


def main(out):
    results=[]
    for row in load_catalog().values():
        try:results.append(inspect(row))
        except Exception as error:results.append({'asset_id':row['id'],'status':'review','reason':str(error)[:200]})
    atomic_json(out,{'kind':'read_only_asset_audit','total':len(results),
                     'passed':sum(r['status']=='pass' for r in results),'records':results})
    print(json.dumps({'total':len(results),'passed':sum(r['status']=='pass' for r in results),
                     'review':[r['asset_id'] for r in results if r['status']!='pass']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();main(args.out)
