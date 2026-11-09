"""Reopen revised Blender files and verify the requested object counts and shell."""
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def fingerprint(obj):
    h = hashlib.sha256()
    for vertex in obj.data.vertices:
        h.update(('%0.9f,%0.9f,%0.9f;' % tuple(vertex.co)).encode('ascii'))
    for face in obj.data.polygons:
        h.update((','.join(str(i) for i in face.vertices) + ';').encode('ascii'))
    return h.hexdigest()


def check_shell(obj, dim):
    radius, y0, y1 = dim['radius'], dim['y0'], dim['y1']
    hits = 0
    for fraction in (.1,.3,.5,.7,.9):
        y = y0 + (y1-y0)*fraction
        for index in range(36):
            a = math.tau*index/36
            vec = Vector((math.cos(a),0,math.sin(a)))
            hit, loc, _, _ = obj.ray_cast(vec*(radius*2)+Vector((0,y,0)), -vec, distance=radius*2.1)
            hits += bool(hit and abs(math.hypot(loc.x,loc.z)-radius)<.001)
    return hits


def main():
    data = json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf8'))
    bpy.ops.wm.open_mainfile(filepath=data['old31'], load_ui=False)
    eye_names = ('rear_clevis','rear_bushing','front_clevis','front_bushing')
    before = {name:fingerprint(bpy.data.objects[name]) for name in eye_names}
    checked = []
    for item in data['items']:
        bpy.ops.wm.open_mainfile(filepath=item['blend'], load_ui=False)
        objects = [obj for obj in bpy.context.scene.objects if obj.type=='MESH' and not obj.hide_render]
        names = [obj.name for obj in objects]
        images = [image for image in bpy.data.images if image.source=='FILE']
        assert images and all(image.packed_file for image in images), 'Unpacked texture in ' + item['id']
        if item['id']=='13':
            assert names.count('Can rear body')==1 and not any(n.startswith('Can front ') for n in names)
        elif item['id'].startswith('19-'):
            assert sum(n.endswith(' 筒身') for n in names)==1
        elif item['id']=='31':
            assert 'barrel_full_opaque' in names and 'barrel' not in names
            assert not any(n=='piston_body' or n.startswith(('piston_ring_','gland_ring_')) for n in names)
            assert {name:fingerprint(bpy.data.objects[name]) for name in eye_names}==before
            assert check_shell(bpy.data.objects['barrel_full_opaque'],data['barrel'])==180
        else:
            raise ValueError('Unexpected item')
        checked.append(item['id'])
        print('VERIFIED',item['id'],'parts',len(names),'packed textures',len(images),flush=True)
    assert set(checked)=={'13','19-1','19-2','19-3','19-4','19-5','19-6','31'}
    print('VERIFIED ALL: one can, six single bottles, opaque barrel and unchanged eye geometry',flush=True)


main()
