"""Reopen both 10 model versions and check every retained mesh, transform and UV."""
import hashlib
import sys
from pathlib import Path

import bpy

REMOVED = {'Detached handle bar', 'Handle bar bore liner', 'Handle bar top rim'}
REQUIRED = {'Jack body', 'Cast base', 'Chrome ram', 'Pump handle socket', 'Carry handle loop'}


def snapshot(path):
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    result = {}
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        h = hashlib.sha256()
        for vertex in obj.data.vertices:
            h.update(('%0.9f,%0.9f,%0.9f;' % tuple(vertex.co)).encode('ascii'))
        for face in obj.data.polygons:
            h.update((','.join(str(i) for i in face.vertices) + ';').encode('ascii'))
        for layer in obj.data.uv_layers:
            h.update(layer.name.encode('utf8'))
            for loop in layer.data:
                h.update(('%0.9f,%0.9f;' % tuple(loop.uv)).encode('ascii'))
        materials = tuple(material.name if material else None for material in obj.data.materials)
        result[obj.name] = (h.hexdigest(), tuple(tuple(row) for row in obj.matrix_world), materials)
    return result


def main():
    old_path, new_path = map(Path, sys.argv[sys.argv.index('--') + 1:])
    before, after = snapshot(old_path), snapshot(new_path)
    assert REMOVED.issubset(before)
    assert after == {name:value for name,value in before.items() if name not in REMOVED}
    assert REQUIRED.issubset(after)
    images = [image for image in bpy.data.images if image.source == 'FILE']
    assert images and all(image.packed_file for image in images)
    print('VERIFIED: removed only the detached handle assembly; retained', len(after),
          'meshes, transforms, UVs and material assignments', flush=True)


main()
