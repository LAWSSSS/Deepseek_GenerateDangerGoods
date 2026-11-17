"""Render every library model in six canonical directions for visual audit."""
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
import blender_worker as worker

req=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf8'))
rows=[json.loads(x) for x in Path(req['library']).read_text(encoding='utf8').splitlines() if x.strip()]
out=Path(req['out']);out.mkdir(parents=True,exist_ok=True)
scene=worker.setup_scene('CPU',512)
scene.render.resolution_x=scene.render.resolution_y=256
scene.cycles.samples=4
for obj in tuple(bpy.data.objects):
    if obj.type=='LIGHT':bpy.data.objects.remove(obj,do_unlink=True)
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value=(1,1,1,1)
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=1.0
directions={'front':(1,0,0),'back':(-1,0,0),'left':(0,1,0),'right':(0,-1,0),
            'top':(0,0,1),'bottom':(0,0,-1)}
records=[]
for row in rows:
    number=row['number'];folder=out/f'{number:02d}';folder.mkdir(exist_ok=True)
    entry={'number':number,'asset_id':row['asset_id'],'file':row['file'],'status':'rendered','views':{}}
    obj=None
    try:
        obj=worker.load_mesh(row['mesh'])
        pts=worker.world_vertices(obj)
        lo=Vector([min(p[i] for p in pts) for i in range(3)])
        hi=Vector([max(p[i] for p in pts) for i in range(3)])
        center=(lo+hi)/2; span=max(hi-lo)
        if span<=0:raise ValueError('empty mesh')
        for name,d in directions.items():
            camera=worker.make_camera(scene,center+Vector(d)*span*5,center,ortho_scale=span*1.5)
            path=folder/f'{name}.png';worker.render(scene,path)
            entry['views'][name]=str(path)
            bpy.data.objects.remove(camera,do_unlink=True)
        entry['bounds']=[list(lo),list(hi)]
    except Exception as exc:
        entry['status']='error';entry['reason']=str(exc)[:300]
    finally:
        if obj is not None:bpy.data.objects.remove(obj,do_unlink=True)
    records.append(entry)
    (out/'render-results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    print('AUDIT',number,entry['status'],flush=True)
