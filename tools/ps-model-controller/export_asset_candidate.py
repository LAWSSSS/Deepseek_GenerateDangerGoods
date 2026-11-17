"""Create a packed diagnostic candidate, then reopen and verify it in Blender."""
import json
import sys
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import bpy

sys.path.insert(0,str(Path(__file__).resolve().parent))
from blender_display_worker import render_parts
from display_batch.render_readback import material_state


def source_inputs():
    """Record actual Principled input graphs, excluding diagnostic output nodes."""
    def value(v):
        try:return list(v)
        except TypeError:return v
    def node_state(node,seen):
        if node in seen:return {'cycle':True}
        seen=seen|{node}
        state={'type':node.type,'inputs':{s.name:(node_state(s.links[0].from_node,seen) if s.is_linked else value(s.default_value))
                                        for s in node.inputs if hasattr(s,'default_value')}}
        if node.type=='TEX_IMAGE' and node.image:
            im=node.image
            state['image_sha256']=hashlib.sha256(bytes(im.packed_file.data)).hexdigest() if im.packed_file else im.filepath
        if hasattr(node,'operation'):state['operation']=node.operation
        return state
    result={}
    for material in bpy.data.materials:
        if not material.use_nodes:continue
        shader=next((n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED' and not n.get('display_deposit')),None)
        if shader:
            result[re.sub(r'\.\d{3}$','',material.name)]={key:(node_state(shader.inputs[key].links[0].from_node,set()) if shader.inputs[key].is_linked else value(shader.inputs[key].default_value)) for key in ('Base Color','Metallic','Roughness','Normal')}
    return result


def compact_readback(baseline,actual):
    def compact(value):
        if not isinstance(value,dict):return {'kind':'constant','value':value}
        return {'kind':'linked','node_type':value['type'],
                'graph_sha256':hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()}
    return {name:{'metallic':compact(values['Metallic']),'roughness':compact(values['Roughness']),
                  'normal':compact(values['Normal']),'base_color':compact(values['Base Color']),
                  'unchanged_from_source':[key for key,value in values.items() if baseline.get(name,{}).get(key)==value]}
            for name,values in actual.items() if name in baseline}


def main():
    request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf8'))
    destination=Path(request.pop('candidate_path')).resolve()
    if destination.suffix!='.blend' or destination.name!='candidate.blend':
        raise ValueError('only a separately named candidate.blend may be written')
    bpy.ops.import_scene.gltf(filepath=request['mesh'])
    baseline=source_inputs()
    render_parts(request)  # Empty parts: setup only, zero renders.
    actual=source_inputs()
    (destination.parent/'candidate-material-readback.json').write_text(json.dumps({
        'source_mesh_hash':request['mesh_hash'],'source_inputs':baseline,'actual_inputs':actual,
        'recipe':request['generated_finish']['recipe'],'controls':request['generated_finish'].get('controls',{}),
        'regions':request['generated_finish']['regions'],'binding':'actual Blender input graphs before export'},ensure_ascii=False,indent=2),encoding='utf8')
    protected={name:baseline.get(name)==actual.get(name) for name,entry in request['generated_finish']['regions']['materials'].items() if not entry.get('eligible')}
    if not all(protected.values()):raise ValueError('protected source material input graph changed')
    scene=bpy.context.scene
    for obj in scene.objects:
        if obj.type=='MESH' and any(m and m.name=='neutral receiver' for m in obj.data.materials):
            obj.visible_camera=False
    scene.render.resolution_x=scene.render.resolution_y=1024
    scene.render.resolution_percentage=100
    scene.view_settings.view_transform='AgX'
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA'
    scene['asset_profile']=json.dumps(request['generated_finish'].get('asset_profile'),ensure_ascii=False)
    scene['material_signature']=request['generated_finish']['signature']
    scene['source_mesh_hash']=request['mesh_hash']
    scene['acceptance']='candidate_only_user_review_required'
    objects=[o for o in scene.objects if o.type=='MESH']
    counts={'mesh_objects':len(objects),'vertices':sum(len(o.data.vertices) for o in objects),
            'faces':sum(len(o.data.polygons) for o in objects),'materials':len(bpy.data.materials)}
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(destination),check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(destination))
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    reopened={'mesh_objects':len(objects),'vertices':sum(len(o.data.vertices) for o in objects),
              'faces':sum(len(o.data.polygons) for o in objects),'materials':len(bpy.data.materials)}
    if reopened!=counts:raise ValueError('candidate reopen geometry/material count mismatch')
    images=[image for image in bpy.data.images if image.source=='FILE']
    if any(not image.packed_file for image in images):raise ValueError('unpacked candidate dependency')
    # The original Principled branch is deliberately retained. Its unchanged
    # inputs cannot prove that the new response survived save/reopen: inspect
    # the actual output-connected branch on the reopened candidate instead.
    materials={m.name:m for obj in objects for m in obj.data.materials if m and m.name!='neutral receiver'}
    reopened_state=material_state(SimpleNamespace(data=SimpleNamespace(materials=list(materials.values()))))
    reference_checks={}
    if request['generated_finish'].get('reference'):
        states={re.sub(r'\.\d{3}$','',row['material']):row for row in reopened_state['materials']}
        for name,entry in request['generated_finish']['regions']['materials'].items():
            if not entry.get('eligible'):continue
            row=states.get(name,{})
            response=row.get('reference_response',{})
            reference_checks[name]=bool(row.get('physical_layers_applied') and response.get('status')=='measured'
                                       and all(r.get('connected_to_surface') for r in response.get('responses',[]))
                                       and len(response.get('responses',[]))==2)
        if not all(reference_checks.values()):raise ValueError('reopened candidate lost connected reference response')
    (destination.parent/'candidate-reopened-materials.json').write_text(
        json.dumps(reopened_state,ensure_ascii=False,indent=2),encoding='utf8')
    (destination.parent/'candidate-check.json').write_text(json.dumps({
        'status':'pass','path':str(destination),'reopened':True,'counts':counts,'packed_images':len(images),
        'blender':bpy.app.version_string,'render_calls':0,'source_mesh_hash':request['mesh_hash'],
        'material_signature':request['generated_finish']['signature'],'geometry_modified':False,
        'material_readback':compact_readback(baseline,actual),
        'reopened_material_readback_path':'candidate-reopened-materials.json',
        'reopened_material_state_sha256':reopened_state['state_sha256'],
        'reference_response_connected':reference_checks,
        'reference_response_scope':'actual_reopened_candidate_output_connections_not_visual_quality',
        'protected_material_graphs_equal':protected,
        'changed_source_inputs':{name:[key for key in baseline.get(name,{}) if baseline[name][key]!=actual.get(name,{}).get(key)] for name in actual if name in baseline}},indent=2),encoding='utf8')


if __name__=='__main__':main()
