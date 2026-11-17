"""Refine evidence-listed edges on existing authored assets and render fixed-view comparisons."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import bmesh
import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_display_worker import canonicalize_mesh, reset
from blender_worker import load_mesh, make_camera
from boost_six_view_oxide import mesh_signature
from display_batch.common import atomic_json, file_hash
from refine_six_view_meshes import packed_images, subject_mesh
from render_six_view_review import rig


PROFILES = {
    'cast': (.028, .003, 4),
    'machine': (.019, .0024, 4),
    'formed': (.022, .0018, 3),
    'polymer': (.026, .002, 3),
}


def points(mesh):
    xyz = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get('co', xyz.ravel())
    return xyz


def mesh_topology(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    euler = len(bm.verts) - len(bm.edges) + len(bm.faces)
    overfull = sum(len(edge.link_faces) > 2 for edge in bm.edges)
    boundary = {edge for edge in bm.edges if edge.is_boundary}
    loops = 0
    while boundary:
        loops += 1
        todo = [boundary.pop()]
        while todo:
            edge = todo.pop()
            for vertex in edge.verts:
                neighbors = boundary.intersection(vertex.link_edges)
                todo.extend(neighbors)
                boundary.difference_update(neighbors)
    bm.free()
    return euler, overfull, loops


def bevel_mesh(obj, profile, overall_span):
    original = obj.data
    raw = points(original)
    span = np.ptp(raw, axis=0)
    if min(span) <= overall_span * 1e-5:
        return {'status':'too_thin'}
    ratio, cap, segments = PROFILES[profile]
    width = min(ratio * float(min(span)), cap * overall_span)
    topology = mesh_topology(original)
    materials = list(original.materials)
    if topology[1]:
        return {'status':'nonmanifold_source'}
    modifier = obj.modifiers.new('Localized manufactured edge', 'BEVEL')
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = 'ANGLE'
    modifier.angle_limit = math.radians(22)
    modifier.use_clamp_overlap = True
    modifier.harden_normals = True
    depsgraph = bpy.context.evaluated_depsgraph_get()
    new = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), preserve_all_data_layers=True,
                                         depsgraph=depsgraph)
    obj.modifiers.remove(modifier)
    growth = len(new.polygons) - len(original.polygons)
    if growth <= 0:
        bpy.data.meshes.remove(new)
        return {'status':'already_curved_or_no_bevel_edge'}
    change = points(new)
    extrema_error = max(float(abs(raw.min(axis=0) - change.min(axis=0)).max()),
                        float(abs(raw.max(axis=0) - change.max(axis=0)).max()))
    if (mesh_topology(new) != topology or extrema_error > width * 1.01
            or [layer.name for layer in original.uv_layers] != [layer.name for layer in new.uv_layers]
            or materials != list(new.materials)):
        bpy.data.meshes.remove(new)
        return {'status':'bevel_geometry_guard'}
    obj.data = new
    return {'status':'beveled', 'new_faces':growth, 'width_fraction':width/overall_span,
            'segments':segments, 'volume_topology':list(topology)}


def setup_render(scene, resolution):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.view_settings.exposure = -.1
    world = scene.world or bpy.data.worlds.new('Inspection world')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.41,.41,.41,1.)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .59
    scene.world = world


def render_view(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def build(row, recipe, folder, resolution):
    source = Path(row['mesh'])
    if file_hash(source) != recipe['source_sha256']:
        raise ValueError('Source SHA-256 changed: '+row['id'])
    if any(not Path(row[key]).is_file() for key in ('board','rust')):
        raise ValueError('Missing evidence board or reference asset')
    folder.mkdir(parents=True)
    scene = reset('OPTIX')
    setup_render(scene, resolution)
    previous = canonicalize_mesh(load_mesh(source))
    previous_xyz = points(previous.data).copy()
    before_faces = len(previous.data.polygons)
    outward = Vector((1.,.34,.26)).normalized()
    up = outward.cross(Vector((0.,0.,1.)).cross(outward)).normalized()
    make_camera(scene, (5.,0.,0.), (0,0,0), ortho_scale=1.)
    rig(scene, previous, outward, up)
    render_view(scene, folder/'before.png')
    bpy.data.objects.remove(previous, do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)
    results = {}

    def refine_parts(objects):
        by_name = {o.name:o for o in objects}
        if not set(recipe['parts']).issubset(by_name):
            raise ValueError('Recipe refers to an absent imported component')
        scale = max(float(np.ptp(points(o.data),axis=0).max()) for o in objects)
        for name in recipe['parts']:
            results[name] = bevel_mesh(by_name[name], recipe['profile'], scale)

    subject = canonicalize_mesh(load_mesh(source, before_join=refine_parts))
    xyz = points(subject.data)
    if not any(report['status']=='beveled' for report in results.values()):
        raise ValueError('No supported edge could be refined')
    if not np.allclose(np.stack((xyz.min(axis=0), xyz.max(axis=0))),
                       np.stack((previous_xyz.min(axis=0), previous_xyz.max(axis=0))), atol=.002, rtol=0.):
        raise ValueError('Refined mesh exceeds source envelope tolerance')
    face_count = len(subject.data.polygons)
    geometry_signature = mesh_signature(subject)
    scene['source_mesh_hash'] = recipe['source_sha256']
    scene['mesh_detail_candidate'] = True
    scene['acceptance'] = 'appearance_review_contact_unverified'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.file.pack_all()
    textures = packed_images()
    after = folder/'candidate.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(after), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(after), load_ui=False, use_scripts=False)
    checked = subject_mesh()
    if mesh_signature(checked)!=geometry_signature or packed_images()!=textures:
        raise ValueError('Candidate geometry, UV or texture readback mismatch')
    if any(im.source=='FILE' and not im.packed_file for im in bpy.data.images):
        raise ValueError('Unpacked texture on readback')
    render_view(bpy.context.scene, folder/'after.png')
    return {'status':'review','id':row['id'],'part_evidence':recipe['basis'],'parts':results,
            'source_mesh_hash':recipe['source_sha256'],'reopened':True,
            'subject':checked.name,'path':str(after),'sha256':file_hash(after),
            'geometry_signature':geometry_signature, 'faces_before':before_faces, 'faces':face_count,
            'packed_textures':len(textures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',nargs='+')
    parser.add_argument('--resolution',type=int,choices=(512,768,1024),default=768)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    recipe = json.loads(args.recipe.read_text(encoding='utf8'))
    if recipe['version'] != 'inventory-detail-v1':
        raise ValueError('Unrecognized detail recipe')
    inventory = {r['id']:r for r in json.loads(args.inventory.read_text(encoding='utf8'))}
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    records = {}
    started = time.monotonic()
    for aid in args.assets or recipe['assets']:
        try:
            records[aid] = build(inventory[aid], recipe['assets'][aid], output/aid, args.resolution)
        except (OSError, ValueError, RuntimeError) as error:
            records[aid] = {'id':aid,'status':'blocked','reason':str(error)}
        print('DETAIL',aid,records[aid]['status'],records[aid].get('reason',''),flush=True)
        atomic_json(output/'asset-candidates.json', {'version':'packed-inventory-candidates-v1',
                    'assets':records,'seconds':round(time.monotonic()-started,3)})


if __name__ == '__main__':
    main()
