"""Create local geometry candidates from the six-view material candidates."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def subject_mesh():
    subject, = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'
                and not any(mat and mat.name == 'neutral receiver' for mat in obj.data.materials)]
    return subject


def slot_for(mesh, case, part):
    region, = [row for material in case['finish']['regions']['materials'].values()
               for row in material['components'] if part in row['parts']]
    slot, = [i for i, mat in enumerate(mesh.materials) if mat and mat.get('component_pbr_region') == region['id']]
    return slot


def add_lathe(bm, center, axis, profile, slot, segments, *, closed=False):
    u = axis.orthogonal().normalized()
    v = axis.cross(u).normalized()
    ring_dirs = [math.cos(i * math.tau / segments) * u + math.sin(i * math.tau / segments) * v
                 for i in range(segments)]
    rings = [[bm.verts.new(center + depth * axis + radius * angle) for angle in ring_dirs]
             for depth, radius in profile]
    faces = []
    for a, b in zip(rings, rings[1:] + (rings[:1] if closed else [])):
        for j in range(segments):
            k = (j + 1) % segments
            face = bm.faces.new((a[j], a[k], b[k], b[j]))
            face.material_index, face.smooth = slot, True
            faces.append(face)
    if not closed:
        for ring in (rings[0][::-1], rings[-1]):
            cap = bm.faces.new(ring)
            cap.material_index = slot
            faces.append(cap)
    bmesh.ops.recalc_face_normals(bm, faces=faces)
    return len(faces)


def threaded_stem(bm, mesh, case, edit):
    slot = slot_for(mesh, case, edit['part'])
    cx, cy = edit['center_xy']
    low, high = edit['span_z']
    pitch, crest = edit['pitch'], edit['height']
    if not .003 <= pitch <= .02 or not .0003 <= crest <= .004:
        raise ValueError('Thread dimensions are outside the inspection range')
    radii = [(v.co.z, math.hypot(v.co.x-cx, v.co.y-cy)) for f in bm.faces if f.material_index == slot for v in f.verts
             if edit['core_profile'][0][0]-.00001 <= v.co.z <= edit['core_profile'][-1][0]+.00001]
    if not radii or abs(max(r for _, r in radii)-edit['core_profile'][0][1]) > .001:
        raise ValueError('The measured stem differs from the threaded part')
    levels, envelopes = zip(*edit['core_profile'])
    turns = (high-low) / pitch
    stations = math.ceil(turns * 72)
    section = [(-.34, -.0011),(-.22, .00018),(-.10, crest),(.08, crest*.91),(.24, .0002),(.34, -.0011)]
    rings = []
    for i in range(stations+1):
        fraction = i/stations
        z = low + (high-low)*fraction
        angle = turns*math.tau*fraction
        fade = min(1., fraction*turns*2., (1.-fraction)*turns*2.)
        ring = []
        for axial, relief in section:
            height = z+axial*pitch
            radius = float(np.interp(height, levels, envelopes)) - .0011 + (relief+.0011)*fade
            ring.append(bm.verts.new((cx+radius*math.cos(angle),cy+radius*math.sin(angle),height)))
        rings.append(ring)
    faces = []
    for first, second in zip(rings, rings[1:]):
        for j in range(len(section)):
            k = (j+1)%len(section)
            face = bm.faces.new((first[j],first[k],second[k],second[j]))
            face.material_index, face.smooth = slot, True
            faces.append(face)
    for cap in (rings[0][::-1],rings[-1]):
        face = bm.faces.new(cap)
        face.material_index = slot
        faces.append(face)
    bmesh.ops.recalc_face_normals(bm, faces=faces)
    return {'kind':'thread','turns':turns,'pitch_model_units':pitch,'added_faces':len(faces)}


def bushing_profile(half, inner, outer, bevel, groove):
    positive = [(half-bevel,inner)]
    for index in range(1,9):
        angle = math.pi*index/16.
        positive.append((half-bevel+bevel*math.sin(angle),inner+bevel*(1.-math.cos(angle))))
    for index in range(1,24):
        t = index/24.
        r = inner+bevel+(outer-inner-2.*bevel)*t
        depth = groove*(math.exp(-((t-.29)/.068)**2) + .8*math.exp(-((t-.72)/.058)**2))
        positive.append((half-depth,r))
    for index in range(9):
        angle = math.pi*index/16.
        positive.append((half-bevel*(1.-math.cos(angle)),outer-bevel+bevel*math.sin(angle)))
    return positive + [(-depth,radius) for depth,radius in reversed(positive)]


def refine_bushing(bm, mesh, case, edit, origin, changed):
    slot = slot_for(mesh, case, edit['part'])
    y, z = edit['center_yz']
    half, inner, outer = (edit[key] for key in ('half_x','inner','outer'))
    bevel = edit['bevel']
    if not 0. < 2.*bevel < outer-inner or not 0. <= edit['groove'] <= .00025:
        raise ValueError('Bushing profile would lose its measured lip')
    def radius(point):
        return math.hypot(point.y-y, point.z-z)
    faces = [face for face in bm.faces if face.material_index == slot
             and all(abs(v.co.x)<=half+.00009 and inner-.00009<=radius(v.co)<=outer+.00009 for v in face.verts)]
    vertices = {v for f in faces for v in f.verts}
    if len(vertices)<100 or abs(min(radius(v.co) for v in vertices)-inner)>.00015:
        raise ValueError('Bushing measurement does not select the original through-bore')
    selected = set(faces)
    if any(f not in selected for v in vertices for f in v.link_faces):
        raise ValueError('Bushing edit intersects an unaudited adjoining mesh')
    changed.update(f[origin] for f in faces)
    bmesh.ops.delete(bm, geom=faces, context='FACES')
    count = add_lathe(bm, Vector((0.,y,z)), Vector((1.,0.,0.)),
                      bushing_profile(half,inner,outer,bevel,edit['groove']), slot, 240, closed=True)
    return {'kind':'bushing','part':edit['part'],'before_faces':len(faces), 'added_faces':count,
            'bore_radius_model_units':inner,'radial_segments':240}


def rounded_shoulder(bm, mesh, case, edit, origin, changed):
    body = slot_for(mesh, case, edit['part'])
    seams = slot_for(mesh, case, edit['seam'])
    source, target = zip(*edit['height_profile'])
    if min(np.diff(source)) <= 0 or min(np.diff(target)) <= 0 or source[-1] != target[-1]:
        raise ValueError('Shoulder height mapping must keep positive sections and the neck')
    cutoff = source[0]
    shoulder = [f for f in bm.faces if f.material_index == body and max(v.co.z for v in f.verts)>cutoff]
    seam = [f for f in bm.faces if f.material_index == seams and min(v.co.z for v in f.verts)>.12]
    if not shoulder or not seam:
        raise ValueError('Missing model shoulder or seam')
    affected = set(shoulder+seam)
    changed.update(f[origin] for f in affected)
    edges = {edge for face in shoulder for edge in face.edges
             if all(f in affected for f in edge.link_faces) and min(v.co.z for v in edge.verts)>=cutoff}
    if len(edges) < 80:
        raise ValueError('Shoulder subdivision lacks a continuous measured band')
    start_count = len(bm.faces)
    bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=2, use_grid_fill=True)
    vertices = {v for f in bm.faces if (f.material_index in (body,seams) and f[origin] in changed)
                for v in f.verts if v.co.z > cutoff}
    for vertex in vertices:
        vertex.co.z = float(np.interp(vertex.co.z,source,target))
    return {'kind':'shoulder','old_band_faces':len(affected),'subdivision_edges':len(edges),
            'added_faces':len(bm.faces)-start_count,'adjusted_vertices':len(vertices)}


def new_surface(mesh, name, colour, roughness, metal=0.):
    surface = bpy.data.materials.new(name)
    surface.use_nodes = True
    shader = surface.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*colour,1.)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metal
    mesh.materials.append(surface)
    return len(mesh.materials)-1, surface, shader


def add_pressure_gauge(bm, mesh, edit, sheet):
    if sha256(sheet.read_bytes()).hexdigest()!=edit['reference_sha256']:
        raise ValueError('Pressure gauge reference crop has changed')
    center = Vector(edit['center'])
    axis = Vector(edit['normal']).normalized()
    up = Vector((0,0,1))-axis*axis.z
    up.normalize()
    right = up.cross(axis)
    steel, _, _ = new_surface(mesh,'Gauge brushed bezel',(.28,.27,.24),.32,.92)
    brass, _, _ = new_surface(mesh,'Gauge brass attachment',(.37,.22,.071),.43,.91)
    dial, dial_mat, dial_bsdf = new_surface(mesh,'Gauge face from front reference',(.42,.41,.37),.74)
    glass, _, glass_bsdf = new_surface(mesh,'Gauge protective lens',(.99,.99,.99),.075)
    glass_bsdf.inputs['Transmission Weight'].default_value = 1.
    glass_bsdf.inputs['IOR'].default_value = 1.47
    nodes = dial_mat.node_tree.nodes
    texture = nodes.new('ShaderNodeTexImage')
    texture.image = bpy.data.images.load(str(sheet.resolve()), check_existing=True)
    texture.image.pack()
    uv_input = nodes.new('ShaderNodeUVMap')
    uv_input.uv_map = 'UVMap'
    dial_mat.node_tree.links.new(uv_input.outputs[0],texture.inputs['Vector'])
    dial_mat.node_tree.links.new(texture.outputs['Color'],dial_bsdf.inputs['Base Color'])
    rim = edit['radius']
    faces = add_lathe(bm,center,axis,[(-.030,rim*.31),(.010,rim*.31),(.014,rim*.47)],brass,72)
    faces += add_lathe(bm,center,axis,[(.001,rim*.94),(.006,rim),(.019,rim),(.024,rim*.91),
                                         (.024,rim*.80),(.018,rim*.79),(.005,rim*.82)],steel,144,closed=True)
    # The small disk is mapped only to the dial, excluding bezel and valve body.
    radius = rim*.805
    disk_center = bm.verts.new(center+axis*.0175)
    ring = [bm.verts.new(disk_center.co+radius*(right*math.cos(i*math.tau/144)+up*math.sin(i*math.tau/144)))
            for i in range(144)]
    uv_layer = bm.loops.layers.uv.get('UVMap')
    x,y,rx,ry = edit['dial_pixels_1080_by_720']
    for i,a in enumerate(ring):
        face = bm.faces.new((disk_center,a,ring[(i+1)%144]))
        face.material_index = dial
        for loop in face.loops:
            point = (loop.vert.co-disk_center.co)/radius
            loop[uv_layer].uv = ((x+rx*point.dot(right))/1080.,1.-(y-ry*point.dot(up))/720.)
    faces += len(ring)
    faces += add_lathe(bm,center,axis,[(.0192,rim*.793),(.0202,rim*.803),(.0216,rim*.803),(.0227,rim*.793)],glass,128)
    return {'kind':'pressure_gauge','added_faces':faces,'diameter_model_units':2.*rim,'face_source':sheet.name}


def face_key(mesh, corner):
    loop = mesh.loops[corner]
    return (tuple(mesh.vertices[loop.vertex_index].co), tuple(tuple(layer.data[corner].uv) for layer in mesh.uv_layers))


def preserve_corners(original, new, marker, changed):
    """Restore the original normals and verify coordinates and UV outside edits."""
    ids = new.attributes[marker].data
    normals = [(0.,0.,0.)]*len(new.loops)
    visited = set()
    for face in new.polygons:
        source = ids[face.index].value
        if not source or source in changed:
            continue
        if source in visited:
            raise ValueError('An unrelated face was tessellated')
        visited.add(source)
        old = original.polygons[source-1]
        before = {face_key(original,k):tuple(original.corner_normals[k].vector) for k in old.loop_indices}
        if len(face.loop_indices)!=len(old.loop_indices) or face.material_index!=old.material_index:
            raise ValueError('An unrelated face lost material or topology')
        for corner in face.loop_indices:
            key = face_key(new,corner)
            if key not in before:
                raise ValueError('An unrelated corner lost position or UV')
            normals[corner] = before[key]
    if len(visited) != len(original.polygons)-len(changed):
        raise ValueError('Original unrelated faces are missing')
    new.normals_split_custom_set(normals)
    new.attributes.remove(new.attributes[marker])
    return len(visited)


def bushing_holes(bm, operations):
    tree = BVHTree.FromBMesh(bm)
    confirmed = []
    for edit in operations:
        if edit['kind'] != 'bushing':
            continue
        y,z = edit['center_yz']
        for offset in (0., .65*edit['inner'], -.65*edit['inner']):
            hit = tree.ray_cast(Vector((-.16,y+offset,z)),Vector((1.,0.,0.)),.32)
            if hit[0] is not None:
                raise ValueError('Edited bushing blocks an existing through-hole')
        confirmed.append(edit['part'])
    return confirmed


def gauge_face_rays(bm, mesh, operations):
    gauge = next((op for op in operations if op['kind']=='gauge'),None)
    if gauge is None:
        return 0
    bm.faces.index_update()
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)
    center, axis = Vector(gauge['center']), Vector(gauge['normal']).normalized()
    up = (Vector((0,0,1)) - axis*axis.z).normalized()
    right = up.cross(axis)
    offsets = [(0.,0.)]+[(.53*math.cos(i*math.tau/8),.53*math.sin(i*math.tau/8)) for i in range(8)]
    for u,v in offsets:
        pos = center + .14*axis + gauge['radius']*(right*u + up*v)
        hit, _, face, _ = tree.ray_cast(pos,-axis,.15)
        if hit is None or mesh.materials[bm.faces[face].material_index].name!='Gauge protective lens':
            raise ValueError('Original valve body overlaps the visible pressure gauge dial')
    return len(offsets)


def packed_images():
    return {image.name:sha256(bytes(image.packed_file.data)).hexdigest()
            for image in bpy.data.images if image.source=='FILE' and image.packed_file}


def dimensions(mesh):
    values = np.asarray([tuple(v.co) for v in mesh.vertices])
    return np.ptp(values, axis=0)


def refine(path, folder, case, operations, sheet):
    record = json.loads((path.parent/'candidate-check.json').read_text(encoding='utf8'))
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    if (record['status'] != 'pass' or record['source_mesh_hash'] != case['mesh_hash']
            or scene.get('source_mesh_hash') != case['mesh_hash']
            or scene.get('material_signature') != record['material_signature']):
        raise ValueError('Candidate does not match the frozen six-view source')
    subject = subject_mesh()
    original = subject.data
    texture_bytes = packed_images()
    before = {'vertices':len(original.vertices),'faces':len(original.polygons),'span':dimensions(original).tolist()}
    new = original.copy()
    bm = bmesh.new()
    bm.from_mesh(original)
    bm.faces.ensure_lookup_table()
    origin = bm.faces.layers.int.new('mesh_detail_origin')
    for i,face in enumerate(bm.faces,1):
        face[origin] = i
    changes, edited = [], set()
    for edit in operations:
        kind = edit['kind']
        if kind == 'thread':
            changes.append(threaded_stem(bm,original,case,edit))
        elif kind == 'bushing':
            changes.append(refine_bushing(bm,original,case,edit,origin,edited))
        elif kind == 'shoulder':
            changes.append(rounded_shoulder(bm,original,case,edit,origin,edited))
        elif kind == 'gauge':
            changes.append(add_pressure_gauge(bm,new,edit,sheet))
        else:
            raise ValueError(f'Unknown local mesh operation {kind}')
    bm.normal_update()
    holes = bushing_holes(bm,operations)
    gauge_rays = gauge_face_rays(bm,new,operations)
    bm.to_mesh(new)
    bm.free()
    new.update()
    retained = preserve_corners(original,new,'mesh_detail_origin',edited)
    after = {'vertices':len(new.vertices),'faces':len(new.polygons),'span':dimensions(new).tolist()}
    if after['faces']<=before['faces'] or max(abs(np.subtract(after['span'],before['span'])))>.0015:
        raise ValueError('Mesh refinement changed the overall proportions')
    subject.data = new
    output = folder/'candidate.blend'
    folder.mkdir(parents=True,exist_ok=False)
    scene['mesh_detail_candidate'] = True
    scene['acceptance'] = 'appearance_and_geometry_review_contact_unverified'
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(output))
    saved = subject_mesh().data
    reopened_textures = packed_images()
    if (len(saved.polygons)!=after['faces'] or len(saved.vertices)!=after['vertices'] or not saved.has_custom_normals
            or not all(image.packed_file for image in bpy.data.images if image.source=='FILE')):
        raise ValueError('Candidate reopened with different mesh or missing packed images')
    if any(reopened_textures.get(name)!=digest for name,digest in texture_bytes.items()):
        raise ValueError('An existing registered texture changed during local mesh editing')
    return {'before':before,'after':after,'unrelated_faces_positions_uv_preserved':retained,
            'verified_open_bushings':holes,'clear_dial_rays':gauge_rays,'original_images_verified':len(texture_bytes),
            'changes':changes,'reopened':True,'sha256':sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    base, output = args.baseline.resolve(), args.output.resolve()
    cases = {case['asset_id']:case for case in json.loads((base/'frozen.json').read_text(encoding='utf8'))['cases']}
    recipe = json.loads(args.recipe.read_text(encoding='utf8'))
    if recipe.get('version') != 'reference-mesh-detail-v1' or recipe['coordinate_frame'] != 'normalized-render-mesh':
        raise ValueError('Expected a detail recipe in the normalized render frame')
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    records = {}
    for aid,detail in recipe['assets'].items():
        records[aid] = refine(base/aid/'candidate.blend',output/aid,cases[aid],detail['operations'],base/aid/'reference-sheet.png')
    (output/'mesh-checks.json').write_text(json.dumps({'assets':records,'seconds':round(time.monotonic()-started,3)},
                                                       ensure_ascii=False,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
