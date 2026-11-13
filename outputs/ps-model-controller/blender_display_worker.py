"""Persistent Blender RPC for display batches. No library meshes are rewritten."""
import json
import math
from pathlib import Path
import sys
import traceback
import re
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
import blender_worker as legacy
from display_batch.geometry import camera_for, contact_geometry, horizontal_angle, pose_points, transformed, material_detail_allowed
from display_batch.common import atomic_json
from display_batch.materials import material_kind, SURFACE_DUST
from display_batch.scanned_shader import matte_dust_surface, overcast_pixels, scanned_surface

BASE = None
BASE_PATH = None
BASE_ROUGHNESS = {}
DEVICE = None
LIVE_RENDER = None


def reset(device='CPU', resolution=512):
    global DEVICE, LIVE_RENDER, BASE, BASE_PATH
    if LIVE_RENDER is not None:
        LIVE_RENDER=None
        if BASE is not None:BASE.use_fake_user=False
        BASE=BASE_PATH=None
    if DEVICE is None:
        scene=legacy.setup_scene(device,resolution)
        DEVICE=device
    else:
        scene=bpy.context.scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj,do_unlink=True)
        if scene.rigidbody_world:
            bpy.ops.rigidbody.world_remove()
        scene.frame_set(1)
        bpy.data.orphans_purge(do_recursive=True)
        scene.render.resolution_x=scene.render.resolution_y=resolution
    scene.render.film_transparent=True
    scene.render.use_border=False
    scene.render.use_persistent_data=False
    scene.view_settings.view_transform='AgX'
    scene.cycles.samples=32
    scene.cycles.use_denoising=True
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA'
    scene.render.image_settings.color_depth='8'
    scene.use_nodes=False
    return scene


def normalized_mesh(path):
    global BASE, BASE_PATH, BASE_ROUGHNESS
    if BASE is None or BASE_PATH!=path:
        if BASE is not None:
            BASE.use_fake_user=False
            # Remove unused textures/materials before loading another large asset.
            bpy.data.orphans_purge(do_recursive=True)
        obj=legacy.load_mesh(path)
        points=np.asarray([list(p) for p in legacy.world_vertices(obj)])
        low,high=points.min(axis=0),points.max(axis=0)
        span=float(max(high-low)); center=(low+high)/2
        mat=Matrix.Scale(1/span,4) @ Matrix.Translation(Vector(-center)) @ obj.matrix_world
        obj.data.transform(mat);obj.matrix_world=Matrix.Identity(4)
        BASE=obj.data;BASE.use_fake_user=True;BASE_PATH=path
        BASE_ROUGHNESS={}
        for material in BASE.materials:
            if material and material.use_nodes:
                for n in material.node_tree.nodes:
                    if n.type=='BSDF_PRINCIPLED':
                        BASE_ROUGHNESS[(material.name,n.name)]=float(n.inputs['Roughness'].default_value)
        return obj
    obj=bpy.data.objects.new('display asset',BASE)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def arrays(obj):
    vertices=np.array([list(v.co) for v in obj.data.vertices],dtype=np.float64)
    obj.data.calc_loop_triangles()
    faces=np.array([t.vertices for t in obj.data.loop_triangles],dtype=np.int32)
    return vertices,faces


def export(request):
    reset(request['device'])
    obj=normalized_mesh(request['mesh']); vertices,faces=arrays(obj)
    np.savez_compressed(request['geometry'],vertices=vertices,faces=faces)
    mats=[]
    for m in obj.data.materials:
        if not m:continue
        nodes=m.node_tree.nodes if m.use_nodes else []
        p=next((n for n in nodes if n.type=='BSDF_PRINCIPLED'),None)
        mats.append({'name':m.name,'metallic':float(p.inputs['Metallic'].default_value) if p else None,
                     'roughness':float(p.inputs['Roughness'].default_value) if p else None,
                     'metallic_linked':bool(p and p.inputs['Metallic'].is_linked),
                     'roughness_linked':bool(p and p.inputs['Roughness'].is_linked),
                     'normal_linked':bool(p and p.inputs['Normal'].is_linked),
                     'textures':[{'name':n.image.name,'space':n.image.colorspace_settings.name}
                                 for n in nodes if n.type=='TEX_IMAGE' and n.image]})
    return {'materials':mats,'vertices':len(vertices),'faces':len(faces),
            'extents':np.ptp(vertices,axis=0).tolist(),'topology_changed':False,
            'geometry':request['geometry']}


def rigid(obj, shape, passive=False):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True);bpy.context.view_layer.objects.active=obj
    bpy.ops.rigidbody.object_add()
    body=obj.rigid_body;body.type='PASSIVE' if passive else 'ACTIVE'
    body.collision_shape=shape;body.friction=.75;body.restitution=0
    body.use_margin=True;body.collision_margin=.001
    body.linear_damping=.4;body.angular_damping=.6
    return body


def make_receiver(request,normal,physical=False):
    patches=request.get('receiver',{}).get('patches',[])
    if patches:
        vertices=[];faces=[]
        for patch in patches:
            offset=len(vertices);vertices.extend(patch['vertices'])
            for face in patch['faces']:
                face=list(face);v=np.asarray(patch['vertices'])[face[:3]]
                if np.cross(v[1]-v[0],v[2]-v[0])[2]<0:face.reverse()
                faces.append([offset+i for i in face])
        mesh=bpy.data.meshes.new('separate support faces');mesh.from_pydata(vertices,[],faces);mesh.update()
        ground=bpy.data.objects.new('finite support patches',mesh);bpy.context.scene.collection.objects.link(ground)
        if physical:rigid(ground,'MESH',True)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1,location=Vector(normal)*-.1)
        ground=bpy.context.object;ground.scale=(8,8,.2)
        ground.rotation_euler=Vector((0,0,1)).rotation_difference(Vector(normal)).to_euler()
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        if physical:rigid(ground,'BOX',True)
    return ground


def settle(request):
    if 'length_m' in request:
        from display_batch.metric_physics import run
        return run(request,reset,rigid,make_receiver)
    data=np.load(request['geometry']);vertices=data['vertices'];faces=data['faces']
    proxy=json.loads(Path(request['proxy']).read_text(encoding='utf8'))
    com=np.asarray(proxy['centroid'])
    normal=np.asarray(request.get('plane_normal',[0,0,1]),dtype=float);normal/=np.linalg.norm(normal)
    results=[]
    for i, initial in enumerate(request['initial_matrices'][:6]):
        scene=reset(request['device'])
        # A compound parent follows the volume centroid; convex children retain local positions.
        mesh=bpy.data.meshes.new('compound_root')
        mesh.from_pydata((vertices-com).tolist(),[],faces.tolist());mesh.update()
        root=bpy.data.objects.new('compound assembly',mesh);scene.collection.objects.link(root)
        body=rigid(root,'COMPOUND');body.mass=max(.01,float(proxy['volume']))
        root.matrix_world=Matrix(initial) @ Matrix.Translation(Vector(com))
        p=transformed(vertices,initial)
        root.location.z+=(.025-float((p@normal).min()))/normal[2]
        for part in proxy['parts']:
            child_mesh=bpy.data.meshes.new('collision_part')
            child_mesh.from_pydata((np.asarray(part['vertices'])-com).tolist(),[],part['faces'])
            child_mesh.update()
            child=bpy.data.objects.new('convex part',child_mesh);scene.collection.objects.link(child)
            child.parent=root;child.matrix_parent_inverse=Matrix.Identity(4)
            rigid(child,'CONVEX_HULL').mass=max(.00001,float(part['volume']))
        floor=make_receiver(request,normal,physical=True)
        scene.gravity=(0,0,-9.81)
        world=scene.rigidbody_world
        world.substeps_per_frame=10;world.solver_iterations=40
        world.point_cache.frame_start=1;world.point_cache.frame_end=400
        last=None;quiet=0;move=999;angle=999
        for frame in range(1,401):
            scene.frame_set(frame)
            if frame%20:continue
            m=root.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
            if last is not None:
                move=(m.translation-last.translation).length
                angle=math.degrees(m.to_quaternion().rotation_difference(last.to_quaternion()).angle)%360
                angle=min(angle,360-angle)
                quiet=quiet+1 if frame>=80 and move<=.002 and angle<=.5 else 0
                if quiet>=3:break
            last=m
        m=root.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy() @ Matrix.Translation(Vector(-com))
        matrix=np.asarray(m).copy(); p=transformed(vertices,matrix)
        minimum=float((p@normal).min()); height=float(np.ptp(p[:,2]))
        tilt=horizontal_angle(request['axis'],matrix)
        contact=p[(p@normal)<=.008]
        if request.get('receiver',{}).get('patches') and len(contact):
            from mathutils.bvhtree import BVHTree
            tree=BVHTree.FromPolygons([v.co for v in floor.data.vertices],
                                     [list(face.vertices) for face in floor.data.polygons],all_triangles=True)
            actual=[]
            for point in contact:
                found=tree.find_nearest(Vector(point))
                if found[0] is not None and found[3]<=.010:actual.append(point)
            contact=np.asarray(actual).reshape(-1,3)
        center=transformed(com[None],matrix)[0]
        supported=False
        if len(contact)>=3:
            # Convex support polygon is valid for the COM only, never for footprint coverage.
            from mathutils.geometry import convex_hull_2d
            vec=[Vector((float(x),float(y))) for x,y in contact[:,:2]]
            hull=np.array([vec[j] for j in convex_hull_2d(vec)])
            if len(hull)>=3:
                edges=np.roll(hull,-1,axis=0)-hull
                side=edges[:,0]*(center[1]-hull[:,1])-edges[:,1]*(center[0]-hull[:,0])
                supported=bool(np.all(side>=-.002) or np.all(side<=.002))
        failures=[]
        if quiet<3:failures.append('not_converged')
        if minimum<-.004:failures.append('ground_penetration')
        if minimum>.008:failures.append('floating')
        if not supported:failures.append('center_of_mass_unsupported')
        if tilt>15:failures.append('upright')
        # Do not teleport a body to conceal proxy error. Only remove sub-millimeter margin.
        if not failures:matrix[:3,3]-=normal*minimum
        results.append({'id':f'P{i}','matrix':matrix.tolist(),'stable':not failures,
                        'failure_codes':failures,'axis_angle_deg':tilt,'height':height,
                        'min_z':minimum,'center_supported':supported,'frames':frame,
                        'motion':move,'rotation_motion_deg':angle,'collision_parts':len(proxy['parts'])})
    return {'poses':results,'mass_basis':proxy['mass_basis'],'proxy_basis':'compound_coacd',
            'assembly':request['assembly'],'plane_normal':normal.tolist()}


def illuminate(scene, camera, environment, scale=1.):
    nodes=scene.world.node_tree.nodes;nodes.clear();links=scene.world.node_tree.links
    output=nodes.new('ShaderNodeOutputWorld');bg=nodes.new('ShaderNodeBackground')
    color=np.array(environment.get('world_color',[.28,.3,.32]))
    width,height=256,128
    yy,xx=np.mgrid[0:height,0:width]
    sky=.12+.55*(yy/height)**2
    reflection=np.exp(-((xx/width-.24)/.035)**2)*np.exp(-((yy/height-.60)/.23)**2)
    reflection+=.5*np.exp(-((xx/width-.76)/.14)**2)*np.exp(-((yy/height-.52)/.22)**2)
    pixels=np.ones((height,width,4),np.float32);pixels[:,:,:3]=(sky+6.*reflection)[:,:,None]*color
    image=bpy.data.images.new('estimated reflection environment',width=width,height=height,float_buffer=True)
    image.colorspace_settings.name='Non-Color';image.pixels.foreach_set(pixels.ravel())
    env=nodes.new('ShaderNodeTexEnvironment');env.image=image
    links.new(env.outputs['Color'],bg.inputs['Color']);bg.inputs['Strength'].default_value=.65
    links.new(bg.outputs[0],output.inputs['Surface'])
    # Broad, non-uniform reflection sources retain metal shape without copying scene texture.
    for name,offset,power,size in [('key',(-2,-3,4),160*scale,3.2),('fill',(3,1,2),65,4.5)]:
        direction=environment.get('key_direction')
        if name=='key' and direction:offset=(direction[0]*3,-2,3+max(0,-direction[1]))
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
        data.color=environment.get('light_color',[1,1,1])
        obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=offset
        obj.rotation_euler=(-obj.location).to_track_quat('-Z','Y').to_euler()


def surface_detail(material,node,environment,dust_gain):
    """Bounded shader-only wear shared by material classes, preserving UV maps and geometry."""
    if not material_detail_allowed(material.name,node.inputs['Metallic'].is_linked):return
    tree=material.node_tree
    local=environment.get('local_context')
    detail=local.get('surface_texture_rms',.018) if local else .025
    deposit=(.12+min(.08,detail*1.5)) if local else (.12+environment.get('haze',0)*.55)
    base_bump=.08 if re.search(r'steel|chrome|metal|atlas',material.name,re.I) else .16
    bump_strength=base_bump*float(np.clip(detail/.025,.8,1.)) if local else base_bump
    if tree.nodes.get('batch_micrograin'):
        tree.nodes['batch_dust_strength'].inputs[1].default_value=min(.38,deposit*dust_gain)
        tree.nodes['batch_dust_tint'].inputs[2].default_value=(*environment.get('dust_color',[.16,.145,.12]),1)
        if tree.nodes.get('batch_micro_bump'):
            tree.nodes['batch_micro_bump'].inputs['Strength'].default_value=bump_strength
        return
    noise=tree.nodes.new('ShaderNodeTexNoise');noise.name='batch_micrograin'
    noise.inputs['Scale'].default_value=160;noise.inputs['Detail'].default_value=3
    bump=tree.nodes.new('ShaderNodeBump');bump.name='batch_micro_bump';bump.inputs['Strength'].default_value=bump_strength
    bump.inputs['Distance'].default_value=.002
    tree.links.new(noise.outputs['Fac'],bump.inputs['Height'])
    if node.inputs['Normal'].is_linked:tree.links.new(node.inputs['Normal'].links[0].from_socket,bump.inputs['Normal'])
    tree.links.new(bump.outputs['Normal'],node.inputs['Normal'])
    broad=tree.nodes.new('ShaderNodeTexNoise');broad.inputs['Scale'].default_value=11
    broad.inputs['Detail'].default_value=4
    remap=tree.nodes.new('ShaderNodeMapRange')
    remap.inputs['From Min'].default_value=.32;remap.inputs['From Max'].default_value=.68;remap.clamp=True
    remap.inputs['To Min'].default_value=.42;remap.inputs['To Max'].default_value=1.
    tree.links.new(broad.outputs['Fac'],remap.inputs['Value'])
    multiply=tree.nodes.new('ShaderNodeMixRGB');multiply.blend_type='MULTIPLY';multiply.inputs[0].default_value=1.
    if node.inputs['Base Color'].is_linked:
        tree.links.new(node.inputs['Base Color'].links[0].from_socket,multiply.inputs[1])
    else:multiply.inputs[1].default_value=node.inputs['Base Color'].default_value
    material_mask=tree.nodes.new('ShaderNodeMixRGB');material_mask.inputs[2].default_value=(1,1,1,1)
    if node.inputs['Metallic'].is_linked:
        tree.links.new(node.inputs['Metallic'].links[0].from_socket,material_mask.inputs[0])
    else:material_mask.inputs[0].default_value=node.inputs['Metallic'].default_value
    tree.links.new(remap.outputs[0],material_mask.inputs[1])
    tree.links.new(material_mask.outputs[0],multiply.inputs[2])
    variation=tree.nodes.new('ShaderNodeMapRange')
    variation.inputs['From Min'].default_value=.3;variation.inputs['From Max'].default_value=.7
    variation.inputs['To Min'].default_value=-.12;variation.inputs['To Max'].default_value=.10;variation.clamp=True
    tree.links.new(broad.outputs['Fac'],variation.inputs['Value'])
    rough=tree.nodes.new('ShaderNodeMath');rough.name='batch_roughness_variation';rough.operation='ADD';rough.use_clamp=True
    if node.inputs['Roughness'].is_linked:tree.links.new(node.inputs['Roughness'].links[0].from_socket,rough.inputs[0])
    else:rough.inputs[0].default_value=node.inputs['Roughness'].default_value
    tree.links.new(variation.outputs[0],rough.inputs[1]);tree.links.new(rough.outputs[0],node.inputs['Roughness'])
    geometry=tree.nodes.new('ShaderNodeNewGeometry')
    upward=tree.nodes.new('ShaderNodeVectorMath');upward.operation='DOT_PRODUCT';upward.inputs[1].default_value=(0,0,1)
    tree.links.new(geometry.outputs['Normal'],upward.inputs[0])
    strength=tree.nodes.new('ShaderNodeMath');strength.name='batch_dust_strength';strength.operation='MULTIPLY';strength.use_clamp=True
    strength.inputs[1].default_value=min(.38,deposit*dust_gain)
    metal_mask=tree.nodes.new('ShaderNodeMath');metal_mask.operation='MULTIPLY_ADD'
    metal_mask.inputs[1].default_value=-.75;metal_mask.inputs[2].default_value=1.
    if node.inputs['Metallic'].is_linked:tree.links.new(node.inputs['Metallic'].links[0].from_socket,metal_mask.inputs[0])
    else:metal_mask.inputs[0].default_value=node.inputs['Metallic'].default_value
    exposed=tree.nodes.new('ShaderNodeMath');exposed.operation='MULTIPLY';exposed.use_clamp=True
    tree.links.new(upward.outputs['Value'],exposed.inputs[0]);tree.links.new(metal_mask.outputs[0],exposed.inputs[1])
    tree.links.new(exposed.outputs[0],strength.inputs[0])
    dust=tree.nodes.new('ShaderNodeMixRGB');dust.name='batch_dust_tint'
    dust.inputs[2].default_value=(*environment.get('dust_color',[.16,.145,.12]),1)
    tree.links.new(strength.outputs[0],dust.inputs[0]);tree.links.new(multiply.outputs[0],dust.inputs[1])
    tree.links.new(dust.outputs[0],node.inputs['Base Color'])


def rust_spot_detail(material,node,strength,material_classes=(),environment=None,dust_scale=0.,rust_scan=None):
    """Add sparse oxide freckles to metal and chromatic paint, excluding rubber and labels."""
    strength=float(np.clip(strength,0.,1.))
    if not strength or not material_detail_allowed(material.name,node.inputs['Metallic'].is_linked):return
    classes=set(material_classes)
    if classes and not classes.intersection({'steel','chrome','paint','rust'}):return
    tree=material.node_tree;nodes=tree.nodes;links=tree.links
    if nodes.get('batch_rust_spot_color'):return
    metallic=node.inputs['Metallic']
    metallic_source=metallic.links[0].from_socket if metallic.is_linked else None
    metallic_value=float(metallic.default_value)
    # Painted metal commonly has a constant Metallic=0 shader input. Do not
    # reject it here: the chroma mask below is what limits oxide to painted steel.

    coordinates=nodes.new('ShaderNodeTexCoord');coordinates.name='batch_rust_coordinates'
    cells=nodes.new('ShaderNodeTexVoronoi');cells.name='batch_rust_cells';cells.feature='F1'
    # Keep individual flecks visible after the object is reduced to camera size.
    cells.distance='EUCLIDEAN';cells.inputs['Scale'].default_value=22
    breakup=nodes.new('ShaderNodeTexNoise');breakup.name='batch_rust_edge_breakup'
    breakup.inputs['Scale'].default_value=190;breakup.inputs['Detail'].default_value=2
    breakup.inputs['Roughness'].default_value=.78
    links.new(coordinates.outputs['Generated'],cells.inputs['Vector'])
    links.new(coordinates.outputs['Generated'],breakup.inputs['Vector'])
    radius=nodes.new('ShaderNodeMath');radius.name='batch_rust_radius';radius.operation='MULTIPLY_ADD'
    radius.inputs[1].default_value=.095;radius.inputs[2].default_value=.11
    links.new(breakup.outputs['Fac'],radius.inputs[0])
    inside=nodes.new('ShaderNodeMath');inside.name='batch_rust_distance';inside.operation='SUBTRACT'
    links.new(radius.outputs[0],inside.inputs[0]);links.new(cells.outputs['Distance'],inside.inputs[1])
    edge=nodes.new('ShaderNodeMapRange');edge.name='batch_rust_soft_edge';edge.clamp=True
    edge.inputs['From Min'].default_value=.008;edge.inputs['From Max'].default_value=.044
    edge.inputs['To Min'].default_value=0.;edge.inputs['To Max'].default_value=1.
    links.new(inside.outputs[0],edge.inputs['Value'])
    density=nodes.new('ShaderNodeTexNoise');density.name='batch_rust_density';density.inputs['Scale'].default_value=18
    density.inputs['Detail'].default_value=2;density.inputs['Roughness'].default_value=.72
    links.new(coordinates.outputs['Generated'],density.inputs['Vector'])
    density_map=nodes.new('ShaderNodeMapRange');density_map.name='batch_rust_cluster_density';density_map.clamp=True
    density_map.inputs['From Min'].default_value=.34;density_map.inputs['From Max'].default_value=.7
    density_map.inputs['To Min'].default_value=.08;density_map.inputs['To Max'].default_value=1.
    links.new(density.outputs['Fac'],density_map.inputs['Value'])
    clustered_spots=nodes.new('ShaderNodeMath');clustered_spots.name='batch_rust_clustered_spots';clustered_spots.operation='MULTIPLY';clustered_spots.use_clamp=True
    links.new(edge.outputs[0],clustered_spots.inputs[0]);links.new(density_map.outputs[0],clustered_spots.inputs[1])
    scars=nodes.new('ShaderNodeTexNoise');scars.name='batch_rust_irregular_scars'
    scars.inputs['Scale'].default_value=9;scars.inputs['Detail'].default_value=5;scars.inputs['Roughness'].default_value=.8
    links.new(coordinates.outputs['Generated'],scars.inputs['Vector'])
    scars_map=nodes.new('ShaderNodeMapRange');scars_map.name='batch_rust_scar_threshold';scars_map.clamp=True
    scars_map.inputs['From Min'].default_value=.47;scars_map.inputs['From Max'].default_value=.61
    scars_map.inputs['To Min'].default_value=0.;scars_map.inputs['To Max'].default_value=1.
    links.new(scars.outputs['Fac'],scars_map.inputs['Value'])
    rust_shape=nodes.new('ShaderNodeMath');rust_shape.name='batch_rust_spots_and_scars';rust_shape.operation='MAXIMUM'
    links.new(clustered_spots.outputs[0],rust_shape.inputs[0]);links.new(scars_map.outputs[0],rust_shape.inputs[1])
    shape_output=rust_shape.outputs[0];scan_image=None;scan_roughness=None;scan_height=None;scan_detail=None
    relief_strength=float(np.clip((rust_scan or {}).get('relief_strength',1.),1.,3.))
    if rust_scan:
        mapping=nodes.new('ShaderNodeVectorMath');mapping.operation='SCALE';mapping.inputs['Scale'].default_value=1.4
        links.new(coordinates.outputs['Generated'],mapping.inputs[0])
        scan_image=nodes.new('ShaderNodeTexImage');scan_image.name='batch_photo_rust'
        scan_image.image=bpy.data.images.load(rust_scan['path'],check_existing=True)
        scan_image.projection='BOX';scan_image.projection_blend=.25
        links.new(mapping.outputs['Vector'],scan_image.inputs['Vector'])
        value=nodes.new('ShaderNodeRGBToBW');links.new(scan_image.outputs['Color'],value.inputs['Color'])
        scan_mask=nodes.new('ShaderNodeMapRange');scan_mask.name='batch_photo_rust_mask';scan_mask.clamp=True
        scan_mask.inputs['From Min'].default_value=.035;scan_mask.inputs['From Max'].default_value=.28
        scan_mask.inputs['To Min'].default_value=1.;scan_mask.inputs['To Max'].default_value=0.
        links.new(value.outputs['Val'],scan_mask.inputs['Value']);shape_output=scan_mask.outputs['Result']
        if rust_scan.get('roughness_path'):
            scan_roughness=nodes.new('ShaderNodeTexImage');scan_roughness.name='batch_photo_rust_roughness'
            scan_roughness.image=bpy.data.images.load(rust_scan['roughness_path'],check_existing=True)
            scan_roughness.image.colorspace_settings.name='Non-Color'
            scan_roughness.projection='BOX';scan_roughness.projection_blend=.25
            links.new(mapping.outputs['Vector'],scan_roughness.inputs['Vector'])
            scan_depth=nodes.new('ShaderNodeTexImage');scan_depth.name='batch_photo_rust_height'
            scan_depth.image=bpy.data.images.load(rust_scan['height_path'],check_existing=True)
            scan_depth.image.colorspace_settings.name='Non-Color'
            scan_depth.projection='BOX';scan_depth.projection_blend=.25
            links.new(mapping.outputs['Vector'],scan_depth.inputs['Vector'])
            scan_height=nodes.new('ShaderNodeMath');scan_height.name='batch_photo_rust_relief';scan_height.operation='MULTIPLY'
            links.new(scan_depth.outputs['Color'],scan_height.inputs[0]);links.new(scan_mask.outputs['Result'],scan_height.inputs[1])
            if rust_scan.get('detail_path'):
                scan_detail=nodes.new('ShaderNodeTexImage');scan_detail.name='batch_photo_rust_detail'
                scan_detail.image=bpy.data.images.load(rust_scan['detail_path'],check_existing=True)
                scan_detail.image.colorspace_settings.name='Non-Color'
                scan_detail.projection='BOX';scan_detail.projection_blend=.25
                links.new(mapping.outputs['Vector'],scan_detail.inputs['Vector'])
    metal_mask=nodes.new('ShaderNodeMath');metal_mask.name='batch_rust_metal_mask';metal_mask.operation='MULTIPLY';metal_mask.use_clamp=True
    if metallic_source is not None:links.new(metallic_source,metal_mask.inputs[0])
    else:metal_mask.inputs[0].default_value=metallic_value
    links.new(shape_output,metal_mask.inputs[1])

    # Painted steel has metallic=0 by design. Use actual albedo chroma to include
    # paint while leaving achromatic rubber, chrome labels and pale markings alone.
    base=node.inputs['Base Color'];base_source=base.links[0].from_socket if base.is_linked else None
    base_value=tuple(base.default_value)
    if base_source is not None:
        separate=nodes.new('ShaderNodeSeparateColor');separate.name='batch_rust_albedo_channels';separate.mode='RGB'
        links.new(base_source,separate.inputs['Color'])
        max_rg=nodes.new('ShaderNodeMath');max_rg.operation='MAXIMUM'
        links.new(separate.outputs['Red'],max_rg.inputs[0]);links.new(separate.outputs['Green'],max_rg.inputs[1])
        max_rgb=nodes.new('ShaderNodeMath');max_rgb.operation='MAXIMUM'
        links.new(max_rg.outputs[0],max_rgb.inputs[0]);links.new(separate.outputs['Blue'],max_rgb.inputs[1])
        min_rg=nodes.new('ShaderNodeMath');min_rg.operation='MINIMUM'
        links.new(separate.outputs['Red'],min_rg.inputs[0]);links.new(separate.outputs['Green'],min_rg.inputs[1])
        min_rgb=nodes.new('ShaderNodeMath');min_rgb.operation='MINIMUM'
        links.new(min_rg.outputs[0],min_rgb.inputs[0]);links.new(separate.outputs['Blue'],min_rgb.inputs[1])
        chroma=nodes.new('ShaderNodeMath');chroma.operation='SUBTRACT'
        links.new(max_rgb.outputs[0],chroma.inputs[0]);links.new(min_rgb.outputs[0],chroma.inputs[1])
    else:
        chroma=nodes.new('ShaderNodeMath');chroma.operation='SUBTRACT'
        chroma.inputs[0].default_value=max(base_value[:3]);chroma.inputs[1].default_value=min(base_value[:3])
    paint_ramp=nodes.new('ShaderNodeMapRange');paint_ramp.name='batch_rust_painted_metal_mask';paint_ramp.clamp=True
    paint_ramp.inputs['From Min'].default_value=.10;paint_ramp.inputs['From Max'].default_value=.28
    paint_ramp.inputs['To Min'].default_value=0.;paint_ramp.inputs['To Max'].default_value=1.
    links.new(chroma.outputs[0],paint_ramp.inputs['Value'])
    painted_spots=nodes.new('ShaderNodeMath');painted_spots.name='batch_rust_spots_on_paint';painted_spots.operation='MULTIPLY';painted_spots.use_clamp=True
    links.new(shape_output,painted_spots.inputs[0]);links.new(paint_ramp.outputs[0],painted_spots.inputs[1])
    include_surface=nodes.new('ShaderNodeMath');include_surface.name='batch_rust_metal_or_paint';include_surface.operation='MAXIMUM';include_surface.use_clamp=True
    links.new(metal_mask.outputs[0],include_surface.inputs[0]);links.new(painted_spots.outputs[0],include_surface.inputs[1])
    strength_node=nodes.new('ShaderNodeMath');strength_node.name='batch_rust_strength';strength_node.operation='MULTIPLY';strength_node.use_clamp=True
    strength_node.inputs[1].default_value=strength;links.new(include_surface.outputs[0],strength_node.inputs[0])

    palette=nodes.new('ShaderNodeTexNoise');palette.name='batch_rust_palette'
    palette.inputs['Scale'].default_value=37;palette.inputs['Detail'].default_value=2
    links.new(coordinates.outputs['Generated'],palette.inputs['Vector'])
    tones=nodes.new('ShaderNodeMixRGB');tones.name='batch_rust_tones';tones.blend_type='MIX'
    tones.inputs[1].default_value=(.07,.022,.007,1.)
    tones.inputs[2].default_value=(.19,.058,.012,1.)
    links.new(palette.outputs['Fac'],tones.inputs[0])
    rust_color=tones.outputs['Color']
    if scan_image:
        local_tint=nodes.new('ShaderNodeMixRGB');local_tint.name='batch_photo_rust_local_color';local_tint.blend_type='MIX'
        local_tint.inputs[0].default_value=.45
        links.new(scan_image.outputs['Color'],local_tint.inputs[1])
        local_tint.inputs[2].default_value=(*((environment or {}).get('dust_color',[.16,.145,.12])),1.)
        rust_color=local_tint.outputs['Color']

    color=nodes.new('ShaderNodeMixRGB');color.name='batch_rust_spot_color';color.blend_type='MIX'
    if base_source is not None:links.new(base_source,color.inputs[1])
    else:color.inputs[1].default_value=base_value
    links.new(rust_color,color.inputs[2]);links.new(strength_node.outputs[0],color.inputs[0])
    # Uneven deposit is a separate layer. Its patchiness and upward-facing bias
    # preserve clean edges and darker recessed areas rather than tinting all RGB.
    eligible=nodes.new('ShaderNodeMath');eligible.name='batch_deposit_eligible';eligible.operation='MAXIMUM';eligible.use_clamp=True
    if metallic_source is not None:links.new(metallic_source,eligible.inputs[0])
    else:eligible.inputs[0].default_value=metallic_value
    links.new(paint_ramp.outputs[0],eligible.inputs[1])
    surface_color=color.outputs['Color']
    if scan_detail:
        detail_gain=nodes.new('ShaderNodeMath');detail_gain.name='batch_photo_detail_gain';detail_gain.operation='MULTIPLY_ADD'
        detail_gain.inputs[1].default_value=1.4*(relief_strength-1.)
        detail_gain.inputs[2].default_value=1.-.7*(relief_strength-1.)
        links.new(scan_detail.outputs['Color'],detail_gain.inputs[0])
        bounded=nodes.new('ShaderNodeClamp');bounded.name='batch_photo_detail_bounded'
        bounded.inputs['Min'].default_value=.45;bounded.inputs['Max'].default_value=1.55
        links.new(detail_gain.outputs[0],bounded.inputs['Value'])
        detail_mix=nodes.new('ShaderNodeMixRGB');detail_mix.name='batch_photo_detail_color';detail_mix.blend_type='MULTIPLY'
        links.new(eligible.outputs[0],detail_mix.inputs[0]);links.new(surface_color,detail_mix.inputs[1]);links.new(bounded.outputs[0],detail_mix.inputs[2])
        surface_color=detail_mix.outputs['Color']
    dust_noise=nodes.new('ShaderNodeTexNoise');dust_noise.name='batch_deposit_patches'
    dust_noise.inputs['Scale'].default_value=7;dust_noise.inputs['Detail'].default_value=5;dust_noise.inputs['Roughness'].default_value=.78
    links.new(coordinates.outputs['Generated'],dust_noise.inputs['Vector'])
    dust_map=nodes.new('ShaderNodeMapRange');dust_map.name='batch_deposit_threshold';dust_map.clamp=True
    dust_map.inputs['From Min'].default_value=.43;dust_map.inputs['From Max'].default_value=.68
    dust_map.inputs['To Min'].default_value=0.;dust_map.inputs['To Max'].default_value=1.
    links.new(dust_noise.outputs['Fac'],dust_map.inputs['Value'])
    geometry=nodes.new('ShaderNodeNewGeometry');orientation=nodes.new('ShaderNodeVectorMath');orientation.operation='DOT_PRODUCT'
    orientation.inputs[1].default_value=(0,0,1);links.new(geometry.outputs['Normal'],orientation.inputs[0])
    upward=nodes.new('ShaderNodeMapRange');upward.name='batch_deposit_upward';upward.clamp=True
    upward.inputs['From Min'].default_value=-.2;upward.inputs['From Max'].default_value=.9
    upward.inputs['To Min'].default_value=.18;upward.inputs['To Max'].default_value=1.
    links.new(orientation.outputs['Value'],upward.inputs['Value'])
    dust_mask=nodes.new('ShaderNodeMath');dust_mask.operation='MULTIPLY';links.new(dust_map.outputs[0],dust_mask.inputs[0]);links.new(upward.outputs[0],dust_mask.inputs[1])
    dust_surface=nodes.new('ShaderNodeMath');dust_surface.operation='MULTIPLY';links.new(dust_mask.outputs[0],dust_surface.inputs[0]);links.new(eligible.outputs[0],dust_surface.inputs[1])
    dust_strength=nodes.new('ShaderNodeMath');dust_strength.name='batch_deposit_strength';dust_strength.operation='MULTIPLY';dust_strength.use_clamp=True
    dust_strength.inputs[1].default_value=.96*float(np.clip(dust_scale,0.,.5))*strength;links.new(dust_surface.outputs[0],dust_strength.inputs[0])
    dust_color=nodes.new('ShaderNodeMixRGB');dust_color.name='batch_irregular_dust_color'
    dust_color.inputs[2].default_value=(*((environment or {}).get('dust_color',[.16,.145,.12])),1.)
    links.new(surface_color,dust_color.inputs[1]);links.new(dust_strength.outputs[0],dust_color.inputs[0])
    for old in list(base.links):links.remove(old)
    links.new(dust_color.outputs['Color'],base)

    def mix_scalar(socket_name,target,label,factor=None):
        factor=factor or strength_node.outputs[0]
        socket=node.inputs[socket_name]
        source=socket.links[0].from_socket if socket.is_linked else None
        value=float(socket.default_value)
        inv=nodes.new('ShaderNodeMath');inv.name='batch_rust_'+label+'_inverse';inv.operation='SUBTRACT'
        inv.inputs[0].default_value=1.;links.new(factor,inv.inputs[1])
        base_part=nodes.new('ShaderNodeMath');base_part.name='batch_rust_'+label+'_base';base_part.operation='MULTIPLY'
        if source is not None:links.new(source,base_part.inputs[0])
        else:base_part.inputs[0].default_value=value
        links.new(inv.outputs[0],base_part.inputs[1])
        spot_part=nodes.new('ShaderNodeMath');spot_part.name='batch_rust_'+label+'_spot';spot_part.operation='MULTIPLY'
        if isinstance(target,(float,int)):spot_part.inputs[1].default_value=target
        else:links.new(target,spot_part.inputs[1])
        links.new(factor,spot_part.inputs[0])
        combined=nodes.new('ShaderNodeMath');combined.name='batch_rust_'+label+'_mix';combined.operation='ADD'
        links.new(base_part.outputs[0],combined.inputs[0]);links.new(spot_part.outputs[0],combined.inputs[1])
        for old in list(socket.links):links.remove(old)
        links.new(combined.outputs[0],socket)

    mix_scalar('Metallic',.06,'metallic')
    mix_scalar('Roughness',scan_roughness.outputs['Color'] if scan_roughness else .86,'roughness')
    mix_scalar('Roughness',.94,'dust_roughness',dust_strength.outputs[0])
    normal=node.inputs['Normal'];normal_source=normal.links[0].from_socket if normal.is_linked else None
    bump=nodes.new('ShaderNodeBump');bump.name='batch_rust_bump'
    bump.inputs['Strength'].default_value=.18*strength;bump.inputs['Distance'].default_value=.0007
    links.new(strength_node.outputs[0],bump.inputs['Height'])
    if normal_source is not None:links.new(normal_source,bump.inputs['Normal'])
    final_normal=bump.outputs['Normal']
    if scan_height:
        fine_bump=nodes.new('ShaderNodeBump');fine_bump.name='batch_photo_rust_microrelief'
        fine_bump.inputs['Strength'].default_value=.28*strength*min(relief_strength,2.)
        fine_bump.inputs['Distance'].default_value=.0008*relief_strength
        links.new(scan_height.outputs[0],fine_bump.inputs['Height']);links.new(final_normal,fine_bump.inputs['Normal'])
        final_normal=fine_bump.outputs['Normal']
    for old in list(normal.links):links.remove(old)
    links.new(final_normal,normal)


def fine_render_mesh(obj,request):
    if request.get('render_detail')!='fine':return
    if obj.modifiers.get('batch_fine_tessellation'):return
    modifier=obj.modifiers.new('batch_fine_tessellation','SUBSURF')
    modifier.subdivision_type='SIMPLE'
    modifier.levels=1;modifier.render_levels=1
    modifier.show_render=True


def pass_outputs(scene, folder):
    layer=scene.view_layers[0]
    layer.use_pass_diffuse_direct=True;layer.use_pass_diffuse_indirect=True;layer.use_pass_diffuse_color=True
    layer.use_pass_glossy_direct=True;layer.use_pass_glossy_indirect=True;layer.use_pass_glossy_color=True
    scene.use_nodes=True;tree=scene.node_tree;tree.nodes.clear()
    source=tree.nodes.new('CompositorNodeRLayers')
    composite=tree.nodes.new('CompositorNodeComposite');tree.links.new(source.outputs['Image'],composite.inputs[0])
    layer.use_pass_z=True
    for label,socket in [('combined','Image'),('depth','Depth'),('metallic','metallic'),('upward','upward')]:
        if label in ('metallic','upward') and not layer.aovs.get(label):
            aov=layer.aovs.add();aov.name=label;aov.type='VALUE'
    bpy.context.view_layer.update()
    for label,socket in [('combined','Image'),('depth','Depth'),('metallic','metallic'),('upward','upward')]:
        node=tree.nodes.new('CompositorNodeOutputFile');node.base_path=str(folder)
        node.format.file_format='OPEN_EXR';node.format.color_depth='32' if label=='depth' else '16'
        node.file_slots[0].path=label;tree.links.new(source.outputs[socket],node.inputs[0])
    for label,prefix in [('diffuse','Diff'),('specular','Gloss')]:
        add=tree.nodes.new('CompositorNodeMixRGB');add.blend_type='ADD';add.inputs[0].default_value=1
        tree.links.new(source.outputs[prefix+'Dir'],add.inputs[1]);tree.links.new(source.outputs[prefix+'Ind'],add.inputs[2])
        mul=tree.nodes.new('CompositorNodeMixRGB');mul.blend_type='MULTIPLY';mul.inputs[0].default_value=1
        tree.links.new(add.outputs[0],mul.inputs[1]);tree.links.new(source.outputs[prefix+'Col'],mul.inputs[2])
        node=tree.nodes.new('CompositorNodeOutputFile');node.base_path=str(folder)
        node.format.file_format='OPEN_EXR';node.format.color_depth='16';node.file_slots[0].path=label
        tree.links.new(mul.outputs[0],node.inputs[0])


def shadow_output(scene,folder,name):
    scene.use_nodes=True;tree=scene.node_tree;tree.nodes.clear()
    source=tree.nodes.new('CompositorNodeRLayers');composite=tree.nodes.new('CompositorNodeComposite')
    tree.links.new(source.outputs['Image'],composite.inputs[0])
    output=tree.nodes.new('CompositorNodeOutputFile');output.base_path=str(folder)
    output.format.file_format='OPEN_EXR';output.format.color_depth='16';output.file_slots[0].path=name+'-linear'
    tree.links.new(source.outputs['Image'],output.inputs[0])
    legacy.render(scene,folder/(name+'.png'))


def configure_materials(obj,request,reference=True):
    scans=request.get('surface_scans')
    matte=request.get('matte_partitions')
    adapted=set()
    for material in obj.data.materials:
        if not material or not material.use_nodes:continue
        for node in list(material.node_tree.nodes):
            if node.get('display_deposit'):continue
            if node.type=='BSDF_PRINCIPLED':
                tree=material.node_tree
                if not tree.nodes.get('batch_metallic_aov'):
                    aov=tree.nodes.new('ShaderNodeOutputAOV');aov.name='batch_metallic_aov';aov.aov_name='metallic'
                    if node.inputs['Metallic'].is_linked:tree.links.new(node.inputs['Metallic'].links[0].from_socket,aov.inputs['Value'])
                    else:aov.inputs['Value'].default_value=node.inputs['Metallic'].default_value
                    geometry=tree.nodes.new('ShaderNodeNewGeometry')
                    dot=tree.nodes.new('ShaderNodeVectorMath');dot.operation='DOT_PRODUCT';dot.inputs[1].default_value=(0,0,1)
                    tree.links.new(geometry.outputs['Normal'],dot.inputs[0])
                    upward=tree.nodes.new('ShaderNodeOutputAOV');upward.aov_name='upward';tree.links.new(dot.outputs['Value'],upward.inputs['Value'])
                if scans is not None:
                    if obj.data.uv_layers.active and scanned_surface(material,node,scans,request['environment']['dust_color']):
                        adapted.add(material.name)
                    rust_spot_detail(material,node,request.get('rust_spots',0.),request.get('material_classes',()),request['environment'],request.get('surface_dust',0.),request.get('rust_scan'))
                    continue
                if matte is not None and request.get('matte_profile'):
                    if obj.data.uv_layers.active and matte_dust_surface(material,node,matte,
                            request['environment'],request['material_classes'],request['matte_profile']):
                        adapted.add(material.name)
                    rust_spot_detail(material,node,request.get('rust_spots',0.),request.get('material_classes',()),request['environment'],request.get('surface_dust',0.),request.get('rust_scan'))
                    continue
                if reference:
                    surface_detail(material,node,request['environment'],request.get('dust_gain',1.))
                    rust_spot_detail(material,node,request.get('rust_spots',0.),request.get('material_classes',()),request['environment'],request.get('surface_dust',0.),request.get('rust_scan'))
                else:
                    scene_surface(material,node,request)
                    continue
            if node.type=='BSDF_PRINCIPLED' and node.inputs['Roughness'].is_linked:
                adjustment=material.node_tree.nodes.get('batch_roughness_delta')
                if adjustment is None:
                    original=node.inputs['Roughness'].links[0].from_socket
                    adjustment=material.node_tree.nodes.new('ShaderNodeMath')
                    adjustment.name='batch_roughness_delta';adjustment.operation='ADD';adjustment.use_clamp=True
                    material.node_tree.links.new(original,adjustment.inputs[0])
                    material.node_tree.links.new(adjustment.outputs[0],node.inputs['Roughness'])
                adjustment.inputs[1].default_value=request.get('roughness_delta',0)
            elif node.type=='BSDF_PRINCIPLED':
                base=BASE_ROUGHNESS[(material.name,node.name)]
                node.inputs['Roughness'].default_value=max(.08,min(.95,base+request.get('roughness_delta',0)))
            if matte is not None and node.type=='BSDF_PRINCIPLED' and obj.data.uv_layers.active:
                if matte_dust_surface(material,node,matte,request['environment'],request['material_classes']):
                    adapted.add(material.name)
    return sorted(adapted)


def render(request):
    folder=Path(request['output']);folder.mkdir(parents=True,exist_ok=True)
    scene=reset(request['device'],request.get('resolution',1024));obj=normalized_mesh(request['mesh'])
    vertices,faces=arrays(obj);p,rot=pose_points(vertices,request['matrix'],request['yaw'])
    matrix=np.eye(4);matrix[:3,:3]=rot
    obj.matrix_world=Matrix(matrix) @ Matrix(request['matrix'])
    camera=request.get('camera') or camera_for(p,request['elevation'])
    cam=legacy.make_camera(scene,camera['eye'],camera['target'],fov=camera['fov'])
    illuminate(scene,cam,request['environment'],request.get('light_gain',1.))
    configure_materials(obj,request)
    bpy.context.view_layer.update()
    normal=np.asarray(request.get('plane_normal',[0,0,1]),dtype=float)
    if request.get('contact_band') and request.get('receiver'):
        from display_batch.contact_geometry import description
        polys,anchor=description(p,faces,camera,request['receiver'],request['contact_band'])
    else:
        polys,anchor=contact_geometry(p,faces,camera,normal=normal,band=request.get('contact_band',.008))
    # The receiver also participates in object indirect lighting. Camera invisibility
    # keeps holes transparent while preventing an artificially illuminated underside.
    ground=make_receiver(request,normal)
    mat=bpy.data.materials.new('neutral receiver');mat.use_nodes=True
    mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.3,.3,.3,1)
    mat.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.9
    ground.data.materials.append(mat)
    ground.visible_camera=False
    pass_outputs(scene,folder)
    legacy.render(scene,folder/'object.png')
    scene.use_nodes=False
    ground.visible_camera=True
    scene.view_settings.view_transform='Standard'
    scene.render.resolution_x=scene.render.resolution_y=512
    scene.cycles.samples=16
    obj.visible_camera=False
    shadow_output(scene,folder,'with-shadow')
    obj.hide_render=True
    shadow_output(scene,folder,'without-shadow')
    return {'anchor_uv':anchor.tolist() if anchor is not None else None,
            'contact_polygons':[q.tolist() for q in polys], 'camera':camera,
            'resolution':request.get('resolution',1024),'shadow_resolution':512,
            'device':DEVICE,'passes':['combined0001.exr','diffuse0001.exr','specular0001.exr',
                'depth0001.exr','metallic0001.exr','upward0001.exr']}


def native_surface(material, bsdf, request):
    tree = material.node_tree
    metallic = bsdf.inputs['Metallic']
    kind = material_kind(material.name, metallic.default_value, metallic.is_linked,
                         request.get('material_classes', []))
    maximum = {'metal':.065, 'paint':.13, 'rust':.08, 'polymer':.035, 'mixed':.025}.get(kind, 0.)
    if not maximum:
        return
    strength = tree.nodes.get('native deposit amount')
    deposit = tree.nodes.get('native dielectric deposit')
    adjustment = tree.nodes.get('native roughness adjustment')
    if strength is None:
        output = next(n for n in tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output)
        original = output.inputs['Surface'].links[0].from_socket
        geometry = tree.nodes.new('ShaderNodeNewGeometry')
        normal = tree.nodes.new('ShaderNodeVectorMath')
        normal.operation = 'DOT_PRODUCT'
        normal.inputs[1].default_value = (0,0,1)
        tree.links.new(geometry.outputs['Normal'], normal.inputs[0])
        exposed = tree.nodes.new('ShaderNodeMapRange')
        exposed.inputs['From Min'].default_value = -1.
        exposed.inputs['To Min'].default_value = .15
        tree.links.new(normal.outputs['Value'], exposed.inputs[0])
        coordinates = tree.nodes.new('ShaderNodeTexCoord')
        variation = tree.nodes.new('ShaderNodeTexNoise')
        variation.inputs['Scale'].default_value = 28.
        tree.links.new(coordinates.outputs['Object'], variation.inputs['Vector'])
        weight = tree.nodes.new('ShaderNodeMath')
        weight.operation = 'MULTIPLY'
        tree.links.new(variation.outputs['Fac'], weight.inputs[0])
        tree.links.new(exposed.outputs[0], weight.inputs[1])
        strength = tree.nodes.new('ShaderNodeMath')
        strength.name = 'native deposit amount'
        strength.operation = 'MULTIPLY'
        tree.links.new(weight.outputs[0], strength.inputs[0])
        deposit = tree.nodes.new('ShaderNodeBsdfPrincipled')
        deposit.name = 'native dielectric deposit'
        deposit['display_deposit'] = True
        deposit.inputs['Metallic'].default_value = 0.
        deposit.inputs['Roughness'].default_value = .85
        mix = tree.nodes.new('ShaderNodeMixShader')
        tree.links.new(strength.outputs[0], mix.inputs[0])
        tree.links.new(original, mix.inputs[1])
        tree.links.new(deposit.outputs['BSDF'], mix.inputs[2])
        tree.links.new(mix.outputs[0], output.inputs['Surface'])
        adjustment = tree.nodes.new('ShaderNodeMath')
        adjustment.name = 'native roughness adjustment'
        adjustment.operation = 'ADD'
        adjustment.use_clamp = True
        roughness = bsdf.inputs['Roughness']
        if roughness.is_linked:
            tree.links.new(roughness.links[0].from_socket, adjustment.inputs[0])
        else:
            adjustment.inputs[0].default_value = roughness.default_value
        tree.links.new(adjustment.outputs[0], roughness)
    # The recognizer's specific surface-clean defect can request a stronger coat.
    # Airlight is never an input, and the original metal or polymer shader remains intact.
    extra = max(0.,request.get('dust_control',1.)-1.)
    coat = maximum + min(extra/.30,1.)*{'metal':.28,'paint':.32,'rust':.12,'polymer':.20,'mixed':.07}.get(kind,0.)
    strength.inputs[1].default_value = min(.5,coat*request.get('dust_gain',1.))
    deposit.inputs['Base Color'].default_value = (*request['environment']['dust_color'], 1)
    adjustment.inputs[1].default_value = float(np.clip(request.get('roughness_delta',0), -.04, .045))


def scene_surface(material, bsdf, request):
    if request.get('appearance')=='native':
        native_surface(material,bsdf,request)
        return
    tree=material.node_tree
    metal=bsdf.inputs['Metallic']
    kind=material.get('display_surface_kind')
    if kind is None:
        kind=material_kind(material.name,metal.default_value,metal.is_linked,request.get('material_classes',[]))
        material['display_surface_kind']=kind
    maximum=SURFACE_DUST.get(kind,0.)
    native=request.get('appearance')=='native'
    if native:
        maximum={'metal':.35,'paint':.30,'rust':.13,'polymer':.15,'mixed':.23}.get(kind,0.)
    if not maximum:
        return
    strength=tree.nodes.get('scene dust amount')
    tint=tree.nodes.get('scene dust tint')
    rough=tree.nodes.get('scene roughness')
    deposit_color=tree.nodes.get('photographed deposit tint')
    if strength is None:
        geometry=tree.nodes.new('ShaderNodeNewGeometry')
        upward=tree.nodes.new('ShaderNodeVectorMath');upward.operation='DOT_PRODUCT'
        upward.inputs[1].default_value=(0,0,1)
        tree.links.new(geometry.outputs['Normal'],upward.inputs[0])
        exposed=tree.nodes.new('ShaderNodeMapRange')
        exposed.inputs['To Min'].default_value=.22
        exposed.inputs['To Max'].default_value=1.
        tree.links.new(upward.outputs['Value'],exposed.inputs[0])
        # The shader's local AO covers recessed holes and parts hidden by the assembly.
        occlusion=tree.nodes.new('ShaderNodeAmbientOcclusion')
        occlusion.samples=4;occlusion.only_local=True;occlusion.inputs['Distance'].default_value=.018
        cover=tree.nodes.new('ShaderNodeMath');cover.operation='MULTIPLY'
        tree.links.new(occlusion.outputs['AO'],cover.inputs[0])
        tree.links.new(exposed.outputs[0],cover.inputs[1])
        variation=tree.nodes.new('ShaderNodeTexNoise')
        variation.inputs['Scale'].default_value=26 if native else 18
        variation.inputs['Detail'].default_value=2
        if native:
            coords=tree.nodes.new('ShaderNodeTexCoord')
            tree.links.new(coords.outputs['Object'],variation.inputs['Vector'])
            grain=tree.nodes.new('ShaderNodeTexNoise')
            grain.inputs['Scale'].default_value=110;grain.inputs['Detail'].default_value=2
            tree.links.new(coords.outputs['Object'],grain.inputs['Vector'])
        mottling=tree.nodes.new('ShaderNodeMapRange')
        mottling.inputs['To Min'].default_value=.15
        mottling.inputs['To Max'].default_value=1.
        tree.links.new(variation.outputs['Fac'],mottling.inputs[0])
        weight=tree.nodes.new('ShaderNodeMath');weight.operation='MULTIPLY'
        tree.links.new(cover.outputs[0],weight.inputs[0]);tree.links.new(mottling.outputs[0],weight.inputs[1])
        strength=tree.nodes.new('ShaderNodeMath');strength.name='scene dust amount';strength.operation='MULTIPLY'
        tree.links.new(weight.outputs[0],strength.inputs[0])
        tint=tree.nodes.new('ShaderNodeMixRGB');tint.name='scene dust tint'
        base=bsdf.inputs['Base Color']
        if base.is_linked:tree.links.new(base.links[0].from_socket,tint.inputs[1])
        else:tint.inputs[1].default_value=base.default_value
        tree.links.new(strength.outputs[0],tint.inputs[0]);tree.links.new(tint.outputs[0],base)
        if native:
            # A 3D deposit layer has irregular shade and micro-normal variation,
            # independent of pixel coordinates or frame crop.
            grain_value=tree.nodes.new('ShaderNodeMapRange')
            grain_value.inputs['From Min'].default_value=.28
            grain_value.inputs['From Max'].default_value=.72
            grain_value.inputs['To Min'].default_value=.35
            grain_value.inputs['To Max'].default_value=1.25
            tree.links.new(grain.outputs['Fac'],grain_value.inputs[0])
            deposit_color=tree.nodes.new('ShaderNodeMixRGB')
            deposit_color.name='photographed deposit tint';deposit_color.blend_type='MULTIPLY'
            deposit_color.inputs[0].default_value=1.
            tree.links.new(grain_value.outputs[0],deposit_color.inputs[2])
            tree.links.new(deposit_color.outputs[0],tint.inputs[2])
            micro=tree.nodes.new('ShaderNodeBump')
            micro.inputs['Distance'].default_value=.0012
            bump_weight=tree.nodes.new('ShaderNodeMath');bump_weight.operation='MULTIPLY'
            bump_weight.inputs[1].default_value=.40 if kind=='polymer' else 1.1
            tree.links.new(strength.outputs[0],bump_weight.inputs[0])
            tree.links.new(bump_weight.outputs[0],micro.inputs['Strength'])
            tree.links.new(grain.outputs['Fac'],micro.inputs['Height'])
            if bsdf.inputs['Normal'].is_linked:
                tree.links.new(bsdf.inputs['Normal'].links[0].from_socket,micro.inputs['Normal'])
            tree.links.new(micro.outputs[0],bsdf.inputs['Normal'])
        rough=tree.nodes.new('ShaderNodeMath');rough.name='scene roughness';rough.operation='ADD';rough.use_clamp=True
        original=bsdf.inputs['Roughness']
        if original.is_linked:tree.links.new(original.links[0].from_socket,rough.inputs[0])
        else:rough.inputs[0].default_value=original.default_value
        tree.links.new(rough.outputs[0],original)
        if native and kind=='metal':
            # Irregular oxide/dust broadens the lobe. Keep linked material values underneath.
            oxide=tree.nodes.new('ShaderNodeMath');oxide.name='oxide roughness'
            oxide.operation='MULTIPLY_ADD'
            oxide.inputs[1].default_value=.14
            tree.links.new(mottling.outputs[0],oxide.inputs[0])
            tree.links.new(oxide.outputs[0],rough.inputs[1])
            deposit=tree.nodes.new('ShaderNodeMath');deposit.operation='MULTIPLY_ADD'
            deposit.inputs[1].default_value=-.7;deposit.inputs[2].default_value=1.
            tree.links.new(strength.outputs[0],deposit.inputs[0])
            layer=tree.nodes.new('ShaderNodeMath');layer.operation='MULTIPLY'
            if metal.is_linked:tree.links.new(metal.links[0].from_socket,layer.inputs[0])
            else:layer.inputs[0].default_value=metal.default_value
            tree.links.new(deposit.outputs[0],layer.inputs[1])
            tree.links.new(layer.outputs[0],metal)
    environment=request['environment']
    dust=request.get('dust_gain',1.)*request.get('dust_control',1.)
    if native:
        dust*=float(np.clip(.75+environment['sensor']['surface_detail_rms']*14,.80,1.35))
    strength.inputs[1].default_value=min(maximum*1.8,maximum*(.7+environment.get('haze',0))*dust)
    if deposit_color is None:
        tint.inputs[2].default_value=(*environment['dust_color'],1)
    else:
        deposit_color.inputs[1].default_value=(*environment['dust_color'],1)
    delta=request.get('roughness_delta',0)
    adjustment=max(-.035,min(.045,delta))*(.25 if kind=='polymer' else 1.)
    oxide=tree.nodes.get('oxide roughness')
    if oxide is not None:
        oxide.inputs[2].default_value=.02+adjustment
    else:
        rough.inputs[1].default_value=adjustment


def scene_lighting(state, camera, environment, gain, reference, hdri=None):
    scene=bpy.context.scene
    local_light=environment.get('local_irradiance')
    key=(camera if not reference or local_light else None,environment,gain,reference,hdri)
    if state.get('lighting_key')==key:return
    state['lighting_key']=key
    if 'world_image' not in state:
        nodes=scene.world.node_tree.nodes;nodes.clear();links=scene.world.node_tree.links
        bg=nodes.new('ShaderNodeBackground');output=nodes.new('ShaderNodeOutputWorld')
        image=bpy.data.images.new('batch environment',width=256,height=128,float_buffer=True)
        image.colorspace_settings.name='Non-Color'
        texture=nodes.new('ShaderNodeTexEnvironment');texture.image=image
        links.new(texture.outputs['Color'],bg.inputs['Color']);links.new(bg.outputs[0],output.inputs['Surface'])
        bg.inputs['Strength'].default_value=.65
        state['world_image']=image
        state['lights']=[]
        for name in ('key','fill'):
            light=bpy.data.lights.new(name,'AREA');light.shape='DISK'
            obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);state['lights'].append(obj)
    yy,xx=np.mgrid[0:128,0:256]
    direction=environment.get('key_direction')
    color=np.asarray(environment['world_color'])
    probe=environment.get('reflection_probe')
    if local_light:
        from display_batch.local_light import light_rig
        sky, lights = light_rig(environment, camera, gain, (128,256))
        target=Vector(camera['target'])
    elif reference:
        sky=.12+.55*(yy/128)**2
        reflection=np.exp(-((xx/256-.24)/.035)**2-((yy/128-.60)/.23)**2)
        reflection+=.5*np.exp(-((xx/256-.76)/.14)**2-((yy/128-.52)/.22)**2)
        sky=sky+6.*reflection
        key_offset=(direction[0]*3,-2,3+max(0,-direction[1])) if direction else (-2,-3,4)
        target=Vector((0,0,0))
        lights=[(key_offset,160*gain,3.2),((3,1,2),65,4.5)]
    else:
        # Broad sky and horizon gradient, with no synthetic specular stripe.
        spread=environment.get('light_spread',.35)
        sky=.60+(.65+.45*spread)*(yy/128)**2
        light=environment.get('light_level',1.)
        target=Vector(camera['target'])
        if direction:
            from display_batch.geometry import illumination_axis
            offset=Vector(illumination_axis(camera,direction))*4.6
            lights=[(offset,210*light*gain,4.6+2*(1-spread)),((0,2.5,2.8),70*light,6.5)]
        else:
            lights=[((0,-.4,4.6),155*light*gain,7.5),((0,2.2,2.7),62*light,6.8)]
        if probe is not None:
            # The observed surroundings dominate reflections. These broad area
            # lights provide modest soft shadows where the sky is unobserved.
            lights=[(offset,power*.46,size*1.28) for offset,power,size in lights]
        if environment.get('calibration'):
            # Light profiles alter the broad key/fill ratio as well as its energy.
            # The soft profile widens the source; it cannot create a narrow studio stripe.
            width=1.30 if gain<1 else .85 if gain>1 else 1.
            offset,power,size=lights[0]
            lights[0]=(offset,power,size*width)
            offset,power,size=lights[1]
            lights[1]=(offset,power*(1.12 if gain<1 else .85 if gain>1 else 1.),size)
    pixels=np.ones((128,256,4),np.float32)
    if hdri is not None:
        pixels[:,:,:3]=overcast_pixels(state,hdri,environment)
        lights=[(offset,power*.55,size*1.20) for offset,power,size in lights]
    elif probe is not None:
        pixels[:,:,:3]=np.repeat(np.repeat(np.asarray(probe,np.float32),2,axis=0),2,axis=1)
        if environment.get('calibration'):
            pixels[:,:,:3]*=gain
    else:
        pixels[:,:,:3]=sky[:,:,None]*color
    image=state['world_image'];image.pixels.foreach_set(pixels.ravel());image.update()
    for obj,(offset,power,size) in zip(state['lights'],lights):
        obj.location=target+Vector(offset)
        obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
        obj.data.energy=power;obj.data.size=size;obj.data.color=environment['light_color']


def set_window(scene, window):
    frame=window['frame'];x,y,r,b=window['rect']
    render=scene.render
    render.resolution_x=render.resolution_y=frame
    render.use_border=window['rect']!=[0,0,frame,frame]
    render.use_crop_to_border=render.use_border
    if render.use_border:
        # Blender truncates floating bounds. Offset by a small subpixel fraction
        # so integer pixel edges round to the intended window in single precision.
        render.border_min_x=(x+.01)/frame;render.border_max_x=min(1.,(r+.01)/frame)
        render.border_min_y=(frame-b+.01)/frame;render.border_max_y=min(1.,(frame-y+.01)/frame)


def shadow_catcher(scene,ground,path):
    """Export the Cycles multiplicative pass in linear RGB."""
    layer=scene.view_layers[0]
    ground.is_shadow_catcher=True
    layer.cycles.use_pass_shadow_catcher=True
    try:
        scene.use_nodes=True
        tree=scene.node_tree
        tree.nodes.clear()
        source=tree.nodes.new('CompositorNodeRLayers')
        composite=tree.nodes.new('CompositorNodeComposite')
        tree.links.new(source.outputs['Image'],composite.inputs[0])
        output=tree.nodes.new('CompositorNodeOutputFile')
        output.base_path=str(path.parent)
        output.format.file_format='OPEN_EXR'
        output.format.color_mode='RGB'
        output.format.color_depth='16'
        output.file_slots[0].path='with-shadow-linear'
        tree.links.new(source.outputs['Shadow Catcher'],output.inputs[0])
        bpy.ops.render.render(write_still=False)
        image=bpy.data.images.load(str(path),check_existing=False)
        try:
            data=np.empty(image.size[0]*image.size[1]*4,np.float32)
            image.pixels.foreach_get(data)
            rgb=data.reshape(-1,4)[:,:3]
            if not np.isfinite(rgb).all() or int((rgb.mean(axis=1)<.98).sum())<12:
                raise ValueError('shadow_catcher_pass_missing_finite_contrast')
        finally:
            bpy.data.images.remove(image)
    finally:
        ground.is_shadow_catcher=False
        layer.cycles.use_pass_shadow_catcher=False
        scene.use_nodes=False


def render_parts(request):
    global LIVE_RENDER, BASE, BASE_PATH
    started=time.monotonic()
    reference=request['appearance'] in ('reference','local')
    matte=request.get('matte_partitions')
    integration_mode=request['environment'].get('integration_mode','legacy')
    mode=(request['appearance'],bool(request.get('surface_scans')),matte is not None,integration_mode)
    if LIVE_RENDER is None or LIVE_RENDER['mode']!=mode:
        scene=reset(request['device'])
        for obj in list(scene.objects):bpy.data.objects.remove(obj,do_unlink=True)
        if BASE is not None:BASE.use_fake_user=False
        BASE=BASE_PATH=None
        bpy.data.orphans_purge(do_recursive=True)
        cam=legacy.make_camera(scene,(0,-10,8),(0,0,0))
        LIVE_RENDER={'camera':cam,'mode':mode}
    scene=bpy.context.scene;state=LIVE_RENDER
    scene.render.use_persistent_data=request.get('persistent_data',True)
    imported=0.;import_seconds=0.
    mesh_key=(request['mesh'],request['mesh_hash'],matte['signature'] if matte is not None else None,
              request.get('matte_profile'),integration_mode,request.get('render_detail','standard'),
              request.get('rust_spots',0.),request.get('surface_dust',0.),
              tuple(request['environment'].get('dust_color',())) if request.get('rust_spots',0.) else (),
              request.get('rust_scan',{}).get('sha256'),request.get('rust_scan',{}).get('roughness_sha256'),
              request.get('rust_scan',{}).get('height_sha256'),request.get('rust_scan',{}).get('detail_sha256'),
              request.get('rust_scan',{}).get('relief_strength',1.))
    if state.get('mesh_key')!=mesh_key:
        begin=time.monotonic()
        if 'asset' in state:
            bpy.data.objects.remove(state['asset'],do_unlink=True)
            BASE.use_fake_user=False
            BASE=BASE_PATH=None
            bpy.data.orphans_purge(do_recursive=True)
        asset=normalized_mesh(request['mesh'])
        state.update(asset=asset,arrays=arrays(asset),mesh_key=mesh_key)
        imported=1;import_seconds=time.monotonic()-begin
    obj=state['asset'];vertices,faces=state['arrays']
    points,rotation=pose_points(vertices,request['matrix'],request['yaw'])
    matrix=np.eye(4);matrix[:3,:3]=rotation
    obj.matrix_world=Matrix(matrix)@Matrix(request['matrix'])
    camera=request.get('camera') or camera_for(points,request['elevation'])
    cam=state['camera'];cam.location=camera['eye']
    cam.rotation_euler=(Vector(camera['target'])-cam.location).to_track_quat('-Z','Y').to_euler()
    if camera.get('basis'):
        b=np.asarray(camera['basis'])
        cam.rotation_euler=Matrix(np.column_stack([b[0],b[1],-b[2]]).tolist()).to_euler()
    cam.data.angle=math.radians(camera['fov'])
    hdri=request.get('scanned_hdri')
    scene_lighting(state,camera,request['environment'],request.get('light_gain',1.),reference and not hdri,hdri)
    adapted=configure_materials(obj,request,reference)
    base_faces=len(obj.data.polygons)
    fine_render_mesh(obj,request)
    normal=np.asarray(request.get('plane_normal',[0,0,1]),dtype=float)
    receiver_key=(request.get('receiver',{}).get('patches',[]),normal.tolist())
    if state.get('receiver_key')!=receiver_key:
        if 'ground' in state:
            old=state['ground'];mesh=old.data
            bpy.data.objects.remove(old,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        ground=make_receiver(request,normal)
        if 'receiver_material' not in state:
            material=bpy.data.materials.new('neutral receiver');material.use_nodes=True
            shader=material.node_tree.nodes.get('Principled BSDF')
            shader.inputs['Base Color'].default_value=(.3,.3,.3,1)
            shader.inputs['Roughness'].default_value=.9
            state['receiver_material']=material
        ground.data.materials.append(state['receiver_material'])
        state.update(ground=ground,receiver_key=receiver_key)
    ground=state['ground']
    if request.get('contact_band') and request.get('receiver'):
        from display_batch.contact_geometry import description
        polys,anchor=description(points,faces,camera,request['receiver'],request['contact_band'])
    else:
        polys,anchor=contact_geometry(points,faces,camera,normal=normal,band=request.get('contact_band',.008))
    timings={'asset_import_seconds':import_seconds,'scene_setup_seconds':time.monotonic()-started-import_seconds}
    scene.render.image_settings.color_mode='RGBA'
    scene.cycles.use_denoising=True
    shadow_basis='scene_linear_EXR_ratio'
    catcher_fallback=None
    for part,path in request['parts'].items():
        folder=Path(path)
        begin=time.monotonic()
        is_object=part=='object'
        obj.hide_render=part=='without_shadow'
        obj.visible_camera=is_object
        ground.visible_camera=not is_object
        set_window(scene,request['windows']['object' if is_object else 'shadow'])
        if request.get('render_detail')=='fine':scene.cycles.samples=128 if is_object else 32
        else:scene.cycles.samples=32 if is_object else 16
        if is_object:
            scene.render.image_settings.file_format='PNG'
            scene.render.image_settings.color_depth='8'
            scene.view_settings.view_transform='AgX'
            pass_outputs(scene,folder)
            legacy.render(scene,folder/'object.png')
        else:
            scene.use_nodes=False
            scene.view_settings.view_transform='Standard'
            scene.render.image_settings.file_format='OPEN_EXR'
            scene.render.image_settings.color_depth='16'
            filename=part.replace('_','-')+'-linear0001.exr'
            if part=='with_shadow' and hdri and request.get('receiver',{}).get('patches'):
                catcher_start=time.monotonic()
                try:
                    shadow_catcher(scene,ground,folder/filename)
                    shadow_basis='cycles_shadow_catcher_ratio'
                except (KeyError,RuntimeError,ValueError,OSError) as error:
                    catcher_fallback=str(error)[:200]
                timings['shadow_catcher_seconds']=time.monotonic()-catcher_start
            elif part=='with_shadow' and hdri:
                catcher_fallback='finite_receiver_unavailable'
            if part!='with_shadow' or shadow_basis!='cycles_shadow_catcher_ratio':
                legacy.render(scene,folder/filename)
        timings[part+'_seconds']=time.monotonic()-begin
    return {'anchor_uv':anchor.tolist() if anchor is not None else None,
            'contact_polygons':[q.tolist() for q in polys],'camera':camera,'device':DEVICE,
            'scanned_materials':adapted if matte is None else [],
            'matte_materials':adapted if matte is not None else [],
            'integration_mode':integration_mode,
            'render_mesh_detail':{'profile':request.get('render_detail','standard'),
                'base_faces':base_faces,'render_faces':base_faces*(4 if request.get('render_detail')=='fine' else 1),
                'subdivision':'simple_level_1' if request.get('render_detail')=='fine' else 'none',
                'cycles_samples':{'object':128,'shadow':32} if request.get('render_detail')=='fine' else {'object':32,'shadow':16},
                'rust_spots':float(request.get('rust_spots',0.)),
                'rust_source':request.get('rust_scan',{}).get('source','procedural'),
                'rust_source_sha256':request.get('rust_scan',{}).get('sha256'),
                'rust_roughness_sha256':request.get('rust_scan',{}).get('roughness_sha256'),
                'rust_height_sha256':request.get('rust_scan',{}).get('height_sha256'),
                'rust_detail_sha256':request.get('rust_scan',{}).get('detail_sha256'),
                'rust_relief_strength':request.get('rust_scan',{}).get('relief_strength',1.)},
            'lighting_basis':request['environment'].get('lighting_basis'),
            'shadow_basis':shadow_basis,'shadow_catcher_fallback':catcher_fallback,
            'imports':int(imported),'timings':timings}


def main():
    for line in sys.stdin:
        try:
            request=json.loads(line)
        except ValueError:continue
        try:
            result={'export':export,'settle':settle,'render':render,'render_parts':render_parts}[request['operation']](request)
            atomic_json(request['reply'],result)
        except Exception as error:
            traceback.print_exc()
            atomic_json(request['reply'],{'error':str(error)[:300]})


if __name__=='__main__':main()
