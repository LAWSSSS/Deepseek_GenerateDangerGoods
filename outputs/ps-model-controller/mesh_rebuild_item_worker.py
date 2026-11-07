"""Blender worker for the review-only parametric 01 valve reconstruction."""
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_worker as legacy

request = json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf8'))
out = Path(request['out'])
config = request['config']
objects = []


def atlas_material():
    material = bpy.data.materials.new('Observed and inferred PBR atlas')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get('Principled BSDF')
    def texture(name, noncolor=False):
        node = nodes.new('ShaderNodeTexImage')
        node.image = bpy.data.images.load(request['textures'][name], check_existing=True)
        node.image.pack()
        if noncolor:
            node.image.colorspace_settings.name = 'Non-Color'
        node.interpolation = 'Linear'
        return node
    color = texture('base_color')
    metal = texture('metallic_roughness', True)
    normal = texture('normal', True)
    split = nodes.new('ShaderNodeSeparateColor')
    links.new(color.outputs['Color'], principled.inputs['Base Color'])
    links.new(metal.outputs['Color'], split.inputs['Color'])
    links.new(split.outputs['Green'], principled.inputs['Roughness'])
    links.new(split.outputs['Blue'], principled.inputs['Metallic'])
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = .45
    links.new(normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
    return material


def assign(obj, tile, *, constant_uv=False):
    obj.data.materials.clear()
    obj.data.materials.append(material)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name='Atlas')
    uv = obj.data.uv_layers.active.data
    col, top_row = tile % 2, tile // 2
    for loop in uv:
        u, v = loop.uv if not constant_uv else (.5, .5)
        loop.uv = (col*.5 + .01 + u*.48, (1-top_row)*.5 + .01 + v*.48)
    objects.append(obj)
    return obj


def finish(obj, name, tile, bevel=.002, smooth=True):
    obj.name = name
    if obj.type != 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH')
        obj = bpy.context.object
    if bevel:
        mod = obj.modifiers.new('Small manufactured edge radius', 'BEVEL')
        mod.width = bevel
        mod.segments = 3
        mod.limit_method = 'ANGLE'
        mod.use_clamp_overlap = True
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
    for poly in obj.data.polygons:
        poly.use_smooth = smooth and len(poly.vertices) == 4
    return assign(obj, tile)


def cylinder(name, radius, z0, z1, tile, *, xy=(0, 0), segments=None, bevel=.0018):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments or config['radial_segments'], radius=radius,
                                        depth=z1-z0, location=(xy[0], xy[1], (z0+z1)/2))
    return finish(bpy.context.object, name, tile, bevel, segments != 6)


def cone(name, radius_bottom, radius_top, z0, z1, tile):
    bpy.ops.mesh.primitive_cone_add(vertices=128, radius1=radius_bottom, radius2=radius_top,
                                    depth=z1-z0, location=(0,0,(z0+z1)/2))
    return finish(bpy.context.object, name, tile, .0015)


def along(name, start, end, radius, tile, segments=32, bevel=.001):
    start, end = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius, depth=(end-start).length,
                                        location=(start+end)/2)
    obj = bpy.context.object
    obj.rotation_euler = (end-start).to_track_quat('Z', 'Y').to_euler()
    return finish(obj, name, tile, bevel, segments != 6)


def torus(name, major, minor, z, tile):
    bpy.ops.mesh.primitive_torus_add(major_segments=config['radial_segments'], minor_segments=20,
                                     location=(0, 0, z), major_radius=major, minor_radius=minor)
    return finish(bpy.context.object, name, tile, bevel=0)


def side_bulge(z):
    # A rounded, intersecting casting around the branch rather than a boxy
    # extruded tube. The buried part is intentionally inside the main body.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=48, location=(0, .148, z))
    obj=bpy.context.object
    obj.scale=(.110,.093,.118)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return finish(obj, 'Rounded cast branch', 0, bevel=0)


def mounting_ring():
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=.151, depth=.022, location=(0,0,.029))
    plate = bpy.context.object
    cutters = []
    for i in range(8):
        angle = math.tau*i/8
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=.010, depth=.05,
                location=(.131*math.cos(angle), .131*math.sin(angle), .029))
        cutters.append(bpy.context.object)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in cutters: obj.select_set(True)
    bpy.context.view_layer.objects.active = cutters[0]
    bpy.ops.object.join()
    cutter = bpy.context.object
    mod = plate.modifiers.new('Eight mounting holes', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'; mod.solver = 'EXACT'; mod.object = cutter
    bpy.context.view_layer.objects.active = plate
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    finish(plate, 'Bottom mounting flange with holes', 1, .001)


def curved_label(text, z, size):
    bpy.ops.object.text_add(location=(0,0,0), rotation=(math.pi/2, 0, math.pi/2))
    obj = bpy.context.object; obj.data.body = text
    obj.data.align_x = 'CENTER'; obj.data.align_y = 'CENTER'; obj.data.size = size
    obj.data.extrude = .0003
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    radius = config['body_radius']
    for vertex in obj.data.vertices:
        y = vertex.co.y
        vertex.co.x = math.sqrt(max(.0001, radius*radius-y*y)) + .0008 + vertex.co.x
        vertex.co.z += z
    obj.name = 'Blue inscription ' + text
    assign(obj, 2, constant_uv=True)


def curved_underline():
    radius=config['body_radius']; verts=[]; faces=[]
    for i in range(65):
        y=-.100 + i*.200/64
        x=math.sqrt(radius*radius-y*y)+.0012
        verts.extend(((x,y,.260),(x,y,.266)))
        if i:
            j=2*i;faces.append((j-2,j,j+1,j-1))
    mesh=bpy.data.meshes.new('Continuous curved painted underline')
    mesh.from_pydata(verts, [], faces);mesh.update()
    obj=bpy.data.objects.new('Continuous curved painted underline',mesh)
    bpy.context.scene.collection.objects.link(obj)
    assign(obj,2,constant_uv=True)


def build():
    mounting_ring()
    cylinder('Lower base neck', .128, .040, .092, 0)
    cylinder('Lower body rim', .174, .083, .116, 0)
    cylinder('Cast valve body', config['body_radius'], config['body_bottom'], config['body_top']-.048, 0, bevel=.003)
    cylinder('Body top gasket', .177, .406, .412, 3, bevel=.0007)
    cylinder('Body top flange', .178, .411, .450, 1, bevel=.0015)
    cylinder('Body top flange lip', .176, .449, .459, 1, bevel=.001)
    for i in range(12):
        angle = math.tau*i/12
        x,y = .151*math.cos(angle), .151*math.sin(angle)
        cylinder(f'Top flange bolt {i+1:02}', .011, .458, .471, 1, xy=(x,y), segments=6, bevel=.0007)
    cylinder('Blue stem base', .058, .454, .528, 2)
    cylinder('Machined stem sleeve', config['stem_radius'], config['stem_bottom'], config['stem_top'], 1, bevel=.0015)
    cylinder('Blue upper stem collar', .055, .695, .730, 2)
    cylinder('Upper flange lower', .086, .728, .746, 1)
    cylinder('Upper flange upper', .088, .746, .762, 1)
    cone('Tapered blue bonnet', .039, .032, .760, .902, 2)
    for i in range(8):
        angle = math.tau*i/8
        x,y = .071*math.cos(angle), .071*math.sin(angle)
        cylinder(f'Bonnet bolt {i+1:02}', .007, .760, .772, 1, xy=(x,y), segments=6, bevel=.0005)
    cylinder('Handwheel hub lower', .035, .897, .929, 2)
    cylinder('Handwheel center hub', .025, .921, .967, 2)
    torus('Four-spoke wheel rim', config['wheel_radius']-config['wheel_tube'], config['wheel_tube'], config['wheel_height'], 2)
    for i in range(4):
        angle=math.tau*i/4
        along(f'Handwheel spoke {i+1}', (.018*math.cos(angle), .018*math.sin(angle), config['wheel_height']),
              ((config['wheel_radius']-.009)*math.cos(angle), (config['wheel_radius']-.009)*math.sin(angle), config['wheel_height']), .008, 2)
    cylinder('Wheel center cap', .012, .965, .978, 1, segments=6, bevel=.001)
    z = config['side_axis_height']
    side_bulge(z)
    along('Side flange inner', (0,.216,z), (0,.246,z), config['side_radius'], 1, 128, .0015)
    along('Side gasket seam', (0,.246,z), (0,.251,z), .120, 3, 128, .0005)
    along('Side cover', (0,.251,z), (0,.266,z), .116, 1, 128, .0015)
    for i in range(8):
        angle=math.tau*i/8
        x,zz=.100*math.cos(angle), z+.100*math.sin(angle)
        along(f'Side cover bolt {i+1:02}', (x,.265,zz), (x,.279,zz), .009, 1, 6, .0005)
    curved_label('DN350', .300, .052)
    curved_underline()
    curved_label('6', .229, .053)
    # Match the existing library: the normalized model is centered near zero.
    for obj in objects: obj.location.z -= .496


def camera(scene, direction, center, span):
    direction=Vector(direction).normalized()
    cam=legacy.make_camera(scene, Vector(center)+direction*span*5, center, ortho_scale=span*1.27)
    if abs(direction.z)>.999:
        cam.rotation_mode='QUATERNION'
        cam.rotation_quaternion=cam.rotation_quaternion @ Quaternion((0,0,1), math.pi/2)
    return cam


def bounds(obs):
    points = [point for obj in obs for point in legacy.world_vertices(obj)]
    lo=Vector([min(p[i] for p in points) for i in range(3)])
    hi=Vector([max(p[i] for p in points) for i in range(3)])
    return lo, hi, (lo+hi)/2, max(hi-lo)


def intersecting_faces(obj):
    """Find non-adjacent overlapping faces inside one component only."""
    bm=bmesh.new();bm.from_mesh(obj.data)
    # Text conversion and UV boundaries may duplicate geometric vertices.
    # Weld only the temporary audit mesh before deciding face adjacency.
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-7)
    bm.faces.ensure_lookup_table()
    tree=BVHTree.FromBMesh(bm,epsilon=1e-8)
    bad=set()
    for a,b in tree.overlap(tree):
        if a>=b:continue
        if set(bm.faces[a].verts).isdisjoint(bm.faces[b].verts):bad.add((a,b))
    bm.free()
    return len(bad)


def draw(scene, path, *, resolution, samples):
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


scene=legacy.setup_scene('CPU', 512)
scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.7
scene.world.node_tree.nodes.get('Background').inputs['Color'].default_value=(1,1,1,1)
for obj in tuple(bpy.data.objects):
    if obj.type == 'LIGHT': bpy.data.objects.remove(obj, do_unlink=True)
for name, location, energy, size in (('large softbox', (3,-4,5), 180, 5), ('fill softbox', (-4,3,2), 90, 5)):
    data=bpy.data.lights.new(name, 'AREA'); data.energy=energy; data.shape='DISK'; data.size=size
    light=bpy.data.objects.new(name, data); scene.collection.objects.link(light)
    light.location=location; light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler()
material=atlas_material()
build()
bpy.context.view_layer.update()
low, high, center, span = bounds(objects)
bpy.ops.object.select_all(action='DESELECT')
for obj in objects: obj.select_set(True)
bpy.context.view_layer.objects.active=objects[0]
bpy.ops.export_scene.gltf(filepath=str(out/'mesh.glb'), export_format='GLB', use_selection=True,
                          export_yup=False, export_apply=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'model.blend'))
parts = [{'name':o.name, 'vertices':len(o.data.vertices), 'faces':len(o.data.polygons)} for o in objects]
intersections={o.name:intersecting_faces(o) for o in objects}
render_dir=out/'renders';render_dir.mkdir(exist_ok=True)
directions={'front':(1,0,.24), 'back':(-1,0,.24), 'left':(0,1,.24), 'right':(0,-1,.24), 'top':(0,0,1), 'bottom':(0,0,-1)}
resolution=512 if request['draft'] else 768
samples=8 if request['draft'] else 24
for name, direction in directions.items():
    cam=camera(scene,direction,center,span);draw(scene,render_dir/f'new-{name}.png',resolution=resolution,samples=samples)
    bpy.data.objects.remove(cam,do_unlink=True)
source=request['source_camera']
cam=legacy.make_camera(scene,(1.9,0,.45),(0,0,0),fov=source['fov_deg'])
cam.data.shift_x=source.get('shift_x',0);cam.data.shift_y=source.get('shift_y',0)
draw(scene,render_dir/'new-source.png',resolution=512,samples=samples)
bpy.data.objects.remove(cam,do_unlink=True)
if not request['draft']:
    for i in range(24):
        angle=math.tau*i/24
        cam=camera(scene,(math.cos(angle),math.sin(angle),.24),center,span)
        draw(scene,render_dir/f'turn-{i:02}.png',resolution=512,samples=8)
        bpy.data.objects.remove(cam,do_unlink=True)
    for name, target, direction, scale in (
        ('wheel', (0,0,.455), (2,-3,2), .52),
        ('stem', (0,0,.287), (3,-3,1), .38),
        ('side', (0,.20,-.23), (1,3,.7), .45),
        ('surface', (.18,0,-.21), (3,-.4,.25), .32)):
        cam=camera(scene,direction,Vector(target),scale)
        draw(scene,render_dir/f'detail-{name}.png',resolution=1024,samples=32)
        bpy.data.objects.remove(cam,do_unlink=True)
    # A second orientation of the broad key light checks that highlights move.
    key=bpy.data.objects['large softbox'];key.location=(-3,4,5)
    key.rotation_euler=(-key.location).to_track_quat('-Z','Y').to_euler()
    cam=camera(scene,directions['front'],center,span)
    draw(scene,render_dir/'new-light-alternate.png',resolution=768,samples=24)
    bpy.data.objects.remove(cam,do_unlink=True)
    key.location=(3,-4,5);key.rotation_euler=(-key.location).to_track_quat('-Z','Y').to_euler()
    wheel_names={'Four-spoke wheel rim','Handwheel center hub','Wheel center cap'} | {f'Handwheel spoke {i}' for i in range(1,5)}
    for obj in objects: obj.hide_render=obj.name not in wheel_names
    cam=camera(scene,(0,0,1),Vector((0,0,config['wheel_height']-.496)),.38)
    draw(scene,render_dir/'wheel-only-top.png',resolution=512,samples=16)
    bpy.data.objects.remove(cam,do_unlink=True)
    for obj in objects: obj.hide_render=obj.name!='Bottom mounting flange with holes'
    cam=camera(scene,(0,0,-1),Vector((0,0,.029-.496)),.32)
    draw(scene,render_dir/'flange-only-bottom.png',resolution=512,samples=16)
    bpy.data.objects.remove(cam,do_unlink=True)
    for obj in objects: obj.hide_render=False
for obj in objects: obj.hide_render=True
old=legacy.load_mesh(request['original_mesh'])
_, _, oldcenter, oldspan=bounds([old])
for name, direction in directions.items():
    cam=camera(scene,direction,oldcenter,oldspan);draw(scene,render_dir/f'old-{name}.png',resolution=resolution,samples=samples)
    bpy.data.objects.remove(cam,do_unlink=True)
data={'parts':parts,'bounds':[list(low),list(high)],'span':span,'wheel_spokes':4,'bottom_mounting_holes':8,
      'polygons':sum(len(o.data.polygons) for o in objects),
      'triangles':sum(max(0,len(p.vertices)-2) for o in objects for p in o.data.polygons), 'texture_size':2048,
      'self_intersecting_components':{k:v for k,v in intersections.items() if v},
      'intersection_check':'BVH 同部件不共享顶点的面交叠；部件之间的装配交叠未判定',
      'geometry_basis':'six-view panels, uncalibrated', 'render_samples':samples}
(out/'worker-result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
