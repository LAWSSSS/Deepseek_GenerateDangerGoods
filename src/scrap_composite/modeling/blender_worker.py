"""Import a static model, pack the actual mesh, and verify the exported assets."""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scrap_composite.data.files import atomic_json
from scrap_composite.runtime import enable_legacy_tools

enable_legacy_tools()
from blender_worker import make_camera, render


def subjects(selector, available):
    if selector:
        kind, _, name = selector.partition(':')
        if kind == 'collection' and name in bpy.data.collections:
            return [obj for obj in bpy.data.collections[name].all_objects if obj.type == 'MESH']
        if kind == 'object' and name in available:
            return [available[name]]
        raise ValueError('主体不在文件中。可用网格：' + ', '.join(sorted(available)))
    if len(available) != 1:
        raise ValueError('请选择主体 object:名称 或 collection:名称。可用网格：' + ', '.join(sorted(available)))
    return list(available.values())


def material_images(meshes):
    materials = {material for obj in meshes for material in obj.data.materials if material}
    pending = [material.node_tree for material in materials if material.node_tree]
    visited = set()
    images = set()
    while pending:
        tree = pending.pop()
        if tree in visited:
            continue
        visited.add(tree)
        for node in tree.nodes:
            if getattr(node, 'object', None) is not None or node.type in ('SCRIPT', 'TEX_POINTDENSITY'):
                raise ValueError('主体材质含外部对象或运行脚本依赖')
            if getattr(node, 'image', None):
                images.add(node.image)
            if node.type == 'GROUP' and node.node_tree:
                pending.append(node.node_tree)
    return images


def packed_images(meshes):
    images = material_images(meshes)
    for image in images:
        if image.source == 'GENERATED':
            continue
        if not image.packed_files:
            raise ValueError('主体贴图未打包或缺失：' + image.name)
    return len(images)


def bounds(obj):
    points = [obj.matrix_world @ point.co for point in obj.data.vertices]
    if not points or not obj.data.polygons or not all(math.isfinite(x) for point in points for x in point):
        raise ValueError('模型网格为空或含无效坐标')
    low = Vector([min(p[i] for p in points) for i in range(3)])
    high = Vector([max(p[i] for p in points) for i in range(3)])
    if min(high - low) <= 1e-8:
        raise ValueError('模型没有有效的三维体积范围')
    return low, high


def use_nodes(obj, finish):
    from scrap_composite.generation.component_shader import finish_surface

    applied = set()
    for material in obj.data.materials:
        if not material or not material.get('component_pbr_region'):
            continue
        shader = next((node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED'), None)
        if shader and finish_surface(material, shader, finish, 0.):
            applied.add(material['component_pbr_region'])
    planned = {region['id'] for record in finish['regions']['materials'].values() for region in record['components']}
    if applied != planned:
        raise ValueError('已验证贴图未完整应用到对应部件')


def import_subject(request):
    source = request['mesh']
    if Path(source).suffix == '.blend':
        bpy.ops.wm.open_mainfile(filepath=source, load_ui=False, use_scripts=False)
        mesh_objects = {obj.name: obj for obj in bpy.context.scene.objects if obj.type == 'MESH'}
        picked = subjects(request.get('subject'), mesh_objects)
        if not picked or any(obj.name not in mesh_objects or obj.library or obj.data.library for obj in picked):
            raise ValueError('主体为空、在当前场景之外或依赖外部 Blender 库')
        packed_images(picked)
        # Snapshot evaluated mesh geometry before removing the authoring rig and lights.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pieces = []
        for obj in picked:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
            pieces.append((obj.name, mesh, evaluated.matrix_world.copy()))
        scene = bpy.data.scenes.new('asset')
        bpy.context.window.scene = scene
        picked = []
        for name, mesh, matrix in pieces:
            obj = bpy.data.objects.new(name, mesh)
            scene.collection.objects.link(obj)
            obj.matrix_world = matrix
            picked.append(obj)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=source)
        picked = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if request.get('legacy_axes'):
            for obj in picked:
                obj.data.transform(Matrix.Rotation(-math.pi/2, 4, 'X'))
    if not picked or any(not obj.data.polygons for obj in picked):
        raise ValueError('主体没有可渲染的网格面')
    finish = request.get('finish')
    if finish:
        from scrap_composite.generation.component_shader import assign_slots
        assign_slots(picked, finish)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in picked:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix
        obj.hide_render = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = picked[0]
    if len(picked) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    low, high = bounds(obj)
    span = max(high - low)
    matrix = Matrix.Scale(1/span, 4) @ Matrix.Translation(-(low + high) / 2) @ obj.matrix_world
    obj.data.transform(matrix)
    obj.matrix_world = Matrix.Identity(4)
    obj.name = 'PublishedSubject'
    if finish:
        use_nodes(obj, finish)
    active_scene = bpy.context.scene
    for scene in list(bpy.data.scenes):
        if scene != active_scene:
            bpy.data.scenes.remove(scene)
    for other in list(bpy.data.objects):
        if other != obj:
            bpy.data.objects.remove(other, do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)
    return obj, list(high-low)


def render_board(folder, obj):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.view_transform = 'AgX'
    scene.world = bpy.data.worlds.new('Preview environment')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .8
    light = bpy.data.lights.new('Broad preview light', 'AREA')
    light.energy, light.shape, light.size = 280, 'DISK', 3
    lamp = bpy.data.objects.new(light.name, light)
    scene.collection.objects.link(lamp)
    lamp.location = (2., -2., 3.)
    lamp.rotation_euler = (-lamp.location).to_track_quat('-Z','Y').to_euler()
    for name, eye in (('front', (1.9, 0, 0)), ('quarter', (2., -2., 1.5))):
        make_camera(scene, eye, (0, 0, 0), ortho_scale=1.65)
        render(scene, folder / (name + '.png'))


def main(request):
    folder = Path(request['output'])
    obj, input_extents = import_subject(request)
    bpy.ops.file.pack_all()
    packed_images([obj])
    pack = folder/'model.blend'
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(pack))
    bpy.ops.export_scene.gltf(filepath=str(folder/'preview.glb'), export_format='GLB', use_selection=True)
    bpy.ops.wm.open_mainfile(filepath=str(pack), load_ui=False, use_scripts=False)
    obj = bpy.data.objects['PublishedSubject']
    low, high = bounds(obj)
    meshes = len(obj.data.vertices), len(obj.data.polygons)
    textures = packed_images([obj])
    render_board(folder, obj)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(folder/'preview.glb'))
    preview = [item for item in bpy.context.selected_objects if item.type == 'MESH']
    if not preview:
        raise ValueError('网页预览回读失败')
    limits = [bounds(item) for item in preview]
    proxy_low = [min(pair[0][axis] for pair in limits) for axis in range(3)]
    proxy_high = [max(pair[1][axis] for pair in limits) for axis in range(3)]
    if any(abs(proxy_low[axis]-low[axis]) > 1e-4 or abs(proxy_high[axis]-high[axis]) > 1e-4
           for axis in range(3)):
        raise ValueError('网页预览与 Blender 主体边界不一致')
    atomic_json(folder/'blender.json', {'reopened': True, 'preview_reimported': True,
        'subject': 'PublishedSubject', 'vertices': meshes[0], 'faces': meshes[1],
        'normalized_extents': list(high-low), 'input_extents': input_extents, 'packed_images': textures,
        'blender': bpy.app.version_string, 'procedure': 'evaluated_mesh_and_self_contained_materials'})


if __name__ == '__main__':
    path = Path(sys.argv[sys.argv.index('--') + 1])
    main(json.loads(path.read_text(encoding='utf8')))
