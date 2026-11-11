"""Blender regression for scalar roughness, material reuse, and rotating reflections."""
import json
import math
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from display_batch.scrap_shader import configure_scrap_materials


def sphere(name,metal):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64,ring_count=32)
    obj = bpy.context.object
    obj.name = name
    for poly in obj.data.polygons:
        poly.use_smooth = True
    original = bpy.data.materials.new(name)
    original.use_nodes = True
    shader = original.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Metallic'].default_value = metal
    shader.inputs['Roughness'].default_value = .23 if metal else .53
    shader.inputs['Base Color'].default_value = (.42,.41,.40,1.)
    obj.data.materials.append(original)
    return obj,original


def point_at(obj,position):
    obj.location = position
    obj.rotation_euler = (-obj.location).to_track_quat('-Z','Y').to_euler()


def render(scene,path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path))
    w,h = image.size
    pixels = np.float32(image.pixels[:]).reshape(h,w,4)
    bpy.data.images.remove(image)
    return pixels


def highlight_center(pixels):
    gray = pixels[...,:3].mean(axis=2)
    mask = pixels[...,3]>.97
    high = np.maximum(gray-np.percentile(gray[mask],86),0)*mask
    return float((high*np.arange(gray.shape[1])[None,:]).sum()/high.sum())


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 2
    scene.render.resolution_x = scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.world = bpy.data.worlds.new('Uniform ambient')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get('Background').inputs['Strength'].default_value = .12
    camera = bpy.data.objects.new('Camera',bpy.data.cameras.new('Camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 2.7
    camera_position = Vector((0,-4,1.7))
    point_at(camera,camera_position)
    key = bpy.data.objects.new('Key',bpy.data.lights.new('Key','AREA'))
    scene.collection.objects.link(key)
    key.data.energy = 460
    key.data.size = 1.4
    left = Vector((-3,-3,4))
    point_at(key,left)
    metal,source = sphere('Steel fixture',1.)
    polymer,plastic_source = sphere('Polymer fixture',0.)
    polymer.hide_render = True
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        texture = bpy.data.images.new('Periodic sample',width=64,height=64,alpha=False)
        y,x = np.mgrid[0:64,0:64]
        rgb = np.ones((64,64,4),np.float32)
        rgb[...,:3] = np.stack([.5+.33*np.sin((x+y)/64*math.tau),
                               .5+.30*np.sin(x/16*math.tau),
                               .5+.36*np.cos((2*y+x)/64*math.tau)],axis=2)
        texture.pixels.foreach_set(rgb.ravel())
        texture.filepath_raw = str(out/'texture.png')
        texture.file_format = 'PNG'
        texture.save()
        profile = {'variant_signature':'test-steel','texture':texture.filepath_raw,
                   'status':'estimated','detail_weight':.18,'metal_roughness':.49,'coating_strength':.52,
                   'palette':{'metal':[.43,.43,.43],'paint':[.25]*3,'rust':[.18]*3,'dust':[.19]*3},
                   'materials':{source.name:{'constant':[1,0,0,0]},plastic_source.name:{'constant':[0,0,0,1]}}}
        request = {'environment':{'scrap_surface':profile},'roughness_delta':0.,
                   'surface_texture_scale':.82,'dust_control':1.,'dust_gain':1.}
        state = {}
        configure_scrap_materials(metal,request,state)
        first = render(scene,out/'left.exr')
        point_at(key,Vector((3,-3,4)))
        right = render(scene,out/'right.exr')
        displacement = highlight_center(right)-highlight_center(first)
        assert displacement>8,displacement

        point_at(key,left)
        rotation = Matrix.Rotation(math.pi/2,3,'Z')
        metal.rotation_euler.z = math.pi/2
        spun = render(scene,out/'spun.exr')
        point_at(camera,rotation@camera_position)
        point_at(key,rotation@left)
        together = render(scene,out/'rotated-together.exr')
        core = first[...,3]>.98
        spin_difference = float(np.mean(abs(first[core,:3]-spun[core,:3])))
        joint_difference = float(np.mean(abs(first[core,:3]-together[core,:3])))
        assert spin_difference>joint_difference+.001,(spin_difference,joint_difference)
        np.testing.assert_array_equal(first[...,3]>.5,right[...,3]>.5)

        # Swap variants and revisit the first object. Each copy has one powder layer.
        configure_scrap_materials(polymer,{**request,'environment':{'scrap_surface':{
            **profile,'variant_signature':'test-polymer'}}},state)
        configure_scrap_materials(metal,{**request,'dust_control':1.3},state)
        for obj,original,metallicity in [(metal,source,1.),(polymer,plastic_source,0.)]:
            assert obj.data.materials[0]==original
            original_shader = original.node_tree.nodes.get('Principled BSDF')
            assert original_shader.inputs['Metallic'].default_value==metallicity
            assert not original_shader.inputs['Roughness'].is_linked
            nodes = obj.material_slots[0].material.node_tree.nodes
            assert sum(n.type=='MIX_SHADER' for n in nodes)==1
            powder = next(n for n in nodes if n.label=='Deposited dust')
            assert powder.inputs['Metallic'].default_value==0.
            assert powder.inputs['Emission Strength'].default_value==0.
        print('SCRAP_SHADER_CHECK '+json.dumps({'highlight_shift_px':displacement,
            'texture_rotation_difference':spin_difference,'camera_rotated_with_object_difference':joint_difference,
            'source_materials_preserved':True,'powder_layers_per_material':1}),flush=True)


if __name__=='__main__':
    main()
