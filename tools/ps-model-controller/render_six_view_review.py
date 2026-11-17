"""Render packed component candidates from the registered sheet directions in Blender."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix, Vector


def cameras(registration):
    primary = next(row for row in registration['surfaces'] if row['mode'] == 'cylinder')
    around, axial, front = (Vector(primary['axes'][key]) for key in ('u','v','n'))
    if primary['direction'] == 'horizontal':
        return {'Front':(front,-around), 'Back':(-front,-around),
                'Left':(axial,-around), 'Right':(-axial,-around),
                'Top':(-around,-front), 'Bottom':(around,front)}
    left = next(view['angle'] for view in primary['views'] if view['view'] == 'Left')
    side = around if left > 0 else -around
    return {'Front':(front,-axial), 'Back':(-front,-axial), 'Left':(side,-axial),
            'Right':(-side,-axial), 'Top':(-axial,-front), 'Bottom':(axial,front)}


def rig(scene, subject, outward, up):
    right = up.cross(outward)
    points = [vertex.co for vertex in subject.data.vertices]
    axes = (right, up, outward)
    bounds = [[min(p.dot(a) for p in points),max(p.dot(a) for p in points)] for a in axes]
    center = sum((a * ((lo+hi)*.5) for a,(lo,hi) in zip(axes,bounds)), Vector((0,0,0)))
    extent = max(hi-lo for lo,hi in bounds[:2])
    camera = scene.camera
    camera.location = center + 5. * outward
    camera.rotation_euler = Matrix(axes).transposed().to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 1.16 * extent
    for light in list(scene.objects):
        if light.type == 'LIGHT':
            bpy.data.objects.remove(light, do_unlink=True)
    for name, position, energy, scale in (
        ('Wide key', 2.6*outward + 2.1*up - 1.45*right, 250., 3.6),
        ('Wide fill', 2.*outward + .65*up + 1.6*right, 78., 3.2),
    ):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.shape, data.size = energy, 'DISK', scale
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = center + position
        light.rotation_euler = (center-light.location).to_track_quat('-Z','Y').to_euler()


def render(candidate, views, output, resolution, expected_axis):
    bpy.ops.wm.open_mainfile(filepath=str(candidate))
    scene = bpy.context.scene
    objects = [obj for obj in scene.objects if obj.type == 'MESH']
    subject, = [obj for obj in objects if not any(m and m.name=='neutral receiver' for m in obj.data.materials)]
    subject.matrix_world = Matrix.Identity(4)
    length_coordinates = [vertex.co.dot(expected_axis) for vertex in subject.data.vertices]
    if max(length_coordinates) - min(length_coordinates) < .8:
        raise ValueError('Reference longitudinal axis disagrees with the canonical mesh')
    for obj in objects:
        if obj != subject:
            obj.hide_render = True
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'OPTIX'
    preferences.refresh_devices()
    for device in preferences.devices:
        device.use = device.type == 'OPTIX'
    if not any(device.use for device in preferences.devices):
        raise RuntimeError('No OptiX device for review render')
    scene.cycles.device = 'GPU'
    scene.cycles.samples, scene.cycles.use_denoising = 40, True
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = 'PNG', 'RGBA'
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True
    scene.use_nodes = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = -.10
    ambient = bpy.data.worlds.new('Neutral view studio')
    ambient.use_nodes = True
    background = ambient.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (.41,.41,.41,1.)
    background.inputs['Strength'].default_value = .59
    scene.world = ambient
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name,(outward,up) in views.items():
        rig(scene, subject, outward, up)
        path = output / f'{name.lower()}.png'
        if path.exists():
            raise FileExistsError(path)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, choices=(512,1024,2048), default=1024)
    parser.add_argument('--views', nargs='+', choices=('Front','Back','Left','Right','Top','Bottom'))
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    assets = json.loads(args.registration.read_text(encoding='utf8'))['assets']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = []
    for aid, registration in assets.items():
        views = cameras(registration)
        primary = next(row for row in registration['surfaces'] if row['mode'] == 'cylinder')
        expected_axis = Vector(primary['axes']['v'])
        pitch = math.radians(registration.get('review_pitch_degrees', 0.))
        for label in ('Front','Back','Left','Right'):
            outward, up = views[label]
            views[label] = (math.cos(pitch)*outward + math.sin(pitch)*up,
                            math.cos(pitch)*up - math.sin(pitch)*outward)
        if args.baseline:
            results.extend(render((args.baseline/aid/'candidate.blend').resolve(),
                                  {'Front':views['Front']}, output/aid/'baseline', args.resolution, expected_axis))
        if args.views:
            views = {name:views[name] for name in args.views}
        results.extend(render((args.candidates/aid/'candidate.blend').resolve(), views, output/aid, args.resolution,
                              expected_axis))
    (output/'render-log.json').write_text(json.dumps({'files':results, 'resolution':args.resolution,
        'renders':len(results), 'seconds':round(time.monotonic()-started,3),'api_calls':0}, ensure_ascii=False, indent=2),
        encoding='utf8')


if __name__ == '__main__':
    main()
