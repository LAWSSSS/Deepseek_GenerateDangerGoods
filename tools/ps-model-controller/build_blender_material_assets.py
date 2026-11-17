"""Create a packed Blender Asset Browser library from downloaded PBR channels."""
import argparse
import json
from pathlib import Path
import sys

import bpy


def image_node(tree, coordinates, paths, channel, location, *, color=False):
    path = Path(paths[channel]).resolve(strict=True)
    node = tree.nodes.new('ShaderNodeTexImage')
    node.label = channel.capitalize()
    node.location = location
    node.image = bpy.data.images.load(str(path), check_existing=True)
    node.image.colorspace_settings.name = 'sRGB' if color else 'Non-Color'
    tree.links.new(coordinates, node.inputs['Vector'])
    return node.outputs['Color']


def add_material(spec):
    material = bpy.data.materials.new(f'{spec["provider"]} / {spec["name"]}')
    material.use_nodes = True
    tree = material.node_tree
    shader = next(node for node in tree.nodes if node.type == 'BSDF_PRINCIPLED')
    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = (-1150, 100)
    paths = spec['channels']
    for i, (channel, socket) in enumerate((('base', 'Base Color'), ('roughness', 'Roughness'),
                                            ('metallic', 'Metallic'))):
        source = image_node(tree, uv.outputs['UV'], paths, channel, (-840, 530 - i * 280),
                            color=(channel == 'base'))
        tree.links.new(source, shader.inputs[socket])
    normal = tree.nodes.new('ShaderNodeNormalMap')
    normal.location = (-355, -470)
    source = image_node(tree, uv.outputs['UV'], paths, 'normal', (-840, -330))
    tree.links.new(source, normal.inputs['Color'])
    tree.links.new(normal.outputs['Normal'], shader.inputs['Normal'])
    material.asset_mark()
    material.asset_data.description = f'CC0 PBR material · {spec["source_url"]}'
    material.asset_data.author = spec['provider']
    material.asset_data.tags.new('Metal')
    material.asset_data.tags.new('CC0')
    material['source_url'] = spec['source_url']
    material['license'] = 'CC0-1.0'
    return material.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    output = args.output.resolve()
    if output.suffix != '.blend' or output.exists():
        raise ValueError('Provide a fresh destination .blend file')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    expected = [add_material(spec) for spec in json.loads(args.spec.read_text(encoding='utf8'))]
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    bpy.ops.wm.open_mainfile(filepath=str(output))
    installed = sorted(material.name for material in bpy.data.materials if material.asset_data)
    if sorted(expected) != installed or any(not image.packed_file for image in bpy.data.images):
        raise ValueError('Packed material library did not reopen with all assets')
    print(json.dumps({'material_assets': installed, 'packed_images': len(bpy.data.images)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
