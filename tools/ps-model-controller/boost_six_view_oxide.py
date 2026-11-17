"""Increase the iron-oxide edges already registered on packed fine-mesh candidates."""
import argparse
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_batch.common import atomic_json, file_hash, signature
from display_batch.scanned_shader import math_node
from refine_six_view_meshes import packed_images, subject_mesh


GAIN_NODE = 'Registered iron oxide edge gain'


def mesh_signature(obj):
    mesh = obj.data
    arrays = []
    for data, prop, width, dtype in (
        (mesh.vertices, 'co', 3, 'f'),
        (mesh.loops, 'vertex_index', 1, 'i'),
        (mesh.polygons, 'loop_total', 1, 'i'),
        (mesh.polygons, 'material_index', 1, 'i'),
        (mesh.corner_normals, 'vector', 3, 'f'),
    ):
        values = np.empty(len(data) * width, dtype=dtype)
        data.foreach_get(prop, values)
        arrays.append(values)
    for layer in mesh.uv_layers:
        values = np.empty(len(mesh.loops) * 2, dtype='f')
        layer.data.foreach_get('uv', values)
        arrays.append(values)
    from hashlib import sha256
    digest = sha256()
    for array in arrays:
        digest.update(array.tobytes())
    return digest.hexdigest()


def boost_material(material, gain):
    tree = material.node_tree
    registered = tree.nodes.get('Registered view surface')
    if registered is None or tree.nodes.get('Component dielectric iron oxide') is None:
        return None
    if GAIN_NODE in tree.nodes:
        raise ValueError('Registered rust was already increased')
    link, = registered.outputs['Color'].links
    if link.to_node.type != 'SEPARATE_COLOR':
        raise ValueError('Registered wear channels are no longer in the expected format')
    oxide = link.to_node.outputs['Red']
    targets = [edge.to_socket for edge in oxide.links]
    if not targets:
        raise ValueError('Registered oxide is disconnected')
    remaining = math_node(tree, 'SUBTRACT', 1., oxide)
    edge = math_node(tree, 'MULTIPLY', oxide, remaining)
    increase = math_node(tree, 'MULTIPLY', edge, gain)
    adjusted = math_node(tree, 'ADD', oxide, increase)
    adjusted.node.name = GAIN_NODE
    for target in targets:
        tree.links.new(adjusted, target)
    # Read the data atlas to measure the actual change with the registered
    # support weight. These are texture samples, not viewed surface area.
    image = registered.image
    if image.colorspace_settings.name != 'Non-Color':
        raise ValueError('Oxide coverage must be read as data')
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(-1, image.channels)
    red, supported = pixels[:, 0], pixels[:, 2] > .15
    before = red[supported]
    after = before + gain * before * (1. - before)
    if len(before) == 0 or np.any(after < before) or np.any(after > 1.):
        raise ValueError('Registered oxide coverage is invalid')
    return {'supported_texels':len(before), 'before_mean':float(before.mean()),
            'after_mean':float(after.mean()), 'before_over_quarter':float((before >= .25).mean()),
            'after_over_quarter':float((after >= .25).mean()),
            'maximum_added_weight':float((after - before).max())}


def build_variant(path, target, mesh_record, gain):
    prior_sha = file_hash(path)
    if prior_sha != mesh_record['sha256'] or not mesh_record['reopened']:
        raise ValueError('Fine-mesh input differs from its validated candidate')
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    if not scene.get('mesh_detail_candidate') or not scene.get('source_mesh_hash'):
        raise ValueError('Input lacks the normalized fine-mesh provenance')
    subject = subject_mesh()
    geometry = mesh_signature(subject)
    images = packed_images()
    gain_for = {}
    for material in subject.data.materials:
        if material and material.use_nodes and material.name not in gain_for:
            evidence = boost_material(material, gain)
            if evidence is not None:
                gain_for[material.name] = evidence
    if not gain_for or not any(row['after_mean'] > row['before_mean'] for row in gain_for.values()):
        raise ValueError('No verifiable increase to the registered iron oxide')
    source_mesh_hash = scene['source_mesh_hash']
    scene['material_signature'] = signature(scene['material_signature'], GAIN_NODE, gain)
    scene['oxide_edge_gain'] = gain
    bpy.context.preferences.filepaths.save_version = 0
    target.parent.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(target), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(target), load_ui=False, use_scripts=False)
    checked = subject_mesh()
    if mesh_signature(checked) != geometry or packed_images() != images:
        raise ValueError('Material adjustment changed geometry, source UVs or existing texture bytes')
    for name in gain_for:
        node = bpy.data.materials[name].node_tree.nodes[GAIN_NODE]
        if node.type != 'MATH' or not node.outputs[0].is_linked:
            raise ValueError('Oxide edge adjustment was disconnected on reopen')
    return {'path':str(target), 'sha256':file_hash(target), 'source_candidate_sha256':prior_sha,
            'source_mesh_hash':source_mesh_hash, 'subject':checked.name, 'geometry_signature':geometry,
            'faces':len(checked.data.polygons), 'oxide_edge_gain':gain, 'materials':gain_for,
            'packed_source_images_verified':len(images), 'reopened':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gain', type=float, default=.24)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not 0 < args.gain <= .30:
        parser.error('gain must be greater than 0 and at most .30')
    baseline, output = args.baseline.resolve(), args.output.resolve()
    records = json.loads((baseline/'mesh-checks.json').read_text(encoding='utf8'))['assets']
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assets = {aid:build_variant(baseline/aid/'candidate.blend', output/aid/'candidate.blend', row, args.gain)
              for aid,row in records.items()}
    atomic_json(output/'asset-candidates.json', {'version':'packed-reference-candidates-v1',
                'assets':assets, 'seconds':round(time.monotonic() - started, 3)})


if __name__ == '__main__':
    main()
