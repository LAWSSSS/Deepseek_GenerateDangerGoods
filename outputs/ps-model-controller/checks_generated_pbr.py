"""Material pack integrity, cache scope and actual Blender shader guards."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from display_batch.common import BLENDER, HERE, atomic_json, file_hash
from display_batch.effects import parameters
from display_batch.generated_material import CHANNELS, FORM_VERSION, VERSION, WEAR_VERSION, derive_maps, finish_regions, load_pack, map_statistics
from display_batch.rendering import RenderCache, part_keys
from display_batch.schema import Action, EffectSettings, apply_actions


class GeneratedFinishTests(unittest.TestCase):
    def test_maps_are_distinct_and_wrap(self):
        source = np.random.default_rng(12).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        oxide, paint, dust = (derive_maps(source, name) for name in CHANNELS)
        self.assertFalse(np.array_equal(oxide, paint))
        self.assertFalse(np.array_equal(paint, dust))
        for tile in (oxide, paint, dust):
            stats = map_statistics(tile)
            self.assertLess(max(stats['seam_mean'].values()), .045)
            self.assertLess(stats['half_tile_correlation'], .48)

    def test_wear_map_retains_chip_mask_and_widens_existing_scratches(self):
        source = np.full((512, 512, 3), 145, np.uint8)
        marks = np.zeros(source.shape[:2], np.float32)
        marks[240:242, 235:277] = .8
        with patch('display_batch.generated_material.narrow_marks', return_value=marks):
            old = derive_maps(source, 'paint')
            new = derive_maps(source, 'paint', WEAR_VERSION)
        np.testing.assert_array_equal(new[..., 0], old[..., 0])
        np.testing.assert_array_equal(new[..., 2], old[..., 2])
        self.assertGreater(np.count_nonzero(new[..., 1] > 10), np.count_nonzero(old[..., 1] > 10))
        self.assertLess(np.count_nonzero(new[..., 1] > 10) / (512 * 512), .02)
        self.assertLess(max(map_statistics(new)['seam_mean'].values()), .045)

    def test_pack_integrity_and_atlas_regions(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            texture = directory / 'tile.png'
            Image.new('RGB', (512, 512)).save(texture)
            mesh_hash = 'a' * 64
            manifest = directory / 'pack.json'
            atomic_json(manifest, {'version': VERSION, 'maps': {kind: {'path': texture.name,
                'size': [512, 512], 'channels': list(channels), 'color_space': 'Non-Color',
                'sha256': file_hash(texture)} for kind, channels in CHANNELS.items()},
                'atlases': {mesh_hash: {'material': 'atlas', 'tiles': ['steel', 'paint', 'nonferrous'] + [None]*13}}})
            pack = load_pack(manifest)
            regions = finish_regions({'mesh_hash': mesh_hash}, pack, directory)['materials']['atlas']
            with Image.open(regions['mask']) as mask:
                self.assertEqual(mask.getpixel((96, 224)), (0, 255, 0))
                self.assertEqual(mask.getpixel((160, 224)), (0, 0, 255))
                self.assertEqual(mask.getpixel((224, 224)), (0, 0, 0))
            Image.new('RGB', (512, 512), 'gray').save(texture)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_pack(manifest)

    def test_render_cache_and_receiver_replay(self):
        request = {'mesh_hash': 'mesh', 'matrix': 'pose', 'yaw': 3., 'roughness_delta': 0.,
            'dust_gain': 1., 'dust_control': 1., 'material_classes': ['steel'], 'light_gain': 1.,
            'environment': {'light_color': [1., 1., 1.]}, 'camera': 'camera',
            'plane_normal': [0, 0, 1], 'appearance': 'local', 'windows': {'object': 'obj', 'shadow': 'sh'},
            'generated_finish': {'signature': 'a', 'regions': {'signature': 'region'}}}
        first = part_keys(request, 'CPU', 'fixture')
        changed = {**request, 'generated_finish': {**request['generated_finish'], 'signature': 'b'}}
        second = part_keys(changed, 'CPU', 'fixture')
        self.assertNotEqual(first['object'], second['object'])
        self.assertNotEqual(first['with_shadow'], second['with_shadow'])
        self.assertEqual(first['without_shadow'], second['without_shadow'])
        with TemporaryDirectory() as temp:
            cache = RenderCache(temp, SimpleNamespace(device='CPU'))
            archived = part_keys(request, 'CPU', 'archive', include_detail=False)['without_shadow']
            folder = cache.root / 'without_shadow' / archived
            folder.mkdir(parents=True)
            (folder / 'without-shadow-linear0001.exr').write_bytes(b'fixture')
            marker = {'path': str(folder)}
            atomic_json(folder / 'result.json', marker)
            ref = {'dependencies': 'archive', 'signature': archived, 'standard_profile_without_detail_key': True}
            self.assertEqual(cache.archived_receiver(changed, ref), marker)
            with self.assertRaisesRegex(ValueError, 'geometry_or_light_changed'):
                cache.archived_receiver({**request, 'light_gain': .8}, ref)

    def test_weld_tile_retains_paint_and_separate_geometry_mask(self):
        atlas = {'material': 'cast', 'tiles': [None, 'paint', 'weld'] + [None] * 13}
        with TemporaryDirectory() as temp:
            record = finish_regions({'mesh_hash': 'mesh'}, {'version': FORM_VERSION, 'atlases': {'mesh': atlas}}, temp)
            entry = record['materials']['cast']
            self.assertEqual(entry['weld_tiles'], [2])
            with Image.open(entry['mask']) as mask:
                self.assertEqual(mask.getpixel((96, 224)), (0, 255, 0, 0))
                self.assertEqual(mask.getpixel((160, 224)), (0, 255, 0, 255))
                self.assertEqual(mask.getpixel((224, 224)), (0, 0, 0, 0))

    def test_review_ignores_presentation_metadata_but_requires_evidence(self):
        from validate_generated_pbr import BlindPair
        scores = dict.fromkeys(('support', 'occlusion', 'lighting', 'edge', 'texture', 'material', 'structure'), 3)
        brief = {'trace': 'visible', 'scores': scores, 'findings': [{
            'factor': 'wear_detail', 'region': [.2, .2, .3, .3], 'severity': 'minor',
            'visibility': 'visible', 'confidence': .8, 'direction': 'unknown', 'cue': '斑驳', 'reference': None}]}
        payload = {'type': 'object', 'a': {**brief, 'findings_order': []}, 'b': brief,
                   'preferred': 'tie', 'reason': '材质接近'}
        reply = BlindPair.model_validate_json(json.dumps(payload))
        self.assertEqual(reply.preferred, 'tie')
        self.assertNotIn('type', reply.model_dump())
        self.assertNotIn('findings_order', reply.a.model_dump())
        with self.assertRaises(ValueError):
            BlindPair.model_validate_json(json.dumps({**payload, 'b': {'trace': 'none'}}))

    def test_mode_and_powder_bounds(self):
        self.assertNotIn('surface_finish', parameters(EffectSettings()))
        inputs = {'surface_finish': 'generated_pbr', 'surface_finish_manifest': 'pack.json'}
        params = parameters(EffectSettings(**inputs))
        self.assertEqual(params['surface_finish_manifest'], 'pack.json')
        for extra in ({'surface_dust': .51}, {'rust_spots': .1}, {'surface_finish_manifest': None}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                EffectSettings(**{**inputs, **extra})
        with self.assertRaises(ValueError):
            apply_actions(params, [Action(kind='surface_dust', value='0.75')], ())

    def test_blender_material_graph(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            texture = folder / 'tile.png'
            Image.new('RGB', (32, 32), (64, 150, 100)).save(texture)
            script = folder / 'shader.py'
            script.write_text(f"""
import sys
sys.path.insert(0, {str(HERE)!r})
import bpy
from display_batch.generated_shader import finish_surface, shape_weld_beads

env = {{'dust_color': [.16, .14, .13]}}
maps = {{kind: {{'path': {str(texture)!r}}} for kind in ('oxide', 'paint', 'dust')}}
for name, weights, strength, expected in (
    ('steel', [1, 0, 0, 0], .5, True), ('paint', [0, 1, 0, 0], .5, True),
    ('paint weathered', [0, 1, 0, 0], .5, True),
    ('zero', [1, 0, 0, 0], 0., True), ('chrome rod', [1, 0, 0, 0], .5, False),
    ('label', [1, 0, 0, 0], .5, False), ('polymer', [0, 0, 0, 1], .5, False),
    ('translucent', [1, 0, 0, 0], .5, False),
):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    source = tree.nodes.get('Principled BSDF')
    norm = tree.nodes.new('ShaderNodeNormalMap')
    tree.links.new(norm.outputs[0], source.inputs['Normal'])
    if name == 'translucent':
        source.inputs['Alpha'].default_value = .5
    alpha = source.inputs['Alpha'].default_value
    profile = {{'maps': maps, 'regions': {{'materials': {{name: {{'coverage': weights, 'constant': weights}}}}}}}}
    if name == 'paint weathered':
        profile['version'] = 'generated-pbr-v2'
    assert finish_surface(material, source, profile, env, strength) == expected
    assert source.inputs['Alpha'].default_value == alpha and not source.inputs['Alpha'].is_linked
    layer = tree.nodes.get('Generated dielectric dust')
    if expected:
        bump = tree.nodes['Generated shallow surface relief']
        assert source.inputs['Normal'].links[0].from_node == bump
        assert bump.inputs['Normal'].links[0].from_node == norm
        assert not tree.nodes['Material Output'].inputs['Displacement'].is_linked
        oxidation = tree.nodes['Generated oxidation']
        shift = oxidation.inputs['Vector'].links[0].from_node
        scale = shift.inputs[0].links[0].from_node
        assert scale.operation == 'SCALE' and scale.inputs[0].links[0].from_socket.name == 'Object'
        assert (layer is not None) == (strength > 0)
        assert ('Generated paint weathering' in tree.nodes) == (name == 'paint weathered')
        if name == 'paint weathered':
            def texture_sources(socket):
                sources = set()
                if socket.is_linked:
                    node = socket.links[0].from_node
                    if node.type == 'TEX_IMAGE':
                        sources.add(node.name)
                    else:
                        for input_socket in node.inputs:
                            sources.update(texture_sources(input_socket))
                return sources
            assert 'Generated paint weathering' in texture_sources(source.inputs['Base Color'])
            assert 'Generated paint weathering' in texture_sources(source.inputs['Roughness'])
            assert 'Generated paint weathering' not in texture_sources(source.inputs['Metallic'])
        if layer:
            assert layer.inputs['Metallic'].default_value == 0.
            coverage = tree.nodes['Generated dust coverage']
            # Top-level coverage is strength times selected region and the
            # independent deposition field. At constant weight the region is
            # exactly one for steel and one half for paint.
            scale = coverage.inputs[0].links[0].from_node
            assert scale.inputs[0].default_value == strength
            region = scale.inputs[1].links[0].from_node
            paint = region.inputs[1].links[0].from_node
            assert paint.inputs[0].default_value == weights[1] and paint.inputs[1].default_value == .5
            assert region.inputs[0].links[0].from_node.operation == 'ADD'
            count = len(tree.nodes)
            assert finish_surface(material, source, profile, env, strength)
            assert len(tree.nodes) == count
    else:
        assert layer is None and source.inputs['Normal'].links[0].from_node == norm

# Corner 2 belongs to both faces. Only vertices unique to the weld face move.
mesh = bpy.data.meshes.new('weld guard')
mesh.from_pydata([(.17, .16, 0), (.52, .11, 0), (.2, .77, 0), (.80, .60, 0), (.90, .50, 0)],
                 [], [(0, 1, 2), (2, 3, 4)])
mesh.update()
uv = mesh.uv_layers.new().uv
for i, corner in enumerate(uv):
    corner.vector = (.38, .64) if i < 3 else (.1, .1)
before = [v.co.copy() for v in mesh.vertices]
topology = [tuple(face.vertices) for face in mesh.polygons]
record = shape_weld_beads(mesh, {{'regions': {{'materials': {{'cast': {{'weld_tiles': [9]}}}}}}}})
assert record['vertices'] == 2 and 0 < record['max_offset'] <= .00078
assert [tuple(face.vertices) for face in mesh.polygons] == topology
assert all(mesh.vertices[i].co == before[i] for i in (2, 3, 4))
assert any(mesh.vertices[i].co != before[i] for i in (0, 1))
print('GENERATED_SHADER_CHECKS_OK')
""", encoding='utf8')
            completed = subprocess.run([str(BLENDER), '--background', '--factory-startup',
                '--python-exit-code', '1', '--python', str(script)], capture_output=True, text=True,
                encoding='utf8', timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('GENERATED_SHADER_CHECKS_OK', completed.stdout)


if __name__ == '__main__':
    unittest.main()
