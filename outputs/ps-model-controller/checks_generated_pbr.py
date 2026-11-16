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
from display_batch.cylinder_wear import cylinder_wear_tile, handling_tile
from display_batch.effects import parameters
from display_batch.generated_material import (
    CHANNELS, CORROSION_VERSION, FORM_VERSION, VERSION, WEAR_VERSION, derive_maps, finish_regions, load_pack,
    map_statistics,
)
from display_batch.rendering import RenderCache, part_keys
from display_batch.schema import Action, EffectSettings, apply_actions
from display_batch.surface_context import match_surface_context


class GeneratedFinishTests(unittest.TestCase):
    def test_handling_maps_are_deterministic_distinct_and_sparse(self):
        tile = handling_tile()
        np.testing.assert_array_equal(tile, handling_tile())
        coverage = (tile > 127).mean(axis=(0, 1))
        for actual, (low, high) in zip(coverage, ((.08, .21), (.20, .45), (.09, .25))):
            self.assertTrue(low < actual < high)
        self.assertLess(max(map_statistics(tile)['seam_mean'].values()), .045)
        self.assertLess(np.max(np.abs(np.corrcoef(tile.reshape(-1, 3).T) - np.eye(3))), .35)

    def test_local_surface_match_leaves_holes_background_and_fine_light_variation(self):
        rgb = np.full((64, 64, 3), .40, np.float32)
        rgb[12:52, 12:52] = [.79, .64, .55]
        rgb[12:52:2, 12:52] += .045
        alpha = np.zeros((64, 64), np.float32)
        alpha[12:52, 12:52] = 1.
        alpha[25:29, 25:29] = 0.
        reference = np.tile([116, 112, 111], (160, 1))
        result, summary = match_surface_context(rgb, alpha, reference, {'basis': 'nearby_same_support_and_depth'})
        np.testing.assert_array_equal(result[alpha == 0], rgb[alpha == 0])
        self.assertEqual(summary['status'], 'applied')
        self.assertLess(summary['luma_gap_after_lab'], summary['luma_gap_before_lab'])
        self.assertGreater(np.linalg.norm(result[20, 20] - result[21, 20]), .05)
        self.assertIs(match_surface_context(rgb, alpha, reference, {'basis': 'bounded_neighbourhood_depth_unknown'})[0], rgb)
        stronger, details = match_surface_context(
            rgb, alpha, reference, {'basis': 'nearby_same_support_and_depth'}, deep_grime=True)
        self.assertLess(details['luma_gap_after_lab'], summary['luma_gap_after_lab'])
        np.testing.assert_array_equal(stronger[alpha == 0], rgb[alpha == 0])
        caked, details_caked = match_surface_context(
            rgb, alpha, reference, {'basis': 'nearby_same_support_and_depth'}, caked_dirt=True)
        np.testing.assert_array_equal(caked[alpha == 0], rgb[alpha == 0])
        self.assertLess(details_caked['luma_gap_after_lab'], details['luma_gap_after_lab'])

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

    def test_cylinder_strokes_are_dense_with_local_steel_and_wrapped_angle(self):
        tile = cylinder_wear_tile()
        np.testing.assert_array_equal(tile, cylinder_wear_tile())
        self.assertTrue(np.all(tile[..., 1] <= tile[..., 0]))
        coverage = (tile > 38).mean(axis=(0, 1))
        for actual, (low, high) in zip(coverage, ((.08, .18), (.02, .07), (.03, .11))):
            self.assertTrue(low < actual < high)
        values = tile.astype(float)
        seam = np.abs(values[0] - values[-1]).mean(axis=0)
        interior = np.abs(np.diff(values, axis=0)).mean(axis=1)
        self.assertTrue(np.all(seam < np.percentile(interior, 95, axis=0)))

    def test_hydraulic_tile_increases_each_channel_with_steel_inside_scrapes(self):
        original = cylinder_wear_tile()
        dense = cylinder_wear_tile(dense=True)
        np.testing.assert_array_equal(dense, cylinder_wear_tile(dense=True))
        self.assertTrue(np.all((dense > 38).mean(axis=(0, 1)) > (original > 38).mean(axis=(0, 1))))
        self.assertTrue(np.all(dense[..., 1] <= dense[..., 0]))

    def test_triple_hydraulic_wear_preserves_old_marks_and_stays_patchy(self):
        base = cylinder_wear_tile(dense=True)
        heavy = cylinder_wear_tile(dense=True, layers=3)
        np.testing.assert_array_equal(heavy, cylinder_wear_tile(dense=True, layers=3))
        self.assertTrue(np.all(heavy >= base))
        self.assertTrue(np.all(heavy[..., 1] <= heavy[..., 0]))
        coverage = (heavy > 38).mean(axis=(0, 1))
        self.assertTrue(np.all(coverage > 2 * (base > 38).mean(axis=(0, 1))))
        self.assertLess(coverage[0], .65)

    def test_body_wear_excludes_protected_label(self):
        layout = {'material': 'body paint', 'cylinder': 'source_uv', 'protected_uv': [.34, .018, .69, .67]}
        with TemporaryDirectory() as temp:
            record = finish_regions({'mesh_hash': 'mesh'}, {'atlases': {'mesh': layout}}, temp)
            self.assertEqual(set(record['materials']), {'body paint'})
            with Image.open(record['materials']['body paint']['mask']) as mask:
                self.assertEqual(mask.getpixel((128, 160)), (0, 0, 0))
                self.assertEqual(mask.getpixel((64, 160)), (0, 255, 0))

    def test_corrosion_requires_named_ferrous_evidence_beyond_asset_class(self):
        asset = {'mesh_hash': 'mesh', 'semantic': {'material_classes': ['steel']}}
        pack = {'version': CORROSION_VERSION, 'atlases': {}}
        for materials, recognized in ((['paint', 'plate'], False),
                                      (['cast steel', 'rubber seal'], True),
                                      (['steel rod', 'aluminium body'], False)):
            regions = {'materials': dict.fromkeys(materials)}
            with self.subTest(materials=materials), patch(
                    'display_batch.generated_material.prepare_uv_masks', return_value=(regions, True)):
                self.assertIs(finish_regions(asset, pack, HERE)['ferrous_source'], recognized)

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

material = bpy.data.materials.new('hydraulic paint')
material.use_nodes = True
tree = material.node_tree
source = tree.nodes.get('Principled BSDF')
profile = {{'version': 'generated-pbr-v6', 'maps': {{**maps, 'cylinder': maps['paint'], 'hydraulic': maps['paint']}},
           'regions': {{'materials': {{material.name: {{
               'coverage': [0, 1, 0], 'constant': [0, 1, 0], 'authored_atlas': True,
               'cylinder': 'normalized_y', 'barrel': [-.29, .22, 0, -.03, .088]}}}}}}}}
assert finish_surface(material, source, profile, env, 0.)
steel = tree.nodes['Hydraulic exposed steel']
assert steel.inputs['Metallic'].default_value == 1.
assert steel.inputs['Roughness'].is_linked and steel.inputs['Tangent'].is_linked
mix = tree.nodes['Material Output'].inputs['Surface'].links[0].from_node
assert mix.inputs[1].links[0].from_node == source and mix.inputs[2].links[0].from_node == steel
aov = next(node for node in tree.nodes if node.type == 'OUTPUT_AOV' and node.aov_name == 'hydraulic_finish')
pack = aov.inputs['Color'].links[0].from_node
assert pack.inputs['Green'].links[0].from_socket == mix.inputs[0].links[0].from_socket

for name, weights, ferrous_source, authored, oxidizes in (
    ('Cast steel fixture', [1, 0, 0, 0], True, False, True),
    ('Chipped steel paint fixture', [0, 1, 0, 0], True, False, True),
    ('Protected chrome rod', [1, 0, 0, 0], True, False, False),
    ('Aluminium fixture', [1, 0, 0, 0], True, False, False),
    ('Unknown metallic surface', [1, 0, 0, 0], False, False, False),
):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    source = tree.nodes.get('Principled BSDF')
    profile = {{'version': 'generated-pbr-v7', 'corrosion': True, 'maps': {{**maps, 'cylinder': maps['paint']}},
               'regions': {{'ferrous_source': ferrous_source, 'materials': {{name: {{
                   'coverage': weights, 'constant': weights, 'authored_atlas': authored}}}}}}}}
    finish_surface(material, source, profile, env, 0.)
    oxide = tree.nodes.get('Matte iron oxide')
    assert (oxide is not None) == oxidizes
    assert source.inputs['Alpha'].default_value == 1.
    if oxide is not None:
        steel = tree.nodes['Weathered exposed steel']
        assert oxide.inputs['Metallic'].default_value == 0. and steel.inputs['Metallic'].default_value == 1.
        oxide_mix = tree.nodes['Material Output'].inputs['Surface'].links[0].from_node
        assert oxide_mix.inputs[2].links[0].from_node == oxide
        aov = next(node for node in tree.nodes if node.type == 'OUTPUT_AOV' and node.aov_name == 'corrosion_finish')
        assert aov.inputs['Color'].links[0].from_node.inputs['Red'].links[0].from_socket == oxide_mix.inputs[0].links[0].from_socket

material = bpy.data.materials.new('Authored aluminum fixture')
material.use_nodes = True
tree = material.node_tree
source = tree.nodes.get('Principled BSDF')
profile = {{'version': 'generated-pbr-v8', 'corrosion': True, 'maps': {{**maps, 'cylinder': maps['paint']}},
            'regions': {{'materials': {{material.name: {{
                'coverage': [0, 0, 1], 'constant': [0, 0, 1], 'authored_atlas': True}}}}}}}}
assert finish_surface(material, source, profile, env, 0.)
patina = tree.nodes['Local nonferrous patina']
before = tuple(patina.outputs[0].default_value)
count = len(tree.nodes)
assert finish_surface(material, source, profile, {{'dust_color': [.23, .26, .27]}}, 0.)
assert tuple(patina.outputs[0].default_value) != before and len(tree.nodes) == count
aov = next(node for node in tree.nodes if node.type == 'OUTPUT_AOV' and node.aov_name == 'corrosion_finish')
channels = aov.inputs['Color'].links[0].from_node
assert channels.inputs['Blue'].is_linked

for material_name, weights, cylinder, version in (
    ('Protected oxidized aluminium fixture', [0, 0, 1], None, 'generated-pbr-v10'),
    ('Painted bottle fixture', [0, 1, 0], 'source_uv', 'generated-pbr-v10'),
    ('Caked dirt on aluminium fixture', [0, 0, 1], None, 'generated-pbr-v11'),
    ('Caked dirt on bottle fixture', [0, 1, 0], 'source_uv', 'generated-pbr-v11'),
    ('Amber ageing on aluminium fixture', [0, 0, 1], None, 'generated-pbr-v12'),
    ('Bottle excluded from amber ageing fixture', [0, 1, 0], 'source_uv', 'generated-pbr-v12'),
    ('Even dirt base on aluminium fixture', [0, 0, 1], None, 'generated-pbr-v13'),
    ('Bottle excluded from even dirt fixture', [0, 1, 0], 'source_uv', 'generated-pbr-v13'),
    ('Even outer dirt on aluminium fixture', [0, 0, 1], None, 'generated-pbr-v14'),
    ('Bottle excluded from outer dirt fixture', [0, 1, 0], 'source_uv', 'generated-pbr-v14'),
):
    material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    tree = material.node_tree
    shader = tree.nodes.get('Principled BSDF')
    region = {{'coverage': weights, 'constant': weights, 'authored_atlas': True, 'cylinder': cylinder}}
    amber_version = version in ('generated-pbr-v12', 'generated-pbr-v13', 'generated-pbr-v14')
    profile = {{'version': version, 'corrosion': True, 'amber_age': amber_version,
        'maps': {{**maps, 'cylinder': maps['paint'], 'handling': maps['paint']}},
        'regions': {{'materials': {{material_name: region}}}}}}
    assert finish_surface(material, shader, profile, env, 0.)
    oxide = tree.nodes.get('Dull nonferrous oxide islands')
    assert (oxide is not None) == (cylinder is None)
    assert ('Axial stains and sediment' in tree.nodes) == (cylinder == 'source_uv')
    grime = tree.nodes['Ground-in nonmetallic grime']
    assert grime.inputs['Metallic'].default_value == 0.
    caked = version == 'generated-pbr-v11' or amber_version
    assert ('Caked dirt and splatter' in tree.nodes) == caked
    if caked:
        assert grime.inputs['Normal'].links[0].from_node.name == 'Packed dirt relief'
        assert any(n.type == 'RGBTOBW' for n in tree.nodes) == (cylinder == 'source_uv')
    amber = tree.nodes.get('Old amber oil film')
    assert (amber is not None) == (amber_version and cylinder is None)
    if amber:
        assert amber.inputs['Metallic'].default_value == 0.
        assert any(n.type == 'OUTPUT_AOV' and n.aov_name == 'amber_film' for n in tree.nodes)
    base_film = tree.nodes.get('Even mineral dirt film')
    assert (base_film is not None) == (version in ('generated-pbr-v13', 'generated-pbr-v14') and cylinder is None)
    if base_film:
        base_mix = base_film.outputs[0].links[0].to_node
        assert base_mix.inputs[1].links[0].from_node.type == 'MIX_SHADER'
        assert base_mix.outputs[0].links[0].to_node == amber.outputs[0].links[0].to_node
    outer_film = tree.nodes.get('Even outer dirt film')
    assert (outer_film is not None) == (version == 'generated-pbr-v14' and cylinder is None)
    if outer_film:
        outer_mix = outer_film.outputs[0].links[0].to_node
        assert outer_mix.inputs[1].links[0].from_node == grime.outputs[0].links[0].to_node
        assert outer_mix.outputs[0].links[0].to_node.type == 'OUTPUT_MATERIAL'
    if oxide:
        assert oxide.inputs['Metallic'].default_value == 0.
        assert oxide.inputs['Roughness'].is_linked
    assert shader.inputs['Alpha'].default_value == 1.

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
