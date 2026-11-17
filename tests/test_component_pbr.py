"""Guard against paint/rust leakage and changed component library inputs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ps-model-controller'))
from display_batch.common import file_hash
from display_batch.component_pbr import VERSION, print_guard, validate
from display_batch.reference_material import ReferenceInputChanged, freeze_library, verify_snapshot


class ComponentPbrTests(unittest.TestCase):
    def fixture(self, folder):
        packs = {}
        for role in ('steel', 'paint', 'oxide', 'rubber'):
            maps = {}
            for key in ('albedo', 'surface'):
                path = folder / f'{role}-{key}.png'
                Image.new('RGB', (32, 32), (120, 130, 140)).save(path)
                maps[key] = {'path': path.name, 'sha256': file_hash(path)}
            packs[role] = {'role': role, 'maps': maps, 'source': {'url': 'https://example.com/pbr', 'license': 'CC0-1.0'},
                           'tile_m': .1, 'relief_um': 10, 'anchor_linear': [.23]*3, 'albedo_variation': .5,
                           'albedo_median_luma': .23}
        return {'version': VERSION, 'status': 'review', 'packages': packs, 'component_regions': [
            {'parts': ['barrel'], 'material': 'metal and paint atlas', 'package': 'paint',
             'substrate': 'steel', 'oxide': 'oxide', 'paint_loss_um': 24},
            {'parts': ['seal'], 'material': 'metal and paint atlas', 'package': 'rubber'}]}

    def test_rust_cannot_attach_to_rubber_or_ambiguous_paint(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            payload = self.fixture(base)
            payload['component_regions'][1]['oxide'] = 'oxide'
            with self.assertRaisesRegex(ValueError, 'nonferrous or polymer'):
                validate(payload, base)
            payload['component_regions'][1].pop('oxide')
            payload['component_regions'][0].pop('substrate')
            with self.assertRaisesRegex(ValueError, 'nonferrous or polymer'):
                validate(payload, base)

    def test_component_cannot_be_assigned_twice_and_film_is_bounded(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            payload = self.fixture(base)
            payload['component_regions'][1]['parts'] = ['barrel']
            with self.assertRaisesRegex(ValueError, 'assigned once'):
                validate(payload, base)
            payload['component_regions'][1]['parts'] = ['seal']
            payload['component_regions'][0]['paint_loss_um'] = 800
            with self.assertRaisesRegex(ValueError, 'paint_loss_um'):
                validate(payload, base)

    def test_snapshot_detects_changed_rubber_map(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            package = base / 'pack.json'
            package.write_text(json.dumps(self.fixture(base)), encoding='utf8')
            library = base / 'library.json'
            library.write_text(json.dumps({'version': 'reference-material-library-v1', 'assets': {
                'a'*64: {'asset_id': '25', 'reference_path': package.name, 'reference_sha256': file_hash(package)}}}),
                encoding='utf8')
            snapshot = freeze_library(library, {'a'*64: '25'})
            self.assertEqual(len(snapshot['files']), 10)
            Image.new('RGB', (32, 32), (100, 100, 100)).save(base/'rubber-surface.png')
            with self.assertRaisesRegex(ReferenceInputChanged, 'rubber-surface'):
                verify_snapshot(snapshot)

    def test_print_guard_protects_ink_and_neighbor_without_keeping_rectangle(self):
        image = np.full((80, 128, 3), (160, 20, 22), dtype=np.uint8)
        image[32:37, 60:64] = [200, 188, 172]
        mask = print_guard(image, 'white_on_red')
        self.assertEqual(mask[35, 62], 255)
        self.assertEqual(mask[34, 59], 255)
        self.assertEqual(mask[24, 55], 0)
        self.assertEqual(mask[60, 100], 0)

    def test_surface_details_stay_on_their_audited_material_roles(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            payload = self.fixture(base)
            paint, seal = payload['component_regions']
            rod = {'parts': ['rod'], 'material': 'metal and paint atlas', 'package': 'steel',
                   'machining': {'axis': [0, 0, 1], 'anisotropy': .34}}
            payload['component_regions'].append(rod)
            paint['scuff_strength'] = .7
            seal['source_roughness_mix'] = .5
            self.assertEqual(len(validate(payload, base)['component_regions']), 3)
            for region, field, invalid, message in (
                (seal, 'scuff_strength', .7, 'audited paint stack'),
                (paint, 'source_roughness_mix', .5, 'opaque polymers'),
                (paint, 'machining', rod['machining'], 'requires a metal part'),
                (rod, 'machining', {'axis': [0, 0, 0], 'anisotropy': .34}, 'unit vector'),
                (rod, 'machining', {'axis': [0, 0, 1], 'anisotropy': .8}, 'anisotropy'),
            ):
                with self.subTest(field=field, invalid=invalid):
                    original = region.get(field)
                    region[field] = invalid
                    with self.assertRaisesRegex(ValueError, message):
                        validate(payload, base)
                    if original is None:
                        region.pop(field)
                    else:
                        region[field] = original


if __name__ == '__main__':
    unittest.main()
