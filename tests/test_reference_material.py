"""Reference inputs must stay tied to the intended source and texture content."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/ps-model-controller'))
from display_batch.common import file_hash
from display_batch.reference_material import MAPS, lookup_reference, validate_reference
from display_batch.schema import EffectSettings
from display_batch.effects import parameters


class ReferenceMaterialTests(unittest.TestCase):
    def fixture(self, folder):
        maps = {}
        for name in MAPS:
            path = folder / (name + '.png')
            Image.new('RGB' if name == 'albedo' else 'L', (32, 32), 128).save(path)
            maps[name] = {'path': str(path), 'sha256': file_hash(path)}
        return {'version': 'reference-pbr-v1', 'status': 'estimated', 'scale_m': .12, 'maps': maps}

    def test_source_binding_and_texture_bytes_are_checked(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp); entry = folder/'entry.json'
            entry.write_text(json.dumps(self.fixture(folder)), encoding='utf8')
            library = folder/'library.json'
            library.write_text(json.dumps({'version':'reference-material-library-v1','assets':{
                'a'*64: {'asset_id':'25', 'reference_path':'entry.json','reference_sha256':file_hash(entry)}}}), encoding='utf8')
            result = lookup_reference(library, 'a'*64)
            self.assertEqual(result['source_mesh_hash'], 'a'*64)
            self.assertTrue(result['estimated'])
            self.assertEqual(result['maps']['albedo']['color_space'], 'sRGB')
            self.assertEqual(result['maps']['roughness']['color_space'], 'Non-Color')
            with self.assertRaisesRegex(ValueError, 'no binding'): lookup_reference(library, 'b'*64)
            Image.new('L',(32,32),100).save(folder/'roughness.png')
            with self.assertRaisesRegex(ValueError, 'hash changed'): lookup_reference(library, 'a'*64)

    def test_height_and_alignment_do_not_claim_real_displacement(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp); value=self.fixture(folder)
            self.assertEqual(validate_reference(value)['height_encoding'], 'signed_midpoint_0.5')
            with self.assertRaisesRegex(ValueError, 'microrelief'):
                validate_reference({**value,'height_amplitude_m':.001})
            Image.new('L',(16,32),128).save(folder/'height.png')
            value['maps']['height']['sha256']=file_hash(folder/'height.png')
            with self.assertRaisesRegex(ValueError,'aligned'): validate_reference(value)

    def test_new_mode_is_explicit_and_legacy_parameters_unchanged(self):
        self.assertNotIn('reference_material_manifest', parameters(EffectSettings()))
        with self.assertRaisesRegex(ValueError, 'physical_layers'):
            EffectSettings(reference_material_manifest='library.json')
        selected=EffectSettings(surface_finish='physical_layers',surface_finish_manifest='pack.json',
                                reference_material_manifest='library.json', surface_dust=.5)
        self.assertEqual(parameters(selected)['reference_material_manifest'], 'library.json')
        with self.assertRaises(ValueError): selected.model_validate({**selected.model_dump(),'surface_dust':.75})


if __name__ == '__main__': unittest.main()
