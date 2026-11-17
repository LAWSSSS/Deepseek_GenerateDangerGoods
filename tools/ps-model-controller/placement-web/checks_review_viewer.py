"""Read-only smoke tests for the local 39-item review viewer."""
import http.client
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import review_feedback
import review_viewer as viewer


class ReviewViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = viewer.load_catalog()
        cls.server = viewer.ReviewServer(('127.0.0.1', 0), viewer.Handler)
        cls.server.catalog = viewer.load_catalog(include_archived=False)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def request(self, path, size=None, host=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        headers = {'Host': host} if host else {}
        conn.request('GET', path, headers=headers)
        response = conn.getresponse()
        body = response.read() if size is None else response.read(size)
        result = response.status, dict(response.getheaders()), body
        conn.close()
        return result

    def test_catalog_lists_active_models_and_hides_storage_tanks(self):
        code, _, data = self.request('/api/items')
        items = json.loads(data)
        self.assertEqual(code, 200)
        expected = [str(x) for x in range(1,19)] + [f'19-{x}' for x in range(1,7)] + [str(x) for x in range(20,40)]
        self.assertEqual([x['id'] for x in items], [k for k in expected if k not in {'5','6','7','38','39'}])
        self.assertEqual(len(self.catalog), 44)
        for tank in ('05','06','07','38','39'):
            self.assertEqual(self.request(f'/api/model/{tank}')[0], 404)
        self.assertTrue(all(x['available'] for x in items))
        self.assertTrue(next(x for x in items if x['id']=='37')['reference_missing'])
        self.assertEqual(sum(x['human_review']=='approved' for x in items), 30)
        self.assertEqual(next(x for x in items if x['id']=='10')['human_review'], 'revision_pending_review')
        self.assertTrue(all(x['quantity']==1 and x['human_review']=='revision_pending_review'
                            for x in items if x['id'].startswith('19-')))
        self.assertFalse(any('_mesh' in x for x in items))

    def test_model_headers_and_image(self):
        for number in (1, 10, 13, 14, '19-1', '19-2', '19-3', '19-4', '19-5', '19-6', 22, 27, 31, 33, 37):
            with self.subTest(number=number):
                code, headers, data = self.request(f'/api/model/{number}', 4)
                self.assertEqual(code, 200)
                self.assertEqual(data, b'glTF')
                self.assertEqual(headers['Content-Type'], 'model/gltf-binary')
        code, _, data = self.request('/api/board/27', 3)
        self.assertEqual((code, data), (200, b'\xff\xd8\xff'))
        code, _, data = self.request('/api/rust/14', 8)
        self.assertEqual((code, data), (200, b'\x89PNG\r\n\x1a\n'))

    def test_revised_models_contain_one_body_and_the_opaque_barrel(self):
        for key, item in self.catalog.items():
            if key not in ('10','13','31') and not key.startswith('19-'):
                continue
            with item['_mesh'].open('rb') as f:
                f.read(12)
                size, _ = struct.unpack('<II', f.read(8))
                doc = json.loads(f.read(size).decode('utf8'))
            names = [node.get('name','') for node in doc.get('nodes',[])]
            if key == '10':
                removed = {'Detached handle bar', 'Handle bar bore liner', 'Handle bar top rim'}
                self.assertFalse(removed.intersection(names))
                self.assertTrue({'Jack body', 'Cast base', 'Chrome ram', 'Pump handle socket',
                                 'Carry handle loop'}.issubset(names))
            elif key == '13':
                self.assertEqual(names.count('Can rear body'),1)
                self.assertFalse(any(n.startswith('Can front ') for n in names))
            elif key.startswith('19-'):
                self.assertEqual(sum(name.endswith(' 筒身') for name in names),1)
            else:
                self.assertIn('barrel_full_opaque',names)
                self.assertNotIn('piston_body',names)
                self.assertFalse(any(n.startswith(('piston_ring_','gland_ring_')) for n in names))
                barrel = next(node for node in doc['nodes'] if node.get('name') == 'barrel_full_opaque')
                for primitive in doc['meshes'][barrel['mesh']]['primitives']:
                    material = doc['materials'][primitive['material']]
                    self.assertEqual(material.get('alphaMode', 'OPAQUE'), 'OPAQUE')
                    pbr = material.get('pbrMetallicRoughness', {})
                    self.assertEqual(pbr.get('baseColorFactor', [1, 1, 1, 1])[3], 1)
                    transmission = material.get('extensions', {}).get('KHR_materials_transmission', {})
                    self.assertEqual(transmission.get('transmissionFactor', 0), 0)

    def test_missing_revision_does_not_reexpose_group_models(self):
        with patch.object(viewer,'load_feedback',return_value={'approved':set(),'requested':{10,13,19,31},'revisions':{},'error':'missing'}):
            current=viewer.load_catalog()
        self.assertFalse(current['10']['available'])
        self.assertFalse(current['13']['available'])
        self.assertFalse(current['19']['available'])
        self.assertFalse(current['31']['available'])

    def test_existing_eight_model_manifest_remains_readable(self):
        review = json.loads(review_feedback.DECISIONS.read_text(encoding='utf8'))
        manifest = json.loads(Path(review['revision_manifest']).read_text(encoding='utf8'))
        review['accepted_numbers'].append(10)
        del review['needs_revision']['10']
        manifest['items'] = [item for item in manifest['items'] if item['id'] != '10']
        with tempfile.TemporaryDirectory() as directory:
            old_manifest = Path(directory) / 'manifest.json'
            old_review = Path(directory) / 'review.json'
            old_manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf8')
            review['revision_manifest'] = str(old_manifest)
            old_review.write_text(json.dumps(review, ensure_ascii=False), encoding='utf8')
            validate = review_feedback.inside_outputs
            def allow_fixture(raw, suffixes):
                return old_manifest if Path(raw) == old_manifest else validate(raw, suffixes)
            with patch.object(review_feedback, 'DECISIONS', old_review), \
                 patch.object(review_feedback, 'inside_outputs', side_effect=allow_fixture):
                prior = review_feedback.load_feedback()
        self.assertIsNone(prior['error'])
        self.assertIn(10, prior['approved'])
        self.assertEqual(prior['requested'], {13, 19, 31})
        self.assertEqual(len(prior['revisions']), 8)

    def test_vendor_and_index(self):
        self.assertEqual(self.request('/')[0], 200)
        self.assertEqual(self.request('/review-viewer.js')[0], 200)
        self.assertEqual(self.request('/vendor/examples/jsm/loaders/GLTFLoader.js', 20)[0], 200)
        self.assertEqual(json.loads(self.request('/api/health')[2])['app'], viewer.IDENTITY)

    def test_reject_unknown_assets_and_external_hosts(self):
        self.assertEqual(self.request('/api/model/40')[0], 404)
        self.assertEqual(self.request('/vendor/../review_viewer.py')[0], 404)
        self.assertEqual(self.request('/api/items', host='example.com')[0], 403)

    def test_a_second_server_cannot_claim_the_port(self):
        with self.assertRaises(OSError):
            viewer.ReviewServer(('127.0.0.1', self.port), viewer.Handler)


if __name__ == '__main__':
    unittest.main()
