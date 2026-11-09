"""Read-only smoke tests for the local 39-item review viewer."""
import http.client
import json
import threading
import unittest

import review_viewer as viewer


class ReviewViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = viewer.load_catalog()
        cls.server = viewer.ReviewServer(('127.0.0.1', 0), viewer.Handler)
        cls.server.catalog = cls.catalog
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

    def test_catalog_has_the_39_real_candidates(self):
        code, _, data = self.request('/api/items')
        items = json.loads(data)
        self.assertEqual(code, 200)
        self.assertEqual([x['number'] for x in items], list(range(1, 40)))
        self.assertTrue(all(x['available'] for x in items))
        self.assertTrue(items[36]['reference_missing'])
        self.assertFalse(any('_mesh' in x for x in items))

    def test_model_headers_and_image(self):
        for number in (1, 14, 22, 27, 33, 37, 39):
            with self.subTest(number=number):
                code, headers, data = self.request(f'/api/model/{number}', 4)
                self.assertEqual(code, 200)
                self.assertEqual(data, b'glTF')
                self.assertEqual(headers['Content-Type'], 'model/gltf-binary')
        code, _, data = self.request('/api/board/27', 3)
        self.assertEqual((code, data), (200, b'\xff\xd8\xff'))
        code, _, data = self.request('/api/rust/14', 8)
        self.assertEqual((code, data), (200, b'\x89PNG\r\n\x1a\n'))

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
