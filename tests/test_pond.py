"""The night pond is served by the service itself: the page, its scene, and nothing else under /pond."""
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from penumbra.api import QuietServer, make_handler


class PondTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = QuietServer(("127.0.0.1", 0), make_handler(mock.MagicMock()))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as res:
            return res.status, res.headers.get("Content-Type"), res.read().decode("utf-8")

    def test_the_page_reads_this_services_own_memory(self):
        status, kind, body = self.get("/pond")
        self.assertEqual(status, 200)
        self.assertIn("text/html", kind)
        self.assertIn("fetch('/memory-core')", body)
        self.assertIn("/pond/scene.js", body)
        self.assertEqual(self.get("/pond/")[0], 200)

    def test_the_scene_is_a_module_with_the_petals(self):
        status, kind, body = self.get("/pond/scene.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", kind)
        self.assertIn("export class Pond", body)
        self.assertIn("setMemories", body)

    def test_nothing_else_is_served_from_the_folder(self):
        for path in ("/pond/../api.py", "/pond/index.html/x", "/pond/secret.txt"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path)
            self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
