import json
import os

os.environ.setdefault("PENUMBRA_EMBEDDING", "none")  # unit tests: deterministic, no model load
import shutil
import tempfile
import threading
import unittest
import urllib.request
from datetime import timedelta
from pathlib import Path

from penumbra import files, index
from penumbra.api import make_handler
from penumbra.config import Config
from penumbra.instance import AlreadyRunning
from penumbra.service import Invalid, NotFound, Penombre
from penumbra.text import bigram_terms

CONV = "c-test-1"


def items(*pairs):
    return [{"id": f"i{n}", "role": role, "content": text, "createdAt": f"2026-09-{10 + n:02d}T12:00:00.000Z"} for n, (role, text) in enumerate(pairs)]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="penumbra-test-"))
        self.svc = Penombre(Config(data_dir=self.tmp))

    def tearDown(self):
        self.svc.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def reopen(self):
        self.svc.close()
        self.svc = Penombre(Config(data_dir=self.tmp))


class TextTest(unittest.TestCase):
    def test_cjk_bigrams_and_latin_words(self):
        self.assertEqual(bigram_terms("秋天的雨 Rain2"), ["秋天", "天的", "的雨", "rain2"])
        self.assertEqual(bigram_terms("雨，好"), ["雨", "好"])


class OriginalsTest(Base):
    def test_ingest_is_idempotent_and_immutable(self):
        first = self.svc.ingest_originals("user", CONV, items(("user", "今天下雨了"), ("assistant", "我陪你听雨")))
        self.assertEqual(len(first["added"]), 2)
        again = self.svc.ingest_originals("user", CONV, items(("user", "今天下雨了")))
        self.assertEqual(again["existing"], first["added"][:1])
        changed = self.svc.ingest_originals("user", CONV, [{"id": "i0", "role": "user", "content": "改过的原文"}])
        self.assertEqual(changed["conflicts"], first["added"][:1])
        self.assertEqual(self.svc.get_original(first["added"][0]).content, "今天下雨了")
        # the refused edit never reached the truth file
        lines = (self.tmp / "originals" / "user" / f"{CONV}.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.svc.ingest_originals("user", "../escape", items(("user", "x")))


class InstanceAndRebuildTest(Base):
    """2026-09-25: several processes on one data dir; two startup rebuilds collided on originals.id."""

    def test_second_instance_on_the_same_dir_is_refused_before_touching_it(self):
        self.svc.ingest_originals("user", CONV, items(("user", "今天下雨了")))
        index_before = (self.tmp / "index.sqlite").stat().st_mtime_ns
        with self.assertRaises(AlreadyRunning):
            Penombre(Config(data_dir=self.tmp))
        self.assertEqual((self.tmp / "index.sqlite").stat().st_mtime_ns, index_before)
        self.reopen()  # released on close: the next process opens normally
        self.assertEqual(self.svc.stats()["originals"], 1)

    def test_rebuild_twice_is_idempotent(self):
        self.svc.ingest_originals("user", CONV, items(("user", "今天下雨了"), ("assistant", "我陪你听雨")))
        first = self.svc.rebuild()
        second = self.svc.rebuild()
        strip = lambda r: {k: v for k, v in r.items() if k != "ms"}  # noqa: E731
        self.assertEqual(strip(first), strip(second))
        self.assertEqual(first["originals"], 2)
        self.assertEqual(self.svc.conn.execute("SELECT COUNT(*) FROM originals_fts").fetchone()[0], 2)

    def test_same_original_twice_is_a_no_op_and_other_content_a_conflict(self):
        self.svc.ingest_originals("user", CONV, items(("user", "今天下雨了")))
        original = self.svc.get_original(self._first_id())
        self.assertFalse(index.put_original(self.svc.conn, original))
        self.assertEqual(self.svc.conn.execute("SELECT COUNT(*) FROM originals_fts").fetchone()[0], 1)
        original.sha256 = "0" * 64
        with self.assertRaises(index.OriginalConflict):
            index.put_original(self.svc.conn, original)

    def test_concurrent_rebuilds_on_one_index_do_not_collide(self):
        self.svc.ingest_originals("user", CONV, items(*[("user", f"第{n}句") for n in range(30)]))
        errors: list[BaseException] = []

        def rebuild_a_few_times():
            conn = index.connect(self.tmp / "index.sqlite")
            try:
                for _ in range(5):
                    index.rebuild(conn, self.tmp / "originals", self.tmp)
            except BaseException as error:  # noqa: BLE001 - collected for the assertion
                errors.append(error)
            finally:
                conn.close()

        threads = [threading.Thread(target=rebuild_a_few_times) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.svc.conn.execute("SELECT COUNT(*) FROM originals").fetchone()[0], 30)

    def _first_id(self) -> str:
        return self.svc.conn.execute("SELECT id FROM originals LIMIT 1").fetchone()[0]


class HttpTest(Base):
    def setUp(self):
        super().setUp()
        from http.server import ThreadingHTTPServer

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.svc))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        for attempt in range(3):  # a loopback connection abort (WinError 10053) can happen on a busy Windows machine
            try:
                with self.opener.open(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())
            except (ConnectionError, urllib.error.URLError):
                if attempt == 2:
                    raise

    def test_roundtrip(self):
        self.assertEqual(self.call("GET", "/health"), (200, {"ok": True}))
        status, ingest = self.call("POST", "/originals", {"source": "user", "conversationId": CONV, "items": items(("user", "想去看海"))})
        self.assertEqual((status, len(ingest["added"])), (200, 1))
        status, one = self.call("GET", "/originals/" + ingest["added"][0])
        self.assertEqual((status, one["original"]["content"]), (200, "想去看海"))
        self.assertEqual(self.call("GET", "/originals/o_missing")[0], 404)
        status, stats = self.call("GET", "/stats")
        self.assertEqual((stats["originals"], stats["memory"]["episodes"]), (1, 0))
        # the retired memory paths answer 410, the unified memory answers under /memory-core
        for path in ("/inject", "/inject/confirm", "/recall", "/review/daily", "/memories", "/manual", "/candidates"):
            self.assertEqual(self.call("POST", path, {})[0], 410, path)
        status, out = self.call("POST", "/memory-core/retrieve", {"query": "周末想去看海", "turnId": "t1", "conversationId": CONV, "sessionId": "s1"})
        self.assertEqual((status, out["status"]), (200, "NO_MEMORY_NEEDED"))  # nothing has been verified into memory yet
        self.assertEqual(self.call("POST", "/memory-core/recall", {})[0], 400)


if __name__ == "__main__":
    unittest.main()
