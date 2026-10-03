"""Preference Learning backend: verbatim documents User publishes - session_pinned (the Session Preference Pack) or
retrieval (locally cut chunks) - plus legacy segmented documents that keep working. Synthetic data only; no model."""
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from penumbra import files
from penumbra.api import make_handler
from penumbra.config import Config
from penumbra.preferences import PINNED_BUDGET_TOKENS, estimate_tokens, local_chunks, split_units
from penumbra.service import Invalid, Penombre
from tests.support import FAKE_GATES, NOW, ConceptEmbeddings, recall

STAMP = NOW.isoformat()

DOC = """# 我们的相处说明（合成测试文档）

聊天的时候我喜欢慢一点，先听我把话说完，再给建议。晚上十一点以后少发长消息。
提醒我做事之前要先问我当天的安排，我说“等一下”就先放着，这条没有例外。
忙完以后陪我聊十分钟别的，帮我把明天的待办理一遍，这是我的收尾习惯。

边界：不要在别人面前开关于我工作的玩笑，也不要替我做决定。
吵架的时候先听我说完，别急着讲道理，等我冷静下来再说；冷战不能超过一个晚上。

另外，我对猫毛过敏，周末喜欢去看展。
"""


class NoModelWorker:
    """Stands in for DeepSeek: any use by the preference path fails the test."""
    provider, model = "none", "none"

    def available(self):
        return False

    def __getattr__(self, name):
        raise AssertionError(f"preference path called the worker ({name})")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="penumbra-pref-"))
        self.open()

    def open(self):
        self.svc = Penombre(Config(data_dir=self.tmp), embeddings=ConceptEmbeddings(), retrieval=FAKE_GATES)
        self.svc._now = lambda: NOW
        self.prefs = self.svc.preferences

    def tearDown(self):
        self.svc.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def draft(self, content=DOC, mode="session_pinned", title="相处说明"):
        return self.prefs.create_document({"owner": "user", "title": title, "originalContent": content, "mode": mode,
                                           "initialLabels": ["relationship"]})

    def live(self, doc_id):
        return [c for c in self.prefs.list_chunks(doc_id) if c["status"] != "archived"]

    def recall_ids(self, query):
        self.svc.vectors.wait_idle(20)
        return [r["id"] for r in recall(self.svc, query)["results"]]

    def pack_ids(self):
        return [d["id"] for d in self.prefs.session_pack()["documents"]]


class SessionPinnedTest(Base):
    def test_publish_is_verbatim_without_model_jobs_or_chunks(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):  # 8: no DeepSeek key needed
            self.svc.memory.verifier.client = NoModelWorker()  # 1: never called
            doc = self.draft()
            self.assertEqual((doc["status"], doc["mode"]), ("draft", "session_pinned"))
            self.assertEqual(self.pack_ids(), [])  # a draft never reaches AI
            doc = self.prefs.publish(doc["id"], {})
        self.assertEqual((doc["status"], doc["enabled"], doc["chunkCount"]), ("active", True, 0))
        self.assertEqual(list((self.tmp / "preferences" / "jobs").iterdir()), [])  # no segmentation job
        pack = self.prefs.session_pack()
        self.assertEqual(len(pack["documents"]), 1)
        entry = pack["documents"][0]
        self.assertEqual((entry["content"], entry["sha256"], entry["version"], entry["title"]), (DOC, files.content_hash(DOC), 1, "相处说明"))
        self.assertEqual(entry["estimatedTokens"], estimate_tokens(DOC))
        self.assertEqual((pack["totalTokens"], pack["budgetTokens"], pack["overBudget"]), (entry["estimatedTokens"], PINNED_BUDGET_TOKENS, False))
        # never a retrieval candidate
        self.assertEqual([i for i in self.recall_ids("吵架的时候先听我说完") if i.startswith("pc_")], [])
        self.assertNotIn("content", self.prefs.session_pack(with_content=False)["documents"][0])

    def test_versions_one_in_effect_and_rotation_identity(self):
        v1 = self.prefs.publish(self.draft()["id"], {})
        hash1 = self.prefs.session_pack()["packHash"]
        # 4: a new draft (same lineage) does not replace the published version until it is published
        v2 = self.prefs.new_version(v1["id"], {})
        self.assertEqual((v2["status"], v2["version"], v2["lineageId"], v2["mode"]), ("draft", 2, v1["lineageId"], "session_pinned"))
        self.prefs.patch_document(v2["id"], {"originalContent": DOC.replace("一个晚上", "两个小时")})
        self.prefs.patch_document(v2["id"], {"originalContent": DOC.replace("一个晚上", "三个小时")})  # drafts edit in place
        self.assertEqual(self.prefs.get_document(v2["id"])["version"], 2)
        self.assertEqual((self.pack_ids(), self.prefs.session_pack()["packHash"]), ([v1["id"]], hash1))
        # 5: publishing v2 retires v1; the pack holds only v2 and its identity changed (-> the bridge rotates)
        self.prefs.publish(v2["id"], {})
        pack = self.prefs.session_pack()
        self.assertEqual([d["id"] for d in pack["documents"]], [v2["id"]])
        self.assertIn("三个小时", pack["documents"][0]["content"])
        self.assertNotEqual(pack["packHash"], hash1)
        old = self.prefs.get_document(v1["id"])
        self.assertEqual((old["status"], old["supersededBy"], old["originalContent"]), ("archived", v2["id"], DOC))
        # editing published text makes a new draft version instead of changing what is in effect
        out = self.prefs.patch_document(v2["id"], {"originalContent": "新的文字"})
        self.assertEqual((out["createdVersion"], out["document"]["version"], self.pack_ids()), (True, 3, [v2["id"]]))
        # a title change is part of the pack identity too
        before = self.prefs.session_pack()["packHash"]
        self.prefs.patch_document(v2["id"], {"title": "相处说明（新）"})
        self.assertNotEqual(self.prefs.session_pack()["packHash"], before)

    def test_disable_unpublish_archive_leave_the_pack(self):
        a = self.prefs.publish(self.draft(title="甲")["id"], {})
        b = self.prefs.publish(self.draft(content="睡前说晚安。", title="乙")["id"], {})
        self.assertEqual(self.pack_ids(), sorted([a["id"], b["id"]]))  # stable order: publish time (frozen here), then id
        h = self.prefs.session_pack()["packHash"]
        self.prefs.set_enabled(a["id"], {}, False)  # 6
        self.assertEqual(self.pack_ids(), [b["id"]])
        self.assertNotEqual(self.prefs.session_pack()["packHash"], h)
        self.prefs.set_enabled(a["id"], {}, True)
        self.assertEqual(self.prefs.session_pack()["packHash"], h)
        self.prefs.unpublish(b["id"], {})
        self.assertEqual((self.prefs.get_document(b["id"])["status"], self.pack_ids()), ("draft", [a["id"]]))
        self.prefs.archive_document(a["id"], {})
        self.assertEqual(self.pack_ids(), [])

    def test_budget_is_enforced_not_truncated(self):
        big = "我" * (PINNED_BUDGET_TOKENS - 10)
        self.prefs.publish(self.draft(content=big, title="大")["id"], {})
        small = self.draft(content="这一份放不下了，还有二十个字左右的内容在这里。", title="小")
        with self.assertRaises(Invalid) as err:
            self.prefs.publish(small["id"], {})
        self.assertIn("预算", str(err.exception))
        self.assertEqual(self.prefs.get_document(small["id"])["status"], "draft")
        self.assertEqual(self.prefs.get_document(small["id"])["originalContent"], "这一份放不下了，还有二十个字左右的内容在这里。")
        # retrieval mode is not limited by the pinned budget
        self.assertEqual(self.prefs.publish(small["id"], {"mode": "retrieval"})["mode"], "retrieval")

    def test_mode_switch_moves_between_pack_and_index(self):
        doc = self.prefs.publish(self.draft(mode="retrieval")["id"], {})
        chunks = self.live(doc["id"])
        conflict = next(c for c in chunks if "吵架" in c["text"])
        self.assertIn(conflict["id"], self.recall_ids("我们闹别扭了怎么哄"))
        self.assertEqual(self.pack_ids(), [])
        self.prefs.patch_document(doc["id"], {"mode": "session_pinned"})
        self.assertEqual(self.pack_ids(), [doc["id"]])
        self.assertNotIn(conflict["id"], self.recall_ids("我们闹别扭了怎么哄"))
        self.prefs.patch_document(doc["id"], {"mode": "retrieval"})
        self.assertEqual(self.pack_ids(), [])
        self.assertIn(conflict["id"], self.recall_ids("我们闹别扭了怎么哄"))


class RetrievalModeTest(Base):
    def test_local_chunks_are_exact_and_recallable(self):
        doc = self.prefs.publish(self.draft(mode="retrieval")["id"], {})
        chunks = self.live(doc["id"])
        self.assertEqual((doc["segmentationVersion"], doc["activeChunkCount"]), ("local-paragraph-v1", len(chunks)))
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual((c["text"], c["contentHash"], c["origin"], c["status"]),
                             (DOC[c["startOffset"]:c["endOffset"]], files.content_hash(c["text"]), "local", "active"))
        self.assertEqual("".join(c["text"] for c in chunks), DOC)
        conflict = next(c for c in chunks if "吵架" in c["text"])
        top = recall(self.svc, "我们闹别扭了怎么哄")["results"][0]
        self.assertEqual((top["id"], top["kind"], top["text"]), (conflict["id"], "PREFERENCE", conflict["text"]))
        self.assertNotIn("过敏", top["text"])
        # restart: chunks come back from the files
        self.svc.close()
        self.open()
        self.assertIn(conflict["id"], self.recall_ids("我们闹别扭了怎么哄"))
        self.prefs.set_enabled(doc["id"], {}, False)
        self.assertNotIn(conflict["id"], self.recall_ids("我们闹别扭了怎么哄"))

    def test_split_merge_boundary_keep_offsets_text_hash(self):
        doc = self.prefs.publish(self.draft(mode="retrieval")["id"], {})
        chunk = next(c for c in self.live(doc["id"]) if "吵架" in c["text"])
        cut = chunk["startOffset"] + chunk["text"].index("冷战")
        a, b = self.prefs.split_chunk(chunk["id"], {"offset": cut})
        self.assertEqual((a["text"] + b["text"], a["endOffset"], b["startOffset"]), (chunk["text"], cut, cut))
        merged = self.prefs.merge_chunks({"chunkIds": [a["id"], b["id"]]})
        self.assertEqual((merged["text"], merged["contentHash"]), (chunk["text"], chunk["contentHash"]))
        live = self.live(doc["id"])
        i = next(n for n, c in enumerate(live) if c["id"] == merged["id"])
        prev = live[i - 1]
        moved = self.prefs.patch_chunk(prev["id"], {"endOffset": merged["startOffset"] + 3})
        after = next(c for c in self.live(doc["id"]) if c["id"] == merged["id"])
        self.assertEqual(moved["text"] + after["text"], prev["text"] + merged["text"])
        self.assertEqual("".join(c["text"] for c in self.live(doc["id"])), DOC)
        with self.assertRaises(Invalid):
            self.prefs.patch_chunk(merged["id"], {"text": "改写"})

    def test_units_and_local_chunks_tile_the_text(self):
        for text in (DOC, "一句话", "没有标点的长文本" * 200, "a\n\n\nb", "  开头空白。结尾没有标点", "短。\n\n" * 50):
            for ranges in (split_units(text), local_chunks(text)):
                self.assertEqual("".join(text[s:e] for s, e in ranges), text)
                self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
        self.assertTrue(all(e - s <= 900 for s, e in local_chunks("这是一句话。" * 400)))


class LegacyDataTest(Base):
    """7: documents chunked by the retired DeepSeek segmentation keep working."""

    def legacy(self, status, chunk_status):
        doc_id = files.new_id("pd")
        units = [(0, DOC.index("边界")), (DOC.index("边界"), len(DOC))]
        doc = {"id": doc_id, "owner": "user", "title": "旧文档", "originalContent": DOC, "sha256": files.content_hash(DOC),
               "status": status, "createdAt": STAMP, "updatedAt": STAMP, "segmentationVersion": "pref-seg-v1", "chunkCount": 2,
               "activeChunkCount": 2 if status == "active" else 0, "initialLabels": [], "processing": {"jobId": "sj_x"},
               "version": 1, "previousVersionId": None, "lineageId": doc_id, "supersededBy": None, "publishedAt": STAMP if status == "active" else None,
               "archivedAt": None}
        chunks = [{"id": files.new_id("pc"), "documentId": doc_id, "startOffset": s, "endOffset": e, "text": DOC[s:e], "labels": ["conflict"],
                   "freeTags": [], "entities": [], "status": chunk_status, "contentHash": files.content_hash(DOC[s:e]), "documentSha256": doc["sha256"],
                   "origin": "segmentation", "segmentationJobId": "sj_x", "replaces": [], "replacedBy": [], "createdAt": STAMP, "updatedAt": STAMP}
                  for s, e in units]
        files.write_json_atomic(self.tmp / "preferences" / "documents" / f"{doc_id}.json", doc)
        files.write_json_atomic(self.tmp / "preferences" / "chunks" / f"{doc_id}.json", {"documentId": doc_id, "chunks": chunks})
        return doc, chunks

    def test_legacy_active_and_review_documents(self):
        active, chunks = self.legacy("active", "active")
        review, review_chunks = self.legacy("review", "review")
        self.svc.close()
        (self.tmp / "index.sqlite").unlink(missing_ok=True)  # rebuilt from the files, as after an upgrade
        self.open()
        self.assertIn(chunks[1]["id"], self.recall_ids("吵架的时候先听我说完"))
        self.assertEqual(self.prefs.list_chunks(active["id"])[1]["text"], chunks[1]["text"])
        self.assertEqual(self.pack_ids(), [])  # a legacy document is retrieval, never pinned
        self.assertNotIn(review_chunks[1]["id"], self.recall_ids("吵架的时候先听我说完"))
        # a legacy review document can still be published (retrieval) - not as Session 常驻
        with self.assertRaises(Invalid):
            self.prefs.publish(review["id"], {"mode": "session_pinned"})
        self.prefs.archive_document(active["id"], {})  # same synthetic text: retrieval would dedupe the two
        self.assertEqual(self.prefs.publish(review["id"], {})["mode"], "retrieval")
        self.assertIn(review_chunks[1]["id"], self.recall_ids("吵架的时候先听我说完"))
        self.assertEqual(self.prefs.list_chunks(review["id"])[1]["origin"], "segmentation")  # reviewed chunks kept as they were

    def test_new_pinned_version_replaces_the_chunked_one(self):
        old, chunks = self.legacy("review", "review")
        draft = self.prefs.new_version(old["id"], {"mode": "session_pinned", "originalContent": DOC + "\n新版补充。"})
        self.assertEqual((draft["version"], draft["lineageId"], draft["mode"]), (2, old["id"], "session_pinned"))
        self.assertEqual(self.prefs.get_document(old["id"])["status"], "review")  # nothing retired before publish
        self.prefs.publish(draft["id"], {})
        retired = self.prefs.get_document(old["id"])
        self.assertEqual((retired["status"], retired["supersededBy"]), ("archived", draft["id"]))
        self.assertTrue(all(c["status"] == "archived" for c in self.prefs.list_chunks(old["id"])))  # kept, not deleted
        self.assertEqual(self.pack_ids(), [draft["id"]])
        self.assertEqual([i for i in self.recall_ids("吵架的时候先听我说完") if i.startswith("pc_")], [])


class PreferenceHttpTest(Base):
    """The routes the Studio calls (bridge prefix /api/memory)."""

    def setUp(self):
        super().setUp()
        from http.server import ThreadingHTTPServer

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.svc))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def call(self, method, path, body=None):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_contract(self):
        s, created = self.call("POST", "/preferences/documents", {"owner": "user", "title": "相处说明", "originalContent": DOC})
        doc_id = created["document"]["id"]
        self.assertEqual((s, created["document"]["status"], created["document"]["mode"]), (200, "draft", "session_pinned"))
        listed = self.call("GET", "/preferences/documents")[1]["documents"][0]
        self.assertEqual((listed["contentLength"], listed["estimatedTokens"], listed["enabled"]), (len(DOC), estimate_tokens(DOC), True))
        self.assertNotIn("originalContent", listed)
        self.assertEqual(self.call("GET", f"/preferences/documents/{doc_id}")[1]["document"]["originalContent"], DOC)
        s, patched = self.call("PATCH", f"/preferences/documents/{doc_id}", {"title": "相处说明 v1", "mode": "retrieval"})
        self.assertEqual((s, patched["createdVersion"], patched["document"]["mode"]), (200, False, "retrieval"))
        self.assertEqual(self.call("POST", f"/preferences/documents/{doc_id}/segment", {})[0], 410)
        self.assertEqual(self.call("POST", f"/preferences/documents/{doc_id}/resegment", {})[0], 410)
        s, pub = self.call("POST", f"/preferences/documents/{doc_id}/publish", {"mode": "session_pinned"})
        self.assertEqual((s, pub["document"]["status"], pub["document"]["mode"]), (200, "active", "session_pinned"))
        pack = self.call("GET", "/preferences/session-pack")[1]
        self.assertEqual((pack["documents"][0]["content"], len(pack["packHash"])), (DOC, 16))
        self.assertNotIn("content", self.call("GET", "/preferences/session-pack?content=0")[1]["documents"][0])
        self.assertEqual(self.call("POST", f"/preferences/documents/{doc_id}/disable", {})[1]["document"]["enabled"], False)
        self.assertEqual(self.call("POST", f"/preferences/documents/{doc_id}/enable", {})[1]["document"]["enabled"], True)
        s, version = self.call("POST", f"/preferences/documents/{doc_id}/version", {})
        self.assertEqual((s, version["document"]["version"], version["document"]["status"]), (200, 2, "draft"))
        self.assertEqual(self.call("POST", f"/preferences/documents/{doc_id}/unpublish", {})[1]["document"]["status"], "draft")
        self.assertEqual(self.call("POST", f"/preferences/documents/{doc_id}/archive", {})[1]["document"]["status"], "archived")
        self.assertEqual(self.call("GET", "/preferences/labels")[1]["modes"], ["session_pinned", "retrieval"])
        self.assertEqual(self.call("POST", "/preferences/documents", {"owner": "user", "title": "x", "originalContent": "y", "actor": "deepseek"})[0], 400)
        self.assertEqual(self.call("POST", "/preferences/documents", {"owner": "user", "title": "x", "originalContent": "y", "mode": "auto"})[0], 400)


if __name__ == "__main__":
    unittest.main()
