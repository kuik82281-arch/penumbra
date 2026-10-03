"""User's 删除 (what 归档 used to be): the Episode / Pattern is gone - store, versions, relations, index, vectors, and
no copy of its text in the audit - while every RAW it came from stays."""
import json
import unittest

from penumbra.memory import actions
from tests.support import Base, episode_answer, run_memory, wrap

CAND = lambda gist: json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "preference", "gist": gist, "entities": ["香菜"]}]})  # noqa: E731


class DeleteTest(Base):
    def remember(self, text, at, pattern=None):
        rid = self.raw(text, at=at)
        self.ollama.answers = [CAND(text)]

        def decide(payload):
            e = episode_answer(payload, f"User 说：{text}", entities=["香菜"])
            if pattern is None:
                return wrap(e, {"action": "CREATE_PATTERN", "ref": "p1", "title": "User 对香菜的态度", "topic": "饮食", "narrative": "User 对香菜。", "current_state": text,
                                "entities": ["香菜"], "confidence": 0.9, "supporting_episode_ids": ["e1"]})
            return wrap(e, {"action": "UPDATE_PATTERN", "pattern_id": pattern, "add_supporting": ["e1"]})
        self.deepseek.decide = decide
        run_memory(self)
        return rid

    def indexed(self, ident):
        return self.svc.conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (ident,)).fetchone()[0]

    def delete(self, kind, ident):
        return self.memory.route("POST", ["edit"], {"actor": "user", "action": f"{kind}.archive", "id": ident})

    def test_deleting_an_episode_keeps_its_raw_and_the_pattern_lets_go(self):
        r1 = self.remember("我以前特别讨厌香菜", "2026-09-01T10:00:00Z")
        pid = self.memory.store.patterns()[0]["pattern_id"]
        self.remember("现在我特别喜欢香菜", "2026-09-10T10:00:00Z", pattern=pid)
        first = next(e for e in self.memory.store.episodes() if r1 in e["source_raw_ids"])
        self.assertEqual(self.indexed(first["episode_id"]), 1)
        self.delete("episode", first["episode_id"])
        self.svc.vectors.wait_idle(20)
        eid = first["episode_id"]
        self.assertEqual(self.memory.store.all("SELECT 1 FROM episodes WHERE episode_id = ?", (eid,)), [])
        self.assertEqual(self.memory.store.all("SELECT 1 FROM episode_versions WHERE episode_id = ?", (eid,)), [])
        self.assertEqual(self.memory.store.all("SELECT 1 FROM relations WHERE src_id = ? OR dst_id = ?", (eid, eid)), [])
        self.assertEqual(self.indexed(eid), 0)
        self.assertNotIn(eid, self.svc.vectors.vectors)
        self.assertEqual(self.memory.raw([r1])[0]["id"], r1)  # the RAW stays
        pattern = self.memory.store.pattern(pid)
        self.assertNotIn(eid, pattern["supporting_episode_ids"])  # the Pattern lives on with its other Episode
        texts = [json.dumps([a["before"], a["after"]], ensure_ascii=False) for a in self.memory.store.audit_rows(eid)]
        self.assertTrue(texts and all("讨厌香菜" not in t for t in texts), texts)

    def test_deleting_the_last_episode_of_a_pattern_deletes_the_pattern(self):
        self.remember("我以前特别讨厌香菜", "2026-09-01T10:00:00Z")
        pid = self.memory.store.patterns()[0]["pattern_id"]
        eid = self.memory.store.episodes()[0]["episode_id"]
        self.delete("episode", eid)
        self.assertEqual(self.memory.store.counts()["patternsAll"], 0)
        self.assertEqual(self.indexed(pid), 0)

    def test_deleting_a_pattern_keeps_its_episodes(self):
        self.remember("我以前特别讨厌香菜", "2026-09-01T10:00:00Z")
        pid = self.memory.store.patterns()[0]["pattern_id"]
        self.delete("pattern", pid)
        self.assertEqual(self.memory.store.all("SELECT 1 FROM pattern_versions WHERE pattern_id = ?", (pid,)), [])
        self.assertEqual(self.indexed(pid), 0)
        self.assertEqual(self.memory.store.counts()["episodes"], 1)
        self.assertEqual(self.memory.read.retrieve({"query": "香菜", "dry": True})["patterns"], [])


if __name__ == "__main__":
    unittest.main()


class TombstoneTest(DeleteTest):
    """Deleted stays deleted when the same RAW is read again (reprocess with onlyEmpty=false); new evidence still counts."""

    def test_a_deleted_memory_is_not_mined_again_from_the_same_raw(self):
        r1 = self.remember("我以前特别讨厌香菜", "2026-09-01T10:00:00Z")
        first = next(e for e in self.memory.store.episodes() if r1 in e["source_raw_ids"])
        self.delete("episode", first["episode_id"])
        tombs = self.memory.store.all("SELECT raw_ids FROM tombstones")
        self.assertEqual([json.loads(t["raw_ids"]) for t in tombs], [[r1]])
        self.assertNotIn("讨厌香菜", json.dumps([dict(t) for t in tombs], ensure_ascii=False))  # no text survives
        # Read every window again; discovery finds the same passage (a new span: a different candidate).
        self.memory.pipeline.reprocess(only_empty=False)
        self.ollama.answers = [json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "preference", "gist": "香菜 again", "entities": ["香菜"]}]})]
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, "User 说她以前讨厌香菜", entities=["香菜"]),
                                                    {"action": "CREATE_PATTERN", "ref": "p1", "title": "香菜", "topic": "饮食", "narrative": "香菜。", "current_state": "讨厌",
                                                     "entities": ["香菜"], "confidence": 0.9, "supporting_episode_ids": ["e1"]})
        run_memory(self)
        self.assertFalse([e for e in self.memory.store.episodes(status="active") if r1 in e["source_raw_ids"]])
        # A fresh decision that cites only that RAW (what a differently-cut span would bring): nothing is created,
        # and the Pattern leaning on it goes too.
        payload = {"RAW": [{"id": r1, "createdAt": "2026-09-01T10:00:00Z"}]}
        answer = wrap(episode_answer(payload, "你以前特别讨厌香菜", entities=["香菜"]),
                      {"action": "CREATE_PATTERN", "ref": "p1", "title": "你和香菜", "topic": "饮食", "narrative": "香菜。", "current_state": "讨厌",
                       "entities": ["香菜"], "confidence": 0.9, "supporting_episode_ids": ["e1"]})
        plan = actions.validate_answer(answer, {r1}, self.memory.store)
        applied = actions.apply_plan(self.memory.store, plan, None, "d-tomb")
        self.assertEqual(applied["tombstoned"], ["e1", "p1"])
        self.assertEqual(applied["episodes_created"], [])
        self.assertFalse(self.memory.store.patterns(status="active"))

    def test_saying_it_again_is_new_evidence(self):
        r1 = self.remember("我以前特别讨厌香菜", "2026-09-01T10:00:00Z")
        self.delete("episode", next(e for e in self.memory.store.episodes() if r1 in e["source_raw_ids"])["episode_id"])
        r2 = self.remember("我真的很讨厌香菜", "2026-09-20T10:00:00Z")
        self.assertTrue([e for e in self.memory.store.episodes(status="active") if r2 in e["source_raw_ids"]])
