"""故事线 stay whole when their Episodes change: a merge moves the links to the merged Episode, a merged-away Episode
is no step of any thread, and a written story whose Episodes changed is written again after a run (a failure is retried
on the next one); a story is always written from the Episodes, never from the story before it."""
import json

from penumbra.memory import actions, threads
from tests.support import Base


class ThreadIntegrityTest(Base):
    def setUp(self):
        super().setUp()
        self.a = self.note("你说下周二一面，有点紧张。", time_start="2026-09-01T10:00:00Z")
        self.b = self.note("一面结束，你说面试官人很好。", time_start="2026-09-02T10:00:00Z")
        self.c = self.note("你拿到了二面通知。", time_start="2026-09-05T10:00:00Z")
        store = self.memory.store
        with store.tx():
            self.tid = threads.attach_proposal(store, self.c, {"new_title": "找工作", "with_episode_ids": [self.a, self.b], "confidence": 0.9}, None)["thread_id"]

    def merge(self, ids):
        store = self.memory.store
        raw = sorted({r for i in ids for r in store.episode(i)["source_raw_ids"]})
        plan = actions.validate_answer({"actions": [{"action": "MERGE_EPISODE", "ref": "m1", "episode_ids": ids, "content": "你说下周二一面，结束后说面试官人很好。",
                                                     "excerpt": "一面前后", "time_start": "2026-09-01T10:00:00Z", "time_end": "2026-09-02T10:00:00Z",
                                                     "entities": [], "topics": [], "state": "", "importance": 0.6, "confidence": 0.9, "source_raw_ids": raw}],
                                        "confidence": 0.9, "reason": "t"}, set(raw), store)
        with store.tx():
            return actions.apply_plan(store, plan, None, "decision_t")["episodes_merged"][0]["into"]

    def test_a_merge_moves_the_thread_to_the_merged_episode(self):
        merged = self.merge([self.a, self.b])
        steps = [s["episode_id"] for s in threads.links(self.memory.store, self.tid)]
        self.assertEqual(steps, [merged, self.c], "the merged Episode takes their place; the merged-away ones are no steps")
        self.assertEqual(threads.threads_of(self.memory.store, merged), [self.tid])
        self.assertEqual(threads.threads_of(self.memory.store, self.a), [])
        self.assertEqual(threads.context_line(self.memory.store, merged)[0]["after"]["episode_id"], self.c)

    def test_a_pending_link_stays_pending_and_a_refusal_is_not_overridden(self):
        store = self.memory.store
        with store.tx():
            store.db.execute("UPDATE thread_links SET status='pending' WHERE episode_id=?", (self.a,))
            store.db.execute("UPDATE thread_links SET status='rejected' WHERE episode_id=?", (self.b,))
        merged = self.merge([self.a, self.b])
        row = store.one("SELECT status FROM thread_links WHERE thread_id=? AND episode_id=?", (self.tid, merged))
        self.assertEqual(row["status"], "pending", "the strongest live status of the old links; a refused one does not make it live")

    def test_a_written_story_is_written_again_when_its_episodes_change(self):
        store, fake = self.memory.store, self.deepseek
        threads.decide(store, {"actor": "user", "action": "close", "thread_id": self.tid})
        fake.decide = lambda payload: {"story": "第一版：" + "；".join(e["content"] for e in payload["episodes"])}
        threads.write_story(store, self.memory.pipeline.verifier.client, self.tid)
        self.assertEqual(threads.refresh_stories(store, self.memory.pipeline.verifier.client), [], "nothing changed")
        store.update_episode(self.c, {"content": "你拿到了二面通知，约在周五下午。"}, actor="user", reason="t")
        fake.decide = lambda payload: RuntimeError("down")
        self.assertEqual(threads.refresh_stories(store, self.memory.pipeline.verifier.client), [], "a failure is left for the next run")
        fake.decide = lambda payload: {"story": "第二版：" + "；".join(e["content"] for e in payload["episodes"])}
        self.assertEqual(threads.refresh_stories(store, self.memory.pipeline.verifier.client), [self.tid])
        t = threads.thread(store, self.tid)
        self.assertIn("周五下午", t["story"])
        self.assertEqual(t["story_version"], 2)
        self.assertNotIn("第一版", json.dumps(fake.payloads[-1], ensure_ascii=False), "written from the Episodes, not from the old story")

    def test_a_story_from_before_is_not_rewritten_just_because_it_has_no_basis(self):
        store = self.memory.store
        with store.tx() as db:
            db.execute("UPDATE threads SET status='closed', story='旧的故事', story_version=1 WHERE thread_id=?", (self.tid,))
        self.deepseek.decide = lambda payload: {"story": "不该被调用"}
        self.assertEqual(threads.refresh_stories(store, self.memory.pipeline.verifier.client), [])
        self.assertEqual(threads.thread(store, self.tid)["story"], "旧的故事")
