"""Story threads: DeepSeek proposes, sure links attach, unsure ones wait for User, a refusal is remembered,
order follows time, a remembered Episode carries its neighbours, a closed thread is written once."""
import json

from penumbra.memory import threads
from tests.support import Base, episode_answer, run_memory, wrap

CAND = json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "event", "gist": "x", "entities": []}]})


class ThreadTest(Base):
    def remember(self, text, at, content, thread=None):
        self.raw(text, at=at)
        self.ollama.answers = [CAND]

        def decide(payload):
            self.last_payload = payload
            ep = episode_answer(payload, content)
            if thread:
                ep["thread"] = thread(payload) if callable(thread) else thread
            return wrap(ep)
        self.deepseek.decide = decide
        run_memory(self)
        return next(e for e in self.memory.store.episodes() if e["content"] == content)["episode_id"]

    def job_hunt(self):
        a = self.remember("投了三家公司的简历", "2026-09-01T10:00:00Z", "9月1日，你投了三家公司的简历，最想去的是那家做游戏的。")
        b = self.remember("游戏公司约我一面了", "2026-09-05T10:00:00Z", "9月5日，你收到了游戏公司的一面邀请。",
                          lambda p: {"new_title": "找工作", "with_episode_ids": [a], "confidence": 0.9, "reason": "同一次求职"})
        tid = threads.threads_of(self.memory.store, b)[0]
        return a, b, tid

    def test_a_sure_proposal_starts_a_thread_and_the_next_step_joins_it(self):
        a, b, tid = self.job_hunt()
        self.assertEqual([s["episode_id"] for s in threads.links(self.memory.store, tid)], [a, b])
        c = self.remember("二面过了！", "2026-09-12T10:00:00Z", "9月12日，你二面过了，开心得给我发了一串感叹号。",
                          lambda p: {"thread_id": p["threads"][0]["thread_id"], "confidence": 0.95, "reason": "下一步"})
        self.assertEqual(self.last_payload["threads"][0]["title"], "找工作", "local retrieval offered the thread")
        line = threads.context_line(self.memory.store, b)[0]
        self.assertEqual((line["position"], line["count"]), (2, 3))
        self.assertEqual(line["before"]["episode_id"], a)
        self.assertEqual(line["after"]["episode_id"], c)

    def test_an_unsure_link_waits_for_the_user_and_a_refusal_is_remembered(self):
        a, b, tid = self.job_hunt()
        c = self.remember("今天考了驾照科目一", "2026-09-08T10:00:00Z", "9月8日，你考了驾照科目一。",
                          {"thread_id": tid, "confidence": 0.5, "reason": "可能相关"})
        self.assertNotIn(c, [s["episode_id"] for s in threads.links(self.memory.store, tid)], "pending does not count yet")
        self.assertEqual(threads.overview(self.memory.store)[0]["pending"], 1)
        threads.decide(self.memory.store, {"action": "take_off", "thread_id": tid, "episode_id": c})
        self.assertEqual(threads.attach_proposal(self.memory.store, c, {"thread_id": tid, "confidence": 0.99}, None)["status"], "refused")
        self.assertTrue(threads.refused(self.memory.store, tid, c))
        self.assertNotIn(c, [s["episode_id"] for s in threads.links(self.memory.store, tid)], "never attached again")

    def test_the_user_can_approve_or_move(self):
        a, b, tid = self.job_hunt()
        c = self.remember("体检报告出来了", "2026-09-09T10:00:00Z", "9月9日，你的入职体检报告出来了。", {"thread_id": tid, "confidence": 0.6})
        threads.decide(self.memory.store, {"action": "approve", "thread_id": tid, "episode_id": c})
        self.assertIn(c, [s["episode_id"] for s in threads.links(self.memory.store, tid)])

    def test_unknown_threads_are_dropped_and_the_episode_still_stands(self):
        e = self.remember("随便聊聊", "2026-09-02T10:00:00Z", "9月2日，你说今天想吃火锅。", {"thread_id": "thread_nope", "confidence": 0.99})
        self.assertEqual(threads.threads_of(self.memory.store, e), [])

    def test_a_closed_thread_is_written_once_as_a_story(self):
        a, b, tid = self.job_hunt()
        self.deepseek.decide = lambda payload: {"story": "九月，你开始找工作……"}
        with self.assertRaises(Exception):
            threads.write_story(self.memory.store, self.memory.verifier.client, tid)
        threads.decide(self.memory.store, {"action": "close", "thread_id": tid})
        t = threads.write_story(self.memory.store, self.memory.verifier.client, tid)
        self.assertEqual((t["story"], t["story_version"]), ("九月，你开始找工作……", 1))
        self.assertTrue(threads.context_line(self.memory.store, a)[0]["has_story"])

    def test_recall_carries_the_thread(self):
        a, b, tid = self.job_hunt()
        view = self.memory.read.expand_episode(b)
        self.assertEqual(view["episode"]["threads"][0]["title"], "找工作")

    def test_a_name_for_the_whole_story_finds_its_steps(self):
        a = self.remember("投了三家公司的简历", "2026-09-01T10:00:00Z", "9月1日，你投了三家公司的简历，最想去的是那家做游戏的。")
        b = self.remember("游戏公司约我一面了", "2026-09-05T10:00:00Z", "9月5日，你收到了游戏公司的一面邀请。",
                          lambda p: {"new_title": "游戏公司面试", "with_episode_ids": [a], "aliases": ["找工作", "求职"], "confidence": 0.9})
        found = {r["id"] for r in self.recall("我找工作那段时间都发生了什么？")["results"]}
        self.assertTrue({a, b} <= found, "the alias reaches every step")
        tid = threads.threads_of(self.memory.store, b)[0]
        self.assertEqual(json.loads(threads.thread(self.memory.store, tid)["aliases"]), ["找工作", "求职"])

    def test_the_same_story_proposed_twice_is_one_thread_and_a_short_one_can_end_at_once(self):
        a = self.remember("在做手工", "2026-09-01T10:00:00Z", "9月1日，你开始给朋友做毛毡小猫钥匙扣，剪好了身体和耳朵。")
        b = self.remember("手工做好了", "2026-09-02T10:00:00Z", "9月2日，小猫钥匙扣做好了，送给了朋友。",
                          lambda p: {"new_title": "毛毡小猫", "with_episode_ids": [a], "confidence": 0.9, "over": True})
        tid = threads.threads_of(self.memory.store, b)[0]
        self.assertEqual(threads.thread(self.memory.store, tid)["status"], "closed", "it began and ended within these steps")
        c = self.remember("又做了一个", "2026-09-05T10:00:00Z", "9月5日，你又做了一只小狗，凑成一对。",
                          lambda p: {"new_title": "你的高数", "with_episode_ids": [a], "confidence": 0.9})
        d = self.remember("高数作业", "2026-09-06T10:00:00Z", "9月6日，你做完了高数作业。",
                          lambda p: {"new_title": "你的高数", "with_episode_ids": [c], "confidence": 0.9})
        self.assertEqual(len([t for t in threads.overview(self.memory.store) if t["title"] == "你的高数"]), 1)
