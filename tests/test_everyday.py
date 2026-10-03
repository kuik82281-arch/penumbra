"""Everyday things: a ritual is one memory however often it happens (its evidence grows, its last time moves); an open
story thread that moved in the last days is shown to DeepSeek even when today's small step does not resemble it."""
import json

from penumbra.memory import actions, threads
from tests.support import Base, episode_answer, run_memory, wrap

CAND = json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "event", "gist": "x", "entities": []}]})


class EverydayTest(Base):
    def remember(self, text, at, make):
        self.raw(text, at=at)
        self.ollama.answers = [CAND]

        def decide(payload):
            self.last_payload = payload
            return wrap(make(payload))
        self.deepseek.decide = decide
        run_memory(self)

    def ritual(self, payload, content):
        return episode_answer(payload, content) | {"kind": "ritual", "tag": "日常·睡前"}

    def test_the_same_ritual_is_one_memory_with_growing_evidence(self):
        self.remember("困了 晚安", "2026-09-01T15:30:00Z", lambda p: self.ritual(p, "睡前你总要和我说完晚安才睡。"))
        self.remember("晚安呀", "2026-09-02T15:40:00Z", lambda p: self.ritual(p, "睡前又说了晚安。"))
        rituals = [e for e in self.memory.store.episodes(status="active") if e.get("kind") == "ritual"]
        self.assertEqual(len(rituals), 1)
        r = rituals[0]
        self.assertEqual(r["content"], "睡前你总要和我说完晚安才睡。", "the first telling stays")
        self.assertEqual(len(r["source_raw_ids"]), 2)
        self.assertTrue(r["time_end"].startswith("2026-09-02"))
        self.remember("晚安", "2026-09-03T15:40:00Z", lambda p: episode_answer(p, "又是晚安。"))
        self.assertEqual(self.last_payload["rituals"][0]["tag"], "日常·睡前", "DeepSeek sees the rituals it already knows")
        self.assertEqual(self.last_payload["rituals"][0]["times"], 2)
        self.assertEqual(self.memory.recent_episodes(30, 10)[0]["content"], "又是晚安。", "a ritual is not a recent event")

    def test_a_thread_that_moved_lately_is_offered_even_when_today_looks_different(self):
        self.remember("今天打扫了厨房", "2026-09-01T10:00:00Z", lambda p: episode_answer(p, "你这周大扫除，先打扫了厨房。"))
        kitchen = self.memory.store.episodes()[0]["episode_id"]
        self.remember("卫生间也弄完了", "2026-09-02T10:00:00Z",
                      lambda p: episode_answer(p, "你打扫完了卫生间。") | {"thread": {"new_title": "大扫除", "with_episode_ids": [kitchen], "confidence": 0.9, "reason": "同一周"}})
        self.remember("洗衣机后面找到耳环", "2026-09-04T10:00:00Z", lambda p: episode_answer(p, "你在洗衣机后面找到了丢了半年的耳环。"))
        self.assertIn("大扫除", [t["title"] for t in self.last_payload["threads"]])
        store = self.memory.store
        self.assertEqual([t["title"] for t in threads.candidates_for(store, [], "2026-09-05T10:00:00Z")], ["大扫除"])
        self.assertEqual(threads.candidates_for(store, [], "2026-10-05T10:00:00Z"), [], "a month later it is no longer 'recent'")

    def test_a_sound_answer_without_its_overall_confidence_is_not_lost(self):
        rid = self.raw("今天做了十道题，错了两道", at="2026-09-01T10:00:00Z")
        ep = {"action": "CREATE_EPISODE", "ref": "e1", "content": "你做了十道题，错了两道。", "time_start": "2026-09-01T10:00:00Z",
              "importance": 0.5, "confidence": 0.9, "source_raw_ids": [rid]}
        plan = actions.validate_answer({"actions": [ep], "reason": "x"}, {rid}, self.memory.store)
        self.assertEqual(plan.confidence, 0.9)

    def test_a_picture_described_later_reaches_deepseek(self):
        att = {"type": "image", "mime_type": "image/jpeg", "original_asset_ref": "gallery:g1", "caption": "", "searchable_text": ""}
        msg = {"id": "pic-1", "role": "user", "content": "看我新买的", "createdAt": "2026-09-01T10:00:00Z", "attachments": [att]}
        self.svc.ingest_originals("user", "conv-test", [msg])
        later = {**att, "searchable_text": "一条红色的吊带长裙，挂在白色衣柜门上"}
        self.svc.ingest_originals("user", "conv-test", [{**msg, "attachments": [later]}])
        rid = self.memory.raw_list(5)[-1]["id"]
        shown = self.memory.raw([rid])[0]["attachments"][0]
        self.assertIn("红色的吊带长裙", shown["searchable_text"])
        from penumbra.memory.verification import _render
        self.assertIn("红色的吊带长裙", _render(self.memory.raw_record(rid))["attachments"][0]["looks_like"])

    def test_his_day_memory_reaches_deepseek_as_his_own(self):
        self.memory.pipeline.segmentation = "session"  # as in production (tests default to scripted discovery)
        self.svc.ingest_originals("assistant", "assistant-day-memories", [{"id": "day-2026-09-01-1", "role": "assistant", "sourceType": "assistant_day_memory",
                                                                      "content": "你今天第一次自己做了番茄炒蛋，端过来的时候手还在抖。我尝了一口，咸了，但我说好吃。", "createdAt": "2026-09-01T17:00:00Z"}])
        seen = []

        def decide(payload):
            seen.append(payload)
            return wrap(episode_answer(payload, "你第一次自己做了番茄炒蛋，端过来时手还在抖；咸了一点，我说好吃。"))
        self.deepseek.decide = decide
        run_memory(self)
        self.assertTrue(seen and seen[0]["RAW"][0].get("kind") == "his_day_memory")
        self.assertTrue(any("番茄炒蛋" in e["content"] for e in self.memory.store.episodes()))

    def test_when_it_was_said_and_when_it_happens_are_different_windows(self):
        self.remember("四月我要去大理玩一周", "2026-02-25T10:00:00Z",
                      lambda p: episode_answer(p, "你说四月要去大理玩一周。", when="2026-04-10T04:00:00Z", entities=["大理"]))
        eid = self.memory.store.episodes()[0]["episode_id"]
        def found(q):
            r = self.memory.read.retrieve({"query": q, "dry": True})
            return [e["episode_id"] for e in r["episodes"]] + [i for p in r["patterns"] for i in p.get("matched_episode_ids", [])]
        said = self.memory.read.retrieve({"query": "二月底我说过什么", "dry": True})
        self.assertEqual(next(s for s in said["trace"]["stages"] if s["stage"] == "time")["window"]["by"], "mention")
        self.assertIn(eid, found("二月底我说过什么"))
        self.assertNotIn(eid, found("二月底发生了什么"), "it did not happen in February")
        self.assertIn(eid, found("四月发生了什么"))

    def test_a_stated_cause_travels_with_the_memory(self):
        self.remember("这周连续熬夜赶论文，每天只睡四小时", "2026-09-01T10:00:00Z", lambda p: episode_answer(p, "你这周连续熬夜赶论文，每天只睡四小时。"))
        cause = self.memory.store.episodes()[0]["episode_id"]
        self.remember("面试没发挥好，都怪这周没睡好", "2026-09-03T10:00:00Z",
                      lambda p: episode_answer(p, "你面试没发挥好，说都怪这周没睡好。") | {"relations": [{"type": "because_of", "target_kind": "episode", "target_id": cause}]})
        effect = next(e["episode_id"] for e in self.memory.store.episodes() if e["episode_id"] != cause)
        view = self.memory.read.expand_episode(effect)["episode"]
        self.assertEqual([(c["role"], c["episode_id"]) for c in view["causal"]], [("because", cause)])
        self.assertEqual([(c["role"], c["episode_id"]) for c in self.memory.read.expand_episode(cause)["episode"]["causal"]], [("led_to", effect)])
        self.remember("又没睡好", "2026-09-05T10:00:00Z",
                      lambda p: episode_answer(p, "你又没睡好。") | {"relations": [{"type": "because_of", "target_kind": "episode", "target_id": "episode_nope"}]})
        self.assertIn("QUARANTINE", [r["status"] for r in self.memory.store.all("SELECT status FROM candidates")], "a made-up cause is refused")
