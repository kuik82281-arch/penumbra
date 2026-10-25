"""Held items: what the model cannot place waits alone for her review, and the rest of the segment is written - in one
transaction, with the held item's own RAW as its evidence, applied through the Studio's existing review."""
import json

from penumbra.memory import edit
from tests.support import Base, episode_answer, run_memory, wrap

CAND = json.dumps({"candidates": [{"from": 1, "to": 2, "kind": "event", "gist": "x", "entities": []}]})


class HoldTest(Base):
    def segment(self):
        self.raw("今天面试过了，下周二终面", at="2026-09-01T10:00:00Z")
        self.raw("晚上又去吃了那家火锅", at="2026-09-01T10:05:00Z")
        self.ollama.answers = [CAND]

    def answer(self, payload, held=None, main=None):
        sure = main if main is not None else [episode_answer(payload, "你说今天面试过了，下周二终面。")]
        out = wrap(*sure)
        out["held"] = held if held is not None else [{**episode_answer(payload, "晚上你又去吃了那家火锅。", ref="h1", raw_index=1),
                                                      "hold_reason": "不确定是不是上次那顿火锅的后续"}]
        return out

    def statuses(self):
        return {r["origin"]: r["status"] for r in self.memory.store.all("SELECT origin, status FROM candidates")}

    def test_the_sure_part_is_written_and_the_unsure_part_waits_alone(self):
        self.segment()
        self.deepseek.decide = lambda payload: self.answer(payload)
        run_memory(self)
        contents = [e["content"] for e in self.memory.store.episodes()]
        self.assertTrue(any("终面" in c for c in contents))
        self.assertFalse(any("火锅" in c for c in contents), "held: not written yet")
        held = self.memory.store.one("SELECT * FROM candidates WHERE origin='hold'")
        self.assertEqual(held["status"], "QUARANTINE")
        self.assertIn("火锅的后续", held["last_error"])
        hot_pot_raw = [r for r in json.loads(held["raw_ids"])]
        self.assertEqual(len(hot_pot_raw), 1, "its evidence is the held item's own RAW, not the whole segment")
        # she confirms it in the Studio: the held action is applied as it was written, validated against its own RAW
        edit.review(self.memory, held["candidate_id"], {"actor": "user", "decision": "apply"})
        self.assertTrue(any("火锅" in e["content"] for e in self.memory.store.episodes()))
        ep = next(e for e in self.memory.store.episodes() if "火锅" in e["content"])
        self.assertEqual(ep["source_raw_ids"], hot_pot_raw)

    def test_only_held_with_no_action(self):
        self.segment()
        self.deepseek.decide = lambda payload: self.answer(payload, main=[{"action": "NO_ACTION", "reason": "只有拿不准的"}])
        run_memory(self)
        self.assertEqual(self.statuses()["hold"], "QUARANTINE")
        self.assertIn("NO_ACTION", [r["status"] for r in self.memory.store.all("SELECT status FROM candidates WHERE origin!='hold'")])
        self.assertEqual(self.memory.store.one("SELECT COUNT(*) AS n FROM candidates WHERE origin='hold' AND status='QUARANTINE'")["n"], 1)
        self.assertFalse(self.memory.store.episodes())

    def test_a_broken_held_item_quarantines_the_whole_decision_as_before(self):
        self.segment()
        self.deepseek.decide = lambda payload: self.answer(payload, held=[{**episode_answer(payload, "火锅。", ref="h1"), "source_raw_ids": ["o_not_in_this_segment"]}])
        run_memory(self)
        self.assertFalse(self.memory.store.episodes(), "nothing half-written")
        self.assertIsNone(self.memory.store.one("SELECT 1 FROM candidates WHERE origin='hold'"))
        self.assertIn("QUARANTINE", [r["status"] for r in self.memory.store.all("SELECT status FROM candidates")])

    def test_an_invented_quote_in_a_held_item_stops_everything(self):
        self.segment()
        self.deepseek.decide = lambda payload: self.answer(payload, held=[{**episode_answer(payload, "你说“我再也不吃火锅了”。", ref="h1", raw_index=1), "hold_reason": "?"}])
        run_memory(self)
        self.assertFalse(self.memory.store.episodes())
        self.assertIsNone(self.memory.store.one("SELECT 1 FROM candidates WHERE origin='hold'"))
