"""Memory gate: DeepSeek says whether a message needs memory (intent) and which candidates really help (judge).
A failure is no gate - the turn searches and keeps what the ranking kept."""
import tests.support  # noqa: F401 - no real model in tests
import unittest

from penumbra.errors import WorkerError
from penumbra.memory.gate import MemoryGate
from tests.support import CONV, Base


class FakeClient:
    """Answers by prompt: intent / judge. An Exception answer raises."""

    def __init__(self, intent=None, judge=None):
        self.answers, self.calls = {"intent": intent, "judge": judge}, []

    def available(self):
        return True

    def chat_json(self, system, user, max_tokens=0):
        kind = "judge" if "候选记忆" in user else "intent"
        self.calls.append((kind, user))
        answer = self.answers[kind]
        if isinstance(answer, Exception):
            raise answer
        return answer, {}


class GateRetrieveTest(Base):
    def setUp(self):
        super().setUp()
        self.fight = self.note("我们吵架的时候，先抱住我，别讲道理，等我冷静下来再说。", topics=["安抚"], importance=0.9)

    def retrieve(self, query, fake, turn="turn-g", **extra):
        self.memory.read.gate = MemoryGate(fake)
        return self.memory.read.retrieve({"query": query, "turnId": turn, "conversationId": CONV, "sessionId": "s1", "gate": "on", **extra})

    def test_need_none_answers_without_a_search(self):
        fake = FakeClient(intent={"need": "none", "seek": ""})
        r = self.retrieve("我们吵架了（抱住）", fake)
        self.assertEqual((r["status"], r["search_count"]), ("NO_MEMORY_NEEDED", 0))
        self.assertEqual([k for k, _ in fake.calls], ["intent"])
        self.assertEqual(r["trace"]["stages"][0]["stage"], "intent")
        self.assertIsNotNone(r["inject_id"], "the skipped turn is still logged")

    def test_asking_for_the_past_is_never_skipped(self):
        fake = FakeClient(intent={"need": "none", "seek": ""}, judge={"keep": [0]})
        r = self.retrieve("还记得我们吵架的时候你该怎么做吗", fake)
        self.assertEqual(r["status"], "LOCKED")

    def test_judge_drops_what_does_not_help(self):
        fake = FakeClient(intent={"need": "maybe", "seek": "吵架后她希望怎么被安抚"}, judge={"keep": []})
        r = self.retrieve("我们吵架了你该怎么做", fake)
        self.assertEqual(r["status"], "NO_MEMORY_NEEDED")
        judge = next(s for s in r["trace"]["stages"] if s["stage"] == "judge")
        self.assertTrue(judge["dropped"])
        self.assertIn("要找的：吵架后她希望怎么被安抚", fake.calls[-1][1])

    def test_judge_keeps_what_helps(self):
        fake = FakeClient(intent={"need": "yes", "seek": "吵架后她希望怎么被安抚"}, judge={"keep": [0]})
        r = self.retrieve("我们吵架了你该怎么做", fake)
        self.assertEqual(r["status"], "LOCKED")
        self.assertIn("seek", [s["stage"] for s in r["trace"]["stages"]])

    # --- degradation: a failing gate never means "inject what the ranking found"
    def test_intent_failing_on_an_ordinary_message_brings_no_memory(self):
        fake = FakeClient(intent=WorkerError("network: timed out", 0), judge={"keep": [0]})
        r = self.retrieve("我们吵架了你该怎么做", fake)
        self.assertEqual((r["status"], r["degraded"], r["retry_hint"]), ("NO_MEMORY_NEEDED", "intent_timeout_skip", False))
        self.assertEqual([k for k, _ in fake.calls], ["intent"], "no search, no judge")

    def test_intent_failing_on_a_history_question_still_searches_and_the_judge_decides(self):
        fake = FakeClient(intent=WorkerError("HTTP 500: oops", 0), judge={"keep": [0]})
        r = self.retrieve("还记得我们吵架的时候你该怎么做吗", fake)
        self.assertEqual((r["status"], r["degraded"]), ("LOCKED", "intent_error_history"))

    def test_judge_failing_keeps_only_strong_candidates_for_a_needed_memory(self):
        down = WorkerError("network: timed out", 0)
        r = self.retrieve("还记得我们吵架的时候你该怎么做吗", FakeClient(intent=down, judge=down))
        self.assertEqual((r["status"], r["degraded"]), ("LOCKED", "intent_timeout_history"), "words agree: kept")
        r = self.retrieve("我们吵架了你该怎么做", FakeClient(intent={"need": "yes", "seek": "吵架后怎么安抚"}, judge=down), turn="turn-y")
        self.assertEqual((r["status"], r["degraded"]), ("LOCKED", "judge_timeout_strong_only"))

    def test_judge_failing_on_a_maybe_brings_nothing_unverified(self):
        fake = FakeClient(intent={"need": "maybe", "seek": "吵架后怎么安抚"}, judge=WorkerError("HTTP 500: oops", 0))
        r = self.retrieve("我们吵架了你该怎么做", fake)
        self.assertEqual((r["status"], r["degraded"]), ("NO_MEMORY_NEEDED", "judge_error_dropped"))

    def test_an_intent_timeout_is_not_followed_by_a_judge_call(self):
        fake = FakeClient(intent=WorkerError("network: timed out", 0), judge={"keep": []})
        r = self.retrieve("还记得我们吵架的时候你该怎么做吗", fake)
        self.assertEqual([k for k, _ in fake.calls], ["intent"], "the judge is not asked to time out a second time")
        self.assertEqual((r["status"], r["degraded"]), ("LOCKED", "intent_timeout_history"), "strong candidates still come")

    def test_a_history_question_left_with_nothing_asks_him_to_recall(self):
        down = WorkerError("network: timed out", 0)
        r = self.retrieve("还记得上次去冰岛看极光的事吗", FakeClient(intent=down, judge=down))
        self.assertEqual(r["status"], "NO_MEMORY_NEEDED")
        self.assertTrue(r["retry_hint"])

    def test_counts_are_kept_for_real_turns_not_for_dry_runs(self):
        down = WorkerError("network: timed out", 0)
        self.retrieve("我们吵架了你该怎么做", FakeClient(intent=down), turn="t1")
        self.retrieve("我们吵架了你该怎么做", FakeClient(intent={"need": "yes", "seek": ""}, judge={"keep": [0]}), turn="t2")
        self.retrieve("我们吵架了你该怎么做", FakeClient(intent=down), turn="t3", dry=True)
        stats = self.memory.read.gate_stats()
        self.assertEqual((stats["turns"], stats["intent_timeout"], stats["intent_ok"], stats["judge_ok"]), (2, 1, 1, 1))
        self.assertEqual((stats["degraded_intent_timeout_skip"], stats["injected"], stats["no_memory"]), (1, 1, 1))

    def test_judge_mode_skips_the_intent(self):
        fake = FakeClient(intent={"need": "none", "seek": ""}, judge={"keep": [0]})
        r = self.retrieve("我们吵架了你该怎么做", fake, gate="judge")
        self.assertEqual(r["status"], "LOCKED")
        self.assertEqual([k for k, _ in fake.calls], ["judge"])

    def test_off_asks_nothing(self):
        fake = FakeClient(intent={"need": "none", "seek": ""}, judge={"keep": []})
        r = self.retrieve("我们吵架了你该怎么做", fake, gate="off")
        self.assertEqual((r["status"], fake.calls), ("LOCKED", []))

    def test_the_intent_is_told_who_a_name_is(self):
        from penumbra.memory import names
        names.save(self.memory.store, {"name": "年糕", "kind": "宠物", "note": "她领养的橘猫"})
        fake = FakeClient(intent={"need": "none", "seek": ""})
        self.retrieve("年糕想吃这个", fake)
        self.assertIn("年糕（宠物）：她领养的橘猫", fake.calls[0][1])
        self.retrieve("我想吃这个", fake, turn="t-plain")
        self.assertNotIn("她自己记下的", fake.calls[1][1])

    def test_an_episode_told_from_the_current_session_is_not_a_memory(self):
        raw = self.memory.store.episode(self.fight)["source_raw_ids"][0]
        row = self.svc.conn.execute("SELECT conversation_id, created_at FROM originals WHERE id = ?", (raw,)).fetchone()
        fake = FakeClient(judge={"keep": [0]})
        ask = lambda since, turn: self.retrieve("我们吵架了你该怎么做", fake, turn=turn, gate="judge",  # noqa: E731
                                                currentSession={"conversationId": row["conversation_id"], "since": since})
        r = ask("2000-01-01T00:00:00.000Z", "t-in")
        self.assertEqual(r["status"], "NO_MEMORY_NEEDED")
        self.assertEqual(next(s for s in r["trace"]["stages"] if s["stage"] == "current_session")["dropped"], [self.fight])
        self.assertEqual(ask("2999-01-01T00:00:00.000Z", "t-before")["status"], "LOCKED")  # said before this session opened


class GateParseTest(unittest.TestCase):
    def test_bad_answers_are_errors_not_decisions(self):
        self.assertIn("error", MemoryGate(FakeClient(intent={"need": "sometimes"})).intent("x", []))
        self.assertIn("error", MemoryGate(FakeClient(judge={"keep": "all"})).judge("x", [], "", ["a"]))

    def test_no_time_left_is_a_timeout_without_a_call(self):
        fake = FakeClient(intent={"need": "yes", "seek": ""})
        out = MemoryGate(fake).intent("x", [], timeout_s=0.1)
        self.assertEqual((out.get("failure"), fake.calls), ("timeout", []))

    def test_after_two_timeouts_deepseek_is_not_asked_for_a_while(self):
        fake = FakeClient(intent=WorkerError("network: timed out", 0))
        gate = MemoryGate(fake)
        gate.intent("x", [])
        gate.intent("x", [])
        out = gate.intent("x", [])
        self.assertEqual((len(fake.calls), out.get("breaker"), out.get("failure")), (2, True, "timeout"))
        gate._open_until = 0  # the minute has passed
        fake.answers["intent"] = {"need": "none", "seek": ""}
        self.assertEqual(gate.intent("x", []).get("need"), "none")
        self.assertEqual(gate._misses, 0, "an answer closes the breaker")

    def test_judge_keeps_only_valid_indices(self):
        out = MemoryGate(FakeClient(judge={"keep": [1, "0", 7, -1]})).judge("x", [], "", ["a", "b"])
        self.assertEqual(out["keep"], [0, 1])


if __name__ == "__main__":
    unittest.main()
