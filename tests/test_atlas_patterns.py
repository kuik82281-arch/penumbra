"""Three Agent Memory Atlas patterns: verified quotes, contradiction as its own verdict, decay by type."""
import json
from datetime import datetime, timedelta, timezone

from penumbra.memory import actions
from penumbra.memory.quotes import quotes_in, unverified_quotes
from tests.support import Base, episode_answer, run_memory, wrap

CAND = json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "event", "gist": "x", "entities": []}]})


class VerifiedQuotesTest(Base):
    def decide_with(self, content):
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, content))

    def last_status(self):
        return self.memory.store.all("SELECT status, last_error FROM candidates ORDER BY created_at DESC LIMIT 1")[0]

    def test_a_real_quote_passes_and_an_invented_one_is_quarantined(self):
        self.raw("我明天要去面试，好紧张啊", at="2026-09-01T10:00:00Z")
        self.ollama.answers = [CAND]
        self.decide_with("9月1日，你说“明天要去面试，好紧张”，我陪你练了一遍自我介绍。")
        run_memory(self)
        self.assertEqual(self.last_status()["status"], "APPLIED")
        self.raw("今天好累", at="2026-09-02T10:00:00Z")
        self.ollama.answers = [CAND]
        self.decide_with("9月2日，你说“我再也不想上班了”，我抱了你很久。")
        run_memory(self)
        row = self.last_status()
        self.assertEqual(row["status"], "QUARANTINE")
        self.assertIn("unverified quote", row["last_error"])
        self.assertFalse([e for e in self.memory.store.episodes() if "再也不想上班" in e["content"]])

    def test_matching_ignores_spacing_and_punctuation_and_skips_tiny_quotes(self):
        self.assertEqual(quotes_in("你说「好想你」，又说『早点睡』"), ["好想你", "早点睡"])
        self.assertEqual(unverified_quotes(["你说“明天 要去面试！”"], ["明天要去面试，好紧张"]), [])
        self.assertEqual(unverified_quotes(["你说“嗯嗯”"], ["好"]), [], "too short to check")
        self.assertEqual(unverified_quotes(["你说“我从来没说过这句”"], ["别的话"]), ["我从来没说过这句"])


class CorrectionTest(Base):
    def pattern(self, state):
        e = self.note(f"你说你{state}。", time="2026-09-01T10:00:00Z")
        return self.memory.store.insert_pattern({"title": "你和香菜", "topic": "饮食", "narrative": "香菜。", "current_state": state, "current_state_since": "2026-09-01T10:00:00Z",
                                                 "states": [{"state_id": "s1", "state": state, "valid_from": "2026-09-01T10:00:00Z", "valid_to": None, "episode_ids": [e]}],
                                                 "supporting_episode_ids": [e], "entities": [], "valid_from": "2026-09-01T10:00:00Z", "confidence": 0.8}, actor="t", reason="t")["pattern_id"]

    def update(self, pid, new_state):
        plan = actions.validate_answer(wrap({"action": "UPDATE_PATTERN", "pattern_id": pid, "new_state": new_state}), set(), self.memory.store)
        return actions.apply_plan(self.memory.store, plan, None, "d1")

    def test_she_changed_keeps_the_old_state_as_history(self):
        pid = self.pattern("喜欢香菜")
        self.update(pid, {"state": "不喜欢香菜", "valid_from": "2026-09-20T10:00:00Z"})
        p = self.memory.store.pattern(pid)
        self.assertEqual(p["current_state"], "不喜欢香菜")
        self.assertEqual([s["state"] for s in p["historical_states"]], ["喜欢香菜"])

    def test_recorded_wrong_is_corrected_in_place_and_never_becomes_history(self):
        pid = self.pattern("喜欢香菜")
        applied = self.update(pid, {"state": "从来不吃香菜", "correction": True})
        p = self.memory.store.pattern(pid)
        self.assertEqual(p["current_state"], "从来不吃香菜")
        self.assertEqual(p["historical_states"], [], "the wrong state is not '以前'")
        current = next(s for s in p["states"] if not s.get("valid_to"))
        self.assertEqual(current["refuted"][0]["state"], "喜欢香菜")
        self.assertTrue(applied["state_changes"][0]["correction"])


class DecayByTypeTest(Base):
    def test_half_lives(self):
        core = self.svc.retrieval
        self.assertEqual(core._half_life("PATTERN", 0.5), 365.0)
        self.assertEqual(core._half_life("EPISODE", 0.85), 365.0, "vows, milestones barely age")
        self.assertEqual(core._half_life("EPISODE", 0.6), 45.0)
        self.assertEqual(core._half_life("EPISODE", 0.4), 14.0, "small everyday things fade fast")
        now = datetime(2026, 10, 2, tzinfo=timezone.utc)
        old = now - timedelta(days=60)
        b = lambda kind, imp: core._boosts(kind=kind, importance=imp, when=old, now=now, entity_share=0, source_quality=0, role=None, mode="memory")["recency"]  # noqa: E731
        self.assertGreater(b("PATTERN", 0.5), b("EPISODE", 0.6))
        self.assertGreater(b("EPISODE", 0.6), b("EPISODE", 0.4))


class LexiconTest(Base):
    """我们的词: memes, pet names, nicknames - kept, one per word, never fading."""

    def remember(self, text, content, tag, at):
        self.raw(text, at=at)
        self.ollama.answers = [CAND]
        self.deepseek.decide = lambda payload: wrap({**episode_answer(payload, content), "kind": "lexicon", "tag": tag})
        run_memory(self)

    def test_a_meme_is_kept_once_and_grows_with_new_uses(self):
        self.remember("以后你叫我小太阳", "9月1日，你让我以后叫你“小太阳”，是你给自己起的。", "昵称·小太阳", "2026-09-01T10:00:00Z")
        self.remember("小太阳要喝奶茶", "9月3日，你又用“小太阳”要奶茶，这成了你使唤我时的自称。", "昵称·小太阳", "2026-09-03T10:00:00Z")
        words = [e for e in self.memory.store.episodes(status="active") if e.get("kind") == "lexicon"]
        self.assertEqual(len(words), 1, "the same word is one memory")
        self.assertIn("使唤我", words[0]["content"])
        self.assertEqual(len(words[0]["source_raw_ids"]), 2)
        self.assertGreaterEqual(words[0]["importance"], 0.6)

    def test_a_tag_is_made_when_missing_and_their_words_never_fade(self):
        doc = actions.new_commitment({"kind": "lexicon", "content": "你教我说“芜湖起飞”，开心到飞起的意思。", "time_start": "2026-09-01T10:00:00Z"})
        self.assertEqual(doc["tag"], "梗·芜湖起飞")
        self.assertEqual(self.svc.retrieval._half_life("EPISODE", 0.6, ["梗·芜湖起飞"]), 365.0)
        self.assertEqual(self.svc.retrieval._half_life("EPISODE", 0.6, ["饮食"]), 45.0)


class LenientTest(Base):
    def test_words_and_numbers_that_used_to_lose_a_decision(self):
        from penumbra.text import is_stop_term
        self.assertFalse(is_stop_term("小说"), "two stop characters can still be a word")
        self.assertFalse(is_stop_term("大学"))
        self.assertTrue(is_stop_term("我的"))
        self.assertEqual(actions._unit("90", "confidence"), 0.9)
        self.assertEqual(actions._unit(85, "confidence"), 0.85)
        self.assertEqual(actions._unit("0.7", "confidence"), 0.7)
