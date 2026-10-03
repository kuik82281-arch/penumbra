"""Retrieval engine over the unified memory: Episodes / Patterns / RAW ranked by BM25 + vector + entity + time (RRF),
with hard negatives, provenance and lifecycle exclusions. Synthetic corpus; `now` is fixed at 2026-09-24."""
import unittest
from datetime import datetime, timezone

from penumbra.text import bigram_terms, is_stop_term, normalize
from penumbra.embeddings import NoEmbeddings
from tests.support import CONV, Base

OTHER = "conv-other"


class ChineseLexicalTest(unittest.TestCase):
    def test_terms(self):
        self.assertEqual(bigram_terms("初雪"), ["初雪"])  # a two-character word is one term
        self.assertEqual(bigram_terms("猫"), ["猫"])  # a single character stays a unigram
        self.assertEqual(bigram_terms("灯塔！！像在眨眼"), ["灯塔", "像在", "在眨", "眨眼"])  # punctuation splits runs
        self.assertEqual(bigram_terms("换了 iPhone 16"), ["换了", "iphone", "16"])  # mixed script, lower-cased
        self.assertEqual(bigram_terms("ＧＰＳ２０２６"), ["gps2026"])  # NFKC: full-width to ASCII
        self.assertEqual(normalize("ＡＩ　Ｍｅｍｏｒｙ"), "ai memory")

    def test_stop_terms(self):
        for term in ("今天", "觉得", "喜欢", "一个", "user", "ai", "我的", "了吗", "a", "7"):
            self.assertTrue(is_stop_term(term), term)
        for term in ("灯塔", "初雪", "拿铁", "iphone", "16", "猫"):
            self.assertFalse(is_stop_term(term), term)


class EpisodeRankingTest(Base):
    """Lexical + entity + time ranking with the vector signal off (its own evaluation is in test_hybrid)."""

    def setUp(self):
        super().setUp()
        self.svc.close()
        self.open(NoEmbeddings())
        self.lighthouse = self.note("User 和 AI 在灯塔边看了初雪，她说像在眨眼。", entities=["灯塔", "初雪"], importance=0.8, time="2026-01-05T10:00:00Z")
        self.phone = self.note("User 把手机换成了 iPhone 16。", entities=["iPhone 16"], time="2026-08-01T10:00:00Z")
        self.cat = self.note("楼下的橘猫叫年糕，User 每天喂它。", entities=["年糕", "橘猫"], time="2026-07-01T10:00:00Z")
        for i in range(10):  # chatter full of high-frequency words
            self.raw(f"User 今天觉得有点累 {i}", at=f"2026-09-0{1 + i % 9}T10:0{i}:00Z")
        self.svc.vectors.wait_idle(20)

    def ids(self, query, **extra):
        return [r["id"] for r in self.recall(query, **extra)["results"]]

    def test_exact_title_and_content_word(self):
        self.assertEqual(self.ids("灯塔")[:1], [self.lighthouse])
        self.assertEqual(self.ids("初雪")[:1], [self.lighthouse])
        self.assertEqual(self.ids("iPhone 16")[:1], [self.phone])

    def test_entity_partial_containment(self):
        self.assertEqual(self.ids("年糕")[:1], [self.cat])
        self.assertIn(self.cat, self.ids("橘猫"))

    def test_hard_negatives_and_no_match(self):
        for query in ("User 今天觉得", "今天觉得喜欢", "我们今天"):
            self.assertEqual(self.recall(query)["results"], [], query)
        self.assertEqual(self.recall("量子力学的诠释")["results"], [])

    def test_an_archived_or_merged_episode_is_never_returned(self):
        self.memory.store.update_episode(self.cat, {"status": "archived"}, actor="t", reason="t")
        self.svc.vectors.wait_idle(20)
        self.assertNotIn(self.cat, self.ids("年糕"))
        self.memory.store.update_episode(self.phone, {"status": "merged", "superseded_by": self.lighthouse}, actor="t", reason="t")
        self.assertNotIn(self.phone, self.ids("iPhone 16"))

    def test_an_updated_episode_answers_with_its_new_text_only(self):
        self.memory.store.update_episode(self.phone, {"content": "User 把手机换成了 Pixel 10。", "entities": ["Pixel 10"]}, actor="t", reason="t")
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.ids("Pixel 10")[:1], [self.phone])
        self.assertNotIn(self.phone, self.ids("iPhone 16"))

    def test_quote_request_prefers_raw(self):
        rid = self.raw("我说过：灯塔的光是暖黄色的，像一杯热茶。", at="2026-01-05T10:05:00Z")
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.ids("我原话说的灯塔的光是什么颜色")[:1], [rid])

    def test_current_session_raw_is_excluded_but_other_sessions_are_not(self):
        old = self.raw("那天在灯塔我说过想再来一次", at="2026-08-01T10:00:00Z", conv=OTHER)
        new = self.raw("刚才又提到灯塔", at="2026-09-24T11:00:00Z")
        ids = self.ids("灯塔", current_session=(CONV, datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)))
        self.assertIn(old, ids)
        self.assertNotIn(new, ids)

    def test_provenance_of_a_memory_hit_points_at_its_raw(self):
        out = self.svc.retrieval_debug({"query": "灯塔"})
        hit = next(h for h in out["top"] if h["hitId"] == self.lighthouse)
        self.assertEqual(hit["kind"], "EPISODE")
        self.assertEqual(hit["sourceOriginalIds"], self.memory.store.episode(self.lighthouse)["source_raw_ids"])

    def test_a_hit_appears_once(self):
        out = self.svc.retrieval_debug({"query": "灯塔 初雪"})
        self.assertEqual(len({h["hitId"] for h in out["top"]}), len(out["top"]))

    def test_inject_is_stricter_than_recall(self):
        self.assertEqual(self.svc.retrieval_debug({"query": "今天午饭吃了面条", "policy": "inject"})["top"], [])

    def test_debug_trace_is_observability_only(self):
        plain = self.ids("灯塔")
        trace = self.svc.retrieval_debug({"query": "灯塔"})["trace"]
        self.assertEqual(self.ids("灯塔"), plain)
        self.assertEqual(trace["traceVersion"], 2)


if __name__ == "__main__":
    unittest.main()
