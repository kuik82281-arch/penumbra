"""Hybrid retrieval over Episodes / Patterns / RAW: vector + entity + lexical fused by RRF. A small concept-based fake
embedding provider makes semantics deterministic here; the real local model is exercised in the live evaluation."""
import unittest
from datetime import datetime, timezone

from penumbra.retrieval import RetrievalRequest
from tests.support import CONCEPTS, CONV, Base, ConceptEmbeddings


class HybridTest(Base):
    def test_semantic_only_hit_is_found_and_traced(self):
        fight = self.note("我们吵架的时候，先抱住我，别讲道理，等我冷静下来再说。", topics=["安抚"])
        self.note("早上一定要一杯燕麦拿铁。")
        query = "我们闹别扭了你该怎么做"
        result = self.recall(query)
        self.assertEqual(result["results"][0]["id"], fight)
        trace = self.svc.retrieval_debug({"query": query})["trace"]
        hit = next(c for c in trace["candidates"] if c["hitId"] == fight)
        self.assertEqual(trace["traceVersion"], 2)
        self.assertIsNone(hit["signals"]["lexical"]["rank"])  # no shared words ...
        self.assertEqual(hit["signals"]["vector"]["rank"], 1)  # ... found semantically
        self.assertEqual((hit["signals"]["vector"]["provider"], hit["signals"]["vector"]["model"]), ("fake", "concepts-v1"))
        self.assertGreater(hit["fusion"]["vectorContribution"], 0)
        self.assertEqual(hit["fusion"]["lexicalContribution"], 0)
        self.assertEqual(trace["fusion"]["method"], "rrf")
        for key in ("queryEmbeddingMs", "vectorSearchMs", "fusionMs", "totalMs"):
            self.assertIn(key, trace["timings"])

    def test_both_signals_beat_one(self):
        both = self.note("吵架了就先哄哄我。")
        lexical_only = self.note("吵架之后记得写日记。")
        vector_only = self.note("我发火的时候请先安慰我。")
        ids = [r["id"] for r in self.recall("吵架了要抱我")["results"]]
        self.assertEqual(ids[0], both)
        self.assertIn(vector_only, ids)
        self.assertIn(lexical_only, ids)

    def test_rrf_fuses_ranks_not_raw_scores(self):
        a = self.note("海边的潮水很好听。")
        debug = self.svc.retrieval_debug({"query": "潮水"})
        hit = next(c for c in debug["trace"]["candidates"] if c["hitId"] == a)
        k = debug["trace"]["fusion"]["k"]
        expected = 1.0 / (k + hit["signals"]["lexical"]["rank"]) + 1.0 / (k + hit["signals"]["vector"]["rank"])
        if hit["signals"]["entity"]["rank"]:
            expected += 0.5 / (k + hit["signals"]["entity"]["rank"])
        self.assertAlmostEqual(hit["fusion"]["fusedScore"], expected, places=5)

    def test_hard_negatives_and_no_topic_queries_stay_empty(self):
        self.note("吵架的时候先抱我。")
        for i in range(12):
            self.raw(f"User 今天觉得有点累 {i}")
        self.svc.vectors.wait_idle(20)
        for query in ("User 今天觉得", "今天觉得喜欢", "我们今天"):
            result = self.svc.retrieval_debug({"query": query})
            self.assertEqual(result["top"], [], query)
            self.assertIn(result["debug"]["vector"]["state"], ("skipped: no informative query terms",), query)

    def test_inject_needs_a_higher_similarity_than_recall(self):
        fight = self.note("我们吵架的时候，先抱住我，别讲道理。")
        inject = lambda q: [h["hitId"] for h in self.svc.retrieval_debug({"query": q, "policy": "inject"})["top"]]  # noqa: E731
        self.assertIn(fight, inject("又闹别扭了，好气"))
        self.assertEqual(inject("今天午饭吃了面条"), [])

    def test_entity_signal_ranks_named_entities(self):
        cat = self.note("User 每天喂楼下那只橘猫。", entities=["年糕", "橘猫"])
        trace = self.svc.retrieval_debug({"query": "年糕"})["trace"]
        hit = next(c for c in trace["candidates"] if c["hitId"] == cat)
        self.assertEqual(hit["signals"]["entity"]["matchedEntities"], ["年糕"])
        self.assertEqual(hit["signals"]["entity"]["rank"], 1)
        self.assertIn("年糕", trace["entity"]["queryEntities"])

    def test_lifecycle_exclusions_hold_for_the_vector_signal(self):
        v1 = self.note("吵架的时候给我一点时间冷静。")
        self.memory.store.update_episode(v1, {"content": "吵架的时候直接抱住我就好。"}, actor="t", reason="t")
        self.svc.vectors.wait_idle(20)
        found = self.recall("闹别扭了怎么办")["results"]
        self.assertEqual([r["id"] for r in found][:1], [v1])
        self.assertIn("直接抱住", found[0]["text"])
        self.assertNotIn("冷静", found[0]["text"])  # only the current version is indexed
        self.memory.store.update_episode(v1, {"status": "archived"}, actor="t", reason="t")
        self.svc.vectors.wait_idle(20)
        after = self.recall("闹别扭了怎么办")["results"]
        self.assertNotIn(v1, [r["id"] for r in after])
        self.assertEqual({r["kind"] for r in after}, {"ORIGINAL"})  # the RAW it came from is immutable and stays searchable
        self.assertNotIn(v1, self.svc.vectors.vectors)

    def test_staging_candidates_are_never_embedded_or_retrieved(self):
        rid = self.raw("吵架时要哄我")
        with self.memory.store.tx() as db:
            db.execute("INSERT INTO candidates (candidate_id, origin, gist, raw_ids, digest, status, created_at, updated_at) VALUES ('cand_x','t','吵架要抱','[]','d','STAGING','x','x')")
        self.svc.vectors.wait_idle(20)
        self.assertNotIn("cand_x", self.svc.vectors.items)
        self.assertNotIn("cand_x", [r["id"] for r in self.recall("吵架")["results"]])
        self.assertIn(rid, [r["id"] for r in self.recall("吵架")["results"]])  # the RAW itself is searchable

    def test_current_session_raw_is_excluded_even_when_found_semantically(self):
        old = self.raw("那次争执之后你抱了我很久", at="2026-08-01T10:00:00Z")
        new = self.raw("刚刚又冷战了", at="2026-09-24T11:00:00Z")
        self.svc.vectors.wait_idle(20)
        since = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)
        ids = [r["id"] for r in self.recall("我们闹别扭", current_session=(CONV, since))["results"]]
        self.assertIn(old, ids)
        self.assertNotIn(new, ids)

    def test_ablation_switches(self):
        fight = self.note("我们吵架的时候，先抱住我。")
        lexical_only = self.svc.retrieval.retrieve(RetrievalRequest(query="闹别扭了", signals=("lexical",)))
        self.assertEqual(lexical_only["top"], [])
        hybrid = self.svc.retrieval.retrieve(RetrievalRequest(query="闹别扭了"))
        self.assertEqual([h["hitId"] for h in hybrid["top"]][:1], [fight])


class SlowModel(ConceptEmbeddings):
    """Like the local bge-m3: it does not know its dimensions (so its cache key) until it has loaded."""

    def __init__(self):
        super().__init__()
        self.loaded = False
        self.dimensions = 0

    def ready(self):
        return self.loaded

    def load(self):
        self.dimensions = len(CONCEPTS) + 8
        self.loaded = True


class StartupCacheTest(Base):
    def test_a_restart_reads_the_cache_once_the_model_is_ready_and_embeds_nothing_again(self):
        a = self.note("海边的潮水声")
        self.raw("今天去看海了")
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.provider.calls, 3)
        self.svc.close()
        slow = SlowModel()
        self.open(slow)  # the model is still loading while the service starts and reloads its index
        self.assertEqual(slow.calls, 0)
        self.assertEqual(self.svc.vectors.stats()["pending"], 0)  # nothing is queued behind a model that cannot run yet
        slow.load()
        self.svc.vectors._wake.set()
        self.svc.vectors.wait_idle(20)
        self.assertEqual(slow.calls, 0, "the cached vectors were used, not recomputed")
        self.assertIn(a, self.svc.vectors.vectors)
        self.assertEqual(self.recall("潮水")["results"][0]["id"], a)


class CacheTest(Base):
    def test_cache_reuse_invalidation_and_restart(self):
        a = self.note("海边的潮水声")
        self.raw("今天去看海了")
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.provider.calls, 3)  # the note's RAW, the note's Episode, the chat RAW
        # restart: nothing is embedded again, the vectors come back from the cache
        self.svc.close()
        self.provider = ConceptEmbeddings()
        self.open()
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.provider.calls, 0)
        self.assertEqual(self.recall("潮水")["results"][0]["id"], a)
        stats = self.svc.vectors.stats()
        self.assertEqual((stats["current"], stats["embedded"], stats["pending"]), (3, 3, 0))
        # an index rebuild does not touch the cache either
        self.svc.rebuild()
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.provider.calls, 0)
        # a new text version is embedded once
        self.memory.store.update_episode(a, {"content": "海边的浪声"}, actor="t", reason="t")
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.provider.calls, 1)
        # a different model / schema invalidates every vector
        self.svc.close()
        other = ConceptEmbeddings()
        other.model_id = "concepts-v2"
        self.open(other)
        self.svc.vectors.wait_idle(20)
        self.assertEqual(other.calls, 3)
        self.assertGreaterEqual(self.svc.vectors.stats()["cache"]["staleRows"], 3)
        self.assertGreaterEqual(self.svc.vectors.purge_stale(), 3)


if __name__ == "__main__":
    unittest.main()
