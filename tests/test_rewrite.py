"""Query rewrite: a local model says what a message means; it only adds to the search, never replaces or blocks it."""
import json
import tests.support  # noqa: F401 - no real model in tests
import unittest

from penumbra.memory import read
from penumbra.memory.rewrite import QueryRewriter


class FakeOllama:
    def __init__(self, answer, loaded=True):
        self.answer, self.loaded, self.timeout_s, self.keep_alive, self.calls = answer, loaded, 0, "", 0

    def health(self):
        return {"modelInstalled": True, "modelLoaded": self.loaded}

    def chat(self, system, user, schema, num_predict=0):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return json.dumps(self.answer, ensure_ascii=False), {}


RECENT = [{"role": "user", "content": "突然想起之前找工作的时候"}, {"role": "assistant", "content": "星屿那次面试？"}]


class RewriteTest(unittest.TestCase):
    def test_a_pronoun_is_resolved_and_invented_keywords_are_dropped(self):
        rw = QueryRewriter(FakeOllama({"topic": "星屿面试的结果", "keywords": ["星屿", "腾讯", "面试"], "image": False}))
        out = rw.rewrite("那个最后怎么样了", RECENT)
        self.assertEqual(out["topic"], "星屿面试的结果")
        self.assertEqual(out["keywords"], ["星屿", "面试"], "腾讯 never appears in the conversation")

    def test_any_failure_is_no_rewrite(self):
        self.assertIsNone(QueryRewriter(FakeOllama(RuntimeError("down"))).rewrite("那个呢", RECENT))
        self.assertIsNone(QueryRewriter(FakeOllama({"topic": "", "keywords": [], "image": False})).rewrite("那个呢", RECENT))

    def test_a_cold_model_is_not_waited_for(self):
        fake = FakeOllama({"topic": "x", "keywords": [], "image": False}, loaded=False)
        rw = QueryRewriter(fake)
        rw.warm = lambda: None
        self.assertIsNone(rw.rewrite("那个呢", RECENT))
        self.assertEqual(fake.calls, 0)


class FuseTest(unittest.TestCase):
    def test_found_by_both_rises_found_only_by_the_rewrite_is_discounted(self):
        original = [{"hitId": "a", "score": 0.5, "signals": {}}, {"hitId": "b", "score": 0.4, "signals": {}}]
        rewritten = [{"hitId": "b", "score": 0.6, "signals": {}}, {"hitId": "c", "score": 0.9, "signals": {}}]
        fused, counts = read._fuse(original, rewritten)
        scores = {h["hitId"]: h["score"] for h in fused}
        self.assertEqual(counts, {"both": 1, "rewrite_only": 1})
        self.assertAlmostEqual(scores["b"], round(0.6 * read.BOTH_BOOST, 4))
        self.assertAlmostEqual(scores["c"], round(0.9 * read.REWRITE_ONLY, 4))
        self.assertEqual(scores["a"], 0.5, "what the original found keeps its score")
        self.assertTrue(fused[0]["signals"].get("rewrite_only"))


if __name__ == "__main__":
    unittest.main()


class ReferentTest(unittest.TestCase):
    def test_a_message_pointing_back_is_caught_a_clear_one_is_not(self):
        from penumbra.memory.rewrite import REFERENT, mode
        for text in ["那个最后怎么样了", "这块我之前是不是老出错", "它还是那么好吃", "就是你说的那家"]:
            self.assertTrue(REFERENT.search(text), text)
        for text in ["我驾照考到哪一步了", "今天好累", "我最爱吃什么"]:
            self.assertFalse(REFERENT.search(text), text)
        self.assertEqual(mode(), "off", "tests run without it (tests/support.py)")
