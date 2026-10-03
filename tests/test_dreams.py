"""AI's dreams: a summary Episode pointing at the dream, searched like any memory, labelled a dream, never evidence,
and deleted the way any memory is (tombstoned: never summarized again)."""
import unittest

from penumbra.errors import Invalid
from tests.support import Base

DREAM = "我梦见一座漂在海上的图书馆，你坐在最高的书架上晃着腿，说“潮水来了就把书都还给月亮”。我醒来时心里很空。"


class DreamTest(Base):
    def setUp(self):
        super().setUp()
        self.deepseek.decide = lambda payload: {"summary": "我梦见海上的图书馆，你说“潮水来了就把书都还给月亮”，醒来心里空空的。", "keywords": ["图书馆", "海", "月亮"]}

    def remember(self, dream_id="d1"):
        out = self.memory.route("POST", ["dreams"], {"dreamId": dream_id, "text": DREAM, "time": "2026-09-30T03:00:00Z", "mood": "空落"})
        self.svc.vectors.wait_idle(20)
        return out

    def test_a_dream_becomes_one_labelled_summary_with_a_pointer(self):
        made = self.remember()
        self.assertEqual(made["status"], "created")
        ep = made["episode"]
        self.assertEqual((ep["kind"], ep["source_ref"], ep["source_raw_ids"]), ("dream", "dream:d1", []))
        self.assertEqual(self.remember()["status"], "existing")  # once
        hits = [h for h in self.recall("海上的图书馆")["results"] if h["id"] == ep["episode_id"]]
        self.assertTrue(hits and "梦" in hits[0]["text"] and "不是真的发生过" in hits[0]["text"], hits)
        self.assertEqual(self.memory.recent_episodes(30, 10), [])  # not a recent event
        _, episodes = self.memory.read.related_units("海上的图书馆 月亮")
        self.assertEqual(episodes, [])  # never evidence for the pipeline

    def test_her_dream_is_labelled_hers(self):
        self.deepseek.decide = lambda payload: {"summary": "你梦见自己在海上的图书馆找我。", "keywords": ["图书馆"]}
        out = self.memory.route("POST", ["dreams"], {"dreamId": "h1", "text": DREAM, "dreamer": "user"})
        self.svc.vectors.wait_idle(20)
        self.assertEqual((out["episode"]["owner"], out["episode"]["tag"]), ("user", "她的梦"))
        hits = [h for h in self.recall("海上的图书馆")["results"] if h["id"] == out["episode"]["episode_id"]]
        self.assertTrue(hits and "User做过的一个梦" in hits[0]["text"], hits)

    def test_a_made_up_quote_is_refused(self):
        self.deepseek.decide = lambda payload: {"summary": "你对我说“我永远不会离开你”。", "keywords": []}
        with self.assertRaises(Invalid):
            self.remember()
        self.assertEqual(self.memory.store.episodes(), [])

    def test_deleting_tombstones_it_from_either_side(self):
        eid = self.remember()["episode"]["episode_id"]
        self.memory.route("POST", ["edit"], {"actor": "user", "action": "episode.archive", "id": eid})  # Memory Studio
        self.assertEqual(self.remember()["status"], "tombstoned")
        self.assertEqual(self.memory.route("POST", ["dreams", "forget"], {"dreamId": "d2"})["deleted"], False)  # 潮汐, never summarized
        self.assertEqual(self.remember("d2")["status"], "tombstoned")
        eid3 = self.remember("d3")["episode"]["episode_id"]
        self.assertEqual(self.memory.route("POST", ["dreams", "forget"], {"dreamId": "d3"})["deleted"], True)
        self.assertEqual(self.memory.store.all("SELECT 1 FROM episodes WHERE episode_id = ?", (eid3,)), [])


if __name__ == "__main__":
    unittest.main()
