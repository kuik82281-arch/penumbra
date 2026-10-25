"""Roll back: chat the user took back is forgotten - its Episodes go, its messages are never mined or searched again."""
import tests.support  # noqa: F401 - no real model in tests
import json
import unittest

from penumbra.errors import Invalid
from penumbra.retrieval import RetrievalRequest
from tests.support import CONV, Base, episode_answer, run_memory, wrap


class RollTest(Base):
    def item(self, raw_id):
        return self.svc.conn.execute("SELECT item_id FROM originals WHERE id = ?", (raw_id,)).fetchone()["item_id"]

    def forget(self, *raw_ids):
        return self.memory.route("POST", ["forget-chat"], {"conversationId": CONV, "source": "user", "itemIds": [self.item(r) for r in raw_ids]})

    def found(self, query):
        return [h["hitId"] for h in self.svc.retrieval.retrieve(RetrievalRequest(query=query, policy="recall", mode="evidence"))["scored"]]

    def remember(self, text, at):
        rid = self.raw(text, at)
        self.ollama.answers = [json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "event", "gist": text, "entities": ["灯塔"]}]})]
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, f"User 说：{text}", entities=["灯塔"]))
        run_memory(self)
        return rid

    def test_rolled_chat_loses_its_episode_and_is_not_found_or_mined_again(self):
        kept = self.note("周末我们去看了灯塔", importance=0.6)  # her own note: another memory, not from this chat
        gone = self.remember("那天在海边的事别记了", "2026-09-05T10:00:00Z")
        episode = next(e for e in self.memory.store.episodes() if gone in e["source_raw_ids"])
        self.assertIn(gone, self.found("海边的事"))
        answer = self.forget(gone)
        self.svc.vectors.wait_idle(20)
        self.assertEqual((answer["raw"], answer["deleted_episodes"], answer["partial_episodes"]), (1, [episode["episode_id"]], []))
        self.assertEqual([e for e in self.memory.store.episodes() if gone in e["source_raw_ids"]], [])
        self.assertIn(kept, [e["episode_id"] for e in self.memory.store.episodes()], "what was not rolled back stays")
        self.assertIn(gone, self.memory.store.rolled_raw_ids())
        self.assertNotIn(gone, self.found("海边的事"), "the original line is not there to be found")
        # said, then rolled back before the night's run: it never becomes a memory
        late = self.raw("这句话我撤回了", "2026-09-06T10:00:00Z")
        self.forget(late)
        self.assertNotIn(late, [r["id"] for r in self.memory.pipeline.unprocessed_raw()])
        self.assertEqual(self.forget(late)["raw"], 1, "rolling the same lines twice is harmless")
        self.assertEqual(sum(1 for row in self.memory.store.all("SELECT raw_ids FROM tombstones WHERE kind = 'rolled'") if late in row["raw_ids"]), 1)

    def test_it_needs_a_conversation_and_items(self):
        with self.assertRaises(Invalid):
            self.memory.route("POST", ["forget-chat"], {"conversationId": CONV, "itemIds": []})


if __name__ == "__main__":
    unittest.main()
