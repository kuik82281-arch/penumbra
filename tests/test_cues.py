"""Recall cues: written per Episode version, checked against the RAW, searchable, never bumping the version."""
from penumbra.memory import cues

from support import Base


class Scripted:
    def __init__(self, answer):
        self.answer = answer

    def chat_json(self, system, user, max_tokens=400):
        return self.answer, {}


class CuesTest(Base):
    def test_cues_are_written_checked_and_searchable(self):
        rid = self.raw("拿铁好好喝，可惜你喝不到", at="2026-09-01T10:00:00Z")
        ep = self.memory.store.insert_episode({"content": "你喝到一杯好喝的拿铁，可惜我喝不到。", "time_start": "2026-09-01T10:00:00Z", "time_end": "2026-09-01T10:00:00Z",
                                               "source_raw_ids": [rid], "origin": "test"}, actor="t", reason="t")
        client = Scripted({"cues": [{"type": "处境", "text": "你喜欢的东西想分给没法亲自体验的我"},
                                    {"type": "细节", "text": "你说“月亮好圆我帮你看”"},  # quotes words nobody said: dropped
                                    {"type": "瞎编", "text": "不认识的类型"}]})
        self.assertEqual(cues.run(self.memory.store, self.memory.raw_record, client), {"written": 1, "failed": 0})
        after = self.memory.store.episode(ep["episode_id"])
        self.assertEqual(after["cues"], "处境：你喜欢的东西想分给没法亲自体验的我")
        self.assertEqual((after["version"], after["cues_for"]), (ep["version"], ep["version"]))
        self.assertEqual(cues.pending(self.memory.store, 10), [])  # nothing left until the Episode changes
        self.memory.store.update_episode(ep["episode_id"], {"state": "想分享"}, actor="t", reason="t")
        self.assertEqual([e["episode_id"] for e in cues.pending(self.memory.store, 10)], [ep["episode_id"]])
