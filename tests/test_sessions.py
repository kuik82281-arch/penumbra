"""Session segmentation: a conversation is cut where it went quiet for half an hour; one segment = one candidate;
the previous segment's end rides along as context; an ongoing chat waits; nothing is cut twice."""
import json
from datetime import datetime, timezone

from tests.support import Base, episode_answer, run_memory, wrap


class SessionTest(Base):
    def setUp(self):
        super().setUp()
        self.memory.pipeline.segmentation = "session"
        self.seen_payloads = []

        def decide(payload):
            self.seen_payloads.append(payload)
            return wrap(episode_answer(payload, "你说周末想去海边，我答应陪你去看日落。"))
        self.deepseek.decide = decide

    def candidates(self):
        return self.memory.store.all("SELECT gist, raw_ids, context_raw_ids, origin FROM candidates ORDER BY created_at")

    def test_half_an_hour_of_silence_ends_a_segment_and_each_segment_is_one_candidate(self):
        a = [self.raw("周末想去海边", at="2026-09-01T10:00:00Z"), self.raw("好啊", at="2026-09-01T10:01:00Z", role="assistant"),
             self.raw("嗯嗯", at="2026-09-01T10:20:00Z")]
        b = [self.raw("我回来啦", at="2026-09-01T11:00:00Z"), self.raw("刚才说到海边", at="2026-09-01T11:02:00Z")]
        run_memory(self)
        rows = self.candidates()
        self.assertEqual([json.loads(r["raw_ids"]) for r in rows], [a, b], "no single-message or overlapping candidates")
        self.assertTrue(all(r["origin"] == "session" for r in rows))
        self.assertEqual(json.loads(rows[1]["context_raw_ids"]), a, "the previous segment's end is the link")
        self.assertIn("接上一段", rows[1]["gist"])
        self.assertEqual(self.seen_payloads[1]["candidate"]["found_by"], "session")
        run_memory(self)
        self.assertEqual(len(self.candidates()), 2, "nothing is cut twice")

    def test_an_ongoing_chat_waits_until_it_has_been_quiet(self):
        now = datetime.now(timezone.utc)
        recent = (now.replace(microsecond=0)).isoformat().replace("+00:00", "Z")
        self.raw("还在聊呢", at=recent)
        run_memory(self)
        self.assertEqual(self.candidates(), [])
        sessions = self.memory.pipeline.build_sessions(self.memory.pipeline.unprocessed_raw(), now=datetime(2099, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(len(sessions), 1, "once it has gone quiet it is taken")

    def test_a_very_long_segment_splits_at_its_longest_pause(self):
        p = self.memory.pipeline
        msgs = [{"id": f"r{i}", "conversationId": "c", "content": "x" * 10, "createdAt": f"2026-09-01T10:{i % 60:02d}:00Z" if i < 60 else f"2026-09-01T11:{i - 60:02d}:30Z"}
                for i in range(100)]
        msgs[60]["createdAt"] = "2026-09-01T10:59:59Z"
        msgs[50]["createdAt"] = "2026-09-01T10:50:00Z"
        for i in range(51, 60):
            msgs[i]["createdAt"] = f"2026-09-01T10:{59 - (59 - i) // 10:02d}:{i:02d}Z"
        parts = p._split_long(msgs)
        self.assertGreater(len(parts), 1)
        self.assertEqual(sum(len(x) for x in parts), 100)
        self.assertTrue(all(len(x) <= 80 for x in parts))
