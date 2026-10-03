"""Promises and plans: a daily commitment is tagged, stays on AI's to-do list until it is done, then sinks; a missed one
is shown to him once; a vow never sinks. DeepSeek is a scripted stand-in (tests/support.py)."""
import json
import unittest
from datetime import datetime, timezone

from penumbra.memory import commitments
from tests.support import Base, episode_answer, run_memory, wrap

CAND = lambda gist: json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "promise", "gist": gist, "entities": []}]})  # noqa: E731
TIE = {"kind": "commitment", "tag": "日常承诺·10月1日·打领带", "owner": "AI", "due_at": "2026-10-01T23:59:00+08:00"}


class CommitmentTest(Base):
    def promise(self, text="明天给你打暗红领带，半温莎结", at="2026-09-30T13:56:00Z", **fields):
        self.raw(text, at=at, role="assistant")
        self.ollama.answers = [CAND(text)]
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, "AI 答应 10 月 1 日早上给 User 打暗红领带，半温莎结，配灰色西装。", importance=0.4) | (fields or TIE))
        run_memory(self)
        return self.memory.store.episodes()[-1]

    def test_a_promise_is_tagged_open_and_on_the_assistants_list(self):
        ep = self.promise()
        self.assertEqual((ep["kind"], ep["tag"], ep["owner"], ep["commit_status"]), ("commitment", "日常承诺·10月1日·打领带", "AI", "open"))
        listed = commitments.for_assistant(self.memory, datetime(2026, 10, 1, 1, tzinfo=timezone.utc))
        self.assertEqual([c["tag"] for c in listed["open"]], ["日常承诺·10月1日·打领带"])
        self.assertNotIn(ep["episode_id"], [e["episode_id"] for e in self.memory.recent_episodes(30)])  # not an "event" of the handoff

    def test_done_updates_the_same_commitment_and_it_sinks(self):
        ep = self.promise()
        self.raw("领带打好了，半温莎", at="2026-10-01T00:30:00Z", role="assistant")
        self.ollama.answers = [CAND("领带打好了")]

        def decide(payload):
            open_ = payload["open_commitments"]
            assert [c["tag"] for c in open_] == ["日常承诺·10月1日·打领带"], "DeepSeek must see the open commitment"
            return wrap({"action": "UPDATE_EPISODE", "episode_id": open_[0]["episode_id"], "patch": {"commit_status": "done"},
                         "source_raw_ids": [payload["RAW"][0]["id"]]})
        self.deepseek.decide = decide
        run_memory(self)
        done = self.memory.store.episode(ep["episode_id"])
        self.assertEqual(done["commit_status"], "done")
        self.assertTrue(done["resolved_at"])
        self.assertLessEqual(done["importance"], 0.15)
        self.assertEqual(len(done["source_raw_ids"]), 2)
        self.assertEqual(self.memory.store.counts()["episodes"], 1)  # updated, never a second copy
        self.assertEqual(commitments.for_assistant(self.memory, datetime(2026, 10, 2, tzinfo=timezone.utc)), {"open": [], "missed": []})

    def test_the_same_promise_again_is_one_commitment(self):
        first = self.promise()
        self.promise("说好了，明天我给你打领带", at="2026-09-30T14:05:00Z")
        eps = self.memory.store.episodes()
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["episode_id"], first["episode_id"])
        self.assertEqual(len(eps[0]["source_raw_ids"]), 2)

    def test_missed_is_shown_once_then_sinks(self):
        ep = self.promise()
        early = datetime(2026, 10, 1, 18, tzinfo=timezone.utc)  # due 15:59Z + 6h grace not over yet
        self.assertEqual(commitments.for_assistant(self.memory, early)["missed"], [])
        later = datetime(2026, 10, 2, 0, tzinfo=timezone.utc)
        listed = commitments.for_assistant(self.memory, later)
        self.assertEqual([c["episode_id"] for c in listed["missed"]], [ep["episode_id"]])
        self.assertEqual(commitments.mark_told(self.memory, [ep["episode_id"]]), [ep["episode_id"]])
        self.assertEqual(commitments.for_assistant(self.memory, later), {"open": [], "missed": []})
        sunk = self.memory.store.episode(ep["episode_id"])
        self.assertEqual(sunk["commit_status"], "missed")
        self.assertTrue(sunk["missed_told_at"])
        self.assertEqual(sunk["status"], "active")  # sunk, not archived: still in memory

    def test_not_missed_while_the_day_is_still_unread(self):
        ep = self.promise()
        self.raw("领带打好了", at="2026-10-01T00:30:00Z", role="assistant")  # not consolidated yet: it may say "done"
        listed = commitments.for_assistant(self.memory, datetime(2026, 10, 2, 12, tzinfo=timezone.utc))
        self.assertEqual(listed["missed"], [])
        self.assertEqual(self.memory.store.episode(ep["episode_id"])["commit_status"], "open")

    def test_a_vow_is_important_and_never_due(self):
        ep = self.promise("以后每年生日我都陪你过", kind="vow", owner="AI")
        self.assertEqual((ep["kind"], ep["commit_status"], ep["due_at"]), ("vow", "", None))
        self.assertGreaterEqual(ep["importance"], 0.85)
        self.assertEqual(commitments.for_assistant(self.memory)["open"], [])

    def test_the_user_can_turn_a_commitment_into_a_vow_and_back(self):
        ep = self.promise()
        self.memory.route("POST", ["edit"], {"actor": "user", "action": "episode.update", "id": ep["episode_id"], "patch": {"kind": "vow"}})
        vow = self.memory.store.episode(ep["episode_id"])
        self.assertEqual((vow["kind"], vow["commit_status"], vow["due_at"]), ("vow", "", None))
        self.memory.route("POST", ["edit"], {"actor": "user", "action": "episode.update", "id": ep["episode_id"], "patch": {"kind": "commitment"}})
        back = self.memory.store.episode(ep["episode_id"])
        self.assertEqual((back["kind"], back["commit_status"]), ("commitment", "open"))
        self.assertTrue(back["tag"] and back["due_at"])
        self.memory.route("POST", ["edit"], {"actor": "user", "action": "episode.update", "id": ep["episode_id"], "patch": {"commit_status": "done"}})
        self.assertEqual(self.memory.store.episode(ep["episode_id"])["commit_status"], "done")

    def test_an_old_database_gains_the_new_columns(self):
        self.svc.close()
        self.open()  # reopening runs the migration again: idempotent
        self.promise()
        self.assertEqual(self.memory.store.episodes()[0]["commit_status"], "open")


if __name__ == "__main__":
    unittest.main()
