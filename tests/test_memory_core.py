"""The unified memory: RAW -> Candidate Discovery -> STAGING -> DeepSeek Verification -> Episode -> Pattern, and its read path.
Ollama and DeepSeek are scripted stand-ins (tests/support.py); the real models are exercised by the live end-to-end check."""
import json
import threading
import unittest
import urllib.request
from datetime import datetime, timedelta, timezone
from unittest import mock

from penumbra.api import make_handler
from penumbra.errors import Invalid, WorkerError
from penumbra.memory import actions
from penumbra.memory.llm import AnswerError, complete_objects, parse_json_answer
from tests.support import CONV, NOW, Base, FakeDeepSeek, FakeOllama, build, episode_answer, run_memory, wrap

CAND = lambda **kw: json.dumps({"candidates": [{"from": 1, "to": 1, "kind": "state_change", "gist": "g", "entities": [], **kw}]})  # noqa: E731


def statuses(base):
    return {r["status"]: r["n"] for r in base.memory.store.all("SELECT status, COUNT(*) AS n FROM candidates GROUP BY status")}


class ParsingTest(unittest.TestCase):
    def test_answer_extraction_is_tolerant_and_never_repairs(self):
        self.assertEqual(parse_json_answer('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_answer('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_answer('好的：{"a": [1, 2]} 以上'), {"a": [1, 2]})
        with self.assertRaises(AnswerError) as cut:
            parse_json_answer('{"candidates": [{"from": 1', "length")
        self.assertEqual(cut.exception.kind, "truncated")
        with self.assertRaises(AnswerError) as empty:
            parse_json_answer("   ")
        self.assertEqual(empty.exception.kind, "empty")

    def test_complete_objects_salvages_the_finished_elements(self):
        text = '{"candidates": [{"from": 1, "to": 2, "kind": "plan", "gist": "a", "entities": []}, {"from": 3, "to": 4, "kin'
        got = complete_objects(text, "candidates")
        self.assertEqual([c["from"] for c in got], [1])


class SchedulerTest(Base):
    """The idle / night / morning scheduler reads the last activity MemoryCore.activity() writes (defaults: 30 idle
    minutes, night 22-06 and morning 8-9 in UTC+8)."""

    ACTIVE = "2026-09-01T14:00:00.000Z"  # 22:00 in UTC+8: inside the night window

    def setUp(self):
        super().setUp()
        self.raw("我们约好下周三一起去看医生", at="2026-09-01T13:50:00Z")  # unprocessed RAW: there is work
        self.deepseek.configured = False
        self.touch(self.ACTIVE)

    def touch(self, stamp):
        """Record activity through the real writer, at a fixed time."""
        with mock.patch("penumbra.memory.now_iso", return_value=stamp):
            self.memory.activity()

    def tick(self, minutes_after_activity, activity=ACTIVE):
        now = datetime.fromisoformat(activity.replace("Z", "+00:00")) + timedelta(minutes=minutes_after_activity)
        return self.memory.pipeline.tick(now=now, background=False)

    def test_recent_activity_means_no_batch(self):
        for minutes in (0, 10, 29):
            result = self.tick(minutes)
            self.assertFalse(result["triggered"], minutes)
            self.assertEqual(result["reason"], "active_or_outside_window")
        self.assertEqual(self.ollama.events, [])
        self.assertEqual(self.memory.store.all("SELECT COUNT(*) AS n FROM runs")[0]["n"], 0)

    def test_inactivity_over_the_threshold_in_the_night_window_runs_a_batch(self):
        result = self.tick(31)  # 22:31 local, 31 idle minutes
        self.assertTrue(result["triggered"])
        self.assertEqual(result["reason"], "night")
        self.assertEqual((result["result"]["status"], result["result"]["summary"]["windows"]), ("completed", 1))
        self.assertEqual(self.ollama.events, ["chat", "unload"])

    def test_the_morning_window_and_the_rest_of_the_day(self):
        self.touch("2026-09-01T23:00:00.000Z")  # 07:00 local
        early = self.tick(45, activity="2026-09-01T23:00:00.000Z")  # 07:45: idle, but no window
        self.assertEqual((early["triggered"], early["reason"]), (False, "active_or_outside_window"))
        morning = self.tick(70, activity="2026-09-01T23:00:00.000Z")  # 08:10: morning re-check
        self.assertEqual((morning["triggered"], morning["reason"]), (True, "morning_recheck"))

    def test_last_activity_persists_across_a_restart(self):
        self.svc.close()
        self.open()
        self.assertEqual(self.memory.store.kv_get("last_activity_at"), self.ACTIVE)
        self.assertEqual(self.memory._checkpoint()["last_activity_at"], self.ACTIVE)  # what Memory Studio shows
        self.assertFalse(self.tick(10)["triggered"])
        self.assertTrue(self.tick(31)["triggered"])
        # The bug this guards: the checkpoint never carries the activity time, so reading it there saw "just now".
        self.assertNotIn("last_activity_at", self.memory.store.kv_get("checkpoint", {}))

    def test_new_raw_counts_as_activity(self):
        with mock.patch("penumbra.memory.now_iso", return_value="2026-09-01T14:20:00.000Z"):
            self.raw("好，我记下来了", at="2026-09-01T14:20:00Z", role="assistant")
        self.assertFalse(self.tick(31)["triggered"])  # only 11 minutes after the new message

    def test_manual_runs_ignore_activity(self):
        self.touch(datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"))  # just now
        self.assertFalse(self.memory.pipeline.tick(background=False)["triggered"])
        result = self.memory.route("POST", ["run-sync"])  # Memory Studio's manual run
        self.assertEqual((result["status"], result["summary"]["windows"]), ("completed", 1))
        self.assertEqual(self.ollama.events, ["chat", "unload"])


class DiscoveryTest(Base):
    def window(self):
        return [self.raw("我们约好下周三一起去看医生", at="2026-09-01T10:00:00Z"), self.raw("好，我记下来了", at="2026-09-01T10:01:00Z", role="assistant")]

    def test_ollama_candidates_are_staged_and_never_written_as_memory(self):
        self.window()
        self.ollama.answers = [CAND(gist="约好下周三看医生")]
        self.deepseek.configured = False  # discovery alone must not create memory
        run_memory(self)
        self.assertEqual(self.memory.store.counts()["episodes"], 0)
        win = self.memory.store.all("SELECT * FROM windows")[0]
        self.assertEqual((win["mode"], win["provider"]), ("ollama", "ollama"))
        self.assertEqual((win["candidate_count"], statuses(self)), (1, {"STAGING": 1}))

    def test_ollama_down_falls_back_to_deterministic_chunks(self):
        self.window()
        self.ollama.ready, self.ollama.error = False, "connection refused"
        self.deepseek.configured = False
        result = run_memory(self)
        self.assertEqual(result["summary"]["modes"], {"deterministic": 1})
        self.assertEqual(result["errors"], [])
        self.assertEqual(statuses(self), {"STAGING": 1})  # awaiting DeepSeek, not lost

    def test_garbage_answer_is_retried_then_falls_back_without_looping(self):
        self.window()
        self.ollama.answers = ["not json at all", "still not json"]
        run_memory(self)
        win = self.memory.store.all("SELECT * FROM windows")[0]
        self.assertEqual(win["status"], "STAGED")  # the window is checkpointed even though Ollama failed
        self.assertEqual(win["mode"], "deterministic")
        calls = len(self.ollama.calls)
        run_memory(self)  # nothing new: Ollama is not asked again
        self.assertEqual(len(self.ollama.calls), calls)

    def test_one_load_serves_the_whole_run_and_is_released_after_it(self):
        self.window()
        self.raw("另一段对话里说好周末去海边", at="2026-09-03T10:00:00Z", conv="conv-2")  # a second window
        self.deepseek.configured = False
        result = run_memory(self)
        self.assertEqual(result["summary"]["windows"], 2)
        self.assertEqual(self.ollama.events, ["chat", "chat", "unload"])  # never unloaded between windows
        self.assertFalse(self.ollama.used)
        run_memory(self)  # nothing new to discover: nothing loaded, nothing to release
        self.assertEqual(self.ollama.events, ["chat", "chat", "unload"])

    def test_a_window_that_fails_after_discovery_still_releases_the_model(self):
        self.window()
        self.deepseek.configured = False
        original = self.memory.pipeline._stage_window
        def stage_then_fail(run_id, msgs):
            original(run_id, msgs)
            raise RuntimeError("boom after discovery")
        self.memory.pipeline._stage_window = stage_then_fail
        run_memory(self)
        self.assertEqual(self.ollama.events, ["chat", "unload"])

    def test_truncated_answer_keeps_its_complete_candidates(self):
        self.window()
        cut = '{"candidates": [{"from": 1, "to": 1, "kind": "plan", "gist": "约好下周三看医生", "entities": []}, {"from": 2, "to": 2, "kind": "pl'
        self.ollama.answers = [cut]
        self.deepseek.configured = False
        run_memory(self)
        win = self.memory.store.all("SELECT * FROM windows")[0]
        self.assertTrue(any("salvaged" in w for w in json.loads(win["warnings"])))
        self.assertGreaterEqual(win["candidate_count"], 1)

    def test_strong_signal_is_not_lost_when_ollama_says_nothing(self):
        self.raw("我决定下个月搬到杭州，已经订了下周三的车票", at="2026-09-01T10:00:00Z")
        self.ollama.answers = [json.dumps({"candidates": []})]
        self.deepseek.configured = False
        run_memory(self)
        win = self.memory.store.all("SELECT * FROM windows")[0]
        self.assertEqual(win["mode"], "ollama+safety")
        self.assertEqual(statuses(self), {"STAGING": 1})

    def test_attitude_statements_survive_a_silent_ollama(self):
        """Regression: qwen2.5:3b returned nothing for 'I used to hate X / I like X now'; the safety net must still stage them."""
        for i, text in enumerate(["我以前特别讨厌香菜，一闻到就想吐", "最近居然吃得下香菜了，觉得还行", "我现在特别喜欢香菜，火锅里都会加一大把"]):
            self.raw(text, at=f"2026-0{3 + 2 * i}-01T10:00:00Z")
        self.ollama.answers = [json.dumps({"candidates": []})]
        self.deepseek.configured = False
        run_memory(self)
        self.assertEqual(statuses(self), {"STAGING": 3})
        self.assertEqual({w["mode"] for w in self.memory.store.all("SELECT mode FROM windows")}, {"ollama+safety"})

    def test_reprocess_reads_empty_windows_again_without_duplicating_candidates(self):
        self.raw("我以前特别讨厌香菜", at="2026-03-01T10:00:00Z")  # one candidate
        self.raw("今天天气不错", at="2026-04-01T10:00:00Z", conv="other")  # none
        self.ollama.ready = False
        self.deepseek.configured = False
        run_memory(self)
        self.assertEqual(self.memory.pipeline.pending_work()["unprocessedRaw"], 0)
        self.assertEqual(self.memory.route("POST", ["reprocess"], {"conversationId": "other"}), {"forgottenWindows": 1, "requeued": 0})
        self.assertEqual(self.memory.pipeline.pending_work()["unprocessedRaw"], 1)
        # a window that produced a candidate is never forgotten
        self.assertEqual(self.memory.route("POST", ["reprocess"], {"conversationId": CONV}), {"forgottenWindows": 0, "requeued": 0})
        run_memory(self)
        self.assertEqual(len(self.memory.store.all("SELECT 1 FROM candidates")), 1)

    def test_reprocess_a_day_again_requeues_what_was_dropped_there(self):
        self.raw("明天给你买冰淇淋", at="2026-09-30T05:00:00Z")
        self.raw("下周给你买戒指", at="2026-09-20T05:00:00Z", conv="other")
        self.ollama.ready = False  # deterministic discovery: the promise signal finds both
        self.deepseek.decide = lambda payload: wrap({"action": "NO_ACTION", "reason": "old rules"})
        run_memory(self)
        self.assertEqual(statuses(self), {"NO_ACTION": 2})
        day = {"onlyEmpty": False, "since": "2026-09-30T00:00:00+08:00", "until": "2026-10-01T00:00:00+08:00"}
        self.assertEqual(self.memory.route("POST", ["reprocess"], day), {"forgottenWindows": 1, "requeued": 1})
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, "AI 答应 9 月 30 日给 User 买冰淇淋。", importance=0.4)
                                                    | {"kind": "commitment", "tag": "日常承诺·9月30日·冰淇淋", "owner": "AI", "due_at": "2026-09-30T23:59:00+08:00"})
        run_memory(self)
        self.assertEqual(statuses(self), {"NO_ACTION": 1, "APPLIED": 1})  # the other day was left alone
        self.assertEqual([e["tag"] for e in self.memory.store.episodes()], ["日常承诺·9月30日·冰淇淋"])
        self.assertEqual(len(self.memory.store.all("SELECT 1 FROM candidates")), 2)  # the same span found again is not a second candidate

    def test_chit_chat_produces_no_candidate(self):
        self.raw("早安呀", at="2026-09-01T10:00:00Z")
        self.raw("早安，今天想吃什么", at="2026-09-01T10:01:00Z", role="assistant")
        self.ollama.ready = False
        run_memory(self)
        self.assertEqual(statuses(self), {})


class VerificationTest(Base):
    def stage(self, text="我决定下个月搬到杭州", at="2026-09-01T10:00:00Z"):
        rid = self.raw(text, at=at)
        self.ollama.answers = [CAND(gist=text[:30])]
        return rid

    def test_create_episode_and_pattern_with_full_provenance(self):
        rid = self.stage("我以前特别讨厌香菜")

        def decide(payload):
            e = episode_answer(payload, "User 说她以前特别讨厌香菜。", state="讨厌香菜", entities=["香菜"], topics=["饮食"])
            p = {"action": "CREATE_PATTERN", "ref": "p1", "title": "User 对香菜的态度", "topic": "饮食偏好", "narrative": "User 以前特别讨厌香菜。", "current_state": "讨厌香菜",
                 "entities": ["香菜"], "confidence": 0.9, "supporting_episode_ids": ["e1"]}
            return wrap(e, p)
        self.deepseek.decide = decide
        run_memory(self)
        counts = self.memory.store.counts()
        self.assertEqual((counts["episodes"], counts["patterns"]), (1, 1))
        pattern = self.memory.store.patterns()[0]
        episode = self.memory.store.episodes()[0]
        self.assertEqual(episode["source_raw_ids"], [rid])
        self.assertEqual(pattern["supporting_episode_ids"], [episode["episode_id"]])
        chain = self.memory.provenance("pattern", pattern["pattern_id"])
        self.assertEqual(chain["pattern"]["pattern_id"], pattern["pattern_id"])
        link = chain["episodes"][0]
        self.assertEqual([link["episode"]["episode_id"]], [episode["episode_id"]])
        self.assertEqual(link["raw"][0]["id"], rid)  # Pattern -> Episode -> RAW
        self.assertEqual(link["candidate"]["status"], "APPLIED")  # ... and the candidate and DeepSeek decision that made it
        self.assertEqual(link["decision"]["status"], "APPLIED")
        self.assertEqual(chain["decisions"][0]["decision_id"], link["decision"]["decision_id"])
        self.assertEqual(statuses(self), {"APPLIED": 1})
        types = {(r["type"], r["src_kind"], r["dst_kind"]) for r in self.memory.store.all("SELECT * FROM relations")}
        self.assertIn(("supports", "episode", "pattern"), types)
        self.assertIn(("derived_from", "pattern", "episode"), types)
        self.assertIn(("involved_in", "entity", "episode"), types)

    def test_state_evolution_keeps_history_and_never_overwrites(self):
        stages = [("我以前特别讨厌香菜", "2026-03-01T10:00:00Z", "讨厌香菜"), ("最近居然吃得下香菜了", "2026-06-01T10:00:00Z", "接受香菜"),
                  ("现在我特别喜欢香菜", "2026-09-01T10:00:00Z", "喜欢香菜")]
        for i, (text, at, state) in enumerate(stages):
            self.stage(text, at)

            def decide(payload, i=i, state=state, text=text):
                e = episode_answer(payload, f"User 说：{text}", ref="e1", state=state, entities=["香菜"])
                if i == 0:
                    return wrap(e, {"action": "CREATE_PATTERN", "ref": "p1", "title": "User 对香菜的态度", "topic": "饮食", "narrative": "User 对香菜的态度。", "current_state": state,
                                    "entities": ["香菜"], "confidence": 0.9, "supporting_episode_ids": ["e1"]})
                existing = payload["existing"]["patterns"]
                assert existing, "DeepSeek must see the existing Pattern"
                return wrap(e, {"action": "UPDATE_PATTERN", "pattern_id": existing[0]["pattern_id"], "add_supporting": ["e1"],
                                "new_state": {"state": state, "valid_from": payload["RAW"][0]["createdAt"]}, "narrative": "User 对香菜的态度：讨厌→接受→喜欢。" if i == 2 else "User 对香菜的态度：讨厌→接受。"})
            self.deepseek.decide = decide
            run_memory(self)
        patterns = self.memory.store.patterns()
        self.assertEqual(len(patterns), 1)
        p = patterns[0]
        self.assertEqual(p["current_state"], "喜欢香菜")
        self.assertEqual([s["state"] for s in p["historical_states"]], ["讨厌香菜", "接受香菜"])
        self.assertTrue(all(s["valid_to"] for s in p["historical_states"]))
        self.assertEqual(len(p["supporting_episode_ids"]), 3)
        self.assertEqual(p["version"], 3)
        versions = self.memory.store.all("SELECT version FROM pattern_versions WHERE pattern_id = ? ORDER BY version", (p["pattern_id"],))
        self.assertEqual([v["version"] for v in versions], [1, 2, 3])
        supersedes = self.memory.store.all("SELECT * FROM relations WHERE type='supersedes'")
        self.assertEqual(len(supersedes), 2)

    def test_answer_citing_unknown_raw_is_quarantined_and_writes_nothing(self):
        self.stage()
        self.deepseek.decide = lambda payload: wrap({**episode_answer(payload, "搬去杭州。"), "source_raw_ids": ["o_invented"]})
        run_memory(self)
        self.assertEqual(statuses(self), {"QUARANTINE": 1})
        self.assertEqual(self.memory.store.counts()["episodes"], 0)
        decision = self.memory.store.all("SELECT * FROM decisions")[0]
        self.assertIn("RAW that was not provided", decision["errors"])

    def test_low_confidence_is_quarantined(self):
        self.stage()
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, "搬去杭州。"), confidence=0.4)
        run_memory(self)
        self.assertEqual(statuses(self), {"QUARANTINE": 1})
        self.assertEqual(self.memory.store.counts()["episodes"], 0)

    def test_terminal_actions_write_nothing(self):
        for action, status in (("NO_ACTION", "NO_ACTION"), ("REJECT", "REJECTED"), ("QUARANTINE", "QUARANTINE")):
            self.stage(f"我决定下个月搬家 {action}", at=f"2026-09-0{1 + ['NO_ACTION', 'REJECT', 'QUARANTINE'].index(action)}T10:00:00Z")
            self.deepseek.decide = lambda payload, a=action: wrap({"action": a, "reason": "test"})
            run_memory(self)
            self.assertIn(status, statuses(self))
        self.assertEqual(self.memory.store.counts()["episodes"], 0)

    def test_role_words_are_replaced_with_names(self):
        self.stage()
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, "用户说她要搬去杭州，助手表示支持。"))
        run_memory(self)
        self.assertEqual(self.memory.store.episodes()[0]["content"], "User说她要搬去杭州，AI表示支持。")

    def test_deepseek_unavailable_keeps_staging_without_burning_attempts(self):
        self.stage()
        self.deepseek.configured = False
        run_memory(self)
        row = self.memory.store.all("SELECT * FROM candidates")[0]
        self.assertEqual((row["status"], row["attempts"]), ("STAGING", 0))
        self.deepseek.configured = True
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, "User 决定下个月搬到杭州。"))
        run_memory(self)
        self.assertEqual(statuses(self), {"APPLIED": 1})

    def test_provider_error_backs_off_and_is_not_retried_in_a_loop(self):
        self.stage()
        self.deepseek.decide = lambda payload: WorkerError("DeepSeek 503")
        run_memory(self)
        row = self.memory.store.all("SELECT * FROM candidates")[0]
        self.assertEqual((row["status"], row["attempts"]), ("STAGING", 1))
        self.assertTrue(row["next_attempt_at"])
        calls = len(self.deepseek.payloads)
        run_memory(self)  # not due yet
        self.assertEqual(len(self.deepseek.payloads), calls)

    def test_repeated_failures_end_in_failed_not_forever(self):
        self.stage()
        self.deepseek.decide = lambda payload: WorkerError("DeepSeek 503")
        run_memory(self)
        for _ in range(8):
            with self.memory.store.tx() as db:
                db.execute("UPDATE candidates SET next_attempt_at = NULL")
            run_memory(self)
        self.assertEqual(statuses(self), {"FAILED": 1})

    def test_checkpoint_each_raw_is_read_once(self):
        self.stage()
        self.deepseek.decide = lambda payload: wrap(episode_answer(payload, "User 决定下个月搬到杭州。"))
        run_memory(self)
        run_memory(self)
        self.assertEqual(len(self.ollama.calls), 1)
        self.assertEqual(len(self.deepseek.payloads), 1)
        self.assertEqual(self.memory.pipeline.pending_work()["unprocessedRaw"], 0)

    def test_restart_forgets_a_half_done_window_and_reads_it_again(self):
        self.stage()
        with self.memory.store.tx() as db:
            db.execute("INSERT INTO windows (window_id, raw_ids, status, created_at, updated_at) VALUES ('w_x', '[]', 'DISCOVERING', 'x', 'x')")
        self.svc.close()
        self.open()
        self.assertIsNone(self.memory.store.one("SELECT 1 FROM windows WHERE window_id='w_x'"))


class ActionsTest(Base):
    def test_merge_episodes_repoints_patterns_and_keeps_sources(self):
        a, b = self.note("User 周三去了医院。", time="2026-09-02T10:00:00Z"), self.note("User 周三去医院复查了。", time="2026-09-02T11:00:00Z")
        store = self.memory.store
        pat = store.insert_pattern({"title": "就医", "topic": "健康", "narrative": "就医。", "current_state": "复查中", "states": [], "supporting_episode_ids": [a, b],
                                    "entities": [], "confidence": 0.8}, actor="test", reason="t")
        plan = actions.validate_answer(wrap({"action": "MERGE_EPISODE", "ref": "m", "episode_ids": [a, b], "content": "User 周三去医院复查。", "time_start": "2026-09-02T10:00:00Z",
                                             "time_end": "2026-09-02T11:00:00Z", "entities": [], "topics": [], "state": "", "importance": 0.6, "confidence": 0.9}), set(), store)
        applied = actions.apply_plan(store, plan, None, "decision_t")
        merged = applied["episodes_merged"][0]["into"]
        self.assertEqual(store.episode(a)["status"], "merged")
        self.assertEqual(sorted(store.episode(merged)["source_raw_ids"]), sorted(store.episode(a)["source_raw_ids"] + store.episode(b)["source_raw_ids"]))
        self.assertEqual(store.pattern(pat["pattern_id"])["supporting_episode_ids"], [merged])

    def test_back_dated_state_is_history_and_a_same_time_conflict_is_refused(self):
        e = self.note("User 现在喜欢香菜。", time="2026-09-01T10:00:00Z")
        store = self.memory.store
        pat = store.insert_pattern({"title": "香菜", "topic": "饮食", "narrative": "香菜。", "current_state": "喜欢香菜", "current_state_since": "2026-09-01T10:00:00Z",
                                    "states": [{"state_id": "s1", "state": "喜欢香菜", "valid_from": "2026-09-01T10:00:00Z", "valid_to": None, "episode_ids": [e]}],
                                    "supporting_episode_ids": [e], "entities": [], "valid_from": "2026-09-01T10:00:00Z", "confidence": 0.8}, actor="t", reason="t")
        early = actions.validate_answer(wrap({"action": "UPDATE_PATTERN", "pattern_id": pat["pattern_id"], "new_state": {"state": "讨厌香菜", "valid_from": "2026-03-01T10:00:00Z"}}), set(), store)
        actions.apply_plan(store, early, None, "d1")
        p = store.pattern(pat["pattern_id"])
        self.assertEqual(p["current_state"], "喜欢香菜")  # the newer state stays current
        self.assertEqual([s["state"] for s in p["historical_states"]], ["讨厌香菜"])
        clash = actions.validate_answer(wrap({"action": "UPDATE_PATTERN", "pattern_id": pat["pattern_id"], "new_state": {"state": "无所谓", "valid_from": "2026-09-01T10:00:00Z"}}), set(), store)
        with self.assertRaises(Invalid):
            actions.apply_plan(store, clash, None, "d2")

    def test_common_answer_shapes_mean_the_same_thing(self):
        store = self.memory.store
        bare = actions.validate_answer({"action": "NO_ACTION", "reason": "闲聊"}, set(), store)  # no wrapper, no confidence
        self.assertEqual((bare.terminal, bare.confidence), ("NO_ACTION", 0.8))
        single = actions.validate_answer({"actions": {"action": "REJECT", "reason": "玩笑"}, "confidence": 0.9}, set(), store)  # actions is one object
        self.assertEqual(single.terminal, "REJECT")
        # ... but a writing action without any stated confidence is never waved through
        with self.assertRaises(Invalid):
            actions.validate_answer({"actions": [{"action": "CREATE_EPISODE", "ref": "e1"}]}, set(), store)
        with self.assertRaises(Invalid):
            actions.validate_answer({"nothing": "here"}, set(), store)

    def test_a_new_pattern_starts_with_the_history_its_episodes_record(self):
        store = self.memory.store
        early = self.note("User 说她以前特别讨厌香菜。", time="2026-03-01T10:00:00Z")
        store.update_episode(early, {"state": "讨厌香菜"}, actor="t", reason="t")
        mid = self.note("User 说最近吃得下香菜了。", time="2026-06-01T10:00:00Z")
        store.update_episode(mid, {"state": "接受香菜"}, actor="t", reason="t")
        late = self.note("User 说她现在特别喜欢香菜。", time="2026-09-01T10:00:00Z")
        store.update_episode(late, {"state": "喜欢香菜"}, actor="t", reason="t")

        def create(**extra):
            plan = actions.validate_answer(wrap({"action": "CREATE_PATTERN", "ref": "p", "title": "香菜", "topic": "饮食", "narrative": "香菜。", "current_state": "喜欢香菜",
                                                 "supporting_episode_ids": [early, mid, late], **extra}), set(), store)
            return store.pattern(actions.apply_plan(store, plan, None, "d")["patterns_created"][0])

        derived = create()  # the model said nothing about earlier states: they are read from the Episodes
        self.assertEqual([(s["state"], s["valid_to"] is None) for s in derived["states"]], [("讨厌香菜", False), ("接受香菜", False), ("喜欢香菜", True)])
        self.assertEqual([len(s["episode_ids"]) for s in derived["states"]], [1, 1, 1])
        self.assertEqual(derived["states"][0]["valid_to"], derived["states"][1]["valid_from"])  # no gaps, nothing overwritten
        self.assertEqual(len(store.all("SELECT 1 FROM relations WHERE type='supersedes'")), 2)
        stated = create(earlier_states=[{"state": "接受香菜", "valid_from": "2026-06-01T10:00:00Z"}, {"state": "讨厌香菜", "valid_from": "2026-03-01T10:00:00Z"}])
        self.assertEqual([s["state"] for s in stated["states"]], ["讨厌香菜", "接受香菜", "喜欢香菜"])  # ordered by time whatever order it came in
        # An "earlier" state that is not earlier than the current one is a slip in the dates: it is dropped, the Pattern
        # is still made (the whole decision used to be lost to QUARANTINE this way).
        made = actions.apply_plan(store, actions.validate_answer(wrap({"action": "CREATE_PATTERN", "ref": "p", "title": "x", "narrative": "x", "current_state": "y", "state_valid_from": "2026-01-01T00:00:00Z",
                                                                       "earlier_states": [{"state": "z", "valid_from": "2026-02-01T00:00:00Z"}], "supporting_episode_ids": [early]}), set(), store), None, "d")
        slipped = store.pattern(made["patterns_created"][0])
        self.assertEqual([s["state"] for s in slipped["states"]], ["y"])

    def test_terminal_action_must_be_alone_and_ids_must_exist(self):
        with self.assertRaises(Invalid):
            actions.validate_answer(wrap({"action": "NO_ACTION"}, {"action": "REJECT"}), set(), self.memory.store)
        with self.assertRaises(Invalid):
            actions.validate_answer(wrap({"action": "UPDATE_PATTERN", "pattern_id": "pattern_nope", "narrative": "x"}), set(), self.memory.store)


class ReadTest(Base):
    def setUp(self):
        super().setUp()
        self.fight = self.note("我们吵架的时候，先抱住我，别讲道理，等我冷静下来再说。", topics=["安抚"], importance=0.9)
        self.coffee = self.note("早上一定要一杯燕麦拿铁。")
        self.pat = self.memory.store.insert_pattern(
            {"title": "吵架后的安抚方式", "topic": "关系", "narrative": "User 吵架后希望先被抱住，不要讲道理。", "current_state": "先抱住再谈", "current_state_since": "2026-09-01T10:00:00Z",
             "states": [{"state_id": "s1", "state": "先抱住再谈", "valid_from": "2026-09-01T10:00:00Z", "valid_to": None, "episode_ids": [self.fight]}],
             "supporting_episode_ids": [self.fight], "entities": [], "valid_from": "2026-09-01T10:00:00Z", "confidence": 0.9}, actor="t", reason="t")
        self.memory.store.add_relation("supports", "episode", self.fight, "pattern", self.pat["pattern_id"])
        self.svc.vectors.wait_idle(20)

    def retrieve(self, query, turn="turn-1", **extra):
        return self.memory.read.retrieve({"query": query, "turnId": turn, "conversationId": CONV, "sessionId": "s1", **extra})

    def test_pattern_first_and_a_lock_is_created(self):
        r = self.retrieve("我们闹别扭了你该怎么做")
        self.assertEqual(r["status"], "LOCKED")
        self.assertEqual([p["pattern_id"] for p in r["patterns"]], [self.pat["pattern_id"]])
        self.assertEqual(r["episodes"], [])  # the Episode is under its Pattern, not beside it
        lock = self.memory.read.lock_for("turn-1")
        self.assertEqual(lock["locked_pattern_ids"], [self.pat["pattern_id"]])
        self.assertIn(self.fight, lock["locked_episode_ids"])

    def test_search_once_per_turn(self):
        first = self.retrieve("我们闹别扭了你该怎么做")
        again = self.retrieve("完全不同的另一句话：咖啡", turn="turn-1")
        self.assertTrue(again["reused_lock"] or again["status"] == "LOCKED")
        self.assertEqual([p["pattern_id"] for p in again["patterns"]], [p["pattern_id"] for p in first["patterns"]])
        self.assertEqual(again["inject_id"], first["inject_id"])  # the reused lock can still be confirmed
        self.assertEqual(again["refs"], first["refs"])
        row = self.memory.store.one("SELECT searches FROM locks WHERE turn_id='turn-1'")
        self.assertEqual(row["searches"], 1)
        follow = self.memory.read.recall({"query": "刚才那个具体怎么说的", "turn_id": "turn-1"})
        self.assertEqual(follow["mode"], "locked")
        self.assertEqual(follow["patterns"][0]["pattern_id"], self.pat["pattern_id"])

    def test_unrelated_message_needs_no_memory(self):
        r = self.retrieve("今天午饭吃了面条", turn="turn-2")
        self.assertEqual((r["status"], r["patterns"], r["episodes"]), ("NO_MEMORY_NEEDED", [], []))
        self.assertIsNone(self.memory.read.lock_for("turn-2"))

    def test_seen_suppression_only_after_the_turn_is_confirmed(self):
        first = self.retrieve("闹别扭了怎么办", turn="t-a")
        self.assertEqual(self.retrieve("闹别扭了怎么办", turn="t-b")["status"], "LOCKED")  # unconfirmed: not seen
        self.memory.read.confirm(first["inject_id"], CONV, "s1", first["refs"])
        self.assertEqual(self.retrieve("闹别扭了怎么办", turn="t-c")["status"], "NO_MEMORY_NEEDED")
        # Another session within the cooldown: not brought up again by itself; asked for the past, it is.
        self.assertEqual(self.memory.read.retrieve({"query": "闹别扭了怎么办", "turnId": "t-d", "conversationId": CONV, "sessionId": "s2"})["status"], "NO_MEMORY_NEEDED")
        self.assertEqual(self.memory.read.retrieve({"query": "上次说的闹别扭了怎么办", "turnId": "t-e", "conversationId": CONV, "sessionId": "s2"})["status"], "LOCKED")

    def test_a_new_pattern_version_is_offered_again(self):
        first = self.retrieve("闹别扭了怎么办", turn="t-a")
        self.memory.read.confirm(first["inject_id"], CONV, "s1", first["refs"])
        self.memory.store.update_pattern(self.pat["pattern_id"], {"narrative": "User 吵架后希望先被抱住，不要讲道理，冷静后再谈。"}, actor="t", reason="t")
        self.assertEqual(self.retrieve("闹别扭了怎么办", turn="t-b")["status"], "LOCKED")

    def test_expand_pattern_to_episode_to_raw(self):
        rid = self.raw("那次吵架之后你抱了我很久", at="2026-08-01T10:00:00Z")
        ep = self.memory.store.insert_episode({"content": "吵架之后 AI 抱了 User 很久。", "time_start": "2026-08-01T10:00:00Z", "time_end": "2026-08-01T10:00:00Z",
                                               "source_raw_ids": [rid], "origin": "test"}, actor="t", reason="t")
        view = self.memory.read.expand_pattern(self.pat["pattern_id"])["pattern"]
        self.assertEqual([e["episode_id"] for e in view["episodes"]], [self.fight])
        expanded = self.memory.read.expand_episode(ep["episode_id"])
        self.assertEqual(expanded["raw"][0]["id"], rid)
        self.assertEqual(expanded["raw"][0]["speaker"], "User")
        current = self.memory.read.expand_episode(ep["episode_id"], {"conversationId": CONV, "since": "2026-07-01T00:00:00Z"})
        self.assertEqual((current["raw"], current["excludedCurrentSession"]), ([], 1))  # already in this session's context

    def test_time_window_finds_episodes_of_that_period(self):
        old = self.note("User 在海边过了生日。", time="2026-09-16T10:00:00Z")
        r = self.memory.read.retrieve({"query": "上周发生了什么", "turnId": "t-time", "dry": True})
        self.assertIsNotNone(r["time_window"])
        self.assertIn(old, [e["episode_id"] for e in r["episodes"]] + [e for p in r["patterns"] for e in p.get("matched_episode_ids", [])])

    def test_a_time_only_question_is_not_dropped_by_the_reranker(self):
        for i in range(3):
            self.note(f"User 在第{i}天去了不同的地方。", time=f"2026-09-1{5 + i}T10:00:00Z")
        self.memory.read.reranker.scores = lambda query, docs: [0.001] * len(docs)  # a cross-encoder finds a date question unrelated to everything
        r = self.memory.read.retrieve({"query": "上周发生了什么", "turnId": "t-time2", "dry": True})
        self.assertEqual(r["status"], "LOCKED")
        self.assertEqual(len(r["episodes"]), 2)  # the per-turn cap on Episodes carried without a Pattern; the rest are one recall away
        self.assertIn("skipped", next(s for s in r["trace"]["stages"] if s["stage"] == "rerank"))

    def test_a_meaning_only_hit_is_verified_by_the_reranker_even_when_alone(self):
        self.memory.read.reranker.scores = lambda query, docs: [0.001] * len(docs)  # the cross-encoder says it is unrelated
        r = self.retrieve("又闹别扭了，好气", turn="t-weak")  # found by meaning (no shared words) -> weak evidence
        stage = next(s for s in r["trace"]["stages"] if s["stage"] == "rerank")
        self.assertTrue(stage["weak"] and stage["applied"])
        self.assertEqual(r["status"], "NO_MEMORY_NEEDED")
        self.memory.read.reranker.scores = lambda query, docs: [0.9] * len(docs)  # ... and when it agrees, the memory is kept
        self.assertEqual(self.retrieve("又闹别扭了，好气", turn="t-weak2")["status"], "LOCKED")
        # a keyword hit is not weak evidence: no reranker call at all
        self.memory.read.reranker.scores = lambda query, docs: (_ for _ in ()).throw(AssertionError("reranker must not run"))
        self.assertEqual(self.retrieve("吵架的时候怎么办", turn="t-strong")["status"], "LOCKED")

    def test_the_reranker_cannot_drop_a_candidate_that_words_and_entities_agree_on(self):
        self.memory.read.reranker.scores = lambda query, docs: [0.0005] * len(docs)  # a vague message: the cross-encoder scores everything ~0
        r = self.retrieve("吵架后的安抚方式还记得吗", turn="t-vague")
        self.assertEqual(r["status"], "LOCKED")
        self.assertEqual([p["pattern_id"] for p in r["patterns"]], [self.pat["pattern_id"]])

    def test_archived_units_leave_the_index(self):
        self.memory.store.update_episode(self.fight, {"status": "archived"}, actor="t", reason="t")
        self.memory.store.update_pattern(self.pat["pattern_id"], {"status": "archived"}, actor="t", reason="t")
        self.svc.vectors.wait_idle(20)
        self.assertEqual(self.retrieve("闹别扭了怎么办", turn="t-z")["status"], "NO_MEMORY_NEEDED")

    def test_projection_is_rebuilt_from_the_store_after_a_restart(self):
        self.svc.close()
        self.open()
        self.assertEqual(self.retrieve("闹别扭了怎么办", turn="t-r")["status"], "LOCKED")

    def test_debug_ranking_carries_a_trace_with_memory_kinds(self):
        out = self.svc.retrieval_debug({"query": "吵架", "policy": "recall"})
        kinds = {h["kind"] for h in out["top"]}
        self.assertTrue(kinds & {"PATTERN", "EPISODE"})
        self.assertEqual(out["trace"]["traceVersion"], 2)


class NotesAndEditTest(Base):
    def test_note_is_raw_plus_episode_and_never_re_enters_the_pipeline(self):
        eid = self.note("我对猫毛过敏。", topics=["健康"])
        ep = self.memory.store.episode(eid)
        raw = self.memory.raw_record(ep["source_raw_ids"][0])
        self.assertEqual((raw["content"], raw["sourceType"]), ("我对猫毛过敏。", "manual_note"))
        self.assertEqual(self.memory.pipeline.unprocessed_raw(), [])

    def test_edit_versions_and_review_of_quarantine(self):
        eid = self.note("User 周三看医生。")
        self.memory.route("POST", ["edit"], {"actor": "user", "action": "episode.update", "id": eid, "patch": {"content": "User 周四看医生。"}})
        ep = self.memory.store.episode(eid)
        self.assertEqual((ep["content"], ep["version"]), ("User 周四看医生。", 2))
        self.assertEqual(len(self.memory.store.all("SELECT * FROM episode_versions WHERE episode_id = ?", (eid,))), 2)


class MigrationTest(Base):
    def test_old_paths_become_episodes_or_quarantine_once(self):
        self.svc.close()
        legacy = self.tmp / "events"
        legacy.mkdir(exist_ok=True)
        rid = None
        self.open()
        rid = self.raw("我们约好了周末去海边", at="2026-08-01T10:00:00Z")
        self.svc.close()
        (legacy / "e1.json").write_text(json.dumps({"eventId": "event_1", "lineageId": "event_1", "version": 1, "status": "confirmed", "title": "海边", "content": "User 和 AI 约好周末去海边。",
                                                     "sources": [rid], "entities": ["海边"], "tags": [], "importance": 0.7, "createdAt": "2026-08-01T10:00:00Z"}, ensure_ascii=False), encoding="utf-8")
        (legacy / "e2.json").write_text(json.dumps({"eventId": "event_2", "lineageId": "event_2", "version": 1, "status": "confirmed", "title": "x", "content": "无法追溯。",
                                                     "sources": ["o_gone"], "createdAt": "2026-08-01T10:00:00Z"}, ensure_ascii=False), encoding="utf-8")
        (self.tmp / "manual").mkdir(exist_ok=True)
        (self.tmp / "manual" / "m1.json").write_text(json.dumps({"memoryId": "manual_1", "lineageId": "manual_1", "version": 1, "status": "confirmed", "content": "我对猫毛过敏。",
                                                                  "createdAt": "2026-08-02T10:00:00Z", "tags": ["健康"]}, ensure_ascii=False), encoding="utf-8")
        self.open()
        self.memory.store.kv_set("migration", {})  # this data dir was already started once (empty); pretend it is an upgrade
        self.svc.close()
        self.open()
        counts = self.memory.store.counts()
        self.assertEqual(counts["episodes"], 2)  # the traceable EVENT and the manual note
        self.assertEqual(counts["quarantine"], 1)  # the EVENT whose RAW is gone
        migrated = next(e for e in self.memory.store.episodes() if "海边" in e["content"])
        self.assertEqual(migrated["source_raw_ids"], [rid])
        self.svc.close()
        self.open()  # a second start migrates nothing again
        self.assertEqual(self.memory.store.counts()["episodes"], 2)
        self.assertEqual(self.memory.store.counts()["quarantine"], 1)


class HttpTest(Base):
    def setUp(self):
        super().setUp()
        from http.server import ThreadingHTTPServer
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.svc))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def call(self, method, path, body=None):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(3):  # a loopback connection abort (WinError 10053) can happen on a busy Windows machine
            try:
                with opener.open(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
            except (ConnectionError, urllib.error.URLError):
                if attempt == 2:
                    raise

    def test_memory_core_routes_and_retired_legacy_routes(self):
        eid = self.note("我们吵架的时候，先抱住我。")
        status, snap = self.call("GET", "/memory-core")
        self.assertEqual(status, 200)
        self.assertEqual(snap["counts"]["episodes"], 1)
        self.assertIn("providers", snap)
        status, out = self.call("POST", "/memory-core/retrieve", {"query": "闹别扭了怎么办", "turnId": "http-1", "conversationId": CONV, "sessionId": "s"})
        self.assertEqual((status, out["status"]), (200, "LOCKED"))
        status, out = self.call("POST", "/memory-core/recall", {"episode_id": eid})
        self.assertEqual((status, out["mode"]), (200, "episode"))
        self.assertEqual(self.call("GET", "/memory-core/health")[0], 200)
        for path in ("/inject", "/recall", "/inject/confirm", "/review/daily"):
            self.assertEqual(self.call("POST", path, {})[0], 410, path)
        self.assertEqual(self.call("GET", "/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
