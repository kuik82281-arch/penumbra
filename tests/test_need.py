"""Local need gate: a name or a noun is not a reason to search; words that lean on the past are. Who someone is and
what happened before are answered separately and can both apply."""
from penumbra.memory import names, need
from tests.support import Base


class NeedTest(Base):
    def test_now_is_not_the_past(self):
        for text in ("我今天穿了一条蓝色格子连衣裙", "中午吃了牛肉面", "团子猫砂该换了"):
            self.assertEqual(need.judge(self.memory.store, text)["episode"], "direct", text)

    def test_words_that_lean_on_the_past_search(self):
        for text in ("我以前穿过类似的连衣裙吗", "小林今天又像上次那样来找我", "你还记得那天的火锅吗", "我平时喜欢吃什么", "那个师兄后来还有找过我吗"):
            self.assertEqual(need.judge(self.memory.store, text)["episode"], "episode", text)

    def test_something_that_existed_before_is_the_past_even_unsaid(self):
        self.assertEqual(need.judge(self.memory.store, "面试结果出来了")["episode"], "episode")
        self.assertEqual(need.judge(self.memory.store, "那本书终于看完了")["episode"], "episode")
        self.assertEqual(need.judge(self.memory.store, "我决定今晚早点睡")["episode"], "direct", "a decision about now")
        recent = [{"role": "user", "content": "我下午去面试"}, {"role": "assistant", "content": "加油"}]
        self.assertEqual(need.judge(self.memory.store, "面试结果出来了", recent)["episode"], "ask", "the lines before may already hold it")

    def test_a_name_is_looked_up_without_an_episode_search(self):
        saved = names.save(self.memory.store, {"name": "小林", "kind": "人"})
        plain = need.judge(self.memory.store, "小林今天来找我了")
        self.assertEqual((plain["episode"], plain["names"]), ("direct", [saved["name_id"]]))
        both = need.judge(self.memory.store, "小林今天又像上次那样来找我")
        self.assertEqual((both["episode"], both["names"]), ("episode", [saved["name_id"]]), "both at once, not either-or")

    def test_wired_behind_its_flag_a_name_brings_who_not_what(self):
        eid = self.note("小林今天来找你借了伞，说下次请你喝奶茶。", importance=0.7)
        saved = names.save(self.memory.store, {"name": "小林", "kind": "人", "note": "同事"})
        with self.memory.store.tx() as db:
            db.execute("UPDATE names SET profile='她的同事，常来借东西。' WHERE name_id=?", (saved["name_id"],))
        off = self.memory.read.retrieve({"query": "小林今天来找我了", "dry": True})
        self.assertIn(eid, [e["episode_id"] for e in off["episodes"]], "no gate: the plain search still matches the words")
        self.assertEqual(next(st for st in off["trace"]["stages"] if st["stage"] == "names").get("skipped"), "not asked about the past")
        on = self.memory.read.retrieve({"query": "小林今天来找我了", "dry": True, "needGate": True})
        self.assertEqual((on["episodes"], on["patterns"], on["profiles"]), ([], [], []), "he knows who 小林 is: not even the profile")
        back = self.memory.read.retrieve({"query": "小林今天又像上次那样来借伞了", "dry": True, "needGate": True})
        self.assertIn("need", [s["stage"] for s in back["trace"]["stages"]])
        self.assertIn(eid, [e["episode_id"] for e in back["episodes"]], "leaning on the past: the search runs as before")

    def test_the_switch_file_turns_it_on_and_off_without_a_restart(self):
        import json, os, time
        from unittest import mock
        from penumbra import trial_report
        self.note("小林今天来找你借了伞。", importance=0.7)
        path = self.svc.config.data_dir / "switches.json"
        with mock.patch.dict(os.environ, {"PENUMBRA_NEED_GATE": ""}):
            self.assertNotIn("need", [s["stage"] for s in self.memory.read.retrieve({"query": "今天下雨了", "turn_id": "t-off"})["trace"]["stages"]])
            path.write_text(json.dumps({"need_gate": True}), "utf-8")
            on = self.memory.read.retrieve({"query": "今天下雨了", "turn_id": "t-on"})
            self.assertEqual(next(s for s in on["trace"]["stages"] if s["stage"] == "need")["episode"], "direct")
            time.sleep(0.05)
            path.write_text(json.dumps({"need_gate": False}), "utf-8")
            os.utime(path, (time.time() + 5, time.time() + 5))
            self.assertNotIn("need", [s["stage"] for s in self.memory.read.retrieve({"query": "今天下雨了", "turn_id": "t-off2"})["trace"]["stages"]])
        searched = self.memory.read.retrieve({"query": "小林借伞那次", "turn_id": "t-search"})
        self.assertIn("engine", [s["stage"] for s in searched["trace"]["stages"]], "the logged trace says where candidates ended")
        days = trial_report.report(self.svc.config.data_dir / "memory" / "core.sqlite", 3650)
        total = {k: sum(d.get(k, 0) for d in days.values()) for k in ("turns", "gate_direct", "gate_off", "searched")}
        self.assertEqual((total["turns"], total["gate_direct"], total["gate_off"]), (4, 1, 3))
        self.assertGreaterEqual(total["searched"], 1)

    def test_a_name_profile_confirmed_without_memories_does_not_come_back_on_the_next_mention(self):
        saved = names.save(self.memory.store, {"name": "团子", "kind": "宠物"})
        with self.memory.store.tx() as db:
            db.execute("UPDATE names SET profile='她养的猫。' WHERE name_id=?", (saved["name_id"],))
        body = {"conversationId": "c1", "sessionId": "s1", "needGate": True}
        passing = self.memory.read.retrieve({**body, "query": "团子在睡觉", "turn_id": "t0"})
        self.assertEqual(passing["profiles"], [], "a name said in passing brings nothing")
        first = self.memory.read.retrieve({**body, "query": "团子之前在睡觉", "turn_id": "t1"})
        self.assertEqual(([p["name"] for p in first["profiles"]], first["refs"]), (["团子"], []))
        # the bridge confirms a turn that carried only a profile (no memory refs)
        self.memory.read.confirm(first["inject_id"], "c1", "s1", [])
        second = self.memory.read.retrieve({**body, "query": "团子之前又在睡觉了", "turn_id": "t2"})
        self.assertEqual(second["profiles"], [], "the same profile is not sent again in this session")

    def test_a_question_without_a_cue_is_left_to_the_intent_call(self):
        self.assertEqual(need.judge(self.memory.store, "我今天想吃什么")["episode"], "ask")
