"""专名库: a name or any alias she keeps finds the memories that mention it under any of its names; a single Chinese
character never counts. 错题集: a recall she marked wrong is replayed and judged."""
from penumbra.errors import Invalid
from penumbra.memory import mistakes, names
from tests.support import Base


class NamesTest(Base):
    def test_an_alias_finds_the_memory_written_with_the_name(self):
        eid = self.note("年糕今天吐了两次，带它去看了兽医，说是肠胃炎。", importance=0.6)
        self.note("你说今天的晚霞是粉紫色的。", importance=0.4)
        before = self.memory.read.retrieve({"query": "糕糕最近好点没", "dry": True})
        self.assertNotIn(eid, [e["episode_id"] for e in before["episodes"]], "without the 专名库, 糕糕 means nothing")
        names.save(self.memory.store, {"name": "年糕", "aliases": ["糕糕", "糕"], "kind": "宠物"})
        after = self.memory.read.retrieve({"query": "糕糕最近好点没", "dry": True})
        self.assertIn(eid, [e["episode_id"] for e in after["episodes"]])
        self.assertIn("names", [s["stage"] for s in after["trace"]["stages"]])
        self.assertEqual(names.mentioned(self.memory.store, "蛋糕好吃"), [], "a single character is not a name")

    def test_names_are_hers_to_keep(self):
        saved = names.save(self.memory.store, {"name": "星屿", "aliases": ["星屿游戏", "星屿"], "kind": "地方"})
        self.assertEqual(saved["aliases"], ["星屿游戏"], "the name itself is not its own alias")
        with self.assertRaises(Invalid):
            names.save(self.memory.store, {"name": "x", "kind": "不存在"})
        names.delete(self.memory.store, saved["name_id"])
        self.assertEqual(names.all_names(self.memory.store), [])


class MistakesTest(Base):
    def test_a_marked_recall_is_replayed_and_judged(self):
        good = self.note("你最喜欢的颜色是薄荷绿。", importance=0.6)
        bad = self.note("你室友最喜欢蓝色。", importance=0.5)
        mistakes.add(self.memory.store, {"query": "我最喜欢什么颜色", "expect_ids": [good], "must_not_ids": [bad], "expect_words": ["薄荷绿"]})
        mistakes.add(self.memory.store, {"query": "我最喜欢什么颜色", "expect_words": ["橙色"]})
        report = mistakes.run(self.memory)
        self.assertEqual(report["total"], 2)
        rows = {m["expect_words"][0]: m for m in mistakes.all_mistakes(self.memory.store)}
        self.assertEqual(rows["橙色"]["last_status"], "fail")
        self.assertIn("没出现「橙色」", rows["橙色"]["last_detail"])
        with self.assertRaises(Invalid):
            mistakes.add(self.memory.store, {"query": "随便"})


class ProfileTest(Base):
    def test_a_profile_is_written_when_its_memories_change_and_travels_with_the_name(self):
        names.save(self.memory.store, {"name": "年糕", "aliases": ["糕糕"], "kind": "宠物"})
        self.note("年糕是你去年领养的橘猫，三岁。", importance=0.6)
        calls = []

        def decide(payload):
            calls.append(payload)
            return {"profile": "年糕是你领养的橘猫，三岁，最近肠胃不好。"}
        self.deepseek.decide = decide
        ident = names.all_names(self.memory.store)[0]["name_id"]
        self.assertTrue(names.write_profile(self.memory.store, self.memory.verifier.client, ident).get("skipped"), "one memory is not enough")
        self.note("糕糕今天吐了两次，兽医说是肠胃炎。", importance=0.6)
        self.assertEqual(names.refresh_profiles(self.memory.store, self.memory.verifier.client), [ident])
        self.assertEqual(len(calls[0]["memories"]), 2, "both memories, under either of its names")
        self.assertEqual(names.refresh_profiles(self.memory.store, self.memory.verifier.client), [], "nothing changed: no new call")
        found = self.memory.read.retrieve({"query": "糕糕最近怎么样", "dry": True})
        self.assertEqual(found["profiles"], [{"name": "年糕", "profile": "年糕是你领养的橘猫，三岁，最近肠胃不好。"}])
