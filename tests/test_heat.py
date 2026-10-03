"""Heat: a memory that has come up in his sessions lately ranks a little higher among equally relevant ones."""
import dataclasses
from datetime import datetime, timedelta, timezone

from tests.support import Base


class HeatTest(Base):
    def seen(self, episode_id, days_ago):
        at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        with self.memory.store.tx() as db:
            db.execute("INSERT INTO seen VALUES (?,?,?,?,?,?)", ("c1", f"s{days_ago}", "episode", episode_id, 1, at))

    def boosts(self, found, ident):
        return next(h for h in found["found"]["top"] if h["hitId"] == ident)["scoreBreakdown"]["boosts"]

    def test_recently_brought_up_memories_warm_up_and_cool_down(self):
        cold = self.note("你说周末想去海边捡贝壳。", importance=0.6)
        warm = self.note("你说周末想去海边看日落。", importance=0.6)
        for d in (0, 1, 3):
            self.seen(warm, d)
        self.seen(cold, 80)  # long ago: almost cold again
        self.svc.retrieval._heat_cache = None
        found = self.recall("周末 海边")
        ids = [r["id"] for r in found["results"]]
        self.assertLess(ids.index(warm), ids.index(cold))
        hot, chilly = self.boosts(found, warm)["heat"], self.boosts(found, cold)["heat"]
        self.assertGreater(hot, 0.04)
        self.assertLessEqual(hot, self.svc.retrieval.cfg.heat_cap)
        self.assertLess(chilly, 0.005)

    def test_heat_only_reorders_what_was_found(self):
        a = self.note("你喜欢草莓味的冰淇淋。", importance=0.6)
        self.note("你说论文开题报告下周交。", importance=0.6)
        for d in range(10):
            self.seen(a, d)
        core = self.svc.retrieval
        cfg = core.cfg
        core._heat_cache = None
        with_heat = {r["id"] for r in self.recall("论文 开题 报告")["results"]}
        core.cfg = dataclasses.replace(cfg, heat_cap=0.0)
        try:
            without = {r["id"] for r in self.recall("论文 开题 报告")["results"]}
        finally:
            core.cfg = cfg
        self.assertEqual(with_heat, without)
