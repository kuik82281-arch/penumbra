"""Daily numbers for the need-gate trial, from the retrievals log (read-only). Counts and ids only - never message text.

  python -m penumbra.trial_report [--days 7] [--json out.json]

Per local day (UTC+8): inject turns; the local gate's verdicts (direct / ask / episode, or off); the intent call (none /
maybe / yes / failed); the engine (searched, candidates, where they ended); the judge (kept something / dropped all /
failed); what was handed over (memories, name profiles only, nothing) and what the bridge confirmed it injected; DeepSeek
calls and tokens; latency p50 / p95 / max; degradations.

Two lists for a person to look at, not miss counts: turns the gate let through that the intent call then refused
(gate_pass_intent_none), and turns that brought nothing and were followed within 3 minutes by his own recall search in
the same conversation (followed_by_recall) - a possible miss until someone confirms it was one.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import load_config

LOCAL = timezone(timedelta(hours=8))


def _pct(values: list[float], q: float) -> int:
    if not values:
        return 0
    v = sorted(values)
    return round(v[min(len(v) - 1, int(len(v) * q))])


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def report(core_path: Path, days: int) -> dict:
    db = sqlite3.connect(f"file:{core_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = db.execute("SELECT retrieval_id, at, conversation_id, policy, status, pattern_ids, episode_ids, latency_ms, trace, confirmed_at "
                      "FROM retrievals WHERE at >= ? ORDER BY at", (since,)).fetchall()
    recalls = defaultdict(list)
    for r in rows:
        if r["policy"] == "recall":
            recalls[r["conversation_id"]].append(_at(r["at"]))
    out: dict[str, dict] = {}
    for r in rows:
        if r["policy"] != "inject":
            continue
        day = _at(r["at"]).astimezone(LOCAL).strftime("%Y-%m-%d")
        d = out.setdefault(day, defaultdict(int) | {"ms": [], "ms_searched": [], "gate_pass_intent_none": [], "followed_by_recall": [], "degraded": defaultdict(int)})
        stages = {s.get("stage"): s for s in (json.loads(r["trace"] or "{}").get("stages") or [])}
        d["turns"] += 1
        need = stages.get("need")
        d[f"gate_{need['episode'] if need else 'off'}"] += 1
        intent = stages.get("intent")
        if intent is not None:
            d["intent_calls"] += 1
            d[f"intent_{intent.get('need') or 'failed'}"] += 1
            if need and need["episode"] != "direct" and intent.get("need") == "none":
                d["gate_pass_intent_none"].append(r["retrieval_id"])
        if "engine" in stages or "threshold" in stages:
            d["searched"] += 1
            d["ms_searched"].append(r["latency_ms"] or 0)
            for end, n in (stages.get("engine") or {}).get("ends", {}).items():
                d[f"candidates_{end}"] += n
        judge = stages.get("judge")
        if judge is not None and judge.get("latency_ms"):
            d["judge_calls"] += 1
            d["judge_failed" if judge.get("error") else ("judge_kept" if "keep" in judge and judge["keep"] else "judge_dropped_all")] += 1
        memories = len(json.loads(r["pattern_ids"])) + len(json.loads(r["episode_ids"]))
        profiles = len((stages.get("names") or {}).get("profiles") or [])
        d["handed_memories" if memories else ("handed_profiles_only" if r["status"] == "LOCKED" and profiles else "handed_nothing")] += 1
        if r["confirmed_at"]:
            d["confirmed_injected"] += 1
        for call in (intent, judge):
            if call and call.get("latency_ms"):
                d["deepseek_calls"] += 1
                d["tokens"] += sum(int((call.get("usage") or {}).get(k) or 0) for k in ("inputTokens", "outputTokens"))
        d["ms"].append(r["latency_ms"] or 0)
        if r["status"] == "NO_MEMORY_NEEDED" and not memories:
            at = _at(r["at"])
            if any(timedelta(0) < t - at <= timedelta(minutes=3) for t in recalls.get(r["conversation_id"], [])):
                d["followed_by_recall"].append(r["retrieval_id"])
        if intent and intent.get("failure"):
            d["degraded"][f"intent_{intent['failure']}"] += 1
        if judge and judge.get("failure"):
            d["degraded"][f"judge_{judge['failure']}"] += 1
    final = {}
    for day, d in sorted(out.items()):
        ms, mss = d.pop("ms"), d.pop("ms_searched")
        final[day] = {**{k: v for k, v in d.items() if not isinstance(v, (list, dict))},
                      "latency_ms": {"p50": _pct(ms, .5), "p95": _pct(ms, .95), "max": round(max(ms, default=0))},
                      "latency_searched_ms": {"p50": _pct(mss, .5), "p95": _pct(mss, .95)},
                      "degraded": dict(d["degraded"]),
                      "gate_pass_intent_none": d["gate_pass_intent_none"], "followed_by_recall": d["followed_by_recall"]}
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--data")
    parser.add_argument("--json")
    args = parser.parse_args()
    result = report(load_config(args.data).data_dir / "memory" / "core.sqlite", args.days)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.json:
        Path(args.json).write_text(text, "utf-8")
    print(text)


if __name__ == "__main__":
    main()
