"""Memory evaluation on a fixed fictional half year (eval/corpus.json, eval/questions.json).

  python -m penumbra.eval            evaluate the built state (builds it first when there is none)
  python -m penumbra.eval --rebuild  rebuild the state: the whole pipeline over the corpus with the real DeepSeek
                                     (about one call per conversation segment), then the user's deletion

The state lives in eval/state (its own data dir: never the real memory), reports in eval/reports. Layer 1 (this file)
asks: does retrieval put the right memory in the top k, and keep out what must not come back? It costs nothing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import PACKAGE_ROOT, Config
from .memory import threads
from .service import Penombre

EVAL_DIR = PACKAGE_ROOT / "eval"
STATE_DIR = EVAL_DIR / "state"
REPORT_DIR = EVAL_DIR / "reports"
LOCAL = timezone(timedelta(hours=8))


def _local(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build(corpus: dict, log=print) -> None:
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
    # A deployment's own names (data/profile.json) go with its own prompts; without one, the defaults are used.
    profile = PACKAGE_ROOT / "data" / "profile.json"
    if profile.exists():
        STATE_DIR.mkdir(parents=True)
        shutil.copy(profile, STATE_DIR / "profile.json")
    svc = Penombre(Config(data_dir=STATE_DIR))
    try:
        if not svc.memory.verifier.available():
            raise SystemExit("DEEPSEEK_API_KEY is not set: the build needs the real DeepSeek")
        svc.memory.pipeline.segmentation = "session"
        items = []
        for seg in corpus["segments"]:
            start = _local(seg["at"])
            for i, (role, text) in enumerate(seg["msgs"]):
                items.append({"id": f"{seg['id']}-{i}", "role": "user" if role == "u" else "assistant", "content": text, "createdAt": _iso(start + timedelta(minutes=2 * i))})
        svc.ingest_originals("user", corpus["conversation_id"], items)
        log(f"[eval] {len(items)} messages in {len(corpus['segments'])} segments")
        for round_ in range(1, 6):
            result = svc.memory.pipeline.run(reason="eval", budget_s=3600)
            log(f"[eval] run {round_}: {result.get('status')} {json.dumps(result.get('summary', {}), ensure_ascii=False)}")
            if not svc.memory.pipeline.pending_work()["any"]:
                break
        # the user deletes one memory (the toilet): it must not come back.
        for e in svc.memory.store.episodes(status="active"):
            if "马桶" in e["content"]:
                svc.memory.route("POST", ["edit"], {"actor": "user", "action": "episode.delete", "id": e["episode_id"]})
                log(f"[eval] deleted {e['episode_id']}")
    finally:
        svc.close()


def _hit_text(hits: list[dict]) -> str:
    return "\n".join(h.get("content", "") for h in hits)


def evaluate(corpus: dict, questions: dict) -> dict:
    now = _local(corpus["now"])
    svc = Penombre(Config(data_dir=STATE_DIR))
    svc._now = lambda: now
    try:
        svc.vectors.wait_idle(120)
        store = svc.memory.store
        results = []
        for q in questions["questions"]:
            if q.get("layer2_only"):
                results.append({**q, "status": "n/a"})
                continue
            # The path his turns use: the memory reader (Patterns first, the time window, the gates), as a dry run.
            found = svc.memory.read.retrieve({"query": q["q"], "recent": q.get("recent") or [], "turnId": f"eval-{q['id']}", "conversationId": "eval", "sessionId": "eval", "dry": True})
            hits = [{"kind": "PATTERN", "hitId": p["pattern_id"], "content": f"{p['title']} {p.get('narrative', '')} {p['current_state']} "
                     + " ".join(e["content"] for e in p.get("matched_episodes") or [])} for p in found.get("patterns") or []]
            # What he sees for an Episode includes its story-thread line (the thread and the steps before / after).
            thread_text = lambda e: " ".join(f"{t['title']} {(t.get('before') or {}).get('content', '')} {(t.get('after') or {}).get('content', '')}"  # noqa: E731
                                             for t in e.get("threads") or [])
            hits += [{"kind": "EPISODE", "hitId": e["episode_id"], "content": f"{e['content']} {thread_text(e)}"} for e in found.get("episodes") or []]
            hits = hits[: questions["top_k"]]
            text = _hit_text(hits)
            missing = [group for group in q.get("expect", []) if not any(word in text for word in group)]
            leaked = [word for word in q.get("must_not", []) if word in text]
            if q.get("expect_none") and hits:  # nothing in memory answers this: anything brought up is noise
                leaked = [f"不该出现：{h['content'][:24]}" for h in hits]
            results.append({**q, "status": "pass" if not missing and not leaked else "fail", "missing": missing, "leaked": leaked,
                            "top": [{"kind": h["kind"], "id": h["hitId"], "content": h.get("content", "")[:120]} for h in hits]})
        active = store.episodes(status="active")
        structure = []
        for s in questions.get("structure", []):
            ok, detail = False, ""
            if "thread_with" in s:
                for t in threads.overview(store, now):
                    steps = [x for x in t["steps"] if x["status"] in threads.LIVE]
                    body = "\n".join(x["content"] for x in steps)
                    if all(w in body for w in s["thread_with"]) and len(steps) >= s["min_steps"]:
                        ok = not any(w in body for w in s.get("thread_without", [])) and (not s.get("thread_closed") or t["status"] == "closed")
                        detail = f"《{t['title']}》{len(steps)} 件 · {'已结束' if t['status'] == 'closed' else '进行中'}" + ("" if ok else "（不符合）")
                        break
                detail = detail or "没有这样的故事线"
            elif "no_episode_with_only" in s:
                bad = [e["content"] for e in active if any(w in e["content"] for w in s["no_episode_with_only"])]
                ok, detail = not bad, "；".join(b[:40] for b in bad) or "没有"
            elif "pattern_with" in s:
                pats = [p for p in store.patterns(status="active") if any(w in f"{p['title']} {p['narrative']} {p['current_state']}" for w in s["pattern_with"])]
                ok, detail = bool(pats), "；".join(f"《{p['title']}》现在：{p['current_state'][:30]}" for p in pats) or "没有"
            elif "episodes_with" in s:
                lo, hi = s.get("between", ["0000", "9999"])
                rows = [e for e in active if any(w in e["content"] for w in s["episodes_with"]) and not any(w in e["content"] for w in s.get("exclude", []))
                        and lo <= _local_day(e["time_start"]) <= hi and (not s.get("kind") or (e.get("kind") or "") == s["kind"])
                        and (not s.get("not_kind") or (e.get("kind") or "") != s["not_kind"])]
                ok = s.get("min", 0) <= len(rows) <= s.get("max", 10 ** 6)
                detail = f"{len(rows)} 条" + ("：" + "；".join(r["content"][:24] for r in rows[:6]) if rows else "")
            elif "ritual_tag" in s:
                rows = [e for e in active if e.get("kind") == "ritual" and s["ritual_tag"] in (e.get("tag") or "")]
                bad = [e for e in rows if any(w in e["content"] for w in s.get("ritual_without", []))]
                ok = len(rows) == 1 and not bad
                detail = "；".join(f"{e['tag']}：{e['content'][:30]}" for e in rows) or "没有这条仪式"
            elif "episodes_or_patterns_with" in s:
                words = s["episodes_or_patterns_with"]
                rows = [e["content"] for e in active if any(w in e["content"] for w in words)]
                rows += [p["title"] + "：" + p["current_state"] for p in store.patterns(status="active") if any(w in f"{p['title']} {p['narrative']} {p['current_state']}" for w in words)]
                ok, detail = len(rows) >= s.get("min", 1), "；".join(r[:30] for r in rows[:4]) or "没有"
            elif "lexicon_tags" in s:
                tags = [e.get("tag") or "" for e in active if e.get("kind") == "lexicon"]
                missing = [w for w in s["lexicon_tags"] if not any(w in t for t in tags)]
                ok, detail = not missing, ("缺：" + "、".join(missing)) if missing else "、".join(tags)
            structure.append({"id": s["id"], "check": s["check"], "status": "pass" if ok else "fail", "detail": detail})
        counts = {"episodes": len(active), "patterns": len(store.patterns(status="active")), "threads": len(threads.overview(store, now)),
                  "lexicon": sum(1 for e in active if e.get("kind") == "lexicon")}
        return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "now": corpus["now"], "results": results, "structure": structure, "counts": counts}
    finally:
        svc.close()


def _local_day(stamp: str) -> str:
    """The local (+08:00) date of an ISO time, for the evaluation's day ranges."""
    return (datetime.fromisoformat(stamp.replace("Z", "+00:00")) + timedelta(hours=8)).strftime("%Y-%m-%d")


def summarize(report: dict) -> str:
    scored = [r for r in report["results"] if r["status"] != "n/a"]
    passed = sum(r["status"] == "pass" for r in scored)
    lines = [f"找得对：{passed}/{len(scored)}（{round(100 * passed / max(1, len(scored)))}%）    结构：{sum(s['status'] == 'pass' for s in report['structure'])}/{len(report['structure'])}    "
             f"记忆：{report['counts']}"]
    cats: dict = {}
    for r in scored:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += r["status"] == "pass"
        c[1] += 1
    lines.append("  " + " · ".join(f"{k} {v[0]}/{v[1]}" for k, v in cats.items()))
    for r in report["results"]:
        if r["status"] == "fail":
            why = []
            if r["missing"]:
                why.append("没找到 " + " / ".join("|".join(g) for g in r["missing"]))
            if r["leaked"]:
                why.append("不该出现 " + "、".join(r["leaked"]))
            lines.append(f"  ✗ [{r['category']}] {r['q']} — {'；'.join(why)}")
    for s in report["structure"]:
        lines.append(f"  {'✓' if s['status'] == 'pass' else '✗'} {s['check']}：{s['detail']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m penumbra.eval")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the evaluation state with the real DeepSeek first")
    args = ap.parse_args(argv)
    corpus = json.loads((EVAL_DIR / "corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    if args.rebuild or not STATE_DIR.exists():
        build(corpus)
    report = evaluate(corpus, questions)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summarize(report))
    print(f"[eval] report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
