"""Replay turns through retrieval and score what came back.

  python -m penumbra.turn_eval --labeled eval/turns-labeled.json [--body ...] [--service ...]
      the fixed, hand-labelled set (no model grades anything): per case need none / maybe / yes, the memories that are
      fine to bring (accept) and the ones a yes-case needs (expect). Reports irrelevant recall (delivered memories
      outside accept; none-cases that got anything), effective recall (yes-cases whose needed memory arrived) and every
      yes-case the gate blocked, with where (intent said none / judge dropped it / never retrieved). A memory written
      after a real message (from that very scene) is a leak of the replay, counted apart.

  The older DeepSeek-graded mode below is kept for a quick look; DeepSeek also runs the gate, so it grades itself.

  python -m penumbra.turn_eval [--n 100] [--label NAME] [--body '{"gate": "off"}'] [--service http://127.0.0.1:8791]

For the last N of her chat messages (with the six lines before each), against the running service (dry: nothing written,
no lock, no seen, no cooldown):
  1. the host's entry gate (a port of context.ts memoryGate) - skipped messages deliver nothing;
  2. POST /memory-core/retrieve {dry: true, ...body} - what would reach the turn;
  3. DeepSeek grades, once per message: does it need long-term memory at all (none / helpful / required), and for each
     delivered memory: does it really help this reply (a memory telling the scene that is going on right now does not -
     it was written later, from these very lines).

Report: false triggers (need none, something delivered), misses (need required, nothing useful delivered), precision
(useful / delivered), latency. Grades are cached per message and per (message, memory) in eval/state/turn-grades.json,
so two variants are graded alike. The report goes to eval/reports/turns-<label>-<time>.json (never committed).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import threading
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import PACKAGE_ROOT
from .memory.llm import DeepSeekClient

DATA = PACKAGE_ROOT / "data"
STATE = PACKAGE_ROOT / "eval" / "state"
REPORTS = PACKAGE_ROOT / "eval" / "reports"
SERVICE = "http://127.0.0.1:8790"
PARALLEL = 1

RECALL_CUE = re.compile(r"上次|上回|还记得|记不记得|记得吗|那次|那回|那天|之前|以前|说过|讲过|聊过")
RITUAL_ONLY = re.compile(r"^(?:早安|早上好|早|午安|晚安|晚安安|好梦|抱|亲亲|亲|摸摸|蹭蹭|嗯+|哦+|噢+|啊+|呜+|好的?|好呀|行|收到|ok|哈+|嘿+|嘻+|在吗|在嘛|在不在|"
                         r"睡了|睡觉|起床了|醒了|想你|想你了|爱你|么么哒?)$", re.IGNORECASE)

GRADE_SYSTEM = """你在评估一个陪伴型 AI（他）的长期记忆检索。给你：最近几句对话、她刚发的消息、系统为这一轮从长期记忆里取出的几条记忆。

回答两件事：
1. need：要回好这句话，他需不需要长期记忆（几天前、更早的事；眼前这几句对话里已经有的不算）？
   none = 不需要（告知当前动作、亲昵的话、情绪、角色扮演里正在进行的动作、眼前对话就能接住）
   helpful = 有的话会更好（涉及她的偏好、以前的经历，但不问也能回）
   required = 需要（明确问过去的事、「上次那个」「那个人」这种指代必须靠过去才懂）
2. useful：每条记忆，能不能实质帮他理解这句话或改善回复？只是有同一个词、同一种气氛不算。
   记忆讲的正是眼前这几句对话里正在发生的事，也算 false（它是事后从这几句写出来的）。

只输出 JSON：{"need": "none|helpful|required", "seek": "如果需要，真正要找的信息，一句话；不需要就空", "useful": {"<记忆编号>": true|false}}"""


def memory_gate(message: str) -> tuple[bool, str]:
    bare = re.sub(r"[\s\W_~～]+", "", unicodedata.normalize("NFKC", message)).lower()
    if RECALL_CUE.search(message):
        return True, "asks for the past"
    if not bare:
        return False, "no words"
    if RITUAL_ONLY.match(bare):
        return False, "ritual"
    if len(bare) <= 2:
        return False, "too short"
    return True, "her words"


def sample(n: int, spread: bool = False) -> list[dict]:
    conn = sqlite3.connect(f"file:{DATA / 'index.sqlite'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, conversation_id, role, content, created_at FROM originals WHERE source_type = 'chat_message' ORDER BY created_at, rowid").fetchall()
    by_conv: dict[str, list] = {}
    for r in rows:
        by_conv.setdefault(r["conversation_id"], []).append(r)
    turns = []
    for conv in by_conv.values():
        for i, r in enumerate(conv):
            if r["role"] == "user" and (r["content"] or "").strip():
                turns.append({"id": r["id"], "at": r["created_at"], "message": r["content"],
                              "recent": [{"role": x["role"], "content": (x["content"] or "")[:400]} for x in conv[max(0, i - 6):i]]})
    turns.sort(key=lambda t: t["at"])
    if spread:  # a fixed, seeded sample across the whole history (the latest messages are often one long scene)
        return sorted(random.Random(7).sample(turns, min(n, len(turns))), key=lambda t: t["at"])
    return turns[-n:]


def retrieve(turn: dict, extra: dict) -> dict:
    body = {"query": turn["message"], "recent": turn["recent"], "dry": True, **extra}
    req = urllib.request.Request(f"{SERVICE}/memory-core/retrieve", data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read())
    data["_ms"] = round((time.perf_counter() - started) * 1000)
    return data


def delivered(result: dict) -> list[tuple[str, str]]:
    out = [(f"pattern:{p['pattern_id']}", f"{p['title']}：{p['narrative'][:300]}（当前：{p['current_state']}）") for p in result.get("patterns") or []]
    out += [(f"episode:{e['episode_id']}", e["content"][:400]) for e in result.get("episodes") or []]
    return out


def lines(turn: dict) -> str:
    return "\n".join(f"{'她' if r['role'] == 'user' else '他'}：{r['content']}" for r in turn["recent"]) or "（没有）"


class Grades:
    def __init__(self):
        self.path = STATE / "turn-grades.json"
        self.data = json.loads(self.path.read_text("utf-8")) if self.path.exists() else {"need": {}, "useful": {}}
        self.lock = threading.Lock()
        self.client = DeepSeekClient(timeout_s=60)

    def grade(self, turn: dict, mems: list[tuple[str, str]]) -> None:
        tid = turn["id"]
        missing = [m for m in mems if f"{tid}|{m[0]}" not in self.data["useful"]]
        if tid in self.data["need"] and not missing:
            return
        ask = missing or mems
        numbered = "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(ask)) or "（这一轮没有取出记忆）"
        user = f"最近的对话：\n{lines(turn)}\n\n她刚说：{turn['message'][:600]}\n\n取出的记忆：\n{numbered}"
        answer, _ = self.client.chat_json(GRADE_SYSTEM, user, max_tokens=600)
        with self.lock:
            if tid not in self.data["need"]:
                self.data["need"][tid] = {"need": answer.get("need") if answer.get("need") in ("none", "helpful", "required") else "helpful", "seek": answer.get("seek") or ""}
            useful = answer.get("useful") or {}
            for i, (key, _) in enumerate(ask):
                self.data["useful"][f"{tid}|{key}"] = bool(useful.get(str(i), False))

    def save(self) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), "utf-8")


def _suffix_in(memory_key: str, ids: list[str]) -> bool:
    return any(memory_key.endswith(i) for i in ids)


def run_labeled(path: str, extra: dict, label: str) -> None:
    data = json.loads(Path(path).read_text("utf-8"))
    ix = sqlite3.connect(f"file:{DATA / 'index.sqlite'}?mode=ro", uri=True)
    ix.row_factory = sqlite3.Row
    core = sqlite3.connect(f"file:{DATA / 'memory' / 'core.sqlite'}?mode=ro", uri=True)
    born = {r[0]: r[1] for r in core.execute("SELECT episode_id, created_at FROM episodes")}
    born.update({r[0]: r[1] for r in core.execute("SELECT pattern_id, created_at FROM patterns")})

    def turn_of(case: dict) -> dict:
        if case["source"] != "real":
            return {"id": case["id"], "message": case["message"], "recent": case.get("recent") or [], "at": None}
        row = ix.execute("SELECT conversation_id, content, created_at FROM originals WHERE id = ?", (case["raw_id"],)).fetchone()
        prev = ix.execute("SELECT role, content FROM originals WHERE conversation_id = ? AND source_type = 'chat_message' AND created_at < ? ORDER BY created_at DESC, rowid DESC LIMIT 6",
                          (row["conversation_id"], row["created_at"])).fetchall()
        return {"id": case["id"], "message": row["content"], "at": row["created_at"],
                "recent": [{"role": r["role"], "content": (r["content"] or "")[:400]} for r in reversed(prev)]}

    turns = {c["id"]: turn_of(c) for c in data["cases"]}  # sqlite stays on this thread

    def run(case: dict) -> dict:
        turn = turns[case["id"]]
        search, why = memory_gate(turn["message"])
        result = retrieve(turn, extra) if search else {"status": "GATED", "_ms": 0}
        stages = {s["stage"]: s for s in (result.get("trace") or {}).get("stages", [])}
        keys = [k for k, _ in delivered(result)]
        leaks = [k for k in keys if turn["at"] and born.get(k.split(":", 1)[1], "") >= turn["at"]]
        kept = [k for k in keys if k not in leaks]
        irrelevant = [k for k in kept if not _suffix_in(k, case["accept"])]
        hit = bool(case["expect"]) and any(_suffix_in(k, case["expect"]) for k in kept)
        where = None
        if case["need"] == "yes" and case["expect"] and not hit:
            intent, judge = stages.get("intent") or {}, stages.get("judge") or {}
            dropped = [d for d in judge.get("dropped") or [] if _suffix_in(d, case["expect"])]
            where = ("host gate" if not search else "intent said none" if intent.get("need") == "none"
                     else "intent failed" if intent.get("error") else "judge dropped it" if dropped
                     else "judge failed" if judge.get("error") else "not retrieved")
        return {"id": case["id"], "category": case["category"], "need": case["need"], "message": turn["message"][:60], "status": result.get("status"),
                "delivered": kept, "leaks": leaks, "irrelevant": irrelevant, "hit": hit, "blocked": where, "ms": result["_ms"],
                "degraded": result.get("degraded"), "intent": (stages.get("intent") or {}).get("need"), "seek": (stages.get("intent") or {}).get("seek")}

    # one at a time, as turns come in production: the engine serialises searches, so parallel cases would eat each
    # other's time budget and show degradations that a real turn never sees
    with ThreadPoolExecutor(PARALLEL) as pool:
        rows = list(pool.map(run, data["cases"]))
    by_need = lambda n: [r for r in rows if r["need"] == n]  # noqa: E731
    needed = [r for r in rows if r["need"] == "yes" and next(c for c in data["cases"] if c["id"] == r["id"])["expect"]]
    items = sum(len(r["delivered"]) for r in rows)
    bad = sum(len(r["irrelevant"]) for r in rows)
    searched = [r for r in rows if r["status"] != "GATED"]
    summary = {
        "label": label, "cases": len(rows), "body": extra,
        "noneCasesWithMemory": f"{sum(1 for r in by_need('none') if r['delivered'])}/{len(by_need('none'))}",
        "maybeCasesWithIrrelevant": f"{sum(1 for r in by_need('maybe') if r['irrelevant'])}/{len(by_need('maybe'))}",
        "irrelevantItems": f"{bad}/{items}" + (f" ({bad / items:.0%})" if items else ""),
        "effectiveRecall": f"{sum(1 for r in needed if r['hit'])}/{len(needed)}",
        "byCategory": {c: f"{sum(1 for r in needed if r['category'] == c and r['hit'])}/{sum(1 for r in needed if r['category'] == c)}" for c in sorted({r["category"] for r in needed})},
        "blocked": [f"{r['id']} [{r['blocked']}] {r['message']}" for r in needed if not r["hit"]],
        "leaks": sum(len(r["leaks"]) for r in rows),
        "degraded": sum(1 for r in rows if r["degraded"]),
        "latencyMs": {"mean": round(sum(r["ms"] for r in searched) / max(1, len(searched))), "max": max((r["ms"] for r in searched), default=0)},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"labeled-{label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--label", default="current")
    parser.add_argument("--spread", action="store_true", help="sample across the whole history instead of the last N")
    parser.add_argument("--body", default="{}", help="extra retrieve body fields, JSON")
    parser.add_argument("--service", default=SERVICE, help="a Penumbra to replay against (e.g. a copy of the data on another port)")
    parser.add_argument("--labeled", help="score against a fixed, hand-labelled set (eval/turns-labeled.json)")
    parser.add_argument("--parallel", type=int, default=1, help="labeled cases at once (1 = like production)")
    args = parser.parse_args()
    globals()["SERVICE"] = args.service
    globals()["PARALLEL"] = args.parallel
    extra = json.loads(args.body)
    health = json.loads(urllib.request.urlopen(f"{SERVICE}/memory-core/health", timeout=10).read())
    if not (health.get("embedding") or {}).get("ready"):
        # without the embedding the engine has no meaning signal: every number below would be about a broken service
        raise SystemExit(f"the service's embedding is not ready ({(health.get('embedding') or {}).get('error')}); wait or restart it")
    if args.labeled:
        return run_labeled(args.labeled, extra, args.label)
    turns = sample(args.n, args.spread)
    grades = Grades()

    def run(turn: dict) -> dict:
        search, why = memory_gate(turn["message"])
        result = retrieve(turn, extra) if search else {"status": "GATED", "_ms": 0}
        mems = delivered(result)
        grades.grade(turn, mems)
        stages = {s["stage"]: s for s in (result.get("trace") or {}).get("stages", [])}
        return {"id": turn["id"], "at": turn["at"], "message": turn["message"][:80], "gate": why, "status": result.get("status"), "ms": result["_ms"],
                "delivered": [k for k, _ in mems], "intent": stages.get("intent"), "judge": stages.get("judge")}

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(run, turns))
    grades.save()
    need = grades.data["need"]
    useful = grades.data["useful"]
    for r in rows:
        r["need"] = need[r["id"]]["need"]
        r["useful"] = [k for k in r["delivered"] if useful.get(f"{r['id']}|{k}")]
    total = len(rows)
    none = [r for r in rows if r["need"] == "none"]
    required = [r for r in rows if r["need"] == "required"]
    delivered_n = sum(len(r["delivered"]) for r in rows)
    useful_n = sum(len(r["useful"]) for r in rows)
    searched = [r for r in rows if r["status"] != "GATED"]
    summary = {
        "label": args.label, "turns": total, "body": extra,
        "need": {k: sum(1 for r in rows if r["need"] == k) for k in ("none", "helpful", "required")},
        "falseTriggers": f"{sum(1 for r in none if r['delivered'])}/{len(none)}",
        "requiredServed": f"{sum(1 for r in required if r['useful'])}/{len(required)}",
        "helpfulServed": f"{sum(1 for r in rows if r['need'] == 'helpful' and r['useful'])}/{sum(1 for r in rows if r['need'] == 'helpful')}",
        "precision": f"{useful_n}/{delivered_n}" + (f" ({useful_n / delivered_n:.0%})" if delivered_n else ""),
        "turnsWithMemory": sum(1 for r in rows if r["delivered"]),
        "latencyMs": {"mean": round(sum(r["ms"] for r in searched) / max(1, len(searched))), "max": max((r["ms"] for r in searched), default=0)},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"turns-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
