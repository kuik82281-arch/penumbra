"""错题集: recalls the user marked as wrong, kept as questions the recall must answer from then on.

From a turn's recall (Memory Studio) she marks what should not have come up, and / or what it should have found (a
memory, or just some words). Each mistake is a question over her real memory: `run` replays every one through the
same retrieve path the turns use (a dry run - nothing is locked, seen or written) and says which still fail. Change a
threshold, a rule or the code, run them again: a fix that breaks an old one shows at once.
"""
from __future__ import annotations

import json

from .. import identity
from ..errors import Invalid, NotFound
from .store import Store, mint, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS mistakes (
  mistake_id TEXT PRIMARY KEY, at TEXT NOT NULL, query TEXT NOT NULL,
  expect_ids TEXT NOT NULL DEFAULT '[]', expect_words TEXT NOT NULL DEFAULT '[]', must_not_ids TEXT NOT NULL DEFAULT '[]',
  note TEXT NOT NULL DEFAULT '', retrieval_id TEXT, last_run_at TEXT, last_status TEXT, last_detail TEXT NOT NULL DEFAULT ''
);
"""


def ensure_schema(store: Store) -> None:
    store.db.executescript(SCHEMA)


def _ids(value) -> list[str]:
    return list(dict.fromkeys(str(x) for x in (value or []) if str(x).startswith(("pattern_", "episode_"))))[:12]


def _view(row) -> dict:
    out = {k: row[k] for k in row.keys()}
    for key in ("expect_ids", "expect_words", "must_not_ids"):
        out[key] = json.loads(out[key])
    return out


def all_mistakes(store: Store) -> list[dict]:
    return [_view(r) for r in store.all("SELECT * FROM mistakes ORDER BY at DESC")]


def add(store: Store, body: dict) -> dict:
    query = str(body.get("query") or "").strip()[:500]
    if not query:
        raise Invalid("query is required")
    expect_ids, must_not = _ids(body.get("expect_ids")), _ids(body.get("must_not_ids"))
    words = [str(w).strip()[:30] for w in body.get("expect_words") or [] if str(w).strip()][:8]
    if not (expect_ids or must_not or words):
        raise Invalid("say what should have come up, or what should not have")
    ident = mint("mistake")
    with store.tx() as db:
        db.execute("INSERT INTO mistakes (mistake_id, at, query, expect_ids, expect_words, must_not_ids, note, retrieval_id) VALUES (?,?,?,?,?,?,?,?)",
                   (ident, now_iso(), query, json.dumps(expect_ids), json.dumps(words, ensure_ascii=False), json.dumps(must_not),
                    str(body.get("note") or "").strip()[:300], body.get("retrieval_id")))
    store.audit("mistake.added", ident, actor=identity.USER_ACTOR)
    return _view(store.one("SELECT * FROM mistakes WHERE mistake_id = ?", (ident,)))


def delete(store: Store, ident: str) -> dict:
    with store.tx() as db:
        gone = db.execute("DELETE FROM mistakes WHERE mistake_id = ?", (ident,)).rowcount
    if not gone:
        raise NotFound(f"no mistake {ident}")
    return {"deleted": ident}


def judge(mistake: dict, found: dict) -> tuple[bool, str]:
    """Does this recall answer the mistake now? Its memories (Patterns, the Episodes under them, orphan Episodes) and words."""
    ids = {p["pattern_id"] for p in found.get("patterns") or []}
    ids |= {e for p in found.get("patterns") or [] for e in p.get("matched_episode_ids") or []}
    ids |= {e["episode_id"] for e in found.get("episodes") or []}
    text = "\n".join([*(f"{p['title']} {p.get('narrative', '')} {p.get('current_state', '')}" for p in found.get("patterns") or []),
                      *(e["content"] for p in found.get("patterns") or [] for e in p.get("matched_episodes") or []),
                      *(e["content"] for e in found.get("episodes") or [])])
    problems = [f"没找到 {i}" for i in mistake["expect_ids"] if i not in ids]
    problems += [f"没出现「{w}」" for w in mistake["expect_words"] if w not in text]
    problems += [f"又出现了 {i}" for i in mistake["must_not_ids"] if i in ids]
    return not problems, "；".join(problems) or "通过"


def run(core) -> dict:
    """Replay every mistake through the live retrieve path (dry) and record pass / fail."""
    results = []
    for m in all_mistakes(core.store):
        found = core.read.retrieve({"query": m["query"], "dry": True})
        ok, detail = judge(m, found)
        with core.store.tx() as db:
            db.execute("UPDATE mistakes SET last_run_at=?, last_status=?, last_detail=? WHERE mistake_id=?", (now_iso(), "pass" if ok else "fail", detail, m["mistake_id"]))
        results.append({"mistake_id": m["mistake_id"], "query": m["query"], "status": "pass" if ok else "fail", "detail": detail})
    return {"results": results, "passed": sum(r["status"] == "pass" for r in results), "total": len(results)}
