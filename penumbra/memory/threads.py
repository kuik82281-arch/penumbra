"""Story threads (故事线): Episodes that are one ongoing story - weeks of exams, an interview, a job hunt - linked in time
order, without being rewritten into one text while the story is still going on.

  propose   when DeepSeek writes an Episode it may say which thread it continues (a thread it was shown) or that it starts
            one together with earlier Episodes, with a confidence. The candidate threads come from local retrieval: the
            threads of the nearest existing Episodes - no extra model call.
  link      confidence >= AUTO_CONFIDENCE: attached at once ("auto"); below: "pending" until the user decides. Either way
            she can look, approve, move it to another thread (candidates from local retrieval), or take it off.
  refuse    a pair she took off is remembered; the same proposal is never attached again.
  order     a thread's Episodes are ordered by time: "before" / "after" follow, nothing to maintain by hand. An Episode
            may belong to two threads (stories run in parallel).
  close     a thread with nothing new for QUIET_DAYS is flagged; when the user closes it, DeepSeek writes the whole story
            once (versioned; a reopened thread can be written again).
  recall    a remembered Episode carries one line: which thread, its place, the Episode before and after it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .. import identity, prompts
from ..errors import Invalid, NotFound
from .store import Store, mint, now_iso, parse_time

AUTO_CONFIDENCE = 0.85


def gist(episode: dict, limit: int = 120) -> str:
    """A step of a thread as one line: its excerpt; an older Episode without one shows the start of its content."""
    return (episode.get("excerpt") or "").strip() or episode["content"][:limit]


QUIET_DAYS = 21
CANDIDATE_THREADS = 6
RECENT_DAYS = 10
LIVE = ("auto", "approved")  # links that count; "pending" waits, "rejected" is remembered as a refusal

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
  thread_id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
  story TEXT NOT NULL DEFAULT '', story_version INTEGER NOT NULL DEFAULT 0, story_at TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, closed_at TEXT
);
CREATE TABLE IF NOT EXISTS thread_links (
  thread_id TEXT NOT NULL, episode_id TEXT NOT NULL, status TEXT NOT NULL,
  confidence REAL, reason TEXT NOT NULL DEFAULT '', decision_id TEXT, created_at TEXT NOT NULL, decided_at TEXT, decided_by TEXT,
  PRIMARY KEY (thread_id, episode_id)
);
CREATE INDEX IF NOT EXISTS thread_links_episode ON thread_links(episode_id);
"""


def ensure_schema(store: Store) -> None:
    store.db.executescript(SCHEMA)
    if "aliases" not in {r[1] for r in store.db.execute("PRAGMA table_info(threads)")}:
        # How she might call the whole story ("找工作", "求职"): searchable on every step of it.
        store.db.execute("ALTER TABLE threads ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'")


def _clean_aliases(raw) -> list[str]:
    out: list[str] = []
    for a in raw if isinstance(raw, list) else []:
        a = str(a).strip()[:12]
        if len(a) >= 2 and a not in out:
            out.append(a)
    return out[:6]


def _touch_steps(store: Store, thread_id: str) -> None:
    """Its steps are indexed again (their 故事线· tags follow the thread's name, aliases and membership)."""
    for r in store.all("SELECT episode_id FROM thread_links WHERE thread_id = ?", (thread_id,)):
        store._dirty["episodes"].add(r["episode_id"])


def _episode_time(store: Store, episode_id: str) -> str:
    try:
        return store.episode(episode_id)["time_start"]
    except NotFound:
        return ""


# ------------------------------------------------------------ reading

def thread(store: Store, thread_id: str) -> dict:
    row = store.one("SELECT * FROM threads WHERE thread_id = ?", (thread_id,))
    if row is None:
        raise NotFound(f"no thread {thread_id}")
    return dict(row)


def links(store: Store, thread_id: str, statuses=LIVE) -> list[dict]:
    marks = ",".join("?" * len(statuses))
    rows = [dict(r) for r in store.all(f"SELECT * FROM thread_links WHERE thread_id = ? AND status IN ({marks})", (thread_id, *statuses))]
    for r in rows:
        r["time_start"] = _episode_time(store, r["episode_id"])
    rows = [r for r in rows if r["time_start"]]  # an Episode deleted since is gone from the thread too
    return sorted(rows, key=lambda r: r["time_start"])


def threads_of(store: Store, episode_id: str, statuses=LIVE) -> list[str]:
    marks = ",".join("?" * len(statuses))
    return [r["thread_id"] for r in store.all(f"SELECT thread_id FROM thread_links WHERE episode_id = ? AND status IN ({marks})", (episode_id, *statuses))]


def refused(store: Store, thread_id: str, episode_id: str) -> bool:
    return store.one("SELECT 1 FROM thread_links WHERE thread_id = ? AND episode_id = ? AND status = 'rejected'", (thread_id, episode_id)) is not None


def candidates_for(store: Store, related_episode_ids: list[str], now: str | None = None) -> list[dict]:
    """The threads DeepSeek is shown: those of the nearest existing Episodes, then the open ones that moved in the last
    RECENT_DAYS (a day's small step rarely resembles the thread's earlier steps closely enough to be found by content),
    each with its latest steps."""
    seen: list[str] = []
    for eid in related_episode_ids:
        for tid in threads_of(store, eid):
            if tid not in seen:
                seen.append(tid)
    if now:
        since = (parse_time(now) - timedelta(days=RECENT_DAYS)).isoformat()
        for row in store.all("SELECT t.thread_id FROM threads t JOIN thread_links l ON l.thread_id = t.thread_id JOIN episodes e ON e.episode_id = l.episode_id "
                             "WHERE t.status = 'open' AND l.status IN ('auto','approved') GROUP BY t.thread_id HAVING MAX(e.time_start) >= ? "
                             "ORDER BY MAX(e.time_start) DESC", (since,)):
            if row["thread_id"] not in seen:
                seen.append(row["thread_id"])
    out = []
    for tid in seen:
        t = thread(store, tid)
        steps = links(store, tid)
        out.append({"thread_id": tid, "title": t["title"], "status": t["status"],
                    "latest": [{"episode_id": s["episode_id"], "time": s["time_start"], "excerpt": gist(store.episode(s["episode_id"]))} for s in steps[-3:]]})
    out.sort(key=lambda t: t["status"] != "open")
    return out[:CANDIDATE_THREADS]


def overview(store: Store, now: datetime | None = None) -> list[dict]:
    """Every thread for the Studio: its steps in order (pending ones included), and whether it has gone quiet."""
    now = now or datetime.now(timezone.utc)
    out = []
    for row in store.all("SELECT * FROM threads ORDER BY updated_at DESC"):
        t = dict(row)
        steps = links(store, t["thread_id"], ("auto", "approved", "pending"))
        live = [s for s in steps if s["status"] in LIVE]
        last = max((s["time_start"] for s in live), default=t["created_at"])
        t["steps"] = [{**s, "content": (ep := store.episode(s["episode_id"]))["content"], "excerpt": ep.get("excerpt") or ""} for s in steps]
        t["pending"] = sum(1 for s in steps if s["status"] == "pending")
        t["quiet"] = t["status"] == "open" and now - parse_time(last) > timedelta(days=QUIET_DAYS)
        out.append(t)
    return out


def context_line(store: Store, episode_id: str) -> list[dict]:
    """For recall / injection: each live thread this Episode is in, its place, and the Episodes before and after."""
    out = []
    for tid in threads_of(store, episode_id):
        t = thread(store, tid)
        steps = links(store, tid)
        ids = [s["episode_id"] for s in steps]
        if episode_id not in ids:
            continue
        i = ids.index(episode_id)
        near = lambda j: ({"episode_id": ids[j], "time": steps[j]["time_start"], "content": gist(store.episode(ids[j]), 80)}  # noqa: E731
                          if 0 <= j < len(ids) else None)
        out.append({"thread_id": tid, "title": t["title"], "status": t["status"], "position": i + 1, "count": len(ids),
                    "before": near(i - 1), "after": near(i + 1), "has_story": bool(t["story"])})
    return out


# ------------------------------------------------------------ writing (inside the caller's transaction)

def _link(store: Store, thread_id: str, episode_id: str, status: str, confidence, reason: str, decision_id: str | None, by: str) -> str:
    now = now_iso()
    store.db.execute("INSERT INTO thread_links (thread_id, episode_id, status, confidence, reason, decision_id, created_at, decided_at, decided_by) VALUES (?,?,?,?,?,?,?,?,?) "
                     "ON CONFLICT(thread_id, episode_id) DO UPDATE SET status=excluded.status, confidence=excluded.confidence, reason=excluded.reason, "
                     "decision_id=excluded.decision_id, decided_at=excluded.decided_at, decided_by=excluded.decided_by",
                     (thread_id, episode_id, status, confidence, reason[:300], decision_id, now, now if status != "pending" else None, by))
    store.db.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
    store._dirty["episodes"].add(episode_id)
    return status


def attach_proposal(store: Store, episode_id: str, proposal: dict, decision_id: str | None) -> dict | None:
    """Apply DeepSeek's thread proposal for a new Episode: continue a thread, or start one with earlier Episodes."""
    conf = float(proposal.get("confidence") or 0)
    status = "auto" if conf >= AUTO_CONFIDENCE else "pending"
    reason = str(proposal.get("reason") or "")
    if proposal.get("thread_id"):
        tid = proposal["thread_id"]
        if refused(store, tid, episode_id):
            return {"thread_id": tid, "status": "refused"}
        more = _clean_aliases(proposal.get("aliases"))
        if more:
            have = json.loads(thread(store, tid).get("aliases") or "[]")
            store.db.execute("UPDATE threads SET aliases = ? WHERE thread_id = ?", (json.dumps(_clean_aliases(have + more), ensure_ascii=False), tid))
            _touch_steps(store, tid)
        _link(store, tid, episode_id, status, conf, reason, decision_id, "deepseek")
        if proposal.get("over") and status == "auto":
            store.db.execute("UPDATE threads SET status = 'closed', closed_at = ?, updated_at = ? WHERE thread_id = ?", (now_iso(), now_iso(), tid))
            return {"thread_id": tid, "status": status, "closed": True}
        return {"thread_id": tid, "status": status}
    title = str(proposal.get("new_title") or "").strip()[:40]
    others = [e for e in proposal.get("with_episode_ids") or [] if e != episode_id]
    if not title or not others:
        return None
    # A "new" thread named like an open one is that one (the same story was proposed twice): join it.
    same = store.one("SELECT thread_id FROM threads WHERE status = 'open' AND replace(title, ' ', '') = ?", (title.replace(" ", ""),))
    if same:
        for eid in [*others, episode_id]:
            if not refused(store, same["thread_id"], eid) and not store.one("SELECT 1 FROM thread_links WHERE thread_id = ? AND episode_id = ? AND status IN ('auto','approved')", (same["thread_id"], eid)):
                _link(store, same["thread_id"], eid, status, conf, reason, decision_id, "deepseek")
        return {"thread_id": same["thread_id"], "status": status, "joined": True}
    now = now_iso()
    tid = mint("thread")
    store.db.execute("INSERT INTO threads (thread_id, title, status, created_at, updated_at, aliases) VALUES (?,?,?,?,?,?)",
                     (tid, title, "open", now, now, json.dumps(_clean_aliases(proposal.get("aliases")), ensure_ascii=False)))
    for eid in [*others, episode_id]:
        _link(store, tid, eid, status, conf, reason, decision_id, "deepseek")
    if proposal.get("over") and status == "auto":  # a short story that began and ended within these steps
        store.db.execute("UPDATE threads SET status = 'closed', closed_at = ?, updated_at = ? WHERE thread_id = ?", (now_iso(), now_iso(), tid))
    return {"thread_id": tid, "status": status, "new": True}


# ------------------------------------------------------------ the user's decisions

def decide(store: Store, body: dict) -> dict:
    """approve / take_off (single, remembered) / move (to another thread) / close / reopen / rename."""
    action = str(body.get("action") or "")
    tid = str(body.get("thread_id") or "")
    eid = str(body.get("episode_id") or "")
    with store.tx() as db:
        if action in ("approve", "take_off", "move"):
            if not store.one("SELECT 1 FROM thread_links WHERE thread_id = ? AND episode_id = ?", (tid, eid)):
                raise NotFound("that Episode is not on that thread")
            _link(store, tid, eid, "approved" if action == "approve" else "rejected", None, "user decision", None, identity.USER_ACTOR)
            if action == "move":
                target = str(body.get("to_thread_id") or "")
                thread(store, target)
                _link(store, target, eid, "approved", None, "moved here by the user", None, identity.USER_ACTOR)
        elif action in ("close", "reopen"):
            thread(store, tid)
            db.execute("UPDATE threads SET status = ?, closed_at = ?, updated_at = ? WHERE thread_id = ?",
                       ("closed" if action == "close" else "open", now_iso() if action == "close" else None, now_iso(), tid))
        elif action == "rename":
            title = str(body.get("title") or "").strip()[:40]
            if not title:
                raise Invalid("title is required")
            db.execute("UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?", (title, now_iso(), tid))
            _touch_steps(store, tid)
        else:
            raise Invalid("action must be approve, take_off, move, close, reopen or rename")
    store.audit(f"thread.{action}", tid or eid, after={"episode_id": eid, "to": body.get("to_thread_id")}, actor=identity.USER_ACTOR)
    return {"ok": True}


def move_candidates(store: Store, read, episode_id: str) -> list[dict]:
    """Other threads this Episode might belong to, found by local retrieval only (no model call)."""
    ep = store.episode(episode_id)
    _, near = read.related_units(ep["content"], k_patterns=0, k_episodes=8)
    have = set(threads_of(store, episode_id, ("auto", "approved", "pending", "rejected")))
    return [t for t in candidates_for(store, [e["episode_id"] for e in near if e["episode_id"] != episode_id]) if t["thread_id"] not in have]


# ------------------------------------------------------------ the whole story, once it is over



def write_story(store: Store, client, thread_id: str) -> dict:
    t = thread(store, thread_id)
    if t["status"] != "closed":
        raise Invalid("close the thread first: a story is written once it is over")
    steps = links(store, thread_id)
    if len(steps) < 2:
        raise Invalid("a story needs at least two Episodes")
    # The excerpts carry the line of the story; the full accounts are there for its details.
    payload = {"title": t["title"], "episodes": [{"time": s["time_start"], "excerpt": (ep := store.episode(s["episode_id"])).get("excerpt") or "", "content": ep["content"]} for s in steps]}
    answer, meta = client.chat_json(prompts.get("story"), json.dumps(payload, ensure_ascii=False), max_tokens=2500)
    story = str((answer or {}).get("story") or "").strip()
    if not story:
        raise Invalid("DeepSeek returned no story")
    with store.tx() as db:
        db.execute("UPDATE threads SET story = ?, story_version = story_version + 1, story_at = ?, updated_at = ? WHERE thread_id = ?",
                   (story[:4000], now_iso(), now_iso(), thread_id))
    store.audit("thread.story", thread_id, after={"episodes": len(steps), "usage": meta.get("usage", {})}, actor="deepseek")
    return thread(store, thread_id)
