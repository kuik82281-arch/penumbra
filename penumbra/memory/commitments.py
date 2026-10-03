"""Promises and plans: the to-do list the assistant carries, and how a commitment sinks.

An Episode with kind='commitment' is a daily promise or plan of either of them ("明天给你打领带", "今天不去找你"). It has a
unique tag (日常承诺·10月1日·领带), an owner, a due time and a status:

    open       the assistant sees it on every turn until it is done
    done       DeepSeek read in later RAW that it happened (or the user marked it) -> it sinks
    cancelled  called off -> it sinks
    missed     its day is over, it was not done, and every RAW up to then has been read: the assistant sees it ONCE as
               "没做到"; after that turn it sinks too

A sunk commitment stays in memory (searchable, with its details); it is only never brought up by itself again.
kind='vow' (a sincere, lasting promise, an important day) never sinks and is found by ordinary retrieval when relevant.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .actions import SUNK_IMPORTANCE
from .store import now_iso, parse_time

MISSED_GRACE_HOURS = 6  # after the due time: the night's consolidation gets to read the day first


def view(e: dict) -> dict:
    return {"episode_id": e["episode_id"], "tag": e.get("tag") or "", "content": e["content"], "owner": e.get("owner") or "", "due_at": e.get("due_at"),
            "status": e.get("commit_status") or "", "made_at": e["time_start"], "version": e["version"]}


def _read_up_to(core, when: datetime) -> bool:
    """Every RAW up to `when` has been through discovery, and nothing is waiting for DeepSeek: a missing "done" means not done."""
    pending = core.pipeline.unprocessed_raw()
    if pending and parse_time(pending[0]["createdAt"]) <= when:
        return False
    return core.store.one("SELECT COUNT(*) FROM candidates WHERE status='STAGING'")[0] == 0


def refresh(core, now: datetime | None = None) -> list[str]:
    """Open commitments whose day is over (and read) become missed. Returns the ids that changed."""
    now = now or datetime.now(timezone.utc)
    changed = []
    for e in core.store.episodes(status="active"):
        if e.get("kind") != "commitment" or e.get("commit_status") != "open" or not e.get("due_at"):
            continue
        due = parse_time(e["due_at"])
        if now >= due + timedelta(hours=MISSED_GRACE_HOURS) and _read_up_to(core, due):
            core.store.update_episode(e["episode_id"], {"commit_status": "missed"}, actor="system", reason="due passed without being done")
            changed.append(e["episode_id"])
    return changed


def for_assistant(core, now: datetime | None = None) -> dict:
    """What the turn shows him: open commitments (soonest first) and missed ones he has not mentioned yet."""
    refresh(core, now)
    rows = [e for e in core.store.episodes(status="active") if e.get("kind") == "commitment"]
    open_ = sorted((e for e in rows if e.get("commit_status") == "open"), key=lambda e: e.get("due_at") or "")
    missed = [e for e in rows if e.get("commit_status") == "missed" and not e.get("missed_told_at")]
    return {"open": [view(e) for e in open_], "missed": [view(e) for e in missed]}


def mark_told(core, episode_ids: list[str]) -> list[str]:
    """the assistant's turn carried these missed commitments: he has had his one chance to bring them up - they sink now."""
    done = []
    for eid in dict.fromkeys(str(i) for i in episode_ids or []):
        e = core.store.episode(eid)
        if e.get("kind") == "commitment" and e.get("commit_status") == "missed" and not e.get("missed_told_at"):
            when = now_iso()
            core.store.update_episode(eid, {"missed_told_at": when, "resolved_at": when, "importance": min(float(e["importance"]), SUNK_IMPORTANCE)},
                                      actor="system", reason="missed commitment shown to the assistant once")
            done.append(eid)
    return done


def for_verification(core) -> list[dict]:
    """Every commitment not settled yet, for DeepSeek: a RAW that fulfils, cancels or repeats one updates it by id."""
    rows = [e for e in core.store.episodes(status="active") if e.get("kind") == "commitment" and e.get("commit_status") in ("open", "missed")]
    return [view(e) for e in rows]
