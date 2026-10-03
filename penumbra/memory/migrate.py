"""One-time migration of every older memory path into RAW -> Episode -> Pattern.

  Pattern Memory (patterns.py, pattern-memory.sqlite)   attachments carried over; Patterns / Episodes converted with their ids
                                                        and RAW refs; proposals and runs recorded in the audit, not revived
  EVENTs (events/*.json)                                each lineage becomes one Episode whose versions are the EVENT's versions
  the user's manual memories (manual/*.json), curated      her words become RAW (source manual_note) and the Episode cites it
    memories (memories/*.md)
  pending candidates                                     QUARANTINE, so the user can send them through DeepSeek Verification again

Anything whose RAW references cannot be resolved goes to QUARANTINE instead of becoming an Episode. The old files are
never modified or deleted: they stay where they are as history, and nothing reads them any more. Each source is migrated
once (markers in the store); every converted item carries `legacy:<kind>:<id>` so a re-run cannot duplicate it.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .. import identity
from .. import files
from ..errors import NotFound
from .store import dumps, mint, normalize_time, now_iso, parse_time

LEGACY_DB = "pattern-memory.sqlite"


def _exists_raw(core, ids: list[str]) -> bool:
    with core.service.lock:
        return bool(ids) and all(core.service.conn.execute("SELECT 1 FROM originals WHERE id = ?", (i,)).fetchone() for i in ids)


def _quarantine(core, gist: str, raw_ids: list[str], origin: str, note: str) -> None:
    digest = mint("legacy")
    with core.store.tx() as db:
        db.execute("INSERT INTO candidates (candidate_id, origin, kind_hint, gist, raw_ids, digest, status, last_error, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (mint("cand"), "migration", "legacy", gist[:160], dumps(raw_ids), f"{origin}:{digest}", "QUARANTINE", note[:300], now_iso(), now_iso()))


def _seen(core, legacy_ref: str) -> bool:
    return core.store.one("SELECT 1 FROM episodes WHERE candidate_id = ?", (legacy_ref,)) is not None


def _pattern_snapshot(core) -> dict:
    path = core.service.config.ledger_dir / LEGACY_DB
    if not path.exists():
        return {"skipped": "no legacy pattern-memory database"}
    report = {"attachments": 0, "episodes": 0, "patterns": 0, "quarantined": 0}
    db = sqlite3.connect(path)
    try:
        row = db.execute("SELECT body FROM snapshot WHERE id = 1").fetchone()
    except sqlite3.Error:
        row = None
    finally:
        db.close()
    state = json.loads(row[0]) if row else {}
    for a in (state.get("attachments") or {}).values():
        try:
            core.store.attachment(a["attachment_id"])
        except NotFound:
            core.store.upsert_attachment(a)
            report["attachments"] += 1
    episodes, patterns = state.get("episodes") or {}, state.get("patterns") or {}
    for pid, old in patterns.items():
        ep_ids, states, stamp = [], [], None
        for eid in old.get("episodes", []):
            e = episodes.get(eid)
            if not e or e.get("status") != "active":
                continue
            ref = f"legacy:episode:{eid}"
            sources = list(e.get("evidence_raw_ids") or [])
            if not _exists_raw(core, sources):
                _quarantine(core, e.get("summary", ""), sources, "legacy-episode", "RAW references could not be resolved")
                report["quarantined"] += 1
                continue
            when = normalize_time(e.get("occurred_at"))
            core.store.insert_episode({"episode_id": eid, "content": e["summary"], "time_start": when, "time_end": when, "entities": [], "topics": list(old.get("tags") or [])[:8],
                                       "state": e.get("state_change") or "", "importance": 0.6, "confidence": 0.8, "source_raw_ids": sources,
                                       "attachment_ids": list(e.get("attachment_ids") or []), "origin": "migration", "candidate_id": ref}, actor="migration", reason="from pattern memory")
            ep_ids.append(eid)
            report["episodes"] += 1
            if e.get("state_change") and (not states or states[-1]["state"] != e["state_change"]):
                if states:
                    states[-1]["valid_to"] = when
                states.append({"state_id": f"s{len(states) + 1}", "state": e["state_change"], "valid_from": when, "valid_to": None, "episode_ids": [eid]})
            elif states:
                states[-1]["episode_ids"].append(eid)
            stamp = when
        if not ep_ids:
            continue
        current = old.get("current_state") or (states[-1]["state"] if states else old["title"])
        if not states or states[-1]["state"] != current:
            first = core.store.episode(ep_ids[-1])["time_end"]
            if states:
                states[-1]["valid_to"] = first
            states.append({"state_id": f"s{len(states) + 1}", "state": current, "valid_from": first, "valid_to": None, "episode_ids": ep_ids[-1:]})
        first_time = min(core.store.episode(i)["time_start"] for i in ep_ids)
        core.store.insert_pattern({"pattern_id": pid, "title": old["title"], "topic": old.get("kind", ""), "narrative": old.get("summary") or old["title"], "current_state": current,
                                   "current_state_since": states[-1]["valid_from"], "states": states, "supporting_episode_ids": ep_ids, "entities": [], "valid_from": first_time,
                                   "confidence": 0.8, "status": "active" if old.get("status") == "active" else "archived"}, actor="migration", reason="from pattern memory")
        for eid in ep_ids:
            core.store.add_relation("supports", "episode", eid, "pattern", pid)
            core.store.add_relation("derived_from", "pattern", pid, "episode", eid)
        report["patterns"] += 1
    core.store.audit("migration.pattern_memory", "pattern-memory", after={**report, "proposals": len(state.get("proposals") or {}), "runs": len(state.get("runs") or {}),
                                                                       "processed_raw": len(state.get("processed") or [])}, actor="migration")
    return report


def _latest_by_lineage(docs: list[dict], id_key: str) -> dict[str, list[dict]]:
    chains: dict[str, list[dict]] = {}
    for d in docs:
        chains.setdefault(d.get("lineageId") or d[id_key], []).append(d)
    return {k: sorted(v, key=lambda d: (d.get("version", 1), d.get("createdAt", ""))) for k, v in chains.items()}


def _events(core, data: Path) -> dict:
    report = {"episodes": 0, "versions": 0, "quarantined": 0}
    for lineage, chain in _latest_by_lineage(list(files.iter_json_docs(data / "events")), "eventId").items():
        ref = f"legacy:event:{lineage}"
        if _seen(core, ref):
            continue
        confirmed = [d for d in chain if d.get("status") == "confirmed"]
        newest = chain[-1]
        sources = list(newest.get("sources") or newest.get("sourceOriginalIds") or [])
        if not _exists_raw(core, sources):
            _quarantine(core, newest.get("title", ""), sources, "legacy-event", "EVENT sources could not be resolved")
            report["quarantined"] += 1
            continue
        when = lambda d: normalize_time((d.get("effectiveAt") or (d.get("eventTime") or {}).get("start") or d.get("createdAt")))  # noqa: E731
        first = chain[0]
        ep = core.store.insert_episode({"content": first["content"], "time_start": when(first), "time_end": when(first), "entities": list(first.get("entities") or [])[:12],
                                        "topics": list(first.get("tags") or [])[:8], "importance": float(first.get("importance", 0.5)), "confidence": 0.85,
                                        "source_raw_ids": list(first.get("sources") or first.get("sourceOriginalIds") or []), "origin": "migration", "candidate_id": ref},
                                       actor="migration", reason=f"EVENT {first['eventId']} v{first.get('version', 1)}")
        report["episodes"] += 1
        for later in chain[1:]:
            core.store.update_episode(ep["episode_id"], {"content": later["content"], "time_start": when(later), "time_end": when(later),
                                                         "entities": list(later.get("entities") or [])[:12], "topics": list(later.get("tags") or [])[:8],
                                                         "source_raw_ids": list(dict.fromkeys([*core.store.episode(ep["episode_id"])["source_raw_ids"], *(later.get("sources") or [])]))},
                                      actor="migration", reason=f"EVENT {later['eventId']} v{later.get('version')}")
            report["versions"] += 1
        if not confirmed:  # the lineage ended archived / superseded
            core.store.update_episode(ep["episode_id"], {"status": "archived"}, actor="migration", reason="EVENT was not confirmed at migration time")
    return report


def _note(core, doc: dict, text: str, when: str, ref: str, tags, entities, importance) -> bool:
    if _seen(core, ref) or not text.strip():
        return False
    result = core.service.ingest_originals(f"{identity.USER_ACTOR}-manual", "manual-notes", [{"id": ref.replace(":", "-"), "role": "user", "content": text, "createdAt": when, "sourceType": "manual_note"}])
    raw = (result["added"] or result["existing"])[0]
    core.store.insert_episode({"content": text, "time_start": when, "time_end": when, "entities": list(entities or [])[:12], "topics": list(tags or [])[:8], "importance": float(importance),
                               "confidence": 1.0, "source_raw_ids": [raw], "origin": "migration", "candidate_id": ref}, actor="migration", reason="manual memory")
    return True


def _manual(core, data: Path) -> dict:
    report = {"episodes": 0, "archived": 0}
    for lineage, chain in _latest_by_lineage(list(files.iter_json_docs(data / "manual")), "memoryId").items():
        newest = chain[-1]
        if newest.get("status") != "confirmed":
            report["archived"] += 1
            continue
        when = normalize_time(newest.get("effectiveAt") or newest.get("createdAt"))
        report["episodes"] += _note(core, newest, newest["content"], when, f"legacy:manual:{lineage}", newest.get("tags"), newest.get("entities"), newest.get("importance", 0.6))
    return report


def _curated(core, data: Path) -> dict:
    report = {"episodes": 0, "quarantined": 0}
    for memory in files.iter_memories(data / "memories"):
        if memory.status == "confirmed" and (identity.is_user(memory.author) or identity.is_assistant(memory.author) or memory.author == "import"):
            when = normalize_time(memory.created)
            report["episodes"] += _note(core, {}, f"{memory.title}\n{memory.body}".strip(), when, f"legacy:memory:{memory.id}", memory.keywords, [], 0.6)
        else:
            _quarantine(core, memory.title, list(memory.sources), "legacy-memory", f"curated memory {memory.id} was {memory.status} ({memory.author})")
            report["quarantined"] += 1
    return report


def _candidates(core, data: Path) -> dict:
    report = {"quarantined": 0}
    for doc in files.iter_json_docs(data / "candidates"):
        if doc.get("status") != "candidate":
            continue
        sources = list(doc.get("sourceOriginalIds") or [])
        _quarantine(core, doc.get("proposedTitle", ""), sources if _exists_raw(core, sources) else [], "legacy-candidate", "pending candidate from the daily event review")
        report["quarantined"] += 1
    return report


def run(core) -> dict:
    marker = core.store.kv_get("migration", {}) | {}
    data = core.service.config.data_dir
    report: dict = {}
    steps = (("pattern_memory", lambda: _pattern_snapshot(core)), ("events", lambda: _events(core, data)), ("manual", lambda: _manual(core, data)),
             ("curated_memories", lambda: _curated(core, data)), ("candidates", lambda: _candidates(core, data)))
    for name, step in steps:
        if marker.get(name):
            continue
        report[name] = step()
        marker[name] = now_iso()
        core.store.kv_set("migration", marker)
    legacy = core.service.config.ledger_dir / LEGACY_DB
    if legacy.exists() and marker.get("pattern_memory"):
        for suffix in ("", "-wal", "-shm"):
            path = legacy.with_name(LEGACY_DB + suffix)
            try:
                if path.exists():
                    path.replace(path.with_name(path.name + ".migrated"))
            except OSError:
                pass  # still open elsewhere: the marker already prevents a second import
    return report
