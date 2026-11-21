"""Recall cues: written once per Episode version, in the background, so a memory can be found from another angle.

After Callback DLC's write-time cues (PolyForm-NC: the idea only, nothing copied) and doc2query: each Episode gets 0-4
short lines - its situation without the topic words, how she might ask about it, a telling detail, what was left open.
They are projected into the search index with the Episode (read._episode_memory) and never shown to anyone. Quotes are
checked against the RAW like any memory text; a cue that quotes words nobody said is dropped. Writing cues never bumps
the Episode's version.
"""
from __future__ import annotations

import json

from .. import prompts
from .quotes import unverified_quotes
from .store import Store

TYPES = ("处境", "会怎么问", "细节", "未完")
MAX_CUES = 4
MAX_CHARS = 40
RAW_CHARS = 1800


def pending(store: Store, limit: int) -> list[dict]:
    rows = store.all("SELECT episode_id FROM episodes WHERE status = 'active' AND kind != 'dream' AND cues_for != version "
                     "ORDER BY updated_at DESC LIMIT ?", (limit,))
    return [store.episode(r["episode_id"]) for r in rows]


def clean(answer: dict, sources: list[str]) -> list[str]:
    out: list[str] = []
    for cue in (answer or {}).get("cues") or []:
        if not isinstance(cue, dict):
            continue
        kind, text = str(cue.get("type") or "").strip(), " ".join(str(cue.get("text") or "").split())
        if kind not in TYPES or not (4 <= len(text) <= MAX_CHARS) or unverified_quotes([text], sources):
            continue
        line = f"{kind}：{text}"
        if line not in out:
            out.append(line)
    return out[:MAX_CUES]


def write(store: Store, raw_record, client, episode: dict) -> list[str]:
    raw: list[str] = []
    used = 0
    for rid in episode["source_raw_ids"][:12]:
        try:
            r = raw_record(rid)
        except Exception:  # noqa: BLE001 - a missing RAW just leaves less to go on
            continue
        text = (r.get("content") or "").strip()
        if not text or used >= RAW_CHARS:
            continue
        raw.append(f"{r.get('role')}: {text[: RAW_CHARS - used]}")
        used += len(text)
    payload = {"episode": episode["content"], "state": episode.get("state") or "", "raw": raw}
    answer, _meta = client.chat_json(prompts.get("cues"), json.dumps(payload, ensure_ascii=False), max_tokens=400)
    lines = clean(answer, [episode["content"], *raw])
    with store.tx() as db:
        db.execute("UPDATE episodes SET cues = ?, cues_for = ? WHERE episode_id = ?", ("\n".join(lines), episode["version"], episode["episode_id"]))
        store._dirty["episodes"].add(episode["episode_id"])  # re-projected into the index; the version stays
    return lines


def run(store: Store, raw_record, client, limit: int = 30) -> dict:
    done, failed = 0, 0
    for episode in pending(store, limit):
        try:
            write(store, raw_record, client, episode)
            done += 1
        except Exception:  # noqa: BLE001 - tried again on the next run
            failed += 1
    return {"written": done, "failed": failed}
