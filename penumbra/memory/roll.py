"""Roll back: the user took a stretch of the chat back (her client removed those messages from the conversation).

The memory follows: Episodes told only from those messages are deleted like any memory she deletes, and the messages
themselves are marked `rolled` - never mined again (pipeline), never found by a search of the originals (retrieval).
The append-only RAW files are not rewritten: the text stays in the archive on her disk, and nothing reads it.

An Episode that also leans on other messages is left alone (it cannot be split here); the answer says how many.
"""
from __future__ import annotations

import json
import os
import uuid

from .. import files, identity
from ..errors import Invalid
from .edit import detach_episode
from .store import now_iso

MAX_ITEMS = 2000


def forget_chat(core, body: dict) -> dict:
    conversation_id = str(body.get("conversationId") or "")
    source = str(body.get("source") or os.environ.get("PENUMBRA_CHAT_SOURCE") or "chat")
    item_ids = [str(i) for i in body.get("itemIds") or [] if str(i or "")][:MAX_ITEMS]
    if not conversation_id or not item_ids:
        raise Invalid("conversationId and itemIds are required")
    wanted = [files.original_id(source, conversation_id, item) for item in item_ids]
    have: set[str] = set()
    with core.service.lock:
        for start in range(0, len(wanted), 400):
            chunk = wanted[start:start + 400]
            rows = core.service.conn.execute(f"SELECT id FROM originals WHERE id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
            have.update(r["id"] for r in rows)
    store = core.store
    deleted, partial = [], []
    with store.tx() as db:
        for episode in store.episodes():
            sources = set(episode["source_raw_ids"])
            if not sources & have:
                continue
            if not sources <= have:
                partial.append(episode["episode_id"])
                continue
            for pattern in store.patterns_supported_by(episode["episode_id"]):
                if pattern["supporting_episode_ids"] == [episode["episode_id"]]:
                    store.delete_pattern(pattern["pattern_id"])
                else:
                    detach_episode(store, pattern["pattern_id"], episode["episode_id"], identity.USER_ACTOR)
            store.delete_episode(episode["episode_id"])
            deleted.append(episode["episode_id"])
        fresh = sorted(have - store.rolled_raw_ids())
        if fresh:
            db.execute("INSERT INTO tombstones (tombstone_id, kind, raw_ids, deleted_at) VALUES (?,?,?,?)",
                       (f"tomb_{uuid.uuid4().hex[:16]}", "rolled", json.dumps(fresh), now_iso()))
        store.audit("chat.rolled", conversation_id, after={"raw": len(have), "episodes": len(deleted), "partial": len(partial)}, actor=identity.USER_ACTOR)
    return {"raw": len(have), "deleted_episodes": deleted, "partial_episodes": partial}
