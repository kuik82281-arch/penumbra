"""the assistant's dreams in long-term memory: a summary only, pointing at the dream itself.

The dream's full text lives in the bridge (潮汐, `dream:<id>`); here DeepSeek writes a short first-person summary of it,
stored as an Episode of kind 'dream' with `source_ref` = "dream:<id>". It is searched like any memory and labelled a
dream wherever it is shown; it is never evidence (read.related_units skips it), so nothing real is updated, corrected or
threaded from a dream. Deleting it is the user's 删除: the Episode goes, a tombstone keeps the ref, and the same dream is
not summarized again.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from .. import identity, prompts
from ..errors import Invalid
from .quotes import unverified_quotes
from .store import normalize_time, now_iso




def ref_of(dream_id: str) -> str:
    return f"dream:{dream_id}"


def _episode_for(store, ref: str) -> dict | None:
    row = store.one("SELECT episode_id FROM episodes WHERE kind = 'dream' AND source_ref = ?", (ref,))
    return store.episode(row["episode_id"]) if row else None


def remember(core, body: dict) -> dict:
    """{dreamId, text, time, mood?, dreamer?} -> the dream's Episode (created once; a deleted one is not created again)."""
    dream_id = str(body.get("dreamId") or "").strip()
    text = str(body.get("text") or "").strip()
    if not dream_id or not text:
        raise Invalid("dreamId and text are required")
    store, ref = core.store, ref_of(dream_id)
    if ref in store.tombstoned_raw_ids():
        return {"status": "tombstoned", "episode": None}
    existing = _episode_for(store, ref)
    if existing:
        return {"status": "existing", "episode": core.read.episode_view(existing)}
    client = core.verifier.client
    if not client.available():
        raise Invalid("DeepSeek is not configured")
    payload = {"dream": text[:6000], "mood_on_waking": str(body.get("mood") or "")[:80]}
    hers = identity.is_user(body.get("dreamer"))  # the user's own dream, told to the assistant
    answer, meta = client.chat_json(prompts.get("her_dream" if hers else "dream"), json.dumps(payload, ensure_ascii=False), max_tokens=600)
    summary = str((answer or {}).get("summary") or "").strip()[:600]
    if not summary:
        raise Invalid("DeepSeek returned no summary")
    bad = unverified_quotes([summary], [text])
    if bad:  # a sentence that is not in the dream cannot become "what I dreamt"
        raise Invalid(f"unverified quote: “{bad[0][:60]}”")
    keywords = [str(k)[:20] for k in (answer.get("keywords") or []) if str(k).strip()][:5]
    stamp = normalize_time(body.get("time")) if body.get("time") else now_iso()
    episode = store.insert_episode({
        "content": summary, "time_start": stamp, "time_end": stamp, "topics": ["她的梦" if hers else "梦", *keywords], "importance": 0.35, "confidence": 1.0,
        "source_raw_ids": [], "origin": "dream", "kind": "dream", "tag": "她的梦" if hers else "梦", "owner": identity.USER_ACTOR if hers else "assistant", "source_ref": ref}, actor="deepseek", reason="dream")
    store.audit("dream.remembered", episode["episode_id"], after={"ref": ref, "usage": meta.get("usage", {})}, actor="deepseek")
    return {"status": "created", "episode": core.read.episode_view(episode)}


def forget(core, dream_id: str) -> dict:
    """She deleted the dream: its summary goes the way of any deleted memory (tombstoned, never made again)."""
    if not dream_id:
        raise Invalid("dreamId is required")
    ref = ref_of(dream_id)
    episode = _episode_for(core.store, ref)
    if episode:
        core.store.delete_episode(episode["episode_id"])
    elif ref not in core.store.tombstoned_raw_ids():
        with core.store.tx() as db:  # deleted before it was ever summarized: still never summarize it
            db.execute("INSERT INTO tombstones (tombstone_id, kind, raw_ids, deleted_at) VALUES (?,?,?,?)",
                       (f"tomb_{uuid.uuid4().hex[:16]}", "dream", json.dumps([ref]), datetime.now(timezone.utc).isoformat()))
    return {"deleted": bool(episode)}
