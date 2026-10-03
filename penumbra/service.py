"""Penumbra core: RAW originals in, the unified memory (memory/) behind them, one retrieval engine on top.

    RAW (immutable originals)  ->  Episode  ->  Pattern        write path and read path: penumbra/memory/
    the user's preference documents (preferences.py) are authored reference documents, not a memory layer.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from . import identity
from . import files, index
from .config import Config
from .embeddings import EmbeddingProvider, provider_from_env
from .errors import Invalid, NotFound  # re-exported: api and tests import them from here
from .instance import DataDirLock
from .memory import MemoryCore
from .preferences import PreferenceStore
from .retrieval import RetrievalConfig, RetrievalCore, RetrievalRequest
from .vectors import VectorStore, embedding_text


def _parse_ts(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _opt_str(value) -> str | None:
    return str(value).strip()[:200] or None if isinstance(value, str) else None


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


class Penombre:
    def __init__(self, config: Config, retrieval: RetrievalConfig | None = None, embeddings: EmbeddingProvider | None = None,
                 discovery=None, verifier=None):
        self.config = config
        identity.load(config.data_dir)  # who the two people are (profile.json / environment), before any prompt is built
        # Before anything reads or rewrites the directory: a second process here raises instance.AlreadyRunning.
        self.data_lock = DataDirLock(config.data_dir)
        try:
            self._open(config, retrieval, embeddings, discovery, verifier)
        except BaseException:
            self.data_lock.release()
            raise

    def _open(self, config: Config, retrieval, embeddings, discovery, verifier) -> None:
        self.lock = threading.RLock()
        for directory in (config.originals_dir, config.memory_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.conn = index.connect(config.index_path)
        # One ranking for the memory read path and the debug views.
        self.retrieval = RetrievalCore(self, retrieval)
        # Embeddings: a persisted cache that outlives index rebuilds; the vector signal is off until a provider is ready.
        self.vectors = VectorStore(config.data_dir, embeddings if embeddings is not None else provider_from_env())
        index.set_listener(self.conn, self._on_index_write)
        self.last_rebuild_ms: float | None = None
        # The unified memory (Episodes, Patterns, staging, audit): its own durable store, projected into the index below.
        self.memory = MemoryCore(self, discovery, verifier)
        # Files are the truth: always start from a fresh projection.
        self.last_rebuild = self.rebuild(reload_vectors=False)
        # the user's preference documents: verbatim originals she publishes (session_pinned -> the Session Preference Pack,
        # retrieval -> locally cut chunks). No model is involved.
        self.preferences = PreferenceStore(self)
        self.memory_start = self.memory.start(scheduler=False)
        self._reload_vectors()

    def close(self) -> None:
        try:
            self.memory.close()
            self.vectors.close()
            self.conn.close()
        finally:
            self.data_lock.release()

    def _on_index_write(self, kind: str, payload: dict) -> None:
        """Every projected write reaches the vector index and the entity lexicon (paused while rebuilding)."""
        if kind == "original":
            self.vectors.sync("raw", payload["id"], "original", payload["content"], True)
            return
        self.retrieval.invalidate_entities()
        text = embedding_text(payload["kind"], payload["title"], payload["body"], payload["entities"], payload["tags"])
        self.vectors.sync(payload["kind"], payload["id"], "memory", text, payload["status"] == "confirmed")

    def _reload_vectors(self) -> None:
        entries = [("raw", r["id"], "original", r["content"]) for r in self.conn.execute("SELECT id, content FROM originals")]
        for r in self.conn.execute("SELECT id, kind, title, body, entities, tags FROM memories WHERE status = 'confirmed'"):
            entries.append((r["kind"], r["id"], "memory",
                            embedding_text(r["kind"], r["title"], r["body"], json.loads(r["entities"]), json.loads(r["tags"]))))
        self.vectors.reload(entries)
        self.retrieval.invalidate_entities()

    def _now(self) -> datetime:
        """Current time (tests replace this)."""
        return datetime.now(timezone.utc)

    def rebuild(self, reload_vectors: bool = True) -> dict:
        with self.lock:
            started = time.perf_counter()
            report = index.rebuild(self.conn, self.config.originals_dir, self.config.data_dir, projectors=(self.memory.read.project_all,))
            if reload_vectors:
                self._reload_vectors()
            self.last_rebuild_ms = round((time.perf_counter() - started) * 1000, 1)
            return {**report, "ms": self.last_rebuild_ms}

    # ------------------------------------------------------------ originals

    def ingest_originals(self, source: str, conversation_id: str, items: list[dict], context: dict | None = None) -> dict:
        """Store each item once, verbatim. `context` (per call) and per-item fields carry provenance: claudeSessionId,
        turnId, sourceType, metadata, eventRefs. Provenance is recorded with the first write and never changes."""
        context = context if isinstance(context, dict) else {}
        if not isinstance(items, list):
            raise Invalid("items must be a list")
        path = files.originals_path(self.config.originals_dir, source, conversation_id)
        added, existing, conflicts, skipped = [], [], [], []
        with self.lock:
            for item in items:
                item_id = str(item.get("id") or "")
                content = item.get("content")
                role = str(item.get("role") or "")
                incoming_attachments = [a for a in _list(item.get("attachments")) if isinstance(a, dict)][:20]
                if not item_id or not isinstance(content, str) or (not content.strip() and not incoming_attachments) or not role:
                    skipped.append(item_id or "?")
                    continue
                oid = files.original_id(source, conversation_id, item_id)
                digest = files.content_hash(content)
                row = self.conn.execute("SELECT sha256 FROM originals WHERE id = ?", (oid,)).fetchone()
                if row is not None:
                    # Originals are immutable: an identical resend is fine, a different text is refused.
                    if row["sha256"] == digest:
                        existing.append(oid)
                        if incoming_attachments:
                            current = self.get_original(oid)
                            current.attachments = incoming_attachments
                            self.memory.register_attachments(current)
                    else:
                        conflicts.append(oid)
                    continue
                original = files.Original(
                    id=oid,
                    source=source,
                    conversation_id=conversation_id,
                    item_id=item_id,
                    role=role,
                    content=content,
                    created_at=str(item.get("createdAt") or files.now_iso()),
                    ingested_at=files.now_iso(),
                    sha256=digest,
                    claude_session_id=_opt_str(item.get("claudeSessionId") or context.get("claudeSessionId")),
                    turn_id=_opt_str(item.get("turnId") or context.get("turnId")),
                    source_type=_opt_str(item.get("sourceType") or context.get("sourceType")) or "chat_message",
                    metadata={**_dict(context.get("metadata")), **_dict(item.get("metadata"))},
                    event_refs=[r for r in [*_list(context.get("eventRefs")), *_list(item.get("eventRefs"))] if isinstance(r, dict)][:20],
                    attachments=incoming_attachments,
                )
                files.append_jsonl(path, original.to_record())
                index.put_original(self.conn, original)
                self.memory.register_attachments(original)
                added.append(oid)
        if added:
            self.memory.activity()
        return {"added": added, "existing": existing, "conflicts": conflicts, "skipped": skipped}

    def get_original(self, original_id: str) -> files.Original:
        row = self.conn.execute("SELECT * FROM originals WHERE id = ?", (original_id,)).fetchone()
        if row is None:
            raise NotFound(f"no original {original_id}")
        return index.row_to_original(row)

    def original_record(self, original_id: str) -> dict:
        return self.memory.raw_record(original_id)

    # ------------------------------------------------------------ debug view of the ranking

    def retrieval_debug(self, payload: dict) -> dict:
        """Read-only full ranking for development: nothing is stored and nothing counts as seen."""
        current = payload.get("currentSession") if isinstance(payload.get("currentSession"), dict) else None
        since = _parse_ts(str(current.get("since") or "")) if current else None
        policy = str(payload.get("policy") or "recall")
        with self.lock:
            found = self.retrieval.retrieve(RetrievalRequest(
                query=str(payload.get("query") or ""), policy=policy, mode=str(payload.get("mode") or ("memory" if policy == "inject" else "balanced")),
                conversation_id=payload.get("conversationId"), session_id=payload.get("sessionId"),
                current_session=(str(current.get("conversationId") or ""), since) if since else None,
                top_k=int(payload["topK"]) if payload.get("topK") else None,
                channels=("memory",) if policy == "inject" else ("memory", "original"), include_static=False, exact=bool(payload.get("exact")),
            ))
        strip = lambda h: {k: v for k, v in h.items() if k != "content"} | {"content": h["content"][:200]}  # noqa: E731
        return {"trace": RetrievalCore.trace(found), "summary": found["summary"], "top": [strip(h) for h in found["top"]], "belowTop": [strip(h) for h in found["hits"][len(found["top"]):]][:20],
                "deduped": [strip(d) for d in found["dropped"]], "debug": found["debug"]}

    # ------------------------------------------------------------ stats

    def stats(self) -> dict:
        one = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "originals": one("SELECT COUNT(*) FROM originals"),
            "conversations": one("SELECT COUNT(DISTINCT conversation_id) FROM originals"),
            "memory": self.memory.store.counts(),
            "lastRebuild": self.last_rebuild,
        }
