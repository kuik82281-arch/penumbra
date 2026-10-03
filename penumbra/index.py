"""index.sqlite: a disposable search index.

RAW (originals/*.jsonl) is the truth of what was said; Episodes and Patterns live in memory/core.sqlite and the user's
preference documents in preferences/. Deleting this file loses nothing: `rebuild()` replays the originals and projects
the Episodes, Patterns and active preference chunks back into the retrieval tables.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import files
from .text import bigram_terms, unigram_terms

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS originals (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  item_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  ingested_at TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  claude_session_id TEXT,
  turn_id TEXT,
  source_type TEXT NOT NULL DEFAULT 'chat_message',
  meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS originals_conv ON originals(conversation_id, created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS originals_fts USING fts5(
  id UNINDEXED, bi, uni, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  injection TEXT NOT NULL,
  status TEXT NOT NULL,
  author TEXT NOT NULL,
  keywords TEXT NOT NULL,
  sources TEXT NOT NULL,
  created TEXT NOT NULL,
  updated TEXT NOT NULL,
  -- Retrieval metadata. kind: pattern | episode (memory/) | preference (a preference-document chunk) | memory (generic).
  kind TEXT NOT NULL DEFAULT 'memory',
  importance REAL NOT NULL DEFAULT 0.5,
  effective_at TEXT,
  lineage_id TEXT,
  version INTEGER NOT NULL DEFAULT 1,
  entities TEXT NOT NULL DEFAULT '[]',
  tags TEXT NOT NULL DEFAULT '[]'
);
-- Field-separated so bm25() can score each field on its own: title / body / entities / tags (curated keywords).
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  id UNINDEXED, title_bi, body_bi, ent_bi, tag_bi, uni, tokenize = 'unicode61 remove_diacritics 2'
);
-- Document frequency per term, for low-information term suppression.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vocab USING fts5vocab(memories_fts, 'row');
CREATE VIRTUAL TABLE IF NOT EXISTS originals_vocab USING fts5vocab(originals_fts, 'row');
"""


# Write listeners per connection (P2-B): the vector store and entity lexicon of the service owning `conn` hear about
# every projected write. Silenced while `rebuild()` replays everything; the owner reloads in bulk afterwards.
_listeners: dict[int, object] = {}
_paused: set[int] = set()


def set_listener(conn: sqlite3.Connection, listener) -> None:
    _listeners[id(conn)] = listener


def _notify(conn: sqlite3.Connection, kind: str, payload: dict) -> None:
    listener = _listeners.get(id(conn))
    if listener is not None and id(conn) not in _paused:
        listener(kind, payload)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


class OriginalConflict(ValueError):
    """An original id already projected with different content: originals are immutable."""


def _schema_statements() -> list[str]:
    """SCHEMA as single statements without its PRAGMAs, so a rebuild can recreate it inside one transaction."""
    statements, current = [], ""
    for line in SCHEMA.splitlines(keepends=True):
        current += line
        if sqlite3.complete_statement(current):
            if not current.strip().upper().startswith("PRAGMA"):
                statements.append(current.strip())
            current = ""
    return statements


def put_original(conn: sqlite3.Connection, original: files.Original) -> bool:
    """Project one original. The same original again is a no-op (False); other content under its id is a conflict."""
    inserted = conn.execute(
        "INSERT INTO originals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
        (
            original.id,
            original.source,
            original.conversation_id,
            original.item_id,
            original.role,
            original.content,
            original.created_at,
            original.ingested_at,
            original.sha256,
            original.claude_session_id,
            original.turn_id,
            original.source_type,
            json.dumps({"metadata": original.metadata, "eventRefs": original.event_refs, "attachments": original.attachments}, ensure_ascii=False),
        ),
    ).rowcount
    if not inserted:
        row = conn.execute("SELECT sha256 FROM originals WHERE id = ?", (original.id,)).fetchone()
        if row is not None and row["sha256"] == original.sha256:
            return False
        raise OriginalConflict(f"original {original.id} is already indexed with different content")
    conn.execute(
        "INSERT INTO originals_fts (id, bi, uni) VALUES (?,?,?)",
        (original.id, " ".join(bigram_terms(original.content)), " ".join(unigram_terms(original.content))),
    )
    _notify(conn, "original", {"id": original.id, "content": original.content})
    return True


def put_memory(conn: sqlite3.Connection, memory: files.Memory, meta: dict | None = None) -> None:
    """Curated memories keep their keywords as tags. EVENTs pass `meta` (entities, tags, importance, time, lineage)."""
    meta = meta or {}
    entities = meta.get("entities", [])
    tags = meta.get("tags", memory.keywords if not meta else [])
    conn.execute("DELETE FROM memories WHERE id = ?", (memory.id,))
    conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory.id,))
    conn.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            memory.id,
            memory.title,
            memory.body,
            memory.injection,
            memory.status,
            memory.author,
            json.dumps(memory.keywords, ensure_ascii=False),
            json.dumps(memory.sources),
            memory.created,
            memory.updated,
            meta.get("kind", "memory"),
            float(meta.get("importance", 0.5)),
            meta.get("effectiveAt"),
            meta.get("lineageId"),
            int(meta.get("version", 1)),
            json.dumps(entities, ensure_ascii=False),
            json.dumps(tags, ensure_ascii=False),
        ),
    )
    conn.execute(
        "INSERT INTO memories_fts (id, title_bi, body_bi, ent_bi, tag_bi, uni) VALUES (?,?,?,?,?,?)",
        (
            memory.id,
            " ".join(bigram_terms(memory.title)),
            " ".join(bigram_terms(memory.body)),
            " ".join(t for e in entities for t in bigram_terms(e)),
            " ".join(t for k in tags for t in bigram_terms(k)),
            " ".join(unigram_terms(f"{memory.title} {' '.join([*entities, *tags])} {memory.body}")),
        ),
    )
    _notify(conn, "memory", {"id": memory.id, "kind": meta.get("kind", "memory"), "status": memory.status, "title": memory.title,
                             "body": memory.body, "entities": entities, "tags": tags})


def delete_memory(conn: sqlite3.Connection, memory_id: str, kind: str = "memory") -> None:
    """A deleted Episode / Pattern leaves the index and its vector (the vector store drops anything not current)."""
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
    _notify(conn, "memory", {"id": memory_id, "kind": kind, "status": "deleted", "title": "", "body": "", "entities": [], "tags": []})


def row_to_memory(row: sqlite3.Row) -> files.Memory:
    return files.Memory(
        id=row["id"],
        title=row["title"],
        body=row["body"],
        injection=row["injection"],
        status=row["status"],
        author=row["author"],
        keywords=json.loads(row["keywords"]),
        sources=json.loads(row["sources"]),
        created=row["created"],
        updated=row["updated"],
    )


def row_to_original(row: sqlite3.Row) -> files.Original:
    meta = json.loads(row["meta"] or "{}")
    return files.Original(
        id=row["id"],
        source=row["source"],
        conversation_id=row["conversation_id"],
        item_id=row["item_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
        sha256=row["sha256"],
        claude_session_id=row["claude_session_id"],
        turn_id=row["turn_id"],
        source_type=row["source_type"],
        metadata=meta.get("metadata") or {},
        event_refs=meta.get("eventRefs") or [],
        attachments=meta.get("attachments") or [],
    )


def rebuild(conn: sqlite3.Connection, originals_dir: Path, data_dir: Path | None = None, projectors: tuple = ()) -> dict:
    """Drop every derived row and replay the truth. `projectors` are callables(conn) that project their own truth (the
    Episodes and Patterns) into the retrieval tables inside this transaction. Returns counts, including immutability conflicts."""
    # One write transaction from the first DROP to the last row: another connection's rebuild waits for it instead
    # of recreating the tables between this one's DROP and its inserts (2026-09-25: two processes started together
    # and the second failed with "UNIQUE constraint failed: originals.id").
    conn.execute("BEGIN IMMEDIATE")
    _paused.add(id(conn))
    try:
        # Projections are disposable: drop and recreate them so schema changes apply to existing data dirs. Tables of the
        # retired lifecycle (candidates / events / batches / audit / seen / recall / inject logs) are dropped for good.
        for table in (
            "memories_vocab", "originals_vocab", "originals", "originals_fts", "memories", "memories_fts",
            "memory_stats", "recall_log", "inject_log", "seen_log", "candidates", "events", "batches", "audit_log", "audit_refs",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        for statement in _schema_statements():
            conn.execute(statement)
        seen: dict[str, str] = {}
        conflicts: list[str] = []
        originals = 0
        for original in files.iter_originals(originals_dir):
            previous = seen.get(original.id)
            if previous is not None:
                # First write wins; a later line with different content is a violation to surface, not apply.
                if previous != original.sha256:
                    conflicts.append(original.id)
                continue
            seen[original.id] = original.sha256
            put_original(conn, original)
            originals += 1
        report: dict = {"originals": originals, "conflicts": conflicts, "preferenceChunks": 0}
        if data_dir is not None:
            from .preferences import project_chunk, retrievable

            for doc in files.iter_json_docs(data_dir / "preferences" / "documents"):
                chunks_file = data_dir / "preferences" / "chunks" / f"{doc['id']}.json"
                if not retrievable(doc) or not chunks_file.exists():  # session_pinned / disabled: never in the index
                    continue
                for chunk in json.loads(chunks_file.read_text(encoding="utf-8"))["chunks"]:
                    if chunk["status"] == "active":
                        project_chunk(conn, doc, chunk, True)
                        report["preferenceChunks"] += 1
        for projector in projectors:
            report.update(projector(conn))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        _paused.discard(id(conn))
    return report
