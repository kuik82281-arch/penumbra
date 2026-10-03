"""The durable store of the unified long-term memory: RAW -> Episode -> Pattern.

RAW is not stored here (it is the immutable original files + index.sqlite); this database holds what is derived from
it and everything engineering needs around it: STAGING candidates, DeepSeek decisions, QUARANTINE, versions, audit,
checkpoints, memory locks and seen state. It is the truth for Episodes and Patterns (WAL, synchronous FULL); the
retrieval index is a projection of it (read.py) and can always be rebuilt from here.

Every write goes through `tx()`: one transaction, nested calls join the outer one, and change callbacks (the index
projection) run after the outermost commit, outside the lock.
"""
from __future__ import annotations

import copy
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..errors import Invalid, NotFound

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  time_start TEXT NOT NULL,
  time_end TEXT NOT NULL,
  entities TEXT NOT NULL DEFAULT '[]',
  topics TEXT NOT NULL DEFAULT '[]',
  relations TEXT NOT NULL DEFAULT '[]',
  state TEXT NOT NULL DEFAULT '',
  importance REAL NOT NULL DEFAULT 0.5,
  confidence REAL NOT NULL DEFAULT 0.8,
  source_raw_ids TEXT NOT NULL,
  attachment_ids TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'active',
  superseded_by TEXT,
  origin TEXT NOT NULL DEFAULT 'pipeline',
  candidate_id TEXT,
  decision_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS episodes_time ON episodes(time_start);
CREATE TABLE IF NOT EXISTS episode_versions (
  episode_id TEXT NOT NULL, version INTEGER NOT NULL, doc TEXT NOT NULL, at TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (episode_id, version)
);
CREATE TABLE IF NOT EXISTS patterns (
  pattern_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  topic TEXT NOT NULL DEFAULT '',
  narrative TEXT NOT NULL,
  current_state TEXT NOT NULL,
  current_state_since TEXT,
  states TEXT NOT NULL,
  supporting_episode_ids TEXT NOT NULL,
  entities TEXT NOT NULL DEFAULT '[]',
  valid_from TEXT,
  valid_to TEXT,
  confidence REAL NOT NULL DEFAULT 0.8,
  status TEXT NOT NULL DEFAULT 'active',
  merged_into TEXT,
  candidate_id TEXT,
  decision_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS pattern_versions (
  pattern_id TEXT NOT NULL, version INTEGER NOT NULL, doc TEXT NOT NULL, at TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (pattern_id, version)
);
-- Auxiliary structure, never a memory layer: supports, related_to, involved_in, associated_with, supersedes, derived_from.
CREATE TABLE IF NOT EXISTS relations (
  relation_id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  src_kind TEXT NOT NULL, src_id TEXT NOT NULL,
  dst_kind TEXT NOT NULL, dst_id TEXT NOT NULL,
  evidence TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'active',
  decision_id TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (type, src_kind, src_id, dst_kind, dst_id)
);
CREATE INDEX IF NOT EXISTS relations_src ON relations(src_kind, src_id);
CREATE INDEX IF NOT EXISTS relations_dst ON relations(dst_kind, dst_id);
-- STAGING: what candidate discovery found and DeepSeek has not decided on yet.
CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY,
  run_id TEXT, window_id TEXT,
  origin TEXT NOT NULL,
  kind_hint TEXT NOT NULL DEFAULT '',
  gist TEXT NOT NULL DEFAULT '',
  entities TEXT NOT NULL DEFAULT '[]',
  topics TEXT NOT NULL DEFAULT '[]',
  raw_ids TEXT NOT NULL,
  context_raw_ids TEXT NOT NULL DEFAULT '[]',
  digest TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  next_attempt_at TEXT,
  decision_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS candidates_status ON candidates(status);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  candidate_id TEXT,
  provider TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  at TEXT NOT NULL,
  status TEXT NOT NULL,
  confidence REAL,
  reason TEXT NOT NULL DEFAULT '',
  actions TEXT NOT NULL DEFAULT '[]',
  answer TEXT NOT NULL DEFAULT '{}',
  errors TEXT NOT NULL DEFAULT '[]',
  applied TEXT NOT NULL DEFAULT '{}',
  usage TEXT NOT NULL DEFAULT '{}',
  raw_hashes TEXT NOT NULL DEFAULT '{}',
  actor TEXT NOT NULL DEFAULT 'deepseek'
);
CREATE TABLE IF NOT EXISTS windows (
  window_id TEXT PRIMARY KEY,
  run_id TEXT, conversation_id TEXT,
  raw_ids TEXT NOT NULL,
  status TEXT NOT NULL,
  mode TEXT NOT NULL DEFAULT '',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  warnings TEXT NOT NULL DEFAULT '[]',
  candidate_count INTEGER NOT NULL DEFAULT 0,
  provider TEXT, model TEXT,
  latency_ms INTEGER,
  first_raw_at TEXT, last_raw_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS windows_status ON windows(status);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  reason TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  summary TEXT NOT NULL DEFAULT '{}',
  errors TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS attachments (
  attachment_id TEXT PRIMARY KEY,
  raw_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  doc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS attachments_raw ON attachments(raw_id);
CREATE TABLE IF NOT EXISTS locks (
  lock_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL UNIQUE,
  conversation_id TEXT,
  query TEXT NOT NULL DEFAULT '',
  locked_pattern_ids TEXT NOT NULL DEFAULT '[]',
  locked_episode_ids TEXT NOT NULL DEFAULT '[]',
  retrieval_trace TEXT NOT NULL DEFAULT '{}',
  searches INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS seen (
  conversation_id TEXT NOT NULL, session_id TEXT NOT NULL,
  ref_kind TEXT NOT NULL, ref_id TEXT NOT NULL, version INTEGER NOT NULL,
  at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS seen_session ON seen(conversation_id, session_id, at);
CREATE TABLE IF NOT EXISTS retrievals (
  retrieval_id TEXT PRIMARY KEY,
  at TEXT NOT NULL,
  turn_id TEXT, conversation_id TEXT,
  policy TEXT NOT NULL, query TEXT NOT NULL,
  status TEXT NOT NULL,
  pattern_ids TEXT NOT NULL DEFAULT '[]', episode_ids TEXT NOT NULL DEFAULT '[]',
  latency_ms REAL,
  trace TEXT NOT NULL DEFAULT '{}',
  confirmed_at TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  audit_id TEXT NOT NULL UNIQUE,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  ref TEXT NOT NULL,
  decision_id TEXT,
  before TEXT, after TEXT
);
CREATE INDEX IF NOT EXISTS audit_ref ON audit(ref);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- the user deleted an Episode: which RAW it came from, nothing else (no text survives a delete). A new Episode drawn only
-- from tombstoned RAW is not created again (actions.apply_plan); one with new evidence is.
CREATE TABLE IF NOT EXISTS tombstones (tombstone_id TEXT PRIMARY KEY, kind TEXT NOT NULL, raw_ids TEXT NOT NULL, deleted_at TEXT NOT NULL);
"""

EPISODE_JSON = ("entities", "topics", "relations", "source_raw_ids", "attachment_ids")
# Promises and plans (commitments.py), added to databases that predate them. kind: '' (an experience), 'commitment' (a
# daily promise / plan: open until done, then it sinks) or 'vow' (a sincere, lasting promise or an important day: never sinks).
EPISODE_COMMITMENT_COLUMNS = (("kind", "TEXT NOT NULL DEFAULT ''"), ("tag", "TEXT NOT NULL DEFAULT ''"), ("owner", "TEXT NOT NULL DEFAULT ''"),
                              ("due_at", "TEXT"), ("commit_status", "TEXT NOT NULL DEFAULT ''"), ("resolved_at", "TEXT"), ("missed_told_at", "TEXT"),
                              # kind 'dream': where the dream itself lives in the bridge ("dream:<id>"); the Episode is only its summary.
                              ("source_ref", "TEXT NOT NULL DEFAULT ''"))
EPISODE_COLUMNS = ("episode_id", "content", "time_start", "time_end", "entities", "topics", "relations", "state", "importance", "confidence",
                   "source_raw_ids", "attachment_ids", "status", "superseded_by", "origin", "candidate_id", "decision_id", "created_at", "updated_at",
                   "version", *(name for name, _ in EPISODE_COMMITMENT_COLUMNS))
EPISODE_DEFAULTS = {"kind": "", "tag": "", "owner": "", "due_at": None, "commit_status": "", "resolved_at": None, "missed_told_at": None, "source_ref": ""}
PATTERN_JSON = ("states", "supporting_episode_ids", "entities")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def mint(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_time(value) -> datetime:
    """A valid ISO-8601 time (date, datetime, with or without Z); anything else is Invalid."""
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise Invalid(f"not a valid time: {text[:40]!r}") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def normalize_time(value) -> str:
    return parse_time(value).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        have = {r[1] for r in self.db.execute("PRAGMA table_info(episodes)")}
        for name, decl in EPISODE_COMMITMENT_COLUMNS:
            if name not in have:
                self.db.execute(f"ALTER TABLE episodes ADD COLUMN {name} {decl}")
        self.lock = threading.RLock()
        self._depth = 0
        self._dirty: dict[str, set] = {"episodes": set(), "patterns": set()}
        self._listeners: list = []

    # ------------------------------------------------------------ transactions

    def on_commit(self, listener) -> None:
        """listener({'episodes': {ids}, 'patterns': {ids}}) runs after every outermost commit that touched them."""
        self._listeners.append(listener)

    @contextmanager
    def tx(self):
        self.lock.acquire()
        outer = self._depth == 0
        released = False
        try:
            if outer:
                self.db.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self.db
            except BaseException:
                self._depth -= 1
                if outer:
                    self.db.execute("ROLLBACK")
                    self._dirty = {"episodes": set(), "patterns": set()}
                raise
            self._depth -= 1
            if outer:
                self.db.execute("COMMIT")
                dirty, self._dirty = self._dirty, {"episodes": set(), "patterns": set()}
                self.lock.release()
                released = True
                if dirty["episodes"] or dirty["patterns"]:
                    for listener in self._listeners:
                        try:
                            listener(dirty)
                        except Exception as error:  # the projection is derived: a failure there never undoes a committed write
                            print(f"[penumbra] memory projection failed: {error!r}")
        finally:
            if not released:
                self.lock.release()

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # ------------------------------------------------------------ small helpers

    def one(self, sql: str, args: tuple = ()):
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def all(self, sql: str, args: tuple = ()) -> list:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def kv_get(self, key: str, default=None):
        row = self.one("SELECT value FROM kv WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default

    def kv_set(self, key: str, value) -> None:
        with self.tx() as db:
            db.execute("INSERT INTO kv VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, dumps(value)))

    def audit(self, action: str, ref: str, before=None, after=None, actor: str = "system", decision_id: str | None = None) -> str:
        audit_id = mint("audit")
        with self.tx() as db:
            db.execute(
                "INSERT INTO audit (audit_id, at, actor, action, ref, decision_id, before, after) VALUES (?,?,?,?,?,?,?,?)",
                (audit_id, now_iso(), actor, action, ref, decision_id, dumps(before) if before is not None else None,
                 dumps(after) if after is not None else None),
            )
        return audit_id

    def audit_rows(self, ref: str | None = None, limit: int = 300) -> list[dict]:
        rows = self.all("SELECT * FROM audit WHERE ref = ? ORDER BY seq DESC LIMIT ?", (ref, limit)) if ref else self.all(
            "SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,))
        return [{**{k: r[k] for k in ("audit_id", "at", "actor", "action", "ref", "decision_id")},
                 "before": json.loads(r["before"]) if r["before"] else None, "after": json.loads(r["after"]) if r["after"] else None} for r in rows]

    # ------------------------------------------------------------ episodes

    @staticmethod
    def _episode(row) -> dict:
        doc = {k: row[k] for k in row.keys()}
        for key in EPISODE_JSON:
            doc[key] = json.loads(doc[key])
        return doc

    def episode(self, episode_id: str) -> dict:
        row = self.one("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
        if row is None:
            raise NotFound(f"no episode {episode_id}")
        return self._episode(row)

    def episodes(self, status: str | None = None, ids: list[str] | None = None) -> list[dict]:
        sql, args = "SELECT * FROM episodes", []
        clauses = []
        if status:
            clauses.append("status = ?")
            args.append(status)
        if ids is not None:
            if not ids:
                return []
            clauses.append(f"episode_id IN ({','.join('?' * len(ids))})")
            args += list(ids)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [self._episode(r) for r in self.all(sql + " ORDER BY time_start, created_at", tuple(args))]

    def insert_episode(self, doc: dict, actor: str = "system", reason: str = "created") -> dict:
        now = now_iso()
        row = {
            "episode_id": doc.get("episode_id") or mint("episode"), "content": doc["content"], "time_start": doc["time_start"],
            "time_end": doc.get("time_end") or doc["time_start"], "entities": doc.get("entities", []), "topics": doc.get("topics", []),
            "relations": doc.get("relations", []), "state": doc.get("state", ""), "importance": float(doc.get("importance", 0.5)),
            "confidence": float(doc.get("confidence", 0.8)), "source_raw_ids": doc["source_raw_ids"], "attachment_ids": doc.get("attachment_ids", []),
            "status": doc.get("status", "active"), "superseded_by": doc.get("superseded_by"), "origin": doc.get("origin", "pipeline"),
            "candidate_id": doc.get("candidate_id"), "decision_id": doc.get("decision_id"),
            "created_at": doc.get("created_at") or now, "updated_at": now, "version": 1,
            **{k: doc.get(k, v) for k, v in EPISODE_DEFAULTS.items()},
        }
        with self.tx() as db:
            db.execute(
                f"INSERT INTO episodes ({','.join(EPISODE_COLUMNS)}) VALUES ({','.join('?' * len(EPISODE_COLUMNS))})",
                tuple(dumps(row[k]) if k in EPISODE_JSON else row[k] for k in EPISODE_COLUMNS),
            )
            db.execute("INSERT INTO episode_versions VALUES (?,?,?,?,?,?)", (row["episode_id"], 1, dumps(row), now, actor, reason))
            self._dirty["episodes"].add(row["episode_id"])
        return row

    def update_episode(self, episode_id: str, patch: dict, actor: str = "system", reason: str = "", decision_id: str | None = None) -> dict:
        """A new version: the previous one stays readable in episode_versions; RAW provenance only ever grows here."""
        with self.tx() as db:
            current = self.episode(episode_id)
            new = copy.deepcopy(current)
            for key, value in patch.items():
                if key in ("episode_id", "created_at", "version"):
                    continue
                new[key] = value
            new["updated_at"] = now_iso()
            new["version"] = current["version"] + 1
            if decision_id:
                new["decision_id"] = decision_id
            new["importance"], new["confidence"] = float(new["importance"]), float(new["confidence"])
            columns = [k for k in EPISODE_COLUMNS if k not in ("episode_id", "created_at")]
            db.execute(f"UPDATE episodes SET {', '.join(k + '=?' for k in columns)} WHERE episode_id=?",
                       (*(dumps(new[k]) if k in EPISODE_JSON else new[k] for k in columns), episode_id))
            db.execute("INSERT INTO episode_versions VALUES (?,?,?,?,?,?)", (episode_id, new["version"], dumps(new), new["updated_at"], actor, reason))
            self._dirty["episodes"].add(episode_id)
        return new

    def episode_versions(self, episode_id: str) -> list[dict]:
        return [{"version": r["version"], "at": r["at"], "actor": r["actor"], "reason": r["reason"], "doc": json.loads(r["doc"])}
                for r in self.all("SELECT * FROM episode_versions WHERE episode_id = ? ORDER BY version", (episode_id,))]

    # ------------------------------------------------------------ patterns

    @staticmethod
    def _pattern(row) -> dict:
        doc = {k: row[k] for k in row.keys()}
        for key in PATTERN_JSON:
            doc[key] = json.loads(doc[key])
        # The spec's view of the same data: everything closed is history, the open one is current.
        doc["historical_states"] = [s for s in doc["states"] if s.get("valid_to")]
        return doc

    def pattern(self, pattern_id: str) -> dict:
        row = self.one("SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,))
        if row is None:
            raise NotFound(f"no pattern {pattern_id}")
        return self._pattern(row)

    def patterns(self, status: str | None = None, ids: list[str] | None = None) -> list[dict]:
        sql, args, clauses = "SELECT * FROM patterns", [], []
        if status:
            clauses.append("status = ?")
            args.append(status)
        if ids is not None:
            if not ids:
                return []
            clauses.append(f"pattern_id IN ({','.join('?' * len(ids))})")
            args += list(ids)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [self._pattern(r) for r in self.all(sql + " ORDER BY created_at", tuple(args))]

    def insert_pattern(self, doc: dict, actor: str = "system", reason: str = "created") -> dict:
        now = now_iso()
        row = {
            "pattern_id": doc.get("pattern_id") or mint("pattern"), "title": doc["title"], "topic": doc.get("topic", ""),
            "narrative": doc["narrative"], "current_state": doc["current_state"], "current_state_since": doc.get("current_state_since"),
            "states": doc["states"], "supporting_episode_ids": doc["supporting_episode_ids"], "entities": doc.get("entities", []),
            "valid_from": doc.get("valid_from"), "valid_to": doc.get("valid_to"), "confidence": float(doc.get("confidence", 0.8)),
            "status": doc.get("status", "active"), "merged_into": doc.get("merged_into"), "candidate_id": doc.get("candidate_id"),
            "decision_id": doc.get("decision_id"), "created_at": doc.get("created_at") or now, "updated_at": now, "version": 1,
        }
        with self.tx() as db:
            db.execute(
                "INSERT INTO patterns VALUES (" + ",".join("?" * 19) + ")",
                (row["pattern_id"], row["title"], row["topic"], row["narrative"], row["current_state"], row["current_state_since"],
                 dumps(row["states"]), dumps(row["supporting_episode_ids"]), dumps(row["entities"]), row["valid_from"], row["valid_to"],
                 row["confidence"], row["status"], row["merged_into"], row["candidate_id"], row["decision_id"], row["created_at"], row["updated_at"], 1),
            )
            db.execute("INSERT INTO pattern_versions VALUES (?,?,?,?,?,?)", (row["pattern_id"], 1, dumps(row), now, actor, reason))
            self._dirty["patterns"].add(row["pattern_id"])
        return row

    def update_pattern(self, pattern_id: str, patch: dict, actor: str = "system", reason: str = "", decision_id: str | None = None) -> dict:
        with self.tx() as db:
            current = self.pattern(pattern_id)
            current.pop("historical_states", None)
            new = copy.deepcopy(current)
            for key, value in patch.items():
                if key in ("pattern_id", "created_at", "version", "historical_states"):
                    continue
                new[key] = value
            new["updated_at"] = now_iso()
            new["version"] = current["version"] + 1
            if decision_id:
                new["decision_id"] = decision_id
            db.execute(
                "UPDATE patterns SET title=?, topic=?, narrative=?, current_state=?, current_state_since=?, states=?, supporting_episode_ids=?, entities=?, "
                "valid_from=?, valid_to=?, confidence=?, status=?, merged_into=?, candidate_id=?, decision_id=?, updated_at=?, version=? WHERE pattern_id=?",
                (new["title"], new["topic"], new["narrative"], new["current_state"], new["current_state_since"], dumps(new["states"]),
                 dumps(new["supporting_episode_ids"]), dumps(new["entities"]), new["valid_from"], new["valid_to"], float(new["confidence"]),
                 new["status"], new["merged_into"], new["candidate_id"], new["decision_id"], new["updated_at"], new["version"], pattern_id),
            )
            db.execute("INSERT INTO pattern_versions VALUES (?,?,?,?,?,?)", (pattern_id, new["version"], dumps(new), new["updated_at"], actor, reason))
            self._dirty["patterns"].add(pattern_id)
        return self._pattern(self.one("SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,)))

    def pattern_versions(self, pattern_id: str) -> list[dict]:
        return [{"version": r["version"], "at": r["at"], "actor": r["actor"], "reason": r["reason"], "doc": json.loads(r["doc"])}
                for r in self.all("SELECT * FROM pattern_versions WHERE pattern_id = ? ORDER BY version", (pattern_id,))]

    def patterns_supported_by(self, episode_id: str) -> list[dict]:
        return [p for p in self.patterns(status="active") if episode_id in p["supporting_episode_ids"]]

    # ------------------------------------------------------------ deletion (the user's "删除": the memory is gone, its RAW stays)

    def _forget(self, db, kind: str, ident: str) -> None:
        """Relations touching it go, and the audit keeps only that it happened: no copy of the text survives anywhere."""
        db.execute("DELETE FROM relations WHERE (src_kind = ? AND (src_id = ? OR src_id LIKE ?)) OR (dst_kind = ? AND (dst_id = ? OR dst_id LIKE ?))",
                   (kind, ident, ident + "#%", kind, ident, ident + "#%"))
        db.execute("DELETE FROM relations WHERE src_kind = 'state' AND (src_id LIKE ? OR dst_id LIKE ?)", (ident + "#%", ident + "#%"))
        db.execute("UPDATE audit SET before = NULL, after = NULL WHERE ref = ?", (ident,))
        self._dirty[kind + "s"].add(ident)  # the projection sees it is gone and drops it from the index

    def delete_episode(self, episode_id: str) -> None:
        with self.tx() as db:
            ep = self.episode(episode_id)  # NotFound if there is nothing to delete
            sources = sorted([*ep["source_raw_ids"], *([ep["source_ref"]] if ep.get("source_ref") else [])])
            if sources:
                db.execute("INSERT INTO tombstones (tombstone_id, kind, raw_ids, deleted_at) VALUES (?,?,?,?)",
                           (f"tomb_{uuid.uuid4().hex[:16]}", "episode", json.dumps(sources), datetime.now(timezone.utc).isoformat()))
            db.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
            db.execute("DELETE FROM episode_versions WHERE episode_id = ?", (episode_id,))
            self._forget(db, "episode", episode_id)

    def tombstoned_raw_ids(self) -> set[str]:
        """Every RAW id a deleted Episode came from."""
        out: set[str] = set()
        for row in self.all("SELECT raw_ids FROM tombstones"):
            out.update(json.loads(row["raw_ids"]))
        return out

    def delete_pattern(self, pattern_id: str) -> None:
        with self.tx() as db:
            self.pattern(pattern_id)
            db.execute("DELETE FROM patterns WHERE pattern_id = ?", (pattern_id,))
            db.execute("DELETE FROM pattern_versions WHERE pattern_id = ?", (pattern_id,))
            self._forget(db, "pattern", pattern_id)

    # ------------------------------------------------------------ relations

    def add_relation(self, rtype: str, src_kind: str, src_id: str, dst_kind: str, dst_id: str, evidence=(), decision_id: str | None = None) -> str | None:
        with self.tx() as db:
            existing = db.execute(
                "SELECT relation_id, status FROM relations WHERE type=? AND src_kind=? AND src_id=? AND dst_kind=? AND dst_id=?",
                (rtype, src_kind, src_id, dst_kind, dst_id)).fetchone()
            if existing:
                if existing["status"] != "active":
                    db.execute("UPDATE relations SET status='active' WHERE relation_id=?", (existing["relation_id"],))
                return existing["relation_id"]
            rid = mint("relation")
            db.execute("INSERT INTO relations VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (rid, rtype, src_kind, src_id, dst_kind, dst_id, dumps(list(evidence)), "active", decision_id, now_iso()))
            return rid

    def archive_relation(self, relation_id: str) -> None:
        with self.tx() as db:
            db.execute("UPDATE relations SET status='archived' WHERE relation_id=?", (relation_id,))

    def relations(self, kind: str | None = None, ident: str | None = None, rtype: str | None = None, active_only: bool = True) -> list[dict]:
        sql, args, clauses = "SELECT * FROM relations", [], []
        if kind and ident:
            clauses.append("((src_kind=? AND src_id=?) OR (dst_kind=? AND dst_id=?))")
            args += [kind, ident, kind, ident]
        if rtype:
            clauses.append("type=?")
            args.append(rtype)
        if active_only:
            clauses.append("status='active'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [{**{k: r[k] for k in r.keys()}, "evidence": json.loads(r["evidence"])} for r in self.all(sql + " ORDER BY created_at", tuple(args))]

    # ------------------------------------------------------------ attachments

    def upsert_attachment(self, doc: dict) -> None:
        with self.tx() as db:
            db.execute("INSERT INTO attachments VALUES (?,?,?,?) ON CONFLICT(attachment_id) DO UPDATE SET status=excluded.status, doc=excluded.doc",
                       (doc["attachment_id"], doc["raw_id"], doc.get("status", "active"), dumps(doc)))

    def attachment(self, attachment_id: str) -> dict:
        row = self.one("SELECT doc FROM attachments WHERE attachment_id = ?", (attachment_id,))
        if row is None:
            raise NotFound(f"no attachment {attachment_id}")
        return json.loads(row["doc"])

    def attachments_for_raw(self, raw_id: str, include_archived: bool = False) -> list[dict]:
        rows = self.all("SELECT doc, status FROM attachments WHERE raw_id = ?", (raw_id,))
        return [json.loads(r["doc"]) for r in rows if include_archived or r["status"] == "active"]

    def attachments(self, include_archived: bool = False) -> list[dict]:
        rows = self.all("SELECT doc, status FROM attachments")
        return [json.loads(r["doc"]) for r in rows if include_archived or r["status"] == "active"]

    # ------------------------------------------------------------ counts

    def counts(self) -> dict:
        one = lambda sql: self.one(sql)[0]  # noqa: E731
        return {
            "episodes": one("SELECT COUNT(*) FROM episodes WHERE status='active'"),
            "episodesAll": one("SELECT COUNT(*) FROM episodes"),
            "patterns": one("SELECT COUNT(*) FROM patterns WHERE status='active'"),
            "patternsAll": one("SELECT COUNT(*) FROM patterns"),
            "relations": one("SELECT COUNT(*) FROM relations WHERE status='active'"),
            "attachments": one("SELECT COUNT(*) FROM attachments WHERE status='active'"),
            "staging": one("SELECT COUNT(*) FROM candidates WHERE status='STAGING'"),
            "quarantine": one("SELECT COUNT(*) FROM candidates WHERE status='QUARANTINE'"),
            "rejected": one("SELECT COUNT(*) FROM candidates WHERE status='REJECTED'"),
            "failed": one("SELECT COUNT(*) FROM candidates WHERE status='FAILED'"),
            "verified": one("SELECT COUNT(*) FROM candidates WHERE status IN ('APPLIED','NO_ACTION')"),
            "candidates": one("SELECT COUNT(*) FROM candidates"),
            "windows": one("SELECT COUNT(*) FROM windows"),
        }
