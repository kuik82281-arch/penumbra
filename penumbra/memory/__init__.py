"""The unified long-term memory: RAW -> Episode -> Pattern.

    WRITE  RAW -> Candidate Discovery (Ollama, deterministic fallback) -> STAGING -> DeepSeek Verification -> Episode -> Pattern
    READ   query -> BM25 + Embedding + Entity + Time -> RRF -> Dedup -> Seen -> Reranker -> Threshold -> Pattern first
           -> Search once -> Lock -> Expand Episode -> Expand RAW

STAGING, QUARANTINE, AUDIT, CHECKPOINT, VERSIONING and SOURCE REFS are engineering aids of this one path, not layers.
"""
from __future__ import annotations

import copy
import hashlib
import json

from datetime import timedelta

from .. import identity
from ..errors import Invalid, NotFound
from . import commitments
from . import edit as edit_module
from .actions import ALL_ACTIONS
from .discovery import Discovery
from .pipeline import LAST_ACTIVITY_KEY, NOTE_SOURCE, Pipeline
from .read import MemoryRead
from .store import Store, mint, normalize_time, now_iso
from . import dreams as dreams_module
from . import mistakes as mistakes_module
from . import names as names_module
from . import threads as threads_module
from .verification import Verifier

__all__ = ["MemoryCore", "ALL_ACTIONS"]


class MemoryCore:
    def __init__(self, service, discovery: Discovery | None = None, verifier: Verifier | None = None):
        self.service = service
        self.store = Store(service.config.memory_dir / "core.sqlite")
        threads_module.ensure_schema(self.store)
        names_module.ensure_schema(self.store)
        mistakes_module.ensure_schema(self.store)
        self.discovery = discovery or Discovery()
        self.verifier = verifier or Verifier()
        self.read = MemoryRead(self)
        self.pipeline = Pipeline(self)
        self.store.on_commit(self.read.on_change)
        self._started = False

    # ------------------------------------------------------------ lifecycle

    def start(self, scheduler: bool = True) -> dict:
        """After the RAW index is rebuilt: recovery, one-time migration of the old paths, attachment backfill, scheduler."""
        from . import migrate

        report = {"recovery": self.pipeline.recover(), "migration": migrate.run(self)}
        with self.service.lock:
            rows = self.service.conn.execute("SELECT id FROM originals WHERE meta LIKE '%\"attachments\": [{%' OR meta LIKE '%\"attachments\":[{%'").fetchall()
        for row in rows:
            original = self.service.get_original(row["id"])
            if original.attachments:
                self.register_attachments(original)
        if scheduler and self.pipeline.settings()["enabled"]:
            self.pipeline.start_scheduler()
        self._started = True
        return report

    def close(self) -> None:
        self.pipeline.close()
        self.store.close()

    # ------------------------------------------------------------ RAW access (the source of truth, read-only)

    def raw_record(self, raw_id: str) -> dict:
        record = self.service.get_original(str(raw_id)).to_record()
        record["attachments"] = self.store.attachments_for_raw(record["id"])
        return record

    def raw(self, ids: list[str]) -> list[dict]:
        if not isinstance(ids, list) or not ids or len(ids) > 100:
            raise Invalid("1..100 RAW ids required")
        return [self.raw_record(str(i)) for i in dict.fromkeys(ids)]

    def raw_list(self, limit: int = 300, conversation_id: str | None = None, before: str | None = None) -> list[dict]:
        """The newest `limit` RAW (optionally of one conversation, optionally only those created before `before`), oldest first."""
        where, args = [], []
        if conversation_id:
            where.append("conversation_id = ?")
            args.append(conversation_id)
        if before:
            where.append("created_at < ?")
            args.append(before)
        sql = "SELECT id FROM originals" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self.service.lock:
            rows = self.service.conn.execute(sql, (*args, limit)).fetchall()
        return [self.raw_record(r["id"]) for r in reversed(rows)]

    def register_attachments(self, original) -> list[str]:
        """Attachment rows are derived from immutable RAW: one per (RAW, asset), idempotent, provenance kept."""
        created = []
        for source in original.attachments:
            asset = str(source.get("original_asset_ref") or "").strip()
            mime = str(source.get("mime_type") or "").strip()
            if not asset or not mime:
                continue
            aid = "attachment_" + hashlib.sha256((original.id + "\x1f" + asset).encode()).hexdigest()[:20]
            try:
                existing = self.store.attachment(aid)
            except NotFound:
                existing = None
            if existing:
                if existing.get("raw_id") != original.id or existing.get("original_asset_ref") != asset:
                    raise Invalid("Attachment identity conflict")
                # What the picture is, in words, often comes later (described after the turn): a resend refreshes it.
                caption, text = str(source.get("caption") or ""), str(source.get("searchable_text") or "")
                if (caption, text) != (existing.get("caption") or "", existing.get("searchable_text") or "") and (caption or text):
                    self.store.upsert_attachment({**existing, "caption": caption or existing.get("caption") or "", "searchable_text": text or existing.get("searchable_text") or ""})
                    self.store.audit("attachment.described", aid, actor="bridge")
                continue
            self.store.upsert_attachment({
                "attachment_id": aid, "raw_id": original.id, "conversation_id": original.conversation_id, "turn_id": original.turn_id or source.get("turn_id"),
                "type": source.get("type") or ("image" if mime.startswith("image/") else "file"), "mime_type": mime, "original_asset_ref": asset,
                "thumbnail_asset_ref": source.get("thumbnail_asset_ref"), "checksum": source.get("checksum"), "created_at": source.get("created_at") or original.created_at,
                "metadata": copy.deepcopy(source.get("metadata") or {}), "caption": str(source.get("caption") or ""), "searchable_text": str(source.get("searchable_text") or ""),
                "embedding_ref": source.get("embedding_ref"), "status": "active"})
            self.store.audit("attachment.registered", aid, actor="bridge")
            created.append(aid)
        return created

    # ------------------------------------------------------------ the user's own notes (RAW + Episode, verbatim)

    def add_note(self, payload: dict) -> dict:
        """A memory the user writes herself: her words are stored as RAW (source manual_note) and the Episode cites it."""
        if not identity.is_user(payload.get("actor") or identity.USER_ACTOR):
            raise Invalid("notes are written by the user")
        content = str(payload.get("content") or "").strip()
        if not content or len(content) > 4000:
            raise Invalid("content is required (max 4000 characters)")
        stamp = normalize_time(payload.get("time")) if payload.get("time") else now_iso()
        item_id = mint("note")
        result = self.service.ingest_originals(f"{identity.USER_ACTOR}-manual", "manual-notes", [{"id": item_id, "role": "user", "content": content, "createdAt": stamp, "sourceType": NOTE_SOURCE}])
        raw_id = result["added"][0]
        episode = self.store.insert_episode({
            "content": content, "time_start": stamp, "time_end": stamp, "entities": [str(e)[:40] for e in payload.get("entities") or []][:12],
            "topics": [str(t)[:40] for t in payload.get("topics") or []][:8], "importance": float(payload.get("importance", 0.7)), "confidence": 1.0,
            "source_raw_ids": [raw_id], "origin": "manual"}, actor=identity.USER_ACTOR, reason="manual note")
        self.store.audit("note.added", episode["episode_id"], after={"raw": raw_id}, actor=identity.USER_ACTOR)
        pattern_id = payload.get("pattern_id")
        if pattern_id:
            edit_module.attach_episode(self.store, str(pattern_id), episode["episode_id"], identity.USER_ACTOR)
        return {"episode": episode, "raw_id": raw_id}

    # ------------------------------------------------------------ status / snapshot

    def activity(self) -> None:
        self.store.kv_set(LAST_ACTIVITY_KEY, now_iso())

    def _checkpoint(self) -> dict:
        checkpoint = self.store.kv_get("checkpoint", {})
        work = self.pipeline.pending_work()
        processed = len(self.pipeline._processed_raw())
        return {**checkpoint, "last_activity_at": self.store.kv_get(LAST_ACTIVITY_KEY),"processed_raw_count": processed, "unprocessed_raw_count": work["unprocessedRaw"],
                "staging_due": work["stagingDue"], "scheduler_error": self.store.kv_get("scheduler_error")}

    def providers(self, probe: bool = True) -> dict:
        ollama = self.discovery.ollama.status(probe=probe) | (self.store.kv_get("ollama_stats", {}) or {})
        deepseek = self.verifier.status() | (self.store.kv_get("deepseek_stats", {}) or {})
        return {"ollama": ollama, "deepseek": deepseek, "reranker": self.read.reranker.status(), "embedding": self.service.vectors.provider.status()}

    def health(self) -> dict:
        return self.providers(probe=True)

    def snapshot(self) -> dict:
        s = self.store
        counts = s.counts()
        with self.service.lock:
            counts["raw"] = self.service.conn.execute("SELECT COUNT(*) FROM originals").fetchone()[0]
        candidates = [self._candidate_view(r) for r in s.all(
            "SELECT c.*, d.status AS decision_status, d.reason AS decision_reason, d.model AS decision_model, d.confidence AS decision_confidence, "
            "d.at AS decision_at FROM candidates c LEFT JOIN decisions d ON d.decision_id = c.decision_id ORDER BY c.created_at DESC LIMIT 400")]
        return {
            "counts": counts,
            "patterns": [{k: v for k, v in p.items() if k != "states"} for p in s.patterns()],
            "episodes": s.episodes(),
            "relations": s.relations(),
            "attachments": s.attachments(include_archived=True),
            "candidates": candidates,
            "windows": [self._window_view(r) for r in s.all("SELECT * FROM windows ORDER BY created_at DESC LIMIT 60")],
            "runs": [self._run_view(r) for r in s.all("SELECT * FROM runs ORDER BY started_at DESC LIMIT 40")],
            "audit": s.audit_rows(limit=120),
            "settings": self.pipeline.settings(),
            "checkpoint": self._checkpoint(),
            "providers": self.providers(probe=False),
            "retrievals": [self._retrieval_view(r) for r in s.all("SELECT * FROM retrievals ORDER BY at DESC LIMIT 30")],
        }

    @staticmethod
    def _candidate_view(r) -> dict:
        out = {k: r[k] for k in r.keys()}
        for key in ("entities", "topics", "raw_ids", "context_raw_ids"):
            out[key] = json.loads(out[key])
        return out

    @staticmethod
    def _window_view(r) -> dict:
        out = {k: r[k] for k in r.keys()}
        out["raw_ids"] = json.loads(out["raw_ids"])
        out["warnings"] = json.loads(out["warnings"])
        return out

    @staticmethod
    def _run_view(r) -> dict:
        out = {k: r[k] for k in r.keys()}
        out["summary"] = json.loads(out["summary"])
        out["errors"] = json.loads(out["errors"])
        return out

    @staticmethod
    def _retrieval_view(r) -> dict:
        out = {k: r[k] for k in r.keys()}
        for key in ("pattern_ids", "episode_ids", "trace"):
            out[key] = json.loads(out[key])
        return out

    def decision(self, decision_id: str) -> dict:
        r = self.store.one("SELECT * FROM decisions WHERE decision_id = ?", (decision_id,))
        if r is None:
            raise NotFound(f"no decision {decision_id}")
        out = {k: r[k] for k in r.keys()}
        for key in ("actions", "answer", "errors", "applied", "usage", "raw_hashes"):
            out[key] = json.loads(out[key])
        return out

    def provenance(self, kind: str, ident: str) -> dict:
        """RAW -> Candidate -> DeepSeek decision -> Episode -> Pattern, for one Episode or Pattern."""
        if kind == "pattern":
            pattern = self.store.pattern(ident)
            episodes = self.store.episodes(ids=pattern["supporting_episode_ids"])
            chains = [self.provenance("episode", e["episode_id"]) for e in episodes]
            decisions = {}
            for v in self.store.pattern_versions(ident):
                did = v["doc"].get("decision_id")
                if did and did not in decisions:
                    try:
                        decisions[did] = self.decision(did)
                    except NotFound:
                        pass
            return {"kind": "pattern", "pattern": pattern, "versions": self.store.pattern_versions(ident), "decisions": list(decisions.values()), "episodes": chains,
                    "relations": self.store.relations("pattern", ident), "audit": self.store.audit_rows(ident, 60)}
        episode = self.store.episode(ident)
        cand = self.store.one("SELECT * FROM candidates WHERE candidate_id = ?", (episode["candidate_id"],)) if episode["candidate_id"] else None
        window = self.store.one("SELECT * FROM windows WHERE window_id = ?", (cand["window_id"],)) if cand else None
        decision = None
        if episode["decision_id"]:
            try:
                decision = self.decision(episode["decision_id"])
            except NotFound:
                decision = None
        raw = []
        for rid in episode["source_raw_ids"][:40]:
            try:
                raw.append(self.raw_record(rid))
            except NotFound:
                raw.append({"id": rid, "missing": True})
        return {"kind": "episode", "episode": episode, "versions": self.store.episode_versions(ident), "raw": raw, "candidate": self._candidate_view(cand) if cand else None,
                "window": self._window_view(window) if window else None, "decision": decision, "patterns": [{"pattern_id": p["pattern_id"], "title": p["title"]} for p in self.store.patterns_supported_by(ident)],
                "relations": self.store.relations("episode", ident), "audit": self.store.audit_rows(ident, 60)}

    # ------------------------------------------------------------ HTTP routing (under /memory-core)

    def route(self, method: str, parts: list[str], body: dict | None = None, query: dict | None = None):
        body, query = body or {}, query or {}
        if method == "GET":
            if not parts:
                return self.snapshot()
            if parts == ["threads"]:
                return {"threads": threads_module.overview(self.store)}
            if len(parts) == 3 and parts[:2] == ["threads", "move-candidates"]:
                return {"threads": threads_module.move_candidates(self.store, self.read, parts[2])}
            if parts == ["health"]:
                return self.health()
            if parts == ["names"]:
                return {"names": names_module.all_names(self.store), "kinds": list(names_module.KINDS)}
            if parts == ["mistakes"]:
                return {"mistakes": mistakes_module.all_mistakes(self.store)}
            if parts == ["recent"]:
                return {"episodes": self.recent_episodes(int(query.get("days") or 7), int(query.get("limit") or 6))}
            if parts == ["commitments"]:
                return commitments.for_assistant(self)
            if parts == ["raw"]:
                return {"raw": self.raw_list(int(query.get("limit") or 300), query.get("conversationId"), query.get("before"))}
            if parts == ["attachments"]:
                needle = str(query.get("q") or "").lower()
                rows = [a for a in self.store.attachments() if not needle or needle in json.dumps([a.get("caption"), a.get("searchable_text"), a.get("metadata")], ensure_ascii=False).lower()]
                return {"attachments": [self.attachment(a["attachment_id"]) for a in rows[:100]]}
            if len(parts) == 2 and parts[0] == "attachments":
                return {"attachment": self.attachment(parts[1])}
            if len(parts) == 3 and parts[0] == "provenance" and parts[1] in ("pattern", "episode"):
                return self.provenance(parts[1], parts[2])
            if len(parts) == 2 and parts[0] == "patterns":
                return self.read.expand_pattern(parts[1])
            if len(parts) == 2 and parts[0] == "episodes":
                return self.read.expand_episode(parts[1])
            if len(parts) == 2 and parts[0] == "decisions":
                return {"decision": self.decision(parts[1])}
            if len(parts) == 2 and parts[0] == "locks":
                lock = self.read.lock_for(parts[1])
                if not lock:
                    raise NotFound("no lock for this turn")
                return {"lock": lock}
        if method == "POST":
            if parts == ["commitments", "told"]:
                return {"sunk": commitments.mark_told(self, body.get("episodeIds") or [])}
            if parts == ["run"]:
                return self.pipeline.start_run("manual")
            if parts == ["run-sync"]:
                return self.pipeline.run("manual")
            if parts == ["tick"]:
                return self.pipeline.tick()
            if parts == ["reprocess"]:
                return self.pipeline.reprocess(str(body.get("conversationId") or "") or None, body.get("onlyEmpty") is not False,
                                               normalize_time(body["since"]) if body.get("since") else None, normalize_time(body["until"]) if body.get("until") else None)
            if parts == ["activity"]:
                self.activity()
                return {"ok": True}
            if parts == ["settings"]:
                return {"settings": self.pipeline.update_settings(body)}
            if parts == ["retrieve"]:
                return self.read.retrieve(body)
            if parts == ["recall"]:
                return self.read.recall(body)
            if parts == ["confirm"]:
                return self.read.confirm(str(body.get("injectId") or ""), str(body.get("conversationId") or ""), str(body.get("sessionId") or ""), body.get("refs") or [])
            if parts == ["note"]:
                return self.add_note(body)
            if parts == ["names"]:
                return {"name": names_module.save(self.store, body)}
            if parts == ["names", "delete"]:
                return names_module.delete(self.store, str(body.get("name_id") or ""))
            if parts == ["mistakes"]:
                return {"mistake": mistakes_module.add(self.store, body)}
            if parts == ["mistakes", "delete"]:
                return mistakes_module.delete(self.store, str(body.get("mistake_id") or ""))
            if parts == ["mistakes", "run"]:
                return mistakes_module.run(self)
            if parts == ["dreams"]:
                return dreams_module.remember(self, body)
            if parts == ["dreams", "forget"]:
                return dreams_module.forget(self, str(body.get("dreamId") or ""))
            if parts == ["threads", "decide"]:
                return threads_module.decide(self.store, body)
            if len(parts) == 3 and parts[:2] == ["threads", "story"]:
                return {"thread": threads_module.write_story(self.store, self.verifier.client, parts[2])}
            if parts == ["edit"]:
                return edit_module.edit(self, body)
            if len(parts) == 2 and parts[0] == "review":
                return edit_module.review(self, parts[1], body)
            if len(parts) == 2 and parts[0] == "retry":
                return edit_module.requeue(self, parts[1], body)
        raise NotFound("Unknown memory-core endpoint")

    def recent_episodes(self, days: int = 7, limit: int = 6) -> list[dict]:
        """The most important Episodes of the last `days` days (Session Handoff: recent_important_events)."""
        since = (self.service._now() - timedelta(days=max(1, min(days, 60)))).isoformat()
        # Commitments are not "events": open ones reach the assistant as his to-do list on every turn, finished ones have sunk.
        rows = [e for e in self.store.episodes(status="active") if e["time_end"] >= since and e.get("kind") not in ("commitment", "dream", "ritual")]
        rows.sort(key=lambda e: e["time_end"], reverse=True)
        rows.sort(key=lambda e: -e["importance"])  # stable: newest first among equally important
        return [{"episode_id": e["episode_id"], "content": e["content"], "time_start": e["time_start"], "time_end": e["time_end"], "importance": e["importance"],
                 "entities": e["entities"], "version": e["version"], "patterns": [{"pattern_id": p["pattern_id"], "title": p["title"]} for p in self.store.patterns_supported_by(e["episode_id"])]}
                for e in rows[:max(1, min(limit, 20))]]

    def attachment(self, aid: str) -> dict:
        a = self.store.attachment(aid)
        episodes = [e for e in self.store.episodes() if aid in e.get("attachment_ids", [])]
        patterns = {}
        for e in episodes:
            for p in self.store.patterns_supported_by(e["episode_id"]):
                patterns[p["pattern_id"]] = {"pattern_id": p["pattern_id"], "title": p["title"]}
        return {**a, "raw": self.raw_record(a["raw_id"]), "episodes": [{"episode_id": e["episode_id"], "content": e["content"]} for e in episodes], "patterns": list(patterns.values())}

