"""Preference Learning backend: the user's own preference documents, kept verbatim, published by her.

Source truth
  PreferenceDocument.originalContent is stored verbatim with its sha256. A draft is edited in place; changing the text of
  a published (or legacy review) document makes a new draft version in the same lineage; the old one stays as it was.

Modes (chosen by the user; a document without a mode is a legacy retrieval document)
  session_pinned  the whole published original rides in every new the assistant runtime session (the bridge's Session
                  Preference Pack, GET /preferences/session-pack). No chunks, no retrieval index, no per-turn selection.
  retrieval       long-tail material: cut into chunks locally and deterministically at publish (paragraphs), recalled
                  per turn like other memories. Chunks are always exact slices: originalContent[startOffset:endOffset].

Workflow
  draft (edit freely) -> publish (session_pinned | retrieval) -> active [enabled / disabled] -> unpublish | archive
  Nothing unpublished ever reaches the assistant. Within a lineage only one version is in effect: publishing a version retires
  the others that were published / under review. Disabled, archived and superseded documents are out of the pack and
  out of retrieval.

DeepSeek segmentation is retired: nothing here calls a model. Legacy documents keep their originals, chunks, jobs and
audit; a legacy document still under review can be published (its reviewed chunks go active) or archived.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import timezone
from typing import TYPE_CHECKING

from . import identity
from . import files, index
from .errors import Invalid, NotFound

if TYPE_CHECKING:
    from .service import Penombre

BASE_LABELS = ("relationship", "boundary", "language", "nickname", "dynamic", "scenario", "comfort", "conflict", "repair",
               "habit", "preference")


def canonical_labels() -> tuple:
    """The built-in labels plus the deployment's own (profile.json "preference_labels")."""
    return BASE_LABELS + tuple(x for x in identity.current().extra_labels if x not in BASE_LABELS)
OWNERS = ("user", "assistant", "relationship")


def _owner_role(value: str) -> str:
    """An owner as its role: the profile's aliases (an older client's words for the two of them) are the same owners."""
    return "user" if identity.is_user(value) else "assistant" if identity.is_assistant(value) else value
MODES = ("session_pinned", "retrieval")
DOC_STATUSES = ("draft", "processing", "review", "active", "failed", "archived")  # processing / review / failed: legacy only
CHUNK_STATUSES = ("review", "active", "excluded", "archived")
LOCAL_CHUNKING = "local-paragraph-v1"
ACTORS = {"user"}
UNIT_ENDERS = set("。！？!?；;…\n")
UNIT_TRAILERS = set("。！？!?；;…”’」』）)\"'")
MAX_UNIT_CHARS = 300
LOCAL_CHUNK_CHARS = 600
LOCAL_CHUNK_MIN = 20
# All published, enabled session_pinned documents together (estimated tokens). Over it, publishing / enabling is refused
# and the user is asked to shorten or move something to retrieval - never truncated or summarised here.
PINNED_BUDGET_TOKENS = int(os.environ.get("PENUMBRA_PINNED_BUDGET_TOKENS") or 20_000)
_WIDE = re.compile(r"[⺀-鿿가-힯豈-﫿＀-￯　-〿]")


def estimate_tokens(text: str) -> int:
    """Rough Claude token estimate: a CJK character ~ 1 token, other text ~ 3.5 characters a token."""
    wide = len(_WIDE.findall(text))
    return math.ceil(wide + (len(text) - wide) / 3.5)


def mode_of(doc: dict) -> str:
    return doc.get("mode") or "retrieval"


def enabled(doc: dict) -> bool:
    return doc.get("enabled", True) is not False


def retrievable(doc: dict) -> bool:
    """A document whose active chunks belong in the retrieval index."""
    return doc["status"] == "active" and mode_of(doc) == "retrieval" and enabled(doc)


def pinned(doc: dict) -> bool:
    """A document that belongs in the Session Preference Pack."""
    return doc["status"] == "active" and mode_of(doc) == "session_pinned" and enabled(doc)


def split_units(text: str) -> list[tuple[int, int]]:
    """Gap-free sentence / line units as (start, end) offsets. Whitespace-only pieces join the unit before them."""
    units: list[tuple[int, int]] = []
    start, i, n = 0, 0, len(text)
    while i < n:
        if text[i] in UNIT_ENDERS or i - start >= MAX_UNIT_CHARS:
            j = i + 1
            while j < n and text[j] in UNIT_TRAILERS:
                j += 1
            while j < n and text[j] in " \t\r\n　":
                j += 1
            units.append((start, j))
            start = i = j
            continue
        i += 1
    if start < n:
        units.append((start, n))
    merged: list[tuple[int, int]] = []
    for s, e in units:
        if merged and not text[s:e].strip():
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    if len(merged) > 1 and not text[merged[0][0]:merged[0][1]].strip():
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged


def local_chunks(text: str, limit: int = LOCAL_CHUNK_CHARS) -> list[tuple[int, int]]:
    """Deterministic retrieval chunks: paragraphs (a blank line ends one), long paragraphs cut at sentence units near
    `limit`, very short paragraphs (headings) joined to what follows. Gap-free: the ranges tile the text exactly."""
    ranges: list[tuple[int, int]] = []
    start = None
    for s, e in split_units(text):
        if start is None:
            start = s
        elif e - start > limit and s - start >= LOCAL_CHUNK_MIN:
            ranges.append((start, s))
            start = s
        piece = text[start:e]
        if re.search(r"\n[ \t　]*\n\s*$", piece) and len(piece.strip()) >= LOCAL_CHUNK_MIN:
            ranges.append((start, e))
            start = None
    if start is not None:
        ranges.append((start, len(text)))
    return ranges


def _tags(value, limit: int = 12, width: int = 30) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()[:width]
        if text and text not in out:
            out.append(text)
    return out[:limit]


def _labels(value) -> tuple[list[str], list[str]]:
    """(canonical labels kept, unknown labels dropped)."""
    kept, dropped = [], []
    labels = canonical_labels()
    for label in _tags(value, limit=len(labels) + 10):
        (kept if label in labels and label not in kept else dropped).append(label)
    return kept, dropped


def _mode(value) -> str:
    if value not in MODES:
        raise Invalid(f"mode must be one of {list(MODES)}")
    return value


class PreferenceStore:
    def __init__(self, service: "Penombre"):
        self.svc = service
        root = service.config.data_dir / "preferences"
        self.docs_dir, self.chunks_dir, self.jobs_dir = root / "documents", root / "chunks", root / "jobs"
        for d in (self.docs_dir, self.chunks_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.last_recovery = self._recover()

    # ------------------------------------------------------------ storage

    def _now(self) -> str:
        return self.svc._now().astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _audit(self, action: str, actor: str, ref: str, **kw) -> None:
        after = kw.get("after")
        if kw.get("reason"):
            after = {**(after if isinstance(after, dict) else {"value": after}), "reason": kw["reason"]}
        self.svc.memory.store.audit(action, ref, before=kw.get("before"), after=after, actor=actor)

    def get_document(self, doc_id: str) -> dict:
        path = self.docs_dir / f"{doc_id}.json"
        if not files.SAFE_SEGMENT.match(doc_id) or not path.exists():
            raise NotFound(f"no preference document {doc_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_document(self, doc: dict) -> None:
        files.write_json_atomic(self.docs_dir / f"{doc['id']}.json", doc)

    def _chunks(self, doc_id: str) -> list[dict]:
        path = self.chunks_dir / f"{doc_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))["chunks"] if path.exists() else []

    def _save_chunks(self, doc: dict, chunks: list[dict]) -> None:
        for chunk in chunks:  # the invariant, checked on every write
            if chunk["text"] != doc["originalContent"][chunk["startOffset"]:chunk["endOffset"]]:
                raise Invalid(f"chunk {chunk['id']} text does not match the original at its offsets")
            if chunk["contentHash"] != files.content_hash(chunk["text"]):
                raise Invalid(f"chunk {chunk['id']} hash mismatch")
        files.write_json_atomic(self.chunks_dir / f"{doc['id']}.json", {"documentId": doc["id"], "chunks": chunks})
        doc["chunkCount"] = sum(1 for c in chunks if c["status"] != "archived")
        doc["activeChunkCount"] = sum(1 for c in chunks if c["status"] == "active")
        doc["updatedAt"] = self._now()
        self._save_document(doc)
        self._project(doc, chunks)

    def _project(self, doc: dict, chunks: list[dict]) -> None:
        """Retrievable = active chunk of an active, enabled retrieval document. Anything else that was ever projected is
        marked not current (a session_pinned document never has current chunks in the index)."""
        for chunk in chunks:
            live = retrievable(doc) and chunk["status"] == "active"
            projected = self.svc.conn.execute("SELECT 1 FROM memories WHERE id = ?", (chunk["id"],)).fetchone() is not None
            if not live and not projected:
                continue
            project_chunk(self.svc.conn, doc, chunk, live)

    def _make_chunk(self, doc: dict, start: int, end: int, *, labels=(), free_tags=(), entities=(), status="review",
                    origin="segmentation", job_id=None, replaces=()) -> dict:
        text = doc["originalContent"][start:end]
        ts = self._now()
        return {
            "id": files.new_id("pc"), "documentId": doc["id"], "startOffset": start, "endOffset": end, "text": text,
            "labels": list(labels), "freeTags": list(free_tags), "entities": list(entities), "status": status,
            "contentHash": files.content_hash(text), "documentSha256": doc["sha256"], "origin": origin, "segmentationJobId": job_id,
            "replaces": list(replaces), "replacedBy": [], "createdAt": ts, "updatedAt": ts,
        }

    @staticmethod
    def _actor(payload: dict) -> str:
        actor = str(payload.get("actor") or "user")
        if not identity.is_user(actor):
            raise Invalid(f"preference documents are reviewed by {sorted(ACTORS)} only")
        return actor

    # ------------------------------------------------------------ documents

    def list_documents(self, status: str | None = None, owner: str | None = None) -> list[dict]:
        docs = [d for d in files.iter_json_docs(self.docs_dir) if (not status or d["status"] == status) and (not owner or _owner_role(d["owner"]) == _owner_role(owner))]
        return sorted(docs, key=lambda d: d["createdAt"])

    def create_document(self, payload: dict, *, previous: dict | None = None) -> dict:
        """A new draft (or a new draft version of `previous`). Nothing is segmented, published or injected here."""
        actor = self._actor(payload)
        content = payload.get("originalContent")
        if not isinstance(content, str) or not content.strip():
            raise Invalid("originalContent is required (stored verbatim)")
        owner = str(payload.get("owner") or (previous or {}).get("owner") or "user")
        if _owner_role(owner) not in OWNERS:
            raise Invalid(f"owner must be one of {list(OWNERS)}")
        title = str(payload.get("title") or (previous or {}).get("title") or "").strip()[:80]
        if not title:
            raise Invalid("title is required")
        mode = _mode(payload.get("mode") or (mode_of(previous) if previous else "session_pinned"))
        labels, dropped = _labels(payload.get("initialLabels", (previous or {}).get("initialLabels", [])))
        ts = self._now()
        doc_id = files.new_id("pd")
        doc = {
            "id": doc_id, "owner": owner, "title": title, "originalContent": content, "sha256": files.content_hash(content),
            "mode": mode, "enabled": True, "status": "draft", "createdAt": ts, "updatedAt": ts, "segmentationVersion": None,
            "chunkCount": 0, "activeChunkCount": 0, "initialLabels": labels, "processing": None,
            "version": (previous["version"] + 1) if previous else 1, "previousVersionId": previous["id"] if previous else None,
            "lineageId": previous["lineageId"] if previous else doc_id, "supersededBy": None, "publishedAt": None, "archivedAt": None,
        }
        with self._lock:
            self._save_document(doc)
            self._audit("preference_document_created", actor, doc_id, after={k: v for k, v in doc.items() if k != "originalContent"},
                        reason=f"sha256 {doc['sha256'][:16]}; dropped labels {dropped}" if dropped else f"sha256 {doc['sha256'][:16]}")
        return doc

    def new_version(self, doc_id: str, payload: dict) -> dict:
        """A new draft in the same lineage, starting from this version's text unless new text is given. The version in
        effect keeps serving until the draft is published."""
        with self._lock:
            doc = self.get_document(doc_id)
            content = payload.get("originalContent", doc["originalContent"])
            return self.create_document({**payload, "originalContent": content}, previous=doc)

    def patch_document(self, doc_id: str, payload: dict) -> dict:
        """Metadata in place. New text: in place on a draft (or legacy failed document), else a new draft version.
        Mode: in place (a published document switches between the pack and retrieval at once, budget permitting)."""
        actor = self._actor(payload)
        with self._lock:
            doc = self.get_document(doc_id)
            if doc["status"] == "archived":
                raise Invalid("archived documents are read-only")
            if "originalContent" in payload and payload["originalContent"] != doc["originalContent"]:
                if doc["status"] in ("review", "active", "processing"):
                    return {"document": self.create_document({**payload, "actor": actor}, previous=doc), "createdVersion": True, "previousId": doc_id}
                content = payload["originalContent"]
                if not isinstance(content, str) or not content.strip():
                    raise Invalid("originalContent cannot be empty")
                chunks = self._chunks(doc_id)
                for chunk in chunks:  # history stays on disk; none of it is current for the new text
                    if chunk["status"] != "archived":
                        chunk.update(status="archived", updatedAt=self._now())
                files.write_json_atomic(self.chunks_dir / f"{doc_id}.json", {"documentId": doc_id, "chunks": chunks})
                doc.update(originalContent=content, sha256=files.content_hash(content), chunkCount=0, activeChunkCount=0, status="draft")
            before = {k: doc.get(k) for k in ("title", "owner", "initialLabels", "mode")}
            if "title" in payload:
                title = str(payload["title"]).strip()[:80]
                if not title:
                    raise Invalid("title cannot be empty")
                doc["title"] = title
            if "owner" in payload:
                if _owner_role(payload["owner"]) not in OWNERS:
                    raise Invalid(f"owner must be one of {list(OWNERS)}")
                doc["owner"] = payload["owner"]
            if "initialLabels" in payload:
                doc["initialLabels"] = _labels(payload["initialLabels"])[0]
            if "mode" in payload and _mode(payload["mode"]) != mode_of(doc):
                if doc["status"] == "active":
                    self._switch_mode(doc, payload["mode"])
                elif doc["status"] == "draft":
                    doc["mode"] = payload["mode"]
                else:
                    raise Invalid(f"the mode of a {doc['status']} document is chosen on a new version")
            doc["updatedAt"] = self._now()
            self._save_document(doc)
            if doc["status"] == "active":
                self._project(doc, self._chunks(doc_id))  # titles / owner shown with the chunks
            self._audit("preference_document_updated", actor, doc_id, before=before, after={k: doc.get(k) for k in before})
        return {"document": doc, "createdVersion": False}

    def _switch_mode(self, doc: dict, mode: str) -> None:
        if mode == "session_pinned":
            self._check_budget(doc)
            doc["mode"] = mode  # its chunks leave the index (projected as not current)
        else:
            doc["mode"] = mode
            if not any(c["status"] == "active" for c in self._chunks(doc["id"])):
                self._chunk_locally(doc)

    def _chunk_locally(self, doc: dict) -> list[dict]:
        """Active retrieval chunks from the deterministic local cut; any earlier chunks are archived (kept)."""
        content = doc["originalContent"]
        chunks = self._chunks(doc["id"])
        for chunk in chunks:
            if chunk["status"] != "archived":
                chunk.update(status="archived", updatedAt=self._now())
        fresh = [self._make_chunk(doc, s, e, labels=doc.get("initialLabels") or [], status="active", origin="local")
                 for s, e in local_chunks(content)]
        if "".join(c["text"] for c in fresh) != content:
            raise Invalid("local chunks do not reproduce the original exactly")
        doc["segmentationVersion"] = LOCAL_CHUNKING
        self._save_chunks(doc, chunks + fresh)
        return fresh

    # ------------------------------------------------------------ publish / effect

    def publish(self, doc_id: str, payload: dict) -> dict:
        """the user publishes. A draft goes live in its mode (session_pinned: whole original, no chunks; retrieval: local
        chunks). A legacy document under review publishes its reviewed chunks (retrieval). Other versions of the same
        lineage that were in effect or under review retire, so exactly one version is in effect."""
        actor = self._actor(payload)
        with self._lock:
            doc = self.get_document(doc_id)
            if doc["status"] == "draft":
                if "mode" in payload:
                    doc["mode"] = _mode(payload["mode"])
                doc.setdefault("mode", "session_pinned")
                if not doc["originalContent"].strip():
                    raise Invalid("an empty document cannot be published")
                if files.content_hash(doc["originalContent"]) != doc["sha256"]:
                    raise Invalid("stored original does not match its sha256")
                if doc["mode"] == "session_pinned":
                    self._check_budget(doc)
                doc.update(status="active", enabled=True, publishedAt=self._now(), processing=None)
                if doc["mode"] == "retrieval":
                    self._chunk_locally(doc)
                else:
                    chunks = self._chunks(doc_id)
                    for chunk in chunks:
                        if chunk["status"] != "archived":
                            chunk.update(status="archived", updatedAt=self._now())
                    self._save_chunks(doc, chunks)
            elif doc["status"] == "review":  # legacy segmented document
                if payload.get("mode", "retrieval") != "retrieval":
                    raise Invalid("a legacy chunked document publishes as retrieval; make a new version for Session 常驻")
                chunks = self._chunks(doc_id)
                live = [c for c in chunks if c["status"] != "archived"]
                if "".join(c["text"] for c in live) != doc["originalContent"]:
                    raise Invalid("current chunks no longer tile the original")
                for chunk in live:
                    if chunk["status"] == "review":
                        chunk.update(status="active", updatedAt=self._now())
                doc.update(status="active", mode="retrieval", enabled=True, publishedAt=self._now())
                self._save_chunks(doc, chunks)
            else:
                raise Invalid(f"only a draft (or a legacy document in review) can be published ({doc_id} is {doc['status']})")
            retired = self._retire_lineage(doc, actor)
            self._audit("preference_document_published", actor, doc_id,
                        after={"mode": doc["mode"], "version": doc["version"], "sha256": doc["sha256"], "retired": retired,
                               "activeChunks": doc.get("activeChunkCount", 0), "estimatedTokens": estimate_tokens(doc["originalContent"])})
        return doc

    def _retire_lineage(self, doc: dict, actor: str) -> list[str]:
        retired = []
        for other in files.iter_json_docs(self.docs_dir):
            if other["id"] == doc["id"] or other.get("lineageId") != doc["lineageId"] or other["status"] in ("draft", "archived"):
                continue
            chunks = self._chunks(other["id"])
            for chunk in chunks:
                if chunk["status"] != "archived":
                    chunk.update(status="archived", updatedAt=self._now())
            other.update(status="archived", archivedAt=self._now(), supersededBy=doc["id"])
            self._save_chunks(other, chunks)
            self._audit("preference_document_superseded", actor, other["id"], event_ids=[doc["id"]], after={"supersededBy": doc["id"]})
            retired.append(other["id"])
        return retired

    def unpublish(self, doc_id: str, payload: dict) -> dict:
        """Out of effect, back to an editable draft (text kept; retrieval chunks archived, re-cut on the next publish)."""
        actor = self._actor(payload)
        with self._lock:
            doc = self.get_document(doc_id)
            if doc["status"] != "active":
                raise Invalid(f"only a published document can be unpublished ({doc_id} is {doc['status']})")
            chunks = self._chunks(doc_id)
            for chunk in chunks:
                if chunk["status"] != "archived":
                    chunk.update(status="archived", updatedAt=self._now())
            doc.update(status="draft", mode=mode_of(doc), lastPublishedAt=doc.get("publishedAt"), publishedAt=None)
            self._save_chunks(doc, chunks)
            self._audit("preference_document_unpublished", actor, doc_id, before={"status": "active"}, after={"status": "draft"},
                        reason=str(payload.get("reason") or ""))
        return doc

    def set_enabled(self, doc_id: str, payload: dict, value: bool) -> dict:
        actor = self._actor(payload)
        with self._lock:
            doc = self.get_document(doc_id)
            if doc["status"] != "active":
                raise Invalid(f"only a published document can be enabled or disabled ({doc_id} is {doc['status']})")
            if value and not enabled(doc) and mode_of(doc) == "session_pinned":
                self._check_budget(doc)
            before = enabled(doc)
            doc.update(enabled=value, mode=mode_of(doc), updatedAt=self._now())
            self._save_document(doc)
            self._project(doc, self._chunks(doc_id))
            self._audit("preference_document_enabled" if value else "preference_document_disabled", actor, doc_id,
                        before={"enabled": before}, after={"enabled": value}, reason=str(payload.get("reason") or ""))
        return doc

    def archive_document(self, doc_id: str, payload: dict) -> dict:
        actor = self._actor(payload)
        with self._lock:
            doc = self.get_document(doc_id)
            if doc["status"] == "archived":
                raise Invalid(f"{doc_id} is already archived")
            before = doc["status"]
            doc.update(status="archived", archivedAt=self._now())
            chunks = self._chunks(doc_id)
            for chunk in chunks:
                if chunk["status"] != "archived":
                    chunk.update(status="archived", updatedAt=self._now())
            self._save_chunks(doc, chunks)
            self._audit("preference_document_archived", actor, doc_id, before={"status": before}, after={"status": "archived"},
                        reason=str(payload.get("reason") or ""))
        return doc

    # ------------------------------------------------------------ Session Preference Pack

    def _pinned_docs(self) -> list[dict]:
        """In effect and pinned, one per lineage (the highest version, should two ever be active), stable order."""
        by_lineage: dict[str, dict] = {}
        for doc in files.iter_json_docs(self.docs_dir):
            if not pinned(doc):
                continue
            key = doc.get("lineageId") or doc["id"]
            if key not in by_lineage or doc["version"] > by_lineage[key]["version"]:
                by_lineage[key] = doc
        return sorted(by_lineage.values(), key=lambda d: (d.get("publishedAt") or "", d["id"]))

    def _check_budget(self, doc: dict) -> None:
        others = sum(estimate_tokens(d["originalContent"]) for d in self._pinned_docs()
                     if (d.get("lineageId") or d["id"]) != (doc.get("lineageId") or doc["id"]))
        total = others + estimate_tokens(doc["originalContent"])
        if total > PINNED_BUDGET_TOKENS:
            raise Invalid(f"Session 常驻偏好会超出预算：约 {total} / {PINNED_BUDGET_TOKENS} tokens。"
                          "请先精简这份文档，或停用 / 改为普通检索其他常驻文档。")

    def session_pack(self, with_content: bool = True) -> dict:
        """What a new the assistant runtime session loads: every published, enabled session_pinned document, verbatim."""
        docs, warnings = [], []
        for doc in self._pinned_docs():
            if files.content_hash(doc["originalContent"]) != doc["sha256"]:
                warnings.append(f"{doc['id']}: original does not match its sha256; left out")
                continue
            entry = {"id": doc["id"], "lineageId": doc.get("lineageId") or doc["id"], "title": doc["title"], "version": doc["version"],
                     "sha256": doc["sha256"], "mode": "session_pinned", "published": True, "enabled": True, "publishedAt": doc.get("publishedAt"),
                     "chars": len(doc["originalContent"]), "estimatedTokens": estimate_tokens(doc["originalContent"])}
            if with_content:
                entry["content"] = doc["originalContent"]
            docs.append(entry)
        identity = [[d["id"], d["version"], d["sha256"], d["title"]] for d in docs]
        total = sum(d["estimatedTokens"] for d in docs)
        return {"documents": docs, "packHash": files.content_hash(json.dumps(identity, ensure_ascii=False))[:16],
                "totalChars": sum(d["chars"] for d in docs), "totalTokens": total, "budgetTokens": PINNED_BUDGET_TOKENS,
                "overBudget": total > PINNED_BUDGET_TOKENS, "warnings": warnings}

    # ------------------------------------------------------------ legacy segmentation records

    def get_job(self, job_id: str) -> dict:
        path = self.jobs_dir / f"{job_id}.json"
        if not files.SAFE_SEGMENT.match(job_id) or not path.exists():
            raise NotFound(f"no segmentation job {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _recover(self) -> dict:
        """Segmentation is retired: a legacy job left queued or running is closed as failed, its document with it."""
        failed = []
        for job in files.iter_json_docs(self.jobs_dir):
            if job["status"] in ("queued", "processing"):
                job.update(status="failed", error="segmentation retired", completedAt=files.now_iso())
                files.write_json_atomic(self.jobs_dir / f"{job['id']}.json", job)
                try:
                    doc = self.get_document(job["documentId"])
                except NotFound:
                    continue
                if doc["status"] == "processing":
                    doc.update(status="failed", processing={**(doc.get("processing") or {}), "error": "segmentation retired"})
                    self._save_document(doc)
                failed.append(job["id"])
        return {"interruptedJobs": failed}

    # ------------------------------------------------------------ chunks (retrieval documents)

    def list_chunks(self, doc_id: str) -> list[dict]:
        self.get_document(doc_id)
        return self._chunks(doc_id)

    def _find_chunk(self, chunk_id: str) -> tuple[dict, list[dict], dict]:
        for path in self.chunks_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for chunk in data["chunks"]:
                if chunk["id"] == chunk_id:
                    doc = self.get_document(data["documentId"])
                    return doc, data["chunks"], chunk
        raise NotFound(f"no preference chunk {chunk_id}")

    def chunk_ref(self, chunk_id: str) -> dict | None:
        """Provenance for retrieval: where a chunk sits in its document."""
        try:
            doc, _, chunk = self._find_chunk(chunk_id)
        except NotFound:
            return None
        return {"documentId": doc["id"], "documentTitle": doc["title"], "owner": doc["owner"], "startOffset": chunk["startOffset"],
                "endOffset": chunk["endOffset"], "documentSha256": doc["sha256"], "labels": chunk["labels"], "version": doc["version"]}

    @staticmethod
    def _editable(doc: dict) -> None:
        if doc["status"] not in ("review", "active") or mode_of(doc) != "retrieval":
            raise Invalid(f"chunks of a {doc['status']} {mode_of(doc)} document cannot be changed")

    def patch_chunk(self, chunk_id: str, payload: dict) -> dict:
        actor = self._actor(payload)
        if "text" in payload:
            raise Invalid("chunk text comes from the original; to change the words, edit the document (a new version)")
        with self._lock:
            doc, chunks, chunk = self._find_chunk(chunk_id)
            self._editable(doc)
            if chunk["status"] == "archived":
                raise Invalid(f"{chunk_id} is archived")
            before = {k: chunk[k] for k in ("labels", "freeTags", "entities", "startOffset", "endOffset")}
            if "labels" in payload:
                chunk["labels"] = _labels(payload["labels"])[0]
            if "freeTags" in payload:
                chunk["freeTags"] = _tags(payload["freeTags"], 8, 20)
            if "entities" in payload:
                chunk["entities"] = _tags(payload["entities"], 8, 30)
            if "endOffset" in payload or "startOffset" in payload:
                self._move_boundary(doc, chunks, chunk, payload)
            chunk["updatedAt"] = self._now()
            self._save_chunks(doc, chunks)
            self._audit("preference_chunk_updated", actor, chunk_id, before=before,
                        after={k: chunk[k] for k in before}, reason=f"document {doc['id']}")
        return chunk

    def _move_boundary(self, doc: dict, chunks: list[dict], chunk: dict, payload: dict) -> None:
        """Move the boundary shared with the adjacent chunk; both texts are re-derived from the original."""
        live = [c for c in chunks if c["status"] != "archived"]
        if "endOffset" in payload:
            new = int(payload["endOffset"])
            neighbour = next((c for c in live if c["startOffset"] == chunk["endOffset"]), None)
            if neighbour is None or not (chunk["startOffset"] < new < neighbour["endOffset"]):
                raise Invalid("endOffset must stay inside this chunk and the next adjacent one")
            chunk["endOffset"], neighbour["startOffset"] = new, new
            pair = (chunk, neighbour)
        else:
            new = int(payload["startOffset"])
            neighbour = next((c for c in live if c["endOffset"] == chunk["startOffset"]), None)
            if neighbour is None or not (neighbour["startOffset"] < new < chunk["endOffset"]):
                raise Invalid("startOffset must stay inside this chunk and the previous adjacent one")
            chunk["startOffset"], neighbour["endOffset"] = new, new
            pair = (neighbour, chunk)
        for c in pair:
            c["text"] = doc["originalContent"][c["startOffset"]:c["endOffset"]]
            c["contentHash"] = files.content_hash(c["text"])
            c["updatedAt"] = self._now()

    def split_chunk(self, chunk_id: str, payload: dict) -> list[dict]:
        actor = self._actor(payload)
        with self._lock:
            doc, chunks, chunk = self._find_chunk(chunk_id)
            self._editable(doc)
            if chunk["status"] == "archived":
                raise Invalid(f"{chunk_id} is archived")
            at = payload.get("offset")
            if at is None and payload.get("relativeOffset") is not None:
                at = chunk["startOffset"] + int(payload["relativeOffset"])
            if at is None or not (chunk["startOffset"] < int(at) < chunk["endOffset"]):
                raise Invalid("offset must fall strictly inside the chunk")
            at = int(at)
            meta = {"labels": chunk["labels"], "free_tags": chunk["freeTags"], "entities": chunk["entities"]}
            parts = [self._make_chunk(doc, chunk["startOffset"], at, status=chunk["status"], origin="split", replaces=[chunk_id], **meta),
                     self._make_chunk(doc, at, chunk["endOffset"], status=chunk["status"], origin="split", replaces=[chunk_id], **meta)]
            chunk.update(status="archived", replacedBy=[p["id"] for p in parts], updatedAt=self._now())
            chunks[chunks.index(chunk) + 1:chunks.index(chunk) + 1] = parts
            self._save_chunks(doc, chunks)
            self._audit("preference_chunk_split", actor, chunk_id, event_ids=[p["id"] for p in parts], after={"at": at})
        return parts

    def merge_chunks(self, payload: dict) -> dict:
        actor = self._actor(payload)
        ids = [str(i) for i in payload.get("chunkIds") or []]
        if len(ids) < 2:
            raise Invalid("merge needs at least two adjacent chunkIds")
        with self._lock:
            doc, chunks, _ = self._find_chunk(ids[0])
            self._editable(doc)
            parts = sorted((c for c in chunks if c["id"] in ids), key=lambda c: c["startOffset"])
            if len(parts) != len(ids) or any(c["status"] == "archived" for c in parts):
                raise Invalid("every chunk must belong to the same document and be current")
            if any(a["endOffset"] != b["startOffset"] for a, b in zip(parts, parts[1:])):
                raise Invalid("only adjacent chunks can be merged")
            status = "active" if any(c["status"] == "active" for c in parts) else parts[0]["status"]
            merged = self._make_chunk(
                doc, parts[0]["startOffset"], parts[-1]["endOffset"], status=status, origin="merge", replaces=ids,
                labels=list(dict.fromkeys(l for c in parts for l in c["labels"])),
                free_tags=_tags([t for c in parts for t in c["freeTags"]], 8, 20),
                entities=_tags([e for c in parts for e in c["entities"]], 8, 30),
            )
            for c in parts:
                c.update(status="archived", replacedBy=[merged["id"]], updatedAt=self._now())
            chunks.insert(chunks.index(parts[-1]) + 1, merged)
            self._save_chunks(doc, chunks)
            self._audit("preference_chunks_merged", actor, merged["id"], event_ids=ids, after={"startOffset": merged["startOffset"],
                                                                                              "endOffset": merged["endOffset"]})
        return merged

    def set_excluded(self, chunk_id: str, payload: dict, excluded: bool) -> dict:
        actor = self._actor(payload)
        with self._lock:
            doc, chunks, chunk = self._find_chunk(chunk_id)
            self._editable(doc)
            if chunk["status"] == "archived":
                raise Invalid(f"{chunk_id} is archived")
            before = chunk["status"]
            chunk["status"] = "excluded" if excluded else ("active" if doc["status"] == "active" else "review")
            chunk["updatedAt"] = self._now()
            self._save_chunks(doc, chunks)
            self._audit("preference_chunk_excluded" if excluded else "preference_chunk_restored", actor, chunk_id,
                        before={"status": before}, after={"status": chunk["status"]}, reason=str(payload.get("reason") or ""))
        return chunk

    # ------------------------------------------------------------ index

    def project_all(self) -> int:
        n = 0
        for doc in files.iter_json_docs(self.docs_dir):
            if retrievable(doc):
                for chunk in self._chunks(doc["id"]):
                    if chunk["status"] == "active":
                        project_chunk(self.svc.conn, doc, chunk, True)
                        n += 1
        return n


def chunk_title(doc: dict, chunk: dict) -> str:
    first = chunk["text"].strip().splitlines()[0] if chunk["text"].strip() else ""
    return f"《{doc['title']}》{first[:40]}{'…' if len(first) > 40 else ''}"[:80]


def project_chunk(conn, doc: dict, chunk: dict, live: bool) -> None:
    memory = files.Memory(
        id=chunk["id"], title=chunk_title(doc, chunk), body=chunk["text"], injection="dynamic",
        status="confirmed" if live else chunk["status"] if chunk["status"] != "active" else "disabled" if doc["status"] == "active" else doc["status"],
        author=doc["owner"], keywords=[*chunk["labels"], *chunk["freeTags"]], sources=[], created=chunk["createdAt"], updated=chunk["updatedAt"],
    )
    index.put_memory(conn, memory, {
        "kind": "preference", "importance": 0.5, "effectiveAt": doc.get("publishedAt") or doc["createdAt"],
        "lineageId": chunk["id"], "version": doc.get("version", 1), "entities": list(chunk["entities"]),
        "tags": [*chunk["labels"], *chunk["freeTags"]],
    })
