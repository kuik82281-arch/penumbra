"""The truth: plain files under data/. Everything in index.sqlite can be rebuilt from here.

originals/<source>/<conversationId>.jsonl   append-only; one immutable original per line
memories/<memoryId>.md                      one curated memory: frontmatter + body
ledger/<YYYY-MM>.jsonl                      append-only inject / recall / memory events
candidates/<candidateId>.json               proposed memory (lifecycle state; every change is audited)
events/<eventId>.json                       confirmed EVENT, one file per version (never rewritten in content)
batches/<batchId>.json                      one Daily Event Pass window
audit/<YYYY-MM>.jsonl                       append-only memory lifecycle audit
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def original_id(source: str, conversation_id: str, item_id: str) -> str:
    digest = hashlib.sha256(f"{source}\x1f{conversation_id}\x1f{item_id}".encode("utf-8")).hexdigest()
    return "o_" + digest[:20]


def new_memory_id() -> str:
    return f"m_{datetime.now(timezone.utc):%Y%m%d}_{secrets.token_hex(3)}"


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%d}_{secrets.token_hex(4)}"


REPLACE_ATTEMPTS = 20
REPLACE_BACKOFF_S = 0.025


def write_json_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Windows refuses to replace a file another handle has open at that instant (a reader polling a job's status):
    # that is transient, so the rename is retried briefly instead of failing the write and leaving the .tmp behind.
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_BACKOFF_S * (attempt + 1))


def iter_json_docs(root: Path) -> Iterator[dict]:
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"[penumbra] skipping unreadable {path}: {error}")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A torn last line (crash mid-write) must not take the whole store down.
                print(f"[penumbra] skipping unreadable line {path}:{number}")


# ---------------------------------------------------------------- originals

@dataclass
class Original:
    id: str
    source: str
    conversation_id: str
    item_id: str
    role: str
    content: str
    created_at: str
    ingested_at: str
    sha256: str
    # Provenance (P1). Older lines lack them; they are never part of the immutability hash.
    claude_session_id: str | None = None
    turn_id: str | None = None
    source_type: str = "chat_message"
    metadata: dict = field(default_factory=dict)
    event_refs: list = field(default_factory=list)  # relationship / tool events that arrived with this turn
    attachments: list = field(default_factory=list)  # immutable asset references; bytes stay in the owning asset store

    def to_record(self) -> dict:
        record = {
            "id": self.id,
            "source": self.source,
            "conversationId": self.conversation_id,
            "itemId": self.item_id,
            "role": self.role,
            "content": self.content,
            "createdAt": self.created_at,
            "ingestedAt": self.ingested_at,
            "sha256": self.sha256,
            "sourceType": self.source_type,
        }
        if self.claude_session_id:
            record["claudeSessionId"] = self.claude_session_id
        if self.turn_id:
            record["turnId"] = self.turn_id
        if self.metadata:
            record["metadata"] = self.metadata
        if self.event_refs:
            record["eventRefs"] = self.event_refs
        if self.attachments:
            record["attachments"] = self.attachments
        return record

    @staticmethod
    def from_record(record: dict) -> "Original":
        return Original(
            id=record["id"],
            source=record["source"],
            conversation_id=record["conversationId"],
            item_id=record["itemId"],
            role=record["role"],
            content=record["content"],
            created_at=record["createdAt"],
            ingested_at=record.get("ingestedAt", ""),
            sha256=record.get("sha256") or content_hash(record["content"]),
            claude_session_id=record.get("claudeSessionId"),
            turn_id=record.get("turnId"),
            source_type=record.get("sourceType") or "chat_message",
            metadata=record.get("metadata") or {},
            event_refs=record.get("eventRefs") or [],
            attachments=record.get("attachments") or [],
        )


def originals_path(root: Path, source: str, conversation_id: str) -> Path:
    if not SAFE_SEGMENT.match(source) or not SAFE_SEGMENT.match(conversation_id):
        raise ValueError("source and conversationId must be simple path-safe ids")
    return root / source / f"{conversation_id}.jsonl"


def iter_originals(root: Path) -> Iterator[Original]:
    if not root.exists():
        return
    for path in sorted(root.glob("*/*.jsonl")):
        for record in read_jsonl(path):
            yield Original.from_record(record)


# ---------------------------------------------------------------- memories

MEMORY_FIELDS = ("id", "title", "injection", "status", "author", "keywords", "sources", "created", "updated")


@dataclass
class Memory:
    id: str
    title: str
    body: str = ""
    injection: str = "dynamic"  # static: always in the directory; dynamic: when relevant
    status: str = "confirmed"  # confirmed | candidate (candidates are never injected)
    author: str = "user"  # user | assistant | import | deepseek (older data: the two names)
    keywords: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # original ids this memory traces back to
    created: str = ""
    updated: str = ""

    def to_public(self, include_body: bool = True) -> dict:
        data = {
            "id": self.id,
            "title": self.title,
            "injection": self.injection,
            "status": self.status,
            "author": self.author,
            "keywords": self.keywords,
            "sources": self.sources,
            "created": self.created,
            "updated": self.updated,
        }
        if include_body:
            data["body"] = self.body
        return data


def _front_value(value) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).replace("\n", " ").strip()
    # Quote anything a reader could mistake for structure.
    if not text or text[0] in "[{\"" or text != text.strip() or ": " in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def write_memory_file(root: Path, memory: Memory) -> Path:
    if not SAFE_SEGMENT.match(memory.id):
        raise ValueError("bad memory id")
    root.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for key in MEMORY_FIELDS:
        lines.append(f"{key}: {_front_value(getattr(memory, key))}")
    lines.append("---")
    text = "\n".join(lines) + "\n" + memory.body.strip() + "\n"
    path = root / f"{memory.id}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def parse_memory_file(path: Path) -> Memory:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name}: missing frontmatter")
    _, front, body = text.split("---", 2)
    meta: dict = {}
    for raw in front.strip().splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        value = value.strip()
        if value[:1] in ("[", "{", '"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        meta[key.strip()] = value
    return Memory(
        id=str(meta.get("id") or path.stem),
        title=str(meta.get("title") or ""),
        body=body.strip("\n"),
        injection=str(meta.get("injection") or "dynamic"),
        status=str(meta.get("status") or "confirmed"),
        author=str(meta.get("author") or "user"),
        keywords=[str(k) for k in meta.get("keywords") or [] if str(k).strip()],
        sources=[str(s) for s in meta.get("sources") or [] if str(s).strip()],
        created=str(meta.get("created") or ""),
        updated=str(meta.get("updated") or meta.get("created") or ""),
    )


def iter_memories(root: Path) -> Iterator[Memory]:
    if not root.exists():
        return
    for path in sorted(root.glob("*.md")):
        try:
            yield parse_memory_file(path)
        except Exception as error:  # a hand-edited file with a typo must not break the service
            print(f"[penumbra] skipping memory {path.name}: {error}")


# ---------------------------------------------------------------- ledger

def ledger_path(root: Path, ts: str) -> Path:
    return root / f"{ts[:7]}.jsonl"


def iter_ledger(root: Path) -> Iterator[dict]:
    if not root.exists():
        return
    for path in sorted(root.glob("*.jsonl")):
        yield from read_jsonl(path)
