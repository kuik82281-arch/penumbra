"""专名库: the user's own list of the names in their world - people, places, pets, things - each with its aliases.

She keeps it by hand (Memory Studio); nothing else writes it. When a message names one of them (the name or any alias),
the memories that mention it (under any of its names) are a precise hit: they pass the gates, and a memory that says
"年糕" is found when she says "糕糕". A single Chinese character never counts as a name (too easily part of another
word); an emoji does.
"""
from __future__ import annotations

import json
import re
import unicodedata

from .. import identity
from ..errors import Invalid, NotFound
from .store import Store, mint, now_iso

KINDS = ("人", "地方", "宠物", "物件", "其他")
MAX_ALIASES = 12

SCHEMA = """
CREATE TABLE IF NOT EXISTS names (
  name_id TEXT PRIMARY KEY, name TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]', kind TEXT NOT NULL DEFAULT '其他',
  note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
"""

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def ensure_schema(store: Store) -> None:
    store.db.executescript(SCHEMA)


def _view(row) -> dict:
    return {"name_id": row["name_id"], "name": row["name"], "aliases": json.loads(row["aliases"]), "kind": row["kind"], "note": row["note"],
            "created_at": row["created_at"], "updated_at": row["updated_at"]}


def all_names(store: Store) -> list[dict]:
    return [_view(r) for r in store.all("SELECT * FROM names ORDER BY kind, name")]


def _word(value) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()[:30]


def save(store: Store, body: dict) -> dict:
    name = _word(body.get("name"))
    if not name:
        raise Invalid("name is required")
    aliases = list(dict.fromkeys(a for a in (_word(x) for x in body.get("aliases") or []) if a and a != name))[:MAX_ALIASES]
    kind = str(body.get("kind") or "其他")
    if kind not in KINDS:
        raise Invalid(f"kind must be one of {KINDS}")
    note = str(body.get("note") or "").strip()[:200]
    ident = str(body.get("name_id") or "")
    with store.tx() as db:
        if ident:
            if not db.execute("SELECT 1 FROM names WHERE name_id = ?", (ident,)).fetchone():
                raise NotFound(f"no name {ident}")
            db.execute("UPDATE names SET name=?, aliases=?, kind=?, note=?, updated_at=? WHERE name_id=?",
                       (name, json.dumps(aliases, ensure_ascii=False), kind, note, now_iso(), ident))
        else:
            ident = mint("name")
            db.execute("INSERT INTO names VALUES (?,?,?,?,?,?,?)", (ident, name, json.dumps(aliases, ensure_ascii=False), kind, note, now_iso(), now_iso()))
    store.audit("name.saved", ident, after={"name": name, "aliases": aliases}, actor=identity.USER_ACTOR)
    return _view(store.one("SELECT * FROM names WHERE name_id = ?", (ident,)))


def delete(store: Store, ident: str) -> dict:
    with store.tx() as db:
        gone = db.execute("DELETE FROM names WHERE name_id = ?", (ident,)).rowcount
    if not gone:
        raise NotFound(f"no name {ident}")
    store.audit("name.deleted", ident, actor=identity.USER_ACTOR)
    return {"deleted": ident}


def _counts(word: str) -> bool:
    """Two characters or more, or an emoji / symbol; never a single Chinese character or letter."""
    if len(word) >= 2:
        return True
    return bool(word) and not _CJK.match(word) and not word.isalnum()


def mentioned(store: Store, text: str) -> list[dict]:
    """The names this text mentions (by the name or an alias), each with every word it goes by."""
    norm = unicodedata.normalize("NFKC", text or "").lower()
    out = []
    for entry in all_names(store):
        words = [entry["name"], *entry["aliases"]]
        said = [w for w in words if _counts(w) and w.lower() in norm]
        if said:
            out.append({**entry, "words": words, "said": said})
    return out
