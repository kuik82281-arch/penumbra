"""专名库: the user's own list of the names in their world - people, places, pets, things - each with its aliases.

Each name can carry a profile: a short, neutral summary of what the memories say about it, rewritten by the LLM in
the background when the memories that mention it change (at least MIN_PROFILE_MEMORIES of them). When a message names
it, the profile travels with that turn's memory as context.

She keeps it by hand (Memory Studio); nothing else writes it. When a message names one of them (the name or any alias),
the memories that mention it (under any of its names) are a precise hit: they pass the gates, and a memory that says
"年糕" is found when she says "糕糕". A single Chinese character never counts as a name (too easily part of another
word); an emoji does.
"""
from __future__ import annotations

import json
import re
import unicodedata

from .. import identity, prompts
from ..errors import Invalid, NotFound
from .store import Store, mint, now_iso

KINDS = ("人", "地方", "宠物", "物件", "其他")
MAX_ALIASES = 12
MIN_PROFILE_MEMORIES = 2
MAX_PROFILE_SOURCES = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS names (
  name_id TEXT PRIMARY KEY, name TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]', kind TEXT NOT NULL DEFAULT '其他',
  note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
"""

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def ensure_schema(store: Store) -> None:
    store.db.executescript(SCHEMA)
    have = {r[1] for r in store.db.execute("PRAGMA table_info(names)")}
    for col, decl in (("profile", "TEXT NOT NULL DEFAULT ''"), ("profile_at", "TEXT"), ("profile_basis", "TEXT NOT NULL DEFAULT ''")):
        if col not in have:
            store.db.execute(f"ALTER TABLE names ADD COLUMN {col} {decl}")


def _view(row) -> dict:
    keys = row.keys()
    return {"name_id": row["name_id"], "name": row["name"], "aliases": json.loads(row["aliases"]), "kind": row["kind"], "note": row["note"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "profile": row["profile"] if "profile" in keys else "", "profile_at": row["profile_at"] if "profile_at" in keys else None}


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
            db.execute("INSERT INTO names (name_id, name, aliases, kind, note, created_at, updated_at) VALUES (?,?,?,?,?,?,?)", (ident, name, json.dumps(aliases, ensure_ascii=False), kind, note, now_iso(), now_iso()))
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


# ------------------------------------------------------------ profiles

def _memories_about(store: Store, entry: dict) -> list[dict]:
    """Active Episodes and Patterns that mention the name under any of its words (dreams excluded: not facts)."""
    words = [w for w in [entry["name"], *entry["aliases"]] if _counts(w)]
    out = []
    for e in store.episodes(status="active"):
        if e.get("kind") != "dream" and any(w in e["content"] or w in " ".join(e["entities"]) for w in words):
            out.append({"id": e["episode_id"], "version": e["version"], "time": e["time_start"], "text": e["content"]})
    for p in store.patterns(status="active"):
        if any(w in f"{p['title']} {p['narrative']} {p['current_state']}" for w in words):
            out.append({"id": p["pattern_id"], "version": p["version"], "time": p["current_state_since"] or p["valid_from"],
                        "text": f"{p['title']}：{p['current_state']}"})
    out.sort(key=lambda m: m["time"] or "")
    return out[-MAX_PROFILE_SOURCES:]


def _basis(memories: list[dict]) -> str:
    return ",".join(f"{m['id']}@{m['version']}" for m in memories)


def write_profile(store: Store, client, ident: str, force: bool = False) -> dict:
    """(Re)write one name's profile from the memories about it; unchanged memories -> nothing to do unless forced."""
    row = store.one("SELECT * FROM names WHERE name_id = ?", (ident,))
    if row is None:
        raise NotFound(f"no name {ident}")
    entry = _view(row)
    memories = _memories_about(store, entry)
    basis = _basis(memories)
    if len(memories) < MIN_PROFILE_MEMORIES or (not force and basis == (row["profile_basis"] or "")):
        return {**entry, "skipped": True}
    if not client.available():
        raise Invalid("the LLM is not configured")
    payload = {"name": entry["name"], "aliases": entry["aliases"], "kind": entry["kind"], "note": entry["note"],
               "memories": [{"time": m["time"], "text": m["text"]} for m in memories]}
    answer, meta = client.chat_json(prompts.get("profile"), json.dumps(payload, ensure_ascii=False), max_tokens=500)
    profile = str((answer or {}).get("profile") or "").strip()[:400]
    if not profile:
        raise Invalid("the LLM returned no profile")
    with store.tx() as db:
        db.execute("UPDATE names SET profile=?, profile_at=?, profile_basis=? WHERE name_id=?", (profile, now_iso(), basis, ident))
    store.audit("name.profiled", ident, after={"memories": len(memories), "usage": meta.get("usage", {})}, actor="llm")
    return _view(store.one("SELECT * FROM names WHERE name_id = ?", (ident,)))


def refresh_profiles(store: Store, client) -> list[str]:
    """After a curation run: rewrite the profiles whose memories changed. Failures are left for the next run."""
    done = []
    for entry in all_names(store):
        try:
            if not write_profile(store, client, entry["name_id"]).get("skipped"):
                done.append(entry["name_id"])
        except Invalid:
            continue
    return done
