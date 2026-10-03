"""The eight decisions DeepSeek can make about a candidate, validated locally and applied atomically.

    CREATE_EPISODE  UPDATE_EPISODE  MERGE_EPISODE  CREATE_PATTERN  UPDATE_PATTERN     write to the long-term store
    QUARANTINE      REJECT          NO_ACTION                                          write nothing (the candidate is resolved)

A model answer is never trusted: shape, ids, RAW provenance, times, state history and text hygiene are all checked here,
and anything that does not hold makes the *whole* decision a QUARANTINE (the user can review and edit it in the Studio) -
never a partial write. `apply_plan` runs in one store transaction; states are never overwritten, only closed.
"""
from __future__ import annotations

import copy
import re
from datetime import timedelta

from .. import identity
from ..errors import Invalid, NotFound
from .store import Store, mint, normalize_time, now_iso, parse_time
from . import threads as threads_module

TERMINAL = ("QUARANTINE", "REJECT", "NO_ACTION")
WRITING = ("CREATE_EPISODE", "UPDATE_EPISODE", "MERGE_EPISODE", "CREATE_PATTERN", "UPDATE_PATTERN")
ALL_ACTIONS = WRITING + TERMINAL
MIN_CONFIDENCE = 0.7
MAX_ACTIONS = 8

# Transport roles are never names in memory: the two of them are always called by their names (identity.py).
_ROLE_WORDS = [(re.compile(r"用户"), lambda m: identity.user()), (re.compile(r"助手"), lambda m: identity.assistant()),
               (re.compile(r"\b[Tt]he user\b|\buser\b"), lambda m: identity.user()),
               (re.compile(r"\b[Aa]ssistant\b"), lambda m: identity.assistant())]


def clean_text(value, field: str, limit: int, required: bool = True) -> str:
    if not isinstance(value, str):
        if required:
            raise Invalid(f"{field} is required")
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    for pattern, name in _ROLE_WORDS:
        text = pattern.sub(name, text)
    if required and not text:
        raise Invalid(f"{field} is required")
    if len(text) > limit:
        raise Invalid(f"{field} is longer than {limit} characters")
    return text


def _strings(value, field: str, limit: int = 12, width: int = 40) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise Invalid(f"{field} must be a list")
    out = [clean_text(v, field, width, required=False) for v in value if isinstance(v, str)]
    return list(dict.fromkeys(t for t in out if t))[:limit]


def _unit(value, field: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    try:
        number = float(str(value).strip().rstrip("%")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise Invalid(f"{field} must be a number between 0 and 1") from None
    if 1.0 < number <= 100.0:  # "90" / "90%": a percentage, not a reason to lose the whole decision
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise Invalid(f"{field} must be between 0 and 1")
    return round(number, 3)


class Plan:
    """A validated decision: the normalized actions plus what the model said about them."""

    def __init__(self, actions: list[dict], confidence: float, reason: str):
        self.actions, self.confidence, self.reason = actions, confidence, reason

    @property
    def terminal(self) -> str | None:
        return self.actions[0]["action"] if self.actions and self.actions[0]["action"] in TERMINAL else None


def normalize_answer(answer):
    """Shapes a model commonly produces for the same meaning, made into {"actions": [...], ...}. Only the wrapper is
    fixed: a single action object, or a bare action object with its own confidence / reason. Content is never invented -
    a missing confidence is only defaulted for the actions that write nothing (NO_ACTION / REJECT / QUARANTINE)."""
    if not isinstance(answer, dict):
        return answer
    if "actions" not in answer and isinstance(answer.get("action"), str):
        answer = {"actions": [{k: v for k, v in answer.items() if k not in ("confidence",)}], "confidence": answer.get("confidence"), "reason": answer.get("reason", "")}
    elif isinstance(answer.get("actions"), dict):
        answer = {**answer, "actions": [answer["actions"]]}
    acts = answer.get("actions")
    if isinstance(acts, list) and acts and all(isinstance(a, dict) and a.get("action") in TERMINAL for a in acts) and answer.get("confidence") is None:
        answer = {**answer, "confidence": 0.8}
    return answer


def validate_answer(answer: dict, allowed_raw: set[str], store: Store, lenient_confidence: bool = False) -> Plan:
    """Turn a model answer into a Plan or raise Invalid (-> QUARANTINE). Existing ids are checked against the store."""
    answer = normalize_answer(answer)
    if not isinstance(answer, dict) or not isinstance(answer.get("actions"), list):
        raise Invalid("answer has no actions list")
    raw_actions = answer["actions"]
    if not 1 <= len(raw_actions) <= MAX_ACTIONS:
        raise Invalid(f"actions must contain 1..{MAX_ACTIONS} items")
    # The overall confidence is sometimes left out of an otherwise sound answer: the least sure of its actions stands in.
    stated = answer.get("confidence")
    if stated is None:
        own = [a.get("confidence") for a in raw_actions if isinstance(a, dict) and isinstance(a.get("confidence"), (int, float))]
        stated = min(own) if own else None
    confidence = _unit(stated, "confidence")
    reason = clean_text(answer.get("reason"), "reason", 600, required=False)
    kinds = [a.get("action") if isinstance(a, dict) else None for a in raw_actions]
    if any(k not in ALL_ACTIONS for k in kinds):
        raise Invalid(f"unknown action in {kinds}")
    if any(k in TERMINAL for k in kinds) and len(kinds) != 1:
        raise Invalid("QUARANTINE / REJECT / NO_ACTION must be the only action")
    if kinds[0] in TERMINAL:
        return Plan([{"action": kinds[0], "reason": reason}], confidence, reason)
    if confidence < MIN_CONFIDENCE and not lenient_confidence:
        raise Invalid(f"confidence {confidence} is below {MIN_CONFIDENCE}")
    refs: dict[str, str] = {}
    plan: list[dict] = []
    for raw in raw_actions:
        kind = raw["action"]
        handler = {"CREATE_EPISODE": _create_episode, "UPDATE_EPISODE": _update_episode, "MERGE_EPISODE": _merge_episode,
                   "CREATE_PATTERN": _create_pattern, "UPDATE_PATTERN": _update_pattern}[kind]
        plan.append(handler(raw, allowed_raw, store, refs))
    return Plan(plan, confidence, reason)


def _evidence(value, allowed_raw: set[str], field: str = "source_raw_ids", required: bool = True) -> list[str]:
    if not isinstance(value, list) or (required and not value):
        raise Invalid(f"{field} must be a non-empty list of RAW ids")
    ids = list(dict.fromkeys(str(v) for v in value))
    stray = [i for i in ids if i not in allowed_raw]
    if stray:
        raise Invalid(f"{field} cites RAW that was not provided: {stray[:3]}")
    return ids


def _episode_fields(raw: dict, allowed_raw: set[str], store: Store, require_all: bool) -> dict:
    out: dict = {}
    if require_all or "content" in raw:
        out["content"] = clean_text(raw.get("content"), "content", 600)
    if require_all or "time_start" in raw:
        out["time_start"] = normalize_time(raw.get("time_start"))
        out["time_end"] = normalize_time(raw.get("time_end") or raw.get("time_start"))
    elif "time_end" in raw:
        out["time_end"] = normalize_time(raw["time_end"])
    if "time_start" in out and out.get("time_end") and parse_time(out["time_end"]) < parse_time(out["time_start"]):
        raise Invalid("time_end is before time_start")
    for key, limit in (("entities", 12), ("topics", 8)):
        if require_all or key in raw:
            out[key] = _strings(raw.get(key), key, limit)
    if require_all or "state" in raw:
        out["state"] = clean_text(raw.get("state") or "", "state", 200, required=False)
    if require_all or "importance" in raw:
        out["importance"] = _unit(raw.get("importance"), "importance", 0.5)
    if require_all or "confidence" in raw:
        out["confidence"] = _unit(raw.get("confidence"), "confidence", 0.8)
    out.update(commitment_fields(raw))
    if "relations" in raw and raw["relations"]:
        rels = []
        for r in raw["relations"]:
            if not isinstance(r, dict) or r.get("type") not in RELATION_TYPES or r.get("target_kind") not in ("pattern", "episode"):
                raise Invalid(f"Episode relations must be one of {RELATION_TYPES} to an existing pattern / episode")
            if r["type"] in CAUSAL and r["target_kind"] != "episode":
                raise Invalid("a cause or an effect is an episode")
            target = str(r.get("target_id"))
            try:
                (store.episode if r["target_kind"] == "episode" else store.pattern)(target)
            except NotFound:
                raise Invalid(f"relation target {target} does not exist") from None
            rels.append({"type": r["type"], "target_kind": r["target_kind"], "target_id": target})
        out["relations"] = rels
    return out


# ------------------------------------------------------------ promises and plans (commitments.py reads them back)

KINDS = ("", "commitment", "vow", "lexicon", "ritual")
# Relations an answer may give a new Episode: loosely related, or cause and effect the RAW states outright.
CAUSAL = ("because_of", "led_to")
RELATION_TYPES = ("related_to", *CAUSAL)
# "我们的词": a meme / joke / dirty joke / pet name / nickname / catchphrase of theirs - tagged 梗·… or 昵称·…, one per word,
# never due, never sinking, and it does not fade (retrieval.py reads the tag).
LEXICON_PREFIXES = ("梗·", "昵称·")
RITUAL_PREFIX = "日常·"  # a ritual of theirs: one memory, however many times it happens
LEXICON_IMPORTANCE = 0.6
COMMIT_STATUSES = ("open", "done", "missed", "cancelled")
SUNK_IMPORTANCE = 0.15  # a commitment that is over: still findable when asked about, never brought up by itself
VOW_IMPORTANCE = 0.85
DEFAULT_DUE_DAYS = 3
BOTH = "两人"


def _owner(value: str) -> str | None:
    """Who made a promise, as the name the memory uses: either of them by name or role, or both."""
    v = value.strip().lower()
    if v in ("两人", "双方", "both"):
        return BOTH
    if identity.is_user(v):
        return identity.user()
    if identity.is_assistant(v):
        return identity.assistant()
    return None


def commitment_fields(raw: dict) -> dict:
    """kind / tag / owner / due_at / commit_status, when the answer or edit names them; anything else is Invalid."""
    out: dict = {}
    if "kind" in raw:
        kind = str(raw.get("kind") or "").strip().lower()
        if kind not in KINDS:
            raise Invalid(f"kind must be one of {KINDS}")
        out["kind"] = kind
    if "tag" in raw:
        out["tag"] = clean_text(raw.get("tag") or "", "tag", 60, required=False)
    if "owner" in raw:
        owner = str(raw.get("owner") or "").strip()
        if owner and _owner(owner) is None:
            raise Invalid(f"owner must be {identity.user()}, {identity.assistant()} or {BOTH}")
        out["owner"] = _owner(owner) if owner else ""
    if "due_at" in raw:
        out["due_at"] = normalize_time(raw["due_at"]) if raw.get("due_at") else None
    if "commit_status" in raw:
        status = str(raw.get("commit_status") or "").strip().lower()
        if status and status not in COMMIT_STATUSES:
            raise Invalid(f"commit_status must be one of {COMMIT_STATUSES}")
        out["commit_status"] = status
    return out


def new_commitment(doc: dict) -> dict:
    """A new Episode's commitment fields made whole: a daily commitment is open, tagged and due (3 days after it was made
    when nobody said when); a vow is never due and starts important."""
    kind = doc.get("kind") or ""
    if kind == "commitment":
        if not doc.get("tag"):
            day = parse_time(doc["time_start"]) + timedelta(hours=8)
            words = re.sub(r"[^\w]", "", doc.get("content") or "")[:8]
            doc["tag"] = f"日常承诺·{day.month}月{day.day}日·{words}"
        doc["commit_status"] = doc.get("commit_status") or "open"
        doc["due_at"] = doc.get("due_at") or normalize_time(parse_time(doc["time_start"]) + timedelta(days=DEFAULT_DUE_DAYS))
    elif kind == "vow":
        doc.update(commit_status="", due_at=None, importance=max(float(doc.get("importance", 0.5)), VOW_IMPORTANCE))
    elif kind == "lexicon":
        tag = (doc.get("tag") or "").strip()
        if not tag.startswith(LEXICON_PREFIXES):
            word = tag or (re.findall(r"[“「『\"]([^”」』\"]{1,20})[”」』\"]", doc.get("content") or "") or [re.sub(r"[^\w]", "", doc.get("content") or "")[:8]])[0]
            tag = f"梗·{word}"
        doc.update(tag=tag[:60], commit_status="", due_at=None, importance=max(float(doc.get("importance", 0.5)), LEXICON_IMPORTANCE))
    elif kind == "ritual":
        tag = (doc.get("tag") or "").strip()
        if not tag.startswith(RITUAL_PREFIX):
            tag = RITUAL_PREFIX + (tag or re.sub(r"[^\w]", "", doc.get("content") or "")[:6])
        doc.update(tag=tag[:40], commit_status="", due_at=None)
    else:
        doc.update(commit_status="", due_at=None)
    return doc


def settle_commitment(current: dict, patch: dict, when: str) -> dict:
    """A patch to an Episode, with what a change of kind or of commitment status implies: done / cancelled sinks it."""
    kind = patch.get("kind", current.get("kind") or "")
    if "kind" in patch and kind != (current.get("kind") or ""):
        merged = new_commitment({**current, **patch, "commit_status": patch.get("commit_status", "")})
        patch.update({k: merged[k] for k in ("commit_status", "due_at", "importance")})
        if kind == "commitment" and current.get("commit_status") in ("done", "cancelled"):
            patch["commit_status"] = current["commit_status"]
    status = patch.get("commit_status", current.get("commit_status") or "")
    if kind == "commitment" and status != (current.get("commit_status") or ""):
        if status in ("done", "cancelled"):
            patch.update(resolved_at=when, importance=min(float(patch.get("importance", current["importance"])), SUNK_IMPORTANCE))
        elif status == "open":
            patch.update(resolved_at=None, missed_told_at=None)
    return patch


def _attachments_for(source_ids: list[str], asked, store: Store) -> list[str]:
    allowed = {a["attachment_id"] for rid in source_ids for a in store.attachments_for_raw(rid)}
    ids = list(dict.fromkeys(str(x) for x in (asked or [])))
    if not set(ids) <= allowed:
        raise Invalid("attachment_ids must belong to the Episode's own RAW")
    return ids


def _active_episode(store: Store, ident: str) -> dict:
    try:
        ep = store.episode(ident)
    except NotFound:
        raise Invalid(f"unknown episode {ident}") from None
    if ep["status"] != "active":
        raise Invalid(f"episode {ident} is {ep['status']}")
    return ep


def _active_pattern(store: Store, ident: str) -> dict:
    try:
        p = store.pattern(ident)
    except NotFound:
        raise Invalid(f"unknown pattern {ident}") from None
    if p["status"] != "active":
        raise Invalid(f"pattern {ident} is {p['status']}")
    return p


def _create_episode(raw: dict, allowed_raw, store, refs) -> dict:
    ref = str(raw.get("ref") or "").strip()
    if not ref or ref in refs:
        raise Invalid("CREATE_EPISODE needs a unique ref")
    refs[ref] = "episode"
    fields = new_commitment(_episode_fields(raw, allowed_raw, store, True))
    sources = _evidence(raw.get("source_raw_ids"), allowed_raw)
    out = {"action": "CREATE_EPISODE", "ref": ref, **fields, "source_raw_ids": sources, "attachment_ids": _attachments_for(sources, raw.get("attachment_ids"), store)}
    proposal = _thread_proposal(raw.get("thread"), store)
    if proposal:
        out["thread"] = proposal
    return out


def _thread_proposal(raw, store) -> dict | None:
    """DeepSeek's story-thread proposal for a new Episode. A proposal that does not hold up (unknown thread, no earlier
    Episode to start one with) is dropped quietly: the Episode itself still stands; the thread is only a link."""
    if not isinstance(raw, dict):
        return None
    try:
        conf = _unit(raw.get("confidence", 0.5), "thread.confidence")
    except Invalid:
        conf = 0.5
    reason = clean_text(raw.get("reason") or "", "thread.reason", 200, required=False)
    tid = str(raw.get("thread_id") or "").strip()
    if tid:
        if not store.one("SELECT 1 FROM threads WHERE thread_id = ?", (tid,)):
            return None
        return {"thread_id": tid, "confidence": conf, "reason": reason, "aliases": [str(a) for a in raw.get("aliases") or []][:6],
                "over": raw.get("over") is True or str(raw.get("over")).lower() == "true"}
    title = clean_text(raw.get("new_title") or "", "thread.new_title", 40, required=False)
    others = [e["episode_id"] for e in store.episodes(status="active", ids=[str(i) for i in raw.get("with_episode_ids") or []][:6])]
    return ({"new_title": title, "with_episode_ids": others, "confidence": conf, "reason": reason, "aliases": [str(a) for a in raw.get("aliases") or []][:6],
             "over": raw.get("over") is True or str(raw.get("over")).lower() == "true"} if title and others else None)


def _update_episode(raw: dict, allowed_raw, store, refs) -> dict:
    target = str(raw.get("episode_id") or "")
    _active_episode(store, target)
    patch = _episode_fields(raw.get("patch") if isinstance(raw.get("patch"), dict) else raw, allowed_raw, store, False)
    add = _evidence(raw.get("source_raw_ids") or [], allowed_raw, required=False)
    if not patch and not add:
        raise Invalid("UPDATE_EPISODE changes nothing")
    return {"action": "UPDATE_EPISODE", "episode_id": target, "patch": patch, "add_source_raw_ids": add,
            "add_attachment_ids": raw.get("attachment_ids") or [], "reason": clean_text(raw.get("reason"), "reason", 300, required=False)}


def _merge_episode(raw: dict, allowed_raw, store, refs) -> dict:
    ids = list(dict.fromkeys(str(i) for i in (raw.get("episode_ids") or [])))
    if len(ids) < 2:
        raise Invalid("MERGE_EPISODE needs at least two episode_ids")
    for i in ids:
        _active_episode(store, i)
    ref = str(raw.get("ref") or "").strip()
    if not ref or ref in refs:
        raise Invalid("MERGE_EPISODE needs a unique ref")
    refs[ref] = "episode"
    merged = raw.get("merged") if isinstance(raw.get("merged"), dict) else raw
    fields = new_commitment(_episode_fields(merged, allowed_raw, store, True))
    return {"action": "MERGE_EPISODE", "ref": ref, "episode_ids": ids, **fields, "add_source_raw_ids": _evidence(raw.get("source_raw_ids") or [], allowed_raw, required=False)}


def _support_ids(values, store: Store, refs: dict, field: str, required: bool) -> list[str]:
    if not isinstance(values, list) or (required and not values):
        raise Invalid(f"{field} must be a non-empty list of episode ids / refs")
    out = []
    for v in values:
        v = str(v)
        if v in refs:
            if refs[v] != "episode":
                raise Invalid(f"{v} is not an episode ref")
            out.append(v)
        else:
            _active_episode(store, v)
            out.append(v)
    return list(dict.fromkeys(out))


def _create_pattern(raw: dict, allowed_raw, store, refs) -> dict:
    ref = str(raw.get("ref") or "").strip()
    if not ref or ref in refs:
        raise Invalid("CREATE_PATTERN needs a unique ref")
    supporting = _support_ids(raw.get("supporting_episode_ids") or raw.get("supporting"), store, refs, "supporting_episode_ids", True)
    refs[ref] = "pattern"
    out = {"action": "CREATE_PATTERN", "ref": ref, "title": clean_text(raw.get("title"), "title", 80), "topic": clean_text(raw.get("topic"), "topic", 60, required=False),
           "narrative": clean_text(raw.get("narrative"), "narrative", 1200), "current_state": clean_text(raw.get("current_state"), "current_state", 300),
           "entities": _strings(raw.get("entities"), "entities", 12), "confidence": _unit(raw.get("confidence"), "confidence", 0.8), "supporting_episode_ids": supporting}
    if raw.get("state_valid_from"):
        out["state_valid_from"] = normalize_time(raw["state_valid_from"])
    earlier = raw.get("earlier_states")
    if earlier:
        if not isinstance(earlier, list) or len(earlier) > 6:
            raise Invalid("earlier_states must be a list of at most 6 states")
        out["earlier_states"] = sorted(
            [{"state": clean_text(e.get("state") if isinstance(e, dict) else None, "earlier_states.state", 300), "valid_from": normalize_time(e.get("valid_from"))} for e in earlier],
            key=lambda e: e["valid_from"])
    return out


def _update_pattern(raw: dict, allowed_raw, store, refs) -> dict:
    target = str(raw.get("pattern_id") or "")
    _active_pattern(store, target)
    out: dict = {"action": "UPDATE_PATTERN", "pattern_id": target, "reason": clean_text(raw.get("reason"), "reason", 300, required=False),
                 "add_supporting": _support_ids(raw.get("add_supporting") or [], store, refs, "add_supporting", False)}
    for key, limit in (("title", 80), ("topic", 60), ("narrative", 1200)):
        if raw.get(key):
            out[key] = clean_text(raw[key], key, limit)
    if raw.get("entities"):
        out["entities"] = _strings(raw["entities"], "entities", 12)
    if raw.get("confidence") is not None:
        out["confidence"] = _unit(raw["confidence"], "confidence")
    ns = raw.get("new_state")
    if ns:
        if not isinstance(ns, dict):
            raise Invalid("new_state must be an object")
        out["new_state"] = {"state": clean_text(ns.get("state"), "new_state.state", 300)}
        if ns.get("valid_from"):
            out["new_state"]["valid_from"] = normalize_time(ns["valid_from"])
        if ns.get("correction") is True or str(ns.get("correction")).strip().lower() == "true":  # the current state was wrong all along, not something she changed
            out["new_state"]["correction"] = True
    if len(out) == 4 and not out["add_supporting"]:
        raise Invalid("UPDATE_PATTERN changes nothing")
    return out


# ------------------------------------------------------------ applying a plan


def _entity_id(name: str) -> str:
    return "entity:" + name


def _link_entities(store: Store, kind: str, ident: str, entities: list[str], decision_id: str | None) -> None:
    for name in entities:
        store.add_relation("involved_in" if kind == "episode" else "associated_with", "entity", _entity_id(name), kind, ident, decision_id=decision_id)


def _union(*lists) -> list:
    return list(dict.fromkeys(x for lst in lists for x in lst))


def apply_plan(store: Store, plan: Plan, candidate_id: str | None, decision_id: str, actor: str = "deepseek") -> dict:
    """Apply the writing actions of a validated plan in the caller's transaction. Returns what changed."""
    applied: dict = {"episodes_created": [], "episodes_updated": [], "episodes_merged": [], "patterns_created": [], "patterns_updated": [],
                     "state_changes": [], "refs": {}}
    refs: dict[str, str] = applied["refs"]

    def resolve(ident: str) -> str:
        return refs.get(ident, ident)

    # Tombstones: an Episode drawn only from RAW that a deleted Episode came from is what the user deleted, found again -
    # it is not created, and nothing else in the plan may lean on it. New evidence (any RAW not tombstoned) passes.
    tomb = store.tombstoned_raw_ids()
    skipped = {a["ref"] for a in plan.actions if a["action"] == "CREATE_EPISODE" and a["source_raw_ids"] and set(a["source_raw_ids"]) <= tomb}
    if skipped:
        applied["tombstoned"] = sorted(skipped)

    with store.tx():
        for a in plan.actions:
            kind = a["action"]
            if a.get("ref") in skipped:
                continue
            if skipped and kind in ("CREATE_PATTERN", "UPDATE_PATTERN"):
                key = "supporting_episode_ids" if kind == "CREATE_PATTERN" else "add_supporting"
                a = {**a, key: [i for i in a.get(key, []) if i not in skipped]}
                if kind == "CREATE_PATTERN" and not a[key]:
                    skipped.add(a["ref"])
                    applied["tombstoned"].append(a["ref"])
                    continue
            if kind == "CREATE_EPISODE":
                dup = next((e for e in store.episodes(status="active") if set(e["source_raw_ids"]) == set(a["source_raw_ids"]) and e["content"] == a["content"]), None)
                if dup:
                    refs[a["ref"]] = dup["episode_id"]
                    continue
                # The same word of theirs again (same 梗· / 昵称· tag): one memory, the newer telling, all the evidence.
                word = a.get("kind") == "lexicon" and next((e for e in store.episodes(status="active") if e.get("kind") == "lexicon"
                                                            and e.get("tag") and e.get("tag") == a.get("tag")), None)
                if word:
                    store.update_episode(word["episode_id"], {"content": a["content"], "source_raw_ids": _union(word["source_raw_ids"], a["source_raw_ids"])},
                                         actor=actor, reason="same word of theirs again", decision_id=decision_id)
                    refs[a["ref"]] = word["episode_id"]
                    applied["episodes_updated"].append(word["episode_id"])
                    continue
                # The same ritual again (same 日常· tag): one memory, its evidence grows, the last time moves; the content stays.
                ritual = a.get("kind") == "ritual" and next((e for e in store.episodes(status="active") if e.get("kind") == "ritual"
                                                              and e.get("tag") == a.get("tag")), None)
                if ritual:
                    store.update_episode(ritual["episode_id"], {"source_raw_ids": _union(ritual["source_raw_ids"], a["source_raw_ids"]),
                                                                "time_end": max(ritual["time_end"], a["time_end"])},
                                         actor=actor, reason="same ritual again", decision_id=decision_id)
                    refs[a["ref"]] = ritual["episode_id"]
                    applied["episodes_updated"].append(ritual["episode_id"])
                    continue
                # The same promise said again (same tag, still open): one commitment, more evidence - never a second copy.
                same = a.get("kind") == "commitment" and next((e for e in store.episodes(status="active") if e.get("kind") == "commitment" and e.get("tag") == a["tag"]
                                                               and e.get("commit_status") in ("open", "missed")), None)
                if same:
                    patch = settle_commitment(same, {k: a[k] for k in ("content", "state", "due_at") if a.get(k)} | {"source_raw_ids": _union(same["source_raw_ids"], a["source_raw_ids"])}, now_iso())
                    store.update_episode(same["episode_id"], patch, actor=actor, reason="same commitment again", decision_id=decision_id)
                    refs[a["ref"]] = same["episode_id"]
                    applied["episodes_updated"].append(same["episode_id"])
                    continue
                doc = {k: a[k] for k in ("content", "time_start", "time_end", "entities", "topics", "state", "importance", "confidence", "source_raw_ids", "attachment_ids",
                                         "kind", "tag", "owner", "due_at", "commit_status") if k in a}
                ep = store.insert_episode({**doc, "relations": a.get("relations", []), "origin": "pipeline", "candidate_id": candidate_id, "decision_id": decision_id}, actor=actor, reason="CREATE_EPISODE")
                refs[a["ref"]] = ep["episode_id"]
                applied["episodes_created"].append(ep["episode_id"])
                _link_entities(store, "episode", ep["episode_id"], ep["entities"], decision_id)
                if a.get("thread"):
                    linked = threads_module.attach_proposal(store, ep["episode_id"], a["thread"], decision_id)
                    if linked:
                        applied.setdefault("threads", []).append({"episode_id": ep["episode_id"], **linked})
                for r in a.get("relations", []):
                    # because_of: this episode happened because of the target; led_to: it brought the target about.
                    if r["type"] == "led_to":
                        store.add_relation("because_of", "episode", r["target_id"], "episode", ep["episode_id"], decision_id=decision_id)
                    else:
                        store.add_relation(r["type"], "episode", ep["episode_id"], r["target_kind"], r["target_id"], decision_id=decision_id)
            elif kind == "UPDATE_EPISODE":
                cur = store.episode(a["episode_id"])
                patch = dict(a["patch"])
                sources = _union(cur["source_raw_ids"], a["add_source_raw_ids"])
                if sources != cur["source_raw_ids"]:
                    patch["source_raw_ids"] = sources
                ask = list(cur["attachment_ids"]) + [str(x) for x in a.get("add_attachment_ids") or []]
                patch["attachment_ids"] = _attachments_for(sources, ask, store)
                if "entities" in patch:
                    patch["entities"] = _union(cur["entities"], patch["entities"])
                if "time_start" in patch and "time_end" not in patch:
                    patch["time_end"] = max(patch["time_start"], cur["time_end"])
                patch = settle_commitment(cur, patch, now_iso())
                new = store.update_episode(a["episode_id"], patch, actor=actor, reason=a.get("reason") or "UPDATE_EPISODE", decision_id=decision_id)
                applied["episodes_updated"].append(a["episode_id"])
                _link_entities(store, "episode", a["episode_id"], new["entities"], decision_id)
            elif kind == "MERGE_EPISODE":
                olds = [store.episode(i) for i in a["episode_ids"]]
                sources = _union(*[e["source_raw_ids"] for e in olds], a["add_source_raw_ids"], [])
                attachments = _union(*[e["attachment_ids"] for e in olds])
                doc = {k: a[k] for k in ("content", "time_start", "time_end", "entities", "topics", "state", "importance", "confidence",
                                         "kind", "tag", "owner", "due_at", "commit_status") if k in a}
                doc["time_start"] = min(doc["time_start"], *[e["time_start"] for e in olds])
                doc["time_end"] = max(doc["time_end"], *[e["time_end"] for e in olds])
                doc["entities"] = _union(doc["entities"], *[e["entities"] for e in olds])[:12]
                merged = store.insert_episode({**doc, "source_raw_ids": sources, "attachment_ids": attachments, "origin": "pipeline", "candidate_id": candidate_id, "decision_id": decision_id}, actor=actor, reason="MERGE_EPISODE")
                refs[a["ref"]] = merged["episode_id"]
                applied["episodes_merged"].append({"into": merged["episode_id"], "from": a["episode_ids"]})
                for old in olds:
                    store.update_episode(old["episode_id"], {"status": "merged", "superseded_by": merged["episode_id"]}, actor=actor, reason=f"merged into {merged['episode_id']}", decision_id=decision_id)
                for pat in store.patterns(status="active"):
                    if set(pat["supporting_episode_ids"]) & set(a["episode_ids"]):
                        support = _union([merged["episode_id"] if e in a["episode_ids"] else e for e in pat["supporting_episode_ids"]])
                        states = copy.deepcopy(pat["states"])
                        for st in states:
                            st["episode_ids"] = _union([merged["episode_id"] if e in a["episode_ids"] else e for e in st["episode_ids"]])
                        store.update_pattern(pat["pattern_id"], {"supporting_episode_ids": support, "states": states}, actor=actor, reason="episode merge", decision_id=decision_id)
                        store.add_relation("supports", "episode", merged["episode_id"], "pattern", pat["pattern_id"], decision_id=decision_id)
                        store.add_relation("derived_from", "pattern", pat["pattern_id"], "episode", merged["episode_id"], decision_id=decision_id)
                _link_entities(store, "episode", merged["episode_id"], merged["entities"], decision_id)
            elif kind == "CREATE_PATTERN":
                support = _union([resolve(i) for i in a["supporting_episode_ids"]])
                eps = [store.episode(i) for i in support]
                if any(e["status"] != "active" for e in eps):
                    raise Invalid("a supporting episode is not active")
                first = min(e["time_start"] for e in eps)
                since = a.get("state_valid_from") or max(e["time_end"] for e in eps)
                states = _initial_states(a, eps, since)
                doc = {"title": a["title"], "topic": a["topic"], "narrative": a["narrative"], "current_state": a["current_state"], "current_state_since": since,
                       "states": states,
                       "supporting_episode_ids": support, "entities": _union(a["entities"], *[e["entities"] for e in eps])[:12], "valid_from": first, "valid_to": None,
                       "confidence": a["confidence"], "candidate_id": candidate_id, "decision_id": decision_id}
                pat = store.insert_pattern(doc, actor=actor, reason="CREATE_PATTERN")
                for older, newer in zip(states, states[1:]):
                    store.add_relation("supersedes", "state", f"{pat['pattern_id']}#{newer['state_id']}", "state", f"{pat['pattern_id']}#{older['state_id']}", decision_id=decision_id)
                refs[a["ref"]] = pat["pattern_id"]
                applied["patterns_created"].append(pat["pattern_id"])
                for e in eps:
                    store.add_relation("supports", "episode", e["episode_id"], "pattern", pat["pattern_id"], decision_id=decision_id)
                    store.add_relation("derived_from", "pattern", pat["pattern_id"], "episode", e["episode_id"], decision_id=decision_id)
                _link_entities(store, "pattern", pat["pattern_id"], pat["entities"], decision_id)
            elif kind == "UPDATE_PATTERN":
                _apply_pattern_update(store, a, resolve, actor, decision_id, applied)
    return applied


def _initial_states(a: dict, eps: list[dict], since: str) -> list[dict]:
    """The state history a new Pattern starts with: the model's `earlier_states`, else the earlier, differently-worded states
    its supporting Episodes themselves record (each Episode names the state it shows). The current state is always last and open."""
    current = a["current_state"]
    spans: list[dict] = []
    if a.get("earlier_states"):
        # An earlier state dated at or after the current one is a slip in the dates, not a reason to lose the whole
        # decision (the memory evaluation lost a change of taste this way): that entry is dropped, the rest stands.
        spans = [{"state": e["state"], "valid_from": e["valid_from"]} for e in a["earlier_states"] if e["state"] != current and e["valid_from"] < since]
    else:
        for e in sorted(eps, key=lambda e: e["time_start"]):
            state = (e.get("state") or "").strip()
            if not state or state == current or e["time_end"] >= since:
                continue
            if spans and spans[-1]["state"] == state:
                continue
            spans.append({"state": state, "valid_from": e["time_start"]})
    states = []
    for i, sp in enumerate(spans):
        end = spans[i + 1]["valid_from"] if i + 1 < len(spans) else since
        states.append({"state_id": f"s{i + 1}", "state": sp["state"], "valid_from": sp["valid_from"], "valid_to": end, "episode_ids": []})
    states.append({"state_id": f"s{len(states) + 1}", "state": current, "valid_from": since, "valid_to": None, "episode_ids": []})
    for e in eps:  # each Episode belongs to the span its time falls in
        home = next((st for st in reversed(states) if e["time_start"] >= st["valid_from"]), states[0])
        home["episode_ids"].append(e["episode_id"])
    return states


def _apply_pattern_update(store: Store, a: dict, resolve, actor: str, decision_id: str, applied: dict) -> None:
    pat = store.pattern(a["pattern_id"])
    if pat["status"] != "active":
        raise Invalid(f"pattern {pat['pattern_id']} is {pat['status']}")
    patch: dict = {}
    add = _union([resolve(i) for i in a["add_supporting"]])
    eps = [store.episode(i) for i in add]
    if any(e["status"] != "active" for e in eps):
        raise Invalid("a supporting episode is not active")
    support = _union(pat["supporting_episode_ids"], add)
    if support != pat["supporting_episode_ids"]:
        patch["supporting_episode_ids"] = support
    states = copy.deepcopy(pat["states"])
    ns = a.get("new_state")
    before_state = pat["current_state"]
    if ns and ns.get("correction"):
        # Refutation, not supersession: the current state was recorded wrong. It is corrected in place, so the wrong
        # text never becomes "以前" (history); it is kept on the state as refuted, with when and by which decision.
        current = next(s for s in states if not s.get("valid_to"))
        current["episode_ids"] = _union(current["episode_ids"], add)
        if ns["state"].strip().lower() != current["state"].strip().lower():
            current.setdefault("refuted", []).append({"state": current["state"], "at": now_iso(), "decision_id": decision_id})
            current["state"] = ns["state"]
            patch["current_state"] = ns["state"]
            applied["state_changes"].append({"pattern_id": pat["pattern_id"], "title": pat["title"], "before": before_state, "after": ns["state"], "correction": True})
        patch["states"] = states
    elif ns:
        latest = max([e["time_end"] for e in eps], default=None)
        valid_from = ns.get("valid_from") or latest or pat["current_state_since"]
        current = next(s for s in states if not s.get("valid_to"))
        if ns["state"].strip().lower() == current["state"].strip().lower():
            current["episode_ids"] = _union(current["episode_ids"], add)  # same state again: more evidence, no change of state
        elif parse_time(valid_from) >= parse_time(current["valid_from"]):
            if parse_time(valid_from) == parse_time(current["valid_from"]):
                raise Invalid("temporal conflict: the new state starts exactly when the current one does")
            current["valid_to"] = valid_from  # the old state is closed, not overwritten
            new_id = f"s{len(states) + 1}"
            states.append({"state_id": new_id, "state": ns["state"], "valid_from": valid_from, "valid_to": None, "episode_ids": add})
            patch.update(current_state=ns["state"], current_state_since=valid_from)
            store.add_relation("supersedes", "state", f"{pat['pattern_id']}#{new_id}", "state", f"{pat['pattern_id']}#{current['state_id']}", decision_id=decision_id)
            applied["state_changes"].append({"pattern_id": pat["pattern_id"], "title": pat["title"], "before": before_state, "after": ns["state"], "valid_from": valid_from})
        else:
            # Older evidence than the current state: it is history, inserted in order; the current state stands.
            later = min((s for s in states if parse_time(s["valid_from"]) > parse_time(valid_from)), key=lambda s: parse_time(s["valid_from"]), default=None)
            earlier = [s for s in states if parse_time(s["valid_from"]) <= parse_time(valid_from)]
            if earlier and earlier[-1].get("valid_to") and parse_time(earlier[-1]["valid_to"]) > parse_time(valid_from):
                raise Invalid("temporal conflict: the back-dated state overlaps an existing one")
            new_id = f"s{len(states) + 1}"
            entry = {"state_id": new_id, "state": ns["state"], "valid_from": valid_from, "valid_to": later["valid_from"] if later else None, "episode_ids": add}
            states.append(entry)
            states.sort(key=lambda s: parse_time(s["valid_from"]))
            if later:
                store.add_relation("supersedes", "state", f"{pat['pattern_id']}#{later['state_id']}", "state", f"{pat['pattern_id']}#{new_id}", decision_id=decision_id)
        patch["states"] = states
    elif add:
        current = next(s for s in states if not s.get("valid_to"))
        current["episode_ids"] = _union(current["episode_ids"], add)
        patch["states"] = states
    for key in ("title", "topic", "narrative", "confidence"):
        if key in a:
            patch[key] = a[key]
    if "entities" in a:
        patch["entities"] = _union(pat["entities"], a["entities"], *[e["entities"] for e in eps])[:12]
    elif eps:
        patch["entities"] = _union(pat["entities"], *[e["entities"] for e in eps])[:12]
    starts = [e["time_start"] for e in eps] + [pat["valid_from"]] if pat["valid_from"] else [e["time_start"] for e in eps]
    if starts:
        patch["valid_from"] = min(starts)
    if not patch:
        raise Invalid("UPDATE_PATTERN changes nothing")
    updated = store.update_pattern(pat["pattern_id"], patch, actor=actor, reason=a.get("reason") or "UPDATE_PATTERN", decision_id=decision_id)
    applied["patterns_updated"].append(pat["pattern_id"])
    for e in eps:
        store.add_relation("supports", "episode", e["episode_id"], "pattern", pat["pattern_id"], decision_id=decision_id)
        store.add_relation("derived_from", "pattern", pat["pattern_id"], "episode", e["episode_id"], decision_id=decision_id)
    _link_entities(store, "pattern", pat["pattern_id"], updated["entities"], decision_id)
