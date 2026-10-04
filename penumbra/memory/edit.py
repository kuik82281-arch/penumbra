"""the user's hand on the memory: corrections, maintenance and review of QUARANTINE. Every change is audited and versioned;
RAW provenance only ever grows, and nothing here can delete a RAW or an Episode's history."""
from __future__ import annotations

import copy
import json

from .. import identity
from ..errors import Invalid, NotFound
from .actions import apply_plan, clean_text, commitment_fields, settle_commitment, validate_answer, _strings, _unit
from .store import dumps, mint, normalize_time, now_iso


def _actor(body: dict) -> str:
    if not identity.is_user(body.get("actor") or identity.USER_ACTOR):
        raise Invalid("manual edits are the user's")
    return identity.USER_ACTOR


def attach_episode(store, pattern_id: str, episode_id: str, actor: str) -> dict:
    """The Episode supports the Pattern (added to the open state's evidence, relations recorded)."""
    with store.tx():
        pattern = store.pattern(pattern_id)
        episode = store.episode(episode_id)
        if pattern["status"] != "active" or episode["status"] != "active":
            raise Invalid("an active Pattern and Episode are required")
        if episode_id in pattern["supporting_episode_ids"]:
            return pattern
        states = copy.deepcopy(pattern["states"])
        current = next(s for s in states if not s.get("valid_to"))
        current["episode_ids"] = list(dict.fromkeys([*current["episode_ids"], episode_id]))
        updated = store.update_pattern(pattern_id, {"supporting_episode_ids": [*pattern["supporting_episode_ids"], episode_id], "states": states,
                                                    "valid_from": min(pattern["valid_from"] or episode["time_start"], episode["time_start"])}, actor=actor, reason=f"attach {episode_id}")
        store.add_relation("supports", "episode", episode_id, "pattern", pattern_id)
        store.add_relation("derived_from", "pattern", pattern_id, "episode", episode_id)
        return updated


def detach_episode(store, pattern_id: str, episode_id: str, actor: str) -> None:
    with store.tx():
        pattern = store.pattern(pattern_id)
        if episode_id not in pattern["supporting_episode_ids"]:
            return
        if len(pattern["supporting_episode_ids"]) == 1:
            raise Invalid("a Pattern needs at least one supporting Episode: archive the Pattern instead")
        states = copy.deepcopy(pattern["states"])
        for s in states:
            s["episode_ids"] = [e for e in s["episode_ids"] if e != episode_id]
        store.update_pattern(pattern_id, {"supporting_episode_ids": [e for e in pattern["supporting_episode_ids"] if e != episode_id], "states": states}, actor=actor, reason=f"detach {episode_id}")
        for r in store.relations("episode", episode_id):
            if r["type"] in ("supports", "derived_from") and pattern_id in (r["src_id"], r["dst_id"]):
                store.archive_relation(r["relation_id"])


def edit(core, body: dict) -> dict:
    actor = _actor(body)
    store = core.store
    action, ident, patch = str(body.get("action") or ""), str(body.get("id") or ""), body.get("patch") if isinstance(body.get("patch"), dict) else {}
    with store.tx():
        before: dict = {}
        if action == "pattern.update":
            pattern = store.pattern(ident)
            before = {k: pattern[k] for k in ("title", "topic", "narrative", "current_state", "entities")}
            change: dict = {}
            for key, limit in (("title", 80), ("topic", 60), ("narrative", 1200)):
                if key in patch:
                    change[key] = clean_text(patch[key], key, limit)
            if "entities" in patch:
                change["entities"] = _strings(patch["entities"], "entities")
            if "current_state" in patch:
                state = clean_text(patch["current_state"], "current_state", 300)
                states = copy.deepcopy(pattern["states"])
                next(s for s in states if not s.get("valid_to"))["state"] = state  # a correction of the open state, not a new state
                change.update(current_state=state, states=states)
            store.update_pattern(ident, change, actor=actor, reason="pattern.update")
        elif action in ("pattern.archive", "pattern.delete"):
            # the user's 删除 (the old 归档): the Pattern is gone for good; its Episodes and all RAW stay.
            store.delete_pattern(ident)
        elif action == "pattern.merge":
            source, target = store.pattern(ident), store.pattern(str(patch.get("target_pattern_id") or ""))
            if source["status"] != "active" or target["status"] != "active" or source["pattern_id"] == target["pattern_id"]:
                raise Invalid("two different active Patterns are required")
            before = {"source": source["pattern_id"], "target": target["pattern_id"]}
            states = sorted([*target["states"], *[dict(s, state_id=f"m{i}_{s['state_id']}", valid_to=s.get("valid_to") or source["current_state_since"]) for i, s in enumerate(source["states"])]],
                            key=lambda s: s["valid_from"])
            support = list(dict.fromkeys([*target["supporting_episode_ids"], *source["supporting_episode_ids"]]))
            store.update_pattern(target["pattern_id"], {"supporting_episode_ids": support, "states": states, "entities": list(dict.fromkeys([*target["entities"], *source["entities"]]))[:12],
                                                        "valid_from": min(x for x in (target["valid_from"], source["valid_from"]) if x),
                                                        "narrative": clean_text(patch.get("narrative") or target["narrative"], "narrative", 1200)}, actor=actor, reason=f"merge {ident}")
            store.update_pattern(ident, {"status": "merged", "merged_into": target["pattern_id"]}, actor=actor, reason="merged")
            for eid in source["supporting_episode_ids"]:
                store.add_relation("supports", "episode", eid, "pattern", target["pattern_id"])
                store.add_relation("derived_from", "pattern", target["pattern_id"], "episode", eid)
        elif action in ("episode.archive", "episode.delete"):
            # Gone for good, RAW untouched: Patterns it supported let go of it, and a Pattern left with nothing goes too.
            store.episode(ident)
            for p in store.patterns_supported_by(ident):
                if p["supporting_episode_ids"] == [ident]:
                    store.delete_pattern(p["pattern_id"])
                else:
                    detach_episode(store, p["pattern_id"], ident, actor)
            store.delete_episode(ident)
        elif action == "episode.update":
            episode = store.episode(ident)
            before = {k: episode.get(k) for k in ("content", "time_start", "time_end", "state", "status", "kind", "tag", "commit_status")}
            change = {}
            if "content" in patch:
                change["content"] = clean_text(patch["content"], "content", 600)
            for key in ("time_start", "time_end"):
                if key in patch:
                    change[key] = normalize_time(patch[key])
            for key, limit in (("entities", 12), ("topics", 8)):
                if key in patch:
                    change[key] = _strings(patch[key], key, limit)
            if "state" in patch:
                change["state"] = clean_text(patch["state"], "state", 200, required=False)
            if "excerpt" in patch:
                change["excerpt"] = clean_text(patch["excerpt"], "excerpt", 120, required=False)
            if "importance" in patch:
                change["importance"] = _unit(patch["importance"], "importance")
            if "source_raw_ids" in patch:
                core.raw([str(i) for i in patch["source_raw_ids"]])  # must exist
                change["source_raw_ids"] = list(dict.fromkeys([*episode["source_raw_ids"], *[str(i) for i in patch["source_raw_ids"]]]))
            # the user's call on a promise: 设为永久约定 / 设为日常承诺 / 普通经历, 已完成 / 取消 / 重新打开.
            change.update(commitment_fields(patch))
            change = settle_commitment(episode, change, now_iso())
            store.update_episode(ident, change, actor=actor, reason="episode.update")
        elif action == "episode.add":
            ids = [str(i) for i in patch.get("source_raw_ids") or []]
            core.raw(ids)
            ep = store.insert_episode({"content": clean_text(patch.get("content"), "content", 600), "time_start": normalize_time(patch.get("time_start")),
                                       "entities": _strings(patch.get("entities"), "entities"), "topics": _strings(patch.get("topics"), "topics", 8),
                                       "importance": float(patch.get("importance", 0.6)), "confidence": 1.0, "source_raw_ids": ids, "origin": "manual"}, actor=actor, reason="episode.add")
            ident = ep["episode_id"]
            if patch.get("pattern_id"):
                attach_episode(store, str(patch["pattern_id"]), ident, actor)
        elif action == "episode.move":
            target = str(patch.get("pattern_id") or "")
            for p in store.patterns_supported_by(ident):
                if p["pattern_id"] != target:
                    detach_episode(store, p["pattern_id"], ident, actor)
            if target:
                attach_episode(store, target, ident, actor)
        elif action in ("relation.add", "relation.delete"):
            if action == "relation.add":
                src, dst = str(patch.get("src_pattern_id") or ""), str(patch.get("dst_pattern_id") or "")
                store.pattern(src), store.pattern(dst)
                if src == dst:
                    raise Invalid("a Pattern cannot relate to itself")
                store.add_relation("related_to", "pattern", src, "pattern", dst)
            else:
                store.archive_relation(ident)
        elif action.startswith("attachment."):
            _attachment_edit(core, action, ident, patch)
        else:
            raise Invalid("Unknown edit action")
        store.audit(action, ident, before=before, after=patch, actor=actor)
        store.kv_set("corrections", (store.kv_get("corrections", []) + [{"at": now_iso(), "action": action, "ref": ident, "type": body.get("correction_type", "other")}])[-500:])
    return core.snapshot()


def _attachment_edit(core, action: str, ident: str, patch: dict) -> None:
    store = core.store
    attachment = store.attachment(ident)
    if action == "attachment.update":
        for key, limit in (("caption", 500), ("searchable_text", 2000)):
            if key in patch:
                attachment[key] = str(patch[key]).strip()[:limit]
        if "metadata" in patch:
            if not isinstance(patch["metadata"], dict):
                raise Invalid("metadata must be an object")
            attachment["metadata"] = copy.deepcopy(patch["metadata"])
        store.upsert_attachment(attachment)
    elif action == "attachment.archive":
        attachment["status"] = "archived"
        store.upsert_attachment(attachment)
        for e in store.episodes():
            if ident in e["attachment_ids"]:
                store.update_episode(e["episode_id"], {"attachment_ids": [x for x in e["attachment_ids"] if x != ident]}, actor=identity.USER_ACTOR, reason="attachment archived")
    else:
        source_id, target_id = str(patch.get("episode_id") or ""), str(patch.get("target_episode_id") or "")
        if action in ("attachment.unlink", "attachment.move"):
            source = store.episode(source_id)
            store.update_episode(source_id, {"attachment_ids": [x for x in source["attachment_ids"] if x != ident]}, actor=identity.USER_ACTOR, reason=action)
        if action in ("attachment.link", "attachment.move"):
            dest = store.episode(target_id if action == "attachment.move" else source_id)
            if attachment["raw_id"] not in dest["source_raw_ids"]:
                raise Invalid("the attachment's RAW must be evidence of the Episode")
            store.update_episode(dest["episode_id"], {"attachment_ids": list(dict.fromkeys([*dest["attachment_ids"], ident]))}, actor=identity.USER_ACTOR, reason=action)


# ------------------------------------------------------------ review of STAGING / QUARANTINE / FAILED candidates


def _candidate(core, cid: str):
    row = core.store.one("SELECT * FROM candidates WHERE candidate_id = ?", (cid,))
    if row is None:
        raise NotFound(f"no candidate {cid}")
    return row


def requeue(core, cid: str, body: dict) -> dict:
    """Back to STAGING for another DeepSeek verification (after a failure, or a fixed API key)."""
    _actor(body)
    row = _candidate(core, cid)
    if row["status"] not in ("FAILED", "QUARANTINE", "REJECTED", "NO_ACTION"):
        raise Invalid(f"a {row['status']} candidate cannot be requeued")
    with core.store.tx() as db:
        db.execute("UPDATE candidates SET status='STAGING', attempts=0, last_error=NULL, next_attempt_at=NULL, decision_id=NULL, updated_at=? WHERE candidate_id=?", (now_iso(), cid))
    core.store.audit("candidate.requeued", cid, actor=identity.USER_ACTOR)
    return core.snapshot()


def review(core, cid: str, body: dict) -> dict:
    """the user resolves a candidate herself: reject it, or apply (edited) actions after reading the RAW."""
    _actor(body)
    row = _candidate(core, cid)
    if row["status"] not in ("STAGING", "QUARANTINE", "FAILED"):
        raise Invalid("only STAGING / QUARANTINE / FAILED candidates can be reviewed")
    store = core.store
    decision_id = mint("decision")
    verdict = body.get("decision")
    if verdict == "reject":
        with store.tx() as db:
            db.execute("INSERT INTO decisions (decision_id, candidate_id, provider, model, prompt_version, at, status, reason, actor) VALUES (?,?,?,?,?,?,?,?,?)",
                       (decision_id, cid, identity.USER_ACTOR, "manual", "review", now_iso(), "REJECTED", str(body.get("reason") or "")[:400], identity.USER_ACTOR))
            db.execute("UPDATE candidates SET status='REJECTED', decision_id=?, updated_at=? WHERE candidate_id=?", (decision_id, now_iso(), cid))
        store.audit("candidate.rejected", cid, actor=identity.USER_ACTOR, decision_id=decision_id)
        return core.snapshot()
    if verdict != "apply":
        raise Invalid("decision must be reject or apply")
    allowed = set(json.loads(row["raw_ids"])) | set(json.loads(row["context_raw_ids"]))
    actions = body.get("actions")
    if actions is None and row["decision_id"]:
        actions = core.decision(row["decision_id"])["answer"].get("actions")
    plan = validate_answer({"actions": actions, "confidence": 1.0, "reason": str(body.get("reason") or "reviewed by the user")}, allowed, store, lenient_confidence=True)
    with store.tx() as db:
        applied = apply_plan(store, plan, cid, decision_id, actor=identity.USER_ACTOR) if not plan.terminal else {}
        status = "APPLIED" if not plan.terminal else {"NO_ACTION": "NO_ACTION", "REJECT": "REJECTED", "QUARANTINE": "QUARANTINE"}[plan.terminal]
        db.execute("INSERT INTO decisions (decision_id, candidate_id, provider, model, prompt_version, at, status, confidence, reason, actions, applied, actor) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (decision_id, cid, identity.USER_ACTOR, "manual", "review", now_iso(), status, 1.0, plan.reason, dumps(plan.actions), dumps(applied), identity.USER_ACTOR))
        db.execute("UPDATE candidates SET status=?, decision_id=?, last_error=NULL, updated_at=? WHERE candidate_id=?", (status, decision_id, now_iso(), cid))
    store.audit(f"candidate.{status.lower()}", cid, after={"actions": plan.actions}, actor=identity.USER_ACTOR, decision_id=decision_id)
    return core.snapshot()
