"""WRITE: RAW -> Candidate Discovery -> STAGING -> DeepSeek Verification -> Episode -> Pattern.

One run:
  1. resume: candidates already in STAGING whose retry time has come (DeepSeek was unavailable / failed before)
  2. new RAW (everything not yet in a window) is cut into windows of whole messages, per conversation, at silences
  3. each window goes through Candidate Discovery (Ollama; deterministic fallback) and its candidates are STAGED in one
     transaction together with the window's checkpoint - a crash leaves either nothing or a complete window
  4. each STAGING candidate goes to DeepSeek with its RAW (+ nearby existing Episodes / Patterns); the validated decision
     is applied atomically with the candidate's new status and the decision record

Segmentation (PENUMBRA_SEGMENTATION, default "session"):
  session    a conversation is cut where nobody wrote for SESSION_GAP_MINUTES; a segment is only taken once it has been
             quiet that long (an ongoing chat waits). The whole segment - one topic from start to pause - is ONE
             candidate for DeepSeek, which records the few things in it worth keeping, each whole. The end of the
             previous segment rides along as context_RAW, so a talk picked up hours later is understood without being
             recorded twice. Very long segments are split at their longest pause. No discovery model is involved.
  discovery  the earlier way: small windows, Ollama points at spans inside them, each span is a candidate.

Nothing here can loop on the same input: a window is checkpointed the moment its candidates are staged, discovery
always returns something (Ollama, or the deterministic fallback), and a candidate that DeepSeek cannot be reached for is
retried with exponential backoff and a cap, then parked as FAILED for the Studio - never retried every few minutes forever.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from .. import identity
from ..errors import Invalid, NotFound, WorkerError
from . import commitments
from .actions import apply_plan, validate_answer
from .discovery import Discovery
from .quotes import plan_texts, unverified_quotes
from .store import Store, dumps, mint, now_iso
from . import names as names_module
from . import threads as threads_module
from .verification import PROMPT_VERSION, Verifier, build_payload

WINDOW_CHARS = 2600
# Session segmentation
SESSION_GAP_MINUTES = 30
SESSION_MAX_MSGS = 80
SESSION_MAX_CHARS = 16000
LINK_TAIL_MSGS = 6
LINK_LOOKBACK_HOURS = 24
LOCAL_OFFSET = timezone(timedelta(hours=8))
WINDOW_MSGS = 24
WINDOW_GAP_HOURS = 3
CONTEXT_MSGS = 3
MAX_RUN_SECONDS = 1500
MAX_VERIFY_ATTEMPTS = 6
BACKOFF_BASE_S = 300
BACKOFF_CAP_S = 6 * 3600
NOTE_SOURCE = "manual_note"

LAST_ACTIVITY_KEY = "last_activity_at"  # kv key: the one place the time of the last new RAW is kept
DEFAULT_SETTINGS = {"idle_minutes": 30, "night_start": 22, "night_end": 6, "morning_hour": 8, "utc_offset": 8, "enabled": True}
SETTING_BOUNDS = {"idle_minutes": (0, 1440), "night_start": (0, 23), "night_end": (0, 23), "morning_hour": (0, 23), "utc_offset": (-12, 14)}


def _later(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")


def _due(stamp: str | None) -> bool:
    return not stamp or stamp <= now_iso()



def _time_of(raw: list[dict]) -> str | None:
    """When a segment happened (its last message): the threads that moved shortly before it are shown to DeepSeek."""
    stamps = [r.get("createdAt") for r in raw if r.get("createdAt")]
    return max(stamps) if stamps else None

class Pipeline:
    def __init__(self, core):
        self.core = core
        self.store: Store = core.store
        self.discovery: Discovery = core.discovery
        self.verifier: Verifier = core.verifier
        self.work_lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.segmentation = os.environ.get("PENUMBRA_SEGMENTATION", "session").strip() or "session"

    # ------------------------------------------------------------ recovery

    def recover(self) -> dict:
        """After a restart: an interrupted run is marked, and a window that never reached STAGED is forgotten so its RAW is read again."""
        with self.store.tx() as db:
            runs = db.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (now_iso(),)).rowcount
            windows = db.execute("DELETE FROM windows WHERE status IN ('DISCOVERING','PENDING')").rowcount
        return {"interruptedRuns": runs, "forgottenWindows": windows}

    def reprocess(self, conversation_id: str | None = None, only_empty: bool = True, since: str | None = None, until: str | None = None) -> dict:
        """Read RAW again after discovery or verification has improved: forget the windows that found nothing (all windows
        with only_empty=False; only those overlapping [since, until) when given), so their RAW is windowed and discovered
        anew on the next run. Applied candidates are never touched and one RAW span can never become two candidates (digest
        is unique); so that a span found again is not lost, the forgotten windows' NO_ACTION / REJECTED candidates go back
        to STAGING and are decided again under the current rules (their earlier decisions stay on record)."""
        with self.store.tx() as db:
            where = "status IN ('STAGED','DONE')" + (" AND candidate_count = 0" if only_empty else "")
            args: list = []
            if conversation_id:
                where += " AND conversation_id = ?"
                args.append(conversation_id)
            if since:
                where += " AND last_raw_at >= ?"
                args.append(since)
            if until:
                where += " AND first_raw_at < ?"
                args.append(until)
            ids = [r["window_id"] for r in db.execute(f"SELECT window_id FROM windows WHERE {where}", args).fetchall()]
            marks = ",".join("?" * len(ids))
            requeued = db.execute(f"UPDATE candidates SET status='STAGING', attempts=0, last_error=NULL, next_attempt_at=NULL, decision_id=NULL, updated_at=? "
                                  f"WHERE window_id IN ({marks}) AND status IN ('NO_ACTION','REJECTED')", (now_iso(), *ids)).rowcount if ids else 0
            forgotten = db.execute(f"DELETE FROM windows WHERE window_id IN ({marks})", ids).rowcount if ids else 0
        self.store.audit("pipeline.reprocess", conversation_id or "*", after={"forgottenWindows": forgotten, "requeued": requeued, "onlyEmpty": only_empty,
                                                                             "since": since, "until": until}, actor=identity.USER_ACTOR)
        return {"forgottenWindows": forgotten, "requeued": requeued}

    # ------------------------------------------------------------ RAW windows

    def _processed_raw(self) -> set[str]:
        out: set[str] = set()
        for row in self.store.all("SELECT raw_ids FROM windows WHERE status IN ('STAGED','DONE')"):
            out.update(json.loads(row["raw_ids"]))
        return out

    def unprocessed_raw(self) -> list[dict]:
        """Chat RAW no window has covered yet, oldest first. Manual notes are already memories and never re-enter."""
        done = self._processed_raw()
        with self.core.service.lock:
            rows = self.core.service.conn.execute(
                "SELECT id, conversation_id, role, content, created_at, source_type FROM originals WHERE source_type != ? ORDER BY created_at, rowid", (NOTE_SOURCE,)).fetchall()
        return [{"id": r["id"], "conversationId": r["conversation_id"], "role": r["role"], "content": r["content"], "createdAt": r["created_at"]} for r in rows if r["id"] not in done]

    def build_windows(self, raw: list[dict]) -> list[list[dict]]:
        windows: list[list[dict]] = []
        cur: list[dict] = []
        size = 0
        for r in raw:
            cost = len(r["content"])
            gap = False
            if cur:
                try:
                    gap = (datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")) - datetime.fromisoformat(cur[-1]["createdAt"].replace("Z", "+00:00"))).total_seconds() > WINDOW_GAP_HOURS * 3600
                except ValueError:
                    gap = False
            if cur and (r["conversationId"] != cur[0]["conversationId"] or gap or size + cost > WINDOW_CHARS or len(cur) >= WINDOW_MSGS):
                windows.append(cur)
                cur, size = [], 0
            cur.append(r)
            size += cost
        if cur:
            windows.append(cur)
        return windows

    # ------------------------------------------------------------ session segmentation

    @staticmethod
    def _ts(stamp: str) -> datetime:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))

    def build_sessions(self, raw: list[dict], now: datetime | None = None) -> list[list[dict]]:
        """Conversation segments: cut at SESSION_GAP_MINUTES of silence, per conversation; the last segment of a
        conversation is left for later while it is not yet quiet that long. Long segments split at their longest pause."""
        now = now or datetime.now(timezone.utc)
        gap = timedelta(minutes=SESSION_GAP_MINUTES)
        by_conv: dict[str, list[dict]] = {}
        for r in raw:
            by_conv.setdefault(r["conversationId"], []).append(r)
        out: list[list[dict]] = []
        for msgs in by_conv.values():
            segs: list[list[dict]] = []
            for r in msgs:
                if segs and self._ts(r["createdAt"]) - self._ts(segs[-1][-1]["createdAt"]) <= gap:
                    segs[-1].append(r)
                else:
                    segs.append([r])
            if segs and now - self._ts(segs[-1][-1]["createdAt"]) < gap:
                segs.pop()  # still going on
            for seg in segs:
                out.extend(self._split_long(seg))
        return sorted(out, key=lambda seg: seg[0]["createdAt"])

    def _split_long(self, seg: list[dict]) -> list[list[dict]]:
        if len(seg) <= SESSION_MAX_MSGS and sum(len(r["content"]) for r in seg) <= SESSION_MAX_CHARS:
            return [seg]
        # the longest pause, away from the very edges
        lo, hi = max(1, len(seg) // 5), min(len(seg) - 1, len(seg) - len(seg) // 5)
        cut = max(range(lo, hi), key=lambda i: self._ts(seg[i]["createdAt"]) - self._ts(seg[i - 1]["createdAt"]), default=len(seg) // 2)
        return self._split_long(seg[:cut]) + self._split_long(seg[cut:])

    def _previous_tail(self, msgs: list[dict]) -> list[str]:
        """The last messages before this segment in the same conversation (within LINK_LOOKBACK_HOURS): the link to it."""
        since = (self._ts(msgs[0]["createdAt"]) - timedelta(hours=LINK_LOOKBACK_HOURS)).isoformat()
        with self.core.service.lock:
            rows = self.core.service.conn.execute(
                "SELECT id, created_at FROM originals WHERE conversation_id = ? AND source_type != ? AND created_at < ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (msgs[0]["conversationId"], NOTE_SOURCE, msgs[0]["createdAt"], LINK_TAIL_MSGS)).fetchall()
        return [r["id"] for r in reversed(rows) if self._ts(r["created_at"]).isoformat() >= since]

    def _stage_session(self, run_id: str, msgs: list[dict]) -> dict:
        window_id = mint("window")
        raw_ids = [m["id"] for m in msgs]
        context = self._previous_tail(msgs)
        a, b = self._ts(msgs[0]["createdAt"]).astimezone(LOCAL_OFFSET), self._ts(msgs[-1]["createdAt"]).astimezone(LOCAL_OFFSET)
        gist = f"对话 {a:%m月%d日 %H:%M}–{b:%H:%M} · {len(msgs)} 条" + ("（接上一段）" if context else "")
        digest = hashlib.sha256(dumps(raw_ids).encode()).hexdigest()
        with self.store.tx() as db:
            db.execute("INSERT INTO windows (window_id, run_id, conversation_id, raw_ids, status, first_raw_at, last_raw_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (window_id, run_id, msgs[0]["conversationId"], dumps(raw_ids), "DISCOVERING", msgs[0]["createdAt"], msgs[-1]["createdAt"], now_iso(), now_iso()))
            made = db.execute(
                "INSERT INTO candidates (candidate_id, run_id, window_id, origin, kind_hint, gist, entities, topics, raw_ids, context_raw_ids, digest, status, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(digest) DO NOTHING",
                (mint("cand"), run_id, window_id, "session", "", gist, dumps([]), dumps([]), dumps(raw_ids), dumps(context), digest, "STAGING", now_iso(), now_iso())).rowcount
            db.execute("UPDATE windows SET status='STAGED', mode='session', attempts=0, candidate_count=?, updated_at=? WHERE window_id=?", (made, now_iso(), window_id))
        return {"window_id": window_id, "mode": "session", "candidates": made, "raw": len(msgs), "warnings": []}

    def pending_work(self) -> dict:
        staging_due = sum(1 for r in self.store.all("SELECT next_attempt_at FROM candidates WHERE status='STAGING'") if _due(r["next_attempt_at"]))
        unprocessed = len(self.unprocessed_raw())
        return {"unprocessedRaw": unprocessed, "stagingDue": staging_due, "any": bool(unprocessed or staging_due)}

    # ------------------------------------------------------------ discovery of one window

    def _stage_window(self, run_id: str, msgs: list[dict]) -> dict:
        raw_full = [self.core.raw_record(m["id"]) for m in msgs]
        window_id = mint("window")
        raw_ids = [m["id"] for m in msgs]
        with self.store.tx() as db:
            db.execute("INSERT INTO windows (window_id, run_id, conversation_id, raw_ids, status, first_raw_at, last_raw_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (window_id, run_id, msgs[0]["conversationId"], dumps(raw_ids), "DISCOVERING", msgs[0]["createdAt"], msgs[-1]["createdAt"], now_iso(), now_iso()))
        result = self.discovery.discover(raw_full)
        made = 0
        with self.store.tx() as db:
            for c in result.candidates:
                span = raw_ids[c["start"]:c["end"] + 1]
                context = raw_ids[max(0, c["start"] - CONTEXT_MSGS):c["start"]] + raw_ids[c["end"] + 1:c["end"] + 1 + CONTEXT_MSGS]
                digest = hashlib.sha256(dumps(span).encode()).hexdigest()
                cid = mint("cand")
                inserted = db.execute(
                    "INSERT INTO candidates (candidate_id, run_id, window_id, origin, kind_hint, gist, entities, topics, raw_ids, context_raw_ids, digest, status, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(digest) DO NOTHING",
                    (cid, run_id, window_id, c.get("origin", result.mode), c.get("kind", ""), c.get("gist", ""), dumps(c.get("entities", [])), dumps(c.get("topics", [])),
                     dumps(span), dumps(context), digest, "STAGING", now_iso(), now_iso())).rowcount
                made += inserted
            db.execute("UPDATE windows SET status='STAGED', mode=?, attempts=?, last_error=?, warnings=?, candidate_count=?, provider=?, model=?, latency_ms=?, updated_at=? WHERE window_id=?",
                       (result.mode, result.attempts, result.error, dumps(result.warnings), made, result.provider, result.model, result.latency_ms, now_iso(), window_id))
        self._note_discovery(result, len(msgs), made)
        return {"window_id": window_id, "mode": result.mode, "candidates": made, "raw": len(msgs), "warnings": result.warnings}

    def _note_discovery(self, result, raw_count: int, made: int) -> None:
        stats = self.store.kv_get("ollama_stats", {}) | {}
        stats["last_run_at"] = now_iso()
        stats["processed_raw_count"] = stats.get("processed_raw_count", 0) + raw_count
        stats["candidate_count"] = stats.get("candidate_count", 0) + made
        if result.mode.startswith("ollama") and result.ollama_calls and not result.error:
            stats["last_success_at"] = now_iso()
            stats["windows_ollama"] = stats.get("windows_ollama", 0) + 1
        if result.error or result.mode == "deterministic":
            stats["last_failure_at"] = now_iso()
            stats["last_error"] = (result.error or "fell back to deterministic chunking")[:300]
            stats["windows_fallback"] = stats.get("windows_fallback", 0) + 1
        self.store.kv_set("ollama_stats", stats)

    # ------------------------------------------------------------ verification of one candidate

    def _backoff(self, attempts: int) -> str:
        return _later(min(BACKOFF_BASE_S * 2 ** max(0, attempts - 1), BACKOFF_CAP_S))

    def _bump_stats(self, outcome: str) -> None:
        stats = self.store.kv_get("deepseek_stats", {}) | {}
        stats["last_verification_at"] = now_iso()
        stats[outcome] = stats.get(outcome, 0) + 1
        self.store.kv_set("deepseek_stats", stats)

    def verify_candidate(self, candidate_id: str, actor_review: dict | None = None) -> str:
        """Decide one STAGING candidate. Returns its new status (or 'skipped')."""
        row = self.store.one("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,))
        if row is None or row["status"] != "STAGING":
            return "skipped"
        raw_ids, context_ids = json.loads(row["raw_ids"]), json.loads(row["context_raw_ids"])
        try:
            raw = [self.core.raw_record(i) for i in raw_ids]
            context = [self.core.raw_record(i) for i in context_ids]
        except NotFound as error:
            return self._finish(candidate_id, None, "QUARANTINE", errors=[f"RAW is missing: {error}"])
        # A whole segment (and the end of the one before) looks for what is already remembered over all of it, so the
        # same thing is not recorded twice; a discovery span only needs its own words.
        query = (" ".join([r["content"] for r in context + raw])[-1600:] if row["origin"] == "session"
                 else " ".join([row["gist"], *[r["content"] for r in raw]])[:400])
        patterns, episodes = self.core.read.related_units(query)
        payload = build_payload({**dict(row), "kind_hint": row["kind_hint"], "entities": json.loads(row["entities"])}, raw, context, patterns, episodes, now_iso(),
                                commitments.for_verification(self.core),
                                threads_module.candidates_for(self.store, [e["episode_id"] for e in episodes], _time_of(raw)),
                                [e for e in self.store.episodes(status="active") if e.get("kind") == "ritual"])
        if not self.verifier.available():
            return self._defer(candidate_id, row, "DeepSeek is not configured (DEEPSEEK_API_KEY is missing)", count_attempt=False)
        try:
            answer, meta = self.verifier.decide(payload)
        except WorkerError as error:
            self._bump_stats("failed")
            return self._defer(candidate_id, row, str(error)[:300])
        allowed = set(raw_ids) | set(context_ids)
        decision_id = mint("decision")
        base = {"decision_id": decision_id, "candidate_id": candidate_id, "provider": self.verifier.provider, "model": self.verifier.model,
                "prompt_version": PROMPT_VERSION, "usage": meta.get("usage", {}), "answer": answer, "raw_hashes": {r["id"]: r["sha256"] for r in raw + context}}
        try:
            plan = validate_answer(answer, allowed, self.store)
            base.update(confidence=plan.confidence, reason=plan.reason, actions=plan.actions)
            # Verified quotes: what it puts in quotation marks must have been said (in this RAW, or already quoted in a
            # memory it was shown). Otherwise nothing is applied and the user sees it in QUARANTINE.
            bad = unverified_quotes(plan_texts(plan.actions), [r["content"] for r in raw + context]
                                    + [e["content"] for e in episodes] + [p["narrative"] for p in patterns])
            if bad and not plan.terminal:
                return self._finish(candidate_id, base, "QUARANTINE", errors=[f"unverified quote: “{q[:60]}”" for q in bad[:5]])
            if plan.terminal:
                status = {"NO_ACTION": "NO_ACTION", "REJECT": "REJECTED", "QUARANTINE": "QUARANTINE"}[plan.terminal]
                return self._finish(candidate_id, base, status, applied={})
            with self.store.tx():
                applied = apply_plan(self.store, plan, candidate_id, decision_id)
                wrote = any(applied[k] for k in ("episodes_created", "episodes_updated", "episodes_merged", "patterns_created", "patterns_updated"))
                if applied.get("tombstoned") and not wrote:  # all of it was something the user had deleted
                    return self._finish(candidate_id, base, "NO_ACTION", applied=applied, errors=["tombstoned: the user deleted this memory before; not created again"])
                return self._finish(candidate_id, base, "APPLIED", applied=applied)
        except Exception as error:  # anything the validation or the apply step cannot accept: the whole decision is quarantined
            base.setdefault("confidence", None)
            base.setdefault("reason", "")
            base.setdefault("actions", [])
            return self._finish(candidate_id, base, "QUARANTINE", errors=[f"{type(error).__name__}: {error}"[:400]])

    def _finish(self, candidate_id: str, decision: dict | None, status: str, applied: dict | None = None, errors: list | None = None) -> str:
        with self.store.tx() as db:
            decision_id = None
            if decision is not None:
                decision_id = decision["decision_id"]
                db.execute("INSERT INTO decisions (decision_id, candidate_id, provider, model, prompt_version, at, status, confidence, reason, actions, answer, errors, applied, usage, raw_hashes) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (decision_id, candidate_id, decision["provider"], decision["model"], decision["prompt_version"], now_iso(), status, decision.get("confidence"),
                            decision.get("reason", ""), dumps(decision.get("actions", [])), dumps(decision.get("answer", {})), dumps(errors or []), dumps(applied or {}),
                            dumps(decision.get("usage", {})), dumps(decision.get("raw_hashes", {}))))
            db.execute("UPDATE candidates SET status=?, decision_id=?, last_error=?, next_attempt_at=NULL, updated_at=? WHERE candidate_id=?",
                       (status, decision_id, "; ".join(errors)[:400] if errors else None, now_iso(), candidate_id))
        self._bump_stats({"APPLIED": "accepted", "NO_ACTION": "no_action", "REJECTED": "rejected", "QUARANTINE": "quarantined"}.get(status, "other"))
        self.store.audit(f"candidate.{status.lower()}", candidate_id, after={"decision_id": decision_id, "errors": errors or []}, actor="deepseek", decision_id=decision_id)
        return status

    def _defer(self, candidate_id: str, row, error: str, count_attempt: bool = True) -> str:
        attempts = row["attempts"] + (1 if count_attempt else 0)
        status = "FAILED" if count_attempt and attempts >= MAX_VERIFY_ATTEMPTS else "STAGING"
        with self.store.tx() as db:
            db.execute("UPDATE candidates SET status=?, attempts=?, last_error=?, next_attempt_at=?, updated_at=? WHERE candidate_id=?",
                       (status, attempts, error, self._backoff(attempts) if status == "STAGING" and count_attempt else None, now_iso(), candidate_id))
        if status == "FAILED":
            self.store.audit("candidate.failed", candidate_id, after={"error": error, "attempts": attempts})
        return status

    # ------------------------------------------------------------ a run

    def run(self, reason: str = "manual", budget_s: float = MAX_RUN_SECONDS, max_windows: int | None = None) -> dict:
        if not self.work_lock.acquire(blocking=False):
            return {"busy": True}
        run_id = mint("run")
        started = time.time()
        summary = {"windows": 0, "raw": 0, "candidates": 0, "modes": {}, "statuses": {}, "resumed": 0}
        errors: list[str] = []
        try:
            with self.store.tx() as db:
                db.execute("INSERT INTO runs (run_id, reason, status, started_at, summary) VALUES (?,?,?,?,?)", (run_id, reason, "running", now_iso(), dumps(summary)))
            self.store.audit("run.started", run_id, after={"reason": reason})
            # 1. candidates from earlier runs that are due again
            for row in self.store.all("SELECT candidate_id, next_attempt_at FROM candidates WHERE status='STAGING' ORDER BY created_at"):
                if time.time() - started > budget_s or self.stop.is_set():
                    break
                if _due(row["next_attempt_at"]):
                    summary["resumed"] += 1
                    status = self.verify_candidate(row["candidate_id"])
                    summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
            # 2. new RAW, window by window
            done = 0
            sessions = self.segmentation == "session"
            for msgs in (self.build_sessions(self.unprocessed_raw()) if sessions else self.build_windows(self.unprocessed_raw())):
                if time.time() - started > budget_s or self.stop.is_set() or (max_windows is not None and done >= max_windows):
                    break
                try:
                    info = self._stage_session(run_id, msgs) if sessions else self._stage_window(run_id, msgs)
                except Exception as error:  # a broken window fails alone; the others go on
                    errors.append(f"window {msgs[0]['id']}: {error!r}"[:300])
                    continue
                done += 1
                summary["windows"] += 1
                summary["raw"] += info["raw"]
                summary["candidates"] += info["candidates"]
                summary["modes"][info["mode"]] = summary["modes"].get(info["mode"], 0) + 1
                for row in self.store.all("SELECT candidate_id FROM candidates WHERE window_id = ? AND status='STAGING'", (info["window_id"],)):
                    if time.time() - started > budget_s or self.stop.is_set():
                        break
                    status = self.verify_candidate(row["candidate_id"])
                    summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
            pending = self.store.one("SELECT COUNT(*) FROM candidates WHERE status='STAGING'")[0]
            status = "pending_verification" if pending else "completed"
            last_raw = self.store.one("SELECT raw_ids FROM windows WHERE status IN ('STAGED','DONE') ORDER BY last_raw_at DESC, created_at DESC LIMIT 1")
            checkpoint = self.store.kv_get("checkpoint", {}) | {}
            checkpoint.update(last_run_id=run_id, last_run_at=now_iso(), last_processed_raw_id=(json.loads(last_raw["raw_ids"])[-1] if last_raw else checkpoint.get("last_processed_raw_id")))
            if status == "completed" and not errors:
                checkpoint["last_success_at"] = now_iso()
            self.store.kv_set("checkpoint", checkpoint)
            with self.store.tx() as db:
                db.execute("UPDATE runs SET status=?, finished_at=?, summary=?, errors=? WHERE run_id=?", (status, now_iso(), dumps(summary), dumps(errors), run_id))
            # The names whose memories changed get their profile rewritten (one LLM call each, only when something changed).
            if summary["statuses"].get("APPLIED") and self.verifier.available():
                summary["profiles"] = len(names_module.refresh_profiles(self.store, self.verifier.client))
            self.store.audit(f"run.{status}", run_id, after=summary)
            return {"run_id": run_id, "status": status, "summary": summary, "errors": errors}
        except Exception as error:
            with self.store.tx() as db:
                db.execute("UPDATE runs SET status='failed', finished_at=?, summary=?, errors=? WHERE run_id=?", (now_iso(), dumps(summary), dumps([*errors, repr(error)[:300]]), run_id))
            self.store.audit("run.failed", run_id, after={"error": repr(error)[:300]})
            return {"run_id": run_id, "status": "failed", "summary": summary, "errors": [*errors, repr(error)[:300]]}
        finally:
            self._release_discovery_model()
            self.work_lock.release()

    def _release_discovery_model(self) -> None:
        """The discovery model is only needed inside a run: one load serves every window of the run, and its memory is
        given back as soon as the run is over (still under the work lock, so it cannot race the next run's load)."""
        ollama = getattr(self.discovery, "ollama", None)
        if ollama is not None and getattr(ollama, "used", False):
            ollama.unload()

    def start_run(self, reason: str = "manual") -> dict:
        if self.work_lock.locked():
            return {"busy": True}
        threading.Thread(target=self.run, args=(reason,), daemon=True, name="memory-pipeline").start()
        return {"started": True}

    # ------------------------------------------------------------ scheduler (idle / night / morning)

    def settings(self) -> dict:
        return DEFAULT_SETTINGS | self.store.kv_get("settings", {})

    def update_settings(self, payload: dict) -> dict:
        current = self.settings()
        for key, (low, high) in SETTING_BOUNDS.items():
            if key in payload:
                value = float(payload[key])
                if not low <= value <= high:
                    raise Invalid(f"{key} out of range")
                current[key] = value
        if "enabled" in payload:
            current["enabled"] = payload["enabled"] is True
        self.store.kv_set("settings", current)
        self.store.audit("settings.updated", "settings", after=current, actor=identity.USER_ACTOR)
        return current

    def tick(self, now: datetime | None = None, background: bool = True) -> dict:
        now = now or datetime.now(timezone.utc)
        cfg = self.settings()
        if not cfg["enabled"]:
            return {"triggered": False, "reason": "disabled"}
        work = self.pending_work()
        if not work["any"]:
            return {"triggered": False, "reason": "no_work", **work}
        hour = (now + timedelta(hours=cfg["utc_offset"])).hour
        night = (hour >= cfg["night_start"] or hour < cfg["night_end"]) if cfg["night_start"] > cfg["night_end"] else cfg["night_start"] <= hour < cfg["night_end"]
        morning = cfg["morning_hour"] <= hour < cfg["morning_hour"] + 1
        checkpoint = self.store.kv_get("checkpoint", {})
        # Last activity lives in its own key, written by MemoryCore.activity() whenever new RAW arrives (never in the
        # checkpoint). No activity on record yet counts as "just now", as before.
        try:
            last = datetime.fromisoformat((self.store.kv_get(LAST_ACTIVITY_KEY) or now.isoformat()).replace("Z", "+00:00"))
        except ValueError:
            last = now
        if (now - last).total_seconds() < cfg["idle_minutes"] * 60 or not (night or morning):
            return {"triggered": False, "reason": "active_or_outside_window", **work}
        attempt = checkpoint.get("last_attempt_at")
        if attempt and (now - datetime.fromisoformat(attempt.replace("Z", "+00:00"))).total_seconds() < 300:
            return {"triggered": False, "reason": "retry_backoff", **work}
        self.store.kv_set("checkpoint", checkpoint | {"last_attempt_at": now.isoformat()})
        reason = "night" if night else "morning_recheck"
        result = self.start_run(reason) if background else self.run(reason)
        return {"triggered": True, "reason": reason, "result": result, **work}

    def start_scheduler(self) -> None:
        def loop():
            while not self.stop.wait(30):
                try:
                    self.tick()
                except Exception as error:
                    self.store.kv_set("scheduler_error", {"at": now_iso(), "error": repr(error)[:300]})
        self.thread = threading.Thread(target=loop, daemon=True, name="memory-idle-scheduler")
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(2)
        with self.work_lock:  # bounded model calls finish before the database closes
            pass
