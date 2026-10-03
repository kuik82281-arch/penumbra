"""Candidate Discovery: which passages of a chat window might deserve a long-term memory.

Ollama (qwen2.5:3b) does this in normal operation. It only *points*: message numbers, a kind, a one-line gist, entities.
It never writes memory text and nothing it says is stored as memory - candidates go to STAGING and DeepSeek decides.

When Ollama is down, slow, or answers with something unusable, discovery falls back to deterministic chunking (time gaps
and simple signals, no model), so the pipeline always moves forward. A window whose Ollama answer is empty but that
carries strong signals (a plan, a change of state, a milestone) also gets deterministic candidates, so a small model
that says "nothing" cannot silently swallow a real event. Every candidate says where it came from (`origin`).

Small-model failure modes handled here (observed on qwen2.5:3b): runaway repetition until the token budget cuts the JSON
(the complete elements are kept, the cut tail dropped), out-of-range message numbers (dropped), duplicate / overlapping
spans (merged), over-eager output (capped).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from .. import identity, prompts
from ..errors import WorkerError
from .llm import AnswerError, OllamaClient, complete_objects, parse_json_answer

PROMPT_VERSION = "discover-v2"
KINDS = ("event", "plan", "promise", "state_change", "preference", "milestone", "fact")
MAX_CANDIDATES = 6
MAX_SPAN = 12
SPLIT_MIN = 8

SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
        "from": {"type": "integer"}, "to": {"type": "integer"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "gist": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    }, "required": ["from", "to", "kind", "gist", "entities"]}}},
    "required": ["candidates"],
}


def speaker(role: str) -> str:
    """A message's role as the name the memory uses."""
    return {"user": identity.user(), "assistant": identity.assistant()}.get(role, role or "?")


def render_window(msgs: list[dict]) -> str:
    lines = []
    for i, m in enumerate(msgs, 1):
        text = str(m.get("content") or "").strip()
        note = f"（附{len(m['attachments'])}张图片）" if m.get("attachments") else ""
        lines.append(f"[{i}] {speaker(m.get('role', ''))}：{text}{note}")
    return "\n".join(lines)


@dataclass
class DiscoveryResult:
    candidates: list[dict] = field(default_factory=list)
    mode: str = "deterministic"  # ollama | ollama+safety | deterministic
    provider: str = "deterministic"
    model: str = "signals-v1"
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    latency_ms: int = 0
    error: str | None = None
    ollama_calls: int = 0


# ------------------------------------------------------------ candidate cleaning


def _clean(item, n: int) -> dict | None:
    """One model candidate as a validated 0-based span, or None. Never raises: a bad item is dropped, not fatal."""
    if not isinstance(item, dict):
        return None
    try:
        start, end = int(item.get("from")), int(item.get("to"))
    except (TypeError, ValueError):
        return None
    if not (1 <= start <= end <= n):
        return None
    kind = str(item.get("kind") or "").strip()
    gist = str(item.get("gist") or "").strip()[:160]
    entities = [str(e).strip()[:40] for e in (item.get("entities") if isinstance(item.get("entities"), list) else []) if str(e).strip()][:6]
    return {"start": start - 1, "end": min(end - 1, start - 1 + MAX_SPAN - 1), "kind": kind if kind in KINDS else "event",
            "gist": gist, "entities": list(dict.fromkeys(entities)), "topics": []}


def merge_spans(cands: list[dict]) -> list[dict]:
    """Overlapping or touching spans become one; the result is ordered, de-duplicated and capped."""
    ordered = sorted(cands, key=lambda c: (c["start"], c["end"]))
    merged: list[dict] = []
    for c in ordered:
        last = merged[-1] if merged else None
        if last and c["start"] <= last["end"] + 0:
            last["end"] = max(last["end"], c["end"])
            last["entities"] = list(dict.fromkeys(last["entities"] + c["entities"]))[:8]
            last["topics"] = list(dict.fromkeys(last["topics"] + c["topics"]))[:6]
            if len(c["gist"]) > len(last["gist"]):
                last["gist"] = c["gist"]
            if last["end"] - last["start"] + 1 > MAX_SPAN:
                last["end"] = last["start"] + MAX_SPAN - 1
        else:
            merged.append(dict(c))
    return merged[:MAX_CANDIDATES]


# ------------------------------------------------------------ deterministic chunking

SIGNALS = {
    "plan": (re.compile(r"决定|计划|打算|约好|约定|安排|订了|预约|下周|下个月|明年|下次|以后每|从.{1,6}起|截止|报名"), 2),
    # A promise of either of them, however small ("明天给你打领带"), and its fulfilment: what the to-do memory is made of.
    "promise": (re.compile(r"答应(你|我)|我保证|说好了|说定了|一言为定|拉钩|承诺|发誓|(明天|今天|今晚|待会|等下|回来|周末)(就)?(给你|帮你|带你|陪你|给我|帮我|带我|陪我)"
                           r"|(给你|帮你)(买|打|做|带|拿|系)|(买|打|做|带)好了"), 5),
    "change": (re.compile(r"搬(到|去|家)|换(了|成)|不再|开始|停止|辞职|毕业|分手|结婚|怀孕|改(成|了)|终于|后来|原来|以前.{0,20}现在|已经(不|没)"), 3),
    "milestone": (re.compile(r"第一次|生日|纪念日|考试|面试|入职|拿到|通过了|获得|完成了|出生|去世|手术|住院"), 3),
    "fact": (re.compile(r"过敏|一直(都)?(喜欢|讨厌|怕)|我(喜欢|讨厌|怕|不吃|不喝)|我(是|有|住|在)[^，。！？]{1,15}(上班|工作|读书|住)"), 1),
    # An attitude, and above all a change of attitude ("以前讨厌香菜，现在喜欢"): what long-term patterns are made of.
    "stance_change": (re.compile(r"(以前|从前|原来|曾经|现在|最近|后来|如今|终于|居然).{0,16}(讨厌|喜欢|爱吃|爱喝|爱上|怕|吃得下|吃不下|接受|不再|习惯|受不了|不敢)|(讨厌|喜欢|爱吃|爱喝|怕|受不了).{0,12}(现在|后来|以前|从前|如今|变了)"), 5),
    "stance": (re.compile(r"(特别|一直|非常|最|很|超级|挺)(讨厌|喜欢|爱吃|爱喝|怕|受不了)|我(不喜欢|讨厌|喜欢|爱吃|爱喝|怕|受不了)"), 3),
    "time": (re.compile(r"\d{1,2}月\d{1,2}[日号]|\d{4}年|周[一二三四五六日天]|星期[一二三四五六日天]|\d{1,2}[点:：]\d{0,2}"), 1),
}
GAP_MINUTES = 25
SEGMENT_CHARS = 1400
SEGMENT_MSGS = 10


def _when(msg: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(str(msg.get("createdAt") or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def _segments(msgs: list[dict]) -> list[tuple[int, int]]:
    """Runs of messages that belong together: cut at a long silence, or when a run gets long."""
    out, start, size = [], 0, 0
    for i, m in enumerate(msgs):
        size += len(str(m.get("content") or ""))
        prev = _when(msgs[i - 1]) if i else None
        cur = _when(m)
        gap = bool(prev and cur and (cur - prev).total_seconds() > GAP_MINUTES * 60)
        if i and (gap or size > SEGMENT_CHARS or i - start >= SEGMENT_MSGS):
            out.append((start, i - 1))
            start, size = i, len(str(m.get("content") or ""))
    out.append((start, len(msgs) - 1))
    return out


def deterministic_candidates(msgs: list[dict], strong_only: bool = False) -> list[dict]:
    """Segments of the window ranked by simple, explainable signals. No model."""
    threshold = 5 if strong_only else 3
    scored = []
    for start, end in _segments(msgs):
        text = " ".join(str(m.get("content") or "") for m in msgs[start:end + 1])
        hits = {name: pattern.search(text) for name, (pattern, _) in SIGNALS.items()}
        score = sum(weight for name, (_, weight) in SIGNALS.items() if hits[name])
        score += 1 if any(m.get("attachments") for m in msgs[start:end + 1]) else 0
        score += 1 if any(m.get("role") == "user" and hits["plan"] or hits["change"] for m in msgs[start:end + 1]) else 0
        if score < threshold:
            continue
        topics = [name for name, hit in hits.items() if hit and name != "time"]
        anchor = next((m for m in msgs[start:end + 1] if m.get("role") == "user" and any(p.search(str(m.get("content"))) for p, _ in SIGNALS.values())), msgs[start])
        gist = re.split(r"[。！？!?\n]", str(anchor.get("content") or "").strip())[0][:80]
        scored.append((score, {"start": start, "end": min(end, start + MAX_SPAN - 1), "kind": ("promise" if hits["promise"] else "state_change" if hits["change"] or hits["stance_change"] else "preference" if hits["stance"] else "plan" if hits["plan"]
                                                                                              else "milestone" if hits["milestone"] else "fact"),
                               "gist": gist, "entities": [], "topics": topics}))
    best = sorted(scored, key=lambda s: -s[0])[:MAX_CANDIDATES]
    return sorted((c for _, c in best), key=lambda c: c["start"])


# ------------------------------------------------------------ discovery


class Discovery:
    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or OllamaClient()

    def _ask(self, msgs: list[dict], result: DiscoveryResult) -> list[dict]:
        """One Ollama call on `msgs`. Returns cleaned candidates (0-based within msgs) or raises WorkerError / AnswerError."""
        content, meta = self.ollama.chat(prompts.get("discover"), render_window(msgs), SCHEMA, num_predict=700)
        result.ollama_calls += 1
        result.latency_ms += meta.get("latency_ms") or 0
        try:
            data = parse_json_answer(content, meta.get("done_reason"))
            raw = data.get("candidates")
            if not isinstance(raw, list):
                raise AnswerError("malformed", "no candidates list")
        except AnswerError as error:
            if error.kind == "empty":
                raise
            raw = complete_objects(content, "candidates")  # the elements it finished; the cut-off tail is dropped
            if not raw:
                raise
            result.warnings.append(f"salvaged {len(raw)} complete candidate(s) from a {error.kind} answer")
        return [c for c in (_clean(item, len(msgs)) for item in raw) if c]

    def discover(self, msgs: list[dict]) -> DiscoveryResult:
        started = time.perf_counter()
        result = DiscoveryResult()
        health = self.ollama.health()
        if not health.get("ready"):
            result.warnings.append(f"ollama unavailable: {health.get('error')}")
            result.error = health.get("error")
            return self._fallback(msgs, result, started)
        result.provider, result.model, result.mode = "ollama", self.ollama.model, "ollama"
        cands: list[dict] | None = None
        for attempt in (1, 2):
            result.attempts = attempt
            try:
                cands = self._ask(msgs, result)
                break
            except (WorkerError, AnswerError) as error:
                result.error = str(error)[:200]
                result.warnings.append(f"attempt {attempt}: {str(error)[:120]}")
                if isinstance(error, WorkerError) and not self.ollama.health(force=True).get("ready"):
                    break  # the server went away: no point asking again
                if len(msgs) >= SPLIT_MIN * 2 and attempt == 1:
                    # A smaller window is a smaller answer: ask each half instead of the same request again.
                    half = len(msgs) // 2
                    try:
                        first = self._ask(msgs[:half], result)
                        second = [dict(c, start=c["start"] + half, end=c["end"] + half) for c in self._ask(msgs[half:], result)]
                        cands = first + second
                        result.warnings.append("window split in half after a failed answer")
                        break
                    except (WorkerError, AnswerError) as again:
                        result.error = str(again)[:200]
                        result.warnings.append(f"split attempt: {str(again)[:120]}")
                        break
        if cands is None:
            return self._fallback(msgs, result, started)
        result.error = None
        for c in cands:
            c["origin"] = "ollama"
        merged = merge_spans(cands)
        # A model answer of "nothing" does not outrank an explicit plan / change / milestone in the text.
        safety = [c for c in deterministic_candidates(msgs, strong_only=True) if not any(c["start"] <= m["end"] and m["start"] <= c["end"] for m in merged)]
        if safety:
            result.mode = "ollama+safety"
            result.warnings.append(f"{len(safety)} strong-signal passage(s) Ollama did not point at were added deterministically")
            merged = merge_spans(merged + [dict(c, origin="deterministic-safety") for c in safety])
        result.candidates = merged
        result.latency_ms = round((time.perf_counter() - started) * 1000)
        return result

    def _fallback(self, msgs: list[dict], result: DiscoveryResult, started: float) -> DiscoveryResult:
        result.mode, result.provider, result.model = "deterministic", "deterministic", "signals-v1"
        result.candidates = [dict(c, origin="deterministic") for c in deterministic_candidates(msgs)]
        result.latency_ms = round((time.perf_counter() - started) * 1000)
        return result
