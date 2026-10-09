"""Memory gate: DeepSeek reads the turn before and after the search, so memory follows what she means, not shared words.

  intent  before the search: does this message need long-term memory at all (none / maybe / yes), and if so, what is
          really being looked for ("又碰见上次那个人了" -> "她多次遇到的那个人是谁"). none -> NO_MEMORY_NEEDED without a
          search; otherwise the search runs once more with that need, fused with the original.
  judge   after the ranking: of the few candidates left, which really help this reply. Those that only share a word, a
          mood or a kind of scene are dropped - a reranker orders candidates, it cannot say that none is needed.

Both are short calls (one small JSON answer, one attempt, no repair retry) under a timeout taken from the turn's time
budget, so the whole retrieval answers before the bridge stops waiting. A failure comes back as {error, failure:
timeout | error} and never means "inject what the ranking found": read.py degrades conservatively (see there).

  PENUMBRA_MEMORY_GATE: "on" (default: intent + judge) | "judge" (judge only) | "intent" (intent only) | "off"
  MEMORY_GATE_BUDGET (5.5 s: intent + search + judge), MEMORY_GATE_TIMEOUT (2 s: one call at most)
"""
from __future__ import annotations

import os
import time

from .. import prompts
from .llm import AnswerError, DeepSeekClient, parse_json_answer

MODES = ("on", "judge", "intent", "off")
MAX_LINE = 300
MAX_CANDIDATES = 5
BUDGET_S = float(os.environ.get("MEMORY_GATE_BUDGET", "") or 5.5)
CALL_TIMEOUT_S = float(os.environ.get("MEMORY_GATE_TIMEOUT", "") or 2.0)
MIN_CALL_S = 0.6  # less time than this left in the budget: the call is not made (it could not answer in time)
# Circuit breaker: after this many timeouts / network errors in a row, DeepSeek is not asked for a while - a turn then
# degrades at once instead of waiting out two timeouts while the service is down.
BREAK_AFTER = 2
BREAK_S = 60.0


def mode(override: str | None = None) -> str:
    value = (override or os.environ.get("PENUMBRA_MEMORY_GATE", "")).strip().lower()
    return value if value in MODES else "on"


def _lines(recent: list[dict]) -> str:
    out = [f"{'她' if str(r.get('role')) == 'user' else '他'}：{str(r.get('content') or '')[:MAX_LINE]}" for r in recent[-6:] if r.get("content")]
    return "\n".join(out) or "（没有）"


def _slow(text: str) -> bool:
    return "timed out" in text or "timeout" in text.lower()


class MemoryGate:
    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient(timeout_s=CALL_TIMEOUT_S, max_retries=0)
        self._misses = 0
        self._open_until = 0.0

    def _settle(self, meta: dict) -> dict:
        """Count a call for the breaker: a timeout / network failure moves it, anything that answered resets it."""
        if meta.get("failure") == "timeout" or str(meta.get("error", "")).startswith("network"):
            self._misses += 1
            if self._misses >= BREAK_AFTER:
                self._open_until = time.monotonic() + BREAK_S
        elif "error" not in meta or meta.get("failure") == "error":
            self._misses, self._open_until = 0, 0.0
        return meta

    def available(self) -> bool:
        return self.client.available()

    def _ask(self, prompt: str, user: str, max_tokens: int, timeout_s: float) -> tuple[dict | None, dict]:
        """One attempt within `timeout_s`. A failure is (None, {error, failure: timeout | error})."""
        if timeout_s < MIN_CALL_S:
            return None, {"latency_ms": 0, "error": f"no time left in the budget ({timeout_s:.1f} s)", "failure": "timeout"}
        if time.monotonic() < self._open_until:
            return None, {"latency_ms": 0, "error": "DeepSeek did not answer the last calls; not asked for a minute", "failure": "timeout", "breaker": True}
        answer, meta = self._call(prompt, user, max_tokens, timeout_s)
        return answer, self._settle(meta)

    def _call(self, prompt: str, user: str, max_tokens: int, timeout_s: float) -> tuple[dict | None, dict]:
        started = time.perf_counter()
        ms = lambda: round((time.perf_counter() - started) * 1000)  # noqa: E731
        if not hasattr(self.client, "_request"):  # a test double answers chat_json directly
            try:
                answer, _meta = self.client.chat_json(prompts.get(prompt), user, max_tokens=max_tokens)
                return answer, {"latency_ms": ms()}
            except Exception as error:
                text = str(error)[:200]
                return None, {"latency_ms": ms(), "error": text, "failure": "timeout" if _slow(text) else "error"}
        # its own client per call: the timeout is this call's, and turns running side by side do not share one
        client = DeepSeekClient(timeout_s=min(CALL_TIMEOUT_S, timeout_s), max_retries=0)
        try:
            result = client._request(prompts.get(prompt), user, max_tokens)
            return parse_json_answer(result["content"], result["finish_reason"]), {"latency_ms": ms()}
        except AnswerError as error:
            return None, {"latency_ms": ms(), "error": f"unusable answer ({error.kind})", "failure": "error"}
        except Exception as error:
            text = str(error)[:200]
            slow = _slow(text) or ms() >= client.timeout_s * 950
            return None, {"latency_ms": ms(), "error": text, "failure": "timeout" if slow else "error"}

    def intent(self, query: str, recent: list[dict], timeout_s: float = CALL_TIMEOUT_S) -> dict:
        """{need: none|maybe|yes, seek, latency_ms} or {error, failure, latency_ms}."""
        answer, meta = self._ask("intent", f"最近的对话：\n{_lines(recent)}\n\n她刚说：{query[:600]}", 200, timeout_s)
        if answer is None:
            return meta
        need = answer.get("need") if answer.get("need") in ("none", "maybe", "yes") else None
        if need is None:
            return {**meta, "error": "no need in the answer", "failure": "error"}
        seek = str(answer.get("seek") or "").strip()[:120] if need != "none" else ""
        return {**meta, "need": need, "seek": seek}

    def judge(self, query: str, recent: list[dict], seek: str, docs: list[str], timeout_s: float = CALL_TIMEOUT_S) -> dict:
        """{keep: [index, ...], latency_ms} or {error, failure, latency_ms}."""
        numbered = "\n".join(f"[{i}] {d[:500]}" for i, d in enumerate(docs))
        user = f"最近的对话：\n{_lines(recent)}\n\n她刚说：{query[:600]}" + (f"\n\n要找的：{seek}" if seek else "") + f"\n\n候选记忆：\n{numbered}"
        answer, meta = self._ask("judge", user, 120, timeout_s)
        if answer is None:
            return meta
        keep = answer.get("keep")
        if not isinstance(keep, list):
            return {**meta, "error": "no keep list in the answer", "failure": "error"}
        return {**meta, "keep": sorted({int(k) for k in keep if isinstance(k, (int, float, str)) and str(k).isdigit() and 0 <= int(k) < len(docs)})}
