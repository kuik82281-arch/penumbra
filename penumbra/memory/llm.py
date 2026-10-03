"""The two model clients of the memory pipeline - each with exactly one job.

Ollama (local, qwen2.5:3b)  Candidate Discovery only: it points at chat passages that might deserve a memory.
                            It never writes memory text and never talks to the long-term store.
DeepSeek                    Verification / Rewrite / Merge / Update Decision, reading the candidate and its RAW.

Both are called with a hard schema, bounded budgets, tolerant-but-honest parsing (extract, never repair) and
metadata-only logs: no API key, no memory text, only status / lengths / positions.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from ..errors import WorkerError

# ------------------------------------------------------------ answer parsing (extract, never repair)


class AnswerError(Exception):
    """A model answer that cannot be used as-is. kind: empty | truncated | malformed | refused."""

    def __init__(self, kind: str, detail: str, position: int | None = None):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.position = position


def balanced_end(text: str, start: int) -> int | None:
    """Index just past the bracket closing the '{' / '[' at `start`, string-aware; None if it never closes."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)(?:```|$)", re.S)


def parse_json_answer(content: str | None, finish_reason: str | None = None) -> dict:
    """The JSON object in a model answer. Extracts from fences / surrounding prose; never adds quotes or brackets."""
    text = (content or "").lstrip("﻿").strip()
    if not text:
        raise AnswerError("empty", f"no content (finish_reason {finish_reason or 'none'})", 0)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise AnswerError("refused", f"no json object in a {len(text)}-char answer", 0)
    end = balanced_end(text, start)
    if end is None:
        raise AnswerError("truncated", f"object never closes ({len(text)} chars, finish_reason {finish_reason or 'none'})", len(text))
    if finish_reason == "length":
        raise AnswerError("truncated", "finish_reason length", len(text))
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as error:
        raise AnswerError("malformed", f"{error.msg} at char {start + error.pos}", start + error.pos) from None
    if not isinstance(data, dict):
        raise AnswerError("malformed", "top level is not an object", start)
    return data


def complete_objects(content: str | None, array_key: str) -> list[dict]:
    """The complete `{...}` elements of `"array_key": [ ... ]` in a possibly cut-off answer.

    Used only for candidate discovery, where each element is an independent pointer that is verified later anyway:
    what the model finished writing is kept, the cut-off tail is dropped. Nothing is patched or completed."""
    text = content or ""
    key = text.find(f'"{array_key}"')
    if key < 0:
        return []
    bracket = text.find("[", key)
    if bracket < 0:
        return []
    out, i = [], bracket + 1
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        end = balanced_end(text, i)
        if end is None:
            break
        try:
            item = json.loads(text[i:end])
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            out.append(item)
        i = end
    return out


# ------------------------------------------------------------ DeepSeek


class DeepSeekClient:
    provider = "deepseek"
    # Whether this endpoint accepted {"thinking": {"type": "disabled"}}; set False once it rejects the field.
    _thinking_off = True

    def __init__(self, timeout_s: float = 150.0, max_retries: int = 2):
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.last_error: str | None = None
        self.last_ok_at: str | None = None

    # The environment is read on every call: the key is never cached on the object or written anywhere.
    @property
    def model(self) -> str:
        return os.environ.get("DEEPSEEK_MODEL", "").strip() or "deepseek-chat"

    def _base(self) -> str:
        return (os.environ.get("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com").rstrip("/")

    def available(self) -> bool:
        return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())

    def status(self) -> dict:
        return {"provider": self.provider, "model": self.model, "baseUrl": self._base(), "configured": self.available(),
                "ready": self.available(), "lastError": self.last_error, "lastOkAt": self.last_ok_at,
                **({} if self.available() else {"error": "DEEPSEEK_API_KEY is not available in this process"})}

    def _request(self, system: str, user: str, max_tokens: int) -> dict:
        """One HTTP round trip (network / 429 / 5xx retried with backoff). Returns content + metadata."""
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2 * attempt, 6))
            thinking_off = self._thinking_off
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": max_tokens if thinking_off else max(max_tokens, 16_000),
            }
            if thinking_off:
                body["thinking"] = {"type": "disabled"}
            request = urllib.request.Request(
                f"{self._base()}/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("DEEPSEEK_API_KEY", "").strip()}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    status = response.status
                    payload = json.loads(response.read().decode("utf-8", "replace"))
                choice = payload["choices"][0]
                usage = payload.get("usage") or {}
                return {"content": (choice.get("message") or {}).get("content"), "finish_reason": choice.get("finish_reason"), "status": status,
                        "usage": {"inputTokens": int(usage.get("prompt_tokens") or 0), "outputTokens": int(usage.get("completion_tokens") or 0)},
                        "thinking_disabled": thinking_off, "attempts": attempt + 1}
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:200]
                last = f"HTTP {error.code}: {detail}"
                if error.code == 400 and thinking_off and "thinking" in detail.lower():
                    DeepSeekClient._thinking_off = False  # this endpoint does not take the field: continue without it
                    continue
                if error.code in (400, 401, 402, 403, 404, 422):
                    raise WorkerError(last, attempt) from None  # retrying cannot fix these
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last = f"network: {getattr(error, 'reason', error)}"
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                last = "unreadable API response"
        raise WorkerError(last, self.max_retries)

    def chat_json(self, system: str, user: str, max_tokens: int = 4000) -> tuple[dict, dict]:
        """(answer object, metadata). An empty / cut-off / malformed answer gets one repair retry with the same input."""
        if not self.available():
            raise WorkerError("DeepSeek is not configured (DEEPSEEK_API_KEY is missing)")
        note = ""
        meta: dict = {}
        for attempt in range(2):
            try:
                result = self._request(system, user + note, max_tokens)
            except WorkerError as error:
                self.last_error = str(error)[:300]
                raise
            meta = {k: v for k, v in result.items() if k != "content"} | {"chars": len(result["content"] or ""), "format_attempts": attempt + 1}
            try:
                data = parse_json_answer(result["content"], result["finish_reason"])
                self.last_error, self.last_ok_at = None, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return data, meta
            except AnswerError as error:
                meta["parse_error"] = f"{error.kind} at {error.position}"
                self.last_error = f"unusable answer ({error.kind})"
                note = f"\n\n上一次的输出无法作为 json 使用（{error.kind}）。请只输出一个完整的 json 对象，格式与要求相同。"
                if error.kind == "truncated":
                    max_tokens = min(max_tokens * 2, 12_000)
        raise WorkerError(f"DeepSeek answer unusable after a repair retry ({meta.get('parse_error')})", 1)


# ------------------------------------------------------------ Ollama


class OllamaClient:
    provider = "ollama"

    def __init__(self):
        self.model = os.environ.get("MEMORY_OLLAMA_MODEL", "").strip() or "qwen2.5:3b"
        base = os.environ.get("OLLAMA_HOST", "").strip() or "http://127.0.0.1:11434"
        self.base = (base if base.startswith("http") else "http://" + base).rstrip("/")
        self.timeout_s = float(os.environ.get("MEMORY_OLLAMA_TIMEOUT", "") or 90)
        self.num_ctx = int(os.environ.get("MEMORY_OLLAMA_NUM_CTX", "") or 8192)
        # How long Ollama keeps the model after each call. It only has to bridge the gaps inside one run (DeepSeek
        # verifies a window's candidates before the next window is discovered); the run itself unloads the model when
        # it ends (unload()), so this is the fallback if a run never gets there.
        self.keep_alive = os.environ.get("MEMORY_OLLAMA_KEEP_ALIVE", "").strip() or "10m"
        self.used = False  # a chat() since the last unload(): the model may be resident
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local service: never through a proxy
        self._health: dict | None = None
        self._health_at = 0.0
        self.last_error: str | None = None
        self.last_ok_at: str | None = None

    def _get(self, path: str, timeout: float = 3.0) -> dict:
        with self._opener.open(self.base + path, timeout=timeout) as response:
            return json.load(response)

    def health(self, force: bool = False, max_age_s: float = 20.0) -> dict:
        """Server reachable, model installed, and whether it is loaded right now. Cached briefly (a chat turn never waits)."""
        if not force and self._health is not None and time.time() - self._health_at < max_age_s:
            return self._health
        started = time.perf_counter()
        result = {"provider": self.provider, "model": self.model, "baseUrl": self.base, "ready": False, "reachable": False,
                  "modelInstalled": False, "modelLoaded": False, "error": None, "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            names = {m.get("name") or m.get("model") for m in self._get("/api/tags").get("models", [])}
            result["reachable"] = True
            result["installedModels"] = sorted(n for n in names if n)
            wanted = {self.model, self.model + ":latest"} if ":" not in self.model else {self.model}
            result["modelInstalled"] = bool(wanted & names)
            if not result["modelInstalled"]:
                result["error"] = f"model {self.model} is not installed (ollama pull {self.model})"
            else:
                try:
                    running = {m.get("name") or m.get("model") for m in self._get("/api/ps").get("models", [])}
                    result["modelLoaded"] = bool(wanted & running)
                except Exception:
                    pass  # /api/ps is informational
                result["ready"] = True
        except Exception as error:
            result["error"] = f"Ollama is not reachable at {self.base}: {getattr(error, 'reason', error)}"
        result["latencyMs"] = round((time.perf_counter() - started) * 1000)
        self._health, self._health_at = result, time.time()
        return result

    def status(self, probe: bool = True) -> dict:
        info = dict(self.health(force=probe) if probe else (self._health or {"provider": self.provider, "model": self.model, "baseUrl": self.base, "ready": None}))
        info["lastError"] = self.last_error
        info["lastOkAt"] = self.last_ok_at
        return info

    def chat(self, system: str, user: str, schema: dict, num_predict: int = 700) -> tuple[str, dict]:
        """One structured completion. Returns the raw content (parsed by the caller) and metadata."""
        body = {
            "model": self.model, "stream": False, "format": schema, "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": num_predict, "repeat_penalty": 1.15},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        request = urllib.request.Request(self.base + "/api/chat", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        self.used = True  # even a call that times out may have loaded the model
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200]
            self.last_error = f"HTTP {error.code}: {detail}"
            self._health = None
            raise WorkerError(f"Ollama {self.last_error}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            self.last_error = f"Ollama request failed: {getattr(error, 'reason', error)}"
            self._health = None
            raise WorkerError(self.last_error) from None
        self.last_error, self.last_ok_at = None, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta = {"latency_ms": round((time.perf_counter() - started) * 1000), "eval_count": payload.get("eval_count"),
                "prompt_eval_count": payload.get("prompt_eval_count"), "done_reason": payload.get("done_reason"),
                "load_ms": round((payload.get("load_duration") or 0) / 1e6)}
        return (payload.get("message") or {}).get("content") or "", meta

    def unload(self) -> bool:
        """Release this model from Ollama's memory now (a request with keep_alive 0 - Ollama's own unload). Only this
        model: whatever else Ollama serves is untouched. The next chat() loads it again."""
        body = {"model": self.model, "keep_alive": 0}
        request = urllib.request.Request(self.base + "/api/generate", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=30) as response:
                json.load(response)
        except Exception as error:  # Ollama down or busy: the keep_alive above still expires on its own
            print(f"[penumbra] ollama unload of {self.model} failed: {getattr(error, 'reason', error)}", flush=True)
            return False
        self.used = False
        self._health = None
        return True
