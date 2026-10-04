"""Query rewrite: before a turn's search, a small local model reads the last few lines and says what the message is about.

  "那个后来怎么样了" finds nothing by its own words; with the conversation before it, it is "轮盘游戏的结果". The local model
  (Ollama, the same one discovery uses) writes:
    topic     what she is talking about, 8-40 characters, every 那个 / 这个 / 它 resolved, no time words, no pet names
    keywords  2-5 rare words for the lexical search, each copied from the conversation (an invented one is dropped)
    image     she means a picture she sent before

  It only ever adds: retrieve() runs the rewritten query next to the original one and fuses the two (read.py), so a wrong
  rewrite cannot lose what the original finds. It never decides whether to search (the host's entry gate does). Any failure
  (Ollama down, slow, unparsable) is simply no rewrite - the turn searches as before.

  PENUMBRA_QUERY_REWRITE: "context" (default: no model, see mode()) | "on" | "shadow" (rewrite and trace it, search with the
  original only) | "off".
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from .. import prompts
from .llm import OllamaClient

SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "image": {"type": "boolean"},
    },
    "required": ["topic", "keywords", "image"],
}

MAX_LINE = 240


# A message that points back at something said before: it needs the lines before it to be searched for.
REFERENT = re.compile(r"那个|这个|那件|这件|那次|这次|那回|那家|这家|那里|那边|这边|那块|这块|那本|这本|那首|这首|那张|这张|那部|这部|那位|"
                      r"那款|这款|那种|这种|那条|这条|它|刚才说的|刚刚说的|你说的那|上面说的")


def mode() -> str:
    """context (default): no model - a message with a REFERENT searches again with the lines before it; on: the local model
    rewrites (needs Ollama, ~0.25 s warm); shadow: the model rewrites, only traced; off."""
    value = os.environ.get("PENUMBRA_QUERY_REWRITE", "").strip().lower()
    return value if value in ("context", "on", "shadow", "off") else "context"


class QueryRewriter:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        # A chat turn waits for it: short, and the model stays warm between turns (it is small).
        self.client.timeout_s = float(os.environ.get("MEMORY_REWRITE_TIMEOUT", "") or 2.5)
        self.client.keep_alive = os.environ.get("MEMORY_REWRITE_KEEP_ALIVE", "").strip() or "30m"

    def warm(self) -> None:
        """Load the model in the background (a cold load takes seconds; a turn never waits for it)."""
        if getattr(self, "_warming", False):
            return
        self._warming = True

        def run():
            client = OllamaClient()
            client.timeout_s, client.keep_alive = 120, self.client.keep_alive
            try:
                client.chat("只回答 JSON。", "预热", SCHEMA, num_predict=8)
            except Exception:
                pass
            finally:
                self._warming = False
        threading.Thread(target=run, name="rewrite-warm", daemon=True).start()

    def rewrite(self, query: str, recent: list[dict]) -> dict | None:
        """{topic, keywords, image, text, latency_ms} or None (no rewrite: search with the original)."""
        health = self.client.health()
        if not health.get("modelInstalled"):
            return None
        if not health.get("modelLoaded"):
            self.warm()  # this turn searches as before; the next one has the model warm
            return None
        lines = [f"{'她' if str(r.get('role')) == 'user' else '他'}：{str(r.get('content') or '')[:MAX_LINE]}" for r in recent[-6:] if r.get("content")]
        user = "最近的对话：\n" + ("\n".join(lines) if lines else "（没有）") + f"\n\n她刚说：{query[:MAX_LINE]}"
        started = time.perf_counter()
        try:
            content, _meta = self.client.chat(prompts.get("rewrite"), user, SCHEMA, num_predict=160)
            answer = json.loads(content)
        except Exception:
            return None
        topic = re.sub(r"\s+", " ", str(answer.get("topic") or "")).strip()[:60]
        context = query + "\n" + "\n".join(lines)
        keywords = []
        for k in answer.get("keywords") or []:
            word = str(k).strip()
            if 2 <= len(word) <= 12 and word in context and word not in keywords:  # copied from the conversation, never invented
                keywords.append(word)
        keywords = keywords[:5]
        if not topic and not keywords:
            return None
        text = " ".join([topic, *[k for k in keywords if k not in topic]]).strip()
        return {"topic": topic, "keywords": keywords, "image": answer.get("image") is True, "text": text,
                "latency_ms": round((time.perf_counter() - started) * 1000)}
