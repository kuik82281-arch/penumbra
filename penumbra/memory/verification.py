"""DeepSeek Verification / Rewrite / Merge / Update Decision.

DeepSeek reads one candidate together with its RAW (the only truth) and, when relevant, the existing Episodes and
Patterns nearest to it. It answers with one of eight decisions (actions.py). It writes Episode / Pattern text itself,
in the user and the assistant's names, only from what the RAW explicitly says. Nothing it returns is applied before local
validation (ids, RAW provenance, times, state history).
"""
from __future__ import annotations

import json

from .. import identity, prompts
from .actions import ALL_ACTIONS
from .discovery import speaker
from .llm import DeepSeekClient

PROMPT_VERSION = "verify-v15"



def _render(record: dict) -> dict:
    return {
        "id": record["id"], "speaker": speaker(record.get("role", "")), "createdAt": record.get("createdAt"), "content": str(record.get("content") or ""),
        **({"kind": "his_day_memory"} if record.get("sourceType") == "assistant_day_memory" else {}),
        "attachments": [{"attachment_id": a.get("attachment_id"), "caption": a.get("caption") or "", "mime_type": a.get("mime_type"),
                         "looks_like": str(a.get("searchable_text") or "")[:400]}
                        for a in record.get("attachments", []) if isinstance(a, dict)],
    }


def build_payload(candidate: dict, raw: list[dict], context: list[dict], patterns: list[dict], episodes: list[dict], now: str,
                  open_commitments: list[dict] | None = None, threads: list[dict] | None = None, rituals: list[dict] | None = None) -> dict:
    return {
        "threads": threads or [],
        "rituals": [{k: r[k] for k in ("episode_id", "tag", "content", "time_end")} | {"times": len(r["source_raw_ids"])} for r in rituals or []],
        "open_commitments": [{k: c[k] for k in ("episode_id", "tag", "content", "owner", "due_at", "status")} for c in open_commitments or []],
        "now": now,
        "candidate": {"candidate_id": candidate["candidate_id"], "kind": candidate.get("kind_hint"), "gist": candidate.get("gist"),
                      "entities": candidate.get("entities"), "found_by": candidate.get("origin")},
        "RAW": [_render(r) for r in raw],
        "context_RAW": [_render(r) for r in context],
        "existing": {
            "patterns": [{"pattern_id": p["pattern_id"], "title": p["title"], "topic": p["topic"], "narrative": p["narrative"],
                          "current_state": p["current_state"], "current_state_since": p["current_state_since"],
                          "historical_states": [{"state": s["state"], "valid_from": s["valid_from"], "valid_to": s["valid_to"]} for s in p["historical_states"]],
                          "supporting_episode_ids": p["supporting_episode_ids"]} for p in patterns],
            "episodes": [{"episode_id": e["episode_id"], "content": e["content"], "time_start": e["time_start"], "time_end": e["time_end"],
                          "state": e["state"], "entities": e["entities"], "source_raw_ids": e["source_raw_ids"][:6], "kind": e.get("kind") or "", "tag": e.get("tag") or "",
                          "patterns": e.get("pattern_ids", [])} for e in episodes],
        },
    }


class Verifier:
    provider = "deepseek"
    prompt_version = PROMPT_VERSION

    def __init__(self, client: DeepSeekClient | None = None):
        self.client = client or DeepSeekClient()

    @property
    def model(self) -> str:
        return self.client.model

    def status(self) -> dict:
        return self.client.status()

    def available(self) -> bool:
        return self.client.available()

    def decide(self, payload: dict) -> tuple[dict, dict]:
        """(answer, metadata). Raises WorkerError when DeepSeek is unavailable or unusable."""
        return self.client.chat_json(prompts.get("verify"), json.dumps(payload, ensure_ascii=False), max_tokens=3500)


__all__ = ["Verifier", "build_payload", "PROMPT_VERSION", "ALL_ACTIONS"]
