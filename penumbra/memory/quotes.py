"""Verified quotes: a memory may only put in quotation marks what was really said.

Every quoted span (“…” 「…」 『…』 "…" ‘…’) in the text a decision writes is looked up in the RAW it read (and in the
existing memories it was shown, for what an earlier memory already quoted). Compared after NFKC, without spaces and
punctuation, as a substring. Very short quotes (a word or two) are not checked. A decision with a quote that is
nowhere is not applied: it goes to QUARANTINE for the user to look at, so a sentence she never said cannot become
"her words" in his memory.
"""
from __future__ import annotations

import re
import unicodedata

QUOTED = re.compile(r"“([^”]{1,300})”|「([^」]{1,300})」|『([^』]{1,300})』|\"([^\"]{1,300})\"|‘([^’]{1,300})’")
MIN_CHARS = 4  # after normalization; "嗯嗯" / "好呀" are not checked


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in text if not (ch.isspace() or unicodedata.category(ch)[0] in "PSZ"))


def quotes_in(text: str) -> list[str]:
    return [next(g for g in m.groups() if g is not None) for m in QUOTED.finditer(text or "")]


def unverified_quotes(texts: list[str], sources: list[str]) -> list[str]:
    """Quoted spans in `texts` that appear in none of `sources`."""
    haystack = "\n".join(_norm(s) for s in sources)
    out: list[str] = []
    for text in texts:
        for q in quotes_in(text):
            n = _norm(q)
            if len(n) >= MIN_CHARS and n not in haystack and q not in out:
                out.append(q)
    return out


def plan_texts(actions: list[dict]) -> list[str]:
    """The prose a validated plan would write."""
    out: list[str] = []
    for a in actions:
        kind = a.get("action")
        if kind in ("CREATE_EPISODE", "MERGE_EPISODE"):
            out += [a.get("content", ""), a.get("state", ""), a.get("excerpt", "")]
        elif kind == "UPDATE_EPISODE":
            patch = a.get("patch") or {}
            out += [patch.get("content", ""), patch.get("state", ""), patch.get("excerpt", "")]
        elif kind == "CREATE_PATTERN":
            out += [a.get("narrative", ""), a.get("current_state", ""), *[s.get("state", "") for s in a.get("earlier_states") or []]]
        elif kind == "UPDATE_PATTERN":
            out += [a.get("narrative", ""), (a.get("new_state") or {}).get("state", "")]
    return [t for t in out if t]
