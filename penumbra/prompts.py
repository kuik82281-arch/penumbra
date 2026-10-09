"""The model prompts, as files.

    penumbra/prompts/private/<name>.md   a deployment's own, tuned prompts (never part of the public edition)
    penumbra/prompts/default/<name>.md   the generic prompts that ship with the code

A private prompt wins when it exists. Either may use {{USER}} / {{AI}} / {{SHE}} / {{HE}}, filled in from identity.py.
Prompts are read once per process; a changed file takes effect on restart.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import identity

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
NAMES = ("verify", "discover", "dream", "her_dream", "story", "rewrite", "intent", "judge")


@lru_cache(maxsize=None)
def _text(name: str) -> str:
    for folder in ("private", "default"):
        path = PROMPTS_DIR / folder / f"{name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no prompt named {name!r} in {PROMPTS_DIR}")


def source(name: str) -> str:
    """'private' or 'default': which prompt this deployment runs."""
    return "private" if (PROMPTS_DIR / "private" / f"{name}.md").exists() else "default"


def get(name: str) -> str:
    """The prompt with this deployment's names in it."""
    return identity.render(_text(name))
