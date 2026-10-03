"""Who the two people in the memory are: the human and the assistant, by the names the memory uses for them.

Every prompt is written with the placeholders {{USER}} and {{AI}} (and {{SHE}} / {{HE}} for their pronouns); `render`
fills them in. The names come from
`<data_dir>/profile.json` ({"user": "...", "assistant": "..."}) or the PENUMBRA_USER_NAME / PENUMBRA_ASSISTANT_NAME
environment variables; without either, a neutral pair is used.

Roles on the wire are not names: a message's role is "user" / "assistant", an editor (actor) is "user". A client that sends
other words there declares them as aliases in the profile; `is_user` / `is_assistant` accept them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_USER = "User"
DEFAULT_ASSISTANT = "AI"
USER_ACTOR = "user"


@dataclass(frozen=True)
class Identity:
    user: str = DEFAULT_USER
    assistant: str = DEFAULT_ASSISTANT
    examples: dict = field(default_factory=dict)
    # Chinese third-person pronouns for each of them in the prompts ({{SHE}} the user, {{HE}} the assistant).
    user_pronoun: str = "她"
    assistant_pronoun: str = "他"
    # Other words a client may use for each role (e.g. the names an older client sent as actor / author / owner).
    user_aliases: tuple = ()
    assistant_aliases: tuple = ()
    # Preference labels this deployment uses beyond the built-in ones (preferences.py).
    extra_labels: tuple = ()


_current = Identity()


def load(data_dir: str | os.PathLike | None) -> Identity:
    """Read the profile (data dir) and the environment; set and return the current identity."""
    global _current
    profile: dict = {}
    if data_dir:
        path = Path(data_dir) / "profile.json"
        if path.exists():
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                profile = {}
    user = os.environ.get("PENUMBRA_USER_NAME") or profile.get("user") or DEFAULT_USER
    assistant = os.environ.get("PENUMBRA_ASSISTANT_NAME") or profile.get("assistant") or DEFAULT_ASSISTANT
    examples = profile.get("examples") if isinstance(profile.get("examples"), dict) else {}
    aliases = profile.get("aliases") if isinstance(profile.get("aliases"), dict) else {}
    _current = Identity(str(user), str(assistant), {str(k): str(v) for k, v in examples.items()},
                        str(profile.get("user_pronoun") or "她"), str(profile.get("assistant_pronoun") or "他"),
                        tuple(str(a).lower() for a in aliases.get("user", [])), tuple(str(a).lower() for a in aliases.get("assistant", [])),
                        tuple(str(x) for x in profile.get("preference_labels", []) if str(x).strip()))
    return _current


def current() -> Identity:
    return _current


def user() -> str:
    return _current.user


def assistant() -> str:
    return _current.assistant


def example(key: str, default: str) -> str:
    """An illustrative example sentence for a prompt: the profile's own, or the neutral default."""
    return _current.examples.get(key, default)


def render(text: str) -> str:
    """A prompt with the names and pronouns filled in."""
    return (text.replace("{{USER}}", _current.user).replace("{{AI}}", _current.assistant)
            .replace("{{SHE}}", _current.user_pronoun).replace("{{HE}}", _current.assistant_pronoun))


def is_user(value: str | None) -> bool:
    """An actor / author / source that means the human."""
    v = str(value or "").strip().lower()
    return v == USER_ACTOR or v == _current.user.lower() or v in _current.user_aliases


def is_assistant(value: str | None) -> bool:
    v = str(value or "").strip().lower()
    return v == "assistant" or v == _current.assistant.lower() or v in _current.assistant_aliases
