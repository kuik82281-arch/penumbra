# Configuration

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PENUMBRA_DATA` | `./data` | data directory (originals, memory database, indexes) |
| `PENUMBRA_HOST` | `127.0.0.1` | address to listen on; the Docker image sets `0.0.0.0` inside the container |
| `PENUMBRA_PORT` | `8790` | HTTP port |
| `DEEPSEEK_API_KEY` | — | key for the curation LLM; without it, segments wait in STAGING |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | any OpenAI-compatible chat completions endpoint |
| `DEEPSEEK_MODEL` | `deepseek-chat` | model name |
| `PENUMBRA_EMBEDDING` | auto | unset: the local bge-m3 in `./models/bge-m3` when present, otherwise none; `none` disables vectors (lexical + entity only) |
| `PENUMBRA_EMBEDDING_DEVICE` | auto | `cpu` / `cuda` |
| `MEMORY_RERANKER` / `MEMORY_RERANKER_PATH` | on / `./models/bge-reranker-v2-m3` | cross-encoder used for re-checks; `0` turns it off |
| `MEMORY_RERANK_FLOOR` | `0.02` | below this score a meaning-only candidate is dropped after re-check |
| `PENUMBRA_COOLDOWN_HOURS` | `24` | a recalled memory is not brought up again by itself for this long |
| `PENUMBRA_SEEN_TTL_HOURS` | `6` | seen suppression within one session |
| `PENUMBRA_SEGMENTATION` | `session` | `session` (conversation segments) or `discovery` (span discovery with a small local model) |
| `PENUMBRA_USER_NAME` / `PENUMBRA_ASSISTANT_NAME` | `User` / `AI` | the two names, overriding the profile |

## Profile (`<data>/profile.json`)

```json
{
 "user": "Mia",
 "assistant": "Echo",
 "user_pronoun": "她",
 "assistant_pronoun": "他",
 "aliases": {"user": [], "assistant": []},
 "preference_labels": [],
 "filler_words": []
}
```

- `user` / `assistant`: the names memories use (prompts are written with `{{USER}}` / `{{AI}}` placeholders).
- `aliases`: other words a client sends for each role as actor / author / owner.
- `preference_labels`: extra labels for preference documents beyond the built-in set.
- `filler_words`: the pet names and calls the two of them use. A message made mostly of these (plus particles) has no
  topic of its own, so retrieval searches with the lines before it instead of finding every tender memory.

The memory gate (DeepSeek before and after the search) is configured by `PENUMBRA_MEMORY_GATE` (on | judge | intent |
off), `MEMORY_GATE_BUDGET` (seconds for the whole retrieval, default 5.5) and `MEMORY_GATE_TIMEOUT` (one call, default 2).

## Prompts

`penumbra/prompts/default/*.md` ship with the code. Put a file with the same name in `penumbra/prompts/private/`
(git-ignored) to use your own; it takes precedence. Prompts are read once per process.

| File | Used by |
|---|---|
| `verify.md` | curation of a segment (the main prompt; its output format is enforced by `memory/actions.py`) |
| `discover.md` | optional span discovery with a small local model (`PENUMBRA_SEGMENTATION=discovery`) |
| `dream.md`, `her_dream.md` | summaries of dreams handed over by the host |
| `story.md` | the narrative of a closed thread |
| `profile.md` | the profile of a name in the name list |
