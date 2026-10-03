# HTTP API

Local only (`127.0.0.1`), JSON in and out. The complete list is in the docstring of `penumbra/api.py`.

## Write

| | |
|---|---|
| `POST /originals` | `{source, conversationId, items[{id, role, content, createdAt, attachments?}], context?}` → `{added, existing, conflicts, skipped}`. Idempotent; a changed text for an existing id is refused. |
| `POST /memory-core/note` | a memory the user writes directly (stored as an original and cited verbatim) |
| `POST /memory-core/dreams` | `{dreamId, text, time, dreamer?}` → a dream summary pointing at `dream:<id>` |
| `POST /memory-core/run` | run the curation pipeline now (normally scheduled) |

## Recall

| | |
|---|---|
| `POST /memory-core/retrieve` | `{query, turnId, conversationId, sessionId?, recent?}` → `LOCKED` with `patterns`, `episodes`, `refs`, or `NO_MEMORY_NEEDED`. One search per turn. |
| `POST /memory-core/confirm` | `{injectId, conversationId, sessionId, refs}`: what the finished turn actually carried (seen / cooldown) |
| `POST /memory-core/recall` | expand within the turn: `pattern_id`, `episode_id` (with its originals and attachments), `raw_ids`, or one `query` when nothing was locked |

## Review and maintenance

| | |
|---|---|
| `GET /memory-core` | snapshot: counts, patterns, episodes, candidates, runs, audit, recent retrievals |
| `GET /memory-core/provenance/<pattern\|episode>/<id>` | pattern → episodes → decision → originals |
| `POST /memory-core/edit` | user edits: delete (tombstoned), correct, convert a commitment, move an attachment |
| `POST /memory-core/review/<candidate>` | resolve a quarantined decision |
| `GET /memory-core/threads`, `POST /memory-core/threads/decide`, `POST /memory-core/threads/story/<id>` | story threads |
| `GET/POST /memory-core/names`, `POST /memory-core/names/delete` | the name list |
| `GET/POST /memory-core/mistakes`, `POST /memory-core/mistakes/run`, `POST /memory-core/mistakes/delete` | the mistake set |
