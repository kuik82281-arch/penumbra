# Changelog

## v0.2.0 — 2026-10-04

- Excerpts: every Episode also gets a one-line excerpt (what it was and what it meant); the full account stays in its
  content. Story threads are judged and walked by excerpts, a closed thread's story is drawn from them, and recall
  carries the most relevant Episode whole and the rest by excerpt. Search still indexes the full account (an A/B on the
  same memories: excerpt as the search title lost two questions).
- Messages that point back: a message with a referent (那个 / 这块 / 它 / 那次 ...) searches once more with the two lines
  before it, fused with the original search, so a clear message searches exactly as before. No model is needed; an
  optional local-model rewrite (PENOMBRE_QUERY_REWRITE=on) scored the same on the evaluation.
- Evaluation: questions can carry the lines before them; a 指代 category.
- The reranker retries a racing first import.

## v0.1.0 — 2026-10-03

First public release.

- Write path: append-only originals, conversation segmentation, LLM curation that keeps only what is new, whole-decision validation (verbatim quotes, tombstones, quarantine).
- Memory model: Episodes, Patterns (changes vs corrections), story Threads, cause-and-effect relations, a name list with profiles; commitments, rituals, lexicon and dreams as kinds.
- Recall path: entry gate, time windows (when it was said vs when it happened), hybrid lexical + vector + entity retrieval, gates and cooldowns, at most a couple of memories per turn.
- Review: audit and versions, review queue, mistake set, a synthetic evaluation corpus.
- Docker image and docker-compose for a one-command start.
- License: AGPL-3.0-or-later (earlier snapshots published the same day were MIT).
