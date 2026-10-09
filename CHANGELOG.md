# Changelog

## v0.4.0 — 2026-10-09

- Memory gate (`memory/gate.py`, prompts `intent.md` / `judge.md`): before the search, DeepSeek reads the message and
  the lines before it and says whether long-term memory is needed at all (none / maybe / yes) and what is really being
  looked for, which is searched as well; after the ranking it keeps only the candidates that really help this reply.
  `PENUMBRA_MEMORY_GATE` = on | judge | intent | off.
- Conservative degradation: a failing gate never falls back to "inject what the ranking found". Without an answer, a
  message that does not lean on the past gets no memory; one that does still searches, and keeps only candidates that
  words, names or entities agree on; a history question left with nothing returns `retry_hint`. One attempt per call
  within a 5.5 s budget (`MEMORY_GATE_BUDGET`, `MEMORY_GATE_TIMEOUT`), and after two timeouts in a row DeepSeek is not
  asked for a minute. Counts: `GET /memory-core/gate-stats`.
- Evaluation without self-grading: `python -m penumbra.turn_eval --labeled <file>` scores replayed messages against
  fixed hand labels (need, acceptable memories, needed memories): irrelevant recall, effective recall by category, and
  every needed case that was blocked and where. Memories written after a replayed message are counted apart.
- No second copy of the same day: Episodes from around the same time reach verification straight from the store (the
  index follows writes in the background, so one applied seconds earlier in the same run was missed); the model still
  decides between update and nothing new. A new plain Episode whose originals mostly back an existing one folds into it.
- Retrieval: bracketed stage directions are not a topic; a message that is mostly calls and particles searches with
  the lines before it (the deployment's own pet names: `filler_words` in profile.json); a memory that only shares a word
  must also be close in meaning.
- Dream summaries can be rewritten (a new version of the same Episode) and are a real compression.

## v0.3.0 — 2026-10-06

- The night pond: `GET /pond` serves a three.js page showing this service's own memory as petals on a pond at night,
  under cherry trees in the rain. Every Episode is a petal on the water; a beam of light (tap to move it) brightens the
  petals beneath it, and tapping one reads its excerpt and full account. Recently recalled petals shimmer. Light and
  wind can be switched off. Nothing but `GET /memory-core` is read.
- `GET /memory-core/pipeline`: whether the curation is getting done — last run and success, unprocessed originals,
  staging due, and the scheduler's last error — so a caller can raise an alarm when work piles up and nothing comes out.

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
