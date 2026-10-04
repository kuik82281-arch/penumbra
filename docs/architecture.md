# Architecture

![Penumbra architecture](images/architecture.svg)

Penumbra has two paths that meet at one SQLite store: a **write path** that turns raw messages into curated memory in
the background, and a **recall path** that runs once per conversation turn.

## Data model

| Layer | What it is | Mutability |
|---|---|---|
| **Original** | one chat message (or an attachment / a note), with provenance | append-only; an identical resend is idempotent, a changed text is refused |
| **Episode** | one concrete thing that happened, two to four sentences, citing the originals it came from, with a one-line excerpt | versioned updates; deletion leaves a tombstone |
| **Pattern** | a long-term state of one subject: current state, earlier states with validity ranges, supporting episodes | versioned; a *change* appends history, a *correction* replaces the wrong state without keeping it |
| **Relation** | `because_of` between two episodes, only when the source states the cause outright; recalled episodes carry their causes and effects one step away | added by curation, validated against existing ids |
| **Name** | a user-curated name with aliases and a short neutral profile, rewritten by the LLM when the memories that mention it change | user-owned list; profile maintained in the background |
| **Thread** | a story across conversations (an exam season, a job hunt): episodes in time order, written up as a narrative once closed | links are `auto` (confidence ≥ 0.85), `pending` (the user decides) or `rejected` (remembered, never re-proposed) |

Episode kinds: experience, `commitment` (open → done / cancelled / missed, then it sinks), `vow` (never sinks),
`lexicon` (nicknames, in-jokes: one memory per word), `ritual` (one memory per routine; each new occurrence adds evidence
and moves `time_end`), `dream` (a summary pointing at the dream text kept by the host; never evidence for anything real).

## Write path

1. **Originals** arrive over HTTP from the host application, with attachments (and, when available, a textual
   description of each image).
2. **Segmentation.** A segment is a conversation between silences of 30 minutes; a segment that resumes an interrupted
   topic carries the last six messages of the previous one as context. Over-long segments split at their longest pause.
   Segmentation is deterministic: no model decides boundaries.
3. **Curation.** One LLM call per segment reads the raw messages, the nearest existing memories (hybrid retrieval), the
   open threads that moved in the last ten days, known routines and open commitments, and returns a list of actions
   (`CREATE_EPISODE`, `UPDATE_EPISODE`, `MERGE_EPISODE`, `CREATE_PATTERN`, `UPDATE_PATTERN`, `NO_ACTION`, `REJECT`,
   `QUARANTINE`). The rule it applies: record what is **new** relative to what is already remembered.
4. **Validation** (`memory/actions.py`). The answer is checked as a whole: schema, ids that must exist, evidence ids that
   must belong to the segment, **verbatim quotes** (`memory/quotes.py`: every quoted span must occur in the source), and
   **tombstones** (an episode drawn only from originals of a deleted episode is not created again). Anything that does
   not hold sends the whole decision to QUARANTINE for review; nothing partial is applied.
5. **Store.** Episodes, patterns and thread links are written in one transaction; the retrieval index (BM25 fields,
   vectors, entity lexicon) is projected from the store, never the other way round.

## Recall path

Run once per turn and locked to the turn id ("search once"): the assistant can expand what was found, not search
again in the same turn.

1. **Entry gate** (host side): greetings, routine words, bare emoji and very short messages do not search; a question
   about the past always does; a message with a picture always does, with the picture's description in the query.
2. **Query understanding.** A message that points back (那个, 这块, 它 ...) is searched once more with the two lines
   before it, and the two searches are fused (found by both rises; found only with the context is discounted). Time expressions (`5月底`, `上个月初`, `三个月前`) become date windows. A question about
   talking ("上个月我们聊过什么") matches the window against when the source messages were written; any other question
   against when the event happened, so "I said in February that I'd go in April" is found by February-said and April-happened,
   not by February-happened. Stop words, and bigrams that only exist because they straddle a stop word, are dropped.
3. **Hybrid retrieval.** BM25 over title / tag / entity / content fields, dense vectors (bge-m3), and entity matching,
   fused by reciprocal rank. The user's **name list** (`memory/names.py`) turns any alias into a precise hit.
4. **Gates.** Per-hit thresholds (term coverage, vector similarity and margin); patterns first (an episode hit stands for
   the patterns it supports); **seen** suppression within a session; a **cooldown** across sessions (24 h, lifted when
   the user asks about the past or names something); de-duplication against what the session context already carries;
   a **cross-encoder** re-check when the order is ambiguous or the evidence is meaning-only.
5. **Delivery.** At most two memories, patterns first with their supporting episodes, each with its thread position
   (the step before and after), its causes and effects one step away, and its attached images by reference. A named
   entity's profile comes first and does not count toward the two.

## Review

- **Audit and versions** for every change; provenance from any pattern down to the originals.
- **Quarantine** and a review queue for decisions the validator refused or the model was unsure about.
- **Mistake set** (`memory/mistakes.py`): a recall the user marks wrong becomes a question that is replayed through the
  live recall path on demand.
- **Evaluation** (`docs/evaluation.md`): a synthetic half-year corpus with questions and structural checks.

## What it deliberately does not do

Rewrite originals · invent a quote · resurrect a deleted memory · treat a correction as a change · treat a dream as a
fact · put a pile of memories into every turn · repeat what was just recalled · degrade silently when the service is
down · record every meal · record a routine once per day · drop a day of progress because it looks like yesterday.
