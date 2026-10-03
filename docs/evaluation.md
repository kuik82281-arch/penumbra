# Evaluation

```bash
python -m penumbra.eval            # score the existing evaluation state (free, seconds)
python -m penumbra.eval --rebuild  # re-curate the corpus with the real LLM first (about one call per segment)
```

## Corpus

`eval/corpus.json` is a fictional half year of chat between a user and an AI companion: 56 segments, each separated
from the next by far more than 30 minutes. Every planted fact is listed under `plants`, for example:

- a preference that **changes** (dislikes coriander → likes it at hotpot) and one that was **misremembered**
  (favourite colour, corrected later);
- a job hunt across five conversations (a **thread**), with an unrelated driving test in the same weeks that must not
  join it;
- a week of cleaning, a novel written over months, a gym habit with a break, a calculus course with daily steps that
  overlap the day before (**progress** that must not be dropped as repetition);
- a nightly check-in (**one ritual**, not five) with one different night (its own memory);
- a week of meals (**nothing to remember**) with one restaurant she will "eat at every day" (remembered);
- a deleted memory that must not come back, a promise kept and one missed, a vow, three nicknames / in-jokes.

## Questions and structural checks

`eval/questions.json` holds recall questions (each must surface given words within the top memories, and must not
surface others) and structural checks over the curated store, e.g. "a thread holds the job hunt and not the driving
test", "four study days are four steps", "the nightly ritual is one memory", "ordinary meals did not become memories".

The recall questions run through the same retrieve path the service uses at runtime.
