# Contributing

Thanks for taking a look.

- **Tests first.** `python -m pytest -q` must pass. Unit tests never call a network or load a model (the LLM, the
  embedding model and the reranker are faked in `tests/support.py`).
- **Behaviour changes to curation or recall** should come with an evaluation case: a segment in `eval/corpus.json` and
  a question or structural check in `eval/questions.json` (see `docs/evaluation.md`).
- **Raw messages stay immutable.** Nothing may rewrite an original; memories change through versioned updates.
- **Prompts** live in `penumbra/prompts/default/`. A deployment can put its own in `penumbra/prompts/private/`
  (git-ignored) without touching code.
- Keep changes small and say why in the commit message.
