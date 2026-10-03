"""Penumbra: a long-term memory service for a personal AI companion.

Principles (the architecture's hard rules):
- Originals are immutable and the only truth. Nothing rewrites, compresses or decays them.
- The index (index.sqlite) is disposable: `python -m penumbra rebuild` recreates it from files.
- Memories are curated entries that point back to originals (source trace). Machine authors
  (e.g. DeepSeek) may only ever produce `status: candidate`; they are never injected.
- Every inject and recall is written to an append-only ledger; access counts derive from it.
"""

__version__ = "0.1.0"
