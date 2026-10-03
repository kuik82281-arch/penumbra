"""Vector index (P2-B): persisted embedding cache + brute-force cosine search over what is currently retrievable.

- Cache: data/embeddings.sqlite, keyed (sourceType, sourceId, providerKey) and checked against the content hash.
  It survives index rebuilds and restarts; an unchanged text under the same provider / model / schema is never
  embedded again. A changed provider key simply finds no rows (invalidation); `purge_stale()` drops them.
- Current set: confirmed EVENTs, active MANUAL, curated memories, active PREFERENCE chunks, and RAW originals.
  Candidates, superseded, archived and excluded items are removed from the matrix the moment they change.
- Embedding runs in one background thread in batches; the chat hot path only ever embeds the query.
- Search: one normalised float32 matrix, dot product, top-N. At personal scale (thousands of rows) this is a few
  milliseconds; an ANN index only replaces `search()` if a benchmark ever says so.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the vector signal is optional
    np = None

from .embeddings import EmbeddingProvider
from .files import content_hash, now_iso

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS embeddings (
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  provider_key TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  dims INTEGER NOT NULL,
  vec BLOB NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (source_type, source_id, provider_key)
);
"""


def embedding_text(kind: str, title: str, body: str, entities: list[str], tags: list[str]) -> str:
    """What gets embedded. Structural fields (ids, hashes, provenance) never enter the semantic body."""
    if kind in ("manual", "preference", "original"):
        return body  # the user's own words / the source text, verbatim
    extra = "、".join([*entities, *tags])
    return f"{title}\n{body}" + (f"\n{extra}" if extra else "")


class VectorStore:
    def __init__(self, data_dir: Path, provider: EmbeddingProvider, batch_size: int = 32):
        self.provider = provider
        self.batch_size = batch_size
        self.path = data_dir / "embeddings.sqlite"
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()
        self.items: dict[str, dict] = {}  # id -> {sourceType, channel, hash, text}
        self.vectors: dict[str, "np.ndarray"] = {}
        self.pending: dict[str, None] = {}
        self._matrix = None
        self._matrix_ids: list[str] = []
        self._dirty = True
        self._wake = threading.Event()
        self._stop = False
        self.stats_counters = {"embedded": 0, "cacheHits": 0, "batches": 0, "embedMs": 0.0, "lastBatchAt": None, "lastReloadMs": None}
        self.enabled = np is not None and provider.provider_id != "none"
        self._thread = None
        # Entries of a reload made before the provider was ready: its cache key includes the vector dimensions, which a
        # local model only knows once loaded, so the cache can only be read after that (see reload()).
        self._reload_when_ready: list[tuple[str, str, str, str]] | None = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, name="penumbra-embedder", daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Stop the embedder, then close the cache. The database is closed under the lock and after the worker has
        left, so a batch that is being written can never meet a closed connection."""
        self._stop = True
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=15)
        with self.lock:
            self.db.close()

    # ------------------------------------------------------------ sync

    def _cached(self, source_type: str, source_id: str, digest: str):
        row = self.db.execute(
            "SELECT content_hash, dims, vec FROM embeddings WHERE source_type = ? AND source_id = ? AND provider_key = ?",
            (source_type, source_id, self.provider.key),
        ).fetchone()
        if row and row[0] == digest:
            return np.frombuffer(row[2], dtype=np.float32)
        return None

    def sync(self, source_type: str, source_id: str, channel: str, text: str, current: bool) -> None:
        """Called for every projected write. Not current -> out of the index; current -> cached vector or queued."""
        if not self.enabled:
            return
        with self.lock:
            if not current or not text.strip():
                if self.items.pop(source_id, None) is not None or source_id in self.vectors:
                    self.vectors.pop(source_id, None)
                    self.pending.pop(source_id, None)
                    self._dirty = True
                return
            digest = content_hash(text)
            known = self.items.get(source_id)
            if known and known["hash"] == digest and (source_id in self.vectors or source_id in self.pending):
                known["channel"] = channel
                return
            self.items[source_id] = {"sourceType": source_type, "channel": channel, "hash": digest, "text": text}
            vec = self._cached(source_type, source_id, digest)
            if vec is not None:
                self.vectors[source_id] = vec
                self.pending.pop(source_id, None)
                self.stats_counters["cacheHits"] += 1
                self._dirty = True
            else:
                self.vectors.pop(source_id, None)
                self._dirty = True
                self.pending[source_id] = None
                self._wake.set()

    def reload(self, entries: list[tuple[str, str, str, str]]) -> None:
        """Full reload after an index rebuild: entries = (sourceType, sourceId, channel, text). Cached vectors are reused."""
        if not self.enabled:
            return
        if not self.provider.ready():
            # The key would read "...:0:..." and match nothing: every vector would be embedded again after each restart,
            # for minutes, while live queries queue behind that work. Wait for the model, then read the cache.
            with self.lock:
                self._reload_when_ready = list(entries)
                self.items.clear()
                self.vectors.clear()
                self.pending.clear()
                for source_type, source_id, channel, text in entries:
                    if text.strip():
                        self.items[source_id] = {"sourceType": source_type, "channel": channel, "hash": content_hash(text), "text": text}
                self._dirty = True
            self._wake.set()
            return
        started = time.perf_counter()
        with self.lock:
            self._reload_when_ready = None
            self.items.clear()
            self.vectors.clear()
            self.pending.clear()
            cache = {
                (r[0], r[1]): (r[2], r[3])
                for r in self.db.execute("SELECT source_type, source_id, content_hash, vec FROM embeddings WHERE provider_key = ?", (self.provider.key,))
            }
            for source_type, source_id, channel, text in entries:
                if not text.strip():
                    continue
                digest = content_hash(text)
                self.items[source_id] = {"sourceType": source_type, "channel": channel, "hash": digest, "text": text}
                hit = cache.get((source_type, source_id))
                if hit and hit[0] == digest:
                    self.vectors[source_id] = np.frombuffer(hit[1], dtype=np.float32)
                    self.stats_counters["cacheHits"] += 1
                else:
                    self.pending[source_id] = None
            self._dirty = True
        self.stats_counters["lastReloadMs"] = round((time.perf_counter() - started) * 1000, 1)
        self._wake.set()

    # ------------------------------------------------------------ background embedding

    def _worker(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=5)
            self._wake.clear()
            while not self._stop and self.provider.ready():
                deferred = self._reload_when_ready
                if deferred is not None:
                    self.reload(deferred)  # the model is ready now: the cache resolves under the right key
                with self.lock:
                    batch = [(i, self.items[i]) for i in list(self.pending)[: self.batch_size] if i in self.items]
                    for i in list(self.pending)[: self.batch_size]:
                        if i not in self.items:
                            self.pending.pop(i, None)
                if not batch:
                    break
                started = time.perf_counter()
                try:
                    vecs = self.provider.embed_documents([item["text"] for _, item in batch])
                except Exception as error:
                    print(f"[penumbra] embedding batch failed: {error!r}")
                    time.sleep(5)
                    break
                elapsed = (time.perf_counter() - started) * 1000
                with self.lock:
                    if self._stop:
                        break
                    for (source_id, item), vec in zip(batch, vecs):
                        arr = np.asarray(vec, dtype=np.float32)
                        self.db.execute(
                            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?,?)",
                            (item["sourceType"], source_id, self.provider.key, item["hash"], int(arr.shape[0]), arr.tobytes(), now_iso()),
                        )
                        current = self.items.get(source_id)
                        if current is not None and current["hash"] == item["hash"]:
                            self.vectors[source_id] = arr
                            self.pending.pop(source_id, None)
                    self._dirty = True
                    c = self.stats_counters
                    c["embedded"] += len(batch)
                    c["batches"] += 1
                    c["embedMs"] += elapsed
                    c["lastBatchAt"] = now_iso()

    def wait_idle(self, timeout: float = 120.0) -> bool:
        """Block until every current item has a vector (tests, rebuild benchmarks)."""
        if not self.enabled:
            return True  # vectors are switched off: nothing will ever be embedded
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.pending and self.provider.ready() and self._reload_when_ready is None:
                return True
            self._wake.set()
            time.sleep(0.05)
        return not self.pending

    # ------------------------------------------------------------ search

    def _refresh(self) -> None:
        if not self._dirty:
            return
        ids = [i for i in self.vectors if i in self.items]
        self._matrix_ids = ids
        self._matrix = np.vstack([self.vectors[i] for i in ids]) if ids else None
        self._dirty = False

    def search(self, query_vec, channels: tuple, depth: int, min_similarity: float) -> list[tuple[str, float]]:
        """Top matches by cosine. `self.last_mean` is the query's mean similarity over everything indexed: a margin over
        it separates a real semantic neighbour from the background similarity every text of a language shares."""
        self.last_mean = None
        if not self.enabled or query_vec is None:
            return []
        with self.lock:
            self._refresh()
            if self._matrix is None:
                return []
            sims = self._matrix @ np.asarray(query_vec, dtype=np.float32)
            self.last_mean = float(sims.mean())
            order = np.argsort(-sims)
            out = []
            for idx in order:
                sim = float(sims[idx])
                if sim < min_similarity:
                    break
                sid = self._matrix_ids[idx]
                if self.items[sid]["channel"] in channels:
                    out.append((sid, sim))
                    if len(out) >= depth:
                        break
            return out

    def similarity(self, query_vec, source_id: str) -> float | None:
        vec = self.vectors.get(source_id)
        if vec is None or query_vec is None:
            return None
        return float(vec @ np.asarray(query_vec, dtype=np.float32))

    def purge_stale(self) -> int:
        with self.lock:
            return self.db.execute("DELETE FROM embeddings WHERE provider_key != ?", (self.provider.key,)).rowcount

    def stats(self) -> dict:
        with self.lock:
            size = self.path.stat().st_size if self.path.exists() else 0
            wal = self.path.with_name(self.path.name + "-wal")
            size += wal.stat().st_size if wal.exists() else 0
            rows = self.db.execute("SELECT COUNT(*) FROM embeddings WHERE provider_key = ?", (self.provider.key,)).fetchone()[0]
            stale = self.db.execute("SELECT COUNT(*) FROM embeddings WHERE provider_key != ?", (self.provider.key,)).fetchone()[0]
            c = self.stats_counters
            return {
                "enabled": self.enabled, "provider": self.provider.status(), "providerKey": self.provider.key,
                "current": len(self.items), "embedded": len(self.vectors), "pending": len(self.pending),
                "cache": {"rows": rows, "staleRows": stale, "bytes": size},
                "work": {**c, "avgMsPerDoc": round(c["embedMs"] / c["embedded"], 2) if c["embedded"] else None},
            }
