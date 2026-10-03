"""Embedding providers (P2-B). The retrieval core only sees this interface; no provider is hard-wired into it.

  provider_id / model_id / dimensions / schema_version
  embed_query(text)        -> list[float]   (one short query, on the chat hot path)
  embed_documents(texts)   -> list[list]    (batches, in the background)
  ready() / status()

Chosen by PENUMBRA_EMBEDDING:
  (unset)            the local BAAI/bge-m3 in ./models/bge-m3 if it is there, else none
  none               no vector signal; retrieval runs lexical + entity
  local:<model>      a sentence-embedding model run locally with transformers (e.g. local:BAAI/bge-m3); the text never
                     leaves this machine. Loaded in a background thread; until it is ready the vector signal is skipped.
  hashing            deterministic character n-gram hashing; tests and plumbing only, it has no semantics
All vectors are L2-normalised, so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from pathlib import Path

from .text import normalize

SCHEMA_VERSION = "emb-v1"


class EmbeddingProvider:
    provider_id = "none"
    model_id = "none"
    dimensions = 0
    schema_version = SCHEMA_VERSION

    @property
    def key(self) -> str:
        """Cache identity of this provider: a change of provider, model, dimensions or schema invalidates every vector."""
        return f"{self.provider_id}:{self.model_id}:{self.dimensions}:{self.schema_version}"

    def ready(self) -> bool:
        return False

    def status(self) -> dict:
        return {"provider": self.provider_id, "model": self.model_id, "dimensions": self.dimensions, "schemaVersion": self.schema_version,
                "ready": self.ready()}

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class NoEmbeddings(EmbeddingProvider):
    pass


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbeddings(EmbeddingProvider):
    """Character 1-3 gram feature hashing. Deterministic and dependency-free; lexical in nature, so only for tests."""

    provider_id = "hashing"
    model_id = "char-ngram-hash"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def ready(self) -> bool:
        return True

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        norm = normalize(text)
        for n in (1, 2, 3):
            for i in range(len(norm) - n + 1):
                gram = norm[i : i + n]
                if gram.isspace():
                    continue
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest[:4], "little") % self.dimensions] += (1.0 if digest[4] & 1 else -1.0) * n
        return _l2(vec)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class LocalTransformerEmbeddings(EmbeddingProvider):
    """A local sentence-embedding model (CLS pooling + L2 norm, the BGE recipe). Loads lazily in a background thread."""

    provider_id = "local"

    def __init__(self, model: str, model_id: str | None = None, device: str | None = None, max_length: int = 512, batch_size: int = 8):
        self.model_path = model
        self.model_id = model_id or model  # the cache identity: a model name, not wherever its files happen to live
        self.max_length = max_length
        self.batch_size = batch_size
        self._device_wanted = device
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._error: str | None = None
        self._loaded_ms: float | None = None
        self.dimensions = 0
        threading.Thread(target=self._load, name="penumbra-embedding-load", daemon=True).start()

    def _load(self) -> None:
        started = time.perf_counter()
        # PyTorch only: an installed TensorFlow would otherwise be imported too (slow, and noisy about numpy ABIs).
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TORCH", "1")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = self._device_wanted or ("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModel.from_pretrained(self.model_path).to(device).eval()
            self._torch, self._device = torch, device
            with self._lock:
                self._tokenizer, self._model = tokenizer, model
                self.dimensions = int(model.config.hidden_size)
            self._encode_ready = True
            self._encode(["预热"])  # the first call pays one-off kernel setup; not a chat query
            self._loaded_ms = round((time.perf_counter() - started) * 1000)
            print(f"[penumbra] embedding model ready: {self.model_id} on {device}, {self.dimensions} dims, {self._loaded_ms} ms")
        except Exception as error:  # the vector signal stays off; lexical + entity keep working
            self._error = f"{type(error).__name__}: {error}"[:300]
            print(f"[penumbra] embedding model unavailable ({self.model_id}): {self._error}")

    def ready(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {**super().status(), "device": getattr(self, "_device", None), "loadMs": self._loaded_ms, "error": self._error}

    def _encode(self, texts: list[str]) -> list[list[float]]:
        torch = self._torch
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            # The model lock is taken per chunk, not per call: a live chat query waits for at most one small chunk of the
            # background embedder's work, never for the whole batch.
            with self._lock, torch.inference_mode():
                batch = self._tokenizer(texts[i : i + self.batch_size], padding=True, truncation=True, max_length=self.max_length,
                                        return_tensors="pt").to(self._device)
                hidden = self._model(**batch).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
                out.extend(hidden.float().cpu().tolist())
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-m3"
DEFAULT_MODEL_ID = "BAAI/bge-m3"


def provider_from_env() -> EmbeddingProvider:
    """PENUMBRA_EMBEDDING: unset -> the local bge-m3 in ./models/bge-m3 when it is there, else none."""
    choice = (os.environ.get("PENUMBRA_EMBEDDING") or "").strip()
    device = os.environ.get("PENUMBRA_EMBEDDING_DEVICE") or None
    if not choice:
        if (DEFAULT_MODEL_DIR / "config.json").exists():
            return LocalTransformerEmbeddings(str(DEFAULT_MODEL_DIR), model_id=DEFAULT_MODEL_ID, device=device)
        return NoEmbeddings()
    if choice.startswith("local:"):
        return LocalTransformerEmbeddings(choice.split(":", 1)[1], device=device)
    if choice == "hashing":
        return HashingEmbeddings()
    return NoEmbeddings()
