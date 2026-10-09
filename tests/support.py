"""Shared test support: a concept-based fake embedding provider, scripted Ollama / DeepSeek stand-ins, service builders.
Synthetic data only; no network, no model."""
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from penumbra.config import Config
from penumbra.embeddings import EmbeddingProvider
from penumbra.errors import WorkerError
from penumbra.memory.discovery import Discovery
from penumbra.memory.verification import Verifier
from penumbra.retrieval import RetrievalConfig, RetrievalRequest
from penumbra.service import Penombre
from penumbra.text import normalize

# Most tests script the discovery model's spans; the session segmentation has its own tests (test_sessions.py).
os.environ.setdefault("PENUMBRA_SEGMENTATION", "discovery")
os.environ.setdefault("PENUMBRA_QUERY_REWRITE", "off")  # tests never call a real model; tests/test_rewrite.py uses a fake one
os.environ.setdefault("PENUMBRA_MEMORY_GATE", "off")  # nor DeepSeek: tests/test_gate.py uses a fake one

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
CONV = "conv-test"
CONCEPTS = {
    "conflict": ["吵架", "争执", "生气", "发火", "闹别扭", "冷战", "气死", "吵"],
    "comfort": ["抱", "哄", "安慰", "拥抱", "摸头", "别讲道理"],
    "weekend": ["周末", "看展", "逛街", "爬山", "出去玩", "公园"],
    "coffee": ["咖啡", "拿铁", "美式", "燕麦奶"],
    "sea": ["海", "潮水", "浪", "海边"],
    "cilantro": ["香菜", "芫荽"],
}

# Vector gates are calibrated per model; the concept fake separates cleanly by absolute similarity and its tiny corpora
# have no meaningful background mean, so it gates on similarity alone.
FAKE_GATES = RetrievalConfig(recall_vector_min=0.5, recall_vector_margin=-1.0, inject_vector_min=0.6, inject_vector_margin=-1.0,
                             raw_recall_vector_min=0.5, raw_recall_vector_margin=-1.0)


class ConceptEmbeddings(EmbeddingProvider):
    """One dimension per concept (+ a faint character residue so unrelated texts are not all identical)."""

    provider_id = "fake"
    model_id = "concepts-v1"

    def __init__(self):
        self.dimensions = len(CONCEPTS) + 8
        self.calls = 0

    def ready(self):
        return True

    def _vec(self, text):
        t = normalize(text)
        vec = [float(sum(t.count(w) for w in words)) for words in CONCEPTS.values()]
        residue = [0.0] * 8
        for ch in t:
            residue[ord(ch) % 8] += 0.02
        vec += residue
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_query(self, text):
        return self._vec(text)

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self._vec(t) for t in texts]


class FakeOllama:
    """Scripted Ollama: `answers` is a list of str / Exception consumed per chat() call (the last one repeats)."""

    model = "fake-ollama"

    def __init__(self, answers=None, ready=True):
        self.answers, self.ready, self.calls = list(answers or []), ready, []
        self.error = None if ready else "connection refused"
        self.used, self.events = False, []  # events: "chat" / "unload", in order

    def unload(self):
        self.events.append("unload")
        self.used = False
        return True

    def health(self, force=False, max_age_s=20.0):
        return {"ready": self.ready, "model": self.model, "error": self.error, "installed": self.ready}

    def status(self, probe=True):
        return {**self.health(), "provider": "ollama"}

    def chat(self, system, user, schema, num_predict=700):
        self.calls.append(user)
        self.events.append("chat")
        self.used = True
        if not self.answers:
            return json.dumps({"candidates": []}), {"latency_ms": 1, "done_reason": "stop"}
        item = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(item, Exception):
            raise item
        return item, {"latency_ms": 1, "done_reason": "stop"}


class FakeDeepSeek:
    """Scripted DeepSeek: `decide(payload) -> answer dict | Exception`; every payload it received is kept in `.payloads`."""

    provider, model = "deepseek", "fake-deepseek"

    def __init__(self, decide=None, configured=True):
        self.decide, self.configured, self.payloads = decide, configured, []
        self.last_error = self.last_ok_at = None

    def available(self):
        return self.configured

    def status(self):
        return {"provider": self.provider, "model": self.model, "configured": self.configured, "ready": self.configured, "baseUrl": "fake"}

    def chat_json(self, system, user, max_tokens=3500):
        payload = json.loads(user)
        self.payloads.append(payload)
        answer = self.decide(payload) if self.decide else {"actions": [{"action": "NO_ACTION", "reason": "nothing"}], "confidence": 0.9, "reason": "nothing"}
        if isinstance(answer, Exception):
            raise answer
        return answer, {"usage": {"total_tokens": 1}, "latency_ms": 1}


def episode_answer(payload, content, ref="e1", state="", entities=None, topics=None, importance=0.7, raw_index=0, when=None):
    """A valid CREATE_EPISODE answer citing the payload's own RAW."""
    raw = payload["RAW"][raw_index]
    when = when or raw["createdAt"]
    return {"action": "CREATE_EPISODE", "ref": ref, "content": content, "time_start": when, "time_end": when, "entities": entities or [], "topics": topics or [],
            "state": state, "importance": importance, "confidence": 0.9, "source_raw_ids": [raw["id"]], "attachment_ids": []}


def wrap(*actions, confidence=0.9, reason="test"):
    return {"actions": list(actions), "confidence": confidence, "reason": reason}


def build(tmp: Path, ollama=None, deepseek=None, embeddings=None) -> Penombre:
    svc = Penombre(Config(data_dir=tmp), embeddings=embeddings or ConceptEmbeddings(), retrieval=FAKE_GATES,
                   discovery=Discovery(ollama or FakeOllama()), verifier=Verifier(deepseek or FakeDeepSeek()))
    svc._now = lambda: NOW
    svc.vectors.wait_idle(20)
    return svc


def recall(svc, query, **extra):
    """Full engine ranking as {'results': [{id, kind, text}], 'found': ...} over RAW + memory (what the old recall tool searched)."""
    limit = extra.pop("limit", 20)
    with svc.lock:
        found = svc.retrieval.retrieve(RetrievalRequest(query=query, policy="recall", include_static=False, top_k=limit, **extra))
    return {"results": [{"id": h["hitId"], "kind": h["kind"], "text": h["content"]} for h in found["top"]], "found": found,
            "excludedCurrentSession": found["summary"].get("excludedCurrentSession", 0)}


class Base(unittest.TestCase):
    """A fresh data dir and service per test (`self.svc`); `self.open()` reopens it (a restart)."""

    ollama = None
    deepseek = None

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="penumbra-test-"))
        self.provider = ConceptEmbeddings()
        self.ollama, self.deepseek = FakeOllama(), FakeDeepSeek()
        self.open()

    def open(self, provider=None):
        self.svc = build(self.tmp, self.ollama, self.deepseek, provider or self.provider)
        self.memory = self.svc.memory

    def tearDown(self):
        self.svc.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def raw(self, text, at="2026-09-01T10:00:00Z", conv=CONV, role="user", n=[0]):
        n[0] += 1
        ids = self.svc.ingest_originals("user", conv, [{"id": f"m{n[0]}", "role": role, "content": text, "createdAt": at}])["added"]
        return ids[0]

    def note(self, content, **extra):
        """User's own memory: RAW (manual_note) + Episode. Returns the episode id."""
        made = self.memory.add_note({"actor": "user", "content": content, **extra})
        self.svc.vectors.wait_idle(20)
        return made["episode"]["episode_id"]

    def recall(self, query, **extra):
        return recall(self.svc, query, **extra)


def run_memory(base: Base, reason="test"):
    result = base.memory.pipeline.run(reason)
    base.svc.vectors.wait_idle(20)
    return result


__all__ = ["recall", "NOW", "CONV", "FAKE_GATES", "ConceptEmbeddings", "FakeOllama", "FakeDeepSeek", "WorkerError", "Base", "build", "episode_answer", "wrap", "run_memory"]
