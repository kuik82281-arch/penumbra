"""Penumbra's local HTTP API (127.0.0.1 only). JSON in, JSON out.

RAW
POST /originals                          {source, conversationId, items[], context?: {claudeSessionId, turnId, sourceType, metadata, eventRefs}}
                                                                              -> {added, existing, conflicts, skipped}  (immutable, idempotent)
GET  /originals/<id>                     the stored original with its provenance and attachments

Unified memory: RAW -> Episode -> Pattern   (everything under /memory-core)
POST /memory-core/retrieve               {query, turnId, conversationId, sessionId, recent?} -> LOCKED | NO_MEMORY_NEEDED
                                         (the turn's one search, Patterns first; creates the turn's memory lock)
POST /memory-core/confirm                {injectId, conversationId, sessionId, refs[{kind,id,version}]}  seen suppression, after the turn
POST /memory-core/recall                 {pattern_id | episode_id | raw_ids[] | query, turnId?, currentSession?}  expand (Search Once)
GET  /memory-core                        snapshot: counts, patterns, episodes, candidates (STAGING / QUARANTINE), runs, providers, audit
GET  /memory-core/health                 Ollama / DeepSeek / reranker / embedding
GET  /memory-core/raw | /patterns | /episodes | /decisions | /locks | /attachments | /provenance/<pattern|episode>/<id>
POST /memory-core/run | /run-sync | /tick | /settings | /activity | /note | /edit | /review | /retry

Retrieval (read-only, development)
POST /retrieval/debug                    {query, policy?, mode?, conversationId?, sessionId?, currentSession?, topK?, exact?} -> full ranking with trace
GET  /retrieval/stats                    index sizes, recent calls, latency p50/p95, config
GET  /stats, GET /health

Preference Learning (bridge path /api/memory/preferences/...; writes need actor "user" (the default); no model involved):
POST  /preferences/documents                 {owner, title, originalContent, mode?, initialLabels?} -> {document} (a draft)
GET   /preferences/documents[?status=&owner=]  (without originalContent; with estimatedTokens)      -> {documents}
GET   /preferences/documents/<id>                                                                 -> {document}
PATCH /preferences/documents/<id>            {title?, owner?, initialLabels?, mode?, originalContent?} -> {document, createdVersion}
POST  /preferences/documents/<id>/version    {originalContent?, title?, mode?}  new draft version  -> {document}
POST  /preferences/documents/<id>/publish    {mode?} | /unpublish | /enable | /disable | /archive  -> {document}
GET   /preferences/session-pack[?content=0]  published enabled session_pinned documents, packHash, token budget
GET   /preferences/documents/<id>/chunks                                                          -> {documentId, chunks}
PATCH /preferences/chunks/<chunkId>          {labels?, freeTags?, entities?, startOffset? | endOffset?} -> {chunk}
POST  /preferences/chunks/<chunkId>/split    {offset | relativeOffset}                            -> {chunks}
POST  /preferences/chunks/merge              {chunkIds[]}                                         -> {chunk}
POST  /preferences/chunks/<chunkId>/exclude | /restore                                           -> {chunk}
GET   /preferences/jobs/<jobId> (legacy records), GET /preferences/labels, GET /preferences/chunks/<chunkId>
POST  /preferences/documents/<id>/segment | /resegment -> 410: DeepSeek segmentation is retired

Retired (410): /inject, /inject/confirm, /recall, /review/daily, /candidates, /events, /manual, /memories, /batches, /patterns.
Their data was migrated into Episodes / Patterns / QUARANTINE (penumbra/memory/migrate.py); the files stay as history.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from . import identity
from .preferences import MODES, OWNERS, canonical_labels, estimate_tokens
from .service import Invalid, NotFound, Penombre

MAX_BODY = 2_000_000
# The client went away mid-request (e.g. the bridge restarted): nothing to answer and nothing wrong with the service.
CLIENT_GONE = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


class QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that logs a client disconnect as one line instead of a traceback."""

    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys

        error = sys.exc_info()[1]
        if isinstance(error, CLIENT_GONE):
            print(f"[penumbra] client {client_address[0]}:{client_address[1]} disconnected ({type(error).__name__})")
            return
        super().handle_error(request, client_address)


def _doc_view(doc: dict) -> dict:
    """Listing view: everything but the full text (fetch one document for originalContent)."""
    return {k: v for k, v in doc.items() if k != "originalContent"} | {
        "contentLength": len(doc["originalContent"]), "estimatedTokens": estimate_tokens(doc["originalContent"]),
        "mode": doc.get("mode") or "retrieval", "enabled": doc.get("enabled", True) is not False}


def _doc_full(doc: dict) -> dict:
    return doc | {"estimatedTokens": estimate_tokens(doc["originalContent"]), "mode": doc.get("mode") or "retrieval",
                  "enabled": doc.get("enabled", True) is not False}


def make_handler(service: Penombre):
    memory = service.memory
    prefs = service.preferences

    class Handler(BaseHTTPRequestHandler):
        server_version = "Penombre/0.2"

        def log_message(self, fmt, *args):  # quieter than the default per-request stderr line
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise Invalid("body too large")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise Invalid("body must be a JSON object")
            return data

        def _run(self, fn) -> None:
            started = time.perf_counter()
            try:
                self._send(200, fn())
            except NotFound as error:
                self._send(404, {"error": str(error)})
            except (Invalid, ValueError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
            except CLIENT_GONE:
                raise  # nobody to answer; QuietServer logs one line
            except Exception as error:  # keep the service alive; the gateway degrades on 5xx
                print(f"[penumbra] {self.command} {self.path} failed: {error!r}")
                self._send(500, {"error": "internal error"})
            finally:
                elapsed = (time.perf_counter() - started) * 1000
                if self.path not in ("/health",):
                    print(f"[penumbra] {self.command} {self.path} {elapsed:.0f}ms")

        def _query(self) -> dict:
            raw = self.path.split("?", 1)[1] if "?" in self.path else ""
            return {k: v[-1] for k, v in parse_qs(raw).items()}

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            parts = path.strip("/").split("/")
            q = self._query()
            if parts[0] == "memory-core":
                return self._run(lambda: memory.route("GET", parts[1:], query=q))
            if path == "/health":
                return self._run(lambda: {"ok": True})
            if path == "/stats":
                return self._run(service.stats)
            if parts[0] == "originals" and len(parts) == 2:
                return self._run(lambda: {"original": service.original_record(parts[1])})
            if path == "/retrieval/stats":
                return self._run(service.retrieval.stats)
            if parts[0] == "preferences":
                return self._preferences_get(parts, q)
            self._send(404, {"error": "not found"})

        # ------------------------------------------------------------ preference learning (P2-B)

        def _preferences_get(self, parts, q):
            if parts[1:] == ["labels"]:
                return self._run(lambda: {"labels": list(canonical_labels()), "owners": list(OWNERS), "modes": list(MODES)})
            if parts[1:] == ["session-pack"]:
                return self._run(lambda: prefs.session_pack(with_content=q.get("content") != "0"))
            if parts[1:] == ["documents"]:
                return self._run(lambda: {"documents": [_doc_view(d) for d in prefs.list_documents(q.get("status"), q.get("owner"))]})
            if len(parts) == 3 and parts[1] == "documents":
                return self._run(lambda: {"document": _doc_full(prefs.get_document(parts[2]))})
            if len(parts) == 4 and parts[1] == "documents" and parts[3] == "chunks":
                return self._run(lambda: {"documentId": parts[2], "chunks": prefs.list_chunks(parts[2])})
            if len(parts) == 3 and parts[1] == "chunks":
                return self._run(lambda: {"chunk": prefs._find_chunk(parts[2])[2]})
            if len(parts) == 3 and parts[1] == "jobs":
                return self._run(lambda: {"job": prefs.get_job(parts[2])})
            self._send(404, {"error": "not found"})

        def _preferences_post(self, parts):
            if parts[1:] == ["documents"]:
                return self._run(lambda: {"document": prefs.create_document(self._body())})
            if parts[1:] == ["chunks", "merge"]:
                return self._run(lambda: {"chunk": prefs.merge_chunks(self._body())})
            if len(parts) == 4 and parts[1] == "documents":
                if parts[3] in ("segment", "resegment"):
                    return self._send(410, {"error": "DeepSeek segmentation is retired: edit the draft and publish it "
                                                     "(Session 常驻 or 普通检索)"})
                actions = {
                    "version": lambda b: {"document": prefs.new_version(parts[2], b)},
                    "publish": lambda b: {"document": prefs.publish(parts[2], b)},
                    "unpublish": lambda b: {"document": prefs.unpublish(parts[2], b)},
                    "enable": lambda b: {"document": prefs.set_enabled(parts[2], b, True)},
                    "disable": lambda b: {"document": prefs.set_enabled(parts[2], b, False)},
                    "archive": lambda b: {"document": prefs.archive_document(parts[2], b)},
                }
                if parts[3] in actions:
                    return self._run(lambda: actions[parts[3]](self._body()))
            if len(parts) == 4 and parts[1] == "chunks":
                actions = {
                    "split": lambda b: {"chunks": prefs.split_chunk(parts[2], b)},
                    "exclude": lambda b: {"chunk": prefs.set_excluded(parts[2], b, True)},
                    "restore": lambda b: {"chunk": prefs.set_excluded(parts[2], b, False)},
                }
                if parts[3] in actions:
                    return self._run(lambda: actions[parts[3]](self._body()))
            self._send(404, {"error": "not found"})

        def do_PATCH(self):  # noqa: N802
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["preferences", "documents"]:
                return self._run(lambda: prefs.patch_document(parts[2], self._body()))
            if len(parts) == 3 and parts[:2] == ["preferences", "chunks"]:
                return self._run(lambda: {"chunk": prefs.patch_chunk(parts[2], self._body())})
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            parts = path.strip("/").split("/")
            if parts[0] == "preferences":
                return self._preferences_post(parts)
            if parts[0] == "memory-core":
                return self._run(lambda: memory.route("POST", parts[1:], self._body()))
            if path == "/originals":
                def ingest():
                    body = self._body()
                    return service.ingest_originals(
                        str(body.get("source") or identity.USER_ACTOR), str(body.get("conversationId") or ""), body.get("items") or [], body.get("context")
                    )
                return self._run(ingest)
            if path == "/retrieval/debug":
                return self._run(lambda: service.retrieval_debug(self._body()))
            if path in ("/inject", "/inject/confirm", "/recall", "/review/daily") or parts[0] in ("candidates", "events", "manual", "memories", "batches", "patterns"):
                return self._send(410, {"error": "retired: use /memory-core (RAW -> Episode -> Pattern)"})
            self._send(404, {"error": "not found"})

    return Handler


def _warm_models(service: Penombre) -> None:
    """Load the reranker once the embedding model is done. Both import `transformers` lazily, and two threads doing that at
    the same time can fail with "cannot import name AutoModel"; one after the other cannot."""
    provider = service.vectors.provider
    for _ in range(300):
        if provider.ready() or getattr(provider, "_error", None) or provider.provider_id != "local":
            break
        time.sleep(1)
    service.memory.read.reranker.warm_async()


def serve(service: Penombre) -> None:
    cfg = service.config
    httpd = QuietServer((cfg.host, cfg.port), make_handler(service))
    print(f"[penumbra] data: {cfg.data_dir}")
    print(f"[penumbra] index rebuilt: {service.last_rebuild}")
    start = service.memory_start
    if any(bool(v) for v in (start.get("recovery") or {}).values()) or start.get("migration"):
        print(f"[penumbra] memory start: {start}")
    counts = service.memory.store.counts()
    print(f"[penumbra] memory: {counts['episodes']} episodes, {counts['patterns']} patterns, {counts['staging']} staging, {counts['quarantine']} quarantine")
    ollama, deepseek = service.memory.discovery.ollama.health(force=True), service.memory.verifier.status()
    print(f"[penumbra] discovery: ollama {ollama.get('model')} ({'ready' if ollama.get('ready') else 'NOT ready: ' + str(ollama.get('error'))}); "
          f"verification: {deepseek['provider']} {deepseek['model']} ({'configured' if deepseek['configured'] else 'NOT configured'})")
    print(f"[penumbra] listening on http://{cfg.host}:{cfg.port}")
    try:
        threading.Thread(target=_warm_models, args=(service,), name="penumbra-warm", daemon=True).start()
        service.memory.pipeline.start_scheduler()
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        service.close()
