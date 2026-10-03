"""CLI: python -m penumbra <command>

serve                       run the local HTTP service (127.0.0.1:8790)
rebuild                     drop index.sqlite projections and replay the files
stats                       counts
memory status               RAW / Episode / Pattern counts, providers (Ollama, DeepSeek), staging and quarantine
memory run                  run the memory pipeline once (discovery -> verification -> Episode / Pattern) and print the report
sync --bridge URL           backfill originals from a host app's conversation history API
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from . import identity
from .config import load_config
from .instance import AlreadyRunning
from .service import Penombre


def _print(data) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _get_json(url: str):
    # Local bridge: never through a system proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="penumbra")
    parser.add_argument("--data", help="data directory (default: $PENUMBRA_DATA or ./data)")
    parser.add_argument("--port", type=int)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    sub.add_parser("rebuild")
    sub.add_parser("stats")
    memory = sub.add_parser("memory")
    msub = memory.add_subparsers(dest="action", required=True)
    msub.add_parser("status")
    msub.add_parser("run")
    sync = sub.add_parser("sync")
    sync.add_argument("--bridge", default="http://127.0.0.1:8787")
    args = parser.parse_args(argv)

    config = load_config(args.data, args.port)
    try:
        service = Penombre(config)
    except AlreadyRunning as error:
        # Nothing was touched. Exit code 3 lets a launcher tell "already serving" from a crash.
        print(f"[penumbra] {error}; not starting a second instance", file=sys.stderr)
        return 3
    if args.command == "serve":
        from .api import serve

        serve(service)
        return 0
    try:
        if args.command == "rebuild":
            _print(service.last_rebuild)
        elif args.command == "stats":
            _print(service.stats())
        elif args.command == "memory" and args.action == "status":
            _print(service.memory.health() | {"counts": service.memory.store.counts()})
        elif args.command == "memory" and args.action == "run":
            _print(service.memory.route("POST", ["run-sync"], {}))
        elif args.command == "sync":
            base = args.bridge.rstrip("/")
            listing = _get_json(f"{base}/api/conversations")
            report = {}
            for conversation in listing.get("conversations", []):
                history = _get_json(f"{base}/api/conversations/{conversation['id']}/history")
                items = [
                    i for i in history.get("items", [])
                    if i.get("role") in ("user", "assistant") and i.get("status", "done") == "done"
                ]
                result = service.ingest_originals(identity.USER_ACTOR, conversation["id"], items)
                report[conversation["id"]] = {k: len(v) for k, v in result.items()}
            _print(report)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
