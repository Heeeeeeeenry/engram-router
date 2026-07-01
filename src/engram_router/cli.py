"""Command line interface for EngramRouter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark import format_report, load_cases, load_conversation, run_benchmark
from .store import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="engram", description="EngramRouter CLI")
    parser.add_argument("--db", default="memory.db", help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    save_p = sub.add_parser("save", help="Save a memory text")
    save_p.add_argument("text", help="Memory text to save")
    save_p.add_argument("--namespace", default="default", help="Tenant namespace")

    recall_p = sub.add_parser("recall", help="Recall memory evidence")
    recall_p.add_argument("query", help="Search query")
    recall_p.add_argument("--top-k", type=int, default=5, help="Number of results (≥1)")
    recall_p.add_argument("--namespace", default="default", help="Tenant namespace")

    gap_p = sub.add_parser("gap-check", help="Check whether recalled memory is sufficient")
    gap_p.add_argument("query", help="Search query")
    gap_p.add_argument("--top-k", type=int, default=5, help="Number of results (≥1)")
    gap_p.add_argument("--namespace", default="default", help="Tenant namespace")
    gap_p.add_argument("--scan-all", action="store_true", help="Bypass recall, scan all memories")

    del_p = sub.add_parser("delete", help="Delete a memory by ID")
    del_p.add_argument("memory_id", help="ID of the memory to delete")

    raw_p = sub.add_parser("save-raw-log", help="Save a raw log entry")
    raw_p.add_argument("text", help="Raw log text")
    raw_p.add_argument("--kind", default="conversation", help="Log kind (conversation, document, etc.)")

    compact_p = sub.add_parser("compact", help="Distill a raw log while preserving evidence refs")
    compact_p.add_argument("raw_log_id", help="Raw log ID to distill")
    compact_p.add_argument("distilled_text", help="Distilled/summarized text")
    compact_p.add_argument("--namespace", default="default", help="Tenant namespace")

    bench_p = sub.add_parser("benchmark", help="Compare evidence recall vs lossy summary baseline")
    bench_p.add_argument("--conversation", required=True, help="Path to conversation markdown")
    bench_p.add_argument("--cases", required=True, help="Path to benchmark questions jsonl")
    bench_p.add_argument("--text", action="store_true", help="Print human-readable report instead of JSON")
    bench_p.add_argument(
        "--gate",
        action="store_true",
        help="Exit non-zero if any hard-gate case fails (known_gap cases never gate). For CI/regression use.",
    )

    # Install / status subcommands delegate to install.py
    sub.add_parser("install", help="Auto-detect and configure MCP for all agents [alias: engram-install]")
    sub.add_parser("status", help="Show which agents are connected to engram [alias: engram-install status]")
    sub.add_parser("uninstall", help="Remove engram MCP config from agents")

    args = parser.parse_args()

    # Delegate install/status/uninstall to install.py
    if args.command in ("install", "status", "uninstall"):
        from .install import cmd_install, cmd_status, cmd_uninstall
        if args.command == "install":
            raise SystemExit(cmd_install())
        elif args.command == "status":
            raise SystemExit(cmd_status())
        else:
            raise SystemExit(cmd_uninstall())

    if args.command == "benchmark":
        turns = load_conversation(args.conversation)
        cases = load_cases(args.cases)
        report = run_benchmark(turns, cases, db_path=Path(args.db))
        if args.text:
            print(format_report(report))
        else:
            _print(report)
        if args.gate and not report.get("gate", {}).get("passed", True):
            raise SystemExit(1)
        return

    try:
        store = MemoryStore(path=Path(args.db))
    except OSError as exc:
        _die(f"Cannot open database: {exc}")

    try:
        with store:
            if args.command == "save":
                if not args.text.strip():
                    _die("text must not be empty")
                memory_id = store.save(args.text, namespace=args.namespace)
                _print({"memory_id": memory_id})
            elif args.command == "recall":
                if args.top_k < 1:
                    _die(f"--top-k must be ≥ 1, got {args.top_k}")
                records = store.recall(args.query, top_k=args.top_k, namespace=args.namespace)
                _print({"memories": [record.to_dict() for record in records]})
            elif args.command == "gap-check":
                records = store.recall(args.query, top_k=args.top_k, namespace=args.namespace)
                _print(store.gap_check(args.query, records, namespace=args.namespace,
                       scan_all=args.scan_all))
            elif args.command == "delete":
                deleted = store.delete(args.memory_id)
                _print({"deleted": deleted, "memory_id": args.memory_id})
            elif args.command == "save-raw-log":
                raw_id = store.save_raw_log(args.text, kind=args.kind)
                _print({"raw_log_id": raw_id})
            elif args.command == "compact":
                try:
                    distilled_id = store.compact(args.raw_log_id, args.distilled_text, namespace=args.namespace)
                except KeyError:
                    _die(f"raw_log_id not found: {args.raw_log_id}")
                _print({"distilled_id": distilled_id})
    except OSError as exc:
        _die(f"Database error: {exc}")


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _die(msg: str) -> None:
    print(json.dumps({"error": msg}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
