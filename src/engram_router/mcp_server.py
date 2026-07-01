"""MCP Server for EngramRouter — stdio JSON-RPC 2.0 transport.

No third-party MCP dependency.  Uses only Python stdlib asyncio.
MemoryStore calls are synchronous — the stdio transport processes one
request at a time, so no thread-offloading is needed.

Protocol: JSON-RPC 2.0 over stdin/stdout.
  Request:  {"jsonrpc": "2.0", "id": N, "method": "...", "params": {...}}
  Response: {"jsonrpc": "2.0", "id": N, "result": ...}
  Error:    {"jsonrpc": "2.0", "id": N, "error": {"code": ..., "message": ...}}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .store import MemoryStore

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 error codes
# ---------------------------------------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_NOT_INITIALIZED = -32002
AUTH_ERROR = -32001

MAX_LINE_LENGTH = 1_048_576  # 1 MB
MAX_JSON_DEPTH = 100


def _safe_json_loads(text: str) -> Any:
    """json.loads with depth-limit pre-scan to prevent recursion/stack traps."""
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise json.JSONDecodeError(
                    f"Exceeded maximum nesting depth of {MAX_JSON_DEPTH}", text, 0
                )
        elif ch in "}]":
            depth -= 1
    return json.loads(text)


class JsonRpcError(Exception):
    """Exception that carries a JSON-RPC error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code

# ---------------------------------------------------------------------------
# Tool definitions (MCP tool/list response format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "memory.save",
        "description": "Save a text to the EngramRouter memory store. Returns the memory ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to save.",
                },
                "source": {
                    "type": "string",
                    "description": "Optional source label (e.g. 'conversation', 'compaction').",
                },
                "namespace": {
                    "type": "string",
                    "description": "Tenant namespace (default: 'default').",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "memory.recall",
        "description": "Recall top-k memories matching a query from the store.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query string to search for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Tenant namespace (default: 'default').",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory.gap_check",
        "description": "Check whether recalled memories are sufficient to answer a query, or if a gap exists.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to check for gaps.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Tenant namespace (default: 'default').",
                },
                "scan_all": {
                    "type": "boolean",
                    "description": "When true, bypass recall and scan all memories for gap analysis (default: false).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory.compact",
        "description": "Distill a raw log into a compact memory while preserving evidence references.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "raw_log_id": {
                    "type": "string",
                    "description": "ID of the raw log to compact.",
                },
                "distilled_text": {
                    "type": "string",
                    "description": "The distilled/compacted text to store.",
                },
            },
            "required": ["raw_log_id", "distilled_text"],
        },
    },
    {
        "name": "memory.consolidate",
        "description": "合并重复实体名 (大小写/空白变体)、清理孤立边和重复边。返回清理统计。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memory.delete",
        "description": "Delete a memory by ID. Returns true/false. USE CAUTIOUSLY: only delete memories you created and no longer need; never delete another agent's memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to delete.",
                },
            },
            "required": ["memory_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# JSON-RPC handler
# ---------------------------------------------------------------------------
class JsonRpcHandler:
    """Handles JSON-RPC 2.0 requests for the MCP server."""

    def __init__(self, store: MemoryStore, api_key: str | None = None) -> None:
        self._store = store
        self._api_key = api_key
        self._initialized: bool = False
        self._methods: dict[str, Any] = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }
        self._tools: dict[str, Any] = {
            "memory.save": self._tool_memory_save,
            "memory.recall": self._tool_memory_recall,
            "memory.gap_check": self._tool_memory_gap_check,
            "memory.compact": self._tool_memory_compact,
            "memory.consolidate": self._tool_memory_consolidate,
            "memory.delete": self._tool_memory_delete,
        }

    def _check_auth(self) -> None:
        """Verify ENGRAM_API_KEY is set and non-empty, raise JsonRpcError(-32001) otherwise."""
        if not self._api_key:
            raise JsonRpcError(
                AUTH_ERROR,
                "Authentication required: ENGRAM_API_KEY environment variable is not set or empty",
            )

    async def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a single JSON-RPC request; returns a response dict or None for notifications."""
        rid = request.get("id")

        # Validate JSON-RPC envelope
        if request.get("jsonrpc") != "2.0":
            return _error(rid, INVALID_REQUEST, "jsonrpc must be '2.0'")

        method = request.get("method")
        if not isinstance(method, str) or not method:
            return _error(rid, INVALID_REQUEST, "method is required")

        params = request.get("params", {})

        handler = self._methods.get(method)
        if handler is None:
            return _error(rid, METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            result = await handler(params)
        except JsonRpcError as exc:
            if rid is None:
                print(f"Notification error ({method}): {exc}", file=sys.stderr)
                return None
            return _error(rid, exc.code, str(exc))
        except Exception as exc:
            if rid is None:
                # Notification — log to stderr, no response
                print(f"Notification error ({method}): {exc}", file=sys.stderr)
                return None
            return _error(rid, INTERNAL_ERROR, str(exc))

        if rid is None:
            return None  # notification — no response
        return _ok(rid, result)

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "engram-router", "version": "0.1.0"},
        }

    async def _handle_initialized(self, params: dict[str, Any]) -> None:
        self._initialized = True
        return None  # notification — no response

    async def _handle_ping(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {}

    # -- MCP protocol methods -------------------------------------------------

    async def _handle_tools_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._check_auth()
        if not self._initialized:
            raise JsonRpcError(SERVER_NOT_INITIALIZED, "Server not initialized")
        return {"tools": TOOLS}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        self._check_auth()
        if not self._initialized:
            raise JsonRpcError(SERVER_NOT_INITIALIZED, "Server not initialized")
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool = self._tools.get(name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            }

        try:
            content = await tool(arguments)
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(content, ensure_ascii=False, default=str),
                }
            ]
        }

    # -- Tool implementations -------------------------------------------------

    async def _tool_memory_save(self, args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("text", "")
        if not text:
            raise ValueError("text is required")
        source = args.get("source", "mcp")
        namespace = args.get("namespace", "default")
        memory_id = self._store.save(text, source=source, namespace=namespace)
        return {"memory_id": memory_id}

    async def _tool_memory_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "")
        if not query:
            raise ValueError("query is required")
        top_k = args.get("top_k", 5)
        namespace = args.get("namespace", "default")
        records = self._store.recall(query, top_k=top_k, namespace=namespace)
        memories = [r.to_dict() for r in records]
        return {"memories": memories}

    async def _tool_memory_gap_check(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "")
        if not query:
            raise ValueError("query is required")
        namespace = args.get("namespace", "default")
        scan_all = args.get("scan_all", False)
        result = self._store.gap_check(query, namespace=namespace, scan_all=scan_all)
        return result

    async def _tool_memory_compact(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_log_id = args.get("raw_log_id", "")
        distilled_text = args.get("distilled_text", "")
        if not raw_log_id or not distilled_text:
            raise ValueError("raw_log_id and distilled_text are required")
        distilled_id = self._store.compact(raw_log_id, distilled_text)
        return {"distilled_id": distilled_id}

    async def _tool_memory_consolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self._store.consolidate()
        return {"status": "ok", "stats": result}

    async def _tool_memory_delete(self, args: dict[str, Any]) -> dict[str, Any]:
        memory_id = args.get("memory_id", "")
        if not memory_id:
            raise ValueError("memory_id is required")
        deleted = self._store.delete(memory_id)
        return {"deleted": deleted}


# ---------------------------------------------------------------------------
# I/O loop
# ---------------------------------------------------------------------------
async def _read_line(reader: asyncio.StreamReader) -> str:
    """Read a single line from stdin (async)."""
    line = await reader.readline()
    return line.decode("utf-8").strip()


def _write_line(line: str) -> None:
    """Write a single line to stdout and flush."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


async def run_server(db_path: str | Path) -> None:
    """Run the MCP server loop: read JSON-RPC requests from stdin, write responses to stdout."""
    api_key = os.environ.get("ENGRAM_API_KEY", "").strip() or None
    store = MemoryStore(path=Path(db_path))
    handler = JsonRpcHandler(store, api_key=api_key)

    loop = asyncio.get_running_loop()
    # Allow headroom so lines slightly over MAX_LINE_LENGTH can be
    # read fully and rejected with an explicit error response.
    reader = asyncio.StreamReader(limit=MAX_LINE_LENGTH + 65536)
    protocol = asyncio.StreamReaderProtocol(reader)

    try:
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    except (OSError, IOError) as exc:
        print(f"Cannot open stdin: {exc}", file=sys.stderr)
        store.close()
        sys.exit(0)

    try:
        while True:
            try:
                line = await _read_line(reader)
            except asyncio.LimitOverrunError:
                _write_line(
                    json.dumps(
                        _error(None, PARSE_ERROR, f"Request line exceeds {MAX_LINE_LENGTH} bytes")
                    )
                )
                # LimitOverrunError corrupts the reader's internal buffer;
                # recreate the reader to continue processing further requests.
                reader = asyncio.StreamReader(limit=MAX_LINE_LENGTH + 65536)
                protocol = asyncio.StreamReaderProtocol(reader)
                await loop.connect_read_pipe(lambda: protocol, sys.stdin)
                continue
            except (BrokenPipeError, EOFError):
                break

            if not line:
                break

            # --- line length limit ---
            if len(line) > MAX_LINE_LENGTH:
                _write_line(
                    json.dumps(
                        _error(None, PARSE_ERROR, f"Request line exceeds {MAX_LINE_LENGTH} bytes")
                    )
                )
                continue

            try:
                request = _safe_json_loads(line)
            except json.JSONDecodeError as exc:
                _write_line(json.dumps(_error(None, PARSE_ERROR, f"Parse error: {exc}")))
                continue

            if not isinstance(request, dict):
                _write_line(json.dumps(_error(None, INVALID_REQUEST, "Request must be a JSON object")))
                continue

            response = await handler.handle(request)
            if response is not None:
                _write_line(json.dumps(response, ensure_ascii=False))
    except KeyboardInterrupt:
        pass
    except (BrokenPipeError, EOFError):
        pass
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ok(rid: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="engram-mcp",
        description="EngramRouter MCP Server (stdio JSON-RPC 2.0)",
    )
    parser.add_argument(
        "--db",
        default="memory.db",
        help="SQLite database path (default: memory.db)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run_server(args.db))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Server error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
