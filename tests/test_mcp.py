"""Tests for the MCP server (stdio JSON-RPC 2.0)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _send(request: dict) -> bytes:
    """Encode a JSON-RPC request as a line for stdin."""
    return (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")


def _recv(proc: subprocess.Popen, timeout: float = 5) -> dict:
    """Read one JSON-RPC response line from stdout."""
    try:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            raise RuntimeError("MCP process closed stdout")
        return json.loads(line.decode("utf-8"))
    except Exception:
        proc.kill()
        raise


def _start_mcp(db_path: str, api_key: str | None = "test-api-key") -> subprocess.Popen:
    """Start the MCP server subprocess."""
    env = {**__import__("os").environ, "PYTHONPATH": SRC_DIR}
    if api_key is not None:
        env["ENGRAM_API_KEY"] = api_key
    return subprocess.Popen(
        [sys.executable, "-m", "engram_router.mcp_server", "--db", db_path],
        cwd=SRC_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _init_mcp(proc: subprocess.Popen) -> None:
    """Send initialize + initialized to the MCP server."""
    proc.stdin.write(
        _send(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }
        )
    )
    proc.stdin.flush()
    resp = _recv(proc)
    assert resp["id"] == 0
    assert "result" in resp

    # Send initialized notification
    proc.stdin.write(
        _send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
    )
    proc.stdin.flush()


class TestMcpServer:
    """Integration tests for the MCP server (subprocess)."""

    def test_tools_list_returns_six_tools(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                # tools/list
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/list",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)

                assert resp["jsonrpc"] == "2.0"
                assert resp["id"] == 1
                assert "result" in resp
                tools = resp["result"]["tools"]
                assert len(tools) == 6

                names = {t["name"] for t in tools}
                assert names == {
                    "memory.save",
                    "memory.recall",
                    "memory.gap_check",
                    "memory.compact",
                    "memory.consolidate",
                    "memory.delete",
                }

                # Verify each tool has inputSchema with type object
                for t in tools:
                    assert "inputSchema" in t
                    assert t["inputSchema"]["type"] == "object"
                    assert "properties" in t["inputSchema"]
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_save_recall_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                # Save a memory
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.save",
                                "arguments": {
                                    "text": "我的机械键盘是HHKB Professional Hybrid Type-S",
                                    "source": "test",
                                },
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert "result" in resp, f"Expected result, got: {resp}"
                content = resp["result"]["content"]
                assert len(content) == 1
                result_data = json.loads(content[0]["text"])
                memory_id = result_data["memory_id"]
                assert memory_id.startswith("mem_")

                # Recall it
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.recall",
                                "arguments": {
                                    "query": "HHKB键盘",
                                    "top_k": 5,
                                },
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp2 = _recv(proc)
                assert "result" in resp2, f"Expected result, got: {resp2}"
                content2 = resp2["result"]["content"]
                recall_data = json.loads(content2[0]["text"])
                memories = recall_data["memories"]

                assert len(memories) >= 1
                # The saved memory should be in the results
                found = any(m["id"] == memory_id for m in memories)
                assert found, f"Memory {memory_id} not found in recall results"
                # Check content matches
                matching = [m for m in memories if m["id"] == memory_id]
                assert matching[0]["raw_text"] == "我的机械键盘是HHKB Professional Hybrid Type-S"
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_invalid_tool_name_returns_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "nonexistent.tool",
                                "arguments": {},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                # Unknown tool returns isError:true in result (not JSON-RPC error)
                assert "result" in resp
                assert resp["result"]["isError"] is True
                assert "Unknown tool" in resp["result"]["content"][0]["text"]
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_invalid_tool_name_returns_error_at_rpc_level(self):
        """Invalid tool name should return isError:true inside a normal JSON-RPC result."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 99,
                            "method": "tools/call",
                            "params": {
                                "name": "nonexistent.tool",
                                "arguments": {},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                # tools/call catches exceptions and returns isError:true in result
                assert resp["jsonrpc"] == "2.0"
                assert resp["id"] == 99
                assert "result" in resp
                assert resp["result"]["isError"] is True
                assert "Unknown tool" in resp["result"]["content"][0]["text"]
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_invalid_jsonrpc_method_returns_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 7,
                            "method": "nonexistent_method",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert resp["jsonrpc"] == "2.0"
                assert resp["id"] == 7
                assert "error" in resp
                assert resp["error"]["code"] == -32601
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_gap_check(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                # Save a memory first for gap_check to have something to recall
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.save",
                                "arguments": {"text": "妈妈今天做了红烧肉，很好吃"},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                _recv(proc)

                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.gap_check",
                                "arguments": {"query": "妈妈做了什么"},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                content = resp["result"]["content"]
                gap_data = json.loads(content[0]["text"])
                assert "sufficient" in gap_data
                assert "missing" in gap_data
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_compact(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                # Compact requires an existing raw_log_id. We expect an error
                # when the raw_log doesn't exist.
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.compact",
                                "arguments": {
                                    "raw_log_id": "raw_nonexistent",
                                    "distilled_text": "distilled version",
                                },
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                # tools/call catches the exception and returns isError:true in result
                assert "result" in resp
                assert resp["result"]["isError"] is True
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_initialize(self):
        """initialize should return protocol version, capabilities, and server info."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "test-client", "version": "1.0"},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert resp["jsonrpc"] == "2.0"
                assert resp["id"] == 1
                result = resp["result"]
                assert result["protocolVersion"] == "2024-11-05"
                assert "tools" in result["capabilities"]
                assert result["serverInfo"]["name"] == "engram-router"
                assert result["serverInfo"]["version"] == "0.1.0"
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_initialized_notification(self):
        """initialized notification should not return a response."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                # Send initialize first (with id, gets response)
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 0,
                            "method": "initialize",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                _recv(proc)  # consume initialize response

                # Send initialized notification (no id)
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "method": "initialized",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                # Then send a normal request to verify server is still alive
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/list",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                # Should get the tools/list response, not a notification response
                assert resp["id"] == 1
                assert "result" in resp
                assert "tools" in resp["result"]
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_notification_no_response_on_error(self):
        """Notification with an error should not leak a response to stdout."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                # Send tools/call as notification (no id) with bad args
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "memory.save",
                                "arguments": {},  # missing required "text"
                            },
                        }
                    )
                )
                proc.stdin.flush()
                # Send a real request — should get THIS response, not any error
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/list",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                # Must receive the tools/list response, not a leaked error
                assert resp["id"] == 1
                assert "result" in resp
                assert "tools" in resp["result"]
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_consolidate_stub(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                _init_mcp(proc)

                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.consolidate",
                                "arguments": {},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                content = resp["result"]["content"]
                data = json.loads(content[0]["text"])
                assert data["status"] == "ok"
                assert "stats" in data
                assert "merged_entities" in data["stats"]
                assert "removed_edges" in data["stats"]
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    # --- New robustness tests ---

    def test_requires_initialize_before_tools(self):
        """tools/list and tools/call must be rejected before initialized notification."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                # tools/list before init — must return error
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/list",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert "error" in resp, f"Expected error, got: {resp}"
                assert resp["error"]["code"] == -32002
                assert resp["error"]["message"] == "Server not initialized"

                # tools/call before init — must return error
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.save",
                                "arguments": {"text": "hello"},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp2 = _recv(proc)
                assert "error" in resp2, f"Expected error, got: {resp2}"
                assert resp2["error"]["code"] == -32002
                assert resp2["error"]["message"] == "Server not initialized"
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_ping(self):
        """ping method should return empty object {}."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "ping",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert resp["jsonrpc"] == "2.0"
                assert resp["id"] == 1
                assert "result" in resp
                assert resp["result"] == {}
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    # --- Security / robustness tests ---

    def test_auth_missing_api_key_returns_error(self):
        """tools/list without ENGRAM_API_KEY must return -32001 error."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name, api_key=None)
            try:
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/list",
                            "params": {},
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert "error" in resp, f"Expected error, got: {resp}"
                assert resp["error"]["code"] == -32001, f"Expected -32001, got: {resp}"
                assert "ENGRAM_API_KEY" in resp["error"]["message"]
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_auth_missing_api_key_tools_call(self):
        """tools/call without ENGRAM_API_KEY must return -32001 error."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name, api_key=None)
            try:
                _init_mcp(proc)
                proc.stdin.write(
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "memory.save",
                                "arguments": {"text": "hello"},
                            },
                        }
                    )
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert "error" in resp, f"Expected error, got: {resp}"
                assert resp["error"]["code"] == -32001, f"Expected -32001, got: {resp}"
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)

    def test_line_too_long_returns_parse_error(self):
        """Request line exceeding 1MB must return PARSE_ERROR."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            proc = _start_mcp(tmp.name)
            try:
                # Confirm server is alive first
                proc.stdin.write(
                    _send({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}})
                )
                proc.stdin.flush()
                resp = _recv(proc)
                assert resp.get("result") == {}

                # Now send a line just over 1 MB
                big = "x" * (1_048_577)  # 1 MB + 1 byte
                proc.stdin.write(big.encode("utf-8") + b"\n")
                proc.stdin.flush()
                resp = _recv(proc)
                assert "error" in resp, f"Expected error, got: {resp}"
                assert resp["error"]["code"] == -32700, f"Expected PARSE_ERROR, got: {resp}"
            finally:
                proc.stdin.close()
                proc.wait(timeout=3)
