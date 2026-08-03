"""Tests for N3 — namespace-scoped cross-tenant isolation."""

from __future__ import annotations

import os
import tempfile

import pytest

from engram_router import MemoryStore


class TestDeleteNamespace:
    """memory.delete — must verify namespace ownership before deleting."""

    def test_delete_wrong_namespace_fails(self):
        """Deleting a memory in ns B should fail when called with ns A."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            mid = s.save("Bob's memory", namespace="tenant_a")

            # Try to delete with wrong namespace
            deleted = s.delete(mid, namespace="tenant_b")
            assert deleted is False, "Should NOT delete memory from wrong namespace"

            # Memory should still exist under tenant_a
            rows = s.conn.execute(
                "SELECT id FROM memories WHERE id = ?", (mid,)
            ).fetchall()
            assert len(rows) == 1

            s.close()

    def test_delete_correct_namespace_succeeds(self):
        """Deleting a memory in ns A with namespace=A should work."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            mid = s.save("Alice's memory", namespace="tenant_a")

            deleted = s.delete(mid, namespace="tenant_a")
            assert deleted is True

            rows = s.conn.execute(
                "SELECT id FROM memories WHERE id = ?", (mid,)
            ).fetchall()
            assert len(rows) == 0

            s.close()

    def test_delete_bogus_id_returns_false(self):
        """Deleting a nonexistent ID should return False."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            deleted = s.delete("nonexistent_id", namespace="default")
            assert deleted is False
            s.close()


class TestEntityNamespace:
    """Entities must be namespace-scoped via _get_or_create_entity."""

    def test_same_name_different_namespace_separate_entities(self):
        """Bob in ns A and Bob in ns B must be different entity rows."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            s.save("Bob 喜欢猫", namespace="tenant_a")
            s.save("Bob 喜欢狗", namespace="tenant_b")

            ents = s.conn.execute(
                "SELECT name, namespace FROM entities WHERE name = 'Bob' ORDER BY namespace"
            ).fetchall()
            assert len(ents) == 2, (
                f"Expected 2 separate Bob entities, got {len(ents)}"
            )
            namespaces = {e["namespace"] for e in ents}
            assert namespaces == {"tenant_a", "tenant_b"}

            s.close()

    def test_entity_retrieval_namespace_scoped(self):
        """Getting entities should filter by namespace."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            s.save("Alice works at Google", namespace="tenant_a")
            s.save("Alice works at Apple", namespace="tenant_b")

            # tenant_a should only see its Alice entity
            ents_a = s.conn.execute(
                "SELECT name, namespace FROM entities WHERE namespace = 'tenant_a'"
            ).fetchall()
            assert all(e["namespace"] == "tenant_a" for e in ents_a)

            # tenant_b should only see its Alice entity
            ents_b = s.conn.execute(
                "SELECT name, namespace FROM entities WHERE namespace = 'tenant_b'"
            ).fetchall()
            assert all(e["namespace"] == "tenant_b" for e in ents_b)

            s.close()


class TestConsolidateNamespace:
    """consolidate() must only merge entities within the same namespace."""

    def test_consolidate_does_not_cross_namespaces(self):
        """Entity alias merging must not cross namespace boundaries."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            s.save("alice 在北京工作", namespace="tenant_a")
            s.save("Alice 在上海工作", namespace="tenant_b")

            # Pre-consolidation: count entities per namespace
            before_a = len(s.conn.execute(
                "SELECT id FROM entities WHERE namespace = 'tenant_a'"
            ).fetchall())
            before_b = len(s.conn.execute(
                "SELECT id FROM entities WHERE namespace = 'tenant_b'"
            ).fetchall())

            result = s.consolidate(namespace="tenant_a")
            after_b = len(s.conn.execute(
                "SELECT id FROM entities WHERE namespace = 'tenant_b'"
            ).fetchall())
            # tenant_b entities must be untouched by tenant_a's consolidate
            assert after_b == before_b, (
                f"Cross-namespace consolidation leaked! "
                f"tenant_b before={before_b}, after={after_b}"
            )

            s.close()

    def test_consolidate_merges_same_namespace_aliases(self):
        """Same-name variants within a single namespace should be merged."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            s.save("Alice 喜欢设计", namespace="tenant_a")
            s.save("alice 喜欢前端", namespace="tenant_a")
            s.save("Alice 住在上海", namespace="tenant_a")

            before = len(s.conn.execute(
                "SELECT id FROM entities WHERE namespace = 'tenant_a'"
            ).fetchall())

            result = s.consolidate(namespace="tenant_a")
            after = len(s.conn.execute(
                "SELECT id FROM entities WHERE namespace = 'tenant_a'"
            ).fetchall())

            assert after < before, (
                f"Expected entity merge within namespace, before={before}, after={after}"
            )
            assert result["merged_entities"] > 0

            s.close()

    def test_consolidate_edge_cleanup_stays_scoped(self):
        """Edge dedup/cleanup in consolidate should only affect target namespace."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            s.save("Alice gave Bob a book", namespace="tenant_a")
            s.save("Charlie gave Dave a pen", namespace="tenant_b")

            result = s.consolidate(namespace="tenant_a")
            # Edges in tenant_b should still exist
            edges_b = s.conn.execute(
                "SELECT COUNT(*) as cnt FROM edges WHERE namespace = 'tenant_b'"
            ).fetchone()
            assert edges_b["cnt"] > 0, (
                "tenant_b edges were removed by tenant_a consolidate!"
            )

            s.close()


class TestTimedEventsNamespace:
    """timed_events must store and scope by namespace."""

    def test_timed_events_namespace_stored(self):
        """Populated timed_events should carry the same namespace as the memory."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            s.save("昨天在北京开会", namespace="tenant_a")

            events = s.conn.execute(
                "SELECT namespace FROM timed_events"
            ).fetchall()
            for e in events:
                assert e["namespace"] == "tenant_a", (
                    f"Timed event NS mismatch: {e['namespace']} != tenant_a"
                )

            s.close()


class TestMCPNamespace:
    """MCP tools should pass namespace through to store methods."""

    def test_delete_mcp_passes_namespace(self):
        """MCP _tool_memory_delete should forward namespace."""
        from engram_router.mcp_server import JsonRpcHandler

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            mid = s.save("test memory", namespace="my_ns")

            import asyncio
            handler = JsonRpcHandler(s, api_key="test-key")

            # Delete with matching namespace should succeed
            async def run():
                await handler._handle_initialize({})
                await handler._handle_initialized({})
                return await handler._handle_tools_call({
                    "name": "memory.delete",
                    "arguments": {"memory_id": mid, "namespace": "my_ns"},
                })

            result = asyncio.run(run())
            content = result["content"][0]
            assert "true" in content["text"].lower() or "True" in content["text"]

            s.close()

    def test_delete_mismatched_namespace(self):
        """MCP delete with wrong namespace should return false."""
        from engram_router.mcp_server import JsonRpcHandler

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            s = MemoryStore(path=tmp.name, enable_vector=False)
            mid = s.save("test memory", namespace="tenant_x")

            import asyncio
            handler = JsonRpcHandler(s, api_key="test-key")

            async def run():
                await handler._handle_initialize({})
                await handler._handle_initialized({})
                return await handler._handle_tools_call({
                    "name": "memory.delete",
                    "arguments": {"memory_id": mid, "namespace": "tenant_y"},
                })

            result = asyncio.run(run())
            content = result["content"][0]
            assert "false" in content["text"].lower() or "False" in content["text"]

            s.close()
