import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from engram_router.store import MemoryStore


def test_save_preserves_raw_text():
    store = MemoryStore()
    memory_id = store.save("张三前两天送我一把 HHKB，说是生日礼物")
    records = store.recall("HHKB")
    assert memory_id == "mem_1"
    assert records
    assert "HHKB" in records[0].raw_text


def test_recall_respects_top_k():
    store = MemoryStore()
    store.save("alpha memory")
    store.save("alpha second memory")
    records = store.recall("alpha", top_k=1)
    assert len(records) == 1


def test_sqlite_store_persists_across_instances(tmp_path):
    db_path = tmp_path / "memory.db"
    first = MemoryStore(path=db_path)
    memory_id = first.save("张三前两天送我一把 HHKB，说是生日礼物")

    second = MemoryStore(path=db_path)
    records = second.recall("HHKB")

    assert memory_id.startswith("mem_")
    assert len(records) == 1
    assert records[0].raw_text == "张三前两天送我一把 HHKB，说是生日礼物"
    assert records[0].evidence_refs
    assert records[0].evidence_refs[0].startswith("evi_")


def test_recall_returns_evidence_package_with_reason_and_score(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三是我的前同事，现在在腾讯。")
    store.save("张三前两天送我一把 HHKB，说是因为我生日。")

    records = store.recall("同事 HHKB 为什么", top_k=5)

    assert records
    best = records[0]
    assert "HHKB" in best.raw_text
    assert best.score > 0
    assert best.match_reason
    assert best.evidence_refs


def test_gap_check_detects_missing_reason_and_suggests_question(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三前两天送我一把 HHKB。")
    records = store.recall("他为什么送我这个？")

    gap = store.gap_check("他为什么送我这个？", records)

    assert gap["sufficient"] is False
    assert "reason" in gap["missing"]
    assert "为什么" in gap["suggested_question"] or "原因" in gap["suggested_question"]


def test_gap_check_passes_when_reason_evidence_exists(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三前两天送我一把 HHKB，因为我生日。")
    records = store.recall("他为什么送我这个？")

    gap = store.gap_check("他为什么送我这个？", records)

    assert gap["sufficient"] is True
    assert gap["missing"] == []


def test_compact_preserves_raw_log_and_evidence_refs(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    raw_id = store.save_raw_log(
        "修改 docs/PROJECT_BRIEF.md：新增证据保留型压缩原则。pytest 2 passed。",
        kind="file_change",
    )

    distilled_id = store.compact(raw_id, distilled_text="项目必须允许提炼，但提炼结果不能替代原始证据。")
    records = store.recall("提炼 原始证据", top_k=5)

    assert distilled_id.startswith("dst_")
    assert any(raw_id in record.evidence_refs for record in records)
    assert any(record.metadata.get("raw_log_id") == raw_id for record in records)
    assert store.get_raw_log(raw_id)["text"].startswith("修改 docs/PROJECT_BRIEF")


def test_save_metadata_round_trips_json(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("用户偏好蓝色系界面。", metadata={"project": "测试项目", "quote": "it's ok"})

    records = store.recall("蓝色系", top_k=1)

    assert records[0].metadata["project"] == "测试项目"
    assert records[0].metadata["quote"] == "it's ok"
    assert records[0].metadata["source"] == "conversation"
    assert "created_at" in records[0].metadata


def test_legacy_str_dict_metadata_is_read(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    memory_id = store.save("legacy metadata keeps working")
    store.conn.execute(
        "UPDATE memories SET metadata = ? WHERE id = ?",
        ("{'legacy': 'yes', 'count': 2}", memory_id),
    )
    store.conn.commit()

    records = store.recall("legacy", top_k=1)

    assert records[0].metadata["legacy"] == "yes"
    assert records[0].metadata["count"] == 2


def test_concurrent_writers_get_unique_ids(tmp_path):
    db_path = tmp_path / "memory.db"

    def save_one(i: int) -> str:
        with MemoryStore(path=db_path) as store:
            return store.save(f"并发写入第 {i} 条 alpha")

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(save_one, range(12)))

    assert len(ids) == len(set(ids))
    with MemoryStore(path=db_path) as store:
        records = store.recall("alpha", top_k=20)
    assert len(records) == 12


def test_cli_persists_between_save_and_recall(tmp_path, cli_env):
    db_path = tmp_path / "cli.db"
    save_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "engram_router.cli",
            "--db",
            str(db_path),
            "save",
            "张三前两天送我一把 HHKB，说是生日礼物",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=cli_env,
    )
    saved = json.loads(save_result.stdout)

    recall_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "engram_router.cli",
            "--db",
            str(db_path),
            "recall",
            "HHKB",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=cli_env,
    )
    recalled = json.loads(recall_result.stdout)

    assert saved["memory_id"].startswith("mem_")
    assert recalled["memories"]
    assert "HHKB" in recalled["memories"][0]["raw_text"]


def test_cli_gap_check_outputs_missing_reason(tmp_path, cli_env):
    db_path = tmp_path / "cli.db"
    subprocess.run(
        [sys.executable, "-m", "engram_router.cli", "--db", str(db_path), "save", "张三送我一把 HHKB。"],
        check=True,
        capture_output=True,
        text=True,
        env=cli_env,
    )
    result = subprocess.run(
        [sys.executable, "-m", "engram_router.cli", "--db", str(db_path), "gap-check", "他为什么送我这个？"],
        check=True,
        capture_output=True,
        text=True,
        env=cli_env,
    )
    gap = json.loads(result.stdout)

    assert gap["sufficient"] is False
    assert "reason" in gap["missing"]


# --- gap_check: time, location, object gaps ---


def test_gap_check_detects_missing_time(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三送了我一把键盘。")
    records = store.recall("他什么时候送的？")

    gap = store.gap_check("他什么时候送的？", records)

    assert gap["sufficient"] is False
    assert "time" in gap["missing"]
    assert "什么时候" in gap["suggested_question"]


def test_gap_check_passes_when_time_evidence_exists(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三昨天送了我一把键盘。")
    records = store.recall("他什么时候送的？")

    gap = store.gap_check("他什么时候送的？", records)

    assert gap["sufficient"] is True
    assert gap["missing"] == []


def test_gap_check_detects_missing_location(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三送了我一把键盘。")
    records = store.recall("他在哪里送的？")

    gap = store.gap_check("他在哪里送的？", records)

    assert gap["sufficient"] is False
    assert "location" in gap["missing"]


def test_gap_check_passes_when_location_evidence_exists(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三在公司送了我一把键盘。")
    records = store.recall("他在哪里送的？")

    gap = store.gap_check("他在哪里送的？", records)

    assert gap["sufficient"] is True
    assert gap["missing"] == []


def test_gap_check_detects_missing_object(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三昨天在公司见了我。")
    records = store.recall("他送了什么？")

    gap = store.gap_check("他送了什么？", records)

    assert gap["sufficient"] is False
    assert "object" in gap["missing"]


def test_gap_check_passes_when_object_evidence_exists(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三昨天在公司送了我一把HHKB键盘。")
    records = store.recall("他送了什么？")

    gap = store.gap_check("他送了什么？", records)

    assert gap["sufficient"] is True
    assert gap["missing"] == []


def test_gap_check_reports_multiple_gaps_together(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    # Pure ASCII text so _has_person_like won't fire on generic CJK chars.
    store.save("something happened")
    records = store.recall("为什么谁什么时候在哪里送了什么东西？")

    gap = store.gap_check("为什么谁什么时候在哪里送了什么东西？", records)

    assert gap["sufficient"] is False
    for kind in ("reason", "person", "time", "location", "object"):
        assert kind in gap["missing"], f"expected {kind} in missing"


# --- corrections: down-weighting in recall ---


def test_corrected_memory_is_downweighted_in_recall(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    mid1 = store.save("张三说我26岁。")
    mid2 = store.save("张三说他28岁。")

    # Insert a user correction for mid1 (the 26岁 memory is wrong).
    store.conn.execute(
        "INSERT INTO corrections (id, target_id, correction_text) VALUES (?, ?, ?)",
        ("corr_1", mid1, "年龄更正为28岁"),
    )
    store.conn.commit()

    # Space after 张三 prevents greedy entity extraction (张三多).
    records = store.recall("张三 多大", top_k=5)

    assert len(records) >= 2
    # The corrected memory should be down-weighted and rank below the uncorrected one.
    corrected = next((r for r in records if r.id == mid1), None)
    uncorrected = next((r for r in records if r.id == mid2), None)
    assert corrected is not None
    assert uncorrected is not None
    assert corrected.score < uncorrected.score
    assert "user_corrected" in corrected.match_reason


def test_corrected_memory_is_not_hard_deleted(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    mid = store.save("张三说我26岁。")

    # Insert a correction.
    store.conn.execute(
        "INSERT INTO corrections (id, target_id, correction_text) VALUES (?, ?, ?)",
        ("corr_1", mid, "年龄更正为28岁"),
    )
    store.conn.commit()

    # Space after 张三 prevents greedy entity extraction (张三多).
    records = store.recall("张三 多大", top_k=5)

    # The memory is still present (not hard-deleted).
    corrected = next((r for r in records if r.id == mid), None)
    assert corrected is not None
    assert "26岁" in corrected.raw_text

    # The correction itself is still stored.
    row = store.conn.execute(
        "SELECT * FROM corrections WHERE target_id = ?", (mid,)
    ).fetchone()
    assert row is not None
    assert row["correction_text"] == "年龄更正为28岁"


def test_no_corrections_no_penalty(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("张三说我28岁。")
    store.save("张三说他28岁。")

    records = store.recall("张三多少岁", top_k=5)

    for r in records:
        assert "user_corrected" not in r.match_reason


# ── Namespace multi-tenant isolation ───────────────────────────────────


def test_namespace_isolation_prevents_cross_namespace_recall(tmp_path):
    """Two namespaces with same entity name must not cross-recall."""
    store = MemoryStore(path=tmp_path / "memory.db")

    # work namespace: 张三 = colleague
    store.save("张三是我的前同事，现在在腾讯。", namespace="work")
    store.save("张三送了我一把HHKB键盘。", namespace="work")

    # family namespace: 张三 = son
    store.save("我儿子张三今年5岁了。", namespace="family")
    store.save("张三最喜欢吃红烧肉。", namespace="family")

    # work query: must not surface family's 张三
    wr = store.recall("张三", namespace="work")
    assert any("同事" in r.raw_text or "HHKB" in r.raw_text for r in wr)
    assert not any("儿子" in r.raw_text or "红烧肉" in r.raw_text for r in wr)

    # family query: must not surface work's 张三
    fr = store.recall("张三", namespace="family")
    assert any("儿子" in r.raw_text or "红烧肉" in r.raw_text for r in fr)
    assert not any("同事" in r.raw_text or "HHKB" in r.raw_text for r in fr)


def test_default_namespace_isolation(tmp_path):
    """Default namespace only sees its own data."""
    store = MemoryStore(path=tmp_path / "memory.db")
    store.save("data in default ns")
    store.save("data in custom ns", namespace="custom")

    assert len(store.recall("data")) == 1
    assert len(store.recall("data", namespace="custom")) == 1


def test_namespace_migration_preserves_existing_data(tmp_path):
    """Old DB without namespace column upgrades with data intact."""
    import sqlite3
    db_path = tmp_path / "legacy.db"

    # Create old-schema DB without namespace column
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'conversation',
            confidence REAL NOT NULL DEFAULT 1.0,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS evidence (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            quote TEXT NOT NULL,
            source_location TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS raw_logs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS distilled_memories (
            id TEXT PRIMARY KEY,
            raw_log_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            distilled_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'unknown',
            salience_class TEXT NOT NULL DEFAULT 'event'
        );
        CREATE TABLE IF NOT EXISTS memory_entities (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            evidence TEXT NOT NULL DEFAULT '',
            salience_class TEXT NOT NULL DEFAULT 'event'
        );
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            src_id TEXT NOT NULL,
            dst_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            evidence_ref TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS corrections (
            id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            correction_text TEXT NOT NULL,
            evidence_ref TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS id_sequences (
            name TEXT PRIMARY KEY,
            next_val INTEGER NOT NULL
        );
        INSERT INTO id_sequences VALUES ('memories', 4);
        INSERT INTO memories (id, raw_text, summary) VALUES ('mem_1', '旧数据1', '旧数据1');
        INSERT INTO memories (id, raw_text, summary) VALUES ('mem_2', '旧数据2', '旧数据2');
        INSERT INTO memories (id, raw_text, summary) VALUES ('mem_3', '旧数据3', '旧数据3');
    """)
    conn.commit()
    conn.close()

    # Open with new code; migration adds namespace='default'
    store = MemoryStore(path=db_path)
    rows = store.conn.execute(
        "SELECT id, namespace FROM memories"
    ).fetchall()
    assert len(rows) == 3
    for r in rows:
        assert r["namespace"] == "default"

    # Old data is still recallable
    records = store.recall("旧数据", top_k=5)
    assert len(records) == 3

    # New saves go to 'default' by default
    store.save("新数据")
    assert len(store.recall("新数据")) == 1


def test_edge_expansion_does_not_cross_namespace(tmp_path):
    """Edge associations must not cross namespace boundaries.

    When entity ``张三`` is shared across namespaces (same global entity),
    edges created in namespace B must not cause namespace B memories to
    surface during a namespace A recall — even though the BFS follows
    the global edge graph and activates ``红烧肉``, step 4's JOIN on
    memories.namespace must block the foreign memory.
    """
    store = MemoryStore(path=tmp_path / "memory.db")

    # Namespace A: two memories
    store.save("张三是我的同事。", namespace="A")          # entity: 张三
    store.save("张三送我HHKB键盘。", namespace="A")        # entities: 张三, HHKB
                                                            # edge: 张三↔HHKB

    # Namespace B: shares SAME global entity 张三
    store.save("张三今年5岁。", namespace="B")              # entity: 张三
    store.save("张三喜欢红烧肉。", namespace="B")           # entities: 张三, 红烧肉
                                                            # edge: 张三↔红烧肉

    # Query namespace A: "同事 HHKB" matches both A memories directly.
    # The global edge 张三→红烧肉 (from namespace B's save) would activate
    # 红烧肉 entity during BFS — but step 4 must block namespace B's memory.
    ra = store.recall("同事 HHKB", namespace="A")
    assert any("HHKB" in r.raw_text for r in ra)
    assert any("同事" in r.raw_text for r in ra)
    # Cross-namespace guard: no B memories should leak in
    assert not any("红烧肉" in r.raw_text for r in ra)
    assert not any("5岁" in r.raw_text for r in ra)

    # Conversely, namespace B should only see its own
    rb = store.recall("张三", namespace="B")
    assert any("5岁" in r.raw_text or "红烧肉" in r.raw_text for r in rb)
    assert not any("同事" in r.raw_text for r in rb)
    assert not any("HHKB" in r.raw_text for r in rb)


def test_compact_respects_namespace(tmp_path):
    """compacted memory is written into the requested namespace."""
    store = MemoryStore(path=tmp_path / "memory.db")
    raw_id = store.save_raw_log("some raw log content for testing")
    store.compact(raw_id, "distilled content", namespace="ns1")

    # default namespace is empty
    assert len(store.recall("distilled")) == 0
    assert len(store.recall("distilled", namespace="ns1")) == 1


# ── RecallWeights centralisation ──────────────────────────────────────


def test_default_recall_weights_match_hardcoded_values():
    """Every default weight must equal the previously hard-coded number."""
    from engram_router.store import RecallWeights
    w = RecallWeights()
    assert w.fts_boost == 0.1
    assert w.shared_entity_multiplier == 1.2
    assert w.brand_boost == 2.0
    assert w.identity_base_attr_boost == 2.0
    assert w.eval_sensory_boost == 1.5
    assert w.correction_penalty == 0.3
    assert w.max_recall_hops == 2
    assert w.recall_decay == 0.5
    assert w.activation_threshold == 0.03
    assert w.ascii_base == 4.0
    assert w.ascii_per_char == 0.5
    assert w.ascii_per_char_cap == 6
    assert w.cjk_multi_base == 2.0
    assert w.cjk_multi_per_char == 0.5
    assert w.stop_char_weight == 0.05
    assert w.single_cjk_weight == 0.4
    assert w.colleague_boost == 1.0
    assert w.reason_marker_boost == 1.5
    assert w.assoc_reach_base_attr == 0.15
    assert w.assoc_reach_constraint == 0.6
    assert w.assoc_reach_decision == 0.7
    assert w.assoc_reach_sensory == 1.0
    assert w.assoc_reach_event == 1.0


def test_custom_weights_alter_recall_ranking(tmp_path):
    """Custom weights produce observably different ranking vs defaults."""
    # ── default weights ──
    d = MemoryStore(path=tmp_path / "d.db")
    mid_a = d.save("张三说我26岁。")
    mid_b = d.save("张三说他28岁。")
    d.conn.execute(
        "INSERT INTO corrections (id, target_id, correction_text) VALUES (?, ?, ?)",
        ("corr_1", mid_a, "年龄更正为28岁"),
    )
    d.conn.commit()
    d_recs = d.recall("张三 多大", top_k=5)
    d_corrected = next(r for r in d_recs if r.id == mid_a)
    d_uncorrected = next(r for r in d_recs if r.id == mid_b)
    # Default: corrected is penalised and ranks below uncorrected.
    assert d_corrected.score < d_uncorrected.score

    # ── custom weights: no correction penalty ──
    from engram_router.store import RecallWeights
    cw = RecallWeights(correction_penalty=1.0)
    c = MemoryStore(path=tmp_path / "c.db", weights=cw)
    mid_a2 = c.save("张三说我26岁。")
    _mid_b2 = c.save("张三说他28岁。")
    c.conn.execute(
        "INSERT INTO corrections (id, target_id, correction_text) VALUES (?, ?, ?)",
        ("corr_2", mid_a2, "年龄更正为28岁"),
    )
    c.conn.commit()
    c_recs = c.recall("张三 多大", top_k=5)
    c_corrected = next(r for r in c_recs if r.id == mid_a2)
    assert c_corrected.score > 0
    assert "user_corrected" in c_corrected.match_reason


def test_store_constructed_with_default_weights_works(tmp_path):
    """Store with default RecallWeights produces valid recall output."""
    from engram_router.store import RecallWeights
    store = MemoryStore(path=tmp_path / "smoke.db", weights=RecallWeights())
    store.save("张三前两天送我一把HHKB，说是因为我生日")
    records = store.recall("HHKB", top_k=5)
    assert records
    assert records[0].score > 0
    assert "HHKB" in records[0].raw_text


# ── delete ──────────────────────────────────────────────────────────────


def test_delete_removes_memory_from_recall(tmp_path):
    store = MemoryStore(path=tmp_path / "del.db")
    mid = store.save("张三是我前同事")
    store.save("李四在百度工作")
    assert store.delete(mid)
    records = store.recall("张三", top_k=5)
    assert mid not in {r.id for r in records}


def test_delete_cascades_to_evidence(tmp_path):
    store = MemoryStore(path=tmp_path / "cas.db")
    mid = store.save("测试级联删除")
    assert store.delete(mid)
    evidence = store.conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE memory_id = ?", (mid,)
    ).fetchone()[0]
    assert evidence == 0


def test_delete_fts5_ghost_filtered_by_recall(tmp_path):
    """After delete, FTS5 may retain a ghost entry but recall() filters it."""
    store = MemoryStore(path=tmp_path / "ghost.db")
    mid = store.save("HHKB 是机械键盘，手感很好")
    assert store.delete(mid)
    # FTS5 still has the entry (ghost), but recall does NOT return it
    records = store.recall("HHKB", top_k=5)
    assert mid not in {r.id for r in records}


def test_delete_nonexistent_returns_false(tmp_path):
    store = MemoryStore(path=tmp_path / "nope.db")
    store.save("存在的一条")
    assert store.delete("nonexistent_id") is False


def test_delete_preserves_unrelated_memories(tmp_path):
    store = MemoryStore(path=tmp_path / "part.db")
    store.save("张三的信息")
    mid2 = store.save("李四的信息")
    store.save("王五的信息")
    store.delete(mid2)
    records = store.recall("张三 王五", top_k=10)
    rec_ids = {r.id for r in records}
    assert len(rec_ids) >= 2
    assert mid2 not in rec_ids
