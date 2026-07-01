"""Phase 2: lightweight entity extraction tests (RED first).

EngramRouter should pull simple, high-signal entities out of each saved turn
so that recall can hop across turns that share no surface tokens.

Concrete failure we target:
  Q: 我那个同事送我的键盘是什么牌子？   (mentions 键盘, not HHKB)
  A-turn: 张三前两天送了我一把 HHKB。  (mentions HHKB, not 键盘)
These two share no multi-char token, so pure keyword recall misses the brand.
With entity extraction we record HHKB as an OBJECT entity and 键盘 as a topic,
linked through the gift event, so recalling on 键盘 can surface the HHKB turn.

We keep extraction deliberately rule-based and conservative (no LLM): only
patterns we can defend. Anything fancier is a later phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engram_router.entities import extract_entities  # noqa: E402
from engram_router.store import MemoryStore  # noqa: E402


def test_extracts_ascii_object_token():
    ents = extract_entities("张三前两天送了我一把 HHKB。")
    kinds = {(e["name"], e["kind"]) for e in ents}
    assert ("HHKB", "object") in kinds
    # 张三 is a person.
    assert ("张三", "person") in kinds


def test_extracts_company_and_reason():
    ents = extract_entities("张三是我前同事，现在在腾讯。")
    names = {e["name"] for e in ents}
    assert "张三" in names
    assert "腾讯" in names
    reason_ents = extract_entities("他说是因为我生日，知道我一直喜欢键盘。")
    assert any(e["kind"] == "reason" for e in reason_ents)


def test_extracts_time_expression():
    ents = extract_entities("张三前两天送了我一把 HHKB。")
    assert any(e["kind"] == "time" for e in ents)


def test_store_persists_entities_on_save():
    store = MemoryStore(path=None)
    mem_id = store.save("张三前两天送了我一把 HHKB。")
    ents = store.entities_for(mem_id)
    names = {e["name"] for e in ents}
    assert "HHKB" in names
    assert "张三" in names


def test_recall_hops_through_shared_entity_topic():
    """The brand question shares no token with the answer turn, but both
    relate to 键盘 -> HHKB. After entity linking, recalling on the brand
    question should surface the HHKB turn in the top results."""
    store = MemoryStore(path=None)
    turns = [
        "张三是我前同事，现在在腾讯。",
        "我最近一直在看机械键盘。",
        "张三前两天送了我一把 HHKB。",
        "他说是因为我生日，知道我一直喜欢键盘。",
        "今天我们聊别的事情，暂时不提键盘了。",
    ]
    for t in turns:
        store.save(t)
    results = store.recall("我那个同事送我的键盘是什么牌子？", top_k=5)
    joined = " ".join(r.raw_text for r in results)
    assert "HHKB" in joined, f"HHKB not surfaced (suppression); got: {joined}"
