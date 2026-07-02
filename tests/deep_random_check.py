#!/usr/bin/env python3
"""
Deeper analysis: check if "pass" was genuine or just random chance.
For scenarios where all memories get the same score, the pass is spurious.
"""

import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from engram_router.store import MemoryStore


def deep_check():
    """For scenarios S5-S10, check if ranking was truly random."""
    
    checks = [
        {
            "id": "S7-deep",
            "query": "我最近胖了",
            "memories": [
                "我这个月体重增加了5公斤，得减肥了。",
                "我最近开始跑步，每天跑5公里。",
                "我换了个新手机，拍照效果不错。",
            ],
            "expected_word": "体重",
        },
        {
            "id": "S8-deep",
            "query": "我为什么心情不好",
            "memories": [
                "今天在会议室被老板批评了。",
                "中午吃了红烧肉，味道还行。",
                "昨天晚上睡得很好。",
            ],
            "expected_word": "批评",
        },
        {
            "id": "S9-deep",
            "query": "我之前说的那个计划",
            "memories": [
                "我想明年去日本旅游，看看樱花。",
                "今天的晚饭还不错。",
                "Python 3.12 的新特性挺多的。",
            ],
            "expected_word": "旅游",
        },
    ]
    
    d = tempfile.mkdtemp()
    
    for check in checks:
        db_path = Path(d) / f"{check['id']}.db"
        store = MemoryStore(path=db_path)
        
        for mem in check["memories"]:
            store.save(mem)
        
        records = store.recall(check["query"], top_k=5)
        
        print(f"\n=== {check['id']}: '{check['query']}' ===")
        print(f"Expected to find: '{check['expected_word']}'")
        
        all_same = len(set(r.score for r in records)) <= 2
        
        for i, r in enumerate(records):
            hit = "⭐" if check["expected_word"] in r.raw_text else "  "
            print(f"  [{i}] {hit} score={r.score:.4f} | rank_reason={r.match_reason[:60]}... | {r.raw_text[:60]}")
        
        if all_same and all(r.score < 1.0 for r in records):
            print(f"  🔴 VERDICT: All scores essentially identical ({records[0].score:.4f}) — ranking is RANDOM")
        else:
            print(f"  🟢 VERDICT: Scores show meaningful differentiation")
        
        store.close()


if __name__ == "__main__":
    deep_check()
