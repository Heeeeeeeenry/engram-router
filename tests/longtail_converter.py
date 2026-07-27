#!/usr/bin/env python3
"""Convert LongMemEval splits to eval_v2-compatible scenario JSON files.

Usage::

    python tests/longtail_converter.py --split s --output tests/data/longtail_500.json
    python tests/longtail_converter.py --split m --output tests/data/longtail_5000.json --limit 100

The output is a list of ``EvalScenario``-shaped dicts that ``eval_v2.py``
can consume directly with ``_build_scenarios_from_json()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))


def _build_scenarios(split: str, limit: int | None) -> list[dict[str, Any]]:
    """Load LongMemEval *split* and convert to EvalScenario JSON dicts."""
    from longmemeval_loader import load_longmemeval

    items = load_longmemeval(split, limit=limit)
    scenarios: list[dict[str, Any]] = []

    for case in items:
        # ── EvalCase ──
        eval_case: dict[str, Any] = {
            "id": case.question_id,
            "query": case.question,
            "expected_ids_or_texts": [case.answer],
            "forbidden_ids_or_texts": [],
            "is_negative": False,
            "description": (
                f"{case.question_type} | "
                f"{case.num_sessions}s | "
                f"{case.num_turns}t | "
                f"{len(case.memories)}m"
            ),
        }

        # ── EvalScenario ──
        scenario: dict[str, Any] = {
            "id": case.question_id,
            "source": f"longmemeval_{split}",
            "memories": case.memories,
            "cases": [eval_case],
        }
        scenarios.append(scenario)

    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval → eval_v2 scenario JSON converter")
    parser.add_argument(
        "--split", choices=["oracle", "s", "m"],
        default="s",
        help="LongMemEval split: oracle / s / m (default: s = 500-turn)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: tests/data/longtail_<split>.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max cases to convert (default: all)",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        _here / "data" / f"longtail_{args.split}.json"
    )

    scenarios = _build_scenarios(args.split, args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)

    total_mems = sum(len(s["memories"]) for s in scenarios)
    print(f"Wrote {len(scenarios)} scenarios ({total_mems} memories) → {output}")


if __name__ == "__main__":
    main()
