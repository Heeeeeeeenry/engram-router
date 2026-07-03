"""Phase 2: evidence-vs-summary benchmark.

This module proves the EngramRouter thesis quantitatively: an on-demand
evidence-recall memory answers history-dependent questions that a rolling
lossy summary cannot.

It compares two strategies over the same conversation + question set:

  SummaryBaseline
    Builds a short rolling summary of the conversation (the kind of lossy
    compression EngramRouter argues against) and answers from that summary
    only.

  EngramRouter recall
    Saves every conversation turn into the SQLite store and answers each
    question by recalling top-k evidence and reading the raw text.

The benchmark reports, per strategy:
  - answer_hits     : cases whose expected answer tokens appear in the answer
  - evidence_hits   : cases whose supporting evidence text was surfaced
  - context_chars   : size of the context each strategy puts in front of the model
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .store import MemoryStore


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    answer_contains: list[str] = field(default_factory=list)
    evidence_contains: list[str] = field(default_factory=list)
    # Anti-contamination: tokens that must NOT appear in recalled context.
    # These catch a ranker dragging a wrong-entity fact (妈妈's 55岁 onto a
    # 张三 age question, the user's 特斯拉 onto 李四's car) into the answer.
    answer_excludes: list[str] = field(default_factory=list)
    # "pass"      -> hard gate; a regression run fails if this case fails.
    # "known_gap" -> tracked debt; reported but never fails the run. Flip to
    #                "pass" once the relevant ranker fix lands.
    expect: str = "pass"
    # Short human label for auditing per-case output.
    tag: str = ""


def load_conversation(path: str | Path) -> list[str]:
    """Extract the ordered user turns from the demo conversation markdown.

    Lines look like ``1. User: 张三是我前同事，现在在腾讯。`` inside the
    ``## Conversation Events`` section. We deliberately keep the final
    meta-turn (the question turn) out of the saved evidence.
    """
    text = Path(path).read_text(encoding="utf-8")
    turns: list[str] = []
    in_events = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Conversation Events"):
            in_events = True
            continue
        if in_events and stripped.startswith("## "):
            break
        if not in_events:
            continue
        # Match "<n>. User: <content>"
        if ". User:" in stripped:
            content = stripped.split("User:", 1)[1].strip()
            # Skip the trailing meta-turn where the user *asks* the questions.
            if content.startswith("很多轮之后") or "我问：" in content:
                continue
            if content:
                turns.append(content)
    return turns


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cases.append(
            BenchmarkCase(
                query=obj["query"],
                answer_contains=list(obj.get("answer_contains", [])),
                evidence_contains=list(obj.get("evidence_contains", [])),
                answer_excludes=list(obj.get("answer_excludes", [])),
                expect=obj.get("expect", "pass"),
                tag=obj.get("tag", ""),
            )
        )
    return cases


class SummaryBaseline:
    """A deliberately lossy rolling-summary baseline.

    It mimics what naive context compression does: collapse many turns into a
    short topical sentence. Specific facts (brand, company, reason) are lost,
    which is exactly the failure mode EngramRouter is built to avoid.
    """

    def __init__(self, turns: list[str]) -> None:
        self.turns = turns
        self.summary = self._compress(turns)

    @staticmethod
    def _compress(turns: list[str]) -> str:
        # Topical, detail-free rollup. This is the anti-pattern under test.
        people = "前同事张三" if any("张三" in t for t in turns) else "某人"
        topics = []
        if any("键盘" in t or "HHKB" in t for t in turns):
            topics.append("键盘相关")
        if any("礼物" in t or "送" in t for t in turns):
            topics.append("礼物相关")
        topic_text = "、".join(topics) if topics else "一些事情"
        return f"用户和{people}聊过{topic_text}内容。"

    def answer(self, query: str) -> str:
        # The baseline can only answer from its lossy summary.
        return self.summary


def _engram_answer(store: MemoryStore, query: str, top_k: int = 3) -> tuple[str, str]:
    """Return (answer_text, evidence_text) from evidence recall."""
    records = store.recall(query, top_k=top_k)
    answer_text = " ".join(r.raw_text for r in records)
    evidence_text = " ".join(
        f"{r.raw_text} [{','.join(r.evidence_refs or [])}]" for r in records
    )
    return answer_text, evidence_text


def _contains_all(haystack: str, needles: list[str]) -> bool:
    return all(n in haystack for n in needles) if needles else True


def _contains_none(haystack: str, needles: list[str]) -> bool:
    """True iff none of the forbidden tokens appear (anti-contamination)."""
    return all(n not in haystack for n in needles) if needles else True


def run_benchmark(
    turns: list[str],
    cases: list[BenchmarkCase],
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run both strategies over the cases and return a structured report."""
    store = MemoryStore(path=db_path)
    try:
        for turn in turns:
            store.save(turn)

        baseline = SummaryBaseline(turns)

        summary_answer_hits = 0
        summary_evidence_hits = 0
        engram_answer_hits = 0
        engram_evidence_hits = 0
        # Hard-gate accounting. A "pass" case is a regression gate; a "known_gap"
        # case is reported but never fails the run.
        hard_total = 0
        hard_passed = 0
        gap_total = 0
        gap_passed = 0
        hard_failures: list[dict[str, Any]] = []
        case_reports: list[dict[str, Any]] = []

        for case in cases:
            # Summary strategy.
            s_answer = baseline.answer(case.query)
            s_answer_ok = _contains_all(s_answer, case.answer_contains)
            s_evidence_ok = _contains_all(s_answer, case.evidence_contains)
            summary_answer_hits += int(s_answer_ok)
            summary_evidence_hits += int(s_evidence_ok)

            # EngramRouter strategy.
            e_answer, e_evidence = _engram_answer(store, case.query)
            e_answer_ok = _contains_all(e_answer, case.answer_contains)
            e_evidence_ok = _contains_all(e_evidence, case.evidence_contains)
            # Anti-contamination check runs against the recalled answer text.
            e_excludes_ok = _contains_none(e_answer, case.answer_excludes)
            engram_answer_hits += int(e_answer_ok)
            engram_evidence_hits += int(e_evidence_ok)

            # A case is satisfied when its positive tokens are all present and no
            # forbidden token leaked. Evidence is not part of the gate (some cases
            # assert only on the answer) but is reported for auditing.
            case_satisfied = e_answer_ok and e_excludes_ok
            leaked = [n for n in case.answer_excludes if n in e_answer]

            if case.expect == "known_gap":
                gap_total += 1
                gap_passed += int(case_satisfied)
            else:
                hard_total += 1
                hard_passed += int(case_satisfied)
                if not case_satisfied:
                    hard_failures.append(
                        {
                            "tag": case.tag,
                            "query": case.query,
                            "missing": [n for n in case.answer_contains if n not in e_answer],
                            "leaked": leaked,
                            "answer": e_answer,
                        }
                    )

            case_reports.append(
                {
                    "tag": case.tag,
                    "query": case.query,
                    "expect": case.expect,
                    "expected_answer": case.answer_contains,
                    "expected_evidence": case.evidence_contains,
                    "forbidden": case.answer_excludes,
                    "satisfied": case_satisfied,
                    "leaked": leaked,
                    "summary": {
                        "answer_ok": s_answer_ok,
                        "evidence_ok": s_evidence_ok,
                        "answer": s_answer,
                    },
                    "engram": {
                        "answer_ok": e_answer_ok,
                        "evidence_ok": e_evidence_ok,
                        "excludes_ok": e_excludes_ok,
                        "answer": e_answer,
                        "evidence": e_evidence,
                    },
                }
            )

        total = len(cases)
        summary_context_chars = len(baseline.summary)
        engram_context_chars = sum(len(t) for t in turns)

        report = {
            "total": total,
            "summary": {
                "answer_hits": summary_answer_hits,
                "evidence_hits": summary_evidence_hits,
                "context_chars": summary_context_chars,
                "note": "Lossy rolling summary. Stored once, never recalls detail.",
            },
            "engram": {
                "answer_hits": engram_answer_hits,
                "evidence_hits": engram_evidence_hits,
                "context_chars_full_corpus": engram_context_chars,
                "note": "On-demand evidence recall. Only top-k evidence enters context per query.",
            },
            "gate": {
                "hard_total": hard_total,
                "hard_passed": hard_passed,
                "hard_failed": hard_total - hard_passed,
                "known_gap_total": gap_total,
                "known_gap_passed": gap_passed,
                "passed": hard_passed == hard_total,
                "failures": hard_failures,
            },
            "verdict": (
                "engram_better"
                if engram_answer_hits > summary_answer_hits
                else "no_improvement"
            ),
            "cases": case_reports,
        }
        return report
    finally:
        store.close()


def format_report(report: dict[str, Any]) -> str:
    """Human-readable plain-text rendering of a benchmark report."""
    lines: list[str] = []
    total = report["total"]
    s = report["summary"]
    e = report["engram"]
    lines.append("EngramRouter vs Summary Baseline")
    lines.append("=" * 36)
    lines.append(f"cases: {total}")
    lines.append("")
    lines.append("Summary baseline (lossy compression):")
    lines.append(f"  answer hits   : {s['answer_hits']}/{total}")
    lines.append(f"  evidence hits : {s['evidence_hits']}/{total}")
    lines.append(f"  context chars : {s['context_chars']}")
    lines.append("")
    lines.append("EngramRouter (evidence recall):")
    lines.append(f"  answer hits   : {e['answer_hits']}/{total}")
    lines.append(f"  evidence hits : {e['evidence_hits']}/{total}")
    lines.append(f"  full corpus   : {e['context_chars_full_corpus']} chars (only top-k recalled per query)")
    lines.append("")
    gate = report.get("gate")
    if gate:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"regression gate: {status}")
        lines.append(f"  hard gates  : {gate['hard_passed']}/{gate['hard_total']} passed")
        lines.append(f"  known gaps  : {gate['known_gap_passed']}/{gate['known_gap_total']} passed (not gated)")
        for fail in gate["failures"]:
            tag = fail.get("tag") or fail["query"]
            bits = []
            if fail["missing"]:
                bits.append(f"missing {fail['missing']}")
            if fail["leaked"]:
                bits.append(f"leaked {fail['leaked']}")
            lines.append(f"    FAIL [{tag}] {'; '.join(bits)}")
        lines.append("")
    lines.append(f"verdict: {report['verdict']}")
    return "\n".join(lines)
