.PHONY: ci-gate test ci-gate-test eval eval-report

# CI gate: runs eval_v2 with gate thresholds.
# Requires ENGRAM_SKIP_VECTOR=1 for deterministic, fast evaluation.
ci-gate:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py --ci

# Run CI gate unit tests.
ci-gate-test:
	ENGRAM_SKIP_VECTOR=1 python -m pytest tests/test_ci_gate.py -q --tb=short

# Full eval run (no CI gate, saves report).
eval:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py

# Eval with held-out split (requires report.json as artifact for diff).
eval-held-out:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py --held-out

# PR shadow diff: compare main vs current.
pr-shadow-diff:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py --pr-shadow-diff-main docs/eval_v2_report.json

# Run all tests.
test:
	ENGRAM_SKIP_VECTOR=1 python -m pytest tests/ -q --tb=short

# Adversarial eval.
eval-adversarial:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py --adversarial

# Multilingual eval.
eval-multilingual:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py --multilingual

# Multi-angle CI gate: precision >= 85%, isolation >= 85%.
ci-gate-multi:
	ENGRAM_SKIP_VECTOR=1 python tests/multi_angle_eval.py
