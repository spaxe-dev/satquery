# Evaluation Harness (model/tool agnostic)

Types: REGRESSION (must not break), QUALITATIVE (human review, no ground truth), BENCHMARK (needs ground truth — none yet, so no accuracy claims).

Run: `python eval/runners/run_baseline.py` → `eval/reports/baseline_heuristic_<ts>.json`
Cases: `eval/cases/*.json` (same 5 demos + grounding probe). No invented ground truth: expected fields are `expected_task`, `expect_refusal`, `min_trace`, qualitative notes only.
