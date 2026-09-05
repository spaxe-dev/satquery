# Evaluation Harness

`python eval/runners/run_baseline.py` → `eval/reports/baseline_<impl>_<ts>.json` (6 cases: 5 REGRESSION demos + 1 QUALITATIVE grounding probe). No ground truth invented; BENCHMARK type reserved until labels exist. Baseline 2026-09-05: 6/6 passed heuristic. Same cases run REAL MODEL later via `SATQUERY_EVAL_IMPL` env + registry selection. Metrics: pass = refusal matches + task matches + trace length + answer non-empty. Perf extension: `time_s` per case recorded; detailed raster/model/tool timings are Phase 19 manual (see handoff).
