# Metrics & Benchmarks — What to Measure, How (2026-09-11)

No invented numbers. Every metric: definition / formula / why / how / baseline / sample / limits / PPT line.

## A. Trust metrics (highest jury value)

1. **Correct refusal / invalid-detection rate** = refused-invalid / total-invalid. Why: proves defensive vs hallucinating baselines. How: 50-case suite (dims, identical, nodata, corrupt, unsupported, ambiguous). Baseline: generic VLM ~0%, SatQuery now ~40% (dims only). Target ≥95% after 4d gates. PPT: bar chart refused vs hallucinated, n=50.
2. **Unsupported-answer rate** = unsupported-answered / unsupported-asked (lower better). How: 20 out-of-scope queries. Target <10%.
3. **Contradiction P/R** = flagged-disagree / true-disagree. How: synthetic agree (same mask) vs disagree (shifted/random) optical-SAR splits, n≥40. Target P≥80%, R≥75%. PPT: confusion matrix + IoU histogram with 0.12 threshold line.
4. **False-confidence rate** = high-conf (>0.8) wrong answers / all wrong. How: log heuristic overall vs human correctness on 50. Target: decrease after contradiction/quality cuts. Never claim calibrated.
5. **ECE / Brier / selective-risk** — log now, report as "being evaluated". Need ≥200 scored answers for stable ECE (10 bins). Do NOT put "calibrated" in PPT until measured. PPT line: "Heuristic evidence-strength; calibration under evaluation (ECE protocol defined)."

## B. Capability metrics (honest baselines)

6. **Routing accuracy** = correct task / total queries (≥60 template+paraphrase queries). Baseline regex unknown — measure today. Target ≥90%. Ablation: regex vs paraphrase drop.
7. **Change sanity**: flood demo 22.37% (retain 18-25% band) + SECOND-sample IoU/F1 (n=20, report, don't chase SOTA 0.8+; classical ~0.3-0.5 expected).
8. **Grounding IoU** on VRSBench-50 (report ~0.2-0.35 heuristic; Grounding-DINO later only if blocks score).
9. **VQA accuracy**: EuroSAT-val classification (>70% after LoRA expected) + RSVQA/VRSBench-50 sample (report, SOTA GeoChat/TEOChat far higher — acknowledge).
10. **Fusion lift**: water IoU agree-case + built-physics explanation; report mean IoU + high-conf vs union % (retain Pair1 0.61 water example as illustration, not claim).
11. **Latency**: single ~25ms, change ~270ms, fusion ~230ms, hero ~400ms CPU 512² (retain, re-measure). PPT: offline CPU bar.

## C. Operational metrics (workflow)

12. **Workflow compression** = manual steps / SatQuery steps. Manual baseline: measure with stopwatch on 2-3 analysts doing QGIS-equivalent (open, stretch, threshold, compare, screenshot, report) — expect 8-12 steps, 15-40min. SatQuery: 1 call, <1min + verify. Report range + n, never single invented %.
13. **Time-to-evidence** = query → visual overlay. Measure median over 20 runs. PPT: timeline screenshot.
14. **Automation rate** = 1 − manual-steps-after / manual-steps-before. Same protocol as 12.

## SatQuery Robustness Benchmark (provisional name — do not brand in PPT yet)

Categories (easiest first): refusal (dims/identical/nodata/corrupt) → routing (60 queries) → contradiction (40 synth pairs) → registration-shift (20 shifted pairs, ECC flag) → cloud-synth (20) → VQA/grounding-50 sample → change-20 sample → OOD-resize (EuroSAT 64→512 stress). Each case: input / expected (refuse/flag/answer+trace) / ground truth / metric / baseline (heuristic now, LoRA later) / viz (overlay + trace screenshot) / failure reading. Implement as `eval/cases/*.json` extensions + `eval/runners/run_robustness.py` that reuses run_baseline.py (no prod change). No invented ground truth — qualitative notes where labels absent (per eval/README).

## Ablation plan (proves agent adds value)

Pipeline A (single VLM/heuristic answers everything) vs B (full: validate→route→specialist→cross-check→fuse→qualify). Ablations: full vs w/o validation (hallucination count ↑ expected), vs w/o contradiction (false-confidence ↑), vs w/o fusion (uncorroborated ↑), vs optical-only / SAR-only (IoU lift). Measure on same 50-100 cases: accuracy, refusal, unsupported, confidence distribution, latency, evidence presence. PPT: single table, full wins on trust metrics even when accuracy ties — that is the story.

## What NOT to claim

No SOTA/OA without test-split eval; no ECE number before ≥200 samples; no time-saving % without timed analyst study; no meter-areas without GSD; no "first/calibrated" language.
