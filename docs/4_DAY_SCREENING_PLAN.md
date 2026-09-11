# 4-Day Screening Plan — Exact Actions (2026-09-11)

Goal: PPT-credible refusal + contradiction + trace story with numbers on ≥50 cases. No prod destabilization, no giant downloads, D: for caches.

## Day 1 — Contradiction + honesty copy (highest demo ROI)

- [ ] `tools/fusion_tool.py`: if min(water_iou,built_iou)<0.12 and either pct>2 → `contradiction=True`, overall −=0.15 (floor 0.05), corroboration prefix "CONTRADICTION: ". Expose `contradiction` in return. (2-3h, test test_tools intact)
- [ ] `backend/orchestrator.py`: surface contradiction badge in trace + final_answer prefix; rename UI copy "confidence (heuristic)" → "evidence strength (uncalibrated, heuristic)". (2h)
- [ ] `tools/vqa_tool.py` wording: RGB-proxy disclaimer in optical answer ("RGB-proxy, not true NDVI/NIR"). (1h)
- [ ] Measure: 40 synth agree/disagree pairs → contradiction P/R. Screenshot: Pair1 agree vs shifted disagree. PPT claim: "Flags disagreement (P≈X%, n=40)" — X measured, not invented.
- Risk LOW (additive flags, no threshold change to masks).

## Day 2 — Refusal gates + why-card

- [ ] `validation.py` + `planner.validate`: identical-hash refuse-note ("identical inputs"), nodata-fraction>90% refuse, unsupported-keyword gate ("owner/caste/tank-type" → unsupported-clarify), ambiguous-temporal-with-1-image redirect message. Each <2h, each with test.
- [ ] `orchestrator.py`: "why this answer" factors dict (input_q, model_c, agreement, cross-modal, quality warnings, contradiction) → returned as `confidence_factors` + rendered card. (3h, additive)
- [ ] Measure: 50-case refusal suite → refusal rate, unsupported rate. Screenshot: Demo5 + identical-pair + unsupported. PPT: refused-vs-hallucinated bar.
- Risk LOW (new branches, existing Demo5 path untouched).

## Day 3 — Routing + registration-warning + numbers

- [ ] `planner.decide_task`: change-verbs dominate over fusion on 2-image opt+SAR + "what changed" (priority tweak) + 10 new tests. (4h)
- [ ] `change_tool` or validation: ECC shift score → if median shift >8px warn "possible misregistration — change APPROXIMATE" (no warp yet). (4h)
- [ ] Run routing-60 + registration-20 + latency-20. Screenshot: trace 9-12 steps + timings. PPT: routing acc, false-change-suppression illustration.
- Risk MED (planner tweak — gate behind tests).

## Day 4 — Benchmark + PPT + red-team

- [ ] `eval/cases/` add refusal/routing/contradiction/registration cases; `eval/runners/run_robustness.py` (reuse run_baseline, no prod import change). Output `eval/reports/robustness_<ts>.json`. (4h)
- [ ] PPT: 1 slide each — problem→before/after workflow (timed, n noted), architecture (existing diagram), 5 demos (screenshots), trust numbers (refusal, contradiction, routing, latency), honesty split WORKING NOW vs PROVABLE SOON vs PLANNED, ISRO relevance (flood/urban/water/infra vignettes, no fabricated workflow).
- [ ] Red-team dry-run: "Why not GeoChat/EarthMind/RS-Agent?" → answers from COMPETITOR_MATRIX. "Where is adaptation?" → EuroSAT dry-run manifest + honest pending-train line. "Calibrated?" → "No — evidence-strength, ECE protocol defined."
- Risk LOW. Do NOT claim: calibrated, SOTA, meters, first.

## Safe quick wins (if ahead)

Duplicate-hash, % -not-meters policy doc, VLM-lifecycle badge polish, cloud-bright heuristic warn-only (no routing change yet).

## Caches

HF/hub → D:\hf_cache (set HF_HOME, HUGGINGFACE_HUB_CACHE). No C: dumps. No dataset download beyond EuroSAT-1500 present.
