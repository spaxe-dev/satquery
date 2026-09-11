# 2-Month Roadmap — Prototype → Final (2026-09-11)

Constraints: D: for caches, 24GB cloud only for training, heuristic floor never broken, no "calibrated/SOTA/first" until measured.

## Week 1 — Trust hardening (from 4-day base)

- Cloud/quality heuristic (bright+low-tex cloud fraction) → qualify + SAR-fallback suggestion text (no auto-reroute yet). Metric: cloud-qualify recall on 20 synth.
- ECC/MI warn → log shift scores for later calibration.
- Start confidence logging (SQLite/JSONL: query/task/tools/IoU/quality/overall/human-correct) for ECE track (need 200+ rows).
- Eval: robustness-100 green.

## Week 2 — Temporal honesty

- Change-confidence (separability + area + brightness/veg coherence) + real-vs-apparent hedge templates ("possible phenology/illumination — verify with dates").
- GSD/metadata pass-through (populate from TIFF tags where present; "pixel-only" disclaimer otherwise).
- Routing paraphrase expansion (60→120 queries, measure drop, regex-tune only).

## Week 3 — Minimal real adaptation (honest small win)

- Cloud 4090: BLIP LoRA r=8 EuroSAT-1500 (15-20min) → load via _try_load_vlm → guard retained. Eval: EuroSAT-val acc + VRSBench/RSVQA-50 sample. Expect classification >70%, VQA modest — report as "adaptation proof, not SOTA". Keep heuristic fallback.
- Alternative BigEarthNet-mini (9MB) instruction synthesis if EuroSAT→Cartosat critique bites (same pipeline).

## Week 4 — Spatial ROI

- ROI op: grounding box or water mask → buffer (dilate) → local change % + local fusion. Query pattern "around/near X". Demo: flood-near-settlement. Metric: local-vs-global agreement on 20.
- Why-card v2 (for/against bullets auto from masks/IoU/quality).

## Weeks 5-6 — Robustness + fusion

- Tiling (512 sliding window + mosaic) for large strips; radiometric percentile-stretch audit.
- Adaptive SAR thresholds (CV-conditioned) + OOD flag (texture/brightness distance from EuroSAT stats) → confidence cut.
- Quality-weighted fusion experiment (SAR weight ↑ when cloud>threshold) vs naive mean; report lift on cloud-synth-20.
- Grounding head spike (Grounding-DINO-tiny) ONLY if Week-3 eval shows grounding blocks benchmark score; else defer.

## Weeks 7-8 — Benchmark, ablation, demo lock

- Full robustness-150 + SECOND-20 + VRSBench-50 + CDVQA-access-check (fallback QAG-360K-sample if gated).
- Ablation table (full vs w/o validation/contradiction/fusion, vs single-VLM, vs opt-only/SAR-only).
- Offline proof (no-net run video), latency re-measure, analyst timing study (n≥5×3), report PDF auto-wire (only now).
- Demo lock: 10 jury demos rehearsed (normal, refusal×3, contradiction, cloud→SAR, ROI, trace-why, baseline-vs-SatQuery, OOD-qualify). Freeze thresholds.

## Dependencies / bottlenecks

- GPU: 1×4090 rental 2-4h total (train + eval); storage ~200GB D: for subsets; net only for initial weights.
- CDVQA access: verify Week 1; fallback QAG-360K-sample + SECOND.
- Biggest risk: RISAT/Cartosat shift untestable without hidden data — mitigation is honest OOD-flag + refusal/qualify demo, not fake robustness.

## What NOT to build (explicit)

Multi-agent/RAG-knowledge (RS-Agent owns, SIH-irrelevant), foundation-scale training, Postgres/MinIO/queue as scored features, meter-areas without GSD, auto-calibration claims.
