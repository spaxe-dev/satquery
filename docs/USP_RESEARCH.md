# USP Research — Strategic Report (2026-09-11)

## Executive takeaway

SatQuery today is a **working deterministic demo with honest scaffolds, not a trained system**. Nothing is fully SIH-compliant (1 GREEN, 10 YELLOW per SIH_REQUIREMENTS_MATRIX). Its only defensible edge is **system behavior, not model quality**: it refuses bad inputs, shows its work, flags disagreement, and runs offline — things GeoChat/TEOChat/EarthMind/VisTA/RS-Agent do not do together. Doubling down on **defensive evidence-aware orchestration** (validate → quality-gate → contradict → qualify/abstain → trace) is the highest-ROI 4-day and 2-month bet. Chasing VQA/SOTA numbers head-on loses (competitors trained on 300k-1.7M samples with 8×A100; SatQuery has 3k EuroSAT untrained).

## What exists (brutal)

- Works: 5 demos, 25+ tests, FastAPI+UI, registry, router, 5 CPU tools, JET/mask/overlays, heuristic confidence (labeled), 9-12-step trace, reports, refusal on dim-mismatch, VLM guard + lifecycle flags, EuroSAT 1500 present + dry-run manifest.
- Heuristic: all VQA/caption/grounding/change/optical/SAR/fusion/confidence. Grounding IoU unmeasured, modest. Optical RGB-proxy (no NIR), SAR luminance-proxy (synthetic-tuned). Fusion IoU only.
- Scaffold: LoRA scripts + manifests, no adapter_model.safetensors loaded; BigEarthNet/VRSBench/RSVQA/CDVQA/SEN1-2 placeholders; PDF manual; no tiling/reproj/calibration/cloud/OOD/ROI/learned-router.
- Will fail hidden: unregistered Cartosat/RISAT pairs (false change), real SAR statistics shift, sub-meter texture shift, cloud-as-change, tiny EuroSAT vs large strips, GSD-less % misread as meters.

## Requirements vs reality

See SIH_REQUIREMENTS_MATRIX + problem_statement.md. Every mandatory scope has a demo but no benchmark score and no trained weight — judge checking weights/metrics marks HIGH risk on RS-adaptation, VQA, change, cross-modal. Trace/GUI are lowest risk. The anti-generic-VLM clause means heuristic-only without at least one real adaptation run fails compliance even if demos pass.

## SOTA summary (numbers that matter)

- GeoChat (CVPR24): single-image multitask, 318k, region dialogue + grounding. No temporal/SAR/agent/validation.
- TEOChat (ICLR25): temporal QA xBD 89.9%/S2Looking 73.4%, zero-shot CDVQA 47.2% vs specialist 63.4%. No SAR/agent.
- EarthMind (2025): RGB-SAR fusion SOTA, 4B beats GPT-4o, VRSBench grounding 55.6%, DIOR CIDEr 428. No temporal/agent/validation. Cost 8×A100.
- VisTA (2024): CDVQA OA 73.1% T1, QAG OA 77.35%, oIoU 84.8%. Change-only, no SAR/agent.
- RS-Agent (2026): planning 95%, 18 tools, DualRAG. Needs API+net, stubs by default, no change-VQA/SAR-language fusion/validation-trace.
- Implication: beating any single number is infeasible in 2mo; combining their gaps (validation+contradiction+trace+offline) is feasible in 4d.

## Recommended USP stack (layered, honest)

**Positioning (to test, not to assert as novel science):**
"SatQuery is not a VLM that answers. It is a deterministic evidence-aware agent that decides how a remote-sensing question should be analyzed, whether the evidence suffices, how modalities agree, and when to qualify or refuse — with every step auditable offline."

- System USP (defensible): defensive validation + contradiction flag + quality qualification + trace + offline determinism together. No competitor does all five.
- NOT scientific novelty: each piece exists somewhere (refusal in software, IoU in fusion lit., trace in agents). Novelty is the evaluated combination under SIH scopes. Say so explicitly to jury.
- Proof burden: refusal rate, contradiction P/R, routing accuracy, false-change suppression, ECE-later — all measurable without inventing numbers.

## MUST / SHOULD / IF-TIME / DO-NOT (summary)

- MUST (4d): extend refusal (identical/nodata/unsupported), contradiction flag + cut, lifecycle+evidence-strength copy fix, why-answer card, % -not-meters policy, trace polish, routing-accuracy + refusal-rate numbers on ≥50 cases.
- SHOULD (2-4wk): cloud heuristic + SAR-fallback routing, ECC/MI registration check, real-vs-apparent hedge, ROI buffer analysis, EuroSAT LoRA minimal train + VRSBench/RSVQA sample eval (honest small numbers).
- IF TIME (2mo): quality-weighted fusion, calibrated confidence track (log now, calibrate later), tiling + adaptive SAR thresholds + OOD flag, grounding head only if eval blocks score.
- DO NOT: multi-agent/RAG-knowledge, frontier foundation training, auto-PDF/Postgres/queue as PPT claims, meter-areas without GSD, "calibrated/SOTA/first" language before experiments.

## Numbers to chase (5-10, no fabrication)

1. Invalid-input detection / correct refusal rate (target ≥95% on 50-case suite).
2. Contradiction detection P/R (target P≥80% on synthetic disagree/agree splits).
3. Agent routing accuracy (target ≥90% on query→task suite).
4. False-change suppression (ECC-gated vs naive on shifted pairs).
5. Change-detection sanity (retain 18-25% flood demo + SECOND-sample IoU, report don't inflate).
6. Grounding IoU sample (VRSBench 50, report honest ~0.2-0.35 heuristic baseline).
7. EuroSAT linear/VQA accuracy after LoRA (expect >70% classification; report val, not SOTA).
8. ECE/Brier/selective-risk (log now, report as "being evaluated", never as calibrated).
9. Latency (retain <500ms CPU hero) + offline proof (no-net run).
10. Workflow compression (manual 8-12 steps → 1 call; time with stopwatch on 3 analysts, n≥5 each, report range not single %).

## Biggest weakness that loses screening/final

Untrained heuristic core + untested domain shift presented as if ready for Cartosat/RISAT. Detection: run resized-EuroSAT + synthetic-SAR-shift + shifted-pair stress today; expect collapse. Mitigation: minimal real LoRA + registration check + adaptive thresholds + honest "provable soon vs planned" PPT split (Phase 24). Demo the mitigation (refuse/qualify on shifted data) rather than hiding it — jury rewards honesty over fake 0.82 bars.
