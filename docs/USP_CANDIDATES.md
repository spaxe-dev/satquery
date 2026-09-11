# USP Candidates — Full Database + Rankings (2026-09-11)

Scoring 0-10: NOV (novelty vs prior art), DEF (defensibility), DEMO (jury demo in 20s), MEAS (measurability), SIH (relevance), FEAS (feasibility now), ADV (competitive advantage). Time: 4D / 1-2W / 1M / 2M / >2M. Risk: L/M/H.

## Candidate table (20)

| Rank | ID | USP | Problem | Prior art | NOV | DEF | DEMO | MEAS | SIH | FEAS | ADV | Time | Risk | Rec? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | U01 | Defensive validation + explicit refusal (dim/identical/nodata/unsupported) | Silent wrong answers on bad inputs | No competitor refuses; RS-Agent plans anyway; GeoChat/TEOChat/EarthMind/VisTA assume clean inputs | 6 | 9 | 10 | 10 | 10 | 10 | 9 | 4D (extend) | L | MUST |
| 2 | U02 | Auditable execution trace as first-class evidence (task/tools/params/timings/outputs) | "Why did you answer this?" for ISRO audit | RS-Agent has LangChain trace (needs API); none produce SIH-observable trace offline deterministically | 5 | 8 | 10 | 10 | 10 | 10 | 8 | 4D (exists, polish) | L | MUST |
| 3 | U03 | Cross-modal contradiction flag + confidence cut (IoU<0.12 → CONTRADICTION badge, −0.15 overall) | Averaging disagreeing sensors misleads | EarthMind fuses (no flag); fusion lit. reports IoU but never abstains/flags for user | 7 | 8 | 10 | 10 | 9 | 9 | 9 | 4D (1d) | L | MUST |
| 4 | U04 | Evidence-strength (honestly uncalibrated) + VLM lifecycle transparency (attempted→loaded→accepted/fallback reason) | Overconfident heuristics presented as probabilities; silent fallbacks | No competitor exposes fallback; all present scores as confidences | 6 | 8 | 9 | 9 | 9 | 10 | 8 | 4D (copy change) | L | MUST |
| 5 | U05 | Cloud/quality-aware qualification + SAR-fallback suggestion | Cloudy optical answered as if clear | RS-Agent has cloud_removal tool but no QA gating; others ignore quality | 7 | 7 | 9 | 9 | 9 | 7 | 8 | 1-2W (heuristic 2-3d + routing 3d) | M | SHOULD |
| 6 | U06 | Registration-aware change (ECC/MI check → APPROXIMATE/warn, no silent resize) | False change from misregistration (biggest hidden risk) | All (VisTA/TEOChat/SatQuery) assume registered; OptSAR lit. notes it but no agent gate | 6 | 8 | 9 | 9 | 10 | 7 | 9 | 1-2W (check 3-5d) | M | SHOULD |
| 7 | U07 | Real vs apparent change hedge (illumination/season + veg/brightness deltas + change-confidence) | Seasonal/shadow reported as urban/flood | VisTA grounds but never hedges; TEOChat QA-only | 6 | 7 | 8 | 8 | 10 | 7 | 8 | 1-2W | M | SHOULD |
| 8 | U08 | Deterministic offline registry-pinned agent (no LLM planner, reproducible, air-gappable) | LLM planners flaky, need API, unevaluable | RS-Agent needs OPENAI_API_KEY + net; others no agent | 5 | 8 | 8 | 9 | 10 | 10 | 8 | 4D (exists) | L | MUST |
| 9 | U09 | Real RS adaptation (EuroSAT LoRA → VRSBench/RSVQA eval, honest numbers) | "Generic VLM disqualified" clause | GeoChat 318k, TEOChat 554k, EarthMind 1M+30k — all far larger; SatQuery 3k EuroSAT tiny | 2 | 5 | 7 | 10 | 10 | 6 | 6 | 1M (train 1d + eval 1wk) | M | SHOULD (do minimal, honest) |
| 10 | U10 | ROI-conditioned analysis ("around lake/river", buffer → local change %) | Whole-image % misleads local question | TEOChat RTQA/TRE does region QA (71.7%); EarthMind SAP segments; SatQuery has masks but no ROI op | 4 | 6 | 9 | 9 | 8 | 6 | 7 | 1M | M | SHOULD |
| 11 | U11 | Quantitative geo-answers in % + pixel counts, never fake meters without GSD | Fake precision (m² without georef) | VisTA gives ratios (CR type); others give text only | 4 | 6 | 8 | 9 | 8 | 9 | 6 | 4D (wording) | L | MUST (policy) |
| 12 | U12 | "Why this answer" evidence card (for/against, quality, agreement, factors) | Trace exists but not decision-readable | None | 6 | 7 | 9 | 8 | 9 | 9 | 7 | 4D (1d expose scores) | L | MUST |
| 13 | U13 | Unsupported-query abstention (capability gate) | Hallucinating ownership/count-by-type | None abstains | 6 | 7 | 9 | 9 | 8 | 8 | 7 | 4D (4h) | L | MUST |
| 14 | U14 | Duplicate/identical-pair detection | Same image twice → fake change% | None | 5 | 6 | 9 | 10 | 7 | 10 | 6 | 4D (2h) | L | SHOULD |
| 15 | U15 | Quality-weighted fusion (SAR up-weighted when optical degraded) | Averaging good+bad sensor | EarthMind mutual attention (learned, 8×A100); SatQuery would be heuristic | 5 | 6 | 9 | 8 | 9 | 5 | 8 | 2M | M | IF TIME |
| 16 | U16 | Learned tool router (classifier over query+meta) | Regex brittle on paraphrase | RS-Agent Task-Aware Retrieval (95%); SatQuery regex only | 3 | 4 | 6 | 9 | 8 | 6 | 5 | 1M | M | IF TIME (measure regex first) |
| 17 | U17 | Calibrated confidence (ECE/Brier/selective-risk on held-out) | Heuristic 0.82 meaningless | None calibrated in RS VLMs (gap is real but hard) | 8 | 8 | 6 | 10 | 9 | 3 | 9 | 2M | H | IF TIME (start logging now, claim later) |
| 18 | U18 | Domain-shift resilience (tiling + radiometric augment + adaptive SAR thresholds + OOD flag) | Sentinel→Cartosat/RISAT collapse | EarthMind multi-sensor training closest; no explicit OOD flag | 6 | 8 | 7 | 9 | 10 | 4 | 9 | 2M | H | SHOULD (tiling+thresholds 2wk, rest 2M) |
| 19 | U19 | Pixel-level grounding upgrade (Grounding-DINO/OWL-ViT head) | Heuristic IoU modest | GeoChat/EarthMind already strong; catching up, not differentiating | 2 | 3 | 7 | 10 | 9 | 5 | 4 | 1M | M | IF TIME (only if eval shows gap blocks score) |
| 20 | U20 | Multi-agent / self-verification loops / RAG knowledge base | Complex queries need knowledge | RS-Agent already does DualRAG+Solution Space; duplicating it loses | 3 | 3 | 5 | 6 | 6 | 3 | 3 | >2M | H | DO NOT BUILD |

Weak candidates kept (U16, U19, U20) to prevent hype-driven roadmapping.

## Rankings

- A (4-day screening): U01, U02, U08, U03, U04, U12, U13, U11, U14.
- B (prototype 2-4wk): U05, U06, U07, U10 + EuroSAT-train-minimal (U09-min).
- C (final 2mo): U18, U15, U17-logging→calibration, U10-full.
- D (moat): U03+U05+U06+U18 combined = "evidence-aware defensive agent" (no single competitor replicates cheaply offline).
- E (demo): U03 contradiction demo, U01 refusal demo, U05 cloud→SAR demo, U02 trace demo, U06 registration demo.
- F (measurable): U01 refusal rate, U03 contradiction P/R, U06 false-change suppression, routing accuracy, ECE-later.
- G (vs "just GeoChat"): U03, U05, U06, U02, U08 (GeoChat has none of these; single-image VQA alone loses).
- H (vs "just LLM wrapper"): U08 offline determinism + U02 trace + U01 refusal + U04 lifecycle (wrappers have none).
- I (vs "just change detection"): U03+U05+U07+U10 (change map alone never contradicts, qualifies, ROIs, or corroborates with SAR).
- J (best combo): U01+U03+U05+U06+U02 = defensive evidence-aware loop. Whole > sum: each gate feeds confidence that feeds abstention that feeds trace.

## Anti-USP verdicts (downgrades applied)

- "Better VQA/captioning/UI/grounding/more models/faster" — downgraded (everyone has it, trivial to copy, not measurable vs GeoChat/EarthMind scale).
- "Learned fusion beats IoU" — downgraded for 4d (EarthMind already SOTA with 8×A100; heuristic flag is cheaper differentiator).
- "RAG knowledge" — rejected (RS-Agent owns it, needs API, SIH-irrelevant).
- "Calibrated confidence now" — rejected as claim (no data); kept as 2M measurement track.
- "First evidence-aware agent" — rejected as wording; honest wording: "only offline deterministic agent in SIH scope with refusal+contradiction+trace evaluated together."
