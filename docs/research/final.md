# SIH26167 — FINAL SYNTHESIS (SatQuery AI)

## What survived scrutiny, and what didn't

**Established facts** (verified or repo-confirmed):
- CDVQA exists: Yuan et al., TGRS 2022, github.com/YZHJessica/CDVQA — repo verified; **archive download never completed by the team**.
- OSVQA/TGFNet is real (arXiv 2411.15770): 6,008 optical–SAR pairs, 140 unique answers, ~10m GSD 256px chips — real but resolution-mismatched to SAC inputs.
- EarthMind ships pretrained + multi-sensor finetuned checkpoints (github.com/HuaDeity/EarthMind). Component-level "no one has our hard part" comfort is dead.
- reBEN v2 = 549k clean image–label pairs claimed; S1 patch-validity fraction unexamined.
- The PS mandates five things: RS adaptation, single-image VQA + one of captioning/grounding, bi-temporal change analysis, optical–SAR joint analysis, agentic orchestration.

**Struck from the record:** the "~74% OA CDVQA" figure (unsourced through three challenges); SSIM for cross-modal validation; all "EXACT MATCH competitor" labels; any numeric benchmark target before answer format is known; Feasibility's blanket LOW/MEDIUM rating (component (c) held at MEDIUM-HIGH).

---

### 1. Problem understanding
Not a VLM project — an orchestration-and-trust project. ISRO wants a natural-language front-end over multiple remote-sensing specialist models, with emphasis on paired imagery (bi-temporal change + optical–SAR), evaluated on hidden Cartosat-2S/RISAT data with undisclosed annotations. A generic LLM/VLM wrapper explicitly fails. The observable execution trace (selected task, tools, parameters, outputs) is what's scored — not chain-of-thought.

### 2. Core conceptual requirements
- RS sensor physics: optical vs SAR complementarity, speckle, backscatter mechanisms, GSD/resolution mismatch.
- Parameter-efficient fine-tuning (LoRA/QLoRA on a shared vision backbone) and domain shift/augmentation.
- Change detection & change-VQA; cross-modal co-registration validation.
- Agentic tool registries: deterministic routing, multi-intent query composition, auditable traces.
- Geospatial engineering: GeoTIFF/CRS/geotransforms, rasterio pipelines.
- Calibration/uncertainty estimation — saying "low confidence" gracefully.

### 3. Best solution
A single shared ~2B-class quantized vision-language backbone with task-specific LoRA adapters, behind a deterministic agent controller with an explicit plan trace, plus:
- Hybrid optical–SAR tool: physics-grounded template floor (backscatter statistics → land-cover statements, always works) + OSVQA-finetuned ceiling if download verifies.
- Metadata-first input validator (rasterio gate), flag-don't-reject warnings; MI registration as stretch goal only.
- Short factual answers trained for the intersection of all plausible scoring regimes.
- EarthMind evaluated as a registry component, not fought head-on.
- Differentiation via execution artifacts: cross-modal contradiction flagging, calibrated uncertainty, live adversarial-input demo.

### 4. Architecture
- Ingest: GeoTIFF/TIFF upload → compatibility check (CRS, geotransform overlap, GSD ratio, date ordering) → warning banners.
- Router: LLM intent parser emits explicit plan into trace; deterministic execution core; tested compound-query path (change tool + grounding tool chained).
- Models: one resident 2B-class backbone, quantized, LoRA-swapped per task (~8–12GB VRAM total), CPU-offload fallback; specialist heads for change detection where masks exist.
- Fusion: optical–SAR consistency check comparing independently derived indicators (e.g., NDWI-water vs low-backscatter smooth regions); disagreement flagged, not rejected.
- Output: evidence-grounded text + visual overlays + confidence + downloadable execution summary/report.
- Web GUI over this backend; end-to-end skeleton with stub models by week 2.

### 5. Critical dependencies
- GPU: owned ≥24GB or rental budget — **unresolved, gates everything**.
- Checksummed downloads: CDVQA archive, OSVQA, reBEN-S1 subset — none done.
- Answer-format clarification from ISRO via SIH channel.
- SAC input characteristics (pan vs Mx optical; RISAT MRS vs FRS) — every resolution decision provisional until known.

### 6. Existing landscape
GeoChat, TEOChat, EarthMind (weights released), TerraScope, RingMo-Agent (prior art/close match pending weight checks), TCSSM domain-generalization CDVQA precedent. Every rival can assemble the same open stack behind a Gradio app; hundreds of teams will submit near-identical architectures.

### 7. Genuine differentiation
No idea-level moat exists — conceded and settled. Real differentiators are execution-only:
1. Live adversarial-input handling (mismatched pairs, cloudy optical, compound queries).
2. Cross-modal consistency checks (does the SAR story contradict the optical story?).
3. Calibrated honesty — "I'm uncertain, here's why."
4. Genuinely auditable traces and honest per-domain test breakdowns.
5. Hybrid SAR floor guaranteeing the optical–SAR demo never fully dies.

### 8. Important questions

RESOLVED QUESTIONS
- Competitor threat level: downgraded to prior art/close match except EarthMind (weights verified).
- CDVQA status: available-pending-download-verification.
- Validator design: metadata-first, flag-don't-reject; SSIM dead; MI stretch goal.
- Compute plan: 2 LoRAs + lightweight fusion head on shared backbone, not four full fine-tunes.
- Novelty location: orchestration/evidence/trust, not components.
- Sequencing accepted: resolve answer format first, then commit numbers.

UNRESOLVED QUESTIONS
- Team GPU ownership/compute budget.
- SAC evaluation answer format (closed-vocab / free-text / LLM-judge).
- Live vs batch evaluation.
- reBEN v2 S1 valid-patch fraction (effective paired training size).
- ISRO metric normalization weights.
- TerraScope/RingMo-Agent checkpoint existence.

QUESTIONS THAT MUST BE ANSWERED BEFORE COMMITTING
- Does at least one byte of CDVQA and OSVQA actually download? (flips component-(c) risk)
- GPU confirmed?
- RISAT mode (MRS ~18m vs FRS) and Cartosat pan/Mx — sets the entire resolution-gap strategy.

### 9. Fatal risks
1. Week 1 produces diagrams instead of artifacts → indistinguishable from median Gradio team, cut at shortlisting. *(the actual killer)*
2. Datasets that don't download (CDVQA archive, OSVQA) → mandatory scope unbuildable as planned.
3. No GPU / late discovery of rental cost → schedule collapse.
4. Integration death: five half-working demos with visible seams (modal SIH failure mode).
5. Confident hallucination on SAC-like imagery in front of SAC judges.
6. Compound-query routing breaking live during the demo.
7. Hidden-set domain shift crushing benchmark scores (hits everyone; hedged, not fatal).

### 10. MVP
The strongest realistic SIH demonstration:
- Upload a real bi-temporal Sentinel pair → "What changed between these dates and where?" → change map overlay + grounded description + execution trace + confidence.
- Same scene, optical+SAR → built-up/water identification with a cross-modal agreement statement ("optical indicates X, SAR backscatter is consistent/inconsistent").
- Then, live: feed a deliberately mismatched pair → system flags it instead of answering garbage.
- Single-image VQA + grounding on VRSBench-style inputs throughout.
Every answer carries confidence and a downloadable audit report. This covers all five mandated capabilities and shows trust machinery rivals won't have.

### 11. Judge attack
1. "EarthMind already does multi-sensor EO reasoning with released weights — what did you build?"
2. "Your optical–SAR model was trained on 10m chips; how does it work on our sub-meter Cartosat/RISAT pairs?" 
3. "Where are your numbers?" (if format unresolved, targets were never committed)
4. "'Agentic' — isn't this just an if-else router over three models?"
5. "Show me it working on OUR imagery, right now." (domain shift, live)
6. "What happens when the two modalities disagree?"
7. "Why should we believe your benchmarks transfer?"

### 12. Defense
1. EarthMind → "We evaluated it as a candidate registry component; our contribution is orchestration, validation, and calibrated evidence above any single specialist — including theirs."
2. Resolution gap → "Known, quantified, hedged: GSD simulation, speckle injection, radiometric augmentation; honest per-domain results in our report; hybrid template floor guarantees sane output regardless."
3. Numbers → "Format-dependent targets deferred deliberately per your sequencing; here are our public-benchmark results and our domain-shift ablation."
4. Agentic → live compound-query composition ("What changed AND highlight the water") with the plan printed in the audit trace; passing test suite shown.
5. Domain shift → adversarial-input demo performed live on judge-supplied-style mismatched inputs.
6. Disagreement → tiered validator flags it with evidence; "flag-don't-reject" is a feature matching the PS's compatibility mandate.
7. Transfer → TCSSM-style domain-generalization recipe + augmentation + honest reporting.

### 13. Final verdict

**PURSUE — strictly gated on the week-1 evidence gate.**

Five rounds produced a top-quartile *plan* attached to a zero-artifact *track record*. Not STRONG PURSUE because nothing has been downloaded, trained, or measured, compute is unconfirmed, and no idea-level moat exists. Not MAYBE because every mandated component traces to a real, open resource and the failure modes are known and fixable. The conditional is non-negotiable: checksummed downloads of CDVQA/OSVQA/reBEN-S1, one end-to-end LoRA run on real downloaded data, GPU confirmation, and the ISRO evaluation-format inquiry sent — all before further architecture work. If those land, this team presents as a genuine winner contender on trust-and-execution grounds; if they don't, expect elimination at shortlisting against a team with one ugly but real end-to-end run.

### 14. Scores

| Criterion | Score | Justification |
|---|---|---|
| Technical depth | 8 | Shared-backbone LoRA design, tiered validation, physics-grounded SAR layer, residency planning — deep, though partly undemonstrated. |
| Novelty | 4 | Conceded: PS-prescribed recipe, same open stack available to all; innovation relocated to trust/calibration mechanisms. |
| Competition | 6 | Hundreds of identical submissions expected; EarthMind weights verified raises the bar; field strength high but beatable via execution. |
| Feasibility | 6 | Buildable from open resources; zero executed to date; compute unconfirmed; integration risk real. |
| Data availability | 5 | All datasets exist as repos/papers; nothing downloaded; reBEN-S1 fraction unknown; OSVQA resolution-mismatched. |
| Demo potential | 8 | Adversarial live demo, hybrid SAR floor, auditable traces — strong even under partial failure. |
| Impact | 8 | Directly serves ISRO/SAC operational need; non-expert NL access to satellite imagery; org fit excellent. |
| Scalability | 7 | Quantized 2B residents + CPU offload scale sensibly to lab deployment; beyond single-GPU unclear. |
| Differentiation | 5 | Execution-only moat; real but fragile — depends entirely on actually shipping the trust features. |
| Winning potential | 7 | Top-quartile plan; wins iff week-1 evidence lands and the live adversarial demo works. |

**Overall: 6.5/10 — PURSUE, conditional.**

Justification: the tension between problem alignment (9-equivalent), demo potential, and impact on one side versus novelty, data credibility, and differentiation on the other *is* the verdict. This is a well-understood, fully-mandated, honestly-scoped build whose upside is execution quality and whose downside is entirely self-inflicted (no downloads, no GPU, no format inquiry). The decisive fact across six rounds of deliberation is that five agents verified almost nothing by running anything. The recommendation is therefore not "build this" but "spend week 1 producing evidence — then build this."