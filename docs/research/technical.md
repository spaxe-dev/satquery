# TECHNICAL CRITIC — SIH26167 SatQuery AI

## technical_claims

| # | Claim (from Stage 1) | Classification | Basis |
|---|---|---|---|
| 1 | LoRA fine-tuning of open 2–7B VLMs on named datasets is feasible for a student team | **VALID** | TEOChat (ICLR 2025) is a Video-LLaVA adaptation; recent CDVQA LoRA letters show 2B-class models reaching ~74% OA. Compute (1×24GB GPU) is sufficient for LoRA, not full FT of 7B. |
| 2 | GeoChat gives captioning/VQA/grounding out of the box | **VALID** | Open weights, mature repo. But note: GeoChat grounding is weak on precise localization (TEOChat paper shows language-token bboxes hurt IoU — TEOChat scored 28.8% IoU vs specialist 66% on xBD localization). Expect modest grounding scores. |
| 3 | TEOChat covers the change-analysis mandate | **PLAUSIBLE** | Strong zero-shot on change QA, beats GPT-4o/Gemini on temporal tasks; but its S2Looking F1 is only 34.5% and it is optical-only. Needs CDVQA-format fine-tuning; not a drop-in. |
| 4 | Optical–SAR fusion tool trained on BigEarthNet-MM will satisfy the cross-modal mandate | **UNCERTAIN** | BigEarthNet-MM gives multi-label *classification* supervision only — no text QA annotations. Any language output must be template-synthesized from labels (weak supervision). Also: S1+S2 fusion improves AP_m by only ~0.5–1.0 pt on reBEN (71.36% vs 68–70% S2-only) — the "complementary information" story is real but marginal for classification; it matters mainly under cloud. |
| 5 | Deterministic-first agent router | **VALID** | Correct engineering call; only the observable execution trace is evaluated, so reproducibility is a scoring asset. |
| 6 | Single 24GB GPU, 4-bit quantized 2–7B VLMs, offline inference | **VALID** | Standard; but multi-model serving (VLM + change model + fusion model + GDAL services) on one GPU requires careful memory budgeting and model swapping — latency per query may reach tens of seconds. |
| 7 | Sentinel-resolution training will generalize to Cartosat-2S (~0.6–1.6m) + RISAT (C-band, varied modes) hidden set | **UNREALISTIC as stated** | See risks. This is the project's single largest failure mode. |
| 8 | CDVQA freely downloadable | **UNCERTAIN** | Dataset exists (2,968 pairs, 122k QA, built on SECOND); access historically via authors/request. Verify in week 1. Fallbacks: QAG-360K, ChangeChat-105k (both with masks). |
| 9 | "BigEarthNet.txt" as adaptation dataset | **VALID with caveat** | reBEN v2: 590,326 S1/S2 pairs, 10m GSD, 120×120, free at bigearth.net. Note the PS filename likely refers to the label/metadata file — teams must confirm what "using BigEarthNet.txt" precisely requires for compliance. |

## required_concepts

- Vision-language fine-tuning: LoRA/QLoRA, instruction tuning, visual token handling for multi-image inputs (Video-LLaVA style interleaving for bi-temporal pairs).
- SAR physics essentials: speckle (multiplicative noise), C-band backscatter, VV/VH polarization, geometric distortions (layover/foreshortening), despeckling (Lee/frost or learned). Any SAR-language component must be speckle-robust.
- Cross-modal alignment: contrastive distillation of SAR features into RGB-pretrained encoder space (MM-OVSeg's CMU: 25k pairs, InfoNCE, lifts mIoU 55→73%) — the proven recipe for the optical–SAR tool.
- Change detection: Siamese encoders, semantic change detection (SECOND classes), pseudo-change suppression (seasonal/illumination artifacts) — the classic failure mode.
- Multi-label classification metrics (mAP, F1/F2) for the BigEarthNet adaptation; closed-vocab accuracy for CDVQA-style VQA.
- Co-registration validation: SSIM/correlation on downsampled patches; geotransform/CRS checks via rasterio/GDAL.
- Agentic orchestration: typed tool registry, intent classification, dispatch tables, confidence propagation, auditable traces.
- Domain shift / domain generalization: optical→SAR adaptation is documented as extremely hard (optical-trained classifier on SAR: 25.6% accuracy without adaptation, ~45% with sophisticated DA — on the *same* Sentinel sensors).

## critical_questions

1. **What exactly does "adapted using BigEarthNet.txt" mean for compliance** — full fine-tune on the archive, or is a LoRA on a subset with documented provenance sufficient? Must be answered before architecture lock.
2. **What resolution/GSD and acquisition modes will the hidden SAC set use?** RISAT-1 spans 1–24m modes; Cartosat-2S ~0.6–1.6m. Without this, preprocessing pipelines are guesses.
3. **Is the hidden-set evaluation per-task or end-to-end?** Are reference answers closed-vocab (CDVQA-style) or free-text? Closed-vocab is scoreable; free-text needs an LLM judge — who runs it?
4. **What happens when the optical image is cloudy** (the entire reason SAR exists)? Does the system need to degrade gracefully to SAR-only answers, and is that scored?
5. **Can a model trained on 10m Sentinel pairs say anything reliable about sub-meter Cartosat scenes?** What is the sanctioned fallback (e.g., tiling + patch-level inference)?
6. **What confidence metric is expected** — calibrated probabilities, verbalized, or just reported tool scores? Is confidence itself scored?
7. **What latency/GPU budget applies during evaluation?** On-site demo vs pre-submitted results changes the whole serving design.
8. **Is internet access permitted at evaluation?** PS implies offline-capable; confirm no API-dependent component.
9. **CDVQA access** — confirmed obtainable? If not, is training on QAG-360K/ChangeChat-105k and evaluating on CDVQA test acceptable?
10. **For grounding, are axis-aligned bounding boxes sufficient**, or are masks expected? (VLM-token grounding is measurably imprecise; a specialist detector may be needed for competitive IoU.)

## assumptions

- The hidden Cartosat/RISAT pairs are pre-georeferenced and co-registered (stated in PS) — this removes registration work but not radiometric/mode diversity.
- Template-generated QA from BigEarthNet-MM labels is acceptable "adaptation" supervision — legitimate but weak; answers will be label-fluent, not scene-fluent.
- Public benchmark test splits are the same ones everyone uses — scores will be directly comparable across teams; being below published SOTA (e.g., VisTA ~73% OA on CDVQA) is visible.
- One GPU at demo; no training at demo time.
- The PS's "may employ multiple specialised models" permits the specialist-tool architecture; a single fine-tuned multi-task VLM is also compliant — a major simplification option.

## technical_risks

1. **CRITICAL — Hidden-set domain shift.** Training data is 10m European Sentinel; hidden set is sub-meter Indian Cartosat + RISAT C-band with undisclosed modes. 3MOS study shows even same-modality matching models fail across SAR sources; optical→SAR adaptation literature shows near-random transfer without adaptation. Mitigation: multi-resolution training (tile high-res to pseudo-10m), aggressive augmentation, mode-diverse SAR data (SEN1-2, 3MOS, SpaceNet6 at 0.5–3m), and honest framing that scores on the hidden set are unpredictable.
2. **HIGH — No public optical–SAR text data.** The cross-modal tool with language output rests entirely on synthesized QA. Judges may probe free-form queries ("is that a harbor?") the templates never covered.
3. **HIGH — Change-VQA benchmark is small and synthetic.** CDVQA: 2,968 pairs from SECOND (aerial, 6 classes). Models can overfit question templates; hidden-set change questions will differ.
4. **MEDIUM — Grounding precision.** VLM token-based boxes are weak (TEOChat 28.8% IoU precedent). If grounding is the chosen second single-image task, budget for a specialist detector head.
5. **MEDIUM — Multi-GPU-model serving on one card**; VRAM thrash and slow tool swapping during live demo.
6. **MEDIUM — SAR modality detection and validation heuristics** (single-band, statistics) will misfire on stretched/RGB-composited SAR GeoTIFFs; the validator needs metadata-first logic with visual fallback.
7. **LOW — Dataset licensing**: BigEarthNet CDLA-Permissive fine; Bhuvan/Bhoonidhi imagery demo-only.

## required_data

- reBEN/BigEarthNet-MM (590k S1+S2 pairs, ~90GB, subsettable) — mandated adaptation + fusion training.
- VRSBench (eval), RSVQA LR/HRBEN (eval + augmentation), CDVQA (eval; access to verify).
- Supplementary: SEN1-2 (282k pairs), SECOND (change masks), QAG-360K / ChangeChat-105k (change QA with masks — fallbacks and augmentation), SpaceNet6 / DFC2023 (high-res optical–SAR, the CMU-Data recipe), xBD (damage assessment).
- Demo imagery: Copernicus Data Space (free, scriptable) for real pre/post scenes; Bhoonidhi for Indian scenes (license-restricted, demo only).

## required_apis

- No external LLM APIs (offline constraint, ISRO context).
- GDAL/rasterio (local), Copernicus OData for demo downloads, HF Hub for weights (download-time only).
- Optional: Bhoonidhi/NRSC portal for Indian demo scenes.

## architecture_requirements

- FastAPI backend + model worker(s); typed tool registry with per-tool schema (inputs, modalities, params, outputs); deterministic intent classifier with LLM-planner fallback; input validator (format/CRS/band/resolution/coregistration/modality); evidence renderer (bboxes/masks overlaid); JSON execution trace (task, tools, params, versions, timings) + PDF report; Postgres + object store; single-GPU serving with quantized models and sequential tool execution.
- Key design decision to settle early: N specialist models vs ONE multi-task fine-tuned VLM with task tokens + two small specialist heads (change-map, fusion classifier). The latter is far easier to serve and fine-tune, and is PS-compliant.

## disagreements

- With Stage 1's optimism on the optical–SAR tool: I downgrade it from "prioritize early" to **UNCERTAIN**. The evidence (marginal fusion gains on reBEN; near-total absence of paired text supervision; brutal cross-sensor transfer results) says this tool will produce label-fluent, template-bound answers, not robust joint reasoning. It satisfies the mandate; it will not wow on free-form queries. Stage 1's own "hardest component" note is correct — its prioritization recommendation understates this.
- With Stage 1's compute claim: LoRA on a 2B model on 24GB is fine; LoRA on 7B with images + multi-image inputs at 4-bit is tight but feasible. No disagreement, just a precision note.

## evidence

- TEOChat (arXiv 2410.06234, ICLR 2025): temporal VLM feasible; localization IoU 28.8% vs specialist 66%; S2Looking F1 34.5% — VLM change/localization is real but imprecise.
- MM-OVSeg (CVPR 2026, arXiv 2603.17528): SAR→RGB alignment via contrastive distillation on 25k pairs lifts mIoU 55.0→73.1% — a concrete, reproducible recipe for the fusion tool.
- reBEN ablations (Clasen et al. 2024): S1+S2 fusion AP_m 71.36% vs S2-only 68–70% — fusion gains are real but small for classification.
- Optical→SAR distribution-shift study (DLR): unadapted transfer 25.6% accuracy (≈ random); best adaptation 44.9% — cross-sensor transfer is genuinely hard even within Sentinel.
- 3MOS (Visual Intelligence 2025): no optical–SAR matching method generalizes across SAR satellites/sources — directly relevant to RISAT generalization.
- CDVQA (Yuan et al., TGRS): dataset built from SECOND, template-generated QA — small and synthetic.
- ISRO U-Net study (arXiv 2411.07581): SAC's own Cartosat-2S/RISAT workflows are per-object specialist models with manual ground truth — hints at the hidden set's style (task-specific reference answers/masks).

## revised_position

No reversal from Stage 1's PURSUE verdict — the problem is well-specified, datasets are real, and every mandatory task has working prior art. But my calibrated assessment is harsher than Stage 1 on two points:

1. The optical–SAR cross-modal tool is a compliance checkbox achievable via BigEarthNet-MM multi-label classification + template QA + MM-OVSeg-style alignment — not a genuine language-reasoning capability. Plan and demo it accordingly (closed-vocab queries, evidence maps).
2. The hidden SAC evaluation set is the decisive, uncontrollable variable. Teams should design for domain shift from day one (multi-resolution training, mode-diverse SAR augmentation, tiling for high-res inputs) and treat public-benchmark scores as the only reliably optimizable metric.

**Preliminary verdict: PURSUE — conditional on (a) confirming CDVQA access and the meaning of "BigEarthNet.txt" compliance, (b) committing to a single multi-task VLM + 2 specialist heads rather than four separate fine-tunes, and (c) accepting that hidden-set performance is a gamble mitigated, not solved, by augmentation. A team without real fine-tuning experience should not attempt this PS.**