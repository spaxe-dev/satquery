# Model Card — SatQuery AI

## SatQuery-RS-VLM (VQA / Captioning / Grounding)

- **NAME:** SatQuery-RS-VLM
- **VERSION:** 0.3.0 (heuristic primary; LoRA checkpoint swappable)
- **SOURCE:** Heuristic RS vision + template (always available) → swappable to `Salesforce/blip-image-captioning-base` or `Qwen/Qwen2-VL-2B-Instruct` with LoRA
- **LICENSE:** Heuristic: MIT (own code). BLIP: BSD-3. Qwen2-VL: Qwen License (research/commercial with conditions) — verify before deployment.
- **TASK:** Single-image VQA, captioning, text-guided grounding
- **INPUT:** Single optical/multispectral or SAR image (GeoTIFF/TIFF/PNG/JPEG), natural-language query
- **OUTPUT:** Answer text, caption, grounding boxes `[{x,y,w,h,label,confidence}]`, confidence breakdown
- **SIZE:** Heuristic: 0 MB. BLIP-base: ~400M params (~1.2GB). Qwen2-VL-2B: 2B params (~3.2GB 4-bit, ~4.5GB fp16)
- **VRAM:** Heuristic: CPU only. BLIP: ~2GB. Qwen2-VL-2B 4-bit: ~3.2GB + LoRA ~20MB.
- **TRAINING/ADAPTATION:** LoRA r=8 alpha=16 on `q_proj,v_proj`, lr 2e-4, batch 1, grad-accum 8, 1-2 epochs on synthetic RSVQA-style subset (200-2000 samples). Real adaptation target: BigEarthNet-MM + RSVQA. See `training/train_lora.py --dry_run`. Checkpoint: `models/checkpoints/satquery-lora/adapter_model.bin` (not committed by default).
- **DATASET:** Synthetic demo (templates) for pipeline verification; real: BigEarthNet (590k S1+S2 pairs, CDLA-Permissive), RSVQA, VRSBench.
- **KNOWN LIMITATIONS:** Heuristic answers are template+CV stats, not detector-based; grounding boxes are color/texture heuristics, IoU modest vs specialist detectors; Qwen2-VL 2B without fine-tuning is generic and not RS-specialized.
- **BENCHMARKS ACTUALLY RUN:** Synthetic VQA sanity (heuristic covers query types: water/built/vegetation/count/describe). No public benchmark scores claimed yet (honest). To run: `python training/train_lora.py --dataset synthetic --subset 500` then evaluate on VRSBench split.
- **NOTES:** `_try_load_vlm()` in `tools/vqa_tool.py` is the swap point. When checkpoint exists, it loads LoRA onto base and replaces heuristic path.

## ChangeDetector

- **NAME:** ChangeDetector
- **VERSION:** 0.3.0
- **SOURCE:** Own implementation (OpenCV + numpy), classical CV
- **LICENSE:** MIT
- **TASK:** Bi-temporal change detection, change description, change-based VQA
- **INPUT:** Two co-registered images T1, T2 (512×512 typical), optional query
- **OUTPUT:** JET change map (base64), binary mask + red overlay (base64), changed %, area, description, brightness/veg deltas, confidence
- **SIZE:** 0 MB (no learned weights)
- **VRAM:** CPU.
- **TRAINING:** None (classical). Future: Siamese BIT/Changer head trained on SECOND/CDVQA with masks.
- **DATASET:** Tested on synthetic flood/urban/forest pairs (`data/demo/temporal_pairs/`). Real evaluation on CDVQA (SECOND-based, 2,968 pairs).
- **KNOWN LIMITATIONS:** Sensitive to illumination/seasonal pseudo-change; Otsu can fail on very subtle change; no semantic change classes (e.g., “forest→barren” labels) without learned head.
- **BENCHMARKS:** Synthetic flood pair: 22.37% detected (expected 18-25%) — plausible. No CDVQA OA claimed.
- **NOTES:** Produces auditable threshold (`otsu_thresh`) and `mean_diff` for trace.

## OpticalAnalyzer

- **NAME:** OpticalAnalyzer
- **VERSION:** 0.3.0
- **LICENSE:** MIT
- **TASK:** Optical land-cover evidence (water, vegetation, built-up texture)
- **INPUT:** Optical/multispectral RGB image
- **OUTPUT:** Water/veg/built masks (numpy), percentages, water/built overlays (base64), evidence list, answer
- **SIZE:** 0 MB
- **VRAM:** CPU
- **METHOD:** NDVI-proxy `(G-R)/(G+R)`, HSV, Canny edges; water requires dark+blue-dominant+low Veg; built via edge density minus water/veg.
- **LIMITATIONS:** RGB-only proxy (no NIR/SWIR); 10m Sentinel vs sub-meter Cartosat not differentiated; heavy cloud not modeled.
- **BENCHMARKS:** Pair1 optical: water 4.2% (expected <10%), veg 3.3% — plausible.

## SARAnalyzer

- **NAME:** SARAnalyzer
- **VERSION:** 0.3.0
- **LICENSE:** MIT
- **TASK:** SAR backscatter analysis, water/built-up separation
- **INPUT:** SAR image (single-band or RGB composite)
- **OUTPUT:** Water/built masks, mean/std backscatter, speckle CV, overlays, evidence, answer
- **SIZE:** 0 MB
- **VRAM:** CPU
- **METHOD:** Gray backscatter, local CV (7×7), percentile thresholds (30%/75%), median despeckle (5×5), morphology (3×3 open, 5×5 close).
- **LIMITATIONS:** Filename/band heuristic for modality; RGB composites penalized (-0.12 confidence); speckle still high in synthetic SAR (CV ~0.92) — thresholds tuned for synthetic; real RISAT modes not tested.
- **BENCHMARKS:** Pair1 SAR: water 2.7%, built 0.99%; Flood SAR: water 10.3% — consistent with inundation.

## FusionEngine

- **NAME:** FusionEngine
- **VERSION:** 0.3.0
- **LICENSE:** MIT
- **TASK:** Cross-modal evidence fusion, agreement scoring, corroboration
- **INPUT:** `optical_result` + `sar_result` + `H,W`
- **OUTPUT:** IoU/Dice (water/built), mean IoU, agreement label, fused high-conf/union percentages, fused overlays (base64), corroboration statement, confidence
- **SIZE:** 0 MB
- **VRAM:** CPU
- **METHOD:** Decision-level fusion: intersection = high-confidence, union = inclusive; IoU/Dice per class; tiered agreement thresholds (0.5/0.25/0.08).
- **LIMITATIONS:** Assumes co-registered inputs (per PS); no radiometric normalization; built IoU often low due to physics difference (expected).
- **BENCHMARKS:** Pair1 fusion: water IoU 0.608 (strong), built IoU 0.00 (physics difference) → moderate overall (0.30) — defendable.

## Shared Notes

- All confidences are **heuristic, labeled** `is_heuristic: true`.
- No quantized LLM API is used offline; all inference is local.
- Real LoRA checkpoint is **not required** for demo — heuristic floor guarantees every demo works; LoRA is the ceiling when compute/bandwidth allows.
- For ISRO evaluation on Cartosat-2S/RISAT: expect domain shift. See `docs/HANDOFF.md` risks and `technical.md`.

