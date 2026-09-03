# Real RS Adaptation Plan — SatQuery AI
# Chosen after filesystem audit 2026-09-04

## Audit Summary
- BigEarthNet/reBEN: [NOT PRESENT] — no files in datasets/raw, only placeholder manifest logic in download_datasets.py. Code never consumed real BigEarthNet patches. License CDLA-Permissive, source https://bigearth.net / Zenodo 10891137, ~110GB full (S2 59GB + S1 51GB), 120x120 10-12 bands, 19 labels. Not practical for local audit, but subset feasible on cloud.
- VRSBench: [NOT PRESENT]
- RSVQA: [NOT PRESENT]
- CDVQA: [NOT PRESENT]
- SEN1-2: [NOT PRESENT]
- Synthetic: [SYNTHETIC] — datasets/processed/synthetic_demo.json (100 entries, 4 templates), data/demo synthetic landscapes. ONLY dataset actually consumed by training/train_lora.py.
- Current training pipeline: [PLACEHOLDER] — writes manifest, no Trainer, no PEFT save, no bitsandbytes, checkpoint not loadable.

## Chosen Formulation
Primary: **EuroSAT RGB** (Sentinel-2, 10 classes, 27k images, 64x64 RGB JPEG, MIT license, Sentinel-2 source).
Why: Satisfies "any open source RS data" per SIH, 90MB vs 110GB, RGB directly usable by VLMs, classification → instruction tuning is valid captioning/VQA, fastest defensible on 24GB, HF `blanchon/EuroSAT_RGB` or Zenodo 7711810, ~98.6% SOTA with ResNet50 shows dataset is well-formed.
Fallback/Documented: BigEarthNet-mini/medium via `jerpint/bigearthnet` (9MB mini, 2.5GB medium) for larger experiment; same instruction synthesis from 19 labels.

## Training Target
Instruction tuning derived from REAL labels:
- Template 1 (captioning): Q: "Describe the land-cover in this Sentinel-2 image." A: "This Sentinel-2 image shows {ClassName}, characterized by {description}."
- Template 2 (VQA): Q: "What land-cover class is visible?" A: "{ClassName}"
- Template 3 (yes/no): Q: "Is this an image of {ClassName}?" A: "Yes" or "No" (negative sampled)
Each real image yields 1-2 instruction pairs; target genuinely derived from RS annotation, reproducible, explicit.

## Model Selection
Primary: `Salesforce/blip-image-captioning-base` (223M, 1.2GB FP16, ~6-8GB VRAM with LoRA) — smallest credible, captioning head aligns with Template 1, fits 4GB for inference (fallback) and 24GB for training.
Alternative: `Qwen/Qwen2-VL-2B-Instruct` (2B, 4-bit QLoRA ~12-14GB) — for instruction following, kept as secondary option, not required for minimal experiment.

## Minimum Real Experiment Spec
See docs section 6 in audit task.
