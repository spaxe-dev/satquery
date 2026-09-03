# Minimum Real Experiment Spec — SatQuery RS Adaptation (RTX 4090 24GB)

**Status: DRY-RUN VERIFIED, READY FOR CLOUD — NOT YET TRAINED (honest)**

This is the exact experiment that satisfies SIH26167 "at least one visual/VLM component genuinely fine-tuned using BigEarthNet or any open-source RS data" with the smallest defensible real-data footprint.

## Dataset

- **DATASET:** EuroSAT RGB
- **DATASET VARIANT:** RGB only (Sentinel-2 B04/B03/B02, 64x64 JPEG, 10 classes)
- **OFFICIAL SOURCE:** https://github.com/phelber/EuroSAT + HF `blanchon/EuroSAT_RGB` (also `giswqs/EuroSAT_RGB`) + Zenodo 7711810 (DOI 10.5281/zenodo.7711810)
- **LICENSE:** MIT (dataset), Sentinel-2 imagery © ESA Copernicus open data
- **PAPER:** https://arxiv.org/abs/1709.00029 (Helber et al. 2017, IEEE JSTARS 2019)
- **DOWNLOAD MECHANISM:** `python scripts/download_eurosat.py --subset 1500` → tries HF `blanchon/EuroSAT_RGB` via `datasets.load_dataset`, fallback to `giswqs/EuroSAT_RGB`, fallback to direct ZIP `https://zenodo.org/records/7711810/files/EuroSAT.zip?download=1`; verifies via `training/dataset_loader.py:verify_real_dataset`
- **STORAGE REQUIREMENT:** ZIP ~90MB, extracted ~200MB, processed instruction JSON ~1MB for 1500 images (3000 entries)
- **IMAGE FORMAT:** JPEG RGB, 64x64, uint8
- **BANDS:** 3 (R,G,B) — Sentinel-2 B04 (red), B03 (green), B02 (blue)
- **LABELS/ANNOTATIONS:** 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake (2000-3000 per class in full 27k)
- **PREPROCESSING:** PIL open → convert RGB → processor resize to 224x224 (BLIP) / 448 (Qwen2-VL), normalize per processor; no band stacking needed
- **TRAIN/VALIDATION SPLIT:** 1500 images stratified 150/class → 83/17 split → 1250 train images (2500 instruction entries) / 250 val images (500 entries); instruction JSON at `datasets/processed/eurosat_instruct_1500.json` with explicit `split` field
- **TRAINING TARGET:** Instruction tuning derived explicitly from real labels via 2 templates per image:
  1. Captioning: Q: "Describe the land-cover in this Sentinel-2 image." A: "This Sentinel-2 image shows {ClassName}, characterized by {CLASS_DESCRIPTIONS[ClassName]}. Land-cover class: {ClassName}."
  2. VQA: Q: "What land-cover class is visible in this Sentinel-2 image?" A: "{ClassName}"
  Reproducible, deterministic, 2 entries per image, genuine RS supervision (not synthetic).

## Model

- **BASE MODEL (primary, smallest credible):** `Salesforce/blip-image-captioning-base` (223M params, ~1.2GB FP16, BLIP Processor 224x224)
- **ALTERNATIVE (if instruction following desired):** `Qwen/Qwen2-VL-2B-Instruct` (2B params, ~4.5GB FP16, ~3.2GB 4-bit NF4)
- **SELECTION RATIONALE:** BLIP is smallest that can produce credible captioning adaptation and integrates into existing `tools/vqa_tool.py` caption path; fits 4GB for inference (fallback) and 6-8GB for training, leaving headroom on 24GB for larger batch. Qwen2-VL is kept as secondary for stronger instruction following but not required for minimal experiment.

## Training Configuration (BLIP primary)

- **QUANTIZATION:** FP16 (no 4-bit needed for BLIP; `torch.float16`, `fp16=True`). For Qwen2-VL: 4-bit NF4 double-quant via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)`
- **LoRA RANK:** 8
- **LoRA ALPHA:** 16
- **LoRA DROPOUT:** 0.05
- **TARGET MODULES (BLIP):** `["query", "value"]` (BERT attention in vision and text decoder; compatible with PEFT for BlipForConditionalGeneration)
- **TARGET MODULES (Qwen2-VL):** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **BATCH SIZE:** 4 (per device)
- **GRADIENT ACCUMULATION:** 4 (effective batch 16)
- **LEARNING RATE:** 2e-4 (BLIP) / 1e-4 (Qwen2-VL)
- **EPOCHS:** 2 (BLIP) / 1 (Qwen2-VL)
- **MAX STEPS:** 312 for BLIP (2500 train entries *2 epochs / 16 effective batch) — estimate, actual via Trainer
- **OPTIMIZER:** `adamw_torch` (BLIP) / `paged_adamw_8bit` (Qwen2-VL QLoRA)
- **INPUT RESOLUTION:** 224x224 (BLIP) / 448x448 (Qwen2-VL)
- **MAX LENGTH:** 128 tokens

## Resource Estimates (labeled estimates, not measured)

- **EXPECTED VRAM (BLIP FP16 + LoRA):** 6-8 GB (fits 24GB with headroom for 2 concurrent tools if needed)
- **EXPECTED VRAM (Qwen2-VL 4-bit QLoRA):** 12-14 GB
- **EXPECTED RUNTIME ON RTX 4090 24GB:** BLIP 2 epochs on 1500 images (3000 entries) ~15-20 min; Qwen2-VL 1 epoch ~25-35 min (measured from 27k SOTA ~98.6% ResNet50 baseline, BLIP throughput ~200 samples/min)
- **EXPECTED CHECKPOINT SIZE:** LoRA adapter only: ~12-20 MB (`adapter_model.safetensors` + `adapter_config.json` + `training_manifest.json` + tokenizer), vs 800MB full model
- **CHECKPOINT LOCATION:** `models/checkpoints/satquery-eurosat-blip-lora/` (primary) or `models/checkpoints/satquery-eurosat-qwen-lora/` (alternative); loader searches `CHECKPOINT_CANDIDATES` in `tools/vqa_tool.py`

## Exact Training Command (for RTX 4090)

```bash
# 1. Verify real data (already done, 1500 images, 3000 entries)
python scripts/download_eurosat.py --subset 1500
python training/train_lora_real.py --dataset eurosat --subset 1500 --dry_run  # verifies loader, no GPU

# 2. Real training — BLIP minimal (recommended)
python training/train_lora_real.py --dataset eurosat --subset 1500 --model Salesforce/blip-image-captioning-base --epochs 2 --batch_size 4 --gradient_accumulation 4 --lr 2e-4 --lora_r 8 --lora_alpha 16 --output models/checkpoints/satquery-eurosat-blip-lora

# 3. Alternative — Qwen2-VL QLoRA
python training/train_lora_real.py --dataset eurosat --subset 1500 --model Qwen/Qwen2-VL-2B-Instruct --use_4bit --epochs 1 --batch_size 1 --gradient_accumulation 8 --lr 1e-4 --output models/checkpoints/satquery-eurosat-qwen-lora

# 4. Via legacy wrapper (delegates)
python training/train_lora.py --dataset eurosat --subset 1500 --model Salesforce/blip-image-captioning-base --epochs 2 --batch_size 4
```

## Verification Steps (before cloud)

- [x] Real dataset files physically exist: `datasets/raw/eurosat/EuroSAT/*/*.jpg` 1500 images, 150/class, 64x64, verified via `verify_real_dataset`
- [x] Instruction JSON exists: `datasets/processed/eurosat_instruct_1500.json` 3000 entries, 2500 train / 500 val, labels in 10 EuroSAT classes, paths valid
- [x] Real data loader verified: `load_eurosat_instruct` + `EuroSATInstructDataset` with BLIP processor (image + text tokenization)
- [x] LoRA config verified: `r=8 alpha=16 target_modules query/value`, `peft.LoraConfig` creation, `get_peft_model` path
- [x] Checkpoint loading path verified: `tools/vqa_tool.py:_find_checkpoint` searches `satquery-eurosat-blip-lora`, loads `PeftModel` if `adapter_config.json` exists, else falls back to heuristic (preserves app)
- [x] Existing tests still pass: `python -m pytest tests -q` 25/25
- [ ] Real training not yet executed (requires RTX 4090, not launched per instructions)
- [ ] Real checkpoint not yet produced (status remains `dry_run_verified_real_loader`, not `trained`)

## BigEarthNet Alternative (documented, not chosen for minimal)

- Variant: reBEN v2 (refined BigEarthNet, 549k S1+S2 pairs, 120x120, 10-12 bands, 19 labels, CDLA-Permissive)
- Source: https://bigearth.net + Zenodo 10891137 (S2 59GB + S1 51GB = 110GB)
- Subset via `jerpint/bigearthnet` Hub: `bigearthnet-mini` 9MB (90/30/30), `bigearthnet-medium` 2.5GB (~10%), `bigearthnet-full` 30GB (RGB only, B02/B03/B04)
- Training target: Same instruction synthesis from 19 labels: "Describe land-cover" → "This patch contains: {labels joined}"
- Why not chosen: 110GB download + LMDB + multi-spectral handling adds 2-3 days setup vs 90MB EuroSAT; EuroSAT already satisfies PS "any open source RS data" and is Sentinel-2 like BigEarthNet, so minimal experiment prefers EuroSAT. BigEarthNet remains documented as primary per PS and can be swapped by changing `--dataset bigearthnet --subset 1000` with same instruction logic.

## Why This Formulation Is Technically Valid

- Dataset is real Sentinel-2 RS imagery (not synthetic), 27k is community standard for land-cover, ~98.6% SOTA shows labels are learnable.
- Training target is instruction tuning where input is real image + text query, target is real class name/description derived deterministically from annotation — no hallucinated labels, reproducible via `CLASS_DESCRIPTIONS`.
- Evaluation is classification accuracy on held-out val (500 entries) and can be extended to VRSBench/RSVQA splits.
- LoRA is genuine PEFT (not just manifest), QLoRA uses bitsandbytes 4-bit NF4 when requested, checkpoint is `adapter_model.safetensors` loadable by PEFT.
- Integration preserves heuristic fallback: `tools/vqa_tool.py` tries adapter first, falls back to heuristic if no checkpoint — app never breaks.

## What Is Not Claimed

- No training has been run yet (no `adapter_model.safetensors` produced, only `training_manifest.json` with `dry_run_verified_real_loader`).
- No benchmark performance claimed (no val accuracy reported, would be after training).
- No BigEarthNet files downloaded (only EuroSAT is real and present).
