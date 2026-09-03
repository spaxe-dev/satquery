# Datasets — SatQuery AI

## Training / Adaptation

| Dataset | Variant | Size | License | Link | Role | Subset for prototype |
|---------|---------|------|---------|------|------|----------------------|
| BigEarthNet / reBEN v2 | 590k S1+S2 pairs, 120×120, 10m GSD | ~90GB full | CDLA-Permissive 1.0 | https://bigearth.net | Mandated RS adaptation | `python scripts/download_datasets.py --dataset bigearthnet --subset 1000` (manifest only) or `python scripts/prepare_datasets.py --dataset synthetic --subset 500` for immediate use |
| BigEarthNet-MM | paired S1+S2 + 19 labels | same | same | — | fusion training (labels→template QA) | same |

## Evaluation

| Dataset | Size | Link | Role | Status |
|---------|------|------|------|--------|
| VRSBench | 29.6k images, 52k refs, 123k QA | https://huggingface.co/datasets/xiang709/VRSBench | caption/ground/VQA | HF, free; placeholder manifest in scaffold |
| RSVQA | LR/HRBEN | https://rsvqa.sylvainlobry.com | VQA augment | placeholder |
| CDVQA | 2,968 pairs, 122k QA (SECOND-based) | https://github.com/YZHJessica/CDVQA | change VQA | exist but download unverified — critical path; fallbacks QAG-360K/ChangeChat-105k |
| SEN1-2 | 282k pairs | mediatum | cross-modal pretrain | optional |

## Demo

Generated locally, no download:

```
python scripts/prepare_demo_data.py
→ data/demo/
   single_image/{scene_1..3.png/.tif}
   temporal_pairs/{flood_T1/T2, urban_T1/T2, forest_T1/T2}.png
   optical_sar_pairs/{pair1_optical/sar, pair2_..., flood_sar}.png
   invalid_inputs/{small_128.png, large_512.png, corrupt.png, not_an_image.txt}
   manifest.json
```

Also: `datasets/processed/synthetic_demo.json` (template QA for LoRA dry-run).

## Scripts

- `scripts/download_datasets.py --list` — lists all datasets with URLs/licenses
- `scripts/download_datasets.py --dataset <name> --subset N --out datasets/raw`
- `scripts/prepare_datasets.py --dataset synthetic --subset 500` — prepares training JSON
- `scripts/verify_datasets.py [path ...]` — checks manifests
- `training/train_lora.py --dry_run --subset 200` — verifies pipeline without GPU

## Directory Structure

```
datasets/
  raw/           # downloaded archives / manifests (gitignored for large)
  processed/     # subset JSONs for training (e.g., synthetic_demo.json)
  metadata/      # CRS/geotransform notes
  manifests/     # checksums
```

## Licenses & Notes

- BigEarthNet: CDLA-Permissive 1.0 — commercial/research allowed, attribution required.
- VRSBench: CC-BY.
- CDVQA/RSVQA: research use; request-gated for CDVQA — verify early (fatal risk if unavailable).
- Bhuvan/Bhoonidhi (for Indian demo scenes): registration required, demo-only, no redistribution.
- Copernicus Data Space (Sentinel): free, scriptable (OData) — best for live flood demo.

## What We Actually Used for This Prototype

- **For adaptation experiment (REAL, verified 2026-09-04)**: EuroSAT RGB — 1500 images (150/class stratified, 64x64 RGB Sentinel-2, JPEG, MIT, source HF `blanchon/EuroSAT_RGB` / Zenodo 7711810, ~90MB ZIP) at `datasets/raw/eurosat/EuroSAT/` + instruction JSON `datasets/processed/eurosat_instruct_1500.json` (3000 entries: 2500 train / 500 val, 2 templates per image: "Describe the land-cover" and "What class is visible?" derived explicitly from real EuroSAT labels). Verified via `training/dataset_loader.py:verify_real_dataset` (all 64x64 images open, labels in 10 classes) and `training/train_lora_real.py --dry_run` (loader OK, LoRA r=8, FP16, manifest `models/checkpoints/satquery-eurosat-blip-lora/training_manifest.json` with status `dry_run_verified_real_loader`). Training not yet executed (requires RTX 4090).
- **Previous synthetic**: `datasets/processed/synthetic_demo.json` (100 entries, 4 templates) remains for fallback and pipeline testing — clearly marked synthetic.
- **For demo/evaluation**: synthetic RS landscapes + simulated SAR (`data/demo/`) — deterministic, offline, covers all 5 SIH scenarios.
- Real BigEarthNet (reBEN v2, 590k, 110GB, CDLA-Permissive, https://bigearth.net) documented as primary per PS; EuroSAT chosen as minimal defensible subset due to 90MB vs 110GB and direct RGB usability. BigEarthNet-mini/medium via `jerpint/bigearthnet` (9MB/2.5GB) remains alternative. VRSBench/RSVQA/CDVQA/SEN1-2 remain scaffolded (not downloaded) — see `scripts/download_datasets.py --list`.

