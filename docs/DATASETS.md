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

- For adaptation experiment: synthetic template dataset (200-500 samples) to verify LoRA scaffolding end-to-end (honest; no fabricated SOTA).
- For demo/evaluation: synthetic RS landscapes + simulated SAR (programmatically generated) — deterministic, offline, covers all 5 SIH scenarios.
- Real BigEarthNet/RSVQA/CDVQA manifests are scaffolded and ready to swap when bandwidth/GPU allow.

