# Datasets — README

## Sources

- **BigEarthNet / reBEN v2**: https://bigearth.net — CDLA-Permissive 1.0 — 590k S1+S2 pairs, 10m GSD, 120×120. Primary for RS adaptation.
- **VRSBench**: https://huggingface.co/datasets/xiang709/VRSBench — NeurIPS 2024, caption/ground/VQA.
- **RSVQA**: https://rsvqa.sylvainlobry.com — LR/HRBEN variants over BigEarthNet + OSM.
- **CDVQA**: https://github.com/YZHJessica/CDVQA — 2,968 pairs, 122k QA, SECOND-based, template QA.
- **SEN1-2**: https://mediatum.ub.tum.de/1436631 — 282k SAR-optical pairs.
- **Copernicus Data Space**: https://dataspace.copernicus.eu — Sentinel-1/2 OData API for demo scenes.
- **Bhoonidhi / Bhuvan (ISRO)**: registration required — demo-only Indian scenes.

## Licenses

- BigEarthNet: CDLA-Permissive 1.0.
- VRSBench: CC-BY.
- CDVQA/RSVQA: research.
- Bhoonidhi: restricted, no redistribution.

## Download Procedure

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --dataset bigearthnet --subset 1000 --out datasets/raw
python scripts/prepare_datasets.py --dataset synthetic --subset 500
```

Full BigEarthNet (~90GB) at https://bigearth.net; RSVQA/VRSBench via their sites/HF.

## Size & Subset Used

Prototype uses synthetic subset (200-500 template QA) for pipeline verification + programmatic demo imagery (no downloads). Real subset 1k-2k recommended for LoRA on 24GB GPU.

## Preprocessing

- GeoTIFF/TIFF via rasterio → PIL fallback; synthetic images are PNG for demo (allowed per PS for benchmarks).
- Training JSON: `{image, query, answer, split}` per entry; see `datasets/processed/synthetic_demo.json`.

## Expected Directory Structure

```
datasets/
  raw/bigearthnet/manifest_subset_1000.json
  processed/synthetic_demo.json
  processed/bigearthnet_subset_1000.json
```

## Demo Data

```
data/demo/
  single_image/scene_*.png/.tif
  temporal_pairs/{flood,urban,forest}_T1/T2.png
  optical_sar_pairs/pair*_optical/sar.png
  invalid_inputs/{small_128.png,large_512.png,corrupt.png}
```

Generated via `python scripts/prepare_demo_data.py`.

