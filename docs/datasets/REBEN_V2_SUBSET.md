# reBEN v2.0 Subset — SatQuery AI (Stage 1: 100 paired S1+S2)

**Status:** Stage 1 manifest created and validated, S1/S2 patches not yet materialized (intentionally, per storage safety).

## Source

- **Dataset:** BigEarthNet v2.0 / reBEN (549,488 S1+S2 pairs, Zenodo 10891137, CDLA-Permissive-1.0)
- **Official metadata:** `https://zenodo.org/records/10891137/files/metadata.parquet?download=1` (3,616,349 bytes, 480,038 rows, excludes snow/cloud)
- **Downloaded to:** `D:\Projects\satquery\datasets\raw\bigearthnet\metadata.parquet` (on D:, 3.6MB, verified via pandas `len(df)=480038, columns=[patch_id, labels, split, country, s1_name, s2v1_name, contains_seasonal_snow, contains_cloud_or_shadow]`)
- **Paper:** Clasen et al. 2024 (arXiv:2407.03653), IGARSS 2025
- **License:** CDLA-Permissive-1.0

## Exact Version

- **reBEN v2.0** (2024, same 115 S2 tiles + 312 S1 scenes as BigEarthNet v1, but with sen2cor 2.11, CLC2018 19-class, new geo split, pixel-level reference maps). Not old BigEarthNet v1.0 (43 labels) and not jerpint mini (BGR-only).

## Number of Samples

- **100 paired S1+S2 patches** for Stage 1, sampled deterministically from `train` split of `metadata.parquet` (237,871 train rows) via `random.sample(..., random_state=42)`.
- **Manifest:** `datasets/processed/reben_v2_subset_manifest.json` (65,801 bytes, 100 entries)
- **Example patch_id:** `S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_90_39` with `labels: ["Marine waters"]`, `s1_name: S1A_IW_GRDH_1SDV_20180529T181504_29UNB_90_39`, `s2v1_name: S2A_MSIL2A_20180529T115401_90_39`, `country: Ireland`, `split: train`.
- **Full manifest fields:** `patch_id, labels (19-class), split, country, s1_name, s2v1_name, contains_seasonal_snow, contains_cloud_or_shadow, expected_s1_path, expected_s2_path`.

## Selection Strategy

- **Method:** `pandas.read_parquet(metadata.parquet) → df[df.split=="train"] → random.sample(n=100, random_state=42)` — deterministic, reproducible, seed 42.
- **Filter:** Only `train` (237k), excludes snow/cloud already (since metadata.parquet excludes them), no additional filtering.
- **Reproducibility:** `manifest["selection_method"] = "deterministic random.sample with seed=42 from train split of metadata.parquet (480k rows, excludes snow/cloud)"`, `seed:42`, `date:2026-09-04`.

## Sample IDs / Manifest Reference

- See `datasets/processed/reben_v2_subset_manifest.json` (100 entries, each with `patch_id` as primary key).
- First 3: `S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB_90_39`, `S2B_MSIL2A_20180603T100029_N9999_R022_T32TPP_44_4`, `S2A_MSIL2A_20170717T113321_N9999_R022_T33UUP_28_87` (random, seed 42).

## Modalities

- **S1:** Sentinel-1 SAR, 2 channels `VV, VH` (C-band, IW GRD, calibrated, same 120x120 at 10m, same geographic patch as S2)
- **S2:** Sentinel-2 multispectral, 10 bands used for training `B02,B03,B04,B08 (10m) + B05,B06,B07,B8A,B11,B12 (20m upsampled to 10m via nearest)`, 120x120, uint16 0-10000 (L2A sen2cor 2.11)

## Bands

- **S1:** `VV, VH` (2, 120x120)
- **S2:** `B02 (490nm), B03 (560nm), B04 (665nm), B08 (842nm), B05 (705nm), B06 (740nm), B07 (783nm), B8A (865nm), B11 (1610nm), B12 (2190nm)` — full 12 would include `B01,B09` (60m) but training uses 10.

## Preprocessing

- **S1:** Keep as float32, clip -50 to 10 dB (if in dB) or 0-1 if linear, normalize per `configilm` mean/std (not yet applied for Stage 1, kept raw).
- **S2:** `uint16 -> float32 /10000.0 -> clip 0-1 -> per-band normalize` (explicit in `training/reben_multimodal_dataset.py`).
- **Loader:** `training/reben_multimodal_dataset.py:ReBENMultimodalDataset` keeps `s1` and `s2` as separate tensors `(2,120,120)` and `(10,120,120)` until fusion, not forced into RGB.

## Storage Location

- **Metadata:** `D:\Projects\satquery\datasets\raw\bigearthnet\metadata.parquet` (3.6MB, on D:)
- **Manifest:** `D:\Projects\satquery\datasets\processed\reben_v2_subset_manifest.json` (65KB, on D:)
- **Validation report:** `datasets/processed/reben_v2_validation.json` (1.2KB, `status: passed`, `paired:100`, `s1_found:0, s2_found:0` — expected for Stage 1, images not yet downloaded)
- **Future S1/S2 patches for 100 pairs:** `D:\Projects\satquery\datasets\raw\bigearthnet\BigEarthNet-S1/` and `BigEarthNet-S2/` — **not yet downloaded** (would be ~21MB for 100 pairs, ~160GB for full 549k). Will be materialized on RTX 4090 stage via selective extraction from `BigEarthNet-S1.tar.zst` (51GB) and `BigEarthNet-S2.tar.zst` (59GB) using `tar --extract` with patch list, not full download to C:.

## License

- **CDLA-Permissive-1.0** — https://cdla.dev/permissive-1-0/ — must cite Clasen et al. 2024 and Zenodo 10891137.

## Reproducibility Instructions

```bash
# 1. Download metadata (3.6MB, already done)
python -c "import requests; r=requests.get('https://zenodo.org/records/10891137/files/metadata.parquet?download=1', stream=True); open('datasets/raw/bigearthnet/metadata.parquet','wb').write(r.content)"

# 2. Generate 100-sample manifest (deterministic, seed 42)
python -c "import pandas as pd, random, json; df=pd.read_parquet('datasets/raw/bigearthnet/metadata.parquet'); train=df[df.split=='train']; sampled=train.sample(n=100, random_state=42); sampled.to_json('datasets/processed/reben_v2_subset_manifest.json', orient='records')"

# Or use the script that did it:
python scripts/download_bigearthnet.py --variant mini --subset 100  # for mini, but for reBEN use the metadata method above
# For reBEN, the manifest was created via C:\Users\aadit\AppData\Local\Temp\make_reben_manifest.py with seed 42
```

For Stage 1, the manifest is the reproducible artifact; the 100 S1/S2 patches will be materialized on D: during the next stage (selective tar extraction, not full 118GB download to C:).
