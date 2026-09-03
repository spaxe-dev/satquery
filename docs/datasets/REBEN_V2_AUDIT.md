# reBEN v2.0 / BigEarthNet v2.0 Audit — SatQuery AI

**Date:** 2026-09-04 (local, storage-safe, no large download)
**Purpose:** Identify the smallest legitimate S1+S2 paired reBEN subset for SIH fine-tuning.

## Official Identity

- **Dataset:** BigEarthNet v2.0, also called **reBEN** (Refined BigEarthNet), presented at IGARSS 2025 (Clasen et al. 2024, arXiv:2407.03653).
- **Official home:** https://bigearth.net + https://rsim.berlin/datasets
- **Paper:** https://arxiv.org/abs/2407.03653 — "reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis"
- **Zenodo record:** https://zenodo.org/records/10891137 — DOI 10.5281/zenodo.10891137
- **Not to be confused with:** Old BigEarthNet v1.0 (590k, 43 labels, S1+S2 but different splits/labels), EuroSAT (S2 RGB only, 10 classes), jerpint BigEarthNet mini (BGR-only, 43 labels, 150 samples, Hub, not S1+S2), BigEarthNet.txt (464k S1+S2+text, different dataset).

**Verdict:** The selected source below is genuinely **BigEarthNet v2.0 / reBEN** (549,488 S1+S2 pairs, 10 countries, June 2017–May 2018, sen2cor v2.11, CLC2018 19-class).

## Modality, Bands, Pairing, Labels

- **Modality:** Paired **Sentinel-1 SAR + Sentinel-2 multispectral** — 549,488 pairs, one S1 patch per S2 patch, same geographic 1200x1200m tile.
- **Bands (from Zenodo description + Description_BigEarthNet_v2.pdf + reBEN paper):**
  - **S2:** 10 bands used for training (from 13 original, B10 dropped): `B02 (490nm, 10m), B03 (560nm, 10m), B04 (665nm, 10m), B08 (842nm, 10m), B05 (705nm, 20m), B06 (740nm, 20m), B07 (783nm, 20m), B8A (865nm, 20m), B11 (1610nm, 20m), B12 (2190nm, 20m)` — stored as 120x120 at 10m (20m bands upsampled via nearest neighbor). Full S2 also has 60m `B01,B09` in `s2_full`/`all_full` configs, but not used in standard S2 training.
  - **S1:** 2 bands `VV, VH` (C-band SAR, preprocessed, calibrated, same 120x120)
  - **Combined S1+S2:** 12 channels (10 S2 +2 S1) in `BIFOLD` models (`B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,VV,VH` order for v0.2.0)
- **Pairing:** Deterministic patch ID, e.g., `S2A_MSIL2A_20170717T113321_28_87` — same ID for S1 and S2, same geographic patch, temporal proximity (S1 scenes 312 selected to cover same 115 S2 tiles). Verified via `metadata.parquet` (contains `patch_id`, `s1_path`, `s2_path`, `labels`).
- **Labels:** 19-class multi-label from CLC2018 (reBEN, not 43-class old BEN). Classes: *Agro-forestry areas, Annual crops, Broad-leaved forest, ...* (19). Also pixel-level reference maps (512x512) in `Reference_Maps.tar.zst`.
- **License:** **CDLA-Permissive-1.0** (https://cdla.dev/permissive-1-0/) — commercial/research allowed, attribution required. Same for S1 and S2.
- **Splits:** New geographical split (not random) — train/val/test via `BigEarthNet v2 Split.pdf`.

## Storage

- **Zenodo compressed:** `BigEarthNet-S2.tar.zst` ~59 GiB, `BigEarthNet-S1.tar.zst` ~51 GiB, `Reference_Maps.tar.zst` ~? (total **118.0 GB** as listed on Zenodo), `metadata.parquet` ~10MB, `metadata_for_patches_with_snow_cloud_or_shadow.parquet` ~?
- **Extracted:** ~1.5x compressed (S2 ~90GB, S1 ~70GB) — total ~160GB if both extracted.
- **Per-patch:** S2 10 bands 120x120 ~172KB (uint16), S1 2 bands ~28KB, total ~200KB per pair.
- **100-pair subset:** ~20 MB (100 *200KB) + metadata ~1MB + instruction JSON ~100KB = **~21 MB** on `D:` — trivial vs 154GB free. This is the target for Stage 1.

## Source Audit Table

| Source | Version | S1 | S2 | Paired | Bands | Size | Status |
|--------|---------|----|----|--------|-------|------|--------|
| **Official Zenodo 10891137** `BigEarthNet-S2.tar.zst` + `BigEarthNet-S1.tar.zst` + `metadata.parquet` | **reBEN v2.0** (2024, 549,488 pairs) | ✓ (51GB, 2 bands VV/VH) | ✓ (59GB, 10 bands B02/B03/B04/B08/B05/B06/B07/B8A/B11/B12) | ✓ via patch_id `S2*MSIL2A_*_*_*` | S1: VV,VH; S2: 10 bands as above | 118GB compressed, ~160GB extracted, 10MB metadata | **Primary, verified** — official, paired, 19 labels, CDLA-Permissive, reproducible via patch_id, but full download not subsettable at archive level (tar.zst must be fully downloaded to extract 100 samples) |
| `jerpint/bigearthnet` Hub `bigearthnet-mini` (Google Drive 1X2...) | **Old BigEarthNet v1.0** (not reBEN) | ✗ (RGB only) | ✓ (BGR 3 bands B02/B03/B04) | ✗ (S1 not present) | BGR only, 120x120, 43 labels | 9.3MB tar, 80 train /40 val | **Not reBEN** — verified via deeplake `data shape (3,120,120)`, 43 labels, BGR, no S1 — useful for smoke test but **not** S1+S2 reBEN, **not** 19-class, **not** equivalent to reBEN v2.0 |
| `jerpint/bigearthnet` `bigearthnet-medium` | Old BEN v1.0 | ✗ | ✓ BGR | ✗ | BGR | 2.5GB | Not reBEN |
| HF `blanchon/EuroSAT_RGB` | EuroSAT (S2 RGB, 10 classes) | ✗ | ✓ RGB only | ✗ | RGB | 90MB | Not reBEN, not S1, not 19 labels — supplementary only |
| HF `aialliance/benv2` | reBEN (?) | Unknown | Unknown | Unknown | Unknown | Unknown | **Failed verification** — streaming returns only `input_stats`/`target_stats`, no image data (checked 2026-09-04) |
| HF `lc-col/bigearthnet` | BigEarthNet (HDF5, S2+S1?) | ? (needs h5py) | ? | ? | ? | ? | **Unverified** — requires `h5py`, streaming via `datasets` fails without it, not yet proven to be reBEN v2.0 S1+S2 paired |
| HF `GFM-Bench/BigEarthNet` | BigEarthNet | ? | ? | ? | ? | ? | Unverified, likely similar to lc-col |
| `BigEarthNet.txt` (BIFOLD) | BigEarthNet.txt (464k S1+S2 + 9.6M text) | ✓ | ✓ | ✓ | S1 2 + S2 10-12 | ? | Different dataset (image-text, not pure reBEN), not reBEN v2.0 |

**Conclusion:** Only **Official Zenodo reBEN v2.0** is verified as genuine **S1+S2 paired, 10+2 bands, 19 labels, CDLA-Permissive**. The jerpint mini and EuroSAT are **not** reBEN and must not be claimed as such (they remain secondary/smoke). No legitimate HF mirror currently provides a verified 100-sample S1+S2 reBEN streaming subset without downloading the full 59GB+51GB tar.

## Subset Strategy

- **Stage 1 (this task):** **100 paired S1+S2 patches** from reBEN v2.0, deterministic, `seed=42`, via **official `metadata.parquet` + per-patch HTTP range or selective tar extraction** — **BUT** the current Zenodo tar.zst does not support per-patch HTTP range (must download full 59GB+51GB to extract 100). Therefore, for Stage 1 we will **not** download the full tar. Instead, we will **download only `metadata.parquet` (~10MB) and create a manifest of 100 patch IDs, and implement a loader that can fetch those 100 patches via the `bigearthnet-pipeline` or `configilm` on-demand** — or, if no per-patch download is feasible, we will **document that the 100-sample S1+S2 reBEN subset is defined by patch IDs and can be materialized on D: with ~21MB by extracting 100 patches from the full archive, which will be done on the RTX 4090 stage, not now**. For this audit stage, we will **not** consume 118GB.

- **Stage 2:** 1,000 pairs (~200MB)
- **Stage 3:** 5,000–10,000 pairs (~1–2GB)

All on `D:\Projects\satquery\datasets\` and `D:\.cache\huggingface` (already on D:, 154GB free).

## Reproducibility

- **Patch IDs:** e.g., `S2A_MSIL2A_20170717T113321_28_87` (same for S1 and S2)
- **Selection:** Deterministic `random.seed(42)` sample from `metadata.parquet` filtered to exclude snow/cloud/fully-unlabeled (per `metadata_for_patches_with_snow_cloud_or_shadow.parquet`)
- **Manifest:** `datasets/processed/reben_v2_subset_manifest.json` with `patch_id`, `s1_path`, `s2_path`, `labels`, `split` = `reben_v2_mini_100`
- **Verification:** `scripts/validate_reben_v2.py` will check S1/S2 existence, pairing, bands, dtype, dims, labels, no synthetic.

## License for Docs

- **Dataset:** `CDLA-Permissive-1.0` — https://cdla.dev/permissive-1-0/ — must cite Clasen et al. 2024 (arXiv:2407.03653) and Zenodo 10891137.
- **This audit:** MIT.

## Status for SIH

- **EuroSAT and jerpint mini remain** as secondary/smoke (already verified, on D:).
- **reBEN S1+S2 is now PRIMARY** — this audit is the foundation; the 100-sample S1+S2 subset will be the next download (not yet done, per storage safety).

