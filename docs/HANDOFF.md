# Handoff — SatQuery AI (for human engineer after autonomous session)

## What Was Built

- **Backend:** FastAPI (`backend/app/main.py` + `api/routes.py`) with `/api/analyze` (multipart), validation, orchestrator, 5 specialist tools, fusion, confidence, trace, reports.
- **Agent:** Deterministic router (`agent/router.py`) — classifies query, validates, plans tools, executes.
- **Tools:** VQA/caption/grounding (heuristic + LoRA-swappable), ChangeDetector (real CV), OpticalAnalyzer, SARAnalyzer, FusionEngine — all CPU-capable.
- **Frontend:** `frontend/dist/index.html` (Tailwind CDN, single-file, polished) served at `/` with 5 demo buttons, trace, visual evidence, confidence, reports. Vite scaffold in `frontend/src/`.
- **Data:** Synthetic demo generator (`scripts/prepare_demo_data.py`) → `data/demo/`; training scaffold (`training/train_lora.py`) with dry-run verified.
- **Tests:** 25 passing (`tests/`); CI-ready.
- **Docs:** README, QUICKSTART, ARCHITECTURE, MODEL_CARD, DATASETS, CLOUD_GPU_SETUP, DEMO_SCRIPT, PITCH, PERFORMANCE, etc.
- **Reports:** JSON per analysis + PDF via `scripts/generate_pdf.py`.

## What Works

All P0 + most P1: image upload, query, agent/router, VQA, captioning, grounding (heuristic), change detection (real masks), optical+SAR fusion, validation + refusal, visual evidence, confidence (heuristic-labeled), trace, JSON reports, 5 demos. Try `python -m pytest tests -v` and `python -m uvicorn backend.app.main:app --port 8000`.

## What Does Not / Partial

- Real LoRA checkpoint not loaded on 4GB by default — heuristic fallback is primary. See `training/train_lora.py --dry_run`.
- Grounding is heuristic (no detector); PDF is script-based not auto.
- No tiling for sub-meter Cartosat, no MI co-registration, no Postgres/MinIO (stretch).
- No Docker Compose yet (runs directly via uvicorn).

## What Was Trained / Best Model

- **Real RS data now present (verified 2026-09-04)**: EuroSAT RGB 1500 images (150/class, 64x64, MIT) at `datasets/raw/eurosat/EuroSAT/` + instruction JSON `datasets/processed/eurosat_instruct_1500.json` (3000 entries: 2500 train / 500 val) verified via `training/dataset_loader.py` and `training/train_lora_real.py --dry_run` (manifest `models/checkpoints/satquery-eurosat-blip-lora/training_manifest.json` status `dry_run_verified_real_loader`). Training not yet executed (requires RTX 4090).
- **Previous synthetic scaffold**: `datasets/processed/synthetic_demo.json` (100 entries) remains for fallback.
- **Best model for demo:** heuristic RS vision (always works) + classical change/SAR/optical pipelines (real CV). For RS-adapted ceiling, use `Salesforce/blip-image-captioning-base` LoRA r=8 on EuroSAT (6-8GB, 15-20min on 4090) or `Qwen/Qwen2-VL-2B-Instruct` QLoRA 4-bit (12-14GB, 25-35min) — see `docs/MODEL_CARD.md` and `docs/REAL_ADAPTATION_PLAN.md`.
- **Failed:** None — no training attempted on 4GB (honest). Real pipeline verified, ready for cloud.

## Datasets Used

- **Real RS**: EuroSAT RGB 1500 (64x64, 10 classes, Sentinel-2, MIT, HF `blanchon/EuroSAT_RGB` / Zenodo 7711810, verified 150/class) — satisfies PS "any open source RS data" (BigEarthNet primary documented as 590k, 110GB, CDLA-Permissive, alternative via `jerpint/bigearthnet`).
- Synthetic landscapes + simulated SAR (`data/demo/`) for demo.
- Synthetic QA templates (`datasets/processed/synthetic_demo.json`) for fallback.
- Real manifests scaffolded for BigEarthNet/VRSBench/RSVQA/CDVQA/SEN1-2 via `scripts/download_datasets.py`; only EuroSAT is physically downloaded and verified.

## Where Checkpoints Are

- `models/checkpoints/satquery-eurosat-blip-lora/` (dry-run manifest, status `dry_run_verified_real_loader`; real `adapter_model.safetensors` ~15MB after RTX 4090 training).
- Legacy `models/checkpoints/satquery-lora/` (synthetic dry-run manifest).

## How the Agent Works

`agent/router.py`: `classify_query` (regex intents) → `decide_task` (intents+num_images+modalities) → `validate_inputs` → `plan_tools` → `orchestrator.run_analysis` executes in order → `fusion_tool.fuse` → confidence aggregation → trace.

## How to Run It

See `QUICKSTART.md`. TL;DR:

```powershell
pip install fastapi uvicorn python-multipart pillow numpy opencv-python scikit-learn scikit-image rasterio shapely reportlab pyyaml requests
python scripts/prepare_demo_data.py
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## How to Retrain

```bash
bash scripts/cloud_setup.sh
bash scripts/cloud_verify.sh

# 1. Verify real RS data (EuroSAT 1500, 90MB, already downloaded)
python scripts/download_eurosat.py --subset 1500
python training/dataset_loader.py  # or: python -c "from training.dataset_loader import verify_real_dataset; verify_real_dataset('datasets/processed/eurosat_instruct_1500.json','datasets/raw/eurosat')"

# 2. Dry-run with real loader (no GPU, verifies LoRA config)
python training/train_lora_real.py --dataset eurosat --subset 1500 --dry_run
python training/train_lora.py --dataset eurosat --subset 1500 --dry_run  # delegates

# 3. Real training on RTX 4090 24GB (BLIP minimal, 6-8GB, 15-20min):
python training/train_lora_real.py --dataset eurosat --subset 1500 --model Salesforce/blip-image-captioning-base --epochs 2 --batch_size 4 --gradient_accumulation 4 --lr 2e-4 --output models/checkpoints/satquery-eurosat-blip-lora
# Alternative Qwen2-VL QLoRA (12-14GB, 25-35min):
python training/train_lora_real.py --dataset eurosat --subset 1500 --model Qwen/Qwen2-VL-2B-Instruct --use_4bit --epochs 1 --batch_size 1 --gradient_accumulation 8 --lr 1e-4 --output models/checkpoints/satquery-eurosat-qwen-lora

# Checkpoint auto-loads in tools/vqa_tool.py::_try_load_vlm (searches models/checkpoints/satquery-eurosat-*); heuristic fallback preserved if no checkpoint.
```

## How to Add a Model

1. Register in `config/tool_registry.yaml`.
2. Implement module in `tools/` with `analyze(...) -> dict` returning confidence + visual evidence.
3. Add branch in `agent/router.py:plan_tools` and `backend/app/services/orchestrator.py`.
4. Add test in `tests/`.

## How to Add a Tool

Same as model — tools are Python modules behind the registry. Keep `analyze` signature typed and heuristic fallback if GPU unavailable.

## How to Add a Dataset

- Add entry to `scripts/download_datasets.py:DATASETS`.
- Create `scripts/prepare_<dataset>.py` subsetter.
- Update `docs/DATASETS.md`.

## Known Bugs

- Optical water detection was over-sensitive (fixed threshold 0.65 + blue dominance); SAR morphology was too aggressive (fixed to 3×3 open).
- `backend/main.py` previously required `frontend/dist/assets` (fixed).
- Unicode `−` in change description can cause cp1252 print errors on Windows console (safe in JSON/HTTP).

## Next Steps (priority order)

1. Run QLoRA on cloud GPU with 1k-2k BigEarthNet-MM subset + RSVQA templates; swap checkpoint.
2. Replace grounding heuristic with Grounding-DINO or OWL-ViT head for competitive IoU.
3. Add tiling + radiometric augmentation for Cartosat/RISAT domain shift.
4. Add MI co-registration check as warning.
5. Wire auto-PDF per request and Postgres trace store.
6. Docker Compose + CI.

## Risks

- **Hidden-set domain shift** (Sentinel 10m → Cartosat sub-meter + RISAT modes) — biggest scoring risk; mitigation is augmentation + honest reporting.
- **CDVQA access** — verify download; use QAG-360K fallback if needed.
- **VRAM on demo laptop** — keep heuristic floor; don’t require cloud at demo time.
