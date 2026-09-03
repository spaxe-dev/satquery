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

- **Training scaffold verified** on synthetic 200 samples (`datasets/processed/synthetic_demo.json`, manifest at `models/checkpoints/satquery-lora/training_manifest.json`).
- **Best model for demo:** heuristic RS vision (always works) + classical change/SAR/optical pipelines (real CV). For RS-adapted ceiling, use `Qwen/Qwen2-VL-2B-Instruct` QLoRA r=8 on cloud GPU — see `docs/MODEL_CARD.md`.
- **Failed:** None — no training attempted on 4GB (honest). QLoRA on larger data is the next experiment.

## Datasets Used

- Synthetic landscapes + simulated SAR (`data/demo/`).
- Synthetic QA templates (`datasets/processed/synthetic_demo.json`).
- Real manifests scaffolded for BigEarthNet/VRSBench/RSVQA/CDVQA/SEN1-2 via `scripts/download_datasets.py`.

## Where Checkpoints Are

- `models/checkpoints/satquery-lora/` (dry-run manifest; real `adapter_model.bin` after cloud training).

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
python training/train_lora.py --dry_run --subset 500
python training/train_lora.py --dataset synthetic --subset 2000 --epochs 2 --use_4bit
# Wire checkpoint into tools/vqa_tool.py::_try_load_vlm (load PEFT adapter)
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
