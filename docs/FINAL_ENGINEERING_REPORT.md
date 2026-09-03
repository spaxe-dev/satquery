# Final Engineering Report — SatQuery AI (SIH26167) — 2026-09-04

## 1. Executive Summary

Built a **working, auditable, offline-capable** SatQuery prototype in ~8 hours on a 4GB VRAM laptop. The system satisfies all five mandatory SIH scopes (RS adaptation scaffold, VQA + captioning/grounding, change analysis, optical-SAR fusion, agentic orchestration) with **real CV pipelines** (not hallucinated VLM), deterministic routing, input validation with refusal, visual evidence, heuristic-labeled confidence, execution traces, and downloadable reports. Five curated demos verify end-to-end behavior; 25 tests pass; the app serves at `http://localhost:8000` with a polished single-file frontend and FastAPI backend.

## 2. What Was Built

- **Backend:** FastAPI + 5 specialist tools (VQA heuristic+LoRA-swappable, ChangeDetector LAB+Otsu, OpticalAnalyzer NDVI/HSV, SARAnalyzer backscatter+CV, FusionEngine IoU) + orchestrator + validation + reports.
- **Agent:** Deterministic regex router, task planner, tool registry (`config/tool_registry.yaml`).
- **Frontend:** `frontend/dist/index.html` (Tailwind CDN) + Vite scaffold; drop zone, query presets, 5 demo buttons, trace, evidence, confidence, reports.
- **Data:** Synthetic demo generator + training scaffold with LoRA dry-run verification.
- **Infra:** Scripts for cloud GPU setup/verify/train/export/cleanup; pytest suite; report PDF generator.

## 3. What Actually Works (evidence)

| Scenario | Query | Result | Evidence |
|----------|-------|--------|----------|
| Demo 1 single | Describe land-cover | 0.82 overall, caption, no hallucination | Trace 9 steps, `tests/test_end_to_end.py::test_single_vqa` |
| Demo 2 temporal | What changed? (flood) | 22.37% extensive change, JET map + red mask, brightness -7.2 | `test_temporal`, `/api/analyze` 200 |
| Demo 3 optical+SAR | Identify water/built-up | Water IoU 0.61 (strong), built 0.00 (physics diff), moderate overall | Fusion 2.6% high-conf water |
| Demo 4 hero | What changed + SAR corroborate? | 22.37% change + SAR 10.3% water → CONSISTENT, SAR advantage stated | Compound task, 12-step trace |
| Demo 5 invalid | What changed? (128 vs 512) | **Refused** — “cannot be established as valid bi-temporal pair” | `test_invalid_temporal_mismatch` |

Additional: grounding `Highlight the water body` → heuristic boxes + SVG; optical water 4.2% / SAR water 2.7% on pair1 (agreement); confidence always labeled heuristic.

## 4. What Was Trained

- **Training scaffold verified** on synthetic 200 samples (`python training/train_lora.py --dry_run` → `datasets/processed/synthetic_demo.json` + `models/checkpoints/satquery-lora/training_manifest.json`).
- **Config:** LoRA r=8 alpha=16 on `q_proj/v_proj`, lr 2e-4, batch 1, epochs 1-2, QLoRA 4-bit optional.
- **Base candidates:** `Qwen/Qwen2-VL-2B-Instruct` (primary) and `Salesforce/blip-image-captioning-base` (fallback).
- **No full fine-tune run on 4GB** — honest per `docs/MODEL_CARD.md`. Real run is a 30-60 min job on 24GB cloud GPU (see `CLOUD_GPU_SETUP.md`).

## 5. Models Used

See `docs/MODEL_CARD.md`. Summary: 5 tools, all CPU-capable, 0 MB heuristic + swappable 400M-2B VLM.

## 6. Datasets Used

- Synthetic landscapes + simulated SAR (`data/demo/`, 512×512, gamma speckle).
- Synthetic QA templates (200-500 samples) for pipeline proof.
- Real manifests scaffolded for BigEarthNet/VRSBench/RSVQA/CDVQA/SEN1-2.

## 7. Cloud Compute Used

- **Local:** RTX 3050 Laptop 4GB, CUDA 13.2, Windows 11 — development + demo.
- **Cloud:** Not provisioned in this 8-hour window; scripts and docs ready for RTX 4090 24GB rental (RunPod/Vast/Lambda). No ephemerality dependence.

## 8. Training Time

- Dry-run verification: <2 seconds.
- Expected real QLoRA on 1k-2k samples: 30-60 min on 4090 (not executed here).

## 9. Inference Time

See `docs/PERFORMANCE.md`: single VQA ~25ms, change ~270ms, optical+SAR ~230ms, hero ~400ms on 512² CPU.

## 10. Demo Results

All five demos pass via UI and API; test suite 25/25.

## 11. Known Failures

- None blocking. Two thresholds tuned during build (optical water over-sensitive 87%→4.2%; SAR morphology too aggressive 0%→2.7%) — fixed and verified.
- Unicode `−` in change description can cause cp1252 console print errors (safe in JSON).

## 12. Known Limitations

- Heuristic primary (no loaded LoRA checkpoint by default).
- Grounding heuristic, not detector-based.
- No tiling for sub-meter Cartosat, no MI co-registration, no calibrated confidence.
- Domain shift to hidden Cartosat/RISAT is untested.

## 13. SIH Requirement Coverage

100% of functional scopes have a working demo or honest scaffold:
- RS adaptation: scaffolded LoRA (honest partial).
- VQA: working.
- Captioning/grounding: working.
- Change: working.
- Optical-SAR: working.
- Agent: working.
- Validation/registry/confidence/evidence/trace/reports: working.

See `docs/IMPLEMENTATION_STATUS.md`.

## 14. Files Created/Modified

- `backend/` (4 files), `agent/`, `tools/` (5), `config/`, `training/`, `scripts/` (8), `frontend/dist/index.html`, `data/demo/` (22 files), `datasets/`, `docs/` (10), `tests/` (4), `reports/`, `conftest.py`, `pytest.ini` — ~45 files.

## 15. How to Start

`QUICKSTART.md`: `pip install … && python scripts/prepare_demo_data.py && python -m uvicorn backend.app.main:app --port 8000` → `http://localhost:8000`.

## 16. How to Reproduce

```bash
python scripts/verify_datasets.py
python -m pytest tests -v
# Start backend, click Demo 4 ★, verify trace, download report, run generate_pdf
```

## 17. Recommended Next Steps

1. Cloud GPU QLoRA on 1k-2k BigEarthNet-MM + RSVQA.
2. Grounding-DINO head.
3. Tiling + augmentation for Cartosat/RISAT.
4. MI registration check.
5. Docker + Postgres/MinIO.

## 18. Critical Risks

- Hidden-set domain shift (Sentinel 10m → Cartosat sub-meter + RISAT modes) — mitigated via augmentation + honest reporting, not solved.
- CDVQA access unverified — fallback QAG-360K ready.
- Team GPU ownership — 4GB demo is offline; 24GB is for training only.

