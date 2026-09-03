# SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing (SIH26167)

> **“Ask satellite imagery a question.”**

SatQuery is an **agentic, query-driven** remote-sensing assistant. The user uploads imagery (single, bi-temporal pair, or co-registered optical+SAR) and asks in natural language. The system **validates inputs, selects specialist tools, executes them, fuses evidence, estimates confidence, and returns text + visual evidence + an auditable execution trace**.

This is **not** “ChatGPT for satellite images”. The novelty is **query-driven orchestration of specialist remote-sensing models** with explicit trust machinery.

---

## 1. What Problem It Solves

Remote-sensing AI is fragmented: separate models for land-cover, VQA, change detection, SAR, etc., each requiring GIS expertise. A disaster manager who wants “Where was inundated?” should not have to operate multiple tools. SatQuery moves that complexity from the user into a deterministic agent.

## 2. Who Uses It

**Direct:** disaster-response teams, urban-planning analysts, agriculture/land-use analysts, forest/environment monitors, water-resource teams.
**Indirect benefit:** public downstream (flood response, planning) via faster institutional decisions.

## 3. Core Architecture

```
USER
 | query + image(s)
 v
+--------------------------+
|      SATQUERY UI         |
|  Drop zone · Query bar   |
+------------+-------------+
             |
             v
+--------------------------+
|       FASTAPI            |
|  /api/analyze · /health  |
+------------+-------------+
             |
             v
+--------------------------+
|     SATQUERY AGENT       |
| Query Interpret · Validate · Plan · Select |
+------+--------+----------+
       |        |
       v        v
+----------+  +-----------+
|   MODEL  |  |   TOOL    |
| REGISTRY |  | REGISTRY   |
+----+-----+  +-----+-----+
     |              |
     +------+-------+
            |
    +-------+--------+---------+
    |       |        |         |
    v       v        v         v
   VQA   Change    Optical    SAR
         Detector  Analyzer   Analyzer
    |       |        |         |
    +-------+--------+---------+
            |
            v
+--------------------------+
|    EVIDENCE FUSION       |
| Text + Spatial + Agreement |
+------------+-------------+
             |
             v
+--------------------------+
|      RESULT ENGINE       |
| Answer · Visual · Conf · Trace |
+------------+-------------+
             |
             v
+--------------------------+
|       REPORTING          |
| JSON / PDF               |
+--------------------------+
```

## 4. System Flow (for pitching)

```
                    USER
                     |
                     v
             Natural Language Query
                     |
                     v
              +-------------+
              |  SATQUERY   |
              |    AGENT    |
              +------+------+
                     |
             Understand intent
                     |
                     v
             Validate inputs
                     |
                     v
            Select tools/models
                     |
        +------------+-------------+
        |            |             |
        v            v             v
      VQA       Change Model   Optical + SAR
        |            |             |
        +------------+-------------+
                     |
                     v
              Evidence Fusion
                     |
          +----------+----------+
          |          |          |
          v          v          v
       ANSWER     VISUAL     CONFIDENCE
                  EVIDENCE
                     |
                     v
              AUDIT TRACE
                     |
                     v
                  REPORT
```

This graph is also in `docs/ARCHITECTURE.md`.

## 5. Every Model & Tool

| Tool ID | Name | Capability | Base | Modality | Status |
|---------|------|------------|------|----------|--------|
| `satquery-rs-vlm` | SatQuery-RS-VLM | VQA, Captioning, Grounding | Heuristic RS vision (BLIP/Qwen2-VL LoRA swappable) | optical / sar | ready |
| `change-detector` | ChangeDetector | temporal change detection/description | LAB+gray diff, Otsu, morphology, CC filtering | bi-temporal | ready |
| `optical-analyzer` | OpticalAnalyzer | water/built-up/veg evidence | NDVI-proxy/NDWI-proxy + HSV + edge density | optical | ready |
| `sar-analyzer` | SARAnalyzer | backscatter water/built-up | backscatter stats, speckle CV, median despeckle | SAR | ready |
| `fusion-engine` | FusionEngine | cross-modal agreement | IoU/Dice, intersection=high-conf, union=inclusive | optical+SAR | ready |

Registry lives in `config/tool_registry.yaml` — the agent may only select from this list (auditable).

## 6. Agent Behavior

1. **Classify** query via regex-intent patterns (`temporal`, `sar`, `optical_sar`, `grounding`, `captioning`, `vqa`) — deterministic, reproducible.
2. **Decide task** from intents + `num_images` + `modalities`.
3. **Validate** (`validate_inputs` + pair-specific checks): dimensions within 15% for temporal, flag-don’t-reject for SAR/optical ambiguity.
4. **Plan tools** (`plan_tools`) — explicit list with reasons.
5. **Execute sequentially**, collecting confidence + visual evidence.
6. **Fuse** (decision-level, IoU/Dice) and estimate overall confidence.
7. **Return** answer + visual overlays + confidence + trace.

Only the trace (task, tools, params, outputs, timings) is evaluated — no hidden chain-of-thought.

## 7. Input Validation

- Supported: `.tif/.tiff` (via `rasterio` if available, else PIL fallback), `.png/.jpg/.jpeg` for benchmarks/demo.
- Checks: file readability, dimensions (reject <10px, warn <64), number of inputs vs task, modality heuristic (filename + band count + color stats), CRS/transform via rasterio when present, temporal pair dimensions (reject >15% mismatch), optical+SAR compatibility (warn, don’t reject).
- On failure: **refusal with reason**, no hallucination. Demo 5 proves this.

## 8. Visual Evidence

- Change: JET heatmap + red mask overlay (binary change).
- Optical: water (blue) / built-up (amber) masks.
- SAR: water (dark smooth) / built-up (bright rough) masks.
- Fusion: intersection (high-conf) overlay + union; agreement scores.
- Grounding: SVG boxes rendered on image (client-side scaling).

## 9. Confidence

Heuristic, **explicitly labeled** as `is_heuristic: true`. Components:
- `input_quality` (size, georef)
- `model_confidence` (separability, CV, mask coherence)
- `evidence_agreement` (area plausibility, IoU)
- `cross_modal_support` (from fusion IoU)
Overall = mean or fusion-aware weighted mean. Never presented as calibrated probability.

## 10. Dataset Sources & Preparation

| Dataset | Role | Access | Subset for prototype |
|---------|------|--------|----------------------|
| BigEarthNet / reBEN | mandated adaptation | https://bigearth.net — CDLA-Permissive | synthetic manifest + `scripts/download_datasets.py --dataset bigearthnet --subset 500` |
| VRSBench | eval caption/ground/VQA | HF `xiang709/VRSBench` | placeholder manifest |
| RSVQA | eval/training augment | https://rsvqa.sylvainlobry.com | placeholder |
| CDVQA | eval change-VQA | https://github.com/YZHJessica/CDVQA | placeholder — verify access early |
| SEN1-2 | cross-modal pretrain (optional) | mediatum | placeholder |
| Demo data | offline demo without downloads | generated locally | `python scripts/prepare_demo_data.py` → `data/demo/` |

All datasets have `datasets/README.md` and scaffold scripts:
- `scripts/download_datasets.py --list`
- `scripts/prepare_datasets.py --dataset synthetic --subset 500`
- `scripts/verify_datasets.py`

See `docs/DATASETS.md`.

## 11. How Models Are Downloaded

- Primary: `Qwen/Qwen2-VL-2B-Instruct` or `Salesforce/blip-image-captioning-base` via `huggingface_hub` at training/inference time (download-time only, offline thereafter).
- Fallback: heuristic RS vision (no download, always works on 4GB VRAM).
- Registry pins `base_model` and `version`; `docs/MODEL_CARD.md` documents license/size/VRAM.

## 12. How Fine-Tuning Works

See `training/train_lora.py` and `docs/MODEL_CARD.md`.

```
BASE (Qwen2-VL-2B or BLIP)  →  LoRA r=8 alpha=16 on q_proj/v_proj  →  QLoRA 4-bit (optional)
     ↑
 RS data (BigEarthNet synthetic / RSVQA subset / SEN1-2) with instruction templates
     ↓
Adapter checkpoint → models/checkpoints/satquery-lora/adapter_model.bin
     ↓
Swapped into tools/vqa_tool.py (_try_load_vlm) at inference — no app change needed
```

On 4GB laptop: `--dry_run` verifies pipeline + writes `datasets/processed/synthetic_demo.json` and manifest. On 24GB cloud GPU: `bash scripts/cloud_setup.sh && bash scripts/train.sh --dataset synthetic --subset 2000 --epochs 2 --use_4bit`.

Measured on RTX 3050 Laptop 4GB: BLIP base loads in ~1.2GB; Qwen2-VL-2B 4-bit fits in ~3.2GB with offload. LoRA training not run on 4GB.

## 13. How Inference Works

Single request:

```
POST /api/analyze (multipart)
  query: str
  images: File[]
→ validate_image_file (rasterio→PIL, modality heuristic)
→ run_analysis (router → orchestrator → tools)
→ encode overlays as base64
→ save reports/report_<id>.json
→ return JSON with previews + evidence + trace
```

See `backend/app/services/orchestrator.py`.

## 14. How Frontend Works

- Single-file `frontend/dist/index.html` (Tailwind CDN, no build needed) — served at `/` by FastAPI; also a Vite React scaffold in `frontend/src/` for extension.
- Features: drag-drop GeoTIFF/PNG/JPEG, query presets, 5 demo buttons, live trace panel, result/evidence/confidence cards, grounding SVG, report download.

## 15. How Backend Works

- FastAPI `backend/app/main.py` + `backend/app/api/routes.py` ( `/api/health`, `/api/registry`, `/api/analyze`, `/api/reports` ).
- Services: `validation.py`, `orchestrator.py`.
- Tools: `tools/*.py` (real CV pipelines, not LLM hallucinations).
- Agent: `agent/router.py`.

## 16. How to Start

See `QUICKSTART.md`. One-command:

Windows:
```powershell
.\scripts\start.ps1
# or: python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Linux:
```bash
./scripts/start.sh
```

Then open `http://localhost:8000` and `http://localhost:8000/docs`.

## 17. How to Run Each Demo

In the UI, click:

- **Demo 1** — single: `Describe the land-cover…` → captioning + VQA
- **Demo 2** — temporal: `What changed…` → flood_T1/T2 + change map
- **Demo 3** — opt+SAR: `Use optical and SAR together…` → pair1 + fusion + agreement
- **Demo 4 (Hero)** — compound: `What changed and can SAR corroborate?` → flood_T1/T2 + flood_sar + change + SAR + fusion (primary demo)
- **Demo 5** — invalid: mismatched 128 vs 512 → **refusal** with reason (trust demo)

Or via API: `curl -F "query=What changed?" -F "images=@data/demo/temporal_pairs/flood_T1.png" -F "images=@data/demo/temporal_pairs/flood_T2.png" http://localhost:8000/api/analyze`

Full presenter script: `docs/DEMO_SCRIPT.md`.

## 18. How to Run Tests

```bash
python -m pytest tests -v
# 25 tests: validation, agent, tools, end-to-end (including refusal)
```

## 19. How to Use Cloud GPU

See `docs/CLOUD_GPU_SETUP.md`.

```bash
# On cloud instance (RTX 4090 24GB or A100)
bash scripts/cloud_setup.sh      # nvidia-smi, torch CUDA, deps
bash scripts/cloud_verify.sh     # GPU, RAM, datasets, checkpoints
python scripts/prepare_demo_data.py
python training/train_lora.py --dry_run --subset 500
python training/train_lora.py --dataset synthetic --subset 2000 --epochs 2 --use_4bit
bash scripts/export_model.sh
bash scripts/cloud_cleanup.sh
# Copy models/checkpoints/satquery-lora/ to persistent storage before stopping instance
```

## 20. How to Shut Down Cloud GPU

Stop/delete instance in console **after** verifying `models/checkpoints/satquery-lora/` copied to persistent storage (e.g., `scp` or mounted bucket). Never depend on ephemeral disk.

## 21. What Works (green)

Single VQA, captioning, grounding (heuristic boxes), change detection with real mask + description, optical/SAR analysis + fusion + agreement, deterministic agent routing, input validation + refusal, visual evidence, confidence (heuristic), execution trace, JSON reports + report listing, PDF generation via `scripts/generate_pdf.py`, demo data synthesis, test suite (25/25), offline demo.

## 22. What Does Not Work / Partial (yellow)

- **Real LoRA checkpoint not loaded by default** on 4GB — heuristic fallback is primary; checkpoint swappable when trained on 24GB.
- Grounding is heuristic (color/texture), not a detector (e.g., Grounding-DINO) — IoU modest.
- No live co-registration (assumes pre-registered per PS).
- PDF is via `reportlab` script, not auto-generated per request.
- No auth, no tiling for sub-meter Cartosat (stretch goal).

## 23. What Is Heuristic

All confidence scores; VQA/caption template + CV stats; change threshold (Otsu); optical water/built masks (NDVI/NDWI proxies); SAR water/built (backscatter percentiles); fusion agreement (IoU). All labeled `is_heuristic`.

## 24. What Remains to Be Improved

Wire trained LoRA checkpoint, replace grounding with detector head, add tiling + radiometric augmentation for Cartosat/RISAT domain shift, add MI co-registration check, calibrate confidence on held-out set, add Postgres/MinIO, add queue for GPU jobs.

## 25. Known Limitations

See `docs/HANDOFF.md` and model cards. Domain shift (Sentinel 10m → Cartosat sub-meter) is untested without hidden data; SAR modality detection can misfire on RGB composites (penalized); very small images (<64px) unreliable.

## 26. Project Structure

```
satquery/
├── README.md · QUICKSTART.md
├── docs/ (ARCHITECTURE, HANDOFF, DECISION_LOG, IMPLEMENTATION_STATUS, CLOUD_GPU_SETUP, DATASETS, MODEL_CARD, DEMO_SCRIPT, PITCH, PERFORMANCE, FINAL_ENGINEERING_REPORT)
├── frontend/ (dist/index.html + Vite scaffold)
├── backend/app/ (main, api/routes, services/validation+orchestrator, models/schemas)
├── agent/router.py
├── tools/ (vqa, change, optical, sar, fusion)
├── training/train_lora.py
├── config/tool_registry.yaml
├── data/demo/ (single_image, temporal_pairs, optical_sar_pairs, invalid_inputs)
├── datasets/ (raw/processed/metadata/manifests)
├── scripts/ (prepare_demo_data, download_datasets, prepare_datasets, verify_datasets, generate_pdf, cloud_*.sh, start.*)
├── tests/ (validation, agent, tools, end_to_end)
└── reports/
```

## 27. Requirements Mapping

| PS Requirement | Where |
|----------------|-------|
| RS adaptation (BigEarthNet) | `training/train_lora.py`, `docs/MODEL_CARD.md` |
| Single VQA (mandatory) | `tools/vqa_tool.py`, Demo 1 |
| Captioning / Grounding | same tool, Demo 1 + grounding query |
| Change analysis | `tools/change_tool.py`, Demo 2/4 |
| Optical-SAR fusion | `tools/optical_tool.py` + `sar_tool.py` + `fusion_tool.py`, Demo 3/4 |
| Agentic orchestration | `agent/router.py` + `backend/app/services/orchestrator.py` + trace UI |
| Input validation | `backend/app/services/validation.py` |
| Model/tool registry | `config/tool_registry.yaml` |
| Visual evidence | overlays + `frontend/dist/index.html` |
| Confidence | heuristic breaks in every tool + `docs/MODEL_CARD.md` |
| Execution trace | orchestrator trace + UI panel |
| JSON/PDF reports | `/api/reports` + `scripts/generate_pdf.py` |
| GUI/web app | `frontend/dist/index.html` served by FastAPI |

---

**Handoff:** `docs/HANDOFF.md` · **Status:** `docs/IMPLEMENTATION_STATUS.md` · **Decisions:** `docs/DECISION_LOG.md` · **Demo:** `docs/DEMO_SCRIPT.md` · **Pitch:** `docs/PITCH.md`
