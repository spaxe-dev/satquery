# Architecture — SatQuery AI (SIH26167)

## High-Level

```
USER
 |
 | natural-language query + image(s)
 v
+--------------------------+
|      SATQUERY UI         |
| React / static (Tailwind)|
+------------+-------------+
             |
             v
+--------------------------+
|       FASTAPI            |
|  /api/analyze            |
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
| Text + Spatial + Agreement|
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

## Data Flow for Pitching

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

## Components

### Frontend (`frontend/dist/index.html`)

- Drag-drop for GeoTIFF/TIFF/PNG/JPEG; previews; modality hint.
- Query textarea with preset buttons for the 5 mandatory queries.
- 5 Demo buttons that fetch `data/demo/...` and auto-analyze.
- Execution Trace panel (mono, step-by-step, with tool badges).
- Result cards: Answer, Caption, Change, Fusion.
- Visual Evidence: JET change map, optical/SAR overlays, fused intersection, grounding SVG.
- Confidence bars (heuristic-labeled) and Models Used chips.
- JSON/PDF report download.

Also a Vite scaffold (`frontend/src/`) for future React work; the static build is intentionally single-file for offline demo.

### Backend (`backend/app/`)

- `main.py` — FastAPI app, CORS, serves frontend, mounts `/assets` if present.
- `api/routes.py` — `GET /api/health`, `GET /api/registry`, `POST /api/analyze` (multipart), `GET /api/reports/{id}`, `GET /api/reports`.
- `services/validation.py` — rasterio-first, PIL fallback, modality heuristic, pair checks.
- `services/orchestrator.py` — router → validation → plan → execute → fuse → confidence → trace.
- `models/schemas.py` — Pydantic schemas for `TaskType`, `ConfidenceBreakdown`, etc.

### Agent (`agent/router.py`)

- `classify_query` — regex intents → list.
- `decide_task` — intents + num_images + modalities → TaskType.
- `plan_tools` — TaskType → ordered tool list with reasons/params.
- `validate_inputs` — task vs num_images + dimension checks.
- `load_registry` — reads `config/tool_registry.yaml`.

Deterministic-first; LLM planner is a future extension, not required for scoring (only trace is scored).

### Tools (`tools/`)

- `vqa_tool.py` — heuristic caption/VQA + grounding boxes (CV-based). BLIP/Qwen2-VL LoRA swappable via `_try_load_vlm`.
- `change_tool.py` — LAB+gray diff, Gaussian blur, Otsu, morphology, CC filtering; outputs JET map + red mask, area %, description with veg/brightness deltas.
- `optical_tool.py` — NDVI-proxy, NDWI-proxy, HSV, Canny edges; water/built/veg masks + overlays.
- `sar_tool.py` — gray backscatter, speckle CV, median despeckle, percentile thresholds; water (low+smooth) / built (bright+rough).
- `fusion_tool.py` — IoU/Dice, intersection=high-conf, union=inclusive, corroboration statements.

All tools are CPU-only, <500ms on 512×512 on laptop.

### Config & Registry

- `config/tool_registry.yaml` — typed registry; agent selects only from here.
- `config/settings.yaml` — app, paths, model, validation, reports.

### Data

- `data/demo/` — synthetic demo imagery (no downloads required).
- `datasets/` — raw/processed/metadata/manifests; scripts scaffold real datasets.

### Reports

- `reports/report_<id>.json` — saved per analysis (query, metas, task, tools, params, result, confidence, trace, timestamp).
- `scripts/generate_pdf.py` — reportlab PDF from JSON.

## Deployment

- Single-GPU Docker Compose is the target; current prototype runs as `uvicorn` directly (no Docker required for demo).
- 4-bit quantized VLMs for inference; sequential tool execution to avoid VRAM thrash.
- Offline-capable after initial `pip install` and optional HF download.

## Technology Choices

- FastAPI over Flask for async, auto-docs, multipart.
- OpenCV + skimage + rasterio for real RS CV (not just LLM).
- Heuristic primary on 4GB VRAM; LoRA/QLoRA scaffold for 24GB cloud GPU (PEFT, bitsandbytes).
- Tailwind CDN for UI polish without build step.

