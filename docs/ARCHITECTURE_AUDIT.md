# SatQuery AI — Architecture Audit (2026-09-05, checkpoint 9dc9c57)

Checkpoint: `9dc9c57 chore: checkpoint before production architecture audit` on `main` (ahead of origin/main by 1, +59 reports/*.json).
Branch: `main`. Untracked after checkpoint: `apps/web/` (Next.js scaffold), `packages/*` (contracts), `apps/api/` empty.
Verified: `packages/*` imports OK (1 pydantic `model_` namespace warning); legacy `agent.router + tools.* + orchestrator` imports OK.

## Executive Verdict

SatQuery is a **working prototype with honest heuristic fallbacks, not a production RS-intelligence system**. The demo path (5 scenarios) runs CPU-only and refuses bad inputs correctly. The new `packages/*` + `apps/web` skeleton is directionally right but **unwired dead code** as of audit.

- Genuinely strong: end-to-end demo works offline (`data/demo/` real PNG/TIF), refusal path (Demo 5), VLM lifecycle metadata (`vlm_attempted/loaded/generation_valid/is_heuristic/fallback_reason` in `tools/vqa_tool.py:607-745`, `orchestrator.py:329-341`), 10 test files CPU-green, static demo serving fix.
- Genuinely weak: no trained RS-VLM in prod path; grounding 100% heuristic; change without registration; optical/SAR physics invalid on RGB demos; `page.tsx` 59KB/1093-line monolith; `apps/api/` empty; `packages/*` duplicated, unused.
- Prototype-only: all specialist tools (`tools/*.py` — LAB+Otsu, NDVI-proxy, luminance-as-backscatter, IoU arithmetic), confidence math, grounding boxes, fusion corroboration text.
- Production-capable: FastAPI multipart validation + refusal JSON, report persistence, health/registry endpoints, demo static mount.
- BS/hype: `config/tool_registry.yaml` claims Siamese/CLIP/learned fusion (code is CV-only); `settings.yaml` claims Qwen2-VL primary (no Qwen checkpoint exists); `page.tsx` references `docs/DATA_SOURCES.md/MODEL_ABSTRACTION.md/TOOL_ABSTRACTION.md` which do not exist; reBEN 100-pair "verified" with 0 bytes on disk.
- Skeptical SIH judge attacks: (1) "Where is BigEarthNet fine-tune? Only manifest + 8-sample smoke loss 9.82." (2) "Grounding confidences 0.72 — calibrated on what?" (3) "Change % without CRS/registration — pixel % ≠ m²." (4) "SAR corroboration on RGB composites?" (5) "`?` regex matches everything — agentic?"
- Breaks first in prod: memory (full-raster + float LAB + b64 overlays, no tiling), event-loop blocking (`async def analyze` calls sync numpy), disk-fill (`reports/*.json` with b64, no cap/auth).
- Retain: orchestrator refusal semantics, VLM metadata, demo fixtures, validation gates, test suite intent.
- Replace: router regex with `packages/agent/planner.QueryPlanner` (currently forked), direct `tools.*` imports with `ToolRegistry`, direct model calls with `ModelRegistry`, monolithic `page.tsx` with split components.
- Delete/quarantine: smoke adapter from prod candidates, `frontend/src/` stub, duplicated `DEMOS` constants, `packages/fusion/__init__` stub until real.

## Current Architecture

```
browser (:8000/ -> frontend/dist/index.html SPA + /data/demo static)
  -> FastAPI backend/app/main.py -> /api/* (routes.py)
    -> orchestrator.run_analysis() (agent/router + validation + tools.* + fusion)
      -> tools/vqa_tool | change_tool | optical_tool | sar_tool | fusion_tool
      -> reports/report_{id}.json (full response + b64)
browser (:3000/ dev) -> apps/web/app/page.tsx -> rewrites /api/*,/data/demo/* -> :8000
packages/* (NEW, UNWIRED): core/schemas | agent/planner | tools/base | models/adapter | geospatial/metadata | observability/logger
```

## Current Data Flow

Upload (1-4 files, ext-check only) -> `await upl.read()` unbounded -> rasterio (bands>3 dropped, radiometry stretched) or PIL fallback (georef lost) -> `meta{width,height,bands,format,modality,has_georef}` -> router classify -> validate (15%/25% dim gates) -> hardcoded `if tid==` dispatch -> tools return `{answer,confidence(heuristic),evidence,masks}` -> orchestrator fuses confidence (mean), builds trace list, VLM flags -> `clean()` strips masks, keeps b64 overlays -> JSON + report file.

## Component Inventory

| Component | File | Live? | Verdict |
|---|---|---|---|
| Router | `agent/router.py` | YES (`orchestrator:12`) | regex, `?` catch-all, live |
| Planner | `packages/agent/planner.py` | NO | typed duplicate, dead |
| Validation | `backend/app/services/validation.py` | YES | ext+dim only, no bytes cap |
| VQA | `tools/vqa_tool.py:745` | YES | heuristic primary, VLM almost never fires |
| Change | `tools/change_tool.py` | YES | LAB+Otsu CV, not Siamese |
| Optical | `tools/optical_tool.py` | YES | RGB proxy, not S2 physics |
| SAR | `tools/sar_tool.py` | YES | luminance proxy, not VV/VH |
| Fusion | `tools/fusion_tool.py` | YES | IoU/Dice arithmetic |
| Orchestrator | `backend/.../orchestrator.py:374` | YES | hardcoded dispatch, ad-hoc trace |
| Schemas (backend) | `backend/app/models/schemas.py` | NO (unused) | diverged duplicate |
| Schemas (new) | `packages/core/schemas.py` | NO | canonical, unused |
| ToolRegistry | `packages/tools/base.py` | NO | `FusionTool.execute` raises |
| ModelRegistry | `packages/models/adapter.py` | NO | tries missing blip/qwen adapters |
| Tiling | `packages/geospatial/metadata.py` | NO | zero callers |
| Logger | `packages/observability/logger.py` | NO | orchestrator uses print |
| API | `backend/app/api/routes.py` | YES | no response_model, cwd-relative reports |
| Old UI | `frontend/dist/index.html` | YES (:8000) | single-file, CDN-dependent |
| New UI | `apps/web/app/page.tsx` | SCAFFOLD | 1093-line monolith, any-typed |

## Dependency Graph

`main.py -> routes.py -> orchestrator.py -> {agent.router, validation, tools.*}`. `tools.*` depend only on numpy/cv2/PIL (+optional torch). `packages/*` depends on `tools.*` via fragile `from tools.X import` (requires `conftest.py sys.path` + `pytest.ini pythonpath=.`), creating reverse dependency new->old. Frontend has no direct model dep (good) but duplicates demo constants.

## Runtime / API / Frontend / Agent / Model / Tool / Geospatial / Reporting Flows

- Runtime: uvicorn sync-blocking; no worker/queue; singleton VLM global, no lock/eviction.
- API: `POST /api/analyze(query Form, images File 1..4)`, `GET /health|registry|reports[/{id}]`, `GET /data/demo/*`. 400 on bad ext/corrupt vs 200-refusal on pair mismatch (inconsistent but Demo 5 relies on latter).
- Frontend: old SPA globals `runDemo/analyze/downloadReport`; new Next.js 12 useState + 2 useEffect health-poll, raw fetch (react-query unused), zod unused, grounding overlay `%`-scaled (breaks on letterbox), before/after slider mis-wired.
- Agent: `classify_query -> decide_task -> validate_inputs -> plan_tools` regex; `decide_task` forces compound if temporal+sar hints; `plan_tools` fixed per task.
- Model: `_find_checkpoint` requires `adapter_config.json` with LoRA keys; 2/3 dirs manifest-only so skipped; only smoke qualifies but torch commented out + guard rejects garbage -> heuristic. Qwen branch unreachable.
- Tool: each `analyze()` returns heuristic dict; orchestrator calls directly, not via registry.
- Geospatial: read-then-discard (crs/transform str, no bounds/GSD/nodata/overviews/windowed-read); `_resize_match` stretches T2 to T1; pixel % reported as change.
- Reporting: full response + metas + b64 persisted; PDF only via `scripts/generate_pdf.py`, UI button is alert+print.

## Strengths / Weaknesses / Smells / Coupling

Strengths: offline demo, refusal correctness, VLM honesty flags, small CPU tests, static fix.
Weaknesses: no real VLM, fake grounding conf, no registration, RGB-as-SAR, monolith UI, empty api/, dead packages.
Smells: forked agent/schemas/validation (same magic numbers 15%/25% in 3 places), registry theater (3 registries, 0 used), fragile imports, modality-by-filename, georef-as-trace-text, `any` UI, duplicated DEMOS/queries.
Coupling: orchestrator knows every tool id; API knows report fs layout; UI knows backend port via rewrite; new packages know old tools paths.

## Bottlenecks / Risks

- Scalability: 4096² RGB ~48MB -> >400MB transient; 4 concurrent OOM; no tiling/overviews/queue/cache; cold VLM fetch blocks.
- Reliability: unbounded read, no timeout/retry/cancel, `traceback.print_exc` leaks, guessable report ids, CWD-relative writes.
- Security: CORS `*`+credentials, no auth/rate-limit, traversal in `get_report`/`generate_pdf` filename, polyglot ext-only check, no MAX_PIXELS, `trust_remote_code=True` + auto-load any adapter_config.
- Geospatial: Cartosat+RISAT same dims fused as co-registered; resize corrupts %; preview thumbnail breaks overlay scale; NDVI-proxy ≠ NDVI; luminance ≠ backscatter.
- ML: smoke loss 9.82, text-decoder-only, 8 samples; EuroSAT manifest copy-paste (labels BigEarthNet); reBEN 0 bytes; no precision/recall/IoU eval.
- UX: two ports, /api/docs 404, PDF theater, heuristic conf labeled but still numeric-impressive, 30-sec judge path buried in 9 nav tabs.

## Target Architecture

```
apps/web (split: workspace/query/trace/evidence/report components, typed api client)
apps/api (FastAPI thin: health/capabilities/models/tools/analyses/events/report, response_model=packages.core)
packages/core (single schemas source)
packages/agent (QueryPlanner + executor, deterministic)
packages/tools (SpecialistTool impls, independently testable)
packages/models (ModelAdapter + Registry, VLM gated by env)
packages/geospatial (extraction + tiling + compat, numeric transforms)
packages/observability (structured events -> trace)
packages/reporting (JSON+PDF builder, absolute paths, caps)
tests/ (unit/integration/e2e/failure)
```

Core forbids FastAPI/React/cv2-UI imports. Orchestrator depends on registries only.

## Migration Plan

1. Single-source schemas (`packages/core`), delete `backend/.../schemas.py`, add `response_model`.
2. Route orchestrator via `ToolRegistry`/`ModelRegistry`; delete `agent/` after parity test.
3. Gate VLM (`SATQUERY_ENABLE_VLM`), quarantine smoke from prod candidates.
4. Absolute `REPORTS_DIR`, id regex, bytes cap, auth note.
5. Split `page.tsx` (workspace, evidence, grounding, trace, report).
6. Wire tiling + scale-factor in preview contract.
7. Relabel grounding/conf as uncalibrated.

## Prioritized Actions (P0-P3)

- P0: bytes cap + traversal fix + report id validation; quarantine smoke; unify schemas import.
- P1: planner swap with regression on 5 demos; registry dispatch; frontend split; scale-factor fix.
- P2: tiling/windowed-read, CRS-compare numeric, co-registration refuse-or-warn, PDF endpoint.
- P3: STAC/catalog, queue/worker, GPU pool, multi-dev CODEOWNERS.
