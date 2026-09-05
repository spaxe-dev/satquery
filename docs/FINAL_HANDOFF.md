# SatQuery AI — Final Engineering Handoff (Phase 2 appended 2026-09-05)

## Phase 2 summary (b7f8c87 → here, no training, inference-only)

- Eval harness `eval/runners/run_baseline.py`: 6 cases (5 REGRESSION demos + 1 QUALITATIVE grounding probe), no invented ground truth; baseline 6/6 passed heuristic (`eval/reports/`).
- Registry real: `packages/tools/dispatcher.py` + orchestrator routes VQA/change/optical/SAR via `registry_dispatch`; `tests/test_registry_dispatch.py` proves replacement without orchestrator edit (FAKE_OPTICAL_MARKER).
- Models tested: BLIP-base (RUNNABLE CPU, BSD-3, optical-only) INTEGRATED gated `SATQUERY_ENABLE_VLM=1` via `packages/models/blip_adapter.py` (grounding explicitly unsupported); Qwen2-VL-2B/GeoChat/EarthMind/TEOChat BLOCKED CPU (see `docs/MODEL_EVALUATION.md`). Fallback: REAL(optical VQA/cap only)→HEURISTIC; SAR/temporal/grounding never sent to RGB-only model as understanding (capability guard).
- Geo/raster: `validation.py` preserves numeric transform/bounds/nodata/GSD (PIL fills None), MAX_BANDS 12, HARD_MAX 64M px, warn >16M; `packages/geospatial/registration.py` returns unregistered/approximately_aligned/registered + aspect-preserved `box_to_preview` (no stretch); change `_resize_match` documented approximate (validation refuses >15%).
- Tiling: contract tested + wired as warn path (large → tiled/windowed required); full tile-execute-aggregate NOT wired (documented limitation).
- Registration: structured status only; pixel fusion on unregistered labeled approximate (no research-grade registration this phase).
- Manifests: eurosat `training_manifest.json` mislabel fixed locally to `eurosat-UNAVAILABLE-needs-regeneration` (file is gitignored under `models/checkpoints`, so fix is local-only, noted here); reBEN 0 bytes already honest UNAVAILABLE-by-note.
- Frontend: additive `ConfidencePanel/EvidencePanel/DemoSelector` + existing heuristic badges; `page.tsx` still monolith (PARTIAL); build PASS 22.1kB/109kB.
- Perf: eval `time_s` per case recorded; no Rust (no profiling justification); bottleneck is numpy/CV + full-raster loads (documented).
- Tests: 79 passed, 1 skipped (was 73/1); demos 5/5; eval 6/6; frontend build PASS.

## Executive Summary (Phase 1, retained)

## Executive Summary

Continued from checkpoint `9dc9c57` without restart. Verified prior summary, preserved uncommitted `packages/*` + `apps/web`, fixed P0 security + contracts, added regression tests, built frontend, verified 5 demos. System remains **heuristic prototype with honest scaffolding** — demoable, not production, no training done (per constraint, no GPU).

## What Was Already Working

- 5 demos offline (Demo4 hero 22.37% + SAR 10.3%, Demo5 refusal), VLM lifecycle flags, report JSON, 10 test files.

## What Was Audited

Full forensic via 3 parallel audits + manual verification: backend live path (`agent/router`, `orchestrator`, `tools/*`), models (manifest-only + smoke loss 9.82), frontend monolith (1093 lines), API gaps, geospatial read-then-discard, security (unbounded read, traversal, CORS*, trust_remote_code), scalability (OOM first).

## What Was Changed

- `packages/core/schemas.py`: +`model_config protected_namespaces` (silences pydantic warning).
- `backend/app/api/routes.py`: `MAX_BYTES 25MB`, absolute `REPORTS_DIR`, `REPORT_ID_RE`, no stack leak (generic 500 + server log).
- `tools/vqa_tool.py`: quarantine `smoke-*` unless `SATQUERY_ALLOW_SMOKE=1`.
- `packages/tools/base.py`: `FusionTool.execute` returns failure dict, no longer raises.
- `apps/api/main.py` (new thin re-export), `apps/web/lib/schemas.ts` (zod + DEMOS), `apps/web/components/TraceView.tsx` (HeuristicBadge + TraceView), `tests/test_contracts.py` (5 contract tests).
- Docs (14 new): ARCHITECTURE_AUDIT, BS_AUDIT, SIH_REQUIREMENTS_MATRIX, PRODUCTION_ARCHITECTURE, SCALABILITY_AUDIT, DATA_AND_INTEGRATION_ROADMAP, SECURITY_AND_RELIABILITY, OBSERVABILITY, UX_AUDIT, API/MODEL/TOOL/GEOSPATIAL_ARCHITECTURE, TESTING_STRATEGY, DEVELOPER_WORKFLOW, PRODUCTION_ROADMAP + this file.

## What Was Deleted / Replaced

Nothing deleted (preserve useful work). `FusionTool` raise replaced with total function. No architecture rewrite — wiring deferred to NEXT with tests.

## Current Architecture / Structure

`apps/web (Next.js, partially split) | apps/api (thin) | packages/core,agent,models,tools,geospatial,observability | backend/ (live FastAPI) | tools/ (live heuristics) | tests/`. Core forbids UI/API deps (documented, not yet enforced by import-lint).

## API / Frontend / Agent / Models / Tools / Geo / Fusion / Observability / Security / Testing

- API: legacy `POST /api/analyze` + `GET /health|registry|reports` live; target `/analyses` documented. P0 fixes in.
- Frontend: `npm run build` PASS (22.1kB route, 109kB first load) after SWC reinstall; `tsc --noEmit` clean except `.next/types/validator` upstream. `page.tsx` still monolith — new `schemas.ts`/`TraceView` are first split, full split NEXT.
- Agent: live `agent/router` regex; `packages/agent/planner` parity-tested but not swapped (needs orchestrator PR).
- Models: `ModelRegistry` prefers non-heuristic; prod returns heuristic (correct without GPU).
- Tools: `ToolRegistry` total (fusion fixed); orchestrator still hardcodes dispatch (NEXT).
- Geo: `TilingStrategy` tested, unwired; numeric CRS/GSD documented.
- Observability: `structured_log` exists, orchestrator still print (NEXT: wire events->trace).
- Security: bytes cap, id regex, filename note, VLM gate; remaining: auth/rate-limit, MAX_PIXELS, magic check.
- Testing: 73 passed, 1 skipped (`pytest -q` 16.7s).

## Demo Workflows

`python scripts/final_verification.py`: Demo1 caption 0.82, Demo2 22.37% 0.84, Demo3 IoU fusion 0.65, Demo4 compound 0.58, Demo5 REFUSED 0.0 — all OK.

## SIH Status

1 GREEN (trace), 10 YELLOW (all functional but heuristic/scaffold). Demo-ready YES; hidden Cartosat/RISAT eval HIGH RISK without real VLM + registration + S1/S2 physics.

## Genuinely Implemented / Heuristic / Scaffolded / NOT

- Implemented: validation gates, refusal, trace, reports, demo fixtures, P0 security, contracts importable.
- Heuristic: VQA/caption/grounding/change/optical/SAR/fusion/confidence (all labeled).
- Scaffolded: ModelRegistry real adapters, ToolRegistry dispatch, planner swap, tiling, PDF endpoint, STAC.
- NOT: training, GPU infer, detector, Siamese, CLIP, queue/DB/storage, auth.

## Requires GPU / External Data

GPU for any LoRA train/infer (Qwen2-VL-2B, BLIP LoRA); 24GB train, ~6GB infer. External: BigEarthNet 59GB, reBEN S1/S2 bytes (0 on disk now), VRSBench/RSVQA/CDVQA eval splits.

## Limitations / Debt / Biggest Weakness / Judge Risk

- Pixel-space % ≠ m²; resize-stretch; preview scale breaks overlays; EuroSAT manifest copy-paste; reBEN 0 bytes; two ports; /api/docs 404; `?` catch-all.
- Debt: forked agent/schemas/validation, fragile `from tools` imports, `any` UI, duplicated DEMOS.
- Weakness: no learned RS model.
- Judge risk: "Show fine-tune weights/metrics" — answer must be "dry-run + smoke quarantined, heuristic baseline, scaffold ready."

## Next 3 Actions

1. Swap orchestrator to `QueryPlanner` + `ToolRegistry` with 5-demo regression (P1).
2. Split `page.tsx` (Workspace/Evidence/Grounding/Report) + Demo4 default + scale-factor contract (P1).
3. Wire tiling + numeric CRS/GSD + eval harness + fix EuroSAT manifest (P2).

## How To Run / Tests / Demo / Add Model-Tool / Parallel

- API: `uvicorn backend.app.main:app --port 8000` ; Web: `cd apps/web; npm run dev` (rewrites to :8000).
- Tests: `python -m pytest -q` (73 passed, 1 skipped). Frontend: `node node_modules/typescript/bin/tsc --noEmit`, `npm run build`.
- Demo: `python scripts/final_verification.py`, or open :8000 + pick Demo4.
- Add model: implement `ModelAdapter`, `register()` in `get_model_registry`, set `SATQUERY_ENABLE_VLM=1`.
- Add tool: subclass `SpecialistTool`, add to `ToolRegistry`, add planner rule + test.
- Parallel: A web, B api, C agent/core, D tools/geo/models, E tests — via `packages/core` stable.

## Verification Results

- `pytest -q`: 73 passed, 1 skipped PASS.
- `test_contracts + test_agent + test_tools`: 20 passed PASS.
- `final_verification.py`: 5/5 PASS (see Demo section).
- Frontend `build`: PASS (22.1kB/109kB). `tsc`: PASS except upstream `.next` file.
- API: exercised via TestClient in `test_end_to_end` + `test_vlm_metadata` PASS.
- Security/geo/scale: static audit PASS, load test NOT TESTED, hidden-eval NOT TESTED.
