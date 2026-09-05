# Production Architecture (target)

## Layout

```
apps/web/ (Next.js split) | apps/api/ (FastAPI thin)
packages/core/ (schemas, single source)
packages/agent/ (QueryPlanner + executor)
packages/tools/ (SpecialistTool impls)
packages/models/ (ModelAdapter + Registry)
packages/geospatial/ (extract + tiling + compat numeric)
packages/observability/ (events -> trace)
packages/reporting/ (JSON + PDF, absolute paths)
tests/ (unit/integration/e2e/failure)
```

## Ownership

- core: shared types, no deps. Forbidden: fastapi, react, cv2.
- agent: depends core only. Owns AnalysisPlan validation.
- tools: depend core + geospatial + observability. Each tool independently testable, `execute(images,query,params)->{answer,evidence,confidence,provenance}` never raises on valid input.
- models: depend core only. `load/unload/health/metadata/infer/device/capabilities`. Registry prefers non-heuristic if `SATQUERY_ENABLE_VLM=1` + checkpoint valid.
- geospatial: numeric CRS/transform/bounds/GSD/tiling. Only place with rasterio.
- observability: `request_id/analysis_id/timestamps/params/durations/events`. Trace derived from events.
- reporting: builds JSON + PDF from AnalysisResult, caps b64 size, absolute dir.
- apps/api: thin handlers, `response_model`, maps HTTP<->core, no domain math.
- apps/web: typed client (`lib/api.ts` + zod), components: Workspace/QueryComposer/TraceView/EvidenceView/GroundingView/ReportView.

No circular deps: api->agent->tools/models; tools -/-> api; core <- all.
