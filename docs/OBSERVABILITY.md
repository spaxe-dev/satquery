# Observability

Every analysis emits events -> derived trace (never hand-written UI strings).

Event: `{request_id, analysis_id, ts, stage, tool/model, params, duration_ms, status, warnings, errors, evidence_refs, confidence}`.
Stages: input_received, metadata_inspected, query_classified, compatibility_validated, tool_selected, tool_executed, fused, answered, reported.
`packages/observability/logger.structured_log` is sink; orchestrator must call it (currently print). Frontend `TraceView` renders events only.
Never log image bytes; log `width,height,modality,has_georef,report_id` only. VLM flags (`vlm_attempted/loaded/generation_valid/fallback_reason`) are first-class fields.
