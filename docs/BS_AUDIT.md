# BS / Hype Audit (evidence-backed, 2026-09-05)

Rule: heuristic = classical CV + templates, no learned RS weights in prod path. Scaffold = interface/registry without wired impl.

## 1. "AI / VLM / RS-adapted / fine-tuned"

- CLAIM: RS-adapted VLM primary (settings.yaml `Qwen2-VL-2B primary`, registry `+CLIP zero-shot`).
- IMPL: `tools/vqa_tool.py:607-745 analyze()` heuristic primary; `_find_checkpoint:411-442` skips 2/3 dirs (manifest-only); `backend/requirements.txt:18-23` torch commented out -> ImportError -> heuristic; smoke adapter 8 samples loss 9.82 rejected by `_is_invalid_generation:292-387`.
- EVIDENCE: `models/checkpoints/*/training_manifest.json (dry_run_verified)`, `smoke_manifest.json loss 9.82`, no `satquery-*-qwen-*` dir, no CLIP import.
- REALITY: No RS fine-tune loaded in prod. Scaffold only.
- RISK: Judge asks for weights/metrics -> fail.
- CAN SAY: "Heuristic baseline with VLM integration scaffold, dry-run manifests, 1 smoke adapter (quarantined)."
- MUST NOT SAY: "Fine-tuned", "RS-adapted VLM", "Qwen-powered".
- MAKE TRUE: Train LoRA on BigEarthNet (GPU), eval on VRSBench/RSVQA, publish metrics + checkpoint hash.

## 2. "Agentic"

- CLAIM: Advanced agentic orchestration.
- IMPL: `agent/router.py` regex `PATTERNS` incl. `?` catch-all -> `single_vqa`; `plan_tools` fixed per task; orchestrator hardcoded `if tid==`.
- REALITY: Deterministic regex router, no planning/search/memory. Honest name: deterministic task router.
- CAN SAY: "Deterministic query router + validated plan + auditable trace."
- MUST NOT SAY: "Autonomous agent", "LLM planner", "self-improving".
- MAKE TRUE: `QueryPlanner` emits validated `AnalysisPlan` (already scaffolded), registry dispatch, policy validation, replanning on tool failure.

## 3. "Multimodal / cross-modal reasoning / fusion"

- CLAIM: Learned optical-SAR fusion.
- IMPL: `fusion_tool.py:8-53` IoU/Dice on threshold masks; `optical_tool` RGB proxy, `sar_tool` luminance proxy.
- REALITY: Decision arithmetic on weak masks. `mean_iou 0.05-0.61` reflects mask overlap, not reasoning.
- CAN SAY: "Decision-level mask agreement (IoU/Dice) as corroboration heuristic."
- MUST NOT SAY: "Cross-modal reasoning", "joint embedding", "learned fusion".

## 4. "Grounding"

- CLAIM: Text-guided region grounding with confidences 0.72/0.65.
- IMPL: `vqa_tool.py:20-95` HSV/Canny/contours + central fallback `conf 0.42`; VLM-success path still uses heuristic boxes `:663-665`.
- REALITY: 100% heuristic, uncalibrated, no IoU eval. `HANDOFF.md:21` admits.
- CAN SAY: "Heuristic region proposals (uncalibrated)."
- MUST NOT SAY: "Grounded", "detected with X% confidence".

## 5. "Change detection / change map"

- CLAIM: Siamese + learned head (`tool_registry.yaml:19-20`).
- IMPL: `change_tool.py:57-240` LAB+gray absdiff + Otsu + morphology + CC 0.1%.
- REALITY: Classical CV, no registration, no radiometric norm, resize-stretch up to 15%.
- CAN SAY: "Classical bi-temporal differencing (requires pre-registered pair)."
- MUST NOT SAY: "Siamese", "learned change", "geospatially accurate m²".

## 6. "Confidence"

- CLAIM: Numeric overall 0-1.
- IMPL: Means like `0.5*input+0.5*model`, `0.35+0.55*iou` — invented weights, always `is_heuristic True`.
- REALITY: Uncalibrated quality score, not probability.
- CAN SAY: "Heuristic quality score (not calibrated)."
- MUST NOT SAY: "Model confidence", "87% confident".

## 7. "Production-ready / scalable / real-time"

- CLAIM: Implied by dashboard polish.
- IMPL: No Dockerfile, no queue, sync-blocking, unbounded reads, cwd-relative reports, CORS *, no auth.
- REALITY: Single-user prototype. 4096² -> OOM risk; concurrent users serialize.
- CAN SAY: "Local prototype, single-user."
- MUST NOT SAY: "Production-ready", "scalable", "real-time".

## 8. "Evidence-grounded / hallucination-free"

- PARTIAL PASS: refusal path correct, VLM flags honest, masks shown. But heuristic captions (ExG %, NDVI-proxy) read as authoritative; fusion "CONSISTENT" text overstates low IoU.
- CAN SAY: "Shows masks + trace + heuristic flags."
- MUST NOT SAY: "Hallucination-free". FIX: prefix heuristic answers, down-weight corroboration language when IoU<0.25.

## Summary table

| Claim | Status | Fix |
|---|---|---|
| RS fine-tune | SCAFFOLD | quarantine smoke, train on GPU later |
| Agentic | HEURISTIC router | rename, use AnalysisPlan |
| Fusion | HEURISTIC arithmetic | relabel |
| Grounding | HEURISTIC | relabel + add detector later |
| Change | HEURISTIC CV | require registration note |
| Confidence | HEURISTIC | label uncalibrated everywhere |
| Scalable/prod | MISSING | local-prototype label |
