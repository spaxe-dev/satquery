# SIH26167 Requirements Matrix (2026-09-05)

Source: `docs/problem_statement.md`. Statuses: GREEN=implemented, YELLOW=partial/scaffold, RED=missing, GRAY=irrelevant.

## Mandatory functional scope

| Requirement | Implementation | Evidence | Status | Limits / Full compliance | Demo | Hidden-eval risk |
|---|---|---|---|---|---|---|
| RS adaptation (BigEarthNet fine-tune ≥1 component) | Dry-run manifests + smoke 8-sample adapter (quarantined) | `models/checkpoints/*/training_manifest.json`, `smoke_manifest.json loss 9.82`, `training/train_lora_real.py` | YELLOW | Need GPU train + eval metrics + checkpoint; EuroSAT manifest copy-paste must fix | N/A | HIGH — judge checks weights/metrics |
| Single-image VQA (mandatory) | Heuristic VQA + VLM scaffold + metadata | `tools/vqa_tool.py:607-745`, `orchestrator.py`, `tests/test_end_to_end.py` | YELLOW | Works demo, heuristic not learned; needs real VQA eval (RSVQA) | Demo1 PASS (heuristic) | HIGH — hidden VQA needs real model |
| + captioning OR grounding (1 more) | Both present heuristically | `vqa_tool._heuristic_caption/_heuristic_grounding:20-95`, Demo1 caption | YELLOW | Caption deterministic fingerprint; grounding uncalibrated, no IoU eval | Demo1 PASS | MEDIUM — caption ok, grounding fails IoU |
| Multi-image change (description or change-VQA + optional map) | Classical change + map + description | `tools/change_tool.py`, `orchestrator temporal_change`, Demo2/4 22.37% | YELLOW | No registration/reproj; pixel % only; illumination false positives | Demo2/4 PASS (registered demos) | HIGH — Cartosat/RISAT unregistered fails |
| Cross-modal optical-SAR pair | Optical+SAR+IoU fusion | `optical_tool/sar_tool/fusion_tool`, Demo3 IoU 0.61 | YELLOW | RGB/luminance proxies, not S1/S2 physics; warn-only modality | Demo3 PASS (demo PNGs) | HIGH — real S1/S2 dB/B04-B08 fails |
| Agentic orchestration (interpret/classify/validate/select/configure/fuse/conf/trace) | Regex classify + validate + hardcoded dispatch + trace | `agent/router.py`, `orchestrator.py:21-374`, trace in response | YELLOW | No registry dispatch, `?` catch-all, no replanning; trace present (good) | All demos trace 9-12 steps | MEDIUM — trace satisfies observable part |

## Input scope

| Single optical/SAR GeoTIFF/TIFF + PNG/JPG for benchmarks | Ext-check + rasterio/PIL, modality-by-filename/bands | `validation.py:45-178` | YELLOW | No benchmark-format gate, no GSD/nodata, bands>3 dropped | PASS | MEDIUM |
| Cross-modal co-registered pair | Dim-check 25% warn-only, CRS warn-only | `validation.validate_optical_sar_pair` | YELLOW | No co-registration proof, no reproj | PASS demo | HIGH |
| Bi-temporal spatially corresponding | Dim 15% refuse, CRS warn | `validate_temporal_pair` | YELLOW | Same-area not verified | PASS demo | HIGH |

## System scope

| GUI + backend + specialist tools + controller + evidence/conf/trace/reports | FastAPI + 2 UIs + 5 tools + router + overlays + reports JSON | `routes.py`, `page.tsx`, `dist/index.html` | YELLOW | PDF manual, `apps/api` empty, page monolith | PASS | LOW for demo, HIGH for prod claims |
| Auditable summary (task/models/tools/params) | trace + tools_used + vlm flags | `orchestrator trace`, `routes 111-159` | GREEN | Good — keep | PASS | LOW |

## Representative queries

| Describe land-cover | captioning | GREEN (heuristic) |
| Highlight water body | grounding heuristic | YELLOW |
| What changed + where | temporal_change + map | YELLOW |
| Optical+SAR built/water | fusion | YELLOW |
| Built-up increased? | temporal + VQA context | YELLOW |

Overall: 1 GREEN, 10 YELLOW, 0 RED (nothing entirely absent, nothing fully compliant). Demo-ready YES; hidden ISRO/SAC co-registered eval AT RISK without real VLM + registration + S1/S2 physics.
