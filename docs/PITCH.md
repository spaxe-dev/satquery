# Pitch — SatQuery AI (SIH26167)

## 30-second

SatQuery is an agentic assistant that lets any user ask satellite imagery a question in natural language — “What changed and can SAR confirm it?” — and automatically runs the right specialist remote-sensing models, validates inputs, fuses evidence, and returns an auditable answer with maps and confidence.

## 1-minute

Today, extracting an answer from satellite imagery requires GIS expertise and multiple fragmented AI tools. SatQuery moves that complexity into a deterministic agent. You upload an optical image, a SAR image, or a bi-temporal pair, ask in plain English, and the agent selects from a registry of RS specialists — VQA, change detection, optical and SAR analyzers, and a fusion engine — validates inputs, runs them, scores agreement, and exposes the full execution trace. No hallucinated generic VLM; every result carries visual evidence and heuristic confidence. We demo it on flood inundation: before, after, and SAR corroboration — plus a live refusal on invalid inputs to prove trust.

## 2-minute

Include the 1-min plus:

- **Why RS adaptation:** Generic VLMs fail on RISAT/Sentinel without BigEarthNet fine-tuning (PS explicitly disqualifies them). We have a real LoRA/QLoRA pipeline (2B backbone, r=8) verified via dry-run and swappable — plus a heuristic floor that guarantees the demo on 4GB VRAM.
- **Why optical+SAR:** Optical sees spectral texture; SAR sees structure through cloud at night. ISRO’s evaluation is Cartosat-2S + RISAT pairs — exactly the pair we demo with IoU-scored corroboration.
- **Why temporal:** Disaster and land-use questions are inherently “what changed?” Our change detector is a real CV pipeline (LAB diff, Otsu, morphology) with a spatial evidence map, not a VLM guess.
- **Why agent:** The value is orchestration: `Question → Validation → Tool Selection → Execution → Fusion → Confidence → Trace → Answer`. That trace is what ISRO scores, and we show it composing change + SAR from one compound query.
- **Differentiation:** Live adversarial demo, cross-modal agreement flags, honest heuristic confidence, and an auditable trace — execution quality over model size.

## Problem Statement (in our words)

Remote-sensing AI is fragmented into task-specific systems that require non-experts to become GIS/model operators. SatQuery inverts this: the user expresses intent (“Where was inundated?”), SatQuery constructs the workflow, executes specialist models on the right imagery (single, bi-temporal, optical+SAR GeoTIFFs), and returns an evidence-grounded, auditable result.

## Why This Matters / Who Uses It

Direct users are institutional analysts (disaster management, urban planning, agriculture, forestry, water, infrastructure) who need *actionable* answers, not a visualization. Public benefit is downstream: faster, auditable decisions on floods, urban growth, forest loss. For ISRO/SAC, it’s a natural-language front-end over Cartosat/RISAT that matches their hidden evaluation set.

## Example Disaster Scenario

**Assam flood (representative):** Pre-flood Sentinel-2 / T1 + post-flood T2 (same area, different dates) + RISAT SAR. Query: “Which areas were inundated and what changed?” SatQuery validates the pair, runs ChangeDetector (22% inundation, darkening), analyzes SAR (10% low-backscatter water), fuses (IoU-scored agreement: SAR corroborates optical), and returns water masks, confidence, and a downloadable report. No fabricated affected-area numbers — we show percentages on the test patches and flag that real-world extents require verified imagery.

## Why Existing Tools Are Insufficient

GeoChat, TEOChat, SkyEyeGPT cover single-image VQA but lack multi-image temporal + SAR + agentic validation. Commercial GIS agents lack RS fine-tuning. A generic VLM wrapper is explicitly disqualified and would hallucinate on SAR.

## Why an Agent Is Necessary

A single model cannot be best at VQA, grounding, change masks, and SAR backscatter simultaneously. The agent is the product: it inspects query + image count + modality + metadata, selects the minimal specialist set, sequences them, and exposes the plan.

## Why Optical+SAR / Temporal / Evidence/Confidence Matter

- **Optical+SAR:** Complementary physics; single modality is unreliable under cloud.
- **Temporal:** Change is the operational question; single frames are insufficient.
- **Evidence/confidence:** Without maps and scores, a user cannot act; without refusal, a user cannot trust.

## What We Built

A working web app (FastAPI + agentic backend + 5 specialist tools + fusion) with a polished UI that makes the agentic process visible, 5 curated demos (including a live refusal), real CV pipelines, deterministic routing, and honest heuristic confidence. LoRA training is scaffolded and checkpoint-swappable; the demo runs offline on a 4GB laptop.

## Architecture Explanation

(See `docs/ARCHITECTURE.md` graph.) Upload → Validate → Agent classifies → Registry selects → Tools execute → Fusion scores agreement → Result engine assembles answer/visual/confidence/trace → Report.

## Demo Sequence

Demo 1 single → Demo 2 temporal (change map) → Demo 3 optical+SAR (agreement) → Demo 4 hero (compound) → Demo 5 invalid (refusal). Each step narrates the trace.

## Competitive Differentiation

Not largest model — most **auditable**: input validation, cross-modal contradiction flagging, heuristic-labeled confidence, refusal, and a downloadable report that reproduces the workflow.

## Limitations

Heuristic primary on 4GB; grounding is not detector-based; no live co-registration; domain shift to Cartosat/RISAT is untested without hidden data — disclosed in `docs/HANDOFF.md`.

## Potential Future Deployment

Single-GPU Docker Compose, then Postgres/MinIO, queue, tiling for sub-meter, EarthMind as registry option, ISRO Bhoonidhi imagery integration, calibrated confidence on held-out set.

