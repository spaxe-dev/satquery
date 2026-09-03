# SIH26167 — SatQuery AI (ISRO) — Deep Research Analysis

## 1. Executive Summary

SatQuery AI asks teams to build an agentic, query-driven assistant that answers natural-language questions about remote-sensing imagery: single optical/SAR images, bi-temporal pairs (change), and co-registered optical–SAR pairs. It is one of the most technically demanding SIH software problems — essentially a mini research project in multimodal remote-sensing AI — but it is also unusually well-specified: the datasets, benchmarks, mandatory tasks, and evaluation method are all named. A strong open-source ecosystem (GeoChat, TEOChat, CDVQA/VisTA, BigEarthNet-MM) makes an MVP buildable by a good team in 6–10 weeks of serious work. Verdict up front: **PURSUE** — high winning potential for a team with real CV/ML capability; a trap for teams planning to "wrap GPT-4V," which is explicitly disqualified.

---

## 2. Problem Interpretation

**What ISRO is asking for:** A web/GUI application where a user uploads GeoTIFF/TIFF images (single optical/multispectral, single SAR, bi-temporal pair, or optical+SAR pair) plus a natural-language question, and gets an evidence-grounded answer (text + visual overlays + confidence + auditable execution trace). An agentic controller must route the query to specialist models from a registry.

**Underlying real-world problem:** Remote-sensing analysis today requires GIS expertise and task-specific models. Non-experts (district disaster officers, agriculture officers, planners) can't ask "what changed here since last year?" and get a reliable answer. Also, single images are often insufficient — cloud cover requires SAR, change requires multitemporal data.

**Users/stakeholders:** ISRO/SAC (evaluator and likely eventual user), NRSC downstream users, state disaster-management and agriculture departments, urban planners, forest departments, students/researchers.

**Official mandatory requirements (explicit):**
1. At least one component fine-tuned/adapted on BigEarthNet.txt (or other open RS data) — generic VLMs explicitly disqualified.
2. Single-image VQA (mandatory) + either captioning OR text-guided grounding.
3. Bi-temporal change description / change-VQA (mandatory); optional change map.
4. Optical–SAR paired-image joint analysis.
5. Agentic orchestration: task classification, input validation (number/modality/format/metadata), model selection from registry, execution, output fusion, confidence, visual evidence, auditable execution summary.
6. Interactive GUI/web app; downloadable reports.

**Success:** Competitive normalized scores on prescribed public benchmark test subsets (VRSBench, RSVQA, CDVQA test splits) plus an undisclosed ISRO/SAC evaluation set of pre-georeferenced Cartosat-2S optical + RISAT-1 SAR pairs.

**My recommendations (not official):** prioritize the optical–SAR cross-modal tool early — it's the least commoditized task and the SAC evaluation set is exactly Cartosat+RISAT pairs; treat the execution trace UI as a first-class product feature, not an afterthought, since only observable traces are evaluated.

---

## 3. Domain Research

| Source | Why it matters |
|---|---|
| **BigEarthNet-MM / reBEN v2.0** (bigearth.net) | The mandated adaptation dataset: 549k–590k Sentinel-1(SAR)+Sentinel-2(optical) patch pairs, GeoTIFF, CDLA-Permissive 1.0 license. Perfectly matches the optical–SAR requirement; v2 adds pixel-level reference maps. |
| **VRSBench** (NeurIPS 2024 D&B; vrsbench.github.io, HF: xiang709/VRSBench) | Named eval benchmark: 29,614 images, human-verified captions, 52k object refs, 123k QA pairs — covers captioning, grounding, VQA. |
| **RSVQA** (rsvqa.sylvainlobry.com; Lobry et al., TGRS 2020/IGARSS 2021) | Named eval benchmark; LR/HRBEN variants built on BigEarthNet with OSM-derived QA. Directly reusable for training too. |
| **CDVQA** (Yuan et al.; SECOND-based, ~122k QA over 2,968 bi-temporal pairs) | Named eval benchmark for change-VQA. Question types: existence, direction, increase/decrease, largest/smallest change, ratios. |
| **GeoChat** (CVPR 2024, github.com/mbzuai-oryx/GeoChat) | First grounded RS VLM (LLaVA-1.5 + LoRA, 318k instructions). Best ready-made single-image backbone: captioning, VQA, grounding out of the box. Apache-style open weights. |
| **TEOChat** (arXiv 2410.06234) | First temporal EO VLM (Video-LLaVA adapted; TEOChatlas dataset). Closest existing work to the change-analysis mandate — both a foundation and a competitor to differentiate from. |
| **SkyEyeGPT** (ISPRS 2025) | Unified RS VLM via instruction tuning (968k samples); single-image focus. |
| **VisTA / QAG-360K / CDQAG** (arXiv 2410.23828) | SOTA CDVQA baseline (~73% OA) that also grounds changes with masks — the exact "answer + evidence" pattern the PS wants. |
| **Recent Qwen3-VL/Qwen3.5 LoRA studies on CDVQA** (2026 letters) | Show Qwen3.5-2B + LoRA hits ~74% OA — proves parameter-efficient fine-tuning of small modern VLMs is the realistic student path. |
| **SEN1-2** (arXiv 1807.01569) | 282k SAR–optical pair dataset for cross-modal learning; supplementary to BigEarthNet-MM. |
| **STSF-Net / Delta-SN6, MM-OVSeg, OptSARChangeDetection (GitHub)** | Optical–SAR fusion/change literature and open code; shows pseudo-change suppression is the hard part of cross-modal analysis. |
| **Agentic AI for Remote Sensing survey** (arXiv 2604.24919) | Frames exactly this agentic-workflow paradigm; useful for architecture citations and identifying research gaps (geospatial consistency validation between tools). |

---

## 4. Existing Solutions and Competition

1. **GeoChat (MBZUAI)** — Grounded RS VLM; captioning/VQA/grounding/classification. Strengths: mature, open weights, handles single-image tasks well. Weaknesses: no multi-image/temporal input, no SAR understanding, no agentic routing. *Differentiation:* SatQuery must add multi-input orchestration + SAR + change.
2. **TEOChat (UMD)** — Temporal EO VLM; change description/damage assessment. Strengths: closest to the change mandate. Weaknesses: optical-only, no SAR, no tool registry/agentic layer, not evaluated on CDVQA-style closed-vocab tasks. *Differentiation:* add SAR modality + agent framework.
3. **SkyEyeGPT / RSGPT / LHRS-Bot / EarthGPT** — single-image unified RS MLLMs. Same gap: no pairs, no agency.
4. **VisTA + QAG-360K** — specialist change-VQA-with-grounding model. Strengths: SOTA on CDVQA. Weaknesses: one task only, no natural interface. *Differentiation:* use its ideas inside our tool chain rather than compete head-on.
5. **General geospatial agents (e.g., ChatGeo/AutoGIS-style LLM-GIS agents, commercial: Google Earth AI, SkyFi)** — route to GIS APIs/models but don't fine-tune VLMs on pixel-level RS tasks and aren't built around paired optical–SAR reasoning. *Differentiation:* pixel-level domain-adapted models + auditable trace.
6. **UniChange (CVPR 2026)** — MLLM unifying binary+semantic change detection with mask output. Strong recent overlap with the change-map option; leverage as reference implementation rather than novelty claim.

Key insight: **no existing system combines all four mandated capabilities under an auditable agentic controller** — that combination is the defensible contribution. But each individual piece has strong prior art; judges at ISRO will know it. Novelty must be honest: integration + cross-modal (SAR) extension, not invented algorithms.

---

## 5. Proposed Solution

**Core product:** Web app ("SatQuery Studio") — upload panel → compatibility checker → chat interface → answer canvas with image overlays → execution-trace sidebar → downloadable PDF/JSON report.

**Major features:**
- Input validator: file type (GeoTIFF/TIFF/PNG/JPEG), CRS/geotransform read via GDAL/rasterio, band count, resolution match, co-registration check (SSIM/correlation threshold on downsampled patches), modality heuristic (SAR detection: single-band, non-natural statistics).
- Tool registry (each tool = typed schema: required inputs, modalities, params, outputs):
  - `vqa-single` — GeoChat or Qwen3-VL LoRA-tuned on RSVQA-LRBEN
  - `caption` / `grounding` — same backbone with task tokens
  - `change-vqa` — bi-temporal VLM (Qwen3.x-2B + LoRA trained toward CDVQA format)
  - `change-map` — lightweight Siamese semantic change detector (e.g., BIT/Changer on SECOND) for mask overlay when masks feasible
  - `optsar-fusion` — dual encoder (optical ViT + SAR CNN) with cross-attention fusion head trained on BigEarthNet-MM labels; outputs per-class land-cover/water/built-up evidence maps + textual summary
- Agentic controller (deterministic-first): rule-based intent classifier (regex/embedding classifier over query templates) → input-config dispatch table (1 img / pair / opt-SAR pair) → tool selection with fallback to LLM planner for ambiguous queries. This matters: deterministic routing is debuggable, reproducible for judging, and avoids LLM-planner flakiness.
- Evidence assembly: answer text + bounding boxes/masks rendered on the uploaded image + confidence (tool-specific calibration or verbalized confidence) + JSON execution trace (task, tools, versions, params, timings).
- Report export: PDF + machine-readable JSON trace.

**Architecture:** React/Next.js frontend ↔ FastAPI backend ↔ model service(s) (GPU worker running PyTorch; tools as Python modules behind an internal registry API) ↔ Postgres (queries, traces, results) + MinIO/S3 for imagery. Queue (Celery/RQ or simple task queue) for GPU jobs. All self-hosted/open-weight so nothing leaves the deployment (important for ISRO context).

**Deployment:** single-GPU (24GB — RTX 4090/A5000 class) Docker Compose stack; quantized (4-bit) 2–7B VLMs for inference. No external LLM API dependency at demo time.

**Offline/low network:** fully offline inference by design (local models); only downloads are initial weights.

---

## 6. Data & APIs

| Dataset | Role | Access reality |
|---|---|---|
| BigEarthNet-MM / reBEN v2.0 | Mandated fine-tuning; optical–SAR fusion training | Free download, ~90GB (subsettable), CDLA-Permissive. Realistic. |
| VRSBench | Eval: captioning/grounding/VQA | HF, free. Realistic. |
| RSVQA LR/HRBEN | Eval + training augmentation | Free. Realistic. |
| CDVQA | Eval: change-VQA | Available via paper authors/GitHub; ~modest size. Verify access early — historically request-gated. |
| SEN1-2, SECOND, xBD/fMoW | Optional extra training for change/SAR | Free. |
| Bhuvan (NRSC)/Bhoonidhi (ISRO) portals | Demo imagery of Indian scenes (LISS, RISAT-derived) | Public registration required; licensing restricts redistribution — use only for demo inputs. |
| Sentinel-1/2 via Copernicus Data Space + OData API | Fresh real-world demo scenes (e.g., flood event pre/post) | Free, scriptable — best source for live demo material. |

Concerns: CDVQA access should be confirmed before committing; the hidden SAC Cartosat/RISAT set cannot be trained against — generalization from Sentinel-resolution (10m) to Cartosat sub-meter is a genuine domain-shift risk and the single biggest unknown in scoring.

---

## 7. Buildability (brutally realistic)

- **Buildable by a student team (4–6 people, ML-capable):** LoRA fine-tuning of open 2–7B VLMs on subsets of the named datasets; a deterministic agent router; FastAPI+React app; rasterio validation; report generation. All demonstrated offline.
- **Demonstrable at SIH:** everything above on prepared scenes + 2–3 live Copernicus downloads.
- **Not buildable:** training a frontier-scale RS foundation model; reliable performance on unseen RISAT/Cartosat characteristics without sample data; robust speckle-noise-robust SAR language grounding at scale.
- **Simulatable:** the SAC-style evaluation (use held-out Sentinel splits as proxy "hidden set"); confidence calibration.
- **Hardest component:** the optical–SAR joint analysis with language output — almost no public paired data with text annotations exists; you'll synthesize QA pairs from BigEarthNet-MM labels (template-generated instructions), which is legitimate but weak supervision.
- **Failure modes:** (a) underestimating GPU/compute needs — LoRA runs need a rented A100/4090 for days; budget ₹30–60k or college cluster time; (b) trying to fine-tune too many components instead of adapting one backbone well; (c) brittle LLM-planner agent producing inconsistent traces during judging; (d) discovering CDVQA access issues in week 8.

---

## 8. MVP

**Essential:** upload + validation; single-image VQA + captioning (GeoChat/Qwen LoRA on RSVQA-LRBEN subset); change description/VQA (LoRA toward CDVQA); optical–SAR joint land-cover/water/built-up answer (fusion classifier + template/textual summarizer); deterministic agent router with visible execution trace; overlay rendering; JSON/PDF report.
**Optional:** grounding boxes, spatial change map, live scene fetch from Copernicus, multi-turn chat, confidence bars.
**Demo flow:** (1) single Sentinel-2 tile + "Describe the land-cover…"; (2) same tile + "Highlight the water body"; (3) bi-temporal pair + "What changed and where? Has built-up increased?"; (4) optical+SAR pair + "Use both images to identify built-up and water regions"; show trace panel each time; end with downloaded report.
**Infra:** 1×24GB GPU (cloud rental or institute), Docker Compose, ~200GB disk for dataset subsets.
**Hardware:** none beyond GPU.

---

## 9. Differentiators

1. **Optical–SAR joint language reasoning** — very few public systems do it; PS makes it mandatory; most competitors will bolt on a classifier without language integration. Feasible via BigEarthNet-MM synthetic instructions.
2. **Auditable execution trace as a first-class UI element** — directly matches what ISRO says it evaluates; most student teams will hide the plumbing. Cheap to build, high judge signal.
3. **Deterministic-hybrid agent router with input-compatibility gate** (co-registration/modality checks before any model runs) — addresses the silent-error problem highlighted in the agentic-EO literature; credible engineering vs pure-LLM planners.
4. **Evidence-grounded change answers** (text + mask overlay from an auxiliary change detector) — follows CDQAG's answer-plus-proof pattern; rare in student builds.
5. **Fully offline open-weight stack** — sovereignty-friendly for ISRO; no GPT dependency.
6. **Domain-shift story**: report per-tool applicability/confidence when input statistics deviate from training distribution (e.g., SAR resolution far below Sentinel norms) — honest handling judges will notice.

---

## 10. Winning Potential (skeptical view)

- Technical depth: very high — arguably the deepest software PS in the space track.
- Novelty: moderate — every component exists; the integration and SAR-language part is new-ish.
- Judge appeal: high with ISRO specifically (their own problem, their own sensors in the eval set).
- Demo potential: excellent if the pipeline is stable; terrible if models hallucinate live.
- Reasons NOT to choose it: (a) extremely heavy ML workload — teams without prior VLM fine-tuning experience will produce a broken wrapper and score near zero because "generic VLM" is explicitly disqualified; (b) scoring depends partly on a hidden SAC dataset you cannot prepare for; (c) compute cost is nontrivial; (d) crowded field — this PS will attract the strongest ML teams, so relative competition is fierce; (e) less "social impact narrative" than disaster/agriculture PSes (though you can frame flood monitoring demos).

## 11. Scoring

| Category | Score |
|---|---|
| Problem importance | 9 |
| Novelty potential | 6 |
| Technical depth | 10 |
| Data availability | 9 |
| Feasibility (for an ML-capable team) | 5 |
| Demo potential | 8 |
| Social impact | 7 |
| Differentiation | 7 |
| Scalability | 7 |
| Winning potential | 7 |
| **Overall (avg)** | **7.5 / 10** |

## 12. Final Recommendation: **PURSUE**

Strong pursue for a team that already has PyTorch/VLM fine-tuning experience and GPU access (effectively 8+/10 for them); WEAK for a team that doesn't — there is no shallow version of this problem that satisfies the explicit anti-generic-VLM clause. The specification quality, named benchmarks, and open ecosystem make planning easy; the depth requirement and hidden SAC eval set make it unforgiving.

## 13. Sources

- bigearth.net (BigEarthNet-MM/reBEN v2.0 descriptions, license)
- vrsbench.github.io / arXiv:2406.12384 / huggingface.co/datasets/xiang709/VRSBench
- rsvqa.sylvainlobry.com
- CDVQA (Yuan et al.) via emergentmind.com/topics/change-detection-visual-question-answering-cdvqa; arXiv:2508.08974 (BrightVQA/domain-generalized CDVQA)
- arXiv:2410.23828 (VisTA, QAG-360K, CDQAG)
- Recent Qwen3 LoRA CDVQA study: arxiv.org/pdf/2604.18429.pdf; themoonlight.io review
- github.com/mbzuai-oryx/GeoChat; arXiv:2311.15826
- arXiv:2410.06234 (TEOChat)
- arXiv:2401.09712 (SkyEyeGPT)
- arXiv:1807.01569 (SEN1-2)
- arXiv:2604.05527 (STSF-Net/Delta-SN6 optical–SAR change)
- github.com/sdubail/OptSARChangeDetection
- UniChange (CVPR 2026) via papernotes.org; github.com/NKU-HLT/UniChange
- arXiv:2604.24919 (Agentic AI for Remote Sensing survey)
- sih.gov.in/sih2026PS#ViewProblemStatement26167 (official PS)