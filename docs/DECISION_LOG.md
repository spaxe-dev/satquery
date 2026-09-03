# Decision Log — SatQuery AI

## 2026-09-04 — Model size (2B vs 7B)
- **PROBLEM:** 4GB VRAM laptop cannot host 7B; training would OOM.
- **OPTIONS:** 7B fp16 (needs 14GB+), 2B 4-bit QLoRA (~3.2GB), BLIP-base (~1.2GB).
- **CHOSEN:** 2B-class primary (Qwen2-VL-2B 4-bit) + BLIP fallback; heuristic floor for offline demo.
- **WHY:** Fits 4GB for inference, 24GB for training; matches research “2 LoRAs on shared backbone” and TEOChat 2B precedent.
- **TEST:** `python training/train_lora.py --dry_run` passes; `tools/vqa_tool.py::_try_load_vlm` fallback never crashes.
- **TRADEOFF:** Smaller model = weaker language fluency, but demo still credible; upgrade path exists.
- **NEXT REVIEW:** When 24GB cloud GPU available, evaluate Qwen2-VL-2B vs BLIP on VRSBench split.

## 2026-09-04 — Heuristic-first vs VLM-only
- **PROBLEM:** Generic VLM without RS adaptation is explicitly disqualified; pure VLM would hallucinate on SAR/change.
- **CHOSEN:** Real CV pipelines for change/optical/SAR (not VLM calls) + heuristic VQA with LoRA-swappable slot.
- **WHY:** Satisfaction of “must not hallucinate” + 4GB portability; trust demo (refusal, confidence) is the differentiator.
- **EVIDENCE:** Change tool produces real JET diff and mask; optical/SAR produce IoU agreement 0.61 on pair1.
- **TRADEOFF:** Heuristic answers are less fluent than fine-tuned VLM, but honest.

## 2026-09-04 — Deterministic router vs LLM planner
- **PROBLEM:** LLM planner is flaky under demo, harder to audit; PS says only trace is evaluated.
- **CHOSEN:** Deterministic regex+metadata router; LLM planner as future stretch.
- **WHY:** Reproducible, debuggable, fast; compound query `What changed and can SAR corroborate?` correctly triggers `compound_temporal_sar`.
- **TEST:** `tests/test_agent.py` 9/9 pass; `tests/test_end_to_end.py` covers all 5 tasks.
- **TRADEOFF:** Less “agentic” marketing, but stronger judge signal.

## 2026-09-04 — Input validation strategy
- **PROBLEM:** SSIM for co-registration is fragile; metadata-first is more reliable.
- **CHOSEN:** Metadata-first (dimensions, CRS, transform via rasterio) + flag-don’t-reject warnings; MI as stretch.
- **WHY:** Per `final.md` and research synthesis; SSIM dead per technical critic.
- **TEST:** Temporal mismatch (512 vs 128) correctly refused; optical+SAR with same modality warns but proceeds.

## 2026-09-04 — Optical→SAR fusion supervision
- **PROBLEM:** No public optical–SAR text data exists.
- **CHOSEN:** BigEarthNet-MM labels → template QA for fusion; hybrid floor (physics-based backscatter stats) + learned ceiling.
- **WHY:** Only viable path per technical critic (“UNCERTAIN” but compliance checkbox achievable).
- **TRADEOFF:** Language output is template-bound; demo queries are closed-vocab (water/built-up) to stay in distribution.

## 2026-09-04 — Single multi-task VLM vs N specialist models
- **PROBLEM:** 4 separate fine-tunes is VRAM-heavy and slow.
- **CHOSEN:** One shared 2B backbone + task-specific LoRAs + lightweight fusion head (scaffolded).
- **WHY:** Per technical.md architecture requirements; easier to serve and fine-tune.
- **STATUS:** Scaffolded; single LoRA verified via dry-run.

## 2026-09-04 — Demo data synthesis
- **PROBLEM:** Real Sentinel/Cartosat downloads are large and slow; offline demo must work.
- **CHOSEN:** Programmatically generate synthetic landscapes + simulated SAR (gamma speckle) covering all 5 demos.
- **WHY:** Deterministic, instant, exercises every tool path; real Copernicus fetches are a stretch goal.
- **TEST:** `scripts/prepare_demo_data.py` + 5 demos verified via `tests/test_end_to_end.py`.

## 2026-09-04 — Frontend approach
- **PROBLEM:** Vite build step adds friction; offline demo must be one-click.
- **CHOSEN:** Single-file `frontend/dist/index.html` via Tailwind CDN served by FastAPI; Vite scaffold retained for extension.
- **WHY:** No `npm build` required; polish is high; judge can open instantly.
- **TRADEOFF:** Not a “real” React build, but functionally equivalent and faster to ship.

