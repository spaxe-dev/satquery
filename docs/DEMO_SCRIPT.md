# Demo Script — SatQuery AI (5 minutes, SIH26167)

## Setup (before judges arrive)

- `python scripts/prepare_demo_data.py` (already done)
- `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000` — keep running
- Open `http://localhost:8000` on projector; also keep `http://localhost:8000/docs` ready
- Have `docs/PITCH.md` printed for reference

## Step 1 — Single Image (45s)

**SAY:** “SatQuery lets you ask satellite imagery a question. No GIS workflow, no model selection — just upload and ask.”

**DO:** Click **DEMO 1**. Upload `scene_1.png`. Query auto-fills: *“Describe the land-cover and major objects visible in this image.”*

**SHOW:** Answer (vegetation + water + texture), caption, confidence (heuristic, 0.82), execution trace: *Input validated → Query classified: captioning → SatQuery-RS-VLM selected → Executed* (9 steps), model used.

**SAY:** “One optical image, one question, one specialist tool — validated and traced.”

## Step 2 — Temporal Change (60s)

**SAY:** “Most operational questions need time, not a single frame. For example, flood response needs before and after.”

**DO:** Click **DEMO 2**. Flood_T1 + T2 load; query: *“What changed between these two dates, and where did the change occur?”*

**SHOW:** Change map (JET) + red mask overlay (22.37% changed — darkening, vegetation loss, consistent with inundation), description with brightness delta, trace: *ChangeDetector + VQA supplement*, confidence.

**SAY:** “This is not a VLM hallucinating a difference. It’s a real diff pipeline: LAB+gray, Otsu, morphology, connected components — with a spatial evidence map.”

## Step 3 — Optical + SAR (60s)

**SAY:** “Optical sees reflectance; SAR sees structure and works through cloud at night. Operationally you need both.”

**DO:** Click **DEMO 3**. Pair1 optical + SAR; query: *“Use the optical and SAR images together to identify built-up and water-covered regions.”*

**SHOW:** Optical water 4.2% (blue overlay), SAR water 2.7% (low backscatter), fused high-confidence water 2.6% with **IoU 0.61 / Dice 0.76** and statement: *“SAR corroborates optical water detection — robust to cloud.”* Built-up low agreement (expected physics difference). Confidence with cross-modal support.

**SAY:** “SAR corroboration is the whole point of ISRO’s RISAT. The fusion engine scores agreement, not just concatenation.”

## Step 4 — Hero (Compound) (90s) ★

**SAY:** “This is the core product — the agent composes multiple specialists from one compound question.”

**DO:** Click **DEMO 4 ★**. Flood_T1 + T2 + flood_sar load; query: *“What changed between the two dates, and can the SAR imagery corroborate the change?”*

**SHOW:** Trace: *Query classified: Temporal + Cross-Modal → Input validated → ChangeDetector → SARAnalyzer → Fusion → Confidence → Answer* (12 steps). Result: *“Temporal: 22.37% extensive change… SAR shows 10.3% water-like low-backscatter — CONSISTENT with inundation. SAR penetrates cloud.”* Agreement label.

**SAY:** “Upload → Question → Agent → Multiple specialists → Evidence → Confidence → Trace → Answer. That is the differentiation over ‘upload → VLM → answer’.”

## Step 5 — Invalid / Refusal (30s) (Trust Demo)

**SAY:** “Trust means refusing when evidence is insufficient.”

**DO:** Click **DEMO 5**. Mismatched 128×128 vs 512×512; query: *“What changed?”*

**SHOW:** **Analysis refused.** Reason: *“Images cannot be established as a valid bi-temporal pair: dimension mismatch.”* No hallucination. No change map. Confidence zeros. Trace shows `error` step.

**SAY:** “A confident hallucination in front of SAC judges would be fatal. Refusal is a feature.”

## Step 6 — Report (15s)

**DO:** Click **⬇ JSON REPORT** (downloads `report_<id>.json` with query, metas, task, tools, params, result, confidence, trace, timestamp). Mention PDF via `python scripts/generate_pdf.py`.

## Likely Judge Questions & Answers (30s each)

**Q: Why can’t a person just compare the images?**
A: For two obvious images they can. The operational problem is 1000 km², subtle change, multiple observations, different sensors, repeatable evidence, confidence, audit trail. SatQuery turns “what changed?” into a reproducible workflow.

**Q: Why do we need an agent / not just if-else?**
A: It *is* a deterministic router today (scored trace is observable). That’s a feature for judging — reproducible. The “agentic” value is automatic composition of 2-3 specialists from one natural question, with validation and fusion. Shown live in Demo 4.

**Q: Why can’t ChatGPT/Gemini do this?**
A: Generic VLMs are explicitly disqualified by the PS — no RS adaptation, no SAR physics, no change masks, no trust machinery. They hallucinate on RISAT without BigEarthNet adaptation. We have a real LO RA scaffold + real CV pipelines.

**Q: Why do we need SAR?**
A: Optical fails under cloud/at night — SAR (C-band, backscatter) sees structure through cloud (Demo 3/4: water is dark smooth in SAR, built-up bright rough). ISRO’s evaluation is Cartosat-2S + RISAT pairs.

**Q: Why multiple models?**
A: One model cannot do VQA, grounding, change masks, and SAR backscatter well. The registry lets the agent pick the right specialist — and the trace proves it.

**Q: How was your model trained?**
A: LoRA r=8 on Qwen2-VL-2B / BLIP, QLoRA 4-bit, lr 2e-4, synthetic RSVQA-style subset (200-500 verified via dry-run). Real adaptation target is BigEarthNet-MM. Checkpoint swappable in `tools/vqa_tool.py`. No fabricated scores — honest per `docs/MODEL_CARD.md`.

**Q: What if wrong?**
A: Confidence is heuristic and labeled as such; trace shows inputs and tools; evidence maps are visual; refusal on invalid inputs. All in the JSON report.

**Q: Confidence?**
A: Heuristic combination of input quality, model confidence, evidence agreement, cross-modal support — labeled heuristic, not calibrated probability. Calibrated version is stretch.

**Q: Invalid imagery?**
A: Demo 5 — refusal with reason, no hallucination.

