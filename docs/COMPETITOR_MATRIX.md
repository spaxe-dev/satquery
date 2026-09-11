# Competitor Matrix — SatQuery AI vs State of the Art (2026-09-11)

Evidence-based. Sources: official papers/GitHub/project pages. UNKNOWN where not evidenced.
Claim discipline: no SOTA claim without number; no "first" without citation.

## Sources (primary)

- GeoChat: Kuckreja et al., CVPR 2024, arXiv:2311.15826, github.com/mbzuai-oryx/GeoChat, LLaVA-1.5 + LoRA, 318k instructions.
- TEOChat: Irvin et al., ICLR 2025, arXiv:2410.06234, github.com/ermongroup/TEOChat, Video-LLaVA adapt, TEOChatlas 554k (245k temporal + 308k single).
- EarthMind: Shu et al., arXiv:2506.01667, github.com/shuyansy/EarthMind, Qwen2.5-3B base, 4B total, SAP + Cross-modal Fusion, EarthMind-Bench 2000+ RGB-SAR pairs, FusionEO 30k.
- VisTA / QAG-360K / CDQAG: Li et al., arXiv:2410.23828, github.com/like413/VisTA, QAG-360K 360k triplets, 10 LC classes, 8 Q types.
- CDVQA: Yuan et al., TGRS 2022, github.com/YZHJessica/CDVQA, SECOND-based, 2968 pairs, ~122k QA.
- RS-Agent: Xu et al., Sci China Inf Sci 2026 (arXiv:2406.07089), github.com/IntelliSensing/RS-Agent, LLM controller + Solution/Knowledge Space + Task-Aware Retrieval + DualRAG.
- SkyEyeGPT: Zhan et al., ISPRS 2025, arXiv:2401.09712, 968k instruction samples, single-image unified.
- UniChange: CVPR 2026, MLLM unifying binary+semantic change + mask.
- VRSBench: NeurIPS 2024 D&B, arXiv:2406.12384, HF xiang709/VRSBench, 29.6k images, 52k refs, 123k QA.
- RSVQA: Lobry et al., TGRS 2020, LR/HRBEN on BigEarthNet + OSM QA.
- BigEarthNet-MM / reBEN v2: 549-590k S1+S2 pairs, GeoTIFF, CDLA-Permissive, bigearth.net.
- SEN1-2: Schmitt et al., 2018, 282k SAR-optical pairs.
- OSVQA / generic optical-SAR fusion: STSF-Net, OptSARChangeDetection (GitHub), Delta-SN6 — fusion literature, no language output.

## Capability matrix

Legend: YES / PARTIAL / NO / UNKNOWN. SatQuery = current repo (heuristic primary, EuroSAT 1500 present untrained).

| Capability | SatQuery now | GeoChat | TEOChat | EarthMind | VisTA | SkyEyeGPT | UniChange | RS-Agent | Generic VLM (GPT-4o/Qwen2-VL) |
|---|---|---|---|---|---|---|---|---|---|
| single-image VQA | PARTIAL (heuristic+template, works demo, not learned) | YES | YES (+3.5 avg over GeoChat zero-shot) | YES (beats GPT-4o on RSVQA-HRBEN/VRSBench-VQA) | NO (change-only) | YES | NO | YES (via GeoChat tool) | PARTIAL (poor RS, hallucinates) |
| scene description / captioning | PARTIAL (deterministic fingerprint caption, numeric, not learned) | YES | YES | YES | NO | YES | NO | YES | PARTIAL |
| grounding (boxes) | PARTIAL (color/texture heuristic, IoU modest, unmeasured) | YES (referring detection, region dialogue) | PARTIAL (TRE 74.9% vs Video-LLaVA 1.9%) | YES (DIOR-RSVG CIDEr 428, VRSBench grounding 55.6%, pixel SAP) | YES (mask grounding for change, oIoU 84.8% QAG) | PARTIAL | PARTIAL (mask output) | YES (via YOLO/DOTA tools) | NO |
| temporal reasoning (bi-temporal) | PARTIAL (LAB+Otsu change %, no semantics, no registration) | NO | YES (xBD QA 89.9%, S2Looking 73.4%, ABCD 85.6%, CDVQA 47.2% zero-shot) | NO | YES (CDVQA OA 73.1% T1, AA 65.9%) | NO | YES | PARTIAL (building damage via changeos/open-cd) | PARTIAL (weak, GPT-4o 38% xBD damage) |
| change detection mask | YES (classical, 22.37% flood demo, Otsu+CC) | NO | PARTIAL (no explicit mask, QA only) | NO | YES | NO | YES | PARTIAL | NO |
| change VQA | PARTIAL (change description + VQA supplement, not CDVQA-trained) | NO | PARTIAL (47.2 CDVQA zero-shot vs specialist 63.4) | NO | YES | NO | PARTIAL | NO | NO |
| optical | YES (RGB proxy) | YES | YES | YES | YES | YES | YES | YES | YES |
| multispectral (S2 bands) | NO (RGB only, B04/B08 not used) | NO (RGB) | NO | PARTIAL (triplet-grouped pseudo-video) | NO | NO | UNKNOWN | NO | NO |
| SAR | PARTIAL (backscatter stats on luminance, synthetic-tuned) | NO | NO | YES (SAR-only strong, multi-sensor training) | NO | NO | NO | YES (DiffDet4SAR, SAR plane) | NO |
| optical-SAR joint | PARTIAL (decision IoU, Pair1 water IoU 0.61, built 0.00) | NO | NO | YES (only open model with explicit fusion; beats GPT-4 stacking which degrades) | NO | NO | NO | PARTIAL (separate tools, no language fusion) | NO (pseudo-RGB stacking degrades fine-grained) |
| cross-modal reasoning (language) | PARTIAL (corroboration sentences from IoU) | NO | NO | YES (FusionEO 30k, HCA attention) | NO | NO | NO | NO | NO |
| agentic planning | PARTIAL (deterministic regex + dispatch, no replanning) | NO | NO | NO | NO | NO | NO | YES (LLM controller + Task-Aware Retrieval, 95% planning acc) | NO |
| specialist tool routing | YES (registry-pinned 5 tools, auditable) | NO | NO | NO | NO | NO | NO | YES (18 tools, plug-and-play) | NO |
| input validation | YES (dims/CRS/modality, refusal Demo5) | NO | NO | NO | NO | NO | NO | NO | NO |
| metadata validation (CRS/transform/bands) | PARTIAL (reads CRS/transform/bounds/nodata via rasterio, warns, no GSD/reproj) | NO | NO | NO | NO | NO | NO | NO | NO |
| evidence fusion | PARTIAL (mean of heuristics, fusion-aware) | NO | NO | PARTIAL (modality mutual attention, no trace) | PARTIAL (text-visual contrastive) | NO | NO | PARTIAL (controller aggregates) | NO |
| contradiction detection | PARTIAL (sentences describe IoU<0.12 as disagree, but no score/threshold policy, no confidence cut) | NO | NO | NO | NO | NO | NO | NO | NO |
| uncertainty | NO (heuristic scores, explicitly uncalibrated) | NO | NO | NO | NO | NO | NO | NO | NO |
| calibrated confidence | NO | NO | NO | NO | NO | NO | NO | NO | NO |
| abstention / refusal | PARTIAL (only dim-mismatch refusal; no quality/OOD abstention) | NO | NO | NO | NO | NO | NO | NO | NO |
| failure awareness | PARTIAL (VLM lifecycle flags, guard vs echo/underscore) | NO | NO | NO | NO | NO | NO | NO | NO |
| visual evidence | YES (JET+mask, overlays, SVG boxes) | PARTIAL (boxes in text) | NO | YES (masks via decoder) | YES (masks) | NO | YES | PARTIAL (tool outputs, no unified canvas) | NO |
| audit trace | YES (9-12 step trace, tools/params/timings, UI panel) | NO | NO | NO | NO | NO | NO | PARTIAL (LangChain trace, not eval trace) | NO |
| reproducibility (deterministic) | YES (regex, no LLM planner) | PARTIAL | PARTIAL | PARTIAL | YES | PARTIAL | PARTIAL | NO (GPT-4o-mini default, flaky) | NO |
| offline deployment | YES (CPU, 0MB heuristic, no API) | YES (open weights 7B) | YES (open) | YES (open 4B) | YES | YES | YES | NO (requires OPENAI_API_KEY, HF download, LangChain hub) | NO |
| GeoTIFF / CRS | PARTIAL (rasterio→PIL, no tiling/reproj) | NO | NO | NO | NO | NO | NO | NO | NO |
| cloud / quality awareness | NO | NO | NO | NO | NO | NO | NO | PARTIAL (cloud_removal/dehaze tools, not QA-aware) | NO |
| human-in-loop | NO (reports only) | NO | NO | NO | NO | NO | NO | PARTIAL (multi-turn) | PARTIAL |
| workflow automation | PARTIAL (single call → answer+evidence+report) | NO | NO | NO | NO | NO | NO | YES | NO |
| benchmark coverage | NO (no public scores claimed; synthetic only) | YES (RS multitask bench) | YES (temporal + zero-shot ABCD/CDVQA) | YES (EarthMind-Bench + AID/UCM/RSVQA/VRSBench/DIOR) | YES (QAG/CDVQA) | YES | YES | YES (9 datasets/18 tasks) | N/A |
| operational usability (GUI) | YES (FastAPI + single-file UI + 5 demos) | PARTIAL (demo) | PARTIAL | PARTIAL | NO | PARTIAL | NO | PARTIAL (demo.py) | YES |

## Where everyone is weak (opportunity)

1. No system does validation + quality-aware routing + contradiction + abstention + calibrated confidence + trace together. SatQuery already has 3 of 6 (validation, trace, refusal scaffold).
2. No open VLM handles Cartosat-submeter + RISAT domain shift explicitly; EarthMind closest but trained on OpenEarthMap-SAR/DFC/MSAW, not Indian sensors.
3. No system exposes modality reliability or image-quality-conditioned confidence; RS-Agent has denoise/cloud tools but does not gate answers on quality.
4. No system distinguishes real vs apparent change (illumination/season/registration) with confidence; VisTA grounds but does not abstain.
5. No system provides analyst-grade audit (inputs→plan→tools→cross-checks→confidence factors→refusal reason) offline and deterministically; SatQuery trace is closest, RS-Agent trace requires API.
6. Optical-SAR language reasoning is covered only by EarthMind (4B, 8×A100 training); no lightweight/agentic alternative exists — SatQuery decision-fusion + corroboration is weaker but far cheaper and auditable.

## What SatQuery must not claim

- NOT SOTA VQA/grounding/change (heuristic vs GeoChat/TEOChat/EarthMind/VisTA numbers above).
- NOT calibrated confidence, NOT learned fusion, NOT multispectral physics, NOT registered change.
- Defensible claim: only deterministic offline agent with registry + validation + refusal + visual evidence + trace covering all 5 SIH scopes in one call.
