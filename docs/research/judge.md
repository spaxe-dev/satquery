I've independently verified the key dataset claims (CDVQA is real, on GitHub at YZHJessica/CDVQA, 2,968 pairs / 122k QA, built on SECOND — matching the researcher's account). The Stage 1 research is largely accurate. Here is my judge assessment.

---

# JUDGE ASSESSMENT — SIH26167 SatQuery AI (ISRO)

## judge_assessment

This is one of the hardest software PSs in SIH 2026, and the proposed approach is directionally correct: fine-tune/adapt open RS VLMs (GeoChat / Qwen-VL class) rather than wrap a generic LLM, build a deterministic-first agentic router, and treat the execution trace as a first-class deliverable. The team appears to understand that this PS is graded on benchmark numbers plus an undisclosed SAC Cartosat-2S/RISAT evaluation set — not on pitch polish.

However, the plan as written is a *compilation of published work* (GeoChat, TEOChat, VisTA, BIT/Changer, BigEarthNet-MM training) with an orchestration layer on top. That is exactly what ISRO asked for, so it's not disqualifying — but a judge must ask: of the ~20 teams that will attempt this, most will also fine-tune GeoChat and build a FastAPI app. What separates the winner is (a) benchmark scores on the named splits, (b) graceful behavior on the hidden SAC data, and (c) a genuinely working optical–SAR language-output tool, which is the one component with almost no public text-annotated paired data. The proposal acknowledges all three but underweights the risk that (c) may be barely functional at demo time.

**Preliminary verdict: PURSUE — conditional.** High winning potential for an ML-strong team; the optical–SAR component is the make-or-break item and needs a de-risked fallback design before commitment.

## strengths

1. **Exact compliance mapping.** Every mandatory clause (adaptation dataset, VQA + captioning/grounding, change-VQA, optical–SAR, agentic registry, auditable trace, GUI, downloadable reports) has a named component. Judges checking a compliance matrix will find no gaps.
2. **Correct reading of the disqualification clause.** "A generic LLM or VLM without RS adaptation will not satisfy the requirements" — the plan centers on BigEarthNet-MM fine-tuning, not GPT-4V wrapping.
3. **Deterministic-first routing.** Rule-based intent classification with LLM-planner fallback is the right engineering call: reproducible for judging, debuggable, and immune to planner flakiness during a live demo.
4. **BigEarthNet-MM is the ideal adaptation dataset** — 549k+ Sentinel-1/2 pairs directly match the optical–SAR mandate; the researcher's license/size facts check out.
5. **Offline-first, self-hosted deployment** is exactly right for ISRO's security posture and eliminates demo-day API-failure risk.
6. **Execution trace as a product feature.** The PS explicitly says only observable traces are evaluated — treating this as a UI feature, not an afterthought, is a genuine edge most teams will miss.
7. **Honest novelty framing.** The researcher does not claim invented algorithms; claims integration + SAR extension. Judges respect this.

## weaknesses

1. **Domain shift is the elephant in the room.** All training data is Sentinel (10m). The hidden SAC set is Cartosat-2S (sub-meter) + RISAT. A model fine-tuned on 10m patches may produce plausible-but-wrong answers at sub-meter resolution. The plan calls this "the single biggest unknown" but proposes no concrete mitigation (no multi-scale training, no test-time tiling strategy, no resolution-augmentation plan).
2. **Optical–SAR language output is weakly supervised.** Template-generated QA from BigEarthNet-MM class labels is legitimate but produces brittle, formulaic answers. On the SAC set, this tool may only handle "identify built-up/water regions" and nothing else.
3. **No benchmark targets stated.** "Competitive normalized scores" is vague. A winning team should commit to numbers: e.g., within X points of published SOTA on VRSBench VQA, CDVQA ~73% OA (VisTA baseline). Without targets, "we fine-tuned it" is unverifiable.
4. **CDVQA access is flagged but not resolved.** It's on GitHub (I verified), so this is solvable — but any dataset-access unknown left open at ideation stage is a planning smell.
5. **Confidence estimation is hand-waved.** "Tool-specific calibration or verbalized confidence" — verbalized confidence from a 2B VLM is notoriously miscalibrated. A judge will probe this.
6. **Compute assumption unexamined.** LoRA fine-tuning 2–7B VLMs on 100k+ instruction samples needs sustained GPU access (A100/4090-class for days). Does the team actually have this? Many teams discover at week 4 that they don't.
7. **Co-registration checking via SSIM threshold** is naive across modalities — optical vs SAR of the same area have near-zero pixel correlation. The validator will false-reject valid optical–SAR pairs. Needs metadata-based (CRS/geotransform) checking as primary.

## judge_questions

1. **"Why does this need an agentic framework at all? A single multi-task VLM could do all four tasks."**
   Good answer must contain: evidence that no single open VLM handles SAR + bi-temporal + grounding simultaneously (TEOChat is optical-only, GeoChat is single-image — cite them); the auditability argument (per-tool traces, per-tool benchmark scores); and a fallback acknowledgment that a unified model is the future but doesn't exist today at student scale.

2. **"Your training data is 10m Sentinel. The SAC evaluation is sub-meter Cartosat + RISAT. Why will your model work at all?"**
   Good answer: multi-scale training strategy (tile sub-meter imagery into 10m-equivalent patches, or train with resolution jitter), a documented resolution-robustness experiment, and honesty that zero-shot transfer is uncertain with a tiling fallback that at least preserves geometric validity. A team that says "it should generalize" gets zero marks here.

3. **"What exactly is novel? GeoChat, TEOChat, VisTA all exist."**
   Good answer: the combination (four capabilities + auditable orchestration + SAR-language output) doesn't exist; the optical–SAR *text-output* tool is the genuinely under-explored piece (paired SAR-optical data with language annotations essentially doesn't exist publicly); and quantified deltas vs each prior system.

4. **"What happens when your model is wrong on the hidden set? Will it know?"**
   Good answer: a real calibration method (temperature scaling on held-out splits, or conformal prediction), an abstention/"low confidence" path in the UI, and demo evidence of the system refusing to answer when inputs are invalid.

5. **"How do you validate your optical–SAR co-registration check when optical and SAR pixels don't correlate?"**
   Good answer: metadata-first (geotransform/CRS equality), with SSIM/correlation only as an advisory signal, not a gate. (The current proposal gets this wrong — SSIM thresholding across modalities fails.)

6. **"Show me your benchmark numbers."**
   Good answer: a table — model, split, metric, published SOTA, our score — on VRSBench, RSVQA, CDVQA test sets, produced before the finale, not promised.

7. **"What is your GPU budget and fine-tuning timeline?"**
   Good answer: named hardware, named training runs with wall-clock estimates, and a plan B (smaller backbone, subset training) if compute falls through.

8. **"Why should ISRO adopt this over just using GeoChat + a change-detection model separately?"**
   Good answer: the non-expert interface + validation + evidence + trace is the operational value; plus demonstrated behavior on Indian scenes (Bhuvan/Bhoonidhi imagery) rather than only European Sentinel data.

9. **"Your SAR answers come from templates over BigEarthNet labels. Isn't that a lookup table, not understanding?"**
   Good answer: acknowledge the weak supervision, show diversity of the QA generation, and demonstrate at least one open-vocabulary SAR query the system handles via the VLM rather than the template path.

10. **"What will you demo live?"**
    Good answer: scripted demo on 3–4 prepared scenes (flood pre/post Sentinel pair, urban change pair, optical–SAR pair) + one live Copernicus download, with the execution trace visible on screen throughout, and the PDF report generated live.

## rejection_reasons

- Generic-VLM wrapper (explicitly disqualified) — would be instant rejection; the plan avoids this but any slippage (e.g., using an external LLM planner as the *primary* router) reopens the risk.
- No fine-tuning actually performed ("we prompt-engineered GeoChat") — fails the mandatory adaptation clause.
- Optical–SAR tool that just concatenates two captions — a judge will see through this in one query.
- No benchmark numbers at finale — the PS names the evaluation method; a team that can't show scores on named splits looks like it didn't read the PS.
- Demo crashes on a GeoTIFF upload — input validation is mandatory scope; failing on the mandated format is fatal.
- Over-promising ("disaster management platform for India") beyond what was demonstrated.

## competitive_position

**GENUINELY DIFFERENT (as an integrated system), INCREMENTALLY BETTER (component-wise).**
- vs GeoChat/SkyEyeGPT/RSGPT: genuinely different — adds temporal, SAR, and orchestration.
- vs TEOChat: incrementally better — adds SAR + agent layer; TEOChat already does temporal reasoning and is a strong foundation the team should consider building *on* rather than beside.
- vs VisTA/QAG-360K: not competitive head-on (it's SOTA on CDVQA) — correctly positioned as a component/idea source.
- vs commercial geo-AI (SkyFi, Google Earth AI): different scope (pixel-level, offline, open).
The defensible claim: **only system combining all four mandated capabilities under an auditable controller** — true today, but it's an integration claim, and 3–5 other SIH teams will make the same claim. Execution quality (scores + hidden-set behavior) decides the winner.

## judge_scores

| Criterion | Score | Reasoning |
|---|---|---|
| Problem alignment | 9 | Near-perfect compliance mapping; every clause addressed. Loses a point for no stated benchmark targets. |
| Impact | 7 | Real accessibility problem, but impact is mediated by ISRO adoption; the pitch shouldn't overclaim field deployment. |
| Technical depth | 8 | Genuine ML work (LoRA fine-tuning, dual-encoder fusion, Siamese change model). Would be 9 with a domain-shift mitigation plan. |
| Innovation | 6 | Honest integration novelty; optical–SAR language output is the only truly novel sliver, and it's weakly supervised. |
| Differentiation | 6 | The combination claim is defensible but replicable by peer teams; component-wise it's mostly prior art. |
| Feasibility | 7 | Buildable in the timeline for an ML-strong team; compute access and the optical–SAR tool are the two real risks. |
| Data credibility | 8 | All named datasets verified real and accessible (I independently confirmed CDVQA on GitHub). Sentinel fallback for demos is solid. |
| Demo potential | 8 | Offline, scripted, visually rich (overlays, traces, reports). Strong demo story if the SAR tool holds up. |
| Scalability | 7 | Docker/GPU stack scales fine; model quality at Indian-sensor resolutions is the scaling limit, not infrastructure. |
| Winning potential | 7 | Top-quartile PS choice with high ceiling; winner only if benchmark numbers + hidden-set robustness are actually delivered. |

Not an average — the profile is "high alignment, moderate novelty, execution-dependent." This team wins or loses on execution, not on idea.

## winning_factors

1. Published benchmark table beating or matching SOTA-adjacent numbers on the named splits.
2. A working, non-trivial optical–SAR demo (fused evidence maps + language answer), even if narrow.
3. Graceful, confident behavior on out-of-distribution Cartosat/RISAT-style inputs (tiling + abstention + trace).
4. The execution-trace UI shown live — directly matches the stated evaluation method.
5. Demo on Indian scenes (Bhuvan/Bhoonidhi), not just European Sentinel data.

## required_changes

1. Add a concrete domain-shift mitigation plan (multi-scale training / resolution jitter / test-time tiling) — this is the #1 judge attack.
2. Fix the co-registration validator: metadata-first, SSIM only advisory (current design fails across modalities).
3. Commit to numeric benchmark targets per named split.
4. Replace "verbalized confidence" with a real calibration method (temperature scaling or conformal) on held-out data.
5. Design a degraded-mode fallback for the optical–SAR tool (e.g., per-class evidence maps + templated summary) so it never demos as a stub.
6. Confirm GPU access and write the training schedule into the plan.
7. Evaluate building change-VQA on TEOChat rather than from scratch — reuse > rebuild.

## disputed_claims

- "Qwen3.5-2B + LoRA hits ~74% OA on CDVQA" — plausible given VisTA's ~73% but I could not independently verify this specific 2026 letter; treat as unconfirmed until reproduced.
- "CDVQA historically request-gated" — disputed; I found it on a public GitHub (YZHJessica/CDVQA). Access risk is lower than stated.
- "No existing system combines all four capabilities" — true today, but this is a fast-moving field (UniChange CVPR 2026, ChangeChat-105k Jul 2025); re-verify at finale time.

## evidence

- CDVQA dataset: Yuan, Mou, Xiong, Zhu, IEEE TGRS 2022 (doi 10.1109/TGRS.2022.3203314); 2,968 pairs, 122k QA, built on SECOND; public at github.com/YZHJessica/CDVQA — verified via IEEE Xplore and TUM portal.
- Active CDVQA research through 2025–26: TCSSM domain-generalization CDVQA (arXiv 2508.08974), QAG-360K, ChangeChat-105k — confirms the field is hot and SOTA is ~73–75% OA; also confirms peer teams can find the same baselines.
- SIH 2026 timeline confirmed (launch Aug 21, 2026; finale Dec 2026) — roughly 10–12 weeks to finale prototype; tight but consistent with the researcher's 6–10 week build estimate.
- SAC has run SIH PSs with the same "train on open data, evaluate on hidden ISRO set" pattern before (sac.gov.in/sih2025), confirming the hidden-set evaluation culture and the domain-shift risk.

## revised_position

After independent verification, I *raise* my assessment slightly on data credibility (CDVQA access is not a real risk) and *lower* it on the co-registration validator design (cross-modal SSIM gating would actively fail the mandated optical–SAR input class). Final position: **PURSUE — this is a winnable PS for this team, but only if items 1–3 of required_changes are in the plan before ideation submission.** The idea alone will not win; among the several teams that will attempt this PS, the winner is decided by benchmark numbers, hidden-set robustness, and a live auditable trace — all execution artifacts, all achievable, none optional.