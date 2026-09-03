# FINAL JUDGE ASSESSMENT — SatQuery AI (SIH26167)

## judge_assessment

After six rounds of adversarial deliberation, this team arrives at the final judging in an unusual position: the idea itself is now unremarkable — the PS hands every one of ~hundreds of teams the same recipe (BigEarthNet adaptation + specialist model registry + agentic controller + GUI). What distinguishes this team is that they are the rare SIH team that spent their deliberation *falsifying their own claims*. The 74% OA figure was struck for lacking a citation, SSIM validation was retracted, "exact match competitor" labels were downgraded to prior art pending checkpoints, and CDVQA availability was downgraded from "VERIFIED" to "pending download verification." That intellectual honesty is worth real points with a judge — but honesty about unverified claims is not the same as having verified anything.

The hard truth at final judging: through six rounds, **zero bytes of training data have been downloaded by anyone on this team, zero weights have been run, GPU ownership is still unknown, and the evaluation answer format remains an open question** — despite four rounds of agents demanding exactly these artifacts. The plan is good. The evidence base is still paper.

## strengths

1. **Complete mandatory-scope coverage, correctly understood.** All five mandated capabilities (RS-adapted VLM, single-image VQA + captioning/grounding, bi-temporal change VQA, optical–SAR joint analysis, agentic orchestration) are addressed with no scope-dodging. Many teams will quietly under-deliver the optical–SAR pair analysis; this team identified it early as the hardest and highest-variance axis.
2. **Correct architectural convergence.** One shared 2B-class quantized vision backbone with task-specific LoRAs (not four unrelated models), resident-simultaneous serving with CPU offload, deterministic router + LLM planner emitting auditable traces. This resolves both integration-death risk and serving latency in one stroke — and matches what the PS actually scores.
3. **Hybrid optical–SAR design is genuinely smart.** Physics-grounded SAR template floor (backscatter statistics → scattering mechanism → land-cover statements, always available even when fine-tuning fails) + supervised ceiling if OSVQA verifies. This gives a guaranteed demo floor on the component most likely to fail.
4. **Domain-shift hedging package.** GSD simulation, speckle injection, radiometric jitter, calibrated "low confidence" outputs, honest per-domain test breakdowns, and adoption of TCSSM's published domain-generalization recipe. Most teams will leaderboard-chase Sentinel/aerial numbers and get punished on the hidden Cartosat/RISAT set; this team plans not to.
5. **Differentiators chosen where scoring actually lives:** cross-modal consistency checks (does optical contradict SAR?), live adversarial-input handling (mismatched pairs, cloudy optical), genuinely auditable execution traces. These beat template Gradio submissions.
6. **Self-correction discipline.** Striking unsourced numbers and retracting dead heuristics mid-debate is exactly the behavior of a team that won't be caught lying in Q&A.

## weaknesses

1. **Nothing has been executed. Six rounds, zero artifacts.** No checksummed download of CDVQA, OSVQA, or reBEN-S1. No LoRA trained. No checkpoint of EarthMind actually pulled. This is the single largest weakness and it is entirely self-inflicted — every dependency in the plan rests on repo existence alone.
2. **Compute ownership unknown in week N.** A student team discovering in week 5 that it must rent ≥24GB cloud GPUs reshapes the entire schedule. This was flagged in round 2 and never resolved.
3. **Evaluation format unknown → all numeric targets deferred.** Closed-vocab vs free-text vs LLM-judged requires different training data and different models. The team accepted "format first, then targets" but has not sent the ISRO inquiry yet.
4. **Hidden-set domain shift is unmitigable, only hedgeable.** All training data (Sentinel 10–20m Europe, aerial US SECOND) differs from Cartosat-2S/RISAT Indian scenes in GSD, bands, radiometry. Every public benchmark number presented is inflated relative to what ISRO measures. OSVQA's ~10m 256px chips do not close this gap to sub-meter Cartosat.
5. **No idea-level moat.** EarthMind ships multi-sensor finetuned checkpoints. GeoChat/TEOChat are open-weight. Any rival can assemble the same stack. Differentiation is execution-only — survivable, but it means the win condition is polish and honesty, not invention.
6. **Multi-intent composition path is designed but untested**, despite the PS's representative queries including compound examples ("What changed AND highlight the water"). Upgraded to MEDIUM-HIGH risk until a passing test exists.

## judge_questions

Q1. "Show me one artifact: a downloaded dataset, a trained adapter, anything."
Good answer contains: checksums/file listings for CDVQA + OSVQA subsets, one completed LoRA run log, GPU confirmation. Anything less = paper team.

Q2. "EarthMind already ships multi-sensor VLM weights. Why does your system exist?"
Good answer contains: EarthMind positioned as a registry *component*, with our value being orchestration, cross-modal consistency checking, calibration, and domain-shift robustness — not "we built the hard part."

Q3. "Your models train on Sentinel and aerial US data. ISRO evaluates on Cartosat/RISAT. Why should we trust any number you show?"
Good answer contains: per-domain test breakdowns, augmentation strategy, graceful low-confidence behavior, TCSSM-style domain-generalization technique, and explicit acknowledgment that public benchmarks overstate hidden-set performance.

Q4. "What happens when your model is wrong on the SAC set?"
Good answer contains: confidence estimation that says 'uncertain', cross-modal disagreement flagging, injection-tested input validator — not silence or hand-waving.

Q5. "Is your agent actually agentic or a hardcoded if-chain?"
Good answer contains: live trace showing query interpretation → plan emission → tool selection → composition across 2–3 chained tools, with a passing test for compound queries like the PS's own examples.

Q6. "What did ISRO say when you asked about evaluation format?"
Good answer contains: proof of inquiry via SIH channel and the fallback (short factual answers viable under all three scoring regimes).

Q7. "Where does your training run happen?"
Good answer: owned ≥24GB GPU or committed rental budget, with the 8–12GB residency math shown.

## expected_good_answers

Covered inline above. Common thread: every convincing answer contains a produced artifact or a measured number, never a claim about a repo the team hasn't opened.

## rejection_reasons

1. Zero verified datasets/compute by presentation time — the entire pipeline is unexecuted speculation.
2. Optical–SAR component fails live on SAC-like inputs because the 10m→sub-meter gap was never bridged with real experiments.
3. Demo shows only cherry-picked benchmark images; breaks on an ISRO-provided Cartosat/RISAT pair.
4. Agentic layer exposed as a static router under questioning ("run it on this compound query" fails).
5. Confident hallucination on the hidden set instead of calibrated uncertainty.
6. Integration seams visible: four half-working specialists behind a Gradio shell.

## competitive_position

**MEANINGFULLY BETTER than the median submission; NOT differentiated at the idea level.**

Versus the field: the median SIH submission here will be GeoChat/TEOChat behind Gradio with template outputs and no optical–SAR credibility. Against that baseline this plan is meaningfully better on trust infrastructure (validators, consistency checks, calibration), domain-shift hedging, and hybrid SAR-language design. Versus EarthMind-class released models: MOSTLY THE SAME at the component level (they concede this and correctly absorb EarthMind into their registry); GENUINELY DIFFERENT only at orchestration/evidence/trust level. Verdict: **INCREMENTALLY-BETTER-to-MEANINGFULLY-BETTER, execution-dependent**.

## judge_scores

| Criterion | Score | Reasoning |
|---|---|---|
| Problem alignment | 9 | Full mandatory scope, correct reading of paired/cross-modal emphasis; only alignment gap is untested compound-query routing. |
| Impact | 7 | Real value for non-expert RS querying; impact narrative constrained by evaluation-format uncertainty and hidden-set risk. |
| Technical depth | 8 | LoRA/shared-backbone design, three-tier validator, physics-grounded SAR layer, residency planning — deep, though MI tier is stretch-goal vaporware currently. |
| Innovation | 5 | No idea-level novelty conceded honestly; PS-prescribed architecture; innovation relocated to trust/calibration mechanisms. |
| Differentiation | 5 | Same open-weight stack available to all rivals; EarthMind exists with weights. Execution-only moat. |
| Feasibility | 6 | Buildable from open resources, but zero execution to date and compute unconfirmed. Conditional, not demonstrated. |
| Data credibility | 5 | CDVQA/OSVQA/reBEN exist as repos/papers but nothing downloaded; effective reBEN-S1 patch fraction unknown; OSVQA resolution mismatch confirmed. |
| Demo potential | 8 | Strong plan: adversarial inputs live, auditable traces, hybrid floor guarantees the optical–SAR demo never fully dies. |
| Scalability | 7 | 2B quantized residents + CPU offload scale sensibly to lab/deployment; unclear beyond single-GPU. |
| Winning potential | 6 | Top-quartile plan quality; winning depends entirely on whether week-1 evidence landed. |

Not averaging these deliberately: the 9 alignment score coexists with a 5 data-credibility score, and that tension *is* the verdict.

## winning_factors

1. Week-1 evidence gate actually executed: checksums, one trained adapter, GPU confirmed, ISRO format inquiry sent.
2. Live adversarial demo — mismatched pairs, cloudy optical, compound queries — handled gracefully while rivals' canned demos break.
3. Calibrated uncertainty and cross-modal contradiction flagging, which read as maturity to SAC evaluators.
4. Hybrid SAR tool guaranteeing a credible optical–SAR answer even if fine-tuning underperforms.
5. Honest per-domain test reporting — evaluators notice teams who don't inflate.

## required_changes

1. **Execute the evidence week before any further design work** — non-negotiable, already committed: downloads with checksums (CDVQA, OSVQA, reBEN-S1 subset), one end-to-end LoRA run, GPU confirmation.
2. **Send the ISRO/SIH evaluation-format inquiry immediately**; train short-factual-answer models viable under all three scoring regimes meanwhile.
3. **Build and pass a multi-intent composition test** before claiming agentic capability.
4. **Evaluate EarthMind checkpoints as a registry component** rather than competing with them head-on.
5. **Keep the MI registration tier as documented stretch goal only**; ship metadata-first validator with flag-don't-reject warnings.
6. **Defer all numeric benchmark commitments** until answer format is known; the 74% figure stays struck.

## disputed_claims

- "~74% OA CDVQA" — struck; no named paper/split after repeated challenge.
- SSIM cross-modal validation — retracted, replaced by metadata-first + MI-secondary + injection-tested.
- "EXACT MATCH competitors" — downgraded to prior art/close match; EarthMind's released weights later verified, partially restoring the threat.
- Feasibility's blanket "LOW/MEDIUM risk" — rejected; optical–SAR output quality held at MEDIUM-HIGH given stacked unverified dependencies.
- Scientist's MI validator operating parameters — never fully specified; demoted accordingly.

## evidence

- CDVQA: Yuan et al., TGRS 2022, github.com/YZHJessica/CDVQA (repo verified; archive download NOT).
- OSVQA/TGFNet: arXiv 2411.15770, 6,008 optical–SAR pairs, ~10m 256px chips, github.com/mmic-lcl/.
- EarthMind: github.com/HuaDeity/EarthMind + shuyansy/Earth-Observation-VLMs — pretrained and multi-sensor finetuned checkpoints released since May 2025 (verified).
- TCSSM arXiv 2508.08974 — domain-generalization CDVQA precedent.
- reBEN v2: 549k clean pairs claimed; S1 patch-quality fraction still unexamined.
- Persistent negatives across six rounds: zero dataset downloads by the team, GPU ownership unresolved, evaluation format unknown, no QLoRA VRAM citation with vision-tower activations.

## revised_position

Unchanged from my round-5 position and reinforced: **PURSUE, strictly conditional.** The conditional has sharpened, not softened: this is a top-tier *plan* attached to a zero-artifact *track record*. If the week-1 gate produces its five facts (checksums, adapter run, GPU, ISRO reply, EarthMind eval), this team presents as a top-quartile finalist with genuine winner potential on trust-and-execution grounds. If week 1 produces more diagrams, they are indistinguishable from the median Gradio-wrapper team and should expect to be cut at shortlisting.

## FINAL VERDICT

**ADVANCE: YES, conditionally — advance them out of internal review into build phase.**
**WINNER: Not yet.** Among 20 presentations today, this team ranks top 4 on plan quality but would lose to a team that shows one working end-to-end run on real downloaded data, because judges reward demonstrated systems over sophisticated intentions.

What makes them lose: arriving with unverified datasets, no GPU plan, a broken compound-query demo, or confident hallucinations on SAC-like imagery.
What makes them win: the evidence-week artifacts plus the live adversarial demo plus calibrated honesty — a system that says "I'm uncertain, here's why" while nineteen rivals bluff.

Verdict: **SURVIVES WITH MAJOR CHANGES — PURSUE, gated on week-1 evidence.**