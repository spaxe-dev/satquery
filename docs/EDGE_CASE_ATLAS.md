# Edge-Case Atlas — SatQuery Failure Atlas (55 cases)

For each: scenario | why | current SatQuery | expected | competitor | detectable? | mitigation | difficulty | importance | metric | demo value | USP?

## 1. Query failures

1. Empty/gibberish query ("asdf ???") — no intent. Current: defaults to vqa, heuristic caption. Expected: clarify-request, not answer. Competitors: all answer anyway. Detectable: yes (len/intent-none). Mitigation: intent-none → ask-clarify (<2h). Importance HIGH. Metric: clarify rate. Demo HIGH. USP yes (refuse-to-guess).
2. Ambiguous single vs temporal ("Has built-up increased?" with 1 image) — planner needs 2. Current: validates temporal needs 2, refuses if task temporal else VQA single. Expected: explain need pair + answer what single shows. Competitors: hallucinate temporal. Detect: yes. Mitigation: task-aware message (2h). Metric: correct-redirect rate. Demo HIGH.
3. Unsupported query ("Who owns this building? Count tanks by type?") — beyond tools. Current: heuristic VQA guesses. Expected: unsupported → abstain + suggest supported. Competitors: hallucinate. Detect: keyword/out-of-scope list. Mitigation: capability gate (4h). Metric: unsupported-abstention rate. Demo HIGH. USP yes.
4. Contradictory query ("Show water decrease using only SAR" but SAR missing) — Current: routes by images not query constraint. Expected: flag missing modality. Detect: yes. Mitigation: constraint check (4h). Importance MED.
5. Multi-intent compound ("What changed near water and can SAR corroborate?") — Current: compound_temporal_sar handles 3-image hero, but 2-image variant fragile. Expected: decompose ROI→change→water→SAR. Competitors: TEOChat handles temporal+region, EarthMind handles SAR, neither does 3-way. Detect: partial. Mitigation: planner decomposition (1-2wk). Metric: routing accuracy. Demo HIGH.

## 2. Image failures

6. Corrupt/undecodable file — Current: refuses with reason (good). Expected: keep. Competitors: crash. Detect: yes. Metric: refusal correctness. Demo LOW.
7. Unsupported ext (.jp2, .hdf) — Current: refuses (good). Keep + suggest convert. Metric: same.
8. Tiny image <64px (EuroSAT 64px edge) — Current: warns, answers anyway low input_q 0.72. Expected: warn + reduce confidence, still answer (benchmarks need it). Competitors: answer full confidence. Detect: yes. Mitigation: already done; add confidence penalty visible (1h). USP partial.
9. Huge image >16M px (Cartosat strip) — Current: warns tiled-required but still loads full (OOM risk). Expected: tile + mosaic or refuse-with-tiling-message. Detect: yes. Mitigation: tiling (1-2wk). Importance HIGH for hidden eval. Metric: no-OOM rate.
10. Empty/all-nodata image — Current: answers (hallucinates). Expected: nodata-fraction check → refuse. Detect: yes via rasterio nodata + flat histogram. Mitigation: 4h. Metric: correct-reject. Demo MED. USP yes.
11. Low-contrast/hazy image — Current: answers same confidence. Expected: contrast metric → lower confidence + note. Detect: yes (std, edge density). Mitigation: quality score (1d). Demo MED.
12. Same image twice as T1/T2 — Current: Otsu → negligible change (correct-ish) but no duplicate warning. Expected: perceptual hash → "identical inputs" note. Detect: yes (mean_diff≈0). Mitigation: 2h. Metric: duplicate-flag rate. Demo HIGH (jury gets it).
13. Reversed temporal order (T2 older) — Current: no date metadata use, brightness_delta sign flips silently. Expected: if dates present warn; else directional wording hedged ("T2−T1"). Detect: partial (needs metadata). Mitigation: date-aware (1d). Importance MED.

## 3. Metadata failures

14. Missing CRS/georef (PNG benchmarks) — Current: has_georef=False, proceeds, marks APPROXIMATE only if compat fails. Expected: explicit "pixel-only, no metric area" disclaimer. Competitors: ignore. Detect: yes. Mitigation: 2h wording. USP yes (no fake meters).
15. CRS mismatch between pair — Current: warns. Expected: warn + mark APPROXIMATE + lower confidence (already does via check_compat). Keep. Metric: warning recall.
16. Resolution/GSD mismatch (10m vs sub-meter) — Current: no GSD field (gsd_m=None never populated), dim-check only. Expected: GSD-aware refuse/warn + no fake % comparison. Detect: partial (needs catalog). Mitigation: populate GSD from TIFF tags/res metadata (1d) + policy (1d). Importance HIGH (Cartosat vs Sentinel). Metric: mismatch-flag rate. USP yes.
17. Band-count mismatch (3-band vs 12-band S2) — Current: drops to 3 bands silently + normalize. Expected: warn "bands>3 truncated; NIR lost". Detect: yes. Mitigation: 2h. Demo LOW.
18. Wrong modality labels (SAR named optical.png) — Current: filename+color heuristic misfires (known: RGB composites penalized). Expected: expose modality + confidence penalty visibly. Detect: partial. Mitigation: backscatter-statistics classifier (2-3d). Metric: modality accuracy. Importance HIGH.
19. Missing acquisition dates for change — Current: ignores. Expected: "dates unknown, seasonal confound possible" hedge. Mitigation: 2h template. Demo MED.

## 4. Temporal failures

20. Unregistered pair (shift/rotation) — Current: pixel diff → false change, no check (HIGHEST hidden risk). Expected: ECC/ORB alignment score → warn/refuse if shift large. Competitors: VisTA/TEOChat also assume registered. Detect: yes (phase correlation). Mitigation: MI/ECC check + warn (3-5d), full warp later (2wk). Metric: false-change suppression. Demo HIGH. USP yes if done.
21. Illumination/seasonal pseudo-change (shadow, phenology) — Current: Otsu fires, veg_delta hint only. Expected: change confidence + "apparent vs real" hedge. Detect: partial (brightness_delta, veg_delta exist). Mitigation: change-confidence + persistence filter (1wk). Metric: selective accuracy. Importance HIGH.
22. Cloud in one date — Current: no cloud mask, clouds → extensive change. Expected: cloud-fraction → qualify + SAR fallback suggestion. Detect: yes (bright+low-tex heuristic 1d). Mitigation: 2-3d. Demo HIGH. USP yes.
23. No-change pair — Current: negligible-change path (good). Keep. Metric: true-negative rate.
24. Subtle change below Otsu floor — Current: Otsu<12 fallback to p85, may miss. Expected: report sensitivity limit. Detect: yes (mean_diff). Mitigation: expose threshold in trace (done) + sensitivity note (2h).
25. Large change >45% (mosaic error) — Current: evidence 0.58 penalty (good). Keep + warn "check registration".

## 5. Optical failures

26. Heavy cloud cover — Current: water/veg masks fire on clouds, no warning. Expected: cloud-aware abstain/qualify. Detect: 1d heuristic. Mitigation: same as 22. USP yes.
27. Haze/saturation — Current: same. Expected: quality-gated confidence. Mitigation: quality score.
28. Night/dark optical — Current: dark→water false positive (87%→4.2% fix helped but still). Expected: brightness gate. Detect: yes. Mitigation: 4h.
29. RGB-only NIR loss (no B08) — Current: NDVI-proxy G−R, weak. Expected: label "RGB-proxy, not true NDVI". Already in code comments but not user-visible. Mitigation: 1h wording. Importance MED (jury: honesty).
30. Small water missed / narrow river — Current: morphology 7×7 removes narrow features. Expected: multi-scale note. Mitigation: tuning (1d). Metric: recall on VRSBench water.

## 6. SAR failures

31. Speckle extreme (CV~0.9 synthetic) — Current: thresholds tuned synthetic, real RISAT will shift. Expected: CV-conditioned thresholds + confidence cut. Detect: yes (speckle_cv exists). Mitigation: adaptive thresholds (2-3d). Importance HIGH.
32. RGB composite fed as SAR — Current: −0.12 penalty hidden in code. Expected: visible "composite, reduced reliability" badge. Mitigation: 2h UI. Demo MED.
33. SAR layover/foreshortening (urban/hill) — Current: bright→built-up, no geometry note. Expected: hedge in built answer. Importance LOW for 4d, MED for 2mo.
34. Wrong polarization/incidence ignored — Current: no metadata use. Expected: expose if present. Mitigation: metadata pass-through (1d).
35. Flood vs wet soil ambiguity — Current: dark+smooth→water, no ambiguity flag. Expected: "water-like, could be wet soil" when SAR-only. Mitigation: wording (2h). USP yes (honest ambiguity).

## 7. Cross-modal failures

36. Optical-SAR disagreement (water IoU 0.0, both >2%) — Current: sentence says disagree + low confidence mapping exists (evidence_agreement 0.35+0.55×IoU) but overall still ~0.6, no explicit flag. Expected: CONTRADICTION badge + confidence cut + "do not trust fused". Detect: yes (IoU). Mitigation: 1d policy (IoU<0.12 → contradiction flag + −0.15 overall). Metric: contradiction precision/recall. Demo VERY HIGH. USP yes — no competitor does this.
37. One modality missing water, other strong — Current: sentence covers, no abstention. Expected: "uncorroborated" label. Mitigation: same 1d.
38. Built IoU ~0.0 always (physics) — Current: correctly explains physics (good). Keep; do not treat as failure. Metric: explanation presence.
39. Unregistered optical-SAR pair — Current: warn-only 25%, resize into compatibility (silent resize in fuse _prep). Expected: no silent resize → APPROXIMATE tag (partially done). Mitigation: enforce check_compat refuse on large mismatch (4h). Importance HIGH.
40. Cloud-optical + good-SAR (fallback case) — Current: no routing change, averages both. Expected: SAR-weighted answer + "optical degraded, SAR primary". Detect: needs cloud score. Mitigation: quality-weighted fusion (1wk). Demo VERY HIGH. USP yes.

## 8. Model failures

41. VLM garbage (echo/underscores) — Current: guard _is_invalid_generation + fallback (good, tested). Keep. Metric: guard precision. Competitors: no guard.
42. VLM hallucinated count ("12 buildings") — Current: connected-components estimate capped 20 + hedge (good). Keep + never present as exact.
43. Grounding box on wrong object — Current: heuristic boxes, conf 0.42-0.72, no IoU eval. Expected: show conf + "heuristic, verify" label. Mitigation: wording (1h). Metric: IoU on VRSBench (measure, don't claim).
44. Change semantic error (says vegetation loss when construction) — Current: type_hints from veg/brightness only. Expected: "possible" language (already hedged). Keep.
45. OOM on large TIFF — Current: loads full array. Expected: tiling/refuse. See 9.

## 9. Grounding/change/agent/confidence/hallucination/domain/workflow

46. Grounding query without location words ("Describe") routes to caption not grounding — Current: correct via planner. Keep.
47. Change query with optical+SAR pair misroutes to fusion not temporal — Current: decide_task prefers compound if SAR intent, else temporal; ambiguous 2-image opt+SAR + "what changed" → fusion (wrong). Expected: prefer temporal when change verbs dominate. Detect: yes. Mitigation: planner priority tweak + test (4h). Metric: routing accuracy. Importance HIGH.
48. Agent silent fallback (VLM→heuristic) — Current: fixed via VLM lifecycle flags (good). Keep + UI badge.
49. Confidence 0.82 on heuristic (Demo1) looks calibrated but is not — Current: labeled is_heuristic (good) but UI bars look like probabilities. Expected: rename "evidence strength (uncalibrated)" + never % probability language. Mitigation: 2h UI copy. USP honesty. Metric: ECE (measure later, don't claim now).
50. Hallucinated area in m² — Current: only % (good, no fake meters). Keep; never add meters without GSD.
51. EuroSAT 64px → Cartosat sub-meter shift — Current: untested, no augmentation. Expected: resolution-aware note + augmentation plan. Mitigation: 2-mo tiling/augment. Metric: cross-res accuracy drop. Importance HIGHEST hidden risk.
52. Sentinel RGB → RISAT SAR shift — Current: synthetic SAR tuning will break. Expected: adaptive thresholds + OOD confidence cut. Mitigation: 1wk.
53. Indian geography OOD (Thar vs Kerala vs urban) — Current: no geography awareness. Expected: uncertainty up on OOD, not wrong answer. Mitigation: OOD detector (brightness/veg/texture distance) 1wk.
54. Analyst needs "why" — Current: trace exists (good) but no confidence-factor breakdown per answer. Expected: "why this answer" card (evidence for/against). Mitigation: 1d (expose existing scores). Demo HIGH. USP yes.
55. Batch workflow (100 tiles) — Current: single-call only, no queue. Expected: out-of-scope for SIH; document as non-goal. Importance LOW.

## Priority for 4 days

Must-fix/flag: 36, 40 (contradiction/fallback policy, 1d), 12, 47, 49, 54, 10, 20-warn (not full warp), 22-cloud-heuristic.
Measure-first: 43 (VRSBench IoU sample), 51-52 (resolution stress test on resized EuroSAT).
