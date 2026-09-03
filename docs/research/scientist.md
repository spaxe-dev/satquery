# Domain & Conceptual Critic Report — SIH26167 SatQuery AI

## domain_summary

This problem belongs to **remote sensing science / Earth observation (EO) image analysis**, sitting at the intersection of:
- **Multimodal sensor physics** — optical/multispectral radiometry vs. synthetic aperture radar (microwave backscatter imaging)
- **Geodesy/geomatics** — georeferencing, co-registration, CRS, spatial resolution matching
- **Change detection theory** — pseudo-change vs. real change, radiometric normalization between acquisition dates
- **Vision-language modelling** adapted to a domain where the "vocabulary" is spectral bands, polarization, incidence angle, GSD — not natural photos
- **Agentic orchestration / model routing** as the software layer on top

The core scientific claim in the PS is sound and important: SAR and optical observe *physically different quantities* (backscatter cross-section vs. surface reflectance). A VLM fine-tuned only on RGB natural images has no basis for interpreting speckle statistics, polarization signatures, or band ratios. The team must internalize that this is fundamentally an **information-fusion problem under heterogeneous sensor physics**, not a chatbot problem.

## required_concepts

1. **SAR fundamentals**: side-looking geometry (layover, foreshortening, shadow), speckle (coherent interference — multiplicative noise, NOT Gaussian), sigma-naught calibration, incidence angle dependence of backscatter, polarization channels (VV/VH for Sentinel-1; RISAT-1 is C-band multi-pol)
2. **Optical physics**: reflectance vs. digital numbers, atmospheric effects, cloud masking, sun elevation/shadow variation across dates
3. **Co-registration**: sub-pixel geometric alignment is a *precondition* for any pixel-wise change detection or optical–SAR fusion; misregistration produces false change
4. **Radiometric/illumination normalization**: bi-temporal optical pairs differ in illumination and phenology — naive pixel differencing detects season, not change ("pseudo-change")
5. **Resolution mismatch**: BigEarthNet patches are 10m Sentinel-1/Sentinel-2; Cartosat-2S pan is ~0.65m and RISAT MRS/CRS are ~18m/~3.6m (approximate modes). The hidden SAC set lives at completely different scales than every training benchmark.
6. **Land-cover semantics from SAR**: water appears dark (specular reflection), built-up areas bright (double-bounce), vegetation moderate volume scattering — but these are probabilistic, not deterministic; bare soil can mimic built-up in C-band.
7. **VQA evaluation mechanics**: closed-vocabulary accuracy (CDVQA OA) vs. open-vocabulary caption metrics (CIDEr/BLEU) vs. grounding IoU — each measures different things and normalizing them (as the PS says) is nontrivial.

## important_terminology

- GSD (ground sample distance), swath, TDI, sun-synchronous orbit
- Backscatter, σ⁰/γ⁰, speckle, layover/shadow, double-bounce, volume scattering, VV/VH polarization, C-band (~5.4 GHz)
- Co-registration, geotransform, CRS/UTM, ortho-rectification, radiometric calibration/atmospheric correction
- Bi-temporal vs. multitemporal; semantic vs. binary change detection; change-VQA
- Fine-tuning/domain adaptation, LoRA, instruction tuning, evidence grounding
- CDLA labels in BigEarthNet: 19-class CORINE-derived nomenclature (multi-label!)

## conceptual_claims

- **Claim (Stage 1):** "BigEarthNet-MM perfectly matches the optical–SAR requirement." — **Partially true but oversimplified.** BEN provides S1/S2 *pairs with multi-label scene classes*, not paired-image QA or region-level correspondence. Template-generated instructions from scene labels give weak supervision: the model learns label vocabulary, not genuine cross-modal reasoning about *where* things are.
- **Claim:** "SSIM/correlation threshold for co-registration check" — **Conceptually shaky.** Optical–SAR pairs have structurally different appearances; SSIM between modalities is near-meaningless. Cross-modal registration needs mutual-information or learned/feature-based methods (or trust provided geotransforms). For bi-temporal optical it's more defensible but still crude.
- **Claim:** "Modality heuristic: single-band, non-natural statistics = SAR" — acceptable engineering, but note multispectral single-band exports exist; better signal is metadata + speckle statistics (Rayleigh-distributed intensity).
- **Claim:** "~74% OA achievable via Qwen3.x LoRA on CDVQA" — plausible per recent literature, but CDVQA answers are template-generated questions over SECOND imagery (aerial, ~0.5–5m); transferring that to Cartosat/RISAT pairs is unproven.

## domain_constraints

1. **Resolution/sensor gap cannot be trained away without data.** No public dataset pairs Cartosat-2S optical with RISAT-1 SAR with text annotations. Generalization from Sentinel 10m → sub-meter optical + medium-res C-band SAR is the single largest scientific unknown.
2. **SAR speckle destroys standard visual tokenizers.** Feeding raw SAR patches into an RGB-pretrained ViT/VLM is physically incoherent without despeckling (e.g., Lee/refined-Lee) or SAR-specific preprocessing (log-amplitude, calibration).
3. **Change ≠ difference.** Any bi-temporal answer must control for: different sun angles, seasonal vegetation state, soil moisture affecting SAR backscatter (moisture change looks like land-cover change in SAR!), and registration error.
4. **BigEarthNet labels are multi-label scene-level** — you can't infer object locations or counts from them; using BEN to teach "grounding" is invalid inference.
5. **Cloud cover asymmetry is the point of SAR** — the system should be able to say "optical unusable here, relying on SAR," which requires explicit usability assessment, not just routing.
6. **Evaluation set is undisclosed and ISRO-specific** — teams optimize blind; robustness across sensors matters more than leaderboard-tuning.

## conceptual_risks

1. **Fatal risk:** treating optical–SAR fusion as "two images concatenated into a VLM." Without encoding what backscatter means, the model will hallucinate plausible-sounding but physically wrong answers (e.g., calling dark SAR fields "water" when they're smooth bare soil).
2. **Pseudo-change hallucination:** change-description outputs driven by illumination/season artifacts rather than real change — judges at SAC will detect this instantly.
3. **Correlation/causation trap:** CDVQA-style question templates are auto-generated; models can score well by exploiting template priors (e.g., answering "increase") rather than reading pixels. High benchmark score may not transfer to SAC data.
4. **Confidence estimation is scientifically undefined** in the PS — verbalized VLM confidence is poorly calibrated; claiming "confidence scores" without calibration methodology is hand-waving.
5. **GeoTIFF handling misconstrued as trivial:** real products are tiled, compressed, possibly with RPCs, multiple overviews, no-data masks — naive PIL/rasterio reads will fail on SAC-supplied files.

## critical_questions

1. What physical quantity does each input pixel represent (reflectance? calibrated σ⁰? raw DN?) — and does the pipeline care?
2. What is ground truth for "change"? Semantic class transition, binary change mask, or textual description consistency? Who adjudicates ambiguous changes?
3. Can the system distinguish "no information available" (e.g., total cloud cover) from "no change detected"? Failure-mode honesty may be scored.
4. How will co-registration quality be verified before any pair analysis, and what happens if inputs are misregistered by >1 pixel?
5. Which SAC-provided image levels will be given (ortho-ready L2? raw?) and what radiometric normalization is expected upstream vs. in-solution?
6. Is soil-moisture-induced SAR brightness change expected to be reported as "change"? What does the reference answer say for such cases?
7. How are the public benchmarks' answer distributions being used — is the model learning templates or images? (Test with adversarial rephrasing.)
8. What confidence definition is used, and is it calibrated against held-out accuracy?

## disputed_claims

- **Dispute Stage 1's "6–10 week MVP buildable" timeline optimism** specifically for the optical–SAR component: the science there (cross-modal alignment + text generation with zero paired-text training data) is closer to an open research problem than an integration task.
- **Dispute SSIM-based co-registration checking** as stated above.
- Stage 1's claim that BigEarthNet-MM "perfectly matches" is too rosy — see conceptual_claims.

## evidence

- Cartosat-2S: 0.65m PAN / <2m MS, ~9.6km swath (vendor specs, ScienceDirect overview) — confirming the severe resolution mismatch with Sentinel-based training sets.
- CDVQA: Yuan et al., arXiv:2112.06343 — auto-generated QA over SECOND aerial bi-temporal pairs; confirms template-generation concern; GitHub (YZHJessica/CDVQA) exists so access is likely feasible.
- CDQAG (like413.github.io/CDQAG): extends change-VQA with grounding — supports Stage 1's VisTA direction.
- Established RS literature consensus: SAR speckle is multiplicative coherent noise requiring dedicated filtering; optical–SAR interpretation differences are documented throughout the SEN1-2 / BigEarthNet-MM papers.

## revised_position

My position aligns with Stage 1's PURSUE verdict but with sharpened conditions: the project is scientifically viable **only if** (a) SAR inputs get physically informed preprocessing (calibration-aware, despeckled) rather than being treated as grayscale photos, (b) the optical–SAR tool is scoped honestly — classification/evidence-map level outputs rather than pretending deep cross-modal dialogue reasoning, and (c) the team tests for template-shortcut learning on CDVQA before trusting benchmark scores. The hidden SAC set punishes exactly the shortcuts that win public benchmarks; design for sensor-level generalization (resolution augmentation, simulated speckle, scale jittering) from day one.