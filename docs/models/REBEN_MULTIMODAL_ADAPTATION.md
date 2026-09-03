# reBEN Multimodal Adaptation Architecture — SatQuery AI

**Status:** Design for Stage 1 (100 S1+S2 pairs) — preserves successful BLIP+LoRA pipeline, adds S1+S2 fusion adapter.

## Architecture Diagram (text)

```
reBEN S1 (2,120,120) VV/VH
        |
        v
  S1 Encoder (frozen, from reBEN S1 model or simple patch embed)
        |
        +----> S1 Proj (trainable, 1x1 conv 2->768) -> S1 embedding (768)
        |
reBEN S2 (10,120,120) B02..B12
        |
        v
  S2 Encoder (frozen, BLIP vision encoder or reBEN S2 model)
        |
        +----> S2 Proj (trainable, 1x1 conv 10->768) -> S2 embedding (768)
        |
        +---- Fusion: concat/add + MLP (trainable, 768*2 -> 768) -> fused visual embedding (768, 384x384? Actually 577 tokens for ViT)
        |
        v
  BLIP Vision Encoder (frozen, 12 layers, qkv) -> already has S1/S2 fused input, so we can bypass and directly feed fused embedding to text_decoder
        |
        v
  BLIP Text Decoder (BERT, 12 layers, query/key/value) — LoRA r=8 on query/value (589k trainable, as verified in smoke test)
        |
        v
  Caption: "This Sentinel-2 image (BigEarthNet, mini) contains: {labels}" (19-class multilabel, real reBEN labels)
        |
        v
  Cross-entropy loss on caption tokens
```

**Simplified defensible version for Stage 1 (100 pairs, CPU smoke):**

```
S1 (2,120,120) --\
                  +-- Channel Proj (trainable: S1 2->3 via 1x1 conv, S2 10->3 via 1x1 conv) -> RGB-like (3,120,120) each
S2 (10,120,120) --/               |
                                 +-- Weighted sum (trainable scalar alpha for S1 vs S2) -> fused RGB (3,120,120)
                                                        |
                                                        v
                                            BLIP Processor (resize 120->384, normalize)
                                                        |
                                                        v
                                            BLIP Vision Encoder (frozen)
                                                        |
                                                        v
                                            BLIP Text Decoder (LoRA query/value, 0.26% trainable)
                                                        |
                                                        v
                                            Caption from 19 labels
```

Even simpler for smoke test (what we will implement first):

```
S1 (2,120,120) -> take VV band, repeat to 3 channels -> (3,120,120) -> BLIP
S2 (10,120,120) -> take B04/B03/B02 (RGB) -> (3,120,120) -> BLIP
Fusion: average S1_RGB and S2_RGB -> fused (3,120,120) -> BLIP
```

This preserves BLIP's expected 3-channel input, allows S1+S2 to reach the model, and keeps vision encoder frozen.

## Tensors/Shapes

- **S1 raw:** `(2, 120, 120)` uint16 or float32, bands [VV, VH], values in dB or linear, will be normalized to 0-1.
- **S2 raw:** `(10, 120, 120)` uint16 0-10000, bands [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12], will be `/10000.0` -> float32 0-1.
- **After channel proj (trainable 1x1 conv):** S1 `(3,120,120)`, S2 `(3,120,120)` — each via `nn.Conv2d(2,3,1)` and `nn.Conv2d(10,3,1)` (trainable, ~6*3 + 30*3 = ~108 params, negligible)
- **Fused:** `(3,120,120)` — `fused = alpha * S1_RGB + (1-alpha) * S2_RGB` where `alpha` is trainable scalar (initialized 0.3 for S1, 0.7 for S2)
- **After BLIP processor:** `(3,384,384)` float32, normalized per BLIP mean/std
- **After BLIP vision encoder:** `(577,768)` tokens (including CLS)
- **Text decoder input_ids:** `(batch, 64)` tokens from caption
- **Loss:** `CrossEntropyLoss` on `labels` (input_ids)

## Fusion Point

- **Late fusion (chosen for Stage 1):** Fuse at **RGB image level before BLIP processor** (weighted sum of S1_RGB and S2_RGB). This is simplest, preserves BLIP's 3-channel expectation, and keeps vision encoder frozen.
- **Alternative (future):** Early fusion via `S1 Proj + S2 Proj -> 768` and feed directly to text decoder's cross-attention, bypassing vision encoder — more powerful but requires more code.

## Trainable vs Frozen

- **Frozen (majority):** BLIP vision encoder (12 layers, `qkv` etc., 86M params), BLIP text decoder base (except LoRA), S1/S2 band statistics.
- **Trainable (minimal):**
  - S1 channel proj: `Conv2d(2,3,1)` — 9 params
  - S2 channel proj: `Conv2d(10,3,1)` — 33 params
  - Fusion alpha: scalar — 1 param
  - LoRA on text_decoder `query/value`: `r=8, alpha=16` → **589,824 params** (0.26% of 224M, verified in smoke test)
  - Total trainable: **~589,867** (dominated by LoRA, as in successful smoke test)

## Loss

- **Captioning:** `CrossEntropyLoss` on caption tokens (teacher forcing, `labels` = `input_ids` from processor)
- **Caption example:** `"This Sentinel-2 image (BigEarthNet, mini) contains: Coniferous forest, Pastures."` — derived from real reBEN 19-class labels (e.g., `["Coniferous forest", "Pastures"]`), not synthetic.
- **No fake VQA:** We do not invent VQA pairs; we use captioning which is valid for multilabel.

## Rationale

1. **Preserves successful BLIP pipeline:** The smoke test proved `BlipForConditionalGeneration` with `model.text_decoder` LoRA works (forward, backward, save, reload). This design keeps that exact pipeline, just adds a tiny fusion adapter before the processor.
2. **Allows S1+S2 to reach the model:** Both modalities are converted to RGB-like and fused, so SAR (S1) information (VV/VH backscatter) and optical (S2) both influence the caption.
3. **Trainable on our hardware:** Only ~590k params trainable, fits on CPU for smoke (8 samples, 1 step) and on RTX 4090 24GB for 100-1000 pairs (6-8GB VRAM, same as EuroSAT).
4. **Does not require training entire BLIP:** Vision encoder frozen, only fusion proj + LoRA.
5. **Can be integrated into SatQuery:** The `ReBENMultimodalDataset` returns `s1` and `s2` separately, and the `SatQuery-RS-VLM` tool can be extended to call the fusion adapter when both modalities are present, else fallback to heuristic.

## Limitations

- **BGR→RGB and 10→3 projection loses spectral information** (10 S2 bands collapsed to 3). For a stronger model, we would keep all 10 bands and use a dedicated S1/S2 encoder, but for Stage 1 (100 pairs), the 3-channel fusion is sufficient to prove S1+S2 reach the model.
- **No SAR-specific encoder:** S1 VV/VH are treated as just 2 channels, not as complex SAR. Future work could add a SAR-specific adapter (e.g., `S1_Adapter` with despeckle).
- **Caption is multilabel, not free-form VQA:** This is intentional to avoid fake VQA; the downstream SatQuery can still answer "What land-cover classes are visible?" by parsing the caption.
- **120→384 upsample:** BLIP processor upsamples 120 to 384, which is okay but loses some detail.

## Integration Boundary

- **Existing:** `BigEarthNet mini → BLIP+LoRA smoke path` (BGR 120, 43 labels, Hub, already verified, remains)
- **New:** `reBEN S1+S2 → multimodal adapter (S1proj+S2proj+alpha) → BLIP+LoRA` (kept behind `training/reben_multimodal_dataset.py` and `docs/models/REBEN_MULTIMODAL_ADAPTATION.md`, not yet wired into production `tools/vqa_tool.py` — that will be done after smoke test passes)

## Next Steps

- Stage 1: 100 S1+S2 pairs via `reben_v2_subset_manifest.json` (already), loader `ReBENMultimodalDataset`, fusion adapter, smoke test 8-16 real pairs (forward/backward with S1+S2 fused, not just S2).
- Stage 2: 1,000 pairs (download 1k S1+S2 patches, ~200MB, on D:)
- Stage 3: 5k-10k pairs (~1-2GB)

No paid GPU until Stage 1 8-sample smoke with S1+S2 fused passes.
