# Model Evaluation (2026-09-05, inference-only, no training)

| Model | Checkpoint | License | Size | CPU/Win | Modalities | VQA/Cap/Ground/Temporal | Verdict |
|---|---|---|---|---|---|---|---|
| BLIP-base caption+VQA | `Salesforce/blip-image-captioning-base`, `blip-vqa-base` | BSD-3 | 223M ~1GB | RUNNABLE CPU ~1GB 20-60s | optical RGB only | VQA yes (vqa-base), cap yes, ground NO, temporal NO | INTEGRATED gated `SATQUERY_ENABLE_VLM=1`, optical-only |
| Qwen2-VL-2B | `Qwen/Qwen2-VL-2B-Instruct` | Apache-2 | 2.2B ~4.5GB | NEEDS GPU (CPU only via Ollama Q4) | optical multi-image | VQA/cap/rough grounding, no SAR | SCAFFOLD, not integrated (no GPU path verified) |
| GeoChat-7B | `MBZUAI/geochat-7B` | Apache-2 (Vicuna non-commercial inherit) | 7B ~14GB | NEEDS GPU 14GB | optical single 504px | VQA/cap/grounding text-coords | BLOCKED CPU |
| EarthMind-4B | `sy1998/EarthMind-4B` | Apache-2 | 4B ~8GB | NEEDS 16-24GB VRAM | optical+SAR+fusion (only SAR-native) | VQA/cap/ground/seg | BLOCKED CPU |
| TEOChat-7B | `jirvin16/TEOChat` | Apache-2 | 7B ~13GB | NEEDS CUDA | temporal optical sequences | temporal QA/change + VQA | BLOCKED CPU |
| BLIP-2/LLaVA-7B | various | BSD/llama2 | 3-7B | NOT-PRACTICAL CPU | optical | VQA/cap | NOT INTEGRATED |

Selection: BLIP-base only CPU-runnable with transformers CPU wheel. SAR/temporal/grounding stay heuristic by capability guard. RS models deferred to GPU service. No capabilities claimed from papers; grounding explicitly unsupported in `BlipAdapter.ground()`.
