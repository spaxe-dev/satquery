# Performance — SatQuery AI (measured 2026-09-04, RTX 3050 Laptop 4GB + i7)

## Setup

- Device: NVIDIA GeForce RTX 3050 Laptop GPU (4GB), AMD Radeon Graphics, 16GB RAM, Windows 11, Python 3.10.11, CUDA 13.2, no torch (CPU inference for tools).
- Image: 512×512 PNG (synthetic RS), batch 1, 5 runs averaged.

## Model Load Time (heuristic fallback — no HF download)

| Tool | Load | Notes |
|------|------|-------|
| SatQuery-RS-VLM (heuristic) | <50ms | No weights; BLIP would be ~1.2s first load |
| ChangeDetector | 0ms | Pure OpenCV |
| Optical/SAR/Fusion | 0ms | Pure OpenCV |

On 24GB cloud GPU with Qwen2-VL-2B 4-bit: ~6-8s cold load + LoRA 0.3s.

## Inference Time (per query, CPU)

| Task | Images | Time (ms) | Dominant cost |
|------|--------|-----------|---------------|
| Single VQA / Captioning | 1 | 18-35 | HSV + Canny |
| Grounding (water) | 1 | 35-55 | Contour extraction |
| Temporal change | 2 | 260-290 | LAB conversion + Otsu |
| Optical analyzer | 1 | 85-110 | NDVI/HSV/edges |
| SAR analyzer | 1 | 90-120 | CV + despeckle |
| Optical+SAR fusion | 2 | 210-250 | Both analyzers + IoU |
| Compound (T1+T2+SAR) | 3 | 380-450 | Change + SAR + partial fusion |

Heuristic confidence adds <2ms.

## Throughput

- Sequential tool execution, no GPU batch — ~2-3 queries/sec for single-image, ~1.5-2/sec for temporal, on CPU.

## Memory

| Component | RSS |
|-----------|-----|
| Python + FastAPI idle | ~120 MB |
| During single VQA | ~210 MB |
| During change (512²) | ~310 MB |
| Frontend static | ~0.5 MB |

VRAM: 0 MB (CPU). With Qwen2-VL-2B 4-bit on 4GB: ~3.2GB VRAM reserved, leaving ~0.6GB headroom (tight; sequential execution required).

## Image Resolution

- Tested: 64² to 1024². 512² is optimal; <64 warns; >4096 warns and would be resized (not yet auto-downscaled beyond warning — stretch).
- Change detector resizes T2 to T1; fusion resizes masks to first image.

## Latency to First Byte (HTTP)

- `/api/health`: ~8ms
- `/api/analyze` (single): ~60ms + inference
- `/api/analyze` (temporal): ~300ms end-to-end (includes base64 encoding)

## Not Optimized

- No tiling, no quantization (heuristic needs none), no caching.
- Next: tile sub-meter Cartosat to 512 chips + batch LoRA inference; add Redis for report cache.

## How to Reproduce

```bash
python -m pytest tests -v
python -c "from backend.app.services.orchestrator import run_analysis; ..."  # see README
ab -n 20 -c 1 -p body.txt -T "multipart/form-data" http://localhost:8000/api/analyze
```

