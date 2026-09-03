# Cloud GPU Setup — SatQuery AI

Target: single-GPU, CUDA-capable, 24GB (RTX 4090 / A5000 / A100) for training; demo runs on 4GB laptop.

Any provider works (RunPod, Vast.ai, Lambda, Paperspace, GCP, AWS). The app expects only `CUDA_VISIBLE_DEVICES` and PyTorch CUDA.

## 1. Create Instance

- Provider console → New GPU instance → 24GB VRAM, Ubuntu 22.04, CUDA 12.1+, 50GB disk.
- Open ports 8000 (app) and 22 (ssh) if needed.

## 2. Connect & Verify

```bash
ssh user@<instance-ip>
nvidia-smi
nvcc --version
python --version   # 3.10+
```

## 3. Verify GPU & Env (or use script)

```bash
git clone <repo-url> && cd satquery
bash scripts/cloud_verify.sh
# Manual:
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Install

```bash
bash scripts/cloud_setup.sh
# Expands to:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes datasets evaluate scikit-learn pillow numpy opencv-python rasterio reportlab
# Plus base deps
pip install fastapi uvicorn python-multipart pillow numpy opencv-python scikit-learn scikit-image rasterio shapely reportlab pyyaml requests
```

## 5. Authenticate (if needed)

```bash
huggingface-cli login  # for Qwen2-VL downloads
# or: export HF_TOKEN=...
```

## 6. Download / Prepare Data

```bash
python scripts/prepare_demo_data.py
python scripts/download_datasets.py --list
python scripts/download_datasets.py --dataset bigearthnet --subset 2000 --out datasets/raw
python scripts/prepare_datasets.py --dataset synthetic --subset 2000
python scripts/verify_datasets.py
```

## 7. Train

```bash
# Dry-run (no GPU):
python training/train_lora.py --dry_run --subset 500

# Real (QLoRA 4-bit, fits 24GB):
python training/train_lora.py --dataset synthetic --subset 2000 --epochs 2 --model Salesforce/blip-image-captioning-base --use_4bit --batch_size 1 --lr 2e-4

# For Qwen2-VL-2B:
python training/train_lora.py --dataset synthetic --subset 1000 --epochs 1 --model Qwen/Qwen2-VL-2B-Instruct --use_4bit
```

Checkpoints → `models/checkpoints/satquery-lora/`.

## 8. Evaluate & Export

```bash
bash scripts/evaluate.sh
python -m pytest tests -v
bash scripts/export_model.sh
ls -lh models/checkpoints/satquery-lora/
```

## 9. Run App on Cloud (optional)

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
# Open http://<instance-ip>:8000
```

## 10. Transfer Artifacts (before shutdown!)

```bash
# From local:
scp -r user@<instance-ip>:~/satquery/models/checkpoints/satquery-lora ./models/checkpoints/
# Or use provider's volume snapshot
ls -lh models/checkpoints/satquery-lora/  # verify not empty
```

## 11. Cleanup / Shutdown

```bash
bash scripts/cloud_cleanup.sh
# Then in provider console: Stop or Terminate instance
# Verify artifacts copied first — ephemeral disk is lost on terminate
```

## VRAM Notes

- RTX 3050 Laptop 4GB: heuristic-only demo; BLIP-base fits but training is slow; Qwen2-VL-2B 4-bit is tight.
- RTX 4090 24GB: full QLoRA 2B training + inference comfortably; can serve all 5 tools sequentially.
- If OOM: reduce batch_size=1, enable 4-bit, gradient_accumulation=8, reduce resolution to 256.

## Cost Estimate (indicative, 2026)

- 4090 rental: ~₹120-200/hr on Indian spot markets; a 2-epoch run on 1k samples ≈ 30-60 min.
- Budget ₹2k-4k for a full adaptation experiment.
