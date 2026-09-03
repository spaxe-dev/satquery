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

# REAL RS data — EuroSAT RGB (recommended minimal, 90MB, verified 2026-09-04)
python scripts/download_eurosat.py --subset 1500 --out datasets/raw/eurosat
# Verify real loader
python -c "from training.dataset_loader import verify_real_dataset; verify_real_dataset('datasets/processed/eurosat_instruct_1500.json','datasets/raw/eurosat')"
python training/train_lora_real.py --dataset eurosat --subset 1500 --dry_run

# Alternative: BigEarthNet subset (110GB full, use mini/medium via jerpint/bigearthnet)
python scripts/download_datasets.py --dataset bigearthnet --subset 1000 --out datasets/raw
python scripts/prepare_datasets.py --dataset synthetic --subset 2000
python scripts/verify_datasets.py
```

## 7. Train

```bash
# Dry-run with REAL EuroSAT loader (no GPU, verifies pipeline):
python training/train_lora_real.py --dataset eurosat --subset 1500 --dry_run
python training/train_lora.py --dataset eurosat --subset 1500 --dry_run  # delegates to real trainer

# Real EuroSAT on RTX 4090 24GB — BLIP minimal (6-8GB, 15-20min, checkpoint ~15MB):
python training/train_lora_real.py --dataset eurosat --subset 1500 --model Salesforce/blip-image-captioning-base --epochs 2 --batch_size 4 --gradient_accumulation 4 --lr 2e-4 --output models/checkpoints/satquery-eurosat-blip-lora

# Real EuroSAT Qwen2-VL QLoRA (12-14GB, 25-35min):
python training/train_lora_real.py --dataset eurosat --subset 1500 --model Qwen/Qwen2-VL-2B-Instruct --use_4bit --epochs 1 --batch_size 1 --gradient_accumulation 8 --lr 1e-4 --output models/checkpoints/satquery-eurosat-qwen-lora

# Legacy synthetic (for testing only, not real RS data):
python training/train_lora.py --dataset synthetic --subset 2000 --epochs 2 --model Salesforce/blip-image-captioning-base --batch_size 1 --lr 2e-4
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
