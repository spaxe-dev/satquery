#!/usr/bin/env bash
set -e
echo "=== SatQuery Cloud GPU Setup ==="
nvidia-smi || echo "nvidia-smi not found — check GPU"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" || echo "torch not installed"
echo "--- Installing deps ---"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes datasets evaluate scikit-learn pillow numpy opencv-python rasterio
echo "--- Verifying ---"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
echo "--- Ready. Run: bash scripts/train.sh --dataset synthetic --subset 500 --epochs 1 --use_4bit ---"
