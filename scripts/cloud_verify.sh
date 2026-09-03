#!/usr/bin/env bash
echo "=== Cloud Verify ==="
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('mem', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB' if torch.cuda.is_available() else '')"
python -c "import transformers, peft; print('transformers', transformers.__version__); print('peft ok')"
ls -lh datasets/processed/ 2>/dev/null || echo "no processed datasets yet"
ls -lh models/checkpoints/ 2>/dev/null || echo "no checkpoints yet"
