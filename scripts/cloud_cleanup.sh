#!/usr/bin/env bash
echo "Cloud cleanup — ensure checkpoints copied!"
ls -lh models/checkpoints/satquery-lora/ 2>/dev/null || echo "no checkpoint"
echo "Shut down instance manually in cloud console after verifying artifacts."
