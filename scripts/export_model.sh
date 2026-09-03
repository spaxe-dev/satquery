#!/usr/bin/env bash
echo "Export model artifacts"
ls -lh models/checkpoints/
echo "Copy to project storage: ensure checkpoints are not on ephemeral disk"
