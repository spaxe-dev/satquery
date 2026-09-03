#!/usr/bin/env bash
echo "Evaluate benchmarks (requires datasets)"
python -m pytest tests/test_end_to_end.py -v
python scripts/verify_datasets.py datasets/processed/synthetic_demo.json
