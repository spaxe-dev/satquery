"""Prepare datasets for training — subsetting, preprocessing, manifests."""
import argparse, json, os
from pathlib import Path

def prepare_synthetic(subset=500, out="datasets/processed/synthetic_demo.json"):
    import sys
    sys.path.insert(0, "training")
    from scripts_synthetic import make_synthetic_dataset
    make_synthetic_dataset(subset, out)
    print(f"Done: {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic","bigearthnet","rsvqa","cdvqa"], default="synthetic")
    p.add_argument("--subset", type=int, default=500)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.dataset == "synthetic":
        out = args.out or "datasets/processed/synthetic_demo.json"
        prepare_synthetic(args.subset, out)
    else:
        # Call download scaffold then prepare
        import subprocess, sys
        print(f"Preparing {args.dataset} — using download scaffold subset {args.subset}")
        out = args.out or f"datasets/processed/{args.dataset}_subset_{args.subset}.json"
        # Reuse synthetic as placeholder structure
        prepare_synthetic(args.subset, out)
        print(f"NOTE: {args.dataset} real preparation requires full download. Placeholder at {out}")

if __name__ == "__main__":
    main()
