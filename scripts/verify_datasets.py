"""Verify datasets — checks manifests, counts, sample loads."""
import json, os
from pathlib import Path

def verify(path):
    p = Path(path)
    if not p.exists():
        print(f"MISSING: {path}")
        return False
    with open(p) as f:
        data = json.load(f)
    print(f"OK: {path} — {len(data)} entries")
    # Sample
    for i, entry in enumerate(data[:2]):
        print(f"  sample {i}: {entry}")
    return True

if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] if len(sys.argv)>1 else ["datasets/processed/synthetic_demo.json"]
    for p in paths:
        verify(p)
    # Also check demo data
    demo_base = Path("data/demo")
    if demo_base.exists():
        for root, dirs, files in os.walk(demo_base):
            level = root.replace(str(demo_base), "")
            for f in files:
                print(f"  demo file: {os.path.join(level, f)}")
