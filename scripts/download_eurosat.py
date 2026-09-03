"""
Real RS dataset downloader — EuroSAT RGB (Sentinel-2, 10 classes, 27k, 64x64, ~90MB)
Reproducible, verified, with fallback chain.

Chain:
1. HF blanchon/EuroSAT_RGB via datasets
2. HF giswqs/EuroSAT_RGB
3. Direct ZIP from GitHub/Zenodo via requests

Produces:
  datasets/raw/eurosat/EuroSAT/  (class folders with JPGs)
  datasets/raw/eurosat/metadata.json
  datasets/processed/eurosat_instruct_{N}.json (instruction pairs)

Usage:
  python scripts/download_eurosat.py --subset 1500 --out datasets/raw/eurosat
  python scripts/download_eurosat.py --subset 2000 --instruction --out_processed datasets/processed/eurosat_instruct.json

Storage: 90MB ZIP, ~200MB extracted, instruction JSON ~1-2MB for 1500 samples.
"""
import argparse, os, json, pathlib, sys, shutil, hashlib, zipfile, requests, tempfile
from pathlib import Path

EUROSAT_URLS = [
    "https://zenodo.org/records/7711810/files/EuroSAT.zip?download=1",  # Zenodo, 90MB
    "http://madm.dfki.de/files/sentinel/EuroSAT.zip",  # Original
    "https://github.com/phelber/eurosat/releases/download/v1.0/EuroSAT.zip",
]

HF_DATASETS = ["blanchon/EuroSAT_RGB", "giswqs/EuroSAT_RGB"]

CLASS_NAMES = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
CLASS_DESCRIPTIONS = {
    "AnnualCrop": "annual cropland with seasonal cultivation patterns",
    "Forest": "dense forest with closed tree canopy",
    "HerbaceousVegetation": "herbaceous vegetation, natural grassland",
    "Highway": "highway or major road infrastructure",
    "Industrial": "industrial buildings and associated impervious surfaces",
    "Pasture": "pasture, permanent grassland for grazing",
    "PermanentCrop": "permanent cropland, orchards or vineyards",
    "Residential": "residential buildings, dense settlement",
    "River": "river, linear water body",
    "SeaLake": "sea or lake, large water body",
}

def try_hf_download(out_dir: Path, subset: int):
    try:
        from datasets import load_dataset
        from PIL import Image
        import io
    except ImportError as e:
        print(f"HF datasets not available: {e}")
        return False

    for hf_id in HF_DATASETS:
        try:
            print(f"Trying HF dataset {hf_id} ...")
            ds = load_dataset(hf_id, split="train")
            print(f"Loaded {hf_id}: {len(ds)} samples")
            # Save subset to disk as images + metadata
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_root = out_dir / "EuroSAT"
            raw_root.mkdir(exist_ok=True)
            # Create class folders
            for cls in CLASS_NAMES:
                (raw_root / cls).mkdir(exist_ok=True)

            # Sample stratified
            import random
            random.seed(42)
            # Stratified: take subset // 10 per class
            per_class = subset // len(CLASS_NAMES)
            remainder = subset % len(CLASS_NAMES)
            # Build index by label
            from collections import defaultdict
            idx_by_label = defaultdict(list)
            for i, ex in enumerate(ds):
                label = ex["label"] if "label" in ex else ex.get("label", 0)
                # For blanchon, label is int; for giswqs, same
                idx_by_label[int(label)].append(i)

            selected = []
            for label_idx, cls_name in enumerate(CLASS_NAMES):
                n = per_class + (1 if label_idx < remainder else 0)
                pool = idx_by_label[label_idx]
                if len(pool) < n:
                    print(f"Warning: class {cls_name} has only {len(pool)} samples, need {n}")
                    n = len(pool)
                chosen = random.sample(pool, n)
                selected.extend(chosen)

            random.shuffle(selected)
            print(f"Selected {len(selected)} stratified samples")

            # Save images
            for idx in selected:
                ex = ds[idx]
                label = int(ex["label"])
                cls_name = CLASS_NAMES[label]
                img = ex["image"]
                # HF gives PIL Image already
                filename = ex.get("filename", f"{cls_name}_{idx}.jpg")
                # Ensure filename is just basename, normalize .tif to .jpg
                filename = Path(filename).name
                # If filename is like AnnualCrop_123.tif, convert to .jpg
                if filename.lower().endswith(".tif"):
                    filename = Path(filename).stem + ".jpg"
                elif not filename.lower().endswith((".jpg",".jpeg",".png")):
                    filename += ".jpg"
                save_path = raw_root / cls_name / filename
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(save_path, "JPEG", quality=95)

            # Write metadata
            meta = {
                "source": f"HF {hf_id}",
                "license": "MIT",
                "doi": "10.5281/zenodo.7711810",
                "paper": "https://arxiv.org/abs/1709.00029",
                "total_samples": len(selected),
                "per_class": per_class,
                "image_size": "64x64",
                "bands": "RGB (Sentinel-2 B04,B03,B02)",
                "format": "JPEG",
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"HF download success: {hf_id} -> {out_dir}")
            return True
        except Exception as e:
            print(f"HF {hf_id} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    return False

def try_direct_zip(out_dir: Path, subset: int):
    # Download ZIP via requests, extract, then prune to subset
    import requests
    from tqdm import tqdm

    for url in EUROSAT_URLS:
        try:
            print(f"Trying direct ZIP {url}")
            # HEAD to check
            head = requests.head(url, allow_redirects=True, timeout=10)
            print(f"HEAD {url} -> {head.status_code} {head.headers.get('Content-Length')}")
            # Download with streaming
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp_zip = out_dir / "EuroSAT.zip.tmp"
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(tmp_zip, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading EuroSAT") as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"Downloaded to {tmp_zip}, size {tmp_zip.stat().st_size}")
            # Verify zip
            with zipfile.ZipFile(tmp_zip, "r") as z:
                # Expect EuroSAT/ with class folders
                namelist = z.namelist()
                print(f"ZIP contains {len(namelist)} files, sample: {namelist[:3]}")
                # Extract
                extract_root = out_dir
                # Check if already extracted
                if (extract_root / "EuroSAT").exists():
                    shutil.rmtree(extract_root / "EuroSAT")
                z.extractall(extract_root)
            tmp_zip.unlink(missing_ok=True)
            raw_root = out_dir / "EuroSAT"
            # Verify structure
            for cls in CLASS_NAMES:
                cls_path = raw_root / cls
                if not cls_path.exists():
                    print(f"Missing class folder {cls}")
                    raise FileNotFoundError(f"Missing {cls}")
                print(f"Class {cls}: {len(list(cls_path.glob('*.jpg')))} images")

            # Prune to subset stratified
            import random
            random.seed(42)
            for cls in CLASS_NAMES:
                cls_path = raw_root / cls
                files = list(cls_path.glob("*.jpg"))
                files_sorted = sorted(files)
                # Keep per_class subset, remove rest
                per_class = subset // len(CLASS_NAMES)
                # Keep first per_class after shuffling
                random.shuffle(files_sorted)
                keep = set(files_sorted[:per_class])
                for f in files_sorted:
                    if f not in keep:
                        f.unlink()
                print(f"Pruned {cls}: kept {len(keep)} / {len(files_sorted)}")

            # Metadata
            meta = {
                "source": url,
                "license": "MIT",
                "total_samples": subset,
                "per_class": subset // len(CLASS_NAMES),
                "image_size": "64x64",
                "bands": "RGB",
                "format": "JPEG",
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Direct ZIP success: {url} -> {out_dir}")
            return True
        except Exception as e:
            print(f"Direct ZIP {url} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    return False

def generate_instruction_json(raw_dir: Path, out_path: Path, subset: int):
    """Generate instruction JSON from real EuroSAT images on disk."""
    raw_root = raw_dir / "EuroSAT"
    if not raw_root.exists():
        raw_root = raw_dir
    # Collect files
    import random
    from collections import defaultdict
    files_by_class = defaultdict(list)
    for cls in CLASS_NAMES:
        cls_path = raw_root / cls
        if cls_path.exists():
            for p in cls_path.glob("*.jpg"):
                files_by_class[cls].append(p)
            for p in cls_path.glob("*.jpeg"):
                files_by_class[cls].append(p)
    # Flatten
    all_files = []
    for cls in CLASS_NAMES:
        for p in files_by_class[cls]:
            all_files.append((cls, p))
    # Shuffle
    random.seed(42)
    random.shuffle(all_files)
    # Take subset
    all_files = all_files[:subset]
    print(f"Generating instruction JSON for {len(all_files)} files")

    # Create train/val split 83/17 stratified
    by_cls = defaultdict(list)
    for cls, p in all_files:
        by_cls[cls].append(p)
    train_entries = []
    val_entries = []
    for cls in CLASS_NAMES:
        lst = by_cls[cls]
        random.shuffle(lst)
        n_val = max(1, int(len(lst) * 0.17))
        n_train = len(lst) - n_val
        for p in lst[:n_train]:
            train_entries.append((cls, p))
        for p in lst[n_train:]:
            val_entries.append((cls, p))

    def make_entries(pairs, split):
        entries = []
        for cls, p in pairs:
            rel_path = str(p.relative_to(raw_dir)) if p.is_relative_to(raw_dir) else str(p)
            # Make path relative to project root for training
            # Use forward slash
            rel_path = rel_path.replace("\\", "/")
            # Templates
            # Template 1: captioning
            caption = f"This Sentinel-2 image shows {cls}, characterized by {CLASS_DESCRIPTIONS[cls]}. Land-cover class: {cls}."
            entries.append({
                "image": rel_path,
                "query": "Describe the land-cover in this Sentinel-2 image.",
                "answer": caption,
                "label": cls,
                "label_id": CLASS_NAMES.index(cls),
                "split": split,
                "source": "EuroSAT",
            })
            # Template 2: VQA classification
            entries.append({
                "image": rel_path,
                "query": "What land-cover class is visible in this Sentinel-2 image?",
                "answer": cls,
                "label": cls,
                "label_id": CLASS_NAMES.index(cls),
                "split": split,
                "source": "EuroSAT",
            })
        return entries

    train_data = make_entries(train_entries, "train")
    val_data = make_entries(val_entries, "val")
    all_data = train_data + val_data
    random.shuffle(all_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Instruction JSON: {len(train_data)} train + {len(val_data)} val = {len(all_data)} entries -> {out_path}")
    # Also write a summary
    print(f"Example train entry: {train_data[0] if train_data else 'none'}")
    print(f"Example val entry: {val_data[0] if val_data else 'none'}")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=1500, help="Total images to keep (stratified)")
    parser.add_argument("--out", type=str, default="datasets/raw/eurosat", help="Raw out dir")
    parser.add_argument("--out_processed", type=str, default=None, help="Processed instruction JSON")
    parser.add_argument("--force", action="store_true", help="Force redownload even if exists")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_processed = Path(args.out_processed) if args.out_processed else Path("datasets/processed") / f"eurosat_instruct_{args.subset}.json"

    # Check if already exists and not force
    if (out_dir / "EuroSAT").exists() and not args.force:
        # Count files
        total = sum(1 for _ in (out_dir / "EuroSAT").rglob("*.jpg"))
        print(f"Found existing EuroSAT at {out_dir} with {total} images, skipping download (use --force to redownload)")
        if total >= args.subset * 0.9:
            # Generate instruction JSON if not exists
            if not out_processed.exists():
                generate_instruction_json(out_dir, out_processed, args.subset)
            else:
                print(f"Instruction JSON already exists: {out_processed}")
            return

    # Try HF first
    success = try_hf_download(out_dir, args.subset)
    if not success:
        print("HF download failed, trying direct ZIP...")
        success = try_direct_zip(out_dir, args.subset)
    if not success:
        print("All download methods failed.")
        print("Fallback: creating synthetic placeholder for verification (marked synthetic)")
        # Do not create fake manifest pointing to nonexistent files — create empty structure
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metadata.json", "w") as f:
            json.dump({"source": "failed", "error": "download failed, synthetic fallback not allowed for real claim", "status": "PLACEHOLDER"}, f, indent=2)
        sys.exit(1)

    # Generate instruction JSON
    generate_instruction_json(out_dir, out_processed, args.subset)
    print(f"Done: raw {out_dir}, processed {out_processed}")

    # Verify
    print("\n=== Verification ===")
    raw_root = out_dir / "EuroSAT"
    for cls in CLASS_NAMES:
        cnt = len(list((raw_root / cls).glob("*.jpg"))) if (raw_root / cls).exists() else 0
        print(f"  {cls}: {cnt}")
    total = sum(1 for _ in raw_root.rglob("*.jpg"))
    print(f"Total images: {total}")
    # Check JSON
    with open(out_processed) as f:
        data = json.load(f)
    print(f"Instruction entries: {len(data)} (expected ~{args.subset*2})")
    # Verify files exist
    missing = 0
    for entry in data[:5]:
        p = out_dir / entry["image"] if not Path(entry["image"]).is_absolute() else Path(entry["image"])
        # Also try relative to raw_dir
        if not p.exists():
            # Try alternative path
            alt = Path("datasets/raw/eurosat") / entry["image"]
            if not alt.exists():
                missing += 1
                print(f"Missing file for entry: {entry['image']}")
    if missing == 0:
        print("First 5 entries: files exist")
    print("Verification complete.")

if __name__ == "__main__":
    main()
