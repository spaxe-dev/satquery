# QUICKSTART — SatQuery AI

Get from zero to hero demo in <5 minutes.

## 1. Prerequisites

- Python 3.10+ (tested 3.10.11)
- 4GB VRAM laptop is fine (heuristic fallback); 24GB for LoRA training
- Windows or Linux

## 2. Clone / Open

```bash
# If using git
git clone <repo-url>
cd satquery
# Or just open D:\Projects\satquery
```

## 3. Install

```powershell
pip install fastapi uvicorn python-multipart pillow numpy opencv-python scikit-learn scikit-image rasterio shapely reportlab pyyaml requests
# For training (cloud GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes
```

## 4. Demo Data (no downloads needed)

```bash
python scripts/prepare_demo_data.py
# Creates data/demo/{single_image,temporal_pairs,optical_sar_pairs,invalid_inputs}
python scripts/verify_datasets.py
python -m pytest tests -v   # 25 tests should pass
```

## 5. Start

Windows:
```powershell
.\scripts\start.ps1
# or directly:
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Linux:
```bash
./scripts/start.sh
```

## 6. Open

- App: http://localhost:8000
- Health: http://localhost:8000/api/health
- Docs: http://localhost:8000/docs
- Registry: http://localhost:8000/api/registry

## 7. Run Demos (in UI)

- Click **DEMO 1** → Describe land-cover (single image)
- Click **DEMO 2** → What changed? (flood_T1/T2)
- Click **DEMO 3** → Optical+SAR (pair1)
- Click **DEMO 4 ★** → Hero: What changed + can SAR corroborate? (T1/T2 + SAR) — **primary**
- Click **DEMO 5** → Invalid (mismatched sizes) → expect **refusal**

Or via API:

```bash
curl -F "query=What changed between these two dates?" -F "images=@data/demo/temporal_pairs/flood_T1.png" -F "images=@data/demo/temporal_pairs/flood_T2.png" http://localhost:8000/api/analyze | jq
```

## 8. Reports

- After any analysis, click **⬇ JSON REPORT** (saves from `/api/reports/<id>`).
- PDF: `python scripts/generate_pdf.py <report_id>` or `python scripts/generate_pdf.py` for latest.

## 9. Train (optional, cloud GPU)

```bash
bash scripts/cloud_setup.sh
bash scripts/cloud_verify.sh
python training/train_lora.py --dry_run --subset 200
python training/train_lora.py --dataset synthetic --subset 1000 --epochs 1 --use_4bit
```

## 10. Troubleshooting

- `Form data requires python-multipart`: `pip install python-multipart`
- `Directory 'assets' does not exist`: fixed in v0.3.0 — ensure you pulled latest.
- Port 8000 in use: `uvicorn ... --port 8001` and open `http://localhost:8001`
- Large images (>4096): auto-resized; very small (<64) may warn.
- Backend offline in UI: check terminal for errors, ensure `pip install` completed.

