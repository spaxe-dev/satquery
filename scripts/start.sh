#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "SatQuery AI — Starting backend..."
pip install -q fastapi uvicorn python-multipart pillow numpy opencv-python scikit-learn scikit-image rasterio shapely reportlab pyyaml || true
if [ ! -f "data/demo/single_image/scene_1.png" ]; then
  echo "Generating demo data..."
  python scripts/prepare_demo_data.py
fi
echo "Starting FastAPI on http://localhost:8000 ..."
echo "Frontend: http://localhost:8000  and  http://localhost:8000/docs"
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
