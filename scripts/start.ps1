Write-Host "SatQuery AI - Starting backend..." -ForegroundColor Cyan
Set-Location $PSScriptRoot/..
# Ensure deps
pip install -q fastapi uvicorn python-multipart pillow numpy opencv-python scikit-learn scikit-image rasterio shapely reportlab pyyaml 2>$null
# Ensure demo data exists
if (-Not (Test-Path "data/demo/single_image/scene_1.png")) {
  Write-Host "Generating demo data..." -ForegroundColor Yellow
  python scripts/prepare_demo_data.py
}
Write-Host "Starting FastAPI on http://localhost:8000 ..." -ForegroundColor Green
Write-Host "Frontend: http://localhost:8000  (served)  and  http://localhost:8000/docs (API)" -ForegroundColor Green
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
