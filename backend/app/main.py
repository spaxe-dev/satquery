from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path

from backend.app.api.routes import router

app = FastAPI(
    title="SatQuery AI",
    description="Agentic Vision-Language Assistant for Multimodal Remote Sensing",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Expose demo assets as static files (must be before SPA catch-all)
# Frontend DEMO 1-5 fetches /data/demo/... and expects image/png, not text/html
DATA_DEMO_DIR = Path(__file__).resolve().parents[2] / "data" / "demo"
if DATA_DEMO_DIR.exists():
    app.mount("/data/demo", StaticFiles(directory=str(DATA_DEMO_DIR)), name="demo-data")

# Serve frontend static if built
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            return {"detail": "Not found"}
        # Don't intercept demo-data or assets — they are already mounted
        if full_path.startswith("data/demo/") or full_path.startswith("assets/"):
            return {"detail": "Not found"}
        idx = FRONTEND_DIST / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return {"detail": "Frontend not built"}

@app.get("/")
def root():
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    return {
        "service": "SatQuery AI",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/api/health",
        "analyze": "POST /api/analyze (multipart: query + images)",
    }
