"""
Regression test for DEMO 4 ★ static asset serving.
Ensures /data/demo/... returns image/png with PNG signature, not text/html (SPA fallback).
This guards against the bug where frontend fetched /data/demo/temporal_pairs/flood_T1.png
and received index.html (31 KB text/html) causing "cannot identify image file".
"""
import io
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

# Demo 4 ★ uses these 3 files — must be real PNGs, not HTML
DEMO4_FILES = [
    "data/demo/temporal_pairs/flood_T1.png",
    "data/demo/temporal_pairs/flood_T2.png",
    "data/demo/optical_sar_pairs/flood_sar.png",
]

# Also test other demo files
ALL_DEMO_FILES = [
    "data/demo/single_image/scene_1.png",
    "data/demo/temporal_pairs/flood_T1.png",
    "data/demo/temporal_pairs/flood_T2.png",
    "data/demo/optical_sar_pairs/pair1_optical.png",
    "data/demo/optical_sar_pairs/pair1_sar.png",
    "data/demo/optical_sar_pairs/flood_sar.png",
]

def test_demo_files_exist_on_disk():
    """Verify actual PNG files exist and are valid."""
    for rel in ALL_DEMO_FILES:
        p = Path(rel)
        assert p.exists(), f"Demo file missing on disk: {rel}"
        # Verify PNG signature
        with open(p, "rb") as f:
            sig = f.read(8)
            assert sig == b"\x89PNG\r\n\x1a\n", f"{rel} invalid PNG signature: {sig.hex()}"
        # Verify PIL can open
        img = Image.open(p)
        img.load()
        assert img.size[0] > 0 and img.size[1] > 0, f"{rel} has invalid dimensions"
        assert img.mode in ("RGB", "RGBA", "L"), f"{rel} unexpected mode {img.mode}"

def test_demo_static_returns_png_not_html():
    """Verify FastAPI serves /data/demo/... as image/png, not text/html SPA fallback."""
    for rel in DEMO4_FILES:
        url = f"/{rel}"  # e.g., /data/demo/temporal_pairs/flood_T1.png
        resp = client.get(url)
        assert resp.status_code == 200, f"{url} status {resp.status_code} != 200"
        ctype = resp.headers.get("content-type", "")
        assert "image/png" in ctype, f"{url} content-type {ctype} != image/png"
        # PNG signature
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n", f"{url} content is not PNG (first 8 bytes {resp.content[:8].hex()})"
        # Must not be HTML
        assert b"<html" not in resp.content[:500].lower(), f"{url} returned HTML instead of PNG"
        # Verify PIL can decode the HTTP response bytes
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        assert img.size == (512, 512), f"{url} decoded size {img.size} != (512,512)"
        assert img.mode == "RGB", f"{url} decoded mode {img.mode} != RGB"

def test_demo_static_all_files():
    """Verify all demo files via HTTP."""
    for rel in ALL_DEMO_FILES:
        url = f"/{rel}"
        resp = client.get(url)
        assert resp.status_code == 200, f"{url} failed"
        assert "image/png" in resp.headers.get("content-type", "")
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

def test_demo_static_404_not_html():
    """Non-existent demo file should be 404, not 200 HTML."""
    resp = client.get("/data/demo/temporal_pairs/nonexistent.png")
    assert resp.status_code == 404, f"Expected 404 for nonexistent demo file, got {resp.status_code}"

def test_spa_fallback_still_works():
    """SPA fallback should still return index.html for unknown routes."""
    resp = client.get("/some/random/spa/route")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert b"<html" in resp.text.lower().encode()[:500]

def test_api_still_works():
    """API should not be affected by static mount."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["service"] == "SatQuery AI"
