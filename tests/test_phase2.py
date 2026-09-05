"""Phase 2 regression: registry, raster limits, geo, coords, models, temporal/SAR."""
import os
import numpy as np

def test_registration_status():
    from packages.geospatial.registration import registration_status, box_to_preview
    m1 = {"width": 512, "height": 512, "crs": "EPSG:32643", "bounds": [0, 0, 1, 1], "transform_list": [1, 0, 0, 0, -1, 0], "has_georef": True}
    assert registration_status(m1, dict(m1)) == "registered"
    assert registration_status({"width": 128, "height": 128}, {"width": 512, "height": 512}) == "unregistered"
    b = box_to_preview({"x": 0, "y": 0, "w": 256, "h": 256}, 512, 512, 512, 512)
    assert abs(b["w"] - 256) < 1e-6  # aspect-preserved, no stretch

def test_raster_limits():
    from backend.app.services.validation import validate_image_file
    from PIL import Image
    import io
    img = Image.new("RGB", (8, 8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = validate_image_file(buf.getvalue(), "tiny.png")
    assert r["valid"] is False and "too small" in r["reason"]

def test_geo_preserved():
    from backend.app.services.validation import validate_image_file
    with open("data/demo/single_image/scene_1.png", "rb") as f:
        r = validate_image_file(f.read(), "scene_1.png")
    assert "bounds" in r["meta"] and "gsd_m" in r["meta"] and "nodata" in r["meta"]

def test_blip_unavailable_honest():
    from packages.models.blip_adapter import BlipAdapter
    a = BlipAdapter()
    # Without env gate must be unavailable, never fake
    assert a.is_available() is False
    g = a.ground(np.zeros((64, 64, 3), np.uint8), "water")
    assert g["boxes"] == [] and "not supported" in g["answer"].lower()

def test_fusion_needs_registration_note():
    from packages.geospatial.registration import registration_status
    assert registration_status({"width": 512, "height": 512, "modality": "optical"}, {"width": 512, "height": 512, "modality": "sar"}) == "unregistered"
