import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
import numpy as np
from backend.app.services.validation import validate_image_file, validate_temporal_pair

def make_png(w=256,h=256, color=(120,140,90)):
    img = Image.new("RGB", (w,h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_valid_single():
    data = make_png()
    r = validate_image_file(data, "test.png")
    assert r["valid"] == True
    assert r["meta"]["width"]==256

def test_invalid_format():
    r = validate_image_file(b"not image", "test.txt")
    assert r["valid"]==False

def test_temporal_valid():
    m1 = {"width":512,"height":512,"modality":"optical"}
    m2 = {"width":512,"height":512,"modality":"optical"}
    ok, reason, w = validate_temporal_pair(m1,m2)
    assert ok==True

def test_temporal_mismatch():
    m1 = {"width":512,"height":512,"modality":"optical"}
    m2 = {"width":128,"height":128,"modality":"optical"}
    ok, reason, w = validate_temporal_pair(m1,m2)
    assert ok==False

def test_corrupt_image():
    r = validate_image_file(b"\x89PNG not real", "corrupt.png")
    assert r["valid"]==False
